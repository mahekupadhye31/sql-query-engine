"""A tiny recursive-descent parser for the supported SQL subset:

    SELECT (* | col (, col)*) FROM table [WHERE col OP literal]

OP is one of = != < <= > >=. Literals are integers or single-quoted
strings. This is intentionally not a general SQL parser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .ast import SelectStmt, WhereClause

KEYWORDS = {"SELECT", "FROM", "WHERE"}

_TOKEN_SPEC = [
    ("STRING", r"'(?:[^'\\]|\\.)*'"),
    ("NUMBER", r"-?\d+"),
    ("LE", r"<="),
    ("GE", r">="),
    ("NE", r"!="),
    ("EQ", r"="),
    ("LT", r"<"),
    ("GT", r">"),
    ("COMMA", r","),
    ("STAR", r"\*"),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("WS", r"\s+"),
]
_MASTER_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC))

_OP_MAP = {"EQ": "=", "NE": "!=", "LT": "<", "LE": "<=", "GT": ">", "GE": ">="}


@dataclass(frozen=True)
class Token:
    kind: str
    value: str


def tokenize(sql: str) -> list[Token]:
    tokens = []
    pos = 0
    while pos < len(sql):
        m = _MASTER_RE.match(sql, pos)
        if not m:
            raise SyntaxError(f"unexpected character {sql[pos]!r} at position {pos}")
        kind = m.lastgroup
        text = m.group()
        pos = m.end()
        if kind == "WS":
            continue
        if kind == "IDENT" and text.upper() in KEYWORDS:
            kind = text.upper()
        tokens.append(Token(kind, text))
    tokens.append(Token("EOF", ""))
    return tokens


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def _expect(self, kind: str) -> Token:
        t = self._advance()
        if t.kind != kind:
            raise SyntaxError(f"expected {kind}, got {t.kind} ({t.value!r})")
        return t

    def parse_select(self) -> SelectStmt:
        self._expect("SELECT")
        columns = self._parse_column_list()
        self._expect("FROM")
        table = self._expect("IDENT").value
        where = None
        if self._peek().kind == "WHERE":
            self._advance()
            where = self._parse_where()
        self._expect("EOF")
        return SelectStmt(columns=columns, table=table, where=where)

    def _parse_column_list(self) -> list[str]:
        if self._peek().kind == "STAR":
            self._advance()
            return ["*"]
        cols = [self._expect("IDENT").value]
        while self._peek().kind == "COMMA":
            self._advance()
            cols.append(self._expect("IDENT").value)
        return cols

    def _parse_where(self) -> WhereClause:
        column = self._expect("IDENT").value
        op_tok = self._advance()
        if op_tok.kind not in _OP_MAP:
            raise SyntaxError(f"expected a comparison operator, got {op_tok.value!r}")
        val_tok = self._advance()
        if val_tok.kind == "NUMBER":
            value = int(val_tok.value)
        elif val_tok.kind == "STRING":
            value = val_tok.value[1:-1].replace("\\'", "'")
        else:
            raise SyntaxError(f"expected a literal value, got {val_tok.value!r}")
        return WhereClause(column=column, op=_OP_MAP[op_tok.kind], value=value)


def parse(sql: str) -> SelectStmt:
    return Parser(tokenize(sql)).parse_select()
