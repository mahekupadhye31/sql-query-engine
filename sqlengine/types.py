"""Column types, schema, and row (de)serialization.

Rows are fixed-order tuples of columns. Supported types are INTEGER (8-byte
signed, struct format 'q') and STRING (uint16 length prefix + utf-8 bytes).
There is no NULL support and no variable schema evolution -- this engine
backs exactly one table per file, defined once at creation time.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ColType(Enum):
    INTEGER = "INTEGER"
    STRING = "STRING"

    @staticmethod
    def parse(name: str) -> "ColType":
        try:
            return ColType[name.upper()]
        except KeyError as e:
            raise ValueError(f"unknown column type: {name!r}") from e


@dataclass(frozen=True)
class Column:
    name: str
    type: ColType


@dataclass(frozen=True)
class Schema:
    table_name: str
    columns: tuple[Column, ...]
    index_column: str | None = None

    def col_index(self, name: str) -> int:
        for i, c in enumerate(self.columns):
            if c.name == name:
                return i
        raise KeyError(f"no such column: {name!r}")

    def col_type(self, name: str) -> ColType:
        return self.columns[self.col_index(name)].type

    def to_json(self) -> dict:
        return {
            "table_name": self.table_name,
            "columns": [{"name": c.name, "type": c.type.value} for c in self.columns],
            "index_column": self.index_column,
        }

    @staticmethod
    def from_json(d: dict) -> "Schema":
        cols = tuple(Column(c["name"], ColType.parse(c["type"])) for c in d["columns"])
        return Schema(table_name=d["table_name"], columns=cols, index_column=d.get("index_column"))


# RowId identifies a row's physical location: the heap page holding it and
# its slot index within that page's slot directory.
@dataclass(frozen=True)
class RowId:
    page_id: int
    slot: int

    def pack(self) -> bytes:
        return struct.pack("<IH", self.page_id, self.slot)

    @staticmethod
    def unpack(buf: bytes, offset: int = 0) -> "RowId":
        page_id, slot = struct.unpack_from("<IH", buf, offset)
        return RowId(page_id, slot)

    SIZE = 6


Row = tuple[Any, ...]


def encode_row(schema: Schema, row: Row) -> bytes:
    if len(row) != len(schema.columns):
        raise ValueError(f"expected {len(schema.columns)} values, got {len(row)}")
    parts: list[bytes] = []
    for col, val in zip(schema.columns, row):
        if col.type is ColType.INTEGER:
            parts.append(struct.pack("<q", int(val)))
        else:
            b = str(val).encode("utf-8")
            if len(b) > 0xFFFF:
                raise ValueError("string value too long")
            parts.append(struct.pack("<H", len(b)) + b)
    return b"".join(parts)


def decode_row(schema: Schema, buf: bytes) -> Row:
    offset = 0
    values: list[Any] = []
    for col in schema.columns:
        if col.type is ColType.INTEGER:
            (v,) = struct.unpack_from("<q", buf, offset)
            offset += 8
            values.append(v)
        else:
            (length,) = struct.unpack_from("<H", buf, offset)
            offset += 2
            v = buf[offset:offset + length].decode("utf-8")
            offset += length
            values.append(v)
    return tuple(values)


def encode_key(col_type: ColType, val: Any) -> Any:
    """Normalize a WHERE-clause literal to the comparable in-memory form
    used by the B+tree (plain Python int/str -- comparisons happen on
    decoded values, not raw bytes, see btree.py)."""
    if col_type is ColType.INTEGER:
        return int(val)
    return str(val)
