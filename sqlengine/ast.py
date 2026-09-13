from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Comparison operators supported in a WHERE clause. Only "=" can be served
# by the B+tree index seek path; everything else forces a full scan.
COMPARISON_OPS = ("=", "!=", "<", "<=", ">", ">=")


@dataclass(frozen=True)
class WhereClause:
    column: str
    op: str
    value: Any


@dataclass(frozen=True)
class SelectStmt:
    columns: list[str]  # ["*"] means all columns
    table: str
    where: WhereClause | None
