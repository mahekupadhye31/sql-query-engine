"""Top-level, single-table database handle."""
from __future__ import annotations

import csv
from pathlib import Path

from .ast import SelectStmt
from .parser import parse
from .planner import ExecutionStats, execute_select
from .table import Table
from .types import Column, ColType, Row, Schema


class Engine:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.table: Table | None = None

    def create_table(self, name: str, columns: list[tuple[str, str]], index_column: str | None = None) -> Table:
        cols = tuple(Column(n, ColType.parse(t)) for n, t in columns)
        schema = Schema(table_name=name, columns=cols, index_column=index_column)
        self.table = Table.create(self.path, schema)
        return self.table

    def open(self) -> Table:
        self.table = Table.open(self.path)
        return self.table

    def insert(self, row: Row):
        return self.table.insert(row)

    def load_csv(self, csv_path: str | Path, has_header: bool = True) -> int:
        count = 0
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            if has_header:
                next(reader)
            for parts in reader:
                row = []
                for col, raw in zip(self.table.schema.columns, parts):
                    row.append(int(raw) if col.type is ColType.INTEGER else raw)
                self.table.insert(tuple(row))
                count += 1
        return count

    def execute(self, sql: str) -> tuple[list[Row], ExecutionStats]:
        stmt: SelectStmt = parse(sql)
        if stmt.table != self.table.schema.table_name:
            raise ValueError(f"unknown table {stmt.table!r}")
        return execute_select(self.table, stmt)

    def close(self) -> None:
        self.table.close()
