"""Interactive REPL: `python3 -m sqlengine [path]`

If PATH doesn't exist, creates it with a small demo table so there's
something to query in the first 30 seconds. Every query prints which plan
the cost-based planner picked and why (page reads, and the scan/seek cost
estimates it compared).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .engine import Engine
from .parser import parse
from .planner import execute_select

DEMO_TABLE = "t"
DEMO_COLUMNS = [("id", "INTEGER"), ("value", "INTEGER"), ("label", "STRING")]
DEMO_INDEX_COLUMN = "id"


def _demo_rows(n: int, cardinality: int, seed: int = 0):
    rng = random.Random(seed)
    for i in range(n):
        yield (i, rng.randrange(cardinality), f"row_{i}")


def _print_table(rows: list, columns: list[str]) -> None:
    if not rows:
        print("(0 rows)")
        return
    str_rows = [[str(v) for v in row] for row in rows]
    widths = [len(c) for c in columns]
    for row in str_rows:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))

    def fmt(vals):
        return "  ".join(v.ljust(w) for v, w in zip(vals, widths))

    print(fmt(columns))
    print(fmt(["-" * w for w in widths]))
    for row in str_rows:
        print(fmt(row))
    print(f"({len(rows)} row{'s' if len(rows) != 1 else ''})")


def _print_help() -> None:
    print("""
Enter SQL: SELECT col[, col...] FROM table [WHERE col OP literal]
  OP is one of = != < <= > >=, literals are integers or 'quoted strings'

Meta commands:
  .schema     show the table's columns and which one is indexed
  .stats      show heap page count, B+tree depth, and index page count
  .help       show this message
  .exit       quit (also: .quit, Ctrl-D)
""")


def _print_schema(table) -> None:
    idx = table.schema.index_column
    print(f"table {table.schema.table_name!r}:")
    for col in table.schema.columns:
        marker = "  [INDEXED]" if col.name == idx else ""
        print(f"  {col.name} {col.type.value}{marker}")


def _print_stats(table) -> None:
    print(f"heap pages:  {table.n_heap_pages}")
    if table.btree is not None:
        index_pages = sum(1 for _ in table.btree.walk_nodes())
        print(f"btree depth: {table.btree.depth()}")
        print(f"btree pages: {index_pages}")
    else:
        print("no index on this table")


def run_repl(engine: Engine) -> None:
    table = engine.table
    print(f"Connected to {engine.path!r}. Type .help for commands, .exit to quit.\n")
    _print_schema(table)
    while True:
        try:
            line = input("\nsqlengine> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue
        if not line:
            continue
        if line in (".exit", ".quit"):
            break
        if line == ".help":
            _print_help()
            continue
        if line == ".schema":
            _print_schema(table)
            continue
        if line == ".stats":
            _print_stats(table)
            continue

        try:
            stmt = parse(line)
        except SyntaxError as e:
            print(f"syntax error: {e}")
            continue
        if stmt.table != table.schema.table_name:
            print(f"error: unknown table {stmt.table!r}")
            continue
        try:
            rows, stats = execute_select(table, stmt)
        except (ValueError, KeyError) as e:
            print(f"error: {e}")
            continue

        col_names = [c.name for c in table.schema.columns] if stmt.columns == ["*"] else stmt.columns
        _print_table(rows, col_names)

        cost = ""
        if stats.estimated_scan_cost is not None:
            cost = f", cost model: scan={stats.estimated_scan_cost}"
            if stats.estimated_seek_cost is not None:
                cost += f" seek={stats.estimated_seek_cost}"
        plural = "s" if stats.page_reads != 1 else ""
        print(f"[{stats.method}] {stats.page_reads} page read{plural}{cost}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="python3 -m sqlengine",
        description="Interactive REPL for the sqlengine query engine.",
    )
    ap.add_argument("path", nargs="?", default="demo.db",
                     help="database file to open (default: demo.db, created with demo data if missing)")
    ap.add_argument("--demo-rows", type=int, default=2000, help="rows to generate for a new demo database")
    ap.add_argument("--cardinality", type=int, default=50,
                     help="distinct values in the demo `value` column (unindexed)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    path = Path(args.path)
    engine = Engine(path)
    if path.exists():
        engine.open()
        print(f"Opened existing database {path}")
    else:
        engine.create_table(DEMO_TABLE, DEMO_COLUMNS, index_column=DEMO_INDEX_COLUMN)
        for row in _demo_rows(args.demo_rows, args.cardinality, args.seed):
            engine.insert(row)
        print(f"Created new demo database {path}: {args.demo_rows} rows in table {DEMO_TABLE!r}, "
              f"indexed on {DEMO_INDEX_COLUMN!r}.")
        print("Try:")
        print("  SELECT * FROM t WHERE id = 42        (indexed column, equality -> seek)")
        print("  SELECT * FROM t WHERE value = 10     (unindexed column -> scan)")
        print("  .stats                                (B+tree depth, page counts)")

    try:
        run_repl(engine)
    finally:
        engine.close()
    print("bye")


if __name__ == "__main__":
    main()
