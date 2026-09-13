"""The planner/executor: given a parsed SELECT, pick a full table scan or
an index seek by actually costing both, then produce rows.

An index seek is only *possible* for an equality predicate on the indexed
column -- the B+tree has no way to serve anything else. Among possible
plans, the choice is cost-based:

    scan_cost = n_heap_pages                 (every heap page, once, sequentially)
    seek_cost = btree.depth() + len(matches) (tree descent, + one heap
                                               fetch per match, worst case)

The trick that makes this cheap to compute rather than merely theoretical:
`btree.search()` only ever touches B+tree pages, never the heap, so getting
the *exact* match count and choosing the cheaper plan costs nothing beyond
what a seek would have paid anyway. If seek turns out cheaper, its rowids
are already in hand and executing it is free; if scan turns out cheaper,
the tree pages read during costing are still counted in ExecutionStats
(they were real I/O, whichever plan ultimately ran).

This cost model is exactly what produces the "crossover selectivity" that
tools/bench.py measures empirically: below some match-count threshold,
depth + matches < n_heap_pages and the planner seeks; above it, scan wins.
`force_method` bypasses this decision (used by bench.py to also measure the
plan the model *didn't* pick, for comparison).
"""
from __future__ import annotations

from dataclasses import dataclass

from .ast import SelectStmt
from .table import Table
from .types import Row, RowId, encode_key


@dataclass
class ExecutionStats:
    method: str = ""          # "scan" or "seek"
    page_reads: int = 0
    rows_examined: int = 0
    rows_returned: int = 0
    estimated_scan_cost: int | None = None
    estimated_seek_cost: int | None = None


def _eval_predicate(schema, row: Row, where) -> bool:
    idx = schema.col_index(where.column)
    val = row[idx]
    target = encode_key(schema.col_type(where.column), where.value)
    op = where.op
    if op == "=":
        return val == target
    if op == "!=":
        return val != target
    if op == "<":
        return val < target
    if op == "<=":
        return val <= target
    if op == ">":
        return val > target
    if op == ">=":
        return val >= target
    raise ValueError(f"unsupported operator {op!r}")


def _seek_available(table: Table, stmt: SelectStmt) -> bool:
    w = stmt.where
    return w is not None and w.op == "=" and table.schema.index_column == w.column and table.btree is not None


def _plan(table: Table, stmt: SelectStmt) -> tuple[str, list[RowId] | None, ExecutionStats]:
    """Cost both available plans and pick the cheaper one. Returns
    (method, precomputed_seek_rowids_or_None, partially-filled stats)."""
    stats = ExecutionStats()
    if not _seek_available(table, stmt):
        stats.estimated_scan_cost = table.n_heap_pages
        return "scan", None, stats

    key = encode_key(table.schema.col_type(stmt.where.column), stmt.where.value)
    matches = table.btree.search(key)  # index-only: no heap I/O yet
    seek_cost = table.btree.depth() + len(matches)
    scan_cost = table.n_heap_pages
    stats.estimated_seek_cost = seek_cost
    stats.estimated_scan_cost = scan_cost

    if seek_cost < scan_cost:
        return "seek", matches, stats
    return "scan", None, stats


def execute_select(
    table: Table, stmt: SelectStmt, force_method: str | None = None
) -> tuple[list[Row], ExecutionStats]:
    start_reads = table.pager.read_count
    results: list[Row] = []

    if force_method is not None:
        if force_method == "seek" and not _seek_available(table, stmt):
            raise ValueError("cannot force a seek: no equality predicate / no index available")
        method, precomputed = force_method, None
        stats = ExecutionStats(method=method)
    else:
        method, precomputed, stats = _plan(table, stmt)
        stats.method = method

    if method == "seek":
        rowids = precomputed if precomputed is not None else table.btree.search(
            encode_key(table.schema.col_type(stmt.where.column), stmt.where.value)
        )
        for rid in rowids:
            results.append(table.get_row(rid))
        stats.rows_examined = len(rowids)
    else:
        for row in table.scan():
            stats.rows_examined += 1
            if stmt.where is None or _eval_predicate(table.schema, row, stmt.where):
                results.append(row)

    stats.page_reads = table.pager.read_count - start_reads
    stats.rows_returned = len(results)

    if stmt.columns != ["*"]:
        idxs = [table.schema.col_index(c) for c in stmt.columns]
        results = [tuple(r[i] for i in idxs) for r in results]

    return results, stats
