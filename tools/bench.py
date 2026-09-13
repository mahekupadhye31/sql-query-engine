#!/usr/bin/env python3
"""Performance benchmarks: page reads, latency, crossover selectivity, and
insert throughput.

Every table built here indexes the `value` column (controlled cardinality)
so an equality predicate on it can be served by either a scan or a seek --
that's what lets us force each method and compare them head to head at a
chosen selectivity, independent of what the planner's static rule would
have picked automatically.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlengine import Engine
from sqlengine.parser import parse
from sqlengine.planner import execute_select
from sqlengine.storage.page import HeapPage
from sqlengine.storage.pager import NULL_PAGE
from tools.gen_data import generate, write_csv

DEFAULT_SIZES = [1_000, 10_000, 100_000]


def percentile(data: list[float], p: float) -> float:
    data = sorted(data)
    if len(data) == 1:
        return data[0]
    k = (len(data) - 1) * p
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def build_table(tmp_dir: Path, n_rows: int, cardinality: int, order: str, seed: int,
                 index_column: str | None = "value") -> Engine:
    csv_path = tmp_dir / f"bench_{n_rows}_{cardinality}_{order}_{seed}.csv"
    db_path = tmp_dir / f"bench_{n_rows}_{cardinality}_{order}_{seed}.db"
    db_path.unlink(missing_ok=True)
    write_csv(csv_path, generate(n_rows, cardinality, order, seed))
    engine = Engine(db_path)
    engine.create_table(
        "t", [("id", "INTEGER"), ("value", "INTEGER"), ("label", "STRING")],
        index_column=index_column,
    )
    engine.load_csv(csv_path)
    return engine


def timed_query(engine: Engine, sql: str, method: str, repeats: int) -> dict:
    stmt = parse(sql)
    times = []
    reads = None
    rows_returned = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        rows, stats = execute_select(engine.table, stmt, force_method=method)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        reads = stats.page_reads
        rows_returned = stats.rows_returned
    return {
        "method": method,
        "p50_ms": round(percentile(times, 0.5), 5),
        "p95_ms": round(percentile(times, 0.95), 5),
        "page_reads": reads,
        "rows_returned": rows_returned,
    }


# -- 1 & 2: page reads + latency, scan vs seek, across table sizes ---------

def bench_scan_vs_seek(sizes: list[int], cardinality: int, tmp_dir: Path, seed: int, repeats: int) -> list[dict]:
    results = []
    for n in sizes:
        engine = build_table(tmp_dir, n, cardinality, "random", seed)
        match_value = 0  # value column is uniform over range(cardinality); 0 always exists for n >= 1
        sql = f"SELECT * FROM t WHERE value = {match_value}"
        scan = timed_query(engine, sql, "scan", repeats)
        seek = timed_query(engine, sql, "seek", repeats)
        results.append({
            "n_rows": n,
            "cardinality": cardinality,
            "selectivity": 1.0 / cardinality,
            "scan": scan,
            "seek": seek,
        })
        engine.close()
    return results


# -- 3: crossover selectivity sweep at a fixed table size -------------------

def _find_crossover(points: list[dict], faster_key: str) -> dict | None:
    """First selectivity, scanning low to high, where seek stops winning."""
    for i in range(1, len(points)):
        if points[i - 1][faster_key] and not points[i][faster_key]:
            return {
                "between_selectivity": [points[i - 1]["selectivity"], points[i]["selectivity"]],
                "between_cardinality": [points[i - 1]["cardinality"], points[i]["cardinality"]],
            }
    return None


def bench_crossover(n_rows: int, cardinalities: list[int], tmp_dir: Path, seed: int, repeats: int) -> dict:
    points = []
    for card in cardinalities:
        engine = build_table(tmp_dir, n_rows, card, "random", seed)
        sql = "SELECT * FROM t WHERE value = 0"
        scan = timed_query(engine, sql, "scan", repeats)
        seek = timed_query(engine, sql, "seek", repeats)
        selectivity = scan["rows_returned"] / n_rows if n_rows else 0.0
        points.append({
            "cardinality": card,
            "selectivity": selectivity,
            "scan_p50_ms": scan["p50_ms"],
            "seek_p50_ms": seek["p50_ms"],
            "scan_page_reads": scan["page_reads"],
            "seek_page_reads": seek["page_reads"],
            "seek_faster_latency": seek["p50_ms"] < scan["p50_ms"],
            "seek_faster_page_reads": seek["page_reads"] < scan["page_reads"],
        })
        engine.close()

    points.sort(key=lambda p: p["selectivity"])
    return {
        "n_rows": n_rows,
        "points": points,
        # Two different notions of "crossover", and they land in very
        # different places: latency also carries Python-level per-row
        # decode cost (paid once per SCANNED row, so it dominates at low
        # table sizes and pushes the observed crossover high); page_reads
        # is the pure I/O signal a real cost-based optimizer uses, and
        # crosses over much earlier -- roughly once seek's O(matches)
        # random heap fetches exceed the scan's fixed O(heap_pages) cost.
        "crossover_latency": _find_crossover(points, "seek_faster_latency"),
        "crossover_page_reads": _find_crossover(points, "seek_faster_page_reads"),
    }


# -- 4: insert throughput, with and without an index ------------------------

def bench_insert_throughput(n_rows: int, cardinality: int, tmp_dir: Path, seed: int) -> dict:
    out = {}
    for label, index_column in [("with_index", "value"), ("without_index", None)]:
        csv_path = tmp_dir / f"insert_{label}.csv"
        db_path = tmp_dir / f"insert_{label}.db"
        db_path.unlink(missing_ok=True)
        write_csv(csv_path, generate(n_rows, cardinality, "random", seed))
        engine = Engine(db_path)
        engine.create_table(
            "t", [("id", "INTEGER"), ("value", "INTEGER"), ("label", "STRING")],
            index_column=index_column,
        )
        t0 = time.perf_counter()
        engine.load_csv(csv_path)
        elapsed = time.perf_counter() - t0
        out[label] = {
            "n_rows": n_rows,
            "elapsed_s": round(elapsed, 4),
            "rows_per_sec": round(n_rows / elapsed, 1),
        }
        engine.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--cardinality", type=int, default=200,
                     help="cardinality used for the scan-vs-seek-by-size sweep")
    ap.add_argument("--crossover-rows", type=int, default=50_000)
    ap.add_argument("--insert-rows", type=int, default=100_000)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmp-dir", default=".sqlengine_bench")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--skip", nargs="*", default=[], choices=["scan-vs-seek", "crossover", "insert"])
    args = ap.parse_args()

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(exist_ok=True)
    report = {}

    if "scan-vs-seek" not in args.skip:
        print(f"== page reads & latency: scan vs seek across sizes {args.sizes} (cardinality={args.cardinality}) ==")
        r = bench_scan_vs_seek(args.sizes, args.cardinality, tmp_dir, args.seed, args.repeats)
        report["scan_vs_seek"] = r
        for row in r:
            print(f"  n={row['n_rows']:>9,}  scan: {row['scan']['page_reads']:>6} reads, "
                  f"p50={row['scan']['p50_ms']:.4f}ms  |  seek: {row['seek']['page_reads']:>3} reads, "
                  f"p50={row['seek']['p50_ms']:.4f}ms")

    if "crossover" not in args.skip:
        n = args.crossover_rows
        cardinalities = sorted({max(1, n // f) for f in
                                 [1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 50, 75, 100,
                                  150, 200, 300, 500, 750, 1000, 2000, n]}, reverse=True)
        print(f"\n== crossover selectivity sweep: n_rows={n}, cardinalities={cardinalities} ==")
        r = bench_crossover(n, cardinalities, tmp_dir, args.seed, args.repeats)
        report["crossover"] = r
        for p in r["points"]:
            marker = "SEEK" if p["seek_faster_latency"] else "SCAN"
            print(f"  selectivity={p['selectivity']:>8.4%}  scan_p50={p['scan_p50_ms']:.4f}ms "
                  f"({p['scan_page_reads']:>5} reads)  seek_p50={p['seek_p50_ms']:.4f}ms "
                  f"({p['seek_page_reads']:>5} reads)  -> faster: {marker}")
        for name, key in [("latency", "crossover_latency"), ("page-reads", "crossover_page_reads")]:
            c = r[key]
            if c:
                lo, hi = c["between_selectivity"]
                print(f"  {name.upper()} CROSSOVER between selectivity {lo:.4%} and {hi:.4%} "
                      f"(seek wins below, scan wins above)")
            else:
                print(f"  no {name} crossover observed in this range")

    if "insert" not in args.skip:
        n = args.insert_rows
        print(f"\n== insert throughput: {n:,} rows, with vs without index ==")
        r = bench_insert_throughput(n, args.cardinality, tmp_dir, args.seed)
        report["insert_throughput"] = r
        for label, stats in r.items():
            print(f"  {label:>14}: {stats['rows_per_sec']:>10,.1f} rows/sec ({stats['elapsed_s']}s total)")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
