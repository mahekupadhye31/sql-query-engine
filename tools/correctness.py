#!/usr/bin/env python3
"""Correctness oracle: SQLite.

1. Load the same synthetic CSV into sqlengine and into an in-memory
   SQLite table with an identical schema.
2. Fire N randomized SELECT queries (which happen to be valid syntax in
   both engines) at both, and assert the (multiset) result sets match.
3. Check B+tree invariants after the load: keys sorted within and across
   leaves, all leaves at equal depth, internal separator/child counts
   consistent.
4. Crash recovery: fork a worker that inserts rows one at a time, SIGKILL
   it at a random point, reopen the file fresh, and verify every row the
   worker reported as committed (i.e. printed after its insert()/fsync
   returned) is still present with the correct value.
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import os
import random
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlengine import Engine
from tools.gen_data import generate, write_csv

COLUMNS = ["id", "value", "label"]
OPS = ["=", "!=", "<", "<=", ">", ">="]


def build_sqlite(csv_path) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER, value INTEGER, label TEXT)")
    with open(csv_path, newline="") as f:
        reader = csv_mod.reader(f)
        next(reader)
        conn.executemany("INSERT INTO t VALUES (?, ?, ?)", ((int(a), int(b), c) for a, b, c in reader))
    conn.commit()
    return conn


def build_sqlengine(csv_path, db_path, index_column="id") -> Engine:
    engine = Engine(db_path)
    engine.create_table(
        "t",
        [("id", "INTEGER"), ("value", "INTEGER"), ("label", "STRING")],
        index_column=index_column,
    )
    engine.load_csv(csv_path)
    return engine


def random_queries(n: int, n_rows: int, cardinality: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    queries = []
    for _ in range(n):
        col = rng.choice(COLUMNS)
        select_cols = rng.choice(["*", "id", "id, value", "label"])
        if col == "id":
            op = rng.choice(OPS)
            lit = str(rng.randrange(-5, n_rows + 5))
        elif col == "value":
            op = rng.choice(OPS)
            lit = str(rng.randrange(cardinality))
        else:
            op = rng.choice(["=", "!="])
            lit = f"'row_{rng.randrange(-5, n_rows + 5)}'"
        queries.append(f"SELECT {select_cols} FROM t WHERE {col} {op} {lit}")
    return queries


def normalize(rows) -> Counter:
    return Counter(tuple(r) for r in rows)


def run_oracle(n_rows=5000, cardinality=50, n_queries=300, seed=0, tmp_dir=".", index_column="id") -> dict:
    tmp_dir = Path(tmp_dir)
    csv_path = tmp_dir / "oracle.csv"
    db_path = tmp_dir / "oracle.db"
    db_path.unlink(missing_ok=True)
    write_csv(csv_path, generate(n_rows, cardinality, order="random", seed=seed))

    sqlite_conn = build_sqlite(csv_path)
    engine = build_sqlengine(csv_path, db_path, index_column=index_column)

    queries = random_queries(n_queries, n_rows, cardinality, seed=seed + 1)
    mismatches = []
    method_counts = {"scan": 0, "seek": 0}

    for sql in queries:
        expected = normalize(sqlite_conn.execute(sql).fetchall())
        actual_rows, stats = engine.execute(sql)
        method_counts[stats.method] += 1
        actual = normalize(actual_rows)
        if actual != expected:
            mismatches.append({"sql": sql, "expected": dict(expected), "actual": dict(actual)})

    invariant_errors = engine.table.btree.validate() if engine.table.btree else []
    cached_heap_pages = engine.table.n_heap_pages
    walked_heap_pages = engine.table.count_heap_pages_by_walking()
    if cached_heap_pages != walked_heap_pages:
        invariant_errors.append(
            f"n_heap_pages counter drifted: cached={cached_heap_pages} walked={walked_heap_pages}"
        )
    engine.close()
    sqlite_conn.close()

    return {
        "n_rows": n_rows,
        "n_queries": n_queries,
        "mismatches": mismatches,
        "agreement_rate": (n_queries - len(mismatches)) / n_queries,
        "method_counts": method_counts,
        "btree_invariant_errors": invariant_errors,
    }


def crash_recovery_test(tmp_dir=".", n_rows=3000, seed=0) -> dict:
    tmp_dir = Path(tmp_dir)
    db_path = tmp_dir / "crash.db"
    db_path.unlink(missing_ok=True)
    worker = str(Path(__file__).resolve().parent / "_crash_worker.py")

    env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))
    proc = subprocess.Popen(
        [sys.executable, worker, str(db_path), str(n_rows), "0"],
        stdout=subprocess.PIPE, text=True, bufsize=1, env=env,
    )
    rng = random.Random(seed)
    kill_after = rng.randint(max(1, n_rows // 10), max(2, n_rows // 2))
    committed = []
    for line in proc.stdout:
        parts = line.split()
        if len(parts) != 2:
            continue
        committed.append(int(parts[1]))
        if len(committed) >= kill_after:
            proc.kill()  # SIGKILL
            break
    proc.wait()

    engine = Engine(db_path)
    engine.open()
    missing = []
    for i in committed:
        rows, _ = engine.execute(f"SELECT * FROM t WHERE id = {i}")
        if not rows or rows[0][1] != f"val_{i}":
            missing.append(i)
    invariant_errors = engine.table.btree.validate() if engine.table.btree else []
    engine.close()

    return {
        "rows_attempted": n_rows,
        "kill_after": kill_after,
        "rows_committed_before_kill": len(committed),
        "rows_missing_after_recovery": missing,
        "btree_invariant_errors": invariant_errors,
        "passed": len(missing) == 0 and len(invariant_errors) == 0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--cardinality", type=int, default=50)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmp-dir", default=".sqlengine_tmp")
    ap.add_argument("--crash-rows", type=int, default=3000)
    ap.add_argument("--skip-crash", action="store_true")
    args = ap.parse_args()

    Path(args.tmp_dir).mkdir(exist_ok=True)

    print(f"== oracle test: {args.rows} rows, cardinality={args.cardinality}, {args.queries} queries ==")
    result = run_oracle(args.rows, args.cardinality, args.queries, args.seed, args.tmp_dir)
    print(f"agreement rate: {result['agreement_rate']:.4%} ({len(result['mismatches'])} mismatches)")
    print(f"method split: {result['method_counts']}")
    print(f"btree invariant errors: {result['btree_invariant_errors'] or 'none'}")
    for m in result["mismatches"][:5]:
        print("MISMATCH:", m["sql"])
        print("  expected:", m["expected"])
        print("  actual:  ", m["actual"])

    ok = not result["mismatches"] and not result["btree_invariant_errors"]

    if not args.skip_crash:
        print(f"\n== crash recovery test: kill -9 mid-insert over {args.crash_rows} rows ==")
        crash_result = crash_recovery_test(args.tmp_dir, args.crash_rows, args.seed)
        print(f"committed before kill: {crash_result['rows_committed_before_kill']} "
              f"(kill scheduled after {crash_result['kill_after']})")
        print(f"missing after recovery: {crash_result['rows_missing_after_recovery'] or 'none'}")
        print(f"btree invariant errors: {crash_result['btree_invariant_errors'] or 'none'}")
        print("PASSED" if crash_result["passed"] else "FAILED")
        ok = ok and crash_result["passed"]

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
