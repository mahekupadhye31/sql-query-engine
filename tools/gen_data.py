#!/usr/bin/env python3
"""Synthetic data generator for benchmarking and correctness testing.

Generates rows with three columns:

    id:    INTEGER, unique, 0..n-1                 (good candidate for a
                                                      point-lookup index)
    value: INTEGER, drawn uniformly from
           range(cardinality)                       (the "selectivity knob":
                                                      an equality match on a
                                                      specific value hits an
                                                      expected fraction of
                                                      1/cardinality of all
                                                      rows -- decoupled from
                                                      n_rows)
    label: STRING, "row_<id>"                        (exercises the STRING
                                                      column type)

`order` controls the ROW order rows are written/inserted in -- this is the
knob for "insert order" in the structure metrics (ascending ids produce
right-heavy B+tree splits typical of auto-increment keys; random insert
order produces a different, usually more balanced, split pattern).
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def generate(n_rows: int, cardinality: int, order: str = "sequential", seed: int = 0):
    rng = random.Random(seed)
    ids = list(range(n_rows))
    values = [rng.randrange(cardinality) for _ in range(n_rows)]

    if order == "sequential":
        row_ids = ids
    elif order == "random":
        row_ids = ids[:]
        rng.shuffle(row_ids)
    elif order == "reverse":
        row_ids = list(reversed(ids))
    else:
        raise ValueError(f"unknown order: {order!r}")

    for rid in row_ids:
        yield (rid, values[rid], f"row_{rid}")


def write_csv(path: str | Path, rows) -> int:
    count = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "value", "label"])
        for row in rows:
            w.writerow(row)
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=10_000)
    ap.add_argument("--cardinality", type=int, default=100,
                     help="distinct values in the `value` column; selectivity of an "
                          "equality match is ~1/cardinality")
    ap.add_argument("--order", choices=["sequential", "random", "reverse"], default="sequential")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="data.csv")
    args = ap.parse_args()

    rows = generate(args.rows, args.cardinality, args.order, args.seed)
    count = write_csv(args.out, rows)
    print(f"wrote {count} rows to {args.out} (cardinality={args.cardinality}, order={args.order})")


if __name__ == "__main__":
    main()
