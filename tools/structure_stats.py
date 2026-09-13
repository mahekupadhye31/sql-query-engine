#!/usr/bin/env python3
"""Structural metrics for the B+tree: depth vs row count, node fanout, page
splits per 10K inserts, and index size as a fraction of table size.

Two views:
  1. A growth curve -- one table, built incrementally, sampled at
     checkpoints -- showing how depth/fanout/splits/size evolve with N.
  2. A fanout comparison between an INTEGER key (fixed 8-byte packed width)
     and a STRING key (variable width, generally wider), at a fixed row
     count, to make the "fanout depends on key size" claim concrete.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlengine import Engine
from sqlengine.btree import BTree
from tools.gen_data import generate, write_csv

DEFAULT_CHECKPOINTS = [1_000, 5_000, 10_000, 50_000, 100_000]


def _summarize(xs: list[int]) -> dict | None:
    if not xs:
        return None
    return {"count": len(xs), "mean": round(statistics.mean(xs), 2), "min": min(xs), "max": max(xs)}


def node_fanout_stats(btree: BTree) -> dict:
    leaf_fanouts, internal_fanouts = [], []
    for _, is_leaf, n in btree.walk_nodes():
        (leaf_fanouts if is_leaf else internal_fanouts).append(n)
    return {"leaf": _summarize(leaf_fanouts), "internal": _summarize(internal_fanouts)}


def index_size_fraction(engine: Engine) -> dict:
    btree = engine.table.btree
    index_pages = sum(1 for _ in btree.walk_nodes())
    heap_pages = engine.table.n_heap_pages
    total = index_pages + heap_pages
    return {
        "index_pages": index_pages,
        "heap_pages": heap_pages,
        "fraction": index_pages / total if total else 0.0,
    }


def growth_curve(checkpoints: list[int], cardinality: int, order: str, tmp_dir: Path,
                  seed: int, index_column: str = "value") -> list[dict]:
    total = checkpoints[-1]
    db_path = tmp_dir / f"structure_growth_{index_column}.db"
    db_path.unlink(missing_ok=True)
    engine = Engine(db_path)
    engine.create_table(
        "t", [("id", "INTEGER"), ("value", "INTEGER"), ("label", "STRING")], index_column=index_column
    )
    btree = engine.table.btree

    results = []
    checkpoint_set = set(checkpoints)
    last_checkpoint = 0
    last_split_count = 0
    for i, row in enumerate(generate(total, cardinality, order, seed), start=1):
        engine.insert(row)
        if i in checkpoint_set:
            splits_since = btree.split_count - last_split_count
            rows_since = i - last_checkpoint
            fanout = node_fanout_stats(btree)
            size = index_size_fraction(engine)
            results.append({
                "n_rows": i,
                "depth": btree.depth(),
                "leaf_fanout_mean": fanout["leaf"]["mean"] if fanout["leaf"] else None,
                "internal_fanout_mean": fanout["internal"]["mean"] if fanout["internal"] else None,
                "splits_since_last_checkpoint": splits_since,
                "splits_per_10k_inserts": round(splits_since / rows_since * 10_000, 2) if rows_since else 0.0,
                "index_pages": size["index_pages"],
                "heap_pages": size["heap_pages"],
                "index_size_fraction": round(size["fraction"], 4),
            })
            last_checkpoint = i
            last_split_count = btree.split_count
    engine.close()
    return results


def fanout_by_key_type(n_rows: int, cardinality: int, tmp_dir: Path, seed: int) -> dict:
    out = {}
    for label, index_column in [("INTEGER (id, unique)", "id"), ("STRING (label)", "label")]:
        db_path = tmp_dir / f"fanout_{index_column}.db"
        csv_path = tmp_dir / f"fanout_{index_column}.csv"
        db_path.unlink(missing_ok=True)
        write_csv(csv_path, generate(n_rows, cardinality, "random", seed))
        engine = Engine(db_path)
        engine.create_table(
            "t", [("id", "INTEGER"), ("value", "INTEGER"), ("label", "STRING")], index_column=index_column
        )
        engine.load_csv(csv_path)
        fanout = node_fanout_stats(engine.table.btree)
        out[label] = {"depth": engine.table.btree.depth(), **fanout}
        engine.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--cardinality", type=int, default=1000)
    ap.add_argument("--order", choices=["sequential", "random", "reverse"], default="random")
    ap.add_argument("--fanout-rows", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmp-dir", default=".sqlengine_structure")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(exist_ok=True)

    print(f"== growth curve: checkpoints={args.checkpoints}, cardinality={args.cardinality}, order={args.order} ==")
    curve = growth_curve(args.checkpoints, args.cardinality, args.order, tmp_dir, args.seed)
    for pt in curve:
        print(f"  n={pt['n_rows']:>9,}  depth={pt['depth']}  "
              f"leaf_fanout={pt['leaf_fanout_mean']:>6}  internal_fanout={pt['internal_fanout_mean']}  "
              f"splits/10K={pt['splits_per_10k_inserts']:>7}  "
              f"index/table pages={pt['index_pages']}/{pt['heap_pages']} "
              f"({pt['index_size_fraction']:.2%})")

    print(f"\n== fanout by key type: {args.fanout_rows:,} rows ==")
    fanout = fanout_by_key_type(args.fanout_rows, args.cardinality, tmp_dir, args.seed)
    for label, stats in fanout.items():
        leaf, internal = stats["leaf"], stats["internal"]
        print(f"  {label}: depth={stats['depth']}  "
              f"leaf fanout mean={leaf['mean'] if leaf else 'n/a'} (min={leaf['min'] if leaf else '-'}, "
              f"max={leaf['max'] if leaf else '-'})  "
              f"internal fanout mean={internal['mean'] if internal else 'n/a'}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"growth_curve": curve, "fanout_by_key_type": fanout}, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
