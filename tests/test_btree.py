import random

from sqlengine.btree import BTree
from sqlengine.storage.pager import Pager
from sqlengine.types import RowId


def make_tree(tmp_path):
    pager = Pager(tmp_path / "idx.db", create=True)
    tree = BTree.create_empty(pager)
    return pager, tree


def test_invariants_hold_after_every_single_insert(tmp_path):
    """Stronger than checking invariants once after a bulk load: re-run
    validate() (keys sorted, leaves at equal depth, separator/child counts
    consistent) after every individual insert, mixing int and duplicate
    keys and both insert orders. This is O(n^2) since each validate() walks
    the whole tree, so n is kept modest -- it's a correctness stress test,
    not something that runs in the hot path."""
    _, tree = make_tree(tmp_path)
    rng = random.Random(7)
    n = 1500
    keys = [rng.randrange(0, n // 3) for _ in range(n)]  # guarantees duplicates
    for i, k in enumerate(keys):
        tree.insert(k, RowId(page_id=i, slot=0))
        errors = tree.validate()
        assert errors == [], f"invariant violated after insert #{i} (key={k}): {errors}"


def test_insert_and_search_single(tmp_path):
    _, tree = make_tree(tmp_path)
    tree.insert(5, RowId(1, 0))
    assert tree.search(5) == [RowId(1, 0)]
    assert tree.search(6) == []


def test_duplicate_keys(tmp_path):
    _, tree = make_tree(tmp_path)
    tree.insert(5, RowId(1, 0))
    tree.insert(5, RowId(1, 1))
    tree.insert(5, RowId(2, 0))
    results = tree.search(5)
    assert sorted(results, key=lambda r: (r.page_id, r.slot)) == [
        RowId(1, 0), RowId(1, 1), RowId(2, 0)
    ]


def test_forces_splits_and_stays_correct(tmp_path):
    _, tree = make_tree(tmp_path)
    n = 5000
    keys = list(range(n))
    random.Random(42).shuffle(keys)
    for i, k in enumerate(keys):
        tree.insert(k, RowId(page_id=i, slot=0))

    assert tree.split_count > 0, "5000 int keys should force at least one split"

    for k in range(0, n, 137):
        results = tree.search(k)
        assert len(results) == 1
        assert results[0].page_id == keys.index(k)

    errors = tree.validate()
    assert errors == [], f"invariant violations: {errors}"


def test_depth_grows_with_size(tmp_path):
    _, tree = make_tree(tmp_path)
    assert tree.depth() == 1
    for i in range(20000):
        tree.insert(i, RowId(i, 0))
    assert tree.depth() > 1


def test_string_keys(tmp_path):
    _, tree = make_tree(tmp_path)
    words = ["banana", "apple", "cherry", "date", "elderberry"]
    for i, w in enumerate(words):
        tree.insert(w, RowId(i, 0))
    assert tree.search("apple") == [RowId(1, 0)]
    assert tree.search("missing") == []
    errors = tree.validate()
    assert errors == []


def test_duplicate_run_spans_multiple_leaves(tmp_path):
    _, tree = make_tree(tmp_path)
    # Every row shares the same key -- the duplicate run must span many
    # leaf pages, and search() must follow next_leaf to collect all of them.
    n = 5000
    for i in range(n):
        tree.insert(0, RowId(page_id=i, slot=0))
    results = tree.search(0)
    assert len(results) == n
    assert {r.page_id for r in results} == set(range(n))
    assert tree.search(1) == []
    errors = tree.validate()
    assert errors == []


def test_sequential_insert_order_also_valid(tmp_path):
    _, tree = make_tree(tmp_path)
    for i in range(10000):
        tree.insert(i, RowId(i, 0))
    errors = tree.validate()
    assert errors == []
    assert tree.search(9999) == [RowId(9999, 0)]
    assert tree.search(0) == [RowId(0, 0)]
