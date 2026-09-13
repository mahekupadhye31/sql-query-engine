"""An on-disk B+tree secondary index.

Each node (leaf or internal) is serialized in full into a single 4KB page.
Leaves hold (key, RowId) pairs in sorted order and are threaded together
via a `next_leaf` pointer for ordered traversal.

Ordering with duplicate keys: a non-unique indexed column is legal SQL (many
rows can share one value), so ties are broken by RowId. Every entry's true
sort position is the composite (value, page_id, slot), never the bare value
alone. Internal separators are promoted as a full composite too -- the
promoted (value, rowid) pair is the exact first entry of the right child
after a split, so a search descends using the *low sentinel* probe
(value, -1, -1), which sorts before every real entry with that value and
therefore always lands at or before the true leftmost matching leaf. (Using
bisect on the bare value would land on whichever leaf a *new* insert of that
value would go to -- the rightmost one -- silently skipping every earlier
leaf holding the same value. This was a real bug caught by
tests/test_btree.py::test_duplicate_run_spans_multiple_leaves, where an
all-duplicate-key tree only returned one leaf's worth of matches.) Landing
early is harmless: search()'s forward scan keeps following next_leaf while
it has consumed a whole leaf without hitting a non-matching key.

Splitting is triggered lazily: an insert always writes the candidate node
first; only if the serialized node would overflow PAGE_SIZE do we split it
and propagate a separator upward, iteratively, allocating a new root if the
split reaches the top. This means fanout is not a fixed constant -- it's
however many entries fit in 4KB, which is exactly why fanout differs for
INTEGER vs STRING keys (see structure_stats.py).
"""
from __future__ import annotations

import bisect
import struct
from dataclasses import dataclass, field
from typing import Any

from .storage.pager import Pager, PAGE_SIZE, NULL_PAGE
from .storage.page import PAGE_TYPE_BTREE_LEAF, PAGE_TYPE_BTREE_INTERNAL
from .types import RowId

_LEAF_HEADER = struct.Struct("<BHI")      # page_type, num_keys, next_leaf
_INTERNAL_HEADER = struct.Struct("<BHI")  # page_type, num_keys, first_child


def _pack_key(key: Any) -> bytes:
    if isinstance(key, int):
        return b"\x00" + struct.pack("<q", key)
    b = str(key).encode("utf-8")
    if len(b) > 0xFFFF:
        raise ValueError("string key too long")
    return b"\x01" + struct.pack("<H", len(b)) + b


def _unpack_key(buf: bytes, offset: int) -> tuple[Any, int]:
    tag = buf[offset]
    offset += 1
    if tag == 0:
        (v,) = struct.unpack_from("<q", buf, offset)
        return v, offset + 8
    (length,) = struct.unpack_from("<H", buf, offset)
    offset += 2
    v = bytes(buf[offset:offset + length]).decode("utf-8")
    return v, offset + length


def _composite(value: Any, rowid: RowId) -> tuple:
    return (value, rowid.page_id, rowid.slot)


def _low_probe(value: Any) -> tuple:
    """Sorts strictly before any real entry with this value (real page ids
    are always >= 1, page 0 is reserved for the meta page)."""
    return (value, -1, -1)


def serialize_leaf(keys: list, rowids: list[RowId], next_leaf: int) -> bytes:
    parts = [_LEAF_HEADER.pack(PAGE_TYPE_BTREE_LEAF, len(keys), next_leaf)]
    for k, rid in zip(keys, rowids):
        parts.append(_pack_key(k))
        parts.append(rid.pack())
    body = b"".join(parts)
    if len(body) > PAGE_SIZE:
        raise OverflowError("leaf node exceeds page size")
    return body + bytes(PAGE_SIZE - len(body))


def deserialize_leaf(buf) -> tuple[list, list[RowId], int]:
    page_type, num_keys, next_leaf = _LEAF_HEADER.unpack_from(buf, 0)
    assert page_type == PAGE_TYPE_BTREE_LEAF
    offset = _LEAF_HEADER.size
    keys, rowids = [], []
    for _ in range(num_keys):
        k, offset = _unpack_key(buf, offset)
        rid = RowId.unpack(buf, offset)
        offset += RowId.SIZE
        keys.append(k)
        rowids.append(rid)
    return keys, rowids, next_leaf


def serialize_internal(keys: list, rowids: list[RowId], children: list[int]) -> bytes:
    """Separators are full (value, RowId) composites -- see module docstring
    for why the tiebreaker is required once duplicate keys are allowed."""
    assert len(children) == len(keys) + 1 == len(rowids) + 1
    parts = [_INTERNAL_HEADER.pack(PAGE_TYPE_BTREE_INTERNAL, len(keys), children[0])]
    for k, rid, child in zip(keys, rowids, children[1:]):
        parts.append(_pack_key(k))
        parts.append(rid.pack())
        parts.append(struct.pack("<I", child))
    body = b"".join(parts)
    if len(body) > PAGE_SIZE:
        raise OverflowError("internal node exceeds page size")
    return body + bytes(PAGE_SIZE - len(body))


def deserialize_internal(buf) -> tuple[list, list[RowId], list[int]]:
    page_type, num_keys, first_child = _INTERNAL_HEADER.unpack_from(buf, 0)
    assert page_type == PAGE_TYPE_BTREE_INTERNAL
    offset = _INTERNAL_HEADER.size
    keys, rowids = [], []
    children = [first_child]
    for _ in range(num_keys):
        k, offset = _unpack_key(buf, offset)
        rid = RowId.unpack(buf, offset)
        offset += RowId.SIZE
        (child,) = struct.unpack_from("<I", buf, offset)
        offset += 4
        keys.append(k)
        rowids.append(rid)
        children.append(child)
    return keys, rowids, children


def is_leaf_page(buf) -> bool:
    return buf[0] == PAGE_TYPE_BTREE_LEAF


@dataclass
class _Frame:
    page_id: int
    is_leaf: bool
    keys: list
    # leaf: rowids are per-row RowIds, next_leaf is the sibling pointer.
    # internal: rowids are separator tiebreakers, children are page ids,
    # child_idx is the index we descended into (where a split promotion
    # gets inserted).
    rowids: list = field(default_factory=list)
    next_leaf: int = NULL_PAGE
    children: list = field(default_factory=list)
    child_idx: int = -1


class BTree:
    def __init__(self, pager: Pager, root_page_id: int, depth: int | None = None):
        self.pager = pager
        self.root_page_id = root_page_id
        self.split_count = 0
        # Maintained incrementally (a root split is the only thing that can
        # change it, and insert() already knows when that happens) so the
        # planner's cost model can read it in O(1) instead of paying for a
        # fresh root-to-leaf descent on every single query just to find out
        # how deep the tree is.
        self._depth = depth if depth is not None else self._walk_depth()

    def _walk_depth(self) -> int:
        depth = 1
        page_id = self.root_page_id
        while True:
            buf = self.pager.read_page(page_id)
            if is_leaf_page(buf):
                return depth
            _, _, children = deserialize_internal(buf)
            page_id = children[0]
            depth += 1

    @staticmethod
    def create_empty(pager: Pager) -> "BTree":
        root_id = pager.allocate_page()
        pager.write_page(root_id, serialize_leaf([], [], NULL_PAGE))
        return BTree(pager, root_id, depth=1)

    # -- search --------------------------------------------------------

    def search(self, key: Any) -> list[RowId]:
        page_id = self.root_page_id
        probe = _low_probe(key)
        while True:
            buf = self.pager.read_page(page_id)
            if is_leaf_page(buf):
                break
            keys, sep_rowids, children = deserialize_internal(buf)
            composites = [_composite(k, r) for k, r in zip(keys, sep_rowids)]
            page_id = children[bisect.bisect_right(composites, probe)]

        keys, rowids, next_leaf = deserialize_leaf(buf)
        pos = bisect.bisect_left(keys, key)
        out = []
        while True:
            while pos < len(keys) and keys[pos] == key:
                out.append(rowids[pos])
                pos += 1
            if pos < len(keys) or next_leaf == NULL_PAGE:
                break
            buf = self.pager.read_page(next_leaf)
            keys, rowids, next_leaf = deserialize_leaf(buf)
            pos = 0
        return out

    # -- insert ----------------------------------------------------------

    def _descend(self, key: Any, rowid: RowId) -> list[_Frame]:
        path = []
        page_id = self.root_page_id
        target = _composite(key, rowid)
        while True:
            buf = self.pager.read_page(page_id)
            if is_leaf_page(buf):
                keys, rowids, next_leaf = deserialize_leaf(buf)
                path.append(_Frame(page_id, True, keys, rowids=rowids, next_leaf=next_leaf))
                return path
            keys, sep_rowids, children = deserialize_internal(buf)
            composites = [_composite(k, r) for k, r in zip(keys, sep_rowids)]
            idx = bisect.bisect_right(composites, target)
            path.append(_Frame(page_id, False, keys, rowids=sep_rowids, children=children, child_idx=idx))
            page_id = children[idx]

    def insert(self, key: Any, rowid: RowId) -> None:
        path = self._descend(key, rowid)
        leaf = path[-1]
        composites = [_composite(k, r) for k, r in zip(leaf.keys, leaf.rowids)]
        pos = bisect.bisect_left(composites, _composite(key, rowid))
        leaf.keys.insert(pos, key)
        leaf.rowids.insert(pos, rowid)

        promote = self._write_leaf_or_split(leaf)
        if promote is None:
            return

        for frame in reversed(path[:-1]):
            sep_key, sep_rowid, right_child_id = promote
            frame.keys.insert(frame.child_idx, sep_key)
            frame.rowids.insert(frame.child_idx, sep_rowid)
            frame.children.insert(frame.child_idx + 1, right_child_id)
            promote = self._write_internal_or_split(frame)
            if promote is None:
                return

        # The root itself split; build a fresh root pointing at both halves.
        sep_key, sep_rowid, right_child_id = promote
        left_root_id = path[0].page_id
        new_root_id = self.pager.allocate_page()
        self.pager.write_page(
            new_root_id, serialize_internal([sep_key], [sep_rowid], [left_root_id, right_child_id])
        )
        self.root_page_id = new_root_id
        self._depth += 1

    def _write_leaf_or_split(self, frame: _Frame) -> tuple[Any, RowId, int] | None:
        try:
            self.pager.write_page(frame.page_id, serialize_leaf(frame.keys, frame.rowids, frame.next_leaf))
            return None
        except OverflowError:
            pass
        mid = len(frame.keys) // 2
        right_keys, right_rowids = frame.keys[mid:], frame.rowids[mid:]
        left_keys, left_rowids = frame.keys[:mid], frame.rowids[:mid]
        new_page_id = self.pager.allocate_page()
        self.pager.write_page(new_page_id, serialize_leaf(right_keys, right_rowids, frame.next_leaf))
        self.pager.write_page(frame.page_id, serialize_leaf(left_keys, left_rowids, new_page_id))
        self.split_count += 1
        return right_keys[0], right_rowids[0], new_page_id

    def _write_internal_or_split(self, frame: _Frame) -> tuple[Any, RowId, int] | None:
        try:
            self.pager.write_page(frame.page_id, serialize_internal(frame.keys, frame.rowids, frame.children))
            return None
        except OverflowError:
            pass
        mid = len(frame.keys) // 2
        up_key, up_rowid = frame.keys[mid], frame.rowids[mid]
        left_keys, right_keys = frame.keys[:mid], frame.keys[mid + 1:]
        left_rowids, right_rowids = frame.rowids[:mid], frame.rowids[mid + 1:]
        left_children, right_children = frame.children[:mid + 1], frame.children[mid + 1:]
        new_page_id = self.pager.allocate_page()
        self.pager.write_page(new_page_id, serialize_internal(right_keys, right_rowids, right_children))
        self.pager.write_page(frame.page_id, serialize_internal(left_keys, left_rowids, left_children))
        self.split_count += 1
        return up_key, up_rowid, new_page_id

    # -- introspection (used by planner.py cost model, structure_stats.py,
    #    correctness.py) ----

    def depth(self) -> int:
        return self._depth

    def leftmost_leaf(self) -> int:
        page_id = self.root_page_id
        while True:
            buf = self.pager.read_page(page_id)
            if is_leaf_page(buf):
                return page_id
            _, _, children = deserialize_internal(buf)
            page_id = children[0]

    def iter_leaves(self):
        """Yield (page_id, keys, rowids) for every leaf, left to right."""
        page_id = self.leftmost_leaf()
        while page_id != NULL_PAGE:
            buf = self.pager.read_page(page_id)
            keys, rowids, next_leaf = deserialize_leaf(buf)
            yield page_id, keys, rowids
            page_id = next_leaf

    def walk_nodes(self):
        """Yield (page_id, is_leaf, num_entries) for every node in the tree,
        via a stack-based DFS from the root."""
        stack = [self.root_page_id]
        seen = set()
        while stack:
            page_id = stack.pop()
            if page_id in seen:
                continue
            seen.add(page_id)
            buf = self.pager.read_page(page_id)
            if is_leaf_page(buf):
                keys, _, _ = deserialize_leaf(buf)
                yield page_id, True, len(keys)
            else:
                keys, _, children = deserialize_internal(buf)
                yield page_id, False, len(children)
                stack.extend(children)

    def validate(self) -> list[str]:
        """Check core B+tree invariants; return a list of violations
        (empty list == healthy tree)."""
        errors = []

        # 1. Every leaf's keys are sorted.
        # 2. Keys are non-decreasing across the whole leaf chain (global order).
        prev_last = None
        for page_id, keys, rowids in self.iter_leaves():
            if keys != sorted(keys):
                errors.append(f"leaf {page_id} keys not sorted: {keys}")
            if len(keys) != len(rowids):
                errors.append(f"leaf {page_id} keys/rowids length mismatch")
            if prev_last is not None and keys and keys[0] < prev_last:
                errors.append(f"leaf {page_id} out of global order (first={keys[0]} < prev_last={prev_last})")
            if keys:
                prev_last = keys[-1]

        # 3. All leaves at equal depth.
        def _depths_from(page_id, depth, acc):
            buf = self.pager.read_page(page_id)
            if is_leaf_page(buf):
                acc.append(depth)
                return
            _, _, children = deserialize_internal(buf)
            for c in children:
                _depths_from(c, depth + 1, acc)

        depths: list[int] = []
        _depths_from(self.root_page_id, 1, depths)
        if len(set(depths)) > 1:
            errors.append(f"leaves at unequal depths: {sorted(set(depths))}")
        elif depths and depths[0] != self._depth:
            errors.append(f"cached depth drifted: cached={self._depth} actual={depths[0]}")

        # 4. Internal separator keys sorted + child/key counts consistent.
        # (Page-size capacity is enforced structurally by serialize_*
        # raising OverflowError, so anything on disk already fits.)
        for page_id, is_leaf, _ in self.walk_nodes():
            buf = self.pager.read_page(page_id)
            if not is_leaf:
                keys, rowids, children = deserialize_internal(buf)
                if keys != sorted(keys):
                    errors.append(f"internal {page_id} separator keys not sorted: {keys}")
                if len(children) != len(keys) + 1:
                    errors.append(f"internal {page_id} has {len(children)} children for {len(keys)} keys")

        return errors
