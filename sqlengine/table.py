"""Ties the heap (table storage) and the B+tree (secondary index) together
behind a single-table API: insert, get_row, and a full scan generator.

Durability boundary: insert() ends with exactly one fsync() call after all
pages touched by the insert (heap page(s), any b+tree split pages, the meta
page) have been written. If the process dies before that fsync returns,
the insert is simply not durable -- nothing before it is affected, because
every prior insert already completed its own fsync before returning.
"""
from __future__ import annotations

import json
from pathlib import Path

from .storage.pager import Pager, NULL_PAGE
from .storage.page import HeapPage, pack_meta, unpack_meta, META_PAGE_ID
from .btree import BTree
from .types import Schema, Row, RowId, encode_row, decode_row, encode_key


class Table:
    def __init__(self, pager: Pager, schema: Schema, heap_first: int, heap_last: int,
                 n_heap_pages: int, btree: BTree | None):
        self.pager = pager
        self.schema = schema
        self.heap_first = heap_first
        self.heap_last = heap_last
        # Maintained as an O(1) counter (incremented only when a new heap
        # page is allocated) rather than recomputed by walking the chain,
        # so the planner's cost model can look up "how many pages would a
        # full scan touch" without that lookup itself costing as much as
        # the scan it's trying to avoid.
        self.n_heap_pages = n_heap_pages
        self.btree = btree

    @staticmethod
    def create(path: str | Path, schema: Schema) -> "Table":
        pager = Pager(path, create=True)
        meta_id = pager.allocate_page()
        assert meta_id == META_PAGE_ID, "meta page must be page 0"
        heap_id = pager.allocate_page()
        pager.write_page(heap_id, HeapPage.new().dump())
        btree = BTree.create_empty(pager) if schema.index_column else None
        table = Table(pager, schema, heap_id, heap_id, 1, btree)
        table._write_meta()
        pager.fsync()
        return table

    @staticmethod
    def open(path: str | Path) -> "Table":
        pager = Pager(path, create=False)
        meta = unpack_meta(pager.read_page(META_PAGE_ID))
        schema = Schema.from_json(meta["schema"])
        btree = None
        if schema.index_column and meta["btree_root"] != NULL_PAGE:
            btree = BTree(pager, meta["btree_root"])
        return Table(pager, schema, meta["heap_first"], meta["heap_last"], meta["n_heap_pages"], btree)

    def _write_meta(self) -> None:
        btree_root = self.btree.root_page_id if self.btree else NULL_PAGE
        schema_json = json.dumps(self.schema.to_json()).encode("utf-8")
        buf = pack_meta(
            self.pager.page_count, self.heap_first, self.heap_last, self.n_heap_pages, btree_root, schema_json
        )
        self.pager.write_page(META_PAGE_ID, buf)

    def insert(self, row: Row) -> RowId:
        data = encode_row(self.schema, row)
        page = HeapPage(self.pager.read_page(self.heap_last))
        target_page_id = self.heap_last

        if not page.can_fit(len(data)):
            new_page_id = self.pager.allocate_page()
            page.next_page_id = new_page_id
            self.pager.write_page(target_page_id, page.dump())
            self.heap_last = new_page_id
            target_page_id = new_page_id
            page = HeapPage.new()
            self.n_heap_pages += 1

        slot = page.insert(data)
        self.pager.write_page(target_page_id, page.dump())
        rowid = RowId(target_page_id, slot)

        if self.btree is not None:
            idx = self.schema.col_index(self.schema.index_column)
            key = encode_key(self.schema.col_type(self.schema.index_column), row[idx])
            self.btree.insert(key, rowid)

        self._write_meta()
        self.pager.fsync()
        return rowid

    def get_row(self, rowid: RowId) -> Row:
        page = HeapPage(self.pager.read_page(rowid.page_id))
        return decode_row(self.schema, page.get(rowid.slot))

    def scan(self):
        page_id = self.heap_first
        while page_id != NULL_PAGE:
            page = HeapPage(self.pager.read_page(page_id))
            for _, data in page.all_rows():
                yield decode_row(self.schema, data)
            page_id = page.next_page_id

    def count_heap_pages_by_walking(self) -> int:
        """O(n_heap_pages) ground truth, used only to validate that the
        cached self.n_heap_pages counter (used by the planner's cost model)
        hasn't drifted -- see tools/correctness.py."""
        count = 0
        page_id = self.heap_first
        while page_id != NULL_PAGE:
            page = HeapPage(self.pager.read_page(page_id))
            count += 1
            page_id = page.next_page_id
        return count

    def close(self) -> None:
        self.pager.close()
