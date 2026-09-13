"""Page layouts: the meta page and the slotted heap page.

Heap pages use a classic slotted-page layout: a slot directory grows
forward from the header, row bytes ("cells") are packed backward from the
end of the page, and free space is whatever lies between the two. This is
the same technique real disk-based databases (SQLite, Postgres) use to pack
variable-length rows into a fixed-size page.
"""
from __future__ import annotations

import json
import struct

from .pager import PAGE_SIZE, NULL_PAGE

META_PAGE_ID = 0

PAGE_TYPE_HEAP = 1
PAGE_TYPE_BTREE_LEAF = 2
PAGE_TYPE_BTREE_INTERNAL = 3

MAGIC = b"SQEN"
VERSION = 1

# magic, version, page_count, heap_first, heap_last, n_heap_pages, btree_root, schema_len
_META_HEADER = struct.Struct("<4sBIIIIII")


def pack_meta(page_count: int, heap_first: int, heap_last: int, n_heap_pages: int,
              btree_root: int, schema_json: bytes) -> bytes:
    header = _META_HEADER.pack(
        MAGIC, VERSION, page_count, heap_first, heap_last, n_heap_pages, btree_root, len(schema_json)
    )
    body = header + schema_json
    if len(body) > PAGE_SIZE:
        raise ValueError("schema too large for meta page")
    return body + bytes(PAGE_SIZE - len(body))


def unpack_meta(buf: bytes) -> dict:
    magic, version, page_count, heap_first, heap_last, n_heap_pages, btree_root, schema_len = \
        _META_HEADER.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError("not a sqlengine file (bad magic)")
    offset = _META_HEADER.size
    schema_json = bytes(buf[offset:offset + schema_len])
    return {
        "page_count": page_count,
        "heap_first": heap_first,
        "heap_last": heap_last,
        "n_heap_pages": n_heap_pages,
        "btree_root": btree_root,
        "schema": json.loads(schema_json.decode("utf-8")),
    }


_HEAP_HEADER = struct.Struct("<BIHH")  # page_type, next_page_id, num_slots, cell_start
_SLOT = struct.Struct("<HH")  # cell offset, cell length


class HeapPage:
    def __init__(self, buf: bytearray):
        page_type, next_page_id, num_slots, cell_start = _HEAP_HEADER.unpack_from(buf, 0)
        if page_type != PAGE_TYPE_HEAP:
            raise ValueError(f"not a heap page (type={page_type})")
        self.buf = buf
        self.next_page_id = next_page_id
        self.num_slots = num_slots
        self.cell_start = cell_start

    @staticmethod
    def new() -> "HeapPage":
        buf = bytearray(PAGE_SIZE)
        _HEAP_HEADER.pack_into(buf, 0, PAGE_TYPE_HEAP, NULL_PAGE, 0, PAGE_SIZE)
        return HeapPage(buf)

    def _flush_header(self) -> None:
        _HEAP_HEADER.pack_into(self.buf, 0, PAGE_TYPE_HEAP, self.next_page_id, self.num_slots, self.cell_start)

    def free_space(self) -> int:
        return self.cell_start - (_HEAP_HEADER.size + _SLOT.size * self.num_slots)

    def can_fit(self, data_len: int) -> bool:
        return self.free_space() >= _SLOT.size + data_len

    def insert(self, data: bytes) -> int:
        if not self.can_fit(len(data)):
            raise ValueError("heap page full")
        new_cell_start = self.cell_start - len(data)
        self.buf[new_cell_start:new_cell_start + len(data)] = data
        slot_off = _HEAP_HEADER.size + _SLOT.size * self.num_slots
        _SLOT.pack_into(self.buf, slot_off, new_cell_start, len(data))
        self.cell_start = new_cell_start
        slot_index = self.num_slots
        self.num_slots += 1
        self._flush_header()
        return slot_index

    def get(self, slot_index: int) -> bytes:
        if slot_index < 0 or slot_index >= self.num_slots:
            raise IndexError(f"slot {slot_index} out of range (num_slots={self.num_slots})")
        slot_off = _HEAP_HEADER.size + _SLOT.size * slot_index
        offset, length = _SLOT.unpack_from(self.buf, slot_off)
        return bytes(self.buf[offset:offset + length])

    def all_rows(self):
        for i in range(self.num_slots):
            yield i, self.get(i)

    def dump(self) -> bytes:
        self._flush_header()
        return bytes(self.buf)
