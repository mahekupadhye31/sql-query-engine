"""Raw fixed-size page I/O over a single file.

Durability model: writes go through pwrite() with no per-call fsync. The
caller (Table.insert) fsyncs once at the end of each logical operation.
A single write(2)/pwrite(2) of a page-aligned, page-sized buffer is atomic
on local filesystems in practice, so a crash never tears a page mid-write --
it only ever loses whichever pages hadn't been fsync'd yet. That is the
durability boundary this engine promises: "insert() returned" == "durable".
There is no WAL and no multi-page transaction atomicity (out of scope --
this engine has no transactions).
"""
from __future__ import annotations

import os
from pathlib import Path

PAGE_SIZE = 4096
NULL_PAGE = 0xFFFFFFFF


class Pager:
    def __init__(self, path: str | Path, create: bool = False):
        self.path = str(path)
        exists = os.path.exists(self.path)
        if not exists and not create:
            raise FileNotFoundError(self.path)
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        self._fd = os.open(self.path, flags, 0o644)
        self.read_count = 0
        self.write_count = 0
        self.fsync_count = 0
        size = os.fstat(self._fd).st_size
        self._page_count = size // PAGE_SIZE

    @property
    def page_count(self) -> int:
        return self._page_count

    def allocate_page(self) -> int:
        page_id = self._page_count
        self._page_count += 1
        os.pwrite(self._fd, bytes(PAGE_SIZE), page_id * PAGE_SIZE)
        return page_id

    def read_page(self, page_id: int) -> bytearray:
        self.read_count += 1
        buf = os.pread(self._fd, PAGE_SIZE, page_id * PAGE_SIZE)
        if len(buf) < PAGE_SIZE:
            buf = buf + bytes(PAGE_SIZE - len(buf))
        return bytearray(buf)

    def write_page(self, page_id: int, data: bytes) -> None:
        if len(data) != PAGE_SIZE:
            raise ValueError("page must be exactly PAGE_SIZE bytes")
        self.write_count += 1
        os.pwrite(self._fd, data, page_id * PAGE_SIZE)

    def fsync(self) -> None:
        self.fsync_count += 1
        os.fsync(self._fd)

    def reset_counters(self) -> None:
        self.read_count = 0
        self.write_count = 0
        self.fsync_count = 0

    def close(self) -> None:
        os.close(self._fd)
