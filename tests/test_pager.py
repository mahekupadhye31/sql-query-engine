from sqlengine.storage.page import HeapPage
from sqlengine.storage.pager import PAGE_SIZE, NULL_PAGE, Pager


def test_pager_allocate_and_readwrite(tmp_path):
    p = Pager(tmp_path / "t.db", create=True)
    pid = p.allocate_page()
    assert pid == 0
    assert p.page_count == 1
    data = bytes([7]) * PAGE_SIZE
    p.write_page(pid, data)
    assert bytes(p.read_page(pid)) == data
    p.close()


def test_pager_persists_across_reopen(tmp_path):
    path = tmp_path / "t.db"
    p = Pager(path, create=True)
    pid = p.allocate_page()
    p.write_page(pid, bytes([9]) * PAGE_SIZE)
    p.fsync()
    p.close()

    p2 = Pager(path, create=False)
    assert p2.page_count == 1
    assert bytes(p2.read_page(pid)) == bytes([9]) * PAGE_SIZE
    p2.close()


def test_heap_page_insert_and_get():
    page = HeapPage.new()
    s0 = page.insert(b"hello")
    s1 = page.insert(b"world!")
    assert page.get(s0) == b"hello"
    assert page.get(s1) == b"world!"
    assert page.next_page_id == NULL_PAGE


def test_heap_page_roundtrip_through_dump():
    page = HeapPage.new()
    page.insert(b"row-a")
    page.insert(b"row-b")
    buf = page.dump()
    assert len(buf) == PAGE_SIZE

    page2 = HeapPage(bytearray(buf))
    assert page2.num_slots == 2
    assert page2.get(0) == b"row-a"
    assert page2.get(1) == b"row-b"


def test_heap_page_fills_up():
    page = HeapPage.new()
    count = 0
    row = b"x" * 100
    while page.can_fit(len(row)):
        page.insert(row)
        count += 1
    assert count > 0
    assert not page.can_fit(len(row))
