from sqlengine import Engine

# Large enough that the table spans several heap pages, so a single-row
# equality seek (cost ~= btree.depth() + 1) genuinely beats a full scan
# (cost = n_heap_pages). At just a handful of rows everything fits on one
# heap page and the cost-based planner correctly prefers scan instead --
# see test_small_table_prefers_scan_even_with_index below.
N = 3000


def build(tmp_path, index_column="id"):
    engine = Engine(tmp_path / "t.db")
    engine.create_table("users", [("id", "INTEGER"), ("name", "STRING")], index_column=index_column)
    for i in range(N):
        engine.insert((i, f"user{i}"))
    return engine


def test_select_star(tmp_path):
    engine = build(tmp_path)
    rows, stats = engine.execute("SELECT * FROM users")
    assert len(rows) == N
    assert stats.method == "scan"
    engine.close()


def test_index_seek_used_for_equality_on_indexed_column(tmp_path):
    engine = build(tmp_path)
    rows, stats = engine.execute("SELECT * FROM users WHERE id = 100")
    assert rows == [(100, "user100")]
    assert stats.method == "seek"
    assert stats.estimated_seek_cost < stats.estimated_scan_cost
    engine.close()


def test_scan_used_for_non_indexed_predicate(tmp_path):
    engine = build(tmp_path)
    rows, stats = engine.execute("SELECT * FROM users WHERE name = 'user100'")
    assert rows == [(100, "user100")]
    assert stats.method == "scan"
    engine.close()


def test_scan_used_for_range_predicate_even_on_indexed_column(tmp_path):
    engine = build(tmp_path)
    rows, stats = engine.execute("SELECT * FROM users WHERE id < 5")
    assert stats.method == "scan"
    assert sorted(r[0] for r in rows) == [0, 1, 2, 3, 4]
    engine.close()


def test_column_projection(tmp_path):
    engine = build(tmp_path)
    rows, _ = engine.execute("SELECT name FROM users WHERE id = 7")
    assert rows == [("user7",)]
    engine.close()


def test_no_index_falls_back_to_scan(tmp_path):
    engine = build(tmp_path, index_column=None)
    rows, stats = engine.execute("SELECT * FROM users WHERE id = 50")
    assert rows == [(50, "user50")]
    assert stats.method == "scan"
    engine.close()


def test_small_table_prefers_scan_even_with_index(tmp_path):
    """The planner is cost-based, not rule-based: an index existing on the
    predicate column isn't enough to trigger a seek. At just a few rows
    everything lives on one heap page (scan_cost=1), which is cheaper than
    any seek (seek_cost = depth + matches >= 2)."""
    engine = Engine(tmp_path / "small.db")
    engine.create_table("t", [("id", "INTEGER"), ("name", "STRING")], index_column="id")
    for i in range(5):
        engine.insert((i, f"n{i}"))
    rows, stats = engine.execute("SELECT * FROM t WHERE id = 3")
    assert rows == [(3, "n3")]
    assert stats.method == "scan"
    assert stats.estimated_scan_cost <= stats.estimated_seek_cost
    engine.close()


def test_high_selectivity_prefers_scan_despite_index(tmp_path):
    """A large table where the predicate matches nearly every row: even
    though the column is indexed and the predicate is an equality, seeking
    means fetching almost every row's heap page one at a time, which costs
    more than a single sequential scan."""
    engine = Engine(tmp_path / "dup.db")
    engine.create_table("t", [("id", "INTEGER"), ("name", "STRING")], index_column="id")
    for i in range(N):
        engine.insert((0, f"n{i}"))  # every row has the same indexed value
    rows, stats = engine.execute("SELECT * FROM t WHERE id = 0")
    assert len(rows) == N
    assert stats.method == "scan"
    engine.close()


def test_reopen_persists_data(tmp_path):
    path = tmp_path / "t.db"
    engine = Engine(path)
    engine.create_table("users", [("id", "INTEGER"), ("name", "STRING")], index_column="id")
    for i in range(N):
        engine.insert((i, f"user{i}"))
    n_heap_pages_before_close = engine.table.n_heap_pages
    engine.close()

    engine2 = Engine(path)
    engine2.open()
    assert engine2.table.n_heap_pages == n_heap_pages_before_close
    rows, stats = engine2.execute("SELECT * FROM users WHERE id = 2")
    assert rows == [(2, "user2")]
    assert stats.method == "seek"
    engine2.close()


def test_load_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,name\n1,alice\n2,bob\n3,carol\n")
    engine = Engine(tmp_path / "t.db")
    engine.create_table("users", [("id", "INTEGER"), ("name", "STRING")], index_column="id")
    count = engine.load_csv(csv_path)
    assert count == 3
    rows, _ = engine.execute("SELECT * FROM users WHERE id = 2")
    assert rows == [(2, "bob")]
    engine.close()
