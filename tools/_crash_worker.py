#!/usr/bin/env python3
"""Child process for the crash-recovery test in correctness.py.

Inserts rows one at a time and prints "OK <id>" immediately after each
insert() call returns -- i.e., after that row's fsync has completed and it
is durable. The parent process reads this stream and SIGKILLs us at an
arbitrary point to simulate a crash mid-insert.
"""
import sys

from sqlengine import Engine


def main():
    db_path, n_rows_str, start_id_str = sys.argv[1], sys.argv[2], sys.argv[3]
    n_rows = int(n_rows_str)
    start_id = int(start_id_str)

    engine = Engine(db_path)
    if start_id == 0:
        engine.create_table("t", [("id", "INTEGER"), ("value", "STRING")], index_column="id")
    else:
        engine.open()

    for i in range(start_id, start_id + n_rows):
        engine.insert((i, f"val_{i}"))
        print(f"OK {i}", flush=True)

    engine.close()


if __name__ == "__main__":
    main()
