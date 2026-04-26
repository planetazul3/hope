import os
import sqlite3
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import tick_collector


class TestTickCollectorBatchInsert(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("""
            CREATE TABLE ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch REAL NOT NULL,
                quote REAL NOT NULL,
                UNIQUE(epoch, quote)
            )
        """)

    def tearDown(self):
        self.conn.close()

    def test_insert_batch_counts_only_new_rows(self):
        first = tick_collector.insert_batch(
            self.conn,
            [1, 2, 3],
            [10.0, 20.0, 30.0],
        )
        second = tick_collector.insert_batch(
            self.conn,
            [2, 3, 4],
            [20.0, 30.0, 40.0],
        )

        self.assertEqual(first, 3)
        self.assertEqual(second, 1)

        total_rows = self.conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
        self.assertEqual(total_rows, 4)


if __name__ == "__main__":
    unittest.main()
