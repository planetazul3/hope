import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import tick_collector


class TestTickCollectorBatchInsert(unittest.TestCase):
    def setUp(self):
        self.store = tick_collector.TickStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_insert_batch_counts_only_new_rows(self):
        first = self.store.insert_batch("R_100", [1, 2, 3], [10.0, 20.0, 30.0])
        second = self.store.insert_batch("R_100", [2, 3, 4], [20.0, 30.0, 40.0])

        self.assertEqual(first, 3)
        self.assertEqual(second, 1)
        self.assertEqual(self.store.get_count("R_100"), 4)


if __name__ == "__main__":
    unittest.main()
