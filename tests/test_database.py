"""
Unit tests for database operations.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.database.sqlite import SQLiteClient


class TestSQLiteClient(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.db = SQLiteClient(self.db_path)
        self.db.connect()

    def tearDown(self):
        self.db.close()
        self.temp_dir.cleanup()

    def test_initialize_schema(self):
        self.db.initialize_schema()
        # Verify tables exist
        tables = ["macro_events", "trades", "feature_store", "predictions", "walk_forward_results", "experiments"]
        for table in tables:
            self.assertTrue(self.db.table_exists(table))

    def test_write_and_read_dataframe(self):
        self.db.initialize_schema()
        
        df = pd.DataFrame({
            "timestamp_utc": ["2023-01-01 10:00:00", "2023-01-02 10:00:00"],
            "event_name": ["CPI", "NFP"],
            "country": ["USD", "USD"],
            "importance": ["high", "high"],
            "actual": [2.1, 1.5],
            "forecast": [2.0, 1.4],
            "previous": [1.9, 1.3],
            "surprise": [0.1, 0.1]
        })
        
        self.db.write_dataframe(df, "macro_events", if_exists="append")
        self.assertEqual(self.db.row_count("macro_events"), 2)
        
        read_df = self.db.read_table("macro_events")
        self.assertEqual(len(read_df), 2)
        self.assertEqual(read_df.iloc[0]["event_name"], "CPI")

    def test_execute_and_fetch(self):
        self.db.initialize_schema()
        self.db.execute("INSERT INTO experiments (experiment_id, model) VALUES (?, ?)", ("exp1", "xgboost"))
        
        row = self.db.fetchone("SELECT * FROM experiments WHERE experiment_id = ?", ("exp1",))
        self.assertIsNotNone(row)
        self.assertEqual(row["model"], "xgboost")
        
        rows = self.db.fetchall("SELECT * FROM experiments")
        self.assertEqual(len(rows), 1)

if __name__ == "__main__":
    unittest.main()
