"""
Unit tests for data cleaning and VWAP deduplication.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import numpy as np

from src.data.cleaning import TradeCleaner
from src.database.sqlite import SQLiteClient
from src.utils.config import PipelineConfig


class TestCleaning(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.db = SQLiteClient(self.db_path)
        self.db.connect()
        self.db.initialize_schema()
        
        # Mock config
        self.config = MagicMock(spec=PipelineConfig)
        self.config.invalid_accounts = ["ACC_TEST", "ACC_INVALID"]
        self.config.dedup_window_ms = 500
        self.config.project_root = Path(self.temp_dir.name)

    def tearDown(self):
        self.db.close()
        self.temp_dir.cleanup()

    def test_vwap_deduplication(self):
        # Two trades within 500ms, same account, same permutation
        raw_trades = pd.DataFrame({
            "timestamp": ["2023-01-01 10:00:00.100", "2023-01-01 10:00:00.200", "2023-01-01 10:05:00.000"],
            "account": ["ACC_1", "ACC_1", "ACC_1"],
            "direction": ["BUY", "BUY", "SELL"],
            "quantity": [10.0, 20.0, 15.0],
            "price": [100.0, 103.0, 105.0],
            "pnl": [5.0, 10.0, -2.0],
            "permutation_id": ["P1", "P1", "P1"],
            "holding_time": [60.0, 60.0, 60.0],
            "threshold": [1.5, 1.5, 1.5]
        })
        self.db.write_dataframe(raw_trades, "trades", if_exists="append")
        
        cleaner = TradeCleaner(self.config, self.db)
        cleaner.clean()
        
        cleaned = self.db.read_table("trades")
        
        # Should have 2 trades (first two grouped, third is separate)
        self.assertEqual(len(cleaned), 2)
        
        # VWAP check for grouped trade
        # (10*100 + 20*103) / 30 = (1000 + 2060) / 30 = 3060 / 30 = 102.0
        grouped_trade = cleaned.iloc[0]
        self.assertAlmostEqual(grouped_trade["price"], 102.0)
        self.assertAlmostEqual(grouped_trade["quantity"], 30.0)
        self.assertAlmostEqual(grouped_trade["pnl"], 15.0)

    def test_invalid_account_filtering(self):
        raw_trades = pd.DataFrame({
            "timestamp": ["2023-01-01 10:00:00.000", "2023-01-01 10:01:00.000"],
            "account": ["ACC_1", "ACC_TEST"],
            "direction": ["BUY", "SELL"],
            "quantity": [10.0, 20.0],
            "price": [100.0, 103.0],
            "pnl": [5.0, 10.0],
            "permutation_id": ["P1", "P1"],
            "holding_time": [60.0, 60.0],
            "threshold": [1.5, 1.5]
        })
        self.db.write_dataframe(raw_trades, "trades", if_exists="append")
        
        cleaner = TradeCleaner(self.config, self.db)
        cleaner.clean()
        
        cleaned = self.db.read_table("trades")
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["account"], "ACC_1")

if __name__ == "__main__":
    unittest.main()
