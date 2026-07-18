"""
Unit tests for feature engineering.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import numpy as np

from src.features.feature_engineer import FeatureEngineer
from src.database.sqlite import SQLiteClient
from src.utils.config import PipelineConfig


class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.db = SQLiteClient(self.db_path)
        self.db.connect()
        self.db.initialize_schema()
        
        self.config = MagicMock(spec=PipelineConfig)
        self.config.feature_version = "1.0"
        self.config.rolling_short = 2
        self.config.rolling_long = 3
        self.config.market_sessions = {
            "asia_open": "00:00", "asia_close": "09:00",
            "europe_open": "07:00", "europe_close": "16:00",
            "us_open": "13:30", "us_close": "20:00"
        }
        self.config.macro_proximity_thresholds = [30, 60, 120]

    def tearDown(self):
        self.db.close()
        self.temp_dir.cleanup()

    def test_rolling_features_shift(self):
        # Insert test trades
        trades = pd.DataFrame({
            "id": [1, 2, 3],
            "timestamp": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
            "account": ["ACC_1", "ACC_1", "ACC_1"],
            "direction": ["BUY", "BUY", "BUY"],
            "quantity": [10.0, 10.0, 10.0],
            "price": [100.0, 101.0, 102.0],
            "pnl": [5.0, 10.0, 15.0],
            "permutation_id": ["P1", "P1", "P1"],
            "holding_time": [60.0, 60.0, 60.0],
            "threshold": [1.5, 1.5, 1.5]
        })
        self.db.write_dataframe(trades, "trades", if_exists="append")
        
        fe = FeatureEngineer(self.config, self.db)
        fe.build_features()
        
        features = self.db.read_table("feature_store").sort_values("feature_timestamp")
        
        # First row should have NaN for rolling features because of shift(1)
        row1 = features.iloc[0]
        self.assertTrue(pd.isna(row1["rolling_pnl_5"]))
        
        # Second row should have rolling stats based on first row only
        row2 = features.iloc[1]
        self.assertAlmostEqual(row2["rolling_pnl_5"], 5.0) # mean of [5.0]
        
        # Third row should have rolling stats based on first two rows
        row3 = features.iloc[2]
        self.assertAlmostEqual(row3["rolling_pnl_5"], 7.5) # mean of [5.0, 10.0]

if __name__ == "__main__":
    unittest.main()
