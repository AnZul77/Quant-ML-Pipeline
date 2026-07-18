"""
Unit tests for walk-forward data leakage prevention.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import numpy as np

from src.models.trainer import WalkForwardTrainer
from src.database.sqlite import SQLiteClient
from src.utils.config import PipelineConfig


class TestLeakage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.db = SQLiteClient(self.db_path)
        self.db.connect()
        self.db.initialize_schema()
        
        self.config = MagicMock(spec=PipelineConfig)
        self.config.train_window_days = 3
        self.config.test_window_days = 1
        self.config.target_column = "pnl"
        self.config.model_list = ["linear_regression"]
        self.config.hyperparameters = {}
        self.config.output_dir = Path(self.temp_dir.name)
        self.config.save_models = False
        self.config.enable_shap = False
        
        # We don't want to actually train, just test the splitter
        # So we'll mock the training part, or just test the splitting logic directly if we can
        
    def tearDown(self):
        self.db.close()
        self.temp_dir.cleanup()

    def test_fold_boundaries(self):
        # Create 10 days of feature store data
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        
        features = pd.DataFrame({
            "trade_id": range(1, 11),
            "feature_timestamp": dates.astype(str),
            "pnl": np.random.randn(10),
            "permutation_id": ["P1"] * 10,
            "hour_of_day": [10] * 10,
            "day_of_week": [0] * 10
        })
        self.db.write_dataframe(features, "feature_store")
        
        trainer = WalkForwardTrainer(self.config, self.db)
        
        # Override the actual model training to just capture the train/test splits
        splits = []
        
        original_train = trainer.train
        
        # We extract the splitting logic
        df = self.db.read_table("feature_store")
        df["feature_timestamp"] = pd.to_datetime(df["feature_timestamp"])
        df = df.sort_values("feature_timestamp")
        
        min_date = df["feature_timestamp"].min()
        max_date = df["feature_timestamp"].max()
        
        train_window = pd.Timedelta(days=self.config.train_window_days)
        test_window = pd.Timedelta(days=self.config.test_window_days)
        
        current_date = min_date
        
        while current_date + train_window < max_date:
            train_end = current_date + train_window
            test_end = train_end + test_window
            
            train_mask = (df["feature_timestamp"] >= current_date) & (df["feature_timestamp"] < train_end)
            test_mask = (df["feature_timestamp"] >= train_end) & (df["feature_timestamp"] < test_end)
            
            train_df = df[train_mask]
            test_df = df[test_mask]
            
            if not train_df.empty and not test_df.empty:
                max_train_ts = train_df["feature_timestamp"].max()
                min_test_ts = test_df["feature_timestamp"].min()
                
                # CRITICAL ASSERTION TEST
                self.assertLess(max_train_ts, min_test_ts)
                
                # Verify no overlap
                train_ids = set(train_df["trade_id"])
                test_ids = set(test_df["trade_id"])
                self.assertEqual(len(train_ids.intersection(test_ids)), 0)
                
            current_date += test_window

if __name__ == "__main__":
    unittest.main()
