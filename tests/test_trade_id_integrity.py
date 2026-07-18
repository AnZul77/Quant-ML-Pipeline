"""
Unit tests for trade_id integrity across the pipeline.
"""

import unittest
from pathlib import Path
import tempfile

import pandas as pd

from src.database.sqlite import SQLiteClient
from src.utils.config import PipelineConfig

class TestTradeIdIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.db = SQLiteClient(self.db_path)
        self.db.connect()
        self.db.initialize_schema()
        
    def tearDown(self):
        self.db.close()
        self.temp_dir.cleanup()

    def test_foreign_key_integrity(self):
        # 1. Insert trades
        trades_df = pd.DataFrame([
            {
                "id": 1, "timestamp": "2023-01-01 10:00:00", "account": "ACC_1",
                "direction": "LONG", "quantity": 100, "price": 50.0, "pnl": 10.0,
                "permutation_id": "P_1", "holding_time": 60, "threshold": 0.5
            },
            {
                "id": 2, "timestamp": "2023-01-01 11:00:00", "account": "ACC_1",
                "direction": "SHORT", "quantity": 100, "price": 51.0, "pnl": -5.0,
                "permutation_id": "P_1", "holding_time": 60, "threshold": 0.5
            }
        ])
        self.db.write_dataframe(trades_df, "trades")
        
        # 2. Insert features with matching trade_ids
        features_df = pd.DataFrame([
            {
                "trade_id": 1, "feature_version": "1.0", "pipeline_version": "1.0",
                "feature_timestamp": "2023-01-01 10:00:00", "market_session": "europe",
                "event_importance": "high", "risk_parameters": "{}", "pnl": 10.0
            },
            {
                "trade_id": 2, "feature_version": "1.0", "pipeline_version": "1.0",
                "feature_timestamp": "2023-01-01 11:00:00", "market_session": "europe",
                "event_importance": "low", "risk_parameters": "{}", "pnl": -5.0
            }
        ])
        self.db.write_dataframe(features_df, "feature_store")
        
        # 3. Insert predictions with matching trade_ids
        preds_df = pd.DataFrame([
            {
                "trade_id": 1, "timestamp": "2023-01-01 10:00:00", "permutation_id": "P_1",
                "predicted_pnl": 8.0, "actual_pnl": 10.0, "walk_forward_fold": 0, "experiment_id": "exp_test"
            },
            {
                "trade_id": 2, "timestamp": "2023-01-01 11:00:00", "permutation_id": "P_1",
                "predicted_pnl": -4.0, "actual_pnl": -5.0, "walk_forward_fold": 0, "experiment_id": "exp_test"
            }
        ])
        self.db.write_dataframe(preds_df, "predictions")
        
        # 4. Verify join works
        query = """
            SELECT p.trade_id, t.pnl as realized_pnl, p.predicted_pnl
            FROM predictions p
            JOIN trades t ON p.trade_id = t.id
            WHERE p.experiment_id = 'exp_test'
        """
        result = self.db.read_sql(query)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["realized_pnl"], 10.0)
        self.assertEqual(result.iloc[1]["realized_pnl"], -5.0)

if __name__ == "__main__":
    unittest.main()
