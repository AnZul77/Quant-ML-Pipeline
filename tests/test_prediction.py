"""
Unit tests for the prediction pipeline.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

from src.models.predictor import PermutationPredictor
from src.database.sqlite import SQLiteClient
from src.utils.config import PipelineConfig


class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.db = SQLiteClient(self.db_path)
        self.db.connect()
        self.db.initialize_schema()
        
        self.config = MagicMock(spec=PipelineConfig)
        self.config.output_dir = Path(self.temp_dir.name)
        
    def tearDown(self):
        self.db.close()
        self.temp_dir.cleanup()

    def test_recommendation_matrix_shape(self):
        # We need to mock the prediction part to just return some dummy data
        predictor = PermutationPredictor(self.config, self.db)
        
        # Mock predict_best_permutations
        dummy_best = []
        for d in range(7):
            for h in range(24):
                dummy_best.append({
                    "day_of_week": d,
                    "hour_of_day": h,
                    "best_permutation": f"P_{d}_{h}",
                    "predicted_pnl": float(d + h)
                })
        
        dummy_df = pd.DataFrame(dummy_best)
        predictor.predict_best_permutations = MagicMock(return_value=dummy_df)
        
        matrix = predictor.generate_recommendation_matrix("exp1")
        
        # Should be 7 rows (days), 24 columns (hours)
        self.assertEqual(matrix.shape, (7, 24))
        
        # Cell (Monday, 00:00) should be P_0_0
        self.assertEqual(matrix.loc["Monday", "00:00"], "P_0_0")

if __name__ == "__main__":
    unittest.main()
