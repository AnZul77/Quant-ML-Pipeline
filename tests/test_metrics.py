"""
Unit tests for trading and regression metrics.
"""

import unittest
from unittest.mock import MagicMock

import pandas as pd
import numpy as np

from src.evaluation.evaluator import PipelineEvaluator
from src.utils.config import PipelineConfig
from src.database.sqlite import SQLiteClient


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock(spec=PipelineConfig)
        self.config.annualization_factor = 252
        self.config.risk_free_rate = 0.0
        
        self.db = MagicMock(spec=SQLiteClient)
        
        self.evaluator = PipelineEvaluator(self.config, self.db)

    def test_regression_metrics(self):
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        
        metrics = self.evaluator.compute_regression_metrics(y_true, y_pred)
        
        self.assertAlmostEqual(metrics["MAE"], 0.5)
        # MSE = (0.25 + 0.25 + 0 + 1) / 4 = 1.5 / 4 = 0.375
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(0.375))
        self.assertIn("R2", metrics)

    def test_trading_metrics(self):
        # Create a simple series of returns
        returns = pd.Series([1.0, 2.0, -1.0, 3.0, -2.0, 1.0])
        
        metrics = self.evaluator.compute_trading_metrics(returns)
        
        # Win rate: 4 positive out of 6
        self.assertAlmostEqual(metrics["Win Rate"], 4/6)
        
        # Average Trade: mean of returns
        self.assertAlmostEqual(metrics["Average Trade"], returns.mean())
        
        # Profit Factor: sum(pos) / abs(sum(neg))
        # pos = 1+2+3+1 = 7, neg = -1-2 = -3 => 7/3
        self.assertAlmostEqual(metrics["Profit Factor"], 7/3)
        
        # Max Drawdown
        # cumsum = [1, 3, 2, 5, 3, 4]
        # cummax = [1, 3, 3, 5, 5, 5]
        # drawdown = [0, 0, -1, 0, -2, -1]
        self.assertAlmostEqual(metrics["Maximum Drawdown"], -2.0)
        
        self.assertIn("Sharpe Ratio", metrics)
        self.assertIn("Sortino Ratio", metrics)
        self.assertIn("Calmar Ratio", metrics)

    def test_empty_trading_metrics(self):
        with self.assertRaises(ValueError):
            self.evaluator.compute_trading_metrics(pd.Series(dtype=float))

if __name__ == "__main__":
    unittest.main()
