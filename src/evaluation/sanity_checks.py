import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PipelineSanityError(Exception):
    """Exception raised when a pipeline sanity check fails."""
    pass

class SanityChecker:
    @staticmethod
    def check_feature_matrix(df: pd.DataFrame, feature_cols: list) -> None:
        """Sanity check the feature matrix before training."""
        logger.info("Running sanity checks on feature matrix...")
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            if df['timestamp'].duplicated().any():
                raise PipelineSanityError("Sanity Check Failed: Duplicate timestamps detected in feature matrix.")
                
        X = df[feature_cols]
        
        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            raise PipelineSanityError("Sanity Check Failed: Infinite values detected in feature matrix.")
            
        # Check for zero variance target
        if 'pnl' in df.columns:
            if df['pnl'].var() == 0:
                raise PipelineSanityError("Sanity Check Failed: Target variance is exactly zero.")

        # Check for rank deficiency
        clean_X = X.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_X) > len(feature_cols):
            rank = np.linalg.matrix_rank(clean_X.values)
            if rank < len(feature_cols):
                logger.warning(f"Sanity Check Warning: Feature matrix is rank deficient (Rank: {rank}, Features: {len(feature_cols)}). Multicollinearity exists.")
                
        logger.info("Feature matrix sanity checks passed.")

    @staticmethod
    def check_optuna_params(params: dict) -> None:
        """Sanity check Optuna parameters."""
        if not params:
            raise PipelineSanityError("Sanity Check Failed: Optuna returned empty parameters.")
        for k, v in params.items():
            if pd.isna(v) or v is None:
                raise PipelineSanityError(f"Sanity Check Failed: Optuna returned invalid parameter {k}={v}")

    @staticmethod
    def check_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Sanity check predictions."""
        logger.info("Running sanity checks on predictions...")
        
        if len(np.unique(y_pred)) == 1:
            raise PipelineSanityError("Sanity Check Failed: All predictions are identical. Model has collapsed.")
            
    @staticmethod
    def check_metrics(metrics: dict) -> None:
        """Sanity check calculated metrics."""
        mae = metrics.get('MAE')
        r2 = metrics.get('R2')
        
        if pd.isna(mae) or pd.isna(r2):
            raise PipelineSanityError(f"Sanity Check Failed: Invalid metrics computed (MAE: {mae}, R2: {r2}).")

    @staticmethod
    def check_feature_importance(importance_dict: dict) -> None:
        """Sanity check feature importances."""
        if not importance_dict:
            return
            
        total_importance = sum(abs(v) for v in importance_dict.values())
        if total_importance == 0:
            raise PipelineSanityError("Sanity Check Failed: All feature importances are exactly zero.")
