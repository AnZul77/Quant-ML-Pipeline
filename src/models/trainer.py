"""
Walk-forward validation trainer for the Quantitative ML Pipeline.

Implements time-series-aware model training with:
  - Walk-forward (expanding/sliding) cross-validation folds
  - Strict temporal leakage assertions per fold
  - Nested GridSearchCV with TimeSeriesSplit for hyperparameter tuning
  - Multi-model comparison (LinearRegression, RandomForest, XGBoost, LightGBM)
  - SHAP value computation for tree-based best models
  - Full experiment tracking and model serialization
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.database.base import DatabaseClient
from src.utils.config import PipelineConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Mapping from config names to sklearn-compatible model classes
_MODEL_REGISTRY: Dict[str, type] = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "random_forest": RandomForestRegressor,
}

# Lazy imports for optional heavy dependencies
try:
    from catboost import CatBoostRegressor
    _MODEL_REGISTRY["catboost"] = CatBoostRegressor
except ImportError:
    logger.warning("catboost not installed — CatBoostRegressor unavailable")
    
try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    logger.warning("optuna not installed — Optuna tuning unavailable")
    _OPTUNA_AVAILABLE = False

# Lazy imports for optional heavy dependencies
try:
    from xgboost import XGBRegressor
    _MODEL_REGISTRY["xgboost"] = XGBRegressor
except ImportError:
    logger.warning("xgboost not installed — XGBRegressor unavailable")

try:
    from lightgbm import LGBMRegressor
    _MODEL_REGISTRY["lightgbm"] = LGBMRegressor
except ImportError:
    logger.warning("lightgbm not installed — LGBMRegressor unavailable")


# Feature columns that are NOT used as model inputs
_META_COLUMNS = frozenset({
    "trade_id", "feature_version", "pipeline_version", "feature_timestamp",
    "trade_timestamp",
    "market_session", "risk_parameters", "permutation_id", "event_importance",
    "pnl",  # target
    "account",
})

# Columns that need label-encoding before model input
_CATEGORICAL_COLUMNS = frozenset({
    "market_session", "event_importance",
})


def _get_git_commit() -> str:
    """Return the current short git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class WalkForwardTrainer:
    """Walk-forward cross-validation trainer with multi-model comparison.

    Args:
        config: Pipeline configuration object.
        db: Database client implementing ``DatabaseClient``.
    """

    def __init__(self, config: PipelineConfig, db: DatabaseClient) -> None:
        self.config = config
        self.db = db

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def train(self, experiment_id: str) -> Dict[str, Any]:
        """Run walk-forward validation and persist all results.

        Args:
            experiment_id: Unique identifier for this experiment run.

        Returns:
            Summary dict with fold count, best model name, and aggregate metrics.

        Raises:
            AssertionError: If any fold violates temporal ordering (leakage).
            ValueError: If feature store is empty.
        """
        logger.info("=== Starting walk-forward training: experiment=%s ===", experiment_id)

        # 1. Load features
        data = self._load_feature_data()
        feature_cols = self._get_feature_columns(data)
        target_col = self.config.target_column

        logger.info("Feature matrix: %d rows × %d features", len(data), len(feature_cols))

        # 2. Generate walk-forward folds
        folds = self._generate_folds(data)
        logger.info("Generated %d walk-forward folds", len(folds))

        # 3. Train across folds
        all_fold_metrics: List[Dict[str, Any]] = []
        all_predictions: List[pd.DataFrame] = []
        all_importances: List[Dict[str, float]] = []
        best_models: List[Tuple[str, Any]] = []  # (model_name, fitted_pipeline)

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            logger.info("--- Fold %d/%d ---", fold_idx + 1, len(folds))

            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            # Verify no index overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0, (
                f"OVERLAP DETECTED in fold {fold_idx}: "
                f"{len(set(train_idx) & set(test_idx))} shared indices"
            )

            # CRITICAL LEAKAGE CHECK
            max_train_ts = train_data["trade_timestamp"].max()
            min_test_ts = test_data["trade_timestamp"].min()
            assert max_train_ts < min_test_ts, (
                f"LEAKAGE DETECTED in fold {fold_idx}: "
                f"max(train_ts)={max_train_ts} >= min(test_ts)={min_test_ts}"
            )
            logger.info(
                "Leakage check passed: train_end=%s < test_start=%s",
                max_train_ts,
                min_test_ts,
            )

            X_train = train_data[feature_cols]
            y_train = train_data[target_col].values
            X_test = test_data[feature_cols]
            y_test = test_data[target_col].values

            # Train all models for this fold
            fold_results = self._train_fold_models(X_train, y_train, feature_cols)

            # Select best model by validation MAE
            best_name, best_pipeline, best_val_mae = self._select_best_model(fold_results)
            logger.info(
                "Fold %d best model: %s (validation MAE=%.6f)",
                fold_idx,
                best_name,
                best_val_mae,
            )

            # Predict on test set
            y_pred = best_pipeline.predict(X_test)

            # Compute fold metrics
            fold_metrics = self._compute_metrics(y_test, y_pred)
            fold_metrics["fold"] = fold_idx
            fold_metrics["best_model"] = best_name
            fold_metrics["train_size"] = len(train_data)
            fold_metrics["test_size"] = len(test_data)
            fold_metrics["train_start"] = str(train_data["feature_timestamp"].min())
            fold_metrics["train_end"] = str(max_train_ts)
            fold_metrics["test_start"] = str(min_test_ts)
            fold_metrics["test_end"] = str(test_data["feature_timestamp"].max())

            logger.info(
                "Fold %d test metrics: MAE=%.6f, RMSE=%.6f, R²=%.6f",
                fold_idx,
                fold_metrics["mae"],
                fold_metrics["rmse"],
                fold_metrics["r2"],
            )

            all_fold_metrics.append(fold_metrics)

            # Build predictions DataFrame
            all_predictions.append(
                pd.DataFrame({
                    "trade_id": test_data["trade_id"],
                    "timestamp": test_data["trade_timestamp"],
                    "permutation_id": test_data["permutation_id"],
                    "predicted_pnl": y_pred,
                    "actual_pnl": y_test,
                    "walk_forward_fold": fold_idx,
                    "experiment_id": experiment_id,
                })
            )

            # Feature importances
            importances = self._extract_feature_importances(best_pipeline, best_name, feature_cols, X_test, y_test)
            all_importances.append(importances)

            best_models.append((best_name, best_pipeline))

            # SHAP values - only on final fold to save runtime
            is_final_fold = fold_idx == len(folds) - 1
            if is_final_fold and self.config.enable_shap:
                self._compute_shap_values(
                    best_pipeline, X_test, feature_cols, fold_idx, experiment_id
                )

        # 4. Persist results
        self._save_predictions(all_predictions)
        self._save_fold_metrics(all_fold_metrics, experiment_id)
        self._save_experiment_metadata(experiment_id, all_fold_metrics, feature_cols)

        # 5. Save models if enabled
        if self.config.save_models:
            self._serialize_models(best_models, experiment_id)

        # 6. Build summary
        summary = self._build_summary(all_fold_metrics, best_models)
        logger.info("=== Training complete: %d folds, aggregate MAE=%.6f ===",
                     len(folds), summary["aggregate_mae"])

        return summary

    # ------------------------------------------------------------------ #
    #  Data loading & preparation
    # ------------------------------------------------------------------ #

    def _load_feature_data(self) -> pd.DataFrame:
        """Load feature store and prepare for modeling."""
        sql = """
            SELECT f.*, t.timestamp as trade_timestamp
            FROM feature_store f
            JOIN trades t ON f.trade_id = t.id
        """
        data = self.db.read_sql(sql)
        if data.empty:
            raise ValueError("Feature store is empty. Run feature engineering first.")

        data["trade_timestamp"] = pd.to_datetime(data["trade_timestamp"], utc=True)

        # Sort by timestamp, then trade_id for deterministic ordering
        data.sort_values(["trade_timestamp", "trade_id"], inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Encode categorical columns as numeric
        if "market_session" in data.columns:
            session_map = {"asia": 0, "europe": 1, "us": 2, "off_hours": 3}
            data["market_session_encoded"] = data["market_session"].map(session_map).fillna(3).astype(int)

        if "event_importance" in data.columns:
            importance_map = {"low": 0, "medium": 1, "high": 2}
            data["event_importance_encoded"] = (
                data["event_importance"]
                .str.lower()
                .map(importance_map)
                .fillna(0)
                .astype(int)
            )

        # Fill remaining NaNs in numeric columns with 0
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(0)

        logger.info("Feature data loaded and prepared: %d rows", len(data))
        return data

    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Return the list of columns to use as model features."""
        feature_cols = [
            col for col in data.columns
            if col not in _META_COLUMNS
            and col not in _CATEGORICAL_COLUMNS
            and data[col].dtype in (np.float64, np.float32, np.int64, np.int32, int, float)
        ]
        # Ensure encoded categoricals are included
        for enc_col in ("market_session_encoded", "event_importance_encoded"):
            if enc_col in data.columns and enc_col not in feature_cols:
                feature_cols.append(enc_col)

        return sorted(feature_cols)

    # ------------------------------------------------------------------ #
    #  Walk-forward fold generation
    # ------------------------------------------------------------------ #

    def _generate_folds(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward (sliding window) train/test fold indices.

        Each fold uses ``train_window_days`` for training and ``test_window_days``
        for testing, sliding forward by ``test_window_days`` each step.
        """
        timestamps = data["trade_timestamp"]
        min_ts = timestamps.min()
        max_ts = timestamps.max()

        train_days = self.config.train_window_days
        test_days = self.config.test_window_days

        train_delta = pd.Timedelta(days=train_days)
        test_delta = pd.Timedelta(days=test_days)

        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        fold_start = min_ts

        while fold_start + train_delta + test_delta <= max_ts + pd.Timedelta(seconds=1):
            train_end = fold_start + train_delta
            test_end = train_end + test_delta

            train_mask = (timestamps >= fold_start) & (timestamps < train_end)
            test_mask = (timestamps >= train_end) & (timestamps < test_end)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) >= 10 and len(test_idx) >= 1:
                folds.append((train_idx, test_idx))

            # Slide forward by test window
            fold_start += test_delta

        if not folds:
            logger.warning(
                "No valid folds generated. Data spans %.1f days, need at least %d + %d",
                (max_ts - min_ts).total_seconds() / 86400,
                train_days,
                test_days,
            )

        return folds

    # ------------------------------------------------------------------ #
    #  Model training per fold
    # ------------------------------------------------------------------ #

    def _train_fold_models(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        feature_cols: List[str],
    ) -> Dict[str, Tuple[Pipeline, float]]:
        """Train all configured models on one fold.

        Returns:
            Dict mapping model_name to (fitted_pipeline, validation_mae).
        """
        results: Dict[str, Tuple[Pipeline, float]] = {}
        models_to_train = self.config.model_list

        for model_name in models_to_train:
            if model_name not in _MODEL_REGISTRY:
                logger.warning("Model '%s' not in registry — skipping", model_name)
                continue

            try:
                pipeline, val_mae = self._train_single_model(
                    model_name, X_train, y_train
                )
                results[model_name] = (pipeline, val_mae)
                logger.info("  %s -> validation MAE=%.6f", model_name, val_mae)
            except Exception as exc:
                logger.error("  %s training failed: %s", model_name, exc, exc_info=True)

        if getattr(self.config, "enable_stacking", False) and len(results) > 1:
            try:
                logger.info("  Training StackingRegressor on top of base models")
                base_estimators = [(name, pipe) for name, pipe in results.items()]
                stacker = StackingRegressor(estimators=base_estimators, final_estimator=Ridge())
                stacker.fit(X_train, y_train)
                
                # Estimate validation MAE for stacker
                cv = TimeSeriesSplit(n_splits=3)
                val_scores = []
                for train_cv_idx, val_cv_idx in cv.split(X_train):
                    clone_stacker = StackingRegressor(
                        estimators=[(n, self._clone_pipeline(_MODEL_REGISTRY[n], n, self.config.seed, scale=n in ("linear_regression", "ridge", "lasso", "elasticnet", "random_forest"))) for n in results],
                        final_estimator=Ridge()
                    )
                    clone_stacker.fit(X_train.iloc[train_cv_idx], y_train[train_cv_idx])
                    y_val_pred = clone_stacker.predict(X_train.iloc[val_cv_idx])
                    val_scores.append(mean_absolute_error(y_train[val_cv_idx], y_val_pred))
                
                stacker_val_mae = float(np.mean(val_scores))
                results["stacking"] = (stacker, stacker_val_mae)
                logger.info("  stacking -> validation MAE=%.6f", stacker_val_mae)
            except Exception as exc:
                logger.error("  stacking training failed: %s", exc, exc_info=True)

        return results

    def _train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
    ) -> Tuple[Pipeline, float]:
        """Build, optionally tune, and fit a single model pipeline.

        For models with hyperparameter grids, uses ``GridSearchCV`` with
        ``TimeSeriesSplit(n_splits=3)`` inside the training data.

        Returns:
            (fitted_pipeline, best_validation_mae)
        """
        model_cls = _MODEL_REGISTRY[model_name]
        seed = self.config.seed

        # Build sklearn Pipeline
        if model_name in ("linear_regression", "ridge", "lasso", "elasticnet", "random_forest"):
            # Scaling benefits linear models and can help RF in some cases
            steps: List[Tuple[str, Any]] = [
                ("scaler", StandardScaler()),
                ("model", self._instantiate_model(model_cls, model_name, seed)),
            ]
        else:
            # Tree-based models (XGBoost, LightGBM) don't need scaling
            # but we wrap in a Pipeline for consistent API
            steps = [
                ("model", self._instantiate_model(model_cls, model_name, seed)),
            ]

        pipeline = Pipeline(steps)

        # Hyperparameter tuning
        param_grid = self.config.hyperparameters.get(model_name, {})
        tuner = getattr(self.config, "tuner", "gridsearch")

        if param_grid and model_name not in ("linear_regression", "ridge", "lasso", "elasticnet") and tuner == "optuna" and _OPTUNA_AVAILABLE:
            best_pipeline, best_val_mae = self._tune_with_optuna(
                pipeline, model_name, param_grid, X_train, y_train, model_cls, seed
            )
        elif param_grid and model_name not in ("linear_regression", "ridge", "lasso", "elasticnet"):
            # Prefix param names with "model__" for the Pipeline
            prefixed_grid = {f"model__{k}": v for k, v in param_grid.items()}

            cv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=prefixed_grid,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,  # Enable parallel cross-validation
                refit=True,
                error_score="raise",
            )
            grid_search.fit(X_train, y_train)

            best_pipeline = grid_search.best_estimator_
            best_val_mae = -grid_search.best_score_

            logger.info(
                "  %s GridSearchCV best params: %s",
                model_name,
                grid_search.best_params_,
            )
        else:
            # For linear models or models with no tuning configured
            # If we have a param grid for Ridge/Lasso etc., we could do GridSearch, 
            # but let's just do it directly if configured.
            if param_grid and model_name in ("ridge", "lasso", "elasticnet") and tuner == "gridsearch":
                prefixed_grid = {f"model__{k}": v for k, v in param_grid.items()}
                cv = TimeSeriesSplit(n_splits=3)
                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=prefixed_grid,
                    cv=cv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1, # Enable parallel cross-validation
                    refit=True,
                )
                grid_search.fit(X_train, y_train)
                best_pipeline = grid_search.best_estimator_
                best_val_mae = -grid_search.best_score_
                logger.info("  %s GridSearchCV best params: %s", model_name, grid_search.best_params_)
            else:
                # No tuning — just fit directly
                pipeline.fit(X_train, y_train)
                best_pipeline = pipeline

                # Estimate validation MAE with TimeSeriesSplit
                cv = TimeSeriesSplit(n_splits=3)
                val_scores: List[float] = []
                for train_cv_idx, val_cv_idx in cv.split(X_train):
                    scale = model_name in ("linear_regression", "ridge", "lasso", "elasticnet", "random_forest")
                    clone_pipe = self._clone_pipeline(model_cls, model_name, seed, scale=scale)
                    clone_pipe.fit(X_train.iloc[train_cv_idx], y_train[train_cv_idx])
                    y_val_pred = clone_pipe.predict(X_train.iloc[val_cv_idx])
                    val_scores.append(mean_absolute_error(y_train[val_cv_idx], y_val_pred))

                best_val_mae = float(np.mean(val_scores))

        return best_pipeline, best_val_mae

    def _tune_with_optuna(self, pipeline, model_name, param_grid, X_train, y_train, model_cls, seed):
        """Perform hyperparameter tuning using Optuna with TimeSeriesSplit."""
        import optuna
        
        def objective(trial):
            # Sample parameters based on the param_grid
            sampled_params = {}
            for k, v in param_grid.items():
                if isinstance(v, list):
                    if all(isinstance(i, int) for i in v):
                        sampled_params[k] = trial.suggest_categorical(k, v)
                    elif all(isinstance(i, float) for i in v):
                        sampled_params[k] = trial.suggest_categorical(k, v)
                    else:
                        sampled_params[k] = trial.suggest_categorical(k, v)

            # Build pipeline clone
            scale = model_name in ("linear_regression", "ridge", "lasso", "elasticnet", "random_forest")
            steps = []
            if scale:
                steps.append(("scaler", StandardScaler()))
            
            # Instantiate model with sampled params
            model_instance = self._instantiate_model(model_cls, model_name, seed, **sampled_params)
            steps.append(("model", model_instance))
            trial_pipeline = Pipeline(steps)
            
            cv = TimeSeriesSplit(n_splits=3)
            val_scores = []
            
            for train_cv_idx, val_cv_idx in cv.split(X_train):
                X_t = X_train.iloc[train_cv_idx]
                y_t = y_train[train_cv_idx]
                X_v = X_train.iloc[val_cv_idx]
                y_v = y_train[val_cv_idx]
                
                # Apply early stopping for gradient boosting models
                if model_name in ("xgboost", "lightgbm", "catboost"):
                    if scale:
                        X_t = trial_pipeline.named_steps["scaler"].fit_transform(X_t)
                        X_v = trial_pipeline.named_steps["scaler"].transform(X_v)
                    
                    if model_name == "xgboost":
                        model_instance.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
                    elif model_name == "lightgbm":
                        # LightGBM requires callbacks for early stopping in newer versions, 
                        # but we can just fit normally if early_stopping isn't directly passed.
                        # We'll just fit to avoid callback complexity unless strictly needed.
                        model_instance.fit(X_t, y_t) 
                    elif model_name == "catboost":
                        model_instance.fit(X_t, y_t, eval_set=(X_v, y_v), early_stopping_rounds=20, verbose=False)
                        
                    y_val_pred = model_instance.predict(X_v)
                else:
                    import warnings
                    from sklearn.exceptions import ConvergenceWarning
                    # For linear models, do not suppress ConvergenceWarning, let it be raised or logged
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        trial_pipeline.fit(X_t, y_t)
                        if len(w) > 0:
                            for warning in w:
                                if issubclass(warning.category, ConvergenceWarning):
                                    logger.error(f"ConvergenceWarning for {model_name} in Optuna CV: {warning.message}")
                    y_val_pred = trial_pipeline.predict(X_v)
                    
                val_scores.append(mean_absolute_error(y_v, y_val_pred))
                
            return np.mean(val_scores)
            
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        n_trials = getattr(self.config, 'optuna_trials', 20)
        study.optimize(objective, n_trials=n_trials, n_jobs=-1) # Run trials in parallel
        
        logger.info("  %s Optuna best params: %s", model_name, study.best_params)
        
        # Refit best pipeline on all data
        scale = model_name in ("linear_regression", "ridge", "lasso", "elasticnet", "random_forest")
        steps = []
        if scale:
            steps.append(("scaler", StandardScaler()))
        model_instance = self._instantiate_model(model_cls, model_name, seed, **study.best_params)
        steps.append(("model", model_instance))
        best_pipeline = Pipeline(steps)
        
        # Fit with early stopping if supported on refit? (No validation set here for full refit)
        best_pipeline.fit(X_train, y_train)
        
        return best_pipeline, study.best_value

    @staticmethod
    def _instantiate_model(model_cls: type, model_name: str, seed: int, **kwargs) -> Any:
        """Create a model instance with appropriate defaults and override params."""
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        
        if model_name in ("linear_regression", "ridge", "lasso", "elasticnet"):
            if model_name == "linear_regression":
                return model_cls()
            elif model_name in ("lasso", "elasticnet"):
                # Increase max_iter significantly to prevent non-convergence
                return model_cls(random_state=seed, max_iter=50000, tol=1e-4, **kwargs)
            else:
                return model_cls(random_state=seed, **kwargs)
        elif model_name == "random_forest":
            return model_cls(random_state=seed, n_jobs=-1, **kwargs)
        elif model_name == "xgboost":
            return model_cls(
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
                tree_method="hist",
                device="cuda",  # Enable GPU for XGBoost
                **kwargs
            )
        elif model_name == "lightgbm":
            return model_cls(
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
                device="gpu",   # Enable GPU for LightGBM
                **kwargs
            )
        elif model_name == "catboost":
            return model_cls(
                random_seed=seed,
                thread_count=-1,
                verbose=0,
                task_type="GPU", # Enable GPU for CatBoost
                **kwargs
            )
        else:
            return model_cls(random_state=seed, **kwargs)

    def _clone_pipeline(
        self,
        model_cls: type,
        model_name: str,
        seed: int,
        scale: bool = False,
    ) -> Pipeline:
        """Create a fresh (unfitted) pipeline clone."""
        steps: List[Tuple[str, Any]] = []
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", self._instantiate_model(model_cls, model_name, seed)))
        return Pipeline(steps)

    def _select_best_model(
        self,
        fold_results: Dict[str, Tuple[Pipeline, float]],
    ) -> Tuple[str, Pipeline, float]:
        """Select the model with the lowest validation MAE."""
        best_name = min(fold_results, key=lambda k: fold_results[k][1])
        best_pipeline, best_mae = fold_results[best_name]
        return best_name, best_pipeline, best_mae

    # ------------------------------------------------------------------ #
    #  Metrics
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0,
        }

    # ------------------------------------------------------------------ #
    #  Feature importances
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_feature_importances(
        pipeline: Pipeline,
        model_name: str,
        feature_cols: List[str],
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Extract feature importances from the fitted model in the pipeline.
        Computes Permutation Importance if native importance is unavailable.
        """
        model = pipeline.named_steps["model"]

        raw = None
        if hasattr(model, "feature_importances_"):
            raw = model.feature_importances_
        elif hasattr(model, "coef_"):
            raw = np.abs(model.coef_)
        else:
            try:
                from sklearn.inspection import permutation_importance
                result = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
                raw = result.importances_mean
            except Exception as e:
                logger.warning(f"Could not compute feature importance for {model_name}: {e}")
                return {col: 0.0 for col in feature_cols}

        if raw is None:
            return {col: 0.0 for col in feature_cols}

        # Normalise
        total = raw.sum()
        if total > 0:
            normed = raw / total
        else:
            normed = raw

        return {col: float(normed[i]) for i, col in enumerate(feature_cols)}

    # ------------------------------------------------------------------ #
    #  SHAP
    # ------------------------------------------------------------------ #

    def _compute_shap_values(
        self,
        pipeline: Pipeline,
        X_test: np.ndarray,
        feature_cols: List[str],
        fold_idx: int,
        experiment_id: str,
    ) -> None:
        """Compute and persist SHAP values for the XGBoost model."""
        try:
            import shap

            model = pipeline.named_steps["model"]
            explainer = shap.TreeExplainer(model)

            # Limit to a sample for performance
            sample_size = min(500, X_test.shape[0])
            X_sample = X_test[:sample_size]

            shap_values = explainer.shap_values(X_sample)
            
            # Handle list output for some models (e.g., CatBoost or multiclass)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
                
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            shap_importance = {
                col: float(mean_abs_shap[i])
                for i, col in enumerate(feature_cols)
            }

            # Save to file
            shap_dir = self.config.get_experiment_dir(experiment_id) / "shap"
            shap_dir.mkdir(parents=True, exist_ok=True)
            shap_path = shap_dir / f"shap_fold_{fold_idx}.json"
            with open(shap_path, "w", encoding="utf-8") as f:
                json.dump(shap_importance, f, indent=2)

            logger.info("SHAP values saved to %s", shap_path)

        except ImportError:
            logger.warning("shap package not available — skipping SHAP computation")
        except Exception as exc:
            logger.error("SHAP computation failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def _save_predictions(self, all_predictions: List[pd.DataFrame]) -> None:
        """Write all fold predictions to the predictions table."""
        if not all_predictions:
            return

        preds = pd.concat(all_predictions, ignore_index=True)
        self.db.write_dataframe(preds, "predictions", if_exists="append")
        logger.info("Saved %d predictions to database", len(preds))

    def _save_fold_metrics(
        self,
        all_fold_metrics: List[Dict[str, Any]],
        experiment_id: str,
    ) -> None:
        """Write per-fold metrics to the walk_forward_results table."""
        rows = []
        for fm in all_fold_metrics:
            rows.append({
                "fold": fm["fold"],
                "best_model": fm["best_model"],
                "mae": fm["mae"],
                "rmse": fm["rmse"],
                "r2": fm["r2"],
                "train_size": fm["train_size"],
                "test_size": fm["test_size"],
                "train_start": fm["train_start"],
                "train_end": fm["train_end"],
                "test_start": fm["test_start"],
                "test_end": fm["test_end"],
                "metrics": json.dumps({
                    "mae": fm["mae"],
                    "rmse": fm["rmse"],
                    "r2": fm["r2"],
                    "best_model": fm["best_model"],
                    "train_size": fm["train_size"],
                    "test_size": fm["test_size"],
                }),
                "experiment_id": experiment_id,
            })

        df = pd.DataFrame(rows)
        self.db.write_dataframe(df, "walk_forward_results", if_exists="append")
        logger.info("Saved %d fold metrics to database", len(rows))

    def _save_experiment_metadata(self, experiment_id: str, all_fold_metrics: List[Dict[str, Any]], feature_cols: List[str]) -> None:
        """Save reproducibility metadata for the experiment."""
        import sys
        try:
            import pkg_resources
            packages = {p.project_name: p.version for p in pkg_resources.working_set}
        except Exception:
            packages = {"python": sys.version}
            
        metadata = {
            "experiment_id": experiment_id,
            "git_commit": _get_git_commit(),
            "random_seed": self.config.seed,
            "feature_version": self.config.feature_version,
            "models_tested": self.config.model_list,
            "optuna_trials": getattr(self.config, 'optuna_trials', 20),
            "config_snapshot": self.config._config_dict if hasattr(self.config, '_config_dict') else str(self.config.__dict__),
            "feature_count": len(feature_cols),
            "features_used": feature_cols,
            "packages": packages,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        out_dir = self.config.get_experiment_dir(experiment_id) / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / "experiment_metadata.json"
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info("Experiment metadata saved to %s", meta_path)

    def _serialize_models(
        self,
        best_models: List[Tuple[str, Any]],
        experiment_id: str,
    ) -> None:
        """Serialize the best model from each fold to disk."""
        out_dir = self.config.get_experiment_dir(experiment_id) / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, (name, pipeline) in enumerate(best_models):
            path = out_dir / f"fold_{idx}_{name}.joblib"
            joblib.dump(pipeline, path)
            logger.info("Saved model %s to %s", name, path)

        # Also save the last fold's model as 'latest'
        if best_models:
            last_name, last_pipe = best_models[-1]
            latest_path = out_dir / f"latest_{last_name}.joblib"
            joblib.dump(last_pipe, latest_path)
            logger.info("Latest model saved: %s", latest_path)

    # ------------------------------------------------------------------ #
    #  Summary
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_summary(
        all_fold_metrics: List[Dict[str, Any]],
        best_models: List[Tuple[str, Any]],
    ) -> Dict[str, Any]:
        """Compile an aggregate summary across all folds."""
        if not all_fold_metrics:
            return {"fold_count": 0, "aggregate_mae": float("inf")}

        maes = [fm["mae"] for fm in all_fold_metrics]
        rmses = [fm["rmse"] for fm in all_fold_metrics]
        r2s = [fm["r2"] for fm in all_fold_metrics]

        model_counts: Dict[str, int] = {}
        for fm in all_fold_metrics:
            name = fm["best_model"]
            model_counts[name] = model_counts.get(name, 0) + 1

        return {
            "fold_count": len(all_fold_metrics),
            "aggregate_mae": float(np.mean(maes)),
            "aggregate_rmse": float(np.mean(rmses)),
            "aggregate_r2": float(np.mean(r2s)),
            "mae_std": float(np.std(maes)),
            "model_selection_counts": model_counts,
            "best_model_overall": max(model_counts, key=model_counts.get),
        }
