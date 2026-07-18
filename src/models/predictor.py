"""
Permutation predictor and recommendation matrix generator.

Loads trained models and feature-store data to predict expected P&L
for every (day_of_week, hour_of_day, permutation_id) combination,
then selects the best permutation per time slot and builds a 7×24
recommendation matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.database.base import DatabaseClient
from src.utils.config import PipelineConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns excluded from model feature vectors (must match trainer.py)
_META_COLUMNS = frozenset({
    "trade_id", "feature_version", "pipeline_version", "feature_timestamp",
    "market_session", "risk_parameters", "permutation_id", "event_importance",
    "pnl",
    "account",
    "trade_timestamp",
})

_CATEGORICAL_COLUMNS = frozenset({
    "market_session", "event_importance",
})


class PermutationPredictor:
    """Predict expected P&L per permutation and generate recommendation matrices.

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

    def predict_best_permutations(self, experiment_id: str) -> pd.DataFrame:
        """Predict expected P&L for all permutations at every (day, hour) slot.

        For each (day_of_week, hour_of_day) combination the method:
          1. Constructs representative feature vectors per permutation using
             historical averages from the feature store.
          2. Predicts expected P&L with the best trained model.
          3. Selects the permutation with the highest predicted P&L.

        Args:
            experiment_id: Experiment whose trained model to use.

        Returns:
            DataFrame with columns:
                day_of_week, hour_of_day, best_permutation, predicted_pnl
        """
        logger.info("Predicting best permutations for experiment=%s", experiment_id)

        model = self._load_best_model(experiment_id)
        data = self._load_feature_data()
        feature_cols = self._get_feature_columns(data)
        permutation_ids = data["permutation_id"].dropna().unique().tolist()

        logger.info(
            "Loaded model and data: %d permutations, %d feature columns",
            len(permutation_ids),
            len(feature_cols),
        )

        # Pre-compute per-permutation feature profiles (historical averages)
        perm_profiles = self._compute_permutation_profiles(data, feature_cols, permutation_ids)

        results: List[Dict[str, Any]] = []

        for dow in range(7):  # 0=Monday … 6=Sunday
            for hour in range(24):
                best_perm: Optional[str] = None
                best_pnl: float = -np.inf

                for perm_id in permutation_ids:
                    feature_vector = self._construct_feature_vector(
                        perm_profiles, perm_id, dow, hour, feature_cols
                    )

                    pred_pnl = float(model.predict(feature_vector.reshape(1, -1))[0])

                    if pred_pnl > best_pnl:
                        best_pnl = pred_pnl
                        best_perm = perm_id

                results.append({
                    "day_of_week": dow,
                    "hour_of_day": hour,
                    "best_permutation": best_perm,
                    "predicted_pnl": best_pnl,
                })

        result_df = pd.DataFrame(results)
        logger.info(
            "Permutation predictions complete: %d time slots evaluated",
            len(result_df),
        )

        return result_df

    def generate_recommendation_matrix(self, experiment_id: str) -> pd.DataFrame:
        """Generate a 7×24 recommendation matrix of best permutation IDs.

        Rows are days of the week (0=Monday … 6=Sunday), columns are hours
        (0–23). Cell values are the permutation IDs predicted to yield the
        highest P&L for that time slot.

        IMPORTANT: Values are derived exclusively from *predicted* P&L —
        never from actual future P&L.

        Args:
            experiment_id: Experiment whose trained model to use.

        Returns:
            DataFrame of shape (7, 24) with permutation IDs as values.
        """
        logger.info("Generating recommendation matrix for experiment=%s", experiment_id)

        predictions = self.predict_best_permutations(experiment_id)

        # Pivot to 7 rows (days) × 24 columns (hours)
        matrix = predictions.pivot(
            index="day_of_week",
            columns="hour_of_day",
            values="best_permutation",
        )

        # Ensure full 7×24 grid even if some slots lack data
        matrix = matrix.reindex(index=range(7), columns=range(24))

        # Label index for readability
        day_labels = ["Monday", "Tuesday", "Wednesday", "Thursday",
                      "Friday", "Saturday", "Sunday"]
        matrix.index = pd.Index(day_labels, name="day_of_week")
        matrix.columns = pd.Index([f"{h:02d}:00" for h in range(24)], name="hour_utc")

        logger.info("Recommendation matrix generated: shape=%s", matrix.shape)
        return matrix

    # ------------------------------------------------------------------ #
    #  Model loading
    # ------------------------------------------------------------------ #

    def _load_best_model(self, experiment_id: str) -> Any:
        """Load the best (latest) model for an experiment.

        Attempts to load from disk first. Falls back to re-loading
        experiment metadata from the database.

        Returns:
            A fitted sklearn Pipeline.

        Raises:
            FileNotFoundError: If no serialised model is found.
        """
        model_dir = self.config.get_experiment_dir(experiment_id) / "models"

        # Look for the 'latest_*.joblib' file
        if model_dir.exists():
            latest_files = sorted(model_dir.glob("latest_*.joblib"))
            if latest_files:
                model_path = latest_files[0]
                logger.info("Loading model from %s", model_path)
                return joblib.load(model_path)

            # Fallback: load the last fold model
            all_models = sorted(model_dir.glob("fold_*.joblib"))
            if all_models:
                model_path = all_models[-1]
                logger.info("Loading fallback model from %s", model_path)
                return joblib.load(model_path)

        raise FileNotFoundError(
            f"No serialised model found for experiment '{experiment_id}' "
            f"in {model_dir}. Ensure training was run with save_models=True."
        )

    # ------------------------------------------------------------------ #
    #  Feature data & column helpers
    # ------------------------------------------------------------------ #

    def _load_feature_data(self) -> pd.DataFrame:
        """Load and prepare feature store data (mirrors trainer logic)."""
        data = self.db.read_table("feature_store")
        if data.empty:
            raise ValueError("Feature store is empty. Run feature engineering first.")

        data["feature_timestamp"] = pd.to_datetime(data["feature_timestamp"], utc=True)

        # Encode categoricals consistently with trainer
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

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(0)

        return data

    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Return sorted list of feature columns (must match trainer)."""
        feature_cols = [
            col for col in data.columns
            if col not in _META_COLUMNS
            and col not in _CATEGORICAL_COLUMNS
            and data[col].dtype in (np.float64, np.float32, np.int64, np.int32, int, float)
        ]
        for enc_col in ("market_session_encoded", "event_importance_encoded"):
            if enc_col in data.columns and enc_col not in feature_cols:
                feature_cols.append(enc_col)

        return sorted(feature_cols)

    # ------------------------------------------------------------------ #
    #  Permutation profiling
    # ------------------------------------------------------------------ #

    def _compute_permutation_profiles(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        permutation_ids: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """Compute per-permutation feature statistics grouped by (day, hour).

        For each permutation, computes the mean of every feature column at
        each (day_of_week, hour_of_day) combination. This becomes the
        "typical" feature vector for prediction.

        Returns:
            Dict mapping permutation_id -> DataFrame indexed by
            (day_of_week, hour_of_day) with feature columns as values.
        """
        profiles: Dict[str, pd.DataFrame] = {}

        # Global fallback profile (across all permutations)
        global_profile = (
            data.groupby(["day_of_week", "hour_of_day"])[feature_cols]
            .mean()
        )

        for perm_id in permutation_ids:
            perm_data = data[data["permutation_id"] == perm_id]

            if perm_data.empty:
                profiles[perm_id] = global_profile
                continue

            perm_profile = (
                perm_data.groupby(["day_of_week", "hour_of_day"])[feature_cols]
                .mean()
            )

            # Fill missing (day, hour) slots with global averages
            perm_profile = perm_profile.reindex(global_profile.index)
            perm_profile = perm_profile.fillna(global_profile)

            profiles[perm_id] = perm_profile

        logger.info("Computed feature profiles for %d permutations", len(profiles))
        return profiles

    def _construct_feature_vector(
        self,
        perm_profiles: Dict[str, pd.DataFrame],
        perm_id: str,
        day_of_week: int,
        hour_of_day: int,
        feature_cols: List[str],
    ) -> np.ndarray:
        """Build a single feature vector for a (permutation, day, hour) combo.

        Falls back to the permutation's overall mean if the specific
        (day, hour) slot has no data.
        """
        profile = perm_profiles.get(perm_id)

        if profile is not None and (day_of_week, hour_of_day) in profile.index:
            row = profile.loc[(day_of_week, hour_of_day)]
            vector = row[feature_cols].values.astype(np.float64)
        elif profile is not None and not profile.empty:
            # Fallback: overall mean for this permutation
            vector = profile[feature_cols].mean().values.astype(np.float64)
        else:
            # Ultimate fallback: zeros
            vector = np.zeros(len(feature_cols), dtype=np.float64)

        # Override the day/hour features to match the target slot
        if "day_of_week" in feature_cols:
            idx = feature_cols.index("day_of_week")
            vector[idx] = float(day_of_week)
        if "hour_of_day" in feature_cols:
            idx = feature_cols.index("hour_of_day")
            vector[idx] = float(hour_of_day)

        # Replace any remaining NaN with 0
        vector = np.nan_to_num(vector, nan=0.0)

        return vector
