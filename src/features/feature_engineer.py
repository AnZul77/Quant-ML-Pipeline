"""
Feature engineering and feature store writer for the Quantitative ML Pipeline.

Computes five categories of features from raw trades and macro events:
  1. Time features -- hour, day, week, month, quarter, market session timing
  2. Rolling trading features -- shifted by 1 to prevent look-ahead bias
  3. Macro proximity features -- temporal distance and surprise from macro events
  4. Enhanced macro features -- density, rolling counts, impact flags, encodings
  5. Strategy features -- holding time, threshold, risk parameters

All features are written to the ``feature_store`` database table.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.database.base import DatabaseClient
from src.utils.config import PipelineConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PIPELINE_VERSION = "2.0.0"


class FeatureEngineer:
    """Build engineered features from raw trades and macro events.

    Args:
        config: Pipeline configuration object.
        db: Database client implementing ``DatabaseClient``.
    """

    def __init__(self, config: PipelineConfig, db: DatabaseClient) -> None:
        self.config = config
        self.db = db
        self._session_ranges = self._parse_market_sessions()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def build_features(self) -> pd.DataFrame:
        """Build all features and persist to the ``feature_store`` table.

        Returns:
            The complete feature DataFrame that was written.

        Raises:
            ValueError: If trades or macro_events tables are empty / missing.
        """
        logger.info("Starting feature engineering (version=%s)", self.config.feature_version)

        trades = self._load_trades()
        macro_series = self._load_macro_series()

        logger.info(
            "Loaded %d trades and %d macro events from database",
            len(trades),
            len(macro_series),
        )

        # --- Compute feature groups ---
        features = trades[["id", "timestamp", "account", "permutation_id"]].copy()
        features.rename(columns={"id": "trade_id"}, inplace=True)

        features = self._add_time_features(features, trades)
        features = self._add_rolling_features(features, trades)

        cols_before = len(features.columns)
        features = self._add_macro_features(features, trades, macro_series)
        features = self._add_enhanced_macro_features(features, trades, macro_series)
        macro_features_count = len(features.columns) - cols_before

        logger.info(
            "Feature Engineering Audit:\n- Number of trades: %d\n- Number of macro events: %d\n- Number of engineered macro features: %d",
            len(trades),
            len(macro_series),
            macro_features_count,
        )

        features = self._add_strategy_features(features, trades)

        # --- Leakage audit ---
        self._audit_leakage(features, trades, macro_series)

        # --- Metadata columns ---
        features["feature_version"] = self.config.feature_version
        features["pipeline_version"] = _PIPELINE_VERSION
        features["feature_timestamp"] = datetime.now(timezone.utc).isoformat()

        # --- Target ---
        features["pnl"] = trades["pnl"].values

        # --- Select final columns in schema order ---
        output_columns = self._schema_columns()
        features = features[output_columns]

        # --- Write to database ---
        self.db.execute("DELETE FROM feature_store")
        self.db.write_dataframe(features, "feature_store", if_exists="append")
        logger.info(
            "Feature store written: %d rows, %d columns",
            len(features),
            len(features.columns),
        )
        
        # --- Generate Feature Lineage Report ---
        self._generate_feature_lineage(features.columns.tolist())

        return features

    def _generate_feature_lineage(self, feature_cols: List[str]) -> None:
        """Generate a feature lineage report for tracking origins."""
        import os
        
        lineage = []
        for col in feature_cols:
            source = "trades"
            transform = "None (Raw)"
            if col.startswith("rolling_"):
                source = "trades"
                transform = "Rolling window aggregation (shifted by 1)"
            elif col.startswith("macro_") or "release" in col:
                source = "macro_series"
                transform = "Temporal proximity or momentum from official FRED data"
            elif col.startswith("lag_"):
                source = "trades (target)"
                transform = "Temporal lag"
            elif col in ["hour_of_day", "day_of_week", "week_of_month", "month", "quarter", "market_session"]:
                source = "trades (timestamp)"
                transform = "Time encoding"
            elif col in ["trade_id", "timestamp", "pnl", "feature_version", "pipeline_version", "feature_timestamp", "permutation_id"]:
                continue
                
            lineage.append({
                "feature": col,
                "source": source,
                "transformation": transform,
                "dependencies": "trade_timestamp, execution_price" if source == "trades" else "macro_series.release_date",
                "version": self.config.feature_version
            })
            
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        report_path = reports_dir / "feature_lineage.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Feature Lineage Report\n\n")
            f.write(f"**Feature Version**: {self.config.feature_version}\n\n")
            f.write("| Feature | Source | Transformation | Dependencies | Version |\n")
            f.write("|---|---|---|---|---|\n")
            for item in lineage:
                f.write(f"| {item['feature']} | {item['source']} | {item['transformation']} | {item['dependencies']} | {item['version']} |\n")
                
        logger.info("Feature lineage report generated at %s", report_path)

    # ------------------------------------------------------------------ #
    #  Data loading
    # ------------------------------------------------------------------ #

    def _load_trades(self) -> pd.DataFrame:
        """Load trades from the database and parse timestamps."""
        trades = self.db.read_table("trades")
        if trades.empty:
            raise ValueError("No trades found in the database. Run data ingestion first.")

        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
        trades.sort_values("timestamp", inplace=True)
        trades.reset_index(drop=True, inplace=True)

        logger.info("Trades date range: %s -> %s", trades["timestamp"].min(), trades["timestamp"].max())
        return trades

    def _load_macro_series(self) -> pd.DataFrame:
        """Load macro series from the database and parse timestamps."""
        macro = self.db.read_table("macro_series")
        if macro.empty:
            logger.warning("No macro series found -- macro features will be NaN")
            return pd.DataFrame(
                columns=["id", "series_id", "date", "value",
                         "release_date", "frequency", "source"]
            )

        macro["release_date"] = pd.to_datetime(macro["release_date"], utc=True)
        macro.sort_values("release_date", inplace=True)
        
        # Precompute per-series percentage change so we can combine them 
        # into a normalized chronological stream
        macro["pct_change"] = macro.groupby("series_id")["value"].pct_change()
        # Fallback to 0 if it's the first release
        macro["pct_change"] = macro["pct_change"].fillna(0.0)
        
        macro.reset_index(drop=True, inplace=True)
        return macro

    # ------------------------------------------------------------------ #
    #  Time features
    # ------------------------------------------------------------------ #

    def _add_time_features(self, features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        """Compute calendar and market-session features."""
        ts = trades["timestamp"]

        features["hour_of_day"] = ts.dt.hour
        features["day_of_week"] = ts.dt.dayofweek  # 0=Monday, 6=Sunday
        features["week_of_month"] = ts.apply(self._week_of_month)
        features["month"] = ts.dt.month
        features["quarter"] = ts.dt.quarter

        # Market session classification
        session_info = ts.apply(self._classify_session)
        features["market_session"] = session_info.apply(lambda x: x[0])
        features["minutes_after_open"] = session_info.apply(lambda x: x[1])
        features["minutes_before_close"] = session_info.apply(lambda x: x[2])

        logger.info("Time features computed")
        return features

    @staticmethod
    def _week_of_month(dt: pd.Timestamp) -> int:
        """Return week of month (1-5) for a timestamp."""
        first_day = dt.replace(day=1)
        adjusted_dom = dt.day + first_day.weekday()
        return int(np.ceil(adjusted_dom / 7.0))

    def _parse_market_sessions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Parse market session open/close times from config into minute-of-day tuples."""
        sessions: Dict[str, Tuple[int, int, int, int]] = {}
        ms = self.config.market_sessions

        for session_name in ("asia", "europe", "us"):
            open_str = ms.get(f"{session_name}_open", "00:00")
            close_str = ms.get(f"{session_name}_close", "00:00")
            oh, om = (int(p) for p in open_str.split(":"))
            ch, cm = (int(p) for p in close_str.split(":"))
            sessions[session_name] = (oh, om, ch, cm)

        return sessions

    def _classify_session(self, ts: pd.Timestamp) -> Tuple[str, float, float]:
        """Classify a UTC timestamp into a market session."""
        ts_minutes = ts.hour * 60 + ts.minute

        for session_name in ("us", "europe", "asia"):
            oh, om, ch, cm = self._session_ranges[session_name]
            open_minutes = oh * 60 + om
            close_minutes = ch * 60 + cm

            if open_minutes <= ts_minutes < close_minutes:
                after_open = float(ts_minutes - open_minutes)
                before_close = float(close_minutes - ts_minutes)
                return session_name, after_open, before_close

        return "off_hours", np.nan, np.nan

    # ------------------------------------------------------------------ #
    #  Rolling trading features
    # ------------------------------------------------------------------ #

    def _add_rolling_features(self, features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling statistics per (account, permutation_id), shifted by 1."""
        short = self.config.rolling_short
        long = self.config.rolling_long

        df = trades[["account", "permutation_id", "timestamp", "pnl", "quantity"]].copy()
        df.sort_values("timestamp", inplace=True)

        group_cols = ["account", "permutation_id"]

        df["pnl_shifted"] = df.groupby(group_cols)["pnl"].shift(1)
        df["quantity_shifted"] = df.groupby(group_cols)["quantity"].shift(1)
        df["win_flag_shifted"] = (df["pnl_shifted"] > 0).astype(float)
        df["trade_flag_shifted"] = df["pnl_shifted"].notna().astype(float)

        features["rolling_pnl_5"] = (
            df.groupby(group_cols)["pnl_shifted"]
            .transform(lambda s: s.rolling(window=short, min_periods=1).mean())
        )

        features["rolling_pnl_20"] = (
            df.groupby(group_cols)["pnl_shifted"]
            .transform(lambda s: s.rolling(window=long, min_periods=1).mean())
        )

        features["rolling_win_rate"] = (
            df.groupby(group_cols)["win_flag_shifted"]
            .transform(lambda s: s.rolling(window=long, min_periods=1).mean())
        )

        features["rolling_avg_quantity"] = (
            df.groupby(group_cols)["quantity_shifted"]
            .transform(lambda s: s.rolling(window=long, min_periods=1).mean())
        )

        features["rolling_trade_frequency"] = (
            df.groupby(group_cols)["trade_flag_shifted"]
            .transform(lambda s: s.rolling(window=long, min_periods=1).sum())
        )

        features["rolling_volatility"] = (
            df.groupby(group_cols)["pnl_shifted"]
            .transform(lambda s: s.rolling(window=long, min_periods=1).std())
        )

        features["rolling_drawdown"] = (
            df.groupby(group_cols)["pnl_shifted"]
            .transform(lambda s: self._rolling_max_drawdown(s, window=long))
        )

        logger.info(
            "Rolling features computed (short=%d, long=%d) with shift(1) leak prevention",
            short, long,
        )
        return features

    @staticmethod
    def _rolling_max_drawdown(series: pd.Series, window: int) -> pd.Series:
        """Compute rolling max drawdown of cumulative P&L over a window."""
        result = pd.Series(np.nan, index=series.index, dtype=float)
        values = series.values
        n = len(values)

        for i in range(n):
            start = max(0, i - window + 1)
            window_vals = values[start: i + 1]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) == 0:
                continue
            cum_pnl = np.nancumsum(valid)
            running_max = np.maximum.accumulate(cum_pnl)
            drawdowns = running_max - cum_pnl
            result.iloc[i] = float(np.max(drawdowns))

        return result

    # ------------------------------------------------------------------ #
    #  Macro Features (Time-Series)
    # ------------------------------------------------------------------ #

    def _add_macro_features(
        self,
        features: pd.DataFrame,
        trades: pd.DataFrame,
        macro: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute advanced time-series macro features using macro_series."""

        # Define default NaNs for the new features
        new_cols = [
            "days_since_release", "days_until_next_release", "release_frequency",
            "rolling_mean", "rolling_std", "rolling_min", "rolling_max",
            "rolling_median", "rolling_percent_change", "rolling_zscore",
            "macro_rolling_volatility", "macro_rolling_variance", "macro_rolling_percentile",
            "macro_ewma", "rolling_skewness", "rolling_kurtosis",
            "expanding_mean", "expanding_std",
            "macro_momentum", "macro_acceleration", "macro_regime",
            "yield_curve_spread", "inflation_momentum",
            "lag_1", "lag_3", "lag_6", "lag_12"
        ]
        
        for col in new_cols:
            features[col] = np.nan

        if macro.empty:
            logger.warning("Macro series empty -- filled with NaN")
            return features

        macro_ts = macro["release_date"].values
        macro_pct = macro["pct_change"].values
        trade_ts = trades["timestamp"].values
        
        # Precompute expanding and rolling stats on the chronological macro_pct array
        # We do this array-wise for the macro stream, then for each trade we just 
        # pick the value corresponding to the most recent macro event.
        s = pd.Series(macro_pct)
        
        # Rolling stats (window=12, e.g., representing roughly a year if monthly, or just 12 events)
        w = 12
        r = s.rolling(window=w, min_periods=1)
        e = s.expanding(min_periods=1)
        
        macro_r_mean = r.mean().values
        macro_r_std = r.std().values
        macro_r_var = r.var().values
        macro_r_min = r.min().values
        macro_r_max = r.max().values
        macro_r_median = r.median().values
        macro_r_skew = r.skew().values
        macro_r_kurt = r.kurt().values
        macro_r_percentile = s.rolling(window=w, min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=True
        ).values
        
        macro_ewma = s.ewm(span=w, min_periods=1).mean().values
        
        macro_e_mean = e.mean().values
        macro_e_std = e.std().values
        
        # Lags
        macro_lag_1 = s.shift(1).values
        macro_lag_3 = s.shift(3).values
        macro_lag_6 = s.shift(6).values
        macro_lag_12 = s.shift(12).values
        
        # We iterate over trades and find the nearest past macro event
        for i, t_ts in enumerate(trade_ts):
            diffs = (t_ts - macro_ts) / np.timedelta64(1, "D")
            past_mask = diffs > 0
            future_mask = diffs <= 0

            if past_mask.any():
                past_diffs = diffs[past_mask]
                closest_past_idx = np.argmin(past_diffs)
                past_indices = np.where(past_mask)[0]
                idx = past_indices[closest_past_idx]
                
                features.at[i, "days_since_release"] = float(past_diffs[closest_past_idx])
                
                features.at[i, "rolling_mean"] = macro_r_mean[idx]
                features.at[i, "rolling_std"] = macro_r_std[idx]
                features.at[i, "rolling_min"] = macro_r_min[idx]
                features.at[i, "rolling_max"] = macro_r_max[idx]
                features.at[i, "rolling_median"] = macro_r_median[idx]
                features.at[i, "rolling_percent_change"] = macro_pct[idx]  # Latest pct change
                
                std_val = macro_r_std[idx]
                if pd.notna(std_val) and std_val > 0:
                    features.at[i, "rolling_zscore"] = (macro_pct[idx] - macro_r_mean[idx]) / std_val
                else:
                    features.at[i, "rolling_zscore"] = 0.0
                    
                features.at[i, "macro_rolling_volatility"] = std_val
                features.at[i, "macro_rolling_variance"] = macro_r_var[idx]
                features.at[i, "macro_rolling_percentile"] = macro_r_percentile[idx]
                features.at[i, "macro_ewma"] = macro_ewma[idx]
                features.at[i, "rolling_skewness"] = macro_r_skew[idx]
                features.at[i, "rolling_kurtosis"] = macro_r_kurt[idx]
                
                features.at[i, "expanding_mean"] = macro_e_mean[idx]
                features.at[i, "expanding_std"] = macro_e_std[idx]
                
                # Momentum and acceleration (simple diffs)
                l1 = macro_lag_1[idx]
                if pd.notna(l1):
                    features.at[i, "macro_momentum"] = macro_pct[idx] - l1
                    l2 = s.shift(2).values[idx]
                    if pd.notna(l2):
                        features.at[i, "macro_acceleration"] = (macro_pct[idx] - l1) - (l1 - l2)
                
                # Regime: 1 if > expanding mean, -1 if < expanding mean
                if pd.notna(macro_pct[idx]) and pd.notna(macro_e_mean[idx]):
                    features.at[i, "macro_regime"] = 1.0 if macro_pct[idx] > macro_e_mean[idx] else -1.0
                
                # Interactions (Yield Curve Spread & Inflation Momentum)
                # We need to find the latest value for specific series before this trade
                # We do this efficiently by filtering the past_mask for specific series_ids
                past_macro = macro.iloc[past_indices]
                
                dgs10_idx = past_macro[past_macro["series_id"] == "DGS10"].last_valid_index()
                dgs2_idx = past_macro[past_macro["series_id"] == "DGS2"].last_valid_index()
                if dgs10_idx is not None and dgs2_idx is not None:
                    features.at[i, "yield_curve_spread"] = macro.at[dgs10_idx, "value"] - macro.at[dgs2_idx, "value"]
                    
                cpi_idx = past_macro[past_macro["series_id"] == "CPIAUCSL"].last_valid_index()
                if cpi_idx is not None:
                    # inflation momentum can be the pct_change of CPI
                    features.at[i, "inflation_momentum"] = macro.at[cpi_idx, "pct_change"]
                
                features.at[i, "lag_1"] = l1
                features.at[i, "lag_3"] = macro_lag_3[idx]
                features.at[i, "lag_6"] = macro_lag_6[idx]
                features.at[i, "lag_12"] = macro_lag_12[idx]
                
                # Frequency proxy (days between this event and the previous one)
                if idx > 0:
                    prev_event_ts = macro_ts[idx - 1]
                    features.at[i, "release_frequency"] = float((macro_ts[idx] - prev_event_ts) / np.timedelta64(1, "D"))

            if future_mask.any():
                future_diffs = np.abs(diffs[future_mask])
                closest_future_idx = np.argmin(future_diffs)
                features.at[i, "days_until_next_release"] = float(future_diffs[closest_future_idx])

        logger.info("Advanced macro time-series features computed for %d trades", len(trades))
        return features

    def _add_enhanced_macro_features(self, features: pd.DataFrame, trades: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
        # We folded enhanced features into _add_macro_features.
        # This is a no-op to maintain the pipeline interface.
        return features

    # ------------------------------------------------------------------ #
    #  Leakage audit
    # ------------------------------------------------------------------ #

    def _audit_leakage(
        self,
        features: pd.DataFrame,
        trades: pd.DataFrame,
        macro: pd.DataFrame,
    ) -> None:
        """Verify that no macro feature uses future information."""
        if macro.empty or features.empty:
            return

        violations = (features["days_since_release"] < 0).sum()
        if violations > 0:
            raise ValueError(
                f"LEAKAGE DETECTED: {violations} trades have negative "
                f"days_since_release, indicating future macro data was used."
            )

        logger.info("Leakage audit passed: 0 violations across %d trades", len(trades))

    # ------------------------------------------------------------------ #
    #  Strategy features
    # ------------------------------------------------------------------ #

    def _add_strategy_features(self, features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        """Add strategy-specific features from the trades table."""
        features["holding_time"] = trades["holding_time"].values
        features["threshold"] = trades["threshold"].values
        features["risk_parameters"] = trades.apply(
            lambda row: json.dumps(
                {"holding_time": row["holding_time"], "threshold": row["threshold"]},
                default=str,
            ),
            axis=1,
        ).values
        features["permutation_id"] = trades["permutation_id"].values

        logger.info("Strategy features added")
        return features

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _schema_columns() -> List[str]:
        """Return the ordered list of columns matching the feature_store schema."""
        return [
            "trade_id",
            "feature_version",
            "pipeline_version",
            "feature_timestamp",
            # Time
            "hour_of_day",
            "day_of_week",
            "week_of_month",
            "month",
            "quarter",
            "market_session",
            "minutes_after_open",
            "minutes_before_close",
            # Rolling
            "rolling_pnl_5",
            "rolling_pnl_20",
            "rolling_win_rate",
            "rolling_avg_quantity",
            "rolling_trade_frequency",
            "rolling_volatility",
            "rolling_drawdown",
            # Macro Features
            "days_since_release",
            "days_until_next_release",
            "release_frequency",
            "rolling_mean",
            "rolling_std",
            "rolling_min",
            "rolling_max",
            "rolling_median",
            "rolling_percent_change",
            "rolling_zscore",
            "macro_rolling_volatility",
            "macro_rolling_variance",
            "macro_rolling_percentile",
            "macro_ewma",
            "rolling_skewness",
            "rolling_kurtosis",
            "expanding_mean",
            "expanding_std",
            "macro_momentum",
            "macro_acceleration",
            "macro_regime",
            "yield_curve_spread",
            "inflation_momentum",
            "lag_1",
            "lag_3",
            "lag_6",
            "lag_12",
            # Strategy
            "holding_time",
            "threshold",
            "risk_parameters",
            "permutation_id",
            # Target
            "pnl",
        ]
