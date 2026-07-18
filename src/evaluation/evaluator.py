import glob
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.database.base import DatabaseClient
from src.utils.config import PipelineConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Set a professional plotting style
plt.style.use("dark_background")


class PipelineEvaluator:
    """Evaluator for walk-forward validation results."""

    def __init__(self, config: PipelineConfig, db: DatabaseClient) -> None:
        self.config = config
        self.db = db

    def compute_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standard regression metrics."""
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)),
        }

    def compute_trading_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Compute trading performance metrics from a series of P&L returns."""
        if len(returns) < 2:
            raise ValueError("Insufficient data to compute trading metrics.")

        ann_factor = self.config.annualization_factor
        rf_rate = self.config.risk_free_rate

        mean_ret = returns.mean()
        std_ret = returns.std()
        
        # Sharpe Ratio
        if std_ret > 0:
            sharpe = ((mean_ret - rf_rate) / std_ret) * np.sqrt(ann_factor)
        else:
            sharpe = 0.0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        if not np.isnan(downside_std) and downside_std > 0:
            sortino = ((mean_ret - rf_rate) / downside_std) * np.sqrt(ann_factor)
        else:
            sortino = 0.0

        # Maximum Drawdown
        cum_returns = returns.cumsum()
        running_max = cum_returns.cummax()
        drawdown = cum_returns - running_max
        max_drawdown = float(drawdown.min())

        # Profit Factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        # Average Trade & Win Rate
        avg_trade = float(mean_ret)
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = float(len(wins) / len(returns)) if len(returns) > 0 else 0.0

        # Calmar Ratio
        annualized_return = mean_ret * ann_factor
        calmar = float(annualized_return / abs(max_drawdown)) if abs(max_drawdown) > 0 else 0.0

        # Expectancy
        loss_rate = 1.0 - win_rate
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        expectancy = float((win_rate * avg_win) - (loss_rate * avg_loss))

        # Recovery Factor
        total_profit = returns.sum()
        recovery_factor = float(total_profit / abs(max_drawdown)) if abs(max_drawdown) > 0 else 0.0

        # Omega Ratio (threshold = 0)
        sum_wins = wins.sum()
        sum_losses = abs(losses.sum())
        omega_ratio = float(sum_wins / sum_losses) if sum_losses > 0 else float("inf")

        # Ulcer Index
        if running_max.iloc[-1] > 0:
            # We need percentage drawdown
            pct_drawdown = drawdown / running_max
            pct_drawdown = pct_drawdown.replace([np.inf, -np.inf], 0).fillna(0)
            ulcer_index = float(np.sqrt(np.mean(pct_drawdown ** 2)))
        else:
            ulcer_index = 0.0

        return {
            "Sharpe Ratio": float(sharpe),
            "Sortino Ratio": float(sortino),
            "Maximum Drawdown": max_drawdown,
            "Profit Factor": profit_factor,
            "Average Trade": avg_trade,
            "Win Rate": win_rate,
            "Calmar Ratio": calmar,
            "Expectancy": expectancy,
            "Recovery Factor": recovery_factor,
            "Omega Ratio": omega_ratio,
            "Ulcer Index": ulcer_index,
        }

    def validate_database(self, experiment_id: str) -> None:
        """Validate database content prior to evaluation."""
        logger.info("Evaluating Database Content")
        
        tables = ["trades", "feature_store", "predictions", "walk_forward_results", "macro_events"]
        for table in tables:
            count = self.db.read_sql(f"SELECT COUNT(*) FROM {table}").iloc[0, 0]
            logger.info("Rows in %s: %d", table, count)
            if count == 0:
                raise ValueError(f"Table '{table}' is empty. Cannot proceed with evaluation.")
                
        # Validate join strictly using trade_id
        join_query = """
            SELECT COUNT(*) 
            FROM predictions p 
            JOIN trades t ON p.trade_id = t.id
            WHERE p.experiment_id = ?
        """
        join_count = self.db.read_sql(join_query, (experiment_id,)).iloc[0, 0]
        logger.info("Rows joined using trade_id: %d", join_count)
        
        if join_count == 0:
            raise ValueError("Join between predictions and trades yielded 0 rows on trade_id.")
            
        # Null checks
        null_trade_id = self.db.read_sql("SELECT COUNT(*) FROM predictions WHERE trade_id IS NULL AND experiment_id = ?", (experiment_id,)).iloc[0, 0]
        null_actual = self.db.read_sql("SELECT COUNT(*) FROM predictions WHERE actual_pnl IS NULL AND experiment_id = ?", (experiment_id,)).iloc[0, 0]
        null_predicted = self.db.read_sql("SELECT COUNT(*) FROM predictions WHERE predicted_pnl IS NULL AND experiment_id = ?", (experiment_id,)).iloc[0, 0]
        missing_trade = self.db.read_sql("SELECT COUNT(*) FROM predictions p LEFT JOIN trades t ON p.trade_id=t.id WHERE t.id IS NULL AND p.experiment_id = ?", (experiment_id,)).iloc[0, 0]

        if null_trade_id > 0:
            raise ValueError(f"Found {null_trade_id} predictions with NULL trade_id.")
        if missing_trade > 0:
            raise ValueError(f"Found {missing_trade} predictions with trade_id not existing in trades.")
        if null_actual > 0:
            raise ValueError(f"Found {null_actual} predictions with NULL actual_pnl.")
        if null_predicted > 0:
            raise ValueError(f"Found {null_predicted} predictions with NULL predicted_pnl.")

    def validate_predictions(self, experiment_id: str) -> None:
        """Assert prediction integrity before plotting."""
        sql = "SELECT trade_id, permutation_id, predicted_pnl, actual_pnl, walk_forward_fold FROM predictions WHERE experiment_id = ?"
        preds = self.db.read_sql(sql, (experiment_id,))
        
        assert len(preds) > 0, "predictions dataframe is empty"
        assert preds["predicted_pnl"].notna().all(), "predicted_pnl contains nulls"
        assert preds["actual_pnl"].notna().all(), "actual_pnl contains nulls"
        assert preds["permutation_id"].notna().all(), "permutation_id contains nulls"
        
        # Verify trade_id is unique within predictions for a fold
        duplicates = preds.groupby(["walk_forward_fold", "trade_id"]).size()
        assert (duplicates == 1).all(), "trade_id is not unique within predictions for a fold"
        
        logger.info("Prediction integrity verified.")

    def generate_baseline_report(self, experiment_id: str, output_dir: Path) -> None:
        """Compute regression metrics for simple baselines."""
        logger.info("Generating baseline regression report")
        
        sql = """
            SELECT p.trade_id, p.timestamp, p.actual_pnl
            FROM predictions p
            WHERE p.experiment_id = ?
            ORDER BY p.timestamp
        """
        preds = self.db.read_sql(sql, (experiment_id,))
        if preds.empty:
            logger.warning("No predictions found for baselines.")
            return
            
        y_true = preds["actual_pnl"].values
        
        # 1. Mean Predictor
        mean_pred = np.full_like(y_true, y_true.mean())
        
        # 2. Previous Trade
        # Shift actual_pnl by 1
        prev_pred = np.roll(y_true, 1)
        prev_pred[0] = y_true[0]
        
        # 3. Random Predictor (Normal distribution matching mean/std)
        np.random.seed(42)
        random_pred = np.random.normal(loc=y_true.mean(), scale=y_true.std(), size=len(y_true))
        
        # 4. Rolling Mean (window=10)
        rolling_pred = preds["actual_pnl"].rolling(window=10, min_periods=1).mean().shift(1).values
        rolling_pred[0] = y_true[0]
        
        # 5. Median Predictor
        median_pred = np.full_like(y_true, np.median(y_true))
        
        baselines = {
            "predict_mean": mean_pred,
            "predict_median": median_pred,
            "previous_trade": prev_pred,
            "random": random_pred,
            "rolling_mean": rolling_pred
        }
        
        results = {}
        for name, y_pred in baselines.items():
            results[name] = self.compute_regression_metrics(y_true, y_pred)
            
        out_path = output_dir / "reports" / "baseline_regression_metrics.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info("Baseline regression metrics saved to %s", out_path)

    def generate_target_audit(self, output_dir: Path) -> None:
        """Analyze the prediction target (PnL) to understand its statistical properties."""
        logger.info("Generating target audit report")
        
        trades = self.db.read_table("trades")
        if trades.empty:
            logger.warning("No trades available for target audit.")
            return
            
        pnl = trades["pnl"]
        
        # Calculate stats
        mean_pnl = pnl.mean()
        std_pnl = pnl.std()
        var_pnl = pnl.var()
        skew_pnl = pnl.skew()
        kurt_pnl = pnl.kurt()
        autocorr = pnl.autocorr(lag=1)
        snr = abs(mean_pnl) / std_pnl if std_pnl > 0 else 0
        
        # Baseline predictability (R2 of predicting mean)
        baseline_preds = np.full_like(pnl, mean_pnl)
        r2_mean = r2_score(pnl, baseline_preds) # Will be exactly 0.0 mathematically
        
        audit_results = {
            "mean": float(mean_pnl),
            "std": float(std_pnl),
            "variance": float(var_pnl),
            "skewness": float(skew_pnl),
            "kurtosis": float(kurt_pnl),
            "autocorrelation_lag1": float(autocorr),
            "signal_to_noise_ratio": float(snr),
            "baseline_r2": float(r2_mean)
        }
        
        # Determine recommendations based on SNR and skewness
        recommendations = []
        if snr < 0.05:
            recommendations.append("The Signal-to-Noise Ratio (SNR) is extremely low. Predicting raw PnL is highly susceptible to noise. Consider predicting directional movement (binary Classification) or normalized return (PnL / volatility).")
        if abs(skew_pnl) > 2.0:
            recommendations.append("The target is highly skewed. Consider log-returns or clipping extreme outliers.")
        if autocorr > 0.1:
            recommendations.append("The target exhibits positive autocorrelation. Trend-following features might be highly predictive.")
            
        audit_results["recommendations"] = recommendations
        
        with open(output_dir / "reports" / "target_audit.json", "w") as f:
            json.dump(audit_results, f, indent=4)
            
        # Plot distribution
        plt.figure(figsize=(10, 5), dpi=150)
        sns.histplot(pnl, bins=50, kde=True, color="cyan")
        plt.title(f"Target (PnL) Distribution (Skew: {skew_pnl:.2f}, Kurtosis: {kurt_pnl:.2f})")
        plt.xlabel("PnL")
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "target_distribution.png")
        plt.close()

    def generate_data_quality_report(self, output_dir: Path) -> None:
        """Generate a macro data quality report."""
        logger.info("Generating macro data quality report")
        
        macro = self.db.read_table("macro_series")
        if macro.empty:
            logger.warning("No macro data available for quality report.")
            return
            
        report = []
        # Pivot to compute correlation easily
        pivot_macro = macro.pivot(index="date", columns="series_id", values="value")
        corr_matrix = pivot_macro.corr()
        
        for series_id, group in macro.groupby("series_id"):
            group = group.sort_values("date")
            start_date = group["date"].min()
            end_date = group["date"].max()
            missing_pct = group["value"].isna().mean() * 100
            variance = group["value"].var()
            
            # Find highly correlated series
            if series_id in corr_matrix.columns:
                corrs = corr_matrix[series_id].drop(series_id).dropna()
                high_corrs = corrs[abs(corrs) > 0.8].to_dict()
            else:
                high_corrs = {}
                
            report.append({
                "series_id": series_id,
                "start_date": start_date,
                "end_date": end_date,
                "missing_pct": missing_pct,
                "variance": variance,
                "high_correlations": high_corrs
            })
            
        report_df = pd.DataFrame(report)
        report_df.to_csv(output_dir / "reports" / "macro_data_quality.csv", index=False)
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10), dpi=150)
        sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, vmin=-1, vmax=1)
        plt.title("Macro Series Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "macro_correlation.png")
        plt.close()

    def generate_feature_drift_report(self, output_dir: Path) -> None:
        """Calculate feature drift across walk-forward folds."""
        logger.info("Generating feature drift report")
        
        sql = """
            SELECT f.*, t.timestamp as trade_timestamp
            FROM feature_store f
            JOIN trades t ON f.trade_id = t.id
            ORDER BY t.timestamp
        """
        data = self.db.read_sql(sql)
        if data.empty:
            return
            
        data["trade_timestamp"] = pd.to_datetime(data["trade_timestamp"], utc=True)
        
        # Split into two halves chronologically to measure drift
        mid_point = data["trade_timestamp"].median()
        first_half = data[data["trade_timestamp"] < mid_point]
        second_half = data[data["trade_timestamp"] >= mid_point]
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        cols_to_ignore = {"trade_id", "permutation_id", "pnl"}
        feature_cols = [c for c in numeric_cols if c not in cols_to_ignore]
        
        drift_report = []
        for col in feature_cols:
            mean1, std1 = first_half[col].mean(), first_half[col].std()
            mean2, std2 = second_half[col].mean(), second_half[col].std()
            
            # Simple PSI (Population Stability Index) approximation using means/stds
            # (A true PSI requires binning, we provide a basic drift metric here)
            drift_score = abs(mean1 - mean2) / (std1 + 1e-9) if std1 > 0 else 0
            
            drift_report.append({
                "feature": col,
                "first_half_mean": mean1,
                "first_half_std": std1,
                "second_half_mean": mean2,
                "second_half_std": std2,
                "drift_score": drift_score
            })
            
        drift_df = pd.DataFrame(drift_report).sort_values("drift_score", ascending=False)
        drift_df.to_csv(output_dir / "reports" / "feature_drift.csv", index=False)
        
        # Plot top 10 drifting features
        top_drift = drift_df.head(10)
        plt.figure(figsize=(10, 6), dpi=150)
        sns.barplot(data=top_drift, x="drift_score", y="feature", palette="Reds_r")
        plt.title("Top 10 Features by Drift Score (1st Half vs 2nd Half)")
        plt.xlabel("Drift Score (|Mean1 - Mean2| / Std1)")
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "feature_drift.png")
        plt.close()

    def generate_model_stability_report(self, experiment_id: str, output_dir: Path) -> None:
        """Analyze model stability across walk-forward folds."""
        logger.info("Generating model stability report")
        
        sql = "SELECT fold, best_model, mae, r2 FROM walk_forward_results WHERE experiment_id = ?"
        results = self.db.read_sql(sql, (experiment_id,))
        if results.empty:
            logger.warning("No walk-forward results found for stability analysis.")
            return
            
        metrics = []
        for model_name, group in results.groupby("best_model"):
            mean_mae = group["mae"].mean()
            std_mae = group["mae"].std()
            cv_mae = std_mae / mean_mae if mean_mae > 0 else 0
            
            # 95% Confidence Interval for MAE
            n = len(group)
            if n > 1:
                se = std_mae / np.sqrt(n)
                ci_margin = 1.96 * se
                ci_lower = mean_mae - ci_margin
                ci_upper = mean_mae + ci_margin
            else:
                ci_lower = ci_upper = mean_mae
                
            metrics.append({
                "model": model_name,
                "folds_won": n,
                "mean_mae": mean_mae,
                "std_mae": std_mae,
                "cv_mae": cv_mae,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper
            })
            
        metrics_df = pd.DataFrame(metrics).sort_values("mean_mae")
        metrics_df.to_csv(output_dir / "reports" / "model_stability.csv", index=False)
        
        # Rank stability - how often did the overall best model win a fold?
        total_folds = results["fold"].nunique()
        overall_best = metrics_df.iloc[0]["model"] if not metrics_df.empty else "N/A"
        win_rate = metrics_df.iloc[0]["folds_won"] / total_folds if total_folds > 0 and not metrics_df.empty else 0
        
        with open(output_dir / "reports" / "model_stability.md", "w") as f:
            f.write("# Model Stability Report\n\n")
            f.write("## Overall Stability Metrics\n")
            f.write("| Model | Folds Won | Mean MAE | Std MAE | CV (Std/Mean) | 95% CI |\n")
            f.write("|---|---|---|---|---|---|\n")
            for _, row in metrics_df.iterrows():
                f.write(f"| {row['model']} | {row['folds_won']} | {row['mean_mae']:.6f} | {row['std_mae']:.6f} | {row['cv_mae']:.4f} | [{row['ci_95_lower']:.6f}, {row['ci_95_upper']:.6f}] |\n")
                
            f.write("\n## Observations\n")
            f.write(f"The best overall model was **{overall_best}**, winning {win_rate*100:.1f}% of the {total_folds} folds. ")
            f.write("A lower Coefficient of Variation (CV) indicates a more stable model across different time periods.\n")
            
            f.write("\n## Interpretation\n")
            f.write("High variance (CV > 0.2) in MAE across folds indicates the model is highly sensitive to the specific time period, suggesting regime dependency or overfitting. ")
            f.write("If the best model changes every fold (low rank stability), the signal is likely extremely weak or non-stationary.\n")
            
            f.write("\n## Recommendations\n")
            f.write("1. If CV is high, consider training on a longer rolling window or incorporating regime-detection features.\n")
            f.write("2. If the best model frequently changes, an ensemble (StackingRegressor) might perform more robustly than picking a single winner.\n")

    def generate_heatmap(self, experiment_id: str, output_dir: Path) -> None:
        """Generate recommendation and predicted P&L heatmaps."""
        logger.info("Generating heatmaps for experiment %s", experiment_id)
        
        sql = """
            SELECT 
                p.timestamp,
                p.permutation_id,
                p.predicted_pnl
            FROM predictions p
            WHERE p.experiment_id = ?
        """
        preds = self.db.read_sql(sql, (experiment_id,))
        if preds.empty:
            raise ValueError("No predictions found for heatmap generation.")

        preds["timestamp"] = pd.to_datetime(preds["timestamp"], format="mixed", utc=True,errors="raise",)
        preds["hour_of_day"] = preds["timestamp"].dt.hour
        preds["day_of_week"] = preds["timestamp"].dt.dayofweek

        # Find best permutation per (day, hour)
        grouped = preds.groupby(["day_of_week", "hour_of_day", "permutation_id"])["predicted_pnl"].mean().reset_index()
        
        # Best permutation map
        idx = grouped.groupby(["day_of_week", "hour_of_day"])["predicted_pnl"].idxmax()
        best_perms = grouped.loc[idx]
        
        if best_perms.empty:
            raise ValueError("No recommended permutations found.")

        # Pivot for heatmap
        perm_matrix = best_perms.pivot(index="day_of_week", columns="hour_of_day", values="permutation_id")
        pnl_matrix = best_perms.pivot(index="day_of_week", columns="hour_of_day", values="predicted_pnl")
        
        # We need numerical labels for categorical permutations to plot a heatmap
        unique_perms = sorted(best_perms["permutation_id"].unique())
        perm_to_num = {p: i for i, p in enumerate(unique_perms)}
        num_matrix = perm_matrix.replace(perm_to_num)
        
        # Ensure num_matrix is fully numeric and handle any missing mapping
        num_matrix = num_matrix.apply(pd.to_numeric, errors="coerce").fillna(-1)
        assert all(np.issubdtype(dtype, np.number) for dtype in num_matrix.dtypes), "num_matrix must contain only numeric values for heatmap generation"

        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        
        # Create compact permutation labels
        perm_label_map = {p: f"P{i+1}" for i, p in enumerate(unique_perms)}
        label_matrix = perm_matrix.replace(perm_label_map)

        # Build legend text
        legend_lines = [f"{v}={k}" for k, v in perm_label_map.items()]
        legend_text = "  ".join(legend_lines)

        # Plot Permutation Heatmap
        fig, ax = plt.subplots(figsize=(16, 8), dpi=150)
        sns.heatmap(num_matrix, cmap="viridis", annot=label_matrix, fmt="", cbar=False, ax=ax, annot_kws={"size": 8})
        ax.set_yticklabels(days[:len(num_matrix)], rotation=0)
        ax.set_title("Recommended Permutation by Day and Hour")
        fig.text(0.5, -0.02, legend_text, ha="center", fontsize=7, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="black", edgecolor="white", alpha=0.8))
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "heatmap_permutation.png", bbox_inches="tight")
        plt.close()

        # Plot PnL Heatmap
        plt.figure(figsize=(16, 8), dpi=150)
        ax = sns.heatmap(pnl_matrix, cmap="RdYlGn", annot=True, fmt=".2f", annot_kws={"size": 8})
        ax.set_yticklabels(days[:len(pnl_matrix)], rotation=0)
        plt.title("Expected P&L of Recommended Permutation")
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "heatmap_pnl.png")
        plt.close()

    # ------------------------------------------------------------------ #
    #  Verification
    # ------------------------------------------------------------------ #
    
    def _verify_pipeline_integrity(self, experiment_id: str) -> None:
        """Verify that all pipeline stages succeeded before generating a report."""
        macro_count = self.db.row_count("macro_events")
        if macro_count == 0:
            raise ValueError("Pipeline integrity failed: macro_events table is empty.")
            
        feature_count = self.db.row_count("feature_store")
        if feature_count == 0:
            raise ValueError("Pipeline integrity failed: feature_store table is empty.")
            
        preds = self.db.read_sql(
            "SELECT COUNT(*) as c FROM predictions WHERE experiment_id = ?",
            (experiment_id,)
        )
        pred_count = preds.iloc[0]["c"] if not preds.empty else 0
        if pred_count == 0:
            raise ValueError(f"Pipeline integrity failed: predictions table is empty for {experiment_id}.")
            
        logger.info(
            "Pipeline integrity verified: %d macro events, %d features, %d predictions.",
            macro_count, feature_count, pred_count
        )

    # ------------------------------------------------------------------ #
    #  Data Loading
    # ------------------------------------------------------------------ #
    
    def _load_predictions(self, experiment_id: str) -> pd.DataFrame:
        sql = """
            SELECT 
                p.timestamp,
                p.permutation_id,
                p.predicted_pnl
            FROM predictions p
            WHERE p.experiment_id = ?
        """
        return self.db.read_sql(sql, (experiment_id,))

    def generate_equity_curve(self, experiment_id: str, output_dir: Path) -> None:
        """Generate cumulative equity curve comparing model portfolio vs baseline using trade_id."""
        logger.info("Generating equity curve for experiment %s", experiment_id)
        
        # 1. Get predictions (includes trade_id)
        sql = """
            SELECT trade_id, timestamp, predicted_pnl
            FROM predictions
            WHERE experiment_id = ?
        """
        preds = self.db.read_sql(sql, (experiment_id,))
        if preds.empty:
            raise ValueError("No predictions found for equity curve.")
            
        preds["timestamp"] = pd.to_datetime(preds["timestamp"], format="mixed", utc=True, errors="raise")
        
        # Find best prediction (highest expected P&L) per timestamp slot    
        idx = preds.groupby("timestamp")["predicted_pnl"].idxmax()
        best_preds = preds.loc[idx, ["trade_id", "timestamp", "predicted_pnl"]]
        
        # 2. Retrieve realized P&L directly using trade_id
        sql_trades = """
            SELECT id AS trade_id, pnl AS realized_pnl, permutation_id
            FROM trades
        """
        trades = self.db.read_sql(sql_trades)
        
        # Merge best predictions with realized trades using trade_id
        model_portfolio = pd.merge(
            best_preds,
            trades,
            on="trade_id",
            how="inner"
        )
        
        if model_portfolio.empty:
            raise ValueError("Equity dataframe is empty after joining trades on trade_id.")
            
        model_portfolio = model_portfolio.sort_values("timestamp").set_index("timestamp")
        model_portfolio["cumulative_pnl"] = model_portfolio["realized_pnl"].cumsum()
        
        if (model_portfolio["cumulative_pnl"] == 0).all():
            logger.warning("Cumulative PnL is completely zero. Check if trades actually yielded zero.")
        
        # 3. Construct Baseline Portfolio (Default Permutation)
        baseline_perm = self.config.baseline_permutation
        
        # Fetch baseline trades directly
        sql_baseline = "SELECT timestamp, pnl FROM trades WHERE permutation_id = ?"
        baseline_trades = self.db.read_sql(sql_baseline, (baseline_perm,))
        baseline_trades["timestamp"] = pd.to_datetime(baseline_trades["timestamp"], format="mixed", utc=True, errors="raise")
        
        # Group by timestamp (in case multiple trades in same timestamp bin)
        baseline_agg = baseline_trades.groupby("timestamp")["pnl"].sum().reset_index()
        baseline_agg = baseline_agg.set_index("timestamp")
        
        # Reindex to match model portfolio
        baseline_portfolio = pd.DataFrame(index=model_portfolio.index)
        baseline_portfolio["pnl"] = baseline_agg["pnl"]
        baseline_portfolio["pnl"] = baseline_portfolio["pnl"].fillna(0)
        baseline_portfolio["cumulative_pnl"] = baseline_portfolio["pnl"].cumsum()

        # Save metrics for report
        model_metrics = self.compute_trading_metrics(model_portfolio["realized_pnl"])
        baseline_metrics = self.compute_trading_metrics(baseline_portfolio["pnl"])
        
        with open(output_dir / "reports" / "trading_metrics.json", "w") as f:
            json.dump({
                "model": model_metrics,
                "baseline": baseline_metrics
            }, f, indent=4)

        # Plot Equity Curves
        plt.figure(figsize=(12, 6), dpi=150)
        plt.plot(model_portfolio.index, model_portfolio["cumulative_pnl"], label="Model Portfolio", color="cyan", linewidth=2)
        plt.plot(baseline_portfolio.index, baseline_portfolio["cumulative_pnl"], label=f"Baseline ({baseline_perm})", color="gray", linestyle="--")
        plt.title("Out-of-Sample Cumulative Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Cumulative P&L")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "equity_curve.png")
        plt.close()

    def generate_error_analysis(self, experiment_id: str, output_dir: Path) -> None:
        """Generate error analysis plots and data."""
        logger.info("Generating error analysis for experiment %s", experiment_id)
        
        # Ensure we strictly pull actual_pnl from trades using trade_id
        sql = """
            SELECT p.trade_id, p.predicted_pnl, t.pnl AS actual_pnl, p.permutation_id
            FROM predictions p
            JOIN trades t ON p.trade_id = t.id
            WHERE p.experiment_id = ?
        """
        preds = self.db.read_sql(sql, (experiment_id,))
        if preds.empty:
            raise ValueError("No predictions found for error analysis after trade_id join.")
            
        preds["residual"] = preds["predicted_pnl"] - preds["actual_pnl"]
        
        # 1. Residual Histogram
        plt.figure(figsize=(10, 5), dpi=150)
        sns.histplot(preds["residual"], bins=50, kde=True, color="purple")
        plt.title("Prediction Residuals Distribution (Predicted - Actual)")
        plt.xlabel("Residual P&L")
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "residual_histogram.png")
        plt.close()
        
        # 2. Residual vs Prediction
        plt.figure(figsize=(10, 5), dpi=150)
        sns.scatterplot(x="predicted_pnl", y="residual", data=preds, alpha=0.5, color="orange")
        plt.axhline(0, color="white", linestyle="--")
        plt.title("Residuals vs. Predicted P&L")
        plt.xlabel("Predicted P&L")
        plt.ylabel("Residual (Pred - Actual)")
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "residual_vs_prediction.png")
        plt.close()
        
        # 3. Worst prediction periods
        preds["abs_error"] = preds["residual"].abs()
        worst_periods = preds.sort_values("abs_error", ascending=False).head(10)
        worst_periods.to_csv(output_dir / "reports" / "worst_predictions.csv", index=False)
        
        # 4. Best/Worst Permutations
        perm_stats = preds.groupby("permutation_id").agg({
            "actual_pnl": "sum",
            "abs_error": "mean"
        }).reset_index()
        
        perm_stats.sort_values("actual_pnl", ascending=False).to_csv(
            output_dir / "reports" / "permutation_performance.csv", index=False
        )

    def generate_feature_importance_plots(self, feature_importances: Dict[str, float], output_dir: Path) -> None:
        """Plot the top 20 features by importance."""
        if not feature_importances:
            return
            
        logger.info("Generating feature importance plot")
        df = pd.DataFrame(list(feature_importances.items()), columns=["Feature", "Importance"])
        df = df.sort_values("Importance", ascending=False).head(20)
        
        plt.figure(figsize=(12, 8), dpi=150)
        sns.barplot(x="Importance", y="Feature", data=df, palette="magma")
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "feature_importance.png")
        plt.close()

    def generate_shap_summary(self, experiment_id: str, output_dir: Path) -> None:
        """Generate a SHAP summary bar chart averaged across walk-forward folds."""
        shap_dir = output_dir / "shap"
        shap_files = sorted(glob.glob(str(shap_dir / "shap_fold_*.json")))

        if not shap_files:
            logger.warning("No SHAP JSON files found in %s — skipping SHAP summary.", shap_dir)
            return

        logger.info("Generating SHAP summary from %d fold files", len(shap_files))

        all_importances: Dict[str, list] = {}
        for path in shap_files:
            with open(path, "r") as f:
                fold_data = json.load(f)
            for feature, value in fold_data.items():
                all_importances.setdefault(feature, []).append(abs(value))

        avg_importance = {k: float(np.mean(v)) for k, v in all_importances.items()}
        df = pd.DataFrame(list(avg_importance.items()), columns=["Feature", "Mean |SHAP|"])
        df = df.sort_values("Mean |SHAP|", ascending=False).head(20)

        plt.figure(figsize=(12, 8), dpi=150)
        sns.barplot(x="Mean |SHAP|", y="Feature", data=df, palette="coolwarm")
        plt.title("Top 20 Features by Mean |SHAP| Value (Averaged Across Folds)")
        plt.tight_layout()
        fig_path = output_dir / "figures" / "shap_summary.png"
        plt.savefig(fig_path)
        plt.close()
        logger.info("SHAP summary plot saved to %s", fig_path)

    def generate_statistical_tests(self, experiment_id: str, output_dir: Path) -> None:
        """Run statistical tests on the final aggregated residuals."""
        logger.info("Running statistical tests on residuals for %s", experiment_id)
        
        preds = self.db.read_table("predictions")
        preds = preds[preds["experiment_id"] == experiment_id].sort_values("timestamp")
        
        if preds.empty:
            logger.warning("No predictions found for statistical tests.")
            return
            
        import statsmodels.api as sm
        from statsmodels.stats.stattools import durbin_watson
        import scipy.stats as stats
        from statsmodels.stats.diagnostic import het_white
        
        residuals = preds["actual_pnl"] - preds["predicted_pnl"]
        
        # 1. Autocorrelation: Durbin-Watson
        # DW ~ 2.0 means no autocorrelation. < 2 means positive, > 2 means negative.
        dw_stat = float(durbin_watson(residuals))
        
        # 2. Normality: Jarque-Bera
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        
        # 3. Heteroskedasticity: White Test
        # We need a regressor for White test. We will use predicted PnL and its square.
        X = sm.add_constant(preds[["predicted_pnl"]])
        X["predicted_pnl_sq"] = X["predicted_pnl"] ** 2
        
        try:
            white_test = het_white(residuals, X)
            lm_stat, lm_pvalue, f_stat, f_pvalue = white_test
        except Exception as e:
            logger.warning("White test failed: %s", e)
            lm_stat, lm_pvalue = np.nan, np.nan
        
        results = {
            "durbin_watson": {
                "statistic": dw_stat,
                "interpretation": "Ideal ~ 2.0. Substantial deviation implies residual autocorrelation."
            },
            "jarque_bera": {
                "statistic": float(jb_stat),
                "p_value": float(jb_pvalue),
                "interpretation": "p < 0.05 implies residuals are non-normal."
            },
            "white_test": {
                "statistic": float(lm_stat),
                "p_value": float(lm_pvalue),
                "interpretation": "p < 0.05 implies heteroskedasticity (variance depends on predicted value)."
            }
        }
        
        out_path = output_dir / "reports" / "statistical_tests.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info("Statistical test results saved to %s", out_path)

    def verify_figures(self, output_dir: Path) -> None:
        """Verify that all required figures were successfully generated before report creation."""
        # Core figures are always required
        required_figures = [
            "heatmap_permutation.png",
            "heatmap_pnl.png",
            "equity_curve.png",
            "residual_histogram.png",
            "residual_vs_prediction.png",
        ]

        for fig in required_figures:
            fig_path = output_dir / "figures" / fig
            assert fig_path.exists(), f"Failed to generate required figure: {fig}"

        # Optional figures — check only if they exist
        optional_figures = ["feature_importance.png", "shap_summary.png"]
        for fig in optional_figures:
            fig_path = output_dir / "figures" / fig
            if fig_path.exists():
                logger.info("Optional figure verified: %s", fig)
            else:
                logger.info("Optional figure not generated (OK): %s", fig)

        logger.info("All required figures validated successfully.")

    def generate_report(self, experiment_id: str, output_dir: Path, results: dict) -> None:
        """Compile a comprehensive Markdown report."""
        logger.info("Generating final evaluation report")
        
        report_path = output_dir / "reports" / "evaluation_report.md"
        
        # Load trading metrics
        metrics_file = output_dir / "reports" / "trading_metrics.json"
        trading_metrics_str = ""
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                tm = json.load(f)
            
            trading_metrics_str = "### Trading Metrics\n\n| Metric | Model Portfolio | Baseline |\n|---|---|---|\n"
            for k in tm["model"].keys():
                model_val = f"{tm['model'][k]:.4f}"
                base_val = f"{tm['baseline'][k]:.4f}"
                trading_metrics_str += f"| {k} | {model_val} | {base_val} |\n"
        else:
            raise ValueError("Trading metrics JSON not found.")
            
        # Load Baseline Regression Metrics
        baseline_reg_file = output_dir / "reports" / "baseline_regression_metrics.json"
        baseline_reg_str = ""
        if baseline_reg_file.exists():
            with open(baseline_reg_file, "r") as f:
                brm = json.load(f)
            baseline_reg_str = "### Baseline Regression Metrics\n\n| Baseline | MAE | RMSE | R² |\n|---|---|---|---|\n"
            for b_name, b_metrics in brm.items():
                baseline_reg_str += f"| {b_name} | {b_metrics['MAE']:.4f} | {b_metrics['RMSE']:.4f} | {b_metrics['R2']:.4f} |\n"
        
        # Load Statistical Tests
        stat_test_file = output_dir / "reports" / "statistical_tests.json"
        stat_test_str = ""
        if stat_test_file.exists():
            with open(stat_test_file, "r") as f:
                st = json.load(f)
            stat_test_str = "### Statistical Tests on Residuals\n\n| Test | Statistic | p-value/Interpretation |\n|---|---|---|\n"
            stat_test_str += f"| Durbin-Watson | {st['durbin_watson']['statistic']:.4f} | {st['durbin_watson']['interpretation']} |\n"
            stat_test_str += f"| Jarque-Bera | {st['jarque_bera']['statistic']:.4f} | p={st['jarque_bera']['p_value']:.4e} |\n"
            stat_test_str += f"| White Test | {st['white_test']['statistic']:.4f} | p={st['white_test']['p_value']:.4e} |\n"


        # Query walk forward results for Model Comparison and Summary
        sql_wf = "SELECT fold, best_model, mae, rmse, r2, train_size, test_size FROM walk_forward_results WHERE experiment_id = ?"
        wf_results = self.db.read_sql(sql_wf, (experiment_id,))
        
        if wf_results.empty:
            raise ValueError("No walk-forward results found for this experiment.")
            
        # Model Comparison Table
        model_comp = wf_results.groupby("best_model")[["mae", "rmse", "r2"]].mean().reset_index()
        model_comp = model_comp.sort_values("mae", ascending=True) # Lowest MAE is best
        best_overall = model_comp.iloc[0]["best_model"]
        
        model_comp_str = "### Model Comparison (Averaged across Folds)\n\n| Model | Mean MAE | Mean RMSE | Mean R² |\n|---|---|---|---|\n"
        for _, row in model_comp.iterrows():
            highlight = "**" if row["best_model"] == best_overall else ""
            model_comp_str += f"| {highlight}{row['best_model']}{highlight} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['r2']:.4f} |\n"
            
        # Walk-Forward Summary
        wf_summary_str = "### Walk-Forward Fold Details\n\n| Fold | Best Model | MAE | RMSE | R² |\n|---|---|---|---|---|\n"
        for _, row in wf_results.iterrows():
            wf_summary_str += f"| {row['fold']} | {row['best_model']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['r2']:.4f} |\n"
            
        avg_mae = wf_results['mae'].mean()
        avg_rmse = wf_results['rmse'].mean()
        avg_r2 = wf_results['r2'].mean()
        
        wf_summary_str += f"\n**Aggregates**: Average MAE: {avg_mae:.4f} | Average RMSE: {avg_rmse:.4f} | Average R²: {avg_r2:.4f}\n"
        wf_summary_str += f"**Folds**: {len(wf_results)} | **Avg Train Size**: {wf_results['train_size'].mean():.0f} | **Avg Test Size**: {wf_results['test_size'].mean():.0f}\n"

        # Feature Importance Table
        fi_str = "### Feature Importance\n\n"
        if "feature_importances" in results and results["feature_importances"]:
            df_fi = pd.DataFrame(list(results["feature_importances"].items()), columns=["Feature", "Importance"])
            df_fi = df_fi.sort_values("Importance", ascending=False).head(20)
            fi_str += "| Rank | Feature | Importance |\n|---|---|---|\n"
            for i, (_, row) in enumerate(df_fi.iterrows(), 1):
                fi_str += f"| {i} | {row['Feature']} | {row['Importance']:.4f} |\n"
            fi_str += "\n![Feature Importance](../figures/feature_importance.png)\n"
        else:
            fi_str += "No feature importance data available.\n"

        # SHAP reference if plot exists
        shap_str = ""
        shap_fig = output_dir / "figures" / "shap_summary.png"
        if shap_fig.exists():
            shap_str = "\n### SHAP Analysis\n\nSHAP (SHapley Additive exPlanations) values provide model-agnostic feature attribution. "
            shap_str += "The following plot shows the top 20 features by mean |SHAP| value averaged across all walk-forward folds.\n\n"
            shap_str += "![SHAP Summary](../figures/shap_summary.png)\n"

        # Discussion section using actual metric values
        best_model_name = model_comp.iloc[0]['best_model']
        best_mae = model_comp.iloc[0]['mae']
        best_r2 = model_comp.iloc[0]['r2']

        discussion_str = f"The best performing model is **{best_model_name}** with an average MAE of **{best_mae:.4f}** and an average R² of **{best_r2:.4f}** across walk-forward folds.\n\n"
        if best_r2 < 0:
            discussion_str += f"⚠️ The R² value is **negative** ({best_r2:.4f}), indicating the model performs worse than simply predicting the mean of the target variable. "
            discussion_str += "This suggests the model may be struggling to capture meaningful patterns in the data, or the prediction target is inherently noisy.\n\n"
        elif best_r2 < 0.1:
            discussion_str += f"The R² value is very low ({best_r2:.4f}), suggesting limited explanatory power. The model captures only a small fraction of variance in the target.\n\n"
        else:
            discussion_str += f"The R² value of {best_r2:.4f} indicates the model explains a meaningful portion of variance in the target variable.\n\n"

        # Trading value assessment
        if metrics_file.exists():
            model_sharpe = tm["model"].get("Sharpe Ratio", 0.0)
            baseline_sharpe = tm["baseline"].get("Sharpe Ratio", 0.0)
            model_pf = tm["model"].get("Profit Factor", 0.0)
            if model_sharpe > baseline_sharpe:
                discussion_str += f"The model portfolio achieves a Sharpe Ratio of **{model_sharpe:.4f}** vs baseline **{baseline_sharpe:.4f}**, suggesting the model adds value over the default strategy.\n\n"
            else:
                discussion_str += f"The model portfolio Sharpe Ratio (**{model_sharpe:.4f}**) does not exceed the baseline (**{baseline_sharpe:.4f}**). The model may not add sufficient value over the default strategy in its current form.\n\n"
            discussion_str += f"The model portfolio Profit Factor is **{model_pf:.4f}**, "
            if model_pf > 1.0:
                discussion_str += "indicating gross profits exceed gross losses.\n"
            else:
                discussion_str += "indicating gross losses exceed or match gross profits — the strategy is not yet profitable on a gross basis.\n"

        # --- Format Markdown Report with Research-Oriented Sections ---
        
        from datetime import datetime, timezone
        markdown_content = f"""# Quantitative ML Pipeline — Evaluation Report

Experiment ID: `{experiment_id}`
Date: {datetime.now(timezone.utc).isoformat()}

---

## 1. Executive Summary

This report evaluates the out-of-sample performance of the Quantitative ML Pipeline. It assesses whether the machine learning models capture sufficient signal to outperform the baseline permutation strategy.

### Observations
{discussion_str}

### Interpretation
- **R² and Predictability**: A low or negative R² indicates the absolute magnitude of PnL is dominated by noise, making exact point forecasting extremely difficult.
- **Trading Value**: Even with low R², if the Sharpe Ratio and Profit Factor are superior to the baseline, the model successfully captures directional edge or risk-avoidance signals.

### Recommendations
- If Sharpe is high but R² is low, deploy the model as a directional filter rather than a precise forecaster.
- If Sharpe is below baseline, the model requires feature engineering improvements or regime-aware conditioning before production use.

---

## 2. Model Diagnostics & Stability

{model_comp_str}

{wf_summary_str}

### Observations
The best performing model on average was **{best_model_name}**. The performance varied across the {len(wf_results)} walk-forward folds. 

### Interpretation
Inconsistent performance across folds (high variance in MAE) indicates regime-dependency. If a tree model consistently beats linear models, the underlying signal contains significant non-linear interactions. If a regularized linear model (Ridge/Lasso) wins, the signal is primarily linear but noisy.

### Recommendations
- Review the `model_stability_report.md` to see if `{best_model_name}` won by a narrow margin or dominated every fold.
- Consider Stacking multiple models if no single model maintains dominance across all market regimes.

---

## 3. Baseline Comparison & Statistical Tests

{baseline_reg_str}
{stat_test_str}

### Observations
We explicitly compare the models against a Median Predictor, Mean Predictor, Random Noise, and a Rolling Window average. Statistical tests on the residuals assess normality and heteroskedasticity.

### Interpretation
- If the ML model cannot beat the Rolling Mean or Median baseline in MAE, the complexity of the ML approach is currently unjustified.
- If the White Test p-value is < 0.05, the model's errors are heteroskedastic (variance changes over time), meaning the model is less confident in certain regimes (e.g., high volatility).

### Recommendations
- Focus on beating the "Previous Trade" and "Rolling Mean" baselines, as beating a static "Mean" is trivial in financial time series.
- If residuals are highly non-normal (Jarque-Bera p < 0.05), consider transforming the target variable (e.g., log, Box-Cox) or using robust loss functions like Huber loss.

---

## 4. Trading Strategy & Equity Curve

{trading_metrics_str}

![Equity Curve](../figures/equity_curve.png)

### Observations
The model portfolio achieves a Sharpe Ratio of {tm['model'].get('Sharpe Ratio', 0.0):.4f} and a Profit Factor of {tm['model'].get('Profit Factor', 0.0):.4f}. The equity curve visually contrasts this against the default baseline strategy.

### Interpretation
The equity curve represents the true "bottom line". A model with poor regression metrics (MAE/R²) can still produce a superior equity curve if its errors are concentrated on small trades but it accurately predicts the direction of large, profitable trades.

### Recommendations
- If the equity curve is superior but draws down heavily in specific periods, investigate the macro regime during those drawdowns.
- If the Profit Factor is < 1.5, transaction costs (slippage/commissions) may erode profitability in live trading.

---

## 5. Feature Importance & Interpretability

{fi_str}
{shap_str}

### Observations
The top features driving predictions typically fall into distinct categories: short-term momentum, macro-economic releases, or temporal seasonality.

### Interpretation
- If lagged PnL features dominate, the system is primarily trend-following.
- If macro features (e.g., `days_since_release`) dominate, the strategy is highly sensitive to the economic calendar.

### Recommendations
- Drop features with exactly 0.0 importance to reduce dimensionality and overfitting.
- Investigate the SHAP summary plot: if a feature has high importance but its SHAP beeswarm shows no clear directional gradient, the model may be using it as a structural split rather than a linear predictor.

---

## 6. Error Analysis

![Residual Histogram](../figures/residual_histogram.png)
![Residual vs Prediction](../figures/residual_vs_prediction.png)

### Observations
The residual plots highlight the distribution of prediction errors and any structural bias across the prediction range.

### Interpretation
- A heavy-tailed residual histogram confirms the target contains extreme outliers (black swans) that the model failed to anticipate.
- A funnel shape in the Residual vs Prediction plot confirms heteroskedasticity.

### Recommendations
- Consider predicting volatility as a separate target and scaling position sizing inversely to predicted volatility.

---

## 7. Recommendation Heatmaps

![Permutation Heatmap](../figures/heatmap_permutation.png)
![P&L Heatmap](../figures/heatmap_pnl.png)


## 13. Future Improvements

Concrete next steps to improve pipeline performance:

1. **Optuna for Bayesian Hyperparameter Optimization** — Replace grid search with Optuna's TPE sampler for more efficient hyperparameter tuning across all model types.
2. **Multi-Asset Extension** — Extend the pipeline to handle multiple correlated assets simultaneously, enabling cross-asset signal generation and portfolio-level optimization.
3. **Real-Time Prediction Serving** — Deploy trained models behind a low-latency inference API for live trading signal generation with sub-second response times.
4. **Alternative Targets** — Experiment with risk-adjusted return targets (e.g., Sharpe ratio, Sortino ratio) instead of raw P&L to encourage more stable model behavior.
5. **Ensemble Methods Across Walk-Forward Folds** — Combine predictions from models trained on different walk-forward windows using stacking or weighted averaging to reduce variance and improve robustness.
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        logger.info("Evaluation report saved to %s", report_path)

    def run_evaluation(
        self,
        experiment_id: str,
        output_dir: Path,
        walk_forward_results: Dict[int, Any],
    ) -> None:
        """Run all evaluation metrics and generate a markdown report.

        Args:
            experiment_id: The ID of the experiment to evaluate.
            output_dir: Directory to save plots and reports.
            walk_forward_results: Output from the trainer.
        """
        logger.info("Starting evaluation for experiment %s", experiment_id)
        
        self._verify_pipeline_integrity(experiment_id)

        df = self._load_predictions(experiment_id)
        if df.empty:
            raise ValueError("No predictions found for evaluation.")

        # 2. Figures Generation
        self.generate_heatmap(experiment_id, output_dir)
        self.generate_equity_curve(experiment_id, output_dir)
        self.generate_error_analysis(experiment_id, output_dir)
        
        # Determine feature importance from provided results if available
        feature_importances = walk_forward_results.get("feature_importances", {})
        if feature_importances:
            self.generate_feature_importance_plots(feature_importances, output_dir)

        self.generate_shap_summary(experiment_id, output_dir)
        
        # New Audits
        self.generate_baseline_report(experiment_id, output_dir)
        self.generate_target_audit(output_dir)
        self.generate_data_quality_report(output_dir)
        self.generate_feature_drift_report(output_dir)
        self.generate_statistical_tests(experiment_id, output_dir)

        # 3. Figure Verification
        self.verify_figures(output_dir)
        
        # 4. Final Report
        self.generate_report(experiment_id, output_dir, walk_forward_results)
