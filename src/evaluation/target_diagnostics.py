import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.utils.config import PipelineConfig
from src.database.base import DatabaseClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TargetDiagnoser:
    def __init__(self, config: PipelineConfig, db: DatabaseClient):
        self.config = config
        self.db = db

    def generate_diagnostics(self, experiment_id: str, output_dir: Path) -> None:
        logger.info("Starting Target Diagnostics")
        
        # Target variable is actual_pnl (or 'pnl') in trades or predictions
        # Let's use feature_store target column (or 'pnl' if target is renamed)
        df = self.db.read_sql("SELECT timestamp, pnl FROM feature_store ORDER BY timestamp")
        if df.empty or 'pnl' not in df.columns:
            logger.warning("Target column 'pnl' not found in feature_store. Skipping target diagnostics.")
            return

        target = df['pnl'].dropna()
        if len(target) < 10:
            logger.warning("Insufficient target data for diagnostics.")
            return
            
        # 1. Distribution Statistics
        mean_val = target.mean()
        var_val = target.var()
        std_val = target.std()
        skew_val = target.skew()
        kurt_val = target.kurtosis()
        
        # SNR (Mean / Std)
        snr = mean_val / std_val if std_val != 0 else np.nan
        
        # Outlier Analysis (Z-score > 3)
        z_scores = np.abs(scipy.stats.zscore(target))
        outliers_pct = (len(z_scores[z_scores > 3]) / len(target)) * 100
        
        # Jarque-Bera Test
        jb_stat, jb_p_value = scipy.stats.jarque_bera(target)
        
        # Stationarity Tests
        try:
            adf_stat, adf_p_value, _, _, _, _ = adfuller(target)
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            adf_stat, adf_p_value = np.nan, np.nan
            
        try:
            kpss_stat, kpss_p_value, _, _ = kpss(target, regression='c', nlags="auto")
        except Exception as e:
            logger.warning(f"KPSS test failed: {e}")
            kpss_stat, kpss_p_value = np.nan, np.nan

        # Generate Plots
        plt.figure(figsize=(10, 6))
        sns.histplot(target, kde=True, bins=50)
        plt.title('Target (PnL) Distribution')
        plt.xlabel('PnL')
        plt.ylabel('Frequency')
        plt.savefig(output_dir / "figures" / "target_distribution.png")
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], target)
        plt.title('Target (PnL) Time Series')
        plt.xlabel('Time')
        plt.ylabel('PnL')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "target_timeseries.png")
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plot_acf(target, lags=40, alpha=0.05)
        plt.title('Target Autocorrelation (ACF)')
        plt.savefig(output_dir / "figures" / "target_acf.png")
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plot_pacf(target, lags=40, alpha=0.05)
        plt.title('Target Partial Autocorrelation (PACF)')
        plt.savefig(output_dir / "figures" / "target_pacf.png")
        plt.close()

        # Markdown Report
        md_path = output_dir / "reports" / "target_analysis.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Target Analysis Report\n\n")
            
            f.write("## Distribution Statistics\n")
            f.write(f"- **Mean**: {mean_val:.4f}\n")
            f.write(f"- **Variance**: {var_val:.4f}\n")
            f.write(f"- **Standard Deviation**: {std_val:.4f}\n")
            f.write(f"- **Skewness**: {skew_val:.4f}\n")
            f.write(f"- **Kurtosis**: {kurt_val:.4f}\n")
            f.write(f"- **Signal-to-Noise Ratio (SNR)**: {snr:.6f}\n")
            f.write(f"- **Outliers (>3 std)**: {outliers_pct:.2f}%\n\n")
            
            f.write("## Statistical Tests\n")
            f.write(f"- **Jarque-Bera (Normality)**: Stat = {jb_stat:.4f}, p-value = {jb_p_value:.4e}\n")
            f.write(f"- **Augmented Dickey-Fuller (Stationarity)**: Stat = {adf_stat:.4f}, p-value = {adf_p_value:.4e}\n")
            f.write(f"- **KPSS (Stationarity)**: Stat = {kpss_stat:.4f}, p-value = {kpss_p_value:.4e}\n\n")
            
            f.write("## Observations\n")
            f.write("The target distribution statistics and stationarity tests have been completed. ")
            f.write(f"The signal-to-noise ratio is {snr:.6f}, which is typical for financial time series but generally low. ")
            if adf_p_value < 0.05:
                f.write("The ADF test indicates the series is stationary. ")
            else:
                f.write("The ADF test indicates the series is non-stationary. ")
            f.write("\n\n")
            
            f.write("## Interpretation\n")
            f.write("A low SNR close to 0 explains why models struggle to achieve high R² scores; the variance is dominated by noise rather than signal. ")
            f.write("Significant kurtosis or skewness indicates fat tails, which linear models with MSE objectives may struggle to fit correctly without outlier clipping or robust loss functions. ")
            f.write("\n\n")
            
            f.write("## Recommendations\n")
            f.write("1. If R² remains near 0, do not interpret this as model failure. In financial data, a small R² with a positive Sharpe ratio is often sufficient.\n")
            f.write("2. Consider applying robust scaling or Winsorization to the target to clip outliers if MSE remains wildly unstable.\n")
            f.write("3. Evaluate models on directional accuracy or rank correlation (Spearman) rather than pure magnitude (MAE/R²) if noise dominates.\n")
            
        logger.info("Target diagnostics report generated successfully.")
