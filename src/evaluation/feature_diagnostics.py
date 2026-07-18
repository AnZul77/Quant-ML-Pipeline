import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.utils.config import PipelineConfig
from src.database.base import DatabaseClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FeatureDiagnoser:
    def __init__(self, config: PipelineConfig, db: DatabaseClient):
        self.config = config
        self.db = db

    def generate_diagnostics(self, experiment_id: str, output_dir: Path) -> None:
        logger.info("Starting Feature Matrix Diagnostics")
        
        df = self.db.read_sql("SELECT * FROM feature_store")
        if df.empty:
            logger.warning("Feature store is empty. Skipping feature diagnostics.")
            return

        exclude_cols = ["trade_id", "timestamp", "pnl", "target", "strategy", "account_id"]
        feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
        X = df[feature_cols].copy()
        
        if X.empty:
            logger.warning("No numeric features found for diagnostics.")
            return

        # 1. Feature Statistics
        stats_list = []
        for col in feature_cols:
            s = X[col]
            n_missing = s.isna().sum()
            n_inf = np.isinf(s).sum()
            
            clean_s = s.replace([np.inf, -np.inf], np.nan).dropna()
            
            mean_val = clean_s.mean() if len(clean_s) > 0 else np.nan
            std_val = clean_s.std() if len(clean_s) > 1 else np.nan
            min_val = clean_s.min() if len(clean_s) > 0 else np.nan
            max_val = clean_s.max() if len(clean_s) > 0 else np.nan
            skew_val = clean_s.skew() if len(clean_s) > 2 else np.nan
            kurt_val = clean_s.kurtosis() if len(clean_s) > 3 else np.nan
            
            stats_list.append({
                "Feature": col,
                "Mean": mean_val,
                "Std": std_val,
                "Min": min_val,
                "Max": max_val,
                "Skewness": skew_val,
                "Kurtosis": kurt_val,
                "Missing": n_missing,
                "PctMissing": (n_missing / len(X)) * 100,
                "Infinite": n_inf,
                "PctInfinite": (n_inf / len(X)) * 100,
                "Constant": (min_val == max_val) if not pd.isna(min_val) else True,
                "FirstAppearance": df["timestamp"].iloc[s.first_valid_index()] if s.first_valid_index() is not None else None,
                "LastAppearance": df["timestamp"].iloc[s.last_valid_index()] if s.last_valid_index() is not None else None,
            })
            
        stats_df = pd.DataFrame(stats_list)
        stats_csv = output_dir / "reports" / "feature_statistics.csv"
        stats_df.to_csv(stats_csv, index=False)
        
        # 2. Condition Number and VIF
        clean_X = X.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_X) > 0:
            try:
                X_scaled = (clean_X - clean_X.mean()) / (clean_X.std() + 1e-8)
                _, s_vals, _ = np.linalg.svd(X_scaled.values)
                condition_number = s_vals[0] / s_vals[-1]
            except Exception as e:
                logger.warning(f"Failed to compute condition number: {e}")
                condition_number = np.nan
            
            vif_data = []
            if len(feature_cols) <= 50:
                X_vif = clean_X.copy()
                X_vif['intercept'] = 1
                try:
                    for i in range(len(X_vif.columns)):
                        col_name = X_vif.columns[i]
                        if col_name != 'intercept':
                            vif_val = variance_inflation_factor(X_vif.values, i)
                            vif_data.append({"Feature": col_name, "VIF": vif_val})
                except Exception as e:
                    logger.warning(f"Failed to compute VIF: {e}")
            
            vif_df = pd.DataFrame(vif_data)
        else:
            condition_number = np.nan
            vif_df = pd.DataFrame()

        # 3. Correlation Matrix and Heatmap
        corr_matrix = clean_X.corr()
        corr_csv = output_dir / "reports" / "correlation_matrix.csv"
        corr_matrix.to_csv(corr_csv)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "correlation_heatmap.png")
        plt.close()
        
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.9:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                    
        # 4. Generate Markdown Report
        md_path = output_dir / "reports" / "feature_diagnostics.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Feature Matrix Diagnostics\n\n")
            
            f.write("## Overall Matrix Health\n")
            f.write(f"- **Total Rows**: {len(X)}\n")
            f.write(f"- **Total Numeric Features**: {len(feature_cols)}\n")
            f.write(f"- **Condition Number**: {condition_number:.2f}\n")
            
            f.write("\n## Suspicious Features\n")
            constant_features = stats_df[stats_df["Constant"] == True]["Feature"].tolist()
            high_missing = stats_df[stats_df["PctMissing"] > 10]["Feature"].tolist()
            inf_features = stats_df[stats_df["PctInfinite"] > 0]["Feature"].tolist()
            
            if constant_features:
                f.write("### Constant Features\n")
                f.write("These features have zero variance and should be removed:\n")
                for c in constant_features:
                    f.write(f"- {c}\n")
            else:
                f.write("No constant features detected.\n")
                
            if high_missing:
                f.write("\n### High Missing Values (>10%)\n")
                for c in high_missing:
                    pct = stats_df.loc[stats_df['Feature'] == c, 'PctMissing'].values[0]
                    f.write(f"- {c} ({pct:.1f}%)\n")
            
            if inf_features:
                f.write("\n### Contains Infinities\n")
                for c in inf_features:
                    f.write(f"- {c}\n")
                    
            f.write("\n## Multicollinearity\n")
            if high_corr:
                f.write("### Pairwise Correlations > 0.9\n")
                for c1, c2, val in high_corr:
                    f.write(f"- {c1} & {c2}: {val:.3f}\n")
            else:
                f.write("No pairwise correlations > 0.9 detected.\n")
                
            if not vif_df.empty:
                high_vif = vif_df[vif_df["VIF"] > 10]
                if not high_vif.empty:
                    f.write("\n### High Variance Inflation Factor (VIF > 10)\n")
                    for _, row in high_vif.iterrows():
                        f.write(f"- {row['Feature']}: {row['VIF']:.2f}\n")
                
            f.write("\n## Observations\n")
            f.write(f"Found {len(constant_features)} constant features, {len(high_missing)} features with >10% missing data, and {len(high_corr)} highly correlated pairs.\n")
            
            f.write("\n## Interpretation\n")
            f.write("Constant features provide zero predictive power. High missingness suggests incomplete data sources. High collinearity (VIF > 10) can cause instability in linear models and bloat tree models.\n")
            
            f.write("\n## Recommendations\n")
            f.write("1. Remove all constant features before training.\n")
            f.write("2. Drop or impute features with high missingness.\n")
            f.write("3. For highly correlated pairs, drop one of the features or apply PCA.\n")

        logger.info("Feature diagnostics report generated successfully.")
