# Quant ML Pipeline

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)

A production-grade Quantitative Machine Learning Pipeline designed to optimize algorithm permutation strategies. This pipeline evaluates parameterised trading strategies using rigorous walk-forward cross-validation, generates predictive features from official macroeconomic data (FRED), and leverages GPU-accelerated gradient boosting to predict expected P&L.

## 🏗️ Pipeline Architecture

The pipeline follows a modular, robust design with strict separation between data ingestion, feature engineering, modeling, and evaluation.

```mermaid
graph TD
    subgraph Data Layer
        A[Trading Logs CSV] -->|Clean & Dedup| C(Trades Table)
        B[FRED Economic Data API] -->|Normalise| D(Macro Series Table)
    end

    subgraph Feature Engineering
        C --> E[Feature Engineer]
        D --> E
        E -->|Time/Rolling/Macro| F(Feature Store Table)
    end

    subgraph Model Training
        F --> G[Walk-Forward Generator]
        G --> H[Model Trainer GPU/Optuna]
        H -->|Best Model per Fold| I(Walk-Forward Results Table)
    end

    subgraph Evaluation & Diagnostics
        H --> J[Evaluator]
        F --> K[Predictor]
        K -->|Score Permutations| L(Predictions Table)
        K --> M[Recommendation Matrix]
        J --> N[Evaluation & Stability Reports]
    end
```

## ✨ Key Features (v2.0)

*   **Official Economic Data**: Integrates directly with the Federal Reserve Economic Data (FRED) API for robust, revision-free macroeconomic indicators (CPI, NFP, GDP, Fed Funds, etc.).
*   **GPU & Multithreading**: Leverages RTX/CUDA acceleration for XGBoost, LightGBM, and CatBoost. Hyperparameter tuning is powered by **Optuna** running concurrently across all available CPU cores.
*   **Leakage Prevention**: Enforces strict shift-based rolling window feature engineering and walk-forward cross-validation to ensure zero look-ahead bias.
*   **Advanced Diagnostics**: Generates complete feature drift (PSI), statistical target distribution, condition numbers, and VIF audits before training even begins.
*   **Research-Grade Reporting**: Outputs a full markdown evaluation report breaking down Equity Curves, Trading Metrics (Sharpe, Sortino, Calmar), Feature Importance, SHAP beeswarm plots, and Residual statistical tests (White Test, Durbin-Watson).

## 🗄️ Database Schema

The pipeline uses SQLite by default (with PostgreSQL support) to maintain relational integrity:

```mermaid
erDiagram
    TRADES ||--o{ FEATURE_STORE : "1:1"
    TRADES ||--o{ PREDICTIONS : "1:N"
    EXPERIMENTS ||--o{ PREDICTIONS : "1:N"
    EXPERIMENTS ||--o{ WALK_FORWARD_RESULTS : "1:N"

    TRADES {
        integer id PK
        datetime timestamp
        string account
        float pnl
        string permutation_id
    }
    
    MACRO_SERIES {
        string series_id
        datetime date
        float value
        datetime release_date
        string frequency
    }

    FEATURE_STORE {
        integer trade_id PK, FK
        datetime feature_timestamp
        string market_session
        float rolling_pnl_5
        float days_since_release
        float pnl "Target"
    }

    PREDICTIONS {
        integer id PK
        integer trade_id FK
        datetime timestamp
        string permutation_id
        float predicted_pnl
        float actual_pnl
        integer walk_forward_fold
        string experiment_id FK
    }
```

## 🚀 Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/QuantML-Pipeline.git
cd QuantML-Pipeline
```

2. **Set up the virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **FRED API Key (Optional but Recommended)**
While the pipeline will work without an API key by using FRED's public tiers, providing a key prevents rate limits.
```bash
export FRED_API_KEY="your_api_key_here"
```

## 🛠️ Usage

The pipeline is driven by a Click-based CLI in `main.py`.

### End-to-End Pipeline
Run the entire sequence (Ingest → Clean → Features → Train → Evaluate → Predict):
```bash
python main.py pipeline
```

### Individual Steps

```bash
# 1. Download official macroeconomic data from FRED
python main.py scrape

# 2. Ingest raw historical trade CSV data
python main.py ingest

# 3. Clean trades (account filtering & VWAP deduplication)
python main.py clean

# 4. Generate features, drift reports, and feature lineage
python main.py features

# 5. Train GPU models using Optuna hyperparameter tuning
python main.py train <experiment_id>

# 6. Evaluate results (reports, heatmaps, equity curves, SHAP)
python main.py evaluate <experiment_id>

# 7. Generate permutation recommendation matrix
python main.py predict <experiment_id>
```

## 📊 Evaluation Methodology

The pipeline uses strict **Walk-Forward Cross-Validation** to prevent data leakage and evaluate real-world performance. 

For a typical dataset:
*   Train window: 30 days (Expanding)
*   Test window: 7 days
*   Inside each train window, nested `TimeSeriesSplit(n_splits=3)` is used for Optuna Bayesian hyperparameter optimization.

Models compared: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, LightGBM, CatBoost, and XGBoost. The pipeline automatically selects the best model per fold based on Validation MAE.

## 📈 Example Outputs

After running the pipeline, check the `outputs/exp_<timestamp>/` directory for a full research audit:

*   **`reports/evaluation_report.md`**: Executive summary, Trading metrics (Sharpe, Drawdown), and model diagnostics.
*   **`reports/feature_lineage.md`**: Tracks every feature back to its mathematical and data origin.
*   **`reports/model_stability.md`**: Coefficient of Variation (CV) and 95% Confidence Intervals for cross-fold model performance.
*   **`reports/experiment_metadata.json`**: Captures exact library versions, random seeds, and git commit hashes for absolute reproducibility.
*   **`figures/equity_curve.png`**: Cumulative P&L of the model vs baseline strategy over time.
*   **`figures/heatmap_pnl.png`**: 7x24 grid showing realized P&L per recommended permutation.
*   **`figures/shap_summary.png`**: Global SHAP values explaining non-linear feature impact.

## ⚠️ Limitations

1. **Transaction Costs**: The current P&L metrics do not deduct slippage or commission, which would alter net profitability.
2. **Order Book Data**: We only use executed trades. Level 2 order book imbalance features would likely improve short-term predictive power.
3. **Static Windows**: The train/test split size is static. Dynamic window sizing based on regime change detection could improve stability.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.
