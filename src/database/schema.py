"""
Database schema definitions for the Quantitative ML Pipeline.

Contains the DDL statements that ``DatabaseClient.initialize_schema``
executes.  The schema is designed to be compatible with both SQLite and
PostgreSQL (with minor adaptation in the Postgres client).
"""

from __future__ import annotations

from typing import List

# Each element is one CREATE TABLE statement.
SCHEMA_SQL: List[str] = [
    # ------------------------------------------------------------------ #
    #  Macroeconomic time-series from official providers
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS macro_series (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        series_id       TEXT    NOT NULL,
        date            TEXT    NOT NULL,
        value           REAL    NOT NULL,
        release_date    TEXT,
        frequency       TEXT,
        source          TEXT
    )
    """,

    # ------------------------------------------------------------------ #
    #  Macroeconomic events (derived from macro_series)
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS macro_events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_utc   TEXT    NOT NULL,
        event_name      TEXT    NOT NULL,
        country         TEXT,
        importance      TEXT,
        actual          REAL,
        forecast        REAL,
        previous        REAL,
        surprise        REAL,
        source          TEXT,
        scraper_version TEXT,
        raw_event_json  TEXT
    )
    """,

    # ------------------------------------------------------------------ #
    #  Raw (cleaned) trades
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS trades (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        account         TEXT,
        direction       TEXT,
        quantity         REAL,
        price           REAL,
        pnl             REAL,
        permutation_id  TEXT,
        holding_time    REAL,
        threshold       REAL
    )
    """,

    # ------------------------------------------------------------------ #
    #  Feature store -- one row per trade_id
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS feature_store (
        trade_id            INTEGER PRIMARY KEY,
        feature_version     TEXT,
        pipeline_version    TEXT,
        feature_timestamp   TEXT,

        -- Time features
        hour_of_day         INTEGER,
        day_of_week         INTEGER,
        week_of_month       INTEGER,
        month               INTEGER,
        quarter             INTEGER,
        market_session      TEXT,
        minutes_after_open  REAL,
        minutes_before_close REAL,

        -- Rolling trading features (shifted by 1)
        rolling_pnl_5       REAL,
        rolling_pnl_20      REAL,
        rolling_win_rate    REAL,
        rolling_avg_quantity REAL,
        rolling_trade_frequency REAL,
        rolling_volatility  REAL,
        rolling_drawdown    REAL,

        -- Macro proximity features
        days_since_release       REAL,
        days_until_next_release  REAL,
        release_frequency        REAL,
        
        -- Time-Series Features (Rolling & Expanding)
        rolling_mean             REAL,
        rolling_std              REAL,
        rolling_min              REAL,
        rolling_max              REAL,
        rolling_median           REAL,
        rolling_percent_change   REAL,
        rolling_zscore           REAL,
        macro_rolling_volatility REAL,
        macro_rolling_variance   REAL,
        macro_rolling_percentile REAL,
        macro_ewma               REAL,
        rolling_skewness         REAL,
        rolling_kurtosis         REAL,
        expanding_mean           REAL,
        expanding_std            REAL,
        
        -- Macro Trend & Interaction Features
        macro_momentum           REAL,
        macro_acceleration       REAL,
        macro_regime             REAL,
        yield_curve_spread       REAL,
        inflation_momentum       REAL,
        
        -- Lags
        lag_1                    REAL,
        lag_3                    REAL,
        lag_6                    REAL,
        lag_12                   REAL,

        -- Strategy features
        holding_time        REAL,
        threshold           REAL,
        risk_parameters     TEXT,
        permutation_id      TEXT,

        -- Target
        pnl                 REAL,

        FOREIGN KEY (trade_id) REFERENCES trades(id)
    )
    """,

    # ------------------------------------------------------------------ #
    #  Out-of-sample predictions
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id            INTEGER NOT NULL,
        timestamp           TEXT,
        permutation_id      TEXT,
        predicted_pnl       REAL,
        actual_pnl          REAL,
        walk_forward_fold   INTEGER,
        experiment_id       TEXT,
        FOREIGN KEY (trade_id) REFERENCES trades(id)
    )
    """,

    # ------------------------------------------------------------------ #
    #  Walk-forward fold metadata
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS walk_forward_results (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        fold        INTEGER,
        best_model  TEXT,
        mae         REAL,
        rmse        REAL,
        r2          REAL,
        train_size  INTEGER,
        test_size   INTEGER,
        train_start TEXT,
        train_end   TEXT,
        test_start  TEXT,
        test_end    TEXT,
        metrics     TEXT,
        experiment_id TEXT
    )
    """,

    # ------------------------------------------------------------------ #
    #  Experiment tracking
    # ------------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id   TEXT PRIMARY KEY,
        model           TEXT,
        parameters      TEXT,
        dataset_version TEXT,
        feature_version TEXT,
        provider_version TEXT,
        feature_count   INTEGER,
        series_used     TEXT,
        git_commit      TEXT,
        random_seed     INTEGER,
        created_at      TEXT
    )
    """,
]
