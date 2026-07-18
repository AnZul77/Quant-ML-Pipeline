"""
Configuration loader for the Quantitative ML Pipeline.

Loads YAML configuration files, merges with environment variables,
and provides typed access to all pipeline settings. Supports CLI
overrides for train-window, test-window, seed, and model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Default path relative to project root
_DEFAULT_CONFIG_PATH = Path("config") / "config.yaml"


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains config/)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "config" / "config.yaml").exists():
            return current
        current = current.parent
    # Fallback: assume CWD
    return Path.cwd()


class PipelineConfig:
    """Centralised, immutable-ish configuration object for the pipeline.

    Attributes:
        raw: The raw dictionary loaded from YAML + env overrides.
    """

    def __init__(self, config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> None:
        self.project_root = _find_project_root()

        if config_path is None:
            config_path_resolved = self.project_root / _DEFAULT_CONFIG_PATH
        else:
            config_path_resolved = Path(config_path).resolve()

        with open(config_path_resolved, "r", encoding="utf-8") as fh:
            self.raw: Dict[str, Any] = yaml.safe_load(fh)

        # Environment variable overrides
        self._apply_env_overrides()

        # CLI argument overrides
        if overrides:
            self._apply_cli_overrides(overrides)

    # ------------------------------------------------------------------ #
    #  Environment variable overrides
    # ------------------------------------------------------------------ #
    def _apply_env_overrides(self) -> None:
        """Override config values with environment variables where set."""
        te_key = os.environ.get("TE_API_KEY", "")
        if te_key:
            self.raw.setdefault("trading_economics", {})["api_key"] = te_key

        pg_password = os.environ.get("PG_PASSWORD", "")
        if pg_password:
            self.raw.setdefault("database", {}).setdefault("postgres", {})["password"] = pg_password

    # ------------------------------------------------------------------ #
    #  CLI overrides
    # ------------------------------------------------------------------ #
    def _apply_cli_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply CLI-level overrides (--seed, --model, --train-window, etc.)."""
        if "seed" in overrides and overrides["seed"] is not None:
            self.raw["seed"] = int(overrides["seed"])
        if "model" in overrides and overrides["model"] is not None:
            self.raw.setdefault("model", {})["default_model"] = overrides["model"]
        if "train_window" in overrides and overrides["train_window"] is not None:
            self.raw.setdefault("walk_forward", {})["train_window_days"] = int(overrides["train_window"])
        if "test_window" in overrides and overrides["test_window"] is not None:
            self.raw.setdefault("walk_forward", {})["test_window_days"] = int(overrides["test_window"])

    # ------------------------------------------------------------------ #
    #  Convenience accessors
    # ------------------------------------------------------------------ #
    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def db_engine(self) -> str:
        return self.raw["database"]["engine"]

    @property
    def db_sqlite_path(self) -> Path:
        return self.project_root / self.raw["database"]["sqlite"]["path"]

    @property
    def db_postgres(self) -> Dict[str, Any]:
        return self.raw["database"]["postgres"]

    @property
    def data_dir(self) -> Path:
        return self.project_root / self.raw["paths"]["data_dir"]

    @property
    def output_dir(self) -> Path:
        return self.project_root / self.raw["paths"]["output_dir"]

    @property
    def raw_logs_path(self) -> Path:
        return self.project_root / self.raw["paths"]["raw_logs"]

    @property
    def cache_dir(self) -> Path:
        return self.project_root / self.raw["paths"].get("cache_dir", "cache")

    # ---- Macro Scraper accessors ----

    @property
    def scraper_headless(self) -> bool:
        return bool(self.raw.get("macro_scraper", {}).get("headless", True))
        
    @property
    def scraper_timeout_ms(self) -> int:
        return int(self.raw.get("macro_scraper", {}).get("timeout_ms", 30000))
        
    @property
    def scraper_max_retries(self) -> int:
        return int(self.raw.get("macro_scraper", {}).get("max_retries", 3))
        
    @property
    def scraper_chunk_size_days(self) -> int:
        return int(self.raw.get("macro_scraper", {}).get("chunk_size_days", 30))
        
    @property
    def raw_macro_dir(self) -> Path:
        return self.project_root / self.raw.get("macro_scraper", {}).get("raw_macro_dir", "data/raw_macro")

    @property
    def scraper_countries(self) -> List[str]:
        return self.raw.get("macro_scraper", {}).get("countries", ["united states"])

    @property
    def scraper_event_types(self) -> List[str]:
        return self.raw.get("macro_scraper", {}).get("event_types", [])

    # ---- Cleaning ----

    @property
    def dedup_window_ms(self) -> int:
        return int(self.raw.get("cleaning", {}).get("dedup_window_ms", 500))

    @property
    def invalid_accounts(self) -> List[str]:
        return self.raw.get("cleaning", {}).get("invalid_accounts", [])

    # ---- Feature Engineering ----

    @property
    def feature_version(self) -> str:
        return self.raw.get("features", {}).get("version", "1.0")

    @property
    def rolling_short(self) -> int:
        return int(self.raw.get("features", {}).get("rolling_windows", {}).get("short", 5))

    @property
    def rolling_long(self) -> int:
        return int(self.raw.get("features", {}).get("rolling_windows", {}).get("long", 20))

    @property
    def train_window_days(self) -> int:
        return int(self.raw.get("walk_forward", {}).get("train_window_days", 30))

    @property
    def test_window_days(self) -> int:
        return int(self.raw.get("walk_forward", {}).get("test_window_days", 7))

    @property
    def baseline_permutation(self) -> str:
        return self.raw.get("model", {}).get("baseline_permutation", "P_DEFAULT")

    @property
    def target_column(self) -> str:
        return self.raw.get("model", {}).get("target_column", "pnl")

    @property
    def model_list(self) -> List[str]:
        return self.raw.get("model", {}).get("models", ["lightgbm"])

    @property
    def default_model(self) -> str:
        return self.raw.get("model", {}).get("default_model", "lightgbm")

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        return self.raw.get("model", {}).get("hyperparameters", {})

    @property
    def risk_free_rate(self) -> float:
        return float(self.raw.get("evaluation", {}).get("risk_free_rate", 0.0))

    @property
    def annualization_factor(self) -> int:
        return int(self.raw.get("evaluation", {}).get("annualization_factor", 252))

    @property
    def enable_shap(self) -> bool:
        return bool(self.raw.get("feature_flags", {}).get("enable_shap", True))

    @property
    def enable_optuna(self) -> bool:
        return bool(self.raw.get("feature_flags", {}).get("enable_optuna", False))

    @property
    def save_models(self) -> bool:
        return bool(self.raw.get("feature_flags", {}).get("save_models", True))

    @property
    def generate_report(self) -> bool:
        return bool(self.raw.get("feature_flags", {}).get("generate_report", True))

    @property
    def market_sessions(self) -> Dict[str, str]:
        return self.raw.get("features", {}).get("market_sessions", {})

    @property
    def macro_proximity_thresholds(self) -> List[int]:
        return self.raw.get("features", {}).get("macro_proximity_thresholds_minutes", [30, 60, 120])

    def get_experiment_dir(self, experiment_id: str) -> Path:
        """Return the output directory for a specific experiment."""
        return self.output_dir / experiment_id

    def to_dict(self) -> Dict[str, Any]:
        """Return a copy of the raw configuration dictionary."""
        import copy
        return copy.deepcopy(self.raw)
