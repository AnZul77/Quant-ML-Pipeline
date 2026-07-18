"""
Structured logging for the Quantitative ML Pipeline.

Provides a pre-configured logger with coloured console output and
optional file logging.  Every pipeline module should obtain its logger
via ``get_logger(__name__)``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_ROOT_LOGGER_NAME = "quant_pipeline"
_initialised = False


def _init_root_logger(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Initialise the root pipeline logger (idempotent)."""
    global _initialised
    if _initialised:
        return

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(file_handler)

    _initialised = True


def get_logger(name: str, level: int = logging.INFO, log_file: Optional[Path] = None) -> logging.Logger:
    """Return a child logger under the pipeline root.

    Args:
        name: Typically ``__name__`` of the calling module.
        level: Logging level (default ``INFO``).
        log_file: Optional path to a log file.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    _init_root_logger(level=level, log_file=log_file)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
