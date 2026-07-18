"""
Abstract database interface for the Quantitative ML Pipeline.

Defines the contract that concrete database implementations (SQLite,
PostgreSQL) must fulfil.  All pipeline modules interact with the database
exclusively through this interface, enabling engine-agnostic code.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional

import pandas as pd


class DatabaseClient(abc.ABC):
    """Abstract base class for database operations.

    Concrete subclasses must implement every abstract method.
    """

    # ------------------------------------------------------------------ #
    #  Connection lifecycle
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def connect(self) -> None:
        """Establish a connection to the database."""

    @abc.abstractmethod
    def close(self) -> None:
        """Close the database connection."""

    @abc.abstractmethod
    def initialize_schema(self) -> None:
        """Create all required tables if they do not already exist."""

    # ------------------------------------------------------------------ #
    #  Generic CRUD
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def execute(self, sql: str, params: Optional[tuple] = None) -> None:
        """Execute a single SQL statement (INSERT / UPDATE / DELETE / DDL)."""

    @abc.abstractmethod
    def executemany(self, sql: str, params_list: List[tuple]) -> None:
        """Execute a parameterised SQL statement for many rows."""

    @abc.abstractmethod
    def fetchall(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT and return all rows as list of dicts."""

    @abc.abstractmethod
    def fetchone(self, sql: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """Execute a SELECT and return the first row as a dict or None."""

    # ------------------------------------------------------------------ #
    #  Pandas helpers
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def read_table(self, table_name: str) -> pd.DataFrame:
        """Read an entire table into a DataFrame."""

    @abc.abstractmethod
    def read_sql(self, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Execute arbitrary SQL and return a DataFrame."""

    @abc.abstractmethod
    def write_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
    ) -> None:
        """Write a DataFrame to a database table.

        Args:
            df: The data to write.
            table_name: Target table name.
            if_exists: One of ``'append'``, ``'replace'``, ``'fail'``.
        """

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """Return True if *table_name* exists in the database."""

    @abc.abstractmethod
    def row_count(self, table_name: str) -> int:
        """Return the number of rows in *table_name*."""
