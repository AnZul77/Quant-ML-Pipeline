"""
SQLite implementation of the abstract DatabaseClient.

Uses the stdlib ``sqlite3`` module wrapped behind the common interface.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.database.base import DatabaseClient
from src.database.schema import SCHEMA_SQL
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteClient(DatabaseClient):
    """Concrete DatabaseClient backed by a local SQLite file."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------ #
    #  Connection lifecycle
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        # Enable WAL for better concurrent read performance
        self._conn.execute("PRAGMA journal_mode=WAL")
        logger.info("Connected to SQLite database at %s", self._db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Closed SQLite connection")

    def initialize_schema(self) -> None:
        assert self._conn is not None, "Database not connected"
        for statement in SCHEMA_SQL:
            self._conn.execute(statement)
        self._conn.commit()
        logger.info("Database schema initialised")

    # ------------------------------------------------------------------ #
    #  Generic CRUD
    # ------------------------------------------------------------------ #
    def execute(self, sql: str, params: Optional[tuple] = None) -> None:
        assert self._conn is not None, "Database not connected"
        if params:
            self._conn.execute(sql, params)
        else:
            self._conn.execute(sql)
        self._conn.commit()

    def executemany(self, sql: str, params_list: List[tuple]) -> None:
        assert self._conn is not None, "Database not connected"
        self._conn.executemany(sql, params_list)
        self._conn.commit()

    def fetchall(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        assert self._conn is not None, "Database not connected"
        cursor = self._conn.execute(sql, params or ())
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def fetchone(self, sql: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        assert self._conn is not None, "Database not connected"
        cursor = self._conn.execute(sql, params or ())
        row = cursor.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ #
    #  Pandas helpers
    # ------------------------------------------------------------------ #
    def read_table(self, table_name: str) -> pd.DataFrame:
        assert self._conn is not None, "Database not connected"
        return pd.read_sql(f"SELECT * FROM {table_name}", self._conn)

    def read_sql(self, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
        assert self._conn is not None, "Database not connected"
        return pd.read_sql(sql, self._conn, params=params)

    def write_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
    ) -> None:
        assert self._conn is not None, "Database not connected"
        df.to_sql(table_name, self._conn, if_exists=if_exists, index=False)
        logger.info("Wrote %d rows to table '%s'", len(df), table_name)

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #
    def table_exists(self, table_name: str) -> bool:
        assert self._conn is not None, "Database not connected"
        result = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return result is not None

    def row_count(self, table_name: str) -> int:
        assert self._conn is not None, "Database not connected"
        result = self._conn.execute(f"SELECT COUNT(*) AS cnt FROM {table_name}").fetchone()
        return int(result["cnt"]) if result else 0
