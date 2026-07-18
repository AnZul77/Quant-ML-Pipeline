"""
PostgreSQL implementation of the abstract DatabaseClient.

This is a **placeholder** that mirrors the SQLite interface.  It requires
``psycopg2`` and a running PostgreSQL instance.  The pipeline defaults to
SQLite; switch to PostgreSQL by setting ``database.engine: postgres`` in
``config.yaml``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.database.base import DatabaseClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PostgresClient(DatabaseClient):
    """Concrete DatabaseClient backed by PostgreSQL.

    .. note::
        This implementation requires ``psycopg2-binary`` to be installed.
        It is intentionally left as a thin stub so that the pipeline can
        be extended to production PostgreSQL without changing any calling
        code.
    """

    def __init__(self, host: str, port: int, database: str, user: str, password: str) -> None:
        self._dsn = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }
        self._conn = None

    # ------------------------------------------------------------------ #
    #  Connection lifecycle
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError as exc:
            raise ImportError(
                "psycopg2-binary is required for PostgreSQL support.  "
                "Install it with: pip install psycopg2-binary"
            ) from exc

        self._conn = psycopg2.connect(**self._dsn)
        self._conn.autocommit = False
        logger.info("Connected to PostgreSQL at %s:%s/%s", self._dsn["host"], self._dsn["port"], self._dsn["database"])

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Closed PostgreSQL connection")

    def initialize_schema(self) -> None:
        from src.database.schema import SCHEMA_SQL

        assert self._conn is not None, "Database not connected"
        cursor = self._conn.cursor()
        for statement in SCHEMA_SQL:
            # Adapt SQLite-specific syntax for PostgreSQL
            adapted = statement.replace("AUTOINCREMENT", "")
            cursor.execute(adapted)
        self._conn.commit()
        logger.info("Database schema initialised (PostgreSQL)")

    # ------------------------------------------------------------------ #
    #  Generic CRUD
    # ------------------------------------------------------------------ #
    def execute(self, sql: str, params: Optional[tuple] = None) -> None:
        assert self._conn is not None, "Database not connected"
        cursor = self._conn.cursor()
        cursor.execute(sql, params)
        self._conn.commit()

    def executemany(self, sql: str, params_list: List[tuple]) -> None:
        assert self._conn is not None, "Database not connected"
        cursor = self._conn.cursor()
        cursor.executemany(sql, params_list)
        self._conn.commit()

    def fetchall(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        assert self._conn is not None, "Database not connected"
        import psycopg2.extras

        cursor = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def fetchone(self, sql: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        assert self._conn is not None, "Database not connected"
        import psycopg2.extras

        cursor = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ #
    #  Pandas helpers
    # ------------------------------------------------------------------ #
    def read_table(self, table_name: str) -> pd.DataFrame:
        return self.read_sql(f"SELECT * FROM {table_name}")

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
        from sqlalchemy import create_engine

        url = (
            f"postgresql://{self._dsn['user']}:{self._dsn['password']}"
            f"@{self._dsn['host']}:{self._dsn['port']}/{self._dsn['database']}"
        )
        engine = create_engine(url)
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        logger.info("Wrote %d rows to table '%s' (PostgreSQL)", len(df), table_name)

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #
    def table_exists(self, table_name: str) -> bool:
        result = self.fetchone(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
            (table_name,),
        )
        return bool(result and result.get("exists", False))

    def row_count(self, table_name: str) -> int:
        result = self.fetchone(f"SELECT COUNT(*) AS cnt FROM {table_name}")
        return int(result["cnt"]) if result else 0
