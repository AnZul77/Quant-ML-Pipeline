import logging
from pathlib import Path
from typing import Dict, Any

from src.utils.config import PipelineConfig
from src.database.base import DatabaseClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DatabaseAuditor:
    def __init__(self, config: PipelineConfig, db: DatabaseClient):
        self.config = config
        self.db = db

    def generate_audit(self, output_dir: Path) -> None:
        logger.info("Starting Database Schema Audit")
        
        # We need to query SQLite system tables to find tables, columns, indexes
        # and see if they are empty or used.
        try:
            tables_df = self.db.read_sql("SELECT name FROM sqlite_master WHERE type='table'")
        except Exception as e:
            logger.warning(f"Database auditor only supports SQLite currently. {e}")
            return
            
        tables = tables_df['name'].tolist()
        
        audit_data = []
        for table in tables:
            # Skip sqlite system tables
            if table.startswith('sqlite_'):
                continue
                
            try:
                # Get column info
                cols_df = self.db.read_sql(f"PRAGMA table_info({table})")
                
                # Get row count
                count_df = self.db.read_sql(f"SELECT COUNT(*) as cnt FROM {table}")
                row_count = count_df.iloc[0]['cnt'] if not count_df.empty else 0
                
                # Identify potential orphaned columns (e.g., all nulls)
                orphaned_cols = []
                if row_count > 0:
                    for col in cols_df['name']:
                        # Skip large tables or just do a quick count
                        col_count = self.db.read_sql(f"SELECT COUNT({col}) as cnt FROM {table}")
                        if not col_count.empty and col_count.iloc[0]['cnt'] == 0:
                            orphaned_cols.append(col)
                            
                # Get indexes
                idx_df = self.db.read_sql(f"PRAGMA index_list({table})")
                
                audit_data.append({
                    "table": table,
                    "columns": len(cols_df),
                    "rows": row_count,
                    "orphaned_cols": orphaned_cols,
                    "indexes": len(idx_df)
                })
            except Exception as e:
                logger.warning(f"Failed to audit table {table}: {e}")

        # Markdown Report
        md_path = output_dir / "reports" / "database_audit.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Database Schema Audit\n\n")
            
            f.write("## Table Summary\n")
            f.write("| Table | Columns | Rows | Indexes | Orphaned Columns (100% NULL) |\n")
            f.write("|---|---|---|---|---|\n")
            for row in audit_data:
                orphans = ", ".join(row['orphaned_cols']) if row['orphaned_cols'] else "None"
                f.write(f"| {row['table']} | {row['columns']} | {row['rows']} | {row['indexes']} | {orphans} |\n")
                
            f.write("\n## Observations\n")
            unused_tables = [r['table'] for r in audit_data if r['rows'] == 0]
            if unused_tables:
                f.write(f"The following tables are completely empty and may be orphaned: {', '.join(unused_tables)}.\n")
            else:
                f.write("All tables contain data.\n")
                
            f.write("\n## Interpretation\n")
            f.write("Empty tables or columns with 100% NULL values represent legacy schema components that are no longer populated by the active pipeline architecture. ")
            f.write("They introduce technical debt and potential confusion during data ingestion and feature engineering.\n")
            
            f.write("\n## Recommendations\n")
            f.write("1. Drop empty tables if they are definitively deprecated.\n")
            f.write("2. Drop 100% NULL columns from the schema definition.\n")
            
        logger.info("Database schema audit generated successfully.")
