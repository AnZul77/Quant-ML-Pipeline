import pandas as pd
from typing import List

from src.database.base import DatabaseClient
from src.utils.config import PipelineConfig
from src.utils.logger import get_logger
from src.macro_ingestion.providers.fred import FredProvider

logger = get_logger(__name__)

class MacroOrchestrator:
    """
    Coordinates data fetching across all active providers, writes to `macro_series`,
    and derives `macro_events` for compatibility with legacy systems.
    """
    
    def __init__(self, config: PipelineConfig, db: DatabaseClient):
        self.config = config
        self.db = db
        self.providers = [FredProvider()]
        
    def run(self, start_date: str, end_date: str) -> None:
        """Execute the ingestion pipeline."""
        logger.info(f"Starting Macro Orchestrator run from {start_date} to {end_date}")
        
        series_ids = getattr(self.config, "macro_series_ids", ["CPIAUCSL", "UNRATE", "GDP", "FEDFUNDS"])
        
        all_series_data = []
        for provider in self.providers:
            for series_id in series_ids:
                df = provider.fetch_series(series_id, start_date, end_date)
                if not df.empty:
                    all_series_data.append(df)
                    
        if not all_series_data:
            logger.warning("No data fetched from any providers.")
            return
            
        final_series_df = pd.concat(all_series_data, ignore_index=True)
        
        # Ensure chronological order
        final_series_df["date"] = pd.to_datetime(final_series_df["date"])
        final_series_df.sort_values(by=["series_id", "date"], inplace=True)
        final_series_df["date"] = final_series_df["date"].dt.strftime("%Y-%m-%d")
        
        # Persist to database
        self._persist_series(final_series_df)
        
        # Derive events
        events_df = self._derive_events(final_series_df)
        self._persist_events(events_df)
        
    def _persist_series(self, df: pd.DataFrame) -> None:
        """Write to macro_series table."""
        self.db.execute("DELETE FROM macro_series")
        self.db.write_dataframe(df, "macro_series", if_exists="append")
        logger.info(f"Persisted {len(df)} rows to macro_series.")
        
    def _derive_events(self, series_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms macro_series into the legacy macro_events structure by 
        calculating `previous` dynamically.
        """
        logger.info("Deriving macro_events from macro_series...")
        
        # Make a copy and parse dates for correct sorting
        df = series_df.copy()
        
        # We assume event_name is just series_id for compatibility
        df.rename(columns={"series_id": "event_name", "value": "actual", "release_date": "timestamp_utc"}, inplace=True)
        
        # Calculate `previous` by shifting within each event_name group
        df.sort_values(["event_name", "date"], inplace=True)
        df["previous"] = df.groupby("event_name")["actual"].shift(1)
        
        # Add nullable legacy columns
        df["forecast"] = None
        df["surprise"] = None
        df["country"] = "US" # All FRED series here are US
        df["importance"] = "high"
        df["scraper_version"] = "2.0"
        df["raw_event_json"] = None
        
        # Select target columns
        cols = [
            "timestamp_utc", "event_name", "country", "importance", 
            "actual", "forecast", "previous", "surprise", 
            "source", "scraper_version", "raw_event_json"
        ]
        
        return df[cols].reset_index(drop=True)

    def _persist_events(self, df: pd.DataFrame) -> None:
        """Write to macro_events table."""
        self.db.execute("DELETE FROM macro_events")
        self.db.write_dataframe(df, "macro_events", if_exists="append")
        logger.info(f"Persisted {len(df)} derived rows to macro_events.")
