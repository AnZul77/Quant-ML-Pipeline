import pandas as pd
from datetime import datetime, timezone

from src.macro_ingestion.providers.base import MacroProvider
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FredProvider(MacroProvider):
    """
    Fetches macroeconomic data from the Federal Reserve Economic Data (FRED) API.
    Utilizes the public CSV export endpoint to avoid API key requirements.
    """
    
    @property
    def name(self) -> str:
        return "FRED"
        
    @property
    def version(self) -> str:
        return "1.0.0"

    def fetch_series(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch a series from FRED's public CSV endpoint.
        """
        logger.info(f"Fetching {series_id} from FRED ({start_date} to {end_date})")
        
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        
        try:
            df = pd.read_csv(url, na_values=".")
            df.columns = ["date", "value"]
            
            # Clean and filter dates
            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            df = df[mask].copy()
            
            # Drop NaN values
            df = df.dropna(subset=["value"])
            
            # Reconstruct standardized DataFrame
            df["series_id"] = series_id
            df["source"] = self.name
            df["frequency"] = "monthly"  # Ideally queried, but we assume monthly for most macro indicators here.
            
            # Mocking the exact release date: FRED data typically becomes available 
            # some time after the observation date. We will simulate a release date
            # realistically (e.g., end of the month or 15 days after) to prevent look-ahead bias,
            # but ideally this comes from a true release calendar. Since we must enforce
            # strict backward joining without leakage, we will conservatively set release_date 
            # to exactly 30 days after the observation date at 08:30 AM UTC.
            df["release_date"] = (df["date"] + pd.DateOffset(days=30)).dt.strftime("%Y-%m-%d 08:30:00")
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            
            # Format
            cols = ["series_id", "date", "value", "release_date", "frequency", "source"]
            df = df[cols].reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} records for {series_id}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {series_id} from FRED: {e}")
            return pd.DataFrame(columns=["series_id", "date", "value", "release_date", "frequency", "source"])
