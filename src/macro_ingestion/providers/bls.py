import pandas as pd
from src.macro_ingestion.providers.base import MacroProvider

class BLSProvider(MacroProvider):
    """
    Placeholder for Bureau of Labor Statistics (BLS) API.
    """
    
    @property
    def name(self) -> str:
        return "BLS"
        
    @property
    def version(self) -> str:
        return "0.0.0"

    def fetch_series(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError("BLS Provider is not yet implemented. Use FRED as a proxy for BLS series.")
