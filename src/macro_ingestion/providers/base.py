"""
Abstract base class for all macroeconomic data providers.
"""
import pandas as pd
from abc import ABC, abstractmethod

class MacroProvider(ABC):
    """
    Abstract base class for macroeconomic data providers.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the provider."""
        pass
        
    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version of the provider implementation."""
        pass

    @abstractmethod
    def fetch_series(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch a macroeconomic time series.
        
        Args:
            series_id: The provider-specific ID for the time series.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            
        Returns:
            A pandas DataFrame with standard columns:
            ['series_id', 'date', 'value', 'release_date', 'frequency', 'source']
        """
        pass
