"""
 Historic Analytics & Rolling Metrics
 Author: Member 3
 """

import pandas as pd
import numpy as np
from database.advanced_models import get_advanced_db_manager
from datetime import datetime, timedelta
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

logger = logging.getLogger(__name__)

class HistoricAnalyzer:
    """Time-series analytics for system monitoring"""

    def __init__(self):
        self.db_manager = get_advanced_db_manager()
        logger.info("HistoricAnalyzer initialized")
        # Ensure the output directory exists
        if not os.path.exists('output/plots'):
            os.makedirs('output/plots')

    def load_time_series(self, metric_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load time series data for a metric"""
        data = self.db_manager.get_historical_metrics(metric_name, start_date=start_date, end_date=end_date)
        df = pd.DataFrame(data)
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Set the timestamp column as the index for better time-series handling
            df = df.set_index('timestamp')
        return df

    def compute_rolling_agg(self, df: pd.DataFrame, value_col: str, window: int = 30) -> pd.DataFrame:
        """Compute rolling mean and std for a given column"""
        if df.empty or value_col not in df.columns:
            logger.warning(f"DataFrame is empty or '{value_col}' not found. Skipping rolling aggregation.")
            return df

        df = df.copy()
        df[f'{value_col}_rolling_mean'] = df[value_col].rolling(window).mean()
        df[f'{value_col}_rolling_std'] = df[value_col].rolling(window).std()
        return df

    def detect_outliers(self, df: pd.DataFrame, value_col: str, threshold: float = 3.0) -> pd.DataFrame:
        """Outlier detection based on Z-score"""
        if df.empty or value_col not in df.columns:
            logger.warning(f"DataFrame is empty or '{value_col}' not found. Skipping outlier detection.")
            return df
        
        df = df.copy()
        mean = df[value_col].mean()
        std = df[value_col].std()
        
        # Avoid division by zero if standard deviation is zero
        if std == 0:
            logger.info("Standard deviation is zero, no outliers detected based on Z-score.")
            df['z_score'] = 0
            df['is_outlier'] = False
        else:
            df['z_score'] = (df[value_col] - mean) / std
            df['is_outlier'] = np.abs(df['z_score']) > threshold
            
        return df

    def create_summary_stats(self, df: pd.DataFrame, value_col: str) -> Dict[str, Any]:
        """Create a dictionary of summary statistics for a given column."""
        if df.empty or value_col not in df.columns:
            logger.warning(f"DataFrame is empty or '{value_col}' not found. Returning empty summary stats.")
            return {}

        return {
            'mean': df[value_col].mean(),
            'std': df[value_col].std(),
            'min': df[value_col].min(),
            'max': df[value_col].max(),
            'percentile_95': np.percentile(df[value_col], 95) if not df[value_col].isnull().all() else np.nan,
            'percentile_99': np.percentile(df[value_col], 99) if not df[value_col].isnull().all() else np.nan
        }

    def plot_metric_trend(self, df: pd.DataFrame, value_col: str, label: str):
        """Plot the trend of a metric and save the figure."""
        if df.empty or value_col not in df.columns:
            logger.warning(f"DataFrame is empty or '{value_col}' not found. Cannot plot trend.")
            return
            
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x=df.index, y=value_col, label=label)
        plt.title(f'Trend of {label}')
        plt.xlabel('Timestamp')
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        file_path = f'output/plots/{label}_trend.png'
        try:
            plt.savefig(file_path)
            logger.info(f"Plot saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
        finally:
            plt.close() # Close the plot to free up memory
