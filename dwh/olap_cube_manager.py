"""
OLAP Cube Manager for Analytics
Author: Member 3
"""

import pandas as pd
from database.advanced_models import get_advanced_db_manager
from typing import Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class OLAPCubeManager:
    """
    Manages the creation and aggregation of OLAP cubes for analytical queries.
    """

    def __init__(self):
        """Initializes the OLAPCubeManager with a database connection."""
        self.db = get_advanced_db_manager()
        logger.info("OLAPCubeManager initialized.")

    def get_cube(self, dimensions: List[str], metrics: List[str], time_range: Dict[str, Any]) -> pd.DataFrame:
        """
        Retrieves raw data and forms a base DataFrame (cube).
        
        Args:
            dimensions (List[str]): A list of column names to serve as dimensions.
            metrics (List[str]): A list of column names for the metrics (values).
            time_range (Dict[str, Any]): A dictionary specifying the time range,
                                         e.g., {'start_date': 'YYYY-MM-DD', 'end_date': 'YYYY-MM-DD'}.

        Returns:
            pd.DataFrame: A DataFrame representing the OLAP cube.
        """
        # The original code had a missing comma and improper alignment.
        data = self.db.get_olap_data(dimensions, metrics, time_range)
        
        if not data:
            logger.warning("No data returned from the database. Returning an empty DataFrame.")
            return pd.DataFrame()
            
        cube = pd.DataFrame(data)
        
        # Ensure that the DataFrame contains the specified dimensions and metrics
        required_cols = dimensions + metrics
        if not all(col in cube.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in cube.columns]
            logger.error(f"Missing required columns in the returned data: {missing_cols}")
            return pd.DataFrame()

        return cube

    def aggregate_cube(self, cube: pd.DataFrame, by: List[str], metrics: List[str]) -> pd.DataFrame:
        """
        Aggregates the OLAP cube data by specified dimensions.
        
        Args:
            cube (pd.DataFrame): The base DataFrame (cube) to be aggregated.
            by (List[str]): A list of column names to group the data by.
            metrics (List[str]): A list of column names to apply aggregations to.
            
        Returns:
            pd.DataFrame: A new DataFrame with aggregated data.
        """
        if cube.empty:
            logger.warning("Input cube is empty. Cannot perform aggregation.")
            return pd.DataFrame()

        if not all(col in cube.columns for col in by):
            missing_dims = [dim for dim in by if dim not in cube.columns]
            logger.error(f"Grouping dimensions not found in cube: {missing_dims}")
            return pd.DataFrame()
            
        if not all(col in cube.columns for col in metrics):
            missing_metrics = [metric for metric in metrics if metric not in cube.columns]
            logger.error(f"Metrics not found in cube: {missing_metrics}")
            return pd.DataFrame()
            
        # Create the aggregation dictionary dynamically
        agg_dict = {m: ['mean', 'max', 'min'] for m in metrics}
        
        # Perform the aggregation
        try:
            aggregated_data = cube.groupby(by).agg(agg_dict)
            return aggregated_data
        except Exception as e:
            logger.error(f"An error occurred during aggregation: {e}")
            return pd.DataFrame()
