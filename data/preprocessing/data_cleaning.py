"""
AIS Data Cleaning Module

This module provides comprehensive data cleaning and validation functions
for AIS (Automatic Identification System) data. It handles common issues such as:
- Invalid coordinates and sensor readings
- Duplicate records
- Outliers and noise
- Missing values
- Impossible position jumps

The cleaning process is designed to be robust while preserving genuine
vessel behavior patterns.

Author: Ship Trajectory Prediction Team
Date: 2025-11-04
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List
from datetime import timedelta
from geopy.distance import geodesic

from config.config import config

logger = logging.getLogger(__name__)


class AISDataCleaner:
    """
    Comprehensive AIS data cleaning and validation pipeline.
    Applies multiple cleaning steps in sequence to ensure data quality.
    """
    
    def __init__(self):
        """Initialize data cleaner with configuration parameters"""
        self.config = config.cleaning
        logger.info("AIS Data Cleaner initialized")
    
    def clean_dataset(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Apply complete cleaning pipeline to AIS dataset
        
        Args:
            df: Input DataFrame with AIS data
            verbose: Print detailed cleaning statistics
            
        Returns:
            Cleaned DataFrame
        """
        initial_count = len(df)
        logger.info(f"Starting data cleaning pipeline with {initial_count} records")
        
        # Create copy to avoid modifying original
        df_clean = df.copy()
        
        # Step 1: Remove duplicates
        df_clean = self.remove_duplicates(df_clean)
        
        # Step 2: Validate and clean basic fields
        df_clean = self.validate_coordinates(df_clean)
        df_clean = self.validate_speed(df_clean)
        df_clean = self.validate_course(df_clean)
        
        # Step 3: Handle missing values
        df_clean = self.handle_missing_values(df_clean)
        
        # Step 4: Sort by vessel and time
        df_clean = df_clean.sort_values(['mmsi', 'timestamp']).reset_index(drop=True)
        
        # Step 5: Remove impossible position jumps
        df_clean = self.remove_position_outliers(df_clean)
        
        # Step 6: Remove impossible speed/course changes
        df_clean = self.remove_motion_outliers(df_clean)
        
        final_count = len(df_clean)
        removed_count = initial_count - final_count
        removal_percentage = (removed_count / initial_count) * 100
        
        logger.info(f"Cleaning complete: {final_count} records remaining "
                   f"({removed_count} removed, {removal_percentage:.2f}%)")
        
        if verbose:
            self._print_cleaning_summary(initial_count, final_count, removed_count)
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate AIS records
        
        Duplicates are identified by: (mmsi, timestamp, latitude, longitude)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame without duplicates
        """
        before = len(df)
        df = df.drop_duplicates(subset=['mmsi', 'timestamp', 'latitude', 'longitude'])
        after = len(df)
        
        logger.info(f"Removed {before - after} duplicate records")
        return df
    
    def validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean latitude/longitude values
        
        Removes records with:
        - Coordinates outside valid ranges
        - Null coordinates
        - Coordinates at (0, 0) which often indicate errors
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with valid coordinates
        """
        before = len(df)
        
        # Remove null coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Validate latitude range
        df = df[
            (df['latitude'] >= self.config.min_latitude) &
            (df['latitude'] <= self.config.max_latitude)
        ]
        
        # Validate longitude range
        df = df[
            (df['longitude'] >= self.config.min_longitude) &
            (df['longitude'] <= self.config.max_longitude)
        ]
        
        # Remove suspicious (0, 0) coordinates
        df = df[~((df['latitude'] == 0) & (df['longitude'] == 0))]
        
        after = len(df)
        logger.info(f"Removed {before - after} records with invalid coordinates")
        
        return df
    
    def validate_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate Speed Over Ground (SOG) values
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with valid speed values
        """
        before = len(df)
        
        # Replace invalid SOG with NaN (will be handled later)
        df.loc[
            (df['sog'] < self.config.min_sog) | 
            (df['sog'] > self.config.max_sog),
            'sog'
        ] = np.nan
        
        after = len(df[df['sog'].notna()])
        logger.info(f"Marked {before - after} records with invalid SOG as missing")
        
        return df
    
    def validate_course(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate Course Over Ground (COG) values
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with valid course values
        """
        before = len(df)
        
        # Replace invalid COG with NaN
        df.loc[
            (df['cog'] < self.config.min_cog) | 
            (df['cog'] > self.config.max_cog),
            'cog'
        ] = np.nan
        
        # Handle COG = 360 (should be 0)
        df.loc[df['cog'] == 360.0, 'cog'] = 0.0
        
        after = len(df[df['cog'].notna()])
        logger.info(f"Marked {before - after} records with invalid COG as missing")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in AIS data
        
        Strategy:
        - Coordinates: Remove records (critical field)
        - SOG/COG: Interpolate within vessel trajectories if gap is small
        - Timestamp: Remove records (critical field)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        before = len(df)
        
        # Remove records with missing timestamps or coordinates (critical)
        df = df.dropna(subset=['timestamp', 'latitude', 'longitude'])
        
        # Interpolate SOG and COG within each vessel's trajectory
        df = df.sort_values(['mmsi', 'timestamp'])
        
        # Group by vessel and interpolate
        for col in ['sog', 'cog']:
            if col in df.columns:
                df[col] = df.groupby('mmsi')[col].transform(
                    lambda x: x.interpolate(method='linear', limit=3)
                )
        
        # Remove records still having missing SOG or COG after interpolation
        df = df.dropna(subset=['sog', 'cog'])
        
        after = len(df)
        logger.info(f"Removed {before - after} records due to missing critical values")
        
        return df
    
    def remove_position_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove records with impossible position jumps
        
        Calculates distance between consecutive positions for each vessel
        and removes points that indicate impossible movement speeds.
        
        Args:
            df: Input DataFrame (must be sorted by mmsi, timestamp)
            
        Returns:
            DataFrame without position outliers
        """
        before = len(df)
        
        # Calculate time difference and distance between consecutive points
        df['time_diff'] = df.groupby('mmsi')['timestamp'].diff().dt.total_seconds()
        
        # Calculate distance using vectorized haversine formula
        df['distance_meters'] = self._calculate_distance_vectorized(df)
        
        # Calculate implied speed (meters per second)
        df['implied_speed_mps'] = df['distance_meters'] / df['time_diff']
        
        # Convert to knots (1 m/s = 1.94384 knots)
        df['implied_speed_knots'] = df['implied_speed_mps'] * 1.94384
        
        # Remove points where implied speed exceeds maximum realistic speed
        # Using 2x max_sog as threshold to be conservative
        max_realistic_speed = self.config.max_sog * 2
        outlier_mask = (
            (df['time_diff'].notna()) & 
            (df['implied_speed_knots'] > max_realistic_speed)
        )
        
        df = df[~outlier_mask]
        
        # Clean up temporary columns
        df = df.drop(columns=['time_diff', 'distance_meters', 
                              'implied_speed_mps', 'implied_speed_knots'])
        
        after = len(df)
        logger.info(f"Removed {before - after} position outliers")
        
        return df
    
    def remove_motion_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove records with impossible speed or course changes
        
        Args:
            df: Input DataFrame (must be sorted by mmsi, timestamp)
            
        Returns:
            DataFrame without motion outliers
        """
        before = len(df)
        
        # Calculate speed change between consecutive points
        df['speed_change'] = df.groupby('mmsi')['sog'].diff().abs()
        
        # Calculate course change (accounting for circular nature of angles)
        df['course_change'] = df.groupby('mmsi')['cog'].transform(
            lambda x: self._calculate_course_change(x)
        )
        
        # Remove points with excessive changes
        outlier_mask = (
            (df['speed_change'] > self.config.max_speed_change_knots) |
            (df['course_change'] > self.config.max_course_change_degrees)
        )
        
        df = df[~outlier_mask]
        
        # Clean up temporary columns
        df = df.drop(columns=['speed_change', 'course_change'])
        
        after = len(df)
        logger.info(f"Removed {before - after} motion outliers")
        
        return df
    
    def _calculate_distance_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate distance between consecutive positions using vectorized operations
        
        Uses simplified haversine formula for reasonable accuracy and speed.
        
        Args:
            df: DataFrame with latitude, longitude, and mmsi columns
            
        Returns:
            Series with distances in meters
        """
        # Get previous positions within same vessel
        prev_lat = df.groupby('mmsi')['latitude'].shift(1)
        prev_lon = df.groupby('mmsi')['longitude'].shift(1)
        
        # Convert to radians
        lat1 = np.radians(prev_lat)
        lon1 = np.radians(prev_lon)
        lat2 = np.radians(df['latitude'])
        lon2 = np.radians(df['longitude'])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in meters
        earth_radius = 6371000
        distance = earth_radius * c
        
        return distance
    
    def _calculate_course_change(self, course_series: pd.Series) -> pd.Series:
        """
        Calculate course change accounting for circular nature of angles
        
        Args:
            course_series: Series of course values in degrees
            
        Returns:
            Series of course changes in degrees
        """
        course_diff = course_series.diff()
        
        # Handle circular nature (e.g., change from 350° to 10° is 20°, not 340°)
        course_change = np.minimum(
            course_diff.abs(),
            360 - course_diff.abs()
        )
        
        return course_change
    
    def filter_stationary_vessels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove or flag vessels that are stationary for extended periods
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with stationary periods handled
        """
        before = len(df)
        
        # Mark low-speed records
        df['is_stationary'] = df['sog'] < self.config.stationary_speed_threshold
        
        # Calculate duration of stationary periods
        df['stationary_group'] = (
            df.groupby('mmsi')['is_stationary']
            .transform(lambda x: (x != x.shift()).cumsum())
        )
        
        # Remove long stationary periods
        for mmsi in df['mmsi'].unique():
            vessel_df = df[df['mmsi'] == mmsi]
            
            for group_id in vessel_df['stationary_group'].unique():
                group_df = vessel_df[
                    (vessel_df['stationary_group'] == group_id) &
                    (vessel_df['is_stationary'] == True)
                ]
                
                if len(group_df) > 0:
                    duration = (group_df['timestamp'].max() - 
                               group_df['timestamp'].min()).total_seconds()
                    
                    if duration > self.config.stationary_duration_seconds:
                        # Remove this stationary period
                        df = df[~df.index.isin(group_df.index)]
        
        # Clean up temporary columns
        df = df.drop(columns=['is_stationary', 'stationary_group'])
        
        after = len(df)
        logger.info(f"Removed {before - after} records from extended stationary periods")
        
        return df
    
    def _print_cleaning_summary(self, initial: int, final: int, removed: int):
        """Print detailed cleaning summary"""
        print("\n" + "="*60)
        print("AIS DATA CLEANING SUMMARY")
        print("="*60)
        print(f"Initial records:     {initial:,}")
        print(f"Final records:       {final:,}")
        print(f"Removed records:     {removed:,}")
        print(f"Removal percentage:  {(removed/initial)*100:.2f}%")
        print(f"Retention rate:      {(final/initial)*100:.2f}%")
        print("="*60 + "\n")


def test_data_cleaning():
    """Test data cleaning functionality with sample data"""
    logger.info("Testing data cleaning module...")
    
    # Create sample data with various issues
    sample_data = {
        'mmsi': [123456789] * 10,
        'latitude': [59.0, 59.1, 59.2, 999.0, 59.4, 59.5, 59.5, 59.6, 59.7, 59.8],
        'longitude': [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.5, 10.6, 10.7, 10.8],
        'sog': [10.0, 10.5, 11.0, 150.0, 10.5, 11.0, 0.2, 11.5, 12.0, 12.5],
        'cog': [45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 50.0, 51.0, 52.0, 53.0],
        'timestamp': pd.date_range('2025-01-01', periods=10, freq='1min')
    }
    
    df = pd.DataFrame(sample_data)
    
    print("\n=== Original Data ===")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    # Clean data
    cleaner = AISDataCleaner()
    df_clean = cleaner.clean_dataset(df, verbose=True)
    
    print("\n=== Cleaned Data ===")
    print(df_clean)
    print(f"\nCleaned shape: {df_clean.shape}")
    
    logger.info("Data cleaning test complete")


if __name__ == "__main__":
    test_data_cleaning()