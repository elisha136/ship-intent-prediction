"""
AIS Data Preprocessing Module

This module handles trajectory segmentation, feature engineering, and dataset
preparation for trajectory prediction models. It transforms cleaned AIS data
into structured trajectory datasets suitable for machine learning.

Key Features:
- Trajectory segmentation based on time gaps
- Feature engineering (velocity, acceleration, turn rate)
- Temporal resampling and interpolation
- Train/validation/test split generation

Author: Ship Trajectory Prediction Team
Date: 2025-11-04
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from datetime import timedelta
from scipy.interpolate import interp1d

from config.config import config

logger = logging.getLogger(__name__)


class TrajectorySegmenter:
    """
    Segments continuous AIS data into discrete trajectories based on
    temporal gaps and quality criteria.
    """
    
    def __init__(self):
        """Initialize trajectory segmenter"""
        self.config = config.cleaning
        logger.info("Trajectory Segmenter initialized")
    
    def segment_trajectories(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Segment AIS data into individual trajectories
        
        A new trajectory is created when:
        1. Time gap exceeds threshold
        2. Vessel changes (different MMSI)
        
        Args:
            df: Cleaned AIS DataFrame sorted by (mmsi, timestamp)
            
        Returns:
            List of trajectory DataFrames
        """
        logger.info(f"Segmenting {len(df)} records into trajectories...")
        
        # Ensure data is sorted
        df = df.sort_values(['mmsi', 'timestamp']).reset_index(drop=True)
        
        # Calculate time differences
        df['time_diff'] = df.groupby('mmsi')['timestamp'].diff()
        
        # Identify trajectory breaks
        df['is_new_trajectory'] = (
            (df['time_diff'].isna()) |  # First record for vessel
            (df['time_diff'] > timedelta(seconds=self.config.max_time_gap_seconds)) |
            (df['mmsi'] != df['mmsi'].shift(1))  # Vessel change
        )
        
        # Assign trajectory IDs
        df['trajectory_id'] = df['is_new_trajectory'].cumsum()
        
        # Split into individual trajectories
        trajectories = []
        for traj_id, traj_df in df.groupby('trajectory_id'):
            # Apply quality filters
            if self._is_valid_trajectory(traj_df):
                # Clean up helper columns
                traj_clean = traj_df.drop(columns=['time_diff', 'is_new_trajectory', 
                                                   'trajectory_id']).reset_index(drop=True)
                trajectories.append(traj_clean)
        
        logger.info(f"Created {len(trajectories)} valid trajectories")
        
        # Print statistics
        self._print_trajectory_statistics(trajectories)
        
        return trajectories
    
    def _is_valid_trajectory(self, traj_df: pd.DataFrame) -> bool:
        """
        Check if trajectory meets quality criteria
        
        Args:
            traj_df: Trajectory DataFrame
            
        Returns:
            Boolean indicating validity
        """
        # Check minimum number of points
        if len(traj_df) < self.config.min_trajectory_points:
            return False
        
        # Check minimum duration
        duration = (traj_df['timestamp'].max() - 
                   traj_df['timestamp'].min()).total_seconds()
        if duration < self.config.min_trajectory_duration_seconds:
            return False
        
        return True
    
    def _print_trajectory_statistics(self, trajectories: List[pd.DataFrame]):
        """Print statistics about segmented trajectories"""
        if not trajectories:
            return
        
        lengths = [len(t) for t in trajectories]
        durations = [(t['timestamp'].max() - t['timestamp'].min()).total_seconds() / 60 
                     for t in trajectories]
        vessels = set([t['mmsi'].iloc[0] for t in trajectories])
        
        print("\n" + "="*60)
        print("TRAJECTORY SEGMENTATION STATISTICS")
        print("="*60)
        print(f"Total trajectories:        {len(trajectories)}")
        print(f"Unique vessels:            {len(vessels)}")
        print(f"\nPoints per trajectory:")
        print(f"  Mean:                    {np.mean(lengths):.1f}")
        print(f"  Median:                  {np.median(lengths):.1f}")
        print(f"  Min:                     {np.min(lengths)}")
        print(f"  Max:                     {np.max(lengths)}")
        print(f"\nDuration (minutes):")
        print(f"  Mean:                    {np.mean(durations):.1f}")
        print(f"  Median:                  {np.median(durations):.1f}")
        print(f"  Min:                     {np.min(durations):.1f}")
        print(f"  Max:                     {np.max(durations):.1f}")
        print("="*60 + "\n")


class FeatureEngineering:
    """
    Feature engineering for AIS trajectory data.
    Creates derived features useful for prediction models.
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.config = config.preprocessing
        logger.info("Feature Engineering initialized")
    
    def add_features(self, traj_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to trajectory
        
        Features include:
        - Velocity changes (acceleration)
        - Course rate (turn rate)
        - Distance traveled
        - Temporal features
        
        Args:
            traj_df: Trajectory DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = traj_df.copy()
        
        # Temporal features
        df = self._add_temporal_features(df)
        
        # Motion features
        df = self._add_motion_features(df)
        
        # Spatial features
        df = self._add_spatial_features(df)
        
        logger.debug(f"Added features. New shape: {df.shape}")
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['time_diff_seconds'] = df['timestamp'].diff().dt.total_seconds()
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['time_since_start'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        
        return df
    
    def _add_motion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add motion-related features"""
        # Speed changes (acceleration)
        df['speed_change'] = df['sog'].diff()
        df['acceleration'] = df['speed_change'] / df['time_diff_seconds']
        
        # Course changes (turn rate)
        df['course_change'] = self._calculate_course_change(df['cog'])
        df['turn_rate'] = df['course_change'] / df['time_diff_seconds']
        
        # Moving averages for smoothing
        for window_size in [3, 5, 10]:
            df[f'sog_ma_{window_size}'] = df['sog'].rolling(window=window_size, 
                                                            min_periods=1).mean()
            df[f'cog_ma_{window_size}'] = df['cog'].rolling(window=window_size, 
                                                            min_periods=1).mean()
        
        return df
    
    def _add_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spatial features"""
        # Distance traveled between points
        df['distance_meters'] = self._calculate_distance(df)
        df['cumulative_distance'] = df['distance_meters'].cumsum()
        
        # Velocity components (useful for prediction)
        df['velocity_east'] = df['sog'] * np.sin(np.radians(df['cog']))
        df['velocity_north'] = df['sog'] * np.cos(np.radians(df['cog']))
        
        return df
    
    def _calculate_distance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate distance between consecutive points"""
        prev_lat = df['latitude'].shift(1)
        prev_lon = df['longitude'].shift(1)
        
        # Simplified haversine
        lat1 = np.radians(prev_lat)
        lon1 = np.radians(prev_lon)
        lat2 = np.radians(df['latitude'])
        lon2 = np.radians(df['longitude'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        distance = 6371000 * c  # Earth radius in meters
        
        return distance
    
    def _calculate_course_change(self, course_series: pd.Series) -> pd.Series:
        """Calculate course change accounting for circular angles"""
        course_diff = course_series.diff()
        
        # Handle wraparound (e.g., 350° to 10° is 20°, not -340°)
        course_change = np.where(
            course_diff.abs() > 180,
            360 - course_diff.abs(),
            course_diff.abs()
        )
        
        # Preserve sign
        course_change = np.where(
            course_diff < 0,
            -course_change,
            course_change
        )
        
        return pd.Series(course_change, index=course_series.index)
    
    def resample_trajectory(self, 
                           traj_df: pd.DataFrame,
                           interval_seconds: int = 60) -> pd.DataFrame:
        """
        Resample trajectory to regular time intervals using interpolation
        
        Args:
            traj_df: Trajectory DataFrame
            interval_seconds: Target sampling interval
            
        Returns:
            Resampled trajectory DataFrame
        """
        # Create target time grid
        start_time = traj_df['timestamp'].iloc[0]
        end_time = traj_df['timestamp'].iloc[-1]
        
        target_times = pd.date_range(start=start_time, end=end_time, 
                                     freq=f'{interval_seconds}s')
        
        # Convert timestamps to seconds for interpolation
        time_seconds = (traj_df['timestamp'] - start_time).dt.total_seconds().values
        target_seconds = (target_times - start_time).total_seconds().values
        
        # Interpolate each field
        resampled_data = {'timestamp': target_times}
        
        for col in ['latitude', 'longitude', 'sog', 'cog']:
            if col in traj_df.columns:
                # Use linear interpolation
                f = interp1d(time_seconds, traj_df[col].values, 
                           kind='linear', fill_value='extrapolate')
                resampled_data[col] = f(target_seconds)
        
        # Copy non-interpolated fields
        resampled_data['mmsi'] = traj_df['mmsi'].iloc[0]
        
        df_resampled = pd.DataFrame(resampled_data)
        
        logger.debug(f"Resampled trajectory: {len(traj_df)} -> {len(df_resampled)} points")
        
        return df_resampled


class DatasetSplitter:
    """
    Split trajectory dataset into train/validation/test sets
    with proper temporal ordering.
    """
    
    def __init__(self):
        """Initialize dataset splitter"""
        self.config = config.preprocessing
        logger.info("Dataset Splitter initialized")
    
    def split_trajectories(self, 
                          trajectories: List[pd.DataFrame],
                          split_by: str = 'temporal') -> Dict[str, List[pd.DataFrame]]:
        """
        Split trajectories into train/validation/test sets
        
        Args:
            trajectories: List of trajectory DataFrames
            split_by: 'temporal' or 'random'
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        if split_by == 'temporal':
            return self._temporal_split(trajectories)
        elif split_by == 'random':
            return self._random_split(trajectories)
        else:
            raise ValueError(f"Unknown split method: {split_by}")
    
    def _temporal_split(self, 
                       trajectories: List[pd.DataFrame]) -> Dict[str, List[pd.DataFrame]]:
        """
        Split based on temporal order (earliest -> train, latest -> test)
        
        This ensures no data leakage from future to past.
        """
        logger.info("Performing temporal split...")
        
        # Sort trajectories by start time
        sorted_trajs = sorted(trajectories, 
                            key=lambda t: t['timestamp'].iloc[0])
        
        n_total = len(sorted_trajs)
        n_train = int(n_total * self.config.train_split)
        n_val = int(n_total * self.config.validation_split)
        
        train_trajs = sorted_trajs[:n_train]
        val_trajs = sorted_trajs[n_train:n_train + n_val]
        test_trajs = sorted_trajs[n_train + n_val:]
        
        logger.info(f"Split: {len(train_trajs)} train, {len(val_trajs)} val, "
                   f"{len(test_trajs)} test trajectories")
        
        return {
            'train': train_trajs,
            'val': val_trajs,
            'test': test_trajs
        }
    
    def _random_split(self, 
                     trajectories: List[pd.DataFrame]) -> Dict[str, List[pd.DataFrame]]:
        """Random split (with fixed seed for reproducibility)"""
        logger.info("Performing random split...")
        
        np.random.seed(42)
        shuffled = trajectories.copy()
        np.random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(n_total * self.config.train_split)
        n_val = int(n_total * self.config.validation_split)
        
        train_trajs = shuffled[:n_train]
        val_trajs = shuffled[n_train:n_train + n_val]
        test_trajs = shuffled[n_train + n_val:]
        
        logger.info(f"Split: {len(train_trajs)} train, {len(val_trajs)} val, "
                   f"{len(test_trajs)} test trajectories")
        
        return {
            'train': train_trajs,
            'val': val_trajs,
            'test': test_trajs
        }


def test_preprocessing():
    """Test preprocessing functionality"""
    logger.info("Testing preprocessing module...")
    
    # Create sample trajectory
    sample_data = {
        'mmsi': [123456789] * 20,
        'latitude': np.linspace(59.0, 59.2, 20),
        'longitude': np.linspace(10.0, 10.2, 20),
        'sog': np.random.uniform(8, 12, 20),
        'cog': np.linspace(45, 55, 20),
        'timestamp': pd.date_range('2025-01-01', periods=20, freq='2min')
    }
    
    traj_df = pd.DataFrame(sample_data)
    
    print("\n=== Original Trajectory ===")
    print(traj_df.head())
    print(f"Shape: {traj_df.shape}")
    
    # Feature engineering
    fe = FeatureEngineering()
    traj_features = fe.add_features(traj_df)
    
    print("\n=== Trajectory with Features ===")
    print(traj_features.columns.tolist())
    print(traj_features[['latitude', 'longitude', 'sog', 'acceleration', 
                         'turn_rate']].head())
    
    # Resampling
    traj_resampled = fe.resample_trajectory(traj_df, interval_seconds=60)
    print("\n=== Resampled Trajectory ===")
    print(f"Original points: {len(traj_df)}, Resampled points: {len(traj_resampled)}")
    print(traj_resampled.head())
    
    logger.info("Preprocessing test complete")


if __name__ == "__main__":
    test_preprocessing()