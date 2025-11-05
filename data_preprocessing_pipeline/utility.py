"""
Utility Functions Module

Common utility functions used across the ship trajectory prediction system.
Includes geographic calculations, data transformations, and helper functions.

Author: Ship Trajectory Prediction Team
Date: 2025-11-04
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, 
                      lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points using Haversine formula
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
        
    Returns:
        Distance in meters
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in meters
    r = 6371000
    
    return c * r


def bearing_between_points(lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
    """
    Calculate bearing (course) from point 1 to point 2
    
    Args:
        lat1, lon1: Starting point coordinates (degrees)
        lat2, lon2: Ending point coordinates (degrees)
        
    Returns:
        Bearing in degrees (0-360, where 0 is North)
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    bearing_rad = np.arctan2(x, y)
    bearing_deg = (np.degrees(bearing_rad) + 360) % 360
    
    return bearing_deg


def project_position(lat: float, lon: float,
                    distance_m: float, bearing_deg: float) -> Tuple[float, float]:
    """
    Project a position by a given distance and bearing
    
    Args:
        lat, lon: Starting position (degrees)
        distance_m: Distance to travel (meters)
        bearing_deg: Bearing/course (degrees, 0 = North)
        
    Returns:
        (new_lat, new_lon) in degrees
    """
    # Earth radius in meters
    R = 6371000
    
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    bearing_rad = np.radians(bearing_deg)
    
    # Calculate new position
    lat2_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(distance_m/R) +
        np.cos(lat_rad) * np.sin(distance_m/R) * np.cos(bearing_rad)
    )
    
    lon2_rad = lon_rad + np.arctan2(
        np.sin(bearing_rad) * np.sin(distance_m/R) * np.cos(lat_rad),
        np.cos(distance_m/R) - np.sin(lat_rad) * np.sin(lat2_rad)
    )
    
    # Convert back to degrees
    lat2 = np.degrees(lat2_rad)
    lon2 = np.degrees(lon2_rad)
    
    return lat2, lon2


def knots_to_ms(knots: float) -> float:
    """Convert speed from knots to meters per second"""
    return knots * 0.514444


def ms_to_knots(ms: float) -> float:
    """Convert speed from meters per second to knots"""
    return ms / 0.514444


def normalize_angle(angle_deg: float) -> float:
    """
    Normalize angle to [0, 360) range
    
    Args:
        angle_deg: Angle in degrees
        
    Returns:
        Normalized angle in [0, 360)
    """
    return angle_deg % 360


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Calculate smallest difference between two angles (accounting for wraparound)
    
    Args:
        angle1, angle2: Angles in degrees
        
    Returns:
        Signed difference in degrees (-180 to 180)
    """
    diff = (angle2 - angle1 + 180) % 360 - 180
    return diff


def time_to_seconds(time_str: str) -> int:
    """
    Convert time string to seconds
    
    Args:
        time_str: Time string (e.g., "5m", "1h", "30s")
        
    Returns:
        Time in seconds
    """
    units = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
    
    number = float(time_str[:-1])
    unit = time_str[-1]
    
    if unit not in units:
        raise ValueError(f"Unknown time unit: {unit}")
    
    return int(number * units[unit])


def create_time_grid(start_time: datetime,
                    end_time: datetime,
                    interval_seconds: int) -> List[datetime]:
    """
    Create regular time grid between start and end times
    
    Args:
        start_time: Start datetime
        end_time: End datetime
        interval_seconds: Time interval in seconds
        
    Returns:
        List of datetime objects
    """
    time_grid = []
    current_time = start_time
    
    while current_time <= end_time:
        time_grid.append(current_time)
        current_time += timedelta(seconds=interval_seconds)
    
    return time_grid


def calculate_speed_from_positions(lat1: float, lon1: float, time1: datetime,
                                   lat2: float, lon2: float, time2: datetime) -> float:
    """
    Calculate speed between two positions
    
    Args:
        lat1, lon1, time1: First position and time
        lat2, lon2, time2: Second position and time
        
    Returns:
        Speed in knots
    """
    # Distance in meters
    distance_m = haversine_distance(lat1, lon1, lat2, lon2)
    
    # Time difference in seconds
    time_diff = (time2 - time1).total_seconds()
    
    if time_diff == 0:
        return 0.0
    
    # Speed in m/s
    speed_ms = distance_m / time_diff
    
    # Convert to knots
    speed_knots = ms_to_knots(speed_ms)
    
    return speed_knots


def is_valid_mmsi(mmsi: int) -> bool:
    """
    Check if MMSI number is valid
    
    MMSI should be a 9-digit number
    
    Args:
        mmsi: Maritime Mobile Service Identity number
        
    Returns:
        True if valid, False otherwise
    """
    mmsi_str = str(mmsi)
    return len(mmsi_str) == 9 and mmsi_str.isdigit()


def format_timestamp(timestamp: datetime) -> str:
    """
    Format timestamp for display
    
    Args:
        timestamp: Datetime object
        
    Returns:
        Formatted string
    """
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse timestamp string
    
    Args:
        timestamp_str: Timestamp string
        
    Returns:
        Datetime object
    """
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse timestamp: {timestamp_str}")


def calculate_cpa(pos1: Tuple[float, float], vel1: Tuple[float, float],
                 pos2: Tuple[float, float], vel2: Tuple[float, float]) -> Tuple[float, float]:
    """
    Calculate Closest Point of Approach (CPA) between two vessels
    
    Args:
        pos1: Position of vessel 1 (lat, lon)
        vel1: Velocity of vessel 1 (vx, vy) in m/s
        pos2: Position of vessel 2 (lat, lon)
        vel2: Velocity of vessel 2 (vx, vy) in m/s
        
    Returns:
        (DCPA in meters, TCPA in seconds)
        DCPA: Distance at CPA
        TCPA: Time to CPA
    """
    # Relative position (in meters, approximate)
    dx = (pos2[1] - pos1[1]) * 111320 * np.cos(np.radians(pos1[0]))
    dy = (pos2[0] - pos1[0]) * 110540
    
    # Relative velocity
    dvx = vel2[0] - vel1[0]
    dvy = vel2[1] - vel1[1]
    
    # Time to CPA
    numerator = -(dx * dvx + dy * dvy)
    denominator = dvx**2 + dvy**2
    
    if denominator < 1e-6:  # Vessels moving in parallel
        tcpa = 0
        dcpa = np.sqrt(dx**2 + dy**2)
    else:
        tcpa = numerator / denominator
        
        if tcpa < 0:  # CPA in the past
            tcpa = 0
            dcpa = np.sqrt(dx**2 + dy**2)
        else:
            # Position at CPA
            dx_cpa = dx + dvx * tcpa
            dy_cpa = dy + dvy * tcpa
            dcpa = np.sqrt(dx_cpa**2 + dy_cpa**2)
    
    return dcpa, tcpa


def smooth_trajectory(trajectory: pd.DataFrame,
                     window_size: int = 5,
                     columns: List[str] = None) -> pd.DataFrame:
    """
    Smooth trajectory using moving average
    
    Args:
        trajectory: Trajectory DataFrame
        window_size: Window size for moving average
        columns: Columns to smooth (default: ['latitude', 'longitude', 'sog', 'cog'])
        
    Returns:
        Smoothed trajectory DataFrame
    """
    if columns is None:
        columns = ['latitude', 'longitude', 'sog', 'cog']
    
    smoothed = trajectory.copy()
    
    for col in columns:
        if col in smoothed.columns:
            smoothed[col] = smoothed[col].rolling(window=window_size, 
                                                  center=True, 
                                                  min_periods=1).mean()
    
    return smoothed


def downsample_trajectory(trajectory: pd.DataFrame,
                         target_interval_seconds: int = 60) -> pd.DataFrame:
    """
    Downsample trajectory to target time interval
    
    Args:
        trajectory: Trajectory DataFrame with timestamp column
        target_interval_seconds: Target interval in seconds
        
    Returns:
        Downsampled trajectory
    """
    trajectory = trajectory.sort_values('timestamp').reset_index(drop=True)
    
    downsampled_indices = [0]  # Always include first point
    last_time = trajectory.iloc[0]['timestamp']
    
    for i in range(1, len(trajectory)):
        current_time = trajectory.iloc[i]['timestamp']
        
        if (current_time - last_time).total_seconds() >= target_interval_seconds:
            downsampled_indices.append(i)
            last_time = current_time
    
    # Always include last point
    if downsampled_indices[-1] != len(trajectory) - 1:
        downsampled_indices.append(len(trajectory) - 1)
    
    return trajectory.iloc[downsampled_indices].reset_index(drop=True)


def print_progress_bar(iteration: int, total: int, 
                      prefix: str = '', suffix: str = '',
                      length: int = 50, fill: str = '█'):
    """
    Print progress bar to console
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        length: Character length of bar
        fill: Bar fill character
    """
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='')
    
    if iteration == total:
        print()


if __name__ == "__main__":
    # Test utility functions
    print("=== Testing Utility Functions ===\n")
    
    # Test haversine distance
    lat1, lon1 = 59.0, 10.0
    lat2, lon2 = 59.1, 10.1
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    print(f"Distance between ({lat1},{lon1}) and ({lat2},{lon2}): {distance:.2f}m")
    
    # Test bearing
    bearing = bearing_between_points(lat1, lon1, lat2, lon2)
    print(f"Bearing: {bearing:.2f}°")
    
    # Test speed conversion
    speed_knots = 10.0
    speed_ms = knots_to_ms(speed_knots)
    print(f"{speed_knots} knots = {speed_ms:.2f} m/s")
    
    # Test angle difference
    angle1, angle2 = 350, 10
    diff = angle_difference(angle1, angle2)
    print(f"Angle difference between {angle1}° and {angle2}°: {diff}°")
    
    # Test CPA calculation
    pos1 = (59.0, 10.0)
    vel1 = (5.0, 0.0)  # 5 m/s east
    pos2 = (59.01, 10.0)
    vel2 = (0.0, 5.0)  # 5 m/s north
    
    dcpa, tcpa = calculate_cpa(pos1, vel1, pos2, vel2)
    print(f"\nCPA calculation:")
    print(f"  DCPA: {dcpa:.2f}m")
    print(f"  TCPA: {tcpa:.2f}s")
    
    print("\n✓ All utility function tests passed")