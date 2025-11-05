"""
Statistical Trajectory Prediction Models

This module implements baseline statistical methods for ship trajectory prediction:
1. Constant Velocity (CV) Model
2. Constant Turn Rate (CTR) Model
3. Kalman Filter with CV motion model

These models serve as baselines for comparison with machine learning approaches.
They are computationally efficient and provide reasonable predictions for
short horizons in steady-state navigation.

Author: Ship Trajectory Prediction Team
Date: 2025-11-04
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from dataclasses import dataclass

from data_preprocessing_pipeline.config import config

logger = logging.getLogger(__name__)


@dataclass
class VesselState:
    """Represents vessel state at a point in time"""
    timestamp: pd.Timestamp
    latitude: float
    longitude: float
    sog: float  # Speed Over Ground (knots)
    cog: float  # Course Over Ground (degrees)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'sog': self.sog,
            'cog': self.cog
        }


class ConstantVelocityModel:
    """
    Constant Velocity (CV) trajectory prediction model.
    
    Assumes vessel maintains constant speed and course.
    Simple but effective for short-term predictions in open water.
    
    Prediction equations:
        lat(t+Δt) = lat(t) + v * cos(θ) * Δt
        lon(t+Δt) = lon(t) + v * sin(θ) * Δt
    
    where:
        v = speed over ground
        θ = course over ground
        Δt = time horizon
    """
    
    def __init__(self):
        """Initialize Constant Velocity model"""
        logger.info("Constant Velocity Model initialized")
    
    def predict(self, 
                current_state: VesselState,
                horizons: List[int]) -> List[VesselState]:
        """
        Predict future positions assuming constant velocity
        
        Args:
            current_state: Current vessel state
            horizons: List of prediction horizons in seconds
            
        Returns:
            List of predicted VesselState objects
        """
        predictions = []
        
        for horizon_seconds in horizons:
            # Convert SOG from knots to degrees per second
            # 1 knot ≈ 1.852 km/h ≈ 0.0005144 m/s
            # At equator: 1 degree latitude ≈ 111 km
            # This is a simplification; more accurate for small distances
            
            # Time in hours
            time_hours = horizon_seconds / 3600.0
            
            # Distance traveled in nautical miles
            distance_nm = current_state.sog * time_hours
            
            # Convert to degrees (approximate)
            # 1 nautical mile ≈ 1/60 degree latitude
            distance_deg = distance_nm / 60.0
            
            # Calculate position change
            cog_rad = np.radians(current_state.cog)
            
            # Latitude change (north-south)
            delta_lat = distance_deg * np.cos(cog_rad)
            
            # Longitude change (east-west), adjusted for latitude
            delta_lon = distance_deg * np.sin(cog_rad) / np.cos(np.radians(current_state.latitude))
            
            # Predicted state
            predicted_state = VesselState(
                timestamp=current_state.timestamp + pd.Timedelta(seconds=horizon_seconds),
                latitude=current_state.latitude + delta_lat,
                longitude=current_state.longitude + delta_lon,
                sog=current_state.sog,  # Constant speed
                cog=current_state.cog   # Constant course
            )
            
            predictions.append(predicted_state)
        
        logger.debug(f"CV prediction: {len(predictions)} horizons")
        return predictions
    
    def predict_trajectory(self,
                          history: pd.DataFrame,
                          horizons: List[int]) -> List[VesselState]:
        """
        Predict using most recent state from trajectory history
        
        Args:
            history: DataFrame with vessel trajectory history
            horizons: Prediction horizons in seconds
            
        Returns:
            List of predicted states
        """
        # Use most recent observation
        latest = history.iloc[-1]
        current_state = VesselState(
            timestamp=latest['timestamp'],
            latitude=latest['latitude'],
            longitude=latest['longitude'],
            sog=latest['sog'],
            cog=latest['cog']
        )
        
        return self.predict(current_state, horizons)


class ConstantTurnRateModel:
    """
    Constant Turn Rate (CTR) trajectory prediction model.
    
    Assumes vessel maintains constant speed and turn rate.
    Better for predicting curved trajectories during maneuvers.
    
    The vessel follows a circular arc with radius:
        R = v / ω
    where:
        v = speed
        ω = turn rate (radians per second)
    """
    
    def __init__(self):
        """Initialize Constant Turn Rate model"""
        self.config = config.model
        logger.info("Constant Turn Rate Model initialized")
    
    def predict(self,
                current_state: VesselState,
                turn_rate: float,
                horizons: List[int]) -> List[VesselState]:
        """
        Predict future positions assuming constant turn rate
        
        Args:
            current_state: Current vessel state
            turn_rate: Turn rate in degrees per second
            horizons: Prediction horizons in seconds
            
        Returns:
            List of predicted states
        """
        predictions = []
        
        # If turn rate is very small, use straight-line approximation
        if abs(turn_rate) < self.config.ctr_min_turn_rate:
            cv_model = ConstantVelocityModel()
            return cv_model.predict(current_state, horizons)
        
        # Convert turn rate to radians per second
        omega = np.radians(turn_rate)
        
        # Calculate turn radius (in nautical miles)
        # SOG is in knots, need to convert to nm/s
        speed_nm_per_sec = current_state.sog / 3600.0
        turn_radius_nm = speed_nm_per_sec / omega if omega != 0 else float('inf')
        
        # Convert to degrees
        turn_radius_deg = turn_radius_nm / 60.0
        
        for horizon_seconds in horizons:
            # Angle traversed (radians)
            angle_traversed = omega * horizon_seconds
            
            # New course
            new_cog = (current_state.cog + np.degrees(angle_traversed)) % 360
            
            # Calculate position on circular arc
            cog_rad = np.radians(current_state.cog)
            
            # Center of circular path
            center_lat = current_state.latitude - turn_radius_deg * np.sin(cog_rad)
            center_lon = current_state.longitude + turn_radius_deg * np.cos(cog_rad) / np.cos(np.radians(current_state.latitude))
            
            # New position on circle
            new_cog_rad = np.radians(new_cog)
            new_lat = center_lat + turn_radius_deg * np.sin(new_cog_rad)
            new_lon = center_lon - turn_radius_deg * np.cos(new_cog_rad) / np.cos(np.radians(new_lat))
            
            predicted_state = VesselState(
                timestamp=current_state.timestamp + pd.Timedelta(seconds=horizon_seconds),
                latitude=new_lat,
                longitude=new_lon,
                sog=current_state.sog,
                cog=new_cog
            )
            
            predictions.append(predicted_state)
        
        logger.debug(f"CTR prediction: {len(predictions)} horizons")
        return predictions
    
    def estimate_turn_rate(self, history: pd.DataFrame) -> float:
        """
        Estimate turn rate from trajectory history
        
        Args:
            history: DataFrame with vessel trajectory history
            
        Returns:
            Estimated turn rate in degrees per second
        """
        if len(history) < 2:
            return 0.0
        
        # Calculate course changes and time differences
        course_changes = []
        time_diffs = []
        
        for i in range(1, len(history)):
            # Course change (handle wraparound)
            cog_diff = history.iloc[i]['cog'] - history.iloc[i-1]['cog']
            if cog_diff > 180:
                cog_diff -= 360
            elif cog_diff < -180:
                cog_diff += 360
            
            # Time difference in seconds
            time_diff = (history.iloc[i]['timestamp'] - 
                        history.iloc[i-1]['timestamp']).total_seconds()
            
            if time_diff > 0:
                course_changes.append(cog_diff)
                time_diffs.append(time_diff)
        
        if not course_changes:
            return 0.0
        
        # Calculate average turn rate
        turn_rates = [cc / td for cc, td in zip(course_changes, time_diffs)]
        avg_turn_rate = np.mean(turn_rates)
        
        return avg_turn_rate
    
    def predict_trajectory(self,
                          history: pd.DataFrame,
                          horizons: List[int]) -> List[VesselState]:
        """
        Predict using estimated turn rate from history
        
        Args:
            history: DataFrame with vessel trajectory history
            horizons: Prediction horizons in seconds
            
        Returns:
            List of predicted states
        """
        # Estimate turn rate from history
        turn_rate = self.estimate_turn_rate(history)
        
        # Use most recent state
        latest = history.iloc[-1]
        current_state = VesselState(
            timestamp=latest['timestamp'],
            latitude=latest['latitude'],
            longitude=latest['longitude'],
            sog=latest['sog'],
            cog=latest['cog']
        )
        
        return self.predict(current_state, turn_rate, horizons)


class KalmanFilter:
    """
    Kalman Filter for vessel trajectory prediction.
    
    State vector: [latitude, longitude, velocity_north, velocity_east]
    Measurement vector: [latitude, longitude, sog, cog]
    
    The Kalman Filter provides optimal state estimation under Gaussian noise
    assumptions and can handle noisy AIS measurements.
    """
    
    def __init__(self):
        """Initialize Kalman Filter"""
        self.config = config.model
        
        # State: [lat, lon, v_north, v_east]
        self.state_dim = 4
        self.measurement_dim = 4
        
        # State vector
        self.x = np.zeros(self.state_dim)
        
        # State covariance matrix
        self.P = np.eye(self.state_dim) * self.config.kalman_initial_covariance
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * self.config.kalman_process_noise
        
        # Measurement noise covariance
        self.R = np.eye(self.measurement_dim) * self.config.kalman_measurement_noise
        
        # Measurement matrix (we measure all state components)
        self.H = np.eye(self.measurement_dim)
        
        self.initialized = False
        
        logger.info("Kalman Filter initialized")
    
    def initialize(self, initial_state: VesselState):
        """
        Initialize filter with first measurement
        
        Args:
            initial_state: Initial vessel state
        """
        # Convert SOG and COG to velocity components
        sog_ms = initial_state.sog * 0.514444  # knots to m/s
        cog_rad = np.radians(initial_state.cog)
        
        v_north = sog_ms * np.cos(cog_rad)
        v_east = sog_ms * np.sin(cog_rad)
        
        # Initialize state
        self.x = np.array([
            initial_state.latitude,
            initial_state.longitude,
            v_north,
            v_east
        ])
        
        self.initialized = True
        logger.debug("Kalman Filter initialized with state")
    
    def predict_step(self, dt: float) -> np.ndarray:
        """
        Prediction step of Kalman Filter
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Predicted state vector
        """
        # State transition matrix (constant velocity model)
        # Convert velocity from m/s to degrees per second
        # Approximate: 1 degree latitude ≈ 111 km
        meters_per_degree = 111000.0
        
        F = np.array([
            [1, 0, dt/meters_per_degree, 0],
            [0, 1, 0, dt/meters_per_degree],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Predict state
        self.x = F @ self.x
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x
    
    def update_step(self, measurement: np.ndarray):
        """
        Update step of Kalman Filter
        
        Args:
            measurement: Measurement vector [lat, lon, v_north, v_east]
        """
        # Innovation (measurement residual)
        y = measurement - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
    
    def predict(self,
                horizons: List[int],
                current_timestamp: pd.Timestamp) -> List[VesselState]:
        """
        Predict future states
        
        Args:
            horizons: Prediction horizons in seconds
            current_timestamp: Current timestamp
            
        Returns:
            List of predicted states
        """
        if not self.initialized:
            raise ValueError("Kalman Filter must be initialized before prediction")
        
        predictions = []
        current_x = self.x.copy()
        current_P = self.P.copy()
        
        last_horizon = 0
        for horizon in horizons:
            # Predict for time difference
            dt = horizon - last_horizon
            
            # State transition
            meters_per_degree = 111000.0
            F = np.array([
                [1, 0, dt/meters_per_degree, 0],
                [0, 1, 0, dt/meters_per_degree],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            current_x = F @ current_x
            
            # Convert velocity components back to SOG/COG
            v_north = current_x[2]
            v_east = current_x[3]
            
            sog_ms = np.sqrt(v_north**2 + v_east**2)
            sog_knots = sog_ms / 0.514444
            
            cog_rad = np.arctan2(v_east, v_north)
            cog_deg = (np.degrees(cog_rad) + 360) % 360
            
            predicted_state = VesselState(
                timestamp=current_timestamp + pd.Timedelta(seconds=horizon),
                latitude=current_x[0],
                longitude=current_x[1],
                sog=sog_knots,
                cog=cog_deg
            )
            
            predictions.append(predicted_state)
            last_horizon = horizon
        
        logger.debug(f"Kalman prediction: {len(predictions)} horizons")
        return predictions
    
    def predict_trajectory(self,
                          history: pd.DataFrame,
                          horizons: List[int]) -> List[VesselState]:
        """
        Filter trajectory history and predict future
        
        Args:
            history: DataFrame with vessel trajectory history
            horizons: Prediction horizons in seconds
            
        Returns:
            List of predicted states
        """
        # Initialize with first observation
        first = history.iloc[0]
        self.initialize(VesselState(
            timestamp=first['timestamp'],
            latitude=first['latitude'],
            longitude=first['longitude'],
            sog=first['sog'],
            cog=first['cog']
        ))
        
        # Filter through all observations
        for i in range(1, len(history)):
            current = history.iloc[i]
            previous = history.iloc[i-1]
            
            # Time difference
            dt = (current['timestamp'] - previous['timestamp']).total_seconds()
            
            # Prediction step
            self.predict_step(dt)
            
            # Measurement
            sog_ms = current['sog'] * 0.514444
            cog_rad = np.radians(current['cog'])
            v_north = sog_ms * np.cos(cog_rad)
            v_east = sog_ms * np.sin(cog_rad)
            
            measurement = np.array([
                current['latitude'],
                current['longitude'],
                v_north,
                v_east
            ])
            
            # Update step
            self.update_step(measurement)
        
        # Now predict future
        latest_timestamp = history.iloc[-1]['timestamp']
        return self.predict(horizons, latest_timestamp)


def test_statistical_models():
    """Test statistical prediction models"""
    logger.info("Testing statistical models...")
    
    # Create sample trajectory
    sample_data = {
        'mmsi': [123456789] * 10,
        'latitude': [59.0, 59.01, 59.02, 59.03, 59.04, 59.05, 59.06, 59.07, 59.08, 59.09],
        'longitude': [10.0, 10.01, 10.02, 10.03, 10.04, 10.05, 10.06, 10.07, 10.08, 10.09],
        'sog': [10.0] * 10,
        'cog': [45.0] * 10,
        'timestamp': pd.date_range('2025-01-01', periods=10, freq='1min')
    }
    
    history = pd.DataFrame(sample_data)
    horizons = config.model.prediction_horizons
    
    print("\n=== Testing Statistical Models ===")
    print(f"History: {len(history)} points")
    print(f"Prediction horizons: {horizons} seconds\n")
    
    # Test Constant Velocity
    print("--- Constant Velocity Model ---")
    cv_model = ConstantVelocityModel()
    cv_predictions = cv_model.predict_trajectory(history, horizons)
    
    for i, pred in enumerate(cv_predictions):
        print(f"Horizon {horizons[i]}s: "
              f"Lat={pred.latitude:.6f}, Lon={pred.longitude:.6f}, "
              f"SOG={pred.sog:.2f}, COG={pred.cog:.2f}")
    
    # Test Constant Turn Rate
    print("\n--- Constant Turn Rate Model ---")
    ctr_model = ConstantTurnRateModel()
    turn_rate = ctr_model.estimate_turn_rate(history)
    print(f"Estimated turn rate: {turn_rate:.4f} deg/s")
    ctr_predictions = ctr_model.predict_trajectory(history, horizons)
    
    for i, pred in enumerate(ctr_predictions):
        print(f"Horizon {horizons[i]}s: "
              f"Lat={pred.latitude:.6f}, Lon={pred.longitude:.6f}, "
              f"SOG={pred.sog:.2f}, COG={pred.cog:.2f}")
    
    # Test Kalman Filter
    print("\n--- Kalman Filter ---")
    kf = KalmanFilter()
    kf_predictions = kf.predict_trajectory(history, horizons)
    
    for i, pred in enumerate(kf_predictions):
        print(f"Horizon {horizons[i]}s: "
              f"Lat={pred.latitude:.6f}, Lon={pred.longitude:.6f}, "
              f"SOG={pred.sog:.2f}, COG={pred.cog:.2f}")
    
    logger.info("Statistical models test complete")


if __name__ == "__main__":
    test_statistical_models()