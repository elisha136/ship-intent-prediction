"""
Configuration Management for Ship Trajectory Prediction System

This module contains all configuration parameters for database connection,
data processing, and model settings. Centralized configuration ensures
easy maintenance and consistent parameter usage across the system.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '5432'))
    database: str = os.getenv('DB_NAME', 'adimalara')
    user: str = os.getenv('DB_USER', 'postgres')
    password: str = os.getenv('DB_PASSWORD', 'changeme-strong-pass')
    
    def get_connection_string(self) -> str:
        """Generate PostgreSQL connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class DataCleaningConfig:
    """Parameters for AIS data cleaning and validation"""
    # Valid ranges for AIS data
    min_latitude: float = -90.0
    max_latitude: float = 90.0
    min_longitude: float = -180.0
    max_longitude: float = 180.0
    
    # Speed Over Ground (SOG) in knots
    min_sog: float = 0.0
    max_sog: float = 50.0  # Maximum realistic ship speed
    
    # Course Over Ground (COG) in degrees
    min_cog: float = 0.0
    max_cog: float = 360.0
    
    # Trajectory segmentation
    max_time_gap_seconds: int = 1800  # 30 minutes gap = new trajectory
    min_trajectory_points: int = 20  # Minimum points for valid trajectory
    min_trajectory_duration_seconds: int = 600  # 10 minutes minimum
    
    # Outlier detection
    max_position_jump_meters: float = 5000.0  # 5km between consecutive points
    max_speed_change_knots: float = 10.0  # Maximum speed change between readings
    max_course_change_degrees: float = 180.0  # Maximum course change
    
    # Stationary vessel filtering
    stationary_speed_threshold: float = 0.5  # Knots
    stationary_duration_seconds: int = 3600  # 1 hour


@dataclass
class PreprocessingConfig:
    """Parameters for data preprocessing and feature engineering"""
    # Temporal resampling
    target_sampling_interval_seconds: int = 60  # Resample to 1 minute intervals
    
    # Feature engineering
    velocity_window_sizes: List[int] = None  # [5, 10, 15] minutes
    acceleration_window_size: int = 300  # 5 minutes for acceleration calculation
    
    # Spatial features
    neighbor_search_radius_nm: float = 5.0  # Nautical miles
    
    # Dataset splits (percentages)
    train_split: float = 0.60
    validation_split: float = 0.20
    test_split: float = 0.20
    
    def __post_init__(self):
        if self.velocity_window_sizes is None:
            self.velocity_window_sizes = [300, 600, 900]  # 5, 10, 15 minutes in seconds


@dataclass
class ModelConfig:
    """Configuration for trajectory prediction models"""
    # Prediction horizons in seconds
    prediction_horizons: List[int] = None  # [30, 60, 120, 180, 300]
    
    # Historical observation window
    observation_window_seconds: int = 1800  # 30 minutes of history
    min_observation_points: int = 10  # Minimum historical points needed
    
    # Kalman Filter parameters
    kalman_process_noise: float = 0.1
    kalman_measurement_noise: float = 1.0
    kalman_initial_covariance: float = 1.0
    
    # Constant Turn Rate parameters
    ctr_min_turn_rate: float = 0.1  # degrees per second
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [30, 60, 120, 180, 300]


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = 'INFO'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_dir: str = 'logs'
    log_file: str = 'logs/pipeline.log'
    console_output: bool = True


class Config:
    """Main configuration class aggregating all configs"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.cleaning = DataCleaningConfig()
        self.preprocessing = PreprocessingConfig()
        self.model = ModelConfig()
        self.logging = LoggingConfig()
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the entire application"""
        # Create logs directory if it doesn't exist
        os.makedirs(self.logging.log_dir, exist_ok=True)

        handlers = []

        # File handler
        file_handler = logging.FileHandler(self.logging.log_file)
        file_handler.setFormatter(logging.Formatter(self.logging.log_format))
        handlers.append(file_handler)

        # Console handler
        if self.logging.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(self.logging.log_format))
            handlers.append(console_handler)

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.logging.log_level),
            handlers=handlers
        )
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        try:
            # Validate dataset splits sum to 1.0
            total_split = (self.preprocessing.train_split + 
                          self.preprocessing.validation_split + 
                          self.preprocessing.test_split)
            assert abs(total_split - 1.0) < 0.001, "Dataset splits must sum to 1.0"
            
            # Validate latitude/longitude ranges
            assert -90 <= self.cleaning.min_latitude <= self.cleaning.max_latitude <= 90
            assert -180 <= self.cleaning.min_longitude <= self.cleaning.max_longitude <= 180
            
            # Validate positive values
            assert self.cleaning.max_time_gap_seconds > 0
            assert self.cleaning.min_trajectory_points > 0
            assert self.model.observation_window_seconds > 0
            
            logging.info("Configuration validation successful")
            return True
            
        except AssertionError as e:
            logging.error(f"Configuration validation failed: {e}")
            return False


# Global configuration instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    print("=== Ship Trajectory Prediction Configuration ===\n")
    
    print(f"Database: {config.database.database}@{config.database.host}:{config.database.port}")
    print(f"Max time gap for trajectory: {config.cleaning.max_time_gap_seconds}s")
    print(f"Prediction horizons: {config.model.prediction_horizons}s")
    print(f"Dataset splits: Train={config.preprocessing.train_split}, "
          f"Val={config.preprocessing.validation_split}, "
          f"Test={config.preprocessing.test_split}")
    
    print(f"\nConfiguration valid: {config.validate()}")