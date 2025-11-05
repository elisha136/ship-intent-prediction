"""
Machine Learning Trajectory Prediction Models

This module implements machine learning approaches for ship trajectory prediction:
1. Random Forest Regressor - Ensemble of decision trees
2. Gradient Boosting - Advanced ensemble method
3. Support Vector Regression (SVR) - Kernel-based regression

These models learn complex patterns from trajectory history that statistical
models cannot capture, including:
- Historical turning patterns
- Speed variation behaviors
- Contextual features (time of day, traffic density)
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import joblib
from pathlib import Path

from config.config import config
from models.statistical.baseline import VesselState
from data.preprocessing.utility import haversine_distance, bearing_between_points

logger = logging.getLogger(__name__)


class MLFeatureEngineering:
    """
    Advanced feature engineering for machine learning models.
    Extracts temporal, spatial, and motion features from trajectory history.
    """
    
    def __init__(self, observation_window_seconds: int = 1800):
        """
        Initialize feature engineer
        
        Args:
            observation_window_seconds: Historical window to use for features
        """
        self.observation_window = observation_window_seconds
        logger.info(f"ML Feature Engineering initialized with {observation_window_seconds}s window")
    
    def extract_features(self, trajectory: pd.DataFrame) -> Dict[str, float]:
        """
        Extract comprehensive feature set from trajectory history
        
        Args:
            trajectory: DataFrame with vessel trajectory history
            
        Returns:
            Dictionary of features
        """
        if len(trajectory) < 2:
            raise ValueError("Need at least 2 trajectory points for feature extraction")
        
        features = {}
        
        # Current state features (most recent observation)
        current = trajectory.iloc[-1]
        features.update(self._extract_current_state(current))
        
        # Historical motion features
        features.update(self._extract_motion_statistics(trajectory))
        
        # Temporal features
        features.update(self._extract_temporal_features(current['timestamp']))
        
        # Trajectory shape features
        features.update(self._extract_trajectory_shape(trajectory))
        
        # Recent behavior features (last few observations)
        features.update(self._extract_recent_behavior(trajectory))
        
        return features
    
    def _extract_current_state(self, observation: pd.Series) -> Dict[str, float]:
        """Extract features from current observation"""
        return {
            'current_lat': observation['latitude'],
            'current_lon': observation['longitude'],
            'current_sog': observation['sog'],
            'current_cog': observation['cog'],
            # Velocity components
            'current_v_north': observation['sog'] * np.cos(np.radians(observation['cog'])),
            'current_v_east': observation['sog'] * np.sin(np.radians(observation['cog']))
        }
    
    def _extract_motion_statistics(self, trajectory: pd.DataFrame) -> Dict[str, float]:
        """Extract statistical features from motion history"""
        features = {}
        
        # Speed statistics
        features['sog_mean'] = trajectory['sog'].mean()
        features['sog_std'] = trajectory['sog'].std()
        features['sog_min'] = trajectory['sog'].min()
        features['sog_max'] = trajectory['sog'].max()
        features['sog_range'] = features['sog_max'] - features['sog_min']
        
        # Course statistics
        features['cog_mean'] = trajectory['cog'].mean()
        features['cog_std'] = trajectory['cog'].std()
        
        # Speed changes (acceleration patterns)
        speed_changes = trajectory['sog'].diff()
        features['speed_change_mean'] = speed_changes.mean()
        features['speed_change_std'] = speed_changes.std()
        features['max_acceleration'] = speed_changes.max()
        features['max_deceleration'] = speed_changes.min()
        
        # Course changes (turning patterns)
        course_changes = trajectory['cog'].diff()
        # Handle wraparound
        course_changes = np.where(course_changes > 180, course_changes - 360, course_changes)
        course_changes = np.where(course_changes < -180, course_changes + 360, course_changes)
        
        features['course_change_mean'] = np.mean(course_changes[~np.isnan(course_changes)])
        features['course_change_std'] = np.std(course_changes[~np.isnan(course_changes)])
        features['max_turn_rate'] = np.abs(course_changes).max()
        
        return features
    
    def _extract_temporal_features(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Extract time-based features"""
        return {
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.dayofweek,
            'is_weekend': 1.0 if timestamp.dayofweek >= 5 else 0.0,
            # Cyclical encoding for hour (preserves 23:00 and 00:00 are close)
            'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
            'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
            # Cyclical encoding for day of week
            'day_sin': np.sin(2 * np.pi * timestamp.dayofweek / 7),
            'day_cos': np.cos(2 * np.pi * timestamp.dayofweek / 7)
        }
    
    def _extract_trajectory_shape(self, trajectory: pd.DataFrame) -> Dict[str, float]:
        """Extract features describing trajectory shape"""
        features = {}
        
        # Total distance traveled
        distances = []
        for i in range(1, len(trajectory)):
            dist = haversine_distance(
                trajectory.iloc[i-1]['latitude'],
                trajectory.iloc[i-1]['longitude'],
                trajectory.iloc[i]['latitude'],
                trajectory.iloc[i]['longitude']
            )
            distances.append(dist)
        
        features['total_distance'] = sum(distances)
        features['avg_distance_per_step'] = np.mean(distances) if distances else 0
        
        # Trajectory duration
        duration = (trajectory.iloc[-1]['timestamp'] - 
                   trajectory.iloc[0]['timestamp']).total_seconds()
        features['trajectory_duration'] = duration
        
        # Straightness (ratio of direct distance to path length)
        direct_distance = haversine_distance(
            trajectory.iloc[0]['latitude'],
            trajectory.iloc[0]['longitude'],
            trajectory.iloc[-1]['latitude'],
            trajectory.iloc[-1]['longitude']
        )
        features['straightness'] = direct_distance / features['total_distance'] if features['total_distance'] > 0 else 1.0
        
        # Average bearing (overall direction of travel)
        overall_bearing = bearing_between_points(
            trajectory.iloc[0]['latitude'],
            trajectory.iloc[0]['longitude'],
            trajectory.iloc[-1]['latitude'],
            trajectory.iloc[-1]['longitude']
        )
        features['overall_bearing'] = overall_bearing
        
        return features
    
    def _extract_recent_behavior(self, trajectory: pd.DataFrame, 
                                 last_n: int = 5) -> Dict[str, float]:
        """Extract features from most recent observations"""
        features = {}
        
        # Use last N points
        recent = trajectory.iloc[-last_n:]
        
        # Recent speed trend
        if len(recent) > 1:
            recent_speeds = recent['sog'].values
            # Linear regression slope
            x = np.arange(len(recent_speeds))
            slope = np.polyfit(x, recent_speeds, 1)[0] if len(x) > 1 else 0
            features['recent_speed_trend'] = slope
            
            # Recent course trend
            recent_courses = recent['cog'].values
            course_slope = np.polyfit(x, recent_courses, 1)[0] if len(x) > 1 else 0
            features['recent_course_trend'] = course_slope
        else:
            features['recent_speed_trend'] = 0.0
            features['recent_course_trend'] = 0.0
        
        # Recent acceleration
        if len(recent) > 1:
            recent_accel = recent['sog'].diff().mean()
            features['recent_acceleration'] = recent_accel
        else:
            features['recent_acceleration'] = 0.0
        
        return features
    
    def create_training_samples(self, 
                                trajectories: List[pd.DataFrame],
                                horizons: List[int],
                                min_history_points: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create training dataset from trajectories
        
        Args:
            trajectories: List of trajectory DataFrames
            horizons: Prediction horizons in seconds
            min_history_points: Minimum historical points needed
            
        Returns:
            (X_features, y_targets) DataFrames
        """
        logger.info(f"Creating training samples from {len(trajectories)} trajectories...")
        
        X_samples = []
        y_samples = []
        
        for traj_idx, trajectory in enumerate(trajectories):
            if len(trajectory) < min_history_points + max(horizons) // 60:
                continue
            
            # Sample multiple windows from this trajectory
            for i in range(min_history_points, len(trajectory) - max(horizons) // 60):
                try:
                    # Extract features from history up to point i
                    history = trajectory.iloc[:i]
                    features = self.extract_features(history)
                    
                    # Extract targets (future positions at each horizon)
                    current_time = trajectory.iloc[i-1]['timestamp']
                    current_state = trajectory.iloc[i-1]
                    targets = {}

                    for horizon in horizons:
                        # Find future observation closest to horizon
                        future_time = current_time + pd.Timedelta(seconds=horizon)
                        time_diffs = (trajectory['timestamp'] - future_time).abs()

                        if time_diffs.min().total_seconds() < 30:  # Within 30 seconds
                            future_idx = time_diffs.argmin()
                            future_obs = trajectory.iloc[future_idx]

                            # Predict DELTAS (changes) instead of absolute positions
                            targets[f'delta_lat_{horizon}s'] = future_obs['latitude'] - current_state['latitude']
                            targets[f'delta_lon_{horizon}s'] = future_obs['longitude'] - current_state['longitude']
                            # Keep SOG and COG as absolute values
                            targets[f'sog_{horizon}s'] = future_obs['sog']
                            targets[f'cog_{horizon}s'] = future_obs['cog']
                        else:
                            # Skip this sample if we don't have ground truth
                            targets = None
                            break
                    
                    if targets:
                        X_samples.append(features)
                        y_samples.append(targets)
                
                except Exception as e:
                    logger.debug(f"Failed to create sample from trajectory {traj_idx}, point {i}: {e}")
                    continue
            
            if traj_idx % 100 == 0:
                logger.info(f"Processed {traj_idx}/{len(trajectories)} trajectories, "
                           f"created {len(X_samples)} samples")
        
        logger.info(f"Created {len(X_samples)} training samples")
        
        X = pd.DataFrame(X_samples)
        y = pd.DataFrame(y_samples)
        
        return X, y


class RandomForestTrajectoryPredictor:
    """
    Random Forest model for trajectory prediction.
    
    Uses ensemble of decision trees to predict future vessel positions.
    Learns complex non-linear patterns from historical data.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 20, 
                 random_state: int = 42):
        """
        Initialize Random Forest predictor
        
        Args:
            n_estimators: Number of trees in forest
            max_depth: Maximum depth of each tree
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Separate model for each output (lat, lon at each horizon)
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_engineer = MLFeatureEngineering()
        self.is_trained = False
        self.feature_names = None
        
        logger.info(f"Random Forest initialized: {n_estimators} trees, max_depth={max_depth}")
    
    def train(self, 
              train_trajectories: List[pd.DataFrame],
              horizons: List[int],
              val_trajectories: Optional[List[pd.DataFrame]] = None):
        """
        Train Random Forest models
        
        Args:
            train_trajectories: List of training trajectory DataFrames
            horizons: Prediction horizons in seconds
            val_trajectories: Optional validation trajectories
        """
        logger.info("Training Random Forest models...")
        
        # Create training samples
        X_train, y_train = self.feature_engineer.create_training_samples(
            train_trajectories, horizons
        )
        
        if len(X_train) == 0:
            raise ValueError("No training samples created!")
        
        logger.info(f"Training on {len(X_train)} samples with {len(X_train.columns)} features")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train separate model for each target variable
        for target_col in y_train.columns:
            logger.info(f"Training model for {target_col}...")
            
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,  # Use all CPU cores
                verbose=0
            )
            
            # Remove NaN targets
            valid_mask = ~y_train[target_col].isna()
            X_valid = X_train_scaled[valid_mask]
            y_valid = y_train[target_col][valid_mask]
            
            model.fit(X_valid, y_valid)
            self.models[target_col] = model
            
            logger.info(f"  Feature importance (top 5):")
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-5:][::-1]
            for idx in top_indices:
                logger.info(f"    {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        self.is_trained = True
        logger.info("Random Forest training complete!")
        
        # Validation if provided
        if val_trajectories:
            self._validate(val_trajectories, horizons)
    
    def _validate(self, val_trajectories: List[pd.DataFrame], horizons: List[int]):
        """Validate model on validation set"""
        logger.info("Validating model...")
        
        X_val, y_val = self.feature_engineer.create_training_samples(
            val_trajectories, horizons
        )
        
        X_val_scaled = self.scaler.transform(X_val)
        
        for target_col in self.models.keys():
            y_pred = self.models[target_col].predict(X_val_scaled)
            y_true = y_val[target_col].values
            
            # Calculate RMSE
            valid_mask = ~np.isnan(y_true)
            rmse = np.sqrt(np.mean((y_pred[valid_mask] - y_true[valid_mask])**2))
            
            logger.info(f"Validation RMSE for {target_col}: {rmse:.4f}")
    
    def predict(self, 
                history: pd.DataFrame,
                horizons: List[int]) -> List[VesselState]:
        """
        Predict future vessel states
        
        Args:
            history: Trajectory history DataFrame
            horizons: Prediction horizons in seconds
            
        Returns:
            List of predicted VesselState objects
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction!")
        
        # Extract features
        features = self.feature_engineer.extract_features(history)
        X = pd.DataFrame([features])
        
        # Ensure feature order matches training
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict for each horizon
        predictions = []
        current_time = history.iloc[-1]['timestamp']
        current_lat = history.iloc[-1]['latitude']
        current_lon = history.iloc[-1]['longitude']

        for horizon in horizons:
            # Predict DELTAS (changes in position)
            delta_lat = self.models[f'delta_lat_{horizon}s'].predict(X_scaled)[0]
            delta_lon = self.models[f'delta_lon_{horizon}s'].predict(X_scaled)[0]

            # Add deltas to current position to get predicted position
            pred_lat = current_lat + delta_lat
            pred_lon = current_lon + delta_lon

            # Predict absolute SOG and COG
            pred_sog = self.models[f'sog_{horizon}s'].predict(X_scaled)[0]
            pred_cog = self.models[f'cog_{horizon}s'].predict(X_scaled)[0]

            # Ensure COG is in valid range [0, 360)
            pred_cog = pred_cog % 360

            # Ensure SOG is non-negative
            pred_sog = max(0.0, pred_sog)

            pred_state = VesselState(
                timestamp=current_time + pd.Timedelta(seconds=horizon),
                latitude=pred_lat,
                longitude=pred_lon,
                sog=pred_sog,
                cog=pred_cog
            )

            predictions.append(pred_state)

        return predictions
    
    def predict_trajectory(self,
                          history: pd.DataFrame,
                          horizons: List[int]) -> List[VesselState]:
        """
        Wrapper for compatibility with evaluation framework
        
        Args:
            history: Trajectory history
            horizons: Prediction horizons
            
        Returns:
            List of predicted states
        """
        return self.predict(history, horizons)
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model!")
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.n_estimators = model_data['n_estimators']
        self.max_depth = model_data['max_depth']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained models
        
        Returns:
            DataFrame with feature importances
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Average importance across all models
        importances = {}
        for feature_name in self.feature_names:
            feature_idx = self.feature_names.index(feature_name)
            avg_importance = np.mean([
                model.feature_importances_[feature_idx] 
                for model in self.models.values()
            ])
            importances[feature_name] = avg_importance
        
        df = pd.DataFrame(list(importances.items()), 
                         columns=['Feature', 'Importance'])
        df = df.sort_values('Importance', ascending=False)
        
        return df


def test_ml_models():
    """Test machine learning models"""
    logger.info("Testing ML models...")
    
    # Create sample trajectories
    np.random.seed(42)
    
    trajectories = []
    for i in range(5):
        n_points = 50
        trajectory = pd.DataFrame({
            'mmsi': [123456789] * n_points,
            'latitude': 59.0 + np.cumsum(np.random.normal(0.0001, 0.00005, n_points)),
            'longitude': 10.0 + np.cumsum(np.random.normal(0.0001, 0.00005, n_points)),
            'sog': np.random.normal(10.0, 0.5, n_points),
            'cog': np.random.normal(45.0, 2.0, n_points),
            'timestamp': pd.date_range('2025-01-01', periods=n_points, freq='1min')
        })
        trajectories.append(trajectory)
    
    print(f"\nCreated {len(trajectories)} sample trajectories")
    
    # Test feature engineering
    print("\n=== Testing Feature Engineering ===")
    fe = MLFeatureEngineering()
    features = fe.extract_features(trajectories[0])
    print(f"Extracted {len(features)} features:")
    for key, value in list(features.items())[:10]:
        print(f"  {key}: {value:.4f}")
    
    # Test Random Forest
    print("\n=== Testing Random Forest ===")
    rf_model = RandomForestTrajectoryPredictor(n_estimators=10, max_depth=10)
    
    horizons = [30, 60, 120]
    
    # Train model
    rf_model.train(trajectories[:3], horizons, val_trajectories=trajectories[3:4])
    
    # Make prediction
    test_history = trajectories[4].iloc[:30]
    predictions = rf_model.predict(test_history, horizons)
    
    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        print(f"  Horizon {horizons[i]}s: "
              f"Lat={pred.latitude:.6f}, Lon={pred.longitude:.6f}, "
              f"SOG={pred.sog:.2f}, COG={pred.cog:.2f}")
    
    # Feature importance
    print("\n=== Feature Importance (Top 10) ===")
    importance_df = rf_model.get_feature_importance()
    print(importance_df.head(10).to_string(index=False))
    
    logger.info("ML models test complete!")


if __name__ == "__main__":
    test_ml_models()