"""
Data-Driven Ensemble Methods for Trajectory Prediction

This module implements intelligent ensemble strategies that combine
statistical models (CV, CTR, KF) with machine learning models (RF, etc.)
using data-driven approaches.

Ensemble Strategies:
1. Learned Weighting - Train a meta-model to predict optimal weights
2. Confidence-Based - Weight by prediction confidence/uncertainty
3. Feature-Based Gating - Select model based on trajectory features
4. Error Prediction - Predict which model will perform best
5. Stacking - Use ML to combine predictions optimally
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from config.config import config
from models.statistical.baseline import VesselState
from data.preprocessing.utility import haversine_distance

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Ensemble prediction with metadata"""
    prediction: VesselState
    weights: Dict[str, float]  # Model name -> weight
    confidence: float  # Overall confidence score


class LearnedWeightingEnsemble:
    """
    Strategy 1: Learned Weighting Ensemble
    
    Trains a meta-model to predict optimal weights for each base model
    based on trajectory features. The weights are learned from validation
    data by minimizing prediction error.
    
    Features used for weight prediction:
    - Speed variability (std, range)
    - Course variability (std, range)
    - Trajectory straightness
    - Recent acceleration
    - Recent turn rate
    """
    
    def __init__(self, models: Dict[str, any]):
        """
        Initialize ensemble with base models
        
        Args:
            models: Dictionary of {model_name: model_object}
        """
        self.models = models
        self.model_names = list(models.keys())
        self.weight_predictors = {}  # One per horizon
        self.scaler = StandardScaler()
        self.is_trained = False
        
        logger.info(f"Learned Weighting Ensemble initialized with {len(models)} models")
    
    def _extract_weighting_features(self, history: pd.DataFrame) -> Dict[str, float]:
        """Extract features for predicting optimal weights"""
        features = {}
        
        # Speed statistics
        features['sog_mean'] = history['sog'].mean()
        features['sog_std'] = history['sog'].std()
        features['sog_range'] = history['sog'].max() - history['sog'].min()
        
        # Course statistics
        features['cog_std'] = history['cog'].std()
        features['cog_range'] = history['cog'].max() - history['cog'].min()
        
        # Trajectory shape
        if len(history) > 1:
            total_distance = sum([
                haversine_distance(
                    history.iloc[i]['latitude'], history.iloc[i]['longitude'],
                    history.iloc[i+1]['latitude'], history.iloc[i+1]['longitude']
                )
                for i in range(len(history)-1)
            ])
            
            straight_distance = haversine_distance(
                history.iloc[0]['latitude'], history.iloc[0]['longitude'],
                history.iloc[-1]['latitude'], history.iloc[-1]['longitude']
            )
            
            features['straightness'] = straight_distance / (total_distance + 1e-6)
        else:
            features['straightness'] = 1.0
        
        # Recent behavior (last 5 points)
        recent = history.tail(5)
        if len(recent) > 1:
            features['recent_speed_change'] = recent['sog'].diff().abs().mean()
            features['recent_course_change'] = recent['cog'].diff().abs().mean()
        else:
            features['recent_speed_change'] = 0.0
            features['recent_course_change'] = 0.0
        
        # Trajectory length
        features['trajectory_length'] = len(history)
        
        return features
    
    def train(self, 
              train_trajectories: List[pd.DataFrame],
              horizons: List[int],
              observation_window: int = 1800):
        """
        Train weight prediction models
        
        For each trajectory, we:
        1. Get predictions from all base models
        2. Calculate error for each model
        3. Compute optimal weights (inverse error)
        4. Train meta-model to predict these weights from features
        """
        logger.info("Training Learned Weighting Ensemble...")
        
        for horizon in horizons:
            logger.info(f"Training weight predictor for horizon {horizon}s...")
            
            X_samples = []
            y_weights = []  # Optimal weights for each model
            
            for traj in train_trajectories:
                if len(traj) < 40:
                    continue
                
                # Sample multiple windows
                for i in range(30, len(traj) - horizon // 60):
                    try:
                        history = traj.iloc[:i]
                        
                        # Extract features
                        features = self._extract_weighting_features(history)
                        
                        # Get predictions from all models
                        predictions = {}
                        for model_name, model in self.models.items():
                            try:
                                preds = model.predict_trajectory(history, [horizon])
                                predictions[model_name] = preds[0]
                            except:
                                continue
                        
                        if len(predictions) < 2:
                            continue
                        
                        # Get ground truth
                        future_time = traj.iloc[i-1]['timestamp'] + pd.Timedelta(seconds=horizon)
                        time_diffs = (traj['timestamp'] - future_time).abs()
                        
                        if time_diffs.min().total_seconds() < 30:
                            truth_idx = time_diffs.argmin()
                            truth = traj.iloc[truth_idx]
                            
                            # Calculate error for each model
                            errors = {}
                            for model_name, pred in predictions.items():
                                error = haversine_distance(
                                    pred.latitude, pred.longitude,
                                    truth['latitude'], truth['longitude']
                                )
                                errors[model_name] = error
                            
                            # Compute optimal weights (inverse error, normalized)
                            # Add small epsilon to avoid division by zero
                            inv_errors = {name: 1.0 / (err + 1.0) for name, err in errors.items()}
                            total = sum(inv_errors.values())
                            weights = {name: inv_err / total for name, inv_err in inv_errors.items()}
                            
                            # Store sample
                            X_samples.append(features)
                            y_weights.append([weights.get(name, 0.0) for name in self.model_names])
                    
                    except Exception as e:
                        continue
            
            if len(X_samples) < 10:
                logger.warning(f"Not enough samples for horizon {horizon}s, using uniform weights")
                continue
            
            # Train weight predictor
            X = pd.DataFrame(X_samples)
            y = np.array(y_weights)
            
            X_scaled = self.scaler.fit_transform(X)
            
            # Use Ridge regression for weight prediction
            predictor = Ridge(alpha=1.0)
            predictor.fit(X_scaled, y)
            
            self.weight_predictors[horizon] = {
                'model': predictor,
                'feature_names': X.columns.tolist()
            }
            
            logger.info(f"  Trained on {len(X_samples)} samples")
        
        self.is_trained = True
        logger.info("Learned Weighting Ensemble training complete!")
    
    def predict_trajectory(self,
                          history: pd.DataFrame,
                          horizons: List[int]) -> List[VesselState]:
        """
        Predict using learned weighted ensemble
        
        Args:
            history: Trajectory history
            horizons: Prediction horizons
            
        Returns:
            List of ensemble predictions
        """
        # Get predictions from all base models
        all_predictions = {}
        for model_name, model in self.models.items():
            try:
                preds = model.predict_trajectory(history, horizons)
                all_predictions[model_name] = preds
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue
        
        if not all_predictions:
            raise ValueError("All models failed to predict!")
        
        # Extract features for weight prediction
        features = self._extract_weighting_features(history)
        X = pd.DataFrame([features])
        
        # Predict weights and combine
        ensemble_predictions = []
        
        for horizon_idx, horizon in enumerate(horizons):
            if self.is_trained and horizon in self.weight_predictors:
                # Use learned weights
                predictor_info = self.weight_predictors[horizon]
                X_ordered = X[predictor_info['feature_names']]
                X_scaled = self.scaler.transform(X_ordered)
                
                predicted_weights = predictor_info['model'].predict(X_scaled)[0]
                
                # Ensure weights are non-negative and sum to 1
                predicted_weights = np.maximum(predicted_weights, 0)
                predicted_weights = predicted_weights / (predicted_weights.sum() + 1e-6)
                
                weights = {name: predicted_weights[i] for i, name in enumerate(self.model_names)}
            else:
                # Use uniform weights
                weights = {name: 1.0 / len(all_predictions) for name in all_predictions.keys()}
            
            # Weighted average of predictions
            weighted_lat = 0.0
            weighted_lon = 0.0
            weighted_sog = 0.0
            weighted_cog_x = 0.0
            weighted_cog_y = 0.0
            
            for model_name, preds in all_predictions.items():
                weight = weights.get(model_name, 0.0)
                pred = preds[horizon_idx]
                
                weighted_lat += weight * pred.latitude
                weighted_lon += weight * pred.longitude
                weighted_sog += weight * pred.sog
                
                # Handle circular COG averaging
                cog_rad = np.radians(pred.cog)
                weighted_cog_x += weight * np.cos(cog_rad)
                weighted_cog_y += weight * np.sin(cog_rad)
            
            # Compute final COG from circular average
            final_cog = np.degrees(np.arctan2(weighted_cog_y, weighted_cog_x)) % 360
            
            ensemble_pred = VesselState(
                timestamp=history.iloc[-1]['timestamp'] + pd.Timedelta(seconds=horizon),
                latitude=weighted_lat,
                longitude=weighted_lon,
                sog=weighted_sog,
                cog=final_cog
            )
            
            ensemble_predictions.append(ensemble_pred)
        
        return ensemble_predictions
    
    def save_model(self, filepath: str):
        """Save trained ensemble"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'weight_predictors': self.weight_predictors,
            'scaler': self.scaler,
            'model_names': self.model_names,
            'is_trained': self.is_trained
        }, filepath)
        logger.info(f"Ensemble saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained ensemble"""
        data = joblib.load(filepath)
        self.weight_predictors = data['weight_predictors']
        self.scaler = data['scaler']
        self.model_names = data['model_names']
        self.is_trained = data['is_trained']
        logger.info(f"Ensemble loaded from {filepath}")


class FeatureBasedGatingEnsemble:
    """
    Strategy 2: Feature-Based Gating Network

    Trains a classifier to SELECT the best model for each prediction
    based on trajectory features. Instead of weighting, this does
    hard selection (gating).

    The gating network learns:
    - When CV works best (straight trajectories, constant speed)
    - When ML works best (turning, accelerating, complex patterns)
    """

    def __init__(self, models: Dict[str, any]):
        """Initialize gating ensemble"""
        self.models = models
        self.model_names = list(models.keys())
        self.gating_networks = {}  # One per horizon
        self.scaler = StandardScaler()
        self.is_trained = False

        logger.info(f"Feature-Based Gating Ensemble initialized with {len(models)} models")

    def _extract_gating_features(self, history: pd.DataFrame) -> Dict[str, float]:
        """Extract features for model selection"""
        features = {}

        # Motion variability (high = use ML, low = use CV)
        features['speed_variability'] = history['sog'].std() / (history['sog'].mean() + 1e-6)
        features['course_variability'] = history['cog'].std()

        # Recent changes (high = use ML)
        recent = history.tail(10)
        if len(recent) > 1:
            features['recent_speed_change_rate'] = recent['sog'].diff().abs().mean()
            features['recent_course_change_rate'] = recent['cog'].diff().abs().mean()
        else:
            features['recent_speed_change_rate'] = 0.0
            features['recent_course_change_rate'] = 0.0

        # Trajectory complexity
        if len(history) > 2:
            # Calculate straightness
            total_dist = sum([
                haversine_distance(
                    history.iloc[i]['latitude'], history.iloc[i]['longitude'],
                    history.iloc[i+1]['latitude'], history.iloc[i+1]['longitude']
                )
                for i in range(len(history)-1)
            ])
            straight_dist = haversine_distance(
                history.iloc[0]['latitude'], history.iloc[0]['longitude'],
                history.iloc[-1]['latitude'], history.iloc[-1]['longitude']
            )
            features['trajectory_complexity'] = 1.0 - (straight_dist / (total_dist + 1e-6))
        else:
            features['trajectory_complexity'] = 0.0

        # Current state
        features['current_speed'] = history.iloc[-1]['sog']
        features['speed_percentile'] = (history.iloc[-1]['sog'] - history['sog'].min()) / (history['sog'].max() - history['sog'].min() + 1e-6)

        return features

    def train(self,
              train_trajectories: List[pd.DataFrame],
              horizons: List[int],
              observation_window: int = 1800):
        """Train gating networks"""
        logger.info("Training Feature-Based Gating Ensemble...")

        for horizon in horizons:
            logger.info(f"Training gating network for horizon {horizon}s...")

            X_samples = []
            y_best_model = []  # Index of best model

            for traj in train_trajectories:
                if len(traj) < 40:
                    continue

                for i in range(30, len(traj) - horizon // 60):
                    try:
                        history = traj.iloc[:i]
                        features = self._extract_gating_features(history)

                        # Get predictions and errors
                        errors = {}
                        for model_name, model in self.models.items():
                            try:
                                preds = model.predict_trajectory(history, [horizon])
                                pred = preds[0]

                                # Get ground truth
                                future_time = traj.iloc[i-1]['timestamp'] + pd.Timedelta(seconds=horizon)
                                time_diffs = (traj['timestamp'] - future_time).abs()

                                if time_diffs.min().total_seconds() < 30:
                                    truth_idx = time_diffs.argmin()
                                    truth = traj.iloc[truth_idx]

                                    error = haversine_distance(
                                        pred.latitude, pred.longitude,
                                        truth['latitude'], truth['longitude']
                                    )
                                    errors[model_name] = error
                            except:
                                continue

                        if len(errors) < 2:
                            continue

                        # Find best model
                        best_model = min(errors, key=errors.get)
                        best_model_idx = self.model_names.index(best_model)

                        X_samples.append(features)
                        y_best_model.append(best_model_idx)

                    except Exception as e:
                        continue

            if len(X_samples) < 10:
                logger.warning(f"Not enough samples for horizon {horizon}s")
                continue

            # Train classifier
            X = pd.DataFrame(X_samples)
            y = np.array(y_best_model)

            X_scaled = self.scaler.fit_transform(X)

            # Use Random Forest classifier
            from sklearn.ensemble import RandomForestClassifier
            classifier = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
            classifier.fit(X_scaled, y)

            self.gating_networks[horizon] = {
                'model': classifier,
                'feature_names': X.columns.tolist()
            }

            # Log model selection distribution
            unique, counts = np.unique(y, return_counts=True)
            logger.info(f"  Model selection distribution:")
            for idx, count in zip(unique, counts):
                logger.info(f"    {self.model_names[idx]}: {count} ({100*count/len(y):.1f}%)")

        self.is_trained = True
        logger.info("Feature-Based Gating Ensemble training complete!")

    def predict_trajectory(self,
                          history: pd.DataFrame,
                          horizons: List[int]) -> List[VesselState]:
        """Predict using gating network"""
        features = self._extract_gating_features(history)
        X = pd.DataFrame([features])

        predictions = []

        for horizon in horizons:
            if self.is_trained and horizon in self.gating_networks:
                # Use gating network to select model
                gate_info = self.gating_networks[horizon]
                X_ordered = X[gate_info['feature_names']]
                X_scaled = self.scaler.transform(X_ordered)

                selected_model_idx = gate_info['model'].predict(X_scaled)[0]
                selected_model_name = self.model_names[selected_model_idx]
            else:
                # Default to first model (usually CV)
                selected_model_name = self.model_names[0]

            # Get prediction from selected model
            selected_model = self.models[selected_model_name]
            pred = selected_model.predict_trajectory(history, [horizon])[0]
            predictions.append(pred)

        return predictions

    def save_model(self, filepath: str):
        """Save trained ensemble"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'gating_networks': self.gating_networks,
            'scaler': self.scaler,
            'model_names': self.model_names,
            'is_trained': self.is_trained
        }, filepath)
        logger.info(f"Gating ensemble saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained ensemble"""
        data = joblib.load(filepath)
        self.gating_networks = data['gating_networks']
        self.scaler = data['scaler']
        self.model_names = data['model_names']
        self.is_trained = data['is_trained']
        logger.info(f"Gating ensemble loaded from {filepath}")


class StackingEnsemble:
    """
    Strategy 3: Stacking Ensemble (Meta-Learning)

    Uses predictions from base models as features for a meta-model.
    The meta-model learns how to optimally combine base predictions.

    This is the most powerful ensemble approach:
    - Learns non-linear combinations
    - Can detect when models agree/disagree
    - Automatically handles model correlations
    """

    def __init__(self, models: Dict[str, any]):
        """Initialize stacking ensemble"""
        self.models = models
        self.model_names = list(models.keys())
        self.meta_models = {}  # One per horizon and output (lat, lon, sog, cog)
        self.scaler = StandardScaler()
        self.is_trained = False

        logger.info(f"Stacking Ensemble initialized with {len(models)} models")

    def _create_meta_features(self,
                             history: pd.DataFrame,
                             base_predictions: Dict[str, VesselState]) -> Dict[str, float]:
        """
        Create features for meta-model from base predictions and history

        Features include:
        - All base model predictions
        - Agreement metrics (variance across models)
        - Trajectory features
        """
        features = {}

        # Base model predictions
        for model_name, pred in base_predictions.items():
            features[f'{model_name}_lat'] = pred.latitude
            features[f'{model_name}_lon'] = pred.longitude
            features[f'{model_name}_sog'] = pred.sog
            features[f'{model_name}_cog'] = pred.cog

        # Agreement metrics
        lats = [p.latitude for p in base_predictions.values()]
        lons = [p.longitude for p in base_predictions.values()]
        sogs = [p.sog for p in base_predictions.values()]
        cogs = [p.cog for p in base_predictions.values()]

        features['lat_variance'] = np.var(lats)
        features['lon_variance'] = np.var(lons)
        features['sog_variance'] = np.var(sogs)
        features['cog_variance'] = np.var(cogs)

        # Trajectory context features
        features['current_sog'] = history.iloc[-1]['sog']
        features['current_cog'] = history.iloc[-1]['cog']
        features['sog_std'] = history['sog'].std()
        features['cog_std'] = history['cog'].std()

        return features

    def train(self,
              train_trajectories: List[pd.DataFrame],
              horizons: List[int],
              observation_window: int = 1800):
        """Train stacking ensemble"""
        logger.info("Training Stacking Ensemble...")

        for horizon in horizons:
            logger.info(f"Training meta-models for horizon {horizon}s...")

            X_samples = []
            y_lat = []
            y_lon = []
            y_sog = []
            y_cog = []

            for traj in train_trajectories:
                if len(traj) < 40:
                    continue

                for i in range(30, len(traj) - horizon // 60):
                    try:
                        history = traj.iloc[:i]

                        # Get predictions from all base models
                        base_predictions = {}
                        for model_name, model in self.models.items():
                            try:
                                preds = model.predict_trajectory(history, [horizon])
                                base_predictions[model_name] = preds[0]
                            except:
                                continue

                        if len(base_predictions) < 2:
                            continue

                        # Get ground truth
                        future_time = traj.iloc[i-1]['timestamp'] + pd.Timedelta(seconds=horizon)
                        time_diffs = (traj['timestamp'] - future_time).abs()

                        if time_diffs.min().total_seconds() < 30:
                            truth_idx = time_diffs.argmin()
                            truth = traj.iloc[truth_idx]

                            # Create meta-features
                            meta_features = self._create_meta_features(history, base_predictions)

                            X_samples.append(meta_features)
                            y_lat.append(truth['latitude'])
                            y_lon.append(truth['longitude'])
                            y_sog.append(truth['sog'])
                            y_cog.append(truth['cog'])

                    except Exception as e:
                        continue

            if len(X_samples) < 10:
                logger.warning(f"Not enough samples for horizon {horizon}s")
                continue

            # Train meta-models for each output
            X = pd.DataFrame(X_samples)
            X_scaled = self.scaler.fit_transform(X)

            # Train separate meta-model for each output
            meta_models = {}

            # Use Gradient Boosting for meta-models
            for output_name, y_data in [('lat', y_lat), ('lon', y_lon), ('sog', y_sog), ('cog', y_cog)]:
                meta_model = GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                meta_model.fit(X_scaled, y_data)
                meta_models[output_name] = meta_model

            self.meta_models[horizon] = {
                'models': meta_models,
                'feature_names': X.columns.tolist()
            }

            logger.info(f"  Trained on {len(X_samples)} samples")

        self.is_trained = True
        logger.info("Stacking Ensemble training complete!")

    def predict_trajectory(self,
                          history: pd.DataFrame,
                          horizons: List[int]) -> List[VesselState]:
        """Predict using stacking ensemble"""
        predictions = []

        for horizon in horizons:
            # Get base model predictions
            base_predictions = {}
            for model_name, model in self.models.items():
                try:
                    preds = model.predict_trajectory(history, [horizon])
                    base_predictions[model_name] = preds[0]
                except Exception as e:
                    logger.warning(f"Model {model_name} failed: {e}")
                    continue

            if not base_predictions:
                raise ValueError("All base models failed!")

            if self.is_trained and horizon in self.meta_models:
                # Use meta-models
                meta_features = self._create_meta_features(history, base_predictions)
                X = pd.DataFrame([meta_features])
                X_ordered = X[self.meta_models[horizon]['feature_names']]
                X_scaled = self.scaler.transform(X_ordered)

                meta_models = self.meta_models[horizon]['models']

                pred_lat = meta_models['lat'].predict(X_scaled)[0]
                pred_lon = meta_models['lon'].predict(X_scaled)[0]
                pred_sog = max(0.0, meta_models['sog'].predict(X_scaled)[0])
                pred_cog = meta_models['cog'].predict(X_scaled)[0] % 360
            else:
                # Fallback to simple average
                pred_lat = np.mean([p.latitude for p in base_predictions.values()])
                pred_lon = np.mean([p.longitude for p in base_predictions.values()])
                pred_sog = np.mean([p.sog for p in base_predictions.values()])

                # Circular average for COG
                cog_x = np.mean([np.cos(np.radians(p.cog)) for p in base_predictions.values()])
                cog_y = np.mean([np.sin(np.radians(p.cog)) for p in base_predictions.values()])
                pred_cog = np.degrees(np.arctan2(cog_y, cog_x)) % 360

            pred_state = VesselState(
                timestamp=history.iloc[-1]['timestamp'] + pd.Timedelta(seconds=horizon),
                latitude=pred_lat,
                longitude=pred_lon,
                sog=pred_sog,
                cog=pred_cog
            )

            predictions.append(pred_state)

        return predictions

    def save_model(self, filepath: str):
        """Save trained ensemble"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'meta_models': self.meta_models,
            'scaler': self.scaler,
            'model_names': self.model_names,
            'is_trained': self.is_trained
        }, filepath)
        logger.info(f"Stacking ensemble saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained ensemble"""
        data = joblib.load(filepath)
        self.meta_models = data['meta_models']
        self.scaler = data['scaler']
        self.model_names = data['model_names']
        self.is_trained = data['is_trained']
        logger.info(f"Stacking ensemble loaded from {filepath}")

