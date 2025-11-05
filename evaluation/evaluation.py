"""
Model Evaluation Module

This module provides comprehensive evaluation metrics and visualization
tools for trajectory prediction models. It implements standard metrics
used in trajectory forecasting research.

Metrics include:
- Average Displacement Error (ADE)
- Final Displacement Error (FDE)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Author: Ship Trajectory Prediction Team
Date: 2025-11-04
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from dataclasses import dataclass

from statistical_baseline_models.baseline import VesselState

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Container for prediction evaluation metrics"""
    ade: float  # Average Displacement Error (meters)
    fde: float  # Final Displacement Error (meters)
    mse: float  # Mean Squared Error (meters^2)
    mae: float  # Mean Absolute Error (meters)
    rmse: float  # Root Mean Squared Error (meters)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'ADE': self.ade,
            'FDE': self.fde,
            'MSE': self.mse,
            'MAE': self.mae,
            'RMSE': self.rmse
        }
    
    def __str__(self) -> str:
        """String representation"""
        return (f"ADE: {self.ade:.2f}m, FDE: {self.fde:.2f}m, "
                f"MSE: {self.mse:.2f}m², MAE: {self.mae:.2f}m, RMSE: {self.rmse:.2f}m")


class TrajectoryEvaluator:
    """
    Evaluates trajectory prediction performance using multiple metrics.
    """
    
    def __init__(self):
        """Initialize trajectory evaluator"""
        logger.info("Trajectory Evaluator initialized")
    
    def calculate_distance(self, 
                          pred_lat: float, pred_lon: float,
                          true_lat: float, true_lon: float) -> float:
        """
        Calculate geodesic distance between predicted and true positions
        
        Args:
            pred_lat, pred_lon: Predicted position
            true_lat, true_lon: True position
            
        Returns:
            Distance in meters
        """
        return geodesic((pred_lat, pred_lon), (true_lat, true_lon)).meters
    
    def evaluate_predictions(self,
                           predictions: List[VesselState],
                           ground_truth: List[VesselState]) -> PredictionMetrics:
        """
        Evaluate predictions against ground truth
        
        Args:
            predictions: List of predicted vessel states
            ground_truth: List of true vessel states
            
        Returns:
            PredictionMetrics object
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        # Calculate displacement errors for all predictions
        errors = []
        for pred, truth in zip(predictions, ground_truth):
            distance = self.calculate_distance(
                pred.latitude, pred.longitude,
                truth.latitude, truth.longitude
            )
            errors.append(distance)
        
        errors = np.array(errors)
        
        # Calculate metrics
        ade = np.mean(errors)  # Average Displacement Error
        fde = errors[-1]  # Final Displacement Error
        mse = np.mean(errors ** 2)  # Mean Squared Error
        mae = np.mean(np.abs(errors))  # Mean Absolute Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        
        return PredictionMetrics(
            ade=ade,
            fde=fde,
            mse=mse,
            mae=mae,
            rmse=rmse
        )
    
    def evaluate_model(self,
                      model,
                      test_trajectories: List[pd.DataFrame],
                      horizons: List[int],
                      observation_window: int = 1800) -> Dict[int, PredictionMetrics]:
        """
        Evaluate model on test trajectories
        
        Args:
            model: Prediction model with predict_trajectory method
            test_trajectories: List of test trajectory DataFrames
            horizons: Prediction horizons in seconds
            observation_window: Observation window in seconds
            
        Returns:
            Dictionary mapping horizons to metrics
        """
        logger.info(f"Evaluating model on {len(test_trajectories)} trajectories...")
        
        # Store errors for each horizon
        horizon_errors = {h: [] for h in horizons}
        
        successful_predictions = 0
        failed_predictions = 0
        
        for traj_idx, trajectory in enumerate(test_trajectories):
            try:
                # Extract observation window and ground truth
                min_length = observation_window + max(horizons)
                if len(trajectory) * 60 < min_length:  # Assuming 1-min sampling
                    continue
                
                # Find split point
                split_idx = len(trajectory) // 2
                
                observation = trajectory.iloc[:split_idx]
                future = trajectory.iloc[split_idx:]
                
                # Make prediction
                predictions = model.predict_trajectory(observation, horizons)
                
                # Match predictions with ground truth
                for horizon_idx, horizon in enumerate(horizons):
                    pred = predictions[horizon_idx]
                    
                    # Find closest ground truth point
                    time_diffs = (future['timestamp'] - pred.timestamp).abs()
                    closest_idx = time_diffs.argmin()
                    
                    if time_diffs.iloc[closest_idx].total_seconds() < 60:  # Within 1 minute
                        truth = future.iloc[closest_idx]
                        
                        error = self.calculate_distance(
                            pred.latitude, pred.longitude,
                            truth['latitude'], truth['longitude']
                        )
                        
                        horizon_errors[horizon].append(error)
                
                successful_predictions += 1
                
            except Exception as e:
                logger.warning(f"Failed to evaluate trajectory {traj_idx}: {e}")
                failed_predictions += 1
        
        # Calculate metrics for each horizon
        results = {}
        for horizon in horizons:
            if horizon_errors[horizon]:
                errors = np.array(horizon_errors[horizon])
                
                results[horizon] = PredictionMetrics(
                    ade=np.mean(errors),
                    fde=np.median(errors),  # Use median for FDE in aggregated results
                    mse=np.mean(errors ** 2),
                    mae=np.mean(np.abs(errors)),
                    rmse=np.sqrt(np.mean(errors ** 2))
                )
            else:
                logger.warning(f"No valid predictions for horizon {horizon}s")
        
        logger.info(f"Evaluation complete: {successful_predictions} successful, "
                   f"{failed_predictions} failed")
        
        return results
    
    def compare_models(self,
                      model_results: Dict[str, Dict[int, PredictionMetrics]],
                      horizons: List[int]) -> pd.DataFrame:
        """
        Create comparison table for multiple models
        
        Args:
            model_results: Dictionary mapping model names to their results
            horizons: List of prediction horizons
            
        Returns:
            DataFrame with comparison results
        """
        rows = []
        
        for model_name, results in model_results.items():
            for horizon in horizons:
                if horizon in results:
                    metrics = results[horizon]
                    rows.append({
                        'Model': model_name,
                        'Horizon (s)': horizon,
                        'ADE (m)': metrics.ade,
                        'FDE (m)': metrics.fde,
                        'RMSE (m)': metrics.rmse,
                        'MAE (m)': metrics.mae
                    })
        
        df = pd.DataFrame(rows)
        return df
    
    def plot_predictions(self,
                        history: pd.DataFrame,
                        predictions: List[VesselState],
                        ground_truth: pd.DataFrame,
                        title: str = "Trajectory Prediction",
                        save_path: str = None):
        """
        Visualize trajectory prediction
        
        Args:
            history: Historical trajectory
            predictions: Predicted future states
            ground_truth: True future trajectory
            title: Plot title
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=(12, 8))
        
        # Plot history
        plt.plot(history['longitude'], history['latitude'], 
                'b-o', label='History', markersize=4, linewidth=2)
        
        # Plot ground truth
        plt.plot(ground_truth['longitude'], ground_truth['latitude'],
                'g-o', label='Ground Truth', markersize=4, linewidth=2)
        
        # Plot predictions
        pred_lons = [p.longitude for p in predictions]
        pred_lats = [p.latitude for p in predictions]
        plt.plot(pred_lons, pred_lats,
                'r-^', label='Prediction', markersize=6, linewidth=2)
        
        # Mark start and end points
        plt.plot(history['longitude'].iloc[0], history['latitude'].iloc[0],
                'go', markersize=10, label='Start')
        plt.plot(pred_lons[-1], pred_lats[-1],
                'rs', markersize=10, label='Final Prediction')
        plt.plot(ground_truth['longitude'].iloc[-1], ground_truth['latitude'].iloc[-1],
                'gs', markersize=10, label='True Final')
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_error_by_horizon(self,
                             model_results: Dict[str, Dict[int, PredictionMetrics]],
                             horizons: List[int],
                             metric: str = 'ADE',
                             save_path: str = None):
        """
        Plot error vs prediction horizon for multiple models
        
        Args:
            model_results: Dictionary mapping model names to results
            horizons: List of prediction horizons
            metric: Metric to plot ('ADE', 'FDE', 'RMSE', 'MAE')
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=(10, 6))
        
        for model_name, results in model_results.items():
            errors = []
            valid_horizons = []
            
            for horizon in horizons:
                if horizon in results:
                    metrics = results[horizon]
                    error_value = getattr(metrics, metric.lower())
                    errors.append(error_value)
                    valid_horizons.append(horizon)
            
            plt.plot(valid_horizons, errors, '-o', label=model_name, linewidth=2)
        
        plt.xlabel('Prediction Horizon (seconds)')
        plt.ylabel(f'{metric} (meters)')
        plt.title(f'{metric} vs Prediction Horizon')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def test_evaluation():
    """Test evaluation module"""
    logger.info("Testing evaluation module...")
    
    # Create sample predictions and ground truth
    predictions = [
        VesselState(pd.Timestamp('2025-01-01 00:00:30'), 59.001, 10.001, 10.0, 45.0),
        VesselState(pd.Timestamp('2025-01-01 00:01:00'), 59.002, 10.002, 10.0, 45.0),
        VesselState(pd.Timestamp('2025-01-01 00:02:00'), 59.004, 10.004, 10.0, 45.0),
    ]
    
    ground_truth = [
        VesselState(pd.Timestamp('2025-01-01 00:00:30'), 59.0011, 10.0011, 10.1, 45.5),
        VesselState(pd.Timestamp('2025-01-01 00:01:00'), 59.0021, 10.0021, 10.2, 46.0),
        VesselState(pd.Timestamp('2025-01-01 00:02:00'), 59.0042, 10.0042, 10.3, 46.5),
    ]
    
    evaluator = TrajectoryEvaluator()
    metrics = evaluator.evaluate_predictions(predictions, ground_truth)
    
    print("\n=== Evaluation Metrics ===")
    print(metrics)
    print("\nMetrics Dictionary:")
    for key, value in metrics.to_dict().items():
        print(f"{key}: {value:.2f}")
    
    logger.info("Evaluation test complete")


if __name__ == "__main__":
    test_evaluation()