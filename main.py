"""
Main Execution Script for Ship Trajectory Prediction

This script demonstrates the complete pipeline:
1. Database connection and data loading
2. Data cleaning
3. Trajectory segmentation and preprocessing
4. Statistical model prediction (CV, CTR, Kalman Filter)
5. Model evaluation and comparison

Usage:
    python main.py --mode [full|test|predict]
    
    full: Run complete pipeline on full dataset
    test: Run on small test dataset
    predict: Make predictions for specific vessel

Author: Ship Trajectory Prediction Team
Date: 2025-11-04
"""

import logging
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List
import sys
from pathlib import Path

# Import our modules
from config.config import config
from data.preprocessing.database import DatabaseConnection, AISDataLoader
from data.preprocessing.data_cleaning import AISDataCleaner
from data.preprocessing.data_preprocessing import TrajectorySegmenter, FeatureEngineering, DatasetSplitter
from models.statistical.baseline import (ConstantVelocityModel, ConstantTurnRateModel,
                                         KalmanFilter, VesselState)
from models.machine_learning.ml import RandomForestTrajectoryPredictor
from models.ensemble.ensemble import LearnedWeightingEnsemble, FeatureBasedGatingEnsemble, StackingEnsemble
from evaluation.evaluation import TrajectoryEvaluator, PredictionMetrics

logger = logging.getLogger(__name__)


class TrajectoryPredictionPipeline:
    """
    Complete pipeline for ship trajectory prediction.
    Orchestrates all components from data loading to evaluation.
    """
    
    def __init__(self):
        """Initialize pipeline components"""
        logger.info("Initializing Trajectory Prediction Pipeline...")

        # Initialize components
        self.db_connection = None
        self.data_loader = None
        self.cleaner = AISDataCleaner()
        self.segmenter = TrajectorySegmenter()
        self.feature_engineer = FeatureEngineering()
        self.splitter = DatasetSplitter()
        self.evaluator = TrajectoryEvaluator()

        # Statistical Models (no training needed)
        self.statistical_models = {
            'Constant Velocity': ConstantVelocityModel(),
            'Constant Turn Rate': ConstantTurnRateModel(),
            'Kalman Filter': KalmanFilter()
        }

        # Machine Learning Models (require training)
        self.ml_models = {
            'Random Forest': RandomForestTrajectoryPredictor(n_estimators=100, max_depth=20)
        }

        # Ensemble Models (require training after base models)
        self.ensemble_models = {}

        # Combined model dictionary
        self.models = {}

        logger.info("Pipeline initialized successfully")
    
    def connect_database(self):
        """Establish database connection"""
        logger.info("Connecting to database...")
        self.db_connection = DatabaseConnection(min_conn=1, max_conn=5)
        self.data_loader = AISDataLoader(self.db_connection)
        logger.info("Database connection established")
    
    def load_data(self, limit: int = None) -> pd.DataFrame:
        """
        Load AIS data from database
        
        Args:
            limit: Maximum number of records to load (None = all)
            
        Returns:
            Raw AIS DataFrame
        """
        logger.info(f"Loading AIS data (limit={limit})...")
        
        # Get table information
        table_info = self.data_loader.get_table_info()
        print("\n" + "="*60)
        print("DATABASE INFORMATION")
        print("="*60)
        for key, value in table_info.items():
            print(f"{key}: {value}")
        print("="*60 + "\n")
        
        # Load data
        df = self.data_loader.load_raw_data(limit=limit)
        logger.info(f"Loaded {len(df)} raw AIS records")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean AIS data
        
        Args:
            df: Raw AIS DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning AIS data...")
        df_clean = self.cleaner.clean_dataset(df, verbose=True)
        return df_clean
    
    def segment_trajectories(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Segment data into individual trajectories
        
        Args:
            df: Cleaned AIS DataFrame
            
        Returns:
            List of trajectory DataFrames
        """
        logger.info("Segmenting trajectories...")
        trajectories = self.segmenter.segment_trajectories(df)
        return trajectories
    
    def split_dataset(self, 
                     trajectories: List[pd.DataFrame]) -> Dict[str, List[pd.DataFrame]]:
        """
        Split trajectories into train/val/test sets
        
        Args:
            trajectories: List of trajectory DataFrames
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        logger.info("Splitting dataset...")
        split_data = self.splitter.split_trajectories(trajectories, split_by='temporal')
        
        print("\n" + "="*60)
        print("DATASET SPLIT")
        print("="*60)
        print(f"Training trajectories:   {len(split_data['train'])}")
        print(f"Validation trajectories: {len(split_data['val'])}")
        print(f"Test trajectories:       {len(split_data['test'])}")
        print("="*60 + "\n")

        return split_data

    def train_ml_models(self,
                       train_trajectories: List[pd.DataFrame],
                       val_trajectories: List[pd.DataFrame],
                       horizons: List[int] = None):
        """
        Train machine learning models

        Args:
            train_trajectories: Training trajectories
            val_trajectories: Validation trajectories
            horizons: Prediction horizons
        """
        if horizons is None:
            horizons = config.model.prediction_horizons

        logger.info(f"Training ML models on {len(train_trajectories)} trajectories...")

        for model_name, model in self.ml_models.items():
            logger.info(f"Training {model_name}...")

            try:
                model.train(
                    train_trajectories=train_trajectories,
                    horizons=horizons,
                    val_trajectories=val_trajectories
                )

                # Save trained model
                model_dir = Path('models/machine_learning/trained')
                model_dir.mkdir(exist_ok=True, parents=True)
                model_path = model_dir / f'{model_name.lower().replace(" ", "_")}.pkl'
                model.save_model(str(model_path))

                logger.info(f"{model_name} training complete, saved to {model_path}")

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                raise

        logger.info("ML model training complete!")

    def train_ensemble_models(self,
                             train_trajectories: List[pd.DataFrame],
                             val_trajectories: List[pd.DataFrame],
                             horizons: List[int] = None):
        """
        Train ensemble models that combine statistical and ML models

        Args:
            train_trajectories: Training trajectories
            val_trajectories: Validation trajectories (not used currently, but kept for consistency)
            horizons: Prediction horizons
        """
        if horizons is None:
            horizons = config.model.prediction_horizons

        logger.info(f"Training ensemble models on {len(train_trajectories)} trajectories...")

        # Create base models dictionary for ensembles
        base_models = {}
        base_models.update(self.statistical_models)
        base_models.update(self.ml_models)

        # Train Learned Weighting Ensemble
        logger.info("Training Learned Weighting Ensemble...")
        try:
            learned_weighting = LearnedWeightingEnsemble(base_models)
            learned_weighting.train(
                train_trajectories=train_trajectories,
                horizons=horizons,
                observation_window=config.model.observation_window_seconds
            )
            self.ensemble_models['Learned Weighting'] = learned_weighting

            # Save trained ensemble
            ensemble_dir = Path('models/ensemble/trained')
            ensemble_dir.mkdir(exist_ok=True, parents=True)
            ensemble_path = ensemble_dir / 'learned_weighting.pkl'
            learned_weighting.save_model(str(ensemble_path))

            logger.info(f"Learned Weighting Ensemble training complete, saved to {ensemble_path}")

        except Exception as e:
            logger.error(f"Failed to train Learned Weighting Ensemble: {e}")
            logger.warning("Continuing without Learned Weighting Ensemble")

        # Train Feature-Based Gating Ensemble
        logger.info("Training Feature-Based Gating Ensemble...")
        try:
            feature_gating = FeatureBasedGatingEnsemble(base_models)
            feature_gating.train(
                train_trajectories=train_trajectories,
                horizons=horizons,
                observation_window=config.model.observation_window_seconds
            )
            self.ensemble_models['Feature Gating'] = feature_gating

            # Save trained ensemble
            ensemble_path = ensemble_dir / 'feature_gating.pkl'
            feature_gating.save_model(str(ensemble_path))

            logger.info(f"Feature Gating Ensemble training complete, saved to {ensemble_path}")

        except Exception as e:
            logger.error(f"Failed to train Feature Gating Ensemble: {e}")
            logger.warning("Continuing without Feature Gating Ensemble")

        # Train Stacking Ensemble
        logger.info("Training Stacking Ensemble...")
        try:
            stacking = StackingEnsemble(base_models)
            stacking.train(
                train_trajectories=train_trajectories,
                horizons=horizons,
                observation_window=config.model.observation_window_seconds
            )
            self.ensemble_models['Stacking'] = stacking

            # Save trained ensemble
            ensemble_path = ensemble_dir / 'stacking.pkl'
            stacking.save_model(str(ensemble_path))

            logger.info(f"Stacking Ensemble training complete, saved to {ensemble_path}")

        except Exception as e:
            logger.error(f"Failed to train Stacking Ensemble: {e}")
            logger.warning("Continuing without Stacking Ensemble")

        logger.info(f"Ensemble model training complete! Trained {len(self.ensemble_models)} ensembles.")

    def evaluate_models(self,
                       test_trajectories: List[pd.DataFrame],
                       horizons: List[int] = None,
                       include_ml: bool = True,
                       include_ensemble: bool = True) -> Dict[str, Dict[int, PredictionMetrics]]:
        """
        Evaluate all models (statistical, ML, and ensemble)

        Args:
            test_trajectories: List of test trajectory DataFrames
            horizons: Prediction horizons (uses config default if None)
            include_ml: Whether to include ML models in evaluation
            include_ensemble: Whether to include ensemble models in evaluation

        Returns:
            Dictionary mapping model names to their results
        """
        if horizons is None:
            horizons = config.model.prediction_horizons

        logger.info(f"Evaluating models on {len(test_trajectories)} test trajectories...")

        # Combine models for evaluation
        models_to_evaluate = dict(self.statistical_models)
        if include_ml:
            models_to_evaluate.update(self.ml_models)
        if include_ensemble:
            models_to_evaluate.update(self.ensemble_models)

        results = {}

        for model_name, model in models_to_evaluate.items():
            logger.info(f"Evaluating {model_name}...")

            try:
                model_results = self.evaluator.evaluate_model(
                    model=model,
                    test_trajectories=test_trajectories,
                    horizons=horizons,
                    observation_window=config.model.observation_window_seconds
                )
                results[model_name] = model_results

                # Print results
                print(f"\n--- {model_name} Results ---")
                for horizon, metrics in model_results.items():
                    print(f"Horizon {horizon}s: {metrics}")

            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")

        return results
    
    def run_full_pipeline(self, data_limit: int = None, train_ml: bool = True, train_ensemble: bool = True):
        """
        Execute complete pipeline

        Args:
            data_limit: Limit on number of records to load (None = all)
            train_ml: Whether to train ML models (can be slow)
            train_ensemble: Whether to train ensemble models (requires ML models)
        """
        logger.info("="*60)
        logger.info("STARTING FULL PIPELINE")
        logger.info("="*60)

        try:
            # Step 1: Connect to database
            self.connect_database()

            # Step 2: Load data
            raw_data = self.load_data(limit=data_limit)

            # Step 3: Clean data
            clean_data = self.clean_data(raw_data)

            # Step 4: Segment trajectories
            trajectories = self.segment_trajectories(clean_data)

            # Step 5: Split dataset
            split_data = self.split_dataset(trajectories)

            # Step 6: Train ML models if requested
            if train_ml:
                logger.info("Training machine learning models...")
                self.train_ml_models(
                    train_trajectories=split_data['train'],
                    val_trajectories=split_data['val']
                )
            else:
                logger.info("Skipping ML model training")

            # Step 7: Train ensemble models if requested
            if train_ensemble and train_ml:
                logger.info("Training ensemble models...")
                self.train_ensemble_models(
                    train_trajectories=split_data['train'],
                    val_trajectories=split_data['val']
                )
            elif train_ensemble and not train_ml:
                logger.warning("Cannot train ensemble models without ML models. Skipping ensemble training.")
            else:
                logger.info("Skipping ensemble model training")

            # Step 8: Evaluate all models on test set
            logger.info("Evaluating models on test set...")
            results = self.evaluate_models(
                test_trajectories=split_data['test'],
                include_ml=train_ml,
                include_ensemble=train_ensemble and train_ml
            )

            # Step 8: Create comparison table
            comparison_df = self.evaluator.compare_models(
                results,
                config.model.prediction_horizons
            )

            print("\n" + "="*60)
            print("MODEL COMPARISON")
            print("="*60)
            print(comparison_df.to_string(index=False))
            print("="*60 + "\n")

            # Step 9: Save results
            self.save_results(comparison_df, results)

            logger.info("Pipeline completed successfully!")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

        finally:
            # Cleanup
            if self.db_connection:
                self.db_connection.close_all_connections()
    
    def save_results(self,
                    comparison_df: pd.DataFrame,
                    detailed_results: Dict):
        """Save results to files"""
        output_dir = Path('./models/statistical/results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison table
        comparison_path = output_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Saved comparison table to {comparison_path}")
        
        # Save detailed results
        results_path = output_dir / 'detailed_results.txt'
        with open(results_path, 'w') as f:
            f.write("DETAILED MODEL EVALUATION RESULTS\n")
            f.write("="*60 + "\n\n")
            
            for model_name, results in detailed_results.items():
                f.write(f"{model_name}\n")
                f.write("-"*60 + "\n")
                for horizon, metrics in results.items():
                    f.write(f"Horizon {horizon}s:\n")
                    for key, value in metrics.to_dict().items():
                        f.write(f"  {key}: {value:.2f}\n")
                    f.write("\n")
                f.write("\n")
        
        logger.info(f"Saved detailed results to {results_path}")


def run_test_mode(train_ml: bool = True, train_ensemble: bool = True):
    """Run pipeline on small test dataset"""
    logger.info("Running in TEST mode with limited data...")

    pipeline = TrajectoryPredictionPipeline()

    # Run with limited data (10,000 records)
    pipeline.run_full_pipeline(data_limit=10000, train_ml=train_ml, train_ensemble=train_ensemble)


def run_full_mode(train_ml: bool = True, train_ensemble: bool = True):
    """Run pipeline on full dataset"""
    logger.info("Running in FULL mode with complete dataset...")

    response = input("This will process the entire 32M record dataset. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return

    pipeline = TrajectoryPredictionPipeline()
    pipeline.run_full_pipeline(data_limit=None, train_ml=train_ml, train_ensemble=train_ensemble)


def run_predict_mode(mmsi: int):
    """Make prediction for specific vessel"""
    logger.info(f"Making prediction for vessel MMSI {mmsi}...")
    
    try:
        # Initialize components
        db = DatabaseConnection(min_conn=1, max_conn=2)
        loader = AISDataLoader(db)
        
        # Load vessel trajectory
        trajectory = loader.load_vessel_trajectory(mmsi)
        
        if trajectory.empty:
            print(f"No data found for MMSI {mmsi}")
            return
        
        print(f"\nLoaded trajectory: {len(trajectory)} points")
        print(f"Time range: {trajectory['timestamp'].min()} to {trajectory['timestamp'].max()}")
        
        # Clean data
        cleaner = AISDataCleaner()
        trajectory_clean = cleaner.clean_dataset(trajectory, verbose=False)
        
        # Use last 30 minutes as observation window
        observation = trajectory_clean.iloc[-30:]
        
        print(f"\nObservation window: {len(observation)} points")
        print(observation[['timestamp', 'latitude', 'longitude', 'sog', 'cog']].tail())
        
        # Make predictions with all models
        horizons = config.model.prediction_horizons
        
        print(f"\n{'Model':<20} | " + " | ".join([f"{h}s" for h in horizons]))
        print("-" * 100)
        
        models = {
            'Constant Velocity': ConstantVelocityModel(),
            'Constant Turn Rate': ConstantTurnRateModel(),
            'Kalman Filter': KalmanFilter()
        }
        
        for model_name, model in models.items():
            predictions = model.predict_trajectory(observation, horizons)
            
            print(f"{model_name:<20} | ", end="")
            for pred in predictions:
                print(f"({pred.latitude:.4f},{pred.longitude:.4f}) | ", end="")
            print()
        
        # Close connection
        db.close_all_connections()
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Ship Trajectory Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode test              # Run on small test dataset with ML training
  python main.py --mode test --skip-ml    # Run test without ML training (faster)
  python main.py --mode full              # Run on full dataset with ML training
  python main.py --mode predict --mmsi 123456789  # Predict specific vessel
        """
    )

    parser.add_argument('--mode', type=str, required=True,
                       choices=['test', 'full', 'predict'],
                       help='Execution mode')

    parser.add_argument('--mmsi', type=int,
                       help='MMSI number for predict mode')

    parser.add_argument('--skip-ml', action='store_true',
                       help='Skip ML model training (use statistical models only)')

    parser.add_argument('--skip-ensemble', action='store_true',
                       help='Skip ensemble model training (train only base models)')

    args = parser.parse_args()

    # Validate configuration
    if not config.validate():
        logger.error("Configuration validation failed")
        sys.exit(1)

    # Execute based on mode
    if args.mode == 'test':
        run_test_mode(train_ml=not args.skip_ml, train_ensemble=not args.skip_ensemble)
    elif args.mode == 'full':
        run_full_mode(train_ml=not args.skip_ml, train_ensemble=not args.skip_ensemble)
    elif args.mode == 'predict':
        if not args.mmsi:
            logger.error("--mmsi required for predict mode")
            sys.exit(1)
        run_predict_mode(args.mmsi)


if __name__ == "__main__":
    main()