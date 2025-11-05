"""
Ensemble Methods for Trajectory Prediction

This package provides data-driven ensemble strategies that intelligently
combine statistical and machine learning models.
"""

from models.ensemble.ensemble import (
    LearnedWeightingEnsemble,
    FeatureBasedGatingEnsemble,
    StackingEnsemble,
    EnsemblePrediction
)

__all__ = [
    'LearnedWeightingEnsemble',
    'FeatureBasedGatingEnsemble',
    'StackingEnsemble',
    'EnsemblePrediction'
]

