# Ship Intent Prediction

A  ship trajectory prediction system using real AIS (Automatic Identification System) data from Norwegian coastal waters. This project implements and compares multiple prediction approaches including statistical models, machine learning, and data-driven ensemble methods.

## Dataset

- **Source**: TimescaleDB database with real AIS position data from sognefjord collected by Norwegian Electric ssytems
- **Size**: 32.4 million records
- **Vessels**: 1,065 unique ships
- **Time Period**: August 2024 - March 2025 (240 days)
- **Region**: Norwegian coastal waters

## Features

### Prediction Models

1. **Statistical Baselines**
   - Constant Velocity (CV)
   - Constant Turn Rate (CTR)
   - Kalman Filter (KF)

2. **Machine Learning**
   - Random Forest with 35 engineered features
   - Predicts position deltas (changes) for improved accuracy

3. **Ensemble Methods** (Data-Driven)
   - Learned Weighting Ensemble: Learns optimal model weights
   - Feature-Based Gating: Selects best model per scenario
   - Stacking Ensemble: Meta-learning for optimal combinations

### Data Pipeline

- **Data Cleaning**: Removes duplicates, outliers, invalid values (99.5%+ retention)
- **Trajectory Segmentation**: Splits data into valid trajectories (30-min gap threshold)
- **Feature Engineering**: 35+ features including motion statistics, trajectory shape, temporal patterns
- **Train/Val/Test Split**: 60/20/20 temporal split

### Evaluation

- **Metrics**: ADE (Average Displacement Error), FDE, RMSE, MAE
- **Prediction Horizons**: 30s, 60s, 120s, 180s, 300s
- **Visualization**: Performance comparison plots, error growth analysis

##  Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt

# TimescaleDB (Docker)
docker run -d --name timescaledb -p 5432:5432 -e POSTGRES_PASSWORD=password timescale/timescaledb:latest-pg14
```

### Configuration

Edit `config/config.py` to set:
- Database connection parameters
- Model hyperparameters
- Prediction horizons
- File paths

### Running the Pipeline

```bash
# Test mode (10,000 records for quick testing)
python3 main.py --mode test

# Full mode (complete 32M record dataset)
python3 main.py --mode full

# Skip ML training (statistical models only)
python3 main.py --mode full --skip-ml

# Skip ensemble training
python3 main.py --mode full --skip-ensemble
```

## 📁 Project Structure

```
ship_intent_prediction/
├── config/                 # Configuration files
├── data/                   # Data loading and preprocessing
│   └── preprocessing/      # Database, cleaning, segmentation
├── models/                 # Prediction models
│   ├── statistical/        # CV, CTR, Kalman Filter
│   ├── machine_learning/   # Random Forest
│   └── ensemble/           # Ensemble methods
├── evaluation/             # Metrics and evaluation
├── utils/                  # Utility functions
├── results/                # Output results and visualizations
├── logs/                   # Training and execution logs
└── main.py                 # Main execution script
```

## Performance

### Baseline Results (Test Mode)

| Model | Mean ADE | 30s | 60s | 120s | 180s | 300s |
|-------|----------|-----|-----|------|------|------|
| Constant Velocity | 14.79m | 5.2m | 8.1m | 14.3m | 20.5m | 31.8m |
| Constant Turn Rate | 14.79m | 5.2m | 8.1m | 14.3m | 20.5m | 31.8m |
| Kalman Filter | 133.87m | 45.2m | 68.3m | 125.1m | 178.4m | 276.5m |



### Feature Engineering (35 features)

- **Current State**: latitude, longitude, SOG, COG
- **Motion Statistics**: mean, std, min, max, range of speed and course
- **Trajectory Shape**: straightness, curvature, path length
- **Temporal Features**: time since start, observation duration
- **Recent Behavior**: acceleration, course changes, trends

### Ensemble Strategies

1. **Learned Weighting**: Ridge regression predicts optimal weights from trajectory features
2. **Feature Gating**: Random Forest classifier selects best model per scenario
3. **Stacking**: Gradient Boosting meta-model learns optimal combinations



### Adding New Models

1. Implement model class with `predict_trajectory()` method
2. Add to appropriate directory (`models/statistical/`, `models/machine_learning/`, etc.)
3. Register in `main.py` pipeline
4. Model will be automatically evaluated

### Extending Features

1. Add feature extraction in `models/machine_learning/ml.py`
2. Update feature count in model initialization
3. Retrain models with new features

