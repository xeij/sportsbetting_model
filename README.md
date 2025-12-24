# Lower League Football Value Bettor

A machine learning system for predicting match outcomes in lower-tier English football leagues and identifying value betting opportunities.

## Overview

This project analyzes historical match data from League One (E2), League Two (E3), and National League (EC) to predict match outcomes and identify bets where the model's probability assessment differs significantly from bookmaker odds. The system uses gradient boosting algorithms and implements Kelly Criterion for stake sizing.

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup Instructions

1. Navigate to the project directory:

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

The installation will include:
- pandas and numpy for data processing
- scikit-learn for baseline models
- xgboost and lightgbm for gradient boosting
- matplotlib and seaborn for visualizations
- requests for data downloading
- pyyaml for configuration management

### Verifying Installation

Test that the installation was successful:
```bash
python -c "import pandas, xgboost, lightgbm; print('Installation successful')"
```

## Usage

The system provides a command-line interface with several modes of operation.

### Download Historical Data

Download match data from football-data.co.uk:
```bash
python main.py --download
```

This will download 16 seasons of data (2010/11 through 2025/26) for all three divisions, totaling approximately 18,000 matches.

### Train Models

Train the prediction models on downloaded data:
```bash
python main.py --train
```

This process will:
- Engineer features from raw match data
- Split data chronologically (85% training, 15% testing)
- Train three models: Logistic Regression, XGBoost, and LightGBM
- Generate evaluation metrics and visualizations
- Save trained models to the models directory

Expected training time: 15-25 minutes depending on hardware.

### Run Backtesting

Simulate betting strategy on historical data:
```bash
python main.py --backtest
```

This evaluates model performance by simulating bets on the test set using Kelly Criterion for stake sizing. Results include ROI, win rate, and detailed bet history.

### Generate Predictions

Identify value bets on recent fixtures:
```bash
python main.py --predict
```

### Complete Pipeline

Run all steps sequentially:
```bash
python main.py --all
```

### Model Selection

Specify which trained model to use for backtesting or predictions:
```bash
python main.py --backtest --model xgboost_model
python main.py --predict --model lightgbm_model
```

## Configuration

Edit config.yaml to customize:

**Data Settings:**
- Seasons to download
- Divisions to include
- Data source URLs

**Feature Engineering:**
- Form calculation windows (default: 5 and 10 games)
- Head-to-head lookback period
- Minimum matches for statistics

**Model Parameters:**
- Train/test split ratio
- XGBoost and LightGBM hyperparameters
- Number of estimators, tree depth, learning rate

**Backtesting:**
- Value threshold for bet identification (default: 5%)
- Kelly fraction for stake sizing (default: 0.25)
- Starting bankroll for simulation
- Odds range filters

## Features

The model creates the following predictive features:

**Form Metrics:**
- Recent points, goals scored/conceded over 5 and 10 game windows
- Home-specific and away-specific performance
- Overall team form across all venues

**Historical Data:**
- Head-to-head records between teams
- Previous meeting outcomes and goal averages

**Situational Factors:**
- Days of rest since previous match
- Goal difference trends

**Market Analysis:**
- Implied probabilities from closing odds
- Bookmaker margin (overround)
- True probabilities with vig removed

## Models

**Baseline Model:**
Multinomial logistic regression with balanced class weights. Provides a performance benchmark.

**XGBoost:**
Gradient boosting with 300 estimators, maximum depth of 8, and regularization. Primary model for predictions.

**LightGBM:**
Fast gradient boosting optimized for efficiency. Alternative to XGBoost with similar performance.

All models output calibrated probabilities for three outcomes: Home Win, Draw, Away Win.

## Evaluation Metrics

Models are evaluated using:

- **Accuracy:** Percentage of correct outcome predictions
- **Log Loss:** Measures probabilistic accuracy
- **Brier Score:** Assesses probability calibration
- **Backtested ROI:** Profitability on historical test data

Calibration curves and feature importance plots are automatically generated during training.

## Data Source

Historical data is sourced from football-data.co.uk, which provides free CSV files containing:

- Match results (full-time home goals, full-time away goals)
- Closing odds from multiple bookmakers
- Match statistics (shots, corners, cards when available)

**Divisions:**
- E2: League One (English third tier)
- E3: League Two (English fourth tier)
- EC: National League (English fifth tier)

Data availability varies by season. Older seasons may have limited statistical features.

## Backtesting Methodology

The backtesting module simulates a betting strategy:

1. **Value Identification:** Bets are placed only when model probability exceeds market probability plus a threshold (default 5%)
2. **Stake Sizing:** Fractional Kelly Criterion (default: quarter Kelly) determines bet size
3. **Risk Management:** Bets filtered by odds range; conservative Kelly fraction reduces variance
4. **Evaluation:** Tracks cumulative profit, ROI, win rate, and bankroll over time

All backtesting uses chronological train/test splits to prevent data leakage.

## Output Files

**Trained Models (models/):**
- baseline_model.joblib
- xgboost_model.joblib
- lightgbm_model.joblib

**Data Files (data/):**
- data/processed/all_matches.csv - Merged historical data
- data/processed/all_matches_features.csv - Feature-engineered dataset

**Visualizations (data/visualizations/):**
- calibration_xgboost.png - Probability calibration curves
- calibration_lightgbm.png
- feature_importance_xgboost.png - Top predictive features
- feature_importance_lightgbm.png
- roi_curve.png - Cumulative profit and bankroll over time
- backtest_report.txt - Detailed performance metrics
- bet_history.csv - Complete record of simulated bets
- predictions.csv - Value bet recommendations

## Technical Notes

**Chronological Validation:**
The system uses strict time-based train/test splits. Models are trained on earlier seasons and tested on recent seasons to simulate real-world deployment.

**Class Imbalance:**
Draws are less frequent than home or away wins. Models use class weighting to handle this imbalance.

**Market Efficiency:**
Lower-tier leagues may exhibit more pricing inefficiencies than top-tier leagues, potentially offering more value betting opportunities.

**Vig Removal:**
The system removes bookmaker margin from odds to estimate true probabilities for comparison with model predictions.

## Troubleshooting

**Import Errors:**
Ensure all dependencies are installed: `pip install -r requirements.txt`

**Data Download Failures:**
Check internet connection. The system retries failed downloads automatically.

**Memory Issues:**
Feature engineering processes all matches sequentially. For very large datasets, consider reducing the number of seasons in config.yaml.

**Long Training Times:**
Training 300-estimator gradient boosting models on 18,000+ matches takes 15-25 minutes. This is expected behavior.
