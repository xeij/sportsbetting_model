# Lower League Football Value Bettor

A production-ready sports betting odds prediction model focused on lower-tier English football leagues (League One, League Two, and National League) to identify value betting opportunities.

## 🎯 Project Overview

This project uses machine learning to predict match outcomes in lower-tier English football and identify value bets where the model's predicted probability exceeds the market's implied probability. The system:

- Downloads free historical data from football-data.co.uk
- Engineers meaningful features (form, head-to-head, rest days, etc.)
- Trains multiple models (Logistic Regression, XGBoost, LightGBM)
- Backtests betting strategies with Kelly criterion stake sizing
- Identifies value betting opportunities on upcoming fixtures

## ⚠️ Disclaimer

**This model is for educational purposes only.** Sports betting involves risk, and past performance does not guarantee future results. Always bet responsibly and within your means. This is not financial advice.

## 🚀 Quick Start

### Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the full pipeline:**
   ```bash
   python main.py --all
   ```

This will:
- Download 10 seasons of historical data for E2, E3, and EC divisions
- Engineer features and train models
- Run backtesting simulation
- Generate predictions on recent fixtures

## 📊 Usage

### Individual Commands

**Download data only:**
```bash
python main.py --download
```

**Train models only:**
```bash
python main.py --train
```

**Run backtest only:**
```bash
python main.py --backtest
```

**Predict on upcoming fixtures:**
```bash
python main.py --predict
```

**Use a specific model:**
```bash
python main.py --backtest --model lightgbm_model
python main.py --predict --model baseline_model
```

Available models: `baseline_model`, `xgboost_model`, `lightgbm_model`

## 📁 Project Structure

```
lower-league-football-value-bettor/
├── data/
│   ├── raw/                    # Downloaded CSV files
│   ├── processed/              # Merged and engineered data
│   └── visualizations/         # Plots and reports
├── models/                     # Saved trained models
├── src/
│   ├── utils.py               # Odds conversion, Kelly criterion
│   ├── data_loader.py         # Data downloading and merging
│   ├── feature_engineering.py # Feature creation
│   ├── modeling.py            # Model training and evaluation
│   ├── prediction.py          # Value bet identification
│   └── backtest.py            # Strategy simulation
├── config.yaml                # Configuration parameters
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Configuration

Edit `config.yaml` to customize:

- **Data source:** Seasons and divisions to download
- **Features:** Form windows, H2H lookback
- **Models:** Hyperparameters for XGBoost and LightGBM
- **Backtesting:** Value threshold, Kelly fraction, bankroll
- **Prediction:** Minimum probabilities, bookmaker preferences

## 📈 Features

The model creates the following features for each match:

### Form Features
- Recent points, goals scored/conceded (last 5 and 10 games)
- Home/away specific form
- Overall team form

### Head-to-Head
- Previous meetings between teams
- Historical win rates and goal averages

### Situational
- Rest days since last match
- Goal difference trends

### Odds-Based
- Implied probabilities from closing odds
- Bookmaker overround (margin)
- True probabilities with vig removed

## 🤖 Models

### Baseline: Logistic Regression
Simple multinomial logistic regression with class balancing.

### XGBoost
Gradient boosting with hyperparameter tuning for multi-class classification.

### LightGBM
Fast gradient boosting optimized for lower-tier leagues.

All models predict probabilities for three outcomes: Home Win, Draw, Away Win.

## 💰 Backtesting

The backtesting module simulates a betting strategy:

1. **Value Identification:** Bet only when model probability > market probability + threshold
2. **Stake Sizing:** Use fractional Kelly criterion for optimal bet sizing
3. **Risk Management:** Filter by odds range, use conservative Kelly fraction
4. **Evaluation:** Track ROI, win rate, and bankroll over time

### Backtest Output

- **Metrics:** Total bets, ROI, win rate, final bankroll
- **Visualizations:** Cumulative profit and bankroll curves
- **Bet History:** Detailed CSV with all simulated bets

## 🎲 Prediction

The prediction module identifies value bets on upcoming fixtures:

```
UPCOMING FIXTURES - VALUE BET PREDICTIONS
================================================================================

2024-03-15 - Portsmouth vs Barnsley
--------------------------------------------------------------------------------
Model Probabilities:
  Home: 52.3%  Draw: 26.1%  Away: 21.6%
Market Odds:
  Home: 2.10  Draw: 3.40  Away: 3.80

*** VALUE BET DETECTED ***
  Bet: Home Win
  Edge: 7.2%
  Expected Value: 9.8%
```

## 📊 Data Source

Historical data is sourced from [football-data.co.uk](https://www.football-data.co.uk/), which provides free CSV files with:

- Match results (FTHG, FTAG)
- Closing odds from multiple bookmakers
- Match statistics (shots, corners, cards)

**Divisions:**
- **E2:** League One (English third tier)
- **E3:** League Two (English fourth tier)
- **EC:** National League (English fifth tier)

## 🔍 Model Evaluation

Models are evaluated using:

- **Accuracy:** Percentage of correct predictions
- **Log Loss:** Probabilistic accuracy measure
- **Brier Score:** Calibration of predicted probabilities
- **Backtested ROI:** Profitability on historical data

Calibration curves and feature importance plots are generated automatically.

## 🛠️ Advanced Usage

### Custom Fixtures

To predict on your own upcoming fixtures:

1. Create a CSV with the same format as the processed data
2. Engineer features using `src/feature_engineering.py`
3. Load the model and call `predict_upcoming_fixtures()`

### Hyperparameter Tuning

Modify `config.yaml` to experiment with different model parameters. Recommended approach:

1. Adjust `xgboost` or `lightgbm` parameters
2. Run `python main.py --train`
3. Compare metrics and backtest results

### Live Betting

For live betting (not included):

1. Fetch latest odds from a bookmaker API
2. Engineer features for upcoming matches
3. Use `src/prediction.py` to identify value bets
4. Place bets manually or via API

## 📝 Notes

- **Chronological Validation:** The model uses strict time-based train/test splits to prevent data leakage
- **Class Imbalance:** Draws are rare; models use class weighting to handle this
- **Market Efficiency:** Lower leagues may have more inefficiencies than top-tier leagues
- **Vig Removal:** The model removes bookmaker margin to get true probabilities

## 🤝 Contributing

This is an educational project. Feel free to:

- Experiment with different features
- Try alternative models (neural networks, ensembles)
- Improve data sources (add more statistics)
- Enhance backtesting (transaction costs, multiple bookmakers)

## 📄 License

This project is provided as-is for educational purposes. Use at your own risk.

## 🙏 Acknowledgments

- Data source: [football-data.co.uk](https://www.football-data.co.uk/)
- Inspired by sports analytics and quantitative betting research

---

**Remember:** The house always has an edge. Bet responsibly! 🎰
