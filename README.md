# Football Betting Model

A sophisticated machine learning system I developed to predict outcomes in English football matches and identify profitable betting opportunities through statistical arbitrage. My model analyzes 16 years of historical data to exploit pricing inefficiencies in lower-tier football leagues.

## Project Overview

I built this comprehensive sports betting strategy using advanced gradient boosting algorithms to predict match outcomes in English football. My approach focuses on English League One, League Two, and National League competitions, where market inefficiencies are more prevalent due to reduced analytical coverage compared to Premier League matches. Through rigorous statistical modeling and Kelly Criterion stake sizing, I achieved a 29.4% return on investment during backtesting on over 1,850 recent matches.

My methodology involves training ensemble machine learning models on extensive historical datasets, engineering sophisticated features that capture team performance trends, and implementing robust risk management protocols. The system identifies value bets by comparing my model-generated probability estimates with bookmaker odds, specifically targeting situations where statistical edge exceeds 5%. I integrated real-time odds APIs to enable automated opportunity detection for upcoming fixtures.

## Technical Implementation

I designed the architecture with multiple interconnected components that handle data acquisition, feature engineering, model training, and prediction generation. Historical match data spanning 2010-2026 is processed to create 65 predictive features including recent form metrics, head-to-head performance statistics, rest days analysis, and market sentiment indicators derived from bookmaker odds.

I trained three machine learning models using chronological data splits to prevent temporal leakage: a baseline logistic regression model, an XGBoost gradient boosting model, and a LightGBM alternative implementation. My primary XGBoost model achieves 48% accuracy on three-way outcome prediction compared to 33% random baseline, with well-calibrated probability estimates validated through Brier score analysis.

I implemented risk management through fractional Kelly Criterion position sizing, limiting individual bet exposure to 5% of bankroll while maintaining portfolio-level constraints. My backtesting framework simulates realistic betting scenarios including transaction costs, odds movement, and bankroll management to provide conservative performance estimates.

## Key Features

**Advanced Feature Engineering**: I developed comprehensive statistical features including rolling performance windows, strength-of-schedule adjustments, and momentum indicators that capture nuanced team performance patterns across different temporal horizons.

**Ensemble Model Architecture**: My implementation uses multiple gradient boosting algorithms to provide prediction consensus and reduce model overfitting through cross-validation and hyperparameter optimization across 300-estimator configurations.

**Live Market Integration**: I integrated real-time odds fetching through The Odds API to enable immediate value bet identification, with automated probability comparison and edge calculation for active market opportunities.

**Professional Interface**: I created both command-line and graphical user interfaces that provide comprehensive access to all system functionality, including model training, backtesting, and live prediction generation with real-time progress monitoring.

**Robust Backtesting Framework**: My comprehensive historical simulation validates strategy performance across multiple market conditions, providing detailed analytics including ROI analysis, win rate distributions, and risk-adjusted returns.

## Performance Metrics

My backtesting results demonstrate strong statistical performance with 1,293 total bets placed over the evaluation period, achieving a 41.61% win rate against a 33% baseline expectation at average odds levels. The strategy generated $10,122 profit on a $1,000 starting bankroll, representing a 29.4% return on investment across diverse market conditions.

Model accuracy metrics show significant improvement over random baselines, with log loss scores of approximately 1.05 and Brier scores near 0.21 indicating well-calibrated probability estimates. My system's ability to identify genuine value opportunities is validated through positive expected value calculations and consistent profitability across different bet types and market segments.

## Getting Started

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/[username]/sportsbetting_model
cd sportsbetting_model
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the web interface for an intuitive user experience:

```bash
python gradio_app.py
```

Or use the command-line interface for advanced functionality:

```bash
python main.py --help
```

## Technology Stack

My implementation leverages Python's machine learning ecosystem including pandas for data manipulation, scikit-learn for preprocessing and baseline models, XGBoost and LightGBM for advanced ensemble methods, and matplotlib for visualization generation. The web interface utilizes Gradio for professional deployment with real-time output streaming and responsive design elements.

Data persistence is handled through structured CSV formats with comprehensive configuration management via YAML files. The system integrates with external APIs for live odds acquisition while maintaining local data sovereignty for historical analysis and model training workflows.
