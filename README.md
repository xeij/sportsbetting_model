# Lower League Football Value Bettor

A machine learning system for predicting match outcomes in English football and identifying profitable betting opportunities.

## How This Model Works

### The Basic Concept

This model helps you find bets where the odds offered by bookmakers are better than they should be. Here's how:

1. **The model predicts match outcomes** using machine learning trained on 16 years of historical data
2. **Bookmakers also predict outcomes** and set their odds based on their predictions
3. **When the model disagrees with bookmakers**, there might be a profitable betting opportunity
4. **The model only recommends bets** where it believes you have a statistical edge

### Step-by-Step Process

**Step 1: Data Collection**

The model downloads historical match data from football-data.co.uk, including:
- Match results (who won, final scores)
- Betting odds from multiple bookmakers
- Match statistics when available

Currently covers:
- Premier League (top tier, 380 matches per season)
- League One (third tier, 552 matches per season)
- League Two (fourth tier, 552 matches per season)
- National League (fifth tier, 552 matches per season)

Total dataset: approximately 18,000 matches from 2010 to present.

**Step 2: Feature Engineering**

The model creates 65 predictive features from raw data:

- **Recent Form**: How well each team has performed in their last 5 and 10 games
  - Points earned
  - Goals scored and conceded
  - Wins, draws, and losses
  
- **Home/Away Performance**: Separate statistics for home and away matches
  - Teams perform differently at home vs away
  
- **Head-to-Head History**: How these two teams have performed against each other
  - Last 5 meetings
  - Goals scored in previous matchups
  
- **Rest Days**: Days since each team's last match
  - Fatigue can affect performance
  
- **Goal Difference Trends**: Whether teams are improving or declining
  
- **Market Analysis**: What bookmakers think (converted from odds to probabilities)
  - Bookmaker margin (overround)
  - True probabilities with margin removed

**Step 3: Model Training**

Three machine learning models are trained:

1. **Logistic Regression (Baseline)**
   - Simple statistical model
   - Provides a performance benchmark
   - Accuracy: approximately 45%

2. **XGBoost (Primary Model)**
   - Advanced gradient boosting algorithm
   - 300 decision trees working together
   - Accuracy: approximately 48%
   - Best for finding value bets

3. **LightGBM (Alternative)**
   - Similar to XGBoost but faster
   - Accuracy: approximately 48%
   - Good for validation

The models predict three outcomes for each match:
- Home Win probability
- Draw probability  
- Away Win probability

**Step 4: Backtesting**

Before using real money, the model is tested on historical data:

- **Training Period**: 2017-2024 (10,500+ matches)
- **Testing Period**: November 2024 - December 2025 (1,850+ matches)
- **Chronological Split**: Models only see past data, never future data
- **Realistic Simulation**: Simulates actual betting with stake sizing

**Backtesting Results:**
- Total Bets Placed: 1,293
- Win Rate: 41.61% (vs 33% expected at average odds)
- Return on Investment: 29.40%
- Starting Bankroll: $1,000
- Final Bankroll: $11,122
- Profit: $10,122

**Step 5: Value Bet Identification**

The model identifies value bets by comparing its predictions to bookmaker odds:

1. **Convert odds to probabilities**
   - Bookmaker odds of 2.50 = 40% implied probability
   
2. **Remove bookmaker margin** (vig)
   - Bookmakers build in profit margin
   - Model removes this to get true market probability
   
3. **Compare to model prediction**
   - If model predicts 50% but market says 40%, there's a 10% edge
   
4. **Apply value threshold**
   - Only bet if edge exceeds 5% (configurable)
   - Ensures statistical significance

5. **Calculate stake size using Kelly Criterion**
   - Mathematical formula for optimal bet sizing
   - Prevents betting too much and going broke
   - Uses fractional Kelly (10%) for safety

### Why This Works

**Market Inefficiency**

Lower-tier football leagues are less analyzed than top leagues:
- Fewer professional analysts
- Less media coverage
- Bookmakers have less information
- More pricing errors

**Statistical Edge**

The model achieves 48% accuracy on a three-way outcome (Home/Draw/Away):
- Random guessing: 33% accuracy
- Model performance: 48% accuracy
- 45% improvement over random

More importantly, the model's **probability estimates are well-calibrated**:
- When it says 60% chance of home win, home teams win approximately 60% of the time
- This calibration is measured by Brier score (0.209, lower is better)

**Risk Management**

The model uses conservative stake sizing:
- Maximum 5% of starting bankroll per bet
- Fractional Kelly Criterion (10% of full Kelly)
- Never risks more than 50% of current bankroll
- Stops betting if bankroll drops too low

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup Instructions

Navigate to the project directory:
```bash
cd c:\Users\xeij\Desktop\code\sportsbetting_model
```

Install required Python packages:
```bash
pip install -r requirements.txt
```

The installation includes:
- pandas and numpy for data processing
- scikit-learn for baseline models
- xgboost and lightgbm for gradient boosting
- matplotlib and seaborn for visualizations
- requests for data downloading
- pyyaml for configuration management

### Verify Installation

Test that the installation was successful:
```bash
python -c "import pandas, xgboost, lightgbm; print('Installation successful')"
```

## Usage

The system provides both a graphical user interface and a command-line interface. The GUI is recommended for most users.

### Graphical User Interface (Recommended)

The GUI provides a simple, visual way to use all features of the betting model without needing command-line knowledge.

**Launch the GUI:**
```bash
python gui.py
```

**Interface Overview:**

The GUI features a dark modern theme with two main panels:

**Left Panel - Actions:**
- Download Data: Get 16 seasons of historical matches with one click
- Train Models: Train XGBoost and LightGBM models (15-20 minutes)
- Run Backtest: See profitability results (29% ROI demonstration)
- Fetch Live Odds: Get current betting odds from The Odds API
- Find Value Bets: Identify profitable opportunities on upcoming matches

**Right Panel - Output:**
- Live terminal-style output display
- Green text on black background
- Scrollable output area
- Real-time progress updates

**Bottom Section:**
- API key input field for The Odds API
- Clickable link to get free API key
- Status bar showing current operation

**Using the GUI:**

1. **First Time Setup:**
   - Click "Download Data" button
   - Wait for download to complete (shows progress in output panel)
   - Click "Train Models" button
   - Wait 15-20 minutes for training to complete
   - Models are now ready to use

2. **Finding Value Bets:**
   - Enter your Odds API key in the text field at bottom left
   - Click "Fetch Live Odds" button
   - Wait for odds to be fetched (shows number of matches found)
   - Click "Find Value Bets" button
   - View recommended bets in output panel

3. **Viewing Backtest Results:**
   - Click "Run Backtest" button
   - See ROI, win rate, and profitability metrics
   - Results also saved to data/visualizations/backtest_report.txt

**Benefits of GUI:**
- No command-line knowledge required
- Visual feedback for all operations
- Easy API key management
- All features accessible from one window
- Suitable for non-technical users
- Clean, professional interface

### Command-Line Interface (Advanced Users)

#### Download Historical Data

Download match data from football-data.co.uk:
```bash
python main.py --download
```

This downloads 16 seasons of data (2010/11 through 2025/26) for all four divisions, totaling approximately 18,000 matches.

#### Train Models

Train the prediction models on downloaded data:
```bash
python main.py --train
```

This process:
- Engineers 65 features from raw match data
- Splits data chronologically (85% training, 15% testing)
- Trains three models: Logistic Regression, XGBoost, and LightGBM
- Generates evaluation metrics and visualizations
- Saves trained models to the models directory

Expected training time: 15-25 minutes depending on hardware.

#### Run Backtesting

Simulate betting strategy on historical data:
```bash
python main.py --backtest
```

This evaluates model performance by simulating bets on the test set using Kelly Criterion for stake sizing. Results include ROI, win rate, and detailed bet history.

#### Fetch Live Odds

Get current betting odds from The Odds API:
```bash
python main.py --fetch-odds
```

Requirements:
- Free API key from the-odds-api.com (500 requests/month)
- Set environment variable: `$env:ODDS_API_KEY="your_key_here"`
- Or pass directly: `python main.py --fetch-odds --api-key YOUR_KEY`

This fetches upcoming matches for:
- Premier League (soccer_epl)
- League One (soccer_england_league1)
- League Two (soccer_england_league2)

Each fetch uses 3 API requests (one per league).

#### Generate Predictions

Identify value bets on fetched fixtures:
```bash
python main.py --predict
```

This:
- Loads your trained model
- Compares model predictions vs live odds
- Identifies value betting opportunities
- Shows recommended bets with edge percentage

#### Complete Pipeline

Run all steps sequentially:
```bash
python main.py --all
```

#### Model Selection

Specify which trained model to use:
```bash
python main.py --backtest --model xgboost_model
python main.py --predict --model lightgbm_model
```

Available models: baseline_model, xgboost_model, lightgbm_model

## Live Odds Integration

### Setup (One-Time)

1. **Get Free API Key**

Visit: https://the-odds-api.com/
- Sign up for a free account
- Get your API key (500 requests/month free)
- Copy the key

2. **Set API Key**

Windows PowerShell:
```powershell
$env:ODDS_API_KEY="your_api_key_here"
```

Or pass directly in command:
```bash
python main.py --fetch-odds --api-key your_api_key_here
```

### Workflow for Finding Value Bets

**Step 1: Fetch Live Odds**
```bash
python main.py --fetch-odds
```

This will:
- Fetch upcoming matches for Premier League, League One, and League Two
- Get current odds from multiple bookmakers
- Save to data/live_odds.csv

**Step 2: Find Value Bets**
```bash
python main.py --predict
```

This will:
- Load your trained model
- Compare model predictions vs live odds
- Identify value betting opportunities
- Show recommended bets with edge percentage

### API Usage Limits

**Free Tier:**
- 500 requests per month
- Each fetch uses 3 requests (one per league)
- Approximately 166 fetches per month
- Recommended: Fetch once per day for best results

**Typical Monthly Usage:**
- Daily fetches: 90 requests
- Testing/debugging: 50 requests
- Total: 140/500 requests (plenty of headroom)

### Supported Leagues

- Premier League (E0) - Available on Polymarket/Kalshi
- League One (E2)
- League Two (E3)
- National League (EC) - Limited API coverage

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
- Train/test split ratio (default: 85/15)
- XGBoost hyperparameters:
  - n_estimators: 300
  - max_depth: 8
  - learning_rate: 0.03
- LightGBM hyperparameters:
  - n_estimators: 300
  - max_depth: 8
  - learning_rate: 0.03

**Backtesting:**
- Value threshold for bet identification (default: 5%)
- Kelly fraction for stake sizing (default: 0.10)
- Starting bankroll for simulation (default: $1,000)
- Odds range filters (min: 1.5, max: 10.0)

## Understanding the Output

### Model Evaluation Metrics

**Accuracy:**
Percentage of correct predictions. For three-way outcomes (Home/Draw/Away), random guessing achieves 33%. The model achieves approximately 48%.

**Log Loss:**
Measures how well the predicted probabilities match actual outcomes. Lower is better. Typical values: 1.0-1.1.

**Brier Score:**
Measures probability calibration. Ranges from 0 (perfect) to 1 (worst). The model achieves approximately 0.21.

### Backtest Report

**Overall Metrics:**
- Total Bets: Number of value bets identified
- Total Staked: Sum of all bet amounts
- Total Profit: Net profit/loss
- ROI: Return on investment percentage
- Win Rate: Percentage of winning bets
- Average Odds: Mean odds of all bets placed
- Final Bankroll: Ending balance

**Bet Type Breakdown:**
Separate statistics for Home Win, Draw, and Away Win bets showing which outcome types are most profitable.

### Value Bet Predictions

For each upcoming match, the output shows:
- Match details (date, teams)
- Model probabilities (Home/Draw/Away)
- Market odds
- Value bet recommendation (if edge exceeds threshold)
- Expected edge percentage
- Expected value

## Data Source

Historical data is sourced from football-data.co.uk, which provides free CSV files containing:

- Match results (full-time home goals, full-time away goals)
- Closing odds from multiple bookmakers
- Match statistics (shots, corners, cards when available)

**Divisions:**
- E0: Premier League (English top tier)
- E2: League One (English third tier)
- E3: League Two (English fourth tier)
- EC: National League (English fifth tier)

Data availability varies by season. Older seasons may have limited statistical features.

## Technical Notes

### Chronological Validation

The system uses strict time-based train/test splits. Models are trained on earlier seasons and tested on recent seasons to simulate real-world deployment. This prevents data leakage where the model learns from future information.

### Class Imbalance

Draws are less frequent than home or away wins (approximately 26% vs 43% home wins vs 31% away wins). Models use class weighting to handle this imbalance and avoid biasing predictions toward more common outcomes.

### Market Efficiency

Lower-tier leagues may exhibit more pricing inefficiencies than top-tier leagues, potentially offering more value betting opportunities. The Premier League is included for comparison and because it's available on prediction markets like Polymarket and Kalshi.

### Vig Removal

Bookmakers build a profit margin (vig or overround) into their odds. The system removes this margin to estimate true probabilities for comparison with model predictions. Two methods are available: margin proportional and power method.

### Stake Sizing

The Kelly Criterion is a mathematical formula that calculates optimal bet size based on:
- Your edge (model probability - market probability)
- The odds offered
- Your current bankroll

Fractional Kelly (10% of full Kelly) is used for safety, reducing variance and risk of ruin.

## Troubleshooting

**Import Errors:**
Ensure all dependencies are installed: `pip install -r requirements.txt`

**Data Download Failures:**
Check internet connection. The system retries failed downloads automatically.

**Memory Issues:**
Feature engineering processes all matches sequentially. For very large datasets, consider reducing the number of seasons in config.yaml.

**Long Training Times:**
Training 300-estimator gradient boosting models on 18,000+ matches takes 15-25 minutes. This is expected behavior.

**No Live Odds Available:**
Ensure API key is set correctly. Check that there are upcoming matches scheduled. Verify API request limit has not been exceeded.
