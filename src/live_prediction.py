"""
Live prediction helper for processing live odds data.
"""

import pandas as pd
import os
from .utils import load_config, print_progress


def load_live_odds_for_prediction(config_path: str = "config.yaml"):
    """
    Load live odds and prepare them for prediction.
    
    Returns:
        DataFrame ready for prediction, or None if no live odds
    """
    live_odds_path = "data/live_odds.csv"
    
    if not os.path.exists(live_odds_path):
        return None
    
    config = load_config(config_path)
    
    print_progress(f"Loading live odds from {live_odds_path}...")
    
    # Load live odds
    live_odds_df = pd.read_csv(live_odds_path, parse_dates=['Date'])
    
    # Remove timezone info to match historical data (which is timezone-naive)
    if live_odds_df['Date'].dt.tz is not None:
        live_odds_df['Date'] = live_odds_df['Date'].dt.tz_localize(None)
    
    # Load historical data with features
    features_path = config['data']['processed_data_path'].replace('.csv', '_features.csv')
    
    if not os.path.exists(features_path):
        print_progress("Historical data not found. Cannot engineer features.", "ERROR")
        return None
    
    # Load historical data
    historical_df = pd.read_csv(features_path, parse_dates=['Date'])
    
    # Prepare live odds to match historical format
    # The live odds already have B365H, B365D, B365A columns
    # We just need to ensure they have the required base columns
    
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']
    
    # Check if live odds has all required columns
    if not all(col in live_odds_df.columns for col in required_cols):
        print_progress(f"Live odds missing required columns. Has: {live_odds_df.columns.tolist()}", "ERROR")
        return None
    
    # Add dummy result columns (will be ignored in prediction)
    live_odds_df['FTHG'] = 0
    live_odds_df['FTAG'] = 0
    live_odds_df['FTR'] = 'D'
    
    # Combine with historical for feature engineering context
    combined_df = pd.concat([historical_df, live_odds_df], ignore_index=True)
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    # Engineer features
    from .feature_engineering import engineer_features
    combined_with_features = engineer_features(combined_df, config_path)
    
    # Extract only the live matches (last N rows)
    live_fixtures = combined_with_features.tail(len(live_odds_df))
    
    print_progress(f"Prepared {len(live_fixtures)} live fixtures for prediction")
    
    return live_fixtures
