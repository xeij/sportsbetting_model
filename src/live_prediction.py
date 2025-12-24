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
    
    # Load historical data with features (already engineered)
    features_path = config['data']['processed_data_path'].replace('.csv', '_features.csv')
    
    if not os.path.exists(features_path):
        print_progress("Historical data not found. Cannot engineer features.", "ERROR")
        return None
    
    # Load historical data (already has features)
    historical_df = pd.read_csv(features_path, parse_dates=['Date'])
    
    # Prepare live odds to match historical format
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']
    
    if not all(col in live_odds_df.columns for col in required_cols):
        print_progress(f"Live odds missing required columns. Has: {live_odds_df.columns.tolist()}", "ERROR")
        return None
    
    # Add dummy result columns (will be ignored in prediction)
    live_odds_df['FTHG'] = 0
    live_odds_df['FTAG'] = 0
    live_odds_df['FTR'] = 'D'
    
    # Store original odds columns to preserve them
    odds_columns = ['B365H', 'B365D', 'B365A']
    live_odds_original = live_odds_df[['Date', 'HomeTeam', 'AwayTeam'] + odds_columns].copy()
    
    # Engineer features ONLY for the new live matches using historical context
    from .feature_engineering import calculate_team_form, calculate_head_to_head, calculate_rest_days, add_odds_features
    
    print_progress(f"Engineering features for {len(live_odds_df)} live matches...")
    
    # Initialize feature columns
    live_fixtures_list = []
    
    for idx, row in live_odds_df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        date = row['Date']
        
        match_features = {
            'Date': date,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'B365H': row['B365H'],
            'B365D': row['B365D'],
            'B365A': row['B365A'],
            'FTHG': 0,
            'FTAG': 0,
            'FTR': 'D'
        }
        
        # Calculate form features using historical data
        for window in [5, 10]:
            # Home team form
            home_form = calculate_team_form(historical_df, home_team, date, window, is_home=True)
            for stat in ['points', 'goals_scored', 'goals_conceded', 'wins', 'draws', 'losses']:
                match_features[f'home_form_{window}_{stat}'] = home_form.get(stat, 0)
            
            home_form_all = calculate_team_form(historical_df, home_team, date, window, is_home=None)
            for stat in ['points', 'goals_scored', 'goals_conceded', 'wins', 'draws', 'losses']:
                match_features[f'home_form_{window}_all_{stat}'] = home_form_all.get(stat, 0)
            
            # Away team form
            away_form = calculate_team_form(historical_df, away_team, date, window, is_home=False)
            for stat in ['points', 'goals_scored', 'goals_conceded', 'wins', 'draws', 'losses']:
                match_features[f'away_form_{window}_{stat}'] = away_form.get(stat, 0)
            
            away_form_all = calculate_team_form(historical_df, away_team, date, window, is_home=None)
            for stat in ['points', 'goals_scored', 'goals_conceded', 'wins', 'draws', 'losses']:
                match_features[f'away_form_{window}_all_{stat}'] = away_form_all.get(stat, 0)
        
        # Head-to-head
        h2h = calculate_head_to_head(historical_df, home_team, away_team, date, 5)
        for stat, value in h2h.items():
            match_features[stat] = value
        
        # Rest days
        match_features['home_rest_days'] = calculate_rest_days(historical_df, home_team, date)
        match_features['away_rest_days'] = calculate_rest_days(historical_df, away_team, date)
        
        # Goal difference
        home_form_10 = calculate_team_form(historical_df, home_team, date, 10, is_home=None)
        away_form_10 = calculate_team_form(historical_df, away_team, date, 10, is_home=None)
        
        if home_form_10['matches_played'] > 0:
            match_features['home_goal_diff'] = (home_form_10['goals_scored'] - home_form_10['goals_conceded']) / home_form_10['matches_played']
        else:
            match_features['home_goal_diff'] = 0
        
        if away_form_10['matches_played'] > 0:
            match_features['away_goal_diff'] = (away_form_10['goals_scored'] - away_form_10['goals_conceded']) / away_form_10['matches_played']
        else:
            match_features['away_goal_diff'] = 0
        
        live_fixtures_list.append(match_features)
    
    # Convert to DataFrame
    live_fixtures = pd.DataFrame(live_fixtures_list)
    
    # Add odds features
    live_fixtures = add_odds_features(live_fixtures, 'B365')
    
    print_progress(f"Prepared {len(live_fixtures)} live fixtures for prediction")
    print_progress(f"Columns with B365: {[c for c in live_fixtures.columns if 'B365' in c]}")
    
    return live_fixtures
