"""
Feature engineering module for creating predictive features.

This module handles all feature creation including form, home/away stats,
head-to-head records, and odds-based features.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import timedelta

from .utils import load_config, print_progress, odds_to_prob, remove_vig


def calculate_team_form(df: pd.DataFrame, team: str, date: pd.Timestamp, 
                        window: int = 5, is_home: bool = None) -> Dict[str, float]:
    """
    Calculate team's recent form statistics.
    
    Args:
        df: DataFrame with match data
        team: Team name
        date: Current match date
        window: Number of previous games to consider
        is_home: If True, only home games; if False, only away games; if None, all games
        
    Returns:
        Dictionary with form statistics
    """
    # Get previous matches for this team
    team_matches = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
        (df['Date'] < date)
    ].tail(window)
    
    if len(team_matches) == 0:
        return {
            'points': 0.0,
            'goals_scored': 0.0,
            'goals_conceded': 0.0,
            'wins': 0.0,
            'draws': 0.0,
            'losses': 0.0,
            'matches_played': 0
        }
    
    points = 0
    goals_scored = 0
    goals_conceded = 0
    wins = 0
    draws = 0
    losses = 0
    
    for _, match in team_matches.iterrows():
        is_home_team = match['HomeTeam'] == team
        
        # Filter by home/away if specified
        if is_home is not None and is_home != is_home_team:
            continue
        
        if is_home_team:
            gf = match['FTHG']
            ga = match['FTAG']
        else:
            gf = match['FTAG']
            ga = match['FTHG']
        
        goals_scored += gf
        goals_conceded += ga
        
        if gf > ga:
            points += 3
            wins += 1
        elif gf == ga:
            points += 1
            draws += 1
        else:
            losses += 1
    
    matches_played = len(team_matches)
    
    return {
        'points': points,
        'goals_scored': goals_scored,
        'goals_conceded': goals_conceded,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'matches_played': matches_played
    }


def calculate_head_to_head(df: pd.DataFrame, home_team: str, away_team: str, 
                          date: pd.Timestamp, lookback: int = 5) -> Dict[str, float]:
    """
    Calculate head-to-head statistics between two teams.
    
    Args:
        df: DataFrame with match data
        home_team: Home team name
        away_team: Away team name
        date: Current match date
        lookback: Number of previous meetings to consider
        
    Returns:
        Dictionary with H2H statistics
    """
    # Get previous meetings
    h2h_matches = df[
        (((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
         ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))) &
        (df['Date'] < date)
    ].tail(lookback)
    
    if len(h2h_matches) == 0:
        return {
            'h2h_home_wins': 0,
            'h2h_draws': 0,
            'h2h_away_wins': 0,
            'h2h_home_goals_avg': 0.0,
            'h2h_away_goals_avg': 0.0,
            'h2h_matches': 0
        }
    
    home_wins = 0
    draws = 0
    away_wins = 0
    home_goals_total = 0
    away_goals_total = 0
    
    for _, match in h2h_matches.iterrows():
        if match['HomeTeam'] == home_team:
            hg = match['FTHG']
            ag = match['FTAG']
        else:
            hg = match['FTAG']
            ag = match['FTHG']
        
        home_goals_total += hg
        away_goals_total += ag
        
        if hg > ag:
            home_wins += 1
        elif hg == ag:
            draws += 1
        else:
            away_wins += 1
    
    matches = len(h2h_matches)
    
    return {
        'h2h_home_wins': home_wins,
        'h2h_draws': draws,
        'h2h_away_wins': away_wins,
        'h2h_home_goals_avg': home_goals_total / matches,
        'h2h_away_goals_avg': away_goals_total / matches,
        'h2h_matches': matches
    }


def calculate_rest_days(df: pd.DataFrame, team: str, date: pd.Timestamp) -> int:
    """
    Calculate days since team's last match.
    
    Args:
        df: DataFrame with match data
        team: Team name
        date: Current match date
        
    Returns:
        Number of days since last match
    """
    previous_matches = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
        (df['Date'] < date)
    ]
    
    if len(previous_matches) == 0:
        return 7  # Default to 1 week
    
    last_match_date = previous_matches['Date'].max()
    rest_days = (date - last_match_date).days
    
    return rest_days


def add_odds_features(df: pd.DataFrame, bookmaker: str = 'B365') -> pd.DataFrame:
    """
    Add features derived from betting odds.
    
    Args:
        df: DataFrame with match data
        bookmaker: Bookmaker prefix to use
        
    Returns:
        DataFrame with odds features added
    """
    df = df.copy()
    
    home_col = f"{bookmaker}H"
    draw_col = f"{bookmaker}D"
    away_col = f"{bookmaker}A"
    
    # Check if columns exist
    if not all(col in df.columns for col in [home_col, draw_col, away_col]):
        print_progress(f"Bookmaker {bookmaker} odds not available", "WARNING")
        return df
    
    # Calculate implied probabilities
    df[f'{bookmaker}_prob_home'] = df[home_col].apply(odds_to_prob)
    df[f'{bookmaker}_prob_draw'] = df[draw_col].apply(odds_to_prob)
    df[f'{bookmaker}_prob_away'] = df[away_col].apply(odds_to_prob)
    
    # Calculate overround (bookmaker margin)
    df[f'{bookmaker}_overround'] = (
        df[f'{bookmaker}_prob_home'] + 
        df[f'{bookmaker}_prob_draw'] + 
        df[f'{bookmaker}_prob_away']
    )
    
    # Remove vig to get true probabilities
    for idx, row in df.iterrows():
        if pd.notna(row[home_col]) and pd.notna(row[draw_col]) and pd.notna(row[away_col]):
            clean_probs = remove_vig(row[home_col], row[draw_col], row[away_col])
            df.at[idx, f'{bookmaker}_true_prob_home'] = clean_probs[0]
            df.at[idx, f'{bookmaker}_true_prob_draw'] = clean_probs[1]
            df.at[idx, f'{bookmaker}_true_prob_away'] = clean_probs[2]
    
    return df


def engineer_features(df: pd.DataFrame, config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Create all features for the dataset.
    
    Args:
        df: DataFrame with match data
        config_path: Path to configuration file
        
    Returns:
        DataFrame with engineered features
    """
    config = load_config(config_path)
    form_windows = config['features']['form_windows']
    h2h_lookback = config['features']['h2h_lookback']
    
    print_progress("Engineering features...")
    
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Initialize feature columns
    feature_cols = []
    
    # Form features for different windows
    for window in form_windows:
        for prefix, is_home in [('home', True), ('away', False)]:
            for stat in ['points', 'goals_scored', 'goals_conceded', 'wins', 'draws', 'losses']:
                col_name = f'{prefix}_form_{window}_{stat}'
                df[col_name] = 0.0
                feature_cols.append(col_name)
            
            # Overall form (all matches)
            col_name = f'{prefix}_form_{window}_all_{stat}'
            df[col_name] = 0.0
            feature_cols.append(col_name)
    
    # Head-to-head features
    h2h_features = ['h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 
                    'h2h_home_goals_avg', 'h2h_away_goals_avg', 'h2h_matches']
    for feat in h2h_features:
        df[feat] = 0.0
        feature_cols.append(feat)
    
    # Rest days
    df['home_rest_days'] = 7
    df['away_rest_days'] = 7
    feature_cols.extend(['home_rest_days', 'away_rest_days'])
    
    # Goal difference
    df['home_goal_diff'] = 0.0
    df['away_goal_diff'] = 0.0
    feature_cols.extend(['home_goal_diff', 'away_goal_diff'])
    
    # Calculate features for each match
    print_progress(f"Calculating features for {len(df)} matches...")
    
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print_progress(f"Processing match {idx}/{len(df)}")
        
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        date = row['Date']
        
        # Form features
        for window in form_windows:
            # Home team form (home matches only)
            home_form = calculate_team_form(df.iloc[:idx], home_team, date, window, is_home=True)
            for stat, value in home_form.items():
                if stat != 'matches_played':
                    df.at[idx, f'home_form_{window}_{stat}'] = value
            
            # Home team overall form
            home_form_all = calculate_team_form(df.iloc[:idx], home_team, date, window, is_home=None)
            for stat, value in home_form_all.items():
                if stat != 'matches_played':
                    df.at[idx, f'home_form_{window}_all_{stat}'] = value
            
            # Away team form (away matches only)
            away_form = calculate_team_form(df.iloc[:idx], away_team, date, window, is_home=False)
            for stat, value in away_form.items():
                if stat != 'matches_played':
                    df.at[idx, f'away_form_{window}_{stat}'] = value
            
            # Away team overall form
            away_form_all = calculate_team_form(df.iloc[:idx], away_team, date, window, is_home=None)
            for stat, value in away_form_all.items():
                if stat != 'matches_played':
                    df.at[idx, f'away_form_{window}_all_{stat}'] = value
        
        # Head-to-head
        h2h = calculate_head_to_head(df.iloc[:idx], home_team, away_team, date, h2h_lookback)
        for stat, value in h2h.items():
            df.at[idx, stat] = value
        
        # Rest days
        df.at[idx, 'home_rest_days'] = calculate_rest_days(df.iloc[:idx], home_team, date)
        df.at[idx, 'away_rest_days'] = calculate_rest_days(df.iloc[:idx], away_team, date)
        
        # Goal difference (from overall form)
        home_form_all = calculate_team_form(df.iloc[:idx], home_team, date, 10, is_home=None)
        away_form_all = calculate_team_form(df.iloc[:idx], away_team, date, 10, is_home=None)
        
        if home_form_all['matches_played'] > 0:
            df.at[idx, 'home_goal_diff'] = (
                home_form_all['goals_scored'] - home_form_all['goals_conceded']
            ) / home_form_all['matches_played']
        
        if away_form_all['matches_played'] > 0:
            df.at[idx, 'away_goal_diff'] = (
                away_form_all['goals_scored'] - away_form_all['goals_conceded']
            ) / away_form_all['matches_played']
    
    # Add odds features
    df = add_odds_features(df, 'B365')
    
    # Create target variable (1X2)
    df['Result'] = 'D'  # Draw
    df.loc[df['FTHG'] > df['FTAG'], 'Result'] = 'H'  # Home win
    df.loc[df['FTHG'] < df['FTAG'], 'Result'] = 'A'  # Away win
    
    print_progress(f"Feature engineering complete. {len(feature_cols)} features created.")
    
    return df
