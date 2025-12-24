"""
Prediction module for identifying value bets.

This module handles loading models and predicting on new fixtures,
identifying value betting opportunities.
"""

from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

from .utils import load_config, print_progress, odds_to_prob, calculate_expected_value, remove_vig
from .modeling import load_model


def predict_match(model: Any, match_features: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    """
    Predict probabilities for a single match.
    
    Args:
        model: Trained model
        match_features: DataFrame with match features (single row)
        feature_cols: List of feature column names
        
    Returns:
        Dictionary with predicted probabilities for each outcome
    """
    # Ensure we have all required features
    X = match_features[feature_cols]
    
    # Get predictions
    probs = model.predict_proba(X)[0]
    
    # Map to outcomes
    if hasattr(model, 'label_encoder'):
        classes = model.label_encoder.classes_
    else:
        classes = model.classes_
    
    result = {}
    for cls, prob in zip(classes, probs):
        if cls == 'H':
            result['home_win'] = prob
        elif cls == 'D':
            result['draw'] = prob
        elif cls == 'A':
            result['away_win'] = prob
    
    return result


def identify_value_bets(model_probs: Dict[str, float], market_odds: Dict[str, float],
                       value_threshold: float = 0.05, vig_method: str = "margin") -> List[Dict]:
    """
    Identify value betting opportunities.
    
    Args:
        model_probs: Model's predicted probabilities
        market_odds: Market odds for each outcome
        value_threshold: Minimum edge required to bet
        vig_method: Method for removing vig
        
    Returns:
        List of value bet dictionaries
    """
    value_bets = []
    
    # Remove vig from market odds
    if all(k in market_odds for k in ['home_win', 'draw', 'away_win']):
        clean_probs = remove_vig(
            market_odds['home_win'],
            market_odds['draw'],
            market_odds['away_win'],
            method=vig_method
        )
        
        market_true_probs = {
            'home_win': clean_probs[0],
            'draw': clean_probs[1],
            'away_win': clean_probs[2]
        }
    else:
        # Fallback to implied probabilities
        market_true_probs = {
            k: odds_to_prob(v) for k, v in market_odds.items()
        }
    
    # Check each outcome for value
    for outcome in ['home_win', 'draw', 'away_win']:
        if outcome not in model_probs or outcome not in market_odds:
            continue
        
        model_prob = model_probs[outcome]
        odds = market_odds[outcome]
        market_prob = market_true_probs[outcome]
        
        # Calculate edge
        edge = model_prob - market_prob
        
        # Calculate expected value
        ev = calculate_expected_value(model_prob, odds)
        
        # Check if it's a value bet
        if edge > value_threshold and ev > 0:
            value_bets.append({
                'outcome': outcome,
                'model_prob': model_prob,
                'market_prob': market_prob,
                'odds': odds,
                'edge': edge,
                'expected_value': ev
            })
    
    return value_bets


def predict_upcoming_fixtures(fixtures_df: pd.DataFrame, model_name: str = "xgboost_model",
                              config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Predict on upcoming fixtures and identify value bets.
    
    Args:
        fixtures_df: DataFrame with upcoming fixtures (must have features engineered)
        model_name: Name of saved model to use
        config_path: Path to configuration file
        
    Returns:
        DataFrame with predictions and value bets
    """
    config = load_config(config_path)
    value_threshold = config['backtest']['value_threshold']
    vig_method = config['backtest']['vig_method']
    bookmakers = config['prediction']['bookmakers']
    
    # Load model
    model, feature_cols = load_model(model_name, config_path)
    
    print_progress(f"Predicting on {len(fixtures_df)} fixtures...")
    
    results = []
    
    for idx, row in fixtures_df.iterrows():
        # Get predictions
        match_features = fixtures_df.iloc[[idx]]
        model_probs = predict_match(model, match_features, feature_cols)
        
        # Get market odds (try bookmakers in order of preference)
        market_odds = {}
        for bookmaker in bookmakers:
            home_col = f"{bookmaker}H"
            draw_col = f"{bookmaker}D"
            away_col = f"{bookmaker}A"
            
            if all(col in row.index for col in [home_col, draw_col, away_col]):
                if pd.notna(row[home_col]) and pd.notna(row[draw_col]) and pd.notna(row[away_col]):
                    market_odds = {
                        'home_win': row[home_col],
                        'draw': row[draw_col],
                        'away_win': row[away_col]
                    }
                    break
        
        if not market_odds:
            print_progress(f"No odds available for match {idx}", "WARNING")
            continue
        
        # Identify value bets
        value_bets = identify_value_bets(model_probs, market_odds, value_threshold, vig_method)
        
        # Store results
        result = {
            'Date': row.get('Date', 'Unknown'),
            'HomeTeam': row.get('HomeTeam', 'Unknown'),
            'AwayTeam': row.get('AwayTeam', 'Unknown'),
            'Model_Home': model_probs.get('home_win', 0),
            'Model_Draw': model_probs.get('draw', 0),
            'Model_Away': model_probs.get('away_win', 0),
            'Odds_Home': market_odds.get('home_win', 0),
            'Odds_Draw': market_odds.get('draw', 0),
            'Odds_Away': market_odds.get('away_win', 0),
            'Value_Bets': len(value_bets),
            'Best_Bet': value_bets[0]['outcome'] if value_bets else None,
            'Best_Bet_Edge': value_bets[0]['edge'] if value_bets else 0,
            'Best_Bet_EV': value_bets[0]['expected_value'] if value_bets else 0
        }
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    print_progress(f"Predictions complete. {len(results_df[results_df['Value_Bets'] > 0])} matches with value bets.")
    
    return results_df


def format_prediction_output(predictions_df: pd.DataFrame) -> str:
    """
    Format predictions for display.
    
    Args:
        predictions_df: DataFrame with predictions
        
    Returns:
        Formatted string
    """
    output = []
    output.append("=" * 80)
    output.append("UPCOMING FIXTURES - VALUE BET PREDICTIONS")
    output.append("=" * 80)
    
    for idx, row in predictions_df.iterrows():
        output.append(f"\n{row['Date']} - {row['HomeTeam']} vs {row['AwayTeam']}")
        output.append("-" * 80)
        
        output.append(f"Model Probabilities:")
        output.append(f"  Home: {row['Model_Home']:.2%}  Draw: {row['Model_Draw']:.2%}  Away: {row['Model_Away']:.2%}")
        
        output.append(f"Market Odds:")
        output.append(f"  Home: {row['Odds_Home']:.2f}  Draw: {row['Odds_Draw']:.2f}  Away: {row['Odds_Away']:.2f}")
        
        if row['Value_Bets'] > 0:
            output.append(f"\n*** VALUE BET DETECTED ***")
            output.append(f"  Bet: {row['Best_Bet'].replace('_', ' ').title()}")
            output.append(f"  Edge: {row['Best_Bet_Edge']:.2%}")
            output.append(f"  Expected Value: {row['Best_Bet_EV']:.2%}")
        else:
            output.append(f"\nNo value bets identified.")
    
    output.append("\n" + "=" * 80)
    
    return "\n".join(output)
