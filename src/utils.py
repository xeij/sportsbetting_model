"""
Utility functions for the betting model.

This module provides helper functions for odds conversion, vig removal,
Kelly criterion calculation, and configuration loading.
"""

from typing import Dict, List, Tuple
import yaml
import numpy as np


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def odds_to_prob(odds: float) -> float:
    """
    Convert decimal odds to implied probability.
    
    Args:
        odds: Decimal odds (e.g., 2.5)
        
    Returns:
        Implied probability (0 to 1)
    """
    if odds <= 1.0:
        return 0.0
    return 1.0 / odds


def remove_vig_margin(probs: List[float]) -> List[float]:
    """
    Remove bookmaker margin (vig) from probabilities using margin method.
    
    This normalizes the probabilities so they sum to 1.0.
    
    Args:
        probs: List of implied probabilities
        
    Returns:
        List of probabilities with vig removed
    """
    total = sum(probs)
    if total == 0:
        return probs
    return [p / total for p in probs]


def remove_vig_power(probs: List[float], k: float = 1.0) -> List[float]:
    """
    Remove bookmaker margin using power method (Shin's method).
    
    Args:
        probs: List of implied probabilities
        k: Power parameter (typically between 0.5 and 2.0)
        
    Returns:
        List of probabilities with vig removed
    """
    powered = [p ** k for p in probs]
    total = sum(powered)
    if total == 0:
        return probs
    return [p / total for p in powered]


def remove_vig(home_odds: float, draw_odds: float, away_odds: float, 
               method: str = "margin") -> Tuple[float, float, float]:
    """
    Remove vig from 1X2 odds and return true probabilities.
    
    Args:
        home_odds: Decimal odds for home win
        draw_odds: Decimal odds for draw
        away_odds: Decimal odds for away win
        method: 'margin' or 'power'
        
    Returns:
        Tuple of (home_prob, draw_prob, away_prob) with vig removed
    """
    # Convert to implied probabilities
    probs = [odds_to_prob(home_odds), odds_to_prob(draw_odds), odds_to_prob(away_odds)]
    
    # Remove vig
    if method == "power":
        clean_probs = remove_vig_power(probs)
    else:
        clean_probs = remove_vig_margin(probs)
    
    return tuple(clean_probs)


def kelly_criterion(prob: float, odds: float, fraction: float = 1.0) -> float:
    """
    Calculate Kelly criterion bet size.
    
    Args:
        prob: Estimated probability of winning (0 to 1)
        odds: Decimal odds offered
        fraction: Fraction of Kelly to use (e.g., 0.25 for quarter Kelly)
        
    Returns:
        Fraction of bankroll to bet (0 to 1)
    """
    if odds <= 1.0 or prob <= 0.0 or prob >= 1.0:
        return 0.0
    
    # Kelly formula: f = (bp - q) / b
    # where b = odds - 1, p = prob, q = 1 - prob
    b = odds - 1.0
    q = 1.0 - prob
    
    kelly = (b * prob - q) / b
    
    # Apply fraction and ensure non-negative
    kelly = max(0.0, kelly * fraction)
    
    # Cap at reasonable maximum (e.g., 25% of bankroll)
    kelly = min(kelly, 0.25)
    
    return kelly


def calculate_expected_value(model_prob: float, odds: float) -> float:
    """
    Calculate expected value of a bet.
    
    Args:
        model_prob: Model's estimated probability
        odds: Decimal odds offered
        
    Returns:
        Expected value (positive means +EV)
    """
    return (model_prob * odds) - 1.0


def get_bookmaker_columns(df_columns: List[str], bookmaker: str = "B365") -> Dict[str, str]:
    """
    Get column names for a specific bookmaker's odds.
    
    Args:
        df_columns: List of all DataFrame columns
        bookmaker: Bookmaker prefix (e.g., 'B365', 'PS', 'Avg')
        
    Returns:
        Dictionary with keys 'home', 'draw', 'away' and column names as values
    """
    home_col = f"{bookmaker}H"
    draw_col = f"{bookmaker}D"
    away_col = f"{bookmaker}A"
    
    # Check if columns exist
    if home_col in df_columns and draw_col in df_columns and away_col in df_columns:
        return {'home': home_col, 'draw': draw_col, 'away': away_col}
    
    return {}


def standardize_team_name(team: str) -> str:
    """
    Standardize team names to handle variations across seasons.
    
    Args:
        team: Team name
        
    Returns:
        Standardized team name
    """
    # Convert to lowercase and strip whitespace
    team = team.lower().strip()
    
    # Common replacements
    replacements = {
        'afc': '',
        'fc': '',
        'utd': 'united',
        '&': 'and',
    }
    
    for old, new in replacements.items():
        team = team.replace(old, new)
    
    # Remove extra whitespace
    team = ' '.join(team.split())
    
    return team


def print_progress(message: str, level: str = "INFO") -> None:
    """
    Print formatted progress message.
    
    Args:
        message: Message to print
        level: Log level (INFO, WARNING, ERROR)
    """
    prefix = f"[{level}]"
    print(f"{prefix} {message}")
