"""
Backtesting module for simulating betting strategies.

This module simulates betting on historical data to evaluate
the profitability of the model's predictions.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from .utils import load_config, print_progress, kelly_criterion, calculate_expected_value
from .prediction import predict_match, identify_value_bets


def simulate_betting(model, X_test: pd.DataFrame, y_test: pd.Series, 
                    test_df: pd.DataFrame, feature_cols: List[str],
                    config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Simulate betting strategy on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        test_df: Original test DataFrame with odds
        feature_cols: List of feature column names
        config_path: Path to configuration file
        
    Returns:
        DataFrame with bet history
    """
    config = load_config(config_path)
    
    value_threshold = config['backtest']['value_threshold']
    kelly_fraction = config['backtest']['kelly_fraction']
    starting_bankroll = config['backtest']['starting_bankroll']
    min_odds = config['backtest']['min_odds']
    max_odds = config['backtest']['max_odds']
    vig_method = config['backtest']['vig_method']
    
    print_progress("Simulating betting strategy...")
    
    bankroll = starting_bankroll
    bet_history = []
    
    for idx in range(len(X_test)):
        # Get match data
        match_features = X_test.iloc[[idx]]
        actual_result = y_test.iloc[idx]
        match_data = test_df.iloc[idx]
        
        # Get model predictions
        model_probs = predict_match(model, match_features, feature_cols)
        
        # Get market odds (use B365 as default)
        market_odds = {}
        if all(col in match_data.index for col in ['B365H', 'B365D', 'B365A']):
            if pd.notna(match_data['B365H']) and pd.notna(match_data['B365D']) and pd.notna(match_data['B365A']):
                market_odds = {
                    'home_win': match_data['B365H'],
                    'draw': match_data['B365D'],
                    'away_win': match_data['B365A']
                }
        
        if not market_odds:
            continue
        
        # Identify value bets
        value_bets = identify_value_bets(model_probs, market_odds, value_threshold, vig_method)
        
        # Filter by odds range
        value_bets = [
            bet for bet in value_bets 
            if min_odds <= bet['odds'] <= max_odds
        ]
        
        if not value_bets:
            continue
        
        # Take the best value bet
        best_bet = max(value_bets, key=lambda x: x['expected_value'])
        
        # Calculate stake using Kelly criterion
        kelly_stake = kelly_criterion(
            best_bet['model_prob'],
            best_bet['odds'],
            kelly_fraction
        )
        
        stake = bankroll * kelly_stake
        
        # Determine if bet won
        outcome_map = {
            'home_win': 'H',
            'draw': 'D',
            'away_win': 'A'
        }
        
        bet_outcome = outcome_map[best_bet['outcome']]
        won = (bet_outcome == actual_result)
        
        # Calculate profit/loss
        if won:
            profit = stake * (best_bet['odds'] - 1)
        else:
            profit = -stake
        
        # Update bankroll
        bankroll += profit
        
        # Record bet
        bet_record = {
            'Date': match_data.get('Date', 'Unknown'),
            'HomeTeam': match_data.get('HomeTeam', 'Unknown'),
            'AwayTeam': match_data.get('AwayTeam', 'Unknown'),
            'Bet': best_bet['outcome'],
            'Odds': best_bet['odds'],
            'Stake': stake,
            'Model_Prob': best_bet['model_prob'],
            'Edge': best_bet['edge'],
            'Result': actual_result,
            'Won': won,
            'Profit': profit,
            'Bankroll': bankroll
        }
        
        bet_history.append(bet_record)
    
    bet_df = pd.DataFrame(bet_history)
    
    print_progress(f"Simulation complete. {len(bet_df)} bets placed.")
    
    return bet_df


def calculate_roi(bet_history: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate return on investment metrics.
    
    Args:
        bet_history: DataFrame with bet history
        
    Returns:
        Dictionary of ROI metrics
    """
    if len(bet_history) == 0:
        return {
            'total_bets': 0,
            'total_staked': 0,
            'total_profit': 0,
            'roi': 0,
            'win_rate': 0,
            'avg_odds': 0,
            'final_bankroll': 0
        }
    
    total_bets = len(bet_history)
    total_staked = bet_history['Stake'].sum()
    total_profit = bet_history['Profit'].sum()
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
    win_rate = bet_history['Won'].mean() * 100
    avg_odds = bet_history['Odds'].mean()
    final_bankroll = bet_history['Bankroll'].iloc[-1]
    
    metrics = {
        'total_bets': total_bets,
        'total_staked': total_staked,
        'total_profit': total_profit,
        'roi': roi,
        'win_rate': win_rate,
        'avg_odds': avg_odds,
        'final_bankroll': final_bankroll
    }
    
    return metrics


def plot_roi_curve(bet_history: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot cumulative ROI over time.
    
    Args:
        bet_history: DataFrame with bet history
        save_path: Optional path to save plot
    """
    if len(bet_history) == 0:
        print_progress("No bets to plot", "WARNING")
        return
    
    print_progress("Generating ROI curve...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Cumulative profit
    bet_history['Cumulative_Profit'] = bet_history['Profit'].cumsum()
    
    axes[0].plot(bet_history.index, bet_history['Cumulative_Profit'], linewidth=2)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Bet Number')
    axes[0].set_ylabel('Cumulative Profit')
    axes[0].set_title('Cumulative Profit Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Bankroll
    axes[1].plot(bet_history.index, bet_history['Bankroll'], linewidth=2, color='green')
    axes[1].axhline(y=bet_history['Bankroll'].iloc[0], color='r', linestyle='--', alpha=0.5, label='Starting Bankroll')
    axes[1].set_xlabel('Bet Number')
    axes[1].set_ylabel('Bankroll')
    axes[1].set_title('Bankroll Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print_progress(f"ROI curve saved to {save_path}")
    
    plt.close()


def generate_backtest_report(bet_history: pd.DataFrame, metrics: Dict[str, float],
                            save_path: str = None) -> str:
    """
    Generate detailed backtest report.
    
    Args:
        bet_history: DataFrame with bet history
        metrics: Dictionary of ROI metrics
        save_path: Optional path to save report
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("BACKTEST REPORT")
    report.append("=" * 80)
    report.append("")
    
    report.append("OVERALL METRICS:")
    report.append(f"  Total Bets: {metrics['total_bets']}")
    report.append(f"  Total Staked: ${metrics['total_staked']:.2f}")
    report.append(f"  Total Profit: ${metrics['total_profit']:.2f}")
    report.append(f"  ROI: {metrics['roi']:.2f}%")
    report.append(f"  Win Rate: {metrics['win_rate']:.2f}%")
    report.append(f"  Average Odds: {metrics['avg_odds']:.2f}")
    report.append(f"  Final Bankroll: ${metrics['final_bankroll']:.2f}")
    report.append("")
    
    if len(bet_history) > 0:
        report.append("BET TYPE BREAKDOWN:")
        for bet_type in ['home_win', 'draw', 'away_win']:
            type_bets = bet_history[bet_history['Bet'] == bet_type]
            if len(type_bets) > 0:
                type_profit = type_bets['Profit'].sum()
                type_staked = type_bets['Stake'].sum()
                type_roi = (type_profit / type_staked * 100) if type_staked > 0 else 0
                type_win_rate = type_bets['Won'].mean() * 100
                
                report.append(f"  {bet_type.replace('_', ' ').title()}:")
                report.append(f"    Bets: {len(type_bets)}")
                report.append(f"    Profit: ${type_profit:.2f}")
                report.append(f"    ROI: {type_roi:.2f}%")
                report.append(f"    Win Rate: {type_win_rate:.2f}%")
        
        report.append("")
        report.append("TOP 5 WINNING BETS:")
        top_wins = bet_history.nlargest(5, 'Profit')
        for idx, row in top_wins.iterrows():
            report.append(f"  {row['Date']} - {row['HomeTeam']} vs {row['AwayTeam']}")
            report.append(f"    Bet: {row['Bet']}, Odds: {row['Odds']:.2f}, Profit: ${row['Profit']:.2f}")
        
        report.append("")
        report.append("TOP 5 LOSING BETS:")
        top_losses = bet_history.nsmallest(5, 'Profit')
        for idx, row in top_losses.iterrows():
            report.append(f"  {row['Date']} - {row['HomeTeam']} vs {row['AwayTeam']}")
            report.append(f"    Bet: {row['Bet']}, Odds: {row['Odds']:.2f}, Loss: ${row['Profit']:.2f}")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report_text)
        print_progress(f"Backtest report saved to {save_path}")
    
    return report_text
