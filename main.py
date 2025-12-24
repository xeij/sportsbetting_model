"""
Lower League Football Value Bettor - Main Entry Point

This script provides a CLI for running the full pipeline or individual components.

Usage:
    python main.py --download          # Download historical data
    python main.py --train             # Train models
    python main.py --backtest          # Run backtesting
    python main.py --predict           # Predict on upcoming fixtures
    python main.py --all               # Run full pipeline
"""

import argparse
import sys
import os

from src.utils import load_config, print_progress
from src.data_loader import download_all_data, merge_data, load_processed_data
from src.feature_engineering import engineer_features
from src.modeling import (
    chronological_split, prepare_data, train_baseline, train_xgboost, train_lightgbm,
    evaluate_model, plot_calibration_curve, plot_feature_importance, save_model
)
from src.backtest import simulate_betting, calculate_roi, plot_roi_curve, generate_backtest_report
from src.prediction import predict_upcoming_fixtures, format_prediction_output


def download_data(config_path: str = "config.yaml") -> None:
    """Download historical match data."""
    print_progress("=" * 80)
    print_progress("DOWNLOADING DATA")
    print_progress("=" * 80)
    
    config = load_config(config_path)
    
    # Download all data
    downloaded_files = download_all_data(config_path)
    
    if not downloaded_files:
        print_progress("No files downloaded. Exiting.", "ERROR")
        return
    
    # Merge data
    output_path = config['data']['processed_data_path']
    merged_df = merge_data(downloaded_files, output_path)
    
    print_progress(f"Data download complete. {len(merged_df)} matches saved to {output_path}")


def train_models(config_path: str = "config.yaml") -> None:
    """Train prediction models."""
    print_progress("=" * 80)
    print_progress("TRAINING MODELS")
    print_progress("=" * 80)
    
    config = load_config(config_path)
    
    # Load processed data
    df = load_processed_data(config_path)
    
    # Engineer features
    df = engineer_features(df, config_path)
    
    # Save engineered features
    features_path = config['data']['processed_data_path'].replace('.csv', '_features.csv')
    df.to_csv(features_path, index=False)
    print_progress(f"Engineered features saved to {features_path}")
    
    # Split data chronologically
    train_df, test_df = chronological_split(df, config['models']['test_size'])
    
    # Prepare data
    X_train, y_train, feature_cols = prepare_data(train_df)
    X_test, y_test, _ = prepare_data(test_df)
    
    # Train baseline model
    print_progress("\n" + "=" * 80)
    baseline_model = train_baseline(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test, "Baseline (Logistic Regression)")
    
    # Train XGBoost
    print_progress("\n" + "=" * 80)
    xgb_model = train_xgboost(X_train, y_train, config)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    # Train LightGBM
    print_progress("\n" + "=" * 80)
    lgb_model = train_lightgbm(X_train, y_train, config)
    lgb_metrics = evaluate_model(lgb_model, X_test, y_test, "LightGBM")
    
    # Save models
    print_progress("\n" + "=" * 80)
    print_progress("SAVING MODELS")
    save_model(baseline_model, feature_cols, "baseline_model", config_path)
    save_model(xgb_model, feature_cols, "xgboost_model", config_path)
    save_model(lgb_model, feature_cols, "lightgbm_model", config_path)
    
    # Generate visualizations
    print_progress("\n" + "=" * 80)
    print_progress("GENERATING VISUALIZATIONS")
    
    viz_dir = config.get('visualization', {}).get('save_dir', 'data/visualizations')
    
    # Calibration curves
    plot_calibration_curve(xgb_model, X_test, y_test, f"{viz_dir}/calibration_xgboost.png")
    plot_calibration_curve(lgb_model, X_test, y_test, f"{viz_dir}/calibration_lightgbm.png")
    
    # Feature importance
    plot_feature_importance(xgb_model, feature_cols, save_path=f"{viz_dir}/feature_importance_xgboost.png")
    plot_feature_importance(lgb_model, feature_cols, save_path=f"{viz_dir}/feature_importance_lightgbm.png")
    
    print_progress("\n" + "=" * 80)
    print_progress("TRAINING COMPLETE")
    print_progress("=" * 80)


def run_backtest(config_path: str = "config.yaml", model_name: str = "xgboost_model") -> None:
    """Run backtesting simulation."""
    print_progress("=" * 80)
    print_progress("RUNNING BACKTEST")
    print_progress("=" * 80)
    
    config = load_config(config_path)
    
    # Load engineered features
    features_path = config['data']['processed_data_path'].replace('.csv', '_features.csv')
    
    if not os.path.exists(features_path):
        print_progress("Engineered features not found. Run training first.", "ERROR")
        return
    
    df = pd.read_csv(features_path, parse_dates=['Date'])
    
    # Split data
    train_df, test_df = chronological_split(df, config['models']['test_size'])
    
    # Prepare test data
    X_test, y_test, feature_cols = prepare_data(test_df)
    
    # Load model
    from src.modeling import load_model
    model, _ = load_model(model_name, config_path)
    
    # Run simulation
    bet_history = simulate_betting(model, X_test, y_test, test_df, feature_cols, config_path)
    
    # Calculate metrics
    metrics = calculate_roi(bet_history)
    
    # Generate report
    viz_dir = config.get('visualization', {}).get('save_dir', 'data/visualizations')
    report = generate_backtest_report(
        bet_history, 
        metrics, 
        save_path=f"{viz_dir}/backtest_report.txt"
    )
    
    print("\n" + report)
    
    # Plot ROI curve
    plot_roi_curve(bet_history, save_path=f"{viz_dir}/roi_curve.png")
    
    # Save bet history
    bet_history.to_csv(f"{viz_dir}/bet_history.csv", index=False)
    print_progress(f"Bet history saved to {viz_dir}/bet_history.csv")
    
    print_progress("\n" + "=" * 80)
    print_progress("BACKTEST COMPLETE")
    print_progress("=" * 80)




def fetch_live_odds(config_path: str = "config.yaml", api_key: str = None) -> None:
    """Fetch live odds from The Odds API."""
    print_progress("=" * 80)
    print_progress("FETCHING LIVE ODDS")
    print_progress("=" * 80)
    
    from src.live_odds import get_live_odds, save_live_odds
    import os
    
    # Get API key from environment or argument
    if not api_key:
        api_key = os.getenv('ODDS_API_KEY')
    
    if not api_key:
        print_progress("ERROR: No API key provided!", "ERROR")
        print_progress("Get a free API key at: https://the-odds-api.com/", "INFO")
        print_progress("Then set it with: set ODDS_API_KEY=your_key_here", "INFO")
        print_progress("Or use: python main.py --fetch-odds --api-key YOUR_KEY", "INFO")
        return
    
    # Fetch odds for all configured leagues
    leagues = [
        'soccer_epl',  # Premier League
        'soccer_england_league1',  # League One
        'soccer_england_league2'   # League Two
    ]
    
    odds_df = get_live_odds(api_key=api_key, leagues=leagues)
    
    if len(odds_df) > 0:
        save_live_odds(odds_df, "data/live_odds.csv")
        print_progress(f"\nFetched {len(odds_df)} upcoming matches with odds")
        print_progress("Saved to: data/live_odds.csv")
        print_progress("\nNow run: python main.py --predict-live")
    else:
        print_progress("No upcoming matches found", "WARNING")


def predict_fixtures(config_path: str = "config.yaml", model_name: str = "xgboost_model") -> None:
    """Predict on upcoming fixtures."""
    print_progress("=" * 80)
    print_progress("PREDICTING UPCOMING FIXTURES")
    print_progress("=" * 80)
    
    # For demo purposes, we'll use the most recent matches from the test set
    # In production, you would load actual upcoming fixtures
    
    config = load_config(config_path)
    
    features_path = config['data']['processed_data_path'].replace('.csv', '_features.csv')
    
    if not os.path.exists(features_path):
        print_progress("Engineered features not found. Run training first.", "ERROR")
        return
    
    df = pd.read_csv(features_path, parse_dates=['Date'])
    
    # Use last 10 matches as "upcoming" fixtures for demo
    upcoming_fixtures = df.tail(10)
    
    # Make predictions
    predictions = predict_upcoming_fixtures(upcoming_fixtures, model_name, config_path)
    
    # Format and display
    output = format_prediction_output(predictions)
    print("\n" + output)
    
    # Save predictions
    viz_dir = config.get('visualization', {}).get('save_dir', 'data/visualizations')
    predictions.to_csv(f"{viz_dir}/predictions.csv", index=False)
    print_progress(f"\nPredictions saved to {viz_dir}/predictions.csv")



def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Lower League Football Value Bettor - Sports Betting Prediction Model"
    )
    
    parser.add_argument('--download', action='store_true', help='Download historical data')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting')
    parser.add_argument('--predict', action='store_true', help='Predict on upcoming fixtures')
    parser.add_argument('--fetch-odds', action='store_true', help='Fetch live odds from The Odds API')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    parser.add_argument('--model', type=str, default='xgboost_model', 
                       help='Model to use (baseline_model, xgboost_model, lightgbm_model)')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--api-key', type=str, default=None,
                       help='The Odds API key (or set ODDS_API_KEY env variable)')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any([args.download, args.train, args.backtest, args.predict, args.fetch_odds, args.all]):
        parser.print_help()
        return
    
    try:
        if args.all:
            download_data(args.config)
            train_models(args.config)
            run_backtest(args.config, args.model)
            predict_fixtures(args.config, args.model)
        else:
            if args.download:
                download_data(args.config)
            
            if args.train:
                train_models(args.config)
            
            if args.backtest:
                run_backtest(args.config, args.model)
            
            if args.fetch_odds:
                fetch_live_odds(args.config, args.api_key)
            
            if args.predict:
                predict_fixtures(args.config, args.model)
        
        print_progress("\n" + "=" * 80)
        print_progress("ALL OPERATIONS COMPLETE")
        print_progress("=" * 80)
        
    except Exception as e:
        print_progress(f"Error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Add pandas import for backtest function
    import pandas as pd
    main()
