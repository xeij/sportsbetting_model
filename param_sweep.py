"""
Parameter sweep to find optimal backtest config settings.
Tests combinations of value_threshold, kelly_fraction, min_odds, max_odds.
Passes parameters directly in memory — no file I/O per iteration.
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import itertools

warnings.filterwarnings('ignore')

# Silence all print output during sweep setup
import builtins
_real_print = builtins.print
builtins.print = lambda *a, **k: None

from src.utils import load_config, kelly_criterion, calculate_expected_value, remove_vig
from src.modeling import chronological_split, prepare_data, load_model
from src.prediction import predict_match, identify_value_bets

builtins.print = _real_print

# ── Load data and model once ──────────────────────────────────────────────────
_real_print("Loading data and model...")

config = load_config("config.yaml")
features_path = config['data']['processed_data_path'].replace('.csv', '_features.csv')
df = pd.read_csv(features_path, parse_dates=['Date'], low_memory=False)

_, test_df = chronological_split(df, config['models']['test_size'])
X_test, y_test, feature_cols = prepare_data(test_df)

builtins.print = lambda *a, **k: None
model, _ = load_model("xgboost_model", "config.yaml")
builtins.print = _real_print

_real_print(f"Test set: {len(X_test)} matches\n")

# Pre-compute model predictions for all test matches (do this ONCE)
_real_print("Pre-computing model predictions...")
all_model_probs = []
all_market_odds = []

for idx in range(len(X_test)):
    match_features = X_test.iloc[[idx]]
    match_data = test_df.iloc[idx]

    probs = predict_match(model, match_features, feature_cols)
    all_model_probs.append(probs)

    odds = {}
    if all(col in match_data.index for col in ['B365H', 'B365D', 'B365A']):
        if pd.notna(match_data['B365H']) and pd.notna(match_data['B365D']) and pd.notna(match_data['B365A']):
            odds = {
                'home_win': match_data['B365H'],
                'draw':     match_data['B365D'],
                'away_win': match_data['B365A']
            }
    all_market_odds.append(odds)

actual_results = y_test.values
_real_print("Pre-computation done.\n")


# ── Fast simulation (no file I/O) ─────────────────────────────────────────────
def fast_simulate(value_threshold, kelly_fraction, min_odds, max_odds,
                  starting_bankroll=1000.0, vig_method='margin'):

    bankroll = starting_bankroll
    max_stake = starting_bankroll * 0.05
    min_stake = 10.0
    total_bets = 0
    total_staked = 0.0
    total_profit = 0.0
    wins = 0

    for idx in range(len(all_model_probs)):
        model_probs = all_model_probs[idx]
        market_odds = all_market_odds[idx]
        actual = actual_results[idx]

        if not market_odds:
            continue

        # Find value bets
        value_bets = identify_value_bets(model_probs, market_odds,
                                         value_threshold, vig_method)
        value_bets = [b for b in value_bets
                      if min_odds <= b['odds'] <= max_odds]

        if not value_bets:
            continue

        best = max(value_bets, key=lambda x: x['expected_value'])

        # Kelly stake
        raw_kelly = kelly_criterion(best['model_prob'], best['odds'], kelly_fraction)
        stake = bankroll * raw_kelly
        stake = min(stake, max_stake)
        stake = max(stake, min_stake)
        if bankroll < min_stake * 2:
            continue
        stake = min(stake, bankroll * 0.5)

        outcome_map = {'home_win': 'H', 'draw': 'D', 'away_win': 'A'}
        won = (outcome_map[best['outcome']] == actual)
        profit = stake * (best['odds'] - 1) if won else -stake

        bankroll += profit
        if bankroll < 0:
            bankroll = 0

        total_bets   += 1
        total_staked += stake
        total_profit += profit
        if won:
            wins += 1

    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0

    return {
        'total_bets':     total_bets,
        'roi_pct':        round(roi, 2),
        'win_rate_pct':   round(win_rate, 2),
        'final_bankroll': round(bankroll, 2),
        'total_profit':   round(total_profit, 2),
    }


# ── Parameter grid ────────────────────────────────────────────────────────────
GRID = {
    'value_threshold': [0.05, 0.07, 0.08, 0.10, 0.12, 0.15],
    'kelly_fraction':  [0.03, 0.05, 0.10],
    'min_odds':        [1.5,  1.8,  2.0],
    'max_odds':        [5.0,  6.0,  7.0],
}
MIN_BETS = 40

combos = list(itertools.product(*GRID.values()))
_real_print(f"Testing {len(combos)} combinations...\n")

results = []
for i, (vt, kf, mn, mx) in enumerate(combos):
    m = fast_simulate(vt, kf, mn, mx)
    if m['total_bets'] >= MIN_BETS:
        results.append({
            'value_threshold': vt,
            'kelly_fraction':  kf,
            'min_odds':        mn,
            'max_odds':        mx,
            **m
        })

# ── Output ────────────────────────────────────────────────────────────────────
if not results:
    _real_print("No combinations produced enough bets. Lower MIN_BETS.")
    sys.exit(1)

res_df = pd.DataFrame(results).sort_values('roi_pct', ascending=False)

_real_print(f"{'='*95}")
_real_print(f"TOP 20 CONFIGURATIONS  (min {MIN_BETS} bets, sorted by ROI)")
_real_print(f"{'='*95}")
_real_print(res_df.head(20).to_string(index=False))

best = res_df.iloc[0]
_real_print(f"\n{'='*95}")
_real_print("BEST CONFIGURATION:")
_real_print(f"{'='*95}")
_real_print(f"  value_threshold : {best['value_threshold']}")
_real_print(f"  kelly_fraction  : {best['kelly_fraction']}")
_real_print(f"  min_odds        : {best['min_odds']}")
_real_print(f"  max_odds        : {best['max_odds']}")
_real_print(f"  ---")
_real_print(f"  ROI             : {best['roi_pct']}%")
_real_print(f"  Win Rate        : {best['win_rate_pct']}%")
_real_print(f"  Total Bets      : {best['total_bets']}")
_real_print(f"  Final Bankroll  : £{best['final_bankroll']}")
_real_print(f"  Total Profit    : £{best['total_profit']}")

os.makedirs('data/visualizations', exist_ok=True)
res_df.to_csv('data/visualizations/param_sweep_results.csv', index=False)
_real_print(f"\nFull results → data/visualizations/param_sweep_results.csv")
