"""
Modeling module for training and evaluating prediction models.

This module handles model training, evaluation, and visualization
of results including calibration curves and feature importance.
"""

import os
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

from .utils import load_config, print_progress


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns for modeling.
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        List of feature column names
    """
    # Exclude non-feature columns
    exclude_cols = [
        'Date', 'HomeTeam', 'AwayTeam', 'HomeTeam_Original', 'AwayTeam_Original',
        'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Result',
        'Division', 'Season', 'Referee'
    ]
    
    # Also exclude odds columns (we use derived features instead)
    exclude_prefixes = ['B365', 'BW', 'IW', 'PS', 'WH', 'VC', 'Avg', 'Max']
    
    feature_cols = []
    for col in df.columns:
        # Skip if in exclude list
        if col in exclude_cols:
            continue
        
        # Skip if starts with bookmaker prefix (unless it's a derived feature)
        is_odds_col = any(col.startswith(prefix) for prefix in exclude_prefixes)
        is_derived = any(x in col for x in ['prob', 'overround', 'true'])
        
        if is_odds_col and not is_derived:
            continue
        
        feature_cols.append(col)
    
    return feature_cols


def chronological_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically to prevent data leakage.
    
    Args:
        df: DataFrame with match data
        test_size: Fraction of data to use for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.sort_values('Date').reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print_progress(f"Train set: {len(train_df)} matches ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print_progress(f"Test set: {len(test_df)} matches ({test_df['Date'].min()} to {test_df['Date'].max()})")
    
    return train_df, test_df


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features and target for modeling.
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        Tuple of (X, y, feature_columns)
    """
    feature_cols = get_feature_columns(df)
    
    # Remove rows with missing features
    df_clean = df.dropna(subset=feature_cols + ['Result'])
    
    X = df_clean[feature_cols]
    y = df_clean['Result']
    
    print_progress(f"Prepared {len(X)} samples with {len(feature_cols)} features")
    print_progress(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Train baseline logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print_progress("Training baseline Logistic Regression...")
    
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        multi_class='multinomial',
        solver='lbfgs'
    )
    
    model.fit(X_train, y_train)
    
    print_progress("Baseline model trained")
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, config: Dict) -> xgb.XGBClassifier:
    """
    Train XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration dictionary
        
    Returns:
        Trained model
    """
    print_progress("Training XGBoost model...")
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # Get hyperparameters from config
    params = config['models']['xgboost']
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=config['models']['random_state'],
        **params
    )
    
    model.fit(X_train, y_train_encoded)
    
    # Store label encoder
    model.label_encoder = le
    
    print_progress("XGBoost model trained")
    return model


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, config: Dict) -> lgb.LGBMClassifier:
    """
    Train LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration dictionary
        
    Returns:
        Trained model
    """
    print_progress("Training LightGBM model...")
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # Get hyperparameters from config
    params = config['models']['lightgbm']
    
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        random_state=config['models']['random_state'],
        verbose=-1,
        **params
    )
    
    model.fit(X_train, y_train_encoded)
    
    # Store label encoder
    model.label_encoder = le
    
    print_progress("LightGBM model trained")
    return model


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                   model_name: str = "Model") -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name for logging
        
    Returns:
        Dictionary of metrics
    """
    print_progress(f"Evaluating {model_name}...")
    
    # Get predictions
    if hasattr(model, 'label_encoder'):
        # XGBoost/LightGBM with encoded labels
        y_test_encoded = model.label_encoder.transform(y_test)
        y_pred_encoded = model.predict(X_test)
        y_pred = model.label_encoder.inverse_transform(y_pred_encoded)
        y_pred_proba = model.predict_proba(X_test)
    else:
        # Logistic Regression
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # For log loss and Brier score, we need to encode labels
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)
    
    logloss = log_loss(y_test_encoded, y_pred_proba)
    
    # Brier score (average across classes)
    brier_scores = []
    for i in range(y_pred_proba.shape[1]):
        y_binary = (y_test_encoded == i).astype(int)
        brier = brier_score_loss(y_binary, y_pred_proba[:, i])
        brier_scores.append(brier)
    
    brier_avg = np.mean(brier_scores)
    
    metrics = {
        'accuracy': accuracy,
        'log_loss': logloss,
        'brier_score': brier_avg
    }
    
    print_progress(f"{model_name} - Accuracy: {accuracy:.4f}, Log Loss: {logloss:.4f}, Brier: {brier_avg:.4f}")
    
    return metrics


def plot_calibration_curve(model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                           save_path: str = None) -> None:
    """
    Plot calibration curve for model probabilities.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        save_path: Optional path to save plot
    """
    print_progress("Generating calibration curve...")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test)
    
    # Encode labels
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    class_names = ['Home Win', 'Draw', 'Away Win']
    
    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        y_binary = (y_test_encoded == i).astype(int)
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_binary, y_pred_proba[:, i], n_bins=10)
        
        # Plot
        ax.plot(prob_pred, prob_true, marker='o', label='Model')
        ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Probability')
        ax.set_title(f'Calibration Curve - {class_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print_progress(f"Calibration curve saved to {save_path}")
    
    plt.close()


def plot_feature_importance(model: Any, feature_names: List[str], 
                            top_n: int = 20, save_path: str = None) -> None:
    """
    Plot feature importance.
    
    Args:
        model: Trained model (must have feature_importances_)
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Optional path to save plot
    """
    if not hasattr(model, 'feature_importances_'):
        print_progress("Model does not have feature importances", "WARNING")
        return
    
    print_progress("Generating feature importance plot...")
    
    # Get importances
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print_progress(f"Feature importance plot saved to {save_path}")
    
    plt.close()


def save_model(model: Any, feature_cols: List[str], model_name: str, 
               config_path: str = "config.yaml") -> str:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        feature_cols: List of feature column names
        model_name: Name for the model file
        config_path: Path to configuration file
        
    Returns:
        Path to saved model
    """
    config = load_config(config_path)
    model_dir = config['models']['model_save_dir']
    
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    
    # Save model and feature columns together
    joblib.dump({
        'model': model,
        'feature_columns': feature_cols
    }, model_path)
    
    print_progress(f"Model saved to {model_path}")
    return model_path


def load_model(model_name: str, config_path: str = "config.yaml") -> Tuple[Any, List[str]]:
    """
    Load trained model from disk.
    
    Args:
        model_name: Name of the model file
        config_path: Path to configuration file
        
    Returns:
        Tuple of (model, feature_columns)
    """
    config = load_config(config_path)
    model_dir = config['models']['model_save_dir']
    
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    data = joblib.load(model_path)
    
    print_progress(f"Model loaded from {model_path}")
    return data['model'], data['feature_columns']
