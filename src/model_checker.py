"""
Utility functions for checking model freshness and data updates.
"""

import os
from datetime import datetime
from pathlib import Path
import joblib


def check_model_freshness(config_path: str = "config.yaml") -> dict:
    """
    Check if trained models are up to date with the data.
    
    Returns:
        Dictionary with status information
    """
    from .utils import load_config
    
    config = load_config(config_path)
    
    # Paths
    features_path = config['data']['processed_data_path'].replace('.csv', '_features.csv')
    model_dir = Path(config['models']['save_dir'])
    
    result = {
        'models_exist': False,
        'data_exists': False,
        'models_up_to_date': False,
        'message': '',
        'model_date': None,
        'data_date': None
    }
    
    # Check if data exists
    if not os.path.exists(features_path):
        result['message'] = "No processed data found. Run 'Download Data' and 'Train Models' first."
        return result
    
    result['data_exists'] = True
    data_modified = datetime.fromtimestamp(os.path.getmtime(features_path))
    result['data_date'] = data_modified
    
    # Check if models exist
    model_files = ['xgboost_model.joblib', 'lightgbm_model.joblib', 'baseline_model.joblib']
    models_found = [model_dir / f for f in model_files if (model_dir / f).exists()]
    
    if not models_found:
        result['message'] = "No trained models found. Click 'Train Models' to train."
        return result
    
    result['models_exist'] = True
    
    # Get oldest model date
    model_dates = [datetime.fromtimestamp(os.path.getmtime(m)) for m in models_found]
    oldest_model_date = min(model_dates)
    result['model_date'] = oldest_model_date
    
    # Compare dates
    if oldest_model_date < data_modified:
        result['message'] = f"Models are outdated. Data updated on {data_modified.strftime('%Y-%m-%d %H:%M')}, but models trained on {oldest_model_date.strftime('%Y-%m-%d %H:%M')}. Click 'Train Models' to retrain."
        result['models_up_to_date'] = False
    else:
        result['message'] = f"Models are up to date (trained on {oldest_model_date.strftime('%Y-%m-%d %H:%M')})."
        result['models_up_to_date'] = True
    
    return result


def get_model_info(model_name: str = "xgboost_model", config_path: str = "config.yaml") -> dict:
    """
    Get information about a trained model.
    
    Returns:
        Dictionary with model metadata
    """
    from .utils import load_config
    
    config = load_config(config_path)
    model_dir = Path(config['models']['save_dir'])
    model_path = model_dir / f"{model_name}.joblib"
    
    if not model_path.exists():
        return {
            'exists': False,
            'message': f"Model {model_name} not found"
        }
    
    # Load model to get metadata
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_cols = model_data['feature_cols']
        
        # Get file info
        modified = datetime.fromtimestamp(os.path.getmtime(model_path))
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        return {
            'exists': True,
            'name': model_name,
            'trained_date': modified,
            'num_features': len(feature_cols),
            'size_mb': round(size_mb, 2),
            'model_type': type(model).__name__
        }
    except Exception as e:
        return {
            'exists': False,
            'message': f"Error loading model: {str(e)}"
        }
