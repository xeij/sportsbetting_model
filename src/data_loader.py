"""
Data loader module for downloading and processing football match data.

This module handles downloading CSV files from football-data.co.uk,
merging multiple seasons and divisions, and cleaning the data.
"""

import os
from typing import List, Optional
import pandas as pd
import requests
from pathlib import Path
import time

from .utils import load_config, print_progress, standardize_team_name


def download_season_data(season: str, division: str, base_url: str, 
                         output_dir: str, retries: int = 3) -> Optional[str]:
    """
    Download CSV data for a specific season and division.
    
    Args:
        season: Season code (e.g., '2425' for 2024/25)
        division: Division code (e.g., 'E2' for League One)
        base_url: Base URL for data source
        output_dir: Directory to save downloaded files
        retries: Number of retry attempts
        
    Returns:
        Path to downloaded file, or None if download failed
    """
    url = f"{base_url}/{season}/{division}.csv"
    output_path = os.path.join(output_dir, f"{division}_{season}.csv")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(output_path):
        print_progress(f"File already exists: {output_path}")
        return output_path
    
    # Download with retries
    for attempt in range(retries):
        try:
            print_progress(f"Downloading {division} {season}... (attempt {attempt + 1}/{retries})")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print_progress(f"Successfully downloaded: {output_path}")
            return output_path
            
        except requests.exceptions.RequestException as e:
            print_progress(f"Download failed: {e}", "WARNING")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print_progress(f"Failed to download {url} after {retries} attempts", "ERROR")
                return None
    
    return None


def download_all_data(config_path: str = "config.yaml") -> List[str]:
    """
    Download all configured seasons and divisions.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        List of paths to downloaded files
    """
    config = load_config(config_path)
    
    base_url = config['data']['base_url']
    divisions = config['data']['divisions']
    seasons = config['data']['seasons']
    output_dir = config['data']['raw_data_dir']
    
    downloaded_files = []
    
    print_progress(f"Starting download of {len(seasons)} seasons × {len(divisions)} divisions")
    
    for season in seasons:
        for division in divisions:
            file_path = download_season_data(season, division, base_url, output_dir)
            if file_path:
                downloaded_files.append(file_path)
    
    print_progress(f"Download complete. {len(downloaded_files)} files downloaded.")
    return downloaded_files


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize DataFrame.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy
    df = df.copy()
    
    # Standardize column names (remove spaces, make lowercase)
    df.columns = df.columns.str.strip()
    
    # Convert date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        # Try alternative format if first one fails
        if df['Date'].isna().all():
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')
    
    # Remove rows with missing essential data (before standardization)
    essential_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
    df = df.dropna(subset=essential_columns)
    
    # Standardize team names
    if 'HomeTeam' in df.columns:
        df['HomeTeam_Original'] = df['HomeTeam']
        df['HomeTeam'] = df['HomeTeam'].apply(standardize_team_name)
    
    if 'AwayTeam' in df.columns:
        df['AwayTeam_Original'] = df['AwayTeam']
        df['AwayTeam'] = df['AwayTeam'].apply(standardize_team_name)
    
    # Ensure numeric columns are numeric
    numeric_columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 
                      'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert odds columns to numeric
    odds_columns = [col for col in df.columns if any(
        col.startswith(prefix) for prefix in ['B365', 'BW', 'IW', 'PS', 'WH', 'VC', 'Avg', 'Max']
    )]
    
    for col in odds_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def merge_data(file_paths: List[str], output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Merge multiple CSV files into a single DataFrame.
    
    Args:
        file_paths: List of paths to CSV files
        output_path: Optional path to save merged data
        
    Returns:
        Merged DataFrame
    """
    print_progress(f"Merging {len(file_paths)} files...")
    
    dfs = []
    for file_path in file_paths:
        try:
            # Extract division and season from filename
            filename = os.path.basename(file_path)
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 2:
                division = parts[0]
                season = parts[1]
            else:
                division = 'Unknown'
                season = 'Unknown'
            
            # Read CSV
            df = pd.read_csv(file_path, encoding='latin-1')
            
            # Add metadata columns
            df['Division'] = division
            df['Season'] = season
            
            dfs.append(df)
            print_progress(f"Loaded {filename}: {len(df)} matches")
            
        except Exception as e:
            print_progress(f"Error loading {file_path}: {e}", "ERROR")
    
    if not dfs:
        raise ValueError("No data files could be loaded")
    
    # Concatenate all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Clean the merged data
    merged_df = clean_data(merged_df)
    
    print_progress(f"Merged data: {len(merged_df)} total matches")
    
    # Save if output path provided
    if output_path:
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        print_progress(f"Saved merged data to {output_path}")
    
    return merged_df


def load_processed_data(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Load processed data from CSV.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DataFrame with processed data
    """
    config = load_config(config_path)
    data_path = config['data']['processed_data_path']
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}. Run download first.")
    
    print_progress(f"Loading processed data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=['Date'])
    print_progress(f"Loaded {len(df)} matches")
    
    return df


def get_available_bookmakers(df: pd.DataFrame) -> List[str]:
    """
    Get list of available bookmakers in the DataFrame.
    
    Args:
        df: DataFrame with match data
        
    Returns:
        List of bookmaker prefixes
    """
    bookmakers = set()
    
    for col in df.columns:
        for prefix in ['B365', 'BW', 'IW', 'PS', 'WH', 'VC', 'Avg', 'Max']:
            if col.startswith(prefix):
                bookmakers.add(prefix)
                break
    
    return sorted(list(bookmakers))
