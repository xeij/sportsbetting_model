"""
Live odds scraper for fetching current betting odds.

This module fetches live odds from The Odds API (free tier available).
Alternative: Can scrape from odds comparison websites.
"""

import requests
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta

from .utils import load_config, print_progress


def fetch_odds_from_api(api_key: str, sport: str = "soccer_epl", 
                        regions: str = "uk") -> List[Dict]:
    """
    Fetch live odds from The Odds API.
    
    Args:
        api_key: API key from the-odds-api.com
        sport: Sport key (soccer_epl for Premier League)
        regions: Regions for odds (uk, us, eu, au)
        
    Returns:
        List of matches with odds
    """
    base_url = "https://api.the-odds-api.com/v4/sports"
    
    # Available sports:
    # soccer_epl - Premier League
    # soccer_england_league1 - League One
    # soccer_england_league2 - League Two
    
    endpoint = f"{base_url}/{sport}/odds"
    
    params = {
        'apiKey': api_key,
        'regions': regions,
        'markets': 'h2h',  # Head to head (1X2)
        'oddsFormat': 'decimal',
        'dateFormat': 'iso'
    }
    
    try:
        print_progress(f"Fetching odds for {sport}...")
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        print_progress(f"Fetched {len(data)} upcoming matches")
        
        # Check remaining requests
        remaining = response.headers.get('x-requests-remaining', 'unknown')
        print_progress(f"API requests remaining: {remaining}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print_progress(f"Error fetching odds: {e}", "ERROR")
        return []


def parse_odds_data(api_data: List[Dict]) -> pd.DataFrame:
    """
    Parse API response into DataFrame format.
    
    Args:
        api_data: Raw API response
        
    Returns:
        DataFrame with matches and odds
    """
    matches = []
    
    for match in api_data:
        # Get match details
        home_team = match.get('home_team', '')
        away_team = match.get('away_team', '')
        commence_time = match.get('commence_time', '')
        
        # Parse commence time
        try:
            match_date = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
        except:
            match_date = datetime.now()
        
        # Get bookmaker odds (use first available bookmaker)
        bookmakers = match.get('bookmakers', [])
        if not bookmakers:
            continue
        
        # Try to find Bet365 or use first bookmaker
        bookmaker = bookmakers[0]
        for bm in bookmakers:
            if 'bet365' in bm.get('key', '').lower():
                bookmaker = bm
                break
        
        # Get h2h market
        markets = bookmaker.get('markets', [])
        h2h_market = None
        for market in markets:
            if market.get('key') == 'h2h':
                h2h_market = market
                break
        
        if not h2h_market:
            continue
        
        # Parse outcomes
        outcomes = h2h_market.get('outcomes', [])
        odds_dict = {}
        for outcome in outcomes:
            name = outcome.get('name', '')
            price = outcome.get('price', 0)
            
            if name == home_team:
                odds_dict['home'] = price
            elif name == away_team:
                odds_dict['away'] = price
            elif 'draw' in name.lower():
                odds_dict['draw'] = price
        
        # Only add if we have all three odds
        if len(odds_dict) == 3:
            matches.append({
                'Date': match_date,
                'HomeTeam': home_team.lower(),
                'AwayTeam': away_team.lower(),
                'B365H': odds_dict['home'],
                'B365D': odds_dict['draw'],
                'B365A': odds_dict['away'],
                'Bookmaker': bookmaker.get('key', 'unknown')
            })
    
    return pd.DataFrame(matches)


def scrape_odds_fallback() -> pd.DataFrame:
    """
    Fallback scraper for when API is not available.
    Scrapes from a free odds comparison website.
    
    Returns:
        DataFrame with matches and odds
    """
    print_progress("API not available, using fallback scraper...", "WARNING")
    
    # This would scrape from a website like:
    # - oddsportal.com
    # - oddschecker.com
    # - flashscore.com
    
    # For now, return empty DataFrame
    # Implementing web scraping requires BeautifulSoup and is fragile
    
    print_progress("Fallback scraper not implemented. Please use API key.", "ERROR")
    return pd.DataFrame()


def get_live_odds(api_key: Optional[str] = None, 
                  leagues: List[str] = ['soccer_epl']) -> pd.DataFrame:
    """
    Get live odds for specified leagues.
    
    Args:
        api_key: The Odds API key (get free key at the-odds-api.com)
        leagues: List of league codes to fetch
        
    Returns:
        DataFrame with all upcoming matches and odds
    """
    if not api_key:
        print_progress("No API key provided. Get free key at: https://the-odds-api.com/", "WARNING")
        return scrape_odds_fallback()
    
    all_matches = []
    
    for league in leagues:
        data = fetch_odds_from_api(api_key, league)
        if data:
            matches_df = parse_odds_data(data)
            matches_df['League'] = league
            all_matches.append(matches_df)
    
    if all_matches:
        return pd.concat(all_matches, ignore_index=True)
    else:
        return pd.DataFrame()


def save_live_odds(df: pd.DataFrame, output_path: str = "data/live_odds.csv") -> None:
    """
    Save fetched odds to CSV.
    
    Args:
        df: DataFrame with odds
        output_path: Path to save CSV
    """
    df.to_csv(output_path, index=False)
    print_progress(f"Saved {len(df)} matches to {output_path}")


# Example usage
if __name__ == "__main__":
    # Get API key from environment or config
    import os
    api_key = os.getenv('ODDS_API_KEY')
    
    if api_key:
        # Fetch odds for Premier League
        odds_df = get_live_odds(
            api_key=api_key,
            leagues=['soccer_epl', 'soccer_england_league1', 'soccer_england_league2']
        )
        
        print(f"Fetched {len(odds_df)} upcoming matches")
        print(odds_df.head())
        
        # Save to file
        save_live_odds(odds_df)
    else:
        print("Set ODDS_API_KEY environment variable")
        print("Get free key at: https://the-odds-api.com/")
