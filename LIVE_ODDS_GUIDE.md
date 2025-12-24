# Live Odds Fetching - Quick Start Guide
## Setup (One-Time)

### 1. Get Free API Key

Visit: https://the-odds-api.com/

- Sign up for a free account
- Get your API key (500 requests/month free)
- Copy the key

### 2. Set API Key

**Windows (PowerShell):**
```powershell
$env:ODDS_API_KEY="your_api_key_here"
```

**Or pass directly:**
```bash
python main.py --fetch-odds --api-key your_api_key_here
```

## Usage

### Step 1: Fetch Live Odds

```bash
python main.py --fetch-odds
```

This will:
- Fetch upcoming matches for Premier League, League One, and League Two
- Get current odds from multiple bookmakers
- Save to `data/live_odds.csv`

### Step 2: Find Value Bets

```bash
python main.py --predict-live
```

This will:
- Load your trained model
- Compare model predictions vs live odds
- Identify value betting opportunities
- Show recommended bets with edge %

## Supported Leagues

- **Premier League** (E0) - Available on Polymarket/Kalshi
- **League One** (E2)
- **League Two** (E3)
- National League (EC) - Limited API coverage

## API Limits

**Free Tier:**
- 500 requests/month
- Each fetch uses 3 requests (one per league)
- ~166 fetches per month
- Fetch once per day for best results

## Example Output

```
FETCHING LIVE ODDS
==================
Fetching odds for soccer_epl...
Fetched 10 upcoming matches
API requests remaining: 497

Fetched 10 upcoming matches with odds
Saved to: data/live_odds.csv

Now run: python main.py --predict-live
```

