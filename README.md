# Tennis Prediction Engine

ATP Tennis Match Prediction Engine with Upset Detection using machine learning.

## Features

- **Daily-updated data** from TML Database (TennisMyLife)
- **Player profiling** with style archetypes (Big Server, Counterpuncher, etc.)
- **Upset detection** based on stylistic matchups and form
- **ELO ratings** with surface-specific adjustments
- **Streamlit dashboard** for predictions and analysis

## Data Source

This project uses [TML Database](https://github.com/Tennismylife/TML-Database) which provides:
- Daily-updated ATP match data (same format as Jeff Sackmann)
- Live `ongoing_tourneys.csv` for in-progress tournaments
- Complete match statistics (aces, double faults, break points, etc.)

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e .

# Run the dashboard
streamlit run dashboard.py
```

## Project Structure

```
tennis-prediction-engine/
├── src/
│   ├── data_loader.py      # TML Database data loading
│   ├── player_profiler.py  # Player style profiling
│   ├── matchup_model.py    # Match prediction model
│   ├── elo_ratings.py      # ELO rating system
│   ├── scraper.py          # TennisExplorer scraper for upcoming matches
│   └── backtester.py       # Model backtesting
├── data/                   # Downloaded CSV files (gitignored)
├── tests/                  # Unit tests
├── dashboard.py            # Streamlit dashboard
└── pyproject.toml          # Project configuration
```

## Model Features

The prediction model uses:
- **H2H records** - Historical head-to-head performance
- **Form** - Recent match results (weighted by recency)
- **Surface preference** - Win rates on hard/clay/grass
- **Style matchups** - How player styles interact
- **Clutch metrics** - Tiebreak %, break point save/convert %
- **Fatigue** - Matches played in last 7/30 days
- **ELO ratings** - Overall and surface-specific

## Branch Strategy

- `main`: production-ready branch; protected and tagged on releases.
- `develop`: integration branch for upcoming work; feature branches merge here via PRs.
- `feature/*`: short-lived branches cut from `develop` (e.g., `feature/improve-elo`); rebase as needed and open PRs into `develop`.
- `hotfix/*`: emergency fixes cut from `main`; merge back to `main` and then into `develop` to keep histories aligned.
- `release/*` (optional): stabilization branches from `develop`; once verified, merge to `main`, tag, and back-merge to `develop`.

## FAQ

**Is this agent free of cost?**  
The built-in repository agent does not add new project costs; standard GitHub/Copilot billing for your account or organization still applies.

## License

For personal/experimental use only.
