"""
Data Loader Module
Downloads and manages historical ATP match data from TML Database (TennisMyLife).
TML provides daily-updated data in the same format as Jeff Sackmann's tennis_atp.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from typing import Optional, List
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TML Database - daily updated ATP data
TML_BASE_URL = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/"
TML_RANKINGS_URL = "https://raw.githubusercontent.com/Tennismylife/TML-Rankings-Database/master/"

# Fallback to Jeff Sackmann for WTA (TML is ATP only)
SACKMANN_WTA_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/"

DATA_DIR = Path(__file__).parent.parent / "data"


def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR


def download_file(url: str, filepath: Path, force: bool = False) -> bool:
    """Download a file if it doesn't exist or force is True."""
    if filepath.exists() and not force:
        logger.info(f"{filepath.name} already exists, skipping.")
        return True
    
    try:
        logger.info(f"Downloading {filepath.name}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        filepath.write_bytes(response.content)
        logger.info(f"Successfully downloaded {filepath.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def get_data_freshness() -> dict:
    """Check the freshness of TML data by looking at recent commits."""
    try:
        api_url = "https://api.github.com/repos/Tennismylife/TML-Database/commits?per_page=1"
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        commits = response.json()
        if commits:
            last_commit = commits[0]
            commit_date = last_commit['commit']['committer']['date']
            return {
                'last_update': commit_date,
                'message': last_commit['commit']['message'],
                'is_fresh': True
            }
    except Exception as e:
        logger.warning(f"Could not check data freshness: {e}")
    
    return {'last_update': None, 'message': None, 'is_fresh': False}


def download_atp_matches(years: List[int], force: bool = False) -> List[Path]:
    """Download ATP match files for specified years from TML Database."""
    ensure_data_dir()
    downloaded = []
    
    for year in years:
        url = f"{TML_BASE_URL}{year}.csv"
        filepath = DATA_DIR / f"atp_matches_{year}.csv"
        if download_file(url, filepath, force):
            downloaded.append(filepath)
    
    return downloaded


def download_ongoing_tournaments(force: bool = True) -> Optional[Path]:
    """
    Download ongoing tournaments data from TML Database.
    This file contains live/in-progress tournament matches.
    Always force refresh for live data.
    """
    ensure_data_dir()
    url = f"{TML_BASE_URL}ongoing_tourneys.csv"
    filepath = DATA_DIR / "ongoing_tourneys.csv"
    if download_file(url, filepath, force):
        return filepath
    return None


def download_atp_players(force: bool = False) -> Optional[Path]:
    """Download ATP player database from TML."""
    ensure_data_dir()
    url = f"{TML_BASE_URL}ATP_Database.csv"
    filepath = DATA_DIR / "atp_players.csv"
    if download_file(url, filepath, force):
        return filepath
    return None


def download_atp_rankings(force: bool = False) -> Optional[Path]:
    """Download current ATP rankings from TML Rankings Database."""
    ensure_data_dir()
    current_year = datetime.now().year
    url = f"{TML_RANKINGS_URL}ATP_Rankings_{current_year}.csv"
    filepath = DATA_DIR / "atp_rankings_current.csv"
    if download_file(url, filepath, force):
        return filepath
    return None


def get_rolling_years(window_size: int = 7) -> List[int]:
    """
    Get a rolling window of years ending at current year.
    
    Args:
        window_size: Number of years to include (default 7)
    
    Returns:
        List of years, e.g., [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    """
    current_year = datetime.now().year
    
    available_years = []
    for year in range(current_year - window_size, current_year + 1):
        filepath = DATA_DIR / f"atp_matches_{year}.csv"
        if filepath.exists():
            available_years.append(year)
        else:
            try:
                url = f"{TML_BASE_URL}{year}.csv"
                response = requests.head(url, timeout=5)
                if response.status_code == 200:
                    available_years.append(year)
            except:
                pass
    
    if len(available_years) > window_size:
        available_years = available_years[-window_size:]
    
    logger.info(f"Rolling window years: {available_years}")
    return available_years


def load_matches(
    years: Optional[List[int]] = None, 
    download_if_missing: bool = True,
    include_ongoing: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Load ATP match data for specified years from TML Database.
    
    Args:
        years: List of years to load. If None, uses rolling 7-year window.
        download_if_missing: Download files if not found locally.
        include_ongoing: Include ongoing tournament matches (live data).
        force_refresh: Force re-download of all data files.
    
    Returns:
        DataFrame with all matches.
    """
    if years is None:
        years = get_rolling_years(window_size=7)
    
    ensure_data_dir()
    
    existing_files = []
    missing_years = []
    
    for year in years:
        filepath = DATA_DIR / f"atp_matches_{year}.csv"
        if filepath.exists() and not force_refresh:
            existing_files.append(filepath)
        else:
            missing_years.append(year)
    
    if missing_years and download_if_missing:
        logger.info(f"Downloading data for years: {missing_years}")
        downloaded = download_atp_matches(missing_years, force=force_refresh)
        existing_files.extend(downloaded)
    
    if not existing_files:
        raise FileNotFoundError("No match files found. Run download first.")
    
    dfs = []
    for f in sorted(set(existing_files)):
        try:
            df = pd.read_csv(f, low_memory=False)
            df['source_file'] = f.name
            df['is_ongoing'] = False
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {f}: {e}")
    
    # Load ongoing tournaments if requested
    if include_ongoing:
        ongoing_file = download_ongoing_tournaments(force=True)
        if ongoing_file and ongoing_file.exists():
            try:
                ongoing_df = pd.read_csv(ongoing_file, low_memory=False)
                ongoing_df['source_file'] = 'ongoing_tourneys.csv'
                ongoing_df['is_ongoing'] = True
                dfs.append(ongoing_df)
                logger.info(f"Loaded {len(ongoing_df)} ongoing tournament matches")
            except Exception as e:
                logger.warning(f"Could not load ongoing tournaments: {e}")
    
    if not dfs:
        raise ValueError("No data loaded successfully.")
    
    matches_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(matches_df):,} matches from {len(dfs)} files.")
    
    return matches_df


def load_players(download_if_missing: bool = True) -> pd.DataFrame:
    """Load ATP player data from TML Database."""
    ensure_data_dir()
    filepath = DATA_DIR / "atp_players.csv"
    
    if not filepath.exists() and download_if_missing:
        download_atp_players()
    
    if not filepath.exists():
        raise FileNotFoundError("Player file not found.")
    
    players_df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(players_df):,} players.")
    return players_df


def load_rankings(download_if_missing: bool = True) -> pd.DataFrame:
    """Load current ATP rankings from TML Rankings Database."""
    ensure_data_dir()
    filepath = DATA_DIR / "atp_rankings_current.csv"
    
    if not filepath.exists() and download_if_missing:
        download_atp_rankings()
    
    if not filepath.exists():
        raise FileNotFoundError("Rankings file not found.")
    
    rankings_df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(rankings_df):,} ranking entries.")
    return rankings_df


def get_player_matches(matches_df: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """Get all matches for a specific player."""
    mask = (
        matches_df['winner_name'].str.contains(player_name, case=False, na=False) |
        matches_df['loser_name'].str.contains(player_name, case=False, na=False)
    )
    return matches_df[mask].copy()


def get_h2h(matches_df: pd.DataFrame, player1: str, player2: str) -> pd.DataFrame:
    """Get head-to-head matches between two players."""
    mask = (
        (matches_df['winner_name'].str.contains(player1, case=False, na=False) &
         matches_df['loser_name'].str.contains(player2, case=False, na=False)) |
        (matches_df['winner_name'].str.contains(player2, case=False, na=False) &
         matches_df['loser_name'].str.contains(player1, case=False, na=False))
    )
    return matches_df[mask].copy()


def preprocess_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess matches dataframe:
    - Parse dates
    - Calculate derived columns
    - Handle missing values
    """
    df = matches_df.copy()
    
    # Parse tournament date (TML uses same format as Sackmann: YYYYMMDD)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df['year'] = df['tourney_date'].dt.year
    
    # Fill missing surfaces
    df['surface'] = df['surface'].fillna('Unknown')
    
    # Calculate rank difference
    df['rank_diff'] = df['loser_rank'] - df['winner_rank']
    
    # Identify upsets (lower-ranked player wins)
    df['is_upset'] = df['winner_rank'] > df['loser_rank']
    
    # Calculate serve stats percentages where available
    if 'w_svpt' in df.columns:
        df['winner_1st_serve_pct'] = df['w_1stIn'] / df['w_svpt']
        df['winner_1st_won_pct'] = df['w_1stWon'] / df['w_1stIn']
        df['winner_2nd_won_pct'] = df['w_2ndWon'] / (df['w_svpt'] - df['w_1stIn'])
        df['winner_ace_pct'] = df['w_ace'] / df['w_svpt']
        df['winner_df_pct'] = df['w_df'] / df['w_svpt']
        
        df['loser_1st_serve_pct'] = df['l_1stIn'] / df['l_svpt']
        df['loser_1st_won_pct'] = df['l_1stWon'] / df['l_1stIn']
        df['loser_2nd_won_pct'] = df['l_2ndWon'] / (df['l_svpt'] - df['l_1stIn'])
        df['loser_ace_pct'] = df['l_ace'] / df['l_svpt']
        df['loser_df_pct'] = df['l_df'] / df['l_svpt']
    
    return df


def refresh_all_data(years: Optional[List[int]] = None) -> dict:
    """
    Force refresh all data from TML Database.
    
    Returns:
        Dict with refresh status and statistics
    """
    if years is None:
        years = get_rolling_years(window_size=7)
    
    results = {
        'years_refreshed': [],
        'matches_loaded': 0,
        'players_loaded': 0,
        'freshness': get_data_freshness()
    }
    
    # Download all years
    downloaded = download_atp_matches(years, force=True)
    results['years_refreshed'] = [int(f.stem.split('_')[-1]) for f in downloaded]
    
    # Download players
    download_atp_players(force=True)
    
    # Download rankings
    download_atp_rankings(force=True)
    
    # Download ongoing
    download_ongoing_tournaments(force=True)
    
    # Load and count
    try:
        matches = load_matches(years, download_if_missing=False, include_ongoing=True)
        results['matches_loaded'] = len(matches)
    except:
        pass
    
    try:
        players = load_players(download_if_missing=False)
        results['players_loaded'] = len(players)
    except:
        pass
    
    logger.info(f"Data refresh complete: {results}")
    return results


if __name__ == "__main__":
    print("TML Database Data Loader")
    print("=" * 50)
    
    # Check data freshness
    freshness = get_data_freshness()
    print(f"\nData freshness: {freshness}")
    
    # Download and load data
    print("\nDownloading ATP data from TML Database...")
    matches = load_matches(years=[2024, 2025, 2026], include_ongoing=True)
    matches = preprocess_matches(matches)
    
    print(f"\nShape: {matches.shape}")
    print(f"\nColumns: {matches.columns.tolist()}")
    print(f"\nSurfaces: {matches['surface'].value_counts()}")
    print(f"\nYears: {matches['year'].value_counts().sort_index()}")
    
    # Show ongoing matches
    ongoing = matches[matches['is_ongoing'] == True]
    if len(ongoing) > 0:
        print(f"\nOngoing tournament matches: {len(ongoing)}")
