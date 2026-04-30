import os
import unicodedata
from pathlib import Path
from typing import List, Optional

import pandas as pd


ODDS_PRIORITY_PAIRS = [
    ("PSW", "PSL", "Pinnacle"),
    ("B365W", "B365L", "Bet365"),
    ("AvgW", "AvgL", "Average"),
    ("MaxW", "MaxL", "Max"),
]


def _select_odds_pair(df: pd.DataFrame) -> pd.DataFrame:
    """
    From Sackmann-style columns, select a single odds pair in priority order and
    expose them as winner_odds / loser_odds along with the chosen source label.
    """
    df = df.copy()
    df["_odds_source"] = None
    df["winner_odds"] = pd.NA
    df["loser_odds"] = pd.NA

    for w_col, l_col, label in ODDS_PRIORITY_PAIRS:
        if w_col in df.columns and l_col in df.columns:
            # Fill if currently missing and new pair has data
            has_w = df["winner_odds"].isna()
            w_vals = df[w_col]
            l_vals = df[l_col]
            fill_mask = has_w & w_vals.notna() & l_vals.notna()
            if fill_mask.any():
                df.loc[fill_mask, "winner_odds"] = w_vals[fill_mask]
                df.loc[fill_mask, "loser_odds"] = l_vals[fill_mask]
                df.loc[fill_mask, "_odds_source"] = label

    return df


def _strip_accents(text: str) -> str:
    if text is None:
        return ""
    try:
        return ''.join(c for c in unicodedata.normalize('NFKD', str(text)) if not unicodedata.combining(c))
    except Exception:
        return str(text)


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Sackmann/TML share same logical keys
    if "tourney_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["tourney_date"]):
        # Support both int YYYYMMDD and string
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df["date_key"] = df["tourney_date"].dt.strftime("%Y%m%d")
    df["tourney_key"] = df["tourney_name"].astype(str).apply(_strip_accents).str.strip().str.lower()
    df["winner_key"] = df["winner_name"].astype(str).apply(_strip_accents).str.strip().str.lower()
    df["loser_key"] = df["loser_name"].astype(str).apply(_strip_accents).str.strip().str.lower()
    return df


def load_sackman_odds(root_dir: Optional[str], years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load Sackmann ATP match CSVs from a local directory and extract a normalized odds view.

    Expected files: atp_matches_YYYY.csv (standard Sackmann naming)

    Returns columns: [tourney_date, tourney_name, winner_name, loser_name, winner_odds, loser_odds, _odds_source]
    """
    if not root_dir:
        root_dir = os.environ.get("SACKMAN_ODDS_DIR")
    if not root_dir:
        raise FileNotFoundError("Sackmann odds directory not provided. Set SACKMAN_ODDS_DIR or pass root_dir.")

    base = Path(root_dir)
    if not base.exists():
        raise FileNotFoundError(f"Sackmann odds directory does not exist: {base}")

    files: List[Path] = []
    if years:
        for y in years:
            f = base / f"atp_matches_{y}.csv"
            if f.exists():
                files.append(f)
    else:
        files = list(base.glob("atp_matches_*.csv"))

    if not files:
        raise FileNotFoundError(f"No Sackmann ATP match files found in {base}")

    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f, low_memory=False)
            # Minimal column check
            required_cols = {"tourney_date", "tourney_name", "winner_name", "loser_name"}
            if not required_cols.issubset(df.columns):
                continue
            df = _select_odds_pair(df)
            # Keep only essentials
            keep = [
                "tourney_date",
                "tourney_name",
                "winner_name",
                "loser_name",
                "winner_odds",
                "loser_odds",
                "_odds_source",
            ]
            df = df[keep]
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        raise ValueError("Failed to load any Sackmann odds data with required columns.")

    odds_df = pd.concat(dfs, ignore_index=True)
    # Normalize keys for join
    odds_df = _normalize_keys(odds_df)
    return odds_df


def attach_sackman_odds(matches_df: pd.DataFrame, root_dir: Optional[str]) -> pd.DataFrame:
    """
    Attach winner/loser pre-match odds from Sackmann data to a TML-format matches dataframe.

    Join keys: tourney_date (YYYYMMDD), tourney_name, winner_name, loser_name (case-insensitive, accent-stripped).
    """
    if matches_df is None or len(matches_df) == 0:
        return matches_df

    # Determine year coverage from matches_df to limit loads
    years = sorted(matches_df["tourney_date"].dt.year.dropna().unique().astype(int).tolist()) if "tourney_date" in matches_df.columns else None
    odds_df = load_sackman_odds(root_dir, years=years)

    df = matches_df.copy()
    # Ensure datetime tourney_date present
    if "tourney_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["tourney_date"]):
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df["date_key"] = df["tourney_date"].dt.strftime("%Y%m%d")
    # Apply accent stripping to TML data to match Sackmann normalization
    df["tourney_key"] = df["tourney_name"].astype(str).apply(_strip_accents).str.strip().str.lower()
    df["winner_key"] = df["winner_name"].astype(str).apply(_strip_accents).str.strip().str.lower()
    df["loser_key"] = df["loser_name"].astype(str).apply(_strip_accents).str.strip().str.lower()

    merged = df.merge(
        odds_df[[
            "date_key", "tourney_key", "winner_key", "loser_key", "winner_odds", "loser_odds", "_odds_source"
        ]],
        on=["date_key", "tourney_key", "winner_key", "loser_key"],
        how="left",
        suffixes=("", "_odds")
    )

    # Clean up
    merged = merged.drop(columns=["date_key", "tourney_key", "winner_key", "loser_key"])  # keep tourney_date original
    return merged
