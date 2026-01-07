"""
ELO Rating System for Tennis
Calculates overall and surface-specific ELO ratings from historical match data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ELO Constants
INITIAL_ELO = 1500
K_FACTOR_BASE = 32  # Base K-factor
K_FACTOR_NEW_PLAYER = 40  # Higher K for players with few matches
K_FACTOR_ESTABLISHED = 24  # Lower K for established players
MATCHES_TO_ESTABLISH = 30  # Matches before player is "established"

# Surface-specific K-factors (slightly lower since it's a subset of matches)
K_SURFACE = 28

# Tournament level multipliers for K-factor
LEVEL_MULTIPLIERS = {
    'G': 1.2,   # Grand Slams matter more
    'M': 1.1,   # Masters 1000
    'A': 1.0,   # ATP 500
    'B': 0.95,  # ATP 250
    'C': 0.85,  # Challengers
    'F': 0.75,  # Futures
    'D': 1.0,   # Davis Cup
}


@dataclass
class PlayerELO:
    """ELO ratings for a player."""
    player_id: str  # TML uses alphanumeric IDs like 'CD85'
    overall_elo: float
    hard_elo: float
    clay_elo: float
    grass_elo: float
    peak_elo: float  # Highest ELO ever achieved
    elo_30d_change: float  # ELO change in last 30 days
    matches_played: int
    surface_matches: Dict[str, int]  # Matches per surface


class ELORatingSystem:
    """
    Calculates and maintains ELO ratings for tennis players.
    Supports overall ratings and surface-specific ratings.
    """
    
    def __init__(self, k_factor: float = K_FACTOR_BASE):
        """Initialize ELO system with configurable K-factor."""
        self.k_factor = k_factor
        self.ratings: Dict[str, Dict] = {}  # player_id -> rating info
        self.rating_history: Dict[str, list] = {}  # player_id -> [(date, elo), ...]
        
    def _get_k_factor(self, player_id: str, level: str = 'A') -> float:
        """
        Get dynamic K-factor based on player experience and tournament level.
        New players have higher K (ratings adjust faster).
        """
        matches = self.ratings.get(player_id, {}).get('matches', 0)
        
        if matches < MATCHES_TO_ESTABLISH:
            base_k = K_FACTOR_NEW_PLAYER
        else:
            base_k = K_FACTOR_ESTABLISHED
            
        level_mult = LEVEL_MULTIPLIERS.get(level, 1.0)
        return base_k * level_mult
    
    def _expected_score(self, elo_a: float, elo_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    
    def _update_elo(self, elo: float, expected: float, actual: float, k: float) -> float:
        """Update ELO rating based on match result."""
        return elo + k * (actual - expected)
    
    def _ensure_player(self, player_id: str) -> None:
        """Ensure player exists in ratings dictionary."""
        if player_id not in self.ratings:
            self.ratings[player_id] = {
                'overall': INITIAL_ELO,
                'hard': INITIAL_ELO,
                'clay': INITIAL_ELO,
                'grass': INITIAL_ELO,
                'peak': INITIAL_ELO,
                'matches': 0,
                'hard_matches': 0,
                'clay_matches': 0,
                'grass_matches': 0,
            }
            self.rating_history[player_id] = []
    
    def process_match(self, winner_id: str, loser_id: str, surface: str, 
                      level: str = 'A', match_date: pd.Timestamp = None) -> Tuple[float, float]:
        """
        Process a single match and update ELO ratings.
        
        Args:
            winner_id: ID of the winning player
            loser_id: ID of the losing player
            surface: Match surface (Hard, Clay, Grass)
            level: Tournament level (G, M, A, B, C, F, D)
            match_date: Date of the match (for tracking history)
            
        Returns:
            Tuple of (winner_new_elo, loser_new_elo)
        """
        self._ensure_player(winner_id)
        self._ensure_player(loser_id)
        
        surface_key = surface.lower() if surface else 'hard'
        if surface_key not in ['hard', 'clay', 'grass']:
            surface_key = 'hard'  # Default
        
        # Get current ratings
        winner_elo = self.ratings[winner_id]['overall']
        loser_elo = self.ratings[loser_id]['overall']
        winner_surf_elo = self.ratings[winner_id][surface_key]
        loser_surf_elo = self.ratings[loser_id][surface_key]
        
        # Calculate expected scores
        expected_winner = self._expected_score(winner_elo, loser_elo)
        expected_loser = 1 - expected_winner
        expected_winner_surf = self._expected_score(winner_surf_elo, loser_surf_elo)
        expected_loser_surf = 1 - expected_winner_surf
        
        # Get K-factors
        k_winner = self._get_k_factor(winner_id, level)
        k_loser = self._get_k_factor(loser_id, level)
        k_surf = K_SURFACE * LEVEL_MULTIPLIERS.get(level, 1.0)
        
        # Update overall ELO
        new_winner_elo = self._update_elo(winner_elo, expected_winner, 1, k_winner)
        new_loser_elo = self._update_elo(loser_elo, expected_loser, 0, k_loser)
        
        # Update surface-specific ELO
        new_winner_surf = self._update_elo(winner_surf_elo, expected_winner_surf, 1, k_surf)
        new_loser_surf = self._update_elo(loser_surf_elo, expected_loser_surf, 0, k_surf)
        
        # Store updated ratings
        self.ratings[winner_id]['overall'] = new_winner_elo
        self.ratings[loser_id]['overall'] = new_loser_elo
        self.ratings[winner_id][surface_key] = new_winner_surf
        self.ratings[loser_id][surface_key] = new_loser_surf
        
        # Update match counts
        self.ratings[winner_id]['matches'] += 1
        self.ratings[loser_id]['matches'] += 1
        self.ratings[winner_id][f'{surface_key}_matches'] += 1
        self.ratings[loser_id][f'{surface_key}_matches'] += 1
        
        # Update peak ELO
        if new_winner_elo > self.ratings[winner_id]['peak']:
            self.ratings[winner_id]['peak'] = new_winner_elo
        
        # Track history for momentum calculation
        if match_date is not None:
            self.rating_history[winner_id].append((match_date, new_winner_elo))
            self.rating_history[loser_id].append((match_date, new_loser_elo))
        
        return new_winner_elo, new_loser_elo
    
    def calculate_ratings_from_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ELO ratings from a DataFrame of historical matches.
        Matches must be sorted chronologically!
        
        Args:
            matches_df: DataFrame with winner_id, loser_id, surface, tourney_level, match_date
            
        Returns:
            DataFrame with player ELO ratings
        """
        logger.info("Calculating ELO ratings from match history...")
        
        # Ensure chronological order
        df = matches_df.copy()
        if 'match_date' not in df.columns:
            df['match_date'] = pd.to_datetime(df['tourney_date'].astype(str), format='mixed', errors='coerce')
        df = df.sort_values('match_date')
        
        # Process each match
        total = len(df)
        for i, (_, match) in enumerate(df.iterrows()):
            if i % 10000 == 0:
                logger.info(f"Processing match {i}/{total}...")
            
            winner_id = match.get('winner_id')
            loser_id = match.get('loser_id')
            
            if pd.isna(winner_id) or pd.isna(loser_id):
                continue
                
            self.process_match(
                winner_id=str(winner_id),
                loser_id=str(loser_id),
                surface=match.get('surface', 'Hard'),
                level=match.get('tourney_level', 'A'),
                match_date=match.get('match_date')
            )
        
        logger.info(f"Calculated ELO for {len(self.ratings)} players")
        return self.get_ratings_dataframe()
    
    def _calculate_elo_momentum(self, player_id: str, days: int = 30) -> float:
        """Calculate ELO change over the last N days."""
        if player_id not in self.rating_history:
            return 0.0
        
        history = self.rating_history[player_id]
        if len(history) < 2:
            return 0.0
        
        # Get ratings from last N days
        latest_date = history[-1][0]
        if pd.isna(latest_date):
            return 0.0
            
        cutoff = latest_date - pd.Timedelta(days=days)
        
        # Find rating at cutoff
        old_rating = INITIAL_ELO
        for date, rating in history:
            if date and date <= cutoff:
                old_rating = rating
            else:
                break
        
        current_rating = history[-1][1]
        return current_rating - old_rating
    
    def get_ratings_dataframe(self) -> pd.DataFrame:
        """Convert ratings to a DataFrame."""
        records = []
        for player_id, ratings in self.ratings.items():
            momentum = self._calculate_elo_momentum(player_id, days=30)
            records.append({
                'player_id': player_id,
                'elo_overall': ratings['overall'],
                'elo_hard': ratings['hard'],
                'elo_clay': ratings['clay'],
                'elo_grass': ratings['grass'],
                'elo_peak': ratings['peak'],
                'elo_30d_change': momentum,
                'elo_matches': ratings['matches'],
                'elo_hard_matches': ratings['hard_matches'],
                'elo_clay_matches': ratings['clay_matches'],
                'elo_grass_matches': ratings['grass_matches'],
            })
        
        return pd.DataFrame(records)
    
    def get_player_elo(self, player_id: str, surface: str = None) -> float:
        """Get ELO rating for a player, optionally surface-specific."""
        if player_id not in self.ratings:
            return INITIAL_ELO
        
        if surface:
            surface_key = surface.lower()
            if surface_key in ['hard', 'clay', 'grass']:
                return self.ratings[player_id][surface_key]
        
        return self.ratings[player_id]['overall']
    
    def predict_match(self, player1_id: int, player2_id: int, surface: str = None) -> Dict:
        """
        Predict match outcome based on ELO ratings.
        
        Returns:
            Dict with p1_win_prob, p2_win_prob, elo_diff, surface_elo_diff
        """
        # Overall ELO
        elo1 = self.get_player_elo(player1_id)
        elo2 = self.get_player_elo(player2_id)
        p1_prob = self._expected_score(elo1, elo2)
        
        # Surface-specific ELO
        if surface:
            surf_elo1 = self.get_player_elo(player1_id, surface)
            surf_elo2 = self.get_player_elo(player2_id, surface)
            p1_surf_prob = self._expected_score(surf_elo1, surf_elo2)
        else:
            surf_elo1 = elo1
            surf_elo2 = elo2
            p1_surf_prob = p1_prob
        
        return {
            'p1_win_prob': p1_prob,
            'p2_win_prob': 1 - p1_prob,
            'elo_diff': elo1 - elo2,
            'surface_elo_diff': surf_elo1 - surf_elo2,
            'p1_surf_prob': p1_surf_prob,
            'p1_elo': elo1,
            'p2_elo': elo2,
            'p1_surf_elo': surf_elo1,
            'p2_surf_elo': surf_elo2,
        }


def calculate_elo_features(matches_df: pd.DataFrame) -> Tuple[ELORatingSystem, pd.DataFrame]:
    """
    Calculate ELO ratings from match data and return the system and ratings DataFrame.
    
    Args:
        matches_df: Historical match data
        
    Returns:
        Tuple of (ELORatingSystem instance, DataFrame with player ELO ratings)
    """
    elo_system = ELORatingSystem()
    ratings_df = elo_system.calculate_ratings_from_matches(matches_df)
    return elo_system, ratings_df
