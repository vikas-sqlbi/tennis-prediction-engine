"""
Matchup Model Module
Predicts match outcomes based on player profiles and identifies potential upsets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SetScorePrediction:
    """Set score prediction for a match."""
    score: str  # e.g., "2-0", "2-1", "3-1"
    probability: float
    tiebreaks_expected: int  # Number of tiebreaks likely


@dataclass
class MatchPrediction:
    """Prediction result for a match."""
    player1_name: str
    player2_name: str
    player1_win_prob: float
    player2_win_prob: float
    predicted_winner: str
    confidence: float
    upset_probability: float
    favorite: str
    underdog: str
    style_matchup: str
    key_factors: List[str]
    upset_explanation: str = ""  # Human-readable explanation for upset/no-upset
    set_scores: List[SetScorePrediction] = None  # Predicted set scores with probabilities
    tiebreak_probability: float = 0.0  # Probability of at least one tiebreak


class MatchupModel:
    """
    Predicts match outcomes based on player profiles and style matchups.
    Identifies potential upsets where underdog has stylistic advantages.
    """
    
    # Style matchup advantages (how style A performs vs style B)
    STYLE_MATCHUPS = {
        ('Big Server', 'Counterpuncher'): -0.05,      # Counterpuncher neutralizes serve
        ('Big Server', 'Grinder'): 0.05,              # Server can overpower
        ('Big Server', 'All-Court'): 0.0,
        ('Big Server', 'Aggressive Baseliner'): 0.02,
        ('Counterpuncher', 'Aggressive Baseliner'): 0.05,  # Frustrates aggressive players
        ('Counterpuncher', 'Grinder'): -0.02,
        ('Counterpuncher', 'All-Court'): 0.0,
        ('Grinder', 'Aggressive Baseliner'): 0.03,    # Wears down aggression
        ('Grinder', 'All-Court'): -0.02,
        ('Aggressive Baseliner', 'All-Court'): 0.02,
    }
    
    def __init__(self, player_profiles: pd.DataFrame, profiler=None):
        """
        Initialize matchup model.
        
        Args:
            player_profiles: DataFrame with player profiles from PlayerProfiler
            profiler: Optional PlayerProfiler instance for H2H lookups
        """
        self.profiles = player_profiles
        self.profiler = profiler  # Used for H2H lookups
        self.model = None
        self.upset_model = None  # Deprecated: Legacy upset prediction model
        # Surface-specific upset models
        self.upset_model_hard = None
        self.upset_model_clay = None
        self.upset_model_grass = None
        self.scaler = StandardScaler()
        # Surface-specific scalers
        self.scaler_hard = StandardScaler()
        self.scaler_clay = StandardScaler()
        self.scaler_grass = StandardScaler()
        self.surface_encoder = LabelEncoder()
        self.level_encoder = LabelEncoder()
        self.feature_cols = []
        self.is_trained = False
        self.upset_model_trained = False  # Legacy flag
        self.surface_models_trained = {}  # Track which surface models are trained
        self._player_cache = {}  # Cache player lookups to avoid repeated warnings
        self._missing_players = set()  # Track players we've already warned about
    
    def _get_player_profile(self, player_name: str) -> Optional[pd.Series]:
        """
        Get profile for a player by name.
        
        Handles multiple formats:
        - Full name: "Jannik Sinner" 
        - Abbreviated: "J. Sinner" or "Sinner J."
        - Last name only: "Sinner"
        """
        import re
        
        if not player_name or len(player_name) < 2:
            return None
        
        # Check cache first
        cache_key = player_name.lower()
        if cache_key in self._player_cache:
            return self._player_cache[cache_key]
        
        # Try exact match first
        mask = self.profiles['player_name'].str.lower() == player_name.lower()
        matches = self.profiles[mask]
        if len(matches) == 1:
            self._player_cache[cache_key] = matches.iloc[0]
            return matches.iloc[0]
        
        # Try contains match (original behavior)
        mask = self.profiles['player_name'].str.contains(player_name, case=False, na=False, regex=False)
        matches = self.profiles[mask]
        if len(matches) == 1:
            self._player_cache[cache_key] = matches.iloc[0]
            return matches.iloc[0]
        
        # Handle abbreviated format: "J. Lastname" or "Lastname J."
        abbrev_match = re.match(r'^([A-Z])\.\s*(.+)$', player_name)
        if abbrev_match:
            initial, lastname = abbrev_match.groups()
            # Match: starts with initial, ends with lastname
            pattern = f'^{initial}[a-z]+\\s+.*{re.escape(lastname)}$|^{initial}[a-z]+\\s+{re.escape(lastname)}$'
            mask = self.profiles['player_name'].str.match(pattern, case=False, na=False)
            matches = self.profiles[mask]
            if len(matches) == 1:
                self._player_cache[cache_key] = matches.iloc[0]
                return matches.iloc[0]
            # Fallback: just match lastname
            mask = self.profiles['player_name'].str.contains(f'\\b{re.escape(lastname)}$', case=False, na=False)
            matches = self.profiles[mask]
            if len(matches) == 1:
                self._player_cache[cache_key] = matches.iloc[0]
                return matches.iloc[0]
        
        # Handle "Lastname F." format
        abbrev_match2 = re.match(r'^(.+)\s+([A-Z])\.$', player_name)
        if abbrev_match2:
            lastname, initial = abbrev_match2.groups()
            pattern = f'^{initial}[a-z]+\\s+.*{re.escape(lastname)}$'
            mask = self.profiles['player_name'].str.match(pattern, case=False, na=False)
            matches = self.profiles[mask]
            if len(matches) == 1:
                self._player_cache[cache_key] = matches.iloc[0]
                return matches.iloc[0]
        
        # Last resort: match by last word (lastname)
        parts = player_name.split()
        if parts:
            lastname = parts[-1] if len(parts[-1]) > 2 else parts[0]
            if len(lastname) > 2:
                mask = self.profiles['player_name'].str.contains(f'\\b{re.escape(lastname)}$', case=False, na=False)
                matches = self.profiles[mask]
                if len(matches) == 1:
                    self._player_cache[cache_key] = matches.iloc[0]
                    return matches.iloc[0]
        
        # Cache the miss to avoid repeated lookups
        self._player_cache[cache_key] = None
        return None
    
    def _get_style_matchup_edge(self, style1: str, style2: str) -> float:
        """Get style matchup edge for player 1 vs player 2."""
        key = (style1, style2)
        if key in self.STYLE_MATCHUPS:
            return self.STYLE_MATCHUPS[key]
        
        # Try reverse
        key_rev = (style2, style1)
        if key_rev in self.STYLE_MATCHUPS:
            return -self.STYLE_MATCHUPS[key_rev]
        
        return 0.0
    
    def create_matchup_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create matchup features for training/prediction using vectorized operations.
        
        For each match, randomly assign winner/loser to player1/player2
        to avoid data leakage (target = 1 if player1 wins).
        """
        logger.info("Creating matchup features (vectorized)...")
        
        # Build player lookup dict for fast access
        player_lookup = {}
        for _, row in self.profiles.iterrows():
            name = row.get('player_name', '')
            if pd.notna(name):
                player_lookup[name.lower()] = row
        
        # Filter matches to those with known players
        df = matches_df.copy()
        df['winner_name_lower'] = df['winner_name'].str.lower()
        df['loser_name_lower'] = df['loser_name'].str.lower()
        
        # Check which matches have both players in profiles
        df['has_winner'] = df['winner_name_lower'].isin(player_lookup.keys())
        df['has_loser'] = df['loser_name_lower'].isin(player_lookup.keys())
        df = df[df['has_winner'] & df['has_loser']].copy()
        
        logger.info(f"Processing {len(df)} matches with known players...")
        
        # Random assignment for each match
        # swap=True means p1=loser, p2=winner -> target=0 (p1 didn't win)
        # swap=False means p1=winner, p2=loser -> target=1 (p1 won)
        np.random.seed(42)
        df['swap'] = np.random.random(len(df)) > 0.5
        df['target'] = (~df['swap']).astype(int)  # Inverted: target=1 when p1=winner (swap=False)
        
        # Get profile columns we need
        profile_cols = [
            'overall_win_pct', 'serve_dominance', 'ace_pct', 'return_points_won_pct',
            'bp_convert_pct', 'clutch_rating', 'tiebreak_win_pct', 'bp_save_pct',
            'style_cluster', 'total_matches', 'recent_form', 'recent_form_weighted',
            'win_streak', 'loss_streak', 'hard_win_pct', 'clay_win_pct', 'grass_win_pct',
            'matches_last_7_days', 'days_since_last_match', 'matches_last_30_days',
            'grand_slam_win_pct', 'masters_win_pct', 'atp500_win_pct', 'atp250_win_pct',
            'challenger_win_pct', 'futures_win_pct', 'player_name'
        ]
        
        # Create profile DataFrames indexed by lowercase name (drop duplicates)
        profiles_indexed = self.profiles.copy()
        profiles_indexed['name_lower'] = profiles_indexed['player_name'].str.lower()
        profiles_indexed = profiles_indexed.drop_duplicates(subset='name_lower').set_index('name_lower')
        
        # Get winner and loser profiles using merge for proper alignment
        df = df.reset_index(drop=True)
        winner_profiles = df[['winner_name_lower']].merge(
            profiles_indexed.reset_index(), 
            left_on='winner_name_lower', 
            right_on='name_lower', 
            how='left'
        ).drop(columns=['winner_name_lower', 'name_lower'])
        
        loser_profiles = df[['loser_name_lower']].merge(
            profiles_indexed.reset_index(), 
            left_on='loser_name_lower', 
            right_on='name_lower', 
            how='left'
        ).drop(columns=['loser_name_lower', 'name_lower'])
        
        # Apply swap (p1 = winner if not swap, else loser)
        swap_mask = df['swap'].values
        
        features = pd.DataFrame(index=df.index)
        features['target'] = df['target'].values
        features['surface'] = df['surface'].values
        features['tourney_level'] = df['tourney_level'].astype(str).values
        
        # Helper to get diff with swap
        def get_diff(col, default=0.5):
            w_vals = winner_profiles[col].fillna(default).values if col in winner_profiles else np.full(len(df), default)
            l_vals = loser_profiles[col].fillna(default).values if col in loser_profiles else np.full(len(df), default)
            # When swap is True, p1=loser, p2=winner, so diff = loser - winner
            # When swap is False, p1=winner, p2=loser, so diff = winner - loser
            return np.where(swap_mask, l_vals - w_vals, w_vals - l_vals)
        
        features['win_pct_diff'] = get_diff('overall_win_pct', 0.5)
        features['serve_dominance_diff'] = get_diff('serve_dominance', 50)
        features['ace_pct_diff'] = get_diff('ace_pct', 0)
        features['return_won_diff'] = get_diff('return_points_won_pct', 0.4)
        features['bp_convert_diff'] = get_diff('bp_convert_pct', 0.4)
        features['clutch_diff'] = get_diff('clutch_rating', 50)
        features['tiebreak_diff'] = get_diff('tiebreak_win_pct', 0.5)
        features['bp_save_diff'] = get_diff('bp_save_pct', 0.6)
        features['experience_diff'] = get_diff('total_matches', 0) / 100
        features['recent_form_diff'] = get_diff('recent_form', 0.5)
        features['recent_form_weighted_diff'] = get_diff('recent_form_weighted', 0.5)
        
        # Momentum
        w_momentum = (winner_profiles['win_streak'].fillna(0) - winner_profiles['loss_streak'].fillna(0)).values
        l_momentum = (loser_profiles['win_streak'].fillna(0) - loser_profiles['loss_streak'].fillna(0)).values
        features['momentum_diff'] = np.where(swap_mask, l_momentum - w_momentum, w_momentum - l_momentum)
        
        # Fatigue
        features['fatigue_diff'] = get_diff('matches_last_7_days', 0)
        features['rest_diff'] = get_diff('days_since_last_match', 7)
        features['schedule_load_diff'] = get_diff('matches_last_30_days', 0)
        
        # Surface-specific win percentage
        def get_surface_diff():
            result = np.zeros(len(df))
            for i, (surf, swap) in enumerate(zip(df['surface'].values, swap_mask)):
                col = f"{surf.lower()}_win_pct" if pd.notna(surf) else 'hard_win_pct'
                if col in winner_profiles.columns:
                    w_val = winner_profiles[col].iloc[i] if pd.notna(winner_profiles[col].iloc[i]) else 0.5
                    l_val = loser_profiles[col].iloc[i] if pd.notna(loser_profiles[col].iloc[i]) else 0.5
                    result[i] = (l_val - w_val) if swap else (w_val - l_val)
            return result
        
        features['surface_win_diff'] = get_surface_diff()
        
        # Tournament level diff
        level_map = {'G': 'grand_slam', 'M': 'masters', 'A': 'atp500', 'B': 'atp250', 'C': 'challenger', 'F': 'futures', 'D': 'davis_cup'}
        def get_level_diff():
            result = np.zeros(len(df))
            for i, (level, swap) in enumerate(zip(df['tourney_level'].astype(str).values, swap_mask)):
                level_name = level_map.get(level, 'atp250')
                col = f"{level_name}_win_pct"
                if col in winner_profiles.columns:
                    w_val = winner_profiles[col].iloc[i] if pd.notna(winner_profiles[col].iloc[i]) else 0.5
                    l_val = loser_profiles[col].iloc[i] if pd.notna(loser_profiles[col].iloc[i]) else 0.5
                    result[i] = (l_val - w_val) if swap else (w_val - l_val)
            return result
        
        features['tourney_level_diff'] = get_level_diff()
        
        # Style matchup edge
        w_styles = winner_profiles['style_cluster'].fillna('All-Court').values
        l_styles = loser_profiles['style_cluster'].fillna('All-Court').values
        style_edges = np.array([
            self._get_style_matchup_edge(
                l_styles[i] if swap_mask[i] else w_styles[i],
                w_styles[i] if swap_mask[i] else l_styles[i]
            ) for i in range(len(df))
        ])
        features['style_matchup_edge'] = style_edges
        
        # H2H advantage (vectorized using player_id lookup)
        if self.profiler is not None and hasattr(self.profiler, 'h2h_records'):
            logger.info("Computing H2H features...")
            w_ids = winner_profiles['player_id'].values
            l_ids = loser_profiles['player_id'].values
            h2h_records = self.profiler.h2h_records
            
            h2h_advantages = np.zeros(len(df))
            h2h_matches_arr = np.zeros(len(df))
            
            for i in range(len(df)):
                # p1 = loser if swap, else winner
                p1_id = str(l_ids[i]) if swap_mask[i] else str(w_ids[i])
                p2_id = str(w_ids[i]) if swap_mask[i] else str(l_ids[i])
                
                if p1_id in h2h_records and p2_id in h2h_records[p1_id]:
                    rec = h2h_records[p1_id][p2_id]
                    total = rec['wins'] + rec['losses']
                    if total > 0:
                        h2h_advantages[i] = rec['wins'] / total - 0.5
                        h2h_matches_arr[i] = total
            
            features['h2h_advantage'] = h2h_advantages
            features['h2h_matches'] = h2h_matches_arr
        else:
            features['h2h_advantage'] = 0.0
            features['h2h_matches'] = 0
        
        features['p1_style'] = np.where(swap_mask, l_styles, w_styles)
        features['p2_style'] = np.where(swap_mask, w_styles, l_styles)
        
        # === NEW ELO FEATURES ===
        features['elo_diff'] = get_diff('elo_overall', 1500) / 100  # Normalize
        features['elo_ratio'] = np.ones(len(df))  # Default
        w_elo = winner_profiles['elo_overall'].fillna(1500).values if 'elo_overall' in winner_profiles else np.full(len(df), 1500)
        l_elo = loser_profiles['elo_overall'].fillna(1500).values if 'elo_overall' in loser_profiles else np.full(len(df), 1500)
        p1_elo = np.where(swap_mask, l_elo, w_elo)
        p2_elo = np.where(swap_mask, w_elo, l_elo)
        features['elo_ratio'] = p1_elo / np.maximum(p2_elo, 1)  # Avoid division by zero
        
        # Surface-specific ELO difference
        def get_surface_elo_diff():
            result = np.zeros(len(df))
            for i, (surf, swap) in enumerate(zip(df['surface'].values, swap_mask)):
                col = f"elo_{surf.lower()}" if pd.notna(surf) else 'elo_hard'
                if col in winner_profiles.columns:
                    w_val = winner_profiles[col].iloc[i] if pd.notna(winner_profiles[col].iloc[i]) else 1500
                    l_val = loser_profiles[col].iloc[i] if pd.notna(loser_profiles[col].iloc[i]) else 1500
                    result[i] = ((l_val - w_val) if swap else (w_val - l_val)) / 100
            return result
        
        features['surface_elo_diff'] = get_surface_elo_diff()
        
        # ELO momentum (30-day change)
        features['elo_momentum_diff'] = get_diff('elo_30d_change', 0) / 50  # Normalize
        
        # Peak ELO difference (indicates ceiling)
        features['elo_peak_diff'] = get_diff('elo_peak', 1500) / 100
        
        # ELO-based prediction as a feature
        features['elo_predicted_p1'] = (p1_elo > p2_elo).astype(float)
        
        # === NEW SERVE FEATURES ===
        features['first_serve_pct_diff'] = get_diff('1st_serve_pct', 0.6)
        features['first_serve_won_diff'] = get_diff('1st_won_pct', 0.7)
        features['second_serve_won_diff'] = get_diff('2nd_won_pct', 0.5)
        features['serve_points_won_diff'] = get_diff('serve_points_won_pct', 0.6)
        features['df_pct_diff'] = get_diff('df_pct', 0.03)  # Lower is better, so this is inverted
        
        # === RANKING TRAJECTORY ===
        features['trajectory_diff'] = get_diff('ranking_trajectory', 0)
        
        # === UPSET-SPECIFIC FEATURES ===
        # Surface upset rate (historical upset frequency on this surface)
        surface_upset_rates = {'Hard': 0.42, 'Clay': 0.38, 'Grass': 0.45, 'Carpet': 0.40}
        features['surface_upset_rate'] = df['surface'].map(surface_upset_rates).fillna(0.42).values
        
        # Recent ranking change (how much ELO has changed in last 30 days)
        features['elo_momentum_diff'] = get_diff('elo_30d_change', 0)
        
        # Upset momentum: trajectory difference weighted by recent performance
        features['upset_momentum'] = features['trajectory_diff'] * (1 + features['recent_form_diff'])
        
        logger.info(f"Created {len(features)} matchup samples with {len(features.columns)} features")
        return features.reset_index(drop=True)
    
    def _calculate_matchup_features(
        self, 
        p1: pd.Series, 
        p2: pd.Series, 
        surface: str = 'Hard',
        tourney_level: str = 'A'
    ) -> Dict:
        """Calculate features for a matchup between two players."""
        
        # Rank-based features
        p1_rank = p1.get('current_rank', p1.get('overall_win_pct', 0.5) * 100)
        p2_rank = p2.get('current_rank', p2.get('overall_win_pct', 0.5) * 100)
        
        features = {
            # Win percentage difference
            'win_pct_diff': p1.get('overall_win_pct', 0.5) - p2.get('overall_win_pct', 0.5),
            
            # Serve metrics difference
            'serve_dominance_diff': p1.get('serve_dominance', 50) - p2.get('serve_dominance', 50),
            'ace_pct_diff': p1.get('ace_pct', 0) - p2.get('ace_pct', 0),
            
            # Return metrics difference
            'return_won_diff': p1.get('return_points_won_pct', 0.4) - p2.get('return_points_won_pct', 0.4),
            'bp_convert_diff': p1.get('bp_convert_pct', 0.4) - p2.get('bp_convert_pct', 0.4),
            
            # Clutch metrics difference
            'clutch_diff': p1.get('clutch_rating', 50) - p2.get('clutch_rating', 50),
            'tiebreak_diff': p1.get('tiebreak_win_pct', 0.5) - p2.get('tiebreak_win_pct', 0.5),
            'bp_save_diff': p1.get('bp_save_pct', 0.6) - p2.get('bp_save_pct', 0.6),
            
            # Surface-specific advantage
            'surface': surface,
        }
        
        # Surface-specific win percentage
        surface_col = f"{surface.lower()}_win_pct"
        if surface_col in p1.index and surface_col in p2.index:
            features['surface_win_diff'] = p1.get(surface_col, 0.5) - p2.get(surface_col, 0.5)
        else:
            features['surface_win_diff'] = 0.0
        
        # Style matchup edge
        style1 = p1.get('style_cluster', 'All-Court')
        style2 = p2.get('style_cluster', 'All-Court')
        features['style_matchup_edge'] = self._get_style_matchup_edge(style1, style2)
        features['p1_style'] = style1
        features['p2_style'] = style2
        
        # Total matches (experience)
        features['experience_diff'] = (
            p1.get('total_matches', 0) - p2.get('total_matches', 0)
        ) / 100  # Normalize
        
        # Recent form features (key for predicting upsets)
        features['recent_form_diff'] = p1.get('recent_form', 0.5) - p2.get('recent_form', 0.5)
        features['recent_form_weighted_diff'] = p1.get('recent_form_weighted', 0.5) - p2.get('recent_form_weighted', 0.5)
        features['momentum_diff'] = (p1.get('win_streak', 0) - p1.get('loss_streak', 0)) - \
                                    (p2.get('win_streak', 0) - p2.get('loss_streak', 0))
        
        # NEW: Fatigue features
        features['fatigue_diff'] = (
            p1.get('matches_last_7_days', 0) - p2.get('matches_last_7_days', 0)
        )
        features['rest_diff'] = (
            p1.get('days_since_last_match', 7) - p2.get('days_since_last_match', 7)
        )
        features['schedule_load_diff'] = (
            p1.get('matches_last_30_days', 0) - p2.get('matches_last_30_days', 0)
        )
        
        # NEW: Tournament level performance
        level_map = {
            'G': 'grand_slam',
            'M': 'masters',
            'A': 'atp500',
            'B': 'atp250',
            'C': 'challenger',
            'F': 'futures',
            'D': 'davis_cup'
        }
        level_name = level_map.get(tourney_level, 'atp250')
        level_col = f"{level_name}_win_pct"
        
        p1_level_pct = p1.get(level_col, p1.get('overall_win_pct', 0.5))
        p2_level_pct = p2.get(level_col, p2.get('overall_win_pct', 0.5))
        features['tourney_level_diff'] = p1_level_pct - p2_level_pct
        features['tourney_level'] = tourney_level
        
        # NEW: Head-to-head (if profiler available)
        features['h2h_advantage'] = 0.0
        if self.profiler is not None:
            p1_name = p1.get('player_name', '')
            p2_name = p2.get('player_name', '')
            h2h = self.profiler.get_h2h_record(p1_name, p2_name)
            if h2h and h2h['total_matches'] > 0:
                features['h2h_advantage'] = h2h['p1_h2h_pct'] - 0.5  # Centered around 0
                features['h2h_matches'] = h2h['total_matches']
            else:
                features['h2h_matches'] = 0
        else:
            features['h2h_matches'] = 0
        
        # === ELO FEATURES ===
        p1_elo = p1.get('elo_overall', 1500)
        p2_elo = p2.get('elo_overall', 1500)
        features['elo_diff'] = (p1_elo - p2_elo) / 100  # Normalized
        features['elo_ratio'] = p1_elo / max(p2_elo, 1)
        
        # Surface-specific ELO
        surface_elo_col = f"elo_{surface.lower()}"
        p1_surf_elo = p1.get(surface_elo_col, p1_elo)
        p2_surf_elo = p2.get(surface_elo_col, p2_elo)
        features['surface_elo_diff'] = (p1_surf_elo - p2_surf_elo) / 100
        
        # ELO momentum
        features['elo_momentum_diff'] = (
            p1.get('elo_30d_change', 0) - p2.get('elo_30d_change', 0)
        ) / 50
        
        # Peak ELO
        features['elo_peak_diff'] = (
            p1.get('elo_peak', 1500) - p2.get('elo_peak', 1500)
        ) / 100
        
        # ELO-based prediction
        features['elo_predicted_p1'] = 1.0 if p1_elo > p2_elo else 0.0
        
        # === SERVE FEATURES ===
        features['first_serve_pct_diff'] = p1.get('1st_serve_pct', 0.6) - p2.get('1st_serve_pct', 0.6)
        features['first_serve_won_diff'] = p1.get('1st_won_pct', 0.7) - p2.get('1st_won_pct', 0.7)
        features['second_serve_won_diff'] = p1.get('2nd_won_pct', 0.5) - p2.get('2nd_won_pct', 0.5)
        features['serve_points_won_diff'] = p1.get('serve_points_won_pct', 0.6) - p2.get('serve_points_won_pct', 0.6)
        features['df_pct_diff'] = p1.get('df_pct', 0.03) - p2.get('df_pct', 0.03)
        
        # === RANKING TRAJECTORY ===
        features['trajectory_diff'] = p1.get('ranking_trajectory', 0) - p2.get('ranking_trajectory', 0)
        
        # === UPSET-SPECIFIC FEATURES ===
        # Surface upset rate
        surface_upset_rates = {'Hard': 0.42, 'Clay': 0.38, 'Grass': 0.45, 'Carpet': 0.40}
        features['surface_upset_rate'] = surface_upset_rates.get(surface, 0.42)
        
        # ELO momentum difference
        features['elo_momentum_diff'] = p1.get('elo_30d_change', 0) - p2.get('elo_30d_change', 0)
        
        # Upset momentum: trajectory difference weighted by recent performance
        p1_recent_form = p1.get('recent_form', 0.5)
        p2_recent_form = p2.get('recent_form', 0.5)
        features['upset_momentum'] = features['trajectory_diff'] * (1 + (p1_recent_form - p2_recent_form))
        
        return features
    
    def train(self, matches_df: pd.DataFrame, model_type: str = 'random_forest') -> Dict:
        """
        Train the matchup prediction model.
        
        Args:
            matches_df: Historical matches dataframe
            model_type: 'random_forest', 'gradient_boosting', or 'logistic'
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Creating matchup features...")
        feature_df = self.create_matchup_features(matches_df)
        
        if len(feature_df) < 100:
            raise ValueError(f"Not enough training data: {len(feature_df)} samples")
        
        logger.info(f"Created {len(feature_df)} matchup samples")
        
        # Encode surface
        feature_df['surface_encoded'] = self.surface_encoder.fit_transform(
            feature_df['surface'].fillna('Hard')
        )
        
        # Encode tournament level (ensure string type)
        feature_df['tourney_level_encoded'] = self.level_encoder.fit_transform(
            feature_df['tourney_level'].fillna('A').astype(str)
        )
        
        # Select feature columns (including new features for better upset detection)
        self.feature_cols = [
            # Core stats
            'win_pct_diff', 'serve_dominance_diff', 'ace_pct_diff',
            'return_won_diff', 'bp_convert_diff', 'clutch_diff',
            'tiebreak_diff', 'bp_save_diff', 'surface_win_diff',
            'style_matchup_edge', 'experience_diff', 'surface_encoded',
            'recent_form_diff', 'recent_form_weighted_diff', 'momentum_diff',
            # Fatigue/schedule
            'fatigue_diff', 'rest_diff', 'schedule_load_diff',
            'tourney_level_diff', 'tourney_level_encoded',
            # H2H
            'h2h_advantage', 'h2h_matches',
            # ELO features (6 features)
            'elo_diff', 'elo_ratio', 'surface_elo_diff', 
            'elo_peak_diff', 'elo_predicted_p1',
            # Serve features
            'first_serve_pct_diff', 'first_serve_won_diff', 
            'second_serve_won_diff', 'serve_points_won_diff', 'df_pct_diff',
            # Ranking trajectory
            'trajectory_diff',
            # Upset-specific features
            'surface_upset_rate', 'elo_momentum_diff', 'upset_momentum',
        ]
        
        X = feature_df[self.feature_cols].fillna(0)
        y = feature_df['target']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=10, 
                min_samples_split=5, random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
        else:
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        
        # Train
        logger.info(f"Training {model_type} model...")
        if model_type in ['random_forest', 'gradient_boosting']:
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]
        else:
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        
        self.is_trained = True
        
        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'model_type': model_type
        }
        
        logger.info(f"Model trained. Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = importance
            self.feature_importance = importance  # Store for later access
            logger.info(f"\n{'='*50}")
            logger.info(f"TOP 10 FEATURE IMPORTANCE:")
            logger.info(f"{'='*50}")
            for i, row in importance.head(10).iterrows():
                bar = '█' * int(row['importance'] * 100)
                logger.info(f"{row['feature']:30s} {row['importance']:.4f} {bar}")
            logger.info(f"{'='*50}")
            logger.info(f"Total features: {len(self.feature_cols)}")
        
        return results
    
    def train_upset_model(self, matches_df: pd.DataFrame, upset_threshold: float = 0.40) -> Dict:
        """
        Train a dedicated upset prediction model.
        
        Args:
            matches_df: Historical matches dataframe
            upset_threshold: Probability threshold to define upsets
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training dedicated upset prediction model...")
        
        # Create features
        feature_df = self.create_matchup_features(matches_df)
        
        if len(feature_df) < 100:
            raise ValueError(f"Not enough training data: {len(feature_df)} samples")
        
        # Encode categorical features
        feature_df['surface_encoded'] = self.surface_encoder.fit_transform(
            feature_df['surface'].fillna('Hard')
        )
        feature_df['tourney_level_encoded'] = self.level_encoder.fit_transform(
            feature_df['tourney_level'].fillna('A').astype(str)
        )
        
        X = feature_df[self.feature_cols].fillna(0)
        y = feature_df['target']
        
        # Create upset labels: 1 if underdog wins, 0 otherwise
        # For upset model, target is whether the predicted underdog wins
        upset_targets = []
        for _, row in feature_df.iterrows():
            # If p1 has lower ranking (higher rank number), they're the underdog
            # This is a simplification - in practice we'd use market odds
            p1_rank = row.get('p1_rank', 100)
            p2_rank = row.get('p2_rank', 100)
            
            # If p1 has higher rank number (worse rank), they're underdog
            is_upset = (p1_rank > p2_rank) and (row['target'] == 1)  # p1 won and was underdog
            upset_targets.append(1 if is_upset else 0)
        
        y_upset = np.array(upset_targets)
        
        # Check if we have enough positive samples
        n_positive = sum(y_upset)
        if n_positive < 10:
            logger.warning(f"Very few upset cases ({n_positive}) - upset model may not be reliable")
            self.upset_model_trained = False
            return {'error': 'insufficient_upset_cases'}
        
        # Train/test split (stratify if possible)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_upset, test_size=0.2, random_state=42, stratify=y_upset
            )
        except ValueError:
            # If stratification fails, do regular split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_upset, test_size=0.2, random_state=42
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train upset model (use simpler model for better generalization)
        self.upset_model = RandomForestClassifier(
            n_estimators=100, max_depth=6,  # Simpler than main model
            min_samples_split=10, random_state=42
        )
        
        self.upset_model.fit(X_train_scaled, y_train)
        y_pred = self.upset_model.predict(X_test_scaled)
        
        # Handle predict_proba for binary classification
        proba = self.upset_model.predict_proba(X_test_scaled)
        if proba.shape[1] > 1:
            y_prob = proba[:, 1]
        else:
            # Only one class predicted
            y_prob = np.zeros(len(y_test))
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(self.upset_model, X, y_upset, cv=5)
        
        self.upset_model_trained = True
        
        results = {
            'upset_accuracy': accuracy,
            'upset_cv_mean': cv_scores.mean(),
            'upset_cv_std': cv_scores.std(),
            'upset_train_samples': len(X_train),
            'upset_test_samples': len(X_test),
            'upset_threshold': upset_threshold
        }
        
        logger.info(f"Upset model trained. Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        
        return results
    
    def train_surface_specific_upset_models(self, matches_df: pd.DataFrame, upset_threshold: float = 0.40) -> Dict:
        """
        Train separate upset prediction models for each surface (Hard, Clay, Grass).
        This addresses the finding that hard court upset predictions have different patterns.
        
        Args:
            matches_df: Historical matches dataframe
            upset_threshold: Probability threshold to define upsets
        
        Returns:
            Dictionary with training metrics for each surface
        """
        logger.info("="*60)
        logger.info("Training surface-specific upset prediction models...")
        logger.info("="*60)
        
        # Create features for all matches
        feature_df = self.create_matchup_features(matches_df)
        
        if len(feature_df) < 100:
            raise ValueError(f"Not enough training data: {len(feature_df)} samples")
        
        # Encode categorical features
        feature_df['surface_encoded'] = self.surface_encoder.fit_transform(
            feature_df['surface'].fillna('Hard')
        )
        feature_df['tourney_level_encoded'] = self.level_encoder.fit_transform(
            feature_df['tourney_level'].fillna('A').astype(str)
        )
        
        # Create upset labels for all matches
        upset_targets = []
        for _, row in feature_df.iterrows():
            p1_rank = row.get('p1_rank', 100)
            p2_rank = row.get('p2_rank', 100)
            is_upset = (p1_rank > p2_rank) and (row['target'] == 1)
            upset_targets.append(1 if is_upset else 0)
        
        feature_df['upset_target'] = upset_targets
        
        results = {}
        
        # Train model for each surface
        for surface in ['Hard', 'Clay', 'Grass']:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {surface} court upset model...")
            logger.info(f"{'='*60}")
            
            # Filter data for this surface
            surface_df = feature_df[feature_df['surface'] == surface].copy()
            
            if len(surface_df) < 50:
                logger.warning(f"Not enough {surface} court data ({len(surface_df)} matches) - skipping")
                results[surface.lower()] = {'error': 'insufficient_data', 'samples': len(surface_df)}
                continue
            
            X = surface_df[self.feature_cols].fillna(0)
            y = surface_df['upset_target']
            
            # Check positive samples
            n_positive = sum(y)
            n_negative = len(y) - n_positive
            logger.info(f"{surface} data: {len(surface_df)} matches, {n_positive} upsets ({n_positive/len(surface_df)*100:.1f}%), {n_negative} favorites")
            
            if n_positive < 10:
                logger.warning(f"Very few upset cases ({n_positive}) for {surface} - model may not be reliable")
                results[surface.lower()] = {'error': 'insufficient_upset_cases', 'upset_count': n_positive}
                continue
            
            # Train/test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # Get surface-specific scaler
            if surface == 'Hard':
                scaler = self.scaler_hard
            elif surface == 'Clay':
                scaler = self.scaler_clay
            else:
                scaler = self.scaler_grass
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train surface-specific upset model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Store model
            if surface == 'Hard':
                self.upset_model_hard = model
            elif surface == 'Clay':
                self.upset_model_clay = model
            else:
                self.upset_model_grass = model
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            X_full_scaled = scaler.fit_transform(X)
            cv_scores = cross_val_score(model, X_full_scaled, y, cv=min(5, len(X)//20))
            
            # Calculate precision/recall
            from sklearn.metrics import precision_score, recall_score
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            
            results[surface.lower()] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'upset_count': n_positive,
                'upset_rate': n_positive / len(surface_df)
            }
            
            self.surface_models_trained[surface.lower()] = True
            
            logger.info(f"{surface} model trained:")
            logger.info(f"  Accuracy: {accuracy:.1%}")
            logger.info(f"  Precision: {precision:.1%}")
            logger.info(f"  Recall: {recall:.1%}")
            logger.info(f"  CV Score: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
        
        logger.info(f"\n{'='*60}")
        logger.info("Surface-specific upset models training complete!")
        logger.info(f"{'='*60}")
        
        return results
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance DataFrame after training."""
        if hasattr(self, 'feature_importance'):
            return self.feature_importance
        return None
    
    def predict_match(
        self, 
        player1_name: str, 
        player2_name: str, 
        surface: str = 'Hard',
        favorite: Optional[str] = None,
        tourney_level: str = 'A',
        tournament_name: str = ''
    ) -> Optional[MatchPrediction]:
        """
        Predict outcome of a match between two players.
        
        Args:
            player1_name: Name of first player
            player2_name: Name of second player
            surface: Playing surface
            favorite: Name of betting favorite (for upset detection)
            tourney_level: Tournament level (G, M, A, B, C, F, D)
            tournament_name: Full tournament name (used to detect best-of-5 formats)
        
        Returns:
            MatchPrediction object
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        p1 = self._get_player_profile(player1_name)
        p2 = self._get_player_profile(player2_name)
        
        if p1 is None:
            # Only log once per player to reduce noise
            if player1_name not in self._missing_players:
                logger.debug(f"Player not found: {player1_name}")
                self._missing_players.add(player1_name)
            return None
        if p2 is None:
            if player2_name not in self._missing_players:
                logger.debug(f"Player not found: {player2_name}")
                self._missing_players.add(player2_name)
            return None
        
        # Calculate features
        features = self._calculate_matchup_features(p1, p2, surface, tourney_level)
        features['surface_encoded'] = self.surface_encoder.transform([surface])[0]
        
        # Encode tourney level (handle unseen levels gracefully)
        try:
            features['tourney_level_encoded'] = self.level_encoder.transform([str(tourney_level)])[0]
        except ValueError:
            features['tourney_level_encoded'] = self.level_encoder.transform(['A'])[0]
        
        # Create feature vector (tolerate missing columns in live inference)
        feature_df = pd.DataFrame([features])
        X = feature_df.reindex(columns=self.feature_cols, fill_value=0)
        
        # Predict
        p1_win_prob = self.model.predict_proba(X)[0][1]
        p2_win_prob = 1 - p1_win_prob
        
        predicted_winner = player1_name if p1_win_prob > 0.5 else player2_name
        confidence = max(p1_win_prob, p2_win_prob)
        
        # FIX: NaN Market Odds Handling (bug/fixing-scraper-odds-nan)
        # Previously, when market odds were NaN, we incorrectly used win_pct as a fallback
        # to determine favorite/underdog. This caused favorites to be mislabeled as underdogs.
        # Now we ONLY use market odds to determine favorite/underdog.
        # If odds are NaN/unavailable, favorite and underdog are set to None.
        if favorite:
            fav = favorite
            underdog = player2_name if favorite == player1_name else player1_name
        else:
            # No market odds available - cannot determine favorite/underdog
            fav = None
            underdog = None
        
        # FIX: Upset probability requires valid market odds
        # Without knowing who the market favorite is, upset probability is meaningless
        # UPDATE: Use surface-specific upset models when available
        if fav is None:
            upset_prob = 0.0  # Cannot calculate upset without market odds
        else:
            # Try to use surface-specific upset model if available
            surface_key = surface.lower() if surface else 'hard'
            surface_model = None
            surface_scaler = None
            
            if surface_key == 'hard' and self.upset_model_hard is not None:
                surface_model = self.upset_model_hard
                surface_scaler = self.scaler_hard
            elif surface_key == 'clay' and self.upset_model_clay is not None:
                surface_model = self.upset_model_clay
                surface_scaler = self.scaler_clay
            elif surface_key == 'grass' and self.upset_model_grass is not None:
                surface_model = self.upset_model_grass
                surface_scaler = self.scaler_grass
            
            # Use surface-specific model if available, otherwise fall back to probability-based
            if surface_model is not None and surface_scaler is not None:
                try:
                    # Scale features for upset model
                    X_scaled = surface_scaler.transform(X)
                    # Get upset probability from surface-specific model
                    # Model was trained to predict: 1 = upset occurred, 0 = favorite won
                    proba = surface_model.predict_proba(X_scaled)[0]
                    upset_prob = proba[1]  # Probability of class 1 (upset)
                    logger.debug(f"Using {surface_key} upset model: upset_prob={upset_prob:.1%}")
                except Exception as e:
                    logger.debug(f"Error using {surface_key} upset model: {e}, falling back to win probability")
                    # Fallback to original method
                    if fav == player1_name:
                        upset_prob = p2_win_prob
                    else:
                        upset_prob = p1_win_prob
            else:
                # Fall back to win probability method
                logger.debug(f"No {surface_key} upset model available, using win probability method")
                if fav == player1_name:
                    upset_prob = p2_win_prob  # Underdog (p2) winning
                else:
                    upset_prob = p1_win_prob  # Underdog (p1) winning
        
        # Style matchup description
        style1 = p1.get('style_cluster', 'All-Court')
        style2 = p2.get('style_cluster', 'All-Court')
        style_matchup = f"{style1} vs {style2}"
        
        # Key factors for upset
        key_factors = self._get_key_factors(p1, p2, surface, fav, underdog)
        
        # Generate upset explanation
        upset_explanation = self._generate_upset_explanation(
            predicted_winner, fav, underdog, key_factors, confidence, p1, p2
        )
        
        # Determine if best-of-5 match (Grand Slams, Next Gen Finals, Davis Cup finals)
        best_of = self._get_best_of_sets(tourney_level, tournament_name)
        
        # Predict set scores and tiebreaks
        set_scores, tiebreak_prob = self._predict_set_scores(
            p1, p2, max(p1_win_prob, p2_win_prob), best_of=best_of
        )
        
        return MatchPrediction(
            player1_name=player1_name,
            player2_name=player2_name,
            player1_win_prob=p1_win_prob,
            player2_win_prob=p2_win_prob,
            predicted_winner=predicted_winner,
            confidence=confidence,
            upset_probability=upset_prob,
            favorite=fav,
            underdog=underdog,
            style_matchup=style_matchup,
            key_factors=key_factors,
            upset_explanation=upset_explanation,
            set_scores=set_scores,
            tiebreak_probability=tiebreak_prob
        )
    
    def _get_best_of_sets(self, tourney_level: str, tournament_name: str = '') -> int:
        """
        Determine if match is best-of-3 or best-of-5 based on tournament level and name.
        
        Best-of-5 formats:
        - Grand Slams (G) - men's singles
        - Davis Cup finals (D) - some ties
        - Next Gen ATP Finals - uses best-of-5 (first to 4 games per set)
        - ATP Finals - round robin and knockouts
        """
        # Grand Slams and Davis Cup use best-of-5 for men
        if tourney_level in ['G', 'D']:
            return 5
        
        # Check tournament name for special events
        if tournament_name:
            name_lower = tournament_name.lower()
            # Next Gen ATP Finals, ATP Finals use best-of-5
            if 'next gen' in name_lower and 'finals' in name_lower:
                return 5
            if 'atp finals' in name_lower and 'next gen' not in name_lower:
                return 5  # ATP Finals (Turin) is also best-of-3 actually, but close matches
            # Grand Slam names
            if any(gs in name_lower for gs in ['australian open', 'roland garros', 'french open', 
                                                 'wimbledon', 'us open']):
                return 5
        
        return 3
    
    def _get_key_factors(
        self, 
        p1: pd.Series, 
        p2: pd.Series, 
        surface: str,
        favorite: Optional[str],
        underdog: Optional[str],
        tourney_level: str = 'A'
    ) -> List[str]:
        """Identify key factors that could lead to an upset."""
        factors = []
        
        # FIX: Handle missing market odds gracefully
        # When favorite/underdog are None (due to NaN odds), return neutral message
        if favorite is None or underdog is None:
            factors.append("No market odds available - cannot assess upset factors")
            return factors
        
        # Determine which profile is the underdog
        if underdog in str(p1.get('player_name', '')):
            ud, fav_p = p1, p2
        else:
            ud, fav_p = p2, p1
        
        # Check for underdog advantages
        if ud.get('clutch_rating', 50) > fav_p.get('clutch_rating', 50):
            factors.append(f"Underdog has higher clutch rating ({ud.get('clutch_rating', 50):.0f} vs {fav_p.get('clutch_rating', 50):.0f})")
        
        surface_col = f"{surface.lower()}_win_pct"
        if surface_col in ud.index and surface_col in fav_p.index:
            if ud.get(surface_col, 0.5) > fav_p.get(surface_col, 0.5):
                factors.append(f"Underdog better on {surface} ({ud.get(surface_col, 0.5):.1%} vs {fav_p.get(surface_col, 0.5):.1%})")
        
        if ud.get('tiebreak_win_pct', 0.5) > fav_p.get('tiebreak_win_pct', 0.5) + 0.05:
            factors.append(f"Underdog stronger in tiebreaks ({ud.get('tiebreak_win_pct', 0.5):.1%})")
        
        # Style matchup
        ud_style = ud.get('style_cluster', 'All-Court')
        fav_style = fav_p.get('style_cluster', 'All-Court')
        edge = self._get_style_matchup_edge(ud_style, fav_style)
        if edge > 0:
            factors.append(f"Favorable style matchup: {ud_style} vs {fav_style}")
        
        # Recent form advantage
        ud_form = ud.get('recent_form', 0.5)
        fav_form = fav_p.get('recent_form', 0.5)
        if ud_form > fav_form + 0.1:
            factors.append(f"Underdog in better recent form ({ud_form:.0%} vs {fav_form:.0%})")
        
        # Momentum (win streak)
        ud_streak = ud.get('win_streak', 0)
        fav_streak = fav_p.get('loss_streak', 0)
        if ud_streak >= 3:
            factors.append(f"Underdog on {ud_streak}-match win streak")
        if fav_streak >= 2:
            factors.append(f"Favorite on {fav_streak}-match losing streak")
        
        # NEW: H2H advantage
        if self.profiler is not None:
            ud_name = ud.get('player_name', '')
            fav_name = fav_p.get('player_name', '')
            h2h = self.profiler.get_h2h_record(ud_name, fav_name)
            if h2h and h2h['total_matches'] >= 3:
                if h2h['p1_h2h_pct'] > 0.55:
                    factors.append(f"Underdog leads H2H {h2h['p1_wins']}-{h2h['p2_wins']}")
        
        # NEW: Fatigue advantage
        ud_fatigue = ud.get('matches_last_7_days', 0)
        fav_fatigue = fav_p.get('matches_last_7_days', 0)
        if fav_fatigue >= 3 and ud_fatigue <= 1:
            factors.append(f"Favorite fatigued ({fav_fatigue} matches in 7 days)")
        
        # NEW: Tournament level experience
        level_map = {'G': 'grand_slam', 'M': 'masters', 'A': 'atp500', 'B': 'atp250'}
        level_name = level_map.get(tourney_level, '')
        if level_name:
            level_col = f"{level_name}_matches"
            ud_exp = ud.get(level_col, 0)
            fav_exp = fav_p.get(level_col, 0)
            if ud_exp > fav_exp * 1.5 and ud_exp >= 10:
                factors.append(f"Underdog more experienced at this level ({ud_exp} vs {fav_exp} matches)")
        
        if not factors:
            factors.append("No significant underdog advantages identified")
        
        return factors
    
    def _generate_upset_explanation(
        self,
        predicted_winner: str,
        favorite: Optional[str],
        underdog: Optional[str],
        key_factors: List[str],
        confidence: float,
        p1: pd.Series,
        p2: pd.Series
    ) -> str:
        """
        Generate a human-readable explanation for why an upset is or isn't predicted.
        """
        # FIX: Provide neutral explanation when market odds are NaN
        # Without market odds, we cannot determine if prediction is an "upset"
        if favorite is None or underdog is None:
            return f"{predicted_winner} predicted to win ({confidence:.0%} confidence). Market odds unavailable - upset status unknown."
        
        is_upset = predicted_winner == underdog
        
        if is_upset:
            # Build upset explanation
            reasons = []
            for factor in key_factors[:3]:  # Top 3 factors
                if "H2H" in factor:
                    reasons.append(f"historical head-to-head dominance")
                elif "form" in factor.lower():
                    reasons.append(f"superior recent form")
                elif "clutch" in factor.lower():
                    reasons.append(f"better clutch performance")
                elif "surface" in factor.lower():
                    reasons.append(f"surface advantage")
                elif "style" in factor.lower():
                    reasons.append(f"favorable style matchup")
                elif "streak" in factor.lower():
                    reasons.append(f"momentum advantage")
                elif "fatigue" in factor.lower():
                    reasons.append(f"opponent fatigue")
            
            if reasons:
                reason_text = ", ".join(reasons[:2])
                return f"UPSET PREDICTED: {underdog} to beat {favorite} ({confidence:.0%} confidence). Key reasons: {reason_text}."
            else:
                return f"UPSET PREDICTED: {underdog} to beat {favorite} ({confidence:.0%} confidence) based on model analysis."
        else:
            # Build no-upset explanation
            fav_profile = p1 if favorite in str(p1.get('player_name', '')) else p2
            und_profile = p2 if favorite in str(p1.get('player_name', '')) else p1
            
            win_diff = fav_profile.get('overall_win_pct', 0.5) - und_profile.get('overall_win_pct', 0.5)
            
            if win_diff > 0.15:
                return f"NO UPSET: {favorite} significantly stronger overall ({fav_profile.get('overall_win_pct', 0.5):.0%} vs {und_profile.get('overall_win_pct', 0.5):.0%} win rate)."
            elif confidence > 0.75:
                return f"NO UPSET: {favorite} heavily favored with {confidence:.0%} confidence."
            else:
                return f"NO UPSET: {favorite} slightly favored. Close match expected ({confidence:.0%} confidence)."
    
    def _predict_set_scores(
        self,
        p1: pd.Series,
        p2: pd.Series,
        winner_prob: float,
        best_of: int = 3
    ) -> Tuple[List[SetScorePrediction], float]:
        """
        Predict likely set scores based on player profiles.
        
        Args:
            p1, p2: Player profiles
            winner_prob: Probability of the predicted winner winning
            best_of: Number of sets (3 or 5)
        
        Returns:
            Tuple of (list of SetScorePrediction, tiebreak probability)
        """
        # Get tiebreak tendencies
        p1_tb = p1.get('tiebreak_win_pct', 0.5)
        p2_tb = p2.get('tiebreak_win_pct', 0.5)
        avg_tb_skill = (p1_tb + p2_tb) / 2
        
        # Get serve dominance - high serve dominance = more tiebreaks
        p1_serve = p1.get('serve_dominance', 50) / 100
        p2_serve = p2.get('serve_dominance', 50) / 100
        avg_serve = (p1_serve + p2_serve) / 2
        
        # Base tiebreak probability: higher serve dominance = more tiebreaks
        # Average ATP match has ~20-25% chance of tiebreak per set
        tiebreak_per_set = 0.2 + (avg_serve * 0.15)  # 20-35% per set
        
        # Probability of at least one tiebreak in the match
        if best_of == 3:
            # P(at least one TB in 2-3 sets)
            expected_sets = 2.5 if winner_prob < 0.65 else 2.2
            tiebreak_prob = 1 - (1 - tiebreak_per_set) ** expected_sets
        else:
            expected_sets = 4 if winner_prob < 0.65 else 3.5
            tiebreak_prob = 1 - (1 - tiebreak_per_set) ** expected_sets
        
        # Predict set scores based on winner probability
        set_scores = []
        
        if best_of == 3:
            # Higher confidence = more likely 2-0
            if winner_prob >= 0.75:
                set_scores = [
                    SetScorePrediction("2-0", 0.55, 0),
                    SetScorePrediction("2-1", 0.35, 1 if tiebreak_prob > 0.4 else 0),
                    SetScorePrediction("0-2", 0.07, 0),
                    SetScorePrediction("1-2", 0.03, 0),
                ]
            elif winner_prob >= 0.60:
                set_scores = [
                    SetScorePrediction("2-0", 0.40, 0),
                    SetScorePrediction("2-1", 0.40, 1 if tiebreak_prob > 0.35 else 0),
                    SetScorePrediction("0-2", 0.12, 0),
                    SetScorePrediction("1-2", 0.08, 0),
                ]
            else:  # Close match
                set_scores = [
                    SetScorePrediction("2-1", 0.45, 1 if tiebreak_prob > 0.3 else 0),
                    SetScorePrediction("2-0", 0.25, 0),
                    SetScorePrediction("1-2", 0.20, 1 if tiebreak_prob > 0.3 else 0),
                    SetScorePrediction("0-2", 0.10, 0),
                ]
        else:  # Best of 5
            if winner_prob >= 0.75:
                set_scores = [
                    SetScorePrediction("3-0", 0.35, 0),
                    SetScorePrediction("3-1", 0.40, 1 if tiebreak_prob > 0.5 else 0),
                    SetScorePrediction("3-2", 0.15, 1 if tiebreak_prob > 0.4 else 0),
                    SetScorePrediction("0-3", 0.05, 0),
                    SetScorePrediction("1-3", 0.04, 0),
                    SetScorePrediction("2-3", 0.01, 0),
                ]
            elif winner_prob >= 0.60:
                set_scores = [
                    SetScorePrediction("3-1", 0.35, 1 if tiebreak_prob > 0.45 else 0),
                    SetScorePrediction("3-0", 0.25, 0),
                    SetScorePrediction("3-2", 0.25, 1 if tiebreak_prob > 0.4 else 0),
                    SetScorePrediction("1-3", 0.08, 0),
                    SetScorePrediction("0-3", 0.04, 0),
                    SetScorePrediction("2-3", 0.03, 0),
                ]
            else:  # Close match
                set_scores = [
                    SetScorePrediction("3-2", 0.35, 1 if tiebreak_prob > 0.35 else 0),
                    SetScorePrediction("3-1", 0.25, 1 if tiebreak_prob > 0.4 else 0),
                    SetScorePrediction("2-3", 0.20, 1 if tiebreak_prob > 0.35 else 0),
                    SetScorePrediction("3-0", 0.10, 0),
                    SetScorePrediction("1-3", 0.07, 0),
                    SetScorePrediction("0-3", 0.03, 0),
                ]
        
        return set_scores, tiebreak_prob
    
    def find_upset_opportunities(
        self, 
        upcoming_matches: pd.DataFrame,
        min_upset_prob: float = 0.40
    ) -> pd.DataFrame:
        """
        Find matches with high upset potential.
        
        Args:
            upcoming_matches: DataFrame with player1, player2, surface, favorite columns
            min_upset_prob: Minimum upset probability threshold
        
        Returns:
            DataFrame with matches sorted by upset probability
        """
        predictions = []
        
        for _, row in upcoming_matches.iterrows():
            pred = self.predict_match(
                row['player1'],
                row['player2'],
                row.get('surface', 'Hard'),
                row.get('favorite', None)
            )
            
            if pred:
                predictions.append({
                    'player1': pred.player1_name,
                    'player2': pred.player2_name,
                    'surface': row.get('surface', 'Hard'),
                    'tournament': row.get('tournament', 'Unknown'),
                    'favorite': pred.favorite,
                    'underdog': pred.underdog,
                    'predicted_winner': pred.predicted_winner,
                    'p1_win_prob': pred.player1_win_prob,
                    'p2_win_prob': pred.player2_win_prob,
                    'upset_probability': pred.upset_probability,
                    'confidence': pred.confidence,
                    'style_matchup': pred.style_matchup,
                    'key_factors': '; '.join(pred.key_factors),
                    'upset_alert': pred.upset_probability >= min_upset_prob
                })
        
        results_df = pd.DataFrame(predictions)
        if len(results_df) > 0:
            results_df = results_df.sort_values('upset_probability', ascending=False)
        
        return results_df
    
    def save_model(self, filepath: Path):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'surface_encoder': self.surface_encoder,
            'feature_cols': self.feature_cols
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path):
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.surface_encoder = model_data['surface_encoder']
        self.feature_cols = model_data['feature_cols']
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    from data_loader import load_matches, preprocess_matches
    from player_profiler import PlayerProfiler
    
    print("Loading data...")
    matches = load_matches(years=[2022, 2023, 2024])
    matches = preprocess_matches(matches)
    
    print("Building player profiles...")
    profiler = PlayerProfiler(matches)
    profiles = profiler.build_all_profiles(min_matches=20)
    
    print("Training matchup model...")
    model = MatchupModel(profiles, profiler=profiler)  # Pass profiler for H2H
    results = model.train(matches, model_type='random_forest')
    
    print(f"\nModel Performance:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  CV Score: {results['cv_mean']:.4f} +/- {results['cv_std']:.4f}")
    
    # Test prediction
    print("\nTest prediction: Sinner vs Alcaraz on Hard")
    pred = model.predict_match("Sinner", "Alcaraz", "Hard")
    if pred:
        print(f"  Predicted winner: {pred.predicted_winner}")
        print(f"  Confidence: {pred.confidence:.1%}")
        print(f"  Style matchup: {pred.style_matchup}")
        print(f"  Key factors: {pred.key_factors}")
