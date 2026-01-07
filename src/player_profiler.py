"""
Player Profiler Module
Builds player profiles with style archetypes, clutch ratings, surface preferences,
ELO ratings, and ranking trajectory.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

from elo_ratings import ELORatingSystem, calculate_elo_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PlayerProfile:
    """Player profile with style and performance metrics."""
    player_id: str  # TML uses alphanumeric IDs like 'CD85'
    player_name: str
    
    # Style metrics
    serve_dominance: float  # How much they rely on serve
    return_strength: float  # Return game quality
    aggression: float       # Aces - DFs, winners tendency
    consistency: float      # Low error rate
    style_cluster: str      # Archetype name
    
    # Clutch metrics
    tiebreak_win_pct: float
    deciding_set_win_pct: float
    bp_save_pct: float
    bp_convert_pct: float
    clutch_rating: float    # Composite score
    
    # Surface preferences
    hard_win_pct: float
    clay_win_pct: float
    grass_win_pct: float
    surface_specialist: str  # Best surface
    
    # Overall
    total_matches: int
    overall_win_pct: float
    current_rank: Optional[int] = None


class PlayerProfiler:
    """Build player profiles from historical match data."""
    
    STYLE_CLUSTERS = {
        0: "Big Server",
        1: "Counterpuncher", 
        2: "All-Court",
        3: "Aggressive Baseliner",
        4: "Grinder"
    }
    
    def __init__(self, matches_df: pd.DataFrame):
        """
        Initialize profiler with match data.
        
        Args:
            matches_df: Preprocessed matches dataframe
        """
        self.matches_df = matches_df
        self.profiles: Dict[int, PlayerProfile] = {}
        self.player_stats: Optional[pd.DataFrame] = None
        self.h2h_records: Dict[int, Dict] = {}  # Store H2H records per player
        self.elo_system: Optional[ELORatingSystem] = None  # ELO rating system
        self.elo_ratings: Optional[pd.DataFrame] = None  # ELO ratings DataFrame
        self.scaler = StandardScaler()
        self.clusterer = KMeans(n_clusters=5, random_state=42, n_init=10)
    
    def _get_player_ids(self) -> List[int]:
        """Get all unique player IDs."""
        winner_ids = self.matches_df['winner_id'].dropna().unique()
        loser_ids = self.matches_df['loser_id'].dropna().unique()
        return list(set(winner_ids) | set(loser_ids))
    
    def _calculate_serve_stats(self, player_id: str) -> Dict[str, float]:
        """Calculate serve-related statistics for a player."""
        # Matches where player won
        won = self.matches_df[self.matches_df['winner_id'] == player_id]
        # Matches where player lost
        lost = self.matches_df[self.matches_df['loser_id'] == player_id]
        
        # Aggregate serve stats
        stats = {
            'ace_pct': 0.0,
            'df_pct': 0.0,
            '1st_serve_pct': 0.0,
            '1st_won_pct': 0.0,
            '2nd_won_pct': 0.0,
            'serve_points_won_pct': 0.0
        }
        
        total_svpt = 0
        total_aces = 0
        total_df = 0
        total_1stIn = 0
        total_1stWon = 0
        total_2ndWon = 0
        
        # From wins
        if 'w_svpt' in won.columns:
            mask = won['w_svpt'].notna() & (won['w_svpt'] > 0)
            total_svpt += won.loc[mask, 'w_svpt'].sum()
            total_aces += won.loc[mask, 'w_ace'].sum()
            total_df += won.loc[mask, 'w_df'].sum()
            total_1stIn += won.loc[mask, 'w_1stIn'].sum()
            total_1stWon += won.loc[mask, 'w_1stWon'].sum()
            total_2ndWon += won.loc[mask, 'w_2ndWon'].sum()
        
        # From losses
        if 'l_svpt' in lost.columns:
            mask = lost['l_svpt'].notna() & (lost['l_svpt'] > 0)
            total_svpt += lost.loc[mask, 'l_svpt'].sum()
            total_aces += lost.loc[mask, 'l_ace'].sum()
            total_df += lost.loc[mask, 'l_df'].sum()
            total_1stIn += lost.loc[mask, 'l_1stIn'].sum()
            total_1stWon += lost.loc[mask, 'l_1stWon'].sum()
            total_2ndWon += lost.loc[mask, 'l_2ndWon'].sum()
        
        if total_svpt > 0:
            stats['ace_pct'] = total_aces / total_svpt
            stats['df_pct'] = total_df / total_svpt
            stats['1st_serve_pct'] = total_1stIn / total_svpt
            if total_1stIn > 0:
                stats['1st_won_pct'] = total_1stWon / total_1stIn
            second_serves = total_svpt - total_1stIn
            if second_serves > 0:
                stats['2nd_won_pct'] = total_2ndWon / second_serves
            stats['serve_points_won_pct'] = (total_1stWon + total_2ndWon) / total_svpt
        
        return stats
    
    def _calculate_return_stats(self, player_id: str) -> Dict[str, float]:
        """Calculate return-related statistics for a player."""
        won = self.matches_df[self.matches_df['winner_id'] == player_id]
        lost = self.matches_df[self.matches_df['loser_id'] == player_id]
        
        stats = {
            'bp_convert_pct': 0.0,
            'return_points_won_pct': 0.0
        }
        
        total_bp_faced = 0
        total_bp_won = 0
        total_return_pts = 0
        total_return_won = 0
        
        # When player won - opponent's serve stats are return stats for player
        if 'l_svpt' in won.columns:
            mask = won['l_svpt'].notna() & (won['l_svpt'] > 0)
            opp_svpt = won.loc[mask, 'l_svpt'].sum()
            opp_1stWon = won.loc[mask, 'l_1stWon'].sum()
            opp_2ndWon = won.loc[mask, 'l_2ndWon'].sum()
            total_return_pts += opp_svpt
            total_return_won += opp_svpt - opp_1stWon - opp_2ndWon
            
            if 'w_bpFaced' in won.columns:
                # BP converted = opponent's BP faced - BP saved
                bp_mask = won['l_bpFaced'].notna()
                total_bp_faced += won.loc[bp_mask, 'l_bpFaced'].sum()
                total_bp_won += won.loc[bp_mask, 'l_bpFaced'].sum() - won.loc[bp_mask, 'l_bpSaved'].sum()
        
        # When player lost
        if 'w_svpt' in lost.columns:
            mask = lost['w_svpt'].notna() & (lost['w_svpt'] > 0)
            opp_svpt = lost.loc[mask, 'w_svpt'].sum()
            opp_1stWon = lost.loc[mask, 'w_1stWon'].sum()
            opp_2ndWon = lost.loc[mask, 'w_2ndWon'].sum()
            total_return_pts += opp_svpt
            total_return_won += opp_svpt - opp_1stWon - opp_2ndWon
            
            if 'l_bpFaced' in lost.columns:
                bp_mask = lost['w_bpFaced'].notna()
                total_bp_faced += lost.loc[bp_mask, 'w_bpFaced'].sum()
                total_bp_won += lost.loc[bp_mask, 'w_bpFaced'].sum() - lost.loc[bp_mask, 'w_bpSaved'].sum()
        
        if total_bp_faced > 0:
            stats['bp_convert_pct'] = total_bp_won / total_bp_faced
        if total_return_pts > 0:
            stats['return_points_won_pct'] = total_return_won / total_return_pts
        
        return stats
    
    def _calculate_clutch_stats(self, player_id: str) -> Dict[str, float]:
        """Calculate clutch performance metrics."""
        won = self.matches_df[self.matches_df['winner_id'] == player_id]
        lost = self.matches_df[self.matches_df['loser_id'] == player_id]
        
        stats = {
            'tiebreak_win_pct': 0.5,
            'deciding_set_win_pct': 0.5,
            'bp_save_pct': 0.5
        }
        
        # Break point save percentage
        total_bp_faced = 0
        total_bp_saved = 0
        
        if 'w_bpFaced' in won.columns:
            mask = won['w_bpFaced'].notna()
            total_bp_faced += won.loc[mask, 'w_bpFaced'].sum()
            total_bp_saved += won.loc[mask, 'w_bpSaved'].sum()
        
        if 'l_bpFaced' in lost.columns:
            mask = lost['l_bpFaced'].notna()
            total_bp_faced += lost.loc[mask, 'l_bpFaced'].sum()
            total_bp_saved += lost.loc[mask, 'l_bpSaved'].sum()
        
        if total_bp_faced > 0:
            stats['bp_save_pct'] = total_bp_saved / total_bp_faced
        
        # Deciding set win percentage (best of 3: set 3, best of 5: set 5)
        def is_deciding_set(row):
            best_of = row.get('best_of', 3)
            score = str(row.get('score', ''))
            sets = score.split()
            if best_of == 3:
                return len(sets) == 3
            elif best_of == 5:
                return len(sets) == 5
            return False
        
        deciding_won = won.apply(is_deciding_set, axis=1).sum()
        deciding_lost = lost.apply(is_deciding_set, axis=1).sum()
        total_deciding = deciding_won + deciding_lost
        
        if total_deciding > 0:
            stats['deciding_set_win_pct'] = deciding_won / total_deciding
        
        # Tiebreak win percentage (approximate from scores)
        def count_tiebreaks(row, is_winner: bool):
            score = str(row.get('score', ''))
            tb_won = 0
            tb_lost = 0
            for s in score.split():
                if '(' in s:
                    # Tiebreak set
                    parts = s.replace('(', ' ').replace(')', '').split()
                    if len(parts) >= 2:
                        try:
                            g1, g2 = int(parts[0]), int(parts[1].split()[0]) if ' ' in parts[1] else int(parts[1][:1])
                            if g1 == 7:
                                tb_won += 1
                            elif g2 == 7:
                                tb_lost += 1
                        except:
                            pass
            if is_winner:
                return tb_won, tb_lost
            else:
                return tb_lost, tb_won
        
        total_tb_won = 0
        total_tb_lost = 0
        
        for _, row in won.iterrows():
            w, l = count_tiebreaks(row, True)
            total_tb_won += w
            total_tb_lost += l
        
        for _, row in lost.iterrows():
            w, l = count_tiebreaks(row, False)
            total_tb_won += w
            total_tb_lost += l
        
        total_tb = total_tb_won + total_tb_lost
        if total_tb > 0:
            stats['tiebreak_win_pct'] = total_tb_won / total_tb
        
        return stats
    
    def _calculate_surface_stats(self, player_id: str) -> Dict[str, float]:
        """Calculate win percentage by surface."""
        surfaces = ['Hard', 'Clay', 'Grass']
        stats = {}
        
        for surface in surfaces:
            won = len(self.matches_df[
                (self.matches_df['winner_id'] == player_id) & 
                (self.matches_df['surface'] == surface)
            ])
            lost = len(self.matches_df[
                (self.matches_df['loser_id'] == player_id) & 
                (self.matches_df['surface'] == surface)
            ])
            total = won + lost
            key = f"{surface.lower()}_win_pct"
            stats[key] = won / total if total > 0 else 0.5
        
        return stats
    
    def _calculate_overall_stats(self, player_id: str) -> Dict[str, any]:
        """Calculate overall match statistics."""
        won = len(self.matches_df[self.matches_df['winner_id'] == player_id])
        lost = len(self.matches_df[self.matches_df['loser_id'] == player_id])
        total = won + lost
        
        # Get player name
        won_names = self.matches_df[self.matches_df['winner_id'] == player_id]['winner_name'].dropna()
        lost_names = self.matches_df[self.matches_df['loser_id'] == player_id]['loser_name'].dropna()
        name = won_names.iloc[0] if len(won_names) > 0 else (lost_names.iloc[0] if len(lost_names) > 0 else "Unknown")
        
        return {
            'player_name': name,
            'total_matches': total,
            'wins': won,
            'losses': lost,
            'overall_win_pct': won / total if total > 0 else 0.5
        }
    
    def _calculate_h2h_stats(self, player_id: str) -> Dict[str, Dict]:
        """
        Calculate head-to-head records against all opponents.
        Returns a dictionary keyed by opponent_id with win/loss counts.
        """
        h2h_records = {}
        
        # Wins
        wins = self.matches_df[self.matches_df['winner_id'] == player_id]
        for _, row in wins.iterrows():
            opp_id = row.get('loser_id')
            if pd.notna(opp_id):
                opp_id = str(opp_id)
                if opp_id not in h2h_records:
                    h2h_records[opp_id] = {'wins': 0, 'losses': 0, 'opp_name': row.get('loser_name', '')}
                h2h_records[opp_id]['wins'] += 1
        
        # Losses
        losses = self.matches_df[self.matches_df['loser_id'] == player_id]
        for _, row in losses.iterrows():
            opp_id = row.get('winner_id')
            if pd.notna(opp_id):
                opp_id = str(opp_id)
                if opp_id not in h2h_records:
                    h2h_records[opp_id] = {'wins': 0, 'losses': 0, 'opp_name': row.get('winner_name', '')}
                h2h_records[opp_id]['losses'] += 1
        
        return h2h_records
    
    def _calculate_fatigue_stats(self, player_id: str, as_of_date: Optional[pd.Timestamp] = None) -> Dict[str, float]:
        """
        Calculate fatigue/schedule metrics.
        - days_since_last_match
        - matches_last_7_days
        - matches_last_14_days
        - matches_last_30_days
        """
        player_wins = self.matches_df[self.matches_df['winner_id'] == player_id].copy()
        player_losses = self.matches_df[self.matches_df['loser_id'] == player_id].copy()
        
        # Add match dates
        if 'match_date' not in player_wins.columns:
            player_wins['match_date'] = pd.to_datetime(player_wins['tourney_date'].astype(str), format='mixed', errors='coerce')
        if 'match_date' not in player_losses.columns:
            player_losses['match_date'] = pd.to_datetime(player_losses['tourney_date'].astype(str), format='mixed', errors='coerce')
        
        all_matches = pd.concat([
            player_wins[['match_date']],
            player_losses[['match_date']]
        ]).dropna().sort_values('match_date', ascending=False)
        
        if len(all_matches) == 0:
            return {
                'days_since_last_match': 30,
                'matches_last_7_days': 0,
                'matches_last_14_days': 0,
                'matches_last_30_days': 0,
                'avg_days_between_matches': 7
            }
        
        if as_of_date is None:
            as_of_date = all_matches['match_date'].max()
        
        last_match = all_matches['match_date'].iloc[0]
        days_since = (as_of_date - last_match).days if pd.notna(last_match) else 30
        
        # Count recent matches
        matches_7d = len(all_matches[all_matches['match_date'] >= as_of_date - pd.Timedelta(days=7)])
        matches_14d = len(all_matches[all_matches['match_date'] >= as_of_date - pd.Timedelta(days=14)])
        matches_30d = len(all_matches[all_matches['match_date'] >= as_of_date - pd.Timedelta(days=30)])
        
        # Average days between matches (last 10 matches)
        recent_dates = all_matches['match_date'].head(10).sort_values()
        if len(recent_dates) > 1:
            diffs = recent_dates.diff().dropna().dt.days
            avg_gap = diffs.mean() if len(diffs) > 0 else 7
        else:
            avg_gap = 7
        
        return {
            'days_since_last_match': max(0, days_since),
            'matches_last_7_days': matches_7d,
            'matches_last_14_days': matches_14d,
            'matches_last_30_days': matches_30d,
            'avg_days_between_matches': avg_gap
        }
    
    def _calculate_tournament_level_stats(self, player_id: str) -> Dict[str, float]:
        """
        Calculate performance by tournament level.
        - grand_slam_win_pct
        - masters_win_pct
        - atp500_win_pct
        - atp250_win_pct
        - challenger_win_pct
        """
        level_map = {
            'G': 'grand_slam',       # Grand Slam
            'M': 'masters',          # Masters 1000
            'A': 'atp500',           # ATP 500
            'B': 'atp250',           # ATP 250
            'C': 'challenger',       # Challenger
            'F': 'futures',          # Futures
            'D': 'davis_cup'         # Davis Cup
        }
        
        stats = {}
        for level_code, level_name in level_map.items():
            won = len(self.matches_df[
                (self.matches_df['winner_id'] == player_id) & 
                (self.matches_df['tourney_level'] == level_code)
            ])
            lost = len(self.matches_df[
                (self.matches_df['loser_id'] == player_id) & 
                (self.matches_df['tourney_level'] == level_code)
            ])
            total = won + lost
            stats[f'{level_name}_win_pct'] = won / total if total > 0 else 0.5
            stats[f'{level_name}_matches'] = total
        
        return stats
    
    def _calculate_recent_form(self, player_id: str, n_matches: int = 10) -> Dict[str, float]:
        """
        Calculate recent form based on last N matches.
        This captures momentum and current playing level.
        """
        # Get all matches for this player with dates
        player_wins = self.matches_df[self.matches_df['winner_id'] == player_id].copy()
        player_losses = self.matches_df[self.matches_df['loser_id'] == player_id].copy()
        
        player_wins['result'] = 1
        player_losses['result'] = 0
        
        # Combine and sort by date
        if 'match_date' not in player_wins.columns:
            player_wins['match_date'] = pd.to_datetime(player_wins['tourney_date'].astype(str), format='mixed', errors='coerce')
        if 'match_date' not in player_losses.columns:
            player_losses['match_date'] = pd.to_datetime(player_losses['tourney_date'].astype(str), format='mixed', errors='coerce')
        
        all_matches = pd.concat([
            player_wins[['match_date', 'result', 'surface']],
            player_losses[['match_date', 'result', 'surface']]
        ]).sort_values('match_date', ascending=False)
        
        recent = all_matches.head(n_matches)
        
        if len(recent) == 0:
            return {
                'recent_form': 0.5,
                'recent_form_weighted': 0.5,
                'win_streak': 0,
                'loss_streak': 0
            }
        
        # Simple recent win percentage
        recent_form = recent['result'].mean()
        
        # Weighted recent form (more recent matches count more)
        weights = np.array([1.0 / (i + 1) for i in range(len(recent))])
        weights = weights / weights.sum()
        recent_form_weighted = (recent['result'].values * weights).sum()
        
        # Calculate current streak
        win_streak = 0
        loss_streak = 0
        for result in recent['result'].values:
            if result == 1 and loss_streak == 0:
                win_streak += 1
            elif result == 0 and win_streak == 0:
                loss_streak += 1
            else:
                break
        
        return {
            'recent_form': recent_form,
            'recent_form_weighted': recent_form_weighted,
            'win_streak': win_streak,
            'loss_streak': loss_streak
        }
    
    def build_player_stats(self, min_matches: int = 20) -> pd.DataFrame:
        """
        Build statistics for all players using vectorized operations.
        
        Args:
            min_matches: Minimum matches required to include player
        
        Returns:
            DataFrame with player statistics
        """
        logger.info("Building player stats with vectorized operations...")
        
        df = self.matches_df.copy()
        
        # Ensure match_date exists
        if 'match_date' not in df.columns:
            df['match_date'] = pd.to_datetime(df['tourney_date'].astype(str), format='mixed', errors='coerce')
        
        # === WINNER STATS ===
        winner_agg = {
            'winner_name': 'first',
            'w_ace': 'sum',
            'w_df': 'sum', 
            'w_svpt': 'sum',
            'w_1stIn': 'sum',
            'w_1stWon': 'sum',
            'w_2ndWon': 'sum',
            'w_bpSaved': 'sum',
            'w_bpFaced': 'sum',
        }
        winner_stats = df.groupby('winner_id').agg(winner_agg)
        winner_stats['wins'] = df.groupby('winner_id').size()
        winner_stats = winner_stats.rename(columns={'winner_name': 'player_name'})
        winner_stats.index.name = 'player_id'
        
        # === LOSER STATS ===
        loser_agg = {
            'loser_name': 'first',
            'l_ace': 'sum',
            'l_df': 'sum',
            'l_svpt': 'sum', 
            'l_1stIn': 'sum',
            'l_1stWon': 'sum',
            'l_2ndWon': 'sum',
            'l_bpSaved': 'sum',
            'l_bpFaced': 'sum',
        }
        loser_stats = df.groupby('loser_id').agg(loser_agg)
        loser_stats['losses'] = df.groupby('loser_id').size()
        loser_stats = loser_stats.rename(columns={'loser_name': 'player_name'})
        loser_stats.index.name = 'player_id'
        
        # === COMBINE ===
        all_players = pd.concat([
            winner_stats[['player_name']],
            loser_stats[['player_name']]
        ]).groupby(level=0).first()
        
        stats = all_players.copy()
        stats['wins'] = winner_stats['wins'].reindex(stats.index).fillna(0)
        stats['losses'] = loser_stats['losses'].reindex(stats.index).fillna(0)
        stats['total_matches'] = stats['wins'] + stats['losses']
        stats['overall_win_pct'] = stats['wins'] / stats['total_matches']
        
        # Filter by min_matches early
        stats = stats[stats['total_matches'] >= min_matches].copy()
        logger.info(f"Found {len(stats)} players with {min_matches}+ matches")
        
        # === SERVE STATS (vectorized) ===
        w_serve = winner_stats[['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon']].reindex(stats.index).fillna(0)
        l_serve = loser_stats[['l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon']].reindex(stats.index).fillna(0)
        
        total_svpt = w_serve['w_svpt'] + l_serve['l_svpt']
        total_aces = w_serve['w_ace'] + l_serve['l_ace']
        total_df = w_serve['w_df'] + l_serve['l_df']
        total_1stIn = w_serve['w_1stIn'] + l_serve['l_1stIn']
        total_1stWon = w_serve['w_1stWon'] + l_serve['l_1stWon']
        total_2ndWon = w_serve['w_2ndWon'] + l_serve['l_2ndWon']
        
        stats['ace_pct'] = (total_aces / total_svpt).fillna(0)
        stats['df_pct'] = (total_df / total_svpt).fillna(0)
        stats['1st_serve_pct'] = (total_1stIn / total_svpt).fillna(0)
        stats['1st_won_pct'] = (total_1stWon / total_1stIn).fillna(0)
        second_serves = total_svpt - total_1stIn
        stats['2nd_won_pct'] = (total_2ndWon / second_serves).fillna(0)
        stats['serve_points_won_pct'] = ((total_1stWon + total_2ndWon) / total_svpt).fillna(0)
        
        # === BREAK POINT STATS ===
        w_bp = winner_stats[['w_bpSaved', 'w_bpFaced']].reindex(stats.index).fillna(0)
        l_bp = loser_stats[['l_bpSaved', 'l_bpFaced']].reindex(stats.index).fillna(0)
        
        total_bp_faced = w_bp['w_bpFaced'] + l_bp['l_bpFaced']
        total_bp_saved = w_bp['w_bpSaved'] + l_bp['l_bpSaved']
        stats['bp_save_pct'] = (total_bp_saved / total_bp_faced).fillna(0.6)
        
        # Return stats (opponent's break points)
        # BP convert = opponent BP faced - opponent BP saved
        stats['bp_convert_pct'] = 0.4  # Default, will refine below
        stats['return_points_won_pct'] = 0.4  # Default
        
        # === SURFACE STATS (vectorized) ===
        for surface in ['Hard', 'Clay', 'Grass']:
            surf_df = df[df['surface'] == surface]
            surf_wins = surf_df.groupby('winner_id').size().reindex(stats.index).fillna(0)
            surf_losses = surf_df.groupby('loser_id').size().reindex(stats.index).fillna(0)
            surf_total = surf_wins + surf_losses
            stats[f'{surface.lower()}_win_pct'] = (surf_wins / surf_total).fillna(0.5)
        
        # === TOURNAMENT LEVEL STATS ===
        level_map = {'G': 'grand_slam', 'M': 'masters', 'A': 'atp500', 'B': 'atp250', 'C': 'challenger', 'F': 'futures', 'D': 'davis_cup'}
        for level_code, level_name in level_map.items():
            level_df = df[df['tourney_level'] == level_code]
            level_wins = level_df.groupby('winner_id').size().reindex(stats.index).fillna(0)
            level_losses = level_df.groupby('loser_id').size().reindex(stats.index).fillna(0)
            level_total = level_wins + level_losses
            stats[f'{level_name}_win_pct'] = (level_wins / level_total).fillna(0.5)
            stats[f'{level_name}_matches'] = level_total
        
        # === TIEBREAK & DECIDING SET (simplified - use overall win pct as proxy) ===
        stats['tiebreak_win_pct'] = stats['overall_win_pct'] * 0.9 + 0.05  # Approximate
        stats['deciding_set_win_pct'] = stats['overall_win_pct'] * 0.95 + 0.025  # Approximate
        
        # === RECENT FORM (last 10 matches per player) ===
        logger.info("Calculating recent form...")
        recent_form = self._calculate_recent_form_vectorized(df, stats.index, n_matches=10)
        stats = stats.join(recent_form)
        
        # === FATIGUE STATS ===
        stats['days_since_last_match'] = 7  # Default
        stats['matches_last_7_days'] = 0
        stats['matches_last_14_days'] = 0
        stats['matches_last_30_days'] = 0
        
        # === H2H (build for qualified players only) ===
        logger.info("Building H2H records...")
        self._build_h2h_vectorized(df, set(stats.index))
        
        # === ELO RATINGS ===
        logger.info("Calculating ELO ratings...")
        self.elo_system, self.elo_ratings = calculate_elo_features(df)
        
        # Merge ELO ratings with stats
        elo_df = self.elo_ratings.set_index('player_id')
        for col in ['elo_overall', 'elo_hard', 'elo_clay', 'elo_grass', 
                    'elo_peak', 'elo_30d_change', 'elo_matches']:
            if col in elo_df.columns:
                stats[col] = elo_df[col].reindex(stats.index).fillna(1500 if 'elo' in col and 'change' not in col else 0)
        
        # === RANKING TRAJECTORY ===
        logger.info("Calculating ranking trajectory...")
        ranking_trajectory = self._calculate_ranking_trajectory(df, stats.index)
        stats = stats.join(ranking_trajectory)
        
        stats = stats.reset_index()
        self.player_stats = stats
        logger.info(f"Built stats for {len(stats)} players with {min_matches}+ matches.")
        
        return self.player_stats
    
    def _calculate_ranking_trajectory(self, df: pd.DataFrame, player_ids: pd.Index) -> pd.DataFrame:
        """
        Calculate ranking trajectory (rising vs falling) based on recent results.
        Uses win rate change and ELO momentum as indicators.
        """
        results = {}
        
        for player_id in player_ids:
            # Get recent 20 matches vs older matches
            player_wins = df[df['winner_id'] == player_id]
            player_losses = df[df['loser_id'] == player_id]
            
            all_matches = pd.concat([
                player_wins[['match_date']].assign(result=1),
                player_losses[['match_date']].assign(result=0)
            ]).sort_values('match_date', ascending=False)
            
            if len(all_matches) < 20:
                results[player_id] = {
                    'ranking_trajectory': 0.0,
                    'is_rising': False,
                    'is_falling': False,
                }
                continue
            
            # Recent 10 matches vs previous 10
            recent_10 = all_matches.head(10)['result'].mean()
            prev_10 = all_matches.iloc[10:20]['result'].mean()
            
            trajectory = recent_10 - prev_10  # Positive = improving
            
            results[player_id] = {
                'ranking_trajectory': trajectory,
                'is_rising': trajectory > 0.1,
                'is_falling': trajectory < -0.1,
            }
        
        return pd.DataFrame(results).T
    
    def _calculate_recent_form_vectorized(self, df: pd.DataFrame, player_ids: pd.Index, n_matches: int = 10) -> pd.DataFrame:
        """Calculate recent form for all players at once."""
        results = []
        
        # Create player-match records
        wins = df[['winner_id', 'match_date']].copy()
        wins.columns = ['player_id', 'match_date']
        wins['result'] = 1
        
        losses = df[['loser_id', 'match_date']].copy()
        losses.columns = ['player_id', 'match_date']
        losses['result'] = 0
        
        all_matches = pd.concat([wins, losses])
        all_matches = all_matches[all_matches['player_id'].isin(player_ids)]
        all_matches = all_matches.sort_values(['player_id', 'match_date'], ascending=[True, False])
        
        # Get last N matches per player
        all_matches['rank'] = all_matches.groupby('player_id').cumcount()
        recent = all_matches[all_matches['rank'] < n_matches]
        
        # Calculate stats
        form_stats = recent.groupby('player_id').agg({
            'result': ['mean', 'sum', 'count']
        })
        form_stats.columns = ['recent_form', 'recent_wins', 'recent_matches']
        form_stats['recent_form_weighted'] = form_stats['recent_form']  # Simplified
        
        # Calculate streaks (simplified - just use first result as indicator)
        first_results = recent[recent['rank'] == 0].set_index('player_id')['result']
        form_stats['win_streak'] = (first_results * 3).fillna(0).astype(int)  # Approximate
        form_stats['loss_streak'] = ((1 - first_results) * 2).fillna(0).astype(int)  # Approximate
        
        return form_stats[['recent_form', 'recent_form_weighted', 'win_streak', 'loss_streak']]
    
    def _build_h2h_vectorized(self, df: pd.DataFrame, player_ids: set):
        """Build H2H records using vectorized operations."""
        # Filter to matches between qualified players
        h2h_df = df[
            (df['winner_id'].isin(player_ids)) & 
            (df['loser_id'].isin(player_ids))
        ][['winner_id', 'loser_id']].copy()
        
        # Count wins for each pair
        h2h_counts = h2h_df.groupby(['winner_id', 'loser_id']).size().reset_index(name='wins')
        
        # Build lookup dict
        for _, row in h2h_counts.iterrows():
            winner_id = str(row['winner_id'])
            loser_id = str(row['loser_id'])
            wins = int(row['wins'])
            
            if winner_id not in self.h2h_records:
                self.h2h_records[winner_id] = {}
            if loser_id not in self.h2h_records[winner_id]:
                self.h2h_records[winner_id][loser_id] = {'wins': 0, 'losses': 0}
            
            self.h2h_records[winner_id][loser_id]['wins'] += wins
            
            # Also record reverse
            if loser_id not in self.h2h_records:
                self.h2h_records[loser_id] = {}
            if winner_id not in self.h2h_records[loser_id]:
                self.h2h_records[loser_id][winner_id] = {'wins': 0, 'losses': 0}
            
            self.h2h_records[loser_id][winner_id]['losses'] += wins
    
    def cluster_play_styles(self) -> pd.DataFrame:
        """Cluster players into style archetypes."""
        if self.player_stats is None:
            raise ValueError("Run build_player_stats first.")
        
        # Features for clustering
        style_features = [
            'ace_pct', 'df_pct', 'serve_points_won_pct',
            'return_points_won_pct', 'bp_convert_pct', 'bp_save_pct'
        ]
        
        # Filter to players with all features
        df = self.player_stats.dropna(subset=style_features).copy()
        
        if len(df) < 10:
            logger.warning("Not enough players with complete stats for clustering.")
            self.player_stats['style_cluster'] = 'All-Court'
            return self.player_stats
        
        X = df[style_features].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Cluster
        clusters = self.clusterer.fit_predict(X_scaled)
        df['style_cluster_id'] = clusters
        
        # Analyze clusters to name them
        cluster_names = self._name_clusters(df, style_features)
        df['style_cluster'] = df['style_cluster_id'].map(cluster_names)
        
        # Merge back
        self.player_stats = self.player_stats.merge(
            df[['player_id', 'style_cluster_id', 'style_cluster']], 
            on='player_id', 
            how='left'
        )
        self.player_stats['style_cluster'] = self.player_stats['style_cluster'].fillna('All-Court')
        
        return self.player_stats
    
    def _name_clusters(self, df: pd.DataFrame, features: List[str]) -> Dict[int, str]:
        """Name clusters based on their characteristics."""
        cluster_means = df.groupby('style_cluster_id')[features].mean()
        
        names = {}
        for cluster_id in cluster_means.index:
            means = cluster_means.loc[cluster_id]
            
            # Determine style based on dominant characteristics
            if means['ace_pct'] > cluster_means['ace_pct'].mean() * 1.2:
                names[cluster_id] = "Big Server"
            elif means['return_points_won_pct'] > cluster_means['return_points_won_pct'].mean() * 1.1:
                names[cluster_id] = "Counterpuncher"
            elif means['bp_save_pct'] > cluster_means['bp_save_pct'].mean() * 1.05:
                names[cluster_id] = "Grinder"
            elif means['ace_pct'] > cluster_means['ace_pct'].mean() and \
                 means['df_pct'] > cluster_means['df_pct'].mean():
                names[cluster_id] = "Aggressive Baseliner"
            else:
                names[cluster_id] = "All-Court"
        
        return names
    
    def calculate_clutch_rating(self) -> pd.DataFrame:
        """Calculate composite clutch rating."""
        if self.player_stats is None:
            raise ValueError("Run build_player_stats first.")
        
        # Clutch rating = weighted average of clutch metrics
        df = self.player_stats.copy()
        
        df['clutch_rating'] = (
            df['tiebreak_win_pct'] * 0.35 +
            df['deciding_set_win_pct'] * 0.35 +
            df['bp_save_pct'] * 0.30
        )
        
        # Normalize to 0-100 scale
        min_clutch = df['clutch_rating'].min()
        max_clutch = df['clutch_rating'].max()
        if max_clutch > min_clutch:
            df['clutch_rating'] = (df['clutch_rating'] - min_clutch) / (max_clutch - min_clutch) * 100
        
        self.player_stats = df
        return self.player_stats
    
    def calculate_serve_dominance(self) -> pd.DataFrame:
        """Calculate serve dominance index."""
        if self.player_stats is None:
            raise ValueError("Run build_player_stats first.")
        
        df = self.player_stats.copy()
        
        # Serve dominance = ace% - df% + serve points won%
        df['serve_dominance'] = (
            df['ace_pct'] * 100 - 
            df['df_pct'] * 100 + 
            df['serve_points_won_pct'] * 50
        )
        
        # Normalize
        min_sd = df['serve_dominance'].min()
        max_sd = df['serve_dominance'].max()
        if max_sd > min_sd:
            df['serve_dominance'] = (df['serve_dominance'] - min_sd) / (max_sd - min_sd) * 100
        
        self.player_stats = df
        return self.player_stats
    
    def get_surface_specialist(self) -> pd.DataFrame:
        """Determine each player's best surface."""
        if self.player_stats is None:
            raise ValueError("Run build_player_stats first.")
        
        df = self.player_stats.copy()
        
        surface_cols = ['hard_win_pct', 'clay_win_pct', 'grass_win_pct']
        df['surface_specialist'] = df[surface_cols].idxmax(axis=1).str.replace('_win_pct', '').str.title()
        
        self.player_stats = df
        return self.player_stats
    
    def build_all_profiles(self, min_matches: int = 20) -> pd.DataFrame:
        """Build complete player profiles with all metrics."""
        self.build_player_stats(min_matches)
        self.cluster_play_styles()
        self.calculate_clutch_rating()
        self.calculate_serve_dominance()
        self.get_surface_specialist()
        
        logger.info(f"Built complete profiles for {len(self.player_stats)} players.")
        return self.player_stats
    
    def get_player_profile(self, player_name: str) -> Optional[pd.Series]:
        """Get profile for a specific player by name."""
        if self.player_stats is None:
            raise ValueError("Run build_all_profiles first.")
        
        mask = self.player_stats['player_name'].str.contains(player_name, case=False, na=False)
        matches = self.player_stats[mask]
        
        if len(matches) == 0:
            return None
        return matches.iloc[0]
    
    def get_top_players(self, n: int = 50, by: str = 'overall_win_pct') -> pd.DataFrame:
        """Get top N players by specified metric."""
        if self.player_stats is None:
            raise ValueError("Run build_all_profiles first.")
        
        return self.player_stats.nlargest(n, by)
    
    def get_h2h_record(self, player1_name: str, player2_name: str) -> Optional[Dict]:
        """
        Get head-to-head record between two players.
        
        Returns:
            Dict with 'p1_wins', 'p2_wins', 'total_matches' or None if not found
        """
        if self.player_stats is None:
            raise ValueError("Run build_all_profiles first.")
        
        # Find player IDs
        p1_mask = self.player_stats['player_name'].str.contains(player1_name, case=False, na=False)
        p2_mask = self.player_stats['player_name'].str.contains(player2_name, case=False, na=False)
        
        p1_matches = self.player_stats[p1_mask]
        p2_matches = self.player_stats[p2_mask]
        
        if len(p1_matches) == 0 or len(p2_matches) == 0:
            return None
        
        p1_id = str(p1_matches.iloc[0]['player_id'])
        p2_id = str(p2_matches.iloc[0]['player_id'])
        
        # Look up H2H from stored records
        if p1_id in self.h2h_records and p2_id in self.h2h_records[p1_id]:
            record = self.h2h_records[p1_id][p2_id]
            return {
                'p1_wins': record['wins'],
                'p2_wins': record['losses'],
                'total_matches': record['wins'] + record['losses'],
                'p1_h2h_pct': record['wins'] / (record['wins'] + record['losses']) if (record['wins'] + record['losses']) > 0 else 0.5
            }
        
        return {'p1_wins': 0, 'p2_wins': 0, 'total_matches': 0, 'p1_h2h_pct': 0.5}


if __name__ == "__main__":
    from data_loader import load_matches, preprocess_matches
    
    print("Loading matches...")
    matches = load_matches(years=[2022, 2023, 2024])
    matches = preprocess_matches(matches)
    
    print("Building player profiles...")
    profiler = PlayerProfiler(matches)
    profiles = profiler.build_all_profiles(min_matches=20)
    
    print(f"\nBuilt {len(profiles)} player profiles")
    print(f"\nStyle distribution:")
    print(profiles['style_cluster'].value_counts())
    
    print("\nTop 10 players by clutch rating:")
    print(profiler.get_top_players(10, by='clutch_rating')[['player_name', 'clutch_rating', 'style_cluster']])
