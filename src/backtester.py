"""
Backtesting Module
Tests predictions against historical outcomes to validate and improve the model.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from data_loader import load_matches, preprocess_matches
from player_profiler import PlayerProfiler
from matchup_model import MatchupModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of a single prediction backtest."""
    match_date: str
    tournament: str
    surface: str
    player1: str
    player2: str
    predicted_winner: str
    actual_winner: str
    correct: bool
    confidence: float
    upset_predicted: bool
    was_upset: bool  # Based on ranking
    upset_correct: bool  # Predicted upset correctly
    style_matchup: str
    key_factors: List[str]
    p1_win_prob: float
    p2_win_prob: float
    rank_diff: float  # Ranking difference


def run_backtest(
    train_end_date: str,
    test_start_date: str,
    test_end_date: str,
    min_matches: int = 15
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run backtest: train on data before train_end_date, test on matches between test dates.
    
    Args:
        train_end_date: Train on matches before this date (YYYY-MM-DD)
        test_start_date: Test on matches from this date
        test_end_date: Test on matches until this date
        min_matches: Minimum matches for player profile
    
    Returns:
        Tuple of (results DataFrame, analysis dict)
    """
    logger.info(f"Loading all historical data...")
    all_matches = load_matches(years=list(range(2018, 2025)), include_lower_level=True)
    all_matches = preprocess_matches(all_matches)
    
    # Parse dates
    train_end = pd.to_datetime(train_end_date)
    test_start = pd.to_datetime(test_start_date)
    test_end = pd.to_datetime(test_end_date)
    
    # Convert tourney_date to datetime (handle mixed formats)
    all_matches['match_date'] = pd.to_datetime(all_matches['tourney_date'].astype(str), format='mixed', errors='coerce')
    
    # Split data
    train_data = all_matches[all_matches['match_date'] < train_end].copy()
    test_data = all_matches[
        (all_matches['match_date'] >= test_start) & 
        (all_matches['match_date'] <= test_end)
    ].copy()
    
    logger.info(f"Training data: {len(train_data)} matches (before {train_end_date})")
    logger.info(f"Test data: {len(test_data)} matches ({test_start_date} to {test_end_date})")
    
    if len(test_data) == 0:
        raise ValueError("No test matches found in date range")
    
    # Build profiles from training data only
    logger.info("Building player profiles from training data...")
    profiler = PlayerProfiler(train_data)
    profiles = profiler.build_all_profiles(min_matches=min_matches)
    logger.info(f"Built {len(profiles)} player profiles")
    
    # Train model on training data only (pass profiler for H2H lookups)
    logger.info("Training model...")
    model = MatchupModel(profiles, profiler=profiler)
    train_result = model.train(train_data, model_type='random_forest')
    logger.info(f"Model trained: accuracy={train_result['accuracy']:.1%}")
    
    # Run predictions on test data
    logger.info("Running predictions on test matches...")
    results = []
    
    for idx, match in test_data.iterrows():
        winner = match['winner_name']
        loser = match['loser_name']
        surface = match.get('surface', 'Hard')
        
        # Determine who was favorite based on ranking
        winner_rank = match.get('winner_rank', 999)
        loser_rank = match.get('loser_rank', 999)
        
        if pd.isna(winner_rank):
            winner_rank = 999
        if pd.isna(loser_rank):
            loser_rank = 999
            
        # Lower rank = better player = favorite
        if winner_rank < loser_rank:
            favorite = winner
            underdog = loser
            was_upset = False
        else:
            favorite = loser
            underdog = winner
            was_upset = True  # Lower-ranked player won
        
        # Get prediction (randomly assign to p1/p2 to avoid bias)
        if np.random.random() > 0.5:
            p1, p2 = winner, loser
        else:
            p1, p2 = loser, winner
        
        tourney_level = match.get('tourney_level', 'A')
        pred = model.predict_match(p1, p2, surface, favorite, tourney_level=str(tourney_level) if pd.notna(tourney_level) else 'A')
        
        if pred is None:
            continue
        
        # Determine if prediction was correct
        predicted_winner = pred.predicted_winner
        correct = predicted_winner == winner
        
        # Determine if upset was predicted
        upset_predicted = pred.upset_probability >= 0.35
        upset_correct = upset_predicted and was_upset and (predicted_winner == underdog)
        
        results.append(BacktestResult(
            match_date=match['match_date'].strftime('%Y-%m-%d'),
            tournament=match.get('tourney_name', 'Unknown'),
            surface=surface,
            player1=p1,
            player2=p2,
            predicted_winner=predicted_winner,
            actual_winner=winner,
            correct=correct,
            confidence=pred.confidence,
            upset_predicted=upset_predicted,
            was_upset=was_upset,
            upset_correct=upset_correct,
            style_matchup=pred.style_matchup,
            key_factors=pred.key_factors,
            p1_win_prob=pred.player1_win_prob,
            p2_win_prob=pred.player2_win_prob,
            rank_diff=abs(winner_rank - loser_rank)
        ))
    
    # Convert to DataFrame
    results_df = pd.DataFrame([r.__dict__ for r in results])
    
    # Analyze results
    analysis = analyze_backtest_results(results_df)
    
    return results_df, analysis


def analyze_backtest_results(results_df: pd.DataFrame) -> Dict:
    """Analyze backtest results to identify patterns in correct/incorrect predictions."""
    
    if len(results_df) == 0:
        return {"error": "No results to analyze"}
    
    analysis = {}
    
    # Overall accuracy
    analysis['total_predictions'] = len(results_df)
    analysis['correct_predictions'] = results_df['correct'].sum()
    analysis['accuracy'] = results_df['correct'].mean()
    
    # Accuracy by confidence level
    high_conf = results_df[results_df['confidence'] >= 0.65]
    med_conf = results_df[(results_df['confidence'] >= 0.55) & (results_df['confidence'] < 0.65)]
    low_conf = results_df[results_df['confidence'] < 0.55]
    
    analysis['accuracy_by_confidence'] = {
        'high (>=65%)': high_conf['correct'].mean() if len(high_conf) > 0 else 0,
        'medium (55-65%)': med_conf['correct'].mean() if len(med_conf) > 0 else 0,
        'low (<55%)': low_conf['correct'].mean() if len(low_conf) > 0 else 0,
        'high_count': len(high_conf),
        'medium_count': len(med_conf),
        'low_count': len(low_conf)
    }
    
    # Accuracy by surface
    analysis['accuracy_by_surface'] = {}
    for surface in results_df['surface'].unique():
        surface_df = results_df[results_df['surface'] == surface]
        analysis['accuracy_by_surface'][surface] = {
            'accuracy': surface_df['correct'].mean(),
            'count': len(surface_df)
        }
    
    # Upset prediction performance
    actual_upsets = results_df[results_df['was_upset']]
    predicted_upsets = results_df[results_df['upset_predicted']]
    
    analysis['upset_analysis'] = {
        'total_upsets': len(actual_upsets),
        'upsets_correctly_predicted': results_df['upset_correct'].sum(),
        'upset_recall': results_df['upset_correct'].sum() / len(actual_upsets) if len(actual_upsets) > 0 else 0,
        'upset_predictions_made': len(predicted_upsets),
        'upset_precision': predicted_upsets['correct'].mean() if len(predicted_upsets) > 0 else 0
    }
    
    # Analyze incorrect predictions - what factors correlate with errors?
    incorrect = results_df[~results_df['correct']]
    correct = results_df[results_df['correct']]
    
    analysis['error_patterns'] = {
        'avg_confidence_correct': correct['confidence'].mean() if len(correct) > 0 else 0,
        'avg_confidence_incorrect': incorrect['confidence'].mean() if len(incorrect) > 0 else 0,
        'avg_rank_diff_correct': correct['rank_diff'].mean() if len(correct) > 0 else 0,
        'avg_rank_diff_incorrect': incorrect['rank_diff'].mean() if len(incorrect) > 0 else 0,
    }
    
    # Style matchup analysis
    analysis['style_matchup_accuracy'] = {}
    for matchup in results_df['style_matchup'].unique():
        matchup_df = results_df[results_df['style_matchup'] == matchup]
        if len(matchup_df) >= 10:  # Only include if enough samples
            analysis['style_matchup_accuracy'][matchup] = {
                'accuracy': matchup_df['correct'].mean(),
                'count': len(matchup_df)
            }
    
    # Key factors analysis - which factors correlate with correct predictions?
    factor_counts = {'correct': {}, 'incorrect': {}}
    for _, row in results_df.iterrows():
        category = 'correct' if row['correct'] else 'incorrect'
        for factor in row['key_factors']:
            factor_key = factor[:50]  # Truncate for grouping
            factor_counts[category][factor_key] = factor_counts[category].get(factor_key, 0) + 1
    
    analysis['factor_analysis'] = factor_counts
    
    return analysis


def print_analysis(analysis: Dict):
    """Pretty print the analysis results."""
    print("\n" + "="*60)
    print("BACKTEST ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n[OVERALL PERFORMANCE]")
    print(f"   Total predictions: {analysis['total_predictions']}")
    print(f"   Correct: {analysis['correct_predictions']}")
    print(f"   Accuracy: {analysis['accuracy']:.1%}")
    
    print(f"\n[ACCURACY BY CONFIDENCE LEVEL]")
    conf = analysis['accuracy_by_confidence']
    print(f"   High (>=65%):   {conf['high (>=65%)']:.1%} ({conf['high_count']} matches)")
    print(f"   Medium (55-65%): {conf['medium (55-65%)']:.1%} ({conf['medium_count']} matches)")
    print(f"   Low (<55%):     {conf['low (<55%)']:.1%} ({conf['low_count']} matches)")
    
    print(f"\n[ACCURACY BY SURFACE]")
    for surface, data in analysis['accuracy_by_surface'].items():
        print(f"   {surface}: {data['accuracy']:.1%} ({data['count']} matches)")
    
    print(f"\n[UPSET PREDICTION PERFORMANCE]")
    upset = analysis['upset_analysis']
    print(f"   Total actual upsets: {upset['total_upsets']}")
    print(f"   Upsets we predicted correctly: {upset['upsets_correctly_predicted']}")
    print(f"   Upset recall (% of upsets caught): {upset['upset_recall']:.1%}")
    print(f"   Upset predictions made: {upset['upset_predictions_made']}")
    print(f"   Upset precision (when we predict upset): {upset['upset_precision']:.1%}")
    
    print(f"\n[ERROR PATTERN ANALYSIS]")
    err = analysis['error_patterns']
    print(f"   Avg confidence (correct): {err['avg_confidence_correct']:.1%}")
    print(f"   Avg confidence (incorrect): {err['avg_confidence_incorrect']:.1%}")
    print(f"   Avg rank diff (correct): {err['avg_rank_diff_correct']:.0f}")
    print(f"   Avg rank diff (incorrect): {err['avg_rank_diff_incorrect']:.0f}")
    
    print(f"\n[STYLE MATCHUP ACCURACY] (min 10 matches)")
    sorted_styles = sorted(
        analysis['style_matchup_accuracy'].items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    for matchup, data in sorted_styles[:10]:
        print(f"   {matchup}: {data['accuracy']:.1%} ({data['count']})")
    
    print("\n" + "="*60)


def get_improvement_recommendations(analysis: Dict) -> List[str]:
    """Generate recommendations for model improvement based on analysis."""
    recommendations = []
    
    # Confidence calibration
    conf = analysis['accuracy_by_confidence']
    if conf['high (>=65%)'] < 0.70:
        recommendations.append(
            "[!] High-confidence predictions underperforming. Consider recalibrating confidence thresholds."
        )
    
    if conf['low (<55%)'] > 0.50:
        recommendations.append(
            "[+] Low-confidence predictions doing better than expected. Model may be too conservative."
        )
    
    # Upset detection
    upset = analysis['upset_analysis']
    if upset['upset_recall'] < 0.20:
        recommendations.append(
            "[!] Missing most upsets. Consider tuning H2H, fatigue, and momentum feature weights."
        )
    
    if upset['upset_precision'] < 0.40:
        recommendations.append(
            "[!] Too many false upset alerts. Consider raising upset threshold or adding filters."
        )
    
    # Error patterns
    err = analysis['error_patterns']
    if err['avg_rank_diff_incorrect'] < err['avg_rank_diff_correct']:
        recommendations.append(
            "[*] Errors more common in close-ranking matches. Add features for form/momentum."
        )
    
    # Surface-specific issues
    for surface, data in analysis['accuracy_by_surface'].items():
        if data['accuracy'] < 0.60 and data['count'] > 50:
            recommendations.append(
                f"[*] {surface} court accuracy low ({data['accuracy']:.1%}). Review surface-specific features."
            )
    
    return recommendations


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    # Run backtest: Train on data before Oct 2024, test on Oct-Dec 2024
    print("Running backtest...")
    print("Training on: 2018-01-01 to 2024-09-30")
    print("Testing on: 2024-10-01 to 2024-12-15")
    print()
    
    results_df, analysis = run_backtest(
        train_end_date="2024-10-01",
        test_start_date="2024-10-01", 
        test_end_date="2024-12-15",
        min_matches=10
    )
    
    # Print analysis
    print_analysis(analysis)
    
    # Get recommendations
    print("\n[IMPROVEMENT RECOMMENDATIONS]")
    print("-" * 40)
    recommendations = get_improvement_recommendations(analysis)
    for rec in recommendations:
        print(f"   {rec}")
    
    # Show some example predictions
    print("\n[SAMPLE PREDICTIONS] (last 20)")
    print("-" * 80)
    sample = results_df.tail(20)[['match_date', 'tournament', 'predicted_winner', 'actual_winner', 'correct', 'confidence']]
    sample.columns = ['Date', 'Tournament', 'Predicted', 'Actual', 'Correct', 'Conf']
    print(sample.to_string(index=False))
    
    # Save results
    results_df.to_csv("backtest_results.csv", index=False)
    print(f"\n[OK] Results saved to backtest_results.csv")
