"""
Upset Prediction Analyzer
Analyzes prediction errors to identify patterns and improvement opportunities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ErrorPattern:
    """A pattern found in prediction errors."""
    name: str
    description: str
    frequency: float  # 0-1
    consistent_across_periods: bool
    periods_found: List[str]
    feature_stats: Dict
    example_count: int


@dataclass
class ImprovementOption:
    """A tested improvement option."""
    name: str
    description: str
    change_type: str  # 'threshold', 'feature_weight', 'feature_add'
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    periods_tested: List[str]
    consistent_improvement: bool
    trade_offs: str


def analyze_error_patterns(results_df: pd.DataFrame, period_name: str) -> Dict:
    """
    Analyze prediction errors to identify patterns.
    
    Args:
        results_df: Backtest results with predictions and actuals
        period_name: Name of the time period being analyzed
    
    Returns:
        Dictionary with error analysis
    """
    if len(results_df) == 0:
        return {"error": "No results to analyze"}
    
    # Categorize predictions
    true_positives = results_df[results_df['upset_correct'] == True]  # Predicted upset correctly
    false_positives = results_df[(results_df['upset_predicted'] == True) & (results_df['was_upset'] == False)]  # Predicted upset, favorite won
    false_negatives = results_df[(results_df['upset_predicted'] == False) & (results_df['was_upset'] == True)]  # Missed upset
    true_negatives = results_df[(results_df['upset_predicted'] == False) & (results_df['was_upset'] == False)]  # Correctly predicted favorite
    
    analysis = {
        'period': period_name,
        'total_matches': len(results_df),
        'confusion_matrix': {
            'true_positives': len(true_positives),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'true_negatives': len(true_negatives)
        }
    }
    
    # Analyze False Positives (predicted upset wrongly)
    if len(false_positives) > 0:
        fp_patterns = []
        
        # Pattern: Confidence distribution
        avg_confidence = false_positives['confidence'].mean()
        fp_patterns.append({
            'name': 'High Confidence False Alarms',
            'description': f'Average confidence in wrong upset predictions: {avg_confidence:.1%}',
            'frequency': len(false_positives[false_positives['confidence'] >= 0.60]) / len(false_positives),
            'count': len(false_positives[false_positives['confidence'] >= 0.60])
        })
        
        # Pattern: Surface distribution
        surface_dist = false_positives['surface'].value_counts()
        for surf, count in surface_dist.items():
            freq = count / len(false_positives)
            if freq > 0.3:  # More than 30% of FPs on this surface
                fp_patterns.append({
                    'name': f'False Alarms on {surf}',
                    'description': f'{count} false positive upset predictions on {surf} courts',
                    'frequency': freq,
                    'count': count
                })
        
        analysis['false_positive_patterns'] = fp_patterns
    
    # Analyze False Negatives (missed upsets)
    if len(false_negatives) > 0:
        fn_patterns = []
        
        # Pattern: Confidence distribution of misses
        avg_confidence = false_negatives['confidence'].mean()
        fn_patterns.append({
            'name': 'High Confidence Missed Upsets',
            'description': f'We were {avg_confidence:.1%} confident in favorite (but upset happened)',
            'frequency': len(false_negatives[false_negatives['confidence'] >= 0.60]) / len(false_negatives),
            'count': len(false_negatives[false_negatives['confidence'] >= 0.60])
        })
        
        # Pattern: Surface distribution
        surface_dist = false_negatives['surface'].value_counts()
        for surf, count in surface_dist.items():
            freq = count / len(false_negatives)
            if freq > 0.3:  # More than 30% of FNs on this surface
                fn_patterns.append({
                    'name': f'Missed Upsets on {surf}',
                    'description': f'{count} upsets we missed on {surf} courts',
                    'frequency': freq,
                    'count': count
                })
        
        analysis['false_negative_patterns'] = fn_patterns
    
    return analysis


def compare_multi_period_patterns(period_analyses: List[Dict]) -> Dict:
    """
    Compare patterns across multiple time periods to find consistent ones.
    
    Args:
        period_analyses: List of analysis results from different periods
    
    Returns:
        Dictionary with consistent patterns
    """
    if len(period_analyses) < 2:
        return {"error": "Need at least 2 periods to compare"}
    
    consistent_patterns = {
        'periods_analyzed': [a['period'] for a in period_analyses],
        'consistent_fp_patterns': [],
        'consistent_fn_patterns': []
    }
    
    # Check FP patterns consistency
    fp_pattern_names = {}
    for analysis in period_analyses:
        if 'false_positive_patterns' in analysis:
            for pattern in analysis['false_positive_patterns']:
                name = pattern['name']
                if name not in fp_pattern_names:
                    fp_pattern_names[name] = []
                fp_pattern_names[name].append({
                    'period': analysis['period'],
                    'frequency': pattern['frequency'],
                    'count': pattern.get('count', 0)
                })
    
    # Identify patterns present in ALL periods
    for name, occurrences in fp_pattern_names.items():
        if len(occurrences) == len(period_analyses):
            avg_freq = np.mean([o['frequency'] for o in occurrences])
            consistent_patterns['consistent_fp_patterns'].append({
                'name': name,
                'avg_frequency': avg_freq,
                'periods': [o['period'] for o in occurrences],
                'consistent': True
            })
    
    # Check FN patterns consistency
    fn_pattern_names = {}
    for analysis in period_analyses:
        if 'false_negative_patterns' in analysis:
            for pattern in analysis['false_negative_patterns']:
                name = pattern['name']
                if name not in fn_pattern_names:
                    fn_pattern_names[name] = []
                fn_pattern_names[name].append({
                    'period': analysis['period'],
                    'frequency': pattern['frequency'],
                    'count': pattern.get('count', 0)
                })
    
    # Identify patterns present in ALL periods
    for name, occurrences in fn_pattern_names.items():
        if len(occurrences) == len(period_analyses):
            avg_freq = np.mean([o['frequency'] for o in occurrences])
            consistent_patterns['consistent_fn_patterns'].append({
                'name': name,
                'avg_frequency': avg_freq,
                'periods': [o['period'] for o in occurrences],
                'consistent': True
            })
    
    return consistent_patterns


def generate_improvement_options(analyses: List[Dict]) -> List[ImprovementOption]:
    """
    Generate testable improvement options based on analysis.
    
    Args:
        analyses: List of analysis results from different periods
    
    Returns:
        List of improvement options to test
    """
    options = []
    
    # Option 1: Test different upset thresholds
    # Current threshold is 40% - test 42.5%, 45%, 47.5%, 50%
    for threshold in [0.425, 0.45, 0.475, 0.50]:
        options.append(ImprovementOption(
            name=f"Increase Upset Threshold to {threshold*100:.1f}%",
            description=f"Flag matches as 'upset likely' only when upset_probability >= {threshold*100:.1f}% (currently 40%)",
            change_type="threshold",
            before_metrics={},
            after_metrics={},
            periods_tested=[a['period'] for a in analyses],
            consistent_improvement=False,  # Will be updated after testing
            trade_offs="Higher threshold = fewer false alarms but may miss some upsets"
        ))
    
    return options


def simulate_threshold_change(results_df: pd.DataFrame, new_threshold: float) -> Dict:
    """
    Simulate what metrics would look like with a different upset threshold.
    
    Args:
        results_df: Backtest results
        new_threshold: New threshold to test (0-1)
    
    Returns:
        Dictionary with simulated metrics
    """
    # Recalculate predictions with new threshold
    simulated = results_df.copy()
    
    # Recalculate upset_predicted based on new threshold
    # Note: We don't have upset_probability in results_df, so we'll use confidence as proxy
    # For upsets: upset_prob = probability that underdog wins
    # When prediction was correct and was_upset=True, confidence indicates upset strength
    
    # Simplified simulation: Use confidence to approximate upset probability
    # If was_upset and correct, confidence is high → likely high upset_prob
    # If not was_upset and correct, confidence is high → likely low upset_prob
    
    # For accurate simulation, we'd need to re-run predictions with new threshold
    # For now, provide estimated metrics
    
    return {
        'threshold': new_threshold,
        'note': 'Accurate testing requires re-running backtest with new threshold',
        'estimated_impact': 'Higher threshold will reduce false positives but may increase false negatives'
    }


if __name__ == "__main__":
    # Test the analyzer with sample data
    print("Upset Analyzer Module - Ready for integration")
