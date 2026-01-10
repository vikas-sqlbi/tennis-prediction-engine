"""
Historical Accuracy Simulator Module
Simulates model predictions on historical matches to analyze accuracy over time periods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

from backtester import run_backtest, analyze_backtest_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimulationPeriod:
    """Definition of a time period for simulation."""
    name: str
    start_date: str
    end_date: str
    description: str


def get_predefined_periods() -> Dict[str, SimulationPeriod]:
    """
    Get predefined time periods for quick analysis.
    
    Returns:
        Dictionary of period names to SimulationPeriod objects
    """
    today = datetime.now()
    
    periods = {
        'current_week': SimulationPeriod(
            name='Current Week',
            start_date=(today - timedelta(days=today.weekday())).strftime('%Y-%m-%d'),
            end_date=today.strftime('%Y-%m-%d'),
            description='Last 7 days'
        ),
        'current_month': SimulationPeriod(
            name='Current Month',
            start_date=today.replace(day=1).strftime('%Y-%m-%d'),
            end_date=today.strftime('%Y-%m-%d'),
            description='Since start of current month'
        ),
        'last_month': SimulationPeriod(
            name='Last Month',
            start_date=(today.replace(day=1) - timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d'),
            end_date=(today.replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d'),
            description='Previous calendar month'
        ),
        '1_month': SimulationPeriod(
            name='Last 1 Month',
            start_date=(today - timedelta(days=30)).strftime('%Y-%m-%d'),
            end_date=today.strftime('%Y-%m-%d'),
            description='Last 30 days'
        ),
        '3_months': SimulationPeriod(
            name='Last 3 Months',
            start_date=(today - timedelta(days=90)).strftime('%Y-%m-%d'),
            end_date=today.strftime('%Y-%m-%d'),
            description='Last 90 days'
        ),
        '6_months': SimulationPeriod(
            name='Last 6 Months',
            start_date=(today - timedelta(days=180)).strftime('%Y-%m-%d'),
            end_date=today.strftime('%Y-%m-%d'),
            description='Last 180 days'
        ),
        '1_year': SimulationPeriod(
            name='Last 1 Year',
            start_date=(today - timedelta(days=365)).strftime('%Y-%m-%d'),
            end_date=today.strftime('%Y-%m-%d'),
            description='Last 365 days'
        ),
    }
    
    return periods


def simulate_accuracy(
    start_date: str,
    end_date: str,
    period_name: str = 'Custom Period',
    use_subprocess: bool = False  # Disabled by default - causes issues on Streamlit Cloud
) -> Dict:
    """
    Simulate historical accuracy by running backtest on a date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        period_name: Display name for the period
        use_subprocess: If True, run backtest in subprocess to free memory after completion
    
    Returns:
        Dictionary with simulation results and analysis
    """
    try:
        logger.info(f"Running accuracy simulation for {period_name}: {start_date} to {end_date}")
        
        # Run backtest for the specified period
        # Training data: from 5 years before start_date to start_date
        # Test data: from start_date to end_date
        training_end = pd.to_datetime(start_date) - timedelta(days=1)
        
        if use_subprocess:
            # Run in subprocess to ensure memory is freed after completion
            import multiprocessing as mp
            import pickle
            import tempfile
            
            def run_backtest_subprocess(train_end, test_start, test_end, output_file):
                """Run backtest in subprocess and save results to file."""
                try:
                    results_df, analysis = run_backtest(
                        train_end_date=train_end,
                        test_start_date=test_start,
                        test_end_date=test_end,
                        min_matches=10
                    )
                    # Save to temp file
                    with open(output_file, 'wb') as f:
                        pickle.dump((results_df, analysis), f)
                except Exception as e:
                    logger.error(f"Subprocess backtest failed: {e}")
                    with open(output_file, 'wb') as f:
                        pickle.dump((None, None), f)
            
            # Create temp file for results
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                output_file = tmp.name
            
            # Run in subprocess
            process = mp.Process(
                target=run_backtest_subprocess,
                args=(training_end.strftime('%Y-%m-%d'), start_date, end_date, output_file)
            )
            process.start()
            process.join(timeout=300)  # 5 minute timeout
            
            if process.is_alive():
                process.terminate()
                process.join()
                raise TimeoutError("Backtest timed out after 5 minutes")
            
            # Load results from temp file
            with open(output_file, 'rb') as f:
                results_df, analysis = pickle.load(f)
            
            # Clean up temp file
            import os
            os.unlink(output_file)
            
            if results_df is None:
                raise ValueError("Backtest failed in subprocess")
        else:
            # Run directly (original behavior)
            results_df, analysis = run_backtest(
                train_end_date=training_end.strftime('%Y-%m-%d'),
                test_start_date=start_date,
                test_end_date=end_date,
                min_matches=10
            )
        
        # Format results for display
        simulation_result = {
            'period_name': period_name,
            'start_date': start_date,
            'end_date': end_date,
            'status': 'success',
            'accuracy': analysis.get('accuracy', 0),
            'total_predictions': analysis.get('total_predictions', 0),
            'correct_predictions': analysis.get('correct_predictions', 0),
            'accuracy_by_confidence': analysis.get('accuracy_by_confidence', {}),
            'accuracy_by_surface': analysis.get('accuracy_by_surface', {}),
            'upset_analysis': analysis.get('upset_analysis', {}),
            'error_patterns': analysis.get('error_patterns', {}),
            'style_matchup_accuracy': analysis.get('style_matchup_accuracy', {}),
            'results_df': results_df
        }
        
        logger.info(f"Simulation complete: {analysis['accuracy']:.1%} accuracy")
        return simulation_result
        
    except Exception as e:
        logger.error(f"Error running accuracy simulation: {e}")
        return {
            'period_name': period_name,
            'start_date': start_date,
            'end_date': end_date,
            'status': 'error',
            'error': str(e),
            'accuracy': None
        }


def format_simulation_results(result: Dict) -> Dict:
    """
    Format simulation results for dashboard display.
    
    Args:
        result: Raw simulation result
    
    Returns:
        Formatted dictionary for display
    """
    if result['status'] == 'error':
        return {
            'period': result['period_name'],
            'error': result['error'],
            'display_accuracy': 'Error'
        }
    
    formatted = {
        'period': result['period_name'],
        'accuracy': result['accuracy'],
        'accuracy_pct': f"{result['accuracy']:.1%}",
        'total_predictions': result['total_predictions'],
        'correct_predictions': result['correct_predictions'],
        'date_range': f"{result['start_date']} to {result['end_date']}"
    }
    
    # Add confidence breakdown
    if result['accuracy_by_confidence']:
        confidence = result['accuracy_by_confidence']
        formatted['confidence_breakdown'] = {
            'high': {
                'accuracy': confidence.get('high (>=65%)', 0),
                'count': confidence.get('high_count', 0)
            },
            'medium': {
                'accuracy': confidence.get('medium (55-65%)', 0),
                'count': confidence.get('medium_count', 0)
            },
            'low': {
                'accuracy': confidence.get('low (<55%)', 0),
                'count': confidence.get('low_count', 0)
            }
        }
    
    # Add surface breakdown
    if result['accuracy_by_surface']:
        formatted['surface_breakdown'] = {}
        for surface, data in result['accuracy_by_surface'].items():
            formatted['surface_breakdown'][surface] = {
                'accuracy': data.get('accuracy', 0),
                'count': data.get('count', 0)
            }
    
    # Add upset analysis
    if result['upset_analysis']:
        upset = result['upset_analysis']
        formatted['upset_analysis'] = {
            'total_upsets': upset.get('total_upsets', 0),
            'upsets_correctly_predicted': upset.get('upsets_correctly_predicted', 0),
            'upset_accuracy': upset.get('upset_recall', 0),
            'upset_predictions': upset.get('upset_predictions_made', 0),
            'upset_precision': upset.get('upset_precision', 0)
        }
    
    return formatted


def get_simulation_summary(result: Dict) -> str:
    """
    Generate a text summary of simulation results.
    
    Args:
        result: Formatted simulation result
    
    Returns:
        String summary for display
    """
    if 'error' in result:
        return f"⚠️ Error: {result['error']}"
    
    accuracy_str = result.get('accuracy_pct', 'N/A')
    predictions_str = result.get('total_predictions', 0)
    
    summary = f"**{result['period']}** | "
    summary += f"Accuracy: {accuracy_str} | "
    summary += f"Predictions: {predictions_str} | "
    summary += f"Period: {result.get('date_range', 'N/A')}"
    
    return summary


if __name__ == "__main__":
    # Test the simulator
    periods = get_predefined_periods()
    
    print("Available predefined periods:")
    for key, period in periods.items():
        print(f"  {key}: {period.name} ({period.description})")
    
    # Run a test simulation for last 3 months
    print("\nRunning test simulation for last 3 months...")
    result = simulate_accuracy(
        start_date=periods['3_months'].start_date,
        end_date=periods['3_months'].end_date,
        period_name=periods['3_months'].name
    )
    
    if result['status'] == 'success':
        print(f"Accuracy: {result['accuracy']:.1%}")
        print(f"Total predictions: {result['total_predictions']}")
    else:
        print(f"Error: {result['error']}")
