"""
Tennis Upset Predictor Dashboard
Interactive Streamlit dashboard for tennis match predictions.

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Tennis Upset Predictor",
    page_icon="🎾",
    layout="wide"
)



from data_loader import load_matches, preprocess_matches
from player_profiler import PlayerProfiler
from matchup_model import MatchupModel
from accuracy_simulator import (
    get_predefined_periods, 
    simulate_accuracy, 
    format_simulation_results,
    get_simulation_summary
)
import re

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


def clean_time_display(time_str: str) -> str:
    """Clean time field for display (removes junk like '03:00Livestreams...')."""
    if not time_str:
        return "TBD"
    if 'live' in str(time_str).lower() and not re.match(r'^\d{1,2}:\d{2}', str(time_str)):
        return "LIVE"
    match = re.match(r'^(\d{1,2}:\d{2})', str(time_str))
    if match:
        return match.group(1)
    return str(time_str)[:5] if len(str(time_str)) > 5 else str(time_str)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data():
    """Load and cache match data from TML Database.
    Uses rolling 7-year window - automatically includes new years and drops oldest.
    Includes ongoing tournament matches for current form data.
    """
    # years=None uses automatic rolling window from get_rolling_years()
    # include_ongoing=True fetches live tournament data
    matches = load_matches(years=None, include_ongoing=True)
    matches = preprocess_matches(matches)
    return matches


@st.cache_resource(show_spinner=False)
def build_profiler_and_profiles(_matches):
    """Build and cache player profiler and profiles."""
    profiler = PlayerProfiler(_matches)
    profiles = profiler.build_all_profiles(min_matches=10)  # Lower threshold to include more players
    
    # Initialize the player matcher with loaded profiles for dynamic name matching
    from scraper import initialize_player_matcher
    initialize_player_matcher(profiles)
    
    return profiler, profiles


@st.cache_resource(show_spinner=False)
def get_trained_model(_matches, _profiles, _profiler):
    """Train and cache the prediction model with H2H support."""
    model = MatchupModel(_profiles, profiler=_profiler)  # Pass profiler for H2H lookups
    result = model.train(_matches, model_type='random_forest')
    return model, result


@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def get_upcoming_matches():
    """Get real upcoming tournament matches from TennisExplorer."""
    from scraper import scrape_tennis_explorer, get_sample_matches
    
    # Try to get real matches
    matches = scrape_tennis_explorer()
    
    if len(matches) == 0:
        st.warning("Could not fetch live matches. Showing sample data.")
        return get_sample_matches()
    
    return matches


def refresh_matches():
    """Clear the match cache to force a refresh."""
    from scraper import reset_te_session
    reset_te_session()
    get_upcoming_matches.clear()


def get_tournament_calendar():
    """Get 2025 tournament calendar."""
    return pd.DataFrame([
        {"tournament": "Australian Open", "start": "2025-01-13", "end": "2025-01-26", "surface": "Hard", "level": "Grand Slam", "location": "Melbourne"},
        {"tournament": "Indian Wells", "start": "2025-03-05", "end": "2025-03-16", "surface": "Hard", "level": "ATP 1000", "location": "California"},
        {"tournament": "Miami Open", "start": "2025-03-19", "end": "2025-03-30", "surface": "Hard", "level": "ATP 1000", "location": "Miami"},
        {"tournament": "Monte Carlo", "start": "2025-04-06", "end": "2025-04-13", "surface": "Clay", "level": "ATP 1000", "location": "Monaco"},
        {"tournament": "Madrid Open", "start": "2025-04-27", "end": "2025-05-04", "surface": "Clay", "level": "ATP 1000", "location": "Madrid"},
        {"tournament": "Italian Open", "start": "2025-05-11", "end": "2025-05-18", "surface": "Clay", "level": "ATP 1000", "location": "Rome"},
        {"tournament": "Roland Garros", "start": "2025-05-25", "end": "2025-06-08", "surface": "Clay", "level": "Grand Slam", "location": "Paris"},
        {"tournament": "Wimbledon", "start": "2025-06-30", "end": "2025-07-13", "surface": "Grass", "level": "Grand Slam", "location": "London"},
        {"tournament": "US Open", "start": "2025-08-25", "end": "2025-09-07", "surface": "Hard", "level": "Grand Slam", "location": "New York"},
        {"tournament": "ATP Finals", "start": "2025-11-09", "end": "2025-11-16", "surface": "Hard", "level": "Finals", "location": "Turin"},
    ])


# =============================================================================
# PAGE FUNCTIONS
# =============================================================================

def page_calendar(model, profiles):
    """Match Calendar page."""
    st.header("📅 Match Calendar & Predictions")
    
    # Timezone diagnostic display
    from scraper import get_current_timezone_offset, USER_TIMEZONE
    from datetime import datetime as dt
    tz_offset = get_current_timezone_offset()
    tz_name = "CDT" if tz_offset == -5 else "CST"
    current_time_cst = dt.now(USER_TIMEZONE).strftime("%I:%M %p")
    st.caption(f"🕐 Times shown in US Central ({tz_name}, GMT{tz_offset}) | Current time: {current_time_cst}")
    
    # Refresh button
    col_refresh, col_info = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Refresh Matches"):
            refresh_matches()
            st.rerun()
    
    upcoming = get_upcoming_matches()
    
    if len(upcoming) == 0:
        st.warning("No upcoming matches found.")
        return
    
    with col_info:
        st.info(f"📡 Scraped **{len(upcoming)} matches** from TennisExplorer (auto-refreshes every 5 min)")
    
    # Ensure date_label column exists (for backwards compatibility)
    if 'date_label' not in upcoming.columns:
        today_str = datetime.now().strftime("%Y-%m-%d")
        yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        tomorrow_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        def assign_label(d):
            if d == today_str:
                return 'Today'
            elif d == yesterday_str:
                return 'Yesterday'
            elif d == tomorrow_str:
                return 'Tomorrow'
            return 'Other'
        upcoming['date_label'] = upcoming['date'].apply(assign_label)
    
    # Determine match status for all matches
    def get_match_status(row):
        if row.get('status') == 'completed':
            return 'finished'
        else:
            return 'upcoming'
    
    upcoming['_status'] = upcoming.apply(get_match_status, axis=1)
    
    # Create datasets for each main tab
    # Upcoming: Today + Tomorrow matches that are not completed
    upcoming_matches = upcoming[
        (upcoming['_status'] == 'upcoming') & 
        (upcoming['date_label'].isin(['Today', 'Tomorrow']))
    ].copy()
    # Sort upcoming matches ascending (earliest first)
    if len(upcoming_matches) > 0:
        upcoming_matches = upcoming_matches.sort_values(by=['date', 'time'], ascending=True)
    
    # Finished: Today + Yesterday completed matches
    finished_matches = upcoming[
        (upcoming['_status'] == 'finished') & 
        (upcoming['date_label'].isin(['Today', 'Yesterday']))
    ].copy()
    # Sort finished matches descending (most recent first)
    if len(finished_matches) > 0:
        finished_matches = finished_matches.sort_values(by=['date', 'time'], ascending=False)
    
    # Filters section - visible before tabs
    st.subheader("🔍 Filters")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        surfaces = st.multiselect("Surface", ["Hard", "Clay", "Grass"], default=["Hard", "Clay", "Grass"])
    with col2:
        tournaments = sorted(upcoming['tournament'].unique().tolist())
        selected_tournament = st.selectbox("Tournament", ["All"] + tournaments)
    with col3:
        show_main_only = st.checkbox("🏆 Main events only", value=True, 
            help="Filter out UTR/Futures/ITF events (recommended for better predictions)")
    with col4:
        show_predicted_only = st.checkbox("🎯 Predicted only", value=True,
            help="Only show matches where we have player predictions")
    with col5:
        show_upsets_only = st.checkbox("⚠️ Upset alerts only", value=False)
    
    st.markdown("---")
    
    # Helper function to check if a match has a prediction
    def has_prediction(match):
        pred = model.predict_match(
            match['player1'], match['player2'], match['surface'],
            match.get('favorite'), tournament_name=match.get('tournament', '')
        )
        return pred is not None
    
    # Helper function to apply filters to a dataset
    def apply_filters(df):
        if len(df) == 0:
            return df
        result = df.copy()
        if surfaces:
            result = result[result['surface'].isin(surfaces)]
        if selected_tournament != "All":
            result = result[result['tournament'] == selected_tournament]
        if show_main_only:
            if 'is_main_tour' in result.columns:
                result = result[result['is_main_tour'] == True]
            else:
                exclude_keywords = ['utr', 'itf', 'futures', 'challenger']
                mask = ~result['tournament'].str.lower().str.contains('|'.join(exclude_keywords), na=False)
                result = result[mask]
        if show_predicted_only:
            # Filter to only matches with predictions
            mask = result.apply(has_prediction, axis=1)
            result = result[mask]
        return result
    
    # Helper function to display a single match card
    def display_match_card(match, model, show_upsets_only=False):
        pred = model.predict_match(
            match['player1'], 
            match['player2'], 
            match['surface'],
            match.get('favorite'),
            tournament_name=match.get('tournament', '')
        )
        
        has_prediction = pred is not None
        is_upset = has_prediction and pred.predicted_winner == pred.underdog
        
        if show_upsets_only and not is_upset:
            return False
        
        with st.container():
            cols = st.columns([1, 3, 2, 2])
            
            with cols[0]:
                match_time = clean_time_display(match.get('time', 'TBD'))
                is_finished = match.get('status') == 'completed'
                
                if is_finished:
                    st.markdown("✅ **FIN**")
                else:
                    st.write(f"**{match_time}**")
                st.caption(f"{match['surface']}")
            
            with cols[1]:
                if is_upset:
                    st.markdown(f"⚠️ **UPSET ALERT**")
                st.write(f"**{match['tournament']}**")
                
                odds1 = match.get('odds_player1')
                odds2 = match.get('odds_player2')
                favorite = match.get('favorite')
                winner = match.get('winner')
                score = match.get('score')
                
                p1_display = match['player1']
                p2_display = match['player2']
                is_finished = match.get('status') == 'completed'
                
                if is_finished and winner:
                    if winner == match['player1']:
                        p1_display = f"🏆 {p1_display}"
                    elif winner == match['player2']:
                        p2_display = f"🏆 {p2_display}"
                else:
                    if favorite == match['player1']:
                        p1_display = f"⭐ {p1_display}"
                    elif favorite == match['player2']:
                        p2_display = f"⭐ {p2_display}"
                
                if odds1:
                    p1_display = f"{p1_display} ({odds1:.2f})"
                if odds2:
                    p2_display = f"{p2_display} ({odds2:.2f})"
                
                st.write(f"🎾 **{p1_display}** vs **{p2_display}**")
                
                if is_finished and score:
                    st.caption(f"📊 Score: **{score}**")
            
            with cols[2]:
                if has_prediction:
                    p1_pct = pred.player1_win_prob * 100
                    p2_pct = pred.player2_win_prob * 100
                    st.caption(f"{match['player1'][:15]}: **{p1_pct:.0f}%**")
                    st.progress(pred.player1_win_prob)
                    st.caption(f"{match['player2'][:15]}: **{p2_pct:.0f}%**")
                    st.progress(pred.player2_win_prob)
                else:
                    if match.get('favorite'):
                        st.caption(f"📊 Odds Favorite: **{match['favorite']}**")
                    else:
                        st.caption("_No prediction available_")
                        st.caption("(Player not in historical data)")
            
            with cols[3]:
                if has_prediction:
                    winner = match.get('winner')
                    
                    # For finished matches, show if prediction was correct
                    if is_finished and winner:
                        prediction_correct = pred.predicted_winner == winner
                        if prediction_correct:
                            st.markdown(f"✅ **Correct!**")
                            st.caption(f"Predicted: {pred.predicted_winner}")
                        else:
                            st.markdown(f"❌ **Wrong**")
                            st.caption(f"Predicted: {pred.predicted_winner}")
                    else:
                        # Upcoming match - show prediction
                        winner_color = "green" if pred.predicted_winner == match.get('favorite') else "orange"
                        st.markdown(f"**Predicted:** :{winner_color}[{pred.predicted_winner}]")
                        
                        if pred.set_scores:
                            top_score = pred.set_scores[0]
                            st.caption(f"📊 Likely: **{top_score.score}** ({top_score.probability:.0%})")
                        
                        if pred.tiebreak_probability > 0.3:
                            st.caption(f"🎯 Tiebreak: {pred.tiebreak_probability:.0%}")
                    
                    if is_upset:
                        st.caption(f"⚠️ Upset conf: **{pred.confidence:.0%}**")
                else:
                    st.caption("—")
            
            if has_prediction and pred.upset_explanation:
                st.caption(f"💡 _{pred.upset_explanation}_")
            
            st.markdown("---")
        return True
    
    # Helper to display matches for a single day
    def display_day_matches(matches_df, model, show_upsets_only, sort_ascending=True):
        if len(matches_df) == 0:
            st.info("No matches found with current filters.")
            return 0
        
        sorted_matches = matches_df.sort_values(
            by='datetime_local', 
            ascending=sort_ascending, 
            na_position='last'
        )
        
        displayed = 0
        for _, match in sorted_matches.iterrows():
            if display_match_card(match, model, show_upsets_only):
                displayed += 1
        
        if displayed == 0:
            st.info("No matches match your filter criteria.")
        return displayed
    
    # Helper to get local date from datetime_local string
    def get_local_date(dt_str):
        if not dt_str:
            return None
        try:
            dt = datetime.fromisoformat(dt_str)
            return dt.strftime("%Y-%m-%d")
        except:
            return None
    
    # Date strings for comparison - use CST to match scraper data
    now_cst = dt.now(USER_TIMEZONE)
    today_str = now_cst.strftime("%Y-%m-%d")
    tomorrow_str = (now_cst + timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_str = (now_cst - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Main tabs: Upcoming and Finished
    tab_upcoming, tab_finished = st.tabs(["⏰ Upcoming", "✅ Finished"])
    
    # UPCOMING TAB with Today/Tomorrow sub-tabs
    with tab_upcoming:
        filtered_upcoming = apply_filters(upcoming_matches)
        
        # Split by date_label AND verify with actual date field to prevent cross-contamination
        today_upcoming = filtered_upcoming[
            (filtered_upcoming['date_label'] == 'Today') | 
            (filtered_upcoming['date'] == today_str)
        ].copy()
        # Tomorrow: must have Tomorrow label AND tomorrow's date (strict check)
        tomorrow_upcoming = filtered_upcoming[
            (filtered_upcoming['date_label'] == 'Tomorrow') & 
            (filtered_upcoming['date'] == tomorrow_str)
        ].copy()
        
        # Count upsets if filter is on
        if show_upsets_only:
            def count_upsets(matches_df):
                count = 0
                for _, match in matches_df.iterrows():
                    pred = model.predict_match(
                        match['player1'], match['player2'], match['surface'],
                        match.get('favorite'), tournament_name=match.get('tournament', '')
                    )
                    if pred is not None and pred.predicted_winner == pred.underdog:
                        count += 1
                return count
            
            today_count = count_upsets(today_upcoming)
            tomorrow_count = count_upsets(tomorrow_upcoming)
            total_count = today_count + tomorrow_count
            st.write(f"**{total_count}** upset predictions")
        else:
            today_count = len(today_upcoming)
            tomorrow_count = len(tomorrow_upcoming)
            st.write(f"**{len(filtered_upcoming)}** upcoming matches total")
        
        sub_today, sub_tomorrow = st.tabs([f"📅 Today ({today_str}) - {today_count}", f"📅 Tomorrow ({tomorrow_str}) - {tomorrow_count}"])
        
        with sub_today:
            display_day_matches(today_upcoming, model, show_upsets_only, sort_ascending=True)
        
        with sub_tomorrow:
            display_day_matches(tomorrow_upcoming, model, show_upsets_only, sort_ascending=True)
    
    # FINISHED TAB - Simple combined view with prediction success/failure tracking
    with tab_finished:
        filtered_finished = apply_filters(finished_matches)
        
        # Pre-calculate prediction outcomes for all matches
        if len(filtered_finished) > 0:
            all_matches = []
            correct_matches = []
            wrong_matches = []
            upset_pred_matches = []
            no_prediction_count = 0
            
            upset_correct_count = 0
            
            for _, match in filtered_finished.iterrows():
                match_data = match.to_dict()
                winner = match.get('winner')
                
                pred = model.predict_match(
                    match['player1'], 
                    match['player2'], 
                    match['surface'],
                    match.get('favorite'),
                    tournament_name=match.get('tournament', '')
                )
                
                # Track prediction outcome
                is_upset_pred = pred is not None and pred.predicted_winner == pred.underdog
                is_correct = pred is not None and winner and pred.predicted_winner == winner
                
                if pred is None or not winner:
                    no_prediction_count += 1
                    match_data['_outcome'] = 'no_prediction'
                elif is_correct:
                    match_data['_outcome'] = 'correct'
                    correct_matches.append(match_data)
                else:
                    match_data['_outcome'] = 'wrong'
                    wrong_matches.append(match_data)
                
                if is_upset_pred:
                    upset_pred_matches.append(match_data)
                    if is_correct:
                        upset_correct_count += 1
                
                all_matches.append(match_data)
            
            success_count = len(correct_matches)
            fail_count = len(wrong_matches)
            upset_count = len(upset_pred_matches)
            total_with_predictions = success_count + fail_count
            
            # Summary metrics - just accuracy and upset accuracy
            col_acc, col_upset_acc = st.columns(2)
            with col_acc:
                if total_with_predictions > 0:
                    accuracy = success_count / total_with_predictions
                    st.metric("📊 Accuracy", f"{accuracy:.1%}", help=f"{success_count} correct, {fail_count} wrong")
                else:
                    st.metric("📊 Accuracy", "N/A")
            with col_upset_acc:
                if upset_count > 0:
                    upset_accuracy = upset_correct_count / upset_count
                    st.metric("⚠️ Upset Accuracy", f"{upset_accuracy:.1%}", help=f"{upset_correct_count} of {upset_count} upset predictions correct")
                else:
                    st.metric("⚠️ Upset Accuracy", "N/A", help="No upset predictions")
            
            if no_prediction_count > 0:
                st.caption(f"({no_prediction_count} matches without predictions)")
            
            st.markdown("---")
            
            # Sub-tabs for filtering by outcome
            tab_all, tab_correct, tab_wrong, tab_upsets = st.tabs([
                f"📋 All ({len(all_matches)})",
                f"✅ Correct ({success_count})",
                f"❌ Wrong ({fail_count})",
                f"⚠️ Upset Predictions ({upset_count})"
            ])
            
            with tab_all:
                if len(all_matches) > 0:
                    display_day_matches(pd.DataFrame(all_matches), model, show_upsets_only=False, sort_ascending=False)
                else:
                    st.info("No matches found.")
            
            with tab_correct:
                if len(correct_matches) > 0:
                    display_day_matches(pd.DataFrame(correct_matches), model, show_upsets_only=False, sort_ascending=False)
                else:
                    st.info("No correct predictions.")
            
            with tab_wrong:
                if len(wrong_matches) > 0:
                    display_day_matches(pd.DataFrame(wrong_matches), model, show_upsets_only=False, sort_ascending=False)
                else:
                    st.info("No wrong predictions.")
            
            with tab_upsets:
                if len(upset_pred_matches) > 0:
                    display_day_matches(pd.DataFrame(upset_pred_matches), model, show_upsets_only=False, sort_ascending=False)
                else:
                    st.info("No upset predictions in finished matches.")
        else:
            st.write("**0** finished matches")
            st.info("No matches found with current filters.")


def page_upsets(model, profiles):
    """Upset Alerts page."""
    from scraper import USER_TIMEZONE
    from datetime import datetime as dt
    st.header("🎯 Upset Alerts")
    st.write("Matches where we predict the underdog to win, sorted by value (confidence × odds)")
    
    # Refresh button
    col_refresh, col_info = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Refresh Matches", key="refresh_upsets"):
            refresh_matches()
            st.rerun()
    
    # Get matches
    upcoming = get_upcoming_matches()
    
    if len(upcoming) == 0:
        st.warning("No matches found.")
        return
    
    # Determine match status
    def get_match_status(row):
        if row.get('status') == 'completed':
            return 'finished'
        else:
            return 'upcoming'
    
    upcoming['_status'] = upcoming.apply(get_match_status, axis=1)
    
    # Create datasets
    upcoming_matches = upcoming[upcoming['_status'] == 'upcoming']
    finished_matches = upcoming[upcoming['_status'] == 'finished']
    
    # Filters section
    st.subheader("🔍 Filters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        surfaces = st.multiselect("Surface", ["Hard", "Clay", "Grass"], default=["Hard", "Clay", "Grass"], key="upset_surfaces")
    with col2:
        tournaments = sorted(upcoming['tournament'].unique().tolist())
        selected_tournament = st.selectbox("Tournament", ["All"] + tournaments, key="upset_tournament")
    with col3:
        show_main_only = st.checkbox("🏆 Main events only", value=True, key="upset_main",
            help="Filter out UTR/Futures/ITF events")
    with col4:
        min_conf = st.slider("Min confidence", 0.50, 0.70, 0.55, 0.05, key="upset_conf",
                             help="Minimum confidence for upset prediction")
    
    st.markdown("---")
    
    # Helper to apply filters
    def apply_filters(df):
        if len(df) == 0:
            return df
        result = df.copy()
        if surfaces:
            result = result[result['surface'].isin(surfaces)]
        if selected_tournament != "All":
            result = result[result['tournament'] == selected_tournament]
        if show_main_only:
            if 'is_main_tour' in result.columns:
                result = result[result['is_main_tour'] == True]
            else:
                exclude_keywords = ['utr', 'itf', 'futures', 'challenger']
                mask = ~result['tournament'].str.lower().str.contains('|'.join(exclude_keywords), na=False)
                result = result[mask]
        return result
    
    # Helper to get local date from datetime_local
    def get_local_date(dt_str):
        if not dt_str:
            return None
        try:
            dt = datetime.fromisoformat(dt_str)
            return dt.strftime("%Y-%m-%d")
        except:
            return None
    
    # Helper to find upsets and calculate value score
    def find_upsets(matches_df):
        upsets = []
        for _, match in matches_df.iterrows():
            match_time = clean_time_display(match.get('time', 'TBD'))
            is_finished = match.get('status') == 'completed'
            
            pred = model.predict_match(
                match['player1'], 
                match['player2'], 
                match['surface'],
                match.get('favorite'),
                tournament_name=match.get('tournament', '')
            )
            
            if pred and pred.predicted_winner == pred.underdog and pred.confidence >= min_conf:
                top_set_score = pred.set_scores[0] if pred.set_scores else None
                
                odds1 = match.get('odds_player1')
                odds2 = match.get('odds_player2')
                underdog_odds = odds2 if pred.underdog == match['player2'] else odds1
                
                # Value score = confidence * underdog_odds (higher = better value bet)
                value_score = pred.confidence * (underdog_odds if underdog_odds else 1.0)
                
                upsets.append({
                    'Time': '✅ FIN' if is_finished else match_time,
                    'datetime_local': match.get('datetime_local'),
                    'Tournament': match['tournament'],
                    'Underdog': pred.underdog,
                    'Favorite': pred.favorite,
                    'Underdog Odds': underdog_odds,
                    'Favorite Odds': odds1 if pred.favorite == match['player1'] else odds2,
                    'Confidence': pred.confidence,
                    'Value Score': value_score,
                    'Surface': match['surface'],
                    'Style': pred.style_matchup,
                    'Key Factor': pred.key_factors[0] if pred.key_factors else "",
                    'Explanation': pred.upset_explanation,
                    'Set Score': f"{top_set_score.score} ({top_set_score.probability:.0%})" if top_set_score else "N/A",
                    'Tiebreak %': pred.tiebreak_probability,
                    'is_finished': is_finished,
                    'Score': match.get('score'),
                    'Winner': match.get('winner')
                })
        return pd.DataFrame(upsets) if upsets else pd.DataFrame()
    
    # Helper to display upset cards sorted by value
    def display_upset_cards(upset_df):
        if len(upset_df) == 0:
            st.info(f"No upset predictions found (confidence >= {min_conf:.0%})")
            return
        
        # Sort by Value Score descending (best value first)
        upset_df = upset_df.sort_values(by='Value Score', ascending=False)
        
        for _, row in upset_df.iterrows():
            finished_badge = "✅ " if row.get('is_finished', False) else ""
            
            underdog_odds = row.get('Underdog Odds')
            odds_str = f" @ {underdog_odds:.2f}" if underdog_odds else ""
            value_str = f" | Value: {row['Value Score']:.2f}" if underdog_odds else ""
            
            result_str = ""
            if row.get('is_finished') and row.get('Winner'):
                if row['Winner'] == row['Underdog']:
                    result_str = " ✓ UPSET!"
                else:
                    result_str = " ✗ Fav won"
            
            with st.expander(f"{finished_badge}⚠️ {row['Underdog']}{odds_str} ({row['Confidence']:.0%}){value_str} over {row['Favorite']}{result_str} | {row['Tournament']}", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{row['Confidence']:.1%}")
                    st.metric("Value Score", f"{row['Value Score']:.2f}" if row.get('Underdog Odds') else "N/A")
                    st.write(f"**Time:** {row['Time']}")
                    
                    if row.get('is_finished') and row.get('Score'):
                        winner_icon = "🏆" if row.get('Winner') == row['Underdog'] else "❌"
                        st.write(f"**Result:** {winner_icon} {row['Score']}")
                    
                with col2:
                    st.write(f"**Underdog:** {row['Underdog']}")
                    if row.get('Underdog Odds'):
                        st.write(f"**Underdog Odds:** {row['Underdog Odds']:.2f}")
                    st.write(f"**Favorite:** {row['Favorite']}")
                    if row.get('Favorite Odds'):
                        st.write(f"**Favorite Odds:** {row['Favorite Odds']:.2f}")
                        
                with col3:
                    st.write(f"**Surface:** {row['Surface']}")
                    st.write(f"**Predicted Score:** {row['Set Score']}")
                    st.write(f"**Tiebreak %:** {row['Tiebreak %']:.0%}")
                
                st.info(f"💡 {row['Explanation']}")
    
    # Date strings for comparison - use CST to match scraper data
    now_cst = dt.now(USER_TIMEZONE)
    today_str = now_cst.strftime("%Y-%m-%d")
    tomorrow_str = (now_cst + timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_str = (now_cst - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Main tabs: Upcoming and Finished
    tab_upcoming, tab_finished = st.tabs(["⏰ Upcoming Upsets", "✅ Finished Upsets"])
    
    # UPCOMING TAB with Today/Tomorrow sub-tabs
    with tab_upcoming:
        filtered_upcoming = apply_filters(upcoming_matches)
        
        # Get upsets for all upcoming matches
        all_upcoming_upsets = find_upsets(filtered_upcoming)
        
        if len(all_upcoming_upsets) == 0:
            st.info(f"No upcoming upset predictions found (confidence >= {min_conf:.0%})")
        else:
            # Split by local date
            all_upcoming_upsets['_local_date'] = all_upcoming_upsets['datetime_local'].apply(get_local_date)
            today_upsets = all_upcoming_upsets[all_upcoming_upsets['_local_date'] == today_str]
            tomorrow_upsets = all_upcoming_upsets[all_upcoming_upsets['_local_date'] == tomorrow_str]
            no_date_upsets = all_upcoming_upsets[all_upcoming_upsets['_local_date'].isna()]
            today_upsets = pd.concat([today_upsets, no_date_upsets])
            
            st.write(f"**{len(all_upcoming_upsets)}** upcoming upset predictions total")
            
            sub_today, sub_tomorrow = st.tabs([f"📅 Today ({today_str}) - {len(today_upsets)}", f"📅 Tomorrow ({tomorrow_str}) - {len(tomorrow_upsets)}"])
            
            with sub_today:
                display_upset_cards(today_upsets)
            
            with sub_tomorrow:
                display_upset_cards(tomorrow_upsets)
    
    # FINISHED TAB with Today/Yesterday sub-tabs
    with tab_finished:
        filtered_finished = apply_filters(finished_matches)
        
        # Get upsets for all finished matches
        all_finished_upsets = find_upsets(filtered_finished)
        
        if len(all_finished_upsets) == 0:
            st.info(f"No finished upset predictions found (confidence >= {min_conf:.0%})")
        else:
            # Split by local date
            all_finished_upsets['_local_date'] = all_finished_upsets['datetime_local'].apply(get_local_date)
            today_fin_upsets = all_finished_upsets[all_finished_upsets['_local_date'] == today_str]
            yesterday_fin_upsets = all_finished_upsets[all_finished_upsets['_local_date'] == yesterday_str]
            no_date_fin = all_finished_upsets[all_finished_upsets['_local_date'].isna()]
            today_fin_upsets = pd.concat([today_fin_upsets, no_date_fin])
            
            st.write(f"**{len(all_finished_upsets)}** finished upset predictions total")
            
            # Calculate success rate
            if len(all_finished_upsets) > 0:
                correct = (all_finished_upsets['Winner'] == all_finished_upsets['Underdog']).sum()
                success_rate = correct / len(all_finished_upsets)
                st.metric("Upset Success Rate", f"{success_rate:.1%}", f"{correct}/{len(all_finished_upsets)} correct")
            
            sub_today_fin, sub_yesterday_fin = st.tabs([f"📅 Today ({today_str}) - {len(today_fin_upsets)}", f"📅 Yesterday ({yesterday_str}) - {len(yesterday_fin_upsets)}"])
            
            with sub_today_fin:
                display_upset_cards(today_fin_upsets)
            
            with sub_yesterday_fin:
                display_upset_cards(yesterday_fin_upsets)


def page_players(profiles, profiler=None):
    """Player Profiles page."""
    st.header("👤 Player Profiles")
    
    # Player selector
    player_names = sorted(profiles['player_name'].dropna().unique().tolist())
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.selectbox("Select Player", player_names, index=0)
    with col2:
        compare = st.selectbox("Compare with (optional)", ["None"] + player_names)
    
    if not selected:
        return
    
    player = profiles[profiles['player_name'] == selected].iloc[0]
    
    # Profile cards
    st.subheader(f"📊 {selected}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Matches", int(player.get('total_matches', 0)))
        st.metric("Win Rate", f"{player.get('overall_win_pct', 0):.1%}")
    
    with col2:
        st.metric("Style", player.get('style_cluster', 'Unknown'))
        st.metric("Clutch Rating", f"{player.get('clutch_rating', 50):.0f}/100")
    
    with col3:
        st.metric("Tiebreak Win %", f"{player.get('tiebreak_win_pct', 0.5):.1%}")
        st.metric("BP Save %", f"{player.get('bp_save_pct', 0.6):.1%}")
    
    with col4:
        st.metric("Best Surface", player.get('surface_specialist', 'Hard').title())
        st.metric("Serve Dominance", f"{player.get('serve_dominance', 50):.0f}/100")
    
    with col5:
        # Recent form indicators
        recent_form = player.get('recent_form', 0.5)
        win_streak = int(player.get('win_streak', 0))
        loss_streak = int(player.get('loss_streak', 0))
        
        form_emoji = "🔥" if recent_form > 0.6 else ("❄️" if recent_form < 0.4 else "➡️")
        streak_text = f"+{win_streak}W" if win_streak > 0 else (f"-{loss_streak}L" if loss_streak > 0 else "—")
        
        st.metric("Recent Form", f"{form_emoji} {recent_form:.0%}")
        st.metric("Current Streak", streak_text)
    
    # Radar chart
    st.subheader("Player Style Radar")
    
    categories = ['Serve', 'Return', 'Clutch', 'Hard Court', 'Clay Court', 'Grass Court']
    
    def get_radar_values(p):
        return [
            min(p.get('serve_dominance', 50) / 100, 1),
            min(p.get('return_points_won_pct', 0.4) * 2.5, 1),
            min(p.get('clutch_rating', 50) / 100, 1),
            p.get('hard_win_pct', 0.5),
            p.get('clay_win_pct', 0.5),
            p.get('grass_win_pct', 0.5)
        ]
    
    fig = go.Figure()
    
    # Selected player
    values = get_radar_values(player)
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=selected
    ))
    
    # Comparison player
    if compare != "None":
        player2 = profiles[profiles['player_name'] == compare].iloc[0]
        values2 = get_radar_values(player2)
        fig.add_trace(go.Scatterpolar(
            r=values2 + [values2[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=compare
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Surface performance table
    st.subheader("Surface Performance")
    surface_data = pd.DataFrame({
        'Surface': ['Hard', 'Clay', 'Grass'],
        'Win %': [
            f"{player.get('hard_win_pct', 0.5):.1%}",
            f"{player.get('clay_win_pct', 0.5):.1%}",
            f"{player.get('grass_win_pct', 0.5):.1%}"
        ]
    })
    st.dataframe(surface_data, use_container_width=True, hide_index=True)


def page_h2h(model, profiles, profiler=None):
    """Head-to-Head page."""
    st.header("⚔️ Head-to-Head Prediction")
    
    player_names = sorted(profiles['player_name'].dropna().unique().tolist())
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        player1 = st.selectbox("Player 1", player_names, index=0, key="h2h_p1")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
    
    with col3:
        player2 = st.selectbox("Player 2", player_names, index=min(1, len(player_names)-1), key="h2h_p2")
    
    if player1 == player2:
        st.warning("Please select two different players")
        return
    
    # Get prediction
    pred = model.predict_match(player1, player2, surface)
    
    if pred is None:
        st.error("Could not generate prediction for these players")
        return
    
    # Get H2H record if profiler available
    h2h_record = None
    if profiler is not None:
        h2h_record = profiler.get_h2h_record(player1, player2)
    
    st.markdown("---")
    
    # H2H Record Banner
    if h2h_record and h2h_record['total_matches'] > 0:
        h2h_col1, h2h_col2, h2h_col3 = st.columns([1, 2, 1])
        with h2h_col2:
            st.info(f"📊 **Head-to-Head Record:** {player1} {h2h_record['p1_wins']} - {h2h_record['p2_wins']} {player2} ({h2h_record['total_matches']} matches)")
    
    # Results
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        p1_profile = profiles[profiles['player_name'] == player1].iloc[0]
        st.subheader(player1)
        st.metric("Win Probability", f"{pred.player1_win_prob:.1%}")
        st.write(f"**Style:** {p1_profile.get('style_cluster', 'Unknown')}")
        st.write(f"**Clutch:** {p1_profile.get('clutch_rating', 50):.0f}/100")
        st.write(f"**{surface} Win %:** {p1_profile.get(f'{surface.lower()}_win_pct', 0.5):.1%}")
        
        # Recent form
        form1 = p1_profile.get('recent_form', 0.5)
        streak1 = int(p1_profile.get('win_streak', 0)) or -int(p1_profile.get('loss_streak', 0))
        form_emoji1 = "🔥" if form1 > 0.6 else ("❄️" if form1 < 0.4 else "➡️")
        st.write(f"**Recent Form:** {form_emoji1} {form1:.0%}")
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center'>⚔️</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center'><b>{surface} Court</b></p>", unsafe_allow_html=True)
    
    with col3:
        p2_profile = profiles[profiles['player_name'] == player2].iloc[0]
        st.subheader(player2)
        st.metric("Win Probability", f"{pred.player2_win_prob:.1%}")
        st.write(f"**Style:** {p2_profile.get('style_cluster', 'Unknown')}")
        st.write(f"**Clutch:** {p2_profile.get('clutch_rating', 50):.0f}/100")
        st.write(f"**{surface} Win %:** {p2_profile.get(f'{surface.lower()}_win_pct', 0.5):.1%}")
        
        # Recent form
        form2 = p2_profile.get('recent_form', 0.5)
        streak2 = int(p2_profile.get('win_streak', 0)) or -int(p2_profile.get('loss_streak', 0))
        form_emoji2 = "🔥" if form2 > 0.6 else ("❄️" if form2 < 0.4 else "➡️")
        st.write(f"**Recent Form:** {form_emoji2} {form2:.0%}")
    
    # Prediction box
    st.markdown("---")
    winner_emoji = "🏆"
    st.success(f"{winner_emoji} **Predicted Winner: {pred.predicted_winner}** (Confidence: {pred.confidence:.0%})")
    
    # Set score and tiebreak predictions
    col_set, col_tb = st.columns(2)
    with col_set:
        st.subheader("📊 Set Score Predictions")
        if pred.set_scores:
            for score_pred in pred.set_scores[:4]:
                bar_pct = int(score_pred.probability * 100)
                tb_indicator = " 🎯" if score_pred.tiebreaks_expected > 0 else ""
                st.write(f"**{score_pred.score}**: {score_pred.probability:.0%}{tb_indicator}")
                st.progress(score_pred.probability)
    
    with col_tb:
        st.subheader("🎯 Tiebreak Analysis")
        st.metric("Tiebreak Probability", f"{pred.tiebreak_probability:.0%}")
        if pred.tiebreak_probability > 0.5:
            st.warning("High likelihood of at least one tiebreak")
        elif pred.tiebreak_probability > 0.3:
            st.info("Moderate chance of a tiebreak")
        else:
            st.success("Low tiebreak probability - expect decisive sets")
    
    # Upset explanation
    st.markdown("---")
    st.info(f"💡 **Analysis:** {pred.upset_explanation}")
    
    # Key factors in expandable section
    with st.expander("🔍 Key Factors", expanded=True):
        for factor in pred.key_factors:
            st.write(f"• {factor}")


def page_leaderboards(profiles):
    """Leaderboards page."""
    st.header("🏆 Player Leaderboards")
    
    metric = st.selectbox(
        "Rank by",
        ["Clutch Rating", "Overall Win %", "Tiebreak Win %", "Hard Court %", "Clay Court %", "Total Matches"],
    )
    
    metric_map = {
        "Clutch Rating": "clutch_rating",
        "Overall Win %": "overall_win_pct",
        "Tiebreak Win %": "tiebreak_win_pct",
        "Hard Court %": "hard_win_pct",
        "Clay Court %": "clay_win_pct",
        "Total Matches": "total_matches"
    }
    
    col = metric_map[metric]
    
    top_players = profiles.nlargest(25, col)[
        ['player_name', 'style_cluster', col, 'total_matches', 'overall_win_pct']
    ].copy()
    top_players.index = range(1, len(top_players) + 1)
    top_players.columns = ['Player', 'Style', metric, 'Matches', 'Win %']
    
    # Format percentages
    if '%' in metric or metric == 'Win %':
        top_players[metric] = top_players[metric].apply(lambda x: f"{x:.1%}" if x < 1 else f"{x:.1f}")
    
    top_players['Win %'] = top_players['Win %'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(top_players, use_container_width=True)
    
    # Style distribution
    st.subheader("Style Distribution")
    style_counts = profiles['style_cluster'].value_counts()
    fig = px.pie(values=style_counts.values, names=style_counts.index, title="Player Styles")
    st.plotly_chart(fig, use_container_width=True)


def page_calendar_tournaments():
    """Tournament calendar page."""
    st.header("📅 2025 Tournament Calendar")
    
    calendar = get_tournament_calendar()
    calendar['start'] = pd.to_datetime(calendar['start'])
    calendar['end'] = pd.to_datetime(calendar['end'])
    
    # Filter by surface
    surface = st.multiselect("Filter by surface", ["Hard", "Clay", "Grass"], default=["Hard", "Clay", "Grass"])
    filtered = calendar[calendar['surface'].isin(surface)]
    
    # Timeline chart
    fig = px.timeline(
        filtered,
        x_start='start',
        x_end='end',
        y='tournament',
        color='surface',
        title="Major Tournaments 2025",
        color_discrete_map={"Hard": "#3498db", "Clay": "#e67e22", "Grass": "#27ae60"}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.subheader("Tournament Details")
    display_df = filtered.copy()
    display_df['start'] = display_df['start'].dt.strftime('%b %d')
    display_df['end'] = display_df['end'].dt.strftime('%b %d')
    display_df.columns = ['Tournament', 'Start', 'End', 'Surface', 'Level', 'Location']
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def page_historical_accuracy():
    """Display historical model accuracy simulation over various time periods."""
    st.title("📊 Historical Accuracy Analysis")
    
    st.markdown("""
    **Simulate and analyze historical model performance** across different time periods.
    The simulator trains on historical data and tests predictions on subsequent matches
    to show how the model would have performed.
    """)
    
    st.markdown("---")
    
    # Get predefined periods
    periods = get_predefined_periods()
    
    # Tabs for different selection methods
    tab1, tab2 = st.tabs(["📅 Predefined Periods", "🎯 Custom Date Range"])
    
    with tab1:
        st.subheader("Quick Analysis with Predefined Periods")
        
        # Create columns for period buttons
        col1, col2, col3 = st.columns(3)
        
        selected_period = None
        with col1:
            if st.button("📆 Current Week", key="btn_week", use_container_width=True):
                selected_period = periods['current_week']
        with col2:
            if st.button("📅 Current Month", key="btn_month", use_container_width=True):
                selected_period = periods['current_month']
        with col3:
            if st.button("📊 Last Month", key="btn_last_month", use_container_width=True):
                selected_period = periods['last_month']
        
        st.markdown("")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Last 1 Month", key="btn_1m", use_container_width=True):
                selected_period = periods['1_month']
        with col2:
            if st.button("📈 Last 3 Months", key="btn_3m", use_container_width=True):
                selected_period = periods['3_months']
        with col3:
            if st.button("📈 Last 6 Months", key="btn_6m", use_container_width=True):
                selected_period = periods['6_months']
        
        st.markdown("")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Last 1 Year", key="btn_1y", use_container_width=True):
                selected_period = periods['1_year']
        
        # Run simulation if period selected
        if selected_period:
            with st.spinner(f"⏳ Running accuracy simulation for {selected_period.name}..."):
                result = simulate_accuracy(
                    start_date=selected_period.start_date,
                    end_date=selected_period.end_date,
                    period_name=selected_period.name
                )
            
            if result['status'] == 'success':
                _display_accuracy_results(result)
            else:
                st.error(f"❌ Simulation Error: {result['error']}")
    
    with tab2:
        st.subheader("Custom Date Range Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=180),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                min_value=start_date,
                max_value=datetime.now()
            )
        
        if st.button("🔍 Analyze Custom Period", use_container_width=True):
            if end_date <= start_date:
                st.error("❌ End date must be after start date")
            else:
                with st.spinner("⏳ Running accuracy simulation..."):
                    result = simulate_accuracy(
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        period_name=f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    )
                
                if result['status'] == 'success':
                    _display_accuracy_results(result)
                else:
                    st.error(f"❌ Simulation Error: {result['error']}")


def _display_accuracy_results(result: Dict):
    """Helper function to display accuracy simulation results."""
    formatted = format_simulation_results(result)
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Overall Accuracy", formatted['accuracy_pct'])
    
    with col2:
        st.metric("✅ Correct Predictions", formatted['correct_predictions'])
    
    with col3:
        st.metric("📈 Total Predictions", formatted['total_predictions'])
    
    with col4:
        st.metric("📅 Period", formatted['date_range'][:20] + "...")
    
    st.markdown("---")
    
    # Confidence breakdown
    if 'confidence_breakdown' in formatted:
        st.subheader("📊 Accuracy by Confidence Level")
        conf_data = formatted['confidence_breakdown']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_acc = conf_data['high']['accuracy']
            high_count = conf_data['high']['count']
            st.metric(
                "🔴 High (≥65%)",
                f"{high_acc:.1%}" if high_acc else "N/A",
                f"{high_count} predictions"
            )
        
        with col2:
            med_acc = conf_data['medium']['accuracy']
            med_count = conf_data['medium']['count']
            st.metric(
                "🟡 Medium (55-65%)",
                f"{med_acc:.1%}" if med_acc else "N/A",
                f"{med_count} predictions"
            )
        
        with col3:
            low_acc = conf_data['low']['accuracy']
            low_count = conf_data['low']['count']
            st.metric(
                "🟢 Low (<55%)",
                f"{low_acc:.1%}" if low_acc else "N/A",
                f"{low_count} predictions"
            )
    
    st.markdown("---")
    
    # Surface breakdown
    if 'surface_breakdown' in formatted and formatted['surface_breakdown']:
        st.subheader("🎾 Accuracy by Surface")
        
        surface_data = []
        for surface, data in formatted['surface_breakdown'].items():
            surface_data.append({
                'Surface': surface,
                'Accuracy': f"{data['accuracy']:.1%}",
                'Predictions': data['count']
            })
        
        surface_df = pd.DataFrame(surface_data)
        st.dataframe(surface_df, use_container_width=True, hide_index=True)
    
    # Upset analysis
    if 'upset_analysis' in formatted:
        st.markdown("---")
        st.subheader("⚠️ Upset Prediction Performance")
        
        upset = formatted['upset_analysis']
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("⚠️ Total Upsets Occurred", upset.get('total_upsets', 0))
        
        with col2:
            st.metric("✅ Upsets Correctly Predicted", upset.get('upsets_correctly_predicted', 0))
        
        with col3:
            upset_acc = upset.get('upset_accuracy', 0)
            st.metric("🎯 Upset Recall", f"{upset_acc:.1%}" if upset_acc else "N/A")
        
        with col4:
            upset_pred = upset.get('upset_predictions', 0)
            st.metric("🔮 Upset Predictions Made", upset_pred)
        
        with col5:
            upset_prec = upset.get('upset_precision', 0)
            st.metric("📊 Upset Precision", f"{upset_prec:.1%}" if upset_prec else "N/A")
        
        # Deep dive error analysis
        results_df = result.get('results_df')
        if results_df is not None and len(results_df) > 0:
            with st.expander("🔬 Deep Dive: Analyze Prediction Errors"):
                st.markdown("**Error Pattern Analysis**")
                st.caption("Analyzing why predictions failed to identify improvement opportunities")
                
                from upset_analyzer import analyze_error_patterns
                
                error_analysis = analyze_error_patterns(results_df, formatted['period'])
                
                if 'confusion_matrix' in error_analysis:
                    cm = error_analysis['confusion_matrix']
                    
                    st.markdown("### Prediction Categories")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("✅ True Positives", cm['true_positives'], 
                                 help="Correctly predicted upsets")
                    with col2:
                        st.metric("❌ False Positives", cm['false_positives'],
                                 help="Predicted upset but favorite won")
                    with col3:
                        st.metric("❌ False Negatives", cm['false_negatives'],
                                 help="Missed upsets")
                    with col4:
                        st.metric("✅ True Negatives", cm['true_negatives'],
                                 help="Correctly predicted favorite wins")
                    
                    # Analyze False Positives
                    if 'false_positive_patterns' in error_analysis and error_analysis['false_positive_patterns']:
                        st.markdown("---")
                        st.markdown("### ❌ False Positive Patterns")
                        st.caption("Why did we predict upsets that didn't happen?")
                        
                        for pattern in error_analysis['false_positive_patterns']:
                            st.write(f"**{pattern['name']}**")
                            st.write(f"  • {pattern['description']}")
                            st.write(f"  • Frequency: {pattern['frequency']:.1%} of false positives")
                            st.write(f"  • Count: {pattern['count']} matches")
                            st.write("")
                    
                    # Analyze False Negatives
                    if 'false_negative_patterns' in error_analysis and error_analysis['false_negative_patterns']:
                        st.markdown("---")
                        st.markdown("### ❌ False Negative Patterns")
                        st.caption("Why did we miss these upsets?")
                        
                        for pattern in error_analysis['false_negative_patterns']:
                            st.write(f"**{pattern['name']}**")
                            st.write(f"  • {pattern['description']}")
                            st.write(f"  • Frequency: {pattern['frequency']:.1%} of missed upsets")
                            st.write(f"  • Count: {pattern['count']} matches")
                            st.write("")
                    
                    # Improvement suggestions section
                    st.markdown("---")
                    st.markdown("### 💡 Next Steps")
                    st.info("""
                    **To improve the model:**
                    1. Review the patterns above
                    2. Consider if they make sense in real tennis
                    3. Report your findings in the chat with approved/rejected patterns
                    4. I'll implement the approved improvements
                    
                    **For multi-period validation:**
                    Run this analysis for 3 months, 6 months, and 12 months to identify consistent patterns.
                    """)
    
    # Summary text
    st.markdown("---")
    st.info(f"""
    **📌 Summary:** {get_simulation_summary(formatted)}
    
    **Interpretation:**
    - **Accuracy**: Percentage of correct predictions
    - **Confidence Levels**: How accuracy varies by prediction confidence
    - **Surface Breakdown**: Performance across different court types
    - **Upset Analysis**: How well the model identifies and predicts upsets
    """)

    # Optional detailed results table (guarded for performance)
    results_df = result.get('results_df')
    if results_df is not None and len(results_df) > 0:
        with st.expander("Show prediction details (may be slower)"):
            display_cols = [
                'match_date', 'tournament', 'surface', 'player1', 'player2',
                'predicted_winner', 'actual_winner', 'correct', 'confidence',
                'upset_predicted', 'was_upset', 'upset_correct', 'p1_win_prob', 'p2_win_prob'
            ]
            existing_cols = [c for c in display_cols if c in results_df.columns]
            preview_df = results_df[existing_cols].copy()
            preview_df['confidence'] = preview_df['confidence'].apply(lambda x: f"{x:.1%}")
            preview_df['p1_win_prob'] = preview_df['p1_win_prob'].apply(lambda x: f"{x:.1%}")
            preview_df['p2_win_prob'] = preview_df['p2_win_prob'].apply(lambda x: f"{x:.1%}")
            max_rows = 200
            if len(preview_df) > max_rows:
                st.caption(f"Showing first {max_rows} of {len(preview_df)} matches")
                preview_df = preview_df.head(max_rows)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)


def page_custom_predictor(model, profiles):
    """Custom match prediction page."""
    st.header("🔮 Custom Match Predictor")
    st.markdown("Predict the outcome of any hypothetical tennis match between two players.")

    # Player selection
    st.subheader("👤 Select Players")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Player 1**")
        player1_options = sorted(profiles['player_name'].unique())
        player1 = st.selectbox("Choose Player 1", player1_options, key="p1")

    with col2:
        st.markdown("**Player 2**")
        player2_options = sorted(profiles['player_name'].unique())
        player2 = st.selectbox("Choose Player 2", player2_options, key="p2")

    # Match conditions
    st.subheader("🎾 Match Conditions")

    col3, col4, col5 = st.columns(3)

    with col3:
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"], index=0)

    with col4:
        tournament_level = st.selectbox("Tournament Level", ["G", "M", "A", "B", "C"], 
                                       format_func=lambda x: {"G": "Grand Slam", "M": "Masters 1000", "A": "ATP 500", "B": "ATP 250", "C": "Challenger"}[x],
                                       index=2)

    with col5:
        # Try to determine favorite based on rankings if available
        p1_profile = profiles[profiles['player_name'] == player1]
        p2_profile = profiles[profiles['player_name'] == player2]

        favorite_options = [f"{player1} (Favorite)", f"{player2} (Favorite)", "Unknown/No Odds"]
        favorite_idx = 2  # Default to unknown

        if not p1_profile.empty and not p2_profile.empty:
            p1_rank = p1_profile['current_rank'].iloc[0] if 'current_rank' in p1_profile.columns else 999
            p2_rank = p2_profile['current_rank'].iloc[0] if 'current_rank' in p2_profile.columns else 999

            if pd.notna(p1_rank) and pd.notna(p2_rank):
                if p1_rank < p2_rank:
                    favorite_options[0] = f"{player1} (Rank {int(p1_rank)})"
                    favorite_options[1] = f"{player2} (Rank {int(p2_rank)})"
                    favorite_idx = 0
                elif p2_rank < p1_rank:
                    favorite_options[0] = f"{player1} (Rank {int(p1_rank)})"
                    favorite_options[1] = f"{player2} (Rank {int(p2_rank)})"
                    favorite_idx = 1

        favorite = st.selectbox("Betting Favorite", favorite_options, index=favorite_idx)

    # Prediction button
    if st.button("🎯 Predict Match", type="primary", use_container_width=True):
        if player1 == player2:
            st.error("Please select two different players!")
            return

        with st.spinner("Analyzing match..."):
            # Extract favorite name
            if "Unknown" in favorite:
                fav_name = None
            else:
                fav_name = favorite.split(" (")[0]

            # Make prediction
            prediction = model.predict_match(
                player1, player2, 
                surface=surface, 
                favorite=fav_name,
                tourney_level=tournament_level
            )

            if prediction is None:
                st.error("Could not make prediction. One or both players may not be in our database.")
                return

        # Display results
        st.success("Prediction Complete!")

        # Main prediction
        col_result1, col_result2 = st.columns(2)

        with col_result1:
            st.metric("Predicted Winner", prediction.predicted_winner)
            st.metric("Confidence", f"{prediction.confidence:.1%}")

        with col_result2:
            st.metric("Player 1 Win Probability", f"{prediction.player1_win_prob:.1%}")
            st.metric("Player 2 Win Probability", f"{prediction.player2_win_prob:.1%}")

        # Upset analysis
        if prediction.upset_probability > 0:
            st.subheader("⚠️ Upset Analysis")

            upset_level = "High" if prediction.upset_probability > 0.4 else "Medium" if prediction.upset_probability > 0.3 else "Low"

            col_upset1, col_upset2, col_upset3 = st.columns(3)

            with col_upset1:
                st.metric("Upset Probability", f"{prediction.upset_probability:.1%}")

            with col_upset2:
                st.metric("Upset Risk Level", upset_level)

            with col_upset3:
                if fav_name:
                    st.metric("Underdog", prediction.underdog if prediction.underdog != fav_name else prediction.favorite)
                else:
                    st.metric("Note", "No favorite set")

            if prediction.upset_explanation:
                st.info(f"💡 {prediction.upset_explanation}")

        # Key factors
        if prediction.key_factors:
            st.subheader("🔑 Key Factors")
            factors_text = "\n".join(f"• {factor}" for factor in prediction.key_factors[:5])  # Top 5
            st.markdown(factors_text)

        # Style matchup
        st.subheader("🎭 Style Matchup")
        st.info(f"**{prediction.style_matchup}**")

        # Player comparison
        st.subheader("📊 Player Comparison")

        # Get player profiles for comparison
        p1_data = profiles[profiles['player_name'] == player1].iloc[0] if not profiles[profiles['player_name'] == player1].empty else None
        p2_data = profiles[profiles['player_name'] == player2].iloc[0] if not profiles[profiles['player_name'] == player2].empty else None

        if p1_data is not None and p2_data is not None:
            # Create comparison table
            comparison_data = {
                "Metric": ["Win %", "Ace %", "DF %", "Recent Form", "Style"],
                player1: [
                    f"{p1_data.get('overall_win_pct', 0):.1%}",
                    f"{p1_data.get('ace_pct', 0):.1f}",
                    f"{p1_data.get('df_pct', 0):.1f}",
                    f"{p1_data.get('recent_form', 0):.2f}",
                    p1_data.get('style_cluster', 'Unknown')
                ],
                player2: [
                    f"{p2_data.get('overall_win_pct', 0):.1%}",
                    f"{p2_data.get('ace_pct', 0):.1f}",
                    f"{p2_data.get('df_pct', 0):.1f}",
                    f"{p2_data.get('recent_form', 0):.2f}",
                    p2_data.get('style_cluster', 'Unknown')
                ]
            }

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # Save to prediction history
        prediction_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'player1': player1,
            'player2': player2,
            'surface': surface,
            'tournament_level': tournament_level,
            'favorite': fav_name,
            'predicted_winner': prediction.predicted_winner,
            'confidence': prediction.confidence,
            'p1_win_prob': prediction.player1_win_prob,
            'p2_win_prob': prediction.player2_win_prob,
            'upset_probability': prediction.upset_probability,
            'style_matchup': prediction.style_matchup
        }
        
        st.session_state.prediction_history.append(prediction_record)
        
        # Show history summary
        st.success(f"✅ Prediction saved! Total predictions made: {len(st.session_state.prediction_history)}")

        # Disclaimer
        st.caption("⚠️ **Disclaimer**: These predictions are for entertainment purposes only. Tennis matches can be unpredictable, and many factors (weather, injuries, form) can affect outcomes.")

    # Prediction History
    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📚 Prediction History")
        
        # Convert to dataframe for display
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Show recent predictions (last 10)
        recent_history = history_df.tail(10).copy()
        recent_history['confidence'] = recent_history['confidence'].apply(lambda x: f"{x:.1%}")
        recent_history['p1_win_prob'] = recent_history['p1_win_prob'].apply(lambda x: f"{x:.1%}")
        recent_history['p2_win_prob'] = recent_history['p2_win_prob'].apply(lambda x: f"{x:.1%}")
        recent_history['upset_probability'] = recent_history['upset_probability'].apply(lambda x: f"{x:.1%}")
        
        # Rename columns for display
        display_cols = {
            'timestamp': 'Time',
            'player1': 'Player 1', 
            'player2': 'Player 2',
            'surface': 'Surface',
            'predicted_winner': 'Predicted Winner',
            'confidence': 'Confidence',
            'upset_probability': 'Upset Risk'
        }
        
        st.dataframe(
            recent_history[list(display_cols.keys())].rename(columns=display_cols),
            use_container_width=True,
            hide_index=True
        )
        
        if len(st.session_state.prediction_history) > 10:
            st.caption(f"Showing last 10 of {len(st.session_state.prediction_history)} predictions")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Title
    st.title("🎾 Tennis Upset Predictor")
    
    # Quick overview
    st.markdown("""
    **AI-powered tennis match predictions with upset detection.** 
    Analyze upcoming matches, explore player profiles, and predict custom matchups.
    """)
    
    col_overview1, col_overview2, col_overview3, col_overview4 = st.columns(4)
    
    with col_overview1:
        st.metric("**Data Source**", "TML Database")
    with col_overview2:
        st.metric("**Model Type**", "Random Forest")
    with col_overview3:
        st.metric("**Key Feature**", "Upset Detection")
    with col_overview4:
        st.metric("**Predictions**", "56% Accurate")
    
    st.markdown("---")
    
    # Load data with progress
    with st.spinner("Loading data and training model... (first load takes ~30 seconds)"):
        try:
            matches = load_all_data()
            profiler, profiles = build_profiler_and_profiles(matches)
            model, train_result = get_trained_model(matches, profiles, profiler)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Try running: `python src/data_loader.py` first to download data")
            return
    
    # Data summary in sidebar
    st.sidebar.title("🎾 Navigation")
    st.sidebar.markdown("---")
    
    # Navigation tabs
    page = st.sidebar.radio(
        "Go to",
        ["📅 Match Calendar", "🎯 Upset Alerts", "👤 Player Profiles", 
         "⚔️ Head-to-Head", "🏆 Leaderboards", "🗓️ Tournament Calendar", "🔮 Custom Predictor", "📊 Historical Accuracy"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Summary")
    st.sidebar.write(f"**Historical Matches:** {len(matches):,}")
    st.sidebar.write(f"**Player Profiles:** {len(profiles):,}")
    st.sidebar.write(f"**Includes:** ATP + Challengers + Futures")
    # Show actual years from data
    if 'tourney_date' in matches.columns:
        min_year = int(str(matches['tourney_date'].min())[:4])
        max_year = int(str(matches['tourney_date'].max())[:4])
        st.sidebar.write(f"**Years:** {min_year}-{max_year} (rolling window)")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Model Performance")
    st.sidebar.write(f"**Accuracy:** {train_result['accuracy']:.1%}")
    st.sidebar.write(f"**CV Score:** {train_result['cv_mean']:.1%} ± {train_result['cv_std']:.1%}")
    st.sidebar.caption("Features: H2H, Fatigue, Form, Style")
    
    # Feature importance chart
    if 'feature_importance' in train_result and train_result['feature_importance'] is not None:
        st.sidebar.markdown("**Top Features:**")
        top_features = train_result['feature_importance'].head(5)
        
        for _, row in top_features.iterrows():
            feature_name = row['feature'][:20] + "..." if len(row['feature']) > 20 else row['feature']
            bar = "█" * int(row['importance'] * 20)  # Scale to 20 chars
            st.sidebar.caption(f"{feature_name}: {bar}")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Data: TML Database (TennisMyLife)")
    
    # Render selected page
    if "Match Calendar" in page:
        page_calendar(model, profiles)
    elif "Upset Alerts" in page:
        page_upsets(model, profiles)
    elif "Player Profiles" in page:
        page_players(profiles, profiler)
    elif "Head-to-Head" in page:
        page_h2h(model, profiles, profiler)
    elif "Leaderboards" in page:
        page_leaderboards(profiles)
    elif "Tournament Calendar" in page:
        page_calendar_tournaments()
    elif "Custom Predictor" in page:
        page_custom_predictor(model, profiles)
    elif "Historical Accuracy" in page:
        page_historical_accuracy()


if __name__ == "__main__":
    main()
