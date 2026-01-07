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
import sys
from zoneinfo import ZoneInfo

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
import re


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


# Common timezones for user selection
TIMEZONE_OPTIONS = {
    "US/Eastern": "🇺🇸 US Eastern (ET)",
    "US/Central": "🇺🇸 US Central (CT)",
    "US/Mountain": "🇺🇸 US Mountain (MT)",
    "US/Pacific": "🇺🇸 US Pacific (PT)",
    "Europe/London": "🇬🇧 UK (GMT/BST)",
    "Europe/Paris": "🇪🇺 Central Europe (CET)",
    "Europe/Berlin": "🇩🇪 Germany (CET)",
    "Asia/Tokyo": "🇯🇵 Japan (JST)",
    "Asia/Shanghai": "🇨🇳 China (CST)",
    "Asia/Kolkata": "🇮🇳 India (IST)",
    "Australia/Sydney": "🇦🇺 Australia (AEST)",
    "Australia/Melbourne": "🇦🇺 Melbourne (AEST)",
    "Pacific/Auckland": "🇳🇿 New Zealand (NZST)",
}


def get_user_now() -> datetime:
    """Get current datetime in user's selected timezone.
    
    FIX: Streamlit Cloud runs on UTC servers. This function uses the user's
    selected timezone to determine 'today', 'tomorrow', etc. correctly.
    """
    tz_name = st.session_state.get('user_timezone', 'US/Central')
    user_tz = ZoneInfo(tz_name)
    return datetime.now(user_tz)


def get_user_timezone_name() -> str:
    """Get the display name of user's selected timezone."""
    tz_name = st.session_state.get('user_timezone', 'US/Central')
    user_tz = ZoneInfo(tz_name)
    now = datetime.now(user_tz)
    return now.strftime("%Z")


def get_cet_now() -> datetime:
    """Get current datetime in CET (TennisExplorer's timezone).
    
    FIX: The scraper stores match dates in server timezone (UTC on Streamlit Cloud).
    We need to compare with CET timezone since TennisExplorer uses CET for match dates.
    """
    cet_tz = ZoneInfo("Europe/Paris")
    return datetime.now(cet_tz)


def convert_cet_to_user_tz(time_str: str, match_date: str = None) -> str:
    """Convert match time from CET (TennisExplorer timezone) to user's selected timezone.
    
    Args:
        time_str: Time string like "14:30" in CET timezone
        match_date: Optional date string "YYYY-MM-DD" for accurate DST handling
        
    Returns:
        Converted time string in user's timezone (e.g., "08:30" for US/Central)
    """
    if not time_str or time_str in ('TBD', 'LIVE', 'FIN', ''):
        return time_str
    
    # Extract just the time portion (HH:MM)
    match = re.match(r'^(\d{1,2}):(\d{2})', str(time_str))
    if not match:
        return time_str
    
    hour, minute = int(match.group(1)), int(match.group(2))
    
    # Use match date if provided, otherwise use CET today
    if match_date:
        try:
            base_date = datetime.strptime(match_date, "%Y-%m-%d")
        except ValueError:
            base_date = get_cet_now()
    else:
        base_date = get_cet_now()
    
    # Create CET datetime
    cet_tz = ZoneInfo("Europe/Paris")
    cet_dt = datetime(base_date.year, base_date.month, base_date.day, hour, minute, tzinfo=cet_tz)
    
    # Convert to user's timezone
    user_tz_name = st.session_state.get('user_timezone', 'US/Central')
    user_tz = ZoneInfo(user_tz_name)
    user_dt = cet_dt.astimezone(user_tz)
    
    return user_dt.strftime("%H:%M")


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

def render_timezone_selector():
    """Render timezone selector prominently at top of page."""
    # Initialize timezone in session state if not set
    if 'user_timezone' not in st.session_state:
        st.session_state['user_timezone'] = 'US/Central'
    
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        current_tz = st.session_state.get('user_timezone', 'US/Central')
        tz_keys = list(TIMEZONE_OPTIONS.keys())
        current_index = tz_keys.index(current_tz) if current_tz in tz_keys else 1
        
        selected_tz = st.selectbox(
            "🕐 Your Timezone",
            options=tz_keys,
            format_func=lambda x: TIMEZONE_OPTIONS[x],
            index=current_index,
            help="Select your timezone - Today/Tomorrow labels will adjust accordingly"
        )
        st.session_state['user_timezone'] = selected_tz
    
    with col2:
        user_now = get_user_now()
        st.metric(
            label="Your Local Time",
            value=user_now.strftime("%I:%M %p"),
            delta=user_now.strftime("%a, %b %d")
        )
    
    with col3:
        cet_now = get_cet_now()
        st.metric(
            label="Tournament Time (CET)",
            value=cet_now.strftime("%I:%M %p"),
            delta=cet_now.strftime("%a, %b %d")
        )
    
    st.divider()


def page_calendar(model, profiles):
    """Match Calendar page."""
    st.header("📅 Match Calendar & Predictions")
    
    # Timezone selector at top of page
    render_timezone_selector()
    
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
        st.info(f"📡 Scraped **{len(upcoming)} matches** from TennisExplorer (auto-refreshes every 30 min)")
    
    # FIX: Use user's timezone for Today/Tomorrow labels
    # Match dates from scraper are in CET, but we want labels based on user's local date
    # This ensures a US user sees "Today" for matches on their current date
    user_now = get_user_now()
    today_str = user_now.strftime("%Y-%m-%d")
    yesterday_str = (user_now - timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow_str = (user_now + timedelta(days=1)).strftime("%Y-%m-%d")
    
    def assign_label(d):
        if d == today_str:
            return 'Today'
        elif d == yesterday_str:
            return 'Yesterday'
        elif d == tomorrow_str:
            return 'Tomorrow'
        return 'Other'
    
    # Always recalculate date_label based on user's timezone
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
    ]
    
    # Finished: Today + Yesterday completed matches
    finished_matches = upcoming[
        (upcoming['_status'] == 'finished') & 
        (upcoming['date_label'].isin(['Today', 'Yesterday']))
    ]
    
    # Filters section - visible before tabs
    st.subheader("🔍 Filters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        surfaces = st.multiselect("Surface", ["Hard", "Clay", "Grass"], default=["Hard", "Clay", "Grass"])
    with col2:
        tournaments = sorted(upcoming['tournament'].unique().tolist())
        selected_tournament = st.selectbox("Tournament", ["All"] + tournaments)
    with col3:
        show_main_only = st.checkbox("🏆 Main events only", value=True, 
            help="Filter out UTR/Futures/ITF events (recommended for better predictions)")
    with col4:
        show_upsets_only = st.checkbox("⚠️ Upset alerts only", value=False)
    
    st.markdown("---")
    
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
                raw_time = clean_time_display(match.get('time', 'TBD'))
                match_date = match.get('date', '')
                match_time = convert_cet_to_user_tz(raw_time, match_date)
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
    
    # Date strings for comparison (using CET - TennisExplorer timezone)
    # FIX: Match dates are stored in CET, so compare with CET dates
    cet_now = get_cet_now()
    today_str = cet_now.strftime("%Y-%m-%d")
    tomorrow_str = (cet_now + timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_str = (cet_now - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Main tabs: Upcoming and Finished
    tab_upcoming, tab_finished = st.tabs(["⏰ Upcoming", "✅ Finished"])
    
    # UPCOMING TAB with Today/Tomorrow sub-tabs
    with tab_upcoming:
        filtered_upcoming = apply_filters(upcoming_matches)
        
        # Split by actual local date from datetime_local
        filtered_upcoming['_local_date'] = filtered_upcoming['datetime_local'].apply(get_local_date)
        today_upcoming = filtered_upcoming[filtered_upcoming['_local_date'] == today_str]
        tomorrow_upcoming = filtered_upcoming[filtered_upcoming['_local_date'] == tomorrow_str]
        # Include matches without datetime_local in "Today"
        no_date = filtered_upcoming[filtered_upcoming['_local_date'].isna()]
        today_upcoming = pd.concat([today_upcoming, no_date])
        
        st.write(f"**{len(filtered_upcoming)}** upcoming matches total")
        
        sub_today, sub_tomorrow = st.tabs([f"📅 Today ({today_str}) - {len(today_upcoming)}", f"📅 Tomorrow ({tomorrow_str}) - {len(tomorrow_upcoming)}"])
        
        with sub_today:
            display_day_matches(today_upcoming, model, show_upsets_only, sort_ascending=True)
        
        with sub_tomorrow:
            display_day_matches(tomorrow_upcoming, model, show_upsets_only, sort_ascending=True)
    
    # FINISHED TAB with Today/Yesterday sub-tabs
    with tab_finished:
        filtered_finished = apply_filters(finished_matches)
        
        # Split by actual local date from datetime_local
        filtered_finished['_local_date'] = filtered_finished['datetime_local'].apply(get_local_date)
        today_finished = filtered_finished[filtered_finished['_local_date'] == today_str]
        yesterday_finished = filtered_finished[filtered_finished['_local_date'] == yesterday_str]
        # Include matches without datetime_local in "Today"
        no_date_fin = filtered_finished[filtered_finished['_local_date'].isna()]
        today_finished = pd.concat([today_finished, no_date_fin])
        
        st.write(f"**{len(filtered_finished)}** finished matches total")
        
        sub_today_fin, sub_yesterday_fin = st.tabs([f"📅 Today ({today_str}) - {len(today_finished)}", f"📅 Yesterday ({yesterday_str}) - {len(yesterday_finished)}"])
        
        with sub_today_fin:
            display_day_matches(today_finished, model, show_upsets_only, sort_ascending=False)
        
        with sub_yesterday_fin:
            display_day_matches(yesterday_finished, model, show_upsets_only, sort_ascending=False)


def page_upsets(model, profiles):
    """Upset Alerts page."""
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
            raw_time = clean_time_display(match.get('time', 'TBD'))
            match_date = match.get('date', '')
            match_time = convert_cet_to_user_tz(raw_time, match_date)
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
    
    # Date strings for comparison (using CET - TennisExplorer timezone)
    # FIX: Match dates are stored in CET, so compare with CET dates
    cet_now = get_cet_now()
    today_str = cet_now.strftime("%Y-%m-%d")
    tomorrow_str = (cet_now + timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_str = (cet_now - timedelta(days=1)).strftime("%Y-%m-%d")
    
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


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Title
    st.title("🎾 Tennis Upset Predictor")
    
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
    
    # Initialize timezone in session state if not set
    if 'user_timezone' not in st.session_state:
        st.session_state['user_timezone'] = 'US/Central'
    
    # Show current time in user's timezone (selector is on main page)
    user_now = get_user_now()
    st.sidebar.caption(f"📍 {user_now.strftime('%a, %b %d • %I:%M %p')} ({get_user_timezone_name()})")
    
    st.sidebar.markdown("---")
    
    # Navigation tabs
    page = st.sidebar.radio(
        "Go to",
        ["📅 Match Calendar", "🎯 Upset Alerts", "👤 Player Profiles", 
         "⚔️ Head-to-Head", "🏆 Leaderboards", "🗓️ Tournament Calendar"],
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


if __name__ == "__main__":
    main()
