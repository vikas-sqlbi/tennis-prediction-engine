"""
Scraper Module
Scrapes real upcoming tennis fixtures from TennisExplorer.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import re
import logging

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TennisExplorer uses CET/CEST timezone
TE_TIMEZONE = ZoneInfo("Europe/Paris")


class PlayerMatcher:
    """
    Dynamic player name matcher that builds mappings from TML Database.
    Uses fuzzy matching for unknown players.
    """
    _instance = None
    _profiles_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._dynamic_map = {}
            cls._instance._lastname_map = {}
            cls._instance._all_players = []
        return cls._instance
    
    def load_from_profiles(self, profiles_df: 'pd.DataFrame') -> None:
        """Build dynamic mappings from TML player profiles."""
        if self._profiles_loaded:
            return
            
        if profiles_df is None or 'player_name' not in profiles_df.columns:
            return
        
        for _, row in profiles_df.iterrows():
            full_name = str(row.get('player_name', '')).strip()
            if not full_name or len(full_name) < 3:
                continue
                
            self._all_players.append(full_name)
            name_lower = full_name.lower()
            
            # Map full name
            self._dynamic_map[name_lower] = full_name
            
            # Map last name (last word)
            parts = full_name.split()
            if parts:
                last_name = parts[-1].lower()
                # Only add if unique or this player has more matches
                if last_name not in self._lastname_map:
                    self._lastname_map[last_name] = full_name
                    
            # Map hyphenated variations
            if '-' in full_name:
                self._dynamic_map[name_lower.replace('-', ' ')] = full_name
            if ' ' in full_name:
                # Handle multi-word last names
                self._dynamic_map[name_lower.replace(' ', '-')] = full_name
        
        self._profiles_loaded = True
        logger.info(f"PlayerMatcher loaded {len(self._all_players)} players from profiles")
    
    def match(self, scraped_name: str) -> Optional[str]:
        """
        Match a scraped player name to a full name from TML Database.
        
        Args:
            scraped_name: The name from TennisExplorer (may be abbreviated)
            
        Returns:
            Full player name from TML, or None if no match found
        """
        if not scraped_name or len(scraped_name) < 2:
            return None
            
        name_clean = scraped_name.strip()
        name_lower = name_clean.lower()
        
        # 1. Check static PLAYER_NAME_MAP first
        if name_lower in PLAYER_NAME_MAP:
            return PLAYER_NAME_MAP[name_lower]
        
        # 2. Check dynamic full name map
        if name_lower in self._dynamic_map:
            return self._dynamic_map[name_lower]
        
        # 3. Handle abbreviated formats: "J. Lastname" or "Lastname J."
        # Format: "J. Sinner"
        match = re.match(r'^([A-Z])\.\s*(.+)$', name_clean)
        if match:
            initial, last_name = match.groups()
            last_lower = last_name.lower().strip()
            
            # Check static map
            if last_lower in PLAYER_NAME_MAP:
                return PLAYER_NAME_MAP[last_lower]
            
            # Check dynamic lastname map
            if last_lower in self._lastname_map:
                candidate = self._lastname_map[last_lower]
                # Verify initial matches
                if candidate and candidate[0].upper() == initial:
                    return candidate
            
            # Search all players for matching last name + initial
            for player in self._all_players:
                if player.lower().endswith(last_lower) and player[0].upper() == initial:
                    return player
        
        # Format: "Sinner J."
        match = re.match(r'^(.+)\s+([A-Z])\.$', name_clean)
        if match:
            last_name, initial = match.groups()
            last_lower = last_name.lower().strip()
            
            if last_lower in PLAYER_NAME_MAP:
                return PLAYER_NAME_MAP[last_lower]
            
            if last_lower in self._lastname_map:
                candidate = self._lastname_map[last_lower]
                if candidate and candidate[0].upper() == initial:
                    return candidate
        
        # 4. Try matching just the last name (last word with 3+ chars)
        parts = name_clean.split()
        if parts:
            last_word = parts[-1].lower() if len(parts[-1]) > 2 else None
            if last_word:
                if last_word in PLAYER_NAME_MAP:
                    return PLAYER_NAME_MAP[last_word]
                if last_word in self._lastname_map:
                    return self._lastname_map[last_word]
        
        # 5. Fuzzy substring match against all players
        for player in self._all_players:
            player_lower = player.lower()
            # Check if scraped name is contained in full name or vice versa
            if name_lower in player_lower or player_lower in name_lower:
                return player
            # Check if last name matches
            player_last = player.split()[-1].lower() if player.split() else ''
            if player_last and player_last == name_lower:
                return player
        
        return None
    
    def get_or_create_mapping(self, scraped_name: str, href: str = None) -> str:
        """
        Get the mapped name or return the cleaned scraped name as-is.
        This ensures we always return something usable.
        """
        matched = self.match(scraped_name)
        if matched:
            return matched
        
        # If no match, try to construct a reasonable name from the input
        # Clean up the scraped name
        clean = re.sub(r'\(\d+\)', '', scraped_name)  # Remove ranking
        clean = re.sub(r'\[\d+\]', '', clean)  # Remove seed
        clean = re.sub(r'\s+\([A-Z]{3}\)', '', clean)  # Remove country code
        clean = ' '.join(clean.split()).strip()
        
        # If still abbreviated, try to expand from URL slug
        if href:
            slug_match = re.search(r'/player/([^/]+?)/?$', href)
            if slug_match:
                slug = slug_match.group(1)
                # Remove hash suffix
                slug = re.sub(r'-[a-f0-9]{5,8}$', '', slug)
                # Convert slug to name (hyphens to spaces, title case)
                expanded = slug.replace('-', ' ').title()
                if len(expanded) > len(clean):
                    return expanded
        
        return clean


# Global player matcher instance
_player_matcher = PlayerMatcher()


def convert_te_time_to_local(date_str: str, time_str: str) -> tuple[datetime, str, str]:
    """
    Convert TennisExplorer time (CET) to local timezone.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        time_str: Time in HH:MM format (CET timezone)
    
    Returns:
        Tuple of (local_datetime, local_date_str, local_time_str)
    """
    try:
        # Parse as CET time
        dt_cet = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        dt_cet = dt_cet.replace(tzinfo=TE_TIMEZONE)
        
        # Convert to local timezone
        dt_local = dt_cet.astimezone()
        
        local_date_str = dt_local.strftime("%Y-%m-%d")
        local_time_str = dt_local.strftime("%H:%M")
        
        return dt_local, local_date_str, local_time_str
    except (ValueError, AttributeError):
        return None, date_str, time_str


def is_match_in_past(date_str: str, time_str: str) -> bool:
    """Check if a match time has already passed (in local timezone)."""
    try:
        dt_cet = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        dt_cet = dt_cet.replace(tzinfo=TE_TIMEZONE)
        dt_local = dt_cet.astimezone()
        
        # Add 3 hour buffer for match duration
        return dt_local < datetime.now().astimezone() - timedelta(hours=3)
    except (ValueError, AttributeError):
        return False

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
}

def parse_odds(odds_text: str) -> Optional[float]:
    """Parse odds text to float. Returns None if invalid."""
    if not odds_text:
        return None
    try:
        cleaned = odds_text.strip().replace(',', '.')
        if cleaned and cleaned not in ['', '-', 'No odds']:
            return float(cleaned)
    except (ValueError, AttributeError):
        pass
    return None


def clean_time_field(time_text: str) -> str:
    """
    Clean time field from TennisExplorer which often has junk appended.
    Examples:
        "03:00Live streams1xBetUnibetbet365" -> "03:00"
        "14:30Livestreams" -> "14:30"
        "LIVE" -> "LIVE"
    """
    if not time_text:
        return "TBD"
    
    # Check for LIVE indicator
    if 'live' in time_text.lower() and not re.match(r'^\d{1,2}:\d{2}', time_text):
        return "LIVE"
    
    # Extract time pattern HH:MM from the beginning
    match = re.match(r'^(\d{1,2}:\d{2})', time_text)
    if match:
        return match.group(1)
    
    return time_text.strip()


def determine_favorite_from_odds(player1: str, player2: str, odds1: Optional[float], odds2: Optional[float]) -> Optional[str]:
    """Determine favorite based on odds. Lower odds = favorite."""
    if odds1 is None or odds2 is None:
        return None
    if odds1 < odds2:
        return player1
    elif odds2 < odds1:
        return player2
    return None  # Equal odds, no favorite


def is_wta_tournament(tournament_name: str) -> bool:
    """
    Check if a tournament is WTA (women's tour).
    Returns True for WTA events, False for ATP/men's events.
    """
    if not tournament_name:
        return False
    
    t = tournament_name.lower()
    
    # Explicit WTA indicators
    wta_indicators = [
        'wta', 'women', 'ladies', 'w-',
        # Women's-only tournaments
        'wuhan', 'zhengzhou', 'nanchang', 'jiujiang',
        'portoroz', 'rabat', 'bogota', 'lausanne',
    ]
    
    # ATP-only indicators (these are definitely men's)
    atp_indicators = [
        'atp', 'challenger', 'davis cup', 'united cup',
        'laver cup', 'next gen', 'hopman',
    ]
    
    # Check ATP first - if it's ATP, it's not WTA
    for indicator in atp_indicators:
        if indicator in t:
            return False
    
    # Check WTA indicators
    for indicator in wta_indicators:
        if indicator in t:
            return True
    
    # Grand Slams and Masters are mixed but we'll keep them (we can filter by player later)
    # Default: assume ATP/mixed (keep the match)
    return False

# ATP player name mappings (last name / slug -> full name)
# This helps match scraped names from TennisExplorer to TML Database historical data
# WTA players removed since we only have ATP historical data
PLAYER_NAME_MAP = {
    # ATP Top 20
    'sinner': 'Jannik Sinner',
    'alcaraz': 'Carlos Alcaraz',
    'djokovic': 'Novak Djokovic',
    'medvedev': 'Daniil Medvedev',
    'zverev': 'Alexander Zverev',
    'rublev': 'Andrey Rublev',
    'ruud': 'Casper Ruud',
    'tsitsipas': 'Stefanos Tsitsipas',
    'fritz': 'Taylor Fritz',
    'de minaur': 'Alex De Minaur',
    'hurkacz': 'Hubert Hurkacz',
    'dimitrov': 'Grigor Dimitrov',
    'shelton': 'Ben Shelton',
    'tiafoe': 'Frances Tiafoe',
    'paul': 'Tommy Paul',
    'rune': 'Holger Rune',
    'khachanov': 'Karen Khachanov',
    'musetti': 'Lorenzo Musetti',
    'draper': 'Jack Draper',
    'auger-aliassime': 'Felix Auger Aliassime',
    'berrettini': 'Matteo Berrettini',
    'bublik': 'Alexander Bublik',
    'baez': 'Sebastian Baez',
    'cerundolo': 'Francisco Cerundolo',
    'humbert': 'Ugo Humbert',
    'jarry': 'Nicolas Jarry',
    'fils': 'Arthur Fils',
    'lehecka': 'Jiri Lehecka',
    'arnaldi': 'Matteo Arnaldi',
    'mensik': 'Jakub Mensik',
    'tien': 'Learner Tien',
    'basavareddy': 'Nishesh Basavareddy',
    'blockx': 'Alexander Blockx',
    'budkov kjaer': 'Nicolai Budkov Kjaer',
    'fonseca': 'Joao Fonseca',
    'nagal': 'Sumit Nagal',
    'nadal': 'Rafael Nadal',
    'federer': 'Roger Federer',
    'murray': 'Andy Murray',
    'wawrinka': 'Stan Wawrinka',
    'thiem': 'Dominic Thiem',
    'nishikori': 'Kei Nishikori',
    'kyrgios': 'Nick Kyrgios',
    'monfils': 'Gael Monfils',
    'bautista agut': 'Roberto Bautista Agut',
    'carreno busta': 'Pablo Carreno Busta',
    'davidovich fokina': 'Alejandro Davidovich Fokina',
    'korda': 'Sebastian Korda',
    'norrie': 'Cameron Norrie',
    'evans': 'Daniel Evans',
    'brooksby': 'Jenson Brooksby',
    'isner': 'John Isner',
    'opelka': 'Reilly Opelka',
    'sock': 'Jack Sock',
    'nakashima': 'Brandon Nakashima',
    'wolf': 'Jeffrey Wolf',
    'giron': 'Marcos Giron',
    'eubanks': 'Christopher Eubanks',
    'kecmanovic': 'Miomir Kecmanovic',
    'djere': 'Laslo Djere',
    'lajovic': 'Dusan Lajovic',
    'krajinovic': 'Filip Krajinovic',
    'sonego': 'Lorenzo Sonego',
    'cobolli': 'Flavio Cobolli',
    'darderi': 'Luciano Darderi',
    'bergs': 'Zizou Bergs',
    'mpetshi perricard': 'Giovanni Mpetshi Perricard',
    'struff': 'Jan Lennard Struff',
    'griekspoor': 'Tallon Griekspoor',
    'van de zandschulp': 'Botic Van De Zandschulp',
    'tabilo': 'Alejandro Tabilo',
    'etcheverry': 'Tomas Martin Etcheverry',
    'machac': 'Tomas Machac',
    'safiullin': 'Roman Safiullin',
    'shang': 'Juncheng Shang',
    'popyrin': 'Alexei Popyrin',
    'thompson': 'Jordan Thompson',
    'kokkinakis': 'Thanasi Kokkinakis',
    'o connell': 'Christopher O Connell',
    'hijikata': 'Rinky Hijikata',
    # Additional ATP players (ranked ~30-100+)
    'mannarino': 'Adrian Mannarino',
    'muller': 'Alexandre Muller',
    'altmaier': 'Daniel Altmaier',
    'borges': 'Nuno Borges',
    'moutet': 'Corentin Moutet',
    'martinez': 'Pedro Martinez',
    'coria': 'Federico Coria',
    'monteiro': 'Thiago Monteiro',
    'diaz acosta': 'Facundo Diaz Acosta',
    'ofner': 'Sebastian Ofner',
    'vukic': 'Aleksandar Vukic',
    'marozsan': 'Fabian Marozsan',
    'nardi': 'Luca Nardi',
    'cazaux': 'Arthur Cazaux',
    'mpetshi': 'Giovanni Mpetshi Perricard',
    'perricard': 'Giovanni Mpetshi Perricard',
    'cressy': 'Maxime Cressy',
    'munar': 'Jaume Munar',
    'ramos-vinolas': 'Albert Ramos Vinolas',
    'ramos vinolas': 'Albert Ramos Vinolas',
    'fucsovics': 'Marton Fucsovics',
    'gasquet': 'Richard Gasquet',
    'bonzi': 'Benjamin Bonzi',
    'rinderknech': 'Arthur Rinderknech',
    'pouille': 'Lucas Pouille',
    'gaston': 'Hugo Gaston',
    'halys': 'Quentin Halys',
    'kotov': 'Pavel Kotov',
    'shevchenko': 'Alexander Shevchenko',
    'karatsev': 'Aslan Karatsev',
    'seyboth wild': 'Thiago Seyboth Wild',
    'hanfmann': 'Yannick Hanfmann',
    'marterer': 'Maximilian Marterer',
    'garin': 'Cristian Garin',
    'zapata miralles': 'Bernabe Zapata Miralles',
    'carballes baena': 'Roberto Carballes Baena',
    'navone': 'Mariano Navone',
    'comesana': 'Francisco Comesana',
    'passaro': 'Francesco Passaro',
    'bellucci': 'Mattia Bellucci',
    'diallo': 'Gabriel Diallo',
    'michelsen': 'Alex Michelsen',
    'schoolkate': 'Tristan Schoolkate',
    'landaluce': 'Martin Landaluce',
    'blanch': 'Darwin Blanch',
    # ATP players ranked ~100-200+
    'coric': 'Borna Coric',
    'daniel': 'Taro Daniel',
    'faria': 'Jaime Faria',
    'habib': 'Hady Habib',
    'koepfer': 'Dominik Koepfer',
    'lestienne': 'Constant Lestienne',
    'medjedovic': 'Hamad Medjedovic',
    'moro canas': 'Alejandro Moro Canas',
    'nagal': 'Sumit Nagal',
    'o connell': 'Christopher O Connell',
    'oconnell': 'Christopher O Connell',
    'otte': 'Oscar Otte',
    'popko': 'Dmitry Popko',
    'purcell': 'Max Purcell',
    'stricker': 'Dominic Stricker',
    'tu': 'Li Tu',
    'virtanen': 'Otto Virtanen',
    'walton': 'Adam Walton',
    'wu': 'Yibing Wu',
    'zhang': 'Zhizhen Zhang',
    'zhizhen zhang': 'Zhizhen Zhang',
    # Young rising stars
    'bu': 'Yunchaokete Bu',
    'fonseca': 'Joao Fonseca',
    'mensik': 'Jakub Mensik',
    'tien': 'Learner Tien',
    'basavareddy': 'Nishesh Basavareddy',
    'blockx': 'Alexander Blockx',
    'budkov kjaer': 'Nicolai Budkov Kjaer',
}


def scrape_tennis_explorer(include_tomorrow: bool = True, include_yesterday: bool = True) -> pd.DataFrame:
    """
    Scrape upcoming matches from TennisExplorer.
    Extracts match times, detects live matches, and fetches scores.
    
    Scrapes multiple category pages to get all matches (ATP, WTA, Challenger, ITF).
    Also fetches results/scores for completed matches.
    
    Args:
        include_tomorrow: If True, also scrape tomorrow's matches
        include_yesterday: If True, also scrape yesterday's completed matches
    """
    seen_matches = {}  # Track unique matches: key -> match dict
    
    # Category URLs to scrape - ATP ONLY (no WTA since we don't have historical data)
    category_types = [
        None,  # Default page (mixed - will filter later)
        'atp-single',
        'ch-single',  # Challengers (men's)
    ]
    
    def make_key(match):
        """Create deduplication key using player pair only.
        
        Players don't play twice in the same tournament round, so player pair
        is sufficient for deduplication. This prevents the same match from
        appearing as duplicates when fetched from multiple day pages.
        """
        p1 = match['player1'].lower().strip()
        p2 = match['player2'].lower().strip()
        # Sort players to handle case where player order might differ
        return tuple(sorted([p1, p2]))
    
    # Scrape today first (all categories)
    # IMPORTANT: TennisExplorer uses CET timezone for its date-based URLs
    # We must use CET dates consistently throughout to avoid timezone mismatch issues
    # FIX: Use CET dates for match 'date' field, not server local time (which is UTC on Streamlit Cloud)
    now_cet = datetime.now(TE_TIMEZONE)
    today = now_cet
    today_str = now_cet.strftime("%Y-%m-%d")  # CET date string for TennisExplorer URLs
    today_cet_str = today_str  # Use CET date for match dates (not server local time)
    yesterday_cet_str = (now_cet - timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow_cet_str = (now_cet + timedelta(days=1)).strftime("%Y-%m-%d")
    
    logger.info(f"Scraping with CET date: {today_str} (all dates stored in CET)")
    
    for cat_type in category_types:
        today_matches = _scrape_tennis_explorer_date(today, category_type=cat_type)
        for match in today_matches:
            # All matches from CET "today" page are considered "Today" for our purposes
            # The user wants to see what's happening now/soon, not strict local date matching
            match['source_day'] = 'Today'
            match['date'] = today_cet_str
            
            key = make_key(match)
            if key not in seen_matches:
                seen_matches[key] = match
            elif match.get('is_live'):
                # Prefer live matches over scheduled
                seen_matches[key] = match
            elif match.get('tournament') and not seen_matches[key].get('tournament'):
                # Add tournament name if missing
                seen_matches[key]['tournament'] = match['tournament']
    
    # Fetch today's results/scores from results page
    # FIX: Use CET date consistently (not server local time)
    today_results = _scrape_results_page(today)
    for match in today_results:
        # Convert the CET datetime to local datetime for sorting
        match_cet_date = match.get('date', '')
        match_time = match.get('time', '')
        
        if match_time and match_time not in ['Finished', 'TBD', '']:
            datetime_local, _, local_time = convert_te_time_to_local(match_cet_date, match_time)
            match['datetime_local'] = datetime_local.isoformat() if datetime_local else None
        
        # Mark as from today's results - use CET date consistently
        match['source_day'] = 'Today'
        match['date'] = today_cet_str  # Use CET date for consistency
        
        key = make_key(match)
        # Always prefer completed results - they have scores
        if key in seen_matches:
            # Update existing match with score
            seen_matches[key]['score'] = match.get('score')
            seen_matches[key]['winner'] = match.get('winner')
            seen_matches[key]['status'] = match.get('status', 'completed')
            seen_matches[key]['source_day'] = 'Today'  # Completed today
            if match.get('datetime_local'):
                seen_matches[key]['datetime_local'] = match['datetime_local']
        else:
            # Add completed match we didn't have
            seen_matches[key] = match
    
    # Scrape tomorrow if requested
    if include_tomorrow:
        tomorrow_cet = now_cet + timedelta(days=1)
        tomorrow_str = tomorrow_cet.strftime("%Y-%m-%d")  # CET date string for URL
        
        for cat_type in category_types:
            tomorrow_matches = _scrape_tennis_explorer_date(tomorrow_cet, category_type=cat_type)
            for match in tomorrow_matches:
                # Matches from tomorrow's page are TOMORROW matches
                # FIX: Use CET date consistently
                match['source_day'] = 'Tomorrow'
                match['date'] = tomorrow_cet_str
                
                key = make_key(match)
                # Only add if we don't already have this match from today's page
                if key not in seen_matches:
                    seen_matches[key] = match
    
    # Scrape yesterday's results if requested - MOST AUTHORITATIVE for completed matches
    if include_yesterday:
        yesterday_cet = now_cet - timedelta(days=1)
        yesterday_str = yesterday_cet.strftime("%Y-%m-%d")  # CET date string for URL
        # Note: yesterday_local_str is already defined at the top
        
        yesterday_results = _scrape_results_page(yesterday_cet)
        for match in yesterday_results:
            # Convert CET datetime to local for sorting
            match_time = match.get('time', '')
            if match_time and match_time not in ['Finished', 'TBD', '']:
                datetime_local, _, _ = convert_te_time_to_local(yesterday_str, match_time)
                match['datetime_local'] = datetime_local.isoformat() if datetime_local else None
            
            # Mark as from yesterday's results - FIX: Use CET date consistently
            match['source_day'] = 'Yesterday'
            match['date'] = yesterday_cet_str
            
            key = make_key(match)
            if key in seen_matches:
                # Same match - update with completed info (more authoritative)
                seen_matches[key]['score'] = match.get('score')
                seen_matches[key]['winner'] = match.get('winner')
                seen_matches[key]['status'] = 'completed'
                seen_matches[key]['date'] = yesterday_cet_str  # Use actual completion date
                seen_matches[key]['source_day'] = 'Yesterday'
                if match.get('datetime_local'):
                    seen_matches[key]['datetime_local'] = match['datetime_local']
            else:
                # New match from yesterday's page
                match['status'] = 'completed'
                seen_matches[key] = match
    
    all_matches = list(seen_matches.values())
    
    # Filter out WTA tournaments (we only have ATP historical data)
    atp_matches = []
    for match in all_matches:
        tournament = match.get('tournament', '')
        if not is_wta_tournament(tournament):
            atp_matches.append(match)
        else:
            logger.debug(f"Filtered out WTA match: {match.get('player1')} vs {match.get('player2')} at {tournament}")
    
    logger.info(f"Filtered {len(all_matches) - len(atp_matches)} WTA matches, kept {len(atp_matches)} ATP matches")
    all_matches = atp_matches
    
    now_local = datetime.now().astimezone()
    
    filtered_matches = []
    for match in all_matches:
        datetime_local_str = match.get('datetime_local')
        
        # Try to parse datetime for time-based filtering
        match_dt = None
        if datetime_local_str:
            try:
                match_dt = datetime.fromisoformat(datetime_local_str)
            except (ValueError, TypeError):
                pass
        
        is_completed = match.get('status') == 'completed'
        is_live = match.get('is_live', False)
        
        # Use source_day for date_label - this is set based on which page we scraped from
        # This is more reliable than timezone-converted dates
        source_day = match.get('source_day')
        if source_day in ['Yesterday', 'Today', 'Tomorrow']:
            match['date_label'] = source_day
        else:
            # Fallback: determine from date string (using CET dates)
            # FIX: Use CET timezone for consistency with stored match dates
            now_cet = datetime.now(TE_TIMEZONE)
            today_cet_str = now_cet.strftime("%Y-%m-%d")
            yesterday_cet_str = (now_cet - timedelta(days=1)).strftime("%Y-%m-%d")
            tomorrow_cet_str = (now_cet + timedelta(days=1)).strftime("%Y-%m-%d")
            match_date = match.get('date', '')
            
            if match_date == today_cet_str:
                match['date_label'] = 'Today'
            elif match_date == yesterday_cet_str:
                match['date_label'] = 'Yesterday'
            elif match_date == tomorrow_cet_str:
                match['date_label'] = 'Tomorrow'
            else:
                continue  # Outside our window
        
        # For scheduled (not completed, not live) TODAY matches, check if time has passed
        # But keep yesterday's matches even if not marked completed (results page may be delayed)
        if not is_completed and not is_live and match_dt and match.get('date_label') == 'Today':
            # Skip if match started more than 3 hours ago (likely finished but not updated)
            if match_dt < now_local - timedelta(hours=3):
                continue
        
        filtered_matches.append(match)
    
    logger.info(f"Scraped {len(all_matches)} matches, kept {len(filtered_matches)} in 3-day window")
    return pd.DataFrame(filtered_matches)


def _scrape_tennis_explorer_date(date: datetime, category_type: str = None) -> List[Dict]:
    """
    Scrape matches for a specific date from TennisExplorer.
    
    Args:
        date: The date to scrape matches for
        category_type: Optional category filter (e.g., 'atp-single', 'wta-single', 'ch-single')
    """
    # TennisExplorer uses /?date=YYYY-MM-DD format
    date_str = date.strftime("%Y-%m-%d")
    is_today = date.date() == datetime.now().date()
    
    # Build URL with optional category type
    if is_today:
        url = "https://www.tennisexplorer.com/matches/"
    else:
        url = f"https://www.tennisexplorer.com/matches/?date={date_str}"
    
    if category_type:
        if '?' in url:
            url += f"&type={category_type}"
        else:
            url += f"?type={category_type}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch TennisExplorer for {date_str}: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    matches = []
    
    current_tournament = ""
    current_surface = "Hard"
    current_date = date_str
    
    # Find all table rows
    all_rows = soup.find_all('tr')
    
    i = 0
    while i < len(all_rows):
        row = all_rows[i]
        
        # Check for tournament header
        header_cell = row.find('td', class_='t-name', colspan=True)
        if header_cell:
            link = header_cell.find('a')
            if link:
                current_tournament = link.get_text(strip=True)
                # Detect surface
                t_lower = current_tournament.lower()
                if any(x in t_lower for x in ['clay', 'roland', 'rome', 'madrid', 'monte carlo', 'barcelona']):
                    current_surface = "Clay"
                elif any(x in t_lower for x in ['grass', 'wimbledon', 'halle', 'queens']):
                    current_surface = "Grass"
                else:
                    current_surface = "Hard"
            i += 1
            continue
        
        # Check for match row (has time cell and player name)
        time_cell = row.find('td', class_='time')
        player_cell = row.find('td', class_='t-name')
        
        if time_cell and player_cell:
            # Get time and clean it (removes junk like "03:00Livestreams1xBet...")
            time_text_raw = time_cell.get_text(strip=True)
            time_text = clean_time_field(time_text_raw)
            
            # Detect if match is live - be more strict
            is_live = False
            
            # Check for explicit live indicator classes on the row
            row_classes = row.get('class', [])
            if any('live' in str(c).lower() or 'inprogress' in str(c).lower() for c in row_classes):
                is_live = True
            
            # Check result cell for actual in-progress scores (e.g., "6-4 3-2")
            # but NOT for scheduled matches showing "0-0" or empty
            result_cell = row.find('td', class_='result')
            if result_cell and not is_live:
                result_text = result_cell.get_text(strip=True)
                # Match is live if result shows actual game scores (not just 0-0)
                # Look for patterns like "6-4", "3-2", "7-6" etc.
                if result_text and re.search(r'[1-9]-\d|[0-9]-[1-9]', result_text):
                    is_live = True
            
            # Get player 1
            link1 = player_cell.find('a')
            if link1 and '/player/' in str(link1.get('href', '')):
                href1 = link1.get('href', '')
                player1 = clean_player_name(link1.get_text(strip=True), href1)
                
                # Extract odds from the first player's row
                # TennisExplorer structure: first row has both odds (H and A columns)
                # H = home/player1 odds, A = away/player2 odds
                all_cells = row.find_all('td')
                odds1 = None
                odds2 = None
                match_url = None
                
                # Collect all decimal odds from this row
                all_odds = []
                for cell in all_cells:
                    cell_text = cell.get_text(strip=True)
                    # Look for match detail link
                    link = cell.find('a')
                    if link and '/match-detail/' in str(link.get('href', '')):
                        match_url = link.get('href', '')
                    # Look for odds (decimal numbers like 1.42, 2.58)
                    if cell_text and re.match(r'^\d+\.\d{2}$', cell_text):
                        parsed = parse_odds(cell_text)
                        if parsed:
                            all_odds.append(parsed)
                
                # First odds value is player 1 (H), second is player 2 (A)
                if len(all_odds) >= 2:
                    odds1 = all_odds[0]
                    odds2 = all_odds[1]
                elif len(all_odds) == 1:
                    odds1 = all_odds[0]
                
                # Get player 2 from next row
                if i + 1 < len(all_rows):
                    next_row = all_rows[i + 1]
                    next_player_cell = next_row.find('td', class_='t-name')
                    if next_player_cell:
                        link2 = next_player_cell.find('a')
                        if link2 and '/player/' in str(link2.get('href', '')):
                            href2 = link2.get('href', '')
                            player2 = clean_player_name(link2.get_text(strip=True), href2)
                            
                            # Skip if either name is empty or too short
                            if len(player1) > 2 and len(player2) > 2:
                                # Format time display
                                if is_live:
                                    display_time = "LIVE"
                                    local_date = current_date
                                    local_time = "LIVE"
                                    datetime_local = None
                                elif time_text and time_text != '&nbsp;' and time_text != '--:--':
                                    # Convert CET time to local timezone
                                    datetime_local, local_date, local_time = convert_te_time_to_local(current_date, time_text)
                                    display_time = local_time
                                else:
                                    display_time = "TBD"
                                    local_date = current_date
                                    local_time = "TBD"
                                    datetime_local = None
                                
                                # Determine favorite based on odds (lower odds = favorite)
                                favorite = determine_favorite_from_odds(player1, player2, odds1, odds2)
                                
                                # Create ISO datetime string in local timezone
                                datetime_local_str = None
                                if datetime_local:
                                    datetime_local_str = datetime_local.strftime("%Y-%m-%dT%H:%M:%S%z")
                                
                                matches.append({
                                    'date': local_date,  # Use local date instead of CET date
                                    'time': display_time,
                                    'datetime_local': datetime_local_str,
                                    'datetime_cet': f"{current_date}T{time_text}:00" if time_text and time_text not in ['--:--', '&nbsp;'] else None,
                                    'tournament': current_tournament,
                                    'surface': current_surface,
                                    'player1': player1,
                                    'player2': player2,
                                    'round': 'TBD',
                                    'odds_player1': odds1,
                                    'odds_player2': odds2,
                                    'favorite': favorite,
                                    'is_live': is_live,
                                    'match_url': f"https://www.tennisexplorer.com{match_url}" if match_url else None
                                })
                            i += 2
                            continue
        
        i += 1
    
    logger.info(f"Scraped {len(matches)} matches for {current_date}" + (f" ({category_type})" if category_type else ""))
    return matches


def _scrape_results_page(date: datetime) -> List[Dict]:
    """
    Scrape completed match results with scores from TennisExplorer results page.
    
    Args:
        date: The date to get results for
    """
    date_str = date.strftime("%Y-%m-%d")
    is_today = date.date() == datetime.now().date()
    
    if is_today:
        url = "https://www.tennisexplorer.com/results/"
    else:
        url = f"https://www.tennisexplorer.com/results/?date={date_str}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch results page for {date_str}: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    matches = []
    
    current_tournament = ""
    current_surface = "Hard"
    
    all_rows = soup.find_all('tr')
    
    i = 0
    while i < len(all_rows):
        row = all_rows[i]
        
        # Check for tournament header
        header_cell = row.find('td', class_='t-name', colspan=True)
        if header_cell:
            link = header_cell.find('a')
            if link:
                current_tournament = link.get_text(strip=True)
                # Detect surface
                t_lower = current_tournament.lower()
                if any(x in t_lower for x in ['clay', 'roland', 'rome', 'madrid', 'monte carlo', 'barcelona']):
                    current_surface = "Clay"
                elif any(x in t_lower for x in ['grass', 'wimbledon', 'halle', 'queens']):
                    current_surface = "Grass"
                else:
                    current_surface = "Hard"
            i += 1
            continue
        
        # Check for match row (has time cell and player name)
        time_cell = row.find('td', class_='time')
        player_cell = row.find('td', class_='t-name')
        
        if time_cell and player_cell:
            link1 = player_cell.find('a')
            if link1 and '/player/' in str(link1.get('href', '')):
                href1 = link1.get('href', '')
                player1 = clean_player_name(link1.get_text(strip=True), href1)
                
                # Get score cells from player 1 row
                cells1 = row.find_all('td')
                p1_scores = []
                for cell in cells1:
                    text = cell.get_text(strip=True)
                    # Match set scores (single digit or with tiebreak like "7-6")
                    if text and re.match(r'^\d$', text):
                        p1_scores.append(int(text))
                
                # Get player 2 and their scores from next row
                if i + 1 < len(all_rows):
                    next_row = all_rows[i + 1]
                    next_player_cell = next_row.find('td', class_='t-name')
                    if next_player_cell:
                        link2 = next_player_cell.find('a')
                        if link2 and '/player/' in str(link2.get('href', '')):
                            href2 = link2.get('href', '')
                            player2 = clean_player_name(link2.get_text(strip=True), href2)
                            
                            # Get score cells from player 2 row
                            cells2 = next_row.find_all('td')
                            p2_scores = []
                            for cell in cells2:
                                text = cell.get_text(strip=True)
                                if text and re.match(r'^\d$', text):
                                    p2_scores.append(int(text))
                            
                            # Skip if either name is empty or too short
                            if len(player1) > 2 and len(player2) > 2:
                                # Format the score (e.g., "6-4 6-3")
                                score_str = ""
                                winner = None
                                
                                # First score is typically sets won
                                if len(p1_scores) >= 1 and len(p2_scores) >= 1:
                                    p1_sets = p1_scores[0]
                                    p2_sets = p2_scores[0]
                                    
                                    # Determine winner
                                    if p1_sets > p2_sets:
                                        winner = player1
                                    elif p2_sets > p1_sets:
                                        winner = player2
                                    
                                    # Build set-by-set score
                                    set_scores = []
                                    # Remaining scores are individual set games
                                    p1_games = p1_scores[1:]
                                    p2_games = p2_scores[1:]
                                    for j in range(min(len(p1_games), len(p2_games))):
                                        set_scores.append(f"{p1_games[j]}-{p2_games[j]}")
                                    
                                    score_str = " ".join(set_scores) if set_scores else f"{p1_sets}-{p2_sets} sets"
                                
                                matches.append({
                                    'date': date_str,
                                    'time': clean_time_field(time_cell.get_text(strip=True)) or "Finished",
                                    'datetime_local': None,
                                    'datetime_cet': None,
                                    'tournament': current_tournament,
                                    'surface': current_surface,
                                    'player1': player1,
                                    'player2': player2,
                                    'round': 'TBD',
                                    'odds_player1': None,
                                    'odds_player2': None,
                                    'favorite': None,
                                    'is_live': False,
                                    'match_url': None,
                                    'status': 'completed',
                                    'score': score_str,
                                    'winner': winner
                                })
                            i += 2
                            continue
        
        i += 1
    
    logger.info(f"Scraped {len(matches)} completed results for {date_str}")
    return matches


def fuzzy_match_player_name(name: str, player_profiles: 'pd.DataFrame' = None) -> Optional[str]:
    """
    Try to fuzzy match a player name against the TML Database profiles.
    
    This handles cases where:
    - TennisExplorer uses abbreviated names: "J. Sinner" -> "Jannik Sinner"
    - Names have diacritics stripped: "Cerundolo" matches "Francisco Cerundolo"
    - Hyphenated names: "Auger Aliassime" or "Auger-Aliassime"
    
    Args:
        name: The scraped player name
        player_profiles: Optional DataFrame with 'player_name' column from TML data
    
    Returns:
        Best matching full name, or None if no good match
    """
    if not name or len(name) < 2:
        return None
    
    # First check our static mapping
    name_lower = name.lower().strip()
    
    # Try exact match in map
    if name_lower in PLAYER_NAME_MAP:
        return PLAYER_NAME_MAP[name_lower]
    
    # Extract last name for lookup
    # Handle "J. Sinner" format
    match = re.match(r'^([A-Z])\.\s*(.+)$', name)
    if match:
        initial, last_name = match.groups()
        last_name_lower = last_name.lower().strip()
        if last_name_lower in PLAYER_NAME_MAP:
            return PLAYER_NAME_MAP[last_name_lower]
    
    # Handle "Lastname F." format
    match = re.match(r'^(.+)\s+([A-Z])\.$', name)
    if match:
        last_name, initial = match.groups()
        last_name_lower = last_name.lower().strip()
        if last_name_lower in PLAYER_NAME_MAP:
            return PLAYER_NAME_MAP[last_name_lower]
    
    # Try last word as last name
    parts = name.split()
    if parts:
        last_word = parts[-1].lower() if len(parts[-1]) > 2 else (parts[0].lower() if parts else '')
        if last_word in PLAYER_NAME_MAP:
            return PLAYER_NAME_MAP[last_word]
    
    # If we have profiles DataFrame, try matching against it
    if player_profiles is not None and 'player_name' in player_profiles.columns:
        # Try exact substring match on last name
        for _, row in player_profiles.iterrows():
            profile_name = str(row.get('player_name', ''))
            if name_lower in profile_name.lower() or profile_name.lower() in name_lower:
                return profile_name
    
    return None


def clean_player_name(name: str, href: str = None) -> str:
    """
    Clean and normalize player name using the dynamic PlayerMatcher.
    
    TennisExplorer shows abbreviated names like "Tien L." but URLs have full names.
    Uses both static mapping and dynamic TML Database lookup.
    
    Examples:
        - href="/player/tien/", name="Tien L." -> "Learner Tien"
        - href="/player/basavareddy/", name="Basavareddy N." -> "Nishesh Basavareddy"
    """
    if not name:
        return ""
    
    # Use the global player matcher for dynamic matching
    return _player_matcher.get_or_create_mapping(name, href)


def initialize_player_matcher(profiles_df: 'pd.DataFrame') -> None:
    """
    Initialize the player matcher with TML Database profiles.
    Call this once after loading profiles to enable dynamic matching.
    """
    _player_matcher.load_from_profiles(profiles_df)


def get_upcoming_matches(use_live: bool = True) -> pd.DataFrame:
    """
    Get upcoming tennis matches.
    
    Args:
        use_live: If True, try to scrape live data first
    
    Returns:
        DataFrame with upcoming matches, sorted with main tour events first
    """
    if use_live:
        matches = scrape_tennis_explorer()
        if len(matches) > 0:
            # Sort matches to prioritize main tour events
            matches = sort_by_tournament_priority(matches)
            return matches
    
    logger.warning("Live scraping returned no matches, using sample data")
    return get_sample_matches()


def sort_by_tournament_priority(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Sort matches by tournament priority (main tour first, amateur last).
    Also adds 'is_main_tour' flag for filtering in dashboard.
    ATP-only version (WTA removed).
    """
    def get_priority(tournament: str) -> int:
        t = tournament.lower()
        # Highest priority: Grand Slams, ATP Finals
        if any(x in t for x in ['grand slam', 'australian open', 'roland garros', 'wimbledon', 'us open', 'atp finals']):
            return 0
        # High: Masters 1000
        if any(x in t for x in ['masters', 'indian wells', 'miami', 'monte carlo', 'madrid', 'rome', 'canada', 'cincinnati', 'shanghai', 'paris']):
            return 1
        # Medium-high: ATP 500
        if 'atp 500' in t or any(x in t for x in ['dubai', 'acapulco', 'rotterdam', 'tokyo', 'beijing', 'vienna', 'basel', 'halle', 'queens']):
            return 2
        # Medium: ATP 250, Challenger
        if 'atp' in t or 'challenger' in t:
            return 3
        # Exhibition/Special events
        if any(x in t for x in ['next gen', 'world tennis league', 'united cup', 'laver cup', 'hopman', 'davis cup']):
            return 4
        # Low priority: ITF Men's
        if 'itf' in t:
            return 5
        # Lowest: UTR, Futures, amateur
        if any(x in t for x in ['utr', 'futures', 'satellite']):
            return 6
        # Unknown - medium-low priority
        return 5
    
    matches = matches.copy()
    matches['_priority'] = matches['tournament'].apply(get_priority)
    matches['is_main_tour'] = matches['_priority'] <= 4
    matches = matches.sort_values('_priority').drop(columns=['_priority'])
    
    return matches


def get_sample_matches() -> pd.DataFrame:
    """
    Return sample matches for demo/testing.
    Clearly marked as sample data.
    """
    today = datetime.now()
    
    sample_data = [
        {
            'date': today.strftime("%Y-%m-%d"),
            'time': '12:00',
            'tournament': '[SAMPLE] Next Gen ATP Finals',
            'surface': 'Hard',
            'player1': 'Learner Tien',
            'player2': 'Nishesh Basavareddy',
            'round': 'RR',
            'favorite': 'Learner Tien',
            'is_sample': True
        },
        {
            'date': today.strftime("%Y-%m-%d"),
            'time': '14:00',
            'tournament': '[SAMPLE] United Cup',
            'surface': 'Hard',
            'player1': 'Jannik Sinner',
            'player2': 'Carlos Alcaraz',
            'round': 'RR',
            'favorite': 'Jannik Sinner',
            'is_sample': True
        },
        {
            'date': today.strftime("%Y-%m-%d"),
            'time': '16:00',
            'tournament': '[SAMPLE] Exhibition',
            'surface': 'Hard',
            'player1': 'Novak Djokovic',
            'player2': 'Alexander Zverev',
            'round': 'F',
            'favorite': 'Novak Djokovic',
            'is_sample': True
        },
    ]
    
    return pd.DataFrame(sample_data)


def get_tournament_calendar() -> pd.DataFrame:
    """Get 2025 ATP tournament calendar."""
    return pd.DataFrame([
        {"tournament": "Australian Open", "start": "2025-01-13", "end": "2025-01-26", "surface": "Hard", "level": "Grand Slam", "location": "Melbourne"},
        {"tournament": "Rotterdam", "start": "2025-02-03", "end": "2025-02-09", "surface": "Hard", "level": "ATP 500", "location": "Netherlands"},
        {"tournament": "Dubai", "start": "2025-02-24", "end": "2025-03-01", "surface": "Hard", "level": "ATP 500", "location": "UAE"},
        {"tournament": "Indian Wells", "start": "2025-03-05", "end": "2025-03-16", "surface": "Hard", "level": "ATP 1000", "location": "California"},
        {"tournament": "Miami Open", "start": "2025-03-19", "end": "2025-03-30", "surface": "Hard", "level": "ATP 1000", "location": "Florida"},
        {"tournament": "Monte Carlo", "start": "2025-04-06", "end": "2025-04-13", "surface": "Clay", "level": "ATP 1000", "location": "Monaco"},
        {"tournament": "Madrid Open", "start": "2025-04-27", "end": "2025-05-04", "surface": "Clay", "level": "ATP 1000", "location": "Spain"},
        {"tournament": "Italian Open", "start": "2025-05-11", "end": "2025-05-18", "surface": "Clay", "level": "ATP 1000", "location": "Rome"},
        {"tournament": "Roland Garros", "start": "2025-05-25", "end": "2025-06-08", "surface": "Clay", "level": "Grand Slam", "location": "Paris"},
        {"tournament": "Wimbledon", "start": "2025-06-30", "end": "2025-07-13", "surface": "Grass", "level": "Grand Slam", "location": "London"},
        {"tournament": "US Open", "start": "2025-08-25", "end": "2025-09-07", "surface": "Hard", "level": "Grand Slam", "location": "New York"},
        {"tournament": "ATP Finals", "start": "2025-11-09", "end": "2025-11-16", "surface": "Hard", "level": "Finals", "location": "Turin"},
    ])


if __name__ == "__main__":
    print("Testing live scraper...")
    matches = scrape_tennis_explorer()
    
    if len(matches) > 0:
        print(f"\nFound {len(matches)} total matches")
        print(f"\nMatches by date:")
        print(matches['date'].value_counts().to_string())
        
        # Show matches with odds
        if 'odds_player1' in matches.columns:
            with_odds = matches[matches['odds_player1'].notna()]
            print(f"\nMatches with odds: {len(with_odds)}")
            if len(with_odds) > 0:
                cols = ['time', 'player1', 'odds_player1', 'player2', 'odds_player2', 'favorite', 'tournament']
                cols = [c for c in cols if c in with_odds.columns]
                print(with_odds[cols].head(15).to_string(index=False))
        
        # Show matches with scores
        if 'score' in matches.columns:
            with_scores = matches[matches['score'].notna() & (matches['score'] != '')]
            print(f"\nCompleted matches with scores: {len(with_scores)}")
            if len(with_scores) > 0:
                cols = ['player1', 'player2', 'score', 'winner']
                cols = [c for c in cols if c in with_scores.columns]
                print(with_scores[cols].head(15).to_string(index=False))
        
        # Show upcoming matches
        upcoming = matches[matches.get('status', 'scheduled') != 'completed'] if 'status' in matches.columns else matches
        print(f"\nUpcoming/live matches: {len(upcoming)}")
        cols = ['time', 'player1', 'player2', 'tournament', 'favorite']
        cols = [c for c in cols if c in upcoming.columns]
        print(upcoming[cols].head(10).to_string(index=False))
    else:
        print("\nNo live matches found.")
