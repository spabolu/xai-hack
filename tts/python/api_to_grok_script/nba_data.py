"""NBA API data fetching module for play-by-play events."""

from typing import Dict, Any, List, Optional
from datetime import datetime


def parse_game_time(period: int, time_string: str) -> float:
    """
    Parse game time and return elapsed minutes from game start.
    
    Args:
        period: Quarter number (1-4, or 5+ for OT)
        time_string: Time remaining in period (e.g., "10:30", "5:45")
    
    Returns:
        Elapsed minutes from game start
    """
    try:
        # Parse time string (MM:SS format)
        if ':' in str(time_string):
            parts = str(time_string).split(':')
            minutes = int(parts[0])
            seconds = int(parts[1]) if len(parts) > 1 else 0
        else:
            # Handle edge cases
            minutes = 0
            seconds = 0
        
        time_remaining = minutes + seconds / 60.0
        
        # Each quarter is 12 minutes
        # Elapsed = (completed quarters * 12) + (12 - time remaining in current quarter)
        quarter_length = 12.0
        elapsed = (period - 1) * quarter_length + (quarter_length - time_remaining)
        
        return max(0, elapsed)
    except (ValueError, TypeError):
        return 0.0


def filter_events_by_duration(
    events: List[Dict[str, Any]], 
    duration_minutes: float,
    is_live: bool = False
) -> List[Dict[str, Any]]:
    """
    Filter events to only include those within the specified duration from game start.
    
    Args:
        events: List of play-by-play events
        duration_minutes: Maximum elapsed game time in minutes
        is_live: Whether events are from live API (different field names)
    
    Returns:
        Filtered list of events
    """
    filtered = []
    
    for event in events:
        if is_live:
            period = event.get('period', 1)
            clock = event.get('clock', '12:00')
        else:
            period = event.get('PERIOD', 1)
            clock = event.get('PCTIMESTRING', '12:00')
        
        elapsed = parse_game_time(period, clock)
        
        if elapsed <= duration_minutes:
            # Add elapsed time to event for reference
            event['_elapsed_minutes'] = round(elapsed, 2)
            filtered.append(event)
    
    return filtered


def get_teams() -> List[Dict[str, Any]]:
    """Get all NBA teams."""
    from nba_api.stats.static import teams
    return teams.get_teams()


def find_team_id(team_name: str) -> Optional[int]:
    """
    Find team ID by name, abbreviation, or nickname.
    
    Args:
        team_name: Team name, abbreviation (e.g., 'LAL'), or nickname (e.g., 'Lakers')
    
    Returns:
        Team ID or None if not found
    """
    from nba_api.stats.static import teams
    
    nba_teams = teams.get_teams()
    team_name_lower = team_name.lower()
    
    for team in nba_teams:
        if (team_name_lower == team['abbreviation'].lower() or
            team_name_lower == team['nickname'].lower() or
            team_name_lower == team['full_name'].lower() or
            team_name_lower in team['full_name'].lower()):
            return team['id']
    
    return None


def get_recent_game_id(team_id: int, season: Optional[str] = None) -> Optional[str]:
    """
    Get the most recent game ID for a team.
    
    Args:
        team_id: NBA team ID
        season: Season string (e.g., '2024-25'). Defaults to current season.
    
    Returns:
        Game ID string or None
    """
    from nba_api.stats.endpoints import leaguegamefinder
    from nba_api.stats.library.parameters import Season, SeasonType
    
    season = season or Season.default
    
    gamefinder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable=SeasonType.regular
    )
    
    games = gamefinder.get_normalized_dict().get('LeagueGameFinderResults', [])
    
    if games:
        return games[0]['GAME_ID']
    
    return None


def get_play_by_play(game_id: str) -> List[Dict[str, Any]]:
    """
    Fetch play-by-play data for a specific game.
    
    Args:
        game_id: NBA game ID (10-digit string)
    
    Returns:
        List of play-by-play event dictionaries
    """
    from nba_api.stats.endpoints import playbyplay
    
    pbp = playbyplay.PlayByPlay(game_id)
    plays = pbp.get_normalized_dict().get('PlayByPlay', [])
    
    return plays


def get_live_play_by_play(game_id: str) -> List[Dict[str, Any]]:
    """
    Fetch live play-by-play data for an ongoing game.
    
    Args:
        game_id: NBA game ID
    
    Returns:
        List of live play-by-play actions
    """
    from nba_api.live.nba.endpoints import playbyplay
    
    pbp = playbyplay.PlayByPlay(game_id)
    data = pbp.get_dict()
    
    return data.get('game', {}).get('actions', [])


def convert_to_commentary_event(play: Dict[str, Any], is_live: bool = False) -> Dict[str, Any]:
    """
    Convert NBA API play-by-play data to commentary event format.
    
    Args:
        play: Raw play-by-play data from NBA API
        is_live: Whether this is live data (different format)
    
    Returns:
        Event dictionary formatted for the commentary agent
    """
    if is_live:
        period = play.get('period', 1)
        clock = play.get('clock', '12:00')
        elapsed = parse_game_time(period, clock)
        
        return {
            "event_type": play.get('actionType', 'unknown'),
            "description": play.get('description', ''),
            "player": play.get('playerName') or play.get('playerNameI', ''),
            "team": play.get('teamTricode', ''),
            "score": f"{play.get('scoreHome', 0)}-{play.get('scoreAway', 0)}",
            "time": f"Q{period} {clock}",
            "quarter": period,
            "elapsed_minutes": round(elapsed, 2),
            "action_number": play.get('actionNumber'),
            "is_field_goal": play.get('isFieldGoal', False),
        }
    else:
        # Historical/stats data format
        event_types = {
            1: "made_shot",
            2: "missed_shot",
            3: "free_throw",
            4: "rebound",
            5: "turnover",
            6: "foul",
            7: "violation",
            8: "substitution",
            9: "timeout",
            10: "jump_ball",
            11: "ejection",
            12: "period_start",
            13: "period_end",
        }
        
        event_type = event_types.get(play.get('EVENTMSGTYPE'), 'unknown')
        
        # Get description from home or visitor
        description = (
            play.get('HOMEDESCRIPTION') or 
            play.get('VISITORDESCRIPTION') or 
            play.get('NEUTRALDESCRIPTION') or 
            ''
        )
        
        # Extract player name from description or player fields
        player = play.get('PLAYER1_NAME', '')
        
        # Get team from player info
        team = play.get('PLAYER1_TEAM_ABBREVIATION', '')
        
        # Format score
        score_home = play.get('SCOREHOME', '')
        score_away = play.get('SCOREAWAY', '')
        score = f"{score_home}-{score_away}" if score_home and score_away else ''
        
        # Calculate elapsed time
        period = play.get('PERIOD', 1)
        clock = play.get('PCTIMESTRING', '12:00')
        elapsed = parse_game_time(period, clock)
        
        return {
            "event_type": event_type,
            "description": description,
            "player": player,
            "team": team,
            "score": score,
            "time": f"Q{period} {clock}",
            "quarter": period,
            "elapsed_minutes": round(elapsed, 2),
            "event_num": play.get('EVENTNUM'),
        }


def fetch_game_events(
    game_id: Optional[str] = None,
    team: Optional[str] = None,
    live: bool = False,
    duration: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Fetch and convert play-by-play events for a game.
    
    Args:
        game_id: Specific game ID. If not provided, uses team's most recent game.
        team: Team name/abbreviation to find recent game (used if game_id not provided)
        live: Whether to fetch live data
        duration: Only include events from the first N minutes of the game
    
    Returns:
        List of events formatted for the commentary agent
    """
    # Get game_id if not provided
    if not game_id:
        if not team:
            raise ValueError("Either game_id or team must be provided")
        
        team_id = find_team_id(team)
        if not team_id:
            raise ValueError(f"Team '{team}' not found")
        
        game_id = get_recent_game_id(team_id)
        if not game_id:
            raise ValueError(f"No recent games found for team '{team}'")
        
        print(f"Using game ID: {game_id}")
    
    # Fetch play-by-play data
    if live:
        plays = get_live_play_by_play(game_id)
    else:
        plays = get_play_by_play(game_id)
    
    # Filter by duration if specified (before conversion)
    if duration:
        plays = filter_events_by_duration(plays, duration, is_live=live)
        print(f"Filtered to first {duration} minutes: {len(plays)} events")
    
    # Convert to commentary events
    events = [convert_to_commentary_event(play, is_live=live) for play in plays]
    
    # Filter out empty/minimal events
    events = [e for e in events if e.get('description')]
    
    return events


def get_today_games() -> List[Dict[str, Any]]:
    """
    Get list of games scheduled for today.
    
    Returns:
        List of game info dictionaries
    """
    from nba_api.live.nba.endpoints import scoreboard
    
    board = scoreboard.ScoreBoard()
    games = board.get_dict().get('scoreboard', {}).get('games', [])
    
    return [
        {
            'game_id': g.get('gameId'),
            'home_team': g.get('homeTeam', {}).get('teamTricode'),
            'away_team': g.get('awayTeam', {}).get('teamTricode'),
            'status': g.get('gameStatusText'),
            'home_score': g.get('homeTeam', {}).get('score'),
            'away_score': g.get('awayTeam', {}).get('score'),
        }
        for g in games
    ]


if __name__ == "__main__":
    # Example usage
    print("Today's Games:")
    print("-" * 40)
    
    games = get_today_games()
    if games:
        for game in games:
            print(f"{game['away_team']} @ {game['home_team']} - {game['status']}")
            if game['home_score'] is not None:
                print(f"  Score: {game['away_score']}-{game['home_score']}")
    else:
        print("No games today")
    
    print("\n" + "=" * 40)
    print("Example: Fetching Lakers' most recent game...")
    print("=" * 40)
    
    try:
        events = fetch_game_events(team="LAL", duration=5)
        for i, event in enumerate(events, 1):
            print(f"\n{i}. [{event['time']}] {event['event_type']}")
            print(f"   {event['description']}")
    except Exception as e:
        print(f"Error: {e}")
