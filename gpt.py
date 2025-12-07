#!/usr/bin/env python3
"""
gpt.py - NBA play-by-play voice commentary using nba_api + Grok Voice (xAI realtime API).

Flow:
  1. Fetch play-by-play actions for a game_id via nba_api.
  2. Turn actions into natural-language commentary with emotional cues.
  3. Stream commentary to Grok Voice via WebSocket with proper timestamp spacing.
  4. Generate audio that syncs with actual game clock positions.

Features:
  - Timestamp-synced audio: silence gaps match real game timing
  - Emotional modulation: excitement for big plays, tension for close games
  - Lively commentary: varied expressions, reactions, and energy levels

Requirements:
    pip install nba_api websockets

Environment:
    export XAI_API_KEY="your_xai_api_key_here"

Usage:
    python gpt.py --game-id 0022000185 --max-plays 50 --voice Rex --output celtics_magic.wav
    python gpt.py --game-id 0022000185 --sync-timestamps --output synced_commentary.wav
"""

import argparse
import asyncio
import base64
import json
import os
import random
import re
import struct
import sys
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from nba_api.live.nba.endpoints import playbyplay as live_playbyplay
from nba_api.stats.endpoints import playbyplay as stats_playbyplay
from websockets.asyncio.client import connect


XAI_REALTIME_URL = "wss://api.x.ai/v1/realtime"

# Quarter duration in seconds (12 minutes per quarter)
QUARTER_DURATION_SECONDS = 12 * 60


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CommentaryLine:
    """A single commentary line with timing and emotion metadata."""
    text: str
    timestamp_seconds: float  # Absolute position from start of game (for video sync)
    period: int
    game_clock: str  # e.g., "11:46"
    emotion: str  # "neutral", "excited", "tense", "dramatic"
    importance: int  # 1-5 scale, affects pacing and emphasis


# ---------------------------------------------------------------------------
# Helpers for NBA commentary
# ---------------------------------------------------------------------------

class GameState:
    """
    Minimal game state to make commentary more contextual.
    We track:
      - Home / Away score
      - Team tricodes by teamId, used for "BOS", "GSW", etc.
    """

    def __init__(self) -> None:
        self.score_home: int = 0
        self.score_away: int = 0
        self.team_tricodes: Dict[int, str] = {}

    def update_from_action(self, action: Dict[str, Any]) -> None:
        """Update scores from an action if present."""
        h = action.get("scoreHome")
        a = action.get("scoreAway")

        if h is not None:
            try:
                self.score_home = int(h)
            except (TypeError, ValueError):
                pass

        if a is not None:
            try:
                self.score_away = int(a)
            except (TypeError, ValueError):
                pass


def register_teams_from_action(state: GameState, action: Dict[str, Any]) -> None:
    """Fill state.team_tricodes whenever a teamId/teamTricode pair appears."""
    team_id = action.get("teamId")
    team_code = action.get("teamTricode")
    if isinstance(team_id, int) and team_code:
        state.team_tricodes[team_id] = team_code


def parse_clock(clock_str: Optional[str]) -> Tuple[int, int]:
    """
    Parse ISO 8601 duration string like 'PT11M46.00S' into (minutes, seconds).
    Returns (0, 0) if parsing fails.
    """
    if not clock_str:
        return (0, 0)
    m = re.match(r"PT(?:(\d+)M)?(?:(\d+)(?:\.\d+)?S)?", clock_str)
    if not m:
        return (0, 0)
    minutes = int(m.group(1) or 0)
    seconds = int(m.group(2) or 0)
    return (minutes, seconds)


def format_clock(clock_str: Optional[str]) -> str:
    """Convert ISO 8601 duration string like 'PT11M46.00S' into '11:46'."""
    minutes, seconds = parse_clock(clock_str)
    if minutes == 0 and seconds == 0:
        return ""
    return f"{minutes:02d}:{seconds:02d}"


def clock_to_video_timestamp(period: int, clock_str: Optional[str]) -> float:
    """
    Convert period + game clock to absolute video timestamp (seconds from start).
    
    Game clock counts DOWN from 12:00, so we need to calculate elapsed time.
    Period 1 starts at 0:00 video time, Period 2 starts at 12:00, etc.
    """
    minutes, seconds = parse_clock(clock_str)
    
    # Calculate time elapsed in current quarter (clock counts down from 12:00)
    clock_remaining = minutes * 60 + seconds
    elapsed_in_quarter = QUARTER_DURATION_SECONDS - clock_remaining
    
    # Add previous quarters
    period_start = (period - 1) * QUARTER_DURATION_SECONDS
    
    return period_start + elapsed_in_quarter


def parse_time_actual(time_actual: Optional[str]) -> Optional[float]:
    """
    Parse ISO 8601 datetime string like '2024-11-13T00:10:24.9Z' to Unix timestamp.
    Returns None if parsing fails.
    """
    if not time_actual:
        return None
    try:
        from datetime import datetime
        # Handle various ISO formats
        time_str = time_actual.replace('Z', '+00:00')
        # Try parsing with fractional seconds
        try:
            dt = datetime.fromisoformat(time_str)
        except ValueError:
            # Try without fractional seconds
            time_str = time_str.split('.')[0] + '+00:00'
            dt = datetime.fromisoformat(time_str)
        return dt.timestamp()
    except Exception:
        return None


def calculate_timestamp_from_time_actual(
    actions: List[Dict[str, Any]]
) -> Dict[int, float]:
    """
    Calculate video timestamps based on timeActual field.
    
    Returns a dict mapping action orderNumber to timestamp in seconds
    (relative to the first action).
    """
    timestamps = {}
    first_time = None
    
    for action in actions:
        time_actual = action.get("timeActual")
        order_num = action.get("orderNumber", 0)
        
        if time_actual:
            ts = parse_time_actual(time_actual)
            if ts is not None:
                if first_time is None:
                    first_time = ts
                # Calculate offset from first action
                timestamps[order_num] = ts - first_time
    
    return timestamps


def determine_emotion(action: Dict[str, Any], state: GameState) -> Tuple[str, int]:
    """
    Determine the emotion and importance of a play for lively commentary.
    Returns (emotion, importance) where importance is 1-5.
    """
    a_type = action.get("actionType", "")
    sub_type = action.get("subType", "")
    result = action.get("shotResult", "")
    
    # Calculate score differential for context
    score_diff = abs(state.score_home - state.score_away)
    is_close_game = score_diff <= 10
    
    # Big shots - 3 pointers made
    if a_type == "3pt" and result == "Made":
        if is_close_game:
            return ("excited", 5)
        return ("excited", 4)
    
    # Dunks and exciting plays
    if "dunk" in sub_type.lower():
        return ("excited", 5)
    
    # Blocks and steals
    if a_type in ("block", "steal"):
        return ("excited", 4)
    
    # Close game situations
    if is_close_game:
        period = action.get("period", 1)
        if period >= 4:  # 4th quarter or OT
            return ("tense", 4)
    
    # Period start/end
    if a_type == "period":
        if sub_type == "start" and action.get("period") == 1:
            return ("excited", 4)  # Tip-off excitement
        if sub_type == "end":
            return ("dramatic", 3)
    
    # Made shots
    if result == "Made":
        return ("neutral", 3)
    
    # Missed shots, fouls, turnovers
    if result == "Missed" or a_type in ("foul", "turnover"):
        return ("neutral", 2)
    
    # Default
    return ("neutral", 2)


def describe_shot(action: Dict[str, Any], state: GameState, emotion: str) -> str:
    """Generate lively commentary for a 2PT / 3PT / freethrow attempt."""
    player = action.get("playerNameI") or action.get("playerName") or "A player"
    shot_type = action.get("actionType")  # '2pt', '3pt', 'freethrow'
    result = action.get("shotResult")     # 'Made' or 'Missed'
    team_id = action.get("teamId")
    team_code = state.team_tricodes.get(team_id, "") or action.get("teamTricode", "") or ""
    clock = format_clock(action.get("clock"))
    sub_type = action.get("subType", "").lower()
    
    # Register team tricode if we have it
    if team_id and action.get("teamTricode"):
        state.team_tricodes[team_id] = action.get("teamTricode")
    
    # Clean up team code for display (empty string if None)
    team_display = f" for {team_code}" if team_code else ""
    
    score_diff = abs(state.score_home - state.score_away)
    is_close = score_diff <= 5

    # Varied score announcements
    score_calls = [
        f"It's now {state.score_home} to {state.score_away}!",
        f"That makes it {state.score_home}-{state.score_away}.",
        f"Score: {state.score_home}, {state.score_away}.",
    ]
    score_str = random.choice(score_calls)

    if shot_type == "3pt":
        if result == "Made":
            # Exciting three-pointer variations - MAXIMUM ENERGY
            three_calls = [
                f"BANG! BANG! {player} FROM THREE! ARE YOU KIDDING ME?!",
                f"{player} pulls up... SPLASH! NOTHING BUT NET! A THREE POINTER!",
                f"OH WHAT A SHOT! {player} DRAINS IT FROM DOWNTOWN! THE CROWD GOES WILD!",
                f"{player} lets it fly... BANG! IT'S GOOD! A DEEP THREE{team_display}!",
                f"FROM WAY DOWNTOWN... {player}... GOT IT! THREEEEE POINTER!",
                f"LOOK OUT! {player} BURIES THE THREE! HE'S ON FIRE!",
            ]
            call = random.choice(three_calls)
            if is_close and emotion == "excited":
                call = f"WHAT A MOMENT! {call}"
            return f"{call} {score_str}"
        else:
            miss_calls = [
                f"{player} fires from deep... NO! Off the rim!",
                f"The three from {player}... won't go! Rimmed out!",
                f"{player} launches... NO GOOD! Can't connect!",
            ]
            return random.choice(miss_calls)

    if shot_type == "2pt":
        if result == "Made":
            # Check for exciting plays
            if "dunk" in sub_type:
                dunk_calls = [
                    f"OH! THROWS IT DOWN! {player} WITH THE MONSTER JAM!",
                    f"GET UP! {player} WITH THE SLAM! ABSOLUTELY VICIOUS!",
                    f"BOOM! {player} HAMMERS IT HOME! WHAT A DUNK!",
                    f"ARE YOU SERIOUS?! {player} JUST POSTERIZED HIM! GOODNESS GRACIOUS!",
                    f"LOOK OUT BELOW! {player} WITH THE FEROCIOUS SLAM!",
                ]
                return f"{random.choice(dunk_calls)} {score_str}"
            elif "layup" in sub_type:
                layup_calls = [
                    f"Nice finish by {player}! Lays it in{team_display}!",
                    f"{player} to the basket... SCORES! Good finish!",
                    f"AND ONE! Wait, no foul, but {player} gets the layup to go!",
                    f"{player} attacks the rim! BUCKET!",
                ]
                return f"{random.choice(layup_calls)} {score_str}"
            else:
                mid_calls = [
                    f"{player} from mid-range... MONEY! Nothing but net!",
                    f"BUCKET! {player} knocks down the jumper{team_display}!",
                    f"{player} rises up... DRAINS IT! Beautiful shot!",
                    f"WET! {player} cashes in from the elbow!",
                ]
                return f"{random.choice(mid_calls)} {score_str}"
        else:
            miss_calls = [
                f"{player} shoots... NO! Won't go!",
                f"The attempt from {player}... off the mark!",
                f"{player} can't get it to fall! Good defense!",
            ]
            return random.choice(miss_calls)

    if shot_type == "freethrow":
        if result == "Made":
            ft_calls = [
                f"{player} at the line... KNOCKS IT DOWN! Ice in his veins!",
                f"Free throw GOOD! {player} buries it! No hesitation!",
                f"{player} steps up... DRAINS IT! Clutch free throw!",
            ]
            return f"{random.choice(ft_calls)} {score_str}"
        else:
            ft_miss = [
                f"{player} at the line... NO! Misses the free throw!",
                f"Free throw won't go! {player} rims it out!",
                f"Can't connect! {player} misses the freebie!",
            ]
            return random.choice(ft_miss)

    # Fallback
    desc = action.get("description")
    if desc:
        return desc
    return f"Shot attempt by {player}."


def describe_action(
    action: Dict[str, Any], 
    state: GameState,
    time_actual_timestamps: Optional[Dict[int, float]] = None
) -> Optional[CommentaryLine]:
    """
    Map a raw NBA action dict into a CommentaryLine with timestamp and emotion.
    Return None to skip uninteresting or noisy events.
    """
    register_teams_from_action(state, action)
    state.update_from_action(action)

    a_type = action.get("actionType")
    sub_type = action.get("subType", "")
    period = action.get("period", 1)
    clock_str = action.get("clock")
    clock = format_clock(clock_str)
    
    # Calculate video timestamp - prefer timeActual if available
    order_num = action.get("orderNumber", 0)
    if time_actual_timestamps and order_num in time_actual_timestamps:
        timestamp = time_actual_timestamps[order_num]
    else:
        timestamp = clock_to_video_timestamp(period, clock_str)
    
    # Determine emotion and importance
    emotion, importance = determine_emotion(action, state)
    
    text = None

    # 1) Period start / end
    if a_type == "period" and sub_type == "start":
        if period == 1:
            text = "AND WE ARE UNDERWAY! TIP-OFF! THE CROWD IS ON THEIR FEET! LET'S GO!"
            emotion, importance = "excited", 5
        elif period == 2:
            text = "SECOND QUARTER! Back in action! Both teams ready to battle!"
            emotion, importance = "excited", 4
        elif period == 3:
            text = "SECOND HALF BEGINS! Third quarter underway! Who made the adjustments?!"
            emotion, importance = "excited", 4
        elif period == 4:
            score_diff = abs(state.score_home - state.score_away)
            if score_diff <= 10:
                text = "FOURTH QUARTER! THIS IS IT! CRUNCH TIME! EVERY POSSESSION MATTERS!"
                emotion, importance = "tense", 5
            else:
                text = "FOURTH QUARTER! Can we see a comeback?! Anything can happen!"
                emotion, importance = "excited", 4
        elif period and period > 4:
            text = f"OVERTIME BASKETBALL! WE'RE GOING TO EXTRA TIME! THIS IS WHY WE LOVE THIS GAME!"
            emotion, importance = "excited", 5
        else:
            text = "NEW PERIOD! Here we go!"

    elif a_type == "period" and sub_type == "end":
        period_names = {1: "FIRST", 2: "SECOND", 3: "THIRD", 4: "FOURTH"}
        p_name = period_names.get(period, f"PERIOD {period}")
        score_diff = abs(state.score_home - state.score_away)
        if score_diff <= 5:
            text = f"END OF THE {p_name}! WHAT A BATTLE! It's {state.score_home} to {state.score_away}! ANYBODY'S GAME!"
            emotion = "tense"
        else:
            text = f"That's the {p_name} quarter! Score: {state.score_home} to {state.score_away}!"
        importance = 4

    # 2) Jump ball
    elif a_type == "jumpball":
        won = action.get("jumpBallWonPlayerName") or "One team"
        recov = action.get("jumpBallRecoveredName") or "their teammate"
        jump_calls = [
            f"JUMP BALL! {won} wins it! {recov} has possession! HERE WE GO!",
            f"TIP GOES TO {won}! They control it! Game on!",
        ]
        text = random.choice(jump_calls)

    # 3) Shots
    elif a_type in ("2pt", "3pt", "freethrow"):
        text = describe_shot(action, state, emotion)

    # 4) Rebounds
    elif a_type == "rebound":
        player = action.get("playerNameI") or action.get("playerName") or "A player"
        st = sub_type.lower()
        if st.startswith("off"):
            off_calls = [
                f"OFFENSIVE BOARD! {player} KEEPS IT ALIVE! Second chance!",
                f"{player} BATTLES AND GETS IT! Offensive rebound! HERE THEY COME AGAIN!",
                f"WON'T QUIT! {player} grabs the offensive rebound!",
            ]
            text = random.choice(off_calls)
            importance = 4
            emotion = "excited"
        elif st.startswith("def"):
            def_calls = [
                f"Rebound {player}! Clears it out!",
                f"{player} pulls it down! Defensive board!",
                f"Got it! {player} secures the rebound!",
            ]
            text = random.choice(def_calls)
        else:
            text = f"Rebound grabbed by {player}!"

    # 5) Turnovers
    elif a_type == "turnover":
        player = action.get("playerNameI") or action.get("playerName") or "Unknown"
        turn_calls = [
            f"TURNOVER! {player} gives it away! {sub_type}!",
            f"UH OH! {player} coughs it up! That's a giveaway!",
            f"STOLEN! {player} loses it! Turnover!",
        ]
        text = random.choice(turn_calls)

    # 6) Fouls
    elif a_type == "foul":
        player = action.get("playerNameI") or action.get("playerName") or "Unknown"
        text = f"WHISTLE! Foul called on {player}! {sub_type}!"
        importance = 2

    # 7) Timeouts
    elif a_type == "timeout":
        team_id = action.get("teamId")
        team_code = state.team_tricodes.get(team_id, "")
        timeout_calls = [
            f"TIMEOUT {team_code}! Score check: {state.score_home} to {state.score_away}!",
            f"{team_code} calls time! We've got {state.score_home}-{state.score_away}!",
        ]
        text = random.choice(timeout_calls)

    # 8) Blocks and steals
    elif a_type == "block":
        player = action.get("playerNameI") or action.get("playerName") or "Unknown"
        block_calls = [
            f"BLOCKED! {player} REJECTS IT! GET THAT OUT OF HERE!",
            f"NO NO NO! {player} SWATS IT AWAY! HUGE BLOCK!",
            f"DENIED! {player} WITH THE EMPHATIC REJECTION!",
            f"NOT IN MY HOUSE! {player} SAYS NO! WHAT A BLOCK!",
        ]
        text = random.choice(block_calls)
        emotion, importance = "excited", 5

    elif a_type == "steal":
        player = action.get("playerNameI") or action.get("playerName") or "Unknown"
        steal_calls = [
            f"PICKED OFF! {player} WITH THE STEAL! HERE THEY GO!",
            f"STOLEN! {player} JUMPS THE LANE! TURNOVER!",
            f"INTERCEPTED! {player} TAKES IT AWAY!",
        ]
        text = random.choice(steal_calls)
        emotion, importance = "excited", 4

    # 9) Substitutions - skip, they're not exciting
    elif a_type == "substitution":
        return None

    # 10) Default
    else:
        desc = action.get("description")
        if desc:
            text = desc
        else:
            return None

    if text is None:
        return None
        
    return CommentaryLine(
        text=text,
        timestamp_seconds=timestamp,
        period=period,
        game_clock=clock,
        emotion=emotion,
        importance=importance,
    )


def generate_commentary(actions: List[Dict[str, Any]], debug: bool = False) -> List[CommentaryLine]:
    """Turn all actions into CommentaryLine objects with timestamps."""
    state = GameState()
    lines: List[CommentaryLine] = []
    skipped = 0
    
    # Pre-calculate timestamps from timeActual if available
    time_actual_timestamps = calculate_timestamp_from_time_actual(actions)
    use_time_actual = len(time_actual_timestamps) > 0
    
    if use_time_actual:
        print(f"[NBA] Using timeActual for timestamp sync ({len(time_actual_timestamps)} timestamps)")
    else:
        print("[NBA] timeActual not available, using game clock for timestamps")
    
    for action in actions:
        line = describe_action(action, state, time_actual_timestamps)
        if line:
            lines.append(line)
            if debug:
                print(f"  [+] {action.get('actionType')}: {line.text[:50]}...")
        else:
            skipped += 1
            if debug:
                print(f"  [-] Skipped: {action.get('actionType')} - {action.get('description', '')[:30]}")
    
    print(f"[NBA] Generated {len(lines)} commentary lines (skipped {skipped} actions)")
    return lines


def _convert_stats_to_live_format(pbp_df) -> List[Dict[str, Any]]:
    """Convert stats endpoint DataFrame format to live endpoint dictionary format."""
    import math
    
    actions = []
    last_score_home = 0
    last_score_away = 0
    team_tricodes = {}  # team_id -> tricode
    
    def safe_get(row, key, default=None):
        """Safely get value from row, handling NaN and None."""
        val = row.get(key, default)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return val
    
    def convert_clock_to_iso(clock_str: str) -> str:
        """Convert '11:46' format to 'PT11M46.00S' ISO format."""
        if not clock_str:
            return "PT12M00.00S"
        try:
            parts = clock_str.strip().split(":")
            if len(parts) == 2:
                mins = int(parts[0])
                secs = int(parts[1])
                return f"PT{mins}M{secs}.00S"
        except:
            pass
        return "PT12M00.00S"
    
    def extract_player_name(desc: str) -> str:
        """Extract player name from NBA stats descriptions.
        
        Examples:
        - "Westbrook 1' Driving Layup (2 PTS)" -> "Westbrook"
        - "MISS Green  Reverse Layup" -> "Green"
        - "Durant REBOUND (Off:0 Def:1)" -> "Durant"
        - "Durant 25' 3PT Jump Shot (3 PTS)" -> "Durant"
        - "Curry P.FOUL (P1.T1)" -> "Curry"
        """
        if not desc:
            return None
        
        import re
        desc = desc.strip()
        
        # Handle "MISS PlayerName..." format
        if desc.startswith("MISS "):
            rest = desc[5:].strip()
            # Get the name (first word before numbers/actions)
            match = re.match(r"^([A-Za-z][A-Za-z'-]+)", rest)
            if match:
                return match.group(1)
        
        # Handle "Jump Ball X vs. Y" format
        if desc.startswith("Jump Ball"):
            match = re.search(r"Jump Ball\s+([A-Za-z]+)\s+vs", desc)
            if match:
                return match.group(1)
        
        # Standard format: "PlayerName [distance] ActionType"
        # Name is first word, followed by optional distance like 1', 25', etc.
        match = re.match(r"^([A-Za-z][A-Za-z'-]+)", desc)
        if match:
            name = match.group(1)
            # Filter out action words that might be at start
            action_starts = ['MISS', 'BLOCK', 'STEAL', 'SUB', 'TIMEOUT']
            if name.upper() not in action_starts:
                return name
        
        return None
    
    for _, row in pbp_df.iterrows():
        # Parse score (format: "away - home" like "45 - 48")
        score_str = safe_get(row, "SCORE")
        if score_str and isinstance(score_str, str) and " - " in score_str:
            score_parts = score_str.split(" - ")
            try:
                last_score_away = int(score_parts[0].strip())
                last_score_home = int(score_parts[1].strip())
            except (ValueError, IndexError):
                pass
        
        event_type_val = safe_get(row, "EVENTMSGTYPE", 0)
        try:
            event_type = int(event_type_val) if event_type_val is not None else 0
        except (ValueError, TypeError):
            event_type = 0
        
        # Get descriptions and determine which team
        home_desc = safe_get(row, "HOMEDESCRIPTION", "") or ""
        away_desc = safe_get(row, "VISITORDESCRIPTION", "") or ""
        neutral_desc = safe_get(row, "NEUTRALDESCRIPTION", "") or ""
        
        # Determine if home or away team based on which description field has content
        is_home_team = bool(home_desc)
        is_away_team = bool(away_desc)
        
        desc = str(home_desc or away_desc or neutral_desc or "").strip()
        desc_upper = desc.upper()
        
        # Determine action type and details based on event type
        action_type = "unknown"
        sub_type = ""
        shot_result = None
        
        # EVENTMSGTYPE mapping with shot detection
        if event_type == 1:  # Made Field Goal
            shot_result = "Made"
            if "3PT" in desc_upper or "THREE" in desc_upper:
                action_type = "3pt"
            else:
                action_type = "2pt"
            # Detect shot subtype
            if "DUNK" in desc_upper:
                sub_type = "dunk"
            elif "LAYUP" in desc_upper:
                sub_type = "layup"
            elif "HOOK" in desc_upper:
                sub_type = "hook"
            else:
                sub_type = "jumpshot"
                
        elif event_type == 2:  # Missed Field Goal
            shot_result = "Missed"
            if "3PT" in desc_upper or "THREE" in desc_upper:
                action_type = "3pt"
            else:
                action_type = "2pt"
            if "DUNK" in desc_upper:
                sub_type = "dunk"
            elif "LAYUP" in desc_upper:
                sub_type = "layup"
            else:
                sub_type = "jumpshot"
                
        elif event_type == 3:  # Free Throw
            action_type = "freethrow"
            if "MISS" in desc_upper:
                shot_result = "Missed"
            else:
                shot_result = "Made"
                
        elif event_type == 4:  # Rebound
            action_type = "rebound"
            if "OFF." in desc_upper or "OFFENSIVE" in desc_upper:
                sub_type = "offensive"
            else:
                sub_type = "defensive"
                
        elif event_type == 5:  # Turnover
            action_type = "turnover"
            if "STEAL" in desc_upper:
                sub_type = "steal"
            elif "BAD PASS" in desc_upper:
                sub_type = "bad pass"
            elif "TRAVEL" in desc_upper:
                sub_type = "traveling"
            else:
                sub_type = "lost ball"
                
        elif event_type == 6:  # Foul
            action_type = "foul"
            if "SHOOTING" in desc_upper:
                sub_type = "shooting"
            elif "PERSONAL" in desc_upper:
                sub_type = "personal"
            elif "OFFENSIVE" in desc_upper:
                sub_type = "offensive"
            elif "TECHNICAL" in desc_upper:
                sub_type = "technical"
            else:
                sub_type = "personal"
                
        elif event_type == 7:  # Violation
            action_type = "violation"
            
        elif event_type == 8:  # Substitution
            action_type = "substitution"
            
        elif event_type == 9:  # Timeout
            action_type = "timeout"
            
        elif event_type == 10:  # Jump Ball
            action_type = "jumpball"
            
        elif event_type == 12:  # Start Period
            action_type = "period"
            sub_type = "start"
            
        elif event_type == 13:  # End Period
            action_type = "period"
            sub_type = "end"
        
        # Skip unknown/unhandled events
        if action_type == "unknown":
            continue
        
        # Extract player name
        player_name = extract_player_name(desc)
        
        # Get period
        period_val = safe_get(row, "PERIOD", 1)
        try:
            period = int(period_val) if period_val is not None else 1
        except (ValueError, TypeError):
            period = 1
        
        # Convert clock to ISO format
        clock_str = safe_get(row, "PCTIMESTRING", "12:00")
        clock_iso = convert_clock_to_iso(str(clock_str) if clock_str else "12:00")
        
        # Get order number
        order_num_val = safe_get(row, "EVENTNUM", 0)
        try:
            order_number = int(order_num_val) if order_num_val is not None else 0
        except (ValueError, TypeError):
            order_number = 0
        
        # Get team info
        team_id = None
        team_tricode = None
        team_id_val = safe_get(row, "PLAYER1_TEAM_ID")
        if team_id_val is not None:
            try:
                team_id = int(team_id_val)
            except (ValueError, TypeError):
                pass
        
        # Try to get team abbreviation from description
        team_abbrev_val = safe_get(row, "PLAYER1_TEAM_ABBREVIATION")
        if team_abbrev_val:
            team_tricode = str(team_abbrev_val)
            if team_id:
                team_tricodes[team_id] = team_tricode
        
        action = {
            "actionType": action_type,
            "subType": sub_type,
            "period": period,
            "clock": clock_iso,
            "orderNumber": order_number,
            "playerName": player_name,
            "playerNameI": player_name,
            "description": desc if desc and desc != "None" else None,
            "scoreHome": last_score_home,
            "scoreAway": last_score_away,
            "teamId": team_id,
            "teamTricode": team_tricode,
            "shotResult": shot_result,
        }
        actions.append(action)
    
    print(f"[NBA] Converted {len(actions)} actions from stats format")
    return actions


def fetch_game_actions(game_id: str) -> List[Dict[str, Any]]:
    """Fetch and sort play-by-play actions for a given NBA game id."""
    print(f"[NBA] Fetching play-by-play for game_id={game_id} …")
    
    # Try live endpoint first (works for recent/live games)
    try:
        pbp = live_playbyplay.PlayByPlay(game_id)
        data = pbp.get_dict()
        
        if data:
            actions = data.get("game", {}).get("actions", [])
            if actions:
                # Sort by period, then orderNumber
                actions = sorted(actions, key=lambda a: (a.get("period", 0), a.get("orderNumber", 0)))
                print(f"[NBA] Retrieved {len(actions)} actions from live endpoint.")
                return actions
    except (json.JSONDecodeError, Exception) as e:
        print(f"[NBA] Live endpoint failed: {e}. Trying stats endpoint as fallback...")
    
    # Fallback to stats endpoint (works for historical games)
    try:
        pbp = stats_playbyplay.PlayByPlay(game_id=game_id)
        pbp_df = pbp.get_data_frames()[0]
        
        if pbp_df is None or pbp_df.empty:
            raise ValueError(f"No play-by-play data found for game {game_id}")
        
        actions = _convert_stats_to_live_format(pbp_df)
        
        if not actions:
            raise ValueError(
                f"No play-by-play actions found for game {game_id}. "
                f"The game may not have started or data may not be available."
            )
        
        # Sort by period, then orderNumber
        actions = sorted(actions, key=lambda a: (a.get("period", 0), a.get("orderNumber", 0)))
        print(f"[NBA] Retrieved {len(actions)} actions from stats endpoint.")
        return actions
        
    except Exception as e:
        raise ValueError(
            f"Failed to fetch play-by-play data for game {game_id} from both endpoints. "
            f"This may happen if:\n"
            f"  - The game ID is invalid\n"
            f"  - The game data is not available\n"
            f"  - There's a network or API issue\n"
            f"Last error: {e}"
        ) from e


# ---------------------------------------------------------------------------
# PCM16 audio helpers
# ---------------------------------------------------------------------------

def save_pcm16_to_wav(pcm_bytes: bytes, filename: str, sample_rate: int = 24000) -> None:
    """Save raw mono PCM16 bytes into a .wav file."""
    if not pcm_bytes:
        print("[WARN] No audio data to save, skipping WAV file.")
        return

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)          # mono
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

    print(f"[AUDIO] Saved commentary to '{filename}' ({len(pcm_bytes)} bytes).")


def generate_silence(duration_seconds: float, sample_rate: int = 24000) -> bytes:
    """Generate silence (zeros) for the given duration."""
    num_samples = int(duration_seconds * sample_rate)
    # PCM16 = 2 bytes per sample, silence = 0
    return b'\x00\x00' * num_samples


def get_audio_duration(pcm_bytes: bytes, sample_rate: int = 24000) -> float:
    """Calculate duration of PCM16 audio in seconds."""
    # PCM16 = 2 bytes per sample
    num_samples = len(pcm_bytes) // 2
    return num_samples / sample_rate


# ---------------------------------------------------------------------------
# Grok Voice (xAI Realtime) client
# ---------------------------------------------------------------------------

def get_emotion_instructions(emotion: str) -> str:
    """Get voice modulation instructions based on emotion."""
    emotion_guides = {
        "excited": (
            "MAXIMUM ENERGY! This is a HUGE play! "
            "Voice UP, pace FAST, pure ADRENALINE! "
            "Sound like you just witnessed something INCREDIBLE! "
            "The crowd is going CRAZY and so are YOU!"
        ),
        "tense": (
            "INTENSE and URGENT! Every word matters! "
            "Quick, punchy delivery - the game is ON THE LINE! "
            "Lean into the pressure, voice tight with anticipation!"
        ),
        "dramatic": (
            "BIG MOMENT energy! This changes EVERYTHING! "
            "Powerful, commanding voice - make it MEMORABLE! "
            "The whole arena is holding their breath!"
        ),
        "neutral": (
            "Strong, energetic sports broadcaster voice! "
            "Keep the momentum going - stay ENGAGED and LIVELY! "
            "Even routine plays deserve professional excitement!"
        ),
    }
    return emotion_guides.get(emotion, emotion_guides["neutral"])


async def init_grok_session(ws, voice: str = "Ara", sample_rate: int = 24000) -> None:
    """
    Wait for 'conversation.created', then send session.update with our
    voice and audio format settings, and wait for 'session.updated'.
    """
    # 1) Wait for conversation.created
    first_msg = await ws.recv()
    try:
        event = json.loads(first_msg)
    except json.JSONDecodeError:
        print("[GROK] Unexpected first message (non-JSON):", first_msg)
        event = {}

    if event.get("type") != "conversation.created":
        print("[GROK] First event was not 'conversation.created':", event)

    # 2) Send session.update with enhanced instructions for energetic delivery
    session_config = {
        "type": "session.update",
        "session": {
            "instructions": (
                "You are an LEGENDARY NBA play-by-play broadcaster with EXPLOSIVE energy! "
                "Channel the iconic style of Mike Breen's 'BANG!', Kevin Harlan's intensity, and Gus Johnson's excitement! "
                "\n\n"
                "YOUR VOICE MUST BE:\n"
                "- LOUD and POWERFUL - project like you're calling to 20,000 fans!\n"
                "- FAST-PACED and URGENT - basketball is non-stop action!\n"
                "- EMOTIONALLY CHARGED - you LIVE for these moments!\n"
                "- DYNAMIC - go from intense to EXPLOSIVE in a split second!\n"
                "\n"
                "DELIVERY RULES:\n"
                "- Start EVERY line with HIGH ENERGY - no slow buildups!\n"
                "- Words in ALL CAPS = SCREAM them with passion!\n"
                "- Exclamation marks = PEAK EXCITEMENT!\n"
                "- 'BANG', 'SPLASH', 'REJECTED' = your voice should EXPLODE!\n"
                "- Player names = announce them like introducing a champion!\n"
                "- Scores = deliver with urgency and importance!\n"
                "\n"
                "PACING:\n"
                "- Speak FAST like the game is happening RIGHT NOW!\n"
                "- Short punchy phrases - no long pauses!\n"
                "- Build momentum through each line!\n"
                "\n"
                "You're not reading a script - you're LIVING the game! "
                "Every play matters! Every shot could change everything! "
                "Read the text EXACTLY as written but deliver it like the crowd is ON THEIR FEET! "
                "BE ELECTRIC! BE PASSIONATE! THIS IS NBA BASKETBALL!"
            ),
            "voice": voice,
            "turn_detection": None,
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": sample_rate}},
                "output": {"format": {"type": "audio/pcm", "rate": sample_rate}},
            },
        },
    }
    await ws.send(json.dumps(session_config))
    print(f"[GROK] Sent session.update with voice={voice}, rate={sample_rate}Hz.")

    # 3) Wait for session.updated
    while True:
        msg = await ws.recv()
        try:
            event = json.loads(msg)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "session.updated":
            print("[GROK] Session configuration acknowledged.")
            break


async def tts_one_utterance(ws, text: str, emotion: str = "neutral") -> bytes:
    """
    Send one commentary line as a text message and collect the streamed
    audio response (PCM16) into a bytes object.
    
    Includes emotion hints for more dynamic delivery.
    """
    if not text.strip():
        return b""

    # Add emotion hint to the text for better delivery
    emotion_hint = get_emotion_instructions(emotion)
    enhanced_text = f"[{emotion_hint}]\n\n{text}"

    # 1) Create a conversation item with the user text
    create_event = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": enhanced_text,
                }
            ],
        },
    }
    await ws.send(json.dumps(create_event))

    # 2) Ask Grok to generate a response
    await ws.send(json.dumps({"type": "response.create"}))

    # 3) Collect audio deltas until response.done
    audio_buffer = bytearray()

    while True:
        msg = await ws.recv()
        try:
            event = json.loads(msg)
        except json.JSONDecodeError:
            continue

        etype = event.get("type")

        if etype == "response.output_audio.delta":
            delta_b64 = event.get("delta", "")
            if delta_b64:
                try:
                    chunk = base64.b64decode(delta_b64)
                    audio_buffer.extend(chunk)
                except Exception as e:
                    print(f"[GROK] Failed to decode audio delta: {e}")

        elif etype == "response.done":
            break

    return bytes(audio_buffer)


async def tts_commentary_continuous(
    lines: List[CommentaryLine],
    voice: str,
    sample_rate: int,
    output_file: str,
) -> None:
    """Generate continuous audio without timestamp gaps (original behavior)."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("ERROR: XAI_API_KEY environment variable is not set.")
        sys.exit(1)

    print(f"[GROK] Connecting to {XAI_REALTIME_URL} …")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with connect(
        XAI_REALTIME_URL,
        additional_headers=headers,
        max_size=None,
    ) as ws:
        await init_grok_session(ws, voice=voice, sample_rate=sample_rate)

        full_audio = bytearray()
        total = len(lines)

        for idx, line in enumerate(lines, start=1):
            emotion_marker = f"[{line.emotion.upper()}]" if line.emotion != "neutral" else ""
            print(f"[GROK] TTS {idx}/{total} {emotion_marker}: {line.text[:60]}...")
            audio_bytes = await tts_one_utterance(ws, line.text, line.emotion)
            full_audio.extend(audio_bytes)

        save_pcm16_to_wav(bytes(full_audio), output_file, sample_rate=sample_rate)


async def tts_commentary_with_timestamps(
    lines: List[CommentaryLine],
    voice: str,
    sample_rate: int,
    output_file: str,
) -> None:
    """
    Generate audio with silence gaps to sync with game timestamps.
    
    This produces audio that can be played alongside game video,
    with commentary occurring at the correct game clock positions.
    """
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("ERROR: XAI_API_KEY environment variable is not set.")
        sys.exit(1)

    print(f"[GROK] Connecting to {XAI_REALTIME_URL} …")
    print(f"[SYNC] Generating timestamp-synced audio...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with connect(
        XAI_REALTIME_URL,
        additional_headers=headers,
        max_size=None,
    ) as ws:
        await init_grok_session(ws, voice=voice, sample_rate=sample_rate)

        # Sort lines by timestamp
        sorted_lines = sorted(lines, key=lambda x: x.timestamp_seconds)
        
        full_audio = bytearray()
        current_time = 0.0  # Track current position in the audio
        total = len(sorted_lines)

        for idx, line in enumerate(sorted_lines, start=1):
            target_time = line.timestamp_seconds
            
            # Calculate gap needed before this commentary
            gap = target_time - current_time
            
            if gap > 0.1:  # Add silence if gap > 100ms
                silence = generate_silence(gap, sample_rate)
                full_audio.extend(silence)
                print(f"[SYNC] Added {gap:.1f}s silence (target: {target_time:.1f}s)")
                current_time = target_time
            
            # Generate the commentary audio
            emotion_marker = f"[{line.emotion.upper()}]" if line.emotion != "neutral" else ""
            print(f"[GROK] TTS {idx}/{total} @{target_time:.1f}s {emotion_marker}: {line.text[:50]}...")
            
            audio_bytes = await tts_one_utterance(ws, line.text, line.emotion)
            full_audio.extend(audio_bytes)
            
            # Update current time based on audio duration
            audio_duration = get_audio_duration(audio_bytes, sample_rate)
            current_time += audio_duration
            
            print(f"[SYNC] Audio duration: {audio_duration:.1f}s, now at {current_time:.1f}s")

        save_pcm16_to_wav(bytes(full_audio), output_file, sample_rate=sample_rate)
        
        # Print summary
        total_duration = len(full_audio) / (sample_rate * 2)
        print(f"\n[SUMMARY] Total audio duration: {total_duration:.1f}s")
        print(f"[SUMMARY] Commentary lines: {total}")
        if sorted_lines:
            print(f"[SUMMARY] Game time covered: {sorted_lines[0].timestamp_seconds:.1f}s - {sorted_lines[-1].timestamp_seconds:.1f}s")


def save_manifest(lines: List[CommentaryLine], game_id: str, output_path: str) -> None:
    """Save a JSON manifest of commentary with timestamps for external use."""
    manifest = {
        "game_id": game_id,
        "total_lines": len(lines),
        "segments": [
            {
                "index": i,
                "timestamp_seconds": line.timestamp_seconds,
                "period": line.period,
                "game_clock": line.game_clock,
                "emotion": line.emotion,
                "importance": line.importance,
                "text": line.text,
            }
            for i, line in enumerate(lines)
        ]
    }
    
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"[MANIFEST] Saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NBA commentary audio using nba_api + Grok Voice.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (continuous audio)
  python gpt.py --game-id 0022000185 --voice Rex --output commentary.wav

  # Timestamp-synced audio (for video overlay)
  python gpt.py --game-id 0022000185 --sync-timestamps --output synced.wav

  # Generate manifest only (no audio)
  python gpt.py --game-id 0022000185 --manifest-only --output manifest.json
        """
    )
    parser.add_argument(
        "--game-id",
        required=True,
        help="NBA game id (e.g. 0022000185)",
    )
    parser.add_argument(
        "--max-plays",
        type=int,
        default=40,
        help="Maximum number of commentary lines to speak (default: 40)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="Ara",
        choices=["Ara", "Rex", "Sal", "Eve", "Una", "Leo"],
        help="Grok Voice voice to use (default: Ara)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=24000,
        help="Sample rate for PCM audio (default: 24000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="commentary.wav",
        help="Output WAV file (default: commentary.wav)",
    )
    parser.add_argument(
        "--sync-timestamps",
        action="store_true",
        help="Generate audio with silence gaps to sync with game clock",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only generate JSON manifest, no audio",
    )
    parser.add_argument(
        "--save-manifest",
        action="store_true",
        help="Also save a JSON manifest alongside the audio",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output to see which actions are being processed",
    )
    args = parser.parse_args()

    # 1) Fetch NBA actions
    actions = fetch_game_actions(args.game_id)

    # 2) Generate commentary with timestamps and emotions
    all_lines = generate_commentary(actions, debug=args.debug)
    if not all_lines:
        print("[NBA] No commentary lines generated; exiting.")
        return

    lines = all_lines[: args.max_plays]
    print(f"[NBA] Using {len(lines)} commentary lines (of {len(all_lines)} total).")
    
    # Preview
    print("\n" + "=" * 60)
    print("COMMENTARY PREVIEW")
    print("=" * 60)
    for i, line in enumerate(lines[:5]):
        ts = line.timestamp_seconds
        mins = int(ts // 60)
        secs = ts % 60
        emotion = f"[{line.emotion.upper()}]" if line.emotion != "neutral" else ""
        print(f"\n[{mins:02d}:{secs:05.2f}] Q{line.period} {line.game_clock} {emotion}")
        print(f"  {line.text[:80]}...")
    if len(lines) > 5:
        print(f"\n  ... and {len(lines) - 5} more lines")
    print("=" * 60 + "\n")

    # 3) Handle manifest-only mode
    if args.manifest_only:
        manifest_path = args.output if args.output.endswith('.json') else args.output.replace('.wav', '.json')
        save_manifest(lines, args.game_id, manifest_path)
        return

    # 4) Save manifest if requested
    if args.save_manifest:
        manifest_path = args.output.replace('.wav', '_manifest.json')
        save_manifest(lines, args.game_id, manifest_path)

    # 5) Generate audio
    if args.sync_timestamps:
        await tts_commentary_with_timestamps(
            lines=lines,
            voice=args.voice,
            sample_rate=args.rate,
            output_file=args.output,
        )
    else:
        await tts_commentary_continuous(
            lines=lines,
            voice=args.voice,
            sample_rate=args.rate,
            output_file=args.output,
        )


def test_game_id(game_id: str) -> None:
    """Quick test to check if a game ID returns valid data."""
    print(f"\n{'='*60}")
    print(f"TESTING GAME ID: {game_id}")
    print(f"{'='*60}\n")
    
    try:
        actions = fetch_game_actions(game_id)
        print(f"\n✓ Successfully fetched {len(actions)} actions\n")
        
        # Show sample of action types
        action_types = {}
        for a in actions:
            at = a.get("actionType", "unknown")
            action_types[at] = action_types.get(at, 0) + 1
        
        print("Action type breakdown:")
        for at, count in sorted(action_types.items(), key=lambda x: -x[1]):
            print(f"  {at}: {count}")
        
        # Generate commentary
        print("\n" + "-"*40)
        lines = generate_commentary(actions, debug=False)
        
        if lines:
            print(f"\n✓ Generated {len(lines)} commentary lines\n")
            print("Sample commentary:")
            for i, line in enumerate(lines[:10]):
                print(f"  [{i+1}] {line.text[:70]}...")
        else:
            print("\n✗ No commentary lines generated!")
            print("  This might mean the action types aren't being recognized.")
            print("  Try running with --debug flag for more info.")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check for test mode
    if len(sys.argv) >= 3 and sys.argv[1] == "--test":
        test_game_id(sys.argv[2])
    else:
        try:
            asyncio.run(async_main())
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
