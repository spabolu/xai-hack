"""Main entrypoint for the NBA commentary agent."""
import json
import sys
from typing import Dict, Any, List, Optional

from agent import NBACommentaryAgent, CommentaryOutput


def process_play_by_play_stream(events: List[Dict[str, Any]], language: str = "en") -> List[CommentaryOutput]:
    """
    Process a stream of play-by-play events.
    
    Args:
        events: List of play-by-play event dictionaries
        language: Language code for commentary (en, es, fr)
    
    Returns:
        List of CommentaryOutput objects
    """
    agent = NBACommentaryAgent(language=language)
    commentaries = []
    
    for event in events:
        try:
            commentary = agent.process_event(event)
            commentaries.append(commentary)
            print(f"✓ Processed event: {event.get('event_type', 'unknown')}", file=sys.stderr)
        except Exception as e:
            print(f"✗ Error processing event: {e}", file=sys.stderr)
            continue
    
    return commentaries


def main():
    """Main entrypoint."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="NBA Commentary Agent - Generate commentary from play-by-play events"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to JSON file containing play-by-play events (or '-' for stdin)",
        default=None
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output JSON file (or '-' for stdout)",
        default="-"
    )
    parser.add_argument(
        "--format",
        choices=["json", "tts"],
        default="json",
        help="Output format: 'json' for structured data, 'tts' for TTS-ready text"
    )
    
    # NBA API options
    parser.add_argument(
        "--team",
        type=str,
        help="Team name or abbreviation (e.g., 'LAL', 'Lakers') to fetch recent game data"
    )
    parser.add_argument(
        "--game-id",
        type=str,
        help="Specific NBA game ID to fetch play-by-play data"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch live game data instead of historical data"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Only include events from the first N minutes of the game (e.g., 10 for first 10 minutes, 25 for first 25 minutes)"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "es", "fr"],
        default="en",
        help="Language for commentary: en (English), es (Spanish), fr (French). Default: en"
    )
    parser.add_argument(
        "--list-games",
        action="store_true",
        help="List today's NBA games and exit"
    )
    
    args = parser.parse_args()
    
    # Handle --list-games
    if args.list_games:
        from nba_data import get_today_games
        
        games = get_today_games()
        if games:
            print("Today's NBA Games:")
            print("-" * 50)
            for game in games:
                print(f"  {game['away_team']} @ {game['home_team']}")
                print(f"    Game ID: {game['game_id']}")
                print(f"    Status: {game['status']}")
                if game['home_score'] is not None:
                    print(f"    Score: {game['away_score']}-{game['home_score']}")
                print()
        else:
            print("No NBA games scheduled for today.")
        return
    
    # Determine data source
    events = []
    
    if args.team or args.game_id:
        # Fetch from NBA API
        from nba_data import fetch_game_events
        
        print(f"Fetching play-by-play data...", file=sys.stderr)
        events = fetch_game_events(
            game_id=args.game_id,
            team=args.team,
            live=args.live,
            duration=args.duration
        )
        print(f"Fetched {len(events)} events", file=sys.stderr)
        
    elif args.input:
        # Read from file or stdin
        if args.input == "-":
            input_data = json.load(sys.stdin)
        else:
            with open(args.input, "r") as f:
                input_data = json.load(f)
        
        # Handle both single event and list of events
        if isinstance(input_data, dict):
            events = [input_data]
        elif isinstance(input_data, list):
            events = input_data
        else:
            raise ValueError("Input must be a JSON object or array of events")
    else:
        # Default to stdin
        input_data = json.load(sys.stdin)
        if isinstance(input_data, dict):
            events = [input_data]
        elif isinstance(input_data, list):
            events = input_data
        else:
            raise ValueError("Input must be a JSON object or array of events")
    
    if not events:
        print("No events to process", file=sys.stderr)
        return
    
    # Process events
    commentaries = process_play_by_play_stream(events, language=args.language)
    
    # Format output
    if args.format == "tts":
        # Format for TTS: simple text output
        output_lines = []
        for commentary in commentaries:
            output_lines.append(commentary.commentary)
            if commentary.game_context:
                output_lines.append(f"[Context] {commentary.game_context}")
            output_lines.append("")  # Blank line between events
        
        output_text = "\n".join(output_lines)
        
        if args.output == "-":
            print(output_text)
        else:
            with open(args.output, "w") as f:
                f.write(output_text)
    else:
        # JSON output
        output_data = [commentary.model_dump() for commentary in commentaries]
        
        if args.output == "-":
            print(json.dumps(output_data, indent=2))
        else:
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Processed {len(commentaries)} events", file=sys.stderr)


if __name__ == "__main__":
    main()
