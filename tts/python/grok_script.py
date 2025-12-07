#!/usr/bin/env python3
"""
Grok Script Generator - Generate NBA commentary from play-by-play data using Grok AI.

This script takes NBA play-by-play JSON data (like gameplay.json) and generates
commentary scripts using xAI's Grok API. It supports real-time pacing based on
the timeActual field to simulate live game commentary.

Usage:
    python grok_script.py --input gameplay.json --output commentary.json
    python grok_script.py --input gameplay.json --realtime  # Simulate live timing
    python grok_script.py --input gameplay.json --duration 5  # First 5 minutes only
"""

import os
import json
import asyncio
import sys
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
script_dir = Path(__file__).parent
env_path = script_dir / ".env"
load_dotenv(dotenv_path=env_path, override=True)


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Application configuration."""
    
    XAI_API_KEY: Optional[str] = os.getenv("XAI_API_KEY")
    GROK_MODEL: str = os.getenv("GROK_MODEL", "grok-3-mini")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
    THREAD_ID: str = os.getenv("THREAD_ID", "nba-commentary-default")
    
    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""
        if not cls.XAI_API_KEY:
            raise ValueError(
                "XAI_API_KEY environment variable is required. "
                "Set it in your .env file or environment."
            )


# =============================================================================
# Output Models
# =============================================================================

class GameContext(BaseModel):
    home_team: Optional[str] = Field(default=None, description="Home team name")
    away_team: Optional[str] = Field(default=None, description="Away team name")
    home_score: Optional[int] = Field(default=None, description="Home team score")
    away_score: Optional[int] = Field(default=None, description="Away team score")
    quarter: Optional[int] = Field(default=None, description="Quarter number")
    time: Optional[str] = Field(default=None, description="Game clock time")


class CommentaryOutput(BaseModel):
    commentary: str = Field(description="Commentary script text")
    game_context: Optional[GameContext] = Field(default=None, description="Game context")
    excitement_level: str = Field(default="medium", description="low, medium, or high")
    tone: str = Field(default="neutral", description="Tone of commentary")
    commentator_name: str = Field(default="ara", description="Commentator name")
    timeActual: Optional[float] = Field(default=None, description="Timestamp in seconds from game start")


# =============================================================================
# Supported Languages
# =============================================================================

SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
}


# =============================================================================
# NBA Commentary Agent
# =============================================================================

class NBACommentaryAgent:
    """AI Agent for generating NBA game commentary from play-by-play events."""

    def __init__(self, language: str = "en"):
        """
        Initialize the agent with Grok API.
        
        Args:
            language: Language code for commentary (en, es, fr). Defaults to 'en'.
        """
        Config.validate()
        
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. Supported: {list(SUPPORTED_LANGUAGES.keys())}")
        
        self.language = language
        self.language_name = SUPPORTED_LANGUAGES[language]
        
        # Import langchain components
        from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_xai import ChatXAI
        
        self.HumanMessage = HumanMessage
        self.AIMessage = AIMessage
        
        self.llm = ChatXAI(
            xai_api_key=Config.XAI_API_KEY,
            model=Config.GROK_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
        )
        
        # Thread management - maintain conversation history
        self.thread_id = Config.THREAD_ID
        self.conversation_history: Dict[str, List[BaseMessage]] = {self.thread_id: []}
        
        # Output parser
        self.output_parser = PydanticOutputParser(pydantic_object=CommentaryOutput)
        
        # System prompt
        self.system_prompt = f"""You are generating commentary for an NBA broadcast with TWO commentators working together.

IMPORTANT: Generate all commentary in {self.language_name}.

THE BROADCAST TEAM:
- Pick two commentators from: ara, eve, leo, rex, sal, una
- Keep the SAME two commentators for the entire game
- One is the PLAY-BY-PLAY announcer (describes the action as it happens)
- One is the COLOR COMMENTATOR (adds analysis, reactions, and personality)

NATURAL FLOW - Make them feel like a real broadcast team:
- They should build off each other's energy
- The play-by-play announcer typically starts with what's happening
- The color commentator reacts, adds insight, or hypes up big moments
- They can finish each other's thoughts across events
- Reference what the other said in previous events ("Like you said earlier...")
- React to each other naturally ("Absolutely!" "You called it!" "I can't believe what we just saw!")

WHO SPEAKS (NOT strict alternating):
- Let the moment dictate who speaks - DO NOT mechanically alternate
- One commentator can speak for 3, 4, or even 5+ events in a row if it feels right
- The play-by-play announcer might carry a fast sequence of plays solo
- The color commentator might take over during a timeout or slow moment for extended analysis
- Switch when it feels natural - when someone has something to add or react to
- Think like a real broadcast: sometimes one voice dominates, then the other chimes in

Guidelines:
- Write the commentary as a script - just the spoken words
- Keep it concise and suitable for TTS (text-to-speech)
- Match the energy and excitement of the play
- Use natural sports commentary language for {self.language_name}-speaking audiences
- Keep player names and team names in their original form

Do NOT include commentator names in the commentary text. The speaker is identified in the commentator_name field."""

        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}\n\n{format_instructions}"),
        ])

    def _get_history(self) -> List:
        """Get conversation history for the current thread."""
        return self.conversation_history.get(self.thread_id, [])

    def _add_to_history(self, human_msg: str, ai_msg: str) -> None:
        """Add messages to conversation history."""
        if self.thread_id not in self.conversation_history:
            self.conversation_history[self.thread_id] = []
        
        self.conversation_history[self.thread_id].append(self.HumanMessage(content=human_msg))
        self.conversation_history[self.thread_id].append(self.AIMessage(content=ai_msg))

    def process_event(self, play_by_play_event: Dict[str, Any]) -> CommentaryOutput:
        """
        Process a play-by-play event and return structured commentary.
        
        Args:
            play_by_play_event: Dictionary containing NBA play-by-play event data
        
        Returns:
            CommentaryOutput with structured commentary
        """
        event_description = self._format_event(play_by_play_event)
        history = self._get_history()
        format_instructions = self.output_parser.get_format_instructions()
        
        prompt = self.prompt_template.format_messages(
            input=event_description,
            history=history,
            format_instructions=format_instructions,
        )
        
        response = self.llm.invoke(prompt)
        response_text = response.content
        
        # Parse structured output
        try:
            commentary_output = self.output_parser.parse(response_text)
        except Exception as e:
            # Fallback: try to extract JSON from response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                commentary_output = CommentaryOutput(**json.loads(json_match.group()))
            else:
                # Create a basic output with the raw text
                commentary_output = CommentaryOutput(
                    commentary=response_text,
                    commentator_name="ara",
                    excitement_level="medium",
                    tone="neutral"
                )
        
        # Preserve timeActual from original event
        if 'timeActual' in play_by_play_event:
            commentary_output.timeActual = play_by_play_event['timeActual']
        
        self._add_to_history(event_description, response_text)
        return commentary_output

    def _format_event(self, event: Dict[str, Any]) -> str:
        """Format play-by-play event into a readable description for Grok."""
        parts = []
        
        # Key fields first
        if "period" in event:
            parts.append(f"Quarter: {event['period']}")
        if "clock" in event:
            parts.append(f"Clock: {event['clock']}")
        if "timeActual" in event:
            parts.append(f"Elapsed Time: {event['timeActual']}s")
        if "actionType" in event:
            parts.append(f"Action: {event['actionType']}")
        if "description" in event:
            parts.append(f"Description: {event['description']}")
        if "playerName" in event or "playerNameI" in event:
            player = event.get('playerNameI') or event.get('playerName', '')
            if player:
                parts.append(f"Player: {player}")
        if "teamTricode" in event:
            parts.append(f"Team: {event['teamTricode']}")
        if "scoreHome" in event and "scoreAway" in event:
            parts.append(f"Score: Home {event['scoreHome']} - Away {event['scoreAway']}")
        
        return "\n".join(parts) if parts else str(event)

    def reset_thread(self) -> None:
        """Reset the conversation history for the current thread."""
        self.conversation_history[self.thread_id] = []


# =============================================================================
# Event Processing Functions
# =============================================================================

def filter_events_by_duration(events: List[Dict], duration_minutes: float) -> List[Dict]:
    """
    Filter events to only include those within the specified duration.
    
    Args:
        events: List of play-by-play events with timeActual field
        duration_minutes: Maximum elapsed time in minutes
    
    Returns:
        Filtered list of events
    """
    duration_seconds = duration_minutes * 60
    return [e for e in events if e.get('timeActual', 0) <= duration_seconds]


def filter_meaningful_events(events: List[Dict]) -> List[Dict]:
    """
    Filter out non-meaningful events (substitutions, timeouts, etc.) for commentary.
    
    Args:
        events: List of play-by-play events
    
    Returns:
        Filtered list with only meaningful events
    """
    # Action types worth commenting on
    meaningful_actions = {
        '2pt', '3pt', 'freethrow', 'rebound', 'turnover', 'steal', 'block',
        'foul', 'jumpball', 'violation', 'period'  # period start/end
    }
    
    filtered = []
    for event in events:
        action_type = event.get('actionType', '').lower()
        
        # Include if it's a meaningful action
        if any(m in action_type for m in meaningful_actions):
            filtered.append(event)
        # Or if it has a non-trivial description
        elif event.get('description') and len(event.get('description', '')) > 10:
            # Skip substitutions and some timeouts
            desc_lower = event.get('description', '').lower()
            if 'sub ' not in desc_lower and 'substitution' not in desc_lower:
                filtered.append(event)
    
    return filtered


async def process_events_realtime(
    events: List[Dict],
    agent: NBACommentaryAgent,
    speed_multiplier: float = 1.0,
    callback=None
) -> List[CommentaryOutput]:
    """
    Process events with real-time pacing based on timeActual field.
    
    Args:
        events: List of play-by-play events with timeActual field
        agent: NBACommentaryAgent instance
        speed_multiplier: Speed up (>1) or slow down (<1) playback
        callback: Optional async callback function(commentary) called after each event
    
    Returns:
        List of CommentaryOutput objects
    """
    commentaries = []
    last_time = 0.0
    
    for i, event in enumerate(events):
        current_time = event.get('timeActual', 0)
        
        # Wait for the appropriate time (simulating real-time)
        if i > 0 and current_time > last_time:
            wait_time = (current_time - last_time) / speed_multiplier
            print(f"⏳ Waiting {wait_time:.1f}s (game time: {current_time:.1f}s)", file=sys.stderr)
            await asyncio.sleep(wait_time)
        
        last_time = current_time
        
        # Process the event
        try:
            print(f"🎤 Processing event {i+1}/{len(events)}: {event.get('description', 'unknown')[:50]}...", file=sys.stderr)
            commentary = agent.process_event(event)
            commentaries.append(commentary)
            
            print(f"   → [{commentary.commentator_name}]: {commentary.commentary[:60]}...", file=sys.stderr)
            
            # Call callback if provided
            if callback:
                await callback(commentary)
                
        except Exception as e:
            print(f"❌ Error processing event: {e}", file=sys.stderr)
            continue
    
    return commentaries


def process_events_batch(
    events: List[Dict],
    agent: NBACommentaryAgent,
) -> List[CommentaryOutput]:
    """
    Process events in batch (no real-time pacing).
    
    Args:
        events: List of play-by-play events
        agent: NBACommentaryAgent instance
    
    Returns:
        List of CommentaryOutput objects
    """
    commentaries = []
    
    for i, event in enumerate(events):
        try:
            print(f"🎤 Processing event {i+1}/{len(events)}: {event.get('description', 'unknown')[:50]}...", file=sys.stderr)
            commentary = agent.process_event(event)
            commentaries.append(commentary)
            print(f"   → [{commentary.commentator_name}]: {commentary.commentary[:60]}...", file=sys.stderr)
        except Exception as e:
            print(f"❌ Error processing event: {e}", file=sys.stderr)
            continue
    
    return commentaries


# =============================================================================
# Main Entry Point
# =============================================================================

async def main_async():
    """Async main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate NBA commentary scripts from play-by-play data using Grok AI"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to JSON file containing play-by-play events (e.g., gameplay.json)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="-",
        help="Path to output JSON file (default: stdout)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Only process events from the first N minutes of the game"
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Process events with real-time pacing based on timeActual"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier for realtime mode (e.g., 2.0 = 2x speed)"
    )
    parser.add_argument(
        "--language",
        choices=["en", "es", "fr"],
        default="en",
        help="Language for commentary (default: en)"
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Filter out non-meaningful events (substitutions, etc.)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit to first N events (for testing)"
    )
    
    args = parser.parse_args()
    
    # Load input data
    print(f"📂 Loading events from {args.input}...", file=sys.stderr)
    with open(args.input, 'r') as f:
        events = json.load(f)
    
    print(f"   Loaded {len(events)} events", file=sys.stderr)
    
    # Filter by duration if specified
    if args.duration:
        events = filter_events_by_duration(events, args.duration)
        print(f"   Filtered to first {args.duration} minutes: {len(events)} events", file=sys.stderr)
    
    # Filter meaningful events if specified
    if args.filter:
        events = filter_meaningful_events(events)
        print(f"   Filtered to meaningful events: {len(events)} events", file=sys.stderr)
    
    # Limit events if specified
    if args.limit:
        events = events[:args.limit]
        print(f"   Limited to first {args.limit} events", file=sys.stderr)
    
    if not events:
        print("❌ No events to process", file=sys.stderr)
        return
    
    # Initialize agent
    print(f"🤖 Initializing Grok agent (language: {args.language})...", file=sys.stderr)
    agent = NBACommentaryAgent(language=args.language)
    
    # Process events
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"🎬 Starting commentary generation...", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)
    
    if args.realtime:
        commentaries = await process_events_realtime(
            events, agent, speed_multiplier=args.speed
        )
    else:
        commentaries = process_events_batch(events, agent)
    
    # Output results
    output_data = [c.model_dump() for c in commentaries]
    
    if args.output == "-":
        print(json.dumps(output_data, indent=2))
    else:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✅ Saved {len(commentaries)} commentary entries to {args.output}", file=sys.stderr)


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
