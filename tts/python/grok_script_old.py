#!/usr/bin/env python3
"""
Grok Script Generator - Generate NBA commentary from play-by-play data using Grok AI.

This script takes NBA play-by-play JSON data (like gameplay.json) and generates
commentary scripts using xAI's Grok API. It supports real-time pacing based on
the timeActual field to simulate live game commentary.

Usage:
    # Batch mode (fast, no timing)
    python grok_script.py --input gameplay.json --output commentary.json

    # Real-time mode (waits based on timestamps)
    python grok_script.py --input gameplay.json --realtime

    # Hybrid mode (lookahead + TTS) - RECOMMENDED for live feel
    python grok_script.py --input gameplay.json --hybrid
    python grok_script.py --input gameplay.json --hybrid --speed 10  # 10x speed for testing
    python grok_script.py --input gameplay.json --hybrid --lookahead 10  # 10s lookahead

    # First N minutes only
    python grok_script.py --input gameplay.json --duration 5 --hybrid
"""

import os
import json
import asyncio
import sys
import re
from typing import Dict, Any, List, Optional
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
    GROK_MODEL: str = os.getenv("GROK_MODEL", "grok-4-1-fast-non-reasoning")
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
        Config.validate()

        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'")

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

        self.thread_id = Config.THREAD_ID
        self.conversation_history: Dict[str, List[BaseMessage]] = {self.thread_id: []}
        self.output_parser = PydanticOutputParser(pydantic_object=CommentaryOutput)

        # STRICTER System Prompt
        self.system_prompt = f"""You are a professional NBA broadcast team.

INSTRUCTIONS:
1. **FORBIDDEN WORDS**: NEVER use "Wow", "Oh my gosh", "Geez", or "Unbelievable". Start sentences directly with the action.
2. **STYLE**: Fast, energetic, and professional. Like Mike Breen or Kevin Harlan.
3. **FORMAT**: Do NOT use "Script:" or commentator names in the text.

IMPORTANT: Generate all commentary in {self.language_name}."""

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}\n\n{format_instructions}"),
            ]
        )

    def _get_history(self) -> List:
        return self.conversation_history.get(self.thread_id, [])

    def _add_to_history(self, human_msg: str, ai_msg: str) -> None:
        if self.thread_id not in self.conversation_history:
            self.conversation_history[self.thread_id] = []
        self.conversation_history[self.thread_id].append(
            self.HumanMessage(content=human_msg)
        )
        self.conversation_history[self.thread_id].append(self.AIMessage(content=ai_msg))

    def process_event(
        self, play_by_play_event: Dict[str, Any], time_budget: float = 10.0
    ) -> CommentaryOutput:
        """
        Process event with a specific time budget constraint.
        """
        # 1. Determine Word Count based on Time Budget
        if time_budget < 8.0:
            word_limit = "EXTREMELY SHORT. Max 6 words."
            style = "Rush delivery. Just the facts."
        else:
            word_limit = "Max 10 words."
            style = "Standard broadcast pacing."

        # 2. Add constraint to the input description
        base_desc = self._format_event(play_by_play_event)
        event_description = (
            f"{base_desc}\n\n"
            f"CONSTRAINT: Next play is in {time_budget:.1f} seconds.\n"
            f"LENGTH: {word_limit}\n"
            f"STYLE: {style}"
        )

        # 3. Invoke
        history = self._get_history()
        format_instructions = self.output_parser.get_format_instructions()

        prompt = self.prompt_template.format_messages(
            input=event_description,
            history=history,
            format_instructions=format_instructions,
        )

        response = self.llm.invoke(prompt)
        response_text = response.content

        try:
            commentary_output = self.output_parser.parse(response_text)
        except Exception:
            # Fallback parsing
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                commentary_output = CommentaryOutput(**json.loads(json_match.group()))
            else:
                commentary_output = CommentaryOutput(
                    commentary=response_text,
                    commentator_name="ara",
                    excitement_level="medium",
                    tone="neutral",
                )

        if "timeActual" in play_by_play_event:
            commentary_output.timeActual = play_by_play_event["timeActual"]

        self._add_to_history(event_description, response_text)
        return commentary_output

    def _format_event(self, event: Dict[str, Any]) -> str:
        parts = []
        if "description" in event:
            parts.append(f"PLAY: {event['description']}")
        if "scoreHome" in event and "scoreAway" in event:
            parts.append(f"SCORE: {event['scoreHome']}-{event['scoreAway']}")
        return "\n".join(parts) if parts else str(event)

    def reset_thread(self) -> None:
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


def process_events_with_tts(
    events: List[Dict],
    agent: NBACommentaryAgent,
    play_audio: bool = True,
) -> List[CommentaryOutput]:
    """
    Process events sequentially: Grok generates commentary, then TTS is created immediately.
    Simple sequential flow: Event → Grok → TTS → Save → Play → Next Event

    Args:
        events: List of play-by-play events
        agent: NBACommentaryAgent instance
        play_audio: Whether to play audio after generating (default: True)

    Returns:
        List of CommentaryOutput objects
    """
    from tts import text_to_speech_excited
    import subprocess

    # Create audio output directory
    audio_dir = Path(__file__).parent.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    commentaries = []

    for i, event in enumerate(events):
        event_num = i + 1
        time_actual = event.get("timeActual", 0)

        # Step 1: Generate commentary with Grok
        try:
            print(f"\n{'─' * 50}", file=sys.stderr)
            print(
                f"📝 Event {event_num}/{len(events)} (time: {time_actual:.1f}s)",
                file=sys.stderr,
            )
            print(f"   {event.get('description', 'unknown')[:60]}", file=sys.stderr)

            commentary = agent.process_event(event)
            commentaries.append(commentary)

            print(
                f"✅ Grok → [{commentary.commentator_name}]: {commentary.commentary[:60]}...",
                file=sys.stderr,
            )

        except Exception as e:
            print(f"❌ Grok error: {e}", file=sys.stderr)
            continue

        # Step 2: Generate TTS immediately
        try:
            voice = commentary.commentator_name.capitalize()
            text = commentary.commentary
            excitement = commentary.excitement_level

            # Generate filename
            safe_voice = voice.lower()
            filename = f"event_{event_num:03d}_{time_actual:.1f}s_{safe_voice}_{excitement}.mp3"
            output_path = audio_dir / filename

            print(f"🔊 Generating TTS ({voice}, {excitement})...", file=sys.stderr)

            saved_path = text_to_speech_excited(
                text=text,
                voice=voice,
                response_format="mp3",
                output_file=str(output_path),
                excitement_level=excitement,
            )

            print(f"💾 Saved: {filename}", file=sys.stderr)

            # Step 3: Play audio if enabled
            if play_audio:
                print(f"▶️  Playing audio...", file=sys.stderr)
                subprocess.run(["afplay", saved_path], check=False)
                print(f"✓  Done playing", file=sys.stderr)

        except Exception as e:
            print(f"❌ TTS error: {e}", file=sys.stderr)
            continue

    return commentaries


# =============================================================================
# Streaming TTS Mode - Low Latency WebSocket Streaming
# =============================================================================


async def process_events_with_streaming_tts(
    events: List[Dict],
    agent: NBACommentaryAgent,
    voice: str = "Leo",
) -> List[CommentaryOutput]:
    """
    Process events with streaming TTS for low-latency playback.

    Uses WebSocket streaming to start audio playback within ~200ms
    instead of waiting 2-5 seconds for full file generation.

    Flow: Event → Grok → Stream TTS (play immediately) → Next Event

    Args:
        events: List of play-by-play events
        agent: NBACommentaryAgent instance
        voice: Voice to use for TTS

    Returns:
        List of CommentaryOutput objects
    """
    from streaming_tts import StreamingTTS, StreamingTTSSimple

    commentaries = []

    # Initialize streaming TTS
    print("🚀 Initializing streaming TTS (low-latency mode)...", file=sys.stderr)

    # Try WebSocket first, fall back to REST streaming
    try:
        tts = StreamingTTS(voice=voice)
        print("   Using WebSocket streaming (~200ms latency)", file=sys.stderr)
    except Exception:
        print(
            "   WebSocket unavailable, falling back to REST streaming", file=sys.stderr
        )
        tts = StreamingTTSSimple(voice=voice)

    try:
        for i, event in enumerate(events):
            event_num = i + 1
            time_actual = event.get("timeActual", 0)

            # Step 1: Generate commentary with Grok
            try:
                print(f"\n{'─' * 50}", file=sys.stderr)
                print(
                    f"📝 Event {event_num}/{len(events)} (time: {time_actual:.1f}s)",
                    file=sys.stderr,
                )
                print(f"   {event.get('description', 'unknown')[:60]}", file=sys.stderr)

                commentary = agent.process_event(event)
                commentaries.append(commentary)

                print(
                    f"✅ Grok → [{commentary.commentator_name}]: {commentary.commentary[:60]}...",
                    file=sys.stderr,
                )

            except Exception as e:
                print(f"❌ Grok error: {e}", file=sys.stderr)
                continue

            # Step 2: Stream TTS immediately (low latency!)
            try:
                text = commentary.commentary
                print(f"🎙️  Streaming TTS...", file=sys.stderr)

                # Stream and play audio
                duration = await tts.speak(text)

                print(f"✓  Streamed {duration:.2f}s of audio", file=sys.stderr)

            except Exception as e:
                print(f"❌ Streaming TTS error: {e}", file=sys.stderr)
                continue

    finally:
        tts.close()

    return commentaries


# =============================================================================
# Hybrid Real-Time Processing with Lookahead and TTS
# =============================================================================


async def generate_tts_audio(commentary: CommentaryOutput, event_index: int = 0) -> str:
    """
    Generate TTS audio file for a commentary entry (without playing).

    Args:
        commentary: CommentaryOutput to convert to speech
        event_index: Event number for filename

    Returns:
        Path to the generated audio file
    """
    from tts import text_to_speech_excited

    voice = commentary.commentator_name.capitalize()
    text = commentary.commentary
    excitement = commentary.excitement_level
    time_actual = commentary.timeActual or 0

    # Create audio output directory
    audio_dir = Path(__file__).parent.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Generate meaningful filename
    safe_voice = voice.lower()
    filename = (
        f"event_{event_index:03d}_{time_actual:.1f}s_{safe_voice}_{excitement}.mp3"
    )
    output_path = audio_dir / filename

    print(
        f"🎵 Generating TTS for event {event_index} ({voice}, {excitement})...",
        file=sys.stderr,
    )

    # Generate TTS (in thread to not block)
    saved_path = await asyncio.to_thread(
        text_to_speech_excited,
        text=text,
        voice=voice,
        response_format="mp3",
        output_file=str(output_path),
        excitement_level=excitement,
    )

    print(f"💾 Pre-generated: {filename}", file=sys.stderr)
    return saved_path


async def play_audio_file(audio_path: str) -> float:
    """
    Play a pre-generated audio file.

    Args:
        audio_path: Path to the audio file

    Returns:
        Duration of playback in seconds
    """
    import subprocess
    import time

    start_time = time.time()
    await asyncio.to_thread(
        subprocess.run,
        ["afplay", audio_path],
        check=False,
    )
    return time.time() - start_time


async def play_tts_for_commentary(
    commentary: CommentaryOutput, event_index: int = 0
) -> float:
    """
    Play TTS for a commentary entry and return the duration.
    Saves audio to tts/audio folder with meaningful filenames.

    Args:
        commentary: CommentaryOutput to speak
        event_index: Event number for filename

    Returns:
        Estimated duration of the TTS audio in seconds
    """
    try:
        # Import TTS functions
        from tts import text_to_speech_excited
        import subprocess
        import time

        voice = commentary.commentator_name.capitalize()
        text = commentary.commentary
        excitement = commentary.excitement_level
        time_actual = commentary.timeActual or 0

        # Create audio output directory
        audio_dir = Path(__file__).parent.parent / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Generate meaningful filename
        # Format: event_001_0.0s_leo_medium.mp3
        safe_voice = voice.lower()
        filename = (
            f"event_{event_index:03d}_{time_actual:.1f}s_{safe_voice}_{excitement}.mp3"
        )
        output_path = audio_dir / filename

        print(f"🔊 [{voice}]: {text[:50]}...", file=sys.stderr)

        # Generate TTS
        saved_path = await asyncio.to_thread(
            text_to_speech_excited,
            text=text,
            voice=voice,
            response_format="mp3",
            output_file=str(output_path),
            excitement_level=excitement,
        )

        print(f"💾 Saved: {filename}", file=sys.stderr)

        # Play audio automatically and measure duration
        start_time = time.time()
        await asyncio.to_thread(
            subprocess.run,
            ["afplay", saved_path],
            check=False,
        )
        duration = time.time() - start_time

        return duration

    except Exception as e:
        print(f"❌ TTS error: {e}", file=sys.stderr)
        # Estimate duration based on text length (roughly 150 words per minute)
        word_count = len(commentary.commentary.split())
        return word_count / 2.5  # ~2.5 words per second


async def process_events_hybrid(
    events: List[Dict],
    agent: NBACommentaryAgent,
    speed_multiplier: float = 1.0,
    lookahead_seconds: float = 5.0,
    play_tts: bool = True,
) -> List[CommentaryOutput]:
    """
    Hybrid real-time processing with lookahead.

    Processes events AHEAD of their timestamp so commentary is ready
    when the game time arrives. Optionally plays TTS immediately.

    Args:
        events: List of play-by-play events with timeActual field
        agent: NBACommentaryAgent instance
        speed_multiplier: Speed up (>1) or slow down (<1) playback
        lookahead_seconds: How far ahead to process events (default 5s)
        play_tts: Whether to play TTS audio

    Returns:
        List of CommentaryOutput objects
    """
    from collections import deque

    if not events:
        return []

    # Queues for managing async processing
    pending_tasks: deque = deque()  # (event_time, event_index, asyncio.Task)
    ready_commentaries: Dict[float, CommentaryOutput] = {}  # event_time -> commentary
    played_times: set = set()  # Track which events have been played
    all_commentaries: List[CommentaryOutput] = []

    # Track game time
    game_start_real_time = asyncio.get_event_loop().time()

    def get_game_time() -> float:
        """Get current game time (adjusted for speed multiplier)."""
        elapsed_real = asyncio.get_event_loop().time() - game_start_real_time
        return elapsed_real * speed_multiplier

    async def process_single_event(event: Dict, index: int) -> tuple:
        """Process one event with Grok and return (event_time, index, commentary)."""
        try:
            commentary = agent.process_event(event)
            return (event.get("timeActual", 0), index, commentary)
        except Exception as e:
            print(f"❌ Error processing event {index}: {e}", file=sys.stderr)
            return (event.get("timeActual", 0), index, None)

    event_index = 0
    total_events = len(events)
    tts_queue: asyncio.Queue = asyncio.Queue()
    tts_playing = False

    print(
        f"🚀 Starting hybrid processing with {lookahead_seconds}s lookahead",
        file=sys.stderr,
    )
    print(
        f"   Speed: {speed_multiplier}x | TTS: {'ON' if play_tts else 'OFF'}",
        file=sys.stderr,
    )

    async def tts_player():
        """Background task to play pre-generated TTS audio at the right time."""
        nonlocal tts_playing
        while True:
            try:
                item = await asyncio.wait_for(tts_queue.get(), timeout=0.5)
                if item is None:  # Shutdown signal
                    break

                # Unpack event time and pre-generated audio path
                event_time, audio_path = item

                # Wait until event time arrives (respecting speed multiplier)
                current_game_time = get_game_time()
                if event_time > current_game_time:
                    wait_seconds = (event_time - current_game_time) / speed_multiplier
                    if wait_seconds > 0:
                        print(
                            f"⏸️  Waiting {wait_seconds:.1f}s for event at {event_time:.1f}s",
                            file=sys.stderr,
                        )
                        await asyncio.sleep(wait_seconds)

                # Play the pre-generated audio instantly!
                tts_playing = True
                print(
                    f"▶️  Playing pre-generated audio for {event_time:.1f}s",
                    file=sys.stderr,
                )
                await play_audio_file(audio_path)
                tts_playing = False
                tts_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"❌ TTS player error: {e}", file=sys.stderr)
                tts_playing = False

    # Start TTS player task if enabled
    tts_task = None
    if play_tts:
        tts_task = asyncio.create_task(tts_player())

    try:
        while event_index < total_events or pending_tasks or ready_commentaries:
            current_game_time = get_game_time()

            # 1. Start processing events within lookahead window
            while event_index < total_events:
                event = events[event_index]
                event_time = event.get("timeActual", 0)

                # If event is within lookahead window, start processing
                if event_time <= current_game_time + lookahead_seconds:
                    print(
                        f"⏱️  [{current_game_time:.1f}s] Queuing event {event_index + 1}/{total_events} (due at {event_time:.1f}s)",
                        file=sys.stderr,
                    )
                    task = asyncio.create_task(process_single_event(event, event_index))
                    pending_tasks.append((event_time, event_index, task))
                    event_index += 1
                else:
                    break

            # 2. Check for completed processing tasks and pre-generate TTS
            completed = []
            for item in list(pending_tasks):
                event_time, idx, task = item
                if task.done():
                    try:
                        result_time, result_idx, commentary = task.result()
                        if commentary:
                            all_commentaries.append(commentary)
                            print(
                                f"✅ [{current_game_time:.1f}s] Commentary ready for event at {result_time:.1f}s",
                                file=sys.stderr,
                            )

                            # Pre-generate TTS immediately (before play time)
                            if play_tts:
                                try:
                                    audio_path = await generate_tts_audio(
                                        commentary, result_idx + 1
                                    )
                                    ready_commentaries[result_time] = (
                                        commentary,
                                        audio_path,
                                    )
                                except Exception as tts_err:
                                    print(
                                        f"⚠️ TTS generation failed: {tts_err}",
                                        file=sys.stderr,
                                    )
                                    ready_commentaries[result_time] = (commentary, None)
                            else:
                                ready_commentaries[result_time] = (commentary, None)
                    except Exception as e:
                        print(f"❌ Task error: {e}", file=sys.stderr)
                    completed.append(item)

            for item in completed:
                pending_tasks.remove(item)

            # 3. Play commentaries that are due (timestamp has arrived)
            for event_time in sorted(ready_commentaries.keys()):
                if event_time <= current_game_time and event_time not in played_times:
                    commentary, audio_path = ready_commentaries[event_time]
                    played_times.add(event_time)

                    print(
                        f"🎤 [{current_game_time:.1f}s] Playing commentary for {event_time:.1f}s",
                        file=sys.stderr,
                    )
                    print(
                        f"   → [{commentary.commentator_name}]: {commentary.commentary[:60]}...",
                        file=sys.stderr,
                    )

                    if play_tts and audio_path:
                        # Queue tuple of (event_time, audio_path) - audio is pre-generated!
                        await tts_queue.put((event_time, audio_path))

                    # Remove from ready queue
                    del ready_commentaries[event_time]

            # 4. Check if we're done
            if (
                event_index >= total_events
                and not pending_tasks
                and not ready_commentaries
                and (not play_tts or tts_queue.empty())
            ):
                break

            # Small delay to prevent busy loop
            await asyncio.sleep(0.05)

    finally:
        # Shutdown TTS player
        if tts_task:
            await tts_queue.put(None)  # Shutdown signal
            # Wait for remaining TTS to finish
            if not tts_queue.empty():
                await tts_queue.join()
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass

    print(
        f"\n✅ Hybrid processing complete: {len(all_commentaries)} commentaries",
        file=sys.stderr,
    )
    return all_commentaries


def merge_close_events(events: List[Dict], threshold: float = 5.0) -> List[Dict]:
    """
    Merges events that occur within `threshold` seconds of each other into single
    'sequence' events. This prevents TTS lag during fast-paced action.

    Args:
        events: Raw list of play-by-play events
        threshold: Time window in seconds to merge (default 5.0s)

    Returns:
        List of merged events
    """
    if not events:
        return []

    merged = []
    current_batch = [events[0]]

    for i in range(1, len(events)):
        current_event = events[i]
        prev_event = current_batch[-1]

        time_diff = current_event.get("timeActual", 0) - prev_event.get("timeActual", 0)

        # If close enough, add to batch
        if time_diff < threshold:
            current_batch.append(current_event)
        else:
            # Flush current batch as one merged event
            merged.append(_create_merged_event(current_batch))
            current_batch = [current_event]

    # Flush final batch
    if current_batch:
        merged.append(_create_merged_event(current_batch))

    print(
        f"📉 Merged {len(events)} raw events into {len(merged)} commentary sequences.",
        file=sys.stderr,
    )
    return merged


def _create_merged_event(batch: List[Dict]) -> Dict:
    """Helper to combine a list of events into one."""
    if len(batch) == 1:
        return batch[0]

    # Use the timestamp of the LAST event (so we speak after the sequence happens)
    final_event = batch[-1]

    # Combine descriptions
    # e.g. "Steal by Curry. FOLLOWED BY: 3pt Shot by Thompson."
    descriptions = [e.get("description", "") for e in batch]
    combined_desc = " -> ".join(descriptions)

    # Create a new event dict
    merged_event = final_event.copy()
    merged_event["description"] = f"SEQUENCE: {combined_desc}"

    # Keep highest importance action type if possible, or just default to the last one
    return merged_event


async def process_events_pipelined(
    events: List[Dict], agent: NBACommentaryAgent, speed_multiplier: float = 1.0
):
    """
    Smart Pipeline:
    - Calculates time gap to next event to enforce brevity.
    - Aggressively cleans 'Wow' and name prefixes.
    """
    import asyncio
    from tts import text_to_speech_excited
    import subprocess
    import time
    from pathlib import Path
    import re

    # Shared Queue
    script_queue = asyncio.Queue()

    # Audio setup
    audio_dir = Path(__file__).parent.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting Smart Pipeline (Speed: {speed_multiplier}x)", file=sys.stderr)

    # --- VOICE WORKER (Consumer) ---
    async def voice_worker():
        print("🎙️  Voice worker started...", file=sys.stderr)

        while True:
            item = await script_queue.get()
            if item is None:
                script_queue.task_done()
                break

            text, commentator, event_id, excitement = item
            voice = commentator.capitalize()

            # --- CLEANING STEP ---
            # 1. Remove Names (Leo:, Ara:)
            clean_text = re.sub(
                r"^(?:Ara|Eve|Leo|Rex|Sal|Una|Commentator):\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )

            # 2. Remove Cliches (Wow!, Oh my gosh!)
            clean_text = re.sub(
                r"^(?:Wow|Oh my gosh|Geez|Unbelievable|Look at that)[!.,]?\s*",
                "",
                clean_text,
                flags=re.IGNORECASE,
            )

            # 3. Capitalize first letter if needed
            if clean_text and clean_text[0].islower():
                clean_text = clean_text[0].upper() + clean_text[1:]

            # Generate File
            filename = f"pipeline_{event_id}_{int(time.time())}.mp3"
            output_path = audio_dir / filename

            try:
                print(
                    f"🎵 Generating audio [{script_queue.qsize()} queued]...",
                    file=sys.stderr,
                )

                audio_file = await asyncio.to_thread(
                    text_to_speech_excited,
                    text=clean_text,
                    voice=voice,
                    response_format="mp3",
                    output_file=str(output_path),
                    excitement_level=excitement,
                )

                print(f"🔊 Speaking: {clean_text[:40]}...", file=sys.stderr)
                await asyncio.to_thread(
                    subprocess.run, ["afplay", audio_file], check=False
                )

            except Exception as e:
                print(f"❌ Audio error: {e}", file=sys.stderr)

            script_queue.task_done()

    voice_task = asyncio.create_task(voice_worker())

    # --- BRAIN WORKER (Producer) ---
    last_event_time = 0.0

    # Simple list of critical terms
    CRITICAL_KEYWORDS = ["made", "dunk", "3pt", "foul", "timeout"]

    for i, event in enumerate(events):
        event_time = event.get("timeActual", 0)
        description = event.get("description", "").lower()

        # --- CALCULATE TIME BUDGET ---
        # Look ahead to the next event to see how much time we have
        if i + 1 < len(events):
            next_event_time = events[i + 1].get("timeActual", 0)
            time_budget = next_event_time - event_time
        else:
            time_budget = 15.0  # Default for last event

        # 1. Pacing
        time_diff = event_time - last_event_time
        if time_diff > 0 and i > 0:
            await asyncio.sleep(time_diff / speed_multiplier)
        last_event_time = event_time

        # 2. Backlog Logic
        backlog_size = script_queue.qsize()
        if backlog_size >= 2:
            is_critical = any(k in description for k in CRITICAL_KEYWORDS)
            if not is_critical:
                print(
                    f"⏩ [Lag: {backlog_size}] Skipping minor event...", file=sys.stderr
                )
                continue

        # 3. Generate with Time Constraint
        print(
            f"📝 [Time: {event_time:.1f}] Generating ({time_budget:.1f}s budget)...",
            file=sys.stderr,
        )
        try:
            # Pass time_budget to process_event
            commentary = await asyncio.wait_for(
                asyncio.to_thread(agent.process_event, event, time_budget), timeout=15.0
            )

            await script_queue.put(
                (
                    commentary.commentary,
                    commentary.commentator_name,
                    i,
                    commentary.excitement_level,
                )
            )

        except asyncio.TimeoutError:
            print(f"❌ Grok timed out...", file=sys.stderr)
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)

    await script_queue.put(None)
    await voice_task
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
        "--tts",
        action="store_true",
        help="TTS mode: generate commentary then immediately create and play TTS audio",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Streaming TTS mode: low-latency (~200ms) audio using WebSocket streaming",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="With --tts: save audio files but don't play them",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Hybrid mode: lookahead processing + real-time TTS playback",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier for realtime/hybrid mode (e.g., 2.0 = 2x speed)",
    )
    parser.add_argument(
        "--lookahead",
        type=float,
        default=5.0,
        help="Lookahead window in seconds for hybrid mode (default: 5.0)",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable TTS playback in hybrid mode (just generate commentary)",
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
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Pipeline mode: Run LLM generation and TTS playback in parallel",
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
    print("🎬 Starting commentary generation...", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    if args.stream:
        # Streaming TTS mode: low-latency WebSocket streaming
        print(
            "⚡ Streaming TTS Mode: Low-latency (~200ms) audio playback",
            file=sys.stderr,
        )
        print(
            "   Audio streams and plays immediately as it's generated\n",
            file=sys.stderr,
        )
        commentaries = await process_events_with_streaming_tts(
            events, agent, voice="Leo"
        )
    elif args.tts:
        # Simple TTS mode: Grok → TTS → Play → Next
        print("🎙️  TTS Mode: Generate commentary → Create audio → Play", file=sys.stderr)
        print(f"   Audio saved to: tts/audio/", file=sys.stderr)
        print(f"   Auto-play: {'OFF' if args.no_play else 'ON'}\n", file=sys.stderr)
        commentaries = process_events_with_tts(
            events, agent, play_audio=not args.no_play
        )
    elif args.hybrid:
        # Hybrid mode: lookahead + TTS
        commentaries = await process_events_hybrid(
            events,
            agent,
            speed_multiplier=args.speed,
            lookahead_seconds=args.lookahead,
            play_tts=not args.no_tts,
        )
    elif args.realtime:
        commentaries = await process_events_realtime(
            events, agent, speed_multiplier=args.speed
        )
    elif args.pipeline:
        # Pipeline mode: Parallel processing
        print("⚡ Pipeline Mode: Parallel LLM generation and Audio playback", file=sys.stderr)
        
        # --- NEW: Merge events before processing ---
        print("🔄 Optimizing event stream...", file=sys.stderr)
        optimized_events = merge_close_events(events, threshold=5.0)
        
        commentaries = await process_events_pipelined(
            optimized_events,  # Pass the merged list, not the raw list
            agent, 
            speed_multiplier=args.speed
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
