#!/usr/bin/env python3
"""
Grok Script Generator - Optimized for Real-Time NBA Commentary
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
    TEMPERATURE: float = 0.6  # Lower temp = less creative/hallucinogenic
    MAX_TOKENS: int = 100  # Default fallback
    THREAD_ID: str = os.getenv("THREAD_ID", "nba-commentary-default")

    @classmethod
    def validate(cls) -> None:
        if not cls.XAI_API_KEY:
            raise ValueError("XAI_API_KEY environment variable is required.")


# =============================================================================
# Output Models
# =============================================================================


class CommentaryOutput(BaseModel):
    commentary: str = Field(description="Commentary script text")
    excitement_level: str = Field(default="high enthusiasm", description="low, medium, or high")
    commentator_name: str = Field(default="ara", description="Commentator name")
    timeActual: Optional[float] = Field(default=None, description="Timestamp")


# =============================================================================
# Text Sanitizer (The "Wow" Killer)
# =============================================================================


def sanitize_text(text: str) -> str:
    """
    Aggressively strips forbidden cliches and commentator names from the start of text.
    Recursive to catch cases like "Leo: Wow! Oh my gosh! The play..."
    """
    # 1. Names to strip
    names = [
        "Ara",
        "Eve",
        "Leo",
        "Rex",
        "Sal",
        "Una",
        "Commentator",
        "Announcer",
        "Play-by-play",
        "Color",
    ]
    # 2. Cliches to strip (Case insensitive)
    banned_starts = [
        "Wow",
        "Wow!",
        "WOW!",
        "WOW",
        "Oh my gosh",
        "Oh my god",
        "Oh my gosh,",
        "Oh my gosh!",
        "Oh my gosh!!",
        "Oh my gosh!!!",
        "Geez",
        "Unbelievable",
        "Incredible",
        "Look at that",
        "Listen",
        "Boy",
        "Man",
        "My goodness",
        "Script",
        "Text",
        "Audio",
    ]

    original = text

    # Simple regex to remove Name: prefix
    text = re.sub(r"^(" + "|".join(names) + r")\s*:\s*", "", text, flags=re.IGNORECASE)

    # Loop to remove multiple stacked cliches (e.g. "Wow! My goodness!")
    while True:
        prev_text = text
        for phrase in banned_starts:
            # Matches "Wow" "Wow," "Wow!" "Wow..." at start of string
            pattern = r"^\s*" + re.escape(phrase) + r"[!.,]*\s*"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        if text == prev_text:
            break  # No changes made, we are clean

    # Capitalize first letter if it got messed up
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text.strip()


# =============================================================================
# NBA Commentary Agent
# =============================================================================


class NBACommentaryAgent:
    """AI Agent for generating NBA game commentary."""

    def __init__(self, language: str = "en"):
        Config.validate()

        from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_xai import ChatXAI

        self.llm = ChatXAI(
            xai_api_key=Config.XAI_API_KEY,
            model=Config.GROK_MODEL,
            temperature=Config.TEMPERATURE,
        )

        self.language = language # Store the language        
        # Map common names to full prompt instructions if needed
        lang_instruction = "English"
        if language.lower().startswith("sp"): lang_instruction = "Spanish (Español)"
        elif language.lower().startswith("fr"): lang_instruction = "French (Français)"

        self.system_prompt = f"""You are a professional NBA TV broadcaster. 
Your goal is to provide play-by-play commentary that is TIGHT, PUNCHY, and SYNCED.

IMPORTANT: Generate all commentary in **{lang_instruction}**.

CRITICAL RULES:
1. MAX 6 WORDS per output. No exceptions.
2. START IMMEDIATELY: Do not use "Wow", "Oh my gosh". Start with the action.
3. BE FAST: Subject + Verb + Result.
"""

    def process_event(
        self, event: Dict[str, Any], time_budget: float = 10.0, max_tokens: int = 25
    ) -> CommentaryOutput:
        """
        Process event with a HARD 6-word limit for maximum speed.
        """
        # FORCE 6 WORDS regardless of time budget
        len_instruction = "STRICT LIMIT: Maximum 6 words."

        description = event.get("description", "Play happening")

        prompt = f"""
    EVENT: {description}
    INSTRUCTION: {len_instruction}
    OUTPUT: Just the spoken commentary.
    """

        # Call LLM with very low token limit
        response = self.llm.invoke(
            [("system", self.system_prompt), ("human", prompt)],
            max_tokens=max_tokens,
        )

        # Basic parsing
        text = response.content

        # Determine excitement based on keywords
        excitement = "high enthusiasm"
        # if any(w in description.lower() for w in ["dunk", "3pt", "steal", "block"]):
        #     excitement = "high"

        return CommentaryOutput(
            commentary=text,
            commentator_name="leo",
            excitement_level=excitement,
            timeActual=event.get("timeActual"),
        )


# =============================================================================
# Helper Functions
# =============================================================================


def merge_close_events(events: List[Dict], threshold: float = 8.0) -> List[Dict]:
    """
    Aggressively merges events within `threshold` seconds (Default 8s).
    This ensures we have bigger 'chunks' of time to speak, preventing overlap.
    """
    if not events:
        return []
    merged = []
    batch = [events[0]]

    for i in range(1, len(events)):
        curr = events[i]
        prev = batch[-1]

        # Check time diff between current event and the START of the batch
        # This prevents a "creeping" batch that gets too long
        batch_start_time = batch[0].get("timeActual", 0)
        curr_time = curr.get("timeActual", 0)

        if (curr_time - batch_start_time) < threshold:
            batch.append(curr)
        else:
            merged.append(_create_merged_event(batch))
            batch = [curr]

    if batch:
        merged.append(_create_merged_event(batch))

    print(
        f"📉 Optimized: {len(events)} events -> {len(merged)} sequences.",
        file=sys.stderr,
    )
    return merged


def _create_merged_event(batch: List[Dict]) -> Dict:
    if len(batch) == 1:
        return batch[0]
    final = batch[-1].copy()

    # Join descriptions
    descs = [e.get("description", "") for e in batch if e.get("description")]
    final["description"] = " -> ".join(descs)
    return final


# =============================================================================
# Pipeline Logic
# =============================================================================


async def process_events_pipelined(
    events: List[Dict], agent: NBACommentaryAgent, speed_multiplier: float = 1.0
):
    """
    Smart Pipeline with Sanitization and Timing Enforcement.
    """
    import asyncio
    from tts import text_to_speech_excited
    import subprocess
    import time

    # 1. Setup
    script_queue = asyncio.Queue()
    # Use absolute path to avoid duplication with tts.py's OUTPUT_DIR
    audio_dir = Path(__file__).parent.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"🚀 Starting Commentary Pipeline (Speed: {speed_multiplier}x)", file=sys.stderr
    )

    # --- CONSUMER (TTS & Playback) ---
    async def voice_worker():
        while True:
            item = await script_queue.get()
            if item is None:
                script_queue.task_done()
                break

            text, voice, event_id, excitement = item

            # THE "WOW" KILLER: Sanitize text before TTS
            clean_text = sanitize_text(text)

            if not clean_text:
                script_queue.task_done()
                continue

            filename = f"seq_{event_id}_{int(time.time())}.mp3"
            path = str(audio_dir / filename)

            try:
                # Generate
                print(f"🎵 Gen: {clean_text[:40]}...", file=sys.stderr)
                await asyncio.to_thread(
                    text_to_speech_excited,
                    text=clean_text,
                    voice=voice.capitalize(),
                    output_file=path,
                    excitement_level=excitement,
                )

                # Play (Blocking, ensuring sequential playback)
                # The Brain worker is still running ahead!
                print(f"🔊 Speaking: {clean_text[:40]}...", file=sys.stderr)
                await asyncio.to_thread(subprocess.run, ["afplay", path], check=False)

            except Exception as e:
                print(f"❌ Audio Error: {e}", file=sys.stderr)

            script_queue.task_done()

    voice_task = asyncio.create_task(voice_worker())

    # --- PRODUCER (Grok) ---
    last_event_time = 0.0

    # Commentator rotation
    voices = ["Leo", "Ara", "Rex"]
    voice_idx = 0

    for i, event in enumerate(events):
        event_time = event.get("timeActual", 0)

        # 1. Calculate Budget
        # Look at next event to see how much time we have
        if i + 1 < len(events):
            next_time = events[i + 1].get("timeActual", 0)
            time_budget = next_time - event_time
        else:
            time_budget = 10.0

        # 2. Hard Limits
        # Avg speaking rate: 3 words/sec. Safe limit: 2.5 words/sec.
        # Approx 1.3 tokens per word -> 3.25 tokens/sec
        safe_token_limit = 25

        # 3. Pacing Wait
        wait_time = (event_time - last_event_time) / speed_multiplier
        if wait_time > 0 and i > 0:
            await asyncio.sleep(wait_time)
        last_event_time = event_time

        # 4. Generate
        # If backlog is high, skip non-critical events
        if script_queue.qsize() >= 1 and time_budget < 5.0:
            print(f"⏩ Skipping tight window ({time_budget:.1f}s)...", file=sys.stderr)
            continue

        print(
            f"📝 [Time: {event_time:.1f}] Generating (Budget: {time_budget:.1f}s)...",
            file=sys.stderr,
        )

        try:
            commentary = await asyncio.wait_for(
                asyncio.to_thread(
                    agent.process_event,
                    event,
                    time_budget=time_budget,
                    max_tokens=safe_token_limit,  # Force brevity
                ),
                timeout=10.0,
            )

            # Rotate voice
            current_voice = voices[voice_idx % len(voices)]
            voice_idx += 1

            await script_queue.put(
                (commentary.commentary, current_voice, i, commentary.excitement_level)
            )

        except Exception as e:
            print(f"❌ Grok Error: {e}", file=sys.stderr)

    await script_queue.put(None)
    await voice_task


# =============================================================================
# Main
# =============================================================================


async def main_async():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--pipeline", action="store_true", help="Run optimized pipeline"
    )
    parser.add_argument("--speed", type=float, default=1.0)
    # 1. We capture the language argument here
    parser.add_argument(
        "--language", type=str, default="English", help="Commentary language"
    )
    args = parser.parse_args()

    print(f"📂 Loading {args.input}...", file=sys.stderr)
    with open(args.input) as f:
        events = json.load(f)

    # 2. CRITICAL FIX: Pass args.language to the class init
    agent = NBACommentaryAgent(language=args.language)

    if args.pipeline:
        # Merge events closer than 8 seconds to allow TTS time to finish
        print(f"🔄 Optimizing event stream for {args.language}...", file=sys.stderr)
        optimized = merge_close_events(events, threshold=5.0)

        await process_events_pipelined(optimized, agent, args.speed)
    else:
        print("Please use --pipeline for this script version.", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main_async())
