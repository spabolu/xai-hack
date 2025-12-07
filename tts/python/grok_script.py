#!/usr/bin/env python3
"""
Grok Script Generator - Optimized for Real-Time NBA Commentary
"""

import os
import json
import asyncio
import sys
import re
import base64
from typing import Dict, Any, List, Optional, AsyncGenerator
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
    TEMPERATURE: float = 0.7  # Slightly higher for more emotional variance
    MAX_TOKENS: int = 100
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
# Text Sanitizer
# =============================================================================


def sanitize_text(text: str) -> str:
    """
    Aggressively strips forbidden cliches and commentator names.
    """
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
    banned_starts = [
        "Wow",
        "Wow!",
        "Oh my gosh",
        "Oh my god",
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
    text = re.sub(r"^(" + "|".join(names) + r")\s*:\s*", "", text, flags=re.IGNORECASE)

    while True:
        prev_text = text
        for phrase in banned_starts:
            pattern = r"^\s*" + re.escape(phrase) + r"[!.,]*\s*"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        if text == prev_text:
            break

    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text.strip()


# =============================================================================
# NBA Commentary Agent
# =============================================================================


class NBACommentaryAgent:
    """AI Agent for generating NBA game commentary with specific team bias."""

    def __init__(self, language: str = "en", team_support: str = "Neither"):
        Config.validate()

        from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_xai import ChatXAI

        self.llm = ChatXAI(
            xai_api_key=Config.XAI_API_KEY,
            model=Config.GROK_MODEL,
            temperature=Config.TEMPERATURE,
        )

        self.language = language

        # 1. Determine Language Instruction
        lang_instruction = "English"
        if language.lower().startswith("sp"):
            lang_instruction = "Spanish (Español)"
        elif language.lower().startswith("fr"):
            lang_instruction = "French (Français)"

        # 2. Determine Persona based on Team Support
        if "neither" in team_support.lower() or "neutral" in team_support.lower():
            persona = "You are a professional, NEUTRAL NBA TV broadcaster."
            bias_instruction = (
                "Maintain objectivity. Get excited for big plays by EITHER team."
            )
        else:
            persona = f"You are a DIE-HARD, BIASED fan commentator for the **{team_support}**."
            bias_instruction = f"""
            **BIAS RULES:**
            - If **{team_support}** scores/steals/blocks: BE ECSTATIC. Use words like "Yes!", "Finally!", "Beautiful!".
            - If the OPPONENT scores: BE SAD, FRUSTRATED, or DISMISSIVE. (e.g., "Ugh, lucky shot," "Defense is sleeping.").
            - Never praise the opponent enthusiastically.
            """

        self.system_prompt = f"""{persona}
Your goal is to provide play-by-play commentary that is TIGHT, PUNCHY, and SYNCED.

IMPORTANT: Generate all commentary in **{lang_instruction}**.

{bias_instruction}

CRITICAL RULES:
1. MAX 8 WORDS per output. No exceptions.
2. START IMMEDIATELY: Do not use "Wow", "Oh my gosh".
3. BE FAST: Subject + Verb + Result.
"""

    def process_event(
        self, event: Dict[str, Any], time_budget: float = 10.0, max_tokens: int = 50
    ) -> CommentaryOutput:
        len_instruction = "STRICT LIMIT: Maximum 15 words."
        description = event.get("description", "Play happening")

        prompt = f"""
    EVENT: {description}
    INSTRUCTION: {len_instruction}
    OUTPUT: Just the spoken commentary.
    """

        response = self.llm.invoke(
            [("system", self.system_prompt), ("human", prompt)],
            max_tokens=max_tokens,
        )

        text = response.content

        # Excitement logic
        # If biased, we might want to manually adjust excitement later,
        # but for now, we let the text carry the emotion.
        excitement = "high enthusiasm"
        # if any(w in description.lower() for w in ["dunk", "3pt", "steal", "block"]):
        #     excitement = "high"

        return CommentaryOutput(
            commentary=text,
            commentator_name="leo",
            excitement_level=excitement,
            timeActual=event.get("timeActual"),
        )

    async def process_event_streaming(
        self, event: Dict[str, Any], max_tokens: int = 50
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Grok LLM for low-latency commentary generation.

        Args:
            event: NBA event dictionary with 'description' field
            max_tokens: Maximum tokens to generate

        Yields:
            Token strings as they are generated
        """
        len_instruction = "STRICT LIMIT: Maximum 15 words."
        description = event.get("description", "Play happening")

        prompt = f"""
    EVENT: {description}
    INSTRUCTION: {len_instruction}
    OUTPUT: Just the spoken commentary.
    """

        async for chunk in self.llm.astream(
            [("system", self.system_prompt), ("human", prompt)],
            max_tokens=max_tokens,
        ):
            if chunk.content:
                yield chunk.content


# =============================================================================
# Streaming TTS (WebSocket - No File Storage)
# =============================================================================


async def stream_to_speaker(text: str, voice: str = "leo") -> None:
    """
    Stream text directly to speaker via WebSocket TTS.
    No file storage - audio chunks play immediately as they arrive.

    Args:
        text: Text to convert to speech
        voice: Voice ID (ara, rex, sal, eve, una, leo)
    """
    import websockets

    # Try to import pyaudio
    try:
        import pyaudio
    except ImportError:
        print("Warning: PyAudio not available, audio will not play", file=sys.stderr)
        return

    api_key = Config.XAI_API_KEY
    if not api_key:
        raise ValueError("XAI_API_KEY not found")

    # WebSocket TTS endpoint
    base_url = os.getenv("BASE_URL", "https://api.x.ai/v1")
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
    uri = f"{ws_url}/realtime/audio/speech"

    # Audio settings (24kHz, mono, 16-bit PCM)
    sample_rate = 24000
    channels = 1

    headers = {"Authorization": f"Bearer {api_key}"}

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    audio_stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        output=True,
    )

    try:
        async with websockets.connect(uri, additional_headers=headers) as websocket:
            # Send config
            config_message = {"type": "config", "data": {"voice_id": voice}}
            await websocket.send(json.dumps(config_message))

            # Send text
            text_message = {
                "type": "text_chunk",
                "data": {"text": text, "is_last": False},
            }
            await websocket.send(json.dumps(text_message))

            # Receive and play audio chunks
            while True:
                try:
                    response = await websocket.recv()
                    data = json.loads(response)

                    # Extract audio data
                    audio_b64 = data["data"]["data"]["audio"]
                    is_last = data["data"]["data"].get("is_last", False)

                    # Decode and play immediately
                    chunk_bytes = base64.b64decode(audio_b64)
                    if len(chunk_bytes) > 0:
                        await asyncio.to_thread(audio_stream.write, chunk_bytes)

                    if is_last:
                        break

                except websockets.exceptions.ConnectionClosedOK:
                    break
                except websockets.exceptions.ConnectionClosedError:
                    break

    except Exception as e:
        print(f"Streaming TTS error: {e}", file=sys.stderr)
    finally:
        audio_stream.stop_stream()
        audio_stream.close()
        p.terminate()


async def stream_tokens_to_speaker(
    token_generator: AsyncGenerator[str, None],
    voice: str = "leo",
) -> str:
    """
    Stream LLM tokens directly to TTS WebSocket as they arrive.

    This enables true real-time streaming: audio starts playing
    while the LLM is still generating tokens.

    Args:
        token_generator: Async generator yielding LLM tokens
        voice: Voice ID (ara, rex, sal, eve, una, leo)

    Returns:
        The accumulated text that was spoken
    """
    import websockets

    # Try to import pyaudio
    try:
        import pyaudio
    except ImportError:
        print("Warning: PyAudio not available, audio will not play", file=sys.stderr)
        # Still consume the generator to get the text
        text = ""
        async for token in token_generator:
            text += token
        return text

    api_key = Config.XAI_API_KEY
    if not api_key:
        raise ValueError("XAI_API_KEY not found")

    # WebSocket TTS endpoint
    base_url = os.getenv("BASE_URL", "https://api.x.ai/v1")
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
    uri = f"{ws_url}/realtime/audio/speech"

    # Audio settings (24kHz, mono, 16-bit PCM)
    sample_rate = 24000
    channels = 1

    headers = {"Authorization": f"Bearer {api_key}"}

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    audio_stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        output=True,
    )

    accumulated_text = ""

    async def receive_and_play_audio(websocket):
        """Receive audio chunks and play them as they arrive."""
        while True:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                data = json.loads(response)

                # Handle different message types
                msg_type = data.get("type", "")
                if msg_type == "error":
                    error_msg = data.get("error", {}).get("message", "Unknown error")
                    print(f"  TTS API error: {error_msg}", file=sys.stderr)
                    break

                # Try to extract audio data
                try:
                    audio_data = data.get("data", {}).get("data", {})
                    audio_b64 = audio_data.get("audio")
                    is_last = audio_data.get("is_last", False)

                    if not audio_b64:
                        # Not an audio message, skip
                        continue

                    # Decode and play immediately
                    chunk_bytes = base64.b64decode(audio_b64)
                    if len(chunk_bytes) > 0:
                        await asyncio.to_thread(audio_stream.write, chunk_bytes)

                    if is_last:
                        break
                except (KeyError, TypeError):
                    # Not an audio message format, skip
                    continue

            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosedOK:
                break
            except websockets.exceptions.ConnectionClosedError:
                break
            except Exception as e:
                print(f"  Audio recv error: {e}", file=sys.stderr)
                break

    audio_task = None
    websocket = None

    try:
        websocket = await websockets.connect(uri, additional_headers=headers)

        # 1. Send config
        config_message = {"type": "config", "data": {"voice_id": voice}}
        await websocket.send(json.dumps(config_message))

        # 2. Start audio receiver task (plays audio as it arrives)
        audio_task = asyncio.create_task(receive_and_play_audio(websocket))

        # 3. Stream tokens as they come from LLM
        async for token in token_generator:
            accumulated_text += token
            # Send each token to TTS immediately
            text_message = {
                "type": "text_chunk",
                "data": {"text": token, "is_last": False},
            }
            try:
                await websocket.send(json.dumps(text_message))
            except Exception as e:
                print(f"  Token send error: {e}", file=sys.stderr)
                break

        # 4. Signal end of text
        end_message = {
            "type": "text_chunk",
            "data": {"text": "", "is_last": True},
        }
        await websocket.send(json.dumps(end_message))

        # 5. Wait for audio to finish playing
        await audio_task

    except asyncio.CancelledError:
        # Interrupted by a new event - clean up gracefully
        raise  # Re-raise so caller knows it was cancelled

    except Exception as e:
        print(f"Streaming TTS error: {e}", file=sys.stderr)

    finally:
        # Cancel audio task if still running
        if audio_task and not audio_task.done():
            audio_task.cancel()
            try:
                await audio_task
            except asyncio.CancelledError:
                pass

        # Close websocket
        if websocket:
            try:
                await websocket.close()
            except Exception:
                pass

        # Clean up PyAudio
        audio_stream.stop_stream()
        audio_stream.close()
        p.terminate()

    return accumulated_text


# =============================================================================
# Streaming Pipeline (LLM Tokens -> TTS Audio Chunks -> Speaker)
# =============================================================================


async def process_events_streaming(
    events: List[Dict],
    agent: NBACommentaryAgent,
    speed_multiplier: float = 1.0,
) -> None:
    """
    Process events with full streaming pipeline.

    Flow: NBA Event -> Grok LLM (token stream) -> TTS (audio chunks) -> Speaker
    No file storage - everything streams directly to speaker.

    Args:
        events: List of NBA events
        agent: NBACommentaryAgent instance
        speed_multiplier: Speed multiplier for event timing
    """
    print(
        f"🚀 Starting Streaming Pipeline (Speed: {speed_multiplier}x)",
        file=sys.stderr,
    )
    print("   LLM tokens -> TTS chunks -> Speaker (no file storage)", file=sys.stderr)

    voices = ["leo", "ara", "rex"]
    voice_idx = 0
    last_event_time = 0.0

    for i, event in enumerate(events):
        event_time = event.get("timeActual", 0)
        description = event.get("description", "")[:50]

        # Pacing: wait based on event timing
        wait_time = (event_time - last_event_time) / speed_multiplier
        if wait_time > 0 and i > 0:
            await asyncio.sleep(wait_time)
        last_event_time = event_time

        print(f"\n[{event_time:.1f}s] {description}...", file=sys.stderr)

        # 1. Stream tokens from Grok LLM
        text = ""
        print("  LLM: ", end="", file=sys.stderr)
        try:
            async for token in agent.process_event_streaming(event):
                text += token
                print(token, end="", flush=True, file=sys.stderr)
            print("", file=sys.stderr)  # newline
        except Exception as e:
            print(f"\n  LLM Error: {e}", file=sys.stderr)
            continue

        # 2. Sanitize text
        clean_text = sanitize_text(text)
        if not clean_text:
            print("  (empty after sanitize, skipping)", file=sys.stderr)
            continue

        # 3. Stream to speaker via WebSocket TTS
        current_voice = voices[voice_idx % len(voices)]
        voice_idx += 1

        print(f"  TTS [{current_voice}]: streaming...", file=sys.stderr)
        try:
            await stream_to_speaker(clean_text, voice=current_voice)
            print(f"  Done.", file=sys.stderr)
        except Exception as e:
            print(f"  TTS Error: {e}", file=sys.stderr)

    print("\n✅ Streaming pipeline complete.", file=sys.stderr)


# =============================================================================
# Pipeline Logic (Batch Mode - Legacy)
# =============================================================================


def merge_close_events(events: List[Dict], threshold: float = 3.0) -> List[Dict]:
    """Merges events within threshold seconds."""
    if not events:
        return []
    merged = []
    batch = [events[0]]
    for i in range(1, len(events)):
        curr = events[i]
        prev = batch[-1]
        if (curr.get("timeActual", 0) - batch[0].get("timeActual", 0)) < threshold:
            batch.append(curr)
        else:
            merged.append(_create_merged_event(batch))
            batch = [curr]
    if batch:
        merged.append(_create_merged_event(batch))
    print(f"📉 Optimized: {len(events)} -> {len(merged)} sequences.", file=sys.stderr)
    return merged


def _create_merged_event(batch: List[Dict]) -> Dict:
    if len(batch) == 1:
        return batch[0]
    final = batch[-1].copy()
    descs = [e.get("description", "") for e in batch if e.get("description")]
    final["description"] = " -> ".join(descs)
    return final


async def process_events_pipelined(
    events: List[Dict], agent: NBACommentaryAgent, speed_multiplier: float = 1.0
):
    import asyncio
    from tts import text_to_speech_excited
    import subprocess
    import time

    script_queue = asyncio.Queue()
    audio_dir = Path(__file__).parent.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"🚀 Starting Commentary Pipeline (Speed: {speed_multiplier}x)", file=sys.stderr
    )

    async def voice_worker():
        while True:
            item = await script_queue.get()
            if item is None:
                script_queue.task_done()
                break

            text, voice, event_id, excitement = item
            clean_text = sanitize_text(text)

            if not clean_text:
                script_queue.task_done()
                continue

            filename = f"seq_{event_id}_{int(time.time())}.mp3"
            path = str(audio_dir / filename)

            try:
                print(f"🎵 Gen: {clean_text[:40]}...", file=sys.stderr)
                await asyncio.to_thread(
                    text_to_speech_excited,
                    text=clean_text,
                    voice=voice.capitalize(),
                    output_file=path,
                    excitement_level=excitement,
                )
                print(f"🔊 Speaking: {clean_text[:40]}...", file=sys.stderr)
                # Playing fast to catch up
                await asyncio.to_thread(
                    subprocess.run, ["afplay", "--rate", "1.3", path], check=False
                )
            except Exception as e:
                print(f"❌ Audio Error: {e}", file=sys.stderr)

            script_queue.task_done()

    voice_task = asyncio.create_task(voice_worker())

    last_event_time = 0.0
    voices = ["Leo", "Ara", "Rex"]
    voice_idx = 0

    for i, event in enumerate(events):
        event_time = event.get("timeActual", 0)

        if i + 1 < len(events):
            next_time = events[i + 1].get("timeActual", 0)
            time_budget = next_time - event_time
        else:
            time_budget = 10.0

        safe_token_limit = 25
        wait_time = (event_time - last_event_time) / speed_multiplier
        if wait_time > 0 and i > 0:
            await asyncio.sleep(wait_time)
        last_event_time = event_time

        if script_queue.qsize() >= 1 and time_budget < 5.0:
            print(f"⏩ Skipping tight window ({time_budget:.1f}s)...", file=sys.stderr)
            continue

        print(f"📝 [Time: {event_time:.1f}] Generating...", file=sys.stderr)

        try:
            commentary = await asyncio.wait_for(
                asyncio.to_thread(
                    agent.process_event,
                    event,
                    time_budget=time_budget,
                    max_tokens=safe_token_limit,
                ),
                timeout=10.0,
            )
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

    parser = argparse.ArgumentParser(
        description="NBA Commentary Generator with Streaming Support"
    )
    parser.add_argument("--input", required=True, help="Input JSON file with events")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Streaming mode: LLM tokens -> TTS chunks -> Speaker (no file storage)",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Legacy pipeline mode: batch LLM -> file TTS -> play",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier")
    parser.add_argument(
        "--language", type=str, default="English", help="Commentary language"
    )
    parser.add_argument(
        "--team_support", type=str, default="Neither", help="Team to support"
    )

    args = parser.parse_args()

    print(f"📂 Loading {args.input}...", file=sys.stderr)
    with open(args.input) as f:
        events = json.load(f)

    print(
        f"ℹ️  Language: {args.language} | Support: {args.team_support}",
        file=sys.stderr,
    )
    agent = NBACommentaryAgent(language=args.language, team_support=args.team_support)

    # Merge events for all modes
    optimized = merge_close_events(events, threshold=3.0)

    if args.stream:
        # Streaming mode: LLM tokens -> TTS chunks -> Speaker
        await process_events_streaming(optimized, agent, args.speed)
    elif args.pipeline:
        # Legacy pipeline mode: batch processing with file storage
        await process_events_pipelined(optimized, agent, args.speed)
    else:
        # Default to streaming mode
        print("Using streaming mode (use --pipeline for legacy mode)", file=sys.stderr)
        await process_events_streaming(optimized, agent, args.speed)


if __name__ == "__main__":
    asyncio.run(main_async())
