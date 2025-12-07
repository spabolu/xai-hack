#!/usr/bin/env python3
"""
Real-Time TTS Processor - Process JSON narration and play immediately
"""

import asyncio
import json
import sys
import os
import subprocess
import tempfile
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the same directory as this script
script_dir = Path(__file__).parent
env_path = script_dir / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Import text enhancement for excitement/tone
from tts import add_excitement, text_to_speech_excited


async def process_narration_entry(entry: Dict, entry_index: int):
    """
    Process a single narration entry and play TTS immediately.

    Args:
        entry: Narration entry dict with commentary, commentator_name, excitement_level, tone
        entry_index: Index in the narration array
    """
    # Map new field names
    voice = entry.get("commentator_name", "ara")  # Changed from "voice"
    text = entry.get("commentary", "")  # Changed from "text"
    excitement_level = entry.get(
        "excitement_level", 5
    )  # Changed from "excitement", numeric 1-10
    tone = entry.get("tone", "neutral")
    game_context = entry.get("game_context", {})  # New field (optional metadata)

    # Convert numeric excitement_level (1-10) to string format ("low", "medium", "high")
    if excitement_level <= 3:
        excitement = "low"
    elif excitement_level <= 7:
        excitement = "medium"
    else:  # 8-10
        excitement = "high"

    # Capitalize first letter for standard API (Ara, Eve, etc.)
    voice_capitalized = voice.capitalize()

    # Detect language from text (optional - defaults to english)
    # You can enhance this with language detection if needed
    language = "spanish" if any(char in text for char in "¡¿áéíóúñ") else "english"

    if not text:
        print(f"⚠️  Entry {entry_index + 1}: Empty commentary, skipping")
        return

    # Display game context if available
    context_info = ""
    if game_context:
        home = game_context.get("home_team", "")
        away = game_context.get("away_team", "")
        score = (
            f"{game_context.get('home_score', 0)}-{game_context.get('away_score', 0)}"
        )
        quarter = game_context.get("quarter", "")
        time = game_context.get("time", "")
        context_info = f" | {home} vs {away} | Score: {score} | Q{quarter} {time}"

    print(f"\n🎤 Processing entry {entry_index + 1}:")
    print(f"   Commentator: {voice_capitalized}")
    print(f"   Commentary: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"   Excitement: {excitement} (level {excitement_level}){context_info}")
    print(f"   Tone: {tone}, Language: {language}")

    try:
        # Set up output directory (same as tts.py)
        output_dir = Path(__file__).parent.parent / "audio"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate meaningful filename
        import time

        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        filename = f"{voice.lower()}_{excitement}_{timestamp}.mp3"
        file_path = output_dir / filename

        # Generate TTS using standard API (run in thread to avoid blocking)
        print(f"🎵 Generating audio...")
        saved_path = await asyncio.to_thread(
            text_to_speech_excited,
            text=text,
            voice=voice_capitalized,
            response_format="mp3",
            output_file=str(file_path),
            excitement_level=excitement,
        )

        print(f"💾 Saved to: {saved_path}")

        # Play immediately using system audio player (non-blocking)
        print(f"🔊 Playing audio...")
        await asyncio.to_thread(
            subprocess.run,
            ["afplay", saved_path],
            check=False,
        )

    except Exception as e:
        print(f"❌ Error processing entry {entry_index + 1}: {e}")

async def process_narration_json(json_data):
    """
    Process complete narration JSON and play all entries in sequence.

    Args:
        json_data: JSON array of narration entries OR dict with "narration" key (backward compatible)
    """
    # Handle both new format (array) and old format (object with "narration" key)
    if isinstance(json_data, list):
        narration = json_data
    elif isinstance(json_data, dict):
        narration = json_data.get("narration", [])
    else:
        print("⚠️  Invalid JSON format: expected array or object with 'narration' key")
        return

    if not narration:
        print("⚠️  No narration entries found in JSON")
        return

    print(f"\n{'=' * 60}")
    print(f"📝 Processing {len(narration)} narration entries")
    print(f"{'=' * 60}")

    # Process each entry sequentially (wait for one to finish before next)
    for i, entry in enumerate(narration):
        await process_narration_entry(entry, i)

        # Small pause between speakers (if multiple)
        if i < len(narration) - 1:
            await asyncio.sleep(0.1)  # 100ms pause between speakers

    print(f"\n✅ Finished processing all narration entries")

async def process_json_from_stdin():
    """Read JSON from stdin and process in real-time."""
    print("🎧 Real-Time TTS Processor - Waiting for JSON input...")
    print("   (Send JSON via stdin, press Ctrl+D when done)")
    print()

    try:
        # Read all input at once (works for complete JSON objects)
        input_data = await asyncio.to_thread(sys.stdin.read)

        if input_data.strip():
            json_data = json.loads(input_data)
            await process_narration_json(json_data)
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")


async def process_json_from_string(json_string: str):
    """
    Process JSON from a string (useful for API/webhook integration).

    Args:
        json_string: JSON string to process
    """
    try:
        json_data = json.loads(json_string)
        await process_narration_json(json_data)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
    except Exception as e:
        print(f"❌ Error processing JSON: {e}")


async def process_json_from_file(file_path: str):
    """
    Process JSON from a file.

    Args:
        file_path: Path to JSON file
    """
    try:
        with open(file_path, "r") as f:
            json_data = json.load(f)
        await process_narration_json(json_data)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in file: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main entry point - supports multiple input methods."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-Time TTS Processor - Process JSON narration and play immediately",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Read from stdin (pipe JSON)
  echo '{"narration": [{"voice": "ara", "text": "Hello!", "excitement": "high"}]}' | python realtime_tts.py
  
  # Read from file
  python realtime_tts.py --file narration.json
  
  # Read from stdin interactively
  python realtime_tts.py
  
Input JSON format:
{
  "narration": [
    {
      "voice": "ara",
      "text": "Commentary text",
      "excitement": "high",
      "tone": "excited",
      "language": "english"
    }
  ]
}
        """,
    )

    parser.add_argument("--file", "-f", help="Read JSON from file instead of stdin")
    parser.add_argument("--json", "-j", help="Process JSON string directly")

    args = parser.parse_args()

    try:
        if args.file:
            asyncio.run(process_json_from_file(args.file))
        elif args.json:
            asyncio.run(process_json_from_string(args.json))
        else:
            # Default: read from stdin
            asyncio.run(process_json_from_stdin())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()