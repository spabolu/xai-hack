#!/usr/bin/env python3
"""
Real-Time TTS Processor - Process JSON narration and play immediately
"""

import asyncio
import json
import sys
import os
import subprocess
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the same directory as this script
script_dir = Path(__file__).parent
env_path = script_dir / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Import text enhancement for excitement/tone
from tts import add_excitement, text_to_speech_excited


async def process_narration_entry(
    commentator_entry: Dict, entry_index: int, game_context: Dict = None
):
    """
    Process a single commentator entry and play TTS immediately.

    Args:
        commentator_entry: Commentator dict with name, line, tone, excitement_level, role
        entry_index: Index in the narration array
        game_context: Optional game context dict
    """
    # Map field names from new JSON structure
    voice = commentator_entry.get("name", "ara")
    text = commentator_entry.get("line", "")  # Changed from "commentary"
    tone = commentator_entry.get("tone", "neutral")  # Can be null, default to "neutral"
    excitement_level_str = commentator_entry.get(
        "excitement_level", "medium"
    )  # String format
    role = commentator_entry.get("role", "play_by_play")  # New field

    # Convert string excitement_level to numeric for processing, then back to string
    excitement_map = {"low": "low", "medium": "medium", "high": "high"}
    excitement = excitement_map.get(excitement_level_str.lower(), "medium")

    # Handle null tone - use role-based default if tone is null
    if tone is None:
        if role == "play_by_play":
            tone = "neutral"
        elif role == "color":
            tone = "analytical"
        else:
            tone = "neutral"

    # Capitalize first letter for standard API (Ara, Eve, etc.)
    voice_capitalized = voice.capitalize()

    # Detect language from text
    language = "spanish" if any(char in text for char in "¡¿áéíóúñ") else "english"

    if not text:
        print(f"⚠️  Entry {entry_index + 1}: Empty line, skipping")
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
        clock = game_context.get("clock", "")
        context_info = f" | {home} vs {away} | Score: {score} | {clock}"

    print(f"\n🎤 Processing entry {entry_index + 1}:")
    print(f"   Commentator: {voice_capitalized} ({role})")
    print(f"   Line: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"   Excitement: {excitement} | Tone: {tone}{context_info}")
    print(f"   Language: {language}")

    try:
        # Set up output directory
        output_dir = Path(__file__).parent.parent / "audio"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate meaningful filename
        import time

        timestamp = int(time.time() * 1000)
        filename = f"{voice.lower()}_{excitement}_{tone}_{timestamp}.mp3"
        file_path = output_dir / filename

        # Apply tone-based text enhancement (if needed)
        enhanced_text = text
        if tone and tone != "neutral":
            enhanced_text = apply_tone_enhancement(text, tone, excitement)

        # Generate TTS using standard API
        print(f"🎵 Generating audio...")
        saved_path = await asyncio.to_thread(
            text_to_speech_excited,
            text=enhanced_text,
            voice=voice_capitalized,
            response_format="mp3",
            output_file=str(file_path),
            excitement_level=excitement,
        )

        print(f"💾 Saved to: {saved_path}")

        # Play immediately
        print(f"🔊 Playing audio...")
        await asyncio.to_thread(
            subprocess.run,
            ["afplay", saved_path],
            check=False,
        )

    except Exception as e:
        print(f"❌ Error processing entry {entry_index + 1}: {e}")


def apply_tone_enhancement(text: str, tone: str, excitement: str) -> str:
    """
    Apply tone-based enhancements to text before TTS.

    Args:
        text: Original text
        tone: Tone value (excited, analytical, sarcastic, dramatic, etc.)
        excitement: Excitement level (low, medium, high)

    Returns:
        Enhanced text
    """
    # First apply excitement (this already adds punctuation)
    enhanced = add_excitement(text, level=excitement)

    # Then apply tone-specific enhancements
    if tone == "excited":
        # Already handled by excitement, but ensure high energy
        if excitement != "high":
            enhanced = f"Wow! {enhanced}"

    elif tone == "analytical":
        # Keep it more measured, less punctuation
        enhanced = text.replace("!!!", ".").replace("!!", ".")
        # Remove excitement interjections for analytical tone
        enhanced = enhanced.replace("Wow! ", "").replace("Oh my gosh! ", "")

    elif tone == "sarcastic":
        # Add subtle sarcasm markers
        if not any(c in enhanced for c in ["?", "!"]):
            enhanced = enhanced.rstrip(".") + "..."
        # Add "Well," prefix for sarcasm
        if not enhanced.startswith("Well"):
            enhanced = f"Well, {enhanced}"

    elif tone == "dramatic":
        # Add pauses and emphasis
        enhanced = enhanced.replace(". ", "... ").replace("! ", "...! ")
        # Add dramatic prefix
        if not enhanced.startswith(("Oh", "And", "The")):
            enhanced = f"And... {enhanced}"

    elif tone == "play_by_play":
        # Fast-paced, factual - remove excessive punctuation
        enhanced = text.replace("!!!", "!").replace("!!", "!")
        # Remove interjections
        enhanced = enhanced.replace("Wow! ", "").replace("Oh my gosh! ", "")

    elif tone == "color":
        # More descriptive, analytical - similar to analytical
        enhanced = text.replace("!!!", ".").replace("!!", ".")

    elif tone == "celebratory":
        # Add celebration markers
        if not enhanced.startswith(("What", "Incredible", "Amazing")):
            enhanced = f"What a moment! {enhanced}"

    elif tone == "concerned":
        # More measured, worried tone
        enhanced = text.replace("!!!", ".").replace("!!", ".")
        if not enhanced.startswith(("That", "Hope", "Looks")):
            enhanced = f"That's concerning. {enhanced}"

    elif tone == "casual":
        # Relaxed, conversational
        enhanced = text.replace("!!!", ".").replace("!!", ".")
        if not enhanced.startswith(("You know", "I mean", "Well")):
            enhanced = f"You know, {enhanced}"

    elif tone == "urgent":
        # Critical moments, add urgency
        enhanced = enhanced.replace(". ", "! ").replace(".", "!")
        if not enhanced.startswith(("They need", "Critical", "Now")):
            enhanced = f"Critical moment! {enhanced}"

    elif tone == "reflective":
        # Historical context, measured
        enhanced = text.replace("!!!", ".").replace("!!", ".")
        if not enhanced.startswith(("This reminds", "You remember", "Back in")):
            enhanced = f"This reminds me... {enhanced}"

    elif tone == "neutral":
        # Default, no special enhancements beyond excitement
        pass

    return enhanced

async def process_narration_json(json_data):
    """
    Process complete narration JSON and play all entries in sequence.

    Args:
        json_data: JSON array of narration entries with commentators array
    """
    # Handle both new format (array with commentators) and old format
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

    entry_counter = 0

    # Process each narration entry
    for narration_entry in narration:
        commentators = narration_entry.get("commentators", [])
        game_context = narration_entry.get("game_context", {})

        # Process each commentator in this narration entry
        for commentator in commentators:
            await process_narration_entry(commentator, entry_counter, game_context)
            entry_counter += 1

            # Small pause between commentators
            if entry_counter < len(
                [c for n in narration for c in n.get("commentators", [])]
            ):
                await asyncio.sleep(0.1)

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
