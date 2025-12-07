#!/usr/bin/env python3
"""
Demo script for calling the Text-to-Speech API endpoints.

This script demonstrates POST methods for generating
speech from text using the TTS API.
"""
import asyncio
import base64
import requests
import os
from dotenv import load_dotenv
import asyncio
import json
import sys
import os
import subprocess
import tempfile
from typing import Dict, List
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("XAI_API_KEY")

BASE_URL = "https://us-east-4.api.x.ai/voice-staging"
ENDPOINT = f"{BASE_URL}/api/v1/text-to-speech/generate"

MAX_INPUT_LENGTH = 4096
MAX_PROMPT_LENGTH = 4096


def file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def tts_request(
    text: str,
    prompt: str = "",
    # vibe: str = "audio",
    voice_file: str | None = None,
    output_file: str = "output.mp3",
):
    print(f"## text {text }")
    print(f"## prompt {prompt }")
    print(f"## voice_file {voice_file }")
    print(f"## output_file {output_file }")

    if voice_file is not None:
        voice_base64 = file_to_base64(voice_file)
    else:
        voice_base64 = None

    text = text[:MAX_INPUT_LENGTH]
    prompt = prompt[:MAX_PROMPT_LENGTH]


    payload = {
        "model": "grok-voice",
        "input": text,
        "response_format": "mp3",
        "instructions": prompt,
        "voice": voice_base64,
        "sampling_params": {
            "max_new_tokens": 512,
            "temperature": 1.0,
            "min_p": 0.01,
        },
    }

    print(f"Making POST request to {ENDPOINT}")
    print(f"Payload: {payload}")

    response = requests.post(ENDPOINT, json=payload, stream=True, headers={"Authorization": f"Bearer {API_KEY}"})

    if response.status_code == 200:
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Audio saved to {output_file}")
        return output_file
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None


async def process_narration_entry(entry: Dict, entry_index: int):
    try:
        voice = entry.get("name", "ara")  # Changed from "voice"
        text = entry.get("commentary", "")  # Changed from "text"
        print(f"text : {text }")
        
        prompt = """You're a NBA game commentator who is enthusiastic and energetic. You are commentating a live basketball game with provided script. 
            Please read the text, in a flow consistent with your earlier commentary as in the audio provided.
            Make sure you stick to the provided script text.
            """
        if entry_index == 0:
            voice_file = "voices/steve-jobs.m4a"
        else:
            voice_file = "voices/steve-jobs.m4a"

        fixed_length = 4
        formatted_index = f"{entry_index:0{fixed_length}d}"
        output_file = f"output/output_{formatted_index}.mp3"

        saved_path = await asyncio.to_thread(
            tts_request,
            text=text,
            prompt=prompt,
            voice_file=voice_file,
            output_file=output_file,
        )
        print(f"💾 Saved to: {saved_path}")

        # Play immediately using system audio player (non-blocking)
        if saved_path:
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
        narration = json_data.get("commentators", [])
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



async def process_json_files_from_directory(text_dir: str, output_dir: str, voice_file: str) -> None:
    """
    Process all JSON files from a directory.
    Each file is expected to be a response object with a 'commentary' field.

    Args:
        text_dir: Path to directory containing response_*.json files
        output_dir: Directory to save generated audio files
        voice_file: Path to voice reference file
    """
    text_path = Path(text_dir)
    output_path = Path(output_dir)
    
    if not text_path.exists():
        print(f"❌ Directory not found: {text_path}")
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all response JSON files, sorted by name
    json_files = sorted(text_path.glob("response_*.json"))
    
    if not json_files:
        print(f"⚠️  No response_*.json files found in {text_path}")
        return
    
    print(f"\n{'=' * 70}")
    print(f"🎙️  Processing {len(json_files)} commentary files")
    print(f"📁 Input: {text_path}")
    print(f"📁 Output: {output_path}")
    print(f"🎤 Voice: {voice_file}")
    print(f"{'=' * 70}")
    
    # Track success/failure
    success_count = 0
    failed_count = 0
    
    # Process each file sequentially
    for idx, json_file in enumerate(json_files):
        try:
            print(f"\n📄 File {idx + 1}/{len(json_files)}: {json_file.name}")
            
            # Load the response JSON
            with open(json_file, "r") as f:
                json_data = json.load(f)
            
            # Extract commentary from the response
            commentary = json_data.get("commentary", "")
            
            # Clean up the commentary (remove extra quotes if present)
            if commentary.startswith('"') and commentary.endswith('"'):
                commentary = commentary[1:-1]
            
            if not commentary:
                print(f"⚠️  No commentary text found, skipping")
                failed_count += 1
                continue
            
            text = commentary
            print(f"📝 Commentary: {text[:80]}...")
            
            # Prompt for the voice model - consistent with TTS instructions
            prompt = """You're a NBA game commentator who is enthusiastic and energetic. You are commentating a live basketball game with provided script. 
            Please read the text, in a flow consistent with your earlier commentary as in the audio provided.
            Make sure you stick to the provided script text.
            """
            
            # Generate output filename
            fixed_length = 4
            formatted_index = f"{idx:0{fixed_length}d}"
            output_file = output_path / f"output_{formatted_index}.mp3"
            
            # Call TTS API in a thread to avoid blocking
            saved_path = await asyncio.to_thread(
                tts_request,
                text=text,
                prompt=prompt,
                voice_file=voice_file,
                output_file=str(output_file),
            )
            
            if saved_path:
                print(f"💾 Saved to: {saved_path}")
                
                # Play immediately using system audio player (non-blocking)
                print(f"🔊 Playing audio...")
                await asyncio.to_thread(
                    subprocess.run,
                    ["afplay", saved_path],
                    check=False,
                )
                success_count += 1
            else:
                print(f"⚠️  Failed to generate audio")
                failed_count += 1
        
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {json_file.name}: {e}")
            failed_count += 1
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")
            failed_count += 1
        
        # Small pause between files
        if idx < len(json_files) - 1:
            await asyncio.sleep(0.5)
    
    print(f"\n{'=' * 70}")
    print(f"✅ Successfully processed: {success_count}/{len(json_files)}")
    if failed_count > 0:
        print(f"❌ Failed: {failed_count}")
    print(f"{'=' * 70}")



def main():
    """
    Demo examples showing different use cases.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate audio from NBA game commentary using Grok Voice TTS"
    )
    parser.add_argument(
        "--text-dir",
        "-t",
        default="text",
        help="Directory containing response_*.json files (default: text)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Directory to save generated audio files (default: output)"
    )
    parser.add_argument(
        "--voice",
        "-v",
        default="voices/steve-jobs.m4a",
        help="Path to voice reference file (default: voices/steve-jobs.m4a)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NBA Audio Generation - Grok Voice TTS")
    print("=" * 70)
    
    # Check if voice file exists
    voice_path = Path(args.voice)
    if not voice_path.exists():
        print(f"❌ Voice file not found: {args.voice}")
        return 1
    
    # Check if API key is set
    if not API_KEY:
        print("❌ XAI_API_KEY environment variable not set")
        return 1
    
    # Run the async function
    asyncio.run(process_json_files_from_directory(
        args.text_dir,
        args.output_dir,
        args.voice
    ))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure to:")
        print("1. Update BASE_URL with your actual domain")
        print("2. Install requests: pip install requests")
        print("3. Ensure the API endpoint is accessible")
