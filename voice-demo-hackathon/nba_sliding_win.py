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


    # # input_text = "The home team is getting ready to shoot a free throw. The crowd is silent in anticipation."
    # text = "Jason missed the first free throw, it looks not good for the home team now."
    # voice_file = "output/example1_dense_1.mp3"
    # voice_base64 = file_to_base64(voice_file)
    # i = 1
    # output_file = "output/example1_dense_2.mp3"

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
        text = entry.get("line", "")  # Changed from "text"
        
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



async def process_json_from_file(file_path: str):
    """
    Process JSON from a file.

    Args:
        file_path: Path to JSON file
    """
    try:
        with open(file_path, "r") as f:
            json_data = json.load(f)
            print(f"## json_data {json_data }")
        await process_narration_json(json_data)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in file: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")



def main():
    """
    Demo examples showing different use cases.
    """
    print("=" * 60)
    print("Text-to-Speech API Demo")
    print("=" * 60)

    print("\n📝 Example 1: Simple POST request")
    print("-" * 60)
    # tts_request(
    #     text="This is a POST request example with voice cloning.",
    #     output_file="example1_arnold.mp3",
    #     voice_file="voices/arnold.m4a",
    # )

    asyncio.run(process_json_from_file("input/input_sentences.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure to:")
        print("1. Update BASE_URL with your actual domain")
        print("2. Install requests: pip install requests")
        print("3. Ensure the API endpoint is accessible")
