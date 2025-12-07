#!/usr/bin/env python3
"""
Demo script for calling the Text-to-Speech API endpoints.

This script demonstrates POST methods for generating
speech from text using the TTS API.
"""

import base64
import requests
import os
from dotenv import load_dotenv

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
    input_text: str,
    prompt: str = "",
    vibe: str = "audio",
    voice_file: str | None = None,
    output_file: str = "output.mp3",
):

    print(f"API_KEY={API_KEY}")

    # if voice_file is not None:
    #     voice_base64 = file_to_base64(voice_file)
    # else:
    #     voice_base64 = None

    input_text = input_text[:MAX_INPUT_LENGTH]
    prompt = prompt[:MAX_PROMPT_LENGTH]


    # input_text = "The home team is getting ready to shoot a free throw. The crowd is silent in anticipation."
    input_text = "Jason missed the first free throw, it looks not good for the home team now."
    instructions = """
    You're a NBA game commentator who is enthusiastic and energetic. You are commentating a live basketball game with provided script. 
    Please read the text, in a flow consistent with your earlier commentary as in the audio provided.
    Make sure you stick to the provided script text.
    """
    voice_file = "output/example1_dense_1.mp3"
    voice_base64 = file_to_base64(voice_file)
    i = 1
    output_file = "output/example1_dense_2.mp3"

    payload = {
        "model": "grok-voice",
        "input": input_text,
        "response_format": "mp3",
        "instructions": instructions,
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


def main():
    """
    Demo examples showing different use cases.
    """
    print("=" * 60)
    print("Text-to-Speech API Demo")
    print("=" * 60)

    print("\n📝 Example 1: Simple POST request")
    print("-" * 60)
    tts_request(
        input_text="This is a POST request example with voice cloning.",
        output_file="example1_arnold.mp3",
        voice_file="voices/arnold.m4a",
    )

    # print("\n📝 Example 1: Simple POST request")
    # print("-" * 60)
    # tts_request(
    #     input_text="This is a POST request example with voice cloning.",
    #     output_file="example1_dense.mp3",
    #     vibe="black american male, aged 60-65",
    # )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure to:")
        print("1. Update BASE_URL with your actual domain")
        print("2. Install requests: pip install requests")
        print("3. Ensure the API endpoint is accessible")
