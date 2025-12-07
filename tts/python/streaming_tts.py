#!/usr/bin/env python3
"""
Streaming TTS using xAI WebSocket Real-time API

Provides near-instant audio playback by streaming audio chunks
as they're generated, instead of waiting for the full file.

Latency: ~200ms to first audio (vs 2-5 seconds with REST API)
"""

import os
import sys
import json
import base64
import asyncio
import struct
from pathlib import Path
from typing import Optional, Callable

try:
    import websockets
except ImportError:
    print("❌ websockets not installed. Run: pip install websockets")
    sys.exit(1)

try:
    import pyaudio
except ImportError:
    print("❌ pyaudio not installed. Run: pip install pyaudio")
    print("   On macOS: brew install portaudio && pip install pyaudio")
    sys.exit(1)

from dotenv import load_dotenv

# Load environment variables
script_dir = Path(__file__).parent
env_path = script_dir / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Configuration
XAI_API_KEY = os.getenv("XAI_API_KEY")
REALTIME_API_URL = os.getenv("REALTIME_API_URL", "wss://api.x.ai/v1/realtime")

# Audio settings for streaming
SAMPLE_RATE = 24000  # 24kHz is common for TTS
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit audio


class StreamingTTS:
    """
    WebSocket-based streaming TTS client for xAI.
    
    Plays audio chunks immediately as they arrive for low-latency playback.
    """
    
    def __init__(
        self,
        voice: str = "Ara",
        sample_rate: int = SAMPLE_RATE,
        on_audio_start: Optional[Callable] = None,
        on_audio_end: Optional[Callable] = None,
    ):
        """
        Initialize streaming TTS.
        
        Args:
            voice: Voice to use (Ara, Rex, Sal, Eve, Una, Leo)
            sample_rate: Audio sample rate (default: 24000)
            on_audio_start: Callback when audio starts playing
            on_audio_end: Callback when audio finishes
        """
        self.voice = voice
        self.sample_rate = sample_rate
        self.on_audio_start = on_audio_start
        self.on_audio_end = on_audio_end
        
        # Audio player
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
        # State
        self.is_playing = False
        self.total_bytes_played = 0
        
    def _init_audio_stream(self):
        """Initialize PyAudio stream for playback."""
        if self.stream is not None:
            self.stream.close()
            
        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=1024,
        )
        
    def _play_chunk(self, audio_data: bytes):
        """Play a chunk of audio data immediately."""
        if self.stream is None:
            self._init_audio_stream()
            
        if not self.is_playing:
            self.is_playing = True
            if self.on_audio_start:
                self.on_audio_start()
                
        self.stream.write(audio_data)
        self.total_bytes_played += len(audio_data)
        
    def _decode_base64_pcm(self, base64_data: str) -> bytes:
        """Decode base64-encoded PCM audio data."""
        return base64.b64decode(base64_data)
    
    async def speak(self, text: str, save_to_file: Optional[str] = None) -> float:
        """
        Stream TTS audio for the given text.
        
        Args:
            text: Text to convert to speech
            save_to_file: Optional path to save the complete audio
            
        Returns:
            Duration of audio played in seconds
        """
        if not XAI_API_KEY:
            raise ValueError("XAI_API_KEY not found in environment variables")
            
        self.total_bytes_played = 0
        self.is_playing = False
        audio_buffer = bytearray()
        
        print(f"🎙️  Streaming TTS: {text[:50]}...", file=sys.stderr)
        
        try:
            # Connect to WebSocket
            # Note: websockets library changed parameter name in v10+
            # Try both 'additional_headers' (new) and 'extra_headers' (old)
            headers = [("Authorization", f"Bearer {XAI_API_KEY}")]
            
            try:
                # websockets >= 10.0
                ws = await websockets.connect(
                    REALTIME_API_URL,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=20,
                )
            except TypeError:
                # websockets < 10.0
                ws = await websockets.connect(
                    REALTIME_API_URL,
                    extra_headers=headers,
                    ping_interval=20,
                    ping_timeout=20,
                )
            
            async with ws:
                
                # Configure session for audio output
                session_config = {
                    "type": "session.update",
                    "session": {
                        "voice": self.voice,
                        "modalities": ["audio", "text"],
                        "audio": {
                            "output": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": self.sample_rate
                                }
                            }
                        }
                    }
                }
                await ws.send(json.dumps(session_config))
                
                # Wait for session confirmation
                response = await ws.recv()
                event = json.loads(response)
                if event.get("type") == "error":
                    raise Exception(f"Session error: {event}")
                    
                # Send text for TTS
                tts_request = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"Please say the following text exactly: {text}"
                            }
                        ]
                    }
                }
                await ws.send(json.dumps(tts_request))
                
                # Request response
                await ws.send(json.dumps({"type": "response.create"}))
                
                # Listen for audio chunks
                first_chunk = True
                async for message in ws:
                    event = json.loads(message)
                    event_type = event.get("type", "")
                    
                    if event_type == "response.audio.delta":
                        # Decode and play audio chunk immediately
                        delta = event.get("delta", "")
                        if delta:
                            audio_chunk = self._decode_base64_pcm(delta)
                            
                            if first_chunk:
                                print(f"▶️  First audio chunk received!", file=sys.stderr)
                                first_chunk = False
                                
                            self._play_chunk(audio_chunk)
                            audio_buffer.extend(audio_chunk)
                            
                    elif event_type == "response.audio.done":
                        # Audio generation complete
                        break
                        
                    elif event_type == "response.done":
                        # Full response complete
                        break
                        
                    elif event_type == "error":
                        print(f"❌ Error: {event}", file=sys.stderr)
                        break
                        
        except websockets.exceptions.ConnectionClosed as e:
            print(f"⚠️  WebSocket closed: {e}", file=sys.stderr)
        except Exception as e:
            print(f"❌ Streaming error: {e}", file=sys.stderr)
            raise
        finally:
            self.is_playing = False
            if self.on_audio_end:
                self.on_audio_end()
                
        # Save to file if requested
        if save_to_file and audio_buffer:
            output_path = Path(save_to_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as raw PCM (can convert to WAV later if needed)
            with open(output_path, "wb") as f:
                f.write(audio_buffer)
            print(f"💾 Saved: {output_path}", file=sys.stderr)
            
        # Calculate duration
        duration = self.total_bytes_played / (self.sample_rate * SAMPLE_WIDTH * CHANNELS)
        print(f"✓  Played {duration:.2f}s of audio", file=sys.stderr)
        
        return duration
    
    def close(self):
        """Clean up audio resources."""
        if self.stream:
            self.stream.close()
        self.pyaudio.terminate()


class StreamingTTSSimple:
    """
    Simplified streaming TTS using REST API with streaming response.
    
    Falls back to this if WebSocket API is not available.
    Uses requests with stream=True for chunk-by-chunk download.
    """
    
    def __init__(self, voice: str = "Ara"):
        self.voice = voice
        self.api_url = os.getenv("BASE_URL", "https://api.x.ai/v1") + "/audio/speech"
        
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
    def _init_audio_stream(self, sample_rate: int = 24000):
        """Initialize PyAudio stream."""
        if self.stream:
            self.stream.close()
            
        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            frames_per_buffer=1024,
        )
        
    async def speak(self, text: str) -> float:
        """
        Stream TTS using REST API with streaming response.
        
        Note: This still has some initial latency but streams the download.
        """
        import requests
        
        if not XAI_API_KEY:
            raise ValueError("XAI_API_KEY not found")
            
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        }
        
        data = {
            "input": text,
            "voice": self.voice,
            "response_format": "pcm",  # Raw PCM for streaming
        }
        
        print(f"🎙️  Streaming TTS (REST): {text[:50]}...", file=sys.stderr)
        
        total_bytes = 0
        self._init_audio_stream()
        
        try:
            # Stream the response
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                stream=True,
            )
            response.raise_for_status()
            
            first_chunk = True
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    if first_chunk:
                        print(f"▶️  First chunk received!", file=sys.stderr)
                        first_chunk = False
                        
                    self.stream.write(chunk)
                    total_bytes += len(chunk)
                    
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            raise
            
        duration = total_bytes / (24000 * 2)  # Assuming 24kHz, 16-bit
        print(f"✓  Played {duration:.2f}s of audio", file=sys.stderr)
        
        return duration
        
    def close(self):
        """Clean up."""
        if self.stream:
            self.stream.close()
        self.pyaudio.terminate()


# Convenience function for quick streaming TTS
async def stream_speak(
    text: str,
    voice: str = "Ara",
    use_websocket: bool = True,
) -> float:
    """
    Quick function to stream TTS audio.
    
    Args:
        text: Text to speak
        voice: Voice to use
        use_websocket: Use WebSocket API (faster) or REST streaming
        
    Returns:
        Duration of audio played
    """
    if use_websocket:
        tts = StreamingTTS(voice=voice)
    else:
        tts = StreamingTTSSimple(voice=voice)
        
    try:
        return await tts.speak(text)
    finally:
        tts.close()


# Test function
async def main():
    """Test streaming TTS."""
    print("=" * 60)
    print("Streaming TTS Test")
    print("=" * 60)
    
    test_text = "Hello! This is a test of the streaming text-to-speech system. The audio should start playing almost immediately!"
    
    print("\n1. Testing WebSocket streaming...")
    try:
        duration = await stream_speak(test_text, voice="Leo", use_websocket=True)
        print(f"   WebSocket: {duration:.2f}s\n")
    except Exception as e:
        print(f"   WebSocket failed: {e}")
        print("   Falling back to REST streaming...\n")
        
        print("2. Testing REST streaming...")
        duration = await stream_speak(test_text, voice="Leo", use_websocket=False)
        print(f"   REST: {duration:.2f}s\n")
    
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
