#!/usr/bin/env python3
"""
playlist.py

Simple script to play MP3 files from an output folder sequentially.

Usage examples:
  # Dry-run (list files that would be played)
  python playlist.py --dir ./output --dry-run

  # Play files in order
  python playlist.py --dir ./output

  # Shuffle and loop forever
  python playlist.py --dir ./output --shuffle --loop

The script tries to select a sensible player depending on the OS:
  - macOS: afplay
  - Linux: mpg123, mpv, ffplay (first available)
  - Windows: attempts to open with the default application (best-effort)

This is intentionally lightweight and uses subprocess to call system players so
no extra Python dependencies are required.
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def find_player() -> Optional[List[str]]:
    """Return a command (list) that can play an mp3 file, or None if none found."""
    import platform

    system = platform.system()
    # macOS: afplay is available by default
    if system == "Darwin":
        if shutil.which("afplay"):
            return ["afplay"]

    # Linux: prefer mpg123, then mpv, then ffplay
    if system == "Linux":
        for cmd in ("mpg123", "mpv", "ffplay"):
            if shutil.which(cmd):
                # ffplay needs args to auto-exit
                if cmd == "ffplay":
                    return ["ffplay", "-nodisp", "-autoexit"]
                return [cmd]

    # Windows or fallback: try to use the 'start' / 'open' behavior by invoking the
    # default application. These commands are not blocking in the same way as a
    # player, but are a reasonable fallback.
    if system == "Windows":
        # Use powershell Start-Process which doesn't block; user can change player
        # If user has 'mpv' or another player installed, prefer that
        for cmd in ("mpv", "mpg123", "ffplay"):
            if shutil.which(cmd):
                if cmd == "ffplay":
                    return ["ffplay", "-nodisp", "-autoexit"]
                return [cmd]
        # fallback to powershell open
        if shutil.which("powershell"):
            return ["powershell", "-Command", "Start-Process"]

    # Generic 'open' on macOS (if afplay not present) or 'xdg-open' on Linux
    if shutil.which("open"):
        return ["open"]
    if shutil.which("xdg-open"):
        return ["xdg-open"]

    return None


def collect_mp3_files(directory: Path) -> List[Path]:
    files = [p for p in sorted(directory.glob("**/*.mp3")) if p.is_file()]
    return files


def play_file(player: Optional[List[str]], file_path: Path) -> None:
    print(f"🔊 Playing: {file_path}")
    if player is None:
        print("⚠️  No player found on this system. Cannot play audio.")
        return

    # If player is a wrapper with extra args, append the file path
    cmd = player + [str(file_path)]

    try:
        # On some systems (open/xdg-open) we want shell=False but let the system
        # open the default app. We block until the command returns to play files
        # sequentially.
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"❌ Failed to play {file_path}: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Play mp3 files from a folder sequentially")
    parser.add_argument("--dir", "-d", default=None, help="Directory containing mp3 files (default: ./output relative to script)")
    parser.add_argument("--shuffle", "-s", action="store_true", help="Shuffle playlist")
    parser.add_argument("--loop", "-l", action="store_true", help="Loop playlist until interrupted")
    parser.add_argument("--dry-run", action="store_true", help="Only list files that would be played")
    args = parser.parse_args(argv)

    script_dir = Path(__file__).parent
    default_dir = script_dir / "output"
    directory = Path(args.dir) if args.dir else default_dir

    if not directory.exists() or not directory.is_dir():
        print(f"❌ Directory not found: {directory}")
        return 2

    files = collect_mp3_files(directory)
    if not files:
        print(f"⚠️  No mp3 files found in {directory}")
        return 0

    if args.shuffle:
        random.shuffle(files)

    print(f"🎧 Found {len(files)} mp3 files in: {directory}")
    for i, f in enumerate(files, start=1):
        print(f"  {i:02d}. {f.name}")

    if args.dry_run:
        print("(dry-run) Done — no audio played")
        return 0

    player = find_player()
    if player:
        print(f"Using player: {' '.join(player)}")
    else:
        print("⚠️  No known audio player found. Attempting to use system opener (may not block).")

    try:
        while True:
            for f in files:
                play_file(player, f)

            if not args.loop:
                break
            # shuffle again on loop if requested
            if args.shuffle:
                random.shuffle(files)

    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
