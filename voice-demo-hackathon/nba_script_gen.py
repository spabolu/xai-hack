#!/usr/bin/env python3
"""
nba_script_gen.py

Read `input/nba_actions.json`, parse play-by-play actions, group them by 5-second
time windows, send prompts to Grok API for commentary generation, and save
annotated results to JSON.

Usage:
  # Generate commentary for all events (requires XAI_API_KEY env var)
  python nba_script_gen.py --input input/nba_actions.json --output output/commentary.json

  # Dry-run: show prompts without calling API
  python nba_script_gen.py --input input/nba_actions.json --dry-run

Environment:
  XAI_API_KEY: Your XAI API key for Grok access

The script:
  1. Loads and sorts actions by orderNumber/timeActual
  2. Groups events by 5-second time windows (events within 5 seconds are combined)
  3. Builds natural-language prompts describing 1-2 events
  4. Calls Grok API to generate live sports commentary
  5. Appends results (events + commentary) to a JSON file
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
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



COMMON_KEYS = [
    "orderNumber",
    "actionNumber",
    "clock",
    "timeActual",
    "period",
    "periodType",
    "actionType",
    "subType",
    "teamTricode",
    "teamId",
    "personId",
    "playerName",
    "description",
    "scoreHome",
    "scoreAway",
    "x",
    "y",
    "shotResult",
    "pointsTotal",
    "assistPlayerNameInitial",
    "reboundTotal",
    "turnoverTotal",
    "qualifiers",
]


def load_actions(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        # some files wrap actions under a key (not the case here but be safe)
        # try to find the list value
        for v in data.values():
            if isinstance(v, list):
                return v
        raise ValueError("No actions list found in JSON file")
    if not isinstance(data, list):
        raise ValueError("Expected JSON array of actions")
    return data


def parse_time(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 timestamp string to datetime object."""
    if not ts:
        return None
    try:
        # handle fractional seconds like 2021-01-16T00:40:29.8Z
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def datetime_to_string(ts: Optional[datetime]) -> Optional[str]:
    """Convert datetime object to ISO string."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.isoformat()
    return str(ts)


def extract_action(a: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in COMMON_KEYS:
        out[k] = a.get(k)

    # Normalize scores: keep as ints when possible
    try:
        if out.get("scoreHome") is not None:
            out["scoreHome"] = int(out["scoreHome"])
    except Exception:
        pass
    try:
        if out.get("scoreAway") is not None:
            out["scoreAway"] = int(out["scoreAway"])
    except Exception:
        pass

    # Normalize numeric totals
    for key in ("pointsTotal", "reboundTotal", "turnoverTotal"):
        v = out.get(key)
        if v is None:
            continue
        try:
            out[key] = int(v) if float(v).is_integer() else float(v)
        except Exception:
            out[key] = v

    # Normalize time
    out["timeActual"] = parse_time(out.get("timeActual"))

    # Keep qualifiers as list
    q = out.get("qualifiers")
    out["qualifiers"] = q if isinstance(q, list) else []

    return out


def group_events_by_time(actions: List[Dict[str, Any]], window_seconds: int = 5) -> List[List[Dict[str, Any]]]:
    """
    Group actions into clusters where each cluster contains events within
    `window_seconds` of the first event in the cluster.
    """
    if not actions:
        return []

    groups: List[List[Dict[str, Any]]] = []
    current_group: List[Dict[str, Any]] = []
    current_group_time: Optional[datetime] = None

    for action in actions:
        ts = action.get("timeActual")
        if isinstance(ts, str):
            ts = parse_time(ts)

        if ts is None:
            # If no timestamp, add to current group anyway
            current_group.append(action)
            continue

        if current_group_time is None:
            # First action in a group
            current_group = [action]
            current_group_time = ts
        elif (ts - current_group_time).total_seconds() <= window_seconds:
            # Within time window, add to current group
            current_group.append(action)
        else:
            # Outside time window, start new group
            if current_group:
                groups.append(current_group)
            current_group = [action]
            current_group_time = ts

    if current_group:
        groups.append(current_group)

    return groups


def build_prompt(group: List[Dict[str, Any]]) -> str:
    """Build a natural-language prompt from a group of actions for Grok."""
    texts = []
    for action in group:
        desc = action.get("description", "Unknown event")
        period = action.get("period", "")
        score_h = action.get("scoreHome")
        score_a = action.get("scoreAway")
        clock = action.get("clock", "")

        # Format time
        time_str = f"{clock}" if clock else ""
        if period:
            time_str += f" P{period}" if time_str else f"Period {period}"

        # Format score
        score_str = ""
        if score_h is not None and score_a is not None:
            score_str = f" (Score: {score_h}-{score_a})"

        texts.append(f"{desc}{score_str}{(' at ' + time_str) if time_str else ''}")

    events_text = "\n".join(texts)

    prompt = f"""You are an enthusiastic NBA play-by-play commentator. Generate a short, exciting live commentary 
for the following game event(s). Be energetic, mention player names when available, and focus on the action. 
Keep the commentary to 1-2 sentences and suitable for audio playback.

Events:
{events_text}

Commentary:"""

    return prompt


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def call_grok_api(prompt: str, dry_run: bool = False) -> str:
    """Call Grok API to generate commentary. Returns the generated text."""
    if dry_run:
        return "[DRY-RUN] Commentary would be generated here"

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY environment variable not set")

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "grok-2",
        "messages": [
            {"role": "system", "content": "You are an enthusiastic NBA commentator."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 150,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Extract message from response
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return "[ERROR] No content in response"

    except requests.exceptions.RequestException as e:
        return f"[ERROR] API call failed: {e}"
    except Exception as e:
        return f"[ERROR] {e}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate NBA game commentary using Grok API")
    parser.add_argument("--input", "-i", default="input/nba_actions.json", help="Path to nba_actions.json")
    parser.add_argument("--output", "-o", default="script_from_event/commentary.json", help="Output JSON file (combined results)")
    parser.add_argument("--text-dir", "-t", default="text", help="Directory to save individual response files")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts without calling API")
    args = parser.parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        print(f"❌ Input file not found: {inp}")
        return 2

    # Create text directory for individual responses
    text_dir = Path(args.text_dir)
    text_dir.mkdir(parents=True, exist_ok=True)

    # Load and sort actions
    actions_raw = load_actions(inp)
    def sort_key(x: Dict[str, Any]):
        if x.get("orderNumber") is not None:
            return int(x.get("orderNumber"))
        t = x.get("timeActual")
        if t:
            try:
                return datetime.fromisoformat(t.replace("Z", "+00:00")) if isinstance(t, str) else t
            except Exception:
                return t
        return 0

    actions_sorted = sorted(actions_raw, key=sort_key)

    # Extract and normalize actions
    extracted: List[Dict[str, Any]] = []
    for a in actions_sorted:
        ea = extract_action(a)
        extracted.append(ea)

    # Group by 5-second time windows
    groups = group_events_by_time(extracted, window_seconds=5)
    print(f"📊 Grouped {len(extracted)} actions into {len(groups)} event clusters")

    # Process each group: build prompt, call Grok, save result
    results: List[Dict[str, Any]] = []

    for i, group in enumerate(groups, start=1):
        # Build prompt
        prompt = build_prompt(group)

        if args.dry_run:
            print(f"\n--- Group {i} (Dry-run) ---")
            print(f"Actions: {len(group)}")
            print(f"Prompt:\n{prompt}\n")
            commentary = "[DRY-RUN] Commentary skipped"
        else:
            print(f"📢 Processing group {i}/{len(groups)} ({len(group)} action(s))...")
            try:
                commentary = call_grok_api(prompt, dry_run=False)
            except Exception as e:
                commentary = f"[ERROR] {e}"
                print(f"   ⚠️  {commentary}")
                continue

        # Build result entry
        result_entry = {
            "group_index": i,
            "num_actions": len(group),
            "actions": group,
            "prompt": prompt,
            "commentary": commentary,
        }
        results.append(result_entry)
        print(f"   ✅ Commentary: {commentary[:80]}...")

        # Save individual response to text/ folder
        individual_file = text_dir / f"response_{i:04d}.json"
        try:
            individual_file.write_text(json.dumps(result_entry, indent=2, ensure_ascii=False, cls=DateTimeEncoder))
            print(f"   💾 Saved to {individual_file}")
        except Exception as e:
            print(f"   ⚠️  Failed to save individual file: {e}")

    # Save combined results to JSON
    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(results, indent=2, ensure_ascii=False, cls=DateTimeEncoder))
        print(f"\n✅ Saved {len(results)} commentary entries to {args.output}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False, cls=DateTimeEncoder))

    print(f"✅ Saved {len(results)} individual response files to {text_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
