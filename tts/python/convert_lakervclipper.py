#!/usr/bin/env python3
"""
Convert timeActual from ISO timestamps to elapsed seconds in lakervclipper.json
"""

import json
import ast
from datetime import datetime
from pathlib import Path


def parse_timestamp(ts_str):
    """Parse ISO 8601 timestamp to datetime object."""
    ts_str = ts_str.rstrip('Z')
    
    if '.' in ts_str:
        parts = ts_str.split('.')
        decimal = parts[1][:6].ljust(6, '0')
        ts_str = parts[0] + '.' + decimal
        return datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S.%f')
    else:
        return datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S')


def main():
    input_file = Path(__file__).parent / "maverickvgrizzlies.json"
    
    print(f"📂 Reading {input_file}...")
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Parse - file uses Python dict format (single quotes)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Parse as Python literals (handles single quotes)
        data = []
        for line in content.strip().split('\n'):
            if line.strip():
                entry = ast.literal_eval(line.strip())
                data.append(entry)
    
    if not data:
        print("❌ No data found")
        return
    
    # Get first timestamp as reference
    first_time_str = data[0].get('timeActual')
    if not first_time_str or not isinstance(first_time_str, str):
        print("❌ First entry has no valid timeActual timestamp")
        return
    
    first_time = parse_timestamp(first_time_str)
    print(f"📍 Reference time (first entry): {first_time_str}")
    
    # Convert all timeActual to elapsed seconds
    for entry in data:
        time_str = entry.get('timeActual')
        if time_str and isinstance(time_str, str):
            current_time = parse_timestamp(time_str)
            elapsed_seconds = (current_time - first_time).total_seconds()
            entry['timeActual'] = round(elapsed_seconds, 2)
    
    # Save
    with open(input_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Converted {len(data)} entries")
    print(f"✅ Saved to '{input_file}'")
    print(f"\nFirst 5 timeActual values:")
    for i, entry in enumerate(data[:5]):
        print(f"  Entry {i}: {entry.get('timeActual')} seconds")
    print(f"\nLast entry timeActual: {data[-1].get('timeActual')} seconds")


if __name__ == '__main__':
    main()

