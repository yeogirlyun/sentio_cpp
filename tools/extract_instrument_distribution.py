#!/usr/bin/env python3
"""
Quick script to extract instrument distribution from audit files.
"""

import sys
import json
from collections import Counter

def extract_instrument_distribution(audit_file):
    """Extract instrument distribution from audit file."""
    
    instruments = []
    
    with open(audit_file, 'r') as f:
        for line in f:
            try:
                # Extract JSON part before sha1
                json_part = line.strip()
                if '","sha1":"' in json_part:
                    json_part = json_part.split('","sha1":"')[0] + '"}'
                elif ',"sha1":"' in json_part:
                    json_part = json_part.split(',"sha1":"')[0] + '}'
                
                event = json.loads(json_part)
                
                # Only count fill events
                if event.get('type') == 'fill':
                    instrument = event.get('inst', 'UNKNOWN')
                    instruments.append(instrument)
                    
            except (json.JSONDecodeError, KeyError):
                continue
    
    # Count and display
    counter = Counter(instruments)
    total = sum(counter.values())
    
    print(f"📊 Instrument Distribution Analysis")
    print(f"📁 File: {audit_file}")
    print(f"📈 Total Trades: {total}")
    print()
    
    for instrument, count in counter.most_common():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{instrument:>6}: {count:>4} trades ({percentage:>5.1f}%)")
    
    return counter

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_instrument_distribution.py <audit_file>")
        sys.exit(1)
    
    audit_file = sys.argv[1]
    extract_instrument_distribution(audit_file)

if __name__ == '__main__':
    main()
