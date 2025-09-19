#!/usr/bin/env python3
"""
Create RTH-only data from existing full trading hours data.
RTH = Regular Trading Hours: 9:30 AM - 4:00 PM ET
"""

import pandas as pd
import sys
from datetime import datetime
import pytz

def filter_rth_data(input_file, output_file):
    """Filter data to only include Regular Trading Hours (9:30 AM - 4:00 PM ET)"""
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    if df.empty:
        print(f"Warning: {input_file} is empty")
        # Create empty output file with header
        df.to_csv(output_file, index=False)
        return 0
    
    # Convert timestamp to datetime with UTC timezone
    # Handle both 'ts' and 'timestamp' column names
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'ts'
    df['timestamp_dt'] = pd.to_datetime(df[ts_col], format='mixed', utc=True)
    
    # Convert to ET timezone
    df['timestamp_et'] = df['timestamp_dt'].dt.tz_convert('America/New_York')
    
    # Filter for RTH: 9:30 AM - 4:00 PM ET
    df['hour'] = df['timestamp_et'].dt.hour
    df['minute'] = df['timestamp_et'].dt.minute
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    
    rth_start = 9 * 60 + 30  # 9:30 AM
    rth_end = 16 * 60        # 4:00 PM
    
    # Filter for RTH and weekdays only
    rth_mask = (df['time_minutes'] >= rth_start) & (df['time_minutes'] < rth_end)
    weekday_mask = df['timestamp_et'].dt.weekday < 5  # Monday=0, Sunday=6
    
    df_rth = df[rth_mask & weekday_mask].copy()
    
    # Drop helper columns and keep original format
    df_rth = df_rth[[ts_col, 'symbol', 'open', 'high', 'low', 'close', 'volume']]
    
    # Rename column back to timestamp for consistency
    if ts_col != 'timestamp':
        df_rth = df_rth.rename(columns={ts_col: 'timestamp'})
    
    # Save RTH data
    df_rth.to_csv(output_file, index=False)
    
    print(f"Filtered {len(df)} bars -> {len(df_rth)} RTH bars: {output_file}")
    return len(df_rth)

def main():
    if len(sys.argv) != 3:
        print("Usage: python create_rth_data.py <input_file> <output_file>")
        print("Example: python create_rth_data.py data/equities/QQQ_NH_ALIGNED.csv data/equities/QQQ_RTH_NH.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        bars_count = filter_rth_data(input_file, output_file)
        if bars_count > 0:
            print(f"✅ Successfully created RTH data: {bars_count} bars")
        else:
            print(f"⚠️  No RTH data found in {input_file}")
    except Exception as e:
        print(f"❌ Error processing {input_file}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
