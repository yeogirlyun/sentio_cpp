#!/usr/bin/env python3
"""
Generate test market data for Sentio C++ backtesting
This creates realistic OHLCV data for testing purposes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_test_data(symbol, days=30, start_price=100.0):
    """Generate realistic OHLCV test data"""
    
    # Create date range (business days only)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days*2)  # Extra days to account for weekends
    
    dates = pd.bdate_range(start=start_date, end=end_date, freq='B')
    dates = dates[-days:]  # Take the last N business days
    
    # Generate price data with realistic patterns
    np.random.seed(42)  # For reproducible data
    
    # Random walk with drift
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Small positive drift, 2% daily volatility
    
    # Add some trend and mean reversion
    trend = np.linspace(0, 0.1, len(dates))  # Slight upward trend
    mean_reversion = -0.001 * np.arange(len(dates))  # Slight mean reversion
    
    returns += trend + mean_reversion
    
    # Generate prices
    prices = [start_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        daily_vol = abs(np.random.normal(0.015, 0.005))  # 1.5% average daily range
        
        high = close * (1 + abs(np.random.normal(0, daily_vol/2)))
        low = close * (1 - abs(np.random.normal(0, daily_vol/2)))
        open_price = close * (1 + np.random.normal(0, daily_vol/4))
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume (higher on volatile days)
        base_volume = 1000000
        volatility_factor = abs(returns[i]) * 10
        volume = int(base_volume * (1 + volatility_factor) * np.random.uniform(0.5, 2.0))
        
        data.append({
            'timestamp': date.strftime('%Y-%m-%dT%H:%M:%S-05:00'),  # EST timezone
            'symbol': symbol,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    return pd.DataFrame(data)

def main():
    """Generate test data for multiple symbols"""
    
    # Create data directory
    os.makedirs('data/equities', exist_ok=True)
    
    # Symbols to generate
    symbols = {
        'QQQ': 400.0,   # Start around $400
        'TQQQ': 50.0,   # Start around $50
        'SQQQ': 20.0,   # Start around $20
        'SPY': 450.0,   # Start around $450
        'IWM': 200.0,   # Start around $200
        'GLD': 180.0,   # Start around $180
        'TLT': 100.0    # Start around $100
    }
    
    print("Generating test market data...")
    
    for symbol, start_price in symbols.items():
        print(f"Generating data for {symbol}...")
        df = generate_test_data(symbol, days=30, start_price=start_price)
        
        # Save to CSV
        filename = f"data/equities/{symbol}.csv"
        df.to_csv(filename, index=False)
        print(f"  Saved {len(df)} bars to {filename}")
    
    print("\nTest data generation complete!")
    print("Files created:")
    for file in os.listdir('data/equities'):
        if file.endswith('.csv'):
            print(f"  data/equities/{file}")

if __name__ == "__main__":
    main()
