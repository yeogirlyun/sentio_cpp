#!/usr/bin/env python3
"""
Analyze trades CSV to understand leverage trading patterns.
"""

import pandas as pd
import sys
from collections import defaultdict

def analyze_trades_csv(csv_file):
    """Analyze trades CSV for leverage trading patterns."""
    
    # Read the CSV
    df = pd.read_csv(csv_file)
    
    print(f"📊 TRADES ANALYSIS")
    print(f"Total Trades: {len(df)}")
    print()
    
    # Analyze by instrument
    instrument_stats = {}
    
    for instrument in df['inst'].unique():
        inst_df = df[df['inst'] == instrument]
        
        stats = {
            'total_trades': len(inst_df),
            'buy_trades': len(inst_df[inst_df['side'] == 1]),
            'sell_trades': len(inst_df[inst_df['side'] == 0]),
            'total_qty': inst_df['qty'].sum(),
            'total_notional': (inst_df['qty'] * inst_df['px']).sum(),
            'total_pnl': inst_df['pnl_d'].sum(),
            'avg_pnl': inst_df['pnl_d'].mean(),
            'winning_trades': len(inst_df[inst_df['pnl_d'] > 0]),
            'losing_trades': len(inst_df[inst_df['pnl_d'] < 0]),
            'win_rate': len(inst_df[inst_df['pnl_d'] > 0]) / len(inst_df) * 100 if len(inst_df) > 0 else 0
        }
        
        instrument_stats[instrument] = stats
    
    # Print analysis
    print("🎯 INSTRUMENT DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    total_trades = len(df)
    
    for instrument in sorted(instrument_stats.keys()):
        stats = instrument_stats[instrument]
        percentage = stats['total_trades'] / total_trades * 100
        
        print(f"\n📈 {instrument} Analysis:")
        print(f"  Total Trades: {stats['total_trades']} ({percentage:.1f}%)")
        print(f"  Buy Trades: {stats['buy_trades']}")
        print(f"  Sell Trades: {stats['sell_trades']}")
        print(f"  Total Quantity: {stats['total_qty']:,.0f}")
        print(f"  Total Notional: ${stats['total_notional']:,.2f}")
        print(f"  Total P&L: ${stats['total_pnl']:,.2f}")
        print(f"  Average P&L per Trade: ${stats['avg_pnl']:,.2f}")
        print(f"  Win Rate: {stats['win_rate']:.1f}% ({stats['winning_trades']}/{stats['total_trades']})")
        print(f"  Winning Trades: {stats['winning_trades']}")
        print(f"  Losing Trades: {stats['losing_trades']}")
    
    # Analyze leverage patterns
    print(f"\n⚡ LEVERAGE TRADING ANALYSIS")
    print("=" * 60)
    
    leverage_instruments = {
        'QQQ': '1x Long (Base)',
        'TQQQ': '3x Long (Bull)',
        'SQQQ': '3x Short (Bear)',
        'PSQ': '1x Short (Inverse)'
    }
    
    for instrument, description in leverage_instruments.items():
        if instrument in instrument_stats:
            stats = instrument_stats[instrument]
            percentage = stats['total_trades'] / total_trades * 100
            
            print(f"\n🔸 {instrument} ({description}):")
            print(f"  Usage: {stats['total_trades']} trades ({percentage:.1f}%)")
            print(f"  P&L: ${stats['total_pnl']:,.2f}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
    
    # Analyze PSQ usage specifically (1x reverse trades)
    if 'PSQ' in instrument_stats:
        psq_stats = instrument_stats['PSQ']
        print(f"\n🔄 PSQ (1x REVERSE) ANALYSIS")
        print("=" * 60)
        print(f"PSQ represents 1x inverse QQQ trades (short exposure)")
        print(f"Total PSQ Trades: {psq_stats['total_trades']}")
        print(f"PSQ Trade Percentage: {psq_stats['total_trades']/total_trades*100:.1f}%")
        print(f"PSQ Total P&L: ${psq_stats['total_pnl']:,.2f}")
        print(f"PSQ Win Rate: {psq_stats['win_rate']:.1f}%")
        
        # Show some PSQ trade examples
        psq_trades = df[df['inst'] == 'PSQ'].head(10)
        if len(psq_trades) > 0:
            print(f"\n📋 Sample PSQ Trades:")
            for _, trade in psq_trades.iterrows():
                side_str = "BUY" if trade['side'] == 1 else "SELL"
                print(f"  {side_str} {trade['qty']:.0f} shares @ ${trade['px']:.2f} | P&L: ${trade['pnl_d']:.2f}")
    
    # Analyze trading patterns
    print(f"\n📈 TRADING PATTERNS")
    print("=" * 60)
    
    # Side distribution
    buy_trades = len(df[df['side'] == 1])
    sell_trades = len(df[df['side'] == 0])
    print(f"Buy Trades: {buy_trades} ({buy_trades/total_trades*100:.1f}%)")
    print(f"Sell Trades: {sell_trades} ({sell_trades/total_trades*100:.1f}%)")
    
    # Overall performance
    total_pnl = df['pnl_d'].sum()
    avg_pnl = df['pnl_d'].mean()
    win_rate = len(df[df['pnl_d'] > 0]) / total_trades * 100
    
    print(f"\nOverall Performance:")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Average P&L per Trade: ${avg_pnl:,.2f}")
    print(f"Overall Win Rate: {win_rate:.1f}%")
    
    return instrument_stats

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_trades_csv.py <trades_csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    instrument_stats = analyze_trades_csv(csv_file)
    
    print(f"\n✅ Analysis complete!")

if __name__ == '__main__':
    main()
