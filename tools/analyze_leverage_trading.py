#!/usr/bin/env python3
"""
Analyze leverage trading patterns in IRE and ASP strategies.
Focuses on instrument distribution (QQQ, PSQ, TQQQ, SQQQ) and P/L analysis.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import statistics

def load_audit_file(file_path):
    """Load and parse audit JSONL file."""
    events = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                # Extract JSON part before the sha1 field
                json_part = line.strip()
                if '","sha1":"' in json_part:
                    json_part = json_part.split('","sha1":"')[0] + '"}'
                elif ',"sha1":"' in json_part:
                    json_part = json_part.split(',"sha1":"')[0] + '}'
                
                event = json.loads(json_part)
                events.append(event)
            except json.JSONDecodeError:
                continue
    return events

def analyze_leverage_trading(events):
    """Analyze leverage trading patterns from audit events."""
    
    # Separate different event types (using actual audit format)
    trades = [e for e in events if e.get('type') == 'fill']
    signals = [e for e in events if e.get('type') == 'signal']
    orders = [e for e in events if e.get('type') == 'order']
    
    print(f"📊 ANALYSIS SUMMARY")
    print(f"Total Events: {len(events)}")
    print(f"Trades: {len(trades)}")
    print(f"Signals: {len(signals)}")
    print(f"Orders: {len(orders)}")
    print()
    
    # Analyze instrument distribution in trades
    instrument_trades = defaultdict(list)
    instrument_stats = defaultdict(lambda: {
        'total_trades': 0,
        'buy_trades': 0,
        'sell_trades': 0,
        'total_qty': 0,
        'total_notional': 0,
        'pnl_by_trade': [],
        'winning_trades': 0,
        'losing_trades': 0
    })
    
    for trade in trades:
        instrument = trade.get('inst', 'UNKNOWN')
        side = trade.get('side', 0)  # 0=sell, 1=buy
        qty = trade.get('qty', 0)
        px = trade.get('px', 0)
        pnl_d = trade.get('pnl_d', 0)
        
        instrument_trades[instrument].append(trade)
        
        stats = instrument_stats[instrument]
        stats['total_trades'] += 1
        stats['total_qty'] += qty
        stats['total_notional'] += qty * px
        
        if side == 1:  # Buy
            stats['buy_trades'] += 1
        else:  # Sell
            stats['sell_trades'] += 1
            
        stats['pnl_by_trade'].append(pnl_d)
        
        if pnl_d > 0:
            stats['winning_trades'] += 1
        elif pnl_d < 0:
            stats['losing_trades'] += 1
    
    # Analyze signal routing patterns from orders
    signal_routing = defaultdict(lambda: defaultdict(int))
    for order in orders:
        instrument = order.get('inst', 'UNKNOWN')
        side = order.get('side', 0)  # 0=sell, 1=buy
        
        if side == 1:  # Buy
            signal_routing[instrument]['long_signals'] += 1
        elif side == 0:  # Sell
            signal_routing[instrument]['short_signals'] += 1
        else:
            signal_routing[instrument]['neutral_signals'] += 1
    
    # Print instrument distribution analysis
    print("🎯 INSTRUMENT DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    total_trades = sum(stats['total_trades'] for stats in instrument_stats.values())
    
    for instrument in sorted(instrument_stats.keys()):
        stats = instrument_stats[instrument]
        trades_count = stats['total_trades']
        percentage = (trades_count / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\n📈 {instrument} Analysis:")
        print(f"  Total Trades: {trades_count} ({percentage:.1f}%)")
        print(f"  Buy Trades: {stats['buy_trades']}")
        print(f"  Sell Trades: {stats['sell_trades']}")
        print(f"  Total Quantity: {stats['total_qty']:,.0f}")
        print(f"  Total Notional: ${stats['total_notional']:,.2f}")
        
        if stats['pnl_by_trade']:
            total_pnl = sum(stats['pnl_by_trade'])
            avg_pnl = statistics.mean(stats['pnl_by_trade'])
            win_rate = (stats['winning_trades'] / trades_count * 100) if trades_count > 0 else 0
            
            print(f"  Total P&L: ${total_pnl:,.2f}")
            print(f"  Average P&L per Trade: ${avg_pnl:,.2f}")
            print(f"  Win Rate: {win_rate:.1f}% ({stats['winning_trades']}/{trades_count})")
            print(f"  Winning Trades: {stats['winning_trades']}")
            print(f"  Losing Trades: {stats['losing_trades']}")
            
            if len(stats['pnl_by_trade']) > 1:
                pnl_std = statistics.stdev(stats['pnl_by_trade'])
                print(f"  P&L Std Dev: ${pnl_std:,.2f}")
    
    # Print signal routing analysis
    print(f"\n🚦 SIGNAL ROUTING ANALYSIS")
    print("=" * 60)
    
    for instrument in sorted(signal_routing.keys()):
        routing = signal_routing[instrument]
        total_signals = sum(routing.values())
        
        if total_signals > 0:
            print(f"\n📡 {instrument} Signal Routing:")
            print(f"  Total Signals: {total_signals}")
            print(f"  Long Signals: {routing['long_signals']} ({routing['long_signals']/total_signals*100:.1f}%)")
            print(f"  Short Signals: {routing['short_signals']} ({routing['short_signals']/total_signals*100:.1f}%)")
            print(f"  Neutral Signals: {routing['neutral_signals']} ({routing['neutral_signals']/total_signals*100:.1f}%)")
    
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
            trades_count = stats['total_trades']
            percentage = (trades_count / total_trades * 100) if total_trades > 0 else 0
            
            print(f"\n🔸 {instrument} ({description}):")
            print(f"  Usage: {trades_count} trades ({percentage:.1f}%)")
            
            if stats['pnl_by_trade']:
                total_pnl = sum(stats['pnl_by_trade'])
                win_rate = (stats['winning_trades'] / trades_count * 100) if trades_count > 0 else 0
                print(f"  P&L: ${total_pnl:,.2f}")
                print(f"  Win Rate: {win_rate:.1f}%")
    
    # Analyze PSQ usage specifically (1x reverse trades)
    if 'PSQ' in instrument_stats:
        psq_stats = instrument_stats['PSQ']
        print(f"\n🔄 PSQ (1x REVERSE) ANALYSIS")
        print("=" * 60)
        print(f"PSQ represents 1x inverse QQQ trades (short exposure)")
        print(f"Total PSQ Trades: {psq_stats['total_trades']}")
        print(f"PSQ Trade Percentage: {psq_stats['total_trades']/total_trades*100:.1f}%")
        
        if psq_stats['pnl_by_trade']:
            psq_pnl = sum(psq_stats['pnl_by_trade'])
            psq_win_rate = (psq_stats['winning_trades'] / psq_stats['total_trades'] * 100) if psq_stats['total_trades'] > 0 else 0
            print(f"PSQ Total P&L: ${psq_pnl:,.2f}")
            print(f"PSQ Win Rate: {psq_win_rate:.1f}%")
    
    return instrument_stats, signal_routing

def main():
    parser = argparse.ArgumentParser(description='Analyze leverage trading patterns')
    parser.add_argument('audit_file', help='Path to audit JSONL file')
    parser.add_argument('--strategy', help='Strategy name for context')
    
    args = parser.parse_args()
    
    audit_file = Path(args.audit_file)
    if not audit_file.exists():
        print(f"Error: Audit file '{audit_file}' not found.")
        sys.exit(1)
    
    print(f"🔍 Analyzing leverage trading patterns...")
    print(f"📁 File: {audit_file}")
    if args.strategy:
        print(f"📊 Strategy: {args.strategy}")
    print()
    
    events = load_audit_file(audit_file)
    if not events:
        print("Error: No valid events found in audit file.")
        sys.exit(1)
    
    instrument_stats, signal_routing = analyze_leverage_trading(events)
    
    print(f"\n✅ Analysis complete!")

if __name__ == '__main__':
    main()
