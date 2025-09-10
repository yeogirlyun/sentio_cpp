from __future__ import annotations
import argparse, pathlib, sys
from audit_analyzer import AuditAnalyzer

def print_usage():
    print("Usage: audit_cli <command> [options]")
    print("Commands:")
    print("  replay <audit_file.jsonl> [--summary] [--trades] [--metrics]")
    print("  format <audit_file.jsonl> [--output <output_file>] [--type <txt|csv>] [--trades-only]")
    print("  trades <audit_file.jsonl> [--output <output_file>]")
    print("  analyze <audit_file.jsonl> [--trades-csv <file>] [--signals-csv <file>] [--daily-csv <file>] [--summary]")
    print("  latest [--max-trades <n>] [--audit-dir <dir>]")

def cmd_replay(args):
    """Replay command - shows basic audit information"""
    analyzer = AuditAnalyzer()
    analyzer.load(args.audit_file)
    s = analyzer.stats()
    
    # Default to showing everything if no specific flags
    show_summary = args.summary or (not args.trades and not args.metrics)
    show_trades = args.trades or (not args.summary and not args.metrics)
    show_metrics = args.metrics or (not args.summary and not args.trades)
    
    if show_summary:
        print("=== AUDIT REPLAY SUMMARY ===")
        print(f"Run ID: {analyzer.run_metadata.get('run', 'unknown')}")
        print(f"Strategy: {analyzer.run_metadata.get('strategy', 'unknown')}")
        print(f"Total Records: {sum(s.values())}")
        print(f"Trades: {s['trades']}")
        print(f"Snapshots: {s['snapshots']}")
        print(f"Signals: {s['signals']}")
        print()
    
    if show_metrics and analyzer.snapshots:
        initial = analyzer.snapshots[0]
        final = analyzer.snapshots[-1]
        
        initial_equity = initial.get('equity', 100000.0)
        final_equity = final.get('equity', 100000.0)
        total_return = (final_equity - initial_equity) / initial_equity
        monthly_return = (final_equity / initial_equity) ** (1.0/3.0) - 1.0
        
        print("=== PERFORMANCE METRICS ===")
        print(f"Initial Equity: ${initial_equity:.2f}")
        print(f"Final Equity: ${final_equity:.2f}")
        print(f"Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
        print(f"Monthly Return: {monthly_return:.4f} ({monthly_return*100:.2f}%)")
        
        if analyzer.trades:
            print(f"Total Trades: {len(analyzer.trades)}")
            print(f"Avg Trades/Day: {len(analyzer.trades) / 63.0:.1f}")
            
            # Add instrument distribution analysis
            print("\n=== INSTRUMENT DISTRIBUTION ===")
            instrument_stats = {}
            total_trades = len(analyzer.trades)
            
            for trade in analyzer.trades:
                instrument = trade.get('inst', 'UNKNOWN')
                pnl = trade.get('pnl_d', 0.0)
                
                if instrument not in instrument_stats:
                    instrument_stats[instrument] = {
                        'count': 0,
                        'total_pnl': 0.0,
                        'winning_trades': 0,
                        'losing_trades': 0
                    }
                
                stats = instrument_stats[instrument]
                stats['count'] += 1
                stats['total_pnl'] += pnl
                
                if pnl > 0:
                    stats['winning_trades'] += 1
                elif pnl < 0:
                    stats['losing_trades'] += 1
            
            # Sort by trade count (most active first)
            sorted_instruments = sorted(instrument_stats.items(), key=lambda x: x[1]['count'], reverse=True)
            
            for instrument, stats in sorted_instruments:
                percentage = (stats['count'] / total_trades * 100) if total_trades > 0 else 0
                win_rate = (stats['winning_trades'] / stats['count'] * 100) if stats['count'] > 0 else 0
                
                print(f"{instrument:>6}: {stats['count']:>4} trades ({percentage:>5.1f}%) | "
                      f"P&L: ${stats['total_pnl']:>8.2f} | Win Rate: {win_rate:>5.1f}%")
        print()
    
    if show_trades and analyzer.trades:
        print("=== RECENT TRADES (Last 10) ===")
        print("Time                Side Instr   Quantity    Price      PnL")
        print("----------------------------------------------------------------")
        
        start_idx = max(0, len(analyzer.trades) - 10)
        for trade in analyzer.trades[start_idx:]:
            side = trade.get('side', '')
            inst = trade.get('inst', '')
            qty = trade.get('qty', 0.0)
            price = trade.get('price', 0.0)
            pnl = trade.get('pnl', 0.0)
            ts = trade.get('ts', 0)
            
            # Convert timestamp
            from datetime import datetime
            dt = datetime.fromtimestamp(ts)
            time_str = dt.strftime("%Y-%m-%d %H:%M")
            
            print(f"{time_str:<19} {side:<5} {inst:<7} {qty:<10.0f} {price:<10.2f} {pnl:<10.2f}")

def cmd_format(args):
    """Format command - converts audit file to human-readable or CSV format"""
    analyzer = AuditAnalyzer()
    analyzer.load(args.audit_file)
    
    output_file = args.output
    if not output_file:
        # Generate default output filename
        base_name = pathlib.Path(args.audit_file).stem
        if args.trades_only:
            output_file = f"audit/{base_name}_trades_only.txt"
        elif args.type == "csv":
            output_file = f"audit/{base_name}_data.csv"
        else:
            output_file = f"audit/{base_name}_human_readable.txt"
    
    # Create output directory if needed
    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        if args.type == "csv":
            f.write("Timestamp,Type,Symbol,Side,Quantity,Price,Trade_PnL,Cash,Realized_PnL,Unrealized_PnL,Total_Equity\n")
        elif args.trades_only:
            f.write("TRADES ONLY - AUDIT LOG\n")
            f.write("=======================\n\n")
            f.write("Format: [#] TIMESTAMP | TICKER | BUY/SELL | QUANTITY @ PRICE | EQUITY_AFTER\n")
            f.write("---------------------------------------------------------------------------------\n\n")
        else:
            f.write("HUMAN-READABLE AUDIT LOG\n")
            f.write("========================\n\n")
        
        line_num = 0
        for event in analyzer.all_events():
            line_num += 1
            event_type = event.get('type', '')
            ts = event.get('ts', 0)
            
            # Convert timestamp
            from datetime import datetime
            dt = datetime.fromtimestamp(ts)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            
            if args.type == "csv":
                f.write(f"{time_str},{event_type}")
                if event_type == "fill":
                    side_str = "BUY" if event.get('side', 0) == 0 else "SELL"
                    f.write(f",{event.get('inst', '')},{side_str},{event.get('qty', 0.0)},{event.get('px', 0.0)},{event.get('pnl_d', 0.0)},,,")
                elif event_type == "snapshot":
                    f.write(f",,,,,,{event.get('cash', 0.0)},{event.get('real', 0.0)},{event.get('equity', 0.0) - event.get('cash', 0.0) - event.get('real', 0.0)},{event.get('equity', 0.0)}")
                else:
                    f.write(",,,,,,,,")
                f.write("\n")
            elif args.trades_only:
                if event_type == "fill":
                    side_str = "BUY" if event.get('side', 0) == 0 else "SELL"
                    f.write(f"[{line_num:4d}] {time_str} | {event.get('inst', ''):<5} | {side_str:<4} | {event.get('qty', 0.0):<8.0f} @ ${event.get('px', 0.0):<8.2f} | Equity: ${event.get('eq_after', 0.0)}\n")
            else:
                f.write(f"[{line_num:4d}] {time_str} ")
                if event_type == "run_start":
                    meta = event.get('meta', {})
                    f.write(f"RUN START - Strategy: {meta.get('strategy', '')}, Series: {meta.get('total_series', 0)}\n")
                elif event_type == "fill":
                    side_str = "BUY" if event.get('side', 0) == 0 else "SELL"
                    f.write(f"TRADE - {side_str} {event.get('qty', 0.0)} {event.get('inst', '')} @ ${event.get('px', 0.0):.2f} (P&L: ${event.get('pnl_d', 0.0)})\n")
                elif event_type == "snapshot":
                    cash = event.get('cash', 0.0)
                    equity = event.get('equity', 0.0)
                    realized = event.get('real', 0.0)
                    unrealized = equity - cash - realized
                    f.write(f"PORTFOLIO - Cash: ${cash:.2f}, Realized P&L: ${realized}, Unrealized P&L: ${unrealized}, Total Equity: ${equity}\n")
                elif event_type == "signal":
                    f.write(f"SIGNAL - {event.get('inst', '')} p={event.get('p', 0.0):.3f} conf={event.get('conf', 0.0)}\n")
                elif event_type == "bar":
                    f.write(f"BAR - {event.get('inst', '')} O:{event.get('o', 0.0):.2f} H:{event.get('h', 0.0):.2f} L:{event.get('l', 0.0):.2f} C:{event.get('c', 0.0):.2f} V:{event.get('v', 0.0):.0f}\n")
                else:
                    f.write(f"{event_type} - {event}\n")
    
    print(f"Formatted audit log written to: {output_file}")

def cmd_trades(args):
    """Trades command - shows only trades"""
    args.type = "txt"
    args.trades_only = True
    cmd_format(args)

def cmd_analyze(args):
    """Analyze command - comprehensive analysis with CSV export"""
    analyzer = AuditAnalyzer()
    analyzer.load(args.audit_file)
    s = analyzer.stats()
    print(f"✅ Loaded: trades={s['trades']} snapshots={s['snapshots']} signals={s['signals']} bars={s['bars']} other={s['other']}")

    if args.summary:
        analyzer.print_summary()

    if args.trades_csv:
        analyzer.export_trades_csv(args.trades_csv)
        print(f"💾 Wrote trades CSV: {args.trades_csv}")
    if args.signals_csv:
        analyzer.export_signals_csv(args.signals_csv)
        print(f"💾 Wrote signals CSV: {args.signals_csv}")
    if args.daily_csv:
        analyzer.export_daily_summary(args.daily_csv)
        print(f"💾 Wrote daily summary CSV: {args.daily_csv}")

def cmd_latest(args):
    """Latest command - automatically find latest audit file and show quick metrics"""
    import glob
    import os
    
    # Find audit directory
    audit_dir = args.audit_dir if args.audit_dir else "audit"
    if not os.path.exists(audit_dir):
        print(f"❌ Audit directory '{audit_dir}' not found")
        return 1
    
    # Find all .jsonl files
    pattern = os.path.join(audit_dir, "*.jsonl")
    audit_files = glob.glob(pattern)
    
    if not audit_files:
        print(f"❌ No audit files found in '{audit_dir}'")
        return 1
    
    # Sort by modification time (newest first)
    audit_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_file = audit_files[0]
    
    print(f"🔍 Found latest audit file: {os.path.basename(latest_file)}")
    print(f"📅 Modified: {pathlib.Path(latest_file).stat().st_mtime}")
    print()
    
    # Load and analyze
    analyzer = AuditAnalyzer()
    analyzer.load(latest_file)
    s = analyzer.stats()
    
    # Quick summary
    print("=== QUICK METRICS ===")
    print(f"Strategy: {analyzer.run_metadata.get('strategy', 'Unknown')}")
    print(f"Total Records: {sum(s.values())}")
    print(f"Trades: {s['trades']}")
    print(f"Snapshots: {s['snapshots']}")
    print(f"Signals: {s['signals']}")
    print()
    
    # Performance metrics
    if analyzer.snapshots:
        initial = analyzer.snapshots[0]
        final = analyzer.snapshots[-1]
        
        initial_equity = initial.get('equity', 100000.0)
        final_equity = final.get('equity', 100000.0)
        total_return = (final_equity - initial_equity) / initial_equity
        
        print("=== PERFORMANCE ===")
        print(f"Initial Equity: ${initial_equity:,.2f}")
        print(f"Final Equity:   ${final_equity:,.2f}")
        print(f"Total Return:   {total_return:.4f} ({total_return*100:+.2f}%)")
        
        if analyzer.trades:
            print(f"Total Trades:   {len(analyzer.trades)}")
            print(f"Avg Trades/Day: {len(analyzer.trades) / 63.0:.1f}")
            
            # Add instrument distribution analysis
            print("\n=== INSTRUMENT DISTRIBUTION ===")
            instrument_stats = {}
            total_trades = len(analyzer.trades)
            
            for trade in analyzer.trades:
                instrument = trade.get('inst', 'UNKNOWN')
                pnl = trade.get('pnl_d', 0.0)
                
                if instrument not in instrument_stats:
                    instrument_stats[instrument] = {
                        'count': 0,
                        'total_pnl': 0.0,
                        'winning_trades': 0,
                        'losing_trades': 0
                    }
                
                stats = instrument_stats[instrument]
                stats['count'] += 1
                stats['total_pnl'] += pnl
                
                if pnl > 0:
                    stats['winning_trades'] += 1
                elif pnl < 0:
                    stats['losing_trades'] += 1
            
            # Sort by trade count (most active first)
            sorted_instruments = sorted(instrument_stats.items(), key=lambda x: x[1]['count'], reverse=True)
            
            for instrument, stats in sorted_instruments:
                percentage = (stats['count'] / total_trades * 100) if total_trades > 0 else 0
                win_rate = (stats['winning_trades'] / stats['count'] * 100) if stats['count'] > 0 else 0
                
                print(f"{instrument:>6}: {stats['count']:>4} trades ({percentage:>5.1f}%) | "
                      f"P&L: ${stats['total_pnl']:>8.2f} | Win Rate: {win_rate:>5.1f}%")
        print()
    
    # Recent trades
    if analyzer.trades:
        max_trades = args.max_trades
        recent_trades = analyzer.trades[-max_trades:] if len(analyzer.trades) > max_trades else analyzer.trades
        
        print(f"=== RECENT TRADES (Last {len(recent_trades)}) ===")
        print("Time                Side Instr   Quantity    Price      PnL")
        print("----------------------------------------------------------------")
        
        for trade in recent_trades:
            side = trade.get('side', '')
            inst = trade.get('inst', '')
            qty = trade.get('qty', 0.0)
            price = trade.get('price', 0.0)
            pnl = trade.get('pnl', 0.0)
            ts = trade.get('ts', 0)
            
            # Convert timestamp
            from datetime import datetime
            dt = datetime.fromtimestamp(ts)
            time_str = dt.strftime("%Y-%m-%d %H:%M")
            
            print(f"{time_str:<19} {side:<5} {inst:<7} {qty:<10.0f} {price:<10.2f} {pnl:<10.2f}")
    else:
        print("=== RECENT TRADES ===")
        print("No trades found in this audit file.")
    
    return 0

def main():
    if len(sys.argv) < 2:
        print_usage()
        return 1
    
    command = sys.argv[1]
    
    if command == "replay":
        parser = argparse.ArgumentParser(description="Replay audit file")
        parser.add_argument("audit_file", help="Path to audit file (.jsonl)")
        parser.add_argument("--summary", action="store_true", help="Show summary")
        parser.add_argument("--trades", action="store_true", help="Show trades")
        parser.add_argument("--metrics", action="store_true", help="Show metrics")
        args = parser.parse_args(sys.argv[2:])
        cmd_replay(args)
        
    elif command == "format":
        parser = argparse.ArgumentParser(description="Format audit file")
        parser.add_argument("audit_file", help="Path to audit file (.jsonl)")
        parser.add_argument("--output", help="Output file path")
        parser.add_argument("--type", choices=["txt", "csv"], default="txt", help="Output format")
        parser.add_argument("--trades-only", action="store_true", help="Show only trades")
        args = parser.parse_args(sys.argv[2:])
        cmd_format(args)
        
    elif command == "trades":
        parser = argparse.ArgumentParser(description="Show trades only")
        parser.add_argument("audit_file", help="Path to audit file (.jsonl)")
        parser.add_argument("--output", help="Output file path")
        args = parser.parse_args(sys.argv[2:])
        args.format_type = "txt"
        args.trades_only = True
        cmd_trades(args)
        
    elif command == "analyze":
        parser = argparse.ArgumentParser(description="Comprehensive audit analysis")
        parser.add_argument("audit_file", help="Path to audit file (.jsonl)")
        parser.add_argument("--trades-csv", help="Export trades CSV")
        parser.add_argument("--signals-csv", help="Export signals CSV")
        parser.add_argument("--daily-csv", help="Export daily summary CSV")
        parser.add_argument("--summary", action="store_true", help="Print detailed summary")
        args = parser.parse_args(sys.argv[2:])
        cmd_analyze(args)
        
    elif command == "latest":
        parser = argparse.ArgumentParser(description="Show latest audit file metrics")
        parser.add_argument("--max-trades", type=int, default=20, help="Maximum number of recent trades to show (default: 20)")
        parser.add_argument("--audit-dir", default="audit", help="Audit directory to search (default: audit)")
        args = parser.parse_args(sys.argv[2:])
        return cmd_latest(args)
        
    else:
        print_usage()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
