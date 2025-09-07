from __future__ import annotations
import csv, sys, pathlib
from typing import Dict, Any, List, Optional, Tuple
from audit_parser import iter_audit_file
from datetime import datetime, timezone
from collections import defaultdict

# --- Minimal "schema" normalization (no pydantic dependency) ---

def normalize_event(e: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure required keys exist; fill safe defaults
    e.setdefault("type", "")
    e.setdefault("run", "")
    e.setdefault("seq", 0)
    e.setdefault("ts", 0)  # epoch millis or nanos per your writer
    return e

# --- Analyzer ---

class AuditAnalyzer:
    def __init__(self) -> None:
        self.trades: List[Dict[str, Any]] = []
        self.snapshots: List[Dict[str, Any]] = []
        self.signals: List[Dict[str, Any]] = []
        self.bars: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.run_metadata: Dict[str, Any] = {}
        self.other: List[Dict[str, Any]] = []

    def load(self, path: str | pathlib.Path) -> None:
        for ln, event, sha1 in iter_audit_file(path):
            e = normalize_event(event)
            if sha1 is not None:
                e["_sha1"] = sha1  # keep association; optional
            t = e.get("type", "")
            try:
                if t == "fill":
                    self.trades.append(e)
                elif t == "snapshot":
                    self.snapshots.append(e)
                elif t == "signal":
                    self.signals.append(e)
                elif t == "bar":
                    inst = e.get("inst", "unknown")
                    self.bars[inst].append(e)
                elif t == "run_start":
                    self.run_metadata = e.get("meta", {})
                    self.other.append(e)
                else:
                    self.other.append(e)
            except Exception as ex:
                # Soft-fail the line, keep going
                if ln <= 5:
                    print(f"⚠️  Processing error on line {ln}: {ex}", file=sys.stderr)

    def stats(self) -> Dict[str, int]:
        return {
            "trades": len(self.trades),
            "snapshots": len(self.snapshots),
            "signals": len(self.signals),
            "bars": sum(len(bars) for bars in self.bars.values()),
            "other": len(self.other),
        }

    def analyze_strategy_performance(self) -> dict:
        """Analyze strategy performance from audit trail"""
        if not self.snapshots:
            return {"error": "No snapshots found"}
        
        # Calculate key metrics
        initial_equity = self.snapshots[0].get("equity", 100000.0) if self.snapshots else 100000.0
        final_equity = self.snapshots[-1].get("equity", initial_equity) if self.snapshots else initial_equity
        total_return = (final_equity - initial_equity) / initial_equity * 100
        
        # Trade analysis
        total_trades = len(self.trades)
        buy_trades = len([t for t in self.trades if t.get("side") == 1 or t.get("side") == "Buy"])
        sell_trades = len([t for t in self.trades if t.get("side") == 0 or t.get("side") == "Sell"])
        
        # Daily analysis
        daily_trades = self._analyze_daily_trades()
        
        return {
            "strategy": self.run_metadata.get("strategy", "Unknown"),
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "total_return_pct": total_return,
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "daily_trades": daily_trades,
            "snapshots_count": len(self.snapshots),
            "signals_count": len(self.signals)
        }
    
    def _analyze_daily_trades(self) -> List[dict]:
        """Analyze trades by day"""
        daily_data = defaultdict(lambda: {'trades': 0, 'volume': 0.0, 'instruments': set()})
        
        for trade in self.trades:
            # Convert timestamp to date (assuming UTC epoch)
            timestamp = trade.get("ts", 0)
            if timestamp > 0:
                date = datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
                daily_data[date]['trades'] += 1
                qty = trade.get("qty", 0)
                price = trade.get("px", 0)
                daily_data[date]['volume'] += abs(qty * price)
                inst = trade.get("inst", "")
                if inst:
                    daily_data[date]['instruments'].add(inst)
        
        # Convert to list and sort by date
        daily_list = []
        for date in sorted(daily_data.keys()):
            data = daily_data[date]
            daily_list.append({
                'date': str(date),
                'trades': data['trades'],
                'volume': data['volume'],
                'instruments': list(data['instruments'])
            })
        
        return daily_list
    
    def get_daily_balance_changes(self) -> List[dict]:
        """Get daily balance changes from snapshots"""
        daily_balances = defaultdict(list)
        
        for snapshot in self.snapshots:
            timestamp = snapshot.get("ts", 0)
            if timestamp > 0:
                date = datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
                daily_balances[date].append(snapshot)
        
        daily_changes = []
        for date in sorted(daily_balances.keys()):
            snapshots = daily_balances[date]
            if snapshots:
                # Use the last snapshot of the day
                last_snapshot = snapshots[-1]
                daily_changes.append({
                    'date': str(date),
                    'cash': last_snapshot.get("cash", 0),
                    'equity': last_snapshot.get("equity", 0),
                    'realized': last_snapshot.get("realized", 0),
                    'snapshots': len(snapshots)
                })
        
        return daily_changes

    def print_summary(self):
        """Print a comprehensive summary of the audit trail"""
        print(f"\n{'='*60}")
        print(f"📊 AUDIT TRAIL ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        # Strategy info
        strategy = self.run_metadata.get("strategy", "Unknown")
        print(f"🎯 Strategy: {strategy}")
        
        # Performance analysis
        perf = self.analyze_strategy_performance()
        if 'error' not in perf:
            print(f"\n💰 PERFORMANCE METRICS:")
            print(f"   Initial Equity: ${perf['initial_equity']:,.2f}")
            print(f"   Final Equity:   ${perf['final_equity']:,.2f}")
            print(f"   Total Return:   {perf['total_return_pct']:.2f}%")
            print(f"   Total Trades:   {perf['total_trades']:,}")
            print(f"   Buy Trades:     {perf['buy_trades']:,}")
            print(f"   Sell Trades:    {perf['sell_trades']:,}")
            print(f"   Signals:        {perf['signals_count']:,}")
        
        # Daily analysis
        daily_trades = perf.get('daily_trades', [])
        if daily_trades:
            avg_daily_trades = sum(d['trades'] for d in daily_trades) / len(daily_trades)
            print(f"\n📈 DAILY ANALYSIS:")
            print(f"   Trading Days:   {len(daily_trades)}")
            print(f"   Avg Daily Trades: {avg_daily_trades:.1f}")
            print(f"   Max Daily Trades: {max(d['trades'] for d in daily_trades)}")
            print(f"   Min Daily Trades: {min(d['trades'] for d in daily_trades)}")
        
        # Balance changes
        daily_balances = self.get_daily_balance_changes()
        if daily_balances:
            print(f"\n💳 DAILY BALANCE CHANGES:")
            print(f"   Days with Snapshots: {len(daily_balances)}")
            if len(daily_balances) >= 2:
                first_equity = daily_balances[0]['equity']
                last_equity = daily_balances[-1]['equity']
                print(f"   First Day Equity: ${first_equity:,.2f}")
                print(f"   Last Day Equity:  ${last_equity:,.2f}")

    # Optional CSV exporters
    def export_trades_csv(self, out_path: str | pathlib.Path) -> None:
        if not self.trades:
            return
        keys = sorted({k for e in self.trades for k in e.keys()})
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for e in self.trades:
                w.writerow(e)

    def export_signals_csv(self, out_path: str | pathlib.Path) -> None:
        if not self.signals:
            return
        keys = sorted({k for e in self.signals for k in e.keys()})
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for e in self.signals:
                w.writerow(e)

    def export_daily_summary(self, output_file: str):
        """Export daily summary to CSV"""
        daily_trades = self._analyze_daily_trades()
        daily_balances = self.get_daily_balance_changes()
        
        # Merge daily data
        daily_data = {}
        for d in daily_trades:
            daily_data[d['date']] = d
        for d in daily_balances:
            if d['date'] in daily_data:
                daily_data[d['date']].update(d)
            else:
                daily_data[d['date']] = d
        
        # Write CSV
        with open(output_file, 'w') as f:
            f.write("date,trades,volume,instruments,cash,equity,realized\n")
            for date in sorted(daily_data.keys()):
                data = daily_data[date]
                instruments = ','.join(data.get('instruments', []))
                f.write(f"{date},{data.get('trades', 0)},{data.get('volume', 0):.2f},"
                       f'"{instruments}",{data.get("cash", 0):.2f},'
                       f'{data.get("equity", 0):.2f},{data.get("realized", 0):.2f}\n')
        
        print(f"📄 Daily summary exported to: {output_file}")
