from __future__ import annotations
import argparse, pathlib
from audit_analyzer_v2 import AuditAnalyzer

def main():
    ap = argparse.ArgumentParser(description="Analyze Sentio audit files")
    ap.add_argument("audit_path", help="Path to audit file (.jsonl or .jsonl.gz)")
    ap.add_argument("--trades-csv", help="Optional path to export trades CSV")
    ap.add_argument("--signals-csv", help="Optional path to export signals CSV")
    ap.add_argument("--daily-csv", help="Optional path to export daily summary CSV")
    ap.add_argument("--summary", action="store_true", help="Print detailed summary")
    args = ap.parse_args()

    analyzer = AuditAnalyzer()
    analyzer.load(args.audit_path)
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

if __name__ == "__main__":
    main()
