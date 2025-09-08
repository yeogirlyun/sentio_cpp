#!/usr/bin/env python3
import sys, json
from datetime import datetime, timezone
from collections import defaultdict

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/emit_last10_trades.py <audit.jsonl>")
        sys.exit(1)
    path = sys.argv[1]
    fills = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Some audit lines append a trailing ,"sha1":"..." after the JSON object.
                # Trim that suffix if present to obtain a valid JSON object.
                raw = line
                cut = raw.find('},"sha1"')
                if cut != -1:
                    raw = raw[:cut+1]
                ev = json.loads(raw)
            except Exception:
                continue
            if ev.get("type") == "fill":
                ts = int(ev.get("ts", 0))
                d = datetime.fromtimestamp(ts, tz=timezone.utc)
                day = d.date().isoformat()
                fills.append((ts, day, ev))
    if not fills:
        print("No fills found.")
        return
    fills.sort(key=lambda x: x[0])
    # collect last 10 distinct days
    days_ordered = []
    seen = set()
    for _, day, _ in fills:
        if day not in seen:
            seen.add(day)
            days_ordered.append(day)
    last10 = set(days_ordered[-10:])
    # group by day
    by_day = defaultdict(list)
    for ts, day, ev in fills:
        if day in last10:
            by_day[day].append((ts, ev))
    for day in sorted(by_day.keys()):
        print(f"=== {day} ===")
        for ts, ev in sorted(by_day[day], key=lambda x: x[0]):
            t = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            inst = ev.get("inst", "")
            side = ev.get("side")
            side_s = "BUY" if side in (0, "Buy") else ("SELL" if side in (1, "Sell") else str(side))
            qty = float(ev.get("qty", 0.0))
            px = float(ev.get("px", 0.0))
            fees = float(ev.get("fees", 0.0))
            pnl_d = float(ev.get("pnl_d", 0.0))
            pos_after = float(ev.get("pos_after", 0.0))
            eq_after = float(ev.get("eq_after", 0.0))
            print(f"{t} {inst:5s} {side_s:4s} qty={qty:.4f} px={px:.4f} fees={fees:.4f} pnl_d={pnl_d:.4f} pos_after={pos_after:.4f} eq_after={eq_after:.2f}")

if __name__ == "__main__":
    main()
