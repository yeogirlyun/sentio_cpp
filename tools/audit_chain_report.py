from __future__ import annotations
import sys, json, pathlib
from collections import defaultdict

def load_events(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                yield ev
            except Exception:
                continue

def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/audit_chain_report.py <audit.jsonl> <out.txt>")
        sys.exit(1)
    audit_path = sys.argv[1]
    out_path = sys.argv[2]

    chains = defaultdict(lambda: {"signal": None, "routes": [], "orders": [], "fills": []})
    for ev in load_events(audit_path):
        t = ev.get("type", "")
        chain = ev.get("chain")
        if not chain:
            continue
        if t == "signal":
            chains[chain]["signal"] = ev
        elif t == "route":
            chains[chain]["routes"].append(ev)
        elif t == "order":
            chains[chain]["orders"].append(ev)
        elif t == "fill":
            chains[chain]["fills"].append(ev)

    lines = []
    lines.append(f"Audit Chain Report for: {audit_path}\n")
    keys = sorted(chains.keys(), key=lambda k: int(k.split(":")[0]))
    for k in keys:
        ch = chains[k]
        sig = ch["signal"] or {}
        ts = sig.get("ts", 0)
        base = sig.get("base", "")
        sig_code = sig.get("sig")
        conf = sig.get("conf")
        sig_name = {0:"BUY",1:"STRONG_BUY",2:"SELL",3:"STRONG_SELL",4:"HOLD"}.get(sig_code, str(sig_code))
        lines.append(f"ts={ts} chain={k} base={base} signal={sig_name} conf={conf}")
        for r in ch["routes"]:
            lines.append(f"  route: inst={r.get('inst')} tw={r.get('tw')}")
        for o in ch["orders"]:
            lines.append(f"  order: inst={o.get('inst')} side={o.get('side')} qty={o.get('qty')} limit={o.get('limit')}")
        for f in ch["fills"]:
            lines.append(f"  fill: inst={f.get('inst')} px={f.get('px')} qty={f.get('qty')} fees={f.get('fees')} side={f.get('side')} pnl_d={f.get('pnl_d')} eq_after={f.get('eq_after')} pos_after={f.get('pos_after')}")
        lines.append("")

    pathlib.Path(out_path).write_text("\n".join(lines), encoding='utf-8')
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()


