#!/usr/bin/env python3

import sqlite3

def debug_audit_position_calculation(db_path, run_id, symbol):
    """Debug the audit position calculation step by step"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all trades for the symbol
    cursor.execute("""
        SELECT ts_millis, side, qty, price 
        FROM audit_events 
        WHERE run_id = ? AND kind = 'FILL' AND symbol = ? 
        ORDER BY ts_millis ASC
    """, (run_id, symbol))
    
    trades = cursor.fetchall()
    
    print(f"=== AUDIT POSITION CALCULATION DEBUG ===")
    print(f"Symbol: {symbol}, Run ID: {run_id}")
    print(f"Total trades: {len(trades)}")
    print()
    
    running_position = 0.0
    buy_total = 0.0
    sell_total = 0.0
    
    print("First 10 trades:")
    for i, (ts, side, qty, price) in enumerate(trades[:10]):
        old_position = running_position
        running_position += qty  # qty is already signed
        
        if side == 'BUY':
            buy_total += qty
        else:
            sell_total += qty
            
        print(f"{i+1:3d}: {side:4s} {qty:+15.10f} -> Position: {old_position:+15.10f} -> {running_position:+15.10f}")
    
    print("\n...")
    print(f"\nLast 10 trades:")
    for i, (ts, side, qty, price) in enumerate(trades[-10:]):
        start_idx = len(trades) - 10 + i
        # Calculate position up to this point
        position_at_start = sum(t[2] for t in trades[:start_idx])
        position_after = position_at_start + qty
        
        print(f"{start_idx+1:3d}: {side:4s} {qty:+15.10f} -> Position: {position_at_start:+15.10f} -> {position_after:+15.10f}")
    
    final_position = sum(t[2] for t in trades)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total BUY quantity:  {buy_total:+15.10f}")
    print(f"Total SELL quantity: {sell_total:+15.10f}")
    print(f"Net position:        {final_position:+15.10f}")
    print(f"Expected (StratTest): 0.00000000000")
    print(f"Discrepancy:         {final_position:+15.10f}")
    
    if abs(final_position) < 1e-6:
        print("✅ Position is effectively zero (within floating-point precision)")
    else:
        print("❌ Significant position discrepancy detected!")
    
    conn.close()

if __name__ == "__main__":
    debug_audit_position_calculation("audit/sentio_audit.sqlite3", "475276", "QQQ")
