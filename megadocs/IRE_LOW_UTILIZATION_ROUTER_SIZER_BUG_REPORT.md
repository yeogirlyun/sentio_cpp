# Bug Report: Low Returns Due to Minimal Capital Deployment (Router/Sizer Config Not Applied)

## Summary
- Symptom: Post-TPA audit shows many tiny fills (e.g., qty≈0.002–0.01 QQQ) and very low monthly returns despite healthy trade frequency after tuning.
- Root cause: Router/Sizer parameters from `configs/strategies/ire.json` were not fully applied in TPA/backtest paths. Only `ire_min_conf_strong_short` was copied into `cfg.router`. The rest of `router` (e.g., `max_position_pct`) and all of `sizer` remained defaults.
- Impact: Router default cap `max_position_pct=0.05` kept target position around 5% of equity. With equity ≈ $107k and QQQ ≈ $570, this yields ≈ 9.4 shares. Combined with fractional sizing and a $1 min-trade threshold in the runner, the system emitted many micro-adjustments with tiny notional, constraining returns.

## Evidence
- Audit sample (`audit/IRE_last10days_trades.txt`):
  - `pos_after≈9.35–9.46` shares of QQQ; equity `≈ $107k`.
  - Numerous fills with `qty≈0.002–0.01` at `px≈$570` → $1–$6 notional per trade.
- Config JSON (`configs/strategies/ire.json`) sets higher caps (e.g., `max_position_pct=0.30`), but these were not reflected at runtime due to incomplete config propagation in `src/main.cpp`.

## Technical Root Cause
- In `src/main.cpp`, both `backtest` and `tpa_test` branches call `load_strategy_config_if_any(...)`, which populates a temporary `tmp_cfg`. However, only this field is copied:
  - `cfg.router.ire_min_conf_strong_short = tmp_cfg.router.ire_min_conf_strong_short;`
- Missing propagation:
  - Full `router` struct (e.g., `min_signal_strength`, `signal_multiplier`, `max_position_pct`, `min_shares`, `lot_size`, family symbols).
  - Full `sizer` struct (e.g., `min_notional`, `max_leverage`, `max_position_pct`, `volatility_target`, `allow_negative_cash`, `cash_reserve_pct`).
- As a result, runtime used defaults from `RouterCfg` and `SizerCfg`, not the tuned JSON, capping position at 5% and allowing fractional micro-deltas.

## Secondary Contributors
- Runner per-trade filter `abs(dq*px) > 1.0` allows $1 notional trades, causing dust fills.
- Fractional shares allowed → fine-grained micro-churn.
- Router and Sizer caps may be inconsistent when JSON isn’t applied, compounding constraints.

## Fix
1) Apply full config structs in `src/main.cpp` after loading `tmp_cfg`:
```
// after load_strategy_config_if_any(...)
cfg.router = tmp_cfg.router;
cfg.sizer  = tmp_cfg.sizer;
```
2) Raise the min notional threshold in the runner to eliminate dust (e.g., `$50–$200`).
```
// runner.cpp
if (std::abs(dq * px) <= 50.0) return; // instead of 1.0
```
3) Optionally set `fractional_allowed=false` or round to a coarser step to avoid micro-churn.
4) Ensure Router cap (`router.max_position_pct`) and Sizer cap (`sizer.max_position_pct`) are coherent.

## Expected Outcome
- With JSON parameters applied (e.g., `max_position_pct=0.15` in Router and larger Sizer limits), the sizer will target materially larger positions.
- Raising the min notional filter will remove dust fills, concentrate trades, and increase realized returns without inflating trade count.

## Relevant Source Modules
- `src/main.cpp` (config propagation bug)
- `src/runner.cpp` (min-notional threshold, fill gating)
- `include/sentio/router.hpp`, `src/router.cpp` (router capping)
- `include/sentio/sizer.hpp` (sizer constraints and equity-based sizing)
- `include/sentio/runner.hpp` (RunnerCfg)
- `configs/strategies/ire.json` (intended parameters)
- `megadocs/IRE_ROUTER_SIZER_UPGRADE_MEGA_DOC.md` (design for improved sizing policy)

## Reproduction Steps
1) Run `./build/sentio_cli tpa_test QQQ --strategy IRE --quarters 1`.
2) Observe fills `audit/IRE_*_q1.jsonl` and export last 10 days via `tools/emit_last10_trades.py`.
3) Note small `qty` deltas and `pos_after≈5% equity / price`.

## Verification Plan
1) Apply the config propagation fix and rebuild.
2) Set `router.max_position_pct` and `sizer.max_position_pct` to coherent values (e.g., 0.15 and 0.30), raise `sizer.min_notional` and runner’s $-threshold.
3) Re-run TPA and compare:
   - Larger per-trade notionals, fewer dust fills.
   - Improved monthly returns and similar/controlled trade frequency.

---

Prepared for: Sentio IRE strategy stack
Owner: Strategy Engineering
Status: Open (fix ready to apply)
