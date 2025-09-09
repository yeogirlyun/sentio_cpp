# IRE Strategy Router/Sizer Upgrade Mega Doc

Generated: Auto

## Executive Summary
- Issue: Very small trade quantities indicate low capital utilization and constrained router/sizer behavior, leading to small profit gains.
- Goal: Define a production-grade, per-strategy router/sizer for IRE that allocates across base (QQQ) and leverage (TQQQ/SQQQ) with robust risk controls, targeting 10–100 trades/day and higher capital utilization while preserving drawdown limits.

## Current IRE Algorithm (Overview)
- Source: `include/sentio/strategy_ire.hpp`, `src/strategy_ire.cpp`, `include/sentio/strategy/rules/integrated_rule_ensemble.hpp`
- Ensemble: Weighted rule probabilities with reliability weighting (Brier+Hit) and agreement boost → emit `p_up ∈ [0,1]`.
- Router (current): Threshold + hysteresis; for IRE, STRONG_BUY/STRONG_SELL only; longs split 50/50 QQQ/TQQQ; shorts only SQQQ gated by `ire_min_conf_strong_short`.
- Sizer (current): Vol-targeting with conservative caps; min notional, round lots; appears to yield very small incremental trades.

## Observed Behavior (Last 10 Trading Days)
- Low per-fill notionals; end-of-day equity roughly unchanged; high cash fraction inferred.
- Audit: `audit/IRE_tpa_test_...jsonl` shows frequent micro-adjustments with minimal capital change.

## Root Causes of Low Utilization
1) Sizer parameters too conservative (low annual vol target, tight max fraction, small round lot/min notional).
2) Router emits many small-strength adjustments rather than larger target steps.
3) IRE-specific rule gating (only strong signals) combined with additional cooldown/hysteresis reduces allocation swings.
4) No explicit target gross exposure policy (e.g., base+leverage budget) → sizer defaults keep exposures tiny.

## Requirements: Professional-Grade Router/Sizer for IRE

### Router (Per-Strategy, Probability-Driven)
- Inputs: `p_up` and optional meta (agreement, variance).
- Threshold bands (strategy-tuned):
  - buy_lo/buy_hi, sell_hi/sell_lo with hysteresis.
  - `ire_min_conf_strong_short` gate for SQQQ entries.
- Directional intent map:
  - p_up ≥ buy_hi → Strong Long intent.
  - p_up ≤ sell_lo → Strong Short intent.
  - Between bands → Hold/maintain.
- Cooldown and min-hold bars per symbol.
- Trade budget signal: produce a normalized conviction score c ∈ [0,1] from |p_up − 0.5| scaled by agreement.

### Sizer (Exposure Policy + Risk Controls)
- Targets per direction:
  - Long: allocate across QQQ and TQQQ with a leverage mix (e.g., 40–60% QQQ, 60–40% TQQQ) configurable.
  - Short: allocate SQQQ only (base short disabled due to alignment constraints).
- Exposure curve:
  - Map conviction c to target gross exposure fraction g(c) using a smooth S-curve:
    - g(c) = g_min + (g_max − g_min) · smoothstep(c; k, mid).
  - Separate curves for long and short sides.
- Volatility targeting overlay:
  - Scale position to hit target annualized volatility; combine with g(c).
- Risk limits:
  - Max gross leverage, max position per instrument, max notional per trade, min notional.
  - Daily loss limit, trailing drawdown clamp.
  - Per-bar clamp delta to avoid churning.
- Rounding/lotting: nearest lot; skip if below `min_notional`.

### Position Split Policy (Long)
- Base-Leverage split based on realized/forecast vol:
  - If realized vol high, bias to QQQ (reduce TQQQ weight).
  - If realized vol low and conviction high, bias to TQQQ.
  - Configurable weights: `(w_base_lo, w_base_hi)` mapping from vol to weights.

### Execution Planning
- Single-bar immediate planning with partial fills allowed.
- Optional TWAP over N bars when target change exceeds threshold.

### Telemetry and Audit
- Log router conviction, thresholds hit, side, and sizer target fractions.
- Log exposure split (QQQ/TQQQ/SQQQ), risk clamps applied, and final planned shares.

## Parameter Set (to Tune per Strategy)
- Router: buy_lo, buy_hi, sell_hi, sell_lo, hysteresis, cooldown, `ire_min_conf_strong_short`.
- Sizer: target_annual_vol_long/short, g_min/g_max, S-curve (k, mid), max_fraction, max_gross_leverage, min_notional, round_lot.
- Split: base/leverage weights by vol bands; thresholds for vol regime.

## Example: Target 10–100 Trades/Day
- Tune buy/sell bands to trigger on moderate-to-strong signals.
- Use per-bar clamp to reduce micro-churn, enabling fewer but larger adjustments.
- Increase g_max and target vol to deploy more capital per trade while respecting risk.

## Implementation Plan (High-Level)
1) Extend `RouterCfg` with IRE conviction and thresholds.
2) Add `SizerCfgIRE` supporting S-curve exposure mapping and split policy.
3) Implement `IREPositionSizer` using exposure policy + vol targeting + clamps.
4) Integrate in `runner.cpp` path for IRE routing; emit audit with conviction and splits.
5) Expose parameters via `configs/strategies/ire.json`; add Bayesian tuner for these fields.

## Relevant Source Modules
- `include/sentio/strategy_ire.hpp`
- `src/strategy_ire.cpp`
- `include/sentio/strategy/rules/integrated_rule_ensemble.hpp`
- `include/sentio/router.hpp`
- `src/runner.cpp`
- `include/sentio/sizer.hpp`
- `include/sentio/audit.hpp`
- `src/audit.cpp`
- `include/sentio/cost_model.hpp`

## Next Steps
- Implement `IREPositionSizer` and router enhancements.
- Tune parameters on recent training window with audit disabled, validate OOS.
- Re-run tpa and publish detailed trade review with improved utilization.
