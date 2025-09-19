# Requirement: Extend SIGOR with Integrated Rule-Based Detectors (Non-Ensemble)

## Objective
Integrate the core logic of our existing rule-based strategies directly into `SignalOrStrategy` (sigor) as internal, reusable “detectors.” The goal is to strengthen the probability output toward high-confidence long/short when a majority of detectors align directionally, and neutralize toward 0.5 when detectors disagree. This is not an ensemble of strategies; rather, sigor remains a single strategy whose probability generator incorporates multiple rule-based detectors.

## Constraints and Principles
- Single source of truth: No duplicate strategy implementations. Extract core rules from previous strategies, refactor as detectors, and consume them inside sigor.
- Strategy-agnostic backend contract: Do not add new Router/Sizer/Runner signatures. Use `BaseStrategy` APIs only.
- Profit maximization defaults: 100% capital usage with maximum leverage (TQQQ/SQQQ, PSQ for weak sell) remains enforced by backend and allocation manager.
- No conflicting positions; one transaction per bar maintained by backend.
- Deterministic and canonical evaluation consistency (strattest ↔ audit) must be preserved.

## High-Level Design
1. Detector Interface
   - Create a lightweight internal interface for detectors inside sigor or a small header in `include/sentio/signal_utils.hpp`:
     - `double score(const std::vector<Bar>& bars, int idx)` returns probability in [0, 1], where <0.5 = short, >0.5 = long, 0.5 = neutral.
     - Detectors must be pure, stateless or minimally stateful, and deterministic.

2. Detectors to Implement (extracted from prior rule strategies)
   - RSI Reversion (from `rsi_strategy`)
   - Bollinger Squeeze/Breakout (from `strategy_bollinger_squeeze_breakout`)
   - Momentum-Volume (from `strategy_momentum_volume` and `rules/momentum_volume_rule.hpp`)
   - Opening Range Breakout (from `strategy_opening_range_breakout`)
   - VWAP Reversion (from `strategy_vwap_reversion`)
   - Order Flow Imbalance (from `strategy_order_flow_imbalance` and `rules/ofi_proxy_rule.hpp`)
   - Market Microstructure/Market-Making Signal (directional filter component only; avoid inventory/spread logic)

3. Aggregation Logic (inside sigor)
   - Compute each detector probability at bar `idx`.
   - Map each to direction: long if p > (0.5 + eps), short if p < (0.5 - eps), neutral otherwise.
   - Majority voting over directional signals:
     - Majority long → boost final p toward high band (e.g., clamp to [0.7, 0.95] depending on unanimity/confidence).
     - Majority short → depress final p toward low band (e.g., clamp to [0.05, 0.3]).
     - Mixed or weak consensus → soften toward 0.5 (neutralization).
   - Confidence shaping:
     - Weight detectors by historical reliability if available (future enhancement), else equal weight.
     - Apply softmax-like sharpening when detectors are aligned to increase separation from 0.5.

4. Signal-to-Allocation Mapping (unchanged contract)
   - The backend Allocation Manager maps probability to instruments:
     - p > 0.7 → TQQQ (100%)
     - 0.51 < p ≤ 0.7 → QQQ (100%)
     - p < 0.3 → SQQQ (100%)
     - 0.3 ≤ p < 0.49 → PSQ (100%)
     - else → CASH
   - Backend enforces no conflicting positions and one transaction per bar.

## Deliverables
1. Refactor points
   - Extract detector cores from prior strategies into pure functions/classes:
     - RSI: `include/sentio/rsi_strategy.hpp`, `src/rsi_strategy.cpp`
     - Bollinger: `include/sentio/strategy_bollinger_squeeze_breakout.hpp`, `src/strategy_bollinger_squeeze_breakout.cpp`
     - Momentum-Volume: `include/sentio/rules/momentum_volume_rule.hpp`, `src/strategy_momentum_volume.cpp`
     - Opening Range: `include/sentio/strategy_opening_range_breakout.hpp`, `src/strategy_opening_range_breakout.cpp`
     - VWAP: `include/sentio/strategy_vwap_reversion.hpp`, `src/strategy_vwap_reversion.cpp`
     - OFI: `include/sentio/rules/ofi_proxy_rule.hpp`, `src/strategy_order_flow_imbalance.cpp`
     - Market-Making directional filter: `include/sentio/strategy_market_making.hpp`, `src/strategy_market_making.cpp`
   - Consolidate detector utilities under `include/sentio/signal_utils.hpp` if needed.

2. SIGOR modifications
   - Add detector registry within `SignalOrStrategy` config (enable/disable, weights, thresholds).
   - Implement `calculate_probability()` to call detectors and perform majority consensus boosting/neutralization.
   - Preserve existing OR mixer for future extensibility, but primary path should be detector-based generation.

3. Tests
   - Unit tests per detector: monotonicity and directional sanity (uptrend favors long; downtrend favors short).
   - Integration test: mixed regimes (trend, mean-reversion, chop) verifying:
     - Majority alignment boosts |p-0.5|; conflicts reduce |p-0.5|.
     - Backend allocation decisions and audit events consistent.
   - Performance checks: higher alignment frequency → higher total return and Sharpe; conflicting regimes → mitigated losses.

4. Audit & Reporting
   - Record `StrategySignalAuditEvent` with per-detector components and final probability.
   - In `audit_cli` signal-flow, add a “detector breakdown” table (counts, avg p, alignment rate).

## Acceptance Criteria
- SIGOR remains a single strategy; no new external strategies introduced.
- All detectors are pure and reusable without side effects; no duplication of legacy strategies.
- Probability output reflects majority consensus with tunable bands; conflicts neutralize toward 0.5.
- Backtests via canonical evaluation show consistent P&L across strattest and audit.
- Instrument distribution tables always include PSQ/QQQ/TQQQ/SQQQ with correct P&L.
- Unit/integration tests pass; audit trail includes per-detector signal attribution.

## Relevant Modules to Include in Mega Document
- Strategy & Signal
  - `include/sentio/strategy_signal_or.hpp`, `src/strategy_signal_or.cpp`
  - `include/sentio/signal_engine.hpp`, `src/signal_engine.cpp`
  - `include/sentio/signal_gate.hpp`, `src/signal_gate.cpp`
  - `include/sentio/signal_utils.hpp`
  - `include/sentio/strategy_bollinger_squeeze_breakout.hpp`, `src/strategy_bollinger_squeeze_breakout.cpp`
  - `include/sentio/rsi_strategy.hpp`, `src/rsi_strategy.cpp`
  - `include/sentio/rules/momentum_volume_rule.hpp`
  - `include/sentio/rules/ofi_proxy_rule.hpp`
  - `include/sentio/strategy_vwap_reversion.hpp`, `src/strategy_vwap_reversion.cpp` (historical reference)
  - `include/sentio/strategy_opening_range_breakout.hpp`, `src/strategy_opening_range_breakout.cpp` (historical reference)
  - `include/sentio/strategy_market_making.hpp`, `src/strategy_market_making.cpp` (directional filter reference)

- Backend & Execution
  - `include/sentio/backend_architecture.hpp`, `src/backend_architecture.cpp`
  - `include/sentio/allocation_manager.hpp`, `src/allocation_manager.cpp`
  - `include/sentio/base_strategy.hpp`
  - `include/sentio/virtual_market.hpp`, `src/virtual_market.cpp`

- Audit & Metrics
  - `audit/src/audit_cli.cpp`
  - `include/sentio/unified_metrics.hpp`, `src/unified_metrics.cpp`
  - `include/sentio/backend_audit_events.hpp`, `src/backend_audit_events.cpp`

- Architecture
  - `docs/architecture.md`

## Rollout Plan
1) Implement detectors (RSI, Bollinger, Momentum-Volume, ORB, VWAP, OFI, MM directional filter).
2) Wire into sigor probability with consensus boosting/neutralization.
3) Add audit logging for detector attribution.
4) Update audit CLI to display detector breakdown.
5) Backtest with canonical evaluation (20 TB) and verify alignment of metrics and P&L.
6) Optimize thresholds and bands; finalize defaults in `SignalOrCfg`.

## Risks & Mitigations
- Overfitting bands/thresholds → keep defaults moderate; expose config with safe ranges.
- Detector disagreement in chop → neutralization toward 0.5 reduces churn; backend enforces single trade per bar.
- Performance regressions → maintain unit/integration tests and canonical evaluation benchmarks.


