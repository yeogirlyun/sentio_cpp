# Sentio PPO Allocation Manager — Requirements

## Goal
Use PPO as the allocation manager in Sentio’s backend to maximize daily profit given a per-minute Transformer probability signal. The system is a high-frequency scalping loop (minute bars), entering/closing positions many times intraday and fully flat at market close. Target 100+ trades per 8-hour session; more trades are acceptable if profit increases after costs.

## Current Architecture (High-Level)
- Strategy layer: `TransformerStrategy` outputs probability p ∈ [0,1] each bar based on enhanced features.
- Backend execution: router + sizer + runner consume signals to create orders and manage positions across QQQ family (QQQ, PSQ, TQQQ, SQQQ).
- Audit: all decisions/events recorded to SQLite via `AuditDBRecorder`; canonical evaluation and metrics via `UnifiedMetrics`.

Data flow
```
Bars → EnhancedFeaturePipeline → TransformerStrategy.prob → Backend (router/sizer/runner) → Orders/Fills → Audit
```

Key behaviors
- Minute-level cadence; deterministic Trading Block evaluation.
- Aggressive leverage usage for strong signals; profit maximization mandate.
- End-of-day (EOD) hard flat: close all positions by market close.

## PPO Allocation Manager — Proposed Design
### Role
Replace or augment router/sizer with a PPO agent that decides per-minute allocation actions using the strategy probability and market context to maximize realized intraday PnL subject to constraints.

### Observation Space (per minute)
- Strategy state: probability p, recent p deltas, rolling mean/var of p.
- Market state: recent returns, realized volatility, microstructure proxies (spread proxy, imbalance proxy), volume rate-of-change.
- Position state: current inventory per instrument, unrealized PnL, time-to-close.
- Risk context: drawdown state, turnover in last N minutes, trade count in session.

### Action Space
- Target weight vector over instruments {QQQ, TQQQ, PSQ, SQQQ} or delta-position actions.
- Optional discrete actions: {increase long 3x, increase short 3x, flatten, rotate to 1x, scale ±k%}.

### Reward Shaping (per step = minute)
- Primary: realized PnL_t − costs_t.
- Costs: commissions + slippage + spread penalty + inventory penalty.
- Regularizers: turnover penalty, drawdown penalty, EOD flatness bonus.
- Optional calibration bonus: alignment of actions with informative probabilities.

### Constraints & Safety
- Position limits per instrument and aggregate leverage cap.
- Hard EOD flat: terminal step forces full close, with penalty if residual.
- Circuit breaker hooks (reuse existing `circuit_breaker`, `execution_verifier`).

### Training Regime
- Offline PPO using historical minute bars (or MarS future QQQ tracks) with a gym-like environment:
  - Episode: one Trading Block session (e.g., 480 bars) with warmup.
  - Observation builder mirrors live backend state.
  - Action applied → router/runner mock execution → fills with cost model.
  - Reward computed; transition logged; GAE advantage and PPO updates.
- Curriculum: start with 1x instruments then add 3x; increase action complexity gradually.

### Inference Path (Live)
```
Bars → TransformerStrategy.prob → PPO Allocation Manager → target weights → Runner → Orders/Fills → Audit
```

### Integration Points
- Replace `router`/`sizer` with `AllocationPolicy` interface; provide `PPOAllocationPolicy` implementation.
- Minimal changes to `runner`: consume target weights produced per bar.
- Keep feature and probability generation unchanged; PPO is strictly allocation.

## KPIs & Evaluation
- Profit: daily PnL, Monthly Projected Return (MPR), Sharpe, max drawdown.
- Microstructure efficiency: slippage per trade, spread capture, realized vs expected turnover.
- Operational: trades per session (≥100), EOD flatness 100%, constraint violations = 0.
- Signal-to-PnL correlation: monotonicity between p and realized returns after actions.

## Interfaces (Proposed)
```cpp
// Narrow, strategy-agnostic interface
struct AllocationDecision { std::string instrument; double target_weight; double confidence; };
class AllocationPolicy {
public:
    virtual ~AllocationPolicy() = default;
    virtual std::vector<AllocationDecision> decide(
        double probability,
        const std::vector<Bar>& bars,
        int current_index,
        const Portfolio& portfolio,
        const RiskState& risk_state,
        const std::vector<std::string>& instruments) = 0;
};

class PPOAllocationPolicy : public AllocationPolicy { /* loads PPO policy, outputs weights */ };
```

## Rollout Plan
1. Define `AllocationPolicy` and wire into `runner` behind a config switch.
2. Build gym-style environment (offline) for PPO using canonical Trading Blocks.
3. Train PPO with conservative costs; validate with evaluator and audit parity.
4. Shadow-mode live: run PPO decisions side-by-side, compare with baseline router.
5. Promote PPO allocation in production when KPIs exceed baseline.

## Risks & Mitigations
- Overtrading: turnover penalty and realistic cost model.
- Distribution shift: continual retraining and drift monitors.
- Latency: per-minute cadence is forgiving; ensure inference < 5ms.
- Stability: clip actions, enforce hard limits, circuit breaker integration.

## Success Criteria
- ≥ 10% monthly projected return with acceptable drawdown.
- ≥ 100 trades/session with positive net expectancy after costs.
- Strong Sharpe improvement vs baseline router/sizer.
- Zero EOD residual positions; zero constraint violations.
