# PPO Allocation Manager (Focused Mega Doc)

**Generated**: 2025-09-20 02:49:51
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: Focused core modules + requirements for PPO allocation over Transformer probability; minute-level scalping, EOD flat.

**Total Files**: 15

---

## 📋 **TABLE OF CONTENTS**

1. [temp_ppo_mega/PPO_ALLOCATION_MANAGER_REQUIREMENTS.md](#file-1)
2. [temp_ppo_mega/include/sentio/base_strategy.hpp](#file-2)
3. [temp_ppo_mega/include/sentio/feature_pipeline.hpp](#file-3)
4. [temp_ppo_mega/include/sentio/router.hpp](#file-4)
5. [temp_ppo_mega/include/sentio/runner.hpp](#file-5)
6. [temp_ppo_mega/include/sentio/sizer.hpp](#file-6)
7. [temp_ppo_mega/include/sentio/strategy_transformer.hpp](#file-7)
8. [temp_ppo_mega/include/sentio/transformer_model.hpp](#file-8)
9. [temp_ppo_mega/include/sentio/transformer_strategy_core.hpp](#file-9)
10. [temp_ppo_mega/src/base_strategy.cpp](#file-10)
11. [temp_ppo_mega/src/feature_pipeline.cpp](#file-11)
12. [temp_ppo_mega/src/router.cpp](#file-12)
13. [temp_ppo_mega/src/runner.cpp](#file-13)
14. [temp_ppo_mega/src/strategy_transformer.cpp](#file-14)
15. [temp_ppo_mega/src/transformer_model.cpp](#file-15)

---

## 📄 **FILE 1 of 15**: temp_ppo_mega/PPO_ALLOCATION_MANAGER_REQUIREMENTS.md

**File Information**:
- **Path**: `temp_ppo_mega/PPO_ALLOCATION_MANAGER_REQUIREMENTS.md`

- **Size**: 106 lines
- **Modified**: 2025-09-20 02:49:51

- **Type**: .md

```text
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

```

## 📄 **FILE 2 of 15**: temp_ppo_mega/include/sentio/base_strategy.hpp

**File Information**:
- **Path**: `temp_ppo_mega/include/sentio/base_strategy.hpp`

- **Size**: 167 lines
- **Modified**: 2025-09-20 02:49:51

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "signal_diag.hpp"
#include "router.hpp"  // for StrategySignal
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <functional>

namespace sentio {

// Forward declarations
struct RouterCfg;
struct SizerCfg;

// Strategy context for bar processing
struct StrategyCtx {
  std::string instrument;     // traded instrument for this stream
  std::int64_t ts_utc_epoch;  // bar timestamp (UTC seconds)
  bool is_rth{true};          // inject from your RTH checker
};

// Minimal strategy interface for ML integration
class IStrategy {
public:
  virtual ~IStrategy() = default;
  virtual void on_bar(const StrategyCtx& ctx, const Bar& b) = 0;
  virtual std::optional<StrategySignal> latest() const = 0;
};

// Parameter types and enums
enum class ParamType { INT, FLOAT };
struct ParamSpec { 
    ParamType type;
    double min_val, max_val;
    double default_val;
};

using ParameterMap = std::unordered_map<std::string, double>;
using ParameterSpace = std::unordered_map<std::string, ParamSpec>;

enum class SignalType { NONE = 0, BUY = 1, SELL = -1, STRONG_BUY = 2, STRONG_SELL = -2 };

struct StrategyState {
    bool in_position = false;
    SignalType last_signal = SignalType::NONE;
    int last_trade_bar = -1000; // Initialize far in the past
    
    void reset() {
        in_position = false;
        last_signal = SignalType::NONE;
        last_trade_bar = -1000;
    }
};

// StrategySignal is now defined in router.hpp

class BaseStrategy {
protected:
    std::string name_;
    ParameterMap params_;
    StrategyState state_;
    SignalDiag diag_;

    bool is_cooldown_active(int current_bar, int cooldown_period) const;
    
public:
    BaseStrategy(const std::string& name) : name_(name) {}
    virtual ~BaseStrategy() = default;

    // **MODIFIED**: Explicitly delete copy and move operations for this polymorphic base class.
    // This prevents object slicing and ownership confusion.
    BaseStrategy(const BaseStrategy&) = delete;
    BaseStrategy& operator=(const BaseStrategy&) = delete;
    BaseStrategy(BaseStrategy&&) = delete;
    BaseStrategy& operator=(BaseStrategy&&) = delete;
    
    std::string get_name() const { return name_; }
    
    virtual ParameterMap get_default_params() const = 0;
    virtual ParameterSpace get_param_space() const = 0;
    virtual void apply_params() = 0;
    
    void set_params(const ParameterMap& params);
    
    // **NEW**: Primary method - strategies should implement this to return probability (0-1)
    virtual double calculate_probability(const std::vector<Bar>& bars, int current_index) = 0;
    
    // **NEW**: Wrapper method that converts probability to StrategySignal
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) {
        double prob = calculate_probability(bars, current_index);
        return StrategySignal::from_probability(prob);
    }
    
    // REMOVED: get_allocation_decisions - strategies only produce probabilities
    // AllocationManager handles instrument selection for maximum profit
    
    // REMOVED: get_router_config - AllocationManager handles routing
    
    // REMOVED: get_sizer_config() - No artificial limits allowed for profit maximization
    // Sizer will always use maximum capital deployment and leverage
    
    // REMOVED: requires_dynamic_allocation - all strategies use same allocation pipeline
    
    // **CONFLICT PREVENTION (HARD DEFAULT)**: Disallow simultaneous positions by default.
    // Coordinator enforces family-level conflict rules; strategies should rarely override this.
    virtual bool allows_simultaneous_positions(const std::string& instrument1, const std::string& instrument2) const {
        (void)instrument1; (void)instrument2; // Suppress unused parameter warnings
        return false;
    }
    
    // **STRATEGY-AGNOSTIC TRANSITION CONTROL**: Let strategy control its own transitions
    virtual bool requires_sequential_transitions() const {
        // Default: allow simultaneous transitions (backward compatible)
        return false;
    }
    
    // **NEW**: Get strategy-specific signal processing (for audit/logging)
    virtual std::string get_signal_description(double probability) const {
        if (probability > 0.8) return "STRONG_BUY";
        if (probability > 0.6) return "BUY";
        if (probability < 0.2) return "STRONG_SELL";
        if (probability < 0.4) return "SELL";
        return "HOLD";
    }
    
    // **NEW**: Strategy evaluation interface
    virtual std::vector<double> get_probability_history() const { return {}; }
    virtual std::vector<double> get_signal_history() const { return {}; }
    virtual void reset_evaluation_data() {}
    virtual bool supports_evaluation() const { return false; }
    virtual void reset_state();
    
    const SignalDiag& get_diag() const { return diag_; }
};

class StrategyFactory {
public:
    using CreateFunction = std::function<std::unique_ptr<BaseStrategy>()>;
    
    static StrategyFactory& instance();
    void register_strategy(const std::string& name, CreateFunction create_func);
    std::unique_ptr<BaseStrategy> create_strategy(const std::string& name);
    std::vector<std::string> get_available_strategies() const;
    
private:
    std::unordered_map<std::string, CreateFunction> strategies_;
};

// Strategy initialization function
bool initialize_strategies();

// **NEW**: The final, more robust registration macro.
// It takes the C++ ClassName and the "Name" to be used by the CLI.
#define REGISTER_STRATEGY(ClassName, Name) \
    namespace { \
        struct ClassName##Registrar { \
            ClassName##Registrar() { \
                StrategyFactory::instance().register_strategy(Name, \
                    []() { return std::make_unique<ClassName>(); }); \
            } \
        }; \
        static ClassName##Registrar g_##ClassName##_registrar; \
    }

} // namespace sentio
```

## 📄 **FILE 3 of 15**: temp_ppo_mega/include/sentio/feature_pipeline.hpp

**File Information**:
- **Path**: `temp_ppo_mega/include/sentio/feature_pipeline.hpp`

- **Size**: 71 lines
- **Modified**: 2025-09-20 02:49:51

- **Type**: .hpp

```text
#pragma once

#include "transformer_strategy_core.hpp"
#include "core.hpp"  // For Bar definition
#include <torch/torch.h>
#include <vector>
#include <deque>
#include <memory>
#include <chrono>

namespace sentio {

// Simple feature pipeline for the transformer strategy
class FeaturePipeline {
public:
    static constexpr int MAX_GENERATION_TIME_US = 500;
    static constexpr int FEATURE_CACHE_SIZE = 10000;
    static constexpr int TOTAL_FEATURES = 128; // Original feature count
    static constexpr int ENHANCED_TOTAL_FEATURES = 173; // Enhanced feature count (128 + 45)
    
    explicit FeaturePipeline(const TransformerConfig::Features& config);
    
    // Main interface
    TransformerFeatureMatrix generate_features(const std::vector<Bar>& bars);
    TransformerFeatureMatrix generate_enhanced_features(const std::vector<Bar>& bars);
    void update_feature_cache(const Bar& new_bar);
    std::vector<Bar> get_cached_bars(int lookback_periods) const;

private:
    TransformerConfig::Features config_;
    std::deque<Bar> feature_cache_;
    std::vector<RunningStats> feature_stats_;
    
    // Feature generation methods
    std::vector<float> generate_price_features(const std::vector<Bar>& bars);
    std::vector<float> generate_volume_features(const std::vector<Bar>& bars);
    std::vector<float> generate_technical_features(const std::vector<Bar>& bars);
    std::vector<float> generate_temporal_features(const Bar& current_bar);
    
    // Enhanced feature groups (incremental to the 128 original features)
    std::vector<float> generate_momentum_persistence_features(const std::vector<Bar>& bars);
    std::vector<float> generate_volatility_regime_features(const std::vector<Bar>& bars);
    std::vector<float> generate_microstructure_features(const std::vector<Bar>& bars);
    std::vector<float> generate_options_features(const std::vector<Bar>& bars);
    
    // Technical indicators
    std::vector<float> calculate_sma(const std::vector<float>& prices, int period);
    std::vector<float> calculate_ema(const std::vector<float>& prices, int period);
    std::vector<float> calculate_rsi(const std::vector<float>& prices, int period = 14);
    
    // Utilities for enhanced features
    float calculate_realized_volatility(const std::vector<Bar>& bars, int period);
    float calculate_average_volume(const std::vector<Bar>& bars, int period);
    float calculate_skewness(const std::vector<float>& values);
    float calculate_kurtosis(const std::vector<float>& values);
    float calculate_trend_strength(const std::vector<Bar>& bars, int lookback);
    float calculate_volume_trend(const std::vector<Bar>& bars, int period);
    float calculate_volatility_persistence(const std::vector<Bar>& bars, int period);
    float calculate_volatility_clustering(const std::vector<Bar>& bars, int period);
    float calculate_mean_reversion_speed(const std::vector<Bar>& bars, int period);
    float calculate_gamma_exposure_proxy(const std::vector<Bar>& bars);
    float detect_unusual_volume_patterns(const std::vector<Bar>& bars, int period);
    float calculate_fear_greed_proxy(const std::vector<Bar>& bars);
    float calculate_resistance_breakthrough(const std::vector<Bar>& bars);
    
    // Normalization
    void update_feature_statistics(const std::vector<float>& features);
    std::vector<float> normalize_features(const std::vector<float>& features);
};

} // namespace sentio

```

## 📄 **FILE 4 of 15**: temp_ppo_mega/include/sentio/router.hpp

**File Information**:
- **Path**: `temp_ppo_mega/include/sentio/router.hpp`

- **Size**: 92 lines
- **Modified**: 2025-09-20 02:49:51

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <optional>
#include <string>

namespace sentio {

enum class OrderSide { Buy, Sell };

struct StrategySignal {
  enum class Type { BUY, STRONG_BUY, SELL, STRONG_SELL, HOLD };
  Type   type{Type::HOLD};
  double confidence{0.0}; // 0..1
  
  // **NEW**: Probability-based signal (0-1 where 1 = strong buy, 0 = strong sell)
  double probability{0.5}; // 0.5 = neutral/no signal
  
  // **NEW**: Convert probability to discrete signal for backward compatibility
  static StrategySignal from_probability(double prob) {
    StrategySignal signal;
    signal.probability = std::clamp(prob, 0.0, 1.0);
    
    if (prob > 0.7) {
      signal.type = Type::STRONG_BUY;
      signal.confidence = (prob - 0.7) / 0.3; // Scale 0.7-1.0 to 0-1
    } else if (prob > 0.505) {  // Narrower BUY threshold: was 0.51
      signal.type = Type::BUY;
      signal.confidence = (prob - 0.505) / 0.195; // Scale 0.505-0.7 to 0-1
    } else if (prob < 0.3) {
      signal.type = Type::STRONG_SELL;
      signal.confidence = (0.3 - prob) / 0.3; // Scale 0.0-0.3 to 0-1
    } else if (prob < 0.495) {  // Narrower SELL threshold: was 0.49
      signal.type = Type::SELL;
      signal.confidence = (0.495 - prob) / 0.195; // Scale 0.3-0.495 to 0-1
    } else {
      signal.type = Type::HOLD;  // Now 0.495-0.505 range (much smaller!)
      signal.confidence = 0.0;
    }
    
    return signal;
  }
};

struct RouterCfg {
  double min_signal_strength = 0.10; // below -> ignore
  double signal_multiplier   = 1.00; // scales target weight
  double max_position_pct    = 0.05; // +/- 5%
  bool   require_rth         = true; // assume ingest enforces RTH
  // family config
  std::string base_symbol{"QQQ"};
  std::string bull3x{"TQQQ"};
  std::string bear3x{"SQQQ"};
  // Note: moderate sell signals now use SHORT base_symbol instead of bear1x ETF
  // sizing
  double min_shares = 1.0;
  double lot_size   = 1.0; // for ETFs typically 1
  // IRE-specific gating for short side (use only SQQQ when confidence above threshold)
  double ire_min_conf_strong_short = 0.85; // trade SQQQ only if strong-sell confidence >= this
};

struct RouteDecision {
  std::string instrument;
  double      target_weight; // [-max, +max]
};

struct AccountSnapshot { double equity{0.0}; double cash{0.0}; };

struct Order {
  std::string instrument;
  OrderSide   side{OrderSide::Buy};
  double      qty{0.0};
  double      notional{0.0};
  double      limit_price{0.0}; // 0 = market
  std::int64_t ts_utc{0};
  std::string signal_id;
};

class PriceBook; // fwd
double last_trade_price(const PriceBook& book, const std::string& instrument);

std::optional<RouteDecision> route(const StrategySignal& s, const RouterCfg& cfg, const std::string& base_symbol);

// High-level convenience: route + size into a market order
Order route_and_create_order(const std::string& signal_id,
                             const StrategySignal& sig,
                             const RouterCfg& cfg,
                             const std::string& base_symbol,
                             const PriceBook& book,
                             const AccountSnapshot& acct,
                             std::int64_t ts_utc);

} // namespace sentio
```

## 📄 **FILE 5 of 15**: temp_ppo_mega/include/sentio/runner.hpp

**File Information**:
- **Path**: `temp_ppo_mega/include/sentio/runner.hpp`

- **Size**: 63 lines
- **Modified**: 2025-09-20 02:49:51

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "audit.hpp"
#include "router.hpp"
#include "safe_sizer.hpp"
// REMOVED: position_manager.hpp - unused legacy file
#include "cost_model.hpp"
#include "symbol_table.hpp"
#include "dataset_metadata.hpp"
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration for canonical evaluation
#include "canonical_evaluation.hpp"

namespace sentio {
    class StrategyProfiler;  // Forward declaration
}

namespace sentio {

enum class AuditLevel { Full, MetricsOnly };

struct RunnerCfg {
    std::string strategy_name = "VWAPReversion";
    std::unordered_map<std::string, std::string> strategy_params;
    RouterCfg router;
    SafeSizerConfig sizer;
    AuditLevel audit_level = AuditLevel::Full;
    int snapshot_stride = 100;
    std::string audit_file = "audit.jsonl";  // JSONL audit file path
    bool skip_audit_run_creation = false;  // Skip audit run creation (for block processing)
};

// NEW: This struct holds the RAW output from a backtest simulation.
// It does not contain any calculated performance metrics.
struct BacktestOutput {
    std::vector<std::pair<std::string, double>> equity_curve;
    // Canonical: raw timestamps aligned with equity_curve entries (milliseconds since epoch)
    std::vector<std::int64_t> equity_curve_ts_ms;
    int total_fills = 0;
    int no_route_events = 0;
    int no_qty_events = 0;
    int run_trading_days = 0;
};

// REMOVED: The old RunResult struct is now obsolete.
// struct RunResult { ... };

// CHANGED: run_backtest now returns the raw BacktestOutput and accepts dataset metadata.
BacktestOutput run_backtest(IAuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                           int base_symbol_id, const RunnerCfg& cfg, const DatasetMetadata& dataset_meta = {}, 
                           StrategyProfiler* persistent_profiler = nullptr);

// NEW: Canonical evaluation using Trading Block system for deterministic performance measurement
CanonicalReport run_canonical_backtest(IAuditRecorder& audit, const SymbolTable& ST, 
                                      const std::vector<std::vector<Bar>>& series, int base_symbol_id, 
                                      const RunnerCfg& cfg, const DatasetMetadata& dataset_meta, 
                                      const TradingBlockConfig& block_config);

} // namespace sentio


```

## 📄 **FILE 6 of 15**: temp_ppo_mega/include/sentio/sizer.hpp

**File Information**:
- **Path**: `temp_ppo_mega/include/sentio/sizer.hpp`

- **Size**: 102 lines
- **Modified**: 2025-09-20 02:49:51

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include "position_validator.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace sentio {

// Profit-Maximizing Sizer Configuration - NO ARTIFICIAL LIMITS
struct SizerCfg {
  bool fractional_allowed = true;
  double min_notional = 1.0;
  // REMOVED: All artificial constraints that limit profit
  // - max_leverage: Always use maximum available leverage
  // - max_position_pct: Always use 100% of capital
  // - allow_negative_cash: Always enabled for margin trading
  // - cash_reserve_pct: No cash reserves, deploy 100% of capital
  double volatility_target = 0.15;  // Keep for volatility targeting only
  int vol_lookback_days = 20;       // Keep for volatility calculation only
};

// Advanced Sizer Class with Multiple Constraints
class AdvancedSizer {
public:
  double calculate_volatility(const std::vector<Bar>& price_history, int lookback) const {
    if (price_history.size() < static_cast<size_t>(lookback)) return 0.05; // Default vol

    std::vector<double> returns;
    returns.reserve(lookback - 1);
    for (size_t i = price_history.size() - lookback + 1; i < price_history.size(); ++i) {
      double prev_close = price_history[i-1].close;
      if (prev_close > 0) {
        returns.push_back(price_history[i].close / prev_close - 1.0);
      }
    }
    
    if (returns.size() < 2) return 0.05;
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - mean) * (ret - mean);
    }
    variance /= returns.size();
    return std::sqrt(variance) * std::sqrt(252.0); // Annualized
  }

  // **PROFIT MAXIMIZATION**: Deploy 100% of capital with maximum leverage
  double calculate_target_quantity(const Portfolio& portfolio,
                                   const SymbolTable& ST,
                                   const std::vector<double>& last_prices,
                                   const std::string& instrument,
                                   double target_weight,
                                                                       [[maybe_unused]] const std::vector<Bar>& price_history,
                                   const SizerCfg& cfg) const {
    
    const double equity = equity_mark_to_market(portfolio, last_prices);
    int instrument_id = ST.get_id(instrument);

    if (equity <= 0 || instrument_id == -1 || last_prices[instrument_id] <= 0) {
        return 0.0;
    }
    
    double instrument_price = last_prices[instrument_id];

    // **PROFIT MAXIMIZATION MANDATE**: Use 100% of capital with maximum leverage
    // No artificial constraints - let the strategy determine optimal allocation
    double desired_notional = equity * std::abs(target_weight);
    
    // Apply minimum notional filter only (to avoid dust trades)
    if (desired_notional < cfg.min_notional) return 0.0;
    
    double qty = desired_notional / instrument_price;
    double final_qty = cfg.fractional_allowed ? qty : std::floor(qty);
    
    // Return with the correct sign (long/short)
    return (target_weight > 0) ? final_qty : -final_qty;
  }

  // **NEW**: Weight-to-shares helper for portfolio allocator integration
  long long target_shares_from_weight(double target_weight, double equity, double price, const SizerCfg& cfg) const {
    if (price <= 0 || equity <= 0) return 0;
    
    // weight = position_notional / equity ⇒ shares = weight * equity / price
    double desired_notional = target_weight * equity;
    long long shares = (long long)std::floor(std::abs(desired_notional) / price);
    
    // Apply min notional filter
    if (shares * price < cfg.min_notional) {
      shares = 0;
    }
    
    // Apply sign
    if (target_weight < 0) shares = -shares;
    
    return shares;
  }
};

} // namespace sentio
```

## 📄 **FILE 7 of 15**: temp_ppo_mega/include/sentio/strategy_transformer.hpp

**File Information**:
- **Path**: `temp_ppo_mega/include/sentio/strategy_transformer.hpp`

- **Size**: 99 lines
- **Modified**: 2025-09-20 02:49:51

- **Type**: .hpp

```text
#pragma once

#include "base_strategy.hpp"
#include "transformer_strategy_core.hpp"
#include "transformer_model.hpp"
#include "feature_pipeline.hpp"
#include "online_trainer.hpp"
#include "adaptive_allocation_manager.hpp"
#include <memory>
#include <deque>
#include <torch/torch.h>

namespace sentio {

struct TransformerCfg {
    // Model configuration
    int feature_dim = 128;
    int sequence_length = 64;
    int d_model = 256;
    int num_heads = 8;
    int num_layers = 6;
    int ffn_hidden = 1024;
    float dropout = 0.1f;
    
    // Strategy parameters
    float buy_threshold = 0.6f;
    float sell_threshold = 0.4f;
    float strong_threshold = 0.8f;
    float conf_floor = 0.5f;
    
    // Training parameters
    bool enable_online_training = true;
    int update_interval_minutes = 60;
    int min_samples_for_update = 1000;
    
    // Model paths
    std::string model_path = "artifacts/Transformer/v1/model.pt";
    std::string artifacts_dir = "artifacts/Transformer/";
    std::string version = "v1";
};

class TransformerStrategy : public BaseStrategy {
public:
    TransformerStrategy();
    explicit TransformerStrategy(const TransformerCfg& cfg);
    
    // BaseStrategy interface
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override;
    
    // Allocation decisions for profit maximization
    std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, 
        int current_index,
        const std::string& base_symbol = "QQQ",
        const std::string& bull3x_symbol = "TQQQ", 
        const std::string& bear3x_symbol = "SQQQ"
    ) const;

private:
    // Configuration
    TransformerCfg cfg_;
    
    // Core components
    std::shared_ptr<TransformerModel> model_;
    std::unique_ptr<FeaturePipeline> feature_pipeline_;
    std::unique_ptr<OnlineTrainer> online_trainer_;
    
    // Feature management
    std::deque<Bar> bar_history_;
    std::vector<double> current_features_;
    
    // Model state
    bool model_initialized_ = false;
    std::atomic<bool> is_training_{false};
    
    // Performance tracking
    mutable std::mutex metrics_mutex_;
    PerformanceMetrics current_metrics_;
    
    // Helper methods
    void initialize_model();
    void update_bar_history(const Bar& bar);
    void maybe_trigger_training();
    TransformerFeatureMatrix generate_features_for_bars(const std::vector<Bar>& bars, int end_index);
    
    // Validation methods
    bool validate_tensor_dimensions(const torch::Tensor& tensor, 
                                   const std::vector<int64_t>& expected_dims,
                                   const std::string& tensor_name);
    bool validate_configuration();
    
    // Feature conversion
    std::vector<Bar> convert_to_transformer_bars(const std::vector<Bar>& sentio_bars) const;
};

} // namespace sentio

```

## 📄 **FILE 8 of 15**: temp_ppo_mega/include/sentio/transformer_model.hpp

**File Information**:
- **Path**: `temp_ppo_mega/include/sentio/transformer_model.hpp`

- **Size**: 43 lines
- **Modified**: 2025-09-20 02:49:51

- **Type**: .hpp

```text
#pragma once

#include "transformer_strategy_core.hpp"
#include <torch/torch.h>
#include <memory>
#include <string>

namespace sentio {

// Simple transformer model implementation for Sentio
class TransformerModel : public torch::nn::Module {
public:
    explicit TransformerModel(const TransformerConfig& config);
    
    // Forward pass
    torch::Tensor forward(const torch::Tensor& input);
    
    // Model management
    void save_model(const std::string& path);
    void load_model(const std::string& path);
    void optimize_for_inference();
    
    // Utilities
    size_t get_parameter_count() const;
    size_t get_memory_usage_bytes() const;
    
private:
    TransformerConfig config_;
    
    // Model components
    torch::nn::Linear input_projection_{nullptr};
    torch::nn::TransformerEncoder transformer_{nullptr};
    torch::nn::LayerNorm layer_norm_{nullptr};
    torch::nn::Linear output_projection_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    
    // Positional encoding
    torch::Tensor pos_encoding_;
    
    void create_positional_encoding();
};

} // namespace sentio
```

## 📄 **FILE 9 of 15**: temp_ppo_mega/include/sentio/transformer_strategy_core.hpp

**File Information**:
- **Path**: `temp_ppo_mega/include/sentio/transformer_strategy_core.hpp`

- **Size**: 251 lines
- **Modified**: 2025-09-20 02:49:51

- **Type**: .hpp

```text
#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <shared_mutex>
#include <chrono>
#include <string>
#include <unordered_map>
#include <torch/torch.h>

namespace sentio {

// Forward declarations
struct Bar;
class Fill;
class BaseStrategy;
struct StrategySignal;

// ==================== Core Data Structures ====================

struct PriceFeatures {
    std::vector<float> ohlc_normalized;     // 4 features
    std::vector<float> returns;             // 5 features (1m, 5m, 15m, 1h, 4h)
    std::vector<float> log_returns;         // 5 features
    std::vector<float> moving_averages;     // 8 features (SMA/EMA 5,10,20,50)
    std::vector<float> bollinger_bands;     // 3 features (upper, lower, %B)
    std::vector<float> rsi_family;          // 4 features (RSI 14, Stoch RSI)
    std::vector<float> momentum;            // 6 features (ROC, Williams %R, etc.)
    std::vector<float> volatility;          // 5 features (ATR, realized vol, etc.)
    
    static constexpr int TOTAL_FEATURES = 40;
};

struct VolumeFeatures {
    std::vector<float> volume_indicators;   // 8 features (VWAP, OBV, etc.)
    std::vector<float> volume_ratios;       // 4 features (vol/avg_vol ratios)
    std::vector<float> price_volume;        // 4 features (PVT, MFI, etc.)
    std::vector<float> volume_profile;      // 4 features (VPOC, VAH, VAL, etc.)
    
    static constexpr int TOTAL_FEATURES = 20;
};

struct MicrostructureFeatures {
    std::vector<float> spread_metrics;      // 5 features (bid-ask spread analysis)
    std::vector<float> order_flow;          // 8 features (tick direction, etc.)
    std::vector<float> market_impact;       // 4 features (Kyle's lambda, etc.)
    std::vector<float> liquidity_metrics;   // 4 features (market depth, etc.)
    std::vector<float> regime_indicators;   // 4 features (volatility regime, etc.)
    
    static constexpr int TOTAL_FEATURES = 25;
};

struct CrossAssetFeatures {
    std::vector<float> correlation_features; // 5 features (SPY, VIX correlation)
    std::vector<float> sector_rotation;      // 5 features (sector momentum)
    std::vector<float> macro_indicators;     // 5 features (yield curve, etc.)
    
    static constexpr int TOTAL_FEATURES = 15;
};

struct TemporalFeatures {
    std::vector<float> time_of_day;         // 8 features (hour encoding)
    std::vector<float> day_of_week;         // 7 features (weekday encoding)
    std::vector<float> monthly_seasonal;    // 12 features (month encoding)
    std::vector<float> market_session;      // 1 feature (RTH/ETH indicator)
    
    static constexpr int TOTAL_FEATURES = 28;
};

using FeatureVector = torch::Tensor;
using TransformerFeatureMatrix = torch::Tensor;

// ==================== Configuration Structures ====================

struct TransformerConfig {
    // Model architecture
    int feature_dim = 128;
    int sequence_length = 64;
    int d_model = 256;
    int num_heads = 8;
    int num_layers = 6;
    int ffn_hidden = 1024;
    float dropout = 0.1f;
    
    // Performance requirements
    float max_inference_latency_ms = 1.0f;
    float max_memory_usage_mb = 1024.0f;
    
    // Training configuration
    struct Training {
        struct Offline {
            int batch_size = 256;
            float learning_rate = 0.001f;
            int max_epochs = 1000;
            int patience = 50;
            float min_delta = 1e-6f;
            std::string optimizer = "AdamW";
            float weight_decay = 1e-4f;
        } offline;
        
        struct Online {
            int update_interval_minutes = 60;
            float learning_rate = 0.0001f;
            int replay_buffer_size = 10000;
            bool enable_regime_detection = true;
            float regime_change_threshold = 0.15f;
            int min_samples_for_update = 1000;
        } online;
    } training;
    
    // Feature configuration
    struct Features {
        enum class NormalizationMethod {
            Z_SCORE, MIN_MAX, ROBUST, QUANTILE_UNIFORM
        };
        NormalizationMethod normalization = NormalizationMethod::Z_SCORE;
        float decay_factor = 0.999f;
    } features;
};

// ==================== Performance Metrics ====================

struct PerformanceMetrics {
    float avg_inference_latency_ms = 0.0f;
    float p95_inference_latency_ms = 0.0f;
    float p99_inference_latency_ms = 0.0f;
    float recent_accuracy = 0.0f;
    float rolling_sharpe_ratio = 0.0f;
    float current_drawdown = 0.0f;
    float memory_usage_mb = 0.0f;
    bool is_training_active = false;
    float training_loss = 0.0f;
    int samples_processed = 0;
};

struct ValidationMetrics {
    float directional_accuracy = 0.0f;
    float sharpe_ratio = 0.0f;
    float max_drawdown = 0.0f;
    float win_rate = 0.0f;
    float profit_factor = 0.0f;
    bool passes_validation = false;
};

struct TrainingResult {
    bool success = false;
    std::string model_path;
    ValidationMetrics validation_metrics;
    std::chrono::system_clock::time_point training_end_time;
    int total_epochs = 0;
};

// ==================== Model Status ====================

enum class ModelStatus {
    UNINITIALIZED,
    LOADING,
    READY,
    TRAINING,
    UPDATING,
    ERROR,
    DISABLED
};

struct UpdateResult {
    bool success = false;
    std::string error_message;
    ValidationMetrics post_update_metrics;
    std::chrono::milliseconds update_duration{0};
};

// ==================== Risk Management ====================

struct RiskLimits {
    float max_position_size = 1.0f;
    float max_daily_trades = 100.0f;
    float max_drawdown_threshold = 0.10f;
    float min_confidence_threshold = 0.6f;
};

struct Alert {
    std::string metric_name;
    float current_value;
    float threshold;
    std::chrono::system_clock::time_point timestamp;
    std::string message;
};

// ==================== Running Statistics for Feature Normalization ====================

class RunningStats {
public:
    RunningStats(float decay_factor = 0.999f) : decay_factor_(decay_factor) {}
    
    void update(float value) {
        if (!initialized_) {
            mean_ = value;
            var_ = 0.0f;
            min_ = max_ = value;
            initialized_ = true;
            count_ = 1;
        } else {
            // Exponential moving average
            mean_ = decay_factor_ * mean_ + (1.0f - decay_factor_) * value;
            float delta = value - mean_;
            var_ = decay_factor_ * var_ + (1.0f - decay_factor_) * delta * delta;
            min_ = std::min(min_, value);
            max_ = std::max(max_, value);
            count_++;
        }
    }
    
    float mean() const { return mean_; }
    float std() const { return std::sqrt(var_); }
    float min() const { return min_; }
    float max() const { return max_; }
    int count() const { return count_; }
    bool is_initialized() const { return initialized_; }
    
private:
    float decay_factor_;
    float mean_ = 0.0f;
    float var_ = 0.0f;
    float min_ = 0.0f;
    float max_ = 0.0f;
    int count_ = 0;
    bool initialized_ = false;
};

// ==================== Core Constants ====================

struct LatencyRequirements {
    static constexpr int MAX_INFERENCE_LATENCY_US = 1000;    // 1ms
    static constexpr int TARGET_INFERENCE_LATENCY_US = 500;  // 0.5ms
    static constexpr int MAX_FEATURE_GEN_LATENCY_US = 500;   // 0.5ms
    static constexpr int MAX_MODEL_UPDATE_TIME_S = 300;      // 5 minutes
    static constexpr int MAX_MEMORY_USAGE_MB = 1024;         // 1GB
    static constexpr int MAX_GPU_MEMORY_MB = 2048;           // 2GB
};

struct AccuracyRequirements {
    static constexpr float MIN_DIRECTIONAL_ACCURACY = 0.52f;
    static constexpr float TARGET_DIRECTIONAL_ACCURACY = 0.55f;
    static constexpr float MIN_SHARPE_RATIO = 1.0f;
    static constexpr float TARGET_SHARPE_RATIO = 2.0f;
    static constexpr float MAX_DRAWDOWN = 0.15f;
    static constexpr float MIN_WIN_RATE = 0.45f;
};

} // namespace sentio

```

## 📄 **FILE 10 of 15**: temp_ppo_mega/src/base_strategy.cpp

**File Information**:
- **Path**: `temp_ppo_mega/src/base_strategy.cpp`

- **Size**: 54 lines
- **Modified**: 2025-09-20 02:49:51

- **Type**: .cpp

```text
#include "sentio/base_strategy.hpp"
#include <iostream>

namespace sentio {

void BaseStrategy::reset_state() {
    state_.reset();
    diag_ = SignalDiag{};
}

bool BaseStrategy::is_cooldown_active(int current_bar, int cooldown_period) const {
    return (current_bar - state_.last_trade_bar) < cooldown_period;
}

// **MODIFIED**: This function now merges incoming parameters (overrides)
// with the existing default parameters, preventing the defaults from being erased.
void BaseStrategy::set_params(const ParameterMap& overrides) {
    // The constructor has already set the defaults.
    // Now, merge the overrides into the existing params_.
    for (const auto& [key, value] : overrides) {
        params_[key] = value;
    }
    
    apply_params();
}

// --- Strategy Factory Implementation ---
StrategyFactory& StrategyFactory::instance() {
    static StrategyFactory factory_instance;
    return factory_instance;
}

void StrategyFactory::register_strategy(const std::string& name, CreateFunction create_func) {
    strategies_[name] = create_func;
}

std::unique_ptr<BaseStrategy> StrategyFactory::create_strategy(const std::string& name) {
    auto it = strategies_.find(name);
    if (it != strategies_.end()) {
        return it->second();
    }
    std::cerr << "Error: Strategy '" << name << "' not found in factory." << std::endl;
    return nullptr;
}

std::vector<std::string> StrategyFactory::get_available_strategies() const {
    std::vector<std::string> names;
    for (const auto& pair : strategies_) {
        names.push_back(pair.first);
    }
    return names;
}

} // namespace sentio
```

## 📄 **FILE 11 of 15**: temp_ppo_mega/src/feature_pipeline.cpp

**File Information**:
- **Path**: `temp_ppo_mega/src/feature_pipeline.cpp`

- **Size**: 788 lines
- **Modified**: 2025-09-20 02:49:51

- **Type**: .cpp

```text
#include "sentio/feature_pipeline.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

namespace sentio {

FeaturePipeline::FeaturePipeline(const TransformerConfig::Features& config)
    : config_(config) {
    // Note: std::deque doesn't have reserve(), but that's okay
    feature_stats_.resize(TOTAL_FEATURES, RunningStats(config.decay_factor));
}

TransformerFeatureMatrix FeaturePipeline::generate_features(const std::vector<Bar>& bars) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (bars.empty()) {
        return torch::zeros({1, TOTAL_FEATURES});
    }
    
    std::vector<float> all_features;
    all_features.reserve(TOTAL_FEATURES);
    
    // Generate different feature categories
    auto price_features = generate_price_features(bars);
    auto volume_features = generate_volume_features(bars);
    auto technical_features = generate_technical_features(bars);
    auto temporal_features = generate_temporal_features(bars.back());
    
    // Combine all features
    all_features.insert(all_features.end(), price_features.begin(), price_features.end());
    all_features.insert(all_features.end(), volume_features.begin(), volume_features.end());
    all_features.insert(all_features.end(), technical_features.begin(), technical_features.end());
    all_features.insert(all_features.end(), temporal_features.begin(), temporal_features.end());
    
    // Pad or truncate to exact feature count
    if (all_features.size() < TOTAL_FEATURES) {
        all_features.resize(TOTAL_FEATURES, 0.0f);
    } else if (all_features.size() > TOTAL_FEATURES) {
        all_features.resize(TOTAL_FEATURES);
    }
    
    // Update statistics and normalize
    update_feature_statistics(all_features);
    auto normalized_features = normalize_features(all_features);
    
    // Convert to tensor
    auto feature_tensor = torch::tensor(normalized_features).unsqueeze(0); // Add batch dimension
    
    // Check latency requirement
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (duration.count() > MAX_GENERATION_TIME_US) {
        std::cerr << "Feature generation exceeded latency requirement: " 
                  << duration.count() << "us > " << MAX_GENERATION_TIME_US << "us" << std::endl;
    }
    
    return feature_tensor;
}

TransformerFeatureMatrix FeaturePipeline::generate_enhanced_features(const std::vector<Bar>& bars) {
    auto start_time = std::chrono::high_resolution_clock::now();
    if (bars.empty()) {
        return torch::zeros({1, ENHANCED_TOTAL_FEATURES});
    }
    // Original 128
    auto base_tensor = generate_features(bars);
    auto base_vec = std::vector<float>(base_tensor.data_ptr<float>(), base_tensor.data_ptr<float>() + TOTAL_FEATURES);

    // Additional feature blocks
    auto momentum = generate_momentum_persistence_features(bars);
    auto volreg   = generate_volatility_regime_features(bars);
    auto micro    = generate_microstructure_features(bars);
    auto options  = generate_options_features(bars);

    std::vector<float> all_features;
    all_features.reserve(ENHANCED_TOTAL_FEATURES);
    all_features.insert(all_features.end(), base_vec.begin(), base_vec.end());
    all_features.insert(all_features.end(), momentum.begin(), momentum.end());
    all_features.insert(all_features.end(), volreg.begin(), volreg.end());
    all_features.insert(all_features.end(), micro.begin(), micro.end());
    all_features.insert(all_features.end(), options.begin(), options.end());

    if (all_features.size() < ENHANCED_TOTAL_FEATURES) {
        all_features.resize(ENHANCED_TOTAL_FEATURES, 0.0f);
    } else if (all_features.size() > ENHANCED_TOTAL_FEATURES) {
        all_features.resize(ENHANCED_TOTAL_FEATURES);
    }

    // For enhanced features we reuse the normalization path but stats are sized to TOTAL_FEATURES.
    // We therefore normalize only the first TOTAL_FEATURES with running stats; rest are left as-is.
    for (size_t i = 0; i < std::min(all_features.size(), feature_stats_.size()); ++i) {
        if (!std::isnan(all_features[i]) && !std::isinf(all_features[i])) {
            feature_stats_[i].update(all_features[i]);
            float z = (all_features[i] - feature_stats_[i].mean()) / (feature_stats_[i].std() + 1e-8f);
            all_features[i] = z;
        } else {
            all_features[i] = 0.0f;
        }
    }

    auto tensor = torch::tensor(all_features).unsqueeze(0);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    if (duration.count() > MAX_GENERATION_TIME_US) {
        std::cerr << "Enhanced feature generation exceeded latency: "
                  << duration.count() << "us > " << MAX_GENERATION_TIME_US << "us" << std::endl;
    }
    return tensor;
}

void FeaturePipeline::update_feature_cache(const Bar& new_bar) {
    if (feature_cache_.size() >= FEATURE_CACHE_SIZE) {
        feature_cache_.pop_front();
    }
    feature_cache_.push_back(new_bar);
}

std::vector<Bar> FeaturePipeline::get_cached_bars(int lookback_periods) const {
    std::vector<Bar> bars;
    
    if (lookback_periods <= 0 || feature_cache_.empty()) {
        return bars;
    }
    
    int start_idx = std::max(0, static_cast<int>(feature_cache_.size()) - lookback_periods);
    bars.reserve(feature_cache_.size() - start_idx);
    
    for (size_t i = start_idx; i < feature_cache_.size(); ++i) {
        bars.push_back(feature_cache_[i]);
    }
    
    return bars;
}

std::vector<float> FeaturePipeline::generate_price_features(const std::vector<Bar>& bars) {
    std::vector<float> features;
    features.reserve(40); // Target 40 price features
    
    if (bars.empty()) {
        features.resize(40, 0.0f);
        return features;
    }
    
    const Bar& current = bars.back();
    
    // Basic OHLC normalized features
    features.push_back(current.open / current.close - 1.0f);
    features.push_back(current.high / current.close - 1.0f);
    features.push_back(current.low / current.close - 1.0f);
    features.push_back(0.0f); // close normalized to itself
    
    // Returns
    if (bars.size() >= 2) {
        features.push_back((current.close / bars[bars.size()-2].close) - 1.0f); // 1-period return
    } else {
        features.push_back(0.0f);
    }
    
    // Multi-period returns
    std::vector<int> periods = {5, 10, 20};
    for (int period : periods) {
        if (bars.size() > period) {
            float ret = (current.close / bars[bars.size()-1-period].close) - 1.0f;
            features.push_back(ret);
        } else {
            features.push_back(0.0f);
        }
    }
    
    // Log returns
    if (bars.size() >= 2) {
        features.push_back(std::log(current.close / bars[bars.size()-2].close));
    } else {
        features.push_back(0.0f);
    }
    
    // Extract close prices for technical indicators
    std::vector<float> closes;
    for (const auto& bar : bars) {
        closes.push_back(bar.close);
    }
    
    // Moving averages (SMA)
    std::vector<int> ma_periods = {5, 10, 20, 50};
    for (int period : ma_periods) {
        auto sma_values = calculate_sma(closes, period);
        if (!sma_values.empty()) {
            features.push_back((current.close / sma_values.back()) - 1.0f);
        } else {
            features.push_back(0.0f);
        }
    }
    
    // Moving averages (EMA)
    for (int period : ma_periods) {
        auto ema_values = calculate_ema(closes, period);
        if (!ema_values.empty()) {
            features.push_back((current.close / ema_values.back()) - 1.0f);
        } else {
            features.push_back(0.0f);
        }
    }
    
    // Volatility features
    if (bars.size() >= 10) {
        float sum_sq = 0.0f;
        float mean_return = 0.0f;
        
        // Calculate returns for volatility
        std::vector<float> returns;
        for (size_t i = bars.size() - 10; i < bars.size() - 1; ++i) {
            float ret = std::log(bars[i+1].close / bars[i].close);
            returns.push_back(ret);
            mean_return += ret;
        }
        mean_return /= returns.size();
        
        // Calculate standard deviation
        for (float ret : returns) {
            sum_sq += (ret - mean_return) * (ret - mean_return);
        }
        float vol = std::sqrt(sum_sq / (returns.size() - 1)) * std::sqrt(252); // Annualized
        features.push_back(vol);
    } else {
        features.push_back(0.0f);
    }
    
    // High-Low ratio
    features.push_back((current.high - current.low) / current.close);
    
    // Momentum features
    if (bars.size() >= 5) {
        float mom_5 = current.close - bars[bars.size()-6].close;
        features.push_back(mom_5 / current.close);
    } else {
        features.push_back(0.0f);
    }
    
    // Pad to exactly 40 features
    while (features.size() < 40) {
        features.push_back(0.0f);
    }
    features.resize(40);
    
    return features;
}

std::vector<float> FeaturePipeline::generate_volume_features(const std::vector<Bar>& bars) {
    std::vector<float> features;
    features.reserve(20); // Target 20 volume features
    
    if (bars.empty()) {
        features.resize(20, 0.0f);
        return features;
    }
    
    const Bar& current = bars.back();
    
    // Current volume normalized
    if (bars.size() >= 10) {
        float avg_vol = 0.0f;
        for (size_t i = bars.size() - 10; i < bars.size(); ++i) {
            avg_vol += bars[i].volume;
        }
        avg_vol /= 10.0f;
        features.push_back(current.volume / (avg_vol + 1e-8f));
    } else {
        features.push_back(1.0f);
    }
    
    // Volume ratios to different period averages
    std::vector<int> periods = {5, 10, 20};
    for (int period : periods) {
        if (bars.size() >= period) {
            float avg_vol = 0.0f;
            for (int i = 1; i <= period; ++i) {
                avg_vol += bars[bars.size() - i].volume;
            }
            avg_vol /= period;
            features.push_back(current.volume / (avg_vol + 1e-8f));
        } else {
            features.push_back(1.0f);
        }
    }
    
    // Volume trend
    if (bars.size() >= 10) {
        float recent_avg = 0.0f, older_avg = 0.0f;
        for (int i = 0; i < 5; ++i) {
            recent_avg += bars[bars.size() - 1 - i].volume;
            if (bars.size() > 10) {
                older_avg += bars[bars.size() - 6 - i].volume;
            }
        }
        recent_avg /= 5.0f;
        older_avg /= 5.0f;
        
        if (older_avg > 0) {
            features.push_back((recent_avg / older_avg) - 1.0f);
        } else {
            features.push_back(0.0f);
        }
    } else {
        features.push_back(0.0f);
    }
    
    // Volume volatility
    if (bars.size() >= 10) {
        float vol_mean = 0.0f;
        for (size_t i = bars.size() - 10; i < bars.size(); ++i) {
            vol_mean += bars[i].volume;
        }
        vol_mean /= 10.0f;
        
        float vol_var = 0.0f;
        for (size_t i = bars.size() - 10; i < bars.size(); ++i) {
            vol_var += (bars[i].volume - vol_mean) * (bars[i].volume - vol_mean);
        }
        vol_var /= 9.0f;
        float vol_std = std::sqrt(vol_var);
        features.push_back(vol_std / (vol_mean + 1e-8f));
    } else {
        features.push_back(0.0f);
    }
    
    // Pad to exactly 20 features
    while (features.size() < 20) {
        features.push_back(0.0f);
    }
    features.resize(20);
    
    return features;
}

std::vector<float> FeaturePipeline::generate_technical_features(const std::vector<Bar>& bars) {
    std::vector<float> features;
    features.reserve(40); // Target 40 technical features
    
    if (bars.empty()) {
        features.resize(40, 0.0f);
        return features;
    }
    
    // Extract close prices
    std::vector<float> closes;
    for (const auto& bar : bars) {
        closes.push_back(bar.close);
    }
    
    // RSI features
    auto rsi_14 = calculate_rsi(closes, 14);
    if (!rsi_14.empty()) {
        features.push_back(rsi_14.back() / 100.0f - 0.5f); // Normalize to [-0.5, 0.5]
    } else {
        features.push_back(0.0f);
    }
    
    auto rsi_7 = calculate_rsi(closes, 7);
    if (!rsi_7.empty()) {
        features.push_back(rsi_7.back() / 100.0f - 0.5f);
    } else {
        features.push_back(0.0f);
    }
    
    // Bollinger Bands approximation
    if (closes.size() >= 20) {
        auto sma_20 = calculate_sma(closes, 20);
        if (!sma_20.empty()) {
            // Calculate standard deviation
            float sum_sq = 0.0f;
            for (size_t i = closes.size() - 20; i < closes.size(); ++i) {
                float diff = closes[i] - sma_20.back();
                sum_sq += diff * diff;
            }
            float std_dev = std::sqrt(sum_sq / 20);
            
            float upper = sma_20.back() + 2.0f * std_dev;
            float lower = sma_20.back() - 2.0f * std_dev;
            
            // %B indicator
            float percent_b = (closes.back() - lower) / (upper - lower + 1e-8f);
            features.push_back(percent_b);
            
            // Bandwidth
            float bandwidth = (upper - lower) / sma_20.back();
            features.push_back(bandwidth);
        } else {
            features.push_back(0.5f);
            features.push_back(0.0f);
        }
    } else {
        features.push_back(0.5f);
        features.push_back(0.0f);
    }
    
    // Pad to exactly 40 features
    while (features.size() < 40) {
        features.push_back(0.0f);
    }
    features.resize(40);
    
    return features;
}

std::vector<float> FeaturePipeline::generate_temporal_features(const Bar& current_bar) {
    std::vector<float> features;
    features.reserve(28); // Target 28 temporal features
    
    auto time_t = static_cast<std::time_t>(current_bar.ts_utc_epoch);
    auto tm = *std::localtime(&time_t);
    
    // Hour of day (8 features - one-hot encoding for market hours)
    for (int h = 0; h < 8; ++h) {
        features.push_back((tm.tm_hour >= 9 + h && tm.tm_hour < 10 + h) ? 1.0f : 0.0f);
    }
    
    // Day of week (7 features)
    for (int d = 0; d < 7; ++d) {
        features.push_back((tm.tm_wday == d) ? 1.0f : 0.0f);
    }
    
    // Month (12 features)
    for (int m = 0; m < 12; ++m) {
        features.push_back((tm.tm_mon == m) ? 1.0f : 0.0f);
    }
    
    // Market session (1 feature - RTH vs ETH)
    bool is_rth = (tm.tm_hour >= 9 && tm.tm_hour < 16);
    features.push_back(is_rth ? 1.0f : 0.0f);
    
    // Ensure exactly 28 features
    features.resize(28, 0.0f);
    
    return features;
}

// ===================== Enhanced feature blocks =====================

std::vector<float> FeaturePipeline::generate_momentum_persistence_features(const std::vector<Bar>& bars) {
    std::vector<float> features;
    features.reserve(15);
    if (bars.size() < 50) { features.resize(15, 0.0f); return features; }

    std::vector<int> periods = {5, 10, 20, 50};
    int consistent_momentum = 0;
    for (int period : periods) {
        if (bars.size() > static_cast<size_t>(period)) {
            float momentum = (bars.back().close / bars[bars.size()-1-period].close) - 1.0f;
            features.push_back(momentum);
            if (momentum > 0.02f) consistent_momentum++; else if (momentum < -0.02f) consistent_momentum--;
        } else { features.push_back(0.0f); }
    }
    features.push_back(static_cast<float>(consistent_momentum) / 4.0f);

    // Price acceleration
    if (bars.size() >= 3) {
        float accel = (bars.back().close - bars[bars.size()-2].close) - (bars[bars.size()-2].close - bars[bars.size()-3].close);
        features.push_back(accel / std::max(1e-8f, static_cast<float>(bars.back().close)));
    } else { features.push_back(0.0f); }

    // Trend strength (R^2)
    if (bars.size() >= 20) { features.push_back(calculate_trend_strength(bars, 20)); } else { features.push_back(0.0f); }

    // Volume-momentum divergence
    if (bars.size() >= 10) {
        float price_mom = (bars.back().close / bars[bars.size()-10].close) - 1.0f;
        float vol_tr = calculate_volume_trend(bars, 10);
        features.push_back(price_mom * vol_tr);
    } else { features.push_back(0.0f); }

    // Momentum decay
    std::vector<int> decay_periods = {2, 5, 10};
    for (int period : decay_periods) {
        if (bars.size() > static_cast<size_t>(period*2)) {
            float recent_m = (bars.back().close / bars[bars.size()-1-period].close) - 1.0f;
            float older_m  = (bars[bars.size()-1-period].close / bars[bars.size()-1-period*2].close) - 1.0f;
            features.push_back(recent_m - older_m);
        } else { features.push_back(0.0f); }
    }

    // Resistance breakthrough
    if (bars.size() >= 50) { features.push_back(calculate_resistance_breakthrough(bars)); } else { features.push_back(0.0f); }

    features.resize(15, 0.0f);
    return features;
}

std::vector<float> FeaturePipeline::generate_volatility_regime_features(const std::vector<Bar>& bars) {
    std::vector<float> features; features.reserve(12);
    if (bars.size() < 30) { features.resize(12, 0.0f); return features; }

    std::vector<int> vol_periods = {5, 10, 20, 50};
    for (int p : vol_periods) { features.push_back(calculate_realized_volatility(bars, p)); }

    float current_vol = calculate_realized_volatility(bars, 10);
    float historical_vol = calculate_realized_volatility(bars, 50);
    float vol_regime = current_vol / (historical_vol + 1e-8f);
    features.push_back(vol_regime);

    if (bars.size() >= 20) {
        float up=0.0f, dn=0.0f; int up_c=0, dn_c=0;
        for (size_t i = bars.size()-20; i < bars.size()-1; ++i) {
            float r = std::log(bars[i+1].close / bars[i].close);
            if (r>0){ up += r*r; up_c++; } else { dn += r*r; dn_c++; }
        }
        float up_vol = up_c>0? std::sqrt(up/up_c) : 0.0f;
        float dn_vol = dn_c>0? std::sqrt(dn/dn_c) : 0.0f;
        features.push_back(up_vol);
        features.push_back(dn_vol);
        features.push_back((dn_vol - up_vol) / (dn_vol + up_vol + 1e-8f));
    } else { features.insert(features.end(), {0.0f,0.0f,0.0f}); }

    std::vector<int> persistence_periods = {3,7,15};
    for (int p : persistence_periods) { features.push_back(calculate_volatility_persistence(bars, p)); }
    features.push_back(calculate_volatility_clustering(bars, 20));
    features.resize(12, 0.0f);
    return features;
}

std::vector<float> FeaturePipeline::generate_microstructure_features(const std::vector<Bar>& bars) {
    std::vector<float> features; features.reserve(10);
    if (bars.empty()) { features.resize(10, 0.0f); return features; }

    const Bar& cur = bars.back();
    features.push_back((cur.high - cur.low) / std::max(1e-8f, static_cast<float>(cur.close)));

    if (bars.size() >= 2) {
        float price_change = (cur.close - bars[bars.size()-2].close) / std::max(1e-8f, static_cast<float>(bars[bars.size()-2].close));
        float vol_factor = cur.volume / (calculate_average_volume(bars, 10) + 1e-8f);
        features.push_back(price_change / (vol_factor + 1e-8f));
    } else { features.push_back(0.0f); }

    float pr = cur.high - cur.low; features.push_back(pr>0? cur.volume / pr : 0.0f);
    float pos = (cur.close - cur.low) / (pr + 1e-8f); features.push_back(pos);
    if (bars.size() >= 2) { features.push_back(cur.close > bars[bars.size()-2].close ? 1.0f : -1.0f); } else { features.push_back(0.0f); }
    float high_vol_proxy = (cur.high - cur.close) / (pr + 1e-8f);
    float low_vol_proxy  = (cur.close - cur.low) / (pr + 1e-8f);
    features.push_back(high_vol_proxy); features.push_back(low_vol_proxy);
    if (bars.size() >= 10) { features.push_back(calculate_mean_reversion_speed(bars, 10)); } else { features.push_back(0.0f); }
    if (bars.size() >= 5) {
        float expected = calculate_average_volume(bars, 5);
        float surprise = (cur.volume - expected) / (expected + 1e-8f);
        features.push_back(surprise);
    } else { features.push_back(0.0f); }
    if (bars.size() >= 2 && cur.volume > 0) {
        float lambda_proxy = std::fabs(cur.close - bars[bars.size()-2].close) / (cur.volume + 1e-8f);
        features.push_back(lambda_proxy);
    } else { features.push_back(0.0f); }

    features.resize(10, 0.0f);
    return features;
}

std::vector<float> FeaturePipeline::generate_options_features(const std::vector<Bar>& bars) {
    std::vector<float> features; features.reserve(8);
    if (bars.size() >= 22) {
        float realized_vol = calculate_realized_volatility(bars, 22) * std::sqrt(252.0f);
        float vix_proxy = realized_vol * 100.0f;
        features.push_back(vix_proxy);
        float short_vol = calculate_realized_volatility(bars, 5) * std::sqrt(252.0f) * 100.0f;
        float long_vol  = calculate_realized_volatility(bars, 44) * std::sqrt(252.0f) * 100.0f;
        float term = (long_vol - short_vol) / (short_vol + 1e-8f);
        features.push_back(term);
    } else { features.push_back(20.0f); features.push_back(0.0f); }

    if (bars.size() >= 10) {
        float down_v=0.0f, up_v=0.0f;
        for (size_t i = bars.size()-10; i < bars.size()-1; ++i) {
            if (bars[i+1].close < bars[i].close) down_v += bars[i+1].volume; else up_v += bars[i+1].volume;
        }
        features.push_back(down_v / (up_v + 1e-8f));
    } else { features.push_back(1.0f); }

    if (bars.size() >= 30) {
        std::vector<float> rets; rets.reserve(29);
        for (size_t i = bars.size()-30; i < bars.size()-1; ++i) { rets.push_back(std::log(bars[i+1].close / bars[i].close)); }
        features.push_back(calculate_skewness(rets));
        features.push_back(calculate_kurtosis(rets));
    } else { features.push_back(0.0f); features.push_back(3.0f); }

    if (bars.size() >= 5) { features.push_back(calculate_gamma_exposure_proxy(bars)); } else { features.push_back(0.0f); }
    if (bars.size() >= 20) { features.push_back(detect_unusual_volume_patterns(bars, 20)); } else { features.push_back(0.0f); }
    features.push_back(calculate_fear_greed_proxy(bars));
    features.resize(8, 0.0f);
    return features;
}

// ===================== Enhanced feature utilities =====================

float FeaturePipeline::calculate_realized_volatility(const std::vector<Bar>& bars, int period) {
    if (bars.size() < static_cast<size_t>(period+1)) return 0.0f;
    float sum=0.0f; for (size_t i = bars.size()-period-1; i < bars.size()-1; ++i) {
        float r = std::log(bars[i+1].close / bars[i].close); sum += r*r; }
    return std::sqrt(sum / period);
}

float FeaturePipeline::calculate_average_volume(const std::vector<Bar>& bars, int period) {
    if (bars.size() < static_cast<size_t>(period)) return bars.back().volume;
    double s=0.0; for (size_t i = bars.size()-period; i < bars.size(); ++i) s += bars[i].volume; return static_cast<float>(s/period);
}

float FeaturePipeline::calculate_skewness(const std::vector<float>& v) {
    if (v.size() < 3) return 0.0f;
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double m2=0.0, m3=0.0; for (double x : v){ double d=x-mean; m2+=d*d; m3+=d*d*d; }
    m2/=v.size(); m3/=v.size(); double s = std::sqrt(m2); return s>0 ? static_cast<float>(m3/(s*s*s)) : 0.0f;
}

float FeaturePipeline::calculate_kurtosis(const std::vector<float>& v) {
    if (v.size() < 4) return 3.0f;
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double m2=0.0, m4=0.0; for (double x : v){ double d=x-mean; m2+=d*d; m4+=d*d*d*d; }
    m2/=v.size(); m4/=v.size(); return m2>0 ? static_cast<float>(m4/(m2*m2)) : 3.0f;
}

float FeaturePipeline::calculate_trend_strength(const std::vector<Bar>& bars, int lookback) {
    if (bars.size() < static_cast<size_t>(lookback)) return 0.0f;
    // Simple linear regression R^2 over closes
    int n = lookback; double sumx=0,sumy=0,sumxy=0,sumxx=0;
    for (int i=0;i<n;++i){ double x=i; double y=bars[bars.size()-lookback+i].close; sumx+=x; sumy+=y; sumxy+=x*y; sumxx+=x*x; }
    double xbar=sumx/n, ybar=sumy/n; double ssxy=sumxy - n*xbar*ybar; double ssxx=sumxx - n*xbar*xbar;
    if (ssxx==0) return 0.0f; double beta=ssxy/ssxx; double sst=0, sse=0;
    for (int i=0;i<n;++i){ double x=i; double y=bars[bars.size()-lookback+i].close; double yhat=ybar+beta*(x-xbar); sst+=(y-ybar)*(y-ybar); sse+=(y-yhat)*(y-yhat);} 
    return sst>0? static_cast<float>(1.0 - sse/sst) : 0.0f;
}

float FeaturePipeline::calculate_volume_trend(const std::vector<Bar>& bars, int period) {
    if (bars.size() < static_cast<size_t>(period*2)) return 0.0f;
    double recent=0, older=0; for (int i=0;i<period;++i){ recent += bars[bars.size()-1-i].volume; older += bars[bars.size()-1-period-i].volume; }
    recent/=period; older/=period; return older>0? static_cast<float>(recent/older - 1.0) : 0.0f;
}

float FeaturePipeline::calculate_volatility_persistence(const std::vector<Bar>& bars, int period) {
    if (bars.size() < static_cast<size_t>(period+1)) return 0.0f;
    std::vector<float> r; r.reserve(period);
    for (size_t i = bars.size()-period-1; i < bars.size()-1; ++i) r.push_back(std::fabs(std::log(bars[i+1].close / bars[i].close)));
    if (r.size()<2) return 0.0f; double mean = std::accumulate(r.begin(), r.end(), 0.0) / r.size(); double acf=0.0, denom=0.0;
    for (size_t i=1;i<r.size();++i){ acf += (r[i]-mean)*(r[i-1]-mean); }
    for (size_t i=0;i<r.size();++i){ denom += (r[i]-mean)*(r[i]-mean); }
    return denom>0? static_cast<float>(acf/denom) : 0.0f;
}

float FeaturePipeline::calculate_volatility_clustering(const std::vector<Bar>& bars, int period) {
    if (bars.size() < static_cast<size_t>(period+1)) return 0.0f;
    // Proxy: mean absolute return vs its std over window
    double sum=0.0; std::vector<float> absr; absr.reserve(period);
    for (size_t i = bars.size()-period-1; i < bars.size()-1; ++i){ float ar = std::fabs(std::log(bars[i+1].close / bars[i].close)); absr.push_back(ar); sum+=ar; }
    double mean = sum/absr.size(); double var=0.0; for (float x:absr){ var += (x-mean)*(x-mean);} var/=std::max<size_t>(1,absr.size()-1);
    return mean>0? static_cast<float>(std::sqrt(var)/mean) : 0.0f;
}

float FeaturePipeline::calculate_mean_reversion_speed(const std::vector<Bar>& bars, int period) {
    if (bars.size() < static_cast<size_t>(period+1)) return 0.0f;
    // OU speed proxy using lag-1 autocorrelation of returns
    double sum=0.0; std::vector<float> r; r.reserve(period);
    for (size_t i = bars.size()-period-1; i < bars.size()-1; ++i) r.push_back(std::log(bars[i+1].close / bars[i].close));
    double mean = std::accumulate(r.begin(), r.end(), 0.0) / r.size(); double num=0.0, den=0.0;
    for (size_t i=1;i<r.size();++i){ num += (r[i]-mean)*(r[i-1]-mean); }
    for (size_t i=0;i<r.size();++i){ den += (r[i]-mean)*(r[i]-mean); }
    double rho = den>0? num/den : 0.0; return static_cast<float>(1.0 - rho);
}

float FeaturePipeline::calculate_gamma_exposure_proxy(const std::vector<Bar>& bars) {
    if (bars.size() < 5) return 0.0f; // simple curvature proxy
    // Use 2nd derivative magnitude of price over last 5
    double curv=0.0; for (size_t i = bars.size()-5; i < bars.size()-2; ++i){ double a = bars[i+2].close - 2*bars[i+1].close + bars[i].close; curv += std::fabs(a); }
    return static_cast<float>(curv / 3.0 / std::max(1e-6f, static_cast<float>(bars.back().close)));
}

float FeaturePipeline::detect_unusual_volume_patterns(const std::vector<Bar>& bars, int period) {
    if (bars.size() < static_cast<size_t>(period)) return 0.0f;
    double mean = 0.0; for (size_t i = bars.size()-period; i < bars.size(); ++i) mean += bars[i].volume; mean/=period;
    double var=0.0; for (size_t i = bars.size()-period; i < bars.size(); ++i){ double d=bars[i].volume-mean; var+=d*d; }
    var/=std::max(1, period-1); double stdv=std::sqrt(var);
    return stdv>0? static_cast<float>((bars.back().volume - mean)/(stdv + 1e-8)) : 0.0f;
}

float FeaturePipeline::calculate_fear_greed_proxy(const std::vector<Bar>& bars) {
    if (bars.size() < 10) return 0.5f;
    // Combine normalized return momentum and volume surprise
    float mom = (bars.back().close / bars[bars.size()-10].close) - 1.0f;
    float vol_surprise = detect_unusual_volume_patterns(bars, 10);
    float scaled = 0.5f + std::tanh(5.0f * (mom + 0.1f*vol_surprise)) * 0.5f;
    return std::clamp(scaled, 0.0f, 1.0f);
}

float FeaturePipeline::calculate_resistance_breakthrough(const std::vector<Bar>& bars) {
    if (bars.size() < 50) return 0.0f;
    float max_price = bars[bars.size()-50].close;
    for (size_t i = bars.size()-49; i < bars.size()-1; ++i) max_price = std::max(max_price, static_cast<float>(bars[i].close));
    float diff = static_cast<float>(bars.back().close) - max_price; return diff / std::max(1e-8f, max_price);
}

std::vector<float> FeaturePipeline::calculate_sma(const std::vector<float>& prices, int period) {
    std::vector<float> result;
    if (prices.size() < period) return result;
    
    result.reserve(prices.size());
    
    for (size_t i = period - 1; i < prices.size(); ++i) {
        float sum = std::accumulate(prices.begin() + i - period + 1, prices.begin() + i + 1, 0.0f);
        result.push_back(sum / period);
    }
    
    return result;
}

std::vector<float> FeaturePipeline::calculate_ema(const std::vector<float>& prices, int period) {
    std::vector<float> result;
    if (prices.empty()) return result;
    
    result.reserve(prices.size());
    float multiplier = 2.0f / (period + 1);
    result.push_back(prices[0]);
    
    for (size_t i = 1; i < prices.size(); ++i) {
        float ema_val = (prices[i] - result[i-1]) * multiplier + result[i-1];
        result.push_back(ema_val);
    }
    
    return result;
}

std::vector<float> FeaturePipeline::calculate_rsi(const std::vector<float>& prices, int period) {
    std::vector<float> result;
    if (prices.size() < period + 1) return result;
    
    std::vector<float> gains, losses;
    
    // Calculate price changes
    for (size_t i = 1; i < prices.size(); ++i) {
        float change = prices[i] - prices[i-1];
        gains.push_back(change > 0 ? change : 0);
        losses.push_back(change < 0 ? -change : 0);
    }
    
    // Calculate RSI
    auto avg_gains = calculate_ema(gains, period);
    auto avg_losses = calculate_ema(losses, period);
    
    result.reserve(avg_gains.size());
    for (size_t i = 0; i < avg_gains.size(); ++i) {
        if (avg_losses[i] == 0) {
            result.push_back(100.0f);
        } else {
            float rs = avg_gains[i] / avg_losses[i];
            result.push_back(100.0f - (100.0f / (1.0f + rs)));
        }
    }
    
    return result;
}

void FeaturePipeline::update_feature_statistics(const std::vector<float>& features) {
    for (size_t i = 0; i < features.size() && i < feature_stats_.size(); ++i) {
        if (!std::isnan(features[i]) && !std::isinf(features[i])) {
            feature_stats_[i].update(features[i]);
        }
    }
}

std::vector<float> FeaturePipeline::normalize_features(const std::vector<float>& features) {
    std::vector<float> normalized;
    normalized.reserve(features.size());
    
    for (size_t i = 0; i < features.size(); ++i) {
        if (i < feature_stats_.size() && feature_stats_[i].is_initialized()) {
            float value = features[i];
            if (!std::isnan(value) && !std::isinf(value)) {
                // Z-score normalization
                float normalized_value = (value - feature_stats_[i].mean()) / (feature_stats_[i].std() + 1e-8f);
                normalized.push_back(normalized_value);
            } else {
                normalized.push_back(0.0f);
            }
        } else {
            normalized.push_back(features[i]);
        }
    }
    
    return normalized;
}

} // namespace sentio

```

## 📄 **FILE 12 of 15**: temp_ppo_mega/src/router.cpp

**File Information**:
- **Path**: `temp_ppo_mega/src/router.cpp`

- **Size**: 92 lines
- **Modified**: 2025-09-20 02:49:51

- **Type**: .cpp

```text
#include "sentio/router.hpp"
#include "sentio/audit.hpp" // for PriceBook definition
#include <algorithm>
#include <cassert>
#include <cmath>

namespace sentio {

static inline int dir_from(const StrategySignal& s) {
  using T = StrategySignal::Type;
  if (s.type==T::BUY || s.type==T::STRONG_BUY) return +1;
  if (s.type==T::SELL|| s.type==T::STRONG_SELL) return -1;
  return 0;
}
static inline bool is_strong(const StrategySignal& s) {
  using T = StrategySignal::Type;
  return s.type==T::STRONG_BUY || s.type==T::STRONG_SELL || s.confidence>=0.90;
}
static inline double clamp(double x,double lo,double hi){ return std::max(lo,std::min(hi,x)); }

static inline std::string map_instrument_qqq_family(bool go_long, bool strong,
                                                    const RouterCfg& cfg,
                                                    const std::string& base_symbol)
{
  if (base_symbol == cfg.base_symbol) {
    if (go_long)   return strong ? cfg.bull3x : cfg.base_symbol;
    else           return strong ? cfg.bear3x : cfg.base_symbol; // SHORT base for moderate sell
  }
  // Unknown family: fall back to base
  return base_symbol;
}

std::optional<RouteDecision> route(const StrategySignal& s, const RouterCfg& cfg, const std::string& base_symbol) {
  int d = dir_from(s);
  if (d==0) return std::nullopt;
  if (s.confidence < cfg.min_signal_strength) return std::nullopt;

  const bool strong = is_strong(s);
  const bool go_long = (d>0);
  const std::string instrument = map_instrument_qqq_family(go_long, strong, cfg, base_symbol);

  const double raw = (d>0?+1.0:-1.0) * (s.confidence * cfg.signal_multiplier);
  const double tw  = clamp(raw, -cfg.max_position_pct, +cfg.max_position_pct);

  return RouteDecision{instrument, tw};
}

// Implemented elsewhere in your codebase; declared in router.hpp.
// Here's a weak reference for clarity:
// double last_trade_price(const PriceBook&, const std::string&);

static inline double round_to_lot(double qty, double lot) {
  if (lot <= 0) return std::floor(qty);
  return std::floor(qty / lot) * lot;
}

Order route_and_create_order(const std::string& signal_id,
                             const StrategySignal& sig,
                             const RouterCfg& cfg,
                             const std::string& base_symbol,
                             const PriceBook& book,
                             const AccountSnapshot& acct,
                             std::int64_t ts_utc)
{
  Order o{};
  o.signal_id = signal_id;
  auto rd = route(sig, cfg, base_symbol);
  if (!rd) return o; // qty 0

  o.instrument = rd->instrument;
  o.side = (rd->target_weight >= 0 ? OrderSide::Buy : OrderSide::Sell);

  // Size by equity * |target_weight|
  double px = last_trade_price(book, o.instrument); // must be routed instrument
  if (!(std::isfinite(px) && px > 0.0)) return o;

  double target_notional = std::abs(rd->target_weight) * acct.equity;
  double raw_qty = target_notional / px;
  double lot = (cfg.lot_size>0 ? cfg.lot_size : 1.0);
  double qty = round_to_lot(raw_qty, lot);
  if (qty < cfg.min_shares) return o;

  o.qty = qty;
  o.limit_price = 0.0; // market
  o.ts_utc = ts_utc;
  o.notional = (o.side==OrderSide::Buy ? +1.0 : -1.0) * px * qty;

  assert(!o.instrument.empty() && "Instrument must be set");
  return o;
}

} // namespace sentio
```

## 📄 **FILE 13 of 15**: temp_ppo_mega/src/runner.cpp

**File Information**:
- **Path**: `temp_ppo_mega/src/runner.cpp`

- **Size**: 1181 lines
- **Modified**: 2025-09-20 02:49:51

- **Type**: .cpp

```text
#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/pricebook.hpp"
#include "sentio/metrics.hpp"
#include "audit/audit_db_recorder.hpp"
#include "sentio/safe_sizer.hpp"
#include "sentio/audit.hpp"
// Strategy-Agnostic Backend Components
#include "sentio/strategy_profiler.hpp"
#include "sentio/adaptive_allocation_manager.hpp"
#include "sentio/universal_position_coordinator.hpp"
#include "sentio/adaptive_eod_manager.hpp"
#include "sentio/feature_feeder.hpp"
#include "sentio/canonical_evaluation.hpp"
// Golden Rule Enforcement Components
#include "sentio/execution_verifier.hpp"
#include "sentio/circuit_breaker.hpp"
#include <iostream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <sqlite3.h>

namespace sentio {

// **GOLDEN RULE ENFORCEMENT PIPELINE**
// Strict phase ordering with enforcement mechanisms that cannot be bypassed
static void execute_bar_pipeline(
    double strategy_probability,
    Portfolio& portfolio, 
    const SymbolTable& ST, 
    const Pricebook& pricebook,
    StrategyProfiler& profiler,
    AdaptiveAllocationManager& allocation_mgr,
    UniversalPositionCoordinator& position_coord, 
    AdaptiveEODManager& eod_mgr,
    const std::vector<std::vector<Bar>>& series, 
    const Bar& bar,
    const std::string& chain_id,
    IAuditRecorder& audit,
    bool logging_enabled,
    int& total_fills,
    const std::string& strategy_name,
    size_t bar_index) {
    
    // ENFORCEMENT COMPONENTS - Cannot be bypassed
    static ExecutionVerifier verifier;
    static CircuitBreaker breaker;
    
    // PHASE 0: Reset and prepare for new bar
    verifier.reset_bar(bar.ts_utc_epoch);
    position_coord.reset_bar(bar.ts_utc_epoch);
    
    auto profile = profiler.get_current_profile();
    
    // PHASE 1: SKIP EOD CHECK (EOD requirement removed per user request)
    verifier.mark_eod_checked(bar.ts_utc_epoch);
    
    // PHASE 2: CIRCUIT BREAKER CHECK (Emergency protection)
    if (!breaker.check_portfolio_integrity(portfolio, ST, bar.ts_utc_epoch)) {
        if (breaker.is_tripped()) {
            
            auto emergency_orders = breaker.get_emergency_closure(portfolio, ST);
            if (!emergency_orders.empty() && verifier.verify_can_execute(bar.ts_utc_epoch, emergency_orders[0].instrument)) {
                const auto& decision = emergency_orders[0];
                
                // Execute emergency close through proper pipeline
                int instrument_id = ST.get_id(decision.instrument);
                if (instrument_id != -1) {
                    double current_qty = portfolio.positions[instrument_id].qty;
                    if (std::abs(current_qty) > 1e-6) {
                        // Calculate trade quantity to close position
                        double trade_qty = -current_qty; // Opposite sign to close
                        double instrument_price = pricebook.last_px[instrument_id];
                        
                        // Apply the trade through proper pipeline
                        apply_fill(portfolio, instrument_id, trade_qty, instrument_price);
                        
                        // Calculate P&L and audit properly
                        double realized_delta = 0.0; // Emergency closure, no realized P&L calculation
                        double equity_after = equity_mark_to_market(portfolio, pricebook.last_px);
                        double pos_after = portfolio.positions[instrument_id].qty;
                        double fees = 0.0; // No fees for emergency closure
                        Side side = (trade_qty > 0) ? Side::Buy : Side::Sell;
                        
                        // Record in audit trail
                        if (logging_enabled) {
                            audit.event_fill_ex(bar.ts_utc_epoch, decision.instrument, 
                                              instrument_price, trade_qty, fees, side,
                                              realized_delta, equity_after, pos_after, chain_id);
                        }
                        
                        total_fills++;
                        verifier.mark_trade_executed(bar.ts_utc_epoch, decision.instrument, true); // is_closing_trade = true
                    }
                }
            }
            
            return; // GOLDEN RULE: No strategy consultation during emergency
        }
    }
    
    // PHASE 3: STRATEGY CONSULTATION (Only if phases 1 & 2 allow)
    if (!verifier.verify_can_execute(bar.ts_utc_epoch)) {
        return;
    }
    
    // Profile the strategy signal
    profiler.observe_signal(strategy_probability, bar.ts_utc_epoch);
    
    // Get adaptive allocations based on profile
    auto allocations = allocation_mgr.get_allocations(strategy_probability, profile);
    
    // Log signal activity
    if (logging_enabled && !allocations.empty()) {
        audit.event_signal_ex(bar.ts_utc_epoch, ST.get_symbol(0), 
                            SigType::BUY, strategy_probability, chain_id);
    }
    
    // Position coordination with enforcement
    verifier.mark_position_coordinated(bar.ts_utc_epoch);
    auto coordination_decisions = position_coord.coordinate(
        allocations, portfolio, ST, bar.ts_utc_epoch, profile
    );
    
    // PHASE 4: EXECUTE APPROVED DECISIONS (With strict enforcement)
    
    for (const auto& coord_decision : coordination_decisions) {
        if (coord_decision.result != CoordinationResult::APPROVED) {
            if (logging_enabled) {
                audit.event_signal_drop(bar.ts_utc_epoch, strategy_name, 
                                      coord_decision.decision.instrument,
                                      DropReason::THRESHOLD, chain_id, 
                                      coord_decision.reason);
            }
            continue;
        }
        
        const auto& decision = coord_decision.decision;
        
        // ENFORCEMENT: Verify execution is allowed
        bool is_closing_trade = (decision.target_weight == 0.0);
        if (!verifier.verify_can_execute(bar.ts_utc_epoch, decision.instrument, is_closing_trade)) {
            continue;
        }
        
        // Use SafeSizer for execution
        SafeSizer sizer;
        double target_qty = sizer.calculate_target_quantity(
            portfolio, ST, pricebook.last_px, 
            decision.instrument, decision.target_weight, 
            bar.ts_utc_epoch, series[ST.get_id(decision.instrument)]
        );
        
        int instrument_id = ST.get_id(decision.instrument);
        if (instrument_id == -1) continue;
        
        double current_qty = portfolio.positions[instrument_id].qty;
        double trade_qty = target_qty - current_qty;
        
        if (std::abs(trade_qty) < 1e-9) continue;
        
        double instrument_price = pricebook.last_px[instrument_id];
        if (instrument_price <= 0) continue;
        
        // Execute trade
        Side side = (trade_qty > 0) ? Side::Buy : Side::Sell;
        
        if (logging_enabled) {
            audit.event_order_ex(bar.ts_utc_epoch, decision.instrument, side, 
                               std::abs(trade_qty), 0.0, chain_id);
        }
        
        // Calculate P&L
        double realized_delta = 0.0;
        const auto& pos_before = portfolio.positions[instrument_id];
        double closing = 0.0;
        if (pos_before.qty > 0 && trade_qty < 0) {
            closing = std::min(std::abs(trade_qty), pos_before.qty);
        } else if (pos_before.qty < 0 && trade_qty > 0) {
            closing = std::min(std::abs(trade_qty), std::abs(pos_before.qty));
        }
        if (closing > 0.0) {
            if (pos_before.qty > 0) {
                realized_delta = (instrument_price - pos_before.avg_price) * closing;
            } else {
                realized_delta = (pos_before.avg_price - instrument_price) * closing;
            }
        }
        
        // Apply fill
        double fees = AlpacaCostModel::calculate_fees(
            decision.instrument, std::abs(trade_qty), instrument_price, side == Side::Sell
        );
        
        apply_fill(portfolio, instrument_id, trade_qty, instrument_price);
        portfolio.cash -= fees;
        
        // **CRITICAL SAFETY CHECK**: Prevent negative cash balance
        if (portfolio.cash < 0) {
            std::cerr << "ERROR: Negative cash balance detected after trade: " 
                      << portfolio.cash << " (instrument: " << decision.instrument 
                      << ", qty: " << trade_qty << ", price: " << instrument_price << ")" << std::endl;
            
            // Only reverse if this was an opening trade (BUY order that consumed cash)
            // Closing trades (SELL orders) should never cause negative cash
            bool is_opening_trade = (trade_qty > 0); // Positive qty = BUY = opening
            
            if (is_opening_trade) {
                std::cerr << "Reversing opening trade that caused negative cash." << std::endl;
                // Emergency reversal: undo the trade
                apply_fill(portfolio, instrument_id, -trade_qty, instrument_price);
                portfolio.cash += fees;
                std::cerr << "Opening trade reversed. Cash restored to: " << portfolio.cash << std::endl;
                continue; // Skip this trade
            } else {
                std::cerr << "CRITICAL: Closing trade caused negative cash - this indicates a deeper bug!" << std::endl;
                std::cerr << "Position before: " << pos_before.qty << ", Position after: " << portfolio.positions[instrument_id].qty << std::endl;
                // Don't reverse closing trades - they should free up cash, not consume it
                // This indicates a serious bug in position tracking or pricing
            }
        }
        
        double equity_after = equity_mark_to_market(portfolio, pricebook.last_px);
        double pos_after = portfolio.positions[instrument_id].qty;
        
        if (logging_enabled) {
            // **CRITICAL FIX**: Use signed trade_qty, not std::abs(trade_qty)
            // SELL orders need negative quantities for proper position tracking
            audit.event_fill_ex(bar.ts_utc_epoch, decision.instrument, 
                              instrument_price, trade_qty, fees, side,
                              realized_delta, equity_after, pos_after, chain_id);
        }
        
        // Update profiler with trade observation
        profiler.observe_trade(strategy_probability, decision.instrument, bar.ts_utc_epoch);
        total_fills++;
        
        // ENFORCEMENT: Mark trade as executed
        verifier.mark_trade_executed(bar.ts_utc_epoch, decision.instrument, is_closing_trade);
    }
}


// CHANGED: The function now returns a BacktestOutput struct with raw data and accepts dataset metadata.
BacktestOutput run_backtest(IAuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                      int base_symbol_id, const RunnerCfg& cfg, const DatasetMetadata& dataset_meta, 
                      StrategyProfiler* persistent_profiler) {
    
    // 1. ============== INITIALIZATION ==============
    BacktestOutput output{}; // NEW: Initialize the output struct
    
    const bool logging_enabled = (cfg.audit_level == AuditLevel::Full);
    // Calculate actual test period in trading days
    int actual_test_days = 0;
    if (!series[base_symbol_id].empty()) {
        // Estimate trading days from bars (assuming ~390 bars per trading day)
        actual_test_days = std::max(1, static_cast<int>(series[base_symbol_id].size()) / 390);
    }
    
    // Determine dataset type based on data source and characteristics
    std::string dataset_type = "historical"; // default
    if (!series[base_symbol_id].empty()) {
        // Check if this looks like future/AI regime data
        // Future data characteristics: ~26k bars (4 weeks), specific timestamp patterns
        size_t bar_count = series[base_symbol_id].size();
        std::int64_t first_ts = series[base_symbol_id][0].ts_utc_epoch;
        std::int64_t last_ts = series[base_symbol_id].back().ts_utc_epoch;
        double time_span_days = (last_ts - first_ts) / (60.0 * 60.0 * 24.0); // Convert seconds to days
        
        // Future data is typically exactly 4 weeks (28 days) with ~26k bars
        if (bar_count >= 25000 && bar_count <= 27000 && time_span_days >= 27 && time_span_days <= 29) {
            dataset_type = "future_ai_regime";
        }
        // Historical data is typically longer periods or different bar counts
    }
    
    // **DEFERRED**: Calculate actual test period metadata after we know the filtered data range
    // This will be done after warmup calculation when we know the exact bars being processed
    
    auto strategy = StrategyFactory::instance().create_strategy(cfg.strategy_name);
    if (!strategy) {
        std::cerr << "FATAL: Could not create strategy '" << cfg.strategy_name << "'. Check registration." << std::endl;
        return output;
    }
    
    ParameterMap params;
    for (const auto& [key, value] : cfg.strategy_params) {
        try {
            params[key] = std::stod(value);
        } catch (...) { /* ignore */ }
    }
    strategy->set_params(params);

    Portfolio portfolio(ST.size());
    Pricebook pricebook(base_symbol_id, ST, series);
    
    // **STRATEGY-AGNOSTIC EXECUTION PIPELINE COMPONENTS**
    StrategyProfiler local_profiler;
    StrategyProfiler& profiler = persistent_profiler ? *persistent_profiler : local_profiler;
    AdaptiveAllocationManager adaptive_allocation_mgr;
    UniversalPositionCoordinator universal_position_coord;
    AdaptiveEODManager adaptive_eod_mgr;
    
    std::vector<std::pair<std::string, double>> equity_curve;
    std::vector<std::int64_t> equity_curve_ts_ms;
    const auto& base_series = series[base_symbol_id];
    equity_curve.reserve(base_series.size());

    int total_fills = 0;
    int no_route_count = 0;
    int no_qty_count = 0;
    double cumulative_realized_pnl = 0.0;  // Track cumulative realized P&L for audit transparency

    // 2. ============== MAIN EVENT LOOP ==============
    size_t total_bars = base_series.size();
    size_t progress_interval = total_bars / 20; // 5% intervals (20 steps)
    
    // Skip first 300 bars to allow technical indicators to warm up
    size_t warmup_bars = 300;
    if (total_bars <= warmup_bars) {
        std::cout << "Warning: Not enough bars for warmup (need " << warmup_bars << ", have " << total_bars << ")" << std::endl;
        warmup_bars = 0;
    }
    
    // **CANONICAL METADATA**: Calculate actual test period from filtered data (post-warmup)
    std::int64_t run_period_start_ts_ms = 0;
    std::int64_t run_period_end_ts_ms = 0;
    int run_trading_days = 0;
    
    if (warmup_bars < base_series.size()) {
        run_period_start_ts_ms = base_series[warmup_bars].ts_utc_epoch * 1000;
        run_period_end_ts_ms = base_series.back().ts_utc_epoch * 1000;
        
        // Count unique trading days in the filtered range
        std::vector<std::int64_t> filtered_timestamps;
        for (size_t i = warmup_bars; i < base_series.size(); ++i) {
            filtered_timestamps.push_back(base_series[i].ts_utc_epoch * 1000);
        }
        run_trading_days = filtered_timestamps.size() / 390.0; // Approximate: 390 bars per trading day
    }
    
    // Start audit run with canonical metadata including dataset information
    std::string meta = "{";
    meta += "\"strategy\":\"" + cfg.strategy_name + "\",";
    meta += "\"base_symbol_id\":" + std::to_string(base_symbol_id) + ",";
    meta += "\"total_series\":" + std::to_string(series.size()) + ",";
    meta += "\"base_series_size\":" + std::to_string(series[base_symbol_id].size()) + ",";
    meta += "\"dataset_type\":\"" + dataset_type + "\",";
    meta += "\"test_period_days\":" + std::to_string(run_trading_days) + ",";
    meta += "\"run_period_start_ts_ms\":" + std::to_string(run_period_start_ts_ms) + ",";
    meta += "\"run_period_end_ts_ms\":" + std::to_string(run_period_end_ts_ms) + ",";
    meta += "\"run_trading_days\":" + std::to_string(run_trading_days) + ",";
    // **DATASET TRACEABILITY**: Include comprehensive dataset metadata
    meta += "\"dataset_source_type\":\"" + (dataset_meta.source_type.empty() ? dataset_type : dataset_meta.source_type) + "\",";
    meta += "\"dataset_file_path\":\"" + dataset_meta.file_path + "\",";
    meta += "\"dataset_file_hash\":\"" + dataset_meta.file_hash + "\",";
    meta += "\"dataset_track_id\":\"" + dataset_meta.track_id + "\",";
    meta += "\"dataset_regime\":\"" + dataset_meta.regime + "\",";
    meta += "\"dataset_bars_count\":" + std::to_string(dataset_meta.bars_count > 0 ? dataset_meta.bars_count : static_cast<int>(series[base_symbol_id].size())) + ",";
    meta += "\"dataset_time_range_start\":" + std::to_string(dataset_meta.time_range_start > 0 ? dataset_meta.time_range_start : run_period_start_ts_ms) + ",";
    meta += "\"dataset_time_range_end\":" + std::to_string(dataset_meta.time_range_end > 0 ? dataset_meta.time_range_end : run_period_end_ts_ms);
    meta += "}";
    
    // Use current time for run timestamp (for proper run ordering)
    std::int64_t start_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (logging_enabled && !cfg.skip_audit_run_creation) audit.event_run_start(start_ts, meta);
    
    for (size_t i = warmup_bars; i < base_series.size(); ++i) {
        
        const auto& bar = base_series[i];
        
        
        // **RENOVATED**: Governor handles day trading automatically - no manual time logic needed
        pricebook.sync_to_base_i(i);
        
        // Log bar data
        AuditBar audit_bar{bar.open, bar.high, bar.low, bar.close, static_cast<double>(bar.volume)};
        if (logging_enabled) audit.event_bar(bar.ts_utc_epoch, ST.get_symbol(base_symbol_id), audit_bar.open, audit_bar.high, audit_bar.low, audit_bar.close, audit_bar.volume);
        
        // **STRATEGY-AGNOSTIC**: Feed features to any strategy that needs them
        FeatureFeeder::feed_features_to_strategy(strategy.get(), base_series, i, strategy->get_name());
        
        // **RENOVATED ARCHITECTURE**: Governor-based target weight system
        
        // **STRATEGY-AGNOSTIC ARCHITECTURE**: Let strategy control its execution path
        std::string chain_id = std::to_string(bar.ts_utc_epoch) + ":" + std::to_string((long long)i);
        
        // Get strategy probability for logging
        double probability = strategy->calculate_probability(base_series, i);
        std::string base_symbol = ST.get_symbol(base_symbol_id);
        
        // **STRATEGY-AGNOSTIC EXECUTION PIPELINE**: Adaptive components that work for any strategy
        // Execute the Golden Rule pipeline with strategy probability
        execute_bar_pipeline(
            probability, portfolio, ST, pricebook,
            profiler, adaptive_allocation_mgr, universal_position_coord, adaptive_eod_mgr,
            series, bar, chain_id, audit, logging_enabled, total_fills, cfg.strategy_name, i
        );
        
        // Audit logging configured based on system settings
        
        // **STRATEGY-AGNOSTIC**: Log signal for diagnostics
        if (logging_enabled) {
            std::string signal_desc = strategy->get_signal_description(probability);
            
            // **STRATEGY-AGNOSTIC**: Convert signal description to SigType enum
            SigType sig_type = SigType::HOLD;
            std::string upper_desc = signal_desc;
            std::transform(upper_desc.begin(), upper_desc.end(), upper_desc.begin(), ::toupper);
            
            if (upper_desc.find("STRONG") != std::string::npos && upper_desc.find("BUY") != std::string::npos) {
                sig_type = SigType::STRONG_BUY;
            } else if (upper_desc.find("STRONG") != std::string::npos && upper_desc.find("SELL") != std::string::npos) {
                sig_type = SigType::STRONG_SELL;
            } else if (upper_desc.find("BUY") != std::string::npos) {
                sig_type = SigType::BUY;
            } else if (upper_desc.find("SELL") != std::string::npos) {
                sig_type = SigType::SELL;
            }
            // Default remains SigType::HOLD for any other signal descriptions
            
            audit.event_signal_ex(bar.ts_utc_epoch, ST.get_symbol(base_symbol_id), 
                                sig_type, probability, chain_id);
        }
        
        
        // 3. ============== SNAPSHOT ==============
        if (i % cfg.snapshot_stride == 0 || i == base_series.size() - 1) {
            double current_equity = equity_mark_to_market(portfolio, pricebook.last_px);
            
            // Fix: Ensure we have a valid timestamp string for metrics calculation
            std::string timestamp = bar.ts_utc;
            if (timestamp.empty()) {
                // Create synthetic progressive timestamps for metrics calculation
                // Start from a base date and add minutes for each bar
                static time_t base_time = 1726200000; // Sept 13, 2024 (recent date)
                time_t synthetic_time = base_time + (i * 60); // Add 1 minute per bar
                
                auto tm_val = *std::gmtime(&synthetic_time);
                char buffer[32];
                std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_val);
                timestamp = std::string(buffer);
            }
            
            equity_curve.emplace_back(timestamp, current_equity);
            equity_curve_ts_ms.emplace_back(static_cast<std::int64_t>(bar.ts_utc_epoch) * 1000);
            
            // Log account snapshot
            // Calculate actual position value and track cumulative realized P&L  
            // double position_value = current_equity - portfolio.cash; // Unused during cleanup
            
            AccountState state;
            state.cash = portfolio.cash;
            state.equity = current_equity;
            state.realized = cumulative_realized_pnl; // Track actual cumulative realized P&L
            if (logging_enabled) audit.event_snapshot(bar.ts_utc_epoch, state);
        }
    }
    
    // 4. ============== METRICS & DIAGNOSTICS ==============
    strategy->get_diag().print(strategy->get_name().c_str());
    
    // Log signal diagnostics to audit trail
    if (logging_enabled) {
        audit.event_signal_diag(series[base_symbol_id].back().ts_utc_epoch, 
                               cfg.strategy_name, strategy->get_diag());
    }

    if (equity_curve.empty()) {
        return output;
    }
    
    // 3. ============== RAW DATA COLLECTION COMPLETE ==============
    // All metric calculation logic moved to UnifiedMetricsCalculator

    // 4. ============== POPULATE OUTPUT & RETURN ==============
    
    // NEW: Populate the output struct with the raw data from the simulation.
    output.equity_curve = equity_curve;
    output.equity_curve_ts_ms = equity_curve_ts_ms;
    output.total_fills = total_fills;
    output.no_route_events = no_route_count;
    output.no_qty_events = no_qty_count;
    output.run_trading_days = run_trading_days;

    // Audit system reconstructs equity curve for metrics

    // Log strategy profile analysis
    auto final_profile = profiler.get_current_profile();
    std::cout << "\n📊 Strategy Profile Analysis:\n";
    std::cout << "  Trading Style: " << 
        (final_profile.style == TradingStyle::AGGRESSIVE ? "AGGRESSIVE" :
         final_profile.style == TradingStyle::CONSERVATIVE ? "CONSERVATIVE" :
         final_profile.style == TradingStyle::BURST ? "BURST" : "ADAPTIVE") << "\n";
    std::cout << "  Avg Signal Frequency: " << std::fixed << std::setprecision(3) << final_profile.avg_signal_frequency << "\n";
    std::cout << "  Signal Volatility: " << std::fixed << std::setprecision(3) << final_profile.signal_volatility << "\n";
    std::cout << "  Trades per Block: " << std::fixed << std::setprecision(1) << final_profile.trades_per_block << "\n";
    std::cout << "  Adaptive Thresholds: 1x=" << std::fixed << std::setprecision(2) << final_profile.adaptive_entry_1x 
              << ", 3x=" << final_profile.adaptive_entry_3x << "\n";
    std::cout << "  Profile Confidence: " << std::fixed << std::setprecision(1) << (final_profile.confidence_level * 100) << "%\n";

    // **PORTFOLIO STATE VERIFICATION** (for single block runs and 1-block canonical evaluations)
    if (!cfg.skip_audit_run_creation || (cfg.skip_audit_run_creation && total_fills > 0)) { // Show for main runs and meaningful sub-blocks
        const std::string RESET = "\033[0m";
        const std::string BOLD = "\033[1m";
        const std::string DIM = "\033[2m";
        const std::string CYAN = "\033[36m";
        const std::string GREEN = "\033[32m";
        const std::string RED = "\033[31m";
        const std::string WHITE = "\033[37m";
        
        std::cout << "\n" << BOLD << CYAN << "💼 PORTFOLIO STATE VERIFICATION" << RESET << std::endl;
        std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
        
        // Starting state
        std::cout << "│ " << BOLD << "Starting Cash:" << RESET << "     " << GREEN << "$   100,000.00" << RESET << " │ Initial capital allocation                  │" << std::endl;
        std::cout << "│ " << BOLD << "Starting Positions:" << RESET << "           0   │ All instruments start at zero               │" << std::endl;
        
        // Current/Final state
        double final_equity = equity_mark_to_market(portfolio, pricebook.last_px);
        std::string cash_color = (portfolio.cash >= 0) ? GREEN : RED;
        std::cout << "│ " << BOLD << "Final Cash:" << RESET << "        " << cash_color << "$" << std::setw(12) << std::fixed << std::setprecision(2) 
                  << portfolio.cash << RESET << "  │ Remaining liquid capital                    │" << std::endl;
        
        // Count non-zero positions
        int active_positions = 0;
        double total_position_value = 0.0;
        for (size_t i = 0; i < portfolio.positions.size(); ++i) {
            if (std::abs(portfolio.positions[i].qty) > 1e-9) {
                active_positions++;
                total_position_value += portfolio.positions[i].qty * pricebook.last_px[i];
            }
        }
        
        std::cout << "│ " << BOLD << "Final Positions:" << RESET << "    " << WHITE << std::setw(8) << active_positions << RESET << "      │ Number of active instrument positions       │" << std::endl;
        
        std::string pos_value_color = (total_position_value >= 0) ? GREEN : RED;
        std::cout << "│ " << BOLD << "Position Value:" << RESET << "     " << pos_value_color << "$" << std::setw(12) << std::fixed << std::setprecision(2) 
                  << total_position_value << RESET << " │ Market value of all positions               │" << std::endl;
        
        std::string equity_color = (final_equity >= 100000.0) ? GREEN : RED;
        std::cout << "│ " << BOLD << "Final Equity:" << RESET << "       " << equity_color << "$" << std::setw(12) << std::fixed << std::setprecision(2) 
                  << final_equity << RESET << " │ Total account value (cash + positions)      │" << std::endl;
        
        // P&L verification
        double calculated_pnl = final_equity - 100000.0;
        std::string pnl_color = (calculated_pnl >= 0) ? GREEN : RED;
        std::cout << "│ " << BOLD << "Calculated P&L:" << RESET << "     " << pnl_color << "$" << std::setw(12) << std::fixed << std::setprecision(2) 
                  << calculated_pnl << RESET << " │ Final Equity - Starting Capital             │" << std::endl;
        
        std::cout << "├─────────────────────────────────────────────────────────────────────────────────┤" << std::endl;
        
        // Position details (if any active positions)
        if (active_positions > 0) {
            std::cout << "│ " << BOLD << "ACTIVE POSITIONS BREAKDOWN:" << RESET << "                                              │" << std::endl;
            for (size_t i = 0; i < portfolio.positions.size(); ++i) {
                if (std::abs(portfolio.positions[i].qty) > 1e-9) {
                    std::string symbol = ST.get_symbol(i);
                    double market_value = portfolio.positions[i].qty * pricebook.last_px[i];
                    std::string mv_color = (market_value >= 0) ? GREEN : RED;
                    
                    printf("│ %s%-8s%s │ %8.2f shares │ %s$%12.2f%s │ $%8.2f/share │\n",
                           BOLD.c_str(), symbol.c_str(), RESET.c_str(),
                           portfolio.positions[i].qty,
                           mv_color.c_str(), market_value, RESET.c_str(),
                           pricebook.last_px[i]);
                }
            }
        } else {
            std::cout << "│ " << DIM << "No active positions - all positions closed at end of test" << RESET << "                       │" << std::endl;
        }
        
        std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
        
        // Integrity checks
        bool has_negative_cash = (portfolio.cash < 0);
        bool has_massive_positions = false;
        for (size_t i = 0; i < portfolio.positions.size(); ++i) {
            if (std::abs(portfolio.positions[i].qty) > 10000) { // More than 10k shares
                has_massive_positions = true;
                break;
            }
        }
        
        if (!has_negative_cash && !has_massive_positions) {
            std::cout << GREEN << "✅ Portfolio Integrity: No negative cash, reasonable position sizes" << RESET << std::endl;
        } else {
            if (has_negative_cash) {
                std::cout << RED << "❌ Portfolio Warning: Negative cash balance detected!" << RESET << std::endl;
            }
            if (has_massive_positions) {
                std::cout << RED << "❌ Portfolio Warning: Unusually large position sizes detected!" << RESET << std::endl;
            }
        }
    }

    // ============== FINAL POSITION LIQUIDATION (AUDITED) ==============
    // Ensure all final position closures are properly recorded in audit system
    bool has_final_positions = false;
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
            has_final_positions = true;
            break;
        }
    }
    
    if (has_final_positions && logging_enabled) {
        // Use the last bar timestamp for final liquidation
        int64_t final_timestamp = base_series.empty() ? 0 : base_series.back().ts_utc_epoch;
        std::string final_chain_id = std::to_string(final_timestamp) + ":final_liquidation";
        
        // Generate final closing orders for all remaining positions
        for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
            double current_qty = portfolio.positions[sid].qty;
            if (std::abs(current_qty) > 1e-6) {
                const std::string& symbol = ST.get_symbol(sid);
                double trade_qty = -current_qty; // Opposite sign to close
                double instrument_price = pricebook.last_px[sid];
                
                // Apply the trade through proper pipeline
                apply_fill(portfolio, sid, trade_qty, instrument_price);
                
                // Calculate P&L and audit properly
                double realized_delta = 0.0; // Final liquidation, no realized P&L calculation
                double equity_after = equity_mark_to_market(portfolio, pricebook.last_px);
                double pos_after = portfolio.positions[sid].qty;
                double fees = 0.0; // No fees for final liquidation
                Side side = (trade_qty > 0) ? Side::Buy : Side::Sell;
                
                // Record in audit trail
                audit.event_fill_ex(final_timestamp, symbol, 
                                  instrument_price, trade_qty, fees, side,
                                  realized_delta, equity_after, pos_after, final_chain_id);
                
                total_fills++;
            }
        }
    }

    // Log the end of the run to the audit trail
    std::string end_meta = "{}";
    if (logging_enabled) {
        std::int64_t end_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        audit.event_run_end(end_ts, end_meta);
    }

    return output;
}

// ============== CANONICAL EVALUATION SYSTEM ==============

CanonicalReport run_canonical_backtest(
    IAuditRecorder& audit, 
    const SymbolTable& ST, 
    const std::vector<std::vector<Bar>>& series, 
    int base_symbol_id, 
    const RunnerCfg& cfg, 
    const DatasetMetadata& dataset_meta,
    const TradingBlockConfig& block_config) {
    
    // Create the main audit run for the canonical backtest
    std::string meta = "{";
    meta += "\"strategy\":\"" + cfg.strategy_name + "\",";
    meta += "\"base_symbol_id\":" + std::to_string(base_symbol_id) + ",";
    meta += "\"trading_blocks\":" + std::to_string(block_config.num_blocks) + ",";
    meta += "\"block_size\":" + std::to_string(block_config.block_size) + ",";
    meta += "\"dataset_source_type\":\"" + dataset_meta.source_type + "\",";
    meta += "\"dataset_file_path\":\"" + dataset_meta.file_path + "\",";
    meta += "\"dataset_regime\":\"" + dataset_meta.regime + "\",";
    meta += "\"evaluation_type\":\"canonical_trading_blocks\"";
    meta += "}";
    
    std::int64_t start_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    audit.event_run_start(start_ts, meta);
    
    CanonicalReport report;
    report.config = block_config;
    report.strategy_name = cfg.strategy_name;
    report.dataset_source = dataset_meta.source_type;
    
    const auto& base_series = series[base_symbol_id];
    
    // Calculate warmup bars - proportional to new 480-bar Trading Blocks
    // Use ~250 bars warmup (about half a Trading Block) for technical indicators
    size_t warmup_bars = 250;
    if (base_series.size() <= warmup_bars) {
        std::cout << "Warning: Not enough bars for warmup (need " << warmup_bars << ", have " << base_series.size() << ")" << std::endl;
        warmup_bars = 0;
    }
    
    // Calculate total bars needed for the canonical test
    size_t total_bars_needed = warmup_bars + block_config.total_bars();
    if (base_series.size() < total_bars_needed) {
        std::cout << "Warning: Not enough data for complete test (need " << total_bars_needed 
                  << ", have " << base_series.size() << "). Running partial test." << std::endl;
        // Adjust block count to fit available data
        size_t available_test_bars = base_series.size() - warmup_bars;
        int possible_blocks = static_cast<int>(available_test_bars / block_config.block_size);
        if (possible_blocks == 0) {
            std::cerr << "Error: Not enough data for even one block" << std::endl;
            return report;
        }
        // Note: We'll process only the possible blocks
    }
    
    // Calculate test period using most recent data (work backwards from end)
    size_t test_end_idx = base_series.size() - 1;
    size_t test_start_idx = test_end_idx - block_config.total_bars() + 1;
    size_t warmup_start_idx = test_start_idx - warmup_bars;
    
    // Store test period metadata
    report.test_start_ts_ms = base_series[test_start_idx].ts_utc_epoch * 1000;
    report.test_end_ts_ms = base_series[test_end_idx].ts_utc_epoch * 1000;
    
    std::vector<BlockResult> block_results;
    
    // **STRATEGY-AGNOSTIC**: Create persistent profiler across all blocks
    // This allows the profiler to learn the strategy's behavior over time
    StrategyProfiler persistent_profiler;
    
    // Process each block (using most recent data)
    for (int block_index = 0; block_index < block_config.num_blocks; ++block_index) {
        size_t block_start_idx = test_start_idx + (block_index * block_config.block_size);
        size_t block_end_idx = block_start_idx + block_config.block_size;
        
        // Check if we have enough data for this block
        if (block_end_idx > base_series.size()) {
            std::cout << "Insufficient data for block " << block_index << ". Stopping at " 
                      << block_results.size() << " completed blocks." << std::endl;
            break;
        }
        
        std::cout << "Processing Trading Block " << (block_index + 1) << "/" << block_config.num_blocks 
                  << " (bars " << block_start_idx << "-" << (block_end_idx - 1) << ")..." << std::endl;
        
        // Create a data slice for this block (including warmup from the correct position)
        std::vector<std::vector<Bar>> block_series;
        block_series.reserve(series.size());
        
        // Calculate the actual warmup start for this block
        size_t block_warmup_start = (block_start_idx >= warmup_bars) ? block_start_idx - warmup_bars : 0;
        
        for (const auto& symbol_series : series) {
            if (symbol_series.size() > block_end_idx) {
                // Include warmup + this block's data (from warmup start to block end)
                std::vector<Bar> slice(symbol_series.begin() + block_warmup_start, symbol_series.begin() + block_end_idx);
                block_series.push_back(slice);
            } else if (symbol_series.size() > block_start_idx) {
                // Partial data case
                std::vector<Bar> slice(symbol_series.begin() + block_warmup_start, symbol_series.end());
                block_series.push_back(slice);
            } else {
                // Empty series for this symbol in this block
                block_series.emplace_back();
            }
        }
        
        // Create block-specific dataset metadata
        DatasetMetadata block_meta = dataset_meta;
        if (!base_series.empty()) {
            block_meta.time_range_start = base_series[block_start_idx].ts_utc_epoch * 1000;
            block_meta.time_range_end = base_series[block_end_idx - 1].ts_utc_epoch * 1000;
            block_meta.bars_count = block_config.block_size;
        }
        
        // Get starting equity for this block
        double starting_equity = 100000.0; // Default starting capital
        if (!block_results.empty()) {
            starting_equity = block_results.back().ending_equity;
        }
        
        // Create block-specific config that skips audit run creation
        RunnerCfg block_cfg = cfg;
        block_cfg.skip_audit_run_creation = true;  // Skip audit run creation for individual blocks
        // ENSURE audit logging is enabled for instrument distribution
        block_cfg.audit_level = AuditLevel::Full;
        
        // Run backtest for this block with persistent profiler
        BacktestOutput block_output = run_backtest(audit, ST, block_series, base_symbol_id, block_cfg, block_meta, &persistent_profiler);
        
        // Calculate block metrics
        BlockResult block_result = CanonicalEvaluator::calculate_block_metrics(
            block_output.equity_curve,
            block_index,
            starting_equity,
            block_output.total_fills,
            base_series[block_start_idx].ts_utc_epoch * 1000,
            base_series[block_end_idx - 1].ts_utc_epoch * 1000
        );
        
        block_results.push_back(block_result);
        
        // **STRATEGY-AGNOSTIC**: Update profiler with block completion
        persistent_profiler.observe_block_complete(block_output.total_fills);
        
        std::cout << "Block " << (block_index + 1) << " completed: "
                  << "RPB=" << std::fixed << std::setprecision(4) << (block_result.return_per_block * 100) << "%, "
                  << "Sharpe=" << std::fixed << std::setprecision(2) << block_result.sharpe_ratio << ", "
                  << "Fills=" << block_result.fills << std::endl;
    }
    
    if (block_results.empty()) {
        std::cerr << "Error: No blocks were processed successfully" << std::endl;
        return report;
    }
    
    // Aggregate all block results
    report = CanonicalEvaluator::aggregate_block_results(block_config, block_results, cfg.strategy_name, dataset_meta.source_type);
    
    // Store block results in audit database (if it supports it)
    try {
        if (auto* db_recorder = dynamic_cast<audit::AuditDBRecorder*>(&audit)) {
            db_recorder->get_db().store_block_results(db_recorder->get_run_id(), block_results);
        }
    } catch (const std::exception& e) {
        std::cout << "Warning: Could not store block results in audit database: " << e.what() << std::endl;
    }
    
    // Helper function to convert timestamp to ISO format
    auto to_iso_string = [](std::int64_t timestamp_ms) -> std::string {
        std::time_t time_sec = timestamp_ms / 1000;
        std::tm* utc_tm = std::gmtime(&time_sec);
        
        char buffer[32];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", utc_tm);
        return std::string(buffer) + "Z";
    };
    
    // Get run ID from audit recorder
    std::string run_id = "unknown";
    if (auto* db_recorder = dynamic_cast<audit::AuditDBRecorder*>(&audit)) {
        run_id = db_recorder->get_run_id();
    }
    
    // Calculate trades per TB
    double trades_per_tb = 0.0;
    if (report.successful_blocks() > 0) {
        trades_per_tb = static_cast<double>(report.total_fills) / report.successful_blocks();
    }
    
    // Calculate MRB (Monthly Return per Block) - projected monthly return
    // Assuming ~20 Trading Blocks per month (480 bars/block, ~390 bars/day, ~20 trading days/month)
    double blocks_per_month = 20.0;
    double mrb = 0.0;
    if (report.mean_rpb != 0.0) {
        // Use compound interest formula: MRB = ((1 + mean_RPB) ^ 20) - 1
        mrb = (std::pow(1.0 + report.mean_rpb, blocks_per_month) - 1.0) * 100.0;
    }
    
    // Calculate MRP20B (Mean Return per 20TB) if we have enough data - for comparison
    double mrp20b = 0.0;
    if (report.successful_blocks() >= 20) {
        double twenty_tb_return = 1.0;
        for (int i = 0; i < 20 && i < static_cast<int>(report.block_results.size()); ++i) {
            twenty_tb_return *= (1.0 + report.block_results[i].return_per_block);
        }
        mrp20b = (twenty_tb_return - 1.0) * 100.0;
    }
    
    // ANSI color codes for enhanced visual formatting
    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";
    const std::string DIM = "\033[2m";
    
    // Colors
    const std::string BLUE = "\033[34m";
    const std::string GREEN = "\033[32m";
    const std::string RED = "\033[31m";
    const std::string YELLOW = "\033[33m";
    const std::string CYAN = "\033[36m";
    const std::string MAGENTA = "\033[35m";
    const std::string WHITE = "\033[37m";
    
    // Background colors
    const std::string BG_BLUE = "\033[44m";
    const std::string BG_GREEN = "\033[42m";
    const std::string BG_RED = "\033[41m";
    const std::string BG_YELLOW = "\033[43m";
    const std::string BG_CYAN = "\033[46m";
    const std::string BG_DARK = "\033[100m";
    
    // Determine performance color based on Mean RPB
    std::string perf_color = RED;
    std::string perf_bg = "";
    if (report.mean_rpb > 0.001) {  // > 0.1%
        perf_color = GREEN;
        perf_bg = "";
    } else if (report.mean_rpb > -0.001) {  // -0.1% to 0.1%
        perf_color = YELLOW;
        perf_bg = "";
    }
    
    // Header with enhanced styling
    std::cout << "\n" << BOLD << BG_BLUE << WHITE << "╔══════════════════════════════════════════════════════════════════════════════════╗" << RESET << std::endl;
    std::cout << BOLD << BG_BLUE << WHITE << "║                        🎯 CANONICAL EVALUATION COMPLETE                          ║" << RESET << std::endl;
    std::cout << BOLD << BG_BLUE << WHITE << "╚══════════════════════════════════════════════════════════════════════════════════╝" << RESET << std::endl;
    
    // Run Information Section
    std::cout << "\n" << BOLD << CYAN << "📋 RUN INFORMATION" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ " << BOLD << "Run ID:" << RESET << "       " << BLUE << run_id << RESET << std::endl;
    std::cout << "│ " << BOLD << "Strategy:" << RESET << "     " << MAGENTA << cfg.strategy_name << RESET << std::endl;
    std::cout << "│ " << BOLD << "Dataset:" << RESET << "      " << DIM << dataset_meta.file_path << RESET << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    // Time Periods Section
    std::cout << "\n" << BOLD << CYAN << "📅 TIME PERIODS" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    
    // Dataset Period
    if (dataset_meta.time_range_start > 0 && dataset_meta.time_range_end > 0) {
        double dataset_days = (dataset_meta.time_range_end - dataset_meta.time_range_start) / (1000.0 * 60.0 * 60.0 * 24.0);
        std::cout << "│ " << BOLD << "Dataset Period:" << RESET << " " << BLUE << to_iso_string(dataset_meta.time_range_start) 
                  << RESET << " → " << BLUE << to_iso_string(dataset_meta.time_range_end) << RESET << " " 
                  << DIM << "(" << std::fixed << std::setprecision(1) << dataset_days << " days)" << RESET << std::endl;
    }
    
    // Test Period (full available period)
    if (report.test_start_ts_ms > 0 && report.test_end_ts_ms > 0) {
        double test_days = (report.test_end_ts_ms - report.test_start_ts_ms) / (1000.0 * 60.0 * 60.0 * 24.0);
        std::cout << "│ " << BOLD << "Test Period:" << RESET << "    " << GREEN << to_iso_string(report.test_start_ts_ms) 
                  << RESET << " → " << GREEN << to_iso_string(report.test_end_ts_ms) << RESET << " " 
                  << DIM << "(" << std::fixed << std::setprecision(1) << test_days << " days)" << RESET << std::endl;
    }
    
    // TB Period (actual Trading Blocks period)
    if (report.successful_blocks() > 0 && !report.block_results.empty()) {
        uint64_t tb_start_ms = report.block_results[0].start_ts_ms;
        uint64_t tb_end_ms = report.block_results[report.successful_blocks() - 1].end_ts_ms;
        double tb_days = (tb_end_ms - tb_start_ms) / (1000.0 * 60.0 * 60.0 * 24.0);
        std::cout << "│ " << BOLD << "TB Period:" << RESET << "      " << YELLOW << to_iso_string(tb_start_ms) 
                  << RESET << " → " << YELLOW << to_iso_string(tb_end_ms) << RESET << " " 
                  << DIM << "(" << std::fixed << std::setprecision(1) << tb_days << " days, " << report.successful_blocks() << " TBs)" << RESET << std::endl;
    }
    
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    // Trading Configuration Section
    std::cout << "\n" << BOLD << CYAN << "⚙️  TRADING CONFIGURATION" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ " << BOLD << "Trading Blocks:" << RESET << "  " << YELLOW << report.successful_blocks() << RESET << "/" 
              << YELLOW << block_config.num_blocks << RESET << " TB " << DIM << "(480 bars each ≈ 8hrs)" << RESET << std::endl;
    std::cout << "│ " << BOLD << "Total Bars:" << RESET << "     " << WHITE << report.total_bars_processed << RESET << " " 
              << DIM << "(" << std::fixed << std::setprecision(1) << (report.total_bars_processed / 390.0) << " trading days)" << RESET << std::endl;
    std::cout << "│ " << BOLD << "Total Fills:" << RESET << "    " << CYAN << report.total_fills << RESET << std::endl;
    std::cout << "│ " << BOLD << "Trades per TB:" << RESET << "  " << CYAN << std::fixed << std::setprecision(1) << trades_per_tb << RESET << " " << DIM << "(≈Daily)" << RESET << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    // **NEW**: Instrument Distribution with P&L Breakdown for Canonical Evaluation
    std::cout << "\n" << BOLD << CYAN << "🎯 INSTRUMENT DISTRIBUTION & P&L BREAKDOWN" << RESET << std::endl;
    std::cout << "┌────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ Instrument │  Total Volume  │  Net P&L       │  Fill Count    │ Avg Fill Size  │" << std::endl;
    std::cout << "├────────────┼────────────────┼────────────────┼────────────────┼────────────────┤" << std::endl;
    
    // Get instrument statistics from audit database
    std::map<std::string, double> instrument_volume;
    std::map<std::string, double> instrument_pnl;
    std::map<std::string, int> instrument_fills;
    
    // Query the audit database for fill events
    if (auto* db_recorder = dynamic_cast<audit::AuditDBRecorder*>(&audit)) {
        std::string run_id = db_recorder->get_run_id();
        sqlite3* db = db_recorder->get_db().get_db();
        
        std::string query = "SELECT symbol, qty, price, pnl_delta FROM audit_events WHERE run_id = ? AND kind = 'FILL' ORDER BY seq ASC";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
            
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                std::string symbol = (char*)sqlite3_column_text(stmt, 0);
                double qty = sqlite3_column_double(stmt, 1);
                double price = sqlite3_column_double(stmt, 2);
                double pnl_delta = sqlite3_column_double(stmt, 3);
                
                instrument_volume[symbol] += std::abs(qty * price);
                instrument_pnl[symbol] += pnl_delta;
                instrument_fills[symbol]++;
            }
            sqlite3_finalize(stmt);
        }
    }
    
    // **FIX P&L MISMATCH**: Get canonical total P&L from final equity
    double canonical_total_pnl = 0.0;
    double starting_capital = 100000.0; // Standard starting capital
    
    // Extract final equity from the last FILL event's note field (matches canonical evaluation)
    if (auto* db_recorder = dynamic_cast<audit::AuditDBRecorder*>(&audit)) {
        std::string run_id = db_recorder->get_run_id();
        sqlite3* db = db_recorder->get_db().get_db();
        
        std::string query = "SELECT note FROM audit_events WHERE run_id = ? AND kind = 'FILL' ORDER BY seq DESC LIMIT 1";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                std::string note = (char*)sqlite3_column_text(stmt, 0);
                size_t eq_pos = note.find("eq_after=");
                if (eq_pos != std::string::npos) {
                    size_t start = eq_pos + 9; // Length of "eq_after="
                    size_t end = note.find(",", start);
                    if (end == std::string::npos) end = note.length();
                    std::string eq_str = note.substr(start, end - start);
                    try {
                        double final_equity = std::stod(eq_str);
                        canonical_total_pnl = final_equity - starting_capital;
                    } catch (...) {
                        // Fall back to sum of pnl_delta if parsing fails
                        canonical_total_pnl = 0.0;
                        for (const auto& [instrument, pnl] : instrument_pnl) {
                            canonical_total_pnl += pnl;
                        }
                    }
                }
            }
            sqlite3_finalize(stmt);
        }
    }
    
    // **FIX**: Display ALL expected instruments (including those with zero activity)
    double total_volume = 0.0;
    double total_instrument_pnl = 0.0; // Sum of individual instrument P&Ls
    int total_fills = 0;
    
    // **ENSURE ALL QQQ FAMILY INSTRUMENTS ARE SHOWN**
    std::vector<std::string> all_expected_instruments = {"PSQ", "QQQ", "TQQQ", "SQQQ"};
    
    for (const std::string& instrument : all_expected_instruments) {
        double volume = instrument_volume.count(instrument) ? instrument_volume[instrument] : 0.0;
        double pnl = instrument_pnl.count(instrument) ? instrument_pnl[instrument] : 0.0;
        int fills = instrument_fills.count(instrument) ? instrument_fills[instrument] : 0;
        double avg_fill_size = (fills > 0) ? volume / fills : 0.0;
        
        total_volume += volume;
        total_instrument_pnl += pnl;
        total_fills += fills;
        
        // Color coding
        const char* pnl_color = (pnl >= 0) ? GREEN.c_str() : RED.c_str();
        
        printf("│ %-10s │ %14.2f │ %s$%+13.2f%s │ %14d │ $%13.2f │\n",
               instrument.c_str(), volume,
               pnl_color, pnl, RESET.c_str(),
               fills, avg_fill_size);
    }
    
    std::cout << "├────────────┼────────────────┼────────────────┼────────────────┼────────────────┤" << std::endl;
    
    // Totals row - use canonical P&L for accuracy
    const char* canonical_pnl_color = (canonical_total_pnl >= 0) ? GREEN.c_str() : RED.c_str();
    printf("│ %-10s │ %14.2f │ %s$%+13.2f%s │ %14d │ $%13.2f │\n",
           "TOTAL", total_volume,
           canonical_pnl_color, canonical_total_pnl, RESET.c_str(),
           total_fills, (total_fills > 0) ? total_volume / total_fills : 0.0);
    
    // **IMPROVED P&L RECONCILIATION**: Show breakdown of realized vs unrealized P&L
    if (std::abs(total_instrument_pnl - canonical_total_pnl) > 1.0) {
        double unrealized_pnl = canonical_total_pnl - total_instrument_pnl;
        printf("│ %-10s │ %14s │ %s$%+13.2f%s │ %14s │ $%13s │\n",
               "Realized", "",
               (total_instrument_pnl >= 0) ? GREEN.c_str() : RED.c_str(), 
               total_instrument_pnl, RESET.c_str(), "", "");
        printf("│ %-10s │ %14s │ %s$%+13.2f%s │ %14s │ $%13s │\n",
               "Unrealized", "",
               (unrealized_pnl >= 0) ? GREEN.c_str() : RED.c_str(),
               unrealized_pnl, RESET.c_str(), "", "");
    }
    
    std::cout << "└────────────┴────────────────┴────────────────┴────────────────┴────────────────┘" << std::endl;
    
    // **NEW**: Transaction Cost Analysis to explain Mean RPB vs Net P&L relationship
    if (auto* db_recorder = dynamic_cast<audit::AuditDBRecorder*>(&audit)) {
        std::string run_id = db_recorder->get_run_id();
        sqlite3* db = db_recorder->get_db().get_db();
        
        // Calculate total transaction costs from FILL events
        double total_transaction_costs = 0.0;
        int sell_count = 0;
        
        std::string cost_query = "SELECT qty, price, note FROM audit_events WHERE run_id = ? AND kind = 'FILL' ORDER BY seq ASC";
        sqlite3_stmt* cost_stmt;
        if (sqlite3_prepare_v2(db, cost_query.c_str(), -1, &cost_stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(cost_stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
            
            while (sqlite3_step(cost_stmt) == SQLITE_ROW) {
                double qty = sqlite3_column_double(cost_stmt, 0);
                double price = sqlite3_column_double(cost_stmt, 1);
                std::string note = (char*)sqlite3_column_text(cost_stmt, 2);
                
                // Extract fees from note (fees=X.XX format)
                size_t fees_pos = note.find("fees=");
                if (fees_pos != std::string::npos) {
                    size_t start = fees_pos + 5; // Length of "fees="
                    size_t end = note.find(",", start);
                    if (end == std::string::npos) end = note.find(")", start);
                    if (end == std::string::npos) end = note.length();
                    std::string fees_str = note.substr(start, end - start);
                    try {
                        double fees = std::stod(fees_str);
                        total_transaction_costs += fees;
                        if (qty < 0) sell_count++; // Count sell transactions (which have SEC/TAF fees)
                    } catch (...) {
                        // Skip if parsing fails
                    }
                }
            }
            sqlite3_finalize(cost_stmt);
        }
        
        // Display transaction cost breakdown
        std::cout << "\n" << BOLD << CYAN << "💰 TRANSACTION COST ANALYSIS" << RESET << std::endl;
        std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
        printf("│ Total Transaction Costs   │ %s$%11.2f%s │ SEC fees + FINRA TAF (sells only)    │\n", 
               RED.c_str(), total_transaction_costs, RESET.c_str());
        printf("│ Sell Transactions         │ %11d  │ Transactions subject to fees         │\n", sell_count);
        printf("│ Avg Cost per Sell         │ $%11.2f │ Average SEC + TAF cost per sell      │\n", 
               (sell_count > 0) ? total_transaction_costs / sell_count : 0.0);
        printf("│ Cost as %% of Net P&L      │ %10.2f%%  │ Transaction costs vs profit          │\n", 
               (canonical_total_pnl != 0) ? (total_transaction_costs / std::abs(canonical_total_pnl)) * 100.0 : 0.0);
        std::cout << "├─────────────────────────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│ " << BOLD << "Mean RPB includes all transaction costs" << RESET << "  │ Block-by-block returns are net       │" << std::endl;
        std::cout << "│ " << BOLD << "Net P&L is final equity difference" << RESET << "      │ Before/after capital comparison       │" << std::endl; 
        std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    }
    
    // Performance insight
    if (canonical_total_pnl >= 0) {
        std::cout << GREEN << "✅ Net Positive P&L: Strategy generated profit across instruments" << RESET << std::endl;
    } else {
        std::cout << RED << "❌ Net Negative P&L: Strategy lost money across instruments" << RESET << std::endl;
    }
    
    // Performance Metrics Section - with color coding
    std::cout << "\n" << BOLD << CYAN << "📈 PERFORMANCE METRICS" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ " << BOLD << "Mean RPB:" << RESET << "       " << perf_color << BOLD << std::fixed << std::setprecision(4) 
              << (report.mean_rpb * 100) << "%" << RESET << " " << DIM << "(Return Per Block - Net of Fees)" << RESET << std::endl;
    std::cout << "│ " << BOLD << "Std Dev RPB:" << RESET << "    " << WHITE << std::fixed << std::setprecision(4) 
              << (report.stdev_rpb * 100) << "%" << RESET << " " << DIM << "(Volatility)" << RESET << std::endl;
    std::cout << "│ " << BOLD << "MRB:" << RESET << "            " << perf_color << BOLD << std::fixed << std::setprecision(2) 
              << mrb << "%" << RESET << " " << DIM << "(Monthly Return)" << RESET << std::endl;
    std::cout << "│ " << BOLD << "ARB:" << RESET << "            " << perf_color << BOLD << std::fixed << std::setprecision(2) 
              << (report.annualized_return_on_block * 100) << "%" << RESET << " " << DIM << "(Annualized Return)" << RESET << std::endl;
    
    // Risk metrics
    std::string sharpe_color = (report.aggregate_sharpe > 1.0) ? GREEN : (report.aggregate_sharpe > 0) ? YELLOW : RED;
    std::cout << "│ " << BOLD << "Sharpe Ratio:" << RESET << "   " << sharpe_color << std::fixed << std::setprecision(2) 
              << report.aggregate_sharpe << RESET << " " << DIM << "(Risk-Adjusted Return)" << RESET << std::endl;
    
    std::string consistency_color = (report.consistency_score < 1.0) ? GREEN : (report.consistency_score < 2.0) ? YELLOW : RED;
    std::cout << "│ " << BOLD << "Consistency:" << RESET << "    " << consistency_color << std::fixed << std::setprecision(4) 
              << report.consistency_score << RESET << " " << DIM << "(Lower = More Consistent)" << RESET << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    // Performance Summary Box
    std::cout << "\n" << BOLD;
    if (report.mean_rpb > 0.001) {
        std::cout << BG_GREEN << WHITE << "🎉 PROFITABLE STRATEGY ";
    } else if (report.mean_rpb > -0.001) {
        std::cout << BG_YELLOW << WHITE << "⚖️  NEUTRAL STRATEGY ";
    } else {
        std::cout << BG_RED << WHITE << "⚠️  LOSING STRATEGY ";
    }
    std::cout << RESET << std::endl;
    
    // End the main audit run
    std::int64_t end_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    audit.event_run_end(end_ts, "{}");
    
    return report;
}

} // namespace sentio
```

## 📄 **FILE 14 of 15**: temp_ppo_mega/src/strategy_transformer.cpp

**File Information**:
- **Path**: `temp_ppo_mega/src/strategy_transformer.cpp`

- **Size**: 466 lines
- **Modified**: 2025-09-20 02:49:51

- **Type**: .cpp

```text
#include "sentio/strategy_transformer.hpp"
#include "sentio/time_utils.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <filesystem>

namespace sentio {

TransformerStrategy::TransformerStrategy()
    : BaseStrategy("Transformer")
    , cfg_()
{
    // CRITICAL: Ensure configuration has correct dimensions
    cfg_.feature_dim = 128;  // Must match FeaturePipeline::TOTAL_FEATURES
    cfg_.sequence_length = 64;
    cfg_.d_model = 256;
    cfg_.num_heads = 8;
    cfg_.num_layers = 6;
    cfg_.dropout = 0.1;
    cfg_.model_path = "artifacts/Transformer/v1/model.pt";
    cfg_.enable_online_training = true;
    
    std::cout << "🔧 Initializing TransformerStrategy with CORRECTED dimensions:" << std::endl;
    std::cout << "  sequence_length: " << cfg_.sequence_length << std::endl;
    std::cout << "  feature_dim: " << cfg_.feature_dim << std::endl;
    std::cout << "  d_model: " << cfg_.d_model << std::endl;
    
    initialize_model();
}

TransformerStrategy::TransformerStrategy(const TransformerCfg& cfg)
    : BaseStrategy("Transformer")
    , cfg_(cfg)
{
    // CRITICAL: Validate and correct configuration dimensions
    if (cfg_.feature_dim != 128) {
        std::cerr << "⚠️  WARNING: feature_dim was " << cfg_.feature_dim << ", correcting to 128" << std::endl;
        cfg_.feature_dim = 128;
    }
    
    if (!validate_configuration()) {
        std::cerr << "❌ Configuration validation failed! Using safe defaults." << std::endl;
        cfg_.feature_dim = 128;
        cfg_.sequence_length = 64;
    }
    
    std::cout << "🔧 Initializing TransformerStrategy with validated dimensions:" << std::endl;
    std::cout << "  sequence_length: " << cfg_.sequence_length << std::endl;
    std::cout << "  feature_dim: " << cfg_.feature_dim << std::endl;
    std::cout << "  d_model: " << cfg_.d_model << std::endl;
    
    initialize_model();
}

void TransformerStrategy::initialize_model() {
    try {
        // Create transformer configuration
        TransformerConfig model_config;
        model_config.feature_dim = cfg_.feature_dim;
        model_config.sequence_length = cfg_.sequence_length;
        model_config.d_model = cfg_.d_model;
        model_config.num_heads = cfg_.num_heads;
        model_config.num_layers = cfg_.num_layers;
        model_config.ffn_hidden = cfg_.ffn_hidden;
        model_config.dropout = cfg_.dropout;
        
        // Initialize model
        model_ = std::make_shared<TransformerModel>(model_config);
        
        // Try to load pre-trained model if it exists
        if (std::filesystem::exists(cfg_.model_path)) {
            std::cout << "Loading pre-trained transformer model from: " << cfg_.model_path << std::endl;
            model_->load_model(cfg_.model_path);
        } else {
            std::cout << "No pre-trained model found at: " << cfg_.model_path << std::endl;
            std::cout << "Using randomly initialized model" << std::endl;
        }
        
        model_->optimize_for_inference();
        
        // Initialize feature pipeline
        TransformerConfig::Features feature_config;
        feature_config.normalization = TransformerConfig::Features::NormalizationMethod::Z_SCORE;
        feature_config.decay_factor = 0.999f;
        feature_pipeline_ = std::make_unique<FeaturePipeline>(feature_config);
        
        // Initialize online trainer if enabled
        if (cfg_.enable_online_training) {
            OnlineTrainer::OnlineConfig trainer_config;
            trainer_config.update_interval_minutes = cfg_.update_interval_minutes;
            trainer_config.min_samples_for_update = cfg_.min_samples_for_update;
            trainer_config.base_learning_rate = 0.0001f;
            trainer_config.replay_buffer_size = 10000;
            trainer_config.enable_regime_detection = true;
            
            online_trainer_ = std::make_unique<OnlineTrainer>(model_, trainer_config);
            std::cout << "Online training enabled with " << cfg_.update_interval_minutes << " minute intervals" << std::endl;
        }
        
        model_initialized_ = true;
        std::cout << "Transformer strategy initialized successfully" << std::endl;
        std::cout << "Model parameters: " << model_->get_parameter_count() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize transformer strategy: " << e.what() << std::endl;
        model_initialized_ = false;
    }
}

ParameterMap TransformerStrategy::get_default_params() const {
    ParameterMap defaults;
    defaults["buy_threshold"] = cfg_.buy_threshold;
    defaults["sell_threshold"] = cfg_.sell_threshold;
    defaults["strong_threshold"] = cfg_.strong_threshold;
    defaults["conf_floor"] = cfg_.conf_floor;
    return defaults;
}

ParameterSpace TransformerStrategy::get_param_space() const {
    return {
        {"buy_threshold", {ParamType::FLOAT, 0.5, 0.8, 0.6}},
        {"sell_threshold", {ParamType::FLOAT, 0.2, 0.5, 0.4}},
        {"strong_threshold", {ParamType::FLOAT, 0.7, 0.9, 0.8}},
        {"conf_floor", {ParamType::FLOAT, 0.4, 0.6, 0.5}}
    };
}

void TransformerStrategy::apply_params() {
    if (params_.count("buy_threshold")) {
        cfg_.buy_threshold = static_cast<float>(params_["buy_threshold"]);
    }
    if (params_.count("sell_threshold")) {
        cfg_.sell_threshold = static_cast<float>(params_["sell_threshold"]);
    }
    if (params_.count("strong_threshold")) {
        cfg_.strong_threshold = static_cast<float>(params_["strong_threshold"]);
    }
    if (params_.count("conf_floor")) {
        cfg_.conf_floor = static_cast<float>(params_["conf_floor"]);
    }
}


void TransformerStrategy::update_bar_history(const Bar& bar) {
    bar_history_.push_back(bar);
    
    // Keep only the required sequence length + some buffer
    const size_t max_history = cfg_.sequence_length + 50;
    while (bar_history_.size() > max_history) {
        bar_history_.pop_front();
    }
}

std::vector<Bar> TransformerStrategy::convert_to_transformer_bars(const std::vector<Bar>& sentio_bars) const {
    std::vector<Bar> transformer_bars;
    transformer_bars.reserve(sentio_bars.size());
    
    for (const auto& sentio_bar : sentio_bars) {
        // Convert Sentio Bar to Transformer Bar format
        Bar transformer_bar;
        transformer_bar.open = sentio_bar.open;
        transformer_bar.high = sentio_bar.high;
        transformer_bar.low = sentio_bar.low;
        transformer_bar.close = sentio_bar.close;
        transformer_bar.volume = sentio_bar.volume;
        transformer_bar.ts_utc_epoch = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count(); // Use current time
        
        transformer_bars.push_back(transformer_bar);
    }
    
    return transformer_bars;
}

bool TransformerStrategy::validate_tensor_dimensions(const torch::Tensor& tensor, 
                                                   const std::vector<int64_t>& expected_dims,
                                                   const std::string& tensor_name) {
    if (tensor.sizes().size() != expected_dims.size()) {
        std::cerr << "Dimension count mismatch for " << tensor_name 
                  << ": expected " << expected_dims.size() 
                  << " dims, got " << tensor.sizes().size() << std::endl;
        return false;
    }
    
    for (size_t i = 0; i < expected_dims.size(); ++i) {
        if (tensor.size(i) != expected_dims[i]) {
            std::cerr << "Dimension " << i << " mismatch for " << tensor_name 
                      << ": expected " << expected_dims[i] 
                      << ", got " << tensor.size(i) << std::endl;
            return false;
        }
    }
    
    return true;
}

bool TransformerStrategy::validate_configuration() {
    // Check model configuration consistency
    if (cfg_.feature_dim != 128) {
        std::cerr << "WARNING: Feature dim should be 128, got " << cfg_.feature_dim << std::endl;
        return false;
    }
    
    if (cfg_.sequence_length <= 0 || cfg_.sequence_length > 1000) {
        std::cerr << "Invalid sequence length: " << cfg_.sequence_length << std::endl;
        return false;
    }
    
    if (cfg_.d_model % cfg_.num_heads != 0) {
        std::cerr << "d_model must be divisible by num_heads" << std::endl;
        return false;
    }
    
    // Check file paths
    if (!std::filesystem::exists(cfg_.model_path)) {
        std::cout << "Model file not found: " << cfg_.model_path << std::endl;
        std::cout << "Will use randomly initialized model" << std::endl;
    }
    
    return true;
}

TransformerFeatureMatrix TransformerStrategy::generate_features_for_bars(const std::vector<Bar>& bars, int end_index) {
    if (!model_initialized_ || bars.empty() || end_index < 0) {
        // Return properly shaped zero tensor: [1, sequence_length, feature_dim]
        return torch::zeros({1, cfg_.sequence_length, cfg_.feature_dim});
    }
    
    // Get sequence of bars
    std::vector<Bar> sequence_bars;
    int start_idx = std::max(0, end_index - cfg_.sequence_length + 1);
    
    for (int i = start_idx; i <= end_index && i < static_cast<int>(bars.size()); ++i) {
        sequence_bars.push_back(bars[i]);
    }
    
    // Convert to transformer bar format
    auto transformer_bars = convert_to_transformer_bars(sequence_bars);
    
    try {
        // CRITICAL FIX: Generate features for each bar in sequence individually
        std::vector<torch::Tensor> sequence_features;
        sequence_features.reserve(cfg_.sequence_length);
        
        // Generate features for each bar in the sequence
        for (size_t i = 0; i < transformer_bars.size(); ++i) {
            std::vector<Bar> single_bar = {transformer_bars[i]};
            auto bar_features = feature_pipeline_->generate_features(single_bar);
            
            // bar_features is [1, 128] - squeeze to get [128]
            auto squeezed_features = bar_features.squeeze(0);
            
            // Validate feature dimensions
            if (squeezed_features.size(0) != cfg_.feature_dim) {
                std::cerr << "Feature dimension mismatch: expected " << cfg_.feature_dim 
                          << ", got " << squeezed_features.size(0) << std::endl;
                return torch::zeros({1, cfg_.sequence_length, cfg_.feature_dim});
            }
            
            sequence_features.push_back(squeezed_features);
        }
        
        // Pad sequence if we don't have enough bars (pad at beginning with zeros)
        while (sequence_features.size() < cfg_.sequence_length) {
            sequence_features.insert(sequence_features.begin(), 
                                   torch::zeros({cfg_.feature_dim}));
        }
        
        // Take only the last sequence_length features if we have too many
        if (sequence_features.size() > cfg_.sequence_length) {
            sequence_features.erase(sequence_features.begin(), 
                                  sequence_features.end() - cfg_.sequence_length);
        }
        
        // Stack into proper tensor: [sequence_length, feature_dim]
        auto stacked_features = torch::stack(sequence_features, 0);
        
        // Add batch dimension: [1, sequence_length, feature_dim]
        auto batched_features = stacked_features.unsqueeze(0);
        
        // Validate final tensor shape using IntArrayRef comparison
        auto expected_shape = torch::IntArrayRef({1, cfg_.sequence_length, cfg_.feature_dim});
        if (batched_features.sizes() != expected_shape) {
            std::cerr << "Final tensor shape mismatch: expected [1, " 
                      << cfg_.sequence_length << ", " << cfg_.feature_dim 
                      << "], got " << batched_features.sizes() << std::endl;
            return torch::zeros({1, cfg_.sequence_length, cfg_.feature_dim});
        }
        
        return batched_features;
        
    } catch (const std::exception& e) {
        std::cerr << "Error generating features sequence: " << e.what() << std::endl;
        return torch::zeros({1, cfg_.sequence_length, cfg_.feature_dim});
    }
}

double TransformerStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (!model_initialized_ || bars.empty() || current_index < 0) {
        std::cerr << "Model not initialized or invalid input" << std::endl;
        return 0.5f;
    }
    
    try {
        // Generate features with proper dimensions
        auto features = generate_features_for_bars(bars, current_index);
        
        // Validate input tensor dimensions before model inference
        auto expected_shape = torch::IntArrayRef({1, cfg_.sequence_length, cfg_.feature_dim});
        if (features.sizes() != expected_shape) {
            std::cerr << "Feature tensor shape mismatch before inference: expected " 
                      << expected_shape << ", got " << features.sizes() << std::endl;
            return 0.5f;
        }
        
        // Ensure model is in eval mode
        model_->eval();
        torch::NoGradGuard no_grad;
        
        // Debug: Print tensor shapes before inference
        std::cout << "Input tensor shape: " << features.sizes() << std::endl;
        
        // Run inference
        auto prediction_tensor = model_->forward(features);
        
        // Debug: Print output tensor shape
        std::cout << "Output tensor shape: " << prediction_tensor.sizes() << std::endl;
        
        float raw_prediction = prediction_tensor.item<float>();
        
        // Debug: Print raw prediction
        std::cout << "Raw prediction: " << raw_prediction << std::endl;
        
        // Convert to probability using sigmoid (more stable than direct sigmoid)
        float probability = 1.0f / (1.0f + std::exp(-std::clamp(raw_prediction, -10.0f, 10.0f)));
        
        // Ensure probability is in valid range and not exactly neutral
        probability = std::clamp(probability, 0.01f, 0.99f);
        
        // Debug: Print final probability
        std::cout << "Final probability: " << probability << std::endl;
        
        // Log significant predictions (non-neutral)
        if (std::abs(probability - 0.5f) > 0.05f) {
            std::cout << "🎯 Non-neutral signal: " << probability << " (raw: " << raw_prediction << ")" << std::endl;
        }
        
        // Update performance metrics
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            current_metrics_.samples_processed++;
        }
        
        return probability;
        
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR in calculate_probability: " << e.what() << std::endl;
        
        // Log detailed tensor information for debugging
        try {
            auto features = generate_features_for_bars(bars, current_index);
            std::cerr << "Debug - Feature tensor shape: " << features.sizes() << std::endl;
            std::cerr << "Debug - Feature tensor dtype: " << features.dtype() << std::endl;
            std::cerr << "Debug - Model expects: [1, " << cfg_.sequence_length 
                      << ", " << cfg_.feature_dim << "]" << std::endl;
        } catch (...) {
            std::cerr << "Failed to generate debug information" << std::endl;
        }
        
        return 0.5f; // Only return neutral as last resort
    }
}

void TransformerStrategy::maybe_trigger_training() {
    if (!cfg_.enable_online_training || !online_trainer_ || is_training_.load()) {
        return;
    }
    
    try {
        // Check if we should update the model
        if (online_trainer_->should_update_model()) {
            std::cout << "Triggering transformer model update..." << std::endl;
            is_training_ = true;
            
            auto result = online_trainer_->update_model();
            if (result.success) {
                std::cout << "Model update completed successfully" << std::endl;
            } else {
                std::cerr << "Model update failed: " << result.error_message << std::endl;
            }
            
            is_training_ = false;
        }
        
        // Check for regime changes
        if (online_trainer_->detect_regime_change()) {
            std::cout << "Market regime change detected, adapting model..." << std::endl;
            online_trainer_->adapt_to_regime_change();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in maybe_trigger_training: " << e.what() << std::endl;
        is_training_ = false;
    }
}


std::vector<AllocationDecision> TransformerStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol, 
    const std::string& bear3x_symbol
) const {
    
    std::vector<AllocationDecision> decisions;
    
    if (!model_initialized_ || bars.empty() || current_index < 0) {
        return decisions;
    }
    
    // Get probability from strategy
    double probability = const_cast<TransformerStrategy*>(this)->calculate_probability(bars, current_index);
    
    // **PROFIT MAXIMIZATION**: Always deploy 100% of capital with maximum leverage
    if (probability > cfg_.strong_threshold) {
        // Strong buy: 100% TQQQ (3x leveraged long)
        decisions.push_back({bull3x_symbol, 1.0, "Transformer strong buy: 100% " + bull3x_symbol + " (3x leverage)"});
        
    } else if (probability > cfg_.buy_threshold) {
        // Moderate buy: 100% QQQ (1x long)
        decisions.push_back({base_symbol, 1.0, "Transformer moderate buy: 100% " + base_symbol});
        
    } else if (probability < (1.0f - cfg_.strong_threshold)) {
        // Strong sell: 100% SQQQ (3x leveraged short)
        decisions.push_back({bear3x_symbol, 1.0, "Transformer strong sell: 100% " + bear3x_symbol + " (3x inverse)"});
        
    } else if (probability < cfg_.sell_threshold) {
        // Weak sell: 100% PSQ (1x inverse)
        decisions.push_back({"PSQ", 1.0, "Transformer weak sell: 100% PSQ (1x inverse)"});
        
    } else {
        // Neutral: Stay in cash (rare case)
        // No positions needed
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol, "PSQ"};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, "Transformer: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

// Register the strategy
REGISTER_STRATEGY(TransformerStrategy, "transformer");

} // namespace sentio

```

## 📄 **FILE 15 of 15**: temp_ppo_mega/src/transformer_model.cpp

**File Information**:
- **Path**: `temp_ppo_mega/src/transformer_model.cpp`

- **Size**: 112 lines
- **Modified**: 2025-09-20 02:49:51

- **Type**: .cpp

```text
#include "sentio/transformer_model.hpp"
#include <cmath>
#include <iostream>

namespace sentio {

TransformerModel::TransformerModel(const TransformerConfig& config) : config_(config) {
    // Input projection
    input_projection_ = register_module("input_projection", 
        torch::nn::Linear(config.feature_dim, config.d_model));
    
    // Transformer encoder
    torch::nn::TransformerEncoderLayerOptions encoder_layer_options(config.d_model, config.num_heads);
    encoder_layer_options.dim_feedforward(config.d_model * 4);
    encoder_layer_options.dropout(config.dropout);
    // Note: batch_first option may not be available in all PyTorch versions
    // encoder_layer_options.batch_first(true);
    
    auto encoder_layer = torch::nn::TransformerEncoderLayer(encoder_layer_options);
    
    torch::nn::TransformerEncoderOptions encoder_options(encoder_layer, config.num_layers);
    transformer_ = register_module("transformer", torch::nn::TransformerEncoder(encoder_options));
    
    // Layer normalization
    layer_norm_ = register_module("layer_norm", torch::nn::LayerNorm(std::vector<int64_t>{config.d_model}));
    
    // Output projection
    output_projection_ = register_module("output_projection", 
        torch::nn::Linear(config.d_model, 1));
    
    // Dropout
    dropout_ = register_module("dropout", torch::nn::Dropout(config.dropout));
    
    // Create positional encoding
    create_positional_encoding();
}

void TransformerModel::create_positional_encoding() {
    pos_encoding_ = torch::zeros({config_.sequence_length, config_.d_model});
    
    auto position = torch::arange(0, config_.sequence_length).unsqueeze(1).to(torch::kFloat);
    auto div_term = torch::exp(torch::arange(0, config_.d_model, 2).to(torch::kFloat) * 
                              -(std::log(10000.0) / config_.d_model));
    
    pos_encoding_.slice(1, 0, config_.d_model, 2) = torch::sin(position * div_term);
    pos_encoding_.slice(1, 1, config_.d_model, 2) = torch::cos(position * div_term);
    
    pos_encoding_ = pos_encoding_.unsqueeze(0); // Add batch dimension
}

torch::Tensor TransformerModel::forward(const torch::Tensor& input) {
    // input shape: [batch_size, sequence_length, feature_dim]
    // auto batch_size = input.size(0);  // Unused for now
    auto seq_len = input.size(1);
    
    // Input projection
    auto x = input_projection_->forward(input);
    
    // Add positional encoding
    auto pos_enc = pos_encoding_.slice(1, 0, seq_len).to(x.device());
    x = x + pos_enc;
    x = dropout_->forward(x);
    
    // Transformer encoding
    x = transformer_->forward(x);
    
    // Layer normalization
    x = layer_norm_->forward(x);
    
    // Global average pooling
    x = torch::mean(x, 1); // [batch_size, d_model]
    
    // Output projection
    auto output = output_projection_->forward(x); // [batch_size, 1]
    
    return output;
}

void TransformerModel::save_model(const std::string& path) {
    torch::serialize::OutputArchive archive;
    this->save(archive);
    archive.save_to(path);
}

void TransformerModel::load_model(const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    this->load(archive);
}

void TransformerModel::optimize_for_inference() {
    eval();
    // Additional optimizations could be added here
}

size_t TransformerModel::get_parameter_count() const {
    size_t count = 0;
    for (const auto& param : parameters()) {
        count += param.numel();
    }
    return count;
}

size_t TransformerModel::get_memory_usage_bytes() const {
    size_t bytes = 0;
    for (const auto& param : parameters()) {
        bytes += param.nbytes();
    }
    return bytes;
}

} // namespace sentio

```

