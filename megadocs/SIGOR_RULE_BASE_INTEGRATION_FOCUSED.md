# SIGOR Rule-Based Detectors Integration

**Generated**: 2025-09-18 10:20:43
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: Requirement + architecture + selected source: sigor, signal utils, and prior rule-based algorithms (RSI, Bollinger, Momentum-Volume, OFI, ORB, VWAP, SMA Cross).

**Total Files**: 15

---

## 📋 **TABLE OF CONTENTS**

1. [megadocs/mega_inputs/sigor_rules/SIGOR_RULE_BASE_INTEGRATION_REQUIREMENTS.md](#file-1)
2. [megadocs/mega_inputs/sigor_rules/architecture.md](#file-2)
3. [megadocs/mega_inputs/sigor_rules/momentum_volume_rule.hpp](#file-3)
4. [megadocs/mega_inputs/sigor_rules/ofi_proxy_rule.hpp](#file-4)
5. [megadocs/mega_inputs/sigor_rules/opening_range_breakout_rule.hpp](#file-5)
6. [megadocs/mega_inputs/sigor_rules/rsi_strategy.cpp](#file-6)
7. [megadocs/mega_inputs/sigor_rules/rsi_strategy.hpp](#file-7)
8. [megadocs/mega_inputs/sigor_rules/signal_utils.hpp](#file-8)
9. [megadocs/mega_inputs/sigor_rules/sma_cross_rule.hpp](#file-9)
10. [megadocs/mega_inputs/sigor_rules/strategy_bollinger_squeeze_breakout.cpp](#file-10)
11. [megadocs/mega_inputs/sigor_rules/strategy_bollinger_squeeze_breakout.hpp](#file-11)
12. [megadocs/mega_inputs/sigor_rules/strategy_signal_or.cpp](#file-12)
13. [megadocs/mega_inputs/sigor_rules/strategy_signal_or.hpp](#file-13)
14. [megadocs/mega_inputs/sigor_rules/strategy_utils.hpp](#file-14)
15. [megadocs/mega_inputs/sigor_rules/vwap_reversion_rule.hpp](#file-15)

---

## 📄 **FILE 1 of 15**: megadocs/mega_inputs/sigor_rules/SIGOR_RULE_BASE_INTEGRATION_REQUIREMENTS.md

**File Information**:
- **Path**: `megadocs/mega_inputs/sigor_rules/SIGOR_RULE_BASE_INTEGRATION_REQUIREMENTS.md`

- **Size**: 125 lines
- **Modified**: 2025-09-18 10:20:37

- **Type**: .md

```text
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



```

## 📄 **FILE 2 of 15**: megadocs/mega_inputs/sigor_rules/architecture.md

**File Information**:
- **Path**: `megadocs/mega_inputs/sigor_rules/architecture.md`

- **Size**: 1064 lines
- **Modified**: 2025-09-18 10:20:37

- **Type**: .md

```text
# Sentio C++ Architecture Document

## Overview

Sentio is a high-performance quantitative trading system built in C++ that implements a strategy-agnostic architecture for systematic trading. The system is designed to maximize profit through sophisticated signal generation, dynamic allocation, and comprehensive audit capabilities.

> **📖 For complete usage instructions, see the [Sentio User Guide](sentio_user_guide.md) which covers both CLI and audit systems.**

## Core Architecture Principles

### 1. Strategy Agnostic Design
- **Runner/Router/Sizer Independence**: Core execution components are completely decoupled from strategy-specific logic
- **Dynamic Strategy Registration**: Strategies are loaded from configuration files without code modifications
- **Unified Signal Interface**: All strategies output probability-based signals (0-1) for consistent processing
- **Extensible Framework**: New strategies integrate seamlessly without architectural changes

### 2. Profit Maximization Mandate
- **Primary Goal**: Maximize monthly projected return rate
- **Capital Efficiency**: Optimize for highest Sharpe score and healthy daily trading range (10-100 trades/day)
- **SOTA Optimization**: Remove artificial constraints, let utility-maximizing frameworks determine optimal allocation
- **Performance Over Safety**: Bias towards aggressive profit-seeking parameters

### 3. Architectural Contract (CRITICAL)
- **Strategy-Agnostic Backend**: Runner, router, and sizer must work with ANY BaseStrategy implementation
- **BaseStrategy API Control**: ALL strategy behavior controlled via BaseStrategy virtual methods
- **100% Capital Deployment**: Always use full available capital for maximum profit
- **Maximum Leverage**: Use leveraged instruments (TQQQ, SQQQ, PSQ) for strong signals
- **Position Integrity**: Never allow negative positions or conflicting long/short positions
- **Extension Protocol**: Extend BaseStrategy API, never modify core systems for specific strategies

## System Components

### Canonical Metrics and Audit Parity

- Single source of truth: All core metrics (MPR, Sharpe, Max Drawdown, Daily Trades) are computed via `UnifiedMetricsCalculator` using day-aware compression.
- Duration-constrained sessions: The expected number of trading sessions is derived from the user input (e.g., `4w -> 28`). Audit independently reconstructs equity from events/BARs and trims the series to the first N distinct US/Eastern sessions to ensure deterministic parity.
- Independence: Audit never trusts strattest outputs; it reconstructs from raw `FILL`/`BAR` events and only uses the input contract (duration/test period) to constrain session counting.
- strattest display policy: For single-simulation runs with a `run_id`, strattest displays the canonical audit metrics (fetched via `audit_db::summarize`) to guarantee that the console report matches audit exactly.
- Effect: Identical metrics across `strattest`, `sentio_audit summarize`, and `sentio_audit position-history` for the same `run_id`, with Trading Days fixed by duration (e.g., 28 for `4w`) and Daily Trades derived consistently.


### 1. Data Filtering Architecture

#### Market Data Processing
- **Raw Data Preservation**: All raw market data from Polygon.io is preserved intact in UTC
- **Pure UTC Processing**: All timestamps are treated as UTC without timezone conversions
- **Holiday Filtering Only**: Data processing excludes US market holidays but keeps all trading hours
- **Extended Hours Support**: Pre-market, regular trading hours, and after-hours data are all utilized
- **Rationale**: More comprehensive data provides better signal generation and market insight

#### Data Pipeline
1. **Raw Download**: Polygon.io data downloaded without time restrictions (UTC timestamps)
2. **Holiday Filtering**: US market holidays removed using pandas_market_calendars
3. **Alignment**: Multiple symbols synchronized to common timestamps
4. **Binary Caching**: Processed data cached as `.bin` files for performance
5. **UTC-Only Processing**: No timezone conversions - all data remains in UTC

### 2. BaseStrategy Extension Pattern

The system uses a rigid architectural pattern to ensure backend systems remain strategy-agnostic:

#### Virtual Method Control
```cpp
class BaseStrategy {
public:
    // Core signal generation
    virtual double calculate_probability(const std::vector<Bar>& bars, int current_index) = 0;
    
    // Allocation decisions for profit maximization
    virtual std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, int current_index,
        const std::string& base_symbol, const std::string& bull3x_symbol, 
        const std::string& bear3x_symbol) = 0;
    
    // Strategy configuration
    virtual RouterCfg get_router_config() const = 0;
    
    // Execution path control
    virtual bool requires_dynamic_allocation() const { return false; }
    
    // Signal description for audit/logging
    virtual std::string get_signal_description(double probability) const;
};
```

#### Strategy-Agnostic Runner Pattern
```cpp
// ✅ CORRECT: Strategy controls its execution path
if (strategy->requires_dynamic_allocation()) {
    // Dynamic allocation path for profit maximization
    auto decisions = strategy->get_allocation_decisions(bars, i, base_symbol, "TQQQ", "SQQQ");
    // Process allocation decisions
} else {
    // Legacy router path for backward compatibility
    auto signal = StrategySignal::from_probability(strategy->calculate_probability(bars, i));
    auto route_decision = route(signal, strategy->get_router_config(), base_symbol);
    // Process router decision
}
```

#### Extension Protocol
1. **Identify New Behavior**: Determine what new capability is needed
2. **Add Virtual Method**: Add virtual method to BaseStrategy with sensible default
3. **Implement in Strategy**: Override method in specific strategy implementations
4. **Update Runner Logic**: Add strategy-agnostic conditional logic in runner
5. **Test Compatibility**: Ensure all existing strategies continue to work

### 3. Strategy Layer

#### Base Strategy Interface
```cpp
class BaseStrategy {
public:
    // Core signal generation - all strategies must implement
    virtual double calculate_probability(const std::vector<Bar>& bars, int current_index) = 0;
    
    // Strategy-agnostic allocation decisions
    virtual std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, int current_index,
        const std::string& base_symbol, const std::string& bull3x_symbol,
        const std::string& bear3x_symbol, const std::string& bear1x_symbol) = 0;
    
    // Strategy-specific configurations
    virtual RouterCfg get_router_config() const = 0;
    virtual SizerCfg get_sizer_config() const = 0;
    
    // Dynamic allocation requirements
    virtual bool requires_dynamic_allocation() const { return false; }
};
```

#### Strategy Signal Standardization
- **Probability Range**: All signals output 0-1 probability where:
  - `1.0` = Very strong buy signal
  - `0.7-1.0` = Strong buy
  - `0.51-0.7` = Buy
  - `0.49-0.51` = Hold/Neutral (no-trade zone)
  - `0.3-0.49` = Sell
  - `0.0-0.3` = Strong sell
  - `0.0` = Very strong sell signal

#### Registered Strategies
1. **IRE (Integrated Rule Ensemble)**: ML-based ensemble with dynamic leverage optimization
2. **TFA (Transformer Financial Analysis)**: Deep learning transformer for sequence modeling
3. **BollingerSqueezeBreakout**: Technical analysis strategy using Bollinger Bands
4. **MomentumVolume**: Price momentum combined with volume analysis
5. **VWAPReversion**: Mean reversion around Volume Weighted Average Price
6. **OrderFlowImbalance**: Market microstructure analysis
7. **MarketMaking**: Bid-ask spread capture strategy
8. **OrderFlowScalping**: High-frequency scalping based on order flow
9. **OpeningRangeBreakout**: Breakout strategy using opening range patterns
10. **KochiPPO**: Reinforcement learning using Proximal Policy Optimization
11. **HybridPPO**: Hybrid approach combining multiple ML techniques

### 2. Execution Layer

#### Runner (Strategy Agnostic)
```cpp
// Core execution loop - completely strategy agnostic
for (const auto& decision : allocation_decisions) {
    if (std::abs(decision.target_weight) > 1e-6) {
        execute_target_position(decision.instrument, decision.target_weight, 
                              portfolio, ST, pricebook, sizer, cfg, series, 
                              bar, chain_id, audit, logging_enabled, total_fills);
    }
}
```

#### Router
- **Instrument Selection**: Routes signals to appropriate instruments (QQQ, PSQ, TQQQ, SQQQ)
- **Leverage Management**: Handles 1x, 3x leveraged instruments
- **Risk Controls**: Implements position limits and exposure controls

#### Sizer
- **Position Sizing**: Calculates optimal position sizes based on volatility targets
- **Risk Management**: Implements maximum position percentage limits
- **Dynamic Allocation**: Adjusts sizing based on market conditions

### 3. Signal Diagnostics System

#### Signal Pipeline Architecture
```
Bar Data → Feature Extraction → Strategy Signal → Signal Gate → Signal Trace → Router
```

#### Diagnostic Components
1. **SignalGate**: Filters and validates signals before execution (RTH filtering removed)
2. **SignalTrace**: Records signal history for analysis
3. **SignalDiag**: Provides real-time signal diagnostics

#### Diagnostic Output Format
```
Signal Diagnostics for [Strategy]:
├── Signal Generation: [Status]
├── Feature Pipeline: [Status]
├── Signal Validation: [Status]
├── Execution Pipeline: [Status]
└── Performance Metrics: [Status]
```

#### Common Signal Issues
- **Feature Pipeline Failures**: Missing or corrupted feature data
- **Signal Validation Errors**: Invalid signal ranges or formats
- **Execution Pipeline Issues**: Router or sizer failures
- **Performance Degradation**: Suboptimal signal quality

#### Troubleshooting Guide
- **No Signals Generated**: Check warmup period, data quality, strategy configuration
- **Low Signal Rate**: Analyze drop reasons, adjust gate parameters
- **Signals But No Trades**: Verify router/sizer configuration
- **High Drop Rate**: Review gate configuration and strategy logic

### 4. Audit System

#### Audit Architecture
```
Execution Events → AuditRecorder → SQLite Database → Audit CLI → Analysis Tools
```

#### Event Types
1. **Signal Events**: Strategy signal generation and validation
2. **Signal Diagnostics**: Signal generation statistics and drop reasons
3. **Order Events**: Order creation, modification, and cancellation
4. **Fill Events**: Trade execution with P&L calculation
5. **Position Events**: Position changes and portfolio updates
6. **Portfolio Snapshots**: Account state and equity tracking

#### Audit Database Schema (SQLite)
```sql
-- Signal diagnostics events
{"ts": 1640995200, "type": "signal_diag", "strategy": "IRE", 
 "emitted": 150, "dropped": 25, "r_min_bars": 5, "r_session": 10, 
 "r_nan": 2, "r_zero_vol": 3, "r_threshold": 3, "r_cooldown": 2, "r_dup": 0}

-- Signal events with chain tracking
{"ts": 1640995200, "type": "signal", "strategy": "IRE", "symbol": "QQQ", 
 "signal": "BUY", "probability": 0.85, "chain_id": "1640995200:123"}

-- Trading events
{"ts": 1640995200, "type": "order", "symbol": "QQQ", "side": "BUY", 
 "quantity": 100, "price": 450.25, "chain_id": "1640995200:123"}
{"ts": 1640995200, "type": "fill", "symbol": "QQQ", "quantity": 100, 
 "price": 450.25, "pnl_d": 1250.50, "chain_id": "1640995200:123"}
```

#### Audit Analysis Tools
- **`sentio_audit`**: Unified C++ CLI for audit analysis
  - **`signal-stats`**: Signal diagnostics and drop analysis
  - **`trade-flow`**: Complete trade sequence visualization
  - **`info`**: Run information and summary statistics
  - **`list`**: List all audit runs with filtering
  - **`export`**: Export data to JSONL/CSV formats
  - **`verify`**: Data integrity validation

#### Signal Diagnostics Integration
The audit system now captures comprehensive signal diagnostics for every run:

**Signal Statistics Tracking**:
- **Emitted Signals**: Total signals generated by strategy
- **Dropped Signals**: Signals filtered out by validation
- **Drop Reasons**: Detailed breakdown of why signals were dropped
  - `WARMUP`: Insufficient warmup period
  - `NAN_INPUT`: Invalid input data
  - `THRESHOLD_TOO_TIGHT`: Below confidence threshold
  - `COOLDOWN_ACTIVE`: Cooldown period active
  - `DUPLICATE_BAR_TS`: Duplicate bar timestamp

**Usage Examples**:
```bash
# View signal diagnostics for latest run
./build/sentio_audit signal-stats

# View signal diagnostics for specific strategy
./build/sentio_audit signal-stats --strategy IRE

# View signal diagnostics for specific run
./build/sentio_audit signal-stats --run IRE_tpa_test_day725_1757576670
```

**Benefits**:
- **Complete Traceability**: Track every signal from generation to execution
- **Performance Analysis**: Understand signal quality and drop patterns
- **Debugging Support**: Identify why signals were filtered out
- **Strategy Optimization**: Analyze signal generation efficiency

#### Instrument Distribution Analysis
```python
def analyze_instrument_distribution(audit_file):
    """Analyze instrument distribution, P&L, and win rates"""
    instruments = defaultdict(lambda: {'trades': 0, 'pnl': 0.0, 'wins': 0})
    
    for event in parse_audit_events(audit_file):
        if event['type'] == 'fill':
            symbol = event['symbol']
            pnl = event['pnl_d']
            instruments[symbol]['trades'] += 1
            instruments[symbol]['pnl'] += pnl
            if pnl > 0:
                instruments[symbol]['wins'] += 1
    
    return instruments
```

### 5. Leverage Trading System

#### Leveraged Instruments
- **QQQ**: 1x NASDAQ-100 ETF (base instrument)
- **PSQ**: 1x inverse NASDAQ-100 ETF (bear 1x)
- **TQQQ**: 3x leveraged NASDAQ-100 ETF (bull 3x)
- **SQQQ**: 3x inverse NASDAQ-100 ETF (bear 3x)

#### Dynamic Leverage Allocation
```cpp
// IRE Strategy Example
std::vector<AllocationDecision> IREStrategy::get_allocation_decisions(...) {
    double probability = calculate_probability(bars, current_index);
    
    if (probability > 0.8) {
        // Strong bullish signal - allocate to TQQQ
        return {{bull3x_symbol, 0.8, probability, "Strong bullish signal"}};
    } else if (probability < 0.2) {
        // Strong bearish signal - allocate to SQQQ
        return {{bear3x_symbol, 0.8, 1.0-probability, "Strong bearish signal"}};
    } else if (probability > 0.6) {
        // Moderate bullish signal - allocate to QQQ
        return {{base_symbol, 0.6, probability, "Moderate bullish signal"}};
    } else if (probability < 0.4) {
        // Moderate bearish signal - allocate to PSQ
        return {{bear1x_symbol, 0.6, 1.0-probability, "Moderate bearish signal"}};
    }
    
    return {}; // No allocation for neutral signals
}
```

#### Leverage Risk Management
- **Position Limits**: Maximum exposure per instrument
- **Volatility Targeting**: Dynamic sizing based on instrument volatility
- **Correlation Controls**: Limits on correlated positions
- **Drawdown Protection**: Automatic position reduction during drawdowns

### 6. Polygon Interface

#### Data Pipeline
```
Polygon API → Data Downloader → Data Aligner → Binary Cache → Strategy Processing
```

#### Data Downloader (`tools/data_downloader.py`)
```python
def download_symbol_data(symbol, years=3, api_key=None):
    """Download historical data from Polygon.io"""
    start_date = datetime.now() - timedelta(days=years*365)
    end_date = datetime.now()
    
    # Download bars with RTH filtering
    bars = polygon_client.get_bars(symbol, start_date, end_date, 
                                 timespan='minute', adjusted=True)
    
    # Filter for Regular Trading Hours
    rth_bars = filter_rth_bars(bars)
    
    # Save to CSV format (all trading hours, no holidays)
    save_bars_to_csv(all_hours_bars, f"{symbol}_NH.csv")
```

#### Data Alignment (`tools/align_bars.py`)
```python
def align_bars(symbols, output_dir="data"):
    """Align timestamps across multiple symbols"""
    all_bars = {}
    for symbol in symbols:
        bars = load_bars_from_csv(f"{symbol}_NH.csv")
        all_bars[symbol] = bars
    
    # Find common timestamps
    common_timestamps = find_common_timestamps(all_bars)
    
    # Align all symbols to common timestamps
    aligned_bars = align_to_timestamps(all_bars, common_timestamps)
    
    # Save aligned data
    for symbol, bars in aligned_bars.items():
        save_bars_to_csv(bars, f"{symbol}.csv")
```

#### Binary Cache System
- **Format**: Custom binary format for fast loading
- **Compression**: LZ4 compression for storage efficiency
- **Indexing**: Fast timestamp-based lookups
- **Validation**: SHA1 checksums for data integrity

### 7. Machine Learning Approaches

#### Offline Training Pipeline
```
Historical Data → Feature Engineering → Model Training → Model Validation → Model Deployment
```

#### ML Strategy Types

##### 1. IRE (Integrated Rule Ensemble)
```cpp
class IREStrategy : public BaseStrategy {
private:
    std::unique_ptr<ml::ModelHandle> model_;
    ml::FeaturePipeline feature_pipeline_;
    
public:
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override {
        // Extract features
        auto features = feature_pipeline_.extract_features(bars, current_index);
        
        // Get model prediction
        auto output = model_->predict(features);
        
        // Convert to probability
        return sigmoid(output.prediction);
    }
};
```

##### 2. TFA (Transformer Financial Analysis)
```cpp
class TFAStrategy : public BaseStrategy {
private:
    std::unique_ptr<ml::TransformerModel> transformer_;
    ml::SequenceContext seq_context_;
    
public:
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override {
        // Build sequence context
        seq_context_.update(bars, current_index);
        
        // Get transformer prediction
        auto output = transformer_->forward(seq_context_);
        
        // Extract probability
        return output.probability;
    }
};
```

##### 3. PPO Strategies (KochiPPO, HybridPPO)
```cpp
class KochiPPOStrategy : public BaseStrategy {
private:
    std::unique_ptr<ml::PPOAgent> ppo_agent_;
    ml::ActionSpace action_space_;
    
public:
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override {
        // Get state representation
        auto state = extract_state(bars, current_index);
        
        // Get PPO action
        auto action = ppo_agent_->act(state);
        
        // Convert action to probability
        return action_to_probability(action);
    }
};
```

#### Feature Engineering
```cpp
class FeaturePipeline {
public:
    struct FeatureSet {
        std::vector<double> technical_indicators;
        std::vector<double> volume_metrics;
        std::vector<double> volatility_measures;
        std::vector<double> momentum_signals;
    };
    
    FeatureSet extract_features(const std::vector<Bar>& bars, int current_index) {
        FeatureSet features;
        
        // Technical indicators
        features.technical_indicators = {
            calculate_rsi(bars, current_index, 14),
            calculate_sma(bars, current_index, 20),
            calculate_bollinger_position(bars, current_index, 20, 2.0)
        };
        
        // Volume metrics
        features.volume_metrics = {
            calculate_volume_ratio(bars, current_index, 20),
            calculate_vwap_deviation(bars, current_index, 20)
        };
        
        // Volatility measures
        features.volatility_measures = {
            calculate_atr(bars, current_index, 14),
            calculate_volatility(bars, current_index, 20)
        };
        
        // Momentum signals
        features.momentum_signals = {
            calculate_momentum(bars, current_index, 10),
            calculate_rate_of_change(bars, current_index, 5)
        };
        
        return features;
    }
};
```

#### Model Training Infrastructure
```python
# tools/ml_training.py
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.feature_pipeline = FeaturePipeline(config.features)
        self.model = self.create_model(config.model_type)
    
    def train(self, training_data, validation_data):
        """Train model with offline data"""
        # Feature extraction
        X_train = self.feature_pipeline.extract_features(training_data)
        y_train = self.extract_labels(training_data)
        
        # Model training
        self.model.fit(X_train, y_train)
        
        # Validation
        X_val = self.feature_pipeline.extract_features(validation_data)
        y_val = self.extract_labels(validation_data)
        
        # Evaluate performance
        metrics = self.model.evaluate(X_val, y_val)
        
        return metrics
    
    def export_model(self, output_path):
        """Export trained model for C++ deployment"""
        self.model.save(output_path)
        self.feature_pipeline.save_config(f"{output_path}_features.json")
```

#### Model Deployment
```cpp
class ModelHandle {
public:
    static std::unique_ptr<ModelHandle> load(const std::string& model_path) {
        auto handle = std::make_unique<ModelHandle>();
        handle->load_model(model_path);
        handle->load_feature_config(f"{model_path}_features.json");
        return handle;
    }
    
    ModelOutput predict(const std::vector<double>& features) {
        // Preprocess features
        auto processed_features = preprocess_features(features);
        
        // Run inference
        auto output = run_inference(processed_features);
        
        // Postprocess output
        return postprocess_output(output);
    }
};
```

### 8. Sanity System

#### Comprehensive Validation Framework
The sanity system provides drop-in C++20 validation checks across the entire pipeline from data ingestion through P&L calculation.

#### Core Components
1. **Sanity Framework**: `SanityIssue` and `SanityReport` for structured error reporting
2. **Deterministic Simulator**: Generates synthetic minute-bar series with realistic market regimes
3. **Property Testing Harness**: Fuzz-like testing framework for invariant validation
4. **Integration Examples**: Complete workflow demonstrations

#### High-Value Bug Detection
- **Time Integrity Issues**: Non-monotonic timestamps, incorrect bar spacing
- **Data Quality Problems**: NaN/Infinity propagation, negative prices, invalid OHLC relationships
- **Instrument Mismatches**: Routed to instruments not in PriceBook
- **Order Execution Errors**: BUY orders with negative quantities, sub-minimum share quantities
- **P&L Calculation Issues**: Equity != cash + realized + mark-to-market
- **Audit Trail Problems**: Fills exceeding orders, missing event sequences

#### Usage Patterns
```cpp
// During data ingestion
SanityReport rep;
sanity::check_bar_monotonic(bars, 60, rep);
sanity::check_bar_values_finite(bars, rep);

// During strategy execution
sanity::check_signal_confidence_range(signal.confidence, rep, timestamp);
sanity::check_routed_instrument_has_price(pricebook, routed_instrument, rep, timestamp);

// During order execution
sanity::check_order_qty_min(qty, min_shares, rep, timestamp);
sanity::check_order_side_qty_sign_consistency(side, qty, rep, timestamp);

// End-of-run validation
sanity::check_equity_consistency(account, positions, pricebook, rep);
sanity::check_audit_counts(event_counts, rep);
```

## Performance Optimization

### 1. Compilation Optimization
- **Release Builds**: `-O3 -march=native -DNDEBUG`
- **Link Time Optimization**: `-flto`
- **Profile Guided Optimization**: `-fprofile-use`

### 2. Runtime Optimization
- **Memory Pool Allocation**: Pre-allocated memory pools for frequent allocations
- **SIMD Instructions**: Vectorized operations for numerical computations
- **Cache-Friendly Data Structures**: Optimized memory layout for hot paths
- **Lock-Free Data Structures**: Minimize synchronization overhead

### 3. Data Access Optimization
- **Binary Cache**: Fast binary format for historical data
- **Memory Mapping**: Memory-mapped files for large datasets
- **Compression**: LZ4 compression for storage efficiency
- **Indexing**: Fast timestamp-based lookups

## Configuration Management

### 1. Strategy Configuration (`configs/strategies.json`)
```json
{
  "strategies": [
    {
      "name": "IRE",
      "class": "IREStrategy",
      "enabled": true,
      "parameters": {
        "lookback_period": 20,
        "volatility_target": 0.15,
        "max_position_pct": 0.8
      }
    },
    {
      "name": "TFA",
      "class": "TFAStrategy",
      "enabled": true,
      "parameters": {
        "sequence_length": 50,
        "attention_heads": 8,
        "max_position_pct": 0.6
      }
    }
  ]
}
```

### 2. TFA Configuration (`configs/tfa.yaml`)
```yaml
model:
  sequence_length: 50
  attention_heads: 8
  hidden_dim: 256
  num_layers: 6

training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  validation_split: 0.2

features:
  technical_indicators: true
  volume_metrics: true
  volatility_measures: true
  momentum_signals: true
```

## Strategy Evaluation Framework

### 1. Signal Quality Evaluation

#### Universal Evaluation Metrics
The system includes a comprehensive evaluation framework for all probability-based trading strategies:

```python
# Core evaluation components
from sentio_trainer.utils.strategy_evaluation import StrategyEvaluator

# Signal quality metrics
evaluator = StrategyEvaluator("StrategyName")
results = evaluator.evaluate_strategy_signal(predictions, actual_returns)
```

#### Evaluation Dimensions
1. **Signal Quality**: Probability range, mean, standard deviation, signal strength
2. **Calibration**: How well probabilities match actual frequencies
3. **Information Content**: Log loss, Brier score, AUC, information ratio
4. **Trading Performance**: Accuracy, precision, recall, F1 score, Sharpe ratio
5. **Overall Assessment**: Weighted score with rating (Excellent/Good/Fair/Poor)

#### Calibration Analysis
- **Calibration Error**: < 0.05 = Excellent, 0.05-0.10 = Good, > 0.10 = Poor
- **Information Ratio**: > 0.2 = High, 0.1-0.2 = Moderate, < 0.1 = Low
- **AUC Score**: 0.5 = Random, 1.0 = Perfect prediction

### 2. Backend Trading System Evaluation

#### Complete Pipeline Evaluation
The system evaluates the complete trading pipeline beyond signal quality:

```
Signal (Probability) → Router → Sizer → Runner → Actual PnL
     ↑                    ↑        ↑        ↑
  Signal              Portfolio    Risk     Execution
  Evaluation          Management   Management Management
```

#### Backend Components Evaluation
1. **Router Performance**: Instrument selection effectiveness, PnL per instrument
2. **Sizer Performance**: Position sizing optimization, risk-adjusted returns
3. **Runner Performance**: Execution costs, slippage, timing efficiency
4. **Signal Effectiveness**: Correlation between signal strength and actual profits

#### Performance Metrics
- **Overall Performance**: Total PnL, win rate, Sharpe ratio, max drawdown, Calmar ratio
- **Execution Quality**: Commission rate, slippage rate, execution cost per trade
- **Risk Management**: Value at Risk (VaR), expected shortfall, turnover rate
- **Signal Correlation**: Signal-to-PnL correlation, signal effectiveness

### 3. Virtual Market Testing (VMTest)

#### VMTest Overview
VMTest provides comprehensive virtual market simulation for strategy testing using multiple data generation approaches. The system supports both synthetic data generation and integration with the MarS (Market Simulation Engine) for realistic market microstructure modeling.

#### VMTest Architecture
```
Data Generation Layer → Strategy Execution → Monte Carlo Testing → Results
         ↑                    ↑                    ↑              ↑
    MarS Engine          SentioStrategy       Statistical    Performance
    Fast Historical      Real Runner          Analysis        Metrics
    Synthetic Data       Integration
```

#### Data Generation Approaches

##### 1. Fast Historical Bridge
- **Purpose**: Instant generation of realistic market data based on historical patterns
- **Speed**: < 1 second per simulation
- **Realism**: Uses actual historical QQQ patterns for volatility, volume, and intraday behavior
- **Time Handling**: Generates timestamps from today's market open (9:30 AM ET)
- **Pattern Analysis**: Extracts mean return, volatility, volume patterns, and hourly multipliers

##### 2. MarS Integration
- **Purpose**: AI-powered market simulation with realistic microstructure
- **Features**: Order-level simulation, market maker behavior, realistic spreads
- **Historical Context**: Uses HistoricalContextAgent for realistic starting conditions
- **AI Continuation**: Optional MarS AI for sophisticated market behavior
- **Performance**: High-quality simulation with realistic market dynamics

##### 3. Synthetic Data Generation
- **Purpose**: Basic synthetic data for rapid testing
- **Speed**: Very fast generation
- **Use Case**: Quick validation and debugging
- **Limitations**: Less realistic than historical or MarS data

#### VMTest CLI Commands

##### Basic VM Test (Fast Historical)
```bash
# Standard VM test with fast historical data
./build/sentio_cli vmtest IRE QQQ --days 30 --simulations 100

# Extended testing with custom parameters
./build/sentio_cli vmtest IRE QQQ --days 70 --simulations 100 --params '{"buy_hi": 0.6, "sell_lo": 0.4}'

# Custom historical data source
./build/sentio_cli strattest ire QQQ --mode historical --duration 14d --simulations 50 --historical-data data/equities/QQQ_NH.csv
```

##### MarS-Powered VM Test
```bash
# MarS simulation with AI
./build/sentio_cli marstest IRE QQQ --days 7 --simulations 20 --regime normal --use-mars-ai

# MarS with historical context
./build/sentio_cli strattest tfa QQQ --mode ai-regime --duration 14d --simulations 10 --regime volatile --historical-data data/equities/QQQ_NH.csv
```

##### Fast Historical Test
```bash
# Direct fast historical test
./build/sentio_cli strattest ire QQQ --mode historical --historical-data data/equities/QQQ_NH.csv --duration 1d --simulations 50
```

#### VMTest Parameters

##### Common Parameters
- **--days <n>**: Number of days to simulate (default: 30)
- **--hours <n>**: Number of hours to simulate (alternative to days)
- **--simulations <n>**: Number of Monte Carlo simulations (default: 100)
- **--params <json>**: Strategy parameters as JSON string
- **--historical-data <file>**: Historical data file for pattern analysis

##### MarS-Specific Parameters
- **--regime <type>**: Market regime (normal, bull_trending, bear_trending, sideways_low_vol, volatile)
- **--use-mars-ai**: Enable MarS AI for sophisticated market behavior
- **--continuation-minutes <n>**: Minutes to simulate beyond historical data

#### VMTest Output Metrics
- **Return Statistics**: Mean, median, standard deviation, min/max returns
- **Confidence Intervals**: 5th, 25th, 75th, 95th percentiles
- **Probability Analysis**: Probability of profit across simulations
- **Performance Metrics**: 
  - Mean Sharpe Ratio
  - Mean MPR (Monthly Projected Return)
  - Mean Daily Trades
- **Signal Diagnostics**: Signal generation and validation metrics
- **Data Quality**: Generated data statistics (price range, volume range, time range)

#### MarS Integration Details

##### Historical Context Agent
```python
class HistoricalContextAgent:
    """Provides realistic starting conditions for MarS simulations"""
    
    def __init__(self, symbol, historical_bars, continuation_minutes):
        self.symbol = symbol
        self.historical_bars = historical_bars
        self.continuation_minutes = continuation_minutes
        
        # Analyze historical patterns
        self.mean_return, self.volatility, self.mean_volume = \
            self._analyze_historical_patterns(historical_bars)
    
    def generate_continuation_orders(self, time):
        """Generate realistic orders based on historical patterns"""
        # Use historical volatility and volume patterns
        # Generate market-making orders with realistic spreads
        # Transition smoothly from historical to synthetic data
```

##### Fast Historical Bridge
```python
def generate_realistic_bars(patterns, start_price, duration_minutes):
    """Generate realistic bars instantly using historical patterns"""
    
    # Use today's market open time
    market_open = get_today_market_open()
    
    # Generate bars with historical patterns
    for i in range(num_bars):
        # Apply hourly volume and volatility multipliers
        volume_multiplier = patterns.hourly_volume_multipliers[hour]
        volatility_multiplier = patterns.hourly_volatility_multipliers[hour]
        
        # Generate realistic price movement
        price_change = np.random.normal(patterns.mean_return, 
                                      patterns.volatility * volatility_multiplier)
        
        # Generate volume with time-of-day patterns
        volume = int(patterns.mean_volume * volume_multiplier * random_factor)
        
        # Create bar with realistic OHLC relationships
        bar = create_bar_with_realistic_ohlc(current_price, price_change, volume)
```

#### VMTest Performance Characteristics

##### Speed Comparison
- **Fast Historical**: < 1 second per simulation
- **MarS (No AI)**: 10-30 seconds per simulation
- **MarS (With AI)**: 30-120 seconds per simulation
- **Synthetic Data**: < 0.1 seconds per simulation

##### Data Quality Ranking
1. **MarS with AI**: Highest realism, sophisticated market behavior
2. **MarS without AI**: High realism, basic market microstructure
3. **Fast Historical**: Good realism, instant generation
4. **Synthetic Data**: Basic realism, fastest generation

##### Use Case Recommendations
- **Development/Testing**: Fast Historical or Synthetic Data
- **Strategy Validation**: MarS without AI
- **Production Simulation**: MarS with AI
- **Quick Diagnostics**: Synthetic Data

### 4. Evaluation Integration

#### Automatic Evaluation During Training
```python
# TFA trainer includes automatic evaluation
python train_models.py --config configs/tfa.yaml
# Output: Comprehensive evaluation metrics + results saved to evaluation_results.json
```

#### Standalone Evaluation Tools
```python
# CLI tool for strategy evaluation
python sentio_trainer/evaluate_strategies.py single --data strategy_data.json --name "MyStrategy"
python sentio_trainer/evaluate_strategies.py compare --data-files strategy1.json strategy2.json
```

#### Programmatic Evaluation
```python
# Quick evaluation
from sentio_trainer.utils.strategy_evaluation import quick_evaluate
results = quick_evaluate(predictions, actual_returns, "StrategyName")

# Detailed evaluation
evaluator = StrategyEvaluator("StrategyName")
results = evaluator.evaluate_strategy_signal(predictions, actual_returns, verbose=True)
evaluator.save_results("results.json")
```

### 5. Evaluation Data Requirements

#### Input Format
- **Predictions**: Raw model outputs (logits), converted to probabilities
- **Actual Returns**: Binary values (1 = price up, 0 = price down)
- **Data Sources**: JSON, NPZ, or CSV formats supported

#### Evaluation Standards
- **Sufficient Data**: Minimum 1000 samples for reliable metrics
- **Balanced Classes**: Avoid extreme class imbalance
- **Clean Data**: Remove outliers and invalid predictions
- **Consistent Time Periods**: Compare strategies on identical data

## Testing and Validation

### 1. Unit Testing
- **Strategy Testing**: Individual strategy signal generation
- **Component Testing**: Router, sizer, and audit components
- **Integration Testing**: End-to-end system testing

### 2. Backtesting Framework
- **Temporal Performance Analysis (TPA)**: Comprehensive backtesting with multiple time periods
- **Performance Metrics**: Sharpe ratio, maximum drawdown, monthly returns
- **Risk Analysis**: Position sizing, leverage analysis, correlation studies

### 3. Live Trading Validation
- **Paper Trading**: Simulated trading with real market data
- **Gradual Deployment**: Phased rollout with position limits
- **Performance Monitoring**: Real-time performance tracking

## Deployment Architecture

### 1. Development Environment
- **Local Development**: macOS development environment
- **Version Control**: Git with GitHub integration
- **Build System**: Makefile-based build system
- **Testing**: Automated testing pipeline

### 2. Production Environment
- **Cloud Deployment**: AWS/GCP cloud infrastructure
- **Containerization**: Docker containers for deployment
- **Monitoring**: Real-time performance monitoring
- **Alerting**: Automated alerting for system issues

### 3. Data Pipeline
- **Real-time Data**: Polygon.io API integration
- **Historical Data**: Comprehensive historical data storage
- **Data Validation**: Automated data quality checks
- **Backup Systems**: Redundant data storage

## Diagnostic Strategy Framework

### 1. Diagnostic Strategy Requirements

The system includes a comprehensive diagnostic strategy framework designed to validate system components and provide baseline performance metrics.

#### Diagnostic Strategy Specifications
- **Purpose**: System validation and infrastructure testing
- **Signal Generation**: RSI-based with aggressive thresholds
- **Frequency**: Minimum 100 signals per day
- **Leverage Support**: QQQ (40%), TQQQ (30%), SQQQ (30%)
- **Objective**: System diagnostics, NOT profit optimization

#### Technical Implementation
```cpp
class DiagnosticStrategy : public BaseStrategy {
private:
    std::vector<double> price_history_;
    int rsi_period_;
    int signal_count_;
    double last_rsi_;
    std::vector<std::string> leverage_symbols_;
    int current_symbol_index_;
    
public:
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override {
        // Calculate RSI from price history
        double rsi = calculate_rsi(price_history_, rsi_period_);
        
        // Generate signals based on RSI thresholds
        if (rsi < 30) return 0.8;      // Strong buy
        if (rsi > 70) return 0.2;      // Strong sell
        if (rsi < 50) return 0.6;      // Moderate buy
        if (rsi > 50) return 0.4;      // Moderate sell
        return 0.5;                    // Neutral
    }
    
    std::vector<AllocationDecision> get_allocation_decisions(...) override {
        // Rotate through leverage symbols
        std::string symbol = select_leverage_ticker();
        double weight = (probability - 0.5) * 2.0; // Convert to -1 to 1
        
        return {{symbol, weight, confidence, "Diagnostic signal"}};
    }
};
```

#### Validation Criteria
- **Signal Generation**: ≥100 signals per day
- **Signal Distribution**: 40% QQQ, 30% TQQQ, 30% SQQQ
- **System Integration**: Successful VM test execution
- **Infrastructure Validation**: Confirms Runner, Router, Sizer functionality

### 2. System Validation Workflow

#### Diagnostic Testing Pipeline
```
Diagnostic Strategy → VM Test → System Validation → Performance Baseline
        ↑              ↑              ↑                    ↑
    RSI Signals    MarS/Fast      Component         Expected
    Generation     Historical     Validation        Metrics
```

#### Validation Steps
1. **Signal Generation Test**: Verify diagnostic strategy generates expected signal frequency
2. **VM Test Integration**: Confirm VM test infrastructure processes signals correctly
3. **Runner Integration**: Validate signal execution and trade generation
4. **Router Validation**: Test multi-symbol routing (QQQ, TQQQ, SQQQ)
5. **Sizer Validation**: Confirm position sizing calculations
6. **Performance Baseline**: Establish expected performance metrics

#### Diagnostic Value
- **Infrastructure Verification**: Confirms all system components are functional
- **Performance Baseline**: Establishes expected signal generation rates
- **Comparison Framework**: Enables comparison with other strategies
- **Debugging Tool**: Helps isolate system vs. strategy issues

## Future Enhancements

### 1. Advanced ML Techniques
- **Reinforcement Learning**: Advanced RL algorithms for strategy optimization
- **Ensemble Methods**: Combining multiple ML models
- **Online Learning**: Continuous model updates with new data
- **Transfer Learning**: Leveraging pre-trained models

### 2. Risk Management
- **Dynamic Risk Controls**: Adaptive risk management based on market conditions
- **Portfolio Optimization**: Multi-strategy portfolio optimization
- **Stress Testing**: Comprehensive stress testing framework
- **Regulatory Compliance**: Automated compliance monitoring

### 3. Performance Optimization
- **GPU Acceleration**: CUDA-based numerical computations
- **Distributed Computing**: Multi-node computation for large-scale analysis
- **Real-time Processing**: Sub-millisecond signal processing
- **Advanced Caching**: Intelligent caching strategies

## Conclusion

The Sentio C++ architecture provides a robust, scalable, and profit-maximizing framework for quantitative trading. The strategy-agnostic design ensures that new strategies can be easily integrated without architectural changes, while the comprehensive audit and diagnostics systems provide full visibility into system performance. The ML integration enables sophisticated signal generation, and the leverage trading system maximizes capital efficiency for optimal returns.

The system is designed to continuously evolve and improve, with a focus on maximizing profit through advanced signal generation, dynamic allocation, and comprehensive risk management. The architecture supports both research and production environments, with robust testing and validation frameworks ensuring reliable performance in live trading scenarios.

```

## 📄 **FILE 3 of 15**: megadocs/mega_inputs/sigor_rules/momentum_volume_rule.hpp

**File Information**:
- **Path**: `megadocs/mega_inputs/sigor_rules/momentum_volume_rule.hpp`

- **Size**: 47 lines
- **Modified**: 2025-09-18 10:20:37

- **Type**: .hpp

```text
#pragma once
#include "irule.hpp"
#include "sentio/rules/utils/validation.hpp"
#include <vector>
#include <deque>
#include <cmath>

namespace sentio::rules {

struct MomentumVolumeRule : IRuleStrategy {
  int mom_win{10}, vol_win{20}; double vol_z{0.0};
  std::vector<double> mom_, vol_ma_, vol_sd_;
  const char* name() const override { return "MOMENTUM_VOLUME"; }
  int warmup() const override { return std::max(mom_win, vol_win)+1; }

  std::optional<RuleOutput> eval(const BarsView& b, int64_t i) override {
    if ((int)mom_.size()!=b.n) build_(b);
    if (i < warmup()) return std::nullopt;
    double mz = (b.volume[i]-vol_ma_[i])/(vol_sd_[i]+1e-9);
    int sig = 0;
    if (mom_[i]>0 && mz>=vol_z) sig=+1;
    else if (mom_[i]<0 && mz>=vol_z) sig=-1;
    return RuleOutput{std::nullopt, sig, (float)mom_[i], 0.6f};
  }

  void build_(const BarsView& b){
    int N=b.n; mom_.assign(N,0); vol_ma_.assign(N,0); vol_sd_.assign(N,1);
    std::vector<double> logc(N,0); logc[0]=std::log(std::max(1e-12,b.close[0]));
    for(int i=1;i<N;i++) logc[i]=std::log(std::max(1e-12,b.close[i]));
    for(int i=0;i<N;i++){ int j=std::max(0,i-mom_win); mom_[i]=logc[i]-logc[j]; }
    
    sentio::rules::utils::SlidingWindow<double> window(vol_win);
    
    for(int i=0;i<N;i++){
      window.push(b.volume[i]);
      
      if (window.has_sufficient_data()) {
        vol_ma_[i] = window.mean();
        vol_sd_[i] = window.standard_deviation();
      }
    }
  }
};

} // namespace sentio::rules



```

## 📄 **FILE 4 of 15**: megadocs/mega_inputs/sigor_rules/ofi_proxy_rule.hpp

**File Information**:
- **Path**: `megadocs/mega_inputs/sigor_rules/ofi_proxy_rule.hpp`

- **Size**: 45 lines
- **Modified**: 2025-09-18 10:20:37

- **Type**: .hpp

```text
#pragma once
#include "irule.hpp"
#include "sentio/rules/utils/validation.hpp"
#include <vector>
#include <deque>
#include <cmath>

namespace sentio::rules {

struct OFIProxyRule : IRuleStrategy {
  int vol_win{20}; double k{1.0};
  std::vector<double> ofi_;
  const char* name() const override { return "OFI_PROXY"; }
  int warmup() const override { return vol_win+1; }

  std::optional<RuleOutput> eval(const BarsView& b, int64_t i) override {
    if ((int)ofi_.size()!=b.n) build_(b);
    if (i < warmup()) return std::nullopt;
    float s = (float)ofi_[i];
    float p = 1.f/(1.f+std::exp(-k*s));
    return RuleOutput{p, std::nullopt, s, 0.6f};
  }

  void build_(const BarsView& b){
    int N=b.n; ofi_.assign(N,0.0);
    std::vector<double> lr(N,0); for(int i=1;i<N;i++) lr[i]=std::log(std::max(1e-12,b.close[i]))-std::log(std::max(1e-12,b.close[i-1]));
    
    sentio::rules::utils::SlidingWindow<double> window(vol_win);
    
    for(int i=0;i<N;i++){
      double x = lr[i]*b.volume[i];
      window.push(x);
      
      if (window.has_sufficient_data()) {
        double m = window.mean();
        double v = window.variance();
        ofi_[i] = (v>0? (x-m)/std::sqrt(v) : 0.0);
      }
    }
  }
};

} // namespace sentio::rules



```

## 📄 **FILE 5 of 15**: megadocs/mega_inputs/sigor_rules/opening_range_breakout_rule.hpp

**File Information**:
- **Path**: `megadocs/mega_inputs/sigor_rules/opening_range_breakout_rule.hpp`

- **Size**: 34 lines
- **Modified**: 2025-09-18 10:20:37

- **Type**: .hpp

```text
#pragma once
#include "irule.hpp"
#include <vector>
#include <algorithm>

namespace sentio::rules {

struct OpeningRangeBreakoutRule : IRuleStrategy {
  int or_bars{30}; double thr{0.000};
  std::vector<double> hi_, lo_;
  const char* name() const override { return "OPENING_RANGE_BRK"; }
  int warmup() const override { return or_bars+1; }

  std::optional<RuleOutput> eval(const BarsView& b, int64_t i) override {
    if ((int)hi_.size()!=b.n) build_(b);
    if (i < warmup()) return std::nullopt;
    double hh=hi_[i], ll=lo_[i], px=b.close[i];
    int sig = (px >= hh*(1.0+thr))? +1 : (px <= ll*(1.0-thr)? -1 : 0);
    return RuleOutput{std::nullopt, sig, std::nullopt, 0.7f};
  }

  void build_(const BarsView& b){
    int N=b.n; hi_.assign(N,b.high[0]); lo_.assign(N,b.low[0]);
    double hh = -1e300, ll = 1e300;
    for(int i=0;i<N;i++){
      if (i<or_bars){ hh=std::max(hh,b.high[i]); ll=std::min(ll,b.low[i]); }
      hi_[i]=hh; lo_[i]=ll;
    }
  }
};

} // namespace sentio::rules



```

## 📄 **FILE 6 of 15**: megadocs/mega_inputs/sigor_rules/rsi_strategy.cpp

**File Information**:
- **Path**: `megadocs/mega_inputs/sigor_rules/rsi_strategy.cpp`

- **Size**: 2 lines
- **Modified**: 2025-09-18 10:20:37

- **Type**: .cpp

```text
#include "sentio/rsi_strategy.hpp"
// Implementations are header-only for simplicity.

```

## 📄 **FILE 7 of 15**: megadocs/mega_inputs/sigor_rules/rsi_strategy.hpp

**File Information**:
- **Path**: `megadocs/mega_inputs/sigor_rules/rsi_strategy.hpp`

- **Size**: 187 lines
- **Modified**: 2025-09-18 10:20:37

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rsi_prob.hpp"
#include "sizer.hpp"
#include <unordered_map>
#include <string>
#include <cmath>

namespace sentio {

class RSIStrategy final : public BaseStrategy {
public:
    RSIStrategy()
    : BaseStrategy("RSI_PROB"),
      rsi_period_(14),
      epsilon_(0.05),             // |w| < eps -> Neutral
      weight_clip_(1.0),
      alpha_(1.0),                // k = ln(2)*alpha ; alpha>1 => steeper
      long_symbol_("QQQ"),
      short_symbol_("SQQQ")
    {}

    ParameterMap get_default_params() const override {
        return {
            {"rsi_period", 14},
            {"epsilon", 0.05},
            {"weight_clip", 1.0},
            {"alpha", 1.0}
        };
    }

    ParameterSpace get_param_space() const override {
        ParameterSpace space;
        space["rsi_period"] = {ParamType::INT, 7, 21, 14};
        space["epsilon"] = {ParamType::FLOAT, 0.01, 0.2, 0.05};
        space["weight_clip"] = {ParamType::FLOAT, 0.5, 2.0, 1.0};
        space["alpha"] = {ParamType::FLOAT, 0.5, 3.0, 1.0};
        return space;
    }

    void apply_params() override {
        auto get = [&](const char* k, double d){
            auto it=params_.find(k); return (it==params_.end() || !std::isfinite(it->second))? d : it->second;
        };
        rsi_period_  = std::max(2, (int)std::llround(get("rsi_period", 14)));
        epsilon_     = std::max(0.0, std::min(0.5, get("epsilon", 0.05)));
        weight_clip_ = std::max(0.1, std::min(2.0, get("weight_clip", 1.0)));
        alpha_       = std::max(0.1, std::min(5.0, get("alpha", 1.0)));
    }

    double calculate_probability(const std::vector<Bar>& bars, int current_index) override {
        if (bars.empty() || current_index < 0 || current_index >= static_cast<int>(bars.size())) {
            diag_.drop(DropReason::MIN_BARS);
            return 0.5; // Neutral
        }
        
        // Need at least rsi_period_ bars to calculate RSI
        if (current_index < rsi_period_) {
            diag_.drop(DropReason::MIN_BARS);
            return 0.5; // Neutral during warmup
        }
        
        // Calculate RSI using the previous rsi_period_ bars
        double rsi = calculate_rsi_from_bars(bars, current_index - rsi_period_, current_index);
        
        // Apply sigmoid transformation
        double probability = rsi_to_prob_tuned(rsi, alpha_);
        
        // Update signal diagnostics
        if (probability != 0.5) {
            diag_.emitted++;
        }
        
        return probability;
    }

    std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, 
        int current_index,
        const std::string& /* base_symbol */,
        const std::string& /* bull3x_symbol */,
        const std::string& /* bear3x_symbol */) override {
        
        std::vector<AllocationDecision> decisions;
        
        if (bars.empty() || current_index < 0 || current_index >= static_cast<int>(bars.size())) {
            return decisions;
        }
        
        double probability = calculate_probability(bars, current_index);
        double weight = 2.0 * (probability - 0.5); // Convert to [-1, 1]
        
        if (std::abs(weight) < epsilon_) {
            // Neutral signal
            AllocationDecision decision;
            decision.instrument = long_symbol_;
            decision.target_weight = 0.0;
            decision.confidence = 0.0;
            decision.reason = "RSI_NEUTRAL";
            decisions.push_back(decision);
        } else {
            // Clip weight
            if (weight > weight_clip_) weight = weight_clip_;
            if (weight < -weight_clip_) weight = -weight_clip_;
            
            AllocationDecision decision;
            if (weight > 0.0) {
                decision.instrument = long_symbol_;
                decision.target_weight = weight;
                decision.confidence = weight;
                decision.reason = "RSI_BULLISH";
            } else {
                decision.instrument = short_symbol_;
                decision.target_weight = weight;
                decision.confidence = -weight;
                decision.reason = "RSI_BEARISH";
            }
            decisions.push_back(decision);
        }
        
        return decisions;
    }

    RouterCfg get_router_config() const override {
        RouterCfg cfg;
        cfg.base_symbol = long_symbol_;
        cfg.bull3x = "TQQQ";
        cfg.bear3x = short_symbol_;
        cfg.bear3x = "SQQQ";
        cfg.max_position_pct = 1.0;
        cfg.min_signal_strength = epsilon_;
        cfg.signal_multiplier = 1.0;
        return cfg;
    }

    // REMOVED: get_sizer_config() - No artificial limits allowed for profit maximization
    // Sizer will use profit-maximizing defaults: 100% capital deployment, maximum leverage

private:
    
    double calculate_rsi_from_bars(const std::vector<Bar>& bars, int start_idx, int end_idx) {
        if (start_idx < 0 || end_idx >= static_cast<int>(bars.size()) || start_idx >= end_idx) {
            return 50.0; // Neutral RSI
        }
        
        // Calculate price changes
        std::vector<double> gains, losses;
        for (int i = start_idx + 1; i <= end_idx; ++i) {
            double change = bars[i].close - bars[i-1].close;
            gains.push_back(change > 0 ? change : 0.0);
            losses.push_back(change < 0 ? -change : 0.0);
        }
        
        // Calculate initial averages (simple average for first period)
        double avg_gain = 0.0, avg_loss = 0.0;
        for (size_t i = 0; i < gains.size(); ++i) {
            avg_gain += gains[i];
            avg_loss += losses[i];
        }
        avg_gain /= gains.size();
        avg_loss /= losses.size();
        
        // Apply Wilder's smoothing for remaining periods
        for (size_t i = 0; i < gains.size(); ++i) {
            avg_gain = (avg_gain * (rsi_period_ - 1) + gains[i]) / rsi_period_;
            avg_loss = (avg_loss * (rsi_period_ - 1) + losses[i]) / rsi_period_;
        }
        
        // Calculate RSI
        if (avg_loss == 0.0) {
            return 100.0; // All gains, no losses
        }
        
        double rs = avg_gain / avg_loss;
        return 100.0 - (100.0 / (1.0 + rs));
    }

    // Params
    int    rsi_period_;
    double epsilon_;
    double weight_clip_;
    double alpha_;
    std::string long_symbol_;
    std::string short_symbol_;
};

} // namespace sentio

```

## 📄 **FILE 8 of 15**: megadocs/mega_inputs/sigor_rules/signal_utils.hpp

**File Information**:
- **Path**: `megadocs/mega_inputs/sigor_rules/signal_utils.hpp`

- **Size**: 126 lines
- **Modified**: 2025-09-18 10:20:37

- **Type**: .hpp

```text
#pragma once

#include "sentio/base_strategy.hpp"
#include <algorithm>
#include <cmath>

namespace sentio::signal_utils {

/**
 * @brief Converts a strategy signal to probability
 * @param sig The strategy signal
 * @param conf_floor Minimum confidence floor
 * @return Probability value (0.0 to 1.0)
 */
inline double signal_to_probability(const StrategySignal& sig, double conf_floor = 0.0) {
    if (sig.confidence < conf_floor) return 0.5; // Neutral
    
    double probability;
    if (sig.type == StrategySignal::Type::BUY) {
        probability = 0.5 + sig.confidence * 0.5; // 0.5 to 1.0
    } else if (sig.type == StrategySignal::Type::SELL) {
        probability = 0.5 - sig.confidence * 0.5; // 0.0 to 0.5
    } else {
        probability = 0.5; // HOLD
    }
    
    return std::clamp(probability, 0.0, 1.0);
}

/**
 * @brief Converts a strategy signal to probability with custom scaling
 * @param sig The strategy signal
 * @param conf_floor Minimum confidence floor
 * @param buy_scale Scaling factor for buy signals
 * @param sell_scale Scaling factor for sell signals
 * @return Probability value (0.0 to 1.0)
 */
inline double signal_to_probability_custom(const StrategySignal& sig, double conf_floor = 0.0,
                                         double buy_scale = 0.5, double sell_scale = 0.5) {
    if (sig.confidence < conf_floor) return 0.5; // Neutral
    
    double probability;
    if (sig.type == StrategySignal::Type::BUY) {
        probability = 0.5 + sig.confidence * buy_scale; // 0.5 to 1.0
    } else if (sig.type == StrategySignal::Type::SELL) {
        probability = 0.5 - sig.confidence * sell_scale; // 0.0 to 0.5
    } else {
        probability = 0.5; // HOLD
    }
    
    return std::clamp(probability, 0.0, 1.0);
}

/**
 * @brief Validates if a signal has sufficient confidence
 * @param sig The strategy signal
 * @param conf_floor Minimum confidence floor
 * @return true if signal has sufficient confidence, false otherwise
 */
inline bool has_sufficient_confidence(const StrategySignal& sig, double conf_floor = 0.0) {
    return sig.confidence >= conf_floor;
}

/**
 * @brief Gets the signal strength (absolute confidence)
 * @param sig The strategy signal
 * @return Signal strength (0.0 to 1.0)
 */
inline double get_signal_strength(const StrategySignal& sig) {
    return std::abs(sig.confidence);
}

/**
 * @brief Determines if signal is a buy signal
 * @param sig The strategy signal
 * @return true if buy signal, false otherwise
 */
inline bool is_buy_signal(const StrategySignal& sig) {
    return sig.type == StrategySignal::Type::BUY;
}

/**
 * @brief Determines if signal is a sell signal
 * @param sig The strategy signal
 * @return true if sell signal, false otherwise
 */
inline bool is_sell_signal(const StrategySignal& sig) {
    return sig.type == StrategySignal::Type::SELL;
}

/**
 * @brief Determines if signal is a hold signal
 * @param sig The strategy signal
 * @return true if hold signal, false otherwise
 */
inline bool is_hold_signal(const StrategySignal& sig) {
    return sig.type == StrategySignal::Type::HOLD;
}

/**
 * @brief Gets the signal direction (-1 for sell, 0 for hold, +1 for buy)
 * @param sig The strategy signal
 * @return Signal direction
 */
inline int get_signal_direction(const StrategySignal& sig) {
    if (sig.type == StrategySignal::Type::BUY) return 1;
    if (sig.type == StrategySignal::Type::SELL) return -1;
    return 0; // HOLD
}

/**
 * @brief Applies confidence floor to a signal
 * @param sig The strategy signal
 * @param conf_floor Minimum confidence floor
 * @return Modified signal with confidence floor applied
 */
inline StrategySignal apply_confidence_floor(const StrategySignal& sig, double conf_floor = 0.0) {
    StrategySignal result = sig;
    if (sig.confidence < conf_floor) {
        result.type = StrategySignal::Type::HOLD;
        result.confidence = 0.0;
    }
    return result;
}

} // namespace sentio::signal_utils

```

## 📄 **FILE 9 of 15**: megadocs/mega_inputs/sigor_rules/sma_cross_rule.hpp

**File Information**:
- **Path**: `megadocs/mega_inputs/sigor_rules/sma_cross_rule.hpp`

- **Size**: 34 lines
- **Modified**: 2025-09-18 10:20:37

- **Type**: .hpp

```text
#pragma once
#include "irule.hpp"
#include <vector>
#include <deque>
#include <algorithm>

namespace sentio::rules {

struct SMACrossRule : IRuleStrategy {
  int fast{10}, slow{20};
  std::vector<double> sma_f_, sma_s_;
  const char* name() const override { return "SMA_CROSS"; }
  int warmup() const override { return std::max(fast, slow); }

  std::optional<RuleOutput> eval(const BarsView& b, int64_t i) override {
    if ((int)sma_f_.size()!=b.n) build_(b);
    if (i < warmup()) return std::nullopt;
    int sig = (sma_f_[i]>sma_s_[i]) ? +1 : (sma_f_[i]<sma_s_[i] ? -1 : 0);
    return RuleOutput{std::nullopt, sig, std::nullopt, 0.6f};
  }

  static void roll_sma_(const double* x, int n, int w, std::vector<double>& out){
    out.assign(n,0.0);
    double s=0; for(int i=0;i<n;i++){ s+=x[i]; if(i>=w) s-=x[i-w]; out[i]=(i>=w-1)? s/w : (i>0? out[i-1]:x[0]); }
  }
  void build_(const BarsView& b){
    sma_f_.assign(b.n,0.0); sma_s_.assign(b.n,0.0);
    roll_sma_(b.close,b.n,fast,sma_f_); roll_sma_(b.close,b.n,slow,sma_s_);
  }
};

} // namespace sentio::rules



```

## 📄 **FILE 10 of 15**: megadocs/mega_inputs/sigor_rules/strategy_bollinger_squeeze_breakout.cpp

**File Information**:
- **Path**: `megadocs/mega_inputs/sigor_rules/strategy_bollinger_squeeze_breakout.cpp`

- **Size**: 200 lines
- **Modified**: 2025-09-18 10:20:37

- **Type**: .cpp

```text
#include "sentio/strategy_bollinger_squeeze_breakout.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

namespace sentio {

    BollingerSqueezeBreakoutStrategy::BollingerSqueezeBreakoutStrategy() 
    : BaseStrategy("BollingerSqueezeBreakout"), bollinger_(20, 2.0) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap BollingerSqueezeBreakoutStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed parameters to be more sensitive to trading opportunities.
    return {
        {"bb_window", 20.0},
        {"bb_k", 1.8},                   // Tighter bands to increase breakout signals
        {"squeeze_percentile", 0.25},    // Squeeze is now top 25% of quietest periods (was 15%)
        {"squeeze_lookback", 60.0},      // Shorter lookback for volatility
        {"hold_max_bars", 120.0},
        {"tp_mult_sd", 1.5},
        {"sl_mult_sd", 1.5},
        {"min_squeeze_bars", 3.0}        // Require at least 3 bars of squeeze
    };
}

ParameterSpace BollingerSqueezeBreakoutStrategy::get_param_space() const { return {}; }

void BollingerSqueezeBreakoutStrategy::apply_params() {
    bb_window_ = static_cast<int>(params_["bb_window"]);
    squeeze_percentile_ = params_["squeeze_percentile"];
    squeeze_lookback_ = static_cast<int>(params_["squeeze_lookback"]);
    hold_max_bars_ = static_cast<int>(params_["hold_max_bars"]);
    tp_mult_sd_ = params_["tp_mult_sd"];
    sl_mult_sd_ = params_["sl_mult_sd"];
    min_squeeze_bars_ = static_cast<int>(params_["min_squeeze_bars"]);
    
    bollinger_ = Bollinger(bb_window_, params_["bb_k"]);
    sd_history_.reserve(squeeze_lookback_);
    reset_state();
}

void BollingerSqueezeBreakoutStrategy::reset_state() {
    BaseStrategy::reset_state();
    state_ = State::Idle;
    bars_in_trade_ = 0;
    squeeze_duration_ = 0;
    sd_history_.clear();
}

double BollingerSqueezeBreakoutStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < squeeze_lookback_) {
        diag_.drop(DropReason::MIN_BARS);
        return 0.5; // Neutral
    }
    
    if (state_ == State::Long || state_ == State::Short) {
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            // Exit signal: return opposite of current position
            double exit_prob = (state_ == State::Long) ? 0.2 : 0.8; // SELL or BUY
            reset_state();
            diag_.emitted++;
            return exit_prob;
        }
        return 0.5; // Hold current position
    }

    update_state_machine(bars[current_index]);

    if (state_ == State::ArmedLong || state_ == State::ArmedShort) {
        if (squeeze_duration_ < min_squeeze_bars_) {
            diag_.drop(DropReason::THRESHOLD);
            state_ = State::Idle;
            return 0.5; // Neutral
        }

        double mid, lo, hi, sd;
        bollinger_.step(bars[current_index].close, mid, lo, hi, sd);
        
        double probability;
        if (state_ == State::ArmedLong) {
            probability = 0.8; // Strong buy signal
            state_ = State::Long;
        } else {
            probability = 0.2; // Strong sell signal  
            state_ = State::Short;
        }
        
        diag_.emitted++;
        bars_in_trade_ = 0;
        return probability;
    } else {
        diag_.drop(DropReason::THRESHOLD);
        return 0.5; // Neutral
    }
}

void BollingerSqueezeBreakoutStrategy::update_state_machine(const Bar& bar) {
    double mid, lo, hi, sd;
    bollinger_.step(bar.close, mid, lo, hi, sd);
    
    sd_history_.push_back(sd);
    if (sd_history_.size() > static_cast<size_t>(squeeze_lookback_)) {
        sd_history_.erase(sd_history_.begin());
    }
    
    double sd_threshold = calculate_volatility_percentile(squeeze_percentile_);
    bool is_squeezed = (sd_history_.size() == static_cast<size_t>(squeeze_lookback_)) && (sd <= sd_threshold);

    switch (state_) {
        case State::Idle:
            if (is_squeezed) {
                state_ = State::Squeezed;
                squeeze_duration_ = 1;
            }
            break;
        case State::Squeezed:
            if (bar.close > hi) state_ = State::ArmedLong;
            else if (bar.close < lo) state_ = State::ArmedShort;
            else if (!is_squeezed) state_ = State::Idle;
            else squeeze_duration_++;
            break;
        default:
            break;
    }
}

// **MODIFIED**: Implemented a proper percentile calculation instead of a stub.
double BollingerSqueezeBreakoutStrategy::calculate_volatility_percentile(double percentile) const {
    if (sd_history_.size() < static_cast<size_t>(squeeze_lookback_)) {
        return std::numeric_limits<double>::max(); // Not enough data, effectively prevents squeeze
    }
    
    std::vector<double> sorted_history = sd_history_;
    std::sort(sorted_history.begin(), sorted_history.end());
    
    int index = static_cast<int>(percentile * (sorted_history.size() - 1));
    return sorted_history[index];
}

std::vector<BaseStrategy::AllocationDecision> BollingerSqueezeBreakoutStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // BollingerSqueezeBreakout uses simple allocation based on signal strength
    if (probability > 0.7) {
        // Strong buy signal
        double conviction = (probability - 0.7) / 0.3; // Scale 0.7-1.0 to 0-1
        double base_weight = 0.4 + (conviction * 0.6); // 40-100% allocation
        
        decisions.push_back({base_symbol, base_weight, conviction, "Bollinger strong buy: 100% QQQ"});
    } else if (probability < 0.3) {
        // Strong sell signal
        double conviction = (0.3 - probability) / 0.3; // Scale 0.0-0.3 to 0-1
        double base_weight = 0.4 + (conviction * 0.6); // 40-100% allocation
        
        // Use SHORT QQQ for moderate sell signals instead of PSQ
        decisions.push_back({base_symbol, -base_weight, conviction, "Bollinger strong sell: SHORT QQQ"});
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "Bollinger: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg BollingerSqueezeBreakoutStrategy::get_router_config() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}

// REMOVED: get_sizer_config() - No artificial limits allowed for profit maximization
// Sizer will use profit-maximizing defaults: 100% capital deployment, maximum leverage

REGISTER_STRATEGY(BollingerSqueezeBreakoutStrategy, "bsb");

} // namespace sentio

```

## 📄 **FILE 11 of 15**: megadocs/mega_inputs/sigor_rules/strategy_bollinger_squeeze_breakout.hpp

**File Information**:
- **Path**: `megadocs/mega_inputs/sigor_rules/strategy_bollinger_squeeze_breakout.hpp`

- **Size**: 57 lines
- **Modified**: 2025-09-18 10:20:37

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "bollinger.hpp"
#include "router.hpp"
#include "sizer.hpp"
#include <vector>
#include <string>

namespace sentio {

class BollingerSqueezeBreakoutStrategy : public BaseStrategy {
private:
    enum class State { Idle, Squeezed, ArmedLong, ArmedShort, Long, Short };
    
    // **MODIFIED**: Cached parameters
    int bb_window_;
    double squeeze_percentile_;
    int squeeze_lookback_;
    int hold_max_bars_;
    double tp_mult_sd_;
    double sl_mult_sd_;
    int min_squeeze_bars_;

    // Strategy state & indicators
    State state_ = State::Idle;
    int bars_in_trade_ = 0;
    int squeeze_duration_ = 0;
    Bollinger bollinger_;
    std::vector<double> sd_history_;
    
    // Helper methods
    double calculate_volatility_percentile(double percentile) const;
    void update_state_machine(const Bar& bar);
    
public:
    BollingerSqueezeBreakoutStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
    
    // **NEW**: Strategy-agnostic allocation interface
    std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, 
        int current_index,
        const std::string& base_symbol,
        const std::string& bull3x_symbol,
        const std::string& bear3x_symbol) override;
    
    RouterCfg get_router_config() const override;
    // REMOVED: get_sizer_config() - No artificial limits allowed for profit maximization
};

} // namespace sentio
```

## 📄 **FILE 12 of 15**: megadocs/mega_inputs/sigor_rules/strategy_signal_or.cpp

**File Information**:
- **Path**: `megadocs/mega_inputs/sigor_rules/strategy_signal_or.cpp`

- **Size**: 274 lines
- **Modified**: 2025-09-18 10:20:37

- **Type**: .cpp

```text
#include "sentio/strategy_signal_or.hpp"
#include "sentio/signal_utils.hpp"
#include "sentio/router.hpp"
#include "sentio/sizer.hpp"
#include "sentio/allocation_manager.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace sentio {

SignalOrStrategy::SignalOrStrategy(const SignalOrCfg& cfg) 
    : BaseStrategy("SignalOR"), cfg_(cfg) {
    // **PROFIT MAXIMIZATION**: Override OR config for more aggressive signals
    cfg_.or_config.aggression = 0.95;      // Maximum aggression for stronger signals
    cfg_.or_config.min_conf = 0.01;       // Lower threshold to capture weak signals
    cfg_.or_config.conflict_soften = 0.2; // Less softening to preserve strong signals
    
    // **MATHEMATICAL ALLOCATION MANAGER**: Initialize with Signal OR tuned parameters
    AllocationConfig alloc_config;
    alloc_config.entry_threshold_1x = cfg_.long_threshold - 0.05;  // Slightly lower for 1x
    alloc_config.entry_threshold_3x = cfg_.long_threshold + 0.15;  // Higher for 3x leverage
    alloc_config.partial_exit_threshold = 0.5 - (cfg_.long_threshold - 0.5) * 0.5; // Dynamic
    alloc_config.full_exit_threshold = 0.5 - (cfg_.long_threshold - 0.5) * 0.8;    // More aggressive
    alloc_config.min_signal_change = cfg_.min_signal_strength;     // Align with strategy config
    
    allocation_manager_ = std::make_unique<AllocationManager>(alloc_config);
    
    apply_params();
}

// Required BaseStrategy methods
double SignalOrStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < 0 || current_index >= static_cast<int>(bars.size())) {
        return 0.5; // Neutral if invalid index
    }
    
    warmup_bars_++;
    
    // Evaluate simple rules and apply Signal-OR mixing
    auto rule_outputs = evaluate_simple_rules(bars, current_index);
    
    if (rule_outputs.empty()) {
        return 0.5; // Neutral if no rules active
    }
    
    // Apply Signal-OR mixing
    double probability = mix_signal_or(rule_outputs, cfg_.or_config);
    
    // **FIXED**: Update signal diagnostics counter
    diag_.emitted++;
    
    return probability;
}

std::vector<SignalOrStrategy::AllocationDecision> SignalOrStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    if (current_index < 0 || current_index >= static_cast<int>(bars.size())) {
        return decisions; // Empty if invalid index
    }
    
    double probability = calculate_probability(bars, current_index);
    
    // **STATE-AWARE TRANSITION ALGORITHM**
    // Determine target position based on signal strength
    std::string target_instrument = "";
    double target_weight = 0.0;
    std::string reason = "";
    
    if (probability > 0.7) {
        // Strong buy: 100% TQQQ (3x leveraged long)
        target_instrument = bull3x_symbol;
        target_weight = 1.0;
        reason = "Signal OR strong buy: 100% TQQQ (3x leverage)";
        
    } else if (probability > cfg_.long_threshold) {
        // Moderate buy: 100% QQQ (1x long)
        target_instrument = base_symbol;
        target_weight = 1.0;
        reason = "Signal OR moderate buy: 100% QQQ";
        
    } else if (probability < 0.3) {
        // Strong sell: 100% SQQQ (3x leveraged short)
        target_instrument = bear3x_symbol;
        target_weight = 1.0;
        reason = "Signal OR strong sell: 100% SQQQ (3x inverse)";
        
    } else if (probability < cfg_.short_threshold) {
        // Weak sell: 100% PSQ (1x inverse)
        target_instrument = "PSQ";
        target_weight = 1.0;
        reason = "Signal OR weak sell: 100% PSQ (1x inverse)";
        
    } else {
        // Neutral: Stay in cash
        target_instrument = "CASH";
        target_weight = 0.0;
        reason = "Signal OR neutral: Stay in cash";
    }
    
    // **TEMPORARY SIMPLE ALLOCATION**: Return target if different from last bar
    bool different_bar = (current_index != last_decision_bar_);
    
    if (different_bar && target_instrument != "CASH") {
        // Return only the target allocation - runner will handle atomic rebalancing
        decisions.push_back({target_instrument, target_weight, probability, reason});
        last_decision_bar_ = current_index;
    }
    // If target is CASH or same bar, return empty decisions (no action needed)
    
    return decisions;
}

RouterCfg SignalOrStrategy::get_router_config() const {
    RouterCfg cfg;
    
    // **PROFIT MAXIMIZATION**: Configure router for maximum leverage and 100% capital deployment
    cfg.min_signal_strength = 0.01;    // Lower threshold to capture more signals
    cfg.signal_multiplier = 1.0;       // No scaling
    cfg.max_position_pct = 1.0;        // 100% position size (profit maximization)
    cfg.require_rth = true;
    
    // Instrument configuration
    cfg.base_symbol = "QQQ";
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: PSQ will be handled via SHORT QQQ for moderate sell signals
    
    cfg.min_shares = 1.0;
    cfg.lot_size = 1.0;
    cfg.ire_min_conf_strong_short = 0.85;
    
    return cfg;
}

// REMOVED: get_sizer_config() - No artificial limits allowed for profit maximization
// Sizer will use profit-maximizing defaults: 100% capital deployment, maximum leverage

// Configuration
ParameterMap SignalOrStrategy::get_default_params() const {
    return {
        {"min_signal_strength", cfg_.min_signal_strength},
        {"long_threshold", cfg_.long_threshold},
        {"short_threshold", cfg_.short_threshold},
        {"hold_threshold", cfg_.hold_threshold},
        {"momentum_window", static_cast<double>(cfg_.momentum_window)},
        {"momentum_scale", cfg_.momentum_scale},
        {"or_aggression", cfg_.or_config.aggression},
        {"or_min_conf", cfg_.or_config.min_conf},
        {"or_conflict_soften", cfg_.or_config.conflict_soften}
    };
}

ParameterSpace SignalOrStrategy::get_param_space() const {
    ParameterSpace space;
    space["min_signal_strength"] = {ParamType::FLOAT, 0.05, 0.3, cfg_.min_signal_strength};
    space["long_threshold"] = {ParamType::FLOAT, 0.55, 0.75, cfg_.long_threshold};
    space["short_threshold"] = {ParamType::FLOAT, 0.25, 0.45, cfg_.short_threshold};
    space["momentum_window"] = {ParamType::INT, 10, 50, static_cast<double>(cfg_.momentum_window)};
    space["momentum_scale"] = {ParamType::FLOAT, 10.0, 50.0, cfg_.momentum_scale};
    space["or_aggression"] = {ParamType::FLOAT, 0.6, 0.95, cfg_.or_config.aggression};
    space["or_min_conf"] = {ParamType::FLOAT, 0.01, 0.2, cfg_.or_config.min_conf};
    space["or_conflict_soften"] = {ParamType::FLOAT, 0.2, 0.6, cfg_.or_config.conflict_soften};
    return space;
}

void SignalOrStrategy::apply_params() {
    // Apply parameters from the parameter map
    if (params_.count("min_signal_strength")) {
        cfg_.min_signal_strength = params_.at("min_signal_strength");
    }
    if (params_.count("long_threshold")) {
        cfg_.long_threshold = params_.at("long_threshold");
    }
    if (params_.count("short_threshold")) {
        cfg_.short_threshold = params_.at("short_threshold");
    }
    if (params_.count("hold_threshold")) {
        cfg_.hold_threshold = params_.at("hold_threshold");
    }
    if (params_.count("momentum_window")) {
        cfg_.momentum_window = static_cast<int>(params_.at("momentum_window"));
    }
    if (params_.count("momentum_scale")) {
        cfg_.momentum_scale = params_.at("momentum_scale");
    }
    if (params_.count("or_aggression")) {
        cfg_.or_config.aggression = params_.at("or_aggression");
    }
    if (params_.count("or_min_conf")) {
        cfg_.or_config.min_conf = params_.at("or_min_conf");
    }
    if (params_.count("or_conflict_soften")) {
        cfg_.or_config.conflict_soften = params_.at("or_conflict_soften");
    }
    
    // Reset state
    warmup_bars_ = 0;
}

// Helper methods
std::vector<RuleOut> SignalOrStrategy::evaluate_simple_rules(const std::vector<Bar>& bars, int current_index) {
    std::vector<RuleOut> outputs;
    
    // Rule 1: Momentum-based probability
    double momentum_prob = calculate_momentum_probability(bars, current_index);
    RuleOut momentum_out;
    momentum_out.p01 = momentum_prob;
    momentum_out.conf01 = std::abs(momentum_prob - 0.5) * 2.0; // Confidence based on deviation from neutral
    outputs.push_back(momentum_out);
    
    // Rule 2: Volume-based probability (if we have volume data)
    if (current_index > 0 && bars[current_index].volume > 0 && bars[current_index - 1].volume > 0) {
        double volume_ratio = static_cast<double>(bars[current_index].volume) / bars[current_index - 1].volume;
        double volume_prob = 0.5 + std::clamp((volume_ratio - 1.0) * 0.1, -0.2, 0.2); // Volume momentum
        RuleOut volume_out;
        volume_out.p01 = volume_prob;
        volume_out.conf01 = std::min(0.5, std::abs(volume_ratio - 1.0) * 0.5); // Confidence based on volume change
        outputs.push_back(volume_out);
    }
    
    // Rule 3: Price volatility-based probability
    if (current_index >= 5) {
        double volatility = 0.0;
        for (int i = current_index - 4; i <= current_index; ++i) {
            double ret = (bars[i].close - bars[i-1].close) / bars[i-1].close;
            volatility += ret * ret;
        }
        volatility = std::sqrt(volatility / 5.0);
        
        // Higher volatility suggests trend continuation
        double vol_prob = 0.5 + std::clamp(volatility * 10.0, -0.2, 0.2);
        RuleOut vol_out;
        vol_out.p01 = vol_prob;
        vol_out.conf01 = std::min(0.3, volatility * 5.0); // Confidence based on volatility
        outputs.push_back(vol_out);
    }
    
    return outputs;
}

double SignalOrStrategy::calculate_momentum_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < cfg_.momentum_window) {
        return 0.5; // Neutral if not enough data
    }
    
    // Calculate moving average
    double ma = 0.0;
    for (int i = current_index - cfg_.momentum_window + 1; i <= current_index; ++i) {
        ma += bars[i].close;
    }
    ma /= cfg_.momentum_window;
    
    // Calculate momentum
    double momentum = (bars[current_index].close - ma) / ma;
    
    // **PROFIT MAXIMIZATION**: Allow extreme probabilities for leverage triggers
    double momentum_prob = 0.5 + std::clamp(momentum * cfg_.momentum_scale, -0.45, 0.45);
    
    return momentum_prob;
}

// **PROFIT MAXIMIZATION**: Old position weight calculation removed
// Now using 100% capital deployment with maximum leverage

} // namespace sentio
```

## 📄 **FILE 13 of 15**: megadocs/mega_inputs/sigor_rules/strategy_signal_or.hpp

**File Information**:
- **Path**: `megadocs/mega_inputs/sigor_rules/strategy_signal_or.hpp`

- **Size**: 108 lines
- **Modified**: 2025-09-18 10:20:37

- **Type**: .hpp

```text
#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/signal_or.hpp"
#include "sentio/allocation_manager.hpp"
#include <algorithm>
#include <vector>
#include <memory>

namespace sentio {

// Signal-OR Strategy Configuration
struct SignalOrCfg {
    // Signal-OR mixer configuration
    OrCfg or_config;
    
    // **PROFIT MAXIMIZATION**: Aggressive thresholds for maximum leverage usage
    double min_signal_strength = 0.05; // Lower threshold to capture more signals
    double long_threshold = 0.55;       // Lower threshold to capture more moderate longs
    double short_threshold = 0.45;      // Higher threshold to capture more moderate shorts
    double hold_threshold = 0.02;       // Tighter hold band to force more action
    
    // **PROFIT MAXIMIZATION**: Remove artificial limits
    // max_position_weight removed - always use 100% capital
    // position_decay removed - not needed for profit maximization
    
    // **PROFIT MAXIMIZATION**: Aggressive momentum for strong signals
    int momentum_window = 10;            // Shorter window for more responsive signals
    double momentum_scale = 50.0;       // Higher scaling for stronger signals
};

// Signal-OR Strategy Implementation
class SignalOrStrategy : public BaseStrategy {
public:
    explicit SignalOrStrategy(const SignalOrCfg& cfg = SignalOrCfg{});
    
    // Required BaseStrategy methods
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override;
    std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, 
        int current_index,
        const std::string& base_symbol,
        const std::string& bull3x_symbol,
        const std::string& bear3x_symbol) override;
    RouterCfg get_router_config() const override;
    // REMOVED: get_sizer_config() - No artificial limits allowed for profit maximization
    
    // **ARCHITECTURAL COMPLIANCE**: Use dynamic allocation with strategy-agnostic conflict prevention
    bool requires_dynamic_allocation() const override { return true; }
    
    // **STRATEGY-SPECIFIC CONFLICT RULES**: Define instrument conflict constraints
    bool allows_simultaneous_positions(const std::string& instrument1, const std::string& instrument2) const override {
        // Define instrument groups
        std::vector<std::string> long_instruments = {"QQQ", "TQQQ"};
        std::vector<std::string> inverse_instruments = {"PSQ", "SQQQ"};
        
        auto is_long = [&](const std::string& inst) {
            return std::find(long_instruments.begin(), long_instruments.end(), inst) != long_instruments.end();
        };
        auto is_inverse = [&](const std::string& inst) {
            return std::find(inverse_instruments.begin(), inverse_instruments.end(), inst) != inverse_instruments.end();
        };
        
        // Rule 1: Cannot hold multiple long instruments simultaneously
        if (is_long(instrument1) && is_long(instrument2)) return false;
        
        // Rule 2: Cannot hold multiple inverse instruments simultaneously  
        if (is_inverse(instrument1) && is_inverse(instrument2)) return false;
        
        // Rule 3: Cannot hold long + inverse simultaneously
        if ((is_long(instrument1) && is_inverse(instrument2)) || 
            (is_inverse(instrument1) && is_long(instrument2))) return false;
        
        return true; // All other combinations allowed
    }
    
    // **STRATEGY-SPECIFIC TRANSITION CONTROL**: Require sequential transitions for conflicts
    bool requires_sequential_transitions() const override { return true; }
    
    // Configuration
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    // Signal-OR specific methods
    void set_or_config(const OrCfg& config) { cfg_.or_config = config; }
    const OrCfg& get_or_config() const { return cfg_.or_config; }

private:
    SignalOrCfg cfg_;
    
    // State tracking
    int warmup_bars_ = 0;
    static constexpr int REQUIRED_WARMUP = 50;
    
    // **MATHEMATICAL ALLOCATION MANAGER**: State-aware portfolio transitions
    std::unique_ptr<AllocationManager> allocation_manager_;
    int last_decision_bar_ = -1;
    
    // Helper methods
    std::vector<RuleOut> evaluate_simple_rules(const std::vector<Bar>& bars, int current_index);
    double calculate_momentum_probability(const std::vector<Bar>& bars, int current_index);
    // **PROFIT MAXIMIZATION**: Old position weight methods removed
};

// Register the strategy with the factory
REGISTER_STRATEGY(SignalOrStrategy, "sigor");

} // namespace sentio

```

## 📄 **FILE 14 of 15**: megadocs/mega_inputs/sigor_rules/strategy_utils.hpp

**File Information**:
- **Path**: `megadocs/mega_inputs/sigor_rules/strategy_utils.hpp`

- **Size**: 65 lines
- **Modified**: 2025-09-18 10:20:37

- **Type**: .hpp

```text
#pragma once

#include <chrono>
#include <unordered_map>

namespace sentio {

/**
 * Utility functions for strategy implementations
 */
class StrategyUtils {
public:
    /**
     * Check if cooldown period is active for a given symbol
     * @param symbol The trading symbol
     * @param last_trade_time Last trade timestamp
     * @param cooldown_seconds Cooldown period in seconds
     * @return true if cooldown is active
     */
    static bool is_cooldown_active(
        const std::string& symbol,
        int64_t last_trade_time,
        int cooldown_seconds
    ) {
        if (cooldown_seconds <= 0) {
            return false;
        }
        
        auto now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        
        return (now - last_trade_time) < cooldown_seconds;
    }
    
    /**
     * Check if cooldown period is active for a given symbol with per-symbol tracking
     * @param symbol The trading symbol
     * @param last_trade_times Map of symbol to last trade time
     * @param cooldown_seconds Cooldown period in seconds
     * @return true if cooldown is active
     */
    static bool is_cooldown_active(
        const std::string& symbol,
        const std::unordered_map<std::string, int64_t>& last_trade_times,
        int cooldown_seconds
    ) {
        if (cooldown_seconds <= 0) {
            return false;
        }
        
        auto it = last_trade_times.find(symbol);
        if (it == last_trade_times.end()) {
            return false;
        }
        
        auto now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        
        return (now - it->second) < cooldown_seconds;
    }
};

} // namespace sentio

```

## 📄 **FILE 15 of 15**: megadocs/mega_inputs/sigor_rules/vwap_reversion_rule.hpp

**File Information**:
- **Path**: `megadocs/mega_inputs/sigor_rules/vwap_reversion_rule.hpp`

- **Size**: 42 lines
- **Modified**: 2025-09-18 10:20:37

- **Type**: .hpp

```text
#pragma once
#include "irule.hpp"
#include <vector>
#include <deque>
#include <cmath>

namespace sentio::rules {

struct VWAPReversionRule : IRuleStrategy {
  int win{20}; double z_lo{-1.0}, z_hi{+1.0};
  std::vector<double> vwap_, sd_;
  const char* name() const override { return "VWAP_REVERSION"; }
  int warmup() const override { return win+1; }

  std::optional<RuleOutput> eval(const BarsView& b, int64_t i) override {
    if ((int)vwap_.size()!=b.n) build_(b);
    if (i < warmup()) return std::nullopt;
    double z=(b.close[i]-vwap_[i])/(sd_[i]+1e-9);
    int sig = (z<=z_lo)? +1 : (z>=z_hi? -1 : 0);
    return RuleOutput{std::nullopt, sig, (float)z, 0.6f};
  }

  void build_(const BarsView& b){
    int N=b.n; vwap_.assign(N,0); sd_.assign(N,1.0);
    std::deque<double> qv,qpv; double sv=0, spv=0; std::deque<double> qdiff; double s2=0;
    for(int i=0;i<N;i++){
      double pv=b.close[i]*b.volume[i];
      qv.push_back(b.volume[i]); qpv.push_back(pv); sv+=b.volume[i]; spv+=pv;
      if((int)qv.size()>win){ sv-=qv.front(); spv-=qpv.front(); qv.pop_front(); qpv.pop_front(); }
      vwap_[i] = (sv>0? spv/sv : b.close[i]);

      double d=b.close[i]-vwap_[i];
      qdiff.push_back(d); s2+=d*d;
      if((int)qdiff.size()>win){ double z=qdiff.front(); qdiff.pop_front(); s2-=z*z; }
      sd_[i] = std::sqrt(std::max(0.0, s2/std::max(1,(int)qdiff.size())));
    }
  }
};

} // namespace sentio::rules



```

