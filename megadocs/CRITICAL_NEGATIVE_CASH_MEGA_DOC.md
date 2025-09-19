# CRITICAL: Negative Cash Balance - Fundamental System Violation

**Generated**: 2025-09-18 19:47:10
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: CRITICAL analysis of negative cash balance indicating fundamental failure in position sizing and risk management systems

**Total Files**: 180

---

## 🐛 **BUG REPORT**

# CRITICAL BUG REPORT: Negative Cash Balance - Fundamental System Violation

## 🚨 CRITICAL SEVERITY - SYSTEM HALT REQUIRED

**Status**: **CRITICAL SYSTEM FAILURE**  
**Impact**: **FUNDAMENTAL TRADING SYSTEM VIOLATION**  
**Risk Level**: **MAXIMUM**  
**Production Status**: **ABSOLUTELY NOT READY**

## Executive Summary

The Sentio trading system has a **CRITICAL FUNDAMENTAL FLAW** in position sizing that allows **impossible negative cash balances**. The system is currently showing:

- **Cash Balance**: **-$1,480,342.37** (IMPOSSIBLE)
- **Position Value**: **$1,581,336.66** 
- **Starting Capital**: **$100,000.00**
- **Effective Leverage**: **15.8x** (WITHOUT MARGIN)

This represents a **complete failure** of basic trading system principles and **must be fixed immediately**.

## Critical Issue Analysis

### 🚨 **Root Cause: Sizer Logic Fundamental Flaw**

**File**: `include/sentio/sizer.hpp` (Lines 68-70)

```cpp
// **PROFIT MAXIMIZATION MANDATE**: Use 100% of capital with maximum leverage
// No artificial constraints - let the strategy determine optimal allocation
double desired_notional = equity * std::abs(target_weight);
```

**The Fatal Flaw**:
The sizer calculates position size based on **total equity** (including unrealized P&L from existing positions) rather than **available cash**. This creates a **compounding leverage effect**:

1. **First Position**: PSQ = $100,994 (based on total equity)
2. **Second Position**: SQQQ = $100,994 (based on total equity INCLUDING PSQ)
3. **Third Position**: TQQQ = $100,994 (based on total equity INCLUDING PSQ + SQQQ)

**Result**: **$300,000+ in positions** on **$100,000 capital** = **Impossible negative cash**

### 🚨 **Current Impossible State**

```
Starting Capital:    $  100,000.00
Current Positions:
├─ PSQ:    3,015 shares × $129.37 = $  389,158.58
├─ SQQQ:  32,851 shares × $ 18.04 = $  564,188.81  
├─ TQQQ:     400 shares × $1558.92 = $  627,989.27
└─ TOTAL POSITIONS:                   $1,581,336.66

Cash Balance:        $-1,480,342.37  ← IMPOSSIBLE!
Total Equity:        $  100,994.29   ← Matches starting capital
```

### 🚨 **System Violations**

1. **Cash Cannot Go Negative**: Fundamental trading system rule violated
2. **No Margin System**: System buying positions it cannot afford
3. **Leverage Without Collateral**: 15.8x leverage on cash account
4. **Position Sizing Failure**: Sizer not checking available cash
5. **Risk Management Failure**: No position size limits enforced

## Technical Analysis

### **Sizer Configuration Issues**

**File**: `include/sentio/sizer.hpp` (Lines 12-23)

```cpp
// Profit-Maximizing Sizer Configuration - NO ARTIFICIAL LIMITS
struct SizerCfg {
  // REMOVED: All artificial constraints that limit profit
  // - max_leverage: Always use maximum available leverage
  // - max_position_pct: Always use 100% of capital
  // - allow_negative_cash: Always enabled for margin trading
  // - cash_reserve_pct: No cash reserves, deploy 100% of capital
};
```

**Critical Issues**:
1. **No Cash Validation**: No check for available cash before sizing
2. **No Leverage Limits**: Unlimited leverage allowed
3. **No Position Limits**: No maximum position size constraints
4. **No Risk Controls**: All risk management removed

### **Portfolio Accounting Issues**

The portfolio accounting system is correctly tracking the impossible state but not preventing it:

- **Equity Calculation**: Correctly shows $100,994 total equity
- **Cash Tracking**: Correctly shows negative cash balance
- **Position Tracking**: Correctly shows over-leveraged positions
- **Problem**: No validation to prevent impossible states

## Impact Assessment

### **Risk Level**: **CRITICAL - MAXIMUM**
- **Financial Risk**: Unlimited leverage exposure
- **System Risk**: Fundamental trading principles violated
- **Operational Risk**: System in impossible state
- **Regulatory Risk**: Violates basic trading regulations

### **Business Impact**: **CRITICAL**
- **Trading Impossible**: System cannot execute real trades
- **Risk Unlimited**: No position size controls
- **Compliance Failure**: Violates financial regulations
- **System Integrity**: Complete loss of system reliability

## Required Immediate Actions

### **Priority 1: IMMEDIATE SYSTEM HALT**
1. **Stop All Trading**: Halt system immediately
2. **Investigate All Positions**: Review all historical trades
3. **Validate All Calculations**: Check all P&L calculations
4. **Assess System Integrity**: Full system audit required

### **Priority 2: CRITICAL FIXES REQUIRED**

#### **Fix 1: Sizer Cash Validation**
```cpp
// REQUIRED: Check available cash before sizing
double available_cash = portfolio.cash;
if (available_cash <= 0) return 0.0; // Cannot buy without cash

double max_notional = available_cash; // Use available cash, not equity
double desired_notional = std::min(max_notional, equity * std::abs(target_weight));
```

#### **Fix 2: Position Size Limits**
```cpp
// REQUIRED: Maximum position size limits
double max_position_pct = 0.95; // Maximum 95% of capital per position
double max_notional = equity * max_position_pct;
```

#### **Fix 3: Leverage Controls**
```cpp
// REQUIRED: Maximum leverage limits
double max_leverage = 1.0; // Cash account = 1x leverage maximum
double total_position_value = calculate_total_positions(portfolio, last_prices);
if (total_position_value / equity > max_leverage) return 0.0;
```

#### **Fix 4: Cash Reserve Requirements**
```cpp
// REQUIRED: Minimum cash reserve
double min_cash_reserve_pct = 0.05; // Minimum 5% cash reserve
double available_for_trading = equity * (1.0 - min_cash_reserve_pct);
```

### **Priority 3: System Validation**
1. **Unit Tests**: Test all position sizing scenarios
2. **Integration Tests**: Test complete trading workflows
3. **Stress Tests**: Test extreme market conditions
4. **Compliance Tests**: Verify regulatory compliance

## Conclusion

The Sentio trading system has a **CRITICAL FUNDAMENTAL FLAW** that makes it **completely unsuitable for production trading**:

1. **Negative Cash Balance**: -$1,480,342.37 (IMPOSSIBLE)
2. **Unlimited Leverage**: 15.8x leverage without margin
3. **No Risk Controls**: All position size limits removed
4. **System Integrity Failure**: Fundamental trading principles violated

**Status**: **CRITICAL SYSTEM FAILURE - IMMEDIATE HALT REQUIRED**

**Recommendation**: **Complete system redesign** of position sizing and risk management before any further trading.

---

**Report Generated**: 2025-09-18  
**Test Run**: 891913  
**Status**: CRITICAL FAILURE - NOT PRODUCTION READY  
**Action Required**: IMMEDIATE SYSTEM HALT


---

## 📋 **TABLE OF CONTENTS**

1. [include/sentio/accurate_leverage_pricing.hpp](#file-1)
2. [include/sentio/all_strategies.hpp](#file-2)
3. [include/sentio/allocation_manager.hpp](#file-3)
4. [include/sentio/alpha.hpp](#file-4)
5. [include/sentio/alpha/sota_linear_policy.hpp](#file-5)
6. [include/sentio/audit.hpp](#file-6)
7. [include/sentio/audit_interface.hpp](#file-7)
8. [include/sentio/base_strategy.hpp](#file-8)
9. [include/sentio/binio.hpp](#file-9)
10. [include/sentio/bo.hpp](#file-10)
11. [include/sentio/bollinger.hpp](#file-11)
12. [include/sentio/canonical_evaluation.hpp](#file-12)
13. [include/sentio/canonical_metrics.hpp](#file-13)
14. [include/sentio/cli_helpers.hpp](#file-14)
15. [include/sentio/core.hpp](#file-15)
16. [include/sentio/core/bar.hpp](#file-16)
17. [include/sentio/cost_model.hpp](#file-17)
18. [include/sentio/csv_loader.hpp](#file-18)
19. [include/sentio/data_downloader.hpp](#file-19)
20. [include/sentio/data_resolver.hpp](#file-20)
21. [include/sentio/dataset_metadata.hpp](#file-21)
22. [include/sentio/day_index.hpp](#file-22)
23. [include/sentio/detectors/bollinger_detector.hpp](#file-23)
24. [include/sentio/detectors/momentum_volume_detector.hpp](#file-24)
25. [include/sentio/detectors/ofi_proxy_detector.hpp](#file-25)
26. [include/sentio/detectors/opening_range_breakout_detector.hpp](#file-26)
27. [include/sentio/detectors/rsi_detector.hpp](#file-27)
28. [include/sentio/detectors/vwap_reversion_detector.hpp](#file-28)
29. [include/sentio/eod_position_manager.hpp](#file-29)
30. [include/sentio/exec/asof_index.hpp](#file-30)
31. [include/sentio/exec_types.hpp](#file-31)
32. [include/sentio/execution/pnl_engine.hpp](#file-32)
33. [include/sentio/family_mapper.hpp](#file-33)
34. [include/sentio/feature/column_projector.hpp](#file-34)
35. [include/sentio/feature/column_projector_safe.hpp](#file-35)
36. [include/sentio/feature/csv_feature_provider.hpp](#file-36)
37. [include/sentio/feature/feature_builder_guarded.hpp](#file-37)
38. [include/sentio/feature/feature_builder_ops.hpp](#file-38)
39. [include/sentio/feature/feature_feeder_guarded.hpp](#file-39)
40. [include/sentio/feature/feature_from_spec.hpp](#file-40)
41. [include/sentio/feature/feature_matrix.hpp](#file-41)
42. [include/sentio/feature/feature_provider.hpp](#file-42)
43. [include/sentio/feature/name_diff.hpp](#file-43)
44. [include/sentio/feature/ops.hpp](#file-44)
45. [include/sentio/feature/sanitize.hpp](#file-45)
46. [include/sentio/feature/standard_scaler.hpp](#file-46)
47. [include/sentio/feature_builder.hpp](#file-47)
48. [include/sentio/feature_cache.hpp](#file-48)
49. [include/sentio/feature_engineering/feature_normalizer.hpp](#file-49)
50. [include/sentio/feature_engineering/kochi_features.hpp](#file-50)
51. [include/sentio/feature_engineering/technical_indicators.hpp](#file-51)
52. [include/sentio/feature_feeder.hpp](#file-52)
53. [include/sentio/feature_utils.hpp](#file-53)
54. [include/sentio/future_qqq_loader.hpp](#file-54)
55. [include/sentio/global_leverage_config.hpp](#file-55)
56. [include/sentio/indicators.hpp](#file-56)
57. [include/sentio/leverage_aware_csv_loader.hpp](#file-57)
58. [include/sentio/leverage_pricing.hpp](#file-58)
59. [include/sentio/mars_data_loader.hpp](#file-59)
60. [include/sentio/metrics.hpp](#file-60)
61. [include/sentio/metrics/mpr.hpp](#file-61)
62. [include/sentio/metrics/session_utils.hpp](#file-62)
63. [include/sentio/ml/feature_pipeline.hpp](#file-63)
64. [include/sentio/ml/feature_window.hpp](#file-64)
65. [include/sentio/ml/iml_model.hpp](#file-65)
66. [include/sentio/ml/model_registry.hpp](#file-66)
67. [include/sentio/ml/ts_model.hpp](#file-67)
68. [include/sentio/of_index.hpp](#file-68)
69. [include/sentio/of_precompute.hpp](#file-69)
70. [include/sentio/orderflow_types.hpp](#file-70)
71. [include/sentio/pnl_accounting.hpp](#file-71)
72. [include/sentio/polygon_client.hpp](#file-72)
73. [include/sentio/portfolio/fee_model.hpp](#file-73)
74. [include/sentio/portfolio/portfolio_allocator.hpp](#file-74)
75. [include/sentio/portfolio/tc_slippage_model.hpp](#file-75)
76. [include/sentio/portfolio/utilization_governor.hpp](#file-76)
77. [include/sentio/position_coordinator.hpp](#file-77)
78. [include/sentio/position_validator.hpp](#file-78)
79. [include/sentio/pricebook.hpp](#file-79)
80. [include/sentio/profiling.hpp](#file-80)
81. [include/sentio/progress_bar.hpp](#file-81)
82. [include/sentio/property_test.hpp](#file-82)
83. [include/sentio/rolling_stats.hpp](#file-83)
84. [include/sentio/router.hpp](#file-84)
85. [include/sentio/rsi_prob.hpp](#file-85)
86. [include/sentio/rules/adapters.hpp](#file-86)
87. [include/sentio/rules/bbands_squeeze_rule.hpp](#file-87)
88. [include/sentio/rules/diversity_weighter.hpp](#file-88)
89. [include/sentio/rules/integrated_rule_ensemble.hpp](#file-89)
90. [include/sentio/rules/irule.hpp](#file-90)
91. [include/sentio/rules/momentum_volume_rule.hpp](#file-91)
92. [include/sentio/rules/ofi_proxy_rule.hpp](#file-92)
93. [include/sentio/rules/online_platt_calibrator.hpp](#file-93)
94. [include/sentio/rules/opening_range_breakout_rule.hpp](#file-94)
95. [include/sentio/rules/registry.hpp](#file-95)
96. [include/sentio/rules/sma_cross_rule.hpp](#file-96)
97. [include/sentio/rules/utils/validation.hpp](#file-97)
98. [include/sentio/rules/vwap_reversion_rule.hpp](#file-98)
99. [include/sentio/run_id_generator.hpp](#file-99)
100. [include/sentio/runner.hpp](#file-100)
101. [include/sentio/side.hpp](#file-101)
102. [include/sentio/signal.hpp](#file-102)
103. [include/sentio/signal_diag.hpp](#file-103)
104. [include/sentio/signal_engine.hpp](#file-104)
105. [include/sentio/signal_gate.hpp](#file-105)
106. [include/sentio/signal_or.hpp](#file-106)
107. [include/sentio/signal_pipeline.hpp](#file-107)
108. [include/sentio/signal_trace.hpp](#file-108)
109. [include/sentio/signal_utils.hpp](#file-109)
110. [include/sentio/sim_data.hpp](#file-110)
111. [include/sentio/sizer.hpp](#file-111)
112. [include/sentio/strategy/intraday_position_governor.hpp](#file-112)
113. [include/sentio/strategy_kochi_ppo.hpp](#file-113)
114. [include/sentio/strategy_signal_or.hpp](#file-114)
115. [include/sentio/strategy_tfa.hpp](#file-115)
116. [include/sentio/strategy_transformer.hpp](#file-116)
117. [include/sentio/strategy_utils.hpp](#file-117)
118. [include/sentio/sym/leverage_registry.hpp](#file-118)
119. [include/sentio/sym/symbol_utils.hpp](#file-119)
120. [include/sentio/symbol_table.hpp](#file-120)
121. [include/sentio/test_strategy.hpp](#file-121)
122. [include/sentio/tfa/artifacts_loader.hpp](#file-122)
123. [include/sentio/tfa/artifacts_safe.hpp](#file-123)
124. [include/sentio/tfa/feature_guard.hpp](#file-124)
125. [include/sentio/tfa/input_shim.hpp](#file-125)
126. [include/sentio/tfa/signal_pipeline.hpp](#file-126)
127. [include/sentio/tfa/tfa_seq_context.hpp](#file-127)
128. [include/sentio/time_utils.hpp](#file-128)
129. [include/sentio/torch/safe_from_blob.hpp](#file-129)
130. [include/sentio/unified_metrics.hpp](#file-130)
131. [include/sentio/unified_strategy_tester.hpp](#file-131)
132. [include/sentio/util/bytes.hpp](#file-132)
133. [include/sentio/util/safe_matrix.hpp](#file-133)
134. [include/sentio/utils/formatting.hpp](#file-134)
135. [include/sentio/utils/validation.hpp](#file-135)
136. [include/sentio/virtual_market.hpp](#file-136)
137. [include/sentio/wf.hpp](#file-137)
138. [src/accurate_leverage_pricing.cpp](#file-138)
139. [src/allocation_manager.cpp](#file-139)
140. [src/audit.cpp](#file-140)
141. [src/base_strategy.cpp](#file-141)
142. [src/canonical_evaluation.cpp](#file-142)
143. [src/canonical_metrics.cpp](#file-143)
144. [src/cli_helpers.cpp](#file-144)
145. [src/csv_loader.cpp](#file-145)
146. [src/data_downloader.cpp](#file-146)
147. [src/eod_position_manager.cpp](#file-147)
148. [src/feature_builder.cpp](#file-148)
149. [src/feature_cache.cpp](#file-149)
150. [src/feature_engineering/feature_normalizer.cpp](#file-150)
151. [src/feature_engineering/kochi_features.cpp](#file-151)
152. [src/feature_engineering/technical_indicators.cpp](#file-152)
153. [src/feature_feeder.cpp](#file-153)
154. [src/future_qqq_loader.cpp](#file-154)
155. [src/global_leverage_config.cpp](#file-155)
156. [src/leverage_aware_csv_loader.cpp](#file-156)
157. [src/leverage_pricing.cpp](#file-157)
158. [src/main.cpp](#file-158)
159. [src/mars_data_loader.cpp](#file-159)
160. [src/ml/model_registry_ts.cpp](#file-160)
161. [src/ml/ts_model.cpp](#file-161)
162. [src/pnl_accounting.cpp](#file-162)
163. [src/poly_fetch_main.cpp](#file-163)
164. [src/polygon_client.cpp](#file-164)
165. [src/position_coordinator.cpp](#file-165)
166. [src/router.cpp](#file-166)
167. [src/run_id_generator.cpp](#file-167)
168. [src/runner.cpp](#file-168)
169. [src/signal_engine.cpp](#file-169)
170. [src/signal_gate.cpp](#file-170)
171. [src/signal_pipeline.cpp](#file-171)
172. [src/signal_trace.cpp](#file-172)
173. [src/strategy/run_rule_ensemble.cpp](#file-173)
174. [src/strategy_initialization.cpp](#file-174)
175. [src/strategy_signal_or.cpp](#file-175)
176. [src/strategy_tfa.cpp](#file-176)
177. [src/test_strategy.cpp](#file-177)
178. [src/time_utils.cpp](#file-178)
179. [src/unified_metrics.cpp](#file-179)
180. [src/virtual_market.cpp](#file-180)

---

## 📄 **FILE 1 of 180**: include/sentio/accurate_leverage_pricing.hpp

**File Information**:
- **Path**: `include/sentio/accurate_leverage_pricing.hpp`

- **Size**: 133 lines
- **Modified**: 2025-09-14 11:29:13

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "sentio/sym/leverage_registry.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <random>

namespace sentio {

// High-accuracy leverage cost model with proper time scaling
struct AccurateLeverageCostModel {
    double expense_ratio{0.0095};        // 0.95% annual expense ratio (default for SQQQ/PSQ)
    double borrowing_cost_rate{0.05};    // 5% annual borrowing cost
    double daily_rebalance_cost{0.0001}; // 0.01% daily rebalancing cost
    double bid_ask_spread{0.0002};       // 0.02% bid-ask spread per trade
    double tracking_error_daily{0.0001}; // 0.01% daily tracking error std
    
    // Create PSQ-specific cost model (1x inverse with 0.95% expense ratio)
    static AccurateLeverageCostModel create_psq_model() {
        AccurateLeverageCostModel model;
        model.expense_ratio = 0.0095;        // 0.95% annual expense ratio
        model.borrowing_cost_rate = 0.02;    // Lower borrowing cost for 1x vs 3x
        model.daily_rebalance_cost = 0.00005; // Lower rebalancing cost for 1x
        model.bid_ask_spread = 0.0001;       // Tighter spread for 1x
        model.tracking_error_daily = 0.00005; // Lower tracking error for 1x
        return model;
    }
    
    // Create TQQQ-specific cost model (3x long)
    static AccurateLeverageCostModel create_tqqq_model() {
        AccurateLeverageCostModel model;
        model.expense_ratio = 0.0086;        // 0.86% annual expense ratio for TQQQ
        model.borrowing_cost_rate = 0.05;    // Higher borrowing cost for 3x
        model.daily_rebalance_cost = 0.0001; // Standard rebalancing cost
        model.bid_ask_spread = 0.0002;       // Standard spread
        model.tracking_error_daily = 0.0001; // Standard tracking error
        return model;
    }
    
    // Calculate minute-level cost rate (properly scaled)
    double minute_cost_rate() const {
        // Scale annual costs to minute level (252 trading days * 390 minutes per day)
        double annual_minutes = 252.0 * 390.0;
        return (expense_ratio + borrowing_cost_rate) / annual_minutes;
    }
    
    // Calculate daily-level cost rate
    double daily_cost_rate() const {
        return (expense_ratio + borrowing_cost_rate) / 252.0 + daily_rebalance_cost;
    }
};

// High-accuracy theoretical leverage pricing engine
class AccurateLeveragePricer {
private:
    AccurateLeverageCostModel cost_model_;
    std::mt19937 rng_;
    std::unordered_map<std::string, double> cumulative_tracking_error_;
    std::unordered_map<std::string, int64_t> last_daily_reset_;
    
    // Reset tracking error daily (simulates daily rebalancing)
    void reset_daily_tracking_if_needed(const std::string& symbol, int64_t current_timestamp);
    
public:
    AccurateLeveragePricer(const AccurateLeverageCostModel& cost_model = AccurateLeverageCostModel{});
    
    // Calculate theoretical price with high accuracy
    double calculate_accurate_theoretical_price(const std::string& leverage_symbol,
                                              double base_price_prev,
                                              double base_price_current,
                                              double leverage_price_prev,
                                              int64_t timestamp = 0);
    
    // Generate theoretical bar with minimal error
    Bar generate_accurate_theoretical_bar(const std::string& leverage_symbol,
                                        const Bar& base_bar_prev,
                                        const Bar& base_bar_current,
                                        const Bar& leverage_bar_prev);
    
    // Generate theoretical series starting from actual first price
    std::vector<Bar> generate_accurate_theoretical_series(const std::string& leverage_symbol,
                                                        const std::vector<Bar>& base_series,
                                                        const std::vector<Bar>& actual_series_for_init);
    
    // Update cost model
    void update_cost_model(const AccurateLeverageCostModel& new_model) { cost_model_ = new_model; }
    
    // Get current cost model
    const AccurateLeverageCostModel& get_cost_model() const { return cost_model_; }
};

// High-accuracy validator for sub-1% error validation
class AccurateLeveragePricingValidator {
private:
    AccurateLeveragePricer pricer_;
    
public:
    struct AccurateValidationResult {
        std::string symbol;
        double price_correlation;
        double return_correlation;
        double mean_price_error_pct;
        double price_error_std_pct;
        double mean_return_error_pct;
        double return_error_std_pct;
        double max_price_error_pct;
        double max_return_error_pct;
        double theoretical_total_return;
        double actual_total_return;
        double return_difference_pct;
        int num_observations;
        bool sub_1pct_accuracy;
    };
    
    AccurateLeveragePricingValidator(const AccurateLeverageCostModel& cost_model = AccurateLeverageCostModel{});
    
    // Validate with sub-1% accuracy target
    AccurateValidationResult validate_accurate_pricing(const std::string& leverage_symbol,
                                                     const std::vector<Bar>& base_series,
                                                     const std::vector<Bar>& actual_leverage_series);
    
    // Print detailed validation report
    void print_accurate_validation_report(const AccurateValidationResult& result);
    
    // Auto-calibrate for sub-1% accuracy
    AccurateLeverageCostModel calibrate_for_accuracy(const std::string& leverage_symbol,
                                                   const std::vector<Bar>& base_series,
                                                   const std::vector<Bar>& actual_leverage_series,
                                                   double target_error_pct = 1.0);
};

} // namespace sentio

```

## 📄 **FILE 2 of 180**: include/sentio/all_strategies.hpp

**File Information**:
- **Path**: `include/sentio/all_strategies.hpp`

- **Size**: 8 lines
- **Modified**: 2025-09-18 14:34:50

- **Type**: .hpp

```text
#pragma once

// This file ensures all strategies are included and registered with the factory.
// Include this header once in your main.cpp.

// Essential strategies for bare minimum system
#include "strategy_tfa.hpp"
#include "strategy_signal_or.hpp"
```

## 📄 **FILE 3 of 180**: include/sentio/allocation_manager.hpp

**File Information**:
- **Path**: `include/sentio/allocation_manager.hpp`

- **Size**: 145 lines
- **Modified**: 2025-09-18 16:51:27

- **Type**: .hpp

```text
#pragma once

#include "core.hpp"
#include "symbol_table.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

namespace sentio {

// **MATHEMATICAL ALLOCATION MANAGER**: State-aware portfolio transitions
// Based on threshold models and optimal stopping theory

enum class PositionType {
    CASH,           // No position
    LONG_1X,        // QQQ (1x long)
    LONG_3X,        // TQQQ (3x long)  
    INVERSE_1X,     // PSQ (1x inverse)
    INVERSE_3X      // SQQQ (3x inverse)
};

enum class AllocationAction {
    HOLD,           // Maintain current position
    PARTIAL_CLOSE,  // Close portion of position (e.g., 3x -> 1x)
    FULL_CLOSE,     // Close entire position -> CASH
    ENTER_NEW       // Enter new position from CASH
};

struct PositionState {
    PositionType type = PositionType::CASH;
    double weight = 0.0;                    // Current position weight (0-1)
    double entry_probability = 0.0;         // Signal strength when entered
    double unrealized_pnl_pct = 0.0;       // Current unrealized P&L %
    int bars_held = 0;                      // Bars since position entry
    double max_favorable_prob = 0.0;       // Strongest favorable signal seen
    double max_adverse_prob = 0.0;          // Strongest adverse signal seen
};

struct InternalAllocationDecision {
    AllocationAction action;
    PositionType target_type;
    double target_weight;
    std::string reason;
    double confidence;                      // Decision confidence (0-1)
};

// **THRESHOLD-BASED ALLOCATION PARAMETERS**
struct AllocationConfig {
    // **ENTRY THRESHOLDS**: Signal strength required to enter new positions
    double entry_threshold_1x = 0.55;      // Enter 1x position
    double entry_threshold_3x = 0.70;      // Enter 3x position
    
    // **EXIT THRESHOLDS**: Opposing signal strength to trigger exits
    double partial_exit_threshold = 0.45;  // Reduce leverage (3x -> 1x)
    double full_exit_threshold = 0.35;     // Full exit to cash
    
    // **POSITION STRENGTH FACTORS**: Influence exit decisions
    double profit_protection_factor = 0.1; // Lower exit threshold if profitable
    double loss_cutting_factor = -0.1;     // Higher exit threshold if losing
    double momentum_decay_factor = 0.02;   // Reduce thresholds over time
    
    // **TRANSACTION COST CONSIDERATIONS**
    double min_signal_change = 0.05;       // Minimum signal change to consider action
    double holding_inertia = 0.02;         // Bias toward holding current position
    
    // **RISK MANAGEMENT**
    double max_adverse_tolerance = 0.25;   // Max adverse signal before forced exit
    int max_holding_period = 240;          // Max bars to hold position (force review)
    
    // **DIRECTIONAL TRANSITION LOGIC**
    int min_holding_period = 5;            // Minimum bars to hold position before direction change
    double strong_signal_bypass_threshold = 0.80; // Strong signals can bypass min holding period
    double weak_opposite_cash_threshold = 0.60;   // Weak opposite signals go to cash during holding
};

class AllocationManager {
private:
    AllocationConfig config_;
    PositionState current_state_;
    double starting_capital_ = 100000.0; // Default starting capital
    
    // **DIRECTIONAL TRANSITION STATE**
    int bars_in_current_position_ = 0;     // Bars since position was established
    bool is_directional_position_ = false; // True if holding long/short (not cash)
    
    // **MATHEMATICAL DECISION FUNCTIONS**
    double calculate_exit_threshold(double base_threshold, const PositionState& state) const;
    double calculate_position_strength(const PositionState& state) const;
    double calculate_signal_momentum(double current_prob, const PositionState& state) const;
    bool is_signal_opposing(double probability, PositionType position_type) const;
    bool is_signal_favorable(double probability, PositionType position_type) const;
    PositionType select_entry_position_type(double probability) const;
    
    // **DIRECTIONAL TRANSITION HELPERS**
    bool is_long_position(PositionType type) const;
    bool is_short_position(PositionType type) const;
    bool is_opposite_direction(PositionType current, PositionType target) const;
    bool can_change_direction(double signal_strength, PositionType current, PositionType target) const;
    
public:
    AllocationManager(const AllocationConfig& config = AllocationConfig{});
    
    // **MAIN ALLOCATION INTERFACE**
    InternalAllocationDecision make_allocation_decision(
        double current_probability,         // Current signal probability (0-1)
        double current_unrealized_pnl_pct,  // Current position P&L %
        int bars_since_last_decision        // Bars since last allocation change
    );
    
    // **RUNNER INTERFACE**: Convert to runner allocation format
    struct RunnerAllocationDecision {
        std::string instrument;
        double target_weight;
        double confidence;
        std::string reason;
    };
    
    std::vector<RunnerAllocationDecision> get_runner_allocations(
        double current_probability,
        const Portfolio& current_portfolio,
        const std::vector<double>& last_prices,
        const SymbolTable& ST
    );
    
    // **STATE MANAGEMENT**
    void update_position_state(const InternalAllocationDecision& decision, double probability);
    void reset_state();
    
    // **ANALYTICS**
    const PositionState& get_current_state() const { return current_state_; }
    double get_decision_confidence() const;
    
    // **CONFIGURATION**
    void update_config(const AllocationConfig& config) { config_ = config; }
    const AllocationConfig& get_config() const { return config_; }
};

// **UTILITY FUNCTIONS**
std::string position_type_to_string(PositionType type);
std::string position_type_to_symbol(PositionType type);
double get_position_leverage(PositionType type);
bool are_positions_conflicting(PositionType type1, PositionType type2);

} // namespace sentio

```

## 📄 **FILE 4 of 180**: include/sentio/alpha.hpp

**File Information**:
- **Path**: `include/sentio/alpha.hpp`

- **Size**: 7 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <string>

namespace sentio {
// Removed: Direction enum and StratSignal struct. These are now defined in core.hpp
} // namespace sentio


```

## 📄 **FILE 5 of 180**: include/sentio/alpha/sota_linear_policy.hpp

**File Information**:
- **Path**: `include/sentio/alpha/sota_linear_policy.hpp`

- **Size**: 84 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <algorithm>
#include <cmath>

namespace sentio::alpha {

// Map probability to conditional mean return (per bar).
// Heuristic: if next-bar sign ~ Bernoulli(p_up), then E[r] ≈ k_ret * (2p-1) * sigma,
// where sigma is realized vol for the bar horizon; k_ret calibrates label/return link.
inline double prob_to_mu(double p_up, double sigma, double k_ret=1.0){
  p_up = std::clamp(p_up, 1e-6, 1.0-1e-6);
  return k_ret * (2.0*p_up - 1.0) * sigma;
}

struct SotaPolicyCfg {
  double gamma     = 4.0;   // risk aversion (higher => smaller positions)
  double k_ret     = 1.0;   // maps prob edge to mean return units
  double lam_tc    = 5e-4;  // turnover penalty (per unit weight change)
  double min_edge  = 0.0;   // optional: deadband in mu minus costs
  double max_abs_w = 0.50;  // per-name cap on weight (leverage)
};

// One-step SOTA linear policy with costs (1-asset version).
// Inputs: p_up in [0,1], sigma (per-bar vol), prev weight w_prev, est. one-shot cost in bps.
inline double sota_linear_weight(double p_up, double sigma, double w_prev, double cost_bps, const SotaPolicyCfg& cfg){
  sigma = std::max(1e-6, sigma);
  // 1) Aim (Merton/Kelly): w* = mu / (gamma * sigma^2)
  const double mu = prob_to_mu(p_up, sigma, cfg.k_ret);
  double w_aim = mu / (cfg.gamma * sigma * sigma);

  // 2) Cost-aware partial adjustment (Gârleanu–Pedersen style)
  // Solve: min_w  (gamma*sigma^2/2)*(w - w_aim)^2 + lam_tc*|w - w_prev|
  // Closed-form with L1 gives a soft-threshold around w_aim; approximate with shrinkage:
  const double k = cfg.gamma * sigma * sigma;
  double w_free = (k*w_aim + cfg.lam_tc*w_prev) / (k + cfg.lam_tc); // Ridge-like blend
  // Apply a small deadband if expected edge can't beat costs
  const double edge_bps = 1e4 * std::abs(mu); // rough edge proxy (bps)
  if (edge_bps < cost_bps + cfg.min_edge) {
    // shrink toward previous to avoid churn
    w_free = 0.5*w_prev + 0.5*w_free;
  }

  // 3) Cap leverage
  w_free = std::clamp(w_free, -cfg.max_abs_w, cfg.max_abs_w);
  return w_free;
}

// Decision helper (hold/long/short) — useful for audits:
enum class Dir { HOLD=0, LONG=+1, SHORT=-1 };
inline Dir direction_from_weight(double w, double tol=1e-3){
  if (w >  tol) return Dir::LONG;
  if (w < -tol) return Dir::SHORT;
  return Dir::HOLD;
}

// Multi-asset version: returns target weights vector given p_up vector and covariance
// This implements the multi-asset Merton rule with partial adjustment
inline std::vector<double> sota_multi_asset_weights(
    const std::vector<double>& p_up,
    const std::vector<double>& sigma,
    const std::vector<double>& w_prev,
    const std::vector<double>& cost_bps,
    const SotaPolicyCfg& cfg) {
    
    const int N = p_up.size();
    std::vector<double> w_target(N, 0.0);
    
    for (int i = 0; i < N; i++) {
        w_target[i] = sota_linear_weight(p_up[i], sigma[i], w_prev[i], cost_bps[i], cfg);
    }
    
    // Apply gross constraint (portfolio-level leverage cap)
    double gross = 0.0;
    for (double w : w_target) gross += std::abs(w);
    
    if (gross > cfg.max_abs_w * N) { // rough gross cap
        double scale = (cfg.max_abs_w * N) / gross;
        for (double& w : w_target) w *= scale;
    }
    
    return w_target;
}

} // namespace sentio::alpha

```

## 📄 **FILE 6 of 180**: include/sentio/audit.hpp

**File Information**:
- **Path**: `include/sentio/audit.hpp`

- **Size**: 112 lines
- **Modified**: 2025-09-11 16:46:18

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <optional>
#include <unordered_map>
#include <vector>
#include "sentio/audit_interface.hpp"

namespace sentio {

// --- Core enums/structs ----
enum class SigType : uint8_t { BUY=0, STRONG_BUY=1, SELL=2, STRONG_SELL=3, HOLD=4 };
enum class Side    : uint8_t { Buy=0, Sell=1 };

// Simple Bar structure for audit system (avoiding conflicts with core.hpp)
struct AuditBar { 
  double open{}, high{}, low{}, close{}, volume{};
};

struct AuditPosition { 
  double qty{0.0}; 
  double avg_px{0.0}; 
};

struct AccountState {
  double cash{0.0};
  double realized{0.0};
  double equity{0.0};
  // computed: equity = cash + realized + sum(qty * mark_px)
};

struct AuditConfig {
  std::string run_id;           // stable id for this run
  std::string file_path;        // where JSONL events are appended
  bool        flush_each=true;  // fsync-ish (fflush) after each write
};

// --- Recorder: append events to JSONL ---
class AuditRecorder : public IAuditRecorder {
public:
  explicit AuditRecorder(const AuditConfig& cfg);
  ~AuditRecorder();

  // lifecycle
  void event_run_start(std::int64_t ts_utc, const std::string& meta_json="{}");
  void event_run_end(std::int64_t ts_utc, const std::string& meta_json="{}");

  // data plane
  void event_bar   (std::int64_t ts_utc, const std::string& instrument, double open, double high, double low, double close, double volume);
  void event_signal(std::int64_t ts_utc, const std::string& base_symbol, SigType type, double confidence);
  void event_route (std::int64_t ts_utc, const std::string& base_symbol, const std::string& instrument, double target_weight);
  void event_order (std::int64_t ts_utc, const std::string& instrument, Side side, double qty, double limit_px);
  void event_fill  (std::int64_t ts_utc, const std::string& instrument, double price, double qty, double fees, Side side);
  void event_snapshot(std::int64_t ts_utc, const AccountState& acct);
  void event_metric (std::int64_t ts_utc, const std::string& key, double value);

  // Extended events with chain id and richer context for precise trade linking and P/L deltas.
  void event_signal_ex(std::int64_t ts_utc, const std::string& base_symbol, SigType type, double confidence,
                       const std::string& chain_id);
  void event_route_ex (std::int64_t ts_utc, const std::string& base_symbol, const std::string& instrument, double target_weight,
                       const std::string& chain_id);
  void event_order_ex (std::int64_t ts_utc, const std::string& instrument, Side side, double qty, double limit_px,
                       const std::string& chain_id);
  void event_fill_ex  (std::int64_t ts_utc, const std::string& instrument, double price, double qty, double fees, Side side,
                       double realized_pnl_delta, double equity_after, double position_after,
                       const std::string& chain_id);

  // Signal diagnostics events
  void event_signal_diag(std::int64_t ts_utc, const std::string& strategy_name, const SignalDiag& diag);
  void event_signal_drop(std::int64_t ts_utc, const std::string& strategy_name, const std::string& symbol, 
                        DropReason reason, const std::string& chain_id, const std::string& note = "");

  // Get current config (for creating new instances)
  AuditConfig get_config() const { return {run_id_, file_path_, flush_each_}; }

private:
  std::string run_id_;
  std::string file_path_;
  std::FILE*  fp_{nullptr};
  std::uint64_t seq_{0};
  bool flush_each_;
  void write_line_(const std::string& s);
  static std::string sha1_hex_(const std::string& s); // tiny local impl
  static std::string json_escape_(const std::string& s);
};

// --- Replayer: read JSONL, rebuild state, recompute P&L, verify ---
struct ReplayResult {
  // recomputed
  std::unordered_map<std::string, AuditPosition> positions;
  AccountState acct{};
  std::size_t  bars{0}, signals{0}, routes{0}, orders{0}, fills{0};
  // mismatches discovered
  std::vector<std::string> issues;
};

class AuditReplayer {
public:
  // price map can be filled from bar events; you may also inject EOD marks
  struct PriceBook { std::unordered_map<std::string, double> last_px; };

  // replay the file; return recomputed account/pnl from fills + marks
  static std::optional<ReplayResult> replay_file(const std::string& file_path,
                                                 const std::string& run_id_expect = "");
private:
  static bool apply_bar_(PriceBook& pb, const std::string& instrument, const AuditBar& b);
  static void mark_to_market_(const PriceBook& pb, ReplayResult& rr);
  static void apply_fill_(ReplayResult& rr, const std::string& inst, double px, double qty, double fees, Side side);
};

} // namespace sentio
```

## 📄 **FILE 7 of 180**: include/sentio/audit_interface.hpp

**File Information**:
- **Path**: `include/sentio/audit_interface.hpp`

- **Size**: 53 lines
- **Modified**: 2025-09-11 16:46:18

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <cstdint>
#include "sentio/signal_diag.hpp"

namespace sentio {

// Forward declarations
struct Bar;
struct AccountState;
enum class SigType : uint8_t;
enum class Side : uint8_t;

// Common interface for all audit recorders
class IAuditRecorder {
public:
    virtual ~IAuditRecorder() = default;
    
    // Run lifecycle events
    virtual void event_run_start(std::int64_t ts, const std::string& meta) = 0;
    virtual void event_run_end(std::int64_t ts, const std::string& meta) = 0;
    
    // Market data events
    virtual void event_bar(std::int64_t ts, const std::string& inst, double open, double high, double low, double close, double volume) = 0;
    
    // Signal events
    virtual void event_signal(std::int64_t ts, const std::string& base, SigType t, double conf) = 0;
    virtual void event_signal_ex(std::int64_t ts, const std::string& base, SigType t, double conf, const std::string& chain_id) = 0;
    
    // Trading events
    virtual void event_route(std::int64_t ts, const std::string& base, const std::string& inst, double tw) = 0;
    virtual void event_route_ex(std::int64_t ts, const std::string& base, const std::string& inst, double tw, const std::string& chain_id) = 0;
    virtual void event_order(std::int64_t ts, const std::string& inst, Side side, double qty, double limit_px) = 0;
    virtual void event_order_ex(std::int64_t ts, const std::string& inst, Side side, double qty, double limit_px, const std::string& chain_id) = 0;
    virtual void event_fill(std::int64_t ts, const std::string& inst, double price, double qty, double fees, Side side) = 0;
    virtual void event_fill_ex(std::int64_t ts, const std::string& inst, double price, double qty, double fees, Side side, 
                              double realized_pnl_delta, double equity_after, double position_after, const std::string& chain_id) = 0;
    
    // Portfolio events
    virtual void event_snapshot(std::int64_t ts, const AccountState& a) = 0;
    
    // Metric events
    virtual void event_metric(std::int64_t ts, const std::string& key, double val) = 0;
    
    // Signal diagnostics events
    virtual void event_signal_diag(std::int64_t ts, const std::string& strategy_name, 
                                  const SignalDiag& diag) = 0;
    virtual void event_signal_drop(std::int64_t ts, const std::string& strategy_name, 
                                  const std::string& symbol, DropReason reason, 
                                  const std::string& chain_id, const std::string& note = "") = 0;
};

} // namespace sentio

```

## 📄 **FILE 8 of 180**: include/sentio/base_strategy.hpp

**File Information**:
- **Path**: `include/sentio/base_strategy.hpp`

- **Size**: 167 lines
- **Modified**: 2025-09-18 11:18:14

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

## 📄 **FILE 9 of 180**: include/sentio/binio.hpp

**File Information**:
- **Path**: `include/sentio/binio.hpp`

- **Size**: 68 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <cstdio>
#include <vector>
#include <string>
#include "core.hpp"

namespace sentio {

inline void save_bin(const std::string& path, const std::vector<Bar>& v) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return;
    
    uint64_t n = v.size();
    std::fwrite(&n, sizeof(n), 1, fp);
    
    for (const auto& bar : v) {
        // Write string length and data
        uint32_t str_len = bar.ts_utc.length();
        std::fwrite(&str_len, sizeof(str_len), 1, fp);
        std::fwrite(bar.ts_utc.c_str(), 1, str_len, fp);
        
        // Write other fields
        std::fwrite(&bar.ts_utc_epoch, sizeof(bar.ts_utc_epoch), 1, fp);
        std::fwrite(&bar.open, sizeof(bar.open), 1, fp);
        std::fwrite(&bar.high, sizeof(bar.high), 1, fp);
        std::fwrite(&bar.low, sizeof(bar.low), 1, fp);
        std::fwrite(&bar.close, sizeof(bar.close), 1, fp);
        std::fwrite(&bar.volume, sizeof(bar.volume), 1, fp);
    }
    std::fclose(fp);
}

inline std::vector<Bar> load_bin(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return {};
    
    uint64_t n = 0; 
    std::fread(&n, sizeof(n), 1, fp);
    std::vector<Bar> v;
    v.reserve(n);
    
    for (uint64_t i = 0; i < n; ++i) {
        Bar bar;
        
        // Read string length and data
        uint32_t str_len = 0;
        std::fread(&str_len, sizeof(str_len), 1, fp);
        if (str_len > 0) {
            std::vector<char> str_data(str_len);
            std::fread(str_data.data(), 1, str_len, fp);
            bar.ts_utc = std::string(str_data.data(), str_len);
        }
        
        // Read other fields
        std::fread(&bar.ts_utc_epoch, sizeof(bar.ts_utc_epoch), 1, fp);
        std::fread(&bar.open, sizeof(bar.open), 1, fp);
        std::fread(&bar.high, sizeof(bar.high), 1, fp);
        std::fread(&bar.low, sizeof(bar.low), 1, fp);
        std::fread(&bar.close, sizeof(bar.close), 1, fp);
        std::fread(&bar.volume, sizeof(bar.volume), 1, fp);
        
        v.push_back(bar);
    }
    std::fclose(fp);
    return v;
}

} // namespace sentio
```

## 📄 **FILE 10 of 180**: include/sentio/bo.hpp

**File Information**:
- **Path**: `include/sentio/bo.hpp`

- **Size**: 384 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
// bo.hpp — minimal, solid C++20 Bayesian Optimization (GP + EI)
// No external deps. Deterministic. Safe. Box bounds + integers + batch ask().

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <functional>
#include <optional>
#include <limits>
#include <cmath>
#include <cassert>
#include <chrono>
#include <string>

// ----------------------------- Utilities -----------------------------
struct BOBounds {
  std::vector<double> lo, hi; // size = D
  bool valid() const { return !lo.empty() && lo.size()==hi.size(); }
  int dim() const { return (int)lo.size(); }
  inline double clamp_unit(double u, int d) const {
    return std::min(1.0, std::max(0.0, u));
  }
  inline double to_real(double u, int d) const {
    u = clamp_unit(u,d);
    return lo[d] + u * (hi[d] - lo[d]);
  }
  inline double to_unit(double x, int d) const {
    const double L = lo[d], H = hi[d];
    if (H<=L) return 0.0;
    double u = (x - L) / (H - L);
    return std::min(1.0, std::max(0.0, u));
  }
};

// Parameter kinds (continuous or integer). If INT, we round to nearest.
enum class ParamKind : uint8_t { CONT, INT };

struct BOOpts {
  int init_design = 16;          // initial random/space-filling points
  int cand_pool   = 2048;        // candidate samples for EI maximization
  int batch_q     = 1;           // q-EI via constant liar
  double jitter   = 1e-10;       // numerical jitter for Cholesky
  double ei_xi    = 0.01;        // EI exploration parameter
  bool  maximize  = true;        // true: maximize objective (default)
  uint64_t seed   = 42;          // deterministic rand seed
  bool  verbose   = false;
};

// ----------------------------- Tiny matrix helpers -----------------------------
// Row-major dense matrix with minimal ops (just what we need for GP Cholesky).
struct Mat {
  int n=0, m=0;                  // n rows, m cols
  std::vector<double> a;         // size n*m
  Mat()=default;
  Mat(int n_, int m_) : n(n_), m(m_), a((size_t)n_*(size_t)m_, 0.0) {}
  inline double& operator()(int i, int j){ return a[(size_t)i*(size_t)m + j]; }
  inline double  operator()(int i, int j) const { return a[(size_t)i*(size_t)m + j]; }
};

// Cholesky decomposition A = L L^T in-place into L (lower). Returns false if fails.
inline bool cholesky(Mat& A, double jitter=1e-10){
  // A must be square, symmetric, positive definite (we'll add jitter on diag).
  assert(A.n == A.m);
  const int n = A.n;
  for (int i=0;i<n;i++) A(i,i) += jitter;

  for (int i=0;i<n;i++){
    for (int j=0;j<=i;j++){
      double sum = A(i,j);
      for (int k=0;k<j;k++) sum -= A(i,k)*A(j,k);
      if (i==j){
        if (sum <= 0.0) return false;
        A(i,j) = std::sqrt(sum);
      } else {
        A(i,j) = sum / A(j,j);
      }
    }
    for (int j=i+1;j<n;j++) A(i,j)=0.0; // zero upper for clarity
  }
  return true;
}

// Solve L y = b (forward)
inline void trisolve_lower(const Mat& L, std::vector<double>& y){
  const int n=L.n;
  for (int i=0;i<n;i++){
    double s = y[i];
    for (int k=0;k<i;k++) s -= L(i,k)*y[k];
    y[i] = s / L(i,i);
  }
}
// Solve L^T x = y (backward)
inline void trisolve_upperT(const Mat& L, std::vector<double>& x){
  const int n=L.n;
  for (int i=n-1;i>=0;i--){
    double s = x[i];
    for (int k=i+1;k<n;k++) s -= L(k,i)*x[k];
    x[i] = s / L(i,i);
  }
}

// ----------------------------- Gaussian Process (RBF-ARD) -----------------------------
struct GP {
  // Hyperparams (on unit cube): signal^2, noise^2, lengthscales (per-dim)
  double sigma_f2 = 1.0;
  double sigma_n2 = 1e-6;
  std::vector<double> ell; // size D, lengthscales

  // Data (unit cube)
  std::vector<std::vector<double>> X; // N x D
  std::vector<double> y;              // N (centered)
  double y_mean = 0.0;

  // Factorization
  Mat L;                 // Cholesky of K = Kf + sigma_n2 I
  std::vector<double> alpha; // (K)^-1 y

  static inline double sqdist_ard(const std::vector<double>& a, const std::vector<double>& b,
                                  const std::vector<double>& ell){
    double s=0.0;
    for (size_t d=0; d<ell.size(); ++d){
      const double z = (a[d]-b[d]) / std::max(1e-12, ell[d]);
      s += z*z;
    }
    return s;
  }

  inline double kf(const std::vector<double>& a, const std::vector<double>& b) const {
    const double s2 = sqdist_ard(a,b,ell);
    return sigma_f2 * std::exp(-0.5 * s2);
  }

  void fit(const std::vector<std::vector<double>>& X_unit,
           const std::vector<double>& y_raw,
           double jitter=1e-10)
  {
    X = X_unit;
    const int N = (int)X.size();
    y_mean = 0.0;
    for (double v: y_raw) y_mean += v;
    y_mean /= std::max(1, N);
    y.resize(N);
    for (int i=0;i<N;i++) y[i] = y_raw[i] - y_mean;

    // Build K
    Mat K(N,N);
    for (int i=0;i<N;i++){
      for (int j=0;j<=i;j++){
        double kij = kf(X[i], X[j]);
        if (i==j) kij += sigma_n2;
        K(i,j) = K(j,i) = kij;
      }
    }
    // Chol
    L = K;
    if (!cholesky(L, jitter)) {
      // increase jitter progressively if needed
      double j = std::max(jitter, 1e-12);
      bool ok=false;
      for (int t=0;t<6 && !ok; ++t) { L = K; ok = cholesky(L, j); j *= 10.0; }
      if (!ok) throw std::runtime_error("GP Cholesky failed (matrix not PD)");
    }
    // alpha = K^{-1} y  via L
    alpha = y;
    trisolve_lower(L, alpha);
    trisolve_upperT(L, alpha);
  }

  // Predictive mean/variance at x (unit cube)
  inline std::pair<double,double> predict(const std::vector<double>& x) const {
    const int N = (int)X.size();
    std::vector<double> k(N);
    for (int i=0;i<N;i++) k[i] = kf(x, X[i]);
    // mean = k^T alpha + y_mean
    double mu = std::inner_product(k.begin(), k.end(), alpha.begin(), 0.0) + y_mean;
    // v = L^{-1} k
    std::vector<double> v = k;
    trisolve_lower(L, v);
    double kxx = sigma_f2; // k(x,x) for RBF with same params is sigma_f2
    double var = kxx - std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    if (var < 1e-18) var = 1e-18;
    return {mu, var};
  }
};

// ----------------------------- Random/Sampling helpers -----------------------------
struct RNG {
  std::mt19937_64 g;
  explicit RNG(uint64_t seed): g(seed) {}
  double uni(){ return std::uniform_real_distribution<double>(0.0,1.0)(g); }
  int randint(int lo, int hi){ return std::uniform_int_distribution<int>(lo,hi)(g); }
};

// Jittered Latin hypercube on unit cube
inline std::vector<std::vector<double>> latin_hypercube(int n, int D, RNG& rng){
  std::vector<std::vector<double>> X(n, std::vector<double>(D,0.0));
  for (int d=0; d<D; ++d){
    std::vector<double> slots(n);
    for (int i=0;i<n;i++) slots[i] = (i + rng.uni()) / n;
    std::shuffle(slots.begin(), slots.end(), rng.g);
    for (int i=0;i<n;i++) X[i][d] = std::min(1.0, std::max(0.0, slots[i]));
  }
  return X;
}

// Round integer dims to nearest feasible grid-point after scaling
inline void apply_kinds_inplace(std::vector<double>& u, const std::vector<ParamKind>& kinds,
                                const BOBounds& B){
  for (int d=0; d<(int)u.size(); ++d){
    if (kinds[d] == ParamKind::INT){
      // snap in real space to integer, then rescale back to unit
      const double x = B.to_real(u[d], d);
      const double xi = std::round(x);
      u[d] = B.to_unit(xi, d);
    }
  }
}

// ----------------------------- Acquisition: Expected Improvement -----------------------------
inline double norm_pdf(double z){ static const double inv_sqrt2pi = 0.3989422804014327; return inv_sqrt2pi*std::exp(-0.5*z*z); }
inline double norm_cdf(double z){
  return 0.5 * std::erfc(-z/std::sqrt(2.0));
}
// EI for maximization: EI = (mu - best - xi) Phi(z) + sigma phi(z), z = (mu - best - xi)/sigma
inline double expected_improvement(double mu, double var, double best, double xi){
  const double sigma = std::sqrt(std::max(1e-18, var));
  const double diff  = mu - best - xi;
  if (sigma <= 1e-12) return std::max(0.0, diff);
  const double z = diff / sigma;
  return diff * norm_cdf(z) + sigma * norm_pdf(z);
}

// ----------------------------- Bayesian Optimizer -----------------------------
struct BO {
  BOBounds bounds;
  std::vector<ParamKind> kinds; // size D
  BOOpts opt;

  // data (real-space for API; internally store unit-cube for GP)
  std::vector<std::vector<double>> X_real; // N x D
  std::vector<std::vector<double>> X;      // N x D (unit)
  std::vector<double> y;                   // N
  GP gp;
  RNG rng;

  BO(const BOBounds& B, const std::vector<ParamKind>& kinds_, BOOpts o)
  : bounds(B), kinds(kinds_), opt(o), gp(), rng(o.seed)
  {
    assert(bounds.valid());
    assert((int)kinds.size() == bounds.dim());
    // init default GP hyperparams
    gp.ell.assign(bounds.dim(), 0.2); // medium lengthscale on unit cube
    gp.sigma_f2 = 1.0;
    gp.sigma_n2 = 1e-6;
  }

  int dim() const { return bounds.dim(); }
  int size() const { return (int)X.size(); }

  void clear(){
    X_real.clear(); X.clear(); y.clear();
  }

  // Append observation
  void tell(const std::vector<double>& xr, double val){
    assert((int)xr.size() == dim());
    X_real.push_back(xr);
    std::vector<double> u(dim());
    for (int d=0; d<dim(); ++d) u[d] = bounds.to_unit(xr[d], d);
    // snap integers
    apply_kinds_inplace(u, kinds, bounds);
    X.push_back(u);
    y.push_back(val);
  }

  // Fit GP (simple hyperparam heuristics; robust & fast)
  void fit(){
    if (X.empty()) return;
    // set GP hyperparams from y stats
    double m=0, s2=0;
    for (double v: y) m += v; m /= (double)y.size();
    for (double v: y) s2 += (v-m)*(v-m);
    s2 /= std::max(1, (int)y.size()-1);
    gp.sigma_f2 = std::max(1e-12, s2);
    gp.sigma_n2 = std::max(1e-9 * gp.sigma_f2, 1e-10);
    // modest ARD scaling: initialize lengthscales to 0.2; (optionally: scale by output sensitivity)
    for (double& e: gp.ell) e = 0.2;
    gp.fit(X, y, opt.jitter);
  }

  // Generate initial design if dataset is empty/small
  void ensure_init_design(){
    const int need = std::max(0, opt.init_design - (int)X.size());
    if (need <= 0) return;
    auto U = latin_hypercube(need, dim(), rng);
    for (auto& u : U){
      apply_kinds_inplace(u, kinds, bounds);
      std::vector<double> xr(dim());
      for (int d=0; d<dim(); ++d) xr[d] = bounds.to_real(u[d], d);
      // placeholder y (user will evaluate and call tell)
      X.push_back(u);
      X_real.push_back(xr);
      y.push_back(std::numeric_limits<double>::quiet_NaN()); // mark unevaluated
    }
  }

  // Ask for q new locations (real-space). Uses constant liar on current model (no new evals yet).
  std::vector<std::vector<double>> ask(int q=1){
    if (q<=0) return {};
    // If we have NaNs from ensure_init_design, return those first (ask-user-to-evaluate)
    std::vector<std::vector<double>> out;
    for (int i=0;i<(int)X.size() && (int)out.size()<q; ++i){
      if (!std::isfinite(y[i])) out.push_back(X_real[i]);
    }
    if ((int)out.size() == q) return out;

    // Ensure we can build a GP on finished points
    // Build filtered dataset of (finite) y's
    std::vector<std::vector<double>> Xf;
    std::vector<double> yf;
    Xf.reserve(X.size()); yf.reserve(y.size());
    for (int i=0;i<(int)X.size(); ++i){
      if (std::isfinite(y[i])) { Xf.push_back(X[i]); yf.push_back(y[i]); }
    }
    if (Xf.size() >= 2) {
      gp.fit(Xf, yf, opt.jitter);
    } else {
      // not enough data to fit GP: just random suggest
      out = random_candidates(q);
      return out;
    }

    const double y_best = opt.maximize ? *std::max_element(yf.begin(), yf.end())
                                       : *std::min_element(yf.begin(), yf.end());

    // batch selection with constant liar (for maximization, lie = y_best)
    std::vector<std::vector<double>> X_aug = Xf;
    std::vector<double> y_aug = yf;

    for (int pick=0; pick<q; ++pick){
      // pool
      double best_ei = -1.0;
      std::vector<double> best_u(dim());
      for (int c=0; c<opt.cand_pool; ++c){
        std::vector<double> u(dim());
        for (int d=0; d<dim(); ++d) u[d] = rng.uni();
        apply_kinds_inplace(u, kinds, bounds);
        // score EI on augmented model (approximate by reusing gp; small bias is fine)
        // NOTE: we approximate: use current gp (no retrain) — fast & works well in practice
        auto [mu, var] = gp.predict(u);
        double ei = opt.maximize ? expected_improvement(mu, var, y_best, opt.ei_xi)
                                 : expected_improvement(-mu, var, -y_best, opt.ei_xi);
        if (ei > best_ei){ best_ei = ei; best_u = u; }
      }
      // map to real, append
      std::vector<double> xr(dim());
      for (int d=0; d<dim(); ++d) xr[d] = bounds.to_real(best_u[d], d);
      out.push_back(xr);

      // constant liar update (no refit for speed; optional: refit small)
      X_aug.push_back(best_u);
      y_aug.push_back(y_best); // lie
      // (Optionally refit gp with augmented data each pick for sharper q-EI; omitted for speed)
    }
    return out;
  }

  // Helper: random suggestions on real-space
  std::vector<std::vector<double>> random_candidates(int q){
    std::vector<std::vector<double>> out;
    out.reserve(q);
    for (int i=0;i<q;++i){
      std::vector<double> u(dim());
      for (int d=0; d<dim(); ++d) u[d] = rng.uni();
      apply_kinds_inplace(u, kinds, bounds);
      std::vector<double> xr(dim());
      for (int d=0; d<dim(); ++d) xr[d] = bounds.to_real(u[d], d);
      out.push_back(xr);
    }
    return out;
  }
};
```

## 📄 **FILE 11 of 180**: include/sentio/bollinger.hpp

**File Information**:
- **Path**: `include/sentio/bollinger.hpp`

- **Size**: 22 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "rolling_stats.hpp"

namespace sentio {

struct Bollinger {
  RollingMeanVar mv;
  double k;
  double eps; // volatility floor to avoid zero-width bands

  Bollinger(int w, double k_=2.0, double eps_=1e-9) : mv(w), k(k_), eps(eps_) {}

  inline void step(double close, double& mid, double& lo, double& hi, double& sd_out){
    auto [m, var] = mv.push(close);
    double sd = std::sqrt(std::max(var, 0.0));
    if (sd < eps) sd = eps;         // <- floor
    mid = m; lo = m - k*sd; hi = m + k*sd;
    sd_out = sd;
  }
};

} // namespace sentio
```

## 📄 **FILE 12 of 180**: include/sentio/canonical_evaluation.hpp

**File Information**:
- **Path**: `include/sentio/canonical_evaluation.hpp`

- **Size**: 151 lines
- **Modified**: 2025-09-16 13:56:24

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace sentio {

/**
 * Canonical Evaluation System for Trading Strategy Performance
 * 
 * This system replaces ambiguous calendar-based time measurements with deterministic
 * bar-based evaluation units called "Trading Blocks". This eliminates discrepancies
 * between strattest and audit systems by creating a single source of truth for
 * performance measurement that is immune to weekends, holidays, and variable trading hours.
 */

/**
 * Configuration for Trading Block-based evaluation
 * Replaces duration strings like "2w" with deterministic Trading Block counts
 * 
 * Standard Units:
 * - 1 Trading Block (TB) = 480 bars ≈ 8 hours of trading
 * - 10 TB ≈ 10 trading days (default for quick tests)  
 * - 20 TB ≈ 1 month of trading (standard benchmark)
 */
struct TradingBlockConfig {
    int block_size = 480;     // Number of bars per Trading Block (TB) - 8 hours
    int num_blocks = 10;      // Number of Trading Blocks to test (default: 10 TB)
    
    // Total bars this configuration will process
    int total_bars() const { return block_size * num_blocks; }
    
    // Helper methods for common configurations
    static TradingBlockConfig quick_test() { return {480, 10}; }      // 10 TB ≈ 2 weeks
    static TradingBlockConfig standard_monthly() { return {480, 20}; } // 20 TB ≈ 1 month
    static TradingBlockConfig extended_test() { return {480, 60}; }    // 60 TB ≈ 3 months
};

/**
 * Performance metrics for a single Trading Block
 * Each block represents exactly block_size consecutive market data bars
 */
struct BlockResult {
    int block_index;                    // 0-based block number
    double return_per_block;            // Total compounded return for this block (RPB)
    double sharpe_ratio;                // Sharpe ratio for this block only
    double max_drawdown_pct;            // Maximum drawdown within this block
    int fills;                          // Number of fill events in this block
    double starting_equity;             // Equity at block start
    double ending_equity;               // Equity at block end
    
    // Timestamps for audit verification - defines exact bar range
    std::int64_t start_ts_ms;           // Timestamp of first bar in block
    std::int64_t end_ts_ms;             // Timestamp of last bar in block
    
    // Additional metrics
    double win_rate_pct = 0.0;          // Win rate for trades in this block
    double avg_trade_pnl = 0.0;         // Average P&L per trade in this block
    int winning_trades = 0;             // Number of winning trades
    int losing_trades = 0;              // Number of losing trades
};

/**
 * Final canonical performance report aggregating all blocks
 * This is the single source of truth for strategy evaluation
 */
struct CanonicalReport {
    TradingBlockConfig config;                  // Configuration used for this test
    std::vector<BlockResult> block_results;     // Results for each block
    
    // Aggregated statistics across all blocks
    double mean_rpb = 0.0;                      // Mean Return Per Block
    double stdev_rpb = 0.0;                     // Standard deviation of RPB
    double annualized_return_on_block = 0.0;    // ARB - annualized equivalent
    double aggregate_sharpe = 0.0;              // Overall Sharpe ratio
    double max_drawdown_across_blocks = 0.0;    // Maximum drawdown across all blocks
    
    // Trading statistics
    int total_fills = 0;                        // Total fills across all blocks
    double avg_fills_per_block = 0.0;           // Average fills per block
    double total_return_pct = 0.0;              // Total compounded return across all blocks
    double consistency_score = 0.0;             // Metric for performance consistency (lower stdev_rpb is better)
    
    // Metadata
    std::string strategy_name;                  // Strategy that was tested
    std::string dataset_source;                 // Source of market data used
    std::int64_t test_start_ts_ms = 0;          // Timestamp when test started
    std::int64_t test_end_ts_ms = 0;            // Timestamp when test ended
    int total_bars_processed = 0;               // Actual number of bars processed
    
    // Helper methods
    bool is_valid() const { return !block_results.empty(); }
    int successful_blocks() const { return static_cast<int>(block_results.size()); }
    double completion_rate() const { 
        return config.num_blocks > 0 ? static_cast<double>(successful_blocks()) / config.num_blocks : 0.0; 
    }
};

/**
 * Canonical Metrics Calculator
 * Single source of truth for all block-level performance calculations
 * Used by both strattest and audit systems to ensure consistency
 */
class CanonicalEvaluator {
public:
    /**
     * Calculate performance metrics for a single block of trading data
     * This is the atomic unit of performance measurement
     */
    static BlockResult calculate_block_metrics(
        const std::vector<std::pair<std::string, double>>& equity_curve,
        int block_index,
        double starting_equity,
        int fills_count,
        std::int64_t start_ts_ms,
        std::int64_t end_ts_ms
    );
    
    /**
     * Aggregate block results into a final canonical report
     * Calculates mean, standard deviation, and annualized metrics
     */
    static CanonicalReport aggregate_block_results(
        const TradingBlockConfig& config,
        const std::vector<BlockResult>& block_results,
        const std::string& strategy_name = "",
        const std::string& dataset_source = ""
    );
    
    /**
     * Calculate Annualized Return on Block (ARB) from mean RPB
     * Uses compound interest formula with trading calendar assumptions
     * 
     * With 480-bar Trading Blocks:
     * - 98,280 bars/year ÷ 480 bars/TB = ~204.75 TB/year
     * - ARB = ((1 + mean_RPB) ^ 204.75) - 1
     */
    static double calculate_annualized_return(
        double mean_rpb, 
        int block_size = 480,
        int bars_per_trading_year = 98280  // 252 days * 390 bars/day for QQQ
    );
    
    /**
     * Calculate consistency score from block results
     * Lower scores indicate more consistent performance
     */
    static double calculate_consistency_score(const std::vector<BlockResult>& block_results);
};

} // namespace sentio

```

## 📄 **FILE 13 of 180**: include/sentio/canonical_metrics.hpp

**File Information**:
- **Path**: `include/sentio/canonical_metrics.hpp`

- **Size**: 98 lines
- **Modified**: 2025-09-15 22:39:24

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <cmath>

namespace sentio {

/**
 * Canonical Metrics Calculator
 * 
 * This is the single source of truth for all performance metrics calculations.
 * All systems (strattest, audit summarize, position-history) must use these functions
 * to ensure consistency across the entire platform.
 */
class CanonicalMetrics {
public:
    struct PerformanceMetrics {
        double total_return_pct;
        double monthly_projected_return_pct;
        double sharpe_ratio;
        double max_drawdown_pct;
        double daily_trades;
        double win_rate_pct;
        int total_trades;
        int trading_days;
        double starting_capital;
        double final_equity;
        double total_pnl;
    };

    /**
     * Calculate all performance metrics from equity curve
     * This is the canonical method that all systems should use
     */
    static PerformanceMetrics calculate_from_equity_curve(
        const std::vector<std::pair<std::string, double>>& equity_curve,
        double starting_capital = 100000.0,
        int total_trades = 0
    );

    /**
     * Calculate Monthly Projected Return (MPR) from daily returns
     * Uses geometric mean for accurate compounding
     */
    static double calculate_mpr_from_daily_returns(const std::vector<double>& daily_returns);

    /**
     * Calculate MPR from total return and trading days
     * Alternative method when daily returns are not available
     */
    static double calculate_mpr_from_total_return(double total_return, int trading_days);

    /**
     * Calculate Sharpe ratio from daily returns
     * Assumes risk-free rate of 0 for simplicity
     */
    static double calculate_sharpe_ratio(const std::vector<double>& daily_returns);

    /**
     * Calculate maximum drawdown from equity curve
     */
    static double calculate_max_drawdown(const std::vector<double>& equity_values);

    /**
     * Extract daily returns from equity curve
     * Handles timestamp parsing and deduplication
     */
    static std::vector<double> extract_daily_returns(
        const std::vector<std::pair<std::string, double>>& equity_curve
    );

    /**
     * Validate metrics consistency between different calculation methods
     * Returns true if all metrics are within acceptable tolerance
     */
    static bool validate_metrics_consistency(
        const PerformanceMetrics& metrics1,
        const PerformanceMetrics& metrics2,
        double tolerance_pct = 1.0  // 1% tolerance by default
    );

    /**
     * Print detailed metrics comparison for debugging
     */
    static void print_metrics_comparison(
        const PerformanceMetrics& canonical,
        const PerformanceMetrics& audit,
        const PerformanceMetrics& strattest,
        const std::string& run_id
    );

private:
    static constexpr double TRADING_DAYS_PER_MONTH = 21.0;
    static constexpr double TRADING_DAYS_PER_YEAR = 252.0;
    static constexpr double TOLERANCE_EPSILON = 1e-6;
};

} // namespace sentio

```

## 📄 **FILE 14 of 180**: include/sentio/cli_helpers.hpp

**File Information**:
- **Path**: `include/sentio/cli_helpers.hpp`

- **Size**: 147 lines
- **Modified**: 2025-09-17 13:20:14

- **Type**: .hpp

```text
#pragma once

#include <string>
#include <vector>
#include <map>
#include <optional>

namespace sentio {

/**
 * CLI Argument Parsing Helpers
 * 
 * Provides standardized argument parsing for canonical sentio_cli interface
 */
class CLIHelpers {
public:
    struct ParsedArgs {
        std::string command;
        std::vector<std::string> positional_args;
        std::map<std::string, std::string> options;
        std::map<std::string, bool> flags;
        bool help_requested = false;
        bool verbose = false;
    };
    
    /**
     * Parse command line arguments into structured format
     */
    static ParsedArgs parse_arguments(int argc, char* argv[]);
    
    /**
     * Get string option with default value
     */
    static std::string get_option(const ParsedArgs& args, const std::string& key, 
                                 const std::string& default_value = "");
    
    /**
     * Get integer option with default value
     */
    static int get_int_option(const ParsedArgs& args, const std::string& key, int default_value = 0);
    
    /**
     * Get double option with default value
     */
    static double get_double_option(const ParsedArgs& args, const std::string& key, double default_value = 0.0);
    
    /**
     * Get boolean flag
     */
    static bool get_flag(const ParsedArgs& args, const std::string& key);
    
    /**
     * Parse period string (e.g., "3y", "6m", "2w", "5d", "4h") to days
     */
    static int parse_period_to_days(const std::string& period);
    
    /**
     * Parse period string to minutes
     */
    static int parse_period_to_minutes(const std::string& period);
    
    /**
     * Validate required positional arguments
     */
    static bool validate_required_args(const ParsedArgs& args, int min_required, 
                                      const std::string& usage_msg = "");
    
    /**
     * Print standardized help message
     */
    static void print_help(const std::string& command, const std::string& usage, 
                          const std::vector<std::string>& options,
                          const std::vector<std::string>& examples);
    
    /**
     * Print error message with usage hint
     */
    static void print_error(const std::string& error_msg, const std::string& usage_hint = "");
    
    /**
     * Validate symbol format
     */
    static bool is_valid_symbol(const std::string& symbol);
    
    /**
     * Validate strategy name format
     */
    static bool is_valid_strategy_name(const std::string& strategy_name);
    
    /**
     * Get available strategies list
     */
    static std::vector<std::string> get_available_strategies();
    
    /**
     * Validate command options against allowed options
     */
    static bool validate_options(const ParsedArgs& args, const std::string& command,
                                const std::vector<std::string>& allowed_options,
                                const std::vector<std::string>& allowed_flags);
    
    /**
     * Print unknown option error with suggestions
     */
    static void print_unknown_option_error(const std::string& unknown_option,
                                          const std::string& command,
                                          const std::vector<std::string>& allowed_options,
                                          const std::vector<std::string>& allowed_flags);
    
    /**
     * Format duration for display (e.g., 1800 minutes -> "30h" or "1.25d")
     */
    static std::string format_duration(int minutes);
    
    /**
     * Parse comma-separated list
     */
    static std::vector<std::string> parse_list(const std::string& list_str, char delimiter = ',');
    
    /**
     * Validate file path exists
     */
    static bool file_exists(const std::string& filepath);
    
    /**
     * Get default data file for symbol
     */
    static std::string get_default_data_file(const std::string& symbol, const std::string& suffix = "_NH.csv");

private:
    /**
     * Normalize option key (remove leading dashes, convert to lowercase)
     */
    static std::string normalize_option_key(const std::string& key);
    
    /**
     * Check if argument is an option (starts with -)
     */
    static bool is_option(const std::string& arg);
    
    /**
     * Check if argument is a flag (starts with -- and has no value)
     */
    static bool is_flag(const std::string& arg);
};

} // namespace sentio

```

## 📄 **FILE 15 of 180**: include/sentio/core.hpp

**File Information**:
- **Path**: `include/sentio/core.hpp`

- **Size**: 74 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath> // For std::sqrt

namespace sentio {

// Forward declarations
struct SymbolTable;

struct Bar {
    std::string ts_utc;
    int64_t ts_utc_epoch;
    double open, high, low, close;
    uint64_t volume;
};

// **MODIFIED**: This struct now holds a vector of Positions, indexed by symbol ID for performance.
struct Position { 
    double qty = 0.0; 
    double avg_price = 0.0; 
};

struct Portfolio {
    double cash = 100000.0;
    std::vector<Position> positions; // Indexed by symbol ID

    Portfolio() = default;
    explicit Portfolio(size_t num_symbols) : positions(num_symbols) {}
};

// **MODIFIED**: Vector-based functions are now the primary way to manage the portfolio.
inline void apply_fill(Portfolio& pf, int sid, double qty_delta, double price) {
    if (sid < 0 || static_cast<size_t>(sid) >= pf.positions.size()) {
        return; // Invalid symbol ID
    }
    
    pf.cash -= qty_delta * price;
    auto& pos = pf.positions[sid];
    
    double new_qty = pos.qty + qty_delta;
    if (std::abs(new_qty) < 1e-9) { // Position is closed
        pos.qty = 0.0;
        pos.avg_price = 0.0;
    } else {
        if (pos.qty * new_qty >= 0) { // Increasing position or opening a new one
            pos.avg_price = (pos.avg_price * pos.qty + price * qty_delta) / new_qty;
        }
        // If flipping from long to short or vice-versa, the new avg_price is just the fill price.
        else if (pos.qty * qty_delta < 0) {
             pos.avg_price = price;
        }
        pos.qty = new_qty;
    }
}

inline double equity_mark_to_market(const Portfolio& pf, const std::vector<double>& last_prices) {
    double eq = pf.cash;
    for (size_t sid = 0; sid < pf.positions.size(); ++sid) {
        if (std::abs(pf.positions[sid].qty) > 0.0 && sid < last_prices.size()) {
            eq += pf.positions[sid].qty * last_prices[sid];
        }
    }
    return eq;
}


// **REMOVED**: Old, simplistic Direction and StratSignal types are now deprecated.

} // namespace sentio
```

## 📄 **FILE 16 of 180**: include/sentio/core/bar.hpp

**File Information**:
- **Path**: `include/sentio/core/bar.hpp`

- **Size**: 9 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <cstdint>

namespace sentio {
struct Bar {
  std::int64_t ts_epoch_us{0};
  double open{0}, high{0}, low{0}, close{0}, volume{0};
};
}

```

## 📄 **FILE 17 of 180**: include/sentio/cost_model.hpp

**File Information**:
- **Path**: `include/sentio/cost_model.hpp`

- **Size**: 120 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <cmath>

namespace sentio {

// Alpaca Trading Cost Model
struct AlpacaCostModel {
  // Commission structure (Alpaca is commission-free for stocks/ETFs)
  static constexpr double commission_per_share = 0.0;
  static constexpr double min_commission = 0.0;
  
  // SEC fees (for sells only)
  static constexpr double sec_fee_rate = 0.0000278; // $0.0278 per $1000 of principal
  
  // FINRA Trading Activity Fee (TAF) - for sells only
  static constexpr double taf_per_share = 0.000145; // $0.000145 per share (max $7.27 per trade)
  static constexpr double taf_max_per_trade = 7.27;
  
  // Slippage model based on market impact
  struct SlippageParams {
    double base_slippage_bps = 1.0;    // Base 1 bps slippage
    double volume_impact_factor = 0.5;  // Additional slippage based on volume
    double volatility_factor = 0.3;     // Additional slippage based on volatility
    double max_slippage_bps = 10.0;     // Cap at 10 bps
  };
  
  static SlippageParams default_slippage;
  
  // Calculate total transaction costs for a trade
  static double calculate_fees([[maybe_unused]] const std::string& symbol, 
                              double quantity, 
                              double price, 
                              bool is_sell) {
    double notional = std::abs(quantity) * price;
    double total_fees = 0.0;
    
    // Commission (free for Alpaca)
    total_fees += commission_per_share * std::abs(quantity);
    total_fees = std::max(total_fees, min_commission);
    
    if (is_sell) {
      // SEC fees (sells only)
      total_fees += notional * sec_fee_rate;
      
      // FINRA TAF (sells only)
      double taf = std::abs(quantity) * taf_per_share;
      total_fees += std::min(taf, taf_max_per_trade);
    }
    
    return total_fees;
  }
  
  // Calculate slippage based on trade characteristics
  static double calculate_slippage_bps(double quantity,
                                      double price, 
                                      double avg_volume,
                                      double volatility,
                                      const SlippageParams& params = default_slippage) {
    double notional = std::abs(quantity) * price;
    
    // Base slippage
    double slippage_bps = params.base_slippage_bps;
    
    // Volume impact (higher for larger trades relative to average volume)
    if (avg_volume > 0) {
      double volume_ratio = notional / (avg_volume * price);
      slippage_bps += params.volume_impact_factor * std::sqrt(volume_ratio) * 100; // Convert to bps
    }
    
    // Volatility impact
    slippage_bps += params.volatility_factor * volatility * 10000; // Convert annual vol to bps
    
    // Cap the slippage
    return std::min(slippage_bps, params.max_slippage_bps);
  }
  
  // Apply slippage to execution price
  static double apply_slippage(double market_price, 
                              double slippage_bps, 
                              bool is_buy) {
    double slippage_factor = slippage_bps / 10000.0; // Convert bps to decimal
    
    if (is_buy) {
      return market_price * (1.0 + slippage_factor); // Pay more when buying
    } else {
      return market_price * (1.0 - slippage_factor); // Receive less when selling
    }
  }
  
  // Complete cost calculation including fees and slippage
  static std::pair<double, double> calculate_total_costs(
      const std::string& symbol,
      double quantity,
      double market_price,
      double avg_volume = 1000000, // Default 1M average volume
      double volatility = 0.20,    // Default 20% annual volatility
      const SlippageParams& params = default_slippage) {
    
    bool is_sell = quantity < 0;
    bool is_buy = quantity > 0;
    
    // Calculate fees
    double fees = calculate_fees(symbol, quantity, market_price, is_sell);
    
    // Calculate slippage
    double slippage_bps = calculate_slippage_bps(quantity, market_price, avg_volume, volatility, params);
    double execution_price = apply_slippage(market_price, slippage_bps, is_buy);
    
    // Slippage cost (difference from market price)
    double slippage_cost = std::abs(quantity) * std::abs(execution_price - market_price);
    
    return {fees, slippage_cost};
  }
};

// Static member definition
inline AlpacaCostModel::SlippageParams AlpacaCostModel::default_slippage = {};

} // namespace sentio
```

## 📄 **FILE 18 of 180**: include/sentio/csv_loader.hpp

**File Information**:
- **Path**: `include/sentio/csv_loader.hpp`

- **Size**: 8 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include <string>

namespace sentio {
bool load_csv(const std::string& path, std::vector<Bar>& out);
} // namespace sentio


```

## 📄 **FILE 19 of 180**: include/sentio/data_downloader.hpp

**File Information**:
- **Path**: `include/sentio/data_downloader.hpp`

- **Size**: 65 lines
- **Modified**: 2025-09-12 19:46:17

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>

namespace sentio {

// **DATA DOWNLOAD UTILITIES**: Extracted from poly_fetch for reuse in sentio_cli

/**
 * Calculate start date for data download based on time period
 * @param years Number of years to go back (0 = ignore)
 * @param months Number of months to go back (0 = ignore) 
 * @param days Number of days to go back (0 = ignore)
 * @return Start date string in YYYY-MM-DD format
 */
std::string calculate_start_date(int years, int months, int days);

/**
 * Get yesterday's date in YYYY-MM-DD format
 * @return Yesterday's date string
 */
std::string get_yesterday_date();

/**
 * Get current date in YYYY-MM-DD format
 * @return Current date string
 */
std::string get_current_date();

/**
 * Map symbol to family name for poly_fetch
 * @param symbol Symbol like "QQQ", "BTC", "TSLA"
 * @return Family name like "qqq", "bitcoin", "tesla"
 */
std::string symbol_to_family(const std::string& symbol);

/**
 * Get symbols for a given family
 * @param family Family name like "qqq", "bitcoin", "tesla"
 * @return Vector of symbols in that family
 */
std::vector<std::string> get_family_symbols(const std::string& family);

/**
 * Download data for a symbol family using Polygon API
 * @param symbol Primary symbol (e.g., "QQQ")
 * @param years Number of years to download (0 = ignore)
 * @param months Number of months to download (0 = ignore)
 * @param days Number of days to download (0 = ignore)
 * @param timespan "day", "hour", or "minute"
 * @param multiplier Aggregation multiplier (default: 1)
 * @param exclude_holidays Whether to exclude holidays (adds _NH suffix)
 * @param output_dir Output directory (default: "data/equities")
 * @return True if successful, false otherwise
 */
bool download_symbol_data(const std::string& symbol,
                         int years = 0,
                         int months = 0, 
                         int days = 0,
                         const std::string& timespan = "day",
                         int multiplier = 1,
                         bool exclude_holidays = false,
                         const std::string& output_dir = "data/equities");

} // namespace sentio

```

## 📄 **FILE 20 of 180**: include/sentio/data_resolver.hpp

**File Information**:
- **Path**: `include/sentio/data_resolver.hpp`

- **Size**: 44 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <filesystem>
#include <cstdlib>

namespace sentio {
enum class TickerFamily { Qqq, Bitcoin, Tesla };

inline const char** family_symbols(TickerFamily f, int& n) {
  static const char* QQQ[] = {"QQQ","TQQQ","SQQQ"}; // PSQ removed
  static const char* BTC[] = {"BTCUSD","ETHUSD"};
  static const char* TSLA[]= {"TSLA","TSLQ"};
  switch (f) {
    case TickerFamily::Qqq: n=3; return QQQ; // Updated count
    case TickerFamily::Bitcoin: n=2; return BTC;
    case TickerFamily::Tesla: n=2; return TSLA;
  }
  n=0; return nullptr;
}

inline std::string resolve_csv(const std::string& symbol,
                               const std::string& equities_root="data/equities",
                               const std::string& crypto_root="data/crypto") {
  namespace fs = std::filesystem;
  std::string up = symbol; for (auto& c: up) c = ::toupper(c);
  auto is_crypto = (up=="BTC"||up=="BTCUSD"||up=="ETH"||up=="ETHUSD");

  const char* env_root = std::getenv("SENTIO_DATA_ROOT");
  const char* env_suffix = std::getenv("SENTIO_DATA_SUFFIX");
  std::string base = env_root ? std::string(env_root) : (is_crypto ? crypto_root : equities_root);
  std::string suffix = env_suffix ? std::string(env_suffix) : std::string("");

  // Prefer suffixed file in base, then non-suffixed, then fallback to default roots
  std::string cand1 = base + "/" + up + suffix + ".csv";
  if (fs::exists(cand1)) return cand1;
  std::string cand2 = base + "/" + up + ".csv";
  if (fs::exists(cand2)) return cand2;
  std::string fallback_base = (is_crypto ? crypto_root : equities_root);
  std::string cand3 = fallback_base + "/" + up + suffix + ".csv";
  if (fs::exists(cand3)) return cand3;
  return fallback_base + "/" + up + ".csv";
}
} // namespace sentio


```

## 📄 **FILE 21 of 180**: include/sentio/dataset_metadata.hpp

**File Information**:
- **Path**: `include/sentio/dataset_metadata.hpp`

- **Size**: 86 lines
- **Modified**: 2025-09-16 15:44:52

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <cstdint>

namespace sentio {

/**
 * Dataset Metadata for Audit Traceability
 * 
 * This structure captures comprehensive information about the dataset used
 * in a backtest, enabling exact reproduction and verification.
 */
struct DatasetMetadata {
    // Source type classification
    std::string source_type = "unknown";        // "historical_csv", "future_qqq_track", "mars_simulation", etc.
    
    // File information
    std::string file_path = "";                 // Full path to the data file
    std::string file_hash = "";                 // SHA256 hash for integrity verification
    
    // Regime/track information (for AI tests)
    std::string track_id = "";                  // For future QQQ tracks (e.g., "track_01")
    std::string regime = "";                    // For AI regime tests (e.g., "normal", "volatile", "trending")
    std::string mode = "hybrid";                // Strategy execution mode ("historical", "hybrid", "live", etc.)
    
    // Dataset characteristics
    int bars_count = 0;                         // Total number of bars in the dataset
    std::int64_t time_range_start = 0;          // First timestamp in the dataset (milliseconds)
    std::int64_t time_range_end = 0;            // Last timestamp in the dataset (milliseconds)
    
    // Trading Block information (canonical evaluation units)
    int available_trading_blocks = 0;           // How many complete 480-bar Trading Blocks available
    int trading_blocks_tested = 0;              // How many Trading Blocks were actually tested
    double dataset_trading_days = 0.0;          // Total trading days equivalent (bars ÷ 390)
    double dataset_calendar_days = 0.0;         // Calendar days span of the dataset
    
    // Performance context
    std::string frequency = "1min";             // Bar frequency (1min, 5min, etc.)
    std::string market_hours = "RTH";           // Regular Trading Hours or Extended
    
    // Helper methods
    bool is_valid() const {
        return !source_type.empty() && source_type != "unknown" && bars_count > 0;
    }
    
    std::string to_json() const {
        std::string json = "{";
        json += "\"source_type\":\"" + source_type + "\",";
        json += "\"file_path\":\"" + file_path + "\",";
        json += "\"file_hash\":\"" + file_hash + "\",";
        json += "\"track_id\":\"" + track_id + "\",";
        json += "\"regime\":\"" + regime + "\",";
        json += "\"bars_count\":" + std::to_string(bars_count) + ",";
        json += "\"time_range_start\":" + std::to_string(time_range_start) + ",";
        json += "\"time_range_end\":" + std::to_string(time_range_end) + ",";
        // Trading Block metadata
        json += "\"available_trading_blocks\":" + std::to_string(available_trading_blocks) + ",";
        json += "\"trading_blocks_tested\":" + std::to_string(trading_blocks_tested) + ",";
        json += "\"dataset_trading_days\":" + std::to_string(dataset_trading_days) + ",";
        json += "\"dataset_calendar_days\":" + std::to_string(dataset_calendar_days) + ",";
        json += "\"frequency\":\"" + frequency + "\",";
        json += "\"market_hours\":\"" + market_hours + "\"";
        json += "}";
        return json;
    }
    
    // Calculate Trading Block availability from dataset
    void calculate_trading_blocks(int block_size = 480, int warmup_bars = 250) {
        if (bars_count > warmup_bars) {
            int usable_bars = bars_count - warmup_bars;
            available_trading_blocks = usable_bars / block_size;
        } else {
            available_trading_blocks = 0;
        }
        
        // Calculate trading days equivalent (assuming 390 bars per trading day)
        dataset_trading_days = static_cast<double>(bars_count) / 390.0;
        
        // Calculate calendar days from timestamp range
        if (time_range_end > time_range_start) {
            dataset_calendar_days = static_cast<double>(time_range_end - time_range_start) / (24 * 60 * 60 * 1000);
        }
    }
};

} // namespace sentio

```

## 📄 **FILE 22 of 180**: include/sentio/day_index.hpp

**File Information**:
- **Path**: `include/sentio/day_index.hpp`

- **Size**: 26 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include <vector>
#include <string_view>

namespace sentio {

// From minute bars with RFC3339 timestamps, return indices of first bar per day
inline std::vector<int> day_start_indices(const std::vector<Bar>& bars) {
    std::vector<int> starts;
    starts.reserve(bars.size() / 300 + 2);
    std::string last_day;
    for (int i=0; i<(int)bars.size(); ++i) {
        std::string_view ts(bars[i].ts_utc);
        if (ts.size() < 10) continue;
        std::string cur{ts.substr(0,10)};
        if (i == 0 || cur != last_day) {
            starts.push_back(i);
            last_day = std::move(cur);
        }
    }
    return starts;
}

} // namespace sentio


```

## 📄 **FILE 23 of 180**: include/sentio/detectors/bollinger_detector.hpp

**File Information**:
- **Path**: `include/sentio/detectors/bollinger_detector.hpp`

- **Size**: 71 lines
- **Modified**: 2025-09-18 10:31:51

- **Type**: .hpp

```text
#pragma once
#include "sentio/signal_utils.hpp"
#include "sentio/bollinger.hpp"
#include <algorithm>
#include <vector>

namespace sentio::detectors {

class BollingerDetector final : public IDetector {
private:
    enum class State { Idle, Squeezed, ArmedLong, ArmedShort };
    State state_ = State::Idle;

    int bb_window_;
    int squeeze_lookback_;
    double squeeze_percentile_;
    int min_squeeze_bars_;

    Bollinger boll_
;
    std::vector<double> sd_history_;
    int squeeze_duration_ = 0;

public:
    explicit BollingerDetector(int bb_win = 20, int sqz_lookback = 60, double sqz_pct = 0.25, int min_sqz_bars = 3)
        : bb_window_(bb_win), squeeze_lookback_(sqz_lookback), squeeze_percentile_(sqz_pct), min_squeeze_bars_(min_sqz_bars),
          boll_(bb_win, 2.0) {}

    std::string_view name() const override { return "BOLLINGER_SQZ_BREAKOUT"; }
    int warmup_period() const override { return std::max(bb_window_, squeeze_lookback_) + 1; }

    DetectorResult score(const std::vector<Bar>& bars, int idx) override {
        const auto& bar = bars[idx];
        double mid, lo, hi, sd;
        boll_.step(bar.close, mid, lo, hi, sd);

        if (sd_history_.size() >= static_cast<size_t>(squeeze_lookback_)) sd_history_.erase(sd_history_.begin());
        sd_history_.push_back(sd);

        update_state_machine(bar, mid, lo, hi, sd);

        double probability = 0.5; int direction = 0;
        if (state_ == State::ArmedLong) { probability = 0.85; direction = 1; state_ = State::Idle; }
        else if (state_ == State::ArmedShort) { probability = 0.15; direction = -1; state_ = State::Idle; }
        return {probability, direction, name()};
    }

private:
    void update_state_machine(const Bar& bar, double mid, double lo, double hi, double sd) {
        if (sd_history_.size() < static_cast<size_t>(squeeze_lookback_)) return;
        auto sds = sd_history_;
        std::sort(sds.begin(), sds.end());
        double sd_threshold = sds[static_cast<size_t>(sds.size() * squeeze_percentile_)];
        bool squeezed = (sd <= sd_threshold);

        if (state_ == State::Idle && squeezed) {
            state_ = State::Squeezed; squeeze_duration_ = 1;
        } else if (state_ == State::Squeezed) {
            if (!squeezed) state_ = State::Idle;
            else if (squeeze_duration_ >= min_squeeze_bars_) {
                if (bar.close > hi) state_ = State::ArmedLong;
                else if (bar.close < lo) state_ = State::ArmedShort;
            }
            squeeze_duration_++;
        }
    }
};

} // namespace sentio::detectors



```

## 📄 **FILE 24 of 180**: include/sentio/detectors/momentum_volume_detector.hpp

**File Information**:
- **Path**: `include/sentio/detectors/momentum_volume_detector.hpp`

- **Size**: 34 lines
- **Modified**: 2025-09-18 10:37:10

- **Type**: .hpp

```text
#pragma once
#include "sentio/signal_utils.hpp"
#include "sentio/rules/momentum_volume_rule.hpp"

namespace sentio::detectors {

class MomentumVolumeDetector final : public IDetector {
public:
    MomentumVolumeDetector() = default;
    std::string_view name() const override { return "MOMENTUM_VOLUME"; }
    int warmup_period() const override { return rule_.warmup(); }
    DetectorResult score(const std::vector<Bar>& bars, int idx) override {
        rules::BarsView v{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0};
        to_view(bars, v);
        auto out = rule_.eval(v, idx);
        if (!out) return {0.5, 0, name()};
        double p = out->p_up ? static_cast<double>(*out->p_up) : 0.5 + 0.5 * (out->signal.value_or(0));
        int dir = out->signal.value_or(0);
        return {p, dir, name()};
    }
private:
    rules::MomentumVolumeRule rule_;
    static void to_view(const std::vector<Bar>& b, rules::BarsView& v){
        // This quick adapter uses contiguous vectors to temporary buffers
        static std::vector<int64_t> ts; static std::vector<double> open,high,low,close,vol;
        size_t N=b.size(); ts.resize(N); open.resize(N); high.resize(N); low.resize(N); close.resize(N); vol.resize(N);
        for(size_t i=0;i<N;i++){ ts[i]=b[i].ts_utc_epoch; open[i]=b[i].open; high[i]=b[i].high; low[i]=b[i].low; close[i]=b[i].close; vol[i]=b[i].volume; }
        v.ts=ts.data(); v.open=open.data(); v.high=high.data(); v.low=low.data(); v.close=close.data(); v.volume=vol.data(); v.n=(int64_t)N;
    }
};

} // namespace sentio::detectors



```

## 📄 **FILE 25 of 180**: include/sentio/detectors/ofi_proxy_detector.hpp

**File Information**:
- **Path**: `include/sentio/detectors/ofi_proxy_detector.hpp`

- **Size**: 32 lines
- **Modified**: 2025-09-18 10:37:10

- **Type**: .hpp

```text
#pragma once
#include "sentio/signal_utils.hpp"
#include "sentio/rules/ofi_proxy_rule.hpp"

namespace sentio::detectors {

class OFIProxyDetector final : public IDetector {
public:
    OFIProxyDetector() = default;
    std::string_view name() const override { return "OFI_PROXY"; }
    int warmup_period() const override { return rule_.warmup(); }
    DetectorResult score(const std::vector<Bar>& bars, int idx) override {
        rules::BarsView v{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0}; to_view(bars, v);
        auto out = rule_.eval(v, idx);
        if (!out) return {0.5, 0, name()};
        double p = out->p_up.value_or(0.5);
        int dir = out->signal.value_or( (p>0.55) ? 1 : (p<0.45 ? -1 : 0) );
        return {p, dir, name()};
    }
private:
    rules::OFIProxyRule rule_;
    static void to_view(const std::vector<Bar>& b, rules::BarsView& v){
        static std::vector<int64_t> ts; static std::vector<double> open,high,low,close,vol;
        size_t N=b.size(); ts.resize(N); open.resize(N); high.resize(N); low.resize(N); close.resize(N); vol.resize(N);
        for(size_t i=0;i<N;i++){ ts[i]=b[i].ts_utc_epoch; open[i]=b[i].open; high[i]=b[i].high; low[i]=b[i].low; close[i]=b[i].close; vol[i]=b[i].volume; }
        v.ts=ts.data(); v.open=open.data(); v.high=high.data(); v.low=low.data(); v.close=close.data(); v.volume=vol.data(); v.n=(int64_t)N;
    }
};

} // namespace sentio::detectors



```

## 📄 **FILE 26 of 180**: include/sentio/detectors/opening_range_breakout_detector.hpp

**File Information**:
- **Path**: `include/sentio/detectors/opening_range_breakout_detector.hpp`

- **Size**: 32 lines
- **Modified**: 2025-09-18 10:37:10

- **Type**: .hpp

```text
#pragma once
#include "sentio/signal_utils.hpp"
#include "sentio/rules/opening_range_breakout_rule.hpp"

namespace sentio::detectors {

class OpeningRangeBreakoutDetector final : public IDetector {
public:
    OpeningRangeBreakoutDetector() = default;
    std::string_view name() const override { return "OPENING_RANGE_BRK"; }
    int warmup_period() const override { return rule_.warmup(); }
    DetectorResult score(const std::vector<Bar>& bars, int idx) override {
        rules::BarsView v{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0}; to_view(bars, v);
        auto out = rule_.eval(v, idx);
        if (!out) return {0.5, 0, name()};
        int dir = out->signal.value_or(0);
        double p = (dir>0)? 0.8 : (dir<0? 0.2 : 0.5);
        return {p, dir, name()};
    }
private:
    rules::OpeningRangeBreakoutRule rule_;
    static void to_view(const std::vector<Bar>& b, rules::BarsView& v){
        static std::vector<int64_t> ts; static std::vector<double> open,high,low,close,vol;
        size_t N=b.size(); ts.resize(N); open.resize(N); high.resize(N); low.resize(N); close.resize(N); vol.resize(N);
        for(size_t i=0;i<N;i++){ ts[i]=b[i].ts_utc_epoch; open[i]=b[i].open; high[i]=b[i].high; low[i]=b[i].low; close[i]=b[i].close; vol[i]=b[i].volume; }
        v.ts=ts.data(); v.open=open.data(); v.high=high.data(); v.low=low.data(); v.close=close.data(); v.volume=vol.data(); v.n=(int64_t)N;
    }
};

} // namespace sentio::detectors



```

## 📄 **FILE 27 of 180**: include/sentio/detectors/rsi_detector.hpp

**File Information**:
- **Path**: `include/sentio/detectors/rsi_detector.hpp`

- **Size**: 44 lines
- **Modified**: 2025-09-18 10:31:51

- **Type**: .hpp

```text
#pragma once
#include "sentio/signal_utils.hpp"
#include "sentio/rsi_prob.hpp"

namespace sentio::detectors {

class RsiDetector final : public IDetector {
public:
    explicit RsiDetector(int period = 14, double alpha = 1.0)
        : period_(period), alpha_(alpha) {}

    std::string_view name() const override { return "RSI_REVERSION"; }
    int warmup_period() const override { return period_ + 1; }

    DetectorResult score(const std::vector<Bar>& bars, int idx) override {
        if (idx < warmup_period()) return {0.5, 0, name()};
        double rsi = calculate_rsi(bars, idx);
        double probability = rsi_to_prob_tuned(rsi, alpha_);
        int direction = (probability > 0.55) ? 1 : (probability < 0.45 ? -1 : 0);
        return {probability, direction, name()};
    }

private:
    int period_;
    double alpha_;

    double calculate_rsi(const std::vector<Bar>& bars, int end_idx) {
        double avg_gain = 0.0, avg_loss = 0.0;
        int start_idx = end_idx - period_;
        for (int i = start_idx + 1; i <= end_idx; ++i) {
            double change = bars[i].close - bars[i - 1].close;
            if (change > 0) avg_gain += change; else avg_loss -= change;
        }
        avg_gain /= period_;
        avg_loss /= period_;
        if (avg_loss < 1e-9) return 100.0;
        double rs = avg_gain / avg_loss;
        return 100.0 - (100.0 / (1.0 + rs));
    }
};

} // namespace sentio::detectors



```

## 📄 **FILE 28 of 180**: include/sentio/detectors/vwap_reversion_detector.hpp

**File Information**:
- **Path**: `include/sentio/detectors/vwap_reversion_detector.hpp`

- **Size**: 32 lines
- **Modified**: 2025-09-18 10:37:10

- **Type**: .hpp

```text
#pragma once
#include "sentio/signal_utils.hpp"
#include "sentio/rules/vwap_reversion_rule.hpp"

namespace sentio::detectors {

class VwapReversionDetector final : public IDetector {
public:
    VwapReversionDetector() = default;
    std::string_view name() const override { return "VWAP_REVERSION"; }
    int warmup_period() const override { return rule_.warmup(); }
    DetectorResult score(const std::vector<Bar>& bars, int idx) override {
        rules::BarsView v{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0}; to_view(bars, v);
        auto out = rule_.eval(v, idx);
        if (!out) return {0.5, 0, name()};
        int dir = out->signal.value_or(0);
        double p = (dir>0)? 0.7 : (dir<0? 0.3 : 0.5);
        return {p, dir, name()};
    }
private:
    rules::VWAPReversionRule rule_;
    static void to_view(const std::vector<Bar>& b, rules::BarsView& v){
        static std::vector<int64_t> ts; static std::vector<double> open,high,low,close,vol;
        size_t N=b.size(); ts.resize(N); open.resize(N); high.resize(N); low.resize(N); close.resize(N); vol.resize(N);
        for(size_t i=0;i<N;i++){ ts[i]=b[i].ts_utc_epoch; open[i]=b[i].open; high[i]=b[i].high; low[i]=b[i].low; close[i]=b[i].close; vol[i]=b[i].volume; }
        v.ts=ts.data(); v.open=open.data(); v.high=high.data(); v.low=low.data(); v.close=close.data(); v.volume=vol.data(); v.n=(int64_t)N;
    }
};

} // namespace sentio::detectors



```

## 📄 **FILE 29 of 180**: include/sentio/eod_position_manager.hpp

**File Information**:
- **Path**: `include/sentio/eod_position_manager.hpp`

- **Size**: 222 lines
- **Modified**: 2025-09-18 16:35:12

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include "audit.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <ctime>

// Forward declaration
class IAuditRecorder;

namespace sentio {

/**
 * Enhanced End-of-Day Position Manager
 * 
 * Mandatory daily position closure system to minimize overnight carry risk,
 * especially critical for leveraged ETFs (TQQQ/SQQQ) that experience decay.
 * 
 * Features:
 * - Multi-stage closure process (warning → partial → mandatory)
 * - Leveraged ETF priority (close high-decay positions first)
 * - Market session awareness (RTH, extended hours)
 * - Configurable closure timing and methods
 * - Comprehensive audit logging
 * - Risk-based position prioritization
 */
class EODPositionManager {
public:
    struct Config {
        // **CLOSURE TIMING CONFIGURATION**
        int warning_minutes_before_close = 30;     // Start warning phase
        int partial_close_minutes = 15;            // Begin partial closure
        int mandatory_close_minutes = 5;           // Force complete closure
        int final_sweep_minutes = 1;               // Final safety sweep
        
        // **POSITION PRIORITIZATION**
        bool prioritize_leveraged_etfs = true;     // Close TQQQ/SQQQ first (high decay)
        bool prioritize_large_positions = true;    // Close large positions first
        double min_position_value = 10.0;          // Minimum $ value to close
        
        // **CLOSURE METHOD CONFIGURATION**
        bool use_market_orders = true;             // Market vs limit orders
        bool allow_partial_closure = true;        // Gradual vs immediate closure
        double partial_closure_fraction = 0.5;    // Fraction to close in partial phase
        
        // **MARKET SESSION AWARENESS**
        bool close_during_extended_hours = false; // Close during after-hours
        bool respect_market_holidays = true;      // Skip closure on holidays
        
        // **RISK MANAGEMENT**
        double max_overnight_exposure = 0.0;      // Maximum $ exposure to carry overnight
        bool force_cash_only_overnight = true;    // Ensure 100% cash overnight
        
        // **AUDIT AND LOGGING**
        bool detailed_logging = true;             // Comprehensive audit trail
        std::string closure_reason_prefix = "EOD_MANDATORY_CLOSE";
    };
    
    enum class ClosurePhase {
        NORMAL_TRADING,    // Regular trading hours, no closure activity
        WARNING,           // Warning phase - alert but no action
        PARTIAL_CLOSE,     // Begin partial position reduction
        MANDATORY_CLOSE,   // Force complete position closure
        FINAL_SWEEP,       // Final safety check and cleanup
        MARKET_CLOSED      // After market close
    };
    
    struct PositionRisk {
        size_t symbol_id;
        std::string symbol;
        double quantity;
        double market_value;
        double decay_risk_score;    // Higher = more urgent to close
        bool is_leveraged_etf;
        int priority_rank;          // 1 = highest priority
    };
    
    EODPositionManager() : EODPositionManager(Config{}) {}
    
    explicit EODPositionManager(const Config& cfg) 
        : config_(cfg), current_phase_(ClosurePhase::NORMAL_TRADING) {}
    
    /**
     * Main processing function - call on every bar during trading hours
     * 
     * @param portfolio Current portfolio state
     * @param pricebook Current market prices
     * @param ST Symbol table for lookups
     * @param current_timestamp Current bar timestamp (UTC epoch)
     * @param audit Audit logger for position closure events
     * @return Number of positions closed in this call
     */
    int process_eod_closure(
        Portfolio& portfolio,
        const std::vector<double>& last_prices,
        const SymbolTable& ST,
        int64_t current_timestamp,
        IAuditRecorder& audit
    );
    
    /**
     * Check if we're in an active closure phase or market is closed
     * Once EOD closure begins, no new trades allowed until next trading day
     */
    bool is_closure_active() const {
        return current_phase_ != ClosurePhase::NORMAL_TRADING;
    }
    
    /**
     * Get current closure phase
     */
    ClosurePhase get_current_phase() const { return current_phase_; }
    
    /**
     * Get minutes until market close (negative if after close)
     */
    int get_minutes_to_close(int64_t current_timestamp) const;
    
    /**
     * Force immediate closure of all positions (emergency use)
     */
    int force_immediate_closure(
        Portfolio& portfolio,
        const std::vector<double>& last_prices,
        const SymbolTable& ST,
        int64_t current_timestamp,
        IAuditRecorder& audit,
        const std::string& reason = "EMERGENCY_CLOSE"
    );
    
    /**
     * Check if position should be exempt from closure (e.g., cash equivalents)
     */
    bool is_position_exempt(const std::string& symbol) const;
    
    /**
     * Get detailed position risk analysis for current portfolio
     */
    std::vector<PositionRisk> analyze_position_risks(
        const Portfolio& portfolio,
        const std::vector<double>& last_prices,
        const SymbolTable& ST
    ) const;
    
    // Configuration access
    const Config& get_config() const { return config_; }
    void update_config(const Config& new_config) { config_ = new_config; }
    
    // Statistics and diagnostics
    struct Statistics {
        int total_closures_today = 0;
        int leveraged_etf_closures = 0;
        int partial_closures = 0;
        int mandatory_closures = 0;
        double total_value_closed = 0.0;
        std::string last_closure_time;
        std::vector<std::string> closure_log;
    };
    
    const Statistics& get_statistics() const { return stats_; }
    void reset_daily_statistics() { stats_ = Statistics{}; }

private:
    Config config_;
    ClosurePhase current_phase_;
    Statistics stats_;
    
    // Market timing helpers
    struct MarketTiming {
        int market_close_hour_utc;
        int market_close_minute_utc;
        bool is_edt;  // Eastern Daylight Time vs EST
    };
    
    MarketTiming get_market_timing(int64_t timestamp) const;
    ClosurePhase determine_closure_phase(int minutes_to_close) const;
    
    // Position analysis and prioritization
    double calculate_decay_risk_score(const std::string& symbol, double market_value) const;
    std::vector<PositionRisk> prioritize_positions(
        const std::vector<PositionRisk>& positions
    ) const;
    
    // Closure execution
    bool close_position(
        Portfolio& portfolio,
        size_t symbol_id,
        double quantity,
        double close_price,
        const SymbolTable& ST,
        int64_t timestamp,
        IAuditRecorder& audit,
        const std::string& reason,
        double closure_fraction = 1.0  // 1.0 = full close, 0.5 = half close
    );
    
    // Leveraged ETF identification
    static const std::unordered_set<std::string> LEVERAGED_ETFS;
    static const std::unordered_set<std::string> HIGH_DECAY_ETFS;
    static const std::unordered_set<std::string> EXEMPT_SYMBOLS;
    
    // Logging helpers
    void log_closure_event(
        const std::string& symbol,
        double quantity,
        double price,
        const std::string& reason,
        IAuditRecorder& audit,
        int64_t timestamp
    );
    
    std::string format_timestamp(int64_t timestamp) const;
    std::string phase_to_string(ClosurePhase phase) const;
};

// Static ETF classifications for risk management (defined in .cpp file)

} // namespace sentio

```

## 📄 **FILE 30 of 180**: include/sentio/exec/asof_index.hpp

**File Information**:
- **Path**: `include/sentio/exec/asof_index.hpp`

- **Size**: 22 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>

namespace sentio {

// Build a map from instrument rows to base rows via as-of (<= ts) alignment.
// base_ts, inst_ts must be monotonic non-decreasing (your loaders should ensure).
inline std::vector<int32_t> build_asof_index(const std::vector<std::int64_t>& base_ts,
                                             const std::vector<std::int64_t>& inst_ts) {
  std::vector<int32_t> idx(inst_ts.size(), -1);
  std::size_t j = 0;
  if (base_ts.empty()) return idx;
  for (std::size_t i = 0; i < inst_ts.size(); ++i) {
    auto t = inst_ts[i];
    while (j + 1 < base_ts.size() && base_ts[j + 1] <= t) ++j;
    idx[i] = static_cast<int32_t>(j);
  }
  return idx;
}

} // namespace sentio

```

## 📄 **FILE 31 of 180**: include/sentio/exec_types.hpp

**File Information**:
- **Path**: `include/sentio/exec_types.hpp`

- **Size**: 15 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
// exec_types.hpp
#pragma once
#include <string>

namespace sentio {

struct ExecutionIntent {
  std::string base_symbol;     // e.g., "QQQ"
  std::string instrument;      // e.g., "TQQQ" or "SQQQ" or "QQQ"
  double      qty = 0.0;
  double      leverage = 1.0;  // informational; actual product carries leverage
  double      score = 0.0;     // signal strength
};

} // namespace sentio
```

## 📄 **FILE 32 of 180**: include/sentio/execution/pnl_engine.hpp

**File Information**:
- **Path**: `include/sentio/execution/pnl_engine.hpp`

- **Size**: 171 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <cmath>
#include <optional>
#include <algorithm>
#include <stdexcept>

namespace sentio {

struct Fill {
  int64_t ts{0};
  double  price{0.0};
  double  qty{0.0};  // +buy, -sell
  double  fee{0.0};
  std::string venue;
};

struct BarPx {
  int64_t ts{0};
  double close{0.0};
  double bid{0.0};
  double ask{0.0};
};

struct Lot {
  double qty{0.0};
  double px{0.0};
  int64_t ts{0};
};

struct PnLSnapshot {
  int64_t ts{0};
  double position{0.0};
  double avg_price{0.0};
  double cash{0.0};
  double realized{0.0};
  double unrealized{0.0};
  double equity{0.0};
  double last_price{0.0};
};

struct FeeModel {
  double per_share{0.0};
  double bps_notional{0.0};
  double min_fee{0.0};
  double compute(double price, double qty) const {
    double absqty = std::abs(qty);
    double f = per_share * absqty + (bps_notional * (price * absqty));
    return std::max(f, min_fee);
  }
};

struct SlippageModel {
  double bps{0.0};
  double apply(double ref_price, double qty) const {
    double sgn = (qty >= 0.0 ? 1.0 : -1.0);
    double mult = 1.0 + sgn * bps;
    return ref_price * mult;
  }
};

class PnLEngine {
public:
  enum class PriceMode { Close, Mid, Bid, Ask };

  explicit PnLEngine(double start_cash = 100000.0)
    : cash_(start_cash), equity_(start_cash) {}

  void set_price_mode(PriceMode m) { price_mode_ = m; }
  void set_fee_model(const FeeModel& f) { fee_model_ = f; auto_fee_ = true; }
  void set_slippage_model(const SlippageModel& s) { slippage_ = s; }

  void on_fill(const Fill& fill) {
    if (std::abs(fill.qty) < 1e-12) return;
    const double qty = fill.qty;
    const double px  = fill.price;
    double fee = fill.fee;
    if (auto_fee_) fee = fee_model_.compute(px, qty);

    double remaining = qty;
    if (same_sign(remaining, position_)) {
      lots_.push_back(Lot{remaining, px, fill.ts});
      position_ += remaining;
      cash_ -= px * qty;
      cash_ -= fee;
    } else {
      while (std::abs(remaining) > 1e-12 && !lots_.empty() && opposite_sign(remaining, lots_.front().qty)) {
        Lot &lot = lots_.front();
        double close_qty = std::min(std::abs(remaining), std::abs(lot.qty));
        double dq = (lot.qty > 0 ? +close_qty : -close_qty);
        realized_ += (px - lot.px) * dq;
        cash_ -= px * (-dq);
        lot.qty -= dq;
        remaining += dq;
        if (std::abs(lot.qty) <= 1e-12) lots_.erase(lots_.begin());
      }
      if (std::abs(remaining) > 1e-12) {
        lots_.push_back(Lot{remaining, px, fill.ts});
        position_ += remaining;
        cash_ -= px * remaining;
      } else {
        position_ = sum_position_from_lots();
      }
      cash_ -= fee;
    }
    avg_price_ = compute_signed_avg();
  }

  void on_bar(const BarPx& bar) {
    last_price_ = reference_price(bar);
    unrealized_ = position_ * (last_price_ - avg_price_);
    equity_     = cash_ + position_ * last_price_;
    snapshots_.push_back(PnLSnapshot{bar.ts, position_, avg_price_, cash_, realized_, unrealized_, equity_, last_price_});
  }

  void reset(double start_cash = 100000.0) {
    lots_.clear(); snapshots_.clear();
    position_=0; avg_price_=0; cash_=start_cash; realized_=0; unrealized_=0; equity_=start_cash; last_price_=0;
  }

  const std::vector<PnLSnapshot>& snapshots() const { return snapshots_; }
  double position()  const { return position_; }
  double avg_price() const { return avg_price_; }
  double cash()      const { return cash_; }
  double realized()  const { return realized_; }
  double unrealized()const { return unrealized_; }
  double equity()    const { return equity_; }

private:
  static bool same_sign(double a, double b){ return (a>=0 && b>=0) || (a<=0 && b<=0); }
  static bool opposite_sign(double a, double b){ return (a>=0 && b<=0) || (a<=0 && b>=0); }

  double compute_signed_avg() const {
    double num = 0.0, den = 0.0;
    for (const auto &l : lots_) { num += l.px * l.qty; den += l.qty; }
    if (std::abs(den) < 1e-12) return 0.0;
    return num / den;
  }
  double sum_position_from_lots() const { double s=0.0; for (const auto &l : lots_) s += l.qty; return s; }

  double reference_price(const BarPx& bar) const {
    switch (price_mode_) {
      case PriceMode::Close: return bar.close;
      case PriceMode::Mid:   return (bar.bid>0 && bar.ask>0) ? 0.5*(bar.bid+bar.ask) : bar.close;
      case PriceMode::Bid:   return (bar.bid>0) ? bar.bid : bar.close;
      case PriceMode::Ask:   return (bar.ask>0) ? bar.ask : bar.close;
    }
    return bar.close;
  }

private:
  std::vector<Lot> lots_;
  std::vector<PnLSnapshot> snapshots_;

  double position_{0.0};
  double avg_price_{0.0};
  double cash_{0.0};
  double realized_{0.0};
  double unrealized_{0.0};
  double equity_{0.0};
  double last_price_{0.0};

  PriceMode price_mode_{PriceMode::Close};
  FeeModel fee_model_{};
  SlippageModel slippage_{};
  bool auto_fee_{false};
};

} // namespace sentio

```

## 📄 **FILE 33 of 180**: include/sentio/family_mapper.hpp

**File Information**:
- **Path**: `include/sentio/family_mapper.hpp`

- **Size**: 42 lines
- **Modified**: 2025-09-11 21:59:53

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace sentio {

class FamilyMapper {
public:
    // Provide complete families you trade; keep in config.
    // Key = family id; Values = member symbols (case-insensitive).
    using Map = std::unordered_map<std::string, std::vector<std::string>>;

    explicit FamilyMapper(Map families) : families_(std::move(families)) {
        // build reverse index
        for (auto& [fam, syms] : families_) {
            for (auto s : syms) {
                auto u = upper(s);
                rev_[u] = fam;
            }
        }
    }

    // Return family for a symbol, or the symbol itself if unknown.
    std::string family_for(const std::string& symbol) const {
        auto u = upper(symbol);
        auto it = rev_.find(u);
        return it==rev_.end() ? u : it->second;
    }

private:
    static std::string upper(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::toupper);
        return s;
    }

    Map families_;
    std::unordered_map<std::string,std::string> rev_;
};

} // namespace sentio

```

## 📄 **FILE 34 of 180**: include/sentio/feature/column_projector.hpp

**File Information**:
- **Path**: `include/sentio/feature/column_projector.hpp`

- **Size**: 41 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace sentio {

struct ColumnProjector {
  std::vector<int> src_to_dst; // index in src for each dst column; -1 -> pad
  float pad{0.0f};

  static ColumnProjector make(const std::vector<std::string>& src,
                              const std::vector<std::string>& dst,
                              float pad_value){
    std::unordered_map<std::string,int> pos;
    for (int i=0;i<(int)src.size();++i) pos[src[i]] = i;
    ColumnProjector P; P.pad = pad_value;
    P.src_to_dst.resize(dst.size(), -1);
    for (int j=0;j<(int)dst.size();++j){
      auto it = pos.find(dst[j]);
      P.src_to_dst[j] = (it==pos.end()) ? -1 : it->second;
    }
    return P;
  }

  void project(const float* X, size_t rows, size_t src_cols, std::vector<float>& Y) const {
    const size_t dst_cols = src_to_dst.size();
    Y.assign(rows * dst_cols, pad);
    for (size_t r=0;r<rows;++r){
      const float* src = X + r*src_cols;
      float* dst = Y.data() + r*dst_cols;
      for (size_t j=0;j<dst_cols;++j){
        int si = src_to_dst[j];
        if (si>=0) dst[j] = src[si];
      }
    }
  }
};

} // namespace sentio

```

## 📄 **FILE 35 of 180**: include/sentio/feature/column_projector_safe.hpp

**File Information**:
- **Path**: `include/sentio/feature/column_projector_safe.hpp`

- **Size**: 90 lines
- **Modified**: 2025-09-18 12:07:25

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <cstddef>
#include <iostream>

namespace sentio {

struct ColumnProjectorSafe {
  std::vector<int> map;   // dst[j] = src index or -1
  size_t F_src{0}, F_dst{0};
  float fill_value{0.0f};

  static ColumnProjectorSafe make(const std::vector<std::string>& src,
                                  const std::vector<std::string>& dst,
                                  float fill_value = 0.0f) {
    ColumnProjectorSafe P; 
    P.F_src = src.size(); 
    P.F_dst = dst.size(); 
    P.fill_value = fill_value;
    P.map.assign(P.F_dst, -1);
    
    std::unordered_map<std::string,int> pos; 
    pos.reserve(src.size()*2);
    for (int i=0;i<(int)src.size();++i) pos[src[i]] = i;
    
    for (int j=0;j<(int)dst.size();++j){
      auto it = pos.find(dst[j]);
      if (it!=pos.end()) {
        P.map[j] = it->second; // Found mapping
      } else {
        P.map[j] = -1; // Will be filled
      }
    }
    
    
    
    return P;
  }

  void project(const float* X_src, size_t rows, size_t Fsrc, std::vector<float>& X_out) const {
    if (Fsrc != F_src) {
      throw std::runtime_error("ColumnProjectorSafe: F_src mismatch expected=" + 
                               std::to_string(F_src) + " got=" + std::to_string(Fsrc));
    }
    
    X_out.assign(rows*F_dst, fill_value);
    
    for (size_t r=0;r<rows;++r){
      const float* src = X_src + r*F_src;
      float* dst = X_out.data() + r*F_dst;
      
      for (size_t j=0;j<F_dst;++j){
        int si = map[j];
        if (si >= 0 && si < (int)F_src) {
          dst[j] = src[(size_t)si];
        } else {
          dst[j] = fill_value;
        }
      }
    }
  }
  
  void project_double(const double* X_src, size_t rows, size_t Fsrc, std::vector<float>& X_out) const {
    if (Fsrc != F_src) {
      throw std::runtime_error("ColumnProjectorSafe: F_src mismatch expected=" + 
                               std::to_string(F_src) + " got=" + std::to_string(Fsrc));
    }
    
    X_out.assign(rows*F_dst, fill_value);
    
    for (size_t r=0;r<rows;++r){
      const double* src = X_src + r*F_src;
      float* dst = X_out.data() + r*F_dst;
      
      for (size_t j=0;j<F_dst;++j){
        int si = map[j];
        if (si >= 0 && si < (int)F_src) {
          dst[j] = static_cast<float>(src[(size_t)si]);
        } else {
          dst[j] = fill_value;
        }
      }
    }
  }
};

} // namespace sentio
```

## 📄 **FILE 36 of 180**: include/sentio/feature/csv_feature_provider.hpp

**File Information**:
- **Path**: `include/sentio/feature/csv_feature_provider.hpp`

- **Size**: 60 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "sentio/feature/feature_provider.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace sentio {

struct CsvFeatureProvider : IFeatureProvider {
  std::string path_;
  std::vector<std::string> names_;
  int T_{64};

  explicit CsvFeatureProvider(std::string path, int T)
  : path_(std::move(path)), T_(T) {
    std::ifstream f(path_);
    if (!f) throw std::runtime_error("missing features csv: " + path_);
    std::string header;
    std::getline(f, header);
    std::stringstream hs(header);
    std::string col; int idx=0;
    while (std::getline(hs, col, ',')) {
      if (idx==0 && (col=="bar_index"||col=="idx")) { idx++; continue; }
      if (col=="ts" || col=="timestamp") { idx++; continue; }
      names_.push_back(col);
      idx++;
    }
  }

  FeatureMatrix get_features_for(const std::string& /*symbol*/) override {
    std::ifstream f(path_);
    if (!f) throw std::runtime_error("missing: " + path_);
    std::string line;
    std::getline(f, line); // header
    std::vector<float> buf; buf.reserve(1<<20);
    int64_t rows=0; const int64_t cols=(int64_t)names_.size();
    while (std::getline(f, line)) {
      if (line.empty()) continue;
      std::stringstream ss(line);
      std::string cell; int colidx=0; bool first=true; bool have_ts=false;
      // optional bar_index, optional timestamp, then features
      while (std::getline(ss, cell, ',')) {
        if (first) { first=false; continue; }
        if (!have_ts) { have_ts=true; continue; }
        buf.push_back(cell.empty()? 0.0f : std::stof(cell));
        colidx++;
      }
      if (colidx != cols) throw std::runtime_error("col mismatch in " + path_);
      rows++;
    }
    return FeatureMatrix{rows, cols, std::move(buf)};
  }

  std::vector<std::string> feature_names() const override { return names_; }
  int seq_len() const override { return T_; }
};

} // namespace sentio

```

## 📄 **FILE 37 of 180**: include/sentio/feature/feature_builder_guarded.hpp

**File Information**:
- **Path**: `include/sentio/feature/feature_builder_guarded.hpp`

- **Size**: 104 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include "sentio/core/bar.hpp"
#include "sentio/feature/feature_matrix.hpp"
#include "sentio/sym/leverage_registry.hpp"
#include "sentio/sym/symbol_utils.hpp"

namespace sentio {

// Throws if symbol is leveraged: you must pass a base ticker (e.g., "QQQ")
inline FeatureMatrix build_features_for_base(const std::string& symbol,
                                             const std::vector<Bar>& bars);

// ---- Implementation (header-only for simplicity) ----
namespace detail {
  inline float s_log_safe(float x) { return std::log(std::max(x, 1e-12f)); }
}

inline FeatureMatrix build_features_for_base(const std::string& symbol,
                                             const std::vector<Bar>& bars)
{
  const auto symU = to_upper(symbol);
  if (is_leveraged(symU)) {
    throw std::invalid_argument("FeatureBuilder: leveraged symbol '" + symU +
                                "' not allowed. Pass base ticker: '" + resolve_base(symU) + "'");
  }

  const std::int64_t N = static_cast<std::int64_t>(bars.size());
  if (N < 64) return {}; // not enough history; adjust to your min

  // Example feature set (extend as needed)
  // 0: close, 1: logret, 2: ema20, 3: ema50, 4: rsi14, 5: zscore20(logret)
  constexpr int F = 6;
  FeatureMatrix M;
  M.rows = N; M.cols = F;
  M.data.resize(static_cast<std::size_t>(N * F));

  std::vector<float> close(N), logret(N, 0.f), ema20(N, 0.f), ema50(N, 0.f), rsi14(N, 0.f), z20(N, 0.f);

  for (std::int64_t i = 0; i < N; ++i) close[i] = static_cast<float>(bars[i].close);
  for (std::int64_t i = 1; i < N; ++i) logret[i] = detail::s_log_safe(close[i] / std::max(close[i-1], 1e-12f));

  auto ema = [&](int period, std::vector<float>& out){
    const float k = 2.f / (period + 1.f);
    float e = close[0];
    out[0] = e;
    for (std::int64_t i=1; i<N; ++i){ e = k*close[i] + (1.f - k)*e; out[i] = e; }
  };
  ema(20, ema20);
  ema(50, ema50);

  // RSI(14) (Wilders)
  {
    const int p = 14;
    float up=0.f, dn=0.f;
    for (int i=1; i<=p && i<N; ++i){
      float d = close[i]-close[i-1];
      up += std::max(d, 0.f);
      dn += std::max(-d, 0.f);
    }
    up/=p; dn/=p;
    for (std::int64_t i=p+1; i<N; ++i){
      float d = close[i]-close[i-1];
      up = (up*(p-1) + std::max(d,0.f)) / p;
      dn = (dn*(p-1) + std::max(-d,0.f)) / p;
      float rs = (dn>1e-12f) ? (up/dn) : 0.f;
      rsi14[i] = 100.f - 100.f/(1.f + rs);
    }
  }

  // Z-score(20) of logret
  {
    const int w = 20;
    if (N > w) {
      double sum=0.0, sum2=0.0;
      for (int i=0; i<w; ++i){ sum += logret[i]; sum2 += logret[i]*logret[i]; }
      for (std::int64_t i=w; i<N; ++i){
        const double mu = sum / w;
        const double var = std::max(0.0, sum2 / w - mu*mu);
        const float sd = static_cast<float>(std::sqrt(var));
        z20[i] = sd > 1e-8f ? static_cast<float>((logret[i]-mu)/sd) : 0.f;
        // slide
        sum += logret[i] - logret[i-w];
        sum2 += logret[i]*logret[i] - logret[i-w]*logret[i-w];
      }
    }
  }

  // Pack row-major
  for (std::int64_t i=0; i<N; ++i) {
    float* r = M.row_ptr(i);
    r[0] = close[i];
    r[1] = logret[i];
    r[2] = ema20[i];
    r[3] = ema50[i];
    r[4] = rsi14[i];
    r[5] = z20[i];
  }
  return M;
}

} // namespace sentio

```

## 📄 **FILE 38 of 180**: include/sentio/feature/feature_builder_ops.hpp

**File Information**:
- **Path**: `include/sentio/feature/feature_builder_ops.hpp`

- **Size**: 66 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace sentio {

inline std::vector<float> fb_ident(const std::vector<float>& x){ return x; }

inline std::vector<float> fb_logret(const std::vector<float>& x){
  std::vector<float> y(x.size(), 0.f);
  for (size_t i=1;i<x.size();++i){
    float a = std::max(x[i],    1e-12f);
    float b = std::max(x[i-1],  1e-12f);
    y[i] = std::log(a) - std::log(b);
  }
  return y;
}

inline std::vector<float> fb_ema(const std::vector<float>& x, int p){
  std::vector<float> e(x.size());
  if (x.empty()) return e;
  float k = 2.f / (p + 1.f);
  e[0] = x[0];
  for (size_t i=1;i<x.size();++i) e[i] = k*x[i] + (1.f-k)*e[i-1];
  return e;
}

inline std::vector<float> fb_rsi(const std::vector<float>& x, int p=14){
  const size_t N=x.size();
  std::vector<float> out(N,0.f), up(N,0.f), dn(N,0.f);
  for (size_t i=1;i<N;++i){
    float d=x[i]-x[i-1]; up[i]=std::max(d,0.f); dn[i]=std::max(-d,0.f);
  }
  std::vector<float> ru(N,0.f), rd(N,0.f);
  if (N> (size_t)p){
    float su=0.f, sd=0.f;
    for (int i=1;i<=p;i++){ su+=up[i]; sd+=dn[i]; }
    ru[p]=su/p; rd[p]=sd/p;
    for (size_t i=p+1;i<N;++i){
      ru[i]=(ru[i-1]*(p-1)+up[i])/p;
      rd[i]=(rd[i-1]*(p-1)+dn[i])/p;
      float rs=(rd[i]>1e-12f)?(ru[i]/rd[i]):0.f;
      out[i]=100.f-100.f/(1.f+rs);
    }
  }
  return out;
}

inline std::vector<float> fb_zwin(const std::vector<float>& x, int w){
  const size_t N=x.size(); std::vector<float> out(N,0.f);
  if (N <= (size_t)w) return out;
  std::vector<double> s(N,0.0), s2(N,0.0);
  s[0]=x[0]; s2[0]=x[0]*x[0];
  for (size_t i=1;i<N;++i){ s[i]=s[i-1]+x[i]; s2[i]=s2[i-1]+x[i]*x[i]; }
  for (size_t i=w;i<N;++i){
    double su = s[i]-s[i-w], su2 = s2[i]-s2[i-w];
    double mu = su/w;
    double var = std::max(0.0, su2/w - mu*mu);
    float sd = (float)std::sqrt(var);
    out[i] = (sd>1e-8f) ? (float)((x[i]-mu)/sd) : 0.f;
  }
  return out;
}

} // namespace sentio

```

## 📄 **FILE 39 of 180**: include/sentio/feature/feature_feeder_guarded.hpp

**File Information**:
- **Path**: `include/sentio/feature/feature_feeder_guarded.hpp`

- **Size**: 54 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "sentio/core/bar.hpp"
#include "sentio/feature/feature_matrix.hpp"
#include "sentio/feature/standard_scaler.hpp"
#include "sentio/sym/leverage_registry.hpp"
#include "sentio/exec/asof_index.hpp"

namespace sentio {

struct FeederInit {
  // All price series loaded (both base and leveraged allowed here)
  std::unordered_map<std::string, std::vector<Bar>> series; // symbol -> bars
  // The single base you want to signal on this run (e.g., "QQQ").
  // If empty, we'll infer it from presence (prefers QQQ if present).
  std::string base_symbol;
};

class FeatureFeederGuarded {
public:
  bool initialize(const FeederInit& init);

  const FeatureMatrix& features() const { return X_; }
  const StandardScaler& scaler() const { return scaler_; }
  const std::vector<std::int64_t>& base_ts() const { return base_ts_; }

  // For execution: map instrument rows to base rows
  // (present only for leveraged family members that exist in input)
  const std::vector<int32_t>* asof_map_for(const std::string& symbol) const {
    auto it = asof_.find(to_upper(symbol));
    if (it == asof_.end()) return nullptr;
    return &it->second;
  }

  // True if symbol is permitted for execution in this run (base or leverage family)
  bool allowed_for_exec(const std::string& symbol) const;

  const std::string& base() const { return base_symU_; }

private:
  std::string base_symU_;
  FeatureMatrix X_;
  StandardScaler scaler_;
  std::vector<std::int64_t> base_ts_;
  std::unordered_map<std::string, std::vector<int32_t>> asof_; // SYM -> asof index into base
  std::unordered_map<std::string, std::vector<Bar>> prices_;   // keep original price series

  bool infer_base_if_needed_(const std::unordered_map<std::string, std::vector<Bar>>& series,
                             std::string& base_out);
};

} // namespace sentio

```

## 📄 **FILE 40 of 180**: include/sentio/feature/feature_from_spec.hpp

**File Information**:
- **Path**: `include/sentio/feature/feature_from_spec.hpp`

- **Size**: 264 lines
- **Modified**: 2025-09-10 21:30:21

- **Type**: .hpp

```text
#pragma once
#include <stdexcept>
#include <string>
#include <vector>
#include "sentio/core.hpp"
#include "sentio/feature/feature_matrix.hpp"
#include "sentio/feature/feature_builder_ops.hpp"
#include "sentio/feature/ops.hpp"
#include "sentio/sym/leverage_registry.hpp"

// nlohmann JSON single-header:
#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace sentio {

inline FeatureMatrix build_features_from_spec_json(
  const std::string& symbol,
  const std::vector<Bar>& bars,
  const std::string& spec_json
){
  if (is_leveraged(symbol))
    throw std::invalid_argument("Leveraged symbol not allowed in FeatureBuilder: " + symbol);

  const size_t N = bars.size();
  if (N == 0) return {};

  // load spec
  json spec = json::parse(spec_json);
  const int F = (int)spec["features"].size();
  int emit_from = (int)spec["alignment_policy"]["emit_from_index"];
  float pad = (float)spec["alignment_policy"]["pad_value"];

  // Build source vectors
  std::vector<float> open(N), high(N), low(N), close(N), volume(N);
  std::vector<float> logret; // Built on demand
  
  for (size_t i = 0; i < N; ++i) {
    open[i] = (float)bars[i].open;
    high[i] = (float)bars[i].high;
    low[i] = (float)bars[i].low;
    close[i] = (float)bars[i].close;
    volume[i] = (float)bars[i].volume;
  }

  auto get_source_vector = [&](const std::string& name) -> const std::vector<float>& {
    if (name == "open") return open;
    if (name == "high") return high;
    if (name == "low") return low;
    if (name == "close") return close;
    if (name == "volume") return volume;
    if (name == "logret") {
      if (logret.empty()) logret = op_LOGRET(close);
      return logret;
    }
    // Handle composite sources
    if (name == "hlc" || name == "hl" || name == "ohlc" || name == "ohlcv" || name == "close_volume" || name == "hlcv") {
      return close; // Return close as default, actual multi-source ops handle this specially
    }
    throw std::runtime_error("unknown source: " + name);
  };

  FeatureMatrix M; 
  M.rows = (std::int64_t)N; 
  M.cols = F; 
  M.data.resize(N * F);

  // PERFORMANCE NOTE: Current implementation processes features column-by-column
  // which causes cache misses due to row-major memory layout. For optimal performance,
  // consider refactoring to process row-by-row: calculate all features for row r,
  // then write them contiguously to M.data[r * F + c] before moving to next row.
  // This would require stateful indicator objects (EMA_Calculator, RSI_Calculator, etc.)
  // and significant refactoring of the op_* functions.
  
  for (int c = 0; c < F; ++c) {
    const auto& f = spec["features"][c];
    const std::string op = f["op"];
    const std::string src = f.value("source", "close");
    
    std::vector<float> col;

    // ============================================================================
    // BASIC OPERATIONS
    // ============================================================================
    if (op == "IDENT") {
      const auto& x = get_source_vector(src);
      col = op_IDENT(x);
    }
    else if (op == "LOGRET") {
      const auto& x = get_source_vector(src);
      col = op_LOGRET(x);
    }
    else if (op == "MOMENTUM") {
      const auto& x = get_source_vector(src);
      col = op_MOMENTUM(x, f.value("window", 5));
    }
    else if (op == "ROC") {
      const auto& x = get_source_vector(src);
      col = op_ROC(x, f.value("window", 10));
    }
    
    // ============================================================================
    // MOVING AVERAGES
    // ============================================================================
    else if (op == "SMA") {
      const auto& x = get_source_vector(src);
      col = op_SMA(x, f.value("window", 20));
    }
    else if (op == "EMA") {
      const auto& x = get_source_vector(src);
      col = op_EMA(x, f.value("window", 20));
    }
    
    // ============================================================================
    // VOLATILITY MEASURES
    // ============================================================================
    else if (op == "VOLATILITY") {
      const auto& x = get_source_vector(src);
      col = op_VOLATILITY(x, f.value("window", 20));
    }
    else if (op == "PARKINSON") {
      col = op_PARKINSON(high, low, f.value("window", 14));
    }
    else if (op == "GARMAN_KLASS") {
      col = op_GARMAN_KLASS(open, high, low, close, f.value("window", 14));
    }
    
    // ============================================================================
    // TECHNICAL INDICATORS
    // ============================================================================
    else if (op == "RSI") {
      const auto& x = get_source_vector(src);
      col = op_RSI(x, f.value("window", 14));
    }
    else if (op == "ZWIN") {
      const auto& x = get_source_vector(src);
      col = op_ZWIN(x, f.value("window", 20));
    }
    else if (op == "ATR") {
      col = op_ATR(high, low, close, f.value("window", 14));
    }
    
    // ============================================================================
    // BOLLINGER BANDS
    // ============================================================================
    else if (op == "BOLLINGER_UPPER") {
      const auto& x = get_source_vector(src);
      auto b = op_BOLLINGER(x, f.value("window", 20), f.value("k", 2.0));
      col = std::move(b.upper);
    }
    else if (op == "BOLLINGER_MIDDLE") {
      const auto& x = get_source_vector(src);
      auto b = op_BOLLINGER(x, f.value("window", 20), f.value("k", 2.0));
      col = std::move(b.middle);
    }
    else if (op == "BOLLINGER_LOWER") {
      const auto& x = get_source_vector(src);
      auto b = op_BOLLINGER(x, f.value("window", 20), f.value("k", 2.0));
      col = std::move(b.lower);
    }
    
    // ============================================================================
    // MACD
    // ============================================================================
    else if (op == "MACD_LINE") {
      const auto& x = get_source_vector(src);
      auto m = op_MACD(x, f.value("fast", 12), f.value("slow", 26), f.value("signal", 9));
      col = std::move(m.line);
    }
    else if (op == "MACD_SIGNAL") {
      const auto& x = get_source_vector(src);
      auto m = op_MACD(x, f.value("fast", 12), f.value("slow", 26), f.value("signal", 9));
      col = std::move(m.signal);
    }
    else if (op == "MACD_HISTOGRAM") {
      const auto& x = get_source_vector(src);
      auto m = op_MACD(x, f.value("fast", 12), f.value("slow", 26), f.value("signal", 9));
      col = std::move(m.histogram);
    }
    
    // ============================================================================
    // STOCHASTIC
    // ============================================================================
    else if (op == "STOCHASTIC_K") {
      auto s = op_STOCHASTIC(high, low, close, f.value("window", 14), f.value("d_period", 3));
      col = std::move(s.k);
    }
    else if (op == "STOCHASTIC_D") {
      auto s = op_STOCHASTIC(high, low, close, f.value("window", 14), f.value("d_period", 3));
      col = std::move(s.d);
    }
    
    // ============================================================================
    // OTHER OSCILLATORS
    // ============================================================================
    else if (op == "WILLIAMS_R") {
      col = op_WILLIAMS_R(high, low, close, f.value("window", 14));
    }
    else if (op == "CCI") {
      col = op_CCI(high, low, close, f.value("window", 20));
    }
    else if (op == "ADX") {
      col = op_ADX(high, low, close, f.value("window", 14));
    }
    
    // ============================================================================
    // VOLUME INDICATORS
    // ============================================================================
    else if (op == "OBV") {
      col = op_OBV(close, volume);
    }
    else if (op == "VPT") {
      col = op_VPT(close, volume);
    }
    else if (op == "AD_LINE") {
      col = op_AD_LINE(high, low, close, volume);
    }
    else if (op == "MFI") {
      col = op_MFI(high, low, close, volume, f.value("window", 14));
    }
    
    // ============================================================================
    // MICROSTRUCTURE INDICATORS
    // ============================================================================
    else if (op == "SPREAD_BP") {
      col = op_SPREAD_BP(open, high, low, close);
    }
    else if (op == "PRICE_IMPACT") {
      col = op_PRICE_IMPACT(open, high, low, close, volume);
    }
    else if (op == "ORDER_FLOW") {
      col = op_ORDER_FLOW(open, high, low, close, volume);
    }
    else if (op == "MARKET_DEPTH") {
      col = op_MARKET_DEPTH(open, high, low, close, volume);
    }
    else if (op == "BID_ASK_RATIO") {
      col = op_BID_ASK_RATIO(open, high, low, close);
    }
    
    // ============================================================================
    // FALLBACK
    // ============================================================================
    else {
      throw std::runtime_error("bad op: " + op);
    }

    // Write column to matrix
    for (size_t r = 0; r < N; ++r) {
      M.data[r * F + c] = col[r];
    }
  }

  // Apply padding policy
  for (std::int64_t r = 0; r < std::min<std::int64_t>(emit_from, M.rows); ++r) {
    for (int c = 0; c < F; ++c) {
      M.data[r * F + c] = pad;
    }
  }

  return M;
}

} // namespace sentio

```

## 📄 **FILE 41 of 180**: include/sentio/feature/feature_matrix.hpp

**File Information**:
- **Path**: `include/sentio/feature/feature_matrix.hpp`

- **Size**: 13 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>

namespace sentio {
struct FeatureMatrix {
  std::vector<float> data; // row-major [rows, cols]
  std::int64_t rows{0};
  std::int64_t cols{0};
  inline float* row_ptr(std::int64_t r) { return data.data() + r*cols; }
  inline const float* row_ptr(std::int64_t r) const { return data.data() + r*cols; }
};
}

```

## 📄 **FILE 42 of 180**: include/sentio/feature/feature_provider.hpp

**File Information**:
- **Path**: `include/sentio/feature/feature_provider.hpp`

- **Size**: 20 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace sentio {

struct FeatureMatrix {
  int64_t rows{0}, cols{0};
  std::vector<float> data; // row-major [rows, cols]
};

struct IFeatureProvider {
  virtual ~IFeatureProvider() = default;
  virtual FeatureMatrix get_features_for(const std::string& symbol) = 0;
  virtual std::vector<std::string> feature_names() const = 0; // authoritative order in source
  virtual int seq_len() const = 0; // sequence length (warmup)
};

} // namespace sentio

```

## 📄 **FILE 43 of 180**: include/sentio/feature/name_diff.hpp

**File Information**:
- **Path**: `include/sentio/feature/name_diff.hpp`

- **Size**: 40 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <unordered_set>

namespace sentio {

inline void print_name_diff(const std::vector<std::string>& src, const std::vector<std::string>& dst){
  std::unordered_set<std::string> S(src.begin(), src.end()), D(dst.begin(), dst.end());
  int miss=0, extra=0, reorder=0;

  std::cout << "[DIFF] Feature name differences:" << std::endl;
  for (auto& n : dst) {
    if (!S.count(n)) { 
      miss++; 
      std::cout << "  MISSING: " << n << std::endl; 
    }
  }
  
  for (auto& n : src) {
    if (!D.count(n)) { 
      extra++; 
      std::cout << "  EXTRA  : " << n << std::endl; 
    }
  }

  if (miss==0 && extra==0 && src.size()==dst.size()){
    for (size_t i=0;i<src.size();++i) {
      if (src[i] != dst[i]) reorder++;
    }
    if (reorder>0) {
      std::cout << "  REORDER count: " << reorder << std::endl;
    }
  }
  
  std::cout << "  Summary: missing=" << miss << " extra=" << extra << " reordered=" << reorder << std::endl;
}

} // namespace sentio

```

## 📄 **FILE 44 of 180**: include/sentio/feature/ops.hpp

**File Information**:
- **Path**: `include/sentio/feature/ops.hpp`

- **Size**: 592 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace sentio {

// ============================================================================
// BASIC OPERATIONS
// ============================================================================

inline std::vector<float> op_IDENT(const std::vector<float>& x) { 
    return x; 
}

inline std::vector<float> op_LOGRET(const std::vector<float>& x) {
    std::vector<float> y(x.size(), 0.f);
    for (size_t i = 1; i < x.size(); ++i) {
        float a = std::max(x[i], 1e-12f), b = std::max(x[i-1], 1e-12f);
        y[i] = std::log(a) - std::log(b);
    }
    return y;
}

inline std::vector<float> op_MOMENTUM(const std::vector<float>& x, int w) {
    std::vector<float> out(x.size(), 0.f);
    for (size_t i = w; i < x.size(); ++i) {
        out[i] = x[i] - x[i-w];
    }
    return out;
}

inline std::vector<float> op_ROC(const std::vector<float>& x, int w) {
    std::vector<float> out(x.size(), 0.f);
    for (size_t i = w; i < x.size(); ++i) {
        float prev = std::max(x[i-w], 1e-12f);
        out[i] = ((x[i] - prev) / prev) * 100.0f;
    }
    return out;
}

// ============================================================================
// MOVING AVERAGES
// ============================================================================

inline std::vector<float> op_SMA(const std::vector<float>& x, int w) {
    std::vector<float> out(x.size(), 0.f);
    if (w <= 1) return x;
    double s = 0.0;
    for (int i = 0; i < (int)x.size(); ++i) {
        s += x[i];
        if (i >= w) s -= x[i-w];
        if (i >= w-1) out[i] = (float)(s/w);
    }
    return out;
}

inline std::vector<float> op_EMA(const std::vector<float>& x, int p) {
    std::vector<float> e(x.size()); 
    if (x.empty()) return e;
    float k = 2.f / (p + 1.f); 
    e[0] = x[0];
    for (size_t i = 1; i < x.size(); ++i) {
        e[i] = k * x[i] + (1.f - k) * e[i-1];
    }
    return e;
}

// ============================================================================
// VOLATILITY AND STATISTICAL MEASURES
// ============================================================================

inline std::vector<float> op_VOLATILITY(const std::vector<float>& x, int w) {
    std::vector<float> out(x.size(), 0.f);
    if ((int)x.size() < w) return out;
    double s = 0.0, s2 = 0.0;
    for (int i = 0; i < w; i++) { 
        s += x[i]; 
        s2 += x[i] * x[i]; 
    }
    for (size_t i = w; i < x.size(); ++i) {
        double mu = s / w;
        double var = std::max(0.0, s2 / w - mu * mu);
        out[i] = (float)std::sqrt(var);
        // slide window
        s += x[i] - x[i-w];
        s2 += x[i] * x[i] - x[i-w] * x[i-w];
    }
    return out;
}

inline std::vector<float> op_PARKINSON(const std::vector<float>& high, 
                                       const std::vector<float>& low, 
                                       int w) {
    std::vector<float> out(high.size(), 0.f);
    std::vector<float> hl_ratio(high.size(), 0.f);
    
    // Calculate log(H/L)^2 for each bar
    for (size_t i = 0; i < high.size(); ++i) {
        if (high[i] > 0 && low[i] > 0) {
            float ratio = std::log(high[i] / low[i]);
            hl_ratio[i] = ratio * ratio;
        }
    }
    
    // Rolling average of (log(H/L))^2 and scale by Parkinson constant
    auto avg_hl = op_SMA(hl_ratio, w);
    const float parkinson_factor = 1.0f / (4.0f * std::log(2.0f));
    
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = std::sqrt(avg_hl[i] * parkinson_factor);
    }
    return out;
}

inline std::vector<float> op_GARMAN_KLASS(const std::vector<float>& open,
                                          const std::vector<float>& high,
                                          const std::vector<float>& low,
                                          const std::vector<float>& close,
                                          int w) {
    std::vector<float> out(open.size(), 0.f);
    std::vector<float> gk_vals(open.size(), 0.f);
    
    // Calculate Garman-Klass estimator for each bar
    for (size_t i = 1; i < open.size(); ++i) {
        if (high[i] > 0 && low[i] > 0 && open[i] > 0 && close[i] > 0) {
            float log_hl = std::log(high[i] / low[i]);
            float log_co = std::log(close[i] / open[i]);
            gk_vals[i] = 0.5f * log_hl * log_hl - (2.0f * std::log(2.0f) - 1.0f) * log_co * log_co;
        }
    }
    
    // Rolling average
    auto avg_gk = op_SMA(gk_vals, w);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = std::sqrt(std::max(0.0f, avg_gk[i]));
    }
    return out;
}

// ============================================================================
// TECHNICAL INDICATORS
// ============================================================================

inline std::vector<float> op_RSI(const std::vector<float>& x, int p = 14) {
    const size_t N = x.size();
    std::vector<float> out(N, 0.f), up(N, 0.f), dn(N, 0.f);
    
    for (size_t i = 1; i < N; ++i) {
        float d = x[i] - x[i-1]; 
        up[i] = std::max(d, 0.f); 
        dn[i] = std::max(-d, 0.f);
    }
    
    std::vector<float> ru(N, 0.f), rd(N, 0.f);
    if (N > (size_t)p) {
        float su = 0.f, sd = 0.f;
        for (int i = 1; i <= p; i++) { 
            su += up[i]; 
            sd += dn[i]; 
        }
        ru[p] = su / p; 
        rd[p] = sd / p;
        
        for (size_t i = p + 1; i < N; ++i) {
            ru[i] = (ru[i-1] * (p-1) + up[i]) / p;
            rd[i] = (rd[i-1] * (p-1) + dn[i]) / p;
            float rs = (rd[i] > 1e-12f) ? (ru[i] / rd[i]) : 0.f;
            out[i] = 100.f - 100.f / (1.f + rs);
        }
    }
    return out;
}

inline std::vector<float> op_ZWIN(const std::vector<float>& x, int w) {
    const size_t N = x.size(); 
    std::vector<float> out(N, 0.f);
    if (N <= (size_t)w) return out;
    
    std::vector<double> s(N, 0.0), s2(N, 0.0);
    s[0] = x[0]; 
    s2[0] = x[0] * x[0];
    
    for (size_t i = 1; i < N; ++i) { 
        s[i] = s[i-1] + x[i]; 
        s2[i] = s2[i-1] + x[i] * x[i]; 
    }
    
    for (size_t i = w; i < N; ++i) {
        double su = s[i] - s[i-w], su2 = s2[i] - s2[i-w];
        double mu = su / w;
        double var = std::max(0.0, su2 / w - mu * mu);
        float sd = (float)std::sqrt(var);
        out[i] = (sd > 1e-8f) ? (float)((x[i] - mu) / sd) : 0.f;
    }
    return out;
}

inline std::vector<float> op_ATR(const std::vector<float>& high,
                                 const std::vector<float>& low,
                                 const std::vector<float>& close,
                                 int p) {
    const size_t N = high.size();
    std::vector<float> tr(N, 0.f), atr(N, 0.f);
    
    for (size_t i = 1; i < N; ++i) {
        float h = high[i], l = low[i], cp = close[i-1];
        float v = std::max({h-l, std::fabs(h-cp), std::fabs(l-cp)});
        tr[i] = v;
    }
    
    // Wilder smoothing
    if (N > (size_t)p) {
        float a = 0.f; 
        for (int i = 1; i <= p; i++) a += tr[i];
        a /= p; 
        atr[p] = a;
        for (size_t i = p + 1; i < N; ++i) {
            atr[i] = (atr[i-1] * (p-1) + tr[i]) / p;
        }
    }
    return atr;
}

// ============================================================================
// BOLLINGER BANDS
// ============================================================================

struct BollingerBands { 
    std::vector<float> upper, middle, lower; 
};

inline BollingerBands op_BOLLINGER(const std::vector<float>& x, int w, float k = 2.0f) {
    BollingerBands b; 
    b.upper.resize(x.size(), 0.f); 
    b.middle.resize(x.size(), 0.f); 
    b.lower.resize(x.size(), 0.f);
    
    auto sma = op_SMA(x, w);
    auto sd = op_VOLATILITY(x, w);
    
    for (size_t i = w; i < x.size(); ++i) {
        b.middle[i] = sma[i];
        b.upper[i] = sma[i] + k * sd[i];
        b.lower[i] = sma[i] - k * sd[i];
    }
    return b;
}

// ============================================================================
// MACD
// ============================================================================

struct MACD { 
    std::vector<float> line, signal, histogram; 
};

inline MACD op_MACD(const std::vector<float>& x, int fast = 12, int slow = 26, int sig = 9) {
    MACD m; 
    m.line.resize(x.size()); 
    m.signal.resize(x.size()); 
    m.histogram.resize(x.size());
    
    auto ema_fast = op_EMA(x, fast);
    auto ema_slow = op_EMA(x, slow);
    
    for (size_t i = 0; i < x.size(); ++i) {
        m.line[i] = ema_fast[i] - ema_slow[i];
    }
    
    m.signal = op_EMA(m.line, sig);
    
    for (size_t i = 0; i < x.size(); ++i) {
        m.histogram[i] = m.line[i] - m.signal[i];
    }
    return m;
}

// ============================================================================
// STOCHASTIC OSCILLATOR
// ============================================================================

struct Stochastic {
    std::vector<float> k, d;
};

inline Stochastic op_STOCHASTIC(const std::vector<float>& high,
                                const std::vector<float>& low,
                                const std::vector<float>& close,
                                int k_period = 14,
                                int d_period = 3) {
    Stochastic stoch;
    stoch.k.resize(close.size(), 0.f);
    stoch.d.resize(close.size(), 0.f);
    
    for (size_t i = k_period; i < close.size(); ++i) {
        float highest = *std::max_element(high.begin() + i - k_period, high.begin() + i + 1);
        float lowest = *std::min_element(low.begin() + i - k_period, low.begin() + i + 1);
        
        if (highest > lowest) {
            stoch.k[i] = ((close[i] - lowest) / (highest - lowest)) * 100.0f;
        }
    }
    
    stoch.d = op_SMA(stoch.k, d_period);
    return stoch;
}

// ============================================================================
// OTHER OSCILLATORS
// ============================================================================

inline std::vector<float> op_WILLIAMS_R(const std::vector<float>& high,
                                        const std::vector<float>& low,
                                        const std::vector<float>& close,
                                        int period = 14) {
    std::vector<float> out(close.size(), 0.f);
    
    for (size_t i = period; i < close.size(); ++i) {
        float highest = *std::max_element(high.begin() + i - period, high.begin() + i + 1);
        float lowest = *std::min_element(low.begin() + i - period, low.begin() + i + 1);
        
        if (highest > lowest) {
            out[i] = ((highest - close[i]) / (highest - lowest)) * -100.0f;
        }
    }
    return out;
}

inline std::vector<float> op_CCI(const std::vector<float>& high,
                                 const std::vector<float>& low,
                                 const std::vector<float>& close,
                                 int period = 20) {
    std::vector<float> out(close.size(), 0.f);
    std::vector<float> tp(close.size()); // typical price
    
    for (size_t i = 0; i < close.size(); ++i) {
        tp[i] = (high[i] + low[i] + close[i]) / 3.0f;
    }
    
    auto sma_tp = op_SMA(tp, period);
    
    for (size_t i = period; i < close.size(); ++i) {
        float mean_dev = 0.0f;
        for (size_t j = i - period + 1; j <= i; ++j) {
            mean_dev += std::fabs(tp[j] - sma_tp[i]);
        }
        mean_dev /= period;
        
        if (mean_dev > 1e-8f) {
            out[i] = (tp[i] - sma_tp[i]) / (0.015f * mean_dev);
        }
    }
    return out;
}

inline std::vector<float> op_ADX(const std::vector<float>& high,
                                 const std::vector<float>& low,
                                 const std::vector<float>& close,
                                 int period = 14) {
    const size_t N = close.size();
    std::vector<float> adx(N, 0.f);
    std::vector<float> dm_plus(N, 0.f), dm_minus(N, 0.f), tr(N, 0.f);
    
    // Calculate directional movement and true range
    for (size_t i = 1; i < N; ++i) {
        float up_move = high[i] - high[i-1];
        float down_move = low[i-1] - low[i];
        
        dm_plus[i] = (up_move > down_move && up_move > 0) ? up_move : 0.0f;
        dm_minus[i] = (down_move > up_move && down_move > 0) ? down_move : 0.0f;
        
        float h = high[i], l = low[i], cp = close[i-1];
        tr[i] = std::max({h-l, std::fabs(h-cp), std::fabs(l-cp)});
    }
    
    // Smooth the values using Wilder's smoothing
    if (N > (size_t)period) {
        float sum_dm_plus = 0, sum_dm_minus = 0, sum_tr = 0;
        for (int i = 1; i <= period; i++) {
            sum_dm_plus += dm_plus[i];
            sum_dm_minus += dm_minus[i];
            sum_tr += tr[i];
        }
        
        float smooth_dm_plus = sum_dm_plus;
        float smooth_dm_minus = sum_dm_minus;
        float smooth_tr = sum_tr;
        
        for (size_t i = period + 1; i < N; ++i) {
            smooth_dm_plus = smooth_dm_plus - smooth_dm_plus/period + dm_plus[i];
            smooth_dm_minus = smooth_dm_minus - smooth_dm_minus/period + dm_minus[i];
            smooth_tr = smooth_tr - smooth_tr/period + tr[i];
            
            float di_plus = (smooth_tr > 0) ? (smooth_dm_plus / smooth_tr) * 100 : 0;
            float di_minus = (smooth_tr > 0) ? (smooth_dm_minus / smooth_tr) * 100 : 0;
            
            float di_sum = di_plus + di_minus;
            float dx = (di_sum > 0) ? std::fabs(di_plus - di_minus) / di_sum * 100 : 0;
            
            // Simple moving average of DX for ADX
            if (i >= period * 2) {
                float adx_sum = 0;
                for (size_t j = i - period + 1; j <= i; ++j) {
                    // Recalculate DX for each period (simplified)
                    adx_sum += dx; // This is simplified; full implementation would store DX values
                }
                adx[i] = adx_sum / period;
            }
        }
    }
    return adx;
}

// ============================================================================
// VOLUME INDICATORS
// ============================================================================

inline std::vector<float> op_OBV(const std::vector<float>& close,
                                 const std::vector<float>& volume) {
    std::vector<float> obv(close.size(), 0.f);
    
    for (size_t i = 1; i < close.size(); ++i) {
        if (close[i] > close[i-1]) {
            obv[i] = obv[i-1] + volume[i];
        } else if (close[i] < close[i-1]) {
            obv[i] = obv[i-1] - volume[i];
        } else {
            obv[i] = obv[i-1];
        }
    }
    return obv;
}

inline std::vector<float> op_VPT(const std::vector<float>& close,
                                 const std::vector<float>& volume) {
    std::vector<float> vpt(close.size(), 0.f);
    
    for (size_t i = 1; i < close.size(); ++i) {
        if (close[i-1] > 0) {
            float pct_change = (close[i] - close[i-1]) / close[i-1];
            vpt[i] = vpt[i-1] + volume[i] * pct_change;
        } else {
            vpt[i] = vpt[i-1];
        }
    }
    return vpt;
}

inline std::vector<float> op_AD_LINE(const std::vector<float>& high,
                                     const std::vector<float>& low,
                                     const std::vector<float>& close,
                                     const std::vector<float>& volume) {
    std::vector<float> ad(close.size(), 0.f);
    
    for (size_t i = 1; i < close.size(); ++i) {
        float hl_diff = high[i] - low[i];
        if (hl_diff > 1e-8f) {
            float mfm = ((close[i] - low[i]) - (high[i] - close[i])) / hl_diff;
            float mfv = mfm * volume[i];
            ad[i] = ad[i-1] + mfv;
        } else {
            ad[i] = ad[i-1];
        }
    }
    return ad;
}

inline std::vector<float> op_MFI(const std::vector<float>& high,
                                 const std::vector<float>& low,
                                 const std::vector<float>& close,
                                 const std::vector<float>& volume,
                                 int period = 14) {
    std::vector<float> mfi(close.size(), 0.f);
    std::vector<float> tp(close.size()); // typical price
    std::vector<float> mf(close.size()); // money flow
    
    for (size_t i = 0; i < close.size(); ++i) {
        tp[i] = (high[i] + low[i] + close[i]) / 3.0f;
        mf[i] = tp[i] * volume[i];
    }
    
    for (size_t i = period; i < close.size(); ++i) {
        float positive_mf = 0, negative_mf = 0;
        
        for (size_t j = i - period + 1; j <= i; ++j) {
            if (j > 0) {
                if (tp[j] > tp[j-1]) {
                    positive_mf += mf[j];
                } else if (tp[j] < tp[j-1]) {
                    negative_mf += mf[j];
                }
            }
        }
        
        if (negative_mf > 1e-8f) {
            float mfr = positive_mf / negative_mf;
            mfi[i] = 100.0f - (100.0f / (1.0f + mfr));
        }
    }
    return mfi;
}

// ============================================================================
// MICROSTRUCTURE INDICATORS (Simplified implementations)
// ============================================================================

inline std::vector<float> op_SPREAD_BP(const std::vector<float>& open,
                                       const std::vector<float>& high,
                                       const std::vector<float>& low,
                                       const std::vector<float>& close) {
    std::vector<float> spread(close.size(), 0.f);
    
    for (size_t i = 0; i < close.size(); ++i) {
        float mid = (high[i] + low[i]) / 2.0f;
        if (mid > 1e-8f) {
            spread[i] = ((high[i] - low[i]) / mid) * 10000.0f; // basis points
        }
    }
    return spread;
}

inline std::vector<float> op_PRICE_IMPACT(const std::vector<float>& open,
                                          const std::vector<float>& high,
                                          const std::vector<float>& low,
                                          const std::vector<float>& close,
                                          const std::vector<float>& volume) {
    std::vector<float> impact(close.size(), 0.f);
    
    for (size_t i = 1; i < close.size(); ++i) {
        float price_change = std::fabs(close[i] - close[i-1]);
        float vol_sqrt = std::sqrt(volume[i]);
        if (vol_sqrt > 1e-8f && close[i-1] > 1e-8f) {
            impact[i] = (price_change / close[i-1]) / vol_sqrt * 1000000.0f; // scaled
        }
    }
    return impact;
}

inline std::vector<float> op_ORDER_FLOW(const std::vector<float>& open,
                                        const std::vector<float>& high,
                                        const std::vector<float>& low,
                                        const std::vector<float>& close,
                                        const std::vector<float>& volume) {
    std::vector<float> flow(close.size(), 0.f);
    
    for (size_t i = 0; i < close.size(); ++i) {
        float hl_diff = high[i] - low[i];
        if (hl_diff > 1e-8f) {
            float buy_pressure = (close[i] - low[i]) / hl_diff;
            float sell_pressure = (high[i] - close[i]) / hl_diff;
            flow[i] = (buy_pressure - sell_pressure) * volume[i];
        }
    }
    return flow;
}

inline std::vector<float> op_MARKET_DEPTH(const std::vector<float>& open,
                                          const std::vector<float>& high,
                                          const std::vector<float>& low,
                                          const std::vector<float>& close,
                                          const std::vector<float>& volume) {
    std::vector<float> depth(close.size(), 0.f);
    
    for (size_t i = 0; i < close.size(); ++i) {
        float range = high[i] - low[i];
        if (volume[i] > 1e-8f && range > 1e-8f) {
            depth[i] = range / std::log(volume[i] + 1.0f);
        }
    }
    return depth;
}

inline std::vector<float> op_BID_ASK_RATIO(const std::vector<float>& open,
                                          const std::vector<float>& high,
                                          const std::vector<float>& low,
                                          const std::vector<float>& close) {
    std::vector<float> ratio(close.size(), 0.f);
    
    for (size_t i = 0; i < close.size(); ++i) {
        // Simplified: use close position within range as proxy
        float range = high[i] - low[i];
        if (range > 1e-8f) {
            float position = (close[i] - low[i]) / range; // 0 = low, 1 = high
            ratio[i] = position / (1.0f - position + 1e-8f); // bid/ask proxy
        }
    }
    return ratio;
}

} // namespace sentio

```

## 📄 **FILE 45 of 180**: include/sentio/feature/sanitize.hpp

**File Information**:
- **Path**: `include/sentio/feature/sanitize.hpp`

- **Size**: 63 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>

namespace sentio {

// NaN/Inf sanitation before model (and a ready mask)
inline std::vector<uint8_t> sanitize_and_ready(float* X, int64_t rows, int64_t cols, int emit_from, float pad=0.0f){
  std::vector<uint8_t> ok((size_t)rows, 0);
  
  // Pad rows before emit_from
  for (int64_t r=0; r<std::min<int64_t>(emit_from, rows); ++r){
    for (int64_t c=0;c<cols;++c) X[r*cols+c] = pad;
  }
  
  // Sanitize and check rows from emit_from onward
  for (int64_t r=emit_from; r<rows; ++r){
    bool good=true;
    float* row = X + r*cols;
    for (int64_t c=0;c<cols;++c){
      float v=row[c];
      if (!std::isfinite(v)){ 
        row[c]=0.0f; 
        good=false; 
      }
    }
    ok[(size_t)r] = good ? 1 : 0;
  }
  return ok;
}

// Overload for cached features (vector<vector<double>>)
inline std::vector<uint8_t> sanitize_cached_features(std::vector<std::vector<double>>& features, int emit_from, double pad=0.0){
  std::vector<uint8_t> ok(features.size(), 0);
  
  if (features.empty()) return ok;
  
  size_t cols = features[0].size();
  
  // Pad rows before emit_from
  for (size_t r=0; r<std::min<size_t>(emit_from, features.size()); ++r){
    for (size_t c=0; c<cols; ++c) {
      features[r][c] = pad;
    }
  }
  
  // Sanitize and check rows from emit_from onward
  for (size_t r=emit_from; r<features.size(); ++r){
    bool good=true;
    for (size_t c=0; c<features[r].size(); ++c){
      double v = features[r][c];
      if (!std::isfinite(v)){ 
        features[r][c] = 0.0; 
        good=false; 
      }
    }
    ok[r] = good ? 1 : 0;
  }
  return ok;
}

} // namespace sentio

```

## 📄 **FILE 46 of 180**: include/sentio/feature/standard_scaler.hpp

**File Information**:
- **Path**: `include/sentio/feature/standard_scaler.hpp`

- **Size**: 42 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace sentio {

struct StandardScaler {
  std::vector<float> mean, inv_std;

  void fit(const float* X, std::int64_t rows, std::int64_t cols) {
    mean.assign(cols, 0.f);
    inv_std.assign(cols, 0.f);

    for (std::int64_t r=0; r<rows; ++r)
      for (std::int64_t c=0; c<cols; ++c)
        mean[c] += X[r*cols + c];
    for (std::int64_t c=0; c<cols; ++c) mean[c] /= std::max<std::int64_t>(1, rows);

    std::vector<double> var(cols, 0.0);
    for (std::int64_t r=0; r<rows; ++r)
      for (std::int64_t c=0; c<cols; ++c) {
        const double d = (double)X[r*cols + c] - (double)mean[c];
        var[c] += d*d;
      }
    for (std::int64_t c=0; c<cols; ++c) {
      const double sd = std::sqrt(std::max(1e-12, var[c] / std::max<std::int64_t>(1, rows)));
      inv_std[c] = (float)(1.0 / sd);
    }
  }

  void transform_inplace(float* X, std::int64_t rows, std::int64_t cols) const {
    for (std::int64_t r=0; r<rows; ++r)
      for (std::int64_t c=0; c<cols; ++c) {
        float& v = X[r*cols + c];
        v = (v - mean[c]) * inv_std[c];
      }
  }
};

} // namespace sentio

```

## 📄 **FILE 47 of 180**: include/sentio/feature_builder.hpp

**File Information**:
- **Path**: `include/sentio/feature_builder.hpp`

- **Size**: 150 lines
- **Modified**: 2025-09-13 14:55:09

- **Type**: .hpp

```text
#pragma once
#include "sentio/core.hpp"
#include <deque>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <optional>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sentio {

// Optional microstructure snapshot (pass when you have it; else omit)
struct MicroTick { double bid{std::numeric_limits<double>::quiet_NaN()}, ask{std::numeric_limits<double>::quiet_NaN()}; };

// A compiled plan for the features (from metadata)
struct FeaturePlan {
  std::vector<std::string> names; // in metadata order
  // sanity: names.size() must match metadata.feature_names
};

// Builder config
struct FeatureBuilderCfg {
  int rsi_period{14};
  int sma_fast{10};
  int sma_slow{30};
  int ret_5m_window{5};      // number of 1m bars
  // If you want volatility as stdev of 1m returns over N bars, set vol_window>1
  int vol_window{20};        // stdev window (bars)
  // spread fallback (bps) when no bid/ask and proxy not computable
  double default_spread_bp{1.5};
};

// Rolling helpers (small, header-only for speed)
class RollingMean {
  double sum_{0}; std::deque<double> q_;
  size_t W_;
public:
  explicit RollingMean(size_t W): W_(W) {}
  void push(double x){ sum_ += x; q_.push_back(x); if(q_.size()>W_){ sum_-=q_.front(); q_.pop_front(); } }
  bool full() const { return q_.size()==W_; }
  double mean() const { return q_.empty()? std::numeric_limits<double>::quiet_NaN() : (sum_/double(q_.size())); }
  size_t size() const { return q_.size(); }
};

class RollingStdWindow {
  std::vector<double> buf_;
  size_t W_, i_{0}, n_{0};
  double sum_{0}, sumsq_{0};
public:
  explicit RollingStdWindow(size_t W): buf_(W, 0.0), W_(W) {}
  inline void push(double x){
    if (n_ < W_) { 
      buf_[n_++] = x; 
      sum_ += x; 
      sumsq_ += x*x; 
      if (n_ == W_) i_ = 0; 
    } else { 
      double old = buf_[i_]; 
      buf_[i_] = x; 
      sum_ += x - old; 
      sumsq_ += x*x - old*old; 
      if (++i_ == W_) i_ = 0; 
    }
  }
  inline bool full() const { return n_ == W_; }
  inline double stdev() const { 
    if (n_ < 2) return std::numeric_limits<double>::quiet_NaN(); 
    double m = sum_ / n_; 
    return std::sqrt(std::max(0.0, sumsq_ / n_ - m * m)); 
  }
  inline size_t size() const { return n_; }
};

class RollingRSI {
  // Wilder's RSI with smoothing; requires first 'period' values to bootstrap
  int period_; bool boot_{true}; int boot_count_{0};
  double up_{0}, dn_{0};
public:
  explicit RollingRSI(int p): period_(p) {}
  // x = current close, px = previous close
  void push(double px, double x){
    double chg = x - px;
    double u = chg>0? chg:0; double d = chg<0? -chg:0;
    if (boot_){
      up_ += u; dn_ += d; ++boot_count_;
      if (boot_count_ == period_) {
        up_ /= period_; dn_ /= period_; boot_ = false;
      }
    } else {
      up_ = (up_*(period_-1) + u) / period_;
      dn_ = (dn_*(period_-1) + d) / period_;
    }
  }
  bool ready() const { return !boot_; }
  double value() const {
    if (boot_) return std::numeric_limits<double>::quiet_NaN();
    if (dn_==0) return 100.0;
    double rs = up_/dn_;
    return 100.0 - 100.0/(1.0+rs);
  }
};

class FeatureBuilder {
public:
  FeatureBuilder(FeaturePlan plan, FeatureBuilderCfg cfg);

  // Feed one 1m bar (RTH-filtered) plus optional bid/ask for spread
  void on_bar(const Bar& b, const std::optional<MicroTick>& mt = std::nullopt);

  // True when all requested features can be computed *and* are finite
  bool ready() const;

  // Returns features in the exact metadata order (size == plan.names.size()).
  // Will return std::nullopt if not ready().
  std::optional<std::vector<double>> build() const;

  // Resets internal buffers
  void reset();

  // Accessors (useful in tests)
  size_t bars_seen() const { return bars_seen_; }

private:
  FeaturePlan plan_;
  FeatureBuilderCfg cfg_;

  // Internal state
  size_t bars_seen_{0};
  std::deque<double> close_q_;             // last N closes for ret/RSI
  RollingMean sma_fast_, sma_slow_;
  RollingStdWindow vol_rtn_;               // stdev of 1m returns (O(1) implementation)
  RollingRSI  rsi_;

  // Cached per-bar computations
  double last_ret_1m_{std::numeric_limits<double>::quiet_NaN()};
  double last_ret_5m_{std::numeric_limits<double>::quiet_NaN()};
  double last_rsi_{std::numeric_limits<double>::quiet_NaN()};
  double last_sma_fast_{std::numeric_limits<double>::quiet_NaN()};
  double last_sma_slow_{std::numeric_limits<double>::quiet_NaN()};
  double last_vol_1m_{std::numeric_limits<double>::quiet_NaN()};
  double last_spread_bp_{std::numeric_limits<double>::quiet_NaN()};

  // helpers
  static inline bool finite(double x){ return std::isfinite(x); }
};

} // namespace sentio

```

## 📄 **FILE 48 of 180**: include/sentio/feature_cache.hpp

**File Information**:
- **Path**: `include/sentio/feature_cache.hpp`

- **Size**: 62 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace sentio {

/**
 * FeatureCache loads and provides access to pre-computed features
 * This eliminates the need for expensive real-time technical indicator calculations
 */
class FeatureCache {
public:
    /**
     * Load pre-computed features from CSV file
     * @param feature_file_path Path to the feature CSV file (e.g., QQQ_RTH_features.csv)
     * @return true if loaded successfully
     */
    bool load_from_csv(const std::string& feature_file_path);

    /**
     * Get features for a specific bar index
     * @param bar_index The bar index (0-based)
     * @return Vector of 55 features, or empty vector if not found
     */
    std::vector<double> get_features(int bar_index) const;

    /**
     * Check if features are available for a given bar index
     */
    bool has_features(int bar_index) const;

    /**
     * Get the total number of bars with features
     */
    size_t get_bar_count() const;

    /**
     * Get the recommended starting bar index (after warmup)
     */
    int get_recommended_start_bar() const;

    /**
     * Get feature names in order
     */
    const std::vector<std::string>& get_feature_names() const;

private:
    // Map from bar_index to feature vector
    std::unordered_map<int, std::vector<double>> features_by_bar_;
    
    // Feature names in order
    std::vector<std::string> feature_names_;
    
    // Recommended starting bar (after warmup period)
    int recommended_start_bar_ = 300;
    
    // Total number of bars
    size_t total_bars_ = 0;
};

} // namespace sentio

```

## 📄 **FILE 49 of 180**: include/sentio/feature_engineering/feature_normalizer.hpp

**File Information**:
- **Path**: `include/sentio/feature_engineering/feature_normalizer.hpp`

- **Size**: 70 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <deque>
#include <mutex>

namespace sentio {
namespace feature_engineering {

struct NormalizationStats {
    double mean{0.0};
    double std{0.0};
    double min{0.0};
    double max{0.0};
    size_t count{0};
};

class FeatureNormalizer {
public:
    FeatureNormalizer(size_t window_size = 252);
    
    // Normalization methods
    std::vector<double> normalize_features(const std::vector<double>& features);
    std::vector<double> denormalize_features(const std::vector<double>& normalized_features);
    
    // Statistics management
    void update_stats(const std::vector<double>& features);
    void reset_stats();
    
    // Feature-specific normalization
    std::vector<double> z_score_normalize(const std::vector<double>& features);
    std::vector<double> min_max_normalize(const std::vector<double>& features);
    std::vector<double> robust_normalize(const std::vector<double>& features);
    
    // Outlier handling
    std::vector<double> clip_outliers(const std::vector<double>& features, double threshold = 3.0);
    std::vector<double> winsorize(const std::vector<double>& features, double percentile = 0.05);
    
    // Validation
    bool is_normalized(const std::vector<double>& features) const;
    std::vector<bool> get_outlier_mask(const std::vector<double>& features, double threshold = 3.0) const;
    
    // Statistics access
    NormalizationStats get_stats(size_t feature_index) const;
    std::vector<NormalizationStats> get_all_stats() const;
    
    // Configuration
    void set_window_size(size_t window_size);
    void set_outlier_threshold(double threshold);
    void set_winsorize_percentile(double percentile);
    
private:
    size_t window_size_;
    double outlier_threshold_{3.0};
    double winsorize_percentile_{0.05};
    
    std::vector<std::deque<double>> feature_history_;
    std::vector<NormalizationStats> stats_;
    mutable std::mutex stats_mutex_;
    
    void update_feature_stats(size_t feature_index, double value);
    double calculate_robust_mean(const std::deque<double>& values);
    double calculate_robust_std(const std::deque<double>& values, double mean);
    double calculate_percentile(const std::deque<double>& values, double percentile);
    void sort_values(std::deque<double>& values);
};

} // namespace feature_engineering
} // namespace sentio

```

## 📄 **FILE 50 of 180**: include/sentio/feature_engineering/kochi_features.hpp

**File Information**:
- **Path**: `include/sentio/feature_engineering/kochi_features.hpp`

- **Size**: 21 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .hpp

```text
#pragma once
#include "sentio/core.hpp"
#include <vector>
#include <string>

namespace sentio {
namespace feature_engineering {

// Returns Kochi feature names in the exact order expected by the trainer.
// This excludes any state features (position one-hot, PnL), which are not used at inference time.
std::vector<std::string> kochi_feature_names();

// Compute Kochi feature vector for a given bar index using bar history.
// Window-dependent features use typical Kochi defaults (e.g., 20 for many).
// The output order matches kochi_feature_names().
std::vector<double> calculate_kochi_features(const std::vector<Bar>& bars, int current_index);

} // namespace feature_engineering
} // namespace sentio



```

## 📄 **FILE 51 of 180**: include/sentio/feature_engineering/technical_indicators.hpp

**File Information**:
- **Path**: `include/sentio/feature_engineering/technical_indicators.hpp`

- **Size**: 127 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "sentio/core.hpp"
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace sentio {
namespace feature_engineering {

// Price-based features
struct PriceFeatures {
    double ret_1m{0.0}, ret_5m{0.0}, ret_15m{0.0}, ret_30m{0.0}, ret_1h{0.0};
    double momentum_5{0.0}, momentum_10{0.0}, momentum_20{0.0};
    double volatility_10{0.0}, volatility_20{0.0}, volatility_30{0.0};
    double atr_14{0.0}, atr_21{0.0};
    double parkinson_vol{0.0}, garman_klass_vol{0.0};
};

// Technical analysis features
struct TechnicalFeatures {
    double rsi_14{0.0}, rsi_21{0.0}, rsi_30{0.0};
    double sma_5{0.0}, sma_10{0.0}, sma_20{0.0}, sma_50{0.0}, sma_200{0.0};
    double ema_5{0.0}, ema_10{0.0}, ema_20{0.0}, ema_50{0.0}, ema_200{0.0};
    double bb_upper_20{0.0}, bb_middle_20{0.0}, bb_lower_20{0.0};
    double bb_upper_50{0.0}, bb_middle_50{0.0}, bb_lower_50{0.0};
    double macd_line{0.0}, macd_signal{0.0}, macd_histogram{0.0};
    double stoch_k{0.0}, stoch_d{0.0};
    double williams_r{0.0};
    double cci_20{0.0};
    double adx_14{0.0};
};

// Volume features
struct VolumeFeatures {
    double volume_sma_10{0.0}, volume_sma_20{0.0}, volume_sma_50{0.0};
    double volume_roc{0.0};
    double obv{0.0};
    double vpt{0.0};
    double ad_line{0.0};
    double mfi_14{0.0};
};

// Market microstructure features
struct MicrostructureFeatures {
    double spread_bp{0.0};
    double price_impact{0.0};
    double order_flow_imbalance{0.0};
    double market_depth{0.0};
    double bid_ask_ratio{0.0};
};

// Main feature calculator
class TechnicalIndicatorCalculator {
public:
    TechnicalIndicatorCalculator();
    
    // Core calculation methods
    PriceFeatures calculate_price_features(const std::vector<Bar>& bars, int current_index);
    TechnicalFeatures calculate_technical_features(const std::vector<Bar>& bars, int current_index);
    VolumeFeatures calculate_volume_features(const std::vector<Bar>& bars, int current_index);
    MicrostructureFeatures calculate_microstructure_features(const std::vector<Bar>& bars, int current_index);
    
    // Combined feature vector
    std::vector<double> calculate_all_features(const std::vector<Bar>& bars, int current_index);
    
    // Feature validation
    bool validate_features(const std::vector<double>& features);
    std::vector<std::string> get_feature_names() const;
    
    // Helper methods
    static std::vector<double> extract_closes(const std::vector<Bar>& bars);
    static std::vector<double> extract_volumes(const std::vector<Bar>& bars);
    static std::vector<double> extract_returns(const std::vector<Bar>& bars);
    
private:
    // Rolling calculations
    double calculate_rsi(const std::vector<double>& closes, int period, int current_index);
    double calculate_sma(const std::vector<double>& values, int period, int current_index);
    double calculate_ema(const std::vector<double>& values, int period, int current_index);
    double calculate_volatility(const std::vector<double>& returns, int period, int current_index);
    double calculate_atr(const std::vector<Bar>& bars, int period, int current_index);
    
    // Bollinger Bands
    struct BollingerBands {
        double upper, middle, lower;
    };
    BollingerBands calculate_bollinger_bands(const std::vector<double>& values, int period, double std_dev, int current_index);
    
    // MACD
    struct MACD {
        double line, signal, histogram;
    };
    MACD calculate_macd(const std::vector<double>& values, int fast, int slow, int signal, int current_index);
    
    // Stochastic
    struct Stochastic {
        double k, d;
    };
    Stochastic calculate_stochastic(const std::vector<Bar>& bars, int k_period, int d_period, int current_index);
    
    // Williams %R
    double calculate_williams_r(const std::vector<Bar>& bars, int period, int current_index);
    
    // CCI
    double calculate_cci(const std::vector<Bar>& bars, int period, int current_index);
    
    // ADX
    double calculate_adx(const std::vector<Bar>& bars, int period, int current_index);
    
    // Volume indicators
    double calculate_obv(const std::vector<Bar>& bars, int current_index);
    double calculate_vpt(const std::vector<Bar>& bars, int current_index);
    double calculate_ad_line(const std::vector<Bar>& bars, int current_index);
    double calculate_mfi(const std::vector<Bar>& bars, int period, int current_index);
    
    // Volatility indicators
    double calculate_parkinson_volatility(const std::vector<Bar>& bars, int period, int current_index);
    double calculate_garman_klass_volatility(const std::vector<Bar>& bars, int period, int current_index);
    
    // Feature names for validation
    std::vector<std::string> feature_names_;
};

} // namespace feature_engineering
} // namespace sentio

```

## 📄 **FILE 52 of 180**: include/sentio/feature_feeder.hpp

**File Information**:
- **Path**: `include/sentio/feature_feeder.hpp`

- **Size**: 107 lines
- **Modified**: 2025-09-12 22:05:37

- **Type**: .hpp

```text
#pragma once
#include "sentio/core.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/feature_engineering/technical_indicators.hpp"
#include "sentio/feature_engineering/feature_normalizer.hpp"
#include "sentio/feature_cache.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <mutex>

namespace sentio {

struct FeatureMetrics {
    std::chrono::microseconds extraction_time{0};
    size_t features_extracted{0};
    size_t features_valid{0};
    size_t features_invalid{0};
    double extraction_rate{0.0}; // features per second
    std::chrono::steady_clock::time_point last_update;
};

struct FeatureHealthReport {
    bool is_healthy{false};
    std::vector<bool> feature_health;
    std::vector<double> feature_quality_scores;
    std::string health_summary;
    double overall_health_score{0.0};
};

class FeatureFeeder {
public:
    // Core functionality
    static std::vector<double> extract_features_from_bar(const Bar& bar, const std::string& strategy_name);
    static std::vector<double> extract_features_from_bars_with_index(const std::vector<Bar>& bars, int current_index, const std::string& strategy_name);
    static bool is_ml_strategy(const std::string& strategy_name);
    static void feed_features_to_strategy(BaseStrategy* strategy, const std::vector<Bar>& bars, int current_index, const std::string& strategy_name);
    
    // Enhanced functionality
    static void initialize_strategy(const std::string& strategy_name);
    static void cleanup_strategy(const std::string& strategy_name);
    
    // **STRATEGY ISOLATION**: Clear all state to prevent cross-strategy contamination
    static void reset_all_state();
    
    // Feature management
    static std::vector<double> get_cached_features(const std::string& strategy_name);
    static void cache_features(const std::string& strategy_name, const std::vector<double>& features);
    static void invalidate_cache(const std::string& strategy_name);
    
    // Performance monitoring
    static FeatureMetrics get_metrics(const std::string& strategy_name);
    static FeatureHealthReport get_health_report(const std::string& strategy_name);
    static void reset_metrics(const std::string& strategy_name);
    
    // Feature validation
    static bool validate_features(const std::vector<double>& features, const std::string& strategy_name);
    static std::vector<std::string> get_feature_names(const std::string& strategy_name);
    
    // Configuration
    static void set_feature_config(const std::string& strategy_name, const std::string& config_key, const std::string& config_value);
    static std::string get_feature_config(const std::string& strategy_name, const std::string& config_key);
    
    // Cached features (for performance)
    static bool load_feature_cache(const std::string& feature_file_path);
    static bool use_cached_features(bool enable = true);
    static bool has_cached_features();
    
    // Batch processing
    static std::vector<std::vector<double>> extract_features_from_bars(const std::vector<Bar>& bars, const std::string& strategy_name);
    static void feed_features_batch(BaseStrategy* strategy, const std::vector<Bar>& bars, const std::string& strategy_name);
    
    // Feature analysis
    static std::vector<double> get_feature_correlation(const std::string& strategy_name);
    static std::vector<double> get_feature_importance(const std::string& strategy_name);
    static void log_feature_performance(const std::string& strategy_name);
    
private:
    // Strategy-specific data
    struct StrategyData {
        std::unique_ptr<feature_engineering::TechnicalIndicatorCalculator> calculator;
        std::unique_ptr<feature_engineering::FeatureNormalizer> normalizer;
        std::vector<double> cached_features;
        FeatureMetrics metrics;
        std::chrono::steady_clock::time_point last_update;
        bool initialized{false};
        std::unordered_map<std::string, std::string> config;
    };
    
    static std::unordered_map<std::string, StrategyData> strategy_data_;
    static std::mutex data_mutex_;
    
    // Feature cache for performance
    static std::unique_ptr<FeatureCache> feature_cache_;
    static bool use_cached_features_;
    
    // Helper methods
    static StrategyData& get_strategy_data(const std::string& strategy_name);
    static void update_metrics(StrategyData& data, const std::vector<double>& features, std::chrono::microseconds extraction_time);
    static FeatureHealthReport calculate_health_report(const StrategyData& data, const std::vector<double>& features);
    static std::vector<std::string> get_strategy_feature_names(const std::string& strategy_name);
    static void initialize_strategy_data(const std::string& strategy_name);
};

} // namespace sentio
```

## 📄 **FILE 53 of 180**: include/sentio/feature_utils.hpp

**File Information**:
- **Path**: `include/sentio/feature_utils.hpp`

- **Size**: 56 lines
- **Modified**: 2025-09-10 13:55:46

- **Type**: .hpp

```text
#pragma once

#include <string>
#include <vector>
#include <algorithm>

namespace sentio {

/**
 * Utility functions for feature operations
 */
class FeatureUtils {
public:
    /**
     * Find strategy data by symbol in a map-like container
     * @param strategy_data The strategy data container
     * @param symbol The symbol to find
     * @return iterator to the found element or end() if not found
     */
    template<typename Container>
    static auto find_strategy_data(Container& strategy_data, const std::string& symbol) {
        return std::find_if(strategy_data.begin(), strategy_data.end(),
            [&symbol](const auto& pair) {
                return pair.first == symbol;
            });
    }
    
    /**
     * Check if strategy data exists for a symbol
     * @param strategy_data The strategy data container
     * @param symbol The symbol to check
     * @return true if data exists for the symbol
     */
    template<typename Container>
    static bool has_strategy_data(const Container& strategy_data, const std::string& symbol) {
        return find_strategy_data(strategy_data, symbol) != strategy_data.end();
    }
    
    /**
     * Get strategy data for a symbol, creating if it doesn't exist
     * @param strategy_data The strategy data container
     * @param symbol The symbol to get/create
     * @return reference to the strategy data
     */
    template<typename Container, typename DataType>
    static DataType& get_or_create_strategy_data(Container& strategy_data, const std::string& symbol) {
        auto it = find_strategy_data(strategy_data, symbol);
        if (it == strategy_data.end()) {
            strategy_data.emplace_back(symbol, DataType{});
            return strategy_data.back().second;
        }
        return it->second;
    }
};

} // namespace sentio

```

## 📄 **FILE 54 of 180**: include/sentio/future_qqq_loader.hpp

**File Information**:
- **Path**: `include/sentio/future_qqq_loader.hpp`

- **Size**: 95 lines
- **Modified**: 2025-09-14 12:01:50

- **Type**: .hpp

```text
#pragma once

#include "sentio/core.hpp"
#include <string>
#include <vector>
#include <map>

namespace sentio {

/**
 * @brief Loader for pre-generated future QQQ data files
 * 
 * This class provides access to the 10 pre-generated future QQQ tracks,
 * each representing 4 weeks (28 days) of different market regimes.
 * 
 * Track Distribution:
 * - Tracks 1, 4, 7, 10: Normal regime (4 tracks)
 * - Tracks 2, 5, 8: Volatile regime (3 tracks)  
 * - Tracks 3, 6, 9: Trending regime (3 tracks)
 */
class FutureQQQLoader {
public:
    /**
     * @brief Market regime types available in future QQQ data
     */
    enum class Regime {
        NORMAL,
        VOLATILE, 
        TRENDING
    };

    /**
     * @brief Load a specific future QQQ track
     * @param track_id Track ID (1-10)
     * @return Vector of bars for the track
     */
    static std::vector<Bar> load_track(int track_id);

    /**
     * @brief Load a random track for the specified regime
     * @param regime Market regime to load
     * @param seed Random seed for reproducible selection (optional)
     * @return Vector of bars for a random track of the specified regime
     */
    static std::vector<Bar> load_regime_track(Regime regime, int seed = -1);

    /**
     * @brief Load a random track for the specified regime (string version)
     * @param regime_str Market regime string ("normal", "volatile", "trending")
     * @param seed Random seed for reproducible selection (optional)
     * @return Vector of bars for a random track of the specified regime
     */
    static std::vector<Bar> load_regime_track(const std::string& regime_str, int seed = -1);

    /**
     * @brief Get all track IDs for a specific regime
     * @param regime Market regime
     * @return Vector of track IDs for the regime
     */
    static std::vector<int> get_regime_tracks(Regime regime);

    /**
     * @brief Convert regime string to enum
     * @param regime_str Regime string ("normal", "volatile", "trending")
     * @return Regime enum value
     */
    static Regime string_to_regime(const std::string& regime_str);

    /**
     * @brief Get the base directory for future QQQ data
     * @return Path to future QQQ data directory
     */
    static std::string get_data_directory();

    /**
     * @brief Check if all future QQQ tracks are available
     * @return True if all 10 tracks are accessible
     */
    static bool validate_tracks();

private:
    /**
     * @brief Get the file path for a specific track
     * @param track_id Track ID (1-10)
     * @return Full path to the CSV file
     */
    static std::string get_track_file_path(int track_id);

    /**
     * @brief Regime to track mapping
     */
    static const std::map<Regime, std::vector<int>> regime_tracks_;
};

} // namespace sentio

```

## 📄 **FILE 55 of 180**: include/sentio/global_leverage_config.hpp

**File Information**:
- **Path**: `include/sentio/global_leverage_config.hpp`

- **Size**: 21 lines
- **Modified**: 2025-09-14 03:59:59

- **Type**: .hpp

```text
#pragma once

namespace sentio {

// Global configuration for leverage pricing system
class GlobalLeverageConfig {
private:
    static bool use_theoretical_leverage_pricing_;
    
public:
    // Enable theoretical leverage pricing globally (default: true)
    static void enable_theoretical_leverage_pricing(bool enable = true) {
        use_theoretical_leverage_pricing_ = enable;
    }
    
    static bool is_theoretical_leverage_pricing_enabled() {
        return use_theoretical_leverage_pricing_;
    }
};

} // namespace sentio

```

## 📄 **FILE 56 of 180**: include/sentio/indicators.hpp

**File Information**:
- **Path**: `include/sentio/indicators.hpp`

- **Size**: 52 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <deque>
#include <cmath>
#include <limits>

namespace sentio {

struct SMA {
  int n{0};
  double sum{0.0};
  std::deque<double> q;
  explicit SMA(int n_) : n(n_) {}
  void reset(){ sum=0; q.clear(); }
  void push(double x){
    if (!std::isfinite(x)) { reset(); return; }
    q.push_back(x); sum += x;
    if ((int)q.size() > n) { sum -= q.front(); q.pop_front(); }
  }
  bool ready() const { return (int)q.size() == n; }
  double value() const { return ready() ? sum / n : std::numeric_limits<double>::quiet_NaN(); }
};

struct RSI {
  int n{14};
  int warm{0};
  double avgGain{0}, avgLoss{0}, prev{NAN};
  explicit RSI(int n_=14):n(n_),warm(0),avgGain(0),avgLoss(0),prev(NAN){}
  void reset(){ warm=0; avgGain=avgLoss=0; prev=NAN; }
  void push(double close){
    if (!std::isfinite(close)) { reset(); return; }
    if (!std::isfinite(prev)) { prev = close; return; }
    double delta = close - prev; prev = close;
    double gain = delta > 0 ? delta : 0.0;
    double loss = delta < 0 ? -delta : 0.0;
    if (warm < n) {
      avgGain += gain; avgLoss += loss; ++warm;
      if (warm==n) { avgGain/=n; avgLoss/=n; }
    } else {
      avgGain = (avgGain*(n-1) + gain)/n;
      avgLoss = (avgLoss*(n-1) + loss)/n;
    }
  }
  bool ready() const { return warm >= n; }
  double value() const {
    if (!ready()) return std::numeric_limits<double>::quiet_NaN();
    if (avgLoss == 0) return 100.0;
    double rs = avgGain/avgLoss;
    return 100.0 - (100.0/(1.0+rs));
  }
};

} // namespace sentio

```

## 📄 **FILE 57 of 180**: include/sentio/leverage_aware_csv_loader.hpp

**File Information**:
- **Path**: `include/sentio/leverage_aware_csv_loader.hpp`

- **Size**: 47 lines
- **Modified**: 2025-09-14 02:16:23

- **Type**: .hpp

```text
#pragma once
#include "sentio/csv_loader.hpp"
#include "sentio/leverage_pricing.hpp"
#include "sentio/data_resolver.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace sentio {

// Drop-in replacement for load_csv that uses theoretical pricing for leverage instruments
bool load_csv_leverage_aware(const std::string& symbol, std::vector<Bar>& out);

// Load multiple symbols with leverage-aware pricing
bool load_family_csv_leverage_aware(const std::vector<std::string>& symbols,
                                   std::unordered_map<std::string, std::vector<Bar>>& series_out);

// Load QQQ family with theoretical leverage pricing
bool load_qqq_family_leverage_aware(std::unordered_map<std::string, std::vector<Bar>>& series_out);

// Global configuration for leverage pricing
class LeveragePricingConfig {
private:
    static LeverageCostModel cost_model_;
    static bool use_theoretical_pricing_;
    
public:
    // Enable/disable theoretical pricing globally
    static void enable_theoretical_pricing(bool enable = true) {
        use_theoretical_pricing_ = enable;
    }
    
    static bool is_theoretical_pricing_enabled() {
        return use_theoretical_pricing_;
    }
    
    // Update the global cost model
    static void set_cost_model(const LeverageCostModel& model) {
        cost_model_ = model;
    }
    
    static const LeverageCostModel& get_cost_model() {
        return cost_model_;
    }
};

} // namespace sentio

```

## 📄 **FILE 58 of 180**: include/sentio/leverage_pricing.hpp

**File Information**:
- **Path**: `include/sentio/leverage_pricing.hpp`

- **Size**: 111 lines
- **Modified**: 2025-09-14 02:16:23

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "sentio/sym/leverage_registry.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <random>

namespace sentio {

// Leverage cost model parameters
struct LeverageCostModel {
    double expense_ratio{0.0095};        // 0.95% annual expense ratio for TQQQ/SQQQ
    double borrowing_cost_rate{0.05};    // 5% annual borrowing cost (varies with interest rates)
    double daily_decay_factor{0.0001};   // Additional daily decay from rebalancing friction
    double bid_ask_spread{0.0005};       // 0.05% bid-ask spread cost per rebalance
    double tracking_error_std{0.0002};   // Daily tracking error standard deviation
    
    // Calculate total daily cost rate
    double daily_cost_rate() const {
        return (expense_ratio + borrowing_cost_rate) / 252.0 + daily_decay_factor + bid_ask_spread;
    }
};

// Theoretical leverage pricing engine
class TheoreticalLeveragePricer {
private:
    LeverageCostModel cost_model_;
    std::unordered_map<std::string, double> last_prices_; // Track last prices for continuity
    std::mt19937 rng_;
    
public:
    TheoreticalLeveragePricer(const LeverageCostModel& cost_model = LeverageCostModel{});
    
    // Generate theoretical leverage price based on base price movement
    double calculate_theoretical_price(const std::string& leverage_symbol,
                                     double base_price_prev,
                                     double base_price_current,
                                     double leverage_price_prev);
    
    // Generate full theoretical bar from base bar
    Bar generate_theoretical_bar(const std::string& leverage_symbol,
                                const Bar& base_bar_prev,
                                const Bar& base_bar_current,
                                const Bar& leverage_bar_prev);
    
    // Generate theoretical price series from base series
    std::vector<Bar> generate_theoretical_series(const std::string& leverage_symbol,
                                                const std::vector<Bar>& base_series,
                                                double initial_price = 0.0);
    
    // Update cost model (e.g., for different interest rate environments)
    void update_cost_model(const LeverageCostModel& new_model) { cost_model_ = new_model; }
    
    // Get current cost model
    const LeverageCostModel& get_cost_model() const { return cost_model_; }
};

// Theoretical pricing validator - compares theoretical vs actual prices
class LeveragePricingValidator {
private:
    TheoreticalLeveragePricer pricer_;
    
public:
    struct ValidationResult {
        std::string symbol;
        double price_correlation;
        double return_correlation;
        double mean_price_error;
        double price_error_std;
        double mean_return_error;
        double return_error_std;
        double theoretical_total_return;
        double actual_total_return;
        double return_difference;
        int num_observations;
    };
    
    LeveragePricingValidator(const LeverageCostModel& cost_model = LeverageCostModel{});
    
    // Validate theoretical pricing against actual data
    ValidationResult validate_pricing(const std::string& leverage_symbol,
                                    const std::vector<Bar>& base_series,
                                    const std::vector<Bar>& actual_leverage_series);
    
    // Run comprehensive validation report
    void print_validation_report(const ValidationResult& result);
    
    // Calibrate cost model to minimize pricing errors
    LeverageCostModel calibrate_cost_model(const std::string& leverage_symbol,
                                         const std::vector<Bar>& base_series,
                                         const std::vector<Bar>& actual_leverage_series);
};

// Leverage-aware data loader that generates theoretical prices
class LeverageAwareDataLoader {
private:
    TheoreticalLeveragePricer pricer_;
    
public:
    LeverageAwareDataLoader(const LeverageCostModel& cost_model = LeverageCostModel{});
    
    // Load data with theoretical leverage pricing
    bool load_symbol_data(const std::string& symbol, std::vector<Bar>& out);
    
    // Load multiple symbols with theoretical leverage pricing
    bool load_family_data(const std::vector<std::string>& symbols,
                         std::unordered_map<std::string, std::vector<Bar>>& series_out);
};

} // namespace sentio

```

## 📄 **FILE 59 of 180**: include/sentio/mars_data_loader.hpp

**File Information**:
- **Path**: `include/sentio/mars_data_loader.hpp`

- **Size**: 123 lines
- **Modified**: 2025-09-12 11:17:38

- **Type**: .hpp

```text
#pragma once

#include "sentio/core.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>

namespace sentio {

/**
 * MarsDataLoader - Loads market data generated by MarS (Microsoft Research Market Simulation Engine)
 * 
 * This class provides integration between MarS-generated realistic market data
 * and our C++ virtual market testing system.
 */
class MarsDataLoader {
public:
    struct MarsBar {
        std::time_t timestamp;
        double open;
        double high;
        double low;
        double close;
        double volume;
        std::string symbol;
    };

    /**
     * Load market data from MarS-generated JSON file
     * 
     * @param filename Path to JSON file generated by MarS bridge
     * @return Vector of MarsBar objects
     */
    static std::vector<MarsBar> load_from_json(const std::string& filename);
    
    /**
     * Convert MarsBar to our standard Bar format
     * 
     * @param mars_bar MarS bar data
     * @return Standard Bar object
     */
    static Bar convert_to_bar(const MarsBar& mars_bar);
    
    /**
     * Convert vector of MarsBar to vector of Bar
     * 
     * @param mars_bars Vector of MarS bars
     * @return Vector of standard Bar objects
     */
    static std::vector<Bar> convert_to_bars(const std::vector<MarsBar>& mars_bars);
    
    /**
     * Generate MarS data using Python bridge
     * 
     * @param symbol Symbol to generate data for
     * @param duration_minutes Duration in minutes
     * @param bar_interval_seconds Bar interval in seconds
     * @param num_simulations Number of simulations
     * @param market_regime Market regime ("normal", "volatile", "trending", "bear")
     * @param output_file Output JSON file path
     * @return True if successful
     */
    static bool generate_mars_data(const std::string& symbol,
                                 int duration_minutes,
                                 int bar_interval_seconds,
                                 int num_simulations,
                                 const std::string& market_regime,
                                 const std::string& output_file);
    
    /**
     * Generate fast historical data using optimized bridge
     * 
     * @param symbol Symbol to generate data for
     * @param historical_data_file Path to historical CSV data
     * @param continuation_minutes Minutes to continue after historical data
     * @param output_file Output JSON file path
     * @return True if successful
     */
    static bool generate_fast_historical_data(const std::string& symbol,
                                            const std::string& historical_data_file,
                                            int continuation_minutes,
                                            const std::string& output_file);
    
    /**
     * Load and convert MarS data in one step
     * 
     * @param symbol Symbol to generate data for
     * @param duration_minutes Duration in minutes
     * @param bar_interval_seconds Bar interval in seconds
     * @param num_simulations Number of simulations
     * @param market_regime Market regime
     * @return Vector of standard Bar objects
     */
    static std::vector<Bar> load_mars_data(const std::string& symbol,
                                         int duration_minutes,
                                         int bar_interval_seconds,
                                         int num_simulations,
                                         const std::string& market_regime);

    /**
     * Load and convert fast historical data in one step
     * 
     * @param symbol Symbol to generate data for
     * @param historical_data_file Path to historical CSV data
     * @param continuation_minutes Minutes to continue after historical data
     * @return Vector of standard Bar objects
     */
    static std::vector<Bar> load_fast_historical_data(const std::string& symbol,
                                                     const std::string& historical_data_file,
                                                     int continuation_minutes);

private:
    /**
     * Execute Python command to generate MarS data
     * 
     * @param command Python command to execute
     * @return True if command executed successfully
     */
    static bool execute_python_command(const std::string& command);
};

} // namespace sentio

```

## 📄 **FILE 60 of 180**: include/sentio/metrics.hpp

**File Information**:
- **Path**: `include/sentio/metrics.hpp`

- **Size**: 123 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <utility>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <string>

namespace sentio {
struct RunSummary {
  int bars{}, trades{};
  double ret_total{}, ret_ann{}, vol_ann{}, sharpe{}, mdd{};
  double monthly_proj{}, daily_trades{};
};

inline RunSummary compute_metrics(const std::vector<std::pair<std::string,double>>& daily_equity,
                                  int fills_count) {
  RunSummary s{};
  if (daily_equity.size() < 2) return s;
  s.bars = (int)daily_equity.size();
  std::vector<double> rets; rets.reserve(daily_equity.size()-1);
  for (size_t i=1;i<daily_equity.size();++i) {
    double e0=daily_equity[i-1].second, e1=daily_equity[i].second;
    rets.push_back(e0>0 ? (e1/e0-1.0) : 0.0);
  }
  double mean = std::accumulate(rets.begin(),rets.end(),0.0)/rets.size();
  double var=0.0; for (double r:rets){ double d=r-mean; var+=d*d; } var/=std::max<size_t>(1,rets.size());
  double sd = std::sqrt(var);
  s.ret_ann = mean * 252.0;
  s.vol_ann = sd * std::sqrt(252.0);
  s.sharpe  = (s.vol_ann>1e-12)? s.ret_ann/s.vol_ann : 0.0;
  double e0 = daily_equity.front().second, e1 = daily_equity.back().second;
  s.ret_total = e0>0 ? (e1/e0-1.0) : 0.0;
  s.monthly_proj = std::pow(1.0 + s.ret_ann, 1.0/12.0) - 1.0;
  s.trades = fills_count;
  s.daily_trades = (s.bars>0) ? (double)s.trades / (double)s.bars : 0.0;
  s.mdd = 0.0; // TODO: compute drawdown if you track running peaks
  return s;
}

// Day-aware metrics computed from bar-level equity series by compressing to day closes
inline RunSummary compute_metrics_day_aware(
    const std::vector<std::pair<std::string,double>>& equity_steps,
    int fills_count) {
  RunSummary s{};
  if (equity_steps.size() < 2) {
    s.trades = fills_count;
    s.daily_trades = 0.0;
    return s;
  }

  // Compress to day closes using ts_utc prefix YYYY-MM-DD
  std::vector<double> day_close;
  day_close.reserve(equity_steps.size() / 300 + 2);

  std::string last_day = equity_steps.front().first.size() >= 10
                         ? equity_steps.front().first.substr(0,10)
                         : std::string{};
  double cur = equity_steps.front().second;

  for (size_t i = 1; i < equity_steps.size(); ++i) {
    const auto& ts = equity_steps[i].first;
    const std::string day = ts.size() >= 10 ? ts.substr(0,10) : last_day;
    if (day != last_day) {
      day_close.push_back(cur); // close of previous day
      last_day = day;
    }
    cur = equity_steps[i].second; // latest equity for current day
  }
  day_close.push_back(cur); // close of final day

  const int D = static_cast<int>(day_close.size());
  s.bars = D;

  if (D < 2) {
    s.trades = fills_count;
    s.daily_trades = 0.0;
    s.ret_total = 0.0; s.ret_ann = 0.0; s.vol_ann = 0.0; s.sharpe = 0.0; s.mdd = 0.0;
    s.monthly_proj = 0.0;
    return s;
  }

  // Daily simple returns
  std::vector<double> r; r.reserve(D - 1);
  for (int i = 1; i < D; ++i) {
    double prev = day_close[i-1];
    double next = day_close[i];
    r.push_back(prev > 0.0 ? (next/prev - 1.0) : 0.0);
  }

  // Mean and variance
  double mean = 0.0; for (double x : r) mean += x; mean /= r.size();
  double var  = 0.0; for (double x : r) { double d = x - mean; var += d*d; } var /= r.size();
  double sd = std::sqrt(var);

  // Annualization on daily series
  s.vol_ann = sd * std::sqrt(252.0);
  double e0 = day_close.front(), e1 = day_close.back();
  s.ret_total = e0 > 0.0 ? (e1/e0 - 1.0) : 0.0;
  double years = (D - 1) / 252.0;
  s.ret_ann = (years > 0.0) ? (std::pow(1.0 + s.ret_total, 1.0/years) - 1.0) : 0.0;
  s.sharpe = (sd > 1e-12) ? (mean / sd) * std::sqrt(252.0) : 0.0;
  s.monthly_proj = std::pow(1.0 + s.ret_ann, 1.0/12.0) - 1.0;

  // Trades
  s.trades = fills_count;
  s.daily_trades = (D > 0) ? static_cast<double>(fills_count) / static_cast<double>(D) : 0.0;

  // Max drawdown on day closes
  double peak = day_close.front();
  double max_dd = 0.0;
  for (int i = 1; i < D; ++i) {
    peak = std::max(peak, day_close[i-1]);
    if (peak > 0.0) {
      double dd = (day_close[i] - peak) / peak; // negative when below peak
      if (dd < max_dd) max_dd = dd;
    }
  }
  s.mdd = -max_dd;
  return s;
}
} // namespace sentio


```

## 📄 **FILE 61 of 180**: include/sentio/metrics/mpr.hpp

**File Information**:
- **Path**: `include/sentio/metrics/mpr.hpp`

- **Size**: 55 lines
- **Modified**: 2025-09-15 00:42:29

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace sentio::metrics {

inline double safe_log1p(double x) {
    // guard tiny negatives from rounding
    if (x <= -1.0) {
        throw std::domain_error("Return <= -100% encountered in log1p.");
    }
    return std::log1p(x);
}

struct MprParams {
    int trading_days_per_month = 21; // equities default
};

inline double compute_mpr_from_daily_returns(
    const std::vector<double>& daily_simple_returns,
    const MprParams& params = {}
) {
    if (daily_simple_returns.empty()) return 0.0;

    long double sum_log = 0.0L;
    for (double r : daily_simple_returns) {
        sum_log += static_cast<long double>(safe_log1p(r));
    }
    const long double mean_log = sum_log / static_cast<long double>(daily_simple_returns.size());
    const long double geo_daily = std::expm1(mean_log); // e^{mean_log} - 1
    const long double mpr = std::pow(1.0L + geo_daily, params.trading_days_per_month) - 1.0L;
    return static_cast<double>(mpr);
}

// Convenience: from equity curve (close-to-close)
inline double compute_mpr_from_equity(
    const std::vector<double>& daily_equity, // length N >= 2
    const MprParams& params = {}
) {
    if (daily_equity.size() < 2) return 0.0;
    std::vector<double> rets;
    rets.reserve(daily_equity.size() - 1);
    for (size_t i = 1; i < daily_equity.size(); ++i) {
        double prev = daily_equity[i-1], cur = daily_equity[i];
        if (!(std::isfinite(prev) && std::isfinite(cur)) || prev <= 0.0) {
            throw std::runtime_error("Non-finite or non-positive equity in compute_mpr_from_equity");
        }
        rets.push_back((cur / prev) - 1.0);
    }
    return compute_mpr_from_daily_returns(rets, params);
}

} // namespace sentio::metrics

```

## 📄 **FILE 62 of 180**: include/sentio/metrics/session_utils.hpp

**File Information**:
- **Path**: `include/sentio/metrics/session_utils.hpp`

- **Size**: 37 lines
- **Modified**: 2025-09-15 00:42:29

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <vector>
#include <set>
#include <cstdint>

namespace sentio::metrics {

// Convert UTC timestamp (milliseconds) to NYSE session date (YYYY-MM-DD)
inline std::string timestamp_to_session_date(std::int64_t ts_millis, const std::string& calendar = "XNYS") {
    // For simplicity, assume NYSE calendar (UTC-5/UTC-4 depending on DST)
    // In production, you'd want proper timezone handling
    std::time_t time_t = ts_millis / 1000;
    
    // Convert to Eastern Time (approximate - doesn't handle DST perfectly)
    // For production use, consider using a proper timezone library
    time_t -= 5 * 3600; // UTC-5 (EST) - this is a simplification
    
    std::tm* tm_info = std::gmtime(&time_t);
    std::ostringstream oss;
    oss << std::put_time(tm_info, "%Y-%m-%d");
    return oss.str();
}

// Count unique trading sessions in a vector of timestamps
inline int count_trading_days(const std::vector<std::int64_t>& timestamps, const std::string& calendar = "XNYS") {
    std::set<std::string> unique_dates;
    for (std::int64_t ts : timestamps) {
        unique_dates.insert(timestamp_to_session_date(ts, calendar));
    }
    return static_cast<int>(unique_dates.size());
}

} // namespace sentio::metrics

```

## 📄 **FILE 63 of 180**: include/sentio/ml/feature_pipeline.hpp

**File Information**:
- **Path**: `include/sentio/ml/feature_pipeline.hpp`

- **Size**: 40 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "iml_model.hpp"
#include <vector>
#include <string>
#include <optional>
#include <cmath>

namespace sentio::ml {

struct FeaturePipeline {
  // Pre-sized to spec.feature_names.size()
  std::vector<float> buf;

  explicit FeaturePipeline(const ModelSpec& spec)
  : buf(spec.feature_names.size(), 0.0f), spec_(&spec) {}

  // raw must match spec.feature_names order/length
  // Applies (x-mean)/std then clips to [clip_lo, clip_hi]
  // Returns pointer to internal buffer if successful, nullptr if failed
  const std::vector<float>* transform(const std::vector<double>& raw) {
    auto N = spec_->feature_names.size();
    if (raw.size()!=N) return nullptr;
    const double lo = spec_->clip2.size()==2 ? spec_->clip2[0] : -5.0;
    const double hi = spec_->clip2.size()==2 ? spec_->clip2[1] :  5.0;
    for (size_t i=0;i<N;++i) {
      double x = raw[i];
      double m = (i<spec_->mean.size()? spec_->mean[i] : 0.0);
      double s = (i<spec_->std.size()?  spec_->std[i]  : 1.0);
      double z = s>0 ? (x-m)/s : x-m;
      if (z<lo) z=lo; if (z>hi) z=hi;
      buf[i] = static_cast<float>(z);
    }
    return &buf;
  }

private:
  const ModelSpec* spec_;
};

} // namespace sentio::ml

```

## 📄 **FILE 64 of 180**: include/sentio/ml/feature_window.hpp

**File Information**:
- **Path**: `include/sentio/ml/feature_window.hpp`

- **Size**: 84 lines
- **Modified**: 2025-09-18 12:07:15

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <deque>
#include <optional>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sentio::ml {

struct WindowSpec {
  int seq_len{64};
  int feat_dim{0};
  std::vector<double> mean, std, clip2; // clip2=[lo,hi]
  // layout: "BTF" or "BF"
  std::string layout{"BTF"};
};

class FeatureWindow {
public:
  explicit FeatureWindow(const WindowSpec& s) : spec_(s) {
    // Validate mean/std lengths match feat_dim to prevent segfaults
    // Temporarily disabled for debugging
    // if ((int)spec_.mean.size() != spec_.feat_dim || (int)spec_.std.size() != spec_.feat_dim) {
    //   throw std::runtime_error("FeatureWindow: mean/std length mismatch feat_dim");
    // }
  }
  
  // push raw features in metadata order (length == feat_dim)
  void push(const std::vector<double>& raw) {
    
    if ((int)raw.size() != spec_.feat_dim) {
      // Diagnostic for size mismatch
      return;
    }
    
    buf_.push_back(raw);
    if ((int)buf_.size() > spec_.seq_len) buf_.pop_front();
    
  }
  
  bool ready() const { return (int)buf_.size() == spec_.seq_len; }

  // Return normalized/ clipped tensor as float vector for ONNX input.
  // For "BTF": size = 1*T*F; For "BF": size = 1*(T*F)
  // Returns reused buffer (no allocations in hot path)
  std::optional<std::vector<float>> to_input() const {
    if (!ready()) return std::nullopt;
    const double lo = spec_.clip2.size() == 2 ? spec_.clip2[0] : -5.0;
    const double hi = spec_.clip2.size() == 2 ? spec_.clip2[1] : 5.0;

    // Pre-size buffer once (no allocations in hot path)
    const size_t need = size_t(spec_.seq_len) * size_t(spec_.feat_dim);
    if (norm_buf_.size() != need) norm_buf_.assign(need, 0.0f);

    auto norm = [&](double x, int i) -> float {
      double m = (i < (int)spec_.mean.size() ? spec_.mean[i] : 0.0);
      double s = (i < (int)spec_.std.size() ? spec_.std[i] : 1.0);
      double z = (s > 0 ? (x - m) / s : (x - m));
      if (z < lo) z = lo; 
      if (z > hi) z = hi;
      return (float)z;
    };

    // Fill row-major [T, F] into reusable buffer (no new allocations)
    for (int t = 0; t < spec_.seq_len; ++t) {
      const auto& r = buf_[t];
      for (int f = 0; f < spec_.feat_dim; ++f) {
        norm_buf_[t * spec_.feat_dim + f] = norm(r[f], f);
      }
    }
    return std::make_optional(norm_buf_);
  }

  int seq_len() const { return spec_.seq_len; }
  int feat_dim() const { return spec_.feat_dim; }

private:
  WindowSpec spec_;
  std::deque<std::vector<double>> buf_;
  mutable std::vector<float> norm_buf_;   // REUSABLE normalized buffer
};

} // namespace sentio::ml

```

## 📄 **FILE 65 of 180**: include/sentio/ml/iml_model.hpp

**File Information**:
- **Path**: `include/sentio/ml/iml_model.hpp`

- **Size**: 40 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>
#include <optional>

namespace sentio::ml {

// Raw model output abstraction: either class probs or a scalar score.
struct ModelOutput {
  std::vector<float> probs;   // e.g., [p_sell, p_hold, p_buy] for discrete
  float score{0.0f};          // for regressors
};

struct ModelSpec {
  std::string model_id;       // "TransAlpha", "HybridPPO"
  std::string version;        // "v1"
  std::vector<std::string> feature_names;
  std::vector<double> mean, std, clip2; // clip2: [lo, hi]
  std::vector<std::string> actions;     // e.g., SELL, HOLD, BUY
  int expected_spacing_sec{60};
  std::string instrument_family;        // e.g., "QQQ"
  std::string notes;                    // optional metadata
  // Sequence model extensions
  int seq_len{1};  // 1 for non-sequence models
  std::string input_layout{"BTF"};  // "BTF" for batch-time-feature, "BF" for flattened
  std::string format{"torchscript"};  // "torchscript", "onnx", etc.
};

// Runtime inference model
class IModel {
public:
  virtual ~IModel() = default;
  virtual const ModelSpec& spec() const = 0;
  // features must match spec().feature_names length and order
  // T, F, layout parameters for sequence models
  virtual std::optional<ModelOutput> predict(const std::vector<float>& features,
                                             int T, int F, const std::string& layout) const = 0;
};

} // namespace sentio::ml
```

## 📄 **FILE 66 of 180**: include/sentio/ml/model_registry.hpp

**File Information**:
- **Path**: `include/sentio/ml/model_registry.hpp`

- **Size**: 20 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "iml_model.hpp"
#include <memory>

namespace sentio::ml {

struct ModelHandle {
  std::unique_ptr<IModel> model;
  ModelSpec spec;
};

class ModelRegistryTS {
public:
  static ModelHandle load_torchscript(const std::string& model_id,
                                      const std::string& version,
                                      const std::string& artifacts_dir,
                                      bool use_cuda = false);
};

} // namespace sentio::ml
```

## 📄 **FILE 67 of 180**: include/sentio/ml/ts_model.hpp

**File Information**:
- **Path**: `include/sentio/ml/ts_model.hpp`

- **Size**: 31 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "iml_model.hpp"
#include <memory>

namespace torch { namespace jit { class Module; } }

namespace sentio::ml {

class TorchScriptModel final : public IModel {
public:
  static std::unique_ptr<TorchScriptModel> load(const std::string& pt_path,
                                                const ModelSpec& spec,
                                                bool use_cuda = false);

  const ModelSpec& spec() const override { return spec_; }
  std::optional<ModelOutput> predict(const std::vector<float>& features,
                                     int T, int F, const std::string& layout) const override;

  ~TorchScriptModel();

private:
  explicit TorchScriptModel(ModelSpec spec);
  ModelSpec spec_;
  std::shared_ptr<torch::jit::Module> mod_;
  // Preallocated input tensor & shape (PIMPL pattern)
  mutable void* input_tensor_;  // torch::Tensor (hidden in .cpp)
  mutable std::vector<int64_t> in_shape_;
  bool cuda_{false};
};

} // namespace sentio::ml

```

## 📄 **FILE 68 of 180**: include/sentio/of_index.hpp

**File Information**:
- **Path**: `include/sentio/of_index.hpp`

- **Size**: 36 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include "core.hpp"
#include "orderflow_types.hpp"

namespace sentio {

struct BarTickSpan { 
    int start=-1, end=-1; // [start,end) ticks for this bar
};

inline std::vector<BarTickSpan> build_tick_spans(const std::vector<Bar>& bars,
                                                 const std::vector<Tick>& ticks)
{
    const int N = (int)bars.size();
    const int M = (int)ticks.size();
    std::vector<BarTickSpan> span(N);

    int i = 0, k = 0;
    int cur_start = 0;

    // assume bars have strictly increasing ts; ticks nondecreasing
    for (; i < N; ++i) {
        const int64_t ts = bars[i].ts_utc_epoch;
        // advance k until tick.ts > ts
        while (k < M && ticks[k].ts_utc_epoch <= ts) ++k;
        span[i].start = cur_start;
        span[i].end   = k;        // [cur_start, k) are ticks for bar i
        cur_start = k;
    }
    return span;
}

} // namespace sentio
```

## 📄 **FILE 69 of 180**: include/sentio/of_precompute.hpp

**File Information**:
- **Path**: `include/sentio/of_precompute.hpp`

- **Size**: 26 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "of_index.hpp"

namespace sentio {

inline void precompute_mid_imb(const std::vector<Tick>& ticks,
                               std::vector<double>& mid,
                               std::vector<double>& imb)
{
    const int M = (int)ticks.size();
    mid.resize(M);
    imb.resize(M);
    for (int k=0; k<M; ++k) {
        double m = (ticks[k].bid_px + ticks[k].ask_px) * 0.5;
        double a = std::max(0.0, ticks[k].ask_sz);
        double b = std::max(0.0, ticks[k].bid_sz);
        double d = a + b;
        mid[k] = m;
        imb[k] = (d > 0.0) ? (a / d) : 0.5;   // neutral if zero depth
    }
}

} // namespace sentio
```

## 📄 **FILE 70 of 180**: include/sentio/orderflow_types.hpp

**File Information**:
- **Path**: `include/sentio/orderflow_types.hpp`

- **Size**: 34 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include "core.hpp"

namespace sentio {

struct Tick {
    int64_t ts_utc_epoch;     // strictly nondecreasing
    double  bid_px, ask_px;
    double  bid_sz, ask_sz;   // L1 sizes (or synthetic from L2)
    // (Optional: trade prints, aggressor flags, etc.)
};

struct OFParams {
    // Signal
    double  min_imbalance = 0.65;     // (ask_sz / (ask_sz + bid_sz)) for long
    int     look_ticks    = 50;       // rolling window
    // Risk
    int     hold_max_ticks = 800;     // hard cap per trade
    double  tp_ticks       = 4.0;     // TP in ticks (half-spread units if you like)
    double  sl_ticks       = 4.0;     // SL in ticks
    // Execution
    double  tick_size      = 0.01;
};

struct Trade {
    int start_k=-1, end_k=-1;  // tick indices
    double entry_px=0, exit_px=0;
    int dir=0;                 // +1 long, -1 short
};

} // namespace sentio
```

## 📄 **FILE 71 of 180**: include/sentio/pnl_accounting.hpp

**File Information**:
- **Path**: `include/sentio/pnl_accounting.hpp`

- **Size**: 24 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
// pnl_accounting.hpp
#pragma once
#include "core.hpp"  // Use Bar from core.hpp
#include <string>
#include <stdexcept>
#include <unordered_map>

namespace sentio {

class PriceBook {
public:
  // instrument -> latest bar (or map<ts,bar> for full history)
  const Bar* get_latest(const std::string& instrument) const;
  
  // Additional helper methods
  void upsert_latest(const std::string& instrument, const Bar& b);
  bool has_instrument(const std::string& instrument) const;
  std::size_t size() const;
};

// Use the instrument actually traded
double last_trade_price(const PriceBook& book, const std::string& instrument);

} // namespace sentio
```

## 📄 **FILE 72 of 180**: include/sentio/polygon_client.hpp

**File Information**:
- **Path**: `include/sentio/polygon_client.hpp`

- **Size**: 31 lines
- **Modified**: 2025-09-16 20:14:43

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>

namespace sentio {
struct AggBar { long long ts_ms; double open, high, low, close, volume; };

struct AggsQuery {
  std::string symbol;
  int multiplier{1};
  std::string timespan{"day"}; // "day","hour","minute"
  std::string from, to;
  bool adjusted{true};
  std::string sort{"asc"};
  int limit{50000};
};

class PolygonClient {
public:
  explicit PolygonClient(std::string api_key);
  std::vector<AggBar> get_aggs_all(const AggsQuery& q, int max_pages=200);
  void write_csv(const std::string& out_path,const std::string& symbol,
                 const std::vector<AggBar>& bars, bool exclude_holidays=false, bool rth_only=false);

private:
  std::string api_key_;
  std::string get_(const std::string& url);
  std::vector<AggBar> get_aggs_chunked(const AggsQuery& q, int max_pages=200);
};
} // namespace sentio


```

## 📄 **FILE 73 of 180**: include/sentio/portfolio/fee_model.hpp

**File Information**:
- **Path**: `include/sentio/portfolio/fee_model.hpp`

- **Size**: 21 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <cstdint>

namespace sentio {

struct TradeCtx {
  double price;       // mid/exec price
  double notional;    // |shares| * price
  long   shares;      // signed
  bool   is_short;    // for borrow fees if modeled
};

class IFeeModel {
public:
  virtual ~IFeeModel() = default;
  virtual double commission(const TradeCtx& t) const = 0;  // $ cost
  virtual double exchange_fees(const TradeCtx& t) const { return 0.0; }
  virtual double borrow_fee_daily_bp(double notional_short) const { return 0.0; }
};

} // namespace sentio

```

## 📄 **FILE 74 of 180**: include/sentio/portfolio/portfolio_allocator.hpp

**File Information**:
- **Path**: `include/sentio/portfolio/portfolio_allocator.hpp`

- **Size**: 115 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "tc_slippage_model.hpp"
#include "fee_model.hpp"
#include "sentio/alpha/sota_linear_policy.hpp"

namespace sentio {

struct InstrumentInputs {
  // Inputs per instrument i
  double p_up;            // probability next bar up [0,1]
  double price;           // last price
  double vol_1d;          // daily vol (sigma)
  double spread_bp;       // bid-ask spread bps
  double adv_notional;    // ADV in $
  double w_prev;          // previous target weight
  double pos_weight;      // live position weight (for turnover)
  bool   tradable{true};  // liquidity/halts
};

struct AllocConfig {
  // SOTA linear policy config
  sentio::alpha::SotaPolicyCfg sota;
  
  // Portfolio-level constraints
  double max_gross       = 1.50;  // portfolio gross cap (150%)
  bool   long_only       = false; // allow shorts?
  
  // Legacy parameters (for backward compatibility)
  double risk_aversion   = 5.0;   // maps to sota.gamma
  double tc_lambda       = 2.0;   // maps to sota.lam_tc
  double max_weight_abs  = 0.20;  // maps to sota.max_abs_w
  double min_edge_bp     = 2.0;   // maps to sota.min_edge
};

struct AllocOutput {
  std::vector<double> w_target;     // target weights per instrument
  std::vector<double> edge_bp;      // per-instrument modeled edge bps
  double gross{0.0};
  double net_gross{0.0};
};

class PortfolioAllocator {
public:
  PortfolioAllocator(TCModel tc, const IFeeModel& fee) : tc_(tc), fee_(fee) {}

  AllocOutput allocate(const std::vector<InstrumentInputs>& X, const AllocConfig& cfg) const {
    const int N = (int)X.size();
    AllocOutput out; out.w_target.assign(N, 0.0); out.edge_bp.assign(N, 0.0);

    // Sync legacy config to SOTA policy config
    auto sota_cfg = cfg.sota;
    sota_cfg.gamma = cfg.risk_aversion;
    sota_cfg.lam_tc = cfg.tc_lambda * 1e-4; // convert to weight units
    sota_cfg.max_abs_w = cfg.max_weight_abs;
    sota_cfg.min_edge = cfg.min_edge_bp;

    // Prepare inputs for SOTA linear policy
    std::vector<double> p_up(N), sigma(N), w_prev(N), cost_bps(N);
    
    for (int i = 0; i < N; i++) {
      const auto& x = X[i];
      if (!x.tradable || !std::isfinite(x.p_up)) {
        p_up[i] = 0.5; // neutral if not tradable
        sigma[i] = 0.01; // small vol
        w_prev[i] = x.w_prev;
        cost_bps[i] = 1000.0; // high cost to discourage
        continue;
      }

      p_up[i] = x.p_up;
      sigma[i] = std::max(1e-6, x.vol_1d / std::sqrt(252.0)); // daily to per-bar vol
      w_prev[i] = x.w_prev;
      
      // Estimate total cost (slippage + fees)
      double notional_est = x.price * 1000.0; // estimate for small trade
      double slippage = tc_.slippage_bp(notional_est, {x.adv_notional, x.spread_bp, x.vol_1d});
      TradeCtx trade_ctx{x.price, notional_est, 1000, false};
      double fees_bp = 1e4 * (fee_.commission(trade_ctx) + fee_.exchange_fees(trade_ctx)) / notional_est;
      cost_bps[i] = slippage + fees_bp;
      
      // Store edge for reporting
      double mu = sentio::alpha::prob_to_mu(x.p_up, sigma[i], sota_cfg.k_ret);
      out.edge_bp[i] = 1e4 * mu - cost_bps[i]; // net edge in bps
    }

    // **SOTA LINEAR POLICY**: Merton/Kelly + Gârleanu-Pedersen
    out.w_target = sentio::alpha::sota_multi_asset_weights(p_up, sigma, w_prev, cost_bps, sota_cfg);
    
    // Apply long-only constraint if needed
    if (cfg.long_only) {
      for (double& w : out.w_target) w = std::max(0.0, w);
    }

    // Apply portfolio gross constraint
    double gross = 0.0; for (double w : out.w_target) gross += std::abs(w);
    if (gross > cfg.max_gross && gross > 0) {
      double scale = cfg.max_gross / gross;
      for (double& w : out.w_target) w *= scale;
      gross = cfg.max_gross;
    }

    out.gross = gross;
    out.net_gross = gross; // could subtract forecast TC here if desired
    return out;
  }

private:
  TCModel tc_;
  const IFeeModel& fee_;
};

} // namespace sentio

```

## 📄 **FILE 75 of 180**: include/sentio/portfolio/tc_slippage_model.hpp

**File Information**:
- **Path**: `include/sentio/portfolio/tc_slippage_model.hpp`

- **Size**: 27 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <algorithm>
#include <cmath>

namespace sentio {

struct LiquidityStats {
  double adv_notional;  // average daily $ volume
  double spread_bp;     // bid-ask spread in bps
  double vol_1d;        // 1-day realized vol (for impact)
};

struct TCModel {
  // Simple slippage model: half-spread + impact
  // slippage = 0.5*spread + k * (trade_notional / ADV) ^ alpha
  double k_impact = 25.0;  // bps at 100% ADV if alpha=0.5
  double alpha    = 0.5;   // square-root impact

  double slippage_bp(double trade_notional, const LiquidityStats& L) const {
    double half_spread = 0.5 * L.spread_bp;
    double adv_frac = (L.adv_notional > 0 ? std::min(1.0, trade_notional / L.adv_notional) : 1.0);
    double impact = k_impact * std::pow(adv_frac, alpha);
    return half_spread + impact; // total bps (one-way)
  }
};

} // namespace sentio

```

## 📄 **FILE 76 of 180**: include/sentio/portfolio/utilization_governor.hpp

**File Information**:
- **Path**: `include/sentio/portfolio/utilization_governor.hpp`

- **Size**: 58 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <algorithm>

namespace sentio {

struct UtilGovConfig {
  double target_gross = 0.60;  // target gross exposure (sum |weights|), e.g. 60%
  double kp_expo      = 0.05;  // proportional gain for exposure gap
  double target_tpd   = 40.0;  // trades per day target (optional)
  double kp_trades    = 0.02;  // proportional gain for trade gap
  float  max_shift    = 0.10f; // max threshold nudge
  double max_vol_adj  = 0.50;  // ±50% of vol target adjustment
};

struct UtilGovState {
  double expo_shift{0.0};      // maps to sizer's vol target multiplier
  float  buy_shift{0.0f};      // router threshold shift
  float  sell_shift{0.0f};
  double integ_expo{0.0};      // optional: add Ki later if needed
  double integ_trades{0.0};
};

class UtilizationGovernor {
public:
  explicit UtilizationGovernor(const UtilGovConfig& c) : cfg_(c) {}

  void daily_update(double realized_gross, int trades_today, UtilGovState& st){
    // Exposure control
    double e_err = cfg_.target_gross - realized_gross;
    st.expo_shift = std::clamp(st.expo_shift + cfg_.kp_expo * e_err,
                               -cfg_.max_vol_adj, cfg_.max_vol_adj);

    // Trades/day control → route thresholds
    double t_err = cfg_.target_tpd - trades_today;
    float delta  = (float)(cfg_.kp_trades * t_err);
    st.buy_shift  = clamp_shift(st.buy_shift  - 0.5f*delta);
    st.sell_shift = clamp_shift(st.sell_shift - 0.5f*delta);
  }

  void get_nudges(struct RouterNudges& nudges, const UtilGovState& st) const {
    nudges.buy_shift = st.buy_shift;
    nudges.sell_shift = st.sell_shift;
  }

private:
  float clamp_shift(float x) const {
    return std::clamp(x, -cfg_.max_shift, cfg_.max_shift);
  }
  UtilGovConfig cfg_;
};

// Forward declare RouterNudges for get_nudges method
struct RouterNudges {
  float buy_shift  = 0.f;
  float sell_shift = 0.f;
};

} // namespace sentio

```

## 📄 **FILE 77 of 180**: include/sentio/position_coordinator.hpp

**File Information**:
- **Path**: `include/sentio/position_coordinator.hpp`

- **Size**: 135 lines
- **Modified**: 2025-09-18 12:02:42

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include "position_validator.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// Forward declarations
namespace sentio {
    class BaseStrategy;
}

namespace sentio {

// **POSITION COORDINATOR**: Strategy-agnostic conflict prevention system
// Ensures no conflicting positions are ever created by coordinating all allocation decisions

enum class CoordinationResult {
    APPROVED,           // Decision approved for execution
    REJECTED_CONFLICT,  // Decision rejected due to conflict
    REJECTED_FREQUENCY, // Decision rejected due to frequency limit
    MODIFIED            // Decision modified to prevent conflict
};

struct CoordinationDecision {
    CoordinationResult result;
    std::string instrument;
    double approved_weight;
    double original_weight;
    std::string reason;
    std::string conflict_details;
};

struct AllocationRequest {
    std::string strategy_name;
    std::string instrument;
    double target_weight;
    double confidence;
    std::string reason;
    std::string chain_id;
};

class PositionCoordinator {
private:
    // **CONFLICT PREVENTION**: ETF family classifications
    static const std::unordered_set<std::string> LONG_ETFS;
    static const std::unordered_set<std::string> INVERSE_ETFS;
    
    // **STATE TRACKING**: Current portfolio state for conflict detection
    std::unordered_map<std::string, double> current_positions_;
    std::unordered_map<std::string, double> pending_positions_;
    
    // **FREQUENCY CONTROL**: Order frequency management
    int orders_this_bar_;
    int max_orders_per_bar_;
    
    // **CONFLICT DETECTION**: Check if positions would conflict
    bool would_create_conflict(const std::unordered_map<std::string, double>& positions) const;
    
    // **POSITION ANALYSIS**: Analyze position implications
    struct PositionAnalysis {
        bool has_long_etf = false;
        bool has_inverse_etf = false;
        bool has_short_qqq = false;
        std::vector<std::string> long_positions;
        std::vector<std::string> short_positions;
        std::vector<std::string> inverse_positions;
    };
    
    PositionAnalysis analyze_positions(const std::unordered_map<std::string, double>& positions) const;
    
    // **CONFLICT RESOLUTION**: Attempt to resolve conflicts by modifying weights
    CoordinationDecision resolve_conflict(const AllocationRequest& request);
    
public:
    PositionCoordinator(int max_orders_per_bar = 1);
    
    // **MAIN COORDINATION**: Coordinate allocation decisions to prevent conflicts
    std::vector<CoordinationDecision> coordinate_allocations(
        const std::vector<AllocationRequest>& requests,
        const Portfolio& current_portfolio,
        const SymbolTable& ST
    );
    
    // **STRATEGY-AGNOSTIC COORDINATION**: Use strategy's conflict rules
    std::vector<CoordinationDecision> coordinate_allocations(
        const std::vector<AllocationRequest>& requests,
        const Portfolio& current_portfolio,
        const SymbolTable& ST,
        const BaseStrategy* strategy
    );
    
    // **BAR RESET**: Reset per-bar state for new bar
    void reset_bar();
    
    // **POSITION SYNC**: Sync current positions from portfolio
    void sync_positions(const Portfolio& portfolio, const SymbolTable& ST);
    
    // **CONFIGURATION**: Set coordination parameters
    void set_max_orders_per_bar(int max_orders) { max_orders_per_bar_ = max_orders; }
    
    // **STATISTICS**: Get coordination statistics
    struct CoordinationStats {
        int total_requests = 0;
        int approved = 0;
        int rejected_conflict = 0;
        int rejected_frequency = 0;
        int modified = 0;
    };
    
    CoordinationStats get_stats() const { return stats_; }
    void reset_stats() { stats_ = CoordinationStats{}; }
    
private:
    CoordinationStats stats_;
};

// **ALLOCATION DECISION STRUCTURE**: Define allocation decision structure
struct AllocationDecision {
    std::string instrument;
    double target_weight;
    double confidence;
    std::string reason;
};

// **ALLOCATION DECISION CONVERTER**: Convert strategy decisions to requests
std::vector<AllocationRequest> convert_allocation_decisions(
    const std::vector<AllocationDecision>& decisions,
    const std::string& strategy_name,
    const std::string& chain_id
);

} // namespace sentio

```

## 📄 **FILE 78 of 180**: include/sentio/position_validator.hpp

**File Information**:
- **Path**: `include/sentio/position_validator.hpp`

- **Size**: 95 lines
- **Modified**: 2025-09-18 16:44:36

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include <string>
#include <unordered_set>
#include <vector>

namespace sentio {

// **UPDATED**: Conflicting position detection - PSQ is inverse ETF, not short position
const std::unordered_set<std::string> LONG_ETFS = {"QQQ", "TQQQ"};
const std::unordered_set<std::string> INVERSE_ETFS = {"SQQQ", "PSQ"}; // PSQ restored as inverse ETF

inline bool has_conflicting_positions(const Portfolio& pf, const SymbolTable& ST) {
    bool has_long_etf = false;
    bool has_inverse_etf = false;
    bool has_short_qqq = false;
    
    for (size_t sid = 0; sid < pf.positions.size(); ++sid) {
        const auto& pos = pf.positions[sid];
        if (std::abs(pos.qty) > 1e-6) {
            const std::string& symbol = ST.get_symbol(sid);
            
            if (LONG_ETFS.count(symbol)) {
                if (pos.qty > 0) {
                    has_long_etf = true;
                } else {
                    // SHORT QQQ or SHORT TQQQ
                    has_short_qqq = true;
                }
            }
            if (INVERSE_ETFS.count(symbol)) {
                has_inverse_etf = true;
            }
        }
    }
    
    // **CONFLICT RULES**:
    // 1. Long ETF (QQQ+, TQQQ+) conflicts with Inverse ETF (SQQQ) or SHORT QQQ (QQQ-, TQQQ-)
    // 2. SHORT QQQ (QQQ-, TQQQ-) conflicts with Long ETF (QQQ+, TQQQ+)
    // 3. Inverse ETF (SQQQ) conflicts with Long ETF (QQQ+, TQQQ+)
    return (has_long_etf && (has_inverse_etf || has_short_qqq)) || 
           (has_short_qqq && has_long_etf);
}

inline std::string get_conflicting_symbols(const Portfolio& pf, const SymbolTable& ST) {
    std::vector<std::string> long_positions;
    std::vector<std::string> short_positions;
    std::vector<std::string> inverse_positions;
    
    for (size_t sid = 0; sid < pf.positions.size(); ++sid) {
        const auto& pos = pf.positions[sid];
        if (std::abs(pos.qty) > 1e-6) {
            const std::string& symbol = ST.get_symbol(sid);
            if (LONG_ETFS.count(symbol)) {
                if (pos.qty > 0) {
                    long_positions.push_back(symbol + "(+" + std::to_string((int)pos.qty) + ")");
                } else {
                    short_positions.push_back("SHORT " + symbol + "(" + std::to_string((int)pos.qty) + ")");
                }
            }
            if (INVERSE_ETFS.count(symbol)) {
                inverse_positions.push_back(symbol + "(" + std::to_string((int)pos.qty) + ")");
            }
        }
    }
    
    std::string result;
    if (!long_positions.empty()) {
        result += "Long: ";
        for (size_t i = 0; i < long_positions.size(); ++i) {
            if (i > 0) result += ", ";
            result += long_positions[i];
        }
    }
    if (!short_positions.empty()) {
        if (!result.empty()) result += "; ";
        result += "Short: ";
        for (size_t i = 0; i < short_positions.size(); ++i) {
            if (i > 0) result += ", ";
            result += short_positions[i];
        }
    }
    if (!inverse_positions.empty()) {
        if (!result.empty()) result += "; ";
        result += "Inverse: ";
        for (size_t i = 0; i < inverse_positions.size(); ++i) {
            if (i > 0) result += ", ";
            result += inverse_positions[i];
        }
    }
    return result;
}

} // namespace sentio

```

## 📄 **FILE 79 of 180**: include/sentio/pricebook.hpp

**File Information**:
- **Path**: `include/sentio/pricebook.hpp`

- **Size**: 59 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include "core.hpp"
#include "symbol_table.hpp"

namespace sentio {

// **MODIFIED**: The Pricebook ensures that all symbol prices are correctly aligned in time.
// It works by advancing an index for each symbol's data series until its timestamp
// matches or just passes the timestamp of the base symbol's current bar.
// This is both fast (amortized O(1)) and correct, handling missing data gracefully.
struct Pricebook {
    const int base_id;
    const SymbolTable& ST;
    const std::vector<std::vector<Bar>>& S; // Reference to all series data
    std::vector<int> idx;                   // Rolling index for each symbol's series
    std::vector<double> last_px;            // Stores the last known price for each symbol ID

    Pricebook(int base, const SymbolTable& st, const std::vector<std::vector<Bar>>& series)
      : base_id(base), ST(st), S(series), idx(S.size(), 0), last_px(S.size(), 0.0) {}

    // Advances the index 'j' for a given series 'V' to the bar corresponding to 'base_ts'
    inline void advance_to_ts(const std::vector<Bar>& V, int& j, int64_t base_ts) {
        const int n = (int)V.size();
        // Move index forward as long as the *next* bar is still at or before the target time
        while (j + 1 < n && V[j + 1].ts_utc_epoch <= base_ts) {
            ++j;
        }
    }

    // Syncs all symbol prices to the timestamp of the i-th bar of the base symbol
    inline void sync_to_base_i(int i) {
        if (S[base_id].empty()) return;
        const int64_t ts = S[base_id][i].ts_utc_epoch;
        
        for (int sid = 0; sid < (int)S.size(); ++sid) {
            if (!S[sid].empty()) {
                advance_to_ts(S[sid], idx[sid], ts);
                last_px[sid] = S[sid][idx[sid]].close;
            }
        }
    }

    // **NEW**: Helper to get the last prices as a map for components needing string keys
    inline std::unordered_map<std::string, double> last_px_map() const {
        std::unordered_map<std::string, double> price_map;
        for (int sid = 0; sid < (int)last_px.size(); ++sid) {
            if (last_px[sid] > 0.0) {
                price_map[ST.get_symbol(sid)] = last_px[sid];
            }
        }
        return price_map;
    }
};

} // namespace sentio
```

## 📄 **FILE 80 of 180**: include/sentio/profiling.hpp

**File Information**:
- **Path**: `include/sentio/profiling.hpp`

- **Size**: 25 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <chrono>
#include <cstdio>

namespace sentio {

struct Tsc {
    std::chrono::high_resolution_clock::time_point t0;
    
    void tic() { 
        t0 = std::chrono::high_resolution_clock::now(); 
    }
    
    double toc_ms() const {
        using namespace std::chrono;
        return duration<double, std::milli>(high_resolution_clock::now() - t0).count();
    }
    
    double toc_sec() const {
        using namespace std::chrono;
        return duration<double>(high_resolution_clock::now() - t0).count();
    }
};

} // namespace sentio
```

## 📄 **FILE 81 of 180**: include/sentio/progress_bar.hpp

**File Information**:
- **Path**: `include/sentio/progress_bar.hpp`

- **Size**: 225 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>

namespace sentio {

class ProgressBar {
public:
    ProgressBar(int total, const std::string& description = "Progress")
        : total_(total), current_(0), description_(description), start_time_(std::chrono::high_resolution_clock::now()) {}
    
    void update(int current) {
        current_ = current;
        if (current_ % std::max(1, total_ / 100) == 0 || current_ == total_) {
            display();
        }
    }
    
    void display() {
        double percentage = (double)current_ / total_ * 100.0;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        
        // Calculate ETA
        double eta_seconds = 0.0;
        if (current_ > 0 && current_ < total_) {
            eta_seconds = (double)elapsed / current_ * (total_ - current_);
        }
        
        std::cout << "\r" << description_ << ": [";
        
        // Draw progress bar (50 characters wide)
        int bar_width = 50;
        int pos = (int)(percentage / 100.0 * bar_width);
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        
        std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "% "
                  << "(" << current_ << "/" << total_ << ") "
                  << "Elapsed: " << elapsed << "s";
        
        if (eta_seconds > 0) {
            std::cout << " ETA: " << (int)eta_seconds << "s";
        }
        
        std::cout << std::flush;
        
        if (current_ == total_) {
            std::cout << std::endl;
        }
    }
    
    void set_description(const std::string& desc) {
        description_ = desc;
    }
    
    int get_current() const { return current_; }
    int get_total() const { return total_; }
    const std::string& get_description() const { return description_; }

private:
    int total_;
    int current_;
    std::string description_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

class TrainingProgressBar : public ProgressBar {
public:
    TrainingProgressBar(int total, const std::string& strategy_name = "TSB")
        : ProgressBar(total, "Training " + strategy_name), best_sharpe_(-999.0), best_return_(-999.0), start_time_(std::chrono::high_resolution_clock::now()) {}
    
    void update_with_metrics(int current, double sharpe_ratio, double total_return, double oos_return = 0.0) {
        // Track best metrics
        if (sharpe_ratio > best_sharpe_) best_sharpe_ = sharpe_ratio;
        if (total_return > best_return_) best_return_ = total_return;
        
        update(current);
        if (current % std::max(1, get_total() / 100) == 0 || current == get_total()) {
            display_with_metrics(sharpe_ratio, total_return, oos_return);
        }
    }
    
    void display_with_metrics(double current_sharpe, double current_return, double oos_return = 0.0) {
        int current = get_current();
        int total = get_total();
        double percentage = (double)current / total * 100.0;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        
        // Calculate ETA
        double eta_seconds = 0.0;
        if (current > 0 && current < total) {
            eta_seconds = (double)elapsed / current * (total - current);
        }
        
        std::cout << "\r" << get_description() << ": [";
        
        // Draw progress bar (50 characters wide)
        int bar_width = 50;
        int pos = (int)(percentage / 100.0 * bar_width);
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        
        std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "% "
                  << "(" << current << "/" << total << ") "
                  << "Elapsed: " << elapsed << "s";
        
        if (eta_seconds > 0) {
            std::cout << " ETA: " << (int)eta_seconds << "s";
        }
        
        // Add metrics
        std::cout << " | Sharpe: " << std::fixed << std::setprecision(3) << current_sharpe
                  << " | Return: " << std::fixed << std::setprecision(2) << (current_return * 100) << "%";
        
        if (oos_return != 0.0) {
            std::cout << " | OOS: " << std::fixed << std::setprecision(2) << (oos_return * 100) << "%";
        }
        
        std::cout << " | Best Sharpe: " << std::fixed << std::setprecision(3) << best_sharpe_
                  << " | Best Return: " << std::fixed << std::setprecision(2) << (best_return_ * 100) << "%";
        
        std::cout << std::flush;
        
        if (current == total) {
            std::cout << std::endl;
        }
    }

private:
    double best_sharpe_;
    double best_return_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

class WFProgressBar : public ProgressBar {
public:
    WFProgressBar(int total, const std::string& strategy_name = "TSB")
        : ProgressBar(total, "WF Test " + strategy_name), best_oos_sharpe_(-999.0), best_oos_return_(-999.0), 
          avg_oos_return_(0.0), successful_folds_(0), start_time_(std::chrono::high_resolution_clock::now()) {}
    
    void update_with_wf_metrics(int current, double oos_sharpe, double oos_return, double train_return = 0.0) {
        // Track best OOS metrics
        if (oos_sharpe > best_oos_sharpe_) best_oos_sharpe_ = oos_sharpe;
        if (oos_return > best_oos_return_) best_oos_return_ = oos_return;
        
        // Update running average
        successful_folds_++;
        avg_oos_return_ = (avg_oos_return_ * (successful_folds_ - 1) + oos_return) / successful_folds_;
        
        update(current);
        if (current % std::max(1, get_total() / 100) == 0 || current == get_total()) {
            display_with_wf_metrics(oos_sharpe, oos_return, train_return);
        }
    }
    
    void display_with_wf_metrics(double current_oos_sharpe, double current_oos_return, double train_return = 0.0) {
        int current = get_current();
        int total = get_total();
        double percentage = (double)current / total * 100.0;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        
        // Calculate ETA
        double eta_seconds = 0.0;
        if (current > 0 && current < total) {
            eta_seconds = (double)elapsed / current * (total - current);
        }
        
        std::cout << "\r" << get_description() << ": [";
        
        // Draw progress bar (50 characters wide)
        int bar_width = 50;
        int pos = (int)(percentage / 100.0 * bar_width);
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        
        std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "% "
                  << "(" << current << "/" << total << ") "
                  << "Elapsed: " << elapsed << "s";
        
        if (eta_seconds > 0) {
            std::cout << " ETA: " << (int)eta_seconds << "s";
        }
        
        // Add WF-specific metrics
        std::cout << " | OOS Sharpe: " << std::fixed << std::setprecision(3) << current_oos_sharpe
                  << " | OOS Return: " << std::fixed << std::setprecision(2) << (current_oos_return * 100) << "%";
        
        if (train_return != 0.0) {
            std::cout << " | Train Return: " << std::fixed << std::setprecision(2) << (train_return * 100) << "%";
        }
        
        std::cout << " | Avg OOS: " << std::fixed << std::setprecision(2) << (avg_oos_return_ * 100) << "%"
                  << " | Best OOS Sharpe: " << std::fixed << std::setprecision(3) << best_oos_sharpe_
                  << " | Folds: " << successful_folds_;
        
        std::cout << std::flush;
        
        if (current == total) {
            std::cout << std::endl;
        }
    }

private:
    double best_oos_sharpe_;
    double best_oos_return_;
    double avg_oos_return_;
    int successful_folds_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace sentio
```

## 📄 **FILE 82 of 180**: include/sentio/property_test.hpp

**File Information**:
- **Path**: `include/sentio/property_test.hpp`

- **Size**: 26 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <functional>
#include <vector>
#include <string>
#include <iostream>

namespace sentio {

struct PropCase { std::string name; std::function<bool()> fn; };

inline int run_properties(const std::vector<PropCase>& cases) {
  int fails = 0;
  for (auto& c : cases) {
    bool ok = false;
    try { ok = c.fn(); }
    catch (const std::exception& e) {
      std::cerr << "[PROP] " << c.name << " threw: " << e.what() << "\n";
      ok = false;
    }
    if (!ok) { std::cerr << "[PROP] FAIL: " << c.name << "\n"; ++fails; }
  }
  if (fails==0) std::cout << "[PROP] all passed ("<<cases.size()<<")\n";
  return fails==0 ? 0 : 1;
}

} // namespace sentio

```

## 📄 **FILE 83 of 180**: include/sentio/rolling_stats.hpp

**File Information**:
- **Path**: `include/sentio/rolling_stats.hpp`

- **Size**: 97 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace sentio {

struct RollingMean {
  int win, idx=0, count=0;
  std::vector<double> buf;
  double sum=0.0;
  
  explicit RollingMean(int w = 1) { reset(w); }

  void reset(int w) {
    win = w > 0 ? w : 1;
    idx = 0;
    count = 0;
    sum = 0.0;
    buf.assign(win, 0.0);
  }

  inline double push(double x){
      if (count < win) { 
          buf[count++] = x; 
          sum += x; 
      } else {
          sum -= buf[idx]; 
          buf[idx]=x; 
          sum += x; 
          idx = (idx+1) % win; 
      }
      return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }

  double mean() const {
      return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }

  size_t size() const {
      return static_cast<size_t>(count);
  }
};


struct RollingMeanVar {
  int win, idx=0, count=0;
  std::vector<double> buf;
  double sum=0.0, sumsq=0.0;

  explicit RollingMeanVar(int w = 1) { reset(w); }

  void reset(int w) {
    win = w > 0 ? w : 1;
    idx = 0;
    count = 0;
    sum = 0.0;
    sumsq = 0.0;
    buf.assign(win, 0.0);
  }

  inline std::pair<double,double> push(double x){
    if (count < win) {
      buf[count++] = x; 
      sum += x; 
      sumsq += x*x;
    } else {
      double old_val = buf[idx];
      sum   -= old_val;
      sumsq -= old_val * old_val;
      buf[idx] = x;
      sum   += x;
      sumsq += x*x;
      idx = (idx+1) % win;
    }
    double m = count > 0 ? sum / static_cast<double>(count) : 0.0;
    double v = count > 0 ? std::max(0.0, (sumsq / static_cast<double>(count)) - (m*m)) : 0.0;
    return {m, v};
  }
  
  double mean() const {
      return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }
  
  double var() const {
      if (count < 2) return 0.0;
      double m = mean();
      return std::max(0.0, (sumsq / static_cast<double>(count)) - (m * m));
  }

  double stddev() const {
      return std::sqrt(var());
  }
};

} // namespace sentio

```

## 📄 **FILE 84 of 180**: include/sentio/router.hpp

**File Information**:
- **Path**: `include/sentio/router.hpp`

- **Size**: 92 lines
- **Modified**: 2025-09-16 16:20:36

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

## 📄 **FILE 85 of 180**: include/sentio/rsi_prob.hpp

**File Information**:
- **Path**: `include/sentio/rsi_prob.hpp`

- **Size**: 23 lines
- **Modified**: 2025-09-11 12:25:09

- **Type**: .hpp

```text
#pragma once
#include <cmath>

namespace sentio {

// Calibrated so that p(30)=0.8, p(50)=0.5, p(70)=0.2
// p(RSI) = 1 / (1 + exp( k * (RSI - 50) / 10 ) )
// Solve: 0.8 = 1/(1+exp(k*(30-50)/10)) -> k = ln(2) ≈ 0.693147
inline double rsi_to_prob(double rsi) {
    constexpr double k = 0.6931471805599453; // ln(2)
    double x = (rsi - 50.0) / 10.0;
    double e = std::exp(k * x);
    return 1.0 / (1.0 + e);
}

// Optionally expose a tunable steepness (k = ln(2)*alpha).
inline double rsi_to_prob_tuned(double rsi, double alpha) {
    double k = 0.6931471805599453 * alpha;
    double x = (rsi - 50.0) / 10.0;
    return 1.0 / (1.0 + std::exp(k * x));
}

} // namespace sentio

```

## 📄 **FILE 86 of 180**: include/sentio/rules/adapters.hpp

**File Information**:
- **Path**: `include/sentio/rules/adapters.hpp`

- **Size**: 26 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "irule.hpp"
#include <algorithm>
#include <cmath>

namespace sentio::rules {

inline float logistic(float x, float k=1.2f){ return 1.f/(1.f+std::exp(-k*x)); }

inline float signal_to_p(float sig_pm, float strength=1.f){
  strength=std::clamp(strength,0.f,1.f);
  if (sig_pm>0)  return std::min(1.f, 0.5f + 0.5f*strength);
  if (sig_pm<0)  return std::max(0.f, 0.5f - 0.5f*strength);
  return 0.5f;
}

inline float to_probability(const RuleOutput& out, float k_logistic=1.2f){
  if (out.p_up)   return std::clamp(*out.p_up, 0.f, 1.f);
  if (out.signal) return signal_to_p((float)*out.signal, out.strength.value_or(1.f));
  if (out.score)  return logistic(*out.score, k_logistic);
  return 0.5f;
}

} // namespace sentio::rules



```

## 📄 **FILE 87 of 180**: include/sentio/rules/bbands_squeeze_rule.hpp

**File Information**:
- **Path**: `include/sentio/rules/bbands_squeeze_rule.hpp`

- **Size**: 45 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "irule.hpp"
#include <vector>
#include <deque>
#include <cmath>

namespace sentio::rules {

struct BBandsSqueezeBreakoutRule : IRuleStrategy {
  int win{20}; double k{2.0}; double squeeze_thr{0.8};
  std::vector<double> ma_, sd_, upper_, lower_, bw_, med_bw_;
  const char* name() const override { return "BBANDS_SQUEEZE_BRK"; }
  int warmup() const override { return win+1; }

  std::optional<RuleOutput> eval(const BarsView& b, int64_t i) override {
    if ((int)ma_.size()!=b.n) build_(b);
    if (i < warmup()) return std::nullopt;
    bool squeeze = (bw_[i] < squeeze_thr * med_bw_[i]);
    int sig = 0;
    if (squeeze){
      if (b.close[i] > upper_[i]) sig = +1;
      else if (b.close[i] < lower_[i]) sig = -1;
    }
    return RuleOutput{std::nullopt, sig, std::nullopt, 0.7f};
  }

  void build_(const BarsView& b){
    int N=b.n; ma_.assign(N,0); sd_.assign(N,0); upper_.assign(N,0); lower_.assign(N,0); bw_.assign(N,0); med_bw_.assign(N,0);
    std::deque<double> q; double s=0,s2=0; std::deque<double> bwq;
    for(int i=0;i<N;i++){
      q.push_back(b.close[i]); s+=b.close[i]; s2+=b.close[i]*b.close[i];
      if((int)q.size()>win){ double z=q.front(); q.pop_front(); s-=z; s2-=z*z; }
      if ((int)q.size()==win){
        double m=s/win, v=std::max(0.0, s2/win - m*m), sd=std::sqrt(v);
        ma_[i]=m; sd_[i]=sd; upper_[i]=m+k*sd; lower_[i]=m-k*sd; bw_[i]=(upper_[i]-lower_[i])/(m+1e-9);
      } else { ma_[i]=(i?ma_[i-1]:b.close[0]); sd_[i]=(i?sd_[i-1]:0.0); upper_[i]=(i?upper_[i-1]:b.close[0]); lower_[i]=(i?lower_[i-1]:b.close[0]); bw_[i]=(i?bw_[i-1]:0.0); }
      if ((int)bwq.size()==win) bwq.pop_front(); bwq.push_back(bw_[i]);
      double m=0; for(double x:bwq) m+=x; med_bw_[i]=(bwq.empty()? bw_[i] : m/bwq.size());
    }
  }
};

} // namespace sentio::rules



```

## 📄 **FILE 88 of 180**: include/sentio/rules/diversity_weighter.hpp

**File Information**:
- **Path**: `include/sentio/rules/diversity_weighter.hpp`

- **Size**: 50 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

namespace sentio::rules {

struct DiversityCfg {
  int   window = 512;
  float shrink = 0.10f;
  float min_w  = 0.10f;
  float max_w  = 3.00f;
};

class DiversityWeighter {
public:
  DiversityWeighter(int members, const DiversityCfg& c = {})
  : M_(members), cfg_(c), mean_(members,0.f), var_(members,1e-4f) {}

  void update(const std::vector<float>& p){
    if ((int)p.size()!=M_) return;
    hist_.push_back(p);
    if ((int)hist_.size()>cfg_.window) hist_.erase(hist_.begin());
    for (int k=0;k<M_;++k){
      mean_[k] = 0.f; for (auto& r: hist_) mean_[k]+=r[k]; mean_[k]/=hist_.size();
      float v=0.f; for (auto& r: hist_){ float d=r[k]-mean_[k]; v+=d*d; }
      var_[k] = std::max(v/std::max(1,(int)hist_.size()-1), 1e-6f);
    }
  }

  std::vector<float> weights() const {
    std::vector<float> w(M_, 1.f);
    for (int k=0;k<M_;++k){
      float inv = 1.f / std::max(var_[k], 1e-6f);
      float ws = (1-cfg_.shrink)*inv + cfg_.shrink*1.f;
      w[k] = std::clamp(ws, cfg_.min_w, cfg_.max_w);
    }
    return w;
  }

private:
  int M_;
  DiversityCfg cfg_{};
  std::vector<std::vector<float>> hist_;
  std::vector<float> mean_, var_;
};

} // namespace sentio::rules



```

## 📄 **FILE 89 of 180**: include/sentio/rules/integrated_rule_ensemble.hpp

**File Information**:
- **Path**: `include/sentio/rules/integrated_rule_ensemble.hpp`

- **Size**: 163 lines
- **Modified**: 2025-09-13 14:55:09

- **Type**: .hpp

```text
#pragma once
#include "irule.hpp"
#include "adapters.hpp"
#include "online_platt_calibrator.hpp"
#include "diversity_weighter.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace sentio::rules {

struct EnsembleConfig {
  float score_logistic_k = 1.2f;
  int   reliability_window = 512;
  std::vector<float> base_weights; // size=rules; default=1
  float agreement_boost = 0.25f;   // scales (p-0.5)
  int   min_rules = 1;
  float eps_clip = 1e-4f;
  bool  use_diversity = true;
  DiversityCfg diversity;
  bool  use_platt = true;
  PlattCfg platt;
};

struct EnsembleMeta {
  int    n_used{0};
  float  agreement{-1.f};
  float  variance{0.f};
  float  weighted_mean{0.f};
  std::vector<float> member_p;
  std::vector<float> member_w;
};

class IntegratedRuleEnsemble {
public:
  IntegratedRuleEnsemble(std::vector<std::unique_ptr<IRuleStrategy>> rules, EnsembleConfig cfg)
  : cfg_(std::move(cfg)), rules_(std::move(rules)),
    div_((int)rules_.size(), cfg_.diversity), platt_(cfg_.platt)
  {
    const size_t R = rules_.size();
    if (cfg_.base_weights.empty()) cfg_.base_weights.assign(R, 1.0f);
    reli_brier_.assign(R, {}); reli_hits_.assign(R, {});
  }

  int warmup() const {
    int w = 0; for (auto& r : rules_) w = std::max(w, r->warmup());
    w = std::max(w, cfg_.reliability_window);
    w = std::max(w, cfg_.diversity.window);
    return w;
  }

  std::optional<float> eval(const BarsView& b, int64_t i,
                            std::optional<float> realized_next_logret,
                            EnsembleMeta* meta=nullptr)
  {
    if (i < warmup()) return std::nullopt;

    raw_p_.clear(); w_eff_.clear();
    const size_t R = rules_.size();

    for (size_t r=0;r<R;r++){
      auto out = rules_[r]->eval(b, i);
      if (!out) { raw_p_.push_back(std::numeric_limits<float>::quiet_NaN()); continue; }
      float p = to_probability(*out, cfg_.score_logistic_k);
      raw_p_.push_back(std::clamp(p, cfg_.eps_clip, 1.f-cfg_.eps_clip));
    }
    if (cfg_.use_diversity){
      std::vector<float> snap(R, 0.5f);
      for (size_t r=0;r<R;r++) snap[r] = std::isfinite(raw_p_[r])? raw_p_[r] : 0.5f;
      div_.update(snap);
      auto w_div = div_.weights();
      for (size_t r=0;r<R;r++){
        if (!std::isfinite(raw_p_[r])) continue;
        float w = cfg_.base_weights[r] * reliability_weight_(r) * w_div[r];
        w_eff_.push_back(std::max(0.f,w));
      }
    } else {
      for (size_t r=0;r<R;r++){
        if (!std::isfinite(raw_p_[r])) continue;
        float w = cfg_.base_weights[r] * reliability_weight_(r);
        w_eff_.push_back(std::max(0.f,w));
      }
    }

    vec_p_.clear(); vec_w_.clear();
    for (size_t r=0, k=0; r<R; r++){
      if (!std::isfinite(raw_p_[r])) continue;
      vec_p_.push_back(raw_p_[r]);
      vec_w_.push_back(w_eff_[k++]);
    }
    if ((int)vec_p_.size() < cfg_.min_rules) return std::nullopt;

    float sw=0.f, sp=0.f;
    for (size_t k=0;k<vec_p_.size();k++){ sw += vec_w_[k]; sp += vec_w_[k]*vec_p_[k]; }
    if (sw<=0.f) return std::nullopt;
    float p_mean = sp / sw;

    float vote = 0.f; for (float p: vec_p_) vote += (p>=0.5f? +1.f : -1.f);
    float agree = std::fabs(vote) / (float)vec_p_.size();
    float boost = 1.f + cfg_.agreement_boost * (agree - 0.5f);
    float p_boosted = std::clamp(0.5f + (p_mean - 0.5f)*boost, cfg_.eps_clip, 1.f - cfg_.eps_clip);

    float p_final = p_boosted;
    if (cfg_.use_platt){ p_final = platt_.calibrate_from_p(p_boosted); }

    if (meta){
      meta->n_used = (int)vec_p_.size();
      meta->agreement = agree; meta->weighted_mean = p_mean;
      float m=0; for(float p:vec_p_) m+=p; m/=vec_p_.size();
      float v=0; for(float p:vec_p_){ float d=p-m; v+=d*d; } v/=std::max<size_t>(1,vec_p_.size()-1);
      meta->variance = v; meta->member_p = vec_p_; meta->member_w = vec_w_;
    }

    if (realized_next_logret){
      float target = (*realized_next_logret > 0.f) ? 1.f : 0.f;
      for (size_t r=0;r<R;r++){
        auto out = rules_[r]->eval(b, i);
        if (!out) continue;
        float p = to_probability(*out, cfg_.score_logistic_k);
        update_reliability_(r, p, target);
      }
      if (cfg_.use_platt){
        float pb = std::clamp(p_boosted, 1e-6f, 1.f-1e-6f);
        float zb = std::log(pb/(1.f-pb));
        platt_.update(zb, target);
      }
    }
    return p_final;
  }

private:
  float reliability_weight_(size_t r) const {
    const auto& B = reli_brier_[r]; const auto& H = reli_hits_[r];
    if (B.empty()) return 1.0f;
    float brier = mean_(B);
    float w_brier = std::clamp(1.5f - brier*3.0f, 0.25f, 2.0f);
    float hit = H.empty()? 0.5f : mean_(H);
    float w_hit = std::clamp(0.5f + (hit-0.5f)*1.0f, 0.25f, 2.0f);
    return 0.5f*w_brier + 0.5f*w_hit;
  }
  void update_reliability_(size_t r, float p, float target){
    float brier = (p - target)*(p - target);
    push_window_(reli_brier_[r], brier, cfg_.reliability_window);
    float hit = ((p>=0.5f) == (target>=0.5f)) ? 1.f : 0.f;
    push_window_(reli_hits_[r], hit, cfg_.reliability_window);
  }
  static void push_window_(std::vector<float>& v, float x, int W){ v.push_back(x); if ((int)v.size()>W) v.erase(v.begin()); }
  static float mean_(const std::vector<float>& v){ if (v.empty()) return 0.f; float s=0.f; for(float x:v) s+=x; return s/v.size(); }

  EnsembleConfig cfg_;
  std::vector<std::unique_ptr<IRuleStrategy>> rules_;
  std::vector<std::vector<float>> reli_brier_, reli_hits_;
  DiversityWeighter div_;
  OnlinePlatt       platt_;
  std::vector<float> raw_p_, w_eff_, vec_p_, vec_w_;
};

} // namespace sentio::rules



```

## 📄 **FILE 90 of 180**: include/sentio/rules/irule.hpp

**File Information**:
- **Path**: `include/sentio/rules/irule.hpp`

- **Size**: 33 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <optional>

namespace sentio::rules {

struct RuleOutput {
  std::optional<float> p_up;     // [0,1]
  std::optional<int>   signal;   // {-1,0,+1}
  std::optional<float> score;    // unbounded or [-1,1]
  std::optional<float> strength; // [0,1]
};

struct BarsView {
  const int64_t* ts;
  const double*  open;
  const double*  high;
  const double*  low;
  const double*  close;
  const double*  volume;
  int64_t        n;
};

struct IRuleStrategy {
  virtual ~IRuleStrategy() = default;
  virtual std::optional<RuleOutput> eval(const BarsView& bars, int64_t i) = 0;
  virtual int  warmup() const { return 20; }
  virtual const char* name() const { return "UnnamedRule"; }
};

} // namespace sentio::rules



```

## 📄 **FILE 91 of 180**: include/sentio/rules/momentum_volume_rule.hpp

**File Information**:
- **Path**: `include/sentio/rules/momentum_volume_rule.hpp`

- **Size**: 47 lines
- **Modified**: 2025-09-11 23:31:46

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

## 📄 **FILE 92 of 180**: include/sentio/rules/ofi_proxy_rule.hpp

**File Information**:
- **Path**: `include/sentio/rules/ofi_proxy_rule.hpp`

- **Size**: 45 lines
- **Modified**: 2025-09-11 23:31:46

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

## 📄 **FILE 93 of 180**: include/sentio/rules/online_platt_calibrator.hpp

**File Information**:
- **Path**: `include/sentio/rules/online_platt_calibrator.hpp`

- **Size**: 46 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace sentio::rules {

struct PlattCfg {
  int   window = 4096;
  float lr     = 0.02f;
  float l2     = 1e-3f;
  float clip   = 10.f;
};

class OnlinePlatt {
public:
  explicit OnlinePlatt(const PlattCfg& c = {}) : cfg_(c) {}

  void update(float z, float target){
    z = std::clamp(z, -cfg_.clip, cfg_.clip);
    float yhat = sigmoid(a_*z + b_);
    float grad_a = (yhat - target)*z + cfg_.l2*a_;
    float grad_b = (yhat - target)     + cfg_.l2*b_;
    a_ -= cfg_.lr * grad_a;
    b_ -= cfg_.lr * grad_b;
    zs_.push_back(z); ys_.push_back(target);
    if ((int)zs_.size()>cfg_.window){ zs_.erase(zs_.begin()); ys_.erase(ys_.begin()); }
  }

  float calibrate_from_p(float p) const {
    p = std::clamp(p, 1e-6f, 1.f-1e-6f);
    float z = std::log(p/(1.f-p));
    float zc = std::clamp(a_*z + b_, -cfg_.clip, cfg_.clip);
    return sigmoid(zc);
  }

private:
  static float sigmoid(float x){ return 1.f/(1.f+std::exp(-x)); }
  PlattCfg cfg_{};
  float a_{1.f}, b_{0.f};
  std::vector<float> zs_, ys_;
};

} // namespace sentio::rules



```

## 📄 **FILE 94 of 180**: include/sentio/rules/opening_range_breakout_rule.hpp

**File Information**:
- **Path**: `include/sentio/rules/opening_range_breakout_rule.hpp`

- **Size**: 34 lines
- **Modified**: 2025-09-10 11:15:18

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

## 📄 **FILE 95 of 180**: include/sentio/rules/registry.hpp

**File Information**:
- **Path**: `include/sentio/rules/registry.hpp`

- **Size**: 26 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "irule.hpp"
#include "sma_cross_rule.hpp"
#include "bbands_squeeze_rule.hpp"
#include "vwap_reversion_rule.hpp"
#include "opening_range_breakout_rule.hpp"
#include "momentum_volume_rule.hpp"
#include "ofi_proxy_rule.hpp"
#include <memory>
#include <string>

namespace sentio::rules {

inline std::unique_ptr<IRuleStrategy> make_rule(const std::string& name){
  if (name=="SMA_CROSS")             return std::make_unique<SMACrossRule>();
  if (name=="BBANDS_SQUEEZE_BRK")    return std::make_unique<BBandsSqueezeBreakoutRule>();
  if (name=="VWAP_REVERSION")        return std::make_unique<VWAPReversionRule>();
  if (name=="OPENING_RANGE_BRK")     return std::make_unique<OpeningRangeBreakoutRule>();
  if (name=="MOMENTUM_VOLUME")       return std::make_unique<MomentumVolumeRule>();
  if (name=="OFI_PROXY")             return std::make_unique<OFIProxyRule>();
  return {};
}

} // namespace sentio::rules



```

## 📄 **FILE 96 of 180**: include/sentio/rules/sma_cross_rule.hpp

**File Information**:
- **Path**: `include/sentio/rules/sma_cross_rule.hpp`

- **Size**: 34 lines
- **Modified**: 2025-09-10 11:15:18

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

## 📄 **FILE 97 of 180**: include/sentio/rules/utils/validation.hpp

**File Information**:
- **Path**: `include/sentio/rules/utils/validation.hpp`

- **Size**: 151 lines
- **Modified**: 2025-09-11 23:11:24

- **Type**: .hpp

```text
#pragma once

#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>

namespace sentio::rules::utils {

/**
 * @brief Validates if there's sufficient data for volume window calculations
 * @param data_size Current data size
 * @param vol_win Volume window size
 * @return true if sufficient data, false otherwise
 */
inline bool has_volume_window_data(int data_size, int vol_win) {
    return data_size > vol_win;
}

/**
 * @brief Validates if there's sufficient data for volume window calculations
 * @param data_size Current data size
 * @param vol_win Volume window size
 * @return true if sufficient data, false otherwise
 */
template<typename Container>
inline bool has_volume_window_data(const Container& data, int vol_win) {
    return static_cast<int>(data.size()) > vol_win;
}

/**
 * @brief Safely manages a sliding window with volume window validation
 * @tparam T Data type
 */
template<typename T>
class SlidingWindow {
private:
    std::deque<T> window_;
    int max_size_;
    T sum_{};
    T sum_squared_{};
    
public:
    explicit SlidingWindow(int max_size) : max_size_(max_size) {}
    
    void push(T value) {
        window_.push_back(value);
        sum_ += value;
        sum_squared_ += value * value;
        
        if (static_cast<int>(window_.size()) > max_size_) {
            T front_value = window_.front();
            window_.pop_front();
            sum_ -= front_value;
            sum_squared_ -= front_value * front_value;
        }
    }
    
    bool has_sufficient_data() const {
        return static_cast<int>(window_.size()) > max_size_;
    }
    
    T mean() const {
        return window_.empty() ? T{} : sum_ / static_cast<T>(window_.size());
    }
    
    T variance() const {
        if (window_.empty()) return T{};
        T m = mean();
        return std::max(T{}, sum_squared_ / static_cast<T>(window_.size()) - m * m);
    }
    
    T standard_deviation() const {
        return std::sqrt(variance());
    }
    
    size_t size() const { return window_.size(); }
    bool empty() const { return window_.empty(); }
};

/**
 * @brief Calculates rolling statistics with volume window validation
 * @param data Input data
 * @param vol_win Volume window size
 * @param output_mean Output mean values
 * @param output_variance Output variance values
 */
template<typename T>
void calculate_rolling_stats(const std::vector<T>& data, int vol_win,
                           std::vector<T>& output_mean, std::vector<T>& output_variance) {
    int N = static_cast<int>(data.size());
    output_mean.assign(N, T{});
    output_variance.assign(N, T{});
    
    SlidingWindow<T> window(vol_win);
    
    for (int i = 0; i < N; ++i) {
        window.push(data[i]);
        
        if (window.has_sufficient_data()) {
            output_mean[i] = window.mean();
            output_variance[i] = window.variance();
        }
    }
}

/**
 * @brief Calculates rolling mean with volume window validation
 * @param data Input data
 * @param vol_win Volume window size
 * @param output Output mean values
 */
template<typename T>
void calculate_rolling_mean(const std::vector<T>& data, int vol_win, std::vector<T>& output) {
    int N = static_cast<int>(data.size());
    output.assign(N, T{});
    
    SlidingWindow<T> window(vol_win);
    
    for (int i = 0; i < N; ++i) {
        window.push(data[i]);
        
        if (window.has_sufficient_data()) {
            output[i] = window.mean();
        }
    }
}

/**
 * @brief Calculates rolling variance with volume window validation
 * @param data Input data
 * @param vol_win Volume window size
 * @param output Output variance values
 */
template<typename T>
void calculate_rolling_variance(const std::vector<T>& data, int vol_win, std::vector<T>& output) {
    int N = static_cast<int>(data.size());
    output.assign(N, T{});
    
    SlidingWindow<T> window(vol_win);
    
    for (int i = 0; i < N; ++i) {
        window.push(data[i]);
        
        if (window.has_sufficient_data()) {
            output[i] = window.variance();
        }
    }
}

} // namespace sentio::rules::utils

```

## 📄 **FILE 98 of 180**: include/sentio/rules/vwap_reversion_rule.hpp

**File Information**:
- **Path**: `include/sentio/rules/vwap_reversion_rule.hpp`

- **Size**: 42 lines
- **Modified**: 2025-09-10 11:15:18

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

## 📄 **FILE 99 of 180**: include/sentio/run_id_generator.hpp

**File Information**:
- **Path**: `include/sentio/run_id_generator.hpp`

- **Size**: 23 lines
- **Modified**: 2025-09-13 09:40:08

- **Type**: .hpp

```text
#pragma once

#include <string>

namespace sentio {

/**
 * Generate a unique 6-digit run ID
 * Format: NNNNNN (e.g., "123456")
 * 
 * Uses a combination of timestamp and random number to ensure uniqueness
 */
std::string generate_run_id();

/**
 * Create a descriptive note for the audit system
 * Format: "Strategy: <strategy_name>, Test: <test_type>, Period: <period_info>"
 */
std::string create_audit_note(const std::string& strategy_name, 
                             const std::string& test_type, 
                             const std::string& period_info = "");

} // namespace sentio

```

## 📄 **FILE 100 of 180**: include/sentio/runner.hpp

**File Information**:
- **Path**: `include/sentio/runner.hpp`

- **Size**: 58 lines
- **Modified**: 2025-09-18 12:22:41

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "audit.hpp"
#include "router.hpp"
#include "sizer.hpp"
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

enum class AuditLevel { Full, MetricsOnly };

struct RunnerCfg {
    std::string strategy_name = "VWAPReversion";
    std::unordered_map<std::string, std::string> strategy_params;
    RouterCfg router;
    SizerCfg sizer;
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
                           int base_symbol_id, const RunnerCfg& cfg, const DatasetMetadata& dataset_meta = {});

// NEW: Canonical evaluation using Trading Block system for deterministic performance measurement
CanonicalReport run_canonical_backtest(IAuditRecorder& audit, const SymbolTable& ST, 
                                      const std::vector<std::vector<Bar>>& series, int base_symbol_id, 
                                      const RunnerCfg& cfg, const DatasetMetadata& dataset_meta, 
                                      const TradingBlockConfig& block_config);

} // namespace sentio


```

## 📄 **FILE 101 of 180**: include/sentio/side.hpp

**File Information**:
- **Path**: `include/sentio/side.hpp`

- **Size**: 37 lines
- **Modified**: 2025-09-11 21:59:53

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <string>

namespace sentio {

enum class PositionSide : int8_t { Flat=0, Long=1, Short=-1 };

inline std::string to_string(PositionSide s) {
    switch (s) { 
        case PositionSide::Flat: return "FLAT";
        case PositionSide::Long: return "LONG";
        case PositionSide::Short: return "SHORT"; 
    }
    return "UNKNOWN";
}

struct Qty { double shares{0.0}; };       // positive magnitude
struct Price { double px{0.0}; };         // last/avg price as needed

struct ExposureKey {
    std::string account;   // e.g., "alpaca:primary"
    std::string family;    // e.g., "QQQ*"  (see family mapper below)

    bool operator==(const ExposureKey& o) const {
        return account==o.account && family==o.family;
    }
};

struct ExposureKeyHash {
    size_t operator()(ExposureKey const& k) const noexcept {
        std::hash<std::string> h;
        return (h(k.account)*1315423911u) ^ h(k.family);
    }
};

} // namespace sentio

```

## 📄 **FILE 102 of 180**: include/sentio/signal.hpp

**File Information**:
- **Path**: `include/sentio/signal.hpp`

- **Size**: 16 lines
- **Modified**: 2025-09-11 12:25:03

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <cstdint>

namespace sentio {

enum class Side { Buy, Sell, Neutral };

struct Signal {
    std::string  symbol;  // e.g., "QQQ", "SQQQ"
    Side         side;    // Buy/Sell/Neutral
    double       weight;  // [-1, +1]
    std::int64_t ts;      // epoch millis or bar index
};

} // namespace sentio
```

## 📄 **FILE 103 of 180**: include/sentio/signal_diag.hpp

**File Information**:
- **Path**: `include/sentio/signal_diag.hpp`

- **Size**: 37 lines
- **Modified**: 2025-09-13 14:55:09

- **Type**: .hpp

```text
// signal_diag.hpp
#pragma once
#include <cstdint>
#include <cstdio>

enum class DropReason : uint8_t {
  NONE, MIN_BARS, SESSION, NAN_INPUT, ZERO_VOL, THRESHOLD, COOLDOWN, DUP_SAME_BAR
};

struct SignalDiag {
  uint64_t emitted=0, dropped=0;
  uint64_t r_min_bars=0, r_session=0, r_nan=0, r_zero_vol=0, r_threshold=0, r_cooldown=0, r_dup=0;

  inline void drop(DropReason r){
    dropped++;
    switch(r){
      case DropReason::MIN_BARS:  r_min_bars++; break;
      case DropReason::SESSION:   r_session++;  break;
      case DropReason::NAN_INPUT: r_nan++;      break;
      case DropReason::ZERO_VOL:  r_zero_vol++; break;
      case DropReason::THRESHOLD: r_threshold++;break;
      case DropReason::COOLDOWN:  r_cooldown++; break;
      case DropReason::DUP_SAME_BAR:r_dup++;    break;
      default: break;
    }
  }

  inline void print(const char* tag) const {
    // Debug: Signal diagnostics (commented out to reduce console noise)
    (void)tag; // Suppress unused parameter warning
    // std::fprintf(stderr, "[SIG %s] emitted=%llu dropped=%llu  min_bars=%llu session=%llu nan=%llu zerovol=%llu thr=%llu cooldown=%llu dup=%llu\n",
    //   tag,(unsigned long long)emitted,(unsigned long long)dropped,
    //   (unsigned long long)r_min_bars,(unsigned long long)r_session,(unsigned long long)r_nan,
    //   (unsigned long long)r_zero_vol,(unsigned long long)r_threshold,(unsigned long long)r_cooldown,
    //   (unsigned long long)r_dup);
  }
};
```

## 📄 **FILE 104 of 180**: include/sentio/signal_engine.hpp

**File Information**:
- **Path**: `include/sentio/signal_engine.hpp`

- **Size**: 22 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "signal_gate.hpp"

namespace sentio {

struct EngineOut {
  std::optional<StrategySignal> signal; // post-gate
  DropReason last_drop{DropReason::NONE};
};

class SignalEngine {
public:
  SignalEngine(IStrategy* strat, const GateCfg& gate_cfg, SignalHealth* health);
  EngineOut on_bar(const StrategyCtx& ctx, const Bar& b, bool inputs_finite=true);
private:
  IStrategy* strat_;
  SignalGate gate_;
  SignalHealth* health_;
};

} // namespace sentio

```

## 📄 **FILE 105 of 180**: include/sentio/signal_gate.hpp

**File Information**:
- **Path**: `include/sentio/signal_gate.hpp`

- **Size**: 44 lines
- **Modified**: 2025-09-12 10:12:46

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <atomic>
#include <optional>

namespace sentio {

enum class DropReason : uint16_t {
  NONE=0, WARMUP, NAN_INPUT, THRESHOLD_TOO_TIGHT,
  COOLDOWN_ACTIVE, DUPLICATE_BAR_TS
};

struct SignalHealth {
  std::atomic<uint64_t> emitted{0};
  std::atomic<uint64_t> dropped{0};
  std::unordered_map<DropReason, std::atomic<uint64_t>> by_reason;
  SignalHealth();
  void incr_emit();
  void incr_drop(DropReason r);
};

struct GateCfg { 
  int cooldown_bars=0; 
  double min_conf=0.05; 
};

class SignalGate {
public:
  explicit SignalGate(const GateCfg& cfg, SignalHealth* health);
  // Returns nullopt if dropped; otherwise passes through with possibly clamped confidence.
  std::optional<double> accept(std::int64_t ts_utc_epoch,
                               bool inputs_finite,
                               bool warmed_up,
                               double confidence);
private:
  GateCfg cfg_;
  SignalHealth* health_;
  std::int64_t last_emit_ts_{-1};
  int cooldown_left_{0};
};

} // namespace sentio

```

## 📄 **FILE 106 of 180**: include/sentio/signal_or.hpp

**File Information**:
- **Path**: `include/sentio/signal_or.hpp`

- **Size**: 78 lines
- **Modified**: 2025-09-11 23:31:46

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace sentio {

// Each rule emits a probability in [0,1] (>0.5 long bias, <0.5 short bias)
// and a confidence in [0,1] (0 = ignore, 1 = strong).
struct RuleOut {
  double p01;
  double conf01;
};

// Tuning knobs for OR behavior
struct OrCfg {
  double min_conf = 0.05;     // ignore components below this confidence
  double aggression = 0.85;   // 0..1, closer to 1 → stronger OR push
  double floor_eps = 1e-6;    // numerical safety
  double neutral_band = 0.015;// tiny band snapped to exact neutral
  double conflict_soften = 0.35; // 0..1 reduce both sides when both high
  size_t min_active = 1;      // require at least this many active rules
};

// Helper: clamp to [0,1]
inline double clamp01(double x) {
  if (!std::isfinite(x)) return 0.5;
  return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

// Noisy-OR on "evidence of long" and "evidence of short", combined safely.
inline double mix_signal_or(const std::vector<RuleOut>& rules, const OrCfg& cfg) {
  double prod_no_long  = 1.0;
  double prod_no_short = 1.0;
  size_t active = 0;

  for (auto r : rules) {
    double p = clamp01(r.p01);
    double c = clamp01(r.conf01);
    if (c < cfg.min_conf) continue;
    // Long evidence in [0,1]: map p ∈ [0.5,1] → [0,1]
    double e_long  = (p <= 0.5) ? 0.0 : (p - 0.5) * 2.0;
    double e_short = (p >= 0.5) ? 0.0 : (0.5 - p) * 2.0;
    // Confidence-weighted evidence
    e_long  = std::pow(e_long,  1.0 - cfg.aggression) * c;
    e_short = std::pow(e_short, 1.0 - cfg.aggression) * c;

    prod_no_long  *= (1.0 - std::max(cfg.floor_eps, e_long));
    prod_no_short *= (1.0 - std::max(cfg.floor_eps, e_short));
    active++;
  }

  if (active < cfg.min_active) return 0.5; // neutral if literally nothing active

  // Noisy-OR results (probability that at least one rule supports the side)
  double p_long  = 1.0 - prod_no_long;
  double p_short = 1.0 - prod_no_short;

  // If both sides are high (conflict), soften both so neutral can occur
  // only when truly balanced and confident on both sides.
  if (p_long > 0.0 && p_short > 0.0) {
    p_long  *= (1.0 - cfg.conflict_soften);
    p_short *= (1.0 - cfg.conflict_soften);
  }

  // Convert side probabilities back to a single p01 in [0,1]
  // Intuition: p = 0.5 + (p_long - p_short)/2, clipped and denoised.
  double p01 = 0.5 + 0.5 * (p_long - p_short);
  p01 = clamp01(p01);

  // Debounce micro-noise to exact neutral for stability near 0.5
  if (std::fabs(p01 - 0.5) < cfg.neutral_band) p01 = 0.5;

  return p01;
}

} // namespace sentio

```

## 📄 **FILE 107 of 180**: include/sentio/signal_pipeline.hpp

**File Information**:
- **Path**: `include/sentio/signal_pipeline.hpp`

- **Size**: 41 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "signal_gate.hpp"
#include "signal_trace.hpp"
#include <cstdint>
#include <string>
#include <optional>

namespace sentio {

// Forward declarations to avoid include conflicts
struct RouterCfg;
struct RouteDecision;
struct Order;
struct AccountSnapshot;
class PriceBook;

struct PipelineCfg {
  GateCfg gate;
  double min_order_shares{1.0};
};

struct PipelineOut {
  std::optional<StrategySignal> signal;
  TraceRow trace;
};

class SignalPipeline {
public:
  SignalPipeline(IStrategy* strat, const PipelineCfg& cfg, void* /*book*/, SignalTrace* trace)
  : strat_(strat), cfg_(cfg), trace_(trace), gate_(cfg.gate, nullptr) {}

  PipelineOut on_bar(const StrategyCtx& ctx, const Bar& b, const void* acct);
private:
  IStrategy* strat_;
  PipelineCfg cfg_;
  SignalTrace* trace_;
  SignalGate gate_;
};

} // namespace sentio

```

## 📄 **FILE 108 of 180**: include/sentio/signal_trace.hpp

**File Information**:
- **Path**: `include/sentio/signal_trace.hpp`

- **Size**: 53 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <optional>

namespace sentio {

enum class TraceReason : uint16_t {
  OK = 0,
  NO_STRATEGY_OUTPUT,
  NOT_RTH,
  HOLIDAY,
  WARMUP,
  NAN_INPUT,
  THRESHOLD_TOO_TIGHT,
  COOLDOWN_ACTIVE,
  DUPLICATE_BAR_TS,
  EMPTY_PRICEBOOK,
  NO_PRICE_FOR_INSTRUMENT,
  ROUTER_REJECTED,
  ORDER_QTY_LT_MIN,
  UNKNOWN
};

struct TraceRow {
  std::int64_t ts_utc{};
  std::string  instrument;     // stream instrument (e.g., QQQ)
  std::string  routed;         // routed instrument (e.g., TQQQ/SQQQ)
  double       close{};
  bool         is_rth{true};
  bool         warmed{true};
  bool         inputs_finite{true};
  double       confidence{0.0};              // raw strategy conf
  double       conf_after_gate{0.0};         // post gate
  double       target_weight{0.0};           // router decision
  double       last_px{0.0};                 // last price for routed
  double       order_qty{0.0};
  TraceReason  reason{TraceReason::UNKNOWN};
  std::string  note;                         // optional detail
};

class SignalTrace {
public:
  void push(const TraceRow& r) { rows_.push_back(r); }
  const std::vector<TraceRow>& rows() const { return rows_; }
  // useful summaries
  std::size_t count(TraceReason r) const;
private:
  std::vector<TraceRow> rows_;
};

} // namespace sentio

```

## 📄 **FILE 109 of 180**: include/sentio/signal_utils.hpp

**File Information**:
- **Path**: `include/sentio/signal_utils.hpp`

- **Size**: 145 lines
- **Modified**: 2025-09-18 10:31:51

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

// Detector interfaces for Signal OR integrated rule-based detectors
namespace sentio::detectors {

struct DetectorResult {
    double probability = 0.5;
    int direction = 0; // -1 short, 0 neutral, +1 long
    std::string_view name;
};

class IDetector {
public:
    virtual ~IDetector() = default;
    virtual std::string_view name() const = 0;
    virtual DetectorResult score(const std::vector<Bar>& bars, int idx) = 0;
    virtual int warmup_period() const = 0;
};

} // namespace sentio::detectors

```

## 📄 **FILE 110 of 180**: include/sentio/sim_data.hpp

**File Information**:
- **Path**: `include/sentio/sim_data.hpp`

- **Size**: 24 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>
#include <random>
#include "sanity.hpp"

namespace sentio {

// Generates a synthetic minute-bar series with regimes (trend, mean-revert, jump).
struct SimCfg {
  std::int64_t start_ts_utc{1'600'000'000};
  int minutes{500};
  double start_price{100.0};
  unsigned seed{42};
  // regime fractions (sum <= 1.0)
  double frac_trend{0.5};
  double frac_mr{0.4};
  double frac_jump{0.1};
  double vol_bps{15.0};    // base noise per min (bps)
};

std::vector<std::pair<std::int64_t, Bar>> generate_minute_series(const SimCfg& cfg);

} // namespace sentio

```

## 📄 **FILE 111 of 180**: include/sentio/sizer.hpp

**File Information**:
- **Path**: `include/sentio/sizer.hpp`

- **Size**: 102 lines
- **Modified**: 2025-09-17 15:47:09

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

## 📄 **FILE 112 of 180**: include/sentio/strategy/intraday_position_governor.hpp

**File Information**:
- **Path**: `include/sentio/strategy/intraday_position_governor.hpp`

- **Size**: 210 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <deque>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sentio {

/**
 * IntradayPositionGovernor - Dynamic day trading position management
 * 
 * Converts probability stream to target weights with:
 * 1. Dynamic percentile-based thresholds (no fixed config)
 * 2. Time-based position decay for graceful EOD exit
 * 3. Signal-adaptive sizing for 10-100 trades/day target
 */
class IntradayPositionGovernor {
public:
    struct Config {
        size_t lookback_window = 60;      // Rolling window for dynamic thresholds (minutes)
        double buy_percentile = 0.85;     // Enter long if p > 85th percentile of recent values
        double sell_percentile = 0.15;    // Enter short if p < 15th percentile
        double max_base_weight = 1.0;     // Maximum position weight before time decay
        double trading_day_hours = 6.5;   // NYSE trading hours (9:30-4:00 ET)
        double min_signal_gap = 0.02;     // Minimum gap between buy/sell thresholds
        double min_abs_edge = 0.05;       // Minimum absolute edge from 0.5 to trade (noise filter)
    };

    IntradayPositionGovernor() : IntradayPositionGovernor(Config{}) {}
    
    explicit IntradayPositionGovernor(const Config& cfg) 
        : config_(cfg)
        , market_open_epoch_(0)
        , trading_day_duration_sec_(cfg.trading_day_hours * 3600.0)
        , last_target_weight_(0.0)
    {}

    /**
     * Main decision function: Convert probability + timestamp → target weight
     * 
     * @param probability IRE ensemble probability [0.0, 1.0]
     * @param current_epoch UTC timestamp of current bar
     * @return Target portfolio weight [-1.0, +1.0] with time decay applied
     */
    double calculate_target_weight(double probability, int64_t current_epoch) {
        // Initialize market open time on first call each day
        detect_new_trading_day(current_epoch);
        
        // Update rolling probability window for dynamic thresholds
        update_probability_history(probability);
        
        // Need sufficient history for reliable thresholds
        if (rolling_p_values_.size() < config_.lookback_window) {
            return apply_time_decay(0.0, current_epoch); // Conservative start
        }

        // Calculate adaptive thresholds based on recent market conditions
        auto [buy_threshold, sell_threshold] = calculate_dynamic_thresholds();
        
        // Determine base position weight from signal strength
        double base_weight = calculate_base_weight(probability, buy_threshold, sell_threshold);
        
        // Apply time decay for graceful position closure toward market close
        double final_weight = apply_time_decay(base_weight, current_epoch);
        
        last_target_weight_ = final_weight;
        return final_weight;
    }

    // Diagnostics for strategy analysis
    double get_current_buy_threshold() const {
        if (rolling_p_values_.size() < config_.lookback_window) return 0.85;
        return calculate_dynamic_thresholds().first;
    }
    
    double get_current_sell_threshold() const {
        if (rolling_p_values_.size() < config_.lookback_window) return 0.15;
        return calculate_dynamic_thresholds().second;
    }
    
    double get_time_decay_multiplier(int64_t current_epoch) const {
        if (market_open_epoch_ == 0) return 1.0;
        double time_into_day = static_cast<double>(current_epoch - market_open_epoch_);
        double time_ratio = std::clamp(time_into_day / trading_day_duration_sec_, 0.0, 1.0);
        return std::cos(time_ratio * M_PI / 2.0); // Smooth decay to 0 at market close
    }

    // Reset for new trading session
    void reset_for_new_day() {
        rolling_p_values_.clear();
        market_open_epoch_ = 0;
        last_target_weight_ = 0.0;
    }

private:
    Config config_;
    int64_t market_open_epoch_;
    double trading_day_duration_sec_;
    double last_target_weight_;
    std::deque<double> rolling_p_values_;

    void detect_new_trading_day(int64_t current_epoch) {
        // Simple day detection - could be enhanced with proper calendar
        const int64_t SECONDS_PER_DAY = 86400;
        int64_t current_day = current_epoch / SECONDS_PER_DAY;
        static int64_t last_day = -1;
        
        if (last_day != current_day) {
            reset_for_new_day();
            market_open_epoch_ = current_epoch;
            last_day = current_day;
        } else if (market_open_epoch_ == 0) {
            market_open_epoch_ = current_epoch;
        }
    }

    void update_probability_history(double probability) {
        if (std::isfinite(probability)) {
            rolling_p_values_.push_back(std::clamp(probability, 0.0, 1.0));
            
            // Maintain fixed window size
            if (rolling_p_values_.size() > config_.lookback_window) {
                rolling_p_values_.pop_front();
            }
        }
    }

    std::pair<double, double> calculate_dynamic_thresholds() const {
        // Copy for sorting without modifying original
        std::vector<double> sorted_p(rolling_p_values_.begin(), rolling_p_values_.end());
        std::sort(sorted_p.begin(), sorted_p.end());
        
        size_t n = sorted_p.size();
        size_t buy_idx = static_cast<size_t>(n * config_.buy_percentile);
        size_t sell_idx = static_cast<size_t>(n * config_.sell_percentile);
        
        // Ensure valid indices
        buy_idx = std::min(buy_idx, n - 1);
        sell_idx = std::min(sell_idx, n - 1);
        
        double buy_threshold = sorted_p[buy_idx];
        double sell_threshold = sorted_p[sell_idx];
        
        // **FIX**: Handle degenerate case when all values are identical
        double signal_range = sorted_p[n-1] - sorted_p[0];
        if (signal_range < 0.001) {  // Nearly constant signals
            // Use fixed thresholds around the constant value
            double center = sorted_p[0];
            buy_threshold = std::min(center + 0.1, 0.9);   // Don't exceed 0.9
            sell_threshold = std::max(center - 0.1, 0.1);  // Don't go below 0.1
        }
        
        // Ensure minimum separation to prevent thrashing
        if (buy_threshold - sell_threshold < config_.min_signal_gap) {
            double mid = (buy_threshold + sell_threshold) * 0.5;
            buy_threshold = mid + config_.min_signal_gap * 0.5;
            sell_threshold = mid - config_.min_signal_gap * 0.5;
        }
        
        // **FIX**: Ensure thresholds are achievable 
        buy_threshold = std::clamp(buy_threshold, 0.0, 0.99);  // Must be < 1.0
        sell_threshold = std::clamp(sell_threshold, 0.01, 1.0); // Must be > 0.0
        
        return {buy_threshold, sell_threshold};
    }

    double calculate_base_weight(double probability, double buy_threshold, double sell_threshold) const {
        // **FIXED**: Now actually use the dynamic thresholds!
        
        // Apply min_abs_edge as an additional filter on top of dynamic thresholds
        double abs_edge_from_neutral = std::abs(probability - 0.5);
        if (abs_edge_from_neutral < config_.min_abs_edge) {
            // Signal too weak - stay flat
            return 0.0; 
        }
        
        // **CORRECTED**: Use dynamic thresholds that adapt to market conditions
        if (probability > buy_threshold) {
            // Strong long signal based on recent probability distribution
            double signal_strength = (probability - 0.5) / 0.5;
            return std::min(config_.max_base_weight, signal_strength);
        } 
        else if (probability < sell_threshold) {
            // Strong short signal based on recent probability distribution  
            double signal_strength = (0.5 - probability) / 0.5;
            return -std::min(config_.max_base_weight, signal_strength);
        }
        else {
            // Signal between thresholds - stay flat
            return 0.0;
        }
    }

    double apply_time_decay(double base_weight, int64_t current_epoch) const {
        if (market_open_epoch_ == 0) return base_weight;
        
        double time_into_day = static_cast<double>(current_epoch - market_open_epoch_);
        double time_ratio = std::clamp(time_into_day / trading_day_duration_sec_, 0.0, 1.0);
        
        // Cosine decay: 1.0 at open → 0.0 at close
        // Provides gentle early decay, aggressive late-day closure
        double time_decay_multiplier = std::cos(time_ratio * M_PI / 2.0);
        
        return base_weight * time_decay_multiplier;
    }
};

} // namespace sentio

```

## 📄 **FILE 113 of 180**: include/sentio/strategy_kochi_ppo.hpp

**File Information**:
- **Path**: `include/sentio/strategy_kochi_ppo.hpp`

- **Size**: 48 lines
- **Modified**: 2025-09-18 12:08:56

- **Type**: .hpp

```text
#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/ml/model_registry.hpp"
#include "sentio/ml/feature_window.hpp"
#include "sentio/router.hpp"
#include "sentio/sizer.hpp"
#include <optional>

namespace sentio {

struct KochiPPOCfg {
  std::string artifacts_dir{"artifacts"};
  std::string model_id{"KochiPPO"};
  std::string version{"v1"};
  bool use_cuda{false};
  double conf_floor{0.05};
};

class KochiPPOStrategy final : public BaseStrategy {
public:
  KochiPPOStrategy();
  explicit KochiPPOStrategy(const KochiPPOCfg& cfg);

  // BaseStrategy interface
  ParameterMap get_default_params() const override;
  ParameterSpace get_param_space() const override;
  void apply_params() override;
  double calculate_probability(const std::vector<Bar>& bars, int current_index) override;
  
  // REMOVED: get_allocation_decisions - AllocationManager handles all instrument decisions
  // REMOVED: get_router_config - AllocationManager handles routing
    // REMOVED: get_sizer_config() - No artificial limits allowed for profit maximization

  // Feed one bar worth of raw features (metadata order, length must match)
  void set_raw_features(const std::vector<double>& raw);

private:
  KochiPPOCfg cfg_;
  ml::ModelHandle handle_;
  ml::FeatureWindow window_;
  std::optional<StrategySignal> last_;

  StrategySignal map_output(const ml::ModelOutput& mo) const;
};

} // namespace sentio



```

## 📄 **FILE 114 of 180**: include/sentio/strategy_signal_or.hpp

**File Information**:
- **Path**: `include/sentio/strategy_signal_or.hpp`

- **Size**: 92 lines
- **Modified**: 2025-09-18 11:18:14

- **Type**: .hpp

```text
#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/signal_or.hpp"
#include "sentio/allocation_manager.hpp"
#include "sentio/signal_utils.hpp"
#include <algorithm>
#include <vector>
#include <memory>
#include <map>

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
    // REMOVED: get_sizer_config() - No artificial limits allowed for profit maximization
    
    // REMOVED: requires_dynamic_allocation - all strategies use same allocation pipeline
    
    // Use hard default from BaseStrategy (no simultaneous positions)
    
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

    // Integrated detector architecture
    std::vector<std::unique_ptr<detectors::IDetector>> detectors_;
    int max_warmup_ = 0;
    double consensus_epsilon_ = 0.05;

    struct AuditContext {
        std::map<std::string, double> detector_probs;
        int long_votes = 0;
        int short_votes = 0;
        double final_probability = 0.5;
    };

    AuditContext run_and_aggregate(const std::vector<Bar>& bars, int idx);

    // Helper methods (legacy simple rules retained for fallback if needed)
    std::vector<RuleOut> evaluate_simple_rules(const std::vector<Bar>& bars, int current_index);
    double calculate_momentum_probability(const std::vector<Bar>& bars, int current_index);
    // **PROFIT MAXIMIZATION**: Old position weight methods removed
};

// Register the strategy with the factory
REGISTER_STRATEGY(SignalOrStrategy, "sigor");

} // namespace sentio

```

## 📄 **FILE 115 of 180**: include/sentio/strategy_tfa.hpp

**File Information**:
- **Path**: `include/sentio/strategy_tfa.hpp`

- **Size**: 68 lines
- **Modified**: 2025-09-18 12:31:29

- **Type**: .hpp

```text
#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/ml/model_registry.hpp"
#include "sentio/ml/feature_window.hpp"
#include "sentio/feature/column_projector.hpp"
#include "sentio/feature/column_projector_safe.hpp"
#include "sentio/tfa/tfa_seq_context.hpp"
#include "sentio/router.hpp"
#include "sentio/sizer.hpp"
#include <optional>
#include <memory>

namespace sentio {

struct TFACfg {
  std::string artifacts_dir{"artifacts"};
  std::string model_id{"TFA"};
  std::string version{"v1"};
  bool use_cuda{false};
  double conf_floor{0.05};
};

class TFAStrategy final : public BaseStrategy {
public:
  TFAStrategy(); // Default constructor for factory
  explicit TFAStrategy(const TFACfg& cfg);

  void set_raw_features(const std::vector<double>& raw);
  void on_bar(const StrategyCtx& ctx, const Bar& b);
  std::optional<StrategySignal> latest() const { return last_; }
  
  // BaseStrategy virtual methods
  ParameterMap get_default_params() const override;
  ParameterSpace get_param_space() const override;
  void apply_params() override;
  double calculate_probability(const std::vector<Bar>& bars, int current_index) override;
  
  // REMOVED: get_allocation_decisions - AllocationManager handles all instrument decisions
  // REMOVED: get_router_config - AllocationManager handles routing
  // REMOVED: get_sizer_config() - No artificial limits allowed for profit maximization
  // REMOVED: requires_dynamic_allocation - all strategies use same allocation pipeline

private:
  TFACfg cfg_;
  ml::ModelHandle handle_;
  ml::FeatureWindow window_;
  std::optional<StrategySignal> last_;
  std::vector<std::vector<double>> feature_buffer_;
  StrategySignal map_output(const ml::ModelOutput& mo) const;
  
  // Feature projection system
  mutable std::unique_ptr<ColumnProjector> projector_;
  mutable std::unique_ptr<ColumnProjectorSafe> projector_safe_;
  mutable bool projector_initialized_{false};
  mutable int expected_feat_dim_{56};
  
  // CRITICAL FIX: Move static state from calculate_probability to class members
  // This prevents data leakage between different test runs and ensures deterministic behavior
  mutable int probability_calls_{0};
  mutable bool seq_context_initialized_{false};
  mutable TfaSeqContext seq_context_;
  mutable std::vector<float> precomputed_probabilities_;
  mutable std::vector<float> probability_history_;
  mutable int cooldown_long_until_{-1};
  mutable int cooldown_short_until_{-1};
};

} // namespace sentio

```

## 📄 **FILE 116 of 180**: include/sentio/strategy_transformer.hpp

**File Information**:
- **Path**: `include/sentio/strategy_transformer.hpp`

- **Size**: 41 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/ml/model_registry.hpp"
#include "sentio/ml/feature_window.hpp"
#include <optional>
#include <memory>

namespace sentio {

struct TransformerCfg {
  std::string artifacts_dir{"artifacts"};
  std::string model_id{"TransAlpha"};
  std::string version{"v1"};
  double conf_floor{0.05};
};

class TransformerSignalStrategy final : public BaseStrategy {
public:
  TransformerSignalStrategy(); // Default constructor for factory
  explicit TransformerSignalStrategy(const TransformerCfg& cfg);

  // BaseStrategy interface
  ParameterMap get_default_params() const override;
  ParameterSpace get_param_space() const override;
  void apply_params() override;
  StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;

  // Push one feature vector per bar, metadata order
  void set_raw_features(const std::vector<double>& raw);

private:
  TransformerCfg cfg_;
  ml::ModelHandle handle_;
  ml::FeatureWindow window_;
  std::optional<StrategySignal> last_;

  StrategySignal map_output(const ml::ModelOutput& mo) const;
  ml::WindowSpec make_window_spec(const ml::ModelSpec& spec) const;
};

} // namespace sentio

```

## 📄 **FILE 117 of 180**: include/sentio/strategy_utils.hpp

**File Information**:
- **Path**: `include/sentio/strategy_utils.hpp`

- **Size**: 65 lines
- **Modified**: 2025-09-10 13:55:46

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

## 📄 **FILE 118 of 180**: include/sentio/sym/leverage_registry.hpp

**File Information**:
- **Path**: `include/sentio/sym/leverage_registry.hpp`

- **Size**: 65 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <unordered_map>
#include <mutex>
#include "sentio/sym/symbol_utils.hpp"

namespace sentio {

// Captures the leveraged instrument's relationship to a base ticker
struct LeverageSpec {
  std::string base;     // e.g., "QQQ"
  float factor{1.f};    // e.g., 3.0 for TQQQ, 1.0 for PSQ (but inverse)
  bool inverse{false};  // true for PSQ/SQQQ (short)
};

// Thread-safe global registry
class LeverageRegistry {
  std::unordered_map<std::string, LeverageSpec> map_; // key: UPPER(symbol)
  std::mutex mu_;
  LeverageRegistry() { seed_defaults_(); }

  void seed_defaults_() {
    // QQQ family (PSQ removed - moderate sell signals now use SHORT QQQ)
    map_.emplace("TQQQ", LeverageSpec{"QQQ", 3.f, false});
    map_.emplace("SQQQ", LeverageSpec{"QQQ", 3.f, true});
    // You can extend similarly for SPY, TSLA, BTC ETFs, etc.
    // Examples:
    // map_.emplace("UPRO", LeverageSpec{"SPY", 3.f, false});
    // map_.emplace("SPXU", LeverageSpec{"SPY", 3.f, true});
    // map_.emplace("TSLQ", LeverageSpec{"TSLA", 1.f, true});
  }

public:
  static LeverageRegistry& instance() {
    static LeverageRegistry x;
    return x;
  }

  void register_leveraged(const std::string& symbol, LeverageSpec spec) {
    std::lock_guard<std::mutex> lk(mu_);
    map_[to_upper(symbol)] = std::move(spec);
  }

  bool lookup(const std::string& symbol, LeverageSpec& out) const {
    const auto key = to_upper(symbol);
    auto it = map_.find(key);
    if (it == map_.end()) return false;
    out = it->second;
    return true;
  }
};

// Convenience helpers
inline bool is_leveraged(const std::string& symbol) {
  LeverageSpec tmp;
  return LeverageRegistry::instance().lookup(symbol, tmp);
}

inline std::string resolve_base(const std::string& symbol) {
  LeverageSpec tmp;
  if (LeverageRegistry::instance().lookup(symbol, tmp)) return tmp.base;
  return to_upper(symbol);
}

} // namespace sentio

```

## 📄 **FILE 119 of 180**: include/sentio/sym/symbol_utils.hpp

**File Information**:
- **Path**: `include/sentio/sym/symbol_utils.hpp`

- **Size**: 10 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <algorithm>

namespace sentio {
inline std::string to_upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::toupper(c); });
  return s;
}
}

```

## 📄 **FILE 120 of 180**: include/sentio/symbol_table.hpp

**File Information**:
- **Path**: `include/sentio/symbol_table.hpp`

- **Size**: 35 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <unordered_map>

namespace sentio {

struct SymbolTable {
  std::vector<std::string> id2sym;
  std::unordered_map<std::string,int> sym2id;

  int intern(const std::string& s){
    auto it = sym2id.find(s);
    if (it != sym2id.end()) return it->second;
    int id = (int)id2sym.size();
    id2sym.push_back(s);
    sym2id.emplace(id2sym.back(), id);
    return id;
  }

  const std::string& get_symbol(int id) const {
    return id2sym[id];
  }

  int get_id(const std::string& sym) const {
    auto it = sym2id.find(sym);
    return it != sym2id.end() ? it->second : -1;
  }

  size_t size() const {
    return id2sym.size();
  }
};

} // namespace sentio
```

## 📄 **FILE 121 of 180**: include/sentio/test_strategy.hpp

**File Information**:
- **Path**: `include/sentio/test_strategy.hpp`

- **Size**: 92 lines
- **Modified**: 2025-09-18 12:31:29

- **Type**: .hpp

```text
#pragma once

#include "sentio/base_strategy.hpp"
#include <vector>
#include <random>

namespace sentio {

// **TEST STRATEGY**: Configurable accuracy for backend testing
// This strategy "cheats" by looking ahead to generate signals with known accuracy

struct TestStrategyConfig {
    double target_accuracy = 0.60;     // Target accuracy (0.0 to 1.0)
    int lookhead_bars = 1;              // How many bars to look ahead
    double signal_threshold = 0.02;     // Minimum return to generate signal
    double noise_factor = 0.1;          // Random noise added to signals
    bool enable_leverage = true;        // Whether to use leveraged instruments
    
    // Signal generation parameters
    double strong_signal_threshold = 0.75;  // Threshold for strong signals (3x leverage)
    double weak_signal_threshold = 0.55;    // Threshold for weak signals (1x)
    double neutral_band = 0.05;             // Neutral zone around 0.5
    
    // Accuracy control
    bool perfect_foresight = false;     // If true, always predict correctly
    unsigned int random_seed = 42;      // For reproducible results
};

class TestStrategy : public BaseStrategy {
private:
    TestStrategyConfig config_;
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<double> uniform_dist_;
    
    // State tracking
    mutable std::vector<double> signal_history_;
    mutable std::vector<double> accuracy_history_;
    mutable int last_signal_bar_ = -1;
    
    // Helper methods
    double calculate_future_return(const std::vector<Bar>& bars, int current_index) const;
    double generate_signal_with_accuracy(double true_signal, double target_accuracy) const;
    double add_signal_noise(double signal) const;
    bool should_use_leverage(double signal_strength) const;
    
public:
    TestStrategy(const TestStrategyConfig& config = TestStrategyConfig{});
    
    // **BaseStrategy Interface Implementation**
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override;
    
    // REMOVED: get_allocation_decisions - AllocationManager handles all instrument decisions
    // REMOVED: get_router_config - AllocationManager handles routing
    
    // **Strategy-Specific Conflict Rules**
    bool allows_simultaneous_positions(const std::string& instrument1, const std::string& instrument2) const override;
    bool requires_sequential_transitions() const override { return true; }
    
    // **Parameter Management**
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    // **Test Strategy Specific Methods**
    void set_target_accuracy(double accuracy);
    double get_actual_accuracy() const;
    std::vector<double> get_signal_history() const override { return signal_history_; }
    const std::vector<double>& get_accuracy_history() const { return accuracy_history_; }
    
    // **Configuration**
    void update_config(const TestStrategyConfig& config);
    const TestStrategyConfig& get_config() const { return config_; }
    
    // **Reset for new test**
    void reset_test_state();
};

// **TEST STRATEGY FACTORY**: Create strategies with different accuracies
class TestStrategyFactory {
public:
    static std::unique_ptr<TestStrategy> create_random_strategy(double accuracy = 0.50);
    static std::unique_ptr<TestStrategy> create_poor_strategy(double accuracy = 0.40);
    static std::unique_ptr<TestStrategy> create_decent_strategy(double accuracy = 0.60);
    static std::unique_ptr<TestStrategy> create_good_strategy(double accuracy = 0.75);
    static std::unique_ptr<TestStrategy> create_excellent_strategy(double accuracy = 0.90);
    static std::unique_ptr<TestStrategy> create_perfect_strategy();
    
    // **Batch Testing**: Create multiple strategies with different accuracies
    static std::vector<std::unique_ptr<TestStrategy>> create_accuracy_test_suite();
};

} // namespace sentio

```

## 📄 **FILE 122 of 180**: include/sentio/tfa/artifacts_loader.hpp

**File Information**:
- **Path**: `include/sentio/tfa/artifacts_loader.hpp`

- **Size**: 106 lines
- **Modified**: 2025-09-15 15:41:51

- **Type**: .hpp

```text
#pragma once
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <iostream>

namespace sentio::tfa {

struct TfaArtifacts {
  torch::jit::Module model;
  nlohmann::json spec;
  nlohmann::json meta;
  
  // Convenience getters
  std::vector<std::string> get_expected_feature_names() const {
    return meta["expects"]["feature_names"].get<std::vector<std::string>>();
  }
  
  int get_expected_input_dim() const {
    return meta["expects"]["input_dim"].get<int>();
  }
  
  std::string get_spec_hash() const {
    return meta["expects"]["spec_hash"].get<std::string>();
  }
  
  float get_pad_value() const {
    return meta["expects"]["pad_value"].get<float>();
  }
  
  int get_emit_from() const {
    return meta["expects"]["emit_from"].get<int>();
  }
};

inline TfaArtifacts load_tfa(const std::string& model_pt,
                             const std::string& feature_spec_json,
                             const std::string& model_meta_json)
{
  TfaArtifacts A;
  
  A.model = torch::jit::load(model_pt, torch::kCPU);
  A.model.eval();
  
  std::ifstream fs(feature_spec_json); 
  if(!fs) throw std::runtime_error("missing feature_spec.json: " + feature_spec_json);
  fs >> A.spec;
  
  std::ifstream fm(model_meta_json); 
  if(!fm) throw std::runtime_error("missing model.meta.json: " + model_meta_json);
  fm >> A.meta;
  
  // Validate metadata structure
  if (!A.meta.contains("expects")) {
    throw std::runtime_error("model.meta.json missing 'expects' section");
  }
  
  auto expects = A.meta["expects"];
  if (!expects.contains("input_dim") || !expects.contains("feature_names") || 
      !expects.contains("spec_hash") || !expects.contains("pad_value") || 
      !expects.contains("emit_from")) {
    throw std::runtime_error("model.meta.json 'expects' section incomplete");
  }
  
  
  return A;
}

// Fallback loader for existing metadata.json (without model.meta.json)
inline TfaArtifacts load_tfa_legacy(const std::string& model_pt,
                                     const std::string& metadata_json)
{
  TfaArtifacts A;
  
  A.model = torch::jit::load(model_pt, torch::kCPU);
  A.model.eval();
  
  std::ifstream fs(metadata_json);
  if(!fs) throw std::runtime_error("missing metadata.json: " + metadata_json);
  
  nlohmann::json legacy_meta;
  fs >> legacy_meta;
  
  // Convert legacy metadata.json to new format
  A.spec = legacy_meta; // Use legacy as spec for now
  
  // Create synthetic model.meta.json structure
  A.meta = {
    {"schema_version", "1.0"},
    {"framework", "torchscript"},
    {"expects", {
      {"input_dim", (int)legacy_meta["feature_names"].size()},
      {"feature_names", legacy_meta["feature_names"]},
      {"spec_hash", "legacy"},
      {"emit_from", 64}, // Default for TFA
      {"pad_value", 0.0f},
      {"dtype", "float32"}
    }}
  };
  
  
  return A;
}

} // namespace sentio::tfa

```

## 📄 **FILE 123 of 180**: include/sentio/tfa/artifacts_safe.hpp

**File Information**:
- **Path**: `include/sentio/tfa/artifacts_safe.hpp`

- **Size**: 113 lines
- **Modified**: 2025-09-15 15:40:46

- **Type**: .hpp

```text
#pragma once
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>

namespace sentio::tfa {

struct TfaArtifactsSafe {
  torch::jit::Module model;
  nlohmann::json spec;
  nlohmann::json meta;
  
  // Convenience getters with validation
  std::vector<std::string> get_expected_feature_names() const {
    if (!meta.contains("expects") || !meta["expects"].contains("feature_names")) {
      throw std::runtime_error("Model metadata missing feature_names");
    }
    return meta["expects"]["feature_names"].get<std::vector<std::string>>();
  }
  
  int get_expected_input_dim() const {
    if (!meta.contains("expects") || !meta["expects"].contains("input_dim")) {
      throw std::runtime_error("Model metadata missing input_dim");
    }
    return meta["expects"]["input_dim"].get<int>();
  }
  
  std::string get_spec_hash() const {
    if (!meta.contains("expects") || !meta["expects"].contains("spec_hash")) {
      throw std::runtime_error("Model metadata missing spec_hash");
    }
    return meta["expects"]["spec_hash"].get<std::string>();
  }
  
  float get_pad_value() const {
    if (!meta.contains("expects") || !meta["expects"].contains("pad_value")) {
      return 0.0f; // Default
    }
    return meta["expects"]["pad_value"].get<float>();
  }
  
  int get_emit_from() const {
    if (!meta.contains("expects") || !meta["expects"].contains("emit_from")) {
      return 64; // Default for TFA
    }
    return meta["expects"]["emit_from"].get<int>();
  }
};

inline TfaArtifactsSafe load_tfa_artifacts_safe(const std::string& model_pt,
                                                const std::string& feature_spec_json,
                                                const std::string& model_meta_json)
{
  TfaArtifactsSafe A;
  
  A.model = torch::jit::load(model_pt, torch::kCPU);
  A.model.eval();
  
  std::ifstream fs(feature_spec_json); 
  if(!fs) throw std::runtime_error("missing feature_spec.json: " + feature_spec_json);
  fs >> A.spec;
  
  std::ifstream fm(model_meta_json); 
  if(!fm) throw std::runtime_error("missing model.meta.json: " + model_meta_json);
  fm >> A.meta;
  
  // Validate metadata structure
  if (!A.meta.contains("expects")) {
    throw std::runtime_error("model.meta.json missing 'expects' section");
  }
  
  auto expects = A.meta["expects"];
  if (!expects.contains("input_dim") || !expects.contains("feature_names") || 
      !expects.contains("spec_hash")) {
    throw std::runtime_error("model.meta.json 'expects' section incomplete");
  }
  
  // Validate spec hash if available
  if (A.spec.contains("content_hash")) {
    std::string spec_hash = A.spec["content_hash"].get<std::string>();
    std::string expected_hash = A.get_spec_hash();
    if (spec_hash != expected_hash) {
    }
  }
  
  
  return A;
}

inline std::vector<std::string> feature_names_from_spec(const nlohmann::json& spec){
  std::vector<std::string> names;
  if (!spec.contains("features")) {
    throw std::runtime_error("Feature spec missing 'features' array");
  }
  
  for (auto& f : spec["features"]){
    if (f.contains("name")) {
      names.push_back(f["name"].get<std::string>());
    } else {
      std::string op = f.value("op", "UNKNOWN");
      std::string src = f.value("source", "");
      std::string w = f.contains("window") ? std::to_string((int)f["window"]) : "";
      std::string k = f.contains("k") ? std::to_string((float)f["k"]) : "";
      names.push_back(op + "_" + src + "_" + w + "_" + k);
    }
  }
  return names;
}

} // namespace sentio::tfa

```

## 📄 **FILE 124 of 180**: include/sentio/tfa/feature_guard.hpp

**File Information**:
- **Path**: `include/sentio/tfa/feature_guard.hpp`

- **Size**: 60 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace sentio::tfa {

struct FeatureGuard {
  int emit_from = 0;
  float pad_value = 0.0f;

  static inline bool is_finite(float x){
    return std::isfinite(x);
  }

  // returns mask: true = usable
  std::vector<uint8_t> build_mask_and_clean(float* X, int64_t rows, int64_t cols) const {
    std::vector<uint8_t> ok(rows, 0);
    // zero/pad early rows, mark them unusable
    for (int64_t r=0; r<std::min<int64_t>(emit_from, rows); ++r){
      for (int64_t c=0; c<cols; ++c) X[r*cols+c] = pad_value;
    }
    // after emit_from: sanitize NaN/Inf
    for (int64_t r=emit_from; r<rows; ++r){
      bool row_ok = true;
      for (int64_t c=0; c<cols; ++c){
        float& v = X[r*cols+c];
        if (!is_finite(v)) { v = 0.0f; row_ok = false; } // clean AND mark not-OK for signal
      }
      ok[r] = row_ok ? 1 : 0;
    }
    return ok;
  }
  
  // Overload for double vectors (from cached features)
  std::vector<uint8_t> build_mask_and_clean(std::vector<std::vector<double>>& features) const {
    const int64_t rows = features.size();
    const int64_t cols = rows > 0 ? features[0].size() : 0;
    std::vector<uint8_t> ok(rows, 0);
    
    // zero/pad early rows, mark them unusable
    for (int64_t r=0; r<std::min<int64_t>(emit_from, rows); ++r){
      for (int64_t c=0; c<cols; ++c) features[r][c] = pad_value;
    }
    
    // after emit_from: sanitize NaN/Inf
    for (int64_t r=emit_from; r<rows; ++r){
      bool row_ok = true;
      for (int64_t c=0; c<cols; ++c){
        double& v = features[r][c];
        if (!std::isfinite(v)) { v = 0.0; row_ok = false; } // clean AND mark not-OK for signal
      }
      ok[r] = row_ok ? 1 : 0;
    }
    return ok;
  }
};

} // namespace sentio::tfa

```

## 📄 **FILE 125 of 180**: include/sentio/tfa/input_shim.hpp

**File Information**:
- **Path**: `include/sentio/tfa/input_shim.hpp`

- **Size**: 55 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <unordered_map>

namespace sentio {

inline std::vector<float> shim_to_expected_input(const float* X_src,
                                                 int64_t rows,
                                                 int64_t F_src,
                                                 const std::vector<std::string>& runtime_names,
                                                 const std::vector<std::string>& expected_names,
                                                 int   F_expected,
                                                 float pad_value = 0.0f)
{
  // Fast path: exact match
  if (F_src == F_expected && runtime_names == expected_names) {
    return std::vector<float>(X_src, X_src + (size_t)rows*(size_t)F_src);
  }

  // Hotfix path: legacy model expects a leading 'ts' column
  const bool model_leads_with_ts = !expected_names.empty() && expected_names.front() == "ts";
  const bool runtime_has_ts      = !runtime_names.empty()   && runtime_names.front()  == "ts";
  if (model_leads_with_ts && !runtime_has_ts && F_src + 1 == F_expected) {
    std::vector<float> out((size_t)rows * (size_t)F_expected, pad_value);
    for (int64_t r=0; r<rows; ++r) {
      float* dst = out.data() + r*F_expected;
      std::memcpy(dst + 1, X_src + r*F_src, sizeof(float) * (size_t)F_src);
      dst[0] = 0.0f; // dummy ts
    }
    std::cerr << "[TFA] HOTFIX: injected dummy 'ts' col to satisfy legacy 56-dim model\n";
    return out;
  }

  // General name-based projection (drops extras, fills missing with pad)
  std::vector<float> out((size_t)rows * (size_t)F_expected, pad_value);
  // build index map
  std::unordered_map<std::string,int> pos;
  pos.reserve(runtime_names.size()*2);
  for (int i=0;i<(int)runtime_names.size();++i) pos[runtime_names[i]] = i;
  for (int64_t r=0; r<rows; ++r) {
    const float* src = X_src + r*F_src;
    float* dst = out.data() + r*F_expected;
    for (int j=0; j<F_expected; ++j) {
      auto it = pos.find(expected_names[j]);
      if (it != pos.end()) dst[j] = src[it->second];
    }
  }
  std::cerr << "[TFA] INFO: name-based projection applied (srcF="<<F_src<<" -> dstF="<<F_expected<<")\n";
  return out;
}

} // namespace sentio

```

## 📄 **FILE 126 of 180**: include/sentio/tfa/signal_pipeline.hpp

**File Information**:
- **Path**: `include/sentio/tfa/signal_pipeline.hpp`

- **Size**: 209 lines
- **Modified**: 2025-09-16 04:18:12

- **Type**: .hpp

```text
#pragma once
#include "sentio/ml/iml_model.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace sentio::tfa {

struct DropCounters {
  int64_t total{0}, not_ready{0}, nan_row{0}, low_conf{0}, session{0}, volume{0}, cooldown{0}, dup{0};
  void log() const {
    std::cout << "[SIG TFA] total="<<total
              << " not_ready="<<not_ready
              << " nan_row="<<nan_row
              << " low_conf="<<low_conf
              << " session="<<session
              << " volume="<<volume
              << " cooldown="<<cooldown
              << " dup="<<dup << std::endl;
  }
};

struct ThresholdPolicy {
  // Either fixed threshold or rolling quantile
  float min_prob = 0.51f;      // fixed - no trade zone 0.49-0.51
  int   q_window = 0;          // if >0, use quantile
  float q_level  = 0.75f;      // 75th percentile

  std::vector<uint8_t> filter(const std::vector<float>& prob) const {
    const int64_t N = (int64_t)prob.size();
    std::vector<uint8_t> keep(N, 0);
    if (q_window <= 0){
      // No trade zone: 0.49 < p < 0.51 (filter out)
      for (int64_t i=0;i<N;++i) {
        keep[i] = (prob[i] >= 0.51f || prob[i] <= 0.49f) ? 1 : 0;
      }
      return keep;
    }
    // rolling quantile
    std::vector<float> win; win.reserve(q_window);
    for (int64_t i=0;i<N;++i){
      int64_t a = std::max<int64_t>(0, i - q_window + 1);
      win.clear();
      for (int64_t j=a;j<=i;++j) win.push_back(prob[j]);
      std::nth_element(win.begin(), win.begin() + (int)(q_level*(win.size()-1)), win.end());
      float thr = win[(int)(q_level*(win.size()-1))];
      keep[i] = (prob[i] >= thr) ? 1 : 0;
    }
    return keep;
  }
};

struct Cooldown {
  int bars = 0; // e.g., 5
  // returns mask where entry allowed, tracking last accepted index
  std::vector<uint8_t> apply(const std::vector<uint8_t>& keep) const {
    if (bars <= 0) return keep;
    std::vector<uint8_t> out(keep.size(), 0);
    int64_t next_ok = 0;
    for (int64_t i=0;i<(int64_t)keep.size(); ++i){
      if (i < next_ok) continue;
      if (keep[i]){ out[i]=1; next_ok = i + bars; }
    }
    return out;
  }
};

// Minimal session & volume filters, customize to your data
struct RowFilters {
  bool rth_only = false;
  double min_volume = 0.0;
  std::vector<uint8_t> session_mask; // 1 if allowed (precomputed per row)
  std::vector<double>  volumes;      // per row

  void ensure_sizes(int64_t N){
    if ((int64_t)session_mask.size()!=N) session_mask.assign(N,1);
    if ((int64_t)volumes.size()!=N) volumes.assign(N, 1.0);
  }
};

struct TfaSignalPipeline {
  ml::IModel* model{nullptr}; // Model interface, returns score/prob
  ThresholdPolicy policy;
  Cooldown cooldown;
  RowFilters rowf;

  struct Result {
    std::vector<uint8_t> emit;     // 1=emit entry
    std::vector<float>   prob;     // model output (after activation)
    DropCounters drops;
  };

  static inline float sigmoid(float x){ return 1.f / (1.f + std::exp(-x)); }

  Result run(float* X, int64_t rows, int64_t cols,
             const std::vector<uint8_t>& ready_mask,
             bool model_outputs_logit=true)
  {
    Result R;
    R.prob.assign(rows, 0.f);
    R.emit.assign(rows, 0);
    R.drops.total = rows;

    // 1) forward model using IModel interface
    for (int64_t i=0; i<rows; ++i){
      std::vector<float> features(cols);
      for (int64_t j=0; j<cols; ++j) {
        features[j] = X[i*cols + j];
      }
      
      auto output = model->predict(features, 1, cols, "BF"); // Single row prediction
      if (output && !output->probs.empty()) {
        float v = output->probs[0]; // Assume single output probability
        R.prob[i] = model_outputs_logit ? sigmoid(v) : v;
      } else if (output) {
        float v = output->score; // Fallback to score
        R.prob[i] = model_outputs_logit ? sigmoid(v) : v;
      } else {
        R.prob[i] = 0.5f; // Default neutral probability
      }
    }

    // 2) ready vs not_ready
    std::vector<uint8_t> keep(rows, 0);
    for (int64_t i=0;i<rows;++i){
      if (!ready_mask[i]) { R.drops.not_ready++; continue; }
      keep[i] = 1;
    }

    // 3) session / volume
    rowf.ensure_sizes(rows);
    for (int64_t i=0;i<rows;++i){
      if (!keep[i]) continue;
      if (!rowf.session_mask[i]){ keep[i]=0; R.drops.session++; continue; }
      if (rowf.volumes[i] <= rowf.min_volume){ keep[i]=0; R.drops.volume++; continue; }
    }

    // 4) thresholding
    auto conf_keep = policy.filter(R.prob);
    for (int64_t i=0;i<rows;++i){
      if (!keep[i]) continue;
      if (!conf_keep[i]){ keep[i]=0; R.drops.low_conf++; }
    }

    // 5) cooldown
    keep = cooldown.apply(keep);
    // count cooldown drops (approx)
    // (We can estimate: entries removed between pre/post)
    // For transparency, compute:
    {
      int64_t pre=0, post=0;
      for (auto v: conf_keep) if (v) pre++;
      for (auto v: keep) if (v) post++;
      R.drops.cooldown += std::max<int64_t>(0, pre - post);
    }

    R.emit = std::move(keep);
    return R;
  }
  
  // Overload for vector<vector<double>> from cached features
  Result run_cached(const std::vector<std::vector<double>>& features,
                    const std::vector<uint8_t>& ready_mask,
                    bool model_outputs_logit=true)
  {
    const int64_t rows = features.size();
    if (rows == 0) {
      Result R;
      R.drops.total = 0;
      return R;
    }
    
    const int64_t cols = features[0].size();
    if (cols == 0) {
      Result R;
      R.drops.total = rows;
      R.drops.nan_row = rows;
      return R;
    }
    
    // Safety check: ensure all rows have same column count
    for (int64_t r = 0; r < rows; ++r) {
      if ((int64_t)features[r].size() != cols) {
        std::cout << "[ERROR] TfaSignalPipeline: Row " << r << " has " << features[r].size() 
                  << " features, expected " << cols << std::endl;
        Result R;
        R.drops.total = rows;
        R.drops.nan_row = rows;
        return R;
      }
    }
    
    // Convert to float array for model
    std::vector<float> X_flat(rows * cols);
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        X_flat[r * cols + c] = static_cast<float>(features[r][c]);
      }
    }
    
    return run(X_flat.data(), rows, cols, ready_mask, model_outputs_logit);
  }
};

} // namespace sentio::tfa

```

## 📄 **FILE 127 of 180**: include/sentio/tfa/tfa_seq_context.hpp

**File Information**:
- **Path**: `include/sentio/tfa/tfa_seq_context.hpp`

- **Size**: 101 lines
- **Modified**: 2025-09-15 15:57:19

- **Type**: .hpp

```text
#pragma once
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <cstring>

#include "sentio/feature/feature_from_spec.hpp"
#include "sentio/feature/column_projector.hpp"
#include "sentio/feature/sanitize.hpp"

namespace sentio {

struct TfaSeqContext {
  torch::jit::Module model;  nlohmann::json spec, meta;
  std::vector<std::string> runtime_names, expected_names;
  int F{55}, T{64}, emit_from{64}; float pad_value{0.f};

  static nlohmann::json load_json(const std::string& p){ std::ifstream f(p); nlohmann::json j; f>>j; return j; }
  static std::vector<std::string> names_from_spec(const nlohmann::json& spec){
    std::vector<std::string> out; out.reserve(spec["features"].size());
    for (auto& f: spec["features"]){
      if (f.contains("name")) out.push_back(f["name"].get<std::string>());
      else {
        std::string op=f["op"].get<std::string>(), src=f.value("source","");
        std::string w=f.contains("window")?std::to_string((int)f["window"]):"";
        std::string k=f.contains("k")?std::to_string((float)f["k"]):"";
        out.push_back(op+"_"+src+"_"+w+"_"+k);
      }
    }
    return out;
  }

  void load(const std::string& model_pt, const std::string& spec_json, const std::string& meta_json){
    model = torch::jit::load(model_pt, torch::kCPU); model.eval();
    spec  = load_json(spec_json);
    meta  = load_json(meta_json);

    runtime_names  = names_from_spec(spec);
    expected_names = meta["expects"]["feature_names"].get<std::vector<std::string>>();
    F         = meta["expects"]["input_dim"].get<int>();
    if (meta["expects"].contains("seq_len")) T = meta["expects"]["seq_len"].get<int>();
    emit_from = meta["expects"]["emit_from"].get<int>();
    pad_value = meta["expects"]["pad_value"].get<float>();

    if (F!=55) std::cerr << "[WARN] model F="<<F<<" expected 55\n";
  }

  template<class Bars>
  void forward_probs(const std::string& symbol, const Bars& bars, std::vector<float>& probs_out)
  {
    // Build features [N,F]
    auto X = sentio::build_features_from_spec_json(symbol, bars, spec.dump());
    // Project if needed
    std::vector<float> Xproj;
    const float* Xp = X.data.data(); int64_t Fs = X.cols;
    if (!(Fs==F && runtime_names==expected_names)){
      auto proj = sentio::ColumnProjector::make(runtime_names, expected_names, pad_value);
      proj.project(X.data.data(), (size_t)X.rows, (size_t)X.cols, Xproj);
      Xp = Xproj.data(); Fs = F;
    }

    // Sanitize
    auto ready = sentio::sanitize_and_ready(const_cast<float*>(Xp), X.rows, Fs, emit_from, pad_value);

    // Slide windows → batch inference
    probs_out.assign((size_t)X.rows, 0.5f);
    torch::NoGradGuard ng; torch::InferenceMode im;
    const int64_t B = 256;
    const int64_t start = std::max<int64_t>({emit_from, T-1});
    const int64_t last  = X.rows - 1;

    for (int64_t i=start; i<=last; ){
      int64_t j = std::min<int64_t>(last+1, i+B);
      int64_t L = j - i;
      auto t = torch::empty({L, T, F}, torch::kFloat32);
      float* dst = t.data_ptr<float>();
      for (int64_t k=0;k<L;++k){
        int64_t idx=i+k, lo=idx-T+1;
        std::memcpy(dst + k*T*F, Xp + lo*F, sizeof(float)*(size_t)(T*F));
      }
      auto y = model.forward({t}).toTensor(); // [L,1] logits
      if (y.dim()==2 && y.size(1)==1) y=y.squeeze(1);
      float* lp = y.contiguous().data_ptr<float>();
      for (int64_t k=0;k<L;++k)
        probs_out[(size_t)(i+k)] = 1.f/(1.f+std::exp(-lp[k])); // sigmoid
      i = j;
    }

    // Stats
    float pmin=1.f, pmax=0.f, ps=0.f; int64_t cnt=0;
    for (int64_t i=start;i<(int64_t)probs_out.size();++i){ pmin=std::min(pmin,probs_out[i]); pmax=std::max(pmax,probs_out[i]); ps+=probs_out[i]; cnt++; }
  }
};

} // namespace sentio



```

## 📄 **FILE 128 of 180**: include/sentio/time_utils.hpp

**File Information**:
- **Path**: `include/sentio/time_utils.hpp`

- **Size**: 18 lines
- **Modified**: 2025-09-16 00:39:14

- **Type**: .hpp

```text
#pragma once
#include <chrono>
#include <string>
#include <variant>

namespace sentio {

// Normalize various timestamp representations to UTC epoch seconds.
std::chrono::sys_seconds to_utc_sys_seconds(const std::variant<std::int64_t, double, std::string>& ts);

// Helpers exposed for tests
bool iso8601_looks_like(const std::string& s);
bool epoch_ms_suspected(double v_ms);

// Calculate start date for data downloads
std::string calculate_start_date(int years, int months, int days);

} // namespace sentio
```

## 📄 **FILE 129 of 180**: include/sentio/torch/safe_from_blob.hpp

**File Information**:
- **Path**: `include/sentio/torch/safe_from_blob.hpp`

- **Size**: 48 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <torch/torch.h>
#include <memory>
#include <vector>

namespace sentio {

// Creates a Tensor that OWNS a heap buffer and will free it when Tensor dies.
// If you already have a std::shared_ptr<float> backing store, prefer that version.
inline torch::Tensor own_copy_tensor(const float* src, int64_t rows, int64_t cols) {
  auto t = torch::empty({rows, cols}, torch::dtype(torch::kFloat32));
  t.copy_(torch::from_blob((void*)src, {rows, cols}, torch::kFloat32)); // safe copy
  return t;
}

// If you insist on zero-copy, give Tensor a deleter tied to a shared_ptr:
inline torch::Tensor tensor_from_shared(std::shared_ptr<std::vector<float>> store, int64_t rows, int64_t cols) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  return torch::from_blob(
      (void*)store->data(),
      {rows, cols},
      [store](void*) mutable { store.reset(); }, // keep alive
      options);
}

// Safe batched forward pass with owned tensors
inline std::vector<float> model_forward_probs(torch::jit::Module& m, const float* X, int64_t rows, int64_t cols, bool logits=true){
  std::vector<float> probs((size_t)rows, 0.f);
  torch::NoGradGuard ng; 
  torch::InferenceMode im;
  const int64_t B = 8192;
  for (int64_t i=0;i<rows;i+=B){
    int64_t b = std::min<int64_t>(B, rows-i);
    auto t = torch::from_blob((void*)(X + i*cols), {b, cols}, torch::kFloat32).clone(); // OWNED
    t = t.contiguous(); // belt & suspenders
    auto y = m.forward({t}).toTensor();
    if (y.dim()==2 && y.size(1)==1) y = y.squeeze(1);
    if (y.dim()!=1 || y.size(0)!=b) throw std::runtime_error("model output shape mismatch");
    auto acc = y.contiguous().data_ptr<float>();
    for (int64_t k=0;k<b;++k){
      float v = acc[k];
      probs[(size_t)(i+k)] = logits ? 1.f/(1.f+std::exp(-v)) : v;
    }
  }
  return probs;
}

} // namespace sentio

```

## 📄 **FILE 130 of 180**: include/sentio/unified_metrics.hpp

**File Information**:
- **Path**: `include/sentio/unified_metrics.hpp`

- **Size**: 124 lines
- **Modified**: 2025-09-15 20:06:17

- **Type**: .hpp

```text
#pragma once

#include "sentio/metrics.hpp"
#include "sentio/cost_model.hpp"
#include "sentio/runner.hpp" // For BacktestOutput
#include <vector>
#include <string>

// Forward declaration to avoid circular dependency
namespace audit {
    struct Event;
}

namespace sentio {

// NEW: This is the canonical, final report struct. All systems will use this.
struct UnifiedMetricsReport {
    double final_equity;
    double total_return;
    double sharpe_ratio;
    double max_drawdown;
    double monthly_projected_return; // Canonical MPR
    double avg_daily_trades;
    int total_fills;
};

/**
 * Unified Metrics Calculator - Single Source of Truth for Performance Metrics
 * 
 * This class provides consistent performance calculation across all systems:
 * - TPA (Temporal Performance Analysis)
 * - Audit System
 * - Live Trading Monitoring
 * 
 * Uses the statistically robust compute_metrics_day_aware methodology
 * with proper compound interest calculations and Alpaca fee modeling.
 */
class UnifiedMetricsCalculator {
public:
    /**
     * NEW: Primary method to generate a unified report from raw backtest data.
     * This is the single source of truth for all metric calculations.
     * 
     * @param output Raw backtest output containing equity curve and trade statistics
     * @return UnifiedMetricsReport with all canonical performance metrics
     */
    static UnifiedMetricsReport calculate_metrics(const BacktestOutput& output);
    
    /**
     * CHANGED: This is now a helper method used by the primary one.
     * Calculate performance metrics from equity curve
     * 
     * @param equity_curve Vector of (timestamp, equity_value) pairs
     * @param fills_count Number of fill events for trade statistics
     * @param include_fees Whether to account for transaction fees in calculations
     * @return RunSummary with all performance metrics
     */
    static RunSummary calculate_from_equity_curve(
        const std::vector<std::pair<std::string, double>>& equity_curve,
        int fills_count,
        bool include_fees = true
    );
    
    /**
     * Calculate performance metrics from audit events
     * Reconstructs equity curve from fill events and applies unified calculation
     * 
     * @param events Vector of audit events (fills, orders, etc.)
     * @param initial_capital Starting capital amount
     * @param include_fees Whether to apply Alpaca fee model
     * @return RunSummary with all performance metrics
     */
    static RunSummary calculate_from_audit_events(
        const std::vector<audit::Event>& events,
        double initial_capital = 100000.0,
        bool include_fees = true
    );
    
    /**
     * Reconstruct equity curve from audit fill events
     * Used by audit system to create consistent equity progression
     * 
     * @param events Vector of audit events
     * @param initial_capital Starting capital
     * @param include_fees Whether to apply transaction fees
     * @return Vector of (timestamp, equity_value) pairs
     */
    static std::vector<std::pair<std::string, double>> reconstruct_equity_curve_from_events(
        const std::vector<audit::Event>& events,
        double initial_capital = 100000.0,
        bool include_fees = true
    );
    
    /**
     * Calculate transaction fees for a trade using Alpaca cost model
     * 
     * @param symbol Trading symbol
     * @param quantity Trade quantity (positive for buy, negative for sell)
     * @param price Execution price
     * @return Total transaction fees
     */
    static double calculate_transaction_fees(
        const std::string& symbol,
        double quantity,
        double price
    );
    
    /**
     * Validate metrics consistency between two calculations
     * Used for testing and verification
     * 
     * @param metrics1 First metrics calculation
     * @param metrics2 Second metrics calculation
     * @param tolerance_pct Acceptable difference percentage (default 1%)
     * @return True if metrics are consistent within tolerance
     */
    static bool validate_metrics_consistency(
        const RunSummary& metrics1,
        const RunSummary& metrics2,
        double tolerance_pct = 1.0
    );
};

} // namespace sentio

```

## 📄 **FILE 131 of 180**: include/sentio/unified_strategy_tester.hpp

**File Information**:
- **Path**: `include/sentio/unified_strategy_tester.hpp`

- **Size**: 262 lines
- **Modified**: 2025-09-16 13:10:27

- **Type**: .hpp

```text
#pragma once

#include "sentio/core.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/runner.hpp"
#include "sentio/virtual_market.hpp"
#include "sentio/mars_data_loader.hpp"
#include "sentio/unified_metrics.hpp"
#include <vector>
#include <string>
#include <memory>
#include <map>

namespace sentio {

/**
 * Unified Strategy Testing Framework
 * 
 * Consolidates vmtest, marstest, and fasttest into a single comprehensive
 * strategy robustness evaluation system focused on Alpaca live trading readiness.
 */
class UnifiedStrategyTester {
public:
    enum class TestMode {
        MONTE_CARLO,    // Pure synthetic data with regime switching
        HISTORICAL,     // Historical patterns with synthetic continuation
        AI_REGIME,      // MarS AI-powered market simulation
        HYBRID          // Combines all three approaches (default)
    };
    
    enum class RiskLevel {
        LOW,
        MEDIUM, 
        HIGH,
        EXTREME
    };
    
    struct ConfidenceInterval {
        double lower;
        double upper;
        double mean;
        double std_dev;
    };
    
    struct TestConfig {
        // Core Parameters
        std::string strategy_name;
        std::string symbol;
        TestMode mode = TestMode::HYBRID;
        int simulations = 50;
        std::string duration = "5d";
        
        // Data Sources
        std::string historical_data_file;
        std::string regime = "normal";
        bool market_hours_only = true;
        
        // Dataset Information for Reporting
        std::string dataset_source = "unknown";
        std::string dataset_period = "unknown";
        std::string test_period = "unknown";
        
        // Robustness Testing
        bool stress_test = false;
        bool regime_switching = false;
        bool liquidity_stress = false;
        double volatility_min = 0.0;
        double volatility_max = 0.0;
        
        // Alpaca Integration
        bool alpaca_fees = true;
        bool alpaca_limits = false;
        bool paper_validation = false;
        
        // Output & Analysis
        double confidence_level = 0.95;
        std::string output_format = "console";
        std::string save_results_file;
        std::string benchmark_symbol = "SPY";
        
        // Performance
        int parallel_jobs = 0; // 0 = auto-detect
        bool quick_mode = false;
        bool comprehensive_mode = false;
        bool holistic_mode = false;
        bool disable_audit_logging = false;
        
        // Strategy Parameters
        std::string params_json = "{}";
        double initial_capital = 100000.0;
    };
    
    struct RobustnessReport {
        // Core Metrics (Alpaca-focused)
        double monthly_projected_return;
        double sharpe_ratio;
        double max_drawdown;
        double win_rate;
        double profit_factor;
        double total_return;
        
        // Robustness Metrics
        double consistency_score;      // Performance consistency across simulations
        double regime_adaptability;    // Performance across different market regimes
        double stress_resilience;      // Performance under stress conditions
        double liquidity_tolerance;    // Performance with liquidity constraints
        
        // Alpaca-Specific Metrics
        double estimated_monthly_fees;
        double capital_efficiency;     // Return per dollar of capital used
        double avg_daily_trades;
        double position_turnover;
        
        // Risk Assessment
        RiskLevel overall_risk;
        std::vector<std::string> risk_warnings;
        std::vector<std::string> recommendations;
        
        // Confidence Intervals
        ConfidenceInterval mpr_ci;
        ConfidenceInterval sharpe_ci;
        ConfidenceInterval drawdown_ci;
        ConfidenceInterval win_rate_ci;
        
        // Simulation Details
        int total_simulations;
        int successful_simulations;
        double test_duration_seconds;
        TestMode mode_used;
        
        // Deployment Readiness
        int deployment_score;          // 0-100 overall readiness score
        bool ready_for_deployment;
        std::string recommended_capital_range;
        int suggested_monitoring_days;
        
        // Audit Information
        std::string run_id;            // Run ID for audit verification
        
        // Dataset Information
        std::string dataset_source = "unknown";
        std::string dataset_period = "unknown";
        std::string test_period = "unknown";
    };
    
    UnifiedStrategyTester();
    ~UnifiedStrategyTester() = default;
    
    /**
     * Run comprehensive strategy robustness test
     */
    RobustnessReport run_comprehensive_test(const TestConfig& config);
    
    /**
     * Print detailed robustness report to console
     */
    void print_robustness_report(const RobustnessReport& report, const TestConfig& config);
    
    /**
     * Save report to file in specified format
     */
    bool save_report(const RobustnessReport& report, const TestConfig& config, 
                     const std::string& filename, const std::string& format);
    
    /**
     * Parse duration string (e.g., "1h", "5d", "2w", "1m")
     */
    static int parse_duration_to_minutes(const std::string& duration);
    
    /**
     * Parse test mode from string
     */
    static TestMode parse_test_mode(const std::string& mode_str);
    
    /**
     * Get default historical data file for symbol
     */
    static std::string get_default_historical_file(const std::string& symbol);
    
    /**
     * Get dataset period from data file
     */
    static std::string get_dataset_period(const std::string& file_path);
    
    /**
     * Get test period from data file
     */
    static std::string get_test_period(const std::string& file_path, int continuation_minutes);

private:
    // Simulation Engines
    VirtualMarketEngine vm_engine_;
    
    // Analysis Components
    // UnifiedMetricsCalculator metrics_calculator_; // TODO: Implement when needed
    
    /**
     * Run Monte Carlo simulations
     */
    std::vector<VirtualMarketEngine::VMSimulationResult> run_monte_carlo_tests(
        const TestConfig& config, int num_simulations);
    
    /**
     * Run historical pattern tests
     */
    std::vector<VirtualMarketEngine::VMSimulationResult> run_historical_tests(
        TestConfig& config, int num_simulations);
    
    /**
     * Run AI regime tests
     */
    std::vector<VirtualMarketEngine::VMSimulationResult> run_ai_regime_tests(
        TestConfig& config, int num_simulations);
    
    /**
     * Run hybrid tests (combination of all modes)
     */
    std::vector<VirtualMarketEngine::VMSimulationResult> run_hybrid_tests(
        TestConfig& config);
    
    /**
     * Run holistic tests (comprehensive multi-scenario testing)
     */
    std::vector<VirtualMarketEngine::VMSimulationResult> run_holistic_tests(
        TestConfig& config);
    
    /**
     * Analyze simulation results for robustness metrics
     */
    RobustnessReport analyze_results(
        const std::vector<VirtualMarketEngine::VMSimulationResult>& results,
        const TestConfig& config);
    
    /**
     * Calculate confidence intervals for metrics
     */
    ConfidenceInterval calculate_confidence_interval(
        const std::vector<double>& values, double confidence_level);
    
    /**
     * Assess overall risk level
     */
    RiskLevel assess_risk_level(const RobustnessReport& report);
    
    /**
     * Generate risk warnings and recommendations
     */
    void generate_recommendations(RobustnessReport& report, const TestConfig& config);
    
    /**
     * Calculate deployment readiness score
     */
    int calculate_deployment_score(const RobustnessReport& report);
    
    /**
     * Apply stress testing scenarios
     */
    void apply_stress_scenarios(std::vector<VirtualMarketEngine::VMSimulationResult>& results,
                               const TestConfig& config);
};

} // namespace sentio

```

## 📄 **FILE 132 of 180**: include/sentio/util/bytes.hpp

**File Information**:
- **Path**: `include/sentio/util/bytes.hpp`

- **Size**: 24 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace sentio {

// Safe memory copy with bounds checking
inline void bytes_copy(void* dst, const void* src, size_t count){
  if (!dst || !src) throw std::runtime_error("bytes_copy: null ptr");
  if (count > 0) {
    std::memcpy(dst, src, count);
  }
}

// Safe memory set with bounds checking
inline void bytes_set(void* ptr, int value, size_t count) {
  if (!ptr) throw std::runtime_error("bytes_set: null ptr");
  if (count > 0) {
    std::memset(ptr, value, count);
  }
}

} // namespace sentio

```

## 📄 **FILE 133 of 180**: include/sentio/util/safe_matrix.hpp

**File Information**:
- **Path**: `include/sentio/util/safe_matrix.hpp`

- **Size**: 38 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <cstring>

namespace sentio {

struct SafeMatrix {
  std::vector<float> buf;
  int64_t rows{0}, cols{0};

  void resize(int64_t r, int64_t c) {
    if (r < 0 || c < 0) throw std::runtime_error("SafeMatrix: negative shape");
    if (c > (int64_t)(std::numeric_limits<size_t>::max()/sizeof(float))/ (r>0?r:1))
      throw std::runtime_error("SafeMatrix: size overflow");
    rows = r; cols = c;
    buf.assign(static_cast<size_t>(r)*static_cast<size_t>(c), 0.0f);
  }

  inline float* row_ptr(int64_t r) {
    if ((uint64_t)r >= (uint64_t)rows) throw std::runtime_error("SafeMatrix: row OOB");
    return buf.data() + (size_t)r*(size_t)cols;
  }
  inline const float* row_ptr(int64_t r) const {
    if ((uint64_t)r >= (uint64_t)rows) throw std::runtime_error("SafeMatrix: row OOB");
    return buf.data() + (size_t)r*(size_t)cols;
  }
  
  // Convenience accessors
  float* data() { return buf.data(); }
  const float* data() const { return buf.data(); }
  size_t size() const { return buf.size(); }
  bool empty() const { return buf.empty(); }
};

} // namespace sentio

```

## 📄 **FILE 134 of 180**: include/sentio/utils/formatting.hpp

**File Information**:
- **Path**: `include/sentio/utils/formatting.hpp`

- **Size**: 95 lines
- **Modified**: 2025-09-11 23:11:24

- **Type**: .hpp

```text
#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

namespace sentio::utils {

/**
 * @brief Formats a value with specified precision
 * @param value The value to format
 * @param precision Number of decimal places
 * @return Formatted string
 */
template<typename T>
std::string format_precision(T value, int precision = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

/**
 * @brief Formats a percentage value
 * @param value The value to format as percentage
 * @param precision Number of decimal places
 * @return Formatted percentage string
 */
template<typename T>
std::string format_percentage(T value, int precision = 2) {
    return format_precision(value * 100.0, precision) + "%";
}

/**
 * @brief Formats a vector of values as a comma-separated string
 * @param values The vector of values
 * @param precision Number of decimal places
 * @return Comma-separated string
 */
template<typename T>
std::string format_vector(const std::vector<T>& values, int precision = 2) {
    if (values.empty()) return "[]";
    
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << format_precision(values[i], precision);
    }
    oss << "]";
    return oss.str();
}

/**
 * @brief Formats a key-value pair
 * @param key The key
 * @param value The value
 * @param precision Number of decimal places for numeric values
 * @return Formatted key-value string
 */
template<typename T>
std::string format_key_value(const std::string& key, T value, int precision = 2) {
    return key + ": " + format_precision(value, precision);
}

/**
 * @brief Formats multiple key-value pairs
 * @param pairs Vector of key-value pairs
 * @param precision Number of decimal places for numeric values
 * @return Comma-separated key-value string
 */
template<typename T>
std::string format_key_values(const std::vector<std::pair<std::string, T>>& pairs, int precision = 2) {
    if (pairs.empty()) return "";
    
    std::ostringstream oss;
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << format_key_value(pairs[i].first, pairs[i].second, precision);
    }
    return oss.str();
}

/**
 * @brief Formats a result summary with key metrics
 * @param metrics Map of metric names to values
 * @param precision Number of decimal places
 * @return Formatted summary string
 */
template<typename T>
std::string format_result_summary(const std::vector<std::pair<std::string, T>>& metrics, int precision = 2) {
    return format_key_values(metrics, precision);
}

} // namespace sentio::utils

```

## 📄 **FILE 135 of 180**: include/sentio/utils/validation.hpp

**File Information**:
- **Path**: `include/sentio/utils/validation.hpp`

- **Size**: 82 lines
- **Modified**: 2025-09-11 23:11:24

- **Type**: .hpp

```text
#pragma once

#include <vector>
#include <string>

namespace sentio::utils {

/**
 * @brief Validates if a series has sufficient data for the given index range
 * @param series The data series to validate
 * @param start_idx The starting index
 * @param end_idx The ending index (optional, defaults to series size)
 * @return true if series has sufficient data, false otherwise
 */
template<typename T>
bool has_sufficient_data(const std::vector<T>& series, int start_idx, int end_idx = -1) {
    if (end_idx == -1) end_idx = static_cast<int>(series.size());
    return static_cast<int>(series.size()) > start_idx && start_idx >= 0 && end_idx > start_idx;
}

/**
 * @brief Validates if a series has sufficient data for the given start index
 * @param series The data series to validate
 * @param start_idx The starting index
 * @return true if series has sufficient data, false otherwise
 */
template<typename T>
bool has_sufficient_data(const std::vector<T>& series, int start_idx) {
    return static_cast<int>(series.size()) > start_idx && start_idx >= 0;
}

/**
 * @brief Validates if a series has sufficient data for volume window
 * @param series The data series to validate
 * @param vol_win The volume window size
 * @return true if series has sufficient data, false otherwise
 */
template<typename T>
bool has_volume_window_data(const std::vector<T>& series, int vol_win) {
    return static_cast<int>(series.size()) > vol_win;
}

/**
 * @brief Safely extracts a sub-range from a series with validation
 * @param series The source series
 * @param start_idx The starting index
 * @param end_idx The ending index
 * @return A new vector containing the sub-range, or empty vector if invalid
 */
template<typename T>
std::vector<T> extract_range(const std::vector<T>& series, int start_idx, int end_idx = -1) {
    if (!has_sufficient_data(series, start_idx, end_idx)) {
        return {};
    }
    
    if (end_idx == -1) end_idx = static_cast<int>(series.size());
    int actual_end = std::min(end_idx, static_cast<int>(series.size()));
    
    return std::vector<T>(series.begin() + start_idx, series.begin() + actual_end);
}

/**
 * @brief Safely extracts multiple sub-ranges from multiple series
 * @param series_vector Vector of series to extract from
 * @param start_idx The starting index
 * @param end_idx The ending index
 * @return Vector of extracted ranges
 */
template<typename T>
std::vector<std::vector<T>> extract_multiple_ranges(const std::vector<std::vector<T>>& series_vector, 
                                                   int start_idx, int end_idx = -1) {
    std::vector<std::vector<T>> result;
    result.reserve(series_vector.size());
    
    for (const auto& series : series_vector) {
        result.emplace_back(extract_range(series, start_idx, end_idx));
    }
    
    return result;
}

} // namespace sentio::utils

```

## 📄 **FILE 136 of 180**: include/sentio/virtual_market.hpp

**File Information**:
- **Path**: `include/sentio/virtual_market.hpp`

- **Size**: 197 lines
- **Modified**: 2025-09-17 15:47:09

- **Type**: .hpp

```text
#pragma once

#include "sentio/core.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/runner.hpp"
#include "sentio/unified_metrics.hpp"
#include <vector>
#include <random>
#include <memory>
#include <chrono>

namespace sentio {

// **PROFIT MAXIMIZATION**: Generate theoretical leverage series for maximum capital deployment
std::vector<Bar> generate_theoretical_leverage_series(const std::vector<Bar>& base_series, double leverage_factor);

/**
 * Virtual Market Engine for Monte Carlo Strategy Testing
 * 
 * Provides synthetic market data generation and Monte Carlo simulation
 * capabilities integrated with the real Sentio strategy components.
 */
class VirtualMarketEngine {
public:
    struct MarketRegime {
        std::string name;
        double volatility;      // Daily volatility (0.01 = 1%)
        double trend;          // Daily trend (-0.05 to +0.05)
        double mean_reversion; // Mean reversion strength (0.0 to 1.0)
        double volume_multiplier; // Volume scaling factor
        int duration_minutes;   // How long this regime lasts
    };

    struct VMSimulationResult {
        // LEGACY: Old calculated metrics (for backward compatibility)
        double total_return;
        double final_capital;
        double sharpe_ratio;
        double max_drawdown;
        double win_rate;
        int total_trades;
        double monthly_projected_return;
        double daily_trades;
        
        // NEW: Raw backtest output for unified metrics calculation
        BacktestOutput raw_output;
        
        // NEW: Unified metrics report (calculated from raw_output)
        UnifiedMetricsReport unified_metrics;
        
        // Audit Information
        std::string run_id;            // Run ID for audit verification
    };

    struct VMTestConfig {
        std::string strategy_name;
        std::string symbol;
        int days = 30;
        int hours = 0;
        int simulations = 100;
        bool fast_mode = true;
        std::string params_json = "{}";
        double initial_capital = 100000.0;
    };

    VirtualMarketEngine();
    ~VirtualMarketEngine() = default;

    /**
     * Generate synthetic market data for a symbol
     */
    std::vector<Bar> generate_market_data(const std::string& symbol, 
                                         int periods, 
                                         int interval_seconds = 60);

    /**
     * Run Monte Carlo simulation with real strategy components
     */
    std::vector<VMSimulationResult> run_monte_carlo_simulation(
        const VMTestConfig& config,
        std::unique_ptr<BaseStrategy> strategy,
        const RunnerCfg& runner_cfg);
    
    /**
     * Run Monte Carlo simulation using MarS-generated realistic data
     */
    std::vector<VMSimulationResult> run_mars_monte_carlo_simulation(
        const VMTestConfig& config,
        std::unique_ptr<BaseStrategy> strategy,
        const RunnerCfg& runner_cfg,
        const std::string& market_regime = "normal");
    
    /**
     * Run MarS virtual market test (convenience method)
     */
    std::vector<VMSimulationResult> run_mars_vm_test(const std::string& strategy_name,
                                                    const std::string& symbol,
                                                    int days,
                                                    int simulations,
                                                    const std::string& market_regime = "normal",
                                                    const std::string& params_json = "");

    /**
     * Run fast historical test using optimized historical patterns
     */
    std::vector<VMSimulationResult> run_fast_historical_test(const std::string& strategy_name,
                                                            const std::string& symbol,
                                                            const std::string& historical_data_file,
                                                            int continuation_minutes,
                                                            int simulations,
                                                            const std::string& params_json = "");

    /**
     * Run single historical test on actual historical data (no simulations)
     */
    std::vector<VMSimulationResult> run_single_historical_test(const std::string& strategy_name,
                                                              const std::string& symbol,
                                                              const std::string& historical_data_file,
                                                              int continuation_minutes,
                                                              const std::string& params_json = "");

    /**
     * Run future QQQ regime test using pre-generated future data
     */
    std::vector<VMSimulationResult> run_future_qqq_regime_test(const std::string& strategy_name,
                                                              const std::string& symbol,
                                                              int simulations,
                                                              const std::string& regime,
                                                              const std::string& params_json = "");

    /**
     * Run single simulation with given market data
     */
    VMSimulationResult run_single_simulation(
        const std::vector<Bar>& market_data,
        std::unique_ptr<BaseStrategy> strategy,
        const RunnerCfg& runner_cfg,
        double initial_capital);

    /**
     * Generate comprehensive test report
     */
    void print_simulation_report(const std::vector<VMSimulationResult>& results,
                                const VMTestConfig& config);

private:
    std::mt19937 rng_;
    std::vector<MarketRegime> market_regimes_;
    std::unordered_map<std::string, double> base_prices_;
    
    MarketRegime current_regime_;
    std::chrono::system_clock::time_point regime_start_time_;
    
    void initialize_market_regimes();
    void select_new_regime();
    bool should_change_regime();
    
    Bar generate_market_bar(const std::string& symbol, 
                           std::chrono::system_clock::time_point timestamp);
    
    double calculate_price_movement(const std::string& symbol, 
                                   double current_price,
                                   const MarketRegime& regime);
    
    VMSimulationResult calculate_performance_metrics(
        const std::vector<double>& returns,
        const std::vector<int>& trades,
        double initial_capital);
};

/**
 * Virtual Market Test Runner
 * 
 * High-level interface for running VM tests with real Sentio components
 */
class VMTestRunner {
public:
    VMTestRunner() = default;
    ~VMTestRunner() = default;

    /**
     * Run virtual market test with real strategy components
     */
    int run_vmtest(const VirtualMarketEngine::VMTestConfig& config);

private:
    VirtualMarketEngine vm_engine_;
    
    std::unique_ptr<BaseStrategy> create_strategy(const std::string& strategy_name,
                                                  const std::string& params_json);
    
    RunnerCfg load_strategy_config(const std::string& strategy_name);
    
    void print_progress(int current, int total, double elapsed_time);
};

} // namespace sentio

```

## 📄 **FILE 137 of 180**: include/sentio/wf.hpp

**File Information**:
- **Path**: `include/sentio/wf.hpp`

- **Size**: 52 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "sentio/core.hpp"
#include "sentio/runner.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/audit.hpp"
#include <vector>
#include <string>

namespace sentio {

struct WfCfg {
    int train_days = 252;  // Training period in days
    int oos_days = 14;     // Out-of-sample period in days
    int step_days = 14;    // Step size between folds
    bool enable_optimization = false;
    std::string optimizer_type = "random";
    int max_optimization_trials = 30;
    std::string optimization_objective = "sharpe_ratio";
    double optimization_timeout_minutes = 15.0;
    bool optimization_verbose = true;
};

struct Gate {
    bool wf_pass = false;
    bool oos_pass = false;
    std::string recommend = "REJECT";
    double oos_monthly_avg_return = 0.0;
    double oos_sharpe = 0.0;
    double oos_max_drawdown = 0.0;
    int successful_optimizations = 0;
    double avg_optimization_improvement = 0.0;
    std::vector<std::pair<std::string, double>> optimization_results;
};

// Walk-forward testing with vector-based data
Gate run_wf_and_gate(AuditRecorder& audit_template,
                     const SymbolTable& ST,
                     const std::vector<std::vector<Bar>>& series,
                     int base_symbol_id,
                     const RunnerCfg& rcfg,
                     const WfCfg& wcfg);

// Walk-forward testing with optimization
Gate run_wf_and_gate_optimized(AuditRecorder& audit_template,
                               const SymbolTable& ST,
                               const std::vector<std::vector<Bar>>& series,
                               int base_symbol_id,
                               const RunnerCfg& base_rcfg,
                               const WfCfg& wcfg);

} // namespace sentio


```

## 📄 **FILE 138 of 180**: src/accurate_leverage_pricing.cpp

**File Information**:
- **Path**: `src/accurate_leverage_pricing.cpp`

- **Size**: 423 lines
- **Modified**: 2025-09-14 11:29:13

- **Type**: .cpp

```text
#include "sentio/accurate_leverage_pricing.hpp"
#include "sentio/csv_loader.hpp"
#include "sentio/data_resolver.hpp"
#include "sentio/sym/symbol_utils.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>

namespace sentio {

AccurateLeveragePricer::AccurateLeveragePricer(const AccurateLeverageCostModel& cost_model)
    : cost_model_(cost_model), rng_(std::chrono::steady_clock::now().time_since_epoch().count()) {
}

void AccurateLeveragePricer::reset_daily_tracking_if_needed(const std::string& symbol, int64_t current_timestamp) {
    // Convert timestamp to day (assuming UTC seconds)
    int64_t current_day = current_timestamp / (24 * 3600);
    
    auto it = last_daily_reset_.find(symbol);
    if (it == last_daily_reset_.end() || it->second != current_day) {
        // Reset tracking error for new day
        cumulative_tracking_error_[symbol] = 0.0;
        last_daily_reset_[symbol] = current_day;
    }
}

double AccurateLeveragePricer::calculate_accurate_theoretical_price(const std::string& leverage_symbol,
                                                                  double base_price_prev,
                                                                  double base_price_current,
                                                                  double leverage_price_prev,
                                                                  int64_t timestamp) {
    LeverageSpec spec;
    if (!LeverageRegistry::instance().lookup(leverage_symbol, spec)) {
        return leverage_price_prev; // Not a leverage instrument
    }
    
    // Select appropriate cost model based on ETF
    AccurateLeverageCostModel current_cost_model = cost_model_;
    if (leverage_symbol == "PSQ") {
        current_cost_model = AccurateLeverageCostModel::create_psq_model();
    } else if (leverage_symbol == "TQQQ") {
        current_cost_model = AccurateLeverageCostModel::create_tqqq_model();
    }
    // SQQQ uses default cost model (0.95% expense ratio)
    
    // Calculate base return
    if (base_price_prev <= 0.0) return leverage_price_prev;
    double base_return = (base_price_current - base_price_prev) / base_price_prev;
    
    // Apply leverage factor and inverse if needed
    double leveraged_return = spec.factor * base_return;
    if (spec.inverse) {
        leveraged_return = -leveraged_return;
    }
    
    // Apply minute-level costs (properly scaled)
    double minute_cost = current_cost_model.minute_cost_rate();
    leveraged_return -= minute_cost;
    
    // Handle daily tracking error reset
    if (timestamp > 0) {
        reset_daily_tracking_if_needed(leverage_symbol, timestamp);
    }
    
    // Add minimal tracking error (much smaller than before)
    std::normal_distribution<double> tracking_noise(0.0, current_cost_model.tracking_error_daily / std::sqrt(390.0)); // Scale to minute level
    double tracking_adjustment = tracking_noise(rng_);
    
    // Accumulate tracking error for daily reset
    cumulative_tracking_error_[leverage_symbol] += tracking_adjustment;
    leveraged_return += tracking_adjustment;
    
    // Calculate new theoretical price
    double theoretical_price = leverage_price_prev * (1.0 + leveraged_return);
    
    return std::max(theoretical_price, 0.01); // Prevent negative prices
}

Bar AccurateLeveragePricer::generate_accurate_theoretical_bar(const std::string& leverage_symbol,
                                                            const Bar& base_bar_prev,
                                                            const Bar& base_bar_current,
                                                            const Bar& leverage_bar_prev) {
    LeverageSpec spec;
    if (!LeverageRegistry::instance().lookup(leverage_symbol, spec)) {
        return leverage_bar_prev; // Not a leverage instrument
    }
    
    Bar theoretical_bar;
    theoretical_bar.ts_utc = base_bar_current.ts_utc;
    theoretical_bar.ts_utc_epoch = base_bar_current.ts_utc_epoch;
    
    // Use close-to-close for primary calculation (most accurate)
    double base_return = (base_bar_current.close - base_bar_prev.close) / base_bar_prev.close;
    double leverage_factor = spec.inverse ? -spec.factor : spec.factor;
    
    // Apply minute-level costs
    double minute_cost = cost_model_.minute_cost_rate();
    double leveraged_return = leverage_factor * base_return - minute_cost;
    
    // Add minimal tracking error
    std::normal_distribution<double> tracking_noise(0.0, cost_model_.tracking_error_daily / std::sqrt(390.0));
    leveraged_return += tracking_noise(rng_);
    
    // Calculate close price first (most important)
    theoretical_bar.close = leverage_bar_prev.close * (1.0 + leveraged_return);
    
    // For OHLC, use the same leverage factor but apply to intrabar movements
    // This is more accurate than applying costs to each OHLC point
    if (base_bar_prev.close > 0.0) {
        double base_open_move = (base_bar_current.open - base_bar_prev.close) / base_bar_prev.close;
        double base_high_move = (base_bar_current.high - base_bar_prev.close) / base_bar_prev.close;
        double base_low_move = (base_bar_current.low - base_bar_prev.close) / base_bar_prev.close;
        
        theoretical_bar.open = leverage_bar_prev.close * (1.0 + leverage_factor * base_open_move);
        theoretical_bar.high = leverage_bar_prev.close * (1.0 + leverage_factor * base_high_move);
        theoretical_bar.low = leverage_bar_prev.close * (1.0 + leverage_factor * base_low_move);
    } else {
        theoretical_bar.open = theoretical_bar.close;
        theoretical_bar.high = theoretical_bar.close;
        theoretical_bar.low = theoretical_bar.close;
    }
    
    // Ensure OHLC consistency
    theoretical_bar.high = std::max({theoretical_bar.high, theoretical_bar.open, theoretical_bar.close});
    theoretical_bar.low = std::min({theoretical_bar.low, theoretical_bar.open, theoretical_bar.close});
    
    // Prevent negative prices
    theoretical_bar.open = std::max(theoretical_bar.open, 0.01);
    theoretical_bar.high = std::max(theoretical_bar.high, 0.01);
    theoretical_bar.low = std::max(theoretical_bar.low, 0.01);
    theoretical_bar.close = std::max(theoretical_bar.close, 0.01);
    
    // Volume scaling
    theoretical_bar.volume = static_cast<uint64_t>(base_bar_current.volume * (1.0 + spec.factor * 0.1));
    
    return theoretical_bar;
}

std::vector<Bar> AccurateLeveragePricer::generate_accurate_theoretical_series(const std::string& leverage_symbol,
                                                                            const std::vector<Bar>& base_series,
                                                                            const std::vector<Bar>& actual_series_for_init) {
    if (base_series.empty() || actual_series_for_init.empty()) {
        return {};
    }
    
    LeverageSpec spec;
    if (!LeverageRegistry::instance().lookup(leverage_symbol, spec)) {
        return base_series; // Not a leverage instrument, return base series
    }
    
    std::vector<Bar> theoretical_series;
    theoretical_series.reserve(base_series.size());
    
    // **KEY FIX**: Start with actual first price to eliminate initial error
    Bar initial_bar = actual_series_for_init[0];
    initial_bar.ts_utc = base_series[0].ts_utc;
    initial_bar.ts_utc_epoch = base_series[0].ts_utc_epoch;
    theoretical_series.push_back(initial_bar);
    
    // Reset tracking error for this series
    cumulative_tracking_error_[leverage_symbol] = 0.0;
    
    // Generate subsequent bars
    for (size_t i = 1; i < base_series.size() && i < actual_series_for_init.size(); ++i) {
        Bar theoretical_bar = generate_accurate_theoretical_bar(leverage_symbol,
                                                              base_series[i-1],
                                                              base_series[i],
                                                              theoretical_series[i-1]);
        theoretical_series.push_back(theoretical_bar);
    }
    
    return theoretical_series;
}

AccurateLeveragePricingValidator::AccurateLeveragePricingValidator(const AccurateLeverageCostModel& cost_model)
    : pricer_(cost_model) {
}

AccurateLeveragePricingValidator::AccurateValidationResult AccurateLeveragePricingValidator::validate_accurate_pricing(
    const std::string& leverage_symbol,
    const std::vector<Bar>& base_series,
    const std::vector<Bar>& actual_leverage_series) {
    
    AccurateValidationResult result;
    result.symbol = leverage_symbol;
    result.num_observations = 0;
    result.sub_1pct_accuracy = false;
    
    if (base_series.empty() || actual_leverage_series.empty()) {
        return result;
    }
    
    // Generate theoretical series starting from actual first price
    auto theoretical_series = pricer_.generate_accurate_theoretical_series(leverage_symbol, base_series, actual_leverage_series);
    
    if (theoretical_series.empty()) {
        return result;
    }
    
    // Align series by timestamp and calculate errors
    size_t min_length = std::min({base_series.size(), actual_leverage_series.size(), theoretical_series.size()});
    
    std::vector<double> theoretical_prices, actual_prices;
    std::vector<double> theoretical_returns, actual_returns;
    std::vector<double> price_errors_pct, return_errors_pct;
    
    for (size_t i = 1; i < min_length; ++i) {
        // Skip if timestamps don't match
        if (base_series[i].ts_utc_epoch != actual_leverage_series[i].ts_utc_epoch) {
            continue;
        }
        
        double theo_price = theoretical_series[i].close;
        double actual_price = actual_leverage_series[i].close;
        
        if (actual_price <= 0.0) continue; // Skip invalid data
        
        double theo_return = (theoretical_series[i].close - theoretical_series[i-1].close) / theoretical_series[i-1].close;
        double actual_return = (actual_leverage_series[i].close - actual_leverage_series[i-1].close) / actual_leverage_series[i-1].close;
        
        theoretical_prices.push_back(theo_price);
        actual_prices.push_back(actual_price);
        theoretical_returns.push_back(theo_return);
        actual_returns.push_back(actual_return);
        
        // Calculate percentage errors
        double price_error_pct = std::abs((theo_price - actual_price) / actual_price) * 100.0;
        double return_error_pct = std::abs(theo_return - actual_return) * 100.0;
        
        price_errors_pct.push_back(price_error_pct);
        return_errors_pct.push_back(return_error_pct);
    }
    
    result.num_observations = price_errors_pct.size();
    
    if (result.num_observations == 0) {
        return result;
    }
    
    // Calculate correlations
    auto calculate_correlation = [](const std::vector<double>& x, const std::vector<double>& y) -> double {
        if (x.size() != y.size() || x.empty()) return 0.0;
        
        double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
        double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
        
        double numerator = 0.0, sum_sq_x = 0.0, sum_sq_y = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double dx = x[i] - mean_x;
            double dy = y[i] - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }
        
        double denominator = std::sqrt(sum_sq_x * sum_sq_y);
        return (denominator > 0) ? numerator / denominator : 0.0;
    };
    
    result.price_correlation = calculate_correlation(theoretical_prices, actual_prices);
    result.return_correlation = calculate_correlation(theoretical_returns, actual_returns);
    
    // Calculate error statistics
    result.mean_price_error_pct = std::accumulate(price_errors_pct.begin(), price_errors_pct.end(), 0.0) / price_errors_pct.size();
    result.mean_return_error_pct = std::accumulate(return_errors_pct.begin(), return_errors_pct.end(), 0.0) / return_errors_pct.size();
    
    result.max_price_error_pct = *std::max_element(price_errors_pct.begin(), price_errors_pct.end());
    result.max_return_error_pct = *std::max_element(return_errors_pct.begin(), return_errors_pct.end());
    
    // Calculate standard deviations
    double price_error_var = 0.0, return_error_var = 0.0;
    for (size_t i = 0; i < price_errors_pct.size(); ++i) {
        price_error_var += (price_errors_pct[i] - result.mean_price_error_pct) * (price_errors_pct[i] - result.mean_price_error_pct);
        return_error_var += (return_errors_pct[i] - result.mean_return_error_pct) * (return_errors_pct[i] - result.mean_return_error_pct);
    }
    result.price_error_std_pct = std::sqrt(price_error_var / price_errors_pct.size());
    result.return_error_std_pct = std::sqrt(return_error_var / return_errors_pct.size());
    
    // Calculate total returns
    if (!theoretical_prices.empty() && !actual_prices.empty()) {
        result.theoretical_total_return = (theoretical_prices.back() - theoretical_prices.front()) / theoretical_prices.front() * 100.0;
        result.actual_total_return = (actual_prices.back() - actual_prices.front()) / actual_prices.front() * 100.0;
        result.return_difference_pct = std::abs(result.theoretical_total_return - result.actual_total_return);
    }
    
    // Check if we achieved sub-1% accuracy
    result.sub_1pct_accuracy = (result.mean_price_error_pct < 1.0 && result.max_price_error_pct < 5.0);
    
    return result;
}

void AccurateLeveragePricingValidator::print_accurate_validation_report(const AccurateValidationResult& result) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "🎯 HIGH-ACCURACY LEVERAGE PRICING VALIDATION" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Symbol:                   " << result.symbol << std::endl;
    std::cout << "Observations:             " << result.num_observations << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "📈 CORRELATION ANALYSIS" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Price Correlation:        " << std::fixed << std::setprecision(6) << result.price_correlation << std::endl;
    std::cout << "Return Correlation:       " << std::fixed << std::setprecision(6) << result.return_correlation << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "🎯 ACCURACY ANALYSIS" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Mean Price Error:         " << std::fixed << std::setprecision(4) << result.mean_price_error_pct << "%" << std::endl;
    std::cout << "Price Error Std Dev:      " << std::fixed << std::setprecision(4) << result.price_error_std_pct << "%" << std::endl;
    std::cout << "Max Price Error:          " << std::fixed << std::setprecision(4) << result.max_price_error_pct << "%" << std::endl;
    std::cout << "Mean Return Error:        " << std::fixed << std::setprecision(4) << result.mean_return_error_pct << "%" << std::endl;
    std::cout << "Return Error Std Dev:     " << std::fixed << std::setprecision(4) << result.return_error_std_pct << "%" << std::endl;
    std::cout << "Max Return Error:         " << std::fixed << std::setprecision(4) << result.max_return_error_pct << "%" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "💰 TOTAL RETURN COMPARISON" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Theoretical Total Return: " << std::fixed << std::setprecision(3) << result.theoretical_total_return << "%" << std::endl;
    std::cout << "Actual Total Return:      " << std::fixed << std::setprecision(3) << result.actual_total_return << "%" << std::endl;
    std::cout << "Return Difference:        " << std::fixed << std::setprecision(3) << result.return_difference_pct << "%" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "🏆 ACCURACY ASSESSMENT" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    if (result.sub_1pct_accuracy) {
        std::cout << "🎯 TARGET ACHIEVED: Sub-1% accuracy!" << std::endl;
    } else {
        std::cout << "❌ TARGET MISSED: Above 1% error" << std::endl;
    }
    
    if (result.mean_price_error_pct < 0.1) {
        std::cout << "🏆 EXCEPTIONAL: Mean error < 0.1%" << std::endl;
    } else if (result.mean_price_error_pct < 0.5) {
        std::cout << "✅ EXCELLENT: Mean error < 0.5%" << std::endl;
    } else if (result.mean_price_error_pct < 1.0) {
        std::cout << "✅ GOOD: Mean error < 1.0%" << std::endl;
    } else if (result.mean_price_error_pct < 2.0) {
        std::cout << "⚠️  FAIR: Mean error < 2.0%" << std::endl;
    } else {
        std::cout << "❌ POOR: Mean error > 2.0%" << std::endl;
    }
    
    if (result.price_correlation > 0.9999) {
        std::cout << "🏆 EXCEPTIONAL: Price correlation > 99.99%" << std::endl;
    } else if (result.price_correlation > 0.999) {
        std::cout << "✅ EXCELLENT: Price correlation > 99.9%" << std::endl;
    } else if (result.price_correlation > 0.99) {
        std::cout << "✅ GOOD: Price correlation > 99%" << std::endl;
    } else {
        std::cout << "⚠️  NEEDS IMPROVEMENT: Price correlation < 99%" << std::endl;
    }
    
    std::cout << std::string(80, '=') << std::endl;
}

AccurateLeverageCostModel AccurateLeveragePricingValidator::calibrate_for_accuracy(
    const std::string& leverage_symbol,
    const std::vector<Bar>& base_series,
    const std::vector<Bar>& actual_leverage_series,
    double target_error_pct) {
    
    std::cout << "🔧 Calibrating for sub-" << target_error_pct << "% accuracy..." << std::endl;
    
    AccurateLeverageCostModel best_model;
    double best_error = 1e9;
    
    // Fine-grained parameter search for high accuracy
    std::vector<double> expense_ratios = {0.005, 0.007, 0.009, 0.0095, 0.01, 0.011, 0.013, 0.015};
    std::vector<double> borrowing_costs = {0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08};
    std::vector<double> rebalance_costs = {0.00005, 0.0001, 0.0002, 0.0003, 0.0005};
    std::vector<double> tracking_errors = {0.00001, 0.00005, 0.0001, 0.0002, 0.0005};
    
    int total_combinations = expense_ratios.size() * borrowing_costs.size() * rebalance_costs.size() * tracking_errors.size();
    int tested = 0;
    
    for (double expense_ratio : expense_ratios) {
        for (double borrowing_cost : borrowing_costs) {
            for (double rebalance_cost : rebalance_costs) {
                for (double tracking_error : tracking_errors) {
                    AccurateLeverageCostModel test_model;
                    test_model.expense_ratio = expense_ratio;
                    test_model.borrowing_cost_rate = borrowing_cost;
                    test_model.daily_rebalance_cost = rebalance_cost;
                    test_model.tracking_error_daily = tracking_error;
                    
                    AccurateLeveragePricingValidator test_validator(test_model);
                    auto result = test_validator.validate_accurate_pricing(leverage_symbol, base_series, actual_leverage_series);
                    
                    if (result.mean_price_error_pct < best_error) {
                        best_error = result.mean_price_error_pct;
                        best_model = test_model;
                    }
                    
                    tested++;
                    if (tested % 100 == 0) {
                        std::cout << "   Tested " << tested << "/" << total_combinations 
                                  << " combinations, best error: " << std::fixed << std::setprecision(4) 
                                  << best_error << "%" << std::endl;
                    }
                    
                    // Early exit if we achieve target
                    if (best_error < target_error_pct) {
                        std::cout << "🎯 Target achieved early! Error: " << std::fixed << std::setprecision(4) 
                                  << best_error << "%" << std::endl;
                        goto calibration_complete;
                    }
                }
            }
        }
    }
    
    calibration_complete:
    std::cout << "✅ Calibration complete. Best error: " << std::fixed << std::setprecision(4) << best_error << "%" << std::endl;
    std::cout << "📊 Optimal parameters:" << std::endl;
    std::cout << "   Expense Ratio: " << std::fixed << std::setprecision(4) << best_model.expense_ratio * 100 << "%" << std::endl;
    std::cout << "   Borrowing Cost: " << std::fixed << std::setprecision(4) << best_model.borrowing_cost_rate * 100 << "%" << std::endl;
    std::cout << "   Rebalance Cost: " << std::fixed << std::setprecision(5) << best_model.daily_rebalance_cost * 100 << "%" << std::endl;
    std::cout << "   Tracking Error: " << std::fixed << std::setprecision(5) << best_model.tracking_error_daily * 100 << "%" << std::endl;
    
    return best_model;
}

} // namespace sentio

```

## 📄 **FILE 139 of 180**: src/allocation_manager.cpp

**File Information**:
- **Path**: `src/allocation_manager.cpp`

- **Size**: 470 lines
- **Modified**: 2025-09-18 16:51:27

- **Type**: .cpp

```text
#include "sentio/allocation_manager.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace sentio {

AllocationManager::AllocationManager(const AllocationConfig& config) 
    : config_(config) {
    reset_state();
}

void AllocationManager::reset_state() {
    current_state_ = PositionState{};
}

// **MATHEMATICAL DECISION ENGINE**: Core allocation logic
InternalAllocationDecision AllocationManager::make_allocation_decision(
    double current_probability,
    double current_unrealized_pnl_pct,
    int bars_since_last_decision) {
    
    // Update current state
    current_state_.unrealized_pnl_pct = current_unrealized_pnl_pct;
    current_state_.bars_held += bars_since_last_decision;
    
    // **DIRECTIONAL TRANSITION STATE TRACKING**: Increment holding period
    if (is_directional_position_) {
        bars_in_current_position_ += bars_since_last_decision;
    }
    
    // Track signal extremes for momentum analysis
    if (current_probability > current_state_.max_favorable_prob) {
        current_state_.max_favorable_prob = current_probability;
    }
    if (current_probability < (1.0 - current_state_.max_adverse_prob)) {
        current_state_.max_adverse_prob = 1.0 - current_probability;
    }
    
    InternalAllocationDecision decision;
    decision.confidence = 0.0;
    
    // **CASE 1: CURRENTLY IN CASH** - Consider entry
    if (current_state_.type == PositionType::CASH) {
        PositionType target_type = select_entry_position_type(current_probability);
        
        if (target_type != PositionType::CASH) {
            decision.action = AllocationAction::ENTER_NEW;
            decision.target_type = target_type;
            decision.target_weight = 1.0;
            decision.confidence = std::abs(current_probability - 0.5) * 2.0; // 0-1 scale
            
            std::ostringstream reason;
            reason << "Enter " << position_type_to_string(target_type) 
                   << " (prob=" << current_probability << ")";
            decision.reason = reason.str();
        } else {
            // Stay in cash
            decision.action = AllocationAction::HOLD;
            decision.target_type = PositionType::CASH;
            decision.target_weight = 0.0;
            decision.reason = "Signal too weak for entry";
            decision.confidence = 0.1;
        }
        
        return decision;
    }
    
    // **CASE 2: CURRENTLY HOLDING POSITION** - Consider hold/exit/transition
    
    bool is_opposing = is_signal_opposing(current_probability, current_state_.type);
    bool is_favorable = is_signal_favorable(current_probability, current_state_.type);
    
    // **RISK MANAGEMENT**: Force exit on extreme adverse signals
    if (is_opposing && std::abs(current_probability - 0.5) > config_.max_adverse_tolerance) {
        decision.action = AllocationAction::FULL_CLOSE;
        decision.target_type = PositionType::CASH;
        decision.target_weight = 0.0;
        decision.reason = "Risk management: Extreme adverse signal";
        decision.confidence = 0.9;
        return decision;
    }
    
    // **HOLDING PERIOD MANAGEMENT**: Force review after max holding period
    if (current_state_.bars_held >= config_.max_holding_period) {
        if (is_opposing) {
            decision.action = AllocationAction::FULL_CLOSE;
            decision.target_type = PositionType::CASH;
            decision.target_weight = 0.0;
            decision.reason = "Max holding period reached with opposing signal";
            decision.confidence = 0.7;
        } else {
            // Reset holding period but maintain position
            current_state_.bars_held = 0;
            decision.action = AllocationAction::HOLD;
            decision.target_type = current_state_.type;
            decision.target_weight = current_state_.weight;
            decision.reason = "Reset holding period, maintain position";
            decision.confidence = 0.3;
        }
        return decision;
    }
    
    if (is_opposing) {
        // **OPPOSING SIGNAL**: Consider directional transition logic
        
        double opposing_strength = std::abs(current_probability - 0.5) * 2.0;
        PositionType target_type = select_entry_position_type(current_probability);
        
        // **DIRECTIONAL TRANSITION LOGIC**
        if (target_type != PositionType::CASH && is_opposite_direction(current_state_.type, target_type)) {
            // Opposing signal wants to change direction
            if (can_change_direction(opposing_strength, current_state_.type, target_type)) {
                // **DIRECTION CHANGE ALLOWED**: Strong signal or minimum holding period met
                decision.action = AllocationAction::ENTER_NEW;
                decision.target_type = target_type;
                decision.target_weight = 1.0;
                decision.confidence = opposing_strength;
                
                std::ostringstream reason;
                reason << "Direction change: " << position_type_to_string(current_state_.type) 
                       << " -> " << position_type_to_string(target_type)
                       << " (strength=" << opposing_strength << ", bars_held=" << bars_in_current_position_ << ")";
                decision.reason = reason.str();
                
            } else if (opposing_strength >= config_.weak_opposite_cash_threshold) {
                // **WEAK OPPOSITE SIGNAL**: Go to cash during holding period
                decision.action = AllocationAction::FULL_CLOSE;
                decision.target_type = PositionType::CASH;
                decision.target_weight = 0.0;
                decision.confidence = opposing_strength * 0.6; // Lower confidence for cash moves
                
                std::ostringstream reason;
                reason << "Cash transition: weak opposite signal during min holding period "
                       << "(strength=" << opposing_strength << ", bars_held=" << bars_in_current_position_ << ")";
                decision.reason = reason.str();
                
            } else {
                // **HOLD**: Signal not strong enough to override holding period
                decision.action = AllocationAction::HOLD;
                decision.target_type = current_state_.type;
                decision.target_weight = current_state_.weight;
                decision.confidence = 0.2;
                decision.reason = "Hold: opposing signal too weak during min holding period";
            }
        } else {
            // Traditional exit logic for non-directional opposing signals
            double partial_threshold = calculate_exit_threshold(config_.partial_exit_threshold, current_state_);
            double full_threshold = calculate_exit_threshold(config_.full_exit_threshold, current_state_);
            
            if (opposing_strength >= full_threshold) {
                // **FULL EXIT**: Strong opposing signal
                decision.action = AllocationAction::FULL_CLOSE;
                decision.target_type = PositionType::CASH;
                decision.target_weight = 0.0;
                decision.confidence = opposing_strength;
                
                std::ostringstream reason;
                reason << "Full exit: opposing signal " << current_probability 
                       << " > threshold " << full_threshold;
                decision.reason = reason.str();
                
            } else if (opposing_strength >= partial_threshold && 
                       (current_state_.type == PositionType::LONG_3X || current_state_.type == PositionType::INVERSE_3X)) {
            // **PARTIAL EXIT**: Reduce leverage (3x -> 1x)
            decision.action = AllocationAction::PARTIAL_CLOSE;
            decision.target_type = (current_state_.type == PositionType::LONG_3X) ? 
                                  PositionType::LONG_1X : PositionType::INVERSE_1X;
            decision.target_weight = 1.0;
            decision.confidence = opposing_strength * 0.7; // Lower confidence for partial moves
            
            std::ostringstream reason;
            reason << "Partial exit: " << position_type_to_string(current_state_.type)
                   << " -> " << position_type_to_string(decision.target_type);
            decision.reason = reason.str();
            
        } else {
            // **HOLD**: Opposing signal not strong enough
            decision.action = AllocationAction::HOLD;
            decision.target_type = current_state_.type;
            decision.target_weight = current_state_.weight;
            decision.confidence = 0.2;
            
            std::ostringstream reason;
            reason << "Hold despite opposing signal " << current_probability 
                   << " (threshold=" << partial_threshold << ")";
            decision.reason = reason.str();
            }
        }
        
    } else if (is_favorable) {
        // **FAVORABLE SIGNAL**: Consider upgrading leverage
        
        double favorable_strength = std::abs(current_probability - 0.5) * 2.0;
        
        if (favorable_strength >= config_.entry_threshold_3x && 
            current_state_.type == PositionType::LONG_1X) {
            // **UPGRADE**: 1x -> 3x long
            decision.action = AllocationAction::ENTER_NEW;
            decision.target_type = PositionType::LONG_3X;
            decision.target_weight = 1.0;
            decision.confidence = favorable_strength;
            decision.reason = "Upgrade QQQ -> TQQQ on strong favorable signal";
            
        } else if (favorable_strength >= config_.entry_threshold_3x && 
                   current_state_.type == PositionType::INVERSE_1X) {
            // **UPGRADE**: 1x -> 3x inverse
            decision.action = AllocationAction::ENTER_NEW;
            decision.target_type = PositionType::INVERSE_3X;
            decision.target_weight = 1.0;
            decision.confidence = favorable_strength;
            decision.reason = "Upgrade PSQ -> SQQQ on strong favorable signal";
            
        } else {
            // **HOLD**: Current position appropriate for signal strength
            decision.action = AllocationAction::HOLD;
            decision.target_type = current_state_.type;
            decision.target_weight = current_state_.weight;
            decision.confidence = 0.4;
            decision.reason = "Hold: favorable signal confirms position";
        }
        
    } else {
        // **NEUTRAL SIGNAL**: Hold current position
        decision.action = AllocationAction::HOLD;
        decision.target_type = current_state_.type;
        decision.target_weight = current_state_.weight;
        decision.confidence = 0.3;
        decision.reason = "Hold: neutral signal";
    }
    
    return decision;
}

// **MATHEMATICAL THRESHOLD ADJUSTMENT**: Dynamic thresholds based on position strength
double AllocationManager::calculate_exit_threshold(double base_threshold, const PositionState& state) const {
    double adjusted_threshold = base_threshold;
    
    // **PROFIT PROTECTION**: Lower threshold if position is profitable
    if (state.unrealized_pnl_pct > 0) {
        adjusted_threshold -= config_.profit_protection_factor * state.unrealized_pnl_pct;
    } else {
        // **LOSS CUTTING**: Higher threshold if position is losing
        adjusted_threshold += config_.loss_cutting_factor * std::abs(state.unrealized_pnl_pct);
    }
    
    // **MOMENTUM DECAY**: Reduce threshold over time (easier to exit old positions)
    double time_decay = config_.momentum_decay_factor * (state.bars_held / 100.0);
    adjusted_threshold -= time_decay;
    
    // **BOUNDS**: Keep threshold reasonable
    adjusted_threshold = std::max(0.1, std::min(0.8, adjusted_threshold));
    
    return adjusted_threshold;
}

double AllocationManager::calculate_position_strength(const PositionState& state) const {
    // Combine P&L, holding period, and signal history
    double pnl_strength = std::tanh(state.unrealized_pnl_pct * 10.0); // -1 to 1
    double time_strength = std::exp(-state.bars_held / 100.0);        // Decay over time
    double signal_strength = state.max_favorable_prob - state.max_adverse_prob;
    
    return (pnl_strength + time_strength + signal_strength) / 3.0;
}

bool AllocationManager::is_signal_opposing(double probability, PositionType position_type) const {
    switch (position_type) {
        case PositionType::LONG_1X:
        case PositionType::LONG_3X:
            return probability < 0.5; // Bearish signal opposes long positions
        case PositionType::INVERSE_1X:
        case PositionType::INVERSE_3X:
            return probability > 0.5; // Bullish signal opposes inverse positions
        case PositionType::CASH:
            return false; // No position to oppose
    }
    return false;
}

bool AllocationManager::is_signal_favorable(double probability, PositionType position_type) const {
    switch (position_type) {
        case PositionType::LONG_1X:
        case PositionType::LONG_3X:
            return probability > 0.5; // Bullish signal favors long positions
        case PositionType::INVERSE_1X:
        case PositionType::INVERSE_3X:
            return probability < 0.5; // Bearish signal favors inverse positions
        case PositionType::CASH:
            return false; // No position to favor
    }
    return false;
}

PositionType AllocationManager::select_entry_position_type(double probability) const {
    double signal_strength = std::abs(probability - 0.5) * 2.0; // 0-1 scale
    
    if (probability > 0.5) {
        // **BULLISH SIGNAL**
        if (signal_strength >= config_.entry_threshold_3x) {
            return PositionType::LONG_3X; // TQQQ
        } else if (signal_strength >= config_.entry_threshold_1x) {
            return PositionType::LONG_1X; // QQQ
        }
    } else {
        // **BEARISH SIGNAL**
        if (signal_strength >= config_.entry_threshold_3x) {
            return PositionType::INVERSE_3X; // SQQQ
        } else if (signal_strength >= config_.entry_threshold_1x) {
            return PositionType::INVERSE_1X; // PSQ
        }
    }
    
    return PositionType::CASH; // Signal too weak
}

// **DIRECTIONAL TRANSITION HELPERS**
bool AllocationManager::is_long_position(PositionType type) const {
    return type == PositionType::LONG_1X || type == PositionType::LONG_3X;
}

bool AllocationManager::is_short_position(PositionType type) const {
    return type == PositionType::INVERSE_1X || type == PositionType::INVERSE_3X;
}

bool AllocationManager::is_opposite_direction(PositionType current, PositionType target) const {
    return (is_long_position(current) && is_short_position(target)) ||
           (is_short_position(current) && is_long_position(target));
}

bool AllocationManager::can_change_direction(double signal_strength, PositionType current, PositionType target) const {
    // If not changing direction, always allowed
    if (!is_opposite_direction(current, target)) {
        return true;
    }
    
    // If we haven't held the position long enough, check conditions
    if (bars_in_current_position_ < config_.min_holding_period) {
        // Strong signals can bypass minimum holding period
        if (signal_strength >= config_.strong_signal_bypass_threshold) {
            return true;
        }
        // Otherwise, must wait for minimum holding period
        return false;
    }
    
    // After minimum holding period, direction changes are allowed
    return true;
}

std::vector<AllocationManager::RunnerAllocationDecision> AllocationManager::get_runner_allocations(
    double current_probability,
    const Portfolio& current_portfolio,
    const std::vector<double>& last_prices,
    const SymbolTable& ST) {
    
    // Calculate current unrealized P&L
    double current_equity = equity_mark_to_market(current_portfolio, last_prices);
    double unrealized_pnl_pct = 0.0;
    if (current_state_.type != PositionType::CASH && current_equity > 0) {
        // Simplified P&L calculation - could be enhanced
        unrealized_pnl_pct = (current_equity - starting_capital_) / starting_capital_;
    }
    
    // Get allocation decision
    InternalAllocationDecision decision = make_allocation_decision(current_probability, unrealized_pnl_pct, 1);
    
    std::vector<RunnerAllocationDecision> runner_decisions;
    
    // Convert internal decision to runner format
    if (decision.action == AllocationAction::HOLD && decision.target_type == PositionType::CASH) {
        // No position needed - flatten all instruments
        std::vector<std::string> all_instruments = {"QQQ", "TQQQ", "SQQQ", "PSQ"};
        for (const auto& instrument : all_instruments) {
            runner_decisions.push_back({instrument, 0.0, decision.confidence, "AllocationManager: Flatten to cash"});
        }
    } else if (decision.target_type != PositionType::CASH) {
        // Target a specific instrument
        std::string target_instrument = position_type_to_symbol(decision.target_type);
        // **FIX**: All positions use positive weights (no negative shorts)
        // PSQ and SQQQ are inverse ETFs, but we buy them with positive quantities
        double target_weight = std::abs(decision.target_weight);
        
        // **FIX**: Only return the target position decision
        // Let PositionCoordinator handle flattening other positions naturally
        runner_decisions.push_back({target_instrument, target_weight, decision.confidence, decision.reason});
    }
    
    // Update internal state
    update_position_state(decision, current_probability);
    
    return runner_decisions;
}

void AllocationManager::update_position_state(const InternalAllocationDecision& decision, double probability) {
    if (decision.action == AllocationAction::ENTER_NEW ||
        decision.action == AllocationAction::PARTIAL_CLOSE) {
        
        // Check if this is a new directional position
        PositionType old_type = current_state_.type;
        bool was_directional = is_directional_position_;
        bool is_new_directional = (decision.target_type != PositionType::CASH);
        
        // Update position
        current_state_.type = decision.target_type;
        current_state_.weight = decision.target_weight;
        current_state_.entry_probability = probability;
        current_state_.bars_held = 0;
        current_state_.max_favorable_prob = probability;
        current_state_.max_adverse_prob = 1.0 - probability;
        
        // **DIRECTIONAL TRANSITION STATE TRACKING**
        if (is_new_directional && (!was_directional || is_opposite_direction(old_type, decision.target_type))) {
            // Starting new directional position or changing direction
            bars_in_current_position_ = 0;
            is_directional_position_ = true;
        } else if (!is_new_directional) {
            // Going to cash
            bars_in_current_position_ = 0;
            is_directional_position_ = false;
        }
        // If continuing same direction (upgrade/downgrade), keep existing bars count
        
    } else if (decision.action == AllocationAction::FULL_CLOSE) {
        // Reset to cash
        reset_state();
        // **DIRECTIONAL TRANSITION STATE TRACKING**
        bars_in_current_position_ = 0;
        is_directional_position_ = false;
    }
    // HOLD action doesn't change position state (already updated in make_allocation_decision)
}

// **UTILITY FUNCTIONS**
std::string position_type_to_string(PositionType type) {
    switch (type) {
        case PositionType::CASH: return "CASH";
        case PositionType::LONG_1X: return "QQQ";
        case PositionType::LONG_3X: return "TQQQ";
        case PositionType::INVERSE_1X: return "PSQ";
        case PositionType::INVERSE_3X: return "SQQQ";
    }
    return "UNKNOWN";
}

std::string position_type_to_symbol(PositionType type) {
    return position_type_to_string(type); // Same for now
}

double get_position_leverage(PositionType type) {
    switch (type) {
        case PositionType::CASH: return 0.0;
        case PositionType::LONG_1X: return 1.0;
        case PositionType::LONG_3X: return 3.0;
        case PositionType::INVERSE_1X: return -1.0;
        case PositionType::INVERSE_3X: return -3.0;
    }
    return 0.0;
}

bool are_positions_conflicting(PositionType type1, PositionType type2) {
    if (type1 == PositionType::CASH || type2 == PositionType::CASH) return false;
    
    double lev1 = get_position_leverage(type1);
    double lev2 = get_position_leverage(type2);
    
    // Conflicting if different signs (long vs inverse)
    return (lev1 > 0 && lev2 < 0) || (lev1 < 0 && lev2 > 0);
}

} // namespace sentio

```

## 📄 **FILE 140 of 180**: src/audit.cpp

**File Information**:
- **Path**: `src/audit.cpp`

- **Size**: 319 lines
- **Modified**: 2025-09-11 16:46:18

- **Type**: .cpp

```text
#include "sentio/audit.hpp"
#include "sentio/signal_diag.hpp"
#include <cstring>
#include <sstream>
#include <iomanip>

// ---- minimal SHA1 (tiny, not constant-time; for tamper detection only) ----
namespace {
struct Sha1 {
  uint32_t h0=0x67452301, h1=0xEFCDAB89, h2=0x98BADCFE, h3=0x10325476, h4=0xC3D2E1F0;
  std::string hex(const std::string& s){
    uint64_t ml = s.size()*8ULL;
    std::string msg = s;
    msg.push_back('\x80');
    while ((msg.size()%64)!=56) msg.push_back('\0');
    for (int i=7;i>=0;--i) msg.push_back(char((ml>>(i*8))&0xFF));
    for (size_t i=0;i<msg.size(); i+=64) {
      uint32_t w[80];
      for (int t=0;t<16;++t) {
        w[t] = (uint32_t(uint8_t(msg[i+4*t]))<<24)
             | (uint32_t(uint8_t(msg[i+4*t+1]))<<16)
             | (uint32_t(uint8_t(msg[i+4*t+2]))<<8)
             | (uint32_t(uint8_t(msg[i+4*t+3])));
      }
      for (int t=16;t<80;++t){ uint32_t v = w[t-3]^w[t-8]^w[t-14]^w[t-16]; w[t]=(v<<1)|(v>>31); }
      uint32_t a=h0,b=h1,c=h2,d=h3,e=h4;
      for (int t=0;t<80;++t){
        uint32_t f,k;
        if (t<20){ f=(b&c)|((~b)&d); k=0x5A827999; }
        else if (t<40){ f=b^c^d; k=0x6ED9EBA1; }
        else if (t<60){ f=(b&c)|(b&d)|(c&d); k=0x8F1BBCDC; }
        else { f=b^c^d; k=0xCA62C1D6; }
        uint32_t temp = ((a<<5)|(a>>27)) + f + e + k + w[t];
        e=d; d=c; c=((b<<30)|(b>>2)); b=a; a=temp;
      }
      h0+=a; h1+=b; h2+=c; h3+=d; h4+=e;
    }
    std::ostringstream os; os<<std::hex<<std::setfill('0')<<std::nouppercase;
    os<<std::setw(8)<<h0<<std::setw(8)<<h1<<std::setw(8)<<h2<<std::setw(8)<<h3<<std::setw(8)<<h4;
    return os.str();
  }
};
}

namespace sentio {

static inline std::string num_s(double v){
  std::ostringstream os; os.setf(std::ios::fixed); os<<std::setprecision(8)<<v; return os.str();
}
static inline std::string num_i(std::int64_t v){
  std::ostringstream os; os<<v; return os.str();
}
std::string AuditRecorder::json_escape_(const std::string& s){
  std::string o; o.reserve(s.size()+8);
  for (char c: s){
    switch(c){
      case '"': o+="\\\""; break;
      case '\\':o+="\\\\"; break;
      case '\b':o+="\\b"; break;
      case '\f':o+="\\f"; break;
      case '\n':o+="\\n"; break;
      case '\r':o+="\\r"; break;
      case '\t':o+="\\t"; break;
      default: o.push_back(c);
    }
  }
  return o;
}

std::string AuditRecorder::sha1_hex_(const std::string& s){
  Sha1 sh; return sh.hex(s);
}

AuditRecorder::AuditRecorder(const AuditConfig& cfg)
: run_id_(cfg.run_id), file_path_(cfg.file_path), flush_each_(cfg.flush_each)
{
  fp_ = std::fopen(cfg.file_path.c_str(), "ab");
  if (!fp_) throw std::runtime_error("Audit open failed: "+cfg.file_path);
}
AuditRecorder::~AuditRecorder(){ if (fp_) std::fclose(fp_); }

void AuditRecorder::write_line_(const std::string& body){
  std::string core = "{\"run\":\""+json_escape_(run_id_)+"\",\"seq\":"+num_i((std::int64_t)seq_)+","+body+"}";
  std::string line = core; // sha1 over core for stability
  std::string h = sha1_hex_(core);
  line.pop_back(); // remove trailing '}'
  line += ",\"sha1\":\""+h+"\"}\n";
  if (std::fwrite(line.data(),1,line.size(),fp_)!=line.size()) throw std::runtime_error("Audit write failed");
  if (flush_each_) std::fflush(fp_);
  ++seq_;
}

void AuditRecorder::event_run_start(std::int64_t ts, const std::string& meta){
  write_line_("\"type\":\"run_start\",\"ts\":"+num_i(ts)+",\"meta\":"+meta+"}");
}
void AuditRecorder::event_run_end(std::int64_t ts, const std::string& meta){
  write_line_("\"type\":\"run_end\",\"ts\":"+num_i(ts)+",\"meta\":"+meta+"}");
}
void AuditRecorder::event_bar(std::int64_t ts, const std::string& inst, double open, double high, double low, double close, double volume){
  write_line_("\"type\":\"bar\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"o\":"+num_s(open)+",\"h\":"+num_s(high)+",\"l\":"+num_s(low)+",\"c\":"+num_s(close)+",\"v\":"+num_s(volume)+"}");
}
void AuditRecorder::event_signal(std::int64_t ts, const std::string& base, SigType t, double conf){
  write_line_("\"type\":\"signal\",\"ts\":"+num_i(ts)+",\"base\":\""+json_escape_(base)+"\",\"sig\":" + num_i((int)t) + ",\"p\":"+num_s(conf)+"}");
}
void AuditRecorder::event_route (std::int64_t ts, const std::string& base, const std::string& inst, double tw){
  write_line_("\"type\":\"route\",\"ts\":"+num_i(ts)+",\"base\":\""+json_escape_(base)+"\",\"inst\":\""+json_escape_(inst)+"\",\"tw\":"+num_s(tw)+"}");
}
void AuditRecorder::event_order (std::int64_t ts, const std::string& inst, Side side, double qty, double limit_px){
  write_line_("\"type\":\"order\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"side\":"+num_i((int)side)+",\"qty\":"+num_s(qty)+",\"limit\":"+num_s(limit_px)+"}");
}
void AuditRecorder::event_fill  (std::int64_t ts, const std::string& inst, double price, double qty, double fees, Side side){
  write_line_("\"type\":\"fill\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"px\":"+num_s(price)+",\"qty\":"+num_s(qty)+",\"fees\":"+num_s(fees)+",\"side\":"+num_i((int)side)+"}");
}
void AuditRecorder::event_snapshot(std::int64_t ts, const AccountState& a){
  write_line_("\"type\":\"snapshot\",\"ts\":"+num_i(ts)+",\"cash\":"+num_s(a.cash)+",\"real\":"+num_s(a.realized)+",\"equity\":"+num_s(a.equity)+"}");
}
void AuditRecorder::event_metric(std::int64_t ts, const std::string& key, double val){
  write_line_("\"type\":\"metric\",\"ts\":"+num_i(ts)+",\"key\":\""+json_escape_(key)+"\",\"val\":"+num_s(val)+"}");
}

// ----------------- Extended events --------------------
void AuditRecorder::event_signal_ex(std::int64_t ts, const std::string& base, SigType t, double conf,
                                    const std::string& chain_id){
  write_line_("\"type\":\"signal\",\"ts\":"+num_i(ts)+",\"base\":\""+json_escape_(base)+"\",\"sig\":" + num_i((int)t) + ",\"p\":"+num_s(conf)+",\"chain\":\""+json_escape_(chain_id)+"\"}");
}
void AuditRecorder::event_route_ex (std::int64_t ts, const std::string& base, const std::string& inst, double tw,
                                    const std::string& chain_id){
  write_line_("\"type\":\"route\",\"ts\":"+num_i(ts)+",\"base\":\""+json_escape_(base)+"\",\"inst\":\""+json_escape_(inst)+"\",\"tw\":"+num_s(tw)+",\"chain\":\""+json_escape_(chain_id)+"\"}");
}
void AuditRecorder::event_order_ex (std::int64_t ts, const std::string& inst, Side side, double qty, double limit_px,
                                    const std::string& chain_id){
  write_line_("\"type\":\"order\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"side\":"+num_i((int)side)+",\"qty\":"+num_s(qty)+",\"limit\":"+num_s(limit_px)+",\"chain\":\""+json_escape_(chain_id)+"\"}");
}
void AuditRecorder::event_fill_ex  (std::int64_t ts, const std::string& inst, double price, double qty, double fees, Side side,
                                    double realized_pnl_delta, double equity_after, double position_after,
                                    const std::string& chain_id){
  write_line_(
    "\"type\":\"fill\",\"ts\":"+num_i(ts)+
    ",\"inst\":\""+json_escape_(inst)+
    "\",\"px\":"+num_s(price)+
    ",\"qty\":"+num_s(qty)+
    ",\"fees\":"+num_s(fees)+
    ",\"side\":"+num_i((int)side)+
    ",\"pnl_d\":"+num_s(realized_pnl_delta)+
    ",\"eq_after\":"+num_s(equity_after)+
    ",\"pos_after\":"+num_s(position_after)+
    ",\"chain\":\""+json_escape_(chain_id)+"\"}"
  );
}

// ----------------- Signal Diagnostics Events --------------------
void AuditRecorder::event_signal_diag(std::int64_t ts, const std::string& strategy_name, const SignalDiag& diag) {
  write_line_(
    "\"type\":\"signal_diag\",\"ts\":"+num_i(ts)+
    ",\"strategy\":\""+json_escape_(strategy_name)+"\""
    ",\"emitted\":"+num_i(diag.emitted)+
    ",\"dropped\":"+num_i(diag.dropped)+
    ",\"r_min_bars\":"+num_i(diag.r_min_bars)+
    ",\"r_session\":"+num_i(diag.r_session)+
    ",\"r_nan\":"+num_i(diag.r_nan)+
    ",\"r_zero_vol\":"+num_i(diag.r_zero_vol)+
    ",\"r_threshold\":"+num_i(diag.r_threshold)+
    ",\"r_cooldown\":"+num_i(diag.r_cooldown)+
    ",\"r_dup\":"+num_i(diag.r_dup)
  );
}

void AuditRecorder::event_signal_drop(std::int64_t ts, const std::string& strategy_name, const std::string& symbol, 
                                      DropReason reason, const std::string& chain_id, const std::string& note) {
  write_line_(
    "\"type\":\"signal_drop\",\"ts\":"+num_i(ts)+
    ",\"strategy\":\""+json_escape_(strategy_name)+"\""
    ",\"symbol\":\""+json_escape_(symbol)+"\""
    ",\"reason\":"+num_i((int)reason)+
    ",\"chain\":\""+json_escape_(chain_id)+"\""
    ",\"note\":\""+json_escape_(note)+"\""
  );
}

// ----------------- Replayer --------------------

static inline bool parse_kv(const std::string& s, const char* key, std::string& out) {
  auto kq = std::string("\"")+key+"\":";
  auto p = s.find(kq); if (p==std::string::npos) return false;
  p += kq.size();
  if (p>=s.size()) return false;
  if (s[p]=='"'){ // string
    auto e = s.find('"', p+1);
    if (e==std::string::npos) return false;
    out = s.substr(p+1, e-(p+1));
    return true;
  } else { // number or enum
    auto e = s.find_first_of(",}", p);
    out = s.substr(p, e-p);
    return true;
  }
}

std::optional<ReplayResult> AuditReplayer::replay_file(const std::string& path,
                                                       const std::string& run_expect)
{
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) return std::nullopt;

  PriceBook pb;
  ReplayResult rr;
  rr.acct.cash = 0.0;
  rr.acct.realized = 0.0;

  char buf[16*1024];
  while (std::fgets(buf, sizeof(buf), fp)) {
    std::string line(buf);
    // very light JSONL parsing (we control writer)

    std::string run; if (!parse_kv(line, "run", run)) continue;
    if (!run_expect.empty() && run!=run_expect) continue;

    std::string type; if (!parse_kv(line, "type", type)) continue;
    // Note: timestamp parsing removed as it's not currently used in replay logic

    if (type=="bar") {
      std::string inst; parse_kv(line, "inst", inst);
      std::string o,h,l,c; parse_kv(line,"o",o); parse_kv(line,"h",h);
      parse_kv(line,"l",l); parse_kv(line,"c",c);
      AuditBar b{std::stod(o),std::stod(h),std::stod(l),std::stod(c)};
      apply_bar_(pb, inst, b);
      ++rr.bars;
    } else if (type=="signal") {
      ++rr.signals;
    } else if (type=="route") {
      ++rr.routes;
    } else if (type=="order") {
      ++rr.orders;
    } else if (type=="fill") {
      std::string inst,px,qty,fees,side_s;
      parse_kv(line,"inst",inst); parse_kv(line,"px",px);
      parse_kv(line,"qty",qty); parse_kv(line,"fees",fees);
      parse_kv(line,"side",side_s);
      Side side = side_s.empty() ? Side::Buy : static_cast<Side>(std::stoi(side_s));
      apply_fill_(rr, inst, std::stod(px), std::stod(qty), std::stod(fees), side);
      ++rr.fills;
      mark_to_market_(pb, rr);
    } else if (type=="snapshot") {
      // snapshots are optional for verification; we recompute anyway
      // you can cross-check here if you also store snapshots
    } else if (type=="run_end") {
      // could verify sha1 continuity/counts here
    }
  }
  std::fclose(fp);

  mark_to_market_(pb, rr);
  return rr;
}

bool AuditReplayer::apply_bar_(PriceBook& pb, const std::string& instrument, const AuditBar& b) {
  pb.last_px[instrument] = b.close;
  return true;
}

void AuditReplayer::apply_fill_(ReplayResult& rr, const std::string& inst, double px, double qty, double fees, Side side) {
  auto& pos = rr.positions[inst];
  
  // Convert order side to position impact
  // Buy orders: positive position qty, Sell orders: negative position qty
  double position_qty = (side == Side::Buy) ? qty : -qty;
  
  // cash impact: buy qty>0 => cash decreases, sell qty>0 => cash increases
  double cash_delta = (side == Side::Buy) ? -(px*qty + fees) : (px*qty - fees);
  rr.acct.cash += cash_delta;

  // position update (VWAP)
  double new_qty = pos.qty + position_qty;
  
  if (new_qty == 0.0) {
    // flat: realize P&L for the round trip
    if (pos.qty != 0.0) {
      rr.acct.realized += (px - pos.avg_px) * pos.qty;
    }
    pos.qty = 0.0; 
    pos.avg_px = 0.0;
  } else if (pos.qty == 0.0) {
    // opening new position
    pos.qty = new_qty;
    pos.avg_px = px;
  } else if ((pos.qty > 0) == (position_qty > 0)) {
    // adding to same side -> new average
    pos.avg_px = (pos.avg_px * pos.qty + px * position_qty) / new_qty;
    pos.qty = new_qty;
  } else {
    // reducing or flipping side -> realize partial P&L
    double closed_qty = std::min(std::abs(position_qty), std::abs(pos.qty));
    if (pos.qty > 0) {
      rr.acct.realized += (px - pos.avg_px) * closed_qty;
    } else {
      rr.acct.realized += (pos.avg_px - px) * closed_qty;
    }
    pos.qty = new_qty;
    if (pos.qty == 0.0) {
      pos.avg_px = 0.0;
    } else {
      pos.avg_px = px; // new average for remaining position
    }
  }
}

void AuditReplayer::mark_to_market_(const PriceBook& pb, ReplayResult& rr) {
  double mtm=0.0;
  for (auto& kv : rr.positions) {
    const auto& inst = kv.first;
    const auto& p = kv.second;
    auto it = pb.last_px.find(inst);
    if (it==pb.last_px.end()) continue;
    mtm += p.qty * it->second;
  }
  rr.acct.equity = rr.acct.cash + rr.acct.realized + mtm;
}

} // namespace sentio
```

## 📄 **FILE 141 of 180**: src/base_strategy.cpp

**File Information**:
- **Path**: `src/base_strategy.cpp`

- **Size**: 54 lines
- **Modified**: 2025-09-11 11:12:37

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

## 📄 **FILE 142 of 180**: src/canonical_evaluation.cpp

**File Information**:
- **Path**: `src/canonical_evaluation.cpp`

- **Size**: 227 lines
- **Modified**: 2025-09-17 16:32:50

- **Type**: .cpp

```text
#include "sentio/canonical_evaluation.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>

namespace sentio {

BlockResult CanonicalEvaluator::calculate_block_metrics(
    const std::vector<std::pair<std::string, double>>& equity_curve,
    int block_index,
    double starting_equity,
    int fills_count,
    std::int64_t start_ts_ms,
    std::int64_t end_ts_ms) {
    
    BlockResult result;
    result.block_index = block_index;
    result.fills = fills_count;
    result.starting_equity = starting_equity;
    result.start_ts_ms = start_ts_ms;
    result.end_ts_ms = end_ts_ms;
    
    if (equity_curve.empty()) {
        result.ending_equity = starting_equity;
        result.return_per_block = 0.0;
        result.sharpe_ratio = 0.0;
        result.max_drawdown_pct = 0.0;
        return result;
    }
    
    result.ending_equity = equity_curve.back().second;
    
    // Calculate Return Per Block (RPB) - NET OF FEES
    // Note: equity_curve already includes Alpaca trading fees via portfolio.cash -= fees
    if (starting_equity > 0.0) {
        result.return_per_block = (result.ending_equity / starting_equity) - 1.0;
    } else {
        result.return_per_block = 0.0;
    }
    
    // Calculate bar-level returns for Sharpe ratio
    std::vector<double> returns;
    returns.reserve(equity_curve.size());
    
    double prev_equity = starting_equity;
    for (const auto& [timestamp, equity] : equity_curve) {
        if (prev_equity > 0.0) {
            double return_rate = (equity / prev_equity) - 1.0;
            returns.push_back(return_rate);
        }
        prev_equity = equity;
    }
    
    // Calculate Sharpe ratio for this block
    if (returns.size() > 1) {
        double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        
        double variance = 0.0;
        for (double ret : returns) {
            double diff = ret - mean_return;
            variance += diff * diff;
        }
        variance /= std::max<size_t>(1, returns.size() - 1); // Sample variance
        
        double std_dev = std::sqrt(variance);
        
        if (std_dev > 1e-12) {
            // Annualize assuming 390 bars per trading day, 252 trading days per year
            double annualized_return = mean_return * 390 * 252;
            double annualized_volatility = std_dev * std::sqrt(390 * 252);
            result.sharpe_ratio = annualized_return / annualized_volatility;
        } else {
            result.sharpe_ratio = 0.0;
        }
    } else {
        result.sharpe_ratio = 0.0;
    }
    
    // Calculate maximum drawdown within this block
    double peak = starting_equity;
    double max_drawdown = 0.0;
    
    for (const auto& [timestamp, equity] : equity_curve) {
        if (equity > peak) {
            peak = equity;
        } else {
            double drawdown = (peak - equity) / peak;
            max_drawdown = std::max(max_drawdown, drawdown);
        }
    }
    
    result.max_drawdown_pct = max_drawdown * 100.0;
    
    return result;
}

CanonicalReport CanonicalEvaluator::aggregate_block_results(
    const TradingBlockConfig& config,
    const std::vector<BlockResult>& block_results,
    const std::string& strategy_name,
    const std::string& dataset_source) {
    
    CanonicalReport report;
    report.config = config;
    report.block_results = block_results;
    report.strategy_name = strategy_name;
    report.dataset_source = dataset_source;
    
    if (block_results.empty()) {
        return report;
    }
    
    // Calculate aggregate statistics
    std::vector<double> rpb_values;
    rpb_values.reserve(block_results.size());
    
    double total_compounded_return = 1.0;
    int total_fills = 0;
    double max_drawdown = 0.0;
    
    for (const auto& block : block_results) {
        rpb_values.push_back(block.return_per_block);
        total_compounded_return *= (1.0 + block.return_per_block);
        total_fills += block.fills;
        max_drawdown = std::max(max_drawdown, block.max_drawdown_pct);
    }
    
    // Calculate mean RPB
    report.mean_rpb = std::accumulate(rpb_values.begin(), rpb_values.end(), 0.0) / rpb_values.size();
    
    // Calculate standard deviation of RPB
    double variance = 0.0;
    for (double rpb : rpb_values) {
        double diff = rpb - report.mean_rpb;
        variance += diff * diff;
    }
    variance /= std::max<size_t>(1, rpb_values.size() - 1); // Sample variance
    report.stdev_rpb = std::sqrt(variance);
    
    // Calculate Annualized Return on Block (ARB)
    report.annualized_return_on_block = calculate_annualized_return(report.mean_rpb, config.block_size);
    
    // Calculate aggregate Sharpe ratio
    std::vector<double> all_sharpe_values;
    for (const auto& block : block_results) {
        if (std::isfinite(block.sharpe_ratio) && block.sharpe_ratio != 0.0) {
            all_sharpe_values.push_back(block.sharpe_ratio);
        }
    }
    
    if (!all_sharpe_values.empty()) {
        report.aggregate_sharpe = std::accumulate(all_sharpe_values.begin(), all_sharpe_values.end(), 0.0) / all_sharpe_values.size();
    }
    
    // Set other aggregated values
    report.total_fills = total_fills;
    report.avg_fills_per_block = static_cast<double>(total_fills) / block_results.size();
    report.total_return_pct = (total_compounded_return - 1.0) * 100.0;
    report.max_drawdown_across_blocks = max_drawdown;
    report.consistency_score = calculate_consistency_score(block_results);
    
    // Set time bounds
    if (!block_results.empty()) {
        report.test_start_ts_ms = block_results.front().start_ts_ms;
        report.test_end_ts_ms = block_results.back().end_ts_ms;
        report.total_bars_processed = config.block_size * static_cast<int>(block_results.size());
    }
    
    return report;
}

double CanonicalEvaluator::calculate_annualized_return(
    double mean_rpb, 
    int block_size, 
    int bars_per_trading_year) {
    
    if (block_size <= 0 || bars_per_trading_year <= 0) {
        return 0.0;
    }
    
    // Calculate number of Trading Blocks in a trading year
    // With default 480-bar blocks: 98,280 / 480 = ~204.75 TB/year
    double blocks_per_year = static_cast<double>(bars_per_trading_year) / block_size;
    
    if (blocks_per_year <= 0) {
        return 0.0;
    }
    
    // Use compound interest formula: ARB = ((1 + mean_RPB) ^ blocks_per_year) - 1
    // This gives us the annualized equivalent of the per-block return
    return std::pow(1.0 + mean_rpb, blocks_per_year) - 1.0;
}

double CanonicalEvaluator::calculate_consistency_score(const std::vector<BlockResult>& block_results) {
    if (block_results.size() < 2) {
        return 0.0; // Perfect consistency for single or no blocks
    }
    
    // Calculate coefficient of variation of RPB values
    std::vector<double> rpb_values;
    rpb_values.reserve(block_results.size());
    
    for (const auto& block : block_results) {
        rpb_values.push_back(block.return_per_block);
    }
    
    double mean = std::accumulate(rpb_values.begin(), rpb_values.end(), 0.0) / rpb_values.size();
    
    if (std::abs(mean) < 1e-12) {
        return 1.0; // High inconsistency if mean is near zero but returns vary
    }
    
    double variance = 0.0;
    for (double rpb : rpb_values) {
        double diff = rpb - mean;
        variance += diff * diff;
    }
    variance /= rpb_values.size();
    
    double std_dev = std::sqrt(variance);
    
    // Return coefficient of variation (higher = less consistent)
    return std_dev / std::abs(mean);
}

} // namespace sentio

```

## 📄 **FILE 143 of 180**: src/canonical_metrics.cpp

**File Information**:
- **Path**: `src/canonical_metrics.cpp`

- **Size**: 245 lines
- **Modified**: 2025-09-15 22:39:24

- **Type**: .cpp

```text
#include "sentio/canonical_metrics.hpp"
#include <iostream>
#include <iomanip>
#include <map>
#include <algorithm>
#include <sstream>

namespace sentio {

CanonicalMetrics::PerformanceMetrics CanonicalMetrics::calculate_from_equity_curve(
    const std::vector<std::pair<std::string, double>>& equity_curve,
    double starting_capital,
    int total_trades) {
    
    PerformanceMetrics metrics = {};
    
    if (equity_curve.empty()) {
        return metrics;
    }
    
    // Extract basic values
    metrics.starting_capital = starting_capital;
    metrics.final_equity = equity_curve.back().second;
    metrics.total_pnl = metrics.final_equity - metrics.starting_capital;
    metrics.total_return_pct = (metrics.total_pnl / metrics.starting_capital) * 100.0;
    metrics.total_trades = total_trades;
    
    // Extract daily returns for advanced calculations
    std::vector<double> daily_returns = extract_daily_returns(equity_curve);
    metrics.trading_days = static_cast<int>(daily_returns.size());
    
    // Calculate derived metrics
    metrics.monthly_projected_return_pct = calculate_mpr_from_daily_returns(daily_returns) * 100.0;
    metrics.sharpe_ratio = calculate_sharpe_ratio(daily_returns);
    
    // Extract equity values for drawdown calculation
    std::vector<double> equity_values;
    equity_values.reserve(equity_curve.size());
    for (const auto& [timestamp, equity] : equity_curve) {
        equity_values.push_back(equity);
    }
    metrics.max_drawdown_pct = calculate_max_drawdown(equity_values) * 100.0;
    
    // Calculate daily trades
    metrics.daily_trades = (metrics.trading_days > 0) ? 
        static_cast<double>(metrics.total_trades) / metrics.trading_days : 0.0;
    
    // Win rate calculation would require individual trade data, set to 0 for now
    metrics.win_rate_pct = 0.0;
    
    return metrics;
}

double CanonicalMetrics::calculate_mpr_from_daily_returns(const std::vector<double>& daily_returns) {
    if (daily_returns.empty()) {
        return 0.0;
    }
    
    // Calculate geometric mean of daily returns
    double cumulative_return = 1.0;
    for (double daily_return : daily_returns) {
        cumulative_return *= (1.0 + daily_return);
    }
    
    // Convert to daily growth rate
    double daily_growth_rate = std::pow(cumulative_return, 1.0 / daily_returns.size()) - 1.0;
    
    // Project to monthly return (21 trading days)
    double monthly_return = std::pow(1.0 + daily_growth_rate, TRADING_DAYS_PER_MONTH) - 1.0;
    
    return monthly_return;
}

double CanonicalMetrics::calculate_mpr_from_total_return(double total_return, int trading_days) {
    if (trading_days <= 0) {
        return 0.0;
    }
    
    // Convert total return to daily growth rate
    double daily_growth_rate = std::pow(1.0 + total_return, 1.0 / trading_days) - 1.0;
    
    // Project to monthly return
    double monthly_return = std::pow(1.0 + daily_growth_rate, TRADING_DAYS_PER_MONTH) - 1.0;
    
    return monthly_return;
}

double CanonicalMetrics::calculate_sharpe_ratio(const std::vector<double>& daily_returns) {
    if (daily_returns.size() < 2) {
        return 0.0;
    }
    
    // Calculate mean and standard deviation of daily returns
    double sum = 0.0;
    for (double ret : daily_returns) {
        sum += ret;
    }
    double mean = sum / daily_returns.size();
    
    double variance = 0.0;
    for (double ret : daily_returns) {
        variance += (ret - mean) * (ret - mean);
    }
    variance /= (daily_returns.size() - 1);
    double std_dev = std::sqrt(variance);
    
    if (std_dev < TOLERANCE_EPSILON) {
        return 0.0;
    }
    
    // Annualize and calculate Sharpe ratio (assuming risk-free rate = 0)
    double annualized_return = mean * TRADING_DAYS_PER_YEAR;
    double annualized_volatility = std_dev * std::sqrt(TRADING_DAYS_PER_YEAR);
    
    return annualized_return / annualized_volatility;
}

double CanonicalMetrics::calculate_max_drawdown(const std::vector<double>& equity_values) {
    if (equity_values.empty()) {
        return 0.0;
    }
    
    double max_drawdown = 0.0;
    double peak = equity_values[0];
    
    for (double equity : equity_values) {
        if (equity > peak) {
            peak = equity;
        }
        
        double drawdown = (peak - equity) / peak;
        if (drawdown > max_drawdown) {
            max_drawdown = drawdown;
        }
    }
    
    return max_drawdown;
}

std::vector<double> CanonicalMetrics::extract_daily_returns(
    const std::vector<std::pair<std::string, double>>& equity_curve) {
    
    if (equity_curve.size() < 2) {
        return {};
    }
    
    // Group by session date (first 10 characters of timestamp: YYYY-MM-DD)
    std::map<std::string, double> daily_equity;
    for (const auto& [timestamp, equity] : equity_curve) {
        std::string session_date = timestamp.substr(0, 10);
        daily_equity[session_date] = equity; // Keep the last equity value for each day
    }
    
    // Calculate daily returns
    std::vector<double> daily_returns;
    auto it = daily_equity.begin();
    double prev_equity = it->second;
    ++it;
    
    for (; it != daily_equity.end(); ++it) {
        double current_equity = it->second;
        double daily_return = (current_equity - prev_equity) / prev_equity;
        daily_returns.push_back(daily_return);
        prev_equity = current_equity;
    }
    
    return daily_returns;
}

bool CanonicalMetrics::validate_metrics_consistency(
    const PerformanceMetrics& metrics1,
    const PerformanceMetrics& metrics2,
    double tolerance_pct) {
    
    auto within_tolerance = [tolerance_pct](double val1, double val2) {
        if (std::abs(val1) < TOLERANCE_EPSILON && std::abs(val2) < TOLERANCE_EPSILON) {
            return true; // Both are effectively zero
        }
        double max_val = std::max(std::abs(val1), std::abs(val2));
        return std::abs(val1 - val2) <= (max_val * tolerance_pct / 100.0);
    };
    
    return within_tolerance(metrics1.total_return_pct, metrics2.total_return_pct) &&
           within_tolerance(metrics1.monthly_projected_return_pct, metrics2.monthly_projected_return_pct) &&
           within_tolerance(metrics1.daily_trades, metrics2.daily_trades);
}

void CanonicalMetrics::print_metrics_comparison(
    const PerformanceMetrics& canonical,
    const PerformanceMetrics& audit,
    const PerformanceMetrics& strattest,
    const std::string& run_id) {
    
    std::cout << "=== CANONICAL METRICS COMPARISON ===" << std::endl;
    std::cout << "Run ID: " << run_id << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "\n┌─────────────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐" << std::endl;
    std::cout << "│ Metric                  │ Canonical       │ Audit Summarize │ Strattest       │ Max Discrepancy │" << std::endl;
    std::cout << "├─────────────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤" << std::endl;
    
    // Total Return
    double max_return_disc = std::max(std::abs(canonical.total_return_pct - audit.total_return_pct), 
                                     std::abs(canonical.total_return_pct - strattest.total_return_pct));
    std::cout << "│ Total Return            │ " << std::setw(14) << canonical.total_return_pct << "% │ " 
              << std::setw(14) << audit.total_return_pct << "% │ " 
              << std::setw(14) << strattest.total_return_pct << "% │ " 
              << std::setw(14) << max_return_disc << "% │" << std::endl;
    
    // MPR
    double max_mpr_disc = std::max(std::abs(canonical.monthly_projected_return_pct - audit.monthly_projected_return_pct), 
                                  std::abs(canonical.monthly_projected_return_pct - strattest.monthly_projected_return_pct));
    std::cout << "│ Monthly Proj. Return    │ " << std::setw(14) << canonical.monthly_projected_return_pct << "% │ " 
              << std::setw(14) << audit.monthly_projected_return_pct << "% │ " 
              << std::setw(14) << strattest.monthly_projected_return_pct << "% │ " 
              << std::setw(14) << max_mpr_disc << "% │" << std::endl;
    
    // Sharpe Ratio
    double max_sharpe_disc = std::max(std::abs(canonical.sharpe_ratio - audit.sharpe_ratio), 
                                     std::abs(canonical.sharpe_ratio - strattest.sharpe_ratio));
    std::cout << "│ Sharpe Ratio            │ " << std::setw(14) << canonical.sharpe_ratio << "  │ " 
              << std::setw(14) << audit.sharpe_ratio << "  │ " 
              << std::setw(14) << strattest.sharpe_ratio << "  │ " 
              << std::setw(14) << max_sharpe_disc << "  │" << std::endl;
    
    // Daily Trades
    double max_trades_disc = std::max(std::abs(canonical.daily_trades - audit.daily_trades), 
                                     std::abs(canonical.daily_trades - strattest.daily_trades));
    std::cout << "│ Daily Trades            │ " << std::setw(14) << canonical.daily_trades << "  │ " 
              << std::setw(14) << audit.daily_trades << "  │ " 
              << std::setw(14) << strattest.daily_trades << "  │ " 
              << std::setw(14) << max_trades_disc << "  │" << std::endl;
    
    std::cout << "└─────────────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘" << std::endl;
    
    // Validation results
    std::cout << "\n=== VALIDATION RESULTS ===" << std::endl;
    bool canonical_audit_consistent = validate_metrics_consistency(canonical, audit, 1.0);
    bool canonical_strattest_consistent = validate_metrics_consistency(canonical, strattest, 5.0); // Higher tolerance for strattest
    
    std::cout << "Canonical vs Audit: " << (canonical_audit_consistent ? "✅ CONSISTENT" : "❌ INCONSISTENT") << std::endl;
    std::cout << "Canonical vs Strattest: " << (canonical_strattest_consistent ? "✅ CONSISTENT" : "❌ INCONSISTENT") << std::endl;
}

} // namespace sentio

```

## 📄 **FILE 144 of 180**: src/cli_helpers.cpp

**File Information**:
- **Path**: `src/cli_helpers.cpp`

- **Size**: 400 lines
- **Modified**: 2025-09-17 13:20:14

- **Type**: .cpp

```text
#include "sentio/cli_helpers.hpp"
#include "sentio/base_strategy.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <regex>
#include <filesystem>
#include <sstream>

namespace sentio {

CLIHelpers::ParsedArgs CLIHelpers::parse_arguments(int argc, char* argv[]) {
    ParsedArgs args;
    
    if (argc < 2) {
        return args;
    }
    
    args.command = argv[1];
    
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        
        // Check for help flag
        if (arg == "--help" || arg == "-h") {
            args.help_requested = true;
            continue;
        }
        
        // Check for verbose flag
        if (arg == "--verbose" || arg == "-v") {
            args.verbose = true;
            args.flags["verbose"] = true;
            continue;
        }
        
        // Check for options (--key value or --key=value)
        if (arg.starts_with("--")) {
            std::string key, value;
            
            size_t eq_pos = arg.find('=');
            if (eq_pos != std::string::npos) {
                // --key=value format
                key = arg.substr(2, eq_pos - 2);
                value = arg.substr(eq_pos + 1);
            } else {
                // --key value format
                key = arg.substr(2);
                if (i + 1 < argc && !(argv[i + 1][0] == '-')) {
                    value = argv[++i];
                } else {
                    // Flag without value
                    args.flags[normalize_option_key(key)] = true;
                    continue;
                }
            }
            
            args.options[normalize_option_key(key)] = value;
        }
        // Check for short options (-k value)
        else if (arg.starts_with("-") && arg.length() == 2) {
            std::string key = arg.substr(1);
            std::string value;
            
            if (i + 1 < argc && !(argv[i + 1][0] == '-')) {
                value = argv[++i];
                args.options[normalize_option_key(key)] = value;
            } else {
                args.flags[normalize_option_key(key)] = true;
            }
        }
        // Positional argument
        else {
            args.positional_args.push_back(arg);
        }
    }
    
    return args;
}

std::string CLIHelpers::get_option(const ParsedArgs& args, const std::string& key, 
                                  const std::string& default_value) {
    std::string normalized_key = normalize_option_key(key);
    auto it = args.options.find(normalized_key);
    return (it != args.options.end()) ? it->second : default_value;
}

int CLIHelpers::get_int_option(const ParsedArgs& args, const std::string& key, int default_value) {
    std::string value = get_option(args, key);
    if (value.empty()) {
        return default_value;
    }
    
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        std::cerr << "Warning: Invalid integer value '" << value << "' for option --" << key 
                  << ", using default " << default_value << std::endl;
        return default_value;
    }
}

double CLIHelpers::get_double_option(const ParsedArgs& args, const std::string& key, double default_value) {
    std::string value = get_option(args, key);
    if (value.empty()) {
        return default_value;
    }
    
    try {
        return std::stod(value);
    } catch (const std::exception&) {
        std::cerr << "Warning: Invalid double value '" << value << "' for option --" << key 
                  << ", using default " << default_value << std::endl;
        return default_value;
    }
}

bool CLIHelpers::get_flag(const ParsedArgs& args, const std::string& key) {
    std::string normalized_key = normalize_option_key(key);
    auto it = args.flags.find(normalized_key);
    return (it != args.flags.end()) ? it->second : false;
}

int CLIHelpers::parse_period_to_days(const std::string& period) {
    std::regex period_regex(R"((\d+)([yMwdh]))");
    std::smatch match;
    
    if (!std::regex_match(period, match, period_regex)) {
        std::cerr << "Warning: Invalid period format '" << period << "', using default 30d" << std::endl;
        return 30;
    }
    
    int value = std::stoi(match[1].str());
    char unit = match[2].str()[0];
    
    switch (unit) {
        case 'y': return value * 365;
        case 'M': return value * 30;
        case 'w': return value * 7;
        case 'd': return value;
        case 'h': return std::max(1, value / 24);
        default: return 30;
    }
}

int CLIHelpers::parse_period_to_minutes(const std::string& period) {
    std::regex period_regex(R"((\d+)([yMwdhm]))");
    std::smatch match;
    
    if (!std::regex_match(period, match, period_regex)) {
        std::cerr << "Warning: Invalid period format '" << period << "', using default 1d" << std::endl;
        return 390; // 1 trading day
    }
    
    int value = std::stoi(match[1].str());
    char unit = match[2].str()[0];
    
    switch (unit) {
        case 'y': return value * 252 * 390; // 252 trading days per year
        case 'M': return value * 22 * 390;  // ~22 trading days per month
        case 'w': return value * 5 * 390;   // 5 trading days per week
        case 'd': return value * 390;       // 390 minutes per trading day
        case 'h': return value * 60;
        case 'm': return value;
        default: return 390;
    }
}

bool CLIHelpers::validate_required_args(const ParsedArgs& args, int min_required, 
                                        const std::string& usage_msg) {
    if (static_cast<int>(args.positional_args.size()) < min_required) {
        print_error("Insufficient arguments", usage_msg);
        return false;
    }
    return true;
}

void CLIHelpers::print_help(const std::string& command, const std::string& usage,
                           const std::vector<std::string>& options,
                           const std::vector<std::string>& examples) {
    (void)command; // Suppress unused parameter warning
    std::cout << "Usage: " << usage << std::endl << std::endl;
    
    if (!options.empty()) {
        std::cout << "Options:" << std::endl;
        for (const auto& option : options) {
            std::cout << "  " << option << std::endl;
        }
        std::cout << std::endl;
    }
    
    if (!examples.empty()) {
        std::cout << "Examples:" << std::endl;
        for (const auto& example : examples) {
            std::cout << "  " << example << std::endl;
        }
        std::cout << std::endl;
    }
}

void CLIHelpers::print_error(const std::string& error_msg, const std::string& usage_hint) {
    std::cerr << "Error: " << error_msg << std::endl;
    if (!usage_hint.empty()) {
        std::cerr << "Usage: " << usage_hint << std::endl;
    }
    std::cerr << "Use --help for more information." << std::endl;
}

bool CLIHelpers::is_valid_symbol(const std::string& symbol) {
    // Basic symbol validation: 1-5 uppercase letters
    std::regex symbol_regex(R"([A-Z]{1,5})");
    return std::regex_match(symbol, symbol_regex);
}

bool CLIHelpers::is_valid_strategy_name(const std::string& strategy_name) {
    // Basic strategy name validation: alphanumeric and underscores
    std::regex strategy_regex(R"([a-zA-Z][a-zA-Z0-9_]*)");
    return std::regex_match(strategy_name, strategy_regex);
}

std::vector<std::string> CLIHelpers::get_available_strategies() {
    // This would ideally query the StrategyFactory for available strategies
    // For now, return a hardcoded list of common strategies
    return {
        "ire", "momentum", "mean_reversion", "rsi", "sma_cross", 
        "bollinger", "macd", "stochastic", "williams_r"
    };
}

std::string CLIHelpers::format_duration(int minutes) {
    if (minutes < 60) {
        return std::to_string(minutes) + "m";
    } else if (minutes < 390) {
        return std::to_string(minutes / 60) + "h";
    } else if (minutes < 390 * 7) {
        double days = static_cast<double>(minutes) / 390.0;
        if (days == static_cast<int>(days)) {
            return std::to_string(static_cast<int>(days)) + "d";
        } else {
            return std::to_string(days).substr(0, 4) + "d";
        }
    } else if (minutes < 390 * 30) {
        return std::to_string(minutes / (390 * 7)) + "w";
    } else {
        return std::to_string(minutes / (390 * 22)) + "M";
    }
}

std::vector<std::string> CLIHelpers::parse_list(const std::string& list_str, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(list_str);
    std::string item;
    
    while (std::getline(ss, item, delimiter)) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    
    return result;
}

bool CLIHelpers::file_exists(const std::string& filepath) {
    return std::filesystem::exists(filepath);
}

std::string CLIHelpers::get_default_data_file(const std::string& symbol, const std::string& suffix) {
    return "data/equities/" + symbol + suffix;
}

std::string CLIHelpers::normalize_option_key(const std::string& key) {
    std::string normalized = key;
    
    // Remove leading dashes
    while (normalized.starts_with("-")) {
        normalized = normalized.substr(1);
    }
    
    // Convert to lowercase
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    
    return normalized;
}

bool CLIHelpers::is_option(const std::string& arg) {
    return arg.starts_with("-");
}

bool CLIHelpers::is_flag(const std::string& arg) {
    return arg.starts_with("--") && arg.find('=') == std::string::npos;
}

bool CLIHelpers::validate_options(const ParsedArgs& args, const std::string& command,
                                 const std::vector<std::string>& allowed_options,
                                 const std::vector<std::string>& allowed_flags) {
    // Check for unknown options
    for (const auto& [option, value] : args.options) {
        bool found = false;
        for (const auto& allowed : allowed_options) {
            if (normalize_option_key(option) == normalize_option_key(allowed)) {
                found = true;
                break;
            }
        }
        if (!found) {
            print_unknown_option_error(option, command, allowed_options, allowed_flags);
            return false;
        }
    }
    
    // Check for unknown flags
    for (const auto& [flag, value] : args.flags) {
        // Skip global flags
        if (flag == "verbose" || flag == "help" || flag == "h" || flag == "v") {
            continue;
        }
        
        bool found = false;
        for (const auto& allowed : allowed_flags) {
            if (normalize_option_key(flag) == normalize_option_key(allowed)) {
                found = true;
                break;
            }
        }
        if (!found) {
            print_unknown_option_error(flag, command, allowed_options, allowed_flags);
            return false;
        }
    }
    
    return true;
}

void CLIHelpers::print_unknown_option_error(const std::string& unknown_option,
                                           const std::string& command,
                                           const std::vector<std::string>& allowed_options,
                                           const std::vector<std::string>& allowed_flags) {
    std::cout << "\033[31m❌ ERROR:\033[0m Unknown option '--" << unknown_option << "' for command '" << command << "'" << std::endl;
    std::cout << std::endl;
    
    // Find similar options (simple string matching)
    std::vector<std::string> suggestions;
    std::string normalized_unknown = normalize_option_key(unknown_option);
    
    // Check for partial matches in allowed options
    for (const auto& allowed : allowed_options) {
        std::string normalized_allowed = normalize_option_key(allowed);
        if (normalized_allowed.find(normalized_unknown) != std::string::npos ||
            normalized_unknown.find(normalized_allowed) != std::string::npos) {
            suggestions.push_back(allowed);
        }
    }
    
    // Check for partial matches in allowed flags
    for (const auto& allowed : allowed_flags) {
        std::string normalized_allowed = normalize_option_key(allowed);
        if (normalized_allowed.find(normalized_unknown) != std::string::npos ||
            normalized_unknown.find(normalized_allowed) != std::string::npos) {
            suggestions.push_back(allowed);
        }
    }
    
    if (!suggestions.empty()) {
        std::cout << "\033[33m💡 Did you mean:\033[0m" << std::endl;
        for (const auto& suggestion : suggestions) {
            std::cout << "  --" << suggestion << std::endl;
        }
        std::cout << std::endl;
    }
    
    std::cout << "\033[36m📋 Available options for '" << command << "':\033[0m" << std::endl;
    
    if (!allowed_options.empty()) {
        std::cout << "\033[1mOptions (require values):\033[0m" << std::endl;
        for (const auto& option : allowed_options) {
            std::cout << "  --" << option << " <value>" << std::endl;
        }
        std::cout << std::endl;
    }
    
    if (!allowed_flags.empty()) {
        std::cout << "\033[1mFlags (no values):\033[0m" << std::endl;
        for (const auto& flag : allowed_flags) {
            std::cout << "  --" << flag << std::endl;
        }
        std::cout << std::endl;
    }
    
    std::cout << "\033[1mGlobal options:\033[0m" << std::endl;
    std::cout << "  --help, -h                Show command help" << std::endl;
    std::cout << "  --verbose, -v             Enable verbose output" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Use 'sentio_cli " << command << " --help' for detailed usage information." << std::endl;
}

} // namespace sentio

```

## 📄 **FILE 145 of 180**: src/csv_loader.cpp

**File Information**:
- **Path**: `src/csv_loader.cpp`

- **Size**: 159 lines
- **Modified**: 2025-09-18 14:34:50

- **Type**: .cpp

```text
#include "sentio/csv_loader.hpp"
#include "sentio/binio.hpp"
#include "sentio/global_leverage_config.hpp"
#include "sentio/leverage_aware_csv_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <cctz/time_zone.h>
#include <cctz/civil_time.h>

namespace sentio {

bool load_csv(const std::string& filename, std::vector<Bar>& out) {
    // Check if this is a leverage instrument and theoretical pricing is enabled
    if (GlobalLeverageConfig::is_theoretical_leverage_pricing_enabled()) {
        // Extract symbol from filename
        std::string symbol;
        size_t last_slash = filename.find_last_of("/\\");
        size_t last_dot = filename.find_last_of(".");
        if (last_slash != std::string::npos && last_dot != std::string::npos && last_dot > last_slash) {
            symbol = filename.substr(last_slash + 1, last_dot - last_slash - 1);
            // Remove any suffix like _NH_ALIGNED
            size_t underscore = symbol.find('_');
            if (underscore != std::string::npos) {
                symbol = symbol.substr(0, underscore);
            }
        }
        
        // If this is a leverage instrument, use theoretical pricing
        if (symbol == "TQQQ" || symbol == "SQQQ" || symbol == "PSQ") {
            std::cout << "🧮 Using theoretical pricing for " << symbol << " (based on QQQ)" << std::endl;
            return load_csv_leverage_aware(symbol, out);
        }
    }
    
    namespace fs = std::filesystem;
    
    // **SMART FRESHNESS-BASED LOADING**: Choose between CSV and binary based on file timestamps
    std::string bin_filename = filename.substr(0, filename.find_last_of('.')) + ".bin";
    
    bool csv_exists = fs::exists(filename);
    bool bin_exists = fs::exists(bin_filename);
    
    // **FRESHNESS COMPARISON**: Use binary only if it's newer than CSV
    bool use_binary = false;
    if (bin_exists && csv_exists) {
        auto csv_time = fs::last_write_time(filename);
        auto bin_time = fs::last_write_time(bin_filename);
        use_binary = (bin_time >= csv_time);
        
        if (use_binary) {
            std::cout << "📦 Using cached binary data (fresher than CSV): " << bin_filename << std::endl;
        } else {
            std::cout << "🔄 CSV file is newer than binary cache, reloading: " << filename << std::endl;
        }
    } else if (bin_exists && !csv_exists) {
        use_binary = true;
        std::cout << "📦 Using binary data (CSV not found): " << bin_filename << std::endl;
    } else if (!bin_exists && csv_exists) {
        use_binary = false;
        std::cout << "📄 Loading CSV data (no binary cache): " << filename << std::endl;
    } else {
        std::cerr << "❌ Neither CSV nor binary file exists: " << filename << std::endl;
        return false;
    }
    
    // **LOAD FROM BINARY**: Use cached binary if it's fresher
    if (use_binary) {
        auto cached = load_bin(bin_filename);
        if (!cached.empty()) {
            out = std::move(cached);
            return true;
        } else {
            std::cerr << "⚠️  Binary cache corrupted, falling back to CSV: " << bin_filename << std::endl;
        }
    }
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string timestamp_str, symbol, open_str, high_str, low_str, close_str, volume_str;
        
        // Standard Polygon format: timestamp,symbol,open,high,low,close,volume
        std::getline(ss, timestamp_str, ',');
        std::getline(ss, symbol, ',');
        std::getline(ss, open_str, ',');
        std::getline(ss, high_str, ',');
        std::getline(ss, low_str, ',');
        std::getline(ss, close_str, ',');
        std::getline(ss, volume_str, ',');
        
        Bar bar;
        bar.ts_utc = timestamp_str;
        
        // **MODIFIED**: Parse ISO 8601 timestamp directly as UTC
        try {
            // Parse the RFC3339 / ISO 8601 timestamp string (e.g., "2023-10-27T13:30:00Z")
            cctz::time_zone utc_tz;
            if (cctz::load_time_zone("UTC", &utc_tz)) {
                cctz::time_point<cctz::seconds> utc_tp;
                if (cctz::parse("%Y-%m-%dT%H:%M:%S%Ez", timestamp_str, utc_tz, &utc_tp)) {
                    bar.ts_utc_epoch = utc_tp.time_since_epoch().count();
                } else {
                    // Try alternative format with Z suffix
                    if (cctz::parse("%Y-%m-%dT%H:%M:%SZ", timestamp_str, utc_tz, &utc_tp)) {
                        bar.ts_utc_epoch = utc_tp.time_since_epoch().count();
                    } else {
                        // Try space format with timezone
                        if (cctz::parse("%Y-%m-%d %H:%M:%S%Ez", timestamp_str, utc_tz, &utc_tp)) {
                            bar.ts_utc_epoch = utc_tp.time_since_epoch().count();
                        } else {
                            // Try space format without timezone (assume UTC)
                            if (cctz::parse("%Y-%m-%d %H:%M:%S", timestamp_str, utc_tz, &utc_tp)) {
                                bar.ts_utc_epoch = utc_tp.time_since_epoch().count();
                            } else {
                                bar.ts_utc_epoch = 0;
                            }
                        }
                    }
                }
            } else {
                bar.ts_utc_epoch = 0; // Could not load timezone
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse timestamp '" << timestamp_str << "'. Error: " << e.what() << std::endl;
            bar.ts_utc_epoch = 0;
        }
        
        try {
            bar.open = std::stod(open_str);
            bar.high = std::stod(high_str);
            bar.low = std::stod(low_str);
            bar.close = std::stod(close_str);
            bar.volume = std::stoull(volume_str);
            out.push_back(bar);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse bar data line: " << line << std::endl;
        }
    }
    
    // **REGENERATE BINARY CACHE**: Save to binary cache for next time
    if (!out.empty()) {
        save_bin(bin_filename, out);
        std::cout << "💾 Regenerated binary cache: " << bin_filename << " (" << out.size() << " bars)" << std::endl;
    }
    
    return true;
}

} // namespace sentio
```

## 📄 **FILE 146 of 180**: src/data_downloader.cpp

**File Information**:
- **Path**: `src/data_downloader.cpp`

- **Size**: 164 lines
- **Modified**: 2025-09-16 00:39:14

- **Type**: .cpp

```text
#include "sentio/data_downloader.hpp"
#include "sentio/polygon_client.hpp"
#include "sentio/time_utils.hpp"
#include <iostream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <algorithm>

namespace sentio {

std::string get_yesterday_date() {
    std::time_t now = std::time(nullptr);
    std::time_t yesterday = now - 24 * 60 * 60; // Subtract 1 day in seconds
    
    std::tm* tm_yesterday = std::gmtime(&yesterday);
    std::ostringstream oss;
    oss << std::put_time(tm_yesterday, "%Y-%m-%d");
    return oss.str();
}

std::string get_current_date() {
    std::time_t now = std::time(nullptr);
    std::tm* tm_now = std::gmtime(&now);
    std::ostringstream oss;
    oss << std::put_time(tm_now, "%Y-%m-%d");
    return oss.str();
}


std::string symbol_to_family(const std::string& symbol) {
    std::string upper_symbol = symbol;
    std::transform(upper_symbol.begin(), upper_symbol.end(), upper_symbol.begin(), ::toupper);
    
    if (upper_symbol == "QQQ" || upper_symbol == "TQQQ" || upper_symbol == "SQQQ") {
        return "qqq";
    } else if (upper_symbol == "BTC" || upper_symbol == "BTCUSD" || upper_symbol == "ETH" || upper_symbol == "ETHUSD") {
        return "bitcoin";
    } else if (upper_symbol == "TSLA" || upper_symbol == "TSLQ") {
        return "tesla";
    } else {
        return "custom";
    }
}

std::vector<std::string> get_family_symbols(const std::string& family) {
    if (family == "qqq") {
        return {"QQQ", "TQQQ", "SQQQ"};
    } else if (family == "bitcoin") {
        return {"X:BTCUSD", "X:ETHUSD"};
    } else if (family == "tesla") {
        return {"TSLA", "TSLQ"};
    } else {
        return {}; // Empty for custom family
    }
}

bool download_symbol_data(const std::string& symbol,
                         int years,
                         int months,
                         int days,
                         const std::string& timespan,
                         int multiplier,
                         bool exclude_holidays,
                         const std::string& output_dir) {
    
    // **POLYGON API KEY CHECK**
    const char* key = std::getenv("POLYGON_API_KEY");
    if (!key || std::string(key).empty()) {
        std::cerr << "❌ Error: POLYGON_API_KEY environment variable not set" << std::endl;
        std::cerr << "   Please set your Polygon API key: export POLYGON_API_KEY=your_key_here" << std::endl;
        return false;
    }
    
    std::string api_key = key;
    PolygonClient cli(api_key);
    
    // **FAMILY DETECTION**
    std::string family = symbol_to_family(symbol);
    std::vector<std::string> symbols;
    
    if (family == "custom") {
        // Single symbol download
        symbols = {symbol};
        std::cout << "📊 Downloading data for symbol: " << symbol << std::endl;
    } else {
        // Family download
        symbols = get_family_symbols(family);
        std::cout << "📊 Downloading data for " << family << " family: ";
        for (size_t i = 0; i < symbols.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << symbols[i];
        }
        std::cout << std::endl;
    }
    
    // **DATE CALCULATION**
    std::string from = sentio::calculate_start_date(years, months, days);
    std::string to = get_yesterday_date();
    
    std::cout << "📅 Current date: " << get_current_date() << std::endl;
    std::cout << "📅 Downloading ";
    if (years > 0) {
        std::cout << years << " year" << (years > 1 ? "s" : "");
    } else if (months > 0) {
        std::cout << months << " month" << (months > 1 ? "s" : "");
    } else if (days > 0) {
        std::cout << days << " day" << (days > 1 ? "s" : "");
    } else {
        std::cout << "3 years (default)";
    }
    std::cout << " of data: " << from << " to " << to << std::endl;
    std::cout << "📈 Timespan: " << timespan << " (multiplier: " << multiplier << ")" << std::endl;
    
    // **DOWNLOAD EACH SYMBOL**
    bool all_success = true;
    for (const auto& sym : symbols) {
        std::cout << "\\n🔄 Downloading " << sym << "..." << std::endl;
        
        AggsQuery q;
        q.symbol = sym;
        q.from = from;
        q.to = to;
        q.timespan = timespan;
        q.multiplier = multiplier;
        q.adjusted = true;
        q.sort = "asc";
        
        try {
            auto bars = cli.get_aggs_all(q);
            
            if (bars.empty()) {
                std::cerr << "⚠️  Warning: No data received for " << sym << std::endl;
                continue;
            }
            
            // **FILE NAMING**
            std::string suffix;
            if (exclude_holidays) suffix += "_NH";
            std::string fname = output_dir + "/" + sym + suffix + ".csv";
            
            // **WRITE CSV**
            cli.write_csv(fname, sym, bars, exclude_holidays);
            
            std::cout << "✅ Wrote " << bars.size() << " bars -> " << fname << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Error downloading " << sym << ": " << e.what() << std::endl;
            all_success = false;
        }
    }
    
    if (all_success) {
        std::cout << "\\n🎉 Download completed successfully!" << std::endl;
        std::cout << "💡 Tip: The smart data loading system will automatically use this fresh data on your next run." << std::endl;
    } else {
        std::cout << "\\n⚠️  Download completed with some errors. Check the output above." << std::endl;
    }
    
    return all_success;
}

} // namespace sentio

```

## 📄 **FILE 147 of 180**: src/eod_position_manager.cpp

**File Information**:
- **Path**: `src/eod_position_manager.cpp`

- **Size**: 444 lines
- **Modified**: 2025-09-18 17:15:13

- **Type**: .cpp

```text
#include "sentio/eod_position_manager.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace sentio {

// Static ETF classifications for risk management
const std::unordered_set<std::string> EODPositionManager::LEVERAGED_ETFS = {
    "TQQQ", "SQQQ", "QLD", "QID", "UPRO", "SPXU", "TNA", "TZA", 
    "TECL", "TECS", "CURE", "RXD", "LABU", "LABD"
};

const std::unordered_set<std::string> EODPositionManager::HIGH_DECAY_ETFS = {
    "TQQQ", "SQQQ",  // 3x leveraged - highest decay
    "QLD", "QID",    // 2x leveraged - moderate decay
    "UPRO", "SPXU"   // 3x S&P - high decay
};

const std::unordered_set<std::string> EODPositionManager::EXEMPT_SYMBOLS = {
    "CASH", "USD", "MMKT"  // Cash equivalents - no need to close
};

int EODPositionManager::process_eod_closure(
    Portfolio& portfolio,
    const std::vector<double>& last_prices,
    const SymbolTable& ST,
    int64_t current_timestamp,
    IAuditRecorder& audit
) {
    int positions_closed = 0;
    
    // Determine current market timing and closure phase
    int minutes_to_close = get_minutes_to_close(current_timestamp);
    ClosurePhase new_phase = determine_closure_phase(minutes_to_close);
    
    // Log phase transitions
    if (new_phase != current_phase_) {
        if (config_.detailed_logging) {
            std::ostringstream log_msg;
            log_msg << "EOD Phase transition: " << phase_to_string(current_phase_) 
                   << " → " << phase_to_string(new_phase)
                   << " (T-" << minutes_to_close << "m)";
            stats_.closure_log.push_back(log_msg.str());
        }
        current_phase_ = new_phase;
    }
    
    // No action needed during normal trading or after market close
    if (current_phase_ == ClosurePhase::NORMAL_TRADING || 
        current_phase_ == ClosurePhase::MARKET_CLOSED) {
        return 0;
    }
    
    // Analyze current position risks
    std::vector<PositionRisk> position_risks = analyze_position_risks(portfolio, last_prices, ST);
    
    if (position_risks.empty()) {
        return 0; // No positions to close
    }
    
    // Prioritize positions based on risk and configuration
    std::vector<PositionRisk> prioritized = prioritize_positions(position_risks);
    
    // Execute closure based on current phase
    switch (current_phase_) {
        case ClosurePhase::WARNING:
            // Warning phase - just log, no action
            if (config_.detailed_logging && !prioritized.empty()) {
                std::ostringstream warning;
                warning << "EOD Warning: " << prioritized.size() << " positions will be closed in " 
                       << (minutes_to_close - config_.partial_close_minutes) << " minutes";
                stats_.closure_log.push_back(warning.str());
            }
            break;
            
        case ClosurePhase::PARTIAL_CLOSE:
            // Partial closure - close high-risk positions partially
            if (config_.allow_partial_closure) {
                for (const auto& risk : prioritized) {
                    if (risk.is_leveraged_etf || risk.decay_risk_score > 0.7) {
                        bool closed = close_position(
                            portfolio, risk.symbol_id, risk.quantity, 
                            last_prices[risk.symbol_id], ST, current_timestamp, audit,
                            config_.closure_reason_prefix + "_PARTIAL",
                            config_.partial_closure_fraction
                        );
                        if (closed) {
                            positions_closed++;
                            stats_.partial_closures++;
                        }
                    }
                }
            }
            break;
            
        case ClosurePhase::MANDATORY_CLOSE:
            // Mandatory closure - close all remaining positions
            for (const auto& risk : prioritized) {
                bool closed = close_position(
                    portfolio, risk.symbol_id, risk.quantity,
                    last_prices[risk.symbol_id], ST, current_timestamp, audit,
                    config_.closure_reason_prefix + "_MANDATORY"
                );
                if (closed) {
                    positions_closed++;
                    stats_.mandatory_closures++;
                }
            }
            break;
            
        case ClosurePhase::FINAL_SWEEP:
            // Final sweep - emergency closure of any remaining positions
            positions_closed = force_immediate_closure(
                portfolio, last_prices, ST, current_timestamp, audit,
                config_.closure_reason_prefix + "_FINAL_SWEEP"
            );
            break;
            
        default:
            break;
    }
    
    // Update statistics
    stats_.total_closures_today += positions_closed;
    if (positions_closed > 0) {
        stats_.last_closure_time = format_timestamp(current_timestamp);
    }
    
    return positions_closed;
}

int EODPositionManager::get_minutes_to_close(int64_t current_timestamp) const {
    MarketTiming timing = get_market_timing(current_timestamp);
    
    // Convert timestamp to UTC time components
    time_t raw_time = current_timestamp;
    struct tm* utc_tm = gmtime(&raw_time);
    int hour_utc = utc_tm->tm_hour;
    int minute_utc = utc_tm->tm_min;
    
    // **ADAPTIVE MARKET CLOSE DETECTION**
    // Historical data often ends before theoretical market close (4:00 PM EDT)
    // Detect if we're near end of trading day and adjust accordingly
    
    int current_minutes = hour_utc * 60 + minute_utc;
    int theoretical_close_minutes = timing.market_close_hour_utc * 60 + timing.market_close_minute_utc;
    int minutes_to_theoretical_close = theoretical_close_minutes - current_minutes;
    
    // **DATA-AWARE CLOSURE TIMING**
    // If we're within 60 minutes of theoretical close, assume data might end early
    // and trigger closure phases more aggressively
    if (minutes_to_theoretical_close <= 60) {
        // **AGGRESSIVE CLOSURE MODE**: Assume current time is close to actual data end
        // Trigger closure phases as if we're much closer to close than theoretical time suggests
        int adjusted_minutes_to_close = std::min(minutes_to_theoretical_close, 10);
        
        // If we're in the last hour of trading, be very aggressive about closure
        if (hour_utc >= 19) { // 7 PM UTC = 3 PM EDT, getting close to 4 PM close
            adjusted_minutes_to_close = std::min(adjusted_minutes_to_close, 5);
        }
        
        return adjusted_minutes_to_close;
    }
    
    // Handle day wrap-around (shouldn't happen with RTH data, but safety check)
    if (minutes_to_theoretical_close < -300) {
        minutes_to_theoretical_close += 24 * 60;
    }
    
    return minutes_to_theoretical_close;
}

EODPositionManager::MarketTiming EODPositionManager::get_market_timing(int64_t timestamp) const {
    MarketTiming timing;
    
    // Convert to UTC time for DST calculation
    time_t raw_time = timestamp;
    struct tm* utc_tm = gmtime(&raw_time);
    int month = utc_tm->tm_mon + 1; // tm_mon is 0-based
    
    // Simple DST check: March-November is EDT (20:00 UTC close), rest is EST (21:00 UTC close)
    // More precise DST rules can be implemented if needed
    timing.is_edt = (month >= 3 && month <= 11);
    timing.market_close_hour_utc = timing.is_edt ? 20 : 21;
    timing.market_close_minute_utc = 0;
    
    return timing;
}

EODPositionManager::ClosurePhase EODPositionManager::determine_closure_phase(int minutes_to_close) const {
    if (minutes_to_close <= 0) {
        return ClosurePhase::MARKET_CLOSED;
    } else if (minutes_to_close <= config_.final_sweep_minutes) {
        return ClosurePhase::FINAL_SWEEP;
    } else if (minutes_to_close <= config_.mandatory_close_minutes) {
        return ClosurePhase::MANDATORY_CLOSE;
    } else if (minutes_to_close <= config_.partial_close_minutes) {
        return ClosurePhase::PARTIAL_CLOSE;
    } else if (minutes_to_close <= config_.warning_minutes_before_close) {
        return ClosurePhase::WARNING;
    } else {
        return ClosurePhase::NORMAL_TRADING;
    }
}

std::vector<EODPositionManager::PositionRisk> EODPositionManager::analyze_position_risks(
    const Portfolio& portfolio,
    const std::vector<double>& last_prices,
    const SymbolTable& ST
) const {
    std::vector<PositionRisk> risks;
    
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        const auto& pos = portfolio.positions[sid];
        
        // Skip positions that are effectively zero or exempt
        if (std::abs(pos.qty) < 1e-6) continue;
        
        std::string symbol = ST.get_symbol(sid);
        if (is_position_exempt(symbol)) continue;
        
        // Calculate market value
        double market_value = std::abs(pos.qty * last_prices[sid]);
        
        // Skip positions below minimum value threshold
        if (market_value < config_.min_position_value) continue;
        
        PositionRisk risk;
        risk.symbol_id = sid;
        risk.symbol = symbol;
        risk.quantity = pos.qty;
        risk.market_value = market_value;
        risk.is_leveraged_etf = LEVERAGED_ETFS.count(symbol) > 0;
        risk.decay_risk_score = calculate_decay_risk_score(symbol, market_value);
        
        risks.push_back(risk);
    }
    
    return risks;
}

double EODPositionManager::calculate_decay_risk_score(const std::string& symbol, double market_value) const {
    double base_score = 0.1; // Base risk for all positions
    
    // **LEVERAGED ETF DECAY RISK**
    if (HIGH_DECAY_ETFS.count(symbol)) {
        base_score += 0.8; // Very high decay risk (3x leveraged)
    } else if (LEVERAGED_ETFS.count(symbol)) {
        base_score += 0.5; // Moderate decay risk (2x leveraged)
    }
    
    // **POSITION SIZE RISK** - Larger positions get higher priority
    if (market_value > 50000) {
        base_score += 0.3; // Large position
    } else if (market_value > 20000) {
        base_score += 0.2; // Medium position
    } else if (market_value > 5000) {
        base_score += 0.1; // Small position
    }
    
    // **SYMBOL-SPECIFIC RISK ADJUSTMENTS**
    if (symbol == "TQQQ" || symbol == "SQQQ") {
        base_score += 0.2; // Extra risk for 3x NASDAQ ETFs
    }
    
    return std::min(1.0, base_score); // Cap at 1.0
}

std::vector<EODPositionManager::PositionRisk> EODPositionManager::prioritize_positions(
    const std::vector<PositionRisk>& positions
) const {
    std::vector<PositionRisk> prioritized = positions;
    
    // Sort by priority: decay risk score (desc), then market value (desc)
    std::sort(prioritized.begin(), prioritized.end(), 
        [](const PositionRisk& a, const PositionRisk& b) {
            // Primary sort: decay risk score (higher = more urgent)
            if (std::abs(a.decay_risk_score - b.decay_risk_score) > 0.01) {
                return a.decay_risk_score > b.decay_risk_score;
            }
            // Secondary sort: market value (larger positions first)
            return a.market_value > b.market_value;
        });
    
    // Assign priority ranks
    for (size_t i = 0; i < prioritized.size(); ++i) {
        prioritized[i].priority_rank = static_cast<int>(i + 1);
    }
    
    return prioritized;
}

bool EODPositionManager::close_position(
    Portfolio& portfolio,
    size_t symbol_id,
    double quantity,
    double close_price,
    const SymbolTable& ST,
    int64_t timestamp,
    IAuditRecorder& audit,
    const std::string& reason,
    double closure_fraction
) {
    if (symbol_id >= portfolio.positions.size() || close_price <= 0) {
        return false;
    }
    
    const auto& pos = portfolio.positions[symbol_id];
    if (std::abs(pos.qty) < 1e-6) {
        return false; // No position to close
    }
    
    // Calculate quantity to close
    double qty_to_close = -pos.qty * closure_fraction; // Negative to close
    if (std::abs(qty_to_close * close_price) < config_.min_position_value) {
        return false; // Position too small to close
    }
    
    std::string symbol = ST.get_symbol(symbol_id);
    
    // Determine order side
    Side close_side = (qty_to_close > 0) ? Side::Buy : Side::Sell;
    
    // Log order event
    if (config_.detailed_logging) {
        audit.event_order_ex(timestamp, symbol, close_side, std::abs(qty_to_close), 0.0, reason);
    }
    
    // **PERFECT EXECUTION FOR EOD CLOSE** - No slippage, no fees
    double fees = 0.0;
    double exec_price = close_price;
    
    // Calculate realized P&L
    double realized_pnl = (pos.qty > 0) 
        ? (exec_price - pos.avg_price) * std::abs(qty_to_close)
        : (pos.avg_price - exec_price) * std::abs(qty_to_close);
    
    // Apply the fill
    apply_fill(portfolio, symbol_id, qty_to_close, exec_price);
    
    // Calculate equity after fill
    double equity_after = portfolio.cash;
    for (size_t i = 0; i < portfolio.positions.size(); ++i) {
        if (i < portfolio.positions.size() && std::abs(portfolio.positions[i].qty) > 1e-6) {
            equity_after += portfolio.positions[i].qty * close_price; // Approximation
        }
    }
    
    // Log fill event
    if (config_.detailed_logging) {
        audit.event_fill_ex(timestamp, symbol, exec_price, std::abs(qty_to_close), 
                           fees, close_side, realized_pnl, equity_after, 0.0, reason);
    }
    
    // Update statistics
    stats_.total_value_closed += std::abs(qty_to_close * exec_price);
    if (LEVERAGED_ETFS.count(symbol)) {
        stats_.leveraged_etf_closures++;
    }
    
    // Log closure event
    log_closure_event(symbol, qty_to_close, exec_price, reason, audit, timestamp);
    
    return true;
}

int EODPositionManager::force_immediate_closure(
    Portfolio& portfolio,
    const std::vector<double>& last_prices,
    const SymbolTable& ST,
    int64_t current_timestamp,
    IAuditRecorder& audit,
    const std::string& reason
) {
    int positions_closed = 0;
    
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        const auto& pos = portfolio.positions[sid];
        
        if (std::abs(pos.qty) > 1e-6 && sid < last_prices.size() && last_prices[sid] > 0) {
            std::string symbol = ST.get_symbol(sid);
            
            // Skip exempt positions
            if (is_position_exempt(symbol)) continue;
            
            bool closed = close_position(
                portfolio, sid, pos.qty, last_prices[sid], ST, 
                current_timestamp, audit, reason + "_FORCE"
            );
            
            if (closed) {
                positions_closed++;
            }
        }
    }
    
    return positions_closed;
}

bool EODPositionManager::is_position_exempt(const std::string& symbol) const {
    return EXEMPT_SYMBOLS.count(symbol) > 0;
}

void EODPositionManager::log_closure_event(
    const std::string& symbol,
    double quantity,
    double price,
    const std::string& reason,
    IAuditRecorder& /* audit */,
    int64_t timestamp
) {
    if (config_.detailed_logging) {
        std::ostringstream log_entry;
        log_entry << format_timestamp(timestamp) << " CLOSED " << symbol 
                 << " qty=" << quantity << " px=" << std::fixed << std::setprecision(2) << price
                 << " reason=" << reason;
        stats_.closure_log.push_back(log_entry.str());
    }
}

std::string EODPositionManager::format_timestamp(int64_t timestamp) const {
    time_t raw_time = timestamp;
    struct tm* utc_tm = gmtime(&raw_time);
    
    std::ostringstream oss;
    oss << std::put_time(utc_tm, "%H:%M:%S");
    return oss.str();
}

std::string EODPositionManager::phase_to_string(ClosurePhase phase) const {
    switch (phase) {
        case ClosurePhase::NORMAL_TRADING: return "NORMAL_TRADING";
        case ClosurePhase::WARNING: return "WARNING";
        case ClosurePhase::PARTIAL_CLOSE: return "PARTIAL_CLOSE";
        case ClosurePhase::MANDATORY_CLOSE: return "MANDATORY_CLOSE";
        case ClosurePhase::FINAL_SWEEP: return "FINAL_SWEEP";
        case ClosurePhase::MARKET_CLOSED: return "MARKET_CLOSED";
        default: return "UNKNOWN";
    }
}

} // namespace sentio

```

## 📄 **FILE 148 of 180**: src/feature_builder.cpp

**File Information**:
- **Path**: `src/feature_builder.cpp`

- **Size**: 109 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .cpp

```text
#include "sentio/feature_builder.hpp"
#include <algorithm>

namespace sentio {

FeatureBuilder::FeatureBuilder(FeaturePlan plan, FeatureBuilderCfg cfg)
: plan_(std::move(plan))
, cfg_(cfg)
, sma_fast_(cfg.sma_fast)
, sma_slow_(cfg.sma_slow)
, vol_rtn_(std::max(2, cfg.vol_window))
, rsi_(cfg.rsi_period)
{}

void FeatureBuilder::reset(){
  *this = FeatureBuilder(plan_, cfg_);
}

void FeatureBuilder::on_bar(const Bar& b, const std::optional<MicroTick>& mt) {
  ++bars_seen_;

  // --- returns ---
  if (!close_q_.empty()) {
    double prev = close_q_.back();
    last_ret_1m_ = (b.close - prev) / std::max(1e-12, prev);
    vol_rtn_.push(last_ret_1m_);
  } else {
    last_ret_1m_ = 0.0;
  }

  close_q_.push_back(b.close);
  if (close_q_.size() > (size_t)std::max({cfg_.ret_5m_window, cfg_.sma_slow, cfg_.rsi_period+1})) {
    close_q_.pop_front();
  }

  if (close_q_.size() >= (size_t)cfg_.ret_5m_window+1) {
    double prev5 = close_q_[close_q_.size()-(cfg_.ret_5m_window+1)];
    last_ret_5m_ = (b.close - prev5) / std::max(1e-12, prev5);
  }

  // --- RSI ---
  if (close_q_.size() >= 2) {
    double prev = close_q_[close_q_.size()-2];
    rsi_.push(prev, b.close);
    last_rsi_ = rsi_.value();
  }

  // --- SMA fast/slow ---
  sma_fast_.push(b.close);
  sma_slow_.push(b.close);
  last_sma_fast_ = sma_fast_.full()? sma_fast_.mean() : NAN;
  last_sma_slow_ = sma_slow_.full()? sma_slow_.mean() : NAN;

  // --- Volatility (stdev of 1m returns) ---
  last_vol_1m_ = vol_rtn_.full()? vol_rtn_.stdev() : NAN;

  // --- Spread bp ---
  if (mt && std::isfinite(mt->bid) && std::isfinite(mt->ask)) {
    double mid = 0.5*(mt->bid + mt->ask);
    if (mid>0) last_spread_bp_ = 1e4 * (mt->ask - mt->bid) / mid;
  } else {
    // Proxy from high/low as a fallback (intrabar range proxy), otherwise default
    double mid = (b.high + b.low) * 0.5;
    if (mid>0) last_spread_bp_ = 1e4 * (b.high - b.low) / std::max(1e-12, mid) * 0.1; // scaled
    else       last_spread_bp_ = cfg_.default_spread_bp;
  }

  // clamp any negatives/NaNs on first bars
  if (!std::isfinite(last_ret_5m_)) last_ret_5m_ = 0.0;
  if (!std::isfinite(last_rsi_))    last_rsi_    = 50.0;
  if (!std::isfinite(last_sma_fast_)) last_sma_fast_ = b.close;
  if (!std::isfinite(last_sma_slow_)) last_sma_slow_ = b.close;
  if (!std::isfinite(last_vol_1m_))   last_vol_1m_   = 0.0;
  if (!std::isfinite(last_spread_bp_)) last_spread_bp_ = cfg_.default_spread_bp;
}

bool FeatureBuilder::ready() const {
  // Require the slowest indicator to be ready (SMA slow & RSI & vol window)
  bool sma_ok = sma_slow_.full();
  bool rsi_ok = rsi_.ready();
  bool vol_ok = vol_rtn_.full();
  // ret_5m needs at least 5+1 bars, covered by SMA slow usually; but check anyway
  bool r5_ok  = close_q_.size() >= (size_t)cfg_.ret_5m_window+1;
  return sma_ok && rsi_ok && vol_ok && r5_ok
      && finite(last_ret_1m_) && finite(last_ret_5m_) && finite(last_rsi_)
      && finite(last_sma_fast_) && finite(last_sma_slow_) && finite(last_vol_1m_) && finite(last_spread_bp_);
}

std::optional<std::vector<double>> FeatureBuilder::build() const {
  if (!ready()) return std::nullopt;
  std::vector<double> out; out.reserve(plan_.names.size());

  for (const auto& name : plan_.names) {
    if (name=="ret_1m")      out.push_back(last_ret_1m_);
    else if (name=="ret_5m") out.push_back(last_ret_5m_);
    else if (name=="rsi_14") out.push_back(last_rsi_);
    else if (name=="sma_10") out.push_back(last_sma_fast_);
    else if (name=="sma_30") out.push_back(last_sma_slow_);
    else if (name=="vol_1m") out.push_back(last_vol_1m_);
    else if (name=="spread_bp") out.push_back(last_spread_bp_);
    else {
      // Unknown feature: fail closed so you'll notice in tests
      throw std::runtime_error("FeatureBuilder: unsupported feature name: " + name);
    }
  }
  return out;
}

} // namespace sentio

```

## 📄 **FILE 149 of 180**: src/feature_cache.cpp

**File Information**:
- **Path**: `src/feature_cache.cpp`

- **Size**: 123 lines
- **Modified**: 2025-09-15 15:57:19

- **Type**: .cpp

```text
#include "sentio/feature_cache.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace sentio {

bool FeatureCache::load_from_csv(const std::string& feature_file_path) {
    std::ifstream file(feature_file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open feature file: " << feature_file_path << std::endl;
        return false;
    }

    std::string line;
    bool is_header = true;
    size_t lines_processed = 0;

    while (std::getline(file, line)) {
        if (is_header) {
            // Parse header to get feature names
            std::stringstream ss(line);
            std::string column;
            
            // Skip bar_index and timestamp columns
            std::getline(ss, column, ','); // bar_index
            std::getline(ss, column, ','); // timestamp
            
            // Read feature names
            feature_names_.clear();
            while (std::getline(ss, column, ',')) {
                feature_names_.push_back(column);
            }
            
            std::cout << "FeatureCache: Loaded " << feature_names_.size() << " feature names" << std::endl;
            is_header = false;
            continue;
        }

        // Parse data line
        std::stringstream ss(line);
        std::string cell;
        
        // Read bar_index
        if (!std::getline(ss, cell, ',')) continue;
        int bar_index = std::stoi(cell);
        
        // Skip timestamp
        if (!std::getline(ss, cell, ',')) continue;
        
        // Read features
        std::vector<double> features;
        features.reserve(feature_names_.size());
        
        while (std::getline(ss, cell, ',')) {
            features.push_back(std::stod(cell));
        }
        
        // Verify feature count  
        if (features.size() != feature_names_.size()) {
            std::cerr << "CRITICAL: Bar " << bar_index << " has " << features.size() 
                      << " features, expected " << feature_names_.size() << std::endl;
            std::cerr << "CSV line: " << line << std::endl;
            std::cerr << "This will cause missing features in get_features()!" << std::endl;
            continue;  // This is the bug - skipping bars causes missing data
        }
        
        // Debug output for first few bars
        if (lines_processed < 3) {
        }
        
        // Store features
        features_by_bar_[bar_index] = std::move(features);
        lines_processed++;
        
        // Progress reporting
        if (lines_processed % 50000 == 0) {
            std::cout << "FeatureCache: Loaded " << lines_processed << " bars..." << std::endl;
        }
    }

    total_bars_ = lines_processed;
    file.close();

    std::cout << "FeatureCache: Successfully loaded " << total_bars_ << " bars with " 
              << feature_names_.size() << " features each" << std::endl;
    std::cout << "FeatureCache: Recommended starting bar: " << recommended_start_bar_ << std::endl;
    
    return true;
}

std::vector<double> FeatureCache::get_features(int bar_index) const {
    auto it = features_by_bar_.find(bar_index);
    if (it != features_by_bar_.end()) {
        static int get_calls = 0;
        get_calls++;
        if (get_calls <= 5) {
        }
        return it->second;
    }
    std::cout << "[ERROR] FeatureCache::get_features(" << bar_index 
              << ") - bar not found! Returning empty vector." << std::endl;
    return {}; // Return empty vector if not found
}

bool FeatureCache::has_features(int bar_index) const {
    return features_by_bar_.find(bar_index) != features_by_bar_.end();
}

size_t FeatureCache::get_bar_count() const {
    return total_bars_;
}

int FeatureCache::get_recommended_start_bar() const {
    return recommended_start_bar_;
}

const std::vector<std::string>& FeatureCache::get_feature_names() const {
    return feature_names_;
}

} // namespace sentio

```

## 📄 **FILE 150 of 180**: src/feature_engineering/feature_normalizer.cpp

**File Information**:
- **Path**: `src/feature_engineering/feature_normalizer.cpp`

- **Size**: 373 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .cpp

```text
#include "sentio/feature_engineering/feature_normalizer.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

namespace sentio {
namespace feature_engineering {

FeatureNormalizer::FeatureNormalizer(size_t window_size) 
    : window_size_(window_size) {
    // Initialize with empty stats - updated to match actual feature count
    stats_.resize(55); // Updated from 52 to 55 to match actual feature count
    feature_history_.resize(55);
}

std::vector<double> FeatureNormalizer::normalize_features(const std::vector<double>& features) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (features.empty()) {
        return features;
    }
    
    // Update statistics with new features
    update_stats(features);
    
    // Apply robust normalization
    return robust_normalize(features);
}

std::vector<double> FeatureNormalizer::denormalize_features(const std::vector<double>& normalized_features) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (normalized_features.empty() || stats_.empty()) {
        return normalized_features;
    }
    
    std::vector<double> denormalized;
    denormalized.reserve(normalized_features.size());
    
    for (size_t i = 0; i < normalized_features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && stat.std > 0) {
            double denorm = (normalized_features[i] * stat.std) + stat.mean;
            denormalized.push_back(denorm);
        } else {
            denormalized.push_back(normalized_features[i]);
        }
    }
    
    return denormalized;
}

void FeatureNormalizer::update_stats(const std::vector<double>& features) {
    for (size_t i = 0; i < features.size() && i < feature_history_.size(); ++i) {
        update_feature_stats(i, features[i]);
    }
}

void FeatureNormalizer::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    for (auto& stat : stats_) {
        stat = NormalizationStats{};
    }
    
    for (auto& history : feature_history_) {
        history.clear();
    }
}

std::vector<double> FeatureNormalizer::z_score_normalize(const std::vector<double>& features) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (features.empty() || stats_.empty()) {
        return features;
    }
    
    std::vector<double> normalized;
    normalized.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && stat.std > 0) {
            double normalized_value = (features[i] - stat.mean) / stat.std;
            normalized.push_back(normalized_value);
        } else {
            normalized.push_back(features[i]);
        }
    }
    
    return normalized;
}

std::vector<double> FeatureNormalizer::min_max_normalize(const std::vector<double>& features) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (features.empty() || stats_.empty()) {
        return features;
    }
    
    std::vector<double> normalized;
    normalized.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && (stat.max - stat.min) > 0) {
            double normalized_value = (features[i] - stat.min) / (stat.max - stat.min);
            normalized.push_back(normalized_value);
        } else {
            normalized.push_back(features[i]);
        }
    }
    
    return normalized;
}

std::vector<double> FeatureNormalizer::robust_normalize(const std::vector<double>& features) {
    // Note: Mutex is already held by caller (normalize_features)
    
    if (features.empty() || stats_.empty()) {
        return features;
    }
    
    std::vector<double> normalized;
    normalized.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && stat.std > 0) {
            // Use robust statistics (median and MAD)
            double robust_mean = calculate_robust_mean(feature_history_[i]);
            double robust_std = calculate_robust_std(feature_history_[i], robust_mean);
            
            if (robust_std > 0) {
                double normalized_value = (features[i] - robust_mean) / robust_std;
                normalized.push_back(normalized_value);
            } else {
                normalized.push_back(features[i]);
            }
        } else {
            normalized.push_back(features[i]);
        }
    }
    
    return normalized;
}

std::vector<double> FeatureNormalizer::clip_outliers(const std::vector<double>& features, double threshold) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (features.empty() || stats_.empty()) {
        return features;
    }
    
    std::vector<double> clipped;
    clipped.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && stat.std > 0) {
            double upper_bound = stat.mean + (threshold * stat.std);
            double lower_bound = stat.mean - (threshold * stat.std);
            
            double clipped_value = std::clamp(features[i], lower_bound, upper_bound);
            clipped.push_back(clipped_value);
        } else {
            clipped.push_back(features[i]);
        }
    }
    
    return clipped;
}

std::vector<double> FeatureNormalizer::winsorize(const std::vector<double>& features, double percentile) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (features.empty() || stats_.empty()) {
        return features;
    }
    
    std::vector<double> winsorized;
    winsorized.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < feature_history_.size(); ++i) {
        if (feature_history_[i].size() < 2) {
            winsorized.push_back(features[i]);
            continue;
        }
        
        // Calculate percentiles from history
        auto sorted_values = feature_history_[i];
        sort_values(sorted_values);
        
        double lower_percentile = calculate_percentile(sorted_values, percentile);
        double upper_percentile = calculate_percentile(sorted_values, 1.0 - percentile);
        
        double winsorized_value = std::clamp(features[i], lower_percentile, upper_percentile);
        winsorized.push_back(winsorized_value);
    }
    
    return winsorized;
}

bool FeatureNormalizer::is_normalized(const std::vector<double>& features) const {
    if (features.empty()) {
        return true;
    }
    
    // Check if features are in reasonable normalized range [-5, 5]
    for (double feature : features) {
        if (std::abs(feature) > 5.0) {
            return false;
        }
    }
    
    return true;
}

std::vector<bool> FeatureNormalizer::get_outlier_mask(const std::vector<double>& features, double threshold) const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    std::vector<bool> outlier_mask;
    outlier_mask.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && stat.std > 0) {
            double upper_bound = stat.mean + (threshold * stat.std);
            double lower_bound = stat.mean - (threshold * stat.std);
            
            bool is_outlier = (features[i] < lower_bound) || (features[i] > upper_bound);
            outlier_mask.push_back(is_outlier);
        } else {
            outlier_mask.push_back(false);
        }
    }
    
    return outlier_mask;
}

NormalizationStats FeatureNormalizer::get_stats(size_t feature_index) const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (feature_index < stats_.size()) {
        return stats_[feature_index];
    }
    
    return NormalizationStats{};
}

std::vector<NormalizationStats> FeatureNormalizer::get_all_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void FeatureNormalizer::set_window_size(size_t window_size) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    window_size_ = window_size;
    
    // Trim existing histories to new window size
    for (auto& history : feature_history_) {
        while (history.size() > window_size_) {
            history.pop_front();
        }
    }
}

void FeatureNormalizer::set_outlier_threshold(double threshold) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    outlier_threshold_ = threshold;
}

void FeatureNormalizer::set_winsorize_percentile(double percentile) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    winsorize_percentile_ = percentile;
}

void FeatureNormalizer::update_feature_stats(size_t feature_index, double value) {
    if (feature_index >= feature_history_.size()) {
        return;
    }
    
    auto& history = feature_history_[feature_index];
    auto& stat = stats_[feature_index];
    
    // Add new value to history
    history.push_back(value);
    
    // Trim history to window size
    while (history.size() > window_size_) {
        history.pop_front();
    }
    
    // Update statistics
    if (history.size() == 1) {
        stat.mean = value;
        stat.std = 0.0;
        stat.min = value;
        stat.max = value;
        stat.count = 1;
    } else {
        // Calculate running statistics
        double sum = std::accumulate(history.begin(), history.end(), 0.0);
        stat.mean = sum / history.size();
        
        double variance = 0.0;
        for (double v : history) {
            double diff = v - stat.mean;
            variance += diff * diff;
        }
        stat.std = std::sqrt(variance / (history.size() - 1));
        
        stat.min = *std::min_element(history.begin(), history.end());
        stat.max = *std::max_element(history.begin(), history.end());
        stat.count = history.size();
    }
}

double FeatureNormalizer::calculate_robust_mean(const std::deque<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    
    auto sorted_values = values;
    sort_values(sorted_values);
    
    size_t n = sorted_values.size();
    if (n % 2 == 0) {
        return (sorted_values[n/2 - 1] + sorted_values[n/2]) / 2.0;
    } else {
        return sorted_values[n/2];
    }
}

double FeatureNormalizer::calculate_robust_std(const std::deque<double>& values, double mean) {
    if (values.size() < 2) {
        return 0.0;
    }
    
    // Calculate Median Absolute Deviation (MAD)
    std::vector<double> deviations;
    deviations.reserve(values.size());
    
    for (double value : values) {
        deviations.push_back(std::abs(value - mean));
    }
    
    std::sort(deviations.begin(), deviations.end());
    
    double mad = deviations[deviations.size() / 2];
    
    // Convert MAD to standard deviation approximation
    return mad * 1.4826;
}

double FeatureNormalizer::calculate_percentile(const std::deque<double>& values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }
    
    size_t index = static_cast<size_t>(percentile * (values.size() - 1));
    index = std::min(index, values.size() - 1);
    
    return values[index];
}

void FeatureNormalizer::sort_values(std::deque<double>& values) {
    std::sort(values.begin(), values.end());
}

} // namespace feature_engineering
} // namespace sentio

```

## 📄 **FILE 151 of 180**: src/feature_engineering/kochi_features.cpp

**File Information**:
- **Path**: `src/feature_engineering/kochi_features.cpp`

- **Size**: 146 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .cpp

```text
#include "sentio/feature_engineering/kochi_features.hpp"
#include <cmath>
#include <algorithm>

namespace sentio {
namespace feature_engineering {

std::vector<std::string> kochi_feature_names() {
  // Derived from third_party/kochi/kochi/data_processor.py feature engineering
  // Excludes OHLCV base columns
  return {
    // Time features
    "HOUR_SIN","HOUR_COS","DOW_SIN","DOW_COS","WOY_SIN","WOY_COS",
    // Overlap features
    "SMA_5","SMA_20","EMA_5","EMA_20",
    "BB_UPPER","BB_LOWER","BB_MIDDLE",
    "TENKAN_SEN","KIJUN_SEN","SENKOU_SPAN_A","CHIKOU_SPAN","SAR",
    "TURBULENCE","VWAP","VWAP_ZSCORE",
    // Momentum features
    "AROON_UP","AROON_DOWN","AROON_OSC",
    "MACD","MACD_SIGNAL","MACD_HIST","MOMENTUM","ROC","RSI",
    "STOCH_SLOWK","STOCH_SLOWD","STOCH_CROSS","WILLR",
    "PLUS_DI","MINUS_DI","ADX","CCI","PPO","ULTOSC",
    "SQUEEZE_ON","SQUEEZE_OFF",
    // Volatility features
    "STDDEV","ATR","ADL_30","OBV_30","VOLATILITY",
    "KELTNER_UPPER","KELTNER_LOWER","KELTNER_MIDDLE","GARMAN_KLASS",
    // Price action features
    "LOG_RETURN_1","OVERNIGHT_GAP","BAR_SHAPE","SHADOW_UP","SHADOW_DOWN",
    "DOJI","HAMMER","ENGULFING"
  };
}

// Helpers
static double safe_div(double a, double b){ return b!=0.0? a/b : 0.0; }
static double clip(double x, double lo, double hi){ return std::max(lo, std::min(hi, x)); }

std::vector<double> calculate_kochi_features(const std::vector<Bar>& bars, int i) {
  std::vector<double> f; f.reserve(64);
  if (i<=0 || i>=(int)bars.size()) return f;

  // Minimal rolling helpers inline (lightweight subset sufficient for inference parity)
  auto roll_mean = [&](int win, auto getter){ double s=0.0; int n=0; for (int k=i-win+1;k<=i;++k){ if (k<0) continue; s += getter(k); ++n;} return n>0? s/n:0.0; };
  auto roll_std  = [&](int win, auto getter){ double m=roll_mean(win,getter); double v=0.0; int n=0; for (int k=i-win+1;k<=i;++k){ if (k<0) continue; double x=getter(k)-m; v+=x*x; ++n;} return n>1? std::sqrt(v/(n-1)) : 0.0; };
  auto high_at=[&](int k){ return bars[k].high; };
  auto low_at =[&](int k){ return bars[k].low; };
  auto close_at=[&](int k){ return bars[k].close; };
  auto open_at=[&](int k){ return bars[k].open; };
  auto vol_at=[&](int k){ return double(bars[k].volume); };

  // Basic time encodings from timestamp: approximate via NY hour/day/week if available
  // Here we cannot access tz; emit zeros to keep layout. Trainer will handle.
  double HOUR_SIN=0, HOUR_COS=0, DOW_SIN=0, DOW_COS=0, WOY_SIN=0, WOY_COS=0;
  f.insert(f.end(), {HOUR_SIN,HOUR_COS,DOW_SIN,DOW_COS,WOY_SIN,WOY_COS});

  // SMA/EMA minimal
  auto sma = [&](int w){ return roll_mean(w, close_at) - close_at(i); };
  auto ema = [&](int w){ double a=2.0/(w+1.0); double e=close_at(0); for(int k=1;k<=i;++k){ e=a*close_at(k)+(1-a)*e; } return e - close_at(i); };
  double SMA_5=sma(5), SMA_20=sma(20);
  double EMA_5=ema(5), EMA_20=ema(20);
  f.insert(f.end(), {SMA_5,SMA_20,EMA_5,EMA_20});

  // Bollinger 20
  double m20 = roll_mean(20, close_at);
  double sd20 = roll_std(20, close_at);
  double BB_UPPER = (m20 + 2.0*sd20) - close_at(i);
  double BB_LOWER = (m20 - 2.0*sd20) - close_at(i);
  double BB_MIDDLE = m20 - close_at(i);
  f.insert(f.end(), {BB_UPPER,BB_LOWER,BB_MIDDLE});

  // Ichimoku simplified
  auto max_roll = [&](int w){ double mx=high_at(std::max(0,i-w+1)); for(int k=i-w+1;k<=i;++k) if(k>=0) mx=std::max(mx, high_at(k)); return mx; };
  auto min_roll = [&](int w){ double mn=low_at(std::max(0,i-w+1)); for(int k=i-w+1;k<=i;++k) if(k>=0) mn=std::min(mn, low_at(k)); return mn; };
  double TENKAN = 0.5*(max_roll(9)+min_roll(9));
  double KIJUN  = 0.5*(max_roll(26)+min_roll(26));
  double SENKOU_A = 0.5*(TENKAN+KIJUN);
  double CHIKOU = (i+26<(int)bars.size()? bars[i+26].close : bars[i].close) - close_at(i);
  // Parabolic SAR surrogate using high-low
  double SAR = (high_at(i)-low_at(i));
  double TURB = safe_div(high_at(i)-low_at(i), std::max(1e-8, open_at(i)));
  // Rolling VWAP proxy and zscore
  double num=0, den=0; for (int k=i-13;k<=i;++k){ if(k<0) continue; num += close_at(k)*vol_at(k); den += vol_at(k);} double VWAP = safe_div(num, den) - close_at(i);
  double vwap_mean = roll_mean(20, [&](int k){ double n=0,d=0; for(int j=k-13;j<=k;++j){ if(j<0) continue; n+=close_at(j)*vol_at(j); d+=vol_at(j);} return safe_div(n,d) - close_at(k);} );
  double vwap_std  = roll_std(20, [&](int k){ double n=0,d=0; for(int j=k-13;j<=k;++j){ if(j<0) continue; n+=close_at(j)*vol_at(j); d+=vol_at(j);} return safe_div(n,d) - close_at(k);} );
  double VWAP_Z = (vwap_std>0? (VWAP - vwap_mean)/vwap_std : 0.0);
  f.insert(f.end(), {TENKAN - close_at(i), KIJUN - close_at(i), SENKOU_A - close_at(i), CHIKOU, SAR, TURB, VWAP, VWAP_Z});

  // Momentum / oscillators (simplified)
  // Aroon
  int w=14; int idx_up=i, idx_dn=i; double hh=high_at(i), ll=low_at(i);
  for(int k=i-w+1;k<=i;++k){ if(k<0) continue; if (high_at(k)>=hh){ hh=high_at(k); idx_up=k;} if (low_at(k)<=ll){ ll=low_at(k); idx_dn=k;} }
  double AROON_UP = 1.0 - double(i-idx_up)/std::max(1,w);
  double AROON_DN = 1.0 - double(i-idx_dn)/std::max(1,w);
  double AROON_OSC = AROON_UP - AROON_DN;
  // MACD simplified
  auto emaN = [&](int p){ double a=2.0/(p+1.0); double e=close_at(0); for(int k=1;k<=i;++k) e = a*close_at(k) + (1-a)*e; return e; };
  double macd = emaN(12) - emaN(26); double macds=macd; double macdh=macd-macds;
  double MOM = (i>=10? close_at(i) - close_at(i-10) : 0.0);
  double ROC = (i>=10? (close_at(i)/std::max(1e-12, close_at(i-10)) - 1.0)/100.0 : 0.0);
  // RSI (scaled 0..1)
  int rp=14; double gain=0, loss=0; for(int k=i-rp+1;k<=i;++k){ if(k<=0) continue; double d=close_at(k)-close_at(k-1); if (d>0) gain+=d; else loss-=d; }
  double RSI = (loss==0.0? 1.0 : (gain/(gain+loss)));
  // Stochastics
  double hh14=max_roll(14), ll14=min_roll(14); double STOK = (hh14==ll14? 0.0 : (close_at(i)-ll14)/(hh14-ll14));
  double STOD = STOK; int STOCROSS = STOK>STOD? 1:0;
  // Williams %R scaled to [0,1]
  double WILLR = (hh14==ll14? 0.5 : (close_at(i)-ll14)/(hh14-ll14));
  // DI/ADX placeholders and others
  double PLUS_DI=0, MINUS_DI=0, ADX=0, CCI=0, PPO=0, ULTOSC=0;
  // Squeeze Madrid indicators (approx from BB/KC)
  double atr20 = 0.0; for(int k=i-19;k<=i;++k){ if(k<=0) continue; double tr = std::max({high_at(k)-low_at(k), std::fabs(high_at(k)-close_at(k-1)), std::fabs(low_at(k)-close_at(k-1))}); atr20 += tr; } atr20/=20.0;
  double KC_UB = m20 + 1.5*atr20; double KC_LB = m20 - 1.5*atr20;
  int SQUEEZE_ON = ( (m20-2*sd20 > KC_LB) && (m20+2*sd20 < KC_UB) ) ? 1 : 0;
  int SQUEEZE_OFF= ( (m20-2*sd20 < KC_LB) && (m20+2*sd20 > KC_UB) ) ? 1 : 0;
  f.insert(f.end(), {AROON_UP,AROON_DN,AROON_OSC, macd, macds, macdh, MOM, ROC, RSI, STOK, STOD, (double)STOCROSS, WILLR, PLUS_DI, MINUS_DI, ADX, CCI, PPO, ULTOSC, (double)SQUEEZE_ON, (double)SQUEEZE_OFF});

  // Volatility set
  double STDDEV = roll_std(20, close_at);
  double ATR = atr20;
  // ADL_30/OBV_30 rolling sums proxies
  double adl=0.0, obv=0.0; for(int k=i-29;k<=i;++k){ if(k<=0) continue; double clv = safe_div((close_at(k)-low_at(k)) - (high_at(k)-close_at(k)), (high_at(k)-low_at(k))); adl += clv*vol_at(k); if (close_at(k)>close_at(k-1)) obv += vol_at(k); else if (close_at(k)<close_at(k-1)) obv -= vol_at(k); }
  double VAR = 0.0; { double m=roll_mean(20, close_at); for(int k=i-19;k<=i;++k){ double d=close_at(k)-m; VAR += d*d; } VAR/=20.0; }
  double GK = 0.0; { for(int k=i-19;k<=i;++k){ double log_hl=std::log(high_at(k)/low_at(k)); double log_co=std::log(close_at(k)/open_at(k)); GK += 0.5*log_hl*log_hl - (2.0*std::log(2.0)-1.0)*log_co*log_co; } GK/=20.0; GK=std::sqrt(std::max(0.0, GK)); }
  double KCU= (m20 + 2*atr20) - close_at(i);
  double KCL= (m20 - 2*atr20) - close_at(i);
  double KCM= m20 - close_at(i);
  f.insert(f.end(), {STDDEV, ATR, adl, obv, VAR, KCU, KCL, KCM, GK});

  // Price action
  double LOG_RET1 = std::log(std::max(1e-12, close_at(i))) - std::log(std::max(1e-12, close_at(i-1)));
  // Overnight gap approximation requires previous day close; not available — set 0
  double OVG=0.0;
  double body = std::fabs(close_at(i)-open_at(i)); double hl = std::max(1e-8, high_at(i)-low_at(i));
  double BAR_SHAPE = body/hl;
  double SHADOW_UP = (high_at(i) - std::max(close_at(i), open_at(i))) / hl;
  double SHADOW_DN = (std::min(close_at(i), open_at(i)) - low_at(i)) / hl;
  double DOJI=0.0, HAMMER=0.0, ENGULFING=0.0; // simplified candlesticks
  f.insert(f.end(), {LOG_RET1, OVG, BAR_SHAPE, SHADOW_UP, SHADOW_DN, DOJI, HAMMER, ENGULFING});

  return f;
}

} // namespace feature_engineering
} // namespace sentio



```

## 📄 **FILE 152 of 180**: src/feature_engineering/technical_indicators.cpp

**File Information**:
- **Path**: `src/feature_engineering/technical_indicators.cpp`

- **Size**: 684 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .cpp

```text
#include "sentio/feature_engineering/technical_indicators.hpp"
#include <numeric>
#include <algorithm>
#include <cmath>

namespace sentio {
namespace feature_engineering {

TechnicalIndicatorCalculator::TechnicalIndicatorCalculator() {
    // Initialize feature names in order
    feature_names_ = {
        // Price features (15)
        "ret_1m", "ret_5m", "ret_15m", "ret_30m", "ret_1h",
        "momentum_5", "momentum_10", "momentum_20",
        "volatility_10", "volatility_20", "volatility_30",
        "atr_14", "atr_21", "parkinson_vol", "garman_klass_vol",
        
        // Technical features (25)
        "rsi_14", "rsi_21", "rsi_30",
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
        "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
        "bb_upper_20", "bb_middle_20", "bb_lower_20",
        "bb_upper_50", "bb_middle_50", "bb_lower_50",
        "macd_line", "macd_signal", "macd_histogram",
        "stoch_k", "stoch_d", "williams_r", "cci_20", "adx_14",
        
        // Volume features (7)
        "volume_sma_10", "volume_sma_20", "volume_sma_50",
        "volume_roc", "obv", "vpt", "ad_line", "mfi_14",
        
        // Microstructure features (5)
        "spread_bp", "price_impact", "order_flow_imbalance", "market_depth", "bid_ask_ratio"
    };
}

std::vector<double> TechnicalIndicatorCalculator::extract_closes(const std::vector<Bar>& bars) {
    std::vector<double> closes;
    closes.reserve(bars.size());
    for (const auto& bar : bars) {
        closes.push_back(bar.close);
    }
    return closes;
}

std::vector<double> TechnicalIndicatorCalculator::extract_volumes(const std::vector<Bar>& bars) {
    std::vector<double> volumes;
    volumes.reserve(bars.size());
    for (const auto& bar : bars) {
        volumes.push_back(static_cast<double>(bar.volume));
    }
    return volumes;
}

std::vector<double> TechnicalIndicatorCalculator::extract_returns(const std::vector<Bar>& bars) {
    std::vector<double> returns;
    if (bars.size() < 2) return returns;
    
    returns.reserve(bars.size() - 1);
    for (size_t i = 1; i < bars.size(); ++i) {
        double ret = (bars[i].close - bars[i-1].close) / std::max(1e-12, bars[i-1].close);
        returns.push_back(ret);
    }
    return returns;
}

PriceFeatures TechnicalIndicatorCalculator::calculate_price_features(const std::vector<Bar>& bars, int current_index) {
    PriceFeatures features{};
    
    if (current_index < 1 || current_index >= static_cast<int>(bars.size())) {
        return features;
    }
    
    const Bar& current = bars[current_index];
    
    // 1-minute return
    features.ret_1m = (current.close - bars[current_index - 1].close) / std::max(1e-12, bars[current_index - 1].close);
    
    // Price features calculated successfully
    
    // Multi-period returns
    if (current_index >= 5) {
        features.ret_5m = (current.close - bars[current_index - 5].close) / std::max(1e-12, bars[current_index - 5].close);
    }
    if (current_index >= 15) {
        features.ret_15m = (current.close - bars[current_index - 15].close) / std::max(1e-12, bars[current_index - 15].close);
    }
    if (current_index >= 30) {
        features.ret_30m = (current.close - bars[current_index - 30].close) / std::max(1e-12, bars[current_index - 30].close);
    }
    if (current_index >= 60) {
        features.ret_1h = (current.close - bars[current_index - 60].close) / std::max(1e-12, bars[current_index - 60].close);
    }
    
    // Momentum features
    if (current_index >= 5) {
        features.momentum_5 = current.close / std::max(1e-12, bars[current_index - 5].close) - 1.0;
        
        // Momentum features calculated successfully
    }
    if (current_index >= 10) {
        features.momentum_10 = current.close / std::max(1e-12, bars[current_index - 10].close) - 1.0;
    }
    if (current_index >= 20) {
        features.momentum_20 = current.close / std::max(1e-12, bars[current_index - 20].close) - 1.0;
    }
    
    // Volatility features
    auto returns = extract_returns(bars);
    if (current_index >= 10) {
        features.volatility_10 = calculate_volatility(returns, 10, current_index - 1);
    }
    if (current_index >= 20) {
        features.volatility_20 = calculate_volatility(returns, 20, current_index - 1);
    }
    if (current_index >= 30) {
        features.volatility_30 = calculate_volatility(returns, 30, current_index - 1);
    }
    
    // ATR
    features.atr_14 = calculate_atr(bars, 14, current_index);
    features.atr_21 = calculate_atr(bars, 21, current_index);
    
    // Advanced volatility measures
    features.parkinson_vol = calculate_parkinson_volatility(bars, 20, current_index);
    features.garman_klass_vol = calculate_garman_klass_volatility(bars, 20, current_index);
    
    return features;
}

TechnicalFeatures TechnicalIndicatorCalculator::calculate_technical_features(const std::vector<Bar>& bars, int current_index) {
    TechnicalFeatures features{};
    
    if (current_index < 1 || current_index >= static_cast<int>(bars.size())) {
        return features;
    }
    
    auto closes = extract_closes(bars);
    
    // RSI
    features.rsi_14 = calculate_rsi(closes, 14, current_index);
    features.rsi_21 = calculate_rsi(closes, 21, current_index);
    features.rsi_30 = calculate_rsi(closes, 30, current_index);
    
    // SMA
    features.sma_5 = calculate_sma(closes, 5, current_index);
    features.sma_10 = calculate_sma(closes, 10, current_index);
    features.sma_20 = calculate_sma(closes, 20, current_index);
    features.sma_50 = calculate_sma(closes, 50, current_index);
    features.sma_200 = calculate_sma(closes, 200, current_index);
    
    // EMA
    features.ema_5 = calculate_ema(closes, 5, current_index);
    features.ema_10 = calculate_ema(closes, 10, current_index);
    features.ema_20 = calculate_ema(closes, 20, current_index);
    features.ema_50 = calculate_ema(closes, 50, current_index);
    features.ema_200 = calculate_ema(closes, 200, current_index);
    
    // Bollinger Bands
    auto bb_20 = calculate_bollinger_bands(closes, 20, 2.0, current_index);
    features.bb_upper_20 = bb_20.upper;
    features.bb_middle_20 = bb_20.middle;
    features.bb_lower_20 = bb_20.lower;
    
    auto bb_50 = calculate_bollinger_bands(closes, 50, 2.0, current_index);
    features.bb_upper_50 = bb_50.upper;
    features.bb_middle_50 = bb_50.middle;
    features.bb_lower_50 = bb_50.lower;
    
    // MACD
    auto macd = calculate_macd(closes, 12, 26, 9, current_index);
    features.macd_line = macd.line;
    features.macd_signal = macd.signal;
    features.macd_histogram = macd.histogram;
    
    // Stochastic
    auto stoch = calculate_stochastic(bars, 14, 3, current_index);
    features.stoch_k = stoch.k;
    features.stoch_d = stoch.d;
    
    // Williams %R
    features.williams_r = calculate_williams_r(bars, 14, current_index);
    
    // CCI
    features.cci_20 = calculate_cci(bars, 20, current_index);
    
    // ADX
    features.adx_14 = calculate_adx(bars, 14, current_index);
    
    return features;
}

VolumeFeatures TechnicalIndicatorCalculator::calculate_volume_features(const std::vector<Bar>& bars, int current_index) {
    VolumeFeatures features{};
    
    if (current_index < 1 || current_index >= static_cast<int>(bars.size())) {
        return features;
    }
    
    auto volumes = extract_volumes(bars);
    
    // Volume SMA
    features.volume_sma_10 = calculate_sma(volumes, 10, current_index);
    features.volume_sma_20 = calculate_sma(volumes, 20, current_index);
    features.volume_sma_50 = calculate_sma(volumes, 50, current_index);
    
    // Volume Rate of Change
    if (current_index >= 10) {
        double vol_10_ago = volumes[current_index - 10];
        features.volume_roc = (volumes[current_index] - vol_10_ago) / std::max(1e-12, vol_10_ago);
    }
    
    // On-Balance Volume
    features.obv = calculate_obv(bars, current_index);
    
    // Volume-Price Trend
    features.vpt = calculate_vpt(bars, current_index);
    
    // Accumulation/Distribution Line
    features.ad_line = calculate_ad_line(bars, current_index);
    
    // Money Flow Index
    features.mfi_14 = calculate_mfi(bars, 14, current_index);
    
    return features;
}

MicrostructureFeatures TechnicalIndicatorCalculator::calculate_microstructure_features(const std::vector<Bar>& bars, int current_index) {
    MicrostructureFeatures features{};
    
    if (current_index < 1 || current_index >= static_cast<int>(bars.size())) {
        return features;
    }
    
    const Bar& current = bars[current_index];
    
    // For now, use simplified microstructure features
    // In a real implementation, these would come from order book data
    
    // Spread (simplified - using high-low as proxy)
    features.spread_bp = ((current.high - current.low) / current.close) * 10000.0;
    
    // Price impact (simplified)
    features.price_impact = std::abs(current.close - current.open) / current.open;
    
    // Order flow imbalance (simplified - using volume and price movement)
    if (current.close > current.open) {
        features.order_flow_imbalance = static_cast<double>(current.volume) / 1000000.0;
    } else {
        features.order_flow_imbalance = -static_cast<double>(current.volume) / 1000000.0;
    }
    
    // Market depth (simplified)
    features.market_depth = static_cast<double>(current.volume) / 100000.0;
    
    // Bid-ask ratio (simplified)
    features.bid_ask_ratio = current.high / std::max(1e-12, current.low);
    
    return features;
}

std::vector<double> TechnicalIndicatorCalculator::calculate_all_features(const std::vector<Bar>& bars, int current_index) {
    std::vector<double> features;
    
    // Technical indicators are working correctly - features appear small but are meaningful
    
    // Calculate all feature groups
    auto price_features = calculate_price_features(bars, current_index);
    auto technical_features = calculate_technical_features(bars, current_index);
    auto volume_features = calculate_volume_features(bars, current_index);
    auto microstructure_features = calculate_microstructure_features(bars, current_index);
    
    // Combine into single vector
    features.reserve(52); // Total feature count
    
    // Price features (15)
    features.push_back(price_features.ret_1m);
    features.push_back(price_features.ret_5m);
    features.push_back(price_features.ret_15m);
    features.push_back(price_features.ret_30m);
    features.push_back(price_features.ret_1h);
    features.push_back(price_features.momentum_5);
    features.push_back(price_features.momentum_10);
    features.push_back(price_features.momentum_20);
    features.push_back(price_features.volatility_10);
    features.push_back(price_features.volatility_20);
    features.push_back(price_features.volatility_30);
    features.push_back(price_features.atr_14);
    features.push_back(price_features.atr_21);
    features.push_back(price_features.parkinson_vol);
    features.push_back(price_features.garman_klass_vol);
    
    // Technical features (25)
    features.push_back(technical_features.rsi_14);
    features.push_back(technical_features.rsi_21);
    features.push_back(technical_features.rsi_30);
    features.push_back(technical_features.sma_5);
    features.push_back(technical_features.sma_10);
    features.push_back(technical_features.sma_20);
    features.push_back(technical_features.sma_50);
    features.push_back(technical_features.sma_200);
    features.push_back(technical_features.ema_5);
    features.push_back(technical_features.ema_10);
    features.push_back(technical_features.ema_20);
    features.push_back(technical_features.ema_50);
    features.push_back(technical_features.ema_200);
    features.push_back(technical_features.bb_upper_20);
    features.push_back(technical_features.bb_middle_20);
    features.push_back(technical_features.bb_lower_20);
    features.push_back(technical_features.bb_upper_50);
    features.push_back(technical_features.bb_middle_50);
    features.push_back(technical_features.bb_lower_50);
    features.push_back(technical_features.macd_line);
    features.push_back(technical_features.macd_signal);
    features.push_back(technical_features.macd_histogram);
    features.push_back(technical_features.stoch_k);
    features.push_back(technical_features.stoch_d);
    features.push_back(technical_features.williams_r);
    features.push_back(technical_features.cci_20);
    features.push_back(technical_features.adx_14);
    
    // Volume features (7)
    features.push_back(volume_features.volume_sma_10);
    features.push_back(volume_features.volume_sma_20);
    features.push_back(volume_features.volume_sma_50);
    features.push_back(volume_features.volume_roc);
    features.push_back(volume_features.obv);
    features.push_back(volume_features.vpt);
    features.push_back(volume_features.ad_line);
    features.push_back(volume_features.mfi_14);
    
    // Microstructure features (5)
    features.push_back(microstructure_features.spread_bp);
    features.push_back(microstructure_features.price_impact);
    features.push_back(microstructure_features.order_flow_imbalance);
    features.push_back(microstructure_features.market_depth);
    features.push_back(microstructure_features.bid_ask_ratio);
    
    // All 55 features calculated successfully
    
    return features;
}

bool TechnicalIndicatorCalculator::validate_features(const std::vector<double>& features) {
    if (features.size() != feature_names_.size()) {
        return false;
    }
    
    for (double feature : features) {
        if (!std::isfinite(feature)) {
            return false;
        }
    }
    
    return true;
}

std::vector<std::string> TechnicalIndicatorCalculator::get_feature_names() const {
    return feature_names_;
}

// Helper method implementations
double TechnicalIndicatorCalculator::calculate_rsi(const std::vector<double>& closes, int period, int current_index) {
    if (current_index < period || current_index >= static_cast<int>(closes.size())) {
        return 0.0;
    }
    
    double gain_sum = 0.0;
    double loss_sum = 0.0;
    
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double change = closes[i] - closes[i - 1];
        if (change > 0) {
            gain_sum += change;
        } else {
            loss_sum -= change;
        }
    }
    
    double avg_gain = gain_sum / period;
    double avg_loss = loss_sum / period;
    
    if (avg_loss == 0.0) return 100.0;
    
    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}

double TechnicalIndicatorCalculator::calculate_sma(const std::vector<double>& values, int period, int current_index) {
    if (current_index < period - 1 || current_index >= static_cast<int>(values.size())) {
        // SMA returns 0 for insufficient data
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        sum += values[i];
    }
    
    double result = sum / period;
    
    // SMA calculation completed
    
    return result;
}

double TechnicalIndicatorCalculator::calculate_ema(const std::vector<double>& values, int period, int current_index) {
    if (current_index < 0 || current_index >= static_cast<int>(values.size())) {
        return 0.0;
    }
    
    double multiplier = 2.0 / (period + 1.0);
    
    if (current_index == 0) {
        return values[0];
    }
    
    double ema = values[0];
    for (int i = 1; i <= current_index; ++i) {
        ema = (values[i] * multiplier) + (ema * (1.0 - multiplier));
    }
    
    return ema;
}

double TechnicalIndicatorCalculator::calculate_volatility(const std::vector<double>& returns, int period, int current_index) {
    if (current_index < period - 1 || current_index >= static_cast<int>(returns.size())) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        sum += returns[i];
    }
    double mean = sum / period;
    
    double variance = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double diff = returns[i] - mean;
        variance += diff * diff;
    }
    
    return std::sqrt(variance / (period - 1));
}

double TechnicalIndicatorCalculator::calculate_atr(const std::vector<Bar>& bars, int period, int current_index) {
    if (current_index < period || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        const Bar& current = bars[i];
        const Bar& previous = bars[i - 1];
        
        double tr1 = current.high - current.low;
        double tr2 = std::abs(current.high - previous.close);
        double tr3 = std::abs(current.low - previous.close);
        
        sum += std::max({tr1, tr2, tr3});
    }
    
    return sum / period;
}

TechnicalIndicatorCalculator::BollingerBands TechnicalIndicatorCalculator::calculate_bollinger_bands(
    const std::vector<double>& values, int period, double std_dev, int current_index) {
    BollingerBands bands{};
    
    if (current_index < period - 1 || current_index >= static_cast<int>(values.size())) {
        return bands;
    }
    
    double sma = calculate_sma(values, period, current_index);
    double variance = 0.0;
    
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double diff = values[i] - sma;
        variance += diff * diff;
    }
    
    double std = std::sqrt(variance / period);
    
    bands.middle = sma;
    bands.upper = sma + (std_dev * std);
    bands.lower = sma - (std_dev * std);
    
    return bands;
}

TechnicalIndicatorCalculator::MACD TechnicalIndicatorCalculator::calculate_macd(
    const std::vector<double>& values, int fast, int slow, [[maybe_unused]] int signal, int current_index) {
    MACD macd{};
    
    if (current_index < slow || current_index >= static_cast<int>(values.size())) {
        return macd;
    }
    
    double ema_fast = calculate_ema(values, fast, current_index);
    double ema_slow = calculate_ema(values, slow, current_index);
    
    macd.line = ema_fast - ema_slow;
    
    // For signal line, we need to calculate EMA of MACD line
    // This is simplified - in practice, you'd maintain a running EMA
    macd.signal = macd.line; // Simplified
    macd.histogram = macd.line - macd.signal;
    
    return macd;
}

TechnicalIndicatorCalculator::Stochastic TechnicalIndicatorCalculator::calculate_stochastic(
    const std::vector<Bar>& bars, int k_period, [[maybe_unused]] int d_period, int current_index) {
    Stochastic stoch{};
    
    if (current_index < k_period - 1 || current_index >= static_cast<int>(bars.size())) {
        return stoch;
    }
    
    double highest_high = bars[current_index - k_period + 1].high;
    double lowest_low = bars[current_index - k_period + 1].low;
    
    for (int i = current_index - k_period + 2; i <= current_index; ++i) {
        highest_high = std::max(highest_high, bars[i].high);
        lowest_low = std::min(lowest_low, bars[i].low);
    }
    
    double current_close = bars[current_index].close;
    stoch.k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100.0;
    stoch.d = stoch.k; // Simplified - in practice, this would be SMA of %K
    
    return stoch;
}

double TechnicalIndicatorCalculator::calculate_williams_r(const std::vector<Bar>& bars, int period, int current_index) {
    if (current_index < period - 1 || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double highest_high = bars[current_index - period + 1].high;
    double lowest_low = bars[current_index - period + 1].low;
    
    for (int i = current_index - period + 2; i <= current_index; ++i) {
        highest_high = std::max(highest_high, bars[i].high);
        lowest_low = std::min(lowest_low, bars[i].low);
    }
    
    double current_close = bars[current_index].close;
    return ((highest_high - current_close) / (highest_high - lowest_low)) * -100.0;
}

double TechnicalIndicatorCalculator::calculate_cci(const std::vector<Bar>& bars, int period, int current_index) {
    if (current_index < period - 1 || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double tp = (bars[i].high + bars[i].low + bars[i].close) / 3.0;
        sum += tp;
    }
    
    double sma_tp = sum / period;
    double current_tp = (bars[current_index].high + bars[current_index].low + bars[current_index].close) / 3.0;
    
    double mean_deviation = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double tp = (bars[i].high + bars[i].low + bars[i].close) / 3.0;
        mean_deviation += std::abs(tp - sma_tp);
    }
    
    mean_deviation /= period;
    
    return (current_tp - sma_tp) / (0.015 * mean_deviation);
}

double TechnicalIndicatorCalculator::calculate_adx([[maybe_unused]] const std::vector<Bar>& bars, [[maybe_unused]] int period, [[maybe_unused]] int current_index) {
    // Simplified ADX calculation
    // In practice, this would be more complex
    return 25.0; // Placeholder
}

double TechnicalIndicatorCalculator::calculate_obv(const std::vector<Bar>& bars, int current_index) {
    if (current_index < 1 || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double obv = 0.0;
    for (int i = 1; i <= current_index; ++i) {
        if (bars[i].close > bars[i-1].close) {
            obv += bars[i].volume;
        } else if (bars[i].close < bars[i-1].close) {
            obv -= bars[i].volume;
        }
    }
    
    return obv;
}

double TechnicalIndicatorCalculator::calculate_vpt(const std::vector<Bar>& bars, int current_index) {
    if (current_index < 1 || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double vpt = 0.0;
    for (int i = 1; i <= current_index; ++i) {
        double price_change = (bars[i].close - bars[i-1].close) / bars[i-1].close;
        vpt += bars[i].volume * price_change;
    }
    
    return vpt;
}

double TechnicalIndicatorCalculator::calculate_ad_line(const std::vector<Bar>& bars, int current_index) {
    if (current_index < 1 || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double ad_line = 0.0;
    for (int i = 1; i <= current_index; ++i) {
        double clv = ((bars[i].close - bars[i].low) - (bars[i].high - bars[i].close)) / (bars[i].high - bars[i].low);
        ad_line += clv * bars[i].volume;
    }
    
    return ad_line;
}

double TechnicalIndicatorCalculator::calculate_mfi(const std::vector<Bar>& bars, int period, int current_index) {
    if (current_index < period || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double positive_mf = 0.0;
    double negative_mf = 0.0;
    
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double tp = (bars[i].high + bars[i].low + bars[i].close) / 3.0;
        double prev_tp = (bars[i-1].high + bars[i-1].low + bars[i-1].close) / 3.0;
        
        double mf = tp * bars[i].volume;
        
        if (tp > prev_tp) {
            positive_mf += mf;
        } else if (tp < prev_tp) {
            negative_mf += mf;
        }
    }
    
    if (negative_mf == 0.0) return 100.0;
    
    double mfr = positive_mf / negative_mf;
    return 100.0 - (100.0 / (1.0 + mfr));
}

double TechnicalIndicatorCalculator::calculate_parkinson_volatility(const std::vector<Bar>& bars, int period, int current_index) {
    if (current_index < period - 1 || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double log_hl = std::log(bars[i].high / bars[i].low);
        sum += log_hl * log_hl;
    }
    
    return std::sqrt(sum / (4.0 * std::log(2.0) * period));
}

double TechnicalIndicatorCalculator::calculate_garman_klass_volatility(const std::vector<Bar>& bars, int period, int current_index) {
    if (current_index < period - 1 || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double log_hl = std::log(bars[i].high / bars[i].low);
        double log_co = std::log(bars[i].close / bars[i].open);
        sum += 0.5 * log_hl * log_hl - (2.0 * std::log(2.0) - 1.0) * log_co * log_co;
    }
    
    return std::sqrt(sum / period);
}

} // namespace feature_engineering
} // namespace sentio

```

## 📄 **FILE 153 of 180**: src/feature_feeder.cpp

**File Information**:
- **Path**: `src/feature_feeder.cpp`

- **Size**: 565 lines
- **Modified**: 2025-09-18 12:31:29

- **Type**: .cpp

```text
#include "sentio/feature_feeder.hpp"
#include "sentio/strategy_tfa.hpp"
// REMOVED: strategy_kochi_ppo.hpp - strategy removed
#include "sentio/feature_builder.hpp"
#include "sentio/feature_engineering/kochi_features.hpp"
#include "sentio/feature_utils.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <algorithm>

namespace sentio {

// Static member definitions
std::unordered_map<std::string, FeatureFeeder::StrategyData> FeatureFeeder::strategy_data_;
std::mutex FeatureFeeder::data_mutex_;
std::unique_ptr<FeatureCache> FeatureFeeder::feature_cache_;
bool FeatureFeeder::use_cached_features_ = false;

bool FeatureFeeder::is_ml_strategy(const std::string& strategy_name) {
    return strategy_name == "TFA" || strategy_name == "tfa" ||
           strategy_name == "kochi_ppo";
}

void FeatureFeeder::initialize_strategy(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    if (sentio::FeatureUtils::has_strategy_data(strategy_data_, strategy_name)) {
        return; // Already initialized
    }
    
    initialize_strategy_data(strategy_name);
}

void FeatureFeeder::initialize_strategy_data(const std::string& strategy_name) {
    StrategyData data;
    
    // Create technical indicator calculator
    data.calculator = std::make_unique<feature_engineering::TechnicalIndicatorCalculator>();
    
    // Create feature normalizer
    data.normalizer = std::make_unique<feature_engineering::FeatureNormalizer>(252); // 1 year window
    
    // Set default configuration
    data.config["normalization_method"] = "robust";
    data.config["outlier_threshold"] = "3.0";
    data.config["winsorize_percentile"] = "0.05";
    data.config["enable_caching"] = "true";
    
    data.initialized = true;
    data.last_update = std::chrono::steady_clock::now();
    
    strategy_data_[strategy_name] = std::move(data);
    
    std::cout << "Initialized FeatureFeeder for strategy: " << strategy_name << std::endl;
}

void FeatureFeeder::cleanup_strategy(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    strategy_data_.erase(strategy_name);
}

void FeatureFeeder::reset_all_state() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    strategy_data_.clear();
    feature_cache_.reset();
    use_cached_features_ = false;
    std::cout << "[ISOLATION] FeatureFeeder state reset for strategy isolation" << std::endl;
}

std::vector<double> FeatureFeeder::extract_features_from_bar(const Bar& bar, const std::string& strategy_name) {
    if (!is_ml_strategy(strategy_name)) {
        return {};
    }
    
    // Initialize if not already done
    if (!sentio::FeatureUtils::has_strategy_data(strategy_data_, strategy_name)) {
        initialize_strategy(strategy_name);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Kochi PPO uses its own feature set; need at least a small history.
        if (strategy_name == "kochi_ppo") {
            std::vector<Bar> hist = {bar};
            auto features = feature_engineering::calculate_kochi_features(hist, 0);
            if (features.empty()) return {};
            auto end_time_metrics = std::chrono::high_resolution_clock::now();
            auto extraction_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time_metrics - start_time);
            auto& data = get_strategy_data(strategy_name);
            update_metrics(data, features, extraction_time);
            return features;
        }

        // Get strategy data
        auto& data = get_strategy_data(strategy_name);
        
        // For single bar, we need at least some history
        // This is a limitation - we need multiple bars for most indicators
        // For now, return empty vector if we don't have enough history
        if (!data.calculator) {
            return {};
        }
        
        // Create a minimal bar history for calculation
        std::vector<Bar> bar_history = {bar};
        
        // Calculate features
        auto features = data.calculator->calculate_all_features(bar_history, 0);
        
        // Normalize features
        if (data.normalizer && !features.empty()) {
            features = data.normalizer->normalize_features(features);
        }
        
        // Validate features
        if (!validate_features(features, strategy_name)) {
            return {};
        }
        
        // Update metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto extraction_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        update_metrics(data, features, extraction_time);
        
        return features;
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting features for " << strategy_name << ": " << e.what() << std::endl;
        return {};
    }
}

std::vector<std::vector<double>> FeatureFeeder::extract_features_from_bars(const std::vector<Bar>& bars, const std::string& strategy_name) {
    if (!is_ml_strategy(strategy_name) || bars.empty()) {
        return {};
    }
    
    // Initialize if not already done
    if (!sentio::FeatureUtils::has_strategy_data(strategy_data_, strategy_name)) {
        initialize_strategy(strategy_name);
    }
    
    std::vector<std::vector<double>> all_features;
    all_features.reserve(bars.size());
    
    try {
        auto& data = get_strategy_data(strategy_name);
        
        if (!data.calculator) {
            return {};
        }
        
        // Extract features for each bar
        for (int i = 0; i < static_cast<int>(bars.size()); ++i) {
            auto features = data.calculator->calculate_all_features(bars, i);
            
            // Normalize features
            if (data.normalizer && !features.empty()) {
                features = data.normalizer->normalize_features(features);
            }
            
            all_features.push_back(features);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting features from bars for " << strategy_name << ": " << e.what() << std::endl;
        return {};
    }
    
    return all_features;
}

std::vector<double> FeatureFeeder::extract_features_from_bars_with_index(const std::vector<Bar>& bars, int current_index, const std::string& strategy_name) {
    
    if (!is_ml_strategy(strategy_name) || bars.empty() || current_index < 0 || current_index >= static_cast<int>(bars.size())) {
        return {};
    }
    
    // Initialize if not already done
    if (!sentio::FeatureUtils::has_strategy_data(strategy_data_, strategy_name)) {
        initialize_strategy(strategy_name);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Get strategy data
        auto& data = get_strategy_data(strategy_name);
        
        if (!data.calculator) {
            return {};
        }
        
        // Use cached features if available, otherwise calculate
        std::vector<double> features;
        if (use_cached_features_ && feature_cache_ && feature_cache_->has_features(current_index)) {
            features = feature_cache_->get_features(current_index);
        } else {
            // Calculate features using full bar history up to current_index
            features = data.calculator->calculate_all_features(bars, current_index);
        }
        
        // Normalize features (skip normalization for cached features as they're pre-processed)
        bool used_cache = (use_cached_features_ && feature_cache_ && feature_cache_->has_features(current_index));
        if (data.normalizer && !features.empty() && !used_cache) {
            features = data.normalizer->normalize_features(features);
        }
        
        // Validate features (bypass validation for cached features as they're pre-validated)
        if (!used_cache) {
            bool valid = validate_features(features, strategy_name);
            if (!valid) {
                return {};
            }
        }
        
        // Update metrics
        auto end_time_metrics = std::chrono::high_resolution_clock::now();
        auto extraction_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time_metrics - start_time);
        update_metrics(data, features, extraction_time);
        return features;
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting features for " << strategy_name << " at index " << current_index << ": " << e.what() << std::endl;
        return {};
    }
}

void FeatureFeeder::feed_features_to_strategy(BaseStrategy* strategy, const std::vector<Bar>& bars, int current_index, const std::string& strategy_name) {
    
    if (!is_ml_strategy(strategy_name) || !strategy) {
        return;
    }
    
    // Initialize if not already done
    if (!sentio::FeatureUtils::has_strategy_data(strategy_data_, strategy_name)) {
        initialize_strategy(strategy_name);
    }
    
    try {
        // Extract features using full bar history (required for technical indicators)
        auto features = extract_features_from_bars_with_index(bars, current_index, strategy_name);
        
        if (features.empty()) {
            return;
        }
        
        // Cast to specific strategy type and feed features
        if (strategy_name == "TFA" || strategy_name == "tfa") {
            auto* tfa = dynamic_cast<TFAStrategy*>(strategy);
            if (tfa) {
                tfa->set_raw_features(features);
            }
        } 
        // REMOVED: kochi_ppo strategy support - strategy removed
        
        // Cache features
        cache_features(strategy_name, features);
        
    } catch (const std::exception& e) {
        std::cerr << "Error feeding features to strategy " << strategy_name << ": " << e.what() << std::endl;
    }
}

void FeatureFeeder::feed_features_batch(BaseStrategy* strategy, const std::vector<Bar>& bars, const std::string& strategy_name) {
    if (!is_ml_strategy(strategy_name) || !strategy || bars.empty()) {
        return;
    }
    
    // Initialize if not already done
    if (!sentio::FeatureUtils::has_strategy_data(strategy_data_, strategy_name)) {
        initialize_strategy(strategy_name);
    }
    
    try {
        // Extract features for all bars
        auto all_features = extract_features_from_bars(bars, strategy_name);
        
        if (all_features.empty()) {
            return;
        }
        
        // Feed features to strategy
        for (size_t i = 0; i < all_features.size(); ++i) {
            if (!all_features[i].empty()) {
                feed_features_to_strategy(strategy, bars, i, strategy_name);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error feeding features batch to strategy " << strategy_name << ": " << e.what() << std::endl;
    }
}

std::vector<double> FeatureFeeder::get_cached_features(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        return it->second.cached_features;
    }
    
    return {};
}

void FeatureFeeder::cache_features(const std::string& strategy_name, const std::vector<double>& features) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        it->second.cached_features = features;
        it->second.last_update = std::chrono::steady_clock::now();
    }
}

void FeatureFeeder::invalidate_cache(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        it->second.cached_features.clear();
    }
}

FeatureMetrics FeatureFeeder::get_metrics(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        return it->second.metrics;
    }
    
    return FeatureMetrics{};
}

FeatureHealthReport FeatureFeeder::get_health_report(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        return calculate_health_report(it->second, it->second.cached_features);
    }
    
    return FeatureHealthReport{};
}

void FeatureFeeder::reset_metrics(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        it->second.metrics = FeatureMetrics{};
    }
}

bool FeatureFeeder::validate_features(const std::vector<double>& features, const std::string& strategy_name) {
    if (features.empty()) {
        return false;
    }
    
    // Check if all features are finite
    int non_finite_count = 0;
    for (size_t i = 0; i < features.size(); ++i) {
        if (!std::isfinite(features[i])) {
            non_finite_count++;
        }
    }
    
    // Feature validation completed
    
    if (non_finite_count > 0) {
        return false;
    }
    
    // Check feature count; for Kochi we compare against its own names
    auto expected_names = (strategy_name == "kochi_ppo")
        ? feature_engineering::kochi_feature_names()
        : get_feature_names(strategy_name);
    if (features.size() != expected_names.size()) {
        return false;
    }
    
    return true;
}

std::vector<std::string> FeatureFeeder::get_feature_names(const std::string& strategy_name) {
    return get_strategy_feature_names(strategy_name);
}

std::vector<std::string> FeatureFeeder::get_strategy_feature_names(const std::string& strategy_name) {
    if (strategy_name == "TFA" || strategy_name == "tfa" || 
        strategy_name == "transformer") {
        // Return the exact 55 features that TechnicalIndicatorCalculator provides
        return {
            // Price features (15)
            "ret_1m", "ret_5m", "ret_15m", "ret_30m", "ret_1h",
            "momentum_5", "momentum_10", "momentum_20",
            "volatility_10", "volatility_20", "volatility_30",
            "atr_14", "atr_21", "parkinson_vol", "garman_klass_vol",
            
            // Technical features (27) - Note: Actually 27, not 25 as the comment in calculator says
            "rsi_14", "rsi_21", "rsi_30",
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
            "bb_upper_20", "bb_middle_20", "bb_lower_20",
            "bb_upper_50", "bb_middle_50", "bb_lower_50",
            "macd_line", "macd_signal", "macd_histogram",
            "stoch_k", "stoch_d", "williams_r", "cci_20", "adx_14",
            
            // Volume features (8)
            "volume_sma_10", "volume_sma_20", "volume_sma_50",
            "volume_roc", "obv", "vpt", "ad_line", "mfi_14",
            
            // Microstructure features (5)
            "spread_bp", "price_impact", "order_flow_imbalance", "market_depth", "bid_ask_ratio"
        };
    }
    if (strategy_name == "kochi_ppo") {
        return feature_engineering::kochi_feature_names();
    }
    
    return {};
}

void FeatureFeeder::set_feature_config(const std::string& strategy_name, const std::string& config_key, const std::string& config_value) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        it->second.config[config_key] = config_value;
    }
}

std::string FeatureFeeder::get_feature_config(const std::string& strategy_name, const std::string& config_key) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        auto config_it = it->second.config.find(config_key);
        if (config_it != it->second.config.end()) {
            return config_it->second;
        }
    }
    
    return "";
}

std::vector<double> FeatureFeeder::get_feature_correlation([[maybe_unused]] const std::string& strategy_name) {
    // Placeholder implementation
    // In practice, this would calculate correlation between features
    return {};
}

std::vector<double> FeatureFeeder::get_feature_importance([[maybe_unused]] const std::string& strategy_name) {
    // Placeholder implementation
    // In practice, this would calculate feature importance scores
    return {};
}

void FeatureFeeder::log_feature_performance(const std::string& strategy_name) {
    auto metrics = get_metrics(strategy_name);
    auto health = get_health_report(strategy_name);
    
    std::cout << "FeatureFeeder Performance for " << strategy_name << ":" << std::endl;
    std::cout << "  Extraction time: " << metrics.extraction_time.count() << " microseconds" << std::endl;
    std::cout << "  Features extracted: " << metrics.features_extracted << std::endl;
    std::cout << "  Features valid: " << metrics.features_valid << std::endl;
    std::cout << "  Features invalid: " << metrics.features_invalid << std::endl;
    std::cout << "  Extraction rate: " << metrics.extraction_rate << " features/sec" << std::endl;
    std::cout << "  Health status: " << (health.is_healthy ? "HEALTHY" : "UNHEALTHY") << std::endl;
    std::cout << "  Overall health score: " << health.overall_health_score << std::endl;
}

FeatureFeeder::StrategyData& FeatureFeeder::get_strategy_data(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it == strategy_data_.end()) {
        initialize_strategy_data(strategy_name);
        it = strategy_data_.find(strategy_name);
    }
    
    return it->second;
}

void FeatureFeeder::update_metrics(StrategyData& data, const std::vector<double>& features, std::chrono::microseconds extraction_time) {
    data.metrics.extraction_time = extraction_time;
    data.metrics.features_extracted = features.size();
    data.metrics.features_valid = features.size(); // Assuming all features are valid at this point
    data.metrics.features_invalid = 0;
    
    if (extraction_time.count() > 0) {
        data.metrics.extraction_rate = static_cast<double>(features.size()) / (extraction_time.count() / 1000000.0);
    }
    
    data.metrics.last_update = std::chrono::steady_clock::now();
}

FeatureHealthReport FeatureFeeder::calculate_health_report([[maybe_unused]] const StrategyData& data, const std::vector<double>& features) {
    FeatureHealthReport report;
    
    if (features.empty()) {
        report.is_healthy = false;
        report.health_summary = "No features available";
        return report;
    }
    
    report.feature_health.resize(features.size(), true);
    report.feature_quality_scores.resize(features.size(), 1.0);
    
    // Check for NaN or infinite values
    for (size_t i = 0; i < features.size(); ++i) {
        if (!std::isfinite(features[i])) {
            report.feature_health[i] = false;
            report.feature_quality_scores[i] = 0.0;
        }
    }
    
    // Calculate overall health
    size_t healthy_features = std::count(report.feature_health.begin(), report.feature_health.end(), true);
    report.overall_health_score = static_cast<double>(healthy_features) / features.size();
    report.is_healthy = report.overall_health_score > 0.8; // 80% threshold
    
    if (report.is_healthy) {
        report.health_summary = "All features are healthy";
    } else {
        report.health_summary = "Some features are unhealthy";
    }
    
    return report;
}

// Cached features implementation
bool FeatureFeeder::load_feature_cache(const std::string& feature_file_path) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    feature_cache_ = std::make_unique<FeatureCache>();
    if (feature_cache_->load_from_csv(feature_file_path)) {
        std::cout << "FeatureFeeder: Successfully loaded feature cache from " << feature_file_path << std::endl;
        return true;
    } else {
        feature_cache_.reset();
        std::cerr << "FeatureFeeder: Failed to load feature cache from " << feature_file_path << std::endl;
        return false;
    }
}

bool FeatureFeeder::use_cached_features(bool enable) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    use_cached_features_ = enable;
    std::cout << "FeatureFeeder: Cached features " << (enable ? "ENABLED" : "DISABLED") << std::endl;
    return true;
}

bool FeatureFeeder::has_cached_features() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return feature_cache_ != nullptr;
}

} // namespace sentio
```

## 📄 **FILE 154 of 180**: src/future_qqq_loader.cpp

**File Information**:
- **Path**: `src/future_qqq_loader.cpp`

- **Size**: 136 lines
- **Modified**: 2025-09-14 12:01:50

- **Type**: .cpp

```text
#include "sentio/future_qqq_loader.hpp"
#include "sentio/csv_loader.hpp"
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>

namespace sentio {

// Define regime to track mapping
const std::map<FutureQQQLoader::Regime, std::vector<int>> FutureQQQLoader::regime_tracks_ = {
    {Regime::NORMAL, {1, 4, 7, 10}},     // 4 normal tracks
    {Regime::VOLATILE, {2, 5, 8}},       // 3 volatile tracks
    {Regime::TRENDING, {3, 6, 9}}        // 3 trending tracks
};

std::vector<Bar> FutureQQQLoader::load_track(int track_id) {
    if (track_id < 1 || track_id > 10) {
        throw std::invalid_argument("Track ID must be between 1 and 10, got: " + std::to_string(track_id));
    }

    std::string file_path = get_track_file_path(track_id);
    
    if (!std::filesystem::exists(file_path)) {
        throw std::runtime_error("Future QQQ track file not found: " + file_path);
    }

    std::cout << "📊 Loading future QQQ track " << track_id << " from: " << file_path << std::endl;

    // Use existing CSV loader to load the data
    std::vector<Bar> bars;
    bool success = load_csv(file_path, bars);
    
    if (!success || bars.empty()) {
        throw std::runtime_error("Failed to load future QQQ track " + std::to_string(track_id));
    }

    std::cout << "✅ Loaded " << bars.size() << " bars from future QQQ track " << track_id << std::endl;
    return bars;
}

std::vector<Bar> FutureQQQLoader::load_regime_track(Regime regime, int seed) {
    auto track_ids = get_regime_tracks(regime);
    
    if (track_ids.empty()) {
        throw std::runtime_error("No tracks available for the specified regime");
    }

    // Select random track from the regime
    int selected_track;
    if (seed >= 0) {
        // Use seed for reproducible selection
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, track_ids.size() - 1);
        selected_track = track_ids[dist(rng)];
    } else {
        // Use random device for true randomness
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> dist(0, track_ids.size() - 1);
        selected_track = track_ids[dist(rng)];
    }

    std::string regime_name;
    switch (regime) {
        case Regime::NORMAL: regime_name = "normal"; break;
        case Regime::VOLATILE: regime_name = "volatile"; break;
        case Regime::TRENDING: regime_name = "trending"; break;
    }

    std::cout << "🎯 Selected track " << selected_track << " for " << regime_name << " regime" << std::endl;
    return load_track(selected_track);
}

std::vector<Bar> FutureQQQLoader::load_regime_track(const std::string& regime_str, int seed) {
    Regime regime = string_to_regime(regime_str);
    return load_regime_track(regime, seed);
}

std::vector<int> FutureQQQLoader::get_regime_tracks(Regime regime) {
    auto it = regime_tracks_.find(regime);
    if (it != regime_tracks_.end()) {
        return it->second;
    }
    return {};
}

FutureQQQLoader::Regime FutureQQQLoader::string_to_regime(const std::string& regime_str) {
    if (regime_str == "normal") {
        return Regime::NORMAL;
    } else if (regime_str == "volatile") {
        return Regime::VOLATILE;
    } else if (regime_str == "trending") {
        return Regime::TRENDING;
    } else {
        // Default to normal for unknown regimes (including "bear", "bull", etc.)
        std::cout << "⚠️  Unknown regime '" << regime_str << "', defaulting to 'normal'" << std::endl;
        return Regime::NORMAL;
    }
}

std::string FutureQQQLoader::get_data_directory() {
    return "data/future_qqq";
}

bool FutureQQQLoader::validate_tracks() {
    std::string base_dir = get_data_directory();
    
    if (!std::filesystem::exists(base_dir)) {
        std::cerr << "❌ Future QQQ data directory not found: " << base_dir << std::endl;
        return false;
    }

    // Check all 10 tracks
    for (int i = 1; i <= 10; ++i) {
        std::string file_path = get_track_file_path(i);
        if (!std::filesystem::exists(file_path)) {
            std::cerr << "❌ Future QQQ track " << i << " not found: " << file_path << std::endl;
            return false;
        }
    }

    std::cout << "✅ All 10 future QQQ tracks validated successfully" << std::endl;
    return true;
}

std::string FutureQQQLoader::get_track_file_path(int track_id) {
    std::string base_dir = get_data_directory();
    
    // Format track ID with leading zero (e.g., "01", "02", ..., "10")
    std::string track_id_str = (track_id < 10) ? "0" + std::to_string(track_id) : std::to_string(track_id);
    
    return base_dir + "/future_qqq_track_" + track_id_str + ".csv";
}

} // namespace sentio

```

## 📄 **FILE 155 of 180**: src/global_leverage_config.cpp

**File Information**:
- **Path**: `src/global_leverage_config.cpp`

- **Size**: 8 lines
- **Modified**: 2025-09-14 03:59:59

- **Type**: .cpp

```text
#include "sentio/global_leverage_config.hpp"

namespace sentio {

// Default to theoretical pricing enabled
bool GlobalLeverageConfig::use_theoretical_leverage_pricing_ = true;

} // namespace sentio

```

## 📄 **FILE 156 of 180**: src/leverage_aware_csv_loader.cpp

**File Information**:
- **Path**: `src/leverage_aware_csv_loader.cpp`

- **Size**: 151 lines
- **Modified**: 2025-09-14 13:14:41

- **Type**: .cpp

```text
#include "sentio/leverage_aware_csv_loader.hpp"
#include "sentio/accurate_leverage_pricing.hpp"
#include "sentio/sym/leverage_registry.hpp"
#include <iostream>

namespace sentio {

// Static member definitions - Use accurate model by default
LeverageCostModel LeveragePricingConfig::cost_model_{};
bool LeveragePricingConfig::use_theoretical_pricing_ = true; // Default to theoretical pricing

// Initialize with calibrated accurate model
static bool initialize_accurate_model() {
    LeverageCostModel accurate_model;
    // Use calibrated parameters from our analysis
    accurate_model.expense_ratio = 0.0095;      // 0.95% (actual TQQQ/SQQQ)
    accurate_model.borrowing_cost_rate = 0.05;  // 5% (current environment)
    accurate_model.daily_decay_factor = 0.0001; // 0.01% daily rebalancing
    accurate_model.bid_ask_spread = 0.0001;     // 0.01% spread
    accurate_model.tracking_error_std = 0.00005; // Minimal tracking error
    LeveragePricingConfig::set_cost_model(accurate_model);
    return true;
}
static bool g_accurate_model_initialized = initialize_accurate_model();

bool load_csv_leverage_aware(const std::string& symbol, std::vector<Bar>& out) {
    if (!LeveragePricingConfig::is_theoretical_pricing_enabled()) {
        // Fall back to normal CSV loading
        std::string data_path = resolve_csv(symbol);
        return load_csv(data_path, out);
    }
    
    // Use the accurate pricing model instead of the basic one
    AccurateLeverageCostModel accurate_model;
    accurate_model.expense_ratio = 0.0095;
    accurate_model.borrowing_cost_rate = 0.05;
    accurate_model.daily_rebalance_cost = 0.0001;
    accurate_model.tracking_error_daily = 0.00005;
    
    AccurateLeveragePricer accurate_pricer(accurate_model);
    
    LeverageSpec spec;
    if (!LeverageRegistry::instance().lookup(symbol, spec)) {
        // Not a leverage instrument, load normally
        std::string data_path = resolve_csv(symbol);
        return load_csv(data_path, out);
    }
    
    // Load base symbol data
    std::vector<Bar> base_series;
    std::string base_data_path = resolve_csv(spec.base);
    if (!load_csv(base_data_path, base_series)) {
        std::cerr << "❌ Failed to load base data for " << spec.base << std::endl;
        return false;
    }
    
    // Generate theoretical leverage series using accurate model
    out = accurate_pricer.generate_accurate_theoretical_series(symbol, base_series, base_series);
    
    std::cout << "🧮 Generated " << out.size() << " accurate theoretical bars for " << symbol 
              << " (based on " << spec.base << ", ~1% accuracy)" << std::endl;
    
    return !out.empty();
}

bool load_family_csv_leverage_aware(const std::vector<std::string>& symbols,
                                   std::unordered_map<std::string, std::vector<Bar>>& series_out) {
    if (!LeveragePricingConfig::is_theoretical_pricing_enabled()) {
        // Fall back to normal CSV loading
        series_out.clear();
        for (const auto& symbol : symbols) {
            std::vector<Bar> series;
            std::string data_path = resolve_csv(symbol);
            if (load_csv(data_path, series)) {
                series_out[symbol] = std::move(series);
            } else {
                std::cerr << "❌ Failed to load data for " << symbol << std::endl;
                return false;
            }
        }
        return true;
    }
    
    // Use accurate pricing model for family loading
    AccurateLeverageCostModel accurate_model;
    accurate_model.expense_ratio = 0.0095;
    accurate_model.borrowing_cost_rate = 0.05;
    accurate_model.daily_rebalance_cost = 0.0001;
    accurate_model.tracking_error_daily = 0.00005;
    
    AccurateLeveragePricer accurate_pricer(accurate_model);
    
    series_out.clear();
    
    // Separate base and leverage symbols
    std::vector<std::string> base_symbols;
    std::vector<std::string> leverage_symbols;
    
    for (const auto& symbol : symbols) {
        LeverageSpec spec;
        if (LeverageRegistry::instance().lookup(symbol, spec)) {
            leverage_symbols.push_back(symbol);
            // Ensure base symbol is loaded
            if (std::find(base_symbols.begin(), base_symbols.end(), spec.base) == base_symbols.end()) {
                base_symbols.push_back(spec.base);
            }
        } else {
            base_symbols.push_back(symbol);
        }
    }
    
    // Load base symbols first
    for (const auto& symbol : base_symbols) {
        std::vector<Bar> series;
        std::string data_path = resolve_csv(symbol);
        if (load_csv(data_path, series)) {
            series_out[symbol] = std::move(series);
            std::cout << "📊 Loaded " << series_out[symbol].size() << " bars for " << symbol << std::endl;
        } else {
            std::cerr << "❌ Failed to load data for base symbol " << symbol << std::endl;
            return false;
        }
    }
    
    // Generate theoretical data for leverage symbols using accurate model
    for (const auto& symbol : leverage_symbols) {
        LeverageSpec spec;
        if (LeverageRegistry::instance().lookup(symbol, spec)) {
            auto base_it = series_out.find(spec.base);
            if (base_it != series_out.end()) {
                auto theoretical_series = accurate_pricer.generate_accurate_theoretical_series(
                    symbol, base_it->second, base_it->second);
                series_out[symbol] = std::move(theoretical_series);
                std::cout << "🧮 Generated " << series_out[symbol].size() << " accurate theoretical bars for " 
                          << symbol << " (based on " << spec.base << ", ~1% accuracy)" << std::endl;
            } else {
                std::cerr << "❌ Base symbol " << spec.base << " not found for " << symbol << std::endl;
                return false;
            }
        }
    }
    
    return true;
}

bool load_qqq_family_leverage_aware(std::unordered_map<std::string, std::vector<Bar>>& series_out) {
    std::vector<std::string> qqq_symbols = {"QQQ", "TQQQ", "SQQQ", "PSQ"};
    return load_family_csv_leverage_aware(qqq_symbols, series_out);
}

} // namespace sentio

```

## 📄 **FILE 157 of 180**: src/leverage_pricing.cpp

**File Information**:
- **Path**: `src/leverage_pricing.cpp`

- **Size**: 424 lines
- **Modified**: 2025-09-14 02:16:23

- **Type**: .cpp

```text
#include "sentio/leverage_pricing.hpp"
#include "sentio/csv_loader.hpp"
#include "sentio/data_resolver.hpp"
#include "sentio/sym/symbol_utils.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>

namespace sentio {

TheoreticalLeveragePricer::TheoreticalLeveragePricer(const LeverageCostModel& cost_model)
    : cost_model_(cost_model), rng_(std::chrono::steady_clock::now().time_since_epoch().count()) {
}

double TheoreticalLeveragePricer::calculate_theoretical_price(const std::string& leverage_symbol,
                                                           double base_price_prev,
                                                           double base_price_current,
                                                           double leverage_price_prev) {
    LeverageSpec spec;
    if (!LeverageRegistry::instance().lookup(leverage_symbol, spec)) {
        return leverage_price_prev; // Not a leverage instrument
    }
    
    // Calculate base return
    double base_return = (base_price_current - base_price_prev) / base_price_prev;
    
    // Apply leverage factor and inverse if needed
    double leveraged_return = spec.factor * base_return;
    if (spec.inverse) {
        leveraged_return = -leveraged_return;
    }
    
    // Apply costs (expense ratio, borrowing costs, decay)
    double daily_cost = cost_model_.daily_cost_rate();
    leveraged_return -= daily_cost;
    
    // Add tracking error noise
    std::normal_distribution<double> tracking_noise(0.0, cost_model_.tracking_error_std);
    leveraged_return += tracking_noise(rng_);
    
    // Calculate new theoretical price
    double theoretical_price = leverage_price_prev * (1.0 + leveraged_return);
    
    return std::max(theoretical_price, 0.01); // Prevent negative prices
}

Bar TheoreticalLeveragePricer::generate_theoretical_bar(const std::string& leverage_symbol,
                                                      const Bar& base_bar_prev,
                                                      const Bar& base_bar_current,
                                                      const Bar& leverage_bar_prev) {
    LeverageSpec spec;
    if (!LeverageRegistry::instance().lookup(leverage_symbol, spec)) {
        return leverage_bar_prev; // Not a leverage instrument
    }
    
    Bar theoretical_bar;
    theoretical_bar.ts_utc = base_bar_current.ts_utc;
    theoretical_bar.ts_utc_epoch = base_bar_current.ts_utc_epoch;
    
    // Calculate theoretical OHLC based on base bar movements
    double base_open_return = (base_bar_current.open - base_bar_prev.close) / base_bar_prev.close;
    double base_high_return = (base_bar_current.high - base_bar_prev.close) / base_bar_prev.close;
    double base_low_return = (base_bar_current.low - base_bar_prev.close) / base_bar_prev.close;
    double base_close_return = (base_bar_current.close - base_bar_prev.close) / base_bar_prev.close;
    
    // Apply leverage and inverse
    double leverage_factor = spec.inverse ? -spec.factor : spec.factor;
    double daily_cost = cost_model_.daily_cost_rate();
    
    // Generate tracking error for each price point
    std::normal_distribution<double> tracking_noise(0.0, cost_model_.tracking_error_std);
    
    double open_return = leverage_factor * base_open_return - daily_cost + tracking_noise(rng_);
    double high_return = leverage_factor * base_high_return - daily_cost + tracking_noise(rng_);
    double low_return = leverage_factor * base_low_return - daily_cost + tracking_noise(rng_);
    double close_return = leverage_factor * base_close_return - daily_cost + tracking_noise(rng_);
    
    // Calculate theoretical prices
    theoretical_bar.open = leverage_bar_prev.close * (1.0 + open_return);
    theoretical_bar.high = leverage_bar_prev.close * (1.0 + high_return);
    theoretical_bar.low = leverage_bar_prev.close * (1.0 + low_return);
    theoretical_bar.close = leverage_bar_prev.close * (1.0 + close_return);
    
    // Ensure OHLC consistency (high >= max(open, close), low <= min(open, close))
    theoretical_bar.high = std::max({theoretical_bar.high, theoretical_bar.open, theoretical_bar.close});
    theoretical_bar.low = std::min({theoretical_bar.low, theoretical_bar.open, theoretical_bar.close});
    
    // Prevent negative prices
    theoretical_bar.open = std::max(theoretical_bar.open, 0.01);
    theoretical_bar.high = std::max(theoretical_bar.high, 0.01);
    theoretical_bar.low = std::max(theoretical_bar.low, 0.01);
    theoretical_bar.close = std::max(theoretical_bar.close, 0.01);
    
    // Scale volume based on leverage factor (leveraged ETFs typically have higher volume)
    theoretical_bar.volume = static_cast<uint64_t>(base_bar_current.volume * (1.0 + spec.factor * 0.2));
    
    return theoretical_bar;
}

std::vector<Bar> TheoreticalLeveragePricer::generate_theoretical_series(const std::string& leverage_symbol,
                                                                      const std::vector<Bar>& base_series,
                                                                      double initial_price) {
    if (base_series.empty()) {
        return {};
    }
    
    LeverageSpec spec;
    if (!LeverageRegistry::instance().lookup(leverage_symbol, spec)) {
        return base_series; // Not a leverage instrument, return base series
    }
    
    std::vector<Bar> theoretical_series;
    theoretical_series.reserve(base_series.size());
    
    // Set initial price if not provided
    if (initial_price <= 0.0) {
        // Use a reasonable initial price based on the base price and leverage factor
        initial_price = base_series[0].close / spec.factor;
        if (leverage_symbol == "TQQQ") initial_price = 120.0; // Use realistic TQQQ price level
        if (leverage_symbol == "SQQQ") initial_price = 120.0; // Use realistic SQQQ price level
    }
    
    // Create initial bar
    Bar initial_bar = base_series[0];
    initial_bar.open = initial_bar.high = initial_bar.low = initial_bar.close = initial_price;
    theoretical_series.push_back(initial_bar);
    
    // Generate subsequent bars
    for (size_t i = 1; i < base_series.size(); ++i) {
        Bar theoretical_bar = generate_theoretical_bar(leverage_symbol,
                                                     base_series[i-1],
                                                     base_series[i],
                                                     theoretical_series[i-1]);
        theoretical_series.push_back(theoretical_bar);
    }
    
    return theoretical_series;
}

LeveragePricingValidator::LeveragePricingValidator(const LeverageCostModel& cost_model)
    : pricer_(cost_model) {
}

LeveragePricingValidator::ValidationResult LeveragePricingValidator::validate_pricing(
    const std::string& leverage_symbol,
    const std::vector<Bar>& base_series,
    const std::vector<Bar>& actual_leverage_series) {
    
    ValidationResult result;
    result.symbol = leverage_symbol;
    result.num_observations = 0;
    
    if (base_series.empty() || actual_leverage_series.empty()) {
        return result;
    }
    
    // Generate theoretical series
    double initial_price = actual_leverage_series[0].close;
    auto theoretical_series = pricer_.generate_theoretical_series(leverage_symbol, base_series, initial_price);
    
    if (theoretical_series.empty()) {
        return result;
    }
    
    // Align series by timestamp (use minimum length)
    size_t min_length = std::min({base_series.size(), actual_leverage_series.size(), theoretical_series.size()});
    
    std::vector<double> theoretical_prices, actual_prices;
    std::vector<double> theoretical_returns, actual_returns;
    std::vector<double> price_errors, return_errors;
    
    for (size_t i = 1; i < min_length; ++i) {
        // Skip if timestamps don't match (simple alignment)
        if (base_series[i].ts_utc_epoch != actual_leverage_series[i].ts_utc_epoch) {
            continue;
        }
        
        double theo_price = theoretical_series[i].close;
        double actual_price = actual_leverage_series[i].close;
        
        double theo_return = (theoretical_series[i].close - theoretical_series[i-1].close) / theoretical_series[i-1].close;
        double actual_return = (actual_leverage_series[i].close - actual_leverage_series[i-1].close) / actual_leverage_series[i-1].close;
        
        theoretical_prices.push_back(theo_price);
        actual_prices.push_back(actual_price);
        theoretical_returns.push_back(theo_return);
        actual_returns.push_back(actual_return);
        
        price_errors.push_back((theo_price - actual_price) / actual_price);
        return_errors.push_back(theo_return - actual_return);
    }
    
    result.num_observations = price_errors.size();
    
    if (result.num_observations == 0) {
        return result;
    }
    
    // Calculate correlations
    auto calculate_correlation = [](const std::vector<double>& x, const std::vector<double>& y) -> double {
        if (x.size() != y.size() || x.empty()) return 0.0;
        
        double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
        double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
        
        double numerator = 0.0, sum_sq_x = 0.0, sum_sq_y = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double dx = x[i] - mean_x;
            double dy = y[i] - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }
        
        double denominator = std::sqrt(sum_sq_x * sum_sq_y);
        return (denominator > 0) ? numerator / denominator : 0.0;
    };
    
    result.price_correlation = calculate_correlation(theoretical_prices, actual_prices);
    result.return_correlation = calculate_correlation(theoretical_returns, actual_returns);
    
    // Calculate error statistics
    result.mean_price_error = std::accumulate(price_errors.begin(), price_errors.end(), 0.0) / price_errors.size();
    result.mean_return_error = std::accumulate(return_errors.begin(), return_errors.end(), 0.0) / return_errors.size();
    
    double price_error_var = 0.0, return_error_var = 0.0;
    for (size_t i = 0; i < price_errors.size(); ++i) {
        price_error_var += (price_errors[i] - result.mean_price_error) * (price_errors[i] - result.mean_price_error);
        return_error_var += (return_errors[i] - result.mean_return_error) * (return_errors[i] - result.mean_return_error);
    }
    result.price_error_std = std::sqrt(price_error_var / price_errors.size());
    result.return_error_std = std::sqrt(return_error_var / return_errors.size());
    
    // Calculate total returns
    if (!theoretical_prices.empty() && !actual_prices.empty()) {
        result.theoretical_total_return = (theoretical_prices.back() - theoretical_prices.front()) / theoretical_prices.front();
        result.actual_total_return = (actual_prices.back() - actual_prices.front()) / actual_prices.front();
        result.return_difference = result.theoretical_total_return - result.actual_total_return;
    }
    
    return result;
}

void LeveragePricingValidator::print_validation_report(const ValidationResult& result) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "📊 LEVERAGE PRICING VALIDATION REPORT" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Symbol:                   " << result.symbol << std::endl;
    std::cout << "Observations:             " << result.num_observations << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "📈 CORRELATION ANALYSIS" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Price Correlation:        " << std::fixed << std::setprecision(4) << result.price_correlation << std::endl;
    std::cout << "Return Correlation:       " << std::fixed << std::setprecision(4) << result.return_correlation << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "📊 ERROR ANALYSIS" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Mean Price Error:         " << std::fixed << std::setprecision(4) << result.mean_price_error * 100 << "%" << std::endl;
    std::cout << "Price Error Std Dev:      " << std::fixed << std::setprecision(4) << result.price_error_std * 100 << "%" << std::endl;
    std::cout << "Mean Return Error:        " << std::fixed << std::setprecision(4) << result.mean_return_error * 100 << "%" << std::endl;
    std::cout << "Return Error Std Dev:     " << std::fixed << std::setprecision(4) << result.return_error_std * 100 << "%" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "💰 TOTAL RETURN COMPARISON" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Theoretical Total Return: " << std::fixed << std::setprecision(2) << result.theoretical_total_return * 100 << "%" << std::endl;
    std::cout << "Actual Total Return:      " << std::fixed << std::setprecision(2) << result.actual_total_return * 100 << "%" << std::endl;
    std::cout << "Return Difference:        " << std::fixed << std::setprecision(2) << result.return_difference * 100 << "%" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Quality assessment
    std::cout << "🎯 MODEL QUALITY ASSESSMENT" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    if (result.return_correlation > 0.95) {
        std::cout << "✅ EXCELLENT: Return correlation > 95%" << std::endl;
    } else if (result.return_correlation > 0.90) {
        std::cout << "✅ GOOD: Return correlation > 90%" << std::endl;
    } else if (result.return_correlation > 0.80) {
        std::cout << "⚠️  FAIR: Return correlation > 80%" << std::endl;
    } else {
        std::cout << "❌ POOR: Return correlation < 80%" << std::endl;
    }
    
    if (std::abs(result.return_difference) < 0.02) {
        std::cout << "✅ EXCELLENT: Total return difference < 2%" << std::endl;
    } else if (std::abs(result.return_difference) < 0.05) {
        std::cout << "✅ GOOD: Total return difference < 5%" << std::endl;
    } else if (std::abs(result.return_difference) < 0.10) {
        std::cout << "⚠️  FAIR: Total return difference < 10%" << std::endl;
    } else {
        std::cout << "❌ POOR: Total return difference > 10%" << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl << std::endl;
}

LeverageCostModel LeveragePricingValidator::calibrate_cost_model(const std::string& leverage_symbol,
                                                               const std::vector<Bar>& base_series,
                                                               const std::vector<Bar>& actual_leverage_series) {
    // Simple grid search calibration
    LeverageCostModel best_model;
    double best_correlation = -1.0;
    
    std::vector<double> expense_ratios = {0.005, 0.0075, 0.0095, 0.012, 0.015};
    std::vector<double> borrowing_costs = {0.02, 0.035, 0.05, 0.065, 0.08};
    std::vector<double> decay_factors = {0.00005, 0.0001, 0.0002, 0.0003, 0.0005};
    
    std::cout << "🔧 Calibrating cost model for " << leverage_symbol << "..." << std::endl;
    
    for (double expense_ratio : expense_ratios) {
        for (double borrowing_cost : borrowing_costs) {
            for (double decay_factor : decay_factors) {
                LeverageCostModel test_model;
                test_model.expense_ratio = expense_ratio;
                test_model.borrowing_cost_rate = borrowing_cost;
                test_model.daily_decay_factor = decay_factor;
                
                TheoreticalLeveragePricer test_pricer(test_model);
                LeveragePricingValidator test_validator(test_model);
                
                auto result = test_validator.validate_pricing(leverage_symbol, base_series, actual_leverage_series);
                
                if (result.return_correlation > best_correlation) {
                    best_correlation = result.return_correlation;
                    best_model = test_model;
                }
            }
        }
    }
    
    std::cout << "✅ Best correlation found: " << std::fixed << std::setprecision(4) << best_correlation << std::endl;
    std::cout << "📊 Optimal parameters:" << std::endl;
    std::cout << "   Expense Ratio: " << std::fixed << std::setprecision(3) << best_model.expense_ratio * 100 << "%" << std::endl;
    std::cout << "   Borrowing Cost: " << std::fixed << std::setprecision(3) << best_model.borrowing_cost_rate * 100 << "%" << std::endl;
    std::cout << "   Daily Decay: " << std::fixed << std::setprecision(5) << best_model.daily_decay_factor * 100 << "%" << std::endl;
    
    return best_model;
}

LeverageAwareDataLoader::LeverageAwareDataLoader(const LeverageCostModel& cost_model)
    : pricer_(cost_model) {
}

bool LeverageAwareDataLoader::load_symbol_data(const std::string& symbol, std::vector<Bar>& out) {
    LeverageSpec spec;
    if (!LeverageRegistry::instance().lookup(symbol, spec)) {
        // Not a leverage instrument, load normally
        std::string data_path = resolve_csv(symbol);
        return load_csv(data_path, out);
    }
    
    // Load base symbol data
    std::vector<Bar> base_series;
    std::string base_data_path = resolve_csv(spec.base);
    if (!load_csv(base_data_path, base_series)) {
        std::cerr << "❌ Failed to load base data for " << spec.base << std::endl;
        return false;
    }
    
    // Generate theoretical leverage series
    out = pricer_.generate_theoretical_series(symbol, base_series);
    
    std::cout << "🧮 Generated " << out.size() << " theoretical bars for " << symbol 
              << " (based on " << spec.base << ")" << std::endl;
    
    return !out.empty();
}

bool LeverageAwareDataLoader::load_family_data(const std::vector<std::string>& symbols,
                                             std::unordered_map<std::string, std::vector<Bar>>& series_out) {
    series_out.clear();
    
    // Separate base and leverage symbols
    std::vector<std::string> base_symbols;
    std::vector<std::string> leverage_symbols;
    
    for (const auto& symbol : symbols) {
        LeverageSpec spec;
        if (LeverageRegistry::instance().lookup(symbol, spec)) {
            leverage_symbols.push_back(symbol);
            // Ensure base symbol is loaded
            if (std::find(base_symbols.begin(), base_symbols.end(), spec.base) == base_symbols.end()) {
                base_symbols.push_back(spec.base);
            }
        } else {
            base_symbols.push_back(symbol);
        }
    }
    
    // Load base symbols first
    for (const auto& symbol : base_symbols) {
        std::vector<Bar> series;
        std::string data_path = resolve_csv(symbol);
        if (load_csv(data_path, series)) {
            series_out[symbol] = std::move(series);
            std::cout << "📊 Loaded " << series_out[symbol].size() << " bars for " << symbol << std::endl;
        } else {
            std::cerr << "❌ Failed to load data for base symbol " << symbol << std::endl;
            return false;
        }
    }
    
    // Generate theoretical data for leverage symbols
    for (const auto& symbol : leverage_symbols) {
        LeverageSpec spec;
        if (LeverageRegistry::instance().lookup(symbol, spec)) {
            auto base_it = series_out.find(spec.base);
            if (base_it != series_out.end()) {
                auto theoretical_series = pricer_.generate_theoretical_series(symbol, base_it->second);
                series_out[symbol] = std::move(theoretical_series);
                std::cout << "🧮 Generated " << series_out[symbol].size() << " theoretical bars for " 
                          << symbol << " (based on " << spec.base << ")" << std::endl;
            } else {
                std::cerr << "❌ Base symbol " << spec.base << " not found for " << symbol << std::endl;
                return false;
            }
        }
    }
    
    return true;
}

} // namespace sentio

```

## 📄 **FILE 158 of 180**: src/main.cpp

**File Information**:
- **Path**: `src/main.cpp`

- **Size**: 818 lines
- **Modified**: 2025-09-17 16:22:34

- **Type**: .cpp

```text
#include "sentio/canonical_evaluation.hpp"
#include "sentio/dataset_metadata.hpp"
#include "sentio/csv_loader.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/all_strategies.hpp"
#include "sentio/data_downloader.hpp"
#include "sentio/cli_helpers.hpp"
#include "sentio/mars_data_loader.hpp"
#include "sentio/run_id_generator.hpp"
#include "sentio/runner.hpp"
#include "sentio/virtual_market.hpp"
#include "audit/audit_db_recorder.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <ctime>
#include <fstream>
#include <iomanip>


void usage() {
    std::cout << "Usage: sentio_cli <command> [options]\n\n"
              << "STRATEGY TESTING:\n"
              << "  strattest <strategy> [symbol] [options]    Unified strategy robustness testing (symbol defaults to QQQ)\n"
              << "  list-strategies [options]                  List all available strategies\n"
              << "\n"
              << "DATA MANAGEMENT:\n"
              << "  download <symbol> [options]               Download historical data from Polygon.io\n"
              << "  probe                                     Show data availability and system status\n"
              << "\n"
              << "DEVELOPMENT & VALIDATION:\n"
              << "  audit-validate                            Validate strategies with audit system\n"
              << "\n"
              << "Global Options:\n"
              << "  --help, -h                                Show command-specific help\n"
              << "  --verbose, -v                             Enable verbose output\n"
              << "  --output <format>                         Output format: console|json|csv\n"
              << "\n"
              << "Examples:\n"
              << "  sentio_cli list-strategies --format table --verbose\n"
              << "  sentio_cli strattest momentum --mode hybrid --blocks 20\n"
              << "  sentio_cli strattest ire --comprehensive --stress-test\n"
              << "  sentio_cli download QQQ --period 3y\n"
              << "  sentio_cli probe\n"
              << "\n"
              << "Use 'sentio_cli <command> --help' for command-specific options.\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // Initialize strategies using factory pattern
    if (!sentio::initialize_strategies()) {
        std::cerr << "Warning: Failed to initialize strategies" << std::endl;
    }
    
    if (argc < 2) {
        usage();
        return 1;
    }

    // Parse arguments using CLI helpers
    auto args = sentio::CLIHelpers::parse_arguments(argc, argv);
    std::string command = args.command;
    
    // Handle global help
    if (command.empty() || (args.help_requested && command.empty())) {
        usage();
        return 0;
    }
    
    if (command == "strattest") {
        if (args.help_requested) {
            sentio::CLIHelpers::print_help("strattest", 
                "sentio_cli strattest <strategy> [symbol] [options]",
                {
                    "=== DATA MODES ===",
                    "--mode <mode>              Data mode: historical|simulation|ai-regime (default: simulation)",
                    "                           • historical: Real QQQ market data (data/equities/)",
                    "                           • simulation: MarS pre-generated tracks (data/future_qqq/)",  
                    "                           • ai-regime: Real-time MarS generation (may take 30-60s)",
                    "",
                    "=== TRADING BLOCK CONFIGURATION ===",
                    "--blocks <n>               Number of Trading Blocks to test (default: 10)",
                    "--block-size <bars>        Bars per Trading Block (default: 480 ≈ 8hrs)",
                    "",
                    "=== REGIME & TRACK OPTIONS ===",
                    "--regime <regime>          Market regime: normal|volatile|trending (default: normal)",
                    "                           • For simulation: selects appropriate track",
                    "                           • For ai-regime: configures MarS generation",
                    "--track <n>                Force specific track (1-10, simulation mode only)",
                    "--alpaca-fees              Use Alpaca fee structure (default: true)",
                    "--alpaca-limits            Apply Alpaca position/order limits",
                    "--confidence <level>       Confidence level: 90|95|99 (default: 95)",
                    "--output <format>          Output format: console|json|csv (default: console)",
                    "--save-results <file>      Save detailed results to file",
                    "--benchmark <symbol>       Benchmark symbol (default: SPY)",
                    "--quick                    Quick mode: 10 TB test",
                    "--comprehensive            Comprehensive mode: 60 TB test (3 months)",
                    "--monthly                  Monthly benchmark: 20 TB test (1 month)",
                    "--params <json>            Strategy parameters as JSON string (default: '{}')",
                    "",
                    "Trading Block Info: 1 TB = 480 bars ≈ 8 hours, 10 TB ≈ 2 weeks, 20 TB ≈ 1 month",
                    "Note: Symbol defaults to QQQ if not specified"
                },
                {
                    "# Historical mode (real market data)",
                    "sentio_cli strattest ire --mode historical --blocks 20",
                    "",
                    "# Simulation mode (MarS pre-generated tracks)", 
                    "sentio_cli strattest momentum --mode simulation --regime volatile --track 2",
                    "sentio_cli strattest tfa --mode simulation --regime trending --blocks 30",
                    "",
                    "# AI Regime mode (real-time MarS generation)",
                    "sentio_cli strattest ire --mode ai-regime --regime volatile --blocks 15",
                    "sentio_cli strattest mm --mode ai-regime --regime normal --monthly"
                });
            return 0;
        }
        
        if (!sentio::CLIHelpers::validate_required_args(args, 1, 
            "sentio_cli strattest <strategy> [symbol] [options]")) {
            return 1;
        }
        
        std::string strategy_name = args.positional_args[0];
        std::string symbol = (args.positional_args.size() > 1) ? args.positional_args[1] : "QQQ";
        
        // Validate command options
        std::vector<std::string> allowed_options = {
            "mode", "blocks", "block-size", "regime", "track", "confidence", 
            "output", "save-results", "benchmark", "params"
        };
        std::vector<std::string> allowed_flags = {
            "alpaca-fees", "alpaca-limits", "quick", "comprehensive", "monthly"
        };
        
        if (!sentio::CLIHelpers::validate_options(args, "strattest", allowed_options, allowed_flags)) {
            return 1;
        }
        
        // Validate inputs
        if (!sentio::CLIHelpers::is_valid_strategy_name(strategy_name)) {
            sentio::CLIHelpers::print_error("Invalid strategy name: " + strategy_name);
            return 1;
        }
        
        if (!sentio::CLIHelpers::is_valid_symbol(symbol)) {
            sentio::CLIHelpers::print_error("Invalid symbol: " + symbol);
            return 1;
        }
        
        // Parse mode and Trading Block configuration  
        std::string mode_str = sentio::CLIHelpers::get_option(args, "mode", "simulation");
        
        // Parse Trading Block configuration (canonical evaluation only)
        int num_blocks = 10;  // Default to 10 TB
        int block_size = 480; // Default to 480 bars per TB (8 hours)
        
        // Check for preset modes first
        if (sentio::CLIHelpers::get_flag(args, "quick")) {
            num_blocks = 10;  // Quick: 10 TB ≈ 2 weeks
        } else if (sentio::CLIHelpers::get_flag(args, "comprehensive")) {
            num_blocks = 60;  // Comprehensive: 60 TB ≈ 3 months
        } else if (sentio::CLIHelpers::get_flag(args, "monthly")) {
            num_blocks = 20;  // Monthly: 20 TB ≈ 1 month
        }
        
        // Allow explicit override of defaults
        num_blocks = sentio::CLIHelpers::get_int_option(args, "blocks", num_blocks);
        block_size = sentio::CLIHelpers::get_int_option(args, "block-size", block_size);
        
        // Show Trading Block configuration
        std::cout << "\n📊 **TRADING BLOCK CONFIGURATION**" << std::endl;
        std::cout << "Strategy: " << strategy_name << " (" << symbol << ")" << std::endl;
        std::cout << "Mode: " << mode_str << std::endl;
        std::cout << "Trading Blocks: " << num_blocks << " TB × " << block_size << " bars" << std::endl;
        std::cout << "Total Duration: " << (num_blocks * block_size) << " bars ≈ " 
                  << std::fixed << std::setprecision(1) << ((num_blocks * block_size) / 390.0) << " trading days" << std::endl;
        std::cout << "Equivalent: ~" << std::fixed << std::setprecision(1) << ((num_blocks * block_size) / 60.0 / 8.0) << " trading days (8hrs/day)" << std::endl;
        
        if (num_blocks >= 20) {
            std::cout << "📈 20TB Benchmark: Available (monthly performance measurement)" << std::endl;
        } else {
            std::cout << "ℹ️  For monthly benchmark (20TB), use --monthly or --blocks 20" << std::endl;
        }
        std::cout << std::endl;

        // Check if we should use the new Trading Block canonical evaluation
        bool use_canonical_evaluation = (num_blocks > 0 && block_size > 0);
        
        if (use_canonical_evaluation) {
            std::cout << "\n🎯 **CANONICAL TRADING BLOCK EVALUATION**" << std::endl;
            std::cout << "Using deterministic Trading Block system instead of legacy duration-based testing" << std::endl;
            std::cout << std::endl;
            
            // Use new canonical evaluation system
            try {
                // Create Trading Block configuration
                sentio::TradingBlockConfig block_config;
                block_config.block_size = block_size;
                block_config.num_blocks = num_blocks;
                
                std::cout << "🚀 Loading data for canonical evaluation..." << std::endl;
                
                // Load data using the profit-maximizing pipeline (QQQ family for leverage)
                sentio::SymbolTable ST;
                std::vector<std::vector<sentio::Bar>> series;
                
                // **PROFIT MAXIMIZATION**: Load full QQQ family for maximum leverage
                int base_symbol_id = ST.intern(symbol);
                int tqqq_id = ST.intern("TQQQ");  // 3x leveraged long
                int sqqq_id = ST.intern("SQQQ");  // 3x leveraged short  
                int psq_id = ST.intern("PSQ");    // 1x inverse
                
                series.resize(4);  // QQQ family
                
                // Select data source based on mode
                std::string data_file;
                std::string dataset_type;
                std::vector<sentio::Bar> bars;
                
                if (mode_str == "historical") {
                    // Mode 1: Use real QQQ RTH historical data from equities folder
                    data_file = "data/equities/QQQ_RTH_NH.csv";
                    dataset_type = "real_historical_qqq_rth";
                    std::cout << "📊 Loading " << symbol << " RTH historical data..." << std::endl;
                    std::cout << "📁 Dataset: " << data_file << " (Real Market Data - RTH Only)" << std::endl;
                    
                } else if (mode_str == "simulation") {
                    // Mode 2: Use MarS pre-generated future QQQ simulation data
                    std::string regime = sentio::CLIHelpers::get_option(args, "regime", "normal");
                    int track_num = sentio::CLIHelpers::get_int_option(args, "track", 1);
                    
                    // Select track based on regime preference
                    if (regime == "volatile") {
                        track_num = (track_num <= 3) ? (2 + (track_num-1)*3) : 2; // Tracks 2,5,8
                    } else if (regime == "trending") {
                        track_num = (track_num <= 3) ? (3 + (track_num-1)*3) : 3; // Tracks 3,6,9
                    } else {
                        // Normal regime: tracks 1,4,7,10
                        if (track_num > 4) track_num = 1;
                        int normal_tracks[] = {1, 4, 7, 10};
                        track_num = normal_tracks[(track_num-1) % 4];
                    }
                    
                    char track_file[64];
                    snprintf(track_file, sizeof(track_file), "data/future_qqq/future_qqq_track_%02d.csv", track_num);
                    data_file = track_file;
                    dataset_type = "mars_simulation_" + regime;
                    
                    std::cout << "📊 Loading " << symbol << " MarS simulation data..." << std::endl;
                    std::cout << "🎯 Simulation Regime: " << regime << " (Track " << track_num << ")" << std::endl;
                    std::cout << "📁 Dataset: " << data_file << " (MarS Simulation)" << std::endl;
                    
                } else if (mode_str == "ai-regime") {
                    // Mode 3: Generate real-time AI data using MarS
                    std::string regime = sentio::CLIHelpers::get_option(args, "regime", "normal");
                    int required_bars = (num_blocks * block_size) + 250; // Include warmup bars
                    // MarS generates about 0.67 bars per minute, so multiply by 1.5 to ensure enough data
                    int duration_minutes = static_cast<int>(required_bars * 1.5);
                    
                    data_file = "temp_ai_regime_" + symbol + "_" + std::to_string(std::time(nullptr)) + ".json";
                    dataset_type = "mars_ai_realtime_" + regime;
                    
                    std::cout << "🤖 Generating " << symbol << " AI regime data..." << std::endl;
                    std::cout << "🎯 AI Regime: " << regime << " (" << duration_minutes << " minutes)" << std::endl;
                    std::cout << "⚡ Real-time generation - this may take 30-60 seconds..." << std::endl;
                    
                    // Generate MarS data in real-time
                    sentio::MarsDataLoader mars_loader;
                    bars = mars_loader.load_mars_data(symbol, duration_minutes, 60, 1, regime);
                    
                    if (bars.empty()) {
                        throw std::runtime_error("Failed to generate AI regime data with MarS");
                    }
                    
                    std::cout << "✅ Generated " << bars.size() << " bars with AI regime: " << regime << std::endl;
                    data_file = "AI-Generated (" + regime + " regime)";
                    
                } else {
                    throw std::runtime_error("Invalid mode: " + mode_str + ". Use: historical, simulation, or ai-regime");
                }
                
                // Load data (if not already loaded by AI regime mode)
                if (mode_str != "ai-regime") {
                    bool load_success = sentio::load_csv(data_file, bars);
                    if (!load_success || bars.empty()) {
                        throw std::runtime_error("Failed to load data from " + data_file);
                    }
                }
                
                std::cout << "✅ Loaded " << bars.size() << " bars" << std::endl;
                
                // Prepare series data structure with leverage instruments for profit maximization
                series.resize(ST.size());
                series[base_symbol_id] = bars;
                
                // **PROFIT MAXIMIZATION**: Generate theoretical leverage data for maximum capital deployment
                std::cout << "🚀 Generating theoretical leverage data for maximum profit..." << std::endl;
                series[tqqq_id] = generate_theoretical_leverage_series(bars, 3.0);   // 3x leveraged long
                series[sqqq_id] = generate_theoretical_leverage_series(bars, -3.0);  // 3x leveraged short  
                series[psq_id] = generate_theoretical_leverage_series(bars, -1.0);   // 1x inverse
                
                std::cout << "✅ TQQQ theoretical data generated (" << series[tqqq_id].size() << " bars, 3x leverage)" << std::endl;
                std::cout << "✅ SQQQ theoretical data generated (" << series[sqqq_id].size() << " bars, -3x leverage)" << std::endl;
                std::cout << "✅ PSQ theoretical data generated (" << series[psq_id].size() << " bars, -1x leverage)" << std::endl;
                
                // Create comprehensive dataset metadata with ISO timestamps
                sentio::DatasetMetadata dataset_meta;
                dataset_meta.source_type = dataset_type;
                dataset_meta.file_path = data_file;
                dataset_meta.bars_count = static_cast<int>(bars.size());
                dataset_meta.mode = mode_str;
                dataset_meta.regime = sentio::CLIHelpers::get_option(args, "regime", "normal");
                
                if (!bars.empty()) {
                    dataset_meta.time_range_start = bars.front().ts_utc_epoch * 1000;
                    dataset_meta.time_range_end = bars.back().ts_utc_epoch * 1000;
                    
                    // Convert timestamps to ISO format for display
                    auto start_time = std::time_t(bars.front().ts_utc_epoch);
                    auto end_time = std::time_t(bars.back().ts_utc_epoch);
                    
                    char start_iso[32], end_iso[32];
                    std::strftime(start_iso, sizeof(start_iso), "%Y-%m-%dT%H:%M:%S", std::gmtime(&start_time));
                    std::strftime(end_iso, sizeof(end_iso), "%Y-%m-%dT%H:%M:%S", std::gmtime(&end_time));
                    
                    std::cout << "📅 Dataset Period: " << start_iso << " to " << end_iso << " UTC" << std::endl;
                    
                    // Calculate and display test period (using most recent data)
                    size_t warmup_bars = 250;
                    size_t test_bars = num_blocks * block_size;
                    if (bars.size() > warmup_bars + test_bars) {
                        // Start from the end and work backwards to get the most recent data
                        size_t test_end_idx = bars.size() - 1;
                        size_t test_start_idx = test_end_idx - test_bars + 1;
                        
                        auto test_start_time = std::time_t(bars[test_start_idx].ts_utc_epoch);
                        auto test_end_time = std::time_t(bars[test_end_idx].ts_utc_epoch);
                        
                        char test_start_iso[32], test_end_iso[32];
                        std::strftime(test_start_iso, sizeof(test_start_iso), "%Y-%m-%dT%H:%M:%S", std::gmtime(&test_start_time));
                        std::strftime(test_end_iso, sizeof(test_end_iso), "%Y-%m-%dT%H:%M:%S", std::gmtime(&test_end_time));
                        
                        std::cout << "🎯 Test Period: " << test_start_iso << " to " << test_end_iso << " UTC" << std::endl;
                        std::cout << "⚡ Warmup Bars: " << warmup_bars << " (excluded from test)" << std::endl;
                    }
                }
                dataset_meta.calculate_trading_blocks(block_config.block_size);
                
                // Create audit recorder
                std::cout << "🔍 Initializing audit system..." << std::endl;
                std::string audit_db_path = "audit/sentio_audit.sqlite3";
                std::string run_id = sentio::generate_run_id();
                auto audit_recorder = std::make_unique<audit::AuditDBRecorder>(audit_db_path, run_id, "Trading Block canonical evaluation");
                
                // Create runner configuration
                sentio::RunnerCfg runner_cfg;
                runner_cfg.strategy_name = strategy_name;
                // Pass mode through strategy parameters (safer than adding to struct)
                runner_cfg.strategy_params["mode"] = mode_str;
                
                std::cout << "🎯 Running canonical Trading Block evaluation..." << std::endl;
                
                // Run the canonical evaluation!
                auto canonical_report = sentio::run_canonical_backtest(
                    *audit_recorder, 
                    ST, 
                    series, 
                    base_symbol_id, 
                    runner_cfg, 
                    dataset_meta, 
                    block_config
                );
                
                std::cout << "\n✅ **CANONICAL EVALUATION COMPLETED SUCCESSFULLY!**" << std::endl;
                std::cout << "🎉 Trading Block system is now fully operational!" << std::endl;
                std::cout << "\nUse './saudit summarize' to verify results in audit system" << std::endl;
                
                return 0; // Exit here - canonical evaluation complete
            
        } catch (const std::exception& e) {
                std::cerr << "❌ Canonical evaluation failed: " << e.what() << std::endl;
            return 1;
            }
        }
        
        // All Trading Block evaluations should succeed and return above
        std::cerr << "❌ Unexpected: Trading Block configuration invalid" << std::endl;
        return 1;
        
    } else if (command == "probe") {
        // ANSI color codes
        const std::string RESET = "\033[0m";
        const std::string BOLD = "\033[1m";
        const std::string DIM = "\033[2m";
        const std::string BLUE = "\033[34m";
        const std::string GREEN = "\033[32m";
        const std::string RED = "\033[31m";
        const std::string YELLOW = "\033[33m";
        const std::string CYAN = "\033[36m";
        const std::string MAGENTA = "\033[35m";
        const std::string WHITE = "\033[37m";
        const std::string BG_BLUE = "\033[44m";
        
        // Enhanced header
        std::cout << "\n" << BOLD << BG_BLUE << WHITE << "╔══════════════════════════════════════════════════════════════════════════════════╗" << RESET << std::endl;
        std::cout << BOLD << BG_BLUE << WHITE << "║                           🔍 SENTIO SYSTEM PROBE                                ║" << RESET << std::endl;
        std::cout << BOLD << BG_BLUE << WHITE << "╚══════════════════════════════════════════════════════════════════════════════════╝" << RESET << std::endl;
        
        auto strategies = sentio::StrategyFactory::instance().get_available_strategies();
        
        // Show available strategies
        std::cout << "\n" << BOLD << CYAN << "📊 AVAILABLE STRATEGIES" << RESET << " " << DIM << "(" << strategies.size() << " total)" << RESET << std::endl;
        std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
        
        // Display strategies in a nice grid format
        int count = 0;
        for (const auto& strategy_name : strategies) {
            if (count % 3 == 0) {
                std::cout << "│ ";
            }
            std::cout << MAGENTA << "• " << strategy_name << RESET;
            
            // Pad to make columns align
            int padding = 25 - strategy_name.length();
            for (int i = 0; i < padding; ++i) {
                std::cout << " ";
            }
            
            count++;
            if (count % 3 == 0) {
                std::cout << "│" << std::endl;
            }
        }
        
        // Handle remaining strategies if not divisible by 3
        if (count % 3 != 0) {
            int remaining = 3 - (count % 3);
            for (int i = 0; i < remaining; ++i) {
                for (int j = 0; j < 25; ++j) {
                    std::cout << " ";
                }
            }
            std::cout << "│" << std::endl;
        }
        
        std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
        
        // Helper function to get file info
        auto get_file_info = [](const std::string& path) -> std::pair<bool, std::pair<std::string, std::string>> {
            std::ifstream file(path);
            if (!file.good()) {
                return {false, {"", ""}};
            }
            
            std::string line;
            std::getline(file, line); // Skip header
            
            std::string start_time = "N/A", end_time = "N/A";
            if (std::getline(file, line)) {
                std::istringstream iss(line);
                std::getline(iss, start_time, ',');
                
                // Find last valid data line (skip any trailing header lines)
                std::string current_line = line;
                while (std::getline(file, line)) {
                    // Skip lines that start with "timestamp" (trailing headers in aligned files)
                    if (line.find("timestamp,") != 0 && !line.empty()) {
                        current_line = line;
                    }
                }
                
                if (!current_line.empty()) {
                    std::istringstream iss2(current_line);
                    std::getline(iss2, end_time, ',');
                }
            }
            
            return {true, {start_time, end_time}};
        };
        
        // Check data availability for key symbols
        std::vector<std::string> symbols = {"QQQ", "SPY", "AAPL", "MSFT", "TSLA"};
        std::cout << "\n" << BOLD << CYAN << "📈 DATA AVAILABILITY CHECK" << RESET << std::endl;
        std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
        
        bool daily_aligned = true, minute_aligned = true;
        
        for (const auto& symbol : symbols) {
            std::cout << "│ " << BOLD << "Symbol: " << BLUE << symbol << RESET << std::endl;
            
            // Check daily data
            std::string daily_path = "data/equities/" + symbol + "_daily.csv";
            auto [daily_exists, daily_range] = get_file_info(daily_path);
            
            std::cout << "│   📅 Daily:  ";
            if (daily_exists) {
                std::cout << GREEN << "✅ Available" << RESET << " " << DIM << "(" << daily_range.first << " to " << daily_range.second << ")" << RESET << std::endl;
            } else {
                std::cout << RED << "❌ Missing" << RESET << std::endl;
                daily_aligned = false;
            }
            
            // Check minute data
            std::string minute_path = "data/equities/" + symbol + "_NH.csv";
            auto [minute_exists, minute_range] = get_file_info(minute_path);
            
            std::cout << "│   ⏰ Minute: ";
            if (minute_exists) {
                std::cout << GREEN << "✅ Available" << RESET << " " << DIM << "(" << minute_range.first << " to " << minute_range.second << ")" << RESET << std::endl;
            } else {
                std::cout << RED << "❌ Missing" << RESET << std::endl;
                minute_aligned = false;
            }
            
            std::cout << "│" << std::endl;
        }
        
        std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
        
        // Summary
        std::cout << "\n" << BOLD << CYAN << "📋 SYSTEM STATUS" << RESET << std::endl;
        std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
        
        if (daily_aligned && minute_aligned) {
            std::cout << "│ " << GREEN << "🎉 ALL SYSTEMS READY" << RESET << " - Data is properly aligned for strategy testing" << std::endl;
            std::cout << "│" << std::endl;
            std::cout << "│ " << BOLD << "Quick Start Commands:" << RESET << std::endl;
            std::cout << "│   " << CYAN << "• ./build/sentio_cli strattest ire --mode simulation --blocks 10" << RESET << std::endl;
            std::cout << "│   " << CYAN << "• ./build/sentio_cli strattest tfa --mode historical --blocks 20" << RESET << std::endl;
            std::cout << "│   " << CYAN << "• ./saudit list --limit 10" << RESET << std::endl;
        } else {
            std::cout << "│ " << YELLOW << "⚠️  PARTIAL DATA AVAILABILITY" << RESET << " - Some data files are missing" << std::endl;
            std::cout << "│" << std::endl;
            std::cout << "│ " << BOLD << "Recommended Actions:" << RESET << std::endl;
            if (!daily_aligned) {
                std::cout << "│   " << RED << "• Download missing daily data files" << RESET << std::endl;
            }
            if (!minute_aligned) {
                std::cout << "│   " << RED << "• Download missing minute data files" << RESET << std::endl;
            }
            std::cout << "│   " << CYAN << "• Use: ./build/sentio_cli download <SYMBOL> --period 3y" << RESET << std::endl;
        }
        
        std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
        
        return 0;
        
    } else if (command == "list-strategies") {
        if (args.help_requested) {
            sentio::CLIHelpers::print_help("list-strategies",
                "sentio_cli list-strategies [options]",
                {
                    "--format <format>          Output format: table|list|json (default: table)",
                    "--category <cat>           Filter by category: momentum|mean-reversion|ml|all (default: all)",
                    "--verbose                  Show detailed strategy information"
                },
                {
                    "sentio_cli list-strategies",
                    "sentio_cli list-strategies --format list",
                    "sentio_cli list-strategies --category ml --verbose"
                });
            return 0;
        }
        
        // Validate command options
        std::vector<std::string> allowed_options = {"format", "category"};
        std::vector<std::string> allowed_flags = {"verbose"};
        
        if (!sentio::CLIHelpers::validate_options(args, "list-strategies", allowed_options, allowed_flags)) {
            return 1;
        }
        
        // Get available strategies from factory
        auto& factory = sentio::StrategyFactory::instance();
        auto strategies = factory.get_available_strategies();
        
        if (strategies.empty()) {
            std::cout << "❌ No strategies available" << std::endl;
            return 1;
        }
        
        // Color constants
        const std::string BOLD = "\033[1m";
        const std::string CYAN = "\033[36m";
        const std::string GREEN = "\033[32m";
        const std::string BLUE = "\033[34m";
        const std::string RESET = "\033[0m";
        
        std::string format = sentio::CLIHelpers::get_option(args, "format", "table");
        std::string category = sentio::CLIHelpers::get_option(args, "category", "all");
        bool verbose = sentio::CLIHelpers::get_flag(args, "verbose");
        
        // Strategy categorization (based on common knowledge)
        std::unordered_map<std::string, std::vector<std::string>> strategy_categories = {
            {"momentum", {"ire", "bollinger_squeeze_breakout"}},
            {"mean-reversion", {"rsi"}},
            {"ml", {"tfa", "kochi_ppo"}},
            {"signal", {"signal_or"}}
        };
        
        // Filter strategies by category if specified
        std::vector<std::string> filtered_strategies = strategies;
        if (category != "all") {
            auto it = strategy_categories.find(category);
            if (it != strategy_categories.end()) {
                filtered_strategies.clear();
                for (const auto& strategy : strategies) {
                    if (std::find(it->second.begin(), it->second.end(), strategy) != it->second.end()) {
                        filtered_strategies.push_back(strategy);
                    }
                }
            }
        }
        
        if (format == "json") {
            std::cout << "{\n";
            std::cout << "  \"strategies\": [\n";
            for (size_t i = 0; i < filtered_strategies.size(); ++i) {
                std::cout << "    \"" << filtered_strategies[i] << "\"";
                if (i < filtered_strategies.size() - 1) std::cout << ",";
                std::cout << "\n";
            }
            std::cout << "  ],\n";
            std::cout << "  \"total\": " << filtered_strategies.size() << ",\n";
            std::cout << "  \"category\": \"" << category << "\"\n";
            std::cout << "}\n";
            
        } else if (format == "list") {
            std::cout << "📋 Available Strategies (" << filtered_strategies.size() << "):\n";
            for (const auto& strategy : filtered_strategies) {
                std::cout << "  • " << strategy << "\n";
            }
            
        } else { // table format (default)
            std::cout << "\n" << BOLD << CYAN << "📋 AVAILABLE STRATEGIES" << RESET << std::endl;
            std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
            std::cout << "│ " << BOLD << "Strategy Name" << RESET << "                    │ " << BOLD << "Category" << RESET << "        │ " << BOLD << "Description" << RESET << "                    │" << std::endl;
            std::cout << "├─────────────────────────────────────────────────────────────────────────────────┤" << std::endl;
            
            for (const auto& strategy : filtered_strategies) {
                std::string category_name = "Other";
                std::string description = "Trading strategy";
                
                // Determine category and description
                if (strategy == "ire") {
                    category_name = "Momentum";
                    description = "Intelligent Regime Engine";
                } else if (strategy == "bollinger_squeeze_breakout") {
                    category_name = "Momentum";
                    description = "Bollinger Band breakout";
                } else if (strategy == "rsi") {
                    category_name = "Mean Reversion";
                    description = "RSI-based reversion";
                } else if (strategy == "tfa") {
                    category_name = "Machine Learning";
                    description = "Transformer-based forecasting";
                } else if (strategy == "kochi_ppo") {
                    category_name = "Machine Learning";
                    description = "PPO reinforcement learning";
                } else if (strategy == "signal_or") {
                    category_name = "Signal";
                    description = "Signal combination logic";
                }
                
                printf("│ %-30s │ %-15s │ %-30s │\n", 
                       strategy.c_str(), category_name.c_str(), description.c_str());
            }
            
            std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
            
            if (verbose) {
                std::cout << "\n" << BOLD << CYAN << "📖 STRATEGY DETAILS" << RESET << std::endl;
                std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
                
                for (const auto& strategy : filtered_strategies) {
                    std::cout << "│ " << BOLD << BLUE << strategy << RESET << std::endl;
                    
                    if (strategy == "ire") {
                        std::cout << "│   📊 Intelligent Regime Engine - Advanced momentum strategy" << std::endl;
                        std::cout << "│   🎯 Uses regime detection and adaptive parameters" << std::endl;
                        std::cout << "│   💡 Best for: Trending markets, volatile conditions" << std::endl;
                    } else if (strategy == "bollinger_squeeze_breakout") {
                        std::cout << "│   📊 Bollinger Band Squeeze Breakout strategy" << std::endl;
                        std::cout << "│   🎯 Detects low volatility periods and trades breakouts" << std::endl;
                        std::cout << "│   💡 Best for: Range-bound markets, volatility expansion" << std::endl;
                    } else if (strategy == "rsi") {
                        std::cout << "│   📊 RSI Mean Reversion strategy" << std::endl;
                        std::cout << "│   🎯 Uses RSI overbought/oversold signals" << std::endl;
                        std::cout << "│   💡 Best for: Range-bound markets, contrarian trading" << std::endl;
                    } else if (strategy == "tfa") {
                        std::cout << "│   🤖 Transformer-based Forecasting Algorithm" << std::endl;
                        std::cout << "│   🎯 Uses deep learning for price prediction" << std::endl;
                        std::cout << "│   💡 Best for: Complex patterns, adaptive to market changes" << std::endl;
                    } else if (strategy == "kochi_ppo") {
                        std::cout << "│   🤖 Kochi PPO Reinforcement Learning strategy" << std::endl;
                        std::cout << "│   🎯 Uses Proximal Policy Optimization for trading decisions" << std::endl;
                        std::cout << "│   💡 Best for: Adaptive learning, complex market dynamics" << std::endl;
                    } else if (strategy == "signal_or") {
                        std::cout << "│   🔗 Signal OR combination strategy" << std::endl;
                        std::cout << "│   🎯 Combines multiple signal sources with OR logic" << std::endl;
                        std::cout << "│   💡 Best for: Signal aggregation, multi-strategy approaches" << std::endl;
                    }
                    
                    std::cout << "│" << std::endl;
                }
                
                std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
            }
        }
        
        std::cout << "\n" << BOLD << GREEN << "💡 Usage Examples:" << RESET << std::endl;
        std::cout << "  " << CYAN << "• sentio_cli strattest ire --mode simulation --blocks 10" << RESET << std::endl;
        std::cout << "  " << CYAN << "• sentio_cli strattest tfa --mode historical --blocks 20" << RESET << std::endl;
        std::cout << "  " << CYAN << "• sentio_cli strattest rsi --mode ai-regime --regime volatile" << RESET << std::endl;
        
        return 0;
        
    } else if (command == "audit-validate") {
        std::cout << "🔍 **STRATEGY-AGNOSTIC AUDIT VALIDATION**" << std::endl;
        std::cout << "⚠️  Audit validation feature removed during cleanup" << std::endl;
        return 0;
        
    } else if (command == "download") {
        if (args.help_requested) {
            sentio::CLIHelpers::print_help("download",
                "sentio_cli download <symbol> [options]",
                {
                    "--period <period>          Time period: 1y, 6m, 3m, 1m, 2w, 5d (default: 3y)",
                    "--timespan <span>          Data resolution: day|hour|minute (default: minute)",
                    "--holidays                 Include market holidays (default: exclude)",
                    "--output <dir>             Output directory (default: data/equities/)",
                    "--family                   Download symbol family (QQQ -> QQQ,TQQQ,SQQQ)",
                    "--force                    Overwrite existing files"
                },
                {
                    "sentio_cli download QQQ --period 3y",
                    "sentio_cli download SPY --period 1y --timespan day",
                    "sentio_cli download QQQ --family --period 6m"
                });
            return 0;
        }
        
        if (!sentio::CLIHelpers::validate_required_args(args, 1,
            "sentio_cli download <symbol> [options]")) {
            return 1;
        }
        
        std::string symbol = args.positional_args[0];
        
        // Validate symbol
        if (!sentio::CLIHelpers::is_valid_symbol(symbol)) {
            sentio::CLIHelpers::print_error("Invalid symbol: " + symbol);
            return 1;
        }
        
        // Parse options
        std::string period = sentio::CLIHelpers::get_option(args, "period", "3y");
        std::string timespan = sentio::CLIHelpers::get_option(args, "timespan", "minute");
        bool include_holidays = sentio::CLIHelpers::get_flag(args, "holidays");
        std::string output_dir = sentio::CLIHelpers::get_option(args, "output", "data/equities/");
        bool download_family = sentio::CLIHelpers::get_flag(args, "family");
        // bool force_overwrite = sentio::CLIHelpers::get_flag(args, "force"); // TODO: Implement force overwrite
        
        // Build symbol list
        std::vector<std::string> symbols_to_download = {symbol};
        if (download_family && symbol == "QQQ") {
            symbols_to_download.push_back("TQQQ");
            symbols_to_download.push_back("SQQQ");
        }
        
        // Convert period to days
        int days = sentio::CLIHelpers::parse_period_to_days(period);
        
        std::cout << "📥 Downloading data for: ";
        for (const auto& sym : symbols_to_download) {
            std::cout << sym << " ";
        }
        std::cout << std::endl;
        std::cout << "⏱️  Period: " << period << " (" << days << " days)" << std::endl;
        std::cout << "📊 Timespan: " << timespan << std::endl;
        std::cout << "🏖️  Holidays: " << (include_holidays ? "included" : "excluded") << std::endl;
        
        try {
            for (const auto& sym : symbols_to_download) {
                std::cout << "\n📈 Downloading " << sym << "..." << std::endl;
                
                bool success = sentio::download_symbol_data(
                    sym, 0, 0, days, timespan, 1, !include_holidays, output_dir
                );
                
                if (success) {
                    std::cout << "✅ " << sym << " downloaded successfully" << std::endl;
                } else {
                    std::cout << "❌ Failed to download " << sym << std::endl;
                    return 1;
                }
            }
            
            std::cout << "\n🎉 All downloads completed successfully!" << std::endl;
            return 0;
            
        } catch (const std::exception& e) {
            std::cerr << "Error downloading data: " << e.what() << std::endl;
            return 1;
        }
        
    } else {
        usage();
        return 1;
    }
    
    return 0;
}

```

## 📄 **FILE 159 of 180**: src/mars_data_loader.cpp

**File Information**:
- **Path**: `src/mars_data_loader.cpp`

- **Size**: 164 lines
- **Modified**: 2025-09-13 15:17:48

- **Type**: .cpp

```text
#include "sentio/mars_data_loader.hpp"
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <sstream>

namespace sentio {

std::vector<MarsDataLoader::MarsBar> MarsDataLoader::load_from_json(const std::string& filename) {
    std::vector<MarsBar> bars;
    
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return bars;
        }
        
        nlohmann::json json_data;
        file >> json_data;
        
        for (const auto& item : json_data) {
            if (item.is_object() && !item.empty()) {
                MarsBar bar;
                bar.timestamp = item["timestamp"];
                bar.open = item["open"];
                bar.high = item["high"];
                bar.low = item["low"];
                bar.close = item["close"];
                bar.volume = item["volume"];
                bar.symbol = item["symbol"];
                bars.push_back(bar);
            }
        }
        
        std::cout << "✅ Loaded " << bars.size() << " bars from MarS data" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading MarS data: " << e.what() << std::endl;
    }
    
    return bars;
}

Bar MarsDataLoader::convert_to_bar(const MarsBar& mars_bar) {
    Bar bar;
    bar.ts_utc_epoch = mars_bar.timestamp;
    bar.open = mars_bar.open;
    bar.high = mars_bar.high;
    bar.low = mars_bar.low;
    bar.close = mars_bar.close;
    bar.volume = mars_bar.volume;
    return bar;
}

std::vector<Bar> MarsDataLoader::convert_to_bars(const std::vector<MarsBar>& mars_bars) {
    std::vector<Bar> bars;
    bars.reserve(mars_bars.size());
    
    for (const auto& mars_bar : mars_bars) {
        bars.push_back(convert_to_bar(mars_bar));
    }
    
    return bars;
}

bool MarsDataLoader::generate_mars_data(const std::string& symbol,
                                       int duration_minutes,
                                       int bar_interval_seconds,
                                       int num_simulations,
                                       const std::string& market_regime,
                                       const std::string& output_file) {
    
    // Construct Python command
    std::stringstream cmd;
    cmd << "python3 tools/mars_bridge.py"
        << " --symbol " << symbol
        << " --duration " << duration_minutes
        << " --interval " << bar_interval_seconds
        << " --simulations " << num_simulations
        << " --regime " << market_regime
        << " --output " << output_file
        << " --quiet";
    
    // Suppress verbose command output
    
    return execute_python_command(cmd.str());
}

bool MarsDataLoader::generate_fast_historical_data(const std::string& symbol,
                                                  const std::string& historical_data_file,
                                                  int continuation_minutes,
                                                  const std::string& output_file) {
    
    // Construct Python command for fast historical bridge
    std::stringstream cmd;
    cmd << "python3 tools/fast_historical_bridge.py"
        << " --symbol " << symbol
        << " --historical-data " << historical_data_file
        << " --continuation-minutes " << continuation_minutes
        << " --output " << output_file
        << " --quiet";
    
    // Suppress verbose command output
    
    return execute_python_command(cmd.str());
}

std::vector<Bar> MarsDataLoader::load_mars_data(const std::string& symbol,
                                               int duration_minutes,
                                               int bar_interval_seconds,
                                               int num_simulations,
                                               const std::string& market_regime) {
    
    // Generate temporary filename
    std::string temp_file = "temp_mars_data_" + symbol + ".json";
    
    // Generate MarS data
    if (!generate_mars_data(symbol, duration_minutes, bar_interval_seconds, 
                           num_simulations, market_regime, temp_file)) {
        std::cerr << "Failed to generate MarS data" << std::endl;
        return {};
    }
    
    // Load and convert data
    auto mars_bars = load_from_json(temp_file);
    auto bars = convert_to_bars(mars_bars);
    
    // Clean up temporary file
    std::filesystem::remove(temp_file);
    
    return bars;
}

std::vector<Bar> MarsDataLoader::load_fast_historical_data(const std::string& symbol,
                                                          const std::string& historical_data_file,
                                                          int continuation_minutes) {
    
    // Generate temporary filename
    std::string temp_file = "temp_fast_historical_" + symbol + ".json";
    
    // Generate fast historical data
    if (!generate_fast_historical_data(symbol, historical_data_file, 
                                      continuation_minutes, temp_file)) {
        std::cerr << "Failed to generate fast historical data" << std::endl;
        return {};
    }
    
    // Load and convert data
    auto mars_bars = load_from_json(temp_file);
    auto bars = convert_to_bars(mars_bars);
    
    // Clean up temporary file
    std::filesystem::remove(temp_file);
    
    return bars;
}

bool MarsDataLoader::execute_python_command(const std::string& command) {
    int result = std::system(command.c_str());
    return result == 0;
}

} // namespace sentio

```

## 📄 **FILE 160 of 180**: src/ml/model_registry_ts.cpp

**File Information**:
- **Path**: `src/ml/model_registry_ts.cpp`

- **Size**: 71 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .cpp

```text
#include "sentio/ml/model_registry.hpp"
#include "sentio/ml/ts_model.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace sentio::ml {

static std::string slurp(const std::string& path){
  std::ifstream f(path); if (!f) throw std::runtime_error("cannot open "+path);
  std::ostringstream ss; ss<<f.rdbuf(); return ss.str();
}
static bool find_val(const std::string& j, const std::string& key, std::string& out){
  auto k="\""+key+"\""; auto p=j.find(k); if (p==std::string::npos) return false;
  p=j.find(':',p); if (p==std::string::npos) return false; ++p;
  while (p<j.size() && isspace((unsigned char)j[p])) ++p;
  if (j[p]=='"'){ auto e=j.find('"',p+1); out=j.substr(p+1,e-(p+1)); return true; }
  auto e=j.find_first_of(",}\n",p); out=j.substr(p,e-p); return true;
}
static std::vector<std::string> parse_sarr(const std::string& j, const std::string& key){
  std::vector<std::string> v; auto k="\""+key+"\""; auto p=j.find(k); if (p==std::string::npos) return v;
  p=j.find('[',p); auto e=j.find(']',p); if (p==std::string::npos||e==std::string::npos) return v;
  auto s=j.substr(p+1,e-(p+1)); size_t i=0;
  while (i<s.size()){ auto q1=s.find('"',i); if (q1==std::string::npos) break; auto q2=s.find('"',q1+1);
    v.push_back(s.substr(q1+1,q2-(q1+1))); i=q2+1; }
  return v;
}
static std::vector<double> parse_darr(const std::string& j, const std::string& key){
  std::vector<double> v; auto k="\""+key+"\""; auto p=j.find(k); if (p==std::string::npos) return v;
  p=j.find('[',p); auto e=j.find(']',p); if (p==std::string::npos||e==std::string::npos) return v;
  auto s=j.substr(p+1,e-(p+1)); size_t i=0;
  while (i<s.size()){ auto j2=s.find_first_of(", \t\n", i);
    auto tok=s.substr(i,(j2==std::string::npos)?std::string::npos:(j2-i));
    if (!tok.empty()) v.push_back(std::stod(tok));
    if (j2==std::string::npos) break; i=j2+1; }
  return v;
}

ModelHandle ModelRegistryTS::load_torchscript(const std::string& model_id,
                                              const std::string& version,
                                              const std::string& artifacts_dir,
                                              bool use_cuda)
{
  const std::string base = artifacts_dir + "/" + model_id + "/" + version + "/";
  const std::string meta_path = base + "metadata.json";
  const std::string pt_path   = base + "model.pt";

  auto js = slurp(meta_path);

  ModelSpec spec;
  spec.model_id = model_id;
  spec.version  = version;
  spec.feature_names = parse_sarr(js, "feature_names");
  spec.mean = parse_darr(js, "mean");
  spec.std  = parse_darr(js, "std");
  auto clip = parse_darr(js, "clip"); if (clip.size()==2) spec.clip2 = clip;
  spec.actions = parse_sarr(js, "actions");

  std::string t; if (find_val(js, "expected_bar_spacing_sec", t)) spec.expected_spacing_sec = std::stoi(t);
  if (find_val(js, "seq_len", t)) spec.seq_len = std::stoi(t);
  std::string layout; if (find_val(js, "input_layout", layout)) spec.input_layout = layout;
  std::string fmt; if (find_val(js, "format", fmt)) spec.format = fmt; else spec.format="torchscript";

  ModelHandle h;
  h.spec = spec;
  h.model = TorchScriptModel::load(pt_path, h.spec, use_cuda);
  return h;
}

} // namespace sentio::ml

```

## 📄 **FILE 161 of 180**: src/ml/ts_model.cpp

**File Information**:
- **Path**: `src/ml/ts_model.cpp`

- **Size**: 81 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .cpp

```text
#include "sentio/ml/ts_model.hpp"
#include <torch/script.h>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace sentio::ml {

TorchScriptModel::TorchScriptModel(ModelSpec spec) : spec_(std::move(spec)), input_tensor_(nullptr) {}

TorchScriptModel::~TorchScriptModel() {
  if (input_tensor_) {
    delete static_cast<torch::Tensor*>(input_tensor_);
  }
}

std::unique_ptr<TorchScriptModel> TorchScriptModel::load(const std::string& pt_path,
                                                         const ModelSpec& spec,
                                                         [[maybe_unused]] bool use_cuda)
{
  auto m = std::unique_ptr<TorchScriptModel>(new TorchScriptModel(spec));
  torch::jit::script::Module mod = torch::jit::load(pt_path);
  mod.eval();
  m->mod_ = std::make_shared<torch::jit::script::Module>(std::move(mod));
  // Disable CUDA for now - CPU-only build
  m->cuda_ = false; // use_cuda && torch::cuda::is_available();
  if (m->cuda_) m->mod_->to(torch::kCUDA);
  // Defer concrete shape until first predict (we need T,F,layout)
  m->in_shape_.clear();
  return m;
}

std::optional<ModelOutput> TorchScriptModel::predict(const std::vector<float>& feats,
                                                     int T, int F, const std::string& layout) const
{
  if (T<=0 || F<=0) return std::nullopt;
  const size_t need = (layout=="BF")? feats.size() : size_t(T)*size_t(F);
  if (feats.size() != need) return std::nullopt;

  torch::NoGradGuard ng; torch::InferenceMode im;

  // Pre-allocate input tensor if needed (persistent tensor approach)
  std::vector<int64_t> need_shape = (layout=="BTF") ? std::vector<int64_t>{1,T,F}
                                                    : std::vector<int64_t>{1,(int64_t)feats.size()};
  
  if (!input_tensor_ || in_shape_ != need_shape) {
    in_shape_ = need_shape;
    if (input_tensor_) {
      delete static_cast<torch::Tensor*>(input_tensor_);
    }
    input_tensor_ = new torch::Tensor(torch::empty(in_shape_, torch::TensorOptions().dtype(torch::kFloat32).device(cuda_?torch::kCUDA:torch::kCPU)));
  }

  // memcpy into persistent tensor (no allocation, no clone)
  torch::Tensor& x = *static_cast<torch::Tensor*>(input_tensor_);
  if (cuda_) {
    // For CUDA, copy via CPU tensor to avoid sync issues
    torch::Tensor host = torch::from_blob((void*)feats.data(), {(int64_t)feats.size()}, torch::kFloat32);
    x.view({-1}).copy_(host);
  } else {
    // For CPU, direct memcpy into tensor data
    std::memcpy(x.data_ptr<float>(), feats.data(), feats.size() * sizeof(float));
  }

  std::vector<torch::jit::IValue> inputs; inputs.emplace_back(x);
  torch::Tensor out = mod_->forward(inputs).toTensor().to(torch::kCPU).contiguous();
  if (out.dim()==2 && out.size(0)==1) out = out.squeeze(0);
  if (out.dim()!=1) return std::nullopt;

  ModelOutput mo; mo.probs.resize(out.numel());
  std::memcpy(mo.probs.data(), out.data_ptr<float>(), mo.probs.size()*sizeof(float));
  float sum=0.f; for (float v: mo.probs) sum += std::isfinite(v)? v : 0.f;
  if (!(sum>0.f)) {
    float mv=*std::max_element(mo.probs.begin(), mo.probs.end());
    float s=0; for (auto& v: mo.probs){ v=std::exp(v-mv); s+=v; } if (s>0) for (auto& v: mo.probs) v/=s;
  } else for (auto& v: mo.probs) v = std::max(0.f, v/sum);
  return mo;
}

} // namespace sentio::ml

```

## 📄 **FILE 162 of 180**: src/pnl_accounting.cpp

**File Information**:
- **Path**: `src/pnl_accounting.cpp`

- **Size**: 62 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .cpp

```text
// src/pnl_accounting.cpp
#include "sentio/pnl_accounting.hpp"
#include <shared_mutex>
#include <unordered_map>
#include <utility>

namespace sentio {

namespace {
// Simple in-TU storage for latest bars keyed by instrument.
// If your header already declares members, remove this and
// implement against those members instead.
struct PriceBookStorage {
    std::unordered_map<std::string, Bar> latest;
    mutable std::shared_mutex mtx;
};

// One global storage per process for the default PriceBook()
// If your PriceBook is an instance with its own fields, move this inside the class.
static PriceBookStorage& storage() {
    static PriceBookStorage s;
    return s;
}
} // anonymous namespace

// ---- Interface implementation ----

const Bar* PriceBook::get_latest(const std::string& instrument) const {
    auto& S = storage();
    std::shared_lock lk(S.mtx);
    auto it = S.latest.find(instrument);
    if (it == S.latest.end()) return nullptr;
    return &it->second; // pointer remains valid as long as map bucket survives
}

// ---- Additional helper methods (not declared in header but useful) ----

void PriceBook::upsert_latest(const std::string& instrument, const Bar& b) {
    auto& S = storage();
    std::unique_lock lk(S.mtx);
    S.latest[instrument] = b;
}

bool PriceBook::has_instrument(const std::string& instrument) const {
    auto& S = storage();
    std::shared_lock lk(S.mtx);
    return S.latest.find(instrument) != S.latest.end();
}

std::size_t PriceBook::size() const {
    auto& S = storage();
    std::shared_lock lk(S.mtx);
    return S.latest.size();
}

double last_trade_price(const PriceBook& book, const std::string& instrument) {
    auto* b = book.get_latest(instrument);
    if (!b) throw std::runtime_error("No bar for instrument: " + instrument);
    return b->close;
}

} // namespace sentio

```

## 📄 **FILE 163 of 180**: src/poly_fetch_main.cpp

**File Information**:
- **Path**: `src/poly_fetch_main.cpp`

- **Size**: 151 lines
- **Modified**: 2025-09-16 20:14:43

- **Type**: .cpp

```text
#include "sentio/polygon_client.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>

using namespace sentio;

std::string get_yesterday_date() {
    std::time_t now = std::time(nullptr);
    std::time_t yesterday = now - 24 * 60 * 60; // Subtract 1 day in seconds
    
    std::tm* tm_yesterday = std::gmtime(&yesterday);
    std::ostringstream oss;
    oss << std::put_time(tm_yesterday, "%Y-%m-%d");
    return oss.str();
}

std::string get_current_date() {
    std::time_t now = std::time(nullptr);
    std::tm* tm_now = std::gmtime(&now);
    std::ostringstream oss;
    oss << std::put_time(tm_now, "%Y-%m-%d");
    return oss.str();
}

std::string calculate_start_date(int years, int months, int days) {
    std::time_t now = std::time(nullptr);
    std::time_t yesterday = now - 24 * 60 * 60; // Start from yesterday
    
    std::tm* tm_start = std::gmtime(&yesterday);
    
    if (years > 0) {
        tm_start->tm_year -= years;
    } else if (months > 0) {
        tm_start->tm_mon -= months;
        if (tm_start->tm_mon < 0) {
            tm_start->tm_mon += 12;
            tm_start->tm_year--;
        }
    } else if (days > 0) {
        tm_start->tm_mday -= days;
        // Let mktime handle month/year overflow
        std::mktime(tm_start);
    } else {
        // Default: 3 years
        tm_start->tm_year -= 3;
    }
    
    std::ostringstream oss;
    oss << std::put_time(tm_start, "%Y-%m-%d");
    return oss.str();
}

int main(int argc,char**argv){
  if(argc<3){
    std::cerr<<"Usage: poly_fetch FAMILY outdir [--years N] [--months N] [--days N] [--timespan day|hour|minute] [--multiplier N] [--symbols SYM1,SYM2,...] [--no-holidays] [--rth-only]\n";
    std::cerr<<"       poly_fetch FAMILY from to outdir [--timespan day|hour|minute] [--multiplier N] [--symbols SYM1,SYM2,...] [--no-holidays] [--rth-only]\n";
    std::cerr<<"Examples:\n";
    std::cerr<<"  poly_fetch qqq data/equities --years 3 --no-holidays --rth-only\n";
    std::cerr<<"  poly_fetch qqq 2022-01-01 2025-01-10 data/equities --timespan minute --rth-only\n";
    return 1;
  }
  
  std::string fam=argv[1];
  std::string from, to, outdir;
  
  // Check if we're using time range options (new format) or explicit dates (old format)
  bool use_time_range = false;
  int years = 0, months = 0, days = 0;
  
  if (argc >= 3) {
    // Check if second argument is a directory (new format) or a date (old format)
    std::string second_arg = argv[2];
    if (second_arg.find('/') != std::string::npos || second_arg == "data" || second_arg == "data/equities") {
      // New format: FAMILY outdir [time options]
      outdir = second_arg;
      use_time_range = true;
    } else if (argc >= 5) {
      // Old format: FAMILY from to outdir
      from = argv[2];
      to = argv[3];
      outdir = argv[4];
    } else {
      std::cerr<<"Error: Invalid arguments. Use --help for usage.\n";
      return 1;
    }
  }
  
  std::string timespan = "day";
  int multiplier = 1;
  std::string symbols_csv;
  bool exclude_holidays=false;
  bool rth_only=false;
  
  int start_idx = use_time_range ? 3 : 5;
  for (int i=start_idx;i<argc;i++) {
    std::string a = argv[i];
    if (a=="--years" && i+1<argc) { years = std::stoi(argv[++i]); }
    else if (a=="--months" && i+1<argc) { months = std::stoi(argv[++i]); }
    else if (a=="--days" && i+1<argc) { days = std::stoi(argv[++i]); }
    else if ((a=="--timespan" || a=="-t") && i+1<argc) { timespan = argv[++i]; }
    else if ((a=="--multiplier" || a=="-m") && i+1<argc) { multiplier = std::stoi(argv[++i]); }
    else if (a=="--symbols" && i+1<argc) { symbols_csv = argv[++i]; }
    else if (a=="--rth-only") { rth_only=true; }
    else if (a=="--no-holidays") { exclude_holidays=true; }
  }
  
  // Calculate dates if using time range options
  if (use_time_range) {
    from = calculate_start_date(years, months, days);
    to = get_yesterday_date();
    std::cerr<<"Current date: " << get_current_date() << "\n";
    std::cerr<<"Downloading " << (years > 0 ? std::to_string(years) + " years" : 
                                  months > 0 ? std::to_string(months) + " months" : 
                                  days > 0 ? std::to_string(days) + " days" : "3 years (default)") 
             << " of data: " << from << " to " << to << "\n";
  }
  const char* key = std::getenv("POLYGON_API_KEY");
  std::string api_key = key? key: "";
  PolygonClient cli(api_key);

  std::vector<std::string> syms;
  if(fam=="qqq") syms={"QQQ","TQQQ","SQQQ"};
  else if(fam=="bitcoin") syms={"X:BTCUSD","X:ETHUSD"};
  else if(fam=="tesla") syms={"TSLA","TSLQ"};
  else if(fam=="custom") {
    if (symbols_csv.empty()) { std::cerr<<"--symbols required for custom family\n"; return 1; }
    size_t start=0; while (start < symbols_csv.size()) {
      size_t pos = symbols_csv.find(',', start);
      std::string tok = (pos==std::string::npos)? symbols_csv.substr(start) : symbols_csv.substr(start, pos-start);
      if (!tok.empty()) syms.push_back(tok);
      if (pos==std::string::npos) break; else start = pos+1;
    }
  } else { std::cerr<<"Unknown family\n"; return 1; }

  for(auto&s:syms){
    AggsQuery q; q.symbol=s; q.from=from; q.to=to; q.timespan=timespan; q.multiplier=multiplier; q.adjusted=true; q.sort="asc";
    auto bars=cli.get_aggs_all(q);
    std::string suffix;
    if (rth_only) suffix += "_RTH";
    if (exclude_holidays) suffix += "_NH";
    std::string fname= outdir + "/" + s + suffix + ".csv";
    cli.write_csv(fname,s,bars,exclude_holidays,rth_only);
    std::cerr<<"Wrote "<<bars.size()<<" bars -> "<<fname<<"\n";
  }
}


```

## 📄 **FILE 164 of 180**: src/polygon_client.cpp

**File Information**:
- **Path**: `src/polygon_client.cpp`

- **Size**: 305 lines
- **Modified**: 2025-09-16 23:16:46

- **Type**: .cpp

```text
#include "sentio/polygon_client.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <cctz/time_zone.h>
#include <cctz/civil_time.h>
#include <fstream>
#include <thread>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>

using json = nlohmann::json;
namespace sentio {

static size_t write_cb(void* contents, size_t size, size_t nmemb, void* userp) {
  size_t total = size * nmemb;
  std::string* s = static_cast<std::string*>(userp);
  s->append(static_cast<char*>(contents), total);
  return total;
}

static std::string rfc3339_utc_from_epoch_ms(long long ms) {
  using namespace std::chrono;
  
  cctz::time_point<cctz::seconds> tp{cctz::seconds{ms / 1000}};
  
  // Get UTC timezone
  cctz::time_zone utc_tz;
  if (!cctz::load_time_zone("UTC", &utc_tz)) {
    return "1970-01-01T00:00:00Z"; // fallback
  }
  
  // Convert to UTC civil time
  auto lt = cctz::convert(tp, utc_tz);
  auto ct = cctz::civil_second(lt);
  
  std::ostringstream oss;
  oss << std::setfill('0') 
      << std::setw(4) << ct.year() << "-"
      << std::setw(2) << ct.month() << "-"
      << std::setw(2) << ct.day() << "T"
      << std::setw(2) << ct.hour() << ":"
      << std::setw(2) << ct.minute() << ":"
      << std::setw(2) << ct.second() << "Z";
  
  return oss.str();
}

// **REMOVED**: RTH filtering - now keeping all trading hours data
// Only holiday filtering remains active

// **NEW**: Holiday check
static bool is_us_market_holiday_utc(int year, int month, int day) {
  // Simple holiday check for common US market holidays
  // This is a simplified version - for production use, integrate with the full calendar system
  
  // New Year's Day (observed)
  if (month == 1 && day == 1) return true;
  if (month == 1 && day == 2) return true; // observed if Jan 1 is Sunday
  
  // MLK Day (3rd Monday in January)
  if (month == 1 && day >= 15 && day <= 21) {
    // Simple check - this could be more precise
    return true;
  }
  
  // Presidents Day (3rd Monday in February)
  if (month == 2 && day >= 15 && day <= 21) {
    return true;
  }
  
  // Good Friday (varies by year)
  if (year == 2022 && month == 4 && day == 15) return true;
  if (year == 2023 && month == 4 && day == 7) return true;
  if (year == 2024 && month == 3 && day == 29) return true;
  if (year == 2025 && month == 4 && day == 18) return true;
  
  // Memorial Day (last Monday in May)
  if (month == 5 && day >= 25 && day <= 31) {
    return true;
  }
  
  // Juneteenth (observed)
  if (month == 6 && day == 19) return true;
  if (month == 6 && day == 20) return true; // observed if Jun 19 is Sunday
  
  // Independence Day (observed)
  if (month == 7 && day == 4) return true;
  if (month == 7 && day == 5) return true; // observed if Jul 4 is Sunday
  
  // Labor Day (1st Monday in September)
  if (month == 9 && day >= 1 && day <= 7) {
    return true;
  }
  
  // Thanksgiving (4th Thursday in November)
  if (month == 11 && day >= 22 && day <= 28) {
    return true;
  }
  
  // Christmas (observed)
  if (month == 12 && day == 25) return true;
  if (month == 12 && day == 26) return true; // observed if Dec 25 is Sunday
  
  return false;
}

PolygonClient::PolygonClient(std::string api_key) : api_key_(std::move(api_key)) {}

std::string PolygonClient::get_(const std::string& url) {
    CURL* curl = curl_easy_init();
    std::string buffer;
    if (!curl) return buffer;
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    struct curl_slist* headers = nullptr;
    std::string auth = "Authorization: Bearer " + api_key_;
    headers = curl_slist_append(headers, auth.c_str());
    headers = curl_slist_append(headers, "Accept: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return buffer;
}

std::vector<AggBar> PolygonClient::get_aggs_all(const AggsQuery& q, int max_pages) {
    std::vector<AggBar> out;
    
    // For minute data with large date ranges, chunk into smaller periods
    if (q.timespan == "minute") {
        return get_aggs_chunked(q, max_pages);
    }
    
    std::string base = "https://api.polygon.io/v2/aggs/ticker/" + q.symbol + "/range/" + std::to_string(q.multiplier) + "/" + q.timespan + "/" + q.from + "/" + q.to + "?adjusted=" + (q.adjusted?"true":"false") + "&sort=" + q.sort + "&limit=" + std::to_string(q.limit);
    std::string url = base;
    
    for (int page=0; page<max_pages; ++page) {
        std::string body = get_(url);
        if (body.empty()) break;
        
        auto j = json::parse(body, nullptr, false);
        if (j.is_discarded()) break;
        
        if (j.contains("results")) {
            for (auto& r : j["results"]) {
                out.push_back({r.value("t", 0LL), r.value("o", 0.0), r.value("h", 0.0), r.value("l", 0.0), r.value("c", 0.0), r.value("v", 0.0)});
            }
        }
        
        if (!j.contains("next_url")) break;
        url = j["next_url"].get<std::string>();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    return out;
}

std::vector<AggBar> PolygonClient::get_aggs_chunked(const AggsQuery& q, int max_pages) {
    std::vector<AggBar> out;
    
    // Parse start and end dates
    std::tm start_tm = {}, end_tm = {};
    std::istringstream start_ss(q.from), end_ss(q.to);
    start_ss >> std::get_time(&start_tm, "%Y-%m-%d");
    end_ss >> std::get_time(&end_tm, "%Y-%m-%d");
    
    if (start_ss.fail() || end_ss.fail()) {
        std::cerr << "Error: Invalid date format. Use YYYY-MM-DD" << std::endl;
        return out;
    }
    
    std::time_t start_time = std::mktime(&start_tm);
    std::time_t end_time = std::mktime(&end_tm);
    
    // Chunk by 30 days to avoid large responses
    const int chunk_days = 30;
    std::time_t current_time = start_time;
    
    while (current_time < end_time) {
        std::time_t chunk_end = current_time + (chunk_days * 24 * 60 * 60);
        if (chunk_end > end_time) chunk_end = end_time;
        
        // Convert back to date strings
        std::tm* current_tm = std::gmtime(&current_time);
        std::tm* chunk_end_tm = std::gmtime(&chunk_end);
        
        char start_str[32], end_str[32];
        std::strftime(start_str, sizeof(start_str), "%Y-%m-%d", current_tm);
        std::strftime(end_str, sizeof(end_str), "%Y-%m-%d", chunk_end_tm);
        
        // Skip if start and end are the same date (avoid duplicate chunks)
        if (std::string(start_str) == std::string(end_str)) {
            current_time = chunk_end + (24 * 60 * 60); // Move to next day properly
            continue;
        }
        
        std::cerr << "Downloading chunk: " << start_str << " to " << end_str << std::endl;
        
        // Create chunk query
        AggsQuery chunk_q = q;
        chunk_q.from = start_str;
        chunk_q.to = end_str;
        
        // Get data for this chunk
        std::string base = "https://api.polygon.io/v2/aggs/ticker/" + chunk_q.symbol + "/range/" + std::to_string(chunk_q.multiplier) + "/" + chunk_q.timespan + "/" + chunk_q.from + "/" + chunk_q.to + "?adjusted=" + (chunk_q.adjusted?"true":"false") + "&sort=" + chunk_q.sort + "&limit=" + std::to_string(chunk_q.limit);
        std::string url = base;
        
        for (int page=0; page<max_pages; ++page) {
            std::string body = get_(url);
            if (body.empty()) break;
            
            auto j = json::parse(body, nullptr, false);
            if (j.is_discarded()) {
                std::cerr << "JSON parsing failed for chunk " << start_str << " to " << end_str << std::endl;
                break;
            }
            
            if (j.contains("results")) {
                for (auto& r : j["results"]) {
                    out.push_back({r.value("t", 0LL), r.value("o", 0.0), r.value("h", 0.0), r.value("l", 0.0), r.value("c", 0.0), r.value("v", 0.0)});
                }
            }
            
            if (!j.contains("next_url")) break;
            url = j["next_url"].get<std::string>();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        
        current_time = chunk_end + (24 * 60 * 60); // Move to next day properly
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Rate limiting between chunks
    }
    
    std::cerr << "Total bars collected: " << out.size() << std::endl;
    return out;
}

void PolygonClient::write_csv(const std::string& out_path,const std::string& symbol,
                              const std::vector<AggBar>& bars, bool exclude_holidays, bool rth_only) {
  std::ofstream f(out_path);
  f << "timestamp,symbol,open,high,low,close,volume\n";
  for (auto& a: bars) {
    // **MODIFIED**: RTH and holiday filtering is now done directly on the UTC timestamp
    // before any string conversion, making it much more reliable.

    // RTH filtering: 9:30 AM - 4:00 PM ET (14:30 - 21:00 UTC)
    if (rth_only) {
        cctz::time_point<cctz::seconds> tp{cctz::seconds{a.ts_ms / 1000}};
        
        // Get ET timezone
        cctz::time_zone et_tz;
        if (cctz::load_time_zone("America/New_York", &et_tz)) {
            auto lt = cctz::convert(tp, et_tz);
            auto ct = cctz::civil_second(lt);
            
            // Check if it's a weekday (Monday=1, Sunday=7)
            auto weekday = cctz::get_weekday(cctz::civil_day(ct));
            if (weekday == cctz::weekday::saturday || weekday == cctz::weekday::sunday) {
                continue; // Skip weekends
            }
            
            // Check if it's within RTH: 9:30 AM - 4:00 PM ET
            int hour = ct.hour();
            int minute = ct.minute();
            int time_minutes = hour * 60 + minute;
            int rth_start = 9 * 60 + 30;  // 9:30 AM
            int rth_end = 16 * 60;        // 4:00 PM
            
            if (time_minutes < rth_start || time_minutes >= rth_end) {
                continue; // Skip pre-market and after-hours
            }
        }
    }
    
    if (exclude_holidays) {
        cctz::time_point<cctz::seconds> tp{cctz::seconds{a.ts_ms / 1000}};
        
        // Get UTC timezone
        cctz::time_zone utc_tz;
        if (cctz::load_time_zone("UTC", &utc_tz)) {
            auto lt = cctz::convert(tp, utc_tz);
            auto ct = cctz::civil_second(lt);
            
            if (is_us_market_holiday_utc(ct.year(), ct.month(), ct.day())) {
                continue;
            }
        }
    }
    
    // The timestamp is converted to a UTC string for writing to the CSV
    std::string ts_str = rfc3339_utc_from_epoch_ms(a.ts_ms);

    f << ts_str << ',' << symbol << ','
      << a.open << ',' << a.high << ',' << a.low << ',' << a.close << ',' << a.volume << '\n';
  }
}

} // namespace sentio

```

## 📄 **FILE 165 of 180**: src/position_coordinator.cpp

**File Information**:
- **Path**: `src/position_coordinator.cpp`

- **Size**: 506 lines
- **Modified**: 2025-09-18 18:34:27

- **Type**: .cpp

```text
#include "sentio/position_coordinator.hpp"
#include "sentio/base_strategy.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace sentio {

// **ETF CLASSIFICATIONS**: PSQ is inverse ETF, not short position
const std::unordered_set<std::string> PositionCoordinator::LONG_ETFS = {"QQQ", "TQQQ"};
const std::unordered_set<std::string> PositionCoordinator::INVERSE_ETFS = {"SQQQ", "PSQ"};

PositionCoordinator::PositionCoordinator(int max_orders_per_bar)
    : orders_this_bar_(0), max_orders_per_bar_(max_orders_per_bar) {
}

void PositionCoordinator::reset_bar() {
    orders_this_bar_ = 0;
    pending_positions_.clear();
}

void PositionCoordinator::sync_positions(const Portfolio& portfolio, const SymbolTable& ST) {
    current_positions_.clear();
    
    for (size_t i = 0; i < portfolio.positions.size(); ++i) {
        const auto& pos = portfolio.positions[i];
        if (std::abs(pos.qty) > 1e-6) {
            const std::string& symbol = ST.get_symbol(i);
            current_positions_[symbol] = pos.qty;
        }
    }
}

PositionCoordinator::PositionAnalysis PositionCoordinator::analyze_positions(
    const std::unordered_map<std::string, double>& positions) const {
    
    PositionAnalysis analysis;
    
    for (const auto& [symbol, qty] : positions) {
        if (std::abs(qty) > 1e-6) {
            if (LONG_ETFS.count(symbol)) {
                if (qty > 0) {
                    analysis.has_long_etf = true;
                    analysis.long_positions.push_back(symbol + "(+" + std::to_string((int)qty) + ")");
                } else {
                    analysis.has_short_qqq = true;
                    analysis.short_positions.push_back("SHORT " + symbol + "(" + std::to_string((int)qty) + ")");
                }
            }
            if (INVERSE_ETFS.count(symbol)) {
                analysis.has_inverse_etf = true;
                analysis.inverse_positions.push_back(symbol + "(" + std::to_string((int)qty) + ")");
            }
        }
    }
    
    return analysis;
}

bool PositionCoordinator::would_create_conflict(
    const std::unordered_map<std::string, double>& positions) const {
    
    auto analysis = analyze_positions(positions);
    
    // **CONFLICT RULES**:
    // 1. Long ETF conflicts with Inverse ETF or SHORT QQQ
    // 2. SHORT QQQ conflicts with Long ETF
    // 3. Inverse ETF conflicts with Long ETF
    return (analysis.has_long_etf && (analysis.has_inverse_etf || analysis.has_short_qqq)) ||
           (analysis.has_short_qqq && analysis.has_long_etf);
}

CoordinationDecision PositionCoordinator::resolve_conflict(const AllocationRequest& request) {
    CoordinationDecision decision;
    decision.instrument = request.instrument;
    decision.original_weight = request.target_weight;
    decision.approved_weight = 0.0; // Default to rejection
    
    // **CONFLICT RESOLUTION STRATEGIES**:
    
    // 1. **ZERO OUT CONFLICTING POSITION**: Set weight to zero to avoid conflict
    decision.result = CoordinationResult::MODIFIED;
    decision.approved_weight = 0.0;
    decision.reason = "CONFLICT_PREVENTION_ZERO";
    
    // Build conflict details
    auto current_analysis = analyze_positions(current_positions_);
    auto pending_analysis = analyze_positions(pending_positions_);
    
    std::string conflict_details = "CONFLICT: ";
    if (!current_analysis.long_positions.empty()) {
        conflict_details += "Current Long: ";
        for (size_t i = 0; i < current_analysis.long_positions.size(); ++i) {
            if (i > 0) conflict_details += ", ";
            conflict_details += current_analysis.long_positions[i];
        }
    }
    if (!current_analysis.short_positions.empty()) {
        if (!current_analysis.long_positions.empty()) conflict_details += "; ";
        conflict_details += "Current Short: ";
        for (size_t i = 0; i < current_analysis.short_positions.size(); ++i) {
            if (i > 0) conflict_details += ", ";
            conflict_details += current_analysis.short_positions[i];
        }
    }
    if (!current_analysis.inverse_positions.empty()) {
        if (!current_analysis.long_positions.empty() || !current_analysis.short_positions.empty()) {
            conflict_details += "; ";
        }
        conflict_details += "Current Inverse: ";
        for (size_t i = 0; i < current_analysis.inverse_positions.size(); ++i) {
            if (i > 0) conflict_details += ", ";
            conflict_details += current_analysis.inverse_positions[i];
        }
    }
    
    conflict_details += " | Requested: " + request.instrument + "(" + std::to_string(request.target_weight) + ")";
    decision.conflict_details = conflict_details;
    
    return decision;
}

std::vector<CoordinationDecision> PositionCoordinator::coordinate_allocations(
    const std::vector<AllocationRequest>& requests,
    const Portfolio& current_portfolio,
    const SymbolTable& ST) {
    
    std::vector<CoordinationDecision> decisions;
    decisions.reserve(requests.size());
    
    // **SYNC CURRENT POSITIONS**
    sync_positions(current_portfolio, ST);
    pending_positions_ = current_positions_; // Start with current positions
    
    // **PROCESS EACH REQUEST**
    for (const auto& request : requests) {
        stats_.total_requests++;
        
        CoordinationDecision decision;
        decision.instrument = request.instrument;
        decision.original_weight = request.target_weight;
        decision.approved_weight = request.target_weight; // Default to approval
        
        // **FREQUENCY CONTROL**: Check order frequency limit
        // Note: Conflict resolution may generate additional close orders, so we need some buffer
        if (orders_this_bar_ >= max_orders_per_bar_ && max_orders_per_bar_ == 1) {
            decision.result = CoordinationResult::REJECTED_FREQUENCY;
            decision.approved_weight = 0.0;
            decision.reason = "FREQUENCY_LIMIT_EXCEEDED";
            decision.conflict_details = "Max " + std::to_string(max_orders_per_bar_) + " orders per bar";
            stats_.rejected_frequency++;
            decisions.push_back(decision);
            continue;
        }
        
        // **SIMULATE POSITION CHANGE WITH AUTOMATIC CONFLICT RESOLUTION**
        std::unordered_map<std::string, double> simulated_positions = pending_positions_;
        
        // Apply the requested position change
        if (std::abs(request.target_weight) > 1e-6) {
            if (request.target_weight > 0) {
                simulated_positions[request.instrument] = std::abs(request.target_weight) * 1000; // Simplified
            } else {
                simulated_positions[request.instrument] = request.target_weight * 1000; // Negative for short
            }
        } else {
            simulated_positions.erase(request.instrument); // Zero weight = close position
        }
        
        // **AUTOMATIC CONFLICT RESOLUTION**: If conflicts exist, close conflicting positions
        if (would_create_conflict(simulated_positions)) {
            // **STRATEGY**: Close all conflicting positions to allow the new position
            std::vector<std::string> positions_to_close;
            
            // Determine what type of position is being requested
            bool requesting_long_etf = (LONG_ETFS.count(request.instrument) && request.target_weight > 0);
            bool requesting_inverse_etf = (INVERSE_ETFS.count(request.instrument) && request.target_weight > 0);
            bool requesting_short_qqq = (LONG_ETFS.count(request.instrument) && request.target_weight < 0);
            
            // Close conflicting existing positions
            for (const auto& [symbol, qty] : simulated_positions) {
                if (symbol != request.instrument && std::abs(qty) > 1e-6) {
                    bool should_close = false;
                    
                    if (requesting_long_etf) {
                        // Close inverse ETFs and short QQQ positions
                        should_close = (INVERSE_ETFS.count(symbol)) || 
                                     (LONG_ETFS.count(symbol) && qty < 0);
                    } else if (requesting_inverse_etf) {
                        // Close long ETF positions
                        should_close = (LONG_ETFS.count(symbol) && qty > 0);
                    } else if (requesting_short_qqq) {
                        // Close long ETF and inverse ETF positions
                        should_close = (LONG_ETFS.count(symbol) && qty > 0) || 
                                     (INVERSE_ETFS.count(symbol));
                    }
                    
                    if (should_close) {
                        positions_to_close.push_back(symbol);
                        simulated_positions.erase(symbol); // Remove conflicting position
                    }
                }
            }
            
            // Build conflict resolution details
            std::string resolution_details = "AUTO_RESOLVED: ";
            if (!positions_to_close.empty()) {
                resolution_details += "Closed conflicting positions: ";
                for (size_t i = 0; i < positions_to_close.size(); ++i) {
                    if (i > 0) resolution_details += ", ";
                    resolution_details += positions_to_close[i];
                }
                resolution_details += " to allow " + request.instrument;
                
                // **GENERATE CLOSE ORDERS**: Add close orders for conflicting positions
                for (const auto& symbol_to_close : positions_to_close) {
                    CoordinationDecision close_decision;
                    close_decision.instrument = symbol_to_close;
                    close_decision.original_weight = 0.0; // This is a generated close order
                    close_decision.approved_weight = 0.0; // Close position
                    close_decision.result = CoordinationResult::APPROVED;
                    close_decision.reason = "AUTO_CLOSE_CONFLICT";
                    close_decision.conflict_details = "Auto-closed to prevent conflict with " + request.instrument;
                    
                    decisions.push_back(close_decision);
                    orders_this_bar_++; // Count each close as an order
                }
            }
            
            decision.result = CoordinationResult::APPROVED;
            decision.reason = "APPROVED_WITH_CONFLICT_RESOLUTION";
            decision.conflict_details = resolution_details;
            
            // Update pending positions for next request
            pending_positions_ = simulated_positions;
            orders_this_bar_++;
            stats_.approved++;
        } else {
            // **APPROVAL**: No conflict detected
            decision.result = CoordinationResult::APPROVED;
            decision.reason = "APPROVED_NO_CONFLICT";
            
            // Update pending positions for next request
            pending_positions_ = simulated_positions;
            orders_this_bar_++;
            stats_.approved++;
        }
        
        decisions.push_back(decision);
    }
    
    return decisions;
}

// **CONVERTER FUNCTION**: Convert strategy allocation decisions to coordination requests
std::vector<AllocationRequest> convert_allocation_decisions(
    const std::vector<AllocationDecision>& decisions,
    const std::string& strategy_name,
    const std::string& chain_id) {
    
    std::vector<AllocationRequest> requests;
    requests.reserve(decisions.size());
    
    for (const auto& decision : decisions) {
        AllocationRequest request;
        request.strategy_name = strategy_name;
        request.instrument = decision.instrument;
        request.target_weight = decision.target_weight;
        request.confidence = decision.confidence;
        request.reason = decision.reason;
        request.chain_id = chain_id;
        
        requests.push_back(request);
    }
    
    return requests;
}

// **STRATEGY-AGNOSTIC COORDINATION**: Use strategy's conflict rules
std::vector<CoordinationDecision> PositionCoordinator::coordinate_allocations(
    const std::vector<AllocationRequest>& requests,
    const Portfolio& current_portfolio,
    const SymbolTable& ST,
    const BaseStrategy* strategy) {
    
    if (!strategy) {
        // Fall back to default coordination if no strategy provided
        return coordinate_allocations(requests, current_portfolio, ST);
    }
    
    // Sync current positions and initialize pending to current
    sync_positions(current_portfolio, ST);
    pending_positions_ = current_positions_;
    
    std::vector<CoordinationDecision> decisions;
    decisions.reserve(requests.size());
    
    // **SEQUENTIAL TRANSITION OPTIMIZATION**: For sequential strategies, process as single rebalancing
    if (strategy->requires_sequential_transitions() && requests.size() > 1) {
        // Find the primary target (non-zero weight)
        const AllocationRequest* primary_target = nullptr;
        for (const auto& request : requests) {
            if (std::abs(request.target_weight) > 1e-6) {
                primary_target = &request;
                break;
            }
        }
        
        if (primary_target && orders_this_bar_ == 0) {
            // Approve only the primary target, reject all others
            for (const auto& request : requests) {
                CoordinationDecision decision;
                decision.instrument = request.instrument;
                decision.original_weight = request.target_weight;
                decision.reason = request.reason;
                
                if (request.instrument == primary_target->instrument) {
                    // Approve primary target
                    decision.result = CoordinationResult::APPROVED;
                    decision.approved_weight = request.target_weight;
                    orders_this_bar_++;
                } else {
                    // Reject all others for sequential transition
                    decision.result = CoordinationResult::REJECTED_FREQUENCY;
                    decision.approved_weight = 0.0;
                    decision.reason = "Sequential transition: only one instrument per bar";
                }
                
                decisions.push_back(decision);
            }
            return decisions;
        } else if (orders_this_bar_ > 0) {
            // Already had a transaction this bar - reject all
            for (const auto& request : requests) {
                CoordinationDecision decision;
                decision.instrument = request.instrument;
                decision.original_weight = request.target_weight;
                decision.approved_weight = 0.0;
                decision.result = CoordinationResult::REJECTED_FREQUENCY;
                decision.reason = "Sequential strategy: max 1 transaction per bar";
                decisions.push_back(decision);
            }
            return decisions;
        }
    }
    
        // **STANDARD PROCESSING**: Process each request individually
    for (const auto& request : requests) {
        CoordinationDecision decision;
        decision.instrument = request.instrument;
        decision.original_weight = request.target_weight;
        decision.approved_weight = request.target_weight;
        decision.reason = request.reason;
        
        // **STRATEGY-AWARE AUTOMATIC CONFLICT RESOLUTION**
        std::unordered_map<std::string, double> simulated_positions = pending_positions_;
        
        // Apply the requested position change
        if (std::abs(request.target_weight) > 1e-6) {
            simulated_positions[request.instrument] = request.target_weight * 1000; // weight->qty proxy
        } else {
            simulated_positions.erase(request.instrument);
        }
        
        // Check for conflicts and automatically resolve them
        bool has_conflict = false;
        std::vector<std::string> positions_to_close;
        std::string conflict_details = "";
        
        // **FAMILY-LEVEL CONFLICT DETECTION AND RESOLUTION**
        if (would_create_conflict(simulated_positions)) {
            has_conflict = true;
            
            // Determine what type of position is being requested
            bool requesting_long_etf = (LONG_ETFS.count(request.instrument) && request.target_weight > 0);
            bool requesting_inverse_etf = (INVERSE_ETFS.count(request.instrument) && request.target_weight > 0);
            bool requesting_short_qqq = (LONG_ETFS.count(request.instrument) && request.target_weight < 0);
            
            // Close conflicting existing positions
            for (const auto& [symbol, qty] : simulated_positions) {
                if (symbol != request.instrument && std::abs(qty) > 1e-6) {
                    bool should_close = false;
                    
                    if (requesting_long_etf) {
                        // Close inverse ETFs and short QQQ positions
                        should_close = (INVERSE_ETFS.count(symbol)) || 
                                     (LONG_ETFS.count(symbol) && qty < 0);
                    } else if (requesting_inverse_etf) {
                        // Close long ETF positions
                        should_close = (LONG_ETFS.count(symbol) && qty > 0);
                    } else if (requesting_short_qqq) {
                        // Close long ETF and inverse ETF positions
                        should_close = (LONG_ETFS.count(symbol) && qty > 0) || 
                                     (INVERSE_ETFS.count(symbol));
                    }
                    
                    if (should_close) {
                        positions_to_close.push_back(symbol);
                        simulated_positions.erase(symbol); // Remove conflicting position
                    }
                }
            }
            
            conflict_details = "AUTO_RESOLVED: Family conflict resolved by closing ";
            for (size_t i = 0; i < positions_to_close.size(); ++i) {
                if (i > 0) conflict_details += ", ";
                conflict_details += positions_to_close[i];
            }
        }
        
        // **STRATEGY-LEVEL CONFLICT DETECTION AND RESOLUTION**
        if (!has_conflict) {
            for (const auto& [existing_instrument, existing_qty] : current_positions_) {
                if (existing_instrument != request.instrument && std::abs(existing_qty) > 1e-6) {
                    if (!strategy->allows_simultaneous_positions(request.instrument, existing_instrument)) {
                        has_conflict = true;
                        positions_to_close.push_back(existing_instrument);
                        simulated_positions.erase(existing_instrument);
                        conflict_details = "AUTO_RESOLVED: Strategy conflict resolved by closing " + existing_instrument;
                        break;
                    }
                }
            }
        }
        
        // Check conflicts with other pending requests (no auto-resolution for pending)
        if (!has_conflict) {
            for (const auto& other_request : requests) {
                if (other_request.instrument != request.instrument && 
                    std::abs(other_request.target_weight) > 1e-6) {
                    if (!strategy->allows_simultaneous_positions(request.instrument, other_request.instrument)) {
                        has_conflict = true;
                        conflict_details = "Strategy prohibits simultaneous " + request.instrument + " + " + other_request.instrument;
                        break;
                    }
                }
            }
        }
        
        // **SEQUENTIAL TRANSITION ENFORCEMENT**: For strategies requiring sequential transitions,
        // enforce maximum one transaction per bar regardless of conflicts
        if (strategy->requires_sequential_transitions() && orders_this_bar_ >= 1) {
            decision.result = CoordinationResult::REJECTED_FREQUENCY;
            decision.approved_weight = 0.0;
            decision.reason = "Sequential strategy: max 1 transaction per bar";
        } else if (!strategy->requires_sequential_transitions() && orders_this_bar_ >= max_orders_per_bar_) {
            // Apply normal frequency limits for non-sequential strategies
            decision.result = CoordinationResult::REJECTED_FREQUENCY;
            decision.approved_weight = 0.0;
            decision.reason = "Frequency limit exceeded";
        } else if (has_conflict && positions_to_close.empty()) {
            // **UNRESOLVABLE CONFLICT**: No automatic resolution possible
            if (strategy->requires_sequential_transitions()) {
                // Strategy requires sequential transitions - reject conflicting request
                decision.result = CoordinationResult::REJECTED_CONFLICT;
                decision.approved_weight = 0.0;
                decision.conflict_details = conflict_details;
            } else {
                // Strategy allows simultaneous transitions - modify to zero weight
                decision.result = CoordinationResult::MODIFIED;
                decision.approved_weight = 0.0;
                decision.conflict_details = conflict_details;
            }
        } else if (has_conflict && !positions_to_close.empty()) {
            // **AUTO-RESOLVED CONFLICT**: Generate close orders and approve request
            
            // Generate close orders for conflicting positions
            for (const auto& symbol_to_close : positions_to_close) {
                CoordinationDecision close_decision;
                close_decision.instrument = symbol_to_close;
                close_decision.original_weight = 0.0; // This is a generated close order
                close_decision.approved_weight = 0.0; // Close position
                close_decision.result = CoordinationResult::APPROVED;
                close_decision.reason = "AUTO_CLOSE_CONFLICT_STRATEGY";
                close_decision.conflict_details = "Auto-closed to prevent conflict with " + request.instrument;
                
                decisions.push_back(close_decision);
                orders_this_bar_++; // Count each close as an order
            }
            
            // Approve the original request
            decision.result = CoordinationResult::APPROVED;
            decision.reason = "APPROVED_WITH_AUTO_RESOLUTION";
            decision.conflict_details = conflict_details;
            orders_this_bar_++;
            
            // Update pending positions for next request
            pending_positions_ = simulated_positions;
        } else {
            // **NO CONFLICT**: Approved
            decision.result = CoordinationResult::APPROVED;
            orders_this_bar_++;
            
            // Update pending positions for next request
            pending_positions_ = simulated_positions;
        }
        
        decisions.push_back(decision);
        
        // Note: pending_positions_ is updated in the decision logic above
    }
    
    return decisions;
}

} // namespace sentio

```

## 📄 **FILE 166 of 180**: src/router.cpp

**File Information**:
- **Path**: `src/router.cpp`

- **Size**: 92 lines
- **Modified**: 2025-09-15 15:04:43

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

## 📄 **FILE 167 of 180**: src/run_id_generator.cpp

**File Information**:
- **Path**: `src/run_id_generator.cpp`

- **Size**: 48 lines
- **Modified**: 2025-09-13 09:40:08

- **Type**: .cpp

```text
#include "sentio/run_id_generator.hpp"
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>

namespace sentio {

std::string generate_run_id() {
    // Use a combination of timestamp and random number for uniqueness
    auto now = std::chrono::steady_clock::now();
    auto timestamp = now.time_since_epoch().count();
    
    // Create a random number generator
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(100000, 999999);
    
    // Generate a 6-digit number
    // Use last 3 digits of timestamp + 3 random digits for better distribution
    int timestamp_part = (timestamp % 1000);
    int random_part = dis(gen) % 1000;
    
    int run_id_num = timestamp_part * 1000 + random_part;
    
    // Ensure it's exactly 6 digits
    run_id_num = 100000 + (run_id_num % 900000);
    
    return std::to_string(run_id_num);
}

std::string create_audit_note(const std::string& strategy_name, 
                             const std::string& test_type, 
                             const std::string& period_info) {
    std::ostringstream note;
    note << "Strategy: " << strategy_name;
    note << ", Test: " << test_type;
    
    if (!period_info.empty()) {
        note << ", Period: " << period_info;
    }
    
    note << ", Generated by sentio_cli";
    
    return note.str();
}

} // namespace sentio

```

## 📄 **FILE 168 of 180**: src/runner.cpp

**File Information**:
- **Path**: `src/runner.cpp`

- **Size**: 1033 lines
- **Modified**: 2025-09-18 18:26:10

- **Type**: .cpp

```text
#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/pricebook.hpp"
#include "sentio/metrics.hpp"
#include "audit/audit_db_recorder.hpp"
#include "sentio/sizer.hpp"
#include "sentio/feature_feeder.hpp"
#include "sentio/position_coordinator.hpp"
#include "sentio/router.hpp"
#include "sentio/allocation_manager.hpp"
#include "sentio/canonical_evaluation.hpp"
#include "sentio/eod_position_manager.hpp"
#include <iostream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <sqlite3.h>

namespace sentio {

// **RENOVATED HELPER**: Execute target position for any instrument
static void execute_target_position(const std::string& instrument, double target_weight,            
                                   Portfolio& portfolio, const SymbolTable& ST, const Pricebook& pricebook,                    
                                   const AdvancedSizer& sizer, const RunnerCfg& cfg,                
                                   const std::vector<std::vector<Bar>>& series, const Bar& bar,     
                                   const std::string& chain_id, IAuditRecorder& audit, bool logging_enabled, int& total_fills) {
    
    int instrument_id = ST.get_id(instrument);
    if (instrument_id == -1) return;
    
    double instrument_price = pricebook.last_px[instrument_id];
    if (instrument_price <= 0) return;

    // Calculate target quantity using sizer
    double target_qty = sizer.calculate_target_quantity(portfolio, ST, pricebook.last_px, 
                                                       instrument, target_weight, 
                                                       series[instrument_id], cfg.sizer);
    
    // **CONFLICT PREVENTION**: Strategy-level conflict prevention should prevent conflicts
    // No need for smart conflict resolution since strategy checks existing positions
    
    double current_qty = portfolio.positions[instrument_id].qty;
    double trade_qty = target_qty - current_qty;

    // **INTEGRITY FIX**: Prevent impossible trades that create negative positions
    // 1. Prevent zero-quantity trades that generate phantom P&L
    // 2. Prevent SELL orders on zero positions (would create negative positions)
    if (std::abs(trade_qty) < 1e-9 || std::abs(trade_qty * instrument_price) <= 10.0) {
        return; // No logging, no execution, no audit entries
    }
    
    // **CRITICAL INTEGRITY CHECK**: Prevent SELL orders that would create negative positions
    if (trade_qty < 0 && current_qty <= 1e-6) {
        // Cannot sell what we don't own - this would create negative positions
        return; // Skip this trade to maintain position integrity
    }

    // **PROFIT MAXIMIZATION**: Execute meaningful trades
    Side side = (trade_qty > 0) ? Side::Buy : Side::Sell;
    
    if (logging_enabled) {
        audit.event_order_ex(bar.ts_utc_epoch, instrument, side, std::abs(trade_qty), 0.0, chain_id);
    }

    // Calculate realized P&L for position changes
    double realized_delta = 0.0;
    const auto& pos_before = portfolio.positions[instrument_id];
    double closing = 0.0;
    if (pos_before.qty > 0 && trade_qty < 0) closing = std::min(std::abs(trade_qty), pos_before.qty);
    if (pos_before.qty < 0 && trade_qty > 0) closing = std::min(std::abs(trade_qty), std::abs(pos_before.qty));
    if (closing > 0.0) {
        if (pos_before.qty > 0) realized_delta = (instrument_price - pos_before.avg_price) * closing;
        else                    realized_delta = (pos_before.avg_price - instrument_price) * closing;
    }

    // **ALPACA COSTS**: Use realistic Alpaca fee model for accurate backtesting
    bool is_sell = (side == Side::Sell);
    double fees = AlpacaCostModel::calculate_fees(instrument, std::abs(trade_qty), instrument_price, is_sell);
    double exec_px = instrument_price; // Perfect execution at market price (no slippage)
    
    apply_fill(portfolio, instrument_id, trade_qty, exec_px);
    portfolio.cash -= fees; // Apply transaction fees
    
    double equity_after = equity_mark_to_market(portfolio, pricebook.last_px);
    double pos_after = portfolio.positions[instrument_id].qty;
    
    if (logging_enabled) {
        audit.event_fill_ex(bar.ts_utc_epoch, instrument, exec_px, std::abs(trade_qty), fees, side,
                           realized_delta, equity_after, pos_after, chain_id);
    }
    total_fills++;
}

// CHANGED: The function now returns a BacktestOutput struct with raw data and accepts dataset metadata.
BacktestOutput run_backtest(IAuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                      int base_symbol_id, const RunnerCfg& cfg, const DatasetMetadata& dataset_meta) {
    
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
    AdvancedSizer sizer;
    Pricebook pricebook(base_symbol_id, ST, series);
    
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
        
        // **ENHANCED EOD POSITION MANAGEMENT**: Multi-stage closure with leveraged ETF priority
        // Create EOD Position Manager with profit-maximizing configuration (static for persistence)
        static EODPositionManager eod_manager([logging_enabled]() {
            EODPositionManager::Config cfg;
            cfg.warning_minutes_before_close = 30;     // Early warning for large positions
            cfg.partial_close_minutes = 15;            // Begin partial closure of high-risk positions
            cfg.mandatory_close_minutes = 5;           // Force complete closure
            cfg.final_sweep_minutes = 2;               // Final safety sweep (2min buffer)
            cfg.prioritize_leveraged_etfs = true;      // Close TQQQ/SQQQ first (high decay)
            cfg.prioritize_large_positions = true;     // Close large positions first
            cfg.min_position_value = 10.0;             // Minimum $ value to close
            cfg.use_market_orders = true;              // Perfect execution for backtesting
            cfg.allow_partial_closure = true;          // Gradual closure to minimize impact
            cfg.partial_closure_fraction = 0.5;       // Close 50% in partial phase
            cfg.max_overnight_exposure = 0.0;          // Zero overnight exposure
            cfg.force_cash_only_overnight = true;      // 100% cash overnight
            cfg.detailed_logging = logging_enabled;    // Match runner logging
            cfg.closure_reason_prefix = "EOD_ENHANCED_CLOSE";
            return cfg;
        }());
        
        if (i > warmup_bars) {
            // Process EOD closure - handles all timing, prioritization, and execution
            int positions_closed = eod_manager.process_eod_closure(
                portfolio, pricebook.last_px, ST, bar.ts_utc_epoch, audit
            );
            
            // Update fill counter for closed positions
            total_fills += positions_closed;
        }
        
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
        
        // **ARCHITECTURAL COMPLIANCE**: Check strategy preference for execution path
        // **NEW ORTHOGONAL ARCHITECTURE**: AllocationManager handles all instrument decisions
        // Strategy only provides probability, AllocationManager maximizes profit
        
        // Create allocation manager for this strategy (could be cached/reused)
        static AllocationManager allocation_manager; // Static for state persistence
        
        // Ensure allocation manager is properly initialized
        if (i == warmup_bars) {
            allocation_manager.reset_state(); // Reset at start of test
        }
        
        // Get allocation decisions from AllocationManager
        
        auto manager_decisions = allocation_manager.get_runner_allocations(
            probability, portfolio, pricebook.last_px, ST);
        
        // Convert to runner format
        std::vector<AllocationDecision> allocation_decisions;
        for (const auto& manager_decision : manager_decisions) {
            AllocationDecision decision;
            decision.instrument = manager_decision.instrument;
            decision.target_weight = manager_decision.target_weight;
            decision.confidence = manager_decision.confidence;
            decision.reason = manager_decision.reason;
            allocation_decisions.push_back(decision);
        }
        
        // **AUDIT**: Log allocation decisions (strategy-agnostic)
        if (logging_enabled) {
            for (const auto& decision : allocation_decisions) {
                if (std::abs(decision.target_weight) > 1e-6) {
                    audit.event_route(bar.ts_utc_epoch, base_symbol, decision.instrument, decision.target_weight);
                }
            }
        }
        
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
        
        
        // **STRATEGY-ISOLATED COORDINATION**: Create fresh coordinator for each run
        PositionCoordinator coordinator(1); // Max 1 order per bar (enforce single trade per bar)
        
        // Convert strategy allocation decisions to coordination format
        std::vector<AllocationDecision> coord_allocation_decisions;
        for (const auto& decision : allocation_decisions) {
            AllocationDecision coord_decision;
            coord_decision.instrument = decision.instrument;
            coord_decision.target_weight = decision.target_weight;
            coord_decision.confidence = decision.confidence;
            coord_decision.reason = decision.reason;
            coord_allocation_decisions.push_back(coord_decision);
        }
        
        // Convert to coordination requests
        auto allocation_requests = convert_allocation_decisions(coord_allocation_decisions, cfg.strategy_name, chain_id);
        
        // **STRATEGY-AGNOSTIC COORDINATION**: Use strategy's conflict rules
        auto coordination_decisions = coordinator.coordinate_allocations(allocation_requests, portfolio, ST, strategy.get());
        
        // Log coordination statistics
        // auto coord_stats = coordinator.get_stats(); // Unused during cleanup
        int approved_orders = 0;
        int rejected_conflicts = 0;
        int rejected_frequency = 0;
        
        // **EOD TRADE BLOCKING**: Prevent new positions during/after EOD closure
        if (eod_manager.is_closure_active()) {
            // Block all new trades once EOD closure begins - trading resumes next day
            for (const auto& coord_decision : coordination_decisions) {
                if (coord_decision.result == CoordinationResult::APPROVED && logging_enabled) {
                    audit.event_signal_drop(bar.ts_utc_epoch, cfg.strategy_name, coord_decision.instrument,
                                          DropReason::THRESHOLD, chain_id,
                                          "EOD_BLOCK: Trading suspended during EOD closure - resumes next trading day");
                }
            }
            // Skip all trade execution for the rest of this trading day
            continue;
        }
        
        // **COORDINATED EXECUTION**: Execute only approved allocation decisions
        for (const auto& coord_decision : coordination_decisions) {
            if (coord_decision.result == CoordinationResult::APPROVED) {
                // **SMART FINAL GUARD**: Allow close orders (zero weight) even when conflicts exist
                // This prevents deadlock where we can't close conflicting positions
                bool is_close_order = (std::abs(coord_decision.approved_weight) < 1e-6);
                bool has_conflicts = has_conflicting_positions(portfolio, ST);
                
                if (has_conflicts && !is_close_order) {
                    if (logging_enabled) {
                        audit.event_signal_drop(bar.ts_utc_epoch, cfg.strategy_name, coord_decision.instrument,
                                              DropReason::THRESHOLD, chain_id,
                                              "FINAL_GUARD_BLOCK: pre-exec conflict detected (non-close order)");
                    }
                    continue;
                }
                // Execute approved decision
                execute_target_position(coord_decision.instrument, coord_decision.approved_weight,
                                      portfolio, ST, pricebook, sizer, cfg, series, bar,
                                      chain_id, audit, logging_enabled, total_fills);
                approved_orders++;
                
            } else if (coord_decision.result == CoordinationResult::REJECTED_CONFLICT) {
                // Log conflict prevention
                if (logging_enabled) {
                    audit.event_signal_drop(bar.ts_utc_epoch, cfg.strategy_name, coord_decision.instrument, 
                                          DropReason::THRESHOLD, chain_id, 
                                          "CONFLICT_PREVENTED: " + coord_decision.conflict_details);
                }
                rejected_conflicts++;
                
            } else if (coord_decision.result == CoordinationResult::REJECTED_FREQUENCY) {
                // Log frequency limit
                if (logging_enabled) {
                    audit.event_signal_drop(bar.ts_utc_epoch, cfg.strategy_name, coord_decision.instrument, 
                                          DropReason::THRESHOLD, chain_id, 
                                          "FREQUENCY_LIMITED: " + coord_decision.reason);
                }
                rejected_frequency++;
                
            } else if (coord_decision.result == CoordinationResult::MODIFIED) {
                // Execute modified decision (usually zero weight to prevent conflict)
                if (std::abs(coord_decision.approved_weight) > 1e-6) {
                    execute_target_position(coord_decision.instrument, coord_decision.approved_weight, 
                                          portfolio, ST, pricebook, sizer, cfg, series, bar, 
                                          chain_id, audit, logging_enabled, total_fills);
                }
                if (logging_enabled) {
                    audit.event_signal_drop(bar.ts_utc_epoch, cfg.strategy_name, coord_decision.instrument, 
                                          DropReason::THRESHOLD, chain_id, 
                                          "MODIFIED_FOR_CONFLICT: " + coord_decision.conflict_details);
                }
            }
        }
        
        // **COORDINATION VALIDATION**: Validate no conflicting positions exist after coordination
        int post_coordination_conflicts = has_conflicting_positions(portfolio, ST) ? 1 : 0;
        
        // **AUDIT COORDINATION STATISTICS**: Log coordination metrics
        if (logging_enabled && (approved_orders > 0 || rejected_conflicts > 0 || rejected_frequency > 0 || post_coordination_conflicts > 0)) {
            std::string coord_stats = "COORDINATION_APPROVED:" + std::to_string(approved_orders) + 
                                    ",CONFLICTS_PREVENTED:" + std::to_string(rejected_conflicts) + 
                                    ",FREQUENCY_LIMITED:" + std::to_string(rejected_frequency) + 
                                    ",POST_COORD_CONFLICTS:" + std::to_string(post_coordination_conflicts);
            audit.event_signal_drop(bar.ts_utc_epoch, cfg.strategy_name, "COORDINATION_STATS", 
                                  DropReason::NONE, chain_id, coord_stats);
        }
        
        // **CRITICAL ASSERTION**: With proper coordination, there should NEVER be post-execution conflicts
        if (post_coordination_conflicts > 0) {
            std::cerr << "CRITICAL ERROR: Position Coordinator failed to prevent conflicts!" << std::endl;
            std::cerr << "This indicates a bug in the coordination logic." << std::endl;
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
        
        // Run backtest for this block
        BacktestOutput block_output = run_backtest(audit, ST, block_series, base_symbol_id, block_cfg, block_meta);
        
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

## 📄 **FILE 169 of 180**: src/signal_engine.cpp

**File Information**:
- **Path**: `src/signal_engine.cpp`

- **Size**: 29 lines
- **Modified**: 2025-09-12 10:12:46

- **Type**: .cpp

```text
#include "sentio/signal_engine.hpp"

namespace sentio {

SignalEngine::SignalEngine(IStrategy* strat, const GateCfg& gate_cfg, SignalHealth* health)
: strat_(strat), gate_(gate_cfg, health), health_(health) {}

EngineOut SignalEngine::on_bar(const StrategyCtx& ctx, const Bar& b, bool inputs_finite) {
  strat_->on_bar(ctx, b);
  auto raw = strat_->latest();
  if (!raw.has_value()) {
    if (health_) health_->incr_drop(DropReason::NONE);
    return {std::nullopt, DropReason::NONE};
  }

  double conf = raw->confidence;
  auto conf2 = gate_.accept(ctx.ts_utc_epoch, inputs_finite,
                            /*warmed_up=*/true, conf);
  if (!conf2) {
    // SignalGate already tallied reason; we return NONE to avoid double counting specific reason here.
    return {std::nullopt, DropReason::NONE};
  }

  StrategySignal out = *raw;
  out.confidence = *conf2;
  return {out, DropReason::NONE};
}

} // namespace sentio

```

## 📄 **FILE 170 of 180**: src/signal_gate.cpp

**File Information**:
- **Path**: `src/signal_gate.cpp`

- **Size**: 51 lines
- **Modified**: 2025-09-12 10:12:46

- **Type**: .cpp

```text
#include "sentio/signal_gate.hpp"
#include <cmath>
#include <algorithm>

namespace sentio {

SignalHealth::SignalHealth() {
  for (auto r :
       {DropReason::NONE, DropReason::WARMUP,
        DropReason::NAN_INPUT, DropReason::THRESHOLD_TOO_TIGHT, DropReason::COOLDOWN_ACTIVE,
        DropReason::DUPLICATE_BAR_TS}) {
    by_reason.emplace(r, 0ULL);
  }
}
void SignalHealth::incr_emit(){ emitted.fetch_add(1, std::memory_order_relaxed); }
void SignalHealth::incr_drop(DropReason r){
  dropped.fetch_add(1, std::memory_order_relaxed);
  auto it = by_reason.find(r);
  if (it != by_reason.end()) it->second.fetch_add(1, std::memory_order_relaxed);
}

SignalGate::SignalGate(const GateCfg& cfg, SignalHealth* health)
: cfg_(cfg), health_(health) {}

std::optional<double> SignalGate::accept(std::int64_t ts_utc_epoch,
                                         bool inputs_finite,
                                         bool warmed_up,
                                         double conf)
{
  // RTH filtering removed - accepting all trading hours data
  if (!inputs_finite)              { if (health_) health_->incr_drop(DropReason::NAN_INPUT); return std::nullopt; }
  if (!warmed_up)                  { if (health_) health_->incr_drop(DropReason::WARMUP);    return std::nullopt; }
  if (!(std::isfinite(conf)))      { if (health_) health_->incr_drop(DropReason::NAN_INPUT); return std::nullopt; }

  conf = std::clamp(conf, 0.0, 1.0);

  if (conf < cfg_.min_conf)        { if (health_) health_->incr_drop(DropReason::THRESHOLD_TOO_TIGHT); return std::nullopt; }

  // Cooldown (optional)
  if (cooldown_left_ > 0)          { --cooldown_left_; if (health_) health_->incr_drop(DropReason::COOLDOWN_ACTIVE); return std::nullopt; }

  // Debounce duplicate timestamps
  if (last_emit_ts_ == ts_utc_epoch){ if (health_) health_->incr_drop(DropReason::DUPLICATE_BAR_TS); return std::nullopt; }

  last_emit_ts_ = ts_utc_epoch;
  cooldown_left_ = cfg_.cooldown_bars;
  if (health_) health_->incr_emit();
  return conf;
}

} // namespace sentio

```

## 📄 **FILE 171 of 180**: src/signal_pipeline.cpp

**File Information**:
- **Path**: `src/signal_pipeline.cpp`

- **Size**: 43 lines
- **Modified**: 2025-09-12 10:12:46

- **Type**: .cpp

```text
#include "sentio/signal_pipeline.hpp"
#include <cmath>

namespace sentio {

PipelineOut SignalPipeline::on_bar(const StrategyCtx& ctx, const Bar& b, const void* acct) {
  (void)acct; // Avoid unused parameter warning
  strat_->on_bar(ctx, b);
  PipelineOut out{};
  TraceRow tr{};
  tr.ts_utc = ctx.ts_utc_epoch;
  tr.instrument = ctx.instrument;
  tr.close = b.close;
  // RTH field removed - no longer filtering by trading hours
  tr.inputs_finite = std::isfinite(b.close);

  auto sig = strat_->latest();
  if (!sig) {
    tr.reason = TraceReason::NO_STRATEGY_OUTPUT;
    if (trace_) trace_->push(tr);
    return out;
  }
  tr.confidence = sig->confidence;

  // Use the existing signal_gate API
  auto conf2 = gate_.accept(ctx.ts_utc_epoch, tr.inputs_finite, true, sig->confidence);
  if (!conf2) {
    tr.reason = TraceReason::THRESHOLD_TOO_TIGHT; // Default to threshold for now
    if (trace_) trace_->push(tr);
    return out;
  }

  StrategySignal sig2 = *sig; sig2.confidence = *conf2;
  tr.conf_after_gate = *conf2;
  out.signal = sig2;

  // For now, just mark as OK since we don't have full routing implemented
  tr.reason = TraceReason::OK;
  if (trace_) trace_->push(tr);
  return out;
}

} // namespace sentio

```

## 📄 **FILE 172 of 180**: src/signal_trace.cpp

**File Information**:
- **Path**: `src/signal_trace.cpp`

- **Size**: 7 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .cpp

```text
#include "sentio/signal_trace.hpp"

namespace sentio {
std::size_t SignalTrace::count(TraceReason r) const {
  std::size_t n=0; for (auto& x: rows_) if (x.reason==r) ++n; return n;
}
} // namespace sentio

```

## 📄 **FILE 173 of 180**: src/strategy/run_rule_ensemble.cpp

**File Information**:
- **Path**: `src/strategy/run_rule_ensemble.cpp`

- **Size**: 79 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .cpp

```text
#include "sentio/rules/registry.hpp"
#include "sentio/rules/integrated_rule_ensemble.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using json = nlohmann::json;

struct Data { std::vector<long long> ts; std::vector<double> o,h,l,c,v; };

static Data load_csv_simple(const std::string& path){
  Data d; std::ifstream f(path); if(!f){ std::cerr<<"Missing csv: "<<path<<"\n"; return d; }
  std::string line; if(!std::getline(f,line)) return d; // header
  while (std::getline(f,line)){
    if(line.empty()) continue; std::stringstream ss(line); std::string cell; int col=0; long long ts=0; double o=0,h=0,l=0,c=0,v=0;
    while(std::getline(ss,cell,',')){
      switch(col){
        case 0: ts = std::stoll(cell); break;
        case 1: o = std::stod(cell); break;
        case 2: h = std::stod(cell); break;
        case 3: l = std::stod(cell); break;
        case 4: c = std::stod(cell); break;
        case 5: v = std::stod(cell); break;
      }
      col++;
    }
    if(col>=6){ d.ts.push_back(ts); d.o.push_back(o); d.h.push_back(h); d.l.push_back(l); d.c.push_back(c); d.v.push_back(v); }
  }
  return d;
}

static sentio::rules::BarsView as_view(const Data& d){
  return sentio::rules::BarsView{ d.ts.data(), d.o.data(), d.h.data(), d.l.data(), d.c.data(), d.v.data(), (long long)d.c.size() };
}

int main(int argc, char** argv){
  std::string cfg_path="configs/strategies/rule_ensemble.json";
  std::string csv_path="data/QQQ.csv";
  for(int i=1;i<argc;i++){
    std::string a=argv[i];
    if (a=="--cfg" && i+1<argc) cfg_path=argv[++i];
    else if (a=="--csv" && i+1<argc) csv_path=argv[++i];
  }
  json cfg = json::parse(std::ifstream(cfg_path));

  std::vector<std::unique_ptr<sentio::rules::IRuleStrategy>> rulesv;
  for (auto& nm : cfg["rules"]) {
    auto r = sentio::rules::make_rule(nm.get<std::string>());
    if (r) rulesv.push_back(std::move(r));
  }

  sentio::rules::EnsembleConfig ec;
  ec.score_logistic_k = cfg.value("score_logistic_k", 1.2f);
  ec.reliability_window = cfg.value("reliability_window", 512);
  ec.agreement_boost    = cfg.value("agreement_boost", 0.25f);
  ec.min_rules          = cfg.value("min_rules", 1);
  if (cfg.contains("base_weights")) for (auto& w : cfg["base_weights"]) ec.base_weights.push_back(w.get<float>());

  sentio::rules::IntegratedRuleEnsemble ers(std::move(rulesv), ec);

  Data d = load_csv_simple(csv_path);
  auto view = as_view(d);
  std::vector<float> lr(view.n,0.f);
  for (long long i=1;i<view.n;i++) lr[i] = (float)(std::log(std::max(1e-9, view.close[i])) - std::log(std::max(1e-9, view.close[i-1])));

  long long start = ers.warmup();
  long long used=0; double sprob=0.0; long long cnt=0;
  for (long long i=start;i<view.n;i++){
    sentio::rules::EnsembleMeta meta;
    auto p = ers.eval(view, i, (i+1<view.n? std::optional<float>(lr[i+1]) : std::nullopt), &meta);
    if (!p) continue; used++; sprob += *p; cnt++;
  }
  std::cerr << "[RuleEnsemble] n=" << view.n << " used="<<used<<" mean_p="<<(cnt? sprob/cnt:0.0) << "\n";
  return 0;
}



```

## 📄 **FILE 174 of 180**: src/strategy_initialization.cpp

**File Information**:
- **Path**: `src/strategy_initialization.cpp`

- **Size**: 29 lines
- **Modified**: 2025-09-18 12:31:29

- **Type**: .cpp

```text
#include "sentio/base_strategy.hpp"
// REMOVED: strategy_ire.hpp - unused legacy strategy
// REMOVED: strategy_bollinger_squeeze_breakout.hpp - unused legacy strategy
// REMOVED: strategy_kochi_ppo.hpp - unused legacy strategy
#include "sentio/strategy_tfa.hpp"
#include "sentio/strategy_signal_or.hpp"
// REMOVED: rsi_strategy.hpp - unused legacy strategy

namespace sentio {

/**
 * Initialize all strategies in the StrategyFactory
 * This replaces the StrategyRegistry system
 * 
 * Note: Individual strategy files use REGISTER_STRATEGY macro
 * which automatically registers strategies, so manual registration
 * is no longer needed here.
 */
bool initialize_strategies() {
    auto& factory = StrategyFactory::instance();
    
    // All strategies are now automatically registered via REGISTER_STRATEGY macro
    // in their respective source files. No manual registration needed.
    
    std::cout << "Registered " << factory.get_available_strategies().size() << " strategies" << std::endl;
    return true;
}

} // namespace sentio

```

## 📄 **FILE 175 of 180**: src/strategy_signal_or.cpp

**File Information**:
- **Path**: `src/strategy_signal_or.cpp`

- **Size**: 222 lines
- **Modified**: 2025-09-18 13:26:59

- **Type**: .cpp

```text
#include "sentio/strategy_signal_or.hpp"
#include "sentio/signal_utils.hpp"
#include "sentio/detectors/rsi_detector.hpp"
#include "sentio/detectors/bollinger_detector.hpp"
#include "sentio/detectors/momentum_volume_detector.hpp"
#include "sentio/detectors/ofi_proxy_detector.hpp"
#include "sentio/detectors/opening_range_breakout_detector.hpp"
#include "sentio/detectors/vwap_reversion_detector.hpp"
// REMOVED: router.hpp - AllocationManager handles routing
// REMOVED: sizer.hpp - handled by runner
// REMOVED: allocation_manager.hpp - handled by runner
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
    
    // **REMOVED**: AllocationManager is handled by runner, not strategy
    // Strategy only provides probability signals
    
    // Initialize integrated detectors
    detectors_.emplace_back(std::make_unique<detectors::RsiDetector>());
    detectors_.emplace_back(std::make_unique<detectors::BollingerDetector>());
    detectors_.emplace_back(std::make_unique<detectors::MomentumVolumeDetector>());
    detectors_.emplace_back(std::make_unique<detectors::OFIProxyDetector>());
    detectors_.emplace_back(std::make_unique<detectors::OpeningRangeBreakoutDetector>());
    detectors_.emplace_back(std::make_unique<detectors::VwapReversionDetector>());
    for (const auto& d : detectors_) max_warmup_ = std::max(max_warmup_, d->warmup_period());

    apply_params();
}

// Required BaseStrategy methods
double SignalOrStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < 0 || current_index >= static_cast<int>(bars.size())) {
        return 0.5; // Neutral if invalid index
    }
    
    warmup_bars_++;

    // If not enough bars for detectors, neutral
    if (current_index < max_warmup_) {
        return 0.5;
    }

    // Run detectors and aggregate majority
    auto ctx = run_and_aggregate(bars, current_index);
    double probability = ctx.final_probability;
    
    // **FIXED**: Update signal diagnostics counter
    diag_.emitted++;
    
    return probability;
}

SignalOrStrategy::AuditContext SignalOrStrategy::run_and_aggregate(const std::vector<Bar>& bars, int idx) {
    AuditContext ctx;
    int total_votes = 0;
    for (const auto& d : detectors_) {
        auto res = d->score(bars, idx);
        ctx.detector_probs[std::string(res.name)] = res.probability;
        if (res.direction == 1) { ctx.long_votes++; total_votes++; }
        else if (res.direction == -1) { ctx.short_votes++; total_votes++; }
    }
    if (total_votes == 0) { ctx.final_probability = 0.5; return ctx; }
    double long_ratio = static_cast<double>(ctx.long_votes) / total_votes;
    if (ctx.long_votes > ctx.short_votes) {
        ctx.final_probability = 0.5 + (long_ratio * 0.5);
        if (long_ratio > 0.8) ctx.final_probability = std::min(0.95, ctx.final_probability + 0.1);
    } else if (ctx.short_votes > ctx.long_votes) {
        double short_ratio = 1.0 - long_ratio;
        ctx.final_probability = 0.5 - (short_ratio * 0.5);
        if (short_ratio > 0.8) ctx.final_probability = std::max(0.05, ctx.final_probability - 0.1);
    } else {
        ctx.final_probability = 0.5;
    }
    return ctx;
}

// REMOVED: get_allocation_decisions - AllocationManager handles all instrument decisions

// REMOVED: get_router_config - AllocationManager handles routing

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

## 📄 **FILE 176 of 180**: src/strategy_tfa.cpp

**File Information**:
- **Path**: `src/strategy_tfa.cpp`

- **Size**: 310 lines
- **Modified**: 2025-09-18 12:31:29

- **Type**: .cpp

```text
#include "sentio/strategy_tfa.hpp"
#include "sentio/tfa/feature_guard.hpp"
#include "sentio/tfa/signal_pipeline.hpp"
#include "sentio/tfa/artifacts_safe.hpp"
#include "sentio/feature/column_projector_safe.hpp"
#include "sentio/feature/name_diff.hpp"
#include "sentio/tfa/tfa_seq_context.hpp"
#include <algorithm>
#include <chrono>

namespace sentio {

static ml::WindowSpec make_spec(const ml::ModelSpec& s){
  ml::WindowSpec w;
  // TFA always uses sequence length of 64 (hardcoded for now since TorchScript doesn't store this)
  w.seq_len = 64;
  w.layout  = s.input_layout.empty()? "BTF" : s.input_layout;
  w.feat_dim = (int)s.feature_names.size();
  // Disable external normalization; model contains its own scaler
  w.mean.clear();
  w.std.clear();
  w.clip2 = s.clip2;
  return w;
}

TFAStrategy::TFAStrategy()
: BaseStrategy("TFA")
, cfg_()
, handle_(ml::ModelRegistryTS::load_torchscript(cfg_.model_id, cfg_.version, cfg_.artifacts_dir, cfg_.use_cuda))
, window_(make_spec(handle_.spec))
{}

TFAStrategy::TFAStrategy(const TFACfg& cfg)
: BaseStrategy("TFA")
, cfg_(cfg)
, handle_(ml::ModelRegistryTS::load_torchscript(cfg.model_id, cfg.version, cfg.artifacts_dir, cfg.use_cuda))
, window_(make_spec(handle_.spec))
{
  // Model loaded successfully
}

void TFAStrategy::apply_params() {
  cfg_.conf_floor = params_["conf_floor"];
}

void TFAStrategy::set_raw_features(const std::vector<double>& raw){
  static int feature_calls = 0;
  feature_calls++;

  // Initialize safe projector on first use
  if (!projector_initialized_) {
    try {
      std::string artifacts_path = cfg_.artifacts_dir + "/" + cfg_.model_id + "/" + cfg_.version + "/";
      auto artifacts = tfa::load_tfa_artifacts_safe(
        artifacts_path + "model.pt",
        artifacts_path + "feature_spec.json",
        artifacts_path + "model.meta.json"
      );

      const int F_expected = artifacts.get_expected_input_dim();
      const auto& expected_names = artifacts.get_expected_feature_names();
      if (F_expected != 55) {
        throw std::runtime_error("Unsupported model input_dim (expect exactly 55)");
      }

      auto runtime_names = tfa::feature_names_from_spec(artifacts.spec);
      float pad_value = artifacts.get_pad_value();


      projector_safe_ = std::make_unique<ColumnProjectorSafe>(
        ColumnProjectorSafe::make(runtime_names, expected_names, pad_value)
      );

      expected_feat_dim_ = F_expected;
      projector_initialized_ = true;
    } catch (const std::exception& e) {
      return;
    }
  }

  // Project raw -> expected order and sanitize, then push into window
  try {
    std::vector<float> proj_f;
    projector_safe_->project_double(raw.data(), 1, raw.size(), proj_f);

    std::vector<double> proj_d;
    proj_d.resize((size_t)expected_feat_dim_);
    for (int i = 0; i < expected_feat_dim_; ++i) {
      float v = (i < (int)proj_f.size() && std::isfinite(proj_f[(size_t)i])) ? proj_f[(size_t)i] : 0.0f;
      proj_d[(size_t)i] = static_cast<double>(v);
    }

    window_.push(proj_d);
  } catch (const std::exception& e) {
  }

}

StrategySignal TFAStrategy::map_output(const ml::ModelOutput& mo) const {
  StrategySignal s;
  // If explicit probabilities are provided
  if (!mo.probs.empty()) {
    if (mo.probs.size() == 1) {
      float prob = mo.probs[0];
      if (prob > 0.5f) {
        s.type = StrategySignal::Type::BUY;
        s.confidence = prob;
      } else {
        s.type = StrategySignal::Type::SELL;
        s.confidence = 1.0f - prob;
      }
      return s;
    }
    // 3-class path
    int argmax = 0; 
    for (int i=1;i<(int)mo.probs.size();++i) 
      if (mo.probs[i]>mo.probs[argmax]) argmax=i;
    float pmax = mo.probs[argmax];
    if      (argmax==2) s.type = StrategySignal::Type::BUY;
    else if (argmax==0) s.type = StrategySignal::Type::SELL;
    else                s.type = StrategySignal::Type::HOLD;
    s.confidence = std::max(cfg_.conf_floor, (double)pmax);
    return s;
  }

  // Fallback: logits-only path (binary)
  float logit = mo.score;
  float prob = 1.0f / (1.0f + std::exp(-logit));
  if (prob > 0.5f) {
    s.type = StrategySignal::Type::BUY;
    s.confidence = prob;
  } else {
    s.type = StrategySignal::Type::SELL;
    s.confidence = 1.0f - prob;
  }
  return s;
}

void TFAStrategy::on_bar(const StrategyCtx& ctx, const Bar& b){
  (void)ctx; (void)b;
  last_.reset();
  
  if (!window_.ready()) {
    return;
  }

  auto in = window_.to_input();
  if (!in) {
    return;
  }

  std::optional<ml::ModelOutput> out;
  try {
    out = handle_.model->predict(*in, window_.seq_len(), window_.feat_dim(), handle_.spec.input_layout);
  } catch (const std::exception& e) {
    return;
  }
  
  if (!out) {
    return;
  }

  auto sig = map_output(*out);
  if (sig.confidence < cfg_.conf_floor) {
    return;
  }
  last_ = sig;
}

ParameterMap TFAStrategy::get_default_params() const {
  return {
    {"conf_floor", cfg_.conf_floor}
  };
}

ParameterSpace TFAStrategy::get_param_space() const {
  return {
    {"conf_floor", {ParamType::FLOAT, 0.0, 1.0, cfg_.conf_floor}}
  };
}

double TFAStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
  (void)current_index; // we will use bars.size() and a member cursor
  ++probability_calls_;

  // One-time: precompute probabilities over the whole series using the sequence context
  if (!seq_context_initialized_) {
    try {
      std::string artifacts_path = cfg_.artifacts_dir + "/" + cfg_.model_id + "/" + cfg_.version + "/";
      seq_context_.load(artifacts_path + "model.pt",
                       artifacts_path + "feature_spec.json",
                       artifacts_path + "model.meta.json");
      // Assume base symbol is QQQ for this test run
      seq_context_.forward_probs("QQQ", bars, precomputed_probabilities_);
      seq_context_initialized_ = true;
    } catch (const std::exception& e) {
      return 0.5; // Neutral
    }
  }

  // Maintain rolling threshold logic with cooldown based on precomputed prob at this call index
  float prob = (probability_calls_-1 < (int)precomputed_probabilities_.size()) ? 
               precomputed_probabilities_[(size_t)(probability_calls_-1)] : 0.5f;

  probability_history_.reserve(4096);
  const int window = 250;
  const float q_long = 0.70f, q_short = 0.30f;
  const float floor_long = 0.51f, ceil_short = 0.49f;
  const int cooldown = 5;

  probability_history_.push_back(prob);

  if ((int)probability_history_.size() >= std::max(window, seq_context_.T)) {
    int end = (int)probability_history_.size() - 1;
    int start = std::max(0, end - window + 1);
    std::vector<float> win(probability_history_.begin() + start, probability_history_.begin() + end + 1);

    int kL = (int)std::floor(q_long * (win.size() - 1));
    std::nth_element(win.begin(), win.begin() + kL, win.end());
    float thrL = std::max(floor_long, win[kL]);

    int kS = (int)std::floor(q_short * (win.size() - 1));
    std::nth_element(win.begin(), win.begin() + kS, win.end());
    float thrS = std::min(ceil_short, win[kS]);

    bool can_long = (probability_calls_ >= cooldown_long_until_);
    bool can_short = (probability_calls_ >= cooldown_short_until_);

    if (can_long && prob >= thrL) {
      cooldown_long_until_ = probability_calls_ + cooldown;
      return prob; // Return the probability directly
    } else if (can_short && prob <= thrS) {
      cooldown_short_until_ = probability_calls_ + cooldown;
      return prob; // Return the probability directly
    }

  }

  return 0.5; // Neutral
}

// REMOVED: get_allocation_decisions - AllocationManager handles all instrument decisions
/*
std::vector<BaseStrategy::AllocationDecision> TFAStrategy::get_allocation_decisions_REMOVED(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // **PROFIT MAXIMIZATION**: Always deploy 100% of capital with maximum leverage
    if (probability > 0.7) {
        // Strong buy: 100% TQQQ (3x leveraged long)
        decisions.push_back({bull3x_symbol, 1.0, probability, "TFA strong buy: 100% TQQQ (3x leverage)"});
        
    } else if (probability > 0.51) {
        // Moderate buy: 100% QQQ (1x long)
        decisions.push_back({base_symbol, 1.0, probability, "TFA moderate buy: 100% QQQ"});
        
    } else if (probability < 0.3) {
        // Strong sell: 100% SQQQ (3x leveraged short)
        decisions.push_back({bear3x_symbol, 1.0, 1.0 - probability, "TFA strong sell: 100% SQQQ (3x inverse)"});
        
    } else if (probability < 0.49) {
        // Weak sell: 100% PSQ (1x inverse) - NEW ADDITION
        decisions.push_back({"PSQ", 1.0, 1.0 - probability, "TFA weak sell: 100% PSQ (1x inverse)"});
        
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
            decisions.push_back({inst, 0.0, 0.0, "TFA: Flatten unused instrument"});
        }
    }
    
    return decisions;
}
*/

// REMOVED: get_router_config - AllocationManager handles routing
/*
RouterCfg TFAStrategy::get_router_config_REMOVED() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}
*/

// REMOVED: get_sizer_config() - No artificial limits allowed
// Sizer will use profit-maximizing defaults: 100% capital deployment, maximum leverage

REGISTER_STRATEGY(TFAStrategy, "tfa");

} // namespace sentio

```

## 📄 **FILE 177 of 180**: src/test_strategy.cpp

**File Information**:
- **Path**: `src/test_strategy.cpp`

- **Size**: 349 lines
- **Modified**: 2025-09-18 12:31:29

- **Type**: .cpp

```text
#include "sentio/test_strategy.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace sentio {

TestStrategy::TestStrategy(const TestStrategyConfig& config)
    : BaseStrategy("TestStrategy"), config_(config), rng_(config_.random_seed), uniform_dist_(0.0, 1.0) {
    reset_test_state();
}

void TestStrategy::reset_test_state() {
    signal_history_.clear();
    accuracy_history_.clear();
    last_signal_bar_ = -1;
    rng_.seed(config_.random_seed);
}

double TestStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < 0 || current_index >= static_cast<int>(bars.size())) {
        return 0.5; // Neutral for invalid index
    }
    
    // **PERFECT FORESIGHT MODE**: Always predict correctly
    if (config_.perfect_foresight) {
        double future_return = calculate_future_return(bars, current_index);
        double probability = 0.5 + future_return * 10.0; // Scale return to probability
        probability = std::max(0.0, std::min(1.0, probability)); // Clamp to [0,1]
        
        signal_history_.push_back(probability);
        accuracy_history_.push_back(1.0); // Perfect accuracy
        return probability;
    }
    
    // **CONTROLLED ACCURACY MODE**: Generate signal with target accuracy
    double future_return = calculate_future_return(bars, current_index);
    
    // Determine true signal direction
    double true_probability = 0.5;
    if (std::abs(future_return) > config_.signal_threshold) {
        true_probability = future_return > 0 ? 0.7 : 0.3; // Strong directional signal
    }
    
    // Generate signal with controlled accuracy
    double generated_signal = generate_signal_with_accuracy(true_probability, config_.target_accuracy);
    
    // Add noise to make it more realistic
    generated_signal = add_signal_noise(generated_signal);
    
    // Clamp to valid probability range
    generated_signal = std::max(0.0, std::min(1.0, generated_signal));
    
    // Track signal for accuracy calculation
    signal_history_.push_back(generated_signal);
    
    // Calculate actual accuracy so far
    if (signal_history_.size() > 1) {
        int correct_predictions = 0;
        for (size_t i = 0; i < signal_history_.size() - 1; ++i) {
            bool predicted_up = signal_history_[i] > 0.5;
            
            // Get actual direction from next bar
            if (i + 1 < bars.size()) {
                double actual_return = (bars[i + 1].close - bars[i].close) / bars[i].close;
                bool actual_up = actual_return > 0;
                
                if (predicted_up == actual_up) {
                    correct_predictions++;
                }
            }
        }
        
        double actual_accuracy = static_cast<double>(correct_predictions) / (signal_history_.size() - 1);
        accuracy_history_.push_back(actual_accuracy);
    }
    
    last_signal_bar_ = current_index;
    return generated_signal;
}

// REMOVED: get_allocation_decisions - AllocationManager handles all instrument decisions
/*
std::vector<TestStrategy::AllocationDecision> TestStrategy::get_allocation_decisions_REMOVED(
    const std::vector<Bar>& bars,
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    if (current_index < 0 || current_index >= static_cast<int>(bars.size())) {
        return decisions;
    }
    
    // Get probability signal
    double probability = calculate_probability(bars, current_index);
    double signal_strength = std::abs(probability - 0.5) * 2.0; // 0 to 1 scale
    
    // **ALLOCATION LOGIC**: Based on signal strength and configuration
    std::string target_instrument;
    double target_weight = 0.0;
    std::string reason;
    
    if (probability > (0.5 + config_.neutral_band)) {
        // **BULLISH SIGNAL**
        if (config_.enable_leverage && signal_strength > config_.strong_signal_threshold) {
            target_instrument = bull3x_symbol; // TQQQ
            target_weight = 1.0;
            reason = "Test Strategy: Strong bullish signal -> 3x leverage";
        } else if (signal_strength > config_.weak_signal_threshold) {
            target_instrument = base_symbol; // QQQ
            target_weight = 1.0;
            reason = "Test Strategy: Moderate bullish signal -> 1x long";
        }
        
    } else if (probability < (0.5 - config_.neutral_band)) {
        // **BEARISH SIGNAL**
        if (config_.enable_leverage && signal_strength > config_.strong_signal_threshold) {
            target_instrument = bear3x_symbol; // SQQQ
            target_weight = 1.0;
            reason = "Test Strategy: Strong bearish signal -> 3x inverse";
        } else if (signal_strength > config_.weak_signal_threshold) {
            target_instrument = "PSQ"; // 1x inverse
            target_weight = 1.0;
            reason = "Test Strategy: Moderate bearish signal -> 1x inverse";
        }
    }
    
    // Only return decision if we have a target and it's a new bar
    if (!target_instrument.empty() && current_index != last_signal_bar_) {
        decisions.push_back({target_instrument, target_weight, signal_strength, reason});
    }
    
    return decisions;
}
*/

double TestStrategy::calculate_future_return(const std::vector<Bar>& bars, int current_index) const {
    if (current_index + config_.lookhead_bars >= static_cast<int>(bars.size())) {
        return 0.0; // No future data available
    }
    
    const Bar& current_bar = bars[current_index];
    const Bar& future_bar = bars[current_index + config_.lookhead_bars];
    
    return (future_bar.close - current_bar.close) / current_bar.close;
}

double TestStrategy::generate_signal_with_accuracy(double true_signal, double target_accuracy) const {
    double random_value = uniform_dist_(rng_);
    
    if (random_value < target_accuracy) {
        // Generate correct signal
        return true_signal;
    } else {
        // Generate incorrect signal (flip direction)
        if (true_signal > 0.5) {
            return 0.5 - (true_signal - 0.5); // Flip to bearish
        } else {
            return 0.5 + (0.5 - true_signal); // Flip to bullish
        }
    }
}

double TestStrategy::add_signal_noise(double signal) const {
    if (config_.noise_factor <= 0.0) return signal;
    
    // Add Gaussian noise
    std::normal_distribution<double> noise_dist(0.0, config_.noise_factor);
    double noise = noise_dist(rng_);
    
    return signal + noise;
}

bool TestStrategy::should_use_leverage(double signal_strength) const {
    return config_.enable_leverage && signal_strength > config_.strong_signal_threshold;
}

// REMOVED: get_router_config - AllocationManager handles routing
/*
RouterCfg TestStrategy::get_router_config_REMOVED() const {
    RouterCfg cfg;
    cfg.base_symbol = "QQQ";
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    cfg.min_signal_strength = config_.signal_threshold;
    cfg.max_position_pct = 1.0; // 100% position for profit maximization
    return cfg;
}
*/

bool TestStrategy::allows_simultaneous_positions(const std::string& instrument1, const std::string& instrument2) const {
    // Define conflicting instrument groups
    std::vector<std::string> long_instruments = {"QQQ", "TQQQ"};
    std::vector<std::string> inverse_instruments = {"PSQ", "SQQQ"};
    
    auto is_long = [&](const std::string& inst) {
        return std::find(long_instruments.begin(), long_instruments.end(), inst) != long_instruments.end();
    };
    auto is_inverse = [&](const std::string& inst) {
        return std::find(inverse_instruments.begin(), inverse_instruments.end(), inst) != inverse_instruments.end();
    };
    
    // Cannot hold multiple long instruments
    if (is_long(instrument1) && is_long(instrument2)) return false;
    
    // Cannot hold multiple inverse instruments
    if (is_inverse(instrument1) && is_inverse(instrument2)) return false;
    
    // Cannot hold long + inverse simultaneously
    if ((is_long(instrument1) && is_inverse(instrument2)) || 
        (is_inverse(instrument1) && is_long(instrument2))) return false;
    
    return true;
}

ParameterMap TestStrategy::get_default_params() const {
    ParameterMap params;
    params["target_accuracy"] = config_.target_accuracy;
    params["lookhead_bars"] = static_cast<double>(config_.lookhead_bars);
    params["signal_threshold"] = config_.signal_threshold;
    params["noise_factor"] = config_.noise_factor;
    params["enable_leverage"] = config_.enable_leverage ? 1.0 : 0.0;
    return params;
}

ParameterSpace TestStrategy::get_param_space() const {
    ParameterSpace space;
    space["target_accuracy"] = {ParamType::FLOAT, 0.0, 1.0, 0.05}; // 0% to 100% accuracy
    space["lookhead_bars"] = {ParamType::INT, 1.0, 10.0, 1.0};   // 1 to 10 bars lookhead
    space["signal_threshold"] = {ParamType::FLOAT, 0.01, 0.10, 0.01}; // 1% to 10% threshold
    space["noise_factor"] = {ParamType::FLOAT, 0.0, 0.5, 0.05};    // 0% to 50% noise
    space["enable_leverage"] = {ParamType::INT, 0.0, 1.0, 1.0};  // Boolean: 0 or 1
    return space;
}

void TestStrategy::apply_params() {
    const auto& params = params_;
    
    if (params.find("target_accuracy") != params.end()) {
        config_.target_accuracy = params.at("target_accuracy");
    }
    if (params.find("lookhead_bars") != params.end()) {
        config_.lookhead_bars = static_cast<int>(params.at("lookhead_bars"));
    }
    if (params.find("signal_threshold") != params.end()) {
        config_.signal_threshold = params.at("signal_threshold");
    }
    if (params.find("noise_factor") != params.end()) {
        config_.noise_factor = params.at("noise_factor");
    }
    if (params.find("enable_leverage") != params.end()) {
        config_.enable_leverage = params.at("enable_leverage") > 0.5;
    }
}

void TestStrategy::set_target_accuracy(double accuracy) {
    config_.target_accuracy = std::max(0.0, std::min(1.0, accuracy));
}

double TestStrategy::get_actual_accuracy() const {
    if (accuracy_history_.empty()) return 0.0;
    return accuracy_history_.back();
}

void TestStrategy::update_config(const TestStrategyConfig& config) {
    config_ = config;
    rng_.seed(config_.random_seed);
}

// **TEST STRATEGY FACTORY IMPLEMENTATION**
std::unique_ptr<TestStrategy> TestStrategyFactory::create_random_strategy(double accuracy) {
    TestStrategyConfig config;
    config.target_accuracy = accuracy;
    config.enable_leverage = false; // Conservative for random strategy
    config.noise_factor = 0.2;      // High noise
    return std::make_unique<TestStrategy>(config);
}

std::unique_ptr<TestStrategy> TestStrategyFactory::create_poor_strategy(double accuracy) {
    TestStrategyConfig config;
    config.target_accuracy = accuracy;
    config.enable_leverage = false;
    config.noise_factor = 0.15;
    config.signal_threshold = 0.05; // Higher threshold (fewer signals)
    return std::make_unique<TestStrategy>(config);
}

std::unique_ptr<TestStrategy> TestStrategyFactory::create_decent_strategy(double accuracy) {
    TestStrategyConfig config;
    config.target_accuracy = accuracy;
    config.enable_leverage = true;  // Enable leverage for decent strategy
    config.noise_factor = 0.1;
    config.signal_threshold = 0.03;
    return std::make_unique<TestStrategy>(config);
}

std::unique_ptr<TestStrategy> TestStrategyFactory::create_good_strategy(double accuracy) {
    TestStrategyConfig config;
    config.target_accuracy = accuracy;
    config.enable_leverage = true;
    config.noise_factor = 0.05;     // Low noise
    config.signal_threshold = 0.02; // Lower threshold (more signals)
    config.strong_signal_threshold = 0.70; // Lower threshold for leverage
    return std::make_unique<TestStrategy>(config);
}

std::unique_ptr<TestStrategy> TestStrategyFactory::create_excellent_strategy(double accuracy) {
    TestStrategyConfig config;
    config.target_accuracy = accuracy;
    config.enable_leverage = true;
    config.noise_factor = 0.02;     // Very low noise
    config.signal_threshold = 0.01; // Very low threshold
    config.strong_signal_threshold = 0.65; // Aggressive leverage usage
    return std::make_unique<TestStrategy>(config);
}

std::unique_ptr<TestStrategy> TestStrategyFactory::create_perfect_strategy() {
    TestStrategyConfig config;
    config.perfect_foresight = true;
    config.target_accuracy = 1.0;
    config.enable_leverage = true;
    config.noise_factor = 0.0;      // No noise
    config.signal_threshold = 0.001; // Capture all movements
    return std::make_unique<TestStrategy>(config);
}

std::vector<std::unique_ptr<TestStrategy>> TestStrategyFactory::create_accuracy_test_suite() {
    std::vector<std::unique_ptr<TestStrategy>> strategies;
    
    // Create strategies with different accuracy levels
    std::vector<double> accuracy_levels = {0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00};
    
    for (double accuracy : accuracy_levels) {
        TestStrategyConfig config;
        config.target_accuracy = accuracy;
        config.enable_leverage = true;
        config.noise_factor = 0.05;
        config.random_seed = 42 + static_cast<unsigned int>(accuracy * 100); // Different seed per strategy
        
        strategies.push_back(std::make_unique<TestStrategy>(config));
    }
    
    return strategies;
}

} // namespace sentio

```

## 📄 **FILE 178 of 180**: src/time_utils.cpp

**File Information**:
- **Path**: `src/time_utils.cpp`

- **Size**: 141 lines
- **Modified**: 2025-09-16 00:39:14

- **Type**: .cpp

```text
#include "sentio/time_utils.hpp"
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <sstream>

#if __has_include(<chrono>)
  #include <chrono>
  using namespace std::chrono;
#else
  #error "Need C++20 <chrono>. If missing tzdb, you can still normalize UTC here and handle ET in calendar."
#endif

namespace sentio {

bool iso8601_looks_like(const std::string& s) {
  // Very light heuristic: "YYYY-MM-DDTHH:MM" and either 'Z' or +/-HH:MM
  return s.size() >= 16 && s[4]=='-' && s[7]=='-' && (s.find('T') != std::string::npos);
}

bool epoch_ms_suspected(double v_ms) {
  // If it's larger than ~1e12 it's probably ms; (1e12 sec is ~31k years)
  return std::isfinite(v_ms) && v_ms > 1.0e11;
}

static inline sys_seconds parse_iso8601_to_utc(const std::string& s) {
  // Minimal ISO8601 handling: require offset or 'Z'.
  // For robustness use Howard Hinnant's date::parse with %FT%T%Ez.
  // Here we support the common forms: 2022-09-06T13:30:00Z and 2022-09-06T09:30:00-04:00
  // We'll implement a tiny parser that splits offset and adjusts.
  auto posT = s.find('T');
  if (posT == std::string::npos) throw std::runtime_error("ISO8601 missing T");
  // Find offset start: last char 'Z' or last '+'/'-'
  int sign = 0;
  int oh=0, om=0;
  bool zulu = false;
  std::size_t offPos = s.rfind('Z');
  if (offPos != std::string::npos && offPos > posT) {
    zulu = true;
  } else {
    std::size_t plus = s.rfind('+');
    std::size_t minus= s.rfind('-');
    std::size_t off  = std::string::npos;
    if (plus!=std::string::npos && plus>posT) { off=plus; sign=+1; }
    else if (minus!=std::string::npos && minus>posT) { off=minus; sign=-1; }
    if (off==std::string::npos) throw std::runtime_error("ISO8601 missing offset/Z");
    // parse HH:MM
    if (off+3 >= s.size()) throw std::runtime_error("Bad offset");
    oh = std::stoi(s.substr(off+1,2));
    if (off+6 <= s.size() && s[off+3]==':') om = std::stoi(s.substr(off+4,2));
  }

  // parse date/time parts (seconds optional)
  int Y = std::stoi(s.substr(0,4));
  int M = std::stoi(s.substr(5,2));
  int D = std::stoi(s.substr(8,2));
  int h = std::stoi(s.substr(posT+1,2));
  int m = std::stoi(s.substr(posT+4,2));
  int sec = 0;
  if (posT+6 < s.size() && s[posT+6]==':') {
    sec = std::stoi(s.substr(posT+7,2));
  }

  // Treat parsed time as local-time-with-offset; compute UTC by subtracting offset
  using namespace std::chrono;
  sys_days sd = sys_days(std::chrono::year{Y}/M/D);
  seconds local = hours{h} + minutes{m} + seconds{sec};
  seconds off = seconds{ (oh*3600 + om*60) * (zulu ? 0 : sign) };
  // If sign=+1 (e.g., +09:00), local = UTC + offset => UTC = local - offset
  seconds utc_sec = local - off;
  return sys_seconds{sd.time_since_epoch() + utc_sec};
}

std::chrono::sys_seconds to_utc_sys_seconds(const std::variant<std::int64_t, double, std::string>& ts) {
  if (std::holds_alternative<std::int64_t>(ts)) {
    // epoch seconds
    return std::chrono::sys_seconds{std::chrono::seconds{std::get<std::int64_t>(ts)}};
  }
  if (std::holds_alternative<double>(ts)) {
    // Could be epoch ms or sec (float). Prefer ms detection and round down.
    double v = std::get<double>(ts);
    if (!std::isfinite(v)) throw std::runtime_error("Non-finite epoch");
    if (epoch_ms_suspected(v)) {
      auto s = static_cast<std::int64_t>(v / 1000.0);
      return std::chrono::sys_seconds{std::chrono::seconds{s}};
    } else {
      auto s = static_cast<std::int64_t>(v);
      return std::chrono::sys_seconds{std::chrono::seconds{s}};
    }
  }
  const std::string& s = std::get<std::string>(ts);
  if (!iso8601_looks_like(s)) {
    // fall back: try integer seconds in string
    std::int64_t v{};
    auto sv = std::string_view{s};
    if (auto [p, ec] = std::from_chars(sv.data(), sv.data()+sv.size(), v); ec == std::errc{}) {
      return std::chrono::sys_seconds{std::chrono::seconds{v}};
    }
    throw std::runtime_error("Unrecognized timestamp format: " + s);
  }
  return parse_iso8601_to_utc(s);
}

std::string calculate_start_date(int years, int months, int days) {
    std::time_t now = std::time(nullptr);
    std::time_t yesterday = now - 24 * 60 * 60; // Start from yesterday
    
    std::tm* tm_start = std::gmtime(&yesterday);
    
    if (years > 0) {
        tm_start->tm_year -= years;
    } else if (months > 0) {
        tm_start->tm_mon -= months;
        if (tm_start->tm_mon < 0) {
            tm_start->tm_mon += 12;
            tm_start->tm_year--;
        }
    } else if (days > 0) {
        tm_start->tm_mday -= days;
        // Let mktime handle month/year overflow
        std::mktime(tm_start);
    } else {
        // Default: 3 years (now explicit default)
        tm_start->tm_year -= 3;
    }
    
    // Normalize the time
    std::mktime(tm_start);
    
    // Format as YYYY-MM-DD
    std::ostringstream oss;
    oss << std::put_time(tm_start, "%Y-%m-%d");
    return oss.str();
}

} // namespace sentio
```

## 📄 **FILE 179 of 180**: src/unified_metrics.cpp

**File Information**:
- **Path**: `src/unified_metrics.cpp`

- **Size**: 280 lines
- **Modified**: 2025-09-16 02:01:42

- **Type**: .cpp

```text
#include "sentio/unified_metrics.hpp"
#include "sentio/metrics.hpp"
#include "sentio/metrics/mpr.hpp"
#include "sentio/cost_model.hpp"
#include "sentio/side.hpp"
#include "audit/audit_db.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <map>
#include <ctime>
#include <sstream>
#include <iomanip>

namespace sentio {

UnifiedMetricsReport UnifiedMetricsCalculator::calculate_metrics(const BacktestOutput& output) {
    UnifiedMetricsReport report{};
    if (output.equity_curve.empty()) {
        return report;
    }

    // 1. Use the existing, robust day-aware calculation
    RunSummary summary = calculate_from_equity_curve(output.equity_curve, output.total_fills);

    // 2. Daily returns are implicit in compute_metrics_day_aware; no separate derivation needed
    
    // 3. Populate the final, unified report
    report.final_equity = output.equity_curve.back().second;
    report.total_return = summary.ret_total;
    report.sharpe_ratio = summary.sharpe;
    report.max_drawdown = summary.mdd;
    report.monthly_projected_return = summary.monthly_proj;
    report.total_fills = output.total_fills;
    report.avg_daily_trades = output.run_trading_days > 0 ? 
        static_cast<double>(output.total_fills) / output.run_trading_days : 0.0;
        
    return report;
}

RunSummary UnifiedMetricsCalculator::calculate_from_equity_curve(
    const std::vector<std::pair<std::string, double>>& equity_curve,
    int fills_count,
    bool include_fees
) {
    // Use the authoritative compute_metrics_day_aware function
    // This ensures consistency across all systems
    return compute_metrics_day_aware(equity_curve, fills_count);
}

RunSummary UnifiedMetricsCalculator::calculate_from_audit_events(
    const std::vector<audit::Event>& events,
    double initial_capital,
    bool include_fees
) {
    // Independent BAR-driven daily-close reconstruction (UTC day buckets)
    double cash = initial_capital;
    std::unordered_map<std::string, double> positions;   // symbol -> qty
    std::unordered_map<std::string, double> last_prices; // symbol -> last price

    // Session-aware trading day bucketing: map timestamps to US/Eastern calendar day.
    // Approximate EST/EDT by using UTC-5h in winter (our current datasets span Jan–Feb).
    // For broader periods, this can be extended to handle DST.
    const std::int64_t eastern_offset_ms = 5LL * 60LL * 60LL * 1000LL; // UTC-5
    auto ts_to_day = [eastern_offset_ms](std::int64_t ts_ms) -> std::string {
        std::int64_t shifted = ts_ms - eastern_offset_ms;
        std::time_t secs = static_cast<std::time_t>(shifted / 1000);
        std::tm* tm_utc = std::gmtime(&secs);
        char buf[16];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d", tm_utc);
        return std::string(buf);
    };

    auto compute_mtm = [&]() -> double {
        double mtm = cash;
        for (const auto& kv : positions) {
            const std::string& sym = kv.first;
            double qty = kv.second;
            auto it = last_prices.find(sym);
            if (std::abs(qty) > 1e-8 && it != last_prices.end()) {
                mtm += qty * it->second;
            }
        }
        return mtm;
    };

    std::vector<std::pair<std::string,double>> daily_equity; // (YYYY-MM-DD 23:59:59, equity)
    std::string current_day;
    double last_mtm = initial_capital;

    int fills_count = 0;

    for (const auto& e : events) {
        if (e.kind == "FILL") {
            // Fees
            double fees = include_fees ? calculate_transaction_fees(e.symbol, (e.side == "SELL" ? -e.qty : e.qty), e.price) : 0.0;
            // Cash
            if (e.side == "SELL") {
                cash += e.price * e.qty - fees;
            } else {
                cash -= e.price * e.qty + fees;
            }
            // Position update
            double delta = (e.side == "SELL") ? -e.qty : e.qty;
            positions[e.symbol] += delta;
            // Update last trade price for symbol
            last_prices[e.symbol] = e.price;
            ++fills_count;
            // Do not emit a day point here; wait for BAR to set day boundary
        } else if (e.kind == "BAR") {
            // Update price book
            last_prices[e.symbol] = e.price;
            // Determine day from BAR timestamp
            std::string day = ts_to_day(e.ts_millis);
            // If day changed, emit previous day's close using last_mtm
            if (!current_day.empty() && day != current_day) {
                daily_equity.emplace_back(current_day + " 23:59:59", last_mtm);
                current_day = day;
            } else if (current_day.empty()) {
                current_day = day;
            }
            // Recompute MTM after this BAR
            last_mtm = compute_mtm();
        }
    }

    // Emit final day close
    if (!current_day.empty()) {
        daily_equity.emplace_back(current_day + " 23:59:59", last_mtm);
    } else {
        // No BARs; fallback to a start/end two-point flat series
        daily_equity.emplace_back("1970-01-01 23:59:59", initial_capital);
        daily_equity.emplace_back("1970-01-02 23:59:59", initial_capital);
    }

    return calculate_from_equity_curve(daily_equity, fills_count, include_fees);
}

std::vector<std::pair<std::string, double>> UnifiedMetricsCalculator::reconstruct_equity_curve_from_events(
    const std::vector<audit::Event>& events,
    double initial_capital,
    bool include_fees
) {
    std::vector<std::pair<std::string, double>> equity_curve;
    
    // Track portfolio state
    double cash = initial_capital;
    std::unordered_map<std::string, double> positions; // symbol -> quantity
    std::unordered_map<std::string, double> avg_prices; // symbol -> average price
    std::unordered_map<std::string, double> last_prices; // symbol -> last known price
    
    // Process events chronologically
    for (const auto& event : events) {
        if (event.kind == "FILL") {
            const std::string& symbol = event.symbol;
            double quantity = event.qty;
            double price = event.price;
            bool is_sell = (event.side == "SELL");
            
            // Convert order side to position impact
            double position_delta = is_sell ? -quantity : quantity;
            
            // Calculate transaction fees if enabled
            double fees = 0.0;
            if (include_fees) {
                fees = calculate_transaction_fees(symbol, position_delta, price);
            }
            
            // Update cash (buy decreases cash, sell increases cash)
            double cash_delta = is_sell ? (price * quantity - fees) : -(price * quantity + fees);
            cash += cash_delta;
            
            // Update position using VWAP
            double current_qty = positions[symbol];
            double new_qty = current_qty + position_delta;
            
            if (std::abs(new_qty) < 1e-8) {
                // Position closed
                positions[symbol] = 0.0;
                avg_prices[symbol] = 0.0;
            } else if (std::abs(current_qty) < 1e-8) {
                // Opening new position
                positions[symbol] = new_qty;
                avg_prices[symbol] = price;
            } else if ((current_qty > 0) == (position_delta > 0)) {
                // Adding to same side - update VWAP
                avg_prices[symbol] = (avg_prices[symbol] * current_qty + price * position_delta) / new_qty;
                positions[symbol] = new_qty;
            } else {
                // Reducing or flipping position
                positions[symbol] = new_qty;
                if (std::abs(new_qty) > 1e-8) {
                    avg_prices[symbol] = price; // New average for remaining position
                }
            }
            
            // Update last known price
            last_prices[symbol] = price;
            
            // Calculate mark-to-market equity
            double mtm_value = cash;
            for (const auto& [sym, qty] : positions) {
                if (std::abs(qty) > 1e-8 && last_prices.count(sym)) {
                    mtm_value += qty * last_prices[sym];
                }
            }
            
            // Add to equity curve with ISO timestamp for proper day compression
            std::time_t secs = static_cast<std::time_t>(event.ts_millis / 1000);
            std::tm* tm_utc = std::gmtime(&secs);
            char buf[32];
            std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm_utc);
            std::string timestamp(buf);
            equity_curve.emplace_back(timestamp, mtm_value);
        }
        else if (event.kind == "BAR") {
            // Update last known prices from bar data
            // This ensures mark-to-market calculations use current prices
            last_prices[event.symbol] = event.price;
            
            // Recalculate mark-to-market equity with updated prices
            double mtm_value = cash;
            for (const auto& [sym, qty] : positions) {
                if (std::abs(qty) > 1e-8 && last_prices.count(sym)) {
                    mtm_value += qty * last_prices[sym];
                }
            }
            
            // Add to equity curve if we have positions or this is a significant update
            if (!positions.empty() || equity_curve.empty()) {
                std::time_t secs = static_cast<std::time_t>(event.ts_millis / 1000);
                std::tm* tm_utc = std::gmtime(&secs);
                char buf[32];
                std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm_utc);
                std::string timestamp(buf);
                equity_curve.emplace_back(timestamp, mtm_value);
            }
        }
    }
    
    // Ensure we have at least initial and final equity points
    if (equity_curve.empty()) {
        equity_curve.emplace_back("start", initial_capital);
        equity_curve.emplace_back("end", initial_capital);
    } else if (equity_curve.size() == 1) {
        equity_curve.emplace_back("end", equity_curve[0].second);
    }
    
    return equity_curve;
}

double UnifiedMetricsCalculator::calculate_transaction_fees(
    const std::string& symbol,
    double quantity,
    double price
) {
    bool is_sell = (quantity < 0);
    return AlpacaCostModel::calculate_fees(symbol, std::abs(quantity), price, is_sell);
}

bool UnifiedMetricsCalculator::validate_metrics_consistency(
    const RunSummary& metrics1,
    const RunSummary& metrics2,
    double tolerance_pct
) {
    auto within_tolerance = [tolerance_pct](double a, double b) -> bool {
        if (std::abs(a) < 1e-8 && std::abs(b) < 1e-8) return true;
        double diff_pct = std::abs(a - b) / std::max(std::abs(a), std::abs(b)) * 100.0;
        return diff_pct <= tolerance_pct;
    };
    
    return within_tolerance(metrics1.ret_total, metrics2.ret_total) &&
           within_tolerance(metrics1.ret_ann, metrics2.ret_ann) &&
           within_tolerance(metrics1.monthly_proj, metrics2.monthly_proj) &&
           within_tolerance(metrics1.sharpe, metrics2.sharpe) &&
           within_tolerance(metrics1.mdd, metrics2.mdd) &&
           (metrics1.trades == metrics2.trades);
}

} // namespace sentio

```

## 📄 **FILE 180 of 180**: src/virtual_market.cpp

**File Information**:
- **Path**: `src/virtual_market.cpp`

- **Size**: 654 lines
- **Modified**: 2025-09-17 15:47:09

- **Type**: .cpp

```text
#include "sentio/virtual_market.hpp"
#include "sentio/runner.hpp"
// #include "sentio/temporal_analysis.hpp" // Removed during cleanup
#include "sentio/symbol_table.hpp"
#include "sentio/audit.hpp"
#include "audit/audit_db_recorder.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/mars_data_loader.hpp"
#include "sentio/future_qqq_loader.hpp"
#include "sentio/leverage_aware_csv_loader.hpp"
#include "sentio/run_id_generator.hpp"
#include "sentio/dataset_metadata.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>

namespace sentio {

// **PROFIT MAXIMIZATION**: Generate theoretical leverage series for maximum capital deployment
std::vector<Bar> generate_theoretical_leverage_series(const std::vector<Bar>& base_series, double leverage_factor) {
    std::vector<Bar> leverage_series;
    leverage_series.reserve(base_series.size());
    
    if (base_series.empty()) return leverage_series;
    
    // Initialize with first bar (no leverage effect on first bar)
    Bar first_bar = base_series[0];
    leverage_series.push_back(first_bar);
    
    // Generate subsequent bars with leverage effect
    for (size_t i = 1; i < base_series.size(); ++i) {
        const Bar& prev_base = base_series[i-1];
        const Bar& curr_base = base_series[i];
        const Bar& prev_leverage = leverage_series[i-1];
        
        // Calculate base return
        double base_return = (curr_base.close - prev_base.close) / prev_base.close;
        
        // Apply leverage factor
        double leverage_return = base_return * leverage_factor;
        
        // Calculate new leverage prices
        Bar leverage_bar;
        leverage_bar.ts_utc_epoch = curr_base.ts_utc_epoch;
        leverage_bar.close = prev_leverage.close * (1.0 + leverage_return);
        
        // Approximate OHLV based on close price movement
        double price_ratio = leverage_bar.close / prev_leverage.close;
        leverage_bar.open = prev_leverage.close;  // Open at previous close
        leverage_bar.high = std::max(leverage_bar.open, leverage_bar.close) * 1.001;  // Small spread
        leverage_bar.low = std::min(leverage_bar.open, leverage_bar.close) * 0.999;   // Small spread
        leverage_bar.volume = curr_base.volume;  // Use base volume
        
        leverage_series.push_back(leverage_bar);
    }
    
    return leverage_series;
}

VirtualMarketEngine::VirtualMarketEngine() 
    : rng_(std::chrono::steady_clock::now().time_since_epoch().count()) {
    initialize_market_regimes();
    
    // Initialize base prices for common symbols (updated to match real market levels)
    base_prices_["QQQ"] = 458.0; // Updated to match real QQQ levels
    base_prices_["SPY"] = 450.0;
    base_prices_["AAPL"] = 175.0;
    base_prices_["MSFT"] = 350.0;
    base_prices_["TSLA"] = 250.0;
    // PSQ removed - moderate sell signals now use SHORT QQQ
    base_prices_["TQQQ"] = 120.0; // 3x QQQ
    base_prices_["SQQQ"] = 120.0; // 3x inverse QQQ
}

void VirtualMarketEngine::initialize_market_regimes() {
    market_regimes_ = {
        {"BULL_TRENDING", 0.015, 0.02, 0.3, 1.2, 60},
        {"BEAR_TRENDING", 0.025, -0.015, 0.2, 1.5, 45},
        {"SIDEWAYS_LOW_VOL", 0.008, 0.001, 0.8, 0.8, 90},
        {"SIDEWAYS_HIGH_VOL", 0.020, 0.002, 0.6, 1.3, 30},
        {"VOLATILE_BREAKOUT", 0.035, 0.025, 0.1, 2.0, 15},
        {"VOLATILE_BREAKDOWN", 0.040, -0.030, 0.1, 2.2, 20},
        {"NORMAL_MARKET", 0.008, 0.001, 0.5, 1.0, 120}
    };
    
    // Select initial regime
    select_new_regime();
}

void VirtualMarketEngine::select_new_regime() {
    // Weight regimes by probability (normal market is most common)
    std::vector<double> weights = {0.15, 0.10, 0.20, 0.15, 0.05, 0.05, 0.30};
    
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    int regime_index = dist(rng_);
    
    current_regime_ = market_regimes_[regime_index];
    regime_start_time_ = std::chrono::system_clock::now();
    
    // Debug: Regime switching (commented out to reduce console noise)
    // std::cout << "🔄 Switched to market regime: " << current_regime_.name 
    //           << " (vol=" << std::fixed << std::setprecision(3) << current_regime_.volatility
    //           << ", trend=" << current_regime_.trend << ")" << std::endl;
}

bool VirtualMarketEngine::should_change_regime() {
    if (current_regime_.name.empty()) return true;
    
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - regime_start_time_);
    return elapsed.count() >= current_regime_.duration_minutes;
}

std::vector<Bar> VirtualMarketEngine::generate_market_data(const std::string& symbol, 
                                                         int periods, 
                                                         int interval_seconds) {
    std::vector<Bar> bars;
    bars.reserve(periods);
    
    auto current_time = std::chrono::system_clock::now();
    double current_price = base_prices_[symbol];
    
    for (int i = 0; i < periods; ++i) {
        // Change regime periodically
        if (i % 100 == 0) {
            select_new_regime();
        }
        
        Bar bar = generate_market_bar(symbol, current_time);
        bars.push_back(bar);
        
        current_price = bar.close;
        current_time += std::chrono::seconds(interval_seconds);
    }
    
    std::cout << "✅ Generated " << bars.size() << " bars for " << symbol << std::endl;
    return bars;
}

Bar VirtualMarketEngine::generate_market_bar(const std::string& symbol, 
                                           std::chrono::system_clock::time_point timestamp) {
    double current_price = base_prices_[symbol];
    
    // Calculate price movement
    double price_move = calculate_price_movement(symbol, current_price, current_regime_);
    double new_price = current_price * (1 + price_move);
    
    // Generate OHLC from price movement
    double open_price = current_price;
    double close_price = new_price;
    
    // High and low with realistic intrabar movement
    double intrabar_range = std::abs(price_move) * (0.3 + (rng_() % 50) / 100.0);
    double high_price = std::max(open_price, close_price) + current_price * intrabar_range * (rng_() % 100) / 100.0;
    double low_price = std::min(open_price, close_price) - current_price * intrabar_range * (rng_() % 100) / 100.0;
    
    // Realistic volume model based on real QQQ data
    int base_volume = 80000 + (rng_() % 120000); // 80K-200K base volume
    double volume_multiplier = current_regime_.volume_multiplier * (1 + std::abs(price_move) * 2.0);
    
    // Add time-of-day volume patterns (higher volume during market open/close)
    double time_multiplier = 1.0;
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);
    std::tm* tm_info = std::localtime(&time_t);
    int hour = tm_info->tm_hour;
    
    // Market hours volume patterns (9:30-16:00 ET)
    if (hour >= 9 && hour <= 16) {
        if (hour == 9 || hour == 16) {
            time_multiplier = 1.5; // Higher volume at open/close
        } else if (hour == 10 || hour == 15) {
            time_multiplier = 1.2; // Moderate volume
        } else {
            time_multiplier = 1.0; // Normal volume
        }
    } else {
        time_multiplier = 0.3; // Lower volume outside market hours
    }
    
    int volume = static_cast<int>(base_volume * volume_multiplier * time_multiplier);
    
    Bar bar;
    bar.ts_utc_epoch = std::chrono::duration_cast<std::chrono::seconds>(timestamp.time_since_epoch()).count();
    bar.open = open_price;
    bar.high = high_price;
    bar.low = low_price;
    bar.close = close_price;
    bar.volume = volume;
    
    return bar;
}

double VirtualMarketEngine::calculate_price_movement(const std::string& symbol, 
                                                    double current_price,
                                                    const MarketRegime& regime) {
    // Calculate price movement components
    double dt = 1.0 / (252 * 390); // 1 minute as fraction of trading year
    
    // Trend component
    double trend_move = regime.trend * dt;
    
    // Random walk component
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    double volatility_move = regime.volatility * std::sqrt(dt) * normal_dist(rng_);
    
    // Mean reversion component
    double base_price = base_prices_[symbol];
    double reversion_move = regime.mean_reversion * (base_price - current_price) * dt * 0.01;
    
    // Microstructure noise
    double microstructure_noise = normal_dist(rng_) * 0.0005;
    
    return trend_move + volatility_move + reversion_move + microstructure_noise;
}

std::vector<VirtualMarketEngine::VMSimulationResult> VirtualMarketEngine::run_monte_carlo_simulation(
    const VMTestConfig& config,
    std::unique_ptr<BaseStrategy> strategy,
    const RunnerCfg& runner_cfg) {
    
    std::cout << "🔄 FIXED DATA: Using Future QQQ tracks instead of random Monte Carlo..." << std::endl;
    
    // **FIXED DATA REPLACEMENT**: Use Future QQQ regime test instead of random generation
    return run_future_qqq_regime_test(
        config.strategy_name, 
        config.symbol, 
        config.simulations, 
        "normal",  // Default to normal regime for Monte Carlo replacement
        config.params_json
    );
}

std::vector<VirtualMarketEngine::VMSimulationResult> VirtualMarketEngine::run_mars_monte_carlo_simulation(
    const VMTestConfig& config,
    std::unique_ptr<BaseStrategy> strategy,
    const RunnerCfg& runner_cfg,
    const std::string& market_regime) {
    
    std::cout << "🔄 FIXED DATA: Using Future QQQ tracks instead of random MarS generation..." << std::endl;
    
    // **FIXED DATA REPLACEMENT**: Use Future QQQ regime test instead of random MarS generation
    return run_future_qqq_regime_test(
        config.strategy_name, 
        config.symbol, 
        config.simulations, 
        market_regime,  // Use the requested regime
        config.params_json
    );
}

std::vector<VirtualMarketEngine::VMSimulationResult> VirtualMarketEngine::run_mars_vm_test(
    const std::string& strategy_name,
    const std::string& symbol,
    int days,
    int simulations,
    const std::string& market_regime,
    const std::string& params_json) {
    
    // Create VM test configuration
    VMTestConfig config;
    config.strategy_name = strategy_name;
    config.symbol = symbol;
    config.days = days;
    config.simulations = simulations;
    config.params_json = params_json;
    config.initial_capital = 100000.0;
    
    // Create strategy instance
    auto strategy = StrategyFactory::instance().create_strategy(strategy_name);
    if (!strategy) {
        std::cerr << "Error: Could not create strategy " << strategy_name << std::endl;
        return {};
    }
    
    // Create runner configuration
    RunnerCfg runner_cfg;
    runner_cfg.strategy_name = strategy_name;
    runner_cfg.strategy_params["buy_hi"] = "0.6";
    runner_cfg.strategy_params["sell_lo"] = "0.4";
    
    // Run MarS Monte Carlo simulation
    return run_mars_monte_carlo_simulation(config, std::move(strategy), runner_cfg, market_regime);
}

std::vector<VirtualMarketEngine::VMSimulationResult> VirtualMarketEngine::run_future_qqq_regime_test(
    const std::string& strategy_name,
    const std::string& symbol,
    int simulations,
    const std::string& market_regime,
    const std::string& params_json) {
    
    std::vector<VMSimulationResult> results;
    results.reserve(simulations);
    
    std::cout << "🚀 Starting Future QQQ Regime Test..." << std::endl;
    std::cout << "📊 Strategy: " << strategy_name << std::endl;
    std::cout << "📈 Symbol: " << symbol << std::endl;
    std::cout << "🎯 Market Regime: " << market_regime << std::endl;
    std::cout << "🎲 Simulations: " << simulations << std::endl;
    
    // Validate future QQQ tracks are available
    if (!FutureQQQLoader::validate_tracks()) {
        std::cerr << "❌ Future QQQ tracks validation failed" << std::endl;
        return results;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < simulations; ++i) {
        // Progress reporting
        if (simulations >= 5 && ((i + 1) % std::max(1, simulations / 2) == 0 || i == simulations - 1)) {
            double progress_pct = (100.0 * (i + 1)) / simulations;
            std::cout << "📊 Progress: " << std::fixed << std::setprecision(0) << progress_pct 
                      << "% (" << (i + 1) << "/" << simulations << ")" << std::endl;
        }
        
        try {
            // Load a random track for the specified regime
            // Use simulation index as seed for reproducible results
            auto future_data = FutureQQQLoader::load_regime_track(market_regime, i);
            
            if (future_data.empty()) {
                std::cerr << "Warning: No future QQQ data loaded for simulation " << (i + 1) << std::endl;
                results.push_back(VMSimulationResult{}); // Add empty result
                continue;
            }
            
            // Create new strategy instance for this simulation
            auto strategy_instance = StrategyFactory::instance().create_strategy(strategy_name);
            
            if (!strategy_instance) {
                std::cerr << "Error: Could not create strategy instance for simulation " << (i + 1) << std::endl;
                results.push_back(VMSimulationResult{}); // Add empty result
                continue;
            }
            
            // Create runner configuration
            RunnerCfg runner_cfg;
            runner_cfg.strategy_name = strategy_name;
            runner_cfg.strategy_params["buy_hi"] = "0.6";
            runner_cfg.strategy_params["sell_lo"] = "0.4";
            runner_cfg.audit_level = AuditLevel::Full; // Ensure full audit logging
            
            // Run simulation with future QQQ data
            auto result = run_single_simulation(
                future_data, std::move(strategy_instance), runner_cfg, 100000.0
            );
            
            results.push_back(result);
            
        } catch (const std::exception& e) {
            std::cerr << "Error in future QQQ simulation " << (i + 1) << ": " << e.what() << std::endl;
            results.push_back(VMSimulationResult{}); // Add empty result
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    std::cout << "⏱️  Completed " << simulations << " future QQQ simulations in " 
              << total_time << " seconds" << std::endl;
    
    // Print comprehensive report
    VMTestConfig config;
    config.strategy_name = strategy_name;
    config.symbol = symbol;
    config.simulations = simulations;
    config.params_json = params_json;
    config.initial_capital = 100000.0;
    print_simulation_report(results, config);
    
    return results;
}

std::vector<VirtualMarketEngine::VMSimulationResult> VirtualMarketEngine::run_fast_historical_test(
    const std::string& strategy_name,
    const std::string& symbol,
    const std::string& historical_data_file,
    int continuation_minutes,
    int simulations,
    const std::string& params_json) {
    
    std::cout << "🔄 FIXED DATA: Using Future QQQ tracks instead of random fast historical generation..." << std::endl;
    std::cout << "📊 Strategy: " << strategy_name << std::endl;
    std::cout << "📈 Symbol: " << symbol << std::endl;
    std::cout << "🎲 Simulations: " << simulations << std::endl;
    
    // **FIXED DATA REPLACEMENT**: Use Future QQQ regime test instead of random fast historical generation
    return run_future_qqq_regime_test(
        strategy_name, 
        symbol, 
        simulations, 
        "normal",  // Default to normal regime for historical replacement
        params_json
    );
}

VirtualMarketEngine::VMSimulationResult VirtualMarketEngine::run_single_simulation(
    const std::vector<Bar>& market_data,
    std::unique_ptr<BaseStrategy> strategy,
    const RunnerCfg& runner_cfg,
    double initial_capital) {
    
    // REAL INTEGRATION: Use actual Runner component
    VMSimulationResult result;
    
    try {
        // 1. Create SymbolTable for QQQ family (profit maximization requires all leverage instruments)
        SymbolTable ST;
        int qqq_id = ST.intern("QQQ");
        int tqqq_id = ST.intern("TQQQ");  // 3x leveraged long
        int sqqq_id = ST.intern("SQQQ");  // 3x leveraged short
        int psq_id = ST.intern("PSQ");    // 1x inverse
        
        // 2. Create series data structure (QQQ family for maximum leverage)
        std::vector<std::vector<Bar>> series(4);
        series[qqq_id] = market_data;  // Base QQQ data
        
        // Generate theoretical leverage data for profit maximization
        std::cout << "🚀 Generating theoretical leverage data for maximum profit..." << std::endl;
        
        // Generate theoretical leverage series directly from QQQ data
        series[tqqq_id] = generate_theoretical_leverage_series(market_data, 3.0);   // 3x leveraged long
        series[sqqq_id] = generate_theoretical_leverage_series(market_data, -3.0);  // 3x leveraged short  
        series[psq_id] = generate_theoretical_leverage_series(market_data, -1.0);   // 1x inverse
        
        std::cout << "✅ TQQQ theoretical data generated (" << series[tqqq_id].size() << " bars, 3x leverage)" << std::endl;
        std::cout << "✅ SQQQ theoretical data generated (" << series[sqqq_id].size() << " bars, -3x leverage)" << std::endl;
        std::cout << "✅ PSQ theoretical data generated (" << series[psq_id].size() << " bars, -1x leverage)" << std::endl;
        
        // 3. Create audit recorder - use in-memory database if audit logging is disabled
        std::string run_id = generate_run_id();
        std::string note = "Strategy: " + runner_cfg.strategy_name + ", Test: vm_test, Generated by strattest";
        
        // Use in-memory database if audit logging is disabled to prevent conflicts
        std::string db_path = (runner_cfg.audit_level == AuditLevel::MetricsOnly) ? ":memory:" : "audit/sentio_audit.sqlite3";
        audit::AuditDBRecorder audit(db_path, run_id, note);
        
        // 4. Run REAL backtest using actual Runner (now returns raw BacktestOutput)
        // **DATASET TRACEABILITY**: Pass comprehensive dataset metadata
        DatasetMetadata dataset_meta;
        dataset_meta.source_type = "future_qqq_track";
        dataset_meta.file_path = "data/future_qqq/future_qqq_track_01.csv";
        dataset_meta.track_id = "track_01";
        dataset_meta.regime = "normal"; // Default regime for fixed data
        dataset_meta.bars_count = static_cast<int>(market_data.size());
        if (!market_data.empty()) {
            dataset_meta.time_range_start = market_data.front().ts_utc_epoch * 1000;
            dataset_meta.time_range_end = market_data.back().ts_utc_epoch * 1000;
        }
        BacktestOutput backtest_output = run_backtest(audit, ST, series, qqq_id, runner_cfg, dataset_meta);
        
        // Suppress debug output for cleaner console
        
        // 5. NEW: Store raw output and calculate unified metrics
        result.raw_output = backtest_output;
        result.unified_metrics = UnifiedMetricsCalculator::calculate_metrics(backtest_output);
        result.run_id = run_id;  // Store run ID for audit verification
        
        // 6. Populate metrics for backward compatibility
        result.total_return = result.unified_metrics.total_return;
        result.final_capital = result.unified_metrics.final_equity;
        result.sharpe_ratio = result.unified_metrics.sharpe_ratio;
        result.max_drawdown = result.unified_metrics.max_drawdown;
        result.win_rate = 0.0; // Not calculated in unified metrics yet
        result.total_trades = result.unified_metrics.total_fills;
        result.monthly_projected_return = result.unified_metrics.monthly_projected_return;
        result.daily_trades = result.unified_metrics.avg_daily_trades;
        
        // Suppress individual simulation output for cleaner console
        
    } catch (const std::exception& e) {
        std::cerr << "❌ VM simulation failed: " << e.what() << std::endl;
        
        // Return zero results on error
        result.raw_output = BacktestOutput{}; // Empty raw output
        result.unified_metrics = UnifiedMetricsReport{}; // Empty unified metrics
        result.total_return = 0.0;
        result.final_capital = initial_capital;
        result.sharpe_ratio = 0.0;
        result.max_drawdown = 0.0;
        result.win_rate = 0.0;
        result.total_trades = 0;
        result.monthly_projected_return = 0.0;
        result.daily_trades = 0.0;
    }
    
    return result;
}

VirtualMarketEngine::VMSimulationResult VirtualMarketEngine::calculate_performance_metrics(
    const std::vector<double>& returns,
    const std::vector<int>& trades,
    double initial_capital) {
    
    VMSimulationResult result;
    
    if (returns.empty()) {
        result.total_return = 0.0;
        result.final_capital = initial_capital;
        result.sharpe_ratio = 0.0;
        result.max_drawdown = 0.0;
        result.win_rate = 0.0;
        result.total_trades = 0;
        result.monthly_projected_return = 0.0;
        result.daily_trades = 0.0;
        return result;
    }
    
    // Calculate basic metrics
    result.total_return = returns.back();
    result.final_capital = initial_capital * (1 + result.total_return);
    result.total_trades = trades.size();
    
    // Calculate Sharpe ratio
    if (returns.size() > 1) {
        std::vector<double> daily_returns;
        for (size_t i = 1; i < returns.size(); ++i) {
            daily_returns.push_back(returns[i] - returns[i-1]);
        }
        
        double mean_return = std::accumulate(daily_returns.begin(), daily_returns.end(), 0.0) / daily_returns.size();
        double variance = 0.0;
        for (double ret : daily_returns) {
            variance += (ret - mean_return) * (ret - mean_return);
        }
        double std_dev = std::sqrt(variance / daily_returns.size());
        
        result.sharpe_ratio = (std_dev > 0) ? (mean_return / std_dev * std::sqrt(252)) : 0.0;
    }
    
    // Calculate max drawdown
    double peak = initial_capital;
    double max_dd = 0.0;
    for (double ret : returns) {
        double current_value = initial_capital * (1 + ret);
        if (current_value > peak) {
            peak = current_value;
        }
        double dd = (peak - current_value) / peak;
        max_dd = std::max(max_dd, dd);
    }
    result.max_drawdown = max_dd;
    
    // Calculate win rate
    int winning_trades = std::count_if(trades.begin(), trades.end(), [](int trade) { return trade > 0; });
    result.win_rate = (trades.empty()) ? 0.0 : static_cast<double>(winning_trades) / trades.size();
    
    // Monthly projected return is already correctly calculated by run_backtest
    // using UnifiedMetricsCalculator - no need to recalculate here
    
    // Calculate daily trades
    result.daily_trades = static_cast<double>(result.total_trades) / returns.size();
    
    return result;
}

void VirtualMarketEngine::print_simulation_report(const std::vector<VMSimulationResult>& results,
                                                 const VMTestConfig& config) {
    if (results.empty()) {
        std::cout << "❌ No simulation results to report" << std::endl;
        return;
    }
    
    // Calculate statistics
    std::vector<double> returns;
    std::vector<double> sharpe_ratios;
    std::vector<double> monthly_returns;
    std::vector<double> daily_trades;
    
    for (const auto& result : results) {
        returns.push_back(result.total_return);
        sharpe_ratios.push_back(result.sharpe_ratio);
        monthly_returns.push_back(result.monthly_projected_return);
        daily_trades.push_back(result.daily_trades);
    }
    
    // Sort for percentiles
    std::sort(returns.begin(), returns.end());
    std::sort(sharpe_ratios.begin(), sharpe_ratios.end());
    std::sort(monthly_returns.begin(), monthly_returns.end());
    std::sort(daily_trades.begin(), daily_trades.end());
    
    // Calculate statistics
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double median_return = returns[returns.size() / 2];
    double std_return = 0.0;
    for (double ret : returns) {
        std_return += (ret - mean_return) * (ret - mean_return);
    }
    std_return = std::sqrt(std_return / returns.size());
    
    double mean_sharpe = std::accumulate(sharpe_ratios.begin(), sharpe_ratios.end(), 0.0) / sharpe_ratios.size();
    double mean_mpr = std::accumulate(monthly_returns.begin(), monthly_returns.end(), 0.0) / monthly_returns.size();
    double mean_daily_trades = std::accumulate(daily_trades.begin(), daily_trades.end(), 0.0) / daily_trades.size();
    
    // Probability analysis
    int profitable_sims = std::count_if(returns.begin(), returns.end(), [](double ret) { return ret > 0; });
    double prob_profit = static_cast<double>(profitable_sims) / returns.size() * 100;
    
    // Report generation moved to UnifiedStrategyTester for consistency
}

// VMTestRunner implementation
int VMTestRunner::run_vmtest(const VirtualMarketEngine::VMTestConfig& config) {
    std::cout << "🚀 Starting Virtual Market Test..." << std::endl;
    std::cout << "📊 Strategy: " << config.strategy_name << std::endl;
    std::cout << "📈 Symbol: " << config.symbol << std::endl;
    std::cout << "⏱️  Duration: " << (config.hours > 0 ? std::to_string(config.hours) + " hours" : std::to_string(config.days) + " days") << std::endl;
    std::cout << "🎲 Simulations: " << config.simulations << std::endl;
    std::cout << "⚡ Fast Mode: " << (config.fast_mode ? "enabled" : "disabled") << std::endl;
    
    // Create strategy
    auto strategy = create_strategy(config.strategy_name, config.params_json);
    if (!strategy) {
        std::cerr << "❌ Failed to create strategy: " << config.strategy_name << std::endl;
        return 1;
    }
    
    // Load strategy configuration
    RunnerCfg runner_cfg = load_strategy_config(config.strategy_name);
    
    // Run Monte Carlo simulation
    auto results = vm_engine_.run_monte_carlo_simulation(config, std::move(strategy), runner_cfg);
    
    // Print comprehensive report
    vm_engine_.print_simulation_report(results, config);
    
    return 0;
}

std::unique_ptr<BaseStrategy> VMTestRunner::create_strategy(const std::string& strategy_name,
                                                           const std::string& params_json) {
    // Use StrategyFactory to create strategy (same as Runner)
    return StrategyFactory::instance().create_strategy(strategy_name);
}

RunnerCfg VMTestRunner::load_strategy_config(const std::string& strategy_name) {
    RunnerCfg cfg;
    cfg.strategy_name = strategy_name;
    cfg.audit_level = AuditLevel::Full;
    cfg.snapshot_stride = 100;
    
    // Load configuration from JSON files
    // TODO: Implement JSON configuration loading
    
    return cfg;
}

} // namespace sentio

```

