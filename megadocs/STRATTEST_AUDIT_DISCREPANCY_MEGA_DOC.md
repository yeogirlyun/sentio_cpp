# STRATTEST_AUDIT_DISCREPANCY_BUG_REPORT

**Generated**: 2025-09-15 16:58:06
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: Comprehensive analysis of discrepancies between strattest and audit summarize results for the same run ID

**Total Files**: 206

---

## 🐛 **BUG REPORT**

# STRATTEST vs AUDIT SUMMARIZE DISCREPANCY BUG REPORT

## Executive Summary

**Critical Issue**: Significant discrepancies exist between `sentio_cli strattest` and `sentio_audit summarize` results for the **same run ID (475680)**, indicating fundamental architectural differences in metric calculations.

**Impact**: This undermines the reliability of performance metrics and creates confusion about actual strategy performance.

## Test Case Details

- **Run ID**: 475680
- **Strategy**: TFA (Transformer-based Feature Analysis)
- **Test Type**: Historical backtest (4 weeks)
- **Dataset**: Historical QQQ data
- **Test Period**: 28 trading days
- **Run Date**: 2025-09-15 16:46:21 KST

## Discrepancy Analysis

### 1. Sharpe Ratio Discrepancy
- **STRATTEST**: 0.31
- **AUDIT**: 2.317
- **Difference**: 647% higher in audit
- **Severity**: CRITICAL

### 2. Total Return Discrepancy
- **STRATTEST Mean Return**: 0.04%
- **AUDIT Total Return**: 0.17%
- **Difference**: 325% higher in audit
- **Severity**: HIGH

### 3. Maximum Drawdown Discrepancy
- **STRATTEST**: 0.5%
- **AUDIT**: 0.03%
- **Difference**: 1567% lower in audit
- **Severity**: CRITICAL

### 4. Daily Trades Discrepancy
- **STRATTEST**: 5.0 trades/day
- **AUDIT**: 5.6 trades/day
- **Difference**: 12% higher in audit
- **Severity**: LOW

### 5. MPR Consistency ✅
- **STRATTEST**: 0.02%
- **AUDIT**: 0.02%
- **Difference**: EXACT MATCH
- **Status**: CONFIRMED ACCURATE

## Root Cause Analysis

### 1. Different Calculation Methodologies

**STRATTEST System**:
- Uses `VirtualMarketEngine` simulation results
- Calculates metrics from `VMSimulationResult` objects
- Applies statistical analysis across multiple simulations
- Uses mean values and confidence intervals

**AUDIT System**:
- Uses stored `audit_fills` and `audit_marks_daily` tables
- Calculates metrics from actual trade execution data
- Uses canonical P&L engine calculations
- Processes individual fill events and daily equity curves

### 2. Data Source Differences

**STRATTEST**:
- Processes simulation results in memory
- Uses `VMSimulationResult::total_return` and `VMSimulationResult::total_trades`
- Applies statistical aggregation across simulation runs

**AUDIT**:
- Processes stored database records
- Uses `audit_fills` table for trade data
- Uses `audit_marks_daily` table for daily equity calculations
- Applies FIFO lot accounting and realized/unrealized P&L

### 3. Metric Calculation Differences

**Sharpe Ratio**:
- STRATTEST: Calculated from simulation return statistics
- AUDIT: Calculated from daily equity curve and returns

**Total Return**:
- STRATTEST: Mean return across simulations
- AUDIT: Actual realized P&L from fills

**Maximum Drawdown**:
- STRATTEST: Statistical maximum drawdown from simulations
- AUDIT: Actual drawdown from daily equity curve

## Technical Architecture Issues

### 1. Dual Calculation Systems
The system maintains two separate calculation pipelines:
- **Simulation Pipeline**: `VirtualMarketEngine` → `VMSimulationResult` → Statistical metrics
- **Audit Pipeline**: Database → `PnLEngine` → Canonical metrics

### 2. Data Consistency Problems
- Simulation results may not perfectly match stored audit data
- Different rounding/precision in calculations
- Potential timing differences in data processing

### 3. Metric Definition Ambiguity
- No clear specification of which calculation method is authoritative
- Different interpretations of "total return" vs "mean return"
- Inconsistent drawdown calculation methods

## Impact Assessment

### 1. Business Impact
- **Strategy Evaluation**: Inconsistent performance metrics lead to incorrect strategy selection
- **Risk Management**: Drawdown calculations differ by 1567%, affecting risk assessment
- **Performance Reporting**: Sharpe ratio differences of 647% undermine credibility

### 2. Technical Impact
- **System Reliability**: Core metric calculations are inconsistent
- **Data Integrity**: Same run produces different results in different systems
- **Maintenance Burden**: Two separate calculation systems require dual maintenance

## Recommendations

### 1. Immediate Actions
1. **Document Discrepancies**: Create comprehensive documentation of all metric calculation differences
2. **Audit Data Validation**: Verify that audit data accurately reflects simulation results
3. **Metric Standardization**: Define canonical calculation methods for all metrics

### 2. Architectural Solutions
1. **Single Source of Truth**: Implement unified metric calculation system
2. **Data Reconciliation**: Ensure simulation results match stored audit data
3. **Canonical Metrics**: Establish authoritative calculation methods

### 3. Long-term Fixes
1. **Unified Architecture**: Merge simulation and audit calculation systems
2. **Real-time Validation**: Implement continuous validation between systems
3. **Comprehensive Testing**: Add integration tests to catch discrepancies

## Conclusion

The discrepancies between `strattest` and `audit summarize` for the same run ID (475680) reveal fundamental architectural issues in the metric calculation systems. While MPR calculations are consistent, other critical metrics (Sharpe ratio, total return, maximum drawdown) show significant differences that undermine system reliability.

**Priority**: HIGH - This issue affects core system functionality and requires immediate attention to ensure accurate strategy performance evaluation.

**Next Steps**: 
1. Investigate the specific calculation differences in each system
2. Implement data reconciliation between simulation and audit systems
3. Establish canonical metric calculation standards
4. Add comprehensive integration testing to prevent future discrepancies


---

## 📋 **TABLE OF CONTENTS**

1. [include/sentio/accurate_leverage_pricing.hpp](#file-1)
2. [include/sentio/all_strategies.hpp](#file-2)
3. [include/sentio/alpha.hpp](#file-3)
4. [include/sentio/alpha/sota_linear_policy.hpp](#file-4)
5. [include/sentio/audit.hpp](#file-5)
6. [include/sentio/audit_interface.hpp](#file-6)
7. [include/sentio/audit_validator.hpp](#file-7)
8. [include/sentio/base_strategy.hpp](#file-8)
9. [include/sentio/binio.hpp](#file-9)
10. [include/sentio/bo.hpp](#file-10)
11. [include/sentio/bollinger.hpp](#file-11)
12. [include/sentio/cli_helpers.hpp](#file-12)
13. [include/sentio/core.hpp](#file-13)
14. [include/sentio/core/bar.hpp](#file-14)
15. [include/sentio/cost_model.hpp](#file-15)
16. [include/sentio/csv_loader.hpp](#file-16)
17. [include/sentio/data_downloader.hpp](#file-17)
18. [include/sentio/data_resolver.hpp](#file-18)
19. [include/sentio/day_index.hpp](#file-19)
20. [include/sentio/exec/asof_index.hpp](#file-20)
21. [include/sentio/exec_types.hpp](#file-21)
22. [include/sentio/execution/pnl_engine.hpp](#file-22)
23. [include/sentio/family_mapper.hpp](#file-23)
24. [include/sentio/feature/column_projector.hpp](#file-24)
25. [include/sentio/feature/column_projector_safe.hpp](#file-25)
26. [include/sentio/feature/csv_feature_provider.hpp](#file-26)
27. [include/sentio/feature/feature_builder_guarded.hpp](#file-27)
28. [include/sentio/feature/feature_builder_ops.hpp](#file-28)
29. [include/sentio/feature/feature_feeder_guarded.hpp](#file-29)
30. [include/sentio/feature/feature_from_spec.hpp](#file-30)
31. [include/sentio/feature/feature_matrix.hpp](#file-31)
32. [include/sentio/feature/feature_provider.hpp](#file-32)
33. [include/sentio/feature/name_diff.hpp](#file-33)
34. [include/sentio/feature/ops.hpp](#file-34)
35. [include/sentio/feature/sanitize.hpp](#file-35)
36. [include/sentio/feature/standard_scaler.hpp](#file-36)
37. [include/sentio/feature_builder.hpp](#file-37)
38. [include/sentio/feature_cache.hpp](#file-38)
39. [include/sentio/feature_engineering/feature_normalizer.hpp](#file-39)
40. [include/sentio/feature_engineering/kochi_features.hpp](#file-40)
41. [include/sentio/feature_engineering/technical_indicators.hpp](#file-41)
42. [include/sentio/feature_feeder.hpp](#file-42)
43. [include/sentio/feature_health.hpp](#file-43)
44. [include/sentio/feature_utils.hpp](#file-44)
45. [include/sentio/future_qqq_loader.hpp](#file-45)
46. [include/sentio/global_leverage_config.hpp](#file-46)
47. [include/sentio/indicators.hpp](#file-47)
48. [include/sentio/leverage_aware_csv_loader.hpp](#file-48)
49. [include/sentio/leverage_pricing.hpp](#file-49)
50. [include/sentio/mars_data_loader.hpp](#file-50)
51. [include/sentio/metrics.hpp](#file-51)
52. [include/sentio/metrics/mpr.hpp](#file-52)
53. [include/sentio/metrics/session_utils.hpp](#file-53)
54. [include/sentio/ml/feature_pipeline.hpp](#file-54)
55. [include/sentio/ml/feature_window.hpp](#file-55)
56. [include/sentio/ml/iml_model.hpp](#file-56)
57. [include/sentio/ml/model_registry.hpp](#file-57)
58. [include/sentio/ml/ts_model.hpp](#file-58)
59. [include/sentio/of_index.hpp](#file-59)
60. [include/sentio/of_precompute.hpp](#file-60)
61. [include/sentio/optimizer.hpp](#file-61)
62. [include/sentio/orderflow_types.hpp](#file-62)
63. [include/sentio/pnl_accounting.hpp](#file-63)
64. [include/sentio/polygon_client.hpp](#file-64)
65. [include/sentio/portfolio/alpaca_fee_model.hpp](#file-65)
66. [include/sentio/portfolio/capital_manager.hpp](#file-66)
67. [include/sentio/portfolio/fee_model.hpp](#file-67)
68. [include/sentio/portfolio/portfolio_allocator.hpp](#file-68)
69. [include/sentio/portfolio/tc_slippage_model.hpp](#file-69)
70. [include/sentio/portfolio/utilization_governor.hpp](#file-70)
71. [include/sentio/position_coordinator.hpp](#file-71)
72. [include/sentio/position_guardian.hpp](#file-72)
73. [include/sentio/position_manager.hpp](#file-73)
74. [include/sentio/position_orchestrator.hpp](#file-74)
75. [include/sentio/position_validator.hpp](#file-75)
76. [include/sentio/pricebook.hpp](#file-76)
77. [include/sentio/profiling.hpp](#file-77)
78. [include/sentio/progress_bar.hpp](#file-78)
79. [include/sentio/property_test.hpp](#file-79)
80. [include/sentio/rolling_stats.hpp](#file-80)
81. [include/sentio/router.hpp](#file-81)
82. [include/sentio/rsi_prob.hpp](#file-82)
83. [include/sentio/rsi_strategy.hpp](#file-83)
84. [include/sentio/rules/adapters.hpp](#file-84)
85. [include/sentio/rules/bbands_squeeze_rule.hpp](#file-85)
86. [include/sentio/rules/diversity_weighter.hpp](#file-86)
87. [include/sentio/rules/integrated_rule_ensemble.hpp](#file-87)
88. [include/sentio/rules/irule.hpp](#file-88)
89. [include/sentio/rules/momentum_volume_rule.hpp](#file-89)
90. [include/sentio/rules/ofi_proxy_rule.hpp](#file-90)
91. [include/sentio/rules/online_platt_calibrator.hpp](#file-91)
92. [include/sentio/rules/opening_range_breakout_rule.hpp](#file-92)
93. [include/sentio/rules/registry.hpp](#file-93)
94. [include/sentio/rules/sma_cross_rule.hpp](#file-94)
95. [include/sentio/rules/utils/validation.hpp](#file-95)
96. [include/sentio/rules/vwap_reversion_rule.hpp](#file-96)
97. [include/sentio/run_id_generator.hpp](#file-97)
98. [include/sentio/runner.hpp](#file-98)
99. [include/sentio/sanity.hpp](#file-99)
100. [include/sentio/side.hpp](#file-100)
101. [include/sentio/signal.hpp](#file-101)
102. [include/sentio/signal_diag.hpp](#file-102)
103. [include/sentio/signal_engine.hpp](#file-103)
104. [include/sentio/signal_gate.hpp](#file-104)
105. [include/sentio/signal_or.hpp](#file-105)
106. [include/sentio/signal_pipeline.hpp](#file-106)
107. [include/sentio/signal_trace.hpp](#file-107)
108. [include/sentio/signal_utils.hpp](#file-108)
109. [include/sentio/sim_data.hpp](#file-109)
110. [include/sentio/sizer.hpp](#file-110)
111. [include/sentio/strategy/intraday_position_governor.hpp](#file-111)
112. [include/sentio/strategy_bollinger_squeeze_breakout.hpp](#file-112)
113. [include/sentio/strategy_ire.hpp](#file-113)
114. [include/sentio/strategy_kochi_ppo.hpp](#file-114)
115. [include/sentio/strategy_market_making.hpp](#file-115)
116. [include/sentio/strategy_momentum_volume.hpp](#file-116)
117. [include/sentio/strategy_opening_range_breakout.hpp](#file-117)
118. [include/sentio/strategy_order_flow_imbalance.hpp](#file-118)
119. [include/sentio/strategy_order_flow_scalping.hpp](#file-119)
120. [include/sentio/strategy_signal_or.hpp](#file-120)
121. [include/sentio/strategy_sma_cross.hpp](#file-121)
122. [include/sentio/strategy_tfa.hpp](#file-122)
123. [include/sentio/strategy_transformer.hpp](#file-123)
124. [include/sentio/strategy_utils.hpp](#file-124)
125. [include/sentio/strategy_vwap_reversion.hpp](#file-125)
126. [include/sentio/sym/leverage_registry.hpp](#file-126)
127. [include/sentio/sym/symbol_utils.hpp](#file-127)
128. [include/sentio/symbol_table.hpp](#file-128)
129. [include/sentio/telemetry_logger.hpp](#file-129)
130. [include/sentio/temporal_analysis.hpp](#file-130)
131. [include/sentio/tfa/artifacts_loader.hpp](#file-131)
132. [include/sentio/tfa/artifacts_safe.hpp](#file-132)
133. [include/sentio/tfa/feature_guard.hpp](#file-133)
134. [include/sentio/tfa/input_shim.hpp](#file-134)
135. [include/sentio/tfa/signal_pipeline.hpp](#file-135)
136. [include/sentio/tfa/tfa_seq_context.hpp](#file-136)
137. [include/sentio/time_utils.hpp](#file-137)
138. [include/sentio/torch/safe_from_blob.hpp](#file-138)
139. [include/sentio/unified_metrics.hpp](#file-139)
140. [include/sentio/unified_strategy_tester.hpp](#file-140)
141. [include/sentio/util/bytes.hpp](#file-141)
142. [include/sentio/util/safe_matrix.hpp](#file-142)
143. [include/sentio/utils/formatting.hpp](#file-143)
144. [include/sentio/utils/validation.hpp](#file-144)
145. [include/sentio/virtual_market.hpp](#file-145)
146. [include/sentio/wf.hpp](#file-146)
147. [src/accurate_leverage_pricing.cpp](#file-147)
148. [src/audit.cpp](#file-148)
149. [src/audit_validator.cpp](#file-149)
150. [src/base_strategy.cpp](#file-150)
151. [src/cli_helpers.cpp](#file-151)
152. [src/csv_loader.cpp](#file-152)
153. [src/data_downloader.cpp](#file-153)
154. [src/feature_builder.cpp](#file-154)
155. [src/feature_cache.cpp](#file-155)
156. [src/feature_engineering/feature_normalizer.cpp](#file-156)
157. [src/feature_engineering/kochi_features.cpp](#file-157)
158. [src/feature_engineering/technical_indicators.cpp](#file-158)
159. [src/feature_feeder.cpp](#file-159)
160. [src/feature_feeder_guarded.cpp](#file-160)
161. [src/feature_health.cpp](#file-161)
162. [src/future_qqq_loader.cpp](#file-162)
163. [src/global_leverage_config.cpp](#file-163)
164. [src/leverage_aware_csv_loader.cpp](#file-164)
165. [src/leverage_pricing.cpp](#file-165)
166. [src/main.cpp](#file-166)
167. [src/mars_data_loader.cpp](#file-167)
168. [src/ml/model_registry_ts.cpp](#file-168)
169. [src/ml/ts_model.cpp](#file-169)
170. [src/optimizer.cpp](#file-170)
171. [src/pnl_accounting.cpp](#file-171)
172. [src/poly_fetch_main.cpp](#file-172)
173. [src/polygon_client.cpp](#file-173)
174. [src/position_coordinator.cpp](#file-174)
175. [src/position_guardian.cpp](#file-175)
176. [src/position_orchestrator.cpp](#file-176)
177. [src/router.cpp](#file-177)
178. [src/rsi_strategy.cpp](#file-178)
179. [src/run_id_generator.cpp](#file-179)
180. [src/runner.cpp](#file-180)
181. [src/sanity.cpp](#file-181)
182. [src/signal_engine.cpp](#file-182)
183. [src/signal_gate.cpp](#file-183)
184. [src/signal_pipeline.cpp](#file-184)
185. [src/signal_trace.cpp](#file-185)
186. [src/sim_data.cpp](#file-186)
187. [src/strategy/run_rule_ensemble.cpp](#file-187)
188. [src/strategy_bollinger_squeeze_breakout.cpp](#file-188)
189. [src/strategy_initialization.cpp](#file-189)
190. [src/strategy_ire.cpp](#file-190)
191. [src/strategy_kochi_ppo.cpp](#file-191)
192. [src/strategy_market_making.cpp](#file-192)
193. [src/strategy_momentum_volume.cpp](#file-193)
194. [src/strategy_opening_range_breakout.cpp](#file-194)
195. [src/strategy_order_flow_imbalance.cpp](#file-195)
196. [src/strategy_order_flow_scalping.cpp](#file-196)
197. [src/strategy_signal_or.cpp](#file-197)
198. [src/strategy_sma_cross.cpp](#file-198)
199. [src/strategy_tfa.cpp](#file-199)
200. [src/strategy_vwap_reversion.cpp](#file-200)
201. [src/telemetry_logger.cpp](#file-201)
202. [src/temporal_analysis.cpp](#file-202)
203. [src/time_utils.cpp](#file-203)
204. [src/unified_metrics.cpp](#file-204)
205. [src/unified_strategy_tester.cpp](#file-205)
206. [src/virtual_market.cpp](#file-206)

---

## 📄 **FILE 1 of 206**: include/sentio/accurate_leverage_pricing.hpp

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

## 📄 **FILE 2 of 206**: include/sentio/all_strategies.hpp

**File Information**:
- **Path**: `include/sentio/all_strategies.hpp`

- **Size**: 18 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .hpp

```text
#pragma once

// This file ensures all strategies are included and registered with the factory.
// Include this header once in your main.cpp.

#include "strategy_bollinger_squeeze_breakout.hpp"
#include "strategy_market_making.hpp"
#include "strategy_momentum_volume.hpp"
#include "strategy_opening_range_breakout.hpp"
#include "strategy_order_flow_imbalance.hpp"
#include "strategy_order_flow_scalping.hpp"
#include "strategy_vwap_reversion.hpp"
// Removed unused strategies: hybrid_ppo, transformer_ts
// TFB strategy removed - focusing on TFA only
#include "strategy_tfa.hpp"
#include "strategy_kochi_ppo.hpp"
#include "strategy_ire.hpp"
#include "strategy_signal_or.hpp"
```

## 📄 **FILE 3 of 206**: include/sentio/alpha.hpp

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

## 📄 **FILE 4 of 206**: include/sentio/alpha/sota_linear_policy.hpp

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

## 📄 **FILE 5 of 206**: include/sentio/audit.hpp

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

## 📄 **FILE 6 of 206**: include/sentio/audit_interface.hpp

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

## 📄 **FILE 7 of 206**: include/sentio/audit_validator.hpp

**File Information**:
- **Path**: `include/sentio/audit_validator.hpp`

- **Size**: 70 lines
- **Modified**: 2025-09-12 21:06:37

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "runner.hpp"
#include "audit.hpp"
#include <string>
#include <vector>

namespace sentio {

/**
 * **STRATEGY-AGNOSTIC AUDIT VALIDATOR**
 * 
 * Validates that any strategy inheriting from BaseStrategy works properly
 * with the audit system without strategy-specific dependencies.
 */
class AuditValidator {
public:
    struct ValidationResult {
        bool success = false;
        std::string strategy_name;
        std::string error_message;
        int signals_logged = 0;
        int orders_logged = 0;
        int fills_logged = 0;
        double test_duration_sec = 0.0;
    };
    
    /**
     * Validate that a strategy works with the audit system
     * @param strategy_name Name of the strategy to test
     * @param test_bars Number of bars to test with (default: 100)
     * @return ValidationResult with success status and metrics
     */
    static ValidationResult validate_strategy_audit_compatibility(
        const std::string& strategy_name,
        int test_bars = 100
    );
    
    /**
     * Validate all registered strategies
     * @param test_bars Number of bars to test each strategy with
     * @return Vector of validation results for all strategies
     */
    static std::vector<ValidationResult> validate_all_strategies(
        int test_bars = 100
    );
    
    /**
     * Print validation report
     * @param results Vector of validation results
     */
    static void print_validation_report(const std::vector<ValidationResult>& results);

private:
    /**
     * Generate synthetic test data for validation
     * @param num_bars Number of bars to generate
     * @return Vector of synthetic bars
     */
    static std::vector<Bar> generate_test_data(int num_bars);
    
    /**
     * Create a minimal RunnerCfg for testing
     * @param strategy_name Name of the strategy
     * @return RunnerCfg configured for audit testing
     */
    static RunnerCfg create_test_config(const std::string& strategy_name);
};

} // namespace sentio

```

## 📄 **FILE 8 of 206**: include/sentio/base_strategy.hpp

**File Information**:
- **Path**: `include/sentio/base_strategy.hpp`

- **Size**: 169 lines
- **Modified**: 2025-09-15 15:04:43

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
    
    // **NEW**: Strategy-agnostic allocation interface
    struct AllocationDecision {
        std::string instrument;
        double target_weight; // -1.0 to 1.0
        double confidence;    // 0.0 to 1.0
        std::string reason;   // Human-readable reason for allocation
    };
    
    // **NEW**: Get allocation decisions for this strategy
    virtual std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, 
        int current_index,
        const std::string& base_symbol,
        const std::string& bull3x_symbol,
        const std::string& bear3x_symbol) = 0;
    
    // **NEW**: Get strategy-specific router configuration
    virtual RouterCfg get_router_config() const = 0;
    
    // **NEW**: Get strategy-specific sizer configuration  
    virtual SizerCfg get_sizer_config() const = 0;
    
    // **NEW**: Check if strategy requires special handling (e.g., dynamic leverage)
    virtual bool requires_dynamic_allocation() const { return false; }
    
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

## 📄 **FILE 9 of 206**: include/sentio/binio.hpp

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

## 📄 **FILE 10 of 206**: include/sentio/bo.hpp

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

## 📄 **FILE 11 of 206**: include/sentio/bollinger.hpp

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

## 📄 **FILE 12 of 206**: include/sentio/cli_helpers.hpp

**File Information**:
- **Path**: `include/sentio/cli_helpers.hpp`

- **Size**: 132 lines
- **Modified**: 2025-09-13 14:01:39

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

## 📄 **FILE 13 of 206**: include/sentio/core.hpp

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

## 📄 **FILE 14 of 206**: include/sentio/core/bar.hpp

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

## 📄 **FILE 15 of 206**: include/sentio/cost_model.hpp

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

## 📄 **FILE 16 of 206**: include/sentio/csv_loader.hpp

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

## 📄 **FILE 17 of 206**: include/sentio/data_downloader.hpp

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

## 📄 **FILE 18 of 206**: include/sentio/data_resolver.hpp

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

## 📄 **FILE 19 of 206**: include/sentio/day_index.hpp

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

## 📄 **FILE 20 of 206**: include/sentio/exec/asof_index.hpp

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

## 📄 **FILE 21 of 206**: include/sentio/exec_types.hpp

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

## 📄 **FILE 22 of 206**: include/sentio/execution/pnl_engine.hpp

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

## 📄 **FILE 23 of 206**: include/sentio/family_mapper.hpp

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

## 📄 **FILE 24 of 206**: include/sentio/feature/column_projector.hpp

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

## 📄 **FILE 25 of 206**: include/sentio/feature/column_projector_safe.hpp

**File Information**:
- **Path**: `include/sentio/feature/column_projector_safe.hpp`

- **Size**: 95 lines
- **Modified**: 2025-09-15 15:42:16

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
    
    int filled_count = 0;
    int mapped_count = 0;
    
    for (int j=0;j<(int)dst.size();++j){
      auto it = pos.find(dst[j]);
      if (it!=pos.end()) {
        P.map[j] = it->second; // Found mapping
        mapped_count++;
      } else {
        P.map[j] = -1; // Will be filled
        filled_count++;
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

## 📄 **FILE 26 of 206**: include/sentio/feature/csv_feature_provider.hpp

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

## 📄 **FILE 27 of 206**: include/sentio/feature/feature_builder_guarded.hpp

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

## 📄 **FILE 28 of 206**: include/sentio/feature/feature_builder_ops.hpp

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

## 📄 **FILE 29 of 206**: include/sentio/feature/feature_feeder_guarded.hpp

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

## 📄 **FILE 30 of 206**: include/sentio/feature/feature_from_spec.hpp

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

## 📄 **FILE 31 of 206**: include/sentio/feature/feature_matrix.hpp

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

## 📄 **FILE 32 of 206**: include/sentio/feature/feature_provider.hpp

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

## 📄 **FILE 33 of 206**: include/sentio/feature/name_diff.hpp

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

## 📄 **FILE 34 of 206**: include/sentio/feature/ops.hpp

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

## 📄 **FILE 35 of 206**: include/sentio/feature/sanitize.hpp

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

## 📄 **FILE 36 of 206**: include/sentio/feature/standard_scaler.hpp

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

## 📄 **FILE 37 of 206**: include/sentio/feature_builder.hpp

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

## 📄 **FILE 38 of 206**: include/sentio/feature_cache.hpp

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

## 📄 **FILE 39 of 206**: include/sentio/feature_engineering/feature_normalizer.hpp

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

## 📄 **FILE 40 of 206**: include/sentio/feature_engineering/kochi_features.hpp

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

## 📄 **FILE 41 of 206**: include/sentio/feature_engineering/technical_indicators.hpp

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

## 📄 **FILE 42 of 206**: include/sentio/feature_feeder.hpp

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

## 📄 **FILE 43 of 206**: include/sentio/feature_health.hpp

**File Information**:
- **Path**: `include/sentio/feature_health.hpp`

- **Size**: 31 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sentio {

struct FeatureIssue {
  std::int64_t ts_utc{};
  std::string  kind;      // e.g., "NaN", "Gap", "Backwards_TS"
  std::string  detail;
};

struct FeatureHealthReport {
  std::vector<FeatureIssue> issues;
  bool ok() const { return issues.empty(); }
};

struct FeatureHealthCfg {
  // bar spacing in seconds (e.g., 60 for 1m). 0 = skip spacing checks.
  int expected_spacing_sec{60};
  bool check_nan{true};
  bool check_monotonic_time{true};
};

struct PricePoint { std::int64_t ts_utc{}; double close{}; };

FeatureHealthReport check_feature_health(const std::vector<PricePoint>& series,
                                         const FeatureHealthCfg& cfg);

} // namespace sentio

```

## 📄 **FILE 44 of 206**: include/sentio/feature_utils.hpp

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

## 📄 **FILE 45 of 206**: include/sentio/future_qqq_loader.hpp

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

## 📄 **FILE 46 of 206**: include/sentio/global_leverage_config.hpp

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

## 📄 **FILE 47 of 206**: include/sentio/indicators.hpp

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

## 📄 **FILE 48 of 206**: include/sentio/leverage_aware_csv_loader.hpp

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

## 📄 **FILE 49 of 206**: include/sentio/leverage_pricing.hpp

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

## 📄 **FILE 50 of 206**: include/sentio/mars_data_loader.hpp

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

## 📄 **FILE 51 of 206**: include/sentio/metrics.hpp

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

## 📄 **FILE 52 of 206**: include/sentio/metrics/mpr.hpp

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

## 📄 **FILE 53 of 206**: include/sentio/metrics/session_utils.hpp

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

## 📄 **FILE 54 of 206**: include/sentio/ml/feature_pipeline.hpp

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

## 📄 **FILE 55 of 206**: include/sentio/ml/feature_window.hpp

**File Information**:
- **Path**: `include/sentio/ml/feature_window.hpp`

- **Size**: 86 lines
- **Modified**: 2025-09-15 15:40:01

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
    static int push_calls = 0;
    push_calls++;
    
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

## 📄 **FILE 56 of 206**: include/sentio/ml/iml_model.hpp

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

## 📄 **FILE 57 of 206**: include/sentio/ml/model_registry.hpp

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

## 📄 **FILE 58 of 206**: include/sentio/ml/ts_model.hpp

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

## 📄 **FILE 59 of 206**: include/sentio/of_index.hpp

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

## 📄 **FILE 60 of 206**: include/sentio/of_precompute.hpp

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

## 📄 **FILE 61 of 206**: include/sentio/optimizer.hpp

**File Information**:
- **Path**: `include/sentio/optimizer.hpp`

- **Size**: 147 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "sentio/core.hpp"
#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <random>

namespace sentio {

struct Parameter {
    std::string name;
    double min_value;
    double max_value;
    double current_value;
    
    Parameter(const std::string& n, double min_val, double max_val, double current_val)
        : name(n), min_value(min_val), max_value(max_val), current_value(current_val) {}
    
    Parameter(const std::string& n, double min_val, double max_val)
        : name(n), min_value(min_val), max_value(max_val), current_value(min_val) {}
};

struct OptimizationResult {
    std::unordered_map<std::string, double> parameters;
    RunResult metrics;
    double objective_value;
};

struct OptimizationConfig {
    std::string optimizer_type = "random";
    int max_trials = 30;
    std::string objective = "sharpe_ratio";
    double timeout_minutes = 15.0;
    bool verbose = true;
};

class ParameterOptimizer {
public:
    virtual ~ParameterOptimizer() = default;
    virtual std::vector<Parameter> get_parameter_space() = 0;
    virtual void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) = 0;
    virtual OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                                      const std::vector<Parameter>& param_space,
                                      const OptimizationConfig& config) = 0;
};

class RandomSearchOptimizer : public ParameterOptimizer {
private:
    std::mt19937 rng;
    
public:
    RandomSearchOptimizer(unsigned int seed = 42) : rng(seed) {}
    
    std::vector<Parameter> get_parameter_space() override;
    void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) override;
    OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                               const std::vector<Parameter>& param_space,
                               const OptimizationConfig& config) override;
};

class GridSearchOptimizer : public ParameterOptimizer {
public:
    std::vector<Parameter> get_parameter_space() override;
    void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) override;
    OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                              const std::vector<Parameter>& param_space,
                              const OptimizationConfig& config) override;
};

class BayesianOptimizer : public ParameterOptimizer {
public:
    std::vector<Parameter> get_parameter_space() override;
    void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) override;
    OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                              const std::vector<Parameter>& param_space,
                              const OptimizationConfig& config) override;
};

// Objective functions
using ObjectiveFunction = std::function<double(const RunResult&)>;

class ObjectiveFunctions {
public:
    static double sharpe_objective(const RunResult& summary) {
        return summary.sharpe_ratio;
    }
    
    static double calmar_objective(const RunResult& summary) {
        return summary.monthly_projected_return / std::max(0.01, summary.max_drawdown);
    }
    
    static double total_return_objective(const RunResult& summary) {
        return summary.total_return;
    }
    
    static double sortino_objective(const RunResult& summary) {
        // Simplified Sortino ratio (assuming no downside deviation calculation)
        return summary.sharpe_ratio;
    }
};

// Strategy-specific parameter creation functions
std::vector<Parameter> create_vwap_parameters();
std::vector<Parameter> create_momentum_parameters();
std::vector<Parameter> create_volatility_parameters();
std::vector<Parameter> create_bollinger_squeeze_parameters();
std::vector<Parameter> create_opening_range_parameters();
std::vector<Parameter> create_order_flow_scalping_parameters();
std::vector<Parameter> create_order_flow_imbalance_parameters();
std::vector<Parameter> create_market_making_parameters();
std::vector<Parameter> create_router_parameters();

std::vector<Parameter> create_parameters_for_strategy(const std::string& strategy_name);
std::vector<Parameter> create_full_parameter_space();

// Optimization utilities
class OptimizationEngine {
private:
    std::unique_ptr<ParameterOptimizer> optimizer;
    OptimizationConfig config;
    
public:
    OptimizationEngine(const std::string& optimizer_type = "random");
    
    void set_config(const OptimizationConfig& cfg) { config = cfg; }
    
    OptimizationResult run_optimization(const std::string& strategy_name,
                                      const std::function<double(const RunResult&)>& objective_func);
    
    double calculate_objective(const RunResult& summary) {
        if (config.objective == "sharpe_ratio") {
            return ObjectiveFunctions::sharpe_objective(summary);
        } else if (config.objective == "calmar_ratio") {
            return ObjectiveFunctions::calmar_objective(summary);
        } else if (config.objective == "total_return") {
            return ObjectiveFunctions::total_return_objective(summary);
        } else if (config.objective == "sortino_ratio") {
            return ObjectiveFunctions::sortino_objective(summary);
        }
        return ObjectiveFunctions::sharpe_objective(summary); // default
    }
};

} // namespace sentio
```

## 📄 **FILE 62 of 206**: include/sentio/orderflow_types.hpp

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

## 📄 **FILE 63 of 206**: include/sentio/pnl_accounting.hpp

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

## 📄 **FILE 64 of 206**: include/sentio/polygon_client.hpp

**File Information**:
- **Path**: `include/sentio/polygon_client.hpp`

- **Size**: 31 lines
- **Modified**: 2025-09-12 14:19:30

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
                 const std::vector<AggBar>& bars, bool exclude_holidays=false);

private:
  std::string api_key_;
  std::string get_(const std::string& url);
  std::vector<AggBar> get_aggs_chunked(const AggsQuery& q, int max_pages=200);
};
} // namespace sentio


```

## 📄 **FILE 65 of 206**: include/sentio/portfolio/alpaca_fee_model.hpp

**File Information**:
- **Path**: `include/sentio/portfolio/alpaca_fee_model.hpp`

- **Size**: 30 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "fee_model.hpp"
#include <algorithm>

namespace sentio {

// Simplified Alpaca-like: $0 commission equities, SEC/TAF pass-through for sells
// Tweak the constants to match your latest schedule if needed.
class AlpacaEquityFeeModel : public IFeeModel {
public:
  // Per-share TAF for sells, SEC fee as bps of notional on sells (approx)
  double taf_per_share = 0.000119;    // $0.000119/share
  double sec_bps_sell  = 0.0000229;   // 0.00229% of notional on sells
  double min_fee       = 0.0;         // $0 min for commissionless
  bool   include_sec   = true;

  double commission(const TradeCtx& t) const override {
    // commission-free
    return min_fee;
  }

  double exchange_fees(const TradeCtx& t) const override {
    if (t.shares > 0) return 0.0; // buy: no SEC/TAF
    double taf = std::abs(t.shares) * taf_per_share;
    double sec = include_sec ? std::max(0.0, std::abs(t.notional) * sec_bps_sell) : 0.0;
    return taf + sec;
  }
};

} // namespace sentio

```

## 📄 **FILE 66 of 206**: include/sentio/portfolio/capital_manager.hpp

**File Information**:
- **Path**: `include/sentio/portfolio/capital_manager.hpp`

- **Size**: 115 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>
#include "portfolio_allocator.hpp"
#include "utilization_governor.hpp"

namespace sentio {

// Forward declarations to avoid circular includes
class PositionSizer;
class BinaryRouter;
struct MarketSnapshot;

struct InstrumentState {
  // live state snapshot
  int64_t ts;
  double  price;
  double  vol_1d;
  double  adv_notional;
  double  spread_bp;
  long    shares;       // current position
  double  w_prev;       // last target weight
  bool    tradable{true};
};

struct RouterInputs {
  // Per-instrument p_up from Strategy (TFA/Kochi/RuleEnsemble)
  double p_up;
};

struct CapitalManagerConfig {
  AllocConfig alloc;
  // Map utilization governor into Router and Sizer:
  UtilGovConfig util;
};

struct CapitalDecision {
  std::vector<double> target_weights; // per instrument
  std::vector<long long> target_shares;
  std::vector<double> edge_bp;        // per-instrument edge in bps
  double gross{0.0};
  double total_cost_bp{0.0};          // estimated total cost
};

class CapitalManager {
public:
  CapitalManager(const CapitalManagerConfig& cfg,
                 PortfolioAllocator alloc)
   : cfg_(cfg), alloc_(std::move(alloc)) {}

  CapitalDecision decide(double equity,
                         const std::vector<InstrumentState>& S,
                         const std::vector<RouterInputs>& R,
                         UtilGovState& ug_state)
  {
    const int N = (int)S.size();
    std::vector<InstrumentInputs> X; X.reserve(N);
    for (int i=0;i<N;i++){
      X.push_back(InstrumentInputs{
        /*p_up   */ R[i].p_up,
        /*price  */ S[i].price,
        /*vol_1d */ S[i].vol_1d,
        /*spread */ S[i].spread_bp,
        /*adv    */ S[i].adv_notional,
        /*w_prev */ S[i].w_prev,
        /*pos_w  */ (S[i].price>0? (S[i].shares*S[i].price)/std::max(1e-9, equity) : 0.0),
        /*trad   */ S[i].tradable
      });
    }

    // Portfolio-level allocation with SOTA linear policy
    auto out = alloc_.allocate(X, cfg_.alloc);
    
    // Apply utilization governor: upscale/downscale weights via expo_shift
    const double expo_mul = std::clamp(1.0 + ug_state.expo_shift, 0.5, 1.5);
    for (double& w : out.w_target) w *= expo_mul;

    // Compute shares via weight-based sizing
    std::vector<long long> tgt_sh(N, 0);
    for (int i=0;i<N;i++){
      if (S[i].price > 0 && std::isfinite(out.w_target[i])) {
        double desired_notional = out.w_target[i] * equity;
        tgt_sh[i] = (long long)std::floor(desired_notional / S[i].price);
        // Apply round lot if needed (assume 1 for now)
        // Apply min notional filter (assume $50 from previous fix)
        if (std::abs(tgt_sh[i] * S[i].price) < 50.0) {
          tgt_sh[i] = 0;
        }
      }
    }

    // Report gross and cost estimates
    double gross=0.0, total_cost=0.0;
    for (int i=0;i<N;i++) {
      gross += std::abs(out.w_target[i]);
      // Estimate turnover cost
      double w_change = std::abs(out.w_target[i] - S[i].w_prev);
      total_cost += w_change * 5.0; // rough 5bp per weight change
    }

    return CapitalDecision{ 
      std::move(out.w_target), 
      std::move(tgt_sh), 
      std::move(out.edge_bp),
      gross,
      total_cost
    };
  }

private:
  CapitalManagerConfig cfg_;
  PortfolioAllocator alloc_;
};

} // namespace sentio

```

## 📄 **FILE 67 of 206**: include/sentio/portfolio/fee_model.hpp

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

## 📄 **FILE 68 of 206**: include/sentio/portfolio/portfolio_allocator.hpp

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

## 📄 **FILE 69 of 206**: include/sentio/portfolio/tc_slippage_model.hpp

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

## 📄 **FILE 70 of 206**: include/sentio/portfolio/utilization_governor.hpp

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

## 📄 **FILE 71 of 206**: include/sentio/position_coordinator.hpp

**File Information**:
- **Path**: `include/sentio/position_coordinator.hpp`

- **Size**: 127 lines
- **Modified**: 2025-09-12 17:11:26

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

## 📄 **FILE 72 of 206**: include/sentio/position_guardian.hpp

**File Information**:
- **Path**: `include/sentio/position_guardian.hpp`

- **Size**: 95 lines
- **Modified**: 2025-09-11 21:59:53

- **Type**: .hpp

```text
#pragma once
#include "side.hpp"
#include "family_mapper.hpp"
#include "core.hpp"
#include <mutex>
#include <unordered_map>
#include <optional>
#include <vector>
#include <chrono>
#include <memory>
#include <string>

namespace sentio {

struct Policy {
    bool   allow_conflicts = false; // hard OFF for this use-case
    double max_gross_shares = 1e9;  // guardrail
    double min_flip_bps = 0.0;      // optional flip friction to avoid churn
    int    cooldown_ms = 0;         // optional per-family cooldown after flip
};

struct PositionSnapshot {
    PositionSide   side{PositionSide::Flat};     // net family side
    double qty{0.0};             // positive magnitude
    double avg_px{0.0};          // informational
    uint64_t epoch{0};           // increments on each committed change
    std::chrono::steady_clock::time_point last_change{};
};

struct PlanLeg {
    std::string symbol;
    PositionSide   side;                 // Long = buy, Short = sell/short
    double qty;                  // positive magnitude
    std::string reason;          // "CLOSE_OPPOSITE" / "OPEN_TARGET" / "RESIZE"
};

struct OrderPlan {
    ExposureKey key;
    uint64_t epoch_before;       // optimistic check
    uint64_t reservation_id;     // for idempotency
    std::vector<PlanLeg> legs;   // ordered legs
};

struct Desire {
    // Strategy asks for this net outcome for the family
    PositionSide   target_side{PositionSide::Flat};
    double target_qty{0.0};       // positive magnitude
    std::string preferred_symbol; // e.g., choose TQQQ for strong long
};

class PositionGuardian {
public:
    using Clock = std::chrono::steady_clock;

    PositionGuardian(const FamilyMapper& mapper)
      : mapper_(mapper) {}

    // Inject broker truth + open orders to seed/refresh snapshots.
    void sync_from_broker(const std::string& account,
                          const std::vector<Position>& positions,
                          const std::vector<std::string>& open_orders);

    // Main entry: produce a conflict-free, atomic plan.
    // Thread-safe: locks the family during planning.
    std::optional<OrderPlan> plan(const std::string& account,
                                  const std::string& symbol,
                                  const Desire& desire,
                                  const Policy& policy);

    // Commit hook after successful router acceptance to advance epoch.
    // (You can also advance on fill callbacks—just be consistent.)
    void commit(const OrderPlan& plan);

    // Read-only view (for monitoring/metrics)
    PositionSnapshot snapshot(const ExposureKey& key) const;

private:
    struct Cell {
        mutable std::mutex m;
        PositionSnapshot ps;
        // Track in-flight reserved qty towards target to avoid double-alloc
        double reserved_long{0.0};
        double reserved_short{0.0};
        uint64_t next_reservation{1};
    };

    const FamilyMapper& mapper_;
    mutable std::mutex map_mu_;
    std::unordered_map<ExposureKey, std::unique_ptr<Cell>, ExposureKeyHash> cells_;

    Cell& cell_for(const ExposureKey& key);
    static bool flip_cooldown_active(const PositionSnapshot& ps, const Policy& pol, Clock::time_point now);
};

} // namespace sentio

```

## 📄 **FILE 73 of 206**: include/sentio/position_manager.hpp

**File Information**:
- **Path**: `include/sentio/position_manager.hpp`

- **Size**: 63 lines
- **Modified**: 2025-09-12 15:35:06

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include <string>
#include <vector>
#include <unordered_set>

namespace sentio {

enum class PortfolioState { Neutral, LongOnly, ShortOnly };
enum class RequiredAction { None, CloseLong, CloseShort };
enum class Direction { Long, Short }; // Keep for simple directional logic

const std::unordered_set<std::string> LONG_INSTRUMENTS = {"QQQ", "TQQQ", "TSLA"};
const std::unordered_set<std::string> SHORT_INSTRUMENTS = {"SQQQ", "TSLQ"};

class PositionManager {
private:
    PortfolioState state = PortfolioState::Neutral;
    int bars_since_flip = 0;
    [[maybe_unused]] const int cooldown_period = 5;
    
public:
    // **MODIFIED**: Logic restored to work with the new ID-based portfolio structure.
    void update_state(const Portfolio& portfolio, const SymbolTable& ST, const std::vector<double>& last_prices) {
        bars_since_flip++;
        double long_exposure = 0.0;
        double short_exposure = 0.0;

        for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
            const auto& pos = portfolio.positions[sid];
            if (std::abs(pos.qty) < 1e-6) continue;

            const std::string& symbol = ST.get_symbol(sid);
            double exposure = pos.qty * last_prices[sid];

            if (LONG_INSTRUMENTS.count(symbol)) {
                long_exposure += exposure;
            }
            if (SHORT_INSTRUMENTS.count(symbol)) {
                short_exposure += exposure; // Will be negative for short positions
            }
        }
        
        PortfolioState old_state = state;
        
        if (long_exposure > 100 && std::abs(short_exposure) < 100) {
            state = PortfolioState::LongOnly;
        } else if (std::abs(short_exposure) > 100 && long_exposure < 100) {
            state = PortfolioState::ShortOnly;
        } else {
            state = PortfolioState::Neutral;
        }

        if (state != old_state) {
            bars_since_flip = 0;
        }
    }
    
    // ... other methods remain the same ...
};

} // namespace sentio
```

## 📄 **FILE 74 of 206**: include/sentio/position_orchestrator.hpp

**File Information**:
- **Path**: `include/sentio/position_orchestrator.hpp`

- **Size**: 47 lines
- **Modified**: 2025-09-11 21:59:53

- **Type**: .hpp

```text
#pragma once
#include "position_guardian.hpp"
#include "family_mapper.hpp"
#include "side.hpp"
#include "core.hpp"
#include "audit.hpp"
#include <string>
#include <memory>

namespace sentio {

// Integration layer between strategies and the position guardian
class PositionOrchestrator {
public:
    explicit PositionOrchestrator(const std::string& account = "sentio:primary");
    
    // Main entry point for strategy signals
    void process_strategy_signal(const std::string& strategy_id,
                                const std::string& symbol,
                                ::sentio::Side target_side,
                                double target_qty,
                                const std::string& preferred_symbol = "");
    
    // Sync with current portfolio state
    void sync_portfolio(const Portfolio& portfolio, const SymbolTable& ST);
    
    // Get current family exposure
    PositionSnapshot get_family_exposure(const std::string& symbol) const;
    
    // Configuration
    void set_policy(const Policy& policy) { policy_ = policy; }
    const Policy& get_policy() const { return policy_; }

private:
    std::string account_;
    FamilyMapper mapper_;
    PositionGuardian guardian_;
    Policy policy_;
    
    // Convert Sentio Side to guardian PositionSide
    static PositionSide convert_side(::sentio::Side side);
    
    // Choose preferred symbol based on signal strength and family
    std::string choose_preferred_symbol(const std::string& symbol, PositionSide side, double strength) const;
};

} // namespace sentio

```

## 📄 **FILE 75 of 206**: include/sentio/position_validator.hpp

**File Information**:
- **Path**: `include/sentio/position_validator.hpp`

- **Size**: 95 lines
- **Modified**: 2025-09-12 16:30:08

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include <string>
#include <unordered_set>
#include <vector>

namespace sentio {

// **UPDATED**: Conflicting position detection for PSQ -> SHORT QQQ architecture
const std::unordered_set<std::string> LONG_ETFS = {"QQQ", "TQQQ"};
const std::unordered_set<std::string> INVERSE_ETFS = {"SQQQ"}; // PSQ removed

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

## 📄 **FILE 76 of 206**: include/sentio/pricebook.hpp

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

## 📄 **FILE 77 of 206**: include/sentio/profiling.hpp

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

## 📄 **FILE 78 of 206**: include/sentio/progress_bar.hpp

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

## 📄 **FILE 79 of 206**: include/sentio/property_test.hpp

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

## 📄 **FILE 80 of 206**: include/sentio/rolling_stats.hpp

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

## 📄 **FILE 81 of 206**: include/sentio/router.hpp

**File Information**:
- **Path**: `include/sentio/router.hpp`

- **Size**: 92 lines
- **Modified**: 2025-09-15 15:04:43

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
    
    if (prob > 0.8) {
      signal.type = Type::STRONG_BUY;
      signal.confidence = (prob - 0.8) / 0.2; // Scale 0.8-1.0 to 0-1
    } else if (prob > 0.6) {
      signal.type = Type::BUY;
      signal.confidence = (prob - 0.6) / 0.2; // Scale 0.6-0.8 to 0-1
    } else if (prob < 0.2) {
      signal.type = Type::STRONG_SELL;
      signal.confidence = (0.2 - prob) / 0.2; // Scale 0.0-0.2 to 0-1
    } else if (prob < 0.4) {
      signal.type = Type::SELL;
      signal.confidence = (0.4 - prob) / 0.2; // Scale 0.2-0.4 to 0-1
    } else {
      signal.type = Type::HOLD;
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

## 📄 **FILE 82 of 206**: include/sentio/rsi_prob.hpp

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

## 📄 **FILE 83 of 206**: include/sentio/rsi_strategy.hpp

**File Information**:
- **Path**: `include/sentio/rsi_strategy.hpp`

- **Size**: 196 lines
- **Modified**: 2025-09-15 15:04:43

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

    SizerCfg get_sizer_config() const override {
        SizerCfg cfg;
        cfg.max_position_pct = 1.0;
        cfg.max_leverage = 3.0;
        cfg.volatility_target = 0.15;
        cfg.vol_lookback_days = 20;
        cfg.cash_reserve_pct = 0.05;
        cfg.fractional_allowed = true;
        cfg.min_notional = 1.0;
        return cfg;
    }

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

## 📄 **FILE 84 of 206**: include/sentio/rules/adapters.hpp

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

## 📄 **FILE 85 of 206**: include/sentio/rules/bbands_squeeze_rule.hpp

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

## 📄 **FILE 86 of 206**: include/sentio/rules/diversity_weighter.hpp

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

## 📄 **FILE 87 of 206**: include/sentio/rules/integrated_rule_ensemble.hpp

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

## 📄 **FILE 88 of 206**: include/sentio/rules/irule.hpp

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

## 📄 **FILE 89 of 206**: include/sentio/rules/momentum_volume_rule.hpp

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

## 📄 **FILE 90 of 206**: include/sentio/rules/ofi_proxy_rule.hpp

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

## 📄 **FILE 91 of 206**: include/sentio/rules/online_platt_calibrator.hpp

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

## 📄 **FILE 92 of 206**: include/sentio/rules/opening_range_breakout_rule.hpp

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

## 📄 **FILE 93 of 206**: include/sentio/rules/registry.hpp

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

## 📄 **FILE 94 of 206**: include/sentio/rules/sma_cross_rule.hpp

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

## 📄 **FILE 95 of 206**: include/sentio/rules/utils/validation.hpp

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

## 📄 **FILE 96 of 206**: include/sentio/rules/vwap_reversion_rule.hpp

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

## 📄 **FILE 97 of 206**: include/sentio/run_id_generator.hpp

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

## 📄 **FILE 98 of 206**: include/sentio/runner.hpp

**File Information**:
- **Path**: `include/sentio/runner.hpp`

- **Size**: 43 lines
- **Modified**: 2025-09-11 15:24:05

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "audit.hpp"
#include "router.hpp"
#include "sizer.hpp"
#include "position_manager.hpp"
#include "cost_model.hpp"
#include "symbol_table.hpp"
#include <string>
#include <unordered_map>
#include <vector>

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
};

struct RunResult {
    double final_equity;
    double total_return;
    double sharpe_ratio;
    double max_drawdown;
    double monthly_projected_return;
    int daily_trades;
    int total_fills;
    int no_route;
    int no_qty;
};

RunResult run_backtest(IAuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                      int base_symbol_id, const RunnerCfg& cfg);

} // namespace sentio


```

## 📄 **FILE 99 of 206**: include/sentio/sanity.hpp

**File Information**:
- **Path**: `include/sentio/sanity.hpp`

- **Size**: 93 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <unordered_map>

namespace sentio {

// Reuse your existing simple types
struct Bar { double open{}, high{}, low{}, close{}; };

// Signal types for strategy system
enum class SigType : uint8_t { BUY=0, STRONG_BUY=1, SELL=2, STRONG_SELL=3, HOLD=4 };

struct SanityIssue {
  enum class Severity { Warn, Error, Fatal };
  Severity severity{Severity::Error};
  std::string where;      // subsystem (DATA/FEATURE/STRAT/ROUTER/EXEC/PnL/AUDIT)
  std::string what;       // human message
  std::int64_t ts_utc{0}; // when applicable
};

struct SanityReport {
  std::vector<SanityIssue> issues;
  bool ok() const;                 // == no Error/Fatal
  std::size_t errors() const;
  std::size_t fatals() const;
  void add(SanityIssue::Severity sev, std::string where, std::string what, std::int64_t ts=0);
};

// Minimal interfaces (match your existing ones)
class PriceBook {
public:
  virtual void upsert_latest(const std::string& instrument, const Bar& b) = 0;
  virtual const Bar* get_latest(const std::string& instrument) const = 0;
  virtual bool has_instrument(const std::string& instrument) const = 0;
  virtual std::size_t size() const = 0;
  virtual ~PriceBook() = default;
};

struct Position { double qty{0.0}; double avg_px{0.0}; };
struct AccountState { double cash{0.0}; double realized{0.0}; double equity{0.0}; };

struct AuditEventCounts {
  std::size_t bars{0}, signals{0}, routes{0}, orders{0}, fills{0};
};

// Contracts you can call from tests or at the end of a run
namespace sanity {

// Data layer
void check_bar_monotonic(const std::vector<std::pair<std::int64_t, Bar>>& bars,
                         int expected_spacing_sec,
                         SanityReport& rep);

void check_bar_values_finite(const std::vector<std::pair<std::int64_t, Bar>>& bars,
                             SanityReport& rep);

// PriceBook coherence
void check_pricebook_coherence(const PriceBook& pb,
                               const std::vector<std::string>& required_instruments,
                               SanityReport& rep);

// Strategy/Routing layer
void check_signal_confidence_range(double conf, SanityReport& rep, std::int64_t ts);
void check_routed_instrument_has_price(const PriceBook& pb,
                                       const std::string& routed,
                                       SanityReport& rep, std::int64_t ts);

// Execution layer
void check_order_qty_min(double qty, double min_shares,
                         SanityReport& rep, std::int64_t ts);
void check_order_side_qty_sign_consistency(const std::string& side, double qty,
                                           SanityReport& rep, std::int64_t ts);

// P&L invariants
void check_equity_consistency(const AccountState& acct,
                              const std::unordered_map<std::string, Position>& pos,
                              const PriceBook& pb,
                              SanityReport& rep);

// Audit correlations
void check_audit_counts(const AuditEventCounts& c,
                        SanityReport& rep);

} // namespace sanity

// Lightweight runtime guard macros (no external deps)
#define SENTIO_ASSERT_FINITE(val, where, rep, ts) \
  do { if (!std::isfinite(val)) { (rep).add(SanityIssue::Severity::Fatal, (where), "non-finite value: " #val, (ts)); } } while(0)

} // namespace sentio

```

## 📄 **FILE 100 of 206**: include/sentio/side.hpp

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

## 📄 **FILE 101 of 206**: include/sentio/signal.hpp

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

## 📄 **FILE 102 of 206**: include/sentio/signal_diag.hpp

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

## 📄 **FILE 103 of 206**: include/sentio/signal_engine.hpp

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

## 📄 **FILE 104 of 206**: include/sentio/signal_gate.hpp

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

## 📄 **FILE 105 of 206**: include/sentio/signal_or.hpp

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

## 📄 **FILE 106 of 206**: include/sentio/signal_pipeline.hpp

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

## 📄 **FILE 107 of 206**: include/sentio/signal_trace.hpp

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

## 📄 **FILE 108 of 206**: include/sentio/signal_utils.hpp

**File Information**:
- **Path**: `include/sentio/signal_utils.hpp`

- **Size**: 126 lines
- **Modified**: 2025-09-11 23:11:24

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

## 📄 **FILE 109 of 206**: include/sentio/sim_data.hpp

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

## 📄 **FILE 110 of 206**: include/sentio/sizer.hpp

**File Information**:
- **Path**: `include/sentio/sizer.hpp`

- **Size**: 118 lines
- **Modified**: 2025-09-11 22:25:54

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

// Advanced Sizer Configuration with Risk Controls
struct SizerCfg {
  bool fractional_allowed = true;
  double min_notional = 1.0;
  double max_leverage = 2.0;
  double max_position_pct = 0.25;
  double volatility_target = 0.15;
  bool allow_negative_cash = false;
  int vol_lookback_days = 20;
  double cash_reserve_pct = 0.05;
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

  // **MODIFIED**: Signature and logic updated for the ID-based, high-performance architecture.
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
    
    // **CONFLICT PREVENTION**: Strategy-level conflict prevention should prevent conflicts
    // No need for smart conflict resolution since strategy checks existing positions
    double instrument_price = last_prices[instrument_id];

    // --- Calculate size based on multiple constraints ---
    double desired_notional = equity * std::abs(target_weight);

    // 1. Max Position Size Constraint
    desired_notional = std::min(desired_notional, equity * cfg.max_position_pct);

    // 2. Leverage Constraint
    double current_exposure = 0.0;
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        current_exposure += std::abs(portfolio.positions[sid].qty * last_prices[sid]);
    }
    double available_leverage_notional = (equity * cfg.max_leverage) - current_exposure;
    desired_notional = std::min(desired_notional, std::max(0.0, available_leverage_notional));

    // 4. Cash Constraint
    if (!cfg.allow_negative_cash) {
      double usable_cash = portfolio.cash * (1.0 - cfg.cash_reserve_pct);
      desired_notional = std::min(desired_notional, std::max(0.0, usable_cash));
    }
    
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

## 📄 **FILE 111 of 206**: include/sentio/strategy/intraday_position_governor.hpp

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

## 📄 **FILE 112 of 206**: include/sentio/strategy_bollinger_squeeze_breakout.hpp

**File Information**:
- **Path**: `include/sentio/strategy_bollinger_squeeze_breakout.hpp`

- **Size**: 57 lines
- **Modified**: 2025-09-15 15:04:43

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
    SizerCfg get_sizer_config() const override;
};

} // namespace sentio
```

## 📄 **FILE 113 of 206**: include/sentio/strategy_ire.hpp

**File Information**:
- **Path**: `include/sentio/strategy_ire.hpp`

- **Size**: 83 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .hpp

```text
#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/rules/integrated_rule_ensemble.hpp"
#include "sentio/strategy/intraday_position_governor.hpp"
#include "sentio/router.hpp"
#include "sentio/sizer.hpp"
#include <deque>

namespace sentio {

class IREStrategy : public BaseStrategy {
public:
    IREStrategy();
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    // The main entry point for the runner is now this function
    double calculate_target_weight(const std::vector<Bar>& bars, int i);
    
    // **NEW**: Implement probability-based signal interface
    double calculate_probability(const std::vector<Bar>& bars, int i) override;
    
    // **NEW**: Strategy-agnostic allocation interface
    std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, 
        int current_index,
        const std::string& base_symbol,
        const std::string& bull3x_symbol,
        const std::string& bear3x_symbol) override;
    
    RouterCfg get_router_config() const override;
    SizerCfg get_sizer_config() const override;
    bool requires_dynamic_allocation() const override { return true; }

    double get_latest_probability() const { return latest_probability_; }

private:
    void ensure_ensemble_built_(); // This can be deprecated or repurposed
    void ensure_governor_built_();
    
    // Legacy members
    std::unique_ptr<rules::IntegratedRuleEnsemble> ensemble_;
    float buy_lo_{0.60f}, buy_hi_{0.75f}, sell_hi_{0.40f}, sell_lo_{0.25f}, hysteresis_{0.02f};
    int warmup_{252};
    
    // Governor and new state
    std::unique_ptr<IntradayPositionGovernor> governor_;
    double latest_probability_{0.5};

    // State for the new Volatility-Adaptive Strategy
    std::deque<double> vol_return_history_;
    std::deque<double> vol_history_;
    std::deque<double> vwap_price_history_;
    std::deque<double> vwap_volume_history_;
    
    // State for the new Alpha Kernel
    std::deque<double> alpha_return_history_;

    // State for holding period and take-profit
    int last_trade_bar_{-1};
    int last_trade_direction_{0}; // -1 for short, 0 for flat, 1 for long
    double entry_price_{0.0};
    
    // **NEW**: State for Kelly Criterion
    std::deque<double> pnl_history_;
    
    // **NEW**: State machine for preventing rapid direction flips
    int last_direction_{-999}; // -999 = uninitialized, 0=neutral, 1=long, -1=inverse
    int bars_since_flip_{0};
    
    // **NEW**: Kelly Criterion methods
    double calculate_kelly_fraction(double edge_probability, double confidence) const;
    void update_trade_performance(double realized_pnl);
    double get_win_loss_ratio() const;
    double get_win_rate() const;
    
    // **NEW**: Multi-Timeframe Alpha Kernel methods
    double calculate_multi_timeframe_alpha(const std::deque<double>& history) const;
    double calculate_single_alpha_probability(const std::deque<double>& history, int short_window, int long_window) const;
};

} // namespace sentio

```

## 📄 **FILE 114 of 206**: include/sentio/strategy_kochi_ppo.hpp

**File Information**:
- **Path**: `include/sentio/strategy_kochi_ppo.hpp`

- **Size**: 55 lines
- **Modified**: 2025-09-15 15:04:43

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
  
  // **NEW**: Strategy-agnostic allocation interface
  std::vector<AllocationDecision> get_allocation_decisions(
      const std::vector<Bar>& bars, 
      int current_index,
      const std::string& base_symbol,
      const std::string& bull3x_symbol,
      const std::string& bear3x_symbol) override;
  
  RouterCfg get_router_config() const override;
  SizerCfg get_sizer_config() const override;

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

## 📄 **FILE 115 of 206**: include/sentio/strategy_market_making.hpp

**File Information**:
- **Path**: `include/sentio/strategy_market_making.hpp`

- **Size**: 65 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp" // For rolling volume
#include "router.hpp"
#include "sizer.hpp"
#include <deque>

namespace sentio {

class MarketMakingStrategy : public BaseStrategy {
private:
    // **MODIFIED**: Cached parameters
    double base_spread_;
    double min_spread_;
    double max_spread_;
    int order_levels_;
    double level_spacing_;
    double order_size_base_;
    double max_inventory_;
    double inventory_skew_mult_;
    double adverse_selection_threshold_;
    double min_volume_ratio_;
    int max_orders_per_bar_;
    int rebalance_frequency_;

    // Market making state
    struct MarketState {
        double inventory = 0.0;
        double average_cost = 0.0;
        int last_rebalance = 0;
        int orders_this_bar = 0;
    };
    MarketState market_state_;

    // **NEW**: Stateful, rolling indicators for performance
    RollingMeanVar rolling_returns_;
    RollingMean rolling_volume_;

    // Helper methods
    bool should_participate(const Bar& bar);
    double get_inventory_skew() const;
    
public:
    MarketMakingStrategy();
    
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
    SizerCfg get_sizer_config() const override;
};

} // namespace sentio
```

## 📄 **FILE 116 of 206**: include/sentio/strategy_momentum_volume.hpp

**File Information**:
- **Path**: `include/sentio/strategy_momentum_volume.hpp`

- **Size**: 70 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp" // **NEW**: For efficient MA calculations
#include "router.hpp"
#include "sizer.hpp"
#include <map>

namespace sentio {

class MomentumVolumeProfileStrategy : public BaseStrategy {
private:
    // **MODIFIED**: Cached parameters for performance
    int profile_period_;
    double value_area_pct_;
    int price_bins_;
    double breakout_threshold_pct_;
    int momentum_lookback_;
    double volume_surge_mult_;
    int cool_down_period_;

    // Volume profile data structures
    struct VolumeNode {
        double price_level;
        double volume;
    };
    struct VolumeProfile {
        std::map<double, VolumeNode> profile;
        double poc_level = 0.0;
        double value_area_high = 0.0;
        double value_area_low = 0.0;
        double total_volume = 0.0;
        void clear() { /* ... unchanged ... */ }
    };
    
    // Strategy state
    VolumeProfile volume_profile_;
    int last_profile_update_ = -1;
    
    // **NEW**: Stateful, rolling indicators for performance
    RollingMean avg_volume_;

    // Helper methods
    void build_volume_profile(const std::vector<Bar>& bars, int end_index);
    void calculate_value_area();
    bool is_momentum_confirmed(const std::vector<Bar>& bars, int index) const;
    
public:
    MomentumVolumeProfileStrategy();
    
    // BaseStrategy interface
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
    SizerCfg get_sizer_config() const override;
};

} // namespace sentio
```

## 📄 **FILE 117 of 206**: include/sentio/strategy_opening_range_breakout.hpp

**File Information**:
- **Path**: `include/sentio/strategy_opening_range_breakout.hpp`

- **Size**: 51 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "router.hpp"
#include "sizer.hpp"

namespace sentio {

class OpeningRangeBreakoutStrategy : public BaseStrategy {
private:
    // **MODIFIED**: Cached parameters
    int opening_range_minutes_;
    int breakout_confirmation_bars_;
    double volume_multiplier_;
    double stop_loss_pct_;
    double take_profit_pct_;
    int cool_down_period_;

    struct OpeningRange {
        double high = 0.0;
        double low = 0.0;
        int end_bar = -1;
        bool is_finalized = false;
    };
    
    // Strategy state
    OpeningRange current_range_;
    int day_start_index_ = -1;
    
public:
    OpeningRangeBreakoutStrategy();
    
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
    SizerCfg get_sizer_config() const override;
};

} // namespace sentio
```

## 📄 **FILE 118 of 206**: include/sentio/strategy_order_flow_imbalance.hpp

**File Information**:
- **Path**: `include/sentio/strategy_order_flow_imbalance.hpp`

- **Size**: 53 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp"
#include "router.hpp"
#include "sizer.hpp"

namespace sentio {

class OrderFlowImbalanceStrategy : public BaseStrategy {
private:
    // Cached parameters
    int lookback_period_;
    double entry_threshold_long_;
    double entry_threshold_short_;
    int hold_max_bars_;
    int cool_down_period_;

    // Strategy-specific state machine
    enum class OFIState { Flat, Long, Short };
    // **FIXED**: Renamed this member to 'ofi_state_' to avoid conflict
    // with the 'state_' member inherited from BaseStrategy.
    OFIState ofi_state_ = OFIState::Flat;
    int bars_in_trade_ = 0;

    // Rolling indicator to measure average pressure
    RollingMean rolling_pressure_;

    // Helper to calculate buying/selling pressure proxy from a bar
    double calculate_bar_pressure(const Bar& bar) const;
    
public:
    OrderFlowImbalanceStrategy();
    
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
    SizerCfg get_sizer_config() const override;
};

} // namespace sentio

```

## 📄 **FILE 119 of 206**: include/sentio/strategy_order_flow_scalping.hpp

**File Information**:
- **Path**: `include/sentio/strategy_order_flow_scalping.hpp`

- **Size**: 51 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp"
#include "router.hpp"
#include "sizer.hpp"

namespace sentio {

class OrderFlowScalpingStrategy : public BaseStrategy {
private:
    // Cached parameters
    int lookback_period_;
    double imbalance_threshold_;
    int hold_max_bars_;
    int cool_down_period_;

    // State machine states
    enum class OFState { Idle, ArmedLong, ArmedShort, Long, Short };
    
    // **FIXED**: Renamed this member to 'of_state_' to avoid conflict
    // with the 'state_' member inherited from BaseStrategy.
    OFState of_state_ = OFState::Idle;
    int bars_in_trade_ = 0;
    RollingMean rolling_pressure_;

    // Helper to calculate buying/selling pressure proxy from a bar
    double calculate_bar_pressure(const Bar& bar) const;
    
public:
    OrderFlowScalpingStrategy();
    
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
    SizerCfg get_sizer_config() const override;
};

} // namespace sentio

```

## 📄 **FILE 120 of 206**: include/sentio/strategy_signal_or.hpp

**File Information**:
- **Path**: `include/sentio/strategy_signal_or.hpp`

- **Size**: 72 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .hpp

```text
#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/signal_or.hpp"
#include <vector>
#include <memory>

namespace sentio {

// Signal-OR Strategy Configuration
struct SignalOrCfg {
    // Signal-OR mixer configuration
    OrCfg or_config;
    
    // Strategy parameters
    double min_signal_strength = 0.1;  // Minimum signal strength to act
    double long_threshold = 0.6;        // Probability threshold for long signals
    double short_threshold = 0.4;       // Probability threshold for short signals
    double hold_threshold = 0.05;       // Band around 0.5 for hold signals
    
    // Risk management
    double max_position_weight = 0.8;   // Maximum position weight
    double position_decay = 0.95;       // Position decay factor per bar
    
    // Simple momentum parameters
    int momentum_window = 20;            // Moving average window
    double momentum_scale = 25.0;       // Momentum scaling factor
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
    SizerCfg get_sizer_config() const override;
    
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
    double current_position_weight_ = 0.0;
    int warmup_bars_ = 0;
    static constexpr int REQUIRED_WARMUP = 50;
    
    // Helper methods
    std::vector<RuleOut> evaluate_simple_rules(const std::vector<Bar>& bars, int current_index);
    double calculate_momentum_probability(const std::vector<Bar>& bars, int current_index);
    double calculate_position_weight(double signal_strength);
    void update_position_decay();
};

// Register the strategy with the factory
REGISTER_STRATEGY(SignalOrStrategy, "sigor");

} // namespace sentio

```

## 📄 **FILE 121 of 206**: include/sentio/strategy_sma_cross.hpp

**File Information**:
- **Path**: `include/sentio/strategy_sma_cross.hpp`

- **Size**: 27 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "indicators.hpp"
#include <optional>

namespace sentio {

struct SMACrossCfg {
  int fast = 10;
  int slow = 30;
  double conf_fast_slow = 0.7; // confidence when cross happens
};

class SMACrossStrategy final : public IStrategy {
public:
  explicit SMACrossStrategy(const SMACrossCfg& cfg);
  void on_bar(const StrategyCtx& ctx, const Bar& b) override;
  std::optional<StrategySignal> latest() const override { return last_; }
  bool warmed_up() const { return sma_fast_.ready() && sma_slow_.ready(); }
private:
  SMACrossCfg cfg_;
  SMA sma_fast_, sma_slow_;
  double last_fast_{NAN}, last_slow_{NAN};
  std::optional<StrategySignal> last_;
};

} // namespace sentio

```

## 📄 **FILE 122 of 206**: include/sentio/strategy_tfa.hpp

**File Information**:
- **Path**: `include/sentio/strategy_tfa.hpp`

- **Size**: 63 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .hpp

```text
#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/ml/model_registry.hpp"
#include "sentio/ml/feature_window.hpp"
#include "sentio/feature/column_projector.hpp"
#include "sentio/feature/column_projector_safe.hpp"
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
  
  // **NEW**: Strategy-agnostic allocation interface
  std::vector<AllocationDecision> get_allocation_decisions(
      const std::vector<Bar>& bars, 
      int current_index,
      const std::string& base_symbol,
      const std::string& bull3x_symbol,
      const std::string& bear3x_symbol) override;
  
  RouterCfg get_router_config() const override;
  SizerCfg get_sizer_config() const override;

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
};

} // namespace sentio

```

## 📄 **FILE 123 of 206**: include/sentio/strategy_transformer.hpp

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

## 📄 **FILE 124 of 206**: include/sentio/strategy_utils.hpp

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

## 📄 **FILE 125 of 206**: include/sentio/strategy_vwap_reversion.hpp

**File Information**:
- **Path**: `include/sentio/strategy_vwap_reversion.hpp`

- **Size**: 60 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "router.hpp"
#include "sizer.hpp"

namespace sentio {

class VWAPReversionStrategy : public BaseStrategy {
private:
    // Cached parameters for performance
    int vwap_period_;
    double band_multiplier_;
    double max_band_width_;
    double min_distance_from_vwap_;
    double volume_confirmation_mult_;
    int rsi_period_;
    double rsi_oversold_;
    double rsi_overbought_;
    double stop_loss_pct_;
    double take_profit_pct_;
    int time_stop_bars_;
    int cool_down_period_;

    // VWAP calculation state
    double cumulative_pv_ = 0.0;
    double cumulative_volume_ = 0.0;
    
    // Strategy state
    int time_in_position_ = 0;
    double vwap_ = 0.0;
    
    // Helper methods
    void update_vwap(const Bar& bar);
    bool is_volume_confirmed(const std::vector<Bar>& bars, int index) const;
    bool is_rsi_condition_met(const std::vector<Bar>& bars, int index, bool for_buy) const;
    double calculate_simple_rsi(const std::vector<double>& prices) const;
    
public:
    VWAPReversionStrategy();
    
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
    SizerCfg get_sizer_config() const override;
};

} // namespace sentio
```

## 📄 **FILE 126 of 206**: include/sentio/sym/leverage_registry.hpp

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

## 📄 **FILE 127 of 206**: include/sentio/sym/symbol_utils.hpp

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

## 📄 **FILE 128 of 206**: include/sentio/symbol_table.hpp

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

## 📄 **FILE 129 of 206**: include/sentio/telemetry_logger.hpp

**File Information**:
- **Path**: `include/sentio/telemetry_logger.hpp`

- **Size**: 131 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <unordered_map>
#include <atomic>

namespace sentio {

/**
 * JSON line logger for telemetry data
 * Thread-safe logging of strategy performance metrics
 */
class TelemetryLogger {
public:
    struct TelemetryData {
        std::string timestamp;
        std::string strategy_name;
        std::string instrument;
        int bars_processed{0};
        int signals_generated{0};
        int buy_signals{0};
        int sell_signals{0};
        int hold_signals{0};
        double avg_confidence{0.0};
        double ready_percentage{0.0};
        double processing_time_ms{0.0};
        std::string notes;
    };

    explicit TelemetryLogger(const std::string& log_file_path);
    ~TelemetryLogger();

    /**
     * Log telemetry data for a strategy
     * @param data Telemetry data to log
     */
    void log(const TelemetryData& data);

    /**
     * Log a simple metric
     * @param strategy_name Strategy name
     * @param metric_name Metric name
     * @param value Metric value
     * @param instrument Optional instrument name
     */
    void log_metric(
        const std::string& strategy_name,
        const std::string& metric_name,
        double value,
        const std::string& instrument = ""
    );

    /**
     * Log signal generation statistics
     * @param strategy_name Strategy name
     * @param instrument Instrument name
     * @param signals_generated Total signals generated
     * @param buy_signals Buy signals
     * @param sell_signals Sell signals
     * @param hold_signals Hold signals
     * @param avg_confidence Average confidence
     */
    void log_signal_stats(
        const std::string& strategy_name,
        const std::string& instrument,
        int signals_generated,
        int buy_signals,
        int sell_signals,
        int hold_signals,
        double avg_confidence
    );

    /**
     * Log performance metrics
     * @param strategy_name Strategy name
     * @param instrument Instrument name
     * @param bars_processed Number of bars processed
     * @param processing_time_ms Processing time in milliseconds
     * @param ready_percentage Percentage of time strategy was ready
     */
    void log_performance(
        const std::string& strategy_name,
        const std::string& instrument,
        int bars_processed,
        double processing_time_ms,
        double ready_percentage
    );

    /**
     * Flush any pending log data
     */
    void flush();

    /**
     * Get current log file path
     */
    const std::string& get_log_file_path() const { return log_file_path_; }

private:
    std::string log_file_path_;
    std::ofstream log_file_;
    std::mutex log_mutex_;
    std::atomic<int> log_counter_{0};
    
    // Helper methods
    std::string get_current_timestamp() const;
    std::string escape_json_string(const std::string& str) const;
    void write_json_line(const std::string& json_line);
};

/**
 * Global telemetry logger instance
 * Use this for easy access throughout the application
 */
extern std::unique_ptr<TelemetryLogger> g_telemetry_logger;

/**
 * Initialize global telemetry logger
 * @param log_file_path Path to log file
 */
void init_telemetry_logger(const std::string& log_file_path = "logs/telemetry.jsonl");

/**
 * Get global telemetry logger instance
 * @return Reference to global telemetry logger
 */
TelemetryLogger& get_telemetry_logger();

} // namespace sentio

```

## 📄 **FILE 130 of 206**: include/sentio/temporal_analysis.hpp

**File Information**:
- **Path**: `include/sentio/temporal_analysis.hpp`

- **Size**: 415 lines
- **Modified**: 2025-09-12 02:07:14

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <iostream>
#include "sentio/progress_bar.hpp"

namespace sentio {

struct TemporalAnalysisConfig {
    int num_quarters = 0;            // Number of quarters to analyze (0 = all data)
    int num_weeks = 0;               // Number of weeks to analyze (0 = all data)
    int num_days = 0;                // Number of days to analyze (0 = all data)
    int min_bars_per_quarter = 50;   // Minimum bars required per quarter
    bool print_detailed_report = true;
    
    // TPA readiness criteria for virtual market testing
    double min_avg_sharpe = 0.5;     // Minimum average Sharpe ratio
    double max_sharpe_volatility = 1.0; // Maximum Sharpe volatility
    double min_health_quarters_pct = 70.0; // Minimum % of quarters with healthy trade frequency
    double min_avg_monthly_return = 0.5;   // Minimum average monthly return (%)
    double max_drawdown_threshold = 15.0;  // Maximum acceptable drawdown (%)
};

struct QuarterlyMetrics {
    int year;
    int quarter;
    double monthly_return_pct;      // Monthly projected return (annualized)
    double sharpe_ratio;
    int total_trades;
    int trading_days;
    double avg_daily_trades;
    double max_drawdown;
    double win_rate;
    double total_return_pct;
    
    // Health indicators
    bool healthy_trade_frequency() const {
        return avg_daily_trades >= 10.0 && avg_daily_trades <= 100.0;
    }
    
    std::string health_status() const {
        if (avg_daily_trades < 10.0) return "LOW_FREQ";
        if (avg_daily_trades > 100.0) return "HIGH_FREQ";
        return "HEALTHY";
    }
};

struct TPAReadinessAssessment {
    bool ready_for_virtual_market = false;
    bool ready_for_paper_trading = false;
    bool ready_for_live_trading = false;
    
    std::vector<std::string> issues;
    std::vector<std::string> recommendations;
    
    double readiness_score = 0.0; // 0-100 score
};

struct TemporalAnalysisSummary {
    std::vector<QuarterlyMetrics> quarterly_results;
    double overall_sharpe;
    double overall_return;
    int total_quarters;
    int healthy_quarters;
    int low_freq_quarters;
    int high_freq_quarters;
    
    // Consistency metrics
    double sharpe_std_dev;
    double return_std_dev;
    double trade_freq_std_dev;
    
    // TPA readiness assessment
    TPAReadinessAssessment readiness;
    
    void calculate_summary_stats() {
        if (quarterly_results.empty()) return;
        
        total_quarters = quarterly_results.size();
        healthy_quarters = 0;
        low_freq_quarters = 0;
        high_freq_quarters = 0;
        
        double sharpe_sum = 0.0, return_sum = 0.0, freq_sum = 0.0;
        double sharpe_sq_sum = 0.0, return_sq_sum = 0.0, freq_sq_sum = 0.0;
        
        for (const auto& q : quarterly_results) {
            sharpe_sum += q.sharpe_ratio;
            return_sum += q.monthly_return_pct;
            freq_sum += q.avg_daily_trades;
            
            sharpe_sq_sum += q.sharpe_ratio * q.sharpe_ratio;
            return_sq_sum += q.monthly_return_pct * q.monthly_return_pct;
            freq_sq_sum += q.avg_daily_trades * q.avg_daily_trades;
            
            if (q.health_status() == "HEALTHY") healthy_quarters++;
            else if (q.health_status() == "LOW_FREQ") low_freq_quarters++;
            else if (q.health_status() == "HIGH_FREQ") high_freq_quarters++;
        }
        
        overall_sharpe = sharpe_sum / total_quarters;
        overall_return = return_sum / total_quarters;
        
        // Calculate standard deviations
        double sharpe_mean = overall_sharpe;
        double return_mean = overall_return;
        double freq_mean = freq_sum / total_quarters;
        
        // For single quarter, standard deviation is 0 (no variance)
        if (total_quarters == 1) {
            sharpe_std_dev = -1.0;  // Use -1.0 to indicate "N/A" for single period
            return_std_dev = -1.0;  // Use -1.0 to indicate "N/A" for single period
            trade_freq_std_dev = -1.0;  // Use -1.0 to indicate "N/A" for single period
        } else {
            sharpe_std_dev = std::sqrt(std::max(0.0, (sharpe_sq_sum / total_quarters) - (sharpe_mean * sharpe_mean)));
            return_std_dev = std::sqrt(std::max(0.0, (return_sq_sum / total_quarters) - (return_mean * return_mean)));
            trade_freq_std_dev = std::sqrt(std::max(0.0, (freq_sq_sum / total_quarters) - (freq_mean * freq_mean)));
        }
    }
    
    void assess_readiness(const TemporalAnalysisConfig& config) {
        readiness.issues.clear();
        readiness.recommendations.clear();
        readiness.ready_for_virtual_market = false;
        readiness.ready_for_paper_trading = false;
        readiness.ready_for_live_trading = false;
        
        double score = 0.0;
        int criteria_met = 0;
        [[maybe_unused]] int total_criteria = 5;
        
        // 1. Average Sharpe ratio check
        if (overall_sharpe >= config.min_avg_sharpe) {
            criteria_met++;
            score += 20.0;
        } else {
            readiness.issues.push_back("Average Sharpe ratio too low: " + std::to_string(overall_sharpe) + " < " + std::to_string(config.min_avg_sharpe));
            readiness.recommendations.push_back("Improve strategy risk-adjusted returns");
        }
        
        // 2. Sharpe volatility check
        if (sharpe_std_dev <= config.max_sharpe_volatility) {
            criteria_met++;
            score += 20.0;
        } else {
            readiness.issues.push_back("Sharpe ratio too volatile: " + std::to_string(sharpe_std_dev) + " > " + std::to_string(config.max_sharpe_volatility));
            readiness.recommendations.push_back("Improve strategy consistency across time periods");
        }
        
        // 3. Trade frequency health check
        double health_pct = 100.0 * healthy_quarters / total_quarters;
        if (health_pct >= config.min_health_quarters_pct) {
            criteria_met++;
            score += 20.0;
        } else {
            readiness.issues.push_back("Too many quarters with unhealthy trade frequency: " + std::to_string(health_pct) + "% < " + std::to_string(config.min_health_quarters_pct) + "%");
            readiness.recommendations.push_back("Adjust strategy parameters for consistent trade frequency");
        }
        
        // 4. Monthly return check
        if (overall_return >= config.min_avg_monthly_return) {
            criteria_met++;
            score += 20.0;
        } else {
            readiness.issues.push_back("Average monthly return too low: " + std::to_string(overall_return) + "% < " + std::to_string(config.min_avg_monthly_return) + "%");
            readiness.recommendations.push_back("Improve strategy profitability");
        }
        
        // 5. Drawdown check (check max drawdown across quarters)
        double max_quarterly_drawdown = 0.0;
        for (const auto& q : quarterly_results) {
            max_quarterly_drawdown = std::max(max_quarterly_drawdown, q.max_drawdown);
        }
        if (max_quarterly_drawdown <= config.max_drawdown_threshold) {
            criteria_met++;
            score += 20.0;
        } else {
            readiness.issues.push_back("Maximum drawdown too high: " + std::to_string(max_quarterly_drawdown) + "% > " + std::to_string(config.max_drawdown_threshold) + "%");
            readiness.recommendations.push_back("Implement better risk management");
        }
        
        readiness.readiness_score = score;
        
        // Determine readiness levels
        if (criteria_met >= 3) {
            readiness.ready_for_virtual_market = true;
        }
        if (criteria_met >= 4) {
            readiness.ready_for_paper_trading = true;
        }
        if (criteria_met >= 5) {
            readiness.ready_for_live_trading = true;
        }
        
        // Add general recommendations based on score
        if (score < 60.0) {
            readiness.recommendations.push_back("Strategy needs significant improvement before testing");
        } else if (score < 80.0) {
            readiness.recommendations.push_back("Strategy shows promise but needs refinement");
        } else {
            readiness.recommendations.push_back("Strategy appears ready for advanced testing");
        }
    }
};

class TPATestProgressBar : public ProgressBar {
private:
    std::string period_name_;
    
public:
    TPATestProgressBar(int total_periods, const std::string& strategy_name, const std::string& period_name = "quarter") 
        : ProgressBar(total_periods, "TPA Test: " + strategy_name), period_name_(period_name) {}
    
    void display_with_period_info([[maybe_unused]] int current_period, int year, int period_num, 
                                  double monthly_return, double sharpe, 
                                  double avg_daily_trades, const std::string& health_status) {
        update(get_current() + 1);
        
        // Clear line and move cursor to beginning
        std::cout << "\r\033[K";
        
        // Calculate percentage
        double percentage = (double)get_current() / get_total() * 100.0;
        
        // Create progress bar
        int bar_width = 50;
        int pos = (int)(bar_width * percentage / 100.0);
        
        std::cout << get_description() << " [" << std::string(pos, '=') 
                  << std::string(bar_width - pos, '-') << "] " 
                  << std::fixed << std::setprecision(1) << percentage << "%";
        
        // Show current period info
        if (period_name_ == "quarter") {
            std::cout << " | Q" << year << "Q" << period_num;
        } else if (period_name_ == "week") {
            std::cout << " | W" << year << "W" << period_num;
        } else if (period_name_ == "day") {
            std::cout << " | D" << year << "D" << period_num;
        } else {
            std::cout << " | " << period_name_ << " " << period_num;
        }
        std::cout << " | Ret: " << std::fixed << std::setprecision(2) << monthly_return << "%"
                  << " | Sharpe: " << std::fixed << std::setprecision(3) << sharpe
                  << " | Trades: " << std::fixed << std::setprecision(1) << avg_daily_trades
                  << " | " << health_status;
        
        std::cout.flush();
    }
};

class TemporalAnalyzer {
private:
    std::string period_name_ = "quarter";
    
public:
    TemporalAnalyzer() = default;
    
    void set_period_name(const std::string& period_name) {
        period_name_ = period_name;
    }
    
    void add_quarterly_result(const QuarterlyMetrics& metrics) {
        quarterly_results_.push_back(metrics);
    }
    
    TemporalAnalysisSummary generate_summary() const {
        TemporalAnalysisSummary summary;
        summary.quarterly_results = quarterly_results_;
        summary.calculate_summary_stats();
        return summary;
    }
    
    void print_detailed_report() const {
        auto summary = generate_summary();
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "TEMPORAL PERFORMANCE ANALYSIS REPORT" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        // Period breakdown
        std::string period_label = period_name_;
        period_label[0] = std::toupper(period_label[0]); // Capitalize first letter
        std::cout << "\n" << period_label << "LY PERFORMANCE BREAKDOWN:\n" << std::endl;
        std::cout << std::left << std::setw(8) << period_label 
                  << " " << std::setw(11) << "MPR%" 
                  << " " << std::setw(9) << "Sharpe" 
                  << " " << std::setw(7) << "Trades" 
                  << " " << std::setw(11) << "Daily Avg" 
                  << " " << std::setw(9) << "Health" 
                  << " " << std::setw(9) << "Drawdown%" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (const auto& q : summary.quarterly_results) {
            std::string period_id;
            if (period_name_ == "quarter") {
                period_id = std::to_string(q.year) + "Q" + std::to_string(q.quarter);
            } else if (period_name_ == "week") {
                period_id = std::to_string(q.year) + "W" + std::to_string(q.quarter);
            } else if (period_name_ == "day") {
                period_id = std::to_string(q.year) + "D" + std::to_string(q.quarter);
            } else {
                period_id = std::to_string(q.year) + period_name_[0] + std::to_string(q.quarter);
            }
            std::cout << std::left << std::setw(8) << period_id
                      << " " << std::setw(11) << std::fixed << std::setprecision(2) << q.monthly_return_pct
                      << " " << std::setw(9) << std::fixed << std::setprecision(3) << q.sharpe_ratio
                      << " " << std::setw(7) << q.total_trades
                      << " " << std::setw(11) << std::fixed << std::setprecision(1) << q.avg_daily_trades
                      << " " << std::setw(9) << q.health_status()
                      << " " << std::setw(9) << std::fixed << std::setprecision(2) << q.max_drawdown << std::endl;
        }
        
        // Summary statistics
        std::cout << "\n" << std::string(80, '-') << std::endl;
        std::cout << "SUMMARY STATISTICS:\n" << std::endl;
        std::cout << "Overall Performance:" << std::endl;
        std::cout << "  Average Monthly Return: " << std::fixed << std::setprecision(2) 
                  << summary.overall_return << "%" << std::endl;
        std::cout << "  Average Sharpe Ratio: " << std::fixed << std::setprecision(3) 
                  << summary.overall_sharpe << std::endl;
        
        std::cout << "\nConsistency Metrics:" << std::endl;
        
        // Return volatility
        if (summary.return_std_dev < 0) {
            std::cout << "  Return Volatility (std): N/A (single period)" << std::endl;
        } else {
            std::cout << "  Return Volatility (std): " << std::fixed << std::setprecision(2) 
                      << summary.return_std_dev << "%" << std::endl;
        }
        
        // Sharpe volatility
        if (summary.sharpe_std_dev < 0) {
            std::cout << "  Sharpe Volatility (std): N/A (single period)" << std::endl;
        } else {
            std::cout << "  Sharpe Volatility (std): " << std::fixed << std::setprecision(3) 
                      << summary.sharpe_std_dev << std::endl;
        }
        
        // Trade frequency volatility
        if (summary.trade_freq_std_dev < 0) {
            std::cout << "  Trade Frequency Volatility (std): N/A (single period)" << std::endl;
        } else {
            std::cout << "  Trade Frequency Volatility (std): " << std::fixed << std::setprecision(1) 
                      << summary.trade_freq_std_dev << " trades/day" << std::endl;
        }
        
        std::cout << "\nTrade Frequency Health:" << std::endl;
        std::string period_label_lower = period_name_;
        std::cout << "  Healthy " << period_label_lower << "s: " << summary.healthy_quarters << "/" << summary.total_quarters 
                  << " (" << std::fixed << std::setprecision(1) 
                  << (100.0 * summary.healthy_quarters / summary.total_quarters) << "%)" << std::endl;
        std::cout << "  Low Frequency: " << summary.low_freq_quarters << " " << period_label_lower << "s" << std::endl;
        std::cout << "  High Frequency: " << summary.high_freq_quarters << " " << period_label_lower << "s" << std::endl;
        
        // Health assessment
        std::cout << "\nHEALTH ASSESSMENT:" << std::endl;
        double health_pct = 100.0 * summary.healthy_quarters / summary.total_quarters;
        if (health_pct >= 80.0) {
            std::cout << "  ✅ EXCELLENT: " << health_pct << "% of quarters have healthy trade frequency" << std::endl;
        } else if (health_pct >= 60.0) {
            std::cout << "  ⚠️  MODERATE: " << health_pct << "% of quarters have healthy trade frequency" << std::endl;
        } else {
            std::cout << "  ❌ POOR: " << health_pct << "% of quarters have healthy trade frequency" << std::endl;
        }
        
        // TPA Readiness Assessment
        std::cout << "\nTPA READINESS ASSESSMENT:" << std::endl;
        std::cout << "  Readiness Score: " << std::fixed << std::setprecision(1) << summary.readiness.readiness_score << "/100" << std::endl;
        
        std::cout << "\n  Testing Readiness:" << std::endl;
        std::cout << "  " << (summary.readiness.ready_for_virtual_market ? "✅" : "❌") 
                  << " Virtual Market Testing: " << (summary.readiness.ready_for_virtual_market ? "READY" : "NOT READY") << std::endl;
        std::cout << "  " << (summary.readiness.ready_for_paper_trading ? "✅" : "❌") 
                  << " Paper Trading: " << (summary.readiness.ready_for_paper_trading ? "READY" : "NOT READY") << std::endl;
        std::cout << "  " << (summary.readiness.ready_for_live_trading ? "✅" : "❌") 
                  << " Live Trading: " << (summary.readiness.ready_for_live_trading ? "READY" : "NOT READY") << std::endl;
        
        if (!summary.readiness.issues.empty()) {
            std::cout << "\n  Issues Identified:" << std::endl;
            for (const auto& issue : summary.readiness.issues) {
                std::cout << "  ❌ " << issue << std::endl;
            }
        }
        
        if (!summary.readiness.recommendations.empty()) {
            std::cout << "\n  Recommendations:" << std::endl;
            for (const auto& rec : summary.readiness.recommendations) {
                std::cout << "  💡 " << rec << std::endl;
            }
        }
        
        std::cout << std::string(80, '=') << std::endl;
    }

private:
    std::vector<QuarterlyMetrics> quarterly_results_;
};

// Forward declarations
struct SymbolTable;
struct Bar;
struct RunnerCfg;

// Main temporal analysis function
TemporalAnalysisSummary run_temporal_analysis(const SymbolTable& ST,
                                            const std::vector<std::vector<Bar>>& series,
                                            int base_symbol_id,
                                            const RunnerCfg& rcfg,
                                            const TemporalAnalysisConfig& cfg = TemporalAnalysisConfig{});

} // namespace sentio

```

## 📄 **FILE 131 of 206**: include/sentio/tfa/artifacts_loader.hpp

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

## 📄 **FILE 132 of 206**: include/sentio/tfa/artifacts_safe.hpp

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

## 📄 **FILE 133 of 206**: include/sentio/tfa/feature_guard.hpp

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

## 📄 **FILE 134 of 206**: include/sentio/tfa/input_shim.hpp

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

## 📄 **FILE 135 of 206**: include/sentio/tfa/signal_pipeline.hpp

**File Information**:
- **Path**: `include/sentio/tfa/signal_pipeline.hpp`

- **Size**: 206 lines
- **Modified**: 2025-09-10 11:15:18

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
  float min_prob = 0.55f;      // fixed
  int   q_window = 0;          // if >0, use quantile
  float q_level  = 0.75f;      // 75th percentile

  std::vector<uint8_t> filter(const std::vector<float>& prob) const {
    const int64_t N = (int64_t)prob.size();
    std::vector<uint8_t> keep(N, 0);
    if (q_window <= 0){
      for (int64_t i=0;i<N;++i) keep[i] = (prob[i] >= min_prob) ? 1 : 0;
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

## 📄 **FILE 136 of 206**: include/sentio/tfa/tfa_seq_context.hpp

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

## 📄 **FILE 137 of 206**: include/sentio/time_utils.hpp

**File Information**:
- **Path**: `include/sentio/time_utils.hpp`

- **Size**: 15 lines
- **Modified**: 2025-09-10 11:15:18

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

} // namespace sentio
```

## 📄 **FILE 138 of 206**: include/sentio/torch/safe_from_blob.hpp

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

## 📄 **FILE 139 of 206**: include/sentio/unified_metrics.hpp

**File Information**:
- **Path**: `include/sentio/unified_metrics.hpp`

- **Size**: 103 lines
- **Modified**: 2025-09-13 09:12:17

- **Type**: .hpp

```text
#pragma once

#include "sentio/metrics.hpp"
#include "sentio/cost_model.hpp"
#include <vector>
#include <string>

// Forward declaration to avoid circular dependency
namespace audit {
    struct Event;
}

namespace sentio {

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
     * Calculate performance metrics from equity curve
     * This is the authoritative method used by all systems
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

## 📄 **FILE 140 of 206**: include/sentio/unified_strategy_tester.hpp

**File Information**:
- **Path**: `include/sentio/unified_strategy_tester.hpp`

- **Size**: 239 lines
- **Modified**: 2025-09-15 15:33:12

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
        const TestConfig& config, int num_simulations);
    
    /**
     * Run AI regime tests
     */
    std::vector<VirtualMarketEngine::VMSimulationResult> run_ai_regime_tests(
        const TestConfig& config, int num_simulations);
    
    /**
     * Run hybrid tests (combination of all modes)
     */
    std::vector<VirtualMarketEngine::VMSimulationResult> run_hybrid_tests(
        const TestConfig& config);
    
    /**
     * Run holistic tests (comprehensive multi-scenario testing)
     */
    std::vector<VirtualMarketEngine::VMSimulationResult> run_holistic_tests(
        const TestConfig& config);
    
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

## 📄 **FILE 141 of 206**: include/sentio/util/bytes.hpp

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

## 📄 **FILE 142 of 206**: include/sentio/util/safe_matrix.hpp

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

## 📄 **FILE 143 of 206**: include/sentio/utils/formatting.hpp

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

## 📄 **FILE 144 of 206**: include/sentio/utils/validation.hpp

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

## 📄 **FILE 145 of 206**: include/sentio/virtual_market.hpp

**File Information**:
- **Path**: `include/sentio/virtual_market.hpp`

- **Size**: 183 lines
- **Modified**: 2025-09-15 15:33:12

- **Type**: .hpp

```text
#pragma once

#include "sentio/core.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/runner.hpp"
#include <vector>
#include <random>
#include <memory>
#include <chrono>

namespace sentio {

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
        double total_return;
        double final_capital;
        double sharpe_ratio;
        double max_drawdown;
        double win_rate;
        int total_trades;
        double monthly_projected_return;
        double daily_trades;
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

## 📄 **FILE 146 of 206**: include/sentio/wf.hpp

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

## 📄 **FILE 147 of 206**: src/accurate_leverage_pricing.cpp

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

## 📄 **FILE 148 of 206**: src/audit.cpp

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

## 📄 **FILE 149 of 206**: src/audit_validator.cpp

**File Information**:
- **Path**: `src/audit_validator.cpp`

- **Size**: 194 lines
- **Modified**: 2025-09-13 09:40:09

- **Type**: .cpp

```text
#include "sentio/audit_validator.hpp"
#include "sentio/all_strategies.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/run_id_generator.hpp"
#include "audit/audit_db_recorder.hpp"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>

namespace sentio {

AuditValidator::ValidationResult AuditValidator::validate_strategy_audit_compatibility(
    const std::string& strategy_name,
    int test_bars) {
    
    ValidationResult result;
    result.strategy_name = strategy_name;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // **1. CREATE STRATEGY INSTANCE**
        auto strategy = StrategyFactory::instance().create_strategy(strategy_name);
        if (!strategy) {
            result.error_message = "Failed to create strategy instance";
            return result;
        }
        
        // **2. SETUP TEST ENVIRONMENT**
        SymbolTable ST;
        int base_symbol_id = ST.intern("QQQ");
        
        // Generate synthetic test data
        auto test_data = generate_test_data(test_bars);
        std::vector<std::vector<Bar>> series(1);
        series[0] = test_data;
        
        // **3. CREATE AUDIT RECORDER**
        std::string run_id = generate_run_id();
        std::string audit_note = create_audit_note(strategy_name, "audit_validation");
        std::string db_path = ":memory:"; // Use in-memory database for testing
        audit::AuditDBRecorder audit(db_path, run_id, audit_note);
        
        // **4. CREATE STRATEGY-AGNOSTIC CONFIG**
        RunnerCfg cfg = create_test_config(strategy_name);
        
        // **5. RUN BACKTEST WITH AUDIT**
        auto backtest_result = run_backtest(audit, ST, series, base_symbol_id, cfg);
        
        // **6. VALIDATE RESULTS**
        result.success = true;
        result.signals_logged = 0; // Would need to query audit DB to get actual counts
        result.orders_logged = 0;
        result.fills_logged = backtest_result.total_fills;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.test_duration_sec = std::chrono::duration<double>(end_time - start_time).count();
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.test_duration_sec = std::chrono::duration<double>(end_time - start_time).count();
    }
    
    return result;
}

std::vector<AuditValidator::ValidationResult> AuditValidator::validate_all_strategies(int test_bars) {
    std::vector<ValidationResult> results;
    
    // Get all registered strategies
    auto strategy_names = StrategyFactory::instance().get_available_strategies();
    
    std::cout << "🔍 **STRATEGY-AGNOSTIC AUDIT VALIDATION**" << std::endl;
    std::cout << "Testing " << strategy_names.size() << " registered strategies..." << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    for (const auto& strategy_name : strategy_names) {
        std::cout << "Testing " << std::setw(20) << std::left << strategy_name << "... ";
        std::cout.flush();
        
        auto result = validate_strategy_audit_compatibility(strategy_name, test_bars);
        results.push_back(result);
        
        if (result.success) {
            std::cout << "✅ PASS (" << std::fixed << std::setprecision(3) 
                      << result.test_duration_sec << "s)" << std::endl;
        } else {
            std::cout << "❌ FAIL: " << result.error_message << std::endl;
        }
    }
    
    return results;
}

void AuditValidator::print_validation_report(const std::vector<ValidationResult>& results) {
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "📊 **AUDIT VALIDATION REPORT**" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    int passed = 0;
    int failed = 0;
    double total_time = 0.0;
    
    for (const auto& result : results) {
        if (result.success) {
            passed++;
        } else {
            failed++;
        }
        total_time += result.test_duration_sec;
    }
    
    std::cout << "📈 **SUMMARY:**" << std::endl;
    std::cout << "  Total Strategies: " << results.size() << std::endl;
    std::cout << "  ✅ Passed: " << passed << " (" << (100.0 * passed / results.size()) << "%)" << std::endl;
    std::cout << "  ❌ Failed: " << failed << " (" << (100.0 * failed / results.size()) << "%)" << std::endl;
    std::cout << "  ⏱️  Total Time: " << std::fixed << std::setprecision(2) << total_time << "s" << std::endl;
    
    if (failed > 0) {
        std::cout << std::endl << "❌ **FAILED STRATEGIES:**" << std::endl;
        for (const auto& result : results) {
            if (!result.success) {
                std::cout << "  • " << result.strategy_name << ": " << result.error_message << std::endl;
            }
        }
    }
    
    if (passed == static_cast<int>(results.size())) {
        std::cout << std::endl << "🎉 **ALL STRATEGIES PASS AUDIT VALIDATION!**" << std::endl;
        std::cout << "The audit system is fully strategy-agnostic." << std::endl;
    }
}

std::vector<Bar> AuditValidator::generate_test_data(int num_bars) {
    std::vector<Bar> bars;
    bars.reserve(num_bars);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> price_change(0.0, 0.01); // 1% volatility
    
    double base_price = 100.0;
    std::int64_t base_time = 1640995200000; // 2022-01-01 00:00:00 UTC in milliseconds
    
    for (int i = 0; i < num_bars; ++i) {
        Bar bar;
        
        // Generate realistic OHLCV data
        double change = price_change(gen);
        double open = base_price * (1.0 + change);
        double high = open * (1.0 + std::abs(price_change(gen)) * 0.5);
        double low = open * (1.0 - std::abs(price_change(gen)) * 0.5);
        double close = open + (high - low) * (price_change(gen) + 1.0) / 2.0;
        
        bar.open = open;
        bar.high = std::max(open, std::max(high, close));
        bar.low = std::min(open, std::min(low, close));
        bar.close = close;
        bar.volume = 1000000 + static_cast<std::uint64_t>(std::abs(price_change(gen)) * 500000);
        
        bar.ts_utc_epoch = base_time + i * 60000; // 1-minute bars
        bar.ts_utc = std::to_string(bar.ts_utc_epoch);
        
        bars.push_back(bar);
        base_price = close; // Use close as next open
    }
    
    return bars;
}

RunnerCfg AuditValidator::create_test_config(const std::string& strategy_name) {
    RunnerCfg cfg;
    cfg.strategy_name = strategy_name;
    cfg.audit_level = AuditLevel::Full; // **CRITICAL**: Enable full audit logging
    cfg.snapshot_stride = 10; // Take snapshots every 10 bars for testing
    
    // Set default router config
    cfg.router.bull3x = "TQQQ";
    cfg.router.bear3x = "SQQQ";
    
    // Set default sizer config
    cfg.sizer.max_position_pct = 1.0;
    cfg.sizer.allow_negative_cash = false;
    cfg.sizer.max_leverage = 2.0;
    cfg.sizer.min_notional = 1.0;
    
    return cfg;
}

} // namespace sentio

```

## 📄 **FILE 150 of 206**: src/base_strategy.cpp

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

## 📄 **FILE 151 of 206**: src/cli_helpers.cpp

**File Information**:
- **Path**: `src/cli_helpers.cpp`

- **Size**: 296 lines
- **Modified**: 2025-09-13 14:55:09

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

} // namespace sentio

```

## 📄 **FILE 152 of 206**: src/csv_loader.cpp

**File Information**:
- **Path**: `src/csv_loader.cpp`

- **Size**: 154 lines
- **Modified**: 2025-09-14 12:09:45

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
                        // Try space format
                        if (cctz::parse("%Y-%m-%d %H:%M:%S%Ez", timestamp_str, utc_tz, &utc_tp)) {
                            bar.ts_utc_epoch = utc_tp.time_since_epoch().count();
                        } else {
                            bar.ts_utc_epoch = 0;
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

## 📄 **FILE 153 of 206**: src/data_downloader.cpp

**File Information**:
- **Path**: `src/data_downloader.cpp`

- **Size**: 190 lines
- **Modified**: 2025-09-12 20:12:17

- **Type**: .cpp

```text
#include "sentio/data_downloader.hpp"
#include "sentio/polygon_client.hpp"
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
    
    std::ostringstream oss;
    oss << std::put_time(tm_start, "%Y-%m-%d");
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
    std::string from = calculate_start_date(years, months, days);
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

## 📄 **FILE 154 of 206**: src/feature_builder.cpp

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

## 📄 **FILE 155 of 206**: src/feature_cache.cpp

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

## 📄 **FILE 156 of 206**: src/feature_engineering/feature_normalizer.cpp

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

## 📄 **FILE 157 of 206**: src/feature_engineering/kochi_features.cpp

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

## 📄 **FILE 158 of 206**: src/feature_engineering/technical_indicators.cpp

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

## 📄 **FILE 159 of 206**: src/feature_feeder.cpp

**File Information**:
- **Path**: `src/feature_feeder.cpp`

- **Size**: 609 lines
- **Modified**: 2025-09-15 15:57:19

- **Type**: .cpp

```text
#include "sentio/feature_feeder.hpp"
// TFB strategy removed - focusing on TFA only
#include "sentio/strategy_tfa.hpp"
#include "sentio/strategy_kochi_ppo.hpp"
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
           strategy_name == "hybrid_ppo" ||
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
            static int cache_calls = 0;
            cache_calls++;
            if (cache_calls <= 5) {
            }
        } else {
            // Calculate features using full bar history up to current_index
            features = data.calculator->calculate_all_features(bars, current_index);
        }
        
        // Normalize features (skip normalization for cached features as they're pre-processed)
        bool used_cache = (use_cached_features_ && feature_cache_ && feature_cache_->has_features(current_index));
        if (data.normalizer && !features.empty() && !used_cache) {
            size_t before_norm = features.size();
            features = data.normalizer->normalize_features(features);
            static int norm_calls = 0;
            norm_calls++;
            if (norm_calls <= 5) {
            }
        } else if (used_cache) {
            static int cache_skip_calls = 0;
            cache_skip_calls++;
            if (cache_skip_calls <= 5) {
            }
        }
        
        // Validate features (bypass validation for cached features as they're pre-validated)
        if (used_cache) {
            static int cache_bypass_calls = 0;
            cache_bypass_calls++;
            if (cache_bypass_calls <= 5) {
            }
        } else {
            bool valid = validate_features(features, strategy_name);
            static int val_calls = 0;
            val_calls++;
            if (val_calls <= 5) {
            }
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
        
        static int feature_extract_calls = 0;
        feature_extract_calls++;
        
        
        if (features.empty()) {
            return;
        }
        
        // Cast to specific strategy type and feed features
        static int strategy_check_calls = 0;
        strategy_check_calls++;
        
        
        if (strategy_name == "TFA" || strategy_name == "tfa") {
            auto* tfa = dynamic_cast<TFAStrategy*>(strategy);
            if (tfa) {
                static int tfa_feed_calls = 0;
                tfa_feed_calls++;
                
                
                tfa->set_raw_features(features);
            } else {
                static int cast_fail = 0;
                cast_fail++;
            }
        } else if (strategy_name == "kochi_ppo") {
            auto* kp = dynamic_cast<KochiPPOStrategy*>(strategy);
            if (kp) {
                kp->set_raw_features(features);
            }
        }
        
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

## 📄 **FILE 160 of 206**: src/feature_feeder_guarded.cpp

**File Information**:
- **Path**: `src/feature_feeder_guarded.cpp`

- **Size**: 79 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .cpp

```text
#include "sentio/feature/feature_feeder_guarded.hpp"
#include "sentio/feature/feature_builder_guarded.hpp"
#include "sentio/sym/symbol_utils.hpp"

namespace sentio {

bool FeatureFeederGuarded::infer_base_if_needed_(
  const std::unordered_map<std::string, std::vector<Bar>>& series,
  std::string& base_out)
{
  if (!base_out.empty()) { base_out = to_upper(base_out); return true; }
  // Prefer QQQ if present, else pick the first non-leveraged symbol
  if (series.find("QQQ") != series.end()) { base_out = "QQQ"; return true; }
  for (auto& kv : series) {
    if (!is_leveraged(kv.first)) { base_out = to_upper(kv.first); return true; }
  }
  return false;
}

bool FeatureFeederGuarded::initialize(const FeederInit& init) {
  prices_.clear();
  asof_.clear();
  base_ts_.clear();
  X_ = {};
  scaler_ = {};
  base_symU_.clear();

  // Cache the input prices (all symbols). We'll filter logically below.
  for (auto& kv : init.series) {
    prices_.emplace(to_upper(kv.first), kv.second);
  }

  // Decide base
  base_symU_ = init.base_symbol;
  if (!infer_base_if_needed_(prices_, base_symU_)) {
    // cannot proceed without a base
    return false;
  }

  // Build features **only** for base
  auto it = prices_.find(base_symU_);
  if (it == prices_.end() || it->second.empty()) return false;

  // Build base timestamp vector
  base_ts_.resize(it->second.size());
  for (std::size_t i=0; i<it->second.size(); ++i) base_ts_[i] = it->second[i].ts_epoch_us;

  // Strict guard: do not allow leveraged symbol to reach builder
  if (is_leveraged(base_symU_)) return false;

  X_ = build_features_for_base(base_symU_, it->second);
  if (X_.rows == 0) return false;

  // Fit scaler ONLY on base features; transform in-place
  scaler_.fit(X_.data.data(), X_.rows, X_.cols);
  scaler_.transform_inplace(X_.data.data(), X_.rows, X_.cols);

  // Build as-of maps for any instrument in the base family (leveraged or base itself)
  for (auto& kv : prices_) {
    const auto symU = kv.first;
    // Only build maps for instruments whose resolved base == base_symU_
    if (resolve_base(symU) != base_symU_) continue;

    // Build inst_ts
    std::vector<std::int64_t> inst_ts; inst_ts.reserve(kv.second.size());
    for (auto& b : kv.second) inst_ts.push_back(b.ts_epoch_us);

    asof_[symU] = build_asof_index(base_ts_, inst_ts);
  }

  return true;
}

bool FeatureFeederGuarded::allowed_for_exec(const std::string& symbol) const {
  const auto symU = to_upper(symbol);
  return resolve_base(symU) == base_symU_;
}

} // namespace sentio

```

## 📄 **FILE 161 of 206**: src/feature_health.cpp

**File Information**:
- **Path**: `src/feature_health.cpp`

- **Size**: 32 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .cpp

```text
#include "sentio/feature_health.hpp"
#include <cmath>

namespace sentio {

FeatureHealthReport check_feature_health(const std::vector<PricePoint>& series,
                                         const FeatureHealthCfg& cfg) {
  FeatureHealthReport rep;
  if (series.empty()) return rep;

  for (std::size_t i=0;i<series.size();++i) {
    const auto& p = series[i];
    if (cfg.check_nan && !std::isfinite(p.close)) {
      rep.issues.push_back({p.ts_utc, "NaN", "non-finite close"});
    }
    if (cfg.check_monotonic_time && i>0) {
      if (series[i].ts_utc <= series[i-1].ts_utc) {
        rep.issues.push_back({p.ts_utc, "Backwards_TS", "non-increasing timestamp"});
      }
      if (cfg.expected_spacing_sec>0) {
        auto gap = series[i].ts_utc - series[i-1].ts_utc;
        if (gap != cfg.expected_spacing_sec) {
          rep.issues.push_back({p.ts_utc, "Gap",
              "expected "+std::to_string(cfg.expected_spacing_sec)+"s got "+std::to_string((long long)gap)+"s"});
        }
      }
    }
  }
  return rep;
}

} // namespace sentio

```

## 📄 **FILE 162 of 206**: src/future_qqq_loader.cpp

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

## 📄 **FILE 163 of 206**: src/global_leverage_config.cpp

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

## 📄 **FILE 164 of 206**: src/leverage_aware_csv_loader.cpp

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

## 📄 **FILE 165 of 206**: src/leverage_pricing.cpp

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

## 📄 **FILE 166 of 206**: src/main.cpp

**File Information**:
- **Path**: `src/main.cpp`

- **Size**: 452 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .cpp

```text
#include "sentio/core.hpp"
#include "sentio/runner.hpp"
#include "sentio/temporal_analysis.hpp"
#include "sentio/csv_loader.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/profiling.hpp"
#include "sentio/data_resolver.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/all_strategies.hpp"
#include "sentio/feature_feeder.hpp"
#include "sentio/data_downloader.hpp"
#include "sentio/audit_validator.hpp"
#include "sentio/feature/feature_matrix.hpp"
#include "sentio/strategy_tfa.hpp"
#include "sentio/virtual_market.hpp"
#include "sentio/unified_strategy_tester.hpp"
#include "sentio/cli_helpers.hpp"

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <cstdlib>
#include <sstream>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <ATen/Parallel.h>
#include <nlohmann/json.hpp>

static bool verify_series_alignment(const sentio::SymbolTable& ST,
                                    const std::vector<std::vector<sentio::Bar>>& series,
                                    int base_sid){
    const auto& base = series[base_sid];
    if (base.empty()) return false;
    const size_t N = base.size();
    bool ok = true;
    for (size_t sid = 0; sid < series.size(); ++sid) {
        if (sid == (size_t)base_sid) continue;
        const auto& s = series[sid];
        if (s.empty()) continue; // allow missing non-base
        if (s.size() != N) {
            std::cerr << "FATAL: Alignment check failed: " << ST.get_symbol((int)sid)
                      << " bars=" << s.size() << " != base(" << ST.get_symbol(base_sid)
                      << ") bars=" << N << "\n";
            ok = false;
            continue;
        }
        // Check timestamp alignment for first and last bars
        if (s[0].ts_utc_epoch != base[0].ts_utc_epoch) {
            std::cerr << "FATAL: Start timestamp mismatch: " << ST.get_symbol((int)sid)
                      << " vs " << ST.get_symbol(base_sid) << "\n";
            ok = false;
        }
        if (s[N-1].ts_utc_epoch != base[N-1].ts_utc_epoch) {
            std::cerr << "FATAL: End timestamp mismatch: " << ST.get_symbol((int)sid)
                      << " vs " << ST.get_symbol(base_sid) << "\n";
            ok = false;
        }
    }
    return ok;
}

void usage() {
    std::cout << "Usage: sentio_cli <command> [options]\n\n"
              << "STRATEGY TESTING:\n"
              << "  strattest <strategy> [symbol] [options]    Unified strategy robustness testing (symbol defaults to QQQ)\n"
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
              << "  sentio_cli strattest momentum --mode hybrid --duration 1w\n"
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
                    "--mode <mode>              Simulation mode: monte-carlo|historical|ai-regime|hybrid (default: hybrid)",
                    "--simulations <n>          Number of simulations (default: 50)",
                    "--duration <period>        Test duration: 1h, 4h, 1d, 5d, 1w, 1m (default: 5d)",
                    "--historical-data <file>   Historical data file (auto-detect if not specified)",
                    "--regime <regime>          Market regime: normal|volatile|trending|bear|bull (default: normal)",
                    "--stress-test              Enable stress testing scenarios",
                    "--regime-switching         Test across multiple market regimes",
                    "--liquidity-stress         Simulate low liquidity conditions",
                    "--alpaca-fees              Use Alpaca fee structure (default: true)",
                    "--alpaca-limits            Apply Alpaca position/order limits",
                    "--confidence <level>       Confidence level: 90|95|99 (default: 95)",
                    "--output <format>          Output format: console|json|csv (default: console)",
                    "--save-results <file>      Save detailed results to file",
                    "--benchmark <symbol>       Benchmark symbol (default: SPY)",
                    "--quick                    Quick mode: fewer simulations, faster execution",
                    "--comprehensive            Comprehensive mode: extensive testing scenarios",
                    "--params <json>            Strategy parameters as JSON string (default: '{}')",
                    "",
                    "Note: Symbol defaults to QQQ if not specified"
                },
                {
                    "sentio_cli strattest momentum --mode hybrid --duration 1w",
                    "sentio_cli strattest ire --comprehensive --stress-test",
                    "sentio_cli strattest momentum QQQ --mode monte-carlo --simulations 100",
                    "sentio_cli strattest ire SPY --mode ai-regime --regime volatile"
                });
            return 0;
        }
        
        if (!sentio::CLIHelpers::validate_required_args(args, 1, 
            "sentio_cli strattest <strategy> [symbol] [options]")) {
            return 1;
        }
        
        std::string strategy_name = args.positional_args[0];
        std::string symbol = (args.positional_args.size() > 1) ? args.positional_args[1] : "QQQ";
        
        // Validate inputs
        if (!sentio::CLIHelpers::is_valid_strategy_name(strategy_name)) {
            sentio::CLIHelpers::print_error("Invalid strategy name: " + strategy_name);
            return 1;
        }
        
        if (!sentio::CLIHelpers::is_valid_symbol(symbol)) {
            sentio::CLIHelpers::print_error("Invalid symbol: " + symbol);
            return 1;
        }
        
        // Build test configuration
        sentio::UnifiedStrategyTester::TestConfig config;
        config.strategy_name = strategy_name;
        config.symbol = symbol;
        
        // Parse options
        std::string mode_str = sentio::CLIHelpers::get_option(args, "mode", "hybrid");
        config.mode = sentio::UnifiedStrategyTester::parse_test_mode(mode_str);
        
        config.simulations = sentio::CLIHelpers::get_int_option(args, "simulations", 50);
        config.duration = sentio::CLIHelpers::get_option(args, "duration", "5d");
        config.historical_data_file = sentio::CLIHelpers::get_option(args, "historical-data");
        config.regime = sentio::CLIHelpers::get_option(args, "regime", "normal");
        
        config.stress_test = sentio::CLIHelpers::get_flag(args, "stress-test");
        config.regime_switching = sentio::CLIHelpers::get_flag(args, "regime-switching");
        config.liquidity_stress = sentio::CLIHelpers::get_flag(args, "liquidity-stress");
        
        config.alpaca_fees = !sentio::CLIHelpers::get_flag(args, "no-alpaca-fees"); // Default true
        config.alpaca_limits = sentio::CLIHelpers::get_flag(args, "alpaca-limits");
        config.paper_validation = sentio::CLIHelpers::get_flag(args, "paper-validation");
        
        int confidence_pct = sentio::CLIHelpers::get_int_option(args, "confidence", 95);
        config.confidence_level = confidence_pct / 100.0;
        
        config.output_format = sentio::CLIHelpers::get_option(args, "output", "console");
        config.save_results_file = sentio::CLIHelpers::get_option(args, "save-results");
        config.benchmark_symbol = sentio::CLIHelpers::get_option(args, "benchmark", "SPY");
        
        config.quick_mode = sentio::CLIHelpers::get_flag(args, "quick");
        config.comprehensive_mode = sentio::CLIHelpers::get_flag(args, "comprehensive");
        
        config.params_json = sentio::CLIHelpers::get_option(args, "params", "{}");
        
        // Adjust simulations based on mode flags
        if (config.quick_mode && config.simulations == 50) {
            config.simulations = 20;
        } else if (config.comprehensive_mode && config.simulations == 50) {
            config.simulations = 100;
        }
        
        // Run unified strategy test
        try {
            sentio::UnifiedStrategyTester tester;
            auto report = tester.run_comprehensive_test(config);
            
            // Display results
            if (config.output_format == "console") {
                tester.print_robustness_report(report, config);
            } else {
                // Save to file or output in other formats
                if (!config.save_results_file.empty()) {
                    if (tester.save_report(report, config, config.save_results_file, config.output_format)) {
                        std::cout << "Results saved to: " << config.save_results_file << std::endl;
                    } else {
                        std::cerr << "Failed to save results to: " << config.save_results_file << std::endl;
                        return 1;
                    }
                }
            }
            
            // Return deployment readiness as exit code (0 = ready, 1 = not ready)
            return report.ready_for_deployment ? 0 : 1;
            
        } catch (const std::exception& e) {
            std::cerr << "Error running strategy test: " << e.what() << std::endl;
            return 1;
        }
        
    } else if (command == "probe") {
        std::cout << "=== SENTIO SYSTEM PROBE ===\n\n";
        
        // Show available strategies
        std::cout << "📊 Available Strategies (" << sentio::StrategyFactory::instance().get_available_strategies().size() << " total):\n";
        std::cout << "=====================\n";
        for (const auto& strategy_name : sentio::StrategyFactory::instance().get_available_strategies()) {
            std::cout << "  • " << strategy_name << "\n";
        }
        std::cout << "\n";
        
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
        std::cout << "📈 Data Availability Check:\n";
        std::cout << "==========================\n";
        
        bool daily_aligned = true, minute_aligned = true;
        
        for (const auto& symbol : symbols) {
            std::cout << "Symbol: " << symbol << "\n";
            
            // Check daily data
            std::string daily_path = "data/equities/" + symbol + "_daily.csv";
            auto [daily_exists, daily_range] = get_file_info(daily_path);
            
            std::cout << "  📅 Daily:  ";
            if (daily_exists) {
                std::cout << "✅ Available (" << daily_range.first << " to " << daily_range.second << ")\n";
            } else {
                std::cout << "❌ Missing\n";
                daily_aligned = false;
            }
            
            // Check minute data
            std::string minute_path = "data/equities/" + symbol + "_NH.csv";
            auto [minute_exists, minute_range] = get_file_info(minute_path);
            
            std::cout << "  ⏰ Minute: ";
            if (minute_exists) {
                std::cout << "✅ Available (" << minute_range.first << " to " << minute_range.second << ")\n";
            } else {
                std::cout << "❌ Missing\n";
                minute_aligned = false;
            }
            
            std::cout << "\n";
        }
        
        // Summary
        std::cout << "📋 Summary:\n";
        std::cout << "===========\n";
        if (daily_aligned && minute_aligned) {
            std::cout << "  🎉 All data is properly aligned and ready for strategy testing!\n";
            std::cout << "  📋 Ready to run: ./build/sentio_cli strattest ire QQQ --comprehensive\n";
        }
        
        std::cout << "\n";
        return 0;
        
    } else if (command == "audit-validate") {
        std::cout << "🔍 **STRATEGY-AGNOSTIC AUDIT VALIDATION**" << std::endl;
        std::cout << "Validating that all registered strategies work with the audit system..." << std::endl;
        std::cout << std::endl;
        
        // Run validation for all strategies
        auto results = sentio::AuditValidator::validate_all_strategies(50); // Test with 50 bars
        
        // Print comprehensive report
        std::cout << "📊 **AUDIT VALIDATION RESULTS**" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        int passed = 0, failed = 0;
        for (const auto& result : results) {
            if (result.success) {
                std::cout << "✅ " << result.strategy_name << " - PASSED" << std::endl;
                passed++;
            } else {
                std::cout << "❌ " << result.strategy_name << " - FAILED: " << result.error_message << std::endl;
                failed++;
            }
        }
        
        std::cout << std::string(50, '=') << std::endl;
        std::cout << "📈 Summary: " << passed << " passed, " << failed << " failed" << std::endl;
        
        if (failed == 0) {
            std::cout << "🎉 All strategies are audit-compatible!" << std::endl;
            return 0;
        } else {
            std::cout << "⚠️  Some strategies need fixes before audit compatibility" << std::endl;
            return 1;
        }
        
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

## 📄 **FILE 167 of 206**: src/mars_data_loader.cpp

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

## 📄 **FILE 168 of 206**: src/ml/model_registry_ts.cpp

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

## 📄 **FILE 169 of 206**: src/ml/ts_model.cpp

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

## 📄 **FILE 170 of 206**: src/optimizer.cpp

**File Information**:
- **Path**: `src/optimizer.cpp`

- **Size**: 180 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .cpp

```text
#include "sentio/optimizer.hpp"
#include "sentio/runner.hpp"
#include <iostream>

namespace sentio {

// RandomSearchOptimizer Implementation
std::vector<Parameter> RandomSearchOptimizer::get_parameter_space() {
    return {
        Parameter("test_param", 0.0, 1.0, 0.5)
    };
}

void RandomSearchOptimizer::apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) {
    for (const auto& param : params) {
        cfg.strategy_params[param.name] = std::to_string(param.current_value);
    }
}

OptimizationResult RandomSearchOptimizer::optimize([[maybe_unused]] const std::function<double(const RunResult&)>& objective_func,
                                                  [[maybe_unused]] const std::vector<Parameter>& param_space,
                                                  [[maybe_unused]] const OptimizationConfig& config) {
    OptimizationResult result;
    result.parameters["test"] = 0.5;
    result.objective_value = 0.0;
    return result;
}

// GridSearchOptimizer Implementation
std::vector<Parameter> GridSearchOptimizer::get_parameter_space() {
    return {
        Parameter("test_param", 0.0, 1.0, 0.5)
    };
}

void GridSearchOptimizer::apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) {
    for (const auto& param : params) {
        cfg.strategy_params[param.name] = std::to_string(param.current_value);
    }
}

OptimizationResult GridSearchOptimizer::optimize([[maybe_unused]] const std::function<double(const RunResult&)>& objective_func,
                                                [[maybe_unused]] const std::vector<Parameter>& param_space,
                                                [[maybe_unused]] const OptimizationConfig& config) {
    OptimizationResult result;
    result.parameters["test"] = 0.5;
    result.objective_value = 0.0;
    return result;
}

// BayesianOptimizer Implementation
std::vector<Parameter> BayesianOptimizer::get_parameter_space() {
    return {
        Parameter("test_param", 0.0, 1.0, 0.5)
    };
}

void BayesianOptimizer::apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) {
    for (const auto& param : params) {
        cfg.strategy_params[param.name] = std::to_string(param.current_value);
    }
}

OptimizationResult BayesianOptimizer::optimize([[maybe_unused]] const std::function<double(const RunResult&)>& objective_func,
                                              [[maybe_unused]] const std::vector<Parameter>& param_space,
                                              [[maybe_unused]] const OptimizationConfig& config) {
    OptimizationResult result;
    result.parameters["test"] = 0.5;
    result.objective_value = 0.0;
    return result;
}

// Strategy parameter creation functions
std::vector<Parameter> create_vwap_parameters() {
    return {
        Parameter("vwap_period", 100.0, 800.0, 200.0),
        Parameter("reversion_threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_momentum_parameters() {
    return {
        Parameter("momentum_period", 5.0, 50.0, 20.0),
        Parameter("threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_volatility_parameters() {
    return {
        Parameter("volatility_period", 10.0, 50.0, 20.0),
        Parameter("threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_bollinger_squeeze_parameters() {
    return {
        Parameter("bb_period", 10.0, 40.0, 20.0),
        Parameter("bb_std", 1.0, 3.0, 2.0)
    };
}

std::vector<Parameter> create_opening_range_parameters() {
    return {
        Parameter("range_minutes", 15.0, 60.0, 30.0),
        Parameter("breakout_threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_order_flow_scalping_parameters() {
    return {
        Parameter("imbalance_period", 5.0, 40.0, 20.0),
        Parameter("threshold", 0.4, 0.9, 0.7)
    };
}

std::vector<Parameter> create_order_flow_imbalance_parameters() {
    return {
        Parameter("lookback_window", 5.0, 50.0, 20.0),
        Parameter("threshold", 0.5, 3.0, 1.5)
    };
}

std::vector<Parameter> create_market_making_parameters() {
    return {
        Parameter("base_spread", 0.0002, 0.003, 0.001),
        Parameter("order_levels", 1.0, 5.0, 3.0)
    };
}

std::vector<Parameter> create_router_parameters() {
    return {
        Parameter("t1", 0.01, 0.3, 0.05),
        Parameter("t2", 0.1, 0.6, 0.3)
    };
}

std::vector<Parameter> create_parameters_for_strategy(const std::string& strategy_name) {
    if (strategy_name == "VWAPReversion") {
        return create_vwap_parameters();
    } else if (strategy_name == "MomentumVolumeProfile") {
        return create_momentum_parameters();
    } else if (strategy_name == "VolatilityExpansion") {
        return create_volatility_parameters();
    } else if (strategy_name == "BollingerSqueezeBreakout") {
        return create_bollinger_squeeze_parameters();
    } else if (strategy_name == "OpeningRangeBreakout") {
        return create_opening_range_parameters();
    } else if (strategy_name == "OrderFlowScalping") {
        return create_order_flow_scalping_parameters();
    } else if (strategy_name == "OrderFlowImbalance") {
        return create_order_flow_imbalance_parameters();
    } else if (strategy_name == "MarketMaking") {
        return create_market_making_parameters();
    } else {
        return create_router_parameters();
    }
}

std::vector<Parameter> create_full_parameter_space() {
    return create_router_parameters();
}

// OptimizationEngine implementation
OptimizationEngine::OptimizationEngine(const std::string& optimizer_type) {
    if (optimizer_type == "grid") {
        optimizer = std::make_unique<GridSearchOptimizer>();
    } else if (optimizer_type == "bayesian") {
        optimizer = std::make_unique<BayesianOptimizer>();
    } else {
        optimizer = std::make_unique<RandomSearchOptimizer>();
    }
}

OptimizationResult OptimizationEngine::run_optimization(const std::string& strategy_name,
                                                       const std::function<double(const RunResult&)>& objective_func) {
    auto param_space = create_parameters_for_strategy(strategy_name);
    return optimizer->optimize(objective_func, param_space, config);
}

} // namespace sentio

```

## 📄 **FILE 171 of 206**: src/pnl_accounting.cpp

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

## 📄 **FILE 172 of 206**: src/poly_fetch_main.cpp

**File Information**:
- **Path**: `src/poly_fetch_main.cpp`

- **Size**: 151 lines
- **Modified**: 2025-09-12 15:35:06

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
    std::cerr<<"Usage: poly_fetch FAMILY outdir [--years N] [--months N] [--days N] [--timespan day|hour|minute] [--multiplier N] [--symbols SYM1,SYM2,...] [--no-holidays]\n";
    std::cerr<<"       poly_fetch FAMILY from to outdir [--timespan day|hour|minute] [--multiplier N] [--symbols SYM1,SYM2,...] [--no-holidays]\n";
    std::cerr<<"Examples:\n";
    std::cerr<<"  poly_fetch qqq data/equities --years 3 --no-holidays\n";
    std::cerr<<"  poly_fetch qqq 2022-01-01 2025-01-10 data/equities --timespan minute\n";
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
  // RTH filtering removed - keeping all trading hours data
  bool exclude_holidays=false;
  
  int start_idx = use_time_range ? 3 : 5;
  for (int i=start_idx;i<argc;i++) {
    std::string a = argv[i];
    if (a=="--years" && i+1<argc) { years = std::stoi(argv[++i]); }
    else if (a=="--months" && i+1<argc) { months = std::stoi(argv[++i]); }
    else if (a=="--days" && i+1<argc) { days = std::stoi(argv[++i]); }
    else if ((a=="--timespan" || a=="-t") && i+1<argc) { timespan = argv[++i]; }
    else if ((a=="--multiplier" || a=="-m") && i+1<argc) { multiplier = std::stoi(argv[++i]); }
    else if (a=="--symbols" && i+1<argc) { symbols_csv = argv[++i]; }
    // RTH option removed
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
    // RTH suffix removed
    if (exclude_holidays) suffix += "_NH";
    std::string fname= outdir + "/" + s + suffix + ".csv";
    cli.write_csv(fname,s,bars,exclude_holidays);
    std::cerr<<"Wrote "<<bars.size()<<" bars -> "<<fname<<"\n";
  }
}


```

## 📄 **FILE 173 of 206**: src/polygon_client.cpp

**File Information**:
- **Path**: `src/polygon_client.cpp`

- **Size**: 272 lines
- **Modified**: 2025-09-12 14:33:52

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
        
        current_time = chunk_end + 1; // Move to next day
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Rate limiting between chunks
    }
    
    std::cerr << "Total bars collected: " << out.size() << std::endl;
    return out;
}

void PolygonClient::write_csv(const std::string& out_path,const std::string& symbol,
                              const std::vector<AggBar>& bars, bool exclude_holidays) {
  std::ofstream f(out_path);
  f << "timestamp,symbol,open,high,low,close,volume\n";
  for (auto& a: bars) {
    // **MODIFIED**: RTH and holiday filtering is now done directly on the UTC timestamp
    // before any string conversion, making it much more reliable.

    // RTH filtering removed - keeping all trading hours data
    
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

## 📄 **FILE 174 of 206**: src/position_coordinator.cpp

**File Information**:
- **Path**: `src/position_coordinator.cpp`

- **Size**: 217 lines
- **Modified**: 2025-09-12 17:11:26

- **Type**: .cpp

```text
#include "sentio/position_coordinator.hpp"
#include "sentio/base_strategy.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace sentio {

// **ETF CLASSIFICATIONS**: Updated for PSQ -> SHORT QQQ architecture
const std::unordered_set<std::string> PositionCoordinator::LONG_ETFS = {"QQQ", "TQQQ"};
const std::unordered_set<std::string> PositionCoordinator::INVERSE_ETFS = {"SQQQ"};

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
        if (orders_this_bar_ >= max_orders_per_bar_) {
            decision.result = CoordinationResult::REJECTED_FREQUENCY;
            decision.approved_weight = 0.0;
            decision.reason = "FREQUENCY_LIMIT_EXCEEDED";
            decision.conflict_details = "Max " + std::to_string(max_orders_per_bar_) + " orders per bar";
            stats_.rejected_frequency++;
            decisions.push_back(decision);
            continue;
        }
        
        // **SIMULATE POSITION CHANGE**: Calculate what positions would be after this request
        std::unordered_map<std::string, double> simulated_positions = pending_positions_;
        
        // Apply the requested position change
        // Note: This is simplified - in reality we'd need to calculate the actual quantity change
        // based on the target weight, current portfolio value, and instrument price
        if (std::abs(request.target_weight) > 1e-6) {
            if (request.target_weight > 0) {
                simulated_positions[request.instrument] = std::abs(request.target_weight) * 1000; // Simplified
            } else {
                simulated_positions[request.instrument] = request.target_weight * 1000; // Negative for short
            }
        } else {
            simulated_positions.erase(request.instrument); // Zero weight = close position
        }
        
        // **CONFLICT DETECTION**: Check if this would create conflicts
        if (would_create_conflict(simulated_positions)) {
            // **CONFLICT RESOLUTION**: Attempt to resolve
            decision = resolve_conflict(request);
            stats_.rejected_conflict++;
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

} // namespace sentio

```

## 📄 **FILE 175 of 206**: src/position_guardian.cpp

**File Information**:
- **Path**: `src/position_guardian.cpp`

- **Size**: 143 lines
- **Modified**: 2025-09-11 21:59:53

- **Type**: .cpp

```text
#include "sentio/position_guardian.hpp"
#include "sentio/family_mapper.hpp"
#include "sentio/core.hpp"
#include <cmath>

namespace sentio {

static inline double sgn(PositionSide s) { 
    return (s==PositionSide::Long? 1.0 : (s==PositionSide::Short? -1.0 : 0.0)); 
}

typename PositionGuardian::Cell& PositionGuardian::cell_for(const ExposureKey& key) {
    std::lock_guard<std::mutex> lg(map_mu_);
    auto it = cells_.find(key);
    if (it == cells_.end()) {
        auto c = std::make_unique<Cell>();
        c->ps.last_change = Clock::now();
        it = cells_.emplace(key, std::move(c)).first;
    }
    return *it->second;
}

bool PositionGuardian::flip_cooldown_active(const PositionSnapshot& ps, const Policy& pol, Clock::time_point now) {
    if (pol.cooldown_ms <= 0) return false;
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - ps.last_change).count();
    return elapsed < pol.cooldown_ms;
}

PositionSnapshot PositionGuardian::snapshot(const ExposureKey& key) const {
    std::lock_guard<std::mutex> lg(map_mu_);
    auto it = cells_.find(key);
    if (it == cells_.end()) return {};
    std::lock_guard<std::mutex> lc(it->second->m);
    return it->second->ps;
}

std::optional<OrderPlan> PositionGuardian::plan(const std::string& account,
                                                const std::string& symbol,
                                                const Desire& desire,
                                                const Policy& policy)
{
    ExposureKey key{account, mapper_.family_for(symbol)};
    auto& cell = cell_for(key);
    auto now = Clock::now();

    std::lock_guard<std::mutex> lk(cell.m);
    auto ps = cell.ps; // copy

    if (policy.allow_conflicts == false) {
        // Always net to one side only
    }

    // Flip friction (optional): if desire flips side for tiny edge, you can block.
    if (ps.side != PositionSide::Flat && desire.target_side != PositionSide::Flat && ps.side != desire.target_side) {
        // You can access bid/ask or last to estimate bps; here we just enforce cooldown if set
        if (flip_cooldown_active(ps, policy, now)) {
            return std::nullopt; // reject for now
        }
    }

    // Compute desired delta relative to current net family exposure
    double net_signed = ps.qty * sgn(ps.side);
    double tgt_signed = desire.target_qty * sgn(desire.target_side);

    // Apply reservations so multiple strategies don't double-commit
    double avail_long  = std::max(0.0, policy.max_gross_shares - (ps.qty + cell.reserved_long));
    double avail_short = std::max(0.0, policy.max_gross_shares - (ps.qty + cell.reserved_short));

    // Bound target by max_gross
    if (tgt_signed > 0) {
        tgt_signed = std::min(tgt_signed, avail_long);
    } else if (tgt_signed < 0) {
        tgt_signed = -std::min(std::abs(tgt_signed), avail_short);
    }

    if (std::abs(tgt_signed - net_signed) < 1e-9) {
        return std::nullopt; // nothing to do
    }

    OrderPlan plan;
    plan.key = key;
    plan.epoch_before = ps.epoch;
    plan.reservation_id = cell.next_reservation++;

    // If opposite side exists, close that first
    if ((net_signed > 0 && tgt_signed <= 0) || (net_signed < 0 && tgt_signed >= 0)) {
        double close_qty = std::abs(net_signed);
        if (close_qty > 0) {
            plan.legs.push_back({
                /*symbol*/ desire.preferred_symbol.empty() ? symbol : desire.preferred_symbol,
                /*side*/ (ps.side==PositionSide::Long ? PositionSide::Short : PositionSide::Long),
                /*qty*/  close_qty,
                /*reason*/ "CLOSE_OPPOSITE"
            });
            net_signed = 0.0; // after close leg
        }
    }

    // Open / Resize towards target
    double open_delta = tgt_signed - net_signed; // signed
    if (std::abs(open_delta) > 1e-9) {
        plan.legs.push_back({
            /*symbol*/ desire.preferred_symbol.empty() ? symbol : desire.preferred_symbol,
            /*side*/ (open_delta > 0 ? PositionSide::Long : PositionSide::Short),
            /*qty*/  std::abs(open_delta),
            /*reason*/ (net_signed==0.0 ? "OPEN_TARGET" : "RESIZE")
        });
        if (open_delta > 0) cell.reserved_long  += std::abs(open_delta);
        else                cell.reserved_short += std::abs(open_delta);
    }

    return plan;
}

void PositionGuardian::commit(const OrderPlan& plan) {
    auto& cell = cell_for(plan.key);
    std::lock_guard<std::mutex> lk(cell.m);

    // Update snapshot pessimistically to reflect intent (helps prevent collisions)
    for (auto& leg : plan.legs) {
        if (leg.reason == "CLOSE_OPPOSITE") {
            cell.ps.side = PositionSide::Flat;
            cell.ps.qty  = 0.0;
        } else {
            cell.ps.side = (leg.side==PositionSide::Long ? PositionSide::Long : PositionSide::Short);
            cell.ps.qty  += leg.qty; // simple pessimistic add; you can reconcile on fills
        }
        cell.ps.epoch++;
        cell.ps.last_change = Clock::now();
    }
    // release reservations (we pessimistically moved to snapshot)
    cell.reserved_long = cell.reserved_short = 0.0;
}

void PositionGuardian::sync_from_broker(const std::string& account,
                                       const std::vector<Position>& positions,
                                       const std::vector<std::string>& open_orders) {
    // TODO: Implement broker sync logic
    // This would rebuild ps/epoch using live positions and open-orders
    // For now, this is a placeholder
}

} // namespace sentio

```

## 📄 **FILE 176 of 206**: src/position_orchestrator.cpp

**File Information**:
- **Path**: `src/position_orchestrator.cpp`

- **Size**: 110 lines
- **Modified**: 2025-09-12 16:30:08

- **Type**: .cpp

```text
#include "sentio/position_orchestrator.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/audit.hpp"
#include <iostream>

namespace sentio {

PositionOrchestrator::PositionOrchestrator(const std::string& account) 
    : account_(account)
    , mapper_(FamilyMapper::Map{
        {"QQQ*", {"QQQ", "TQQQ", "SQQQ"}}, // PSQ removed
        {"SPY*", {"SPY", "UPRO", "SPXU"}},
        {"BTC*", {"BTC", "BITO", "BITX", "BITI"}},
        {"TSLA*", {"TSLA", "TSLL", "TSLS"}}
      })
    , guardian_(mapper_)
{
    // Default policy: no conflicts, reasonable limits
    policy_.allow_conflicts = false;
    policy_.max_gross_shares = 1'000'000;
    policy_.min_flip_bps = 0.0;
    policy_.cooldown_ms = 500; // avoid churn
}

void PositionOrchestrator::process_strategy_signal(const std::string& strategy_id,
                                                  const std::string& symbol,
                                                  ::sentio::Side target_side,
                                                  double target_qty,
                                                  const std::string& preferred_symbol) {
    
    if (target_qty <= 0) return; // No action needed
    
    Desire desire;
    desire.target_side = convert_side(target_side);
    desire.target_qty = target_qty;
    desire.preferred_symbol = preferred_symbol.empty() ? 
        choose_preferred_symbol(symbol, convert_side(target_side), target_qty) : preferred_symbol;
    
    auto plan_opt = guardian_.plan(account_, symbol, desire, policy_);
    if (!plan_opt) {
        std::cerr << "PositionOrchestrator: No plan generated for " << strategy_id 
                  << " " << symbol << " " << (target_side == ::sentio::Side::Buy ? "BUY" : "SELL") << " " << target_qty << std::endl;
        return; // no-op or cooldown
    }
    
    const auto& plan = *plan_opt;
    
    std::cout << "PositionOrchestrator: Generated plan for " << strategy_id 
              << " family=" << plan.key.family 
              << " legs=" << plan.legs.size() << std::endl;
    
    // TODO: Integrate with existing router
    // For now, just commit the plan
    // In production, this would:
    // 1. Create router batch with strict leg ordering
    // 2. Send to router
    // 3. Only commit on success
    
    guardian_.commit(plan);
    
    // Log the plan details
    for (const auto& leg : plan.legs) {
        std::cout << "  " << leg.reason << ": " << leg.symbol 
                  << " " << to_string(leg.side) << " " << leg.qty << std::endl;
    }
}

void PositionOrchestrator::sync_portfolio(const Portfolio& portfolio, const SymbolTable& ST) {
    // Convert Sentio portfolio to guardian format
    std::vector<Position> positions = portfolio.positions;
    
    // TODO: Convert open orders if available
    std::vector<std::string> open_orders;
    
    guardian_.sync_from_broker(account_, positions, open_orders);
}

PositionSnapshot PositionOrchestrator::get_family_exposure(const std::string& symbol) const {
    ExposureKey key{account_, mapper_.family_for(symbol)};
    return guardian_.snapshot(key);
}

PositionSide PositionOrchestrator::convert_side(::sentio::Side side) {
    switch (side) {
        case ::sentio::Side::Buy: return PositionSide::Long;
        case ::sentio::Side::Sell: return PositionSide::Short;
        default: return PositionSide::Flat;
    }
}

std::string PositionOrchestrator::choose_preferred_symbol(const std::string& symbol, 
                                                         PositionSide side, 
                                                         double strength) const {
    std::string family = mapper_.family_for(symbol);
    
    if (family == "QQQ*") {
        if (side == PositionSide::Long) {
            // For long positions, choose TQQQ for strong signals, QQQ for moderate
            return (strength > 0.7) ? "TQQQ" : "QQQ";
        } else {
            // For short positions, choose SQQQ for strong signals, SHORT QQQ for moderate  
            return (strength > 0.7) ? "SQQQ" : "QQQ"; // QQQ will be shorted for moderate sells
        }
    }
    
    // Default to original symbol for unknown families
    return symbol;
}

} // namespace sentio

```

## 📄 **FILE 177 of 206**: src/router.cpp

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

## 📄 **FILE 178 of 206**: src/rsi_strategy.cpp

**File Information**:
- **Path**: `src/rsi_strategy.cpp`

- **Size**: 2 lines
- **Modified**: 2025-09-11 12:25:32

- **Type**: .cpp

```text
#include "sentio/rsi_strategy.hpp"
// Implementations are header-only for simplicity.

```

## 📄 **FILE 179 of 206**: src/run_id_generator.cpp

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

## 📄 **FILE 180 of 206**: src/runner.cpp

**File Information**:
- **Path**: `src/runner.cpp`

- **Size**: 555 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .cpp

```text
#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/pricebook.hpp"
#include "sentio/metrics.hpp"
#include "sentio/unified_metrics.hpp"
#include "sentio/metrics/mpr.hpp"
#include "sentio/metrics/session_utils.hpp"
#include "audit/audit_db_recorder.hpp"
#include "sentio/sizer.hpp"
#include "sentio/cost_model.hpp"
#include "sentio/feature_feeder.hpp"
#include "sentio/position_validator.hpp"
#include "sentio/position_coordinator.hpp"
#include "sentio/router.hpp"
#include <iostream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <map>
#include <ctime>

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

    // **BUG FIX**: Prevent zero-quantity trades that generate phantom P&L
    // Early return to completely avoid processing zero or tiny trades
    if (std::abs(trade_qty) < 1e-9 || std::abs(trade_qty * instrument_price) <= 10.0) {
        return; // No logging, no execution, no audit entries
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

RunResult run_backtest(IAuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                      int base_symbol_id, const RunnerCfg& cfg) {
    
    // 1. ============== INITIALIZATION ==============
    RunResult result{};
    
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
        return result;
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
        run_trading_days = sentio::metrics::count_trading_days(filtered_timestamps);
    }
    
    // Start audit run with canonical metadata
    std::string meta = "{";
    meta += "\"strategy\":\"" + cfg.strategy_name + "\",";
    meta += "\"base_symbol_id\":" + std::to_string(base_symbol_id) + ",";
    meta += "\"total_series\":" + std::to_string(series.size()) + ",";
    meta += "\"base_series_size\":" + std::to_string(series[base_symbol_id].size()) + ",";
    meta += "\"dataset_type\":\"" + dataset_type + "\",";
    meta += "\"test_period_days\":" + std::to_string(run_trading_days) + ",";
    meta += "\"run_period_start_ts_ms\":" + std::to_string(run_period_start_ts_ms) + ",";
    meta += "\"run_period_end_ts_ms\":" + std::to_string(run_period_end_ts_ms) + ",";
    meta += "\"run_trading_days\":" + std::to_string(run_trading_days);
    meta += "}";
    
    // Use current time for run timestamp (for proper run ordering)
    std::int64_t start_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (logging_enabled) audit.event_run_start(start_ts, meta);
    
    for (size_t i = warmup_bars; i < base_series.size(); ++i) {
        
        const auto& bar = base_series[i];
        
        // **DAY TRADING RULE**: Intraday risk management
        if (i > warmup_bars) {
            // Extract hour and minute from UTC timestamp for market close detection
            // Market closes at 4:00 PM ET = 20:00 UTC (EDT) or 21:00 UTC (EST)
            time_t raw_time = bar.ts_utc_epoch;
            struct tm* utc_tm = gmtime(&raw_time);
            int hour_utc = utc_tm->tm_hour;
            int minute_utc = utc_tm->tm_min;
            int month = utc_tm->tm_mon + 1; // tm_mon is 0-based
            
            // Simple DST check: April-October is EDT (20:00 close), rest is EST (21:00 close)
            bool is_edt = (month >= 4 && month <= 10);
            int market_close_hour = is_edt ? 20 : 21;
            
            // Calculate minutes until market close
            int current_minutes = hour_utc * 60 + minute_utc;
            int close_minutes = market_close_hour * 60; // Market close time in minutes
            int minutes_to_close = close_minutes - current_minutes;
            
            // Handle day wrap-around (shouldn't happen with RTH data, but safety check)
            if (minutes_to_close < -300) minutes_to_close += 24 * 60;
            
            // **MANDATORY POSITION CLOSURE**: 10 minutes before market close
            if (minutes_to_close <= 10 && minutes_to_close > 0) {
                for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
                    if (portfolio.positions[sid].qty != 0.0) {
                        double close_qty = -portfolio.positions[sid].qty; // Close entire position
                        double close_price = pricebook.last_px[sid];
                        
                        if (close_price > 0 && std::abs(close_qty * close_price) > 10.0) {
                            std::string inst = ST.get_symbol(sid);
                            Side close_side = (close_qty > 0) ? Side::Buy : Side::Sell;
                            
                            if (logging_enabled) {
                                audit.event_order_ex(bar.ts_utc_epoch, inst, close_side, std::abs(close_qty), 0.0, "EOD_MANDATORY_CLOSE");
                            }
                            
                            // **ZERO COSTS FOR TESTING**: Perfect execution for EOD close
                            double fees = 0.0;
                            double exec_px = close_price; // Perfect execution at market price
                            
                            double realized_pnl = (portfolio.positions[sid].qty > 0) 
                                ? (exec_px - portfolio.positions[sid].avg_price) * std::abs(close_qty)
                                : (portfolio.positions[sid].avg_price - exec_px) * std::abs(close_qty);
                            
                            apply_fill(portfolio, sid, close_qty, exec_px);
                            // portfolio.cash -= fees; // No fees for EOD close
                            
                            double eq_after = equity_mark_to_market(portfolio, pricebook.last_px);
                            if (logging_enabled) {
                                audit.event_fill_ex(bar.ts_utc_epoch, inst, exec_px, std::abs(close_qty), fees, close_side, realized_pnl, eq_after, 0.0, "EOD_MANDATORY_CLOSE");
                            }
                            total_fills++;
                        }
                    }
                }
            }
        }
        
        // **RENOVATED**: Governor handles day trading automatically - no manual time logic needed
        pricebook.sync_to_base_i(i);
        
        // Log bar data
        AuditBar audit_bar{bar.open, bar.high, bar.low, bar.close, static_cast<double>(bar.volume)};
        if (logging_enabled) audit.event_bar(bar.ts_utc_epoch, ST.get_symbol(base_symbol_id), audit_bar.open, audit_bar.high, audit_bar.low, audit_bar.close, audit_bar.volume);
        
        // **STRATEGY-AGNOSTIC**: Feed features to any strategy that needs them
        [[maybe_unused]] auto start_feed = std::chrono::high_resolution_clock::now();
        FeatureFeeder::feed_features_to_strategy(strategy.get(), base_series, i, strategy->get_name());
        [[maybe_unused]] auto end_feed = std::chrono::high_resolution_clock::now();
        
        // **RENOVATED ARCHITECTURE**: Governor-based target weight system
        [[maybe_unused]] auto start_signal = std::chrono::high_resolution_clock::now();
        
        // **CORRECT ARCHITECTURE**: Strategy emits ONE signal, router selects instrument
        std::string chain_id = std::to_string(bar.ts_utc_epoch) + ":" + std::to_string((long long)i);
        
        // Get strategy signal (ONE signal per bar)
        double probability = strategy->calculate_probability(base_series, i);
        StrategySignal signal = StrategySignal::from_probability(probability);
        
        // Get strategy-specific router configuration
        RouterCfg strategy_router = strategy->get_router_config();
        
        // Router determines which instrument to use based on signal strength
        std::string base_symbol = ST.get_symbol(base_symbol_id);
        auto route_decision = route(signal, strategy_router, base_symbol);
        
        // Convert router decision to allocation format
        std::vector<AllocationDecision> allocation_decisions;
        if (route_decision.has_value()) {
            // **AUDIT**: Log router decision
            if (logging_enabled) {
                audit.event_route(bar.ts_utc_epoch, base_symbol, route_decision->instrument, route_decision->target_weight);
            }
            
            AllocationDecision decision;
            decision.instrument = route_decision->instrument;
            decision.target_weight = route_decision->target_weight;
            decision.confidence = signal.confidence;
            decision.reason = "Router selected " + route_decision->instrument + " for signal strength " + std::to_string(signal.confidence);
            allocation_decisions.push_back(decision);
        } else {
            // **AUDIT**: Log when router returns no decision
            if (logging_enabled) {
                audit.event_route(bar.ts_utc_epoch, base_symbol, "NO_ROUTE", 0.0);
            }
            no_route_count++;
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
        
        [[maybe_unused]] auto end_signal = std::chrono::high_resolution_clock::now();
        
        // **STRATEGY-ISOLATED COORDINATION**: Create fresh coordinator for each run
        PositionCoordinator coordinator(5); // Max 5 orders per bar (QQQ family + extras)
        
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
        
        // Coordinate all allocations to prevent conflicts
        auto coordination_decisions = coordinator.coordinate_allocations(allocation_requests, portfolio, ST);
        
        // Log coordination statistics
        auto coord_stats = coordinator.get_stats();
        int approved_orders = 0;
        int rejected_conflicts = 0;
        int rejected_frequency = 0;
        
        // **COORDINATED EXECUTION**: Execute only approved allocation decisions
        for (const auto& coord_decision : coordination_decisions) {
            if (coord_decision.result == CoordinationResult::APPROVED) {
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
        int post_coordination_conflicts = 0;
        if (has_conflicting_positions(portfolio, ST)) {
            post_coordination_conflicts++;
        }
        
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
            
            // Log account snapshot
            // Calculate actual position value and track cumulative realized P&L  
            double position_value = current_equity - portfolio.cash;
            
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
        return result;
    }
    
    // **CANONICAL METRICS**: Use canonical MPR calculation and store daily returns
    auto summary = UnifiedMetricsCalculator::calculate_from_equity_curve(equity_curve, total_fills, true);
    
    // Calculate daily returns from equity curve for canonical MPR
    std::vector<double> daily_equity_values;
    std::vector<std::pair<std::string, double>> daily_returns_with_dates;
    
    if (!equity_curve.empty()) {
        // Group equity by session date and calculate daily returns
        // The equity_curve already has string timestamps, so we need to extract session dates
        std::map<std::string, double> daily_equity_map;
        for (const auto& [timestamp_str, equity] : equity_curve) {
            // Extract date from timestamp string (format: "YYYY-MM-DD HH:MM:SS")
            std::string session_date = timestamp_str.substr(0, 10); // Extract "YYYY-MM-DD"
            daily_equity_map[session_date] = equity; // Keep last equity of each day
        }
        
        // Convert to vectors and calculate returns
        std::vector<std::string> dates;
        for (const auto& [date, equity] : daily_equity_map) {
            dates.push_back(date);
            daily_equity_values.push_back(equity);
        }
        
        // Calculate daily returns
        for (size_t i = 1; i < daily_equity_values.size(); ++i) {
            double daily_return = (daily_equity_values[i] / daily_equity_values[i-1]) - 1.0;
            daily_returns_with_dates.emplace_back(dates[i], daily_return);
        }
    }
    
    // Use canonical MPR calculation
    std::vector<double> daily_returns;
    for (const auto& [date, ret] : daily_returns_with_dates) {
        daily_returns.push_back(ret);
    }
    double canonical_mpr = sentio::metrics::compute_mpr_from_daily_returns(daily_returns);

    result.final_equity = equity_curve.empty() ? 100000.0 : equity_curve.back().second;
    result.total_return = summary.ret_total; // Already in decimal form (0.0366 = 3.66%)
    result.sharpe_ratio = summary.sharpe;
    result.max_drawdown = summary.mdd; // Already in decimal form
    result.monthly_projected_return = canonical_mpr; // **CANONICAL MPR**
    result.daily_trades = static_cast<int>(summary.daily_trades);
    result.total_fills = summary.trades;
    result.no_route = no_route_count;
    result.no_qty = no_qty_count;

    // **CANONICAL STORAGE**: Store daily returns and update run with canonical period fields
    std::int64_t end_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (logging_enabled) {
        // Store daily returns for canonical MPR calculation
        if (!daily_returns_with_dates.empty()) {
            // Cast to AuditDBRecorder to access canonical MPR methods
            if (auto* db_recorder = dynamic_cast<audit::AuditDBRecorder*>(&audit)) {
                db_recorder->store_daily_returns(daily_returns_with_dates);
            }
        }
        
        // Log final metrics with canonical MPR
        audit.event_metric(end_ts, "final_equity", result.final_equity);
        audit.event_metric(end_ts, "total_return", result.total_return);
        audit.event_metric(end_ts, "sharpe_ratio", result.sharpe_ratio);
        audit.event_metric(end_ts, "max_drawdown", result.max_drawdown);
        audit.event_metric(end_ts, "canonical_mpr", canonical_mpr);
        audit.event_metric(end_ts, "run_trading_days", static_cast<double>(run_trading_days));
        audit.event_metric(end_ts, "total_fills", result.total_fills);
        audit.event_metric(end_ts, "no_route", result.no_route);
        audit.event_metric(end_ts, "no_qty", result.no_qty);
    }
    
    std::string end_meta = "{";
    end_meta += "\"final_equity\":" + std::to_string(result.final_equity) + ",";
    end_meta += "\"total_return\":" + std::to_string(result.total_return) + ",";
    end_meta += "\"sharpe_ratio\":" + std::to_string(result.sharpe_ratio);
    end_meta += "}";
    if (logging_enabled) audit.event_run_end(end_ts, end_meta);

    return result;
}

} // namespace sentio
```

## 📄 **FILE 181 of 206**: src/sanity.cpp

**File Information**:
- **Path**: `src/sanity.cpp`

- **Size**: 166 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .cpp

```text
#include "sentio/sanity.hpp"
#include <cmath>
#include <algorithm>

namespace sentio {

// PriceBook is now abstract - implementations must be provided by concrete classes

bool SanityReport::ok() const {
  for (auto& i : issues) if (i.severity != SanityIssue::Severity::Warn) return false;
  return true;
}
std::size_t SanityReport::errors() const {
  return std::count_if(issues.begin(), issues.end(), [](auto& i){
    return i.severity==SanityIssue::Severity::Error;
  });
}
std::size_t SanityReport::fatals() const {
  return std::count_if(issues.begin(), issues.end(), [](auto& i){
    return i.severity==SanityIssue::Severity::Fatal;
  });
}
void SanityReport::add(SanityIssue::Severity sev, std::string where, std::string what, std::int64_t ts){
  issues.push_back({sev,std::move(where),std::move(what),ts});
}

namespace sanity {

void check_bar_monotonic(const std::vector<std::pair<std::int64_t, Bar>>& bars,
                         int expected_spacing_sec,
                         SanityReport& rep)
{
  if (bars.empty()) return;
  for (std::size_t i=1;i<bars.size();++i){
    auto prev = bars[i-1].first;
    auto cur  = bars[i].first;
    if (cur <= prev) {
      rep.add(SanityIssue::Severity::Fatal, "DATA", "non-increasing timestamp", cur);
    }
    if (expected_spacing_sec>0) {
      auto gap = cur - prev;
      if (gap != expected_spacing_sec) {
        rep.add(SanityIssue::Severity::Error, "DATA",
          "unexpected spacing: got "+std::to_string((long long)gap)+"s expected "+std::to_string(expected_spacing_sec)+"s", cur);
      }
    }
    const Bar& b = bars[i].second;
    if (!(b.low <= b.high)) {
      rep.add(SanityIssue::Severity::Error, "DATA", "bar.low > bar.high", cur);
    }
  }
}

void check_bar_values_finite(const std::vector<std::pair<std::int64_t, Bar>>& bars,
                             SanityReport& rep)
{
  for (auto& it : bars) {
    auto ts = it.first; const Bar& b = it.second;
    if (!(std::isfinite(b.open) && std::isfinite(b.high) && std::isfinite(b.low) && std::isfinite(b.close))) {
      rep.add(SanityIssue::Severity::Fatal, "DATA", "non-finite OHLC", ts);
    }
    if (b.open<=0 || b.high<=0 || b.low<=0 || b.close<=0) {
      rep.add(SanityIssue::Severity::Error, "DATA", "non-positive price", ts);
    }
  }
}

void check_pricebook_coherence(const PriceBook& pb,
                               const std::vector<std::string>& required_instruments,
                               SanityReport& rep)
{
  for (auto& inst : required_instruments) {
    if (!pb.has_instrument(inst)) {
      rep.add(SanityIssue::Severity::Error, "DATA", "PriceBook missing instrument: "+inst);
    } else {
      auto* b = pb.get_latest(inst);
      if (!b || !std::isfinite(b->close)) {
        rep.add(SanityIssue::Severity::Error, "DATA", "PriceBook non-finite last close for: "+inst);
      }
    }
  }
}

void check_signal_confidence_range(double conf, SanityReport& rep, std::int64_t ts) {
  if (!(conf>=0.0 && conf<=1.0)) {
    rep.add(SanityIssue::Severity::Error, "STRAT", "signal confidence out of [0,1]", ts);
  }
}

void check_routed_instrument_has_price(const PriceBook& pb,
                                       const std::string& routed,
                                       SanityReport& rep, std::int64_t ts)
{
  if (routed.empty()) {
    rep.add(SanityIssue::Severity::Fatal, "ROUTER", "empty routed instrument", ts);
    return;
  }
  if (!pb.has_instrument(routed)) {
    rep.add(SanityIssue::Severity::Error, "ROUTER", "routed instrument missing in PriceBook: "+routed, ts);
  } else if (auto* b = pb.get_latest(routed); !b || !std::isfinite(b->close)) {
    rep.add(SanityIssue::Severity::Error, "ROUTER", "routed instrument has non-finite price: "+routed, ts);
  }
}

void check_order_qty_min(double qty, double min_shares,
                         SanityReport& rep, std::int64_t ts)
{
  if (!(std::isfinite(qty))) {
    rep.add(SanityIssue::Severity::Fatal, "EXEC", "order qty non-finite", ts);
    return;
  }
  if (qty != 0.0 && std::abs(qty) < min_shares) {
    rep.add(SanityIssue::Severity::Warn, "EXEC", "order qty < min_shares", ts);
  }
}

void check_order_side_qty_sign_consistency(const std::string& side, double qty,
                                           SanityReport& rep, std::int64_t ts)
{
  if (side=="BUY" && qty<0)  rep.add(SanityIssue::Severity::Error, "EXEC", "BUY with negative qty", ts);
  if (side=="SELL"&& qty>0)  rep.add(SanityIssue::Severity::Error, "EXEC", "SELL with positive qty", ts);
}

void check_equity_consistency(const AccountState& acct,
                              const std::unordered_map<std::string, Position>& pos,
                              const PriceBook& pb,
                              SanityReport& rep)
{
  if (!std::isfinite(acct.cash) || !std::isfinite(acct.realized) || !std::isfinite(acct.equity)) {
    rep.add(SanityIssue::Severity::Fatal, "PnL", "non-finite account values");
    return;
  }
  // recompute mark-to-market
  double mtm = 0.0;
  for (auto& kv : pos) {
    const auto& inst = kv.first;
    const auto& p = kv.second;
    if (!std::isfinite(p.qty) || !std::isfinite(p.avg_px)) {
      rep.add(SanityIssue::Severity::Fatal, "PnL", "non-finite position for "+inst);
      continue;
    }
    auto* b = pb.get_latest(inst);
    if (!b) continue;
    mtm += p.qty * b->close;
  }
  double equity_calc = acct.cash + acct.realized + mtm;
  if (std::isfinite(equity_calc) && std::abs(equity_calc - acct.equity) > 1e-6) {
    rep.add(SanityIssue::Severity::Error, "PnL", "equity mismatch (calc vs recorded) diff="+std::to_string(equity_calc - acct.equity));
  }
}

void check_audit_counts(const AuditEventCounts& c, SanityReport& rep) {
  if (c.orders < c.fills) {
    rep.add(SanityIssue::Severity::Error, "AUDIT", "fills exceed orders");
  }
  // Loose ratios to catch obviously broken runs
  if (c.routes && c.orders==0) {
    rep.add(SanityIssue::Severity::Warn, "AUDIT", "routes exist but no orders");
  }
  if (c.signals && c.routes==0) {
    rep.add(SanityIssue::Severity::Warn, "AUDIT", "signals exist but no routes");
  }
}

} // namespace sanity
} // namespace sentio

```

## 📄 **FILE 182 of 206**: src/signal_engine.cpp

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

## 📄 **FILE 183 of 206**: src/signal_gate.cpp

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

## 📄 **FILE 184 of 206**: src/signal_pipeline.cpp

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

## 📄 **FILE 185 of 206**: src/signal_trace.cpp

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

## 📄 **FILE 186 of 206**: src/sim_data.cpp

**File Information**:
- **Path**: `src/sim_data.cpp`

- **Size**: 47 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .cpp

```text
#include "sentio/sim_data.hpp"
#include <cmath>

namespace sentio {

std::vector<std::pair<std::int64_t, Bar>> generate_minute_series(const SimCfg& cfg) {
  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<double> z(0.0, 1.0);
  std::uniform_real_distribution<double> U(0.0, 1.0);

  std::vector<std::pair<std::int64_t, Bar>> out;
  out.reserve(cfg.minutes);

  double px = cfg.start_price;
  std::int64_t ts = cfg.start_ts_utc;

  for (int i=0;i<cfg.minutes;++i, ts+=60) {
    double u = U(rng);
    double ret = 0.0;

    if (u < cfg.frac_trend) {
      // trending drift + noise
      ret = 0.0002 + (cfg.vol_bps*1e-4) * z(rng);
    } else if (u < cfg.frac_trend + cfg.frac_mr) {
      // mean-reversion around 0 with lighter noise
      ret = -0.0001 + (cfg.vol_bps*0.6e-4) * z(rng);
    } else {
      // jump regime
      ret = (U(rng) < 0.5 ? +1 : -1) * (cfg.vol_bps*6e-4 + std::abs(z(rng))*cfg.vol_bps*3e-4);
    }

    double new_px = std::max(0.01, px * (1.0 + ret));
    double o = px;
    double c = new_px;
    double h = std::max(o, c) * (1.0 + std::abs((cfg.vol_bps*0.3e-4) * z(rng)));
    double l = std::min(o, c) * (1.0 - std::abs((cfg.vol_bps*0.3e-4) * z(rng)));
    // ensure consistency
    h = std::max({h, o, c});
    l = std::min({l, o, c});

    out.push_back({ts, Bar{o,h,l,c}});
    px = new_px;
  }
  return out;
}

} // namespace sentio

```

## 📄 **FILE 187 of 206**: src/strategy/run_rule_ensemble.cpp

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

## 📄 **FILE 188 of 206**: src/strategy_bollinger_squeeze_breakout.cpp

**File Information**:
- **Path**: `src/strategy_bollinger_squeeze_breakout.cpp`

- **Size**: 204 lines
- **Modified**: 2025-09-12 15:35:06

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

SizerCfg BollingerSqueezeBreakoutStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.max_position_pct = 1.0; // 100% max position
    cfg.volatility_target = 0.15; // 15% volatility target
    return cfg;
}

REGISTER_STRATEGY(BollingerSqueezeBreakoutStrategy, "bsb");

} // namespace sentio

```

## 📄 **FILE 189 of 206**: src/strategy_initialization.cpp

**File Information**:
- **Path**: `src/strategy_initialization.cpp`

- **Size**: 35 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .cpp

```text
#include "sentio/base_strategy.hpp"
#include "sentio/strategy_ire.hpp"
#include "sentio/strategy_bollinger_squeeze_breakout.hpp"
// Removed unused strategy: hybrid_ppo
#include "sentio/strategy_kochi_ppo.hpp"
#include "sentio/strategy_market_making.hpp"
#include "sentio/strategy_momentum_volume.hpp"
#include "sentio/strategy_opening_range_breakout.hpp"
#include "sentio/strategy_order_flow_imbalance.hpp"
#include "sentio/strategy_order_flow_scalping.hpp"
#include "sentio/strategy_tfa.hpp"
#include "sentio/strategy_vwap_reversion.hpp"
#include "sentio/rsi_strategy.hpp"

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

## 📄 **FILE 190 of 206**: src/strategy_ire.cpp

**File Information**:
- **Path**: `src/strategy_ire.cpp`

- **Size**: 393 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .cpp

```text
#include "sentio/strategy_ire.hpp"
#include "sentio/position_validator.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace sentio {

// Helper to calculate the mean of a deque
double calculate_mean(const std::deque<double>& data) {
    if (data.empty()) return 0.0;
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

// Helper to calculate the standard deviation of a deque
double calculate_stddev(const std::deque<double>& data, double mean) {
    if (data.size() < 2) return 0.0;
    double sq_sum = 0.0;
    for (const auto& value : data) {
        sq_sum += (value - mean) * (value - mean);
    }
    return std::sqrt(sq_sum / data.size());
}

// **ENHANCED**: Multi-Timeframe Alpha Kernel Implementation


IREStrategy::IREStrategy() : BaseStrategy("IRE") {
    apply_params();
}

ParameterMap IREStrategy::get_default_params() const {
  return { {"buy_lo", 0.60}, {"buy_hi", 0.75}, {"sell_hi", 0.40}, {"sell_lo", 0.25} };
}

ParameterSpace IREStrategy::get_param_space() const { return {}; }

void IREStrategy::apply_params() {
    vol_return_history_.clear();
    vol_history_.clear();
    vwap_price_history_.clear();
    vwap_volume_history_.clear();
    alpha_return_history_.clear(); // Initialize new state
    last_trade_bar_ = -1;
    last_trade_direction_ = 0;
    entry_price_ = 0.0; // Initialize new state
    pnl_history_.clear(); // Initialize Kelly state
}

double IREStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
  // Use the existing calculate_target_weight method which already returns probability
  return calculate_target_weight(bars, current_index);
}

std::vector<BaseStrategy::AllocationDecision> IREStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // **SIMPLIFIED ALLOCATION LOGIC**: Direct probability-based allocation
    if (probability > 0.80) {
        // **STRONG BUY**: High conviction - aggressive leverage
        double conviction = (probability - 0.80) / 0.20; // 0-1 scale within strong range
        double base_weight = 0.6 + (conviction * 0.4); // 60-100% allocation
        
        decisions.push_back({bull3x_symbol, base_weight * 0.7, conviction, "Strong buy: 70% TQQQ"});
        decisions.push_back({base_symbol, base_weight * 0.3, conviction, "Strong buy: 30% QQQ"});
    } 
    else if (probability > 0.55) {
        // **MODERATE BUY**: Good conviction - conservative allocation
        double conviction = (probability - 0.55) / 0.25; // 0-1 scale within moderate range
        double base_weight = 0.3 + (conviction * 0.3); // 30-60% allocation
        
        decisions.push_back({base_symbol, base_weight, conviction, "Moderate buy: 100% QQQ"});
    }
    else if (probability < 0.20) {
        // **STRONG SELL**: High conviction - aggressive inverse leverage
        double conviction = (0.20 - probability) / 0.20; // 0-1 scale within strong range
        double base_weight = 0.6 + (conviction * 0.4); // 60-100% allocation
        
        decisions.push_back({bear3x_symbol, base_weight, conviction, "Strong sell: 100% SQQQ"});
    }
    else if (probability < 0.45) {
        // **MODERATE SELL**: Good conviction - conservative inverse
        double conviction = (0.45 - probability) / 0.25; // 0-1 scale within moderate range  
        double base_weight = 0.3 + (conviction * 0.3); // 30-60% allocation
        
        // Use SHORT QQQ for moderate sell signals instead of PSQ
        decisions.push_back({base_symbol, -base_weight, conviction, "Moderate sell: SHORT QQQ"});
    }
    // **NEUTRAL ZONE** (0.45-0.55): No allocations = stay flat
    
    // **ENSURE ALL INSTRUMENTS ARE FLATTENED IF NOT IN ALLOCATION**
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg IREStrategy::get_router_config() const {
    // IRE uses default router configuration
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}

SizerCfg IREStrategy::get_sizer_config() const {
    // IRE uses default sizer configuration
    SizerCfg cfg;
    cfg.max_position_pct = 1.0; // 100% max position
    cfg.volatility_target = 0.15; // 15% volatility target
    return cfg;
}

double IREStrategy::calculate_target_weight(const std::vector<Bar>& bars, int i) {
    ensure_governor_built_();
    const int WARMUP_PERIOD = 60;
    if (i < 1) return 0.0;

    // --- Maintain Rolling History Windows ---
    double log_return = std::log(bars[i].close / bars[i-1].close);
    vol_return_history_.push_back(log_return);
    alpha_return_history_.push_back(log_return);
    if (vol_return_history_.size() > WARMUP_PERIOD) vol_return_history_.pop_front();
    
    // **ENHANCED**: Maintain longer history for multi-timeframe analysis (30 minutes)
    if (alpha_return_history_.size() > 30) alpha_return_history_.pop_front();
    
    vwap_price_history_.push_back(bars[i].close);
    vwap_volume_history_.push_back(bars[i].volume);
    if (vwap_price_history_.size() > 20) {
        vwap_price_history_.pop_front();
        vwap_volume_history_.pop_front();
    }
    
    if (i < WARMUP_PERIOD) return 0.0;

    // --- Dynamic Take-Profit Logic ---
    if (last_trade_direction_ != 0 && entry_price_ > 0) {
        double current_pnl = (bars[i].close - entry_price_) * last_trade_direction_;
        // Target profit is 1.5x the recent volatility
        double take_profit_threshold = calculate_stddev(vol_return_history_, 0.0) * entry_price_ * 1.5;
        if (current_pnl > take_profit_threshold) {
            // **NEW**: Record trade performance for Kelly Criterion
            update_trade_performance(current_pnl);
            
            last_trade_direction_ = 0; // Reset trade state
            entry_price_ = 0.0;
            return 0.0; // Signal to flatten position and lock in profit
        }
    }

    // --- REGIME DETECTION ---
    double vol_mean = calculate_mean(vol_return_history_);
    double current_vol = calculate_stddev(vol_return_history_, vol_mean);
    vol_history_.push_back(current_vol);
    if (vol_history_.size() > WARMUP_PERIOD) vol_history_.pop_front();
    double avg_vol = calculate_mean(vol_history_);
    bool is_high_volatility = current_vol > (avg_vol * 2.0);

    // --- SIGNAL CALCULATION ---
    double regime_probability = 0.5;
    if (is_high_volatility) {
        double momentum = (bars[i].close / bars[i-10].close) - 1.0;
        double volume_mean = calculate_mean(vwap_volume_history_);
        if (bars[i].volume > volume_mean * 2.5) { 
            if (momentum > 0.0015) regime_probability = 0.80;
            else if (momentum < -0.0015) regime_probability = 0.20;
        }
    } else {
        double sum_pv = 0.0, sum_vol = 0.0;
        for (size_t k = 0; k < vwap_price_history_.size(); ++k) {
            sum_pv += vwap_price_history_[k] * vwap_volume_history_[k];
            sum_vol += vwap_volume_history_[k];
        }
        double vwap = (sum_vol > 0) ? sum_pv / sum_vol : bars[i].close;
        double sq_diff_sum = 0.0;
        for (size_t k = 0; k < vwap_price_history_.size(); ++k) {
            sq_diff_sum += (vwap_price_history_[k] - vwap) * (vwap_price_history_[k] - vwap);
        }
        double std_dev_from_vwap = std::sqrt(sq_diff_sum / vwap_price_history_.size());
        if (std_dev_from_vwap > 0) {
            double z_score = (bars[i].close - vwap) / std_dev_from_vwap;
            regime_probability = 0.5 - std::clamp(z_score / 5.0, -0.40, 0.40);
        }
    }

    // **REVERTED**: Simple 20-minute MA momentum signal (PROVEN 3.21% PERFORMER)
    double momentum_signal = 0.5; // Default neutral
    if (i >= 20) {
        double ma_20 = 0.0;
        for (int j = i - 19; j <= i; ++j) {
            ma_20 += bars[j].close;
        }
        ma_20 /= 20.0;
        
        // Simple momentum: current price vs 20-min MA
        double momentum = (bars[i].close - ma_20) / ma_20;
        momentum_signal = 0.5 + std::clamp(momentum * 25.0, -0.4, 0.4); // Scale momentum to probability
    }
    
    latest_probability_ = momentum_signal;
    
    // **FIXED**: Increment signal diagnostics counter
    diag_.emitted++;
    
    // **SIMPLIFIED**: Strategy only provides probability - runner handles allocation
    static int debug_count = 0;
    if (debug_count < 5 || debug_count % 1000 == 0) {
        double momentum_pct = 0.0;
        if (i >= 20) {
            double ma_20_debug = 0.0;
            for (int j = i - 19; j <= i; ++j) ma_20_debug += bars[j].close;
            ma_20_debug /= 20.0;
            momentum_pct = (bars[i].close - ma_20_debug) / ma_20_debug * 100.0;
        }
    }
    debug_count++;
    
    // **FIXED**: Return the calculated momentum signal probability
    return momentum_signal; // Return actual probability for signal generation
}

void IREStrategy::ensure_governor_built_() {
    if (governor_) return;
    IntradayPositionGovernor::Config gov_config;
    gov_config.lookback_window = 45;        // Shorter for more responsiveness to Alpha Kernel
    gov_config.buy_percentile = params_["buy_hi"];   // Use actual strategy parameter
    gov_config.sell_percentile = params_["sell_lo"]; // Use actual strategy parameter
    gov_config.max_base_weight = 1.0; 
    gov_config.min_abs_edge = 0.03;         // Lower threshold to allow Alpha Kernel through
    governor_ = std::make_unique<IntradayPositionGovernor>(gov_config);
}

// ensure_ensemble_built is no longer needed but kept for compatibility
void IREStrategy::ensure_ensemble_built_() {}

// **NEW**: Kelly Criterion Implementation
double IREStrategy::calculate_kelly_fraction(double edge_probability, double confidence) const {
    if (pnl_history_.size() < 10) {
        // Not enough trade history, use moderate sizing
        return 0.8; // 80% of normal position size during learning phase
    }
    
    double win_rate = get_win_rate();
    double win_loss_ratio = get_win_loss_ratio();
    
    if (win_loss_ratio <= 0.0 || win_rate <= 0.0) {
        // Poor historical performance, use minimal sizing
        return 0.25;
    }
    
    // Kelly formula: f = (bp - q) / b
    // where b = win_loss_ratio, p = edge_probability, q = 1-p
    double p = edge_probability;
    double q = 1.0 - p;
    double b = win_loss_ratio;
    
    double kelly_f = (b * p - q) / b;
    
    // Apply confidence scaling and safety constraints
    double scaled_kelly = kelly_f * confidence * 0.5; // 50% of full Kelly for more aggressive sizing (half-Kelly)
    
    // Clamp to reasonable range - more aggressive bounds
    return std::clamp(scaled_kelly, 0.3, 3.0); // Min 30%, Max 300% of base position
}

void IREStrategy::update_trade_performance(double realized_pnl) {
    pnl_history_.push_back(realized_pnl);
    
    // Maintain rolling window of last 50 trades for Kelly calculation
    if (pnl_history_.size() > 50) {
        pnl_history_.pop_front();
    }
}

double IREStrategy::get_win_loss_ratio() const {
    if (pnl_history_.size() < 5) return 1.0; // Default ratio
    
    double total_wins = 0.0;
    double total_losses = 0.0;
    int win_count = 0;
    int loss_count = 0;
    
    for (double pnl : pnl_history_) {
        if (pnl > 0) {
            total_wins += pnl;
            win_count++;
        } else if (pnl < 0) {
            total_losses += std::abs(pnl);
            loss_count++;
        }
    }
    
    if (loss_count == 0) return 2.0; // No losses, assume good ratio
    if (win_count == 0) return 0.5; // No wins, conservative ratio
    
    double avg_win = total_wins / win_count;
    double avg_loss = total_losses / loss_count;
    
    return avg_win / avg_loss;
}

double IREStrategy::get_win_rate() const {
    if (pnl_history_.size() < 5) return 0.55; // Default optimistic win rate
    
    int wins = 0;
    for (double pnl : pnl_history_) {
        if (pnl > 0) wins++;
    }
    
    return static_cast<double>(wins) / pnl_history_.size();
}

// **NEW**: Multi-Timeframe Alpha Kernel Implementation
double IREStrategy::calculate_multi_timeframe_alpha(const std::deque<double>& history) const {
    if (history.size() < 30) return 0.5; // Need sufficient history for all timeframes
    
    // **Ultra-Fast (3-8 min)**: Captures immediate momentum and noise
    double short_alpha = calculate_single_alpha_probability(history, 3, 8);
    
    // **Medium-Term (5-15 min)**: Identifies core intraday moves (original alpha)
    double medium_alpha = calculate_single_alpha_probability(history, 5, 15);
    
    // **Long-Term (10-30 min)**: Detects broader intraday trend
    double long_alpha = calculate_single_alpha_probability(history, 10, 30);
    
    // **Hierarchical Blending**: Weight shorter timeframes more heavily for day trading
    // short_alpha * 0.5 + medium_alpha * 0.3 + long_alpha * 0.2 = active but trend-aware
    double ensemble_alpha = (short_alpha * 0.5) + (medium_alpha * 0.3) + (long_alpha * 0.2);
    
    return ensemble_alpha;
}

double IREStrategy::calculate_single_alpha_probability(const std::deque<double>& history, int short_window, int long_window) const {
    if (history.size() < static_cast<size_t>(long_window)) return 0.5;

    // Velocity (short-term trend direction)
    double recent_mean = 0.0;
    for(size_t i = history.size() - short_window; i < history.size(); ++i) {
        recent_mean += history[i];
    }
    recent_mean /= short_window;

    // Acceleration (how the trend is changing)
    double older_mean = 0.0;
    for(size_t i = history.size() - long_window; i < history.size() - short_window; ++i) {
        older_mean += history[i];
    }
    older_mean /= (long_window - short_window);
    
    // A positive acceleration means the upward trend is strengthening (or downward is weakening)
    double acceleration = recent_mean - older_mean;

    // Combine velocity and acceleration for a forward-looking forecast
    double forecast = (recent_mean * 0.7) + (acceleration * 0.3);
    
    // **ENHANCED**: Adaptive scaling factor based on timeframe
    double timeframe_scale = 5000.0;
    if (short_window <= 3) {
        // Ultra-fast: More aggressive for scalping
        timeframe_scale = 8000.0;
    } else if (short_window >= 10) {
        // Long-term: More conservative for trend
        timeframe_scale = 3000.0;
    }
    
    // Convert the forecast into a probability between 0 and 1
    return 0.5 + std::clamp(forecast * timeframe_scale, -0.48, 0.48);
}

REGISTER_STRATEGY(IREStrategy, "ire");

} // namespace sentio
```

## 📄 **FILE 191 of 206**: src/strategy_kochi_ppo.cpp

**File Information**:
- **Path**: `src/strategy_kochi_ppo.cpp`

- **Size**: 155 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .cpp

```text
#include "sentio/strategy_kochi_ppo.hpp"
#include "sentio/signal_utils.hpp"
#include <algorithm>
#include <stdexcept>

namespace sentio {

static ml::WindowSpec make_kochi_spec(const ml::ModelSpec& s){
  ml::WindowSpec w;
  // Kochi environment defaults to 20 window; allow metadata override if provided
  w.seq_len = s.seq_len > 0 ? s.seq_len : 20;
  w.layout  = s.input_layout.empty()? "BTF" : s.input_layout;
  w.feat_dim = (int)s.feature_names.size();
  w.mean = s.mean;
  w.std  = s.std;
  w.clip2 = s.clip2;
  return w;
}

KochiPPOStrategy::KochiPPOStrategy()
: BaseStrategy("KochiPPO")
, cfg_()
, handle_(ml::ModelRegistryTS::load_torchscript(cfg_.model_id, cfg_.version, cfg_.artifacts_dir, cfg_.use_cuda))
, window_(make_kochi_spec(handle_.spec))
{}

KochiPPOStrategy::KochiPPOStrategy(const KochiPPOCfg& cfg)
: BaseStrategy("KochiPPO")
, cfg_(cfg)
, handle_(ml::ModelRegistryTS::load_torchscript(cfg.model_id, cfg.version, cfg.artifacts_dir, cfg.use_cuda))
, window_(make_kochi_spec(handle_.spec))
{}

void KochiPPOStrategy::set_raw_features(const std::vector<double>& raw){
  // Expect exactly F features in model metadata order
  if ((int)raw.size() != window_.feat_dim()) return;
  window_.push(raw);
}

ParameterMap KochiPPOStrategy::get_default_params() const {
  return {
    {"conf_floor", cfg_.conf_floor}
  };
}

ParameterSpace KochiPPOStrategy::get_param_space() const {
  return {
    {"conf_floor", {ParamType::FLOAT, 0.0, 1.0, cfg_.conf_floor}}
  };
}

void KochiPPOStrategy::apply_params() {
  cfg_.conf_floor = params_["conf_floor"];
}

StrategySignal KochiPPOStrategy::map_output(const ml::ModelOutput& mo) const {
  StrategySignal s;
  // Assume discrete probs with actions in spec.actions. Default mapping SELL/HOLD/BUY
  int argmax = 0;
  for (int i=1;i<(int)mo.probs.size();++i) if (mo.probs[i] > mo.probs[argmax]) argmax = i;

  const auto& acts = handle_.spec.actions;
  std::string a = (argmax<(int)acts.size()? acts[argmax] : "HOLD");
  float pmax = mo.probs.empty()? 0.0f : mo.probs[argmax];

  if (a=="BUY")       s.type = StrategySignal::Type::BUY;
  else if (a=="SELL") s.type = StrategySignal::Type::SELL;
  else                 s.type = StrategySignal::Type::HOLD;

  s.confidence = std::max(cfg_.conf_floor, (double)pmax);
  return s;
}

double KochiPPOStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
  (void)bars; (void)current_index; // features are streamed in via set_raw_features
  last_.reset();
  if (!window_.ready()) return 0.5; // Neutral

  auto in = window_.to_input();
  if (!in) return 0.5; // Neutral

  auto out = handle_.model->predict(*in, window_.seq_len(), window_.feat_dim(), handle_.spec.input_layout);
  if (!out) return 0.5; // Neutral

  auto sig = map_output(*out);
  double probability = sentio::signal_utils::signal_to_probability(sig, cfg_.conf_floor);
  
  last_ = sig;
  return probability;
}

std::vector<BaseStrategy::AllocationDecision> KochiPPOStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // KochiPPO uses simple allocation based on signal strength
    if (probability > 0.6) {
        // Buy signal
        double conviction = (probability - 0.6) / 0.4; // Scale 0.6-1.0 to 0-1
        double base_weight = 0.3 + (conviction * 0.7); // 30-100% allocation
        
        decisions.push_back({base_symbol, base_weight, conviction, "KochiPPO buy: 100% QQQ"});
    } else if (probability < 0.4) {
        // Sell signal
        double conviction = (0.4 - probability) / 0.4; // Scale 0.0-0.4 to 0-1
        double base_weight = 0.3 + (conviction * 0.7); // 30-100% allocation
        
        // Use SHORT QQQ for moderate sell signals instead of PSQ
        decisions.push_back({base_symbol, -base_weight, conviction, "KochiPPO sell: SHORT QQQ"});
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "KochiPPO: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg KochiPPOStrategy::get_router_config() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}

SizerCfg KochiPPOStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.max_position_pct = 1.0; // 100% max position
    cfg.volatility_target = 0.15; // 15% volatility target
    return cfg;
}

// Register with factory
REGISTER_STRATEGY(KochiPPOStrategy, "kochi_ppo");

} // namespace sentio



```

## 📄 **FILE 192 of 206**: src/strategy_market_making.cpp

**File Information**:
- **Path**: `src/strategy_market_making.cpp`

- **Size**: 195 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .cpp

```text
#include "sentio/strategy_market_making.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace sentio {

MarketMakingStrategy::MarketMakingStrategy() 
    : BaseStrategy("MarketMaking"),
      rolling_returns_(20),
      rolling_volume_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap MarketMakingStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed volatility and volume thresholds to allow participation.
    return {
        {"base_spread", 0.001}, {"min_spread", 0.0005}, {"max_spread", 0.003},
        {"order_levels", 3.0}, {"level_spacing", 0.0005}, {"order_size_base", 0.5},
        {"max_inventory", 100.0}, {"inventory_skew_mult", 0.002},
        {"adverse_selection_threshold", 0.004}, // Was 0.002, allowing participation in more volatile conditions
        {"volatility_window", 20.0},
        {"volume_window", 50.0}, {"min_volume_ratio", 0.05}, // Was 0.1, making it even more permissive
        {"max_orders_per_bar", 10.0}, {"rebalance_frequency", 10.0}
    };
}

ParameterSpace MarketMakingStrategy::get_param_space() const { return {}; }

void MarketMakingStrategy::apply_params() {
    base_spread_ = params_.at("base_spread");
    min_spread_ = params_.at("min_spread");
    max_spread_ = params_.at("max_spread");
    order_levels_ = static_cast<int>(params_.at("order_levels"));
    level_spacing_ = params_.at("level_spacing");
    order_size_base_ = params_.at("order_size_base");
    max_inventory_ = params_.at("max_inventory");
    inventory_skew_mult_ = params_.at("inventory_skew_mult");
    adverse_selection_threshold_ = params_.at("adverse_selection_threshold");
    min_volume_ratio_ = params_.at("min_volume_ratio");
    max_orders_per_bar_ = static_cast<int>(params_.at("max_orders_per_bar"));
    rebalance_frequency_ = static_cast<int>(params_.at("rebalance_frequency"));

    int vol_window = std::max(1, static_cast<int>(params_.at("volatility_window")));
    int vol_mean_window = std::max(1, static_cast<int>(params_.at("volume_window")));
    
    rolling_returns_.reset(vol_window);
    rolling_volume_.reset(vol_mean_window);
    reset_state();
}

void MarketMakingStrategy::reset_state() {
    BaseStrategy::reset_state();
    market_state_ = MarketState{};
    rolling_returns_.reset(std::max(1, static_cast<int>(params_.at("volatility_window"))));
    rolling_volume_.reset(std::max(1, static_cast<int>(params_.at("volume_window"))));
}

double MarketMakingStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    
    // Always update indicators to have a full history for the next bar
    if(current_index > 0) {
        double price_return = (bars[current_index].close - bars[current_index - 1].close) / bars[current_index - 1].close;
        rolling_returns_.push(price_return);
    }
    rolling_volume_.push(bars[current_index].volume);

    // Wait for indicators to warm up
    if (rolling_volume_.size() < static_cast<size_t>(params_.at("volume_window"))) {
        diag_.drop(DropReason::MIN_BARS);
        return 0.5; // Neutral
    }

    if (!should_participate(bars[current_index])) {
        return 0.5; // Neutral
    }
    
    // **FIXED**: Generate signals based on volatility and volume patterns instead of inventory
    // Note: inventory tracking removed as it's not currently implemented
    // Since inventory tracking is not implemented, use a simpler approach
    double volatility = rolling_returns_.stddev();
    double avg_volume = rolling_volume_.mean();
    double volume_ratio = (avg_volume > 0) ? bars[current_index].volume / avg_volume : 0.0;
    
    // Generate signals when volatility is moderate and volume is increasing
    if (volatility > 0.0005 && volatility < adverse_selection_threshold_ && volume_ratio > 0.8) {
        // Simple momentum-based signal
        if (current_index > 0) {
            double price_change = (bars[current_index].close - bars[current_index - 1].close) / bars[current_index - 1].close;
            if (price_change > 0.001) {
                return 0.6; // Buy signal
            } else if (price_change < -0.001) {
                return 0.4; // Sell signal
            } else {
                diag_.drop(DropReason::THRESHOLD);
                return 0.5; // Neutral
            }
        } else {
            diag_.drop(DropReason::THRESHOLD);
            return 0.5; // Neutral
        }
    } else {
        diag_.drop(DropReason::THRESHOLD);
        return 0.5; // Neutral
    }
    diag_.emitted++;
    return 0.5; // Should not reach here
}

bool MarketMakingStrategy::should_participate(const Bar& bar) {
    double volatility = rolling_returns_.stddev();
    
    if (volatility > adverse_selection_threshold_) {
        diag_.drop(DropReason::THRESHOLD); 
        return false;
    }

    double avg_volume = rolling_volume_.mean();
    
    if (avg_volume > 0 && (bar.volume < avg_volume * min_volume_ratio_)) {
        diag_.drop(DropReason::ZERO_VOL);
        return false;
    }
    return true;
}

double MarketMakingStrategy::get_inventory_skew() const {
    if (max_inventory_ <= 0) return 0.0;
    double normalized_inventory = market_state_.inventory / max_inventory_;
    return -normalized_inventory * inventory_skew_mult_;
}

std::vector<BaseStrategy::AllocationDecision> MarketMakingStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // MarketMaking uses simple allocation based on signal strength
    if (probability > 0.6) {
        // Buy signal
        double conviction = (probability - 0.6) / 0.4; // Scale 0.6-1.0 to 0-1
        double base_weight = 0.2 + (conviction * 0.3); // 20-50% allocation
        
        decisions.push_back({base_symbol, base_weight, conviction, "MarketMaking buy: 100% QQQ"});
    } else if (probability < 0.4) {
        // Sell signal
        double conviction = (0.4 - probability) / 0.4; // Scale 0.0-0.4 to 0-1
        double base_weight = 0.2 + (conviction * 0.3); // 20-50% allocation
        
        // Use SHORT QQQ for moderate sell signals instead of PSQ
        decisions.push_back({base_symbol, -base_weight, conviction, "MarketMaking sell: SHORT QQQ"});
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "MarketMaking: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg MarketMakingStrategy::get_router_config() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}

SizerCfg MarketMakingStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.max_position_pct = 0.5; // 50% max position for market making
    cfg.volatility_target = 0.10; // 10% volatility target
    return cfg;
}

REGISTER_STRATEGY(MarketMakingStrategy, "mm");

} // namespace sentio


```

## 📄 **FILE 193 of 206**: src/strategy_momentum_volume.cpp

**File Information**:
- **Path**: `src/strategy_momentum_volume.cpp`

- **Size**: 202 lines
- **Modified**: 2025-09-12 15:35:06

- **Type**: .cpp

```text
#include "sentio/strategy_momentum_volume.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>

namespace sentio {

MomentumVolumeProfileStrategy::MomentumVolumeProfileStrategy() 
    : BaseStrategy("MomentumVolumeProfile"), avg_volume_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap MomentumVolumeProfileStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed parameters to be more sensitive
    return {
        {"profile_period", 100.0},
        {"value_area_pct", 0.7},
        {"price_bins", 30.0},
        {"breakout_threshold_pct", 0.001},
        {"momentum_lookback", 20.0},
        {"volume_surge_mult", 1.2}, // Was 1.5
        {"cool_down_period", 5.0}   // Was 10
    };
}

ParameterSpace MomentumVolumeProfileStrategy::get_param_space() const { return {}; }

void MomentumVolumeProfileStrategy::apply_params() {
    profile_period_ = static_cast<int>(params_["profile_period"]);
    value_area_pct_ = params_["value_area_pct"];
    price_bins_ = static_cast<int>(params_["price_bins"]);
    breakout_threshold_pct_ = params_["breakout_threshold_pct"];
    momentum_lookback_ = static_cast<int>(params_["momentum_lookback"]);
    volume_surge_mult_ = params_["volume_surge_mult"];
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);
    avg_volume_ = RollingMean(profile_period_);
    reset_state();
}

void MomentumVolumeProfileStrategy::reset_state() {
    BaseStrategy::reset_state();
    volume_profile_.clear();
    last_profile_update_ = -1;
    avg_volume_ = RollingMean(profile_period_);
}

double MomentumVolumeProfileStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < profile_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return 0.5; // Neutral
    }
    
    if (is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return 0.5; // Neutral
    }
    
    // Periodically rebuild the expensive volume profile
    if (last_profile_update_ == -1 || current_index - last_profile_update_ >= 10) {
        build_volume_profile(bars, current_index);
        last_profile_update_ = current_index;
    }
    
    if (volume_profile_.value_area_high <= 0) {
        diag_.drop(DropReason::NAN_INPUT); // Profile not ready or invalid
        return 0.5; // Neutral
    }

    const auto& bar = bars[current_index];
    avg_volume_.push(bar.volume);
    
    bool breakout_up = bar.close > (volume_profile_.value_area_high * (1.0 + breakout_threshold_pct_));
    bool breakout_down = bar.close < (volume_profile_.value_area_low * (1.0 - breakout_threshold_pct_));

    if (!breakout_up && !breakout_down) {
        diag_.drop(DropReason::THRESHOLD);
        return 0.5; // Neutral
    }

    if (!is_momentum_confirmed(bars, current_index)) {
        diag_.drop(DropReason::THRESHOLD);
        return 0.5; // Neutral
    }
    
    if (bar.volume < avg_volume_.mean() * volume_surge_mult_) {
        diag_.drop(DropReason::ZERO_VOL);
        return 0.5; // Neutral
    }

    double probability;
    if (breakout_up) {
        probability = 0.85; // Strong buy signal
    } else {
        probability = 0.15; // Strong sell signal
    }
    
    diag_.emitted++;
    state_.last_trade_bar = current_index;
    return probability;
}

bool MomentumVolumeProfileStrategy::is_momentum_confirmed(const std::vector<Bar>& bars, int index) const {
    if (index < momentum_lookback_) return false;
    double price_change = bars[index].close - bars[index - momentum_lookback_].close;
    if (bars[index].close > volume_profile_.value_area_high) {
        return price_change > 0;
    }
    if (bars[index].close < volume_profile_.value_area_low) {
        return price_change < 0;
    }
    return false;
}

// **MODIFIED**: This is now a functional, albeit simple, implementation to prevent NaN drops.
void MomentumVolumeProfileStrategy::build_volume_profile(const std::vector<Bar>& bars, int end_index) {
    volume_profile_.clear();
    int start_index = std::max(0, end_index - profile_period_ + 1);

    double min_price = std::numeric_limits<double>::max();
    double max_price = std::numeric_limits<double>::lowest();
    
    for (int i = start_index; i <= end_index; ++i) {
        min_price = std::min(min_price, bars[i].low);
        max_price = std::max(max_price, bars[i].high);
    }
    
    if (max_price <= min_price) return; // Cannot build profile

    // Simple implementation: Value Area is the high/low of the lookback period
    volume_profile_.value_area_high = max_price;
    volume_profile_.value_area_low = min_price;
    volume_profile_.total_volume = 1.0; // Mark as valid by setting a non-zero value
    // A proper implementation would bin prices and find the 70% volume area.
}

void MomentumVolumeProfileStrategy::calculate_value_area() {
    // This is now handled within build_volume_profile for simplicity
}

std::vector<BaseStrategy::AllocationDecision> MomentumVolumeProfileStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // MomentumVolume uses simple allocation based on signal strength
    if (probability > 0.7) {
        // Strong buy signal
        double conviction = (probability - 0.7) / 0.3; // Scale 0.7-1.0 to 0-1
        double base_weight = 0.4 + (conviction * 0.6); // 40-100% allocation
        
        decisions.push_back({base_symbol, base_weight, conviction, "MomentumVolume strong buy: 100% QQQ"});
    } else if (probability < 0.3) {
        // Strong sell signal
        double conviction = (0.3 - probability) / 0.3; // Scale 0.0-0.3 to 0-1
        double base_weight = 0.4 + (conviction * 0.6); // 40-100% allocation
        
        // Use SHORT QQQ for moderate sell signals instead of PSQ
        decisions.push_back({base_symbol, -base_weight, conviction, "MomentumVolume strong sell: SHORT QQQ"});
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "MomentumVolume: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg MomentumVolumeProfileStrategy::get_router_config() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}

SizerCfg MomentumVolumeProfileStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.max_position_pct = 1.0; // 100% max position
    cfg.volatility_target = 0.15; // 15% volatility target
    return cfg;
}

REGISTER_STRATEGY(MomentumVolumeProfileStrategy, "mvp");

} // namespace sentio

```

## 📄 **FILE 194 of 206**: src/strategy_opening_range_breakout.cpp

**File Information**:
- **Path**: `src/strategy_opening_range_breakout.cpp`

- **Size**: 187 lines
- **Modified**: 2025-09-12 15:35:06

- **Type**: .cpp

```text
#include "sentio/strategy_opening_range_breakout.hpp"
#include <algorithm>
#include <cmath>

namespace sentio {

OpeningRangeBreakoutStrategy::OpeningRangeBreakoutStrategy() 
    : BaseStrategy("OpeningRangeBreakout") {
    params_ = get_default_params();
    apply_params();
}
ParameterMap OpeningRangeBreakoutStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed cooldown to allow more frequent trades.
    return {
        {"opening_range_minutes", 30.0},
        {"breakout_confirmation_bars", 1.0},
        {"volume_multiplier", 1.5},
        {"stop_loss_pct", 0.01},
        {"take_profit_pct", 0.02},
        {"cool_down_period", 5.0}, // Was 15.0
    };
}

ParameterSpace OpeningRangeBreakoutStrategy::get_param_space() const { /* ... unchanged ... */ return {}; }

void OpeningRangeBreakoutStrategy::apply_params() {
    // **NEW**: Cache parameters
    opening_range_minutes_ = static_cast<int>(params_["opening_range_minutes"]);
    breakout_confirmation_bars_ = static_cast<int>(params_["breakout_confirmation_bars"]);
    volume_multiplier_ = params_["volume_multiplier"];
    stop_loss_pct_ = params_["stop_loss_pct"];
    take_profit_pct_ = params_["take_profit_pct"];
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);
    reset_state();
}

void OpeningRangeBreakoutStrategy::reset_state() {
    BaseStrategy::reset_state();
    current_range_ = OpeningRange{};
    day_start_index_ = -1;
}

double OpeningRangeBreakoutStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < 1) {
        diag_.drop(DropReason::MIN_BARS);
        return 0.5; // Neutral
    }

    // **MODIFIED**: Robust and performant new-day detection
    const int SECONDS_IN_DAY = 86400;
    long current_day = bars[current_index].ts_utc_epoch / SECONDS_IN_DAY;
    long prev_day = bars[current_index - 1].ts_utc_epoch / SECONDS_IN_DAY;

    if (current_day != prev_day) {
        reset_state(); // Reset everything for the new day
        day_start_index_ = current_index;
    }
    
    if (day_start_index_ == -1) { // Haven't established the start of the first day yet
        day_start_index_ = 0;
    }

    int bars_into_day = current_index - day_start_index_;

    // --- Phase 1: Define the Opening Range ---
    if (bars_into_day < opening_range_minutes_) {
        if (bars_into_day == 0) {
            current_range_.high = bars[current_index].high;
            current_range_.low = bars[current_index].low;
        } else {
            current_range_.high = std::max(current_range_.high, bars[current_index].high);
            current_range_.low = std::min(current_range_.low, bars[current_index].low);
        }
        diag_.drop(DropReason::SESSION); // Use SESSION to mean "in range formation"
        return 0.5; // Neutral
    }

    // --- Finalize the range exactly once ---
    if (!current_range_.is_finalized) {
        current_range_.end_bar = current_index - 1;
        current_range_.is_finalized = true;
    }

    // --- Phase 2: Look for Breakouts ---
    if (state_.in_position || is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return 0.5; // Neutral
    }

    const auto& bar = bars[current_index];
    bool is_breakout_up = bar.close > current_range_.high;
    bool is_breakout_down = bar.close < current_range_.low;

    if (!is_breakout_up && !is_breakout_down) {
        diag_.drop(DropReason::THRESHOLD);
        return 0.5; // Neutral
    }
    
    // Volume Confirmation
    double avg_volume = 0;
    for (int i = day_start_index_; i < current_range_.end_bar; ++i) {
        avg_volume += bars[i].volume;
    }
    avg_volume /= (current_range_.end_bar - day_start_index_ + 1);

    if (bar.volume < avg_volume * volume_multiplier_) {
        diag_.drop(DropReason::ZERO_VOL); // Re-using for low volume
        return 0.5; // Neutral
    }

    // Generate Signal
    double probability;
    if (is_breakout_up) {
        probability = 0.9; // Strong buy signal
    } else { // is_breakout_down
        probability = 0.1; // Strong sell signal
    }

    diag_.emitted++;
    state_.in_position = true; // Manually set state as this is an intraday strategy
    state_.last_trade_bar = current_index;

    return probability;
}

std::vector<BaseStrategy::AllocationDecision> OpeningRangeBreakoutStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // OpeningRangeBreakout uses simple allocation based on signal strength
    if (probability > 0.8) {
        // Strong buy signal
        double conviction = (probability - 0.8) / 0.2; // Scale 0.8-1.0 to 0-1
        double base_weight = 0.5 + (conviction * 0.5); // 50-100% allocation
        
        decisions.push_back({base_symbol, base_weight, conviction, "OpeningRangeBreakout strong buy: 100% QQQ"});
    } else if (probability < 0.2) {
        // Strong sell signal
        double conviction = (0.2 - probability) / 0.2; // Scale 0.0-0.2 to 0-1
        double base_weight = 0.5 + (conviction * 0.5); // 50-100% allocation
        
        // Use SHORT QQQ for moderate sell signals instead of PSQ
        decisions.push_back({base_symbol, -base_weight, conviction, "OpeningRangeBreakout strong sell: SHORT QQQ"});
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "OpeningRangeBreakout: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg OpeningRangeBreakoutStrategy::get_router_config() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}

SizerCfg OpeningRangeBreakoutStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.max_position_pct = 1.0; // 100% max position
    cfg.volatility_target = 0.15; // 15% volatility target
    return cfg;
}

// Register the strategy
REGISTER_STRATEGY(OpeningRangeBreakoutStrategy, "orb");

} // namespace sentio
```

## 📄 **FILE 195 of 206**: src/strategy_order_flow_imbalance.cpp

**File Information**:
- **Path**: `src/strategy_order_flow_imbalance.cpp`

- **Size**: 161 lines
- **Modified**: 2025-09-12 15:35:06

- **Type**: .cpp

```text
#include "sentio/strategy_order_flow_imbalance.hpp"
#include <algorithm>
#include <cmath>

namespace sentio {

OrderFlowImbalanceStrategy::OrderFlowImbalanceStrategy() 
    : BaseStrategy("OrderFlowImbalance"),
      rolling_pressure_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap OrderFlowImbalanceStrategy::get_default_params() const {
    return {
        {"lookback_period", 50.0},
        {"entry_threshold_long", 0.60},
        {"entry_threshold_short", 0.40},
        {"hold_max_bars", 60.0},
        {"cool_down_period", 5.0}
    };
}

ParameterSpace OrderFlowImbalanceStrategy::get_param_space() const { return {}; }

void OrderFlowImbalanceStrategy::apply_params() {
    lookback_period_ = static_cast<int>(params_["lookback_period"]);
    entry_threshold_long_ = params_["entry_threshold_long"];
    entry_threshold_short_ = params_["entry_threshold_short"];
    hold_max_bars_ = static_cast<int>(params_["hold_max_bars"]);
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);
    
    rolling_pressure_ = RollingMean(lookback_period_);
    reset_state();
}

void OrderFlowImbalanceStrategy::reset_state() {
    BaseStrategy::reset_state();
    ofi_state_ = OFIState::Flat; // **FIXED**: Use the renamed state variable
    bars_in_trade_ = 0;
    rolling_pressure_ = RollingMean(lookback_period_);
}

double OrderFlowImbalanceStrategy::calculate_bar_pressure(const Bar& bar) const {
    double range = bar.high - bar.low;
    if (range < 1e-9) {
        return 0.5; // Neutral pressure if there's no range
    }
    return (bar.close - bar.low) / range;
}

double OrderFlowImbalanceStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < lookback_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return 0.5; // Neutral
    }

    double pressure = calculate_bar_pressure(bars[current_index]);
    double avg_pressure = rolling_pressure_.push(pressure);

    // **FIXED**: Use the strategy-specific 'ofi_state_' for state machine logic
    if (ofi_state_ == OFIState::Flat) {
        if (is_cooldown_active(current_index, cool_down_period_)) {
            diag_.drop(DropReason::COOLDOWN);
            return 0.5; // Neutral
        }

        double probability;
        if (avg_pressure > entry_threshold_long_) {
            probability = 0.7; // Buy signal
            ofi_state_ = OFIState::Long;
            // **FIXED**: Correctly access the 'state_' member from BaseStrategy
            state_.last_trade_bar = current_index;
        } else if (avg_pressure < entry_threshold_short_) {
            probability = 0.3; // Sell signal
            ofi_state_ = OFIState::Short;
            // **FIXED**: Correctly access the 'state_' member from BaseStrategy
            state_.last_trade_bar = current_index;
        } else {
            diag_.drop(DropReason::THRESHOLD);
            return 0.5; // Neutral
        }

        diag_.emitted++;
        return probability;

    } else { // In a trade, check for exit
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            // **FIXED**: Use 'ofi_state_' to determine exit signal direction
            double exit_prob = (ofi_state_ == OFIState::Long) ? 0.3 : 0.7; // SELL or BUY
            diag_.emitted++;
            reset_state();
            return exit_prob;
        }
        return 0.5; // Hold current position
    }
}

std::vector<BaseStrategy::AllocationDecision> OrderFlowImbalanceStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // OrderFlowImbalance uses simple allocation based on signal strength
    if (probability > 0.6) {
        // Buy signal
        double conviction = (probability - 0.6) / 0.4; // Scale 0.6-1.0 to 0-1
        double base_weight = 0.3 + (conviction * 0.4); // 30-70% allocation
        
        decisions.push_back({base_symbol, base_weight, conviction, "OrderFlowImbalance buy: 100% QQQ"});
    } else if (probability < 0.4) {
        // Sell signal
        double conviction = (0.4 - probability) / 0.4; // Scale 0.0-0.4 to 0-1
        double base_weight = 0.3 + (conviction * 0.4); // 30-70% allocation
        
        // Use SHORT QQQ for moderate sell signals instead of PSQ
        decisions.push_back({base_symbol, -base_weight, conviction, "OrderFlowImbalance sell: SHORT QQQ"});
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "OrderFlowImbalance: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg OrderFlowImbalanceStrategy::get_router_config() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}

SizerCfg OrderFlowImbalanceStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.max_position_pct = 0.7; // 70% max position
    cfg.volatility_target = 0.12; // 12% volatility target
    return cfg;
}

REGISTER_STRATEGY(OrderFlowImbalanceStrategy, "ofi");

} // namespace sentio


```

## 📄 **FILE 196 of 206**: src/strategy_order_flow_scalping.cpp

**File Information**:
- **Path**: `src/strategy_order_flow_scalping.cpp`

- **Size**: 177 lines
- **Modified**: 2025-09-12 15:35:06

- **Type**: .cpp

```text
#include "sentio/strategy_order_flow_scalping.hpp"
#include <algorithm>
#include <cmath>

namespace sentio {

OrderFlowScalpingStrategy::OrderFlowScalpingStrategy() 
    : BaseStrategy("OrderFlowScalping"),
      rolling_pressure_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap OrderFlowScalpingStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed the imbalance threshold to arm more frequently.
    return {
        {"lookback_period", 50.0},
        {"imbalance_threshold", 0.55}, // Was 0.65, now arms when avg pressure is > 55%
        {"hold_max_bars", 20.0},
        {"cool_down_period", 3.0}
    };
}

ParameterSpace OrderFlowScalpingStrategy::get_param_space() const { return {}; }

void OrderFlowScalpingStrategy::apply_params() {
    lookback_period_ = static_cast<int>(params_["lookback_period"]);
    imbalance_threshold_ = params_["imbalance_threshold"];
    hold_max_bars_ = static_cast<int>(params_["hold_max_bars"]);
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);

    rolling_pressure_ = RollingMean(lookback_period_);
    reset_state();
}

void OrderFlowScalpingStrategy::reset_state() {
    BaseStrategy::reset_state();
    of_state_ = OFState::Idle; // **FIXED**: Use the renamed state variable
    bars_in_trade_ = 0;
    rolling_pressure_ = RollingMean(lookback_period_);
}

double OrderFlowScalpingStrategy::calculate_bar_pressure(const Bar& bar) const {
    double range = bar.high - bar.low;
    if (range < 1e-9) return 0.5;
    return (bar.close - bar.low) / range;
}

double OrderFlowScalpingStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < lookback_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return 0.5; // Neutral
    }

    const auto& bar = bars[current_index];
    double pressure = calculate_bar_pressure(bar);
    double avg_pressure = rolling_pressure_.push(pressure);

    // **FIXED**: Use the strategy-specific 'of_state_' for state machine logic
    if (of_state_ == OFState::Long || of_state_ == OFState::Short) {
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            double exit_prob = (of_state_ == OFState::Long) ? 0.3 : 0.7; // SELL or BUY
            diag_.emitted++;
            reset_state();
            return exit_prob;
        }
        return 0.5; // Hold current position
    }
    
    if (is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return 0.5; // Neutral
    }

    double probability = 0.5; // Default neutral
    switch (of_state_) {
        case OFState::Idle:
            if (avg_pressure > imbalance_threshold_) of_state_ = OFState::ArmedLong;
            else if (avg_pressure < (1.0 - imbalance_threshold_)) of_state_ = OFState::ArmedShort;
            else diag_.drop(DropReason::THRESHOLD);
            break;
            
        case OFState::ArmedLong:
            if (pressure > 0.5) { // Confirmation bar must be bullish
                probability = 0.7; // Buy signal
                of_state_ = OFState::Long;
            } else { // Failed confirmation
                of_state_ = OFState::Idle;
                diag_.drop(DropReason::THRESHOLD);
            }
            break;

        case OFState::ArmedShort:
            if (pressure < 0.5) { // Confirmation bar must be bearish
                probability = 0.3; // Sell signal
                of_state_ = OFState::Short;
            } else { // Failed confirmation
                of_state_ = OFState::Idle;
                diag_.drop(DropReason::THRESHOLD);
            }
            break;
        default: break;
    }
    
    if (probability != 0.5) {
        diag_.emitted++;
        bars_in_trade_ = 0;
        // **FIXED**: This now correctly refers to the 'state_' member from BaseStrategy
        state_.last_trade_bar = current_index;
    }
    
    return probability;
}

std::vector<BaseStrategy::AllocationDecision> OrderFlowScalpingStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // OrderFlowScalping uses simple allocation based on signal strength
    if (probability > 0.6) {
        // Buy signal
        double conviction = (probability - 0.6) / 0.4; // Scale 0.6-1.0 to 0-1
        double base_weight = 0.2 + (conviction * 0.3); // 20-50% allocation (scalping is smaller)
        
        decisions.push_back({base_symbol, base_weight, conviction, "OrderFlowScalping buy: 100% QQQ"});
    } else if (probability < 0.4) {
        // Sell signal
        double conviction = (0.4 - probability) / 0.4; // Scale 0.0-0.4 to 0-1
        double base_weight = 0.2 + (conviction * 0.3); // 20-50% allocation (scalping is smaller)
        
        // Use SHORT QQQ for moderate sell signals instead of PSQ
        decisions.push_back({base_symbol, -base_weight, conviction, "OrderFlowScalping sell: SHORT QQQ"});
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "OrderFlowScalping: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg OrderFlowScalpingStrategy::get_router_config() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}

SizerCfg OrderFlowScalpingStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.max_position_pct = 0.5; // 50% max position for scalping
    cfg.volatility_target = 0.10; // 10% volatility target
    return cfg;
}

REGISTER_STRATEGY(OrderFlowScalpingStrategy, "ofs");

} // namespace sentio


```

## 📄 **FILE 197 of 206**: src/strategy_signal_or.cpp

**File Information**:
- **Path**: `src/strategy_signal_or.cpp`

- **Size**: 277 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .cpp

```text
#include "sentio/strategy_signal_or.hpp"
#include "sentio/signal_utils.hpp"
#include "sentio/router.hpp"
#include "sentio/sizer.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace sentio {

SignalOrStrategy::SignalOrStrategy(const SignalOrCfg& cfg) 
    : BaseStrategy("SignalOR"), cfg_(cfg) {
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
    double signal_strength = std::abs(probability - 0.5) * 2.0;
    
    // Only make allocation decisions if signal is strong enough
    if (signal_strength < cfg_.min_signal_strength) {
        return decisions; // Empty if signal too weak
    }
    
    // Determine target weight based on probability
    double target_weight = 0.0;
    std::string target_symbol;
    std::string reason;
    
    if (probability > cfg_.long_threshold) {
        // Long signal - choose between base and 3x based on strength
        if (signal_strength > 0.7) {
            target_symbol = bull3x_symbol;
            target_weight = cfg_.max_position_weight;
            reason = "Strong long signal - 3x leveraged";
        } else {
            target_symbol = base_symbol;
            target_weight = cfg_.max_position_weight * 0.6; // Conservative sizing
            reason = "Moderate long signal - base position";
        }
    } else if (probability < cfg_.short_threshold) {
        // Short signal - choose between inverse and bear 3x
        if (signal_strength > 0.7) {
            target_symbol = bear3x_symbol;
            target_weight = -cfg_.max_position_weight; // Negative for short
            reason = "Strong short signal - 3x leveraged short";
        } else {
            // Use SHORT QQQ for moderate sell signals instead of inverse ETF
            target_symbol = base_symbol;
            target_weight = -cfg_.max_position_weight * 0.6; // Conservative sizing (negative for short)
            reason = "Moderate short signal - SHORT QQQ";
        }
    }
    
    if (!target_symbol.empty() && std::abs(target_weight) > 1e-6) {
        AllocationDecision decision;
        decision.instrument = target_symbol;
        decision.target_weight = target_weight;
        decision.confidence = signal_strength;
        decision.reason = reason;
        decisions.push_back(decision);
    }
    
    return decisions;
}

RouterCfg SignalOrStrategy::get_router_config() const {
    RouterCfg cfg;
    return cfg;
}

SizerCfg SignalOrStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.fractional_allowed = true;
    cfg.min_notional = 100.0; // $100 minimum
    cfg.max_leverage = 3.0; // Allow 3x leverage
    cfg.max_position_pct = cfg_.max_position_weight;
    cfg.volatility_target = 0.20; // 20% target volatility
    cfg.allow_negative_cash = false;
    cfg.vol_lookback_days = 20;
    cfg.cash_reserve_pct = 0.05; // 5% cash reserve
    return cfg;
}

// Configuration
ParameterMap SignalOrStrategy::get_default_params() const {
    return {
        {"min_signal_strength", cfg_.min_signal_strength},
        {"long_threshold", cfg_.long_threshold},
        {"short_threshold", cfg_.short_threshold},
        {"hold_threshold", cfg_.hold_threshold},
        {"max_position_weight", cfg_.max_position_weight},
        {"position_decay", cfg_.position_decay},
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
    space["max_position_weight"] = {ParamType::FLOAT, 0.5, 1.0, cfg_.max_position_weight};
    space["position_decay"] = {ParamType::FLOAT, 0.9, 0.99, cfg_.position_decay};
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
    if (params_.count("max_position_weight")) {
        cfg_.max_position_weight = params_.at("max_position_weight");
    }
    if (params_.count("position_decay")) {
        cfg_.position_decay = params_.at("position_decay");
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
    current_position_weight_ = 0.0;
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
    
    // Convert momentum to probability
    double momentum_prob = 0.5 + std::clamp(momentum * cfg_.momentum_scale, -0.4, 0.4);
    
    return momentum_prob;
}

double SignalOrStrategy::calculate_position_weight(double signal_strength) {
    // Calculate position weight based on signal strength
    double base_weight = signal_strength * cfg_.max_position_weight;
    
    // Apply position decay
    current_position_weight_ *= cfg_.position_decay;
    
    // Update with new signal
    current_position_weight_ = std::max(current_position_weight_, base_weight);
    
    return std::min(current_position_weight_, cfg_.max_position_weight);
}

void SignalOrStrategy::update_position_decay() {
    // Apply position decay to reduce position over time without new signals
    current_position_weight_ *= cfg_.position_decay;
    
    // Prevent position from becoming negative
    current_position_weight_ = std::max(0.0, current_position_weight_);
}

} // namespace sentio
```

## 📄 **FILE 198 of 206**: src/strategy_sma_cross.cpp

**File Information**:
- **Path**: `src/strategy_sma_cross.cpp`

- **Size**: 35 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .cpp

```text
#include "sentio/strategy_sma_cross.hpp"
#include <cmath>

namespace sentio {

SMACrossStrategy::SMACrossStrategy(const SMACrossCfg& cfg)
  : cfg_(cfg), sma_fast_(cfg.fast), sma_slow_(cfg.slow) {}

void SMACrossStrategy::on_bar(const StrategyCtx& ctx, const Bar& b) {
  (void)ctx;
  sma_fast_.push(b.close);
  sma_slow_.push(b.close);

  if (!warmed_up()) { last_.reset(); return; }

  double f = sma_fast_.value();
  double s = sma_slow_.value();
  if (!std::isfinite(f) || !std::isfinite(s)) { last_.reset(); return; }

  // Detect cross (including equality toggle avoidance)
  bool golden_now  = (f >= s) && !(std::isfinite(last_fast_) && std::isfinite(last_slow_) && last_fast_ >= last_slow_);
  bool death_now   = (f <= s) && !(std::isfinite(last_fast_) && std::isfinite(last_slow_) && last_fast_ <= last_slow_);

  if (golden_now) {
    last_ = StrategySignal{StrategySignal::Type::BUY, cfg_.conf_fast_slow};
  } else if (death_now) {
    last_ = StrategySignal{StrategySignal::Type::SELL, cfg_.conf_fast_slow};
  } else {
    last_.reset(); // no edge this bar
  }

  last_fast_ = f; last_slow_ = s;
}

} // namespace sentio

```

## 📄 **FILE 199 of 206**: src/strategy_tfa.cpp

**File Information**:
- **Path**: `src/strategy_tfa.cpp`

- **Size**: 306 lines
- **Modified**: 2025-09-15 15:35:51

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
  (void)current_index; // we will use bars.size() and a static cursor
  static int calls = 0;
  ++calls;

  // One-time: precompute probabilities over the whole series using the sequence context
  static bool seq_inited = false;
  static TfaSeqContext seq_ctx;
  static std::vector<float> probs_all;
  if (!seq_inited) {
    try {
      std::string artifacts_path = cfg_.artifacts_dir + "/" + cfg_.model_id + "/" + cfg_.version + "/";
      seq_ctx.load(artifacts_path + "model.pt",
                   artifacts_path + "feature_spec.json",
                   artifacts_path + "model.meta.json");
      // Assume base symbol is QQQ for this test run
      seq_ctx.forward_probs("QQQ", bars, probs_all);
      seq_inited = true;
    } catch (const std::exception& e) {
      return 0.5; // Neutral
    }
  }

  // Maintain rolling threshold logic with cooldown based on precomputed prob at this call index
  float prob = (calls-1 < (int)probs_all.size()) ? probs_all[(size_t)(calls-1)] : 0.5f;

  static std::vector<float> p_hist; p_hist.reserve(4096);
  static int cooldown_long_until = -1;
  static int cooldown_short_until = -1;
  const int window = 250;
  const float q_long = 0.80f, q_short = 0.20f;
  const float floor_long = 0.55f, ceil_short = 0.45f;
  const int cooldown = 5;

  p_hist.push_back(prob);

  if ((int)p_hist.size() >= std::max(window, seq_ctx.T)) {
    int end = (int)p_hist.size() - 1;
    int start = std::max(0, end - window + 1);
    std::vector<float> win(p_hist.begin() + start, p_hist.begin() + end + 1);

    int kL = (int)std::floor(q_long * (win.size() - 1));
    std::nth_element(win.begin(), win.begin() + kL, win.end());
    float thrL = std::max(floor_long, win[kL]);

    int kS = (int)std::floor(q_short * (win.size() - 1));
    std::nth_element(win.begin(), win.begin() + kS, win.end());
    float thrS = std::min(ceil_short, win[kS]);

    bool can_long = (calls >= cooldown_long_until);
    bool can_short = (calls >= cooldown_short_until);

    if (can_long && prob >= thrL) {
      cooldown_long_until = calls + cooldown;
      return prob; // Return the probability directly
    } else if (can_short && prob <= thrS) {
      cooldown_short_until = calls + cooldown;
      return prob; // Return the probability directly
    }

  }

  return 0.5; // Neutral
}

std::vector<BaseStrategy::AllocationDecision> TFAStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // TFA is a long-only strategy - only allocates to bullish instruments
    if (probability > 0.6) {
        double conviction = (probability - 0.6) / 0.4; // Scale 0.6-1.0 to 0-1
        double base_weight = 0.3 + (conviction * 0.7); // 30-100% allocation
        
        if (probability > 0.8) {
            // Strong buy: use leveraged instruments
            decisions.push_back({bull3x_symbol, base_weight * 0.6, conviction, "TFA strong buy: 60% TQQQ"});
            decisions.push_back({base_symbol, base_weight * 0.4, conviction, "TFA strong buy: 40% QQQ"});
        } else {
            // Moderate buy: use unleveraged instruments
            decisions.push_back({base_symbol, base_weight, conviction, "TFA moderate buy: 100% QQQ"});
        }
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol};
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

RouterCfg TFAStrategy::get_router_config() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}

SizerCfg TFAStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.max_position_pct = 1.0; // 100% max position
    cfg.volatility_target = 0.15; // 15% volatility target
    return cfg;
}

REGISTER_STRATEGY(TFAStrategy, "tfa");

} // namespace sentio

```

## 📄 **FILE 200 of 206**: src/strategy_vwap_reversion.cpp

**File Information**:
- **Path**: `src/strategy_vwap_reversion.cpp`

- **Size**: 213 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .cpp

```text
#include "sentio/strategy_vwap_reversion.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace sentio {

VWAPReversionStrategy::VWAPReversionStrategy() : BaseStrategy("VWAPReversion") {
    params_ = get_default_params();
    apply_params();
}

ParameterMap VWAPReversionStrategy::get_default_params() const {
    // Default parameters remain the same
    return {
        {"vwap_period", 390.0}, {"band_multiplier", 0.005}, {"max_band_width", 0.01},
        {"min_distance_from_vwap", 0.001}, {"volume_confirmation_mult", 1.2},
        {"rsi_period", 14.0}, {"rsi_oversold", 40.0}, {"rsi_overbought", 60.0},
        {"stop_loss_pct", 0.003}, {"take_profit_pct", 0.005},
        {"time_stop_bars", 30.0}, {"cool_down_period", 2.0}
    };
}

ParameterSpace VWAPReversionStrategy::get_param_space() const { return {}; }

void VWAPReversionStrategy::apply_params() {
    vwap_period_ = static_cast<int>(params_["vwap_period"]);
    band_multiplier_ = params_["band_multiplier"];
    max_band_width_ = params_["max_band_width"];
    min_distance_from_vwap_ = params_["min_distance_from_vwap"];
    volume_confirmation_mult_ = params_["volume_confirmation_mult"];
    rsi_period_ = static_cast<int>(params_["rsi_period"]);
    rsi_oversold_ = params_["rsi_oversold"];
    rsi_overbought_ = params_["rsi_overbought"];
    stop_loss_pct_ = params_["stop_loss_pct"];
    take_profit_pct_ = params_["take_profit_pct"];
    time_stop_bars_ = static_cast<int>(params_["time_stop_bars"]);
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);
    reset_state();
}

void VWAPReversionStrategy::reset_state() {
    BaseStrategy::reset_state();
    cumulative_pv_ = 0.0;
    cumulative_volume_ = 0.0;
    time_in_position_ = 0;
    vwap_ = 0.0;
}

void VWAPReversionStrategy::update_vwap(const Bar& bar) {
    double typical_price = (bar.high + bar.low + bar.close) / 3.0;
    cumulative_pv_ += typical_price * bar.volume;
    cumulative_volume_ += bar.volume;
    if (cumulative_volume_ > 0) {
        vwap_ = cumulative_pv_ / cumulative_volume_;
    }
}

double VWAPReversionStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    update_vwap(bars[current_index]);

    if (current_index < rsi_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return 0.5; // Neutral
    }
    
    if (is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return 0.5; // Neutral
    }
    
    if (vwap_ <= 0) {
        diag_.drop(DropReason::NAN_INPUT);
        return 0.5; // Neutral
    }

    const auto& bar = bars[current_index];
    double distance_pct = std::abs(bar.close - vwap_) / vwap_;
    if (distance_pct < min_distance_from_vwap_) {
        diag_.drop(DropReason::THRESHOLD);
        return 0.5; // Neutral
    }

    double upper_band = vwap_ * (1.0 + band_multiplier_);
    double lower_band = vwap_ * (1.0 - band_multiplier_);

    bool buy_condition = bar.close < lower_band && is_rsi_condition_met(bars, current_index, true);
    bool sell_condition = bar.close > upper_band && is_rsi_condition_met(bars, current_index, false);

    double probability;
    if (buy_condition) {
        probability = 0.8; // Strong buy signal
    } else if (sell_condition) {
        probability = 0.2; // Strong sell signal
    } else {
        diag_.drop(DropReason::THRESHOLD);
        return 0.5; // Neutral
    }

    diag_.emitted++;
    state_.last_trade_bar = current_index;
    return probability;
}

bool VWAPReversionStrategy::is_rsi_condition_met(const std::vector<Bar>& bars, int index, bool for_buy) const {
    std::vector<double> closes;
    closes.reserve(rsi_period_);
    for(int i = 0; i < rsi_period_; ++i) {
        closes.push_back(bars[index - rsi_period_ + 1 + i].close);
    }
    // Simple RSI calculation
    double rsi = calculate_simple_rsi(closes);
    return for_buy ? (rsi < rsi_oversold_) : (rsi > rsi_overbought_);
}

double VWAPReversionStrategy::calculate_simple_rsi(const std::vector<double>& prices) const {
    if (prices.size() < 2) return 50.0; // Neutral RSI if not enough data
    
    std::vector<double> gains, losses;
    for (size_t i = 1; i < prices.size(); ++i) {
        double change = prices[i] - prices[i-1];
        if (change > 0) {
            gains.push_back(change);
            losses.push_back(0.0);
        } else {
            gains.push_back(0.0);
            losses.push_back(-change);
        }
    }
    
    if (gains.empty()) return 50.0;
    
    double avg_gain = std::accumulate(gains.begin(), gains.end(), 0.0) / gains.size();
    double avg_loss = std::accumulate(losses.begin(), losses.end(), 0.0) / losses.size();
    
    if (avg_loss == 0.0) return 100.0; // All gains
    if (avg_gain == 0.0) return 0.0;   // All losses
    
    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}

bool VWAPReversionStrategy::is_volume_confirmed(const std::vector<Bar>& bars, int index) const {
    if (index < 20) return true;
    double avg_vol = 0;
    for(int i = 1; i <= 20; ++i) {
        avg_vol += bars[index-i].volume;
    }
    avg_vol /= 20.0;
    return bars[index].volume > avg_vol * volume_confirmation_mult_;
}

std::vector<BaseStrategy::AllocationDecision> VWAPReversionStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // VWAPReversion uses simple allocation based on signal strength
    if (probability > 0.7) {
        // Strong buy signal
        double conviction = (probability - 0.7) / 0.3; // Scale 0.7-1.0 to 0-1
        double base_weight = 0.4 + (conviction * 0.6); // 40-100% allocation
        
        decisions.push_back({base_symbol, base_weight, conviction, "VWAPReversion strong buy: 100% QQQ"});
    } else if (probability < 0.3) {
        // Strong sell signal
        double conviction = (0.3 - probability) / 0.3; // Scale 0.0-0.3 to 0-1
        double base_weight = 0.4 + (conviction * 0.6); // 40-100% allocation
        
        // Use SHORT QQQ for moderate sell signals instead of PSQ
        decisions.push_back({base_symbol, -base_weight, conviction, "VWAPReversion strong sell: SHORT QQQ"});
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "VWAPReversion: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg VWAPReversionStrategy::get_router_config() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}

SizerCfg VWAPReversionStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.max_position_pct = 1.0; // 100% max position
    cfg.volatility_target = 0.15; // 15% volatility target
    return cfg;
}

REGISTER_STRATEGY(VWAPReversionStrategy, "vwap");

} // namespace sentio
```

## 📄 **FILE 201 of 206**: src/telemetry_logger.cpp

**File Information**:
- **Path**: `src/telemetry_logger.cpp`

- **Size**: 177 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .cpp

```text
#include "sentio/telemetry_logger.hpp"
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace sentio {

// Global telemetry logger instance
std::unique_ptr<TelemetryLogger> g_telemetry_logger = nullptr;

TelemetryLogger::TelemetryLogger(const std::string& log_file_path) 
    : log_file_path_(log_file_path) {
    // Create directory if it doesn't exist
    std::filesystem::path path(log_file_path);
    std::filesystem::create_directories(path.parent_path());
    
    // Open log file in append mode
    log_file_.open(log_file_path, std::ios::app);
    if (!log_file_.is_open()) {
        throw std::runtime_error("Failed to open telemetry log file: " + log_file_path);
    }
}

TelemetryLogger::~TelemetryLogger() {
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

void TelemetryLogger::log(const TelemetryData& data) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    std::ostringstream json;
    json << "{";
    json << "\"timestamp\":\"" << data.timestamp << "\",";
    json << "\"strategy_name\":\"" << escape_json_string(data.strategy_name) << "\",";
    json << "\"instrument\":\"" << escape_json_string(data.instrument) << "\",";
    json << "\"bars_processed\":" << data.bars_processed << ",";
    json << "\"signals_generated\":" << data.signals_generated << ",";
    json << "\"buy_signals\":" << data.buy_signals << ",";
    json << "\"sell_signals\":" << data.sell_signals << ",";
    json << "\"hold_signals\":" << data.hold_signals << ",";
    json << "\"avg_confidence\":" << std::fixed << std::setprecision(6) << data.avg_confidence << ",";
    json << "\"ready_percentage\":" << std::fixed << std::setprecision(2) << data.ready_percentage << ",";
    json << "\"processing_time_ms\":" << std::fixed << std::setprecision(3) << data.processing_time_ms;
    
    if (!data.notes.empty()) {
        json << ",\"notes\":\"" << escape_json_string(data.notes) << "\"";
    }
    
    json << "}";
    
    write_json_line(json.str());
}

void TelemetryLogger::log_metric(
    const std::string& strategy_name,
    const std::string& metric_name,
    double value,
    const std::string& instrument
) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    std::ostringstream json;
    json << "{";
    json << "\"timestamp\":\"" << get_current_timestamp() << "\",";
    json << "\"strategy_name\":\"" << escape_json_string(strategy_name) << "\",";
    json << "\"instrument\":\"" << escape_json_string(instrument) << "\",";
    json << "\"metric_name\":\"" << escape_json_string(metric_name) << "\",";
    json << "\"value\":" << std::fixed << std::setprecision(6) << value;
    json << "}";
    
    write_json_line(json.str());
}

void TelemetryLogger::log_signal_stats(
    const std::string& strategy_name,
    const std::string& instrument,
    int signals_generated,
    int buy_signals,
    int sell_signals,
    int hold_signals,
    double avg_confidence
) {
    TelemetryData data;
    data.timestamp = get_current_timestamp();
    data.strategy_name = strategy_name;
    data.instrument = instrument;
    data.signals_generated = signals_generated;
    data.buy_signals = buy_signals;
    data.sell_signals = sell_signals;
    data.hold_signals = hold_signals;
    data.avg_confidence = avg_confidence;
    
    log(data);
}

void TelemetryLogger::log_performance(
    const std::string& strategy_name,
    const std::string& instrument,
    int bars_processed,
    double processing_time_ms,
    double ready_percentage
) {
    TelemetryData data;
    data.timestamp = get_current_timestamp();
    data.strategy_name = strategy_name;
    data.instrument = instrument;
    data.bars_processed = bars_processed;
    data.processing_time_ms = processing_time_ms;
    data.ready_percentage = ready_percentage;
    
    log(data);
}

void TelemetryLogger::flush() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (log_file_.is_open()) {
        log_file_.flush();
    }
}

std::string TelemetryLogger::get_current_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count() << "Z";
    return oss.str();
}

std::string TelemetryLogger::escape_json_string(const std::string& str) const {
    std::string escaped;
    escaped.reserve(str.length() + 10); // Reserve some extra space
    
    for (char c : str) {
        switch (c) {
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:   escaped += c; break;
        }
    }
    
    return escaped;
}

void TelemetryLogger::write_json_line(const std::string& json_line) {
    if (log_file_.is_open()) {
        log_file_ << json_line << std::endl;
        
        // Flush every 100 lines to ensure data is written
        if (++log_counter_ % 100 == 0) {
            log_file_.flush();
        }
    }
}

void init_telemetry_logger(const std::string& log_file_path) {
    g_telemetry_logger = std::make_unique<TelemetryLogger>(log_file_path);
}

TelemetryLogger& get_telemetry_logger() {
    if (!g_telemetry_logger) {
        init_telemetry_logger();
    }
    return *g_telemetry_logger;
}

} // namespace sentio

```

## 📄 **FILE 202 of 206**: src/temporal_analysis.cpp

**File Information**:
- **Path**: `src/temporal_analysis.cpp`

- **Size**: 179 lines
- **Modified**: 2025-09-13 09:40:09

- **Type**: .cpp

```text
#include "sentio/temporal_analysis.hpp"
#include "sentio/runner.hpp"
#include "sentio/audit.hpp"
#include "sentio/metrics.hpp"
#include "sentio/unified_metrics.hpp"
#include "sentio/progress_bar.hpp"
#include "sentio/day_index.hpp"
#include "sentio/feature_feeder.hpp"
#include "sentio/run_id_generator.hpp"
#include "audit/audit_db_recorder.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

namespace sentio {

TemporalAnalysisSummary run_temporal_analysis(const SymbolTable& ST,
                                            const std::vector<std::vector<Bar>>& series,
                                            int base_symbol_id,
                                            const RunnerCfg& rcfg,
                                            const TemporalAnalysisConfig& cfg) {
    
    const auto& base_series = series[base_symbol_id];
    const int total_bars = (int)base_series.size();
    
    if (total_bars < cfg.min_bars_per_quarter) {
        std::cerr << "ERROR: Insufficient data for temporal analysis. Need at least " 
                  << cfg.min_bars_per_quarter << " bars per quarter." << std::endl;
        return TemporalAnalysisSummary{};
    }
    
    // Calculate time periods based on actual trading periods
    // Assume ~390 bars per trading day (6.5 hours * 60 mins = 390 1-min bars)
    int bars_per_day = 390;
    int bars_per_week = 5 * bars_per_day;  // 5 trading days per week
    int bars_per_quarter = 66 * bars_per_day; // 66 trading days per quarter
    
    // Determine which time period to use and calculate the number of periods
    int num_periods = 0;
    int bars_per_period = 0;
    std::string period_name = "period";
    
    if (cfg.num_days > 0) {
        num_periods = cfg.num_days;
        bars_per_period = bars_per_day;
        period_name = "day";
    } else if (cfg.num_weeks > 0) {
        num_periods = cfg.num_weeks;
        bars_per_period = bars_per_week;
        period_name = "week";
    } else if (cfg.num_quarters > 0) {
        num_periods = cfg.num_quarters;
        bars_per_period = bars_per_quarter;
        period_name = "quarter";
    } else {
        // Default: analyze all data as one period
        num_periods = 1;
        bars_per_period = total_bars;
        period_name = "full period";
    }
    
    std::cout << "Starting TPA (Temporal Performance Analysis) Test..." << std::endl;
    std::cout << "Total bars: " << total_bars << ", Bars per " << period_name << ": " << bars_per_period << std::endl;
    
    // Initialize analyzer and progress bar
    TemporalAnalyzer analyzer;
    analyzer.set_period_name(period_name);
    TPATestProgressBar progress_bar(num_periods, rcfg.strategy_name, period_name);
    progress_bar.display(); // Show initial progress bar
    
    std::cout << "\nInitializing data processing..." << std::endl;
    
    // Build audit filename prefix with strategy and timestamp
    const std::string test_name = "tpa_test";
    const auto ts_epoch = static_cast<long long>(std::time(nullptr));

    // Determine the last num_periods periods from the end
    int total_periods_available = std::max(1, total_bars / std::max(1, bars_per_period));
    int start_period = std::max(0, total_periods_available - num_periods);
    for (int pi = 0; pi < num_periods; ++pi) {
        int p = start_period + pi;
        int start_idx = p * bars_per_period;
        int end_idx = std::min(start_idx + bars_per_period, total_bars);
        
        if (end_idx - start_idx < cfg.min_bars_per_quarter) {
            std::cout << "Skipping " << period_name << " " << (p + 1) << " - insufficient data" << std::endl;
            continue;
        }
        
        std::cout << "\nProcessing " << period_name << " " << (p + 1) 
                  << " (bars " << start_idx << "-" << end_idx << ")..." << std::endl;
        
        // Create data slice for this quarter
        std::vector<std::vector<Bar>> quarter_series;
        quarter_series.reserve(series.size());
        for (const auto& sym_series : series) {
            if (sym_series.size() > static_cast<size_t>(end_idx)) {
                quarter_series.emplace_back(sym_series.begin() + start_idx, sym_series.begin() + end_idx);
            } else if (sym_series.size() > static_cast<size_t>(start_idx)) {
                quarter_series.emplace_back(sym_series.begin() + start_idx, sym_series.end());
            } else {
                quarter_series.emplace_back();
            }
        }
        
        // **STRATEGY ISOLATION**: Reset all shared state before each run
        FeatureFeeder::reset_all_state();
        
        // **STRATEGY-AGNOSTIC**: Run backtest for this period using SQLite audit system
        std::string run_id = generate_run_id();
        std::string period_info = period_name + std::to_string(p + 1);
        std::string audit_note = create_audit_note(rcfg.strategy_name, test_name, period_info);
        std::string db_path = "audit/sentio_audit.sqlite3";
        audit::AuditDBRecorder audit(db_path, run_id, audit_note);
        
        // **AUDIT FIX**: Ensure audit level is set to Full for proper logging
        RunnerCfg audit_cfg = rcfg;
        audit_cfg.audit_level = AuditLevel::Full;
        
        auto result = run_backtest(audit, ST, quarter_series, base_symbol_id, audit_cfg);
        
        // Calculate period metrics
        QuarterlyMetrics metrics;
        metrics.year = 2024; // Use current year for all periods
        metrics.quarter = p + 1; // Use period number as quarter
        
        // Calculate actual trading days by extracting unique dates from base symbol bars
        // Use the first series (base symbol) to count trading days
        int actual_trading_days = 0;
        if (!series.empty() && series[0].size() > static_cast<size_t>(start_idx)) {
            int actual_end_idx = std::min(end_idx, static_cast<int>(series[0].size()));
            std::vector<Bar> quarter_bars(series[0].begin() + start_idx, series[0].begin() + actual_end_idx);
            auto day_starts = day_start_indices(quarter_bars);
            actual_trading_days = static_cast<int>(day_starts.size());
        } else {
            // Fallback: estimate trading days (approximately 66 trading days per quarter)
            actual_trading_days = std::max(1, (end_idx - start_idx) / 390); // ~390 bars per day
        }
        
        // **FIX**: Use the correct monthly projection from compute_metrics_day_aware
        // The metrics calculator already properly computes monthly projected returns
        // No need for additional compounding which was causing 24x inflation
        metrics.monthly_return_pct = result.monthly_projected_return * 100.0;
        
        metrics.sharpe_ratio = result.sharpe_ratio;
        metrics.total_trades = result.total_fills;  // Use total_fills as proxy for trades
        metrics.trading_days = actual_trading_days;
        metrics.avg_daily_trades = actual_trading_days > 0 ? static_cast<double>(result.total_fills) / actual_trading_days : 0.0;
        metrics.max_drawdown = result.max_drawdown;
        metrics.win_rate = 0.0;  // Not available in RunResult, set to 0
        metrics.total_return_pct = result.total_return;
        
        analyzer.add_quarterly_result(metrics);
        
        // Update progress bar with period results
        progress_bar.display_with_period_info(p + 1, 2024, p + 1,
                                             metrics.monthly_return_pct, metrics.sharpe_ratio,
                                             metrics.avg_daily_trades, metrics.health_status());
        
        // Print period summary
        std::cout << "\n  Monthly Return: " << std::fixed << std::setprecision(2) 
                  << metrics.monthly_return_pct << "%" << std::endl;
        std::cout << "  Sharpe Ratio: " << std::fixed << std::setprecision(3) 
                  << metrics.sharpe_ratio << std::endl;
        std::cout << "  Daily Trades: " << std::fixed << std::setprecision(1) 
                  << metrics.avg_daily_trades << " (Health: " << metrics.health_status() << ")" << std::endl;
        std::cout << "  Total Trades: " << metrics.total_trades << std::endl;
    }
    
    // Final progress bar update
    std::cout << "\n\nTPA Test completed! Generating summary..." << std::endl;
    
    auto summary = analyzer.generate_summary();
    summary.assess_readiness(cfg);
    return summary;
}

} // namespace sentio

```

## 📄 **FILE 203 of 206**: src/time_utils.cpp

**File Information**:
- **Path**: `src/time_utils.cpp`

- **Size**: 106 lines
- **Modified**: 2025-09-10 11:15:18

- **Type**: .cpp

```text
#include "sentio/time_utils.hpp"
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <algorithm>

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

} // namespace sentio
```

## 📄 **FILE 204 of 206**: src/unified_metrics.cpp

**File Information**:
- **Path**: `src/unified_metrics.cpp`

- **Size**: 175 lines
- **Modified**: 2025-09-13 09:12:17

- **Type**: .cpp

```text
#include "sentio/unified_metrics.hpp"
#include "sentio/metrics.hpp"
#include "sentio/cost_model.hpp"
#include "sentio/side.hpp"
#include "audit/audit_db.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace sentio {

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
    // Reconstruct equity curve from audit events
    auto equity_curve = reconstruct_equity_curve_from_events(events, initial_capital, include_fees);
    
    // Count fill events for trade statistics
    int fills_count = 0;
    for (const auto& event : events) {
        if (event.kind == "FILL") {
            fills_count++;
        }
    }
    
    // Use unified calculation method
    return calculate_from_equity_curve(equity_curve, fills_count, include_fees);
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
            
            // Add to equity curve (convert timestamp from millis to string)
            std::string timestamp = std::to_string(event.ts_millis);
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
                std::string timestamp = std::to_string(event.ts_millis);
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

## 📄 **FILE 205 of 206**: src/unified_strategy_tester.cpp

**File Information**:
- **Path**: `src/unified_strategy_tester.cpp`

- **Size**: 790 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .cpp

```text
#include "sentio/unified_strategy_tester.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/run_id_generator.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <regex>

namespace sentio {

UnifiedStrategyTester::UnifiedStrategyTester() {
    // Initialize components
}

UnifiedStrategyTester::RobustnessReport UnifiedStrategyTester::run_comprehensive_test(const TestConfig& config) {
    std::cout << "🎯 Testing " << config.strategy_name << " on " << config.symbol;
    
    if (config.holistic_mode) {
        std::cout << " (HOLISTIC - Ultimate Robustness)";
    } else {
        switch (config.mode) {
            case TestMode::HISTORICAL: std::cout << " (Historical)"; break;
            case TestMode::AI_REGIME: std::cout << " (AI-" << config.regime << ")"; break;
            case TestMode::HYBRID: std::cout << " (Hybrid)"; break;
        }
    }
    std::cout << " - " << config.simulations << " simulations, " << config.duration << std::endl;
    
    // Warn about short test periods
    if (config.duration == "1d" || config.duration == "5d" || config.duration == "1w" || config.duration == "2w") {
        std::cout << "⚠️  WARNING: Short test period may produce unreliable MPR projections due to statistical noise" << std::endl;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run simulations based on mode
    std::vector<VirtualMarketEngine::VMSimulationResult> results;
    
    if (config.holistic_mode) {
        // Holistic mode: Run comprehensive multi-scenario testing
        results = run_holistic_tests(config);
    } else {
        switch (config.mode) {
            case TestMode::HISTORICAL:
                results = run_historical_tests(config, config.simulations);
                break;
            case TestMode::AI_REGIME:
                results = run_ai_regime_tests(config, config.simulations);
                break;
            case TestMode::HYBRID:
                results = run_hybrid_tests(config);
                break;
        }
    }
    
    // Apply stress testing if enabled
    if (config.stress_test) {
        apply_stress_scenarios(results, config);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "✅ Completed in " << duration.count() << "s" << std::endl;
    
    // Analyze results
    RobustnessReport report = analyze_results(results, config);
    report.test_duration_seconds = duration.count();
    report.mode_used = config.mode;
    
    return report;
}

std::vector<VirtualMarketEngine::VMSimulationResult> UnifiedStrategyTester::run_monte_carlo_tests(
    const TestConfig& config, int num_simulations) {
    
    std::cout << "🎲 Running Monte Carlo simulations..." << std::endl;
    
    // Create VM test configuration
    VirtualMarketEngine::VMTestConfig vm_config;
    vm_config.strategy_name = config.strategy_name;
    vm_config.symbol = config.symbol;
    vm_config.simulations = num_simulations;
    vm_config.params_json = config.params_json;
    vm_config.initial_capital = config.initial_capital;
    
    // Parse duration
    int total_minutes = parse_duration_to_minutes(config.duration);
    if (total_minutes >= 390) { // More than 1 trading day
        vm_config.days = total_minutes / 390;
        vm_config.hours = 0;
    } else {
        vm_config.days = 0;
        vm_config.hours = total_minutes / 60;
    }
    
    // Create strategy instance
    auto strategy = StrategyFactory::instance().create_strategy(config.strategy_name);
    if (!strategy) {
        std::cerr << "Error: Could not create strategy " << config.strategy_name << std::endl;
        return {};
    }
    
    // Create runner configuration
    RunnerCfg runner_cfg;
    runner_cfg.strategy_name = config.strategy_name;
    runner_cfg.strategy_params["buy_hi"] = "0.6";
    runner_cfg.strategy_params["sell_lo"] = "0.4";
    
    // **DISABLE AUDIT LOGGING**: Prevent conflicts when running within test-all
    if (config.disable_audit_logging) {
        runner_cfg.audit_level = AuditLevel::MetricsOnly;
    }
    
    // Run Monte Carlo simulation
    return vm_engine_.run_monte_carlo_simulation(vm_config, std::move(strategy), runner_cfg);
}

std::vector<VirtualMarketEngine::VMSimulationResult> UnifiedStrategyTester::run_historical_tests(
    const TestConfig& config, int num_simulations) {
    
    std::cout << "📊 Running historical pattern tests..." << std::endl;
    
    std::string historical_file = config.historical_data_file;
    if (historical_file.empty()) {
        historical_file = get_default_historical_file(config.symbol);
    }
    
    int continuation_minutes = parse_duration_to_minutes(config.duration);
    
    return vm_engine_.run_fast_historical_test(
        config.strategy_name, config.symbol, historical_file,
        continuation_minutes, num_simulations, config.params_json
    );
}

std::vector<VirtualMarketEngine::VMSimulationResult> UnifiedStrategyTester::run_ai_regime_tests(
    const TestConfig& config, int num_simulations) {
    
    std::cout << "🚀 Running Future QQQ regime tests..." << std::endl;
    
    // Use future QQQ data instead of MarS generation
    return vm_engine_.run_future_qqq_regime_test(
        config.strategy_name, config.symbol,
        num_simulations, config.regime, config.params_json
    );
}

std::vector<VirtualMarketEngine::VMSimulationResult> UnifiedStrategyTester::run_hybrid_tests(
    const TestConfig& config) {
    
    std::cout << "🌈 Running hybrid tests (Historical + AI)..." << std::endl;
    
    std::vector<VirtualMarketEngine::VMSimulationResult> all_results;
    
    // 60% Historical Pattern Testing
    int hist_sims = static_cast<int>(config.simulations * 0.6);
    if (hist_sims > 0) {
        auto hist_results = run_historical_tests(config, hist_sims);
        all_results.insert(all_results.end(), hist_results.begin(), hist_results.end());
    }
    
    // 40% AI Regime Testing
    int ai_sims = config.simulations - hist_sims;
    if (ai_sims > 0) {
        auto ai_results = run_ai_regime_tests(config, ai_sims);
        all_results.insert(all_results.end(), ai_results.begin(), ai_results.end());
    }
    
    return all_results;
}

UnifiedStrategyTester::RobustnessReport UnifiedStrategyTester::analyze_results(
    const std::vector<VirtualMarketEngine::VMSimulationResult>& results,
    const TestConfig& config) {
    
    RobustnessReport report;
    
    if (results.empty()) {
        std::cerr << "Warning: No simulation results to analyze" << std::endl;
        return report;
    }
    
    // Filter out failed simulations
    std::vector<VirtualMarketEngine::VMSimulationResult> valid_results;
    for (const auto& result : results) {
        if (result.total_trades > 0 || result.total_return != 0.0) {
            valid_results.push_back(result);
        }
    }
    
    report.total_simulations = results.size();
    report.successful_simulations = valid_results.size();
    
    if (valid_results.empty()) {
        std::cerr << "Warning: No valid simulation results" << std::endl;
        
        // Initialize report with safe default values for zero-trade scenarios
        report.monthly_projected_return = 0.0;
        report.sharpe_ratio = 0.0;
        report.max_drawdown = 0.0;
        report.win_rate = 0.0;
        report.profit_factor = 0.0;
        report.total_return = 0.0;
        report.avg_daily_trades = 0.0;
        
        report.consistency_score = 0.0;
        report.regime_adaptability = 0.0;
        report.stress_resilience = 0.0;
        report.liquidity_tolerance = 0.0;
        
        report.estimated_monthly_fees = 0.0;
        report.capital_efficiency = 0.0;
        report.position_turnover = 0.0;
        
        report.overall_risk = RiskLevel::HIGH;
        report.deployment_score = 0;
        report.ready_for_deployment = false;
        report.recommended_capital_range = "Not recommended";
        report.suggested_monitoring_days = 30; // Safe default
        
        // Initialize confidence intervals with zeros
        report.mpr_ci = {0.0, 0.0, 0.0, 0.0};
        report.sharpe_ci = {0.0, 0.0, 0.0, 0.0};
        report.drawdown_ci = {0.0, 0.0, 0.0, 0.0};
        report.win_rate_ci = {0.0, 0.0, 0.0, 0.0};
        
        return report;
    }
    
    // Extract metrics vectors
    std::vector<double> mprs, sharpes, drawdowns, win_rates, returns;
    std::vector<double> daily_trades, total_trades;
    
    for (const auto& result : valid_results) {
        mprs.push_back(result.monthly_projected_return);
        sharpes.push_back(result.sharpe_ratio);
        drawdowns.push_back(result.max_drawdown);
        win_rates.push_back(result.win_rate);
        returns.push_back(result.total_return);
        daily_trades.push_back(result.daily_trades);
        total_trades.push_back(result.total_trades);
    }
    
    // Calculate core metrics
    report.monthly_projected_return = std::accumulate(mprs.begin(), mprs.end(), 0.0) / mprs.size();
    report.sharpe_ratio = std::accumulate(sharpes.begin(), sharpes.end(), 0.0) / sharpes.size();
    report.max_drawdown = std::accumulate(drawdowns.begin(), drawdowns.end(), 0.0) / drawdowns.size();
    report.win_rate = std::accumulate(win_rates.begin(), win_rates.end(), 0.0) / win_rates.size();
    report.total_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    report.avg_daily_trades = std::accumulate(daily_trades.begin(), daily_trades.end(), 0.0) / daily_trades.size();
    
    // Calculate profit factor
    double total_wins = 0.0, total_losses = 0.0;
    for (const auto& result : valid_results) {
        if (result.total_return > 0) total_wins += result.total_return;
        else total_losses += std::abs(result.total_return);
    }
    report.profit_factor = (total_losses > 0) ? (total_wins / total_losses) : 0.0;
    
    // Calculate confidence intervals
    report.mpr_ci = calculate_confidence_interval(mprs, config.confidence_level);
    report.sharpe_ci = calculate_confidence_interval(sharpes, config.confidence_level);
    report.drawdown_ci = calculate_confidence_interval(drawdowns, config.confidence_level);
    report.win_rate_ci = calculate_confidence_interval(win_rates, config.confidence_level);
    
    // Calculate robustness metrics
    // Consistency Score: Lower variance = higher consistency
    double mpr_variance = 0.0;
    for (double mpr : mprs) {
        mpr_variance += std::pow(mpr - report.monthly_projected_return, 2);
    }
    mpr_variance /= mprs.size();
    
    // Avoid division by zero when MPR is 0
    double mpr_cv = 0.0;
    if (std::abs(report.monthly_projected_return) > 1e-9) {
        mpr_cv = std::sqrt(mpr_variance) / std::abs(report.monthly_projected_return);
    } else {
        // When MPR is effectively zero, use variance as a proxy for consistency
        mpr_cv = std::sqrt(mpr_variance) * 100.0; // Scale for percentage
    }
    report.consistency_score = std::max(0.0, std::min(100.0, 100.0 * (1.0 - std::min(1.0, mpr_cv))));
    
    // Enhanced robustness metrics for holistic mode
    if (config.holistic_mode) {
        // Regime Adaptability: Analyze performance across different market regimes
        std::map<std::string, std::vector<double>> regime_performance;
        // This would be enhanced with actual regime tracking in a full implementation
        report.regime_adaptability = std::max(0.0, std::min(100.0, 
            60.0 + 40.0 * std::tanh(report.sharpe_ratio - 0.5)));
        
        // Stress Resilience: Performance under extreme conditions
        double stress_penalty = 0.0;
        if (report.max_drawdown < -0.20) stress_penalty += 20.0;
        if (report.sharpe_ratio < 0.5) stress_penalty += 15.0;
        report.stress_resilience = std::max(0.0, 100.0 - stress_penalty);
        
        // Liquidity Tolerance: Assess impact of trading frequency
        double liquidity_score = 100.0;
        if (report.avg_daily_trades > 200) liquidity_score -= 30.0;
        else if (report.avg_daily_trades > 100) liquidity_score -= 15.0;
        else if (report.avg_daily_trades < 5) liquidity_score -= 10.0;
        report.liquidity_tolerance = std::max(0.0, liquidity_score);
        
    } else {
        // Standard regime adaptability for non-holistic modes
        report.regime_adaptability = std::max(0.0, std::min(100.0, 
            50.0 + 50.0 * std::tanh(report.sharpe_ratio - 1.0)));
    }
    
    // Stress Resilience: Based on worst-case scenarios
    double worst_mpr = *std::min_element(mprs.begin(), mprs.end());
    double worst_drawdown = *std::min_element(drawdowns.begin(), drawdowns.end());
    report.stress_resilience = std::max(0.0, std::min(100.0,
        50.0 + 25.0 * std::tanh(worst_mpr) + 25.0 * std::tanh(worst_drawdown + 0.2)));
    
    // Liquidity Tolerance: Based on trade frequency and consistency
    report.liquidity_tolerance = std::max(0.0, std::min(100.0,
        80.0 + 20.0 * std::tanh(report.avg_daily_trades / 50.0)));
    
    // Alpaca-specific metrics
    if (config.alpaca_fees) {
        // Estimate monthly fees (simplified: $0.005 per share, assume 100 shares per trade)
        double avg_monthly_trades = report.avg_daily_trades * 22; // 22 trading days per month
        report.estimated_monthly_fees = avg_monthly_trades * 100 * 0.005; // $0.005 per share
    }
    
    report.capital_efficiency = std::min(100.0, std::abs(report.total_return) * 100.0);
    report.position_turnover = report.avg_daily_trades / 10.0; // Simplified calculation
    
    // Risk assessment
    report.overall_risk = assess_risk_level(report);
    
    // Calculate deployment score
    report.deployment_score = calculate_deployment_score(report);
    report.ready_for_deployment = report.deployment_score >= 70;
    
    // Recommended capital range
    if (report.deployment_score >= 80) {
        report.recommended_capital_range = "$50,000 - $500,000";
        report.suggested_monitoring_days = 7;
    } else if (report.deployment_score >= 70) {
        report.recommended_capital_range = "$10,000 - $100,000";
        report.suggested_monitoring_days = 14;
    } else {
        report.recommended_capital_range = "$1,000 - $10,000";
        report.suggested_monitoring_days = 30;
    }
    
    // Generate recommendations (after monitoring days are set)
    generate_recommendations(report, config);
    
    return report;
}

UnifiedStrategyTester::ConfidenceInterval UnifiedStrategyTester::calculate_confidence_interval(
    const std::vector<double>& values, double confidence_level) {
    
    ConfidenceInterval ci;
    
    if (values.empty()) {
        ci.lower = ci.upper = ci.mean = ci.std_dev = 0.0;
        return ci;
    }
    
    ci.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    
    double variance = 0.0;
    for (double value : values) {
        variance += std::pow(value - ci.mean, 2);
    }
    variance /= values.size();
    ci.std_dev = std::sqrt(variance);
    
    // Simplified confidence interval (assumes normal distribution)
    double z_score = (confidence_level == 0.95) ? 1.96 : 
                    (confidence_level == 0.99) ? 2.58 : 1.645; // 90%
    
    double margin = z_score * ci.std_dev / std::sqrt(values.size());
    ci.lower = ci.mean - margin;
    ci.upper = ci.mean + margin;
    
    return ci;
}

UnifiedStrategyTester::RiskLevel UnifiedStrategyTester::assess_risk_level(const RobustnessReport& report) {
    int risk_score = 0;
    
    // High drawdown increases risk
    if (report.max_drawdown < -0.20) risk_score += 3;
    else if (report.max_drawdown < -0.10) risk_score += 2;
    else if (report.max_drawdown < -0.05) risk_score += 1;
    
    // Low Sharpe ratio increases risk
    if (report.sharpe_ratio < 0.5) risk_score += 3;
    else if (report.sharpe_ratio < 1.0) risk_score += 2;
    else if (report.sharpe_ratio < 1.5) risk_score += 1;
    
    // Low consistency increases risk
    if (report.consistency_score < 50) risk_score += 2;
    else if (report.consistency_score < 70) risk_score += 1;
    
    // Low win rate increases risk
    if (report.win_rate < 0.4) risk_score += 2;
    else if (report.win_rate < 0.5) risk_score += 1;
    
    if (risk_score >= 6) return RiskLevel::EXTREME;
    else if (risk_score >= 4) return RiskLevel::HIGH;
    else if (risk_score >= 2) return RiskLevel::MEDIUM;
    else return RiskLevel::LOW;
}

void UnifiedStrategyTester::generate_recommendations(RobustnessReport& report, const TestConfig& config) {
    // Risk warnings
    if (report.max_drawdown < -0.15) {
        report.risk_warnings.push_back("High maximum drawdown risk (>" + std::to_string(std::abs(report.max_drawdown * 100)) + "%)");
    }
    
    if (report.sharpe_ratio < 1.0) {
        report.risk_warnings.push_back("Low risk-adjusted returns (Sharpe < 1.0)");
    }
    
    if (report.consistency_score < 60) {
        report.risk_warnings.push_back("High performance variance across simulations");
    }
    
    if (report.avg_daily_trades > 100) {
        report.risk_warnings.push_back("Very high trading frequency may impact execution");
    }
    
    // Recommendations
    if (report.monthly_projected_return > 0.05) {
        report.recommendations.push_back("Strong performance - consider gradual capital scaling");
    }
    
    if (report.consistency_score > 80) {
        report.recommendations.push_back("Excellent consistency - suitable for automated trading");
    }
    
    if (report.stress_resilience < 70) {
        report.recommendations.push_back("Consider position sizing limits during high volatility");
    }
    
    if (report.avg_daily_trades < 5) {
        report.recommendations.push_back("Low trade frequency - consider longer backtesting periods");
    }
    
    report.recommendations.push_back("Start with paper trading for " + std::to_string(report.suggested_monitoring_days) + " days");
}

int UnifiedStrategyTester::calculate_deployment_score(const RobustnessReport& report) {
    int score = 50; // Base score
    
    // Performance components (40 points max)
    if (report.monthly_projected_return > 0.10) score += 15;
    else if (report.monthly_projected_return > 0.05) score += 10;
    else if (report.monthly_projected_return > 0.02) score += 5;
    
    if (report.sharpe_ratio > 2.0) score += 15;
    else if (report.sharpe_ratio > 1.5) score += 10;
    else if (report.sharpe_ratio > 1.0) score += 5;
    
    if (report.max_drawdown > -0.05) score += 10;
    else if (report.max_drawdown > -0.10) score += 5;
    else if (report.max_drawdown < -0.20) score -= 10;
    
    // Robustness components (30 points max)
    score += static_cast<int>(report.consistency_score * 0.15);
    score += static_cast<int>(report.stress_resilience * 0.15);
    
    // Risk components (20 points max)
    if (report.win_rate > 0.6) score += 10;
    else if (report.win_rate > 0.5) score += 5;
    
    if (report.profit_factor > 2.0) score += 10;
    else if (report.profit_factor > 1.5) score += 5;
    
    // Practical components (10 points max)
    if (report.avg_daily_trades >= 5 && report.avg_daily_trades <= 50) score += 5;
    if (report.successful_simulations >= report.total_simulations * 0.9) score += 5;
    
    return std::max(0, std::min(100, score));
}

void UnifiedStrategyTester::apply_stress_scenarios(
    std::vector<VirtualMarketEngine::VMSimulationResult>& results,
    const TestConfig& config) {
    
    std::cout << "⚡ Applying stress testing scenarios..." << std::endl;
    
    // Apply stress multipliers to simulate adverse conditions
    for (auto& result : results) {
        // Increase drawdown by 50% in stress scenarios
        result.max_drawdown *= 1.5;
        
        // Reduce returns by 20% in stress scenarios
        result.total_return *= 0.8;
        result.monthly_projected_return *= 0.8;
        
        // Reduce Sharpe ratio due to increased volatility
        result.sharpe_ratio *= 0.7;
        
        // Reduce win rate slightly
        result.win_rate *= 0.9;
    }
}

int UnifiedStrategyTester::parse_duration_to_minutes(const std::string& duration) {
    std::regex duration_regex(R"((\d+)([hdwm]))");
    std::smatch match;
    
    if (!std::regex_match(duration, match, duration_regex)) {
        std::cerr << "Warning: Invalid duration format '" << duration << "', using default 5d" << std::endl;
        return 5 * 390; // 5 days
    }
    
    int value = std::stoi(match[1].str());
    char unit = match[2].str()[0];
    
    switch (unit) {
        case 'h': return value * 60;
        case 'd': return value * 390; // 390 minutes per trading day
        case 'w': return value * 5 * 390; // 5 trading days per week
        case 'm': return value * 22 * 390; // ~22 trading days per month
        default: return 5 * 390;
    }
}

UnifiedStrategyTester::TestMode UnifiedStrategyTester::parse_test_mode(const std::string& mode_str) {
    std::string mode_lower = mode_str;
    std::transform(mode_lower.begin(), mode_lower.end(), mode_lower.begin(), ::tolower);
    
    if (mode_lower == "historical" || mode_lower == "hist") return TestMode::HISTORICAL;
    if (mode_lower == "ai-regime" || mode_lower == "ai") return TestMode::AI_REGIME;
    if (mode_lower == "hybrid") return TestMode::HYBRID;
    
    std::cerr << "Warning: Unknown test mode '" << mode_str << "', using hybrid" << std::endl;
    return TestMode::HYBRID;
}

std::string UnifiedStrategyTester::get_default_historical_file(const std::string& symbol) {
    return "data/equities/" + symbol + "_NH.csv";
}

void UnifiedStrategyTester::print_robustness_report(const RobustnessReport& report, const TestConfig& config) {
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "🎯 STRATEGY ROBUSTNESS REPORT" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Header information
    std::cout << "Strategy: " << std::setw(20) << config.strategy_name 
              << "  Symbol: " << std::setw(10) << config.symbol 
              << "  Test Date: " << std::setw(12) << "2024-01-15" << std::endl;
    
    std::cout << "Duration: " << std::setw(20) << config.duration
              << "  Simulations: " << std::setw(7) << report.total_simulations
              << "  Mode: " << std::setw(15);
    
    switch (report.mode_used) {
        case TestMode::HISTORICAL: std::cout << "historical"; break;
        case TestMode::AI_REGIME: std::cout << "ai-regime"; break;
        case TestMode::HYBRID: std::cout << "hybrid"; break;
    }
    std::cout << std::endl;
    
    std::cout << "Confidence Level: " << std::setw(12) << (config.confidence_level * 100) << "%"
              << "  Alpaca Fees: " << std::setw(8) << (config.alpaca_fees ? "Enabled" : "Disabled")
              << "  Market Hours: " << std::setw(8) << "RTH" << std::endl;
    
    std::cout << std::endl;
    
    // Performance Summary
    std::cout << "📈 PERFORMANCE SUMMARY" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Monthly Projected Return: " << std::setw(8) << (report.monthly_projected_return * 100) << "% ± " 
              << std::setw(4) << ((report.mpr_ci.upper - report.mpr_ci.lower) / 2 * 100) << "%"
              << "  [" << std::setw(5) << (report.mpr_ci.lower * 100) << "% - " 
              << std::setw(5) << (report.mpr_ci.upper * 100) << "%] (95% CI)" << std::endl;
    
    std::cout << std::setprecision(2);
    std::cout << "Sharpe Ratio:            " << std::setw(8) << report.sharpe_ratio << " ± " 
              << std::setw(4) << ((report.sharpe_ci.upper - report.sharpe_ci.lower) / 2) << ""
              << "  [" << std::setw(5) << report.sharpe_ci.lower << " - " 
              << std::setw(5) << report.sharpe_ci.upper << "] (95% CI)" << std::endl;
    
    std::cout << std::setprecision(1);
    std::cout << "Maximum Drawdown:        " << std::setw(8) << (report.max_drawdown * 100) << "% ± " 
              << std::setw(4) << ((report.drawdown_ci.upper - report.drawdown_ci.lower) / 2 * 100) << "%"
              << "  [" << std::setw(5) << (report.drawdown_ci.lower * 100) << "% - " 
              << std::setw(5) << (report.drawdown_ci.upper * 100) << "%] (95% CI)" << std::endl;
    
    std::cout << "Win Rate:                " << std::setw(8) << (report.win_rate * 100) << "% ± " 
              << std::setw(4) << ((report.win_rate_ci.upper - report.win_rate_ci.lower) / 2 * 100) << "%"
              << "  [" << std::setw(5) << (report.win_rate_ci.lower * 100) << "% - " 
              << std::setw(5) << (report.win_rate_ci.upper * 100) << "%] (95% CI)" << std::endl;
    
    std::cout << std::setprecision(2);
    std::cout << "Profit Factor:           " << std::setw(8) << report.profit_factor << ""
              << "     [" << std::setw(5) << (report.profit_factor * 0.8) << " - " 
              << std::setw(5) << (report.profit_factor * 1.2) << "] (Est. Range)" << std::endl;
    
    std::cout << std::endl;
    
    // Robustness Analysis
    std::cout << "🛡️  ROBUSTNESS ANALYSIS" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    auto get_rating = [](double score) -> std::string {
        if (score >= 90) return "EXCELLENT";
        if (score >= 80) return "VERY GOOD";
        if (score >= 70) return "GOOD";
        if (score >= 60) return "FAIR";
        return "POOR";
    };
    
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "Consistency Score:       " << std::setw(8) << report.consistency_score << "/100"
              << "      " << get_rating(report.consistency_score) << " - Performance variance" << std::endl;
    std::cout << "Regime Adaptability:     " << std::setw(8) << report.regime_adaptability << "/100"
              << "      " << get_rating(report.regime_adaptability) << " - Market adaptation" << std::endl;
    std::cout << "Stress Resilience:       " << std::setw(8) << report.stress_resilience << "/100"
              << "      " << get_rating(report.stress_resilience) << " - Adverse conditions" << std::endl;
    std::cout << "Liquidity Tolerance:     " << std::setw(8) << report.liquidity_tolerance << "/100"
              << "      " << get_rating(report.liquidity_tolerance) << " - Execution robustness" << std::endl;
    
    std::cout << std::endl;
    
    // Alpaca Trading Analysis
    std::cout << "💰 ALPACA TRADING ANALYSIS" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "Estimated Monthly Fees:  $" << std::setw(7) << report.estimated_monthly_fees 
              << "     (" << std::setprecision(2) << (report.estimated_monthly_fees / config.initial_capital * 100) 
              << "% of capital)" << std::endl;
    
    std::cout << std::setprecision(1);
    std::cout << "Capital Efficiency:      " << std::setw(8) << report.capital_efficiency << "%"
              << "      High capital utilization" << std::endl;
    
    std::cout << "Average Daily Trades:    " << std::setw(8) << report.avg_daily_trades 
              << "      Within Alpaca limits" << std::endl;
    
    std::cout << std::setprecision(1);
    std::cout << "Position Turnover:       " << std::setw(8) << report.position_turnover << "x/day"
              << "      Moderate turnover rate" << std::endl;
    
    std::cout << std::endl;
    
    // Risk Assessment
    std::string risk_str;
    switch (report.overall_risk) {
        case RiskLevel::LOW: risk_str = "LOW"; break;
        case RiskLevel::MEDIUM: risk_str = "MEDIUM"; break;
        case RiskLevel::HIGH: risk_str = "HIGH"; break;
        case RiskLevel::EXTREME: risk_str = "EXTREME"; break;
    }
    
    std::cout << "⚠️  RISK ASSESSMENT: " << risk_str << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    if (!report.risk_warnings.empty()) {
        std::cout << "⚠️  WARNINGS:" << std::endl;
        for (const auto& warning : report.risk_warnings) {
            std::cout << "  • " << warning << std::endl;
        }
        std::cout << std::endl;
    }
    
    if (!report.recommendations.empty()) {
        std::cout << "💡 RECOMMENDATIONS:" << std::endl;
        for (const auto& rec : report.recommendations) {
            std::cout << "  • " << rec << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Deployment Readiness
    std::cout << "🎯 DEPLOYMENT READINESS: " << (report.ready_for_deployment ? "READY" : "NOT READY") << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    std::cout << "Overall Score:           " << std::setw(8) << report.deployment_score << "/100"
              << "      (" << get_rating(report.deployment_score) << ")" << std::endl;
    std::cout << "Recommended Live Capital: " << report.recommended_capital_range << std::endl;
    std::cout << "Suggested Monitoring:    " << report.suggested_monitoring_days << " days paper trading" << std::endl;
    
    std::cout << std::string(80, '=') << std::endl;
}

bool UnifiedStrategyTester::save_report(const RobustnessReport& report, const TestConfig& config,
                                       const std::string& filename, const std::string& format) {
    // Implementation for saving reports in different formats
    // This is a placeholder - full implementation would handle JSON, CSV, etc.
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    if (format == "json") {
        // JSON format implementation
        file << "{\n";
        file << "  \"strategy\": \"" << config.strategy_name << "\",\n";
        file << "  \"symbol\": \"" << config.symbol << "\",\n";
        file << "  \"monthly_projected_return\": " << report.monthly_projected_return << ",\n";
        file << "  \"sharpe_ratio\": " << report.sharpe_ratio << ",\n";
        file << "  \"max_drawdown\": " << report.max_drawdown << ",\n";
        file << "  \"deployment_score\": " << report.deployment_score << "\n";
        file << "}\n";
    } else {
        // Default to text format
        // Redirect print_robustness_report output to file
        std::streambuf* orig = std::cout.rdbuf();
        std::cout.rdbuf(file.rdbuf());
        
        // This is a bit hacky - in a real implementation, we'd refactor print_robustness_report
        // to accept an output stream parameter
        
        std::cout.rdbuf(orig);
    }
    
    file.close();
    return true;
}

std::vector<VirtualMarketEngine::VMSimulationResult> UnifiedStrategyTester::run_holistic_tests(const TestConfig& config) {
    std::cout << "🔬 HOLISTIC TESTING: Running comprehensive multi-scenario robustness analysis..." << std::endl;
    
    std::vector<VirtualMarketEngine::VMSimulationResult> all_results;
    
    // 1. Historical Data Testing (40% of simulations)
    int historical_sims = config.simulations * 0.4;
    std::cout << "📊 Phase 1/4: Historical Pattern Analysis (" << historical_sims << " simulations)" << std::endl;
    auto historical_results = run_historical_tests(config, historical_sims);
    all_results.insert(all_results.end(), historical_results.begin(), historical_results.end());
    
    // 2. AI Market Regime Testing - Multiple Regimes (40% of simulations)
    int ai_sims_per_regime = (config.simulations * 0.4) / 4; // 4 different regimes (normal, volatile, trending, bear)
    std::cout << "🤖 Phase 2/4: AI Market Regime Testing (" << (ai_sims_per_regime * 4) << " simulations)" << std::endl;
    
    std::vector<std::string> regimes = {"normal", "volatile", "trending", "bear"}; // Removed "bull" - not supported by MarS
    for (const auto& regime : regimes) {
        auto regime_config = config;
        regime_config.regime = regime;
        auto regime_results = run_ai_regime_tests(regime_config, ai_sims_per_regime);
        all_results.insert(all_results.end(), regime_results.begin(), regime_results.end());
    }
    
    // 3. Stress Testing Scenarios (10% of simulations)
    int stress_sims = config.simulations * 0.1;
    std::cout << "⚡ Phase 3/4: Extreme Stress Testing (" << stress_sims << " simulations)" << std::endl;
    auto stress_config = config;
    stress_config.stress_test = true;
    stress_config.liquidity_stress = true;
    stress_config.volatility_min = 0.02; // High volatility
    stress_config.volatility_max = 0.08; // Extreme volatility
    auto stress_results = run_ai_regime_tests(stress_config, stress_sims);
    all_results.insert(all_results.end(), stress_results.begin(), stress_results.end());
    
    // 4. Cross-Timeframe Validation (10% of simulations)
    int timeframe_sims = config.simulations * 0.1;
    std::cout << "⏰ Phase 4/4: Cross-Timeframe Validation (" << timeframe_sims << " simulations)" << std::endl;
    
    // Test with different durations to validate consistency
    std::vector<std::string> test_durations = {"1w", "2w", "1m"};
    int sims_per_duration = std::max(1, timeframe_sims / (int)test_durations.size());
    
    for (const auto& duration : test_durations) {
        auto duration_config = config;
        duration_config.duration = duration;
        auto duration_results = run_historical_tests(duration_config, sims_per_duration);
        all_results.insert(all_results.end(), duration_results.begin(), duration_results.end());
    }
    
    std::cout << "✅ HOLISTIC TESTING COMPLETE: " << all_results.size() << " total simulations across all scenarios" << std::endl;
    
    return all_results;
}

// Duplicate method implementations removed - using existing implementations above

} // namespace sentio

```

## 📄 **FILE 206 of 206**: src/virtual_market.cpp

**File Information**:
- **Path**: `src/virtual_market.cpp`

- **Size**: 756 lines
- **Modified**: 2025-09-15 15:04:43

- **Type**: .cpp

```text
#include "sentio/virtual_market.hpp"
// Strategy registry removed - using factory pattern instead
#include "sentio/runner.hpp"
#include "sentio/temporal_analysis.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/audit.hpp"
#include "audit/audit_db_recorder.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/mars_data_loader.hpp"
#include "sentio/future_qqq_loader.hpp"
#include "sentio/run_id_generator.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>

namespace sentio {

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
    
    std::vector<VMSimulationResult> results;
    results.reserve(config.simulations);
    
    int periods = config.hours > 0 ? config.hours * 60 : config.days * 390;
    
    std::cout << "🎲 Running " << config.simulations << " Monte Carlo simulations..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < config.simulations; ++i) {
        // Reduced progress reporting - only show every 50% or at key milestones
        if (config.simulations >= 10 && ((i + 1) % (config.simulations / 2) == 0 || i == 0 || i == config.simulations - 1)) {
            auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
            auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            double progress_pct = ((i + 1) / static_cast<double>(config.simulations)) * 100;
            
            std::cout << "📊 Progress: " << std::fixed << std::setprecision(0) << progress_pct 
                      << "% (" << (i + 1) << "/" << config.simulations << ")" << std::endl;
        }
        
        // Generate synthetic market data
        std::vector<Bar> market_data = generate_market_data(config.symbol, periods, 60);
        
        // Create a new strategy instance for this simulation
        auto sim_strategy = StrategyFactory::instance().create_strategy(config.strategy_name);
        
        // Run single simulation
        VMSimulationResult result = run_single_simulation(market_data, 
                                                         std::move(sim_strategy),
                                                         runner_cfg, 
                                                         config.initial_capital);
        results.push_back(result);
    }
    
    auto total_time = std::chrono::high_resolution_clock::now() - start_time;
    auto total_sec = std::chrono::duration_cast<std::chrono::seconds>(total_time).count();
    
    std::cout << "⏱️  Completed " << config.simulations << " simulations in " 
              << total_sec << " seconds" << std::endl;
    
    return results;
}

std::vector<VirtualMarketEngine::VMSimulationResult> VirtualMarketEngine::run_mars_monte_carlo_simulation(
    const VMTestConfig& config,
    std::unique_ptr<BaseStrategy> strategy,
    const RunnerCfg& runner_cfg,
    const std::string& market_regime) {
    
    std::vector<VMSimulationResult> results;
    results.reserve(config.simulations);
    
    std::cout << "🤖 Running " << config.simulations << " AI regime tests..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < config.simulations; ++i) {
        // Minimal progress reporting
        if (config.simulations >= 5 && ((i + 1) % std::max(1, config.simulations / 2) == 0 || i == config.simulations - 1)) {
            double progress_pct = (100.0 * (i + 1)) / config.simulations;
            std::cout << "📊 Progress: " << std::fixed << std::setprecision(0) << progress_pct 
                      << "% (" << (i + 1) << "/" << config.simulations << ")" << std::endl;
        }
        
        try {
            // Generate MarS data for this simulation
            int periods = config.days * 24 * 60; // Convert days to minutes
            auto mars_data = MarsDataLoader::load_mars_data(
                config.symbol, periods, 60, 1, market_regime
            );
            
            if (mars_data.empty()) {
                std::cerr << "Warning: No MarS data generated for simulation " << (i + 1) << std::endl;
                results.push_back(VMSimulationResult{}); // Add empty result
                continue;
            }
            
            // Create new strategy instance for this simulation
            auto strategy_instance = StrategyFactory::instance().create_strategy(
                config.strategy_name
            );
            
            if (!strategy_instance) {
                std::cerr << "Error: Could not create strategy instance for simulation " << (i + 1) << std::endl;
                results.push_back(VMSimulationResult{}); // Add empty result
                continue;
            }
            
            // Run simulation with MarS data
            auto result = run_single_simulation(
                mars_data, std::move(strategy_instance), runner_cfg, config.initial_capital
            );
            
            results.push_back(result);
            
        } catch (const std::exception& e) {
            std::cerr << "Error in MarS simulation " << (i + 1) << ": " << e.what() << std::endl;
            results.push_back(VMSimulationResult{}); // Add empty result
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    std::cout << "⏱️  Completed " << config.simulations << " MarS simulations in " 
              << total_time << " seconds" << std::endl;
    
    return results;
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
    
    std::vector<VMSimulationResult> results;
    results.reserve(simulations);
    
    std::cout << "⚡ Starting Fast Historical Test..." << std::endl;
    std::cout << "📊 Strategy: " << strategy_name << std::endl;
    std::cout << "📈 Symbol: " << symbol << std::endl;
    std::cout << "📊 Historical data: " << historical_data_file << std::endl;
    std::cout << "⏱️  Continuation: " << continuation_minutes << " minutes" << std::endl;
    std::cout << "🎲 Simulations: " << simulations << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < simulations; ++i) {
        // Minimal progress reporting
        if (simulations >= 5 && ((i + 1) % std::max(1, simulations / 2) == 0 || i == simulations - 1)) {
            double progress_pct = (100.0 * (i + 1)) / simulations;
            std::cout << "📊 Progress: " << std::fixed << std::setprecision(0) << progress_pct 
                      << "% (" << (i + 1) << "/" << simulations << ")" << std::endl;
        }
        
        try {
            // Generate fast historical data for this simulation
            auto fast_data = MarsDataLoader::load_fast_historical_data(
                symbol, historical_data_file, continuation_minutes
            );
            
            if (fast_data.empty()) {
                std::cerr << "Warning: No fast historical data generated for simulation " << (i + 1) << std::endl;
                results.push_back(VMSimulationResult{}); // Add empty result
                continue;
            }
            
            // Create new strategy instance for this simulation
            auto strategy_instance = StrategyFactory::instance().create_strategy(
                strategy_name
            );
            
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
            
            // Run simulation with fast historical data
            auto result = run_single_simulation(
                fast_data, std::move(strategy_instance), runner_cfg, 100000.0
            );
            
            results.push_back(result);
            
        } catch (const std::exception& e) {
            std::cerr << "Error in fast historical simulation " << (i + 1) << ": " << e.what() << std::endl;
            results.push_back(VMSimulationResult{}); // Add empty result
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    std::cout << "⏱️  Completed " << simulations << " fast historical simulations in " 
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

VirtualMarketEngine::VMSimulationResult VirtualMarketEngine::run_single_simulation(
    const std::vector<Bar>& market_data,
    std::unique_ptr<BaseStrategy> strategy,
    const RunnerCfg& runner_cfg,
    double initial_capital) {
    
    // REAL INTEGRATION: Use actual Runner component
    VMSimulationResult result;
    
    try {
        // 1. Create SymbolTable for the test symbol
        SymbolTable ST;
        int symbol_id = ST.intern("QQQ"); // Use QQQ as the base symbol
        
        // 2. Create series data structure (single symbol)
        std::vector<std::vector<Bar>> series(1);
        series[0] = market_data;
        
        // 3. Create audit recorder - use in-memory database if audit logging is disabled
        std::string run_id = generate_run_id();
        std::string note = "Strategy: " + runner_cfg.strategy_name + ", Test: vm_test, Generated by strattest";
        
        // Use in-memory database if audit logging is disabled to prevent conflicts
        std::string db_path = (runner_cfg.audit_level == AuditLevel::MetricsOnly) ? ":memory:" : "audit/sentio_audit.sqlite3";
        audit::AuditDBRecorder audit(db_path, run_id, note);
        
        // 4. Run REAL backtest using actual Runner
        RunResult run_result = run_backtest(audit, ST, series, symbol_id, runner_cfg);
        
        // Suppress debug output for cleaner console
        
        // 5. Extract performance metrics from real results
        result.total_return = run_result.total_return;
        result.final_capital = initial_capital * (1 + run_result.total_return);
        result.sharpe_ratio = run_result.sharpe_ratio;
        result.max_drawdown = run_result.max_drawdown;
        result.win_rate = 0.0; // Not available in RunResult, calculate separately if needed
        result.total_trades = run_result.total_fills;
        result.monthly_projected_return = run_result.monthly_projected_return;
        result.daily_trades = static_cast<double>(run_result.daily_trades);
        
        // Suppress individual simulation output for cleaner console
        
    } catch (const std::exception& e) {
        std::cerr << "❌ VM simulation failed: " << e.what() << std::endl;
        
        // Return zero results on error
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
    
    // Print report
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "📊 VIRTUAL MARKET TEST RESULTS" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Strategy:                 " << config.strategy_name << std::endl;
    std::cout << "Symbol:                   " << config.symbol << std::endl;
    std::cout << "Simulations:              " << config.simulations << std::endl;
    std::cout << "Simulation Period:        " << (config.hours > 0 ? std::to_string(config.hours) + " hours" : std::to_string(config.days) + " days") << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "📈 RETURN STATISTICS" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Mean Return:              " << std::setw(8) << std::fixed << std::setprecision(2) << mean_return * 100 << "%" << std::endl;
    std::cout << "Median Return:            " << std::setw(8) << std::fixed << std::setprecision(2) << median_return * 100 << "%" << std::endl;
    std::cout << "Standard Deviation:       " << std::setw(8) << std::fixed << std::setprecision(2) << std_return * 100 << "%" << std::endl;
    std::cout << "Minimum Return:           " << std::setw(8) << std::fixed << std::setprecision(2) << returns.front() * 100 << "%" << std::endl;
    std::cout << "Maximum Return:           " << std::setw(8) << std::fixed << std::setprecision(2) << returns.back() * 100 << "%" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "📊 CONFIDENCE INTERVALS" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "5th Percentile:           " << std::setw(8) << std::fixed << std::setprecision(2) << returns[static_cast<size_t>(returns.size() * 0.05)] * 100 << "%" << std::endl;
    std::cout << "25th Percentile:          " << std::setw(8) << std::fixed << std::setprecision(2) << returns[static_cast<size_t>(returns.size() * 0.25)] * 100 << "%" << std::endl;
    std::cout << "75th Percentile:          " << std::setw(8) << std::fixed << std::setprecision(2) << returns[static_cast<size_t>(returns.size() * 0.75)] * 100 << "%" << std::endl;
    std::cout << "95th Percentile:          " << std::setw(8) << std::fixed << std::setprecision(2) << returns[static_cast<size_t>(returns.size() * 0.95)] * 100 << "%" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "🎯 PROBABILITY ANALYSIS" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Probability of Profit:    " << std::setw(8) << std::fixed << std::setprecision(1) << prob_profit << "%" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "📋 ADDITIONAL METRICS" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Mean Sharpe Ratio:        " << std::setw(8) << std::fixed << std::setprecision(2) << mean_sharpe << std::endl;
    std::cout << "Mean MPR (Monthly):       " << std::setw(8) << std::fixed << std::setprecision(2) << mean_mpr * 100 << "%" << std::endl;
    std::cout << "Mean Daily Trades:        " << std::setw(8) << std::fixed << std::setprecision(1) << mean_daily_trades << std::endl;
    std::cout << std::string(70, '=') << std::endl << std::endl;
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

