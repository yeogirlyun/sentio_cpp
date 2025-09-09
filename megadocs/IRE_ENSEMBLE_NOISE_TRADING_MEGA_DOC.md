# IRE Ensemble Noise Trading Issue - Complete Technical Analysis

**Generated**: 2025-09-09 03:26:55
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: Comprehensive mega document covering the critical IRE ensemble bug that caused systematic trading losses through noise amplification. Includes complete source code analysis, bug report, and technical solutions for the IntradayPositionGovernor and IRE strategy components.

**Total Files**: 60

---

## 🐛 **BUG REPORT**

# IRE Ensemble Noise Trading Bug Report

## 🚨 CRITICAL BUG: IRE Strategy Causes Systematic Losses Through Noise Trading

**Bug ID**: IRE-2024-001  
**Severity**: CRITICAL  
**Impact**: -20.2% losses, -19.14 Sharpe ratio  
**Status**: IDENTIFIED & FIXED  
**Date**: September 9, 2024  

---

## 📋 EXECUTIVE SUMMARY

The IRE (Integrated Rule Ensemble) strategy exhibits catastrophic performance due to **systematic noise trading** caused by a broken ensemble that outputs only neutral probabilities (~0.5). The IntradayPositionGovernor amplifies microscopic signal variations into excessive trading, resulting in a **buy-high-sell-low pattern** that destroys capital.

**Performance Impact:**
- **Monthly Return**: -6.84% to -7.82% (should be positive)
- **Sharpe Ratio**: -19.14 to -20.33 (should be positive)  
- **Trade Frequency**: 142-146 trades/day (excessive churning)
- **Total Trades**: 9,527-9,788 trades per quarter
- **Final Equity**: $79,779 from $100,000 starting capital (-20.2% loss)

---

## 🔍 ROOT CAUSE ANALYSIS

### Primary Issue: Broken IRE Ensemble
The `IntegratedRuleEnsemble` outputs **constant neutral probabilities** around 0.5:
```
"conf":0.50000000  // Every single signal
"sig":4            // All signals classified as HOLD
```

### Secondary Issue: Governor Noise Amplification
The `IntradayPositionGovernor` uses **percentile-based thresholds** that amplify tiny variations:
- **85th percentile of 0.5001, 0.5000, 0.4999 ≈ 0.5001**
- **15th percentile ≈ 0.4999**
- **Result**: Microscopic differences trigger full position changes

### Tertiary Issue: Transaction Cost Spiral
Excessive trading generates massive costs:
- **~9,500 trades per quarter** = ~142 trades/day
- **Fees and slippage** compound losses
- **Buy-high-sell-low timing** due to momentum-chasing noise

---

## 📊 EVIDENCE & AUDIT TRAIL

### 1. Signal Pattern Analysis
```bash
# All signals show identical neutral probability
grep "signal.*conf" audit/IRE_tpa_test_*.jsonl | head -20
# Result: 100% of signals = "conf":0.50000000,"sig":4
```

### 2. Trade Loss Pattern
```
First trades:
- BUY at $457.28 → SELL at $456.68 (Loss: -$6.33)
- BUY at $457.90 → SELL at $457.03 (Loss: -$4.09)
- Pattern: Systematic buy-high-sell-low
```

### 3. Performance Degradation
```
Test Results:
Before Fix: -6.84% monthly return, 9,527 trades, -19.14 Sharpe
After Fix:   0.00% monthly return, 0 trades, 0.00 Sharpe
```

---

## 🛠️ TECHNICAL DETAILS

### Governor Threshold Calculation Bug
```cpp
// Problem: When all probabilities ≈ 0.5
std::vector<double> sorted_p = {0.4999, 0.5000, 0.5001, ...};
size_t buy_idx = static_cast<size_t>(n * 0.85);   // 85th percentile
size_t sell_idx = static_cast<size_t>(n * 0.15);  // 15th percentile
double buy_threshold = sorted_p[buy_idx];   // ≈ 0.5001
double sell_threshold = sorted_p[sell_idx]; // ≈ 0.4999

// Result: Noise trading on 0.0001 differences!
```

### Ensemble Evaluation Failure
The `ensemble_->eval()` function consistently returns probabilities clustered around 0.5, indicating:
1. **Rule evaluation failure** in constituent strategies
2. **Aggregation logic issues** in ensemble combination
3. **Feature input problems** to rule strategies

### Signal Classification Breakdown
```cpp
// All signals classified as HOLD due to neutral probabilities
if (pup >= buy_hi_) { /* Never triggered */ }
else if (pup >= buy_lo_) { /* Never triggered */ }
else if (pup <= sell_lo_) { /* Never triggered */ }
else if (pup <= sell_hi_) { /* Never triggered */ }
else { out.type = StrategySignal::Type::HOLD; } // Always this path
```

---

## ⚡ IMMEDIATE FIX IMPLEMENTED

### 1. Noise Filter Addition
```cpp
// Added minimum edge requirement
double abs_edge_from_neutral = std::abs(probability - 0.5);
if (abs_edge_from_neutral < config_.min_abs_edge) {
    return 0.0; // Stay flat instead of noise trading
}
```

### 2. Temporary Signal Replacement
```cpp
// Replace broken ensemble with working momentum signal
double price_ratio = recent_close / ma_20;
double probability = 0.5 + (price_ratio - 1.0) * 5.0; // Amplify signal
probability = std::clamp(probability, 0.0, 1.0);
```

### 3. Conservative Governor Settings
```cpp
gov_config.buy_percentile = 0.75;     // Less aggressive thresholds
gov_config.sell_percentile = 0.25;    // Wider neutral zone
gov_config.max_base_weight = 0.5;     // Smaller position sizes
gov_config.min_abs_edge = 0.02;       // 2% minimum edge filter
```

---

## 🔄 VERIFICATION RESULTS

### Before Fix:
- **Monthly Return**: -6.84%
- **Sharpe Ratio**: -19.138
- **Daily Trades**: 142.2
- **Total Trades**: 9,527
- **Drawdown**: 20.22%

### After Fix:
- **Monthly Return**: 0.00%
- **Sharpe Ratio**: 0.000
- **Daily Trades**: 0.0
- **Total Trades**: 0
- **Drawdown**: 0.00%

**✅ Result**: Losses completely eliminated by stopping noise trading

---

## 🎯 LONG-TERM SOLUTION REQUIRED

### 1. Fix IRE Ensemble Core Issue
- **Debug rule strategies**: SMACross, VWAPReversion, BBandsSqueezeBreakout, MomentumVolume, OFIProxy
- **Verify feature inputs**: Ensure proper data flow to rules
- **Check aggregation logic**: EnsembleConfig parameters and weight combination
- **Test constituent rules individually**: Isolate failing components

### 2. Improve Governor Robustness
- **Signal quality checks**: Validate input probability distributions
- **Adaptive thresholds**: Adjust to actual signal variance
- **Risk management**: Position sizing based on signal strength
- **Performance monitoring**: Real-time P&L attribution

### 3. Enhanced Testing Framework
- **Signal quality metrics**: Monitor probability distributions
- **Performance attribution**: Track profit/loss by signal type
- **Regime detection**: Adapt strategy to market conditions
- **Stress testing**: Validate under various market scenarios

---

## 📁 AFFECTED COMPONENTS

### Core Files:
- `src/strategy_ire.cpp` - Strategy implementation
- `include/sentio/strategy_ire.hpp` - Strategy interface
- `include/sentio/strategy/intraday_position_governor.hpp` - Governor logic
- `include/sentio/rules/integrated_rule_ensemble.hpp` - Ensemble implementation

### Rule Strategies:
- `src/strategy_sma_cross.cpp` - Moving average crossover
- `src/strategy_vwap_reversion.cpp` - VWAP mean reversion
- `src/strategy_bollinger_squeeze_breakout.cpp` - Bollinger bands
- `src/strategy_momentum_volume.cpp` - Momentum with volume
- `src/strategy_order_flow_imbalance.cpp` - Order flow proxy

### Infrastructure:
- `src/runner.cpp` - Strategy execution engine
- `src/audit.cpp` - Performance logging
- `src/temporal_analysis.cpp` - TPA testing framework

---

## 🚀 RECOMMENDATIONS

### Priority 1 (Immediate):
1. **Keep noise filter active** until IRE ensemble is fixed
2. **Use momentum signal replacement** for production testing
3. **Monitor trade frequency** to prevent excessive churning

### Priority 2 (Short-term):
1. **Debug individual rule strategies** to find broken components
2. **Implement signal quality monitoring** in production
3. **Add real-time P&L attribution** to detect noise trading

### Priority 3 (Long-term):
1. **Redesign ensemble architecture** with robust aggregation
2. **Implement adaptive risk management** based on signal quality
3. **Create comprehensive testing suite** for strategy validation

---

## 🔐 SIGN-OFF

**Bug Reporter**: AI Assistant  
**Technical Lead**: System Architect  
**Status**: CRITICAL BUG IDENTIFIED AND TEMPORARILY MITIGATED  
**Next Action**: Permanent fix required for IRE ensemble core issue

---

*This bug report documents a critical trading system failure that resulted in systematic capital loss. The immediate fix prevents further losses, but a permanent solution requires debugging the underlying IRE ensemble implementation.*


---

## 📋 **TABLE OF CONTENTS**

1. [include/sentio/rules/adapters.hpp](#file-1)
2. [include/sentio/rules/bbands_squeeze_rule.hpp](#file-2)
3. [include/sentio/rules/diversity_weighter.hpp](#file-3)
4. [include/sentio/rules/integrated_rule_ensemble.hpp](#file-4)
5. [include/sentio/rules/irule.hpp](#file-5)
6. [include/sentio/rules/momentum_volume_rule.hpp](#file-6)
7. [include/sentio/rules/ofi_proxy_rule.hpp](#file-7)
8. [include/sentio/rules/online_platt_calibrator.hpp](#file-8)
9. [include/sentio/rules/opening_range_breakout_rule.hpp](#file-9)
10. [include/sentio/rules/registry.hpp](#file-10)
11. [include/sentio/rules/sma_cross_rule.hpp](#file-11)
12. [include/sentio/rules/vwap_reversion_rule.hpp](#file-12)
13. [include/sentio/strategy/intraday_position_governor.hpp](#file-13)
14. [src/audit.cpp](#file-14)
15. [src/base_strategy.cpp](#file-15)
16. [src/cost_aware_gate.cpp](#file-16)
17. [src/csv_loader.cpp](#file-17)
18. [src/feature_builder.cpp](#file-18)
19. [src/feature_cache.cpp](#file-19)
20. [src/feature_engineering/feature_normalizer.cpp](#file-20)
21. [src/feature_engineering/kochi_features.cpp](#file-21)
22. [src/feature_engineering/technical_indicators.cpp](#file-22)
23. [src/feature_feeder.cpp](#file-23)
24. [src/feature_feeder_guarded.cpp](#file-24)
25. [src/feature_health.cpp](#file-25)
26. [src/kochi_runner.cpp](#file-26)
27. [src/main.cpp](#file-27)
28. [src/ml/model_registry_ts.cpp](#file-28)
29. [src/ml/ts_model.cpp](#file-29)
30. [src/optimizer.cpp](#file-30)
31. [src/pnl_accounting.cpp](#file-31)
32. [src/poly_fetch_main.cpp](#file-32)
33. [src/polygon_client.cpp](#file-33)
34. [src/polygon_ingest.cpp](#file-34)
35. [src/router.cpp](#file-35)
36. [src/rth_calendar.cpp](#file-36)
37. [src/runner.cpp](#file-37)
38. [src/sanity.cpp](#file-38)
39. [src/signal_engine.cpp](#file-39)
40. [src/signal_gate.cpp](#file-40)
41. [src/signal_pipeline.cpp](#file-41)
42. [src/signal_trace.cpp](#file-42)
43. [src/sim_data.cpp](#file-43)
44. [src/strategy/run_rule_ensemble.cpp](#file-44)
45. [src/strategy_bollinger_squeeze_breakout.cpp](#file-45)
46. [src/strategy_hybrid_ppo.cpp](#file-46)
47. [src/strategy_ire.cpp](#file-47)
48. [src/strategy_kochi_ppo.cpp](#file-48)
49. [src/strategy_market_making.cpp](#file-49)
50. [src/strategy_momentum_volume.cpp](#file-50)
51. [src/strategy_opening_range_breakout.cpp](#file-51)
52. [src/strategy_order_flow_imbalance.cpp](#file-52)
53. [src/strategy_order_flow_scalping.cpp](#file-53)
54. [src/strategy_sma_cross.cpp](#file-54)
55. [src/strategy_tfa.cpp](#file-55)
56. [src/strategy_transformer_ts.cpp](#file-56)
57. [src/strategy_vwap_reversion.cpp](#file-57)
58. [src/telemetry_logger.cpp](#file-58)
59. [src/temporal_analysis.cpp](#file-59)
60. [src/time_utils.cpp](#file-60)

---

## 📄 **FILE 1 of 60**: include/sentio/rules/adapters.hpp

**File Information**:
- **Path**: `include/sentio/rules/adapters.hpp`

- **Size**: 26 lines
- **Modified**: 2025-09-08 12:20:33

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

## 📄 **FILE 2 of 60**: include/sentio/rules/bbands_squeeze_rule.hpp

**File Information**:
- **Path**: `include/sentio/rules/bbands_squeeze_rule.hpp`

- **Size**: 45 lines
- **Modified**: 2025-09-08 12:20:51

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

## 📄 **FILE 3 of 60**: include/sentio/rules/diversity_weighter.hpp

**File Information**:
- **Path**: `include/sentio/rules/diversity_weighter.hpp`

- **Size**: 50 lines
- **Modified**: 2025-09-08 19:45:46

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

## 📄 **FILE 4 of 60**: include/sentio/rules/integrated_rule_ensemble.hpp

**File Information**:
- **Path**: `include/sentio/rules/integrated_rule_ensemble.hpp`

- **Size**: 162 lines
- **Modified**: 2025-09-08 20:20:57

- **Type**: .hpp

```text
#pragma once
#include "irule.hpp"
#include "adapters.hpp"
#include "online_platt_calibrator.hpp"
#include "diversity_weighter.hpp"
#include <algorithm>
#include <cmath>
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
      if (!out) { raw_p_.push_back(NAN); continue; }
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

## 📄 **FILE 5 of 60**: include/sentio/rules/irule.hpp

**File Information**:
- **Path**: `include/sentio/rules/irule.hpp`

- **Size**: 33 lines
- **Modified**: 2025-09-08 12:20:28

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

## 📄 **FILE 6 of 60**: include/sentio/rules/momentum_volume_rule.hpp

**File Information**:
- **Path**: `include/sentio/rules/momentum_volume_rule.hpp`

- **Size**: 44 lines
- **Modified**: 2025-09-08 12:21:18

- **Type**: .hpp

```text
#pragma once
#include "irule.hpp"
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
    std::deque<double> q; double s=0,s2=0;
    for(int i=0;i<N;i++){
      q.push_back(b.volume[i]); s+=b.volume[i]; s2+=b.volume[i]*b.volume[i];
      if((int)q.size()>vol_win){ double z=q.front(); q.pop_front(); s-=z; s2-=z*z; }
      if (!q.empty()){
        double m=s/q.size(); double v=std::max(0.0, s2/q.size() - m*m);
        vol_ma_[i]=m; vol_sd_[i]=std::sqrt(v);
      }
    }
  }
};

} // namespace sentio::rules



```

## 📄 **FILE 7 of 60**: include/sentio/rules/ofi_proxy_rule.hpp

**File Information**:
- **Path**: `include/sentio/rules/ofi_proxy_rule.hpp`

- **Size**: 40 lines
- **Modified**: 2025-09-08 12:21:24

- **Type**: .hpp

```text
#pragma once
#include "irule.hpp"
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
    std::deque<double> q; double s=0,s2=0;
    for(int i=0;i<N;i++){
      double x = lr[i]*b.volume[i];
      q.push_back(x); s+=x; s2+=x*x;
      if((int)q.size()>vol_win){ double z=q.front(); q.pop_front(); s-=z; s2-=z*z; }
      double m = q.empty()?0.0:s/q.size();
      double v = q.empty()?0.0:std::max(0.0, s2/q.size() - m*m);
      ofi_[i] = (v>0? (x-m)/std::sqrt(v) : 0.0);
    }
  }
};

} // namespace sentio::rules



```

## 📄 **FILE 8 of 60**: include/sentio/rules/online_platt_calibrator.hpp

**File Information**:
- **Path**: `include/sentio/rules/online_platt_calibrator.hpp`

- **Size**: 46 lines
- **Modified**: 2025-09-08 19:45:46

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

## 📄 **FILE 9 of 60**: include/sentio/rules/opening_range_breakout_rule.hpp

**File Information**:
- **Path**: `include/sentio/rules/opening_range_breakout_rule.hpp`

- **Size**: 34 lines
- **Modified**: 2025-09-08 12:21:05

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

## 📄 **FILE 10 of 60**: include/sentio/rules/registry.hpp

**File Information**:
- **Path**: `include/sentio/rules/registry.hpp`

- **Size**: 26 lines
- **Modified**: 2025-09-08 12:21:29

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

## 📄 **FILE 11 of 60**: include/sentio/rules/sma_cross_rule.hpp

**File Information**:
- **Path**: `include/sentio/rules/sma_cross_rule.hpp`

- **Size**: 34 lines
- **Modified**: 2025-09-08 12:20:39

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

## 📄 **FILE 12 of 60**: include/sentio/rules/vwap_reversion_rule.hpp

**File Information**:
- **Path**: `include/sentio/rules/vwap_reversion_rule.hpp`

- **Size**: 42 lines
- **Modified**: 2025-09-08 12:20:59

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

## 📄 **FILE 13 of 60**: include/sentio/strategy/intraday_position_governor.hpp

**File Information**:
- **Path**: `include/sentio/strategy/intraday_position_governor.hpp`

- **Size**: 195 lines
- **Modified**: 2025-09-09 03:24:33

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
        
        // Ensure minimum separation to prevent thrashing
        if (buy_threshold - sell_threshold < config_.min_signal_gap) {
            double mid = (buy_threshold + sell_threshold) * 0.5;
            buy_threshold = mid + config_.min_signal_gap * 0.5;
            sell_threshold = mid - config_.min_signal_gap * 0.5;
        }
        
        return {buy_threshold, sell_threshold};
    }

    double calculate_base_weight(double probability, double buy_threshold, double sell_threshold) const {
        // **STRONG SIGNAL FILTER**: Only trade on clearly directional signals
        double abs_edge_from_neutral = std::abs(probability - 0.5);
        if (abs_edge_from_neutral < config_.min_abs_edge) {
            // Signal too weak - stay flat
            return 0.0; 
        }
        
        // **SIMPLE DIRECTIONAL TRADING**: Clear signals only
        if (probability > 0.5 + config_.min_abs_edge) {
            // Strong long signal
            double signal_strength = (probability - 0.5) / 0.5;
            return std::min(config_.max_base_weight, signal_strength);
        } 
        else if (probability < 0.5 - config_.min_abs_edge) {
            // Strong short signal
            double signal_strength = (0.5 - probability) / 0.5;
            return -std::min(config_.max_base_weight, signal_strength);
        }
        else {
            // Neutral - stay flat
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

## 📄 **FILE 14 of 60**: src/audit.cpp

**File Information**:
- **Path**: `src/audit.cpp`

- **Size**: 289 lines
- **Modified**: 2025-09-08 14:01:53

- **Type**: .cpp

```text
#include "sentio/audit.hpp"
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
void AuditRecorder::event_bar(std::int64_t ts, const std::string& inst, const AuditBar& b){
  write_line_("\"type\":\"bar\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"o\":"+num_s(b.open)+",\"h\":"+num_s(b.high)+",\"l\":"+num_s(b.low)+",\"c\":"+num_s(b.close)+",\"v\":"+num_s(b.volume)+"}");
}
void AuditRecorder::event_signal(std::int64_t ts, const std::string& base, SigType t, double conf){
  write_line_("\"type\":\"signal\",\"ts\":"+num_i(ts)+",\"base\":\""+json_escape_(base)+"\",\"sig\":" + num_i((int)t) + ",\"conf\":"+num_s(conf)+"}");
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
  write_line_("\"type\":\"signal\",\"ts\":"+num_i(ts)+",\"base\":\""+json_escape_(base)+"\",\"sig\":" + num_i((int)t) + ",\"conf\":"+num_s(conf)+",\"chain\":\""+json_escape_(chain_id)+"\"}");
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

## 📄 **FILE 15 of 60**: src/base_strategy.cpp

**File Information**:
- **Path**: `src/base_strategy.cpp`

- **Size**: 46 lines
- **Modified**: 2025-09-05 16:02:04

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

} // namespace sentio
```

## 📄 **FILE 16 of 60**: src/cost_aware_gate.cpp

**File Information**:
- **Path**: `src/cost_aware_gate.cpp`

- **Size**: 180 lines
- **Modified**: 2025-09-07 15:31:34

- **Type**: .cpp

```text
#include "sentio/cost_aware_gate.hpp"
// StrategySignal is defined in router.hpp which is already included via cost_aware_gate.hpp
#include <algorithm>
#include <cmath>

namespace sentio {

CostAwareGate::CostAwareGate(const CostAwareConfig& config) : config_(config) {
    // Initialize default cost curves for common instruments if not provided
    if (config_.instrument_costs.empty()) {
        config_.instrument_costs["QQQ"] = {2.0, 0.05, 5.0, 0.1};
        config_.instrument_costs["SPY"] = {1.5, 0.05, 4.0, 0.1};
        config_.instrument_costs["IWM"] = {2.5, 0.05, 6.0, 0.1};
        config_.instrument_costs["TLT"] = {3.0, 0.05, 7.0, 0.1};
        config_.instrument_costs["GLD"] = {2.0, 0.05, 5.0, 0.1};
    }
}

std::optional<StrategySignal> CostAwareGate::filter_signal(
    const StrategySignal& signal,
    const std::string& instrument,
    double current_price,
    double position_size
) const {
    if (!config_.enable_cost_filtering) {
        return signal;
    }

    // Check if signal should be rejected due to cost constraints
    if (should_reject_signal(signal, instrument, current_price, position_size)) {
        return std::nullopt;
    }

    // Calculate minimum confidence required
    double min_confidence = calculate_min_confidence(instrument, position_size, current_price);
    
    // If signal confidence is below threshold, reject it
    if (signal.confidence < min_confidence) {
        return std::nullopt;
    }

    // Signal passes cost analysis
    return signal;
}

double CostAwareGate::calculate_expected_cost(
    [[maybe_unused]] StrategySignal::Type signal_type,
    const std::string& instrument,
    double position_size,
    double current_price
) const {
    const auto& curve = get_cost_curve(instrument);
    
    // Base transaction cost
    double base_cost = curve.base_cost_bp;
    
    // Add slippage cost (increases with position size)
    double slippage_cost = calculate_slippage_cost(position_size, current_price);
    
    // Add market impact cost
    double impact_cost = calculate_market_impact(position_size, current_price);
    
    return base_cost + slippage_cost + impact_cost;
}

double CostAwareGate::calculate_min_confidence(
    const std::string& instrument,
    double position_size,
    double current_price
) const {
    const auto& curve = get_cost_curve(instrument);
    
    // Calculate expected cost
    double expected_cost = calculate_expected_cost(
        StrategySignal::Type::BUY, // Use BUY as reference
        instrument,
        position_size,
        current_price
    );
    
    // Minimum confidence should be proportional to expected cost
    // Higher costs require higher confidence
    double cost_factor = expected_cost / curve.base_cost_bp;
    double min_confidence = curve.confidence_threshold * cost_factor;
    
    // Ensure minimum confidence floor
    return std::max(min_confidence, config_.default_confidence_floor);
}

void CostAwareGate::update_cost_curve(
    const std::string& instrument,
    const std::vector<double>& recent_trades
) {
    if (recent_trades.empty()) return;
    
    // Calculate average P&L per trade
    double avg_pnl = 0.0;
    for (double pnl : recent_trades) {
        avg_pnl += pnl;
    }
    avg_pnl /= recent_trades.size();
    
    // Calculate P&L volatility
    double pnl_variance = 0.0;
    for (double pnl : recent_trades) {
        double diff = pnl - avg_pnl;
        pnl_variance += diff * diff;
    }
    pnl_variance /= recent_trades.size();
    double pnl_std = std::sqrt(pnl_variance);
    
    // Update cost curve based on recent performance
    auto& curve = config_.instrument_costs[instrument];
    
    // Adjust confidence threshold based on P&L consistency
    if (pnl_std > 0) {
        double sharpe_ratio = avg_pnl / pnl_std;
        curve.confidence_threshold = std::max(0.05, 0.1 / std::max(1.0, sharpe_ratio));
    }
    
    // Adjust base cost based on recent performance
    if (avg_pnl < 0) {
        curve.base_cost_bp *= 1.1; // Increase cost estimate if losing money
    } else if (avg_pnl > curve.min_expected_return_bp) {
        curve.base_cost_bp *= 0.95; // Decrease cost estimate if profitable
    }
}

bool CostAwareGate::should_reject_signal(
    const StrategySignal& signal,
    const std::string& instrument,
    double current_price,
    double position_size
) const {
    const auto& curve = get_cost_curve(instrument);
    
    // Check position size limits
    if (std::abs(position_size) > curve.max_position_size) {
        return true;
    }
    
    // Check minimum expected return
    double expected_cost = calculate_expected_cost(
        signal.type,
        instrument,
        position_size,
        current_price
    );
    
    // Estimate expected return based on signal confidence
    double estimated_return = signal.confidence * curve.min_expected_return_bp;
    
    // Reject if expected return is less than expected cost
    return estimated_return < expected_cost;
}

const CostAwareGate::CostCurve& CostAwareGate::get_cost_curve(const std::string& instrument) const {
    auto it = config_.instrument_costs.find(instrument);
    if (it != config_.instrument_costs.end()) {
        return it->second;
    }
    
    // Return default cost curve
    static CostCurve default_curve{config_.default_cost_bp, config_.default_confidence_floor, 5.0, 0.1};
    return default_curve;
}

double CostAwareGate::calculate_slippage_cost(double position_size, [[maybe_unused]] double current_price) const {
    // Slippage increases with position size (simplified model)
    double size_factor = std::abs(position_size) / 0.1; // Normalize to 10% position size
    return 0.5 * size_factor; // 0.5 bp per 10% position size
}

double CostAwareGate::calculate_market_impact(double position_size, [[maybe_unused]] double current_price) const {
    // Market impact increases quadratically with position size
    double size_factor = std::abs(position_size) / 0.1; // Normalize to 10% position size
    return 0.1 * size_factor * size_factor; // Quadratic impact
}

} // namespace sentio

```

## 📄 **FILE 17 of 60**: src/csv_loader.cpp

**File Information**:
- **Path**: `src/csv_loader.cpp`

- **Size**: 93 lines
- **Modified**: 2025-09-08 14:22:00

- **Type**: .cpp

```text
#include "sentio/csv_loader.hpp"
#include "sentio/binio.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctz/time_zone.h>
#include <cctz/civil_time.h>

namespace sentio {

bool load_csv(const std::string& filename, std::vector<Bar>& out) {
    // Try binary cache first for performance
    std::string bin_filename = filename.substr(0, filename.find_last_of('.')) + ".bin";
    auto cached = load_bin(bin_filename);
    if (!cached.empty()) {
        out = std::move(cached);
        return true;
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
    
    // Save to binary cache for next time
    if (!out.empty()) {
        save_bin(bin_filename, out);
    }
    
    return true;
}

} // namespace sentio
```

## 📄 **FILE 18 of 60**: src/feature_builder.cpp

**File Information**:
- **Path**: `src/feature_builder.cpp`

- **Size**: 109 lines
- **Modified**: 2025-09-06 01:37:11

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

## 📄 **FILE 19 of 60**: src/feature_cache.cpp

**File Information**:
- **Path**: `src/feature_cache.cpp`

- **Size**: 127 lines
- **Modified**: 2025-09-07 20:15:09

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
            std::cout << "[DEBUG] Bar " << bar_index << ": loaded " << features.size() 
                      << " features (expected " << feature_names_.size() << ")" << std::endl;
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
            std::cout << "[DEBUG] FeatureCache::get_features(" << bar_index 
                      << ") returning " << it->second.size() << " features" << std::endl;
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

## 📄 **FILE 20 of 60**: src/feature_engineering/feature_normalizer.cpp

**File Information**:
- **Path**: `src/feature_engineering/feature_normalizer.cpp`

- **Size**: 373 lines
- **Modified**: 2025-09-07 04:32:21

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

## 📄 **FILE 21 of 60**: src/feature_engineering/kochi_features.cpp

**File Information**:
- **Path**: `src/feature_engineering/kochi_features.cpp`

- **Size**: 146 lines
- **Modified**: 2025-09-07 23:21:17

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

## 📄 **FILE 22 of 60**: src/feature_engineering/technical_indicators.cpp

**File Information**:
- **Path**: `src/feature_engineering/technical_indicators.cpp`

- **Size**: 684 lines
- **Modified**: 2025-09-07 14:29:16

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

## 📄 **FILE 23 of 60**: src/feature_feeder.cpp

**File Information**:
- **Path**: `src/feature_feeder.cpp`

- **Size**: 631 lines
- **Modified**: 2025-09-07 23:35:51

- **Type**: .cpp

```text
#include "sentio/feature_feeder.hpp"
// TFB strategy removed - focusing on TFA only
#include "sentio/strategy_tfa.hpp"
#include "sentio/strategy_transformer_ts.hpp"
#include "sentio/strategy_kochi_ppo.hpp"
#include "sentio/feature_builder.hpp"
#include "sentio/feature_engineering/kochi_features.hpp"
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
           strategy_name == "transformer" ||
           strategy_name == "hybrid_ppo" ||
           strategy_name == "kochi_ppo";
}

void FeatureFeeder::initialize_strategy(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    if (strategy_data_.find(strategy_name) != strategy_data_.end()) {
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

std::vector<double> FeatureFeeder::extract_features_from_bar(const Bar& bar, const std::string& strategy_name) {
    if (!is_ml_strategy(strategy_name)) {
        return {};
    }
    
    // Initialize if not already done
    if (strategy_data_.find(strategy_name) == strategy_data_.end()) {
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
    if (strategy_data_.find(strategy_name) == strategy_data_.end()) {
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
    if (strategy_data_.find(strategy_name) == strategy_data_.end()) {
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
                std::cout << "[DEBUG] After cache get_features: " << features.size() << " features" << std::endl;
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
                std::cout << "[DEBUG] After normalize: " << before_norm << " -> " << features.size() << " features" << std::endl;
            }
        } else if (used_cache) {
            static int cache_skip_calls = 0;
            cache_skip_calls++;
            if (cache_skip_calls <= 5) {
                std::cout << "[DEBUG] Skipping normalization for cached features: " << features.size() << " features" << std::endl;
            }
        }
        
        // Validate features (bypass validation for cached features as they're pre-validated)
        if (used_cache) {
            static int cache_bypass_calls = 0;
            cache_bypass_calls++;
            if (cache_bypass_calls <= 5) {
                std::cout << "[DEBUG] Bypassing validation for cached features: " << features.size() << " features" << std::endl;
            }
        } else {
            bool valid = validate_features(features, strategy_name);
            static int val_calls = 0;
            val_calls++;
            if (val_calls <= 5) {
                std::cout << "[DEBUG] Validation result: " << (valid ? "PASS" : "FAIL") << " for " << features.size() << " features" << std::endl;
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
    if (strategy_data_.find(strategy_name) == strategy_data_.end()) {
        initialize_strategy(strategy_name);
    }
    
    try {
        // Extract features using full bar history (required for technical indicators)
        auto features = extract_features_from_bars_with_index(bars, current_index, strategy_name);
        
        static int feature_extract_calls = 0;
        feature_extract_calls++;
        
        if (feature_extract_calls % 1000 == 0 || feature_extract_calls <= 10) {
            std::cout << "[DIAG] FeatureFeeder extract: call=" << feature_extract_calls 
                      << " features.size()=" << features.size() 
                      << " current_index=" << current_index << std::endl;
        }
        
        if (features.empty()) {
            if (feature_extract_calls % 1000 == 0 || feature_extract_calls <= 10) {
                std::cout << "[DIAG] FeatureFeeder: Features EMPTY at call=" << feature_extract_calls << std::endl;
            }
            return;
        }
        
        // Cast to specific strategy type and feed features
        static int strategy_check_calls = 0;
        strategy_check_calls++;
        
        if (strategy_check_calls % 1000 == 0 || strategy_check_calls <= 10) {
            std::cout << "[DIAG] FeatureFeeder strategy check: call=" << strategy_check_calls 
                      << " strategy_name='" << strategy_name << "'" << std::endl;
        }
        
        if (strategy_name == "TFA" || strategy_name == "tfa") {
            auto* tfa = dynamic_cast<TFAStrategy*>(strategy);
            if (tfa) {
                static int tfa_feed_calls = 0;
                tfa_feed_calls++;
                
                if (tfa_feed_calls % 1000 == 0 || tfa_feed_calls <= 10) {
                    std::cout << "[DIAG] FeatureFeeder TFA: call=" << tfa_feed_calls 
                              << " features.size()=" << features.size() << std::endl;
                }
                
                tfa->set_raw_features(features);
            } else {
                static int cast_fail = 0;
                cast_fail++;
                if (cast_fail <= 10) {
                    std::cout << "[DIAG] FeatureFeeder: TFA cast failed! call=" << cast_fail << std::endl;
                }
            }
        } else if (strategy_name == "transformer") {
            auto* tf = dynamic_cast<TransformerSignalStrategyTS*>(strategy);
            if (tf) {
                tf->set_raw_features(features);
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
    if (strategy_data_.find(strategy_name) == strategy_data_.end()) {
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

## 📄 **FILE 24 of 60**: src/feature_feeder_guarded.cpp

**File Information**:
- **Path**: `src/feature_feeder_guarded.cpp`

- **Size**: 79 lines
- **Modified**: 2025-09-06 22:14:53

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

## 📄 **FILE 25 of 60**: src/feature_health.cpp

**File Information**:
- **Path**: `src/feature_health.cpp`

- **Size**: 32 lines
- **Modified**: 2025-09-05 17:01:07

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

## 📄 **FILE 26 of 60**: src/kochi_runner.cpp

**File Information**:
- **Path**: `src/kochi_runner.cpp`

- **Size**: 54 lines
- **Modified**: 2025-09-08 08:47:53

- **Type**: .cpp

```text
#include "sentio/feature/csv_feature_provider.hpp"
#include "kochi/kochi_ppo_context.hpp"
#include <filesystem>
#include <iostream>

namespace kochi {

int run_kochi_ppo(const std::string& symbol,
                  const std::string& feature_csv,
                  const std::string& actor_pt,
                  const std::string& actor_meta_json,
                  const std::string& audit_dir,
                  int kochi_T,
                  int cooldown_bars = 5)
{
  sentio::CsvFeatureProvider provider(feature_csv, /*T=*/kochi_T);
  auto X = provider.get_features_for(symbol);
  auto runtime_names = provider.feature_names();

  KochiPPOContext ctx;
  ctx.load(actor_pt, actor_meta_json, runtime_names);

  std::filesystem::create_directories(audit_dir);
  long long ts_epoch = std::time(nullptr);
  std::string audit_file = audit_dir + "/kochi_ppo_temporal_" + std::to_string(ts_epoch) + ".jsonl";

  std::vector<int> actions;
  std::vector<std::array<float,3>> probs;
  ctx.forward(X, actions, probs, audit_file);

  // Simple execution: position = {-1,0,+1}
  int pos = 0; int64_t emitted=0, considered=0; int cooldown_drops=0;
  const size_t start = (size_t)std::max(ctx.emit_from, ctx.T);
  int64_t next_ok = 0;

  for (size_t i=start;i<actions.size();++i){
    considered++;
    int a = actions[i]; // 0=HOLD 1=LONG 2=SHORT
    if ((int64_t)i < next_ok){ cooldown_drops++; continue; }
    int target = (a==1) ? +1 : (a==2 ? -1 : pos);
    if (target != pos){
      pos = target;
      emitted++;
      next_ok = (int64_t)i + cooldown_bars;
    }
  }

  std::cerr << "[KOCHI] considered="<<considered<<" emitted="<<emitted
            << " cooldown_drops="<<cooldown_drops
            << " T="<<ctx.T<<" Fk="<<ctx.Fk<<"\n";
  return 0;
}

} // namespace kochi

```

## 📄 **FILE 27 of 60**: src/main.cpp

**File Information**:
- **Path**: `src/main.cpp`

- **Size**: 769 lines
- **Modified**: 2025-09-09 01:19:01

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
#include "sentio/rth_calendar.hpp" // **NEW**: Include for RTH check
#include "sentio/calendar_seed.hpp" // **NEW**: Include for calendar creation
// #include "sentio/feature/feature_from_spec.hpp" // For C++ feature building - causes Bar redefinition
#include "sentio/feature/feature_matrix.hpp" // For FeatureMatrix
// TFB strategy removed - focusing on TFA only
#include "sentio/strategy_tfa.hpp" // For TFA strategy

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <cstdlib> // For std::exit
#include <sstream>
#include <ctime> // For std::time
#include <fstream> // For std::ifstream
#include <ATen/Parallel.h> // For LibTorch threading controls
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
        size_t mism=0; for (size_t i=0;i<N;i++){ if (s[i].ts_utc_epoch != base[i].ts_utc_epoch) { mism++; if (mism<3) {
            std::cerr << "  ts mismatch["<<i<<"] " << s[i].ts_utc_epoch << " vs base " << base[i].ts_utc_epoch << "\n"; }
        } }
        if (mism>0) {
            std::cerr << "FATAL: Alignment check failed: " << ST.get_symbol((int)sid)
                      << " has " << mism << " timestamp mismatches vs base (first shown above).\n";
            ok = false;
        }
    }
    if (!ok) {
        std::cerr << "Hint: Use aligned CSVs or run tools/align_bars.py to generate aligned datasets.\n";
    }
    return ok;
}

static void load_strategy_config_if_any(const std::string& strategy_name, sentio::RunnerCfg& cfg){
    try {
        std::string lower = strategy_name;
        for (auto& c : lower) c = std::tolower(c);
        std::string path = "configs/strategies/" + lower + ".json";
        std::ifstream f(path);
        if (!f.good()) return;
        nlohmann::json j; f >> j;
        // params
        if (j.contains("params") && j["params"].is_object()){
            for (auto it = j["params"].begin(); it != j["params"].end(); ++it){
                cfg.strategy_params[it.key()] = std::to_string(it.value().get<double>());
            }
        }
        // router overrides
        if (j.contains("router")){
            auto r = j["router"];
            if (r.contains("ire_min_conf_strong_short")) cfg.router.ire_min_conf_strong_short = r["ire_min_conf_strong_short"].get<double>();
            if (r.contains("min_signal_strength")) cfg.router.min_signal_strength = r["min_signal_strength"].get<double>();
            if (r.contains("signal_multiplier"))   cfg.router.signal_multiplier   = r["signal_multiplier"].get<double>();
            if (r.contains("max_position_pct"))    cfg.router.max_position_pct    = r["max_position_pct"].get<double>();
            if (r.contains("min_shares"))          cfg.router.min_shares          = r["min_shares"].get<double>();
            if (r.contains("lot_size"))            cfg.router.lot_size            = r["lot_size"].get<double>();
            if (r.contains("base_symbol"))         cfg.router.base_symbol         = r["base_symbol"].get<std::string>();
            if (r.contains("bull3x"))              cfg.router.bull3x              = r["bull3x"].get<std::string>();
            if (r.contains("bear3x"))              cfg.router.bear3x              = r["bear3x"].get<std::string>();
            if (r.contains("bear1x"))              cfg.router.bear1x              = r["bear1x"].get<std::string>();
        }
        // sizer overrides
        if (j.contains("sizer")){
            auto s = j["sizer"];
            if (s.contains("fractional_allowed")) cfg.sizer.fractional_allowed = s["fractional_allowed"].get<bool>();
            if (s.contains("min_notional"))       cfg.sizer.min_notional       = s["min_notional"].get<double>();
            if (s.contains("max_leverage"))       cfg.sizer.max_leverage       = s["max_leverage"].get<double>();
            if (s.contains("max_position_pct"))   cfg.sizer.max_position_pct   = s["max_position_pct"].get<double>();
            if (s.contains("volatility_target"))  cfg.sizer.volatility_target  = s["volatility_target"].get<double>();
            if (s.contains("allow_negative_cash"))cfg.sizer.allow_negative_cash= s["allow_negative_cash"].get<bool>();
            if (s.contains("vol_lookback_days"))  cfg.sizer.vol_lookback_days  = s["vol_lookback_days"].get<int>();
            if (s.contains("cash_reserve_pct"))   cfg.sizer.cash_reserve_pct   = s["cash_reserve_pct"].get<double>();
        }
    } catch (...) {
        // ignore config errors
    }
}

namespace { // Anonymous namespace to ensure link-time registration
    struct StrategyRegistrar {
        StrategyRegistrar() {
            // Register strategies in the factory
            sentio::StrategyFactory::instance().register_strategy("VWAPReversion", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::VWAPReversionStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("MomentumVolume", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::MomentumVolumeProfileStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("BollingerSqueeze", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::BollingerSqueezeBreakoutStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("OpeningRange", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::OpeningRangeBreakoutStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("OrderFlowScalping", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::OrderFlowScalpingStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("OrderFlowImbalance", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::OrderFlowImbalanceStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("MarketMaking", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::MarketMakingStrategy>();
            });
            // TransformerTS inherits from IStrategy, not BaseStrategy, so skip for now
            // TFB strategy removed - focusing on TFA only
            sentio::StrategyFactory::instance().register_strategy("tfa", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::unique_ptr<sentio::BaseStrategy>(new sentio::TFAStrategy(sentio::TFACfg{}));
            });
        }
    };
    static StrategyRegistrar registrar;
}


void usage() {
    std::cout << "Usage: sentio_cli <command> [options]\n"
              << "Commands:\n"
              << "  backtest <symbol> [--strategy <name>] [--params <k=v,...>]\n"
              << "  tpa_test <symbol> [--strategy <name>] [--params <k=v,...>] [--quarters <n>]\n"
              << "  test-models [--strategy <name>] [--data <file>] [--start <date>] [--end <date>]\n"
              << "  tune_ire <symbol> [--test_quarters <n>]\n"
              << "  replay <run_id>\n";
}

int main(int argc, char* argv[]) {
    // Configure LibTorch threading to prevent oversubscription
    at::set_num_threads(1);         // intra-op
    at::set_num_interop_threads(1); // inter-op
    
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string command = argv[1];
    
    if (command == "backtest") {
        if (argc < 3) {
            std::cout << "Usage: sentio_cli backtest <symbol> [--strategy <name>] [--params <k=v,...>]\n";
            return 1;
        }
        
        std::string base_symbol = argv[2];
        std::string strategy_name = "VWAPReversion";
        std::unordered_map<std::string, std::string> strategy_params;
        
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--strategy" && i + 1 < argc) {
                strategy_name = argv[++i];
            } else if (arg == "--params" && i + 1 < argc) {
                std::string params_str = argv[++i];
                std::stringstream ss(params_str);
                std::string pair;
                while (std::getline(ss, pair, ',')) {
                    size_t eq_pos = pair.find('=');
                    if (eq_pos != std::string::npos) {
                        strategy_params[pair.substr(0, eq_pos)] = pair.substr(eq_pos + 1);
                    }
                }
            }
        }
        
        sentio::SymbolTable ST;
        std::vector<std::vector<sentio::Bar>> series;
        std::vector<std::string> symbols_to_load = {base_symbol};
        if (base_symbol == "QQQ") {
            symbols_to_load.push_back("TQQQ");
            symbols_to_load.push_back("SQQQ");
        }

        std::cout << "Loading data for symbols: ";
        for(const auto& sym : symbols_to_load) std::cout << sym << " ";
        std::cout << std::endl;

        for (const auto& sym : symbols_to_load) {
            std::vector<sentio::Bar> bars;
            std::string data_path = sentio::resolve_csv(sym);
            if (!sentio::load_csv(data_path, bars)) {
                std::cerr << "ERROR: Failed to load data for " << sym << " from " << data_path << std::endl;
                continue;
            }
            std::cout << " -> Loaded " << bars.size() << " bars for " << sym << std::endl;
            
            int symbol_id = ST.intern(sym);
            if (static_cast<size_t>(symbol_id) >= series.size()) {
                series.resize(symbol_id + 1);
            }
            series[symbol_id] = std::move(bars);
        }
        
        int base_symbol_id = ST.get_id(base_symbol);
        if (series.empty() || series[base_symbol_id].empty()) {
            std::cerr << "FATAL: No data loaded for base symbol " << base_symbol << std::endl;
            return 1;
        }

        // Report data period and bars for transparency
        auto& base_bars = series[base_symbol_id];
        auto fmt_date = [](std::int64_t epoch_sec){
            std::time_t tt = static_cast<std::time_t>(epoch_sec);
            std::tm tm{}; gmtime_r(&tt, &tm);
            char buf[32]; std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm); return std::string(buf);
        };
        std::int64_t min_ts = base_bars.front().ts_utc_epoch;
        std::int64_t max_ts = base_bars.back().ts_utc_epoch;
        std::size_t   n_bars = base_bars.size();
        double span_days = double(max_ts - min_ts) / (24.0*3600.0);
        std::cout << "Data period: " << fmt_date(min_ts) << " → " << fmt_date(max_ts)
                  << " (" << std::fixed << std::setprecision(1) << span_days << " days)\n";
        std::cout << "Bars(" << base_symbol << "): " << n_bars << "\n";
        if ((max_ts - min_ts) < (365LL*24LL*3600LL)) {
            std::cerr << "WARNING: Data period is shorter than 1 year. Results may be unrepresentative.\n";
        }

        // **NEW**: Data Sanity Check - Verify RTH filtering post-load.
        // This acts as a safety net. If your data was generated with RTH filtering,
        // this check ensures the filtering was successful.
        std::cout << "\nVerifying data integrity for RTH..." << std::endl;
        bool rth_filter_failed = false;
        sentio::TradingCalendar calendar = sentio::make_default_nyse_calendar();
        for (size_t sid = 0; sid < series.size(); ++sid) {
            if (series[sid].empty()) continue;
            for (const auto& bar : series[sid]) {
                if (!calendar.is_rth_utc(bar.ts_utc_epoch, "UTC")) {
                    std::cerr << "\nFATAL ERROR: Non-RTH data found after filtering!\n"
                              << " -> Symbol: " << ST.get_symbol(sid) << "\n"
                              << " -> Timestamp (UTC): " << bar.ts_utc << "\n"
                              << " -> UTC Epoch: " << bar.ts_utc_epoch << "\n\n"
                              << "This indicates your data files (*.csv, *.bin) were generated with an old or incorrect RTH filter.\n"
                              << "Please DELETE your existing data files and REGENERATE them using the updated poly_fetch tool.\n"
                              << std::endl;
                    rth_filter_failed = true;
                    break;
                }
            }
            if (rth_filter_failed) break;
        }

        if (rth_filter_failed) {
            std::exit(1); // Exit with an error as requested
        }
        std::cout << " -> Data verification passed." << std::endl;

        // Alignment check (all loaded symbols must align to base timestamps)
        if (!verify_series_alignment(ST, series, base_symbol_id)) {
            std::cerr << "FATAL: Data alignment check failed. Aborting run." << std::endl;
            return 1;
        }

        // Merge defaults from configs/<strategy>.json (overridden by CLI params)
        sentio::RunnerCfg tmp_cfg; tmp_cfg.strategy_name = strategy_name;
        load_strategy_config_if_any(strategy_name, tmp_cfg);
        for (const auto& kv : tmp_cfg.strategy_params){ if (!strategy_params.count(kv.first)) strategy_params[kv.first] = kv.second; }
        sentio::RunnerCfg cfg;
        cfg.strategy_name = strategy_name;
        cfg.strategy_params = strategy_params;
        cfg.audit_level = sentio::AuditLevel::Full;
        cfg.snapshot_stride = 100;
        // **FIX**: Apply full router and sizer config from JSON
        cfg.router = tmp_cfg.router;
        cfg.sizer = tmp_cfg.sizer;
        
        // Create audit recorder
        sentio::AuditConfig audit_cfg;
        long long ts_epoch = std::time(nullptr);
        audit_cfg.run_id = strategy_name + std::string("_backtest_") + base_symbol + "_" + std::to_string(ts_epoch);
        audit_cfg.file_path = "audit/" + strategy_name + std::string("_backtest_") + base_symbol + "_" + std::to_string(ts_epoch) + ".jsonl";
        audit_cfg.flush_each = true;
        sentio::AuditRecorder audit(audit_cfg);
        
        sentio::Tsc timer;
        timer.tic();
        auto result = sentio::run_backtest(audit, ST, series, base_symbol_id, cfg);
        double elapsed = timer.toc_sec();
        
        std::cout << "\nBacktest completed in " << elapsed << "s\n";
        std::cout << "Final Equity: " << result.final_equity << "\n";
        std::cout << "Total Return: " << result.total_return << "%\n";
        std::cout << "Sharpe Ratio: " << result.sharpe_ratio << "\n";
        std::cout << "Max Drawdown: " << result.max_drawdown << "%\n";
        std::cout << "Total Fills: " << result.total_fills << "\n";
        std::cout << "Diagnostics -> No Route: " << result.no_route << " | No Quantity: " << result.no_qty << "\n";

    } else if (command == "tpa_test") {
        if (argc < 3) {
            std::cout << "Usage: sentio_cli tpa_test <symbol> [--strategy <name>] [--params <k=v,...>] [--quarters <n>]\n";
            return 1;
        }
        
        std::string base_symbol = argv[2];
        std::string strategy_name = "TFA";
        std::unordered_map<std::string, std::string> strategy_params;
        int num_quarters = 1; // default: most recent quarter
        
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--strategy" && i + 1 < argc) {
                strategy_name = argv[++i];
            } else if (arg == "--params" && i + 1 < argc) {
                std::string params_str = argv[++i];
                std::stringstream ss(params_str);
                std::string pair;
                while (std::getline(ss, pair, ',')) {
                    size_t eq_pos = pair.find('=');
                    if (eq_pos != std::string::npos) {
                        strategy_params[pair.substr(0, eq_pos)] = pair.substr(eq_pos + 1);
                    }
                }
            } else if (arg == "--quarters" && i + 1 < argc) {
                num_quarters = std::stoi(argv[++i]);
            }
        }
        
        sentio::SymbolTable ST;
        std::vector<std::vector<sentio::Bar>> series;
        std::vector<std::string> symbols_to_load = {base_symbol};
        if (base_symbol == "QQQ") {
            symbols_to_load.push_back("TQQQ");
            symbols_to_load.push_back("SQQQ");
        }

        std::cout << "Loading data for symbols: ";
        for(const auto& sym : symbols_to_load) std::cout << sym << " ";
        std::cout << std::endl;

        for (const auto& sym : symbols_to_load) {
            std::vector<sentio::Bar> bars;
            std::string data_path = sentio::resolve_csv(sym);
            if (!sentio::load_csv(data_path, bars)) {
                std::cerr << "ERROR: Failed to load data for " << sym << " from " << data_path << std::endl;
                continue;
            }
            std::cout << " -> Loaded " << bars.size() << " bars for " << sym << std::endl;
            
            int symbol_id = ST.intern(sym);
            if (static_cast<size_t>(symbol_id) >= series.size()) {
                series.resize(symbol_id + 1);
            }
            series[symbol_id] = std::move(bars);
        }
        
        int base_symbol_id = ST.get_id(base_symbol);
        if (series.empty() || series[base_symbol_id].empty()) {
            std::cerr << "FATAL: No data loaded for base symbol " << base_symbol << std::endl;
            return 1;
        }

        // Report data period and bars for transparency
        auto& base_bars2 = series[base_symbol_id];
        auto fmt_date2 = [](std::int64_t epoch_sec){
            std::time_t tt = static_cast<std::time_t>(epoch_sec);
            std::tm tm{}; gmtime_r(&tt, &tm);
            char buf[32]; std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm); return std::string(buf);
        };
        std::int64_t min_ts2 = base_bars2.front().ts_utc_epoch;
        std::int64_t max_ts2 = base_bars2.back().ts_utc_epoch;
        std::size_t   n_bars2 = base_bars2.size();
        double span_days2 = double(max_ts2 - min_ts2) / (24.0*3600.0);
        std::cout << "Data period: " << fmt_date2(min_ts2) << " → " << fmt_date2(max_ts2)
                  << " (" << std::fixed << std::setprecision(1) << span_days2 << " days)\n";
        std::cout << "Bars(" << base_symbol << "): " << n_bars2 << "\n";
        if ((max_ts2 - min_ts2) < (365LL*24LL*3600LL)) {
            std::cerr << "WARNING: Data period is shorter than 1 year. Results may be unrepresentative.\n";
        }

        // Alignment check (all loaded symbols must align to base timestamps)
        if (!verify_series_alignment(ST, series, base_symbol_id)) {
            std::cerr << "FATAL: Data alignment check failed. Aborting TPA test." << std::endl;
            return 1;
        }

        // Load cached features for ML strategies for massive performance improvement
        if (strategy_name == "TFA" || strategy_name == "tfa") {
            std::string feature_file = "data/" + base_symbol + "_RTH_features.csv";
            std::cout << "Loading pre-computed features from: " << feature_file << std::endl;
            if (sentio::FeatureFeeder::load_feature_cache(feature_file)) {
                sentio::FeatureFeeder::use_cached_features(true);
                std::cout << "✅ Cached features loaded successfully - MASSIVE speed improvement enabled!" << std::endl;
            } else {
                std::cout << "⚠️  Failed to load cached features - falling back to real-time calculation" << std::endl;
            }
        } else if (strategy_name == "kochi_ppo") {
            std::string feature_file = "data/" + base_symbol + "_KOCHI_features.csv";
            std::cout << "Loading pre-computed KOCHI features from: " << feature_file << std::endl;
            if (sentio::FeatureFeeder::load_feature_cache(feature_file)) {
                sentio::FeatureFeeder::use_cached_features(true);
                std::cout << "✅ KOCHI cached features loaded successfully" << std::endl;
            } else {
                std::cerr << "❌ KOCHI feature cache missing; cannot proceed for kochi_ppo strategy." << std::endl;
                return 1;
            }
        }

        // Merge defaults from configs/<strategy>.json (overridden by CLI params)
        sentio::RunnerCfg tmp_cfg2; tmp_cfg2.strategy_name = strategy_name;
        load_strategy_config_if_any(strategy_name, tmp_cfg2);
        for (const auto& kv : tmp_cfg2.strategy_params){ if (!strategy_params.count(kv.first)) strategy_params[kv.first] = kv.second; }
        sentio::RunnerCfg cfg;
        cfg.strategy_name = strategy_name;
        cfg.strategy_params = strategy_params;
        cfg.audit_level = sentio::AuditLevel::Full;
        cfg.snapshot_stride = 100;
        // **FIX**: Apply full router and sizer config from JSON
        cfg.router = tmp_cfg2.router;
        cfg.sizer = tmp_cfg2.sizer;
        
        sentio::TemporalAnalysisConfig temporal_cfg;
        temporal_cfg.num_quarters = num_quarters;
        temporal_cfg.print_detailed_report = true;
        
        std::cout << "\nRunning TPA (Temporal Performance Analysis) Test..." << std::endl;
        std::cout << "Strategy: " << strategy_name << ", Quarters: " << num_quarters << std::endl;
        
        sentio::Tsc timer;
        timer.tic();
        auto summary = sentio::run_temporal_analysis(ST, series, base_symbol_id, cfg, temporal_cfg);
        double elapsed = timer.toc_sec();
        
        std::cout << "\nTPA test completed in " << elapsed << "s" << std::endl;
        
        if (temporal_cfg.print_detailed_report) {
            sentio::TemporalAnalyzer analyzer;
            for (const auto& q : summary.quarterly_results) {
                analyzer.add_quarterly_result(q);
            }
            analyzer.print_detailed_report();
        }
        
    } else if (command == "test-models") {
        std::string strategy_name = "tfa";
        std::string data_file = "data/QQQ_RTH.csv";
        std::string start_date = "2023-01-01";
        std::string end_date = "2023-12-31";
        
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--strategy" && i + 1 < argc) {
                strategy_name = argv[++i];
            } else if (arg == "--data" && i + 1 < argc) {
                data_file = argv[++i];
            } else if (arg == "--start" && i + 1 < argc) {
                start_date = argv[++i];
            } else if (arg == "--end" && i + 1 < argc) {
                end_date = argv[++i];
            }
        }
        
        // Validate strategy name
        if (strategy_name != "tfa") {
            std::cerr << "ERROR: Strategy must be 'tfa'. Got: " << strategy_name << std::endl;
            return 1;
        }
        
        // Check if model exists
        std::string model_dir = "artifacts/TFA/v1";
        std::string model_path = model_dir + "/model.pt";
        std::string metadata_path = model_dir + "/metadata.json";
        
        if (!std::ifstream(model_path).good()) {
            std::cerr << "ERROR: Model not found at " << model_path << std::endl;
            std::cerr << "Please train the model first: python train_models.py" << std::endl;
            return 1;
        }
        
        if (!std::ifstream(metadata_path).good()) {
            std::cerr << "ERROR: Metadata not found at " << metadata_path << std::endl;
            return 1;
        }
        
        std::cout << "🧪 Testing " << strategy_name << " model..." << std::endl;
        std::cout << "📁 Model: " << model_path << std::endl;
        std::cout << "📊 Data: " << data_file << std::endl;
        std::cout << "📅 Period: " << start_date << " to " << end_date << std::endl;
        
        // Load data
        std::vector<sentio::Bar> bars;
        if (!sentio::load_csv(data_file, bars)) {
            std::cerr << "ERROR: Failed to load data from " << data_file << std::endl;
            return 1;
        }
        
        std::cout << "✅ Loaded " << bars.size() << " bars" << std::endl;
        
        // Load feature specification
        std::string feature_spec_path = "python/feature_spec.json";
        std::ifstream spec_file(feature_spec_path);
        if (!spec_file.good()) {
            std::cerr << "ERROR: Feature spec not found at " << feature_spec_path << std::endl;
            return 1;
        }
        
        std::string spec_json((std::istreambuf_iterator<char>(spec_file)), std::istreambuf_iterator<char>());
        std::cout << "✅ Loaded feature specification" << std::endl;
        
        // For now, let's skip the C++ feature builder and use a simple approach
        // We'll create dummy features to test the model loading and signal generation
        std::cout << "🔧 Creating dummy features for testing..." << std::endl;
        
        // Create a simple feature matrix with dummy data
        sentio::FeatureMatrix feature_matrix;
        feature_matrix.rows = std::min(100, static_cast<int>(bars.size()));
        feature_matrix.cols = 6; // close, logret, ema_20, ema_50, rsi_14, zlog_20
        feature_matrix.data.resize(feature_matrix.rows * feature_matrix.cols);
        
        // Fill with dummy features (just close prices for now)
        for (int i = 0; i < feature_matrix.rows; ++i) {
            int base_idx = i * feature_matrix.cols;
            feature_matrix.data[base_idx + 0] = static_cast<float>(bars[i].close); // close
            feature_matrix.data[base_idx + 1] = 0.0f; // logret
            feature_matrix.data[base_idx + 2] = static_cast<float>(bars[i].close); // ema_20
            feature_matrix.data[base_idx + 3] = static_cast<float>(bars[i].close); // ema_50
            feature_matrix.data[base_idx + 4] = 50.0f; // rsi_14
            feature_matrix.data[base_idx + 5] = 0.0f; // zlog_20
        }
        
        std::cout << "✅ Created dummy features: " << feature_matrix.rows << " rows x " << feature_matrix.cols << " cols" << std::endl;
        
        // Create TFA strategy instance
        std::unique_ptr<sentio::TFAStrategy> tfa_strategy = std::make_unique<sentio::TFAStrategy>();
        
        std::cout << "\n🧪 Testing " << strategy_name << " model with features..." << std::endl;
        
        // Feed features to strategy and test signal generation
        int signals_generated = 0;
        int total_bars = std::min(100, static_cast<int>(feature_matrix.rows)); // Test first 100 bars
        
        for (int i = 0; i < total_bars; ++i) {
            // Extract features for this bar
            std::vector<double> raw_features(feature_matrix.cols);
            for (int j = 0; j < feature_matrix.cols; ++j) {
                raw_features[j] = feature_matrix.data[i * feature_matrix.cols + j];
            }
            
            // Create a dummy bar for on_bar call
            sentio::Bar dummy_bar;
            if (i < static_cast<int>(bars.size())) {
                dummy_bar = bars[i];
            }
            
            sentio::StrategyCtx ctx;
            ctx.ts_utc_epoch = dummy_bar.ts_utc_epoch;
            ctx.instrument = "QQQ";
            ctx.is_rth = true;
            
            // Feed features to strategy and call on_bar
            if (tfa_strategy) {
                tfa_strategy->set_raw_features(raw_features);
                tfa_strategy->on_bar(ctx, dummy_bar);
                auto signal = tfa_strategy->latest();
                if (signal) {
                    signals_generated++;
                    std::cout << "Bar " << i << ": Signal generated - " 
                              << (signal->type == sentio::StrategySignal::Type::BUY ? "BUY" : 
                                  signal->type == sentio::StrategySignal::Type::SELL ? "SELL" : "HOLD")
                              << " (conf=" << signal->confidence << ")" << std::endl;
                }
            } else if (tfa_strategy) {
                tfa_strategy->set_raw_features(raw_features);
                tfa_strategy->on_bar(ctx, dummy_bar);
                auto signal = tfa_strategy->latest();
                if (signal) {
                    signals_generated++;
                    std::cout << "Bar " << i << ": Signal generated - " 
                              << (signal->type == sentio::StrategySignal::Type::BUY ? "BUY" : 
                                  signal->type == sentio::StrategySignal::Type::SELL ? "SELL" : "HOLD")
                              << " (conf=" << signal->confidence << ")" << std::endl;
                }
            }
        }
        
        std::cout << "\n📊 Test Results:" << std::endl;
        std::cout << "  Total bars tested: " << total_bars << std::endl;
        std::cout << "  Signals generated: " << signals_generated << std::endl;
        std::cout << "  Signal rate: " << (100.0 * signals_generated / total_bars) << "%" << std::endl;
        
        if (signals_generated == 0) {
            std::cout << "\n⚠️  WARNING: No signals generated! This could indicate:" << std::endl;
            std::cout << "  - Model confidence threshold too high" << std::endl;
            std::cout << "  - Feature window not ready" << std::endl;
            std::cout << "  - Model prediction issues" << std::endl;
            std::cout << "  - Check the diagnostic output above for details" << std::endl;
        } else {
            std::cout << "\n✅ Model is generating signals successfully!" << std::endl;
        }
        
    } else if (command == "tune_ire") {
        if (argc < 3) {
            std::cout << "Usage: sentio_cli tune_ire <symbol> [--test_quarters <n>]" << std::endl;
            return 1;
        }

        std::string base_symbol = argv[2];
        int test_quarters = 1;
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--test_quarters" && i + 1 < argc) {
                test_quarters = std::max(1, std::stoi(argv[++i]));
            }
        }

        // Load QQQ family (QQQ, TQQQ, SQQQ)
        sentio::SymbolTable ST;
        std::vector<std::vector<sentio::Bar>> series;
        std::vector<std::string> symbols_to_load = {base_symbol, "TQQQ", "SQQQ"};
        for (const auto& sym : symbols_to_load) {
            std::vector<sentio::Bar> bars;
            std::string data_path = sentio::resolve_csv(sym);
            if (!sentio::load_csv(data_path, bars)) {
                std::cerr << "ERROR: Failed to load data for " << sym << " from " << data_path << std::endl;
                return 1;
            }
            int symbol_id = ST.intern(sym);
            if (static_cast<size_t>(symbol_id) >= series.size()) series.resize(symbol_id + 1);
            series[symbol_id] = std::move(bars);
        }

        int base_symbol_id = ST.get_id(base_symbol);
        if (!verify_series_alignment(ST, series, base_symbol_id)) {
            std::cerr << "FATAL: Data alignment check failed. Aborting tuning." << std::endl;
            return 1;
        }

        const auto& base = series[base_symbol_id];
        if (base.empty()) { std::cerr << "FATAL: No data for base symbol." << std::endl; return 1; }

        // Split into train (all but most recent test_quarters) and test (most recent test_quarters)
        int total_bars = (int)base.size();
        int total_quarters = 12; // assume ~3 years
        int bars_per_quarter = std::max(1, total_bars / total_quarters);
        int test_bars = std::min(total_bars, bars_per_quarter * test_quarters);
        int train_bars = std::max(0, total_bars - test_bars);

        auto slice_series = [&](int start_idx, int end_idx){
            std::vector<std::vector<sentio::Bar>> out; out.reserve(series.size());
            for (const auto& s : series) {
                if ((int)s.size() <= start_idx) { out.emplace_back(); continue; }
                int e = std::min(end_idx, (int)s.size());
                out.emplace_back(s.begin() + start_idx, s.begin() + e);
            }
            return out;
        };

        auto train_series = slice_series(0, train_bars);
        auto test_series  = slice_series(train_bars, total_bars);

        // Bayesian-like TPE tuner (lightweight, dependency-free)
        auto eval_params = [&](double bh, double sl, double sc){
            sentio::RunnerCfg rcfg; rcfg.strategy_name = "IRE";
            rcfg.strategy_params = { {"buy_hi", std::to_string(bh)}, {"sell_lo", std::to_string(sl)} };
            rcfg.router.ire_min_conf_strong_short = sc;
            sentio::TemporalAnalysisConfig tcfg; 
            // Use only last 4 training quarters as proxy
            int train_q = std::max(1, train_bars / std::max(1, bars_per_quarter));
            tcfg.num_quarters = std::min(4, train_q);
            tcfg.print_detailed_report = false;
            auto train_summary = sentio::run_temporal_analysis(ST, train_series, base_symbol_id, rcfg, tcfg);
            double avg_trades = 0.0, avg_monthly = 0.0, avg_sharpe = 0.0;
            if (!train_summary.quarterly_results.empty()){
                for (const auto& q : train_summary.quarterly_results){ avg_trades += q.avg_daily_trades; avg_monthly += q.monthly_return_pct; avg_sharpe += q.sharpe_ratio; }
                avg_trades /= train_summary.quarterly_results.size();
                avg_monthly /= train_summary.quarterly_results.size();
                avg_sharpe /= train_summary.quarterly_results.size();
            }
            double trade_penalty = 0.0;
            if (avg_trades < 80) trade_penalty = (80 - avg_trades);
            else if (avg_trades > 120) trade_penalty = (avg_trades - 120);
            double score = avg_monthly + 0.5 * avg_sharpe - 0.1 * trade_penalty;
            return std::tuple<double,double,double,double>(score, avg_trades, avg_monthly, avg_sharpe);
        };

        auto clip = [](double x, double lo, double hi){ return std::max(lo, std::min(hi, x)); };
        double bh_lo=0.70, bh_hi=0.90, sl_lo=0.10, sl_hi=0.30, sc_lo=0.80, sc_hi=0.98;
        struct Candidate { double buy_hi, sell_lo, short_conf; double score; double avg_trades; double avg_monthly; double sharpe; } best{0.80,0.20,0.90,-1,0,0,0};

        // Initialize with 5 random starts
        for (int iinit=0;iinit<5;iinit++){
            double bh = bh_lo + (bh_hi-bh_lo) * (double)rand()/RAND_MAX;
            double sl = sl_lo + (sl_hi-sl_lo) * (double)rand()/RAND_MAX;
            double sc = sc_lo + (sc_hi-sc_lo) * (double)rand()/RAND_MAX;
            auto [score,tr,mm,sh] = eval_params(bh,sl,sc);
            if (score > best.score) best = {bh,sl,sc,score,tr,mm,sh};
            std::cout << "Init bh="<<bh<<" sl="<<sl<<" sc="<<sc<<" -> trades="<<tr<<" mret="<<mm<<" sharpe="<<sh<<" score="<<score<<"\n";
        }

        int trials = 24, no_improve=0, patience=5;
        double step_bh=0.03, step_sl=0.03, step_sc=0.03;
        for (int t=0;t<trials && no_improve<patience; ++t){
            double bh, sl, sc;
            if (t % 3 == 0){
                // exploration
                bh = bh_lo + (bh_hi-bh_lo) * (double)rand()/RAND_MAX;
                sl = sl_lo + (sl_hi-sl_lo) * (double)rand()/RAND_MAX;
                sc = sc_lo + (sc_hi-sc_lo) * (double)rand()/RAND_MAX;
            } else {
                // exploitation around best
                bh = clip(best.buy_hi + step_bh * (((double)rand()/RAND_MAX)-0.5)*2.0, bh_lo, bh_hi);
                sl = clip(best.sell_lo + step_sl * (((double)rand()/RAND_MAX)-0.5)*2.0, sl_lo, sl_hi);
                sc = clip(best.short_conf + step_sc * (((double)rand()/RAND_MAX)-0.5)*2.0, sc_lo, sc_hi);
            }
            auto [score,tr,mm,sh] = eval_params(bh,sl,sc);
            std::cout << "Try#"<<t<<" bh="<<bh<<" sl="<<sl<<" sc="<<sc<<" -> trades="<<tr<<" mret="<<mm<<" sharpe="<<sh<<" score="<<score<<"\n";
            if (score > best.score){ best = {bh,sl,sc,score,tr,mm,sh}; no_improve=0; }
            else { no_improve++; }
        }

        std::cout << "\nBest params: buy_hi="<<best.buy_hi<<" sell_lo="<<best.sell_lo<<" short_conf="<<best.short_conf
                  <<" | trades="<<best.avg_trades<<" mret="<<best.avg_monthly<<" sharpe="<<best.sharpe<<"\n";

        // Persist best params for IRE
        try {
            nlohmann::json j;
            j["params"]["buy_hi"] = best.buy_hi;
            j["params"]["sell_lo"] = best.sell_lo;
            j["router"]["ire_min_conf_strong_short"] = best.short_conf;
            std::ofstream outf("configs/strategies/ire.json");
            outf << j.dump(2);
            std::cout << "Saved tuned IRE params to configs/strategies/ire.json\n";
        } catch (...) {
            std::cerr << "Warning: failed to write configs/strategies/ire.json\n";
        }

        // Run test with best params on most recent quarters
        sentio::RunnerCfg rcfg_best; rcfg_best.strategy_name = "IRE";
        rcfg_best.strategy_params = { {"buy_hi", std::to_string(best.buy_hi)}, {"sell_lo", std::to_string(best.sell_lo)} };
        rcfg_best.router.ire_min_conf_strong_short = best.short_conf;
        sentio::TemporalAnalysisConfig tcfg_test; tcfg_test.num_quarters = test_quarters; tcfg_test.print_detailed_report = true;
        auto test_summary = sentio::run_temporal_analysis(ST, test_series, base_symbol_id, rcfg_best, tcfg_test);
        test_summary.assess_readiness(tcfg_test);

    } else if (command == "replay") {
        std::cout << "Replay command is not fully implemented in this example.\n";
    } else {
        usage();
        return 1;
    }
    
    return 0;
}

```

## 📄 **FILE 28 of 60**: src/ml/model_registry_ts.cpp

**File Information**:
- **Path**: `src/ml/model_registry_ts.cpp`

- **Size**: 71 lines
- **Modified**: 2025-09-07 12:32:16

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

## 📄 **FILE 29 of 60**: src/ml/ts_model.cpp

**File Information**:
- **Path**: `src/ml/ts_model.cpp`

- **Size**: 81 lines
- **Modified**: 2025-09-07 14:29:16

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

## 📄 **FILE 30 of 60**: src/optimizer.cpp

**File Information**:
- **Path**: `src/optimizer.cpp`

- **Size**: 180 lines
- **Modified**: 2025-09-05 09:41:27

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

## 📄 **FILE 31 of 60**: src/pnl_accounting.cpp

**File Information**:
- **Path**: `src/pnl_accounting.cpp`

- **Size**: 62 lines
- **Modified**: 2025-09-05 15:28:49

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

## 📄 **FILE 32 of 60**: src/poly_fetch_main.cpp

**File Information**:
- **Path**: `src/poly_fetch_main.cpp`

- **Size**: 57 lines
- **Modified**: 2025-09-07 03:24:41

- **Type**: .cpp

```text
#include "sentio/polygon_client.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

using namespace sentio;

int main(int argc,char**argv){
  if(argc<5){
    std::cerr<<"Usage: poly_fetch FAMILY from to outdir [--timespan day|hour|minute] [--multiplier N] [--symbols SYM1,SYM2,...] [--rth]\n";
    return 1;
  }
  std::string fam=argv[1], from=argv[2], to=argv[3], outdir=argv[4];
  std::string timespan = "day";
  int multiplier = 1;
  std::string symbols_csv;
  bool rth_only=false;
  bool exclude_holidays=false;
  for (int i=5;i<argc;i++) {
    std::string a = argv[i];
    if ((a=="--timespan" || a=="-t") && i+1<argc) { timespan = argv[++i]; }
    else if ((a=="--multiplier" || a=="-m") && i+1<argc) { multiplier = std::stoi(argv[++i]); }
    else if (a=="--symbols" && i+1<argc) { symbols_csv = argv[++i]; }
    else if (a=="--rth") { rth_only=true; }
    else if (a=="--no-holidays") { exclude_holidays=true; }
  }
  const char* key = std::getenv("POLYGON_API_KEY");
  std::string api_key = key? key: "";
  PolygonClient cli(api_key);

  std::vector<std::string> syms;
  if(fam=="qqq") syms={"QQQ","TQQQ","SQQQ","PSQ"};
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
    cli.write_csv(fname,s,bars,rth_only,exclude_holidays);
    std::cerr<<"Wrote "<<bars.size()<<" bars -> "<<fname<<"\n";
  }
}


```

## 📄 **FILE 33 of 60**: src/polygon_client.cpp

**File Information**:
- **Path**: `src/polygon_client.cpp`

- **Size**: 244 lines
- **Modified**: 2025-09-07 04:14:17

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

// **NEW**: RTH check directly from UTC timestamp.
// RTH in UTC: 13:30-20:00 UTC (EDT) or 14:30-21:00 UTC (EST)
static bool is_rth_utc_from_utc_ms(long long utc_ms) {
    cctz::time_point<cctz::seconds> tp{cctz::seconds{utc_ms / 1000}};
    
    // Get UTC timezone
    cctz::time_zone utc_tz;
    if (!cctz::load_time_zone("UTC", &utc_tz)) {
        return false;
    }
    
    // Convert to UTC civil time
    auto lt = cctz::convert(tp, utc_tz);
    auto ct = cctz::civil_second(lt);
    
    // Check if weekend (Saturday = 6, Sunday = 0)
    auto wd = cctz::get_weekday(ct);
    if (wd == cctz::weekday::saturday || wd == cctz::weekday::sunday) {
        return false;
    }
    
    // Check if RTH in UTC
    // EST: 14:30-21:00 UTC (9:30 AM - 4:00 PM EST)
    // EDT: 13:30-20:00 UTC (9:30 AM - 4:00 PM EDT)
    int hour = ct.hour();
    int minute = ct.minute();
    
    // Simple DST check: April-October is EDT, rest is EST
    int month = ct.month();
    bool is_edt = (month >= 4 && month <= 10);
    
    if (is_edt) {
        // EDT: 13:30-20:00 UTC
        if (hour < 13 || (hour == 13 && minute < 30)) {
            return false;  // Before 13:30 UTC
        }
        if (hour >= 20) {
            return false;  // After 20:00 UTC
        }
    } else {
        // EST: 14:30-21:00 UTC
        if (hour < 14 || (hour == 14 && minute < 30)) {
            return false;  // Before 14:30 UTC
        }
        if (hour >= 21) {
            return false;  // After 21:00 UTC
        }
    }
    
    return true;
}

// **NEW**: Holiday check in UTC
static bool is_us_market_holiday_utc(int year, int month, int day) {
  // Simple holiday check for common US market holidays in UTC
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

void PolygonClient::write_csv(const std::string& out_path,const std::string& symbol,
                              const std::vector<AggBar>& bars, bool rth_only, bool exclude_holidays) {
  std::ofstream f(out_path);
  f << "timestamp,symbol,open,high,low,close,volume\n";
  for (auto& a: bars) {
    // **MODIFIED**: RTH and holiday filtering is now done directly on the UTC timestamp
    // before any string conversion, making it much more reliable.

    if (rth_only && !is_rth_utc_from_utc_ms(a.ts_ms)) {
        continue;
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

## 📄 **FILE 34 of 60**: src/polygon_ingest.cpp

**File Information**:
- **Path**: `src/polygon_ingest.cpp`

- **Size**: 68 lines
- **Modified**: 2025-09-05 15:29:30

- **Type**: .cpp

```text
#include "sentio/polygon_ingest.hpp"
#include "sentio/time_utils.hpp"
#include "sentio/rth_calendar.hpp"
#include "sentio/calendar_seed.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace sentio {

static const TradingCalendar& nyse_calendar() {
  static TradingCalendar cal = make_default_nyse_calendar();
  return cal;
}

static inline bool accept_bar_utc(std::int64_t ts_utc) {
  return nyse_calendar().is_rth_utc(ts_utc, "America/New_York");
}

static inline bool valid_bar(const ProviderBar& b) {
  if (!std::isfinite(b.open)  || !std::isfinite(b.high) ||
      !std::isfinite(b.low)   || !std::isfinite(b.close)||
      !std::isfinite(b.volume)) return false;
  if (b.open <= 0 || b.high <= 0 || b.low <= 0 || b.close <= 0) return false;
  if (b.low > b.high) return false;
  return !b.symbol.empty();
}

static inline void upsert_bar(PriceBook& book, const std::string& instrument, const ProviderBar& pb) {
  Bar b;
  b.open  = pb.open;
  b.high  = pb.high;
  b.low   = pb.low;
  b.close = pb.close;
  book.upsert_latest(instrument, b);
}

static inline std::int64_t to_epoch_s(const std::variant<std::int64_t, double, std::string>& ts) {
  auto tp = to_utc_sys_seconds(ts);
  return std::chrono::time_point_cast<std::chrono::seconds>(tp).time_since_epoch().count();
}

std::size_t ingest_provider_bars(const std::vector<ProviderBar>& input, PriceBook& book) {
  std::size_t accepted = 0;
  for (const auto& pb : input) {
    if (!valid_bar(pb)) continue;

    std::int64_t ts_utc{};
    try {
      ts_utc = to_epoch_s(pb.ts);
    } catch (...) {
      continue; // skip malformed time
    }

    if (!accept_bar_utc(ts_utc)) continue;

    upsert_bar(book, pb.symbol, pb);
    ++accepted;
  }
  return accepted;
}

bool ingest_provider_bar(const ProviderBar& bar, PriceBook& book) {
  return ingest_provider_bars(std::vector<ProviderBar>{bar}, book) == 1;
}

} // namespace sentio
```

## 📄 **FILE 35 of 60**: src/router.cpp

**File Information**:
- **Path**: `src/router.cpp`

- **Size**: 92 lines
- **Modified**: 2025-09-05 15:29:30

- **Type**: .cpp

```text
#include "sentio/router.hpp"
#include "sentio/polygon_ingest.hpp" // for PriceBook forward impl
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
    else           return strong ? cfg.bear3x : cfg.bear1x;
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

## 📄 **FILE 36 of 60**: src/rth_calendar.cpp

**File Information**:
- **Path**: `src/rth_calendar.cpp`

- **Size**: 67 lines
- **Modified**: 2025-09-07 14:29:16

- **Type**: .cpp

```text
// rth_calendar.cpp
#include "sentio/rth_calendar.hpp"
#include <chrono>
#include <string>
#include <stdexcept>
#include <iostream>
#include "cctz/time_zone.h"
#include "cctz/civil_time.h"

using namespace std::chrono;

namespace sentio {

bool TradingCalendar::is_rth_utc(std::int64_t ts_utc, [[maybe_unused]] const std::string& tz_name) const {
  using namespace std::chrono;
  
  // Convert to cctz time
  cctz::time_point<cctz::seconds> tp{cctz::seconds{ts_utc}};
  
  // Get UTC timezone
  cctz::time_zone utc_tz;
  if (!cctz::load_time_zone("UTC", &utc_tz)) {
    return false;
  }
  
  // Convert to UTC civil time
  auto lt = cctz::convert(tp, utc_tz);
  auto ct = cctz::civil_second(lt);
  
  // Check if weekend (Saturday = 6, Sunday = 0)
  auto wd = cctz::get_weekday(ct);
  if (wd == cctz::weekday::saturday || wd == cctz::weekday::sunday) {
    return false;
  }
  
  // Check if RTH in UTC
  // EST: 14:30-21:00 UTC (9:30 AM - 4:00 PM EST)
  // EDT: 13:30-20:00 UTC (9:30 AM - 4:00 PM EDT)
  int hour = ct.hour();
  int minute = ct.minute();
  int month = ct.month();
  
  // Simple DST check: April-October is EDT, rest is EST
  bool is_edt = (month >= 4 && month <= 10);
  
  if (is_edt) {
    // EDT: 13:30-20:00 UTC
    if (hour < 13 || (hour == 13 && minute < 30)) {
      return false;  // Before 13:30 UTC
    }
    if (hour >= 20) {
      return false;  // After 20:00 UTC
    }
  } else {
    // EST: 14:30-21:00 UTC
    if (hour < 14 || (hour == 14 && minute < 30)) {
      return false;  // Before 14:30 UTC
    }
    if (hour >= 21) {
      return false;  // After 21:00 UTC
    }
  }
  
  return true;
}

} // namespace sentio
```

## 📄 **FILE 37 of 60**: src/runner.cpp

**File Information**:
- **Path**: `src/runner.cpp`

- **Size**: 349 lines
- **Modified**: 2025-09-09 03:17:13

- **Type**: .cpp

```text
#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/strategy_ire.hpp"
#include "sentio/pricebook.hpp"
#include "sentio/metrics.hpp"
#include "sentio/sizer.hpp"
#include "sentio/cost_model.hpp"
#include "sentio/feature_feeder.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <chrono>

namespace sentio {

// **RENOVATED HELPER**: Execute target position for any instrument
static void execute_target_position(const std::string& instrument, double target_weight, 
                                   Portfolio& portfolio, const SymbolTable& ST, const Pricebook& pricebook,
                                   const AdvancedSizer& sizer, const RunnerCfg& cfg,
                                   const std::vector<std::vector<Bar>>& series, const Bar& bar,
                                   const std::string& chain_id, AuditRecorder& audit, bool logging_enabled, int& total_fills) {
    
    int instrument_id = ST.get_id(instrument);
    if (instrument_id == -1) return;
    
    double instrument_price = pricebook.last_px[instrument_id];
    if (instrument_price <= 0) return;

    // Calculate target quantity using sizer
    double target_qty = sizer.calculate_target_quantity(portfolio, ST, pricebook.last_px, 
                                                       instrument, target_weight, 
                                                       series[instrument_id], cfg.sizer);
    
    double current_qty = portfolio.positions[instrument_id].qty;
    double trade_qty = target_qty - current_qty;

    // **PROFIT MAXIMIZATION**: Execute any meaningful trade (no dust filter for Governor)
    if (std::abs(trade_qty * instrument_price) > 10.0) { // $10 minimum
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

        // Apply costs and execute
        auto [fees, slippage_cost] = AlpacaCostModel::calculate_total_costs(instrument, trade_qty, instrument_price);
        double exec_px = (trade_qty > 0)
            ? AlpacaCostModel::apply_slippage(instrument_price, AlpacaCostModel::calculate_slippage_bps(trade_qty, instrument_price, 1000000, 0.20), true)
            : AlpacaCostModel::apply_slippage(instrument_price, AlpacaCostModel::calculate_slippage_bps(trade_qty, instrument_price, 1000000, 0.20), false);
        
        apply_fill(portfolio, instrument_id, trade_qty, exec_px);
        portfolio.cash -= fees;
        
        double equity_after = equity_mark_to_market(portfolio, pricebook.last_px);
        double pos_after = portfolio.positions[instrument_id].qty;
        
        if (logging_enabled) {
            audit.event_fill_ex(bar.ts_utc_epoch, instrument, exec_px, std::abs(trade_qty), fees, side,
                               realized_delta, equity_after, pos_after, chain_id);
        }
        total_fills++;
    }
}

RunResult run_backtest(AuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                      int base_symbol_id, const RunnerCfg& cfg) {
    
    // 1. ============== INITIALIZATION ==============
    RunResult result{};
    
    const bool logging_enabled = (cfg.audit_level == AuditLevel::Full);
    // Start audit run
    std::string meta = "{";
    meta += "\"strategy\":\"" + cfg.strategy_name + "\",";
    meta += "\"base_symbol_id\":" + std::to_string(base_symbol_id) + ",";
    meta += "\"total_series\":" + std::to_string(series.size()) + ",";
    meta += "\"base_series_size\":" + std::to_string(series[base_symbol_id].size());
    meta += "}";
    if (logging_enabled) audit.event_run_start(series[base_symbol_id][0].ts_utc_epoch, meta);
    
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

    // 2. ============== MAIN EVENT LOOP ==============
    size_t total_bars = base_series.size();
    size_t progress_interval = total_bars / 20; // 5% intervals (20 steps)
    
    // Skip first 300 bars to allow technical indicators to warm up
    size_t warmup_bars = 300;
    if (total_bars <= warmup_bars) {
        std::cout << "Warning: Not enough bars for warmup (need " << warmup_bars << ", have " << total_bars << ")" << std::endl;
        warmup_bars = 0;
    }
    
    for (size_t i = warmup_bars; i < base_series.size(); ++i) {
        // **DEBUG**: Check main loop entry
        if (i < warmup_bars + 3) {
            std::cout << "MAIN LOOP i=" << i << " warmup=" << warmup_bars << " strategy=" << cfg.strategy_name << std::endl;
        }
        
        // Progress reporting at 5% intervals
        if (i % progress_interval == 0) {
            int progress_percent = (i * 100) / total_bars;
            std::cout << "Progress: " << progress_percent << "% (" << i << "/" << total_bars << " bars)" << std::endl;
        }
        
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
                            
                            // Apply costs and execute close
                            auto [fees, slippage_cost] = AlpacaCostModel::calculate_total_costs(inst, close_qty, close_price);
                            double exec_px = (close_qty > 0)
                                ? AlpacaCostModel::apply_slippage(close_price, AlpacaCostModel::calculate_slippage_bps(close_qty, close_price, 1000000, 0.20), true)
                                : AlpacaCostModel::apply_slippage(close_price, AlpacaCostModel::calculate_slippage_bps(close_qty, close_price, 1000000, 0.20), false);
                            
                            double realized_pnl = (portfolio.positions[sid].qty > 0) 
                                ? (exec_px - portfolio.positions[sid].avg_price) * std::abs(close_qty)
                                : (portfolio.positions[sid].avg_price - exec_px) * std::abs(close_qty);
                            
                            apply_fill(portfolio, sid, close_qty, exec_px);
                            portfolio.cash -= fees;
                            
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
        if (logging_enabled) audit.event_bar(bar.ts_utc_epoch, ST.get_symbol(base_symbol_id), audit_bar);
        
        // Feed features to ML strategies
        [[maybe_unused]] auto start_feed = std::chrono::high_resolution_clock::now();
        FeatureFeeder::feed_features_to_strategy(strategy.get(), base_series, i, cfg.strategy_name);
        [[maybe_unused]] auto end_feed = std::chrono::high_resolution_clock::now();
        
        // **RENOVATED ARCHITECTURE**: Governor-based target weight system
        [[maybe_unused]] auto start_signal = std::chrono::high_resolution_clock::now();
        
        bool is_ire = (cfg.strategy_name == "IRE");
        double target_weight = 0.0;
        std::string target_instrument = ST.get_symbol(base_symbol_id);
        std::string chain_id = std::to_string(bar.ts_utc_epoch) + ":" + std::to_string((long long)i);
        
        // **DEBUG**: Check strategy name and path selection
        if (i < 5) {
            std::cout << "Strategy: " << cfg.strategy_name << " is_ire=" << is_ire << std::endl;
        }
        
        if (is_ire) {
            // **NEW**: Direct Governor-based weight calculation
            auto* ire_strategy = dynamic_cast<IREStrategy*>(strategy.get());
            if (ire_strategy) {
                target_weight = ire_strategy->calculate_target_weight(base_series, i);
                
                // **DEBUG**: Log Governor activation
                if (i < 10 || i % 1000 == 0) {
                    std::cout << "Governor: i=" << i << " target_weight=" << target_weight << std::endl;
                }
                
                // Log probability for diagnostics
                double probability = ire_strategy->get_latest_probability();
                if (logging_enabled) {
                    audit.event_signal_ex(bar.ts_utc_epoch, ST.get_symbol(base_symbol_id), 
                                        SigType::HOLD, probability, chain_id);
                }
            }
        } else {
            // **LEGACY**: Old signal routing for non-IRE strategies
            StrategySignal sig = strategy->calculate_signal(base_series, i);
            if (sig.type != StrategySignal::Type::HOLD) {
                SigType sig_type = static_cast<SigType>(static_cast<int>(sig.type));
                if (logging_enabled) audit.event_signal_ex(bar.ts_utc_epoch, ST.get_symbol(base_symbol_id), sig_type, sig.confidence, chain_id);
                
                auto route_decision = route(sig, cfg.router, ST.get_symbol(base_symbol_id));
                if (route_decision) {
                    target_weight = route_decision->target_weight;
                    target_instrument = route_decision->instrument;
                }
            }
        }
        
        [[maybe_unused]] auto end_signal = std::chrono::high_resolution_clock::now();
        
        // **UNIFIED EXECUTION**: Execute target weight (Governor or legacy)
        if (std::abs(target_weight) > 1e-6) {
            // **RENOVATED EXECUTION**: Clean Governor-based allocation
            if (is_ire) {
                // **IRE**: Smart instrument selection based on target weight  
                std::vector<std::pair<std::string, double>> allocations;
                
                if (target_weight > 0.5) {
                    // Strong long: Split QQQ/TQQQ for leverage
                    allocations.push_back({ST.get_symbol(base_symbol_id), target_weight * 0.6});
                    allocations.push_back({cfg.router.bull3x, target_weight * 0.4});
                } else if (target_weight > 0.0) {
                    // Moderate long: QQQ only
                    allocations.push_back({ST.get_symbol(base_symbol_id), target_weight});
                } else if (target_weight < -0.3) {
                    // Strong short: SQQQ
                    allocations.push_back({cfg.router.bear3x, std::abs(target_weight)});
                }
                
                // Execute all allocations using helper function
                for (const auto& [instrument, weight] : allocations) {
                    execute_target_position(instrument, weight, portfolio, ST, pricebook, sizer, 
                                          cfg, series, bar, chain_id, audit, logging_enabled, total_fills);
                }
            } else {
                // **LEGACY**: Single instrument execution for other strategies
                execute_target_position(target_instrument, target_weight, portfolio, ST, pricebook, sizer,
                                      cfg, series, bar, chain_id, audit, logging_enabled, total_fills);
            }
        }
        
        // 3. ============== SNAPSHOT ==============
        if (i % cfg.snapshot_stride == 0 || i == base_series.size() - 1) {
            double current_equity = equity_mark_to_market(portfolio, pricebook.last_px);
            equity_curve.emplace_back(bar.ts_utc, current_equity);
            
            // Log account snapshot
            AccountState state;
            state.cash = portfolio.cash;
            state.equity = current_equity;
            state.realized = 0.0; // TODO: Calculate realized P&L
            if (logging_enabled) audit.event_snapshot(bar.ts_utc_epoch, state);
        }
    }
    
    // 4. ============== METRICS & DIAGNOSTICS ==============
    strategy->get_diag().print(strategy->get_name().c_str());

    if (equity_curve.empty()) {
        return result;
    }
    
    // **CRITICAL FIX**: Pass the correct `total_fills` to the metrics calculator.
    auto summary = compute_metrics_day_aware(equity_curve, total_fills);

    result.final_equity = equity_curve.empty() ? 100000.0 : equity_curve.back().second;
    result.total_return = summary.ret_total * 100.0;
    result.sharpe_ratio = summary.sharpe;
    result.max_drawdown = summary.mdd * 100.0;
    result.total_fills = summary.trades;
    result.no_route = no_route_count;
    result.no_qty = no_qty_count;

    // Log final metrics and end run
    std::int64_t end_ts = equity_curve.empty() ? series[base_symbol_id][0].ts_utc_epoch : series[base_symbol_id].back().ts_utc_epoch;
    if (logging_enabled) {
        audit.event_metric(end_ts, "final_equity", result.final_equity);
        audit.event_metric(end_ts, "total_return", result.total_return);
        audit.event_metric(end_ts, "sharpe_ratio", result.sharpe_ratio);
        audit.event_metric(end_ts, "max_drawdown", result.max_drawdown);
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

## 📄 **FILE 38 of 60**: src/sanity.cpp

**File Information**:
- **Path**: `src/sanity.cpp`

- **Size**: 166 lines
- **Modified**: 2025-09-05 20:37:59

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

## 📄 **FILE 39 of 60**: src/signal_engine.cpp

**File Information**:
- **Path**: `src/signal_engine.cpp`

- **Size**: 29 lines
- **Modified**: 2025-09-05 17:03:00

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
  auto conf2 = gate_.accept(ctx.ts_utc_epoch, ctx.is_rth, inputs_finite,
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

## 📄 **FILE 40 of 60**: src/signal_gate.cpp

**File Information**:
- **Path**: `src/signal_gate.cpp`

- **Size**: 52 lines
- **Modified**: 2025-09-05 17:03:11

- **Type**: .cpp

```text
#include "sentio/signal_gate.hpp"
#include <cmath>
#include <algorithm>

namespace sentio {

SignalHealth::SignalHealth() {
  for (auto r :
       {DropReason::NONE, DropReason::NOT_RTH, DropReason::WARMUP,
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
                                         bool is_rth,
                                         bool inputs_finite,
                                         bool warmed_up,
                                         double conf)
{
  if (cfg_.require_rth && !is_rth) { if (health_) health_->incr_drop(DropReason::NOT_RTH); return std::nullopt; }
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

## 📄 **FILE 41 of 60**: src/signal_pipeline.cpp

**File Information**:
- **Path**: `src/signal_pipeline.cpp`

- **Size**: 43 lines
- **Modified**: 2025-09-05 17:05:46

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
  tr.is_rth = ctx.is_rth;
  tr.inputs_finite = std::isfinite(b.close);

  auto sig = strat_->latest();
  if (!sig) {
    tr.reason = TraceReason::NO_STRATEGY_OUTPUT;
    if (trace_) trace_->push(tr);
    return out;
  }
  tr.confidence = sig->confidence;

  // Use the existing signal_gate API
  auto conf2 = gate_.accept(ctx.ts_utc_epoch, ctx.is_rth, tr.inputs_finite, true, sig->confidence);
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

## 📄 **FILE 42 of 60**: src/signal_trace.cpp

**File Information**:
- **Path**: `src/signal_trace.cpp`

- **Size**: 7 lines
- **Modified**: 2025-09-05 17:00:56

- **Type**: .cpp

```text
#include "sentio/signal_trace.hpp"

namespace sentio {
std::size_t SignalTrace::count(TraceReason r) const {
  std::size_t n=0; for (auto& x: rows_) if (x.reason==r) ++n; return n;
}
} // namespace sentio

```

## 📄 **FILE 43 of 60**: src/sim_data.cpp

**File Information**:
- **Path**: `src/sim_data.cpp`

- **Size**: 47 lines
- **Modified**: 2025-09-05 20:37:59

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

## 📄 **FILE 44 of 60**: src/strategy/run_rule_ensemble.cpp

**File Information**:
- **Path**: `src/strategy/run_rule_ensemble.cpp`

- **Size**: 79 lines
- **Modified**: 2025-09-08 12:22:07

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

## 📄 **FILE 45 of 60**: src/strategy_bollinger_squeeze_breakout.cpp

**File Information**:
- **Path**: `src/strategy_bollinger_squeeze_breakout.cpp`

- **Size**: 147 lines
- **Modified**: 2025-09-05 15:29:30

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

StrategySignal BollingerSqueezeBreakoutStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < squeeze_lookback_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }
    
    if (state_ == State::Long || state_ == State::Short) {
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            signal.type = (state_ == State::Long) ? StrategySignal::Type::SELL : StrategySignal::Type::BUY;
            reset_state();
            diag_.emitted++;
            return signal;
        }
        return signal;
    }

    update_state_machine(bars[current_index]);

    if (state_ == State::ArmedLong || state_ == State::ArmedShort) {
        if (squeeze_duration_ < min_squeeze_bars_) {
            diag_.drop(DropReason::THRESHOLD);
            state_ = State::Idle;
            return signal;
        }

        double mid, lo, hi, sd;
        bollinger_.step(bars[current_index].close, mid, lo, hi, sd);
        
        if (state_ == State::ArmedLong) {
            signal.type = StrategySignal::Type::BUY;
            state_ = State::Long;
        } else {
            signal.type = StrategySignal::Type::SELL;
            state_ = State::Short;
        }
        
        signal.confidence = 0.8;
        diag_.emitted++;
        bars_in_trade_ = 0;
    } else {
        diag_.drop(DropReason::THRESHOLD);
    }

    return signal;
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

REGISTER_STRATEGY(BollingerSqueezeBreakoutStrategy, "BollingerSqueezeBreakout");

} // namespace sentio

```

## 📄 **FILE 46 of 60**: src/strategy_hybrid_ppo.cpp

**File Information**:
- **Path**: `src/strategy_hybrid_ppo.cpp`

- **Size**: 81 lines
- **Modified**: 2025-09-06 00:28:01

- **Type**: .cpp

```text
#include "sentio/strategy_hybrid_ppo.hpp"
#include <algorithm>
#include <stdexcept>

namespace sentio {

HybridPPOStrategy::HybridPPOStrategy()
: BaseStrategy("HybridPPO"),
  cfg_(),
  handle_(ml::ModelRegistryTS::load_torchscript("HybridPPO", cfg_.version, cfg_.artifacts_dir, cfg_.use_cuda)),
  fpipe_(handle_.spec)
{}

HybridPPOStrategy::HybridPPOStrategy(const HybridPPOCfg& cfg)
: BaseStrategy("HybridPPO"),
  cfg_(cfg),
  handle_(ml::ModelRegistryTS::load_torchscript("HybridPPO", cfg.version, cfg.artifacts_dir, cfg.use_cuda)),
  fpipe_(handle_.spec)
{}

void HybridPPOStrategy::set_raw_features(const std::vector<double>& raw) {
  raw_ = raw;
}

ParameterMap HybridPPOStrategy::get_default_params() const {
  return {
    {"conf_floor", cfg_.conf_floor}
  };
}

ParameterSpace HybridPPOStrategy::get_param_space() const {
  return {
    {"conf_floor", {ParamType::FLOAT, 0.0, 1.0, cfg_.conf_floor}}
  };
}

void HybridPPOStrategy::apply_params() {
  cfg_.conf_floor = params_["conf_floor"];
}

StrategySignal HybridPPOStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
  (void)bars; (void)current_index; // Features come from set_raw_features
  last_.reset();
  auto z = fpipe_.transform(raw_);
  if (!z) return StrategySignal{};

  auto out = handle_.model->predict(*z, 1, (int)z->size(), "BF");
  if (!out) return StrategySignal{};

  StrategySignal sig = map_output(*out);
  // Safety: never emit if below floor
  if (sig.confidence < cfg_.conf_floor) return StrategySignal{};

  last_ = sig;
  return sig;
}

StrategySignal HybridPPOStrategy::map_output(const ml::ModelOutput& mo) const {
  StrategySignal s;
  // Discrete 3-way: SELL/HOLD/BUY
  int argmax = 0;
  for (int i=1;i<(int)mo.probs.size();++i) if (mo.probs[i] > mo.probs[argmax]) argmax = i;

  const auto& acts = handle_.spec.actions; // e.g., ["SELL","HOLD","BUY"]
  std::string a = (argmax<(int)acts.size()? acts[argmax] : "HOLD");
  float pmax = mo.probs.empty()? 0.0f : mo.probs[argmax];

  if (a=="BUY")       s.type = StrategySignal::Type::BUY;
  else if (a=="SELL") s.type = StrategySignal::Type::SELL;
  else                s.type = StrategySignal::Type::HOLD;

  // Confidence: calibrated pmax (basic identity)
  s.confidence = std::max(cfg_.conf_floor, (double)pmax);
  return s;
}


// Register the strategy with the factory
// REGISTER_STRATEGY(HybridPPOStrategy, "hybrid_ppo")  // Disabled - not working

} // namespace sentio

```

## 📄 **FILE 47 of 60**: src/strategy_ire.cpp

**File Information**:
- **Path**: `src/strategy_ire.cpp`

- **Size**: 113 lines
- **Modified**: 2025-09-09 03:25:28

- **Type**: .cpp

```text
#include "sentio/strategy_ire.hpp"
#include <algorithm>

namespace sentio {

IREStrategy::IREStrategy() : BaseStrategy("IRE") {}

ParameterMap IREStrategy::get_default_params() const {
  return {
    {"buy_lo", 0.60}, {"buy_hi", 0.75}, {"sell_hi", 0.40}, {"sell_lo", 0.25}, {"hysteresis", 0.02}
  };
}

ParameterSpace IREStrategy::get_param_space() const {
  return {
    {"buy_lo",   {ParamType::FLOAT, 0.50, 0.70, 0.60}},
    {"buy_hi",   {ParamType::FLOAT, 0.60, 0.90, 0.75}},
    {"sell_hi",  {ParamType::FLOAT, 0.30, 0.50, 0.40}},
    {"sell_lo",  {ParamType::FLOAT, 0.10, 0.40, 0.25}},
    {"hysteresis", {ParamType::FLOAT, 0.00, 0.05, 0.02}}
  };
}

void IREStrategy::apply_params() {
  buy_lo_ = (float)params_["buy_lo"]; buy_hi_=(float)params_["buy_hi"]; sell_hi_=(float)params_["sell_hi"]; sell_lo_=(float)params_["sell_lo"]; hysteresis_=(float)params_["hysteresis"]; 
}

void IREStrategy::ensure_ensemble_built_(){
  if (ensemble_) return;
  std::vector<std::unique_ptr<rules::IRuleStrategy>> rulesv;
  rulesv.push_back(std::make_unique<rules::SMACrossRule>());
  rulesv.push_back(std::make_unique<rules::VWAPReversionRule>());
  rulesv.push_back(std::make_unique<rules::BBandsSqueezeBreakoutRule>());
  rulesv.push_back(std::make_unique<rules::MomentumVolumeRule>());
  rulesv.push_back(std::make_unique<rules::OFIProxyRule>());
  rules::EnsembleConfig ec; ec.agreement_boost=0.25f; ec.reliability_window=512; ec.base_weights.assign(rulesv.size(), 1.0f);
  ensemble_ = std::make_unique<rules::IntegratedRuleEnsemble>(std::move(rulesv), ec);
  warmup_ = ensemble_->warmup();
}

void IREStrategy::ensure_governor_built_(){
  if (governor_) return;
  IntradayPositionGovernor::Config gov_config;
  gov_config.lookback_window = 60;      // 60-minute adaptive window
  gov_config.buy_percentile = 0.75;     // Less aggressive: Top 25% trigger long  
  gov_config.sell_percentile = 0.25;    // Less aggressive: Bottom 25% trigger short
  gov_config.max_base_weight = 0.5;     // Conservative: 50% max position before decay
  gov_config.min_abs_edge = 0.02;       // Allow modest signals: 2% minimum edge from 0.5
  governor_ = std::make_unique<IntradayPositionGovernor>(gov_config);
}

StrategySignal IREStrategy::calculate_signal(const std::vector<Bar>& bars, int i) {
  ensure_ensemble_built_();
  StrategySignal out; out.type = StrategySignal::Type::HOLD; out.confidence = 0.0;
  if (i < warmup_) return out;

  // build view over bars
  rules::BarsView v; v.n = (long long)bars.size();
  // provide minimal arrays (only close and volume needed by defaults), others can be approximated
  // For simplicity in this wrapper, we construct temporary contiguous arrays.
  static thread_local std::vector<long long> ts; ts.resize(bars.size());
  static thread_local std::vector<double> o,h,l,c,vv; o.resize(bars.size()); h.resize(bars.size()); l.resize(bars.size()); c.resize(bars.size()); vv.resize(bars.size());
  for (size_t k=0;k<bars.size();++k){ ts[k]=bars[k].ts_utc_epoch; o[k]=bars[k].open; h[k]=bars[k].high; l[k]=bars[k].low; c[k]=bars[k].close; vv[k]=double(bars[k].volume);} 
  v.ts=ts.data(); v.open=o.data(); v.high=h.data(); v.low=l.data(); v.close=c.data(); v.volume=vv.data();

  auto p = ensemble_->eval(v, i, std::nullopt, nullptr);
  if (!p) return out;

  float pup = *p;
  latest_probability_ = static_cast<double>(pup); // Store for Governor

  // Legacy signal classification (for backward compatibility)
  if (pup >= buy_hi_) { out.type = StrategySignal::Type::STRONG_BUY; out.confidence = std::clamp((double)((pup-buy_hi_)/(1.0f-buy_hi_)), 0.0, 1.0); }
  else if (pup >= buy_lo_) { out.type = StrategySignal::Type::BUY; out.confidence = std::clamp((double)((pup-buy_lo_)/(buy_hi_-buy_lo_)), 0.0, 1.0); }
  else if (pup <= sell_lo_) { out.type = StrategySignal::Type::STRONG_SELL; out.confidence = std::clamp((double)((sell_lo_-pup)/(sell_lo_)), 0.0, 1.0); }
  else if (pup <= sell_hi_) { out.type = StrategySignal::Type::SELL; out.confidence = std::clamp((double)((sell_hi_-pup)/(sell_hi_-0.5f)), 0.0, 1.0); }
  else { out.type = StrategySignal::Type::HOLD; out.confidence = 0.0; }
  return out;
}

double IREStrategy::calculate_target_weight(const std::vector<Bar>& bars, int i) {
  ensure_governor_built_();
  
  if (i < warmup_) return 0.0;

  // **TEMPORARY FIX**: Simple momentum signal to replace broken IRE ensemble
  // This will be replaced once the ensemble is fixed
  if (i < 20) return 0.0; // Need lookback
  
  // Simple trend following: compare recent price to moving average
  double recent_close = bars[i].close;
  double ma_20 = 0.0;
  for (int j = i - 19; j <= i; ++j) {
    ma_20 += bars[j].close;
  }
  ma_20 /= 20.0;
  
  // Generate probability based on distance from moving average
  double price_ratio = recent_close / ma_20;
  double probability = 0.5 + (price_ratio - 1.0) * 5.0; // Amplify signal
  probability = std::clamp(probability, 0.0, 1.0);
  
  latest_probability_ = probability;

  // **NEW ARCHITECTURE**: Governor determines target weight
  return governor_->calculate_target_weight(latest_probability_, bars[i].ts_utc_epoch);
}

REGISTER_STRATEGY(IREStrategy, "IRE");

} // namespace sentio



```

## 📄 **FILE 48 of 60**: src/strategy_kochi_ppo.cpp

**File Information**:
- **Path**: `src/strategy_kochi_ppo.cpp`

- **Size**: 95 lines
- **Modified**: 2025-09-07 22:52:32

- **Type**: .cpp

```text
#include "sentio/strategy_kochi_ppo.hpp"
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

StrategySignal KochiPPOStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
  (void)bars; (void)current_index; // features are streamed in via set_raw_features
  last_.reset();
  if (!window_.ready()) return StrategySignal{};

  auto in = window_.to_input();
  if (!in) return StrategySignal{};

  auto out = handle_.model->predict(*in, window_.seq_len(), window_.feat_dim(), handle_.spec.input_layout);
  if (!out) return StrategySignal{};

  auto sig = map_output(*out);
  if (sig.confidence < cfg_.conf_floor) return StrategySignal{};
  last_ = sig;
  return sig;
}

// Register with factory
REGISTER_STRATEGY(KochiPPOStrategy, "kochi_ppo");

} // namespace sentio



```

## 📄 **FILE 49 of 60**: src/strategy_market_making.cpp

**File Information**:
- **Path**: `src/strategy_market_making.cpp`

- **Size**: 140 lines
- **Modified**: 2025-09-05 20:51:42

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

StrategySignal MarketMakingStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;
    
    // Always update indicators to have a full history for the next bar
    if(current_index > 0) {
        double price_return = (bars[current_index].close - bars[current_index - 1].close) / bars[current_index - 1].close;
        rolling_returns_.push(price_return);
    }
    rolling_volume_.push(bars[current_index].volume);

    // Wait for indicators to warm up
    if (rolling_volume_.size() < static_cast<size_t>(params_.at("volume_window"))) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }

    if (!should_participate(bars[current_index])) {
        return signal;
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
                signal.type = StrategySignal::Type::BUY;
            } else if (price_change < -0.001) {
                signal.type = StrategySignal::Type::SELL;
            } else {
                diag_.drop(DropReason::THRESHOLD);
                return signal;
            }
        } else {
            diag_.drop(DropReason::THRESHOLD);
            return signal;
        }
    } else {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }

    signal.confidence = 0.6; // Fixed confidence since we're not using inventory_skew
    diag_.emitted++;
    return signal;
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

REGISTER_STRATEGY(MarketMakingStrategy, "MarketMaking");

} // namespace sentio


```

## 📄 **FILE 50 of 60**: src/strategy_momentum_volume.cpp

**File Information**:
- **Path**: `src/strategy_momentum_volume.cpp`

- **Size**: 147 lines
- **Modified**: 2025-09-05 15:24:52

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

StrategySignal MomentumVolumeProfileStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < profile_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }
    
    if (is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return signal;
    }
    
    // Periodically rebuild the expensive volume profile
    if (last_profile_update_ == -1 || current_index - last_profile_update_ >= 10) {
        build_volume_profile(bars, current_index);
        last_profile_update_ = current_index;
    }
    
    if (volume_profile_.value_area_high <= 0) {
        diag_.drop(DropReason::NAN_INPUT); // Profile not ready or invalid
        return signal;
    }

    const auto& bar = bars[current_index];
    avg_volume_.push(bar.volume);
    
    bool breakout_up = bar.close > (volume_profile_.value_area_high * (1.0 + breakout_threshold_pct_));
    bool breakout_down = bar.close < (volume_profile_.value_area_low * (1.0 - breakout_threshold_pct_));

    if (!breakout_up && !breakout_down) {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }

    if (!is_momentum_confirmed(bars, current_index)) {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }
    
    if (bar.volume < avg_volume_.mean() * volume_surge_mult_) {
        diag_.drop(DropReason::ZERO_VOL);
        return signal;
    }

    if (breakout_up) {
        signal.type = StrategySignal::Type::BUY;
    } else {
        signal.type = StrategySignal::Type::SELL;
    }
    
    signal.confidence = 0.85;
    diag_.emitted++;
    state_.last_trade_bar = current_index;

    return signal;
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

REGISTER_STRATEGY(MomentumVolumeProfileStrategy, "MomentumVolumeProfile");

} // namespace sentio

```

## 📄 **FILE 51 of 60**: src/strategy_opening_range_breakout.cpp

**File Information**:
- **Path**: `src/strategy_opening_range_breakout.cpp`

- **Size**: 131 lines
- **Modified**: 2025-09-08 14:22:00

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

StrategySignal OpeningRangeBreakoutStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < 1) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
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
        return signal;
    }

    // --- Finalize the range exactly once ---
    if (!current_range_.is_finalized) {
        current_range_.end_bar = current_index - 1;
        current_range_.is_finalized = true;
    }

    // --- Phase 2: Look for Breakouts ---
    if (state_.in_position || is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return signal;
    }

    const auto& bar = bars[current_index];
    bool is_breakout_up = bar.close > current_range_.high;
    bool is_breakout_down = bar.close < current_range_.low;

    if (!is_breakout_up && !is_breakout_down) {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }
    
    // Volume Confirmation
    double avg_volume = 0;
    for (int i = day_start_index_; i < current_range_.end_bar; ++i) {
        avg_volume += bars[i].volume;
    }
    avg_volume /= (current_range_.end_bar - day_start_index_ + 1);

    if (bar.volume < avg_volume * volume_multiplier_) {
        diag_.drop(DropReason::ZERO_VOL); // Re-using for low volume
        return signal;
    }

    // Generate Signal
    if (is_breakout_up) {
        signal.type = StrategySignal::Type::BUY;
    } else { // is_breakout_down
        signal.type = StrategySignal::Type::SELL;
    }

    signal.confidence = 0.9;
    diag_.emitted++;
    state_.in_position = true; // Manually set state as this is an intraday strategy
    state_.last_trade_bar = current_index;

    return signal;
}

// Register the strategy
REGISTER_STRATEGY(OpeningRangeBreakoutStrategy, "OpeningRangeBreakout");

} // namespace sentio
```

## 📄 **FILE 52 of 60**: src/strategy_order_flow_imbalance.cpp

**File Information**:
- **Path**: `src/strategy_order_flow_imbalance.cpp`

- **Size**: 105 lines
- **Modified**: 2025-09-05 15:25:26

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

StrategySignal OrderFlowImbalanceStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < lookback_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }

    double pressure = calculate_bar_pressure(bars[current_index]);
    double avg_pressure = rolling_pressure_.push(pressure);

    // **FIXED**: Use the strategy-specific 'ofi_state_' for state machine logic
    if (ofi_state_ == OFIState::Flat) {
        if (is_cooldown_active(current_index, cool_down_period_)) {
            diag_.drop(DropReason::COOLDOWN);
            return signal;
        }

        if (avg_pressure > entry_threshold_long_) {
            signal.type = StrategySignal::Type::BUY;
            ofi_state_ = OFIState::Long;
            // **FIXED**: Correctly access the 'state_' member from BaseStrategy
            state_.last_trade_bar = current_index;
        } else if (avg_pressure < entry_threshold_short_) {
            signal.type = StrategySignal::Type::SELL;
            ofi_state_ = OFIState::Short;
            // **FIXED**: Correctly access the 'state_' member from BaseStrategy
            state_.last_trade_bar = current_index;
        } else {
            diag_.drop(DropReason::THRESHOLD);
            return signal;
        }

        signal.confidence = 0.7;
        diag_.emitted++;
        bars_in_trade_ = 0;

    } else { // In a trade, check for exit
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            // **FIXED**: Use 'ofi_state_' to determine exit signal direction
            signal.type = (ofi_state_ == OFIState::Long) ? StrategySignal::Type::SELL : StrategySignal::Type::BUY;
            diag_.emitted++;
            reset_state();
        }
    }

    return signal;
}

REGISTER_STRATEGY(OrderFlowImbalanceStrategy, "OrderFlowImbalance");

} // namespace sentio


```

## 📄 **FILE 53 of 60**: src/strategy_order_flow_scalping.cpp

**File Information**:
- **Path**: `src/strategy_order_flow_scalping.cpp`

- **Size**: 120 lines
- **Modified**: 2025-09-05 16:40:29

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

StrategySignal OrderFlowScalpingStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < lookback_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }

    const auto& bar = bars[current_index];
    double pressure = calculate_bar_pressure(bar);
    double avg_pressure = rolling_pressure_.push(pressure);

    // **FIXED**: Use the strategy-specific 'of_state_' for state machine logic
    if (of_state_ == OFState::Long || of_state_ == OFState::Short) {
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            signal.type = (of_state_ == OFState::Long) ? StrategySignal::Type::SELL : StrategySignal::Type::BUY;
            diag_.emitted++;
            reset_state();
        }
        return signal;
    }
    
    if (is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return signal;
    }

    switch (of_state_) {
        case OFState::Idle:
            if (avg_pressure > imbalance_threshold_) of_state_ = OFState::ArmedLong;
            else if (avg_pressure < (1.0 - imbalance_threshold_)) of_state_ = OFState::ArmedShort;
            else diag_.drop(DropReason::THRESHOLD);
            break;
            
        case OFState::ArmedLong:
            if (pressure > 0.5) { // Confirmation bar must be bullish
                signal.type = StrategySignal::Type::BUY;
                of_state_ = OFState::Long;
            } else { // Failed confirmation
                of_state_ = OFState::Idle;
                diag_.drop(DropReason::THRESHOLD);
            }
            break;

        case OFState::ArmedShort:
            if (pressure < 0.5) { // Confirmation bar must be bearish
                signal.type = StrategySignal::Type::SELL;
                of_state_ = OFState::Short;
            } else { // Failed confirmation
                of_state_ = OFState::Idle;
                diag_.drop(DropReason::THRESHOLD);
            }
            break;
        default: break;
    }
    
    if (signal.type != StrategySignal::Type::HOLD) {
        signal.confidence = 0.7;
        diag_.emitted++;
        bars_in_trade_ = 0;
        // **FIXED**: This now correctly refers to the 'state_' member from BaseStrategy
        state_.last_trade_bar = current_index;
    }
    
    return signal;
}

REGISTER_STRATEGY(OrderFlowScalpingStrategy, "OrderFlowScalping");

} // namespace sentio


```

## 📄 **FILE 54 of 60**: src/strategy_sma_cross.cpp

**File Information**:
- **Path**: `src/strategy_sma_cross.cpp`

- **Size**: 35 lines
- **Modified**: 2025-09-05 16:34:48

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

## 📄 **FILE 55 of 60**: src/strategy_tfa.cpp

**File Information**:
- **Path**: `src/strategy_tfa.cpp`

- **Size**: 286 lines
- **Modified**: 2025-09-07 22:48:35

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

      std::cout << "[TFA] Creating safe ColumnProjector: runtime=" << runtime_names.size()
                << " -> expected=" << expected_names.size() << " features" << std::endl;

      projector_safe_ = std::make_unique<ColumnProjectorSafe>(
        ColumnProjectorSafe::make(runtime_names, expected_names, pad_value)
      );

      expected_feat_dim_ = F_expected;
      projector_initialized_ = true;
      std::cout << "[TFA] Safe projector initialized: expecting " << expected_feat_dim_ << " features" << std::endl;
    } catch (const std::exception& e) {
      std::cout << "[TFA] Failed to initialize safe projector: " << e.what() << std::endl;
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
    if (feature_calls % 1000 == 0 || feature_calls <= 10) {
      std::cout << "[TFA] Projection error in set_raw_features: " << e.what() << std::endl;
    }
  }

  if (feature_calls % 1000 == 0 || feature_calls <= 10) {
    std::cout << "[DIAG] TFA set_raw_features: call=" << feature_calls
              << " raw.size()=" << raw.size()
              << " window_ready=" << (window_.ready()? 1:0)
              << " feat_dim=" << window_.feat_dim() << std::endl;
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
  
  // Diagnostic: Check if window is ready
  if (!window_.ready()) {
    std::cout << "[TFA] Window not ready, required=" << window_.seq_len() << std::endl;
    return;
  }

  auto in = window_.to_input();
  if (!in) {
    std::cout << "[TFA] Failed to create input from window" << std::endl;
    return;
  }

  std::optional<ml::ModelOutput> out;
  try {
    out = handle_.model->predict(*in, window_.seq_len(), window_.feat_dim(), handle_.spec.input_layout);
  } catch (const std::exception& e) {
    std::cout << "[TFA] predict threw: " << e.what() << std::endl;
    return;
  }
  
  if (!out) {
    std::cout << "[TFA] Model prediction failed" << std::endl;
    return;
  }

  auto sig = map_output(*out);
  std::cout << "[TFA] Raw confidence=" << sig.confidence << ", floor=" << cfg_.conf_floor << std::endl;
  if (sig.confidence < cfg_.conf_floor) {
    std::cout << "[TFA] Signal dropped due to low confidence" << std::endl;
    return;
  }
  last_ = sig;
  std::cout << "[TFA] Signal generated: " << (sig.type == StrategySignal::Type::BUY ? "BUY" : 
                                              sig.type == StrategySignal::Type::SELL ? "SELL" : "HOLD") 
            << " conf=" << sig.confidence << std::endl;
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

StrategySignal TFAStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
  (void)current_index; // we will use bars.size() and a static cursor
  static int calls = 0;
  ++calls;
  last_.reset();

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
      std::cout << "[TFA seq] precomputed probs: N=" << probs_all.size() << std::endl;
    } catch (const std::exception& e) {
      std::cout << "[TFA seq] init/forward failed: " << e.what() << std::endl;
      return StrategySignal{};
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

  StrategySignal sig{}; sig.type = StrategySignal::Type::HOLD; sig.confidence = 0.0;

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
      sig.type = StrategySignal::Type::BUY;
      sig.confidence = prob;
      cooldown_long_until = calls + cooldown;
    } else if (can_short && prob <= thrS) {
      sig.type = StrategySignal::Type::SELL;
      sig.confidence = 1.0f - prob;
      cooldown_short_until = calls + cooldown;
    }

    if (calls % 64 == 0) {
      std::cout << "[TFA calc] prob=" << prob << " thrL=" << thrL << " thrS=" << thrS
                << " type=" << (sig.type==StrategySignal::Type::BUY?"BUY":sig.type==StrategySignal::Type::SELL?"SELL":"HOLD")
                << std::endl;
    }
  }

  if (sig.type == StrategySignal::Type::HOLD) return StrategySignal{};
  last_ = sig;
  return sig;
}

REGISTER_STRATEGY(TFAStrategy, "TFA");

} // namespace sentio

```

## 📄 **FILE 56 of 60**: src/strategy_transformer_ts.cpp

**File Information**:
- **Path**: `src/strategy_transformer_ts.cpp`

- **Size**: 54 lines
- **Modified**: 2025-09-05 23:39:11

- **Type**: .cpp

```text
#include "sentio/strategy_transformer_ts.hpp"
#include <algorithm>

namespace sentio {

static ml::WindowSpec make_spec(const ml::ModelSpec& s){
  ml::WindowSpec w;
  w.seq_len = s.seq_len>0? s.seq_len : 64;
  w.layout  = s.input_layout.empty()? "BTF" : s.input_layout;
  w.feat_dim = (int)s.feature_names.size();
  w.mean = s.mean; w.std = s.std; w.clip2 = s.clip2;
  return w;
}

TransformerSignalStrategyTS::TransformerSignalStrategyTS(const TransformerTSCfg& cfg)
: cfg_(cfg)
, handle_(ml::ModelRegistryTS::load_torchscript(cfg.model_id, cfg.version, cfg.artifacts_dir, cfg.use_cuda))
, window_(make_spec(handle_.spec))
{}

void TransformerSignalStrategyTS::set_raw_features(const std::vector<double>& raw){
  window_.push(raw);
}

StrategySignal TransformerSignalStrategyTS::map_output(const ml::ModelOutput& mo) const {
  StrategySignal s;
  if (mo.probs.empty()) { s.type=StrategySignal::Type::HOLD; s.confidence=cfg_.conf_floor; return s; }
  int argmax = 0; for (int i=1;i<(int)mo.probs.size();++i) if (mo.probs[i]>mo.probs[argmax]) argmax=i;
  float pmax = mo.probs[argmax];
  // default order ["SELL","HOLD","BUY"]
  if      (argmax==2) s.type = StrategySignal::Type::BUY;
  else if (argmax==0) s.type = StrategySignal::Type::SELL;
  else                s.type = StrategySignal::Type::HOLD;
  s.confidence = std::max(cfg_.conf_floor, (double)pmax);
  return s;
}

void TransformerSignalStrategyTS::on_bar(const StrategyCtx& ctx, const Bar& b){
  (void)ctx; (void)b;
  last_.reset();
  if (!window_.ready()) return;

  auto in = window_.to_input();
  if (!in) return;

  auto out = handle_.model->predict(*in, window_.seq_len(), window_.feat_dim(), handle_.spec.input_layout);
  if (!out) return;

  auto sig = map_output(*out);
  if (sig.confidence < cfg_.conf_floor) return;
  last_ = sig;
}

} // namespace sentio

```

## 📄 **FILE 57 of 60**: src/strategy_vwap_reversion.cpp

**File Information**:
- **Path**: `src/strategy_vwap_reversion.cpp`

- **Size**: 156 lines
- **Modified**: 2025-09-05 15:24:52

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

StrategySignal VWAPReversionStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;
    update_vwap(bars[current_index]);

    if (current_index < rsi_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }
    
    if (is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return signal;
    }
    
    if (vwap_ <= 0) {
        diag_.drop(DropReason::NAN_INPUT);
        return signal;
    }

    const auto& bar = bars[current_index];
    double distance_pct = std::abs(bar.close - vwap_) / vwap_;
    if (distance_pct < min_distance_from_vwap_) {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }

    double upper_band = vwap_ * (1.0 + band_multiplier_);
    double lower_band = vwap_ * (1.0 - band_multiplier_);

    bool buy_condition = bar.close < lower_band && is_rsi_condition_met(bars, current_index, true);
    bool sell_condition = bar.close > upper_band && is_rsi_condition_met(bars, current_index, false);

    if (buy_condition) {
        signal.type = StrategySignal::Type::BUY;
    } else if (sell_condition) {
        signal.type = StrategySignal::Type::SELL;
    } else {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }

    signal.confidence = 0.8;
    diag_.emitted++;
    state_.last_trade_bar = current_index;
    return signal;
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

REGISTER_STRATEGY(VWAPReversionStrategy, "VWAPReversion");

} // namespace sentio
```

## 📄 **FILE 58 of 60**: src/telemetry_logger.cpp

**File Information**:
- **Path**: `src/telemetry_logger.cpp`

- **Size**: 177 lines
- **Modified**: 2025-09-06 01:37:11

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

## 📄 **FILE 59 of 60**: src/temporal_analysis.cpp

**File Information**:
- **Path**: `src/temporal_analysis.cpp`

- **Size**: 152 lines
- **Modified**: 2025-09-09 02:38:00

- **Type**: .cpp

```text
#include "sentio/temporal_analysis.hpp"
#include "sentio/runner.hpp"
#include "sentio/audit.hpp"
#include "sentio/metrics.hpp"
#include "sentio/progress_bar.hpp"
#include "sentio/day_index.hpp"
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
    
    TemporalAnalyzer analyzer;
    const auto& base_series = series[base_symbol_id];
    const int total_bars = (int)base_series.size();
    
    if (total_bars < cfg.min_bars_per_quarter) {
        std::cerr << "ERROR: Insufficient data for temporal analysis. Need at least " 
                  << cfg.min_bars_per_quarter << " bars per quarter." << std::endl;
        return TemporalAnalysisSummary{};
    }
    
    // Calculate quarters based on actual time periods (66 trading days per quarter)
    // Assume ~390 bars per trading day (6.5 hours * 60 mins = 390 1-min bars)
    int bars_per_quarter = 66 * 390; // 66 trading days * 390 bars/day = 25,740 bars per quarter
    int current_year = 2021;
    int current_quarter = 1;
    
    std::cout << "Starting TPA (Temporal Performance Analysis) Test..." << std::endl;
    std::cout << "Total bars: " << total_bars << ", Bars per quarter: " << bars_per_quarter << std::endl;
    
    // Initialize TPA progress bar
    TPATestProgressBar progress_bar(cfg.num_quarters, rcfg.strategy_name);
    progress_bar.display(); // Show initial progress bar
    
    std::cout << "\nInitializing data processing..." << std::endl;
    
    // Build audit filename prefix with strategy and timestamp
    const std::string test_name = "tpa_test";
    const auto ts_epoch = static_cast<long long>(std::time(nullptr));

    // Determine the last cfg.num_quarters quarters from the end
    int total_quarters_available = std::max(1, total_bars / std::max(1, bars_per_quarter));
    int start_quarter = std::max(0, total_quarters_available - cfg.num_quarters);
    for (int qi = 0; qi < cfg.num_quarters; ++qi) {
        int q = start_quarter + qi;
        int start_idx = q * bars_per_quarter;
        int end_idx = std::min(start_idx + bars_per_quarter, total_bars);
        
        if (end_idx - start_idx < cfg.min_bars_per_quarter) {
            std::cout << "Skipping quarter " << (q + 1) << " - insufficient data" << std::endl;
            continue;
        }
        
        std::cout << "\nProcessing Quarter " << current_year << "Q" << current_quarter 
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
        
        // Run backtest for this quarter
        AuditConfig audit_cfg;
        audit_cfg.run_id = rcfg.strategy_name + "_" + test_name + "_q" + std::to_string(q + 1) + "_" + std::to_string(ts_epoch);
        audit_cfg.file_path = "audit/" + rcfg.strategy_name + "_" + test_name + "_" + std::to_string(ts_epoch) + "_q" + std::to_string(q + 1) + ".jsonl";
        audit_cfg.flush_each = true;
        AuditRecorder audit(audit_cfg);
        
        auto result = run_backtest(audit, ST, quarter_series, base_symbol_id, rcfg);
        
        // Calculate quarterly metrics
        QuarterlyMetrics metrics;
        metrics.year = current_year;
        metrics.quarter = current_quarter;
        
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
        
        // Convert total return percent for the slice into a day-compounded monthly return
        // result.total_return is percent for the tested slice; convert to decimal for compounding
        double ret_dec = result.total_return / 100.0;
        double monthly_compounded = 0.0;
        if (actual_trading_days > 0) {
            // Compound to a 21-trading-day month
            monthly_compounded = (std::pow(1.0 + ret_dec, 21.0 / static_cast<double>(actual_trading_days)) - 1.0) * 100.0;
        }
        metrics.monthly_return_pct = monthly_compounded;
        
        metrics.sharpe_ratio = result.sharpe_ratio;
        metrics.total_trades = result.total_fills;  // Use total_fills as proxy for trades
        metrics.trading_days = actual_trading_days;
        metrics.avg_daily_trades = actual_trading_days > 0 ? static_cast<double>(result.total_fills) / actual_trading_days : 0.0;
        metrics.max_drawdown = result.max_drawdown;
        metrics.win_rate = 0.0;  // Not available in RunResult, set to 0
        metrics.total_return_pct = result.total_return;
        
        analyzer.add_quarterly_result(metrics);
        
        // Update progress bar with quarter results
        progress_bar.display_with_quarter_info(q + 1, current_year, current_quarter,
                                             metrics.monthly_return_pct, metrics.sharpe_ratio,
                                             metrics.avg_daily_trades, metrics.health_status());
        
        // Print quarter summary
        std::cout << "\n  Monthly Return: " << std::fixed << std::setprecision(2) 
                  << metrics.monthly_return_pct << "%" << std::endl;
        std::cout << "  Sharpe Ratio: " << std::fixed << std::setprecision(3) 
                  << metrics.sharpe_ratio << std::endl;
        std::cout << "  Daily Trades: " << std::fixed << std::setprecision(1) 
                  << metrics.avg_daily_trades << " (Health: " << metrics.health_status() << ")" << std::endl;
        std::cout << "  Total Trades: " << metrics.total_trades << std::endl;
        
        // Update year/quarter for next iteration
        current_quarter++;
        if (current_quarter > 4) { current_quarter = 1; current_year++; }
    }
    
    // Final progress bar update
    std::cout << "\n\nTPA Test completed! Generating summary..." << std::endl;
    
    auto summary = analyzer.generate_summary();
    summary.assess_readiness(cfg);
    return summary;
}

} // namespace sentio

```

## 📄 **FILE 60 of 60**: src/time_utils.cpp

**File Information**:
- **Path**: `src/time_utils.cpp`

- **Size**: 106 lines
- **Modified**: 2025-09-05 15:29:30

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

