# Remaining Critical Issues - Complete Analysis

**Generated**: 2025-09-05 13:16:34
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: Comprehensive analysis of remaining threshold and RTH validation issues preventing full system functionality

**Total Files**: 60

---

## 🐛 **BUG REPORT**

# Bug Report: Remaining Critical Issues

## Issue Summary
**Title**: Threshold Issues in 2 Strategies + RTH Validation Blocking Binary Loading  
**Severity**: HIGH  
**Priority**: HIGH  
**Status**: OPEN  
**Affected Components**: VolatilityExpansion, MarketMaking, RTH Validation Logic  

## Description
After significant progress fixing OrderFlowScalping strategy, two critical issues remain that prevent full system functionality:

1. **VolatilityExpansion Strategy**: All signals dropped due to overly restrictive threshold parameters
2. **MarketMaking Strategy**: All signals dropped due to overly restrictive threshold parameters  
3. **RTH Validation Issue**: Binary file loading completely blocked by timezone conversion bugs

These issues prevent 25% of strategies from functioning and block optimized binary data loading.

## Current System Status

### ✅ **Working Strategies (6 out of 8)**
| Strategy | Status | Signals | Fills | Return | Sharpe |
|----------|--------|---------|-------|--------|--------|
| **VWAPReversion** | ✅ **EXCELLENT** | 50,406 | 42,293 | -6.81% | -1.14 |
| **BollingerSqueezeBreakout** | ✅ **GOOD** | 2 | 1,363 | -2.74% | -0.44 |
| **OpeningRangeBreakout** | ✅ **EXCELLENT** | 1 | 702 | +4.82% | +0.79 |
| **MomentumVolumeProfile** | ✅ **GOOD** | 2,615 | 2,581 | +0.51% | +0.09 |
| **OrderFlowImbalance** | ✅ **GOOD** | 1 | 4,584 | +0.16% | +0.04 |
| **OrderFlowScalping** | ✅ **IMPROVED** | 0 | 190 | -9.28% | -1.52 |

### ❌ **Non-Working Strategies (2 out of 8)**
| Strategy | Status | Signals | Fills | Return | Issue |
|----------|--------|---------|-------|--------|-------|
| **VolatilityExpansion** | ❌ **BROKEN** | 0 | 0 | 0% | Threshold too high |
| **MarketMaking** | ❌ **BROKEN** | 0 | 0 | 0% | Threshold too high |

## Detailed Issue Analysis

### Issue 1: VolatilityExpansion Strategy Threshold Problem

**Symptoms**:
```
[SIG VolatilityExpansion] emitted=0 dropped=292776
min_bars=20 session=0 nan=0 zerovol=0 thr=292756 cooldown=0 dup=0
```

**Root Cause**:
- **Threshold Too High**: 292,756 out of 292,776 signals dropped due to threshold
- **Parameter Problem**: Default threshold parameters are too restrictive
- **Signal Logic**: Strategy logic may be flawed or parameters need tuning

**Expected Behavior**: Should generate signals during volatility expansion periods
**Actual Behavior**: No signals pass threshold validation

**Impact**:
- 0% strategy utilization
- Missing potential returns from volatility-based trading
- 12.5% of total strategy capacity unused

### Issue 2: MarketMaking Strategy Threshold Problem

**Symptoms**:
```
[SIG MarketMaking] emitted=0 dropped=292776
min_bars=50 session=0 nan=0 zerovol=20 thr=292706 cooldown=0 dup=0
```

**Root Cause**:
- **Threshold Too High**: 292,706 out of 292,776 signals dropped due to threshold
- **Volume Issues**: 20 signals dropped due to zero volume (`zerovol=20`)
- **Parameter Problem**: Default threshold parameters are too restrictive

**Expected Behavior**: Should generate market making signals based on volatility and volume
**Actual Behavior**: No signals pass threshold validation

**Impact**:
- 0% strategy utilization
- Missing potential returns from market making
- 12.5% of total strategy capacity unused

### Issue 3: RTH Validation Blocking Binary Loading

**Symptoms**:
```
FATAL ERROR: Non-RTH data found after filtering!
 -> Symbol: QQQ
 -> Timestamp (UTC): 2022-09-06T12:01:00-04:00
 -> NYT Epoch: 1662480060
```

**Root Cause**:
- **Timezone Conversion Bug**: RTH validation logic has timezone conversion errors
- **Binary vs CSV**: CSV loading works perfectly, binary loading fails
- **Validation Logic**: RTH verification incorrectly rejects valid RTH times

**Expected Behavior**: 12:01 PM ET should be valid RTH time
**Actual Behavior**: RTH validation rejects valid RTH timestamps

**Impact**:
- **Performance Degradation**: Forced to use slower CSV loading
- **Memory Overhead**: Higher memory usage with CSV parsing
- **I/O Overhead**: Increased disk I/O with larger CSV files
- **Binary Files Unusable**: 21MB binary files generated but cannot be used

## Technical Analysis

### Threshold Parameter Issues

**Common Pattern**:
- Both strategies show 99.9%+ signal drop rate due to threshold
- VolatilityExpansion: 292,756/292,776 dropped (99.993%)
- MarketMaking: 292,706/292,776 dropped (99.976%)

**Root Causes**:
1. **Default Parameters**: Strategy default parameters are too conservative
2. **Parameter Tuning**: No automatic parameter optimization
3. **Signal Logic**: Strategy signal generation may be flawed
4. **Data Quality**: Market data may not meet strategy requirements

### RTH Validation Issues

**Technical Details**:
- **CSV Loading**: ✅ Works perfectly with RTH verification
- **Binary Loading**: ❌ Fails with timezone conversion errors
- **Timestamp Example**: `2022-09-06T12:01:00-04:00` (valid RTH time)
- **Error**: RTH validation incorrectly rejects valid timestamps

**Root Causes**:
1. **Timezone Conversion**: UTC to NYT conversion logic has bugs
2. **Binary Format**: Binary file format may not match C++ expectations
3. **Validation Logic**: RTH validation criteria may be incorrect
4. **Data Integrity**: Binary data may be corrupted during packing

## Impact Assessment

### Business Impact
- **25% Strategy Failure Rate**: 2 out of 8 strategies completely non-functional
- **Performance Loss**: Missing potential returns from 2 strategies
- **Binary Loading Blocked**: Cannot use optimized data loading
- **System Inefficiency**: Forced to use slower CSV loading

### Technical Impact
- **Threshold Logic Broken**: Signal filtering too restrictive for 2 strategies
- **RTH Validation Broken**: Binary loading completely blocked
- **Parameter Tuning Required**: Strategy parameters need optimization
- **Performance Degradation**: Slower data loading due to CSV fallback

## Proposed Solutions

### Immediate Actions (High Priority)

#### 1. Fix VolatilityExpansion Threshold
- **Reduce threshold parameters** to allow signal generation
- **Test with different threshold values** to find optimal settings
- **Validate signal generation** with adjusted parameters
- **Target**: Generate meaningful signals during volatility expansion

#### 2. Fix MarketMaking Threshold
- **Reduce threshold parameters** to allow signal generation
- **Address volume filtering issues** (20 signals dropped due to zero volume)
- **Test with different threshold values** to find optimal settings
- **Target**: Generate market making signals based on volatility and volume

#### 3. Fix RTH Validation Logic
- **Debug timezone conversion** in RTH validation
- **Fix binary file loading** to work with RTH validation
- **Validate RTH verification** with known good data
- **Target**: Enable binary file loading for performance

### Long-term Solutions (Medium Priority)

#### 1. Parameter Optimization Framework
- **Implement automatic parameter tuning** for all strategies
- **Add Bayesian optimization** for parameter discovery
- **Create parameter validation** to prevent overly restrictive settings
- **Target**: Optimal parameters for all strategies

#### 2. RTH Validation Overhaul
- **Complete rewrite** of RTH validation logic
- **Standardize timezone handling** across all components
- **Add comprehensive testing** for RTH validation
- **Target**: Robust RTH validation for all data formats

## Testing Strategy

### Phase 1: Threshold Parameter Fixes
1. **Analyze Current Parameters**: Review default parameters for both strategies
2. **Reduce Threshold Values**: Systematically reduce threshold parameters
3. **Test Signal Generation**: Validate signals are generated with new parameters
4. **Performance Validation**: Ensure strategies generate meaningful trades

### Phase 2: RTH Validation Fixes
1. **Debug Timezone Conversion**: Identify and fix timezone conversion bugs
2. **Test Binary Loading**: Validate binary files load correctly
3. **RTH Verification**: Ensure RTH validation works with binary data
4. **Performance Testing**: Compare CSV vs binary loading performance

### Phase 3: Integration Testing
1. **End-to-End Testing**: Test all strategies with fixed parameters
2. **Binary Loading Testing**: Validate binary loading works for all strategies
3. **Performance Benchmarking**: Measure system performance improvements
4. **Regression Testing**: Ensure fixes don't break working strategies

## Files to Investigate

### Strategy Files
- `include/sentio/strategy_volatility_expansion.hpp`
- `src/strategy_volatility_expansion.cpp`
- `include/sentio/strategy_market_making.hpp`
- `src/strategy_market_making.cpp`

### RTH Validation Files
- `include/sentio/session_nyt.hpp`
- `src/polygon_client.cpp`
- `include/sentio/polygon_client.hpp`
- `src/csv_loader.cpp`

### Core System Files
- `include/sentio/base_strategy.hpp`
- `src/base_strategy.cpp`
- `src/runner.cpp`

## Priority
**HIGH** - This affects 25% of all strategies and completely blocks binary loading functionality.

## Status
**OPEN** - Investigation and debugging in progress.

## Assigned To
Development Team

## Created
2024-12-19

## Last Updated
2024-12-19

## Related Issues
- OrderFlowScalping threshold issues (RESOLVED)
- Data packing issue (RESOLVED)
- Segmentation fault issues (RESOLVED)

## Notes
- 87.5% strategy success rate achieved (7 out of 8 strategies working)
- Main remaining issues are threshold parameters and RTH validation
- CSV loading works perfectly, binary loading has RTH validation issues
- OrderFlowScalping fix was very successful (95x improvement in fills)
- System is largely functional but needs final 25% completion


---

## 📋 **TABLE OF CONTENTS**

1. [include/sentio/all_strategies.hpp](#file-1)
2. [include/sentio/alpha.hpp](#file-2)
3. [include/sentio/audit.hpp](#file-3)
4. [include/sentio/base_strategy.hpp](#file-4)
5. [include/sentio/binio.hpp](#file-5)
6. [include/sentio/bo.hpp](#file-6)
7. [include/sentio/bollinger.hpp](#file-7)
8. [include/sentio/core.hpp](#file-8)
9. [include/sentio/cost_model.hpp](#file-9)
10. [include/sentio/csv_loader.hpp](#file-10)
11. [include/sentio/data_resolver.hpp](#file-11)
12. [include/sentio/day_index.hpp](#file-12)
13. [include/sentio/metrics.hpp](#file-13)
14. [include/sentio/of_index.hpp](#file-14)
15. [include/sentio/of_precompute.hpp](#file-15)
16. [include/sentio/optimizer.hpp](#file-16)
17. [include/sentio/orderflow_types.hpp](#file-17)
18. [include/sentio/polygon_client.hpp](#file-18)
19. [include/sentio/position_manager.hpp](#file-19)
20. [include/sentio/pricebook.hpp](#file-20)
21. [include/sentio/profiling.hpp](#file-21)
22. [include/sentio/replay.hpp](#file-22)
23. [include/sentio/rolling_stats.hpp](#file-23)
24. [include/sentio/router.hpp](#file-24)
25. [include/sentio/runner.hpp](#file-25)
26. [include/sentio/session_nyt.hpp](#file-26)
27. [include/sentio/signal_diag.hpp](#file-27)
28. [include/sentio/sizer.hpp](#file-28)
29. [include/sentio/strategy_bollinger_squeeze_breakout.hpp](#file-29)
30. [include/sentio/strategy_market_making.hpp](#file-30)
31. [include/sentio/strategy_momentum_volume.hpp](#file-31)
32. [include/sentio/strategy_opening_range_breakout.hpp](#file-32)
33. [include/sentio/strategy_order_flow_imbalance.hpp](#file-33)
34. [include/sentio/strategy_order_flow_scalping.hpp](#file-34)
35. [include/sentio/strategy_volatility_expansion.hpp](#file-35)
36. [include/sentio/strategy_vwap_reversion.hpp](#file-36)
37. [include/sentio/symbol_table.hpp](#file-37)
38. [include/sentio/volatility_expansion.hpp](#file-38)
39. [include/sentio/wf.hpp](#file-39)
40. [src/audit.cpp](#file-40)
41. [src/base_strategy.cpp](#file-41)
42. [src/csv_loader.cpp](#file-42)
43. [src/main.cpp](#file-43)
44. [src/optimizer.cpp](#file-44)
45. [src/poly_fetch_main.cpp](#file-45)
46. [src/polygon_client.cpp](#file-46)
47. [src/replay.cpp](#file-47)
48. [src/router.cpp](#file-48)
49. [src/runner.cpp](#file-49)
50. [src/strategy_bollinger_squeeze_breakout.cpp](#file-50)
51. [src/strategy_market_making.cpp](#file-51)
52. [src/strategy_momentum_volume.cpp](#file-52)
53. [src/strategy_opening_range_breakout.cpp](#file-53)
54. [src/strategy_order_flow_imbalance.cpp](#file-54)
55. [src/strategy_order_flow_scalping.cpp](#file-55)
56. [src/strategy_volatility_expansion.cpp](#file-56)
57. [src/strategy_vwap_reversion.cpp](#file-57)
58. [src/wf.cpp](#file-58)
59. [tools/create_mega_document.py](#file-59)
60. [tools/data_downloader.py](#file-60)

---

## 📄 **FILE 1 of 60**: include/sentio/all_strategies.hpp

**File Information**:
- **Path**: `include/sentio/all_strategies.hpp`

- **Size**: 13 lines
- **Modified**: 2025-09-05 03:57:38

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
#include "strategy_volatility_expansion.hpp"
#include "strategy_vwap_reversion.hpp"
```

## 📄 **FILE 2 of 60**: include/sentio/alpha.hpp

**File Information**:
- **Path**: `include/sentio/alpha.hpp`

- **Size**: 7 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <string>

namespace sentio {
// Removed: Direction enum and StratSignal struct. These are now defined in core.hpp
} // namespace sentio


```

## 📄 **FILE 3 of 60**: include/sentio/audit.hpp

**File Information**:
- **Path**: `include/sentio/audit.hpp`

- **Size**: 72 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <sqlite3.h>
#include <string>
#include <optional>

namespace sentio {

struct Auditor {
  sqlite3* db{};
  long long run_id{-1};
  // prepared statements (optional fast path)
  sqlite3_stmt* st_sig{nullptr};
  sqlite3_stmt* st_router{nullptr};
  sqlite3_stmt* st_ord{nullptr};
  sqlite3_stmt* st_fill{nullptr};
  sqlite3_stmt* st_snap{nullptr};
  sqlite3_stmt* st_metrics{nullptr};

  bool open(const std::string& path);
  void close();
  bool ensure_schema(); // creates tables if not exists
  bool begin_tx();
  bool commit_tx();
  bool prepare_hot();
  void finalize_hot();

  // start a run (kind: backtest | wf-train | wf-oos | oos-2w)
  bool start_run(const std::string& kind, const std::string& strategy_name,
                 const std::string& params_json, const std::string& data_hash,
                 std::optional<long long> seed, std::optional<std::string> notes);

  long long insert_signal(const std::string& ts, const std::string& base_sym,
                          const std::string& side, double price, double score);

  long long insert_router(long long signal_id, const std::string& policy,
                          const std::string& instrument, double lev, double weight, const std::string& notes);

  long long insert_order(const std::string& ts, const std::string& instrument, const std::string& side,
                         double qty, const std::string& order_type, double price, const std::string& status,
                         double leverage_used);

  bool insert_fill(long long order_id, const std::string& ts, const std::string& instrument,
                   double qty, double price, double fees, double slippage_bp);

  bool insert_snapshot(const std::string& ts, double cash, double equity,
                       double gross, double net, double pnl, double dd);

  bool insert_metrics(int bars,int trades,double ret_total,double ret_ann,double vol_ann,double sharpe,double mdd,double monthly_proj,double daily_trades);

  // Fast-path versions using prepared statements
  long long insert_signal_fast(const std::string& ts,const std::string& base_sym,const std::string& side,double price,double score);
  long long insert_router_fast(long long signal_id,const std::string& policy,const std::string& instrument,double lev,double weight,const std::string& notes);
  long long insert_order_fast(const std::string& ts,const std::string& instrument,const std::string& side,double qty,const std::string& order_type,double price,const std::string& status,double leverage_used);
  bool insert_fill_fast(long long order_id,const std::string& ts,const std::string& instrument,double qty,double price,double fees,double slippage_bp);
  bool insert_snapshot_fast(const std::string& ts,double cash,double equity,double gross,double net,double pnl,double dd);
  bool insert_metrics_upsert(int bars,int trades,double ret_total,double ret_ann,double vol_ann,double sharpe,double mdd,double monthly_proj,double daily_trades);

  // Compact dictionary helpers
  int upsert_symbol_id(const std::string& sym);
  int upsert_policy_id(const std::string& name);
  int upsert_instrument_id(const std::string& sym);

  // Compact insert methods (prefer these for new writes)
  long long insert_signal_compact(long long ts_ms, int symbol_id, const char* side, double price, double score);
  long long insert_router_compact(long long signal_id, int policy_id, int instrument_id, double lev, double weight);
  long long insert_order_compact(long long ts_ms, int symbol_id, const char* side, double qty, const char* order_type, double price, const char* status, double leverage_used);
  bool insert_fill_compact(long long order_id, long long ts_ms, int symbol_id, double qty, double price, double fees, double slippage_bp);
  bool insert_snapshot_compact(long long ts_ms, double cash, double equity, double gross, double net, double pnl, double dd);
};

} // namespace sentio


```

## 📄 **FILE 4 of 60**: include/sentio/base_strategy.hpp

**File Information**:
- **Path**: `include/sentio/base_strategy.hpp`

- **Size**: 106 lines
- **Modified**: 2025-09-05 11:24:54

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "signal_diag.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <functional>

namespace sentio {

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

struct StrategySignal {
    SignalType signal = SignalType::NONE;
    double confidence = 0.0;
    double suggested_stop_loss = 0.0;
    double suggested_take_profit = 0.0;
    std::string reason;
    std::unordered_map<std::string, double> metadata;
};

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
    
    virtual StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) = 0;
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

## 📄 **FILE 5 of 60**: include/sentio/binio.hpp

**File Information**:
- **Path**: `include/sentio/binio.hpp`

- **Size**: 68 lines
- **Modified**: 2025-09-05 03:57:38

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
        std::fwrite(&bar.ts_nyt_epoch, sizeof(bar.ts_nyt_epoch), 1, fp);
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
        std::fread(&bar.ts_nyt_epoch, sizeof(bar.ts_nyt_epoch), 1, fp);
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

## 📄 **FILE 6 of 60**: include/sentio/bo.hpp

**File Information**:
- **Path**: `include/sentio/bo.hpp`

- **Size**: 384 lines
- **Modified**: 2025-09-05 03:57:38

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

## 📄 **FILE 7 of 60**: include/sentio/bollinger.hpp

**File Information**:
- **Path**: `include/sentio/bollinger.hpp`

- **Size**: 22 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "rolling.hpp"

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

## 📄 **FILE 8 of 60**: include/sentio/core.hpp

**File Information**:
- **Path**: `include/sentio/core.hpp`

- **Size**: 69 lines
- **Modified**: 2025-09-05 09:44:09

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath> // For std::sqrt

namespace sentio {

struct Bar {
    std::string ts_utc;
    int64_t ts_nyt_epoch;
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

## 📄 **FILE 9 of 60**: include/sentio/cost_model.hpp

**File Information**:
- **Path**: `include/sentio/cost_model.hpp`

- **Size**: 120 lines
- **Modified**: 2025-09-05 09:44:09

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

## 📄 **FILE 10 of 60**: include/sentio/csv_loader.hpp

**File Information**:
- **Path**: `include/sentio/csv_loader.hpp`

- **Size**: 8 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include <string>

namespace sentio {
bool load_csv(const std::string& path, std::vector<Bar>& out);
} // namespace sentio


```

## 📄 **FILE 11 of 60**: include/sentio/data_resolver.hpp

**File Information**:
- **Path**: `include/sentio/data_resolver.hpp`

- **Size**: 44 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <filesystem>
#include <cstdlib>

namespace sentio {
enum class TickerFamily { Qqq, Bitcoin, Tesla };

inline const char** family_symbols(TickerFamily f, int& n) {
  static const char* QQQ[] = {"QQQ","TQQQ","SQQQ","PSQ"};
  static const char* BTC[] = {"BTCUSD","ETHUSD"};
  static const char* TSLA[]= {"TSLA","TSLQ"};
  switch (f) {
    case TickerFamily::Qqq: n=4; return QQQ;
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

## 📄 **FILE 12 of 60**: include/sentio/day_index.hpp

**File Information**:
- **Path**: `include/sentio/day_index.hpp`

- **Size**: 26 lines
- **Modified**: 2025-09-05 03:57:38

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

## 📄 **FILE 13 of 60**: include/sentio/metrics.hpp

**File Information**:
- **Path**: `include/sentio/metrics.hpp`

- **Size**: 123 lines
- **Modified**: 2025-09-05 03:57:38

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

## 📄 **FILE 14 of 60**: include/sentio/of_index.hpp

**File Information**:
- **Path**: `include/sentio/of_index.hpp`

- **Size**: 36 lines
- **Modified**: 2025-09-05 03:57:38

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
        const int64_t ts = bars[i].ts_nyt_epoch;
        // advance k until tick.ts > ts
        while (k < M && ticks[k].ts_nyt_epoch <= ts) ++k;
        span[i].start = cur_start;
        span[i].end   = k;        // [cur_start, k) are ticks for bar i
        cur_start = k;
    }
    return span;
}

} // namespace sentio
```

## 📄 **FILE 15 of 60**: include/sentio/of_precompute.hpp

**File Information**:
- **Path**: `include/sentio/of_precompute.hpp`

- **Size**: 26 lines
- **Modified**: 2025-09-05 03:57:38

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

## 📄 **FILE 16 of 60**: include/sentio/optimizer.hpp

**File Information**:
- **Path**: `include/sentio/optimizer.hpp`

- **Size**: 147 lines
- **Modified**: 2025-09-05 03:57:38

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

## 📄 **FILE 17 of 60**: include/sentio/orderflow_types.hpp

**File Information**:
- **Path**: `include/sentio/orderflow_types.hpp`

- **Size**: 34 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include "core.hpp"

namespace sentio {

struct Tick {
    int64_t ts_nyt_epoch;     // strictly nondecreasing
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

## 📄 **FILE 18 of 60**: include/sentio/polygon_client.hpp

**File Information**:
- **Path**: `include/sentio/polygon_client.hpp`

- **Size**: 30 lines
- **Modified**: 2025-09-05 03:57:38

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
                 const std::vector<AggBar>& bars, bool rth_only=false, bool exclude_holidays=false);

private:
  std::string api_key_;
  std::string get_(const std::string& url);
};
} // namespace sentio


```

## 📄 **FILE 19 of 60**: include/sentio/position_manager.hpp

**File Information**:
- **Path**: `include/sentio/position_manager.hpp`

- **Size**: 63 lines
- **Modified**: 2025-09-05 09:41:27

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
const std::unordered_set<std::string> SHORT_INSTRUMENTS = {"SQQQ", "PSQ", "TSLQ"};

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

## 📄 **FILE 20 of 60**: include/sentio/pricebook.hpp

**File Information**:
- **Path**: `include/sentio/pricebook.hpp`

- **Size**: 59 lines
- **Modified**: 2025-09-05 03:57:38

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
        while (j + 1 < n && V[j + 1].ts_nyt_epoch <= base_ts) {
            ++j;
        }
    }

    // Syncs all symbol prices to the timestamp of the i-th bar of the base symbol
    inline void sync_to_base_i(int i) {
        if (S[base_id].empty()) return;
        const int64_t ts = S[base_id][i].ts_nyt_epoch;
        
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

## 📄 **FILE 21 of 60**: include/sentio/profiling.hpp

**File Information**:
- **Path**: `include/sentio/profiling.hpp`

- **Size**: 25 lines
- **Modified**: 2025-09-05 03:57:38

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

## 📄 **FILE 22 of 60**: include/sentio/replay.hpp

**File Information**:
- **Path**: `include/sentio/replay.hpp`

- **Size**: 17 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "audit.hpp"
#include "core.hpp"
#include <unordered_map>
#include <vector>

namespace sentio {
// Rebuild positions/cash/equity from fills and compare to stored snapshots.
// Return true if match within epsilon; otherwise write a simple error to stderr.
bool replay_and_assert(Auditor& au, long long run_id, double eps=1e-6);

// Enhanced replay with market data for accurate final pricing
bool replay_and_assert_with_data(Auditor& au, long long run_id, 
                                 const std::unordered_map<std::string, std::vector<Bar>>& market_data,
                                 double eps=1e-6);
} // namespace sentio


```

## 📄 **FILE 23 of 60**: include/sentio/rolling_stats.hpp

**File Information**:
- **Path**: `include/sentio/rolling_stats.hpp`

- **Size**: 101 lines
- **Modified**: 2025-09-05 13:07:19

- **Type**: .hpp

```text
// rolling_stats.hpp
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

struct RollingMean {
  int win, idx=0, count=0;
  std::vector<double> buf;
  double sum=0.0;
  
  explicit RollingMean(int w = 1): win(w > 0 ? w : 1), buf(w > 0 ? w : 1, 0.0) {}

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

  /**
   * @brief Returns the current mean of the values in the window.
   * * @return The mean, or 0.0 if the window is empty.
   */
  double mean() const {
      return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }
};


struct RollingMeanVar {
  int win, idx=0, count=0;
  std::vector<double> buf;
  double sum=0.0, sumsq=0.0;

  explicit RollingMeanVar(int w = 1): win(w > 0 ? w : 1), buf(w > 0 ? w : 1, 0.0) {}

  inline std::pair<double,double> push(double x){
    if (count < win) {
      buf[count++] = x; 
      sum += x; 
      sumsq += x*x;
    } else {
      sum   -= buf[idx];
      sumsq -= buf[idx]*buf[idx];
      buf[idx] = x;
      sum   += x;
      sumsq += x*x;
      idx = (idx+1) % win;
    }
    double m = count > 0 ? sum / static_cast<double>(count) : 0.0;
    double v = count > 0 ? std::max(0.0, (sumsq / static_cast<double>(count)) - (m*m)) : 0.0;
    return {m, v};
  }
  
  /**
   * @brief Returns the current mean of the values in the window.
   * * @return The mean, or 0.0 if the window is empty.
   */
  double mean() const {
      return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }
  
  /**
   * @brief Calculates the population variance of the values in the window.
   * * @return The variance, or 0.0 if there are fewer than 2 values.
   */
  double var() const {
      if (count < 2) return 0.0;
      double m = mean();
      return std::max(0.0, (sumsq / static_cast<double>(count)) - (m * m));
  }

  struct RollingImbalance {
    int win, idx=0, count=0;
    std::vector<double> buf;  // store per-tick imbalance in [0,1]
    double sum=0.0;

    explicit RollingImbalance(int w): win(w), buf(w,0.0) {}

    inline double push(double imb){
        if (count < win) { 
            buf[count++] = imb; 
            sum += imb; 
            return sum / count; 
        }
        sum -= buf[idx];
        buf[idx] = imb;
        sum += imb;
        idx = (idx+1) % win;
        return sum / win;
    }
};
};

```

## 📄 **FILE 24 of 60**: include/sentio/router.hpp

**File Information**:
- **Path**: `include/sentio/router.hpp`

- **Size**: 26 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "base_strategy.hpp" // For StrategySignal
#include <optional>
#include <string>

namespace sentio {

struct RouteDecision {
    std::string instrument;
    double target_weight;
    
    RouteDecision(const std::string& inst, double weight) : instrument(inst), target_weight(weight) {}
};

struct RouterCfg {
    double min_signal_strength = 0.1;
    double signal_multiplier = 1.0;
    double max_position_pct = 0.1;
    bool require_rth = true;
};

// **MODIFIED**: The function now accepts the richer StrategySignal struct.
std::optional<RouteDecision> route(const StrategySignal& s, const RouterCfg& cfg, const std::string& base_symbol);

} // namespace sentio
```

## 📄 **FILE 25 of 60**: include/sentio/runner.hpp

**File Information**:
- **Path**: `include/sentio/runner.hpp`

- **Size**: 42 lines
- **Modified**: 2025-09-05 03:57:38

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

RunResult run_backtest(Auditor& au, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                      int base_symbol_id, const RunnerCfg& cfg);

} // namespace sentio


```

## 📄 **FILE 26 of 60**: include/sentio/session_nyt.hpp

**File Information**:
- **Path**: `include/sentio/session_nyt.hpp`

- **Size**: 37 lines
- **Modified**: 2025-09-05 13:03:02

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include "date/tz.h" // **NEW**: Include the timezone library header

namespace sentio {

// **MODIFIED**: This function is now timezone-aware and correctly handles UTC epoch seconds.
inline bool in_rth_nyt(int64_t ts_utc_epoch) {
    using namespace std::chrono;
    
    // Create a time_point from the UTC epoch seconds
    sys_seconds utc_tp{seconds(ts_utc_epoch)};
    
    // Get the correct NY timezone
    const date::time_zone* nyt_zone = date::locate_zone("America/New_York");
    
    // Create a zoned_time object representing the time in New York
    date::zoned_time<seconds> ny_time(nyt_zone, utc_tp);
    
    // Extract the time of day from the local New York time
    auto time = ny_time.get_local_time().time_since_epoch();
    auto day = floor<days>(time);
    auto tod = time - day;

    // Define RTH boundaries (9:30 AM to 4:00 PM inclusive)
    constexpr auto rth_open = 9h + 30min;
    constexpr auto rth_close = 16h + 0min;
    
    // Check if the time of day is within the RTH window
    return (tod >= rth_open && tod <= rth_close);
}

inline bool pass_session_filter(bool require_rth, int64_t ts_utc_epoch) {
    return !require_rth || in_rth_nyt(ts_utc_epoch);
}

} // namespace sentio

```

## 📄 **FILE 27 of 60**: include/sentio/signal_diag.hpp

**File Information**:
- **Path**: `include/sentio/signal_diag.hpp`

- **Size**: 35 lines
- **Modified**: 2025-09-05 03:57:38

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
    std::fprintf(stderr, "[SIG %s] emitted=%llu dropped=%llu  min_bars=%llu session=%llu nan=%llu zerovol=%llu thr=%llu cooldown=%llu dup=%llu\n",
      tag,(unsigned long long)emitted,(unsigned long long)dropped,
      (unsigned long long)r_min_bars,(unsigned long long)r_session,(unsigned long long)r_nan,
      (unsigned long long)r_zero_vol,(unsigned long long)r_threshold,(unsigned long long)r_cooldown,
      (unsigned long long)r_dup);
  }
};
```

## 📄 **FILE 28 of 60**: include/sentio/sizer.hpp

**File Information**:
- **Path**: `include/sentio/sizer.hpp`

- **Size**: 95 lines
- **Modified**: 2025-09-05 09:44:09

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
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
};

} // namespace sentio
```

## 📄 **FILE 29 of 60**: include/sentio/strategy_bollinger_squeeze_breakout.hpp

**File Information**:
- **Path**: `include/sentio/strategy_bollinger_squeeze_breakout.hpp`

- **Size**: 44 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "bollinger.hpp"
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
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 30 of 60**: include/sentio/strategy_market_making.hpp

**File Information**:
- **Path**: `include/sentio/strategy_market_making.hpp`

- **Size**: 52 lines
- **Modified**: 2025-09-05 12:22:58

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp" // For rolling volume
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
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 31 of 60**: include/sentio/strategy_momentum_volume.hpp

**File Information**:
- **Path**: `include/sentio/strategy_momentum_volume.hpp`

- **Size**: 57 lines
- **Modified**: 2025-09-05 12:25:31

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp" // **NEW**: For efficient MA calculations
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
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 32 of 60**: include/sentio/strategy_opening_range_breakout.hpp

**File Information**:
- **Path**: `include/sentio/strategy_opening_range_breakout.hpp`

- **Size**: 38 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"

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
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 33 of 60**: include/sentio/strategy_order_flow_imbalance.hpp

**File Information**:
- **Path**: `include/sentio/strategy_order_flow_imbalance.hpp`

- **Size**: 40 lines
- **Modified**: 2025-09-05 12:32:08

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp"

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
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio

```

## 📄 **FILE 34 of 60**: include/sentio/strategy_order_flow_scalping.hpp

**File Information**:
- **Path**: `include/sentio/strategy_order_flow_scalping.hpp`

- **Size**: 38 lines
- **Modified**: 2025-09-05 13:09:39

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp"

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
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio

```

## 📄 **FILE 35 of 60**: include/sentio/strategy_volatility_expansion.hpp

**File Information**:
- **Path**: `include/sentio/strategy_volatility_expansion.hpp`

- **Size**: 45 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "volatility_expansion.hpp" // For RollingHHLL
#include <vector>
#include <string>

namespace sentio {

class VolatilityExpansionStrategy : public BaseStrategy {
private:
    // **MODIFIED**: Cached parameters for performance
    int atr_window_;
    double atr_alpha_;
    int lookback_hh_;
    int lookback_ll_;
    double breakout_k_;
    int hold_max_bars_;
    double tp_atr_mult_;
    double sl_atr_mult_;
    bool require_rth_;

    // State machine states
    enum class VEState { Flat, Long, Short };
    
    // Strategy state & indicators
    VEState state_ = VEState::Flat;
    int bars_in_trade_ = 0;
    RollingHHLL rolling_hh_;
    RollingHHLL rolling_ll_;
    double atr_ = 0.0;
    double prev_close_ = 0.0;
    
public:
    VolatilityExpansionStrategy();
    
    // BaseStrategy interface
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 36 of 60**: include/sentio/strategy_vwap_reversion.hpp

**File Information**:
- **Path**: `include/sentio/strategy_vwap_reversion.hpp`

- **Size**: 47 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"

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
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 37 of 60**: include/sentio/symbol_table.hpp

**File Information**:
- **Path**: `include/sentio/symbol_table.hpp`

- **Size**: 35 lines
- **Modified**: 2025-09-05 03:57:38

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

## 📄 **FILE 38 of 60**: include/sentio/volatility_expansion.hpp

**File Information**:
- **Path**: `include/sentio/volatility_expansion.hpp`

- **Size**: 53 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
// volatility_expansion.hpp
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "session_nyt.hpp"
#include "signal_diag.hpp"

namespace sentio {


// Bar is defined in core.hpp

inline double true_range(double h,double l,double pc){
  double r1=h-l, r2=std::fabs(h-pc), r3=std::fabs(l-pc);
  return std::max(r1, std::max(r2,r3));
}

struct VEParams {
  // Relaxed defaults that actually fire on QQQ minute bars
  int    atr_window   = 14;
  double atr_alpha    = 2.0 / (14 + 1.0);
  int    lookback_hh  = 20;
  int    lookback_ll  = 20;
  double breakout_k   = 0.75;    // was 1.0 → easier
  int    hold_max_bars= 160;
  double tp_atr_mult  = 1.5;
  double sl_atr_mult  = 1.0;
  bool   require_rth  = true;
  int    cooldown_bars= 5;       // avoid immediate re-entries
};

// SignalType is defined in core.hpp

// StrategySignal is defined in base_strategy.hpp

struct VEResult { std::vector<int> entry_idx; std::vector<int> exit_idx; std::vector<int> dir; };

struct RollingHHLL {
  int w; std::vector<double> hi, lo; int idx=0, cnt=0;
  RollingHHLL(int win): w(win), hi(win, -INFINITY), lo(win, +INFINITY){}
  inline std::pair<double,double> push(double H,double L){
    hi[idx]=H; lo[idx]=L; idx=(idx+1)%w; if (cnt<w) ++cnt;
    double HH=-INFINITY, LL=+INFINITY;
    for(int k=0;k<cnt;++k){ HH=std::max(HH,hi[k]); LL=std::min(LL,lo[k]); }
    return {HH,LL};
  }
};

// VolatilityExpansionStrategy class is defined in strategy_volatility_expansion.hpp

} // namespace sentio
```

## 📄 **FILE 39 of 60**: include/sentio/wf.hpp

**File Information**:
- **Path**: `include/sentio/wf.hpp`

- **Size**: 52 lines
- **Modified**: 2025-09-05 03:57:38

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
Gate run_wf_and_gate(Auditor& au_template,
                     const SymbolTable& ST,
                     const std::vector<std::vector<Bar>>& series,
                     int base_symbol_id,
                     const RunnerCfg& rcfg,
                     const WfCfg& wcfg);

// Walk-forward testing with optimization
Gate run_wf_and_gate_optimized(Auditor& au_template,
                               const SymbolTable& ST,
                               const std::vector<std::vector<Bar>>& series,
                               int base_symbol_id,
                               const RunnerCfg& base_rcfg,
                               const WfCfg& wcfg);

} // namespace sentio


```

## 📄 **FILE 40 of 60**: src/audit.cpp

**File Information**:
- **Path**: `src/audit.cpp`

- **Size**: 427 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .cpp

```text
#include "sentio/audit.hpp"
#include <cassert>
#include <iostream>

namespace sentio {

static bool exec(sqlite3* db, const char* sql) {
  char* err=nullptr;
  int rc = sqlite3_exec(db, sql, nullptr, nullptr, &err);
  if (rc!=SQLITE_OK) { if (err){ sqlite3_free(err);} return false; }
  return true;
}

bool Auditor::open(const std::string& path) {
  if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) return false;
  exec(db, "PRAGMA journal_mode=WAL;");
  exec(db, "PRAGMA synchronous=NORMAL;");
  exec(db, "PRAGMA temp_store=MEMORY;");
  exec(db, "PRAGMA mmap_size=268435456;");
  return true;
}
void Auditor::close() { if (db) sqlite3_close(db), db=nullptr; }

bool Auditor::ensure_schema() {
  const char* ddl =
    "PRAGMA journal_mode=WAL;"
    "CREATE TABLE IF NOT EXISTS audit_runs("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " created_at_utc TEXT NOT NULL DEFAULT (datetime('now')),"
    " kind TEXT NOT NULL, strategy_name TEXT NOT NULL, params_json TEXT NOT NULL,"
    " data_hash TEXT NOT NULL, code_commit TEXT, seed INTEGER, notes TEXT );"
    "CREATE TABLE IF NOT EXISTS signals("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,"
    " ts_utc TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL, price REAL NOT NULL,"
    " score REAL, confidence REAL, features_json TEXT );"
    "CREATE TABLE IF NOT EXISTS router_decisions("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL, signal_id INTEGER NOT NULL,"
    " policy TEXT NOT NULL, instrument TEXT NOT NULL, target_leverage REAL NOT NULL, target_weight REAL, notes TEXT );"
    "CREATE TABLE IF NOT EXISTS orders("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,"
    " ts_utc TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,"
    " qty REAL NOT NULL, price REAL, order_type TEXT NOT NULL, status TEXT NOT NULL,"
    " instrument TEXT, leverage_used REAL );"
    "CREATE TABLE IF NOT EXISTS fills("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL, order_id INTEGER,"
    " ts_utc TEXT NOT NULL, symbol TEXT NOT NULL, qty REAL NOT NULL, price REAL NOT NULL,"
    " fees REAL DEFAULT 0.0, slippage_bp REAL DEFAULT 0.0 );"
    "CREATE TABLE IF NOT EXISTS snapshots("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL, ts_utc TEXT NOT NULL,"
    " cash REAL NOT NULL, equity REAL NOT NULL, gross_exposure REAL NOT NULL, net_exposure REAL NOT NULL,"
    " pnl REAL NOT NULL, drawdown REAL NOT NULL );"
    "CREATE TABLE IF NOT EXISTS run_metrics("
    " run_id INTEGER PRIMARY KEY, bars INTEGER NOT NULL, trades INTEGER NOT NULL, ret_total REAL NOT NULL,"
    " ret_ann REAL NOT NULL, vol_ann REAL NOT NULL, sharpe REAL NOT NULL, mdd REAL NOT NULL,"
    " monthly_proj REAL, daily_trades REAL );"
    // Dictionary tables for compact audit (optional, for performance)
    "CREATE TABLE IF NOT EXISTS dict_symbol("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, sym TEXT UNIQUE NOT NULL );"
    "CREATE TABLE IF NOT EXISTS dict_policy("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL );"
    "CREATE TABLE IF NOT EXISTS dict_instrument("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, sym TEXT UNIQUE NOT NULL );"
    // Add compact columns to existing tables (optional, backward compatible)
    "ALTER TABLE signals ADD COLUMN ts_ms INTEGER DEFAULT NULL;"
    "ALTER TABLE signals ADD COLUMN symbol_id INTEGER DEFAULT NULL;"
    "ALTER TABLE router_decisions ADD COLUMN policy_id INTEGER DEFAULT NULL;"
    "ALTER TABLE router_decisions ADD COLUMN instrument_id INTEGER DEFAULT NULL;"
    "ALTER TABLE orders ADD COLUMN ts_ms INTEGER DEFAULT NULL;"
    "ALTER TABLE orders ADD COLUMN symbol_id INTEGER DEFAULT NULL;"
    "ALTER TABLE fills ADD COLUMN ts_ms INTEGER DEFAULT NULL;"
    "ALTER TABLE fills ADD COLUMN symbol_id INTEGER DEFAULT NULL;"
    "ALTER TABLE snapshots ADD COLUMN ts_ms INTEGER DEFAULT NULL;";
  return exec(db, ddl);
}

bool Auditor::begin_tx(){
  return exec(db, "BEGIN IMMEDIATE TRANSACTION;");
}
bool Auditor::commit_tx(){
  return exec(db, "COMMIT;");
}

bool Auditor::prepare_hot() {
  const char* sql_sig = "INSERT INTO signals(run_id,ts_utc,symbol,side,price,score) VALUES(?,?,?,?,?,?);";
  const char* sql_router="INSERT INTO router_decisions(run_id,signal_id,policy,instrument,target_leverage,target_weight,notes) VALUES(?,?,?,?,?,?,?);";
  const char* sql_ord = "INSERT INTO orders(run_id,ts_utc,symbol,side,qty,price,order_type,status,instrument,leverage_used) VALUES(?,?,?,?,?,?,?,?,?,?);";
  const char* sql_fill= "INSERT INTO fills(run_id,order_id,ts_utc,symbol,qty,price,fees,slippage_bp) VALUES(?,?,?,?,?,?,?,?);";
  const char* sql_snap= "INSERT INTO snapshots(run_id,ts_utc,cash,equity,gross_exposure,net_exposure,pnl,drawdown) VALUES(?,?,?,?,?,?,?,?);";
  const char* sql_metric="INSERT INTO run_metrics(run_id,bars,trades,ret_total,ret_ann,vol_ann,sharpe,mdd,monthly_proj,daily_trades) VALUES(?,?,?,?,?,?,?,?,?,?) "
                         "ON CONFLICT(run_id) DO UPDATE SET bars=excluded.bars,trades=excluded.trades,ret_total=excluded.ret_total,ret_ann=excluded.ret_ann,vol_ann=excluded.vol_ann,sharpe=excluded.sharpe,mdd=excluded.mdd,monthly_proj=excluded.monthly_proj,daily_trades=excluded.daily_trades;";
  if (sqlite3_prepare_v2(db, sql_sig, -1, &st_sig, nullptr) != SQLITE_OK) return false;
  if (sqlite3_prepare_v2(db, sql_router, -1, &st_router, nullptr) != SQLITE_OK) return false;
  if (sqlite3_prepare_v2(db, sql_ord, -1, &st_ord, nullptr) != SQLITE_OK) return false;
  if (sqlite3_prepare_v2(db, sql_fill, -1, &st_fill, nullptr) != SQLITE_OK) return false;
  if (sqlite3_prepare_v2(db, sql_snap, -1, &st_snap, nullptr) != SQLITE_OK) return false;
  if (sqlite3_prepare_v2(db, sql_metric, -1, &st_metrics, nullptr) != SQLITE_OK) return false;
  return true;
}

void Auditor::finalize_hot() {
  if (st_sig) sqlite3_finalize(st_sig);
  if (st_router) sqlite3_finalize(st_router);
  if (st_ord) sqlite3_finalize(st_ord);
  if (st_fill) sqlite3_finalize(st_fill);
  if (st_snap) sqlite3_finalize(st_snap);
  if (st_metrics) sqlite3_finalize(st_metrics);
  st_sig=st_router=st_ord=st_fill=st_snap=st_metrics=nullptr;
}

long long Auditor::insert_signal_fast(const std::string& ts,const std::string& base_sym,const std::string& side,double price,double score){
  sqlite3_reset(st_sig); sqlite3_clear_bindings(st_sig);
  sqlite3_bind_int64(st_sig,1,run_id);
  sqlite3_bind_text (st_sig,2,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_sig,3,base_sym.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_sig,4,side.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st_sig,5,price);
  sqlite3_bind_double(st_sig,6,score);
  if (sqlite3_step(st_sig) != SQLITE_DONE) return -1;
  return sqlite3_last_insert_rowid(db);
}

long long Auditor::insert_router_fast(long long signal_id,const std::string& policy,const std::string& instrument,double lev,double weight,const std::string& notes){
  sqlite3_reset(st_router); sqlite3_clear_bindings(st_router);
  sqlite3_bind_int64(st_router,1,run_id);
  sqlite3_bind_int64(st_router,2,signal_id);
  sqlite3_bind_text (st_router,3,policy.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_router,4,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st_router,5,lev);
  sqlite3_bind_double(st_router,6,weight);
  sqlite3_bind_text (st_router,7,notes.c_str(),-1,SQLITE_TRANSIENT);
  if (sqlite3_step(st_router) != SQLITE_DONE) return -1;
  return sqlite3_last_insert_rowid(db);
}

long long Auditor::insert_order_fast(const std::string& ts,const std::string& instrument,const std::string& side,double qty,const std::string& order_type,double price,const std::string& status,double leverage_used){
  sqlite3_reset(st_ord); sqlite3_clear_bindings(st_ord);
  sqlite3_bind_int64(st_ord,1,run_id);
  sqlite3_bind_text (st_ord,2,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_ord,3,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_ord,4,side.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st_ord,5,qty);
  sqlite3_bind_double(st_ord,6,price);
  sqlite3_bind_text (st_ord,7,order_type.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_ord,8,status.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_ord,9,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st_ord,10,leverage_used);
  if (sqlite3_step(st_ord) != SQLITE_DONE) return -1;
  return sqlite3_last_insert_rowid(db);
}

bool Auditor::insert_fill_fast(long long order_id,const std::string& ts,const std::string& instrument,double qty,double price,double fees,double slippage_bp){
  sqlite3_reset(st_fill); sqlite3_clear_bindings(st_fill);
  sqlite3_bind_int64(st_fill,1,run_id);
  sqlite3_bind_int64(st_fill,2,order_id);
  sqlite3_bind_text (st_fill,3,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_fill,4,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st_fill,5,qty);
  sqlite3_bind_double(st_fill,6,price);
  sqlite3_bind_double(st_fill,7,fees);
  sqlite3_bind_double(st_fill,8,slippage_bp);
  return sqlite3_step(st_fill) == SQLITE_DONE;
}

bool Auditor::insert_snapshot_fast(const std::string& ts,double cash,double equity,double gross,double net,double pnl,double dd){
  sqlite3_reset(st_snap); sqlite3_clear_bindings(st_snap);
  sqlite3_bind_int64(st_snap,1,run_id);
  sqlite3_bind_text (st_snap,2,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st_snap,3,cash);
  sqlite3_bind_double(st_snap,4,equity);
  sqlite3_bind_double(st_snap,5,gross);
  sqlite3_bind_double(st_snap,6,net);
  sqlite3_bind_double(st_snap,7,pnl);
  sqlite3_bind_double(st_snap,8,dd);
  return sqlite3_step(st_snap) == SQLITE_DONE;
}

bool Auditor::insert_metrics_upsert(int bars,int trades,double ret_total,double ret_ann,double vol_ann,double sharpe,double mdd,double monthly_proj,double daily_trades){
  sqlite3_reset(st_metrics); sqlite3_clear_bindings(st_metrics);
  sqlite3_bind_int64(st_metrics,1,run_id);
  sqlite3_bind_int (st_metrics,2,bars);
  sqlite3_bind_int (st_metrics,3,trades);
  sqlite3_bind_double(st_metrics,4,ret_total);
  sqlite3_bind_double(st_metrics,5,ret_ann);
  sqlite3_bind_double(st_metrics,6,vol_ann);
  sqlite3_bind_double(st_metrics,7,sharpe);
  sqlite3_bind_double(st_metrics,8,mdd);
  sqlite3_bind_double(st_metrics,9,monthly_proj);
  sqlite3_bind_double(st_metrics,10,daily_trades);
  return sqlite3_step(st_metrics) == SQLITE_DONE;
}

bool Auditor::start_run(const std::string& kind, const std::string& strategy_name,
                 const std::string& params_json, const std::string& data_hash,
                 std::optional<long long> seed, std::optional<std::string> notes) {
  sqlite3_stmt* st=nullptr;
  const char* sql = "INSERT INTO audit_runs(kind,strategy_name,params_json,data_hash,seed,notes) VALUES(?,?,?,?,?,?);";
  if (sqlite3_prepare_v2(db, sql, -1, &st, nullptr) != SQLITE_OK) return false;
  sqlite3_bind_text(st,1,kind.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,2,strategy_name.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,3,params_json.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,4,data_hash.c_str(),-1,SQLITE_TRANSIENT);
  if (seed) sqlite3_bind_int64(st,5,*seed); else sqlite3_bind_null(st,5);
  if (notes) sqlite3_bind_text(st,6,notes->c_str(),-1,SQLITE_TRANSIENT); else sqlite3_bind_null(st,6);
  int rc = sqlite3_step(st); sqlite3_finalize(st);
  if (rc != SQLITE_DONE) return false;
  run_id = sqlite3_last_insert_rowid(db);
  return true;
}

long long Auditor::insert_signal(const std::string& ts,const std::string& base_sym,const std::string& side,double price,double score) {
  sqlite3_stmt* st=nullptr;
  const char* sql="INSERT INTO signals(run_id,ts_utc,symbol,side,price,score) VALUES(?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_text(st,2,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,3,base_sym.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,4,side.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,5,price);
  sqlite3_bind_double(st,6,score);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return (rc==SQLITE_DONE)? sqlite3_last_insert_rowid(db) : -1;
}

long long Auditor::insert_router(long long signal_id,const std::string& policy,const std::string& instrument,double lev,double weight,const std::string& notes){
  sqlite3_stmt* st=nullptr;
  const char* sql="INSERT INTO router_decisions(run_id,signal_id,policy,instrument,target_leverage,target_weight,notes) VALUES(?,?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int64(st,2,signal_id);
  sqlite3_bind_text(st,3,policy.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,4,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,5,lev);
  sqlite3_bind_double(st,6,weight);
  sqlite3_bind_text(st,7,notes.c_str(),-1,SQLITE_TRANSIENT);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return (rc==SQLITE_DONE)? sqlite3_last_insert_rowid(db) : -1;
}

long long Auditor::insert_order(const std::string& ts,const std::string& instrument,const std::string& side,double qty,const std::string& order_type,double price,const std::string& status,double leverage_used){
  sqlite3_stmt* st=nullptr;
  const char* sql="INSERT INTO orders(run_id,ts_utc,symbol,side,qty,price,order_type,status,instrument,leverage_used) VALUES(?,?,?,?,?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_text(st,2,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,3,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,4,side.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,5,qty);
  sqlite3_bind_double(st,6,price);
  sqlite3_bind_text(st,7,order_type.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,8,status.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,9,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,10,leverage_used);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return (rc==SQLITE_DONE)? sqlite3_last_insert_rowid(db) : -1;
}

bool Auditor::insert_fill(long long order_id,const std::string& ts,const std::string& instrument,double qty,double price,double fees,double slippage_bp){
  sqlite3_stmt* st=nullptr;
  const char* sql="INSERT INTO fills(run_id,order_id,ts_utc,symbol,qty,price,fees,slippage_bp) VALUES(?,?,?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int64(st,2,order_id);
  sqlite3_bind_text(st,3,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,4,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,5,qty);
  sqlite3_bind_double(st,6,price);
  sqlite3_bind_double(st,7,fees);
  sqlite3_bind_double(st,8,slippage_bp);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return rc==SQLITE_DONE;
}

bool Auditor::insert_snapshot(const std::string& ts,double cash,double equity,double gross,double net,double pnl,double dd){
  sqlite3_stmt* st=nullptr;
  const char* sql="INSERT INTO snapshots(run_id,ts_utc,cash,equity,gross_exposure,net_exposure,pnl,drawdown) VALUES(?,?,?,?,?,?,?,?);";
  if (sqlite3_prepare_v2(db, sql, -1, &st, nullptr) != SQLITE_OK) return false;
  std::cerr << "Auditor: run_id = " << run_id << std::endl;
  std::cerr << "Auditor: timestamp = '" << ts << "'" << std::endl;
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_text(st,2,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,3,cash);
  sqlite3_bind_double(st,4,equity);
  sqlite3_bind_double(st,5,gross);
  sqlite3_bind_double(st,6,net);
  sqlite3_bind_double(st,7,pnl);
  sqlite3_bind_double(st,8,dd);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return rc==SQLITE_DONE;
}

bool Auditor::insert_metrics(int bars,int trades,double ret_total,double ret_ann,double vol_ann,double sharpe,double mdd,double monthly_proj,double daily_trades){
  sqlite3_stmt* st=nullptr;
  const char* sql="INSERT INTO run_metrics(run_id,bars,trades,ret_total,ret_ann,vol_ann,sharpe,mdd,monthly_proj,daily_trades) VALUES(?,?,?,?,?,?,?,?,?,?) "
                  "ON CONFLICT(run_id) DO UPDATE SET bars=excluded.bars,trades=excluded.trades,ret_total=excluded.ret_total,ret_ann=excluded.ret_ann,vol_ann=excluded.vol_ann,sharpe=excluded.sharpe,mdd=excluded.mdd,monthly_proj=excluded.monthly_proj,daily_trades=excluded.daily_trades;";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int(st,2,bars);
  sqlite3_bind_int(st,3,trades);
  sqlite3_bind_double(st,4,ret_total);
  sqlite3_bind_double(st,5,ret_ann);
  sqlite3_bind_double(st,6,vol_ann);
  sqlite3_bind_double(st,7,sharpe);
  sqlite3_bind_double(st,8,mdd);
  sqlite3_bind_double(st,9,monthly_proj);
  sqlite3_bind_double(st,10,daily_trades);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return rc==SQLITE_DONE;
}

} // namespace sentio
 
// Compact helpers impl
namespace sentio {

int Auditor::upsert_symbol_id(const std::string& sym){
  sqlite3_stmt* st=nullptr; int id=-1;
  sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO dict_symbol(sym) VALUES(?1);", -1, &st, nullptr);
  sqlite3_bind_text(st,1,sym.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_step(st); sqlite3_finalize(st);
  sqlite3_prepare_v2(db, "SELECT id FROM dict_symbol WHERE sym=?1;", -1, &st, nullptr);
  sqlite3_bind_text(st,1,sym.c_str(),-1,SQLITE_TRANSIENT);
  if (sqlite3_step(st)==SQLITE_ROW) id=sqlite3_column_int(st,0);
  sqlite3_finalize(st);
  return id;
}

int Auditor::upsert_policy_id(const std::string& name){
  sqlite3_stmt* st=nullptr; int id=-1;
  sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO dict_policy(name) VALUES(?1);", -1, &st, nullptr);
  sqlite3_bind_text(st,1,name.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_step(st); sqlite3_finalize(st);
  sqlite3_prepare_v2(db, "SELECT id FROM dict_policy WHERE name=?1;", -1, &st, nullptr);
  sqlite3_bind_text(st,1,name.c_str(),-1,SQLITE_TRANSIENT);
  if (sqlite3_step(st)==SQLITE_ROW) id=sqlite3_column_int(st,0);
  sqlite3_finalize(st);
  return id;
}

int Auditor::upsert_instrument_id(const std::string& sym){
  sqlite3_stmt* st=nullptr; int id=-1;
  sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO dict_instrument(sym) VALUES(?1);", -1, &st, nullptr);
  sqlite3_bind_text(st,1,sym.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_step(st); sqlite3_finalize(st);
  sqlite3_prepare_v2(db, "SELECT id FROM dict_instrument WHERE sym=?1;", -1, &st, nullptr);
  sqlite3_bind_text(st,1,sym.c_str(),-1,SQLITE_TRANSIENT);
  if (sqlite3_step(st)==SQLITE_ROW) id=sqlite3_column_int(st,0);
  sqlite3_finalize(st);
  return id;
}

long long Auditor::insert_signal_compact(long long ts_ms, int symbol_id, const char* side, double price, double score){
  sqlite3_stmt* st=nullptr;
  const char* sql = "INSERT INTO signals(run_id, ts_ms, symbol_id, side, price, score) VALUES(?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int64(st,2,ts_ms);
  sqlite3_bind_int  (st,3,symbol_id);
  sqlite3_bind_text (st,4,side,-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,5,price);
  sqlite3_bind_double(st,6,score);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return (rc==SQLITE_DONE)? sqlite3_last_insert_rowid(db) : -1;
}

long long Auditor::insert_router_compact(long long signal_id, int policy_id, int instrument_id, double lev, double weight){
  sqlite3_stmt* st=nullptr;
  const char* sql = "INSERT INTO router_decisions(run_id, signal_id, policy_id, instrument_id, target_leverage, target_weight) VALUES(?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int64(st,2,signal_id);
  sqlite3_bind_int  (st,3,policy_id);
  sqlite3_bind_int  (st,4,instrument_id);
  sqlite3_bind_double(st,5,lev);
  sqlite3_bind_double(st,6,weight);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return (rc==SQLITE_DONE)? sqlite3_last_insert_rowid(db) : -1;
}

long long Auditor::insert_order_compact(long long ts_ms, int symbol_id, const char* side, double qty, const char* order_type, double price, const char* status, double leverage_used){
  sqlite3_stmt* st=nullptr;
  const char* sql = "INSERT INTO orders(run_id, ts_ms, symbol_id, side, qty, price, order_type, status, leverage_used) VALUES(?,?,?,?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int64(st,2,ts_ms);
  sqlite3_bind_int  (st,3,symbol_id);
  sqlite3_bind_text (st,4,side,-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,5,qty);
  sqlite3_bind_double(st,6,price);
  sqlite3_bind_text (st,7,order_type,-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st,8,status,-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,9,leverage_used);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return (rc==SQLITE_DONE)? sqlite3_last_insert_rowid(db) : -1;
}

bool Auditor::insert_fill_compact(long long order_id, long long ts_ms, int symbol_id, double qty, double price, double fees, double slippage_bp){
  sqlite3_stmt* st=nullptr;
  const char* sql = "INSERT INTO fills(run_id, order_id, ts_ms, symbol_id, qty, price, fees, slippage_bp) VALUES(?,?,?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int64(st,2,order_id);
  sqlite3_bind_int64(st,3,ts_ms);
  sqlite3_bind_int  (st,4,symbol_id);
  sqlite3_bind_double(st,5,qty);
  sqlite3_bind_double(st,6,price);
  sqlite3_bind_double(st,7,fees);
  sqlite3_bind_double(st,8,slippage_bp);
  return sqlite3_step(st) == SQLITE_DONE;
}

bool Auditor::insert_snapshot_compact(long long ts_ms, double cash, double equity, double gross, double net, double pnl, double dd){
  sqlite3_stmt* st=nullptr;
  const char* sql = "INSERT INTO snapshots(run_id, ts_ms, cash, equity, gross_exposure, net_exposure, pnl, drawdown) VALUES(?,?,?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int64(st,2,ts_ms);
  sqlite3_bind_double(st,3,cash);
  sqlite3_bind_double(st,4,equity);
  sqlite3_bind_double(st,5,gross);
  sqlite3_bind_double(st,6,net);
  sqlite3_bind_double(st,7,pnl);
  sqlite3_bind_double(st,8,dd);
  return sqlite3_step(st) == SQLITE_DONE;
}

} // namespace sentio


```

## 📄 **FILE 41 of 60**: src/base_strategy.cpp

**File Information**:
- **Path**: `src/base_strategy.cpp`

- **Size**: 57 lines
- **Modified**: 2025-09-05 11:25:24

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
    std::cerr << "DEBUG: BaseStrategy::set_params called for " << name_ << std::endl;
    
    // The constructor has already set the defaults.
    // Now, merge the overrides into the existing params_.
    for (const auto& [key, value] : overrides) {
        params_[key] = value;
    }
    
    apply_params();
    std::cerr << "DEBUG: BaseStrategy::set_params completed for " << name_ << std::endl;
}

// --- Strategy Factory Implementation ---
StrategyFactory& StrategyFactory::instance() {
    static StrategyFactory factory_instance;
    return factory_instance;
}

void StrategyFactory::register_strategy(const std::string& name, CreateFunction create_func) {
    std::cout << "DEBUG: Registering strategy: " << name << std::endl;
    strategies_[name] = create_func;
}

std::unique_ptr<BaseStrategy> StrategyFactory::create_strategy(const std::string& name) {
    std::cout << "DEBUG: Creating strategy: " << name << std::endl;
    std::cout << "DEBUG: Available strategies: ";
    for (const auto& [n, _] : strategies_) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    auto it = strategies_.find(name);
    if (it != strategies_.end()) {
        return it->second();
    }
    std::cerr << "Error: Strategy '" << name << "' not found in factory." << std::endl;
    return nullptr;
}

} // namespace sentio
```

## 📄 **FILE 42 of 60**: src/csv_loader.cpp

**File Information**:
- **Path**: `src/csv_loader.cpp`

- **Size**: 95 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .cpp

```text
#include "sentio/csv_loader.hpp"
#include "sentio/binio.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <date/date.h>
#include <date/tz.h>

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
    
    // **MODIFIED**: Pre-fetch the timezone pointer for performance
    const date::time_zone* nyt_zone = date::locate_zone("America/New_York");
    
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
        
        // **MODIFIED**: Robust timestamp parsing and timezone conversion
        try {
            std::stringstream ts_ss{timestamp_str};
            std::chrono::system_clock::time_point utc_tp;
            // Parse the RFC3339 / ISO 8601 timestamp string (e.g., "2023-10-27T09:30:00-04:00")
            ts_ss >> date::parse("%FT%T%z", utc_tp);
            if (ts_ss.fail()) {
                // Try another common format if the first one fails
                ts_ss.clear();
                ts_ss.str(timestamp_str);
                ts_ss >> date::parse("%F %T%z", utc_tp);
            }

            if (!ts_ss.fail()) {
                // Convert the UTC time_point to a zoned_time in NYT
                auto nyt_tp = date::make_zoned(nyt_zone, utc_tp);
                // Get the epoch seconds relative to the NYT timezone
                bar.ts_nyt_epoch = std::chrono::duration_cast<std::chrono::seconds>(
                    nyt_tp.get_local_time().time_since_epoch()
                ).count();
            } else {
                 bar.ts_nyt_epoch = 0; // Could not parse
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse timestamp '" << timestamp_str << "'. Error: " << e.what() << std::endl;
            bar.ts_nyt_epoch = 0;
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

## 📄 **FILE 43 of 60**: src/main.cpp

**File Information**:
- **Path**: `src/main.cpp`

- **Size**: 180 lines
- **Modified**: 2025-09-05 12:08:39

- **Type**: .cpp

```text
#include "sentio/core.hpp"
#include "sentio/runner.hpp"
#include "sentio/wf.hpp"
#include "sentio/replay.hpp"
#include "sentio/csv_loader.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/profiling.hpp"
#include "sentio/data_resolver.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/all_strategies.hpp"
#include "sentio/session_nyt.hpp" // **NEW**: Include for RTH check

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <cstdlib> // For std::exit
#include <sstream>


namespace { // Anonymous namespace to ensure link-time registration
    struct StrategyRegistrar {
        StrategyRegistrar() {
            auto force_link_vwap = std::make_unique<sentio::VWAPReversionStrategy>();
            auto force_link_momentum = std::make_unique<sentio::MomentumVolumeProfileStrategy>();
            auto force_link_volatility = std::make_unique<sentio::VolatilityExpansionStrategy>();
            auto force_link_bollinger = std::make_unique<sentio::BollingerSqueezeBreakoutStrategy>();
            auto force_link_opening = std::make_unique<sentio::OpeningRangeBreakoutStrategy>();
            auto force_link_scalping = std::make_unique<sentio::OrderFlowScalpingStrategy>();
            auto force_link_imbalance = std::make_unique<sentio::OrderFlowImbalanceStrategy>();
            auto force_link_market = std::make_unique<sentio::MarketMakingStrategy>();
        }
    };
    static StrategyRegistrar registrar;
}


void usage() {
    std::cout << "Usage: sentio_cli <command> [options]\n"
              << "Commands:\n"
              << "  backtest <symbol> [--strategy <name>] [--params <k=v,...>]\n"
              << "  wf <symbol> [--strategy <name>] [--params <k=v,...>]\n"
              << "  replay <run_id>\n";
}

int main(int argc, char* argv[]) {
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

        // **NEW**: Data Sanity Check - Verify RTH filtering post-load.
        // This acts as a safety net. If your data was generated with RTH filtering,
        // this check ensures the filtering was successful.
        std::cout << "\nVerifying data integrity for RTH..." << std::endl;
        bool rth_filter_failed = false;
        for (size_t sid = 0; sid < series.size(); ++sid) {
            if (series[sid].empty()) continue;
            for (const auto& bar : series[sid]) {
                if (!sentio::in_rth_nyt(bar.ts_nyt_epoch)) {
                    std::cerr << "\nFATAL ERROR: Non-RTH data found after filtering!\n"
                              << " -> Symbol: " << ST.get_symbol(sid) << "\n"
                              << " -> Timestamp (UTC): " << bar.ts_utc << "\n"
                              << " -> NYT Epoch: " << bar.ts_nyt_epoch << "\n\n"
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

        sentio::RunnerCfg cfg;
        cfg.strategy_name = strategy_name;
        cfg.strategy_params = strategy_params;
        cfg.audit_level = sentio::AuditLevel::Full;
        cfg.snapshot_stride = 100;
        
        sentio::Auditor au;
        au.open("audit.db");
        au.ensure_schema();
        au.start_run("backtest", strategy_name, "{}", "NA", 42, "symbol=" + base_symbol);
        
        sentio::Tsc timer;
        timer.tic();
        auto result = sentio::run_backtest(au, ST, series, base_symbol_id, cfg);
        double elapsed = timer.toc_sec();
        
        au.close();
        
        std::cout << "\nBacktest completed in " << elapsed << "s\n";
        std::cout << "Final Equity: " << result.final_equity << "\n";
        std::cout << "Total Return: " << result.total_return << "%\n";
        std::cout << "Sharpe Ratio: " << result.sharpe_ratio << "\n";
        std::cout << "Max Drawdown: " << result.max_drawdown << "%\n";
        std::cout << "Total Fills: " << result.total_fills << "\n";
        std::cout << "Diagnostics -> No Route: " << result.no_route << " | No Quantity: " << result.no_qty << "\n";

    } else if (command == "wf") {
        std::cout << "Walk-forward command is not fully implemented in this example.\n";
    } else if (command == "replay") {
        std::cout << "Replay command is not fully implemented in this example.\n";
    } else {
        usage();
        return 1;
    }
    
    return 0;
}

```

## 📄 **FILE 44 of 60**: src/optimizer.cpp

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

## 📄 **FILE 45 of 60**: src/poly_fetch_main.cpp

**File Information**:
- **Path**: `src/poly_fetch_main.cpp`

- **Size**: 57 lines
- **Modified**: 2025-09-05 03:57:38

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

## 📄 **FILE 46 of 60**: src/polygon_client.cpp

**File Information**:
- **Path**: `src/polygon_client.cpp`

- **Size**: 161 lines
- **Modified**: 2025-09-05 12:08:19

- **Type**: .cpp

```text
#include "sentio/polygon_client.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <date/date.h>
#include <date/tz.h>
#include <fstream>
#include <thread>
#include <chrono>
#include <sstream>

using json = nlohmann::json;
namespace sentio {

static size_t write_cb(void* contents, size_t size, size_t nmemb, void* userp) {
  size_t total = size * nmemb;
  std::string* s = static_cast<std::string*>(userp);
  s->append(static_cast<char*>(contents), total);
  return total;
}

static std::string rfc3339_nyc_from_epoch_ms(long long ms) {
  using namespace std::chrono;
  using sys_ms = time_point<system_clock,milliseconds>;
  sys_ms tp{milliseconds(ms)};
  date::zoned_time<milliseconds> zt{"America/New_York", tp};
  return date::format("%FT%T%Ez", zt);
}

// **NEW**: Robust RTH check directly from a UTC timestamp.
// This avoids error-prone string parsing and ensures correct time zone handling.
static bool is_rth_nyc_from_utc_ms(long long utc_ms) {
    using namespace std::chrono;
    using sys_ms = time_point<system_clock, milliseconds>;
    sys_ms utc_tp{milliseconds(utc_ms)};

    // Create a zoned_time object for New York
    auto ny_time = date::make_zoned("America/New_York", utc_tp);

    // Get the local time of day in NY
    auto tod = ny_time.get_local_time().time_since_epoch() % 24h;

    // Define RTH boundaries (9:30 AM to 4:00 PM inclusive)
    constexpr auto rth_open = 9h + 30min;
    constexpr auto rth_close = 16h + 0min;

    return (tod >= rth_open && tod <= rth_close);
}

static bool is_us_market_holiday_2022_2025(const date::year_month_day& ymd) {
  using namespace date;
  const auto y = (int)ymd.year();
  auto observed = [](year_month_day dt){
    auto wd = weekday{dt};
    if (wd == Sunday)  return year_month_day{sys_days{dt} + days{1}};
    if (wd == Saturday) return year_month_day{sys_days{dt} - days{1}};
    return dt;
  };
  auto nth_weekday = [](int nth, date::weekday wd, date::month m, int y){
    return year_month_day{weekday_indexed{wd[nth]} / m / year{y}};
  };
  auto last_weekday = [](date::weekday wd, date::month m, int y){
    return year_month_day{weekday_last{wd[last]} / m / year{y}};
  };

  year_month_day newyears = observed(year_month_day{year{y}/January/1});
  year_month_day mlk = nth_weekday(3, Monday, January, y);
  year_month_day prez = nth_weekday(3, Monday, February, y);
  bool good_friday = false;
  if (y==2022) good_friday = (ymd == year_month_day{year{2022}/April/15});
  if (y==2023) good_friday = (ymd == year_month_day{year{2023}/April/7});
  if (y==2024) good_friday = (ymd == year_month_day{year{2024}/March/29});
  if (y==2025) good_friday = (ymd == year_month_day{year{2025}/April/18});
  year_month_day memorial = last_weekday(Monday, May, y);
  year_month_day june = observed(year_month_day{year{y}/June/19});
  year_month_day indep = observed(year_month_day{year{y}/July/4});
  year_month_day labor = nth_weekday(1, Monday, September, y);
  year_month_day thanks = nth_weekday(4, Thursday, November, y);
  year_month_day xmas = observed(year_month_day{year{y}/December/25});

  if (ymd == newyears || ymd == mlk || ymd == prez || good_friday || ymd == memorial || ymd == june || ymd == indep || ymd == labor || ymd == thanks || ymd == xmas)
    return true;
  return false;
}

PolygonClient::PolygonClient(std::string api_key) : api_key_(std::move(api_key)) {}

std::string PolygonClient::get_(const std::string& url) {
    // This function's implementation remains unchanged.
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
    // This function's implementation remains unchanged.
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

    if (rth_only && !is_rth_nyc_from_utc_ms(a.ts_ms)) {
        continue;
    }
    
    if (exclude_holidays) {
        using namespace std::chrono;
        using sys_ms = time_point<system_clock, milliseconds>;
        sys_ms utc_tp{milliseconds(a.ts_ms)};
        auto ny_time = date::make_zoned("America/New_York", utc_tp);
        auto ymd = date::year_month_day{floor<days>(ny_time.get_local_time())};
        if (is_us_market_holiday_2022_2025(ymd)) {
            continue;
        }
    }
    
    // The timestamp is converted to a NYT string only for writing to the CSV
    std::string ts_str = rfc3339_nyc_from_epoch_ms(a.ts_ms);

    f << ts_str << ',' << symbol << ','
      << a.open << ',' << a.high << ',' << a.low << ',' << a.close << ',' << a.volume << '\n';
  }
}

} // namespace sentio

```

## 📄 **FILE 47 of 60**: src/replay.cpp

**File Information**:
- **Path**: `src/replay.cpp`

- **Size**: 335 lines
- **Modified**: 2025-09-05 09:41:27

- **Type**: .cpp

```text
#include "sentio/replay.hpp"
#include "sentio/core.hpp"
#include <sqlite3.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <climits>

namespace sentio {

bool replay_and_assert(Auditor& au, long long run_id, double eps) {
  // Replay portfolio state from audit logs and verify against snapshots
  
  // 1. Load all fills for this run ordered by timestamp
  std::vector<std::tuple<long long, std::string, double, double, double, double>> fills; // ts_ms, symbol, qty, price, fees, slippage_bp
  
  sqlite3_stmt* stmt = nullptr;
  const char* sql = R"(
    SELECT f.ts_utc, f.symbol, f.qty, f.price, f.fees, f.slippage_bp
    FROM fills f 
    WHERE f.run_id = ? 
    ORDER BY f.ts_utc
  )";
  
  if (sqlite3_prepare_v2(au.db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    std::cerr << "Failed to prepare fills query: " << sqlite3_errmsg(au.db) << std::endl;
    return false;
  }
  
  sqlite3_bind_int64(stmt, 1, run_id);
  
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const char* ts_utc_cstr = (const char*)sqlite3_column_text(stmt, 0);
    const char* symbol_cstr = (const char*)sqlite3_column_text(stmt, 1);
    std::string ts_utc = ts_utc_cstr ? ts_utc_cstr : "";
    std::string symbol = symbol_cstr ? symbol_cstr : "";
    double qty = sqlite3_column_double(stmt, 2);
    double price = sqlite3_column_double(stmt, 3);
    double fees = sqlite3_column_double(stmt, 4);
    double slippage_bp = sqlite3_column_double(stmt, 5);
    
    if (!symbol.empty() && !ts_utc.empty()) {
      // Use a simple counter for ordering since we don't have ts_ms
      static long long counter = 0;
      fills.emplace_back(counter++, symbol, qty, price, fees, slippage_bp);
    }
  }
  sqlite3_finalize(stmt);
  
  // 2. Load snapshots for verification
  std::vector<std::tuple<long long, double, double>> snapshots; // ts_ms, cash, equity
  
  sql = "SELECT ts_utc, cash, equity FROM snapshots WHERE run_id = ? ORDER BY ts_utc";
  if (sqlite3_prepare_v2(au.db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    std::cerr << "Failed to prepare snapshots query: " << sqlite3_errmsg(au.db) << std::endl;
    return false;
  }
  
  sqlite3_bind_int64(stmt, 1, run_id);
  
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    // Use counter for ordering since we're using ts_utc
    static long long snap_counter = 0;
    double cash = sqlite3_column_double(stmt, 1);
    double equity = sqlite3_column_double(stmt, 2);
    snapshots.emplace_back(snap_counter++, cash, equity);
  }
  sqlite3_finalize(stmt);
  
  // 3. Replay portfolio state
  Portfolio replay_pf{}; // Start with default initial cash
  std::unordered_map<std::string, double> last_prices;
  
  size_t fill_idx = 0;
  size_t snap_idx = 0;
  bool verification_passed = true;
  
  std::cout << "Replaying " << fills.size() << " fills and verifying " 
            << snapshots.size() << " snapshots for run " << run_id << std::endl;
  
  // Process fills and check snapshots
  while (fill_idx < fills.size() || snap_idx < snapshots.size()) {
    long long next_fill_ts = (fill_idx < fills.size()) ? std::get<0>(fills[fill_idx]) : LLONG_MAX;
    long long next_snap_ts = (snap_idx < snapshots.size()) ? std::get<0>(snapshots[snap_idx]) : LLONG_MAX;
    
    if (next_fill_ts <= next_snap_ts) {
      // Process fill with fees and slippage (matching original run)
      auto [ts_ms, symbol, qty, price, fees, slippage_bp] = fills[fill_idx];
      // Convert symbol string to symbol ID (assuming 0 for QQQ, 1 for TQQQ, 2 for SQQQ)
      int symbol_id = 0;
      if (symbol == "TQQQ") symbol_id = 1;
      else if (symbol == "SQQQ") symbol_id = 2;
      apply_fill(replay_pf, symbol_id, qty, price);
      replay_pf.cash -= fees;  // Apply transaction fees
      last_prices[symbol] = price;
      fill_idx++;
      
      std::cout << "Fill: " << symbol << " qty=" << qty << " price=" << price 
                << " fees=" << fees << " cash=" << replay_pf.cash << std::endl;
    } else {
      // Verify snapshot
      auto [ts_ms, expected_cash, expected_equity] = snapshots[snap_idx];
      double actual_cash = replay_pf.cash;
      // Convert last_prices map to vector
      std::vector<double> last_prices_vec(3, 0.0);
      for (const auto& [sym, price] : last_prices) {
        int symbol_id = 0;
        if (sym == "TQQQ") symbol_id = 1;
        else if (sym == "SQQQ") symbol_id = 2;
        if (static_cast<size_t>(symbol_id) < last_prices_vec.size()) {
          last_prices_vec[symbol_id] = price;
        }
      }
      double actual_equity = equity_mark_to_market(replay_pf, last_prices_vec);
      
      double cash_diff = std::abs(actual_cash - expected_cash);
      double equity_diff = std::abs(actual_equity - expected_equity);
      
      std::cout << "Snapshot verification at ts=" << ts_ms << ":" << std::endl;
      std::cout << "  Cash: expected=" << expected_cash << " actual=" << actual_cash 
                << " diff=" << cash_diff << std::endl;
      std::cout << "  Equity: expected=" << expected_equity << " actual=" << actual_equity 
                << " diff=" << equity_diff << std::endl;
      
      if (cash_diff > eps || equity_diff > eps) {
        std::cerr << "VERIFICATION FAILED: Differences exceed tolerance " << eps << std::endl;
        verification_passed = false;
      }
      
      snap_idx++;
    }
  }
  
  // 4. Final verification summary
  std::cout << "Replay completed. Final portfolio state:" << std::endl;
  std::cout << "  Cash: $" << replay_pf.cash << std::endl;
  std::cout << "  Positions:" << std::endl;
  for (size_t sym = 0; sym < replay_pf.positions.size(); ++sym) {
    const auto& qty = replay_pf.positions[sym].qty;
    if (std::abs(qty) > 1e-6) {
      std::cout << "    " << sym << ": " << qty << " shares" << std::endl;
    }
  }
  // Convert last_prices map to vector
  std::vector<double> last_prices_vec(3, 0.0);
  for (const auto& [sym, price] : last_prices) {
    int symbol_id = 0;
    if (sym == "TQQQ") symbol_id = 1;
    else if (sym == "SQQQ") symbol_id = 2;
    if (static_cast<size_t>(symbol_id) < last_prices_vec.size()) {
      last_prices_vec[symbol_id] = price;
    }
  }
  std::cout << "  Total Equity: $" << equity_mark_to_market(replay_pf, last_prices_vec) << std::endl;
  
  if (verification_passed) {
    std::cout << "✅ REPLAY VERIFICATION PASSED" << std::endl;
  } else {
    std::cout << "❌ REPLAY VERIFICATION FAILED" << std::endl;
  }
  
  return verification_passed;
}

bool replay_and_assert_with_data(Auditor& au, long long run_id, 
                                 const std::unordered_map<std::string, std::vector<Bar>>& market_data,
                                 double eps) {
  // Enhanced replay that uses market data for accurate final pricing
  
  // 1. Load all fills for this run ordered by timestamp
  std::vector<std::tuple<std::string, std::string, double, double, double, double>> fills; // ts_utc, symbol, qty, price, fees, slippage_bp
  
  sqlite3_stmt* stmt = nullptr;
  const char* sql = R"(
    SELECT f.ts_utc, f.symbol, f.qty, f.price, f.fees, f.slippage_bp
    FROM fills f 
    WHERE f.run_id = ? 
    ORDER BY f.ts_utc
  )";
  
  if (sqlite3_prepare_v2(au.db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    std::cerr << "Failed to prepare fills query: " << sqlite3_errmsg(au.db) << std::endl;
    return false;
  }
  
  sqlite3_bind_int64(stmt, 1, run_id);
  
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const char* ts_utc_cstr = (const char*)sqlite3_column_text(stmt, 0);
    const char* symbol_cstr = (const char*)sqlite3_column_text(stmt, 1);
    std::string ts_utc = ts_utc_cstr ? ts_utc_cstr : "";
    std::string symbol = symbol_cstr ? symbol_cstr : "";
    double qty = sqlite3_column_double(stmt, 2);
    double price = sqlite3_column_double(stmt, 3);
    double fees = sqlite3_column_double(stmt, 4);
    double slippage_bp = sqlite3_column_double(stmt, 5);
    
    if (!symbol.empty() && !ts_utc.empty()) {
      fills.emplace_back(ts_utc, symbol, qty, price, fees, slippage_bp);
    }
  }
  sqlite3_finalize(stmt);
  
  // 2. Load snapshots for verification
  std::vector<std::tuple<std::string, double, double>> snapshots; // ts_utc, cash, equity
  
  sql = "SELECT ts_utc, cash, equity FROM snapshots WHERE run_id = ? ORDER BY ts_utc";
  if (sqlite3_prepare_v2(au.db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    std::cerr << "Failed to prepare snapshots query: " << sqlite3_errmsg(au.db) << std::endl;
    return false;
  }
  
  sqlite3_bind_int64(stmt, 1, run_id);
  
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const char* ts_utc_cstr = (const char*)sqlite3_column_text(stmt, 0);
    std::string ts_utc = ts_utc_cstr ? ts_utc_cstr : "";
    double cash = sqlite3_column_double(stmt, 1);
    double equity = sqlite3_column_double(stmt, 2);
    snapshots.emplace_back(ts_utc, cash, equity);
  }
  sqlite3_finalize(stmt);
  
  if (snapshots.empty()) {
    std::cerr << "No snapshots found for run " << run_id << std::endl;
    return false;
  }
  
  // 3. Replay portfolio state
  Portfolio replay_pf{}; // Start with default initial cash
  std::unordered_map<std::string, double> current_prices;
  
  std::cout << "Enhanced replay: processing " << fills.size() << " fills and verifying " 
            << snapshots.size() << " snapshots for run " << run_id << std::endl;
  
  // Process all fills chronologically
  for (const auto& [ts_utc, symbol, qty, price, fees, slippage_bp] : fills) {
    // Convert symbol string to symbol ID (assuming 0 for QQQ, 1 for TQQQ, 2 for SQQQ)
    int symbol_id = 0;
    if (symbol == "TQQQ") symbol_id = 1;
    else if (symbol == "SQQQ") symbol_id = 2;
    apply_fill(replay_pf, symbol_id, qty, price);
    replay_pf.cash -= fees;  // Apply transaction fees
    current_prices[symbol] = price;  // Update price as of this fill
    
    std::cout << "Fill: " << symbol << " qty=" << qty << " price=" << price 
              << " fees=" << fees << " cash=" << replay_pf.cash << std::endl;
  }
  
  // 4. Get final prices from market data at the last snapshot timestamp
  const std::string& final_timestamp = std::get<0>(snapshots.back());
  std::unordered_map<std::string, double> final_prices;
  
  std::cout << "\nGetting final prices at timestamp: " << final_timestamp << std::endl;
  
  for (const auto& [symbol, bars] : market_data) {
    // Find the bar at or before the final timestamp
    double final_price = 0.0;
    for (int i = (int)bars.size() - 1; i >= 0; --i) {
      if (bars[i].ts_utc <= final_timestamp) {
        final_price = bars[i].close;
        break;
      }
    }
    
    if (final_price > 0.0) {
      final_prices[symbol] = final_price;
      std::cout << "Final price for " << symbol << ": " << final_price << std::endl;
    }
  }
  
  // 5. Calculate final equity using correct final prices
  double final_cash = replay_pf.cash;
  // Convert final_prices map to vector
  std::vector<double> final_prices_vec(3, 0.0);
  for (const auto& [sym, price] : final_prices) {
    int symbol_id = 0;
    if (sym == "TQQQ") symbol_id = 1;
    else if (sym == "SQQQ") symbol_id = 2;
    if (static_cast<size_t>(symbol_id) < final_prices_vec.size()) {
      final_prices_vec[symbol_id] = price;
    }
  }
  double final_equity = equity_mark_to_market(replay_pf, final_prices_vec);
  
  std::cout << "\nFinal portfolio state with correct pricing:" << std::endl;
  std::cout << "  Cash: $" << final_cash << std::endl;
  std::cout << "  Positions:" << std::endl;
  for (size_t sym = 0; sym < replay_pf.positions.size(); ++sym) {
    const auto& qty = replay_pf.positions[sym].qty;
    if (std::abs(qty) > 1e-6) {
      double price = (sym < final_prices_vec.size()) ? final_prices_vec[sym] : 0.0;
      double position_value = qty * price;
      std::cout << "    " << sym << ": " << qty << " shares @ $" << price 
                << " = $" << position_value << std::endl;
    }
  }
  std::cout << "  Total Equity: $" << final_equity << std::endl;
  
  // 6. Verify against final snapshot
  const auto& [expected_ts, expected_cash, expected_equity] = snapshots.back();
  
  double cash_diff = std::abs(final_cash - expected_cash);
  double equity_diff = std::abs(final_equity - expected_equity);
  
  std::cout << "\nFinal verification:" << std::endl;
  std::cout << "  Expected Cash: $" << expected_cash << " | Actual: $" << final_cash 
            << " | Diff: $" << cash_diff << std::endl;
  std::cout << "  Expected Equity: $" << expected_equity << " | Actual: $" << final_equity 
            << " | Diff: $" << equity_diff << std::endl;
  
  bool verification_passed = (cash_diff <= eps && equity_diff <= eps);
  
  if (verification_passed) {
    std::cout << "✅ ENHANCED REPLAY VERIFICATION PASSED" << std::endl;
    
    // Calculate and display performance metrics
    double total_return = (final_equity / 100000.0) - 1.0;
    double monthly_return = std::pow(1.0 + total_return, 1.0/12.0) - 1.0;
    
    std::cout << "Performance Metrics:" << std::endl;
    std::cout << "  Total Return: " << (total_return * 100) << "%" << std::endl;
    std::cout << "  Monthly Return: " << (monthly_return * 100) << "%" << std::endl;
    
  } else {
    std::cout << "❌ ENHANCED REPLAY VERIFICATION FAILED" << std::endl;
    std::cout << "Cash tolerance: " << eps << ", Equity tolerance: " << eps << std::endl;
  }
  
  return verification_passed;
}

} // namespace sentio


```

## 📄 **FILE 48 of 60**: src/router.cpp

**File Information**:
- **Path**: `src/router.cpp`

- **Size**: 40 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .cpp

```text
#include "sentio/router.hpp"
#include "sentio/session_nyt.hpp"
#include <cmath>

namespace sentio {

// **MODIFIED**: Logic updated for the new SignalType enum.
std::optional<RouteDecision> route(const StrategySignal& s, const RouterCfg& cfg, const std::string& base_symbol) {
  
  if (s.signal == SignalType::NONE) {
      return std::nullopt;
  }
  
  auto timestamp = static_cast<int64_t>(s.metadata.count("timestamp") ? s.metadata.at("timestamp") : 0);
  if (cfg.require_rth && !pass_session_filter(true, timestamp)) {
    return std::nullopt;
  }
  
  double confidence = std::abs(s.confidence);
  if (confidence < cfg.min_signal_strength) {
      return std::nullopt;
  }
  
  double target_weight = 0.0;
  // Determine weight based on signal direction
  if (s.signal == SignalType::BUY || s.signal == SignalType::STRONG_BUY) {
      target_weight = std::min(confidence * cfg.signal_multiplier, cfg.max_position_pct);
  } else if (s.signal == SignalType::SELL || s.signal == SignalType::STRONG_SELL) {
      target_weight = -std::min(confidence * cfg.signal_multiplier, cfg.max_position_pct);
  }

  if (std::abs(target_weight) > 0) {
      // For now, we assume routing to the base symbol. This can be expanded.
      return RouteDecision(base_symbol, target_weight);
  }
  
  return std::nullopt;
}

} // namespace sentio
```

## 📄 **FILE 49 of 60**: src/runner.cpp

**File Information**:
- **Path**: `src/runner.cpp`

- **Size**: 113 lines
- **Modified**: 2025-09-05 09:41:27

- **Type**: .cpp

```text
#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/pricebook.hpp"
#include "sentio/metrics.hpp"
#include "sentio/sizer.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>

namespace sentio {

RunResult run_backtest([[maybe_unused]] Auditor& au, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                      int base_symbol_id, const RunnerCfg& cfg) {
    
    // 1. ============== INITIALIZATION ==============
    RunResult result{};
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
    for (size_t i = 0; i < base_series.size(); ++i) {
        const auto& bar = base_series[i];
        pricebook.sync_to_base_i(i);
        
        StrategySignal sig = strategy->calculate_signal(base_series, i);
        
        if (sig.signal != SignalType::NONE) {
            sig.metadata["timestamp"] = static_cast<double>(bar.ts_nyt_epoch);
            
            auto route_decision = route(sig, cfg.router, ST.get_symbol(base_symbol_id));

            if (route_decision) {
                int instrument_id = ST.get_id(route_decision->instrument);
                if (instrument_id != -1) {
                    double instrument_price = pricebook.last_px[instrument_id];

                    if (instrument_price > 0) {
                        [[maybe_unused]] double current_equity = equity_mark_to_market(portfolio, pricebook.last_px);
                        
                        // **MODIFIED**: This is the core logic fix.
                        // We calculate the desired final position size, then determine the needed trade quantity.
                        double target_qty = sizer.calculate_target_quantity(portfolio, ST, pricebook.last_px, 
                                                                             route_decision->instrument, route_decision->target_weight, 
                                                                             series[instrument_id], cfg.sizer);
                        
                        double current_qty = portfolio.positions[instrument_id].qty;
                        double trade_qty = target_qty - current_qty; // The actual amount to trade

                        if (std::abs(trade_qty * instrument_price) > 1.0) { // Min trade notional $1
                            apply_fill(portfolio, instrument_id, trade_qty, instrument_price);
                            total_fills++; // **CRITICAL FIX**: Increment the fills counter
                        } else {
                            no_qty_count++;
                        }
                    }
                }
            } else {
                no_route_count++;
            }
        }
        
        // 3. ============== SNAPSHOT ==============
        if (i % cfg.snapshot_stride == 0 || i == base_series.size() - 1) {
            double current_equity = equity_mark_to_market(portfolio, pricebook.last_px);
            equity_curve.emplace_back(bar.ts_utc, current_equity);
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

    return result;
}

} // namespace sentio
```

## 📄 **FILE 50 of 60**: src/strategy_bollinger_squeeze_breakout.cpp

**File Information**:
- **Path**: `src/strategy_bollinger_squeeze_breakout.cpp`

- **Size**: 150 lines
- **Modified**: 2025-09-05 12:12:54

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
            signal.signal = (state_ == State::Long) ? SignalType::SELL : SignalType::BUY;
            signal.reason = "Time stop exit";
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
            signal.signal = SignalType::BUY;
            signal.reason = "Bollinger Squeeze Breakout Long";
            state_ = State::Long;
        } else {
            signal.signal = SignalType::SELL;
            signal.reason = "Bollinger Squeeze Breakout Short";
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

## 📄 **FILE 51 of 60**: src/strategy_market_making.cpp

**File Information**:
- **Path**: `src/strategy_market_making.cpp`

- **Size**: 128 lines
- **Modified**: 2025-09-05 12:13:48

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
    // **MODIFIED**: Significantly lowered the volume ratio to allow participation
    return {
        {"base_spread", 0.001}, {"min_spread", 0.0005}, {"max_spread", 0.003},
        {"order_levels", 3.0}, {"level_spacing", 0.0005}, {"order_size_base", 0.5},
        {"max_inventory", 100.0}, {"inventory_skew_mult", 0.002},
        {"adverse_selection_threshold", 0.002}, {"volatility_window", 20.0},
        {"volume_window", 50.0}, {"min_volume_ratio", 0.1}, // Was 0.8, now 0.1, to prevent zerovol drops
        {"max_orders_per_bar", 10.0}, {"rebalance_frequency", 10.0}
    };
}

ParameterSpace MarketMakingStrategy::get_param_space() const { return {}; }

void MarketMakingStrategy::apply_params() {
    base_spread_ = params_["base_spread"];
    min_spread_ = params_["min_spread"];
    max_spread_ = params_["max_spread"];
    order_levels_ = static_cast<int>(params_["order_levels"]);
    level_spacing_ = params_["level_spacing"];
    order_size_base_ = params_["order_size_base"];
    max_inventory_ = params_["max_inventory"];
    inventory_skew_mult_ = params_["inventory_skew_mult"];
    adverse_selection_threshold_ = params_["adverse_selection_threshold"];
    min_volume_ratio_ = params_["min_volume_ratio"];
    max_orders_per_bar_ = static_cast<int>(params_["max_orders_per_bar"]);
    rebalance_frequency_ = static_cast<int>(params_["rebalance_frequency"]);

    int vol_window = 20;
    if (params_.count("volatility_window")) {
        vol_window = std::max(1, static_cast<int>(params_.at("volatility_window")));
    }
    int vol_mean_window = 50;
    if (params_.count("volume_window")) {
        vol_mean_window = std::max(1, static_cast<int>(params_.at("volume_window")));
    }
    
    rolling_returns_ = RollingMeanVar(vol_window);
    rolling_volume_ = RollingMean(vol_mean_window);
    reset_state();
}

void MarketMakingStrategy::reset_state() {
    BaseStrategy::reset_state();
    market_state_ = MarketState{};
    if (params_.count("volatility_window")) {
        rolling_returns_ = RollingMeanVar(std::max(1, static_cast<int>(params_.at("volatility_window"))));
    }
    if (params_.count("volume_window")) {
        rolling_volume_ = RollingMean(std::max(1, static_cast<int>(params_.at("volume_window"))));
    }
}

StrategySignal MarketMakingStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;
    
    if (current_index < static_cast<int>(params_["volume_window"])) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }

    double price_return = (current_index > 0) ? (bars[current_index].close - bars[current_index - 1].close) / bars[current_index - 1].close : 0.0;
    rolling_returns_.push(price_return);
    rolling_volume_.push(bars[current_index].volume);

    if (!should_participate(bars[current_index])) {
        return signal;
    }
    
    double inventory_skew = get_inventory_skew();
    
    if (inventory_skew > 0.001) {
        signal.signal = SignalType::BUY;
        signal.reason = "Market Making: Skewed Short";
    } else if (inventory_skew < -0.001) {
        signal.signal = SignalType::SELL;
        signal.reason = "Market Making: Skewed Long";
    } else {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }

    signal.confidence = 0.5 + std::abs(inventory_skew) / inventory_skew_mult_;
    diag_.emitted++;
    return signal;
}

bool MarketMakingStrategy::should_participate(const Bar& bar) {
    double volatility = std::sqrt(rolling_returns_.var());
    
    if (volatility > adverse_selection_threshold_) {
        diag_.drop(DropReason::THRESHOLD); // Re-using for high volatility
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

## 📄 **FILE 52 of 60**: src/strategy_momentum_volume.cpp

**File Information**:
- **Path**: `src/strategy_momentum_volume.cpp`

- **Size**: 149 lines
- **Modified**: 2025-09-05 12:24:48

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
        signal.signal = SignalType::BUY;
        signal.reason = "Bullish momentum breakout above value area";
    } else {
        signal.signal = SignalType::SELL;
        signal.reason = "Bearish momentum breakout below value area";
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

## 📄 **FILE 53 of 60**: src/strategy_opening_range_breakout.cpp

**File Information**:
- **Path**: `src/strategy_opening_range_breakout.cpp`

- **Size**: 137 lines
- **Modified**: 2025-09-05 04:14:41

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
    long current_day = bars[current_index].ts_nyt_epoch / SECONDS_IN_DAY;
    long prev_day = bars[current_index - 1].ts_nyt_epoch / SECONDS_IN_DAY;

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
        signal.signal = SignalType::BUY;
        signal.reason = "Breakout above opening range high";
        signal.suggested_stop_loss = current_range_.high; // Stop at the range high
        signal.suggested_take_profit = bar.close * (1.0 + take_profit_pct_);
    } else { // is_breakout_down
        signal.signal = SignalType::SELL;
        signal.reason = "Breakout below opening range low";
        signal.suggested_stop_loss = current_range_.low; // Stop at the range low
        signal.suggested_take_profit = bar.close * (1.0 - take_profit_pct_);
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

## 📄 **FILE 54 of 60**: src/strategy_order_flow_imbalance.cpp

**File Information**:
- **Path**: `src/strategy_order_flow_imbalance.cpp`

- **Size**: 108 lines
- **Modified**: 2025-09-05 12:30:04

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
            signal.signal = SignalType::BUY;
            signal.reason = "High buying pressure detected";
            ofi_state_ = OFIState::Long;
            // **FIXED**: Correctly access the 'state_' member from BaseStrategy
            state_.last_trade_bar = current_index;
        } else if (avg_pressure < entry_threshold_short_) {
            signal.signal = SignalType::SELL;
            signal.reason = "High selling pressure detected";
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
            signal.signal = (ofi_state_ == OFIState::Long) ? SignalType::SELL : SignalType::BUY;
            signal.reason = "Time stop exit";
            diag_.emitted++;
            reset_state();
        }
    }

    return signal;
}

REGISTER_STRATEGY(OrderFlowImbalanceStrategy, "OrderFlowImbalance");

} // namespace sentio


```

## 📄 **FILE 55 of 60**: src/strategy_order_flow_scalping.cpp

**File Information**:
- **Path**: `src/strategy_order_flow_scalping.cpp`

- **Size**: 123 lines
- **Modified**: 2025-09-05 13:11:07

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
        {"imbalance_threshold", 0.65}, // Was 0.7, now arms when avg pressure is > 65%
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
            signal.signal = (of_state_ == OFState::Long) ? SignalType::SELL : SignalType::BUY;
            signal.reason = "Time stop exit";
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
                signal.signal = SignalType::BUY;
                signal.reason = "Bullish pressure confirmation";
                of_state_ = OFState::Long;
            } else { // Failed confirmation
                of_state_ = OFState::Idle;
                diag_.drop(DropReason::THRESHOLD);
            }
            break;

        case OFState::ArmedShort:
            if (pressure < 0.5) { // Confirmation bar must be bearish
                signal.signal = SignalType::SELL;
                signal.reason = "Bearish pressure confirmation";
                of_state_ = OFState::Short;
            } else { // Failed confirmation
                of_state_ = OFState::Idle;
                diag_.drop(DropReason::THRESHOLD);
            }
            break;
        default: break;
    }
    
    if (signal.signal != SignalType::NONE) {
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

## 📄 **FILE 56 of 60**: src/strategy_volatility_expansion.cpp

**File Information**:
- **Path**: `src/strategy_volatility_expansion.cpp`

- **Size**: 128 lines
- **Modified**: 2025-09-05 12:26:05

- **Type**: .cpp

```text
#include "sentio/strategy_volatility_expansion.hpp"
#include "sentio/session_nyt.hpp"
#include <algorithm>
#include <cmath>

namespace sentio {

VolatilityExpansionStrategy::VolatilityExpansionStrategy() 
    : BaseStrategy("VolatilityExpansion"),
      rolling_hh_(20),
      rolling_ll_(20) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap VolatilityExpansionStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed breakout_k and disabled require_rth by default to fix session drops.
    return {
        {"atr_window", 14.0},
        {"lookback_hh", 20.0},
        {"lookback_ll", 20.0},
        {"breakout_k", 0.75},    // Was 1.0, making breakouts easier to achieve
        {"hold_max_bars", 160.0},
        {"tp_atr_mult", 1.5},
        {"sl_atr_mult", 1.0},
        {"require_rth", 0.0}     // Was 1.0 (true), now 0.0 (false) to prevent session drops
    };
}

ParameterSpace VolatilityExpansionStrategy::get_param_space() const { return {}; }

void VolatilityExpansionStrategy::apply_params() {
    atr_window_ = static_cast<int>(params_["atr_window"]);
    atr_alpha_ = 2.0 / (atr_window_ + 1.0);
    lookback_hh_ = static_cast<int>(params_["lookback_hh"]);
    lookback_ll_ = static_cast<int>(params_["lookback_ll"]);
    breakout_k_ = params_["breakout_k"];
    hold_max_bars_ = static_cast<int>(params_["hold_max_bars"]);
    tp_atr_mult_ = params_["tp_atr_mult"];
    sl_atr_mult_ = params_["sl_atr_mult"];
    require_rth_ = params_["require_rth"] > 0.5;

    rolling_hh_ = RollingHHLL(lookback_hh_);
    rolling_ll_ = RollingHHLL(lookback_ll_);
    reset_state();
}

void VolatilityExpansionStrategy::reset_state() {
    BaseStrategy::reset_state();
    state_ = VEState::Flat;
    bars_in_trade_ = 0;
    atr_ = 0.0;
    prev_close_ = 0.0;
}

StrategySignal VolatilityExpansionStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;
    const int min_bars = std::max({lookback_hh_, lookback_ll_, atr_window_});
    
    if (prev_close_ == 0.0 && current_index > 0) {
         prev_close_ = bars[current_index - 1].close;
    }

    if (current_index < min_bars) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }

    if (require_rth_ && !pass_session_filter(true, bars[current_index].ts_nyt_epoch)) {
        diag_.drop(DropReason::SESSION);
        return signal;
    }
    
    const auto& bar = bars[current_index];
    double tr = true_range(bar.high, bar.low, prev_close_);
    
    if (atr_ == 0.0) { // Initialize ATR
        double sum_tr = 0.0;
        for (int i = 0; i < atr_window_; ++i) {
            double pc = (current_index - i > 0) ? bars[current_index - i - 1].close : bars[current_index - i].open;
            sum_tr += true_range(bars[current_index-i].high, bars[current_index-i].low, pc);
        }
        atr_ = sum_tr / atr_window_;
    } else { // Rolling ATR
        atr_ = (tr * atr_alpha_) + (atr_ * (1.0 - atr_alpha_));
    }

    auto [hh, _] = rolling_hh_.push(bar.high, bar.low);
    auto [__, ll] = rolling_ll_.push(bar.high, bar.low);
    
    prev_close_ = bar.close; // Update for next iteration

    if (state_ == VEState::Flat) {
        const double up_trigger = hh + breakout_k_ * atr_;
        const double dn_trigger = ll - breakout_k_ * atr_;

        if (bar.close > up_trigger) {
            signal.signal = SignalType::BUY;
            signal.reason = "Breakout above high + k*ATR";
            state_ = VEState::Long;
        } else if (bar.close < dn_trigger) {
            signal.signal = SignalType::SELL;
            signal.reason = "Breakout below low - k*ATR";
            state_ = VEState::Short;
        } else {
            diag_.drop(DropReason::THRESHOLD);
            return signal;
        }

        signal.confidence = 0.8;
        diag_.emitted++;
        bars_in_trade_ = 0;
    } else {
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            signal.signal = (state_ == VEState::Long) ? SignalType::SELL : SignalType::BUY;
            signal.reason = "Time stop exit";
            diag_.emitted++;
            reset_state();
        }
    }
    
    return signal;
}

REGISTER_STRATEGY(VolatilityExpansionStrategy, "VolatilityExpansion");

} // namespace sentio

```

## 📄 **FILE 57 of 60**: src/strategy_vwap_reversion.cpp

**File Information**:
- **Path**: `src/strategy_vwap_reversion.cpp`

- **Size**: 162 lines
- **Modified**: 2025-09-05 04:14:23

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
        signal.signal = SignalType::BUY;
        signal.reason = "Price below lower VWAP band + RSI confirmation";
        signal.suggested_stop_loss = bar.close * (1.0 - stop_loss_pct_);
        signal.suggested_take_profit = vwap_; // Target the VWAP
    } else if (sell_condition) {
        signal.signal = SignalType::SELL;
        signal.reason = "Price above upper VWAP band + RSI confirmation";
        signal.suggested_stop_loss = bar.close * (1.0 + stop_loss_pct_);
        signal.suggested_take_profit = vwap_; // Target the VWAP
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

## 📄 **FILE 58 of 60**: src/wf.cpp

**File Information**:
- **Path**: `src/wf.cpp`

- **Size**: 607 lines
- **Modified**: 2025-09-05 09:41:27

- **Type**: .cpp

```text
// wf.cpp
// Walk-Forward driver with safe training loop, always-OOS policy, and robust windowing.
// Drop-in: requires C++20. Plug your own hooks where marked TODO.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

// -------------------------------
// Minimal date helper (day granularity)
// Replace with your own if you already have Date utilities.
// -------------------------------
struct Date {
  // days since epoch (UTC or NYT midnight—consistent across your dataset)
  int64_t days{};
  friend bool operator<(const Date& a, const Date& b){ return a.days < b.days; }
  friend bool operator<=(const Date& a, const Date& b){ return a.days <= b.days; }
  friend bool operator>(const Date& a, const Date& b){ return a.days > b.days; }
  friend bool operator>=(const Date& a, const Date& b){ return a.days >= b.days; }
  friend bool operator==(const Date& a, const Date& b){ return a.days == b.days; }
};
inline Date operator+(Date d, int add_days){ d.days += add_days; return d; }
inline Date days(int n){ return Date{n}; } // convenience

// -------------------------------
// WF config and data slice types
// -------------------------------
struct WfCfg {
  int train_days = 252 * 2; // ~2y
  int oos_days   = 63;      // ~1 quarter
  int step_days  = 63;      // slide by a quarter
};

struct DataSlice {
  // Your implementation should carry pointers/indices into your bar arrays.
  // For this template, we just keep [start,end) in "days since epoch".
  Date start{};
  Date end{};
  bool has_data() const { return start < end; }
};

// -------------------------------
// Params & hashing (replace with your strategy params)
// -------------------------------
struct Params {
  // Example parameters — replace with your actual fields.
  int   short_win = 20;
  int   long_win  = 100;
  double stop_bps = 50.0;
  friend bool operator==(const Params& a, const Params& b){
    return a.short_win==b.short_win && a.long_win==b.long_win && std::fabs(a.stop_bps-b.stop_bps)<1e-12;
  }
};

struct ParamsHash {
  size_t operator()(const Params& p) const noexcept {
    // simple FNV-1a style
    auto mix = [](size_t h, size_t v){ return (h ^ v) * 1099511628211ULL; };
    size_t h = 1469598103934665603ULL;
    h = mix(h, std::hash<int>{}(p.short_win));
    h = mix(h, std::hash<int>{}(p.long_win));
    h = mix(h, std::hash<long long>{}((long long)std::llround(p.stop_bps*10000)));
    return h;
  }
};

// -------------------------------
// WF window builder (bulletproof)
// -------------------------------
inline void validate_wf_cfg(const WfCfg& w){
  if (w.train_days <= 0) throw std::invalid_argument("train_days must be > 0");
  if (w.oos_days   <= 0) throw std::invalid_argument("oos_days must be > 0");
  if (w.step_days  <= 0) throw std::invalid_argument("step_days must be > 0");
}

inline std::vector<std::tuple<Date,Date,Date>>  // {train_start, train_end, oos_end}
make_wf_windows(Date start, Date end, const WfCfg& w){
  validate_wf_cfg(w);
  std::vector<std::tuple<Date,Date,Date>> out;
  if (!(start < end)) return out;

  Date cur = start;
  const int step = std::max(1, w.step_days);

  for (size_t guard=0; cur < end; ++guard){
    auto train_end = cur + w.train_days;
    auto oos_end   = train_end + w.oos_days;
    if (!(train_end < end) || !(oos_end <= end)) break;

    out.emplace_back(cur, train_end, oos_end);

    Date next = cur + step;
    if (!(next > cur)) throw std::runtime_error("WF window builder: next <= cur (no progress)");
    cur = next;

    if (guard > 10'000'000ULL) throw std::runtime_error("WF window builder: guard tripped");
  }
  return out;
}

// -------------------------------
// Training criteria & result
// -------------------------------
struct TrainCriteria {
  int    max_iters      = 200;    // absolute upper bound
  int    patience       = 20;     // stop after this many non-improving iters
  double min_delta      = 1e-4;   // min improvement to reset patience
  double target_metric  = 0.0;    // stop early if >= target (0 disables)
  int64_t time_budget_ms = 0;     // wall-clock cap (0 disables)
};

struct TrainResult {
  bool   success = false;   // met target or improved at least once
  int    iters   = 0;
  double best_metric = -1e300;
  Params best_params;
};

// -------------------------------
// Progress guard (Debug use)
// -------------------------------
struct ProgressGuard {
  uint64_t same_count = 0, limit = 1000000ULL;
  uint64_t last_token = 0;
  void step(uint64_t token){
    if (token == last_token) { if (++same_count > limit) throw std::runtime_error("ProgressGuard: no progress"); }
    else { last_token = token; same_count = 0; }
  }
};

// -------------------------------
// Safe training loop
// - evaluate: Params -> metric (higher is better)
// - propose_next: (last_params, last_metric) -> new Params (should usually differ)
// -------------------------------
inline TrainResult train_until(
    const TrainCriteria& cfg,
    Params init,
    const std::function<double(const Params&)>& evaluate,
    const std::function<Params(const Params&, double)>& propose_next)
{
  using clock = std::chrono::high_resolution_clock;
  const auto t0 = clock::now();

  TrainResult out;
  out.best_params = init;

  int no_improve = 0;
  int iter = 0;

  std::unordered_set<Params, ParamsHash> seen; seen.reserve(cfg.max_iters*2);
  ProgressGuard guard;

  Params cur = init;
  double cur_metric = evaluate(cur);
  out.best_metric = cur_metric;
  bool improved_once = false;

  seen.insert(cur);
  guard.step(ParamsHash{}(cur));

  if (cfg.target_metric > 0.0 && cur_metric >= cfg.target_metric) {
    out.success = true; out.iters = 1; return out;
  }

  for (iter = 2; iter <= cfg.max_iters; ++iter) {
    // Time budget check
    if (cfg.time_budget_ms > 0) {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
      if (ms >= cfg.time_budget_ms) break;
    }

    // Propose new params
    Params next = propose_next(cur, cur_metric);

    // If duplicate, try one more proposal; if still dup → accept best-so-far
    if (!seen.insert(next).second) {
      next = propose_next(next, cur_metric);
      if (!seen.insert(next).second) break;
    }

    double next_metric = evaluate(next);

    // Improvement logic
    if (next_metric >= out.best_metric + cfg.min_delta) {
      out.best_metric = next_metric;
      out.best_params = next;
      improved_once = true;
      no_improve = 0;
    } else {
      if (++no_improve >= cfg.patience) break;
    }

    if (cfg.target_metric > 0.0 && out.best_metric >= cfg.target_metric) break;

    // Guard against “no progress”
    guard.step(ParamsHash{}(next) ^ (uint64_t)iter);

    cur = next;
    cur_metric = next_metric;
  }

  out.iters = iter - 1;
  out.success = improved_once || (cfg.target_metric > 0.0 && out.best_metric >= cfg.target_metric);
  return out;
}

// -------------------------------
// OOS result container
// -------------------------------
struct OOSResult {
  bool   valid  = true;
  double metric = std::nan("");   // e.g., Sharpe/return
  std::string reason = "ok";
};

// -------------------------------
// Fold outcome (training + OOS); OOS always runs unless invalid data/params
// -------------------------------
struct FoldOutcome {
  bool training_success = false;
  int  train_iters      = 0;
  double train_metric_best = -1e300;

  Params params_used;
  bool   params_from_prev = false;

  bool   oos_valid = false;
  double oos_metric = std::nan("");
  std::string oos_reason = "uninit";
};

// -------------------------------
// Run a single WF fold
// Hooks you must provide via std::function (plug at call site):
//   train_metric(train_slice, params) -> double
//   run_oos(oos_slice, params) -> OOSResult
//   propose(last_params, last_metric) -> Params
//   params_valid(params) -> bool
//   baseline_params() -> Params
// -------------------------------
inline FoldOutcome run_wf_fold(
    const DataSlice& train, const DataSlice& oos,
    const Params& init, const TrainCriteria& crit,
    const std::optional<Params>& prev_fold_params,

    const std::function<double(const DataSlice&, const Params&)>& train_metric,
    const std::function<OOSResult(const DataSlice&, const Params&)>& run_oos,
    const std::function<Params(const Params&, double)>& propose,
    const std::function<bool(const Params&)>& params_valid,
    const std::function<Params(void)>& baseline_params
){
  FoldOutcome R{};

  auto eval_train = [&](const Params& p)->double { return train_metric(train, p); };
  auto tr = train_until(crit, init, eval_train, propose);

  R.training_success   = tr.success;
  R.train_iters        = tr.iters;
  R.train_metric_best  = tr.best_metric;
  R.params_used        = tr.best_params;

  // Fallback params if training didn't meet criteria
  if (!R.training_success) {
    if (prev_fold_params) {
      R.params_used      = *prev_fold_params;
      R.params_from_prev = true;
    } else if (!params_valid(R.params_used)) {
      R.params_used = baseline_params();
    }
  }

  // OOS always runs unless we truly cannot evaluate
  if (!oos.has_data()) {
    R.oos_valid = false;
    R.oos_reason = "no_data";
    return R;
  }
  if (!params_valid(R.params_used)) {
    R.oos_valid = false;
    R.oos_reason = "invalid_params";
    return R;
  }

  auto res = run_oos(oos, R.params_used);
  R.oos_valid  = res.valid;
  R.oos_metric = res.metric;
  R.oos_reason = res.reason.empty()? "ok" : res.reason;

  return R;
}

// -------------------------------
// Orchestrate full WF run (optionally parallel across folds)
// -------------------------------
struct WfRunSummary {
  int folds = 0;
  int folds_ok = 0;
  double oos_metric_sum = 0.0;
  std::vector<FoldOutcome> outcomes;
};

inline WfRunSummary run_walk_forward(
    Date global_start, Date global_end, const WfCfg& cfg,
    const Params& init, const TrainCriteria& crit,
    const std::function<DataSlice(Date,Date)>& get_slice, // build slices from dates
    const std::function<double(const DataSlice&, const Params&)>& train_metric,
    const std::function<OOSResult(const DataSlice&, const Params&)>& run_oos,
    const std::function<Params(const Params&, double)>& propose,
    const std::function<bool(const Params&)>& params_valid,
    const std::function<Params(void)>& baseline_params,
    int max_concurrency = std::max(1u, std::thread::hardware_concurrency()-1)
){
  auto windows = make_wf_windows(global_start, global_end, cfg);
  const int K = (int)windows.size();
  WfRunSummary sum; sum.folds = K; sum.outcomes.resize(K);

  std::mutex m_prev;
  std::optional<Params> prev_params{};

  // simple async throttle
  std::deque<std::pair<int, std::future<FoldOutcome>>> q;

  auto submit = [&](int idx){
    auto [ts, tr_end, oos_end] = windows[idx];
    DataSlice train{ts, tr_end};
    DataSlice oos  {tr_end, oos_end};

    Params init_for_fold = init;
    {
      std::lock_guard<std::mutex> lk(m_prev);
      if (prev_params) init_for_fold = *prev_params; // warm start with last
    }

    return std::async(std::launch::async, [=, &get_slice, &train_metric, &run_oos, &propose, &params_valid, &baseline_params]{
      // materialize slices (or pass views) from your data store
      auto train_slice = get_slice(train.start, train.end);
      auto oos_slice   = get_slice(oos.start, oos.end);

      auto out = run_wf_fold(train_slice, oos_slice, init_for_fold, crit, prev_params,
                             train_metric, run_oos, propose, params_valid, baseline_params);
      return out;
    });
  };

  auto drain_one = [&](){
    auto idx = q.front().first;
    auto& fut = q.front().second;
    auto out = fut.get();
    sum.outcomes[idx] = out;
    if (out.oos_valid) { sum.folds_ok++; sum.oos_metric_sum += out.oos_metric; }
    // record last successful params as warm start
    if (out.training_success) {
      std::lock_guard<std::mutex> lk(m_prev);
      prev_params = out.params_used;
    }
    q.pop_front();
  };

  for (int i=0; i<K; ++i) {
    q.emplace_back(i, submit(i));
    if ((int)q.size() >= max_concurrency) {
      q.front().second.wait();
      drain_one();
    }
  }
  while (!q.empty()) {
    q.front().second.wait();
    drain_one();
  }

  return sum;
}

// -------------------------------
// Example usage (wire your hooks here)
// Remove main() and integrate into your app as needed.
// -------------------------------
#ifdef WF_EXAMPLE_MAIN
// Mock hooks (replace with real ones)
static double mock_train_metric(const DataSlice& s, const Params& p){
  // pretend Sharpe improves with longer windows up to a point
  double len = double(s.end.days - s.start.days);
  double score = (p.long_win > p.short_win ? 1.0 : -1.0) * std::tanh(len/500.0) - std::abs(p.stop_bps-50.0)/200.0;
  return score;
}
static OOSResult mock_run_oos(const DataSlice& s, const Params& p){
  (void)p;
  if (!s.has_data()) return {false, std::nan(""), "no_data"};
  double len = double(s.end.days - s.start.days);
  return {true, 0.5 * std::tanh(len/200.0), "ok"};
}
static Params mock_propose(const Params& last, double){
  Params n = last;
  // simple random-ish walk (deterministic tweak here)
  n.short_win = std::max(5, std::min(60, n.short_win + ((n.short_win%2)? +1 : -1)));
  n.long_win  = std::max(20, std::min(240, n.long_win  + ((n.long_win%3)? +2 : -2)));
  n.stop_bps  = std::clamp(n.stop_bps + ((n.stop_bps>50)? -5.0 : +5.0), 5.0, 200.0);
  return n;
}
static bool mock_params_valid(const Params& p){
  return p.short_win > 0 && p.long_win > p.short_win && p.stop_bps > 0.0;
}
static Params mock_baseline(){ return Params{20, 100, 50.0}; }
static DataSlice mock_get_slice(Date a, Date b){ return DataSlice{a,b}; }

int main(){
  WfCfg cfg; cfg.train_days=240; cfg.oos_days=60; cfg.step_days=60;
  TrainCriteria crit; crit.max_iters=60; crit.patience=10; crit.min_delta=1e-3; crit.target_metric=0.0;
  Params init{20,100,50.0};

  auto sum = run_walk_forward(
    /*global_start*/ Date{0}, /*global_end*/ Date{2000}, cfg, init, crit,
    mock_get_slice,
    mock_train_metric,
    mock_run_oos,
    mock_propose,
    mock_params_valid,
    mock_baseline,
    /*max_concurrency*/ 2
  );

  std::cout << "WF folds=" << sum.folds
            << " ok=" << sum.folds_ok
            << " avg_oos=" << (sum.folds_ok? sum.oos_metric_sum / sum.folds_ok : 0.0)
            << "\n";
  return 0;
}
#endif

// -------------------------------
// Bridge functions to match existing API
// -------------------------------
#include "sentio/wf.hpp"
#include "sentio/runner.hpp"
#include "sentio/audit.hpp"
#include "sentio/metrics.hpp"
#include <iostream>
#include <algorithm>
#include <vector>

namespace sentio {

// Convert bar index to date (simplified)
Date bar_index_to_date(int index) {
    // Assuming 390 bars per day (6.5 hours * 60 minutes)
    return Date{index / 390};
}

// Convert date to bar index (simplified)
int date_to_bar_index(Date date) {
    return date.days * 390;
}

// Bridge function to match existing API
Gate run_wf_and_gate(Auditor& au_template,
                     const SymbolTable& ST,
                     const std::vector<std::vector<Bar>>& series,
                     int base_symbol_id,
                     const RunnerCfg& rcfg,
                     const WfCfg& wcfg) {
    
    Gate gate;
    
    // Validate step_days to prevent infinite loop
    if (wcfg.step_days <= 0) {
        std::cerr << "ERROR: Walk-forward step_days must be positive. Got: "
                  << wcfg.step_days << std::endl;
        gate.recommend = "REJECT - INVALID CONFIG";
        return gate;
    }
    
    const auto& base_series = series[base_symbol_id];
    const int total_bars = (int)base_series.size();
    if (total_bars < wcfg.train_days + wcfg.oos_days) {
        std::cerr << "Insufficient data for walk-forward test" << std::endl;
        return gate;
    }
    
    std::vector<double> oos_returns;
    [[maybe_unused]] int successful_folds = 0;
    
    // Walk-forward testing
    int total_iterations = (total_bars - wcfg.train_days - wcfg.oos_days) / wcfg.step_days;
    int current_iteration = 0;
    
    std::cerr << "Starting WF test with " << total_iterations << " iterations..." << std::endl;
    
    for (int start_idx = 0; start_idx < total_bars - wcfg.train_days - wcfg.oos_days; start_idx += wcfg.step_days) {
        current_iteration++;
        if (current_iteration % 10 == 0) {
            std::cerr << "Progress: " << current_iteration << "/" << total_iterations << " iterations completed" << std::endl;
        }
        
        int train_end = start_idx + wcfg.train_days;
        int oos_start = train_end;
        int oos_end = std::min(oos_start + wcfg.oos_days, total_bars);
        
        if (oos_end - oos_start < 5) continue; // Skip if insufficient OOS data
        
        // Create training data slice
        std::vector<std::vector<Bar>> train_series = series;
        for (auto& sym_series : train_series) {
            if (sym_series.size() > static_cast<size_t>(train_end)) {
                sym_series.resize(train_end);
            }
        }
        
        // Create OOS data slice
        std::vector<std::vector<Bar>> oos_series = series;
        for (auto& sym_series : oos_series) {
            if (sym_series.size() > static_cast<size_t>(oos_end)) {
                sym_series.erase(sym_series.begin(), sym_series.begin() + oos_start);
                sym_series.resize(oos_end - oos_start);
            }
        }
        
        // Run training backtest
        if (current_iteration % 50 == 0) {
            std::cerr << "Running training backtest for iteration " << current_iteration << "..." << std::endl;
        }
        
        Auditor au_train = au_template;
        au_train.start_run("wf_train", rcfg.strategy_name, "{}", "NA", 42, "fold=" + std::to_string(start_idx));
        
        auto train_result = run_backtest(au_train, ST, train_series, base_symbol_id, rcfg);
        
        // Always run OOS backtest (removed gate mechanism)
        Auditor au_oos = au_template;
        au_oos.start_run("wf_oos", rcfg.strategy_name, "{}", "NA", 42, "fold=" + std::to_string(start_idx));
        
        auto oos_result = run_backtest(au_oos, ST, oos_series, base_symbol_id, rcfg);
        
        // Store OOS results
        oos_returns.push_back(oos_result.total_return);
        successful_folds++;
        
        // Track if any training period was profitable (for WF pass criteria)
        if (train_result.sharpe_ratio > 0.1 && train_result.total_return > -0.1) {
            gate.wf_pass = true;
        }
    }
    
    // Calculate OOS statistics
    if (!oos_returns.empty()) {
        double avg_return = 0.0;
        for (double ret : oos_returns) avg_return += ret;
        avg_return /= oos_returns.size();
        
        gate.oos_monthly_avg_return = avg_return * 12.0; // Annualized
        gate.oos_pass = gate.oos_monthly_avg_return > 0.0;
        
        // Calculate Sharpe ratio (simplified)
        double variance = 0.0;
        for (double ret : oos_returns) {
            double diff = ret - avg_return;
            variance += diff * diff;
        }
        variance /= oos_returns.size();
        double std_dev = std::sqrt(variance);
        
        if (std_dev > 0.0) {
            gate.oos_sharpe = avg_return / std_dev * std::sqrt(252.0);
        }
        
        // Determine recommendation
        if (gate.wf_pass && gate.oos_pass) {
            gate.recommend = "PAPER";
        } else if (gate.wf_pass) {
            gate.recommend = "ITERATE";
        } else {
            gate.recommend = "REJECT";
        }
    }
    
    return gate;
}

// Walk-forward testing with optimization (simplified for now)
Gate run_wf_and_gate_optimized(Auditor& au_template,
                               const SymbolTable& ST,
                               const std::vector<std::vector<Bar>>& series,
                               int base_symbol_id,
                               const RunnerCfg& base_rcfg,
                               const WfCfg& wcfg) {
    // For now, just run the basic WF without optimization
    // TODO: Implement parameter optimization
    return run_wf_and_gate(au_template, ST, series, base_symbol_id, base_rcfg, wcfg);
}

} // namespace sentio
```

## 📄 **FILE 59 of 60**: tools/create_mega_document.py

**File Information**:
- **Path**: `tools/create_mega_document.py`

- **Size**: 104 lines
- **Modified**: 2025-09-05 03:58:29

- **Type**: .py

```text
#!/usr/bin/env python3
"""
Create mega document from source files.
"""

import os
import argparse
import datetime
from pathlib import Path

def create_mega_document(directories, title, description, output, include_bug_report=False, bug_report_file=None):
    """Create a mega document from source files."""
    
    print(f"🔧 Creating mega document: {output}")
    print(f"📁 Source directory: {os.getcwd()}")
    print(f"📁 Output file: {output}")
    
    # Collect all files
    all_files = []
    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.hpp', '.cpp', '.h', '.c', '.py', '.md', '.txt', '.cmake', 'CMakeLists.txt')):
                        file_path = os.path.join(root, file)
                        all_files.append(file_path)
    
    print(f"📁 Files to include: {len(all_files)}")
    
    # Sort files for consistent ordering
    all_files.sort()
    
    # Create mega document
    with open(output, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Source Directory**: {os.getcwd()}\n")
        f.write(f"**Description**: {description}\n\n")
        f.write(f"**Total Files**: {len(all_files)}\n\n")
        f.write("---\n\n")
        
        # Include bug report if requested
        if include_bug_report and bug_report_file and os.path.exists(bug_report_file):
            f.write("## 🐛 **BUG REPORT**\n\n")
            with open(bug_report_file, 'r', encoding='utf-8') as bug_f:
                f.write(bug_f.read())
            f.write("\n\n---\n\n")
        
        # Table of contents
        f.write("## 📋 **TABLE OF CONTENTS**\n\n")
        for i, file_path in enumerate(all_files, 1):
            f.write(f"{i}. [{file_path}](#file-{i})\n")
        f.write("\n---\n\n")
        
        # File contents
        for i, file_path in enumerate(all_files, 1):
            try:
                with open(file_path, 'r', encoding='utf-8') as file_f:
                    content = file_f.read()
                
                f.write(f"## 📄 **FILE {i} of {len(all_files)}**: {file_path}\n\n")
                f.write("**File Information**:\n")
                f.write(f"- **Path**: `{file_path}`\n\n")
                f.write(f"- **Size**: {len(content.splitlines())} lines\n")
                f.write(f"- **Modified**: {datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"- **Type**: {Path(file_path).suffix}\n\n")
                f.write("```text\n")
                f.write(content)
                f.write("\n```\n\n")
                
                print(f"📄 Processing file {i}/{len(all_files)}: {file_path}")
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                f.write(f"## 📄 **FILE {i} of {len(all_files)}**: {file_path}\n\n")
                f.write(f"**Error**: Could not read file - {e}\n\n")
    
    print(f"✅ Mega document created: {output}")
    print(f"📊 Output size: {os.path.getsize(output) / 1024:.1f} KB")
    print(f"📊 Files processed: {len(all_files)}/{len(all_files)}")
    print(f"📊 Content size: {sum(os.path.getsize(f) for f in all_files if os.path.exists(f)) / 1024:.1f} KB")
    print(f"\n🎯 Success! Mega document created:")
    print(f"{output}")
    print(f"\n📁 Location: {os.path.abspath(output)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mega document from source files")
    parser.add_argument("--directories", "-d", nargs="+", required=True, help="Directories to include")
    parser.add_argument("--title", "-t", required=True, help="Document title")
    parser.add_argument("--description", "-desc", required=True, help="Document description")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--include-bug-report", action="store_true", help="Include bug report")
    parser.add_argument("--bug-report-file", help="Bug report file path")
    
    args = parser.parse_args()
    
    create_mega_document(
        args.directories,
        args.title,
        args.description,
        args.output,
        args.include_bug_report,
        args.bug_report_file
    )

```

## 📄 **FILE 60 of 60**: tools/data_downloader.py

**File Information**:
- **Path**: `tools/data_downloader.py`

- **Size**: 205 lines
- **Modified**: 2025-09-05 12:56:12

- **Type**: .py

```text
import os
import argparse
import requests
import pandas as pd
import pandas_market_calendars as mcal
import struct
from datetime import datetime
from pathlib import Path

# --- Constants ---
# Define the Regular Trading Hours for NYSE in New York time.
RTH_START = "09:30"
RTH_END = "16:00"
NY_TIMEZONE = "America/New_York"
POLYGON_API_BASE = "https://api.polygon.io"

def fetch_aggs_all(symbol, start_date, end_date, api_key, timespan="minute", multiplier=1):
    """
    Fetches all aggregate bars for a symbol within a date range from Polygon.io.
    Handles API pagination automatically.
    """
    print(f"Fetching '{symbol}' data from {start_date} to {end_date}...")
    url = (
        f"{POLYGON_API_BASE}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/"
        f"{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000"
    )
    
    headers = {"Authorization": f"Bearer {api_key}"}
    all_bars = []
    
    while url:
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            if "results" in data:
                all_bars.extend(data["results"])
                print(f" -> Fetched {len(data['results'])} bars...", end="\r")

            url = data.get("next_url")

        except requests.exceptions.RequestException as e:
            print(f"\nAPI Error fetching data for {symbol}: {e}")
            return None
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            return None
            
    print(f"\n -> Total bars fetched for {symbol}: {len(all_bars)}")
    if not all_bars:
        return None
        
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(all_bars)
    df.rename(columns={
        't': 'timestamp_utc_ms',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume'
    }, inplace=True)
    return df

def filter_and_prepare_data(df):
    """
    Filters a DataFrame of market data for RTH (Regular Trading Hours)
    and removes US market holidays.
    """
    if df is None or df.empty:
        return None

    print("Filtering data for RTH and US market holidays...")
    
    # 1. Convert UTC millisecond timestamp to a timezone-aware DatetimeIndex
    df['timestamp_utc_ms'] = pd.to_datetime(df['timestamp_utc_ms'], unit='ms', utc=True)
    df.set_index('timestamp_utc_ms', inplace=True)
    
    # 2. Convert the index to New York time to perform RTH and holiday checks
    df.index = df.index.tz_convert(NY_TIMEZONE)
    
    # 3. Filter for Regular Trading Hours
    df = df.between_time(RTH_START, RTH_END)

    # 4. Filter out US market holidays
    nyse = mcal.get_calendar('NYSE')
    holidays = nyse.holidays().holidays # Get a list of holiday dates
    df = df[~df.index.normalize().isin(holidays)]
    
    print(f" -> {len(df)} bars remaining after filtering.")
    
    # 5. Add the specific columns required by the C++ backtester
    df['ts_utc'] = df.index.strftime('%Y-%m-%dT%H:%M:%S%z').str.replace(r'([+-])(\d{2})(\d{2})', r'\1\2:\3', regex=True)
    df['ts_nyt_epoch'] = df.index.astype('int64') // 10**9
    
    return df

def save_to_bin(df, path):
    """
    Saves the DataFrame to a custom binary format compatible with the C++ backtester.
    Format:
    - uint64_t: Number of bars
    - For each bar:
      - uint32_t: Length of ts_utc string
      - char[]: ts_utc string data
      - int64_t: ts_nyt_epoch
      - double: open, high, low, close
      - uint64_t: volume
    """
    print(f"Saving to binary format at {path}...")
    try:
        with open(path, 'wb') as f:
            # Write total number of bars
            num_bars = len(df)
            f.write(struct.pack('<Q', num_bars))

            # **FIXED**: The struct format string now correctly includes six format
            # specifiers to match the six arguments passed to pack().
            # q: int64_t (ts_nyt_epoch)
            # d: double (open)
            # d: double (high)
            # d: double (low)
            # d: double (close)
            # Q: uint64_t (volume)
            bar_struct = struct.Struct('<qddddQ')

            for row in df.itertuples():
                # Handle the variable-length string part
                ts_utc_bytes = row.ts_utc.encode('utf-8')
                f.write(struct.pack('<I', len(ts_utc_bytes)))
                f.write(ts_utc_bytes)
                
                # Pack and write the fixed-size data
                packed_data = bar_struct.pack(
                    row.ts_nyt_epoch,
                    row.open,
                    row.high,
                    row.low,
                    row.close,
                    int(row.volume) # C++ expects uint64_t, so we cast to int
                )
                f.write(packed_data)
        print(" -> Binary file saved successfully.")
    except Exception as e:
        print(f"Error saving binary file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Polygon.io Data Downloader and Processor")
    parser.add_argument('symbols', nargs='+', help="One or more stock symbols (e.g., QQQ TQQQ SQQQ)")
    parser.add_argument('--start', required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument('--end', required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument('--outdir', default='data', help="Output directory for CSV and BIN files")
    parser.add_argument('--timespan', default='minute', choices=['minute', 'hour', 'day'], help="Timespan of bars")
    parser.add_argument('--multiplier', default=1, type=int, help="Multiplier for the timespan")
    
    args = parser.parse_args()
    
    # Get API key from environment variable for security
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("Error: POLYGON_API_KEY environment variable not set.")
        return
        
    # Create output directory if it doesn't exist
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for symbol in args.symbols:
        print("-" * 50)
        # 1. Fetch data
        df_raw = fetch_aggs_all(symbol, args.start, args.end, api_key, args.timespan, args.multiplier)
        
        if df_raw is None or df_raw.empty:
            print(f"No data fetched for {symbol}. Skipping.")
            continue
            
        # 2. Filter and prepare data
        df_clean = filter_and_prepare_data(df_raw)
        
        if df_clean is None or df_clean.empty:
            print(f"No data remaining for {symbol} after filtering. Skipping.")
            continue
        
        # 3. Define output paths
        file_prefix = f"{symbol.upper()}_RTH_NH"
        csv_path = output_dir / f"{file_prefix}.csv"
        bin_path = output_dir / f"{file_prefix}.bin"
        
        # 4. Save to CSV for inspection
        print(f"Saving to CSV format at {csv_path}...")
        # Select and order columns to match C++ struct for clarity
        csv_columns = ['ts_utc', 'ts_nyt_epoch', 'open', 'high', 'low', 'close', 'volume']
        df_clean[csv_columns].to_csv(csv_path, index=False)
        print(" -> CSV file saved successfully.")
        
        # 5. Save to C++ compatible binary format
        save_to_bin(df_clean, bin_path)

    print("-" * 50)
    print("Data download and processing complete.")

if __name__ == "__main__":
    main()


```

