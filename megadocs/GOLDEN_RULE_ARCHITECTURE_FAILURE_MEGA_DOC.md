# Golden Rule Architecture Failure Analysis

**Generated**: 2025-09-19 03:16:21
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: Comprehensive analysis of why the Golden Rule execution pipeline (EOD/safety first, strategy second) is not preventing EOD violations and conflicting positions despite correct implementation

**Total Files**: 11

---

## 🐛 **BUG REPORT**

# 🐛 CRITICAL BUG REPORT: Golden Rule Architecture Still Has Violations

## 📋 **Bug Summary**
Despite implementing a comprehensive "Golden Rule" architectural redesign (EOD/safety checks first, strategy consultation second), the system continues to exhibit critical violations in 20-block runs, indicating fundamental gaps in the safety system enforcement.

## 🔍 **Latest Status (Run ID: 266910)**
- **Strategy**: sigor (20 trading blocks)
- **Total Fills**: 3,930 trades (196.5 per block - extremely high frequency)
- **Net P&L**: +$57.85 (barely profitable with violations)

## ❌ **Critical Violations Detected**

### **1. Position Conflicts (Mixed Directional Exposure)**
```
FINAL POSITIONS: PSQ:685.2 QQQ:293.4 SQQQ:1754.3 TQQQ:76.2
ISSUE: Both long ETFs (QQQ, TQQQ) and inverse ETFs (PSQ, SQQQ) held simultaneously
STATUS: UniversalPositionCoordinator conflict detection failed despite Golden Rule architecture
```

### **2. EOD Violations (6 days with overnight positions)**
```
❌ 6 days with overnight positions detected
RISK: Overnight carry risk, leveraged ETF decay, gap risk exposure
STATUS: AdaptiveEODManager closure logic failed despite multi-layered approach
```

### **3. Excessive Trading Frequency**
```
TOTAL FILLS: 3,930 trades (196.5 per block)
ISSUE: Extremely high frequency trading not controlled by safety systems
STATUS: One-trade-per-bar rule not being enforced
```

## 🔧 **Implemented Fixes That Are Not Working**

### **Golden Rule Architecture (Latest Attempt)**
- ✅ **Architectural Redesign**: EOD/safety checks first, strategy consultation second
- ✅ **Pipeline Restructure**: `execute_bar_pipeline()` with mandatory EOD priority
- ✅ **Simplified Logic**: Removed complex deferred trades and hysteresis
- ❌ **Result**: Still 6 days with EOD violations and mixed directional exposure

### **Multi-Layered EOD Manager**
- ✅ **Layer 1**: Extended 60-minute closure window for AGGRESSIVE strategies
- ✅ **Layer 2**: Mandatory 5-minute "nuclear option" liquidation
- ✅ **Robust Day Tracking**: Year+day-of-year unique ID for foolproof resets
- ❌ **Result**: Still 6 days with overnight positions (worse than before)

### **Universal Position Coordinator**
- ✅ **Strict Conflict Resolution**: "Close first, then open later" policy
- ✅ **Immediate Rejection**: New conflicting trades rejected
- ✅ **One Trade Per Bar**: Enforced frequency limits
- ❌ **Result**: Still mixed directional exposure and 3,930 fills (should be ~20)

## 🚨 **Root Cause Analysis**

### **Critical Finding: Golden Rule Architecture Not Being Enforced**
Despite implementing the theoretically sound "Golden Rule" architecture, the practical results show:

1. **Safety Systems Are Not Being Invoked**: The 3,930 fills (vs expected ~20) indicate the one-trade-per-bar rule is completely ignored
2. **EOD Logic Is Not Triggered**: 6 days with overnight positions suggest the EOD manager is never called or always bypassed
3. **Conflict Detection Is Ineffective**: Mixed directional exposure shows the position coordinator is not preventing conflicts

### **Hypothesis 1: Execution Pipeline Bypass**
The Golden Rule architecture may be bypassed by:
- Strategy continuing to use old execution paths
- Safety systems not being called in the main loop
- Conditional logic that skips safety checks under certain conditions

### **Hypothesis 2: Component Integration Failure**
The new components may not be properly wired:
- `execute_bar_pipeline()` may not be called correctly
- Parameters may not be passed properly between components
- State may not be synchronized across the safety systems

### **Hypothesis 3: Logic Inversion**
The system may still be operating under the old paradigm:
- Strategy decisions processed first, safety checks second
- EOD manager called only when strategy allows it
- Position coordinator acting as advisory rather than mandatory

## 🔍 **Diagnostic Evidence**

### **Strategy Profile Behavior**
```
Block 1: CONSERVATIVE → Fills=378 (high activity but classified as conservative)
Block 2: AGGRESSIVE → Fills=0 (no activity despite aggressive classification)
...
Pattern: Inconsistent classification vs. actual trading behavior
```

### **Conflicting Position Pattern**
```
Consistent pattern: QQQ + TQQQ (long) vs. SQQQ (inverse)
Timing: Multiple consecutive minutes with same conflict
Location: 2025-08-16 01:44:00 to 01:48:00 (5-minute window)
```

## 🎯 **Required Investigation**

### **1. Verify Fix Implementation**
- Confirm `AdaptiveEODManager::get_eod_allocations()` is being called
- Verify `UniversalPositionCoordinator::coordinate()` logic execution
- Check if `FINAL_LIQUIDATION_MINUTES` path is ever reached

### **2. Debug Timing Logic**
- Validate market close time calculations (UTC vs local time)
- Confirm `tm_yday` day boundary detection
- Verify `minutes_to_close` calculations

### **3. Strategy Profile Debugging**
- Check if `observe_block_complete()` is being called
- Verify `trades_per_block` calculation accuracy
- Confirm style classification logic

## 🚀 **Proposed Solutions**

### **Option 1: Enhanced Debugging**
Add comprehensive logging to track:
- EOD manager decision points
- Position coordinator rejection reasons
- Strategy profile classification changes

### **Option 2: Simplified Approach**
Implement a more aggressive, simpler EOD system:
- Force close ALL positions at 15 minutes before close
- Disable new position opening in final 30 minutes
- Override all other logic during EOD window

### **Option 3: Architecture Review**
- Investigate if the strategy-agnostic backend is properly integrated
- Verify that all components are using the new adaptive system
- Check for legacy code paths that bypass the new logic

## ⚠️ **Impact Assessment**
- **Risk Level**: CRITICAL
- **Live Trading**: BLOCKED until resolved
- **Performance**: Profitable but unsafe
- **Compliance**: Multiple principle violations

## 📊 **Success Criteria**
1. **Zero conflicting positions** in any run
2. **Zero EOD violations** in any run  
3. **100% integrity check pass rate**
4. **Consistent behavior** across different block counts (3, 10, 20)

---
**Status**: ACTIVE BUG - Requires immediate investigation and resolution
**Priority**: P0 - Blocking live trading deployment
**Assigned**: Backend Architecture Team


---

## 📋 **TABLE OF CONTENTS**

1. [temp_mega_doc/REMAINING_EOD_POSITION_CONFLICTS_BUG_REPORT.md](#file-1)
2. [temp_mega_doc/adaptive_allocation_manager.cpp](#file-2)
3. [temp_mega_doc/adaptive_allocation_manager.hpp](#file-3)
4. [temp_mega_doc/adaptive_eod_manager.cpp](#file-4)
5. [temp_mega_doc/adaptive_eod_manager.hpp](#file-5)
6. [temp_mega_doc/runner.cpp](#file-6)
7. [temp_mega_doc/runner.hpp](#file-7)
8. [temp_mega_doc/strategy_profiler.cpp](#file-8)
9. [temp_mega_doc/strategy_profiler.hpp](#file-9)
10. [temp_mega_doc/universal_position_coordinator.cpp](#file-10)
11. [temp_mega_doc/universal_position_coordinator.hpp](#file-11)

---

## 📄 **FILE 1 of 11**: temp_mega_doc/REMAINING_EOD_POSITION_CONFLICTS_BUG_REPORT.md

**File Information**:
- **Path**: `temp_mega_doc/REMAINING_EOD_POSITION_CONFLICTS_BUG_REPORT.md`

- **Size**: 149 lines
- **Modified**: 2025-09-19 03:16:14

- **Type**: .md

```text
# 🐛 CRITICAL BUG REPORT: Golden Rule Architecture Still Has Violations

## 📋 **Bug Summary**
Despite implementing a comprehensive "Golden Rule" architectural redesign (EOD/safety checks first, strategy consultation second), the system continues to exhibit critical violations in 20-block runs, indicating fundamental gaps in the safety system enforcement.

## 🔍 **Latest Status (Run ID: 266910)**
- **Strategy**: sigor (20 trading blocks)
- **Total Fills**: 3,930 trades (196.5 per block - extremely high frequency)
- **Net P&L**: +$57.85 (barely profitable with violations)

## ❌ **Critical Violations Detected**

### **1. Position Conflicts (Mixed Directional Exposure)**
```
FINAL POSITIONS: PSQ:685.2 QQQ:293.4 SQQQ:1754.3 TQQQ:76.2
ISSUE: Both long ETFs (QQQ, TQQQ) and inverse ETFs (PSQ, SQQQ) held simultaneously
STATUS: UniversalPositionCoordinator conflict detection failed despite Golden Rule architecture
```

### **2. EOD Violations (6 days with overnight positions)**
```
❌ 6 days with overnight positions detected
RISK: Overnight carry risk, leveraged ETF decay, gap risk exposure
STATUS: AdaptiveEODManager closure logic failed despite multi-layered approach
```

### **3. Excessive Trading Frequency**
```
TOTAL FILLS: 3,930 trades (196.5 per block)
ISSUE: Extremely high frequency trading not controlled by safety systems
STATUS: One-trade-per-bar rule not being enforced
```

## 🔧 **Implemented Fixes That Are Not Working**

### **Golden Rule Architecture (Latest Attempt)**
- ✅ **Architectural Redesign**: EOD/safety checks first, strategy consultation second
- ✅ **Pipeline Restructure**: `execute_bar_pipeline()` with mandatory EOD priority
- ✅ **Simplified Logic**: Removed complex deferred trades and hysteresis
- ❌ **Result**: Still 6 days with EOD violations and mixed directional exposure

### **Multi-Layered EOD Manager**
- ✅ **Layer 1**: Extended 60-minute closure window for AGGRESSIVE strategies
- ✅ **Layer 2**: Mandatory 5-minute "nuclear option" liquidation
- ✅ **Robust Day Tracking**: Year+day-of-year unique ID for foolproof resets
- ❌ **Result**: Still 6 days with overnight positions (worse than before)

### **Universal Position Coordinator**
- ✅ **Strict Conflict Resolution**: "Close first, then open later" policy
- ✅ **Immediate Rejection**: New conflicting trades rejected
- ✅ **One Trade Per Bar**: Enforced frequency limits
- ❌ **Result**: Still mixed directional exposure and 3,930 fills (should be ~20)

## 🚨 **Root Cause Analysis**

### **Critical Finding: Golden Rule Architecture Not Being Enforced**
Despite implementing the theoretically sound "Golden Rule" architecture, the practical results show:

1. **Safety Systems Are Not Being Invoked**: The 3,930 fills (vs expected ~20) indicate the one-trade-per-bar rule is completely ignored
2. **EOD Logic Is Not Triggered**: 6 days with overnight positions suggest the EOD manager is never called or always bypassed
3. **Conflict Detection Is Ineffective**: Mixed directional exposure shows the position coordinator is not preventing conflicts

### **Hypothesis 1: Execution Pipeline Bypass**
The Golden Rule architecture may be bypassed by:
- Strategy continuing to use old execution paths
- Safety systems not being called in the main loop
- Conditional logic that skips safety checks under certain conditions

### **Hypothesis 2: Component Integration Failure**
The new components may not be properly wired:
- `execute_bar_pipeline()` may not be called correctly
- Parameters may not be passed properly between components
- State may not be synchronized across the safety systems

### **Hypothesis 3: Logic Inversion**
The system may still be operating under the old paradigm:
- Strategy decisions processed first, safety checks second
- EOD manager called only when strategy allows it
- Position coordinator acting as advisory rather than mandatory

## 🔍 **Diagnostic Evidence**

### **Strategy Profile Behavior**
```
Block 1: CONSERVATIVE → Fills=378 (high activity but classified as conservative)
Block 2: AGGRESSIVE → Fills=0 (no activity despite aggressive classification)
...
Pattern: Inconsistent classification vs. actual trading behavior
```

### **Conflicting Position Pattern**
```
Consistent pattern: QQQ + TQQQ (long) vs. SQQQ (inverse)
Timing: Multiple consecutive minutes with same conflict
Location: 2025-08-16 01:44:00 to 01:48:00 (5-minute window)
```

## 🎯 **Required Investigation**

### **1. Verify Fix Implementation**
- Confirm `AdaptiveEODManager::get_eod_allocations()` is being called
- Verify `UniversalPositionCoordinator::coordinate()` logic execution
- Check if `FINAL_LIQUIDATION_MINUTES` path is ever reached

### **2. Debug Timing Logic**
- Validate market close time calculations (UTC vs local time)
- Confirm `tm_yday` day boundary detection
- Verify `minutes_to_close` calculations

### **3. Strategy Profile Debugging**
- Check if `observe_block_complete()` is being called
- Verify `trades_per_block` calculation accuracy
- Confirm style classification logic

## 🚀 **Proposed Solutions**

### **Option 1: Enhanced Debugging**
Add comprehensive logging to track:
- EOD manager decision points
- Position coordinator rejection reasons
- Strategy profile classification changes

### **Option 2: Simplified Approach**
Implement a more aggressive, simpler EOD system:
- Force close ALL positions at 15 minutes before close
- Disable new position opening in final 30 minutes
- Override all other logic during EOD window

### **Option 3: Architecture Review**
- Investigate if the strategy-agnostic backend is properly integrated
- Verify that all components are using the new adaptive system
- Check for legacy code paths that bypass the new logic

## ⚠️ **Impact Assessment**
- **Risk Level**: CRITICAL
- **Live Trading**: BLOCKED until resolved
- **Performance**: Profitable but unsafe
- **Compliance**: Multiple principle violations

## 📊 **Success Criteria**
1. **Zero conflicting positions** in any run
2. **Zero EOD violations** in any run  
3. **100% integrity check pass rate**
4. **Consistent behavior** across different block counts (3, 10, 20)

---
**Status**: ACTIVE BUG - Requires immediate investigation and resolution
**Priority**: P0 - Blocking live trading deployment
**Assigned**: Backend Architecture Team

```

## 📄 **FILE 2 of 11**: temp_mega_doc/adaptive_allocation_manager.cpp

**File Information**:
- **Path**: `temp_mega_doc/adaptive_allocation_manager.cpp`

- **Size**: 112 lines
- **Modified**: 2025-09-19 03:16:09

- **Type**: .cpp

```text
#include "sentio/adaptive_allocation_manager.hpp"
#include <cmath>
#include <algorithm>

namespace sentio {

AdaptiveAllocationManager::AdaptiveAllocationManager() {}

std::vector<AllocationDecision> AdaptiveAllocationManager::get_allocations(
    double probability,
    const StrategyProfiler::StrategyProfile& profile) {
    
    update_thresholds(profile);
    
    // Filter signal based on profile
    double filtered_prob = filter_signal(probability, profile);
    
    // Check if signal is strong enough to trade
    if (!should_trade(filtered_prob, profile)) {
        return {};  // No allocation
    }
    
    std::vector<AllocationDecision> decisions;
    double signal_strength = std::abs(filtered_prob - 0.5);
    bool is_bullish = filtered_prob > 0.5;
    
    if (is_bullish) {
        if (filtered_prob >= thresholds_.entry_3x) {
            decisions.push_back({"TQQQ", 1.0, 
                "Strong bullish (p=" + std::to_string(filtered_prob) + "), adaptive 3x"});
        } else if (filtered_prob >= thresholds_.entry_1x) {
            decisions.push_back({"QQQ", 1.0, 
                "Moderate bullish (p=" + std::to_string(filtered_prob) + "), adaptive 1x"});
        }
    } else {
        double inverse_prob = 1.0 - filtered_prob;
        if (inverse_prob >= thresholds_.entry_3x) {
            decisions.push_back({"SQQQ", 1.0, 
                "Strong bearish (p=" + std::to_string(filtered_prob) + "), adaptive 3x inverse"});
        } else if (inverse_prob >= thresholds_.entry_1x) {
            decisions.push_back({"PSQ", 1.0, 
                "Moderate bearish (p=" + std::to_string(filtered_prob) + "), adaptive 1x inverse"});
        }
    }
    
    if (!decisions.empty()) {
        signals_since_trade_ = 0;
    } else {
        signals_since_trade_++;
    }
    
    last_signal_ = filtered_prob;
    return decisions;
}

bool AdaptiveAllocationManager::should_trade(double probability, 
    const StrategyProfiler::StrategyProfile& profile) const {
    
    double signal_strength = std::abs(probability - 0.5);
    
    // Adaptive noise filtering
    if (signal_strength < profile.adaptive_noise_floor) {
        return false;  // Signal is noise
    }
    
    // For aggressive strategies, require stronger signals to reduce churning
    if (profile.style == TradingStyle::AGGRESSIVE) {
        return signal_strength > thresholds_.signal_strength_min * 1.5;
    }
    
    // For conservative strategies, respect smaller signals
    if (profile.style == TradingStyle::CONSERVATIVE) {
        return signal_strength > thresholds_.signal_strength_min * 0.7;
    }
    
    return signal_strength > thresholds_.signal_strength_min;
}

void AdaptiveAllocationManager::update_thresholds(const StrategyProfiler::StrategyProfile& profile) {
    // Use adaptive thresholds from profile
    thresholds_.entry_1x = profile.adaptive_entry_1x;
    thresholds_.entry_3x = profile.adaptive_entry_3x;
    thresholds_.noise_floor = profile.adaptive_noise_floor;
    
    // Adjust signal strength minimum based on trading frequency
    if (profile.trades_per_block > 200) {
        // Very high frequency - increase minimum signal strength
        thresholds_.signal_strength_min = 0.15;
    } else if (profile.trades_per_block > 50) {
        // Moderate frequency
        thresholds_.signal_strength_min = 0.10;
    } else {
        // Low frequency - allow weaker signals
        thresholds_.signal_strength_min = 0.05;
    }
}

double AdaptiveAllocationManager::filter_signal(double raw_probability, 
    const StrategyProfiler::StrategyProfile& profile) const {
    
    // Apply smoothing for volatile strategies
    if (profile.signal_volatility > 0.2) {
        // Exponential smoothing
        double alpha = 0.3;  // Smoothing factor
        return alpha * raw_probability + (1 - alpha) * last_signal_;
    }
    
    // For stable strategies, use raw signal
    return raw_probability;
}

} // namespace sentio

```

## 📄 **FILE 3 of 11**: temp_mega_doc/adaptive_allocation_manager.hpp

**File Information**:
- **Path**: `temp_mega_doc/adaptive_allocation_manager.hpp`

- **Size**: 46 lines
- **Modified**: 2025-09-19 03:16:09

- **Type**: .hpp

```text
#pragma once
#include "sentio/strategy_profiler.hpp"
#include <memory>
#include <string>
#include <vector>

namespace sentio {

/**
 * @brief Represents a concrete allocation decision for the runner.
 */
struct AllocationDecision {
    std::string instrument;
    double target_weight; // e.g., 1.0 for 100%
    std::string reason;
};

class AdaptiveAllocationManager {
public:
    AdaptiveAllocationManager();
    
    std::vector<AllocationDecision> get_allocations(
        double probability,
        const StrategyProfiler::StrategyProfile& profile
    );
    
    // Signal filtering based on profile
    bool should_trade(double probability, const StrategyProfiler::StrategyProfile& profile) const;
    
private:
    struct DynamicThresholds {
        double entry_1x = 0.60;
        double entry_3x = 0.75;
        double noise_floor = 0.05;
        double signal_strength_min = 0.10;
    };
    
    DynamicThresholds thresholds_;
    double last_signal_ = 0.5;
    int signals_since_trade_ = 0;
    
    void update_thresholds(const StrategyProfiler::StrategyProfile& profile);
    double filter_signal(double raw_probability, const StrategyProfiler::StrategyProfile& profile) const;
};

} // namespace sentio

```

## 📄 **FILE 4 of 11**: temp_mega_doc/adaptive_eod_manager.cpp

**File Information**:
- **Path**: `temp_mega_doc/adaptive_eod_manager.cpp`

- **Size**: 140 lines
- **Modified**: 2025-09-19 03:16:09

- **Type**: .cpp

```text
#include "sentio/adaptive_eod_manager.hpp"
#include <ctime>

namespace sentio {

AdaptiveEODManager::AdaptiveEODManager() {}

std::vector<AllocationDecision> AdaptiveEODManager::get_eod_allocations(
    int64_t current_timestamp_utc,
    const Portfolio& portfolio,
    const SymbolTable& ST,
    const StrategyProfiler::StrategyProfile& profile) {
    
    adapt_config(profile);
    
    std::vector<AllocationDecision> closing_decisions;
    
    // Convert to time components
    time_t time_secs = current_timestamp_utc;
    tm* utc_tm = gmtime(&time_secs);
    if (!utc_tm) return closing_decisions;
    
    int current_hour = utc_tm->tm_hour;
    int current_minute = utc_tm->tm_min;
    int current_day = utc_tm->tm_yday;  // Day of year
    
    // --- BUG FIX: Robust Day Tracking ---
    // Using tm_yday alone fails on year changes (e.g., Dec 31 -> Jan 1).
    // Combine year and day-of-year for a unique day ID.
    int unique_day_id = (utc_tm->tm_year * 1000) + utc_tm->tm_yday;
    
    if (unique_day_id != last_processed_day_) {
        closed_today_.clear();
        last_processed_day_ = unique_day_id;
    }
    
    // Calculate minutes until close
    int current_minutes = current_hour * 60 + current_minute;
    int close_minutes = config_.market_close_hour_utc * 60 + config_.market_close_minute_utc;
    int minutes_to_close = close_minutes - current_minutes;

    // LAYER 1: MANDATORY LIQUIDATION ("NUCLEAR OPTION")
    // This is the final, non-negotiable closure phase.
    constexpr int FINAL_LIQUIDATION_MINUTES = 5;
    if (minutes_to_close <= FINAL_LIQUIDATION_MINUTES && minutes_to_close > 0) {
        closure_active_ = true;
        for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
            // If any position still exists, generate a closing order.
            if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
                const std::string& symbol = ST.get_symbol(sid);
                closing_decisions.push_back({
                    symbol, 0.0, 
                    "MANDATORY EOD LIQUIDATION (" + std::to_string(minutes_to_close) + " min)"
                });
            }
        }
        // Return immediately to ensure no other logic interferes.
        return closing_decisions;
    }

    // LAYER 2: STANDARD ADAPTIVE CLOSURE WINDOW
    // This phase runs before the final liquidation.
    if (minutes_to_close <= config_.closure_start_minutes && minutes_to_close > FINAL_LIQUIDATION_MINUTES) {
        closure_active_ = true;
        
        bool force_close = (minutes_to_close <= config_.mandatory_close_minutes);
        
        for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
            if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
                const std::string& symbol = ST.get_symbol(sid);
                
                // The closed_today_ check is appropriate here to avoid repeated orders
                // during the "soft" close period. The mandatory phase above will catch anything missed.
                if (closed_today_.count(symbol)) continue;
                
                if (force_close || should_close_position(symbol, profile)) {
                    closing_decisions.push_back({
                        symbol, 0.0, 
                        "EOD closure (adaptive, " + std::to_string(minutes_to_close) + " min to close)"
                    });
                    closed_today_.insert(symbol);
                }
            }
        }
        
        if (!closing_decisions.empty()) {
            last_close_timestamp_ = current_timestamp_utc;
        }
    } else {
        closure_active_ = false;
    }
    
    return closing_decisions;
}

void AdaptiveEODManager::adapt_config(const StrategyProfiler::StrategyProfile& profile) {
    // Adjust closure timing based on strategy profile
    switch (profile.style) {
        case TradingStyle::AGGRESSIVE:
            // WIDER window for aggressive strategies to ensure all positions can be closed.
            config_.closure_start_minutes = 60;  // Start 1 hour before market close.
            config_.mandatory_close_minutes = 15; // Start forcing 15 min before close.
            break;
            
        case TradingStyle::CONSERVATIVE:
            // Standard closure for conservative strategies
            config_.closure_start_minutes = 15;
            config_.mandatory_close_minutes = 10;
            break;
            
        case TradingStyle::BURST:
            // Flexible closure for burst strategies
            config_.closure_start_minutes = 25;
            config_.mandatory_close_minutes = 12;
            break;
            
        default:
            // Keep defaults
            break;
    }
}

bool AdaptiveEODManager::should_close_position(const std::string& symbol,
    const StrategyProfiler::StrategyProfile& profile) const {
    
    // For aggressive strategies, always close
    if (profile.style == TradingStyle::AGGRESSIVE) {
        return true;
    }
    
    // For leveraged positions, always close
    if (symbol == "TQQQ" || symbol == "SQQQ") {
        return true;
    }
    
    // For conservative strategies, can be more selective
    return closure_active_;
}

} // namespace sentio

```

## 📄 **FILE 5 of 11**: temp_mega_doc/adaptive_eod_manager.hpp

**File Information**:
- **Path**: `temp_mega_doc/adaptive_eod_manager.hpp`

- **Size**: 44 lines
- **Modified**: 2025-09-19 03:16:09

- **Type**: .hpp

```text
#pragma once
#include "sentio/adaptive_allocation_manager.hpp"
#include "sentio/strategy_profiler.hpp"
#include "sentio/core.hpp"
#include "sentio/symbol_table.hpp"
#include <unordered_set>

namespace sentio {

class AdaptiveEODManager {
public:
    AdaptiveEODManager();
    
    std::vector<AllocationDecision> get_eod_allocations(
        int64_t current_timestamp_utc,
        const Portfolio& portfolio,
        const SymbolTable& ST,
        const StrategyProfiler::StrategyProfile& profile
    );
    
    bool is_closure_active() const { return closure_active_; }
    
        private:
            struct AdaptiveConfig {
                int closure_start_minutes = 15;
                int mandatory_close_minutes = 10;
                int market_close_hour_utc = 20;
                int market_close_minute_utc = 0;
            };
            
            AdaptiveConfig config_;
            bool closure_active_ = false;
            int64_t last_close_timestamp_ = 0;
            std::unordered_set<std::string> closed_today_;
            
            // **FIX**: Add robust day tracking to prevent EOD violations
            int last_processed_day_ = -1;
    
    void adapt_config(const StrategyProfiler::StrategyProfile& profile);
    bool should_close_position(const std::string& symbol, 
                              const StrategyProfiler::StrategyProfile& profile) const;
};

} // namespace sentio

```

## 📄 **FILE 6 of 11**: temp_mega_doc/runner.cpp

**File Information**:
- **Path**: `temp_mega_doc/runner.cpp`

- **Size**: 988 lines
- **Modified**: 2025-09-19 03:16:09

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
#include <iostream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <sqlite3.h>

namespace sentio {

// **ARCHITECTURAL FIX: GOLDEN RULE EXECUTION PIPELINE**
// System integrity and risk management rules are checked FIRST. Strategy is only consulted if system allows it.
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
    
    auto profile = profiler.get_current_profile();

    // --- ARCHITECTURAL FIX: EOD Check First ---
    // The EOD Manager gets absolute priority. If it issues closing orders, the strategy is ignored.
    auto eod_allocations = eod_mgr.get_eod_allocations(bar.ts_utc_epoch, portfolio, ST, profile);
    if (!eod_allocations.empty()) {
        // EOD is active. Coordinate and execute ONLY the closing orders.
        position_coord.reset_bar(bar.ts_utc_epoch);
        auto coordination_decisions = position_coord.coordinate(eod_allocations, portfolio, ST, bar.ts_utc_epoch, profile);
        
        // Execute EOD closing orders
        for (const auto& coord_decision : coordination_decisions) {
            if (coord_decision.result == CoordinationResult::APPROVED) {
                // Execute trade logic (same as below)
                const auto& decision = coord_decision.decision;
                SafeSizer sizer;
                double target_qty = sizer.calculate_target_quantity(
                    portfolio, ST, pricebook.last_px, 
                    decision.instrument, decision.target_weight, 
                    bar.ts_utc_epoch, series[ST.get_id(decision.instrument)]
                );
                
                int instrument_id = ST.get_id(decision.instrument);
                if (instrument_id != -1) {
                    double current_qty = portfolio.positions[instrument_id].qty;
                    double trade_qty = target_qty - current_qty;
                    
                    if (std::abs(trade_qty) >= 1e-9) {
                        double instrument_price = pricebook.last_px[instrument_id];
                        if (instrument_price > 0) {
                            portfolio.positions[instrument_id].qty = target_qty;
                            total_fills++;
                            
                            if (logging_enabled) {
                                Side side = (trade_qty > 0) ? Side::Buy : Side::Sell;
                                audit.event_fill(bar.ts_utc_epoch, decision.instrument, 
                                                instrument_price, std::abs(trade_qty), 0.0, side);
                            }
                        }
                    }
                }
            }
        }
        return; // End of processing for this bar - strategy is not consulted during EOD
    }

    // --- Standard Pipeline (If not EOD) ---
    // 1. Profile the strategy signal
    profiler.observe_signal(strategy_probability, bar.ts_utc_epoch);
    
    // 2. Get adaptive allocations based on profile
    auto allocations = allocation_mgr.get_allocations(strategy_probability, profile);
    
    // Log signal activity
    if (logging_enabled && !allocations.empty()) {
        audit.event_signal_ex(bar.ts_utc_epoch, ST.get_symbol(0), 
                            SigType::BUY, strategy_probability, chain_id);
    }
    
    // 3. Universal position coordination
    position_coord.reset_bar(bar.ts_utc_epoch);
    auto coordination_decisions = position_coord.coordinate(
        allocations, portfolio, ST, bar.ts_utc_epoch, profile
    );
    
    // STEP 5: Execute approved decisions
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
        
        double equity_after = equity_mark_to_market(portfolio, pricebook.last_px);
        double pos_after = portfolio.positions[instrument_id].qty;
        
        if (logging_enabled) {
            audit.event_fill_ex(bar.ts_utc_epoch, decision.instrument, 
                              instrument_price, std::abs(trade_qty), fees, side,
                              realized_delta, equity_after, pos_after, chain_id);
        }
        
        // Update profiler with trade observation
        profiler.observe_trade(strategy_probability, decision.instrument, bar.ts_utc_epoch);
        total_fills++;
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

## 📄 **FILE 7 of 11**: temp_mega_doc/runner.hpp

**File Information**:
- **Path**: `temp_mega_doc/runner.hpp`

- **Size**: 63 lines
- **Modified**: 2025-09-19 03:16:09

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

## 📄 **FILE 8 of 11**: temp_mega_doc/strategy_profiler.cpp

**File Information**:
- **Path**: `temp_mega_doc/strategy_profiler.cpp`

- **Size**: 175 lines
- **Modified**: 2025-09-19 03:16:09

- **Type**: .cpp

```text
#include "sentio/strategy_profiler.hpp"
#include <algorithm>
#include <numeric>

namespace sentio {

StrategyProfiler::StrategyProfiler() {
    reset_profile();
}

void StrategyProfiler::reset_profile() {
    profile_ = StrategyProfile();
    signal_history_.clear();
    signal_timestamps_.clear();
    trade_signals_.clear();
    block_trade_counts_.clear();
}

void StrategyProfiler::observe_signal(double probability, int64_t timestamp) {
    signal_history_.push_back(probability);
    signal_timestamps_.push_back(timestamp);
    
    // Maintain window size
    while (signal_history_.size() > WINDOW_SIZE) {
        signal_history_.pop_front();
        signal_timestamps_.pop_front();
    }
    
    profile_.observation_count++;
    update_profile();
}

void StrategyProfiler::observe_trade(double probability, const std::string& instrument, int64_t timestamp) {
    trade_signals_.push_back(probability);
    
    while (trade_signals_.size() > WINDOW_SIZE) {
        trade_signals_.pop_front();
    }
}

void StrategyProfiler::observe_block_complete(int trades_in_block) {
    block_trade_counts_.push_back(trades_in_block);
    
    // Keep last 10 blocks for recent behavior
    while (block_trade_counts_.size() > 10) {
        block_trade_counts_.pop_front();
    }
    
    if (!block_trade_counts_.empty()) {
        profile_.trades_per_block = std::accumulate(block_trade_counts_.begin(), 
                                                   block_trade_counts_.end(), 0.0) 
                                  / block_trade_counts_.size();
    }
}

void StrategyProfiler::update_profile() {
    if (signal_history_.size() < MIN_OBSERVATIONS) {
        profile_.confidence_level = signal_history_.size() / double(MIN_OBSERVATIONS);
        return;
    }
    
    // Calculate signal frequency (signals that deviate from 0.5)
    int active_signals = 0;
    for (double prob : signal_history_) {
        if (std::abs(prob - 0.5) > 0.05) {  // Signal threshold
            active_signals++;
        }
    }
    profile_.avg_signal_frequency = double(active_signals) / signal_history_.size();
    
    // Calculate mean and volatility
    double sum = std::accumulate(signal_history_.begin(), signal_history_.end(), 0.0);
    profile_.signal_mean = sum / signal_history_.size();
    
    double sq_sum = 0.0;
    for (double prob : signal_history_) {
        double diff = prob - profile_.signal_mean;
        sq_sum += diff * diff;
    }
    profile_.signal_volatility = std::sqrt(sq_sum / signal_history_.size());
    
    // Update noise threshold and trading style
    profile_.noise_threshold = calculate_noise_threshold();
    detect_trading_style();
    calculate_adaptive_thresholds();
    
    profile_.confidence_level = std::min(1.0, signal_history_.size() / double(WINDOW_SIZE));
}

void StrategyProfiler::detect_trading_style() {
    // --- BUG FIX: Simplified and More Direct Style Detection ---
    // The previous hysteresis logic was getting stuck and misclassifying.
    // This new logic is a direct mapping from recent behavior.
    
    if (block_trade_counts_.empty()) {
        profile_.style = TradingStyle::CONSERVATIVE; // Default style
        return;
    }

    // Use the most recent block's trade count as the primary indicator.
    double recent_trades = block_trade_counts_.back();

    // Define clear, non-overlapping thresholds.
    constexpr double AGGRESSIVE_THRESHOLD = 100.0;
    constexpr double CONSERVATIVE_THRESHOLD = 20.0;
    constexpr double BURST_VOLATILITY_THRESHOLD = 0.25;

    if (recent_trades > AGGRESSIVE_THRESHOLD) {
        profile_.style = TradingStyle::AGGRESSIVE;
    } else if (recent_trades < CONSERVATIVE_THRESHOLD) {
        profile_.style = TradingStyle::CONSERVATIVE;
    } else if (profile_.signal_volatility > BURST_VOLATILITY_THRESHOLD) {
        // Moderate trade count but high signal volatility indicates a "bursty" strategy.
        profile_.style = TradingStyle::BURST;
    } else {
        // In-between behavior is classified as adaptive.
        profile_.style = TradingStyle::ADAPTIVE;
    }
}

void StrategyProfiler::calculate_adaptive_thresholds() {
    // Adapt thresholds based on observed trading style
    switch (profile_.style) {
        case TradingStyle::AGGRESSIVE:
            // Higher thresholds for aggressive strategies to filter noise
            profile_.adaptive_entry_1x = 0.65;
            profile_.adaptive_entry_3x = 0.80;
            profile_.adaptive_noise_floor = 0.10;
            break;
            
        case TradingStyle::CONSERVATIVE:
            // Lower thresholds for conservative strategies
            profile_.adaptive_entry_1x = 0.55;
            profile_.adaptive_entry_3x = 0.70;
            profile_.adaptive_noise_floor = 0.02;
            break;
            
        case TradingStyle::BURST:
            // Dynamic thresholds for burst strategies
            profile_.adaptive_entry_1x = 0.60;
            profile_.adaptive_entry_3x = 0.75;
            profile_.adaptive_noise_floor = 0.05;
            break;
            
        default:
            // Keep defaults for adaptive
            break;
    }
    
    // Further adjust based on signal volatility
    if (profile_.signal_volatility > 0.15) {
        profile_.adaptive_noise_floor *= 1.5;  // Increase noise floor for volatile signals
    }
}

double StrategyProfiler::calculate_noise_threshold() {
    if (signal_history_.size() < MIN_OBSERVATIONS) {
        return 0.05;  // Default
    }
    
    // Calculate signal change frequency to detect noise
    std::vector<double> signal_changes;
    for (size_t i = 1; i < signal_history_.size(); ++i) {
        signal_changes.push_back(std::abs(signal_history_[i] - signal_history_[i-1]));
    }
    
    if (signal_changes.empty()) return 0.05;
    
    // Noise threshold is the 25th percentile of signal changes
    std::sort(signal_changes.begin(), signal_changes.end());
    size_t idx = signal_changes.size() / 4;
    return signal_changes[idx];
}

} // namespace sentio

```

## 📄 **FILE 9 of 11**: temp_mega_doc/strategy_profiler.hpp

**File Information**:
- **Path**: `temp_mega_doc/strategy_profiler.hpp`

- **Size**: 61 lines
- **Modified**: 2025-09-19 03:16:09

- **Type**: .hpp

```text
#pragma once
#include <deque>
#include <unordered_map>
#include <cstdint>
#include <cmath>

namespace sentio {

enum class TradingStyle {
    CONSERVATIVE,  // Low frequency, high conviction (like TFA)
    AGGRESSIVE,    // High frequency, many signals (like sigor)
    BURST,        // Intermittent high activity
    ADAPTIVE      // Changes behavior dynamically
};

class StrategyProfiler {
public:
    struct StrategyProfile {
        double avg_signal_frequency = 0.0;    // signals per bar
        double signal_volatility = 0.0;       // signal strength variance
        double signal_mean = 0.5;            // average signal value
        double noise_threshold = 0.0;        // auto-detected noise level
        double confidence_level = 0.0;       // profile confidence (0-1)
        TradingStyle style = TradingStyle::CONSERVATIVE;
        int observation_count = 0;
        double trades_per_block = 0.0;      // recent trading frequency
        
        // Adaptive thresholds based on observed behavior
        double adaptive_entry_1x = 0.60;    
        double adaptive_entry_3x = 0.75;
        double adaptive_noise_floor = 0.05;
    };
    
    StrategyProfiler();
    
    void observe_signal(double probability, int64_t timestamp);
    void observe_trade(double probability, const std::string& instrument, int64_t timestamp);
    void observe_block_complete(int trades_in_block);
    
    StrategyProfile get_current_profile() const { return profile_; }
    void reset_profile();
    
        private:
            static constexpr size_t WINDOW_SIZE = 500;  // Bars to analyze
            static constexpr size_t MIN_OBSERVATIONS = 50;  // Minimum for confidence
            
            StrategyProfile profile_;
            std::deque<double> signal_history_;
            std::deque<int64_t> signal_timestamps_;
            std::deque<double> trade_signals_;  // Signals that resulted in trades
            std::deque<int> block_trade_counts_;
            
            // **FIX**: Add hysteresis state tracking to prevent oscillation
            
            void update_profile();
            void detect_trading_style();
            void calculate_adaptive_thresholds();
            double calculate_noise_threshold();
};

} // namespace sentio

```

## 📄 **FILE 10 of 11**: temp_mega_doc/universal_position_coordinator.cpp

**File Information**:
- **Path**: `temp_mega_doc/universal_position_coordinator.cpp`

- **Size**: 134 lines
- **Modified**: 2025-09-19 03:16:09

- **Type**: .cpp

```text
#include "sentio/universal_position_coordinator.hpp"
#include <algorithm>

namespace sentio {

UniversalPositionCoordinator::UniversalPositionCoordinator() {}

void UniversalPositionCoordinator::reset_bar(int64_t timestamp) {
    // Reset for new bar - no state to track since we simplified the coordinator
    (void)timestamp; // Suppress unused parameter warning
}

std::vector<CoordinationDecision> UniversalPositionCoordinator::coordinate(
    const std::vector<AllocationDecision>& allocations,
    const Portfolio& portfolio,
    const SymbolTable& ST,
    int64_t current_timestamp,
    const StrategyProfiler::StrategyProfile& profile) {
    
    std::vector<CoordinationDecision> results;

    // If allocations are empty, it's a signal to hold or close.
    // The coordinator's job here is to generate closing orders if needed.
    if (allocations.empty()) {
        if (trades_per_timestamp_[current_timestamp].empty()) { // Only allow one closing transaction
            for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
                if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
                    const std::string& symbol = ST.get_symbol(sid);
                    results.push_back({{symbol, 0.0, "Close position, no active signal"}, CoordinationResult::APPROVED, "Closing position."});
                    // Since we are closing everything, we can add multiple close orders in one go.
                }
            }
            if (!results.empty()) {
                trades_per_timestamp_[current_timestamp].push_back({
                    current_timestamp, "CLOSE_ALL", 0.0
                });
            }
        }
        return results;
    }
    
    // We only process one new allocation request per bar to enforce the one-trade-per-bar rule.
    const auto& primary_decision = allocations[0];

    // PRINCIPLE 3: ONE TRADE PER BAR
    if (!trades_per_timestamp_[current_timestamp].empty()) {
        results.push_back({primary_decision, CoordinationResult::REJECTED_FREQUENCY, "One transaction per bar limit reached."});
        return results;
    }

    // PRINCIPLE 2: NO CONFLICTING POSITIONS
    if (would_create_conflict(primary_decision.instrument, portfolio, ST)) {
        // A conflict exists. The ONLY valid action is to close existing positions.
        // The new trade is REJECTED for this bar.
        results.push_back({primary_decision, CoordinationResult::REJECTED_CONFLICT, "Existing position conflicts. Closing first."});
        
        // Generate closing orders for ALL existing positions to ensure a clean slate.
        for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
            if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
                const std::string& symbol_to_close = ST.get_symbol(sid);
                results.push_back({{symbol_to_close, 0.0, "Closing conflicting position"}, CoordinationResult::APPROVED, "Conflict Resolution: Flattening."});
            }
        }
    } else {
        // No conflict detected. Approve the primary decision.
        results.push_back({primary_decision, CoordinationResult::APPROVED, "Approved by coordinator."});
    }

    // Record the trade if at least one trade (open or close) was approved.
    if (!results.empty()) {
        trades_per_timestamp_[current_timestamp].push_back({
            current_timestamp, 
            primary_decision.instrument,
            std::abs(primary_decision.target_weight)
        });
    }
    
    return results;
}

bool UniversalPositionCoordinator::would_create_conflict(
    const std::string& instrument,
    const Portfolio& portfolio,
    const SymbolTable& ST) const {
    
    bool wants_long = LONG_ETFS.count(instrument);
    bool wants_inverse = INVERSE_ETFS.count(instrument);
    
    if (!wants_long && !wants_inverse) return false;
    
    // Check existing positions
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
            const std::string& symbol = ST.get_symbol(sid);
            bool has_long = LONG_ETFS.count(symbol);
            bool has_inverse = INVERSE_ETFS.count(symbol);
            
            if ((wants_long && has_inverse) || (wants_inverse && has_long)) {
                return true;
            }
        }
    }
    
    return false;
}

bool UniversalPositionCoordinator::can_trade_at_timestamp(
    const std::string& instrument,
    int64_t timestamp,
    const StrategyProfiler::StrategyProfile& profile) {
    
    auto it = trades_per_timestamp_.find(timestamp);
    if (it == trades_per_timestamp_.end()) {
        return true;  // No trades yet at this timestamp
    }
    
    // For aggressive strategies, allow multiple trades per bar
    if (profile.style == TradingStyle::AGGRESSIVE) {
        // Allow up to 3 trades per bar for aggressive strategies
        return it->second.size() < 3;
    }
    
    // For conservative strategies, strict one trade per bar
    for (const auto& record : it->second) {
        if (record.instrument == instrument) {
            return false;  // Already traded this instrument
        }
    }
    
    return it->second.empty();  // Allow if no trades yet
}


} // namespace sentio

```

## 📄 **FILE 11 of 11**: temp_mega_doc/universal_position_coordinator.hpp

**File Information**:
- **Path**: `temp_mega_doc/universal_position_coordinator.hpp`

- **Size**: 66 lines
- **Modified**: 2025-09-19 03:16:09

- **Type**: .hpp

```text
#pragma once
#include "sentio/adaptive_allocation_manager.hpp"
#include "sentio/strategy_profiler.hpp"
#include "sentio/core.hpp"
#include "sentio/symbol_table.hpp"
#include <unordered_map>
#include <unordered_set>
#include <queue>

namespace sentio {

/**
 * @brief The result of a coordination check for a requested trade.
 */
enum class CoordinationResult {
    APPROVED,
    REJECTED_CONFLICT,
    REJECTED_FREQUENCY
};

/**
 * @brief The output of the coordinator for a single allocation request.
 */
struct CoordinationDecision {
    AllocationDecision decision;
    CoordinationResult result;
    std::string reason;
};

class UniversalPositionCoordinator {
public:
    UniversalPositionCoordinator();
    
    std::vector<CoordinationDecision> coordinate(
        const std::vector<AllocationDecision>& allocations,
        const Portfolio& portfolio,
        const SymbolTable& ST,
        int64_t current_timestamp,
        const StrategyProfiler::StrategyProfile& profile
    );
    
    void reset_bar(int64_t timestamp);
    
private:
    struct TradeRecord {
        int64_t timestamp;
        std::string instrument;
        double signal_strength;
    };
    
    std::unordered_map<int64_t, std::vector<TradeRecord>> trades_per_timestamp_;
    
    const std::unordered_set<std::string> LONG_ETFS = {"QQQ", "TQQQ"};
    const std::unordered_set<std::string> INVERSE_ETFS = {"SQQQ", "PSQ"};
    
    bool would_create_conflict(const std::string& instrument, 
                              const Portfolio& portfolio, 
                              const SymbolTable& ST) const;
    
    bool can_trade_at_timestamp(const std::string& instrument, 
                               int64_t timestamp,
                               const StrategyProfiler::StrategyProfile& profile);
    
};

} // namespace sentio

```

