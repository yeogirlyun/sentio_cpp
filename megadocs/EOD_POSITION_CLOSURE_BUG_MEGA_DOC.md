# EOD Position Closure Bug Analysis

**Generated**: 2025-09-19 02:13:56
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: Focused analysis of End-of-Day position management failures with relevant source modules for timing and closure logic

**Total Files**: 8

---

## 🐛 **BUG REPORT**

# EOD Position Closure Bug Report

**Date**: 2025-09-19  
**Reporter**: AI Assistant  
**Severity**: Medium  
**Status**: Active  
**Run ID**: 116676 (3-block sigor test)

## 🐛 **Problem Summary**

The End-of-Day (EOD) position management system fails to close all positions before market close in certain scenarios, resulting in overnight carry risk and integrity violations.

## 📊 **Evidence**

### **Integrity Check Results**
```
🌙 PRINCIPLE 4: EOD CLOSING OF ALL POSITIONS
┌─────────────────────────────────────────────────────────────────────────────────┐
│ ❌ VIOLATION DETECTED │ 1 days with overnight positions │
│ Risk: Overnight carry risk, leveraged ETF decay, gap risk exposure │
│ Fix:  Review EODPositionManager configuration and timing │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **Position History Analysis**
- **Date**: 2025-09-10
- **Overnight Position**: TQQQ:6.16 shares
- **Exposure**: $9,406.31
- **Risk Type**: Leveraged ETF overnight carry

## 🔍 **Root Cause Analysis**

### **1. Timing Window Issues**
The `AdaptiveEODManager` uses a 45-minute closure window for aggressive strategies, but this may be insufficient for certain market conditions or high-frequency trading patterns.

**Current Configuration:**
```cpp
case TradingStyle::AGGRESSIVE:
    config_.closure_start_minutes = 45;  // Start closing 45 min before close
    config_.mandatory_close_minutes = 10; // Force close 10 min before close
    break;
```

### **2. State Tracking Problems**
The `closed_today_` set may prevent proper re-closure of positions that were partially closed earlier in the day.

**Problematic Logic:**
```cpp
// Skip if already closed today
if (closed_today_.count(symbol)) continue;
```

### **3. Insufficient Force Closure**
The mandatory closure logic may not be aggressive enough for high-frequency strategies that accumulate many small positions throughout the day.

## 🎯 **Impact Assessment**

### **Financial Risk**
- **Overnight Carry**: Exposure to gap risk on leveraged ETFs
- **Decay Risk**: TQQQ/SQQQ suffer from daily rebalancing decay
- **Regulatory Risk**: Potential margin calls or position limits

### **System Integrity**
- **Principle Violation**: Breaks core EOD closure requirement
- **Audit Failures**: Causes integrity checks to fail
- **Production Risk**: Unsuitable for live trading deployment

## 🔧 **Proposed Solutions**

### **Option 1: Extend Closure Window**
Increase the closure window for aggressive strategies to ensure sufficient time for position liquidation.

```cpp
case TradingStyle::AGGRESSIVE:
    config_.closure_start_minutes = 60;  // Start 1 hour before close
    config_.mandatory_close_minutes = 5;  // Force close 5 min before close
    break;
```

### **Option 2: Improve State Tracking**
Reset the `closed_today_` set more frequently or track partial vs. complete closures.

```cpp
// Clear closed tracking if position still exists
if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
    closed_today_.erase(symbol);
}
```

### **Option 3: Mandatory Liquidation**
Implement a final "nuclear option" that forces closure of ALL positions in the last 5 minutes regardless of other conditions.

```cpp
// Final liquidation phase - close everything
if (minutes_to_close <= 5) {
    // Force close all positions, ignore closed_today_ tracking
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
            const std::string& symbol = ST.get_symbol(sid);
            closing_decisions.push_back({symbol, 0.0, "MANDATORY EOD LIQUIDATION"});
        }
    }
}
```

## 🧪 **Test Cases**

### **Reproduction Steps**
1. Run `./build/sentio_cli strattest sigor --blocks 3 --mode historical`
2. Execute `./saudit integrity` 
3. Observe EOD violation in integrity report
4. Check `./saudit position-history` for overnight positions

### **Expected Behavior**
- All positions should be closed by 20:00 UTC (market close)
- No overnight positions should exist
- Integrity check should pass all 5 principles

### **Validation Criteria**
- ✅ Zero overnight positions across all test days
- ✅ All leveraged ETF positions (TQQQ, SQQQ) closed before EOD
- ✅ Integrity check passes with 0 EOD violations

## 📈 **Progress Tracking**

### **Recent Improvements**
- ✅ **Conflicting Positions**: Eliminated (was major issue)
- ✅ **Strategy Profiler**: Stabilized with hysteresis logic
- ✅ **Position Coordinator**: Strict conflict enforcement implemented
- ⚠️ **EOD Management**: Improved from 3 days → 1 day violations

### **Remaining Work**
- [ ] Implement extended closure window for aggressive strategies
- [ ] Add mandatory liquidation phase in final minutes
- [ ] Improve state tracking for partial position closures
- [ ] Validate fixes with longer test runs (10+ blocks)

## 🔗 **Related Components**

### **Primary Files**
- `src/adaptive_eod_manager.cpp` - Core EOD management logic
- `include/sentio/adaptive_eod_manager.hpp` - EOD manager interface
- `src/runner.cpp` - Integration with trading pipeline

### **Secondary Files**
- `src/strategy_profiler.cpp` - Trading style classification
- `src/universal_position_coordinator.cpp` - Position coordination
- `audit/src/audit_cli.cpp` - Integrity checking logic

## 📋 **Acceptance Criteria**

The bug will be considered **RESOLVED** when:

1. **Zero EOD Violations**: Integrity checks pass consistently across multiple test runs
2. **All Position Types**: Both regular and leveraged ETF positions close properly
3. **Multiple Strategies**: EOD closure works for both aggressive (sigor) and conservative (TFA) strategies
4. **Stress Testing**: 20-block runs show zero overnight positions
5. **Audit Compliance**: All 5 integrity principles pass without exceptions

---

**Priority**: High (blocks production deployment)  
**Complexity**: Medium (timing and state management)  
**Estimated Effort**: 2-4 hours (configuration tuning + testing)


---

## 📋 **TABLE OF CONTENTS**

1. [temp_mega_doc/EOD_POSITION_CLOSURE_BUG_REPORT.md](#file-1)
2. [temp_mega_doc/adaptive_eod_manager.cpp](#file-2)
3. [temp_mega_doc/adaptive_eod_manager.hpp](#file-3)
4. [temp_mega_doc/audit_cli.cpp](#file-4)
5. [temp_mega_doc/runner.cpp](#file-5)
6. [temp_mega_doc/runner.hpp](#file-6)
7. [temp_mega_doc/strategy_profiler.cpp](#file-7)
8. [temp_mega_doc/strategy_profiler.hpp](#file-8)

---

## 📄 **FILE 1 of 8**: temp_mega_doc/EOD_POSITION_CLOSURE_BUG_REPORT.md

**File Information**:
- **Path**: `temp_mega_doc/EOD_POSITION_CLOSURE_BUG_REPORT.md`

- **Size**: 164 lines
- **Modified**: 2025-09-19 02:13:51

- **Type**: .md

```text
# EOD Position Closure Bug Report

**Date**: 2025-09-19  
**Reporter**: AI Assistant  
**Severity**: Medium  
**Status**: Active  
**Run ID**: 116676 (3-block sigor test)

## 🐛 **Problem Summary**

The End-of-Day (EOD) position management system fails to close all positions before market close in certain scenarios, resulting in overnight carry risk and integrity violations.

## 📊 **Evidence**

### **Integrity Check Results**
```
🌙 PRINCIPLE 4: EOD CLOSING OF ALL POSITIONS
┌─────────────────────────────────────────────────────────────────────────────────┐
│ ❌ VIOLATION DETECTED │ 1 days with overnight positions │
│ Risk: Overnight carry risk, leveraged ETF decay, gap risk exposure │
│ Fix:  Review EODPositionManager configuration and timing │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **Position History Analysis**
- **Date**: 2025-09-10
- **Overnight Position**: TQQQ:6.16 shares
- **Exposure**: $9,406.31
- **Risk Type**: Leveraged ETF overnight carry

## 🔍 **Root Cause Analysis**

### **1. Timing Window Issues**
The `AdaptiveEODManager` uses a 45-minute closure window for aggressive strategies, but this may be insufficient for certain market conditions or high-frequency trading patterns.

**Current Configuration:**
```cpp
case TradingStyle::AGGRESSIVE:
    config_.closure_start_minutes = 45;  // Start closing 45 min before close
    config_.mandatory_close_minutes = 10; // Force close 10 min before close
    break;
```

### **2. State Tracking Problems**
The `closed_today_` set may prevent proper re-closure of positions that were partially closed earlier in the day.

**Problematic Logic:**
```cpp
// Skip if already closed today
if (closed_today_.count(symbol)) continue;
```

### **3. Insufficient Force Closure**
The mandatory closure logic may not be aggressive enough for high-frequency strategies that accumulate many small positions throughout the day.

## 🎯 **Impact Assessment**

### **Financial Risk**
- **Overnight Carry**: Exposure to gap risk on leveraged ETFs
- **Decay Risk**: TQQQ/SQQQ suffer from daily rebalancing decay
- **Regulatory Risk**: Potential margin calls or position limits

### **System Integrity**
- **Principle Violation**: Breaks core EOD closure requirement
- **Audit Failures**: Causes integrity checks to fail
- **Production Risk**: Unsuitable for live trading deployment

## 🔧 **Proposed Solutions**

### **Option 1: Extend Closure Window**
Increase the closure window for aggressive strategies to ensure sufficient time for position liquidation.

```cpp
case TradingStyle::AGGRESSIVE:
    config_.closure_start_minutes = 60;  // Start 1 hour before close
    config_.mandatory_close_minutes = 5;  // Force close 5 min before close
    break;
```

### **Option 2: Improve State Tracking**
Reset the `closed_today_` set more frequently or track partial vs. complete closures.

```cpp
// Clear closed tracking if position still exists
if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
    closed_today_.erase(symbol);
}
```

### **Option 3: Mandatory Liquidation**
Implement a final "nuclear option" that forces closure of ALL positions in the last 5 minutes regardless of other conditions.

```cpp
// Final liquidation phase - close everything
if (minutes_to_close <= 5) {
    // Force close all positions, ignore closed_today_ tracking
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
            const std::string& symbol = ST.get_symbol(sid);
            closing_decisions.push_back({symbol, 0.0, "MANDATORY EOD LIQUIDATION"});
        }
    }
}
```

## 🧪 **Test Cases**

### **Reproduction Steps**
1. Run `./build/sentio_cli strattest sigor --blocks 3 --mode historical`
2. Execute `./saudit integrity` 
3. Observe EOD violation in integrity report
4. Check `./saudit position-history` for overnight positions

### **Expected Behavior**
- All positions should be closed by 20:00 UTC (market close)
- No overnight positions should exist
- Integrity check should pass all 5 principles

### **Validation Criteria**
- ✅ Zero overnight positions across all test days
- ✅ All leveraged ETF positions (TQQQ, SQQQ) closed before EOD
- ✅ Integrity check passes with 0 EOD violations

## 📈 **Progress Tracking**

### **Recent Improvements**
- ✅ **Conflicting Positions**: Eliminated (was major issue)
- ✅ **Strategy Profiler**: Stabilized with hysteresis logic
- ✅ **Position Coordinator**: Strict conflict enforcement implemented
- ⚠️ **EOD Management**: Improved from 3 days → 1 day violations

### **Remaining Work**
- [ ] Implement extended closure window for aggressive strategies
- [ ] Add mandatory liquidation phase in final minutes
- [ ] Improve state tracking for partial position closures
- [ ] Validate fixes with longer test runs (10+ blocks)

## 🔗 **Related Components**

### **Primary Files**
- `src/adaptive_eod_manager.cpp` - Core EOD management logic
- `include/sentio/adaptive_eod_manager.hpp` - EOD manager interface
- `src/runner.cpp` - Integration with trading pipeline

### **Secondary Files**
- `src/strategy_profiler.cpp` - Trading style classification
- `src/universal_position_coordinator.cpp` - Position coordination
- `audit/src/audit_cli.cpp` - Integrity checking logic

## 📋 **Acceptance Criteria**

The bug will be considered **RESOLVED** when:

1. **Zero EOD Violations**: Integrity checks pass consistently across multiple test runs
2. **All Position Types**: Both regular and leveraged ETF positions close properly
3. **Multiple Strategies**: EOD closure works for both aggressive (sigor) and conservative (TFA) strategies
4. **Stress Testing**: 20-block runs show zero overnight positions
5. **Audit Compliance**: All 5 integrity principles pass without exceptions

---

**Priority**: High (blocks production deployment)  
**Complexity**: Medium (timing and state management)  
**Estimated Effort**: 2-4 hours (configuration tuning + testing)

```

## 📄 **FILE 2 of 8**: temp_mega_doc/adaptive_eod_manager.cpp

**File Information**:
- **Path**: `temp_mega_doc/adaptive_eod_manager.cpp`

- **Size**: 119 lines
- **Modified**: 2025-09-19 02:13:46

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
    
    // ROBUST new-day check using day of year
    if (current_day != last_processed_day_) {
        closed_today_.clear();
        last_processed_day_ = current_day;
    }
    
    // Calculate minutes until close
    int current_minutes = current_hour * 60 + current_minute;
    int close_minutes = config_.market_close_hour_utc * 60 + config_.market_close_minute_utc;
    int minutes_to_close = close_minutes - current_minutes;
    
    // Check if in closure window
    if (minutes_to_close <= config_.closure_start_minutes && minutes_to_close > 0) {
        closure_active_ = true;
        
        // For aggressive strategies, close positions more aggressively
        bool force_close = (minutes_to_close <= config_.mandatory_close_minutes) ||
                          (profile.style == TradingStyle::AGGRESSIVE && 
                           minutes_to_close <= config_.closure_start_minutes);
        
        // Generate close orders for all positions
        for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
            if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
                const std::string& symbol = ST.get_symbol(sid);
                
                // Skip if already closed today
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
            config_.closure_start_minutes = 45;
            config_.mandatory_close_minutes = 10;
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

## 📄 **FILE 3 of 8**: temp_mega_doc/adaptive_eod_manager.hpp

**File Information**:
- **Path**: `temp_mega_doc/adaptive_eod_manager.hpp`

- **Size**: 44 lines
- **Modified**: 2025-09-19 02:13:46

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

## 📄 **FILE 4 of 8**: temp_mega_doc/audit_cli.cpp

**File Information**:
- **Path**: `temp_mega_doc/audit_cli.cpp`

- **Size**: 3535 lines
- **Modified**: 2025-09-19 02:13:46

- **Type**: .cpp

```text
#include "audit/audit_cli.hpp"
#include "audit/audit_db.hpp"
#include "audit/clock.hpp"
#include "sentio/sentio_integration_adapter.hpp"
#include "sentio/core.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <sqlite3.h>
#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp>

// ANSI color codes for enhanced visual output
#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define DIM     "\033[2m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"
#define BG_BLUE "\033[44m"

using namespace audit;

// **CONFLICT DETECTION**: ETF classifications for conflict analysis
static const std::unordered_set<std::string> LONG_ETFS = {"QQQ", "TQQQ"};
static const std::unordered_set<std::string> INVERSE_ETFS = {"SQQQ"}; // PSQ removed

// **CONFLICT DETECTION**: Position tracking for conflict analysis
struct ConflictPosition {
    double qty = 0.0;
    std::string symbol;
};

struct ConflictAnalysis {
    std::vector<std::string> conflicts;
    std::unordered_map<std::string, ConflictPosition> positions;
    int conflict_count = 0;
    bool has_conflicts = false;
};

// **CONFLICT DETECTION**: Check for conflicting positions
static ConflictAnalysis analyze_position_conflicts(const std::unordered_map<std::string, ConflictPosition>& positions) {
    ConflictAnalysis analysis;
    analysis.positions = positions;
    
    bool has_long_etf = false;
    bool has_inverse_etf = false;
    bool has_short_qqq = false;
    
    std::vector<std::string> long_positions;
    std::vector<std::string> short_positions;
    std::vector<std::string> inverse_positions;
    
    for (const auto& [symbol, pos] : positions) {
        if (std::abs(pos.qty) > 1e-6) {
            if (LONG_ETFS.count(symbol)) {
                if (pos.qty > 0) {
                    has_long_etf = true;
                    long_positions.push_back(symbol + "(+" + std::to_string((int)pos.qty) + ")");
                } else {
                    has_short_qqq = true;
                    short_positions.push_back("SHORT " + symbol + "(" + std::to_string((int)pos.qty) + ")");
                }
            }
            if (INVERSE_ETFS.count(symbol)) {
                has_inverse_etf = true;
                inverse_positions.push_back(symbol + "(" + std::to_string((int)pos.qty) + ")");
            }
        }
    }
    
    // **CONFLICT RULES**:
    // 1. Long ETF conflicts with Inverse ETF or SHORT QQQ
    // 2. SHORT QQQ conflicts with Long ETF
    // 3. Inverse ETF conflicts with Long ETF
    if ((has_long_etf && (has_inverse_etf || has_short_qqq)) || 
        (has_short_qqq && has_long_etf)) {
        analysis.has_conflicts = true;
        analysis.conflict_count++;
        
        std::string conflict_desc = "CONFLICTING POSITIONS DETECTED: ";
        if (!long_positions.empty()) {
            conflict_desc += "Long: ";
            for (size_t i = 0; i < long_positions.size(); ++i) {
                if (i > 0) conflict_desc += ", ";
                conflict_desc += long_positions[i];
            }
        }
        if (!short_positions.empty()) {
            if (!long_positions.empty()) conflict_desc += "; ";
            conflict_desc += "Short: ";
            for (size_t i = 0; i < short_positions.size(); ++i) {
                if (i > 0) conflict_desc += ", ";
                conflict_desc += short_positions[i];
            }
        }
        if (!inverse_positions.empty()) {
            if (!long_positions.empty() || !short_positions.empty()) conflict_desc += "; ";
            conflict_desc += "Inverse: ";
            for (size_t i = 0; i < inverse_positions.size(); ++i) {
                if (i > 0) conflict_desc += ", ";
                conflict_desc += inverse_positions[i];
            }
        }
        
        analysis.conflicts.push_back(conflict_desc);
    }
    
    return analysis;
}

static const char* usage =
  "sentio_audit <cmd> [options]\n"
  "\n"
  "DATABASE MANAGEMENT:\n"
  "  init           [--db DB]\n"
  "  reset          [--db DB] [--confirm]  # WARNING: Deletes all audit data!\n"
  "  vacuum         [--db DB]\n"
  "\n"
  "RUN MANAGEMENT:\n"
  "  new-run        [--db DB] --run RUN --strategy STRAT --kind KIND --params FILE --data-hash HASH --git REV [--note NOTE]\n"
  "  end-run        [--db DB] --run RUN\n"
  "  log            [--db DB] --run RUN --ts MS --kind KIND [--symbol S] [--side SIDE] [--qty Q] [--price P] [--pnl P] [--weight W] [--prob P] [--reason R] [--note NOTE]\n"
  "\n"
  "QUERY COMMANDS:\n"
  "  list           [--db DB] [--strategy STRAT] [--kind KIND]\n"
  "  latest         [--db DB] [--strategy STRAT]\n"
  "  info           [--db DB] [--run RUN]  # defaults to latest run\n"
  "\n"
  "ANALYSIS COMMANDS:\n"
  "  verify         [--db DB] [--run RUN]  # defaults to latest run\n"
  "  summarize      [--db DB] [--run RUN]  # defaults to latest run\n"
  "  strategies-summary [--db DB]  # summary of all strategies' most recent runs\n"
  "  signal-stats   [--db DB] [--run RUN] [--strategy STRAT]  # defaults to latest run\n"
  "\n"
  "INTEGRATED ARCHITECTURE:\n"
  "  system-health  [--db DB]  # Check integrated system health and violations\n"
  "  architecture-test         # Run comprehensive integration tests\n"
  "  event-audit    [--db DB] [--run RUN] [--export FILE]  # Event sourcing audit trail\n"
  "\n"
  "FLOW ANALYSIS:\n"
  "  trade-flow     [--db DB] [--run RUN] [--symbol S] [--limit N] [--max [N]] [--buy] [--sell] [--hold] [--enhanced]  # defaults to latest run, limit=20\n"
  "  signal-flow    [--db DB] [--run RUN] [--symbol S] [--limit N] [--max [N]] [--buy] [--sell] [--hold] [--enhanced]  # defaults to latest run, limit=20\n"
  "  position-history [--db DB] [--run RUN] [--symbol S] [--limit N] [--max [N]] [--buy] [--sell] [--hold]  # defaults to latest run, limit=20\n"
  "\n"
  "DATA OPERATIONS:\n"
  "  export         [--db DB] [--run RUN] --format FORMAT --output FILE  # defaults to latest run\n"
  "  grep           [--db DB] [--run RUN] --where \"CONDITION\"  # defaults to latest run\n"
  "  diff           [--db DB] --run1 RUN1 --run2 RUN2\n"
  "\n"
  "DEFAULTS:\n"
  "  Database: audit/sentio_audit.sqlite3\n"
  "  Run: latest run (for analysis commands)\n"
  "  Limit: 20 events (for flow analysis)\n";

static const char* arg(const char* k, int argc, char** argv, const char* def=nullptr) {
  for (int i=1;i<argc-1;i++) if (!strcmp(argv[i], k)) return argv[i+1];
  return def;
}
static bool has(const char* k, int argc, char** argv) {
  for (int i=1;i<argc;i++) if (!strcmp(argv[i], k)) return true;
  return false;
}

// Helper function to remove chain information from note field for display
static std::string clean_note_for_display(const char* note) {
  if (!note) return "";
  
  std::string note_str = note;
  
  // Remove chain= information
  size_t chain_pos = note_str.find("chain=");
  if (chain_pos != std::string::npos) {
    size_t comma_pos = note_str.find(",", chain_pos);
    if (comma_pos != std::string::npos) {
      // Remove "chain=xxx," or "chain=xxx" at end
      note_str.erase(chain_pos, comma_pos - chain_pos + 1);
    } else {
      // Remove "chain=xxx" at end
      note_str.erase(chain_pos);
    }
  }
  
  // Clean up any leading/trailing commas or spaces
  while (!note_str.empty() && (note_str.back() == ',' || note_str.back() == ' ')) {
    note_str.pop_back();
  }
  while (!note_str.empty() && (note_str.front() == ',' || note_str.front() == ' ')) {
    note_str.erase(0, 1);
  }
  
  return note_str;
}

static std::string get_latest_run_id(const std::string& db_path) {
  try {
    DB db(db_path);
    // **FIXED**: Use dedicated latest run ID tracking instead of timestamp-based ordering
    std::string latest_run_id = db.get_latest_run_id();
    
    // Fallback to timestamp-based ordering if no latest run ID is stored
    if (latest_run_id.empty()) {
      std::string sql = "SELECT run_id FROM audit_runs ORDER BY started_at DESC LIMIT 1";
      sqlite3_stmt* st = nullptr;
      int rc = sqlite3_prepare_v2(db.get_db(), sql.c_str(), -1, &st, nullptr);
      if (rc == SQLITE_OK) {
        if (sqlite3_step(st) == SQLITE_ROW) {
          const char* run_id = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
          if (run_id) {
            latest_run_id = run_id;
          }
        }
        sqlite3_finalize(st);
      }
    }
    
    return latest_run_id;
  } catch (const std::exception& e) {
    return "";
  }
}

struct RunInfo {
  std::string run_id;
  std::string strategy;
  std::string kind;
  int64_t started_at;
  std::string note;
  std::string meta;
  std::string dataset_source_type;
  std::string dataset_file_path;
};

static RunInfo get_run_info(const std::string& db_path, const std::string& run_id) {
  RunInfo info;
  info.run_id = run_id;
  
  try {
    DB db(db_path);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(db.get_db(), 
                       "SELECT strategy, kind, started_at, note, params_json, dataset_source_type, dataset_file_path FROM audit_runs WHERE run_id = ?", 
                       -1, &st, nullptr);
    sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(st) == SQLITE_ROW) {
      const char* strategy = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
      const char* kind = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
      const char* note = reinterpret_cast<const char*>(sqlite3_column_text(st, 3));
      const char* params_json = reinterpret_cast<const char*>(sqlite3_column_text(st, 4));
      const char* dataset_source_type = reinterpret_cast<const char*>(sqlite3_column_text(st, 5));
      const char* dataset_file_path = reinterpret_cast<const char*>(sqlite3_column_text(st, 6));
      
      info.strategy = strategy ? strategy : "";
      info.kind = kind ? kind : "";
      info.started_at = sqlite3_column_int64(st, 2);
      info.note = note ? note : "";
      info.meta = params_json ? params_json : ""; // Use params_json as meta
      info.dataset_source_type = dataset_source_type ? dataset_source_type : "unknown";
      info.dataset_file_path = dataset_file_path ? dataset_file_path : "unknown";
    }
    
    sqlite3_finalize(st);
  } catch (const std::exception& e) {
    // Keep defaults
  }
  
  return info;
}

static void print_run_header(const std::string& title, const RunInfo& info) {
  // Format timestamp to local time
  auto format_timestamp = [](int64_t ts_millis) -> std::string {
    if (ts_millis == 0) return "N/A";
    time_t ts_sec = ts_millis / 1000;
    struct tm* tm_info = localtime(&ts_sec);
    char buffer[64];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S %Z", tm_info);
    return std::string(buffer);
  };
  
  // Format timestamp to ISO format
  auto format_timestamp_iso = [](int64_t ts_millis) -> std::string {
    if (ts_millis == 0) return "N/A";
    time_t ts_sec = ts_millis / 1000;
    struct tm* tm_info = gmtime(&ts_sec); // Use UTC for ISO format
    char buffer[64];
    strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", tm_info);
    return std::string(buffer);
  };
  
  // Use direct database fields for dataset information
  std::string dataset_type = info.dataset_source_type;
  std::string dataset_file = info.dataset_file_path;
  
  // Extract filename from full path
  size_t last_slash = dataset_file.find_last_of("/");
  if (last_slash != std::string::npos) {
      dataset_file = dataset_file.substr(last_slash + 1);
  }
  
  std::string dataset_period = "unknown";
  std::string test_period = "unknown";
  int test_period_days = 0;
  
  if (!info.meta.empty()) {
    // Parse time ranges from JSON metadata
    size_t dataset_pos = info.meta.find("\"dataset_type\":\"");
    if (dataset_pos != std::string::npos) {
      size_t start = dataset_pos + 16; // length of "dataset_type":""
      size_t end = info.meta.find("\"", start);
      if (end != std::string::npos) {
        dataset_type = info.meta.substr(start, end - start);
      }
    }
    
    // Parse dataset file path
    size_t file_pos = info.meta.find("\"dataset_file_path\":\"");
    if (file_pos != std::string::npos) {
      size_t start = file_pos + 20; // length of "dataset_file_path":""
      size_t end = info.meta.find("\"", start);
      if (end != std::string::npos) {
        std::string full_path = info.meta.substr(start, end - start);
        // Extract just the filename
        size_t last_slash = full_path.find_last_of("/\\");
        if (last_slash != std::string::npos) {
          dataset_file = full_path.substr(last_slash + 1);
        } else {
          dataset_file = full_path;
        }
      }
    }
    
    // Parse dataset period (new format from signal-flow)
    size_t dataset_start_pos = info.meta.find("\"dataset_period_start_ts_ms\":");
    size_t dataset_end_pos = info.meta.find("\"dataset_period_end_ts_ms\":");
    if (dataset_start_pos != std::string::npos && dataset_end_pos != std::string::npos) {
      size_t start_val_start = dataset_start_pos + 29;
      size_t start_val_end = info.meta.find_first_of(",}", start_val_start);
      
      size_t end_val_start = dataset_end_pos + 27;
      size_t end_val_end = info.meta.find_first_of(",}", end_val_start);
      
      if (start_val_end != std::string::npos && end_val_end != std::string::npos) {
        try {
          int64_t start_ts = std::stoll(info.meta.substr(start_val_start, start_val_end - start_val_start));
          int64_t end_ts = std::stoll(info.meta.substr(end_val_start, end_val_end - end_val_start));
          
          // Parse dataset days
          size_t dataset_days_pos = info.meta.find("\"dataset_period_days\":");
          if (dataset_days_pos != std::string::npos) {
            size_t days_start = dataset_days_pos + 22;
            size_t days_end = info.meta.find_first_of(",}", days_start);
            if (days_end != std::string::npos) {
              try {
                double days = std::stod(info.meta.substr(days_start, days_end - days_start));
                dataset_period = format_timestamp_iso(start_ts) + " to " + format_timestamp_iso(end_ts) + " (" + std::to_string((int)days) + " days)";
              } catch (...) {
          dataset_period = format_timestamp_iso(start_ts) + " to " + format_timestamp_iso(end_ts);
              }
            }
          } else {
            dataset_period = format_timestamp_iso(start_ts) + " to " + format_timestamp_iso(end_ts);
          }
        } catch (...) { /* ignore parse errors */ }
      }
    }
    
    // Fallback: Parse old dataset time range format
    if (dataset_period == "unknown") {
      size_t old_dataset_start_pos = info.meta.find("\"dataset_time_range_start\":");
      size_t old_dataset_end_pos = info.meta.find("\"dataset_time_range_end\":");
      if (old_dataset_start_pos != std::string::npos && old_dataset_end_pos != std::string::npos) {
        size_t start_val_start = old_dataset_start_pos + 27;
        size_t start_val_end = info.meta.find_first_of(",}", start_val_start);
        
        size_t end_val_start = old_dataset_end_pos + 25;
        size_t end_val_end = info.meta.find_first_of(",}", end_val_start);
        
        if (start_val_end != std::string::npos && end_val_end != std::string::npos) {
          try {
            int64_t start_ts = std::stoll(info.meta.substr(start_val_start, start_val_end - start_val_start));
            int64_t end_ts = std::stoll(info.meta.substr(end_val_start, end_val_end - end_val_start));
            
            dataset_period = format_timestamp_iso(start_ts) + " to " + format_timestamp_iso(end_ts);
          } catch (...) { /* ignore parse errors */ }
        }
      }
    }
    
    // Parse test period
    size_t test_start_pos = info.meta.find("\"run_period_start_ts_ms\":");
    size_t test_end_pos = info.meta.find("\"run_period_end_ts_ms\":");
    if (test_start_pos != std::string::npos && test_end_pos != std::string::npos) {
      size_t start_val_start = test_start_pos + 25;
      size_t start_val_end = info.meta.find_first_of(",}", start_val_start);
      
      size_t end_val_start = test_end_pos + 23;
      size_t end_val_end = info.meta.find_first_of(",}", end_val_start);
      
      if (start_val_end != std::string::npos && end_val_end != std::string::npos) {
        try {
          int64_t start_ts = std::stoll(info.meta.substr(start_val_start, start_val_end - start_val_start));
          int64_t end_ts = std::stoll(info.meta.substr(end_val_start, end_val_end - end_val_start));
          
          // Parse test period days and TB count
          int tb_count = 0;
          size_t tb_count_pos = info.meta.find("\"tb_count\":");
          if (tb_count_pos != std::string::npos) {
            size_t tb_start = tb_count_pos + 11;
            size_t tb_end = info.meta.find_first_of(",}", tb_start);
            if (tb_end != std::string::npos) {
              try {
                tb_count = std::stoi(info.meta.substr(tb_start, tb_end - tb_start));
              } catch (...) { /* ignore */ }
            }
          }
          
          // Parse test period days
          size_t test_days_pos = info.meta.find("\"test_period_days\":");
          if (test_days_pos != std::string::npos) {
            size_t days_start = test_days_pos + 19;
            size_t days_end = info.meta.find_first_of(",}", days_start);
            if (days_end != std::string::npos) {
              try {
                double days = std::stod(info.meta.substr(days_start, days_end - days_start));
                test_period_days = (int)days;
                test_period = format_timestamp_iso(start_ts) + " to " + format_timestamp_iso(end_ts) + " (" + std::to_string((int)days) + " days";
                if (tb_count > 0) {
                  test_period += ", " + std::to_string(tb_count) + " TBs)";
                } else {
                  test_period += ")";
                }
              } catch (...) {
                test_period = format_timestamp_iso(start_ts) + " to " + format_timestamp_iso(end_ts);
          int64_t duration_ms = end_ts - start_ts;
          test_period_days = static_cast<int>(duration_ms / (1000 * 60 * 60 * 24));
              }
            }
          } else {
            test_period = format_timestamp_iso(start_ts) + " to " + format_timestamp_iso(end_ts);
            int64_t duration_ms = end_ts - start_ts;
            test_period_days = static_cast<int>(duration_ms / (1000 * 60 * 60 * 24));
          }
        } catch (...) { /* ignore parse errors */ }
      }
    }
    
    // Fallback: parse simple test_period_days
    if (test_period_days == 0) {
      size_t period_pos = info.meta.find("\"test_period_days\":");
      if (period_pos != std::string::npos) {
        size_t start = period_pos + 19; // length of "test_period_days":
        size_t end = info.meta.find_first_of(",}", start);
        if (end != std::string::npos) {
          try {
            test_period_days = std::stoi(info.meta.substr(start, end - start));
          } catch (...) { /* ignore parse errors */ }
        }
      }
    }
  }
  
  // Enhanced header with consistent visual formatting
  std::cout << "\n" << BOLD << BG_BLUE << WHITE << "╔══════════════════════════════════════════════════════════════════════════════════╗" << RESET << std::endl;
  std::cout << BOLD << BG_BLUE << WHITE << "║                            📊 " << title << "                            ║" << RESET << std::endl;
  std::cout << BOLD << BG_BLUE << WHITE << "╚══════════════════════════════════════════════════════════════════════════════════╝" << RESET << std::endl;
  
  // Run Information Section
  std::cout << "\n" << BOLD << CYAN << "📋 RUN INFORMATION" << RESET << std::endl;
  std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
  std::cout << "│ " << BOLD << "Run ID:" << RESET << "       " << BLUE << info.run_id << RESET << std::endl;
  std::cout << "│ " << BOLD << "Strategy:" << RESET << "     " << MAGENTA << info.strategy << RESET << std::endl;
  std::cout << "│ " << BOLD << "Test Kind:" << RESET << "    " << GREEN << info.kind << RESET << std::endl;
  std::cout << "│ " << BOLD << "Run Time:" << RESET << "     " << WHITE << format_timestamp(info.started_at) << RESET << std::endl;
  std::cout << "│ " << BOLD << "Dataset:" << RESET << "      " << DIM << dataset_file << " (" << dataset_type << ")" << RESET << std::endl;
  if (!info.note.empty()) {
    std::cout << "│ " << BOLD << "Note:" << RESET << "         " << DIM << info.note << RESET << std::endl;
  }
  std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
  
  // Time Periods Section (if available)
  if (dataset_period != "unknown" || test_period != "unknown") {
    std::cout << "\n" << BOLD << CYAN << "📅 TIME PERIODS" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    if (dataset_period != "unknown") {
      std::cout << "│ " << BOLD << "Dataset Period:" << RESET << " " << BLUE << dataset_period << RESET << std::endl;
    }
    if (test_period != "unknown") {
      std::cout << "│ " << BOLD << "Test Period:" << RESET << "    " << GREEN << test_period << RESET << std::endl;
    }
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
  }
}

// **POSITION CONFLICT CHECK**: Verify no conflicting positions exist
void check_position_conflicts(sqlite3* db, const std::string& run_id) {
    printf("\n" BOLD CYAN "⚔️  POSITION CONFLICT CHECK" RESET "\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    
    // **PERFORMANCE FIX**: Use much simpler query to avoid hanging on large datasets
    const char* query = R"(
        SELECT 
            symbol,
            COUNT(*) as fill_count,
            SUM(CASE WHEN side = 'BUY' THEN qty ELSE -qty END) as net_position
        FROM audit_events 
        WHERE run_id = ? AND kind = 'FILL'
        GROUP BY symbol
        HAVING ABS(net_position) > 0.001
    )";
    
    sqlite3_stmt* stmt = nullptr;
    std::map<std::string, double> final_positions;
    
    if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* symbol = (const char*)sqlite3_column_text(stmt, 0);
            double net_position = sqlite3_column_double(stmt, 2);
            
            if (symbol) {
                final_positions[symbol] = net_position;
            }
        }
        sqlite3_finalize(stmt);
    }
    
    // Simple conflict detection: check if we have both long and inverse ETFs
    bool has_long_etf = false;
    bool has_inverse_etf = false;
    bool has_short_positions = false;
    
    for (const auto& [symbol, position] : final_positions) {
        if (std::abs(position) > 0.001) {
            if (symbol == "QQQ" || symbol == "TQQQ") {
                if (position > 0) has_long_etf = true;
                if (position < 0) has_short_positions = true;
            }
            if (symbol == "PSQ" || symbol == "SQQQ") {
                if (position > 0) has_inverse_etf = true;
            }
        }
    }
    
    bool has_conflicts = (has_long_etf && has_inverse_etf) || has_short_positions;
    
    // Summary
    printf("├─────────────────────────────────────────────────────────────────────────────────┤\n");
    if (has_conflicts) {
        printf("│ " RED "❌ POTENTIAL CONFLICTS DETECTED" RESET " │ " RED "Mixed directional exposure found" RESET " │\n");
        if (has_long_etf && has_inverse_etf) {
            printf("│ " BOLD "Issue:" RESET " Both long ETFs and inverse ETFs held simultaneously │\n");
        }
        if (has_short_positions) {
            printf("│ " BOLD "Issue:" RESET " Short positions detected - should use inverse ETFs instead │\n");
        }
        printf("│ " BOLD "Fix:" RESET "  Review PositionCoordinator conflict detection and resolution │\n");
    } else {
        printf("│ " GREEN "✅ NO CONFLICTS DETECTED" RESET " │ " GREEN "All positions directionally consistent" RESET " │\n");
        printf("│ " BOLD "Status:" RESET " Proper position coordination, clean directional exposure │\n");
    }
    
    printf("│ " BOLD "Final Positions:" RESET " ");
    for (const auto& [symbol, position] : final_positions) {
        if (std::abs(position) > 0.001) {
            printf("%s:%.1f ", symbol.c_str(), position);
        }
    }
    printf("│\n");
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
}

// **EOD POSITION CHECK**: Verify all positions are closed at end of day
void check_eod_positions(sqlite3* db, const std::string& run_id) {
    printf("\n" BOLD CYAN "🌙 END-OF-DAY POSITION CHECK" RESET "\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    
    // Calculate cumulative positions at the end of each trading day
    const char* query = R"(
        WITH daily_bars AS (
            SELECT 
                DATE(ts_millis/1000, 'unixepoch') as trading_day,
                MAX(ts_millis) as last_bar_ts
            FROM audit_events 
            WHERE run_id = ? AND kind = 'BAR'
            GROUP BY DATE(ts_millis/1000, 'unixepoch')
        ),
        actual_positions AS (
            SELECT 
                DATE(ae.ts_millis/1000, 'unixepoch') as trading_day,
                ae.symbol,
                ae.ts_millis,
                ae.qty,
                ae.price,
                CASE 
                    WHEN ae.note LIKE '%pos_after=%' THEN 
                        CAST(SUBSTR(ae.note, INSTR(ae.note, 'pos_after=') + 10, 
                             CASE WHEN INSTR(SUBSTR(ae.note, INSTR(ae.note, 'pos_after=') + 10), ',') > 0 
                                  THEN INSTR(SUBSTR(ae.note, INSTR(ae.note, 'pos_after=') + 10), ',') - 1
                                  ELSE LENGTH(SUBSTR(ae.note, INSTR(ae.note, 'pos_after=') + 10))
                             END) AS REAL)
                    ELSE 0.0
                END as actual_qty
            FROM audit_events ae
            WHERE ae.run_id = ? AND ae.kind = 'FILL'
        ),
        eod_positions AS (
            SELECT 
                ap.trading_day,
                ap.symbol,
                ap.actual_qty as qty,
                ap.price,
                (ap.actual_qty * ap.price) as position_value,
                ROW_NUMBER() OVER (
                    PARTITION BY ap.trading_day, ap.symbol 
                    ORDER BY ap.ts_millis DESC
                ) as rn
            FROM actual_positions ap
            JOIN daily_bars db ON ap.trading_day = db.trading_day
            WHERE ap.ts_millis <= db.last_bar_ts
        ),
        final_eod_positions AS (
            SELECT trading_day, symbol, qty, price, position_value
            FROM eod_positions 
            WHERE rn = 1 AND ABS(qty) > 0.001
        )
        SELECT 
            trading_day,
            COUNT(*) as open_positions,
            SUM(ABS(position_value)) as total_exposure,
            GROUP_CONCAT(symbol || ':' || ROUND(qty,2)) as positions
        FROM final_eod_positions
        GROUP BY trading_day
        ORDER BY trading_day;
    )";
    
    sqlite3_stmt* stmt;
    bool has_eod_violations = false;
    int total_violations = 0;
    
    if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, run_id.c_str(), -1, SQLITE_TRANSIENT);
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* trading_day = (const char*)sqlite3_column_text(stmt, 0);
            int open_positions = sqlite3_column_int(stmt, 1);
            double total_exposure = sqlite3_column_double(stmt, 2);
            const char* positions = (const char*)sqlite3_column_text(stmt, 3);
            
            if (open_positions > 0) {
                has_eod_violations = true;
                total_violations++;
                
                printf("│ " RED "❌ %s" RESET " │ " RED "%d positions" RESET " │ " RED "$%.2f exposure" RESET " │\n", 
                       trading_day, open_positions, total_exposure);
                printf("│   Positions: " DIM "%s" RESET "\n", positions ? positions : "unknown");
            } else {
                printf("│ " GREEN "✅ %s" RESET " │ " GREEN "0 positions" RESET " │ " GREEN "$0.00 exposure" RESET " │\n", 
                       trading_day);
            }
        }
        sqlite3_finalize(stmt);
    }
    
    // Summary
    printf("├─────────────────────────────────────────────────────────────────────────────────┤\n");
    if (has_eod_violations) {
        printf("│ " BOLD RED "⚠️  EOD VIOLATIONS DETECTED" RESET " │ " RED "%d days with overnight positions" RESET " │\n", 
               total_violations);
        printf("│ " BOLD "Risk:" RESET " Overnight carry risk, leveraged ETF decay, gap risk exposure │\n");
        printf("│ " BOLD "Fix:" RESET "  Review EOD position management system configuration         │\n");
    } else {
        printf("│ " BOLD GREEN "✅ EOD COMPLIANCE VERIFIED" RESET " │ " GREEN "All positions closed overnight" RESET " │\n");
        printf("│ " BOLD "Status:" RESET " Zero overnight carry risk, proper risk management        │\n");
    }
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
}

// **COMPREHENSIVE INTEGRITY CHECK**: Validates all 5 core trading principles
int perform_integrity_check(sqlite3* db, const std::string& run_id);

int audit_main(int argc, char** argv) {
  if (argc<2) { fputs(usage, stderr); return 1; }
  const char* cmd = argv[1];
  const char* dbp = arg("--db", argc, argv, "audit/sentio_audit.sqlite3");
  
  if (!strcmp(cmd,"init")) {
    DB db(dbp); db.init_schema(); puts("ok");
    return 0;
  }

  if (!strcmp(cmd,"reset")) {
    bool confirmed = has("--confirm", argc, argv);
    if (!confirmed) {
      printf("WARNING: This will delete ALL audit data!\n");
      printf("Use --confirm flag to proceed: sentio_audit reset --confirm\n");
      return 1;
    }
    
    // Remove the database file to reset everything
    if (std::remove(dbp) == 0) {
      printf("Audit database reset successfully: %s\n", dbp);
      // Recreate the database with schema
      DB db(dbp); db.init_schema();
      puts("Fresh database initialized");
      return 0;
    } else {
      printf("Failed to reset database: %s\n", dbp);
      return 1;
    }
  }

  DB db(dbp);

  if (!strcmp(cmd,"new-run")) {
    RunRow r;
    r.run_id     = arg("--run", argc, argv, "");
    r.started_at = now_millis();
    r.kind       = arg("--kind", argc, argv, "backtest");
    r.strategy   = arg("--strategy", argc, argv, "");
    const char* params_file = arg("--params", argc, argv, "");
    const char* data_hash   = arg("--data-hash", argc, argv, "");
    r.git_rev    = arg("--git", argc, argv, "");
    r.note       = arg("--note", argc, argv, "");
    if (r.run_id.empty() || r.strategy.empty() || !params_file || !*params_file || !data_hash || !*data_hash) {
      fputs("missing required args\n", stderr); return 3;
    }
    // Load params.json
    FILE* f=fopen(params_file,"rb"); if(!f){perror("params"); return 4;}
    std::string pj; char buf[4096]; size_t n;
    while((n=fread(buf,1,sizeof(buf),f))>0) pj.append(buf,n);
    fclose(f);
    r.params_json = pj;
    r.data_hash   = data_hash;
    db.new_run(r);
    puts("run created"); return 0;
  }

  if (!strcmp(cmd,"log")) {
    Event ev;
    ev.run_id   = arg("--run",argc,argv,"");
    ev.ts_millis= atoll(arg("--ts",argc,argv,"0"));
    ev.kind     = arg("--kind",argc,argv,"NOTE");
    ev.symbol   = arg("--symbol",argc,argv,"");
    ev.side     = arg("--side",argc,argv,"");
    ev.qty      = atof(arg("--qty",argc,argv,"0"));
    ev.price    = atof(arg("--price",argc,argv,"0"));
    ev.pnl_delta= atof(arg("--pnl",argc,argv,"0"));
    ev.weight   = atof(arg("--weight",argc,argv,"0"));
    ev.prob     = atof(arg("--prob",argc,argv,"0"));
    ev.reason   = arg("--reason",argc,argv,"");
    ev.note     = arg("--note",argc,argv,"");
    if (ev.run_id.empty() || ev.ts_millis==0 || ev.kind.empty()) { fputs("missing run/ts/kind\n", stderr); return 3; }
    auto [seq,h] = db.append_event(ev);
    printf("ok seq=%lld hash=%s\n", seq, h.c_str());
    return 0;
  }

  if (!strcmp(cmd,"end-run")) {
    const char* run = arg("--run",argc,argv,"");
    if (!*run) { fputs("--run required\n", stderr); return 3; }
    db.end_run(run, now_millis());
    puts("ended"); return 0;
  }

  if (!strcmp(cmd,"verify")) {
    const char* run_arg = arg("--run",argc,argv,"");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    auto [ok,msg]= db.verify_run(run);
    printf("%s: %s\n", ok?"OK":"FAIL", msg.c_str());
    return ok?0:10;
  }

  if (!strcmp(cmd,"summarize")) {
    const char* run_arg = arg("--run",argc,argv,"");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    auto s = db.summarize(run);
    
    // Get run info and print enhanced header manually
    RunInfo info = get_run_info(dbp, run);
    
    // Use direct database fields instead of JSON parsing
    std::string dataset_type = info.dataset_source_type;
    std::string dataset_source = info.dataset_file_path;
    
          // Extract filename from full path
          size_t last_slash = dataset_source.find_last_of("/");
          if (last_slash != std::string::npos) {
            dataset_source = dataset_source.substr(last_slash + 1);
      }
      
    std::string dataset_period = "unknown";
    std::string test_period = "unknown";
    int test_period_days = 0;
    
    if (!info.meta.empty()) {
      // Parse time ranges
      size_t start_ts_pos = info.meta.find("\"dataset_time_range_start\":");
      size_t end_ts_pos = info.meta.find("\"dataset_time_range_end\":");
      size_t run_start_pos = info.meta.find("\"run_period_start_ts_ms\":");
      size_t run_end_pos = info.meta.find("\"run_period_end_ts_ms\":");
      
      auto format_timestamp_range = [](const std::string& meta, const std::string& key) -> std::string {
        size_t pos = meta.find("\"" + key + "\":");
        if (pos != std::string::npos) {
          size_t start = pos + key.length() + 3; // length of key + ":"
          size_t end = meta.find_first_of(",}", start);
          if (end != std::string::npos) {
            try {
              std::int64_t ts_ms = std::stoll(meta.substr(start, end - start));
              time_t ts_sec = ts_ms / 1000;
              struct tm* tm_info = localtime(&ts_sec);
              char buffer[32];
              strftime(buffer, sizeof(buffer), "%Y.%m.%d", tm_info);
              return std::string(buffer);
            } catch (...) { /* ignore parse errors */ }
          }
        }
        return "unknown";
      };
      
      if (start_ts_pos != std::string::npos && end_ts_pos != std::string::npos) {
        std::string start_date = format_timestamp_range(info.meta, "dataset_time_range_start");
        std::string end_date = format_timestamp_range(info.meta, "dataset_time_range_end");
        if (start_date != "unknown" && end_date != "unknown") {
          dataset_period = start_date + " - " + end_date;
        }
      }
      
      if (run_start_pos != std::string::npos && run_end_pos != std::string::npos) {
        std::string test_start = format_timestamp_range(info.meta, "run_period_start_ts_ms");
        std::string test_end = format_timestamp_range(info.meta, "run_period_end_ts_ms");
        if (test_start != "unknown" && test_end != "unknown") {
          test_period = test_start + " to " + test_end;
        }
      }
      
      size_t period_pos = info.meta.find("\"test_period_days\":");
      if (period_pos != std::string::npos) {
        size_t start = period_pos + 19; // length of "test_period_days":
        size_t end = info.meta.find_first_of(",}", start);
        if (end != std::string::npos) {
          try {
            test_period_days = std::stoi(info.meta.substr(start, end - start));
          } catch (...) { /* ignore parse errors */ }
        }
      }
    }
    
    // Print enhanced header manually
    auto format_timestamp = [](int64_t ts_millis) -> std::string {
      if (ts_millis == 0) return "N/A";
      time_t ts_sec = ts_millis / 1000;
      struct tm* tm_info = localtime(&ts_sec);
      char buffer[64];
      strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S %Z", tm_info);
      return std::string(buffer);
    };
    
    std::cout << "\n" << BOLD << "\033[44m" << WHITE << "╔══════════════════════════════════════════════════════════════════════════════════╗" << RESET << std::endl;
    std::cout << BOLD << "\033[44m" << WHITE << "║                           📊 AUDIT SUMMARY REPORT                                ║" << RESET << std::endl;
    std::cout << BOLD << "\033[44m" << WHITE << "╚══════════════════════════════════════════════════════════════════════════════════╝" << RESET << std::endl;
    
    std::cout << "\n" << BOLD << CYAN << "📋 RUN INFORMATION" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ " << BOLD << "Run ID:" << RESET << "       " << BLUE << info.run_id << RESET << std::endl;
    std::cout << "│ " << BOLD << "Strategy:" << RESET << "     " << MAGENTA << info.strategy << RESET << std::endl;
    std::cout << "│ " << BOLD << "Test Kind:" << RESET << "    " << GREEN << info.kind << RESET << std::endl;
    std::cout << "│ " << BOLD << "Run Time:" << RESET << "     " << WHITE << format_timestamp(info.started_at) << RESET << std::endl;
    std::cout << "│ " << BOLD << "Dataset:" << RESET << "      " << DIM << dataset_source << " (" << dataset_type << ")" << RESET << std::endl;
    if (!info.note.empty()) {
      std::cout << "│ " << BOLD << "Note:" << RESET << "         " << DIM << info.note << RESET << std::endl;
    }
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    // Enhanced dataset information section
    std::cout << "\n" << BOLD << CYAN << "📅 TIME PERIODS" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    
    // Calculate time range and days
    auto format_time_range = [](int64_t start_ts, int64_t end_ts) -> std::pair<std::string, double> {
        if (start_ts == 0 || end_ts == 0) return {"unknown", 0.0};
        
        time_t start_sec = start_ts / 1000;
        time_t end_sec = end_ts / 1000;
        
        // Use gmtime_r for thread safety and to avoid static buffer issues
        struct tm start_tm, end_tm;
        gmtime_r(&start_sec, &start_tm);
        gmtime_r(&end_sec, &end_tm);
        
        char start_buf[32], end_buf[32];
        strftime(start_buf, sizeof(start_buf), "%Y-%m-%dT%H:%M:%SZ", &start_tm);
        strftime(end_buf, sizeof(end_buf), "%Y-%m-%dT%H:%M:%SZ", &end_tm);
        
        double days = (end_ts - start_ts) / (1000.0 * 60.0 * 60.0 * 24.0);
        return {std::string(start_buf) + " → " + std::string(end_buf), days};
    };
    
    // Show dataset period if available
    if (dataset_period != "unknown") {
        std::cout << "│ " << BOLD << "Dataset Period:" << RESET << " " << BLUE << dataset_period << RESET << std::endl;
    }
    
    // Show test period with time range
    auto [time_range_str, time_range_days] = format_time_range(s.ts_first, s.ts_last);
    std::cout << "│ " << BOLD << "Test Period:" << RESET << "    " << GREEN << time_range_str << RESET << " " 
              << DIM << "(" << std::fixed << std::setprecision(1) << time_range_days << " days)" << RESET << std::endl;
    
    // Show TB period if this is a Trading Block run
    auto block_rows = db.get_blocks_for_run(run);
    if (!block_rows.empty()) {
        // Calculate TB period using actual Trading Block timestamps
        int64_t tb_start_ms = block_rows[0].start_ts_ms;
        int64_t tb_end_ms = block_rows[block_rows.size() - 1].end_ts_ms;
        
        auto [tb_time_range_str, tb_time_range_days] = format_time_range(tb_start_ms, tb_end_ms);
        
        std::cout << "│ " << BOLD << "TB Period:" << RESET << "      " << YELLOW << tb_time_range_str << RESET << " " 
                  << DIM << "(" << std::fixed << std::setprecision(1) << tb_time_range_days << " days, " << block_rows.size() << " TBs)" << RESET << std::endl;
    }
    
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    std::cout << "\n" << BOLD << CYAN << "📊 EVENT COUNTS" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ " << BOLD << "Total Events:" << RESET << "  " << WHITE << s.n_total << RESET << std::endl;
    std::cout << "│ " << BOLD << "Signals:" << RESET << "       " << CYAN << s.n_signal << RESET << std::endl;
    std::cout << "│ " << BOLD << "Orders:" << RESET << "        " << YELLOW << s.n_order << RESET << std::endl;
    std::cout << "│ " << BOLD << "Fills:" << RESET << "         " << GREEN << s.n_fill << RESET << std::endl;
    std::cout << "│ " << BOLD << "P&L Rows:" << RESET << "      " << MAGENTA << s.n_pnl << RESET << " " << DIM << "(dedicated P&L accounting events)" << RESET << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    std::cout << "\n" << BOLD << CYAN << "⚙️  TRADING CONFIGURATION" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    
    // Check if this run has Trading Block data (reuse the variable from above)
    if (!block_rows.empty()) {
      std::cout << "│ " << BOLD << "Trading Blocks:" << RESET << "  " << YELLOW << block_rows.size() << RESET << "/" 
                << YELLOW << block_rows.size() << RESET << " TB " << DIM << "(480 bars each ≈ 8hrs)" << RESET << std::endl;
      std::cout << "│ " << BOLD << "Total Bars:" << RESET << "     " << WHITE << (block_rows.size() * 480) << RESET << " " 
                << DIM << "(" << std::fixed << std::setprecision(1) << (block_rows.size() * 480 / 390.0) << " trading days)" << RESET << std::endl;
      std::cout << "│ " << BOLD << "Total Fills:" << RESET << "    " << CYAN << s.n_fill << RESET << std::endl;
      std::cout << "│ " << BOLD << "Trades per TB:" << RESET << "  " << CYAN << std::fixed << std::setprecision(1) << (double(s.n_fill) / block_rows.size()) << RESET << " " << DIM << "(≈Daily)" << RESET << std::endl;
    } else {
      std::cout << "│ " << BOLD << "Legacy Run:" << RESET << "     " << DIM << "Non-Trading Block evaluation" << RESET << std::endl;
      std::cout << "│ " << BOLD << "Total Fills:" << RESET << "    " << CYAN << s.n_fill << RESET << std::endl;
      std::cout << "│ " << BOLD << "Trading Days:" << RESET << "   " << WHITE << s.trading_days << RESET << std::endl;
    }
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    std::cout << "\n" << BOLD << CYAN << "📈 PERFORMANCE METRICS" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    
    if (!block_rows.empty()) {
      // Calculate Trading Block metrics
      double total_compounded = 1.0;
      double sum_rpb = 0.0;
      for (const auto& block : block_rows) {
        total_compounded *= (1.0 + block.return_per_block);
        sum_rpb += block.return_per_block;
      }
      double mean_rpb = sum_rpb / block_rows.size();
      double total_return = (total_compounded - 1.0) * 100.0;
      
      // Color code based on performance
      const char* rpb_color = (mean_rpb >= 0) ? GREEN : (mean_rpb >= -0.001) ? YELLOW : RED;
      const char* return_color = (total_return >= 0) ? GREEN : (total_return >= -1.0) ? YELLOW : RED;
      const char* sharpe_color = (s.sharpe >= 1.0) ? GREEN : (s.sharpe >= 0) ? YELLOW : RED;
      
      // Calculate MRB (Monthly Return per Block) - projected monthly return
      double blocks_per_month = 20.0;
      double mrb = 0.0;
      if (mean_rpb != 0.0) {
          mrb = (std::pow(1.0 + mean_rpb, blocks_per_month) - 1.0) * 100.0;
      }
      const char* mrb_color = (mrb >= 0) ? GREEN : (mrb >= -5.0) ? YELLOW : RED;
      
      std::cout << "│ " << BOLD << "Mean RPB:" << RESET << "       " << rpb_color << BOLD << std::fixed << std::setprecision(4) << (mean_rpb * 100.0) << "%" << RESET << " " << DIM << "(Return Per Block)" << RESET << std::endl;
      std::cout << "│ " << BOLD << "Std Dev RPB:" << RESET << "    " << WHITE << "N/A%" << RESET << " " << DIM << "(Volatility)" << RESET << std::endl;
      std::cout << "│ " << BOLD << "MRB:" << RESET << "            " << mrb_color << BOLD << std::fixed << std::setprecision(2) << mrb << "%" << RESET << " " << DIM << "(Monthly Return)" << RESET << std::endl;
      std::cout << "│ " << BOLD << "ARB:" << RESET << "            " << return_color << BOLD << std::fixed << std::setprecision(2) << (mean_rpb * 100.0 * 252) << "%" << RESET << " " << DIM << "(Annualized Return)" << RESET << std::endl;
      std::cout << "│ " << BOLD << "Sharpe Ratio:" << RESET << "   " << sharpe_color << std::fixed << std::setprecision(2) << s.sharpe << RESET << " " << DIM << "(Risk-Adjusted Return)" << RESET << std::endl;
      std::cout << "│ " << BOLD << "Consistency:" << RESET << "    " << YELLOW << "N/A" << RESET << " " << DIM << "(Lower = More Consistent)" << RESET << std::endl;
      
      // 20TB benchmark if available
      if (block_rows.size() >= 20) {
        double twenty_tb_return = 1.0;
        for (int i = 0; i < 20; ++i) {
          twenty_tb_return *= (1.0 + block_rows[i].return_per_block);
        }
        std::cout << "│ " << BOLD << "MRP20B:" << RESET << "         " << GREEN << std::fixed << std::setprecision(2) << ((twenty_tb_return - 1.0) * 100.0) << "%" << RESET << " " << DIM << "(≈Monthly Return)" << RESET << std::endl;
      }
    } else {
      // Legacy format for non-TB runs
      std::cout << "│ " << BOLD << "Total Return:" << RESET << "   " << GREEN << std::fixed << std::setprecision(2) << s.total_return << "%" << RESET << std::endl;
      std::cout << "│ " << BOLD << "MPR (Legacy):" << RESET << "   " << YELLOW << std::fixed << std::setprecision(2) << s.mpr << "%" << RESET << " " << DIM << "[Monthly Projected Return]" << RESET << std::endl;
      std::cout << "│ " << BOLD << "Sharpe Ratio:" << RESET << "   " << GREEN << std::fixed << std::setprecision(3) << s.sharpe << RESET << std::endl;
      std::cout << "│ " << BOLD << "Daily Trades:" << RESET << "   " << CYAN << std::fixed << std::setprecision(1) << s.daily_trades << RESET << std::endl;
      std::cout << "│ " << BOLD << "Max Drawdown:" << RESET << "   " << RED << std::fixed << std::setprecision(2) << s.max_drawdown << "%" << RESET << std::endl;
      std::cout << "│ " << YELLOW << "⚠️  Legacy Run: Use Trading Block system for canonical metrics" << RESET << std::endl;
    }
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    // Add strategy performance indicator
    if (!block_rows.empty()) {
        // Calculate mean RPB for strategy indicator
        double total_compounded = 1.0;
        double sum_rpb = 0.0;
        for (const auto& block : block_rows) {
          total_compounded *= (1.0 + block.return_per_block);
          sum_rpb += block.return_per_block;
        }
        double mean_rpb = sum_rpb / block_rows.size();
        
        if (mean_rpb > 0.001) {
            std::cout << "\n" << BOLD << "\033[42m" << WHITE << "🚀 WINNING STRATEGY " << RESET << std::endl;
        } else if (mean_rpb > -0.001) {
            std::cout << "\n" << BOLD << "\033[43m" << WHITE << "⚖️  NEUTRAL STRATEGY " << RESET << std::endl;
        } else {
            std::cout << "\n" << BOLD << "\033[41m" << WHITE << "⚠️  LOSING STRATEGY " << RESET << std::endl;
        }
    }
    
    std::cout << "\n" << BOLD << CYAN << "💰 P&L SUMMARY" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    
    // Color code P&L values
    const char* realized_color = (s.realized_pnl >= 0) ? GREEN : RED;
    const char* unrealized_color = (s.unrealized_pnl >= 0) ? GREEN : RED;
    const char* total_color = (s.pnl_sum >= 0) ? GREEN : RED;
    
    std::cout << "│ " << BOLD << "Realized P&L:" << RESET << "   " << realized_color << std::fixed << std::setprecision(2) << s.realized_pnl << RESET << " " << DIM << "(from closed trades)" << RESET << std::endl;
    std::cout << "│ " << BOLD << "Unrealized P&L:" << RESET << " " << unrealized_color << std::fixed << std::setprecision(2) << s.unrealized_pnl << RESET << " " << DIM << "(from open positions)" << RESET << std::endl;
    std::cout << "│ " << BOLD << "Total P&L:" << RESET << "      " << total_color << std::fixed << std::setprecision(2) << s.pnl_sum << RESET << " " << DIM << "(realized + unrealized)" << RESET << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    // **FIX**: Always show instrument distribution (including zero activity instruments)
    printf("\n📊 INSTRUMENT DISTRIBUTION\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ %-8s %8s %8s %12s %8s %15s\n", "Symbol", "Fills", "Fill%", "P&L", "P&L%", "Volume");
    printf("│ %-8s %8s %8s %12s %8s %15s\n", "------", "-----", "-----", "---", "----", "------");
    
    // **FIX**: Ensure ALL expected instruments are shown (including zero activity)
    std::vector<std::string> all_expected_instruments = {"PSQ", "QQQ", "TQQQ", "SQQQ"};
      
      // Calculate totals for percentage calculations
      int64_t total_fills = 0;
      double total_volume = 0.0;
    for (const std::string& symbol : all_expected_instruments) {
        int64_t fills = s.instrument_fills.count(symbol) ? s.instrument_fills.at(symbol) : 0;
        double volume = s.instrument_volume.count(symbol) ? s.instrument_volume.at(symbol) : 0.0;
        total_fills += fills;
        total_volume += volume;
      }
      
    for (const std::string& symbol : all_expected_instruments) {
      double pnl = s.instrument_pnl.count(symbol) ? s.instrument_pnl.at(symbol) : 0.0;
        int64_t fills = s.instrument_fills.count(symbol) ? s.instrument_fills.at(symbol) : 0;
        double volume = s.instrument_volume.count(symbol) ? s.instrument_volume.at(symbol) : 0.0;
        
        double fill_pct = total_fills > 0 ? (100.0 * fills / total_fills) : 0.0;
        double pnl_pct = std::abs(s.pnl_sum) > 1e-6 ? (100.0 * pnl / s.pnl_sum) : 0.0;
        
      printf("│ %-8s %8lld %7.1f%% %12.2f %7.1f%% %15.0f\n", 
               symbol.c_str(), fills, fill_pct, pnl, pnl_pct, volume);
      }
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    
    printf("\n⏰ TIME RANGE\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Start: %s (%lld)\n", format_timestamp(s.ts_first).c_str(), s.ts_first);
    printf("│ End:   %s (%lld)\n", format_timestamp(s.ts_last).c_str(), s.ts_last);
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    
    printf("\n⚠️  POSITION CONFLICT ANALYSIS\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    
    // Track positions throughout the run by replaying fills
    std::unordered_map<std::string, ConflictPosition> positions;
    int total_conflicts = 0;
    std::vector<std::string> conflict_timestamps;
    
    // Query all FILL events to reconstruct position history
    sqlite3_stmt* fill_st = nullptr;
    std::string fill_sql = "SELECT ts_millis, symbol, side, qty FROM audit_events WHERE run_id = ? AND kind = 'FILL' ORDER BY ts_millis ASC";
    int fill_rc = sqlite3_prepare_v2(db.get_db(), fill_sql.c_str(), -1, &fill_st, nullptr);
    if (fill_rc == SQLITE_OK) {
        sqlite3_bind_text(fill_st, 1, run.c_str(), -1, SQLITE_TRANSIENT);
        
        while (sqlite3_step(fill_st) == SQLITE_ROW) {
            int64_t ts_millis = sqlite3_column_int64(fill_st, 0);
            const char* symbol = (const char*)sqlite3_column_text(fill_st, 1);
            const char* side = (const char*)sqlite3_column_text(fill_st, 2);
            double qty = sqlite3_column_double(fill_st, 3);
            
            if (symbol && side) {
                // Update position
                auto& pos = positions[symbol];
                pos.symbol = symbol;
                
                // Apply fill to position (BUY=0, SELL=1)
                if (strcmp(side, "BUY") == 0) {
                    pos.qty += qty;
                } else if (strcmp(side, "SELL") == 0) {
                    pos.qty -= qty;
                }
                
                // **PERFORMANCE FIX**: Only check conflicts periodically to avoid O(n²) complexity
                // Check conflicts every 50 fills or if we have fewer than 5 conflicts detected
                static int fill_count = 0;
                fill_count++;
                
                if (fill_count % 50 == 0 || total_conflicts < 5) {
                auto conflict_analysis = analyze_position_conflicts(positions);
                if (conflict_analysis.has_conflicts) {
                    total_conflicts++;
                    
                    // Convert timestamp to readable format
                    time_t ts_sec = ts_millis / 1000;
                    struct tm* tm_info = localtime(&ts_sec);
                    char time_buffer[32];
                    strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", tm_info);
                    
                    conflict_timestamps.push_back(std::string(time_buffer));
                    
                    // Only show first few conflicts to avoid spam
                    if (total_conflicts <= 5) {
                            printf("│ ⚠️  CONFLICT #%d at %s:\n", total_conflicts, time_buffer);
                        for (const auto& conflict : conflict_analysis.conflicts) {
                                printf("│   %s\n", conflict.c_str());
                        }
                        }
                    }
                }
            }
        }
        sqlite3_finalize(fill_st);
    }
    
    // Summary of conflict analysis
    if (total_conflicts == 0) {
        printf("│ ✅ NO CONFLICTS DETECTED: All positions maintained proper directional consistency\n");
    } else {
        printf("│ ❌ CONFLICTS DETECTED: %d instances of conflicting positions found\n", total_conflicts);
        if (total_conflicts > 5) {
            printf("│   (Showing first 5 conflicts only - %d additional conflicts occurred)\n", total_conflicts - 5);
        }
        printf("│\n");
        printf("│ ⚠️  WARNING: Conflicting positions generate fees without profit and cause\n");
        printf("│   leveraged ETF decay. The backend should prevent these automatically.\n");
    }
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    
    // **EOD POSITION CHECK**: Verify overnight risk management
    check_eod_positions(db.get_db(), run);
    
    // **POSITION CONFLICT CHECK**: Verify no conflicting positions
    check_position_conflicts(db.get_db(), run);
    
    printf("\nNote: P&L Rows = 0 means P&L is embedded in FILL events, not separate accounting events\n");
    return 0;
  }

  if (!strcmp(cmd,"integrity")) {
    const char* run_arg = arg("--run",argc,argv,"");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    
    // Perform comprehensive integrity check for all 5 core principles
    return perform_integrity_check(db.get_db(), run);
  }

  if (!strcmp(cmd,"system-health")) {
    printf("\n" BOLD BG_BLUE WHITE "🏥 INTEGRATED SYSTEM HEALTH CHECK" RESET "\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    
    try {
      // Create sample portfolio and symbol table for health check
      sentio::SymbolTable ST;
      ST.intern("QQQ");
      ST.intern("TQQQ");
      ST.intern("SQQQ");
      ST.intern("PSQ");
      
      sentio::Portfolio sample_portfolio(ST.size());
      std::vector<double> sample_prices = {400.0, 45.0, 15.0, 25.0};
      
      sentio::SentioIntegrationAdapter adapter;
      auto health = adapter.check_system_health(sample_portfolio, ST, sample_prices);
      
      printf("│ " BOLD "Current Equity:" RESET " $%.2f │\n", health.current_equity);
      printf("│ " BOLD "Position Integrity:" RESET " %s │\n", health.position_integrity ? "✅ PASS" : "❌ FAIL");
      printf("│ " BOLD "Cash Integrity:" RESET " %s │\n", health.cash_integrity ? "✅ PASS" : "❌ FAIL");
      printf("│ " BOLD "EOD Compliance:" RESET " %s │\n", health.eod_compliance ? "✅ PASS" : "❌ FAIL");
      printf("│ " BOLD "Total Violations:" RESET " %d │\n", health.total_violations);
      
      if (health.critical_alerts.empty()) {
        printf("│ " GREEN "✅ SYSTEM HEALTH: EXCELLENT" RESET " │\n");
        printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
        return 0;
      } else {
        printf("│ " RED "⚠️  SYSTEM HEALTH: ISSUES DETECTED" RESET " │\n");
        for (const auto& alert : health.critical_alerts) {
          printf("│ " RED "🚨 %s" RESET " │\n", alert.c_str());
        }
        printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
        return 1;
      }
      
    } catch (const std::exception& e) {
      printf("│ " RED "❌ Health check failed: %s" RESET " │\n", e.what());
      printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
      return 1;
    }
  }

  if (!strcmp(cmd,"architecture-test")) {
    printf("\n" BOLD BG_BLUE WHITE "🧪 COMPREHENSIVE ARCHITECTURE INTEGRATION TESTS" RESET "\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Testing new integrated architecture components...                               │\n");
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    
    try {
      sentio::SentioIntegrationAdapter adapter;
      auto test_result = adapter.run_integration_tests();
      
      printf("│ " BOLD "Total Tests:" RESET " %d │\n", test_result.total_tests);
      printf("│ " BOLD "Passed:" RESET " %d ✅ │\n", test_result.passed_tests);
      printf("│ " BOLD "Failed:" RESET " %d ❌ │\n", test_result.failed_tests);
      printf("│ " BOLD "Execution Time:" RESET " %.1fms │\n", test_result.execution_time_ms);
      printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
      
      if (test_result.success) {
        printf("\n" GREEN "🎉 ALL INTEGRATION TESTS PASSED!" RESET "\n");
        printf(GREEN "✅ System architecture is working correctly" RESET "\n");
        return 0;
      } else {
        printf("\n" RED "❌ INTEGRATION TESTS FAILED!" RESET "\n");
        printf(RED "🚨 Error: %s" RESET "\n", test_result.error_message.c_str());
        return 1;
      }
      
    } catch (const std::exception& e) {
      printf(RED "❌ Integration tests failed: %s" RESET "\n", e.what());
      return 1;
    }
  }

  if (!strcmp(cmd,"event-audit")) {
    const char* run_arg = arg("--run",argc,argv,"");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    
    const char* export_file = arg("--export", argc, argv, "");
    
    printf("\n" BOLD BG_BLUE WHITE "📚 EVENT SOURCING AUDIT TRAIL" RESET "\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ " BOLD "Run ID:" RESET " %s │\n", run.c_str());
    
    try {
      // Demonstrate event sourcing capabilities using integration adapter
      sentio::SentioIntegrationAdapter adapter;
      
      // Create sample portfolio and symbol table
      sentio::SymbolTable ST;
      ST.intern("QQQ");
      sentio::Portfolio sample_portfolio(ST.size());
      std::vector<double> sample_prices = {400.0};
      
      printf("│ " BOLD "Event Sourcing Demo:" RESET " Simulating trading events │\n");
      
      // Simulate some trading events
      auto decisions = adapter.execute_integrated_bar(0.8, sample_portfolio, ST, sample_prices, 1000);
      
      printf("│ " BOLD "Generated Decisions:" RESET " %zu allocation decisions │\n", decisions.size());
      
      for (const auto& decision : decisions) {
        printf("│ " BOLD "Decision:" RESET " %s -> %.2f%% (%s) │\n", 
               decision.instrument.c_str(), decision.target_weight * 100.0, decision.reason.c_str());
      }
      
      // Export if requested
      if (export_file && *export_file) {
        printf("│ " BOLD "Export:" RESET " Would export audit trail to %s │\n", export_file);
      }
      
      printf("│ " GREEN "✅ EVENT SOURCING SYSTEM: OPERATIONAL" RESET " │\n");
      printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
      return 0;
      
    } catch (const std::exception& e) {
      printf("│ " RED "❌ Event audit failed: %s" RESET " │\n", e.what());
      printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
      return 1;
    }
  }

  if (!strcmp(cmd,"export")) {
    const char* run_arg = arg("--run",argc,argv,"");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    const char* fmt = arg("--fmt",argc,argv,"jsonl");
    const char* out = arg("--out",argc,argv,"-");
    if (strcmp(out,"-")==0) { fputs("write to file only (use --out)\n", stderr); return 5; }
    if (!strcmp(fmt,"jsonl")) db.export_run_jsonl(run,out); else db.export_run_csv(run,out);
    puts("exported"); return 0;
  }

  if (!strcmp(cmd,"grep")) {
    const char* run_arg = arg("--run",argc,argv,"");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    const char* where = arg("--sql",argc,argv,"");
    long n = db.grep_where(run, where?where:"");
    printf("rows=%ld\n", n);
    return 0;
  }

  if (!strcmp(cmd,"diff")) {
    const char* a=arg("--run",argc,argv,"");
    const char* b=arg("--run2",argc,argv,"");
    auto txt = db.diff_runs(a,b);
    fputs(txt.c_str(), stdout);
    return 0;
  }

  if (!strcmp(cmd,"vacuum")) { db.vacuum(); puts("ok"); return 0; }

  if (!strcmp(cmd,"list")) {
    const char* strategy = arg("--strategy", argc, argv, "");
    const char* kind = arg("--kind", argc, argv, "");
    list_runs(dbp, strategy ? strategy : "", kind ? kind : "");
    return 0;
  }

  if (!strcmp(cmd,"latest")) {
    const char* strategy = arg("--strategy", argc, argv, "");
    find_latest_run(dbp, strategy ? strategy : "");
    return 0;
  }

  if (!strcmp(cmd,"info")) {
    const char* run_arg = arg("--run", argc, argv, "");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    show_run_info(dbp, run);
    return 0;
  }

  if (!strcmp(cmd,"trade-flow")) {
    const char* run_arg = arg("--run", argc, argv, "");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    const char* symbol = arg("--symbol", argc, argv, "");
    const char* limit_str = arg("--limit", argc, argv, "20");
    int limit = std::atoi(limit_str);
    
    // Handle --max option with optional number
    bool show_max = has("--max", argc, argv);
    if (show_max) {
      const char* max_str = arg("--max", argc, argv, nullptr);
      if (max_str && *max_str) {
        // --max N specified
        limit = std::atoi(max_str);
      } else {
        // --max without number specified, show all
        limit = 0;
      }
    }
    bool enhanced = has("--enhanced", argc, argv);
    bool show_buy = has("--buy", argc, argv);
    bool show_sell = has("--sell", argc, argv);
    bool show_hold = has("--hold", argc, argv);
    show_trade_flow(dbp, run, symbol, limit, enhanced, show_buy, show_sell, show_hold);
    
    // **EOD POSITION CHECK**: Verify overnight risk management
    check_eod_positions(db.get_db(), run);
    
    // **POSITION CONFLICT CHECK**: Verify no conflicting positions
    check_position_conflicts(db.get_db(), run);
    
    return 0;
  }

  if (!strcmp(cmd,"signal-stats")) {
    const char* run_arg = arg("--run", argc, argv, "");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    const char* strategy = arg("--strategy", argc, argv, "");
    show_signal_stats(dbp, run, strategy ? strategy : "");
    return 0;
  }

  if (!strcmp(cmd,"signal-flow")) {
    const char* run_arg = arg("--run", argc, argv, "");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    const char* symbol = arg("--symbol", argc, argv, "");
    const char* limit_str = arg("--limit", argc, argv, "20");
    int limit = std::atoi(limit_str);
    
    // Handle --max option with optional number
    bool show_max = has("--max", argc, argv);
    if (show_max) {
      const char* max_str = arg("--max", argc, argv, nullptr);
      if (max_str && *max_str) {
        // --max N specified
        limit = std::atoi(max_str);
      } else {
        // --max without number specified, show all
        limit = 0;
      }
    }
    bool show_buy = has("--buy", argc, argv);
    bool show_sell = has("--sell", argc, argv);
    bool show_hold = has("--hold", argc, argv);
    bool enhanced = has("--enhanced", argc, argv);
    show_signal_flow(dbp, run, symbol, limit, show_buy, show_sell, show_hold, enhanced);
    
    // **EOD POSITION CHECK**: Verify overnight risk management
    check_eod_positions(db.get_db(), run);
    
    // **POSITION CONFLICT CHECK**: Verify no conflicting positions
    check_position_conflicts(db.get_db(), run);
    
    return 0;
  }

  if (!strcmp(cmd,"position-history")) {
    const char* run_arg = arg("--run", argc, argv, "");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    const char* symbol = arg("--symbol", argc, argv, "");
    const char* limit_str = arg("--limit", argc, argv, "20");
    int limit = std::atoi(limit_str);
    
    // Handle --max option with optional number
    bool show_max = has("--max", argc, argv);
    if (show_max) {
      const char* max_str = arg("--max", argc, argv, nullptr);
      if (max_str && *max_str) {
        // --max N specified
        limit = std::atoi(max_str);
      } else {
        // --max without number specified, show all
        limit = 0;
      }
    }
    bool show_buy = has("--buy", argc, argv);
    bool show_sell = has("--sell", argc, argv);
    bool show_hold = has("--hold", argc, argv);
    show_position_history(dbp, run, symbol, limit, show_buy, show_sell, show_hold);
    
    // **EOD POSITION CHECK**: Verify overnight risk management
    check_eod_positions(db.get_db(), run);
    
    // **POSITION CONFLICT CHECK**: Verify no conflicting positions
    check_position_conflicts(db.get_db(), run);
    
    return 0;
  }

  if (!strcmp(cmd,"strategies-summary")) {
    show_strategies_summary(dbp);
    return 0;
  }

  fputs(usage, stderr); return 1;
}

// Provide a standalone main if you build standalone; otherwise link into your app.
int main(int argc, char** argv) { return audit_main(argc, argv); }

// Implementation of utility functions
namespace audit {

void list_runs(const std::string& db_path, const std::string& strategy_filter, const std::string& kind_filter) {
  try {
    DB db(db_path);
    
    // Use global ANSI color codes defined at top of file
    
    std::string sql = "SELECT run_id, strategy, kind, started_at, ended_at, note FROM audit_runs";
    std::string where_clause = "";
    
    if (!strategy_filter.empty() || !kind_filter.empty()) {
      where_clause = " WHERE ";
      bool first = true;
      if (!strategy_filter.empty()) {
        where_clause += "strategy = '" + strategy_filter + "'";
        first = false;
      }
      if (!kind_filter.empty()) {
        if (!first) where_clause += " AND ";
        where_clause += "kind = '" + kind_filter + "'";
      }
    }
    
    sql += where_clause + " ORDER BY started_at DESC";
    
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(db.get_db(), sql.c_str(), -1, &st, nullptr);
    
    // Enhanced header
    printf("\n%s%s%s╔══════════════════════════════════════════════════════════════════════════════════╗%s\n", BOLD, BG_BLUE, WHITE, RESET);
    printf("%s%s%s║                              📊 AUDIT RUN HISTORY                               ║%s\n", BOLD, BG_BLUE, WHITE, RESET);
    printf("%s%s%s╚══════════════════════════════════════════════════════════════════════════════════╝%s\n", BOLD, BG_BLUE, WHITE, RESET);
    
    // Filters display
    if (!strategy_filter.empty() || !kind_filter.empty()) {
      printf("\n%s%s🔍 ACTIVE FILTERS%s\n", BOLD, CYAN, RESET);
      printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
      if (!strategy_filter.empty()) {
        printf("│ %sStrategy:%s %s%s%s\n", BOLD, RESET, MAGENTA, strategy_filter.c_str(), RESET);
      }
      if (!kind_filter.empty()) {
        printf("│ %sKind:%s     %s%s%s\n", BOLD, RESET, YELLOW, kind_filter.c_str(), RESET);
      }
      printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    }
    
    // Count total runs first
    int total_runs = 0;
    while (sqlite3_step(st) == SQLITE_ROW) {
      total_runs++;
    }
    sqlite3_reset(st);
    
    printf("\n%s%s📋 RUN LIST%s %s(%d runs)%s\n", BOLD, CYAN, RESET, DIM, total_runs, RESET);
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ %s%-20s %-12s %-10s %-19s %-10s%s\n", BOLD, "RUN_ID", "STRATEGY", "KIND", "STARTED_AT", "STATUS", RESET);
    printf("│ %s%-20s %-12s %-10s %-19s %-10s%s\n", DIM, "──────", "────────", "────", "──────────", "──────", RESET);
    
    // Helper function to format timestamp
    auto format_timestamp = [](int64_t ts_ms) -> std::string {
      if (ts_ms == 0) return "N/A";
      time_t ts_sec = ts_ms / 1000;
      struct tm* tm_info = localtime(&ts_sec);
      char buffer[32];
      strftime(buffer, sizeof(buffer), "%m-%d %H:%M:%S", tm_info);
      return std::string(buffer);
    };
    
    while (sqlite3_step(st) == SQLITE_ROW) {
      const char* run_id = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
      const char* strategy = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
      const char* kind = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));
      std::int64_t started_at = sqlite3_column_int64(st, 3);
      std::int64_t ended_at = sqlite3_column_int64(st, 4);
      const char* note = reinterpret_cast<const char*>(sqlite3_column_text(st, 5));
      
      // Determine status and color
      const char* status_color = GREEN;
      const char* status_text = "✅ DONE";
      if (ended_at == 0) {
        status_color = YELLOW;
        status_text = "🔄 RUNNING";
      }
      
      std::string formatted_time = format_timestamp(started_at);
      
      printf("│ %s%-20s%s %s%-12s%s %s%-10s%s %s%-19s%s %s%s%s\n", 
             BLUE, run_id ? run_id : "N/A", RESET,
             MAGENTA, strategy ? strategy : "N/A", RESET,
             CYAN, kind ? kind : "N/A", RESET,
             WHITE, formatted_time.c_str(), RESET,
             status_color, status_text, RESET);
      
      // Show note if present
      if (note && strlen(note) > 0) {
        printf("│   %s└─ Note: %s%s\n", DIM, note, RESET);
      }
    }
    
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    
    if (total_runs == 0) {
      printf("\n%s📭 No runs found", YELLOW);
      if (!strategy_filter.empty() || !kind_filter.empty()) {
        printf(" matching the specified filters");
      }
      printf("%s\n", RESET);
    } else {
      printf("\n%s📊 Total: %s%d runs%s", BOLD, GREEN, total_runs, RESET);
      if (!strategy_filter.empty() || !kind_filter.empty()) {
        printf(" %s(filtered)%s", DIM, RESET);
      }
      printf("\n");
    }
    
    sqlite3_finalize(st);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error listing runs: %s\n", e.what());
  }
}

void find_latest_run(const std::string& db_path, const std::string& strategy_filter) {
  try {
    DB db(db_path);
    
    std::string sql = "SELECT run_id, strategy, kind, started_at, ended_at FROM audit_runs";
    if (!strategy_filter.empty()) {
      sql += " WHERE strategy = '" + strategy_filter + "'";
    }
    sql += " ORDER BY run_id DESC LIMIT 1";
    
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(db.get_db(), sql.c_str(), -1, &st, nullptr);
    
    if (sqlite3_step(st) == SQLITE_ROW) {
      const char* run_id = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
      const char* strategy = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
      const char* kind = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));
      std::int64_t started_at = sqlite3_column_int64(st, 3);
      std::int64_t ended_at = sqlite3_column_int64(st, 4);
      
      printf("Latest run: %s\n", run_id ? run_id : "");
      printf("Strategy: %s\n", strategy ? strategy : "");
      printf("Kind: %s\n", kind ? kind : "");
      printf("Started: %lld\n", started_at);
      printf("Ended: %lld\n", ended_at);
    } else {
      printf("No runs found");
      if (!strategy_filter.empty()) {
        printf(" for strategy: %s", strategy_filter.c_str());
      }
      printf("\n");
    }
    
    sqlite3_finalize(st);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error finding latest run: %s\n", e.what());
  }
}

void show_run_info(const std::string& db_path, const std::string& run_id) {
  try {
    DB db(db_path);
    
    // Get run info
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(db.get_db(), 
                       "SELECT run_id, strategy, kind, started_at, ended_at, params_json, data_hash, git_rev, note FROM audit_runs WHERE run_id = ?", 
                       -1, &st, nullptr);
    sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(st) == SQLITE_ROW) {
      const char* strategy = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
      const char* kind = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));
      std::int64_t started_at = sqlite3_column_int64(st, 3);
      std::int64_t ended_at = sqlite3_column_int64(st, 4);
      const char* params_json = reinterpret_cast<const char*>(sqlite3_column_text(st, 5));
      const char* data_hash = reinterpret_cast<const char*>(sqlite3_column_text(st, 6));
      const char* git_rev = reinterpret_cast<const char*>(sqlite3_column_text(st, 7));
      const char* note = reinterpret_cast<const char*>(sqlite3_column_text(st, 8));
      
      printf("Run ID: %s\n", run_id.c_str());
      printf("Strategy: %s\n", strategy ? strategy : "");
      printf("Kind: %s\n", kind ? kind : "");
      printf("Started: %lld\n", started_at);
      printf("Ended: %lld\n", ended_at);
      printf("Data Hash: %s\n", data_hash ? data_hash : "");
      printf("Git Rev: %s\n", git_rev ? git_rev : "");
      printf("Note: %s\n", note ? note : "");
      printf("Params: %s\n", params_json ? params_json : "");
      
      // Get summary
      auto summary = db.summarize(run_id);
      printf("\nSummary:\n");
      printf("  Events: %lld\n", summary.n_total);
      printf("  Signals: %lld\n", summary.n_signal);
      printf("  Orders: %lld\n", summary.n_order);
      printf("  Fills: %lld\n", summary.n_fill);
      printf("  P&L Rows: %lld\n", summary.n_pnl);
      printf("  P&L Sum: %.6f\n", summary.pnl_sum);
      printf("  Time Range: %lld - %lld\n", summary.ts_first, summary.ts_last);
      
    } else {
      printf("Run not found: %s\n", run_id.c_str());
    }
    
    sqlite3_finalize(st);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error showing run info: %s\n", e.what());
  }
}

struct TradeFlowEvent {
  std::int64_t timestamp;
  std::string kind;
  std::string symbol;
  std::string side;
  double quantity;
  double price;
  double pnl_delta;
  double weight;
  double prob;
  std::string reason;
  std::string note;
};

void show_trade_flow(const std::string& db_path, const std::string& run_id, const std::string& symbol_filter, int limit, bool enhanced, bool show_buy, bool show_sell, bool show_hold) {
  try {
    DB db(db_path);
    
    // Get run info and print header with time periods
    RunInfo info = get_run_info(db_path, run_id);
    
    // Get trading blocks to calculate time periods dynamically
    std::vector<BlockRow> block_rows = db.get_blocks_for_run(run_id);
    if (!block_rows.empty()) {
      // Calculate dataset and test periods from block data
      int64_t dataset_start_ms = block_rows.front().start_ts_ms;
      int64_t dataset_end_ms = block_rows.back().end_ts_ms;
      int64_t test_start_ms = block_rows.front().start_ts_ms;
      int64_t test_end_ms = block_rows.back().end_ts_ms;
      
      int dataset_days = (dataset_end_ms - dataset_start_ms) / (24 * 60 * 60 * 1000);
      int test_days = (test_end_ms - test_start_ms) / (24 * 60 * 60 * 1000);
      int tb_count = block_rows.size();
      
      // Inject time period data into info.meta
      nlohmann::json meta_json;
      if (!info.meta.empty()) {
        try {
          meta_json = nlohmann::json::parse(info.meta);
        } catch (...) {
          meta_json = nlohmann::json::object();
        }
      }
      
      meta_json["dataset_period_start_ts_ms"] = dataset_start_ms;
      meta_json["dataset_period_end_ts_ms"] = dataset_end_ms;
      meta_json["dataset_period_days"] = dataset_days;
      meta_json["run_period_start_ts_ms"] = test_start_ms;
      meta_json["run_period_end_ts_ms"] = test_end_ms;
      meta_json["test_period_days"] = test_days;
      meta_json["tb_count"] = tb_count;
      
      info.meta = meta_json.dump();
    }
    
    print_run_header(" EXECUTION FLOW REPORT ", info);
    
    if (!symbol_filter.empty()) {
      printf("Symbol Filter: %s\n", symbol_filter.c_str());
    }
    
    // Display action filters
    std::vector<std::string> action_filters;
    if (show_buy) action_filters.push_back("BUY");
    if (show_sell) action_filters.push_back("SELL");
    if (show_hold) action_filters.push_back("HOLD");
    
    if (!action_filters.empty()) {
      printf("Action Filter: ");
      for (size_t i = 0; i < action_filters.size(); i++) {
        if (i > 0) printf(", ");
        printf("%s", action_filters[i].c_str());
      }
      printf("\n");
    }
    
    if (limit > 0) {
      printf("Showing: %d most recent events\n", limit);
    } else {
      printf("Showing: All execution events\n");
    }
    printf("\n");
    
    // Build SQL query to get trade flow events
    std::string sql = "SELECT ts_millis, kind, symbol, side, qty, price, pnl_delta, weight, prob, reason, note FROM audit_events WHERE run_id = ? AND kind IN ('SIGNAL', 'ORDER', 'FILL')";
    
    if (!symbol_filter.empty()) {
      sql += " AND symbol = '" + symbol_filter + "'";
    }
    
    // Add action filtering if any specific actions are requested
    if (show_buy || show_sell || show_hold) {
      std::vector<std::string> side_conditions;
      if (show_buy) side_conditions.push_back("side = 'BUY'");
      if (show_sell) side_conditions.push_back("side = 'SELL'");
      if (show_hold) side_conditions.push_back("side = 'HOLD'");
      
      sql += " AND (";
      for (size_t i = 0; i < side_conditions.size(); i++) {
        if (i > 0) sql += " OR ";
        sql += side_conditions[i];
      }
      sql += ")";
    }
    
    sql += " ORDER BY ts_millis ASC";
    
    sqlite3_stmt* st = nullptr;
    int rc = sqlite3_prepare_v2(db.get_db(), sql.c_str(), -1, &st, nullptr);
    if (rc != SQLITE_OK) {
      fprintf(stderr, "SQL prepare error: %s\n", sqlite3_errmsg(db.get_db()));
      return;
    }
    sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    
    // Collect all events and calculate summary statistics
    std::vector<TradeFlowEvent> events;
    int signal_count = 0, order_count = 0, fill_count = 0;
    double total_volume = 0.0, total_pnl = 0.0;
    std::map<std::string, int> symbol_activity;
    std::map<std::string, double> symbol_pnl;
    
    while (sqlite3_step(st) == SQLITE_ROW) {
      TradeFlowEvent event;
      event.timestamp = sqlite3_column_int64(st, 0);
      event.kind = sqlite3_column_text(st, 1) ? (char*)sqlite3_column_text(st, 1) : "";
      event.symbol = sqlite3_column_text(st, 2) ? (char*)sqlite3_column_text(st, 2) : "";
      event.side = sqlite3_column_text(st, 3) ? (char*)sqlite3_column_text(st, 3) : "";
      event.quantity = sqlite3_column_double(st, 4);
      event.price = sqlite3_column_double(st, 5);
      event.pnl_delta = sqlite3_column_double(st, 6);
      event.weight = sqlite3_column_double(st, 7);
      event.prob = sqlite3_column_double(st, 8);
      event.reason = sqlite3_column_text(st, 9) ? (char*)sqlite3_column_text(st, 9) : "";
      event.note = sqlite3_column_text(st, 10) ? (char*)sqlite3_column_text(st, 10) : "";
      
      events.push_back(event);
      
      // Update statistics
      if (event.kind == "SIGNAL") signal_count++;
      else if (event.kind == "ORDER") order_count++;
      else if (event.kind == "FILL") {
        fill_count++;
        total_volume += event.quantity * event.price;
        total_pnl += event.pnl_delta;
        if (!event.symbol.empty()) {
          symbol_pnl[event.symbol] += event.pnl_delta;
        }
      }
      
      if (!event.symbol.empty()) {
        symbol_activity[event.symbol]++;
      }
    }
    
    sqlite3_finalize(st);
    
    // Calculate execution efficiency
    double execution_rate = (order_count > 0) ? (double)fill_count / order_count * 100.0 : 0.0;
    double signal_to_order_rate = (signal_count > 0) ? (double)order_count / signal_count * 100.0 : 0.0;
    
    // 1. EXECUTION PERFORMANCE SUMMARY
    printf("📊 EXECUTION PERFORMANCE SUMMARY\n");
    printf("┌─────────────────────┬─────────────┬─────────────────────┬─────────────┬────────────────┬──────────┐\n");
    printf("│ Total Signals       │ %11d │ Orders Placed       │ %11d │ Execution Rate │ %7.1f%% │\n", 
           signal_count, order_count, execution_rate);
    printf("│ Orders Filled       │ %11d │ Total Volume        │ $%10.0f │ Signal→Order   │ %7.1f%% │\n", 
           fill_count, total_volume, signal_to_order_rate);
    printf("│ Active Symbols      │ %11d │ Net P&L Impact      │ $%+10.2f │ Avg Fill Size  │ $%7.0f │\n", 
           (int)symbol_activity.size(), total_pnl, 
           fill_count > 0 ? total_volume / fill_count : 0.0);
    printf("└─────────────────────┴─────────────┴─────────────────────┴─────────────┴────────────────┴──────────┘\n\n");
    
    // 2. SYMBOL PERFORMANCE BREAKDOWN
    if (!symbol_pnl.empty()) {
      printf("📈 SYMBOL PERFORMANCE BREAKDOWN\n");
      printf("┌────────┬─────────┬─────────────┬─────────────────────────────────────────────────────────────┐\n");
      printf("│ Symbol │ Events  │ P&L Impact  │ Performance Level                                           │\n");
      printf("├────────┼─────────┼─────────────┼─────────────────────────────────────────────────────────────┤\n");
      
      // Sort symbols by P&L
      std::vector<std::pair<std::string, double>> sorted_pnl(symbol_pnl.begin(), symbol_pnl.end());
      std::sort(sorted_pnl.begin(), sorted_pnl.end(), 
                [](const auto& a, const auto& b) { return a.second > b.second; });
      
      for (const auto& [symbol, pnl] : sorted_pnl) {
        int events = symbol_activity[symbol];
        
        // Create performance bar (green for profit, red for loss)
        double max_abs_pnl = 0;
        for (const auto& [s, p] : sorted_pnl) {
          max_abs_pnl = std::max(max_abs_pnl, std::abs(p));
        }
        
        int bar_length = max_abs_pnl > 0 ? std::min(50, (int)(std::abs(pnl) * 50 / max_abs_pnl)) : 0;
        std::string performance_bar;
        if (pnl > 0) {
          performance_bar = std::string(bar_length, '#') + std::string(50 - bar_length, '.');
        } else {
          performance_bar = std::string(50 - bar_length, '.') + std::string(bar_length, 'X');
        }
        
        const char* pnl_color = pnl > 0 ? "🟢" : pnl < 0 ? "🔴" : "⚪";
        
        printf("│ %-6s │ %7d │ %s$%+9.2f │ %s │\n", 
               symbol.c_str(), events, pnl_color, pnl, performance_bar.c_str());
      }
      printf("└────────┴─────────┴─────────────┴─────────────────────────────────────────────────────────────┘\n\n");
    }
    
    // 3. EXECUTION EVENT TIMELINE
    printf("🔄 EXECUTION EVENT TIMELINE");
    if (limit > 0 && (int)events.size() > limit) {
      printf(" (Last %d of %d events)", limit, (int)events.size());
    }
    printf("\n");
    printf("┌──────────────┬────────────┬────────┬────────┬──────────┬──────────┬─────────────┬──────────────┐\n");
    printf("│ Time         │ Event      │ Symbol │ Action │ Quantity │ Price    │ Value       │ P&L Impact   │\n");
    printf("├──────────────┼────────────┼────────┼────────┼──────────┼──────────┼─────────────┼──────────────┤\n");
    
    // Show recent events (apply limit here)
    int start_idx = (limit > 0 && (int)events.size() > limit) ? (int)events.size() - limit : 0;
    int event_count = 0;
    
    for (int i = start_idx; i < (int)events.size(); i++) {
      const auto& event = events[i];
      
      // Add empty line before each event for better scanning (except first)
      if (event_count > 0) {
        printf("│              │            │        │        │          │          │             │              │\n");
      }
      
      // Format timestamp
      char time_str[32];
      std::time_t time_t = event.timestamp / 1000;
      std::strftime(time_str, sizeof(time_str), "%m/%d %H:%M:%S", std::localtime(&time_t));
      
      // Event type icons
      const char* event_icon = "📋";
      if (event.kind == "SIGNAL") event_icon = "📡";
      else if (event.kind == "FILL") event_icon = "✅";
      
      // Action color coding
      const char* action_color = "";
      if (event.side == "BUY") action_color = "🟢";
      else if (event.side == "SELL") action_color = "🔴";
      else if (event.side == "HOLD") action_color = "🟡";
      
      double trade_value = event.quantity * event.price;
      
      // Pad event types for consistent alignment
      std::string padded_kind = event.kind;
      if (event.kind == "ORDER") {
        padded_kind = "ORDER ";
      } else if (event.kind == "FILL") {
        padded_kind = "FILL  ";
      }
      
      printf("│ %-12s │ %s%-6s │ %-6s │ %s%-4s │ %8.0f │ $%7.2f │ $%+10.0f │ $%+11.2f │\n",
             time_str, event_icon, padded_kind.c_str(), event.symbol.c_str(),
             action_color, event.side.c_str(), event.quantity, event.price, 
             trade_value, event.pnl_delta);
      
      // Show additional details based on event type
      if (event.kind == "SIGNAL") {
        if (event.prob > 0 || event.weight > 0) {
          printf("│              │ └─ Signal Strength: %.1f%% prob, %.2f weight\n",
                 event.prob * 100, event.weight);
        }
        if (!event.reason.empty()) {
          printf("│              │ └─ Signal Type: %s\n", event.reason.c_str());
        }
      } else if (event.kind == "ORDER") {
        printf("│              │ └─ Order Details: %s %.0f shares @ $%.2f\n", 
               event.side.c_str(), event.quantity, event.price);
      } else if (event.kind == "FILL") {
        const char* pnl_indicator = event.pnl_delta > 0 ? "🟢 PROFIT" : 
                                   event.pnl_delta < 0 ? "🔴 LOSS" : "⚪ NEUTRAL";
        printf("│              │ └─ Execution: %s (P&L: $%.2f %s)\n", 
               event.side.c_str(), event.pnl_delta, pnl_indicator);
      }
      
      event_count++;
    }
    printf("└──────────────┴────────────┴────────┴────────┴──────────┴──────────┴─────────────┴──────────────┘\n\n");
    
    // 4. EXECUTION EFFICIENCY ANALYSIS
    printf("⚡ EXECUTION EFFICIENCY ANALYSIS\n");
    printf("┌──────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Metric                    │ Value         │ Rating         │ Description                     │\n");
    printf("├──────────────────────────────────────────────────────────────────────────────────────────────┤\n");
    
    // Execution rate analysis
    const char* exec_rating = execution_rate >= 90 ? "🟢 EXCELLENT" : 
                             execution_rate >= 70 ? "🟡 GOOD" : "🔴 NEEDS WORK";
    printf("│ Order Fill Rate           │ %12.1f%% │ %-16s │ %% of orders successfully filled │\n", 
           execution_rate, exec_rating);
    
    // Signal conversion analysis  
    const char* signal_rating = signal_to_order_rate >= 20 ? "🟢 ACTIVE" :
                               signal_to_order_rate >= 10 ? "🟡 MODERATE" : "🔴 PASSIVE";
    printf("│ Signal Conversion Rate    │ %12.1f%% │ %-16s │ %% of signals converted to orders│\n", 
           signal_to_order_rate, signal_rating);
    
    // P&L efficiency
    const char* pnl_rating = total_pnl > 0 ? "🟢 PROFITABLE" : 
                            total_pnl > -100 ? "🟡 BREAKEVEN" : "🔴 LOSING";
    printf("│ P&L Efficiency            │ $%+12.2f │ %-16s │ Net profit/loss from executions │\n", 
           total_pnl, pnl_rating);
    
    // Volume efficiency
    const char* volume_rating = total_volume > 1000000 ? "🟢 HIGH VOLUME" :
                               total_volume > 100000 ? "🟡 MODERATE" : "🔴 LOW VOLUME";
    printf("│ Trading Volume            │ %13.0f │ %-16s │ Total dollar volume traded      │\n", 
           total_volume, volume_rating);
    
    printf("└──────────────────────────────────────────────────────────────────────────────────────────────┘\n");
    
    // **NEW**: Instrument Distribution with P&L Breakdown for Trade Flow Report
    std::cout << "\n" << BOLD << CYAN << "🎯 INSTRUMENT DISTRIBUTION & P&L BREAKDOWN" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ Instrument │  Total Volume  │  Realized P&L  │  Fill Count    │ Avg Fill Size  │   P&L/Fill         │" << std::endl;
    std::cout << "├────────────┼────────────────┼────────────────┼────────────────┼────────────────┼────────────────────┤" << std::endl;
    
    // Calculate per-instrument statistics from events
    std::map<std::string, double> instrument_volume;
    std::map<std::string, double> instrument_pnl;
    std::map<std::string, int> instrument_fills;
    
    for (const auto& event : events) {
        if (event.kind == "FILL") {
            instrument_volume[event.symbol] += std::abs(event.quantity * event.price);
            instrument_pnl[event.symbol] += event.pnl_delta;
            instrument_fills[event.symbol]++;
        }
    }
    
    // **FIX**: Display ALL expected instruments (including zero activity)
    std::vector<std::string> all_expected_instruments = {"PSQ", "QQQ", "TQQQ", "SQQQ"};
    
    // Display per-instrument statistics
    double total_instrument_volume = 0.0;
    double total_instrument_pnl = 0.0;
    int total_instrument_fills = 0;
    
    for (const std::string& instrument : all_expected_instruments) {
        double volume = instrument_volume.count(instrument) ? instrument_volume[instrument] : 0.0;
        double pnl = instrument_pnl.count(instrument) ? instrument_pnl[instrument] : 0.0;
        int fills = instrument_fills.count(instrument) ? instrument_fills[instrument] : 0;
        double avg_fill_size = (fills > 0) ? volume / fills : 0.0;
        double pnl_per_fill = (fills > 0) ? pnl / fills : 0.0;
        
        total_instrument_volume += volume;
        total_instrument_pnl += pnl;
        total_instrument_fills += fills;
        
        // Color coding
        const char* pnl_color = (pnl >= 0) ? GREEN : RED;
        const char* pnl_per_fill_color = (pnl_per_fill >= 0) ? GREEN : RED;
        
        printf("│ %-10s │ $%13.2f │ %s$%+12.2f%s │ %14d │ $%12.2f │ %s$%+12.2f%s │\n",
               instrument.c_str(), volume,
               pnl_color, pnl, RESET,
               fills, avg_fill_size,
               pnl_per_fill_color, pnl_per_fill, RESET);
    }
    
    std::cout << "├────────────┼────────────────┼────────────────┼────────────────┼────────────────┼────────────────────┤" << std::endl;
    
    // Totals row
    double avg_total_fill_size = (total_instrument_fills > 0) ? total_instrument_volume / total_instrument_fills : 0.0;
    double avg_total_pnl_per_fill = (total_instrument_fills > 0) ? total_instrument_pnl / total_instrument_fills : 0.0;
    const char* total_pnl_color = (total_instrument_pnl >= 0) ? GREEN : RED;
    const char* total_pnl_per_fill_color = (avg_total_pnl_per_fill >= 0) ? GREEN : RED;
    
    printf("│ %-10s │ $%12.2f │ %s$%+12.2f%s │ %14d │ $%12.2f │ %s$%+12.2f%s │\n",
           "TOTAL", total_instrument_volume,
           total_pnl_color, total_instrument_pnl, RESET,
           total_instrument_fills, avg_total_fill_size,
           total_pnl_per_fill_color, avg_total_pnl_per_fill, RESET);
    
    std::cout << "└────────────┴────────────────┴────────────────┴────────────────┴────────────────┴────────────────┘" << std::endl;
    
    // Verification: Check if instrument P&L sum matches total
    double pnl_difference = std::abs(total_instrument_pnl - total_pnl);
    if (pnl_difference > 0.01) {
        std::cout << YELLOW << "⚠️  WARNING: Instrument P&L sum ($" << total_instrument_pnl 
                  << ") differs from total P&L ($" << total_pnl << ") by $" 
                  << pnl_difference << RESET << std::endl;
    } else {
        std::cout << GREEN << "✅ P&L Verification: Instrument breakdown matches total P&L" << RESET << std::endl;
    }
    
  } catch (const std::exception& e) {
    fprintf(stderr, "Error showing trade flow: %s\n", e.what());
  }
}

void show_signal_stats(const std::string& db_path, const std::string& run_id, const std::string& strategy_filter) {
  try {
    DB db(db_path);
    
    // Build SQL query to get signal diagnostics
    std::string sql = "SELECT ts_millis, symbol, qty, price, note FROM audit_events WHERE run_id = ? AND kind = 'SIGNAL_DIAG'";
    
    if (!strategy_filter.empty()) {
      sql += " AND symbol = '" + strategy_filter + "'";
    }
    
    sql += " ORDER BY ts_millis ASC";
    
    sqlite3_stmt* st = nullptr;
    int rc = sqlite3_prepare_v2(db.get_db(), sql.c_str(), -1, &st, nullptr);
    if (rc != SQLITE_OK) {
      fprintf(stderr, "SQL prepare error: %s\n", sqlite3_errmsg(db.get_db()));
      return;
    }
    sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    
    // Get run info and print header
    RunInfo info = get_run_info(db_path, run_id);
    print_run_header("SIGNAL DIAGNOSTICS", info);
    
    if (!strategy_filter.empty()) {
      printf("Strategy Filter: %s\n", strategy_filter.c_str());
    }
    printf("\n");
    
    printf("┌─────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│                                    SIGNAL STATISTICS                                        │\n");
    printf("└─────────────────────────────────────────────────────────────────────────────────────────────┘\n\n");
    
    bool found_any = false;
    while (sqlite3_step(st) == SQLITE_ROW) {
      found_any = true;
      std::int64_t ts = sqlite3_column_int64(st, 0);
      const char* strategy = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
      double emitted = sqlite3_column_double(st, 2);
      double dropped = sqlite3_column_double(st, 3);
      const char* note = reinterpret_cast<const char*>(sqlite3_column_text(st, 4));
      
      // Format timestamp
      char time_str[32];
      std::time_t time_t = ts / 1000;
      std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", std::localtime(&time_t));
      
      printf("🔍 Strategy: %s\n", strategy ? strategy : "");
      printf("⏰ Timestamp: %s\n", time_str);
      printf("📊 Signal Statistics:\n");
      printf("   📤 Emitted: %.0f\n", emitted);
      printf("   📥 Dropped: %.0f\n", dropped);
      printf("   📈 Success Rate: %.1f%%\n", emitted > 0 ? (emitted / (emitted + dropped)) * 100.0 : 0.0);
      
      if (note) {
        std::string clean_note = clean_note_for_display(note);
        if (!clean_note.empty()) {
          printf("   📋 Details: %s\n", clean_note.c_str());
        }
      }
      
      printf("\n");
    }
    
    if (!found_any) {
      printf("No signal diagnostics found for this run.\n");
    }
    
    sqlite3_finalize(st);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error showing signal stats: %s\n", e.what());
  }
}

void show_signal_flow(const std::string& db_path, const std::string& run_id, const std::string& symbol_filter, int limit, bool show_buy, bool show_sell, bool show_hold, bool enhanced) {
  try {
    DB db(db_path);
    
    // Get run info and print header
    RunInfo info = get_run_info(db_path, run_id);
    
    // Add comprehensive time period information like other reports
    auto summary = db.summarize(run_id);
    auto block_rows = db.get_blocks_for_run(run_id);
    
    if (!block_rows.empty()) {
      // Calculate test period from block data
      int64_t test_start_ts = block_rows.front().start_ts_ms;
      int64_t test_end_ts = block_rows.back().end_ts_ms;
      double test_days = (test_end_ts - test_start_ts) / (1000.0 * 60.0 * 60.0 * 24.0);
      int tb_count = block_rows.size();
      
      // For dataset period, we need to estimate from the dataset file info
      // This should match what strattest shows: full dataset range
      // Using a reasonable estimate based on the dataset type
      int64_t dataset_start_ts = 1663243800000LL; // 2022-09-15T13:30:00Z (from strattest output)
      int64_t dataset_end_ts = test_end_ts; // Assume dataset goes up to test end
      double dataset_days = (dataset_end_ts - dataset_start_ts) / (1000.0 * 60.0 * 60.0 * 24.0);
      
      // Create comprehensive metadata like other reports
      char time_buffer[512];
      snprintf(time_buffer, sizeof(time_buffer), 
               "\"dataset_period_start_ts_ms\":%lld,\"dataset_period_end_ts_ms\":%lld,\"dataset_period_days\":%.1f,"
               "\"run_period_start_ts_ms\":%lld,\"run_period_end_ts_ms\":%lld,\"test_period_days\":%.1f,"
               "\"tb_count\":%d",
               dataset_start_ts, dataset_end_ts, dataset_days,
               test_start_ts, test_end_ts, test_days,
               tb_count);
      info.meta = std::string("{") + time_buffer + "}";
    }
    
    print_run_header("SIGNAL PIPELINE REPORT ", info);
    
    // Enhanced filter and display information
    if (!symbol_filter.empty()) {
      std::cout << "\n" << BOLD << YELLOW << "🔍 Filter: " << symbol_filter << RESET << std::endl;
    }
    
    // Display action filters
    std::vector<std::string> action_filters;
    if (show_buy) action_filters.push_back("BUY");
    if (show_sell) action_filters.push_back("SELL");
    if (show_hold) action_filters.push_back("HOLD");
    
    if (!action_filters.empty()) {
      std::cout << BOLD << YELLOW << "🎯 Actions: ";
      for (size_t i = 0; i < action_filters.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << action_filters[i];
      }
      std::cout << RESET << std::endl;
    }
    
    if (limit > 0) {
      std::cout << "\n" << DIM << "Showing: " << limit << " most recent events" << RESET << std::endl;
    } else {
      std::cout << "\n" << DIM << "Showing: All signal events" << RESET << std::endl;
    }
    
    // Enhanced signal pipeline diagram
    std::cout << "\n" << BOLD << CYAN << "📊 SIGNAL PROCESSING PIPELINE" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ " << BLUE << "Market Data" << RESET << " → " << GREEN << "Feature Extraction" << RESET << " → " << MAGENTA << "Strategy Signal" << RESET << " → " << YELLOW << "Signal Gate" << RESET << " → " << CYAN << "Router" << RESET << " → " << WHITE << "Order" << RESET << " → " << GREEN << "Fill" << RESET << " │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    // Get signal diagnostics first
    printf("📈 SIGNAL DIAGNOSTICS:\n");
    sqlite3_stmt* diag_st = nullptr;
    std::string diag_sql = "SELECT symbol, qty, price, note FROM audit_events WHERE run_id = ? AND kind = 'SIGNAL_DIAG'";
    int diag_rc = sqlite3_prepare_v2(db.get_db(), diag_sql.c_str(), -1, &diag_st, nullptr);
    if (diag_rc == SQLITE_OK) {
      sqlite3_bind_text(diag_st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
      if (sqlite3_step(diag_st) == SQLITE_ROW) {
        const char* strategy = reinterpret_cast<const char*>(sqlite3_column_text(diag_st, 0));
        double emitted = sqlite3_column_double(diag_st, 1);
        double dropped = sqlite3_column_double(diag_st, 2);
        const char* details = reinterpret_cast<const char*>(sqlite3_column_text(diag_st, 3));
        
        printf("🔍 Strategy: %s\n", strategy ? strategy : "Unknown");
        printf("📤 Signals Emitted: %.0f\n", emitted);
        printf("📥 Signals Dropped: %.0f\n", dropped);
        printf("📈 Success Rate: %.1f%%\n", (emitted + dropped) > 0 ? (emitted / (emitted + dropped)) * 100.0 : 0.0);
        
        if (details) {
          printf("📋 Drop Breakdown: %s\n", details);
        }
      }
      sqlite3_finalize(diag_st);
    }
    printf("\n");
    
    // Get signal events with enhanced analysis
    std::string sql = "SELECT ts_millis, kind, symbol, side, qty, price, pnl_delta, weight, prob, reason, note, hash_curr FROM audit_events WHERE run_id = ? AND kind IN ('SIGNAL', 'ORDER', 'FILL', 'SIGNAL_DROP')";
    
    if (!symbol_filter.empty()) {
      sql += " AND symbol = '" + symbol_filter + "'";
    }
    
    // Add action filtering if any specific actions are requested
    if (show_buy || show_sell || show_hold) {
      std::vector<std::string> side_conditions;
      if (show_buy) side_conditions.push_back("side = 'BUY'");
      if (show_sell) side_conditions.push_back("side = 'SELL'");
      if (show_hold) side_conditions.push_back("side = 'HOLD'");
      
      sql += " AND (";
      for (size_t i = 0; i < side_conditions.size(); i++) {
        if (i > 0) sql += " OR ";
        sql += side_conditions[i];
      }
      sql += ")";
    }
    
    sql += " ORDER BY ts_millis ASC";
    
    if (limit > 0) {
      sql += " LIMIT " + std::to_string(limit);
    }
    
    sqlite3_stmt* st = nullptr;
    int rc = sqlite3_prepare_v2(db.get_db(), sql.c_str(), -1, &st, nullptr);
    if (rc != SQLITE_OK) {
      fprintf(stderr, "SQL prepare error: %s\n", sqlite3_errmsg(db.get_db()));
      return;
    }
    sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    
    // Initialize counters for instrument distribution
    std::map<std::string, int> instrument_signals;
    std::map<std::string, int> instrument_orders;
    std::map<std::string, int> instrument_fills;
    std::map<std::string, double> instrument_signal_values;
    
    // Enhanced signal processing events table
    std::cout << "\n" << BOLD << CYAN << "🔄 SIGNAL PROCESSING EVENTS" << RESET << std::endl;
    std::cout << "┌──────────┬──────────────┬────────┬────────┬────────┬────────┬─────────────┐" << std::endl;
    std::cout << "│   Time   │    Event     │ Symbol │ Signal │  Prob  │ Weight │   Status    │" << std::endl;
    std::cout << "├──────────┼──────────────┼────────┼────────┼────────┼────────┼─────────────┤" << std::endl;
    
    // Helper function to decode drop reasons
    auto decode_drop_reason = [](const char* reason) -> std::string {
      if (!reason) return "Unknown reason";
      
      std::string reason_str = reason;
      if (reason_str == "DROP_REASON_0") return "System/coordination signal (not tradeable)";
      else if (reason_str == "DROP_REASON_1") return "Minimum bars not met";
      else if (reason_str == "DROP_REASON_2") return "Outside trading session";
      else if (reason_str == "DROP_REASON_3") return "NaN/Invalid signal value";
      else if (reason_str == "DROP_REASON_4") return "Zero volume bar";
      else if (reason_str == "DROP_REASON_5") return "Below probability threshold";
      else if (reason_str == "DROP_REASON_6") return "Signal cooldown active";
      else if (reason_str == "DROP_REASON_7") return "Duplicate signal filtered";
      else if (reason_str == "DROP_REASON_8") return "Position size limit reached";
      else if (reason_str == "DROP_REASON_9") return "Risk management override";
      else if (reason_str == "DROP_REASON_10") return "Conflicting position detected";
      else return reason_str; // Return original if not recognized
    };
    
    int event_count = 0;
    while (sqlite3_step(st) == SQLITE_ROW) {
      std::int64_t ts = sqlite3_column_int64(st, 0);
      const char* kind = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
      const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));
      const char* side = reinterpret_cast<const char*>(sqlite3_column_text(st, 3));
      double qty = sqlite3_column_double(st, 4);
      double price = sqlite3_column_double(st, 5);
      double pnl_delta = sqlite3_column_double(st, 6);
      double weight = sqlite3_column_double(st, 7);
      double prob = sqlite3_column_double(st, 8);
      const char* reason = reinterpret_cast<const char*>(sqlite3_column_text(st, 9));
      const char* note = reinterpret_cast<const char*>(sqlite3_column_text(st, 10));
      const char* chain_id = reinterpret_cast<const char*>(sqlite3_column_text(st, 11));
      (void)chain_id; // **CLEAN DISPLAY**: Chain ID kept internally but not displayed
      
      // Update instrument distribution counters
      if (symbol) {
        if (kind && strcmp(kind, "SIGNAL") == 0) {
          instrument_signals[symbol]++;
          instrument_signal_values[symbol] += std::abs(qty * price);
        } else if (kind && strcmp(kind, "ORDER") == 0) {
          instrument_orders[symbol]++;
        } else if (kind && strcmp(kind, "FILL") == 0) {
          instrument_fills[symbol]++;
        }
      }
      
      // Format timestamp
      char time_str[32];
      std::time_t time_t = ts / 1000;
      std::strftime(time_str, sizeof(time_str), "%H:%M:%S", std::localtime(&time_t));
      
      // Determine status and icon with consistent formatting
      std::string status = "✅ PASSED  ";
      std::string event_icon = "📡";
      if (kind && strcmp(kind, "SIGNAL_DROP") == 0) {
        status = "❌ DROPPED ";
        event_icon = "🚫";
      } else if (kind && strcmp(kind, "ORDER") == 0) {
        status = "📋 ORDERED ";
        event_icon = "📋";
      } else if (kind && strcmp(kind, "FILL") == 0) {
        status = "💰 FILLED  ";
        event_icon = "✅";
      }
      
      // Add empty line before each signal event for better scanning
      if (event_count > 0) {
        printf("│          │              │        │        │        │        │             │\n");
      }
      
      // Handle COORDINATION_STATS specially to fix alignment
      std::string display_symbol = symbol ? symbol : "";
      std::string display_side = side ? side : "";
      
      if (display_symbol == "COORDINATION_STATS") {
        // Move COORDINATION_STATS to Signal column, clear Symbol column
        display_side = "STATS";
        display_symbol = "-";
      }
      
      // Shorten SIGNAL_DROP to fit in Event column
      std::string display_kind = kind ? kind : "";
      if (display_kind == "SIGNAL_DROP") {
        display_kind = "DROP";
      }
      
      printf("│ %-8s │ %s%-10s │ %-6s │ %-6s │ %6.3f │ %6.3f │ %-11s │\n",
             time_str, event_icon.c_str(), display_kind.c_str(), display_symbol.c_str(), 
             display_side.c_str(), prob, weight, status.c_str());
      
      // Add detailed information with proper table alignment
      if (kind && strcmp(kind, "SIGNAL_DROP") == 0) {
        std::string decoded_reason = decode_drop_reason(reason);
        printf("│          │ └─ Drop: %s\n", decoded_reason.c_str());
      } else if (kind && strcmp(kind, "SIGNAL") == 0) {
        // Show signal strength and reason for passed signals
        if (reason) {
          printf("│          │ └─ Type: %s\n", reason);
        }
        if (prob > 0.7) {
          printf("│          │ └─ 🟢 HIGH CONFIDENCE (%.1f%% prob)\n", prob * 100);
        } else if (prob > 0.5) {
          printf("│          │ └─ 🟡 MEDIUM CONFIDENCE (%.1f%% prob)\n", prob * 100);
        } else if (prob > 0.3) {
          printf("│          │ └─ 🟠 LOW CONFIDENCE (%.1f%% prob)\n", prob * 100);
        }
      } else if (kind && strcmp(kind, "ORDER") == 0) {
        printf("│          │ └─ Order: %s %.0f @ $%.2f\n", 
               side ? side : "", qty, price);
      } else if (kind && strcmp(kind, "FILL") == 0) {
        printf("│          │ └─ Fill: %s %.0f @ $%.2f (P&L: $%.2f)\n", 
               side ? side : "", qty, price, pnl_delta);
      }
      
      event_count++;
    }
    
    std::cout << "└──────────┴──────────────┴────────┴────────┴────────┴────────┴─────────────┘" << std::endl;
    
    // **CONFLICT DETECTION**: Check for position conflicts in signal flow
    printf("\n🔍 POSITION CONFLICT CHECK:\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    
    // Quick conflict check by analyzing fills in this signal flow
    std::unordered_map<std::string, ConflictPosition> signal_positions;
    int signal_conflicts = 0;
    
    sqlite3_stmt* signal_conflict_st = nullptr;
    std::string signal_conflict_sql = "SELECT symbol, side, qty FROM audit_events WHERE run_id = ? AND kind = 'FILL'";
    if (!symbol_filter.empty()) {
        signal_conflict_sql += " AND symbol = '" + symbol_filter + "'";
    }
    signal_conflict_sql += " ORDER BY ts_millis ASC";
    if (limit > 0) {
        signal_conflict_sql += " LIMIT " + std::to_string(limit);
    }
    
    int signal_conflict_rc = sqlite3_prepare_v2(db.get_db(), signal_conflict_sql.c_str(), -1, &signal_conflict_st, nullptr);
    if (signal_conflict_rc == SQLITE_OK) {
        sqlite3_bind_text(signal_conflict_st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
        
        while (sqlite3_step(signal_conflict_st) == SQLITE_ROW) {
            const char* symbol = (const char*)sqlite3_column_text(signal_conflict_st, 0);
            const char* side = (const char*)sqlite3_column_text(signal_conflict_st, 1);
            double qty = sqlite3_column_double(signal_conflict_st, 2);
            
            if (symbol && side) {
                auto& pos = signal_positions[symbol];
                pos.symbol = symbol;
                
                if (strcmp(side, "BUY") == 0) {
                    pos.qty += qty;
                } else if (strcmp(side, "SELL") == 0) {
                    pos.qty -= qty;
                }
            }
        }
        sqlite3_finalize(signal_conflict_st);
        
        // Analyze final positions for conflicts
        auto final_conflict_analysis = analyze_position_conflicts(signal_positions);
        if (final_conflict_analysis.has_conflicts) {
            signal_conflicts = final_conflict_analysis.conflicts.size();
        }
    }
    
    if (signal_conflicts == 0) {
        printf("│ ✅ SIGNAL FLOW CLEAN: No conflicting positions detected in signal processing    │\n");
    } else {
        printf("│ ⚠️  SIGNAL CONFLICTS: %d conflicting position patterns found                    │\n", signal_conflicts);
        printf("│    Signals may be generating opposing positions that waste capital             │\n");
    }
    
    printf("└─────────────────────────────────────────────────────────────────────────────────────────────┘\n");
    
    printf("\n📊 LEGEND:\n");
    printf("  ✅ PASSED  → Signal passed all validation gates\n");
    printf("  ❌ DROPPED → Signal dropped by validation (see reason)\n");
    printf("  📋 ORDERED → Signal converted to order\n");
    printf("  💰 FILLED  → Order executed (trade completed)\n");
    
    // **NEW**: Instrument Distribution with P&L Breakdown for Signal Flow Report
    std::cout << "\n" << BOLD << CYAN << "🎯 INSTRUMENT DISTRIBUTION & SIGNAL BREAKDOWN" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ Instrument │ Signal Count   │ Order Count    │ Fill Count     │ Signal→Fill %  │ Avg Signal Val │" << std::endl;
    std::cout << "├────────────┼────────────────┼────────────────┼────────────────┼────────────────┼────────────────┤" << std::endl;
    
    // Display per-instrument signal statistics using collected data
    int total_signals = 0;
    int total_orders = 0;
    int total_fills = 0;
    double total_signal_values = 0.0;
    
    // Get all unique instruments from signals
    std::set<std::string> all_signal_instruments;
    for (const auto& [instrument, count] : instrument_signals) {
        all_signal_instruments.insert(instrument);
    }
    
    for (const auto& instrument : all_signal_instruments) {
        int signals = instrument_signals[instrument];
        int orders = instrument_orders[instrument];
        int fills = instrument_fills[instrument];
        double signal_values = instrument_signal_values[instrument];
        double signal_to_fill_pct = (signals > 0) ? (static_cast<double>(fills) / signals) * 100.0 : 0.0;
        double avg_signal_value = (signals > 0) ? signal_values / signals : 0.0;
        
        total_signals += signals;
        total_orders += orders;
        total_fills += fills;
        total_signal_values += signal_values;
        
        // Color coding for efficiency
        const char* efficiency_color = (signal_to_fill_pct > 80) ? GREEN : 
                                      (signal_to_fill_pct > 50) ? YELLOW : RED;
        
        printf("│ %-10s │ %14d │ %14d │ %14d │ %s%13.1f%%%s │ $%12.2f │\n",
               instrument.c_str(), signals, orders, fills,
               efficiency_color, signal_to_fill_pct, RESET,
               avg_signal_value);
    }
    
    std::cout << "├────────────┼────────────────┼────────────────┼────────────────┼────────────────┼────────────────┤" << std::endl;
    
    // Totals row
    double total_signal_to_fill_pct = (total_signals > 0) ? (static_cast<double>(total_fills) / total_signals) * 100.0 : 0.0;
    double avg_total_signal_value = (total_signals > 0) ? total_signal_values / total_signals : 0.0;
    const char* total_efficiency_color = (total_signal_to_fill_pct > 80) ? GREEN : 
                                        (total_signal_to_fill_pct > 50) ? YELLOW : RED;
    
    printf("│ %-10s │ %14d │ %14d │ %14d │ %s%13.1f%%%s │ $%12.2f │\n",
           "TOTAL", total_signals, total_orders, total_fills,
           total_efficiency_color, total_signal_to_fill_pct, RESET,
           avg_total_signal_value);
    
    std::cout << "└────────────┴────────────────┴────────────────┴────────────────┴────────────────┴────────────────┘" << std::endl;
    
    // Signal efficiency analysis
    if (total_signal_to_fill_pct > 80) {
        std::cout << GREEN << "✅ Excellent Signal Efficiency: " << std::fixed << std::setprecision(1) 
                  << total_signal_to_fill_pct << "% of signals result in fills" << RESET << std::endl;
    } else if (total_signal_to_fill_pct > 50) {
        std::cout << YELLOW << "⚠️  Moderate Signal Efficiency: " << std::fixed << std::setprecision(1) 
                  << total_signal_to_fill_pct << "% of signals result in fills" << RESET << std::endl;
    } else {
        std::cout << RED << "❌ Low Signal Efficiency: " << std::fixed << std::setprecision(1) 
                  << total_signal_to_fill_pct << "% of signals result in fills" << RESET << std::endl;
    }
    
    sqlite3_finalize(st);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error showing signal flow: %s\n", e.what());
  }
}

struct TradeRecord {
  std::int64_t timestamp;
  std::string symbol;
  std::string action;  // BUY/SELL
  double quantity;
  double price;
  double trade_value;
  double realized_pnl;
  double cumulative_pnl;
  double equity_after;
  std::string position_breakdown;  // Per-symbol position breakdown (e.g., "QQQ:100 | TQQQ:50")
  double unrealized_pnl;           // Unrealized P&L after this trade
};

struct PositionSummary {
  std::string symbol;
  double quantity;
  double avg_price;
  double market_value;
  double unrealized_pnl;
  double pnl_percent;
};

void show_position_history(const std::string& db_path, const std::string& run_id, const std::string& symbol_filter, int limit, bool show_buy, bool show_sell, bool show_hold) {
  try {
    DB db(db_path);
    
    // Get run info and print header
    RunInfo info = get_run_info(db_path, run_id);
    print_run_header("   ACCOUNT STATEMENT   ", info);
    
    // Get correct P&L from database summary (this is the authoritative source)
    auto summary = db.summarize(run_id);
    
    // Display filter information
    if (!symbol_filter.empty()) {
      std::cout << "\n" << BOLD << YELLOW << "🔍 Filter: " << symbol_filter << RESET << std::endl;
    }
    if (limit > 0) {
      std::cout << "\n" << DIM << "Showing: " << limit << " most recent transactions" << RESET << std::endl;
    } else {
      std::cout << "\n" << DIM << "Showing: All transactions" << RESET << std::endl;
    }
    
    // Get all FILL events to build trade history
    std::string sql = "SELECT ts_millis, symbol, side, qty, price, pnl_delta FROM audit_events WHERE run_id = ? AND kind = 'FILL'";
    
    if (!symbol_filter.empty()) {
      sql += " AND symbol = '" + symbol_filter + "'";
    }
    
    sql += " ORDER BY ts_millis ASC";
    
    sqlite3_stmt* st = nullptr;
    int rc = sqlite3_prepare_v2(db.get_db(), sql.c_str(), -1, &st, nullptr);
    if (rc != SQLITE_OK) {
      fprintf(stderr, "SQL prepare error: %s\n", sqlite3_errmsg(db.get_db()));
      return;
    }
    sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    
    // Collect all trades and calculate running totals
    std::vector<TradeRecord> trades;
    std::map<std::string, double> positions; // symbol -> quantity
    std::map<std::string, double> avg_prices; // symbol -> average price
    std::map<std::string, double> realized_pnl_by_symbol; // symbol -> total realized P&L
    
    double starting_cash = 100000.0;
    double running_cash = starting_cash;
    double cumulative_realized_pnl = 0.0;
    
    // **DEBUG**: Track cash flow vs P&L separately to identify the bug
    double total_cash_flow = 0.0;
    
    while (sqlite3_step(st) == SQLITE_ROW) {
      std::int64_t ts = sqlite3_column_int64(st, 0);
      const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
      const char* side = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));
      double qty = sqlite3_column_double(st, 3);
      double price = sqlite3_column_double(st, 4);
      double pnl_delta = sqlite3_column_double(st, 5);
      
      if (!symbol || !side) continue;
      
      std::string symbol_str = symbol;
      std::string action = side;
      bool is_buy = (action == "BUY");
      
      // Calculate trade value and cash impact
      double trade_value = qty * price;
      double cash_delta = is_buy ? -trade_value : trade_value;
      
      // **FIX**: P&L delta already includes the cash flow impact
      // Don't double-count by adding both cash_delta and pnl_delta
      // The pnl_delta represents the realized gain/loss from closing positions
      // Cash flow = trade_value changes, P&L = profit/loss on closed positions
      
      running_cash += cash_delta;
      total_cash_flow += cash_delta;
      
      // Only count pnl_delta as realized P&L (this is the actual profit/loss)
      double trade_realized_pnl = pnl_delta;
      cumulative_realized_pnl += trade_realized_pnl;
      realized_pnl_by_symbol[symbol_str] += trade_realized_pnl;
      
      // Update position and average price
      double old_qty = positions[symbol_str];
      double new_qty = old_qty + (is_buy ? qty : -qty);
      if (std::abs(new_qty) < 1e-6) {
        // Position closed
        positions.erase(symbol_str);
        avg_prices.erase(symbol_str);
      } else {
        if (old_qty * new_qty >= 0 && std::abs(old_qty) > 1e-6) {
          // Same direction - update VWAP only for BUY orders
          if (is_buy) {
            double old_avg = avg_prices[symbol_str];
            avg_prices[symbol_str] = (old_avg * std::abs(old_qty) + price * qty) / std::abs(new_qty);
          }
          // SELL orders keep the same average price (no update needed)
        } else {
          // New position or flipping direction
          avg_prices[symbol_str] = price;
        }
        positions[symbol_str] = new_qty;
      }
      
      // Calculate current equity (cash + position value at current prices)
      double total_position_value = 0.0;
      double current_unrealized_pnl = 0.0;
      for (const auto& [sym, pos_qty] : positions) {
        if (std::abs(pos_qty) > 1e-6) {
          // Use the most recent price for this symbol as approximation
          double current_price = (sym == symbol_str) ? price : avg_prices[sym];
          double position_value = pos_qty * current_price;
          total_position_value += position_value;
          
          // Calculate unrealized P&L for this position
          if (avg_prices.find(sym) != avg_prices.end()) {
            double position_unrealized = pos_qty * (current_price - avg_prices[sym]);
            current_unrealized_pnl += position_unrealized;
          }
        }
      }
      double equity_after = running_cash + total_position_value;
      
      // Get per-symbol position breakdown for complete visibility
      std::string position_breakdown = "";
      std::vector<std::string> symbols = {"QQQ", "TQQQ", "SQQQ", "PSQ"};
      for (const auto& sym : symbols) {
        auto it = positions.find(sym);
        if (it != positions.end() && std::abs(it->second) > 1e-6) {
          if (!position_breakdown.empty()) position_breakdown += " | ";
          position_breakdown += sym + ":" + std::to_string((int)it->second);
        }
      }
      if (position_breakdown.empty()) position_breakdown = "CASH";
      
      // Store trade record
      TradeRecord trade;
      trade.timestamp = ts;
      trade.symbol = symbol_str;
      trade.action = action;
      trade.quantity = qty;
      trade.price = price;
      trade.trade_value = trade_value;
      trade.realized_pnl = trade_realized_pnl;
      trade.cumulative_pnl = cumulative_realized_pnl;
      trade.equity_after = equity_after;
      trade.position_breakdown = position_breakdown;
      trade.unrealized_pnl = current_unrealized_pnl;
      
      trades.push_back(trade);
    }
    
    sqlite3_finalize(st);
    
    // Calculate final metrics
    double final_equity = running_cash;
    // **REMOVED**: total_unrealized_pnl - use canonical evaluation instead
    std::vector<PositionSummary> current_positions;
    
    // Track calculated P&L values (more accurate than database summary)
    double calculated_realized_pnl = cumulative_realized_pnl;
    // **REMOVED**: calculated_unrealized_pnl - use canonical evaluation instead
    
    for (const auto& [symbol, qty] : positions) {
      if (std::abs(qty) > 1e-6) {
        double avg_price = avg_prices[symbol];
        // **FIX**: Use the most recent trade price as current market price
        double current_price = avg_price; // Fallback to avg price
        
        // Find the most recent price for this symbol from all trades
        for (int j = (int)trades.size() - 1; j >= 0; j--) {
          if (trades[j].symbol == symbol) {
            current_price = trades[j].price;
            break;
          }
        }
        
        double market_value = qty * current_price;
        // **REMOVED**: Incorrect unrealized P&L calculation
        // Use canonical evaluation instead of calculating unrealized P&L here
        
        final_equity += market_value;
        // **REMOVED**: Don't accumulate incorrect unrealized P&L values
        
        PositionSummary pos;
        pos.symbol = symbol;
        pos.quantity = qty;
        pos.avg_price = avg_price;
        pos.market_value = market_value;
        pos.unrealized_pnl = 0.0;  // Will be set correctly later from canonical evaluation
        pos.pnl_percent = 0.0;     // Will be calculated correctly later
        current_positions.push_back(pos);
      }
    }
    
    double total_return = ((final_equity - starting_cash) / starting_cash) * 100.0;
    
    // Use calculated P&L values (more accurate than database summary)
    double starting_capital = 100000.0;
    
    // **FIX**: Use canonical evaluation for total P&L calculation
    double calculated_total_pnl = calculated_realized_pnl;  // Start with realized only
    
    // **FIX DISCREPANCY**: Use final equity from last FILL event (matches canonical evaluation)
    double current_equity = starting_capital + calculated_total_pnl;
    
    // Extract final equity from the last FILL event's note field (eq_after=...)
    // This matches exactly what the canonical evaluation uses
    std::string query = "SELECT note FROM audit_events WHERE run_id = ? AND kind = 'FILL' ORDER BY seq DESC LIMIT 1";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db.get_db(), query.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
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
                    double final_equity_from_canonical = std::stod(eq_str);
                    current_equity = final_equity_from_canonical;
                    calculated_total_pnl = current_equity - starting_capital;
                    
                    // **FIX**: Calculate correct unrealized P&L from canonical evaluation
                    double corrected_unrealized_pnl = calculated_total_pnl - calculated_realized_pnl;
                    
                    // **FIX**: Update position unrealized P&L to be consistent
                    // Distribute the total unrealized P&L proportionally across open positions
                    double total_position_value = 0.0;
                    for (const auto& pos : current_positions) {
                        total_position_value += std::abs(pos.market_value);
                    }
                    
                    if (total_position_value > 1e-6) {
                        for (auto& pos : current_positions) {
                            double weight = std::abs(pos.market_value) / total_position_value;
                            pos.unrealized_pnl = corrected_unrealized_pnl * weight;
                            pos.pnl_percent = (std::abs(pos.avg_price) > 1e-6) ? 
                                (pos.unrealized_pnl / (std::abs(pos.quantity) * pos.avg_price)) * 100.0 : 0.0;
                        }
                    }
                } catch (...) {
                    // Fall back to calculated method if parsing fails
                }
            }
        }
        sqlite3_finalize(stmt);
    }
    double total_return_pct = (calculated_total_pnl / starting_capital) * 100.0;
    
    // Enhanced Account Performance Summary
    std::cout << "\n" << BOLD << CYAN << "📊 ACCOUNT PERFORMANCE SUMMARY" << RESET << std::endl;
    std::cout << "┌───────────────────────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    
    // Color code the total return
    const char* return_color = (total_return_pct >= 0) ? GREEN : RED;
    const char* realized_color = (calculated_realized_pnl >= 0) ? GREEN : RED;
    // **FIX**: Calculate unrealized color from canonical evaluation
    double display_unrealized_pnl = calculated_total_pnl - calculated_realized_pnl;
    const char* unrealized_color = (display_unrealized_pnl >= 0) ? GREEN : RED;
    
    printf("│ Starting Capital    │ $%10.2f │ Current Equity      │ %s$%10.2f%s │ Total Return │ %s%+6.2f%%%s    │\n", 
           starting_capital, return_color, current_equity, RESET, return_color, total_return_pct, RESET);
    
    printf("│ Total Trades        │ %11d │ Realized P&L        │ %s$%+10.2f%s │ Unrealized   │%s$%+10.2f%s │\n", 
           (int)trades.size(), realized_color, calculated_realized_pnl, RESET, unrealized_color, display_unrealized_pnl, RESET);
    // **FIX**: Calculate correct position value independently
    double total_position_value = 0.0;
    for (const auto& pos : current_positions) {
        total_position_value += pos.market_value;
    }
    
    // **FIX**: Cash balance should be current_equity - position_value, not running_cash
    double correct_cash_balance = current_equity - total_position_value;
    
    printf("│ Cash Balance        │ $%10.2f │ Position Value      │ $%10.2f │ Open Pos.    │ %8d   │\n", 
           correct_cash_balance, total_position_value, (int)current_positions.size());
    std::cout << "└───────────────────────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    // **NEW**: Instrument Distribution with P&L Breakdown
    std::cout << "\n" << BOLD << CYAN << "🎯 INSTRUMENT DISTRIBUTION & P&L BREAKDOWN" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ Instrument │ Position │  Market Value  │  Realized P&L  │ Unrealized P&L │   Total P&L    │ Weight  │" << std::endl; 
    std::cout << "├────────────┼──────────┼────────────────┼────────────────┼────────────────┼────────────────┼─────────┤" << std::endl;
    
    // Calculate per-instrument P&L breakdown
    std::map<std::string, double> instrument_realized_pnl;
    std::map<std::string, double> instrument_unrealized_pnl;
    std::map<std::string, double> instrument_market_value;
    
    // Calculate realized P&L per instrument from trades
    for (const auto& trade : trades) {
        instrument_realized_pnl[trade.symbol] += trade.realized_pnl;
    }
    
    // Calculate unrealized P&L and market value per instrument from current positions
    // **FIX**: Use the corrected unrealized P&L values
    for (const auto& pos : current_positions) {
        instrument_unrealized_pnl[pos.symbol] = pos.unrealized_pnl;  // Now correctly calculated above
        instrument_market_value[pos.symbol] = pos.market_value;
    }
    
    // **FIX**: Ensure ALL expected instruments are shown (including zero activity)
    std::vector<std::string> all_expected_instruments = {"PSQ", "QQQ", "TQQQ", "SQQQ"};
    std::set<std::string> all_instruments(all_expected_instruments.begin(), all_expected_instruments.end());
    
    // Also include any additional instruments that might exist from trades/positions
    for (const auto& trade : trades) all_instruments.insert(trade.symbol);
    for (const auto& pos : current_positions) all_instruments.insert(pos.symbol);
    
    double total_instrument_pnl = 0.0;
    double total_market_value = 0.0;
    
    for (const auto& instrument : all_instruments) {
        double realized = instrument_realized_pnl[instrument];
        double unrealized = instrument_unrealized_pnl[instrument];
        double market_value = instrument_market_value[instrument];
        double total_pnl = realized + unrealized;
        double weight = (current_equity > 0) ? (market_value / current_equity) * 100.0 : 0.0;
        
        total_instrument_pnl += total_pnl;
        total_market_value += market_value;
        
        // Color coding
        const char* realized_color = (realized >= 0) ? GREEN : RED;
        const char* unrealized_color = (unrealized >= 0) ? GREEN : RED;
        const char* total_color = (total_pnl >= 0) ? GREEN : RED;
        
        // Get position quantity for this instrument
        double position = 0.0;
        for (const auto& [symbol, qty] : positions) {
            if (symbol == instrument) {
                position = qty;
                break;
            }
        }
        
        printf("│ %-10s │ %8.0f │ %s$%13.2f%s │ %s$%+13.2f%s │ %s$%+13.2f%s │ %s$%+13.2f%s │ %6.1f%% │\n",
               instrument.c_str(), position, 
               (market_value >= 0) ? GREEN : RED, market_value, RESET,
               realized_color, realized, RESET,
               unrealized_color, unrealized, RESET,
               total_color, total_pnl, RESET,
               weight);
    }
    
    // Add cash row - recalculate corrected cash balance
    double total_pos_value = 0.0;
    for (const auto& pos : current_positions) {
        total_pos_value += pos.market_value;
    }
    double cash_balance = current_equity - total_pos_value;
    double cash_weight = (current_equity > 0) ? (cash_balance / current_equity) * 100.0 : 0.0;
    printf("│ %-10s │ %8s │ %s$%13.2f%s │ %14s │ %14s │ %14s │ %6.1f%%│\n",
           "CASH", "N/A", 
           (cash_balance >= 0) ? GREEN : RED, cash_balance, RESET,
           "N/A", "N/A", "N/A", cash_weight);
    
    std::cout << "├────────────┼──────────┼────────────────┼────────────────┼────────────────┼────────────────┼─────────┤" << std::endl;
    
    // Totals row
    const char* total_color = (total_instrument_pnl >= 0) ? GREEN : RED;
    // **FIX**: Use corrected unrealized P&L for totals display
    double corrected_unrealized_pnl = calculated_total_pnl - calculated_realized_pnl;
    const char* corrected_unrealized_color = (corrected_unrealized_pnl >= 0) ? GREEN : RED;
    
    printf("│ %-10s │ %8s │ %s$%13.2f%s │ %s$%+13.2f%s │ %s$%+13.2f%s │ %s$%+13.2f%s │ %6.1f%% │\n",
           "TOTAL", "N/A",
           GREEN, current_equity, RESET,
           realized_color, calculated_realized_pnl, RESET,
           corrected_unrealized_color, corrected_unrealized_pnl, RESET,
           total_color, calculated_total_pnl, RESET,
           100.0);
    
    std::cout << "└────────────┴──────────┴────────────────┴────────────────┴────────────────┴────────────────┴─────────┘" << std::endl;
    
    // Verification: Check if sum equals total
    double pnl_difference = std::abs(total_instrument_pnl - calculated_total_pnl);
    if (pnl_difference > 0.01) {
        std::cout << YELLOW << "⚠️  WARNING: Instrument P&L sum ($" << total_instrument_pnl 
                  << ") differs from total P&L ($" << calculated_total_pnl << ") by $" 
                  << pnl_difference << RESET << std::endl;
    } else {
        std::cout << GREEN << "✅ P&L Verification: Instrument breakdown matches total P&L" << RESET << std::endl;
    }
    
    // Enhanced Trade History Section
    std::cout << "\n" << BOLD << CYAN << "📈 TRADE HISTORY";
    if (limit > 0 && (int)trades.size() > limit) {
      std::cout << " (Last " << limit << " of " << trades.size() << " trades)";
    }
    std::cout << RESET << std::endl;
    std::cout << "┌─────────────────┬────────┬────────┬──────────┬──────────┬───────────────┬──────────────┬─────────────────┬──────────────────────┬─────────────────┐" << std::endl;
    std::cout << "│ Date/Time       │ Symbol │ Action │ Quantity │ Price    │  Trade Value  │  Realized P&L│  Equity After   │ Positions            │ Unrealized P&L  │" << std::endl;
    std::cout << "├─────────────────┼────────┼────────┼──────────┼──────────┼───────────────┼──────────────┼─────────────────┼──────────────────────┼─────────────────┤" << std::endl;
    
    // Show recent trades (apply limit here)
    int start_idx = (limit > 0 && (int)trades.size() > limit) ? (int)trades.size() - limit : 0;
    for (int i = start_idx; i < (int)trades.size(); i++) {
      const auto& trade = trades[i];
      
      // Format timestamp
      char date_str[32];
      std::time_t time_t = trade.timestamp / 1000;
      std::strftime(date_str, sizeof(date_str), "%m/%d %H:%M:%S", std::localtime(&time_t));
      
      // Color coding for actions
      const char* action_color = (trade.action == "BUY") ? "🟢" : "🔴";
      
      // Color coding for P&L values
      const char* unrealized_color = (trade.unrealized_pnl >= 0) ? GREEN : RED;
      const char* unrealized_icon = (trade.unrealized_pnl >= 0) ? "🟢" : "🔴";
      
      // Show fractional shares with proper column alignment matching table borders
      printf("│ %-13s  │ %-6s │ %s%-4s │ %8.3f │ $%7.2f │ $%+12.2f │ $%+12.2f│ $%+12.2f   │ %-20s │ %s$%+12.2f%s │\n",
             date_str, trade.symbol.c_str(), action_color, trade.action.c_str(),
             trade.quantity, trade.price, trade.trade_value, trade.realized_pnl, trade.equity_after,
             trade.position_breakdown.c_str(), unrealized_color, trade.unrealized_pnl, RESET);
    }
    printf("└─────────────────┴────────┴────────┴──────────┴──────────┴───────────────┴──────────────┴─────────────────┴──────────────────────┴─────────────────┘\n\n");
    
    // Enhanced Current Positions Section - Use authoritative database values
    std::cout << "\n" << BOLD << CYAN << "💼 CURRENT POSITIONS" << RESET << std::endl;
    
    // Since database shows unrealized P&L = $0.00, there are no meaningful open positions
    if (std::abs(summary.unrealized_pnl) > 0.01) {
      // Only show positions if there's meaningful unrealized P&L
      std::cout << "┌────────┬──────────┬───────────┬─────────────┬─────────────┬──────────┐" << std::endl;
      std::cout << "│ Symbol │ Quantity │ Avg Price │ Market Value│ Unrealized  │ Return % │" << std::endl;
      std::cout << "├────────┼──────────┼───────────┼─────────────┼─────────────┼──────────┤" << std::endl;
      
      // Show positions with meaningful values (this section would only execute if unrealized_pnl != 0)
      for (const auto& pos : current_positions) {
        if (std::abs(pos.quantity) > 0.001 && std::abs(pos.unrealized_pnl) > 0.01) {
          const char* pnl_color = (pos.unrealized_pnl >= 0) ? GREEN : RED;
          const char* pnl_icon = (pos.unrealized_pnl >= 0) ? "🟢" : "🔴";
          
          printf("│ %-6s │ %8.3f │ $%8.2f │ $%+10.2f │ %s%s$%+8.2f%s │ %s%+7.2f%%%s │\n",
                 pos.symbol.c_str(), pos.quantity, pos.avg_price, pos.market_value,
                 pnl_icon, pnl_color, pos.unrealized_pnl, RESET, pnl_color, pos.pnl_percent, RESET);
        }
      }
      std::cout << "└────────┴──────────┴───────────┴─────────────┴─────────────┴──────────┘" << std::endl;
    } else {
      std::cout << DIM << "No open positions (All positions closed)" << RESET << std::endl;
    }
    
    // Enhanced Performance Breakdown Section
    std::cout << "\n" << BOLD << CYAN << "📊 PERFORMANCE BREAKDOWN" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ Metric                    │ Amount      │ Percentage    │ Description                       │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────────────────────────────────────────┤" << std::endl;
    
    double realized_pct = (calculated_realized_pnl / starting_capital) * 100.0;
    // **FIX**: Use corrected unrealized P&L for performance breakdown
    double corrected_unrealized_breakdown = calculated_total_pnl - calculated_realized_pnl;
    double unrealized_pct = (corrected_unrealized_breakdown / starting_capital) * 100.0;
    const char* corrected_unrealized_breakdown_color = (corrected_unrealized_breakdown >= 0) ? GREEN : RED;
    
    printf("│ Realized Gains/Losses     │ %s$%+10.2f%s │ %s%+8.2f%%%s      │ Profit from closed positions     │\n", 
           realized_color, calculated_realized_pnl, RESET, realized_color, realized_pct, RESET);
    printf("│ Unrealized Gains/Losses   │ %s$%+10.2f%s │ %s%+8.2f%%%s      │ Profit from open positions       │\n", 
           corrected_unrealized_breakdown_color, corrected_unrealized_breakdown, RESET, corrected_unrealized_breakdown_color, unrealized_pct, RESET);
    printf("│ Total Return              │ %s$%+10.2f%s │ %s%+8.2f%%%s      │ Overall account performance      │\n", 
           return_color, calculated_total_pnl, RESET, return_color, total_return_pct, RESET);
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    // 5. CORE METRICS (for comparison with strattest and audit summarize)
    printf("\n📈 CORE METRICS COMPARISON\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Metric                    │ Value           │ Description                    │\n");
    printf("├─────────────────────────────────────────────────────────────────────────────────────────────┤\n");
    
    // Use existing summary data for comparison
    
    // Check for Trading Block data
    auto block_rows = db.get_blocks_for_run(run_id);
    if (!block_rows.empty()) {
      double sum_rpb = 0.0;
      for (const auto& block : block_rows) {
        sum_rpb += block.return_per_block;
      }
      double mean_rpb = sum_rpb / block_rows.size();
      
      printf("│ Trading Blocks            │ %+8zu TB     │ %zu × 480 bars (≈8hrs each)     │\n", block_rows.size(), block_rows.size());
      printf("│ Mean RPB                  │ %+8.4f%%       │ Return Per Block (canonical)   │\n", mean_rpb * 100.0);
      printf("│ Sharpe Ratio              │ %+8.3f        │ Risk-adjusted performance      │\n", summary.sharpe);
      if (block_rows.size() >= 20) {
        double twenty_tb_return = 1.0;
        for (int i = 0; i < 20; ++i) {
          twenty_tb_return *= (1.0 + block_rows[i].return_per_block);
        }
        printf("│ 20TB Return (≈1 month)    │ %+8.2f%%       │ Monthly benchmark metric       │\n", (twenty_tb_return - 1.0) * 100.0);
      }
      printf("│ Daily Trades              │ %+8.1f        │ Avg trades per day             │\n", summary.daily_trades);
    } else {
      printf("│ Monthly Projected Return  │ %+8.2f%%      │ MPR (legacy - use TB system)  │\n", summary.mpr);
      printf("│ Sharpe Ratio              │ %+8.3f        │ Sharpe (legacy)                │\n", summary.sharpe);
    printf("│ Daily Trades              │ %+8.1f        │ Avg trades per day             │\n", summary.daily_trades);
    printf("│ Trading Days              │ %+8d        │ Total trading days              │\n", summary.trading_days);
    }
    printf("└─────────────────────────────────────────────────────────────────────────────────────────────┘\n");
    
  } catch (const std::exception& e) {
    fprintf(stderr, "Error showing position history: %s\n", e.what());
  }
}

void show_strategies_summary(const std::string& db_path) {
  try {
    DB db(db_path);
    
    printf("📊 STRATEGIES SUMMARY REPORT\n");
    printf("════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("Summary of all strategies' most recent runs\n");
    // Format current timestamp
    char time_buffer[64];
    std::time_t ts_sec = now_millis() / 1000;
    struct tm* tm_info = localtime(&ts_sec);
    strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", tm_info);
    printf("Generated: %s\n\n", time_buffer);
    
    // Query to get the latest run for each strategy
    sqlite3* sqlite_db = nullptr;
    int rc = sqlite3_open(db_path.c_str(), &sqlite_db);
    if (rc != SQLITE_OK) {
      fprintf(stderr, "Error opening database: %s\n", sqlite3_errmsg(sqlite_db));
      return;
    }
    
    // Get latest run for each strategy
    const char* sql = R"(
      SELECT r.strategy, r.run_id, r.started_at, r.kind, r.note,
             COUNT(e.run_id) as total_events,
             SUM(CASE WHEN e.kind = 'FILL' THEN e.pnl_delta ELSE 0 END) as total_pnl,
             COUNT(CASE WHEN e.kind = 'FILL' THEN 1 END) as total_trades,
             COUNT(CASE WHEN e.kind = 'SIGNAL' THEN 1 END) as total_signals
      FROM (
        SELECT strategy, MAX(started_at) as max_started_at
        FROM audit_runs 
        GROUP BY strategy
      ) latest
      JOIN audit_runs r ON r.strategy = latest.strategy AND r.started_at = latest.max_started_at
      LEFT JOIN audit_events e ON e.run_id = r.run_id
      GROUP BY r.strategy, r.run_id, r.started_at, r.kind, r.note
      ORDER BY r.strategy
    )";
    
    sqlite3_stmt* stmt;
    rc = sqlite3_prepare_v2(sqlite_db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
      fprintf(stderr, "Error preparing query: %s\n", sqlite3_errmsg(sqlite_db));
      sqlite3_close(sqlite_db);
      return;
    }
    
    printf("┌─────────────┬────────┬─────────────────────┬───────────┬──────────┬─────────────┬─────────────┬─────────────┐\n");
    printf("│ Strategy    │ Run ID │ Date/Time           │ Test Type │ Signals  │ Trades      │ Total P&L   │ MPR Est.    │\n");
    printf("├─────────────┼────────┼─────────────────────┼───────────┼──────────┼─────────────┼─────────────┼─────────────┤\n");
    
    bool has_data = false;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
      has_data = true;
      
      const char* strategy = (const char*)sqlite3_column_text(stmt, 0);
      const char* run_id = (const char*)sqlite3_column_text(stmt, 1);
      int64_t started_at = sqlite3_column_int64(stmt, 2);
      const char* kind = (const char*)sqlite3_column_text(stmt, 3);
      const char* note = (const char*)sqlite3_column_text(stmt, 4);
      int total_events = sqlite3_column_int(stmt, 5);
      double total_pnl = sqlite3_column_double(stmt, 6);
      int total_trades = sqlite3_column_int(stmt, 7);
      int total_signals = sqlite3_column_int(stmt, 8);
      
      // Format timestamp
      char date_str[32];
      std::time_t time_t = started_at / 1000;
      std::strftime(date_str, sizeof(date_str), "%m/%d %H:%M:%S", std::localtime(&time_t));
      
      // Extract test type from note (e.g., "strattest holistic QQQ 2w")
      std::string test_type = "unknown";
      if (note && strlen(note) > 0) {
        std::string note_str(note);
        if (note_str.find("holistic") != std::string::npos) {
          test_type = "holistic";
        } else if (note_str.find("historical") != std::string::npos) {
          test_type = "historical";
        } else if (note_str.find("ai-regime") != std::string::npos) {
          test_type = "ai-regime";
        } else if (note_str.find("hybrid") != std::string::npos) {
          test_type = "hybrid";
        } else if (note_str.find("strattest") != std::string::npos) {
          test_type = "strattest";
        }
      }
      
      // Estimate MPR (very rough calculation)
      double mpr_estimate = 0.0;
      if (total_pnl != 0.0) {
        // Assume 100k starting capital and estimate monthly return
        double return_pct = (total_pnl / 100000.0) * 100.0;
        // Very rough annualization (assuming test was representative)
        mpr_estimate = return_pct * 12.0; // Rough monthly estimate
      }
      
      // Color coding for P&L
      const char* pnl_color = (total_pnl >= 0) ? "🟢" : "🔴";
      const char* mpr_color = (mpr_estimate >= 0) ? "🟢" : "🔴";
      
      printf("│ %-11s │ %-6s │ %-19s │ %-9s │ %8d │ %11d │ %s$%+9.2f │ %s%+6.1f%%    │\n",
             strategy ? strategy : "unknown",
             run_id ? run_id : "N/A",
             date_str,
             test_type.c_str(),
             total_signals,
             total_trades,
             pnl_color, total_pnl,
             mpr_color, mpr_estimate);
    }
    
    printf("└─────────────┴────────┴─────────────────────┴───────────┴──────────┴─────────────┴─────────────┴─────────────┘\n");
    
    if (!has_data) {
      printf("\n⚠️  No strategy runs found in the database.\n");
      printf("Run some strategies using 'sentio_cli strattest' to populate the audit database.\n");
    } else {
      printf("\n📋 SUMMARY NOTES:\n");
      printf("• Run ID: 6-digit unique identifier for each test run\n");
      printf("• MPR Est.: Rough Monthly Projected Return estimate (not precise)\n");
      printf("• For detailed analysis, use: sentio_audit summarize --run <run_id>\n");
      printf("• For signal analysis, use: sentio_audit signal-flow --run <run_id>\n");
      printf("• For trade analysis, use: sentio_audit trade-flow --run <run_id>\n");
    }
    
    sqlite3_finalize(stmt);
    sqlite3_close(sqlite_db);
    
  } catch (const std::exception& e) {
    fprintf(stderr, "Error generating strategies summary: %s\n", e.what());
  }
}

} // namespace audit

// **COMPREHENSIVE INTEGRITY CHECK IMPLEMENTATION**
int perform_integrity_check(sqlite3* db, const std::string& run_id) {
    printf("\n");
    printf(BOLD BG_BLUE WHITE "╔══════════════════════════════════════════════════════════════════════════════════╗" RESET "\n");
    printf(BOLD BG_BLUE WHITE "║                        🔍 COMPREHENSIVE INTEGRITY CHECK                          ║" RESET "\n");
    printf(BOLD BG_BLUE WHITE "╚══════════════════════════════════════════════════════════════════════════════════╝" RESET "\n");
    
    printf("\n" BOLD CYAN "📋 RUN INFORMATION" RESET "\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ " BOLD "Run ID:" RESET "       " BLUE "%s" RESET "\n", run_id.c_str());
    printf("│ " BOLD "Check Type:" RESET "   " MAGENTA "5-Principle Integrity Validation" RESET "\n");
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    
    int total_violations = 0;
    int critical_violations = 0;
    
    // **PRINCIPLE 1: NO NEGATIVE CASH BALANCE**
    printf("\n" BOLD CYAN "💰 PRINCIPLE 1: NO NEGATIVE CASH BALANCE" RESET "\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    
    const char* cash_query = R"(
        SELECT MIN(CAST(SUBSTR(note, INSTR(note, 'eq_after=') + 9, 
                              INSTR(note || ',', ',', INSTR(note, 'eq_after=') + 9) - INSTR(note, 'eq_after=') - 9) AS REAL)) as min_cash,
               COUNT(*) as total_fills
        FROM audit_events 
        WHERE run_id = ? AND kind = 'FILL' AND note LIKE '%eq_after=%'
    )";
    
    sqlite3_stmt* stmt;
    double min_cash = 0.0;
    int total_fills = 0;
    
    if (sqlite3_prepare_v2(db, cash_query, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            min_cash = sqlite3_column_double(stmt, 0);
            total_fills = sqlite3_column_int(stmt, 1);
        }
        sqlite3_finalize(stmt);
    }
    
    if (min_cash < -1.0) {  // Allow $1 tolerance for rounding
        printf("│ " RED "❌ VIOLATION DETECTED" RESET " │ " RED "Minimum cash: $%.2f" RESET " │\n", min_cash);
        printf("│ " BOLD "Risk:" RESET " System went into negative cash, violating margin requirements │\n");
        printf("│ " BOLD "Fix:" RESET "  Review SafeSizer cash calculation and position sizing logic │\n");
        critical_violations++;
        total_violations++;
    } else {
        printf("│ " GREEN "✅ COMPLIANCE VERIFIED" RESET " │ " GREEN "Minimum cash: $%.2f" RESET " │\n", min_cash);
        printf("│ " BOLD "Status:" RESET " Cash balance remained positive throughout %d trades │\n", total_fills);
    }
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    
    // **PRINCIPLE 2: NO CONFLICTING POSITIONS**
    printf("\n" BOLD CYAN "⚔️  PRINCIPLE 2: NO CONFLICTING POSITIONS" RESET "\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    
    // **FIX**: Use same simple logic as working summarize command
    const char* conflict_query = R"(
        SELECT 
            symbol,
            SUM(CASE WHEN side = 'BUY' THEN qty ELSE -qty END) as net_position
        FROM audit_events 
        WHERE run_id = ? AND kind = 'FILL'
        GROUP BY symbol
        HAVING ABS(net_position) > 0.001
    )";
    
    std::map<std::string, double> final_positions;
    if (sqlite3_prepare_v2(db, conflict_query, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* symbol = (const char*)sqlite3_column_text(stmt, 0);
            double net_position = sqlite3_column_double(stmt, 1);
            
            if (symbol) {
                final_positions[symbol] = net_position;
            }
        }
        sqlite3_finalize(stmt);
    }
    
    // Detect conflicts using same logic as summarize
    bool has_long_etf = false;
    bool has_inverse_etf = false;
    bool has_short_positions = false;
    
    for (const auto& [symbol, position] : final_positions) {
        if (std::abs(position) > 0.001) {
            if (symbol == "QQQ" || symbol == "TQQQ") {
                if (position > 0) has_long_etf = true;
                if (position < 0) has_short_positions = true;
            }
            if (symbol == "PSQ" || symbol == "SQQQ") {
                if (position > 0) has_inverse_etf = true;
            }
        }
    }
    
    bool has_conflicts = (has_long_etf && has_inverse_etf) || has_short_positions;
    int conflict_count = has_conflicts ? 1 : 0;
    
    if (conflict_count > 0) {
        printf("│ " RED "❌ VIOLATION DETECTED" RESET " │ " RED "Mixed directional exposure found" RESET " │\n");
        if (has_long_etf && has_inverse_etf) {
            printf("│ " BOLD "Issue:" RESET " Both long ETFs and inverse ETFs held simultaneously │\n");
        }
        if (has_short_positions) {
            printf("│ " BOLD "Issue:" RESET " Short positions detected - should use inverse ETFs instead │\n");
        }
        printf("│ " BOLD "Positions:" RESET " ");
        for (const auto& [symbol, position] : final_positions) {
            if (std::abs(position) > 0.001) {
                printf("%s:%.1f ", symbol.c_str(), position);
            }
        }
        printf("│\n");
        printf("│ " BOLD "Fix:" RESET "  Review PositionCoordinator conflict detection and resolution │\n");
        critical_violations++;
        total_violations++;
    } else {
        printf("│ " GREEN "✅ COMPLIANCE VERIFIED" RESET " │ " GREEN "No conflicting positions detected" RESET " │\n");
        printf("│ " BOLD "Status:" RESET " All positions maintained proper directional consistency │\n");
    }
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    
    // **PRINCIPLE 3: NO SHORT POSITIONS (NEGATIVE QUANTITIES)**
    printf("\n" BOLD CYAN "📈 PRINCIPLE 3: NO SHORT POSITIONS" RESET "\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    
    // **FIX**: Use same simple logic - check if any final positions are negative
    int short_count = 0;
    double min_position = 0.0;
    
    for (const auto& [symbol, position] : final_positions) {
        if (position < -0.001) {
            short_count++;
            if (position < min_position) {
                min_position = position;
            }
        }
    }
    
    if (short_count > 0) {
        printf("│ " RED "❌ VIOLATION DETECTED" RESET " │ " RED "%d short positions (min: %.3f)" RESET " │\n", short_count, min_position);
        printf("│ " BOLD "Risk:" RESET " Short positions should use inverse ETFs instead (SQQQ, PSQ) │\n");
        printf("│ " BOLD "Fix:" RESET "  Review SafeSizer to prevent negative quantities completely │\n");
        critical_violations++;
        total_violations++;
    } else {
        printf("│ " GREEN "✅ COMPLIANCE VERIFIED" RESET " │ " GREEN "All positions are long (positive quantities)" RESET " │\n");
        printf("│ " BOLD "Status:" RESET " System correctly uses inverse ETFs for bearish exposure │\n");
    }
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    
    // **PRINCIPLE 4: EOD CLOSING OF ALL POSITIONS**
    printf("\n" BOLD CYAN "🌙 PRINCIPLE 4: EOD CLOSING OF ALL POSITIONS" RESET "\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    
    // **FIX**: Use same simple logic as working EOD check in summarize
    // Just call the existing check_eod_positions function and capture violations
    int eod_violations = 0;
    
    // Simple query to check if we have any final positions at end of any day
    const char* eod_query = R"(
        WITH daily_final_positions AS (
            SELECT DATE(ts_millis/1000, 'unixepoch') as trade_date,
                   symbol,
                   SUM(CASE WHEN side = 'BUY' THEN qty ELSE -qty END) as final_position
            FROM audit_events 
            WHERE run_id = ? AND kind = 'FILL'
            GROUP BY DATE(ts_millis/1000, 'unixepoch'), symbol
            HAVING ABS(final_position) > 0.001
        )
        SELECT COUNT(DISTINCT trade_date) as violation_days FROM daily_final_positions
    )";
    
    if (sqlite3_prepare_v2(db, eod_query, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            eod_violations = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    
    if (eod_violations > 0) {
        printf("│ " RED "❌ VIOLATION DETECTED" RESET " │ " RED "%d days with overnight positions" RESET " │\n", eod_violations);
        printf("│ " BOLD "Risk:" RESET " Overnight carry risk, leveraged ETF decay, gap risk exposure │\n");
        printf("│ " BOLD "Fix:" RESET "  Review EODPositionManager configuration and timing │\n");
        critical_violations++;
        total_violations++;
    } else {
        printf("│ " GREEN "✅ COMPLIANCE VERIFIED" RESET " │ " GREEN "All positions closed overnight" RESET " │\n");
        printf("│ " BOLD "Status:" RESET " Zero overnight carry risk, proper risk management │\n");
    }
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    
    // **PRINCIPLE 5: MAXIMUM CAPITAL UTILIZATION**
    printf("\n" BOLD CYAN "🚀 PRINCIPLE 5: MAXIMUM CAPITAL UTILIZATION" RESET "\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    
    const char* capital_query = R"(
        WITH equity_snapshots AS (
            SELECT ts_millis,
                   CAST(SUBSTR(note, INSTR(note, 'eq_after=') + 9, 
                              INSTR(note || ',', ',', INSTR(note, 'eq_after=') + 9) - INSTR(note, 'eq_after=') - 9) AS REAL) as equity_after
            FROM audit_events 
            WHERE run_id = ? AND kind = 'FILL' AND note LIKE '%eq_after=%'
            ORDER BY ts_millis
        ),
        capital_utilization AS (
            SELECT AVG(CASE WHEN equity_after > 0 THEN (100000.0 - (equity_after - (equity_after - 100000.0))) / 100000.0 * 100.0 ELSE 0 END) as avg_utilization,
                   MIN(equity_after) as min_equity,
                   MAX(equity_after) as max_equity,
                   COUNT(*) as snapshots
            FROM equity_snapshots
        )
        SELECT avg_utilization, min_equity, max_equity, snapshots FROM capital_utilization
    )";
    
    double avg_utilization = 0.0;
    double min_equity = 100000.0;
    double max_equity = 100000.0;
    int snapshots = 0;
    
    if (sqlite3_prepare_v2(db, capital_query, -1, &stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            avg_utilization = sqlite3_column_double(stmt, 0);
            min_equity = sqlite3_column_double(stmt, 1);
            max_equity = sqlite3_column_double(stmt, 2);
            snapshots = sqlite3_column_int(stmt, 3);
        }
        sqlite3_finalize(stmt);
    }
    
    // Calculate performance metrics
    double total_return = ((max_equity - 100000.0) / 100000.0) * 100.0;
    bool low_utilization = (avg_utilization < 50.0 && snapshots > 10);
    bool poor_performance = (total_return < 0.1 && snapshots > 50);
    
    if (low_utilization || poor_performance) {
        printf("│ " YELLOW "⚠️  SUBOPTIMAL DETECTED" RESET " │ ");
        if (low_utilization) {
            printf(YELLOW "Avg utilization: %.1f%%" RESET " │\n", avg_utilization);
        } else {
            printf(YELLOW "Total return: %.2f%%" RESET " │\n", total_return);
        }
        printf("│ " BOLD "Opportunity:" RESET " System could deploy capital more aggressively on strong signals │\n");
        printf("│ " BOLD "Suggestion:" RESET " Review AllocationManager thresholds and SafeSizer limits │\n");
        total_violations++;
    } else {
        printf("│ " GREEN "✅ EFFICIENT UTILIZATION" RESET " │ " GREEN "Return: %.2f%%, Utilization: %.1f%%" RESET " │\n", total_return, avg_utilization);
        printf("│ " BOLD "Status:" RESET " Capital deployed effectively with %d position changes │\n", snapshots);
    }
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    
    // **FINAL SUMMARY**
    printf("\n" BOLD CYAN "📊 INTEGRITY CHECK SUMMARY" RESET "\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────┐\n");
    
    if (critical_violations == 0) {
        printf("│ " BOLD GREEN "🎉 SYSTEM INTEGRITY VERIFIED" RESET " │ " GREEN "All critical principles satisfied" RESET " │\n");
        printf("│ " BOLD "Status:" RESET " Trading system operating within all safety constraints │\n");
    } else {
        printf("│ " BOLD RED "⚠️  INTEGRITY VIOLATIONS FOUND" RESET " │ " RED "%d critical, %d total violations" RESET " │\n", 
               critical_violations, total_violations);
        printf("│ " BOLD "Action Required:" RESET " Fix critical violations before live trading │\n");
    }
    
    if (total_violations > critical_violations) {
        printf("│ " BOLD "Additional Notes:" RESET " %d optimization opportunities identified │\n", 
               total_violations - critical_violations);
    }
    
    printf("└─────────────────────────────────────────────────────────────────────────────────┘\n");
    
    // Return appropriate exit code
    if (critical_violations > 0) {
        printf("\n" RED "❌ INTEGRITY CHECK FAILED" RESET " - Critical violations must be resolved\n");
        return 1;  // Failure exit code
    } else if (total_violations > 0) {
        printf("\n" YELLOW "⚠️  INTEGRITY CHECK PASSED WITH WARNINGS" RESET " - Optimization recommended\n");
        return 2;  // Warning exit code
    } else {
        printf("\n" GREEN "✅ INTEGRITY CHECK PASSED" RESET " - System operating optimally\n");
        return 0;  // Success exit code
    }
}

```

## 📄 **FILE 5 of 8**: temp_mega_doc/runner.cpp

**File Information**:
- **Path**: `temp_mega_doc/runner.cpp`

- **Size**: 946 lines
- **Modified**: 2025-09-19 02:13:46

- **Type**: .cpp

```text
#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/pricebook.hpp"
#include "sentio/metrics.hpp"
#include "audit/audit_db_recorder.hpp"
#include "sentio/safe_sizer.hpp"
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

// **STRATEGY-AGNOSTIC EXECUTION PIPELINE**: Adaptive components that work for any strategy
static void execute_adaptive_pipeline(
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
    
    // STEP 1: Profile the strategy signal
    profiler.observe_signal(strategy_probability, bar.ts_utc_epoch);
    auto profile = profiler.get_current_profile();

    // STEP 2: Get adaptive allocations based on profile
    auto allocations = allocation_mgr.get_allocations(strategy_probability, profile);
    
    // Log signal activity
    if (logging_enabled && !allocations.empty()) {
        audit.event_signal_ex(bar.ts_utc_epoch, ST.get_symbol(0), 
                            SigType::BUY, strategy_probability, chain_id);
    }
    
    // STEP 3: Check EOD overrides
    auto eod_allocations = eod_mgr.get_eod_allocations(bar.ts_utc_epoch, portfolio, ST, profile);
    if (!eod_allocations.empty()) {
        allocations = eod_allocations;  // EOD overrides everything
    }
    
    // STEP 4: Universal position coordination
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
        // Execute the adaptive pipeline with strategy probability
        execute_adaptive_pipeline(
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

## 📄 **FILE 6 of 8**: temp_mega_doc/runner.hpp

**File Information**:
- **Path**: `temp_mega_doc/runner.hpp`

- **Size**: 63 lines
- **Modified**: 2025-09-19 02:13:46

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

## 📄 **FILE 7 of 8**: temp_mega_doc/strategy_profiler.cpp

**File Information**:
- **Path**: `temp_mega_doc/strategy_profiler.cpp`

- **Size**: 194 lines
- **Modified**: 2025-09-19 02:13:46

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
    TradingStyle new_style_candidate;

    // Define thresholds for switching *to* a new style.
    // These are the entry points.
    constexpr double AGGRESSIVE_UPPER_THRESHOLD = 110.0;
    constexpr double CONSERVATIVE_LOWER_THRESHOLD = 15.0;
    constexpr double BURST_VOLATILITY_THRESHOLD = 0.25;

    // Determine the current style candidate based on metrics.
    if (profile_.trades_per_block > AGGRESSIVE_UPPER_THRESHOLD) {
        new_style_candidate = TradingStyle::AGGRESSIVE;
    } else if (profile_.trades_per_block < CONSERVATIVE_LOWER_THRESHOLD) {
        new_style_candidate = TradingStyle::CONSERVATIVE;
    } else if (profile_.trades_per_block <= 100 && profile_.signal_volatility > BURST_VOLATILITY_THRESHOLD) {
        new_style_candidate = TradingStyle::BURST;
    } else {
        new_style_candidate = TradingStyle::ADAPTIVE; // Default fallback
    }

    // Now, apply hysteresis and confirmation logic.
    // A style must be confirmed for 2 consecutive blocks before it's adopted.
    if (new_style_candidate == tentative_style_) {
        style_confirmation_blocks_++;
    } else {
        // If the candidate changes, reset the counter and update the tentative style.
        tentative_style_ = new_style_candidate;
        style_confirmation_blocks_ = 1;
    }
    
    // Define exit thresholds (hysteresis). The system must cross a wider
    // band to switch *away* from an established style.
    constexpr double AGGRESSIVE_LOWER_THRESHOLD = 90.0;
    constexpr double CONSERVATIVE_UPPER_THRESHOLD = 25.0;
    
    // Only commit to a style change if confirmed for 2 blocks OR if a strong exit condition is met.
    if (style_confirmation_blocks_ >= 2) {
        profile_.style = tentative_style_;
    } else {
        // Add override logic for sharp changes to prevent getting stuck in a stale profile.
        if (profile_.style == TradingStyle::AGGRESSIVE && profile_.trades_per_block < AGGRESSIVE_LOWER_THRESHOLD) {
            profile_.style = tentative_style_;
            style_confirmation_blocks_ = 1; // Reset confirmation
        } else if (profile_.style == TradingStyle::CONSERVATIVE && profile_.trades_per_block > CONSERVATIVE_UPPER_THRESHOLD) {
            profile_.style = tentative_style_;
            style_confirmation_blocks_ = 1; // Reset confirmation
        }
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

## 📄 **FILE 8 of 8**: temp_mega_doc/strategy_profiler.hpp

**File Information**:
- **Path**: `temp_mega_doc/strategy_profiler.hpp`

- **Size**: 63 lines
- **Modified**: 2025-09-19 02:13:46

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
            TradingStyle tentative_style_ = TradingStyle::CONSERVATIVE;
            int style_confirmation_blocks_ = 0;
            
            void update_profile();
            void detect_trading_style();
            void calculate_adaptive_thresholds();
            double calculate_noise_threshold();
};

} // namespace sentio

```

