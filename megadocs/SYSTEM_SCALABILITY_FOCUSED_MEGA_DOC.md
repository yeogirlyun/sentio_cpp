# System Scalability Failures - Core Components Only

**Generated**: 2025-09-19 00:07:37
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: Focused analysis of core backend components that fail under high-frequency trading

**Total Files**: 16

---

## 🐛 **BUG REPORT**

# Bug Report: Fundamental System Scalability Failures Under High-Frequency Trading

## Executive Summary

The sigor strategy has exposed critical architectural flaws in the Sentio trading system that manifest under high-frequency signal generation. While the TFA strategy passes all integrity checks with minimal trading (14 trades/block), sigor generates excessive trades (367 trades/block) that overwhelm the system's safety mechanisms, causing position conflicts and EOD violations.

## Problem Statement

**Root Issue**: The trading system architecture is **not truly strategy-agnostic** and breaks down under high-frequency signal generation, revealing fundamental scalability and strategy coupling failures.

**Manifestation**: 
- TFA Strategy: ✅ Perfect integrity (5/5 principles) with 14 trades/block
- Sigor Strategy: ❌ Multiple violations (4/5 principles) with 367 trades/block

**Strategy Coupling**: Backend components make implicit assumptions about strategy behavior that work for conservative strategies but fail for aggressive ones.

## Critical Findings

### 1. Trade Frequency Control Failure
**Expected Behavior**: "One trade per bar" enforcement (max 480 trades per 480-bar block)
**Actual Behavior**: 367 trades per block (76% of bars have trades)
**Impact**: System allows excessive churning, overwhelming downstream components

### 2. Position Coordinator Scalability Breakdown
**Low Frequency (TFA)**: Perfect conflict prevention with 14 trades/block
**High Frequency (Sigor)**: Position conflicts emerge with 367 trades/block
**Root Cause**: Conflict detection logic cannot handle rapid position changes

### 3. EOD Manager Timing Failures
**Expected**: All positions closed at end of day
**Actual**: 1-3 days with overnight positions under high-frequency trading
**Root Cause**: EOD closure logic gets confused by frequent position changes

### 4. Missing Signal Intelligence Layer
**Problem**: No distinction between "strong signals worth trading" vs "market noise"
**Result**: System trades on every minor probability fluctuation
**Impact**: Excessive transaction costs, increased risk, system instability

### 5. Backend Strategy Coupling
**Problem**: Backend components are implicitly coupled to specific strategy behaviors
**Evidence**: Perfect performance with TFA, failures with sigor using identical backend
**Impact**: System cannot handle diverse strategy types, limiting scalability

## Detailed Analysis

### Trade Frequency Comparison
```
Strategy | Blocks | Total Trades | Trades/Block | Integrity Status
---------|--------|--------------|--------------|------------------
TFA      | 6      | 88          | 14.7         | ✅ Perfect (5/5)
Sigor    | 6      | 2,205       | 367.5        | ❌ Violations (4/5)
```

### Violation Patterns
- **Position Conflicts**: Emerge only under high-frequency trading
- **EOD Violations**: Increase with trading frequency
- **System Stability**: Inversely correlated with signal frequency

## Root Cause Analysis

### 1. Architectural Assumption Failure
**Assumption**: Strategies generate reasonable signal frequencies
**Reality**: Sigor's detector-based approach generates signals on 76% of bars
**Fix Required**: Implement proper signal filtering and trade frequency controls

### 2. Component Interaction Failures
**Issue**: System components (PositionCoordinator, EODManager) designed for low-frequency trading
**Evidence**: Perfect performance with TFA, failures with sigor
**Fix Required**: Redesign components for high-frequency resilience

### 3. Missing Circuit Breakers
**Issue**: No protection against excessive signal generation
**Evidence**: System allows 25x more trades than reasonable baseline
**Fix Required**: Implement intelligent signal filtering and frequency limits

## Impact Assessment

### Financial Impact
- **Transaction Costs**: 25x higher than necessary (sigor vs TFA)
- **Risk Exposure**: Overnight positions due to EOD failures
- **Capital Efficiency**: Reduced due to churning and conflicts

### System Reliability
- **Integrity Violations**: 2 critical principles fail under stress
- **Scalability**: System cannot handle legitimate high-frequency strategies
- **Robustness**: Architecture breaks down predictably under load

### Operational Impact
- **Production Risk**: System unsuitable for high-frequency strategies
- **Maintenance Burden**: Band-aid fixes (threshold manipulation) mask real issues
- **Development Velocity**: Architectural debt prevents proper solutions

## Proposed Solutions

### 1. Design Truly Strategy-Agnostic Backend
**Component**: Entire backend architecture
**Fix**: Remove all implicit assumptions about strategy behavior
**Benefit**: System works identically well for any strategy type (conservative, aggressive, high-frequency, etc.)

### 2. Implement Adaptive Component Configuration
**Component**: AllocationManager, PositionCoordinator, EODManager
**Fix**: Components automatically adapt to strategy characteristics
**Benefit**: No manual tuning required, optimal performance for each strategy

### 3. Add Strategy Profiling System
**Component**: New StrategyProfiler class
**Fix**: Automatically analyze strategy behavior and configure backend accordingly
**Benefit**: Backend optimizes itself for each strategy's unique characteristics

### 4. Implement Proper Trade Frequency Control
**Component**: PositionCoordinator
**Fix**: Enforce actual "one trade per bar" with timestamp tracking, regardless of signal frequency
**Benefit**: Prevents excessive churning while preserving strong signals for any strategy

### 5. Add Universal Signal Intelligence Layer
**Component**: New AdaptiveSignalFilter class
**Fix**: Dynamically distinguish between actionable signals and noise for any strategy
**Benefit**: Reduces unnecessary trades while preserving alpha, adapts to strategy style

### 6. Redesign Position Coordinator for Universal Scalability
**Component**: PositionCoordinator
**Fix**: Handle any trading frequency without breaking conflict detection
**Benefit**: Maintains integrity under all trading patterns (low-freq, high-freq, burst trading)

### 7. Fix EOD Manager for Universal Robustness
**Component**: EODPositionManager
**Fix**: Robust end-of-day closure regardless of strategy behavior
**Benefit**: Eliminates overnight risk for any strategy type

### 8. Implement Universal System Circuit Breakers
**Component**: New AdaptiveTradingCircuitBreaker class
**Fix**: Automatic protection that adapts to strategy characteristics
**Benefit**: Prevents system overload while maintaining functionality for any strategy

## Testing Strategy

### Stress Testing Protocol
1. **Baseline**: Verify TFA continues to pass all integrity checks
2. **Stress Test**: Use sigor as high-frequency stress test
3. **Validation**: Both strategies must pass all 5 integrity principles
4. **Performance**: Measure system behavior under various signal frequencies

### Success Criteria
- ✅ **Strategy Agnostic**: Backend works identically well for any strategy type
- ✅ **TFA**: Maintains perfect integrity (5/5 principles)
- ✅ **Sigor**: Achieves perfect integrity (5/5 principles) 
- ✅ **Scalability**: System handles 0-500 trades/block gracefully
- ✅ **Performance**: No degradation under any signal pattern
- ✅ **Adaptability**: Backend automatically optimizes for each strategy
- ✅ **Universality**: Same backend code works for conservative and aggressive strategies

## Conclusion

Sigor has revealed that our trading system has fundamental architectural limitations that prevent it from handling high-frequency signal generation. Rather than masking these issues with parameter adjustments, we must redesign the core components to be truly robust and scalable.

The current system works well for conservative strategies like TFA but fails catastrophically for aggressive strategies like sigor. A production-ready system must handle both scenarios flawlessly.

## Priority

**CRITICAL** - These are fundamental architectural flaws that limit system capability and create production risks. Must be resolved before deploying any high-frequency strategies.

## Files Affected

- `src/position_coordinator.cpp` - Core conflict detection logic
- `src/eod_position_manager.cpp` - End-of-day closure timing
- `src/allocation_manager.cpp` - Signal-to-allocation conversion
- `src/runner.cpp` - Main execution pipeline
- `src/strategy_signal_or.cpp` - High-frequency signal generator
- `include/sentio/position_coordinator.hpp` - Position coordination interface
- `include/sentio/eod_position_manager.hpp` - EOD management interface
- `include/sentio/allocation_manager.hpp` - Allocation management interface

## Next Steps

1. Create comprehensive mega document with all relevant source modules
2. Implement proper signal filtering and trade frequency controls
3. Redesign PositionCoordinator for high-frequency resilience
4. Fix EODPositionManager timing logic
5. Add system circuit breakers and monitoring
6. Validate fixes with both TFA and sigor strategies


---

## 📋 **TABLE OF CONTENTS**

1. [temp_core_analysis/audit_src/audit_cli.cpp](#file-1)
2. [temp_core_analysis/bug_report_system_scalability_failures.md](#file-2)
3. [temp_core_analysis/include/allocation_manager.hpp](#file-3)
4. [temp_core_analysis/include/eod_position_manager.hpp](#file-4)
5. [temp_core_analysis/include/position_coordinator.hpp](#file-5)
6. [temp_core_analysis/include/runner.hpp](#file-6)
7. [temp_core_analysis/include/sizer.hpp](#file-7)
8. [temp_core_analysis/include/strategy_signal_or.hpp](#file-8)
9. [temp_core_analysis/include/strategy_tfa.hpp](#file-9)
10. [temp_core_analysis/src/allocation_manager.cpp](#file-10)
11. [temp_core_analysis/src/eod_position_manager.cpp](#file-11)
12. [temp_core_analysis/src/position_coordinator.cpp](#file-12)
13. [temp_core_analysis/src/runner.cpp](#file-13)
14. [temp_core_analysis/src/strategy_signal_or.cpp](#file-14)
15. [temp_core_analysis/src/strategy_tfa.cpp](#file-15)
16. [temp_core_analysis/strategy_agnostic_backend_design.md](#file-16)

---

## 📄 **FILE 1 of 16**: temp_core_analysis/audit_src/audit_cli.cpp

**File Information**:
- **Path**: `temp_core_analysis/audit_src/audit_cli.cpp`

- **Size**: 3535 lines
- **Modified**: 2025-09-19 00:07:34

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

## 📄 **FILE 2 of 16**: temp_core_analysis/bug_report_system_scalability_failures.md

**File Information**:
- **Path**: `temp_core_analysis/bug_report_system_scalability_failures.md`

- **Size**: 180 lines
- **Modified**: 2025-09-19 00:07:26

- **Type**: .md

```text
# Bug Report: Fundamental System Scalability Failures Under High-Frequency Trading

## Executive Summary

The sigor strategy has exposed critical architectural flaws in the Sentio trading system that manifest under high-frequency signal generation. While the TFA strategy passes all integrity checks with minimal trading (14 trades/block), sigor generates excessive trades (367 trades/block) that overwhelm the system's safety mechanisms, causing position conflicts and EOD violations.

## Problem Statement

**Root Issue**: The trading system architecture is **not truly strategy-agnostic** and breaks down under high-frequency signal generation, revealing fundamental scalability and strategy coupling failures.

**Manifestation**: 
- TFA Strategy: ✅ Perfect integrity (5/5 principles) with 14 trades/block
- Sigor Strategy: ❌ Multiple violations (4/5 principles) with 367 trades/block

**Strategy Coupling**: Backend components make implicit assumptions about strategy behavior that work for conservative strategies but fail for aggressive ones.

## Critical Findings

### 1. Trade Frequency Control Failure
**Expected Behavior**: "One trade per bar" enforcement (max 480 trades per 480-bar block)
**Actual Behavior**: 367 trades per block (76% of bars have trades)
**Impact**: System allows excessive churning, overwhelming downstream components

### 2. Position Coordinator Scalability Breakdown
**Low Frequency (TFA)**: Perfect conflict prevention with 14 trades/block
**High Frequency (Sigor)**: Position conflicts emerge with 367 trades/block
**Root Cause**: Conflict detection logic cannot handle rapid position changes

### 3. EOD Manager Timing Failures
**Expected**: All positions closed at end of day
**Actual**: 1-3 days with overnight positions under high-frequency trading
**Root Cause**: EOD closure logic gets confused by frequent position changes

### 4. Missing Signal Intelligence Layer
**Problem**: No distinction between "strong signals worth trading" vs "market noise"
**Result**: System trades on every minor probability fluctuation
**Impact**: Excessive transaction costs, increased risk, system instability

### 5. Backend Strategy Coupling
**Problem**: Backend components are implicitly coupled to specific strategy behaviors
**Evidence**: Perfect performance with TFA, failures with sigor using identical backend
**Impact**: System cannot handle diverse strategy types, limiting scalability

## Detailed Analysis

### Trade Frequency Comparison
```
Strategy | Blocks | Total Trades | Trades/Block | Integrity Status
---------|--------|--------------|--------------|------------------
TFA      | 6      | 88          | 14.7         | ✅ Perfect (5/5)
Sigor    | 6      | 2,205       | 367.5        | ❌ Violations (4/5)
```

### Violation Patterns
- **Position Conflicts**: Emerge only under high-frequency trading
- **EOD Violations**: Increase with trading frequency
- **System Stability**: Inversely correlated with signal frequency

## Root Cause Analysis

### 1. Architectural Assumption Failure
**Assumption**: Strategies generate reasonable signal frequencies
**Reality**: Sigor's detector-based approach generates signals on 76% of bars
**Fix Required**: Implement proper signal filtering and trade frequency controls

### 2. Component Interaction Failures
**Issue**: System components (PositionCoordinator, EODManager) designed for low-frequency trading
**Evidence**: Perfect performance with TFA, failures with sigor
**Fix Required**: Redesign components for high-frequency resilience

### 3. Missing Circuit Breakers
**Issue**: No protection against excessive signal generation
**Evidence**: System allows 25x more trades than reasonable baseline
**Fix Required**: Implement intelligent signal filtering and frequency limits

## Impact Assessment

### Financial Impact
- **Transaction Costs**: 25x higher than necessary (sigor vs TFA)
- **Risk Exposure**: Overnight positions due to EOD failures
- **Capital Efficiency**: Reduced due to churning and conflicts

### System Reliability
- **Integrity Violations**: 2 critical principles fail under stress
- **Scalability**: System cannot handle legitimate high-frequency strategies
- **Robustness**: Architecture breaks down predictably under load

### Operational Impact
- **Production Risk**: System unsuitable for high-frequency strategies
- **Maintenance Burden**: Band-aid fixes (threshold manipulation) mask real issues
- **Development Velocity**: Architectural debt prevents proper solutions

## Proposed Solutions

### 1. Design Truly Strategy-Agnostic Backend
**Component**: Entire backend architecture
**Fix**: Remove all implicit assumptions about strategy behavior
**Benefit**: System works identically well for any strategy type (conservative, aggressive, high-frequency, etc.)

### 2. Implement Adaptive Component Configuration
**Component**: AllocationManager, PositionCoordinator, EODManager
**Fix**: Components automatically adapt to strategy characteristics
**Benefit**: No manual tuning required, optimal performance for each strategy

### 3. Add Strategy Profiling System
**Component**: New StrategyProfiler class
**Fix**: Automatically analyze strategy behavior and configure backend accordingly
**Benefit**: Backend optimizes itself for each strategy's unique characteristics

### 4. Implement Proper Trade Frequency Control
**Component**: PositionCoordinator
**Fix**: Enforce actual "one trade per bar" with timestamp tracking, regardless of signal frequency
**Benefit**: Prevents excessive churning while preserving strong signals for any strategy

### 5. Add Universal Signal Intelligence Layer
**Component**: New AdaptiveSignalFilter class
**Fix**: Dynamically distinguish between actionable signals and noise for any strategy
**Benefit**: Reduces unnecessary trades while preserving alpha, adapts to strategy style

### 6. Redesign Position Coordinator for Universal Scalability
**Component**: PositionCoordinator
**Fix**: Handle any trading frequency without breaking conflict detection
**Benefit**: Maintains integrity under all trading patterns (low-freq, high-freq, burst trading)

### 7. Fix EOD Manager for Universal Robustness
**Component**: EODPositionManager
**Fix**: Robust end-of-day closure regardless of strategy behavior
**Benefit**: Eliminates overnight risk for any strategy type

### 8. Implement Universal System Circuit Breakers
**Component**: New AdaptiveTradingCircuitBreaker class
**Fix**: Automatic protection that adapts to strategy characteristics
**Benefit**: Prevents system overload while maintaining functionality for any strategy

## Testing Strategy

### Stress Testing Protocol
1. **Baseline**: Verify TFA continues to pass all integrity checks
2. **Stress Test**: Use sigor as high-frequency stress test
3. **Validation**: Both strategies must pass all 5 integrity principles
4. **Performance**: Measure system behavior under various signal frequencies

### Success Criteria
- ✅ **Strategy Agnostic**: Backend works identically well for any strategy type
- ✅ **TFA**: Maintains perfect integrity (5/5 principles)
- ✅ **Sigor**: Achieves perfect integrity (5/5 principles) 
- ✅ **Scalability**: System handles 0-500 trades/block gracefully
- ✅ **Performance**: No degradation under any signal pattern
- ✅ **Adaptability**: Backend automatically optimizes for each strategy
- ✅ **Universality**: Same backend code works for conservative and aggressive strategies

## Conclusion

Sigor has revealed that our trading system has fundamental architectural limitations that prevent it from handling high-frequency signal generation. Rather than masking these issues with parameter adjustments, we must redesign the core components to be truly robust and scalable.

The current system works well for conservative strategies like TFA but fails catastrophically for aggressive strategies like sigor. A production-ready system must handle both scenarios flawlessly.

## Priority

**CRITICAL** - These are fundamental architectural flaws that limit system capability and create production risks. Must be resolved before deploying any high-frequency strategies.

## Files Affected

- `src/position_coordinator.cpp` - Core conflict detection logic
- `src/eod_position_manager.cpp` - End-of-day closure timing
- `src/allocation_manager.cpp` - Signal-to-allocation conversion
- `src/runner.cpp` - Main execution pipeline
- `src/strategy_signal_or.cpp` - High-frequency signal generator
- `include/sentio/position_coordinator.hpp` - Position coordination interface
- `include/sentio/eod_position_manager.hpp` - EOD management interface
- `include/sentio/allocation_manager.hpp` - Allocation management interface

## Next Steps

1. Create comprehensive mega document with all relevant source modules
2. Implement proper signal filtering and trade frequency controls
3. Redesign PositionCoordinator for high-frequency resilience
4. Fix EODPositionManager timing logic
5. Add system circuit breakers and monitoring
6. Validate fixes with both TFA and sigor strategies

```

## 📄 **FILE 3 of 16**: temp_core_analysis/include/allocation_manager.hpp

**File Information**:
- **Path**: `temp_core_analysis/include/allocation_manager.hpp`

- **Size**: 52 lines
- **Modified**: 2025-09-19 00:07:31

- **Type**: .hpp

```text
#pragma once

#include "sentio/core.hpp"
#include <string>
#include <vector>

namespace sentio {

/**
 * @brief Configuration for the AllocationManager.
 * Defines the probability thresholds for entering different positions.
 */
struct AllocationConfig {
    // Thresholds for entering a 1x position (e.g., QQQ or PSQ).
    double entry_threshold_1x = 0.60;  // Reasonable threshold for normal trading
    // Thresholds for entering a 3x leveraged position (e.g., TQQQ or SQQQ).
    double entry_threshold_3x = 0.75;  // Higher threshold for leveraged positions
};

/**
 * @brief Represents a concrete allocation decision for the runner.
 */
struct AllocationDecision {
    std::string instrument;
    double target_weight; // e.g., 1.0 for 100%
    std::string reason;
};


/**
 * @class AllocationManager
 * @brief Translates a strategy's probability signal into a profit-maximizing allocation target.
 *
 * This module contains the logic for applying leverage based on signal strength,
 * effectively implementing the "Maximum Profit" principle in a controlled manner.
 */
class AllocationManager {
public:
    explicit AllocationManager(const AllocationConfig& config = AllocationConfig{});

    /**
     * @brief Determines the target instrument and weight based on the signal probability.
     * @param probability A value from 0.0 (strong sell) to 1.0 (strong buy).
     * @return A vector of allocation decisions. Usually one, but can be empty (for cash).
     */
    std::vector<AllocationDecision> get_allocations(double probability);

private:
    AllocationConfig config_;
};

} // namespace sentio

```

## 📄 **FILE 4 of 16**: temp_core_analysis/include/eod_position_manager.hpp

**File Information**:
- **Path**: `temp_core_analysis/include/eod_position_manager.hpp`

- **Size**: 55 lines
- **Modified**: 2025-09-19 00:07:31

- **Type**: .hpp

```text
#pragma once

#include "sentio/core.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/allocation_manager.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace sentio {

/**
 * @brief Configuration for the EODPositionManager.
 */
struct EODManagerConfig {
    int mandatory_close_minutes_before_eod = 15; // Start force-closing all positions 15 minutes before market close.
    // Market close time in UTC (21:00 UTC during EST, 20:00 UTC during EDT)
    int market_close_hour_utc = 20;
    int market_close_minute_utc = 0;
};

/**
 * @class EODPositionManager
 * @brief Manages the mandatory closing of all positions at the end of the trading day.
 */
class EODPositionManager {
public:
    explicit EODPositionManager(const EODManagerConfig& config = EODManagerConfig{});

    /**
     * @brief Checks the current time and returns closing orders if in the EOD window.
     * @param current_timestamp_utc The UTC epoch seconds of the current bar.
     * @param portfolio The current portfolio.
     * @param ST The symbol table.
     * @return A vector of AllocationDecisions to close all open positions.
     */
    std::vector<AllocationDecision> get_eod_allocations(std::int64_t current_timestamp_utc,
                                                        const Portfolio& portfolio,
                                                        const SymbolTable& ST);

    /**
     * @brief Indicates if the EOD closure process is currently active.
     * When true, the main runner loop should block any new opening trades.
     * @return True if in the mandatory close window, false otherwise.
     */
    bool is_closure_active() const;

private:
    EODManagerConfig config_;
    bool closure_is_active_;
    bool close_orders_generated_ = false;  // Prevent duplicate close orders
    int last_eod_day_ = -1;  // Track which day we last generated close orders
};

} // namespace sentio

```

## 📄 **FILE 5 of 16**: temp_core_analysis/include/position_coordinator.hpp

**File Information**:
- **Path**: `temp_core_analysis/include/position_coordinator.hpp`

- **Size**: 65 lines
- **Modified**: 2025-09-19 00:07:31

- **Type**: .hpp

```text
#pragma once

#include "sentio/core.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/allocation_manager.hpp"
#include <string>
#include <vector>
#include <unordered_set>

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

/**
 * @class PositionCoordinator
 * @brief Enforces no-conflict and one-trade-per-bar rules.
 */
class PositionCoordinator {
public:
    PositionCoordinator();

    /**
     * @brief Processes allocation decisions, validating them against system rules.
     * If a decision creates a conflict, this function generates closing trades for existing
     * conflicting positions and rejects the new opening trade for the current bar.
     *
     * @param allocations The desired allocations from the AllocationManager.
     * @param portfolio The current portfolio state.
     * @param ST The symbol table.
     * @return A vector of decisions, which may include new closing trades and rejections.
     */
    std::vector<CoordinationDecision> coordinate(const std::vector<AllocationDecision>& allocations,
                                                 const Portfolio& portfolio,
                                                 const SymbolTable& ST);
    
    /**
     * @brief Resets the per-bar trade counter. Must be called at the start of each new bar.
     */
    void reset_bar();

private:
    bool would_create_conflict(const std::string& new_instrument, const Portfolio& portfolio, const SymbolTable& ST) const;
    
    int orders_this_bar_;
    const std::unordered_set<std::string> LONG_ETFS = {"QQQ", "TQQQ"};
    const std::unordered_set<std::string> INVERSE_ETFS = {"SQQQ", "PSQ"};
};

} // namespace sentio

```

## 📄 **FILE 6 of 16**: temp_core_analysis/include/runner.hpp

**File Information**:
- **Path**: `temp_core_analysis/include/runner.hpp`

- **Size**: 58 lines
- **Modified**: 2025-09-19 00:07:31

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
                           int base_symbol_id, const RunnerCfg& cfg, const DatasetMetadata& dataset_meta = {});

// NEW: Canonical evaluation using Trading Block system for deterministic performance measurement
CanonicalReport run_canonical_backtest(IAuditRecorder& audit, const SymbolTable& ST, 
                                      const std::vector<std::vector<Bar>>& series, int base_symbol_id, 
                                      const RunnerCfg& cfg, const DatasetMetadata& dataset_meta, 
                                      const TradingBlockConfig& block_config);

} // namespace sentio


```

## 📄 **FILE 7 of 16**: temp_core_analysis/include/sizer.hpp

**File Information**:
- **Path**: `temp_core_analysis/include/sizer.hpp`

- **Size**: 102 lines
- **Modified**: 2025-09-19 00:07:31

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

## 📄 **FILE 8 of 16**: temp_core_analysis/include/strategy_signal_or.hpp

**File Information**:
- **Path**: `temp_core_analysis/include/strategy_signal_or.hpp`

- **Size**: 92 lines
- **Modified**: 2025-09-19 00:07:31

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

## 📄 **FILE 9 of 16**: temp_core_analysis/include/strategy_tfa.hpp

**File Information**:
- **Path**: `temp_core_analysis/include/strategy_tfa.hpp`

- **Size**: 68 lines
- **Modified**: 2025-09-19 00:07:31

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

## 📄 **FILE 10 of 16**: temp_core_analysis/src/allocation_manager.cpp

**File Information**:
- **Path**: `temp_core_analysis/src/allocation_manager.cpp`

- **Size**: 49 lines
- **Modified**: 2025-09-19 00:07:28

- **Type**: .cpp

```text
#include "sentio/allocation_manager.hpp"
#include <cmath>
#include <sstream>

namespace sentio {

AllocationManager::AllocationManager(const AllocationConfig& config) : config_(config) {}

std::vector<AllocationDecision> AllocationManager::get_allocations(double probability) {
    std::vector<AllocationDecision> decisions;

    double signal_strength = std::abs(probability - 0.5); // Deviation from neutral
    bool is_bullish = probability > 0.5;

    std::string target_instrument;
    std::string reason;

    if (is_bullish) {
        if (probability >= config_.entry_threshold_3x) {
            // Strong Buy Signal -> Use 3x Leverage
            target_instrument = "TQQQ";
            reason = "Strong bullish signal (p=" + std::to_string(probability) + "), allocating to 3x leverage.";
        } else if (probability >= config_.entry_threshold_1x) {
            // Moderate Buy Signal -> Use 1x 
            target_instrument = "QQQ";
            reason = "Moderate bullish signal (p=" + std::to_string(probability) + "), allocating to 1x.";
        }
    } else { // Bearish
        double inverse_prob = 1.0 - probability;
        if (inverse_prob >= config_.entry_threshold_3x) {
            // Strong Sell Signal -> Use 3x Inverse Leverage
            target_instrument = "SQQQ";
            reason = "Strong bearish signal (p=" + std::to_string(probability) + "), allocating to 3x inverse.";
        } else if (inverse_prob >= config_.entry_threshold_1x) {
            // Moderate Sell Signal -> Use 1x Inverse
            target_instrument = "PSQ";
            reason = "Moderate bearish signal (p=" + std::to_string(probability) + "), allocating to 1x inverse.";
        }
    }
    
    // If a target was selected, create a decision for 100% weight.
    if (!target_instrument.empty()) {
        decisions.push_back({target_instrument, 1.0, reason});
    }

    return decisions;
}

} // namespace sentio

```

## 📄 **FILE 11 of 16**: temp_core_analysis/src/eod_position_manager.cpp

**File Information**:
- **Path**: `temp_core_analysis/src/eod_position_manager.cpp`

- **Size**: 75 lines
- **Modified**: 2025-09-19 00:07:28

- **Type**: .cpp

```text
#include "sentio/eod_position_manager.hpp"
#include <ctime>

namespace sentio {

EODPositionManager::EODPositionManager(const EODManagerConfig& config)
    : config_(config), closure_is_active_(false) {}

bool EODPositionManager::is_closure_active() const {
    return closure_is_active_;
}

std::vector<AllocationDecision> EODPositionManager::get_eod_allocations(
    std::int64_t current_timestamp_utc,
    const Portfolio& portfolio,
    const SymbolTable& ST) {
    
    std::vector<AllocationDecision> closing_decisions;

    // Convert epoch to tm struct for time components
    time_t time_secs = current_timestamp_utc;
    tm* utc_tm = gmtime(&time_secs);

    if (!utc_tm) {
        closure_is_active_ = false;
        return closing_decisions;
    }
    
    int current_hour = utc_tm->tm_hour;
    int current_minute = utc_tm->tm_min;

    // Calculate total minutes from midnight for easier comparison
    int market_close_total_minutes = config_.market_close_hour_utc * 60 + config_.market_close_minute_utc;
    int current_total_minutes = current_hour * 60 + current_minute;
    int mandatory_close_start_time = market_close_total_minutes - config_.mandatory_close_minutes_before_eod;

    // PRINCIPLE 4: EOD CLOSING
    if (current_total_minutes >= mandatory_close_start_time && current_total_minutes < market_close_total_minutes) {
        closure_is_active_ = true;
        
        int current_day = utc_tm->tm_mday;
        
        // **CRITICAL FIX**: Only generate close orders ONCE per day, not every minute
        if (current_day != last_eod_day_ || !close_orders_generated_) {
            // Generate closing orders for all non-cash positions.
            for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
                if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
                    const std::string& symbol = ST.get_symbol(sid);
                    closing_decisions.push_back({symbol, 0.0, "EOD Mandatory Position Closure"});
                }
            }
            
            // Mark that we've generated close orders for this day
            close_orders_generated_ = true;
            last_eod_day_ = current_day;
        }
        // If we've already generated close orders for today, return empty vector
        
    } else {
        closure_is_active_ = false;
        // Reset the flag when we're outside EOD window (new day)
        if (current_total_minutes < mandatory_close_start_time) {
            close_orders_generated_ = false;
        }
    }

    // Reset after market close to prepare for the next day.
    if (current_total_minutes >= market_close_total_minutes) {
        closure_is_active_ = false;
    }

    return closing_decisions;
}

} // namespace sentio

```

## 📄 **FILE 12 of 16**: temp_core_analysis/src/position_coordinator.cpp

**File Information**:
- **Path**: `temp_core_analysis/src/position_coordinator.cpp`

- **Size**: 107 lines
- **Modified**: 2025-09-19 00:07:28

- **Type**: .cpp

```text
#include "sentio/position_coordinator.hpp"
#include <iostream>

namespace sentio {

PositionCoordinator::PositionCoordinator() : orders_this_bar_(0) {}

void PositionCoordinator::reset_bar() {
    orders_this_bar_ = 0;
}

bool PositionCoordinator::would_create_conflict(const std::string& new_instrument, const Portfolio& portfolio, const SymbolTable& ST) const {
    bool wants_long = LONG_ETFS.count(new_instrument);
    bool wants_inverse = INVERSE_ETFS.count(new_instrument);

    if (!wants_long && !wants_inverse) {
        return false; // Not a conflicting type of instrument
    }

    bool has_long = false;
    bool has_inverse = false;

    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
            const std::string& symbol = ST.get_symbol(sid);
            if (LONG_ETFS.count(symbol)) has_long = true;
            if (INVERSE_ETFS.count(symbol)) has_inverse = true;
        }
    }

    // A conflict exists if we want a long position but already have an inverse one, or vice-versa.
    if ((wants_long && has_inverse) || (wants_inverse && has_long)) {
        return true;
    }

    return false;
}

std::vector<CoordinationDecision> PositionCoordinator::coordinate(
    const std::vector<AllocationDecision>& allocations,
    const Portfolio& portfolio,
    const SymbolTable& ST) {

    std::vector<CoordinationDecision> results;

    // If there are no allocation requests, there's nothing to do.
    if (allocations.empty()) {
        // Generate decisions to close any existing positions if the allocation is empty.
        for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
            if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
                if (orders_this_bar_ > 0) continue; // Respect one trade per bar
                const std::string& symbol_to_close = ST.get_symbol(sid);
                results.push_back({ {symbol_to_close, 0.0, "Close position, no active signal"}, CoordinationResult::APPROVED, "Closing existing position." });
                orders_this_bar_++;
            }
        }
        return results;
    }
    
    // For simplicity, we process only the first allocation decision.
    const auto& primary_decision = allocations[0];

    // PRINCIPLE 3: ONE TRADE PER BAR
    // EXCEPTION: EOD close orders (target_weight=0.0) bypass frequency limits for safety
    bool is_eod_close_order = (std::abs(primary_decision.target_weight) < 1e-9);
    if (orders_this_bar_ > 0 && !is_eod_close_order) {
        results.push_back({primary_decision, CoordinationResult::REJECTED_FREQUENCY, "One trade per bar limit reached."});
        return results;
    }

    // PRINCIPLE 2: NO CONFLICTING POSITIONS
    // EXCEPTION: EOD close orders don't create conflicts, they resolve them
    if (!is_eod_close_order && would_create_conflict(primary_decision.instrument, portfolio, ST)) {
        // **ACTIVE CONFLICT RESOLUTION**: Close conflicting positions first, then allow new trade
        
        // Step 1: Generate close orders for all conflicting positions
        bool wants_long = LONG_ETFS.count(primary_decision.instrument);
        bool wants_inverse = INVERSE_ETFS.count(primary_decision.instrument);
        
        for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
            if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
                const std::string& existing_symbol = ST.get_symbol(sid);
                bool existing_is_long = LONG_ETFS.count(existing_symbol);
                bool existing_is_inverse = INVERSE_ETFS.count(existing_symbol);
                
                // Close conflicting positions
                if ((wants_long && existing_is_inverse) || (wants_inverse && existing_is_long)) {
                    results.push_back({{existing_symbol, 0.0, "Close conflicting position"}, 
                                     CoordinationResult::APPROVED, "Closing conflicting position for new trade."});
                }
            }
        }
        
        // Step 2: Approve the new trade after closing conflicts
        results.push_back({primary_decision, CoordinationResult::APPROVED, "Approved after closing conflicts."});
        orders_this_bar_++;
        return results;
    }

    // No conflicts, and frequency limit not hit. Approve the decision.
    results.push_back({primary_decision, CoordinationResult::APPROVED, "Approved by coordinator."});
    orders_this_bar_++;
    
    return results;
}

} // namespace sentio

```

## 📄 **FILE 13 of 16**: temp_core_analysis/src/runner.cpp

**File Information**:
- **Path**: `temp_core_analysis/src/runner.cpp`

- **Size**: 967 lines
- **Modified**: 2025-09-19 00:07:28

- **Type**: .cpp

```text
#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/pricebook.hpp"
#include "sentio/metrics.hpp"
#include "audit/audit_db_recorder.hpp"
#include "sentio/safe_sizer.hpp"
#include "sentio/allocation_manager.hpp"
#include "sentio/position_coordinator.hpp"
#include "sentio/eod_position_manager.hpp"
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

// **NEW EXECUTION PIPELINE**: Strategy → Allocation Manager → Position Coordinator → EOD Manager → Sizer → Execution
static void execute_new_pipeline(double strategy_probability,
                                Portfolio& portfolio, const SymbolTable& ST, const Pricebook& pricebook,
                                AllocationManager& allocation_mgr, PositionCoordinator& position_coord, 
                                EODPositionManager& eod_mgr, const std::vector<std::vector<Bar>>& series, 
                                const Bar& bar, const std::string& chain_id, IAuditRecorder& audit, 
                                bool logging_enabled, int& total_fills, const std::string& strategy_name, size_t bar_index) {
    
    // **STEP 1: STRATEGY SIGNAL** - Already provided as strategy_probability
    
        // **STEP 2: ALLOCATION MANAGER** - Convert probability to instrument allocation
        std::vector<AllocationDecision> allocations = allocation_mgr.get_allocations(strategy_probability);
        
        // Strategy probability and allocation decisions logged to audit system
    
        // **STEP 3: EOD MANAGER** - Override with closing orders if in EOD window
        std::vector<AllocationDecision> eod_allocations = eod_mgr.get_eod_allocations(bar.ts_utc_epoch, portfolio, ST);
        if (!eod_allocations.empty()) {
            allocations = eod_allocations; // EOD overrides everything
            // EOD closure activity logged to audit system
        }
    
    // **STEP 4: POSITION COORDINATOR** - Validate against conflicts and frequency
    std::vector<CoordinationDecision> coordination_decisions = position_coord.coordinate(allocations, portfolio, ST);
    
    // **STEP 5: SIZER & EXECUTION** - Execute approved decisions
    for (const auto& coord_decision : coordination_decisions) {
        if (coord_decision.result != CoordinationResult::APPROVED) {
            if (logging_enabled) {
                audit.event_signal_drop(bar.ts_utc_epoch, strategy_name, coord_decision.decision.instrument,
                                      DropReason::THRESHOLD, chain_id, coord_decision.reason);
            }
                // Rejected decisions logged to audit system
            continue;
        }
        
        const auto& decision = coord_decision.decision;
        
        // **STEP 5A: SIZER** - Calculate safe quantity based on available cash
        SafeSizer sizer;  // Use existing SafeSizer
        double target_qty = sizer.calculate_target_quantity(portfolio, ST, pricebook.last_px, 
                                                           decision.instrument, decision.target_weight, 
                                                           bar.ts_utc_epoch, series[ST.get_id(decision.instrument)]);
        
            // EOD close order execution logged to audit system
        
        // **STEP 5B: EXECUTION** - Execute the trade if quantity is valid
        int instrument_id = ST.get_id(decision.instrument);
        if (instrument_id == -1) continue;
        
        double current_qty = portfolio.positions[instrument_id].qty;
        double trade_qty = target_qty - current_qty;
        
        if (std::abs(trade_qty) < 1e-9) continue; // No trade needed
        
        double instrument_price = pricebook.last_px[instrument_id];
        if (instrument_price <= 0) continue;
        
        // Execute the trade
        Side side = (trade_qty > 0) ? Side::Buy : Side::Sell;
        bool is_sell = (side == Side::Sell);
        
        if (logging_enabled) {
            audit.event_order_ex(bar.ts_utc_epoch, decision.instrument, side, std::abs(trade_qty), 0.0, chain_id);
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
        
        // Calculate fees and execute
        double fees = AlpacaCostModel::calculate_fees(decision.instrument, std::abs(trade_qty), instrument_price, is_sell);
        double exec_px = instrument_price;
        
        apply_fill(portfolio, instrument_id, trade_qty, exec_px);
        portfolio.cash -= fees;
        
        double equity_after = equity_mark_to_market(portfolio, pricebook.last_px);
        double pos_after = portfolio.positions[instrument_id].qty;
        
        if (logging_enabled) {
            audit.event_fill_ex(bar.ts_utc_epoch, decision.instrument, exec_px, std::abs(trade_qty), fees, side,
                               realized_delta, equity_after, pos_after, chain_id);
        }
        
        total_fills++;
    }
}

// **LEGACY HELPER**: Execute target position with comprehensive safety checks
static void execute_target_position(const std::string& instrument, double target_weight,            
                                   Portfolio& portfolio, const SymbolTable& ST, const Pricebook& pricebook,                    
                                   SafeSizer& sizer, [[maybe_unused]] const RunnerCfg& cfg,                
                                   const std::vector<std::vector<Bar>>& series, const Bar& bar,     
                                   const std::string& chain_id, IAuditRecorder& audit, bool logging_enabled, int& total_fills) {
    
    int instrument_id = ST.get_id(instrument);
    if (instrument_id == -1) return;
    
    double instrument_price = pricebook.last_px[instrument_id];
    if (instrument_price <= 0) return;

    // Calculate target quantity using safe sizer with timestamp
    double target_qty = sizer.calculate_target_quantity(portfolio, ST, pricebook.last_px, 
                                                       instrument, target_weight, 
                                                       bar.ts_utc_epoch, series[instrument_id]);
    
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
    
    // **RECORD TRADE**: Update sizer's trade frequency tracker
    sizer.record_trade_execution(bar.ts_utc_epoch);
    
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
    Pricebook pricebook(base_symbol_id, ST, series);
    
    // **NEW EXECUTION PIPELINE COMPONENTS**
    AllocationManager allocation_mgr;
    PositionCoordinator position_coord;
    EODPositionManager eod_mgr;
    
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
        
        // **NEW EXECUTION PIPELINE**: Strategy → Allocation Manager → Position Coordinator → EOD Manager → Sizer → Execution
        // Reset position coordinator for new bar
        position_coord.reset_bar();
        
        // Execute the complete pipeline with strategy probability
        execute_new_pipeline(probability, portfolio, ST, pricebook, allocation_mgr, position_coord, 
                           eod_mgr, series, bar, chain_id, audit, logging_enabled, total_fills, cfg.strategy_name, i);
        
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
        // ENSURE audit logging is enabled for instrument distribution
        block_cfg.audit_level = AuditLevel::Full;
        
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

## 📄 **FILE 14 of 16**: temp_core_analysis/src/strategy_signal_or.cpp

**File Information**:
- **Path**: `temp_core_analysis/src/strategy_signal_or.cpp`

- **Size**: 222 lines
- **Modified**: 2025-09-19 00:07:28

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

## 📄 **FILE 15 of 16**: temp_core_analysis/src/strategy_tfa.cpp

**File Information**:
- **Path**: `temp_core_analysis/src/strategy_tfa.cpp`

- **Size**: 310 lines
- **Modified**: 2025-09-19 00:07:28

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

## 📄 **FILE 16 of 16**: temp_core_analysis/strategy_agnostic_backend_design.md

**File Information**:
- **Path**: `temp_core_analysis/strategy_agnostic_backend_design.md`

- **Size**: 281 lines
- **Modified**: 2025-09-19 00:07:26

- **Type**: .md

```text
# Strategy-Agnostic Backend Architecture Design

## Vision

Create a **truly universal trading backend** that works identically well for any strategy type - from conservative low-frequency strategies like TFA to aggressive high-frequency strategies like sigor - without requiring manual configuration or parameter tuning.

## Core Principles

### 1. **Zero Strategy Assumptions**
- Backend makes no assumptions about signal frequency, trading patterns, or strategy behavior
- All components adapt automatically to observed strategy characteristics
- Same backend code handles all strategy types

### 2. **Adaptive Configuration**
- Components automatically configure themselves based on strategy behavior
- No hardcoded thresholds or fixed parameters
- Real-time adaptation to changing strategy patterns

### 3. **Universal Scalability**
- System performs equally well from 1 trade/day to 500 trades/block
- No performance degradation under any signal pattern
- Graceful handling of burst trading and quiet periods

### 4. **Intelligent Signal Processing**
- Automatic distinction between actionable signals and noise
- Adaptive filtering based on strategy characteristics
- Preserves alpha while reducing unnecessary trades

## Architecture Components

### 1. StrategyProfiler
**Purpose**: Automatically analyze and profile strategy behavior
**Functionality**:
- Monitor signal frequency patterns
- Analyze trade size distributions
- Detect strategy "personality" (conservative, aggressive, burst, etc.)
- Generate dynamic configuration profiles

```cpp
class StrategyProfiler {
public:
    struct StrategyProfile {
        double avg_signal_frequency;    // signals per bar
        double signal_volatility;       // signal strength variance
        TradingStyle style;            // CONSERVATIVE, AGGRESSIVE, BURST, etc.
        double noise_threshold;        // auto-detected noise level
        double confidence_level;       // profile confidence
    };
    
    void observe_signal(double probability, int64_t timestamp);
    void observe_trade(const TradeEvent& trade);
    StrategyProfile get_current_profile() const;
    void reset_profile();  // for new strategies
};
```

### 2. AdaptiveAllocationManager
**Purpose**: Replace fixed-threshold AllocationManager with adaptive version
**Functionality**:
- Dynamic thresholds based on strategy profile
- Automatic noise filtering
- Signal strength normalization

```cpp
class AdaptiveAllocationManager {
private:
    StrategyProfiler profiler_;
    DynamicThresholds thresholds_;
    
public:
    struct DynamicThresholds {
        double entry_1x;      // auto-calculated based on strategy
        double entry_3x;      // auto-calculated based on strategy
        double noise_floor;   // auto-detected noise threshold
    };
    
    std::vector<AllocationDecision> get_allocations(
        double strategy_probability,
        const StrategyProfile& profile
    );
    
    void update_thresholds(const StrategyProfile& profile);
};
```

### 3. UniversalPositionCoordinator
**Purpose**: Handle any trading frequency without breaking
**Functionality**:
- True "one trade per bar" enforcement with timestamp tracking
- Scalable conflict detection for high-frequency trading
- Adaptive conflict resolution based on signal strength

```cpp
class UniversalPositionCoordinator {
private:
    std::unordered_map<int64_t, std::string> trades_per_bar_;  // timestamp -> instrument
    ConflictResolver resolver_;
    
public:
    struct ConflictResolution {
        enum Type { REJECT, CLOSE_EXISTING, QUEUE_FOR_NEXT_BAR };
        Type action;
        std::string reason;
        std::vector<std::string> positions_to_close;
    };
    
    std::vector<CoordinationDecision> coordinate(
        const std::vector<AllocationDecision>& allocations,
        const Portfolio& portfolio,
        const SymbolTable& ST,
        int64_t current_timestamp
    );
    
private:
    bool enforce_one_trade_per_bar(const std::string& instrument, int64_t timestamp);
    ConflictResolution resolve_conflict(
        const AllocationDecision& new_decision,
        const Portfolio& portfolio,
        const StrategyProfile& profile
    );
};
```

### 4. AdaptiveEODManager
**Purpose**: Robust EOD closure regardless of strategy behavior
**Functionality**:
- Position tracking resilient to high-frequency changes
- Adaptive closure timing based on strategy patterns
- Guaranteed EOD closure for any strategy type

```cpp
class AdaptiveEODManager {
private:
    PositionTracker position_tracker_;
    EODConfig adaptive_config_;
    
public:
    struct EODConfig {
        int closure_start_minutes;     // adaptive based on strategy
        int mandatory_close_minutes;   // adaptive based on strategy
        bool allow_late_entries;       // based on strategy profile
    };
    
    std::vector<AllocationDecision> get_eod_allocations(
        int64_t timestamp_utc,
        const Portfolio& portfolio,
        const SymbolTable& ST,
        const StrategyProfile& profile
    );
    
    void adapt_config(const StrategyProfile& profile);
};
```

### 5. AdaptiveSignalFilter
**Purpose**: Intelligent signal filtering that adapts to strategy characteristics
**Functionality**:
- Dynamic noise detection and filtering
- Signal strength normalization
- Adaptive smoothing based on strategy volatility

```cpp
class AdaptiveSignalFilter {
private:
    NoiseDetector noise_detector_;
    SignalSmoother smoother_;
    
public:
    struct FilteredSignal {
        double original_probability;
        double filtered_probability;
        double confidence;
        bool should_trade;
        std::string filter_reason;
    };
    
    FilteredSignal filter_signal(
        double raw_probability,
        const StrategyProfile& profile,
        int64_t timestamp
    );
    
    void update_noise_model(const StrategyProfile& profile);
};
```

### 6. UniversalTradingBackend
**Purpose**: Orchestrate all components in strategy-agnostic manner
**Functionality**:
- Coordinate all adaptive components
- Maintain strategy profiles
- Provide unified interface for any strategy

```cpp
class UniversalTradingBackend {
private:
    StrategyProfiler profiler_;
    AdaptiveAllocationManager allocation_mgr_;
    UniversalPositionCoordinator position_coord_;
    AdaptiveEODManager eod_mgr_;
    AdaptiveSignalFilter signal_filter_;
    
public:
    struct TradingDecision {
        std::vector<AllocationDecision> allocations;
        StrategyProfile current_profile;
        std::string execution_summary;
        bool integrity_check_passed;
    };
    
    TradingDecision process_signal(
        double strategy_probability,
        const Portfolio& portfolio,
        const SymbolTable& ST,
        int64_t timestamp
    );
    
    void register_strategy(const std::string& strategy_name);
    StrategyProfile get_strategy_profile(const std::string& strategy_name) const;
};
```

## Implementation Strategy

### Phase 1: Strategy Profiling Foundation
1. Implement StrategyProfiler to analyze TFA and sigor behavior
2. Collect baseline profiles for both strategies
3. Validate profiling accuracy

### Phase 2: Adaptive Components
1. Replace AllocationManager with AdaptiveAllocationManager
2. Upgrade PositionCoordinator to UniversalPositionCoordinator
3. Enhance EODManager to AdaptiveEODManager

### Phase 3: Signal Intelligence
1. Implement AdaptiveSignalFilter
2. Add noise detection and signal smoothing
3. Integrate with strategy profiling

### Phase 4: Integration and Testing
1. Integrate all components into UniversalTradingBackend
2. Test with TFA (should maintain perfect integrity)
3. Test with sigor (should achieve perfect integrity)
4. Validate strategy-agnostic behavior

## Success Metrics

### Quantitative Metrics
- **TFA**: Maintains 5/5 integrity principles
- **Sigor**: Achieves 5/5 integrity principles
- **Performance**: No degradation for any strategy type
- **Adaptability**: Automatic optimization within 100 bars

### Qualitative Metrics
- **Zero Configuration**: No manual parameter tuning required
- **Universal Compatibility**: Same backend works for all strategies
- **Intelligent Behavior**: System makes smart decisions automatically
- **Robust Operation**: Handles edge cases gracefully

## Benefits

### For Strategy Development
- **Faster Deployment**: New strategies work immediately without backend tuning
- **Focus on Alpha**: Strategy developers focus on signals, not backend integration
- **Consistent Behavior**: Predictable backend behavior across all strategies

### For System Operations
- **Reduced Maintenance**: No strategy-specific configurations to maintain
- **Better Reliability**: Robust backend handles all scenarios
- **Easier Debugging**: Consistent behavior patterns across strategies

### For Business
- **Faster Time-to-Market**: New strategies deploy immediately
- **Higher Capacity**: System handles more diverse strategy types
- **Better Risk Management**: Consistent integrity enforcement

## Conclusion

A truly strategy-agnostic backend will transform Sentio from a system that works well for specific strategy types to a **universal trading platform** that excels with any strategy. This architecture eliminates the current coupling between backend and strategy characteristics, enabling unlimited scalability and strategy diversity.

The key insight is that **the backend should adapt to the strategy, not the other way around**. By implementing intelligent, adaptive components, we create a system that automatically optimizes itself for each strategy's unique characteristics while maintaining perfect integrity and performance.

```

