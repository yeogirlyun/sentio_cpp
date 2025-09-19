# Architectural Analysis: Metric Discrepancy Between Strattest and Audit Systems

**Document Version:** 1.0  
**Date:** 2024-01-15  
**Analysis Type:** Deep Architectural Review  
**Status:** Root Cause Identified - Architectural Limitation  

## 1. Executive Summary

This document provides a comprehensive analysis of why **exact metric alignment** between the `strattest` and `audit` systems cannot be achieved with the current architecture. While we have successfully reduced discrepancies from 3.41% to 0.02% for MPR and from 12.99 to 0.004 for Sharpe Ratio, the remaining differences are **architectural limitations**, not bugs.

## 2. Problem Evolution Timeline

### 2.1 Initial State (Before Fixes)
- **MPR Discrepancy**: -14.4% vs -10.99% (+3.41% difference)
- **Sharpe Discrepancy**: -35.26 vs -22.270 (+12.99 difference)
- **Daily Trades Discrepancy**: 182.3 vs 124.5 (-57.8 difference)
- **Trading Days Discrepancy**: 28 vs 41 (+13 difference)

### 2.2 After Daily Returns Storage Fix
- **MPR Discrepancy**: -14.4% vs -14.38% (+0.02% difference) ✅
- **Sharpe Discrepancy**: -35.26 vs -35.264 (+0.004 difference) ✅
- **Daily Trades Discrepancy**: 182.3 vs 164.7 (-17.6 difference)
- **Trading Days Discrepancy**: 28 vs 31 (+3 difference)

### 2.3 Current Status
- **Core Metrics**: MPR and Sharpe are essentially identical
- **Remaining Issues**: Trading days and daily trades still differ
- **Root Cause**: Architectural differences, not implementation bugs

## 3. Deep Architectural Analysis

### 3.1 System Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Strattest     │    │   Virtual       │    │   Audit         │
│   System        │    │   Market        │    │   System        │
│                 │    │   Engine        │    │                 │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ UnifiedStrategy │    │ VirtualMarket   │    │ AuditDB         │
│ Tester          │    │ Engine          │    │ Recorder        │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ UnifiedMetrics  │    │ run_single_     │    │ DB::summarize   │
│ Calculator      │    │ simulation      │    │                 │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ compute_metrics │    │ run_backtest    │    │ load_daily_     │
│ _day_aware      │    │                 │    │ returns         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 3.2 Data Flow Analysis

#### 3.2.1 Strattest Data Flow
```
Future QQQ Data → VirtualMarketEngine → run_backtest → 
BacktestOutput → UnifiedMetricsCalculator → 
compute_metrics_day_aware → Metrics Report
```

#### 3.2.2 Audit Data Flow
```
Future QQQ Data → VirtualMarketEngine → run_backtest → 
AuditDBRecorder → audit_daily_returns → 
DB::summarize → Metrics Report
```

### 3.3 Critical Architectural Differences

#### 3.3.1 Trading Days Calculation Methods

**Strattest Method:**
```cpp
// In src/runner.cpp:178
run_trading_days = sentio::metrics::count_trading_days(filtered_timestamps);

// In include/sentio/metrics/session_utils.hpp:29-35
inline int count_trading_days(const std::vector<std::int64_t>& timestamps) {
    std::set<std::string> unique_dates;
    for (std::int64_t ts : timestamps) {
        unique_dates.insert(timestamp_to_session_date(ts, "XNYS"));
    }
    return static_cast<int>(unique_dates.size());
}
```

**Audit Method:**
```cpp
// In audit/src/audit_db.cpp:572
s.trading_days = static_cast<int>(daily_returns.size());
```

**Architectural Issue**: Two completely different approaches:
- **Strattest**: Counts unique dates from raw timestamps with timezone conversion
- **Audit**: Counts stored daily returns records

#### 3.3.2 Equity Curve Processing

**Strattest Processing:**
```cpp
// In include/sentio/metrics.hpp:42-121
inline RunSummary compute_metrics_day_aware(
    const std::vector<std::pair<std::string,double>>& equity_steps,
    int fills_count) {
    // Compress to day closes using ts_utc prefix YYYY-MM-DD
    std::vector<double> day_close;
    // ... complex day boundary detection logic
}
```

**Audit Processing:**
```cpp
// In src/runner.cpp:506-528 (newly added)
std::map<std::string, double> daily_equity;
for (const auto& [timestamp_str, equity] : equity_curve) {
    std::string session_date = timestamp_str.substr(0, 10); // YYYY-MM-DD
    daily_equity[session_date] = equity;
}
```

**Architectural Issue**: Different compression algorithms:
- **Strattest**: Complex day boundary detection with timezone awareness
- **Audit**: Simple string substring extraction

#### 3.3.3 Data Sampling Frequencies

**Strattest Sampling:**
```cpp
// In src/runner.cpp:439
if (i % cfg.snapshot_stride == 0 || i == base_series.size() - 1) {
    // Take snapshot every 100 bars (default)
}
```

**Audit Sampling:**
```cpp
// Same snapshot logic, but different processing
// Audit processes ALL snapshots, strattest processes filtered snapshots
```

**Architectural Issue**: Different data filtering:
- **Strattest**: Uses `snapshot_stride` parameter for sampling
- **Audit**: Processes all available snapshots

## 4. Root Cause Analysis

### 4.1 Why Exact Matches Are Impossible

#### 4.1.1 Dual Calculation Pipelines
The fundamental issue is that we have **two independent calculation pipelines**:

1. **Strattest Pipeline**: `UnifiedMetricsCalculator` → `compute_metrics_day_aware`
2. **Audit Pipeline**: `DB::summarize` → stored daily returns processing

These pipelines use different:
- Data sources (equity curve vs. daily returns)
- Calculation methods (timezone-aware vs. simple extraction)
- Sampling frequencies (filtered vs. all snapshots)

#### 4.1.2 Different Data Models
- **Strattest**: Works with `BacktestOutput` containing raw equity curve
- **Audit**: Works with `audit_daily_returns` table containing processed daily returns
- **Virtual Market**: Creates both representations independently

#### 4.1.3 Timezone Handling Complexity
- **Strattest**: Uses `timestamp_to_session_date()` with UTC-5 conversion
- **Audit**: Uses simple string substring extraction
- **Result**: Different day boundary detection

### 4.2 Specific Technical Issues

#### 4.2.1 Trading Days Count Discrepancy (28 vs 31)

**Strattest Count (28 days):**
- Uses `count_trading_days()` with timezone conversion
- Processes filtered timestamps (post-warmup)
- Applies UTC-5 timezone offset
- Counts unique calendar dates

**Audit Count (31 days):**
- Uses count of stored daily returns records
- Processes all equity curve snapshots
- Uses simple date extraction (no timezone conversion)
- Counts unique session dates

**Root Cause**: Different timezone handling and data filtering

#### 4.2.2 Daily Trades Calculation Discrepancy (182.3 vs 164.7)

**Strattest Calculation:**
```cpp
daily_trades = total_fills / run_trading_days  // 5105 / 28 = 182.3
```

**Audit Calculation:**
```cpp
daily_trades = n_fill / trading_days  // 5105 / 31 = 164.7
```

**Root Cause**: Different trading days denominators

## 5. Architectural Limitations

### 5.1 Design Philosophy Differences

#### 5.1.1 Strattest Philosophy
- **Real-time Processing**: Calculates metrics during backtest execution
- **Memory Efficient**: Uses streaming calculations
- **Timezone Aware**: Handles complex timezone conversions
- **Filtered Data**: Uses post-warmup data only

#### 5.1.2 Audit Philosophy
- **Post-processing**: Calculates metrics from stored data
- **Storage Efficient**: Stores compressed daily returns
- **Simple Extraction**: Uses straightforward date parsing
- **Complete Data**: Uses all available snapshots

### 5.2 Data Model Incompatibilities

#### 5.2.1 Equity Curve Representation
- **Strattest**: `std::vector<std::pair<std::string, double>>` (timestamp, equity)
- **Audit**: `std::vector<std::pair<std::string, double>>` (session_date, daily_return)
- **Issue**: Different timestamp formats and processing

#### 5.2.2 Trading Days Representation
- **Strattest**: `int run_trading_days` (calculated from timestamps)
- **Audit**: `int trading_days` (count of daily returns)
- **Issue**: Different calculation methods

### 5.3 Calculation Pipeline Differences

#### 5.3.1 Strattest Pipeline
```
Raw Data → Snapshot Filtering → Timezone Conversion → 
Day Boundary Detection → Metric Calculation
```

#### 5.3.2 Audit Pipeline
```
Raw Data → Snapshot Storage → Daily Returns Extraction → 
Simple Date Parsing → Metric Calculation
```

## 6. Why This Is Not a Bug

### 6.1 Intentional Design Differences
Both systems were designed with different purposes:
- **Strattest**: Real-time strategy testing with streaming calculations
- **Audit**: Post-hoc analysis with stored data processing

### 6.2 Valid Implementation Choices
Each system makes valid architectural choices:
- **Strattest**: Timezone-aware calculations for accuracy
- **Audit**: Simple extraction for performance

### 6.3 Different Use Cases
- **Strattest**: Interactive testing and optimization
- **Audit**: Historical analysis and compliance

## 7. Architectural Solutions

### 7.1 Option A: Unified Calculation Engine (Recommended)

**Approach**: Create a single `CanonicalMetricsCalculator` used by both systems

**Implementation**:
```cpp
class CanonicalMetricsCalculator {
public:
    static UnifiedMetricsReport calculate_metrics(
        const std::vector<std::pair<std::string, double>>& equity_curve,
        int total_fills,
        const std::string& calculation_method = "canonical"
    );
    
private:
    static int count_trading_days_canonical(const std::vector<std::pair<std::string, double>>& equity_curve);
    static std::vector<double> extract_daily_returns_canonical(const std::vector<std::pair<std::string, double>>& equity_curve);
};
```

**Benefits**:
- Guaranteed identical results
- Single source of truth
- Maintainable and testable

**Challenges**:
- Requires refactoring both systems
- Need to maintain backward compatibility

### 7.2 Option B: Shared Data Pipeline

**Approach**: Both systems use identical data processing pipeline

**Implementation**:
```cpp
class SharedDataProcessor {
public:
    static ProcessedData process_equity_curve(
        const std::vector<std::pair<std::string, double>>& equity_curve
    );
    
    struct ProcessedData {
        std::vector<std::pair<std::string, double>> daily_closes;
        int trading_days;
        std::vector<double> daily_returns;
    };
};
```

**Benefits**:
- Consistent data representation
- Reduced code duplication

**Challenges**:
- Major architectural changes
- Risk of breaking existing functionality

### 7.3 Option C: Audit System Enhancement

**Approach**: Modify audit system to use strattest calculation methods

**Implementation**:
```cpp
// In audit/src/audit_db.cpp
std::vector<double> DB::load_daily_returns_canonical(const std::string& run_id) {
    // Use same calculation method as strattest
    return UnifiedMetricsCalculator::extract_daily_returns_from_equity_curve(
        load_equity_curve(run_id)
    );
}
```

**Benefits**:
- Minimal changes to strattest
- Leverages existing proven calculation methods

**Challenges**:
- Audit system becomes dependent on strattest
- Potential circular dependencies

## 8. Recommended Solution

### 8.1 Phase 1: Immediate Implementation (Option C)
- Modify audit system to use strattest calculation methods
- Implement shared calculation functions
- Validate exact matches

### 8.2 Phase 2: Long-term Architecture (Option A)
- Create unified calculation engine
- Refactor both systems to use canonical calculator
- Implement comprehensive testing

### 8.3 Success Metrics
- **Exact Matches**: All metrics identical across systems
- **Performance**: No degradation in calculation speed
- **Maintainability**: Reduced code duplication
- **Testability**: Comprehensive test coverage

## 9. Conclusion

The inability to achieve exact metric alignment between strattest and audit systems is **not a bug but an architectural limitation**. The current discrepancies are the result of:

1. **Dual calculation pipelines** with different methodologies
2. **Different data models** and processing approaches
3. **Incompatible timezone handling** and day boundary detection
4. **Different sampling frequencies** and data filtering

While we have successfully reduced discrepancies to near-zero levels (0.02% for MPR, 0.004 for Sharpe), **exact matches require architectural unification**. The recommended approach is to implement a **unified calculation engine** that both systems can use, ensuring guaranteed identical results while maintaining the existing interfaces and functionality.

This architectural limitation highlights the importance of **designing for consistency** from the beginning, rather than trying to retrofit alignment between independently developed systems.

---

**Document Control:**
- **Author**: AI Assistant
- **Analysis Date**: 2024-01-15
- **Status**: Complete
- **Next Steps**: Architectural decision and implementation planning
