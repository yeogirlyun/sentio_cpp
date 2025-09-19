# Requirement: Exact Metric Alignment Between Strattest and Audit Systems

**Document Version:** 1.0  
**Date:** 2024-01-15  
**Status:** Architectural Challenge Identified  
**Priority:** HIGH  

## 1. Executive Summary

This document specifies the requirement for **exact metric alignment** between the `strattest` and `audit` systems in the Sentio C++ trading platform. Currently, while core metrics (MPR and Sharpe Ratio) are now within 0.02% and 0.004% respectively, **exact matches** are not achievable due to fundamental architectural differences in how trading days are calculated and counted.

## 2. Current Status

### 2.1 Achieved Improvements
- **Monthly Projected Return (MPR)**: Reduced discrepancy from +3.41% to +0.02%
- **Sharpe Ratio**: Reduced discrepancy from +12.99 to +0.004
- **Daily Returns Storage**: Successfully implemented in audit database
- **Core Metrics Pipeline**: Unified calculation methods implemented

### 2.2 Remaining Discrepancies
| Metric | strattest | audit | Discrepancy | Root Cause |
|--------|-----------|-------|-------------|------------|
| **Trading Days** | 28 | 31 | +3 days | Different calculation methods |
| **Daily Trades** | 182.3 | 164.7 | -17.6 trades | Different denominators |

## 3. Functional Requirements

### 3.1 Primary Requirement
**REQ-001: Exact Metric Alignment**
- **Description**: All core performance metrics (MPR, Sharpe Ratio, Daily Trades, Trading Days) must produce **identical values** for the same run ID across `strattest`, `audit summarize`, and `audit position-history`
- **Acceptance Criteria**: 
  - MPR difference ≤ 0.001%
  - Sharpe Ratio difference ≤ 0.001
  - Daily Trades difference ≤ 0.1 trades
  - Trading Days difference = 0 days
- **Priority**: HIGH

### 3.2 Secondary Requirements
**REQ-002: Dataset Traceability**
- **Description**: Both systems must clearly display the exact dataset and test period used for calculations
- **Acceptance Criteria**: Reports must show dataset file, track ID, time range, and bar count

**REQ-003: Calculation Transparency**
- **Description**: Both systems must use identical calculation methods and data sources
- **Acceptance Criteria**: No fallback calculations or different methodologies

## 4. Technical Requirements

### 4.1 Trading Days Calculation Unification
**REQ-004: Single Source of Truth for Trading Days**
- **Description**: Both systems must use the same method to count trading days
- **Current Issue**: 
  - `strattest` uses `count_trading_days()` with timezone conversion
  - `audit` uses stored daily returns count
- **Solution**: Unify on a single calculation method

### 4.2 Data Source Alignment
**REQ-005: Identical Data Sources**
- **Description**: Both systems must process the exact same dataset
- **Current Issue**: Different equity curve sampling frequencies
- **Solution**: Ensure identical snapshot frequencies and data processing

### 4.3 Metric Calculation Pipeline
**REQ-006: Unified Calculation Pipeline**
- **Description**: Both systems must use identical calculation logic
- **Current Issue**: Different equity curve compression methods
- **Solution**: Share calculation functions between systems

## 5. Architectural Challenges

### 5.1 Fundamental Design Differences

#### 5.1.1 Trading Days Calculation Methods
- **strattest**: Uses `count_trading_days()` with timezone-aware date extraction
- **audit**: Uses count of stored daily returns records
- **Challenge**: Different timezone handling and date boundary detection

#### 5.1.2 Data Sampling Frequencies
- **strattest**: Uses `snapshot_stride` parameter (default: 100 bars)
- **audit**: Uses stored daily returns from equity curve compression
- **Challenge**: Different sampling frequencies lead to different day counts

#### 5.1.3 Equity Curve Processing
- **strattest**: Processes equity curve in `UnifiedMetricsCalculator`
- **audit**: Processes stored daily returns in `DB::summarize`
- **Challenge**: Different compression and aggregation methods

### 5.2 Root Cause Analysis

The inability to achieve exact matches is **not a bug but an architectural limitation**:

1. **Dual Calculation Pipelines**: Two independent systems calculating metrics from different data representations
2. **Different Data Models**: Equity curve vs. daily returns vs. raw audit events
3. **Sampling Frequency Mismatch**: Different snapshot frequencies create different day counts
4. **Timezone Handling**: Complex timezone conversions vs. simple date extraction

## 6. Proposed Solutions

### 6.1 Option A: Unified Calculation Engine (Recommended)
- **Approach**: Create a single `CanonicalMetricsCalculator` used by both systems
- **Benefits**: Guaranteed identical results
- **Effort**: Medium - requires refactoring both systems
- **Risk**: Low - maintains existing interfaces

### 6.2 Option B: Shared Data Pipeline
- **Approach**: Both systems use identical data processing pipeline
- **Benefits**: Consistent data representation
- **Effort**: High - requires major architectural changes
- **Risk**: Medium - could break existing functionality

### 6.3 Option C: Audit System Enhancement
- **Approach**: Modify audit system to use strattest calculation methods
- **Benefits**: Minimal changes to strattest
- **Effort**: Low - only audit system changes
- **Risk**: Low - audit system is less critical

## 7. Implementation Plan

### 7.1 Phase 1: Immediate Fixes (Completed)
- ✅ Implement daily returns storage in audit database
- ✅ Fix timestamp conversion issues
- ✅ Ensure audit logging is enabled

### 7.2 Phase 2: Architectural Unification (Recommended)
1. **Create CanonicalMetricsCalculator**
   - Extract common calculation logic
   - Implement unified trading days counting
   - Standardize equity curve processing

2. **Modify Both Systems**
   - Update strattest to use canonical calculator
   - Update audit system to use canonical calculator
   - Remove duplicate calculation code

3. **Validation and Testing**
   - Create comprehensive test suite
   - Validate exact matches across all metrics
   - Performance regression testing

### 7.3 Phase 3: Enhanced Reporting
1. **Dataset Information Display**
   - Add dataset details to strattest reports
   - Add dataset details to audit reports
   - Include calculation method information

2. **Transparency Improvements**
   - Show calculation steps in reports
   - Include data source information
   - Add validation checksums

## 8. Success Criteria

### 8.1 Functional Success
- **Exact Metric Matches**: All core metrics identical across systems
- **Dataset Traceability**: Clear dataset information in all reports
- **Calculation Transparency**: Identical calculation methods documented

### 8.2 Technical Success
- **Single Source of Truth**: One calculation engine for all metrics
- **Maintainability**: Reduced code duplication
- **Performance**: No degradation in calculation speed

### 8.3 Quality Success
- **Test Coverage**: Comprehensive test suite for metric calculations
- **Documentation**: Clear documentation of calculation methods
- **Validation**: Automated checks for metric consistency

## 9. Risk Assessment

### 9.1 Technical Risks
- **Calculation Changes**: Modifying core calculation logic could introduce bugs
- **Performance Impact**: Unified calculation engine might be slower
- **Data Consistency**: Changes could affect historical data interpretation

### 9.2 Mitigation Strategies
- **Comprehensive Testing**: Extensive test suite before deployment
- **Gradual Rollout**: Phase implementation with validation at each step
- **Backup Systems**: Maintain ability to revert to previous calculations

## 10. Conclusion

While significant progress has been made in aligning the core metrics between strattest and audit systems, **exact matches require architectural unification**. The current discrepancies are not bugs but fundamental design differences that can only be resolved through:

1. **Unified calculation engine** for both systems
2. **Identical data processing pipelines**
3. **Single source of truth** for trading days calculation

The recommended approach is **Option A: Unified Calculation Engine** as it provides the best balance of effort, risk, and maintainability while guaranteeing exact metric alignment.

---

**Document Control:**
- **Author**: AI Assistant
- **Reviewers**: Development Team
- **Approval**: Pending
- **Next Review**: After architectural decision
