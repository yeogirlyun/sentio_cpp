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
