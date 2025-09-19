# STRATTEST vs AUDIT SUMMARIZE DISCREPANCY BUG REPORT

## Executive Summary

This report documents significant discrepancies between `sentio_cli strattest` results and `sentio_audit summarize` results for the same strategy runs. The analysis reveals multiple inconsistencies in metrics calculation, data interpretation, and reporting formats that suggest fundamental architectural issues in the dual-system approach.

## Test Methodology

**Test Configuration:**
- Duration: 4 weeks (28 trading days)
- Simulations: 1 (for precise comparison)
- Data Source: Historical
- Strategies Tested: TFA, OFI
- Test Date: 2025-09-15

## Detailed Comparison Analysis

### TFA Strategy Results

#### STRATTEST Results:
```
📈 RETURN STATISTICS
Mean Return:                  0.04%
Mean Sharpe Ratio:            0.31
Mean MPR (Monthly):           0.02%
Mean Daily Trades:             5.0

📈 PERFORMANCE SUMMARY
Monthly Projected Return:      0.0% ±  0.0%  [  0.0% -   0.0%] (95% CI)
Sharpe Ratio:                0.31 ± 0.00  [ 0.31 -  0.31] (95% CI)
Maximum Drawdown:             0.5% ±  0.0%  [  0.5% -   0.5%] (95% CI)
```

#### AUDIT SUMMARIZE Results:
```
Trading Performance:
  Total Return: 0.17%
  Monthly Projected Return (MPR): 0.02%
  Sharpe Ratio: 2.317
  Daily Trades: 5.6
  Max Drawdown: 0.03%
  Trading Days: 31

P&L Summary:
  Total P&L: 166.420682 (sum of all P&L from FILL events)
```

### OFI Strategy Results

#### STRATTEST Results:
```
📈 RETURN STATISTICS
Mean Return:                 -2.43%
Mean Sharpe Ratio:          -11.90
Mean MPR (Monthly):          -1.65%
Mean Daily Trades:            22.0

📈 PERFORMANCE SUMMARY
Monthly Projected Return:     -1.7% ±  0.0%  [ -1.7% -  -1.7%] (95% CI)
Sharpe Ratio:              -11.90 ± 0.00  [-11.90 - -11.90] (95% CI)
Maximum Drawdown:             2.4% ±  0.0%  [  2.4% -   2.4%] (95% CI)
```

#### AUDIT SUMMARIZE Results:
```
Trading Performance:
  Total Return: -2.26%
  Monthly Projected Return (MPR): -1.65%
  Sharpe Ratio: -8.547
  Daily Trades: 22.8
  Max Drawdown: 0.45%
  Trading Days: 31

P&L Summary:
  Total P&L: -2264.700766 (sum of all P&L from FILL events)
```

## Critical Discrepancies Identified

### 1. Sharpe Ratio Calculation Discrepancy

**TFA Strategy:**
- STRATTEST: 0.31
- AUDIT: 2.317
- **Discrepancy: 647% difference**

**OFI Strategy:**
- STRATTEST: -11.90
- AUDIT: -8.547
- **Discrepancy: 28% difference**

**Root Cause Analysis:**
- Different volatility calculations
- Different risk-free rate assumptions
- Different time period calculations
- Potential different return series used for calculation

### 2. Total Return vs Mean Return Discrepancy

**TFA Strategy:**
- STRATTEST Mean Return: 0.04%
- AUDIT Total Return: 0.17%
- **Discrepancy: 325% difference**

**OFI Strategy:**
- STRATTEST Mean Return: -2.43%
- AUDIT Total Return: -2.26%
- **Discrepancy: 7.5% difference**

**Root Cause Analysis:**
- Different return calculation methodologies
- Compounding vs simple return calculations
- Different time period definitions

### 3. Maximum Drawdown Discrepancy

**TFA Strategy:**
- STRATTEST: 0.5%
- AUDIT: 0.03%
- **Discrepancy: 1567% difference**

**OFI Strategy:**
- STRATTEST: 2.4%
- AUDIT: 0.45%
- **Discrepancy: 433% difference**

**Root Cause Analysis:**
- Different drawdown calculation algorithms
- Different peak/trough identification methods
- Different time window definitions

### 4. Daily Trades Discrepancy

**TFA Strategy:**
- STRATTEST: 5.0 trades/day
- AUDIT: 5.6 trades/day
- **Discrepancy: 12% difference**

**OFI Strategy:**
- STRATTEST: 22.0 trades/day
- AUDIT: 22.8 trades/day
- **Discrepancy: 3.6% difference**

**Root Cause Analysis:**
- Different trade counting methodologies
- Different time period calculations
- Different trade classification criteria

### 5. Trading Days Discrepancy

**Both Strategies:**
- STRATTEST: Reports "30 days" simulation period
- AUDIT: Reports "31 trading days" and "28 trading days" test period
- **Inconsistency: Multiple conflicting day counts**

**Root Cause Analysis:**
- Different calendar calculations
- Different session definitions
- Different data range interpretations

## Architectural Issues Identified

### 1. Dual-System Architecture Problem

The system maintains two separate calculation engines:
- `sentio_cli strattest`: Virtual market simulation engine
- `sentio_audit`: Audit trail analysis engine

**Problems:**
- No synchronization between systems
- Different calculation methodologies
- Different data interpretations
- No validation mechanism

### 2. Data Source Inconsistency

**STRATTEST:**
- Uses virtual market simulation
- Generates synthetic market data
- Uses future QQQ tracks for AI regime testing

**AUDIT:**
- Uses actual historical data
- Records real fill events
- Uses different time periods

### 3. Metric Calculation Divergence

**Sharpe Ratio:**
- Different volatility calculations
- Different risk-free rate assumptions
- Different return series definitions

**Return Calculations:**
- Different compounding methods
- Different time period definitions
- Different baseline calculations

**Drawdown Calculations:**
- Different peak identification algorithms
- Different trough detection methods
- Different time window definitions

### 4. Time Period Confusion

**Multiple Conflicting Reports:**
- "30 days" simulation period
- "31 trading days" 
- "28 trading days" test period
- Different start/end timestamps

## Impact Assessment

### 1. Trust and Reliability Issues

- **High Impact**: Users cannot trust either system's results
- **Medium Impact**: Decision-making based on unreliable metrics
- **High Impact**: Potential financial losses from incorrect strategy evaluation

### 2. System Integration Problems

- **High Impact**: Dual-system architecture creates maintenance burden
- **Medium Impact**: No single source of truth for performance metrics
- **High Impact**: Difficult to debug and validate results

### 3. User Experience Degradation

- **Medium Impact**: Confusing and contradictory reports
- **High Impact**: Loss of confidence in system accuracy
- **Medium Impact**: Increased support burden

## Root Cause Analysis

### 1. Historical Architecture Debt

The system evolved from separate components:
- Original audit system for recording trades
- New virtual market system for strategy testing
- No integration or synchronization between systems

### 2. Different Calculation Philosophies

**STRATTEST Philosophy:**
- Monte Carlo simulation approach
- Statistical sampling methodology
- Virtual market data generation

**AUDIT Philosophy:**
- Event-driven recording approach
- Historical data analysis methodology
- Real market data processing

### 3. Lack of Validation Framework

- No cross-validation between systems
- No reconciliation mechanism
- No error detection or correction

### 4. Time Period Calculation Errors

- Different calendar systems
- Different session definitions
- Different data range interpretations
- Inconsistent timestamp handling

## Recommendations

### 1. Immediate Actions (High Priority)

1. **Implement Cross-Validation**
   - Add reconciliation checks between strattest and audit
   - Flag discrepancies above threshold (e.g., 5%)
   - Generate warning reports for users

2. **Standardize Metric Calculations**
   - Define canonical calculation methods
   - Implement shared calculation libraries
   - Ensure consistent formulas across systems

3. **Fix Time Period Calculations**
   - Standardize calendar definitions
   - Implement consistent session calculations
   - Validate time period consistency

### 2. Medium-Term Solutions (Medium Priority)

1. **Unified Calculation Engine**
   - Merge calculation logic into single engine
   - Use audit system as primary data source
   - Make strattest use audit data for reporting

2. **Enhanced Validation Framework**
   - Implement automated testing for metric consistency
   - Add regression testing for calculation changes
   - Create validation reports for each release

3. **Improved Documentation**
   - Document calculation methodologies
   - Explain differences between systems
   - Provide user guidance on interpretation

### 3. Long-Term Architecture (Low Priority)

1. **Single-Source-of-Truth Architecture**
   - Eliminate dual-system approach
   - Implement unified performance calculation
   - Use single audit trail for all reporting

2. **Real-Time Validation**
   - Implement continuous validation during runs
   - Add real-time discrepancy detection
   - Provide immediate feedback on inconsistencies

## Conclusion

The discrepancies between `sentio_cli strattest` and `sentio_audit summarize` results represent a critical architectural issue that undermines the reliability and trustworthiness of the entire system. The dual-system approach has created fundamental inconsistencies in metric calculations, time period definitions, and data interpretation.

**Key Findings:**
- Sharpe ratio discrepancies up to 647%
- Total return discrepancies up to 325%
- Maximum drawdown discrepancies up to 1567%
- Conflicting time period reports
- No validation or reconciliation mechanism

**Critical Impact:**
- Users cannot trust either system's results
- Decision-making based on unreliable metrics
- Potential financial losses from incorrect evaluations
- System maintenance complexity

**Immediate Action Required:**
- Implement cross-validation between systems
- Standardize metric calculation methodologies
- Fix time period calculation inconsistencies
- Add discrepancy detection and warning mechanisms

This bug report should be treated as a **CRITICAL PRIORITY** requiring immediate attention to restore system reliability and user trust.

---

**Report Generated:** 2025-09-15  
**Test Data:** TFA and OFI strategies, 4-week duration, 1 simulation  
**Discrepancy Threshold:** >5% difference flagged as significant  
**Status:** CRITICAL - Requires immediate resolution
