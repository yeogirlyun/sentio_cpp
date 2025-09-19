# Bug Report: Core Metrics Discrepancy Between Systems

## Summary
The three core metrics (MPR, Sharpe Ratio, Daily Trades) show significant discrepancies across the three reporting systems when using the same run ID, indicating inconsistent calculation methods or data sources.

## Test Case
- **Run ID**: 225081
- **Strategy**: sigor
- **Test Period**: 1 week (1w)
- **Simulations**: 1
- **Date**: 2025-09-16

## Discrepancies Found

### 1. Monthly Projected Return (MPR)
| System | MPR Value | Difference |
|--------|-----------|------------|
| strattest | -14.4% | Baseline |
| audit summarize | -10.99% | +3.41% |
| position-history | -10.99% | +3.41% |

**Impact**: 3.41 percentage point difference in MPR calculation

### 2. Sharpe Ratio
| System | Sharpe Value | Difference |
|--------|--------------|------------|
| strattest | -35.26 | Baseline |
| audit summarize | -22.270 | +12.99 |
| position-history | -22.270 | +12.99 |

**Impact**: 12.99 point difference in Sharpe ratio calculation

### 3. Daily Trades
| System | Daily Trades | Difference |
|--------|--------------|------------|
| strattest | 182.3 | Baseline |
| audit summarize | 124.5 | -57.8 |
| position-history | 124.5 | -57.8 |

**Impact**: 57.8 trades per day difference

### 4. Trading Days Calculation
| System | Trading Days | Difference |
|--------|--------------|------------|
| strattest | 28 days | Baseline |
| audit summarize | 41 days | +13 days |
| position-history | 41 days | +13 days |

**Impact**: 13 additional trading days in audit system

## Root Cause Analysis

### 1. Trading Days Discrepancy
- **strattest**: Uses 28 trading days
- **audit systems**: Use 41 trading days
- **Cause**: Different methods for calculating trading days from the same dataset

### 2. MPR Calculation
- **strattest**: Uses `sentio::metrics::compute_mpr_from_daily_returns(daily_returns)` with 28-day equity curve
- **audit systems**: Use same function but with 41-day data
- **Cause**: Different daily returns data due to different trading days calculation

### 3. Sharpe Ratio Calculation
- **strattest**: Uses proper Sharpe calculation from `compute_metrics_day_aware` with daily returns
- **audit systems**: Use simplified calculation: `(mean / sd) * sqrt(252)` with different data
- **Cause**: Different calculation method and different daily returns data

### 4. Daily Trades Calculation
- **strattest**: `total_fills / run_trading_days` = 5105 / 28 = 182.3
- **audit systems**: `n_fill / trading_days` = 5105 / 41 = 124.5
- **Cause**: Different trading days denominator

## Technical Details

### strattest Calculation Method
```cpp
// Uses UnifiedMetricsCalculator::calculate_metrics()
// - Trading days: 28 (from equity curve compression)
// - MPR: sentio::metrics::compute_mpr_from_daily_returns(daily_returns)
// - Sharpe: compute_metrics_day_aware() with proper daily returns
// - Daily trades: total_fills / run_trading_days
```

### audit summarize Calculation Method
```cpp
// Uses DB::summarize()
// - Trading days: 41 (from daily_returns.size() or time span calculation)
// - MPR: sentio::metrics::compute_mpr_from_daily_returns(daily_returns) with 41-day data
// - Sharpe: Simplified calculation with different data
// - Daily trades: n_fill / trading_days
```

## Impact Assessment

### Severity: HIGH
- **Consistency**: Core metrics should be identical across all systems for the same run ID
- **Trust**: Users cannot rely on audit verification if metrics differ significantly
- **Decision Making**: Different MPR and Sharpe values could lead to different trading decisions

### Business Impact
- **Audit Verification**: Cannot verify strattest results using audit systems
- **Performance Tracking**: Inconsistent performance metrics across systems
- **Risk Management**: Different Sharpe ratios affect risk assessment

## Recommended Fixes

### 1. Unify Trading Days Calculation
- Ensure all systems use the same method to calculate trading days
- Use the same equity curve compression logic across systems

### 2. Unify Sharpe Calculation
- Replace simplified Sharpe calculation in audit systems with proper `compute_metrics_day_aware` method
- Use the same daily returns data across all systems

### 3. Unify Daily Returns Data
- Ensure all systems use the same daily returns data source
- Use consistent equity curve compression methodology

### 4. Add Validation
- Add cross-system validation to ensure metrics match for the same run ID
- Implement automated tests to catch future discrepancies

## Files Affected
- `src/unified_metrics.cpp` - strattest calculation
- `audit/src/audit_db.cpp` - audit summarize calculation
- `audit/src/audit_cli.cpp` - position-history display

## Test Commands
```bash
# Generate test run
./build/sentio_cli strattest sigor --simulations 1 --duration 1w

# Compare results (replace RUN_ID with actual run ID)
./build/sentio_audit summarize RUN_ID
./build/sentio_audit position-history RUN_ID
```

## Priority
**HIGH** - Core metrics consistency is critical for system reliability and user trust.
