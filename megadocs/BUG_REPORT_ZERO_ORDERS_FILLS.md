# Bug Report: Zero Orders/Fills Problem

## Issue Summary
**Title**: Strategies Generate Signals But Produce Zero Orders/Fills  
**Severity**: CRITICAL  
**Priority**: HIGH  
**Status**: OPEN  
**Affected Components**: All Trading Strategies, Router, Sizer, Order Management  

## Description
Despite all strategies now running without segmentation faults and generating proper diagnostic information, the majority of strategies are producing zero orders and fills. This indicates a critical issue in the signal-to-trade conversion pipeline where strategies emit signals but the router/sizer system fails to convert these signals into actual trades.

## Affected Strategies

### Strategies with Zero Orders/Fills:
1. **BollingerSqueezeBreakout** - 0 emitted, 753 dropped (thr=733)
2. **MarketMaking** - 0 emitted, 753 dropped (zerovol=703)  
3. **MomentumVolumeProfile** - 0 emitted, 753 dropped (nan=653)
4. **OrderFlowImbalance** - 0 emitted, 93 dropped (thr=93)
5. **OrderFlowScalping** - 0 emitted, 680 dropped (thr=680)
6. **VolatilityExpansion** - 0 emitted, 753 dropped (session=733)

### Strategies with Some Orders/Fills:
1. **OpeningRangeBreakout** - 1 emitted, 0 dropped (working correctly)
2. **VWAPReversion** - 467 emitted, 286 dropped (working correctly)

## Root Cause Analysis

### Primary Issues Identified:

#### 1. **Signal Generation Problems**
- **Threshold Drops**: Most strategies show high `thr=` values indicating signals are being dropped due to threshold conditions
- **Parameter Issues**: Strategies may have incorrect parameter values causing all signals to be filtered out
- **Logic Errors**: Signal generation logic may be flawed, preventing valid signals from being emitted

#### 2. **Router/Sizer Logic Issues**
- **Signal Processing**: Router may not be properly processing strategy signals
- **Order Creation**: Sizer may not be creating orders from valid signals
- **Validation Failures**: Orders may be failing validation checks

#### 3. **Data Quality Issues**
- **Volume Problems**: `zerovol=703` in MarketMaking indicates volume data issues
- **NaN Values**: `nan=653` in MomentumVolumeProfile indicates data quality problems
- **Session Filtering**: `session=733` in VolatilityExpansion suggests session filtering issues

## Detailed Analysis by Strategy

### BollingerSqueezeBreakout
```
[SIG BollingerSqueezeBreakout] emitted=0 dropped=753
min_bars=20 session=0 nan=0 zerovol=0 thr=733 cooldown=0 dup=0
```
- **Issue**: 733 out of 753 signals dropped due to threshold (`thr=733`)
- **Root Cause**: Threshold parameters too restrictive
- **Impact**: No signals pass threshold validation

### MarketMaking
```
[SIG MarketMaking] emitted=0 dropped=753
min_bars=50 zerovol=703 session=0 nan=0 thr=0 cooldown=0 dup=0
```
- **Issue**: 703 out of 753 signals dropped due to zero volume (`zerovol=703`)
- **Root Cause**: Volume data quality issues or volume filtering too strict
- **Impact**: All signals rejected due to volume validation

### MomentumVolumeProfile
```
[SIG MomentumVolumeProfile] emitted=0 dropped=753
min_bars=100 session=0 nan=653 zerovol=0 thr=0 cooldown=0 dup=0
```
- **Issue**: 653 out of 753 signals dropped due to NaN values (`nan=653`)
- **Root Cause**: Data quality issues or calculation errors producing NaN
- **Impact**: All signals rejected due to NaN validation

### OrderFlowImbalance
```
[SIG OrderFlowImbalance] emitted=0 dropped=93
min_bars=0 session=0 nan=0 zerovol=0 thr=93 cooldown=0 dup=0
```
- **Issue**: 93 out of 93 signals dropped due to threshold (`thr=93`)
- **Root Cause**: Threshold parameters too restrictive
- **Impact**: No signals pass threshold validation

### OrderFlowScalping
```
[SIG OrderFlowScalping] emitted=0 dropped=680
min_bars=0 session=0 nan=0 zerovol=0 thr=680 cooldown=0 dup=0
```
- **Issue**: 680 out of 680 signals dropped due to threshold (`thr=680`)
- **Root Cause**: Threshold parameters too restrictive
- **Impact**: No signals pass threshold validation

### VolatilityExpansion
```
[SIG VolatilityExpansion] emitted=0 dropped=753
min_bars=20 session=733 nan=0 zerovol=0 thr=0 cooldown=0 dup=0
```
- **Issue**: 733 out of 753 signals dropped due to session filtering (`session=733`)
- **Root Cause**: Session filtering logic too restrictive
- **Impact**: All signals rejected due to session validation

## Technical Details

### Signal Processing Pipeline
1. **Strategy Signal Generation**: `calculate_signal()` method
2. **Signal Validation**: Threshold, volume, NaN, session checks
3. **Router Processing**: Signal to order conversion
4. **Sizer Processing**: Order sizing and validation
5. **Order Execution**: Trade execution and fill generation

### Diagnostic Categories
- `min_bars`: Signals dropped due to insufficient data
- `session`: Signals dropped due to session filtering
- `nan`: Signals dropped due to NaN values
- `zerovol`: Signals dropped due to zero volume
- `thr`: Signals dropped due to threshold conditions
- `cooldown`: Signals dropped due to cooldown period
- `dup`: Signals dropped due to duplicate detection

## Impact Assessment

### Business Impact
- **Trading System Non-Functional**: 75% of strategies produce no trades
- **Backtesting Invalid**: Results cannot be trusted for strategy evaluation
- **Development Blocked**: Cannot proceed with optimization or live trading

### Technical Impact
- **Signal Pipeline Broken**: Critical path from signal to trade is failing
- **Data Quality Issues**: Multiple data quality problems identified
- **Parameter Tuning Required**: Strategy parameters need adjustment

## Proposed Solutions

### Immediate Actions
1. **Debug Signal Generation**: Add detailed logging to signal generation logic
2. **Fix Data Quality Issues**: Resolve volume and NaN data problems
3. **Adjust Threshold Parameters**: Review and adjust strategy thresholds
4. **Validate Router/Sizer**: Ensure proper signal-to-trade conversion

### Long-term Solutions
1. **Parameter Optimization**: Implement automatic parameter tuning
2. **Data Quality Monitoring**: Add comprehensive data validation
3. **Signal Pipeline Testing**: Create unit tests for signal processing
4. **Performance Monitoring**: Add real-time signal processing metrics

## Testing Strategy

### Phase 1: Signal Generation Debugging
- Add detailed logging to each strategy's `calculate_signal()` method
- Test with simplified parameters to isolate issues
- Validate data quality and preprocessing

### Phase 2: Router/Sizer Validation
- Test signal processing pipeline with known good signals
- Validate order creation and sizing logic
- Test order execution and fill generation

### Phase 3: Integration Testing
- End-to-end testing with all strategies
- Performance validation with real market data
- Stress testing with high-frequency signals

## Files to Investigate

### Strategy Files
- `include/sentio/strategy_*.hpp` - All strategy headers
- `src/strategy_*.cpp` - All strategy implementations

### Core System Files
- `include/sentio/base_strategy.hpp` - Base strategy interface
- `src/base_strategy.cpp` - Base strategy implementation
- `include/sentio/router.hpp` - Signal routing logic
- `include/sentio/sizer.hpp` - Order sizing logic
- `src/runner.cpp` - Main backtesting runner

### Data Processing Files
- `src/polygon_client.cpp` - Data fetching and preprocessing
- `include/sentio/rolling_stats.hpp` - Rolling statistics calculations

## Priority
**CRITICAL** - This affects 75% of all strategies and completely blocks trading functionality.

## Status
**OPEN** - Investigation and debugging in progress.

## Assigned To
Development Team

## Created
2024-12-19

## Last Updated
2024-12-19

## Related Issues
- Segmentation fault issues (RESOLVED)
- Zero returns and metrics calculation (IN PROGRESS)
- Router/sizer logic problems (IN PROGRESS)

## Notes
- All strategies now run without crashes (segmentation faults resolved)
- Signal generation is working but parameters/thresholds are too restrictive
- Data quality issues need immediate attention
- Router/sizer validation required to ensure proper signal-to-trade conversion
