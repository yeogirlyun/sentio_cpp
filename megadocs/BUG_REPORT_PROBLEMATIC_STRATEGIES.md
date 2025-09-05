# Bug Report: Problematic Strategies and RTH Validation Issues

## Issue Summary
**Title**: Three Strategies Not Working + RTH Validation Issue  
**Severity**: HIGH  
**Priority**: HIGH  
**Status**: OPEN  
**Affected Components**: VolatilityExpansion, MarketMaking, OrderFlowScalping, RTH Validation Logic  

## Description
After comprehensive testing of all 8 trading strategies with fresh 3-year RTH data, three strategies are experiencing critical issues that prevent them from generating meaningful signals and trades. Additionally, there's an RTH validation issue that prevents binary file loading despite CSV files working correctly.

## Affected Strategies

### 1. VolatilityExpansion Strategy
**Status**: ❌ **NOT WORKING**  
**Issue**: All signals dropped due to threshold conditions  
**Symptoms**:
- 0 signals emitted, 292,776 signals dropped
- All signals rejected due to threshold (`thr=292756`)
- 0 fills, 0% return, 0 Sharpe ratio
- Final equity unchanged at 100,000

**Root Cause**: Threshold parameters are too restrictive, filtering out all valid signals

### 2. MarketMaking Strategy  
**Status**: ❌ **NOT WORKING**  
**Issue**: All signals dropped due to threshold conditions  
**Symptoms**:
- 0 signals emitted, 292,776 signals dropped
- All signals rejected due to threshold (`thr=292706`)
- 0 fills, 0% return, 0 Sharpe ratio
- Final equity unchanged at 100,000

**Root Cause**: Threshold parameters are too restrictive, filtering out all valid signals

### 3. OrderFlowScalping Strategy
**Status**: ⚠️ **PARTIALLY WORKING**  
**Issue**: Very few signals generated despite strategy functioning  
**Symptoms**:
- 0 signals emitted, 111,774 signals dropped
- All signals rejected due to threshold (`thr=111774`)
- Only 2 fills generated
- Positive performance (+1.54% return, +0.39 Sharpe) but minimal activity

**Root Cause**: Threshold parameters too restrictive, allowing only minimal signal generation

## RTH Validation Issue

### Problem Description
**Status**: ❌ **CRITICAL**  
**Issue**: RTH validation logic incorrectly rejects valid RTH data when loading binary files  
**Symptoms**:
- Binary files load successfully with CSV data
- Binary files fail with RTH validation error
- Error message: "Non-RTH data found after filtering!"
- Timestamp example: `2022-09-06T12:01:00-04:00` (valid RTH time)

**Root Cause**: RTH verification logic has timezone conversion or validation bugs

## Detailed Analysis

### VolatilityExpansion Strategy Analysis
```
[SIG VolatilityExpansion] emitted=0 dropped=292776
min_bars=20 session=0 nan=0 zerovol=0 thr=292756 cooldown=0 dup=0
```

**Issues Identified**:
1. **Threshold Too High**: 292,756 out of 292,776 signals dropped due to threshold
2. **Parameter Problem**: Threshold parameters need adjustment
3. **Signal Logic**: Strategy logic may be flawed

**Expected Behavior**: Should generate signals during volatility expansion periods
**Actual Behavior**: No signals pass threshold validation

### MarketMaking Strategy Analysis
```
[SIG MarketMaking] emitted=0 dropped=292776
min_bars=50 session=0 nan=0 zerovol=20 thr=292706 cooldown=0 dup=0
```

**Issues Identified**:
1. **Threshold Too High**: 292,706 out of 292,776 signals dropped due to threshold
2. **Volume Issues**: 20 signals dropped due to zero volume (`zerovol=20`)
3. **Parameter Problem**: Threshold parameters need adjustment

**Expected Behavior**: Should generate market making signals based on volatility and volume
**Actual Behavior**: No signals pass threshold validation

### OrderFlowScalping Strategy Analysis
```
[SIG OrderFlowScalping] emitted=0 dropped=111774
min_bars=0 session=0 nan=0 zerovol=0 thr=111774 cooldown=0 dup=0
```

**Issues Identified**:
1. **Threshold Too High**: 111,774 out of 111,774 signals dropped due to threshold
2. **Partial Functionality**: Strategy generates some trades (2 fills) despite no signals
3. **Parameter Problem**: Threshold parameters need adjustment

**Expected Behavior**: Should generate frequent scalping signals
**Actual Behavior**: Minimal signal generation, some trades executed

### RTH Validation Issue Analysis
```
FATAL ERROR: Non-RTH data found after filtering!
 -> Symbol: QQQ
 -> Timestamp (UTC): 2022-09-06T12:01:00-04:00
 -> NYT Epoch: 1662480060
```

**Issues Identified**:
1. **Timezone Conversion**: RTH validation logic has timezone conversion bugs
2. **Binary vs CSV**: CSV loading works, binary loading fails
3. **Validation Logic**: RTH verification incorrectly rejects valid times

**Expected Behavior**: 12:01 PM ET should be valid RTH time
**Actual Behavior**: RTH validation rejects valid RTH timestamps

## Impact Assessment

### Business Impact
- **25% Strategy Failure Rate**: 2 out of 8 strategies completely non-functional
- **12.5% Partial Failure Rate**: 1 out of 8 strategies barely functional
- **Performance Loss**: Missing potential returns from 3 strategies
- **Binary Loading Blocked**: Cannot use optimized binary data loading

### Technical Impact
- **Threshold Logic Broken**: Signal filtering too restrictive
- **RTH Validation Broken**: Binary loading completely blocked
- **Parameter Tuning Required**: Strategy parameters need optimization
- **Performance Degradation**: Forced to use slower CSV loading

## Root Cause Analysis

### Threshold Issues
1. **Parameter Values**: Default threshold parameters are too high
2. **Signal Logic**: Strategy signal generation logic may be flawed
3. **Data Quality**: Market data may not meet strategy requirements
4. **Parameter Tuning**: No automatic parameter optimization

### RTH Validation Issues
1. **Timezone Conversion**: UTC to NYT conversion logic has bugs
2. **Binary Format**: Binary file format may not match C++ expectations
3. **Validation Logic**: RTH validation criteria may be incorrect
4. **Data Integrity**: Binary data may be corrupted during packing

## Proposed Solutions

### Immediate Actions
1. **Fix Threshold Parameters**: Adjust threshold values for all three strategies
2. **Debug RTH Validation**: Fix timezone conversion and validation logic
3. **Parameter Optimization**: Implement automatic parameter tuning
4. **Signal Logic Review**: Review and fix strategy signal generation logic

### Long-term Solutions
1. **Parameter Optimization Framework**: Implement Bayesian optimization
2. **RTH Validation Overhaul**: Complete rewrite of RTH validation logic
3. **Binary Format Standardization**: Ensure binary format matches C++ expectations
4. **Comprehensive Testing**: Add unit tests for all strategy parameters

## Testing Strategy

### Phase 1: Threshold Parameter Fixes
- Reduce threshold values for VolatilityExpansion and MarketMaking
- Test with different threshold values to find optimal settings
- Validate signal generation with adjusted parameters

### Phase 2: RTH Validation Fixes
- Debug timezone conversion logic
- Fix binary file loading issues
- Validate RTH verification with known good data

### Phase 3: Strategy Logic Review
- Review signal generation logic for all three strategies
- Implement parameter optimization
- Add comprehensive logging for debugging

## Files to Investigate

### Strategy Files
- `include/sentio/strategy_volatility_expansion.hpp`
- `src/strategy_volatility_expansion.cpp`
- `include/sentio/strategy_market_making.hpp`
- `src/strategy_market_making.cpp`
- `include/sentio/strategy_order_flow_scalping.hpp`
- `src/strategy_order_flow_scalping.cpp`

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
**HIGH** - This affects 37.5% of all strategies and completely blocks binary loading functionality.

## Status
**OPEN** - Investigation and debugging in progress.

## Assigned To
Development Team

## Created
2024-12-19

## Last Updated
2024-12-19

## Related Issues
- Zero orders/fills problem (PARTIALLY RESOLVED)
- Data packing issue (RESOLVED)
- Segmentation fault issues (RESOLVED)

## Notes
- 6 out of 8 strategies are working correctly (75% success rate)
- Main issues are threshold parameters and RTH validation logic
- CSV loading works perfectly, binary loading has RTH validation issues
- Strategies need parameter optimization for better performance
