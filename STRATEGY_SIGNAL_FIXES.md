# Strategy Signal Generation Fixes

## Problem Summary
Three strategies are generating 0 signals due to overly restrictive thresholds:
- **MarketMaking**: 1515 threshold drops (inventory logic bug)
- **OrderFlowScalping**: 1512 threshold drops (imbalance threshold too high)  
- **VolatilityExpansion**: 1544 threshold drops (breakout threshold too high)

## Root Cause Analysis

### 1. MarketMaking Strategy
**Issue**: Inventory never updates, always stays at 0
```cpp
// Current logic:
double normalized_inventory = market_state_.inventory / max_inventory_; // Always 0/100 = 0
double inventory_skew = -normalized_inventory * inventory_skew_mult_; // Always 0
if (inventory_skew > 0.001) { // Never true
    signal.type = StrategySignal::Type::BUY;
}
```

**Fix**: Track inventory based on fills
```cpp
// Add to MarketMakingStrategy class:
void update_inventory(const Order& fill) {
    if (fill.side == OrderSide::BUY) {
        market_state_.inventory += fill.quantity;
    } else {
        market_state_.inventory -= fill.quantity;
    }
}
```

### 2. OrderFlowScalping Strategy  
**Issue**: Imbalance threshold 0.65 is too restrictive
```cpp
// Current logic:
if (avg_pressure > imbalance_threshold_) // 0.65 - too high
    of_state_ = OFState::ArmedLong;
```

**Fix**: Lower threshold to 0.55
```cpp
{"imbalance_threshold", 0.55}, // Was 0.65
```

### 3. VolatilityExpansion Strategy
**Issue**: Breakout threshold 0.9 is too high
```cpp
// Current logic:
const double up_trigger = hh + breakout_k_ * atr_; // 0.9 * ATR - too large
if (bar.close > up_trigger) { // Rarely triggered
```

**Fix**: Lower breakout_k to 0.6
```cpp
{"breakout_k", 0.6}, // Was 0.9
```

## Implementation Priority
1. **High Impact**: OrderFlowScalping (simple parameter change)
2. **Medium Impact**: VolatilityExpansion (simple parameter change)  
3. **Low Impact**: MarketMaking (requires inventory tracking system)

## Expected Results
After fixes:
- OrderFlowScalping: Should generate 50-100 signals
- VolatilityExpansion: Should generate 20-50 signals
- MarketMaking: Will need inventory tracking to generate signals
