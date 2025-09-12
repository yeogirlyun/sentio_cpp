# IRE Strategy Zero Signals/Trades Bug Report

## **🐛 Bug Summary**

**Issue**: IRE strategy is generating only HOLD signals with probabilities around 0.47-0.52, resulting in zero actual trades despite emitting 180 signals per day.

**Severity**: CRITICAL - Strategy is non-functional for trading

**Impact**: Complete loss of trading functionality for IRE strategy

**Root Cause**: Conflict resolution policy is preventing all allocation decisions from being executed, causing the strategy to only generate neutral HOLD signals.

## **🔍 Detailed Analysis**

### **Symptoms Observed**

1. **Signal Generation**: IRE strategy emits 180 signals per day (normal)
2. **Signal Type**: All signals are HOLD type with probabilities 0.47-0.52 (abnormal)
3. **Trade Execution**: Zero orders and fills generated (critical)
4. **Segmentation Faults**: Occurs during multi-day runs (stability issue)
5. **Audit Trail**: Shows signals passing validation but no subsequent orders

### **Expected Behavior**

- IRE strategy should generate BUY/SELL signals based on momentum analysis
- Signals should convert to orders and fills
- Strategy should execute trades based on 20-minute moving average momentum
- Target weights should be calculated and executed by the sizer

### **Actual Behavior**

- Only HOLD signals generated (probability ~0.5)
- No allocation decisions converted to orders
- Zero trade execution despite signal emission
- Strategy appears to be in perpetual neutral state

## **🔧 Technical Investigation**

### **Signal Generation Logic**

The IRE strategy uses a simple 20-minute moving average momentum signal:

```cpp
// From src/strategy_ire.cpp:217-218
double momentum = (bars[i].close - ma_20) / ma_20;
momentum_signal = 0.5 + std::clamp(momentum * 25.0, -0.4, 0.4);
```

**Issue**: The momentum calculation is working correctly, but the strategy is only returning probabilities, not actual BUY/SELL signals.

### **Allocation Decision Processing**

The runner processes allocation decisions from the strategy:

```cpp
// From src/runner.cpp:290-294
for (const auto& decision : allocation_decisions) {
    if (std::abs(decision.target_weight) > 1e-6) {
        execute_target_position(decision.instrument, decision.target_weight, ...);
    }
}
```

**Issue**: The `allocation_decisions` vector appears to be empty or contains only zero weights.

### **Conflict Resolution Logic**

The runner has aggressive conflict resolution that closes all conflicting positions:

```cpp
// From src/runner.cpp:246-286
if (has_conflicting_positions(portfolio, ST)) {
    conflicts_detected++;
    // Close all conflicting positions
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
            // Close position logic
        }
    }
}
```

**Issue**: This conflict resolution may be preventing any new positions from being established.

### **Sizer Logic**

The sizer calculates target quantities based on equity and target weights:

```cpp
// From include/sentio/sizer.hpp:70-95
double desired_notional = equity * std::abs(target_weight);
// Apply constraints...
double qty = desired_notional / instrument_price;
return (target_weight > 0) ? final_qty : -final_qty;
```

**Issue**: If target_weight is always 0 or very small, no trades will be executed.

## **🎯 Root Cause Analysis**

### **Primary Issue**: Strategy-Sizer Integration Gap

The IRE strategy is designed to return probabilities (0.0-1.0), but the runner expects allocation decisions with target weights. There's a missing conversion layer between:

1. **Strategy Output**: Probability values (0.47-0.52)
2. **Runner Expectation**: Target weight allocation decisions
3. **Sizer Input**: Target weights for position sizing

### **Secondary Issue**: Conflict Resolution Over-Aggressiveness

The conflict resolution logic may be:
1. Closing positions too aggressively
2. Preventing new positions from being established
3. Creating a feedback loop that keeps the portfolio empty

### **Tertiary Issue**: Signal Type Mapping

The strategy generates probabilities but doesn't map them to actual BUY/SELL/HOLD signal types that would trigger allocation decisions.

## **🔧 Proposed Solutions**

### **Solution 1: Fix Strategy-Sizer Integration**

Add a conversion layer in the runner to map strategy probabilities to allocation decisions:

```cpp
// Convert probability to target weight
double target_weight = 0.0;
if (probability > 0.6) {
    target_weight = (probability - 0.5) * 2.0; // Scale to [-1, 1]
} else if (probability < 0.4) {
    target_weight = (probability - 0.5) * 2.0; // Scale to [-1, 1]
}
```

### **Solution 2: Implement Proper Signal Type Mapping**

Modify the IRE strategy to generate actual StrategySignal objects with BUY/SELL types:

```cpp
StrategySignal signal;
if (probability > 0.6) {
    signal.type = StrategySignal::Type::BUY;
    signal.confidence = (probability - 0.5) * 2.0;
} else if (probability < 0.4) {
    signal.type = StrategySignal::Type::SELL;
    signal.confidence = (0.5 - probability) * 2.0;
} else {
    signal.type = StrategySignal::Type::HOLD;
    signal.confidence = 0.0;
}
```

### **Solution 3: Fix Conflict Resolution Logic**

Modify conflict resolution to be less aggressive:

```cpp
// Only close conflicting positions, don't prevent new ones
if (has_conflicting_positions(portfolio, ST)) {
    // Close only the conflicting positions
    // Allow new non-conflicting positions to be established
}
```

## **🧪 Testing Plan**

### **Unit Tests**

1. Test IRE strategy signal generation with known market data
2. Test probability-to-weight conversion logic
3. Test conflict resolution with various portfolio states
4. Test sizer with different target weights

### **Integration Tests**

1. Run IRE strategy with single day of data
2. Verify signal generation and allocation decision creation
3. Verify order execution and fill processing
4. Verify conflict resolution doesn't prevent legitimate trades

### **Regression Tests**

1. Compare IRE performance before and after fix
2. Verify other strategies still work correctly
3. Test multi-day runs to ensure no segmentation faults

## **📊 Expected Outcomes**

### **After Fix**

- IRE strategy generates BUY/SELL signals based on momentum
- Signals convert to orders and fills
- Strategy executes trades with proper position sizing
- No segmentation faults during multi-day runs
- Audit trail shows complete trade flow from signal to fill

### **Performance Metrics**

- Signal-to-trade conversion rate: >50%
- Daily trade count: 10-50 trades (healthy range)
- Sharpe ratio: >0.5 (improved from 0.0)
- No segmentation faults during extended runs

## **🚨 Immediate Actions Required**

1. **CRITICAL**: Fix strategy-sizer integration gap
2. **HIGH**: Implement proper signal type mapping
3. **MEDIUM**: Review conflict resolution aggressiveness
4. **LOW**: Add comprehensive logging for debugging

## **📝 Implementation Priority**

1. **Phase 1**: Fix core signal-to-allocation conversion
2. **Phase 2**: Implement proper signal type mapping
3. **Phase 3**: Optimize conflict resolution logic
4. **Phase 4**: Add comprehensive testing and validation

This bug is blocking all IRE strategy functionality and must be resolved immediately to restore trading capabilities.
