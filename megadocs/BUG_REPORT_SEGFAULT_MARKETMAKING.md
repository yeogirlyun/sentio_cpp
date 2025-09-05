# Bug Report: Segmentation Fault in MarketMaking Strategy

## Summary
The MarketMaking strategy is experiencing a segmentation fault during backtesting when trying to push data to the `rolling_returns_` object. The issue occurs because the `rolling_returns_` object has `win=0` when it should have `win=20`.

## Environment
- OS: macOS 24.6.0
- Compiler: g++ with C++20
- Data: QQQ_RTH_NH.csv (753 bars, holiday-free data from 2022-01-01 to 2024-12-31)

## Symptoms
1. **Segmentation fault** occurs in `rolling_returns_.push(price_return)`
2. **Debug output shows**:
   - Constructor creates strategy with `rolling_returns_ win=20`
   - `set_params` is called successfully
   - When `calculate_signal` is called, `rolling_returns_ win=0`
   - Same memory address (`this=0x152e10a30`) but corrupted state

## Root Cause Analysis

### Issue 1: Object Corruption
The strategy object is being corrupted between `set_params` and `calculate_signal` calls. The `rolling_returns_` object shows:
- `win=20` after constructor and `apply_params()`
- `win=0` when `calculate_signal` is called

### Issue 2: Memory Management
The strategy is created twice:
1. During registration (shows debug output)
2. During actual use (shows debug output but with corrupted state)

Both instances have the same memory address, suggesting the object is being copied or moved incorrectly.

### Issue 3: RollingMeanVar Initialization
The `RollingMeanVar` object is not properly initialized when the strategy object is used. The constructor initializes it with `win=20`, but by the time `calculate_signal` is called, it has `win=0`.

## Debug Output
```
DEBUG: MarketMaking constructor called
DEBUG: About to call apply_params
DEBUG: After apply_params, rolling_returns_ win=20
DEBUG: Registering strategy: MarketMaking
...
DEBUG: Creating strategy: MarketMaking
DEBUG: MarketMaking constructor called
DEBUG: About to call apply_params
DEBUG: After apply_params, rolling_returns_ win=20
DEBUG: BaseStrategy::set_params called for MarketMaking
DEBUG: BaseStrategy::set_params completed for MarketMaking
DEBUG: MarketMaking calculate_signal called with current_index=0
DEBUG: this=0x152e10a30 rolling_returns_ win=0
DEBUG: MarketMaking calculate_signal called with current_index=1
DEBUG: this=0x152e10a30 rolling_returns_ win=0
DEBUG: About to calculate price_return
DEBUG: price_return calculated: -0.0129705
DEBUG: About to push to rolling_returns_
DEBUG: rolling_returns_ win=0 count=0
zsh: segmentation fault
```

## Code Analysis

### MarketMakingStrategy Constructor
```cpp
MarketMakingStrategy::MarketMakingStrategy() 
    : BaseStrategy("MarketMaking"),
      rolling_returns_(20),
      rolling_volume_(50) {
    std::cerr << "DEBUG: MarketMaking constructor called" << std::endl;
    params_ = get_default_params();
    std::cerr << "DEBUG: About to call apply_params" << std::endl;
    apply_params();
    std::cerr << "DEBUG: After apply_params, rolling_returns_ win=" << rolling_returns_.win << std::endl;
}
```

### apply_params Method
```cpp
void MarketMakingStrategy::apply_params() {
    // ... parameter assignments ...
    
    // Always recreate rolling objects with correct window sizes
    rolling_returns_ = RollingMeanVar(static_cast<int>(params_["volatility_window"]));
    rolling_volume_ = RollingMean(static_cast<int>(params_["volume_window"]));
    reset_state();
}
```

### RollingMeanVar Structure
```cpp
struct RollingMeanVar {
  int win, idx=0, count=0;
  std::vector<double> buf;
  double sum=0.0, sumsq=0.0;

  explicit RollingMeanVar(int w): win(w), buf(w,0.0) {}
  // ... rest of implementation
};
```

## Potential Causes

### 1. Copy Constructor Issues
The strategy object might be getting copied somewhere, and the copy doesn't have proper initialization. The `RollingMeanVar` and `RollingMean` objects don't have default constructors, so copying might cause issues.

### 2. Memory Corruption
The object might be getting corrupted due to:
- Buffer overflow
- Use after free
- Double free
- Memory alignment issues

### 3. Container Issues
The strategy might be stored in a container that causes copying or moving, leading to object corruption.

### 4. Virtual Function Issues
The strategy is accessed through a `BaseStrategy*` pointer, and there might be issues with virtual function calls or object slicing.

## Impact
- **Critical**: Complete failure of MarketMaking strategy
- **Data Loss**: No backtesting results can be generated
- **User Experience**: Segmentation fault crashes the application

## Recommended Fixes

### 1. Add Copy Constructor and Assignment Operator
```cpp
// In MarketMakingStrategy header
MarketMakingStrategy(const MarketMakingStrategy& other);
MarketMakingStrategy& operator=(const MarketMakingStrategy& other);
```

### 2. Add Move Constructor and Assignment Operator
```cpp
// In MarketMakingStrategy header
MarketMakingStrategy(MarketMakingStrategy&& other) noexcept;
MarketMakingStrategy& operator=(MarketMakingStrategy&& other) noexcept;
```

### 3. Add Debug Output to Track Object Lifecycle
Add debug output to:
- Copy constructor
- Assignment operator
- Move constructor
- Move assignment operator
- Destructor

### 4. Use Smart Pointers
Ensure the strategy is properly managed with `std::unique_ptr` or `std::shared_ptr`.

### 5. Add Memory Sanitizer
Compile with AddressSanitizer to detect memory corruption:
```bash
g++ -fsanitize=address -g ...
```

## Test Cases
1. **Basic Backtest**: `./build/sentio_cli backtest QQQ --strategy MarketMaking`
2. **Walk-Forward Test**: `./build/sentio_cli wf --db audit.sqlite --symbol QQQ --csv-root data`
3. **Memory Sanitizer Test**: Compile with AddressSanitizer and run backtest

## Priority
**HIGH** - This is a critical bug that prevents the MarketMaking strategy from working at all.

## Status
**OPEN** - Bug is still present and needs to be fixed.

## Assigned To
Development Team

## Created
2024-12-19

## Last Updated
2024-12-19
