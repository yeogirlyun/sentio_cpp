# Bug Report: Missing Function Implementations

## 🚨 **CRITICAL ISSUE**

**Status**: 🔴 **ACTIVE**  
**Severity**: **CRITICAL** - Build failure due to missing implementations  
**Impact**: **100% build failure** - Executable cannot be created  
**Date**: September 5, 2024  

## 📋 **Summary**

The build process is failing due to missing function implementations that are declared in headers but not implemented in source files. This prevents the creation of the `sentio_cli` executable and blocks all testing of the RTH validation fixes.

## 🔍 **Problem Description**

### **Build Status**
- ✅ **Compilation**: All source files compile successfully
- ❌ **Linking**: Fails due to undefined symbols
- ❌ **Executable**: `sentio_cli` cannot be created
- ❌ **Testing**: Cannot test RTH validation fixes

### **Error Details**
```
Undefined symbols for architecture arm64:
  "sentio::route(sentio::StrategySignal const&, sentio::RouterCfg const&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> const&)", referenced from:
      sentio::run_backtest(...) in runner.o

  "sentio::PriceBook::get_latest(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> const&) const", referenced from:
      sentio::route_and_create_order(...) in router.o
```

## 🐛 **Missing Functions**

### **1. `sentio::route` Function**
- **Declaration**: `include/sentio/router.hpp:24`
- **Signature**: `std::optional<RouteDecision> route(const StrategySignal& s, const RouterCfg& cfg, const std::string& base_symbol)`
- **Status**: ❌ **NOT IMPLEMENTED**
- **Impact**: Critical - Required by `run_backtest` function

### **2. `PriceBook::get_latest` Method**
- **Declaration**: `include/sentio/pnl_accounting.hpp:16`
- **Signature**: `const Bar* get_latest(const std::string& instrument) const`
- **Status**: ❌ **NOT IMPLEMENTED**
- **Impact**: Critical - Required by `route_and_create_order` function

## 🔧 **Root Cause Analysis**

### **Missing Implementations**
The following functions are declared in headers but have no corresponding implementations:

1. **`route` function**: 
   - Declared in `include/sentio/router.hpp`
   - Should be implemented in `src/router.cpp`
   - Required for strategy signal routing

2. **`PriceBook::get_latest` method**:
   - Declared in `include/sentio/pnl_accounting.hpp`
   - Should be implemented in `src/pnl_accounting.cpp` (file doesn't exist)
   - Required for price lookup

### **File Structure Issues**
- `src/pnl_accounting.cpp` does not exist
- `src/router.cpp` is missing the `route` function implementation
- Headers declare functions but source files don't implement them

## 📊 **Impact Assessment**

### **Immediate Impact**
- **Build failure**: Cannot create executable
- **Testing blocked**: Cannot test RTH validation fixes
- **Development halted**: All strategy testing impossible

### **Affected Components**
- Router system
- Price book system
- Strategy execution pipeline
- Backtesting framework

## 🛠️ **Reproduction Steps**

1. **Build the project**:
   ```bash
   make clean && make
   ```

2. **Observe failure**:
   ```
   Undefined symbols for architecture arm64:
     "sentio::route(...)", referenced from:
         sentio::run_backtest(...) in runner.o
     "sentio::PriceBook::get_latest(...)", referenced from:
         sentio::route_and_create_order(...) in router.o
   ```

3. **Verify missing executable**:
   ```bash
   ls -la build/sentio_cli
   # No such file or directory
   ```

## 🔍 **Technical Details**

### **Missing Function Signatures**

#### **1. Route Function**
```cpp
// Declaration in include/sentio/router.hpp:24
std::optional<RouteDecision> route(const StrategySignal& s, const RouterCfg& cfg, const std::string& base_symbol);

// Expected implementation in src/router.cpp
std::optional<RouteDecision> route(const StrategySignal& s, const RouterCfg& cfg, const std::string& base_symbol) {
    // Implementation needed
}
```

#### **2. PriceBook::get_latest Method**
```cpp
// Declaration in include/sentio/pnl_accounting.hpp:16
const Bar* get_latest(const std::string& instrument) const;

// Expected implementation in src/pnl_accounting.cpp
const Bar* PriceBook::get_latest(const std::string& instrument) const {
    // Implementation needed
}
```

### **Dependencies**
- **Router system**: Depends on `route` function
- **Price lookup**: Depends on `PriceBook::get_latest`
- **Strategy execution**: Depends on both functions

## 🎯 **Proposed Solutions**

### **Immediate Fix**
1. **Implement `route` function** in `src/router.cpp`
2. **Create `src/pnl_accounting.cpp`** with `PriceBook::get_latest` implementation
3. **Add missing method implementations** to existing classes

### **Implementation Requirements**

#### **1. Route Function**
- Take strategy signal and routing configuration
- Return optional route decision
- Handle different signal types (Buy/Sell)
- Apply routing rules and constraints

#### **2. PriceBook::get_latest Method**
- Look up latest bar for given instrument
- Return pointer to Bar or nullptr if not found
- Handle instrument name mapping
- Provide price data for routing decisions

### **Long-term Fix**
1. **Code review**: Ensure all declared functions are implemented
2. **Build validation**: Add checks for missing implementations
3. **Documentation**: Document all required implementations

## 📈 **Success Criteria**

### **Immediate Success**
- ✅ Build completes without linking errors
- ✅ `sentio_cli` executable is created
- ✅ RTH validation can be tested

### **Long-term Success**
- ✅ All declared functions have implementations
- ✅ Build process is robust and reliable
- ✅ No missing function errors in future

## 🚨 **Priority**

**CRITICAL** - This issue blocks all testing and development work.

## 📝 **Notes**

- This issue was discovered after installing C++20 timezone libraries
- The compilation succeeded but linking failed due to missing implementations
- Once fixed, we can test the RTH validation regression fix
- The missing functions are core to the strategy execution pipeline

## 🔗 **Related Issues**

- **RTH Validation Regression**: Cannot test due to build failure
- **Strategy Testing**: Blocked until executable is created
- **Timezone Library Installation**: Completed but cannot be tested

---

**Report Generated**: September 5, 2024  
**Last Updated**: September 5, 2024  
**Status**: 🔴 **ACTIVE**
