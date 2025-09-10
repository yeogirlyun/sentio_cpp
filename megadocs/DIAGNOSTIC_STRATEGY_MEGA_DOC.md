# DIAGNOSTIC_STRATEGY_MEGA_DOC

**Generated**: 2025-09-11 03:43:28
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: Comprehensive documentation for implementing a diagnostic strategy that generates guaranteed signals for system validation

**Total Files**: 0

---

## 🐛 **BUG REPORT**

# Diagnostic Strategy Requirements Document

## 1. Overview

### 1.1 Purpose
Create a simple diagnostic strategy that is guaranteed to generate signals for any meaningful real data, with a minimum of 100 signals per day. This strategy will serve as a system diagnostic tool to verify that the VM test infrastructure, Runner, Router, and Sizer components are functioning correctly.

### 1.2 Objectives
- **Primary**: Generate consistent, predictable signals for system validation
- **Secondary**: Test leverage ticker support (TQQQ, SQQQ)
- **Tertiary**: Validate signal routing and position sizing mechanisms
- **Note**: Profit optimization is NOT an objective

## 2. Strategy Specifications

### 2.1 Core Logic
- **Indicator**: RSI (Relative Strength Index) with aggressive thresholds
- **Signal Generation**: Multiple signals per bar based on RSI conditions
- **Frequency**: Minimum 100 signals per day (approximately 1 signal every 4 minutes)
- **Volatility**: High sensitivity to market movements

### 2.2 Technical Parameters
- **RSI Period**: 14 bars (standard)
- **RSI Overbought**: 70 (triggers sell signals)
- **RSI Oversold**: 30 (triggers buy signals)
- **RSI Neutral**: 50 (triggers neutral/reversal signals)
- **Signal Multiplier**: 3x signals per RSI condition (buy, sell, neutral)

### 2.3 Signal Types
1. **RSI < 30**: Strong buy signal (weight: +0.8)
2. **RSI > 70**: Strong sell signal (weight: -0.8)
3. **RSI 30-50**: Moderate buy signal (weight: +0.4)
4. **RSI 50-70**: Moderate sell signal (weight: -0.4)
5. **RSI = 50**: Neutral signal (weight: 0.0)

### 2.4 Leverage Ticker Support
- **Primary Symbol**: QQQ
- **Leverage Tickers**: TQQQ (3x), SQQQ (-3x)
- **Signal Distribution**: 
  - QQQ: 40% of signals
  - TQQQ: 30% of signals  
  - SQQQ: 30% of signals

## 3. Implementation Requirements

### 3.1 Class Structure
```cpp
class DiagnosticStrategy : public BaseStrategy {
private:
    // RSI calculation
    std::vector<double> price_history_;
    int rsi_period_;
    
    // Signal generation
    int signal_count_;
    double last_rsi_;
    
    // Leverage ticker management
    std::vector<std::string> leverage_symbols_;
    int current_symbol_index_;
    
public:
    DiagnosticStrategy();
    ~DiagnosticStrategy() override;
    
    // Core strategy methods
    double calculate_target_weight(const Bar& bar) override;
    void on_bar(const Bar& bar) override;
    void initialize() override;
    
    // Diagnostic methods
    int get_signal_count() const;
    double get_last_rsi() const;
    std::string get_current_symbol() const;
};
```

### 3.2 Key Methods

#### 3.2.1 RSI Calculation
```cpp
double calculate_rsi(const std::vector<double>& prices, int period) {
    // Standard RSI calculation using price changes
    // Returns value between 0-100
}
```

#### 3.2.2 Signal Generation
```cpp
double calculate_target_weight(const Bar& bar) override {
    // 1. Calculate RSI from price history
    // 2. Determine signal type based on RSI value
    // 3. Select leverage ticker (QQQ/TQQQ/SQQQ)
    // 4. Return target weight (-1.0 to +1.0)
    // 5. Increment signal counter
}
```

#### 3.2.3 Leverage Ticker Selection
```cpp
std::string select_leverage_ticker() {
    // Rotate through QQQ, TQQQ, SQQQ
    // Return current symbol for signal generation
}
```

## 4. Integration Requirements

### 4.1 Strategy Factory Registration
- Register as "DIAGNOSTIC" in StrategyFactory
- Ensure proper instantiation and parameter passing

### 4.2 VM Test Integration
- Compatible with existing VM test infrastructure
- Support for synthetic data generation
- Proper signal counting and reporting

### 4.3 Runner Integration
- Compatible with existing Runner component
- Support for multiple symbol routing
- Proper position sizing and risk management

## 5. Testing Requirements

### 5.1 Unit Tests
- RSI calculation accuracy
- Signal generation frequency
- Leverage ticker rotation
- Weight calculation correctness

### 5.2 Integration Tests
- VM test compatibility
- Runner integration
- Router and Sizer functionality
- Signal counting accuracy

### 5.3 Performance Tests
- Minimum 100 signals per day
- Consistent signal generation across different market conditions
- Proper leverage ticker distribution

## 6. Validation Criteria

### 6.1 Signal Generation
- **Minimum**: 100 signals per day
- **Consistency**: Signals generated across all market conditions
- **Distribution**: Proper leverage ticker usage

### 6.2 System Integration
- **VM Test**: Successful execution in virtual market environment
- **Runner**: Proper signal processing and execution
- **Router**: Correct symbol routing and position management
- **Sizer**: Appropriate position sizing based on signals

### 6.3 Diagnostic Value
- **Verification**: Confirms system components are functional
- **Baseline**: Establishes expected signal generation rate
- **Comparison**: Enables comparison with other strategies

## 7. File Structure

### 7.1 Header Files
- `include/sentio/strategy_diagnostic.hpp`

### 7.2 Source Files
- `src/strategy_diagnostic.cpp`

### 7.3 Test Files
- `tests/test_diagnostic_strategy.cpp`

### 7.4 Configuration Files
- `configs/strategies/diagnostic.json`

## 8. Dependencies

### 8.1 Core Dependencies
- `BaseStrategy` (inheritance)
- `Bar` (data structure)
- `StrategyFactory` (registration)
- `Runner` (execution)

### 8.2 External Dependencies
- Standard C++ libraries
- No external mathematical libraries required

## 9. Success Metrics

### 9.1 Primary Metrics
- **Signal Count**: ≥100 signals per day
- **Signal Distribution**: 40% QQQ, 30% TQQQ, 30% SQQQ
- **RSI Accuracy**: Correct RSI calculation and thresholds

### 9.2 Secondary Metrics
- **System Integration**: Successful VM test execution
- **Performance**: Consistent signal generation across market conditions
- **Diagnostic Value**: Confirms system functionality

## 10. Implementation Timeline

### 10.1 Phase 1: Core Implementation
- RSI calculation logic
- Basic signal generation
- Strategy class structure

### 10.2 Phase 2: Leverage Integration
- Leverage ticker support
- Symbol rotation logic
- Multi-symbol signal distribution

### 10.3 Phase 3: Testing & Validation
- Unit test development
- Integration testing
- VM test validation

### 10.4 Phase 4: Documentation & Deployment
- Code documentation
- Configuration setup
- System integration

## 11. Risk Assessment

### 11.1 Technical Risks
- **RSI Calculation**: Potential accuracy issues
- **Signal Frequency**: May not meet 100 signals/day requirement
- **Integration**: Compatibility issues with existing system

### 11.2 Mitigation Strategies
- **Thorough Testing**: Comprehensive unit and integration tests
- **Fallback Logic**: Alternative signal generation methods
- **Monitoring**: Real-time signal counting and validation

## 12. Conclusion

This diagnostic strategy will serve as a critical tool for validating the Sentio C++ trading system infrastructure. By generating predictable, frequent signals across multiple leverage tickers, it will enable comprehensive testing of all system components and provide a baseline for comparing other strategies.

The strategy's simplicity and high signal frequency make it ideal for system diagnostics, while its leverage ticker support ensures comprehensive testing of the routing and sizing mechanisms.


---

## 📋 **TABLE OF CONTENTS**

1. [Bug Report](#-bug-report)
2. [Source Modules](#-source-modules)

---

## 📁 **SOURCE MODULES**

### Base Strategy Interface
**File**: `include/sentio/base_strategy.hpp`

The base strategy interface defines the core structure that all strategies must implement, including:
- Parameter management system
- Signal generation interface
- Allocation decision framework
- Strategy evaluation capabilities

### Strategy Registry
**File**: `include/sentio/strategy_registry.hpp`

The strategy registry provides dynamic strategy loading and management:
- Strategy registration and factory pattern
- Configuration-based strategy loading
- Strategy discovery and instantiation

### Router Interface
**File**: `include/sentio/router.hpp`

The router handles signal routing and position management:
- Strategy signal processing
- Multi-symbol routing (QQQ, TQQQ, SQQQ)
- Position tracking and management

### Sizer Interface
**File**: `include/sentio/sizer.hpp`

The sizer manages position sizing and risk management:
- Capital allocation
- Risk-adjusted position sizing
- Dynamic sizing based on market conditions

### Runner Interface
**File**: `include/sentio/runner.hpp`

The runner orchestrates strategy execution:
- Strategy execution engine
- Performance tracking
- Trade history management

### Virtual Market Engine
**File**: `include/sentio/virtual_market.hpp`

The virtual market engine provides synthetic data generation and testing:
- Monte Carlo simulation
- MarS integration
- Fast historical data generation

### Mars Data Loader
**File**: `include/sentio/mars_data_loader.hpp`

The MarS data loader handles integration with the MarS simulation engine:
- MarS data generation
- Data format conversion
- Python bridge integration

---

