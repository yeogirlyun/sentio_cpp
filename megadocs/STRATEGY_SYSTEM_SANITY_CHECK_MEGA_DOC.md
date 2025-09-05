# Strategy System Sanity Check Requirements and Source Code

**Generated**: 2025-09-05 20:22:23
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: Complete requirements document for strategy system validation along with all source code for the C++ trading system

**Total Files**: 92

---

## 📋 **TABLE OF CONTENTS**

1. [docs/AUDIT_TRAIL_REQUIREMENTS.md](#file-1)
2. [docs/STRATEGY_SYSTEM_SANITY_CHECK_REQUIREMENTS.md](#file-2)
3. [include/sentio/all_strategies.hpp](#file-3)
4. [include/sentio/alpha.hpp](#file-4)
5. [include/sentio/audit.hpp](#file-5)
6. [include/sentio/base_strategy.hpp](#file-6)
7. [include/sentio/binio.hpp](#file-7)
8. [include/sentio/bo.hpp](#file-8)
9. [include/sentio/bollinger.hpp](#file-9)
10. [include/sentio/calendar_seed.hpp](#file-10)
11. [include/sentio/core.hpp](#file-11)
12. [include/sentio/cost_model.hpp](#file-12)
13. [include/sentio/csv_loader.hpp](#file-13)
14. [include/sentio/data_resolver.hpp](#file-14)
15. [include/sentio/day_index.hpp](#file-15)
16. [include/sentio/exec_types.hpp](#file-16)
17. [include/sentio/feature_health.hpp](#file-17)
18. [include/sentio/indicators.hpp](#file-18)
19. [include/sentio/metrics.hpp](#file-19)
20. [include/sentio/of_index.hpp](#file-20)
21. [include/sentio/of_precompute.hpp](#file-21)
22. [include/sentio/optimizer.hpp](#file-22)
23. [include/sentio/orderflow_types.hpp](#file-23)
24. [include/sentio/pnl_accounting.hpp](#file-24)
25. [include/sentio/polygon_client.hpp](#file-25)
26. [include/sentio/polygon_ingest.hpp](#file-26)
27. [include/sentio/position_manager.hpp](#file-27)
28. [include/sentio/pricebook.hpp](#file-28)
29. [include/sentio/profiling.hpp](#file-29)
30. [include/sentio/rolling_stats.hpp](#file-30)
31. [include/sentio/router.hpp](#file-31)
32. [include/sentio/rth_calendar.hpp](#file-32)
33. [include/sentio/runner.hpp](#file-33)
34. [include/sentio/signal_diag.hpp](#file-34)
35. [include/sentio/signal_engine.hpp](#file-35)
36. [include/sentio/signal_gate.hpp](#file-36)
37. [include/sentio/signal_pipeline.hpp](#file-37)
38. [include/sentio/signal_trace.hpp](#file-38)
39. [include/sentio/sizer.hpp](#file-39)
40. [include/sentio/strategy_base.hpp](#file-40)
41. [include/sentio/strategy_bollinger_squeeze_breakout.hpp](#file-41)
42. [include/sentio/strategy_market_making.hpp](#file-42)
43. [include/sentio/strategy_momentum_volume.hpp](#file-43)
44. [include/sentio/strategy_opening_range_breakout.hpp](#file-44)
45. [include/sentio/strategy_order_flow_imbalance.hpp](#file-45)
46. [include/sentio/strategy_order_flow_scalping.hpp](#file-46)
47. [include/sentio/strategy_sma_cross.hpp](#file-47)
48. [include/sentio/strategy_vwap_reversion.hpp](#file-48)
49. [include/sentio/symbol_table.hpp](#file-49)
50. [include/sentio/time_utils.hpp](#file-50)
51. [include/sentio/wf.hpp](#file-51)
52. [src/audit.cpp](#file-52)
53. [src/base_strategy.cpp](#file-53)
54. [src/csv_loader.cpp](#file-54)
55. [src/feature_health.cpp](#file-55)
56. [src/main.cpp](#file-56)
57. [src/optimizer.cpp](#file-57)
58. [src/pnl_accounting.cpp](#file-58)
59. [src/poly_fetch_main.cpp](#file-59)
60. [src/polygon_client.cpp](#file-60)
61. [src/polygon_ingest.cpp](#file-61)
62. [src/router.cpp](#file-62)
63. [src/rth_calendar.cpp](#file-63)
64. [src/runner.cpp](#file-64)
65. [src/signal_engine.cpp](#file-65)
66. [src/signal_gate.cpp](#file-66)
67. [src/signal_pipeline.cpp](#file-67)
68. [src/signal_trace.cpp](#file-68)
69. [src/strategy_bollinger_squeeze_breakout.cpp](#file-69)
70. [src/strategy_market_making.cpp](#file-70)
71. [src/strategy_momentum_volume.cpp](#file-71)
72. [src/strategy_opening_range_breakout.cpp](#file-72)
73. [src/strategy_order_flow_imbalance.cpp](#file-73)
74. [src/strategy_order_flow_scalping.cpp](#file-74)
75. [src/strategy_sma_cross.cpp](#file-75)
76. [src/strategy_vwap_reversion.cpp](#file-76)
77. [src/test_rth.cpp](#file-77)
78. [src/time_utils.cpp](#file-78)
79. [src/wf.cpp](#file-79)
80. [tests/test_audit_replay.cpp](#file-80)
81. [tests/test_audit_simple.cpp](#file-81)
82. [tests/test_pipeline_emits.cpp](#file-82)
83. [tests/test_sma_cross_emit.cpp](#file-83)
84. [tools/create_mega_document.py](#file-84)
85. [tools/data_downloader.py](#file-85)
86. [tools/detailed_strategy_diagnostics.cpp](#file-86)
87. [tools/extended_strategy_test.cpp](#file-87)
88. [tools/integration_example.cpp](#file-88)
89. [tools/signal_diagnostics.cpp](#file-89)
90. [tools/simple_strategy_diagnostics.cpp](#file-90)
91. [tools/strategy_diagnostics.cpp](#file-91)
92. [tools/trace_analyzer.cpp](#file-92)

---

## 📄 **FILE 1 of 92**: docs/AUDIT_TRAIL_REQUIREMENTS.md

**File Information**:
- **Path**: `docs/AUDIT_TRAIL_REQUIREMENTS.md`

- **Size**: 486 lines
- **Modified**: 2025-09-05 17:30:53

- **Type**: .md

```text
# Audit Trail Requirements for SQLite Replay and P/L Verification

## Overview

This document defines the requirements for implementing a comprehensive audit trail system using SQLite to enable complete replay of trading sessions and accurate P/L verification. The system must maintain minimal performance degradation while providing complete traceability of all trading decisions.

## 1. Core Requirements

### 1.1 Complete Traceability
- **Every bar processed** must be recorded with full context
- **Every signal generated** must be logged with reasoning
- **Every order created** must be tracked from signal to execution
- **Every trade executed** must be recorded with exact timestamps
- **Every P/L calculation** must be traceable to source trades

### 1.2 Replay Capability
- **Exact sequence reproduction**: Any test run must be perfectly replayable
- **State reconstruction**: All strategy states must be reconstructible
- **Deterministic results**: Same input must produce identical output
- **Step-by-step debugging**: Ability to pause and inspect at any point

### 1.3 Performance Requirements
- **Minimal overhead**: < 1% performance impact on backtesting
- **Efficient storage**: Compressed data storage with minimal disk usage
- **Fast queries**: Sub-second response times for replay queries
- **Batch operations**: Bulk insert/update operations for efficiency

## 2. Database Schema Design

### 2.1 Core Tables

#### `audit_sessions`
```sql
CREATE TABLE audit_sessions (
    session_id INTEGER PRIMARY KEY,
    start_time INTEGER NOT NULL,           -- UTC epoch
    end_time INTEGER,                      -- UTC epoch
    strategy_name TEXT NOT NULL,
    instrument TEXT NOT NULL,
    config_hash TEXT NOT NULL,            -- Hash of strategy config
    data_source TEXT NOT NULL,            -- Source data file/stream
    status TEXT NOT NULL,                 -- RUNNING, COMPLETED, FAILED
    total_bars INTEGER DEFAULT 0,
    signals_generated INTEGER DEFAULT 0,
    orders_created INTEGER DEFAULT 0,
    trades_executed INTEGER DEFAULT 0,
    final_pnl REAL DEFAULT 0.0,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);
```

#### `audit_bars`
```sql
CREATE TABLE audit_bars (
    bar_id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,
    bar_index INTEGER NOT NULL,           -- Sequential bar number
    timestamp INTEGER NOT NULL,           -- UTC epoch
    instrument TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    is_rth BOOLEAN NOT NULL,
    strategy_state BLOB,                  -- Serialized strategy state
    processing_time_us INTEGER,           -- Microseconds to process
    FOREIGN KEY (session_id) REFERENCES audit_sessions(session_id)
);
```

#### `audit_signals`
```sql
CREATE TABLE audit_signals (
    signal_id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,
    bar_id INTEGER NOT NULL,
    signal_type TEXT NOT NULL,            -- BUY, SELL, STRONG_BUY, etc.
    confidence REAL NOT NULL,
    reasoning TEXT,                       -- Human-readable reason
    strategy_params BLOB,                 -- Strategy parameters at time of signal
    drop_reason TEXT,                     -- If signal was dropped
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES audit_sessions(session_id),
    FOREIGN KEY (bar_id) REFERENCES audit_bars(bar_id)
);
```

#### `audit_orders`
```sql
CREATE TABLE audit_orders (
    order_id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,
    signal_id INTEGER NOT NULL,
    bar_id INTEGER NOT NULL,
    order_type TEXT NOT NULL,             -- MARKET, LIMIT, etc.
    side TEXT NOT NULL,                   -- BUY, SELL
    quantity REAL NOT NULL,
    price REAL,                           -- NULL for market orders
    instrument TEXT NOT NULL,
    status TEXT NOT NULL,                 -- PENDING, FILLED, CANCELLED, REJECTED
    fill_price REAL,
    fill_quantity REAL,
    fill_time INTEGER,                    -- UTC epoch
    commission REAL DEFAULT 0.0,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES audit_sessions(session_id),
    FOREIGN KEY (signal_id) REFERENCES audit_signals(signal_id),
    FOREIGN KEY (bar_id) REFERENCES audit_bars(bar_id)
);
```

#### `audit_trades`
```sql
CREATE TABLE audit_trades (
    trade_id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,
    order_id INTEGER NOT NULL,
    bar_id INTEGER NOT NULL,
    side TEXT NOT NULL,                   -- BUY, SELL
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    instrument TEXT NOT NULL,
    timestamp INTEGER NOT NULL,           -- UTC epoch
    pnl REAL DEFAULT 0.0,                 -- P&L for this trade
    cumulative_pnl REAL DEFAULT 0.0,      -- Running P&L
    position_after REAL DEFAULT 0.0,      -- Position after this trade
    commission REAL DEFAULT 0.0,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES audit_sessions(session_id),
    FOREIGN KEY (order_id) REFERENCES audit_orders(order_id),
    FOREIGN KEY (bar_id) REFERENCES audit_bars(bar_id)
);
```

#### `audit_positions`
```sql
CREATE TABLE audit_positions (
    position_id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,
    bar_id INTEGER NOT NULL,
    instrument TEXT NOT NULL,
    quantity REAL NOT NULL,
    avg_price REAL NOT NULL,
    unrealized_pnl REAL DEFAULT 0.0,
    realized_pnl REAL DEFAULT 0.0,
    timestamp INTEGER NOT NULL,           -- UTC epoch
    FOREIGN KEY (session_id) REFERENCES audit_sessions(session_id),
    FOREIGN KEY (bar_id) REFERENCES audit_bars(bar_id)
);
```

### 2.2 Indexes for Performance

```sql
-- Session queries
CREATE INDEX idx_audit_sessions_time ON audit_sessions(start_time, end_time);
CREATE INDEX idx_audit_sessions_strategy ON audit_sessions(strategy_name, instrument);

-- Bar queries
CREATE INDEX idx_audit_bars_session ON audit_bars(session_id, bar_index);
CREATE INDEX idx_audit_bars_time ON audit_bars(timestamp);

-- Signal queries
CREATE INDEX idx_audit_signals_session ON audit_signals(session_id, bar_id);
CREATE INDEX idx_audit_signals_type ON audit_signals(signal_type, created_at);

-- Order queries
CREATE INDEX idx_audit_orders_session ON audit_orders(session_id, bar_id);
CREATE INDEX idx_audit_orders_status ON audit_orders(status, created_at);

-- Trade queries
CREATE INDEX idx_audit_trades_session ON audit_trades(session_id, bar_id);
CREATE INDEX idx_audit_trades_time ON audit_trades(timestamp);

-- Position queries
CREATE INDEX idx_audit_positions_session ON audit_positions(session_id, bar_id);
CREATE INDEX idx_audit_positions_instrument ON audit_positions(instrument, timestamp);
```

## 3. Implementation Requirements

### 3.1 Audit Manager Class

```cpp
class AuditManager {
public:
    // Session management
    int start_session(const std::string& strategy_name, 
                     const std::string& instrument,
                     const std::string& config_hash,
                     const std::string& data_source);
    void end_session(int session_id, const std::string& status);
    
    // Bar logging
    void log_bar(int session_id, int bar_index, const Bar& bar, 
                 bool is_rth, const std::string& strategy_state,
                 int processing_time_us);
    
    // Signal logging
    int log_signal(int session_id, int bar_id, const StrategySignal& signal,
                   const std::string& reasoning, const std::string& drop_reason);
    
    // Order logging
    int log_order(int session_id, int signal_id, int bar_id, 
                  const Order& order);
    void update_order_status(int order_id, const std::string& status,
                            double fill_price, double fill_quantity,
                            int fill_time, double commission);
    
    // Trade logging
    int log_trade(int session_id, int order_id, int bar_id,
                  const Trade& trade, double pnl, double cumulative_pnl,
                  double position_after, double commission);
    
    // Position logging
    void log_position(int session_id, int bar_id, const std::string& instrument,
                     double quantity, double avg_price, double unrealized_pnl,
                     double realized_pnl);
    
    // Replay functionality
    std::vector<Bar> replay_bars(int session_id);
    std::vector<StrategySignal> replay_signals(int session_id);
    std::vector<Order> replay_orders(int session_id);
    std::vector<Trade> replay_trades(int session_id);
    double calculate_final_pnl(int session_id);
    
    // Performance optimization
    void enable_batch_mode();
    void flush_batch();
    void optimize_database();
    
private:
    sqlite3* db_;
    bool batch_mode_;
    std::vector<std::string> batch_queries_;
};
```

### 3.2 Performance Optimizations

#### 3.2.1 Batch Operations
- **Batch inserts**: Collect multiple operations and execute in single transaction
- **Prepared statements**: Reuse prepared statements for repeated operations
- **Connection pooling**: Maintain persistent database connections
- **Async logging**: Non-blocking audit operations where possible

#### 3.2.2 Data Compression
- **Strategy state compression**: Use efficient serialization (MessagePack/Protocol Buffers)
- **Text compression**: Compress reasoning and parameter text fields
- **Binary storage**: Store complex objects as BLOB with compression

#### 3.2.3 Query Optimization
- **Materialized views**: Pre-compute common aggregations
- **Partitioning**: Partition tables by session_id for better performance
- **Vacuum operations**: Regular database maintenance during off-hours

### 3.3 Integration Points

#### 3.3.1 Strategy Integration
```cpp
class BaseStrategy {
protected:
    AuditManager* audit_manager_;
    int current_session_id_;
    int current_bar_id_;
    
public:
    virtual void on_bar(const Bar& bar) override {
        // Log bar processing
        auto start = std::chrono::high_resolution_clock::now();
        
        // Process bar (existing logic)
        StrategySignal signal = calculate_signal(bar);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto processing_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Log bar and signal
        audit_manager_->log_bar(current_session_id_, current_bar_id_, bar, 
                               is_rth, serialize_state(), processing_time.count());
        
        if (signal.type != StrategySignal::Type::HOLD) {
            audit_manager_->log_signal(current_session_id_, current_bar_id_, signal,
                                     generate_reasoning(signal), "");
        }
    }
};
```

#### 3.3.2 Order Management Integration
```cpp
class OrderManager {
private:
    AuditManager* audit_manager_;
    
public:
    Order create_order(const StrategySignal& signal, const Bar& bar) {
        Order order = generate_order(signal, bar);
        
        // Log order creation
        int order_id = audit_manager_->log_order(current_session_id_, 
                                                signal.signal_id, 
                                                current_bar_id_, 
                                                order);
        order.id = order_id;
        
        return order;
    }
    
    void execute_trade(const Order& order, double fill_price, double fill_quantity) {
        Trade trade = create_trade(order, fill_price, fill_quantity);
        
        // Calculate P&L
        double pnl = calculate_pnl(trade);
        double cumulative_pnl = get_cumulative_pnl() + pnl;
        double position_after = update_position(trade);
        
        // Log trade
        audit_manager_->log_trade(current_session_id_, order.id, current_bar_id_,
                                 trade, pnl, cumulative_pnl, position_after, 
                                 calculate_commission(trade));
    }
};
```

## 4. Replay and Verification Features

### 4.1 Complete Replay
```cpp
class ReplayEngine {
public:
    // Replay entire session
    void replay_session(int session_id) {
        auto bars = audit_manager_->replay_bars(session_id);
        auto signals = audit_manager_->replay_signals(session_id);
        auto orders = audit_manager_->replay_orders(session_id);
        auto trades = audit_manager_->replay_trades(session_id);
        
        // Reconstruct exact sequence
        for (const auto& bar : bars) {
            // Process bar in exact same order
            process_bar(bar);
            
            // Verify signals match
            verify_signals(bar, signals);
            
            // Verify orders match
            verify_orders(bar, orders);
            
            // Verify trades match
            verify_trades(bar, trades);
        }
    }
    
    // Step-by-step debugging
    void replay_step(int session_id, int bar_index) {
        // Replay up to specific bar
        auto bars = audit_manager_->replay_bars(session_id);
        for (int i = 0; i <= bar_index; ++i) {
            process_bar(bars[i]);
        }
    }
};
```

### 4.2 P/L Verification
```cpp
class PnLVerifier {
public:
    struct PnLReport {
        double total_realized_pnl;
        double total_unrealized_pnl;
        double total_commission;
        double net_pnl;
        std::vector<Trade> trades;
        std::vector<Position> positions;
    };
    
    PnLReport verify_session_pnl(int session_id) {
        auto trades = audit_manager_->replay_trades(session_id);
        auto positions = audit_manager_->replay_positions(session_id);
        
        PnLReport report;
        report.total_realized_pnl = 0.0;
        report.total_commission = 0.0;
        
        for (const auto& trade : trades) {
            report.total_realized_pnl += trade.pnl;
            report.total_commission += trade.commission;
            report.trades.push_back(trade);
        }
        
        for (const auto& position : positions) {
            report.total_unrealized_pnl += position.unrealized_pnl;
            report.positions.push_back(position);
        }
        
        report.net_pnl = report.total_realized_pnl + report.total_unrealized_pnl - report.total_commission;
        
        return report;
    }
};
```

## 5. Configuration and Maintenance

### 5.1 Database Configuration
```cpp
struct AuditConfig {
    std::string db_path = "audit.db";
    bool enable_compression = true;
    bool enable_batch_mode = true;
    int batch_size = 1000;
    int vacuum_interval_hours = 24;
    int max_retention_days = 365;
    bool enable_async_logging = true;
};
```

### 5.2 Maintenance Operations
- **Daily vacuum**: Optimize database performance
- **Data archival**: Move old sessions to archive tables
- **Index rebuilding**: Rebuild indexes for optimal performance
- **Data validation**: Verify data integrity and consistency

## 6. Testing and Validation

### 6.1 Unit Tests
- Test audit logging accuracy
- Test replay functionality
- Test P/L calculation correctness
- Test performance benchmarks

### 6.2 Integration Tests
- End-to-end audit trail verification
- Replay accuracy validation
- Performance regression testing
- Data integrity validation

## 7. Monitoring and Alerts

### 7.1 Performance Monitoring
- Database query performance
- Audit logging latency
- Disk usage growth
- Memory usage patterns

### 7.2 Data Quality Monitoring
- Missing audit records
- Inconsistent P/L calculations
- Replay accuracy validation
- Data corruption detection

## 8. Implementation Timeline

### Phase 1: Core Infrastructure (Week 1-2)
- Database schema implementation
- Basic AuditManager class
- Integration with existing strategies

### Phase 2: Performance Optimization (Week 3-4)
- Batch operations implementation
- Query optimization
- Compression and storage optimization

### Phase 3: Replay and Verification (Week 5-6)
- ReplayEngine implementation
- PnLVerifier implementation
- Testing and validation

### Phase 4: Production Deployment (Week 7-8)
- Performance tuning
- Monitoring implementation
- Documentation and training

## 9. Success Criteria

- **100% traceability**: Every trading decision is logged and traceable
- **Perfect replay**: Any session can be exactly reproduced
- **Accurate P/L**: P/L calculations are verified and consistent
- **< 1% performance impact**: Minimal overhead on backtesting performance
- **Sub-second queries**: Fast response times for replay operations
- **Data integrity**: 100% data consistency and accuracy

This audit trail system will provide complete transparency and traceability for all trading operations while maintaining high performance and efficiency.

```

## 📄 **FILE 2 of 92**: docs/STRATEGY_SYSTEM_SANITY_CHECK_REQUIREMENTS.md

**File Information**:
- **Path**: `docs/STRATEGY_SYSTEM_SANITY_CHECK_REQUIREMENTS.md`

- **Size**: 324 lines
- **Modified**: 2025-09-05 20:22:19

- **Type**: .md

```text
# Strategy System Sanity Check Requirements

## Overview

This document defines the requirements for a comprehensive sanity check of the entire strategy system, from signal generation through trade execution to audit verification. The goal is to establish a reliable baseline for strategy testing and experimentation that can support high-performance trading systems.

## 1. System Architecture Validation

### 1.1 Signal Generation Pipeline
- **Requirement**: Verify that all strategy components can generate valid signals
- **Components to Test**:
  - Strategy base classes and interfaces
  - All registered strategies (VWAPReversion, MarketMaking, MomentumVolume, etc.)
  - Signal validation and filtering mechanisms
  - RTH (Regular Trading Hours) filtering
  - Warmup period handling
  - NaN and invalid data handling

### 1.2 Signal Processing Chain
- **Requirement**: Ensure signals flow correctly through the processing pipeline
- **Components to Test**:
  - Signal gate and filtering logic
  - Router decision making
  - Order generation and sizing
  - Position management
  - P&L calculation accuracy

### 1.3 Audit Trail Integrity
- **Requirement**: Verify complete traceability from signals to final P&L
- **Components to Test**:
  - JSONL event logging
  - Event sequence integrity
  - SHA1 hash verification
  - Replay accuracy
  - P&L reconstruction

## 2. Data Quality Assurance

### 2.1 Input Data Validation
- **Requirement**: Ensure all input data meets quality standards
- **Tests**:
  - Bar data completeness (OHLCV)
  - Timestamp monotonicity
  - Price data validity (no negative prices, reasonable ranges)
  - Volume data consistency
  - RTH calendar accuracy

### 2.2 Feature Health Monitoring
- **Requirement**: Detect and report data quality issues
- **Tests**:
  - Gap detection in time series
  - NaN/infinity propagation tracking
  - Outlier detection
  - Data continuity validation

## 3. Strategy Performance Validation

### 3.1 Signal Quality Metrics
- **Requirement**: Measure and validate signal quality
- **Metrics**:
  - Signal frequency and distribution
  - Signal-to-noise ratio
  - False positive/negative rates
  - Signal persistence and stability

### 3.2 Execution Quality
- **Requirement**: Verify trade execution accuracy
- **Metrics**:
  - Order fill rates
  - Slippage analysis
  - Execution timing accuracy
  - Position sizing accuracy

### 3.3 P&L Accuracy
- **Requirement**: Ensure P&L calculations are correct and auditable
- **Tests**:
  - Cash flow accuracy
  - Position valuation
  - Realized vs unrealized P&L
  - Fee and cost accounting

## 4. System Robustness Testing

### 4.1 Error Handling
- **Requirement**: System must handle errors gracefully
- **Tests**:
  - Invalid input data handling
  - Network/API failures
  - Memory allocation failures
  - File I/O errors
  - Database connection issues

### 4.2 Performance Under Load
- **Requirement**: System must maintain performance under realistic load
- **Tests**:
  - High-frequency data processing
  - Multiple strategy execution
  - Concurrent audit logging
  - Memory usage optimization

### 4.3 Edge Case Handling
- **Requirement**: System must handle edge cases correctly
- **Tests**:
  - Market open/close transitions
  - Holiday calendar handling
  - Leap year date handling
  - Timezone conversions
  - Very small or very large numbers

## 5. Audit Trail Verification

### 5.1 Event Completeness
- **Requirement**: All events must be captured and logged
- **Tests**:
  - Every bar generates a bar event
  - Every signal generates signal/route/order/fill events
  - Every trade generates complete audit trail
  - No missing or duplicate events

### 5.2 Event Integrity
- **Requirement**: All events must be accurate and tamper-proof
- **Tests**:
  - SHA1 hash verification
  - Event sequence validation
  - Data consistency checks
  - Timestamp accuracy

### 5.3 Replay Accuracy
- **Requirement**: Audit trail must allow perfect replay
- **Tests**:
  - P&L reconstruction accuracy
  - Position reconstruction accuracy
  - Cash flow reconstruction accuracy
  - State consistency verification

## 6. Integration Testing

### 6.1 End-to-End Workflow
- **Requirement**: Complete workflow must function correctly
- **Tests**:
  - Data loading → Signal generation → Trade execution → Audit logging
  - Multiple strategies running simultaneously
  - Walk-forward testing integration
  - Optimization workflow integration

### 6.2 Cross-Component Communication
- **Requirement**: All components must communicate correctly
- **Tests**:
  - Strategy → Router communication
  - Router → Order management communication
  - Order management → Audit communication
  - Audit → Replay communication

## 7. Performance Benchmarks

### 7.1 Latency Requirements
- **Requirement**: System must meet latency targets
- **Targets**:
  - Signal generation: < 1ms per bar
  - Order processing: < 100μs per order
  - Audit logging: < 10μs per event
  - Replay processing: < 1ms per 1000 events

### 7.2 Throughput Requirements
- **Requirement**: System must handle required throughput
- **Targets**:
  - Bar processing: > 10,000 bars/second
  - Event logging: > 100,000 events/second
  - Replay processing: > 1,000,000 events/second

### 7.3 Memory Requirements
- **Requirement**: System must use memory efficiently
- **Targets**:
  - Base memory usage: < 100MB
  - Memory growth: < 1MB per 10,000 events
  - No memory leaks over 24-hour runs

## 8. Validation Test Suite

### 8.1 Unit Tests
- **Requirement**: All components must have comprehensive unit tests
- **Coverage**:
  - Strategy signal generation
  - Router decision logic
  - Order management
  - Audit logging and replay
  - P&L calculations

### 8.2 Integration Tests
- **Requirement**: Component interactions must be tested
- **Tests**:
  - Strategy → Router → Order → Audit pipeline
  - Multi-strategy execution
  - Walk-forward testing
  - Optimization workflows

### 8.3 Stress Tests
- **Requirement**: System must handle stress conditions
- **Tests**:
  - High-frequency data processing
  - Memory pressure conditions
  - Network failure scenarios
  - Disk I/O failures

### 8.4 Regression Tests
- **Requirement**: Changes must not break existing functionality
- **Tests**:
  - Automated test suite execution
  - Performance regression detection
  - Accuracy regression detection
  - Compatibility verification

## 9. Monitoring and Alerting

### 9.1 Real-time Monitoring
- **Requirement**: System must provide real-time monitoring
- **Metrics**:
  - Signal generation rates
  - Order execution rates
  - P&L tracking
  - Error rates
  - Performance metrics

### 9.2 Alerting System
- **Requirement**: System must alert on critical issues
- **Alerts**:
  - Signal generation failures
  - Order execution failures
  - Audit trail corruption
  - Performance degradation
  - System errors

## 10. Documentation and Maintenance

### 10.1 System Documentation
- **Requirement**: Complete system documentation must be maintained
- **Components**:
  - Architecture diagrams
  - API documentation
  - Configuration guides
  - Troubleshooting guides
  - Performance tuning guides

### 10.2 Maintenance Procedures
- **Requirement**: Clear maintenance procedures must be established
- **Procedures**:
  - Regular system health checks
  - Performance monitoring
  - Error log analysis
  - System updates and patches
  - Backup and recovery procedures

## 11. Compliance and Security

### 11.1 Audit Compliance
- **Requirement**: System must meet audit requirements
- **Standards**:
  - Complete event logging
  - Immutable audit trail
  - Tamper detection
  - Data retention policies
  - Access controls

### 11.2 Security Requirements
- **Requirement**: System must be secure
- **Standards**:
  - Data encryption at rest
  - Secure communication channels
  - Access authentication
  - Audit trail protection
  - Vulnerability management

## 12. Success Criteria

### 12.1 Functional Requirements
- ✅ All strategies generate valid signals
- ✅ All signals are properly routed and executed
- ✅ All trades are accurately recorded and auditable
- ✅ P&L calculations are correct and verifiable
- ✅ System handles all edge cases gracefully

### 12.2 Performance Requirements
- ✅ System meets latency and throughput targets
- ✅ Memory usage is within acceptable limits
- ✅ No performance degradation over time
- ✅ System scales with increased load

### 12.3 Quality Requirements
- ✅ 100% test coverage for critical components
- ✅ Zero data loss or corruption
- ✅ Perfect audit trail integrity
- ✅ Complete replay accuracy

## 13. Implementation Plan

### Phase 1: Core System Validation (Week 1)
1. Implement comprehensive unit tests
2. Validate signal generation pipeline
3. Test audit trail integrity
4. Verify P&L calculations

### Phase 2: Integration Testing (Week 2)
1. End-to-end workflow testing
2. Multi-strategy execution testing
3. Walk-forward testing validation
4. Performance benchmarking

### Phase 3: Stress Testing (Week 3)
1. High-frequency data processing tests
2. Memory and performance stress tests
3. Error handling validation
4. Edge case testing

### Phase 4: Production Readiness (Week 4)
1. Monitoring and alerting setup
2. Documentation completion
3. Security validation
4. Compliance verification

## 14. Conclusion

This sanity check framework ensures that the strategy system is robust, accurate, and ready for high-performance trading. By validating every component from signal generation through audit verification, we establish a solid foundation for strategy development and experimentation.

The comprehensive test suite and monitoring systems provide ongoing confidence in system reliability and performance, enabling rapid iteration and optimization of trading strategies while maintaining data integrity and audit compliance.

```

## 📄 **FILE 3 of 92**: include/sentio/all_strategies.hpp

**File Information**:
- **Path**: `include/sentio/all_strategies.hpp`

- **Size**: 12 lines
- **Modified**: 2025-09-05 17:25:23

- **Type**: .hpp

```text
#pragma once

// This file ensures all strategies are included and registered with the factory.
// Include this header once in your main.cpp.

#include "strategy_bollinger_squeeze_breakout.hpp"
#include "strategy_market_making.hpp"
#include "strategy_momentum_volume.hpp"
#include "strategy_opening_range_breakout.hpp"
#include "strategy_order_flow_imbalance.hpp"
#include "strategy_order_flow_scalping.hpp"
#include "strategy_vwap_reversion.hpp"
```

## 📄 **FILE 4 of 92**: include/sentio/alpha.hpp

**File Information**:
- **Path**: `include/sentio/alpha.hpp`

- **Size**: 7 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <string>

namespace sentio {
// Removed: Direction enum and StratSignal struct. These are now defined in core.hpp
} // namespace sentio


```

## 📄 **FILE 5 of 92**: include/sentio/audit.hpp

**File Information**:
- **Path**: `include/sentio/audit.hpp`

- **Size**: 95 lines
- **Modified**: 2025-09-05 20:13:14

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <optional>
#include <unordered_map>
#include <vector>

namespace sentio {

// --- Core enums/structs ----
enum class SigType : uint8_t { BUY=0, STRONG_BUY=1, SELL=2, STRONG_SELL=3, HOLD=4 };
enum class Side    : uint8_t { Buy=0, Sell=1 };

// Simple Bar structure for audit system (avoiding conflicts with core.hpp)
struct AuditBar { 
  double open{}, high{}, low{}, close{}, volume{};
};

struct AuditPosition { 
  double qty{0.0}; 
  double avg_px{0.0}; 
};

struct AccountState {
  double cash{0.0};
  double realized{0.0};
  double equity{0.0};
  // computed: equity = cash + realized + sum(qty * mark_px)
};

struct AuditConfig {
  std::string run_id;           // stable id for this run
  std::string file_path;        // where JSONL events are appended
  bool        flush_each=true;  // fsync-ish (fflush) after each write
};

// --- Recorder: append events to JSONL ---
class AuditRecorder {
public:
  explicit AuditRecorder(const AuditConfig& cfg);
  ~AuditRecorder();

  // lifecycle
  void event_run_start(std::int64_t ts_utc, const std::string& meta_json="{}");
  void event_run_end(std::int64_t ts_utc, const std::string& meta_json="{}");

  // data plane
  void event_bar   (std::int64_t ts_utc, const std::string& instrument, const AuditBar& b);
  void event_signal(std::int64_t ts_utc, const std::string& base_symbol, SigType type, double confidence);
  void event_route (std::int64_t ts_utc, const std::string& base_symbol, const std::string& instrument, double target_weight);
  void event_order (std::int64_t ts_utc, const std::string& instrument, Side side, double qty, double limit_px);
  void event_fill  (std::int64_t ts_utc, const std::string& instrument, double price, double qty, double fees, Side side);
  void event_snapshot(std::int64_t ts_utc, const AccountState& acct);
  void event_metric (std::int64_t ts_utc, const std::string& key, double value);

  // Get current config (for creating new instances)
  AuditConfig get_config() const { return {run_id_, file_path_, flush_each_}; }

private:
  std::string run_id_;
  std::string file_path_;
  std::FILE*  fp_{nullptr};
  std::uint64_t seq_{0};
  bool flush_each_;
  void write_line_(const std::string& s);
  static std::string sha1_hex_(const std::string& s); // tiny local impl
  static std::string json_escape_(const std::string& s);
};

// --- Replayer: read JSONL, rebuild state, recompute P&L, verify ---
struct ReplayResult {
  // recomputed
  std::unordered_map<std::string, AuditPosition> positions;
  AccountState acct{};
  std::size_t  bars{0}, signals{0}, routes{0}, orders{0}, fills{0};
  // mismatches discovered
  std::vector<std::string> issues;
};

class AuditReplayer {
public:
  // price map can be filled from bar events; you may also inject EOD marks
  struct PriceBook { std::unordered_map<std::string, double> last_px; };

  // replay the file; return recomputed account/pnl from fills + marks
  static std::optional<ReplayResult> replay_file(const std::string& file_path,
                                                 const std::string& run_id_expect = "");
private:
  static bool apply_bar_(PriceBook& pb, const std::string& instrument, const AuditBar& b);
  static void mark_to_market_(const PriceBook& pb, ReplayResult& rr);
  static void apply_fill_(ReplayResult& rr, const std::string& inst, double px, double qty, double fees, Side side);
};

} // namespace sentio
```

## 📄 **FILE 6 of 92**: include/sentio/base_strategy.hpp

**File Information**:
- **Path**: `include/sentio/base_strategy.hpp`

- **Size**: 100 lines
- **Modified**: 2025-09-05 15:29:30

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "signal_diag.hpp"
#include "router.hpp"  // for StrategySignal
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <functional>

namespace sentio {

// Parameter types and enums
enum class ParamType { INT, FLOAT };
struct ParamSpec { 
    ParamType type;
    double min_val, max_val;
    double default_val;
};

using ParameterMap = std::unordered_map<std::string, double>;
using ParameterSpace = std::unordered_map<std::string, ParamSpec>;

enum class SignalType { NONE = 0, BUY = 1, SELL = -1, STRONG_BUY = 2, STRONG_SELL = -2 };

struct StrategyState {
    bool in_position = false;
    SignalType last_signal = SignalType::NONE;
    int last_trade_bar = -1000; // Initialize far in the past
    
    void reset() {
        in_position = false;
        last_signal = SignalType::NONE;
        last_trade_bar = -1000;
    }
};

// StrategySignal is now defined in router.hpp

class BaseStrategy {
protected:
    std::string name_;
    ParameterMap params_;
    StrategyState state_;
    SignalDiag diag_;

    bool is_cooldown_active(int current_bar, int cooldown_period) const;
    
public:
    BaseStrategy(const std::string& name) : name_(name) {}
    virtual ~BaseStrategy() = default;

    // **MODIFIED**: Explicitly delete copy and move operations for this polymorphic base class.
    // This prevents object slicing and ownership confusion.
    BaseStrategy(const BaseStrategy&) = delete;
    BaseStrategy& operator=(const BaseStrategy&) = delete;
    BaseStrategy(BaseStrategy&&) = delete;
    BaseStrategy& operator=(BaseStrategy&&) = delete;
    
    std::string get_name() const { return name_; }
    
    virtual ParameterMap get_default_params() const = 0;
    virtual ParameterSpace get_param_space() const = 0;
    virtual void apply_params() = 0;
    
    void set_params(const ParameterMap& params);
    
    virtual StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) = 0;
    virtual void reset_state();
    
    const SignalDiag& get_diag() const { return diag_; }
};

class StrategyFactory {
public:
    using CreateFunction = std::function<std::unique_ptr<BaseStrategy>()>;
    
    static StrategyFactory& instance();
    void register_strategy(const std::string& name, CreateFunction create_func);
    std::unique_ptr<BaseStrategy> create_strategy(const std::string& name);
    std::vector<std::string> get_available_strategies() const;
    
private:
    std::unordered_map<std::string, CreateFunction> strategies_;
};

// **NEW**: The final, more robust registration macro.
// It takes the C++ ClassName and the "Name" to be used by the CLI.
#define REGISTER_STRATEGY(ClassName, Name) \
    namespace { \
        struct ClassName##Registrar { \
            ClassName##Registrar() { \
                StrategyFactory::instance().register_strategy(Name, \
                    []() { return std::make_unique<ClassName>(); }); \
            } \
        }; \
        static ClassName##Registrar g_##ClassName##_registrar; \
    }

} // namespace sentio
```

## 📄 **FILE 7 of 92**: include/sentio/binio.hpp

**File Information**:
- **Path**: `include/sentio/binio.hpp`

- **Size**: 68 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <cstdio>
#include <vector>
#include <string>
#include "core.hpp"

namespace sentio {

inline void save_bin(const std::string& path, const std::vector<Bar>& v) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return;
    
    uint64_t n = v.size();
    std::fwrite(&n, sizeof(n), 1, fp);
    
    for (const auto& bar : v) {
        // Write string length and data
        uint32_t str_len = bar.ts_utc.length();
        std::fwrite(&str_len, sizeof(str_len), 1, fp);
        std::fwrite(bar.ts_utc.c_str(), 1, str_len, fp);
        
        // Write other fields
        std::fwrite(&bar.ts_nyt_epoch, sizeof(bar.ts_nyt_epoch), 1, fp);
        std::fwrite(&bar.open, sizeof(bar.open), 1, fp);
        std::fwrite(&bar.high, sizeof(bar.high), 1, fp);
        std::fwrite(&bar.low, sizeof(bar.low), 1, fp);
        std::fwrite(&bar.close, sizeof(bar.close), 1, fp);
        std::fwrite(&bar.volume, sizeof(bar.volume), 1, fp);
    }
    std::fclose(fp);
}

inline std::vector<Bar> load_bin(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return {};
    
    uint64_t n = 0; 
    std::fread(&n, sizeof(n), 1, fp);
    std::vector<Bar> v;
    v.reserve(n);
    
    for (uint64_t i = 0; i < n; ++i) {
        Bar bar;
        
        // Read string length and data
        uint32_t str_len = 0;
        std::fread(&str_len, sizeof(str_len), 1, fp);
        if (str_len > 0) {
            std::vector<char> str_data(str_len);
            std::fread(str_data.data(), 1, str_len, fp);
            bar.ts_utc = std::string(str_data.data(), str_len);
        }
        
        // Read other fields
        std::fread(&bar.ts_nyt_epoch, sizeof(bar.ts_nyt_epoch), 1, fp);
        std::fread(&bar.open, sizeof(bar.open), 1, fp);
        std::fread(&bar.high, sizeof(bar.high), 1, fp);
        std::fread(&bar.low, sizeof(bar.low), 1, fp);
        std::fread(&bar.close, sizeof(bar.close), 1, fp);
        std::fread(&bar.volume, sizeof(bar.volume), 1, fp);
        
        v.push_back(bar);
    }
    std::fclose(fp);
    return v;
}

} // namespace sentio
```

## 📄 **FILE 8 of 92**: include/sentio/bo.hpp

**File Information**:
- **Path**: `include/sentio/bo.hpp`

- **Size**: 384 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
// bo.hpp — minimal, solid C++20 Bayesian Optimization (GP + EI)
// No external deps. Deterministic. Safe. Box bounds + integers + batch ask().

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <functional>
#include <optional>
#include <limits>
#include <cmath>
#include <cassert>
#include <chrono>
#include <string>

// ----------------------------- Utilities -----------------------------
struct BOBounds {
  std::vector<double> lo, hi; // size = D
  bool valid() const { return !lo.empty() && lo.size()==hi.size(); }
  int dim() const { return (int)lo.size(); }
  inline double clamp_unit(double u, int d) const {
    return std::min(1.0, std::max(0.0, u));
  }
  inline double to_real(double u, int d) const {
    u = clamp_unit(u,d);
    return lo[d] + u * (hi[d] - lo[d]);
  }
  inline double to_unit(double x, int d) const {
    const double L = lo[d], H = hi[d];
    if (H<=L) return 0.0;
    double u = (x - L) / (H - L);
    return std::min(1.0, std::max(0.0, u));
  }
};

// Parameter kinds (continuous or integer). If INT, we round to nearest.
enum class ParamKind : uint8_t { CONT, INT };

struct BOOpts {
  int init_design = 16;          // initial random/space-filling points
  int cand_pool   = 2048;        // candidate samples for EI maximization
  int batch_q     = 1;           // q-EI via constant liar
  double jitter   = 1e-10;       // numerical jitter for Cholesky
  double ei_xi    = 0.01;        // EI exploration parameter
  bool  maximize  = true;        // true: maximize objective (default)
  uint64_t seed   = 42;          // deterministic rand seed
  bool  verbose   = false;
};

// ----------------------------- Tiny matrix helpers -----------------------------
// Row-major dense matrix with minimal ops (just what we need for GP Cholesky).
struct Mat {
  int n=0, m=0;                  // n rows, m cols
  std::vector<double> a;         // size n*m
  Mat()=default;
  Mat(int n_, int m_) : n(n_), m(m_), a((size_t)n_*(size_t)m_, 0.0) {}
  inline double& operator()(int i, int j){ return a[(size_t)i*(size_t)m + j]; }
  inline double  operator()(int i, int j) const { return a[(size_t)i*(size_t)m + j]; }
};

// Cholesky decomposition A = L L^T in-place into L (lower). Returns false if fails.
inline bool cholesky(Mat& A, double jitter=1e-10){
  // A must be square, symmetric, positive definite (we'll add jitter on diag).
  assert(A.n == A.m);
  const int n = A.n;
  for (int i=0;i<n;i++) A(i,i) += jitter;

  for (int i=0;i<n;i++){
    for (int j=0;j<=i;j++){
      double sum = A(i,j);
      for (int k=0;k<j;k++) sum -= A(i,k)*A(j,k);
      if (i==j){
        if (sum <= 0.0) return false;
        A(i,j) = std::sqrt(sum);
      } else {
        A(i,j) = sum / A(j,j);
      }
    }
    for (int j=i+1;j<n;j++) A(i,j)=0.0; // zero upper for clarity
  }
  return true;
}

// Solve L y = b (forward)
inline void trisolve_lower(const Mat& L, std::vector<double>& y){
  const int n=L.n;
  for (int i=0;i<n;i++){
    double s = y[i];
    for (int k=0;k<i;k++) s -= L(i,k)*y[k];
    y[i] = s / L(i,i);
  }
}
// Solve L^T x = y (backward)
inline void trisolve_upperT(const Mat& L, std::vector<double>& x){
  const int n=L.n;
  for (int i=n-1;i>=0;i--){
    double s = x[i];
    for (int k=i+1;k<n;k++) s -= L(k,i)*x[k];
    x[i] = s / L(i,i);
  }
}

// ----------------------------- Gaussian Process (RBF-ARD) -----------------------------
struct GP {
  // Hyperparams (on unit cube): signal^2, noise^2, lengthscales (per-dim)
  double sigma_f2 = 1.0;
  double sigma_n2 = 1e-6;
  std::vector<double> ell; // size D, lengthscales

  // Data (unit cube)
  std::vector<std::vector<double>> X; // N x D
  std::vector<double> y;              // N (centered)
  double y_mean = 0.0;

  // Factorization
  Mat L;                 // Cholesky of K = Kf + sigma_n2 I
  std::vector<double> alpha; // (K)^-1 y

  static inline double sqdist_ard(const std::vector<double>& a, const std::vector<double>& b,
                                  const std::vector<double>& ell){
    double s=0.0;
    for (size_t d=0; d<ell.size(); ++d){
      const double z = (a[d]-b[d]) / std::max(1e-12, ell[d]);
      s += z*z;
    }
    return s;
  }

  inline double kf(const std::vector<double>& a, const std::vector<double>& b) const {
    const double s2 = sqdist_ard(a,b,ell);
    return sigma_f2 * std::exp(-0.5 * s2);
  }

  void fit(const std::vector<std::vector<double>>& X_unit,
           const std::vector<double>& y_raw,
           double jitter=1e-10)
  {
    X = X_unit;
    const int N = (int)X.size();
    y_mean = 0.0;
    for (double v: y_raw) y_mean += v;
    y_mean /= std::max(1, N);
    y.resize(N);
    for (int i=0;i<N;i++) y[i] = y_raw[i] - y_mean;

    // Build K
    Mat K(N,N);
    for (int i=0;i<N;i++){
      for (int j=0;j<=i;j++){
        double kij = kf(X[i], X[j]);
        if (i==j) kij += sigma_n2;
        K(i,j) = K(j,i) = kij;
      }
    }
    // Chol
    L = K;
    if (!cholesky(L, jitter)) {
      // increase jitter progressively if needed
      double j = std::max(jitter, 1e-12);
      bool ok=false;
      for (int t=0;t<6 && !ok; ++t) { L = K; ok = cholesky(L, j); j *= 10.0; }
      if (!ok) throw std::runtime_error("GP Cholesky failed (matrix not PD)");
    }
    // alpha = K^{-1} y  via L
    alpha = y;
    trisolve_lower(L, alpha);
    trisolve_upperT(L, alpha);
  }

  // Predictive mean/variance at x (unit cube)
  inline std::pair<double,double> predict(const std::vector<double>& x) const {
    const int N = (int)X.size();
    std::vector<double> k(N);
    for (int i=0;i<N;i++) k[i] = kf(x, X[i]);
    // mean = k^T alpha + y_mean
    double mu = std::inner_product(k.begin(), k.end(), alpha.begin(), 0.0) + y_mean;
    // v = L^{-1} k
    std::vector<double> v = k;
    trisolve_lower(L, v);
    double kxx = sigma_f2; // k(x,x) for RBF with same params is sigma_f2
    double var = kxx - std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    if (var < 1e-18) var = 1e-18;
    return {mu, var};
  }
};

// ----------------------------- Random/Sampling helpers -----------------------------
struct RNG {
  std::mt19937_64 g;
  explicit RNG(uint64_t seed): g(seed) {}
  double uni(){ return std::uniform_real_distribution<double>(0.0,1.0)(g); }
  int randint(int lo, int hi){ return std::uniform_int_distribution<int>(lo,hi)(g); }
};

// Jittered Latin hypercube on unit cube
inline std::vector<std::vector<double>> latin_hypercube(int n, int D, RNG& rng){
  std::vector<std::vector<double>> X(n, std::vector<double>(D,0.0));
  for (int d=0; d<D; ++d){
    std::vector<double> slots(n);
    for (int i=0;i<n;i++) slots[i] = (i + rng.uni()) / n;
    std::shuffle(slots.begin(), slots.end(), rng.g);
    for (int i=0;i<n;i++) X[i][d] = std::min(1.0, std::max(0.0, slots[i]));
  }
  return X;
}

// Round integer dims to nearest feasible grid-point after scaling
inline void apply_kinds_inplace(std::vector<double>& u, const std::vector<ParamKind>& kinds,
                                const BOBounds& B){
  for (int d=0; d<(int)u.size(); ++d){
    if (kinds[d] == ParamKind::INT){
      // snap in real space to integer, then rescale back to unit
      const double x = B.to_real(u[d], d);
      const double xi = std::round(x);
      u[d] = B.to_unit(xi, d);
    }
  }
}

// ----------------------------- Acquisition: Expected Improvement -----------------------------
inline double norm_pdf(double z){ static const double inv_sqrt2pi = 0.3989422804014327; return inv_sqrt2pi*std::exp(-0.5*z*z); }
inline double norm_cdf(double z){
  return 0.5 * std::erfc(-z/std::sqrt(2.0));
}
// EI for maximization: EI = (mu - best - xi) Phi(z) + sigma phi(z), z = (mu - best - xi)/sigma
inline double expected_improvement(double mu, double var, double best, double xi){
  const double sigma = std::sqrt(std::max(1e-18, var));
  const double diff  = mu - best - xi;
  if (sigma <= 1e-12) return std::max(0.0, diff);
  const double z = diff / sigma;
  return diff * norm_cdf(z) + sigma * norm_pdf(z);
}

// ----------------------------- Bayesian Optimizer -----------------------------
struct BO {
  BOBounds bounds;
  std::vector<ParamKind> kinds; // size D
  BOOpts opt;

  // data (real-space for API; internally store unit-cube for GP)
  std::vector<std::vector<double>> X_real; // N x D
  std::vector<std::vector<double>> X;      // N x D (unit)
  std::vector<double> y;                   // N
  GP gp;
  RNG rng;

  BO(const BOBounds& B, const std::vector<ParamKind>& kinds_, BOOpts o)
  : bounds(B), kinds(kinds_), opt(o), gp(), rng(o.seed)
  {
    assert(bounds.valid());
    assert((int)kinds.size() == bounds.dim());
    // init default GP hyperparams
    gp.ell.assign(bounds.dim(), 0.2); // medium lengthscale on unit cube
    gp.sigma_f2 = 1.0;
    gp.sigma_n2 = 1e-6;
  }

  int dim() const { return bounds.dim(); }
  int size() const { return (int)X.size(); }

  void clear(){
    X_real.clear(); X.clear(); y.clear();
  }

  // Append observation
  void tell(const std::vector<double>& xr, double val){
    assert((int)xr.size() == dim());
    X_real.push_back(xr);
    std::vector<double> u(dim());
    for (int d=0; d<dim(); ++d) u[d] = bounds.to_unit(xr[d], d);
    // snap integers
    apply_kinds_inplace(u, kinds, bounds);
    X.push_back(u);
    y.push_back(val);
  }

  // Fit GP (simple hyperparam heuristics; robust & fast)
  void fit(){
    if (X.empty()) return;
    // set GP hyperparams from y stats
    double m=0, s2=0;
    for (double v: y) m += v; m /= (double)y.size();
    for (double v: y) s2 += (v-m)*(v-m);
    s2 /= std::max(1, (int)y.size()-1);
    gp.sigma_f2 = std::max(1e-12, s2);
    gp.sigma_n2 = std::max(1e-9 * gp.sigma_f2, 1e-10);
    // modest ARD scaling: initialize lengthscales to 0.2; (optionally: scale by output sensitivity)
    for (double& e: gp.ell) e = 0.2;
    gp.fit(X, y, opt.jitter);
  }

  // Generate initial design if dataset is empty/small
  void ensure_init_design(){
    const int need = std::max(0, opt.init_design - (int)X.size());
    if (need <= 0) return;
    auto U = latin_hypercube(need, dim(), rng);
    for (auto& u : U){
      apply_kinds_inplace(u, kinds, bounds);
      std::vector<double> xr(dim());
      for (int d=0; d<dim(); ++d) xr[d] = bounds.to_real(u[d], d);
      // placeholder y (user will evaluate and call tell)
      X.push_back(u);
      X_real.push_back(xr);
      y.push_back(std::numeric_limits<double>::quiet_NaN()); // mark unevaluated
    }
  }

  // Ask for q new locations (real-space). Uses constant liar on current model (no new evals yet).
  std::vector<std::vector<double>> ask(int q=1){
    if (q<=0) return {};
    // If we have NaNs from ensure_init_design, return those first (ask-user-to-evaluate)
    std::vector<std::vector<double>> out;
    for (int i=0;i<(int)X.size() && (int)out.size()<q; ++i){
      if (!std::isfinite(y[i])) out.push_back(X_real[i]);
    }
    if ((int)out.size() == q) return out;

    // Ensure we can build a GP on finished points
    // Build filtered dataset of (finite) y's
    std::vector<std::vector<double>> Xf;
    std::vector<double> yf;
    Xf.reserve(X.size()); yf.reserve(y.size());
    for (int i=0;i<(int)X.size(); ++i){
      if (std::isfinite(y[i])) { Xf.push_back(X[i]); yf.push_back(y[i]); }
    }
    if (Xf.size() >= 2) {
      gp.fit(Xf, yf, opt.jitter);
    } else {
      // not enough data to fit GP: just random suggest
      out = random_candidates(q);
      return out;
    }

    const double y_best = opt.maximize ? *std::max_element(yf.begin(), yf.end())
                                       : *std::min_element(yf.begin(), yf.end());

    // batch selection with constant liar (for maximization, lie = y_best)
    std::vector<std::vector<double>> X_aug = Xf;
    std::vector<double> y_aug = yf;

    for (int pick=0; pick<q; ++pick){
      // pool
      double best_ei = -1.0;
      std::vector<double> best_u(dim());
      for (int c=0; c<opt.cand_pool; ++c){
        std::vector<double> u(dim());
        for (int d=0; d<dim(); ++d) u[d] = rng.uni();
        apply_kinds_inplace(u, kinds, bounds);
        // score EI on augmented model (approximate by reusing gp; small bias is fine)
        // NOTE: we approximate: use current gp (no retrain) — fast & works well in practice
        auto [mu, var] = gp.predict(u);
        double ei = opt.maximize ? expected_improvement(mu, var, y_best, opt.ei_xi)
                                 : expected_improvement(-mu, var, -y_best, opt.ei_xi);
        if (ei > best_ei){ best_ei = ei; best_u = u; }
      }
      // map to real, append
      std::vector<double> xr(dim());
      for (int d=0; d<dim(); ++d) xr[d] = bounds.to_real(best_u[d], d);
      out.push_back(xr);

      // constant liar update (no refit for speed; optional: refit small)
      X_aug.push_back(best_u);
      y_aug.push_back(y_best); // lie
      // (Optionally refit gp with augmented data each pick for sharper q-EI; omitted for speed)
    }
    return out;
  }

  // Helper: random suggestions on real-space
  std::vector<std::vector<double>> random_candidates(int q){
    std::vector<std::vector<double>> out;
    out.reserve(q);
    for (int i=0;i<q;++i){
      std::vector<double> u(dim());
      for (int d=0; d<dim(); ++d) u[d] = rng.uni();
      apply_kinds_inplace(u, kinds, bounds);
      std::vector<double> xr(dim());
      for (int d=0; d<dim(); ++d) xr[d] = bounds.to_real(u[d], d);
      out.push_back(xr);
    }
    return out;
  }
};
```

## 📄 **FILE 9 of 92**: include/sentio/bollinger.hpp

**File Information**:
- **Path**: `include/sentio/bollinger.hpp`

- **Size**: 22 lines
- **Modified**: 2025-09-05 13:23:38

- **Type**: .hpp

```text
#pragma once
#include "rolling_stats.hpp"

namespace sentio {

struct Bollinger {
  RollingMeanVar mv;
  double k;
  double eps; // volatility floor to avoid zero-width bands

  Bollinger(int w, double k_=2.0, double eps_=1e-9) : mv(w), k(k_), eps(eps_) {}

  inline void step(double close, double& mid, double& lo, double& hi, double& sd_out){
    auto [m, var] = mv.push(close);
    double sd = std::sqrt(std::max(var, 0.0));
    if (sd < eps) sd = eps;         // <- floor
    mid = m; lo = m - k*sd; hi = m + k*sd;
    sd_out = sd;
  }
};

} // namespace sentio
```

## 📄 **FILE 10 of 92**: include/sentio/calendar_seed.hpp

**File Information**:
- **Path**: `include/sentio/calendar_seed.hpp`

- **Size**: 30 lines
- **Modified**: 2025-09-05 14:04:17

- **Type**: .hpp

```text
// calendar_seed.hpp
#pragma once
#include "rth_calendar.hpp"

namespace sentio {

inline TradingCalendar make_default_nyse_calendar() {
  TradingCalendar c;

  // Full-day holidays (partial sample; fill your range robustly)
  // 2022: New Year (obs 2021-12-31), MLK 2022-01-17, Presidents 02-21,
  // Good Friday 04-15, Memorial 05-30, Juneteenth (obs 06-20), Independence 07-04,
  // Labor 09-05, Thanksgiving 11-24, Christmas (obs 2022-12-26)
  c.full_holidays.insert(20220117);
  c.full_holidays.insert(20220221);
  c.full_holidays.insert(20220415);
  c.full_holidays.insert(20220530);
  c.full_holidays.insert(20220620);
  c.full_holidays.insert(20220704);
  c.full_holidays.insert(20220905);
  c.full_holidays.insert(20221124);
  c.full_holidays.insert(20221226);

  // Early closes (sample): Black Friday 2022-11-25 @ 13:00 ET
  c.early_close_sec.emplace(20221125, 13*3600);

  return c;
}

} // namespace sentio
```

## 📄 **FILE 11 of 92**: include/sentio/core.hpp

**File Information**:
- **Path**: `include/sentio/core.hpp`

- **Size**: 69 lines
- **Modified**: 2025-09-05 09:44:09

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath> // For std::sqrt

namespace sentio {

struct Bar {
    std::string ts_utc;
    int64_t ts_nyt_epoch;
    double open, high, low, close;
    uint64_t volume;
};

// **MODIFIED**: This struct now holds a vector of Positions, indexed by symbol ID for performance.
struct Position { 
    double qty = 0.0; 
    double avg_price = 0.0; 
};

struct Portfolio {
    double cash = 100000.0;
    std::vector<Position> positions; // Indexed by symbol ID

    Portfolio() = default;
    explicit Portfolio(size_t num_symbols) : positions(num_symbols) {}
};

// **MODIFIED**: Vector-based functions are now the primary way to manage the portfolio.
inline void apply_fill(Portfolio& pf, int sid, double qty_delta, double price) {
    if (sid < 0 || static_cast<size_t>(sid) >= pf.positions.size()) {
        return; // Invalid symbol ID
    }
    
    pf.cash -= qty_delta * price;
    auto& pos = pf.positions[sid];
    
    double new_qty = pos.qty + qty_delta;
    if (std::abs(new_qty) < 1e-9) { // Position is closed
        pos.qty = 0.0;
        pos.avg_price = 0.0;
    } else {
        if (pos.qty * new_qty >= 0) { // Increasing position or opening a new one
            pos.avg_price = (pos.avg_price * pos.qty + price * qty_delta) / new_qty;
        }
        // If flipping from long to short or vice-versa, the new avg_price is just the fill price.
        else if (pos.qty * qty_delta < 0) {
             pos.avg_price = price;
        }
        pos.qty = new_qty;
    }
}

inline double equity_mark_to_market(const Portfolio& pf, const std::vector<double>& last_prices) {
    double eq = pf.cash;
    for (size_t sid = 0; sid < pf.positions.size(); ++sid) {
        if (std::abs(pf.positions[sid].qty) > 0.0 && sid < last_prices.size()) {
            eq += pf.positions[sid].qty * last_prices[sid];
        }
    }
    return eq;
}

// **REMOVED**: Old, simplistic Direction and StratSignal types are now deprecated.

} // namespace sentio
```

## 📄 **FILE 12 of 92**: include/sentio/cost_model.hpp

**File Information**:
- **Path**: `include/sentio/cost_model.hpp`

- **Size**: 120 lines
- **Modified**: 2025-09-05 09:44:09

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <cmath>

namespace sentio {

// Alpaca Trading Cost Model
struct AlpacaCostModel {
  // Commission structure (Alpaca is commission-free for stocks/ETFs)
  static constexpr double commission_per_share = 0.0;
  static constexpr double min_commission = 0.0;
  
  // SEC fees (for sells only)
  static constexpr double sec_fee_rate = 0.0000278; // $0.0278 per $1000 of principal
  
  // FINRA Trading Activity Fee (TAF) - for sells only
  static constexpr double taf_per_share = 0.000145; // $0.000145 per share (max $7.27 per trade)
  static constexpr double taf_max_per_trade = 7.27;
  
  // Slippage model based on market impact
  struct SlippageParams {
    double base_slippage_bps = 1.0;    // Base 1 bps slippage
    double volume_impact_factor = 0.5;  // Additional slippage based on volume
    double volatility_factor = 0.3;     // Additional slippage based on volatility
    double max_slippage_bps = 10.0;     // Cap at 10 bps
  };
  
  static SlippageParams default_slippage;
  
  // Calculate total transaction costs for a trade
  static double calculate_fees([[maybe_unused]] const std::string& symbol, 
                              double quantity, 
                              double price, 
                              bool is_sell) {
    double notional = std::abs(quantity) * price;
    double total_fees = 0.0;
    
    // Commission (free for Alpaca)
    total_fees += commission_per_share * std::abs(quantity);
    total_fees = std::max(total_fees, min_commission);
    
    if (is_sell) {
      // SEC fees (sells only)
      total_fees += notional * sec_fee_rate;
      
      // FINRA TAF (sells only)
      double taf = std::abs(quantity) * taf_per_share;
      total_fees += std::min(taf, taf_max_per_trade);
    }
    
    return total_fees;
  }
  
  // Calculate slippage based on trade characteristics
  static double calculate_slippage_bps(double quantity,
                                      double price, 
                                      double avg_volume,
                                      double volatility,
                                      const SlippageParams& params = default_slippage) {
    double notional = std::abs(quantity) * price;
    
    // Base slippage
    double slippage_bps = params.base_slippage_bps;
    
    // Volume impact (higher for larger trades relative to average volume)
    if (avg_volume > 0) {
      double volume_ratio = notional / (avg_volume * price);
      slippage_bps += params.volume_impact_factor * std::sqrt(volume_ratio) * 100; // Convert to bps
    }
    
    // Volatility impact
    slippage_bps += params.volatility_factor * volatility * 10000; // Convert annual vol to bps
    
    // Cap the slippage
    return std::min(slippage_bps, params.max_slippage_bps);
  }
  
  // Apply slippage to execution price
  static double apply_slippage(double market_price, 
                              double slippage_bps, 
                              bool is_buy) {
    double slippage_factor = slippage_bps / 10000.0; // Convert bps to decimal
    
    if (is_buy) {
      return market_price * (1.0 + slippage_factor); // Pay more when buying
    } else {
      return market_price * (1.0 - slippage_factor); // Receive less when selling
    }
  }
  
  // Complete cost calculation including fees and slippage
  static std::pair<double, double> calculate_total_costs(
      const std::string& symbol,
      double quantity,
      double market_price,
      double avg_volume = 1000000, // Default 1M average volume
      double volatility = 0.20,    // Default 20% annual volatility
      const SlippageParams& params = default_slippage) {
    
    bool is_sell = quantity < 0;
    bool is_buy = quantity > 0;
    
    // Calculate fees
    double fees = calculate_fees(symbol, quantity, market_price, is_sell);
    
    // Calculate slippage
    double slippage_bps = calculate_slippage_bps(quantity, market_price, avg_volume, volatility, params);
    double execution_price = apply_slippage(market_price, slippage_bps, is_buy);
    
    // Slippage cost (difference from market price)
    double slippage_cost = std::abs(quantity) * std::abs(execution_price - market_price);
    
    return {fees, slippage_cost};
  }
};

// Static member definition
inline AlpacaCostModel::SlippageParams AlpacaCostModel::default_slippage = {};

} // namespace sentio
```

## 📄 **FILE 13 of 92**: include/sentio/csv_loader.hpp

**File Information**:
- **Path**: `include/sentio/csv_loader.hpp`

- **Size**: 8 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include <string>

namespace sentio {
bool load_csv(const std::string& path, std::vector<Bar>& out);
} // namespace sentio


```

## 📄 **FILE 14 of 92**: include/sentio/data_resolver.hpp

**File Information**:
- **Path**: `include/sentio/data_resolver.hpp`

- **Size**: 44 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <filesystem>
#include <cstdlib>

namespace sentio {
enum class TickerFamily { Qqq, Bitcoin, Tesla };

inline const char** family_symbols(TickerFamily f, int& n) {
  static const char* QQQ[] = {"QQQ","TQQQ","SQQQ","PSQ"};
  static const char* BTC[] = {"BTCUSD","ETHUSD"};
  static const char* TSLA[]= {"TSLA","TSLQ"};
  switch (f) {
    case TickerFamily::Qqq: n=4; return QQQ;
    case TickerFamily::Bitcoin: n=2; return BTC;
    case TickerFamily::Tesla: n=2; return TSLA;
  }
  n=0; return nullptr;
}

inline std::string resolve_csv(const std::string& symbol,
                               const std::string& equities_root="data/equities",
                               const std::string& crypto_root="data/crypto") {
  namespace fs = std::filesystem;
  std::string up = symbol; for (auto& c: up) c = ::toupper(c);
  auto is_crypto = (up=="BTC"||up=="BTCUSD"||up=="ETH"||up=="ETHUSD");

  const char* env_root = std::getenv("SENTIO_DATA_ROOT");
  const char* env_suffix = std::getenv("SENTIO_DATA_SUFFIX");
  std::string base = env_root ? std::string(env_root) : (is_crypto ? crypto_root : equities_root);
  std::string suffix = env_suffix ? std::string(env_suffix) : std::string("");

  // Prefer suffixed file in base, then non-suffixed, then fallback to default roots
  std::string cand1 = base + "/" + up + suffix + ".csv";
  if (fs::exists(cand1)) return cand1;
  std::string cand2 = base + "/" + up + ".csv";
  if (fs::exists(cand2)) return cand2;
  std::string fallback_base = (is_crypto ? crypto_root : equities_root);
  std::string cand3 = fallback_base + "/" + up + suffix + ".csv";
  if (fs::exists(cand3)) return cand3;
  return fallback_base + "/" + up + ".csv";
}
} // namespace sentio


```

## 📄 **FILE 15 of 92**: include/sentio/day_index.hpp

**File Information**:
- **Path**: `include/sentio/day_index.hpp`

- **Size**: 26 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include <vector>
#include <string_view>

namespace sentio {

// From minute bars with RFC3339 timestamps, return indices of first bar per day
inline std::vector<int> day_start_indices(const std::vector<Bar>& bars) {
    std::vector<int> starts;
    starts.reserve(bars.size() / 300 + 2);
    std::string last_day;
    for (int i=0; i<(int)bars.size(); ++i) {
        std::string_view ts(bars[i].ts_utc);
        if (ts.size() < 10) continue;
        std::string cur{ts.substr(0,10)};
        if (i == 0 || cur != last_day) {
            starts.push_back(i);
            last_day = std::move(cur);
        }
    }
    return starts;
}

} // namespace sentio


```

## 📄 **FILE 16 of 92**: include/sentio/exec_types.hpp

**File Information**:
- **Path**: `include/sentio/exec_types.hpp`

- **Size**: 15 lines
- **Modified**: 2025-09-05 14:08:05

- **Type**: .hpp

```text
// exec_types.hpp
#pragma once
#include <string>

namespace sentio {

struct ExecutionIntent {
  std::string base_symbol;     // e.g., "QQQ"
  std::string instrument;      // e.g., "TQQQ" or "SQQQ" or "QQQ"
  double      qty = 0.0;
  double      leverage = 1.0;  // informational; actual product carries leverage
  double      score = 0.0;     // signal strength
};

} // namespace sentio
```

## 📄 **FILE 17 of 92**: include/sentio/feature_health.hpp

**File Information**:
- **Path**: `include/sentio/feature_health.hpp`

- **Size**: 31 lines
- **Modified**: 2025-09-05 17:01:01

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sentio {

struct FeatureIssue {
  std::int64_t ts_utc{};
  std::string  kind;      // e.g., "NaN", "Gap", "Backwards_TS"
  std::string  detail;
};

struct FeatureHealthReport {
  std::vector<FeatureIssue> issues;
  bool ok() const { return issues.empty(); }
};

struct FeatureHealthCfg {
  // bar spacing in seconds (e.g., 60 for 1m). 0 = skip spacing checks.
  int expected_spacing_sec{60};
  bool check_nan{true};
  bool check_monotonic_time{true};
};

struct PricePoint { std::int64_t ts_utc{}; double close{}; };

FeatureHealthReport check_feature_health(const std::vector<PricePoint>& series,
                                         const FeatureHealthCfg& cfg);

} // namespace sentio

```

## 📄 **FILE 18 of 92**: include/sentio/indicators.hpp

**File Information**:
- **Path**: `include/sentio/indicators.hpp`

- **Size**: 52 lines
- **Modified**: 2025-09-05 16:34:30

- **Type**: .hpp

```text
#pragma once
#include <deque>
#include <cmath>
#include <limits>

namespace sentio {

struct SMA {
  int n{0};
  double sum{0.0};
  std::deque<double> q;
  explicit SMA(int n_) : n(n_) {}
  void reset(){ sum=0; q.clear(); }
  void push(double x){
    if (!std::isfinite(x)) { reset(); return; }
    q.push_back(x); sum += x;
    if ((int)q.size() > n) { sum -= q.front(); q.pop_front(); }
  }
  bool ready() const { return (int)q.size() == n; }
  double value() const { return ready() ? sum / n : std::numeric_limits<double>::quiet_NaN(); }
};

struct RSI {
  int n{14};
  int warm{0};
  double avgGain{0}, avgLoss{0}, prev{NAN};
  explicit RSI(int n_=14):n(n_),warm(0),avgGain(0),avgLoss(0),prev(NAN){}
  void reset(){ warm=0; avgGain=avgLoss=0; prev=NAN; }
  void push(double close){
    if (!std::isfinite(close)) { reset(); return; }
    if (!std::isfinite(prev)) { prev = close; return; }
    double delta = close - prev; prev = close;
    double gain = delta > 0 ? delta : 0.0;
    double loss = delta < 0 ? -delta : 0.0;
    if (warm < n) {
      avgGain += gain; avgLoss += loss; ++warm;
      if (warm==n) { avgGain/=n; avgLoss/=n; }
    } else {
      avgGain = (avgGain*(n-1) + gain)/n;
      avgLoss = (avgLoss*(n-1) + loss)/n;
    }
  }
  bool ready() const { return warm >= n; }
  double value() const {
    if (!ready()) return std::numeric_limits<double>::quiet_NaN();
    if (avgLoss == 0) return 100.0;
    double rs = avgGain/avgLoss;
    return 100.0 - (100.0/(1.0+rs));
  }
};

} // namespace sentio

```

## 📄 **FILE 19 of 92**: include/sentio/metrics.hpp

**File Information**:
- **Path**: `include/sentio/metrics.hpp`

- **Size**: 123 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <utility>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <string>

namespace sentio {
struct RunSummary {
  int bars{}, trades{};
  double ret_total{}, ret_ann{}, vol_ann{}, sharpe{}, mdd{};
  double monthly_proj{}, daily_trades{};
};

inline RunSummary compute_metrics(const std::vector<std::pair<std::string,double>>& daily_equity,
                                  int fills_count) {
  RunSummary s{};
  if (daily_equity.size() < 2) return s;
  s.bars = (int)daily_equity.size();
  std::vector<double> rets; rets.reserve(daily_equity.size()-1);
  for (size_t i=1;i<daily_equity.size();++i) {
    double e0=daily_equity[i-1].second, e1=daily_equity[i].second;
    rets.push_back(e0>0 ? (e1/e0-1.0) : 0.0);
  }
  double mean = std::accumulate(rets.begin(),rets.end(),0.0)/rets.size();
  double var=0.0; for (double r:rets){ double d=r-mean; var+=d*d; } var/=std::max<size_t>(1,rets.size());
  double sd = std::sqrt(var);
  s.ret_ann = mean * 252.0;
  s.vol_ann = sd * std::sqrt(252.0);
  s.sharpe  = (s.vol_ann>1e-12)? s.ret_ann/s.vol_ann : 0.0;
  double e0 = daily_equity.front().second, e1 = daily_equity.back().second;
  s.ret_total = e0>0 ? (e1/e0-1.0) : 0.0;
  s.monthly_proj = std::pow(1.0 + s.ret_ann, 1.0/12.0) - 1.0;
  s.trades = fills_count;
  s.daily_trades = (s.bars>0) ? (double)s.trades / (double)s.bars : 0.0;
  s.mdd = 0.0; // TODO: compute drawdown if you track running peaks
  return s;
}

// Day-aware metrics computed from bar-level equity series by compressing to day closes
inline RunSummary compute_metrics_day_aware(
    const std::vector<std::pair<std::string,double>>& equity_steps,
    int fills_count) {
  RunSummary s{};
  if (equity_steps.size() < 2) {
    s.trades = fills_count;
    s.daily_trades = 0.0;
    return s;
  }

  // Compress to day closes using ts_utc prefix YYYY-MM-DD
  std::vector<double> day_close;
  day_close.reserve(equity_steps.size() / 300 + 2);

  std::string last_day = equity_steps.front().first.size() >= 10
                         ? equity_steps.front().first.substr(0,10)
                         : std::string{};
  double cur = equity_steps.front().second;

  for (size_t i = 1; i < equity_steps.size(); ++i) {
    const auto& ts = equity_steps[i].first;
    const std::string day = ts.size() >= 10 ? ts.substr(0,10) : last_day;
    if (day != last_day) {
      day_close.push_back(cur); // close of previous day
      last_day = day;
    }
    cur = equity_steps[i].second; // latest equity for current day
  }
  day_close.push_back(cur); // close of final day

  const int D = static_cast<int>(day_close.size());
  s.bars = D;

  if (D < 2) {
    s.trades = fills_count;
    s.daily_trades = 0.0;
    s.ret_total = 0.0; s.ret_ann = 0.0; s.vol_ann = 0.0; s.sharpe = 0.0; s.mdd = 0.0;
    s.monthly_proj = 0.0;
    return s;
  }

  // Daily simple returns
  std::vector<double> r; r.reserve(D - 1);
  for (int i = 1; i < D; ++i) {
    double prev = day_close[i-1];
    double next = day_close[i];
    r.push_back(prev > 0.0 ? (next/prev - 1.0) : 0.0);
  }

  // Mean and variance
  double mean = 0.0; for (double x : r) mean += x; mean /= r.size();
  double var  = 0.0; for (double x : r) { double d = x - mean; var += d*d; } var /= r.size();
  double sd = std::sqrt(var);

  // Annualization on daily series
  s.vol_ann = sd * std::sqrt(252.0);
  double e0 = day_close.front(), e1 = day_close.back();
  s.ret_total = e0 > 0.0 ? (e1/e0 - 1.0) : 0.0;
  double years = (D - 1) / 252.0;
  s.ret_ann = (years > 0.0) ? (std::pow(1.0 + s.ret_total, 1.0/years) - 1.0) : 0.0;
  s.sharpe = (sd > 1e-12) ? (mean / sd) * std::sqrt(252.0) : 0.0;
  s.monthly_proj = std::pow(1.0 + s.ret_ann, 1.0/12.0) - 1.0;

  // Trades
  s.trades = fills_count;
  s.daily_trades = (D > 0) ? static_cast<double>(fills_count) / static_cast<double>(D) : 0.0;

  // Max drawdown on day closes
  double peak = day_close.front();
  double max_dd = 0.0;
  for (int i = 1; i < D; ++i) {
    peak = std::max(peak, day_close[i-1]);
    if (peak > 0.0) {
      double dd = (day_close[i] - peak) / peak; // negative when below peak
      if (dd < max_dd) max_dd = dd;
    }
  }
  s.mdd = -max_dd;
  return s;
}
} // namespace sentio


```

## 📄 **FILE 20 of 92**: include/sentio/of_index.hpp

**File Information**:
- **Path**: `include/sentio/of_index.hpp`

- **Size**: 36 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include "core.hpp"
#include "orderflow_types.hpp"

namespace sentio {

struct BarTickSpan { 
    int start=-1, end=-1; // [start,end) ticks for this bar
};

inline std::vector<BarTickSpan> build_tick_spans(const std::vector<Bar>& bars,
                                                 const std::vector<Tick>& ticks)
{
    const int N = (int)bars.size();
    const int M = (int)ticks.size();
    std::vector<BarTickSpan> span(N);

    int i = 0, k = 0;
    int cur_start = 0;

    // assume bars have strictly increasing ts; ticks nondecreasing
    for (; i < N; ++i) {
        const int64_t ts = bars[i].ts_nyt_epoch;
        // advance k until tick.ts > ts
        while (k < M && ticks[k].ts_nyt_epoch <= ts) ++k;
        span[i].start = cur_start;
        span[i].end   = k;        // [cur_start, k) are ticks for bar i
        cur_start = k;
    }
    return span;
}

} // namespace sentio
```

## 📄 **FILE 21 of 92**: include/sentio/of_precompute.hpp

**File Information**:
- **Path**: `include/sentio/of_precompute.hpp`

- **Size**: 26 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "of_index.hpp"

namespace sentio {

inline void precompute_mid_imb(const std::vector<Tick>& ticks,
                               std::vector<double>& mid,
                               std::vector<double>& imb)
{
    const int M = (int)ticks.size();
    mid.resize(M);
    imb.resize(M);
    for (int k=0; k<M; ++k) {
        double m = (ticks[k].bid_px + ticks[k].ask_px) * 0.5;
        double a = std::max(0.0, ticks[k].ask_sz);
        double b = std::max(0.0, ticks[k].bid_sz);
        double d = a + b;
        mid[k] = m;
        imb[k] = (d > 0.0) ? (a / d) : 0.5;   // neutral if zero depth
    }
}

} // namespace sentio
```

## 📄 **FILE 22 of 92**: include/sentio/optimizer.hpp

**File Information**:
- **Path**: `include/sentio/optimizer.hpp`

- **Size**: 147 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "sentio/core.hpp"
#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <random>

namespace sentio {

struct Parameter {
    std::string name;
    double min_value;
    double max_value;
    double current_value;
    
    Parameter(const std::string& n, double min_val, double max_val, double current_val)
        : name(n), min_value(min_val), max_value(max_val), current_value(current_val) {}
    
    Parameter(const std::string& n, double min_val, double max_val)
        : name(n), min_value(min_val), max_value(max_val), current_value(min_val) {}
};

struct OptimizationResult {
    std::unordered_map<std::string, double> parameters;
    RunResult metrics;
    double objective_value;
};

struct OptimizationConfig {
    std::string optimizer_type = "random";
    int max_trials = 30;
    std::string objective = "sharpe_ratio";
    double timeout_minutes = 15.0;
    bool verbose = true;
};

class ParameterOptimizer {
public:
    virtual ~ParameterOptimizer() = default;
    virtual std::vector<Parameter> get_parameter_space() = 0;
    virtual void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) = 0;
    virtual OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                                      const std::vector<Parameter>& param_space,
                                      const OptimizationConfig& config) = 0;
};

class RandomSearchOptimizer : public ParameterOptimizer {
private:
    std::mt19937 rng;
    
public:
    RandomSearchOptimizer(unsigned int seed = 42) : rng(seed) {}
    
    std::vector<Parameter> get_parameter_space() override;
    void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) override;
    OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                               const std::vector<Parameter>& param_space,
                               const OptimizationConfig& config) override;
};

class GridSearchOptimizer : public ParameterOptimizer {
public:
    std::vector<Parameter> get_parameter_space() override;
    void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) override;
    OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                              const std::vector<Parameter>& param_space,
                              const OptimizationConfig& config) override;
};

class BayesianOptimizer : public ParameterOptimizer {
public:
    std::vector<Parameter> get_parameter_space() override;
    void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) override;
    OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                              const std::vector<Parameter>& param_space,
                              const OptimizationConfig& config) override;
};

// Objective functions
using ObjectiveFunction = std::function<double(const RunResult&)>;

class ObjectiveFunctions {
public:
    static double sharpe_objective(const RunResult& summary) {
        return summary.sharpe_ratio;
    }
    
    static double calmar_objective(const RunResult& summary) {
        return summary.monthly_projected_return / std::max(0.01, summary.max_drawdown);
    }
    
    static double total_return_objective(const RunResult& summary) {
        return summary.total_return;
    }
    
    static double sortino_objective(const RunResult& summary) {
        // Simplified Sortino ratio (assuming no downside deviation calculation)
        return summary.sharpe_ratio;
    }
};

// Strategy-specific parameter creation functions
std::vector<Parameter> create_vwap_parameters();
std::vector<Parameter> create_momentum_parameters();
std::vector<Parameter> create_volatility_parameters();
std::vector<Parameter> create_bollinger_squeeze_parameters();
std::vector<Parameter> create_opening_range_parameters();
std::vector<Parameter> create_order_flow_scalping_parameters();
std::vector<Parameter> create_order_flow_imbalance_parameters();
std::vector<Parameter> create_market_making_parameters();
std::vector<Parameter> create_router_parameters();

std::vector<Parameter> create_parameters_for_strategy(const std::string& strategy_name);
std::vector<Parameter> create_full_parameter_space();

// Optimization utilities
class OptimizationEngine {
private:
    std::unique_ptr<ParameterOptimizer> optimizer;
    OptimizationConfig config;
    
public:
    OptimizationEngine(const std::string& optimizer_type = "random");
    
    void set_config(const OptimizationConfig& cfg) { config = cfg; }
    
    OptimizationResult run_optimization(const std::string& strategy_name,
                                      const std::function<double(const RunResult&)>& objective_func);
    
    double calculate_objective(const RunResult& summary) {
        if (config.objective == "sharpe_ratio") {
            return ObjectiveFunctions::sharpe_objective(summary);
        } else if (config.objective == "calmar_ratio") {
            return ObjectiveFunctions::calmar_objective(summary);
        } else if (config.objective == "total_return") {
            return ObjectiveFunctions::total_return_objective(summary);
        } else if (config.objective == "sortino_ratio") {
            return ObjectiveFunctions::sortino_objective(summary);
        }
        return ObjectiveFunctions::sharpe_objective(summary); // default
    }
};

} // namespace sentio
```

## 📄 **FILE 23 of 92**: include/sentio/orderflow_types.hpp

**File Information**:
- **Path**: `include/sentio/orderflow_types.hpp`

- **Size**: 34 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include "core.hpp"

namespace sentio {

struct Tick {
    int64_t ts_nyt_epoch;     // strictly nondecreasing
    double  bid_px, ask_px;
    double  bid_sz, ask_sz;   // L1 sizes (or synthetic from L2)
    // (Optional: trade prints, aggressor flags, etc.)
};

struct OFParams {
    // Signal
    double  min_imbalance = 0.65;     // (ask_sz / (ask_sz + bid_sz)) for long
    int     look_ticks    = 50;       // rolling window
    // Risk
    int     hold_max_ticks = 800;     // hard cap per trade
    double  tp_ticks       = 4.0;     // TP in ticks (half-spread units if you like)
    double  sl_ticks       = 4.0;     // SL in ticks
    // Execution
    double  tick_size      = 0.01;
};

struct Trade {
    int start_k=-1, end_k=-1;  // tick indices
    double entry_px=0, exit_px=0;
    int dir=0;                 // +1 long, -1 short
};

} // namespace sentio
```

## 📄 **FILE 24 of 92**: include/sentio/pnl_accounting.hpp

**File Information**:
- **Path**: `include/sentio/pnl_accounting.hpp`

- **Size**: 24 lines
- **Modified**: 2025-09-05 15:29:35

- **Type**: .hpp

```text
// pnl_accounting.hpp
#pragma once
#include "core.hpp"  // Use Bar from core.hpp
#include <string>
#include <stdexcept>
#include <unordered_map>

namespace sentio {

class PriceBook {
public:
  // instrument -> latest bar (or map<ts,bar> for full history)
  const Bar* get_latest(const std::string& instrument) const;
  
  // Additional helper methods
  void upsert_latest(const std::string& instrument, const Bar& b);
  bool has_instrument(const std::string& instrument) const;
  std::size_t size() const;
};

// Use the instrument actually traded
double last_trade_price(const PriceBook& book, const std::string& instrument);

} // namespace sentio
```

## 📄 **FILE 25 of 92**: include/sentio/polygon_client.hpp

**File Information**:
- **Path**: `include/sentio/polygon_client.hpp`

- **Size**: 30 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>

namespace sentio {
struct AggBar { long long ts_ms; double open, high, low, close, volume; };

struct AggsQuery {
  std::string symbol;
  int multiplier{1};
  std::string timespan{"day"}; // "day","hour","minute"
  std::string from, to;
  bool adjusted{true};
  std::string sort{"asc"};
  int limit{50000};
};

class PolygonClient {
public:
  explicit PolygonClient(std::string api_key);
  std::vector<AggBar> get_aggs_all(const AggsQuery& q, int max_pages=200);
  void write_csv(const std::string& out_path,const std::string& symbol,
                 const std::vector<AggBar>& bars, bool rth_only=false, bool exclude_holidays=false);

private:
  std::string api_key_;
  std::string get_(const std::string& url);
};
} // namespace sentio


```

## 📄 **FILE 26 of 92**: include/sentio/polygon_ingest.hpp

**File Information**:
- **Path**: `include/sentio/polygon_ingest.hpp`

- **Size**: 34 lines
- **Modified**: 2025-09-05 15:28:51

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace sentio {

// Polygon-like bar from feeds/adapters.
// ts can be epoch seconds (UTC), epoch milliseconds (UTC), or ISO8601 string.
struct ProviderBar {
  std::string symbol;                                   // instrument actually traded (QQQ/TQQQ/SQQQ/…)
  std::variant<std::int64_t, double, std::string> ts;   // epoch sec (int64), epoch ms (double), or ISO8601
  double open{};
  double high{};
  double low{};
  double close{};
  double volume{};
};

struct Bar { double open{}, high{}, low{}, close{}; };

class PriceBook {
public:
  void upsert_latest(const std::string& instrument, const Bar& b);
  const Bar* get_latest(const std::string& instrument) const;
  bool has_instrument(const std::string& instrument) const;
  std::size_t size() const;
};

std::size_t ingest_provider_bars(const std::vector<ProviderBar>& input, PriceBook& book);
bool        ingest_provider_bar(const ProviderBar& bar, PriceBook& book);

} // namespace sentio

```

## 📄 **FILE 27 of 92**: include/sentio/position_manager.hpp

**File Information**:
- **Path**: `include/sentio/position_manager.hpp`

- **Size**: 63 lines
- **Modified**: 2025-09-05 09:41:27

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include <string>
#include <vector>
#include <unordered_set>

namespace sentio {

enum class PortfolioState { Neutral, LongOnly, ShortOnly };
enum class RequiredAction { None, CloseLong, CloseShort };
enum class Direction { Long, Short }; // Keep for simple directional logic

const std::unordered_set<std::string> LONG_INSTRUMENTS = {"QQQ", "TQQQ", "TSLA"};
const std::unordered_set<std::string> SHORT_INSTRUMENTS = {"SQQQ", "PSQ", "TSLQ"};

class PositionManager {
private:
    PortfolioState state = PortfolioState::Neutral;
    int bars_since_flip = 0;
    [[maybe_unused]] const int cooldown_period = 5;
    
public:
    // **MODIFIED**: Logic restored to work with the new ID-based portfolio structure.
    void update_state(const Portfolio& portfolio, const SymbolTable& ST, const std::vector<double>& last_prices) {
        bars_since_flip++;
        double long_exposure = 0.0;
        double short_exposure = 0.0;

        for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
            const auto& pos = portfolio.positions[sid];
            if (std::abs(pos.qty) < 1e-6) continue;

            const std::string& symbol = ST.get_symbol(sid);
            double exposure = pos.qty * last_prices[sid];

            if (LONG_INSTRUMENTS.count(symbol)) {
                long_exposure += exposure;
            }
            if (SHORT_INSTRUMENTS.count(symbol)) {
                short_exposure += exposure; // Will be negative for short positions
            }
        }
        
        PortfolioState old_state = state;
        
        if (long_exposure > 100 && std::abs(short_exposure) < 100) {
            state = PortfolioState::LongOnly;
        } else if (std::abs(short_exposure) > 100 && long_exposure < 100) {
            state = PortfolioState::ShortOnly;
        } else {
            state = PortfolioState::Neutral;
        }

        if (state != old_state) {
            bars_since_flip = 0;
        }
    }
    
    // ... other methods remain the same ...
};

} // namespace sentio
```

## 📄 **FILE 28 of 92**: include/sentio/pricebook.hpp

**File Information**:
- **Path**: `include/sentio/pricebook.hpp`

- **Size**: 59 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include "core.hpp"
#include "symbol_table.hpp"

namespace sentio {

// **MODIFIED**: The Pricebook ensures that all symbol prices are correctly aligned in time.
// It works by advancing an index for each symbol's data series until its timestamp
// matches or just passes the timestamp of the base symbol's current bar.
// This is both fast (amortized O(1)) and correct, handling missing data gracefully.
struct Pricebook {
    const int base_id;
    const SymbolTable& ST;
    const std::vector<std::vector<Bar>>& S; // Reference to all series data
    std::vector<int> idx;                   // Rolling index for each symbol's series
    std::vector<double> last_px;            // Stores the last known price for each symbol ID

    Pricebook(int base, const SymbolTable& st, const std::vector<std::vector<Bar>>& series)
      : base_id(base), ST(st), S(series), idx(S.size(), 0), last_px(S.size(), 0.0) {}

    // Advances the index 'j' for a given series 'V' to the bar corresponding to 'base_ts'
    inline void advance_to_ts(const std::vector<Bar>& V, int& j, int64_t base_ts) {
        const int n = (int)V.size();
        // Move index forward as long as the *next* bar is still at or before the target time
        while (j + 1 < n && V[j + 1].ts_nyt_epoch <= base_ts) {
            ++j;
        }
    }

    // Syncs all symbol prices to the timestamp of the i-th bar of the base symbol
    inline void sync_to_base_i(int i) {
        if (S[base_id].empty()) return;
        const int64_t ts = S[base_id][i].ts_nyt_epoch;
        
        for (int sid = 0; sid < (int)S.size(); ++sid) {
            if (!S[sid].empty()) {
                advance_to_ts(S[sid], idx[sid], ts);
                last_px[sid] = S[sid][idx[sid]].close;
            }
        }
    }

    // **NEW**: Helper to get the last prices as a map for components needing string keys
    inline std::unordered_map<std::string, double> last_px_map() const {
        std::unordered_map<std::string, double> price_map;
        for (int sid = 0; sid < (int)last_px.size(); ++sid) {
            if (last_px[sid] > 0.0) {
                price_map[ST.get_symbol(sid)] = last_px[sid];
            }
        }
        return price_map;
    }
};

} // namespace sentio
```

## 📄 **FILE 29 of 92**: include/sentio/profiling.hpp

**File Information**:
- **Path**: `include/sentio/profiling.hpp`

- **Size**: 25 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <chrono>
#include <cstdio>

namespace sentio {

struct Tsc {
    std::chrono::high_resolution_clock::time_point t0;
    
    void tic() { 
        t0 = std::chrono::high_resolution_clock::now(); 
    }
    
    double toc_ms() const {
        using namespace std::chrono;
        return duration<double, std::milli>(high_resolution_clock::now() - t0).count();
    }
    
    double toc_sec() const {
        using namespace std::chrono;
        return duration<double>(high_resolution_clock::now() - t0).count();
    }
};

} // namespace sentio
```

## 📄 **FILE 30 of 92**: include/sentio/rolling_stats.hpp

**File Information**:
- **Path**: `include/sentio/rolling_stats.hpp`

- **Size**: 97 lines
- **Modified**: 2025-09-05 13:47:04

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace sentio {

struct RollingMean {
  int win, idx=0, count=0;
  std::vector<double> buf;
  double sum=0.0;
  
  explicit RollingMean(int w = 1) { reset(w); }

  void reset(int w) {
    win = w > 0 ? w : 1;
    idx = 0;
    count = 0;
    sum = 0.0;
    buf.assign(win, 0.0);
  }

  inline double push(double x){
      if (count < win) { 
          buf[count++] = x; 
          sum += x; 
      } else {
          sum -= buf[idx]; 
          buf[idx]=x; 
          sum += x; 
          idx = (idx+1) % win; 
      }
      return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }

  double mean() const {
      return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }

  size_t size() const {
      return static_cast<size_t>(count);
  }
};


struct RollingMeanVar {
  int win, idx=0, count=0;
  std::vector<double> buf;
  double sum=0.0, sumsq=0.0;

  explicit RollingMeanVar(int w = 1) { reset(w); }

  void reset(int w) {
    win = w > 0 ? w : 1;
    idx = 0;
    count = 0;
    sum = 0.0;
    sumsq = 0.0;
    buf.assign(win, 0.0);
  }

  inline std::pair<double,double> push(double x){
    if (count < win) {
      buf[count++] = x; 
      sum += x; 
      sumsq += x*x;
    } else {
      double old_val = buf[idx];
      sum   -= old_val;
      sumsq -= old_val * old_val;
      buf[idx] = x;
      sum   += x;
      sumsq += x*x;
      idx = (idx+1) % win;
    }
    double m = count > 0 ? sum / static_cast<double>(count) : 0.0;
    double v = count > 0 ? std::max(0.0, (sumsq / static_cast<double>(count)) - (m*m)) : 0.0;
    return {m, v};
  }
  
  double mean() const {
      return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }
  
  double var() const {
      if (count < 2) return 0.0;
      double m = mean();
      return std::max(0.0, (sumsq / static_cast<double>(count)) - (m * m));
  }

  double stddev() const {
      return std::sqrt(var());
  }
};

} // namespace sentio

```

## 📄 **FILE 31 of 92**: include/sentio/router.hpp

**File Information**:
- **Path**: `include/sentio/router.hpp`

- **Size**: 62 lines
- **Modified**: 2025-09-05 15:29:30

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <optional>
#include <string>

namespace sentio {

enum class OrderSide { Buy, Sell };

struct StrategySignal {
  enum class Type { BUY, STRONG_BUY, SELL, STRONG_SELL, HOLD };
  Type   type{Type::HOLD};
  double confidence{0.0}; // 0..1
};

struct RouterCfg {
  double min_signal_strength = 0.10; // below -> ignore
  double signal_multiplier   = 1.00; // scales target weight
  double max_position_pct    = 0.05; // +/- 5%
  bool   require_rth         = true; // assume ingest enforces RTH
  // family config
  std::string base_symbol{"QQQ"};
  std::string bull3x{"TQQQ"};
  std::string bear3x{"SQQQ"};
  std::string bear1x{"PSQ"};
  // sizing
  double min_shares = 1.0;
  double lot_size   = 1.0; // for ETFs typically 1
};

struct RouteDecision {
  std::string instrument;
  double      target_weight; // [-max, +max]
};

struct AccountSnapshot { double equity{0.0}; double cash{0.0}; };

struct Order {
  std::string instrument;
  OrderSide   side{OrderSide::Buy};
  double      qty{0.0};
  double      notional{0.0};
  double      limit_price{0.0}; // 0 = market
  std::int64_t ts_utc{0};
  std::string signal_id;
};

class PriceBook; // fwd
double last_trade_price(const PriceBook& book, const std::string& instrument);

std::optional<RouteDecision> route(const StrategySignal& s, const RouterCfg& cfg, const std::string& base_symbol);

// High-level convenience: route + size into a market order
Order route_and_create_order(const std::string& signal_id,
                             const StrategySignal& sig,
                             const RouterCfg& cfg,
                             const std::string& base_symbol,
                             const PriceBook& book,
                             const AccountSnapshot& acct,
                             std::int64_t ts_utc);

} // namespace sentio
```

## 📄 **FILE 32 of 92**: include/sentio/rth_calendar.hpp

**File Information**:
- **Path**: `include/sentio/rth_calendar.hpp`

- **Size**: 35 lines
- **Modified**: 2025-09-05 14:18:43

- **Type**: .hpp

```text
// rth_calendar.hpp
#pragma once
#include <chrono>
#include <string>
#include <unordered_set>
#include <unordered_map>

namespace sentio {

struct TradingCalendar {
  // Holidays: YYYYMMDD integers for fast lookups
  std::unordered_set<int> full_holidays;
  // Early closes (e.g., Black Friday): YYYYMMDD -> close second-of-day (e.g., 13:00 = 13*3600)
  std::unordered_map<int,int> early_close_sec;

  // Regular RTH bounds (seconds from midnight ET)
  static constexpr int RTH_OPEN_SEC  = 9*3600 + 30*60;  // 09:30:00
  static constexpr int RTH_CLOSE_SEC = 16*3600;         // 16:00:00

  // Return yyyymmdd in ET from a zoned_time (no allocations)
  static int yyyymmdd_from_local(const std::chrono::hh_mm_ss<std::chrono::seconds>& tod,
                                 std::chrono::year_month_day ymd) {
    int y = int(ymd.year());
    unsigned m = unsigned(ymd.month());
    unsigned d = unsigned(ymd.day());
    return y*10000 + int(m)*100 + int(d);
  }

  // Main predicate:
  //   ts_utc  = UTC wall clock in seconds since epoch
  //   tz_name = "America/New_York"
  bool is_rth_utc(std::int64_t ts_utc, const std::string& tz_name = "America/New_York") const;
};

} // namespace sentio
```

## 📄 **FILE 33 of 92**: include/sentio/runner.hpp

**File Information**:
- **Path**: `include/sentio/runner.hpp`

- **Size**: 43 lines
- **Modified**: 2025-09-05 20:13:14

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "audit.hpp"
#include "router.hpp"
#include "sizer.hpp"
#include "position_manager.hpp"
#include "cost_model.hpp"
#include "symbol_table.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace sentio {

enum class AuditLevel { Full, MetricsOnly };

struct RunnerCfg {
    std::string strategy_name = "VWAPReversion";
    std::unordered_map<std::string, std::string> strategy_params;
    RouterCfg router;
    SizerCfg sizer;
    AuditLevel audit_level = AuditLevel::Full;
    int snapshot_stride = 100;
    std::string audit_file = "audit.jsonl";  // JSONL audit file path
};

struct RunResult {
    double final_equity;
    double total_return;
    double sharpe_ratio;
    double max_drawdown;
    double monthly_projected_return;
    int daily_trades;
    int total_fills;
    int no_route;
    int no_qty;
};

RunResult run_backtest(AuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                      int base_symbol_id, const RunnerCfg& cfg);

} // namespace sentio


```

## 📄 **FILE 34 of 92**: include/sentio/signal_diag.hpp

**File Information**:
- **Path**: `include/sentio/signal_diag.hpp`

- **Size**: 35 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
// signal_diag.hpp
#pragma once
#include <cstdint>
#include <cstdio>

enum class DropReason : uint8_t {
  NONE, MIN_BARS, SESSION, NAN_INPUT, ZERO_VOL, THRESHOLD, COOLDOWN, DUP_SAME_BAR
};

struct SignalDiag {
  uint64_t emitted=0, dropped=0;
  uint64_t r_min_bars=0, r_session=0, r_nan=0, r_zero_vol=0, r_threshold=0, r_cooldown=0, r_dup=0;

  inline void drop(DropReason r){
    dropped++;
    switch(r){
      case DropReason::MIN_BARS:  r_min_bars++; break;
      case DropReason::SESSION:   r_session++;  break;
      case DropReason::NAN_INPUT: r_nan++;      break;
      case DropReason::ZERO_VOL:  r_zero_vol++; break;
      case DropReason::THRESHOLD: r_threshold++;break;
      case DropReason::COOLDOWN:  r_cooldown++; break;
      case DropReason::DUP_SAME_BAR:r_dup++;    break;
      default: break;
    }
  }

  inline void print(const char* tag) const {
    std::fprintf(stderr, "[SIG %s] emitted=%llu dropped=%llu  min_bars=%llu session=%llu nan=%llu zerovol=%llu thr=%llu cooldown=%llu dup=%llu\n",
      tag,(unsigned long long)emitted,(unsigned long long)dropped,
      (unsigned long long)r_min_bars,(unsigned long long)r_session,(unsigned long long)r_nan,
      (unsigned long long)r_zero_vol,(unsigned long long)r_threshold,(unsigned long long)r_cooldown,
      (unsigned long long)r_dup);
  }
};
```

## 📄 **FILE 35 of 92**: include/sentio/signal_engine.hpp

**File Information**:
- **Path**: `include/sentio/signal_engine.hpp`

- **Size**: 22 lines
- **Modified**: 2025-09-05 16:34:56

- **Type**: .hpp

```text
#pragma once
#include "strategy_base.hpp"
#include "signal_gate.hpp"

namespace sentio {

struct EngineOut {
  std::optional<StrategySignal> signal; // post-gate
  DropReason last_drop{DropReason::NONE};
};

class SignalEngine {
public:
  SignalEngine(IStrategy* strat, const GateCfg& gate_cfg, SignalHealth* health);
  EngineOut on_bar(const StrategyCtx& ctx, const Bar& b, bool inputs_finite=true);
private:
  IStrategy* strat_;
  SignalGate gate_;
  SignalHealth* health_;
};

} // namespace sentio

```

## 📄 **FILE 36 of 92**: include/sentio/signal_gate.hpp

**File Information**:
- **Path**: `include/sentio/signal_gate.hpp`

- **Size**: 46 lines
- **Modified**: 2025-09-05 17:01:14

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <atomic>
#include <optional>

namespace sentio {

enum class DropReason : uint16_t {
  NONE=0, NOT_RTH, WARMUP, NAN_INPUT, THRESHOLD_TOO_TIGHT,
  COOLDOWN_ACTIVE, DUPLICATE_BAR_TS
};

struct SignalHealth {
  std::atomic<uint64_t> emitted{0};
  std::atomic<uint64_t> dropped{0};
  std::unordered_map<DropReason, std::atomic<uint64_t>> by_reason;
  SignalHealth();
  void incr_emit();
  void incr_drop(DropReason r);
};

struct GateCfg { 
  bool require_rth=true; 
  int cooldown_bars=0; 
  double min_conf=0.05; 
};

class SignalGate {
public:
  explicit SignalGate(const GateCfg& cfg, SignalHealth* health);
  // Returns nullopt if dropped; otherwise passes through with possibly clamped confidence.
  std::optional<double> accept(std::int64_t ts_utc_epoch,
                               bool is_rth,
                               bool inputs_finite,
                               bool warmed_up,
                               double confidence);
private:
  GateCfg cfg_;
  SignalHealth* health_;
  std::int64_t last_emit_ts_{-1};
  int cooldown_left_{0};
};

} // namespace sentio

```

## 📄 **FILE 37 of 92**: include/sentio/signal_pipeline.hpp

**File Information**:
- **Path**: `include/sentio/signal_pipeline.hpp`

- **Size**: 42 lines
- **Modified**: 2025-09-05 17:05:41

- **Type**: .hpp

```text
#pragma once
#include "strategy_base.hpp"
#include "signal_gate.hpp"
#include "signal_trace.hpp"
#include <cstdint>
#include <string>
#include <optional>

namespace sentio {

// Forward declarations to avoid include conflicts
struct RouterCfg;
struct RouteDecision;
struct Order;
struct AccountSnapshot;
class PriceBook;

struct PipelineCfg {
  GateCfg gate;
  double min_order_shares{1.0};
};

struct PipelineOut {
  std::optional<StrategySignal> signal;
  TraceRow trace;
};

class SignalPipeline {
public:
  SignalPipeline(IStrategy* strat, const PipelineCfg& cfg, void* book, SignalTrace* trace)
  : strat_(strat), cfg_(cfg), book_(book), trace_(trace), gate_(cfg.gate, nullptr) {}

  PipelineOut on_bar(const StrategyCtx& ctx, const Bar& b, const void* acct);
private:
  IStrategy* strat_;
  PipelineCfg cfg_;
  void* book_;  // Avoid include conflicts
  SignalTrace* trace_;
  SignalGate gate_;
};

} // namespace sentio

```

## 📄 **FILE 38 of 92**: include/sentio/signal_trace.hpp

**File Information**:
- **Path**: `include/sentio/signal_trace.hpp`

- **Size**: 53 lines
- **Modified**: 2025-09-05 17:00:48

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <optional>

namespace sentio {

enum class TraceReason : uint16_t {
  OK = 0,
  NO_STRATEGY_OUTPUT,
  NOT_RTH,
  HOLIDAY,
  WARMUP,
  NAN_INPUT,
  THRESHOLD_TOO_TIGHT,
  COOLDOWN_ACTIVE,
  DUPLICATE_BAR_TS,
  EMPTY_PRICEBOOK,
  NO_PRICE_FOR_INSTRUMENT,
  ROUTER_REJECTED,
  ORDER_QTY_LT_MIN,
  UNKNOWN
};

struct TraceRow {
  std::int64_t ts_utc{};
  std::string  instrument;     // stream instrument (e.g., QQQ)
  std::string  routed;         // routed instrument (e.g., TQQQ/SQQQ)
  double       close{};
  bool         is_rth{true};
  bool         warmed{true};
  bool         inputs_finite{true};
  double       confidence{0.0};              // raw strategy conf
  double       conf_after_gate{0.0};         // post gate
  double       target_weight{0.0};           // router decision
  double       last_px{0.0};                 // last price for routed
  double       order_qty{0.0};
  TraceReason  reason{TraceReason::UNKNOWN};
  std::string  note;                         // optional detail
};

class SignalTrace {
public:
  void push(const TraceRow& r) { rows_.push_back(r); }
  const std::vector<TraceRow>& rows() const { return rows_; }
  // useful summaries
  std::size_t count(TraceReason r) const;
private:
  std::vector<TraceRow> rows_;
};

} // namespace sentio

```

## 📄 **FILE 39 of 92**: include/sentio/sizer.hpp

**File Information**:
- **Path**: `include/sentio/sizer.hpp`

- **Size**: 95 lines
- **Modified**: 2025-09-05 09:44:09

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace sentio {

// Advanced Sizer Configuration with Risk Controls
struct SizerCfg {
  bool fractional_allowed = true;
  double min_notional = 1.0;
  double max_leverage = 2.0;
  double max_position_pct = 0.25;
  double volatility_target = 0.15;
  bool allow_negative_cash = false;
  int vol_lookback_days = 20;
  double cash_reserve_pct = 0.05;
};

// Advanced Sizer Class with Multiple Constraints
class AdvancedSizer {
public:
  double calculate_volatility(const std::vector<Bar>& price_history, int lookback) const {
    if (price_history.size() < static_cast<size_t>(lookback)) return 0.05; // Default vol

    std::vector<double> returns;
    returns.reserve(lookback - 1);
    for (size_t i = price_history.size() - lookback + 1; i < price_history.size(); ++i) {
      double prev_close = price_history[i-1].close;
      if (prev_close > 0) {
        returns.push_back(price_history[i].close / prev_close - 1.0);
      }
    }
    
    if (returns.size() < 2) return 0.05;
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - mean) * (ret - mean);
    }
    variance /= returns.size();
    return std::sqrt(variance) * std::sqrt(252.0); // Annualized
  }

  // **MODIFIED**: Signature and logic updated for the ID-based, high-performance architecture.
  double calculate_target_quantity(const Portfolio& portfolio,
                                   const SymbolTable& ST,
                                   const std::vector<double>& last_prices,
                                   const std::string& instrument,
                                   double target_weight,
                                                                       [[maybe_unused]] const std::vector<Bar>& price_history,
                                   const SizerCfg& cfg) const {
    
    const double equity = equity_mark_to_market(portfolio, last_prices);
    int instrument_id = ST.get_id(instrument);

    if (equity <= 0 || instrument_id == -1 || last_prices[instrument_id] <= 0) {
        return 0.0;
    }
    double instrument_price = last_prices[instrument_id];

    // --- Calculate size based on multiple constraints ---
    double desired_notional = equity * std::abs(target_weight);

    // 1. Max Position Size Constraint
    desired_notional = std::min(desired_notional, equity * cfg.max_position_pct);

    // 2. Leverage Constraint
    double current_exposure = 0.0;
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        current_exposure += std::abs(portfolio.positions[sid].qty * last_prices[sid]);
    }
    double available_leverage_notional = (equity * cfg.max_leverage) - current_exposure;
    desired_notional = std::min(desired_notional, std::max(0.0, available_leverage_notional));

    // 4. Cash Constraint
    if (!cfg.allow_negative_cash) {
      double usable_cash = portfolio.cash * (1.0 - cfg.cash_reserve_pct);
      desired_notional = std::min(desired_notional, std::max(0.0, usable_cash));
    }
    
    if (desired_notional < cfg.min_notional) return 0.0;
    
    double qty = desired_notional / instrument_price;
    double final_qty = cfg.fractional_allowed ? qty : std::floor(qty);
    
    // Return with the correct sign (long/short)
    return (target_weight > 0) ? final_qty : -final_qty;
  }
};

} // namespace sentio
```

## 📄 **FILE 40 of 92**: include/sentio/strategy_base.hpp

**File Information**:
- **Path**: `include/sentio/strategy_base.hpp`

- **Size**: 29 lines
- **Modified**: 2025-09-05 16:34:35

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <string>
#include <optional>

namespace sentio {

struct Bar { double open{}, high{}, low{}, close{}; };

struct StrategySignal {
  enum class Type { BUY, STRONG_BUY, SELL, STRONG_SELL, HOLD };
  Type   type{Type::HOLD};
  double confidence{0.0}; // 0..1
};

struct StrategyCtx {
  std::string instrument;     // traded instrument for this stream
  std::int64_t ts_utc_epoch;  // bar timestamp (UTC seconds)
  bool is_rth{true};          // inject from your RTH checker
};

class IStrategy {
public:
  virtual ~IStrategy() = default;
  virtual void on_bar(const StrategyCtx& ctx, const Bar& b) = 0;
  virtual std::optional<StrategySignal> latest() const = 0;
};

} // namespace sentio

```

## 📄 **FILE 41 of 92**: include/sentio/strategy_bollinger_squeeze_breakout.hpp

**File Information**:
- **Path**: `include/sentio/strategy_bollinger_squeeze_breakout.hpp`

- **Size**: 44 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "bollinger.hpp"
#include <vector>
#include <string>

namespace sentio {

class BollingerSqueezeBreakoutStrategy : public BaseStrategy {
private:
    enum class State { Idle, Squeezed, ArmedLong, ArmedShort, Long, Short };
    
    // **MODIFIED**: Cached parameters
    int bb_window_;
    double squeeze_percentile_;
    int squeeze_lookback_;
    int hold_max_bars_;
    double tp_mult_sd_;
    double sl_mult_sd_;
    int min_squeeze_bars_;

    // Strategy state & indicators
    State state_ = State::Idle;
    int bars_in_trade_ = 0;
    int squeeze_duration_ = 0;
    Bollinger bollinger_;
    std::vector<double> sd_history_;
    
    // Helper methods
    double calculate_volatility_percentile(double percentile) const;
    void update_state_machine(const Bar& bar);
    
public:
    BollingerSqueezeBreakoutStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 42 of 92**: include/sentio/strategy_market_making.hpp

**File Information**:
- **Path**: `include/sentio/strategy_market_making.hpp`

- **Size**: 52 lines
- **Modified**: 2025-09-05 12:22:58

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp" // For rolling volume
#include <deque>

namespace sentio {

class MarketMakingStrategy : public BaseStrategy {
private:
    // **MODIFIED**: Cached parameters
    double base_spread_;
    double min_spread_;
    double max_spread_;
    int order_levels_;
    double level_spacing_;
    double order_size_base_;
    double max_inventory_;
    double inventory_skew_mult_;
    double adverse_selection_threshold_;
    double min_volume_ratio_;
    int max_orders_per_bar_;
    int rebalance_frequency_;

    // Market making state
    struct MarketState {
        double inventory = 0.0;
        double average_cost = 0.0;
        int last_rebalance = 0;
        int orders_this_bar = 0;
    };
    MarketState market_state_;

    // **NEW**: Stateful, rolling indicators for performance
    RollingMeanVar rolling_returns_;
    RollingMean rolling_volume_;

    // Helper methods
    bool should_participate(const Bar& bar);
    double get_inventory_skew() const;
    
public:
    MarketMakingStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 43 of 92**: include/sentio/strategy_momentum_volume.hpp

**File Information**:
- **Path**: `include/sentio/strategy_momentum_volume.hpp`

- **Size**: 57 lines
- **Modified**: 2025-09-05 12:25:31

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp" // **NEW**: For efficient MA calculations
#include <map>

namespace sentio {

class MomentumVolumeProfileStrategy : public BaseStrategy {
private:
    // **MODIFIED**: Cached parameters for performance
    int profile_period_;
    double value_area_pct_;
    int price_bins_;
    double breakout_threshold_pct_;
    int momentum_lookback_;
    double volume_surge_mult_;
    int cool_down_period_;

    // Volume profile data structures
    struct VolumeNode {
        double price_level;
        double volume;
    };
    struct VolumeProfile {
        std::map<double, VolumeNode> profile;
        double poc_level = 0.0;
        double value_area_high = 0.0;
        double value_area_low = 0.0;
        double total_volume = 0.0;
        void clear() { /* ... unchanged ... */ }
    };
    
    // Strategy state
    VolumeProfile volume_profile_;
    int last_profile_update_ = -1;
    
    // **NEW**: Stateful, rolling indicators for performance
    RollingMean avg_volume_;

    // Helper methods
    void build_volume_profile(const std::vector<Bar>& bars, int end_index);
    void calculate_value_area();
    bool is_momentum_confirmed(const std::vector<Bar>& bars, int index) const;
    
public:
    MomentumVolumeProfileStrategy();
    
    // BaseStrategy interface
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 44 of 92**: include/sentio/strategy_opening_range_breakout.hpp

**File Information**:
- **Path**: `include/sentio/strategy_opening_range_breakout.hpp`

- **Size**: 38 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"

namespace sentio {

class OpeningRangeBreakoutStrategy : public BaseStrategy {
private:
    // **MODIFIED**: Cached parameters
    int opening_range_minutes_;
    int breakout_confirmation_bars_;
    double volume_multiplier_;
    double stop_loss_pct_;
    double take_profit_pct_;
    int cool_down_period_;

    struct OpeningRange {
        double high = 0.0;
        double low = 0.0;
        int end_bar = -1;
        bool is_finalized = false;
    };
    
    // Strategy state
    OpeningRange current_range_;
    int day_start_index_ = -1;
    
public:
    OpeningRangeBreakoutStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 45 of 92**: include/sentio/strategy_order_flow_imbalance.hpp

**File Information**:
- **Path**: `include/sentio/strategy_order_flow_imbalance.hpp`

- **Size**: 40 lines
- **Modified**: 2025-09-05 12:32:08

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp"

namespace sentio {

class OrderFlowImbalanceStrategy : public BaseStrategy {
private:
    // Cached parameters
    int lookback_period_;
    double entry_threshold_long_;
    double entry_threshold_short_;
    int hold_max_bars_;
    int cool_down_period_;

    // Strategy-specific state machine
    enum class OFIState { Flat, Long, Short };
    // **FIXED**: Renamed this member to 'ofi_state_' to avoid conflict
    // with the 'state_' member inherited from BaseStrategy.
    OFIState ofi_state_ = OFIState::Flat;
    int bars_in_trade_ = 0;

    // Rolling indicator to measure average pressure
    RollingMean rolling_pressure_;

    // Helper to calculate buying/selling pressure proxy from a bar
    double calculate_bar_pressure(const Bar& bar) const;
    
public:
    OrderFlowImbalanceStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio

```

## 📄 **FILE 46 of 92**: include/sentio/strategy_order_flow_scalping.hpp

**File Information**:
- **Path**: `include/sentio/strategy_order_flow_scalping.hpp`

- **Size**: 38 lines
- **Modified**: 2025-09-05 13:09:39

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp"

namespace sentio {

class OrderFlowScalpingStrategy : public BaseStrategy {
private:
    // Cached parameters
    int lookback_period_;
    double imbalance_threshold_;
    int hold_max_bars_;
    int cool_down_period_;

    // State machine states
    enum class OFState { Idle, ArmedLong, ArmedShort, Long, Short };
    
    // **FIXED**: Renamed this member to 'of_state_' to avoid conflict
    // with the 'state_' member inherited from BaseStrategy.
    OFState of_state_ = OFState::Idle;
    int bars_in_trade_ = 0;
    RollingMean rolling_pressure_;

    // Helper to calculate buying/selling pressure proxy from a bar
    double calculate_bar_pressure(const Bar& bar) const;
    
public:
    OrderFlowScalpingStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio

```

## 📄 **FILE 47 of 92**: include/sentio/strategy_sma_cross.hpp

**File Information**:
- **Path**: `include/sentio/strategy_sma_cross.hpp`

- **Size**: 27 lines
- **Modified**: 2025-09-05 16:34:40

- **Type**: .hpp

```text
#pragma once
#include "strategy_base.hpp"
#include "indicators.hpp"
#include <optional>

namespace sentio {

struct SMACrossCfg {
  int fast = 10;
  int slow = 30;
  double conf_fast_slow = 0.7; // confidence when cross happens
};

class SMACrossStrategy final : public IStrategy {
public:
  explicit SMACrossStrategy(const SMACrossCfg& cfg);
  void on_bar(const StrategyCtx& ctx, const Bar& b) override;
  std::optional<StrategySignal> latest() const override { return last_; }
  bool warmed_up() const { return sma_fast_.ready() && sma_slow_.ready(); }
private:
  SMACrossCfg cfg_;
  SMA sma_fast_, sma_slow_;
  double last_fast_{NAN}, last_slow_{NAN};
  std::optional<StrategySignal> last_;
};

} // namespace sentio

```

## 📄 **FILE 48 of 92**: include/sentio/strategy_vwap_reversion.hpp

**File Information**:
- **Path**: `include/sentio/strategy_vwap_reversion.hpp`

- **Size**: 47 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"

namespace sentio {

class VWAPReversionStrategy : public BaseStrategy {
private:
    // Cached parameters for performance
    int vwap_period_;
    double band_multiplier_;
    double max_band_width_;
    double min_distance_from_vwap_;
    double volume_confirmation_mult_;
    int rsi_period_;
    double rsi_oversold_;
    double rsi_overbought_;
    double stop_loss_pct_;
    double take_profit_pct_;
    int time_stop_bars_;
    int cool_down_period_;

    // VWAP calculation state
    double cumulative_pv_ = 0.0;
    double cumulative_volume_ = 0.0;
    
    // Strategy state
    int time_in_position_ = 0;
    double vwap_ = 0.0;
    
    // Helper methods
    void update_vwap(const Bar& bar);
    bool is_volume_confirmed(const std::vector<Bar>& bars, int index) const;
    bool is_rsi_condition_met(const std::vector<Bar>& bars, int index, bool for_buy) const;
    double calculate_simple_rsi(const std::vector<double>& prices) const;
    
public:
    VWAPReversionStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 49 of 92**: include/sentio/symbol_table.hpp

**File Information**:
- **Path**: `include/sentio/symbol_table.hpp`

- **Size**: 35 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <unordered_map>

namespace sentio {

struct SymbolTable {
  std::vector<std::string> id2sym;
  std::unordered_map<std::string,int> sym2id;

  int intern(const std::string& s){
    auto it = sym2id.find(s);
    if (it != sym2id.end()) return it->second;
    int id = (int)id2sym.size();
    id2sym.push_back(s);
    sym2id.emplace(id2sym.back(), id);
    return id;
  }

  const std::string& get_symbol(int id) const {
    return id2sym[id];
  }

  int get_id(const std::string& sym) const {
    auto it = sym2id.find(sym);
    return it != sym2id.end() ? it->second : -1;
  }

  size_t size() const {
    return id2sym.size();
  }
};

} // namespace sentio
```

## 📄 **FILE 50 of 92**: include/sentio/time_utils.hpp

**File Information**:
- **Path**: `include/sentio/time_utils.hpp`

- **Size**: 15 lines
- **Modified**: 2025-09-05 15:29:30

- **Type**: .hpp

```text
#pragma once
#include <chrono>
#include <string>
#include <variant>

namespace sentio {

// Normalize various timestamp representations to UTC epoch seconds.
std::chrono::sys_seconds to_utc_sys_seconds(const std::variant<std::int64_t, double, std::string>& ts);

// Helpers exposed for tests
bool iso8601_looks_like(const std::string& s);
bool epoch_ms_suspected(double v_ms);

} // namespace sentio
```

## 📄 **FILE 51 of 92**: include/sentio/wf.hpp

**File Information**:
- **Path**: `include/sentio/wf.hpp`

- **Size**: 52 lines
- **Modified**: 2025-09-05 20:13:14

- **Type**: .hpp

```text
#pragma once
#include "sentio/core.hpp"
#include "sentio/runner.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/audit.hpp"
#include <vector>
#include <string>

namespace sentio {

struct WfCfg {
    int train_days = 252;  // Training period in days
    int oos_days = 14;     // Out-of-sample period in days
    int step_days = 14;    // Step size between folds
    bool enable_optimization = false;
    std::string optimizer_type = "random";
    int max_optimization_trials = 30;
    std::string optimization_objective = "sharpe_ratio";
    double optimization_timeout_minutes = 15.0;
    bool optimization_verbose = true;
};

struct Gate {
    bool wf_pass = false;
    bool oos_pass = false;
    std::string recommend = "REJECT";
    double oos_monthly_avg_return = 0.0;
    double oos_sharpe = 0.0;
    double oos_max_drawdown = 0.0;
    int successful_optimizations = 0;
    double avg_optimization_improvement = 0.0;
    std::vector<std::pair<std::string, double>> optimization_results;
};

// Walk-forward testing with vector-based data
Gate run_wf_and_gate(AuditRecorder& audit_template,
                     const SymbolTable& ST,
                     const std::vector<std::vector<Bar>>& series,
                     int base_symbol_id,
                     const RunnerCfg& rcfg,
                     const WfCfg& wcfg);

// Walk-forward testing with optimization
Gate run_wf_and_gate_optimized(AuditRecorder& audit_template,
                               const SymbolTable& ST,
                               const std::vector<std::vector<Bar>>& series,
                               int base_symbol_id,
                               const RunnerCfg& base_rcfg,
                               const WfCfg& wcfg);

} // namespace sentio


```

## 📄 **FILE 52 of 92**: src/audit.cpp

**File Information**:
- **Path**: `src/audit.cpp`

- **Size**: 260 lines
- **Modified**: 2025-09-05 20:13:14

- **Type**: .cpp

```text
#include "sentio/audit.hpp"
#include <cstring>
#include <sstream>
#include <iomanip>

// ---- minimal SHA1 (tiny, not constant-time; for tamper detection only) ----
namespace {
struct Sha1 {
  uint32_t h0=0x67452301, h1=0xEFCDAB89, h2=0x98BADCFE, h3=0x10325476, h4=0xC3D2E1F0;
  std::string hex(const std::string& s){
    uint64_t ml = s.size()*8ULL;
    std::string msg = s;
    msg.push_back('\x80');
    while ((msg.size()%64)!=56) msg.push_back('\0');
    for (int i=7;i>=0;--i) msg.push_back(char((ml>>(i*8))&0xFF));
    for (size_t i=0;i<msg.size(); i+=64) {
      uint32_t w[80];
      for (int t=0;t<16;++t) {
        w[t] = (uint32_t(uint8_t(msg[i+4*t]))<<24)
             | (uint32_t(uint8_t(msg[i+4*t+1]))<<16)
             | (uint32_t(uint8_t(msg[i+4*t+2]))<<8)
             | (uint32_t(uint8_t(msg[i+4*t+3])));
      }
      for (int t=16;t<80;++t){ uint32_t v = w[t-3]^w[t-8]^w[t-14]^w[t-16]; w[t]=(v<<1)|(v>>31); }
      uint32_t a=h0,b=h1,c=h2,d=h3,e=h4;
      for (int t=0;t<80;++t){
        uint32_t f,k;
        if (t<20){ f=(b&c)|((~b)&d); k=0x5A827999; }
        else if (t<40){ f=b^c^d; k=0x6ED9EBA1; }
        else if (t<60){ f=(b&c)|(b&d)|(c&d); k=0x8F1BBCDC; }
        else { f=b^c^d; k=0xCA62C1D6; }
        uint32_t temp = ((a<<5)|(a>>27)) + f + e + k + w[t];
        e=d; d=c; c=((b<<30)|(b>>2)); b=a; a=temp;
      }
      h0+=a; h1+=b; h2+=c; h3+=d; h4+=e;
    }
    std::ostringstream os; os<<std::hex<<std::setfill('0')<<std::nouppercase;
    os<<std::setw(8)<<h0<<std::setw(8)<<h1<<std::setw(8)<<h2<<std::setw(8)<<h3<<std::setw(8)<<h4;
    return os.str();
  }
};
}

namespace sentio {

static inline std::string num_s(double v){
  std::ostringstream os; os.setf(std::ios::fixed); os<<std::setprecision(8)<<v; return os.str();
}
static inline std::string num_i(std::int64_t v){
  std::ostringstream os; os<<v; return os.str();
}
std::string AuditRecorder::json_escape_(const std::string& s){
  std::string o; o.reserve(s.size()+8);
  for (char c: s){
    switch(c){
      case '"': o+="\\\""; break;
      case '\\':o+="\\\\"; break;
      case '\b':o+="\\b"; break;
      case '\f':o+="\\f"; break;
      case '\n':o+="\\n"; break;
      case '\r':o+="\\r"; break;
      case '\t':o+="\\t"; break;
      default: o.push_back(c);
    }
  }
  return o;
}

std::string AuditRecorder::sha1_hex_(const std::string& s){
  Sha1 sh; return sh.hex(s);
}

AuditRecorder::AuditRecorder(const AuditConfig& cfg)
: run_id_(cfg.run_id), file_path_(cfg.file_path), flush_each_(cfg.flush_each)
{
  fp_ = std::fopen(cfg.file_path.c_str(), "ab");
  if (!fp_) throw std::runtime_error("Audit open failed: "+cfg.file_path);
}
AuditRecorder::~AuditRecorder(){ if (fp_) std::fclose(fp_); }

void AuditRecorder::write_line_(const std::string& body){
  std::string core = "{\"run\":\""+json_escape_(run_id_)+"\",\"seq\":"+num_i((std::int64_t)seq_)+","+body+"}";
  std::string line = core; // sha1 over core for stability
  std::string h = sha1_hex_(core);
  line.pop_back(); // remove trailing '}'
  line += ",\"sha1\":\""+h+"\"}\n";
  if (std::fwrite(line.data(),1,line.size(),fp_)!=line.size()) throw std::runtime_error("Audit write failed");
  if (flush_each_) std::fflush(fp_);
  ++seq_;
}

void AuditRecorder::event_run_start(std::int64_t ts, const std::string& meta){
  write_line_("\"type\":\"run_start\",\"ts\":"+num_i(ts)+",\"meta\":"+meta+"}");
}
void AuditRecorder::event_run_end(std::int64_t ts, const std::string& meta){
  write_line_("\"type\":\"run_end\",\"ts\":"+num_i(ts)+",\"meta\":"+meta+"}");
}
void AuditRecorder::event_bar(std::int64_t ts, const std::string& inst, const AuditBar& b){
  write_line_("\"type\":\"bar\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"o\":"+num_s(b.open)+",\"h\":"+num_s(b.high)+",\"l\":"+num_s(b.low)+",\"c\":"+num_s(b.close)+",\"v\":"+num_s(b.volume)+"}");
}
void AuditRecorder::event_signal(std::int64_t ts, const std::string& base, SigType t, double conf){
  write_line_("\"type\":\"signal\",\"ts\":"+num_i(ts)+",\"base\":\""+json_escape_(base)+"\",\"sig\":" + num_i((int)t) + ",\"conf\":"+num_s(conf)+"}");
}
void AuditRecorder::event_route (std::int64_t ts, const std::string& base, const std::string& inst, double tw){
  write_line_("\"type\":\"route\",\"ts\":"+num_i(ts)+",\"base\":\""+json_escape_(base)+"\",\"inst\":\""+json_escape_(inst)+"\",\"tw\":"+num_s(tw)+"}");
}
void AuditRecorder::event_order (std::int64_t ts, const std::string& inst, Side side, double qty, double limit_px){
  write_line_("\"type\":\"order\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"side\":"+num_i((int)side)+",\"qty\":"+num_s(qty)+",\"limit\":"+num_s(limit_px)+"}");
}
void AuditRecorder::event_fill  (std::int64_t ts, const std::string& inst, double price, double qty, double fees, Side side){
  write_line_("\"type\":\"fill\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"px\":"+num_s(price)+",\"qty\":"+num_s(qty)+",\"fees\":"+num_s(fees)+",\"side\":"+num_i((int)side)+"}");
}
void AuditRecorder::event_snapshot(std::int64_t ts, const AccountState& a){
  write_line_("\"type\":\"snapshot\",\"ts\":"+num_i(ts)+",\"cash\":"+num_s(a.cash)+",\"real\":"+num_s(a.realized)+",\"equity\":"+num_s(a.equity)+"}");
}
void AuditRecorder::event_metric(std::int64_t ts, const std::string& key, double val){
  write_line_("\"type\":\"metric\",\"ts\":"+num_i(ts)+",\"key\":\""+json_escape_(key)+"\",\"val\":"+num_s(val)+"}");
}

// ----------------- Replayer --------------------

static inline bool parse_kv(const std::string& s, const char* key, std::string& out) {
  auto kq = std::string("\"")+key+"\":";
  auto p = s.find(kq); if (p==std::string::npos) return false;
  p += kq.size();
  if (p>=s.size()) return false;
  if (s[p]=='"'){ // string
    auto e = s.find('"', p+1);
    if (e==std::string::npos) return false;
    out = s.substr(p+1, e-(p+1));
    return true;
  } else { // number or enum
    auto e = s.find_first_of(",}", p);
    out = s.substr(p, e-p);
    return true;
  }
}

std::optional<ReplayResult> AuditReplayer::replay_file(const std::string& path,
                                                       const std::string& run_expect)
{
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) return std::nullopt;

  PriceBook pb;
  ReplayResult rr;
  rr.acct.cash = 0.0;
  rr.acct.realized = 0.0;

  char buf[16*1024];
  while (std::fgets(buf, sizeof(buf), fp)) {
    std::string line(buf);
    // very light JSONL parsing (we control writer)

    std::string run; if (!parse_kv(line, "run", run)) continue;
    if (!run_expect.empty() && run!=run_expect) continue;

    std::string type; if (!parse_kv(line, "type", type)) continue;
    std::string ts_s; parse_kv(line, "ts", ts_s);
    std::int64_t ts = ts_s.empty()?0:std::stoll(ts_s);

    if (type=="bar") {
      std::string inst; parse_kv(line, "inst", inst);
      std::string o,h,l,c; parse_kv(line,"o",o); parse_kv(line,"h",h);
      parse_kv(line,"l",l); parse_kv(line,"c",c);
      AuditBar b{std::stod(o),std::stod(h),std::stod(l),std::stod(c)};
      apply_bar_(pb, inst, b);
      ++rr.bars;
    } else if (type=="signal") {
      ++rr.signals;
    } else if (type=="route") {
      ++rr.routes;
    } else if (type=="order") {
      ++rr.orders;
    } else if (type=="fill") {
      std::string inst,px,qty,fees,side_s;
      parse_kv(line,"inst",inst); parse_kv(line,"px",px);
      parse_kv(line,"qty",qty); parse_kv(line,"fees",fees);
      parse_kv(line,"side",side_s);
      Side side = side_s.empty() ? Side::Buy : static_cast<Side>(std::stoi(side_s));
      apply_fill_(rr, inst, std::stod(px), std::stod(qty), std::stod(fees), side);
      ++rr.fills;
      mark_to_market_(pb, rr);
    } else if (type=="snapshot") {
      // snapshots are optional for verification; we recompute anyway
      // you can cross-check here if you also store snapshots
    } else if (type=="run_end") {
      // could verify sha1 continuity/counts here
    }
  }
  std::fclose(fp);

  mark_to_market_(pb, rr);
  return rr;
}

bool AuditReplayer::apply_bar_(PriceBook& pb, const std::string& instrument, const AuditBar& b) {
  pb.last_px[instrument] = b.close;
  return true;
}

void AuditReplayer::apply_fill_(ReplayResult& rr, const std::string& inst, double px, double qty, double fees, Side side) {
  auto& pos = rr.positions[inst];
  
  // Convert order side to position impact
  // Buy orders: positive position qty, Sell orders: negative position qty
  double position_qty = (side == Side::Buy) ? qty : -qty;
  
  // cash impact: buy qty>0 => cash decreases, sell qty>0 => cash increases
  double cash_delta = (side == Side::Buy) ? -(px*qty + fees) : (px*qty - fees);
  rr.acct.cash += cash_delta;

  // position update (VWAP)
  double new_qty = pos.qty + position_qty;
  
  if (new_qty == 0.0) {
    // flat: realize P&L for the round trip
    if (pos.qty != 0.0) {
      rr.acct.realized += (px - pos.avg_px) * pos.qty;
    }
    pos.qty = 0.0; 
    pos.avg_px = 0.0;
  } else if (pos.qty == 0.0) {
    // opening new position
    pos.qty = new_qty;
    pos.avg_px = px;
  } else if ((pos.qty > 0) == (position_qty > 0)) {
    // adding to same side -> new average
    pos.avg_px = (pos.avg_px * pos.qty + px * position_qty) / new_qty;
    pos.qty = new_qty;
  } else {
    // reducing or flipping side -> realize partial P&L
    double closed_qty = std::min(std::abs(position_qty), std::abs(pos.qty));
    if (pos.qty > 0) {
      rr.acct.realized += (px - pos.avg_px) * closed_qty;
    } else {
      rr.acct.realized += (pos.avg_px - px) * closed_qty;
    }
    pos.qty = new_qty;
    if (pos.qty == 0.0) {
      pos.avg_px = 0.0;
    } else {
      pos.avg_px = px; // new average for remaining position
    }
  }
}

void AuditReplayer::mark_to_market_(const PriceBook& pb, ReplayResult& rr) {
  double mtm=0.0;
  for (auto& kv : rr.positions) {
    const auto& inst = kv.first;
    const auto& p = kv.second;
    auto it = pb.last_px.find(inst);
    if (it==pb.last_px.end()) continue;
    mtm += p.qty * it->second;
  }
  rr.acct.equity = rr.acct.cash + rr.acct.realized + mtm;
}

} // namespace sentio
```

## 📄 **FILE 53 of 92**: src/base_strategy.cpp

**File Information**:
- **Path**: `src/base_strategy.cpp`

- **Size**: 46 lines
- **Modified**: 2025-09-05 16:02:04

- **Type**: .cpp

```text
#include "sentio/base_strategy.hpp"
#include <iostream>

namespace sentio {

void BaseStrategy::reset_state() {
    state_.reset();
    diag_ = SignalDiag{};
}

bool BaseStrategy::is_cooldown_active(int current_bar, int cooldown_period) const {
    return (current_bar - state_.last_trade_bar) < cooldown_period;
}

// **MODIFIED**: This function now merges incoming parameters (overrides)
// with the existing default parameters, preventing the defaults from being erased.
void BaseStrategy::set_params(const ParameterMap& overrides) {
    // The constructor has already set the defaults.
    // Now, merge the overrides into the existing params_.
    for (const auto& [key, value] : overrides) {
        params_[key] = value;
    }
    
    apply_params();
}

// --- Strategy Factory Implementation ---
StrategyFactory& StrategyFactory::instance() {
    static StrategyFactory factory_instance;
    return factory_instance;
}

void StrategyFactory::register_strategy(const std::string& name, CreateFunction create_func) {
    strategies_[name] = create_func;
}

std::unique_ptr<BaseStrategy> StrategyFactory::create_strategy(const std::string& name) {
    auto it = strategies_.find(name);
    if (it != strategies_.end()) {
        return it->second();
    }
    std::cerr << "Error: Strategy '" << name << "' not found in factory." << std::endl;
    return nullptr;
}

} // namespace sentio
```

## 📄 **FILE 54 of 92**: src/csv_loader.cpp

**File Information**:
- **Path**: `src/csv_loader.cpp`

- **Size**: 89 lines
- **Modified**: 2025-09-05 15:48:15

- **Type**: .cpp

```text
#include "sentio/csv_loader.hpp"
#include "sentio/binio.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctz/time_zone.h>
#include <cctz/civil_time.h>

namespace sentio {

bool load_csv(const std::string& filename, std::vector<Bar>& out) {
    // Try binary cache first for performance
    std::string bin_filename = filename.substr(0, filename.find_last_of('.')) + ".bin";
    auto cached = load_bin(bin_filename);
    if (!cached.empty()) {
        out = std::move(cached);
        return true;
    }
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string timestamp_str, symbol, open_str, high_str, low_str, close_str, volume_str;
        
        std::getline(ss, timestamp_str, ',');
        std::getline(ss, symbol, ',');
        std::getline(ss, open_str, ',');
        std::getline(ss, high_str, ',');
        std::getline(ss, low_str, ',');
        std::getline(ss, close_str, ',');
        std::getline(ss, volume_str, ',');
        
        Bar bar;
        bar.ts_utc = timestamp_str;
        
        // **MODIFIED**: Parse ISO 8601 timestamp using cctz
        try {
            // Parse the RFC3339 / ISO 8601 timestamp string (e.g., "2023-10-27T09:30:00-04:00")
            cctz::time_zone tz;
            if (cctz::load_time_zone("America/New_York", &tz)) {
                cctz::time_point<cctz::seconds> tp;
                if (cctz::parse("%Y-%m-%dT%H:%M:%S%Ez", timestamp_str, tz, &tp)) {
                    // Convert to UTC epoch seconds
                    bar.ts_nyt_epoch = tp.time_since_epoch().count();
                } else {
                    // Try alternative format
                    if (cctz::parse("%Y-%m-%d %H:%M:%S%Ez", timestamp_str, tz, &tp)) {
                        bar.ts_nyt_epoch = tp.time_since_epoch().count();
                    } else {
                        bar.ts_nyt_epoch = 0; // Could not parse
                    }
                }
            } else {
                bar.ts_nyt_epoch = 0; // Could not load timezone
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse timestamp '" << timestamp_str << "'. Error: " << e.what() << std::endl;
            bar.ts_nyt_epoch = 0;
        }
        
        try {
            bar.open = std::stod(open_str);
            bar.high = std::stod(high_str);
            bar.low = std::stod(low_str);
            bar.close = std::stod(close_str);
            bar.volume = std::stoull(volume_str);
            out.push_back(bar);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse bar data line: " << line << std::endl;
        }
    }
    
    // Save to binary cache for next time
    if (!out.empty()) {
        save_bin(bin_filename, out);
    }
    
    return true;
}

} // namespace sentio
```

## 📄 **FILE 55 of 92**: src/feature_health.cpp

**File Information**:
- **Path**: `src/feature_health.cpp`

- **Size**: 32 lines
- **Modified**: 2025-09-05 17:01:07

- **Type**: .cpp

```text
#include "sentio/feature_health.hpp"
#include <cmath>

namespace sentio {

FeatureHealthReport check_feature_health(const std::vector<PricePoint>& series,
                                         const FeatureHealthCfg& cfg) {
  FeatureHealthReport rep;
  if (series.empty()) return rep;

  for (std::size_t i=0;i<series.size();++i) {
    const auto& p = series[i];
    if (cfg.check_nan && !std::isfinite(p.close)) {
      rep.issues.push_back({p.ts_utc, "NaN", "non-finite close"});
    }
    if (cfg.check_monotonic_time && i>0) {
      if (series[i].ts_utc <= series[i-1].ts_utc) {
        rep.issues.push_back({p.ts_utc, "Backwards_TS", "non-increasing timestamp"});
      }
      if (cfg.expected_spacing_sec>0) {
        auto gap = series[i].ts_utc - series[i-1].ts_utc;
        if (gap != cfg.expected_spacing_sec) {
          rep.issues.push_back({p.ts_utc, "Gap",
              "expected "+std::to_string(cfg.expected_spacing_sec)+"s got "+std::to_string((long long)gap)+"s"});
        }
      }
    }
  }
  return rep;
}

} // namespace sentio

```

## 📄 **FILE 56 of 92**: src/main.cpp

**File Information**:
- **Path**: `src/main.cpp`

- **Size**: 181 lines
- **Modified**: 2025-09-05 20:13:14

- **Type**: .cpp

```text
#include "sentio/core.hpp"
#include "sentio/runner.hpp"
#include "sentio/wf.hpp"
#include "sentio/csv_loader.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/profiling.hpp"
#include "sentio/data_resolver.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/all_strategies.hpp"
#include "sentio/rth_calendar.hpp" // **NEW**: Include for RTH check
#include "sentio/calendar_seed.hpp" // **NEW**: Include for calendar creation

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <cstdlib> // For std::exit
#include <sstream>
#include <ctime> // For std::time


namespace { // Anonymous namespace to ensure link-time registration
    struct StrategyRegistrar {
        StrategyRegistrar() {
            auto force_link_vwap = std::make_unique<sentio::VWAPReversionStrategy>();
            auto force_link_momentum = std::make_unique<sentio::MomentumVolumeProfileStrategy>();
            auto force_link_bollinger = std::make_unique<sentio::BollingerSqueezeBreakoutStrategy>();
            auto force_link_opening = std::make_unique<sentio::OpeningRangeBreakoutStrategy>();
            auto force_link_scalping = std::make_unique<sentio::OrderFlowScalpingStrategy>();
            auto force_link_imbalance = std::make_unique<sentio::OrderFlowImbalanceStrategy>();
            auto force_link_market = std::make_unique<sentio::MarketMakingStrategy>();
        }
    };
    static StrategyRegistrar registrar;
}


void usage() {
    std::cout << "Usage: sentio_cli <command> [options]\n"
              << "Commands:\n"
              << "  backtest <symbol> [--strategy <name>] [--params <k=v,...>]\n"
              << "  wf <symbol> [--strategy <name>] [--params <k=v,...>]\n"
              << "  replay <run_id>\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string command = argv[1];
    
    if (command == "backtest") {
        if (argc < 3) {
            std::cout << "Usage: sentio_cli backtest <symbol> [--strategy <name>] [--params <k=v,...>]\n";
            return 1;
        }
        
        std::string base_symbol = argv[2];
        std::string strategy_name = "VWAPReversion";
        std::unordered_map<std::string, std::string> strategy_params;
        
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--strategy" && i + 1 < argc) {
                strategy_name = argv[++i];
            } else if (arg == "--params" && i + 1 < argc) {
                std::string params_str = argv[++i];
                std::stringstream ss(params_str);
                std::string pair;
                while (std::getline(ss, pair, ',')) {
                    size_t eq_pos = pair.find('=');
                    if (eq_pos != std::string::npos) {
                        strategy_params[pair.substr(0, eq_pos)] = pair.substr(eq_pos + 1);
                    }
                }
            }
        }
        
        sentio::SymbolTable ST;
        std::vector<std::vector<sentio::Bar>> series;
        std::vector<std::string> symbols_to_load = {base_symbol};
        if (base_symbol == "QQQ") {
            symbols_to_load.push_back("TQQQ");
            symbols_to_load.push_back("SQQQ");
        }

        std::cout << "Loading data for symbols: ";
        for(const auto& sym : symbols_to_load) std::cout << sym << " ";
        std::cout << std::endl;

        for (const auto& sym : symbols_to_load) {
            std::vector<sentio::Bar> bars;
            std::string data_path = sentio::resolve_csv(sym);
            if (!sentio::load_csv(data_path, bars)) {
                std::cerr << "ERROR: Failed to load data for " << sym << " from " << data_path << std::endl;
                continue;
            }
            std::cout << " -> Loaded " << bars.size() << " bars for " << sym << std::endl;
            
            int symbol_id = ST.intern(sym);
            if (static_cast<size_t>(symbol_id) >= series.size()) {
                series.resize(symbol_id + 1);
            }
            series[symbol_id] = std::move(bars);
        }
        
        int base_symbol_id = ST.get_id(base_symbol);
        if (series.empty() || series[base_symbol_id].empty()) {
            std::cerr << "FATAL: No data loaded for base symbol " << base_symbol << std::endl;
            return 1;
        }

        // **NEW**: Data Sanity Check - Verify RTH filtering post-load.
        // This acts as a safety net. If your data was generated with RTH filtering,
        // this check ensures the filtering was successful.
        std::cout << "\nVerifying data integrity for RTH..." << std::endl;
        bool rth_filter_failed = false;
        sentio::TradingCalendar calendar = sentio::make_default_nyse_calendar();
        for (size_t sid = 0; sid < series.size(); ++sid) {
            if (series[sid].empty()) continue;
            for (const auto& bar : series[sid]) {
                if (!calendar.is_rth_utc(bar.ts_nyt_epoch, "America/New_York")) {
                    std::cerr << "\nFATAL ERROR: Non-RTH data found after filtering!\n"
                              << " -> Symbol: " << ST.get_symbol(sid) << "\n"
                              << " -> Timestamp (UTC): " << bar.ts_utc << "\n"
                              << " -> NYT Epoch: " << bar.ts_nyt_epoch << "\n\n"
                              << "This indicates your data files (*.csv, *.bin) were generated with an old or incorrect RTH filter.\n"
                              << "Please DELETE your existing data files and REGENERATE them using the updated poly_fetch tool.\n"
                              << std::endl;
                    rth_filter_failed = true;
                    break;
                }
            }
            if (rth_filter_failed) break;
        }

        if (rth_filter_failed) {
            std::exit(1); // Exit with an error as requested
        }
        std::cout << " -> Data verification passed." << std::endl;

        sentio::RunnerCfg cfg;
        cfg.strategy_name = strategy_name;
        cfg.strategy_params = strategy_params;
        cfg.audit_level = sentio::AuditLevel::Full;
        cfg.snapshot_stride = 100;
        
        // Create audit recorder
        sentio::AuditConfig audit_cfg;
        audit_cfg.run_id = "backtest_" + base_symbol + "_" + std::to_string(std::time(nullptr));
        audit_cfg.file_path = "audit/backtest_" + base_symbol + ".jsonl";
        audit_cfg.flush_each = true;
        sentio::AuditRecorder audit(audit_cfg);
        
        sentio::Tsc timer;
        timer.tic();
        auto result = sentio::run_backtest(audit, ST, series, base_symbol_id, cfg);
        double elapsed = timer.toc_sec();
        
        std::cout << "\nBacktest completed in " << elapsed << "s\n";
        std::cout << "Final Equity: " << result.final_equity << "\n";
        std::cout << "Total Return: " << result.total_return << "%\n";
        std::cout << "Sharpe Ratio: " << result.sharpe_ratio << "\n";
        std::cout << "Max Drawdown: " << result.max_drawdown << "%\n";
        std::cout << "Total Fills: " << result.total_fills << "\n";
        std::cout << "Diagnostics -> No Route: " << result.no_route << " | No Quantity: " << result.no_qty << "\n";

    } else if (command == "wf") {
        std::cout << "Walk-forward command is not fully implemented in this example.\n";
    } else if (command == "replay") {
        std::cout << "Replay command is not fully implemented in this example.\n";
    } else {
        usage();
        return 1;
    }
    
    return 0;
}

```

## 📄 **FILE 57 of 92**: src/optimizer.cpp

**File Information**:
- **Path**: `src/optimizer.cpp`

- **Size**: 180 lines
- **Modified**: 2025-09-05 09:41:27

- **Type**: .cpp

```text
#include "sentio/optimizer.hpp"
#include "sentio/runner.hpp"
#include <iostream>

namespace sentio {

// RandomSearchOptimizer Implementation
std::vector<Parameter> RandomSearchOptimizer::get_parameter_space() {
    return {
        Parameter("test_param", 0.0, 1.0, 0.5)
    };
}

void RandomSearchOptimizer::apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) {
    for (const auto& param : params) {
        cfg.strategy_params[param.name] = std::to_string(param.current_value);
    }
}

OptimizationResult RandomSearchOptimizer::optimize([[maybe_unused]] const std::function<double(const RunResult&)>& objective_func,
                                                  [[maybe_unused]] const std::vector<Parameter>& param_space,
                                                  [[maybe_unused]] const OptimizationConfig& config) {
    OptimizationResult result;
    result.parameters["test"] = 0.5;
    result.objective_value = 0.0;
    return result;
}

// GridSearchOptimizer Implementation
std::vector<Parameter> GridSearchOptimizer::get_parameter_space() {
    return {
        Parameter("test_param", 0.0, 1.0, 0.5)
    };
}

void GridSearchOptimizer::apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) {
    for (const auto& param : params) {
        cfg.strategy_params[param.name] = std::to_string(param.current_value);
    }
}

OptimizationResult GridSearchOptimizer::optimize([[maybe_unused]] const std::function<double(const RunResult&)>& objective_func,
                                                [[maybe_unused]] const std::vector<Parameter>& param_space,
                                                [[maybe_unused]] const OptimizationConfig& config) {
    OptimizationResult result;
    result.parameters["test"] = 0.5;
    result.objective_value = 0.0;
    return result;
}

// BayesianOptimizer Implementation
std::vector<Parameter> BayesianOptimizer::get_parameter_space() {
    return {
        Parameter("test_param", 0.0, 1.0, 0.5)
    };
}

void BayesianOptimizer::apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) {
    for (const auto& param : params) {
        cfg.strategy_params[param.name] = std::to_string(param.current_value);
    }
}

OptimizationResult BayesianOptimizer::optimize([[maybe_unused]] const std::function<double(const RunResult&)>& objective_func,
                                              [[maybe_unused]] const std::vector<Parameter>& param_space,
                                              [[maybe_unused]] const OptimizationConfig& config) {
    OptimizationResult result;
    result.parameters["test"] = 0.5;
    result.objective_value = 0.0;
    return result;
}

// Strategy parameter creation functions
std::vector<Parameter> create_vwap_parameters() {
    return {
        Parameter("vwap_period", 100.0, 800.0, 200.0),
        Parameter("reversion_threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_momentum_parameters() {
    return {
        Parameter("momentum_period", 5.0, 50.0, 20.0),
        Parameter("threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_volatility_parameters() {
    return {
        Parameter("volatility_period", 10.0, 50.0, 20.0),
        Parameter("threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_bollinger_squeeze_parameters() {
    return {
        Parameter("bb_period", 10.0, 40.0, 20.0),
        Parameter("bb_std", 1.0, 3.0, 2.0)
    };
}

std::vector<Parameter> create_opening_range_parameters() {
    return {
        Parameter("range_minutes", 15.0, 60.0, 30.0),
        Parameter("breakout_threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_order_flow_scalping_parameters() {
    return {
        Parameter("imbalance_period", 5.0, 40.0, 20.0),
        Parameter("threshold", 0.4, 0.9, 0.7)
    };
}

std::vector<Parameter> create_order_flow_imbalance_parameters() {
    return {
        Parameter("lookback_window", 5.0, 50.0, 20.0),
        Parameter("threshold", 0.5, 3.0, 1.5)
    };
}

std::vector<Parameter> create_market_making_parameters() {
    return {
        Parameter("base_spread", 0.0002, 0.003, 0.001),
        Parameter("order_levels", 1.0, 5.0, 3.0)
    };
}

std::vector<Parameter> create_router_parameters() {
    return {
        Parameter("t1", 0.01, 0.3, 0.05),
        Parameter("t2", 0.1, 0.6, 0.3)
    };
}

std::vector<Parameter> create_parameters_for_strategy(const std::string& strategy_name) {
    if (strategy_name == "VWAPReversion") {
        return create_vwap_parameters();
    } else if (strategy_name == "MomentumVolumeProfile") {
        return create_momentum_parameters();
    } else if (strategy_name == "VolatilityExpansion") {
        return create_volatility_parameters();
    } else if (strategy_name == "BollingerSqueezeBreakout") {
        return create_bollinger_squeeze_parameters();
    } else if (strategy_name == "OpeningRangeBreakout") {
        return create_opening_range_parameters();
    } else if (strategy_name == "OrderFlowScalping") {
        return create_order_flow_scalping_parameters();
    } else if (strategy_name == "OrderFlowImbalance") {
        return create_order_flow_imbalance_parameters();
    } else if (strategy_name == "MarketMaking") {
        return create_market_making_parameters();
    } else {
        return create_router_parameters();
    }
}

std::vector<Parameter> create_full_parameter_space() {
    return create_router_parameters();
}

// OptimizationEngine implementation
OptimizationEngine::OptimizationEngine(const std::string& optimizer_type) {
    if (optimizer_type == "grid") {
        optimizer = std::make_unique<GridSearchOptimizer>();
    } else if (optimizer_type == "bayesian") {
        optimizer = std::make_unique<BayesianOptimizer>();
    } else {
        optimizer = std::make_unique<RandomSearchOptimizer>();
    }
}

OptimizationResult OptimizationEngine::run_optimization(const std::string& strategy_name,
                                                       const std::function<double(const RunResult&)>& objective_func) {
    auto param_space = create_parameters_for_strategy(strategy_name);
    return optimizer->optimize(objective_func, param_space, config);
}

} // namespace sentio

```

## 📄 **FILE 58 of 92**: src/pnl_accounting.cpp

**File Information**:
- **Path**: `src/pnl_accounting.cpp`

- **Size**: 62 lines
- **Modified**: 2025-09-05 15:28:49

- **Type**: .cpp

```text
// src/pnl_accounting.cpp
#include "sentio/pnl_accounting.hpp"
#include <shared_mutex>
#include <unordered_map>
#include <utility>

namespace sentio {

namespace {
// Simple in-TU storage for latest bars keyed by instrument.
// If your header already declares members, remove this and
// implement against those members instead.
struct PriceBookStorage {
    std::unordered_map<std::string, Bar> latest;
    mutable std::shared_mutex mtx;
};

// One global storage per process for the default PriceBook()
// If your PriceBook is an instance with its own fields, move this inside the class.
static PriceBookStorage& storage() {
    static PriceBookStorage s;
    return s;
}
} // anonymous namespace

// ---- Interface implementation ----

const Bar* PriceBook::get_latest(const std::string& instrument) const {
    auto& S = storage();
    std::shared_lock lk(S.mtx);
    auto it = S.latest.find(instrument);
    if (it == S.latest.end()) return nullptr;
    return &it->second; // pointer remains valid as long as map bucket survives
}

// ---- Additional helper methods (not declared in header but useful) ----

void PriceBook::upsert_latest(const std::string& instrument, const Bar& b) {
    auto& S = storage();
    std::unique_lock lk(S.mtx);
    S.latest[instrument] = b;
}

bool PriceBook::has_instrument(const std::string& instrument) const {
    auto& S = storage();
    std::shared_lock lk(S.mtx);
    return S.latest.find(instrument) != S.latest.end();
}

std::size_t PriceBook::size() const {
    auto& S = storage();
    std::shared_lock lk(S.mtx);
    return S.latest.size();
}

double last_trade_price(const PriceBook& book, const std::string& instrument) {
    auto* b = book.get_latest(instrument);
    if (!b) throw std::runtime_error("No bar for instrument: " + instrument);
    return b->close;
}

} // namespace sentio

```

## 📄 **FILE 59 of 92**: src/poly_fetch_main.cpp

**File Information**:
- **Path**: `src/poly_fetch_main.cpp`

- **Size**: 57 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .cpp

```text
#include "sentio/polygon_client.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

using namespace sentio;

int main(int argc,char**argv){
  if(argc<5){
    std::cerr<<"Usage: poly_fetch FAMILY from to outdir [--timespan day|hour|minute] [--multiplier N] [--symbols SYM1,SYM2,...] [--rth]\n";
    return 1;
  }
  std::string fam=argv[1], from=argv[2], to=argv[3], outdir=argv[4];
  std::string timespan = "day";
  int multiplier = 1;
  std::string symbols_csv;
  bool rth_only=false;
  bool exclude_holidays=false;
  for (int i=5;i<argc;i++) {
    std::string a = argv[i];
    if ((a=="--timespan" || a=="-t") && i+1<argc) { timespan = argv[++i]; }
    else if ((a=="--multiplier" || a=="-m") && i+1<argc) { multiplier = std::stoi(argv[++i]); }
    else if (a=="--symbols" && i+1<argc) { symbols_csv = argv[++i]; }
    else if (a=="--rth") { rth_only=true; }
    else if (a=="--no-holidays") { exclude_holidays=true; }
  }
  const char* key = std::getenv("POLYGON_API_KEY");
  std::string api_key = key? key: "";
  PolygonClient cli(api_key);

  std::vector<std::string> syms;
  if(fam=="qqq") syms={"QQQ","TQQQ","SQQQ","PSQ"};
  else if(fam=="bitcoin") syms={"X:BTCUSD","X:ETHUSD"};
  else if(fam=="tesla") syms={"TSLA","TSLQ"};
  else if(fam=="custom") {
    if (symbols_csv.empty()) { std::cerr<<"--symbols required for custom family\n"; return 1; }
    size_t start=0; while (start < symbols_csv.size()) {
      size_t pos = symbols_csv.find(',', start);
      std::string tok = (pos==std::string::npos)? symbols_csv.substr(start) : symbols_csv.substr(start, pos-start);
      if (!tok.empty()) syms.push_back(tok);
      if (pos==std::string::npos) break; else start = pos+1;
    }
  } else { std::cerr<<"Unknown family\n"; return 1; }

  for(auto&s:syms){
    AggsQuery q; q.symbol=s; q.from=from; q.to=to; q.timespan=timespan; q.multiplier=multiplier; q.adjusted=true; q.sort="asc";
    auto bars=cli.get_aggs_all(q);
    std::string suffix;
    if (rth_only) suffix += "_RTH";
    if (exclude_holidays) suffix += "_NH";
    std::string fname= outdir + "/" + s + suffix + ".csv";
    cli.write_csv(fname,s,bars,rth_only,exclude_holidays);
    std::cerr<<"Wrote "<<bars.size()<<" bars -> "<<fname<<"\n";
  }
}


```

## 📄 **FILE 60 of 92**: src/polygon_client.cpp

**File Information**:
- **Path**: `src/polygon_client.cpp`

- **Size**: 207 lines
- **Modified**: 2025-09-05 15:51:59

- **Type**: .cpp

```text
#include "sentio/polygon_client.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <cctz/time_zone.h>
#include <cctz/civil_time.h>
#include <fstream>
#include <thread>
#include <chrono>
#include <sstream>
#include <iomanip>

using json = nlohmann::json;
namespace sentio {

static size_t write_cb(void* contents, size_t size, size_t nmemb, void* userp) {
  size_t total = size * nmemb;
  std::string* s = static_cast<std::string*>(userp);
  s->append(static_cast<char*>(contents), total);
  return total;
}

static std::string rfc3339_nyc_from_epoch_ms(long long ms) {
  using namespace std::chrono;
  cctz::time_zone tz;
  if (!cctz::load_time_zone("America/New_York", &tz)) {
    return "1970-01-01T00:00:00+00:00"; // fallback
  }
  
  cctz::time_point<cctz::seconds> tp{cctz::seconds{ms / 1000}};
  auto lt = cctz::convert(tp, tz);
  auto ct = cctz::civil_second(lt);
  
  std::ostringstream oss;
  oss << std::setfill('0') 
      << std::setw(4) << ct.year() << "-"
      << std::setw(2) << ct.month() << "-"
      << std::setw(2) << ct.day() << "T"
      << std::setw(2) << ct.hour() << ":"
      << std::setw(2) << ct.minute() << ":"
      << std::setw(2) << ct.second() << "-04:00"; // EDT offset
  
  return oss.str();
}

// **NEW**: Robust RTH check directly from a UTC timestamp.
// This avoids error-prone string parsing and ensures correct time zone handling.
static bool is_rth_nyc_from_utc_ms(long long utc_ms) {
    cctz::time_zone tz;
    if (!cctz::load_time_zone("America/New_York", &tz)) {
        return false;
    }
    
    cctz::time_point<cctz::seconds> tp{cctz::seconds{utc_ms / 1000}};
    auto lt = cctz::convert(tp, tz);
    auto ct = cctz::civil_second(lt);
    
    // Check if weekend (Saturday = 6, Sunday = 0)
    auto wd = cctz::get_weekday(ct);
    if (wd == cctz::weekday::saturday || wd == cctz::weekday::sunday) {
        return false;
    }
    
    // Check if RTH (9:30 AM - 4:00 PM ET)
    if (ct.hour() < 9 || (ct.hour() == 9 && ct.minute() < 30)) {
        return false;  // Before 9:30 AM
    }
    if (ct.hour() >= 16) {
        return false;  // After 4:00 PM
    }
    
    return true;
}

static bool is_us_market_holiday_2022_2025(int year, int month, int day) {
  // Simple holiday check for common US market holidays
  // This is a simplified version - for production use, integrate with the full calendar system
  
  // New Year's Day (observed)
  if (month == 1 && day == 1) return true;
  if (month == 1 && day == 2) return true; // observed if Jan 1 is Sunday
  
  // MLK Day (3rd Monday in January)
  if (month == 1 && day >= 15 && day <= 21) {
    // Simple check - this could be more precise
    return true;
  }
  
  // Presidents Day (3rd Monday in February)
  if (month == 2 && day >= 15 && day <= 21) {
    return true;
  }
  
  // Good Friday (varies by year)
  if (year == 2022 && month == 4 && day == 15) return true;
  if (year == 2023 && month == 4 && day == 7) return true;
  if (year == 2024 && month == 3 && day == 29) return true;
  if (year == 2025 && month == 4 && day == 18) return true;
  
  // Memorial Day (last Monday in May)
  if (month == 5 && day >= 25 && day <= 31) {
    return true;
  }
  
  // Juneteenth (observed)
  if (month == 6 && day == 19) return true;
  if (month == 6 && day == 20) return true; // observed if Jun 19 is Sunday
  
  // Independence Day (observed)
  if (month == 7 && day == 4) return true;
  if (month == 7 && day == 5) return true; // observed if Jul 4 is Sunday
  
  // Labor Day (1st Monday in September)
  if (month == 9 && day >= 1 && day <= 7) {
    return true;
  }
  
  // Thanksgiving (4th Thursday in November)
  if (month == 11 && day >= 22 && day <= 28) {
    return true;
  }
  
  // Christmas (observed)
  if (month == 12 && day == 25) return true;
  if (month == 12 && day == 26) return true; // observed if Dec 25 is Sunday
  
  return false;
}

PolygonClient::PolygonClient(std::string api_key) : api_key_(std::move(api_key)) {}

std::string PolygonClient::get_(const std::string& url) {
    // This function's implementation remains unchanged.
    CURL* curl = curl_easy_init();
    std::string buffer;
    if (!curl) return buffer;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    struct curl_slist* headers = nullptr;
    std::string auth = "Authorization: Bearer " + api_key_;
    headers = curl_slist_append(headers, auth.c_str());
    headers = curl_slist_append(headers, "Accept: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return buffer;
}

std::vector<AggBar> PolygonClient::get_aggs_all(const AggsQuery& q, int max_pages) {
    // This function's implementation remains unchanged.
    std::vector<AggBar> out;
    std::string base = "https://api.polygon.io/v2/aggs/ticker/" + q.symbol + "/range/" + std::to_string(q.multiplier) + "/" + q.timespan + "/" + q.from + "/" + q.to + "?adjusted=" + (q.adjusted?"true":"false") + "&sort=" + q.sort + "&limit=" + std::to_string(q.limit);
    std::string url = base;
    for (int page=0; page<max_pages; ++page) {
        std::string body = get_(url);
        if (body.empty()) break;
        auto j = json::parse(body, nullptr, false);
        if (j.is_discarded()) break;
        if (j.contains("results")) {
            for (auto& r : j["results"]) {
                out.push_back({r.value("t", 0LL), r.value("o", 0.0), r.value("h", 0.0), r.value("l", 0.0), r.value("c", 0.0), r.value("v", 0.0)});
            }
        }
        if (!j.contains("next_url")) break;
        url = j["next_url"].get<std::string>();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return out;
}

void PolygonClient::write_csv(const std::string& out_path,const std::string& symbol,
                              const std::vector<AggBar>& bars, bool rth_only, bool exclude_holidays) {
  std::ofstream f(out_path);
  f << "timestamp,symbol,open,high,low,close,volume\n";
  for (auto& a: bars) {
    // **MODIFIED**: RTH and holiday filtering is now done directly on the UTC timestamp
    // before any string conversion, making it much more reliable.

    if (rth_only && !is_rth_nyc_from_utc_ms(a.ts_ms)) {
        continue;
    }
    
    if (exclude_holidays) {
        cctz::time_zone tz;
        if (cctz::load_time_zone("America/New_York", &tz)) {
            cctz::time_point<cctz::seconds> tp{cctz::seconds{a.ts_ms / 1000}};
            auto lt = cctz::convert(tp, tz);
            auto ct = cctz::civil_second(lt);
            
            if (is_us_market_holiday_2022_2025(ct.year(), ct.month(), ct.day())) {
                continue;
            }
        }
    }
    
    // The timestamp is converted to a NYT string only for writing to the CSV
    std::string ts_str = rfc3339_nyc_from_epoch_ms(a.ts_ms);

    f << ts_str << ',' << symbol << ','
      << a.open << ',' << a.high << ',' << a.low << ',' << a.close << ',' << a.volume << '\n';
  }
}

} // namespace sentio

```

## 📄 **FILE 61 of 92**: src/polygon_ingest.cpp

**File Information**:
- **Path**: `src/polygon_ingest.cpp`

- **Size**: 68 lines
- **Modified**: 2025-09-05 15:29:30

- **Type**: .cpp

```text
#include "sentio/polygon_ingest.hpp"
#include "sentio/time_utils.hpp"
#include "sentio/rth_calendar.hpp"
#include "sentio/calendar_seed.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace sentio {

static const TradingCalendar& nyse_calendar() {
  static TradingCalendar cal = make_default_nyse_calendar();
  return cal;
}

static inline bool accept_bar_utc(std::int64_t ts_utc) {
  return nyse_calendar().is_rth_utc(ts_utc, "America/New_York");
}

static inline bool valid_bar(const ProviderBar& b) {
  if (!std::isfinite(b.open)  || !std::isfinite(b.high) ||
      !std::isfinite(b.low)   || !std::isfinite(b.close)||
      !std::isfinite(b.volume)) return false;
  if (b.open <= 0 || b.high <= 0 || b.low <= 0 || b.close <= 0) return false;
  if (b.low > b.high) return false;
  return !b.symbol.empty();
}

static inline void upsert_bar(PriceBook& book, const std::string& instrument, const ProviderBar& pb) {
  Bar b;
  b.open  = pb.open;
  b.high  = pb.high;
  b.low   = pb.low;
  b.close = pb.close;
  book.upsert_latest(instrument, b);
}

static inline std::int64_t to_epoch_s(const std::variant<std::int64_t, double, std::string>& ts) {
  auto tp = to_utc_sys_seconds(ts);
  return std::chrono::time_point_cast<std::chrono::seconds>(tp).time_since_epoch().count();
}

std::size_t ingest_provider_bars(const std::vector<ProviderBar>& input, PriceBook& book) {
  std::size_t accepted = 0;
  for (const auto& pb : input) {
    if (!valid_bar(pb)) continue;

    std::int64_t ts_utc{};
    try {
      ts_utc = to_epoch_s(pb.ts);
    } catch (...) {
      continue; // skip malformed time
    }

    if (!accept_bar_utc(ts_utc)) continue;

    upsert_bar(book, pb.symbol, pb);
    ++accepted;
  }
  return accepted;
}

bool ingest_provider_bar(const ProviderBar& bar, PriceBook& book) {
  return ingest_provider_bars(std::vector<ProviderBar>{bar}, book) == 1;
}

} // namespace sentio
```

## 📄 **FILE 62 of 92**: src/router.cpp

**File Information**:
- **Path**: `src/router.cpp`

- **Size**: 92 lines
- **Modified**: 2025-09-05 15:29:30

- **Type**: .cpp

```text
#include "sentio/router.hpp"
#include "sentio/polygon_ingest.hpp" // for PriceBook forward impl
#include <algorithm>
#include <cassert>
#include <cmath>

namespace sentio {

static inline int dir_from(const StrategySignal& s) {
  using T = StrategySignal::Type;
  if (s.type==T::BUY || s.type==T::STRONG_BUY) return +1;
  if (s.type==T::SELL|| s.type==T::STRONG_SELL) return -1;
  return 0;
}
static inline bool is_strong(const StrategySignal& s) {
  using T = StrategySignal::Type;
  return s.type==T::STRONG_BUY || s.type==T::STRONG_SELL || s.confidence>=0.90;
}
static inline double clamp(double x,double lo,double hi){ return std::max(lo,std::min(hi,x)); }

static inline std::string map_instrument_qqq_family(bool go_long, bool strong,
                                                    const RouterCfg& cfg,
                                                    const std::string& base_symbol)
{
  if (base_symbol == cfg.base_symbol) {
    if (go_long)   return strong ? cfg.bull3x : cfg.base_symbol;
    else           return strong ? cfg.bear3x : cfg.bear1x;
  }
  // Unknown family: fall back to base
  return base_symbol;
}

std::optional<RouteDecision> route(const StrategySignal& s, const RouterCfg& cfg, const std::string& base_symbol) {
  int d = dir_from(s);
  if (d==0) return std::nullopt;
  if (s.confidence < cfg.min_signal_strength) return std::nullopt;

  const bool strong = is_strong(s);
  const bool go_long = (d>0);
  const std::string instrument = map_instrument_qqq_family(go_long, strong, cfg, base_symbol);

  const double raw = (d>0?+1.0:-1.0) * (s.confidence * cfg.signal_multiplier);
  const double tw  = clamp(raw, -cfg.max_position_pct, +cfg.max_position_pct);

  return RouteDecision{instrument, tw};
}

// Implemented elsewhere in your codebase; declared in router.hpp.
// Here's a weak reference for clarity:
// double last_trade_price(const PriceBook&, const std::string&);

static inline double round_to_lot(double qty, double lot) {
  if (lot <= 0) return std::floor(qty);
  return std::floor(qty / lot) * lot;
}

Order route_and_create_order(const std::string& signal_id,
                             const StrategySignal& sig,
                             const RouterCfg& cfg,
                             const std::string& base_symbol,
                             const PriceBook& book,
                             const AccountSnapshot& acct,
                             std::int64_t ts_utc)
{
  Order o{};
  o.signal_id = signal_id;
  auto rd = route(sig, cfg, base_symbol);
  if (!rd) return o; // qty 0

  o.instrument = rd->instrument;
  o.side = (rd->target_weight >= 0 ? OrderSide::Buy : OrderSide::Sell);

  // Size by equity * |target_weight|
  double px = last_trade_price(book, o.instrument); // must be routed instrument
  if (!(std::isfinite(px) && px > 0.0)) return o;

  double target_notional = std::abs(rd->target_weight) * acct.equity;
  double raw_qty = target_notional / px;
  double lot = (cfg.lot_size>0 ? cfg.lot_size : 1.0);
  double qty = round_to_lot(raw_qty, lot);
  if (qty < cfg.min_shares) return o;

  o.qty = qty;
  o.limit_price = 0.0; // market
  o.ts_utc = ts_utc;
  o.notional = (o.side==OrderSide::Buy ? +1.0 : -1.0) * px * qty;

  assert(!o.instrument.empty() && "Instrument must be set");
  return o;
}

} // namespace sentio
```

## 📄 **FILE 63 of 92**: src/rth_calendar.cpp

**File Information**:
- **Path**: `src/rth_calendar.cpp`

- **Size**: 49 lines
- **Modified**: 2025-09-05 16:02:04

- **Type**: .cpp

```text
// rth_calendar.cpp
#include "sentio/rth_calendar.hpp"
#include <chrono>
#include <string>
#include <stdexcept>
#include <iostream>
#include "cctz/time_zone.h"
#include "cctz/civil_time.h"

using namespace std::chrono;

namespace sentio {

bool TradingCalendar::is_rth_utc(std::int64_t ts_utc, const std::string& tz_name) const {
  using namespace std::chrono;
  
  // Convert to cctz time
  cctz::time_point<cctz::seconds> tp{cctz::seconds{ts_utc}};
  
  // Get timezone
  cctz::time_zone tz;
  if (!cctz::load_time_zone(tz_name, &tz)) {
    return false;
  }
  
  // Convert to local time
  auto lt = cctz::convert(tp, tz);
  
  // Extract civil time components
  auto ct = cctz::civil_second(lt);
  
  // Check if weekend (Saturday = 6, Sunday = 0)
  auto wd = cctz::get_weekday(ct);
  if (wd == cctz::weekday::saturday || wd == cctz::weekday::sunday) {
    return false;
  }
  
  // Check if RTH (9:30 AM - 4:00 PM ET inclusive)
  if (ct.hour() < 9 || (ct.hour() == 9 && ct.minute() < 30)) {
    return false;  // Before 9:30 AM
  }
  if (ct.hour() > 16 || (ct.hour() == 16 && ct.minute() > 0)) {
    return false;  // After 4:00 PM
  }
  
  return true;
}

} // namespace sentio
```

## 📄 **FILE 64 of 92**: src/runner.cpp

**File Information**:
- **Path**: `src/runner.cpp`

- **Size**: 161 lines
- **Modified**: 2025-09-05 20:13:14

- **Type**: .cpp

```text
#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/pricebook.hpp"
#include "sentio/metrics.hpp"
#include "sentio/sizer.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>

namespace sentio {

RunResult run_backtest(AuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                      int base_symbol_id, const RunnerCfg& cfg) {
    
    // 1. ============== INITIALIZATION ==============
    RunResult result{};
    
    // Start audit run
    std::string meta = "{";
    meta += "\"strategy\":\"" + cfg.strategy_name + "\",";
    meta += "\"base_symbol_id\":" + std::to_string(base_symbol_id) + ",";
    meta += "\"total_series\":" + std::to_string(series.size()) + ",";
    meta += "\"base_series_size\":" + std::to_string(series[base_symbol_id].size());
    meta += "}";
    audit.event_run_start(series[base_symbol_id][0].ts_nyt_epoch, meta);
    
    auto strategy = StrategyFactory::instance().create_strategy(cfg.strategy_name);
    if (!strategy) {
        std::cerr << "FATAL: Could not create strategy '" << cfg.strategy_name << "'. Check registration." << std::endl;
        return result;
    }
    
    ParameterMap params;
    for (const auto& [key, value] : cfg.strategy_params) {
        try {
            params[key] = std::stod(value);
        } catch (...) { /* ignore */ }
    }
    strategy->set_params(params);

    Portfolio portfolio(ST.size());
    AdvancedSizer sizer;
    Pricebook pricebook(base_symbol_id, ST, series);
    
    std::vector<std::pair<std::string, double>> equity_curve;
    const auto& base_series = series[base_symbol_id];
    equity_curve.reserve(base_series.size());

    int total_fills = 0;
    int no_route_count = 0;
    int no_qty_count = 0;

    // 2. ============== MAIN EVENT LOOP ==============
    for (size_t i = 0; i < base_series.size(); ++i) {
        const auto& bar = base_series[i];
        pricebook.sync_to_base_i(i);
        
        // Log bar data
        AuditBar audit_bar{bar.open, bar.high, bar.low, bar.close, static_cast<double>(bar.volume)};
        audit.event_bar(bar.ts_nyt_epoch, ST.get_symbol(base_symbol_id), audit_bar);
        
        StrategySignal sig = strategy->calculate_signal(base_series, i);
        
        if (sig.type != StrategySignal::Type::HOLD) {
            // Log signal
            SigType sig_type = static_cast<SigType>(static_cast<int>(sig.type));
            audit.event_signal(bar.ts_nyt_epoch, ST.get_symbol(base_symbol_id), sig_type, sig.confidence);
            
            auto route_decision = route(sig, cfg.router, ST.get_symbol(base_symbol_id));

            if (route_decision) {
                // Log route decision
                audit.event_route(bar.ts_nyt_epoch, ST.get_symbol(base_symbol_id), route_decision->instrument, route_decision->target_weight);
                
                int instrument_id = ST.get_id(route_decision->instrument);
                if (instrument_id != -1) {
                    double instrument_price = pricebook.last_px[instrument_id];

                    if (instrument_price > 0) {
                        [[maybe_unused]] double current_equity = equity_mark_to_market(portfolio, pricebook.last_px);
                        
                        // **MODIFIED**: This is the core logic fix.
                        // We calculate the desired final position size, then determine the needed trade quantity.
                        double target_qty = sizer.calculate_target_quantity(portfolio, ST, pricebook.last_px, 
                                                                             route_decision->instrument, route_decision->target_weight, 
                                                                             series[instrument_id], cfg.sizer);
                        
                        double current_qty = portfolio.positions[instrument_id].qty;
                        double trade_qty = target_qty - current_qty; // The actual amount to trade

                        if (std::abs(trade_qty * instrument_price) > 1.0) { // Min trade notional $1
                            // Log order and fill
                            Side side = (trade_qty > 0) ? Side::Buy : Side::Sell;
                            audit.event_order(bar.ts_nyt_epoch, route_decision->instrument, side, std::abs(trade_qty), 0.0);
                            audit.event_fill(bar.ts_nyt_epoch, route_decision->instrument, instrument_price, std::abs(trade_qty), 0.0, side);
                            
                            apply_fill(portfolio, instrument_id, trade_qty, instrument_price);
                            total_fills++; // **CRITICAL FIX**: Increment the fills counter
                        } else {
                            no_qty_count++;
                        }
                    }
                }
            } else {
                no_route_count++;
            }
        }
        
        // 3. ============== SNAPSHOT ==============
        if (i % cfg.snapshot_stride == 0 || i == base_series.size() - 1) {
            double current_equity = equity_mark_to_market(portfolio, pricebook.last_px);
            equity_curve.emplace_back(bar.ts_utc, current_equity);
            
            // Log account snapshot
            AccountState state;
            state.cash = portfolio.cash;
            state.equity = current_equity;
            state.realized = 0.0; // TODO: Calculate realized P&L
            audit.event_snapshot(bar.ts_nyt_epoch, state);
        }
    }
    
    // 4. ============== METRICS & DIAGNOSTICS ==============
    strategy->get_diag().print(strategy->get_name().c_str());

    if (equity_curve.empty()) {
        return result;
    }
    
    // **CRITICAL FIX**: Pass the correct `total_fills` to the metrics calculator.
    auto summary = compute_metrics_day_aware(equity_curve, total_fills);

    result.final_equity = equity_curve.empty() ? 100000.0 : equity_curve.back().second;
    result.total_return = summary.ret_total * 100.0;
    result.sharpe_ratio = summary.sharpe;
    result.max_drawdown = summary.mdd * 100.0;
    result.total_fills = summary.trades;
    result.no_route = no_route_count;
    result.no_qty = no_qty_count;

    // Log final metrics and end run
    std::int64_t end_ts = equity_curve.empty() ? series[base_symbol_id][0].ts_nyt_epoch : series[base_symbol_id].back().ts_nyt_epoch;
    audit.event_metric(end_ts, "final_equity", result.final_equity);
    audit.event_metric(end_ts, "total_return", result.total_return);
    audit.event_metric(end_ts, "sharpe_ratio", result.sharpe_ratio);
    audit.event_metric(end_ts, "max_drawdown", result.max_drawdown);
    audit.event_metric(end_ts, "total_fills", result.total_fills);
    audit.event_metric(end_ts, "no_route", result.no_route);
    audit.event_metric(end_ts, "no_qty", result.no_qty);
    
    std::string end_meta = "{";
    end_meta += "\"final_equity\":" + std::to_string(result.final_equity) + ",";
    end_meta += "\"total_return\":" + std::to_string(result.total_return) + ",";
    end_meta += "\"sharpe_ratio\":" + std::to_string(result.sharpe_ratio);
    end_meta += "}";
    audit.event_run_end(end_ts, end_meta);

    return result;
}

} // namespace sentio
```

## 📄 **FILE 65 of 92**: src/signal_engine.cpp

**File Information**:
- **Path**: `src/signal_engine.cpp`

- **Size**: 29 lines
- **Modified**: 2025-09-05 17:03:00

- **Type**: .cpp

```text
#include "sentio/signal_engine.hpp"

namespace sentio {

SignalEngine::SignalEngine(IStrategy* strat, const GateCfg& gate_cfg, SignalHealth* health)
: strat_(strat), gate_(gate_cfg, health), health_(health) {}

EngineOut SignalEngine::on_bar(const StrategyCtx& ctx, const Bar& b, bool inputs_finite) {
  strat_->on_bar(ctx, b);
  auto raw = strat_->latest();
  if (!raw.has_value()) {
    if (health_) health_->incr_drop(DropReason::NONE);
    return {std::nullopt, DropReason::NONE};
  }

  double conf = raw->confidence;
  auto conf2 = gate_.accept(ctx.ts_utc_epoch, ctx.is_rth, inputs_finite,
                            /*warmed_up=*/true, conf);
  if (!conf2) {
    // SignalGate already tallied reason; we return NONE to avoid double counting specific reason here.
    return {std::nullopt, DropReason::NONE};
  }

  StrategySignal out = *raw;
  out.confidence = *conf2;
  return {out, DropReason::NONE};
}

} // namespace sentio

```

## 📄 **FILE 66 of 92**: src/signal_gate.cpp

**File Information**:
- **Path**: `src/signal_gate.cpp`

- **Size**: 52 lines
- **Modified**: 2025-09-05 17:03:11

- **Type**: .cpp

```text
#include "sentio/signal_gate.hpp"
#include <cmath>
#include <algorithm>

namespace sentio {

SignalHealth::SignalHealth() {
  for (auto r :
       {DropReason::NONE, DropReason::NOT_RTH, DropReason::WARMUP,
        DropReason::NAN_INPUT, DropReason::THRESHOLD_TOO_TIGHT, DropReason::COOLDOWN_ACTIVE,
        DropReason::DUPLICATE_BAR_TS}) {
    by_reason.emplace(r, 0ULL);
  }
}
void SignalHealth::incr_emit(){ emitted.fetch_add(1, std::memory_order_relaxed); }
void SignalHealth::incr_drop(DropReason r){
  dropped.fetch_add(1, std::memory_order_relaxed);
  auto it = by_reason.find(r);
  if (it != by_reason.end()) it->second.fetch_add(1, std::memory_order_relaxed);
}

SignalGate::SignalGate(const GateCfg& cfg, SignalHealth* health)
: cfg_(cfg), health_(health) {}

std::optional<double> SignalGate::accept(std::int64_t ts_utc_epoch,
                                         bool is_rth,
                                         bool inputs_finite,
                                         bool warmed_up,
                                         double conf)
{
  if (cfg_.require_rth && !is_rth) { if (health_) health_->incr_drop(DropReason::NOT_RTH); return std::nullopt; }
  if (!inputs_finite)              { if (health_) health_->incr_drop(DropReason::NAN_INPUT); return std::nullopt; }
  if (!warmed_up)                  { if (health_) health_->incr_drop(DropReason::WARMUP);    return std::nullopt; }
  if (!(std::isfinite(conf)))      { if (health_) health_->incr_drop(DropReason::NAN_INPUT); return std::nullopt; }

  conf = std::clamp(conf, 0.0, 1.0);

  if (conf < cfg_.min_conf)        { if (health_) health_->incr_drop(DropReason::THRESHOLD_TOO_TIGHT); return std::nullopt; }

  // Cooldown (optional)
  if (cooldown_left_ > 0)          { --cooldown_left_; if (health_) health_->incr_drop(DropReason::COOLDOWN_ACTIVE); return std::nullopt; }

  // Debounce duplicate timestamps
  if (last_emit_ts_ == ts_utc_epoch){ if (health_) health_->incr_drop(DropReason::DUPLICATE_BAR_TS); return std::nullopt; }

  last_emit_ts_ = ts_utc_epoch;
  cooldown_left_ = cfg_.cooldown_bars;
  if (health_) health_->incr_emit();
  return conf;
}

} // namespace sentio

```

## 📄 **FILE 67 of 92**: src/signal_pipeline.cpp

**File Information**:
- **Path**: `src/signal_pipeline.cpp`

- **Size**: 43 lines
- **Modified**: 2025-09-05 17:05:46

- **Type**: .cpp

```text
#include "sentio/signal_pipeline.hpp"
#include <cmath>

namespace sentio {

PipelineOut SignalPipeline::on_bar(const StrategyCtx& ctx, const Bar& b, const void* acct) {
  (void)acct; // Avoid unused parameter warning
  strat_->on_bar(ctx, b);
  PipelineOut out{};
  TraceRow tr{};
  tr.ts_utc = ctx.ts_utc_epoch;
  tr.instrument = ctx.instrument;
  tr.close = b.close;
  tr.is_rth = ctx.is_rth;
  tr.inputs_finite = std::isfinite(b.close);

  auto sig = strat_->latest();
  if (!sig) {
    tr.reason = TraceReason::NO_STRATEGY_OUTPUT;
    if (trace_) trace_->push(tr);
    return out;
  }
  tr.confidence = sig->confidence;

  // Use the existing signal_gate API
  auto conf2 = gate_.accept(ctx.ts_utc_epoch, ctx.is_rth, tr.inputs_finite, true, sig->confidence);
  if (!conf2) {
    tr.reason = TraceReason::THRESHOLD_TOO_TIGHT; // Default to threshold for now
    if (trace_) trace_->push(tr);
    return out;
  }

  StrategySignal sig2 = *sig; sig2.confidence = *conf2;
  tr.conf_after_gate = *conf2;
  out.signal = sig2;

  // For now, just mark as OK since we don't have full routing implemented
  tr.reason = TraceReason::OK;
  if (trace_) trace_->push(tr);
  return out;
}

} // namespace sentio

```

## 📄 **FILE 68 of 92**: src/signal_trace.cpp

**File Information**:
- **Path**: `src/signal_trace.cpp`

- **Size**: 7 lines
- **Modified**: 2025-09-05 17:00:56

- **Type**: .cpp

```text
#include "sentio/signal_trace.hpp"

namespace sentio {
std::size_t SignalTrace::count(TraceReason r) const {
  std::size_t n=0; for (auto& x: rows_) if (x.reason==r) ++n; return n;
}
} // namespace sentio

```

## 📄 **FILE 69 of 92**: src/strategy_bollinger_squeeze_breakout.cpp

**File Information**:
- **Path**: `src/strategy_bollinger_squeeze_breakout.cpp`

- **Size**: 147 lines
- **Modified**: 2025-09-05 15:29:30

- **Type**: .cpp

```text
#include "sentio/strategy_bollinger_squeeze_breakout.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

namespace sentio {

    BollingerSqueezeBreakoutStrategy::BollingerSqueezeBreakoutStrategy() 
    : BaseStrategy("BollingerSqueezeBreakout"), bollinger_(20, 2.0) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap BollingerSqueezeBreakoutStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed parameters to be more sensitive to trading opportunities.
    return {
        {"bb_window", 20.0},
        {"bb_k", 1.8},                   // Tighter bands to increase breakout signals
        {"squeeze_percentile", 0.25},    // Squeeze is now top 25% of quietest periods (was 15%)
        {"squeeze_lookback", 60.0},      // Shorter lookback for volatility
        {"hold_max_bars", 120.0},
        {"tp_mult_sd", 1.5},
        {"sl_mult_sd", 1.5},
        {"min_squeeze_bars", 3.0}        // Require at least 3 bars of squeeze
    };
}

ParameterSpace BollingerSqueezeBreakoutStrategy::get_param_space() const { return {}; }

void BollingerSqueezeBreakoutStrategy::apply_params() {
    bb_window_ = static_cast<int>(params_["bb_window"]);
    squeeze_percentile_ = params_["squeeze_percentile"];
    squeeze_lookback_ = static_cast<int>(params_["squeeze_lookback"]);
    hold_max_bars_ = static_cast<int>(params_["hold_max_bars"]);
    tp_mult_sd_ = params_["tp_mult_sd"];
    sl_mult_sd_ = params_["sl_mult_sd"];
    min_squeeze_bars_ = static_cast<int>(params_["min_squeeze_bars"]);
    
    bollinger_ = Bollinger(bb_window_, params_["bb_k"]);
    sd_history_.reserve(squeeze_lookback_);
    reset_state();
}

void BollingerSqueezeBreakoutStrategy::reset_state() {
    BaseStrategy::reset_state();
    state_ = State::Idle;
    bars_in_trade_ = 0;
    squeeze_duration_ = 0;
    sd_history_.clear();
}

StrategySignal BollingerSqueezeBreakoutStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < squeeze_lookback_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }
    
    if (state_ == State::Long || state_ == State::Short) {
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            signal.type = (state_ == State::Long) ? StrategySignal::Type::SELL : StrategySignal::Type::BUY;
            reset_state();
            diag_.emitted++;
            return signal;
        }
        return signal;
    }

    update_state_machine(bars[current_index]);

    if (state_ == State::ArmedLong || state_ == State::ArmedShort) {
        if (squeeze_duration_ < min_squeeze_bars_) {
            diag_.drop(DropReason::THRESHOLD);
            state_ = State::Idle;
            return signal;
        }

        double mid, lo, hi, sd;
        bollinger_.step(bars[current_index].close, mid, lo, hi, sd);
        
        if (state_ == State::ArmedLong) {
            signal.type = StrategySignal::Type::BUY;
            state_ = State::Long;
        } else {
            signal.type = StrategySignal::Type::SELL;
            state_ = State::Short;
        }
        
        signal.confidence = 0.8;
        diag_.emitted++;
        bars_in_trade_ = 0;
    } else {
        diag_.drop(DropReason::THRESHOLD);
    }

    return signal;
}

void BollingerSqueezeBreakoutStrategy::update_state_machine(const Bar& bar) {
    double mid, lo, hi, sd;
    bollinger_.step(bar.close, mid, lo, hi, sd);
    
    sd_history_.push_back(sd);
    if (sd_history_.size() > static_cast<size_t>(squeeze_lookback_)) {
        sd_history_.erase(sd_history_.begin());
    }
    
    double sd_threshold = calculate_volatility_percentile(squeeze_percentile_);
    bool is_squeezed = (sd_history_.size() == static_cast<size_t>(squeeze_lookback_)) && (sd <= sd_threshold);

    switch (state_) {
        case State::Idle:
            if (is_squeezed) {
                state_ = State::Squeezed;
                squeeze_duration_ = 1;
            }
            break;
        case State::Squeezed:
            if (bar.close > hi) state_ = State::ArmedLong;
            else if (bar.close < lo) state_ = State::ArmedShort;
            else if (!is_squeezed) state_ = State::Idle;
            else squeeze_duration_++;
            break;
        default:
            break;
    }
}

// **MODIFIED**: Implemented a proper percentile calculation instead of a stub.
double BollingerSqueezeBreakoutStrategy::calculate_volatility_percentile(double percentile) const {
    if (sd_history_.size() < static_cast<size_t>(squeeze_lookback_)) {
        return std::numeric_limits<double>::max(); // Not enough data, effectively prevents squeeze
    }
    
    std::vector<double> sorted_history = sd_history_;
    std::sort(sorted_history.begin(), sorted_history.end());
    
    int index = static_cast<int>(percentile * (sorted_history.size() - 1));
    return sorted_history[index];
}

REGISTER_STRATEGY(BollingerSqueezeBreakoutStrategy, "BollingerSqueezeBreakout");

} // namespace sentio

```

## 📄 **FILE 70 of 92**: src/strategy_market_making.cpp

**File Information**:
- **Path**: `src/strategy_market_making.cpp`

- **Size**: 141 lines
- **Modified**: 2025-09-05 17:13:47

- **Type**: .cpp

```text
#include "sentio/strategy_market_making.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace sentio {

MarketMakingStrategy::MarketMakingStrategy() 
    : BaseStrategy("MarketMaking"),
      rolling_returns_(20),
      rolling_volume_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap MarketMakingStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed volatility and volume thresholds to allow participation.
    return {
        {"base_spread", 0.001}, {"min_spread", 0.0005}, {"max_spread", 0.003},
        {"order_levels", 3.0}, {"level_spacing", 0.0005}, {"order_size_base", 0.5},
        {"max_inventory", 100.0}, {"inventory_skew_mult", 0.002},
        {"adverse_selection_threshold", 0.004}, // Was 0.002, allowing participation in more volatile conditions
        {"volatility_window", 20.0},
        {"volume_window", 50.0}, {"min_volume_ratio", 0.05}, // Was 0.1, making it even more permissive
        {"max_orders_per_bar", 10.0}, {"rebalance_frequency", 10.0}
    };
}

ParameterSpace MarketMakingStrategy::get_param_space() const { return {}; }

void MarketMakingStrategy::apply_params() {
    base_spread_ = params_.at("base_spread");
    min_spread_ = params_.at("min_spread");
    max_spread_ = params_.at("max_spread");
    order_levels_ = static_cast<int>(params_.at("order_levels"));
    level_spacing_ = params_.at("level_spacing");
    order_size_base_ = params_.at("order_size_base");
    max_inventory_ = params_.at("max_inventory");
    inventory_skew_mult_ = params_.at("inventory_skew_mult");
    adverse_selection_threshold_ = params_.at("adverse_selection_threshold");
    min_volume_ratio_ = params_.at("min_volume_ratio");
    max_orders_per_bar_ = static_cast<int>(params_.at("max_orders_per_bar"));
    rebalance_frequency_ = static_cast<int>(params_.at("rebalance_frequency"));

    int vol_window = std::max(1, static_cast<int>(params_.at("volatility_window")));
    int vol_mean_window = std::max(1, static_cast<int>(params_.at("volume_window")));
    
    rolling_returns_.reset(vol_window);
    rolling_volume_.reset(vol_mean_window);
    reset_state();
}

void MarketMakingStrategy::reset_state() {
    BaseStrategy::reset_state();
    market_state_ = MarketState{};
    rolling_returns_.reset(std::max(1, static_cast<int>(params_.at("volatility_window"))));
    rolling_volume_.reset(std::max(1, static_cast<int>(params_.at("volume_window"))));
}

StrategySignal MarketMakingStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;
    
    // Always update indicators to have a full history for the next bar
    if(current_index > 0) {
        double price_return = (bars[current_index].close - bars[current_index - 1].close) / bars[current_index - 1].close;
        rolling_returns_.push(price_return);
    }
    rolling_volume_.push(bars[current_index].volume);

    // Wait for indicators to warm up
    if (rolling_volume_.size() < static_cast<size_t>(params_.at("volume_window"))) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }

    if (!should_participate(bars[current_index])) {
        return signal;
    }
    
    double inventory_skew = get_inventory_skew();
    
    // **FIXED**: Generate signals based on volatility and volume patterns instead of inventory
    // Since inventory tracking is not implemented, use a simpler approach
    double volatility = rolling_returns_.stddev();
    double avg_volume = rolling_volume_.mean();
    double volume_ratio = (avg_volume > 0) ? bars[current_index].volume / avg_volume : 0.0;
    
    // Generate signals when volatility is moderate and volume is increasing
    if (volatility > 0.0005 && volatility < adverse_selection_threshold_ && volume_ratio > 0.8) {
        // Simple momentum-based signal
        if (current_index > 0) {
            double price_change = (bars[current_index].close - bars[current_index - 1].close) / bars[current_index - 1].close;
            if (price_change > 0.001) {
                signal.type = StrategySignal::Type::BUY;
            } else if (price_change < -0.001) {
                signal.type = StrategySignal::Type::SELL;
            } else {
                diag_.drop(DropReason::THRESHOLD);
                return signal;
            }
        } else {
            diag_.drop(DropReason::THRESHOLD);
            return signal;
        }
    } else {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }

    signal.confidence = 0.6; // Fixed confidence since we're not using inventory_skew
    diag_.emitted++;
    return signal;
}

bool MarketMakingStrategy::should_participate(const Bar& bar) {
    double volatility = rolling_returns_.stddev();
    
    if (volatility > adverse_selection_threshold_) {
        diag_.drop(DropReason::THRESHOLD); 
        return false;
    }

    double avg_volume = rolling_volume_.mean();
    
    if (avg_volume > 0 && (bar.volume < avg_volume * min_volume_ratio_)) {
        diag_.drop(DropReason::ZERO_VOL);
        return false;
    }
    return true;
}

double MarketMakingStrategy::get_inventory_skew() const {
    if (max_inventory_ <= 0) return 0.0;
    double normalized_inventory = market_state_.inventory / max_inventory_;
    return -normalized_inventory * inventory_skew_mult_;
}

REGISTER_STRATEGY(MarketMakingStrategy, "MarketMaking");

} // namespace sentio


```

## 📄 **FILE 71 of 92**: src/strategy_momentum_volume.cpp

**File Information**:
- **Path**: `src/strategy_momentum_volume.cpp`

- **Size**: 147 lines
- **Modified**: 2025-09-05 15:24:52

- **Type**: .cpp

```text
#include "sentio/strategy_momentum_volume.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>

namespace sentio {

MomentumVolumeProfileStrategy::MomentumVolumeProfileStrategy() 
    : BaseStrategy("MomentumVolumeProfile"), avg_volume_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap MomentumVolumeProfileStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed parameters to be more sensitive
    return {
        {"profile_period", 100.0},
        {"value_area_pct", 0.7},
        {"price_bins", 30.0},
        {"breakout_threshold_pct", 0.001},
        {"momentum_lookback", 20.0},
        {"volume_surge_mult", 1.2}, // Was 1.5
        {"cool_down_period", 5.0}   // Was 10
    };
}

ParameterSpace MomentumVolumeProfileStrategy::get_param_space() const { return {}; }

void MomentumVolumeProfileStrategy::apply_params() {
    profile_period_ = static_cast<int>(params_["profile_period"]);
    value_area_pct_ = params_["value_area_pct"];
    price_bins_ = static_cast<int>(params_["price_bins"]);
    breakout_threshold_pct_ = params_["breakout_threshold_pct"];
    momentum_lookback_ = static_cast<int>(params_["momentum_lookback"]);
    volume_surge_mult_ = params_["volume_surge_mult"];
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);
    avg_volume_ = RollingMean(profile_period_);
    reset_state();
}

void MomentumVolumeProfileStrategy::reset_state() {
    BaseStrategy::reset_state();
    volume_profile_.clear();
    last_profile_update_ = -1;
    avg_volume_ = RollingMean(profile_period_);
}

StrategySignal MomentumVolumeProfileStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < profile_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }
    
    if (is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return signal;
    }
    
    // Periodically rebuild the expensive volume profile
    if (last_profile_update_ == -1 || current_index - last_profile_update_ >= 10) {
        build_volume_profile(bars, current_index);
        last_profile_update_ = current_index;
    }
    
    if (volume_profile_.value_area_high <= 0) {
        diag_.drop(DropReason::NAN_INPUT); // Profile not ready or invalid
        return signal;
    }

    const auto& bar = bars[current_index];
    avg_volume_.push(bar.volume);
    
    bool breakout_up = bar.close > (volume_profile_.value_area_high * (1.0 + breakout_threshold_pct_));
    bool breakout_down = bar.close < (volume_profile_.value_area_low * (1.0 - breakout_threshold_pct_));

    if (!breakout_up && !breakout_down) {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }

    if (!is_momentum_confirmed(bars, current_index)) {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }
    
    if (bar.volume < avg_volume_.mean() * volume_surge_mult_) {
        diag_.drop(DropReason::ZERO_VOL);
        return signal;
    }

    if (breakout_up) {
        signal.type = StrategySignal::Type::BUY;
    } else {
        signal.type = StrategySignal::Type::SELL;
    }
    
    signal.confidence = 0.85;
    diag_.emitted++;
    state_.last_trade_bar = current_index;

    return signal;
}

bool MomentumVolumeProfileStrategy::is_momentum_confirmed(const std::vector<Bar>& bars, int index) const {
    if (index < momentum_lookback_) return false;
    double price_change = bars[index].close - bars[index - momentum_lookback_].close;
    if (bars[index].close > volume_profile_.value_area_high) {
        return price_change > 0;
    }
    if (bars[index].close < volume_profile_.value_area_low) {
        return price_change < 0;
    }
    return false;
}

// **MODIFIED**: This is now a functional, albeit simple, implementation to prevent NaN drops.
void MomentumVolumeProfileStrategy::build_volume_profile(const std::vector<Bar>& bars, int end_index) {
    volume_profile_.clear();
    int start_index = std::max(0, end_index - profile_period_ + 1);

    double min_price = std::numeric_limits<double>::max();
    double max_price = std::numeric_limits<double>::lowest();
    
    for (int i = start_index; i <= end_index; ++i) {
        min_price = std::min(min_price, bars[i].low);
        max_price = std::max(max_price, bars[i].high);
    }
    
    if (max_price <= min_price) return; // Cannot build profile

    // Simple implementation: Value Area is the high/low of the lookback period
    volume_profile_.value_area_high = max_price;
    volume_profile_.value_area_low = min_price;
    volume_profile_.total_volume = 1.0; // Mark as valid by setting a non-zero value
    // A proper implementation would bin prices and find the 70% volume area.
}

void MomentumVolumeProfileStrategy::calculate_value_area() {
    // This is now handled within build_volume_profile for simplicity
}

REGISTER_STRATEGY(MomentumVolumeProfileStrategy, "MomentumVolumeProfile");

} // namespace sentio

```

## 📄 **FILE 72 of 92**: src/strategy_opening_range_breakout.cpp

**File Information**:
- **Path**: `src/strategy_opening_range_breakout.cpp`

- **Size**: 131 lines
- **Modified**: 2025-09-05 15:24:52

- **Type**: .cpp

```text
#include "sentio/strategy_opening_range_breakout.hpp"
#include <algorithm>
#include <cmath>

namespace sentio {

OpeningRangeBreakoutStrategy::OpeningRangeBreakoutStrategy() 
    : BaseStrategy("OpeningRangeBreakout") {
    params_ = get_default_params();
    apply_params();
}
ParameterMap OpeningRangeBreakoutStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed cooldown to allow more frequent trades.
    return {
        {"opening_range_minutes", 30.0},
        {"breakout_confirmation_bars", 1.0},
        {"volume_multiplier", 1.5},
        {"stop_loss_pct", 0.01},
        {"take_profit_pct", 0.02},
        {"cool_down_period", 5.0}, // Was 15.0
    };
}

ParameterSpace OpeningRangeBreakoutStrategy::get_param_space() const { /* ... unchanged ... */ return {}; }

void OpeningRangeBreakoutStrategy::apply_params() {
    // **NEW**: Cache parameters
    opening_range_minutes_ = static_cast<int>(params_["opening_range_minutes"]);
    breakout_confirmation_bars_ = static_cast<int>(params_["breakout_confirmation_bars"]);
    volume_multiplier_ = params_["volume_multiplier"];
    stop_loss_pct_ = params_["stop_loss_pct"];
    take_profit_pct_ = params_["take_profit_pct"];
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);
    reset_state();
}

void OpeningRangeBreakoutStrategy::reset_state() {
    BaseStrategy::reset_state();
    current_range_ = OpeningRange{};
    day_start_index_ = -1;
}

StrategySignal OpeningRangeBreakoutStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < 1) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }

    // **MODIFIED**: Robust and performant new-day detection
    const int SECONDS_IN_DAY = 86400;
    long current_day = bars[current_index].ts_nyt_epoch / SECONDS_IN_DAY;
    long prev_day = bars[current_index - 1].ts_nyt_epoch / SECONDS_IN_DAY;

    if (current_day != prev_day) {
        reset_state(); // Reset everything for the new day
        day_start_index_ = current_index;
    }
    
    if (day_start_index_ == -1) { // Haven't established the start of the first day yet
        day_start_index_ = 0;
    }

    int bars_into_day = current_index - day_start_index_;

    // --- Phase 1: Define the Opening Range ---
    if (bars_into_day < opening_range_minutes_) {
        if (bars_into_day == 0) {
            current_range_.high = bars[current_index].high;
            current_range_.low = bars[current_index].low;
        } else {
            current_range_.high = std::max(current_range_.high, bars[current_index].high);
            current_range_.low = std::min(current_range_.low, bars[current_index].low);
        }
        diag_.drop(DropReason::SESSION); // Use SESSION to mean "in range formation"
        return signal;
    }

    // --- Finalize the range exactly once ---
    if (!current_range_.is_finalized) {
        current_range_.end_bar = current_index - 1;
        current_range_.is_finalized = true;
    }

    // --- Phase 2: Look for Breakouts ---
    if (state_.in_position || is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return signal;
    }

    const auto& bar = bars[current_index];
    bool is_breakout_up = bar.close > current_range_.high;
    bool is_breakout_down = bar.close < current_range_.low;

    if (!is_breakout_up && !is_breakout_down) {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }
    
    // Volume Confirmation
    double avg_volume = 0;
    for (int i = day_start_index_; i < current_range_.end_bar; ++i) {
        avg_volume += bars[i].volume;
    }
    avg_volume /= (current_range_.end_bar - day_start_index_ + 1);

    if (bar.volume < avg_volume * volume_multiplier_) {
        diag_.drop(DropReason::ZERO_VOL); // Re-using for low volume
        return signal;
    }

    // Generate Signal
    if (is_breakout_up) {
        signal.type = StrategySignal::Type::BUY;
    } else { // is_breakout_down
        signal.type = StrategySignal::Type::SELL;
    }

    signal.confidence = 0.9;
    diag_.emitted++;
    state_.in_position = true; // Manually set state as this is an intraday strategy
    state_.last_trade_bar = current_index;

    return signal;
}

// Register the strategy
REGISTER_STRATEGY(OpeningRangeBreakoutStrategy, "OpeningRangeBreakout");

} // namespace sentio
```

## 📄 **FILE 73 of 92**: src/strategy_order_flow_imbalance.cpp

**File Information**:
- **Path**: `src/strategy_order_flow_imbalance.cpp`

- **Size**: 105 lines
- **Modified**: 2025-09-05 15:25:26

- **Type**: .cpp

```text
#include "sentio/strategy_order_flow_imbalance.hpp"
#include <algorithm>
#include <cmath>

namespace sentio {

OrderFlowImbalanceStrategy::OrderFlowImbalanceStrategy() 
    : BaseStrategy("OrderFlowImbalance"),
      rolling_pressure_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap OrderFlowImbalanceStrategy::get_default_params() const {
    return {
        {"lookback_period", 50.0},
        {"entry_threshold_long", 0.60},
        {"entry_threshold_short", 0.40},
        {"hold_max_bars", 60.0},
        {"cool_down_period", 5.0}
    };
}

ParameterSpace OrderFlowImbalanceStrategy::get_param_space() const { return {}; }

void OrderFlowImbalanceStrategy::apply_params() {
    lookback_period_ = static_cast<int>(params_["lookback_period"]);
    entry_threshold_long_ = params_["entry_threshold_long"];
    entry_threshold_short_ = params_["entry_threshold_short"];
    hold_max_bars_ = static_cast<int>(params_["hold_max_bars"]);
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);
    
    rolling_pressure_ = RollingMean(lookback_period_);
    reset_state();
}

void OrderFlowImbalanceStrategy::reset_state() {
    BaseStrategy::reset_state();
    ofi_state_ = OFIState::Flat; // **FIXED**: Use the renamed state variable
    bars_in_trade_ = 0;
    rolling_pressure_ = RollingMean(lookback_period_);
}

double OrderFlowImbalanceStrategy::calculate_bar_pressure(const Bar& bar) const {
    double range = bar.high - bar.low;
    if (range < 1e-9) {
        return 0.5; // Neutral pressure if there's no range
    }
    return (bar.close - bar.low) / range;
}

StrategySignal OrderFlowImbalanceStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < lookback_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }

    double pressure = calculate_bar_pressure(bars[current_index]);
    double avg_pressure = rolling_pressure_.push(pressure);

    // **FIXED**: Use the strategy-specific 'ofi_state_' for state machine logic
    if (ofi_state_ == OFIState::Flat) {
        if (is_cooldown_active(current_index, cool_down_period_)) {
            diag_.drop(DropReason::COOLDOWN);
            return signal;
        }

        if (avg_pressure > entry_threshold_long_) {
            signal.type = StrategySignal::Type::BUY;
            ofi_state_ = OFIState::Long;
            // **FIXED**: Correctly access the 'state_' member from BaseStrategy
            state_.last_trade_bar = current_index;
        } else if (avg_pressure < entry_threshold_short_) {
            signal.type = StrategySignal::Type::SELL;
            ofi_state_ = OFIState::Short;
            // **FIXED**: Correctly access the 'state_' member from BaseStrategy
            state_.last_trade_bar = current_index;
        } else {
            diag_.drop(DropReason::THRESHOLD);
            return signal;
        }

        signal.confidence = 0.7;
        diag_.emitted++;
        bars_in_trade_ = 0;

    } else { // In a trade, check for exit
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            // **FIXED**: Use 'ofi_state_' to determine exit signal direction
            signal.type = (ofi_state_ == OFIState::Long) ? StrategySignal::Type::SELL : StrategySignal::Type::BUY;
            diag_.emitted++;
            reset_state();
        }
    }

    return signal;
}

REGISTER_STRATEGY(OrderFlowImbalanceStrategy, "OrderFlowImbalance");

} // namespace sentio


```

## 📄 **FILE 74 of 92**: src/strategy_order_flow_scalping.cpp

**File Information**:
- **Path**: `src/strategy_order_flow_scalping.cpp`

- **Size**: 120 lines
- **Modified**: 2025-09-05 16:40:29

- **Type**: .cpp

```text
#include "sentio/strategy_order_flow_scalping.hpp"
#include <algorithm>
#include <cmath>

namespace sentio {

OrderFlowScalpingStrategy::OrderFlowScalpingStrategy() 
    : BaseStrategy("OrderFlowScalping"),
      rolling_pressure_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap OrderFlowScalpingStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed the imbalance threshold to arm more frequently.
    return {
        {"lookback_period", 50.0},
        {"imbalance_threshold", 0.55}, // Was 0.65, now arms when avg pressure is > 55%
        {"hold_max_bars", 20.0},
        {"cool_down_period", 3.0}
    };
}

ParameterSpace OrderFlowScalpingStrategy::get_param_space() const { return {}; }

void OrderFlowScalpingStrategy::apply_params() {
    lookback_period_ = static_cast<int>(params_["lookback_period"]);
    imbalance_threshold_ = params_["imbalance_threshold"];
    hold_max_bars_ = static_cast<int>(params_["hold_max_bars"]);
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);

    rolling_pressure_ = RollingMean(lookback_period_);
    reset_state();
}

void OrderFlowScalpingStrategy::reset_state() {
    BaseStrategy::reset_state();
    of_state_ = OFState::Idle; // **FIXED**: Use the renamed state variable
    bars_in_trade_ = 0;
    rolling_pressure_ = RollingMean(lookback_period_);
}

double OrderFlowScalpingStrategy::calculate_bar_pressure(const Bar& bar) const {
    double range = bar.high - bar.low;
    if (range < 1e-9) return 0.5;
    return (bar.close - bar.low) / range;
}

StrategySignal OrderFlowScalpingStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < lookback_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }

    const auto& bar = bars[current_index];
    double pressure = calculate_bar_pressure(bar);
    double avg_pressure = rolling_pressure_.push(pressure);

    // **FIXED**: Use the strategy-specific 'of_state_' for state machine logic
    if (of_state_ == OFState::Long || of_state_ == OFState::Short) {
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            signal.type = (of_state_ == OFState::Long) ? StrategySignal::Type::SELL : StrategySignal::Type::BUY;
            diag_.emitted++;
            reset_state();
        }
        return signal;
    }
    
    if (is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return signal;
    }

    switch (of_state_) {
        case OFState::Idle:
            if (avg_pressure > imbalance_threshold_) of_state_ = OFState::ArmedLong;
            else if (avg_pressure < (1.0 - imbalance_threshold_)) of_state_ = OFState::ArmedShort;
            else diag_.drop(DropReason::THRESHOLD);
            break;
            
        case OFState::ArmedLong:
            if (pressure > 0.5) { // Confirmation bar must be bullish
                signal.type = StrategySignal::Type::BUY;
                of_state_ = OFState::Long;
            } else { // Failed confirmation
                of_state_ = OFState::Idle;
                diag_.drop(DropReason::THRESHOLD);
            }
            break;

        case OFState::ArmedShort:
            if (pressure < 0.5) { // Confirmation bar must be bearish
                signal.type = StrategySignal::Type::SELL;
                of_state_ = OFState::Short;
            } else { // Failed confirmation
                of_state_ = OFState::Idle;
                diag_.drop(DropReason::THRESHOLD);
            }
            break;
        default: break;
    }
    
    if (signal.type != StrategySignal::Type::HOLD) {
        signal.confidence = 0.7;
        diag_.emitted++;
        bars_in_trade_ = 0;
        // **FIXED**: This now correctly refers to the 'state_' member from BaseStrategy
        state_.last_trade_bar = current_index;
    }
    
    return signal;
}

REGISTER_STRATEGY(OrderFlowScalpingStrategy, "OrderFlowScalping");

} // namespace sentio


```

## 📄 **FILE 75 of 92**: src/strategy_sma_cross.cpp

**File Information**:
- **Path**: `src/strategy_sma_cross.cpp`

- **Size**: 35 lines
- **Modified**: 2025-09-05 16:34:48

- **Type**: .cpp

```text
#include "sentio/strategy_sma_cross.hpp"
#include <cmath>

namespace sentio {

SMACrossStrategy::SMACrossStrategy(const SMACrossCfg& cfg)
  : cfg_(cfg), sma_fast_(cfg.fast), sma_slow_(cfg.slow) {}

void SMACrossStrategy::on_bar(const StrategyCtx& ctx, const Bar& b) {
  (void)ctx;
  sma_fast_.push(b.close);
  sma_slow_.push(b.close);

  if (!warmed_up()) { last_.reset(); return; }

  double f = sma_fast_.value();
  double s = sma_slow_.value();
  if (!std::isfinite(f) || !std::isfinite(s)) { last_.reset(); return; }

  // Detect cross (including equality toggle avoidance)
  bool golden_now  = (f >= s) && !(std::isfinite(last_fast_) && std::isfinite(last_slow_) && last_fast_ >= last_slow_);
  bool death_now   = (f <= s) && !(std::isfinite(last_fast_) && std::isfinite(last_slow_) && last_fast_ <= last_slow_);

  if (golden_now) {
    last_ = StrategySignal{StrategySignal::Type::BUY, cfg_.conf_fast_slow};
  } else if (death_now) {
    last_ = StrategySignal{StrategySignal::Type::SELL, cfg_.conf_fast_slow};
  } else {
    last_.reset(); // no edge this bar
  }

  last_fast_ = f; last_slow_ = s;
}

} // namespace sentio

```

## 📄 **FILE 76 of 92**: src/strategy_vwap_reversion.cpp

**File Information**:
- **Path**: `src/strategy_vwap_reversion.cpp`

- **Size**: 156 lines
- **Modified**: 2025-09-05 15:24:52

- **Type**: .cpp

```text
#include "sentio/strategy_vwap_reversion.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace sentio {

VWAPReversionStrategy::VWAPReversionStrategy() : BaseStrategy("VWAPReversion") {
    params_ = get_default_params();
    apply_params();
}

ParameterMap VWAPReversionStrategy::get_default_params() const {
    // Default parameters remain the same
    return {
        {"vwap_period", 390.0}, {"band_multiplier", 0.005}, {"max_band_width", 0.01},
        {"min_distance_from_vwap", 0.001}, {"volume_confirmation_mult", 1.2},
        {"rsi_period", 14.0}, {"rsi_oversold", 40.0}, {"rsi_overbought", 60.0},
        {"stop_loss_pct", 0.003}, {"take_profit_pct", 0.005},
        {"time_stop_bars", 30.0}, {"cool_down_period", 2.0}
    };
}

ParameterSpace VWAPReversionStrategy::get_param_space() const { return {}; }

void VWAPReversionStrategy::apply_params() {
    vwap_period_ = static_cast<int>(params_["vwap_period"]);
    band_multiplier_ = params_["band_multiplier"];
    max_band_width_ = params_["max_band_width"];
    min_distance_from_vwap_ = params_["min_distance_from_vwap"];
    volume_confirmation_mult_ = params_["volume_confirmation_mult"];
    rsi_period_ = static_cast<int>(params_["rsi_period"]);
    rsi_oversold_ = params_["rsi_oversold"];
    rsi_overbought_ = params_["rsi_overbought"];
    stop_loss_pct_ = params_["stop_loss_pct"];
    take_profit_pct_ = params_["take_profit_pct"];
    time_stop_bars_ = static_cast<int>(params_["time_stop_bars"]);
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);
    reset_state();
}

void VWAPReversionStrategy::reset_state() {
    BaseStrategy::reset_state();
    cumulative_pv_ = 0.0;
    cumulative_volume_ = 0.0;
    time_in_position_ = 0;
    vwap_ = 0.0;
}

void VWAPReversionStrategy::update_vwap(const Bar& bar) {
    double typical_price = (bar.high + bar.low + bar.close) / 3.0;
    cumulative_pv_ += typical_price * bar.volume;
    cumulative_volume_ += bar.volume;
    if (cumulative_volume_ > 0) {
        vwap_ = cumulative_pv_ / cumulative_volume_;
    }
}

StrategySignal VWAPReversionStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;
    update_vwap(bars[current_index]);

    if (current_index < rsi_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }
    
    if (is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return signal;
    }
    
    if (vwap_ <= 0) {
        diag_.drop(DropReason::NAN_INPUT);
        return signal;
    }

    const auto& bar = bars[current_index];
    double distance_pct = std::abs(bar.close - vwap_) / vwap_;
    if (distance_pct < min_distance_from_vwap_) {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }

    double upper_band = vwap_ * (1.0 + band_multiplier_);
    double lower_band = vwap_ * (1.0 - band_multiplier_);

    bool buy_condition = bar.close < lower_band && is_rsi_condition_met(bars, current_index, true);
    bool sell_condition = bar.close > upper_band && is_rsi_condition_met(bars, current_index, false);

    if (buy_condition) {
        signal.type = StrategySignal::Type::BUY;
    } else if (sell_condition) {
        signal.type = StrategySignal::Type::SELL;
    } else {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }

    signal.confidence = 0.8;
    diag_.emitted++;
    state_.last_trade_bar = current_index;
    return signal;
}

bool VWAPReversionStrategy::is_rsi_condition_met(const std::vector<Bar>& bars, int index, bool for_buy) const {
    std::vector<double> closes;
    closes.reserve(rsi_period_);
    for(int i = 0; i < rsi_period_; ++i) {
        closes.push_back(bars[index - rsi_period_ + 1 + i].close);
    }
    // Simple RSI calculation
    double rsi = calculate_simple_rsi(closes);
    return for_buy ? (rsi < rsi_oversold_) : (rsi > rsi_overbought_);
}

double VWAPReversionStrategy::calculate_simple_rsi(const std::vector<double>& prices) const {
    if (prices.size() < 2) return 50.0; // Neutral RSI if not enough data
    
    std::vector<double> gains, losses;
    for (size_t i = 1; i < prices.size(); ++i) {
        double change = prices[i] - prices[i-1];
        if (change > 0) {
            gains.push_back(change);
            losses.push_back(0.0);
        } else {
            gains.push_back(0.0);
            losses.push_back(-change);
        }
    }
    
    if (gains.empty()) return 50.0;
    
    double avg_gain = std::accumulate(gains.begin(), gains.end(), 0.0) / gains.size();
    double avg_loss = std::accumulate(losses.begin(), losses.end(), 0.0) / losses.size();
    
    if (avg_loss == 0.0) return 100.0; // All gains
    if (avg_gain == 0.0) return 0.0;   // All losses
    
    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}

bool VWAPReversionStrategy::is_volume_confirmed(const std::vector<Bar>& bars, int index) const {
    if (index < 20) return true;
    double avg_vol = 0;
    for(int i = 1; i <= 20; ++i) {
        avg_vol += bars[index-i].volume;
    }
    avg_vol /= 20.0;
    return bars[index].volume > avg_vol * volume_confirmation_mult_;
}

REGISTER_STRATEGY(VWAPReversionStrategy, "VWAPReversion");

} // namespace sentio
```

## 📄 **FILE 77 of 92**: src/test_rth.cpp

**File Information**:
- **Path**: `src/test_rth.cpp`

- **Size**: 51 lines
- **Modified**: 2025-09-05 14:30:31

- **Type**: .cpp

```text
// test_rth.cpp (use your preferred test framework)
#include "sentio/rth_calendar.hpp"
#include "sentio/calendar_seed.hpp"
#include <cassert>
#include "cctz/time_zone.h"
#include "cctz/civil_time.h"

using namespace std::chrono;

static std::int64_t to_epoch_utc(int y,int m,int d,int hh,int mm,int ss,const char* tz_name){
  using namespace std::chrono;
  // Interpret given local wall time (tz_name) and convert to UTC epoch
  cctz::time_zone tz;
  if (!cctz::load_time_zone(tz_name, &tz)) {
    return 0;
  }
  
  auto ct = cctz::civil_second(y, m, d, hh, mm, ss);
  auto tp = cctz::convert(ct, tz);
  return tp.time_since_epoch().count();
}

int main() {
  auto cal = sentio::make_default_nyse_calendar();

  // 2022-09-06 09:30:00 ET should be RTH (day after Labor Day)
  auto t1 = to_epoch_utc(2022,9,6,9,30,0,"America/New_York");
  assert(cal.is_rth_utc(t1));

  // 2022-11-25 12:59:59 ET (Black Friday early close @ 13:00) -> RTH
  auto t2 = to_epoch_utc(2022,11,25,12,59,59,"America/New_York");
  assert(cal.is_rth_utc(t2));

  // 2022-11-25 13:00:01 ET -> NOT RTH (just past early close)
  auto t3 = to_epoch_utc(2022,11,25,13,0,1,"America/New_York");
  assert(!cal.is_rth_utc(t3));

  // Labor Day 2022-09-05 10:00 ET -> NOT RTH (holiday)
  auto t4 = to_epoch_utc(2022,9,5,10,0,0,"America/New_York");
  assert(!cal.is_rth_utc(t4));

  // 2022-09-06 09:29:59 ET -> Pre-open (NOT RTH)
  auto t5 = to_epoch_utc(2022,9,6,9,29,59,"America/New_York");
  assert(!cal.is_rth_utc(t5));

  // 2022-09-06 16:00:00 ET -> Still RTH (inclusive)
  auto t6 = to_epoch_utc(2022,9,6,16,0,0,"America/New_York");
  assert(cal.is_rth_utc(t6));

  return 0;
}
```

## 📄 **FILE 78 of 92**: src/time_utils.cpp

**File Information**:
- **Path**: `src/time_utils.cpp`

- **Size**: 106 lines
- **Modified**: 2025-09-05 15:29:30

- **Type**: .cpp

```text
#include "sentio/time_utils.hpp"
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <algorithm>

#if __has_include(<chrono>)
  #include <chrono>
  using namespace std::chrono;
#else
  #error "Need C++20 <chrono>. If missing tzdb, you can still normalize UTC here and handle ET in calendar."
#endif

namespace sentio {

bool iso8601_looks_like(const std::string& s) {
  // Very light heuristic: "YYYY-MM-DDTHH:MM" and either 'Z' or +/-HH:MM
  return s.size() >= 16 && s[4]=='-' && s[7]=='-' && (s.find('T') != std::string::npos);
}

bool epoch_ms_suspected(double v_ms) {
  // If it's larger than ~1e12 it's probably ms; (1e12 sec is ~31k years)
  return std::isfinite(v_ms) && v_ms > 1.0e11;
}

static inline sys_seconds parse_iso8601_to_utc(const std::string& s) {
  // Minimal ISO8601 handling: require offset or 'Z'.
  // For robustness use Howard Hinnant's date::parse with %FT%T%Ez.
  // Here we support the common forms: 2022-09-06T13:30:00Z and 2022-09-06T09:30:00-04:00
  // We'll implement a tiny parser that splits offset and adjusts.
  auto posT = s.find('T');
  if (posT == std::string::npos) throw std::runtime_error("ISO8601 missing T");
  // Find offset start: last char 'Z' or last '+'/'-'
  int sign = 0;
  int oh=0, om=0;
  bool zulu = false;
  std::size_t offPos = s.rfind('Z');
  if (offPos != std::string::npos && offPos > posT) {
    zulu = true;
  } else {
    std::size_t plus = s.rfind('+');
    std::size_t minus= s.rfind('-');
    std::size_t off  = std::string::npos;
    if (plus!=std::string::npos && plus>posT) { off=plus; sign=+1; }
    else if (minus!=std::string::npos && minus>posT) { off=minus; sign=-1; }
    if (off==std::string::npos) throw std::runtime_error("ISO8601 missing offset/Z");
    // parse HH:MM
    if (off+3 >= s.size()) throw std::runtime_error("Bad offset");
    oh = std::stoi(s.substr(off+1,2));
    if (off+6 <= s.size() && s[off+3]==':') om = std::stoi(s.substr(off+4,2));
  }

  // parse date/time parts (seconds optional)
  int Y = std::stoi(s.substr(0,4));
  int M = std::stoi(s.substr(5,2));
  int D = std::stoi(s.substr(8,2));
  int h = std::stoi(s.substr(posT+1,2));
  int m = std::stoi(s.substr(posT+4,2));
  int sec = 0;
  if (posT+6 < s.size() && s[posT+6]==':') {
    sec = std::stoi(s.substr(posT+7,2));
  }

  // Treat parsed time as local-time-with-offset; compute UTC by subtracting offset
  using namespace std::chrono;
  sys_days sd = sys_days(std::chrono::year{Y}/M/D);
  seconds local = hours{h} + minutes{m} + seconds{sec};
  seconds off = seconds{ (oh*3600 + om*60) * (zulu ? 0 : sign) };
  // If sign=+1 (e.g., +09:00), local = UTC + offset => UTC = local - offset
  seconds utc_sec = local - off;
  return sys_seconds{sd.time_since_epoch() + utc_sec};
}

std::chrono::sys_seconds to_utc_sys_seconds(const std::variant<std::int64_t, double, std::string>& ts) {
  if (std::holds_alternative<std::int64_t>(ts)) {
    // epoch seconds
    return std::chrono::sys_seconds{std::chrono::seconds{std::get<std::int64_t>(ts)}};
  }
  if (std::holds_alternative<double>(ts)) {
    // Could be epoch ms or sec (float). Prefer ms detection and round down.
    double v = std::get<double>(ts);
    if (!std::isfinite(v)) throw std::runtime_error("Non-finite epoch");
    if (epoch_ms_suspected(v)) {
      auto s = static_cast<std::int64_t>(v / 1000.0);
      return std::chrono::sys_seconds{std::chrono::seconds{s}};
    } else {
      auto s = static_cast<std::int64_t>(v);
      return std::chrono::sys_seconds{std::chrono::seconds{s}};
    }
  }
  const std::string& s = std::get<std::string>(ts);
  if (!iso8601_looks_like(s)) {
    // fall back: try integer seconds in string
    std::int64_t v{};
    auto sv = std::string_view{s};
    if (auto [p, ec] = std::from_chars(sv.data(), sv.data()+sv.size(), v); ec == std::errc{}) {
      return std::chrono::sys_seconds{std::chrono::seconds{v}};
    }
    throw std::runtime_error("Unrecognized timestamp format: " + s);
  }
  return parse_iso8601_to_utc(s);
}

} // namespace sentio
```

## 📄 **FILE 79 of 92**: src/wf.cpp

**File Information**:
- **Path**: `src/wf.cpp`

- **Size**: 612 lines
- **Modified**: 2025-09-05 20:13:14

- **Type**: .cpp

```text
// wf.cpp
// Walk-Forward driver with safe training loop, always-OOS policy, and robust windowing.
// Drop-in: requires C++20. Plug your own hooks where marked TODO.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

// -------------------------------
// Minimal date helper (day granularity)
// Replace with your own if you already have Date utilities.
// -------------------------------
struct Date {
  // days since epoch (UTC or NYT midnight—consistent across your dataset)
  int64_t days{};
  friend bool operator<(const Date& a, const Date& b){ return a.days < b.days; }
  friend bool operator<=(const Date& a, const Date& b){ return a.days <= b.days; }
  friend bool operator>(const Date& a, const Date& b){ return a.days > b.days; }
  friend bool operator>=(const Date& a, const Date& b){ return a.days >= b.days; }
  friend bool operator==(const Date& a, const Date& b){ return a.days == b.days; }
};
inline Date operator+(Date d, int add_days){ d.days += add_days; return d; }
inline Date days(int n){ return Date{n}; } // convenience

// -------------------------------
// WF config and data slice types
// -------------------------------
struct WfCfg {
  int train_days = 252 * 2; // ~2y
  int oos_days   = 63;      // ~1 quarter
  int step_days  = 63;      // slide by a quarter
};

struct DataSlice {
  // Your implementation should carry pointers/indices into your bar arrays.
  // For this template, we just keep [start,end) in "days since epoch".
  Date start{};
  Date end{};
  bool has_data() const { return start < end; }
};

// -------------------------------
// Params & hashing (replace with your strategy params)
// -------------------------------
struct Params {
  // Example parameters — replace with your actual fields.
  int   short_win = 20;
  int   long_win  = 100;
  double stop_bps = 50.0;
  friend bool operator==(const Params& a, const Params& b){
    return a.short_win==b.short_win && a.long_win==b.long_win && std::fabs(a.stop_bps-b.stop_bps)<1e-12;
  }
};

struct ParamsHash {
  size_t operator()(const Params& p) const noexcept {
    // simple FNV-1a style
    auto mix = [](size_t h, size_t v){ return (h ^ v) * 1099511628211ULL; };
    size_t h = 1469598103934665603ULL;
    h = mix(h, std::hash<int>{}(p.short_win));
    h = mix(h, std::hash<int>{}(p.long_win));
    h = mix(h, std::hash<long long>{}((long long)std::llround(p.stop_bps*10000)));
    return h;
  }
};

// -------------------------------
// WF window builder (bulletproof)
// -------------------------------
inline void validate_wf_cfg(const WfCfg& w){
  if (w.train_days <= 0) throw std::invalid_argument("train_days must be > 0");
  if (w.oos_days   <= 0) throw std::invalid_argument("oos_days must be > 0");
  if (w.step_days  <= 0) throw std::invalid_argument("step_days must be > 0");
}

inline std::vector<std::tuple<Date,Date,Date>>  // {train_start, train_end, oos_end}
make_wf_windows(Date start, Date end, const WfCfg& w){
  validate_wf_cfg(w);
  std::vector<std::tuple<Date,Date,Date>> out;
  if (!(start < end)) return out;

  Date cur = start;
  const int step = std::max(1, w.step_days);

  for (size_t guard=0; cur < end; ++guard){
    auto train_end = cur + w.train_days;
    auto oos_end   = train_end + w.oos_days;
    if (!(train_end < end) || !(oos_end <= end)) break;

    out.emplace_back(cur, train_end, oos_end);

    Date next = cur + step;
    if (!(next > cur)) throw std::runtime_error("WF window builder: next <= cur (no progress)");
    cur = next;

    if (guard > 10'000'000ULL) throw std::runtime_error("WF window builder: guard tripped");
  }
  return out;
}

// -------------------------------
// Training criteria & result
// -------------------------------
struct TrainCriteria {
  int    max_iters      = 200;    // absolute upper bound
  int    patience       = 20;     // stop after this many non-improving iters
  double min_delta      = 1e-4;   // min improvement to reset patience
  double target_metric  = 0.0;    // stop early if >= target (0 disables)
  int64_t time_budget_ms = 0;     // wall-clock cap (0 disables)
};

struct TrainResult {
  bool   success = false;   // met target or improved at least once
  int    iters   = 0;
  double best_metric = -1e300;
  Params best_params;
};

// -------------------------------
// Progress guard (Debug use)
// -------------------------------
struct ProgressGuard {
  uint64_t same_count = 0, limit = 1000000ULL;
  uint64_t last_token = 0;
  void step(uint64_t token){
    if (token == last_token) { if (++same_count > limit) throw std::runtime_error("ProgressGuard: no progress"); }
    else { last_token = token; same_count = 0; }
  }
};

// -------------------------------
// Safe training loop
// - evaluate: Params -> metric (higher is better)
// - propose_next: (last_params, last_metric) -> new Params (should usually differ)
// -------------------------------
inline TrainResult train_until(
    const TrainCriteria& cfg,
    Params init,
    const std::function<double(const Params&)>& evaluate,
    const std::function<Params(const Params&, double)>& propose_next)
{
  using clock = std::chrono::high_resolution_clock;
  const auto t0 = clock::now();

  TrainResult out;
  out.best_params = init;

  int no_improve = 0;
  int iter = 0;

  std::unordered_set<Params, ParamsHash> seen; seen.reserve(cfg.max_iters*2);
  ProgressGuard guard;

  Params cur = init;
  double cur_metric = evaluate(cur);
  out.best_metric = cur_metric;
  bool improved_once = false;

  seen.insert(cur);
  guard.step(ParamsHash{}(cur));

  if (cfg.target_metric > 0.0 && cur_metric >= cfg.target_metric) {
    out.success = true; out.iters = 1; return out;
  }

  for (iter = 2; iter <= cfg.max_iters; ++iter) {
    // Time budget check
    if (cfg.time_budget_ms > 0) {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
      if (ms >= cfg.time_budget_ms) break;
    }

    // Propose new params
    Params next = propose_next(cur, cur_metric);

    // If duplicate, try one more proposal; if still dup → accept best-so-far
    if (!seen.insert(next).second) {
      next = propose_next(next, cur_metric);
      if (!seen.insert(next).second) break;
    }

    double next_metric = evaluate(next);

    // Improvement logic
    if (next_metric >= out.best_metric + cfg.min_delta) {
      out.best_metric = next_metric;
      out.best_params = next;
      improved_once = true;
      no_improve = 0;
    } else {
      if (++no_improve >= cfg.patience) break;
    }

    if (cfg.target_metric > 0.0 && out.best_metric >= cfg.target_metric) break;

    // Guard against “no progress”
    guard.step(ParamsHash{}(next) ^ (uint64_t)iter);

    cur = next;
    cur_metric = next_metric;
  }

  out.iters = iter - 1;
  out.success = improved_once || (cfg.target_metric > 0.0 && out.best_metric >= cfg.target_metric);
  return out;
}

// -------------------------------
// OOS result container
// -------------------------------
struct OOSResult {
  bool   valid  = true;
  double metric = std::nan("");   // e.g., Sharpe/return
  std::string reason = "ok";
};

// -------------------------------
// Fold outcome (training + OOS); OOS always runs unless invalid data/params
// -------------------------------
struct FoldOutcome {
  bool training_success = false;
  int  train_iters      = 0;
  double train_metric_best = -1e300;

  Params params_used;
  bool   params_from_prev = false;

  bool   oos_valid = false;
  double oos_metric = std::nan("");
  std::string oos_reason = "uninit";
};

// -------------------------------
// Run a single WF fold
// Hooks you must provide via std::function (plug at call site):
//   train_metric(train_slice, params) -> double
//   run_oos(oos_slice, params) -> OOSResult
//   propose(last_params, last_metric) -> Params
//   params_valid(params) -> bool
//   baseline_params() -> Params
// -------------------------------
inline FoldOutcome run_wf_fold(
    const DataSlice& train, const DataSlice& oos,
    const Params& init, const TrainCriteria& crit,
    const std::optional<Params>& prev_fold_params,

    const std::function<double(const DataSlice&, const Params&)>& train_metric,
    const std::function<OOSResult(const DataSlice&, const Params&)>& run_oos,
    const std::function<Params(const Params&, double)>& propose,
    const std::function<bool(const Params&)>& params_valid,
    const std::function<Params(void)>& baseline_params
){
  FoldOutcome R{};

  auto eval_train = [&](const Params& p)->double { return train_metric(train, p); };
  auto tr = train_until(crit, init, eval_train, propose);

  R.training_success   = tr.success;
  R.train_iters        = tr.iters;
  R.train_metric_best  = tr.best_metric;
  R.params_used        = tr.best_params;

  // Fallback params if training didn't meet criteria
  if (!R.training_success) {
    if (prev_fold_params) {
      R.params_used      = *prev_fold_params;
      R.params_from_prev = true;
    } else if (!params_valid(R.params_used)) {
      R.params_used = baseline_params();
    }
  }

  // OOS always runs unless we truly cannot evaluate
  if (!oos.has_data()) {
    R.oos_valid = false;
    R.oos_reason = "no_data";
    return R;
  }
  if (!params_valid(R.params_used)) {
    R.oos_valid = false;
    R.oos_reason = "invalid_params";
    return R;
  }

  auto res = run_oos(oos, R.params_used);
  R.oos_valid  = res.valid;
  R.oos_metric = res.metric;
  R.oos_reason = res.reason.empty()? "ok" : res.reason;

  return R;
}

// -------------------------------
// Orchestrate full WF run (optionally parallel across folds)
// -------------------------------
struct WfRunSummary {
  int folds = 0;
  int folds_ok = 0;
  double oos_metric_sum = 0.0;
  std::vector<FoldOutcome> outcomes;
};

inline WfRunSummary run_walk_forward(
    Date global_start, Date global_end, const WfCfg& cfg,
    const Params& init, const TrainCriteria& crit,
    const std::function<DataSlice(Date,Date)>& get_slice, // build slices from dates
    const std::function<double(const DataSlice&, const Params&)>& train_metric,
    const std::function<OOSResult(const DataSlice&, const Params&)>& run_oos,
    const std::function<Params(const Params&, double)>& propose,
    const std::function<bool(const Params&)>& params_valid,
    const std::function<Params(void)>& baseline_params,
    int max_concurrency = std::max(1u, std::thread::hardware_concurrency()-1)
){
  auto windows = make_wf_windows(global_start, global_end, cfg);
  const int K = (int)windows.size();
  WfRunSummary sum; sum.folds = K; sum.outcomes.resize(K);

  std::mutex m_prev;
  std::optional<Params> prev_params{};

  // simple async throttle
  std::deque<std::pair<int, std::future<FoldOutcome>>> q;

  auto submit = [&](int idx){
    auto [ts, tr_end, oos_end] = windows[idx];
    DataSlice train{ts, tr_end};
    DataSlice oos  {tr_end, oos_end};

    Params init_for_fold = init;
    {
      std::lock_guard<std::mutex> lk(m_prev);
      if (prev_params) init_for_fold = *prev_params; // warm start with last
    }

    return std::async(std::launch::async, [=, &get_slice, &train_metric, &run_oos, &propose, &params_valid, &baseline_params]{
      // materialize slices (or pass views) from your data store
      auto train_slice = get_slice(train.start, train.end);
      auto oos_slice   = get_slice(oos.start, oos.end);

      auto out = run_wf_fold(train_slice, oos_slice, init_for_fold, crit, prev_params,
                             train_metric, run_oos, propose, params_valid, baseline_params);
      return out;
    });
  };

  auto drain_one = [&](){
    auto idx = q.front().first;
    auto& fut = q.front().second;
    auto out = fut.get();
    sum.outcomes[idx] = out;
    if (out.oos_valid) { sum.folds_ok++; sum.oos_metric_sum += out.oos_metric; }
    // record last successful params as warm start
    if (out.training_success) {
      std::lock_guard<std::mutex> lk(m_prev);
      prev_params = out.params_used;
    }
    q.pop_front();
  };

  for (int i=0; i<K; ++i) {
    q.emplace_back(i, submit(i));
    if ((int)q.size() >= max_concurrency) {
      q.front().second.wait();
      drain_one();
    }
  }
  while (!q.empty()) {
    q.front().second.wait();
    drain_one();
  }

  return sum;
}

// -------------------------------
// Example usage (wire your hooks here)
// Remove main() and integrate into your app as needed.
// -------------------------------
#ifdef WF_EXAMPLE_MAIN
// Mock hooks (replace with real ones)
static double mock_train_metric(const DataSlice& s, const Params& p){
  // pretend Sharpe improves with longer windows up to a point
  double len = double(s.end.days - s.start.days);
  double score = (p.long_win > p.short_win ? 1.0 : -1.0) * std::tanh(len/500.0) - std::abs(p.stop_bps-50.0)/200.0;
  return score;
}
static OOSResult mock_run_oos(const DataSlice& s, const Params& p){
  (void)p;
  if (!s.has_data()) return {false, std::nan(""), "no_data"};
  double len = double(s.end.days - s.start.days);
  return {true, 0.5 * std::tanh(len/200.0), "ok"};
}
static Params mock_propose(const Params& last, double){
  Params n = last;
  // simple random-ish walk (deterministic tweak here)
  n.short_win = std::max(5, std::min(60, n.short_win + ((n.short_win%2)? +1 : -1)));
  n.long_win  = std::max(20, std::min(240, n.long_win  + ((n.long_win%3)? +2 : -2)));
  n.stop_bps  = std::clamp(n.stop_bps + ((n.stop_bps>50)? -5.0 : +5.0), 5.0, 200.0);
  return n;
}
static bool mock_params_valid(const Params& p){
  return p.short_win > 0 && p.long_win > p.short_win && p.stop_bps > 0.0;
}
static Params mock_baseline(){ return Params{20, 100, 50.0}; }
static DataSlice mock_get_slice(Date a, Date b){ return DataSlice{a,b}; }

int main(){
  WfCfg cfg; cfg.train_days=240; cfg.oos_days=60; cfg.step_days=60;
  TrainCriteria crit; crit.max_iters=60; crit.patience=10; crit.min_delta=1e-3; crit.target_metric=0.0;
  Params init{20,100,50.0};

  auto sum = run_walk_forward(
    /*global_start*/ Date{0}, /*global_end*/ Date{2000}, cfg, init, crit,
    mock_get_slice,
    mock_train_metric,
    mock_run_oos,
    mock_propose,
    mock_params_valid,
    mock_baseline,
    /*max_concurrency*/ 2
  );

  std::cout << "WF folds=" << sum.folds
            << " ok=" << sum.folds_ok
            << " avg_oos=" << (sum.folds_ok? sum.oos_metric_sum / sum.folds_ok : 0.0)
            << "\n";
  return 0;
}
#endif

// -------------------------------
// Bridge functions to match existing API
// -------------------------------
#include "sentio/wf.hpp"
#include "sentio/runner.hpp"
#include "sentio/audit.hpp"
#include "sentio/metrics.hpp"
#include <iostream>
#include <algorithm>
#include <vector>

namespace sentio {

// Convert bar index to date (simplified)
Date bar_index_to_date(int index) {
    // Assuming 390 bars per day (6.5 hours * 60 minutes)
    return Date{index / 390};
}

// Convert date to bar index (simplified)
int date_to_bar_index(Date date) {
    return date.days * 390;
}

// Bridge function to match existing API
Gate run_wf_and_gate(AuditRecorder& audit_template,
                     const SymbolTable& ST,
                     const std::vector<std::vector<Bar>>& series,
                     int base_symbol_id,
                     const RunnerCfg& rcfg,
                     const WfCfg& wcfg) {
    
    Gate gate;
    
    // Validate step_days to prevent infinite loop
    if (wcfg.step_days <= 0) {
        std::cerr << "ERROR: Walk-forward step_days must be positive. Got: "
                  << wcfg.step_days << std::endl;
        gate.recommend = "REJECT - INVALID CONFIG";
        return gate;
    }
    
    const auto& base_series = series[base_symbol_id];
    const int total_bars = (int)base_series.size();
    if (total_bars < wcfg.train_days + wcfg.oos_days) {
        std::cerr << "Insufficient data for walk-forward test" << std::endl;
        return gate;
    }
    
    std::vector<double> oos_returns;
    [[maybe_unused]] int successful_folds = 0;
    
    // Walk-forward testing
    int total_iterations = (total_bars - wcfg.train_days - wcfg.oos_days) / wcfg.step_days;
    int current_iteration = 0;
    
    std::cerr << "Starting WF test with " << total_iterations << " iterations..." << std::endl;
    
    for (int start_idx = 0; start_idx < total_bars - wcfg.train_days - wcfg.oos_days; start_idx += wcfg.step_days) {
        current_iteration++;
        if (current_iteration % 10 == 0) {
            std::cerr << "Progress: " << current_iteration << "/" << total_iterations << " iterations completed" << std::endl;
        }
        
        int train_end = start_idx + wcfg.train_days;
        int oos_start = train_end;
        int oos_end = std::min(oos_start + wcfg.oos_days, total_bars);
        
        if (oos_end - oos_start < 5) continue; // Skip if insufficient OOS data
        
        // Create training data slice
        std::vector<std::vector<Bar>> train_series = series;
        for (auto& sym_series : train_series) {
            if (sym_series.size() > static_cast<size_t>(train_end)) {
                sym_series.resize(train_end);
            }
        }
        
        // Create OOS data slice
        std::vector<std::vector<Bar>> oos_series = series;
        for (auto& sym_series : oos_series) {
            if (sym_series.size() > static_cast<size_t>(oos_end)) {
                sym_series.erase(sym_series.begin(), sym_series.begin() + oos_start);
                sym_series.resize(oos_end - oos_start);
            }
        }
        
        // Run training backtest
        if (current_iteration % 50 == 0) {
            std::cerr << "Running training backtest for iteration " << current_iteration << "..." << std::endl;
        }
        
        // Create new audit recorder for training
        AuditConfig train_cfg = audit_template.get_config();
        train_cfg.run_id = "wf_train_" + std::to_string(start_idx);
        train_cfg.file_path = "audit/wf_train_" + std::to_string(start_idx) + ".jsonl";
        AuditRecorder au_train(train_cfg);
        
        auto train_result = run_backtest(au_train, ST, train_series, base_symbol_id, rcfg);
        
        // Always run OOS backtest (removed gate mechanism)
        AuditConfig oos_cfg = audit_template.get_config();
        oos_cfg.run_id = "wf_oos_" + std::to_string(start_idx);
        oos_cfg.file_path = "audit/wf_oos_" + std::to_string(start_idx) + ".jsonl";
        AuditRecorder au_oos(oos_cfg);
        
        auto oos_result = run_backtest(au_oos, ST, oos_series, base_symbol_id, rcfg);
        
        // Store OOS results
        oos_returns.push_back(oos_result.total_return);
        successful_folds++;
        
        // Track if any training period was profitable (for WF pass criteria)
        if (train_result.sharpe_ratio > 0.1 && train_result.total_return > -0.1) {
            gate.wf_pass = true;
        }
    }
    
    // Calculate OOS statistics
    if (!oos_returns.empty()) {
        double avg_return = 0.0;
        for (double ret : oos_returns) avg_return += ret;
        avg_return /= oos_returns.size();
        
        gate.oos_monthly_avg_return = avg_return * 12.0; // Annualized
        gate.oos_pass = gate.oos_monthly_avg_return > 0.0;
        
        // Calculate Sharpe ratio (simplified)
        double variance = 0.0;
        for (double ret : oos_returns) {
            double diff = ret - avg_return;
            variance += diff * diff;
        }
        variance /= oos_returns.size();
        double std_dev = std::sqrt(variance);
        
        if (std_dev > 0.0) {
            gate.oos_sharpe = avg_return / std_dev * std::sqrt(252.0);
        }
        
        // Determine recommendation
        if (gate.wf_pass && gate.oos_pass) {
            gate.recommend = "PAPER";
        } else if (gate.wf_pass) {
            gate.recommend = "ITERATE";
        } else {
            gate.recommend = "REJECT";
        }
    }
    
    return gate;
}

// Walk-forward testing with optimization (simplified for now)
Gate run_wf_and_gate_optimized(AuditRecorder& audit_template,
                               const SymbolTable& ST,
                               const std::vector<std::vector<Bar>>& series,
                               int base_symbol_id,
                               const RunnerCfg& base_rcfg,
                               const WfCfg& wcfg) {
    // For now, just run the basic WF without optimization
    // TODO: Implement parameter optimization
    return run_wf_and_gate(audit_template, ST, series, base_symbol_id, base_rcfg, wcfg);
}

} // namespace sentio
```

## 📄 **FILE 80 of 92**: tests/test_audit_replay.cpp

**File Information**:
- **Path**: `tests/test_audit_replay.cpp`

- **Size**: 56 lines
- **Modified**: 2025-09-05 20:13:14

- **Type**: .cpp

```text
#include "../include/sentio/audit.hpp"
#include <cassert>
#include <cstdio>
using namespace sentio;

int main(){
  const char* path="audit_test.jsonl";
  {
    AuditRecorder ar({.run_id="t1", .file_path=path, .flush_each=true});
    ar.event_run_start(1000);
    ar.event_bar(1000,"QQQ",AuditBar{100,101,99,100,0});
    ar.event_signal(1000,"QQQ",SigType::BUY,0.8);
    ar.event_route(1000,"QQQ","TQQQ",0.05);
    ar.event_order(1000,"TQQQ",Side::Buy,10,0.0);
    ar.event_fill(1000,"TQQQ",30.0,10,0.1,Side::Buy); // buy 10 @ 30
    ar.event_bar(1060,"TQQQ",AuditBar{30,31,29,31,0});
    ar.event_signal(1120,"QQQ",SigType::SELL,0.8);
    ar.event_route(1120,"QQQ","TQQQ",-0.05);
    ar.event_order(1120,"TQQQ",Side::Sell,10,0.0);
    ar.event_fill(1120,"TQQQ",31.5,10,0.1,Side::Sell); // sell 10 @ 31.5
    ar.event_run_end(1200);
  }
  auto rr = AuditReplayer::replay_file(path,"t1");
  assert(rr.has_value());
  
  // Debug output
  printf("Cash: %.6f\n", rr->acct.cash);
  printf("Realized: %.6f\n", rr->acct.realized);
  printf("Equity: %.6f\n", rr->acct.equity);
  printf("Bars: %zu, Signals: %zu, Routes: %zu, Orders: %zu, Fills: %zu\n", 
         rr->bars, rr->signals, rr->routes, rr->orders, rr->fills);
  
  // Expected calculation:
  // Buy 10 @ 30: cash = -30*10 - 0.1 = -300.1, position = +10 @ 30
  // Sell 10 @ 31.5: cash = +31.5*10 - 0.1 = 314.9, position = 0
  // Total cash = -300.1 + 314.9 = 14.8
  // Realized P&L = (31.5 - 30) * 10 = 15.0
  // Equity = cash + realized + mtm = 14.8 + 15.0 + 0 = 29.8
  
  // Check that we have the expected number of events
  assert(rr->bars >= 2);
  assert(rr->signals >= 2);
  assert(rr->routes >= 2);
  assert(rr->orders >= 2);
  assert(rr->fills >= 2);
  
  // Check that we have some realized P&L (should be positive)
  assert(rr->acct.realized > 0.0);
  
  // Check that equity is reasonable
  assert(rr->acct.equity > 0.0);
  
  // Clean up test file
  std::remove(path);
  return 0;
}

```

## 📄 **FILE 81 of 92**: tests/test_audit_simple.cpp

**File Information**:
- **Path**: `tests/test_audit_simple.cpp`

- **Size**: 38 lines
- **Modified**: 2025-09-05 19:42:58

- **Type**: .cpp

```text
#include "../include/sentio/audit.hpp"
#include <cassert>
#include <cstdio>
using namespace sentio;

int main(){
  const char* path="audit_simple_test.jsonl";
  {
    AuditRecorder ar({.run_id="simple", .file_path=path, .flush_each=true});
    ar.event_run_start(1000);
    ar.event_bar(1000,"TQQQ",AuditBar{30,31,29,30,1000});
    ar.event_order(1000,"TQQQ",Side::Buy,10,0.0);
    ar.event_fill(1000,"TQQQ",30.0,10,0.1,Side::Buy); // buy 10 @ 30
    ar.event_bar(1060,"TQQQ",AuditBar{30,31,29,31,1000});
    ar.event_order(1060,"TQQQ",Side::Sell,10,0.0);
    ar.event_fill(1060,"TQQQ",31.0,10,0.1,Side::Sell); // sell 10 @ 31
    ar.event_run_end(1200);
  }
  auto rr = AuditReplayer::replay_file(path,"simple");
  assert(rr.has_value());
  
  // Debug output
  printf("Cash: %.6f\n", rr->acct.cash);
  printf("Realized: %.6f\n", rr->acct.realized);
  printf("Equity: %.6f\n", rr->acct.equity);
  printf("Bars: %zu, Fills: %zu\n", rr->bars, rr->fills);
  
  // Expected:
  // Buy 10 @ 30: cash = -30*10 - 0.1 = -300.1
  // Sell 10 @ 31: cash = +31*10 - 0.1 = 309.9
  // Total cash = -300.1 + 309.9 = 9.8
  // Realized P&L = (31 - 30) * 10 = 10.0
  // Equity = 9.8 + 10.0 = 19.8
  
  // Clean up test file
  std::remove(path);
  return 0;
}

```

## 📄 **FILE 82 of 92**: tests/test_pipeline_emits.cpp

**File Information**:
- **Path**: `tests/test_pipeline_emits.cpp`

- **Size**: 67 lines
- **Modified**: 2025-09-05 17:05:51

- **Type**: .cpp

```text
#include "../include/sentio/strategy_sma_cross.hpp"
#include "../include/sentio/signal_pipeline.hpp"
#include <cassert>
#include <vector>
#include <iostream>
#include <unordered_map>

using namespace sentio;

struct PB {
  void upsert_latest(const std::string& i, const Bar& b) { latest[i]=b; }
  const Bar* get_latest(const std::string& i) const { auto it=latest.find(i); return it==latest.end()?nullptr:&it->second; }
  bool has_instrument(const std::string& i) const { return latest.count(i)>0; }
  std::size_t size() const { return latest.size(); }
  std::unordered_map<std::string,Bar> latest;
};

int main() {
  std::cout << "🧪 Testing Signal Pipeline End-to-End\n";
  std::cout << "====================================\n\n";
  
  PB book;
  SMACrossCfg scfg{5, 10, 0.8};
  SMACrossStrategy strat(scfg);

  SignalTrace trace;
  PipelineCfg pcfg;
  pcfg.gate = GateCfg{true, 0, 0.01};
  pcfg.min_order_shares = 1.0;

  SignalPipeline pipe(&strat, pcfg, &book, &trace);
  // Simple account struct to avoid include conflicts
  struct SimpleAccount { double equity=100000; double cash=100000; };
  SimpleAccount acct;

  // Rising closes to trigger BUY cross; book price for routed instruments
  std::vector<double> closes;
  for (int i=0;i<30;++i) closes.push_back(100+i*0.5);

  int emits=0;
  for (int i=0;i<(int)closes.size();++i) {
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=1'000'000+i*60, .is_rth=true};
    Bar b{closes[i],closes[i],closes[i],closes[i]};
    // keep both QQQ and TQQQ book updated to avoid "no price"
    book.upsert_latest("QQQ", b);
    book.upsert_latest("TQQQ", Bar{b.open*3,b.high*3,b.low*3,b.close*3});
    auto out = pipe.on_bar(ctx,b,&acct);
    if (out.signal) ++emits;
  }

  std::cout << "📊 Pipeline Test Results:\n";
  std::cout << "  Signals emitted: " << emits << "\n";
  std::cout << "  Total trace rows: " << trace.rows().size() << "\n";
  std::cout << "  OK signals: " << trace.count(TraceReason::OK) << "\n";
  std::cout << "  No strategy output: " << trace.count(TraceReason::NO_STRATEGY_OUTPUT) << "\n";
  std::cout << "  Threshold too tight: " << trace.count(TraceReason::THRESHOLD_TOO_TIGHT) << "\n";
  std::cout << "  WARMUP: " << trace.count(TraceReason::WARMUP) << "\n";
  std::cout << "  NOT_RTH: " << trace.count(TraceReason::NOT_RTH) << "\n";
  std::cout << "  NAN_INPUT: " << trace.count(TraceReason::NAN_INPUT) << "\n";
  std::cout << "  Other reasons: " << trace.count(TraceReason::UNKNOWN) << "\n\n";

  assert(emits>=1);
  assert(trace.count(TraceReason::OK)>=1);
  
  std::cout << "✅ Test passed! Signal pipeline successfully emitted signals.\n";
  return 0;
}

```

## 📄 **FILE 83 of 92**: tests/test_sma_cross_emit.cpp

**File Information**:
- **Path**: `tests/test_sma_cross_emit.cpp`

- **Size**: 52 lines
- **Modified**: 2025-09-05 16:36:36

- **Type**: .cpp

```text
#include "../include/sentio/strategy_sma_cross.hpp"
#include "../include/sentio/signal_engine.hpp"
#include <cassert>
#include <vector>
#include <iostream>

using namespace sentio;

int main() {
  std::cout << "🧪 Testing SMA Cross Strategy Signal Emission\n";
  
  // Rising series should eventually trigger a BUY (golden cross)
  SMACrossCfg cfg{5, 10, 0.8};
  SMACrossStrategy strat(cfg);
  SignalHealth health;
  GateCfg gate{.require_rth=true, .cooldown_bars=0, .min_conf=0.01};
  SignalEngine engine(&strat, gate, &health);

  std::vector<double> closes;
  for (int i=0;i<25;++i) closes.push_back(100.0 + i*0.5);

  std::optional<StrategySignal> last;
  int emits=0;
  for (int i=0;i<(int)closes.size();++i) {
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=1'000'000+i*60, .is_rth=true};
    Bar b{closes[i], closes[i], closes[i], closes[i]};
    auto out = engine.on_bar(ctx, b, /*inputs_finite=*/true);
    if (out.signal) { 
      last = out.signal; 
      ++emits; 
      std::cout << "📈 Signal emitted at bar " << i << ": " 
                << (out.signal->type == StrategySignal::Type::BUY ? "BUY" : "SELL")
                << " (conf: " << out.signal->confidence << ")\n";
    }
  }

  std::cout << "📊 Final Results:\n";
  std::cout << "  Signals emitted: " << emits << "\n";
  std::cout << "  Signals dropped: " << health.dropped.load() << "\n";
  std::cout << "  Health by reason:\n";
  for (const auto& [reason, count] : health.by_reason) {
    if (count.load() > 0) {
      std::cout << "    " << static_cast<int>(reason) << ": " << count.load() << "\n";
    }
  }

  assert(emits >= 1);
  assert(last && (last->type == StrategySignal::Type::BUY));
  
  std::cout << "✅ Test passed! SMA Cross strategy successfully emitted signals.\n";
  return 0;
}

```

## 📄 **FILE 84 of 92**: tools/create_mega_document.py

**File Information**:
- **Path**: `tools/create_mega_document.py`

- **Size**: 104 lines
- **Modified**: 2025-09-05 03:58:29

- **Type**: .py

```text
#!/usr/bin/env python3
"""
Create mega document from source files.
"""

import os
import argparse
import datetime
from pathlib import Path

def create_mega_document(directories, title, description, output, include_bug_report=False, bug_report_file=None):
    """Create a mega document from source files."""
    
    print(f"🔧 Creating mega document: {output}")
    print(f"📁 Source directory: {os.getcwd()}")
    print(f"📁 Output file: {output}")
    
    # Collect all files
    all_files = []
    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.hpp', '.cpp', '.h', '.c', '.py', '.md', '.txt', '.cmake', 'CMakeLists.txt')):
                        file_path = os.path.join(root, file)
                        all_files.append(file_path)
    
    print(f"📁 Files to include: {len(all_files)}")
    
    # Sort files for consistent ordering
    all_files.sort()
    
    # Create mega document
    with open(output, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Source Directory**: {os.getcwd()}\n")
        f.write(f"**Description**: {description}\n\n")
        f.write(f"**Total Files**: {len(all_files)}\n\n")
        f.write("---\n\n")
        
        # Include bug report if requested
        if include_bug_report and bug_report_file and os.path.exists(bug_report_file):
            f.write("## 🐛 **BUG REPORT**\n\n")
            with open(bug_report_file, 'r', encoding='utf-8') as bug_f:
                f.write(bug_f.read())
            f.write("\n\n---\n\n")
        
        # Table of contents
        f.write("## 📋 **TABLE OF CONTENTS**\n\n")
        for i, file_path in enumerate(all_files, 1):
            f.write(f"{i}. [{file_path}](#file-{i})\n")
        f.write("\n---\n\n")
        
        # File contents
        for i, file_path in enumerate(all_files, 1):
            try:
                with open(file_path, 'r', encoding='utf-8') as file_f:
                    content = file_f.read()
                
                f.write(f"## 📄 **FILE {i} of {len(all_files)}**: {file_path}\n\n")
                f.write("**File Information**:\n")
                f.write(f"- **Path**: `{file_path}`\n\n")
                f.write(f"- **Size**: {len(content.splitlines())} lines\n")
                f.write(f"- **Modified**: {datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"- **Type**: {Path(file_path).suffix}\n\n")
                f.write("```text\n")
                f.write(content)
                f.write("\n```\n\n")
                
                print(f"📄 Processing file {i}/{len(all_files)}: {file_path}")
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                f.write(f"## 📄 **FILE {i} of {len(all_files)}**: {file_path}\n\n")
                f.write(f"**Error**: Could not read file - {e}\n\n")
    
    print(f"✅ Mega document created: {output}")
    print(f"📊 Output size: {os.path.getsize(output) / 1024:.1f} KB")
    print(f"📊 Files processed: {len(all_files)}/{len(all_files)}")
    print(f"📊 Content size: {sum(os.path.getsize(f) for f in all_files if os.path.exists(f)) / 1024:.1f} KB")
    print(f"\n🎯 Success! Mega document created:")
    print(f"{output}")
    print(f"\n📁 Location: {os.path.abspath(output)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mega document from source files")
    parser.add_argument("--directories", "-d", nargs="+", required=True, help="Directories to include")
    parser.add_argument("--title", "-t", required=True, help="Document title")
    parser.add_argument("--description", "-desc", required=True, help="Document description")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--include-bug-report", action="store_true", help="Include bug report")
    parser.add_argument("--bug-report-file", help="Bug report file path")
    
    args = parser.parse_args()
    
    create_mega_document(
        args.directories,
        args.title,
        args.description,
        args.output,
        args.include_bug_report,
        args.bug_report_file
    )

```

## 📄 **FILE 85 of 92**: tools/data_downloader.py

**File Information**:
- **Path**: `tools/data_downloader.py`

- **Size**: 205 lines
- **Modified**: 2025-09-05 12:56:12

- **Type**: .py

```text
import os
import argparse
import requests
import pandas as pd
import pandas_market_calendars as mcal
import struct
from datetime import datetime
from pathlib import Path

# --- Constants ---
# Define the Regular Trading Hours for NYSE in New York time.
RTH_START = "09:30"
RTH_END = "16:00"
NY_TIMEZONE = "America/New_York"
POLYGON_API_BASE = "https://api.polygon.io"

def fetch_aggs_all(symbol, start_date, end_date, api_key, timespan="minute", multiplier=1):
    """
    Fetches all aggregate bars for a symbol within a date range from Polygon.io.
    Handles API pagination automatically.
    """
    print(f"Fetching '{symbol}' data from {start_date} to {end_date}...")
    url = (
        f"{POLYGON_API_BASE}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/"
        f"{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000"
    )
    
    headers = {"Authorization": f"Bearer {api_key}"}
    all_bars = []
    
    while url:
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            if "results" in data:
                all_bars.extend(data["results"])
                print(f" -> Fetched {len(data['results'])} bars...", end="\r")

            url = data.get("next_url")

        except requests.exceptions.RequestException as e:
            print(f"\nAPI Error fetching data for {symbol}: {e}")
            return None
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            return None
            
    print(f"\n -> Total bars fetched for {symbol}: {len(all_bars)}")
    if not all_bars:
        return None
        
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(all_bars)
    df.rename(columns={
        't': 'timestamp_utc_ms',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume'
    }, inplace=True)
    return df

def filter_and_prepare_data(df):
    """
    Filters a DataFrame of market data for RTH (Regular Trading Hours)
    and removes US market holidays.
    """
    if df is None or df.empty:
        return None

    print("Filtering data for RTH and US market holidays...")
    
    # 1. Convert UTC millisecond timestamp to a timezone-aware DatetimeIndex
    df['timestamp_utc_ms'] = pd.to_datetime(df['timestamp_utc_ms'], unit='ms', utc=True)
    df.set_index('timestamp_utc_ms', inplace=True)
    
    # 2. Convert the index to New York time to perform RTH and holiday checks
    df.index = df.index.tz_convert(NY_TIMEZONE)
    
    # 3. Filter for Regular Trading Hours
    df = df.between_time(RTH_START, RTH_END)

    # 4. Filter out US market holidays
    nyse = mcal.get_calendar('NYSE')
    holidays = nyse.holidays().holidays # Get a list of holiday dates
    df = df[~df.index.normalize().isin(holidays)]
    
    print(f" -> {len(df)} bars remaining after filtering.")
    
    # 5. Add the specific columns required by the C++ backtester
    df['ts_utc'] = df.index.strftime('%Y-%m-%dT%H:%M:%S%z').str.replace(r'([+-])(\d{2})(\d{2})', r'\1\2:\3', regex=True)
    df['ts_nyt_epoch'] = df.index.astype('int64') // 10**9
    
    return df

def save_to_bin(df, path):
    """
    Saves the DataFrame to a custom binary format compatible with the C++ backtester.
    Format:
    - uint64_t: Number of bars
    - For each bar:
      - uint32_t: Length of ts_utc string
      - char[]: ts_utc string data
      - int64_t: ts_nyt_epoch
      - double: open, high, low, close
      - uint64_t: volume
    """
    print(f"Saving to binary format at {path}...")
    try:
        with open(path, 'wb') as f:
            # Write total number of bars
            num_bars = len(df)
            f.write(struct.pack('<Q', num_bars))

            # **FIXED**: The struct format string now correctly includes six format
            # specifiers to match the six arguments passed to pack().
            # q: int64_t (ts_nyt_epoch)
            # d: double (open)
            # d: double (high)
            # d: double (low)
            # d: double (close)
            # Q: uint64_t (volume)
            bar_struct = struct.Struct('<qddddQ')

            for row in df.itertuples():
                # Handle the variable-length string part
                ts_utc_bytes = row.ts_utc.encode('utf-8')
                f.write(struct.pack('<I', len(ts_utc_bytes)))
                f.write(ts_utc_bytes)
                
                # Pack and write the fixed-size data
                packed_data = bar_struct.pack(
                    row.ts_nyt_epoch,
                    row.open,
                    row.high,
                    row.low,
                    row.close,
                    int(row.volume) # C++ expects uint64_t, so we cast to int
                )
                f.write(packed_data)
        print(" -> Binary file saved successfully.")
    except Exception as e:
        print(f"Error saving binary file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Polygon.io Data Downloader and Processor")
    parser.add_argument('symbols', nargs='+', help="One or more stock symbols (e.g., QQQ TQQQ SQQQ)")
    parser.add_argument('--start', required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument('--end', required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument('--outdir', default='data', help="Output directory for CSV and BIN files")
    parser.add_argument('--timespan', default='minute', choices=['minute', 'hour', 'day'], help="Timespan of bars")
    parser.add_argument('--multiplier', default=1, type=int, help="Multiplier for the timespan")
    
    args = parser.parse_args()
    
    # Get API key from environment variable for security
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("Error: POLYGON_API_KEY environment variable not set.")
        return
        
    # Create output directory if it doesn't exist
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for symbol in args.symbols:
        print("-" * 50)
        # 1. Fetch data
        df_raw = fetch_aggs_all(symbol, args.start, args.end, api_key, args.timespan, args.multiplier)
        
        if df_raw is None or df_raw.empty:
            print(f"No data fetched for {symbol}. Skipping.")
            continue
            
        # 2. Filter and prepare data
        df_clean = filter_and_prepare_data(df_raw)
        
        if df_clean is None or df_clean.empty:
            print(f"No data remaining for {symbol} after filtering. Skipping.")
            continue
        
        # 3. Define output paths
        file_prefix = f"{symbol.upper()}_RTH_NH"
        csv_path = output_dir / f"{file_prefix}.csv"
        bin_path = output_dir / f"{file_prefix}.bin"
        
        # 4. Save to CSV for inspection
        print(f"Saving to CSV format at {csv_path}...")
        # Select and order columns to match C++ struct for clarity
        csv_columns = ['ts_utc', 'ts_nyt_epoch', 'open', 'high', 'low', 'close', 'volume']
        df_clean[csv_columns].to_csv(csv_path, index=False)
        print(" -> CSV file saved successfully.")
        
        # 5. Save to C++ compatible binary format
        save_to_bin(df_clean, bin_path)

    print("-" * 50)
    print("Data download and processing complete.")

if __name__ == "__main__":
    main()


```

## 📄 **FILE 86 of 92**: tools/detailed_strategy_diagnostics.cpp

**File Information**:
- **Path**: `tools/detailed_strategy_diagnostics.cpp`

- **Size**: 112 lines
- **Modified**: 2025-09-05 17:26:02

- **Type**: .cpp

```text
#include "../include/sentio/strategy_market_making.hpp"
#include "../include/sentio/core.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace sentio;

void test_strategy_detailed(const std::string& strategy_name, BaseStrategy* strategy, const std::vector<double>& closes) {
  std::cout << "\n🔍 DETAILED Testing " << strategy_name << " Strategy\n";
  std::cout << "==========================================\n";
  
  // Create a vector of bars
  std::vector<Bar> bars;
  for (int i = 0; i < (int)closes.size(); ++i) {
    Bar b;
    b.ts_utc = std::to_string(1'000'000 + i * 60);
    b.ts_nyt_epoch = 1'000'000 + i * 60;
    b.open = closes[i];
    b.high = closes[i] * 1.001;  // Slight high
    b.low = closes[i] * 0.999;   // Slight low
    b.close = closes[i];
    b.volume = 1000;
    bars.push_back(b);
  }
  
  int signals_emitted = 0;
  std::cout << "Detailed bar-by-bar analysis:\n";
  std::cout << "Bar | Price  | Signal | Confidence | Reason\n";
  std::cout << "----|--------|--------|------------|-------\n";
  
  for (int i = 0; i < (int)bars.size(); ++i) {
    // Calculate signal for this bar
    StrategySignal signal = strategy->calculate_signal(bars, i);
    
    std::cout << std::setw(3) << i << " | " 
              << std::fixed << std::setprecision(2) << std::setw(6) << bars[i].close << " | ";
    
    if (signal.type != StrategySignal::Type::HOLD) {
      signals_emitted++;
      switch (signal.type) {
        case StrategySignal::Type::BUY: std::cout << "BUY    "; break;
        case StrategySignal::Type::SELL: std::cout << "SELL   "; break;
        case StrategySignal::Type::STRONG_BUY: std::cout << "S_BUY  "; break;
        case StrategySignal::Type::STRONG_SELL: std::cout << "S_SELL "; break;
        default: std::cout << "UNKNOWN"; break;
      }
      std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(10) << signal.confidence << " | SUCCESS";
    } else {
      std::cout << "NONE   | " << std::fixed << std::setprecision(3) << std::setw(10) << signal.confidence << " | DROPPED";
    }
    std::cout << "\n";
    
    // Show first 10 bars and every 10th bar after that
    if (i >= 10 && i % 10 != 0) continue;
  }
  
  std::cout << "\n📊 Summary for " << strategy_name << ":\n";
  std::cout << "  Total bars processed: " << bars.size() << "\n";
  std::cout << "  Signals emitted: " << signals_emitted << "\n";
  std::cout << "  Success rate: " << std::fixed << std::setprecision(2) 
            << (bars.size() > 0 ? (signals_emitted * 100.0 / bars.size()) : 0.0) << "%\n";
  
  // Get diagnostic info
  const SignalDiag& diag = strategy->get_diag();
  std::cout << "  Diagnostic info:\n";
  std::cout << "    Signals emitted: " << diag.emitted << "\n";
  std::cout << "    Signals dropped: " << diag.dropped << "\n";
  std::cout << "    Drop reasons:\n";
  std::cout << "      Min bars: " << diag.r_min_bars << "\n";
  std::cout << "      Session: " << diag.r_session << "\n";
  std::cout << "      NaN input: " << diag.r_nan << "\n";
  std::cout << "      Zero volume: " << diag.r_zero_vol << "\n";
  std::cout << "      Threshold: " << diag.r_threshold << "\n";
  std::cout << "      Cooldown: " << diag.r_cooldown << "\n";
  std::cout << "      Duplicate: " << diag.r_dup << "\n";
}

int main() {
  std::cout << "🔍 DETAILED STRATEGY DIAGNOSTIC TOOL\n";
  std::cout << "====================================\n";
  std::cout << "Testing problematic strategies with detailed bar-by-bar analysis\n\n";
  
  // Create realistic test data with some volatility
  std::vector<double> closes;
  double base_price = 100.0;
  srand(42); // For reproducible results
  
  for (int i = 0; i < 50; ++i) { // Smaller dataset for detailed analysis
    // Add some realistic price movement with occasional spikes
    double trend = i * 0.2;  // Stronger upward trend
    double noise = (rand() % 100 - 50) * 0.02;  // More noise
    double spike = (i % 10 == 0) ? (rand() % 100 - 50) * 0.1 : 0;  // More frequent spikes
    closes.push_back(base_price + trend + noise + spike);
  }
  
  std::cout << "📈 Test data: " << closes.size() << " bars, price range: " 
            << *std::min_element(closes.begin(), closes.end()) << " - "
            << *std::max_element(closes.begin(), closes.end()) << "\n\n";
  
  // Test Market Making Strategy
  std::cout << "Creating Market Making Strategy...\n";
  MarketMakingStrategy mm_strategy;
  test_strategy_detailed("Market Making", &mm_strategy, closes);
  
  std::cout << "\n🎯 DETAILED DIAGNOSTIC COMPLETE\n";
  std::cout << "================================\n";
  std::cout << "Check the detailed bar-by-bar analysis above to see exactly\n";
  std::cout << "what's happening in each bar and why signals are being dropped.\n";
  
  return 0;
}

```

## 📄 **FILE 87 of 92**: tools/extended_strategy_test.cpp

**File Information**:
- **Path**: `tools/extended_strategy_test.cpp`

- **Size**: 113 lines
- **Modified**: 2025-09-05 17:26:11

- **Type**: .cpp

```text
#include "../include/sentio/strategy_market_making.hpp"
#include "../include/sentio/core.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace sentio;

void test_strategy_extended(const std::string& strategy_name, BaseStrategy* strategy, const std::vector<double>& closes) {
  std::cout << "\n🔍 EXTENDED Testing " << strategy_name << " Strategy\n";
  std::cout << "==========================================\n";
  
  // Create a vector of bars
  std::vector<Bar> bars;
  for (int i = 0; i < (int)closes.size(); ++i) {
    Bar b;
    b.ts_utc = std::to_string(1'000'000 + i * 60);
    b.ts_nyt_epoch = 1'000'000 + i * 60;
    b.open = closes[i];
    b.high = closes[i] * 1.001;  // Slight high
    b.low = closes[i] * 0.999;   // Slight low
    b.close = closes[i];
    b.volume = 1000;
    bars.push_back(b);
  }
  
  int signals_emitted = 0;
  std::cout << "Extended bar-by-bar analysis (showing every 10th bar):\n";
  std::cout << "Bar | Price  | Signal | Confidence | Reason\n";
  std::cout << "----|--------|--------|------------|-------\n";
  
  for (int i = 0; i < (int)bars.size(); ++i) {
    // Calculate signal for this bar
    StrategySignal signal = strategy->calculate_signal(bars, i);
    
    // Show every 10th bar and any bars with signals
    if (i % 10 == 0 || signal.type != StrategySignal::Type::HOLD) {
      std::cout << std::setw(3) << i << " | " 
                << std::fixed << std::setprecision(2) << std::setw(6) << bars[i].close << " | ";
      
      if (signal.type != StrategySignal::Type::HOLD) {
        signals_emitted++;
        switch (signal.type) {
          case StrategySignal::Type::BUY: std::cout << "BUY    "; break;
          case StrategySignal::Type::SELL: std::cout << "SELL   "; break;
          case StrategySignal::Type::STRONG_BUY: std::cout << "S_BUY  "; break;
          case StrategySignal::Type::STRONG_SELL: std::cout << "S_SELL "; break;
          default: std::cout << "UNKNOWN"; break;
        }
        std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(10) << signal.confidence << " | SUCCESS";
      } else {
        std::cout << "NONE   | " << std::fixed << std::setprecision(3) << std::setw(10) << signal.confidence << " | DROPPED";
      }
      std::cout << "\n";
    }
  }
  
  std::cout << "\n📊 Summary for " << strategy_name << ":\n";
  std::cout << "  Total bars processed: " << bars.size() << "\n";
  std::cout << "  Signals emitted: " << signals_emitted << "\n";
  std::cout << "  Success rate: " << std::fixed << std::setprecision(2) 
            << (bars.size() > 0 ? (signals_emitted * 100.0 / bars.size()) : 0.0) << "%\n";
  
  // Get diagnostic info
  const SignalDiag& diag = strategy->get_diag();
  std::cout << "  Diagnostic info:\n";
  std::cout << "    Signals emitted: " << diag.emitted << "\n";
  std::cout << "    Signals dropped: " << diag.dropped << "\n";
  std::cout << "    Drop reasons:\n";
  std::cout << "      Min bars: " << diag.r_min_bars << "\n";
  std::cout << "      Session: " << diag.r_session << "\n";
  std::cout << "      NaN input: " << diag.r_nan << "\n";
  std::cout << "      Zero volume: " << diag.r_zero_vol << "\n";
  std::cout << "      Threshold: " << diag.r_threshold << "\n";
  std::cout << "      Cooldown: " << diag.r_cooldown << "\n";
  std::cout << "      Duplicate: " << diag.r_dup << "\n";
}

int main() {
  std::cout << "🔍 EXTENDED STRATEGY TEST\n";
  std::cout << "=========================\n";
  std::cout << "Testing strategies with extended data to satisfy warmup requirements\n\n";
  
  // Create extended test data with more volatility
  std::vector<double> closes;
  double base_price = 100.0;
  srand(42); // For reproducible results
  
  // Generate 200 bars to ensure warmup requirements are met
  for (int i = 0; i < 200; ++i) {
    // Add some realistic price movement with occasional spikes
    double trend = i * 0.05;  // Gentle upward trend
    double noise = (rand() % 100 - 50) * 0.01;  // Random noise
    double spike = (i % 30 == 0) ? (rand() % 100 - 50) * 0.05 : 0;  // Occasional spikes
    closes.push_back(base_price + trend + noise + spike);
  }
  
  std::cout << "📈 Test data: " << closes.size() << " bars, price range: " 
            << *std::min_element(closes.begin(), closes.end()) << " - "
            << *std::max_element(closes.begin(), closes.end()) << "\n\n";
  
  // Test Market Making Strategy
  std::cout << "Creating Market Making Strategy...\n";
  MarketMakingStrategy mm_strategy;
  test_strategy_extended("Market Making", &mm_strategy, closes);
  
  std::cout << "\n🎯 EXTENDED TEST COMPLETE\n";
  std::cout << "==========================\n";
  std::cout << "This test uses 200 bars to ensure warmup requirements are satisfied.\n";
  std::cout << "If strategies still don't generate signals, the thresholds are too restrictive.\n";
  
  return 0;
}

```

## 📄 **FILE 88 of 92**: tools/integration_example.cpp

**File Information**:
- **Path**: `tools/integration_example.cpp`

- **Size**: 184 lines
- **Modified**: 2025-09-05 16:38:27

- **Type**: .cpp

```text
#include "../include/sentio/strategy_sma_cross.hpp"
#include "../include/sentio/signal_engine.hpp"
#include "../include/sentio/rth_calendar.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace sentio;

// Example integration showing how to use the diagnostic system
// with existing QQQ data to surface exactly why signals are dropped

struct QQQBar {
  double open, high, low, close;
  std::int64_t ts_utc;
  double volume;
};

// Convert QQQ data to our diagnostic format
std::vector<QQQBar> load_qqq_sample() {
  std::vector<QQQBar> bars;
  
  // Sample QQQ data (rising trend to trigger golden cross)
  std::vector<double> prices = {
    100.0, 100.5, 101.0, 101.2, 101.5, 102.0, 102.3, 102.8, 103.0, 103.5,
    104.0, 104.2, 104.8, 105.0, 105.5, 106.0, 106.3, 106.8, 107.0, 107.5,
    108.0, 108.2, 108.8, 109.0, 109.5, 110.0, 110.3, 110.8, 111.0, 111.5
  };
  
  for (size_t i = 0; i < prices.size(); ++i) {
    double price = prices[i];
    bars.push_back({
      .open = price - 0.1,
      .high = price + 0.2,
      .low = price - 0.2,
      .close = price,
      .ts_utc = static_cast<std::int64_t>(1000000 + i * 60), // 1-minute bars
      .volume = static_cast<double>(1000000 + i * 10000)
    });
  }
  
  return bars;
}

void print_detailed_health(const SignalHealth& health) {
  std::cout << "\n🔍 DETAILED SIGNAL HEALTH ANALYSIS\n";
  std::cout << "===================================\n";
  std::cout << "Total Emitted: " << health.emitted.load() << "\n";
  std::cout << "Total Dropped: " << health.dropped.load() << "\n";
  
  if (health.dropped.load() > 0) {
    std::cout << "\nDrop Reason Breakdown:\n";
    std::cout << "---------------------\n";
    
    const char* reason_names[] = {
      "NONE", "NO_DATA", "NOT_RTH", "HOLIDAY", "WARMUP", 
      "NAN_INPUT", "THRESHOLD_TOO_TIGHT", "COOLDOWN_ACTIVE", 
      "DEBOUNCE", "DUPLICATE_BAR_TS", "UNKNOWN"
    };
    
    for (const auto& [reason, count] : health.by_reason) {
      if (count.load() > 0) {
        int idx = static_cast<int>(reason);
        if (idx >= 0 && idx < 11) {
          double percentage = (double)count.load() / health.dropped.load() * 100.0;
          std::cout << "  " << std::setw(20) << reason_names[idx] << ": " 
                    << std::setw(6) << count.load() 
                    << " (" << std::fixed << std::setprecision(1) << percentage << "%)\n";
        }
      }
    }
  }
}

void run_integration_test() {
  std::cout << "🚀 SIGNAL DIAGNOSTIC INTEGRATION TEST\n";
  std::cout << "=====================================\n\n";
  
  // Load sample data
  auto qqq_bars = load_qqq_sample();
  std::cout << "📊 Loaded " << qqq_bars.size() << " QQQ bars\n\n";
  
  // Test 1: Basic SMA Cross with permissive settings
  std::cout << "Test 1: SMA Cross with permissive settings\n";
  std::cout << "------------------------------------------\n";
  
  SMACrossCfg cfg{5, 15, 0.6}; // Fast=5, Slow=15, Conf=0.6
  SMACrossStrategy strat(cfg);
  SignalHealth health1;
  GateCfg gate1{.require_rth=false, .cooldown_bars=0, .min_conf=0.01};
  SignalEngine engine1(&strat, gate1, &health1);
  
  int signals_1 = 0;
  for (size_t i = 0; i < qqq_bars.size(); ++i) {
    const auto& bar = qqq_bars[i];
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=bar.ts_utc, .is_rth=true};
    Bar b{bar.open, bar.high, bar.low, bar.close};
    auto out = engine1.on_bar(ctx, b, true);
    
    if (out.signal) {
      signals_1++;
      std::cout << "  Bar " << std::setw(2) << i << ": " 
                << (out.signal->type == StrategySignal::Type::BUY ? "BUY" : "SELL")
                << " (conf: " << std::fixed << std::setprecision(2) << out.signal->confidence << ")\n";
    }
  }
  
  print_detailed_health(health1);
  std::cout << "\nResult: " << signals_1 << " signals emitted\n\n";
  
  // Test 2: RTH validation enabled
  std::cout << "Test 2: SMA Cross with RTH validation\n";
  std::cout << "-------------------------------------\n";
  
  SMACrossStrategy strat2(cfg);
  SignalHealth health2;
  GateCfg gate2{.require_rth=true, .cooldown_bars=0, .min_conf=0.01};
  SignalEngine engine2(&strat2, gate2, &health2);
  
  int signals_2 = 0;
  for (size_t i = 0; i < qqq_bars.size(); ++i) {
    const auto& bar = qqq_bars[i];
    // Simulate RTH validation (all bars are RTH in this example)
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=bar.ts_utc, .is_rth=true};
    Bar b{bar.open, bar.high, bar.low, bar.close};
    auto out = engine2.on_bar(ctx, b, true);
    
    if (out.signal) {
      signals_2++;
      std::cout << "  Bar " << std::setw(2) << i << ": " 
                << (out.signal->type == StrategySignal::Type::BUY ? "BUY" : "SELL")
                << " (conf: " << std::fixed << std::setprecision(2) << out.signal->confidence << ")\n";
    }
  }
  
  print_detailed_health(health2);
  std::cout << "\nResult: " << signals_2 << " signals emitted\n\n";
  
  // Test 3: High confidence threshold
  std::cout << "Test 3: SMA Cross with high confidence threshold\n";
  std::cout << "------------------------------------------------\n";
  
  SMACrossStrategy strat3(cfg);
  SignalHealth health3;
  GateCfg gate3{.require_rth=false, .cooldown_bars=0, .min_conf=0.9}; // Very high threshold
  SignalEngine engine3(&strat3, gate3, &health3);
  
  int signals_3 = 0;
  for (size_t i = 0; i < qqq_bars.size(); ++i) {
    const auto& bar = qqq_bars[i];
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=bar.ts_utc, .is_rth=true};
    Bar b{bar.open, bar.high, bar.low, bar.close};
    auto out = engine3.on_bar(ctx, b, true);
    
    if (out.signal) {
      signals_3++;
      std::cout << "  Bar " << std::setw(2) << i << ": " 
                << (out.signal->type == StrategySignal::Type::BUY ? "BUY" : "SELL")
                << " (conf: " << std::fixed << std::setprecision(2) << out.signal->confidence << ")\n";
    }
  }
  
  print_detailed_health(health3);
  std::cout << "\nResult: " << signals_3 << " signals emitted\n\n";
  
  // Summary
  std::cout << "📋 INTEGRATION TEST SUMMARY\n";
  std::cout << "===========================\n";
  std::cout << "Permissive settings: " << signals_1 << " signals\n";
  std::cout << "RTH validation:     " << signals_2 << " signals\n";
  std::cout << "High threshold:     " << signals_3 << " signals\n\n";
  
  std::cout << "🎯 KEY INSIGHTS:\n";
  std::cout << "- The diagnostic system successfully identifies why signals are dropped\n";
  std::cout << "- NO_DATA drops indicate warmup periods or strategy logic issues\n";
  std::cout << "- THRESHOLD_TOO_TIGHT drops show when confidence thresholds are too high\n";
  std::cout << "- NOT_RTH drops occur when RTH validation is too strict\n";
  std::cout << "- This system can be integrated into existing strategies to debug signal issues\n";
}

int main() {
  run_integration_test();
  return 0;
}

```

## 📄 **FILE 89 of 92**: tools/signal_diagnostics.cpp

**File Information**:
- **Path**: `tools/signal_diagnostics.cpp`

- **Size**: 153 lines
- **Modified**: 2025-09-05 16:36:32

- **Type**: .cpp

```text
#include "../include/sentio/strategy_sma_cross.hpp"
#include "../include/sentio/signal_engine.hpp"
#include "../include/sentio/rth_calendar.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace sentio;

struct BarData {
  double open, high, low, close;
  std::int64_t ts_utc;
  bool is_rth;
};

void print_health_summary(const SignalHealth& health) {
  std::cout << "\n📊 SIGNAL HEALTH SUMMARY\n";
  std::cout << "========================\n";
  std::cout << "Emitted: " << health.emitted.load() << "\n";
  std::cout << "Dropped: " << health.dropped.load() << "\n";
  std::cout << "\nDrop Reasons:\n";
  
  const char* reason_names[] = {
    "NONE", "NO_DATA", "NOT_RTH", "HOLIDAY", "WARMUP", 
    "NAN_INPUT", "THRESHOLD_TOO_TIGHT", "COOLDOWN_ACTIVE", 
    "DEBOUNCE", "DUPLICATE_BAR_TS", "UNKNOWN"
  };
  
  for (const auto& [reason, count] : health.by_reason) {
    if (count.load() > 0) {
      int idx = static_cast<int>(reason);
      if (idx >= 0 && idx < 11) {
        std::cout << "  " << reason_names[idx] << ": " << count.load() << "\n";
      }
    }
  }
}

void run_diagnostic_test() {
  std::cout << "🔍 SIGNAL DIAGNOSTIC TOOL\n";
  std::cout << "==========================\n\n";
  
  // Test 1: Basic SMA Cross with rising data
  std::cout << "Test 1: Rising data (should generate BUY signals)\n";
  std::cout << "------------------------------------------------\n";
  
  SMACrossCfg cfg{5, 10, 0.8};
  SMACrossStrategy strat(cfg);
  SignalHealth health1;
  GateCfg gate{.require_rth=false, .cooldown_bars=0, .min_conf=0.01};
  SignalEngine engine1(&strat, gate, &health1);

  std::vector<BarData> rising_data;
  for (int i = 0; i < 20; ++i) {
    double price = 100.0 + i * 0.5;
    rising_data.push_back({price, price, price, price, 1000000 + i * 60, true});
  }

  int signals_1 = 0;
  for (size_t i = 0; i < rising_data.size(); ++i) {
    const auto& bar = rising_data[i];
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=bar.ts_utc, .is_rth=bar.is_rth};
    Bar b{bar.open, bar.high, bar.low, bar.close};
    auto out = engine1.on_bar(ctx, b, true);
    
    if (out.signal) {
      signals_1++;
      std::cout << "  Bar " << std::setw(2) << i << ": " 
                << (out.signal->type == StrategySignal::Type::BUY ? "BUY" : "SELL")
                << " (conf: " << std::fixed << std::setprecision(2) << out.signal->confidence << ")\n";
    }
  }
  
  print_health_summary(health1);
  std::cout << "\nResult: " << signals_1 << " signals emitted\n\n";

  // Test 2: RTH validation test
  std::cout << "Test 2: RTH validation (should drop non-RTH bars)\n";
  std::cout << "------------------------------------------------\n";
  
  SMACrossStrategy strat2(cfg);
  SignalHealth health2;
  GateCfg gate_rth{.require_rth=true, .cooldown_bars=0, .min_conf=0.01};
  SignalEngine engine2(&strat2, gate_rth, &health2);

  std::vector<BarData> mixed_data;
  for (int i = 0; i < 15; ++i) {
    double price = 100.0 + i * 0.3;
    // Mix RTH and non-RTH bars
    bool is_rth = (i % 3 != 0); // Every 3rd bar is non-RTH
    mixed_data.push_back({price, price, price, price, 1000000 + i * 60, is_rth});
  }

  int signals_2 = 0;
  for (size_t i = 0; i < mixed_data.size(); ++i) {
    const auto& bar = mixed_data[i];
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=bar.ts_utc, .is_rth=bar.is_rth};
    Bar b{bar.open, bar.high, bar.low, bar.close};
    auto out = engine2.on_bar(ctx, b, true);
    
    if (out.signal) {
      signals_2++;
      std::cout << "  Bar " << std::setw(2) << i << " (RTH:" << (bar.is_rth ? "Y" : "N") << "): " 
                << (out.signal->type == StrategySignal::Type::BUY ? "BUY" : "SELL") << "\n";
    } else {
      std::cout << "  Bar " << std::setw(2) << i << " (RTH:" << (bar.is_rth ? "Y" : "N") << "): DROPPED\n";
    }
  }
  
  print_health_summary(health2);
  std::cout << "\nResult: " << signals_2 << " signals emitted\n\n";

  // Test 3: Threshold test
  std::cout << "Test 3: High threshold (should drop low confidence signals)\n";
  std::cout << "--------------------------------------------------------\n";
  
  SMACrossStrategy strat3(cfg);
  SignalHealth health3;
  GateCfg gate_high{.require_rth=false, .cooldown_bars=0, .min_conf=0.9}; // Very high threshold
  SignalEngine engine3(&strat3, gate_high, &health3);

  int signals_3 = 0;
  for (size_t i = 0; i < rising_data.size(); ++i) {
    const auto& bar = rising_data[i];
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=bar.ts_utc, .is_rth=bar.is_rth};
    Bar b{bar.open, bar.high, bar.low, bar.close};
    auto out = engine3.on_bar(ctx, b, true);
    
    if (out.signal) {
      signals_3++;
      std::cout << "  Bar " << std::setw(2) << i << ": " 
                << (out.signal->type == StrategySignal::Type::BUY ? "BUY" : "SELL")
                << " (conf: " << std::fixed << std::setprecision(2) << out.signal->confidence << ")\n";
    }
  }
  
  print_health_summary(health3);
  std::cout << "\nResult: " << signals_3 << " signals emitted\n\n";

  std::cout << "🎯 DIAGNOSTIC COMPLETE\n";
  std::cout << "======================\n";
  std::cout << "If you see 0 signals in any test, check the drop reasons above.\n";
  std::cout << "Common issues:\n";
  std::cout << "  - WARMUP: Not enough data for indicators\n";
  std::cout << "  - NOT_RTH: RTH validation too strict\n";
  std::cout << "  - THRESHOLD_TOO_TIGHT: Confidence threshold too high\n";
  std::cout << "  - NAN_INPUT: Invalid data in bars\n";
}

int main() {
  run_diagnostic_test();
  return 0;
}

```

## 📄 **FILE 90 of 92**: tools/simple_strategy_diagnostics.cpp

**File Information**:
- **Path**: `tools/simple_strategy_diagnostics.cpp`

- **Size**: 110 lines
- **Modified**: 2025-09-05 17:25:53

- **Type**: .cpp

```text
#include "../include/sentio/strategy_market_making.hpp"
#include "../include/sentio/core.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace sentio;

void test_strategy_direct(const std::string& strategy_name, BaseStrategy* strategy, const std::vector<double>& closes) {
  std::cout << "\n🔍 Testing " << strategy_name << " Strategy\n";
  std::cout << "=====================================\n";
  
  // Create a vector of bars
  std::vector<Bar> bars;
  for (int i = 0; i < (int)closes.size(); ++i) {
    Bar b;
    b.ts_utc = std::to_string(1'000'000 + i * 60);
    b.ts_nyt_epoch = 1'000'000 + i * 60;
    b.open = closes[i];
    b.high = closes[i] * 1.001;  // Slight high
    b.low = closes[i] * 0.999;   // Slight low
    b.close = closes[i];
    b.volume = 1000;
    bars.push_back(b);
  }
  
  int signals_emitted = 0;
  std::cout << "Bar-by-bar analysis:\n";
  
  for (int i = 0; i < (int)bars.size(); ++i) {
    // Calculate signal for this bar
    StrategySignal signal = strategy->calculate_signal(bars, i);
    
    if (signal.type != StrategySignal::Type::HOLD) {
      signals_emitted++;
      std::cout << "  ✅ Bar " << std::setw(3) << i 
                << " (price: " << std::fixed << std::setprecision(2) << bars[i].close << "): ";
      
      switch (signal.type) {
        case StrategySignal::Type::BUY: std::cout << "BUY"; break;
        case StrategySignal::Type::SELL: std::cout << "SELL"; break;
        case StrategySignal::Type::STRONG_BUY: std::cout << "STRONG_BUY"; break;
        case StrategySignal::Type::STRONG_SELL: std::cout << "STRONG_SELL"; break;
        default: std::cout << "UNKNOWN"; break;
      }
      
      std::cout << " (conf: " << std::fixed << std::setprecision(3) << signal.confidence << ")\n";
    } else if (i % 20 == 0) {
      // Show progress every 20 bars
      std::cout << "  ⏳ Bar " << std::setw(3) << i 
                << " (price: " << std::fixed << std::setprecision(2) << bars[i].close 
                << "): No signal\n";
    }
  }
  
  std::cout << "\n📊 Summary for " << strategy_name << ":\n";
  std::cout << "  Total bars processed: " << bars.size() << "\n";
  std::cout << "  Signals emitted: " << signals_emitted << "\n";
  std::cout << "  Success rate: " << std::fixed << std::setprecision(2) 
            << (bars.size() > 0 ? (signals_emitted * 100.0 / bars.size()) : 0.0) << "%\n";
  
  // Get diagnostic info
  const SignalDiag& diag = strategy->get_diag();
  std::cout << "  Diagnostic info:\n";
  std::cout << "    Signals emitted: " << diag.emitted << "\n";
  std::cout << "    Signals dropped: " << diag.dropped << "\n";
  std::cout << "    Drop reasons:\n";
  std::cout << "      Min bars: " << diag.r_min_bars << "\n";
  std::cout << "      Session: " << diag.r_session << "\n";
  std::cout << "      NaN input: " << diag.r_nan << "\n";
  std::cout << "      Zero volume: " << diag.r_zero_vol << "\n";
  std::cout << "      Threshold: " << diag.r_threshold << "\n";
  std::cout << "      Cooldown: " << diag.r_cooldown << "\n";
  std::cout << "      Duplicate: " << diag.r_dup << "\n";
}

int main() {
  std::cout << "🔍 SIMPLE STRATEGY DIAGNOSTIC TOOL\n";
  std::cout << "==================================\n";
  std::cout << "Testing problematic strategies with direct signal calculation\n\n";
  
  // Create realistic test data with some volatility
  std::vector<double> closes;
  double base_price = 100.0;
  srand(42); // For reproducible results
  
  for (int i = 0; i < 100; ++i) {
    // Add some realistic price movement with occasional spikes
    double trend = i * 0.1;  // Slight upward trend
    double noise = (rand() % 100 - 50) * 0.01;  // Random noise
    double spike = (i % 20 == 0) ? (rand() % 100 - 50) * 0.05 : 0;  // Occasional spikes
    closes.push_back(base_price + trend + noise + spike);
  }
  
  std::cout << "📈 Test data: " << closes.size() << " bars, price range: " 
            << *std::min_element(closes.begin(), closes.end()) << " - "
            << *std::max_element(closes.begin(), closes.end()) << "\n\n";
  
  // Test Market Making Strategy
  std::cout << "Creating Market Making Strategy...\n";
  MarketMakingStrategy mm_strategy;
  test_strategy_direct("Market Making", &mm_strategy, closes);
  
  std::cout << "\n🎯 DIAGNOSTIC COMPLETE\n";
  std::cout << "======================\n";
  std::cout << "Check the bar-by-bar analysis above to see exactly when\n";
  std::cout << "each strategy should have generated signals but didn't.\n";
  
  return 0;
}

```

## 📄 **FILE 91 of 92**: tools/strategy_diagnostics.cpp

**File Information**:
- **Path**: `tools/strategy_diagnostics.cpp`

- **Size**: 168 lines
- **Modified**: 2025-09-05 17:26:20

- **Type**: .cpp

```text
#include "../include/sentio/strategy_market_making.hpp"
#include "../include/sentio/signal_pipeline.hpp"
#include "../include/sentio/feature_health.hpp"
#include "../include/sentio/core.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <unordered_map>

using namespace sentio;

struct PB {
  void upsert_latest(const std::string& i, const Bar& b) { latest[i]=b; }
  const Bar* get_latest(const std::string& i) const { auto it=latest.find(i); return it==latest.end()?nullptr:&it->second; }
  bool has_instrument(const std::string& i) const { return latest.count(i)>0; }
  std::size_t size() const { return latest.size(); }
  std::unordered_map<std::string,Bar> latest;
};

const char* reason_to_string(TraceReason r) {
  switch (r) {
    case TraceReason::OK: return "OK";
    case TraceReason::NO_STRATEGY_OUTPUT: return "NO_STRATEGY_OUTPUT";
    case TraceReason::NOT_RTH: return "NOT_RTH";
    case TraceReason::HOLIDAY: return "HOLIDAY";
    case TraceReason::WARMUP: return "WARMUP";
    case TraceReason::NAN_INPUT: return "NAN_INPUT";
    case TraceReason::THRESHOLD_TOO_TIGHT: return "THRESHOLD_TOO_TIGHT";
    case TraceReason::COOLDOWN_ACTIVE: return "COOLDOWN_ACTIVE";
    case TraceReason::DUPLICATE_BAR_TS: return "DUPLICATE_BAR_TS";
    case TraceReason::EMPTY_PRICEBOOK: return "EMPTY_PRICEBOOK";
    case TraceReason::NO_PRICE_FOR_INSTRUMENT: return "NO_PRICE_FOR_INSTRUMENT";
    case TraceReason::ROUTER_REJECTED: return "ROUTER_REJECTED";
    case TraceReason::ORDER_QTY_LT_MIN: return "ORDER_QTY_LT_MIN";
    case TraceReason::UNKNOWN: return "UNKNOWN";
    default: return "UNKNOWN";
  }
}

void print_strategy_summary(const std::string& strategy_name, const SignalTrace& trace) {
  std::cout << "\n📊 " << strategy_name << " DIAGNOSTIC RESULTS\n";
  std::cout << "==========================================\n";
  std::cout << "Total bars processed: " << trace.rows().size() << "\n";
  std::cout << "OK signals: " << trace.count(TraceReason::OK) << "\n";
  std::cout << "No strategy output: " << trace.count(TraceReason::NO_STRATEGY_OUTPUT) << "\n";
  std::cout << "Threshold too tight: " << trace.count(TraceReason::THRESHOLD_TOO_TIGHT) << "\n";
  std::cout << "WARMUP: " << trace.count(TraceReason::WARMUP) << "\n";
  std::cout << "NOT_RTH: " << trace.count(TraceReason::NOT_RTH) << "\n";
  std::cout << "NAN_INPUT: " << trace.count(TraceReason::NAN_INPUT) << "\n";
  std::cout << "COOLDOWN_ACTIVE: " << trace.count(TraceReason::COOLDOWN_ACTIVE) << "\n";
  std::cout << "DUPLICATE_BAR_TS: " << trace.count(TraceReason::DUPLICATE_BAR_TS) << "\n";
  std::cout << "EMPTY_PRICEBOOK: " << trace.count(TraceReason::EMPTY_PRICEBOOK) << "\n";
  std::cout << "NO_PRICE_FOR_INSTRUMENT: " << trace.count(TraceReason::NO_PRICE_FOR_INSTRUMENT) << "\n";
  std::cout << "ROUTER_REJECTED: " << trace.count(TraceReason::ROUTER_REJECTED) << "\n";
  std::cout << "ORDER_QTY_LT_MIN: " << trace.count(TraceReason::ORDER_QTY_LT_MIN) << "\n";
  std::cout << "UNKNOWN: " << trace.count(TraceReason::UNKNOWN) << "\n";
  
  double success_rate = trace.rows().size() > 0 ? 
    (trace.count(TraceReason::OK) * 100.0 / trace.rows().size()) : 0.0;
  std::cout << "Success rate: " << std::fixed << std::setprecision(2) << success_rate << "%\n";
}

void print_detailed_trace(const SignalTrace& trace, int limit = 20) {
  std::cout << "\n📋 DETAILED TRACE (Last " << limit << " bars)\n";
  std::cout << "============================================\n";
  std::cout << "ts_utc,instrument,close,is_rth,warmed,inputs_finite,confidence,conf_after_gate,reason\n";
  
  int start = std::max(0, (int)trace.rows().size() - limit);
  for (int i = start; i < (int)trace.rows().size(); ++i) {
    const auto& row = trace.rows()[i];
    std::cout << row.ts_utc << ","
              << row.instrument << ","
              << std::fixed << std::setprecision(2) << row.close << ","
              << (row.is_rth ? "true" : "false") << ","
              << (row.warmed ? "true" : "false") << ","
              << (row.inputs_finite ? "true" : "false") << ","
              << std::fixed << std::setprecision(3) << row.confidence << ","
              << row.conf_after_gate << ","
              << reason_to_string(row.reason) << "\n";
  }
}

void test_strategy(const std::string& strategy_name, BaseStrategy* strategy, const std::vector<double>& closes) {
  std::cout << "\n🔍 Testing " << strategy_name << " Strategy\n";
  std::cout << "=====================================\n";
  
  PB book;
  SignalTrace trace;
  PipelineCfg pcfg;
  pcfg.gate = GateCfg{true, 0, 0.01};  // Very permissive gate
  pcfg.min_order_shares = 1.0;

  SignalPipeline pipe(strategy, pcfg, &book, &trace);
  struct SimpleAccount { double equity=100000; double cash=100000; };
  SimpleAccount acct;

  int signals_emitted = 0;
  for (int i = 0; i < (int)closes.size(); ++i) {
    // Create Bar with proper structure
    Bar b;
    b.ts_utc = std::to_string(1'000'000 + i * 60);
    b.ts_nyt_epoch = 1'000'000 + i * 60;
    b.open = closes[i];
    b.high = closes[i] * 1.001;  // Slight high
    b.low = closes[i] * 0.999;   // Slight low
    b.close = closes[i];
    b.volume = 1000;
    
    // Update price book
    book.upsert_latest("QQQ", b);
    
    // Create TQQQ bar
    Bar tqqq_bar = b;
    tqqq_bar.open *= 3;
    tqqq_bar.high *= 3;
    tqqq_bar.low *= 3;
    tqqq_bar.close *= 3;
    book.upsert_latest("TQQQ", tqqq_bar);
    
    // Test strategy directly instead of through pipeline
    strategy->on_bar(b);
    auto signal = strategy->get_latest_signal();
    
    if (signal.has_value()) {
      signals_emitted++;
      std::cout << "  ✅ Signal at bar " << i << ": " 
                << (signal->type == StrategySignal::Type::BUY ? "BUY" : 
                    signal->type == StrategySignal::Type::SELL ? "SELL" : "HOLD")
                << " (conf: " << std::fixed << std::setprecision(3) << signal->confidence << ")\n";
    }
  }
  
  std::cout << "\nSignals emitted: " << signals_emitted << "\n";
  print_strategy_summary(strategy_name, trace);
  print_detailed_trace(trace, 15);
}

int main() {
  std::cout << "🔍 STRATEGY DIAGNOSTIC TOOL\n";
  std::cout << "===========================\n";
  std::cout << "Testing problematic strategies with detailed tracing\n\n";
  
  // Create realistic test data with some volatility
  std::vector<double> closes;
  double base_price = 100.0;
  for (int i = 0; i < 100; ++i) {
    // Add some realistic price movement with occasional spikes
    double trend = i * 0.1;  // Slight upward trend
    double noise = (rand() % 100 - 50) * 0.01;  // Random noise
    double spike = (i % 20 == 0) ? (rand() % 100 - 50) * 0.05 : 0;  // Occasional spikes
    closes.push_back(base_price + trend + noise + spike);
  }
  
  std::cout << "📈 Test data: " << closes.size() << " bars, price range: " 
            << *std::min_element(closes.begin(), closes.end()) << " - "
            << *std::max_element(closes.begin(), closes.end()) << "\n\n";
  
  // Test Market Making Strategy
  MarketMakingStrategy mm_strategy;
  test_strategy("Market Making", &mm_strategy, closes);
  
  std::cout << "\n🎯 DIAGNOSTIC COMPLETE\n";
  std::cout << "======================\n";
  std::cout << "Check the detailed traces above to identify why each strategy\n";
  std::cout << "is not generating signals. Look for patterns in the reason codes.\n";
  
  return 0;
}

```

## 📄 **FILE 92 of 92**: tools/trace_analyzer.cpp

**File Information**:
- **Path**: `tools/trace_analyzer.cpp`

- **Size**: 133 lines
- **Modified**: 2025-09-05 17:06:27

- **Type**: .cpp

```text
#include "../include/sentio/strategy_sma_cross.hpp"
#include "../include/sentio/signal_pipeline.hpp"
#include "../include/sentio/feature_health.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace sentio;

struct PB {
  void upsert_latest(const std::string& i, const Bar& b) { latest[i]=b; }
  const Bar* get_latest(const std::string& i) const { auto it=latest.find(i); return it==latest.end()?nullptr:&it->second; }
  bool has_instrument(const std::string& i) const { return latest.count(i)>0; }
  std::size_t size() const { return latest.size(); }
  std::unordered_map<std::string,Bar> latest;
};

const char* reason_to_string(TraceReason r) {
  switch (r) {
    case TraceReason::OK: return "OK";
    case TraceReason::NO_STRATEGY_OUTPUT: return "NO_STRATEGY_OUTPUT";
    case TraceReason::NOT_RTH: return "NOT_RTH";
    case TraceReason::HOLIDAY: return "HOLIDAY";
    case TraceReason::WARMUP: return "WARMUP";
    case TraceReason::NAN_INPUT: return "NAN_INPUT";
    case TraceReason::THRESHOLD_TOO_TIGHT: return "THRESHOLD_TOO_TIGHT";
    case TraceReason::COOLDOWN_ACTIVE: return "COOLDOWN_ACTIVE";
    case TraceReason::DUPLICATE_BAR_TS: return "DUPLICATE_BAR_TS";
    case TraceReason::EMPTY_PRICEBOOK: return "EMPTY_PRICEBOOK";
    case TraceReason::NO_PRICE_FOR_INSTRUMENT: return "NO_PRICE_FOR_INSTRUMENT";
    case TraceReason::ROUTER_REJECTED: return "ROUTER_REJECTED";
    case TraceReason::ORDER_QTY_LT_MIN: return "ORDER_QTY_LT_MIN";
    case TraceReason::UNKNOWN: return "UNKNOWN";
    default: return "UNKNOWN";
  }
}

void print_trace_summary(const SignalTrace& trace) {
  std::cout << "\n📊 TRACE SUMMARY\n";
  std::cout << "================\n";
  std::cout << "Total bars processed: " << trace.rows().size() << "\n";
  std::cout << "OK signals: " << trace.count(TraceReason::OK) << "\n";
  std::cout << "No strategy output: " << trace.count(TraceReason::NO_STRATEGY_OUTPUT) << "\n";
  std::cout << "Threshold too tight: " << trace.count(TraceReason::THRESHOLD_TOO_TIGHT) << "\n";
  std::cout << "WARMUP: " << trace.count(TraceReason::WARMUP) << "\n";
  std::cout << "NOT_RTH: " << trace.count(TraceReason::NOT_RTH) << "\n";
  std::cout << "NAN_INPUT: " << trace.count(TraceReason::NAN_INPUT) << "\n";
  std::cout << "COOLDOWN_ACTIVE: " << trace.count(TraceReason::COOLDOWN_ACTIVE) << "\n";
  std::cout << "DUPLICATE_BAR_TS: " << trace.count(TraceReason::DUPLICATE_BAR_TS) << "\n";
  std::cout << "EMPTY_PRICEBOOK: " << trace.count(TraceReason::EMPTY_PRICEBOOK) << "\n";
  std::cout << "NO_PRICE_FOR_INSTRUMENT: " << trace.count(TraceReason::NO_PRICE_FOR_INSTRUMENT) << "\n";
  std::cout << "ROUTER_REJECTED: " << trace.count(TraceReason::ROUTER_REJECTED) << "\n";
  std::cout << "ORDER_QTY_LT_MIN: " << trace.count(TraceReason::ORDER_QTY_LT_MIN) << "\n";
  std::cout << "UNKNOWN: " << trace.count(TraceReason::UNKNOWN) << "\n\n";
}

void print_csv_header() {
  std::cout << "ts_utc,instrument,routed,close,is_rth,warmed,inputs_finite,confidence,conf_after_gate,target_weight,last_px,order_qty,reason,note\n";
}

void print_trace_csv(const SignalTrace& trace, int limit = 50) {
  std::cout << "\n📋 LAST " << limit << " TRACE ROWS (CSV)\n";
  std::cout << "=====================================\n";
  print_csv_header();
  
  int start = std::max(0, (int)trace.rows().size() - limit);
  for (int i = start; i < (int)trace.rows().size(); ++i) {
    const auto& row = trace.rows()[i];
    std::cout << row.ts_utc << ","
              << row.instrument << ","
              << row.routed << ","
              << std::fixed << std::setprecision(2) << row.close << ","
              << (row.is_rth ? "true" : "false") << ","
              << (row.warmed ? "true" : "false") << ","
              << (row.inputs_finite ? "true" : "false") << ","
              << std::fixed << std::setprecision(3) << row.confidence << ","
              << row.conf_after_gate << ","
              << row.target_weight << ","
              << row.last_px << ","
              << row.order_qty << ","
              << reason_to_string(row.reason) << ","
              << row.note << "\n";
  }
}

void run_diagnostic_test() {
  std::cout << "🔍 SIGNAL PIPELINE DIAGNOSTIC TOOL\n";
  std::cout << "==================================\n\n";
  
  PB book;
  SMACrossCfg scfg{5, 10, 0.8};
  SMACrossStrategy strat(scfg);

  SignalTrace trace;
  PipelineCfg pcfg;
  pcfg.gate = GateCfg{true, 0, 0.01};
  pcfg.min_order_shares = 1.0;

  SignalPipeline pipe(&strat, pcfg, &book, &trace);
  // Simple account struct to avoid include conflicts
  struct SimpleAccount { double equity=100000; double cash=100000; };
  SimpleAccount acct;

  // Test with rising data to trigger signals
  std::vector<double> closes;
  for (int i=0;i<50;++i) closes.push_back(100+i*0.3);

  int emits=0;
  for (int i=0;i<(int)closes.size();++i) {
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=1'000'000+i*60, .is_rth=true};
    Bar b{closes[i],closes[i],closes[i],closes[i]};
    
    // Update price book for both QQQ and TQQQ
    book.upsert_latest("QQQ", b);
    book.upsert_latest("TQQQ", Bar{b.open*3,b.high*3,b.low*3,b.close*3});
    
    auto out = pipe.on_bar(ctx,b,&acct);
    if (out.signal) ++emits;
  }

  print_trace_summary(trace);
  print_trace_csv(trace, 20);
  
  std::cout << "\n🎯 DIAGNOSTIC COMPLETE\n";
  std::cout << "======================\n";
  std::cout << "Orders emitted: " << emits << "\n";
  std::cout << "Success rate: " << (trace.count(TraceReason::OK) * 100.0 / trace.rows().size()) << "%\n";
}

int main() {
  run_diagnostic_test();
  return 0;
}

```

