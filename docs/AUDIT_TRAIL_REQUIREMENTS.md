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
