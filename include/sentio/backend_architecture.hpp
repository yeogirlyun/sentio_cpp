#pragma once

#include "sentio/base_strategy.hpp"
#include "sentio/router.hpp"
#include "sentio/sizer.hpp"
#include "sentio/allocation_manager.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/backend_audit_events.hpp"
#include "sentio/core.hpp"
#include <vector>
#include <memory>
#include <string>

namespace sentio {

// **STRATEGY-AGNOSTIC BACKEND ARCHITECTURE**
// Clean separation: Strategy-specific logic in BaseStrategy, everything else generic

struct BackendConfig {
    // Sizer configuration  
    SizerCfg sizer_config;
    
    // Allocation manager configuration
    AllocationConfig allocation_config;
    
    // Execution settings
    bool enable_transaction_costs = true;
    double transaction_cost_bps = 1.0;  // 1 basis point per trade
    bool enable_slippage = false;
    double slippage_bps = 0.5;          // 0.5 basis points slippage
    
    // Risk management
    double max_drawdown_pct = 0.20;     // 20% max drawdown
    bool enable_position_limits = true;
    double max_position_pct = 1.0;      // 100% max position (profit maximization)
};

struct ExecutionResult {
    std::string instrument;
    double quantity;
    double price;
    double transaction_cost;
    double slippage_cost;
    std::string reason;
    bool success;
};

struct BackendPerformance {
    double total_return_pct;
    double sharpe_ratio;
    double max_drawdown_pct;
    double win_rate;
    int total_trades;
    double avg_trade_return_pct;
    double transaction_costs_total;
    double profit_factor;           // Gross profit / Gross loss
    
    // Strategy-specific metrics
    double signal_accuracy;         // Percentage of correct signals
    double signal_utilization;      // Percentage of signals that resulted in trades
    double capital_efficiency;      // Average capital deployed
};

// **STRATEGY-AGNOSTIC EXECUTION BACKEND**
class TradingBackend {
private:
    BackendConfig config_;
    std::unique_ptr<AdvancedSizer> sizer_;
    std::unique_ptr<AllocationManager> allocation_manager_;
    std::unique_ptr<BackendAuditRecorder> audit_recorder_;
    
    // State tracking
    Portfolio current_portfolio_;
    SymbolTable symbol_table_;
    std::vector<ExecutionResult> execution_history_;
    std::vector<double> equity_curve_;
    std::vector<double> last_prices_; // Track last prices for equity calculation
    double starting_capital_;
    
    // Audit and performance tracking
    bool audit_enabled_;
    std::string current_chain_id_;
    
    // Performance tracking
    BackendPerformance performance_stats_;
    
    // Internal methods
    std::vector<ExecutionResult> execute_allocation_decisions(
        const std::vector<BaseStrategy::AllocationDecision>& decisions,
        const std::vector<std::vector<Bar>>& market_data,
        int current_bar_index
    );
    
    void update_portfolio_state(const std::vector<ExecutionResult>& executions);
    void calculate_performance_metrics();
    double calculate_transaction_cost(const std::string& instrument, double quantity, double price);
    double calculate_slippage(const std::string& instrument, double quantity, double price);
    
public:
    TradingBackend(const BackendConfig& config, double starting_capital = 100000.0);
    TradingBackend(const BackendConfig& config, double starting_capital, audit::DB* audit_db, const std::string& run_id);
    
    // **MAIN EXECUTION INTERFACE**: Strategy-agnostic execution pipeline
    std::vector<ExecutionResult> process_strategy_signal(
        BaseStrategy* strategy,
        const std::vector<std::vector<Bar>>& market_data,
        int current_bar_index
    );
    
    // **BACKTESTING INTERFACE**: Run full backtest with strategy
    BackendPerformance run_backtest(
        BaseStrategy* strategy,
        const std::vector<std::vector<Bar>>& market_data,
        int start_bar = 0,
        int end_bar = -1
    );
    
    // **STATE ACCESS**
    const Portfolio& get_current_portfolio() const { return current_portfolio_; }
    const std::vector<ExecutionResult>& get_execution_history() const { return execution_history_; }
    
    // **AUDIT INTEGRATION**
    void enable_audit(audit::DB* audit_db, const std::string& run_id);
    void disable_audit();
    bool is_audit_enabled() const { return audit_enabled_; }
    BackendAuditRecorder* get_audit_recorder() const { return audit_recorder_.get(); }
    const std::vector<double>& get_equity_curve() const { return equity_curve_; }
    const BackendPerformance& get_performance() const { return performance_stats_; }
    
    // **CONFIGURATION**
    void update_config(const BackendConfig& config);
    void reset_state();
};

// **UTILITY FUNCTIONS**
BackendConfig create_profit_maximizing_config();
BackendConfig create_conservative_config();
BackendConfig create_test_config();

} // namespace sentio
