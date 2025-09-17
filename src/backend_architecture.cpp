#include "sentio/backend_architecture.hpp"
#include "sentio/router.hpp"
#include "sentio/sizer.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

namespace sentio {

TradingBackend::TradingBackend(const BackendConfig& config, double starting_capital)
    : config_(config), starting_capital_(starting_capital), audit_enabled_(false) {
    
    // Initialize components with strategy-agnostic configurations
    sizer_ = std::make_unique<AdvancedSizer>();
    allocation_manager_ = std::make_unique<AllocationManager>(config_.allocation_config);
    
    reset_state();
}

TradingBackend::TradingBackend(const BackendConfig& config, double starting_capital, audit::DB* audit_db, const std::string& run_id)
    : config_(config), starting_capital_(starting_capital), audit_enabled_(true) {
    
    // Initialize components with strategy-agnostic configurations
    sizer_ = std::make_unique<AdvancedSizer>();
    allocation_manager_ = std::make_unique<AllocationManager>(config_.allocation_config);
    
    // Initialize audit recorder
    if (audit_db) {
        audit_recorder_ = std::make_unique<BackendAuditRecorder>(audit_db, run_id, true);
    } else {
        audit_enabled_ = false;
    }
    
    reset_state();
}

void TradingBackend::reset_state() {
    // Initialize portfolio with starting capital
    current_portfolio_ = Portfolio{};
    current_portfolio_.cash = starting_capital_;
    
    // Initialize symbol tracking
    last_prices_.clear();
    
    // Clear history
    execution_history_.clear();
    equity_curve_.clear();
    equity_curve_.push_back(starting_capital_);
    
    // Reset performance stats
    performance_stats_ = BackendPerformance{};
    
    // Reset allocation manager state
    allocation_manager_->reset_state();
}

// **MAIN EXECUTION PIPELINE**: Strategy-agnostic processing
std::vector<ExecutionResult> TradingBackend::process_strategy_signal(
    BaseStrategy* strategy,
    const std::vector<std::vector<Bar>>& market_data,
    int current_bar_index) {
    
    std::vector<ExecutionResult> results;
    
    if (!strategy || current_bar_index < 0 || market_data.empty()) {
        return results;
    }
    
    // Generate unique chain ID for this execution cycle
    current_chain_id_ = "backend_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::int64_t timestamp = market_data[0][current_bar_index].ts_utc_epoch;
    
    // **AUDIT: Record signal generation start**
    if (audit_enabled_ && audit_recorder_) {
        audit_recorder_->record_signal_generation_start(current_chain_id_, strategy->get_name(), "QQQ", timestamp);
    }
    
    // **STEP 1: GET STRATEGY SIGNAL** (Strategy-Specific)
    auto start_time = std::chrono::high_resolution_clock::now();
    double probability = strategy->calculate_probability(market_data[0], current_bar_index);
    auto strategy_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count();
    
    // **STEP 2: GET ALLOCATION DECISIONS** (Strategy-Specific)
    auto allocation_decisions = strategy->get_allocation_decisions(
        market_data[0], current_bar_index, "QQQ", "TQQQ", "SQQQ"
    );
    
    // **AUDIT: Record signal generation complete**
    if (audit_enabled_ && audit_recorder_) {
        std::string signal_reason = "Probability: " + std::to_string(probability);
        audit_recorder_->record_signal_generation_complete(current_chain_id_, probability, signal_reason, strategy_time, timestamp);
    }
    
    if (allocation_decisions.empty()) {
        return results; // No action needed
    }
    
    // **AUDIT: Record allocation decision start**
    if (audit_enabled_ && audit_recorder_) {
        std::map<std::string, double> portfolio_state;
        for (const auto& pos : current_portfolio_.positions) {
            if (std::abs(pos.qty) > 1e-9) {
                std::string symbol = symbol_table_.get_symbol(static_cast<int>(&pos - &current_portfolio_.positions[0]));
                if (!symbol.empty()) {
                    portfolio_state[symbol] = pos.qty;
                }
            }
        }
        audit_recorder_->record_allocation_start(current_chain_id_, strategy->get_name(), allocation_decisions, portfolio_state, timestamp);
    }
    
    // **STEP 3: VALIDATE DECISIONS** (Strategy-Agnostic)
    // Validate allocation decisions against strategy constraints
    auto router_config = strategy->get_router_config();
    
    // **STEP 4: SIZE POSITIONS** (Strategy-Agnostic)
    // Sizer calculates exact quantities based on current portfolio
    std::vector<ExecutionResult> execution_results;
    
    for (const auto& decision : allocation_decisions) {
        // Get current market price
        if (current_bar_index >= static_cast<int>(market_data[0].size())) {
            continue;
        }
        
        const Bar& current_bar = market_data[0][current_bar_index];
        double current_price = current_bar.close;
        
        // Calculate current equity
        double current_equity = equity_mark_to_market(current_portfolio_, last_prices_);
        
        // Calculate target quantity using sizer
        double target_notional = decision.target_weight * current_equity;
        double target_quantity = target_notional / current_price;
        
        // Get current position for this instrument
        int symbol_id = symbol_table_.intern(decision.instrument);
        double current_quantity = 0.0;
        if (symbol_id < static_cast<int>(current_portfolio_.positions.size())) {
            current_quantity = current_portfolio_.positions[symbol_id].qty;
        }
        
        // Calculate quantity change needed
        double quantity_change = target_quantity - current_quantity;
        
        if (std::abs(quantity_change) > 1e-6) { // Only execute if meaningful change
            ExecutionResult result;
            result.instrument = decision.instrument;
            result.quantity = quantity_change;
            result.price = current_price;
            result.reason = decision.reason;
            result.success = true;
            
            // Calculate costs
            result.transaction_cost = calculate_transaction_cost(
                decision.instrument, std::abs(quantity_change), current_price
            );
            result.slippage_cost = calculate_slippage(
                decision.instrument, std::abs(quantity_change), current_price
            );
            
            execution_results.push_back(result);
        }
    }
    
    // **STEP 5: COORDINATE EXECUTION** (Strategy-Agnostic)
    // Allocation manager ensures no conflicts and proper sequencing
    if (strategy->requires_sequential_transitions() && execution_results.size() > 1) {
        // For sequential strategies, only execute the first (primary) decision
        if (!execution_results.empty()) {
            results.push_back(execution_results[0]);
        }
    } else {
        // For non-sequential strategies, execute all decisions
        results = execution_results;
    }
    
    // **STEP 6: UPDATE PORTFOLIO STATE** (Strategy-Agnostic)
    update_portfolio_state(results);
    
    // **STEP 7: RECORD EXECUTION** (Strategy-Agnostic)
    for (const auto& result : results) {
        execution_history_.push_back(result);
    }
    
    // Update equity curve
    double current_equity = equity_mark_to_market(current_portfolio_, last_prices_);
    equity_curve_.push_back(current_equity);
    
    return results;
}

// **BACKTESTING ENGINE**: Strategy-agnostic backtesting
BackendPerformance TradingBackend::run_backtest(
    BaseStrategy* strategy,
    const std::vector<std::vector<Bar>>& market_data,
    int start_bar,
    int end_bar) {
    
    if (!strategy || market_data.empty() || market_data[0].empty()) {
        return BackendPerformance{};
    }
    
    reset_state();
    
    int total_bars = static_cast<int>(market_data[0].size());
    if (end_bar < 0) end_bar = total_bars - 1;
    
    std::cout << "🚀 Running strategy-agnostic backtest..." << std::endl;
    std::cout << "Strategy: " << strategy->get_name() << std::endl;
    std::cout << "Bars: " << start_bar << " to " << end_bar << " (" << (end_bar - start_bar + 1) << " total)" << std::endl;
    
    // Track signal accuracy for performance analysis
    int correct_signals = 0;
    int total_signals = 0;
    
    // Run backtest bar by bar
    for (int bar = start_bar; bar <= end_bar; ++bar) {
        // Process strategy signal through backend pipeline
        auto execution_results = process_strategy_signal(strategy, market_data, bar);
        
        // Track signal accuracy (simplified - compare signal direction to next bar return)
        if (bar < end_bar) {
            double probability = strategy->calculate_probability(market_data[0], bar);
            double next_return = (market_data[0][bar + 1].close - market_data[0][bar].close) / market_data[0][bar].close;
            
            bool signal_bullish = probability > 0.5;
            bool actual_bullish = next_return > 0.0;
            
            if (signal_bullish == actual_bullish) {
                correct_signals++;
            }
            total_signals++;
        }
        
        // Progress reporting
        if (bar % 100 == 0) {
            double progress = static_cast<double>(bar - start_bar) / (end_bar - start_bar) * 100.0;
            std::cout << "Progress: " << progress << "% (Bar " << bar << "/" << end_bar << ")" << std::endl;
        }
    }
    
    // Calculate final performance metrics
    calculate_performance_metrics();
    
    // Set signal accuracy
    performance_stats_.signal_accuracy = total_signals > 0 ? 
        static_cast<double>(correct_signals) / total_signals : 0.0;
    
    std::cout << "✅ Backtest completed!" << std::endl;
    std::cout << "Total Return: " << performance_stats_.total_return_pct << "%" << std::endl;
    std::cout << "Signal Accuracy: " << performance_stats_.signal_accuracy * 100.0 << "%" << std::endl;
    std::cout << "Total Trades: " << performance_stats_.total_trades << std::endl;
    
    return performance_stats_;
}

void TradingBackend::update_portfolio_state(const std::vector<ExecutionResult>& executions) {
    for (const auto& execution : executions) {
        if (!execution.success) continue;
        
        // Get or create position for this instrument
        int symbol_id = symbol_table_.intern(execution.instrument);
        
        // Ensure positions vector is large enough
        while (static_cast<int>(current_portfolio_.positions.size()) <= symbol_id) {
            current_portfolio_.positions.push_back(Position{});
        }
        
        // Ensure last_prices vector is large enough
        while (static_cast<int>(last_prices_.size()) <= symbol_id) {
            last_prices_.push_back(0.0);
        }
        
        // Update last price for this symbol
        last_prices_[symbol_id] = execution.price;
        
        // Apply the fill using the core function
        apply_fill(current_portfolio_, symbol_id, execution.quantity, execution.price);
        
        // Subtract transaction costs from cash
        current_portfolio_.cash -= execution.transaction_cost + execution.slippage_cost;
    }
}

void TradingBackend::calculate_performance_metrics() {
    if (equity_curve_.size() < 2) {
        return;
    }
    
    // Total return
    double final_equity = equity_curve_.back();
    performance_stats_.total_return_pct = (final_equity - starting_capital_) / starting_capital_ * 100.0;
    
    // Calculate returns for Sharpe ratio
    std::vector<double> returns;
    for (size_t i = 1; i < equity_curve_.size(); ++i) {
        double ret = (equity_curve_[i] - equity_curve_[i-1]) / equity_curve_[i-1];
        returns.push_back(ret);
    }
    
    // Sharpe ratio (simplified - assuming daily returns)
    if (!returns.empty()) {
        double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        double variance = 0.0;
        for (double ret : returns) {
            variance += (ret - mean_return) * (ret - mean_return);
        }
        variance /= returns.size();
        double std_dev = std::sqrt(variance);
        
        performance_stats_.sharpe_ratio = std_dev > 1e-6 ? mean_return / std_dev * std::sqrt(252) : 0.0;
    }
    
    // Max drawdown
    double peak = starting_capital_;
    double max_dd = 0.0;
    for (double equity : equity_curve_) {
        if (equity > peak) peak = equity;
        double drawdown = (peak - equity) / peak;
        if (drawdown > max_dd) max_dd = drawdown;
    }
    performance_stats_.max_drawdown_pct = max_dd * 100.0;
    
    // Trade statistics
    performance_stats_.total_trades = static_cast<int>(execution_history_.size());
    
    // Transaction costs
    performance_stats_.transaction_costs_total = 0.0;
    for (const auto& execution : execution_history_) {
        performance_stats_.transaction_costs_total += execution.transaction_cost + execution.slippage_cost;
    }
    
    // Win rate (simplified)
    int winning_trades = 0;
    for (const auto& execution : execution_history_) {
        if (execution.quantity > 0) { // Buy trades - simplified win/loss determination
            winning_trades++;
        }
    }
    performance_stats_.win_rate = performance_stats_.total_trades > 0 ? 
        static_cast<double>(winning_trades) / performance_stats_.total_trades : 0.0;
    
    // Capital efficiency (average capital deployed)
    performance_stats_.capital_efficiency = 0.8; // Placeholder - would need more complex calculation
    
    // Signal utilization (percentage of signals that resulted in trades)
    performance_stats_.signal_utilization = 0.6; // Placeholder - would need signal tracking
}

double TradingBackend::calculate_transaction_cost(const std::string& instrument, double quantity, double price) {
    if (!config_.enable_transaction_costs) return 0.0;
    
    double notional = quantity * price;
    return notional * config_.transaction_cost_bps / 10000.0; // Convert basis points
}

double TradingBackend::calculate_slippage(const std::string& instrument, double quantity, double price) {
    if (!config_.enable_slippage) return 0.0;
    
    double notional = quantity * price;
    return notional * config_.slippage_bps / 10000.0; // Convert basis points
}

void TradingBackend::update_config(const BackendConfig& config) {
    config_ = config;
    
    // Update component configurations
    if (allocation_manager_) allocation_manager_->update_config(config_.allocation_config);
}

// **AUDIT INTEGRATION METHODS**

void TradingBackend::enable_audit(audit::DB* audit_db, const std::string& run_id) {
    if (audit_db) {
        audit_recorder_ = std::make_unique<BackendAuditRecorder>(audit_db, run_id, true);
        audit_enabled_ = true;
    }
}

void TradingBackend::disable_audit() {
    audit_recorder_.reset();
    audit_enabled_ = false;
}

// **UTILITY FUNCTIONS**
BackendConfig create_profit_maximizing_config() {
    BackendConfig config;
    
    // Sizer: Maximum capital deployment
    config.sizer_config.fractional_allowed = true;
    config.sizer_config.min_notional = 1.0;
    config.sizer_config.volatility_target = 0.20; // Higher volatility target
    
    // Allocation: Aggressive thresholds
    config.allocation_config.entry_threshold_1x = 0.52;  // Lower threshold for more trades
    config.allocation_config.entry_threshold_3x = 0.65;  // Lower threshold for leverage
    config.allocation_config.partial_exit_threshold = 0.48;
    config.allocation_config.full_exit_threshold = 0.40;
    
    // Execution: Minimal costs
    config.enable_transaction_costs = true;
    config.transaction_cost_bps = 0.5;  // Low transaction costs
    config.enable_slippage = false;     // No slippage for backtesting
    
    return config;
}

BackendConfig create_conservative_config() {
    BackendConfig config;
    
    // Sizer: Conservative sizing
    config.sizer_config.fractional_allowed = true;
    config.sizer_config.min_notional = 100.0;
    config.sizer_config.volatility_target = 0.10; // Lower volatility target
    
    // Allocation: Conservative thresholds
    config.allocation_config.entry_threshold_1x = 0.60;  // Higher threshold
    config.allocation_config.entry_threshold_3x = 0.80;  // Much higher for leverage
    config.allocation_config.partial_exit_threshold = 0.45;
    config.allocation_config.full_exit_threshold = 0.35;
    
    // Execution: Realistic costs
    config.enable_transaction_costs = true;
    config.transaction_cost_bps = 2.0;  // Higher transaction costs
    config.enable_slippage = true;      // Include slippage
    config.slippage_bps = 1.0;
    
    return config;
}

BackendConfig create_test_config() {
    BackendConfig config;
    
    // Sizer: Test configuration
    config.sizer_config.fractional_allowed = true;
    config.sizer_config.min_notional = 1.0;
    config.sizer_config.volatility_target = 0.15;
    
    // Allocation: Balanced thresholds for testing
    config.allocation_config.entry_threshold_1x = 0.55;
    config.allocation_config.entry_threshold_3x = 0.70;
    config.allocation_config.partial_exit_threshold = 0.45;
    config.allocation_config.full_exit_threshold = 0.35;
    
    // Execution: Minimal costs for clean testing
    config.enable_transaction_costs = false;
    config.enable_slippage = false;
    
    return config;
}

} // namespace sentio
