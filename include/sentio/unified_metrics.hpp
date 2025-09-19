#pragma once

#include "sentio/metrics.hpp"
#include "sentio/cost_model.hpp"
#include "sentio/runner.hpp" // For BacktestOutput
#include <vector>
#include <string>

// Forward declaration to avoid circular dependency
namespace audit {
    struct Event;
}

namespace sentio {

// NEW: This is the canonical, final report struct. All systems will use this.
struct UnifiedMetricsReport {
    double final_equity;
    double total_return;
    double sharpe_ratio;
    double max_drawdown;
    double monthly_projected_return; // Canonical MPR
    double avg_daily_trades;
    int total_fills;
};

/**
 * Unified Metrics Calculator - Single Source of Truth for Performance Metrics
 * 
 * This class provides consistent performance calculation across all systems:
 * - TPA (Temporal Performance Analysis)
 * - Audit System
 * - Live Trading Monitoring
 * 
 * Uses the statistically robust compute_metrics_day_aware methodology
 * with proper compound interest calculations and Alpaca fee modeling.
 */
class UnifiedMetricsCalculator {
public:
    /**
     * NEW: Primary method to generate a unified report from raw backtest data.
     * This is the single source of truth for all metric calculations.
     * 
     * @param output Raw backtest output containing equity curve and trade statistics
     * @return UnifiedMetricsReport with all canonical performance metrics
     */
    static UnifiedMetricsReport calculate_metrics(const BacktestOutput& output);
    
    /**
     * CHANGED: This is now a helper method used by the primary one.
     * Calculate performance metrics from equity curve
     * 
     * @param equity_curve Vector of (timestamp, equity_value) pairs
     * @param fills_count Number of fill events for trade statistics
     * @param include_fees Whether to account for transaction fees in calculations
     * @return RunSummary with all performance metrics
     */
    static RunSummary calculate_from_equity_curve(
        const std::vector<std::pair<std::string, double>>& equity_curve,
        int fills_count,
        bool include_fees = true
    );
    
    /**
     * Calculate performance metrics from audit events
     * Reconstructs equity curve from fill events and applies unified calculation
     * 
     * @param events Vector of audit events (fills, orders, etc.)
     * @param initial_capital Starting capital amount
     * @param include_fees Whether to apply Alpaca fee model
     * @return RunSummary with all performance metrics
     */
    static RunSummary calculate_from_audit_events(
        const std::vector<audit::Event>& events,
        double initial_capital = 100000.0,
        bool include_fees = true
    );
    
    /**
     * Reconstruct equity curve from audit fill events
     * Used by audit system to create consistent equity progression
     * 
     * @param events Vector of audit events
     * @param initial_capital Starting capital
     * @param include_fees Whether to apply transaction fees
     * @return Vector of (timestamp, equity_value) pairs
     */
    static std::vector<std::pair<std::string, double>> reconstruct_equity_curve_from_events(
        const std::vector<audit::Event>& events,
        double initial_capital = 100000.0,
        bool include_fees = true
    );
    
    /**
     * Calculate transaction fees for a trade using Alpaca cost model
     * 
     * @param symbol Trading symbol
     * @param quantity Trade quantity (positive for buy, negative for sell)
     * @param price Execution price
     * @return Total transaction fees
     */
    static double calculate_transaction_fees(
        const std::string& symbol,
        double quantity,
        double price
    );
    
    /**
     * Validate metrics consistency between two calculations
     * Used for testing and verification
     * 
     * @param metrics1 First metrics calculation
     * @param metrics2 Second metrics calculation
     * @param tolerance_pct Acceptable difference percentage (default 1%)
     * @return True if metrics are consistent within tolerance
     */
    static bool validate_metrics_consistency(
        const RunSummary& metrics1,
        const RunSummary& metrics2,
        double tolerance_pct = 1.0
    );
};

} // namespace sentio
