#pragma once
#include <vector>
#include <string>
#include <cmath>

namespace sentio {

/**
 * Canonical Metrics Calculator
 * 
 * This is the single source of truth for all performance metrics calculations.
 * All systems (strattest, audit summarize, position-history) must use these functions
 * to ensure consistency across the entire platform.
 */
class CanonicalMetrics {
public:
    struct PerformanceMetrics {
        double total_return_pct;
        double monthly_projected_return_pct;
        double sharpe_ratio;
        double max_drawdown_pct;
        double daily_trades;
        double win_rate_pct;
        int total_trades;
        int trading_days;
        double starting_capital;
        double final_equity;
        double total_pnl;
    };

    /**
     * Calculate all performance metrics from equity curve
     * This is the canonical method that all systems should use
     */
    static PerformanceMetrics calculate_from_equity_curve(
        const std::vector<std::pair<std::string, double>>& equity_curve,
        double starting_capital = 100000.0,
        int total_trades = 0
    );

    /**
     * Calculate Monthly Projected Return (MPR) from daily returns
     * Uses geometric mean for accurate compounding
     */
    static double calculate_mpr_from_daily_returns(const std::vector<double>& daily_returns);

    /**
     * Calculate MPR from total return and trading days
     * Alternative method when daily returns are not available
     */
    static double calculate_mpr_from_total_return(double total_return, int trading_days);

    /**
     * Calculate Sharpe ratio from daily returns
     * Assumes risk-free rate of 0 for simplicity
     */
    static double calculate_sharpe_ratio(const std::vector<double>& daily_returns);

    /**
     * Calculate maximum drawdown from equity curve
     */
    static double calculate_max_drawdown(const std::vector<double>& equity_values);

    /**
     * Extract daily returns from equity curve
     * Handles timestamp parsing and deduplication
     */
    static std::vector<double> extract_daily_returns(
        const std::vector<std::pair<std::string, double>>& equity_curve
    );

    /**
     * Validate metrics consistency between different calculation methods
     * Returns true if all metrics are within acceptable tolerance
     */
    static bool validate_metrics_consistency(
        const PerformanceMetrics& metrics1,
        const PerformanceMetrics& metrics2,
        double tolerance_pct = 1.0  // 1% tolerance by default
    );

    /**
     * Print detailed metrics comparison for debugging
     */
    static void print_metrics_comparison(
        const PerformanceMetrics& canonical,
        const PerformanceMetrics& audit,
        const PerformanceMetrics& strattest,
        const std::string& run_id
    );

private:
    static constexpr double TRADING_DAYS_PER_MONTH = 21.0;
    static constexpr double TRADING_DAYS_PER_YEAR = 252.0;
    static constexpr double TOLERANCE_EPSILON = 1e-6;
};

} // namespace sentio
