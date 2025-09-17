#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace sentio {

/**
 * Canonical Evaluation System for Trading Strategy Performance
 * 
 * This system replaces ambiguous calendar-based time measurements with deterministic
 * bar-based evaluation units called "Trading Blocks". This eliminates discrepancies
 * between strattest and audit systems by creating a single source of truth for
 * performance measurement that is immune to weekends, holidays, and variable trading hours.
 */

/**
 * Configuration for Trading Block-based evaluation
 * Replaces duration strings like "2w" with deterministic Trading Block counts
 * 
 * Standard Units:
 * - 1 Trading Block (TB) = 480 bars ≈ 8 hours of trading
 * - 10 TB ≈ 10 trading days (default for quick tests)  
 * - 20 TB ≈ 1 month of trading (standard benchmark)
 */
struct TradingBlockConfig {
    int block_size = 480;     // Number of bars per Trading Block (TB) - 8 hours
    int num_blocks = 10;      // Number of Trading Blocks to test (default: 10 TB)
    
    // Total bars this configuration will process
    int total_bars() const { return block_size * num_blocks; }
    
    // Helper methods for common configurations
    static TradingBlockConfig quick_test() { return {480, 10}; }      // 10 TB ≈ 2 weeks
    static TradingBlockConfig standard_monthly() { return {480, 20}; } // 20 TB ≈ 1 month
    static TradingBlockConfig extended_test() { return {480, 60}; }    // 60 TB ≈ 3 months
};

/**
 * Performance metrics for a single Trading Block
 * Each block represents exactly block_size consecutive market data bars
 */
struct BlockResult {
    int block_index;                    // 0-based block number
    double return_per_block;            // Total compounded return for this block (RPB)
    double sharpe_ratio;                // Sharpe ratio for this block only
    double max_drawdown_pct;            // Maximum drawdown within this block
    int fills;                          // Number of fill events in this block
    double starting_equity;             // Equity at block start
    double ending_equity;               // Equity at block end
    
    // Timestamps for audit verification - defines exact bar range
    std::int64_t start_ts_ms;           // Timestamp of first bar in block
    std::int64_t end_ts_ms;             // Timestamp of last bar in block
    
    // Additional metrics
    double win_rate_pct = 0.0;          // Win rate for trades in this block
    double avg_trade_pnl = 0.0;         // Average P&L per trade in this block
    int winning_trades = 0;             // Number of winning trades
    int losing_trades = 0;              // Number of losing trades
};

/**
 * Final canonical performance report aggregating all blocks
 * This is the single source of truth for strategy evaluation
 */
struct CanonicalReport {
    TradingBlockConfig config;                  // Configuration used for this test
    std::vector<BlockResult> block_results;     // Results for each block
    
    // Aggregated statistics across all blocks
    double mean_rpb = 0.0;                      // Mean Return Per Block
    double stdev_rpb = 0.0;                     // Standard deviation of RPB
    double annualized_return_on_block = 0.0;    // ARB - annualized equivalent
    double aggregate_sharpe = 0.0;              // Overall Sharpe ratio
    double max_drawdown_across_blocks = 0.0;    // Maximum drawdown across all blocks
    
    // Trading statistics
    int total_fills = 0;                        // Total fills across all blocks
    double avg_fills_per_block = 0.0;           // Average fills per block
    double total_return_pct = 0.0;              // Total compounded return across all blocks
    double consistency_score = 0.0;             // Metric for performance consistency (lower stdev_rpb is better)
    
    // Metadata
    std::string strategy_name;                  // Strategy that was tested
    std::string dataset_source;                 // Source of market data used
    std::int64_t test_start_ts_ms = 0;          // Timestamp when test started
    std::int64_t test_end_ts_ms = 0;            // Timestamp when test ended
    int total_bars_processed = 0;               // Actual number of bars processed
    
    // Helper methods
    bool is_valid() const { return !block_results.empty(); }
    int successful_blocks() const { return static_cast<int>(block_results.size()); }
    double completion_rate() const { 
        return config.num_blocks > 0 ? static_cast<double>(successful_blocks()) / config.num_blocks : 0.0; 
    }
};

/**
 * Canonical Metrics Calculator
 * Single source of truth for all block-level performance calculations
 * Used by both strattest and audit systems to ensure consistency
 */
class CanonicalEvaluator {
public:
    /**
     * Calculate performance metrics for a single block of trading data
     * This is the atomic unit of performance measurement
     */
    static BlockResult calculate_block_metrics(
        const std::vector<std::pair<std::string, double>>& equity_curve,
        int block_index,
        double starting_equity,
        int fills_count,
        std::int64_t start_ts_ms,
        std::int64_t end_ts_ms
    );
    
    /**
     * Aggregate block results into a final canonical report
     * Calculates mean, standard deviation, and annualized metrics
     */
    static CanonicalReport aggregate_block_results(
        const TradingBlockConfig& config,
        const std::vector<BlockResult>& block_results,
        const std::string& strategy_name = "",
        const std::string& dataset_source = ""
    );
    
    /**
     * Calculate Annualized Return on Block (ARB) from mean RPB
     * Uses compound interest formula with trading calendar assumptions
     * 
     * With 480-bar Trading Blocks:
     * - 98,280 bars/year ÷ 480 bars/TB = ~204.75 TB/year
     * - ARB = ((1 + mean_RPB) ^ 204.75) - 1
     */
    static double calculate_annualized_return(
        double mean_rpb, 
        int block_size = 480,
        int bars_per_trading_year = 98280  // 252 days * 390 bars/day for QQQ
    );
    
    /**
     * Calculate consistency score from block results
     * Lower scores indicate more consistent performance
     */
    static double calculate_consistency_score(const std::vector<BlockResult>& block_results);
};

} // namespace sentio
