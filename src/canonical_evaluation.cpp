#include "sentio/canonical_evaluation.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>

namespace sentio {

BlockResult CanonicalEvaluator::calculate_block_metrics(
    const std::vector<std::pair<std::string, double>>& equity_curve,
    int block_index,
    double starting_equity,
    int fills_count,
    std::int64_t start_ts_ms,
    std::int64_t end_ts_ms) {
    
    BlockResult result;
    result.block_index = block_index;
    result.fills = fills_count;
    result.starting_equity = starting_equity;
    result.start_ts_ms = start_ts_ms;
    result.end_ts_ms = end_ts_ms;
    
    if (equity_curve.empty()) {
        result.ending_equity = starting_equity;
        result.return_per_block = 0.0;
        result.sharpe_ratio = 0.0;
        result.max_drawdown_pct = 0.0;
        return result;
    }
    
    result.ending_equity = equity_curve.back().second;
    
    // Calculate Return Per Block (RPB) - NET OF FEES
    // Note: equity_curve already includes Alpaca trading fees via portfolio.cash -= fees
    if (starting_equity > 0.0) {
        result.return_per_block = (result.ending_equity / starting_equity) - 1.0;
    } else {
        result.return_per_block = 0.0;
    }
    
    // Calculate bar-level returns for Sharpe ratio
    std::vector<double> returns;
    returns.reserve(equity_curve.size());
    
    double prev_equity = starting_equity;
    for (const auto& [timestamp, equity] : equity_curve) {
        if (prev_equity > 0.0) {
            double return_rate = (equity / prev_equity) - 1.0;
            returns.push_back(return_rate);
        }
        prev_equity = equity;
    }
    
    // Calculate Sharpe ratio for this block
    if (returns.size() > 1) {
        double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        
        double variance = 0.0;
        for (double ret : returns) {
            double diff = ret - mean_return;
            variance += diff * diff;
        }
        variance /= std::max<size_t>(1, returns.size() - 1); // Sample variance
        
        double std_dev = std::sqrt(variance);
        
        if (std_dev > 1e-12) {
            // Annualize assuming 390 bars per trading day, 252 trading days per year
            double annualized_return = mean_return * 390 * 252;
            double annualized_volatility = std_dev * std::sqrt(390 * 252);
            result.sharpe_ratio = annualized_return / annualized_volatility;
        } else {
            result.sharpe_ratio = 0.0;
        }
    } else {
        result.sharpe_ratio = 0.0;
    }
    
    // Calculate maximum drawdown within this block
    double peak = starting_equity;
    double max_drawdown = 0.0;
    
    for (const auto& [timestamp, equity] : equity_curve) {
        if (equity > peak) {
            peak = equity;
        } else {
            double drawdown = (peak - equity) / peak;
            max_drawdown = std::max(max_drawdown, drawdown);
        }
    }
    
    result.max_drawdown_pct = max_drawdown * 100.0;
    
    return result;
}

CanonicalReport CanonicalEvaluator::aggregate_block_results(
    const TradingBlockConfig& config,
    const std::vector<BlockResult>& block_results,
    const std::string& strategy_name,
    const std::string& dataset_source) {
    
    CanonicalReport report;
    report.config = config;
    report.block_results = block_results;
    report.strategy_name = strategy_name;
    report.dataset_source = dataset_source;
    
    if (block_results.empty()) {
        return report;
    }
    
    // Calculate aggregate statistics
    std::vector<double> rpb_values;
    rpb_values.reserve(block_results.size());
    
    double total_compounded_return = 1.0;
    int total_fills = 0;
    double max_drawdown = 0.0;
    
    for (const auto& block : block_results) {
        rpb_values.push_back(block.return_per_block);
        total_compounded_return *= (1.0 + block.return_per_block);
        total_fills += block.fills;
        max_drawdown = std::max(max_drawdown, block.max_drawdown_pct);
    }
    
    // Calculate mean RPB
    report.mean_rpb = std::accumulate(rpb_values.begin(), rpb_values.end(), 0.0) / rpb_values.size();
    
    // Calculate standard deviation of RPB
    double variance = 0.0;
    for (double rpb : rpb_values) {
        double diff = rpb - report.mean_rpb;
        variance += diff * diff;
    }
    variance /= std::max<size_t>(1, rpb_values.size() - 1); // Sample variance
    report.stdev_rpb = std::sqrt(variance);
    
    // Calculate Annualized Return on Block (ARB)
    report.annualized_return_on_block = calculate_annualized_return(report.mean_rpb, config.block_size);
    
    // Calculate aggregate Sharpe ratio
    std::vector<double> all_sharpe_values;
    for (const auto& block : block_results) {
        if (std::isfinite(block.sharpe_ratio) && block.sharpe_ratio != 0.0) {
            all_sharpe_values.push_back(block.sharpe_ratio);
        }
    }
    
    if (!all_sharpe_values.empty()) {
        report.aggregate_sharpe = std::accumulate(all_sharpe_values.begin(), all_sharpe_values.end(), 0.0) / all_sharpe_values.size();
    }
    
    // Set other aggregated values
    report.total_fills = total_fills;
    report.avg_fills_per_block = static_cast<double>(total_fills) / block_results.size();
    report.total_return_pct = (total_compounded_return - 1.0) * 100.0;
    report.max_drawdown_across_blocks = max_drawdown;
    report.consistency_score = calculate_consistency_score(block_results);
    
    // Set time bounds
    if (!block_results.empty()) {
        report.test_start_ts_ms = block_results.front().start_ts_ms;
        report.test_end_ts_ms = block_results.back().end_ts_ms;
        report.total_bars_processed = config.block_size * static_cast<int>(block_results.size());
    }
    
    return report;
}

double CanonicalEvaluator::calculate_annualized_return(
    double mean_rpb, 
    int block_size, 
    int bars_per_trading_year) {
    
    if (block_size <= 0 || bars_per_trading_year <= 0) {
        return 0.0;
    }
    
    // Calculate number of Trading Blocks in a trading year
    // With default 480-bar blocks: 98,280 / 480 = ~204.75 TB/year
    double blocks_per_year = static_cast<double>(bars_per_trading_year) / block_size;
    
    if (blocks_per_year <= 0) {
        return 0.0;
    }
    
    // Use compound interest formula: ARB = ((1 + mean_RPB) ^ blocks_per_year) - 1
    // This gives us the annualized equivalent of the per-block return
    return std::pow(1.0 + mean_rpb, blocks_per_year) - 1.0;
}

double CanonicalEvaluator::calculate_consistency_score(const std::vector<BlockResult>& block_results) {
    if (block_results.size() < 2) {
        return 0.0; // Perfect consistency for single or no blocks
    }
    
    // Calculate coefficient of variation of RPB values
    std::vector<double> rpb_values;
    rpb_values.reserve(block_results.size());
    
    for (const auto& block : block_results) {
        rpb_values.push_back(block.return_per_block);
    }
    
    double mean = std::accumulate(rpb_values.begin(), rpb_values.end(), 0.0) / rpb_values.size();
    
    if (std::abs(mean) < 1e-12) {
        return 1.0; // High inconsistency if mean is near zero but returns vary
    }
    
    double variance = 0.0;
    for (double rpb : rpb_values) {
        double diff = rpb - mean;
        variance += diff * diff;
    }
    variance /= rpb_values.size();
    
    double std_dev = std::sqrt(variance);
    
    // Return coefficient of variation (higher = less consistent)
    return std_dev / std::abs(mean);
}

} // namespace sentio
