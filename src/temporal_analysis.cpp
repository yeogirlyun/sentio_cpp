#include "sentio/temporal_analysis.hpp"
#include "sentio/runner.hpp"
#include "sentio/audit.hpp"
#include "sentio/metrics.hpp"
#include "sentio/progress_bar.hpp"
#include "sentio/day_index.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

namespace sentio {

TemporalAnalysisSummary run_temporal_analysis(const SymbolTable& ST,
                                            const std::vector<std::vector<Bar>>& series,
                                            int base_symbol_id,
                                            const RunnerCfg& rcfg,
                                            const TemporalAnalysisConfig& cfg) {
    
    const auto& base_series = series[base_symbol_id];
    const int total_bars = (int)base_series.size();
    
    if (total_bars < cfg.min_bars_per_quarter) {
        std::cerr << "ERROR: Insufficient data for temporal analysis. Need at least " 
                  << cfg.min_bars_per_quarter << " bars per quarter." << std::endl;
        return TemporalAnalysisSummary{};
    }
    
    // Calculate time periods based on actual trading periods
    // Assume ~390 bars per trading day (6.5 hours * 60 mins = 390 1-min bars)
    int bars_per_day = 390;
    int bars_per_week = 5 * bars_per_day;  // 5 trading days per week
    int bars_per_quarter = 66 * bars_per_day; // 66 trading days per quarter
    
    // Determine which time period to use and calculate the number of periods
    int num_periods = 0;
    int bars_per_period = 0;
    std::string period_name = "period";
    
    if (cfg.num_days > 0) {
        num_periods = cfg.num_days;
        bars_per_period = bars_per_day;
        period_name = "day";
    } else if (cfg.num_weeks > 0) {
        num_periods = cfg.num_weeks;
        bars_per_period = bars_per_week;
        period_name = "week";
    } else if (cfg.num_quarters > 0) {
        num_periods = cfg.num_quarters;
        bars_per_period = bars_per_quarter;
        period_name = "quarter";
    } else {
        // Default: analyze all data as one period
        num_periods = 1;
        bars_per_period = total_bars;
        period_name = "full period";
    }
    
    std::cout << "Starting TPA (Temporal Performance Analysis) Test..." << std::endl;
    std::cout << "Total bars: " << total_bars << ", Bars per " << period_name << ": " << bars_per_period << std::endl;
    
    // Initialize analyzer and progress bar
    TemporalAnalyzer analyzer;
    analyzer.set_period_name(period_name);
    TPATestProgressBar progress_bar(num_periods, rcfg.strategy_name, period_name);
    progress_bar.display(); // Show initial progress bar
    
    std::cout << "\nInitializing data processing..." << std::endl;
    
    // Build audit filename prefix with strategy and timestamp
    const std::string test_name = "tpa_test";
    const auto ts_epoch = static_cast<long long>(std::time(nullptr));

    // Determine the last num_periods periods from the end
    int total_periods_available = std::max(1, total_bars / std::max(1, bars_per_period));
    int start_period = std::max(0, total_periods_available - num_periods);
    for (int pi = 0; pi < num_periods; ++pi) {
        int p = start_period + pi;
        int start_idx = p * bars_per_period;
        int end_idx = std::min(start_idx + bars_per_period, total_bars);
        
        if (end_idx - start_idx < cfg.min_bars_per_quarter) {
            std::cout << "Skipping " << period_name << " " << (p + 1) << " - insufficient data" << std::endl;
            continue;
        }
        
        std::cout << "\nProcessing " << period_name << " " << (p + 1) 
                  << " (bars " << start_idx << "-" << end_idx << ")..." << std::endl;
        
        // Create data slice for this quarter
        std::vector<std::vector<Bar>> quarter_series;
        quarter_series.reserve(series.size());
        for (const auto& sym_series : series) {
            if (sym_series.size() > static_cast<size_t>(end_idx)) {
                quarter_series.emplace_back(sym_series.begin() + start_idx, sym_series.begin() + end_idx);
            } else if (sym_series.size() > static_cast<size_t>(start_idx)) {
                quarter_series.emplace_back(sym_series.begin() + start_idx, sym_series.end());
            } else {
                quarter_series.emplace_back();
            }
        }
        
        // Run backtest for this period
        AuditConfig audit_cfg;
        audit_cfg.run_id = rcfg.strategy_name + "_" + test_name + "_" + period_name + std::to_string(p + 1) + "_" + std::to_string(ts_epoch);
        audit_cfg.file_path = "audit/" + rcfg.strategy_name + "_" + test_name + "_" + std::to_string(ts_epoch) + "_" + period_name + std::to_string(p + 1) + ".jsonl";
        audit_cfg.flush_each = true;
        AuditRecorder audit(audit_cfg);
        
        auto result = run_backtest(audit, ST, quarter_series, base_symbol_id, rcfg);
        
        // Calculate period metrics
        QuarterlyMetrics metrics;
        metrics.year = 2024; // Use current year for all periods
        metrics.quarter = p + 1; // Use period number as quarter
        
        // Calculate actual trading days by extracting unique dates from base symbol bars
        // Use the first series (base symbol) to count trading days
        int actual_trading_days = 0;
        if (!series.empty() && series[0].size() > static_cast<size_t>(start_idx)) {
            int actual_end_idx = std::min(end_idx, static_cast<int>(series[0].size()));
            std::vector<Bar> quarter_bars(series[0].begin() + start_idx, series[0].begin() + actual_end_idx);
            auto day_starts = day_start_indices(quarter_bars);
            actual_trading_days = static_cast<int>(day_starts.size());
        } else {
            // Fallback: estimate trading days (approximately 66 trading days per quarter)
            actual_trading_days = std::max(1, (end_idx - start_idx) / 390); // ~390 bars per day
        }
        
        // Convert total return percent for the slice into a day-compounded monthly return
        // result.total_return is percent for the tested slice; convert to decimal for compounding
        double ret_dec = result.total_return / 100.0;
        double monthly_compounded = 0.0;
        if (actual_trading_days > 0) {
            // Compound to a 21-trading-day month
            monthly_compounded = (std::pow(1.0 + ret_dec, 21.0 / static_cast<double>(actual_trading_days)) - 1.0) * 100.0;
        }
        metrics.monthly_return_pct = monthly_compounded;
        
        metrics.sharpe_ratio = result.sharpe_ratio;
        metrics.total_trades = result.total_fills;  // Use total_fills as proxy for trades
        metrics.trading_days = actual_trading_days;
        metrics.avg_daily_trades = actual_trading_days > 0 ? static_cast<double>(result.total_fills) / actual_trading_days : 0.0;
        metrics.max_drawdown = result.max_drawdown;
        metrics.win_rate = 0.0;  // Not available in RunResult, set to 0
        metrics.total_return_pct = result.total_return;
        
        analyzer.add_quarterly_result(metrics);
        
        // Update progress bar with period results
        progress_bar.display_with_period_info(p + 1, 2024, p + 1,
                                             metrics.monthly_return_pct, metrics.sharpe_ratio,
                                             metrics.avg_daily_trades, metrics.health_status());
        
        // Print period summary
        std::cout << "\n  Monthly Return: " << std::fixed << std::setprecision(2) 
                  << metrics.monthly_return_pct << "%" << std::endl;
        std::cout << "  Sharpe Ratio: " << std::fixed << std::setprecision(3) 
                  << metrics.sharpe_ratio << std::endl;
        std::cout << "  Daily Trades: " << std::fixed << std::setprecision(1) 
                  << metrics.avg_daily_trades << " (Health: " << metrics.health_status() << ")" << std::endl;
        std::cout << "  Total Trades: " << metrics.total_trades << std::endl;
    }
    
    // Final progress bar update
    std::cout << "\n\nTPA Test completed! Generating summary..." << std::endl;
    
    auto summary = analyzer.generate_summary();
    summary.assess_readiness(cfg);
    return summary;
}

} // namespace sentio
