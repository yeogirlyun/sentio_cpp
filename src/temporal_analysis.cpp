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
    
    TemporalAnalyzer analyzer;
    const auto& base_series = series[base_symbol_id];
    const int total_bars = (int)base_series.size();
    
    if (total_bars < cfg.min_bars_per_quarter) {
        std::cerr << "ERROR: Insufficient data for temporal analysis. Need at least " 
                  << cfg.min_bars_per_quarter << " bars per quarter." << std::endl;
        return TemporalAnalysisSummary{};
    }
    
    // Calculate quarters based on data
    int bars_per_quarter = total_bars / cfg.num_quarters;
    int current_year = 2021; // Starting year
    int current_quarter = 1;
    
    std::cout << "Starting TPA (Temporal Performance Analysis) Test..." << std::endl;
    std::cout << "Total bars: " << total_bars << ", Bars per quarter: " << bars_per_quarter << std::endl;
    
    // Initialize TPA progress bar
    TPATestProgressBar progress_bar(cfg.num_quarters, rcfg.strategy_name);
    progress_bar.display(); // Show initial progress bar
    
    std::cout << "\nInitializing data processing..." << std::endl;
    
    for (int q = 0; q < cfg.num_quarters; ++q) {
        int start_idx = q * bars_per_quarter;
        int end_idx = std::min(start_idx + bars_per_quarter, total_bars);
        
        if (end_idx - start_idx < cfg.min_bars_per_quarter) {
            std::cout << "Skipping quarter " << (q + 1) << " - insufficient data" << std::endl;
            continue;
        }
        
        std::cout << "\nProcessing Quarter " << current_year << "Q" << current_quarter 
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
        
        // Run backtest for this quarter
        AuditConfig audit_cfg;
        audit_cfg.run_id = "temporal_q" + std::to_string(q + 1);
        audit_cfg.file_path = "audit/temporal_q" + std::to_string(q + 1) + ".jsonl";
        audit_cfg.flush_each = true;
        AuditRecorder audit(audit_cfg);
        
        auto result = run_backtest(audit, ST, quarter_series, base_symbol_id, rcfg);
        
        // Calculate quarterly metrics
        QuarterlyMetrics metrics;
        metrics.year = current_year;
        metrics.quarter = current_quarter;
        
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
        
        double months_in_quarter = actual_trading_days / 21.0; // ~21 trading days per month
        metrics.monthly_return_pct = (result.total_return * 12.0) / months_in_quarter;
        
        metrics.sharpe_ratio = result.sharpe_ratio;
        metrics.total_trades = result.total_fills;  // Use total_fills as proxy for trades
        metrics.trading_days = actual_trading_days;
        metrics.avg_daily_trades = actual_trading_days > 0 ? static_cast<double>(result.total_fills) / actual_trading_days : 0.0;
        metrics.max_drawdown = result.max_drawdown;
        metrics.win_rate = 0.0;  // Not available in RunResult, set to 0
        metrics.total_return_pct = result.total_return * 100.0;
        
        analyzer.add_quarterly_result(metrics);
        
        // Update progress bar with quarter results
        progress_bar.display_with_quarter_info(q + 1, current_year, current_quarter,
                                             metrics.monthly_return_pct, metrics.sharpe_ratio,
                                             metrics.avg_daily_trades, metrics.health_status());
        
        // Print quarter summary
        std::cout << "\n  Monthly Return: " << std::fixed << std::setprecision(2) 
                  << metrics.monthly_return_pct << "%" << std::endl;
        std::cout << "  Sharpe Ratio: " << std::fixed << std::setprecision(3) 
                  << metrics.sharpe_ratio << std::endl;
        std::cout << "  Daily Trades: " << std::fixed << std::setprecision(1) 
                  << metrics.avg_daily_trades << " (Health: " << metrics.health_status() << ")" << std::endl;
        std::cout << "  Total Trades: " << metrics.total_trades << std::endl;
        
        // Update year/quarter for next iteration
        current_quarter++;
        if (current_quarter > 4) {
            current_quarter = 1;
            current_year++;
        }
    }
    
    // Final progress bar update
    std::cout << "\n\nTPA Test completed! Generating summary..." << std::endl;
    
    auto summary = analyzer.generate_summary();
    summary.assess_readiness(cfg);
    return summary;
}

} // namespace sentio
