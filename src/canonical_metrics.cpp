#include "sentio/canonical_metrics.hpp"
#include <iostream>
#include <iomanip>
#include <map>
#include <algorithm>
#include <sstream>

namespace sentio {

CanonicalMetrics::PerformanceMetrics CanonicalMetrics::calculate_from_equity_curve(
    const std::vector<std::pair<std::string, double>>& equity_curve,
    double starting_capital,
    int total_trades) {
    
    PerformanceMetrics metrics = {};
    
    if (equity_curve.empty()) {
        return metrics;
    }
    
    // Extract basic values
    metrics.starting_capital = starting_capital;
    metrics.final_equity = equity_curve.back().second;
    metrics.total_pnl = metrics.final_equity - metrics.starting_capital;
    metrics.total_return_pct = (metrics.total_pnl / metrics.starting_capital) * 100.0;
    metrics.total_trades = total_trades;
    
    // Extract daily returns for advanced calculations
    std::vector<double> daily_returns = extract_daily_returns(equity_curve);
    metrics.trading_days = static_cast<int>(daily_returns.size());
    
    // Calculate derived metrics
    metrics.monthly_projected_return_pct = calculate_mpr_from_daily_returns(daily_returns) * 100.0;
    metrics.sharpe_ratio = calculate_sharpe_ratio(daily_returns);
    
    // Extract equity values for drawdown calculation
    std::vector<double> equity_values;
    equity_values.reserve(equity_curve.size());
    for (const auto& [timestamp, equity] : equity_curve) {
        equity_values.push_back(equity);
    }
    metrics.max_drawdown_pct = calculate_max_drawdown(equity_values) * 100.0;
    
    // Calculate daily trades
    metrics.daily_trades = (metrics.trading_days > 0) ? 
        static_cast<double>(metrics.total_trades) / metrics.trading_days : 0.0;
    
    // Win rate calculation would require individual trade data, set to 0 for now
    metrics.win_rate_pct = 0.0;
    
    return metrics;
}

double CanonicalMetrics::calculate_mpr_from_daily_returns(const std::vector<double>& daily_returns) {
    if (daily_returns.empty()) {
        return 0.0;
    }
    
    // Calculate geometric mean of daily returns
    double cumulative_return = 1.0;
    for (double daily_return : daily_returns) {
        cumulative_return *= (1.0 + daily_return);
    }
    
    // Convert to daily growth rate
    double daily_growth_rate = std::pow(cumulative_return, 1.0 / daily_returns.size()) - 1.0;
    
    // Project to monthly return (21 trading days)
    double monthly_return = std::pow(1.0 + daily_growth_rate, TRADING_DAYS_PER_MONTH) - 1.0;
    
    return monthly_return;
}

double CanonicalMetrics::calculate_mpr_from_total_return(double total_return, int trading_days) {
    if (trading_days <= 0) {
        return 0.0;
    }
    
    // Convert total return to daily growth rate
    double daily_growth_rate = std::pow(1.0 + total_return, 1.0 / trading_days) - 1.0;
    
    // Project to monthly return
    double monthly_return = std::pow(1.0 + daily_growth_rate, TRADING_DAYS_PER_MONTH) - 1.0;
    
    return monthly_return;
}

double CanonicalMetrics::calculate_sharpe_ratio(const std::vector<double>& daily_returns) {
    if (daily_returns.size() < 2) {
        return 0.0;
    }
    
    // Calculate mean and standard deviation of daily returns
    double sum = 0.0;
    for (double ret : daily_returns) {
        sum += ret;
    }
    double mean = sum / daily_returns.size();
    
    double variance = 0.0;
    for (double ret : daily_returns) {
        variance += (ret - mean) * (ret - mean);
    }
    variance /= (daily_returns.size() - 1);
    double std_dev = std::sqrt(variance);
    
    if (std_dev < TOLERANCE_EPSILON) {
        return 0.0;
    }
    
    // Annualize and calculate Sharpe ratio (assuming risk-free rate = 0)
    double annualized_return = mean * TRADING_DAYS_PER_YEAR;
    double annualized_volatility = std_dev * std::sqrt(TRADING_DAYS_PER_YEAR);
    
    return annualized_return / annualized_volatility;
}

double CanonicalMetrics::calculate_max_drawdown(const std::vector<double>& equity_values) {
    if (equity_values.empty()) {
        return 0.0;
    }
    
    double max_drawdown = 0.0;
    double peak = equity_values[0];
    
    for (double equity : equity_values) {
        if (equity > peak) {
            peak = equity;
        }
        
        double drawdown = (peak - equity) / peak;
        if (drawdown > max_drawdown) {
            max_drawdown = drawdown;
        }
    }
    
    return max_drawdown;
}

std::vector<double> CanonicalMetrics::extract_daily_returns(
    const std::vector<std::pair<std::string, double>>& equity_curve) {
    
    if (equity_curve.size() < 2) {
        return {};
    }
    
    // Group by session date (first 10 characters of timestamp: YYYY-MM-DD)
    std::map<std::string, double> daily_equity;
    for (const auto& [timestamp, equity] : equity_curve) {
        std::string session_date = timestamp.substr(0, 10);
        daily_equity[session_date] = equity; // Keep the last equity value for each day
    }
    
    // Calculate daily returns
    std::vector<double> daily_returns;
    auto it = daily_equity.begin();
    double prev_equity = it->second;
    ++it;
    
    for (; it != daily_equity.end(); ++it) {
        double current_equity = it->second;
        double daily_return = (current_equity - prev_equity) / prev_equity;
        daily_returns.push_back(daily_return);
        prev_equity = current_equity;
    }
    
    return daily_returns;
}

bool CanonicalMetrics::validate_metrics_consistency(
    const PerformanceMetrics& metrics1,
    const PerformanceMetrics& metrics2,
    double tolerance_pct) {
    
    auto within_tolerance = [tolerance_pct](double val1, double val2) {
        if (std::abs(val1) < TOLERANCE_EPSILON && std::abs(val2) < TOLERANCE_EPSILON) {
            return true; // Both are effectively zero
        }
        double max_val = std::max(std::abs(val1), std::abs(val2));
        return std::abs(val1 - val2) <= (max_val * tolerance_pct / 100.0);
    };
    
    return within_tolerance(metrics1.total_return_pct, metrics2.total_return_pct) &&
           within_tolerance(metrics1.monthly_projected_return_pct, metrics2.monthly_projected_return_pct) &&
           within_tolerance(metrics1.daily_trades, metrics2.daily_trades);
}

void CanonicalMetrics::print_metrics_comparison(
    const PerformanceMetrics& canonical,
    const PerformanceMetrics& audit,
    const PerformanceMetrics& strattest,
    const std::string& run_id) {
    
    std::cout << "=== CANONICAL METRICS COMPARISON ===" << std::endl;
    std::cout << "Run ID: " << run_id << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "\n┌─────────────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐" << std::endl;
    std::cout << "│ Metric                  │ Canonical       │ Audit Summarize │ Strattest       │ Max Discrepancy │" << std::endl;
    std::cout << "├─────────────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤" << std::endl;
    
    // Total Return
    double max_return_disc = std::max(std::abs(canonical.total_return_pct - audit.total_return_pct), 
                                     std::abs(canonical.total_return_pct - strattest.total_return_pct));
    std::cout << "│ Total Return            │ " << std::setw(14) << canonical.total_return_pct << "% │ " 
              << std::setw(14) << audit.total_return_pct << "% │ " 
              << std::setw(14) << strattest.total_return_pct << "% │ " 
              << std::setw(14) << max_return_disc << "% │" << std::endl;
    
    // MPR
    double max_mpr_disc = std::max(std::abs(canonical.monthly_projected_return_pct - audit.monthly_projected_return_pct), 
                                  std::abs(canonical.monthly_projected_return_pct - strattest.monthly_projected_return_pct));
    std::cout << "│ Monthly Proj. Return    │ " << std::setw(14) << canonical.monthly_projected_return_pct << "% │ " 
              << std::setw(14) << audit.monthly_projected_return_pct << "% │ " 
              << std::setw(14) << strattest.monthly_projected_return_pct << "% │ " 
              << std::setw(14) << max_mpr_disc << "% │" << std::endl;
    
    // Sharpe Ratio
    double max_sharpe_disc = std::max(std::abs(canonical.sharpe_ratio - audit.sharpe_ratio), 
                                     std::abs(canonical.sharpe_ratio - strattest.sharpe_ratio));
    std::cout << "│ Sharpe Ratio            │ " << std::setw(14) << canonical.sharpe_ratio << "  │ " 
              << std::setw(14) << audit.sharpe_ratio << "  │ " 
              << std::setw(14) << strattest.sharpe_ratio << "  │ " 
              << std::setw(14) << max_sharpe_disc << "  │" << std::endl;
    
    // Daily Trades
    double max_trades_disc = std::max(std::abs(canonical.daily_trades - audit.daily_trades), 
                                     std::abs(canonical.daily_trades - strattest.daily_trades));
    std::cout << "│ Daily Trades            │ " << std::setw(14) << canonical.daily_trades << "  │ " 
              << std::setw(14) << audit.daily_trades << "  │ " 
              << std::setw(14) << strattest.daily_trades << "  │ " 
              << std::setw(14) << max_trades_disc << "  │" << std::endl;
    
    std::cout << "└─────────────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘" << std::endl;
    
    // Validation results
    std::cout << "\n=== VALIDATION RESULTS ===" << std::endl;
    bool canonical_audit_consistent = validate_metrics_consistency(canonical, audit, 1.0);
    bool canonical_strattest_consistent = validate_metrics_consistency(canonical, strattest, 5.0); // Higher tolerance for strattest
    
    std::cout << "Canonical vs Audit: " << (canonical_audit_consistent ? "✅ CONSISTENT" : "❌ INCONSISTENT") << std::endl;
    std::cout << "Canonical vs Strattest: " << (canonical_strattest_consistent ? "✅ CONSISTENT" : "❌ INCONSISTENT") << std::endl;
}

} // namespace sentio
