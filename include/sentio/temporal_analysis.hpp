#pragma once
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <iostream>
#include "sentio/progress_bar.hpp"

namespace sentio {

struct TemporalAnalysisConfig {
    int num_quarters = 12;           // Number of quarters to analyze
    int min_bars_per_quarter = 50;   // Minimum bars required per quarter
    bool print_detailed_report = true;
    
    // TPA readiness criteria for virtual market testing
    double min_avg_sharpe = 0.5;     // Minimum average Sharpe ratio
    double max_sharpe_volatility = 1.0; // Maximum Sharpe volatility
    double min_health_quarters_pct = 70.0; // Minimum % of quarters with healthy trade frequency
    double min_avg_monthly_return = 0.5;   // Minimum average monthly return (%)
    double max_drawdown_threshold = 15.0;  // Maximum acceptable drawdown (%)
};

struct QuarterlyMetrics {
    int year;
    int quarter;
    double monthly_return_pct;      // Monthly projected return (annualized)
    double sharpe_ratio;
    int total_trades;
    int trading_days;
    double avg_daily_trades;
    double max_drawdown;
    double win_rate;
    double total_return_pct;
    
    // Health indicators
    bool healthy_trade_frequency() const {
        return avg_daily_trades >= 10.0 && avg_daily_trades <= 100.0;
    }
    
    std::string health_status() const {
        if (avg_daily_trades < 10.0) return "LOW_FREQ";
        if (avg_daily_trades > 100.0) return "HIGH_FREQ";
        return "HEALTHY";
    }
};

struct TPAReadinessAssessment {
    bool ready_for_virtual_market = false;
    bool ready_for_paper_trading = false;
    bool ready_for_live_trading = false;
    
    std::vector<std::string> issues;
    std::vector<std::string> recommendations;
    
    double readiness_score = 0.0; // 0-100 score
};

struct TemporalAnalysisSummary {
    std::vector<QuarterlyMetrics> quarterly_results;
    double overall_sharpe;
    double overall_return;
    int total_quarters;
    int healthy_quarters;
    int low_freq_quarters;
    int high_freq_quarters;
    
    // Consistency metrics
    double sharpe_std_dev;
    double return_std_dev;
    double trade_freq_std_dev;
    
    // TPA readiness assessment
    TPAReadinessAssessment readiness;
    
    void calculate_summary_stats() {
        if (quarterly_results.empty()) return;
        
        total_quarters = quarterly_results.size();
        healthy_quarters = 0;
        low_freq_quarters = 0;
        high_freq_quarters = 0;
        
        double sharpe_sum = 0.0, return_sum = 0.0, freq_sum = 0.0;
        double sharpe_sq_sum = 0.0, return_sq_sum = 0.0, freq_sq_sum = 0.0;
        
        for (const auto& q : quarterly_results) {
            sharpe_sum += q.sharpe_ratio;
            return_sum += q.monthly_return_pct;
            freq_sum += q.avg_daily_trades;
            
            sharpe_sq_sum += q.sharpe_ratio * q.sharpe_ratio;
            return_sq_sum += q.monthly_return_pct * q.monthly_return_pct;
            freq_sq_sum += q.avg_daily_trades * q.avg_daily_trades;
            
            if (q.health_status() == "HEALTHY") healthy_quarters++;
            else if (q.health_status() == "LOW_FREQ") low_freq_quarters++;
            else if (q.health_status() == "HIGH_FREQ") high_freq_quarters++;
        }
        
        overall_sharpe = sharpe_sum / total_quarters;
        overall_return = return_sum / total_quarters;
        
        // Calculate standard deviations
        double sharpe_mean = overall_sharpe;
        double return_mean = overall_return;
        double freq_mean = freq_sum / total_quarters;
        
        // For single quarter, standard deviation is 0 (no variance)
        if (total_quarters == 1) {
            sharpe_std_dev = 0.0;
            return_std_dev = 0.0;
            trade_freq_std_dev = 0.0;
        } else {
            sharpe_std_dev = std::sqrt(std::max(0.0, (sharpe_sq_sum / total_quarters) - (sharpe_mean * sharpe_mean)));
            return_std_dev = std::sqrt(std::max(0.0, (return_sq_sum / total_quarters) - (return_mean * return_mean)));
            trade_freq_std_dev = std::sqrt(std::max(0.0, (freq_sq_sum / total_quarters) - (freq_mean * freq_mean)));
        }
    }
    
    void assess_readiness(const TemporalAnalysisConfig& config) {
        readiness.issues.clear();
        readiness.recommendations.clear();
        readiness.ready_for_virtual_market = false;
        readiness.ready_for_paper_trading = false;
        readiness.ready_for_live_trading = false;
        
        double score = 0.0;
        int criteria_met = 0;
        [[maybe_unused]] int total_criteria = 5;
        
        // 1. Average Sharpe ratio check
        if (overall_sharpe >= config.min_avg_sharpe) {
            criteria_met++;
            score += 20.0;
        } else {
            readiness.issues.push_back("Average Sharpe ratio too low: " + std::to_string(overall_sharpe) + " < " + std::to_string(config.min_avg_sharpe));
            readiness.recommendations.push_back("Improve strategy risk-adjusted returns");
        }
        
        // 2. Sharpe volatility check
        if (sharpe_std_dev <= config.max_sharpe_volatility) {
            criteria_met++;
            score += 20.0;
        } else {
            readiness.issues.push_back("Sharpe ratio too volatile: " + std::to_string(sharpe_std_dev) + " > " + std::to_string(config.max_sharpe_volatility));
            readiness.recommendations.push_back("Improve strategy consistency across time periods");
        }
        
        // 3. Trade frequency health check
        double health_pct = 100.0 * healthy_quarters / total_quarters;
        if (health_pct >= config.min_health_quarters_pct) {
            criteria_met++;
            score += 20.0;
        } else {
            readiness.issues.push_back("Too many quarters with unhealthy trade frequency: " + std::to_string(health_pct) + "% < " + std::to_string(config.min_health_quarters_pct) + "%");
            readiness.recommendations.push_back("Adjust strategy parameters for consistent trade frequency");
        }
        
        // 4. Monthly return check
        if (overall_return >= config.min_avg_monthly_return) {
            criteria_met++;
            score += 20.0;
        } else {
            readiness.issues.push_back("Average monthly return too low: " + std::to_string(overall_return) + "% < " + std::to_string(config.min_avg_monthly_return) + "%");
            readiness.recommendations.push_back("Improve strategy profitability");
        }
        
        // 5. Drawdown check (check max drawdown across quarters)
        double max_quarterly_drawdown = 0.0;
        for (const auto& q : quarterly_results) {
            max_quarterly_drawdown = std::max(max_quarterly_drawdown, q.max_drawdown);
        }
        if (max_quarterly_drawdown <= config.max_drawdown_threshold) {
            criteria_met++;
            score += 20.0;
        } else {
            readiness.issues.push_back("Maximum drawdown too high: " + std::to_string(max_quarterly_drawdown) + "% > " + std::to_string(config.max_drawdown_threshold) + "%");
            readiness.recommendations.push_back("Implement better risk management");
        }
        
        readiness.readiness_score = score;
        
        // Determine readiness levels
        if (criteria_met >= 3) {
            readiness.ready_for_virtual_market = true;
        }
        if (criteria_met >= 4) {
            readiness.ready_for_paper_trading = true;
        }
        if (criteria_met >= 5) {
            readiness.ready_for_live_trading = true;
        }
        
        // Add general recommendations based on score
        if (score < 60.0) {
            readiness.recommendations.push_back("Strategy needs significant improvement before testing");
        } else if (score < 80.0) {
            readiness.recommendations.push_back("Strategy shows promise but needs refinement");
        } else {
            readiness.recommendations.push_back("Strategy appears ready for advanced testing");
        }
    }
};

class TPATestProgressBar : public ProgressBar {
public:
    TPATestProgressBar(int total_quarters, const std::string& strategy_name) 
        : ProgressBar(total_quarters, "TPA Test: " + strategy_name) {}
    
    void display_with_quarter_info([[maybe_unused]] int current_quarter, int year, int quarter, 
                                   double monthly_return, double sharpe, 
                                   double avg_daily_trades, const std::string& health_status) {
        update(get_current() + 1);
        
        // Clear line and move cursor to beginning
        std::cout << "\r\033[K";
        
        // Calculate percentage
        double percentage = (double)get_current() / get_total() * 100.0;
        
        // Create progress bar
        int bar_width = 50;
        int pos = (int)(bar_width * percentage / 100.0);
        
        std::cout << get_description() << " [" << std::string(pos, '=') 
                  << std::string(bar_width - pos, '-') << "] " 
                  << std::fixed << std::setprecision(1) << percentage << "%";
        
        // Show current quarter info
        std::cout << " | Q" << year << "Q" << quarter 
                  << " | Ret: " << std::fixed << std::setprecision(2) << monthly_return << "%"
                  << " | Sharpe: " << std::fixed << std::setprecision(3) << sharpe
                  << " | Trades: " << std::fixed << std::setprecision(1) << avg_daily_trades
                  << " | " << health_status;
        
        std::cout.flush();
    }
};

class TemporalAnalyzer {
public:
    TemporalAnalyzer() = default;
    
    void add_quarterly_result(const QuarterlyMetrics& metrics) {
        quarterly_results_.push_back(metrics);
    }
    
    TemporalAnalysisSummary generate_summary() const {
        TemporalAnalysisSummary summary;
        summary.quarterly_results = quarterly_results_;
        summary.calculate_summary_stats();
        return summary;
    }
    
    void print_detailed_report() const {
        auto summary = generate_summary();
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "TEMPORAL PERFORMANCE ANALYSIS REPORT" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        // Quarterly breakdown
        std::cout << "\nQUARTERLY PERFORMANCE BREAKDOWN:\n" << std::endl;
        std::cout << std::left << std::setw(8) << "Quarter" 
                  << std::setw(12) << "Monthly Ret%" 
                  << std::setw(10) << "Sharpe" 
                  << std::setw(8) << "Trades" 
                  << std::setw(12) << "Daily Avg" 
                  << std::setw(10) << "Health" 
                  << std::setw(10) << "Drawdown%" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (const auto& q : summary.quarterly_results) {
            std::cout << std::left << std::setw(8) << (std::to_string(q.year) + "Q" + std::to_string(q.quarter))
                      << std::setw(12) << std::fixed << std::setprecision(2) << q.monthly_return_pct
                      << std::setw(10) << std::fixed << std::setprecision(3) << q.sharpe_ratio
                      << std::setw(8) << q.total_trades
                      << std::setw(12) << std::fixed << std::setprecision(1) << q.avg_daily_trades
                      << std::setw(10) << q.health_status()
                      << std::setw(10) << std::fixed << std::setprecision(2) << q.max_drawdown << std::endl;
        }
        
        // Summary statistics
        std::cout << "\n" << std::string(80, '-') << std::endl;
        std::cout << "SUMMARY STATISTICS:\n" << std::endl;
        std::cout << "Overall Performance:" << std::endl;
        std::cout << "  Average Monthly Return: " << std::fixed << std::setprecision(2) 
                  << summary.overall_return << "%" << std::endl;
        std::cout << "  Average Sharpe Ratio: " << std::fixed << std::setprecision(3) 
                  << summary.overall_sharpe << std::endl;
        
        std::cout << "\nConsistency Metrics:" << std::endl;
        std::cout << "  Return Volatility (std): " << std::fixed << std::setprecision(2) 
                  << summary.return_std_dev << "%" << std::endl;
        std::cout << "  Sharpe Volatility (std): " << std::fixed << std::setprecision(3) 
                  << summary.sharpe_std_dev << std::endl;
        std::cout << "  Trade Frequency Volatility (std): " << std::fixed << std::setprecision(1) 
                  << summary.trade_freq_std_dev << " trades/day" << std::endl;
        
        std::cout << "\nTrade Frequency Health:" << std::endl;
        std::cout << "  Healthy Quarters: " << summary.healthy_quarters << "/" << summary.total_quarters 
                  << " (" << std::fixed << std::setprecision(1) 
                  << (100.0 * summary.healthy_quarters / summary.total_quarters) << "%)" << std::endl;
        std::cout << "  Low Frequency: " << summary.low_freq_quarters << " quarters" << std::endl;
        std::cout << "  High Frequency: " << summary.high_freq_quarters << " quarters" << std::endl;
        
        // Health assessment
        std::cout << "\nHEALTH ASSESSMENT:" << std::endl;
        double health_pct = 100.0 * summary.healthy_quarters / summary.total_quarters;
        if (health_pct >= 80.0) {
            std::cout << "  ✅ EXCELLENT: " << health_pct << "% of quarters have healthy trade frequency" << std::endl;
        } else if (health_pct >= 60.0) {
            std::cout << "  ⚠️  MODERATE: " << health_pct << "% of quarters have healthy trade frequency" << std::endl;
        } else {
            std::cout << "  ❌ POOR: " << health_pct << "% of quarters have healthy trade frequency" << std::endl;
        }
        
        // TPA Readiness Assessment
        std::cout << "\nTPA READINESS ASSESSMENT:" << std::endl;
        std::cout << "  Readiness Score: " << std::fixed << std::setprecision(1) << summary.readiness.readiness_score << "/100" << std::endl;
        
        std::cout << "\n  Testing Readiness:" << std::endl;
        std::cout << "  " << (summary.readiness.ready_for_virtual_market ? "✅" : "❌") 
                  << " Virtual Market Testing: " << (summary.readiness.ready_for_virtual_market ? "READY" : "NOT READY") << std::endl;
        std::cout << "  " << (summary.readiness.ready_for_paper_trading ? "✅" : "❌") 
                  << " Paper Trading: " << (summary.readiness.ready_for_paper_trading ? "READY" : "NOT READY") << std::endl;
        std::cout << "  " << (summary.readiness.ready_for_live_trading ? "✅" : "❌") 
                  << " Live Trading: " << (summary.readiness.ready_for_live_trading ? "READY" : "NOT READY") << std::endl;
        
        if (!summary.readiness.issues.empty()) {
            std::cout << "\n  Issues Identified:" << std::endl;
            for (const auto& issue : summary.readiness.issues) {
                std::cout << "  ❌ " << issue << std::endl;
            }
        }
        
        if (!summary.readiness.recommendations.empty()) {
            std::cout << "\n  Recommendations:" << std::endl;
            for (const auto& rec : summary.readiness.recommendations) {
                std::cout << "  💡 " << rec << std::endl;
            }
        }
        
        std::cout << std::string(80, '=') << std::endl;
    }

private:
    std::vector<QuarterlyMetrics> quarterly_results_;
};

// Forward declarations
struct SymbolTable;
struct Bar;
struct RunnerCfg;

// Main temporal analysis function
TemporalAnalysisSummary run_temporal_analysis(const SymbolTable& ST,
                                            const std::vector<std::vector<Bar>>& series,
                                            int base_symbol_id,
                                            const RunnerCfg& rcfg,
                                            const TemporalAnalysisConfig& cfg = TemporalAnalysisConfig{});

} // namespace sentio
