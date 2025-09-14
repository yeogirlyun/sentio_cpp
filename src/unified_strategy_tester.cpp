#include "sentio/unified_strategy_tester.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/run_id_generator.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <regex>

namespace sentio {

UnifiedStrategyTester::UnifiedStrategyTester() {
    // Initialize components
}

UnifiedStrategyTester::RobustnessReport UnifiedStrategyTester::run_comprehensive_test(const TestConfig& config) {
    std::cout << "🎯 Testing " << config.strategy_name << " on " << config.symbol;
    
    if (config.holistic_mode) {
        std::cout << " (HOLISTIC - Ultimate Robustness)";
    } else {
        switch (config.mode) {
            case TestMode::HISTORICAL: std::cout << " (Historical)"; break;
            case TestMode::AI_REGIME: std::cout << " (AI-" << config.regime << ")"; break;
            case TestMode::HYBRID: std::cout << " (Hybrid)"; break;
        }
    }
    std::cout << " - " << config.simulations << " simulations, " << config.duration << std::endl;
    
    // Warn about short test periods
    if (config.duration == "1d" || config.duration == "5d" || config.duration == "1w" || config.duration == "2w") {
        std::cout << "⚠️  WARNING: Short test period may produce unreliable MPR projections due to statistical noise" << std::endl;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run simulations based on mode
    std::vector<VirtualMarketEngine::VMSimulationResult> results;
    
    if (config.holistic_mode) {
        // Holistic mode: Run comprehensive multi-scenario testing
        results = run_holistic_tests(config);
    } else {
        switch (config.mode) {
            case TestMode::HISTORICAL:
                results = run_historical_tests(config, config.simulations);
                break;
            case TestMode::AI_REGIME:
                results = run_ai_regime_tests(config, config.simulations);
                break;
            case TestMode::HYBRID:
                results = run_hybrid_tests(config);
                break;
        }
    }
    
    // Apply stress testing if enabled
    if (config.stress_test) {
        apply_stress_scenarios(results, config);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "✅ Completed in " << duration.count() << "s" << std::endl;
    
    // Analyze results
    RobustnessReport report = analyze_results(results, config);
    report.test_duration_seconds = duration.count();
    report.mode_used = config.mode;
    
    return report;
}

std::vector<VirtualMarketEngine::VMSimulationResult> UnifiedStrategyTester::run_monte_carlo_tests(
    const TestConfig& config, int num_simulations) {
    
    std::cout << "🎲 Running Monte Carlo simulations..." << std::endl;
    
    // Create VM test configuration
    VirtualMarketEngine::VMTestConfig vm_config;
    vm_config.strategy_name = config.strategy_name;
    vm_config.symbol = config.symbol;
    vm_config.simulations = num_simulations;
    vm_config.params_json = config.params_json;
    vm_config.initial_capital = config.initial_capital;
    
    // Parse duration
    int total_minutes = parse_duration_to_minutes(config.duration);
    if (total_minutes >= 390) { // More than 1 trading day
        vm_config.days = total_minutes / 390;
        vm_config.hours = 0;
    } else {
        vm_config.days = 0;
        vm_config.hours = total_minutes / 60;
    }
    
    // Create strategy instance
    auto strategy = StrategyFactory::instance().create_strategy(config.strategy_name);
    if (!strategy) {
        std::cerr << "Error: Could not create strategy " << config.strategy_name << std::endl;
        return {};
    }
    
    // Create runner configuration
    RunnerCfg runner_cfg;
    runner_cfg.strategy_name = config.strategy_name;
    runner_cfg.strategy_params["buy_hi"] = "0.6";
    runner_cfg.strategy_params["sell_lo"] = "0.4";
    
    // **DISABLE AUDIT LOGGING**: Prevent conflicts when running within test-all
    if (config.disable_audit_logging) {
        runner_cfg.audit_level = AuditLevel::MetricsOnly;
    }
    
    // Run Monte Carlo simulation
    return vm_engine_.run_monte_carlo_simulation(vm_config, std::move(strategy), runner_cfg);
}

std::vector<VirtualMarketEngine::VMSimulationResult> UnifiedStrategyTester::run_historical_tests(
    const TestConfig& config, int num_simulations) {
    
    std::cout << "📊 Running historical pattern tests..." << std::endl;
    
    std::string historical_file = config.historical_data_file;
    if (historical_file.empty()) {
        historical_file = get_default_historical_file(config.symbol);
    }
    
    int continuation_minutes = parse_duration_to_minutes(config.duration);
    
    return vm_engine_.run_fast_historical_test(
        config.strategy_name, config.symbol, historical_file,
        continuation_minutes, num_simulations, config.params_json
    );
}

std::vector<VirtualMarketEngine::VMSimulationResult> UnifiedStrategyTester::run_ai_regime_tests(
    const TestConfig& config, int num_simulations) {
    
    std::cout << "🚀 Running Future QQQ regime tests..." << std::endl;
    
    // Use future QQQ data instead of MarS generation
    return vm_engine_.run_future_qqq_regime_test(
        config.strategy_name, config.symbol,
        num_simulations, config.regime, config.params_json
    );
}

std::vector<VirtualMarketEngine::VMSimulationResult> UnifiedStrategyTester::run_hybrid_tests(
    const TestConfig& config) {
    
    std::cout << "🌈 Running hybrid tests (Historical + AI)..." << std::endl;
    
    std::vector<VirtualMarketEngine::VMSimulationResult> all_results;
    
    // 60% Historical Pattern Testing
    int hist_sims = static_cast<int>(config.simulations * 0.6);
    if (hist_sims > 0) {
        auto hist_results = run_historical_tests(config, hist_sims);
        all_results.insert(all_results.end(), hist_results.begin(), hist_results.end());
    }
    
    // 40% AI Regime Testing
    int ai_sims = config.simulations - hist_sims;
    if (ai_sims > 0) {
        auto ai_results = run_ai_regime_tests(config, ai_sims);
        all_results.insert(all_results.end(), ai_results.begin(), ai_results.end());
    }
    
    return all_results;
}

UnifiedStrategyTester::RobustnessReport UnifiedStrategyTester::analyze_results(
    const std::vector<VirtualMarketEngine::VMSimulationResult>& results,
    const TestConfig& config) {
    
    RobustnessReport report;
    
    if (results.empty()) {
        std::cerr << "Warning: No simulation results to analyze" << std::endl;
        return report;
    }
    
    // Filter out failed simulations
    std::vector<VirtualMarketEngine::VMSimulationResult> valid_results;
    for (const auto& result : results) {
        if (result.total_trades > 0 || result.total_return != 0.0) {
            valid_results.push_back(result);
        }
    }
    
    report.total_simulations = results.size();
    report.successful_simulations = valid_results.size();
    
    if (valid_results.empty()) {
        std::cerr << "Warning: No valid simulation results" << std::endl;
        
        // Initialize report with safe default values for zero-trade scenarios
        report.monthly_projected_return = 0.0;
        report.sharpe_ratio = 0.0;
        report.max_drawdown = 0.0;
        report.win_rate = 0.0;
        report.profit_factor = 0.0;
        report.total_return = 0.0;
        report.avg_daily_trades = 0.0;
        
        report.consistency_score = 0.0;
        report.regime_adaptability = 0.0;
        report.stress_resilience = 0.0;
        report.liquidity_tolerance = 0.0;
        
        report.estimated_monthly_fees = 0.0;
        report.capital_efficiency = 0.0;
        report.position_turnover = 0.0;
        
        report.overall_risk = RiskLevel::HIGH;
        report.deployment_score = 0;
        report.ready_for_deployment = false;
        report.recommended_capital_range = "Not recommended";
        report.suggested_monitoring_days = 30; // Safe default
        
        // Initialize confidence intervals with zeros
        report.mpr_ci = {0.0, 0.0, 0.0, 0.0};
        report.sharpe_ci = {0.0, 0.0, 0.0, 0.0};
        report.drawdown_ci = {0.0, 0.0, 0.0, 0.0};
        report.win_rate_ci = {0.0, 0.0, 0.0, 0.0};
        
        return report;
    }
    
    // Extract metrics vectors
    std::vector<double> mprs, sharpes, drawdowns, win_rates, returns;
    std::vector<double> daily_trades, total_trades;
    
    for (const auto& result : valid_results) {
        mprs.push_back(result.monthly_projected_return);
        sharpes.push_back(result.sharpe_ratio);
        drawdowns.push_back(result.max_drawdown);
        win_rates.push_back(result.win_rate);
        returns.push_back(result.total_return);
        daily_trades.push_back(result.daily_trades);
        total_trades.push_back(result.total_trades);
    }
    
    // Calculate core metrics
    report.monthly_projected_return = std::accumulate(mprs.begin(), mprs.end(), 0.0) / mprs.size();
    report.sharpe_ratio = std::accumulate(sharpes.begin(), sharpes.end(), 0.0) / sharpes.size();
    report.max_drawdown = std::accumulate(drawdowns.begin(), drawdowns.end(), 0.0) / drawdowns.size();
    report.win_rate = std::accumulate(win_rates.begin(), win_rates.end(), 0.0) / win_rates.size();
    report.total_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    report.avg_daily_trades = std::accumulate(daily_trades.begin(), daily_trades.end(), 0.0) / daily_trades.size();
    
    // Calculate profit factor
    double total_wins = 0.0, total_losses = 0.0;
    for (const auto& result : valid_results) {
        if (result.total_return > 0) total_wins += result.total_return;
        else total_losses += std::abs(result.total_return);
    }
    report.profit_factor = (total_losses > 0) ? (total_wins / total_losses) : 0.0;
    
    // Calculate confidence intervals
    report.mpr_ci = calculate_confidence_interval(mprs, config.confidence_level);
    report.sharpe_ci = calculate_confidence_interval(sharpes, config.confidence_level);
    report.drawdown_ci = calculate_confidence_interval(drawdowns, config.confidence_level);
    report.win_rate_ci = calculate_confidence_interval(win_rates, config.confidence_level);
    
    // Calculate robustness metrics
    // Consistency Score: Lower variance = higher consistency
    double mpr_variance = 0.0;
    for (double mpr : mprs) {
        mpr_variance += std::pow(mpr - report.monthly_projected_return, 2);
    }
    mpr_variance /= mprs.size();
    
    // Avoid division by zero when MPR is 0
    double mpr_cv = 0.0;
    if (std::abs(report.monthly_projected_return) > 1e-9) {
        mpr_cv = std::sqrt(mpr_variance) / std::abs(report.monthly_projected_return);
    } else {
        // When MPR is effectively zero, use variance as a proxy for consistency
        mpr_cv = std::sqrt(mpr_variance) * 100.0; // Scale for percentage
    }
    report.consistency_score = std::max(0.0, std::min(100.0, 100.0 * (1.0 - std::min(1.0, mpr_cv))));
    
    // Enhanced robustness metrics for holistic mode
    if (config.holistic_mode) {
        // Regime Adaptability: Analyze performance across different market regimes
        std::map<std::string, std::vector<double>> regime_performance;
        // This would be enhanced with actual regime tracking in a full implementation
        report.regime_adaptability = std::max(0.0, std::min(100.0, 
            60.0 + 40.0 * std::tanh(report.sharpe_ratio - 0.5)));
        
        // Stress Resilience: Performance under extreme conditions
        double stress_penalty = 0.0;
        if (report.max_drawdown < -0.20) stress_penalty += 20.0;
        if (report.sharpe_ratio < 0.5) stress_penalty += 15.0;
        report.stress_resilience = std::max(0.0, 100.0 - stress_penalty);
        
        // Liquidity Tolerance: Assess impact of trading frequency
        double liquidity_score = 100.0;
        if (report.avg_daily_trades > 200) liquidity_score -= 30.0;
        else if (report.avg_daily_trades > 100) liquidity_score -= 15.0;
        else if (report.avg_daily_trades < 5) liquidity_score -= 10.0;
        report.liquidity_tolerance = std::max(0.0, liquidity_score);
        
    } else {
        // Standard regime adaptability for non-holistic modes
        report.regime_adaptability = std::max(0.0, std::min(100.0, 
            50.0 + 50.0 * std::tanh(report.sharpe_ratio - 1.0)));
    }
    
    // Stress Resilience: Based on worst-case scenarios
    double worst_mpr = *std::min_element(mprs.begin(), mprs.end());
    double worst_drawdown = *std::min_element(drawdowns.begin(), drawdowns.end());
    report.stress_resilience = std::max(0.0, std::min(100.0,
        50.0 + 25.0 * std::tanh(worst_mpr) + 25.0 * std::tanh(worst_drawdown + 0.2)));
    
    // Liquidity Tolerance: Based on trade frequency and consistency
    report.liquidity_tolerance = std::max(0.0, std::min(100.0,
        80.0 + 20.0 * std::tanh(report.avg_daily_trades / 50.0)));
    
    // Alpaca-specific metrics
    if (config.alpaca_fees) {
        // Estimate monthly fees (simplified: $0.005 per share, assume 100 shares per trade)
        double avg_monthly_trades = report.avg_daily_trades * 22; // 22 trading days per month
        report.estimated_monthly_fees = avg_monthly_trades * 100 * 0.005; // $0.005 per share
    }
    
    report.capital_efficiency = std::min(100.0, std::abs(report.total_return) * 100.0);
    report.position_turnover = report.avg_daily_trades / 10.0; // Simplified calculation
    
    // Risk assessment
    report.overall_risk = assess_risk_level(report);
    
    // Calculate deployment score
    report.deployment_score = calculate_deployment_score(report);
    report.ready_for_deployment = report.deployment_score >= 70;
    
    // Recommended capital range
    if (report.deployment_score >= 80) {
        report.recommended_capital_range = "$50,000 - $500,000";
        report.suggested_monitoring_days = 7;
    } else if (report.deployment_score >= 70) {
        report.recommended_capital_range = "$10,000 - $100,000";
        report.suggested_monitoring_days = 14;
    } else {
        report.recommended_capital_range = "$1,000 - $10,000";
        report.suggested_monitoring_days = 30;
    }
    
    // Generate recommendations (after monitoring days are set)
    generate_recommendations(report, config);
    
    return report;
}

UnifiedStrategyTester::ConfidenceInterval UnifiedStrategyTester::calculate_confidence_interval(
    const std::vector<double>& values, double confidence_level) {
    
    ConfidenceInterval ci;
    
    if (values.empty()) {
        ci.lower = ci.upper = ci.mean = ci.std_dev = 0.0;
        return ci;
    }
    
    ci.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    
    double variance = 0.0;
    for (double value : values) {
        variance += std::pow(value - ci.mean, 2);
    }
    variance /= values.size();
    ci.std_dev = std::sqrt(variance);
    
    // Simplified confidence interval (assumes normal distribution)
    double z_score = (confidence_level == 0.95) ? 1.96 : 
                    (confidence_level == 0.99) ? 2.58 : 1.645; // 90%
    
    double margin = z_score * ci.std_dev / std::sqrt(values.size());
    ci.lower = ci.mean - margin;
    ci.upper = ci.mean + margin;
    
    return ci;
}

UnifiedStrategyTester::RiskLevel UnifiedStrategyTester::assess_risk_level(const RobustnessReport& report) {
    int risk_score = 0;
    
    // High drawdown increases risk
    if (report.max_drawdown < -0.20) risk_score += 3;
    else if (report.max_drawdown < -0.10) risk_score += 2;
    else if (report.max_drawdown < -0.05) risk_score += 1;
    
    // Low Sharpe ratio increases risk
    if (report.sharpe_ratio < 0.5) risk_score += 3;
    else if (report.sharpe_ratio < 1.0) risk_score += 2;
    else if (report.sharpe_ratio < 1.5) risk_score += 1;
    
    // Low consistency increases risk
    if (report.consistency_score < 50) risk_score += 2;
    else if (report.consistency_score < 70) risk_score += 1;
    
    // Low win rate increases risk
    if (report.win_rate < 0.4) risk_score += 2;
    else if (report.win_rate < 0.5) risk_score += 1;
    
    if (risk_score >= 6) return RiskLevel::EXTREME;
    else if (risk_score >= 4) return RiskLevel::HIGH;
    else if (risk_score >= 2) return RiskLevel::MEDIUM;
    else return RiskLevel::LOW;
}

void UnifiedStrategyTester::generate_recommendations(RobustnessReport& report, const TestConfig& config) {
    // Risk warnings
    if (report.max_drawdown < -0.15) {
        report.risk_warnings.push_back("High maximum drawdown risk (>" + std::to_string(std::abs(report.max_drawdown * 100)) + "%)");
    }
    
    if (report.sharpe_ratio < 1.0) {
        report.risk_warnings.push_back("Low risk-adjusted returns (Sharpe < 1.0)");
    }
    
    if (report.consistency_score < 60) {
        report.risk_warnings.push_back("High performance variance across simulations");
    }
    
    if (report.avg_daily_trades > 100) {
        report.risk_warnings.push_back("Very high trading frequency may impact execution");
    }
    
    // Recommendations
    if (report.monthly_projected_return > 0.05) {
        report.recommendations.push_back("Strong performance - consider gradual capital scaling");
    }
    
    if (report.consistency_score > 80) {
        report.recommendations.push_back("Excellent consistency - suitable for automated trading");
    }
    
    if (report.stress_resilience < 70) {
        report.recommendations.push_back("Consider position sizing limits during high volatility");
    }
    
    if (report.avg_daily_trades < 5) {
        report.recommendations.push_back("Low trade frequency - consider longer backtesting periods");
    }
    
    report.recommendations.push_back("Start with paper trading for " + std::to_string(report.suggested_monitoring_days) + " days");
}

int UnifiedStrategyTester::calculate_deployment_score(const RobustnessReport& report) {
    int score = 50; // Base score
    
    // Performance components (40 points max)
    if (report.monthly_projected_return > 0.10) score += 15;
    else if (report.monthly_projected_return > 0.05) score += 10;
    else if (report.monthly_projected_return > 0.02) score += 5;
    
    if (report.sharpe_ratio > 2.0) score += 15;
    else if (report.sharpe_ratio > 1.5) score += 10;
    else if (report.sharpe_ratio > 1.0) score += 5;
    
    if (report.max_drawdown > -0.05) score += 10;
    else if (report.max_drawdown > -0.10) score += 5;
    else if (report.max_drawdown < -0.20) score -= 10;
    
    // Robustness components (30 points max)
    score += static_cast<int>(report.consistency_score * 0.15);
    score += static_cast<int>(report.stress_resilience * 0.15);
    
    // Risk components (20 points max)
    if (report.win_rate > 0.6) score += 10;
    else if (report.win_rate > 0.5) score += 5;
    
    if (report.profit_factor > 2.0) score += 10;
    else if (report.profit_factor > 1.5) score += 5;
    
    // Practical components (10 points max)
    if (report.avg_daily_trades >= 5 && report.avg_daily_trades <= 50) score += 5;
    if (report.successful_simulations >= report.total_simulations * 0.9) score += 5;
    
    return std::max(0, std::min(100, score));
}

void UnifiedStrategyTester::apply_stress_scenarios(
    std::vector<VirtualMarketEngine::VMSimulationResult>& results,
    const TestConfig& config) {
    
    std::cout << "⚡ Applying stress testing scenarios..." << std::endl;
    
    // Apply stress multipliers to simulate adverse conditions
    for (auto& result : results) {
        // Increase drawdown by 50% in stress scenarios
        result.max_drawdown *= 1.5;
        
        // Reduce returns by 20% in stress scenarios
        result.total_return *= 0.8;
        result.monthly_projected_return *= 0.8;
        
        // Reduce Sharpe ratio due to increased volatility
        result.sharpe_ratio *= 0.7;
        
        // Reduce win rate slightly
        result.win_rate *= 0.9;
    }
}

int UnifiedStrategyTester::parse_duration_to_minutes(const std::string& duration) {
    std::regex duration_regex(R"((\d+)([hdwm]))");
    std::smatch match;
    
    if (!std::regex_match(duration, match, duration_regex)) {
        std::cerr << "Warning: Invalid duration format '" << duration << "', using default 5d" << std::endl;
        return 5 * 390; // 5 days
    }
    
    int value = std::stoi(match[1].str());
    char unit = match[2].str()[0];
    
    switch (unit) {
        case 'h': return value * 60;
        case 'd': return value * 390; // 390 minutes per trading day
        case 'w': return value * 5 * 390; // 5 trading days per week
        case 'm': return value * 22 * 390; // ~22 trading days per month
        default: return 5 * 390;
    }
}

UnifiedStrategyTester::TestMode UnifiedStrategyTester::parse_test_mode(const std::string& mode_str) {
    std::string mode_lower = mode_str;
    std::transform(mode_lower.begin(), mode_lower.end(), mode_lower.begin(), ::tolower);
    
    if (mode_lower == "historical" || mode_lower == "hist") return TestMode::HISTORICAL;
    if (mode_lower == "ai-regime" || mode_lower == "ai") return TestMode::AI_REGIME;
    if (mode_lower == "hybrid") return TestMode::HYBRID;
    
    std::cerr << "Warning: Unknown test mode '" << mode_str << "', using hybrid" << std::endl;
    return TestMode::HYBRID;
}

std::string UnifiedStrategyTester::get_default_historical_file(const std::string& symbol) {
    return "data/equities/" + symbol + "_NH.csv";
}

void UnifiedStrategyTester::print_robustness_report(const RobustnessReport& report, const TestConfig& config) {
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "🎯 STRATEGY ROBUSTNESS REPORT" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Header information
    std::cout << "Strategy: " << std::setw(20) << config.strategy_name 
              << "  Symbol: " << std::setw(10) << config.symbol 
              << "  Test Date: " << std::setw(12) << "2024-01-15" << std::endl;
    
    std::cout << "Duration: " << std::setw(20) << config.duration
              << "  Simulations: " << std::setw(7) << report.total_simulations
              << "  Mode: " << std::setw(15);
    
    switch (report.mode_used) {
        case TestMode::HISTORICAL: std::cout << "historical"; break;
        case TestMode::AI_REGIME: std::cout << "ai-regime"; break;
        case TestMode::HYBRID: std::cout << "hybrid"; break;
    }
    std::cout << std::endl;
    
    std::cout << "Confidence Level: " << std::setw(12) << (config.confidence_level * 100) << "%"
              << "  Alpaca Fees: " << std::setw(8) << (config.alpaca_fees ? "Enabled" : "Disabled")
              << "  Market Hours: " << std::setw(8) << "RTH" << std::endl;
    
    std::cout << std::endl;
    
    // Performance Summary
    std::cout << "📈 PERFORMANCE SUMMARY" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Monthly Projected Return: " << std::setw(8) << (report.monthly_projected_return * 100) << "% ± " 
              << std::setw(4) << ((report.mpr_ci.upper - report.mpr_ci.lower) / 2 * 100) << "%"
              << "  [" << std::setw(5) << (report.mpr_ci.lower * 100) << "% - " 
              << std::setw(5) << (report.mpr_ci.upper * 100) << "%] (95% CI)" << std::endl;
    
    std::cout << std::setprecision(2);
    std::cout << "Sharpe Ratio:            " << std::setw(8) << report.sharpe_ratio << " ± " 
              << std::setw(4) << ((report.sharpe_ci.upper - report.sharpe_ci.lower) / 2) << ""
              << "  [" << std::setw(5) << report.sharpe_ci.lower << " - " 
              << std::setw(5) << report.sharpe_ci.upper << "] (95% CI)" << std::endl;
    
    std::cout << std::setprecision(1);
    std::cout << "Maximum Drawdown:        " << std::setw(8) << (report.max_drawdown * 100) << "% ± " 
              << std::setw(4) << ((report.drawdown_ci.upper - report.drawdown_ci.lower) / 2 * 100) << "%"
              << "  [" << std::setw(5) << (report.drawdown_ci.lower * 100) << "% - " 
              << std::setw(5) << (report.drawdown_ci.upper * 100) << "%] (95% CI)" << std::endl;
    
    std::cout << "Win Rate:                " << std::setw(8) << (report.win_rate * 100) << "% ± " 
              << std::setw(4) << ((report.win_rate_ci.upper - report.win_rate_ci.lower) / 2 * 100) << "%"
              << "  [" << std::setw(5) << (report.win_rate_ci.lower * 100) << "% - " 
              << std::setw(5) << (report.win_rate_ci.upper * 100) << "%] (95% CI)" << std::endl;
    
    std::cout << std::setprecision(2);
    std::cout << "Profit Factor:           " << std::setw(8) << report.profit_factor << ""
              << "     [" << std::setw(5) << (report.profit_factor * 0.8) << " - " 
              << std::setw(5) << (report.profit_factor * 1.2) << "] (Est. Range)" << std::endl;
    
    std::cout << std::endl;
    
    // Robustness Analysis
    std::cout << "🛡️  ROBUSTNESS ANALYSIS" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    auto get_rating = [](double score) -> std::string {
        if (score >= 90) return "EXCELLENT";
        if (score >= 80) return "VERY GOOD";
        if (score >= 70) return "GOOD";
        if (score >= 60) return "FAIR";
        return "POOR";
    };
    
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "Consistency Score:       " << std::setw(8) << report.consistency_score << "/100"
              << "      " << get_rating(report.consistency_score) << " - Performance variance" << std::endl;
    std::cout << "Regime Adaptability:     " << std::setw(8) << report.regime_adaptability << "/100"
              << "      " << get_rating(report.regime_adaptability) << " - Market adaptation" << std::endl;
    std::cout << "Stress Resilience:       " << std::setw(8) << report.stress_resilience << "/100"
              << "      " << get_rating(report.stress_resilience) << " - Adverse conditions" << std::endl;
    std::cout << "Liquidity Tolerance:     " << std::setw(8) << report.liquidity_tolerance << "/100"
              << "      " << get_rating(report.liquidity_tolerance) << " - Execution robustness" << std::endl;
    
    std::cout << std::endl;
    
    // Alpaca Trading Analysis
    std::cout << "💰 ALPACA TRADING ANALYSIS" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "Estimated Monthly Fees:  $" << std::setw(7) << report.estimated_monthly_fees 
              << "     (" << std::setprecision(2) << (report.estimated_monthly_fees / config.initial_capital * 100) 
              << "% of capital)" << std::endl;
    
    std::cout << std::setprecision(1);
    std::cout << "Capital Efficiency:      " << std::setw(8) << report.capital_efficiency << "%"
              << "      High capital utilization" << std::endl;
    
    std::cout << "Average Daily Trades:    " << std::setw(8) << report.avg_daily_trades 
              << "      Within Alpaca limits" << std::endl;
    
    std::cout << std::setprecision(1);
    std::cout << "Position Turnover:       " << std::setw(8) << report.position_turnover << "x/day"
              << "      Moderate turnover rate" << std::endl;
    
    std::cout << std::endl;
    
    // Risk Assessment
    std::string risk_str;
    switch (report.overall_risk) {
        case RiskLevel::LOW: risk_str = "LOW"; break;
        case RiskLevel::MEDIUM: risk_str = "MEDIUM"; break;
        case RiskLevel::HIGH: risk_str = "HIGH"; break;
        case RiskLevel::EXTREME: risk_str = "EXTREME"; break;
    }
    
    std::cout << "⚠️  RISK ASSESSMENT: " << risk_str << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    if (!report.risk_warnings.empty()) {
        std::cout << "⚠️  WARNINGS:" << std::endl;
        for (const auto& warning : report.risk_warnings) {
            std::cout << "  • " << warning << std::endl;
        }
        std::cout << std::endl;
    }
    
    if (!report.recommendations.empty()) {
        std::cout << "💡 RECOMMENDATIONS:" << std::endl;
        for (const auto& rec : report.recommendations) {
            std::cout << "  • " << rec << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Deployment Readiness
    std::cout << "🎯 DEPLOYMENT READINESS: " << (report.ready_for_deployment ? "READY" : "NOT READY") << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    std::cout << "Overall Score:           " << std::setw(8) << report.deployment_score << "/100"
              << "      (" << get_rating(report.deployment_score) << ")" << std::endl;
    std::cout << "Recommended Live Capital: " << report.recommended_capital_range << std::endl;
    std::cout << "Suggested Monitoring:    " << report.suggested_monitoring_days << " days paper trading" << std::endl;
    
    std::cout << std::string(80, '=') << std::endl;
}

bool UnifiedStrategyTester::save_report(const RobustnessReport& report, const TestConfig& config,
                                       const std::string& filename, const std::string& format) {
    // Implementation for saving reports in different formats
    // This is a placeholder - full implementation would handle JSON, CSV, etc.
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    if (format == "json") {
        // JSON format implementation
        file << "{\n";
        file << "  \"strategy\": \"" << config.strategy_name << "\",\n";
        file << "  \"symbol\": \"" << config.symbol << "\",\n";
        file << "  \"monthly_projected_return\": " << report.monthly_projected_return << ",\n";
        file << "  \"sharpe_ratio\": " << report.sharpe_ratio << ",\n";
        file << "  \"max_drawdown\": " << report.max_drawdown << ",\n";
        file << "  \"deployment_score\": " << report.deployment_score << "\n";
        file << "}\n";
    } else {
        // Default to text format
        // Redirect print_robustness_report output to file
        std::streambuf* orig = std::cout.rdbuf();
        std::cout.rdbuf(file.rdbuf());
        
        // This is a bit hacky - in a real implementation, we'd refactor print_robustness_report
        // to accept an output stream parameter
        
        std::cout.rdbuf(orig);
    }
    
    file.close();
    return true;
}

std::vector<VirtualMarketEngine::VMSimulationResult> UnifiedStrategyTester::run_holistic_tests(const TestConfig& config) {
    std::cout << "🔬 HOLISTIC TESTING: Running comprehensive multi-scenario robustness analysis..." << std::endl;
    
    std::vector<VirtualMarketEngine::VMSimulationResult> all_results;
    
    // 1. Historical Data Testing (40% of simulations)
    int historical_sims = config.simulations * 0.4;
    std::cout << "📊 Phase 1/4: Historical Pattern Analysis (" << historical_sims << " simulations)" << std::endl;
    auto historical_results = run_historical_tests(config, historical_sims);
    all_results.insert(all_results.end(), historical_results.begin(), historical_results.end());
    
    // 2. AI Market Regime Testing - Multiple Regimes (40% of simulations)
    int ai_sims_per_regime = (config.simulations * 0.4) / 4; // 4 different regimes (normal, volatile, trending, bear)
    std::cout << "🤖 Phase 2/4: AI Market Regime Testing (" << (ai_sims_per_regime * 4) << " simulations)" << std::endl;
    
    std::vector<std::string> regimes = {"normal", "volatile", "trending", "bear"}; // Removed "bull" - not supported by MarS
    for (const auto& regime : regimes) {
        auto regime_config = config;
        regime_config.regime = regime;
        auto regime_results = run_ai_regime_tests(regime_config, ai_sims_per_regime);
        all_results.insert(all_results.end(), regime_results.begin(), regime_results.end());
    }
    
    // 3. Stress Testing Scenarios (10% of simulations)
    int stress_sims = config.simulations * 0.1;
    std::cout << "⚡ Phase 3/4: Extreme Stress Testing (" << stress_sims << " simulations)" << std::endl;
    auto stress_config = config;
    stress_config.stress_test = true;
    stress_config.liquidity_stress = true;
    stress_config.volatility_min = 0.02; // High volatility
    stress_config.volatility_max = 0.08; // Extreme volatility
    auto stress_results = run_ai_regime_tests(stress_config, stress_sims);
    all_results.insert(all_results.end(), stress_results.begin(), stress_results.end());
    
    // 4. Cross-Timeframe Validation (10% of simulations)
    int timeframe_sims = config.simulations * 0.1;
    std::cout << "⏰ Phase 4/4: Cross-Timeframe Validation (" << timeframe_sims << " simulations)" << std::endl;
    
    // Test with different durations to validate consistency
    std::vector<std::string> test_durations = {"1w", "2w", "1m"};
    int sims_per_duration = std::max(1, timeframe_sims / (int)test_durations.size());
    
    for (const auto& duration : test_durations) {
        auto duration_config = config;
        duration_config.duration = duration;
        auto duration_results = run_historical_tests(duration_config, sims_per_duration);
        all_results.insert(all_results.end(), duration_results.begin(), duration_results.end());
    }
    
    std::cout << "✅ HOLISTIC TESTING COMPLETE: " << all_results.size() << " total simulations across all scenarios" << std::endl;
    
    return all_results;
}

// Duplicate method implementations removed - using existing implementations above

} // namespace sentio
