#pragma once

#include "sentio/core.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/runner.hpp"
#include "sentio/virtual_market.hpp"
#include "sentio/mars_data_loader.hpp"
#include "sentio/unified_metrics.hpp"
#include <vector>
#include <string>
#include <memory>
#include <map>

namespace sentio {

/**
 * Unified Strategy Testing Framework
 * 
 * Consolidates vmtest, marstest, and fasttest into a single comprehensive
 * strategy robustness evaluation system focused on Alpaca live trading readiness.
 */
class UnifiedStrategyTester {
public:
    enum class TestMode {
        MONTE_CARLO,    // Pure synthetic data with regime switching
        HISTORICAL,     // Historical patterns with synthetic continuation
        AI_REGIME,      // MarS AI-powered market simulation
        HYBRID          // Combines all three approaches (default)
    };
    
    enum class RiskLevel {
        LOW,
        MEDIUM, 
        HIGH,
        EXTREME
    };
    
    struct ConfidenceInterval {
        double lower;
        double upper;
        double mean;
        double std_dev;
    };
    
    struct TestConfig {
        // Core Parameters
        std::string strategy_name;
        std::string symbol;
        TestMode mode = TestMode::HYBRID;
        int simulations = 50;
        std::string duration = "5d";
        
        // Data Sources
        std::string historical_data_file;
        std::string regime = "normal";
        bool market_hours_only = true;
        
        // Dataset Information for Reporting
        std::string dataset_source = "unknown";
        std::string dataset_period = "unknown";
        std::string test_period = "unknown";
        
        // Robustness Testing
        bool stress_test = false;
        bool regime_switching = false;
        bool liquidity_stress = false;
        double volatility_min = 0.0;
        double volatility_max = 0.0;
        
        // Alpaca Integration
        bool alpaca_fees = true;
        bool alpaca_limits = false;
        bool paper_validation = false;
        
        // Output & Analysis
        double confidence_level = 0.95;
        std::string output_format = "console";
        std::string save_results_file;
        std::string benchmark_symbol = "SPY";
        
        // Performance
        int parallel_jobs = 0; // 0 = auto-detect
        bool quick_mode = false;
        bool comprehensive_mode = false;
        bool holistic_mode = false;
        bool disable_audit_logging = false;
        
        // Strategy Parameters
        std::string params_json = "{}";
        double initial_capital = 100000.0;
    };
    
    struct RobustnessReport {
        // Core Metrics (Alpaca-focused)
        double monthly_projected_return;
        double sharpe_ratio;
        double max_drawdown;
        double win_rate;
        double profit_factor;
        double total_return;
        
        // Robustness Metrics
        double consistency_score;      // Performance consistency across simulations
        double regime_adaptability;    // Performance across different market regimes
        double stress_resilience;      // Performance under stress conditions
        double liquidity_tolerance;    // Performance with liquidity constraints
        
        // Alpaca-Specific Metrics
        double estimated_monthly_fees;
        double capital_efficiency;     // Return per dollar of capital used
        double avg_daily_trades;
        double position_turnover;
        
        // Risk Assessment
        RiskLevel overall_risk;
        std::vector<std::string> risk_warnings;
        std::vector<std::string> recommendations;
        
        // Confidence Intervals
        ConfidenceInterval mpr_ci;
        ConfidenceInterval sharpe_ci;
        ConfidenceInterval drawdown_ci;
        ConfidenceInterval win_rate_ci;
        
        // Simulation Details
        int total_simulations;
        int successful_simulations;
        double test_duration_seconds;
        TestMode mode_used;
        
        // Deployment Readiness
        int deployment_score;          // 0-100 overall readiness score
        bool ready_for_deployment;
        std::string recommended_capital_range;
        int suggested_monitoring_days;
        
        // Audit Information
        std::string run_id;            // Run ID for audit verification
        
        // Dataset Information
        std::string dataset_source = "unknown";
        std::string dataset_period = "unknown";
        std::string test_period = "unknown";
    };
    
    UnifiedStrategyTester();
    ~UnifiedStrategyTester() = default;
    
    /**
     * Run comprehensive strategy robustness test
     */
    RobustnessReport run_comprehensive_test(const TestConfig& config);
    
    /**
     * Print detailed robustness report to console
     */
    void print_robustness_report(const RobustnessReport& report, const TestConfig& config);
    
    /**
     * Save report to file in specified format
     */
    bool save_report(const RobustnessReport& report, const TestConfig& config, 
                     const std::string& filename, const std::string& format);
    
    /**
     * Parse duration string (e.g., "1h", "5d", "2w", "1m")
     */
    static int parse_duration_to_minutes(const std::string& duration);
    
    /**
     * Parse test mode from string
     */
    static TestMode parse_test_mode(const std::string& mode_str);
    
    /**
     * Get default historical data file for symbol
     */
    static std::string get_default_historical_file(const std::string& symbol);
    
    /**
     * Get dataset period from data file
     */
    static std::string get_dataset_period(const std::string& file_path);
    
    /**
     * Get test period from data file
     */
    static std::string get_test_period(const std::string& file_path, int continuation_minutes);

private:
    // Simulation Engines
    VirtualMarketEngine vm_engine_;
    
    // Analysis Components
    // UnifiedMetricsCalculator metrics_calculator_; // TODO: Implement when needed
    
    /**
     * Run Monte Carlo simulations
     */
    std::vector<VirtualMarketEngine::VMSimulationResult> run_monte_carlo_tests(
        const TestConfig& config, int num_simulations);
    
    /**
     * Run historical pattern tests
     */
    std::vector<VirtualMarketEngine::VMSimulationResult> run_historical_tests(
        TestConfig& config, int num_simulations);
    
    /**
     * Run AI regime tests
     */
    std::vector<VirtualMarketEngine::VMSimulationResult> run_ai_regime_tests(
        TestConfig& config, int num_simulations);
    
    /**
     * Run hybrid tests (combination of all modes)
     */
    std::vector<VirtualMarketEngine::VMSimulationResult> run_hybrid_tests(
        TestConfig& config);
    
    /**
     * Run holistic tests (comprehensive multi-scenario testing)
     */
    std::vector<VirtualMarketEngine::VMSimulationResult> run_holistic_tests(
        TestConfig& config);
    
    /**
     * Analyze simulation results for robustness metrics
     */
    RobustnessReport analyze_results(
        const std::vector<VirtualMarketEngine::VMSimulationResult>& results,
        const TestConfig& config);
    
    /**
     * Calculate confidence intervals for metrics
     */
    ConfidenceInterval calculate_confidence_interval(
        const std::vector<double>& values, double confidence_level);
    
    /**
     * Assess overall risk level
     */
    RiskLevel assess_risk_level(const RobustnessReport& report);
    
    /**
     * Generate risk warnings and recommendations
     */
    void generate_recommendations(RobustnessReport& report, const TestConfig& config);
    
    /**
     * Calculate deployment readiness score
     */
    int calculate_deployment_score(const RobustnessReport& report);
    
    /**
     * Apply stress testing scenarios
     */
    void apply_stress_scenarios(std::vector<VirtualMarketEngine::VMSimulationResult>& results,
                               const TestConfig& config);
};

} // namespace sentio
