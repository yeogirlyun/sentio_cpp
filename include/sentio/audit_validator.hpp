#pragma once
#include "base_strategy.hpp"
#include "runner.hpp"
#include "audit.hpp"
#include <string>
#include <vector>

namespace sentio {

/**
 * **STRATEGY-AGNOSTIC AUDIT VALIDATOR**
 * 
 * Validates that any strategy inheriting from BaseStrategy works properly
 * with the audit system without strategy-specific dependencies.
 */
class AuditValidator {
public:
    struct ValidationResult {
        bool success = false;
        std::string strategy_name;
        std::string error_message;
        int signals_logged = 0;
        int orders_logged = 0;
        int fills_logged = 0;
        double test_duration_sec = 0.0;
    };
    
    /**
     * Validate that a strategy works with the audit system
     * @param strategy_name Name of the strategy to test
     * @param test_bars Number of bars to test with (default: 100)
     * @return ValidationResult with success status and metrics
     */
    static ValidationResult validate_strategy_audit_compatibility(
        const std::string& strategy_name,
        int test_bars = 100
    );
    
    /**
     * Validate all registered strategies
     * @param test_bars Number of bars to test each strategy with
     * @return Vector of validation results for all strategies
     */
    static std::vector<ValidationResult> validate_all_strategies(
        int test_bars = 100
    );
    
    /**
     * Print validation report
     * @param results Vector of validation results
     */
    static void print_validation_report(const std::vector<ValidationResult>& results);

private:
    /**
     * Generate synthetic test data for validation
     * @param num_bars Number of bars to generate
     * @return Vector of synthetic bars
     */
    static std::vector<Bar> generate_test_data(int num_bars);
    
    /**
     * Create a minimal RunnerCfg for testing
     * @param strategy_name Name of the strategy
     * @return RunnerCfg configured for audit testing
     */
    static RunnerCfg create_test_config(const std::string& strategy_name);
};

} // namespace sentio
