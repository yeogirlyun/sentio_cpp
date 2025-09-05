#pragma once
#include "sentio/core.hpp"
#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <random>

namespace sentio {

struct Parameter {
    std::string name;
    double min_value;
    double max_value;
    double current_value;
    
    Parameter(const std::string& n, double min_val, double max_val, double current_val)
        : name(n), min_value(min_val), max_value(max_val), current_value(current_val) {}
    
    Parameter(const std::string& n, double min_val, double max_val)
        : name(n), min_value(min_val), max_value(max_val), current_value(min_val) {}
};

struct OptimizationResult {
    std::unordered_map<std::string, double> parameters;
    RunResult metrics;
    double objective_value;
};

struct OptimizationConfig {
    std::string optimizer_type = "random";
    int max_trials = 30;
    std::string objective = "sharpe_ratio";
    double timeout_minutes = 15.0;
    bool verbose = true;
};

class ParameterOptimizer {
public:
    virtual ~ParameterOptimizer() = default;
    virtual std::vector<Parameter> get_parameter_space() = 0;
    virtual void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) = 0;
    virtual OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                                      const std::vector<Parameter>& param_space,
                                      const OptimizationConfig& config) = 0;
};

class RandomSearchOptimizer : public ParameterOptimizer {
private:
    std::mt19937 rng;
    
public:
    RandomSearchOptimizer(unsigned int seed = 42) : rng(seed) {}
    
    std::vector<Parameter> get_parameter_space() override;
    void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) override;
    OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                               const std::vector<Parameter>& param_space,
                               const OptimizationConfig& config) override;
};

class GridSearchOptimizer : public ParameterOptimizer {
public:
    std::vector<Parameter> get_parameter_space() override;
    void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) override;
    OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                              const std::vector<Parameter>& param_space,
                              const OptimizationConfig& config) override;
};

class BayesianOptimizer : public ParameterOptimizer {
public:
    std::vector<Parameter> get_parameter_space() override;
    void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) override;
    OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                              const std::vector<Parameter>& param_space,
                              const OptimizationConfig& config) override;
};

// Objective functions
using ObjectiveFunction = std::function<double(const RunResult&)>;

class ObjectiveFunctions {
public:
    static double sharpe_objective(const RunResult& summary) {
        return summary.sharpe_ratio;
    }
    
    static double calmar_objective(const RunResult& summary) {
        return summary.monthly_projected_return / std::max(0.01, summary.max_drawdown);
    }
    
    static double total_return_objective(const RunResult& summary) {
        return summary.total_return;
    }
    
    static double sortino_objective(const RunResult& summary) {
        // Simplified Sortino ratio (assuming no downside deviation calculation)
        return summary.sharpe_ratio;
    }
};

// Strategy-specific parameter creation functions
std::vector<Parameter> create_vwap_parameters();
std::vector<Parameter> create_momentum_parameters();
std::vector<Parameter> create_volatility_parameters();
std::vector<Parameter> create_bollinger_squeeze_parameters();
std::vector<Parameter> create_opening_range_parameters();
std::vector<Parameter> create_order_flow_scalping_parameters();
std::vector<Parameter> create_order_flow_imbalance_parameters();
std::vector<Parameter> create_market_making_parameters();
std::vector<Parameter> create_router_parameters();

std::vector<Parameter> create_parameters_for_strategy(const std::string& strategy_name);
std::vector<Parameter> create_full_parameter_space();

// Optimization utilities
class OptimizationEngine {
private:
    std::unique_ptr<ParameterOptimizer> optimizer;
    OptimizationConfig config;
    
public:
    OptimizationEngine(const std::string& optimizer_type = "random");
    
    void set_config(const OptimizationConfig& cfg) { config = cfg; }
    
    OptimizationResult run_optimization(const std::string& strategy_name,
                                      const std::function<double(const RunResult&)>& objective_func);
    
    double calculate_objective(const RunResult& summary) {
        if (config.objective == "sharpe_ratio") {
            return ObjectiveFunctions::sharpe_objective(summary);
        } else if (config.objective == "calmar_ratio") {
            return ObjectiveFunctions::calmar_objective(summary);
        } else if (config.objective == "total_return") {
            return ObjectiveFunctions::total_return_objective(summary);
        } else if (config.objective == "sortino_ratio") {
            return ObjectiveFunctions::sortino_objective(summary);
        }
        return ObjectiveFunctions::sharpe_objective(summary); // default
    }
};

} // namespace sentio