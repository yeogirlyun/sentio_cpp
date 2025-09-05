#include "sentio/optimizer.hpp"
#include "sentio/runner.hpp"
#include <iostream>

namespace sentio {

// RandomSearchOptimizer Implementation
std::vector<Parameter> RandomSearchOptimizer::get_parameter_space() {
    return {
        Parameter("test_param", 0.0, 1.0, 0.5)
    };
}

void RandomSearchOptimizer::apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) {
    for (const auto& param : params) {
        cfg.strategy_params[param.name] = std::to_string(param.current_value);
    }
}

OptimizationResult RandomSearchOptimizer::optimize([[maybe_unused]] const std::function<double(const RunResult&)>& objective_func,
                                                  [[maybe_unused]] const std::vector<Parameter>& param_space,
                                                  [[maybe_unused]] const OptimizationConfig& config) {
    OptimizationResult result;
    result.parameters["test"] = 0.5;
    result.objective_value = 0.0;
    return result;
}

// GridSearchOptimizer Implementation
std::vector<Parameter> GridSearchOptimizer::get_parameter_space() {
    return {
        Parameter("test_param", 0.0, 1.0, 0.5)
    };
}

void GridSearchOptimizer::apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) {
    for (const auto& param : params) {
        cfg.strategy_params[param.name] = std::to_string(param.current_value);
    }
}

OptimizationResult GridSearchOptimizer::optimize([[maybe_unused]] const std::function<double(const RunResult&)>& objective_func,
                                                [[maybe_unused]] const std::vector<Parameter>& param_space,
                                                [[maybe_unused]] const OptimizationConfig& config) {
    OptimizationResult result;
    result.parameters["test"] = 0.5;
    result.objective_value = 0.0;
    return result;
}

// BayesianOptimizer Implementation
std::vector<Parameter> BayesianOptimizer::get_parameter_space() {
    return {
        Parameter("test_param", 0.0, 1.0, 0.5)
    };
}

void BayesianOptimizer::apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) {
    for (const auto& param : params) {
        cfg.strategy_params[param.name] = std::to_string(param.current_value);
    }
}

OptimizationResult BayesianOptimizer::optimize([[maybe_unused]] const std::function<double(const RunResult&)>& objective_func,
                                              [[maybe_unused]] const std::vector<Parameter>& param_space,
                                              [[maybe_unused]] const OptimizationConfig& config) {
    OptimizationResult result;
    result.parameters["test"] = 0.5;
    result.objective_value = 0.0;
    return result;
}

// Strategy parameter creation functions
std::vector<Parameter> create_vwap_parameters() {
    return {
        Parameter("vwap_period", 100.0, 800.0, 200.0),
        Parameter("reversion_threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_momentum_parameters() {
    return {
        Parameter("momentum_period", 5.0, 50.0, 20.0),
        Parameter("threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_volatility_parameters() {
    return {
        Parameter("volatility_period", 10.0, 50.0, 20.0),
        Parameter("threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_bollinger_squeeze_parameters() {
    return {
        Parameter("bb_period", 10.0, 40.0, 20.0),
        Parameter("bb_std", 1.0, 3.0, 2.0)
    };
}

std::vector<Parameter> create_opening_range_parameters() {
    return {
        Parameter("range_minutes", 15.0, 60.0, 30.0),
        Parameter("breakout_threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_order_flow_scalping_parameters() {
    return {
        Parameter("imbalance_period", 5.0, 40.0, 20.0),
        Parameter("threshold", 0.4, 0.9, 0.7)
    };
}

std::vector<Parameter> create_order_flow_imbalance_parameters() {
    return {
        Parameter("lookback_window", 5.0, 50.0, 20.0),
        Parameter("threshold", 0.5, 3.0, 1.5)
    };
}

std::vector<Parameter> create_market_making_parameters() {
    return {
        Parameter("base_spread", 0.0002, 0.003, 0.001),
        Parameter("order_levels", 1.0, 5.0, 3.0)
    };
}

std::vector<Parameter> create_router_parameters() {
    return {
        Parameter("t1", 0.01, 0.3, 0.05),
        Parameter("t2", 0.1, 0.6, 0.3)
    };
}

std::vector<Parameter> create_parameters_for_strategy(const std::string& strategy_name) {
    if (strategy_name == "VWAPReversion") {
        return create_vwap_parameters();
    } else if (strategy_name == "MomentumVolumeProfile") {
        return create_momentum_parameters();
    } else if (strategy_name == "VolatilityExpansion") {
        return create_volatility_parameters();
    } else if (strategy_name == "BollingerSqueezeBreakout") {
        return create_bollinger_squeeze_parameters();
    } else if (strategy_name == "OpeningRangeBreakout") {
        return create_opening_range_parameters();
    } else if (strategy_name == "OrderFlowScalping") {
        return create_order_flow_scalping_parameters();
    } else if (strategy_name == "OrderFlowImbalance") {
        return create_order_flow_imbalance_parameters();
    } else if (strategy_name == "MarketMaking") {
        return create_market_making_parameters();
    } else {
        return create_router_parameters();
    }
}

std::vector<Parameter> create_full_parameter_space() {
    return create_router_parameters();
}

// OptimizationEngine implementation
OptimizationEngine::OptimizationEngine(const std::string& optimizer_type) {
    if (optimizer_type == "grid") {
        optimizer = std::make_unique<GridSearchOptimizer>();
    } else if (optimizer_type == "bayesian") {
        optimizer = std::make_unique<BayesianOptimizer>();
    } else {
        optimizer = std::make_unique<RandomSearchOptimizer>();
    }
}

OptimizationResult OptimizationEngine::run_optimization(const std::string& strategy_name,
                                                       const std::function<double(const RunResult&)>& objective_func) {
    auto param_space = create_parameters_for_strategy(strategy_name);
    return optimizer->optimize(objective_func, param_space, config);
}

} // namespace sentio
