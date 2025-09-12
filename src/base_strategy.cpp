#include "sentio/base_strategy.hpp"
#include <iostream>

namespace sentio {

void BaseStrategy::reset_state() {
    state_.reset();
    diag_ = SignalDiag{};
}

bool BaseStrategy::is_cooldown_active(int current_bar, int cooldown_period) const {
    return (current_bar - state_.last_trade_bar) < cooldown_period;
}

// **MODIFIED**: This function now merges incoming parameters (overrides)
// with the existing default parameters, preventing the defaults from being erased.
void BaseStrategy::set_params(const ParameterMap& overrides) {
    // The constructor has already set the defaults.
    // Now, merge the overrides into the existing params_.
    for (const auto& [key, value] : overrides) {
        params_[key] = value;
    }
    
    apply_params();
}

// --- Strategy Factory Implementation ---
StrategyFactory& StrategyFactory::instance() {
    static StrategyFactory factory_instance;
    return factory_instance;
}

void StrategyFactory::register_strategy(const std::string& name, CreateFunction create_func) {
    strategies_[name] = create_func;
}

std::unique_ptr<BaseStrategy> StrategyFactory::create_strategy(const std::string& name) {
    auto it = strategies_.find(name);
    if (it != strategies_.end()) {
        return it->second();
    }
    std::cerr << "Error: Strategy '" << name << "' not found in factory." << std::endl;
    return nullptr;
}

std::vector<std::string> StrategyFactory::get_available_strategies() const {
    std::vector<std::string> names;
    for (const auto& pair : strategies_) {
        names.push_back(pair.first);
    }
    return names;
}

} // namespace sentio