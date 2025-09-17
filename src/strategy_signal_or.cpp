#include "sentio/strategy_signal_or.hpp"
#include "sentio/signal_utils.hpp"
#include "sentio/router.hpp"
#include "sentio/sizer.hpp"
#include "sentio/allocation_manager.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace sentio {

SignalOrStrategy::SignalOrStrategy(const SignalOrCfg& cfg) 
    : BaseStrategy("SignalOR"), cfg_(cfg) {
    // **PROFIT MAXIMIZATION**: Override OR config for more aggressive signals
    cfg_.or_config.aggression = 0.95;      // Maximum aggression for stronger signals
    cfg_.or_config.min_conf = 0.01;       // Lower threshold to capture weak signals
    cfg_.or_config.conflict_soften = 0.2; // Less softening to preserve strong signals
    
    // **MATHEMATICAL ALLOCATION MANAGER**: Initialize with Signal OR tuned parameters
    AllocationConfig alloc_config;
    alloc_config.entry_threshold_1x = cfg_.long_threshold - 0.05;  // Slightly lower for 1x
    alloc_config.entry_threshold_3x = cfg_.long_threshold + 0.15;  // Higher for 3x leverage
    alloc_config.partial_exit_threshold = 0.5 - (cfg_.long_threshold - 0.5) * 0.5; // Dynamic
    alloc_config.full_exit_threshold = 0.5 - (cfg_.long_threshold - 0.5) * 0.8;    // More aggressive
    alloc_config.min_signal_change = cfg_.min_signal_strength;     // Align with strategy config
    
    allocation_manager_ = std::make_unique<AllocationManager>(alloc_config);
    
    apply_params();
}

// Required BaseStrategy methods
double SignalOrStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < 0 || current_index >= static_cast<int>(bars.size())) {
        return 0.5; // Neutral if invalid index
    }
    
    warmup_bars_++;
    
    // Evaluate simple rules and apply Signal-OR mixing
    auto rule_outputs = evaluate_simple_rules(bars, current_index);
    
    if (rule_outputs.empty()) {
        return 0.5; // Neutral if no rules active
    }
    
    // Apply Signal-OR mixing
    double probability = mix_signal_or(rule_outputs, cfg_.or_config);
    
    // **FIXED**: Update signal diagnostics counter
    diag_.emitted++;
    
    return probability;
}

std::vector<SignalOrStrategy::AllocationDecision> SignalOrStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    if (current_index < 0 || current_index >= static_cast<int>(bars.size())) {
        return decisions; // Empty if invalid index
    }
    
    double probability = calculate_probability(bars, current_index);
    
    // **STATE-AWARE TRANSITION ALGORITHM**
    // Determine target position based on signal strength
    std::string target_instrument = "";
    double target_weight = 0.0;
    std::string reason = "";
    
    if (probability > 0.7) {
        // Strong buy: 100% TQQQ (3x leveraged long)
        target_instrument = bull3x_symbol;
        target_weight = 1.0;
        reason = "Signal OR strong buy: 100% TQQQ (3x leverage)";
        
    } else if (probability > cfg_.long_threshold) {
        // Moderate buy: 100% QQQ (1x long)
        target_instrument = base_symbol;
        target_weight = 1.0;
        reason = "Signal OR moderate buy: 100% QQQ";
        
    } else if (probability < 0.3) {
        // Strong sell: 100% SQQQ (3x leveraged short)
        target_instrument = bear3x_symbol;
        target_weight = 1.0;
        reason = "Signal OR strong sell: 100% SQQQ (3x inverse)";
        
    } else if (probability < cfg_.short_threshold) {
        // Weak sell: 100% PSQ (1x inverse)
        target_instrument = "PSQ";
        target_weight = 1.0;
        reason = "Signal OR weak sell: 100% PSQ (1x inverse)";
        
    } else {
        // Neutral: Stay in cash
        target_instrument = "CASH";
        target_weight = 0.0;
        reason = "Signal OR neutral: Stay in cash";
    }
    
    // **TEMPORARY SIMPLE ALLOCATION**: Return target if different from last bar
    bool different_bar = (current_index != last_decision_bar_);
    
    if (different_bar && target_instrument != "CASH") {
        // Return only the target allocation - runner will handle atomic rebalancing
        decisions.push_back({target_instrument, target_weight, probability, reason});
        last_decision_bar_ = current_index;
    }
    // If target is CASH or same bar, return empty decisions (no action needed)
    
    return decisions;
}

RouterCfg SignalOrStrategy::get_router_config() const {
    RouterCfg cfg;
    
    // **PROFIT MAXIMIZATION**: Configure router for maximum leverage and 100% capital deployment
    cfg.min_signal_strength = 0.01;    // Lower threshold to capture more signals
    cfg.signal_multiplier = 1.0;       // No scaling
    cfg.max_position_pct = 1.0;        // 100% position size (profit maximization)
    cfg.require_rth = true;
    
    // Instrument configuration
    cfg.base_symbol = "QQQ";
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: PSQ will be handled via SHORT QQQ for moderate sell signals
    
    cfg.min_shares = 1.0;
    cfg.lot_size = 1.0;
    cfg.ire_min_conf_strong_short = 0.85;
    
    return cfg;
}

// REMOVED: get_sizer_config() - No artificial limits allowed for profit maximization
// Sizer will use profit-maximizing defaults: 100% capital deployment, maximum leverage

// Configuration
ParameterMap SignalOrStrategy::get_default_params() const {
    return {
        {"min_signal_strength", cfg_.min_signal_strength},
        {"long_threshold", cfg_.long_threshold},
        {"short_threshold", cfg_.short_threshold},
        {"hold_threshold", cfg_.hold_threshold},
        {"momentum_window", static_cast<double>(cfg_.momentum_window)},
        {"momentum_scale", cfg_.momentum_scale},
        {"or_aggression", cfg_.or_config.aggression},
        {"or_min_conf", cfg_.or_config.min_conf},
        {"or_conflict_soften", cfg_.or_config.conflict_soften}
    };
}

ParameterSpace SignalOrStrategy::get_param_space() const {
    ParameterSpace space;
    space["min_signal_strength"] = {ParamType::FLOAT, 0.05, 0.3, cfg_.min_signal_strength};
    space["long_threshold"] = {ParamType::FLOAT, 0.55, 0.75, cfg_.long_threshold};
    space["short_threshold"] = {ParamType::FLOAT, 0.25, 0.45, cfg_.short_threshold};
    space["momentum_window"] = {ParamType::INT, 10, 50, static_cast<double>(cfg_.momentum_window)};
    space["momentum_scale"] = {ParamType::FLOAT, 10.0, 50.0, cfg_.momentum_scale};
    space["or_aggression"] = {ParamType::FLOAT, 0.6, 0.95, cfg_.or_config.aggression};
    space["or_min_conf"] = {ParamType::FLOAT, 0.01, 0.2, cfg_.or_config.min_conf};
    space["or_conflict_soften"] = {ParamType::FLOAT, 0.2, 0.6, cfg_.or_config.conflict_soften};
    return space;
}

void SignalOrStrategy::apply_params() {
    // Apply parameters from the parameter map
    if (params_.count("min_signal_strength")) {
        cfg_.min_signal_strength = params_.at("min_signal_strength");
    }
    if (params_.count("long_threshold")) {
        cfg_.long_threshold = params_.at("long_threshold");
    }
    if (params_.count("short_threshold")) {
        cfg_.short_threshold = params_.at("short_threshold");
    }
    if (params_.count("hold_threshold")) {
        cfg_.hold_threshold = params_.at("hold_threshold");
    }
    if (params_.count("momentum_window")) {
        cfg_.momentum_window = static_cast<int>(params_.at("momentum_window"));
    }
    if (params_.count("momentum_scale")) {
        cfg_.momentum_scale = params_.at("momentum_scale");
    }
    if (params_.count("or_aggression")) {
        cfg_.or_config.aggression = params_.at("or_aggression");
    }
    if (params_.count("or_min_conf")) {
        cfg_.or_config.min_conf = params_.at("or_min_conf");
    }
    if (params_.count("or_conflict_soften")) {
        cfg_.or_config.conflict_soften = params_.at("or_conflict_soften");
    }
    
    // Reset state
    warmup_bars_ = 0;
}

// Helper methods
std::vector<RuleOut> SignalOrStrategy::evaluate_simple_rules(const std::vector<Bar>& bars, int current_index) {
    std::vector<RuleOut> outputs;
    
    // Rule 1: Momentum-based probability
    double momentum_prob = calculate_momentum_probability(bars, current_index);
    RuleOut momentum_out;
    momentum_out.p01 = momentum_prob;
    momentum_out.conf01 = std::abs(momentum_prob - 0.5) * 2.0; // Confidence based on deviation from neutral
    outputs.push_back(momentum_out);
    
    // Rule 2: Volume-based probability (if we have volume data)
    if (current_index > 0 && bars[current_index].volume > 0 && bars[current_index - 1].volume > 0) {
        double volume_ratio = static_cast<double>(bars[current_index].volume) / bars[current_index - 1].volume;
        double volume_prob = 0.5 + std::clamp((volume_ratio - 1.0) * 0.1, -0.2, 0.2); // Volume momentum
        RuleOut volume_out;
        volume_out.p01 = volume_prob;
        volume_out.conf01 = std::min(0.5, std::abs(volume_ratio - 1.0) * 0.5); // Confidence based on volume change
        outputs.push_back(volume_out);
    }
    
    // Rule 3: Price volatility-based probability
    if (current_index >= 5) {
        double volatility = 0.0;
        for (int i = current_index - 4; i <= current_index; ++i) {
            double ret = (bars[i].close - bars[i-1].close) / bars[i-1].close;
            volatility += ret * ret;
        }
        volatility = std::sqrt(volatility / 5.0);
        
        // Higher volatility suggests trend continuation
        double vol_prob = 0.5 + std::clamp(volatility * 10.0, -0.2, 0.2);
        RuleOut vol_out;
        vol_out.p01 = vol_prob;
        vol_out.conf01 = std::min(0.3, volatility * 5.0); // Confidence based on volatility
        outputs.push_back(vol_out);
    }
    
    return outputs;
}

double SignalOrStrategy::calculate_momentum_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < cfg_.momentum_window) {
        return 0.5; // Neutral if not enough data
    }
    
    // Calculate moving average
    double ma = 0.0;
    for (int i = current_index - cfg_.momentum_window + 1; i <= current_index; ++i) {
        ma += bars[i].close;
    }
    ma /= cfg_.momentum_window;
    
    // Calculate momentum
    double momentum = (bars[current_index].close - ma) / ma;
    
    // **PROFIT MAXIMIZATION**: Allow extreme probabilities for leverage triggers
    double momentum_prob = 0.5 + std::clamp(momentum * cfg_.momentum_scale, -0.45, 0.45);
    
    return momentum_prob;
}

// **PROFIT MAXIMIZATION**: Old position weight calculation removed
// Now using 100% capital deployment with maximum leverage

} // namespace sentio