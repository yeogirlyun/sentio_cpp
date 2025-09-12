#include "sentio/strategy_signal_or.hpp"
#include "sentio/signal_utils.hpp"
#include "sentio/router.hpp"
#include "sentio/sizer.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace sentio {

SignalOrStrategy::SignalOrStrategy(const SignalOrCfg& cfg) 
    : BaseStrategy("SignalOR"), cfg_(cfg) {
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
    const std::string& bear3x_symbol,
    const std::string& inverse_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    if (current_index < 0 || current_index >= static_cast<int>(bars.size())) {
        return decisions; // Empty if invalid index
    }
    
    double probability = calculate_probability(bars, current_index);
    double signal_strength = std::abs(probability - 0.5) * 2.0;
    
    // Only make allocation decisions if signal is strong enough
    if (signal_strength < cfg_.min_signal_strength) {
        return decisions; // Empty if signal too weak
    }
    
    // Determine target weight based on probability
    double target_weight = 0.0;
    std::string target_symbol;
    std::string reason;
    
    if (probability > cfg_.long_threshold) {
        // Long signal - choose between base and 3x based on strength
        if (signal_strength > 0.7) {
            target_symbol = bull3x_symbol;
            target_weight = cfg_.max_position_weight;
            reason = "Strong long signal - 3x leveraged";
        } else {
            target_symbol = base_symbol;
            target_weight = cfg_.max_position_weight * 0.6; // Conservative sizing
            reason = "Moderate long signal - base position";
        }
    } else if (probability < cfg_.short_threshold) {
        // Short signal - choose between inverse and bear 3x
        if (signal_strength > 0.7) {
            target_symbol = bear3x_symbol;
            target_weight = -cfg_.max_position_weight; // Negative for short
            reason = "Strong short signal - 3x leveraged short";
        } else {
            target_symbol = inverse_symbol;
            target_weight = -cfg_.max_position_weight * 0.6; // Conservative sizing
            reason = "Moderate short signal - inverse position";
        }
    }
    
    if (!target_symbol.empty() && std::abs(target_weight) > 1e-6) {
        AllocationDecision decision;
        decision.instrument = target_symbol;
        decision.target_weight = target_weight;
        decision.confidence = signal_strength;
        decision.reason = reason;
        decisions.push_back(decision);
    }
    
    return decisions;
}

RouterCfg SignalOrStrategy::get_router_config() const {
    RouterCfg cfg;
    return cfg;
}

SizerCfg SignalOrStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.fractional_allowed = true;
    cfg.min_notional = 100.0; // $100 minimum
    cfg.max_leverage = 3.0; // Allow 3x leverage
    cfg.max_position_pct = cfg_.max_position_weight;
    cfg.volatility_target = 0.20; // 20% target volatility
    cfg.allow_negative_cash = false;
    cfg.vol_lookback_days = 20;
    cfg.cash_reserve_pct = 0.05; // 5% cash reserve
    return cfg;
}

// Configuration
ParameterMap SignalOrStrategy::get_default_params() const {
    return {
        {"min_signal_strength", cfg_.min_signal_strength},
        {"long_threshold", cfg_.long_threshold},
        {"short_threshold", cfg_.short_threshold},
        {"hold_threshold", cfg_.hold_threshold},
        {"max_position_weight", cfg_.max_position_weight},
        {"position_decay", cfg_.position_decay},
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
    space["max_position_weight"] = {ParamType::FLOAT, 0.5, 1.0, cfg_.max_position_weight};
    space["position_decay"] = {ParamType::FLOAT, 0.9, 0.99, cfg_.position_decay};
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
    if (params_.count("max_position_weight")) {
        cfg_.max_position_weight = params_.at("max_position_weight");
    }
    if (params_.count("position_decay")) {
        cfg_.position_decay = params_.at("position_decay");
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
    current_position_weight_ = 0.0;
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
    
    // Convert momentum to probability
    double momentum_prob = 0.5 + std::clamp(momentum * cfg_.momentum_scale, -0.4, 0.4);
    
    return momentum_prob;
}

double SignalOrStrategy::calculate_position_weight(double signal_strength) {
    // Calculate position weight based on signal strength
    double base_weight = signal_strength * cfg_.max_position_weight;
    
    // Apply position decay
    current_position_weight_ *= cfg_.position_decay;
    
    // Update with new signal
    current_position_weight_ = std::max(current_position_weight_, base_weight);
    
    return std::min(current_position_weight_, cfg_.max_position_weight);
}

void SignalOrStrategy::update_position_decay() {
    // Apply position decay to reduce position over time without new signals
    current_position_weight_ *= cfg_.position_decay;
    
    // Prevent position from becoming negative
    current_position_weight_ = std::max(0.0, current_position_weight_);
}

} // namespace sentio