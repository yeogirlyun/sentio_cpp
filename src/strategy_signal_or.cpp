#include "sentio/strategy_signal_or.hpp"
#include "sentio/signal_utils.hpp"
#include "sentio/detectors/rsi_detector.hpp"
#include "sentio/detectors/bollinger_detector.hpp"
#include "sentio/detectors/momentum_volume_detector.hpp"
#include "sentio/detectors/ofi_proxy_detector.hpp"
#include "sentio/detectors/opening_range_breakout_detector.hpp"
#include "sentio/detectors/vwap_reversion_detector.hpp"
// REMOVED: router.hpp - AllocationManager handles routing
// REMOVED: sizer.hpp - handled by runner
// REMOVED: allocation_manager.hpp - handled by runner
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
    
    // **REMOVED**: AllocationManager is handled by runner, not strategy
    // Strategy only provides probability signals
    
    // Initialize integrated detectors
    detectors_.emplace_back(std::make_unique<detectors::RsiDetector>());
    detectors_.emplace_back(std::make_unique<detectors::BollingerDetector>());
    detectors_.emplace_back(std::make_unique<detectors::MomentumVolumeDetector>());
    detectors_.emplace_back(std::make_unique<detectors::OFIProxyDetector>());
    detectors_.emplace_back(std::make_unique<detectors::OpeningRangeBreakoutDetector>());
    detectors_.emplace_back(std::make_unique<detectors::VwapReversionDetector>());
    for (const auto& d : detectors_) max_warmup_ = std::max(max_warmup_, d->warmup_period());

    apply_params();
}

// Required BaseStrategy methods
double SignalOrStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < 0 || current_index >= static_cast<int>(bars.size())) {
        return 0.5; // Neutral if invalid index
    }
    
    warmup_bars_++;

    // If not enough bars for detectors, neutral
    if (current_index < max_warmup_) {
        return 0.5;
    }

    // Run detectors and aggregate majority
    auto ctx = run_and_aggregate(bars, current_index);
    double probability = ctx.final_probability;
    
    // **FIXED**: Update signal diagnostics counter
    diag_.emitted++;
    
    return probability;
}

SignalOrStrategy::AuditContext SignalOrStrategy::run_and_aggregate(const std::vector<Bar>& bars, int idx) {
    AuditContext ctx;
    int total_votes = 0;
    for (const auto& d : detectors_) {
        auto res = d->score(bars, idx);
        ctx.detector_probs[std::string(res.name)] = res.probability;
        if (res.direction == 1) { ctx.long_votes++; total_votes++; }
        else if (res.direction == -1) { ctx.short_votes++; total_votes++; }
    }
    if (total_votes == 0) { ctx.final_probability = 0.5; return ctx; }
    double long_ratio = static_cast<double>(ctx.long_votes) / total_votes;
    if (ctx.long_votes > ctx.short_votes) {
        ctx.final_probability = 0.5 + (long_ratio * 0.5);
        if (long_ratio > 0.8) ctx.final_probability = std::min(0.95, ctx.final_probability + 0.1);
    } else if (ctx.short_votes > ctx.long_votes) {
        double short_ratio = 1.0 - long_ratio;
        ctx.final_probability = 0.5 - (short_ratio * 0.5);
        if (short_ratio > 0.8) ctx.final_probability = std::max(0.05, ctx.final_probability - 0.1);
    } else {
        ctx.final_probability = 0.5;
    }
    return ctx;
}

// REMOVED: get_allocation_decisions - AllocationManager handles all instrument decisions

// REMOVED: get_router_config - AllocationManager handles routing

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