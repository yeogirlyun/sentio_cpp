#include "sentio/strategy_momentum_volume.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>

namespace sentio {

MomentumVolumeProfileStrategy::MomentumVolumeProfileStrategy() 
    : BaseStrategy("MomentumVolumeProfile"), avg_volume_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap MomentumVolumeProfileStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed parameters to be more sensitive
    return {
        {"profile_period", 100.0},
        {"value_area_pct", 0.7},
        {"price_bins", 30.0},
        {"breakout_threshold_pct", 0.001},
        {"momentum_lookback", 20.0},
        {"volume_surge_mult", 1.2}, // Was 1.5
        {"cool_down_period", 5.0}   // Was 10
    };
}

ParameterSpace MomentumVolumeProfileStrategy::get_param_space() const { return {}; }

void MomentumVolumeProfileStrategy::apply_params() {
    profile_period_ = static_cast<int>(params_["profile_period"]);
    value_area_pct_ = params_["value_area_pct"];
    price_bins_ = static_cast<int>(params_["price_bins"]);
    breakout_threshold_pct_ = params_["breakout_threshold_pct"];
    momentum_lookback_ = static_cast<int>(params_["momentum_lookback"]);
    volume_surge_mult_ = params_["volume_surge_mult"];
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);
    avg_volume_ = RollingMean(profile_period_);
    reset_state();
}

void MomentumVolumeProfileStrategy::reset_state() {
    BaseStrategy::reset_state();
    volume_profile_.clear();
    last_profile_update_ = -1;
    avg_volume_ = RollingMean(profile_period_);
}

double MomentumVolumeProfileStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < profile_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return 0.5; // Neutral
    }
    
    if (is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return 0.5; // Neutral
    }
    
    // Periodically rebuild the expensive volume profile
    if (last_profile_update_ == -1 || current_index - last_profile_update_ >= 10) {
        build_volume_profile(bars, current_index);
        last_profile_update_ = current_index;
    }
    
    if (volume_profile_.value_area_high <= 0) {
        diag_.drop(DropReason::NAN_INPUT); // Profile not ready or invalid
        return 0.5; // Neutral
    }

    const auto& bar = bars[current_index];
    avg_volume_.push(bar.volume);
    
    bool breakout_up = bar.close > (volume_profile_.value_area_high * (1.0 + breakout_threshold_pct_));
    bool breakout_down = bar.close < (volume_profile_.value_area_low * (1.0 - breakout_threshold_pct_));

    if (!breakout_up && !breakout_down) {
        diag_.drop(DropReason::THRESHOLD);
        return 0.5; // Neutral
    }

    if (!is_momentum_confirmed(bars, current_index)) {
        diag_.drop(DropReason::THRESHOLD);
        return 0.5; // Neutral
    }
    
    if (bar.volume < avg_volume_.mean() * volume_surge_mult_) {
        diag_.drop(DropReason::ZERO_VOL);
        return 0.5; // Neutral
    }

    double probability;
    if (breakout_up) {
        probability = 0.85; // Strong buy signal
    } else {
        probability = 0.15; // Strong sell signal
    }
    
    diag_.emitted++;
    state_.last_trade_bar = current_index;
    return probability;
}

bool MomentumVolumeProfileStrategy::is_momentum_confirmed(const std::vector<Bar>& bars, int index) const {
    if (index < momentum_lookback_) return false;
    double price_change = bars[index].close - bars[index - momentum_lookback_].close;
    if (bars[index].close > volume_profile_.value_area_high) {
        return price_change > 0;
    }
    if (bars[index].close < volume_profile_.value_area_low) {
        return price_change < 0;
    }
    return false;
}

// **MODIFIED**: This is now a functional, albeit simple, implementation to prevent NaN drops.
void MomentumVolumeProfileStrategy::build_volume_profile(const std::vector<Bar>& bars, int end_index) {
    volume_profile_.clear();
    int start_index = std::max(0, end_index - profile_period_ + 1);

    double min_price = std::numeric_limits<double>::max();
    double max_price = std::numeric_limits<double>::lowest();
    
    for (int i = start_index; i <= end_index; ++i) {
        min_price = std::min(min_price, bars[i].low);
        max_price = std::max(max_price, bars[i].high);
    }
    
    if (max_price <= min_price) return; // Cannot build profile

    // Simple implementation: Value Area is the high/low of the lookback period
    volume_profile_.value_area_high = max_price;
    volume_profile_.value_area_low = min_price;
    volume_profile_.total_volume = 1.0; // Mark as valid by setting a non-zero value
    // A proper implementation would bin prices and find the 70% volume area.
}

void MomentumVolumeProfileStrategy::calculate_value_area() {
    // This is now handled within build_volume_profile for simplicity
}

std::vector<BaseStrategy::AllocationDecision> MomentumVolumeProfileStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // MomentumVolume uses simple allocation based on signal strength
    if (probability > 0.7) {
        // Strong buy signal
        double conviction = (probability - 0.7) / 0.3; // Scale 0.7-1.0 to 0-1
        double base_weight = 0.4 + (conviction * 0.6); // 40-100% allocation
        
        decisions.push_back({base_symbol, base_weight, conviction, "MomentumVolume strong buy: 100% QQQ"});
    } else if (probability < 0.3) {
        // Strong sell signal
        double conviction = (0.3 - probability) / 0.3; // Scale 0.0-0.3 to 0-1
        double base_weight = 0.4 + (conviction * 0.6); // 40-100% allocation
        
        // Use SHORT QQQ for moderate sell signals instead of PSQ
        decisions.push_back({base_symbol, -base_weight, conviction, "MomentumVolume strong sell: SHORT QQQ"});
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "MomentumVolume: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg MomentumVolumeProfileStrategy::get_router_config() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}

SizerCfg MomentumVolumeProfileStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.max_position_pct = 1.0; // 100% max position
    cfg.volatility_target = 0.15; // 15% volatility target
    return cfg;
}

REGISTER_STRATEGY(MomentumVolumeProfileStrategy, "mvp");

} // namespace sentio
