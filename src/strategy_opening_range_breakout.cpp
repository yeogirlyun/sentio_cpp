#include "sentio/strategy_opening_range_breakout.hpp"
#include <algorithm>
#include <cmath>

namespace sentio {

OpeningRangeBreakoutStrategy::OpeningRangeBreakoutStrategy() 
    : BaseStrategy("OpeningRangeBreakout") {
    params_ = get_default_params();
    apply_params();
}
ParameterMap OpeningRangeBreakoutStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed cooldown to allow more frequent trades.
    return {
        {"opening_range_minutes", 30.0},
        {"breakout_confirmation_bars", 1.0},
        {"volume_multiplier", 1.5},
        {"stop_loss_pct", 0.01},
        {"take_profit_pct", 0.02},
        {"cool_down_period", 5.0}, // Was 15.0
    };
}

ParameterSpace OpeningRangeBreakoutStrategy::get_param_space() const { /* ... unchanged ... */ return {}; }

void OpeningRangeBreakoutStrategy::apply_params() {
    // **NEW**: Cache parameters
    opening_range_minutes_ = static_cast<int>(params_["opening_range_minutes"]);
    breakout_confirmation_bars_ = static_cast<int>(params_["breakout_confirmation_bars"]);
    volume_multiplier_ = params_["volume_multiplier"];
    stop_loss_pct_ = params_["stop_loss_pct"];
    take_profit_pct_ = params_["take_profit_pct"];
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);
    reset_state();
}

void OpeningRangeBreakoutStrategy::reset_state() {
    BaseStrategy::reset_state();
    current_range_ = OpeningRange{};
    day_start_index_ = -1;
}

double OpeningRangeBreakoutStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < 1) {
        diag_.drop(DropReason::MIN_BARS);
        return 0.5; // Neutral
    }

    // **MODIFIED**: Robust and performant new-day detection
    const int SECONDS_IN_DAY = 86400;
    long current_day = bars[current_index].ts_utc_epoch / SECONDS_IN_DAY;
    long prev_day = bars[current_index - 1].ts_utc_epoch / SECONDS_IN_DAY;

    if (current_day != prev_day) {
        reset_state(); // Reset everything for the new day
        day_start_index_ = current_index;
    }
    
    if (day_start_index_ == -1) { // Haven't established the start of the first day yet
        day_start_index_ = 0;
    }

    int bars_into_day = current_index - day_start_index_;

    // --- Phase 1: Define the Opening Range ---
    if (bars_into_day < opening_range_minutes_) {
        if (bars_into_day == 0) {
            current_range_.high = bars[current_index].high;
            current_range_.low = bars[current_index].low;
        } else {
            current_range_.high = std::max(current_range_.high, bars[current_index].high);
            current_range_.low = std::min(current_range_.low, bars[current_index].low);
        }
        diag_.drop(DropReason::SESSION); // Use SESSION to mean "in range formation"
        return 0.5; // Neutral
    }

    // --- Finalize the range exactly once ---
    if (!current_range_.is_finalized) {
        current_range_.end_bar = current_index - 1;
        current_range_.is_finalized = true;
    }

    // --- Phase 2: Look for Breakouts ---
    if (state_.in_position || is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return 0.5; // Neutral
    }

    const auto& bar = bars[current_index];
    bool is_breakout_up = bar.close > current_range_.high;
    bool is_breakout_down = bar.close < current_range_.low;

    if (!is_breakout_up && !is_breakout_down) {
        diag_.drop(DropReason::THRESHOLD);
        return 0.5; // Neutral
    }
    
    // Volume Confirmation
    double avg_volume = 0;
    for (int i = day_start_index_; i < current_range_.end_bar; ++i) {
        avg_volume += bars[i].volume;
    }
    avg_volume /= (current_range_.end_bar - day_start_index_ + 1);

    if (bar.volume < avg_volume * volume_multiplier_) {
        diag_.drop(DropReason::ZERO_VOL); // Re-using for low volume
        return 0.5; // Neutral
    }

    // Generate Signal
    double probability;
    if (is_breakout_up) {
        probability = 0.9; // Strong buy signal
    } else { // is_breakout_down
        probability = 0.1; // Strong sell signal
    }

    diag_.emitted++;
    state_.in_position = true; // Manually set state as this is an intraday strategy
    state_.last_trade_bar = current_index;

    return probability;
}

std::vector<BaseStrategy::AllocationDecision> OpeningRangeBreakoutStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // OpeningRangeBreakout uses simple allocation based on signal strength
    if (probability > 0.8) {
        // Strong buy signal
        double conviction = (probability - 0.8) / 0.2; // Scale 0.8-1.0 to 0-1
        double base_weight = 0.5 + (conviction * 0.5); // 50-100% allocation
        
        decisions.push_back({base_symbol, base_weight, conviction, "OpeningRangeBreakout strong buy: 100% QQQ"});
    } else if (probability < 0.2) {
        // Strong sell signal
        double conviction = (0.2 - probability) / 0.2; // Scale 0.0-0.2 to 0-1
        double base_weight = 0.5 + (conviction * 0.5); // 50-100% allocation
        
        // Use SHORT QQQ for moderate sell signals instead of PSQ
        decisions.push_back({base_symbol, -base_weight, conviction, "OpeningRangeBreakout strong sell: SHORT QQQ"});
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "OpeningRangeBreakout: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg OpeningRangeBreakoutStrategy::get_router_config() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}

SizerCfg OpeningRangeBreakoutStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.max_position_pct = 1.0; // 100% max position
    cfg.volatility_target = 0.15; // 15% volatility target
    return cfg;
}

// Register the strategy
REGISTER_STRATEGY(OpeningRangeBreakoutStrategy, "orb");

} // namespace sentio