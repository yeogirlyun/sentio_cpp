#include "sentio/strategy_volatility_expansion.hpp"
#include "sentio/rth_calendar.hpp"
#include "sentio/calendar_seed.hpp"
#include <algorithm>
#include <cmath>

namespace sentio {

VolatilityExpansionStrategy::VolatilityExpansionStrategy() 
    : BaseStrategy("VolatilityExpansion"),
      rolling_hh_(20),
      rolling_ll_(20) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap VolatilityExpansionStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed breakout_k to be more sensitive.
    // 'require_rth' is now correctly handled by the fixed session logic, so it can be enabled.
    return {
        {"atr_window", 14.0},
        {"lookback_hh", 20.0},
        {"lookback_ll", 20.0},
        {"breakout_k", 0.01}, // Extremely sensitive - was 0.05, now 0.01
        {"hold_max_bars", 160.0},
        {"tp_atr_mult", 1.5},
        {"sl_atr_mult", 1.0},
        {"require_rth", 0.0} // Disabled to test if RTH is causing issues
    };
}

ParameterSpace VolatilityExpansionStrategy::get_param_space() const { return {}; }

void VolatilityExpansionStrategy::apply_params() {
    atr_window_ = static_cast<int>(params_["atr_window"]);
    atr_alpha_ = 2.0 / (atr_window_ + 1.0);
    lookback_hh_ = static_cast<int>(params_["lookback_hh"]);
    lookback_ll_ = static_cast<int>(params_["lookback_ll"]);
    breakout_k_ = params_["breakout_k"];
    hold_max_bars_ = static_cast<int>(params_["hold_max_bars"]);
    tp_atr_mult_ = params_["tp_atr_mult"];
    sl_atr_mult_ = params_["sl_atr_mult"];
    require_rth_ = params_["require_rth"] > 0.5;

    rolling_hh_ = RollingHHLL(lookback_hh_);
    rolling_ll_ = RollingHHLL(lookback_ll_);
    reset_state();
}

void VolatilityExpansionStrategy::reset_state() {
    BaseStrategy::reset_state();
    state_ = VEState::Flat;
    bars_in_trade_ = 0;
    atr_ = 0.0;
    prev_close_ = 0.0;
}

StrategySignal VolatilityExpansionStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;
    const int min_bars = std::max({lookback_hh_, lookback_ll_, atr_window_});
    
    if (current_index < min_bars) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }
    
    // Initialize prev_close on the first valid bar
    if (prev_close_ == 0.0) {
        prev_close_ = bars[current_index - 1].close;
    }

    if (require_rth_) {
        static sentio::TradingCalendar calendar = sentio::make_default_nyse_calendar();
        if (!calendar.is_rth_utc(bars[current_index].ts_nyt_epoch, "America/New_York")) {
            diag_.drop(DropReason::SESSION);
            return signal;
        }
    }
    
    const auto& bar = bars[current_index];
    double tr = true_range(bar.high, bar.low, prev_close_);
    
    if (atr_ == 0.0) { // Initialize ATR with a simple moving average
        double sum_tr = 0.0;
        for (int i = 0; i < atr_window_; ++i) {
            double pc = (current_index - i > 0) ? bars[current_index - i - 1].close : bars[current_index - i].open;
            sum_tr += true_range(bars[current_index - i].high, bars[current_index - i].low, pc);
        }
        atr_ = sum_tr / atr_window_;
    } else { // Use exponential moving average for subsequent bars
        atr_ = (tr * atr_alpha_) + (atr_ * (1.0 - atr_alpha_));
    }

    auto [hh, _] = rolling_hh_.push(bar.high, bar.low);
    auto [__, ll] = rolling_ll_.push(bar.high, bar.low);
    
    prev_close_ = bar.close; // Update for the next iteration

    if (state_ == VEState::Flat) {
        const double up_trigger = hh + breakout_k_ * atr_;
        const double dn_trigger = ll - breakout_k_ * atr_;

        if (bar.close > up_trigger) {
            signal.type = StrategySignal::Type::BUY;
            state_ = VEState::Long;
        } else if (bar.close < dn_trigger) {
            signal.type = StrategySignal::Type::SELL;
            state_ = VEState::Short;
        } else {
            diag_.drop(DropReason::THRESHOLD);
            return signal;
        }

        signal.confidence = 0.8;
        diag_.emitted++;
        bars_in_trade_ = 0;
    } else {
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            signal.type = (state_ == VEState::Long) ? StrategySignal::Type::SELL : StrategySignal::Type::BUY;
            diag_.emitted++;
            reset_state();
        }
    }
    
    return signal;
}

REGISTER_STRATEGY(VolatilityExpansionStrategy, "VolatilityExpansion");

} // namespace sentio

