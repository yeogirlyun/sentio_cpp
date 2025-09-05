#include "sentio/strategy_order_flow_imbalance.hpp"
#include <algorithm>
#include <cmath>

namespace sentio {

OrderFlowImbalanceStrategy::OrderFlowImbalanceStrategy() 
    : BaseStrategy("OrderFlowImbalance"),
      rolling_pressure_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap OrderFlowImbalanceStrategy::get_default_params() const {
    return {
        {"lookback_period", 50.0},
        {"entry_threshold_long", 0.60},
        {"entry_threshold_short", 0.40},
        {"hold_max_bars", 60.0},
        {"cool_down_period", 5.0}
    };
}

ParameterSpace OrderFlowImbalanceStrategy::get_param_space() const { return {}; }

void OrderFlowImbalanceStrategy::apply_params() {
    lookback_period_ = static_cast<int>(params_["lookback_period"]);
    entry_threshold_long_ = params_["entry_threshold_long"];
    entry_threshold_short_ = params_["entry_threshold_short"];
    hold_max_bars_ = static_cast<int>(params_["hold_max_bars"]);
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);
    
    rolling_pressure_ = RollingMean(lookback_period_);
    reset_state();
}

void OrderFlowImbalanceStrategy::reset_state() {
    BaseStrategy::reset_state();
    ofi_state_ = OFIState::Flat; // **FIXED**: Use the renamed state variable
    bars_in_trade_ = 0;
    rolling_pressure_ = RollingMean(lookback_period_);
}

double OrderFlowImbalanceStrategy::calculate_bar_pressure(const Bar& bar) const {
    double range = bar.high - bar.low;
    if (range < 1e-9) {
        return 0.5; // Neutral pressure if there's no range
    }
    return (bar.close - bar.low) / range;
}

StrategySignal OrderFlowImbalanceStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < lookback_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }

    double pressure = calculate_bar_pressure(bars[current_index]);
    double avg_pressure = rolling_pressure_.push(pressure);

    // **FIXED**: Use the strategy-specific 'ofi_state_' for state machine logic
    if (ofi_state_ == OFIState::Flat) {
        if (is_cooldown_active(current_index, cool_down_period_)) {
            diag_.drop(DropReason::COOLDOWN);
            return signal;
        }

        if (avg_pressure > entry_threshold_long_) {
            signal.type = StrategySignal::Type::BUY;
            ofi_state_ = OFIState::Long;
            // **FIXED**: Correctly access the 'state_' member from BaseStrategy
            state_.last_trade_bar = current_index;
        } else if (avg_pressure < entry_threshold_short_) {
            signal.type = StrategySignal::Type::SELL;
            ofi_state_ = OFIState::Short;
            // **FIXED**: Correctly access the 'state_' member from BaseStrategy
            state_.last_trade_bar = current_index;
        } else {
            diag_.drop(DropReason::THRESHOLD);
            return signal;
        }

        signal.confidence = 0.7;
        diag_.emitted++;
        bars_in_trade_ = 0;

    } else { // In a trade, check for exit
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            // **FIXED**: Use 'ofi_state_' to determine exit signal direction
            signal.type = (ofi_state_ == OFIState::Long) ? StrategySignal::Type::SELL : StrategySignal::Type::BUY;
            diag_.emitted++;
            reset_state();
        }
    }

    return signal;
}

REGISTER_STRATEGY(OrderFlowImbalanceStrategy, "OrderFlowImbalance");

} // namespace sentio

