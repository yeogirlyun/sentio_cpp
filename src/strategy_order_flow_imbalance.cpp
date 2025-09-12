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

double OrderFlowImbalanceStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < lookback_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return 0.5; // Neutral
    }

    double pressure = calculate_bar_pressure(bars[current_index]);
    double avg_pressure = rolling_pressure_.push(pressure);

    // **FIXED**: Use the strategy-specific 'ofi_state_' for state machine logic
    if (ofi_state_ == OFIState::Flat) {
        if (is_cooldown_active(current_index, cool_down_period_)) {
            diag_.drop(DropReason::COOLDOWN);
            return 0.5; // Neutral
        }

        double probability;
        if (avg_pressure > entry_threshold_long_) {
            probability = 0.7; // Buy signal
            ofi_state_ = OFIState::Long;
            // **FIXED**: Correctly access the 'state_' member from BaseStrategy
            state_.last_trade_bar = current_index;
        } else if (avg_pressure < entry_threshold_short_) {
            probability = 0.3; // Sell signal
            ofi_state_ = OFIState::Short;
            // **FIXED**: Correctly access the 'state_' member from BaseStrategy
            state_.last_trade_bar = current_index;
        } else {
            diag_.drop(DropReason::THRESHOLD);
            return 0.5; // Neutral
        }

        diag_.emitted++;
        return probability;

    } else { // In a trade, check for exit
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            // **FIXED**: Use 'ofi_state_' to determine exit signal direction
            double exit_prob = (ofi_state_ == OFIState::Long) ? 0.3 : 0.7; // SELL or BUY
            diag_.emitted++;
            reset_state();
            return exit_prob;
        }
        return 0.5; // Hold current position
    }
}

std::vector<BaseStrategy::AllocationDecision> OrderFlowImbalanceStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol,
    const std::string& bear1x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // OrderFlowImbalance uses simple allocation based on signal strength
    if (probability > 0.6) {
        // Buy signal
        double conviction = (probability - 0.6) / 0.4; // Scale 0.6-1.0 to 0-1
        double base_weight = 0.3 + (conviction * 0.4); // 30-70% allocation
        
        decisions.push_back({base_symbol, base_weight, conviction, "OrderFlowImbalance buy: 100% QQQ"});
    } else if (probability < 0.4) {
        // Sell signal
        double conviction = (0.4 - probability) / 0.4; // Scale 0.0-0.4 to 0-1
        double base_weight = 0.3 + (conviction * 0.4); // 30-70% allocation
        
        decisions.push_back({bear1x_symbol, base_weight, conviction, "OrderFlowImbalance sell: 100% PSQ"});
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol, bear1x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "OrderFlowImbalance: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg OrderFlowImbalanceStrategy::get_router_config() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    cfg.bear1x = "PSQ";
    return cfg;
}

SizerCfg OrderFlowImbalanceStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.max_position_pct = 0.7; // 70% max position
    cfg.volatility_target = 0.12; // 12% volatility target
    return cfg;
}

REGISTER_STRATEGY(OrderFlowImbalanceStrategy, "ofi");

} // namespace sentio

