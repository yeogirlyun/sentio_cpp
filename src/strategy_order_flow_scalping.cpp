#include "sentio/strategy_order_flow_scalping.hpp"
#include <algorithm>
#include <cmath>

namespace sentio {

OrderFlowScalpingStrategy::OrderFlowScalpingStrategy() 
    : BaseStrategy("OrderFlowScalping"),
      rolling_pressure_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap OrderFlowScalpingStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed the imbalance threshold to arm more frequently.
    return {
        {"lookback_period", 50.0},
        {"imbalance_threshold", 0.55}, // Was 0.65, now arms when avg pressure is > 55%
        {"hold_max_bars", 20.0},
        {"cool_down_period", 3.0}
    };
}

ParameterSpace OrderFlowScalpingStrategy::get_param_space() const { return {}; }

void OrderFlowScalpingStrategy::apply_params() {
    lookback_period_ = static_cast<int>(params_["lookback_period"]);
    imbalance_threshold_ = params_["imbalance_threshold"];
    hold_max_bars_ = static_cast<int>(params_["hold_max_bars"]);
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);

    rolling_pressure_ = RollingMean(lookback_period_);
    reset_state();
}

void OrderFlowScalpingStrategy::reset_state() {
    BaseStrategy::reset_state();
    of_state_ = OFState::Idle; // **FIXED**: Use the renamed state variable
    bars_in_trade_ = 0;
    rolling_pressure_ = RollingMean(lookback_period_);
}

double OrderFlowScalpingStrategy::calculate_bar_pressure(const Bar& bar) const {
    double range = bar.high - bar.low;
    if (range < 1e-9) return 0.5;
    return (bar.close - bar.low) / range;
}

double OrderFlowScalpingStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < lookback_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return 0.5; // Neutral
    }

    const auto& bar = bars[current_index];
    double pressure = calculate_bar_pressure(bar);
    double avg_pressure = rolling_pressure_.push(pressure);

    // **FIXED**: Use the strategy-specific 'of_state_' for state machine logic
    if (of_state_ == OFState::Long || of_state_ == OFState::Short) {
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            double exit_prob = (of_state_ == OFState::Long) ? 0.3 : 0.7; // SELL or BUY
            diag_.emitted++;
            reset_state();
            return exit_prob;
        }
        return 0.5; // Hold current position
    }
    
    if (is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return 0.5; // Neutral
    }

    double probability = 0.5; // Default neutral
    switch (of_state_) {
        case OFState::Idle:
            if (avg_pressure > imbalance_threshold_) of_state_ = OFState::ArmedLong;
            else if (avg_pressure < (1.0 - imbalance_threshold_)) of_state_ = OFState::ArmedShort;
            else diag_.drop(DropReason::THRESHOLD);
            break;
            
        case OFState::ArmedLong:
            if (pressure > 0.5) { // Confirmation bar must be bullish
                probability = 0.7; // Buy signal
                of_state_ = OFState::Long;
            } else { // Failed confirmation
                of_state_ = OFState::Idle;
                diag_.drop(DropReason::THRESHOLD);
            }
            break;

        case OFState::ArmedShort:
            if (pressure < 0.5) { // Confirmation bar must be bearish
                probability = 0.3; // Sell signal
                of_state_ = OFState::Short;
            } else { // Failed confirmation
                of_state_ = OFState::Idle;
                diag_.drop(DropReason::THRESHOLD);
            }
            break;
        default: break;
    }
    
    if (probability != 0.5) {
        diag_.emitted++;
        bars_in_trade_ = 0;
        // **FIXED**: This now correctly refers to the 'state_' member from BaseStrategy
        state_.last_trade_bar = current_index;
    }
    
    return probability;
}

std::vector<BaseStrategy::AllocationDecision> OrderFlowScalpingStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // OrderFlowScalping uses simple allocation based on signal strength
    if (probability > 0.6) {
        // Buy signal
        double conviction = (probability - 0.6) / 0.4; // Scale 0.6-1.0 to 0-1
        double base_weight = 0.2 + (conviction * 0.3); // 20-50% allocation (scalping is smaller)
        
        decisions.push_back({base_symbol, base_weight, conviction, "OrderFlowScalping buy: 100% QQQ"});
    } else if (probability < 0.4) {
        // Sell signal
        double conviction = (0.4 - probability) / 0.4; // Scale 0.0-0.4 to 0-1
        double base_weight = 0.2 + (conviction * 0.3); // 20-50% allocation (scalping is smaller)
        
        // Use SHORT QQQ for moderate sell signals instead of PSQ
        decisions.push_back({base_symbol, -base_weight, conviction, "OrderFlowScalping sell: SHORT QQQ"});
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "OrderFlowScalping: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg OrderFlowScalpingStrategy::get_router_config() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}

SizerCfg OrderFlowScalpingStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.max_position_pct = 0.5; // 50% max position for scalping
    cfg.volatility_target = 0.10; // 10% volatility target
    return cfg;
}

REGISTER_STRATEGY(OrderFlowScalpingStrategy, "ofs");

} // namespace sentio

