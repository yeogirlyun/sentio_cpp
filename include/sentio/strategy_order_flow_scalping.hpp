#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp"

namespace sentio {

class OrderFlowScalpingStrategy : public BaseStrategy {
private:
    // Cached parameters
    int lookback_period_;
    double imbalance_threshold_;
    int hold_max_bars_;
    int cool_down_period_;

    // State machine states
    enum class OFState { Idle, ArmedLong, ArmedShort, Long, Short };
    
    // **FIXED**: Renamed this member to 'of_state_' to avoid conflict
    // with the 'state_' member inherited from BaseStrategy.
    OFState of_state_ = OFState::Idle;
    int bars_in_trade_ = 0;
    RollingMean rolling_pressure_;

    // Helper to calculate buying/selling pressure proxy from a bar
    double calculate_bar_pressure(const Bar& bar) const;
    
public:
    OrderFlowScalpingStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
