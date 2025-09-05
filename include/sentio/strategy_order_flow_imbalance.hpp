#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp"

namespace sentio {

class OrderFlowImbalanceStrategy : public BaseStrategy {
private:
    // Cached parameters
    int lookback_period_;
    double entry_threshold_long_;
    double entry_threshold_short_;
    int hold_max_bars_;
    int cool_down_period_;

    // Strategy-specific state machine
    enum class OFIState { Flat, Long, Short };
    // **FIXED**: Renamed this member to 'ofi_state_' to avoid conflict
    // with the 'state_' member inherited from BaseStrategy.
    OFIState ofi_state_ = OFIState::Flat;
    int bars_in_trade_ = 0;

    // Rolling indicator to measure average pressure
    RollingMean rolling_pressure_;

    // Helper to calculate buying/selling pressure proxy from a bar
    double calculate_bar_pressure(const Bar& bar) const;
    
public:
    OrderFlowImbalanceStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
