#pragma once
#include "base_strategy.hpp"

namespace sentio {

class OpeningRangeBreakoutStrategy : public BaseStrategy {
private:
    // **MODIFIED**: Cached parameters
    int opening_range_minutes_;
    int breakout_confirmation_bars_;
    double volume_multiplier_;
    double stop_loss_pct_;
    double take_profit_pct_;
    int cool_down_period_;

    struct OpeningRange {
        double high = 0.0;
        double low = 0.0;
        int end_bar = -1;
        bool is_finalized = false;
    };
    
    // Strategy state
    OpeningRange current_range_;
    int day_start_index_ = -1;
    
public:
    OpeningRangeBreakoutStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio