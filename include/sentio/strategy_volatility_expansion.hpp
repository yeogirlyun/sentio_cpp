#pragma once
#include "base_strategy.hpp"
#include "volatility_expansion.hpp" // For RollingHHLL
#include <vector>
#include <string>

namespace sentio {

class VolatilityExpansionStrategy : public BaseStrategy {
private:
    // **MODIFIED**: Cached parameters for performance
    int atr_window_;
    double atr_alpha_;
    int lookback_hh_;
    int lookback_ll_;
    double breakout_k_;
    int hold_max_bars_;
    double tp_atr_mult_;
    double sl_atr_mult_;
    bool require_rth_;

    // State machine states
    enum class VEState { Flat, Long, Short };
    
    // Strategy state & indicators
    VEState state_ = VEState::Flat;
    int bars_in_trade_ = 0;
    RollingHHLL rolling_hh_;
    RollingHHLL rolling_ll_;
    double atr_ = 0.0;
    double prev_close_ = 0.0;
    
public:
    VolatilityExpansionStrategy();
    
    // BaseStrategy interface
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio