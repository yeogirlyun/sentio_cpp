#pragma once
#include "base_strategy.hpp"
#include "bollinger.hpp"
#include "router.hpp"
#include "sizer.hpp"
#include <vector>
#include <string>

namespace sentio {

class BollingerSqueezeBreakoutStrategy : public BaseStrategy {
private:
    enum class State { Idle, Squeezed, ArmedLong, ArmedShort, Long, Short };
    
    // **MODIFIED**: Cached parameters
    int bb_window_;
    double squeeze_percentile_;
    int squeeze_lookback_;
    int hold_max_bars_;
    double tp_mult_sd_;
    double sl_mult_sd_;
    int min_squeeze_bars_;

    // Strategy state & indicators
    State state_ = State::Idle;
    int bars_in_trade_ = 0;
    int squeeze_duration_ = 0;
    Bollinger bollinger_;
    std::vector<double> sd_history_;
    
    // Helper methods
    double calculate_volatility_percentile(double percentile) const;
    void update_state_machine(const Bar& bar);
    
public:
    BollingerSqueezeBreakoutStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
    
    // **NEW**: Strategy-agnostic allocation interface
    std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, 
        int current_index,
        const std::string& base_symbol,
        const std::string& bull3x_symbol,
        const std::string& bear3x_symbol) override;
    
    RouterCfg get_router_config() const override;
    SizerCfg get_sizer_config() const override;
};

} // namespace sentio