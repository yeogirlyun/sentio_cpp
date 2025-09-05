#pragma once
#include "base_strategy.hpp"

namespace sentio {

class VWAPReversionStrategy : public BaseStrategy {
private:
    // Cached parameters for performance
    int vwap_period_;
    double band_multiplier_;
    double max_band_width_;
    double min_distance_from_vwap_;
    double volume_confirmation_mult_;
    int rsi_period_;
    double rsi_oversold_;
    double rsi_overbought_;
    double stop_loss_pct_;
    double take_profit_pct_;
    int time_stop_bars_;
    int cool_down_period_;

    // VWAP calculation state
    double cumulative_pv_ = 0.0;
    double cumulative_volume_ = 0.0;
    
    // Strategy state
    int time_in_position_ = 0;
    double vwap_ = 0.0;
    
    // Helper methods
    void update_vwap(const Bar& bar);
    bool is_volume_confirmed(const std::vector<Bar>& bars, int index) const;
    bool is_rsi_condition_met(const std::vector<Bar>& bars, int index, bool for_buy) const;
    double calculate_simple_rsi(const std::vector<double>& prices) const;
    
public:
    VWAPReversionStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio