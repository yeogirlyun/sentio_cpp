#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp" // For rolling volume
#include <deque>

namespace sentio {

class MarketMakingStrategy : public BaseStrategy {
private:
    // **MODIFIED**: Cached parameters
    double base_spread_;
    double min_spread_;
    double max_spread_;
    int order_levels_;
    double level_spacing_;
    double order_size_base_;
    double max_inventory_;
    double inventory_skew_mult_;
    double adverse_selection_threshold_;
    double min_volume_ratio_;
    int max_orders_per_bar_;
    int rebalance_frequency_;

    // Market making state
    struct MarketState {
        double inventory = 0.0;
        double average_cost = 0.0;
        int last_rebalance = 0;
        int orders_this_bar = 0;
    };
    MarketState market_state_;

    // **NEW**: Stateful, rolling indicators for performance
    RollingMeanVar rolling_returns_;
    RollingMean rolling_volume_;

    // Helper methods
    bool should_participate(const Bar& bar);
    double get_inventory_skew() const;
    
public:
    MarketMakingStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio