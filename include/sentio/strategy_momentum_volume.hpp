#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp" // **NEW**: For efficient MA calculations
#include <map>

namespace sentio {

class MomentumVolumeProfileStrategy : public BaseStrategy {
private:
    // **MODIFIED**: Cached parameters for performance
    int profile_period_;
    double value_area_pct_;
    int price_bins_;
    double breakout_threshold_pct_;
    int momentum_lookback_;
    double volume_surge_mult_;
    int cool_down_period_;

    // Volume profile data structures
    struct VolumeNode {
        double price_level;
        double volume;
    };
    struct VolumeProfile {
        std::map<double, VolumeNode> profile;
        double poc_level = 0.0;
        double value_area_high = 0.0;
        double value_area_low = 0.0;
        double total_volume = 0.0;
        void clear() { /* ... unchanged ... */ }
    };
    
    // Strategy state
    VolumeProfile volume_profile_;
    int last_profile_update_ = -1;
    
    // **NEW**: Stateful, rolling indicators for performance
    RollingMean avg_volume_;

    // Helper methods
    void build_volume_profile(const std::vector<Bar>& bars, int end_index);
    void calculate_value_area();
    bool is_momentum_confirmed(const std::vector<Bar>& bars, int index) const;
    
public:
    MomentumVolumeProfileStrategy();
    
    // BaseStrategy interface
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio