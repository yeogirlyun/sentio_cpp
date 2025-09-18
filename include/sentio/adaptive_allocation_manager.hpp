#pragma once
#include "sentio/strategy_profiler.hpp"
#include <memory>
#include <string>
#include <vector>

namespace sentio {

/**
 * @brief Represents a concrete allocation decision for the runner.
 */
struct AllocationDecision {
    std::string instrument;
    double target_weight; // e.g., 1.0 for 100%
    std::string reason;
};

class AdaptiveAllocationManager {
public:
    AdaptiveAllocationManager();
    
    std::vector<AllocationDecision> get_allocations(
        double probability,
        const StrategyProfiler::StrategyProfile& profile
    );
    
    // Signal filtering based on profile
    bool should_trade(double probability, const StrategyProfiler::StrategyProfile& profile) const;
    
private:
    struct DynamicThresholds {
        double entry_1x = 0.60;
        double entry_3x = 0.75;
        double noise_floor = 0.05;
        double signal_strength_min = 0.10;
    };
    
    DynamicThresholds thresholds_;
    double last_signal_ = 0.5;
    int signals_since_trade_ = 0;
    
    void update_thresholds(const StrategyProfiler::StrategyProfile& profile);
    double filter_signal(double raw_probability, const StrategyProfiler::StrategyProfile& profile) const;
};

} // namespace sentio
