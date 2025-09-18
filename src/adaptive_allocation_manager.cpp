#include "sentio/adaptive_allocation_manager.hpp"
#include <cmath>
#include <algorithm>

namespace sentio {

AdaptiveAllocationManager::AdaptiveAllocationManager() {}

std::vector<AllocationDecision> AdaptiveAllocationManager::get_allocations(
    double probability,
    const StrategyProfiler::StrategyProfile& profile) {
    
    update_thresholds(profile);
    
    // Filter signal based on profile
    double filtered_prob = filter_signal(probability, profile);
    
    // Check if signal is strong enough to trade
    if (!should_trade(filtered_prob, profile)) {
        return {};  // No allocation
    }
    
    std::vector<AllocationDecision> decisions;
    // Signal strength calculation (currently unused but may be needed for future enhancements)
    // double signal_strength = std::abs(filtered_prob - 0.5);
    bool is_bullish = filtered_prob > 0.5;
    
    if (is_bullish) {
        if (filtered_prob >= thresholds_.entry_3x) {
            decisions.push_back({"TQQQ", 1.0, 
                "Strong bullish (p=" + std::to_string(filtered_prob) + "), adaptive 3x"});
        } else if (filtered_prob >= thresholds_.entry_1x) {
            decisions.push_back({"QQQ", 1.0, 
                "Moderate bullish (p=" + std::to_string(filtered_prob) + "), adaptive 1x"});
        }
    } else {
        double inverse_prob = 1.0 - filtered_prob;
        if (inverse_prob >= thresholds_.entry_3x) {
            decisions.push_back({"SQQQ", 1.0, 
                "Strong bearish (p=" + std::to_string(filtered_prob) + "), adaptive 3x inverse"});
        } else if (inverse_prob >= thresholds_.entry_1x) {
            decisions.push_back({"PSQ", 1.0, 
                "Moderate bearish (p=" + std::to_string(filtered_prob) + "), adaptive 1x inverse"});
        }
    }
    
    if (!decisions.empty()) {
        signals_since_trade_ = 0;
    } else {
        signals_since_trade_++;
    }
    
    last_signal_ = filtered_prob;
    return decisions;
}

bool AdaptiveAllocationManager::should_trade(double probability, 
    const StrategyProfiler::StrategyProfile& profile) const {
    
    double signal_strength = std::abs(probability - 0.5);
    
    // Adaptive noise filtering
    if (signal_strength < profile.adaptive_noise_floor) {
        return false;  // Signal is noise
    }
    
    // For aggressive strategies, require stronger signals to reduce churning
    if (profile.style == TradingStyle::AGGRESSIVE) {
        return signal_strength > thresholds_.signal_strength_min * 1.5;
    }
    
    // For conservative strategies, respect smaller signals
    if (profile.style == TradingStyle::CONSERVATIVE) {
        return signal_strength > thresholds_.signal_strength_min * 0.7;
    }
    
    return signal_strength > thresholds_.signal_strength_min;
}

void AdaptiveAllocationManager::update_thresholds(const StrategyProfiler::StrategyProfile& profile) {
    // Use adaptive thresholds from profile
    thresholds_.entry_1x = profile.adaptive_entry_1x;
    thresholds_.entry_3x = profile.adaptive_entry_3x;
    thresholds_.noise_floor = profile.adaptive_noise_floor;
    
    // Adjust signal strength minimum based on trading frequency
    if (profile.trades_per_block > 200) {
        // Very high frequency - increase minimum signal strength
        thresholds_.signal_strength_min = 0.15;
    } else if (profile.trades_per_block > 50) {
        // Moderate frequency
        thresholds_.signal_strength_min = 0.10;
    } else {
        // Low frequency - allow weaker signals
        thresholds_.signal_strength_min = 0.05;
    }
}

double AdaptiveAllocationManager::filter_signal(double raw_probability, 
    const StrategyProfiler::StrategyProfile& profile) const {
    
    // Apply smoothing for volatile strategies
    if (profile.signal_volatility > 0.2) {
        // Exponential smoothing
        double alpha = 0.3;  // Smoothing factor
        return alpha * raw_probability + (1 - alpha) * last_signal_;
    }
    
    // For stable strategies, use raw signal
    return raw_probability;
}

} // namespace sentio
