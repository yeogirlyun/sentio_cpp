#pragma once

#include <chrono>
#include <unordered_map>

namespace sentio {

/**
 * Utility functions for strategy implementations
 */
class StrategyUtils {
public:
    /**
     * Check if cooldown period is active for a given symbol
     * @param symbol The trading symbol
     * @param last_trade_time Last trade timestamp
     * @param cooldown_seconds Cooldown period in seconds
     * @return true if cooldown is active
     */
    static bool is_cooldown_active(
        const std::string& symbol,
        int64_t last_trade_time,
        int cooldown_seconds
    ) {
        if (cooldown_seconds <= 0) {
            return false;
        }
        
        auto now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        
        return (now - last_trade_time) < cooldown_seconds;
    }
    
    /**
     * Check if cooldown period is active for a given symbol with per-symbol tracking
     * @param symbol The trading symbol
     * @param last_trade_times Map of symbol to last trade time
     * @param cooldown_seconds Cooldown period in seconds
     * @return true if cooldown is active
     */
    static bool is_cooldown_active(
        const std::string& symbol,
        const std::unordered_map<std::string, int64_t>& last_trade_times,
        int cooldown_seconds
    ) {
        if (cooldown_seconds <= 0) {
            return false;
        }
        
        auto it = last_trade_times.find(symbol);
        if (it == last_trade_times.end()) {
            return false;
        }
        
        auto now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        
        return (now - it->second) < cooldown_seconds;
    }
};

} // namespace sentio
