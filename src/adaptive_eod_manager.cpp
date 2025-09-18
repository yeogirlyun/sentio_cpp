#include "sentio/adaptive_eod_manager.hpp"
#include <ctime>

namespace sentio {

AdaptiveEODManager::AdaptiveEODManager() {}

std::vector<AllocationDecision> AdaptiveEODManager::get_eod_allocations(
    int64_t current_timestamp_utc,
    const Portfolio& portfolio,
    const SymbolTable& ST,
    const StrategyProfiler::StrategyProfile& profile) {
    
    adapt_config(profile);
    
    std::vector<AllocationDecision> closing_decisions;
    
    // Convert to time components
    time_t time_secs = current_timestamp_utc;
    tm* utc_tm = gmtime(&time_secs);
    if (!utc_tm) return closing_decisions;
    
    int current_hour = utc_tm->tm_hour;
    int current_minute = utc_tm->tm_min;
    // --- ROBUST MARKET DAY TRACKING ---
    // Market day starts at 9:30 ET, not midnight UTC
    // Convert to ET timezone for proper market day detection
    bool is_new_trading_day = false;
    
    // Simple ET approximation: UTC-5 (EST) or UTC-4 (EDT)
    // For production, use proper timezone library
    int et_hour = (utc_tm->tm_hour - 5 + 24) % 24; // Approximate EST
    
    // Create unique trading day ID based on ET date
    int et_day = utc_tm->tm_yday;
    if (et_hour < 9 || (et_hour == 9 && utc_tm->tm_min < 30)) {
        // Before market open, still previous trading day
        et_day = (et_day - 1 + 365) % 365;
    }
    int unique_trading_day_id = (utc_tm->tm_year * 1000) + et_day;
    
    if (unique_trading_day_id != last_processed_day_) {
        closed_today_.clear();
        last_processed_day_ = unique_trading_day_id;
        is_new_trading_day = true;
    }
    
    // Suppress unused variable warning
    (void)is_new_trading_day;
    
    // Calculate minutes until close
    int current_minutes = current_hour * 60 + current_minute;
    int close_minutes = config_.market_close_hour_utc * 60 + config_.market_close_minute_utc;
    int minutes_to_close = close_minutes - current_minutes;

    // LAYER 1: MANDATORY LIQUIDATION ("NUCLEAR OPTION")
    // This is the final, non-negotiable closure phase.
    constexpr int FINAL_LIQUIDATION_MINUTES = 5;
    if (minutes_to_close <= FINAL_LIQUIDATION_MINUTES && minutes_to_close > 0) {
        closure_active_ = true;
        for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
            // If any position still exists, generate a closing order.
            if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
                const std::string& symbol = ST.get_symbol(sid);
                closing_decisions.push_back({
                    symbol, 0.0, 
                    "MANDATORY EOD LIQUIDATION (" + std::to_string(minutes_to_close) + " min)"
                });
            }
        }
        // Return immediately to ensure no other logic interferes.
        return closing_decisions;
    }

    // LAYER 2: STANDARD ADAPTIVE CLOSURE WINDOW
    // This phase runs before the final liquidation.
    if (minutes_to_close <= config_.closure_start_minutes && minutes_to_close > FINAL_LIQUIDATION_MINUTES) {
        closure_active_ = true;
        
        bool force_close = (minutes_to_close <= config_.mandatory_close_minutes);
        
        for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
            if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
                const std::string& symbol = ST.get_symbol(sid);
                
                // The closed_today_ check is appropriate here to avoid repeated orders
                // during the "soft" close period. The mandatory phase above will catch anything missed.
                if (closed_today_.count(symbol)) continue;
                
                if (force_close || should_close_position(symbol, profile)) {
                    closing_decisions.push_back({
                        symbol, 0.0, 
                        "EOD closure (adaptive, " + std::to_string(minutes_to_close) + " min to close)"
                    });
                    closed_today_.insert(symbol);
                }
            }
        }
        
        if (!closing_decisions.empty()) {
            last_close_timestamp_ = current_timestamp_utc;
        }
    } else {
        closure_active_ = false;
    }
    
    return closing_decisions;
}

void AdaptiveEODManager::adapt_config(const StrategyProfiler::StrategyProfile& profile) {
    // Adjust closure timing based on strategy profile
    switch (profile.style) {
        case TradingStyle::AGGRESSIVE:
            // WIDER window for aggressive strategies to ensure all positions can be closed.
            config_.closure_start_minutes = 60;  // Start 1 hour before market close.
            config_.mandatory_close_minutes = 15; // Start forcing 15 min before close.
            break;
            
        case TradingStyle::CONSERVATIVE:
            // Standard closure for conservative strategies
            config_.closure_start_minutes = 15;
            config_.mandatory_close_minutes = 10;
            break;
            
        case TradingStyle::BURST:
            // Flexible closure for burst strategies
            config_.closure_start_minutes = 25;
            config_.mandatory_close_minutes = 12;
            break;
            
        default:
            // Keep defaults
            break;
    }
}

bool AdaptiveEODManager::should_close_position(const std::string& symbol,
    const StrategyProfiler::StrategyProfile& profile) const {
    
    // For aggressive strategies, always close
    if (profile.style == TradingStyle::AGGRESSIVE) {
        return true;
    }
    
    // For leveraged positions, always close
    if (symbol == "TQQQ" || symbol == "SQQQ") {
        return true;
    }
    
    // For conservative strategies, can be more selective
    return closure_active_;
}

} // namespace sentio
