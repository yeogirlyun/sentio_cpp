#pragma once
#include "sentio/adaptive_allocation_manager.hpp"
#include "sentio/strategy_profiler.hpp"
#include "sentio/core.hpp"
#include "sentio/symbol_table.hpp"
#include <unordered_set>

namespace sentio {

class AdaptiveEODManager {
public:
    AdaptiveEODManager();
    
    std::vector<AllocationDecision> get_eod_allocations(
        int64_t current_timestamp_utc,
        const Portfolio& portfolio,
        const SymbolTable& ST,
        const StrategyProfiler::StrategyProfile& profile
    );
    
    bool is_closure_active() const { return closure_active_; }
    
        private:
            struct AdaptiveConfig {
                int closure_start_minutes = 15;
                int mandatory_close_minutes = 10;
                int market_close_hour_utc = 20;
                int market_close_minute_utc = 0;
            };
            
            AdaptiveConfig config_;
            bool closure_active_ = false;
            int64_t last_close_timestamp_ = 0;
            std::unordered_set<std::string> closed_today_;
            
            // **FIX**: Add robust day tracking to prevent EOD violations
            int last_processed_day_ = -1;
    
    void adapt_config(const StrategyProfiler::StrategyProfile& profile);
    bool should_close_position(const std::string& symbol, 
                              const StrategyProfiler::StrategyProfile& profile) const;
};

} // namespace sentio
