#pragma once
#include "sentio/adaptive_allocation_manager.hpp"
#include "sentio/strategy_profiler.hpp"
#include "sentio/core.hpp"
#include "sentio/symbol_table.hpp"
#include <unordered_set>

namespace sentio {

enum class CoordinationResult {
    APPROVED,
    REJECTED_CONFLICT,
    REJECTED_FREQUENCY
};

struct CoordinationDecision {
    AllocationDecision decision;
    CoordinationResult result;
    std::string reason;
};

class UniversalPositionCoordinator {
public:
    UniversalPositionCoordinator();
    
    /**
     * Coordinate allocation decisions with portfolio state.
     * Enforces:
     * 1. No conflicting positions (long vs inverse)
     * 2. Maximum one OPENING trade per bar (closing trades unlimited)
     */
    std::vector<CoordinationDecision> coordinate(
        const std::vector<AllocationDecision>& allocations,
        const Portfolio& portfolio,
        const SymbolTable& ST,
        int64_t current_timestamp,
        const StrategyProfiler::StrategyProfile& profile
    );
    
    void reset_bar(int64_t timestamp);
    
private:
    const std::unordered_set<std::string> LONG_ETFS = {"QQQ", "TQQQ"};
    const std::unordered_set<std::string> INVERSE_ETFS = {"SQQQ", "PSQ"};
    
    int64_t current_bar_timestamp_ = -1;
    int opening_trades_this_bar_ = 0;
    
    bool check_portfolio_conflicts(const Portfolio& portfolio, 
                                  const SymbolTable& ST) const;
    
    bool would_create_conflict(const std::string& instrument, 
                              const Portfolio& portfolio, 
                              const SymbolTable& ST) const;
    
    bool portfolio_has_positions(const Portfolio& portfolio) const;
};

} // namespace sentio
