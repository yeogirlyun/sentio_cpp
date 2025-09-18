#pragma once
#include "sentio/adaptive_allocation_manager.hpp"
#include "sentio/strategy_profiler.hpp"
#include "sentio/core.hpp"
#include "sentio/symbol_table.hpp"
#include <unordered_map>
#include <unordered_set>
#include <queue>

namespace sentio {

/**
 * @brief The result of a coordination check for a requested trade.
 */
enum class CoordinationResult {
    APPROVED,
    REJECTED_CONFLICT,
    REJECTED_FREQUENCY
};

/**
 * @brief The output of the coordinator for a single allocation request.
 */
struct CoordinationDecision {
    AllocationDecision decision;
    CoordinationResult result;
    std::string reason;
};

class UniversalPositionCoordinator {
public:
    UniversalPositionCoordinator();
    
    std::vector<CoordinationDecision> coordinate(
        const std::vector<AllocationDecision>& allocations,
        const Portfolio& portfolio,
        const SymbolTable& ST,
        int64_t current_timestamp,
        const StrategyProfiler::StrategyProfile& profile
    );
    
    void reset_bar(int64_t timestamp);
    
private:
    struct TradeRecord {
        int64_t timestamp;
        std::string instrument;
        double signal_strength;
    };
    
    std::unordered_map<int64_t, std::vector<TradeRecord>> trades_per_timestamp_;
    
    const std::unordered_set<std::string> LONG_ETFS = {"QQQ", "TQQQ"};
    const std::unordered_set<std::string> INVERSE_ETFS = {"SQQQ", "PSQ"};
    
    bool would_create_conflict(const std::string& instrument, 
                              const Portfolio& portfolio, 
                              const SymbolTable& ST) const;
    
    bool can_trade_at_timestamp(const std::string& instrument, 
                               int64_t timestamp,
                               const StrategyProfiler::StrategyProfile& profile);
    
};

} // namespace sentio
