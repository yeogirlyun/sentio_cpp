#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include "position_validator.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// Forward declarations
namespace sentio {
    class BaseStrategy;
}

namespace sentio {

// **POSITION COORDINATOR**: Strategy-agnostic conflict prevention system
// Ensures no conflicting positions are ever created by coordinating all allocation decisions

enum class CoordinationResult {
    APPROVED,           // Decision approved for execution
    REJECTED_CONFLICT,  // Decision rejected due to conflict
    REJECTED_FREQUENCY, // Decision rejected due to frequency limit
    MODIFIED            // Decision modified to prevent conflict
};

struct CoordinationDecision {
    CoordinationResult result;
    std::string instrument;
    double approved_weight;
    double original_weight;
    std::string reason;
    std::string conflict_details;
};

struct AllocationRequest {
    std::string strategy_name;
    std::string instrument;
    double target_weight;
    double confidence;
    std::string reason;
    std::string chain_id;
};

class PositionCoordinator {
private:
    // **CONFLICT PREVENTION**: ETF family classifications
    static const std::unordered_set<std::string> LONG_ETFS;
    static const std::unordered_set<std::string> INVERSE_ETFS;
    
    // **STATE TRACKING**: Current portfolio state for conflict detection
    std::unordered_map<std::string, double> current_positions_;
    std::unordered_map<std::string, double> pending_positions_;
    
    // **FREQUENCY CONTROL**: Order frequency management
    int orders_this_bar_;
    int max_orders_per_bar_;
    
    // **CONFLICT DETECTION**: Check if positions would conflict
    bool would_create_conflict(const std::unordered_map<std::string, double>& positions) const;
    
    // **POSITION ANALYSIS**: Analyze position implications
    struct PositionAnalysis {
        bool has_long_etf = false;
        bool has_inverse_etf = false;
        bool has_short_qqq = false;
        std::vector<std::string> long_positions;
        std::vector<std::string> short_positions;
        std::vector<std::string> inverse_positions;
    };
    
    PositionAnalysis analyze_positions(const std::unordered_map<std::string, double>& positions) const;
    
    // **CONFLICT RESOLUTION**: Attempt to resolve conflicts by modifying weights
    CoordinationDecision resolve_conflict(const AllocationRequest& request);
    
public:
    PositionCoordinator(int max_orders_per_bar = 1);
    
    // **MAIN COORDINATION**: Coordinate allocation decisions to prevent conflicts
    std::vector<CoordinationDecision> coordinate_allocations(
        const std::vector<AllocationRequest>& requests,
        const Portfolio& current_portfolio,
        const SymbolTable& ST
    );
    
    // **BAR RESET**: Reset per-bar state for new bar
    void reset_bar();
    
    // **POSITION SYNC**: Sync current positions from portfolio
    void sync_positions(const Portfolio& portfolio, const SymbolTable& ST);
    
    // **CONFIGURATION**: Set coordination parameters
    void set_max_orders_per_bar(int max_orders) { max_orders_per_bar_ = max_orders; }
    
    // **STATISTICS**: Get coordination statistics
    struct CoordinationStats {
        int total_requests = 0;
        int approved = 0;
        int rejected_conflict = 0;
        int rejected_frequency = 0;
        int modified = 0;
    };
    
    CoordinationStats get_stats() const { return stats_; }
    void reset_stats() { stats_ = CoordinationStats{}; }
    
private:
    CoordinationStats stats_;
};

// **ALLOCATION DECISION STRUCTURE**: Define allocation decision structure
struct AllocationDecision {
    std::string instrument;
    double target_weight;
    double confidence;
    std::string reason;
};

// **ALLOCATION DECISION CONVERTER**: Convert strategy decisions to requests
std::vector<AllocationRequest> convert_allocation_decisions(
    const std::vector<AllocationDecision>& decisions,
    const std::string& strategy_name,
    const std::string& chain_id
);

} // namespace sentio
