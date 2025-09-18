#include "sentio/universal_position_coordinator.hpp"
#include <algorithm>

namespace sentio {

UniversalPositionCoordinator::UniversalPositionCoordinator() {}

void UniversalPositionCoordinator::reset_bar(int64_t timestamp) {
    // Reset for new bar - no state to track since we simplified the coordinator
    (void)timestamp; // Suppress unused parameter warning
}

std::vector<CoordinationDecision> UniversalPositionCoordinator::coordinate(
    const std::vector<AllocationDecision>& allocations,
    const Portfolio& portfolio,
    const SymbolTable& ST,
    int64_t current_timestamp,
    const StrategyProfiler::StrategyProfile& profile) {
    
    std::vector<CoordinationDecision> results;

    // If allocations are empty, it's a signal to hold or close.
    // The coordinator's job here is to generate closing orders if needed.
    if (allocations.empty()) {
        if (trades_per_timestamp_[current_timestamp].empty()) { // Only allow one closing transaction
            for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
                if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
                    const std::string& symbol = ST.get_symbol(sid);
                    results.push_back({{symbol, 0.0, "Close position, no active signal"}, CoordinationResult::APPROVED, "Closing position."});
                    // Since we are closing everything, we can add multiple close orders in one go.
                }
            }
            if (!results.empty()) {
                trades_per_timestamp_[current_timestamp].push_back({
                    current_timestamp, "CLOSE_ALL", 0.0
                });
            }
        }
        return results;
    }
    
    // We only process one new allocation request per bar to enforce the one-trade-per-bar rule.
    const auto& primary_decision = allocations[0];

    // PRINCIPLE 3: ONE TRADE PER BAR
    if (!trades_per_timestamp_[current_timestamp].empty()) {
        results.push_back({primary_decision, CoordinationResult::REJECTED_FREQUENCY, "One transaction per bar limit reached."});
        return results;
    }

    // PRINCIPLE 2: NO CONFLICTING POSITIONS
    if (would_create_conflict(primary_decision.instrument, portfolio, ST)) {
        // A conflict exists. The ONLY valid action is to close existing positions.
        // The new trade is REJECTED for this bar.
        results.push_back({primary_decision, CoordinationResult::REJECTED_CONFLICT, "Existing position conflicts. Closing first."});
        
        // Generate closing orders for ALL existing positions to ensure a clean slate.
        for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
            if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
                const std::string& symbol_to_close = ST.get_symbol(sid);
                results.push_back({{symbol_to_close, 0.0, "Closing conflicting position"}, CoordinationResult::APPROVED, "Conflict Resolution: Flattening."});
            }
        }
    } else {
        // No conflict detected. Approve the primary decision.
        results.push_back({primary_decision, CoordinationResult::APPROVED, "Approved by coordinator."});
    }

    // Record the trade if at least one trade (open or close) was approved.
    if (!results.empty()) {
        trades_per_timestamp_[current_timestamp].push_back({
            current_timestamp, 
            primary_decision.instrument,
            std::abs(primary_decision.target_weight)
        });
    }
    
    return results;
}

bool UniversalPositionCoordinator::would_create_conflict(
    const std::string& instrument,
    const Portfolio& portfolio,
    const SymbolTable& ST) const {
    
    bool wants_long = LONG_ETFS.count(instrument);
    bool wants_inverse = INVERSE_ETFS.count(instrument);
    
    if (!wants_long && !wants_inverse) return false;
    
    // Check existing positions
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
            const std::string& symbol = ST.get_symbol(sid);
            bool has_long = LONG_ETFS.count(symbol);
            bool has_inverse = INVERSE_ETFS.count(symbol);
            
            if ((wants_long && has_inverse) || (wants_inverse && has_long)) {
                return true;
            }
        }
    }
    
    return false;
}

bool UniversalPositionCoordinator::can_trade_at_timestamp(
    const std::string& instrument,
    int64_t timestamp,
    const StrategyProfiler::StrategyProfile& profile) {
    
    auto it = trades_per_timestamp_.find(timestamp);
    if (it == trades_per_timestamp_.end()) {
        return true;  // No trades yet at this timestamp
    }
    
    // For aggressive strategies, allow multiple trades per bar
    if (profile.style == TradingStyle::AGGRESSIVE) {
        // Allow up to 3 trades per bar for aggressive strategies
        return it->second.size() < 3;
    }
    
    // For conservative strategies, strict one trade per bar
    for (const auto& record : it->second) {
        if (record.instrument == instrument) {
            return false;  // Already traded this instrument
        }
    }
    
    return it->second.empty();  // Allow if no trades yet
}


} // namespace sentio
