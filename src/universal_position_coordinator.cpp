#include "sentio/universal_position_coordinator.hpp"
#include <algorithm>
#include <iostream>

namespace sentio {

UniversalPositionCoordinator::UniversalPositionCoordinator() {
}

void UniversalPositionCoordinator::reset_bar(int64_t timestamp) {
    current_bar_timestamp_ = timestamp;
    opening_trades_this_bar_ = 0;
}

std::vector<CoordinationDecision> UniversalPositionCoordinator::coordinate(
    const std::vector<AllocationDecision>& allocations,
    const Portfolio& portfolio,
    const SymbolTable& ST,
    int64_t current_timestamp,
    const StrategyProfiler::StrategyProfile& profile) {
    
    std::vector<CoordinationDecision> results;
    
    
    // STEP 1: Atomic conflict resolution - capture portfolio state first
    std::vector<std::pair<size_t, double>> conflicted_positions;
    bool has_conflicts = false;
    
    // Atomic check and capture of conflicted positions
    {
        bool has_long = false;
        bool has_inverse = false;
        
        for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
            if (has_position(portfolio.positions[sid])) {
                const std::string& symbol = ST.get_symbol(sid);
                
                if (LONG_ETFS.count(symbol)) {
                    has_long = true;
                }
                if (INVERSE_ETFS.count(symbol)) {
                    has_inverse = true;
                }
                
                // Capture all positions for potential closure
                conflicted_positions.push_back({sid, portfolio.positions[sid].qty});
            }
        }
        
        has_conflicts = (has_long && has_inverse);
    }
    
    if (has_conflicts) {
        // Generate closing orders for ALL captured positions atomically
        for (const auto& [sid, qty] : conflicted_positions) {
            const std::string& symbol = ST.get_symbol(sid);
            results.push_back({
                {symbol, 0.0, "IMMEDIATE CONFLICT RESOLUTION"},
                CoordinationResult::APPROVED,
                "Closing position to resolve portfolio conflict."
            });
        }
        
        // STEP 2: Reject ALL incoming allocations for this bar to prevent worsening the state
        for (const auto& allocation : allocations) {
            results.push_back({
                allocation,
                CoordinationResult::REJECTED_CONFLICT,
                "Rejected: Portfolio conflict resolution is active."
            });
        }
        
        return results; // Immediately return with resolution orders
    }
    
    // PHASE 2: NORMAL OPERATION (no conflicts exist)
    
    // Process empty allocations
    if (allocations.empty()) {
        return results;
    }
    
    // Process each allocation
    for (const auto& allocation : allocations) {
        bool is_closing = (allocation.target_weight == 0.0);
        
        if (is_closing) {
            // Always approve closing trades
            results.push_back({
                allocation,
                CoordinationResult::APPROVED,
                "Closing trade approved"
            });
        } else {
            // Opening trade - apply strict checks
            
            // Verify no conflict would be created
            if (would_create_conflict(allocation.instrument, portfolio, ST)) {
                results.push_back({
                    allocation,
                    CoordinationResult::REJECTED_CONFLICT,
                    "Would create conflict"
                });
                continue;
            }
            
            // Enforce one opening trade per bar
            if (opening_trades_this_bar_ >= 1) {
                results.push_back({
                    allocation,
                    CoordinationResult::REJECTED_FREQUENCY,
                    "One opening per bar limit"
                });
                continue;
            }
            
            // Approve the trade
            results.push_back({
                allocation,
                CoordinationResult::APPROVED,
                "Opening trade approved"
            });
            
            opening_trades_this_bar_++;
        }
    }
    
    return results;
}

bool UniversalPositionCoordinator::check_portfolio_conflicts(
    const Portfolio& portfolio,
    const SymbolTable& ST) const {
    
    bool has_long = false;
    bool has_inverse = false;
    
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        if (has_position(portfolio.positions[sid])) {
            const std::string& symbol = ST.get_symbol(sid);
            
            if (LONG_ETFS.count(symbol)) {
                has_long = true;
            }
            if (INVERSE_ETFS.count(symbol)) {
                has_inverse = true;
            }
        }
    }
    
    return (has_long && has_inverse);
}

bool UniversalPositionCoordinator::would_create_conflict(
    const std::string& instrument,
    const Portfolio& portfolio,
    const SymbolTable& ST) const {
    
    bool wants_long = LONG_ETFS.count(instrument) > 0;
    bool wants_inverse = INVERSE_ETFS.count(instrument) > 0;
    
    if (!wants_long && !wants_inverse) {
        return false;
    }
    
    // CORRECTED: Only check for DIRECTIONAL conflicts (long vs inverse)
    // Multiple inverse ETFs (PSQ+SQQQ) are ALLOWED - same direction, optimal allocation
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        if (has_position(portfolio.positions[sid])) {
            const std::string& symbol = ST.get_symbol(sid);
            bool has_long = LONG_ETFS.count(symbol) > 0;
            bool has_inverse = INVERSE_ETFS.count(symbol) > 0;
            
            // Only conflict if OPPOSITE directions: long vs inverse
            if ((wants_long && has_inverse) || (wants_inverse && has_long)) {
                return true;
            }
        }
    }
    
    return false;
}


bool UniversalPositionCoordinator::portfolio_has_positions(const Portfolio& portfolio) const {
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        if (has_position(portfolio.positions[sid])) {
            return true;
        }
    }
    return false;
}


} // namespace sentio