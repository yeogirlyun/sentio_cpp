#include "sentio/position_coordinator.hpp"
#include "sentio/base_strategy.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace sentio {

// **ETF CLASSIFICATIONS**: Updated for PSQ -> SHORT QQQ architecture
const std::unordered_set<std::string> PositionCoordinator::LONG_ETFS = {"QQQ", "TQQQ"};
const std::unordered_set<std::string> PositionCoordinator::INVERSE_ETFS = {"SQQQ"};

PositionCoordinator::PositionCoordinator(int max_orders_per_bar)
    : orders_this_bar_(0), max_orders_per_bar_(max_orders_per_bar) {
}

void PositionCoordinator::reset_bar() {
    orders_this_bar_ = 0;
    pending_positions_.clear();
}

void PositionCoordinator::sync_positions(const Portfolio& portfolio, const SymbolTable& ST) {
    current_positions_.clear();
    
    for (size_t i = 0; i < portfolio.positions.size(); ++i) {
        const auto& pos = portfolio.positions[i];
        if (std::abs(pos.qty) > 1e-6) {
            const std::string& symbol = ST.get_symbol(i);
            current_positions_[symbol] = pos.qty;
        }
    }
}

PositionCoordinator::PositionAnalysis PositionCoordinator::analyze_positions(
    const std::unordered_map<std::string, double>& positions) const {
    
    PositionAnalysis analysis;
    
    for (const auto& [symbol, qty] : positions) {
        if (std::abs(qty) > 1e-6) {
            if (LONG_ETFS.count(symbol)) {
                if (qty > 0) {
                    analysis.has_long_etf = true;
                    analysis.long_positions.push_back(symbol + "(+" + std::to_string((int)qty) + ")");
                } else {
                    analysis.has_short_qqq = true;
                    analysis.short_positions.push_back("SHORT " + symbol + "(" + std::to_string((int)qty) + ")");
                }
            }
            if (INVERSE_ETFS.count(symbol)) {
                analysis.has_inverse_etf = true;
                analysis.inverse_positions.push_back(symbol + "(" + std::to_string((int)qty) + ")");
            }
        }
    }
    
    return analysis;
}

bool PositionCoordinator::would_create_conflict(
    const std::unordered_map<std::string, double>& positions) const {
    
    auto analysis = analyze_positions(positions);
    
    // **CONFLICT RULES**:
    // 1. Long ETF conflicts with Inverse ETF or SHORT QQQ
    // 2. SHORT QQQ conflicts with Long ETF
    // 3. Inverse ETF conflicts with Long ETF
    return (analysis.has_long_etf && (analysis.has_inverse_etf || analysis.has_short_qqq)) ||
           (analysis.has_short_qqq && analysis.has_long_etf);
}

CoordinationDecision PositionCoordinator::resolve_conflict(const AllocationRequest& request) {
    CoordinationDecision decision;
    decision.instrument = request.instrument;
    decision.original_weight = request.target_weight;
    decision.approved_weight = 0.0; // Default to rejection
    
    // **CONFLICT RESOLUTION STRATEGIES**:
    
    // 1. **ZERO OUT CONFLICTING POSITION**: Set weight to zero to avoid conflict
    decision.result = CoordinationResult::MODIFIED;
    decision.approved_weight = 0.0;
    decision.reason = "CONFLICT_PREVENTION_ZERO";
    
    // Build conflict details
    auto current_analysis = analyze_positions(current_positions_);
    auto pending_analysis = analyze_positions(pending_positions_);
    
    std::string conflict_details = "CONFLICT: ";
    if (!current_analysis.long_positions.empty()) {
        conflict_details += "Current Long: ";
        for (size_t i = 0; i < current_analysis.long_positions.size(); ++i) {
            if (i > 0) conflict_details += ", ";
            conflict_details += current_analysis.long_positions[i];
        }
    }
    if (!current_analysis.short_positions.empty()) {
        if (!current_analysis.long_positions.empty()) conflict_details += "; ";
        conflict_details += "Current Short: ";
        for (size_t i = 0; i < current_analysis.short_positions.size(); ++i) {
            if (i > 0) conflict_details += ", ";
            conflict_details += current_analysis.short_positions[i];
        }
    }
    if (!current_analysis.inverse_positions.empty()) {
        if (!current_analysis.long_positions.empty() || !current_analysis.short_positions.empty()) {
            conflict_details += "; ";
        }
        conflict_details += "Current Inverse: ";
        for (size_t i = 0; i < current_analysis.inverse_positions.size(); ++i) {
            if (i > 0) conflict_details += ", ";
            conflict_details += current_analysis.inverse_positions[i];
        }
    }
    
    conflict_details += " | Requested: " + request.instrument + "(" + std::to_string(request.target_weight) + ")";
    decision.conflict_details = conflict_details;
    
    return decision;
}

std::vector<CoordinationDecision> PositionCoordinator::coordinate_allocations(
    const std::vector<AllocationRequest>& requests,
    const Portfolio& current_portfolio,
    const SymbolTable& ST) {
    
    std::vector<CoordinationDecision> decisions;
    decisions.reserve(requests.size());
    
    // **SYNC CURRENT POSITIONS**
    sync_positions(current_portfolio, ST);
    pending_positions_ = current_positions_; // Start with current positions
    
    // **PROCESS EACH REQUEST**
    for (const auto& request : requests) {
        stats_.total_requests++;
        
        CoordinationDecision decision;
        decision.instrument = request.instrument;
        decision.original_weight = request.target_weight;
        decision.approved_weight = request.target_weight; // Default to approval
        
        // **FREQUENCY CONTROL**: Check order frequency limit
        if (orders_this_bar_ >= max_orders_per_bar_) {
            decision.result = CoordinationResult::REJECTED_FREQUENCY;
            decision.approved_weight = 0.0;
            decision.reason = "FREQUENCY_LIMIT_EXCEEDED";
            decision.conflict_details = "Max " + std::to_string(max_orders_per_bar_) + " orders per bar";
            stats_.rejected_frequency++;
            decisions.push_back(decision);
            continue;
        }
        
        // **SIMULATE POSITION CHANGE**: Calculate what positions would be after this request
        std::unordered_map<std::string, double> simulated_positions = pending_positions_;
        
        // Apply the requested position change
        // Note: This is simplified - in reality we'd need to calculate the actual quantity change
        // based on the target weight, current portfolio value, and instrument price
        if (std::abs(request.target_weight) > 1e-6) {
            if (request.target_weight > 0) {
                simulated_positions[request.instrument] = std::abs(request.target_weight) * 1000; // Simplified
            } else {
                simulated_positions[request.instrument] = request.target_weight * 1000; // Negative for short
            }
        } else {
            simulated_positions.erase(request.instrument); // Zero weight = close position
        }
        
        // **CONFLICT DETECTION**: Check if this would create conflicts
        if (would_create_conflict(simulated_positions)) {
            // **CONFLICT RESOLUTION**: Attempt to resolve
            decision = resolve_conflict(request);
            stats_.rejected_conflict++;
        } else {
            // **APPROVAL**: No conflict detected
            decision.result = CoordinationResult::APPROVED;
            decision.reason = "APPROVED_NO_CONFLICT";
            
            // Update pending positions for next request
            pending_positions_ = simulated_positions;
            orders_this_bar_++;
            stats_.approved++;
        }
        
        decisions.push_back(decision);
    }
    
    return decisions;
}

// **CONVERTER FUNCTION**: Convert strategy allocation decisions to coordination requests
std::vector<AllocationRequest> convert_allocation_decisions(
    const std::vector<AllocationDecision>& decisions,
    const std::string& strategy_name,
    const std::string& chain_id) {
    
    std::vector<AllocationRequest> requests;
    requests.reserve(decisions.size());
    
    for (const auto& decision : decisions) {
        AllocationRequest request;
        request.strategy_name = strategy_name;
        request.instrument = decision.instrument;
        request.target_weight = decision.target_weight;
        request.confidence = decision.confidence;
        request.reason = decision.reason;
        request.chain_id = chain_id;
        
        requests.push_back(request);
    }
    
    return requests;
}

} // namespace sentio
