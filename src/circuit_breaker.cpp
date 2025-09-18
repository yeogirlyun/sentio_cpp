#include "sentio/circuit_breaker.hpp"
#include <iostream>
#include <cmath>

namespace sentio {

CircuitBreaker::CircuitBreaker() {}

bool CircuitBreaker::has_conflicting_positions(const Portfolio& portfolio, const SymbolTable& ST) const {
    bool has_long = false;
    bool has_inverse = false;
    
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        if (std::abs(portfolio.positions[sid].qty) > MIN_POSITION_THRESHOLD) {
            const std::string& symbol = ST.get_symbol(sid);
            
            if (LONG_ETFS.count(symbol)) {
                has_long = true;
            }
            if (INVERSE_ETFS.count(symbol)) {
                has_inverse = true;
            }
        }
    }
    
    return has_long && has_inverse;
}

void CircuitBreaker::log_violation(const std::string& reason, int64_t timestamp) {
    // Log violation for internal tracking
    (void)reason; (void)timestamp; // Suppress unused parameter warnings
}

bool CircuitBreaker::check_portfolio_integrity(const Portfolio& portfolio, 
                                              const SymbolTable& ST,
                                              int64_t timestamp) {
    // If already tripped, don't check further
    if (tripped_) {
        return false;
    }
    
    // Check for conflicting positions
    if (has_conflicting_positions(portfolio, ST)) {
        consecutive_violations_++;
        log_violation("Conflicting directional positions detected", timestamp);
        
        if (consecutive_violations_ >= MAX_CONSECUTIVE_VIOLATIONS) {
            tripped_ = true;
            trip_timestamp_ = timestamp;
            trip_reason_ = "Consecutive conflicting positions exceeded threshold";
        }
        
        return false;
    } else {
        // Reset violation counter on clean portfolio
        if (consecutive_violations_ > 0) {
            consecutive_violations_ = 0;
        }
        return true;
    }
}

bool CircuitBreaker::is_tripped() const {
    return tripped_;
}

std::vector<AllocationDecision> CircuitBreaker::get_emergency_closure(const Portfolio& portfolio, 
                                                                     const SymbolTable& ST) const {
    std::vector<AllocationDecision> emergency_orders;
    
    if (!tripped_) {
        return emergency_orders;
    }
    
    // Find the largest position to close first (one trade per bar)
    double largest_exposure = 0.0;
    std::string largest_symbol;
    
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        if (std::abs(portfolio.positions[sid].qty) > MIN_POSITION_THRESHOLD) {
            const std::string& symbol = ST.get_symbol(sid);
            double exposure = std::abs(portfolio.positions[sid].qty);
            
            if (exposure > largest_exposure) {
                largest_exposure = exposure;
                largest_symbol = symbol;
            }
        }
    }
    
    if (!largest_symbol.empty()) {
        emergency_orders.push_back({
            largest_symbol, 
            0.0, 
            "EMERGENCY CIRCUIT BREAKER CLOSURE - " + trip_reason_
        });
    }
    
    return emergency_orders;
}

void CircuitBreaker::reset() {
    tripped_ = false;
    consecutive_violations_ = 0;
    trip_timestamp_ = 0;
    trip_reason_.clear();
}

CircuitBreaker::Status CircuitBreaker::get_status() const {
    return {
        tripped_,
        consecutive_violations_,
        trip_timestamp_,
        trip_reason_
    };
}

} // namespace sentio
