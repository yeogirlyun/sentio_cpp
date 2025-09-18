#pragma once
#include "sentio/core.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/adaptive_allocation_manager.hpp"
#include <unordered_set>

namespace sentio {

/**
 * @brief CircuitBreaker provides emergency protection against system violations
 * 
 * This component:
 * 1. Detects conflicting positions (long + inverse ETFs)
 * 2. Triggers emergency closure when violations persist
 * 3. Prevents further trading when tripped
 * 4. Provides failsafe protection against Golden Rule bypasses
 */
class CircuitBreaker {
private:
    int consecutive_violations_ = 0;
    bool tripped_ = false;
    int64_t trip_timestamp_ = 0;
    std::string trip_reason_;
    
    // ETF classifications
    const std::unordered_set<std::string> LONG_ETFS = {"QQQ", "TQQQ"};
    const std::unordered_set<std::string> INVERSE_ETFS = {"PSQ", "SQQQ"};
    
    // Configuration
    static constexpr int MAX_CONSECUTIVE_VIOLATIONS = 3;
    static constexpr double MIN_POSITION_THRESHOLD = 1e-6;
    
    bool has_conflicting_positions(const Portfolio& portfolio, const SymbolTable& ST) const;
    void log_violation(const std::string& reason, int64_t timestamp);
    
public:
    CircuitBreaker();
    
    /**
     * @brief Check portfolio integrity and trip breaker if violations detected
     * @param portfolio Current portfolio state
     * @param ST Symbol table for position lookup
     * @param timestamp Current timestamp
     * @return true if portfolio is clean, false if violations detected
     */
    bool check_portfolio_integrity(const Portfolio& portfolio, 
                                  const SymbolTable& ST,
                                  int64_t timestamp);
    
    /**
     * @brief Check if circuit breaker is tripped
     * @return true if breaker is tripped and trading should stop
     */
    bool is_tripped() const;
    
    /**
     * @brief Get emergency closure orders when breaker is tripped
     * @param portfolio Current portfolio
     * @param ST Symbol table
     * @return Single emergency close order (one trade per bar)
     */
    std::vector<AllocationDecision> get_emergency_closure(const Portfolio& portfolio, 
                                                         const SymbolTable& ST) const;
    
    /**
     * @brief Reset breaker (use with caution)
     */
    void reset();
    
    /**
     * @brief Get breaker status for diagnostics
     */
    struct Status {
        bool tripped;
        int consecutive_violations;
        int64_t trip_timestamp;
        std::string trip_reason;
    };
    
    Status get_status() const;
};

} // namespace sentio
