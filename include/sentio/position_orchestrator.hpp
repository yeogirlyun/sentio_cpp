#pragma once
#include "position_guardian.hpp"
#include "family_mapper.hpp"
#include "side.hpp"
#include "core.hpp"
#include "audit.hpp"
#include <string>
#include <memory>

namespace sentio {

// Integration layer between strategies and the position guardian
class PositionOrchestrator {
public:
    explicit PositionOrchestrator(const std::string& account = "sentio:primary");
    
    // Main entry point for strategy signals
    void process_strategy_signal(const std::string& strategy_id,
                                const std::string& symbol,
                                ::sentio::Side target_side,
                                double target_qty,
                                const std::string& preferred_symbol = "");
    
    // Sync with current portfolio state
    void sync_portfolio(const Portfolio& portfolio, const SymbolTable& ST);
    
    // Get current family exposure
    PositionSnapshot get_family_exposure(const std::string& symbol) const;
    
    // Configuration
    void set_policy(const Policy& policy) { policy_ = policy; }
    const Policy& get_policy() const { return policy_; }

private:
    std::string account_;
    FamilyMapper mapper_;
    PositionGuardian guardian_;
    Policy policy_;
    
    // Convert Sentio Side to guardian PositionSide
    static PositionSide convert_side(::sentio::Side side);
    
    // Choose preferred symbol based on signal strength and family
    std::string choose_preferred_symbol(const std::string& symbol, PositionSide side, double strength) const;
};

} // namespace sentio
