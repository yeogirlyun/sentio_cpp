#include "sentio/position_orchestrator.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/audit.hpp"
#include <iostream>

namespace sentio {

PositionOrchestrator::PositionOrchestrator(const std::string& account) 
    : account_(account)
    , mapper_(FamilyMapper::Map{
        {"QQQ*", {"QQQ", "TQQQ", "SQQQ"}}, // PSQ removed
        {"SPY*", {"SPY", "UPRO", "SPXU"}},
        {"BTC*", {"BTC", "BITO", "BITX", "BITI"}},
        {"TSLA*", {"TSLA", "TSLL", "TSLS"}}
      })
    , guardian_(mapper_)
{
    // Default policy: no conflicts, reasonable limits
    policy_.allow_conflicts = false;
    policy_.max_gross_shares = 1'000'000;
    policy_.min_flip_bps = 0.0;
    policy_.cooldown_ms = 500; // avoid churn
}

void PositionOrchestrator::process_strategy_signal(const std::string& strategy_id,
                                                  const std::string& symbol,
                                                  ::sentio::Side target_side,
                                                  double target_qty,
                                                  const std::string& preferred_symbol) {
    
    if (target_qty <= 0) return; // No action needed
    
    Desire desire;
    desire.target_side = convert_side(target_side);
    desire.target_qty = target_qty;
    desire.preferred_symbol = preferred_symbol.empty() ? 
        choose_preferred_symbol(symbol, convert_side(target_side), target_qty) : preferred_symbol;
    
    auto plan_opt = guardian_.plan(account_, symbol, desire, policy_);
    if (!plan_opt) {
        std::cerr << "PositionOrchestrator: No plan generated for " << strategy_id 
                  << " " << symbol << " " << (target_side == ::sentio::Side::Buy ? "BUY" : "SELL") << " " << target_qty << std::endl;
        return; // no-op or cooldown
    }
    
    const auto& plan = *plan_opt;
    
    std::cout << "PositionOrchestrator: Generated plan for " << strategy_id 
              << " family=" << plan.key.family 
              << " legs=" << plan.legs.size() << std::endl;
    
    // TODO: Integrate with existing router
    // For now, just commit the plan
    // In production, this would:
    // 1. Create router batch with strict leg ordering
    // 2. Send to router
    // 3. Only commit on success
    
    guardian_.commit(plan);
    
    // Log the plan details
    for (const auto& leg : plan.legs) {
        std::cout << "  " << leg.reason << ": " << leg.symbol 
                  << " " << to_string(leg.side) << " " << leg.qty << std::endl;
    }
}

void PositionOrchestrator::sync_portfolio(const Portfolio& portfolio, const SymbolTable& ST) {
    // Convert Sentio portfolio to guardian format
    std::vector<Position> positions = portfolio.positions;
    
    // TODO: Convert open orders if available
    std::vector<std::string> open_orders;
    
    guardian_.sync_from_broker(account_, positions, open_orders);
}

PositionSnapshot PositionOrchestrator::get_family_exposure(const std::string& symbol) const {
    ExposureKey key{account_, mapper_.family_for(symbol)};
    return guardian_.snapshot(key);
}

PositionSide PositionOrchestrator::convert_side(::sentio::Side side) {
    switch (side) {
        case ::sentio::Side::Buy: return PositionSide::Long;
        case ::sentio::Side::Sell: return PositionSide::Short;
        default: return PositionSide::Flat;
    }
}

std::string PositionOrchestrator::choose_preferred_symbol(const std::string& symbol, 
                                                         PositionSide side, 
                                                         double strength) const {
    std::string family = mapper_.family_for(symbol);
    
    if (family == "QQQ*") {
        if (side == PositionSide::Long) {
            // For long positions, choose TQQQ for strong signals, QQQ for moderate
            return (strength > 0.7) ? "TQQQ" : "QQQ";
        } else {
            // For short positions, choose SQQQ for strong signals, SHORT QQQ for moderate  
            return (strength > 0.7) ? "SQQQ" : "QQQ"; // QQQ will be shorted for moderate sells
        }
    }
    
    // Default to original symbol for unknown families
    return symbol;
}

} // namespace sentio
