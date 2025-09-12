#include "sentio/base_strategy.hpp"
#include "sentio/strategy_ire.hpp"
#include "sentio/strategy_bollinger_squeeze_breakout.hpp"
// Removed unused strategy: hybrid_ppo
#include "sentio/strategy_kochi_ppo.hpp"
#include "sentio/strategy_market_making.hpp"
#include "sentio/strategy_momentum_volume.hpp"
#include "sentio/strategy_opening_range_breakout.hpp"
#include "sentio/strategy_order_flow_imbalance.hpp"
#include "sentio/strategy_order_flow_scalping.hpp"
#include "sentio/strategy_tfa.hpp"
#include "sentio/strategy_vwap_reversion.hpp"
#include "sentio/rsi_strategy.hpp"

namespace sentio {

/**
 * Initialize all strategies in the StrategyFactory
 * This replaces the StrategyRegistry system
 * 
 * Note: Individual strategy files use REGISTER_STRATEGY macro
 * which automatically registers strategies, so manual registration
 * is no longer needed here.
 */
bool initialize_strategies() {
    auto& factory = StrategyFactory::instance();
    
    // All strategies are now automatically registered via REGISTER_STRATEGY macro
    // in their respective source files. No manual registration needed.
    
    std::cout << "Registered " << factory.get_available_strategies().size() << " strategies" << std::endl;
    return true;
}

} // namespace sentio
