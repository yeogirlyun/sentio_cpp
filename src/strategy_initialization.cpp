#include "sentio/base_strategy.hpp"
// REMOVED: strategy_ire.hpp - unused legacy strategy
// REMOVED: strategy_bollinger_squeeze_breakout.hpp - unused legacy strategy
// REMOVED: strategy_kochi_ppo.hpp - unused legacy strategy
#include "sentio/strategy_tfa.hpp"
#include "sentio/strategy_signal_or.hpp"
#include "sentio/strategy_transformer.hpp"
// REMOVED: rsi_strategy.hpp - unused legacy strategy

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
