#include "sentio/strategy_registry.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/strategy_ire.hpp"
#include "sentio/strategy_bollinger_squeeze_breakout.hpp"
#include "sentio/strategy_hybrid_ppo.hpp"
#include "sentio/strategy_kochi_ppo.hpp"
#include "sentio/strategy_market_making.hpp"
#include "sentio/strategy_momentum_volume.hpp"
#include "sentio/strategy_opening_range_breakout.hpp"
#include "sentio/strategy_order_flow_imbalance.hpp"
#include "sentio/strategy_order_flow_scalping.hpp"
#include "sentio/strategy_sma_cross.hpp"
#include "sentio/strategy_tfa.hpp"
#include "sentio/strategy_transformer_ts.hpp"
#include "sentio/strategy_vwap_reversion.hpp"
#include <fstream>
#include <iostream>

namespace sentio {

std::unordered_map<std::string, StrategyRegistry::StrategyFactory>& StrategyRegistry::get_factories() {
    static std::unordered_map<std::string, StrategyRegistry::StrategyFactory> factories;
    return factories;
}

void StrategyRegistry::register_strategy(const std::string& name, StrategyFactory factory) {
    get_factories()[name] = factory;
}

std::unique_ptr<BaseStrategy> StrategyRegistry::create_strategy(const std::string& name) {
    auto& factories = get_factories();
    auto it = factories.find(name);
    if (it != factories.end()) {
        return it->second();
    }
    return nullptr;
}

bool StrategyRegistry::load_from_config(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open strategy config file: " << config_path << std::endl;
        return false;
    }
    
    nlohmann::json config;
    try {
        file >> config;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse strategy config: " << e.what() << std::endl;
        return false;
    }
    
    // Register all strategies
    register_strategy("IRE", []() -> std::unique_ptr<BaseStrategy> {
        return std::make_unique<IREStrategy>();
    });
    
    register_strategy("BollingerSqueezeBreakout", []() -> std::unique_ptr<BaseStrategy> {
        return std::make_unique<BollingerSqueezeBreakoutStrategy>();
    });
    
    register_strategy("HybridPPO", []() -> std::unique_ptr<BaseStrategy> {
        return std::make_unique<HybridPPOStrategy>();
    });
    
    register_strategy("KochiPPO", []() -> std::unique_ptr<BaseStrategy> {
        return std::make_unique<KochiPPOStrategy>();
    });
    
    register_strategy("MarketMaking", []() -> std::unique_ptr<BaseStrategy> {
        return std::make_unique<MarketMakingStrategy>();
    });
    
    register_strategy("MomentumVolume", []() -> std::unique_ptr<BaseStrategy> {
        return std::make_unique<MomentumVolumeProfileStrategy>();
    });
    
    register_strategy("OpeningRangeBreakout", []() -> std::unique_ptr<BaseStrategy> {
        return std::make_unique<OpeningRangeBreakoutStrategy>();
    });
    
    register_strategy("OrderFlowImbalance", []() -> std::unique_ptr<BaseStrategy> {
        return std::make_unique<OrderFlowImbalanceStrategy>();
    });
    
    register_strategy("OrderFlowScalping", []() -> std::unique_ptr<BaseStrategy> {
        return std::make_unique<OrderFlowScalpingStrategy>();
    });
    
    // Note: SMACrossStrategy inherits from IStrategy, not BaseStrategy
    // register_strategy("SMACross", []() -> std::unique_ptr<BaseStrategy> {
    //     return std::make_unique<SMACrossStrategy>();
    // });
    
    register_strategy("TFA", []() -> std::unique_ptr<BaseStrategy> {
        return std::make_unique<TFAStrategy>();
    });
    
    // Note: TransformerSignalStrategyTS inherits from IStrategy, not BaseStrategy
    // register_strategy("TransformerTS", []() -> std::unique_ptr<BaseStrategy> {
    //     return std::make_unique<TransformerSignalStrategyTS>();
    // });
    
    register_strategy("VWAPReversion", []() -> std::unique_ptr<BaseStrategy> {
        return std::make_unique<VWAPReversionStrategy>();
    });
    
    std::cout << "Registered " << get_factories().size() << " strategies from config" << std::endl;
    return true;
}

std::vector<std::string> StrategyRegistry::get_registered_strategies() {
    std::vector<std::string> names;
    for (const auto& pair : get_factories()) {
        names.push_back(pair.first);
    }
    return names;
}

bool StrategyRegistry::is_registered(const std::string& name) {
    return get_factories().find(name) != get_factories().end();
}

} // namespace sentio
