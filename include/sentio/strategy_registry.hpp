#pragma once

#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace sentio {

class BaseStrategy;

/**
 * Strategy registry for dynamic strategy loading from configuration
 */
class StrategyRegistry {
public:
    using StrategyFactory = std::function<std::unique_ptr<BaseStrategy>()>;
    
    /**
     * Register a strategy with its factory function
     * @param name Strategy name
     * @param factory Factory function to create strategy instance
     */
    static void register_strategy(const std::string& name, StrategyFactory factory);
    
    /**
     * Create a strategy instance by name
     * @param name Strategy name
     * @return unique_ptr to strategy instance or nullptr if not found
     */
    static std::unique_ptr<BaseStrategy> create_strategy(const std::string& name);
    
    /**
     * Load strategies from configuration file
     * @param config_path Path to strategies.json configuration file
     * @return true if loading succeeded
     */
    static bool load_from_config(const std::string& config_path = "configs/strategies.json");
    
    /**
     * Get list of registered strategy names
     * @return vector of strategy names
     */
    static std::vector<std::string> get_registered_strategies();
    
    /**
     * Check if a strategy is registered
     * @param name Strategy name
     * @return true if strategy is registered
     */
    static bool is_registered(const std::string& name);

private:
    static std::unordered_map<std::string, StrategyFactory>& get_factories();
};

} // namespace sentio
