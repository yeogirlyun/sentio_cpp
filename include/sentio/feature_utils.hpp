#pragma once

#include <string>
#include <vector>
#include <algorithm>

namespace sentio {

/**
 * Utility functions for feature operations
 */
class FeatureUtils {
public:
    /**
     * Find strategy data by symbol in a map-like container
     * @param strategy_data The strategy data container
     * @param symbol The symbol to find
     * @return iterator to the found element or end() if not found
     */
    template<typename Container>
    static auto find_strategy_data(Container& strategy_data, const std::string& symbol) {
        return std::find_if(strategy_data.begin(), strategy_data.end(),
            [&symbol](const auto& pair) {
                return pair.first == symbol;
            });
    }
    
    /**
     * Check if strategy data exists for a symbol
     * @param strategy_data The strategy data container
     * @param symbol The symbol to check
     * @return true if data exists for the symbol
     */
    template<typename Container>
    static bool has_strategy_data(const Container& strategy_data, const std::string& symbol) {
        return find_strategy_data(strategy_data, symbol) != strategy_data.end();
    }
    
    /**
     * Get strategy data for a symbol, creating if it doesn't exist
     * @param strategy_data The strategy data container
     * @param symbol The symbol to get/create
     * @return reference to the strategy data
     */
    template<typename Container, typename DataType>
    static DataType& get_or_create_strategy_data(Container& strategy_data, const std::string& symbol) {
        auto it = find_strategy_data(strategy_data, symbol);
        if (it == strategy_data.end()) {
            strategy_data.emplace_back(symbol, DataType{});
            return strategy_data.back().second;
        }
        return it->second;
    }
};

} // namespace sentio
