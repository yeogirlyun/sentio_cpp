#pragma once
#include "sentio/csv_loader.hpp"
#include "sentio/leverage_pricing.hpp"
#include "sentio/data_resolver.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace sentio {

// Drop-in replacement for load_csv that uses theoretical pricing for leverage instruments
bool load_csv_leverage_aware(const std::string& symbol, std::vector<Bar>& out);

// Load multiple symbols with leverage-aware pricing
bool load_family_csv_leverage_aware(const std::vector<std::string>& symbols,
                                   std::unordered_map<std::string, std::vector<Bar>>& series_out);

// Load QQQ family with theoretical leverage pricing
bool load_qqq_family_leverage_aware(std::unordered_map<std::string, std::vector<Bar>>& series_out);

// Global configuration for leverage pricing
class LeveragePricingConfig {
private:
    static LeverageCostModel cost_model_;
    static bool use_theoretical_pricing_;
    
public:
    // Enable/disable theoretical pricing globally
    static void enable_theoretical_pricing(bool enable = true) {
        use_theoretical_pricing_ = enable;
    }
    
    static bool is_theoretical_pricing_enabled() {
        return use_theoretical_pricing_;
    }
    
    // Update the global cost model
    static void set_cost_model(const LeverageCostModel& model) {
        cost_model_ = model;
    }
    
    static const LeverageCostModel& get_cost_model() {
        return cost_model_;
    }
};

} // namespace sentio
