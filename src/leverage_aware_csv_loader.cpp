#include "sentio/leverage_aware_csv_loader.hpp"
#include "sentio/accurate_leverage_pricing.hpp"
#include "sentio/sym/leverage_registry.hpp"
#include <iostream>

namespace sentio {

// Static member definitions - Use accurate model by default
LeverageCostModel LeveragePricingConfig::cost_model_{};
bool LeveragePricingConfig::use_theoretical_pricing_ = true; // Default to theoretical pricing

// Initialize with calibrated accurate model
static bool initialize_accurate_model() {
    LeverageCostModel accurate_model;
    // Use calibrated parameters from our analysis
    accurate_model.expense_ratio = 0.0095;      // 0.95% (actual TQQQ/SQQQ)
    accurate_model.borrowing_cost_rate = 0.05;  // 5% (current environment)
    accurate_model.daily_decay_factor = 0.0001; // 0.01% daily rebalancing
    accurate_model.bid_ask_spread = 0.0001;     // 0.01% spread
    accurate_model.tracking_error_std = 0.00005; // Minimal tracking error
    LeveragePricingConfig::set_cost_model(accurate_model);
    return true;
}
static bool g_accurate_model_initialized = initialize_accurate_model();

bool load_csv_leverage_aware(const std::string& symbol, std::vector<Bar>& out) {
    if (!LeveragePricingConfig::is_theoretical_pricing_enabled()) {
        // Fall back to normal CSV loading
        std::string data_path = resolve_csv(symbol);
        return load_csv(data_path, out);
    }
    
    // Use the accurate pricing model instead of the basic one
    AccurateLeverageCostModel accurate_model;
    accurate_model.expense_ratio = 0.0095;
    accurate_model.borrowing_cost_rate = 0.05;
    accurate_model.daily_rebalance_cost = 0.0001;
    accurate_model.tracking_error_daily = 0.00005;
    
    AccurateLeveragePricer accurate_pricer(accurate_model);
    
    LeverageSpec spec;
    if (!LeverageRegistry::instance().lookup(symbol, spec)) {
        // Not a leverage instrument, load normally
        std::string data_path = resolve_csv(symbol);
        return load_csv(data_path, out);
    }
    
    // Load base symbol data
    std::vector<Bar> base_series;
    std::string base_data_path = resolve_csv(spec.base);
    if (!load_csv(base_data_path, base_series)) {
        std::cerr << "❌ Failed to load base data for " << spec.base << std::endl;
        return false;
    }
    
    // Generate theoretical leverage series using accurate model
    out = accurate_pricer.generate_accurate_theoretical_series(symbol, base_series, base_series);
    
    std::cout << "🧮 Generated " << out.size() << " accurate theoretical bars for " << symbol 
              << " (based on " << spec.base << ", ~1% accuracy)" << std::endl;
    
    return !out.empty();
}

bool load_family_csv_leverage_aware(const std::vector<std::string>& symbols,
                                   std::unordered_map<std::string, std::vector<Bar>>& series_out) {
    if (!LeveragePricingConfig::is_theoretical_pricing_enabled()) {
        // Fall back to normal CSV loading
        series_out.clear();
        for (const auto& symbol : symbols) {
            std::vector<Bar> series;
            std::string data_path = resolve_csv(symbol);
            if (load_csv(data_path, series)) {
                series_out[symbol] = std::move(series);
            } else {
                std::cerr << "❌ Failed to load data for " << symbol << std::endl;
                return false;
            }
        }
        return true;
    }
    
    // Use accurate pricing model for family loading
    AccurateLeverageCostModel accurate_model;
    accurate_model.expense_ratio = 0.0095;
    accurate_model.borrowing_cost_rate = 0.05;
    accurate_model.daily_rebalance_cost = 0.0001;
    accurate_model.tracking_error_daily = 0.00005;
    
    AccurateLeveragePricer accurate_pricer(accurate_model);
    
    series_out.clear();
    
    // Separate base and leverage symbols
    std::vector<std::string> base_symbols;
    std::vector<std::string> leverage_symbols;
    
    for (const auto& symbol : symbols) {
        LeverageSpec spec;
        if (LeverageRegistry::instance().lookup(symbol, spec)) {
            leverage_symbols.push_back(symbol);
            // Ensure base symbol is loaded
            if (std::find(base_symbols.begin(), base_symbols.end(), spec.base) == base_symbols.end()) {
                base_symbols.push_back(spec.base);
            }
        } else {
            base_symbols.push_back(symbol);
        }
    }
    
    // Load base symbols first
    for (const auto& symbol : base_symbols) {
        std::vector<Bar> series;
        std::string data_path = resolve_csv(symbol);
        if (load_csv(data_path, series)) {
            series_out[symbol] = std::move(series);
            std::cout << "📊 Loaded " << series_out[symbol].size() << " bars for " << symbol << std::endl;
        } else {
            std::cerr << "❌ Failed to load data for base symbol " << symbol << std::endl;
            return false;
        }
    }
    
    // Generate theoretical data for leverage symbols using accurate model
    for (const auto& symbol : leverage_symbols) {
        LeverageSpec spec;
        if (LeverageRegistry::instance().lookup(symbol, spec)) {
            auto base_it = series_out.find(spec.base);
            if (base_it != series_out.end()) {
                auto theoretical_series = accurate_pricer.generate_accurate_theoretical_series(
                    symbol, base_it->second, base_it->second);
                series_out[symbol] = std::move(theoretical_series);
                std::cout << "🧮 Generated " << series_out[symbol].size() << " accurate theoretical bars for " 
                          << symbol << " (based on " << spec.base << ", ~1% accuracy)" << std::endl;
            } else {
                std::cerr << "❌ Base symbol " << spec.base << " not found for " << symbol << std::endl;
                return false;
            }
        }
    }
    
    return true;
}

bool load_qqq_family_leverage_aware(std::unordered_map<std::string, std::vector<Bar>>& series_out) {
    std::vector<std::string> qqq_symbols = {"QQQ", "TQQQ", "SQQQ", "PSQ"};
    return load_family_csv_leverage_aware(qqq_symbols, series_out);
}

} // namespace sentio
