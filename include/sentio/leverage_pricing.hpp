#pragma once
#include "core.hpp"
#include "sentio/sym/leverage_registry.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <random>

namespace sentio {

// Leverage cost model parameters
struct LeverageCostModel {
    double expense_ratio{0.0095};        // 0.95% annual expense ratio for TQQQ/SQQQ
    double borrowing_cost_rate{0.05};    // 5% annual borrowing cost (varies with interest rates)
    double daily_decay_factor{0.0001};   // Additional daily decay from rebalancing friction
    double bid_ask_spread{0.0005};       // 0.05% bid-ask spread cost per rebalance
    double tracking_error_std{0.0002};   // Daily tracking error standard deviation
    
    // Calculate total daily cost rate
    double daily_cost_rate() const {
        return (expense_ratio + borrowing_cost_rate) / 252.0 + daily_decay_factor + bid_ask_spread;
    }
};

// Theoretical leverage pricing engine
class TheoreticalLeveragePricer {
private:
    LeverageCostModel cost_model_;
    std::unordered_map<std::string, double> last_prices_; // Track last prices for continuity
    std::mt19937 rng_;
    
public:
    TheoreticalLeveragePricer(const LeverageCostModel& cost_model = LeverageCostModel{});
    
    // Generate theoretical leverage price based on base price movement
    double calculate_theoretical_price(const std::string& leverage_symbol,
                                     double base_price_prev,
                                     double base_price_current,
                                     double leverage_price_prev);
    
    // Generate full theoretical bar from base bar
    Bar generate_theoretical_bar(const std::string& leverage_symbol,
                                const Bar& base_bar_prev,
                                const Bar& base_bar_current,
                                const Bar& leverage_bar_prev);
    
    // Generate theoretical price series from base series
    std::vector<Bar> generate_theoretical_series(const std::string& leverage_symbol,
                                                const std::vector<Bar>& base_series,
                                                double initial_price = 0.0);
    
    // Update cost model (e.g., for different interest rate environments)
    void update_cost_model(const LeverageCostModel& new_model) { cost_model_ = new_model; }
    
    // Get current cost model
    const LeverageCostModel& get_cost_model() const { return cost_model_; }
};

// Theoretical pricing validator - compares theoretical vs actual prices
class LeveragePricingValidator {
private:
    TheoreticalLeveragePricer pricer_;
    
public:
    struct ValidationResult {
        std::string symbol;
        double price_correlation;
        double return_correlation;
        double mean_price_error;
        double price_error_std;
        double mean_return_error;
        double return_error_std;
        double theoretical_total_return;
        double actual_total_return;
        double return_difference;
        int num_observations;
    };
    
    LeveragePricingValidator(const LeverageCostModel& cost_model = LeverageCostModel{});
    
    // Validate theoretical pricing against actual data
    ValidationResult validate_pricing(const std::string& leverage_symbol,
                                    const std::vector<Bar>& base_series,
                                    const std::vector<Bar>& actual_leverage_series);
    
    // Run comprehensive validation report
    void print_validation_report(const ValidationResult& result);
    
    // Calibrate cost model to minimize pricing errors
    LeverageCostModel calibrate_cost_model(const std::string& leverage_symbol,
                                         const std::vector<Bar>& base_series,
                                         const std::vector<Bar>& actual_leverage_series);
};

// Leverage-aware data loader that generates theoretical prices
class LeverageAwareDataLoader {
private:
    TheoreticalLeveragePricer pricer_;
    
public:
    LeverageAwareDataLoader(const LeverageCostModel& cost_model = LeverageCostModel{});
    
    // Load data with theoretical leverage pricing
    bool load_symbol_data(const std::string& symbol, std::vector<Bar>& out);
    
    // Load multiple symbols with theoretical leverage pricing
    bool load_family_data(const std::vector<std::string>& symbols,
                         std::unordered_map<std::string, std::vector<Bar>>& series_out);
};

} // namespace sentio
