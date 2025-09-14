#pragma once
#include "core.hpp"
#include "sentio/sym/leverage_registry.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <random>

namespace sentio {

// High-accuracy leverage cost model with proper time scaling
struct AccurateLeverageCostModel {
    double expense_ratio{0.0095};        // 0.95% annual expense ratio (default for SQQQ/PSQ)
    double borrowing_cost_rate{0.05};    // 5% annual borrowing cost
    double daily_rebalance_cost{0.0001}; // 0.01% daily rebalancing cost
    double bid_ask_spread{0.0002};       // 0.02% bid-ask spread per trade
    double tracking_error_daily{0.0001}; // 0.01% daily tracking error std
    
    // Create PSQ-specific cost model (1x inverse with 0.95% expense ratio)
    static AccurateLeverageCostModel create_psq_model() {
        AccurateLeverageCostModel model;
        model.expense_ratio = 0.0095;        // 0.95% annual expense ratio
        model.borrowing_cost_rate = 0.02;    // Lower borrowing cost for 1x vs 3x
        model.daily_rebalance_cost = 0.00005; // Lower rebalancing cost for 1x
        model.bid_ask_spread = 0.0001;       // Tighter spread for 1x
        model.tracking_error_daily = 0.00005; // Lower tracking error for 1x
        return model;
    }
    
    // Create TQQQ-specific cost model (3x long)
    static AccurateLeverageCostModel create_tqqq_model() {
        AccurateLeverageCostModel model;
        model.expense_ratio = 0.0086;        // 0.86% annual expense ratio for TQQQ
        model.borrowing_cost_rate = 0.05;    // Higher borrowing cost for 3x
        model.daily_rebalance_cost = 0.0001; // Standard rebalancing cost
        model.bid_ask_spread = 0.0002;       // Standard spread
        model.tracking_error_daily = 0.0001; // Standard tracking error
        return model;
    }
    
    // Calculate minute-level cost rate (properly scaled)
    double minute_cost_rate() const {
        // Scale annual costs to minute level (252 trading days * 390 minutes per day)
        double annual_minutes = 252.0 * 390.0;
        return (expense_ratio + borrowing_cost_rate) / annual_minutes;
    }
    
    // Calculate daily-level cost rate
    double daily_cost_rate() const {
        return (expense_ratio + borrowing_cost_rate) / 252.0 + daily_rebalance_cost;
    }
};

// High-accuracy theoretical leverage pricing engine
class AccurateLeveragePricer {
private:
    AccurateLeverageCostModel cost_model_;
    std::mt19937 rng_;
    std::unordered_map<std::string, double> cumulative_tracking_error_;
    std::unordered_map<std::string, int64_t> last_daily_reset_;
    
    // Reset tracking error daily (simulates daily rebalancing)
    void reset_daily_tracking_if_needed(const std::string& symbol, int64_t current_timestamp);
    
public:
    AccurateLeveragePricer(const AccurateLeverageCostModel& cost_model = AccurateLeverageCostModel{});
    
    // Calculate theoretical price with high accuracy
    double calculate_accurate_theoretical_price(const std::string& leverage_symbol,
                                              double base_price_prev,
                                              double base_price_current,
                                              double leverage_price_prev,
                                              int64_t timestamp = 0);
    
    // Generate theoretical bar with minimal error
    Bar generate_accurate_theoretical_bar(const std::string& leverage_symbol,
                                        const Bar& base_bar_prev,
                                        const Bar& base_bar_current,
                                        const Bar& leverage_bar_prev);
    
    // Generate theoretical series starting from actual first price
    std::vector<Bar> generate_accurate_theoretical_series(const std::string& leverage_symbol,
                                                        const std::vector<Bar>& base_series,
                                                        const std::vector<Bar>& actual_series_for_init);
    
    // Update cost model
    void update_cost_model(const AccurateLeverageCostModel& new_model) { cost_model_ = new_model; }
    
    // Get current cost model
    const AccurateLeverageCostModel& get_cost_model() const { return cost_model_; }
};

// High-accuracy validator for sub-1% error validation
class AccurateLeveragePricingValidator {
private:
    AccurateLeveragePricer pricer_;
    
public:
    struct AccurateValidationResult {
        std::string symbol;
        double price_correlation;
        double return_correlation;
        double mean_price_error_pct;
        double price_error_std_pct;
        double mean_return_error_pct;
        double return_error_std_pct;
        double max_price_error_pct;
        double max_return_error_pct;
        double theoretical_total_return;
        double actual_total_return;
        double return_difference_pct;
        int num_observations;
        bool sub_1pct_accuracy;
    };
    
    AccurateLeveragePricingValidator(const AccurateLeverageCostModel& cost_model = AccurateLeverageCostModel{});
    
    // Validate with sub-1% accuracy target
    AccurateValidationResult validate_accurate_pricing(const std::string& leverage_symbol,
                                                     const std::vector<Bar>& base_series,
                                                     const std::vector<Bar>& actual_leverage_series);
    
    // Print detailed validation report
    void print_accurate_validation_report(const AccurateValidationResult& result);
    
    // Auto-calibrate for sub-1% accuracy
    AccurateLeverageCostModel calibrate_for_accuracy(const std::string& leverage_symbol,
                                                   const std::vector<Bar>& base_series,
                                                   const std::vector<Bar>& actual_leverage_series,
                                                   double target_error_pct = 1.0);
};

} // namespace sentio
