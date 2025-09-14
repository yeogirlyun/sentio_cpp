#include "sentio/accurate_leverage_pricing.hpp"
#include "sentio/csv_loader.hpp"
#include "sentio/data_resolver.hpp"
#include "sentio/sym/symbol_utils.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>

namespace sentio {

AccurateLeveragePricer::AccurateLeveragePricer(const AccurateLeverageCostModel& cost_model)
    : cost_model_(cost_model), rng_(std::chrono::steady_clock::now().time_since_epoch().count()) {
}

void AccurateLeveragePricer::reset_daily_tracking_if_needed(const std::string& symbol, int64_t current_timestamp) {
    // Convert timestamp to day (assuming UTC seconds)
    int64_t current_day = current_timestamp / (24 * 3600);
    
    auto it = last_daily_reset_.find(symbol);
    if (it == last_daily_reset_.end() || it->second != current_day) {
        // Reset tracking error for new day
        cumulative_tracking_error_[symbol] = 0.0;
        last_daily_reset_[symbol] = current_day;
    }
}

double AccurateLeveragePricer::calculate_accurate_theoretical_price(const std::string& leverage_symbol,
                                                                  double base_price_prev,
                                                                  double base_price_current,
                                                                  double leverage_price_prev,
                                                                  int64_t timestamp) {
    LeverageSpec spec;
    if (!LeverageRegistry::instance().lookup(leverage_symbol, spec)) {
        return leverage_price_prev; // Not a leverage instrument
    }
    
    // Select appropriate cost model based on ETF
    AccurateLeverageCostModel current_cost_model = cost_model_;
    if (leverage_symbol == "PSQ") {
        current_cost_model = AccurateLeverageCostModel::create_psq_model();
    } else if (leverage_symbol == "TQQQ") {
        current_cost_model = AccurateLeverageCostModel::create_tqqq_model();
    }
    // SQQQ uses default cost model (0.95% expense ratio)
    
    // Calculate base return
    if (base_price_prev <= 0.0) return leverage_price_prev;
    double base_return = (base_price_current - base_price_prev) / base_price_prev;
    
    // Apply leverage factor and inverse if needed
    double leveraged_return = spec.factor * base_return;
    if (spec.inverse) {
        leveraged_return = -leveraged_return;
    }
    
    // Apply minute-level costs (properly scaled)
    double minute_cost = current_cost_model.minute_cost_rate();
    leveraged_return -= minute_cost;
    
    // Handle daily tracking error reset
    if (timestamp > 0) {
        reset_daily_tracking_if_needed(leverage_symbol, timestamp);
    }
    
    // Add minimal tracking error (much smaller than before)
    std::normal_distribution<double> tracking_noise(0.0, current_cost_model.tracking_error_daily / std::sqrt(390.0)); // Scale to minute level
    double tracking_adjustment = tracking_noise(rng_);
    
    // Accumulate tracking error for daily reset
    cumulative_tracking_error_[leverage_symbol] += tracking_adjustment;
    leveraged_return += tracking_adjustment;
    
    // Calculate new theoretical price
    double theoretical_price = leverage_price_prev * (1.0 + leveraged_return);
    
    return std::max(theoretical_price, 0.01); // Prevent negative prices
}

Bar AccurateLeveragePricer::generate_accurate_theoretical_bar(const std::string& leverage_symbol,
                                                            const Bar& base_bar_prev,
                                                            const Bar& base_bar_current,
                                                            const Bar& leverage_bar_prev) {
    LeverageSpec spec;
    if (!LeverageRegistry::instance().lookup(leverage_symbol, spec)) {
        return leverage_bar_prev; // Not a leverage instrument
    }
    
    Bar theoretical_bar;
    theoretical_bar.ts_utc = base_bar_current.ts_utc;
    theoretical_bar.ts_utc_epoch = base_bar_current.ts_utc_epoch;
    
    // Use close-to-close for primary calculation (most accurate)
    double base_return = (base_bar_current.close - base_bar_prev.close) / base_bar_prev.close;
    double leverage_factor = spec.inverse ? -spec.factor : spec.factor;
    
    // Apply minute-level costs
    double minute_cost = cost_model_.minute_cost_rate();
    double leveraged_return = leverage_factor * base_return - minute_cost;
    
    // Add minimal tracking error
    std::normal_distribution<double> tracking_noise(0.0, cost_model_.tracking_error_daily / std::sqrt(390.0));
    leveraged_return += tracking_noise(rng_);
    
    // Calculate close price first (most important)
    theoretical_bar.close = leverage_bar_prev.close * (1.0 + leveraged_return);
    
    // For OHLC, use the same leverage factor but apply to intrabar movements
    // This is more accurate than applying costs to each OHLC point
    if (base_bar_prev.close > 0.0) {
        double base_open_move = (base_bar_current.open - base_bar_prev.close) / base_bar_prev.close;
        double base_high_move = (base_bar_current.high - base_bar_prev.close) / base_bar_prev.close;
        double base_low_move = (base_bar_current.low - base_bar_prev.close) / base_bar_prev.close;
        
        theoretical_bar.open = leverage_bar_prev.close * (1.0 + leverage_factor * base_open_move);
        theoretical_bar.high = leverage_bar_prev.close * (1.0 + leverage_factor * base_high_move);
        theoretical_bar.low = leverage_bar_prev.close * (1.0 + leverage_factor * base_low_move);
    } else {
        theoretical_bar.open = theoretical_bar.close;
        theoretical_bar.high = theoretical_bar.close;
        theoretical_bar.low = theoretical_bar.close;
    }
    
    // Ensure OHLC consistency
    theoretical_bar.high = std::max({theoretical_bar.high, theoretical_bar.open, theoretical_bar.close});
    theoretical_bar.low = std::min({theoretical_bar.low, theoretical_bar.open, theoretical_bar.close});
    
    // Prevent negative prices
    theoretical_bar.open = std::max(theoretical_bar.open, 0.01);
    theoretical_bar.high = std::max(theoretical_bar.high, 0.01);
    theoretical_bar.low = std::max(theoretical_bar.low, 0.01);
    theoretical_bar.close = std::max(theoretical_bar.close, 0.01);
    
    // Volume scaling
    theoretical_bar.volume = static_cast<uint64_t>(base_bar_current.volume * (1.0 + spec.factor * 0.1));
    
    return theoretical_bar;
}

std::vector<Bar> AccurateLeveragePricer::generate_accurate_theoretical_series(const std::string& leverage_symbol,
                                                                            const std::vector<Bar>& base_series,
                                                                            const std::vector<Bar>& actual_series_for_init) {
    if (base_series.empty() || actual_series_for_init.empty()) {
        return {};
    }
    
    LeverageSpec spec;
    if (!LeverageRegistry::instance().lookup(leverage_symbol, spec)) {
        return base_series; // Not a leverage instrument, return base series
    }
    
    std::vector<Bar> theoretical_series;
    theoretical_series.reserve(base_series.size());
    
    // **KEY FIX**: Start with actual first price to eliminate initial error
    Bar initial_bar = actual_series_for_init[0];
    initial_bar.ts_utc = base_series[0].ts_utc;
    initial_bar.ts_utc_epoch = base_series[0].ts_utc_epoch;
    theoretical_series.push_back(initial_bar);
    
    // Reset tracking error for this series
    cumulative_tracking_error_[leverage_symbol] = 0.0;
    
    // Generate subsequent bars
    for (size_t i = 1; i < base_series.size() && i < actual_series_for_init.size(); ++i) {
        Bar theoretical_bar = generate_accurate_theoretical_bar(leverage_symbol,
                                                              base_series[i-1],
                                                              base_series[i],
                                                              theoretical_series[i-1]);
        theoretical_series.push_back(theoretical_bar);
    }
    
    return theoretical_series;
}

AccurateLeveragePricingValidator::AccurateLeveragePricingValidator(const AccurateLeverageCostModel& cost_model)
    : pricer_(cost_model) {
}

AccurateLeveragePricingValidator::AccurateValidationResult AccurateLeveragePricingValidator::validate_accurate_pricing(
    const std::string& leverage_symbol,
    const std::vector<Bar>& base_series,
    const std::vector<Bar>& actual_leverage_series) {
    
    AccurateValidationResult result;
    result.symbol = leverage_symbol;
    result.num_observations = 0;
    result.sub_1pct_accuracy = false;
    
    if (base_series.empty() || actual_leverage_series.empty()) {
        return result;
    }
    
    // Generate theoretical series starting from actual first price
    auto theoretical_series = pricer_.generate_accurate_theoretical_series(leverage_symbol, base_series, actual_leverage_series);
    
    if (theoretical_series.empty()) {
        return result;
    }
    
    // Align series by timestamp and calculate errors
    size_t min_length = std::min({base_series.size(), actual_leverage_series.size(), theoretical_series.size()});
    
    std::vector<double> theoretical_prices, actual_prices;
    std::vector<double> theoretical_returns, actual_returns;
    std::vector<double> price_errors_pct, return_errors_pct;
    
    for (size_t i = 1; i < min_length; ++i) {
        // Skip if timestamps don't match
        if (base_series[i].ts_utc_epoch != actual_leverage_series[i].ts_utc_epoch) {
            continue;
        }
        
        double theo_price = theoretical_series[i].close;
        double actual_price = actual_leverage_series[i].close;
        
        if (actual_price <= 0.0) continue; // Skip invalid data
        
        double theo_return = (theoretical_series[i].close - theoretical_series[i-1].close) / theoretical_series[i-1].close;
        double actual_return = (actual_leverage_series[i].close - actual_leverage_series[i-1].close) / actual_leverage_series[i-1].close;
        
        theoretical_prices.push_back(theo_price);
        actual_prices.push_back(actual_price);
        theoretical_returns.push_back(theo_return);
        actual_returns.push_back(actual_return);
        
        // Calculate percentage errors
        double price_error_pct = std::abs((theo_price - actual_price) / actual_price) * 100.0;
        double return_error_pct = std::abs(theo_return - actual_return) * 100.0;
        
        price_errors_pct.push_back(price_error_pct);
        return_errors_pct.push_back(return_error_pct);
    }
    
    result.num_observations = price_errors_pct.size();
    
    if (result.num_observations == 0) {
        return result;
    }
    
    // Calculate correlations
    auto calculate_correlation = [](const std::vector<double>& x, const std::vector<double>& y) -> double {
        if (x.size() != y.size() || x.empty()) return 0.0;
        
        double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
        double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
        
        double numerator = 0.0, sum_sq_x = 0.0, sum_sq_y = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double dx = x[i] - mean_x;
            double dy = y[i] - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }
        
        double denominator = std::sqrt(sum_sq_x * sum_sq_y);
        return (denominator > 0) ? numerator / denominator : 0.0;
    };
    
    result.price_correlation = calculate_correlation(theoretical_prices, actual_prices);
    result.return_correlation = calculate_correlation(theoretical_returns, actual_returns);
    
    // Calculate error statistics
    result.mean_price_error_pct = std::accumulate(price_errors_pct.begin(), price_errors_pct.end(), 0.0) / price_errors_pct.size();
    result.mean_return_error_pct = std::accumulate(return_errors_pct.begin(), return_errors_pct.end(), 0.0) / return_errors_pct.size();
    
    result.max_price_error_pct = *std::max_element(price_errors_pct.begin(), price_errors_pct.end());
    result.max_return_error_pct = *std::max_element(return_errors_pct.begin(), return_errors_pct.end());
    
    // Calculate standard deviations
    double price_error_var = 0.0, return_error_var = 0.0;
    for (size_t i = 0; i < price_errors_pct.size(); ++i) {
        price_error_var += (price_errors_pct[i] - result.mean_price_error_pct) * (price_errors_pct[i] - result.mean_price_error_pct);
        return_error_var += (return_errors_pct[i] - result.mean_return_error_pct) * (return_errors_pct[i] - result.mean_return_error_pct);
    }
    result.price_error_std_pct = std::sqrt(price_error_var / price_errors_pct.size());
    result.return_error_std_pct = std::sqrt(return_error_var / return_errors_pct.size());
    
    // Calculate total returns
    if (!theoretical_prices.empty() && !actual_prices.empty()) {
        result.theoretical_total_return = (theoretical_prices.back() - theoretical_prices.front()) / theoretical_prices.front() * 100.0;
        result.actual_total_return = (actual_prices.back() - actual_prices.front()) / actual_prices.front() * 100.0;
        result.return_difference_pct = std::abs(result.theoretical_total_return - result.actual_total_return);
    }
    
    // Check if we achieved sub-1% accuracy
    result.sub_1pct_accuracy = (result.mean_price_error_pct < 1.0 && result.max_price_error_pct < 5.0);
    
    return result;
}

void AccurateLeveragePricingValidator::print_accurate_validation_report(const AccurateValidationResult& result) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "🎯 HIGH-ACCURACY LEVERAGE PRICING VALIDATION" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Symbol:                   " << result.symbol << std::endl;
    std::cout << "Observations:             " << result.num_observations << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "📈 CORRELATION ANALYSIS" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Price Correlation:        " << std::fixed << std::setprecision(6) << result.price_correlation << std::endl;
    std::cout << "Return Correlation:       " << std::fixed << std::setprecision(6) << result.return_correlation << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "🎯 ACCURACY ANALYSIS" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Mean Price Error:         " << std::fixed << std::setprecision(4) << result.mean_price_error_pct << "%" << std::endl;
    std::cout << "Price Error Std Dev:      " << std::fixed << std::setprecision(4) << result.price_error_std_pct << "%" << std::endl;
    std::cout << "Max Price Error:          " << std::fixed << std::setprecision(4) << result.max_price_error_pct << "%" << std::endl;
    std::cout << "Mean Return Error:        " << std::fixed << std::setprecision(4) << result.mean_return_error_pct << "%" << std::endl;
    std::cout << "Return Error Std Dev:     " << std::fixed << std::setprecision(4) << result.return_error_std_pct << "%" << std::endl;
    std::cout << "Max Return Error:         " << std::fixed << std::setprecision(4) << result.max_return_error_pct << "%" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "💰 TOTAL RETURN COMPARISON" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Theoretical Total Return: " << std::fixed << std::setprecision(3) << result.theoretical_total_return << "%" << std::endl;
    std::cout << "Actual Total Return:      " << std::fixed << std::setprecision(3) << result.actual_total_return << "%" << std::endl;
    std::cout << "Return Difference:        " << std::fixed << std::setprecision(3) << result.return_difference_pct << "%" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "🏆 ACCURACY ASSESSMENT" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    if (result.sub_1pct_accuracy) {
        std::cout << "🎯 TARGET ACHIEVED: Sub-1% accuracy!" << std::endl;
    } else {
        std::cout << "❌ TARGET MISSED: Above 1% error" << std::endl;
    }
    
    if (result.mean_price_error_pct < 0.1) {
        std::cout << "🏆 EXCEPTIONAL: Mean error < 0.1%" << std::endl;
    } else if (result.mean_price_error_pct < 0.5) {
        std::cout << "✅ EXCELLENT: Mean error < 0.5%" << std::endl;
    } else if (result.mean_price_error_pct < 1.0) {
        std::cout << "✅ GOOD: Mean error < 1.0%" << std::endl;
    } else if (result.mean_price_error_pct < 2.0) {
        std::cout << "⚠️  FAIR: Mean error < 2.0%" << std::endl;
    } else {
        std::cout << "❌ POOR: Mean error > 2.0%" << std::endl;
    }
    
    if (result.price_correlation > 0.9999) {
        std::cout << "🏆 EXCEPTIONAL: Price correlation > 99.99%" << std::endl;
    } else if (result.price_correlation > 0.999) {
        std::cout << "✅ EXCELLENT: Price correlation > 99.9%" << std::endl;
    } else if (result.price_correlation > 0.99) {
        std::cout << "✅ GOOD: Price correlation > 99%" << std::endl;
    } else {
        std::cout << "⚠️  NEEDS IMPROVEMENT: Price correlation < 99%" << std::endl;
    }
    
    std::cout << std::string(80, '=') << std::endl;
}

AccurateLeverageCostModel AccurateLeveragePricingValidator::calibrate_for_accuracy(
    const std::string& leverage_symbol,
    const std::vector<Bar>& base_series,
    const std::vector<Bar>& actual_leverage_series,
    double target_error_pct) {
    
    std::cout << "🔧 Calibrating for sub-" << target_error_pct << "% accuracy..." << std::endl;
    
    AccurateLeverageCostModel best_model;
    double best_error = 1e9;
    
    // Fine-grained parameter search for high accuracy
    std::vector<double> expense_ratios = {0.005, 0.007, 0.009, 0.0095, 0.01, 0.011, 0.013, 0.015};
    std::vector<double> borrowing_costs = {0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08};
    std::vector<double> rebalance_costs = {0.00005, 0.0001, 0.0002, 0.0003, 0.0005};
    std::vector<double> tracking_errors = {0.00001, 0.00005, 0.0001, 0.0002, 0.0005};
    
    int total_combinations = expense_ratios.size() * borrowing_costs.size() * rebalance_costs.size() * tracking_errors.size();
    int tested = 0;
    
    for (double expense_ratio : expense_ratios) {
        for (double borrowing_cost : borrowing_costs) {
            for (double rebalance_cost : rebalance_costs) {
                for (double tracking_error : tracking_errors) {
                    AccurateLeverageCostModel test_model;
                    test_model.expense_ratio = expense_ratio;
                    test_model.borrowing_cost_rate = borrowing_cost;
                    test_model.daily_rebalance_cost = rebalance_cost;
                    test_model.tracking_error_daily = tracking_error;
                    
                    AccurateLeveragePricingValidator test_validator(test_model);
                    auto result = test_validator.validate_accurate_pricing(leverage_symbol, base_series, actual_leverage_series);
                    
                    if (result.mean_price_error_pct < best_error) {
                        best_error = result.mean_price_error_pct;
                        best_model = test_model;
                    }
                    
                    tested++;
                    if (tested % 100 == 0) {
                        std::cout << "   Tested " << tested << "/" << total_combinations 
                                  << " combinations, best error: " << std::fixed << std::setprecision(4) 
                                  << best_error << "%" << std::endl;
                    }
                    
                    // Early exit if we achieve target
                    if (best_error < target_error_pct) {
                        std::cout << "🎯 Target achieved early! Error: " << std::fixed << std::setprecision(4) 
                                  << best_error << "%" << std::endl;
                        goto calibration_complete;
                    }
                }
            }
        }
    }
    
    calibration_complete:
    std::cout << "✅ Calibration complete. Best error: " << std::fixed << std::setprecision(4) << best_error << "%" << std::endl;
    std::cout << "📊 Optimal parameters:" << std::endl;
    std::cout << "   Expense Ratio: " << std::fixed << std::setprecision(4) << best_model.expense_ratio * 100 << "%" << std::endl;
    std::cout << "   Borrowing Cost: " << std::fixed << std::setprecision(4) << best_model.borrowing_cost_rate * 100 << "%" << std::endl;
    std::cout << "   Rebalance Cost: " << std::fixed << std::setprecision(5) << best_model.daily_rebalance_cost * 100 << "%" << std::endl;
    std::cout << "   Tracking Error: " << std::fixed << std::setprecision(5) << best_model.tracking_error_daily * 100 << "%" << std::endl;
    
    return best_model;
}

} // namespace sentio
