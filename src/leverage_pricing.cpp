#include "sentio/leverage_pricing.hpp"
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

TheoreticalLeveragePricer::TheoreticalLeveragePricer(const LeverageCostModel& cost_model)
    : cost_model_(cost_model), rng_(std::chrono::steady_clock::now().time_since_epoch().count()) {
}

double TheoreticalLeveragePricer::calculate_theoretical_price(const std::string& leverage_symbol,
                                                           double base_price_prev,
                                                           double base_price_current,
                                                           double leverage_price_prev) {
    LeverageSpec spec;
    if (!LeverageRegistry::instance().lookup(leverage_symbol, spec)) {
        return leverage_price_prev; // Not a leverage instrument
    }
    
    // Calculate base return
    double base_return = (base_price_current - base_price_prev) / base_price_prev;
    
    // Apply leverage factor and inverse if needed
    double leveraged_return = spec.factor * base_return;
    if (spec.inverse) {
        leveraged_return = -leveraged_return;
    }
    
    // Apply costs (expense ratio, borrowing costs, decay)
    double daily_cost = cost_model_.daily_cost_rate();
    leveraged_return -= daily_cost;
    
    // Add tracking error noise
    std::normal_distribution<double> tracking_noise(0.0, cost_model_.tracking_error_std);
    leveraged_return += tracking_noise(rng_);
    
    // Calculate new theoretical price
    double theoretical_price = leverage_price_prev * (1.0 + leveraged_return);
    
    return std::max(theoretical_price, 0.01); // Prevent negative prices
}

Bar TheoreticalLeveragePricer::generate_theoretical_bar(const std::string& leverage_symbol,
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
    
    // Calculate theoretical OHLC based on base bar movements
    double base_open_return = (base_bar_current.open - base_bar_prev.close) / base_bar_prev.close;
    double base_high_return = (base_bar_current.high - base_bar_prev.close) / base_bar_prev.close;
    double base_low_return = (base_bar_current.low - base_bar_prev.close) / base_bar_prev.close;
    double base_close_return = (base_bar_current.close - base_bar_prev.close) / base_bar_prev.close;
    
    // Apply leverage and inverse
    double leverage_factor = spec.inverse ? -spec.factor : spec.factor;
    double daily_cost = cost_model_.daily_cost_rate();
    
    // Generate tracking error for each price point
    std::normal_distribution<double> tracking_noise(0.0, cost_model_.tracking_error_std);
    
    double open_return = leverage_factor * base_open_return - daily_cost + tracking_noise(rng_);
    double high_return = leverage_factor * base_high_return - daily_cost + tracking_noise(rng_);
    double low_return = leverage_factor * base_low_return - daily_cost + tracking_noise(rng_);
    double close_return = leverage_factor * base_close_return - daily_cost + tracking_noise(rng_);
    
    // Calculate theoretical prices
    theoretical_bar.open = leverage_bar_prev.close * (1.0 + open_return);
    theoretical_bar.high = leverage_bar_prev.close * (1.0 + high_return);
    theoretical_bar.low = leverage_bar_prev.close * (1.0 + low_return);
    theoretical_bar.close = leverage_bar_prev.close * (1.0 + close_return);
    
    // Ensure OHLC consistency (high >= max(open, close), low <= min(open, close))
    theoretical_bar.high = std::max({theoretical_bar.high, theoretical_bar.open, theoretical_bar.close});
    theoretical_bar.low = std::min({theoretical_bar.low, theoretical_bar.open, theoretical_bar.close});
    
    // Prevent negative prices
    theoretical_bar.open = std::max(theoretical_bar.open, 0.01);
    theoretical_bar.high = std::max(theoretical_bar.high, 0.01);
    theoretical_bar.low = std::max(theoretical_bar.low, 0.01);
    theoretical_bar.close = std::max(theoretical_bar.close, 0.01);
    
    // Scale volume based on leverage factor (leveraged ETFs typically have higher volume)
    theoretical_bar.volume = static_cast<uint64_t>(base_bar_current.volume * (1.0 + spec.factor * 0.2));
    
    return theoretical_bar;
}

std::vector<Bar> TheoreticalLeveragePricer::generate_theoretical_series(const std::string& leverage_symbol,
                                                                      const std::vector<Bar>& base_series,
                                                                      double initial_price) {
    if (base_series.empty()) {
        return {};
    }
    
    LeverageSpec spec;
    if (!LeverageRegistry::instance().lookup(leverage_symbol, spec)) {
        return base_series; // Not a leverage instrument, return base series
    }
    
    std::vector<Bar> theoretical_series;
    theoretical_series.reserve(base_series.size());
    
    // Set initial price if not provided
    if (initial_price <= 0.0) {
        // Use a reasonable initial price based on the base price and leverage factor
        initial_price = base_series[0].close / spec.factor;
        if (leverage_symbol == "TQQQ") initial_price = 120.0; // Use realistic TQQQ price level
        if (leverage_symbol == "SQQQ") initial_price = 120.0; // Use realistic SQQQ price level
    }
    
    // Create initial bar
    Bar initial_bar = base_series[0];
    initial_bar.open = initial_bar.high = initial_bar.low = initial_bar.close = initial_price;
    theoretical_series.push_back(initial_bar);
    
    // Generate subsequent bars
    for (size_t i = 1; i < base_series.size(); ++i) {
        Bar theoretical_bar = generate_theoretical_bar(leverage_symbol,
                                                     base_series[i-1],
                                                     base_series[i],
                                                     theoretical_series[i-1]);
        theoretical_series.push_back(theoretical_bar);
    }
    
    return theoretical_series;
}

LeveragePricingValidator::LeveragePricingValidator(const LeverageCostModel& cost_model)
    : pricer_(cost_model) {
}

LeveragePricingValidator::ValidationResult LeveragePricingValidator::validate_pricing(
    const std::string& leverage_symbol,
    const std::vector<Bar>& base_series,
    const std::vector<Bar>& actual_leverage_series) {
    
    ValidationResult result;
    result.symbol = leverage_symbol;
    result.num_observations = 0;
    
    if (base_series.empty() || actual_leverage_series.empty()) {
        return result;
    }
    
    // Generate theoretical series
    double initial_price = actual_leverage_series[0].close;
    auto theoretical_series = pricer_.generate_theoretical_series(leverage_symbol, base_series, initial_price);
    
    if (theoretical_series.empty()) {
        return result;
    }
    
    // Align series by timestamp (use minimum length)
    size_t min_length = std::min({base_series.size(), actual_leverage_series.size(), theoretical_series.size()});
    
    std::vector<double> theoretical_prices, actual_prices;
    std::vector<double> theoretical_returns, actual_returns;
    std::vector<double> price_errors, return_errors;
    
    for (size_t i = 1; i < min_length; ++i) {
        // Skip if timestamps don't match (simple alignment)
        if (base_series[i].ts_utc_epoch != actual_leverage_series[i].ts_utc_epoch) {
            continue;
        }
        
        double theo_price = theoretical_series[i].close;
        double actual_price = actual_leverage_series[i].close;
        
        double theo_return = (theoretical_series[i].close - theoretical_series[i-1].close) / theoretical_series[i-1].close;
        double actual_return = (actual_leverage_series[i].close - actual_leverage_series[i-1].close) / actual_leverage_series[i-1].close;
        
        theoretical_prices.push_back(theo_price);
        actual_prices.push_back(actual_price);
        theoretical_returns.push_back(theo_return);
        actual_returns.push_back(actual_return);
        
        price_errors.push_back((theo_price - actual_price) / actual_price);
        return_errors.push_back(theo_return - actual_return);
    }
    
    result.num_observations = price_errors.size();
    
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
    result.mean_price_error = std::accumulate(price_errors.begin(), price_errors.end(), 0.0) / price_errors.size();
    result.mean_return_error = std::accumulate(return_errors.begin(), return_errors.end(), 0.0) / return_errors.size();
    
    double price_error_var = 0.0, return_error_var = 0.0;
    for (size_t i = 0; i < price_errors.size(); ++i) {
        price_error_var += (price_errors[i] - result.mean_price_error) * (price_errors[i] - result.mean_price_error);
        return_error_var += (return_errors[i] - result.mean_return_error) * (return_errors[i] - result.mean_return_error);
    }
    result.price_error_std = std::sqrt(price_error_var / price_errors.size());
    result.return_error_std = std::sqrt(return_error_var / return_errors.size());
    
    // Calculate total returns
    if (!theoretical_prices.empty() && !actual_prices.empty()) {
        result.theoretical_total_return = (theoretical_prices.back() - theoretical_prices.front()) / theoretical_prices.front();
        result.actual_total_return = (actual_prices.back() - actual_prices.front()) / actual_prices.front();
        result.return_difference = result.theoretical_total_return - result.actual_total_return;
    }
    
    return result;
}

void LeveragePricingValidator::print_validation_report(const ValidationResult& result) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "📊 LEVERAGE PRICING VALIDATION REPORT" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Symbol:                   " << result.symbol << std::endl;
    std::cout << "Observations:             " << result.num_observations << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "📈 CORRELATION ANALYSIS" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Price Correlation:        " << std::fixed << std::setprecision(4) << result.price_correlation << std::endl;
    std::cout << "Return Correlation:       " << std::fixed << std::setprecision(4) << result.return_correlation << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "📊 ERROR ANALYSIS" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Mean Price Error:         " << std::fixed << std::setprecision(4) << result.mean_price_error * 100 << "%" << std::endl;
    std::cout << "Price Error Std Dev:      " << std::fixed << std::setprecision(4) << result.price_error_std * 100 << "%" << std::endl;
    std::cout << "Mean Return Error:        " << std::fixed << std::setprecision(4) << result.mean_return_error * 100 << "%" << std::endl;
    std::cout << "Return Error Std Dev:     " << std::fixed << std::setprecision(4) << result.return_error_std * 100 << "%" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "💰 TOTAL RETURN COMPARISON" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Theoretical Total Return: " << std::fixed << std::setprecision(2) << result.theoretical_total_return * 100 << "%" << std::endl;
    std::cout << "Actual Total Return:      " << std::fixed << std::setprecision(2) << result.actual_total_return * 100 << "%" << std::endl;
    std::cout << "Return Difference:        " << std::fixed << std::setprecision(2) << result.return_difference * 100 << "%" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Quality assessment
    std::cout << "🎯 MODEL QUALITY ASSESSMENT" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    if (result.return_correlation > 0.95) {
        std::cout << "✅ EXCELLENT: Return correlation > 95%" << std::endl;
    } else if (result.return_correlation > 0.90) {
        std::cout << "✅ GOOD: Return correlation > 90%" << std::endl;
    } else if (result.return_correlation > 0.80) {
        std::cout << "⚠️  FAIR: Return correlation > 80%" << std::endl;
    } else {
        std::cout << "❌ POOR: Return correlation < 80%" << std::endl;
    }
    
    if (std::abs(result.return_difference) < 0.02) {
        std::cout << "✅ EXCELLENT: Total return difference < 2%" << std::endl;
    } else if (std::abs(result.return_difference) < 0.05) {
        std::cout << "✅ GOOD: Total return difference < 5%" << std::endl;
    } else if (std::abs(result.return_difference) < 0.10) {
        std::cout << "⚠️  FAIR: Total return difference < 10%" << std::endl;
    } else {
        std::cout << "❌ POOR: Total return difference > 10%" << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl << std::endl;
}

LeverageCostModel LeveragePricingValidator::calibrate_cost_model(const std::string& leverage_symbol,
                                                               const std::vector<Bar>& base_series,
                                                               const std::vector<Bar>& actual_leverage_series) {
    // Simple grid search calibration
    LeverageCostModel best_model;
    double best_correlation = -1.0;
    
    std::vector<double> expense_ratios = {0.005, 0.0075, 0.0095, 0.012, 0.015};
    std::vector<double> borrowing_costs = {0.02, 0.035, 0.05, 0.065, 0.08};
    std::vector<double> decay_factors = {0.00005, 0.0001, 0.0002, 0.0003, 0.0005};
    
    std::cout << "🔧 Calibrating cost model for " << leverage_symbol << "..." << std::endl;
    
    for (double expense_ratio : expense_ratios) {
        for (double borrowing_cost : borrowing_costs) {
            for (double decay_factor : decay_factors) {
                LeverageCostModel test_model;
                test_model.expense_ratio = expense_ratio;
                test_model.borrowing_cost_rate = borrowing_cost;
                test_model.daily_decay_factor = decay_factor;
                
                TheoreticalLeveragePricer test_pricer(test_model);
                LeveragePricingValidator test_validator(test_model);
                
                auto result = test_validator.validate_pricing(leverage_symbol, base_series, actual_leverage_series);
                
                if (result.return_correlation > best_correlation) {
                    best_correlation = result.return_correlation;
                    best_model = test_model;
                }
            }
        }
    }
    
    std::cout << "✅ Best correlation found: " << std::fixed << std::setprecision(4) << best_correlation << std::endl;
    std::cout << "📊 Optimal parameters:" << std::endl;
    std::cout << "   Expense Ratio: " << std::fixed << std::setprecision(3) << best_model.expense_ratio * 100 << "%" << std::endl;
    std::cout << "   Borrowing Cost: " << std::fixed << std::setprecision(3) << best_model.borrowing_cost_rate * 100 << "%" << std::endl;
    std::cout << "   Daily Decay: " << std::fixed << std::setprecision(5) << best_model.daily_decay_factor * 100 << "%" << std::endl;
    
    return best_model;
}

LeverageAwareDataLoader::LeverageAwareDataLoader(const LeverageCostModel& cost_model)
    : pricer_(cost_model) {
}

bool LeverageAwareDataLoader::load_symbol_data(const std::string& symbol, std::vector<Bar>& out) {
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
    
    // Generate theoretical leverage series
    out = pricer_.generate_theoretical_series(symbol, base_series);
    
    std::cout << "🧮 Generated " << out.size() << " theoretical bars for " << symbol 
              << " (based on " << spec.base << ")" << std::endl;
    
    return !out.empty();
}

bool LeverageAwareDataLoader::load_family_data(const std::vector<std::string>& symbols,
                                             std::unordered_map<std::string, std::vector<Bar>>& series_out) {
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
    
    // Generate theoretical data for leverage symbols
    for (const auto& symbol : leverage_symbols) {
        LeverageSpec spec;
        if (LeverageRegistry::instance().lookup(symbol, spec)) {
            auto base_it = series_out.find(spec.base);
            if (base_it != series_out.end()) {
                auto theoretical_series = pricer_.generate_theoretical_series(symbol, base_it->second);
                series_out[symbol] = std::move(theoretical_series);
                std::cout << "🧮 Generated " << series_out[symbol].size() << " theoretical bars for " 
                          << symbol << " (based on " << spec.base << ")" << std::endl;
            } else {
                std::cerr << "❌ Base symbol " << spec.base << " not found for " << symbol << std::endl;
                return false;
            }
        }
    }
    
    return true;
}

} // namespace sentio
