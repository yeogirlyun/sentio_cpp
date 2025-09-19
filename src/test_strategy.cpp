#include "sentio/test_strategy.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace sentio {

TestStrategy::TestStrategy(const TestStrategyConfig& config)
    : BaseStrategy("TestStrategy"), config_(config), rng_(config_.random_seed), uniform_dist_(0.0, 1.0) {
    reset_test_state();
}

void TestStrategy::reset_test_state() {
    signal_history_.clear();
    accuracy_history_.clear();
    last_signal_bar_ = -1;
    rng_.seed(config_.random_seed);
}

double TestStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (current_index < 0 || current_index >= static_cast<int>(bars.size())) {
        return 0.5; // Neutral for invalid index
    }
    
    // **PERFECT FORESIGHT MODE**: Always predict correctly
    if (config_.perfect_foresight) {
        double future_return = calculate_future_return(bars, current_index);
        double probability = 0.5 + future_return * 10.0; // Scale return to probability
        probability = std::max(0.0, std::min(1.0, probability)); // Clamp to [0,1]
        
        signal_history_.push_back(probability);
        accuracy_history_.push_back(1.0); // Perfect accuracy
        return probability;
    }
    
    // **CONTROLLED ACCURACY MODE**: Generate signal with target accuracy
    double future_return = calculate_future_return(bars, current_index);
    
    // Determine true signal direction
    double true_probability = 0.5;
    if (std::abs(future_return) > config_.signal_threshold) {
        true_probability = future_return > 0 ? 0.7 : 0.3; // Strong directional signal
    }
    
    // Generate signal with controlled accuracy
    double generated_signal = generate_signal_with_accuracy(true_probability, config_.target_accuracy);
    
    // Add noise to make it more realistic
    generated_signal = add_signal_noise(generated_signal);
    
    // Clamp to valid probability range
    generated_signal = std::max(0.0, std::min(1.0, generated_signal));
    
    // Track signal for accuracy calculation
    signal_history_.push_back(generated_signal);
    
    // Calculate actual accuracy so far
    if (signal_history_.size() > 1) {
        int correct_predictions = 0;
        for (size_t i = 0; i < signal_history_.size() - 1; ++i) {
            bool predicted_up = signal_history_[i] > 0.5;
            
            // Get actual direction from next bar
            if (i + 1 < bars.size()) {
                double actual_return = (bars[i + 1].close - bars[i].close) / bars[i].close;
                bool actual_up = actual_return > 0;
                
                if (predicted_up == actual_up) {
                    correct_predictions++;
                }
            }
        }
        
        double actual_accuracy = static_cast<double>(correct_predictions) / (signal_history_.size() - 1);
        accuracy_history_.push_back(actual_accuracy);
    }
    
    last_signal_bar_ = current_index;
    return generated_signal;
}

// REMOVED: get_allocation_decisions - AllocationManager handles all instrument decisions
/*
std::vector<TestStrategy::AllocationDecision> TestStrategy::get_allocation_decisions_REMOVED(
    const std::vector<Bar>& bars,
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    if (current_index < 0 || current_index >= static_cast<int>(bars.size())) {
        return decisions;
    }
    
    // Get probability signal
    double probability = calculate_probability(bars, current_index);
    double signal_strength = std::abs(probability - 0.5) * 2.0; // 0 to 1 scale
    
    // **ALLOCATION LOGIC**: Based on signal strength and configuration
    std::string target_instrument;
    double target_weight = 0.0;
    std::string reason;
    
    if (probability > (0.5 + config_.neutral_band)) {
        // **BULLISH SIGNAL**
        if (config_.enable_leverage && signal_strength > config_.strong_signal_threshold) {
            target_instrument = bull3x_symbol; // TQQQ
            target_weight = 1.0;
            reason = "Test Strategy: Strong bullish signal -> 3x leverage";
        } else if (signal_strength > config_.weak_signal_threshold) {
            target_instrument = base_symbol; // QQQ
            target_weight = 1.0;
            reason = "Test Strategy: Moderate bullish signal -> 1x long";
        }
        
    } else if (probability < (0.5 - config_.neutral_band)) {
        // **BEARISH SIGNAL**
        if (config_.enable_leverage && signal_strength > config_.strong_signal_threshold) {
            target_instrument = bear3x_symbol; // SQQQ
            target_weight = 1.0;
            reason = "Test Strategy: Strong bearish signal -> 3x inverse";
        } else if (signal_strength > config_.weak_signal_threshold) {
            target_instrument = "PSQ"; // 1x inverse
            target_weight = 1.0;
            reason = "Test Strategy: Moderate bearish signal -> 1x inverse";
        }
    }
    
    // Only return decision if we have a target and it's a new bar
    if (!target_instrument.empty() && current_index != last_signal_bar_) {
        decisions.push_back({target_instrument, target_weight, signal_strength, reason});
    }
    
    return decisions;
}
*/

double TestStrategy::calculate_future_return(const std::vector<Bar>& bars, int current_index) const {
    if (current_index + config_.lookhead_bars >= static_cast<int>(bars.size())) {
        return 0.0; // No future data available
    }
    
    const Bar& current_bar = bars[current_index];
    const Bar& future_bar = bars[current_index + config_.lookhead_bars];
    
    return (future_bar.close - current_bar.close) / current_bar.close;
}

double TestStrategy::generate_signal_with_accuracy(double true_signal, double target_accuracy) const {
    double random_value = uniform_dist_(rng_);
    
    if (random_value < target_accuracy) {
        // Generate correct signal
        return true_signal;
    } else {
        // Generate incorrect signal (flip direction)
        if (true_signal > 0.5) {
            return 0.5 - (true_signal - 0.5); // Flip to bearish
        } else {
            return 0.5 + (0.5 - true_signal); // Flip to bullish
        }
    }
}

double TestStrategy::add_signal_noise(double signal) const {
    if (config_.noise_factor <= 0.0) return signal;
    
    // Add Gaussian noise
    std::normal_distribution<double> noise_dist(0.0, config_.noise_factor);
    double noise = noise_dist(rng_);
    
    return signal + noise;
}

bool TestStrategy::should_use_leverage(double signal_strength) const {
    return config_.enable_leverage && signal_strength > config_.strong_signal_threshold;
}

// REMOVED: get_router_config - AllocationManager handles routing
/*
RouterCfg TestStrategy::get_router_config_REMOVED() const {
    RouterCfg cfg;
    cfg.base_symbol = "QQQ";
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    cfg.min_signal_strength = config_.signal_threshold;
    cfg.max_position_pct = 1.0; // 100% position for profit maximization
    return cfg;
}
*/

bool TestStrategy::allows_simultaneous_positions(const std::string& instrument1, const std::string& instrument2) const {
    // Define conflicting instrument groups
    std::vector<std::string> long_instruments = {"QQQ", "TQQQ"};
    std::vector<std::string> inverse_instruments = {"PSQ", "SQQQ"};
    
    auto is_long = [&](const std::string& inst) {
        return std::find(long_instruments.begin(), long_instruments.end(), inst) != long_instruments.end();
    };
    auto is_inverse = [&](const std::string& inst) {
        return std::find(inverse_instruments.begin(), inverse_instruments.end(), inst) != inverse_instruments.end();
    };
    
    // Cannot hold multiple long instruments
    if (is_long(instrument1) && is_long(instrument2)) return false;
    
    // Cannot hold multiple inverse instruments
    if (is_inverse(instrument1) && is_inverse(instrument2)) return false;
    
    // Cannot hold long + inverse simultaneously
    if ((is_long(instrument1) && is_inverse(instrument2)) || 
        (is_inverse(instrument1) && is_long(instrument2))) return false;
    
    return true;
}

ParameterMap TestStrategy::get_default_params() const {
    ParameterMap params;
    params["target_accuracy"] = config_.target_accuracy;
    params["lookhead_bars"] = static_cast<double>(config_.lookhead_bars);
    params["signal_threshold"] = config_.signal_threshold;
    params["noise_factor"] = config_.noise_factor;
    params["enable_leverage"] = config_.enable_leverage ? 1.0 : 0.0;
    return params;
}

ParameterSpace TestStrategy::get_param_space() const {
    ParameterSpace space;
    space["target_accuracy"] = {ParamType::FLOAT, 0.0, 1.0, 0.05}; // 0% to 100% accuracy
    space["lookhead_bars"] = {ParamType::INT, 1.0, 10.0, 1.0};   // 1 to 10 bars lookhead
    space["signal_threshold"] = {ParamType::FLOAT, 0.01, 0.10, 0.01}; // 1% to 10% threshold
    space["noise_factor"] = {ParamType::FLOAT, 0.0, 0.5, 0.05};    // 0% to 50% noise
    space["enable_leverage"] = {ParamType::INT, 0.0, 1.0, 1.0};  // Boolean: 0 or 1
    return space;
}

void TestStrategy::apply_params() {
    const auto& params = params_;
    
    if (params.find("target_accuracy") != params.end()) {
        config_.target_accuracy = params.at("target_accuracy");
    }
    if (params.find("lookhead_bars") != params.end()) {
        config_.lookhead_bars = static_cast<int>(params.at("lookhead_bars"));
    }
    if (params.find("signal_threshold") != params.end()) {
        config_.signal_threshold = params.at("signal_threshold");
    }
    if (params.find("noise_factor") != params.end()) {
        config_.noise_factor = params.at("noise_factor");
    }
    if (params.find("enable_leverage") != params.end()) {
        config_.enable_leverage = params.at("enable_leverage") > 0.5;
    }
}

void TestStrategy::set_target_accuracy(double accuracy) {
    config_.target_accuracy = std::max(0.0, std::min(1.0, accuracy));
}

double TestStrategy::get_actual_accuracy() const {
    if (accuracy_history_.empty()) return 0.0;
    return accuracy_history_.back();
}

void TestStrategy::update_config(const TestStrategyConfig& config) {
    config_ = config;
    rng_.seed(config_.random_seed);
}

// **TEST STRATEGY FACTORY IMPLEMENTATION**
std::unique_ptr<TestStrategy> TestStrategyFactory::create_random_strategy(double accuracy) {
    TestStrategyConfig config;
    config.target_accuracy = accuracy;
    config.enable_leverage = false; // Conservative for random strategy
    config.noise_factor = 0.2;      // High noise
    return std::make_unique<TestStrategy>(config);
}

std::unique_ptr<TestStrategy> TestStrategyFactory::create_poor_strategy(double accuracy) {
    TestStrategyConfig config;
    config.target_accuracy = accuracy;
    config.enable_leverage = false;
    config.noise_factor = 0.15;
    config.signal_threshold = 0.05; // Higher threshold (fewer signals)
    return std::make_unique<TestStrategy>(config);
}

std::unique_ptr<TestStrategy> TestStrategyFactory::create_decent_strategy(double accuracy) {
    TestStrategyConfig config;
    config.target_accuracy = accuracy;
    config.enable_leverage = true;  // Enable leverage for decent strategy
    config.noise_factor = 0.1;
    config.signal_threshold = 0.03;
    return std::make_unique<TestStrategy>(config);
}

std::unique_ptr<TestStrategy> TestStrategyFactory::create_good_strategy(double accuracy) {
    TestStrategyConfig config;
    config.target_accuracy = accuracy;
    config.enable_leverage = true;
    config.noise_factor = 0.05;     // Low noise
    config.signal_threshold = 0.02; // Lower threshold (more signals)
    config.strong_signal_threshold = 0.70; // Lower threshold for leverage
    return std::make_unique<TestStrategy>(config);
}

std::unique_ptr<TestStrategy> TestStrategyFactory::create_excellent_strategy(double accuracy) {
    TestStrategyConfig config;
    config.target_accuracy = accuracy;
    config.enable_leverage = true;
    config.noise_factor = 0.02;     // Very low noise
    config.signal_threshold = 0.01; // Very low threshold
    config.strong_signal_threshold = 0.65; // Aggressive leverage usage
    return std::make_unique<TestStrategy>(config);
}

std::unique_ptr<TestStrategy> TestStrategyFactory::create_perfect_strategy() {
    TestStrategyConfig config;
    config.perfect_foresight = true;
    config.target_accuracy = 1.0;
    config.enable_leverage = true;
    config.noise_factor = 0.0;      // No noise
    config.signal_threshold = 0.001; // Capture all movements
    return std::make_unique<TestStrategy>(config);
}

std::vector<std::unique_ptr<TestStrategy>> TestStrategyFactory::create_accuracy_test_suite() {
    std::vector<std::unique_ptr<TestStrategy>> strategies;
    
    // Create strategies with different accuracy levels
    std::vector<double> accuracy_levels = {0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00};
    
    for (double accuracy : accuracy_levels) {
        TestStrategyConfig config;
        config.target_accuracy = accuracy;
        config.enable_leverage = true;
        config.noise_factor = 0.05;
        config.random_seed = 42 + static_cast<unsigned int>(accuracy * 100); // Different seed per strategy
        
        strategies.push_back(std::make_unique<TestStrategy>(config));
    }
    
    return strategies;
}

} // namespace sentio
