#pragma once

#include "sentio/base_strategy.hpp"
#include <vector>
#include <random>

namespace sentio {

// **TEST STRATEGY**: Configurable accuracy for backend testing
// This strategy "cheats" by looking ahead to generate signals with known accuracy

struct TestStrategyConfig {
    double target_accuracy = 0.60;     // Target accuracy (0.0 to 1.0)
    int lookhead_bars = 1;              // How many bars to look ahead
    double signal_threshold = 0.02;     // Minimum return to generate signal
    double noise_factor = 0.1;          // Random noise added to signals
    bool enable_leverage = true;        // Whether to use leveraged instruments
    
    // Signal generation parameters
    double strong_signal_threshold = 0.75;  // Threshold for strong signals (3x leverage)
    double weak_signal_threshold = 0.55;    // Threshold for weak signals (1x)
    double neutral_band = 0.05;             // Neutral zone around 0.5
    
    // Accuracy control
    bool perfect_foresight = false;     // If true, always predict correctly
    unsigned int random_seed = 42;      // For reproducible results
};

class TestStrategy : public BaseStrategy {
private:
    TestStrategyConfig config_;
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<double> uniform_dist_;
    
    // State tracking
    mutable std::vector<double> signal_history_;
    mutable std::vector<double> accuracy_history_;
    mutable int last_signal_bar_ = -1;
    
    // Helper methods
    double calculate_future_return(const std::vector<Bar>& bars, int current_index) const;
    double generate_signal_with_accuracy(double true_signal, double target_accuracy) const;
    double add_signal_noise(double signal) const;
    bool should_use_leverage(double signal_strength) const;
    
public:
    TestStrategy(const TestStrategyConfig& config = TestStrategyConfig{});
    
    // **BaseStrategy Interface Implementation**
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override;
    
    std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars,
        int current_index,
        const std::string& base_symbol,
        const std::string& bull3x_symbol,
        const std::string& bear3x_symbol) override;
    
    RouterCfg get_router_config() const override;
    
    // **Strategy-Specific Conflict Rules**
    bool allows_simultaneous_positions(const std::string& instrument1, const std::string& instrument2) const override;
    bool requires_sequential_transitions() const override { return true; }
    
    // **Parameter Management**
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    // **Test Strategy Specific Methods**
    void set_target_accuracy(double accuracy);
    double get_actual_accuracy() const;
    std::vector<double> get_signal_history() const override { return signal_history_; }
    const std::vector<double>& get_accuracy_history() const { return accuracy_history_; }
    
    // **Configuration**
    void update_config(const TestStrategyConfig& config);
    const TestStrategyConfig& get_config() const { return config_; }
    
    // **Reset for new test**
    void reset_test_state();
};

// **TEST STRATEGY FACTORY**: Create strategies with different accuracies
class TestStrategyFactory {
public:
    static std::unique_ptr<TestStrategy> create_random_strategy(double accuracy = 0.50);
    static std::unique_ptr<TestStrategy> create_poor_strategy(double accuracy = 0.40);
    static std::unique_ptr<TestStrategy> create_decent_strategy(double accuracy = 0.60);
    static std::unique_ptr<TestStrategy> create_good_strategy(double accuracy = 0.75);
    static std::unique_ptr<TestStrategy> create_excellent_strategy(double accuracy = 0.90);
    static std::unique_ptr<TestStrategy> create_perfect_strategy();
    
    // **Batch Testing**: Create multiple strategies with different accuracies
    static std::vector<std::unique_ptr<TestStrategy>> create_accuracy_test_suite();
};

} // namespace sentio
