#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/signal_or.hpp"
#include <vector>
#include <memory>

namespace sentio {

// Signal-OR Strategy Configuration
struct SignalOrCfg {
    // Signal-OR mixer configuration
    OrCfg or_config;
    
    // Strategy parameters
    double min_signal_strength = 0.1;  // Minimum signal strength to act
    double long_threshold = 0.6;        // Probability threshold for long signals
    double short_threshold = 0.4;       // Probability threshold for short signals
    double hold_threshold = 0.05;       // Band around 0.5 for hold signals
    
    // Risk management
    double max_position_weight = 0.8;   // Maximum position weight
    double position_decay = 0.95;       // Position decay factor per bar
    
    // Simple momentum parameters
    int momentum_window = 20;            // Moving average window
    double momentum_scale = 25.0;       // Momentum scaling factor
};

// Signal-OR Strategy Implementation
class SignalOrStrategy : public BaseStrategy {
public:
    explicit SignalOrStrategy(const SignalOrCfg& cfg = SignalOrCfg{});
    
    // Required BaseStrategy methods
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override;
    std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, 
        int current_index,
        const std::string& base_symbol,
        const std::string& bull3x_symbol,
        const std::string& bear3x_symbol,
        const std::string& inverse_symbol) override;
    RouterCfg get_router_config() const override;
    SizerCfg get_sizer_config() const override;
    
    // Configuration
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    // Signal-OR specific methods
    void set_or_config(const OrCfg& config) { cfg_.or_config = config; }
    const OrCfg& get_or_config() const { return cfg_.or_config; }

private:
    SignalOrCfg cfg_;
    
    // State tracking
    double current_position_weight_ = 0.0;
    int warmup_bars_ = 0;
    static constexpr int REQUIRED_WARMUP = 50;
    
    // Helper methods
    std::vector<RuleOut> evaluate_simple_rules(const std::vector<Bar>& bars, int current_index);
    double calculate_momentum_probability(const std::vector<Bar>& bars, int current_index);
    double calculate_position_weight(double signal_strength);
    void update_position_decay();
};

// Register the strategy with the factory
REGISTER_STRATEGY(SignalOrStrategy, "sigor");

} // namespace sentio
