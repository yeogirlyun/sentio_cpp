#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/signal_or.hpp"
#include "sentio/signal_utils.hpp"
#include <algorithm>
#include <vector>
#include <memory>
#include <map>

namespace sentio {

// Signal-OR Strategy Configuration
struct SignalOrCfg {
    // Signal-OR mixer configuration
    OrCfg or_config;
    
    // **PROFIT MAXIMIZATION**: Aggressive thresholds for maximum leverage usage
    double min_signal_strength = 0.05; // Lower threshold to capture more signals
    double long_threshold = 0.55;       // Lower threshold to capture more moderate longs
    double short_threshold = 0.45;      // Higher threshold to capture more moderate shorts
    double hold_threshold = 0.02;       // Tighter hold band to force more action
    
    // **PROFIT MAXIMIZATION**: Remove artificial limits
    // max_position_weight removed - always use 100% capital
    // position_decay removed - not needed for profit maximization
    
    // **PROFIT MAXIMIZATION**: Aggressive momentum for strong signals
    int momentum_window = 10;            // Shorter window for more responsive signals
    double momentum_scale = 50.0;       // Higher scaling for stronger signals
};

// Signal-OR Strategy Implementation
class SignalOrStrategy : public BaseStrategy {
public:
    explicit SignalOrStrategy(const SignalOrCfg& cfg = SignalOrCfg{});
    
    // Required BaseStrategy methods
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override;
    // REMOVED: get_sizer_config() - No artificial limits allowed for profit maximization
    
    // REMOVED: requires_dynamic_allocation - all strategies use same allocation pipeline
    
    // Use hard default from BaseStrategy (no simultaneous positions)
    
    // **STRATEGY-SPECIFIC TRANSITION CONTROL**: Require sequential transitions for conflicts
    bool requires_sequential_transitions() const override { return true; }
    
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
    int warmup_bars_ = 0;
    static constexpr int REQUIRED_WARMUP = 50;
    
    // **REMOVED**: AllocationManager is now handled by the strategy-agnostic backend
    // Strategies only provide probability signals

    // Integrated detector architecture
    std::vector<std::unique_ptr<detectors::IDetector>> detectors_;
    int max_warmup_ = 0;

    struct AuditContext {
        std::map<std::string, double> detector_probs;
        int long_votes = 0;
        int short_votes = 0;
        double final_probability = 0.5;
    };

    AuditContext run_and_aggregate(const std::vector<Bar>& bars, int idx);

    // Helper methods (legacy simple rules retained for fallback if needed)
    std::vector<RuleOut> evaluate_simple_rules(const std::vector<Bar>& bars, int current_index);
    double calculate_momentum_probability(const std::vector<Bar>& bars, int current_index);
    // **PROFIT MAXIMIZATION**: Old position weight methods removed
};

// Register the strategy with the factory
REGISTER_STRATEGY(SignalOrStrategy, "sigor");

} // namespace sentio
