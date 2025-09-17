#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/signal_or.hpp"
#include "sentio/allocation_manager.hpp"
#include <algorithm>
#include <vector>
#include <memory>

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
    std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, 
        int current_index,
        const std::string& base_symbol,
        const std::string& bull3x_symbol,
        const std::string& bear3x_symbol) override;
    RouterCfg get_router_config() const override;
    // REMOVED: get_sizer_config() - No artificial limits allowed for profit maximization
    
    // **ARCHITECTURAL COMPLIANCE**: Use dynamic allocation with strategy-agnostic conflict prevention
    bool requires_dynamic_allocation() const override { return true; }
    
    // **STRATEGY-SPECIFIC CONFLICT RULES**: Define instrument conflict constraints
    bool allows_simultaneous_positions(const std::string& instrument1, const std::string& instrument2) const override {
        // Define instrument groups
        std::vector<std::string> long_instruments = {"QQQ", "TQQQ"};
        std::vector<std::string> inverse_instruments = {"PSQ", "SQQQ"};
        
        auto is_long = [&](const std::string& inst) {
            return std::find(long_instruments.begin(), long_instruments.end(), inst) != long_instruments.end();
        };
        auto is_inverse = [&](const std::string& inst) {
            return std::find(inverse_instruments.begin(), inverse_instruments.end(), inst) != inverse_instruments.end();
        };
        
        // Rule 1: Cannot hold multiple long instruments simultaneously
        if (is_long(instrument1) && is_long(instrument2)) return false;
        
        // Rule 2: Cannot hold multiple inverse instruments simultaneously  
        if (is_inverse(instrument1) && is_inverse(instrument2)) return false;
        
        // Rule 3: Cannot hold long + inverse simultaneously
        if ((is_long(instrument1) && is_inverse(instrument2)) || 
            (is_inverse(instrument1) && is_long(instrument2))) return false;
        
        return true; // All other combinations allowed
    }
    
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
    
    // **MATHEMATICAL ALLOCATION MANAGER**: State-aware portfolio transitions
    std::unique_ptr<AllocationManager> allocation_manager_;
    int last_decision_bar_ = -1;
    
    // Helper methods
    std::vector<RuleOut> evaluate_simple_rules(const std::vector<Bar>& bars, int current_index);
    double calculate_momentum_probability(const std::vector<Bar>& bars, int current_index);
    // **PROFIT MAXIMIZATION**: Old position weight methods removed
};

// Register the strategy with the factory
REGISTER_STRATEGY(SignalOrStrategy, "sigor");

} // namespace sentio
