#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

namespace sentio {

// **MATHEMATICAL ALLOCATION MANAGER**: State-aware portfolio transitions
// Based on threshold models and optimal stopping theory

enum class PositionType {
    CASH,           // No position
    LONG_1X,        // QQQ (1x long)
    LONG_3X,        // TQQQ (3x long)  
    INVERSE_1X,     // PSQ (1x inverse)
    INVERSE_3X      // SQQQ (3x inverse)
};

enum class AllocationAction {
    HOLD,           // Maintain current position
    PARTIAL_CLOSE,  // Close portion of position (e.g., 3x -> 1x)
    FULL_CLOSE,     // Close entire position -> CASH
    ENTER_NEW       // Enter new position from CASH
};

struct PositionState {
    PositionType type = PositionType::CASH;
    double weight = 0.0;                    // Current position weight (0-1)
    double entry_probability = 0.0;         // Signal strength when entered
    double unrealized_pnl_pct = 0.0;       // Current unrealized P&L %
    int bars_held = 0;                      // Bars since position entry
    double max_favorable_prob = 0.0;       // Strongest favorable signal seen
    double max_adverse_prob = 0.0;          // Strongest adverse signal seen
};

struct AllocationDecision {
    AllocationAction action;
    PositionType target_type;
    double target_weight;
    std::string reason;
    double confidence;                      // Decision confidence (0-1)
};

// **THRESHOLD-BASED ALLOCATION PARAMETERS**
struct AllocationConfig {
    // **ENTRY THRESHOLDS**: Signal strength required to enter new positions
    double entry_threshold_1x = 0.55;      // Enter 1x position
    double entry_threshold_3x = 0.70;      // Enter 3x position
    
    // **EXIT THRESHOLDS**: Opposing signal strength to trigger exits
    double partial_exit_threshold = 0.45;  // Reduce leverage (3x -> 1x)
    double full_exit_threshold = 0.35;     // Full exit to cash
    
    // **POSITION STRENGTH FACTORS**: Influence exit decisions
    double profit_protection_factor = 0.1; // Lower exit threshold if profitable
    double loss_cutting_factor = -0.1;     // Higher exit threshold if losing
    double momentum_decay_factor = 0.02;   // Reduce thresholds over time
    
    // **TRANSACTION COST CONSIDERATIONS**
    double min_signal_change = 0.05;       // Minimum signal change to consider action
    double holding_inertia = 0.02;         // Bias toward holding current position
    
    // **RISK MANAGEMENT**
    double max_adverse_tolerance = 0.25;   // Max adverse signal before forced exit
    int max_holding_period = 240;          // Max bars to hold position (force review)
};

class AllocationManager {
private:
    AllocationConfig config_;
    PositionState current_state_;
    
    // **MATHEMATICAL DECISION FUNCTIONS**
    double calculate_exit_threshold(double base_threshold, const PositionState& state) const;
    double calculate_position_strength(const PositionState& state) const;
    double calculate_signal_momentum(double current_prob, const PositionState& state) const;
    bool is_signal_opposing(double probability, PositionType position_type) const;
    bool is_signal_favorable(double probability, PositionType position_type) const;
    PositionType select_entry_position_type(double probability) const;
    
public:
    AllocationManager(const AllocationConfig& config = AllocationConfig{});
    
    // **MAIN ALLOCATION INTERFACE**
    AllocationDecision make_allocation_decision(
        double current_probability,         // Current signal probability (0-1)
        double current_unrealized_pnl_pct,  // Current position P&L %
        int bars_since_last_decision        // Bars since last allocation change
    );
    
    // **STATE MANAGEMENT**
    void update_position_state(const AllocationDecision& decision, double probability);
    void reset_state();
    
    // **ANALYTICS**
    const PositionState& get_current_state() const { return current_state_; }
    double get_decision_confidence() const;
    
    // **CONFIGURATION**
    void update_config(const AllocationConfig& config) { config_ = config; }
    const AllocationConfig& get_config() const { return config_; }
};

// **UTILITY FUNCTIONS**
std::string position_type_to_string(PositionType type);
std::string position_type_to_symbol(PositionType type);
double get_position_leverage(PositionType type);
bool are_positions_conflicting(PositionType type1, PositionType type2);

} // namespace sentio
