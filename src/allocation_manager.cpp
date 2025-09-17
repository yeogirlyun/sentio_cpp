#include "sentio/allocation_manager.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace sentio {

AllocationManager::AllocationManager(const AllocationConfig& config) 
    : config_(config) {
    reset_state();
}

void AllocationManager::reset_state() {
    current_state_ = PositionState{};
}

// **MATHEMATICAL DECISION ENGINE**: Core allocation logic
AllocationDecision AllocationManager::make_allocation_decision(
    double current_probability,
    double current_unrealized_pnl_pct,
    int bars_since_last_decision) {
    
    // Update current state
    current_state_.unrealized_pnl_pct = current_unrealized_pnl_pct;
    current_state_.bars_held += bars_since_last_decision;
    
    // Track signal extremes for momentum analysis
    if (current_probability > current_state_.max_favorable_prob) {
        current_state_.max_favorable_prob = current_probability;
    }
    if (current_probability < (1.0 - current_state_.max_adverse_prob)) {
        current_state_.max_adverse_prob = 1.0 - current_probability;
    }
    
    AllocationDecision decision;
    decision.confidence = 0.0;
    
    // **CASE 1: CURRENTLY IN CASH** - Consider entry
    if (current_state_.type == PositionType::CASH) {
        PositionType target_type = select_entry_position_type(current_probability);
        
        if (target_type != PositionType::CASH) {
            decision.action = AllocationAction::ENTER_NEW;
            decision.target_type = target_type;
            decision.target_weight = 1.0;
            decision.confidence = std::abs(current_probability - 0.5) * 2.0; // 0-1 scale
            
            std::ostringstream reason;
            reason << "Enter " << position_type_to_string(target_type) 
                   << " (prob=" << current_probability << ")";
            decision.reason = reason.str();
        } else {
            // Stay in cash
            decision.action = AllocationAction::HOLD;
            decision.target_type = PositionType::CASH;
            decision.target_weight = 0.0;
            decision.reason = "Signal too weak for entry";
            decision.confidence = 0.1;
        }
        
        return decision;
    }
    
    // **CASE 2: CURRENTLY HOLDING POSITION** - Consider hold/exit/transition
    
    bool is_opposing = is_signal_opposing(current_probability, current_state_.type);
    bool is_favorable = is_signal_favorable(current_probability, current_state_.type);
    
    // **RISK MANAGEMENT**: Force exit on extreme adverse signals
    if (is_opposing && std::abs(current_probability - 0.5) > config_.max_adverse_tolerance) {
        decision.action = AllocationAction::FULL_CLOSE;
        decision.target_type = PositionType::CASH;
        decision.target_weight = 0.0;
        decision.reason = "Risk management: Extreme adverse signal";
        decision.confidence = 0.9;
        return decision;
    }
    
    // **HOLDING PERIOD MANAGEMENT**: Force review after max holding period
    if (current_state_.bars_held >= config_.max_holding_period) {
        if (is_opposing) {
            decision.action = AllocationAction::FULL_CLOSE;
            decision.target_type = PositionType::CASH;
            decision.target_weight = 0.0;
            decision.reason = "Max holding period reached with opposing signal";
            decision.confidence = 0.7;
        } else {
            // Reset holding period but maintain position
            current_state_.bars_held = 0;
            decision.action = AllocationAction::HOLD;
            decision.target_type = current_state_.type;
            decision.target_weight = current_state_.weight;
            decision.reason = "Reset holding period, maintain position";
            decision.confidence = 0.3;
        }
        return decision;
    }
    
    if (is_opposing) {
        // **OPPOSING SIGNAL**: Consider partial or full exit
        
        // Calculate dynamic exit thresholds based on position strength
        double partial_threshold = calculate_exit_threshold(config_.partial_exit_threshold, current_state_);
        double full_threshold = calculate_exit_threshold(config_.full_exit_threshold, current_state_);
        
        double opposing_strength = std::abs(current_probability - 0.5) * 2.0;
        
        if (opposing_strength >= full_threshold) {
            // **FULL EXIT**: Strong opposing signal
            decision.action = AllocationAction::FULL_CLOSE;
            decision.target_type = PositionType::CASH;
            decision.target_weight = 0.0;
            decision.confidence = opposing_strength;
            
            std::ostringstream reason;
            reason << "Full exit: opposing signal " << current_probability 
                   << " > threshold " << full_threshold;
            decision.reason = reason.str();
            
        } else if (opposing_strength >= partial_threshold && 
                   (current_state_.type == PositionType::LONG_3X || current_state_.type == PositionType::INVERSE_3X)) {
            // **PARTIAL EXIT**: Reduce leverage (3x -> 1x)
            decision.action = AllocationAction::PARTIAL_CLOSE;
            decision.target_type = (current_state_.type == PositionType::LONG_3X) ? 
                                  PositionType::LONG_1X : PositionType::INVERSE_1X;
            decision.target_weight = 1.0;
            decision.confidence = opposing_strength * 0.7; // Lower confidence for partial moves
            
            std::ostringstream reason;
            reason << "Partial exit: " << position_type_to_string(current_state_.type)
                   << " -> " << position_type_to_string(decision.target_type);
            decision.reason = reason.str();
            
        } else {
            // **HOLD**: Opposing signal not strong enough
            decision.action = AllocationAction::HOLD;
            decision.target_type = current_state_.type;
            decision.target_weight = current_state_.weight;
            decision.confidence = 0.2;
            
            std::ostringstream reason;
            reason << "Hold despite opposing signal " << current_probability 
                   << " (threshold=" << partial_threshold << ")";
            decision.reason = reason.str();
        }
        
    } else if (is_favorable) {
        // **FAVORABLE SIGNAL**: Consider upgrading leverage
        
        double favorable_strength = std::abs(current_probability - 0.5) * 2.0;
        
        if (favorable_strength >= config_.entry_threshold_3x && 
            current_state_.type == PositionType::LONG_1X) {
            // **UPGRADE**: 1x -> 3x long
            decision.action = AllocationAction::ENTER_NEW;
            decision.target_type = PositionType::LONG_3X;
            decision.target_weight = 1.0;
            decision.confidence = favorable_strength;
            decision.reason = "Upgrade QQQ -> TQQQ on strong favorable signal";
            
        } else if (favorable_strength >= config_.entry_threshold_3x && 
                   current_state_.type == PositionType::INVERSE_1X) {
            // **UPGRADE**: 1x -> 3x inverse
            decision.action = AllocationAction::ENTER_NEW;
            decision.target_type = PositionType::INVERSE_3X;
            decision.target_weight = 1.0;
            decision.confidence = favorable_strength;
            decision.reason = "Upgrade PSQ -> SQQQ on strong favorable signal";
            
        } else {
            // **HOLD**: Current position appropriate for signal strength
            decision.action = AllocationAction::HOLD;
            decision.target_type = current_state_.type;
            decision.target_weight = current_state_.weight;
            decision.confidence = 0.4;
            decision.reason = "Hold: favorable signal confirms position";
        }
        
    } else {
        // **NEUTRAL SIGNAL**: Hold current position
        decision.action = AllocationAction::HOLD;
        decision.target_type = current_state_.type;
        decision.target_weight = current_state_.weight;
        decision.confidence = 0.3;
        decision.reason = "Hold: neutral signal";
    }
    
    return decision;
}

// **MATHEMATICAL THRESHOLD ADJUSTMENT**: Dynamic thresholds based on position strength
double AllocationManager::calculate_exit_threshold(double base_threshold, const PositionState& state) const {
    double adjusted_threshold = base_threshold;
    
    // **PROFIT PROTECTION**: Lower threshold if position is profitable
    if (state.unrealized_pnl_pct > 0) {
        adjusted_threshold -= config_.profit_protection_factor * state.unrealized_pnl_pct;
    } else {
        // **LOSS CUTTING**: Higher threshold if position is losing
        adjusted_threshold += config_.loss_cutting_factor * std::abs(state.unrealized_pnl_pct);
    }
    
    // **MOMENTUM DECAY**: Reduce threshold over time (easier to exit old positions)
    double time_decay = config_.momentum_decay_factor * (state.bars_held / 100.0);
    adjusted_threshold -= time_decay;
    
    // **BOUNDS**: Keep threshold reasonable
    adjusted_threshold = std::max(0.1, std::min(0.8, adjusted_threshold));
    
    return adjusted_threshold;
}

double AllocationManager::calculate_position_strength(const PositionState& state) const {
    // Combine P&L, holding period, and signal history
    double pnl_strength = std::tanh(state.unrealized_pnl_pct * 10.0); // -1 to 1
    double time_strength = std::exp(-state.bars_held / 100.0);        // Decay over time
    double signal_strength = state.max_favorable_prob - state.max_adverse_prob;
    
    return (pnl_strength + time_strength + signal_strength) / 3.0;
}

bool AllocationManager::is_signal_opposing(double probability, PositionType position_type) const {
    switch (position_type) {
        case PositionType::LONG_1X:
        case PositionType::LONG_3X:
            return probability < 0.5; // Bearish signal opposes long positions
        case PositionType::INVERSE_1X:
        case PositionType::INVERSE_3X:
            return probability > 0.5; // Bullish signal opposes inverse positions
        case PositionType::CASH:
            return false; // No position to oppose
    }
    return false;
}

bool AllocationManager::is_signal_favorable(double probability, PositionType position_type) const {
    switch (position_type) {
        case PositionType::LONG_1X:
        case PositionType::LONG_3X:
            return probability > 0.5; // Bullish signal favors long positions
        case PositionType::INVERSE_1X:
        case PositionType::INVERSE_3X:
            return probability < 0.5; // Bearish signal favors inverse positions
        case PositionType::CASH:
            return false; // No position to favor
    }
    return false;
}

PositionType AllocationManager::select_entry_position_type(double probability) const {
    double signal_strength = std::abs(probability - 0.5) * 2.0; // 0-1 scale
    
    if (probability > 0.5) {
        // **BULLISH SIGNAL**
        if (signal_strength >= config_.entry_threshold_3x) {
            return PositionType::LONG_3X; // TQQQ
        } else if (signal_strength >= config_.entry_threshold_1x) {
            return PositionType::LONG_1X; // QQQ
        }
    } else {
        // **BEARISH SIGNAL**
        if (signal_strength >= config_.entry_threshold_3x) {
            return PositionType::INVERSE_3X; // SQQQ
        } else if (signal_strength >= config_.entry_threshold_1x) {
            return PositionType::INVERSE_1X; // PSQ
        }
    }
    
    return PositionType::CASH; // Signal too weak
}

void AllocationManager::update_position_state(const AllocationDecision& decision, double probability) {
    if (decision.action == AllocationAction::ENTER_NEW || 
        decision.action == AllocationAction::PARTIAL_CLOSE) {
        
        // Update position
        current_state_.type = decision.target_type;
        current_state_.weight = decision.target_weight;
        current_state_.entry_probability = probability;
        current_state_.bars_held = 0;
        current_state_.max_favorable_prob = probability;
        current_state_.max_adverse_prob = 1.0 - probability;
        
    } else if (decision.action == AllocationAction::FULL_CLOSE) {
        // Reset to cash
        reset_state();
    }
    // HOLD action doesn't change position state (already updated in make_allocation_decision)
}

// **UTILITY FUNCTIONS**
std::string position_type_to_string(PositionType type) {
    switch (type) {
        case PositionType::CASH: return "CASH";
        case PositionType::LONG_1X: return "QQQ";
        case PositionType::LONG_3X: return "TQQQ";
        case PositionType::INVERSE_1X: return "PSQ";
        case PositionType::INVERSE_3X: return "SQQQ";
    }
    return "UNKNOWN";
}

std::string position_type_to_symbol(PositionType type) {
    return position_type_to_string(type); // Same for now
}

double get_position_leverage(PositionType type) {
    switch (type) {
        case PositionType::CASH: return 0.0;
        case PositionType::LONG_1X: return 1.0;
        case PositionType::LONG_3X: return 3.0;
        case PositionType::INVERSE_1X: return -1.0;
        case PositionType::INVERSE_3X: return -3.0;
    }
    return 0.0;
}

bool are_positions_conflicting(PositionType type1, PositionType type2) {
    if (type1 == PositionType::CASH || type2 == PositionType::CASH) return false;
    
    double lev1 = get_position_leverage(type1);
    double lev2 = get_position_leverage(type2);
    
    // Conflicting if different signs (long vs inverse)
    return (lev1 > 0 && lev2 < 0) || (lev1 < 0 && lev2 > 0);
}

} // namespace sentio
