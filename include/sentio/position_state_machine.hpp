#pragma once

#include <string>
#include <unordered_map>
#include <chrono>

namespace sentio {

/**
 * @brief Explicit state machine for position lifecycle management
 * 
 * Addresses the core architectural issue: implicit state transitions
 * led to conflicts and EOD violations. This provides explicit,
 * validated state transitions with enforcement.
 */
enum class PositionState {
    CLOSED = 0,      // No position, ready for new trades
    OPENING,         // Order placed, waiting for fill
    OPEN,            // Position established, normal trading
    CLOSING,         // Close order placed, waiting for fill
    CONFLICTED,      // Detected conflict, requires resolution
    EOD_CLOSING,     // Mandatory EOD closure in progress
    FROZEN           // System halt due to violations
};

enum class PositionEvent {
    OPEN_ORDER,      // Strategy wants to open position
    FILL_OPEN,       // Open order filled
    CLOSE_ORDER,     // Strategy wants to close position
    FILL_CLOSE,      // Close order filled
    CONFLICT_DETECTED, // Conflict with other positions
    EOD_TRIGGER,     // End of day closure required
    VIOLATION_HALT,  // System safety halt
    CONFLICT_RESOLVED // Conflict manually resolved
};

/**
 * @brief State machine for individual position lifecycle
 */
class PositionStateMachine {
public:
    explicit PositionStateMachine(const std::string& symbol) 
        : symbol_(symbol), state_(PositionState::CLOSED) {}
    
    // Core state transition with validation
    bool transition(PositionEvent event);
    
    // State queries
    PositionState get_state() const { return state_; }
    bool can_open() const { return state_ == PositionState::CLOSED; }
    bool can_close() const { 
        return state_ == PositionState::OPEN || 
               state_ == PositionState::CONFLICTED; 
    }
    bool is_open() const { 
        return state_ == PositionState::OPEN || 
               state_ == PositionState::CONFLICTED; 
    }
    bool requires_eod_close() const {
        return state_ == PositionState::OPEN || 
               state_ == PositionState::CONFLICTED ||
               state_ == PositionState::OPENING;
    }
    bool is_frozen() const { return state_ == PositionState::FROZEN; }
    
    // Metadata
    const std::string& symbol() const { return symbol_; }
    std::chrono::system_clock::time_point last_transition() const { return last_transition_; }
    
private:
    std::string symbol_;
    PositionState state_;
    std::chrono::system_clock::time_point last_transition_;
    
    bool is_valid_transition(PositionState from, PositionEvent event, PositionState to) const;
};

/**
 * @brief Central orchestrator for all position state management
 * 
 * This is the "single source of truth" that enforces invariants
 * across all components, addressing the integration failure.
 */
class PositionOrchestrator {
public:
    PositionOrchestrator() = default;
    
    // Core orchestration methods
    bool can_open_position(const std::string& symbol) const;
    bool can_close_position(const std::string& symbol) const;
    
    // State transitions with conflict checking
    bool request_open(const std::string& symbol);
    bool confirm_open(const std::string& symbol);
    bool request_close(const std::string& symbol);
    bool confirm_close(const std::string& symbol);
    
    // Conflict management
    bool detect_conflicts() const;
    std::vector<std::string> get_conflicted_positions() const;
    bool resolve_conflict(const std::string& symbol);
    
    // EOD management
    std::vector<std::string> get_eod_required_closes() const;
    bool trigger_eod_closure();
    
    // Safety circuit breaker
    bool should_halt_trading() const;
    void emergency_halt();
    
    // State queries
    PositionState get_position_state(const std::string& symbol) const;
    std::vector<std::string> get_open_positions() const;
    size_t get_conflict_count() const;
    
private:
    std::unordered_map<std::string, PositionStateMachine> positions_;
    size_t max_conflicts_ = 5;  // Circuit breaker threshold
    bool emergency_halt_ = false;
    
    bool would_create_directional_conflict(const std::string& new_symbol) const;
    void update_conflict_states();
};

} // namespace sentio
