#include "sentio/execution_verifier.hpp"
#include <iostream>
#include <algorithm>

namespace sentio {

ExecutionVerifier::ExecutionVerifier() {}

void ExecutionVerifier::cleanup_old_states(int64_t current_timestamp) {
    // Keep only the last 100 bars to prevent memory growth
    if (bar_states_.size() > 100) {
        auto cutoff_time = current_timestamp - (100 * 60 * 1000); // 100 minutes ago
        auto it = bar_states_.lower_bound(cutoff_time);
        bar_states_.erase(bar_states_.begin(), it);
    }
}

bool ExecutionVerifier::verify_can_execute(int64_t timestamp, const std::string& instrument) {
    cleanup_old_states(timestamp);
    
    auto& state = bar_states_[timestamp];
    state.timestamp = timestamp;
    
    // GOLDEN RULE ENFORCEMENT: EOD must be checked first
    if (!state.eod_checked) {
        throw std::runtime_error("GOLDEN RULE VIOLATION: EOD check must occur before any execution at timestamp " + std::to_string(timestamp));
    }
    
    // ENFORCEMENT: One trade per bar maximum
    if (state.trades_executed >= 1) {
        return false;
    }
    
    // ENFORCEMENT: No duplicate instrument trades
    if (!instrument.empty() && state.instruments_traded.count(instrument)) {
        return false;
    }
    
    return true;
}

void ExecutionVerifier::mark_eod_checked(int64_t timestamp) {
    auto& state = bar_states_[timestamp];
    state.eod_checked = true;
    state.timestamp = timestamp;
}

void ExecutionVerifier::mark_position_coordinated(int64_t timestamp) {
    auto& state = bar_states_[timestamp];
    state.position_coordinated = true;
    state.timestamp = timestamp;
}

void ExecutionVerifier::mark_trade_executed(int64_t timestamp, const std::string& instrument) {
    auto& state = bar_states_[timestamp];
    state.trades_executed++;
    state.instruments_traded.insert(instrument);
    state.timestamp = timestamp;
}

void ExecutionVerifier::reset_bar(int64_t timestamp) {
    current_bar_timestamp_ = timestamp;
    
    // Clear state for new bar
    auto& state = bar_states_[timestamp];
    state = BarState{};
    state.timestamp = timestamp;
}

ExecutionVerifier::BarState ExecutionVerifier::get_bar_state(int64_t timestamp) const {
    auto it = bar_states_.find(timestamp);
    if (it != bar_states_.end()) {
        return it->second;
    }
    return BarState{};
}

bool ExecutionVerifier::is_enforcement_active() const {
    return !bar_states_.empty();
}

} // namespace sentio
