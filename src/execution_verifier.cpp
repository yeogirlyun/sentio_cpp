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

bool ExecutionVerifier::verify_can_execute(int64_t timestamp, const std::string& instrument, bool is_closing_trade) {
    cleanup_old_states(timestamp);
    
    auto& state = bar_states_[timestamp];
    state.timestamp = timestamp;
    
    // GOLDEN RULE ENFORCEMENT: EOD must be checked first (now skipped since EOD removed)
    if (!state.eod_checked) {
        // Since EOD requirement was removed, automatically mark as checked
        state.eod_checked = true;
    }
    
    // Closing trades are ALWAYS allowed (no limit)
    if (is_closing_trade) {
        state.closing_trades_executed++;
        return true;
    }
    
    // Opening trades: enforce one per bar limit
    if (state.opening_trades_executed >= 1) {
        return false;  // Limit reached
    }
    
    // Check for duplicate instrument trades (same instrument in same bar)
    if (!instrument.empty() && state.instruments_traded.count(instrument)) {
        return false;  // Already traded this instrument
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

void ExecutionVerifier::mark_trade_executed(int64_t timestamp, const std::string& instrument, bool is_closing_trade) {
    auto& state = bar_states_[timestamp];
    
    if (is_closing_trade) {
        state.closing_trades_executed++;
    } else {
        state.opening_trades_executed++;
    }
    
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
