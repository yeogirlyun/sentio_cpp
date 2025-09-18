#pragma once
#include <map>
#include <set>
#include <string>
#include <stdexcept>

namespace sentio {

/**
 * @brief ExecutionVerifier ensures the Golden Rule is enforced
 * 
 * This component verifies that:
 * 1. EOD checks occur before any execution
 * 2. One trade per bar rule is enforced
 * 3. No duplicate instrument trades per bar
 * 4. Safety systems cannot be bypassed
 */
class ExecutionVerifier {
private:
    struct BarState {
        bool eod_checked = false;
        bool position_coordinated = false;
        int trades_executed = 0;
        std::set<std::string> instruments_traded;
        int64_t timestamp = 0;
    };
    
    std::map<int64_t, BarState> bar_states_;
    int64_t current_bar_timestamp_ = -1;
    
    void cleanup_old_states(int64_t current_timestamp);
    
public:
    ExecutionVerifier();
    
    /**
     * @brief Verify if execution can proceed for this bar
     * @param timestamp Bar timestamp
     * @param instrument Instrument to trade (empty for general check)
     * @return true if execution is allowed
     * @throws std::runtime_error if Golden Rule is violated
     */
    bool verify_can_execute(int64_t timestamp, const std::string& instrument = "");
    
    /**
     * @brief Mark that EOD check has been performed for this bar
     * @param timestamp Bar timestamp
     */
    void mark_eod_checked(int64_t timestamp);
    
    /**
     * @brief Mark that position coordination has been performed
     * @param timestamp Bar timestamp
     */
    void mark_position_coordinated(int64_t timestamp);
    
    /**
     * @brief Mark that a trade has been executed
     * @param timestamp Bar timestamp
     * @param instrument Instrument traded
     */
    void mark_trade_executed(int64_t timestamp, const std::string& instrument);
    
    /**
     * @brief Reset state for new bar
     * @param timestamp New bar timestamp
     */
    void reset_bar(int64_t timestamp);
    
    /**
     * @brief Get current bar statistics for debugging
     */
    BarState get_bar_state(int64_t timestamp) const;
    
    /**
     * @brief Check if Golden Rule enforcement is working
     */
    bool is_enforcement_active() const;
};

} // namespace sentio
