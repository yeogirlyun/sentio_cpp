#pragma once
#include <deque>
#include <unordered_map>
#include <cstdint>
#include <cmath>

namespace sentio {

enum class TradingStyle {
    CONSERVATIVE,  // Low frequency, high conviction (like TFA)
    AGGRESSIVE,    // High frequency, many signals (like sigor)
    BURST,        // Intermittent high activity
    ADAPTIVE      // Changes behavior dynamically
};

class StrategyProfiler {
public:
    struct StrategyProfile {
        double avg_signal_frequency = 0.0;    // signals per bar
        double signal_volatility = 0.0;       // signal strength variance
        double signal_mean = 0.5;            // average signal value
        double noise_threshold = 0.0;        // auto-detected noise level
        double confidence_level = 0.0;       // profile confidence (0-1)
        TradingStyle style = TradingStyle::CONSERVATIVE;
        int observation_count = 0;
        double trades_per_block = 0.0;      // recent trading frequency
        
        // Adaptive thresholds based on observed behavior
        double adaptive_entry_1x = 0.60;    
        double adaptive_entry_3x = 0.75;
        double adaptive_noise_floor = 0.05;
    };
    
    StrategyProfiler();
    
    void observe_signal(double probability, int64_t timestamp);
    void observe_trade(double probability, const std::string& instrument, int64_t timestamp);
    void observe_block_complete(int trades_in_block);
    
    StrategyProfile get_current_profile() const { return profile_; }
    void reset_profile();
    
        private:
            static constexpr size_t WINDOW_SIZE = 500;  // Bars to analyze
            static constexpr size_t MIN_OBSERVATIONS = 50;  // Minimum for confidence
            
            StrategyProfile profile_;
            std::deque<double> signal_history_;
            std::deque<int64_t> signal_timestamps_;
            std::deque<double> trade_signals_;  // Signals that resulted in trades
            std::deque<int> block_trade_counts_;
            
            // **FIX**: Add hysteresis state tracking to prevent oscillation
            
            void update_profile();
            void detect_trading_style();
            void calculate_adaptive_thresholds();
            double calculate_noise_threshold();
};

} // namespace sentio
