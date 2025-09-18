#include "sentio/strategy_profiler.hpp"
#include <algorithm>
#include <numeric>

namespace sentio {

StrategyProfiler::StrategyProfiler() {
    reset_profile();
}

void StrategyProfiler::reset_profile() {
    profile_ = StrategyProfile();
    signal_history_.clear();
    signal_timestamps_.clear();
    trade_signals_.clear();
    block_trade_counts_.clear();
}

void StrategyProfiler::observe_signal(double probability, int64_t timestamp) {
    signal_history_.push_back(probability);
    signal_timestamps_.push_back(timestamp);
    
    // Maintain window size
    while (signal_history_.size() > WINDOW_SIZE) {
        signal_history_.pop_front();
        signal_timestamps_.pop_front();
    }
    
    profile_.observation_count++;
    update_profile();
}

void StrategyProfiler::observe_trade(double probability, const std::string& instrument, int64_t timestamp) {
    trade_signals_.push_back(probability);
    
    while (trade_signals_.size() > WINDOW_SIZE) {
        trade_signals_.pop_front();
    }
}

void StrategyProfiler::observe_block_complete(int trades_in_block) {
    block_trade_counts_.push_back(trades_in_block);
    
    // Keep last 10 blocks for recent behavior
    while (block_trade_counts_.size() > 10) {
        block_trade_counts_.pop_front();
    }
    
    if (!block_trade_counts_.empty()) {
        profile_.trades_per_block = std::accumulate(block_trade_counts_.begin(), 
                                                   block_trade_counts_.end(), 0.0) 
                                  / block_trade_counts_.size();
    }
}

void StrategyProfiler::update_profile() {
    if (signal_history_.size() < MIN_OBSERVATIONS) {
        profile_.confidence_level = signal_history_.size() / double(MIN_OBSERVATIONS);
        return;
    }
    
    // Calculate signal frequency (signals that deviate from 0.5)
    int active_signals = 0;
    for (double prob : signal_history_) {
        if (std::abs(prob - 0.5) > 0.05) {  // Signal threshold
            active_signals++;
        }
    }
    profile_.avg_signal_frequency = double(active_signals) / signal_history_.size();
    
    // Calculate mean and volatility
    double sum = std::accumulate(signal_history_.begin(), signal_history_.end(), 0.0);
    profile_.signal_mean = sum / signal_history_.size();
    
    double sq_sum = 0.0;
    for (double prob : signal_history_) {
        double diff = prob - profile_.signal_mean;
        sq_sum += diff * diff;
    }
    profile_.signal_volatility = std::sqrt(sq_sum / signal_history_.size());
    
    // Update noise threshold and trading style
    profile_.noise_threshold = calculate_noise_threshold();
    detect_trading_style();
    calculate_adaptive_thresholds();
    
    profile_.confidence_level = std::min(1.0, signal_history_.size() / double(WINDOW_SIZE));
}

void StrategyProfiler::detect_trading_style() {
    // --- BUG FIX: Simplified and More Direct Style Detection ---
    // The previous hysteresis logic was getting stuck and misclassifying.
    // This new logic is a direct mapping from recent behavior.
    
    if (block_trade_counts_.empty()) {
        profile_.style = TradingStyle::CONSERVATIVE; // Default style
        return;
    }

    // Use the most recent block's trade count as the primary indicator.
    double recent_trades = block_trade_counts_.back();

    // Define clear, non-overlapping thresholds.
    constexpr double AGGRESSIVE_THRESHOLD = 100.0;
    constexpr double CONSERVATIVE_THRESHOLD = 20.0;
    constexpr double BURST_VOLATILITY_THRESHOLD = 0.25;

    if (recent_trades > AGGRESSIVE_THRESHOLD) {
        profile_.style = TradingStyle::AGGRESSIVE;
    } else if (recent_trades < CONSERVATIVE_THRESHOLD) {
        profile_.style = TradingStyle::CONSERVATIVE;
    } else if (profile_.signal_volatility > BURST_VOLATILITY_THRESHOLD) {
        // Moderate trade count but high signal volatility indicates a "bursty" strategy.
        profile_.style = TradingStyle::BURST;
    } else {
        // In-between behavior is classified as adaptive.
        profile_.style = TradingStyle::ADAPTIVE;
    }
}

void StrategyProfiler::calculate_adaptive_thresholds() {
    // Adapt thresholds based on observed trading style
    switch (profile_.style) {
        case TradingStyle::AGGRESSIVE:
            // Higher thresholds for aggressive strategies to filter noise
            profile_.adaptive_entry_1x = 0.65;
            profile_.adaptive_entry_3x = 0.80;
            profile_.adaptive_noise_floor = 0.10;
            break;
            
        case TradingStyle::CONSERVATIVE:
            // Lower thresholds for conservative strategies
            profile_.adaptive_entry_1x = 0.55;
            profile_.adaptive_entry_3x = 0.70;
            profile_.adaptive_noise_floor = 0.02;
            break;
            
        case TradingStyle::BURST:
            // Dynamic thresholds for burst strategies
            profile_.adaptive_entry_1x = 0.60;
            profile_.adaptive_entry_3x = 0.75;
            profile_.adaptive_noise_floor = 0.05;
            break;
            
        default:
            // Keep defaults for adaptive
            break;
    }
    
    // Further adjust based on signal volatility
    if (profile_.signal_volatility > 0.15) {
        profile_.adaptive_noise_floor *= 1.5;  // Increase noise floor for volatile signals
    }
}

double StrategyProfiler::calculate_noise_threshold() {
    if (signal_history_.size() < MIN_OBSERVATIONS) {
        return 0.05;  // Default
    }
    
    // Calculate signal change frequency to detect noise
    std::vector<double> signal_changes;
    for (size_t i = 1; i < signal_history_.size(); ++i) {
        signal_changes.push_back(std::abs(signal_history_[i] - signal_history_[i-1]));
    }
    
    if (signal_changes.empty()) return 0.05;
    
    // Noise threshold is the 25th percentile of signal changes
    std::sort(signal_changes.begin(), signal_changes.end());
    size_t idx = signal_changes.size() / 4;
    return signal_changes[idx];
}

} // namespace sentio
