#pragma once
#include <deque>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sentio {

/**
 * IntradayPositionGovernor - Dynamic day trading position management
 * 
 * Converts probability stream to target weights with:
 * 1. Dynamic percentile-based thresholds (no fixed config)
 * 2. Time-based position decay for graceful EOD exit
 * 3. Signal-adaptive sizing for 10-100 trades/day target
 */
class IntradayPositionGovernor {
public:
    struct Config {
        size_t lookback_window = 60;      // Rolling window for dynamic thresholds (minutes)
        double buy_percentile = 0.85;     // Enter long if p > 85th percentile of recent values
        double sell_percentile = 0.15;    // Enter short if p < 15th percentile
        double max_base_weight = 1.0;     // Maximum position weight before time decay
        double trading_day_hours = 6.5;   // NYSE trading hours (9:30-4:00 ET)
        double min_signal_gap = 0.02;     // Minimum gap between buy/sell thresholds
        double min_abs_edge = 0.05;       // Minimum absolute edge from 0.5 to trade (noise filter)
    };

    IntradayPositionGovernor() : IntradayPositionGovernor(Config{}) {}
    
    explicit IntradayPositionGovernor(const Config& cfg) 
        : config_(cfg)
        , market_open_epoch_(0)
        , trading_day_duration_sec_(cfg.trading_day_hours * 3600.0)
        , last_target_weight_(0.0)
    {}

    /**
     * Main decision function: Convert probability + timestamp → target weight
     * 
     * @param probability IRE ensemble probability [0.0, 1.0]
     * @param current_epoch UTC timestamp of current bar
     * @return Target portfolio weight [-1.0, +1.0] with time decay applied
     */
    double calculate_target_weight(double probability, int64_t current_epoch) {
        // Initialize market open time on first call each day
        detect_new_trading_day(current_epoch);
        
        // Update rolling probability window for dynamic thresholds
        update_probability_history(probability);
        
        // Need sufficient history for reliable thresholds
        if (rolling_p_values_.size() < config_.lookback_window) {
            return apply_time_decay(0.0, current_epoch); // Conservative start
        }

        // Calculate adaptive thresholds based on recent market conditions
        auto [buy_threshold, sell_threshold] = calculate_dynamic_thresholds();
        
        // Determine base position weight from signal strength
        double base_weight = calculate_base_weight(probability, buy_threshold, sell_threshold);
        
        // Apply time decay for graceful position closure toward market close
        double final_weight = apply_time_decay(base_weight, current_epoch);
        
        last_target_weight_ = final_weight;
        return final_weight;
    }

    // Diagnostics for strategy analysis
    double get_current_buy_threshold() const {
        if (rolling_p_values_.size() < config_.lookback_window) return 0.85;
        return calculate_dynamic_thresholds().first;
    }
    
    double get_current_sell_threshold() const {
        if (rolling_p_values_.size() < config_.lookback_window) return 0.15;
        return calculate_dynamic_thresholds().second;
    }
    
    double get_time_decay_multiplier(int64_t current_epoch) const {
        if (market_open_epoch_ == 0) return 1.0;
        double time_into_day = static_cast<double>(current_epoch - market_open_epoch_);
        double time_ratio = std::clamp(time_into_day / trading_day_duration_sec_, 0.0, 1.0);
        return std::cos(time_ratio * M_PI / 2.0); // Smooth decay to 0 at market close
    }

    // Reset for new trading session
    void reset_for_new_day() {
        rolling_p_values_.clear();
        market_open_epoch_ = 0;
        last_target_weight_ = 0.0;
    }

private:
    Config config_;
    int64_t market_open_epoch_;
    double trading_day_duration_sec_;
    double last_target_weight_;
    std::deque<double> rolling_p_values_;

    void detect_new_trading_day(int64_t current_epoch) {
        // Simple day detection - could be enhanced with proper calendar
        const int64_t SECONDS_PER_DAY = 86400;
        int64_t current_day = current_epoch / SECONDS_PER_DAY;
        static int64_t last_day = -1;
        
        if (last_day != current_day) {
            reset_for_new_day();
            market_open_epoch_ = current_epoch;
            last_day = current_day;
        } else if (market_open_epoch_ == 0) {
            market_open_epoch_ = current_epoch;
        }
    }

    void update_probability_history(double probability) {
        if (std::isfinite(probability)) {
            rolling_p_values_.push_back(std::clamp(probability, 0.0, 1.0));
            
            // Maintain fixed window size
            if (rolling_p_values_.size() > config_.lookback_window) {
                rolling_p_values_.pop_front();
            }
        }
    }

    std::pair<double, double> calculate_dynamic_thresholds() const {
        // Copy for sorting without modifying original
        std::vector<double> sorted_p(rolling_p_values_.begin(), rolling_p_values_.end());
        std::sort(sorted_p.begin(), sorted_p.end());
        
        size_t n = sorted_p.size();
        size_t buy_idx = static_cast<size_t>(n * config_.buy_percentile);
        size_t sell_idx = static_cast<size_t>(n * config_.sell_percentile);
        
        // Ensure valid indices
        buy_idx = std::min(buy_idx, n - 1);
        sell_idx = std::min(sell_idx, n - 1);
        
        double buy_threshold = sorted_p[buy_idx];
        double sell_threshold = sorted_p[sell_idx];
        
        // **FIX**: Handle degenerate case when all values are identical
        double signal_range = sorted_p[n-1] - sorted_p[0];
        if (signal_range < 0.001) {  // Nearly constant signals
            // Use fixed thresholds around the constant value
            double center = sorted_p[0];
            buy_threshold = std::min(center + 0.1, 0.9);   // Don't exceed 0.9
            sell_threshold = std::max(center - 0.1, 0.1);  // Don't go below 0.1
        }
        
        // Ensure minimum separation to prevent thrashing
        if (buy_threshold - sell_threshold < config_.min_signal_gap) {
            double mid = (buy_threshold + sell_threshold) * 0.5;
            buy_threshold = mid + config_.min_signal_gap * 0.5;
            sell_threshold = mid - config_.min_signal_gap * 0.5;
        }
        
        // **FIX**: Ensure thresholds are achievable 
        buy_threshold = std::clamp(buy_threshold, 0.0, 0.99);  // Must be < 1.0
        sell_threshold = std::clamp(sell_threshold, 0.01, 1.0); // Must be > 0.0
        
        return {buy_threshold, sell_threshold};
    }

    double calculate_base_weight(double probability, double buy_threshold, double sell_threshold) const {
        // **FIXED**: Now actually use the dynamic thresholds!
        
        // Apply min_abs_edge as an additional filter on top of dynamic thresholds
        double abs_edge_from_neutral = std::abs(probability - 0.5);
        if (abs_edge_from_neutral < config_.min_abs_edge) {
            // Signal too weak - stay flat
            return 0.0; 
        }
        
        // **CORRECTED**: Use dynamic thresholds that adapt to market conditions
        if (probability > buy_threshold) {
            // Strong long signal based on recent probability distribution
            double signal_strength = (probability - 0.5) / 0.5;
            return std::min(config_.max_base_weight, signal_strength);
        } 
        else if (probability < sell_threshold) {
            // Strong short signal based on recent probability distribution  
            double signal_strength = (0.5 - probability) / 0.5;
            return -std::min(config_.max_base_weight, signal_strength);
        }
        else {
            // Signal between thresholds - stay flat
            return 0.0;
        }
    }

    double apply_time_decay(double base_weight, int64_t current_epoch) const {
        if (market_open_epoch_ == 0) return base_weight;
        
        double time_into_day = static_cast<double>(current_epoch - market_open_epoch_);
        double time_ratio = std::clamp(time_into_day / trading_day_duration_sec_, 0.0, 1.0);
        
        // Cosine decay: 1.0 at open → 0.0 at close
        // Provides gentle early decay, aggressive late-day closure
        double time_decay_multiplier = std::cos(time_ratio * M_PI / 2.0);
        
        return base_weight * time_decay_multiplier;
    }
};

} // namespace sentio
