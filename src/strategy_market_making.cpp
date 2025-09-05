#include "sentio/strategy_market_making.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace sentio {

MarketMakingStrategy::MarketMakingStrategy() 
    : BaseStrategy("MarketMaking"),
      rolling_returns_(20),
      rolling_volume_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap MarketMakingStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed volatility and volume thresholds to allow participation.
    return {
        {"base_spread", 0.001}, {"min_spread", 0.0005}, {"max_spread", 0.003},
        {"order_levels", 3.0}, {"level_spacing", 0.0005}, {"order_size_base", 0.5},
        {"max_inventory", 100.0}, {"inventory_skew_mult", 0.002},
        {"adverse_selection_threshold", 0.004}, // Was 0.002, allowing participation in more volatile conditions
        {"volatility_window", 20.0},
        {"volume_window", 50.0}, {"min_volume_ratio", 0.05}, // Was 0.1, making it even more permissive
        {"max_orders_per_bar", 10.0}, {"rebalance_frequency", 10.0}
    };
}

ParameterSpace MarketMakingStrategy::get_param_space() const { return {}; }

void MarketMakingStrategy::apply_params() {
    base_spread_ = params_.at("base_spread");
    min_spread_ = params_.at("min_spread");
    max_spread_ = params_.at("max_spread");
    order_levels_ = static_cast<int>(params_.at("order_levels"));
    level_spacing_ = params_.at("level_spacing");
    order_size_base_ = params_.at("order_size_base");
    max_inventory_ = params_.at("max_inventory");
    inventory_skew_mult_ = params_.at("inventory_skew_mult");
    adverse_selection_threshold_ = params_.at("adverse_selection_threshold");
    min_volume_ratio_ = params_.at("min_volume_ratio");
    max_orders_per_bar_ = static_cast<int>(params_.at("max_orders_per_bar"));
    rebalance_frequency_ = static_cast<int>(params_.at("rebalance_frequency"));

    int vol_window = std::max(1, static_cast<int>(params_.at("volatility_window")));
    int vol_mean_window = std::max(1, static_cast<int>(params_.at("volume_window")));
    
    rolling_returns_.reset(vol_window);
    rolling_volume_.reset(vol_mean_window);
    reset_state();
}

void MarketMakingStrategy::reset_state() {
    BaseStrategy::reset_state();
    market_state_ = MarketState{};
    rolling_returns_.reset(std::max(1, static_cast<int>(params_.at("volatility_window"))));
    rolling_volume_.reset(std::max(1, static_cast<int>(params_.at("volume_window"))));
}

StrategySignal MarketMakingStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;
    
    // Always update indicators to have a full history for the next bar
    if(current_index > 0) {
        double price_return = (bars[current_index].close - bars[current_index - 1].close) / bars[current_index - 1].close;
        rolling_returns_.push(price_return);
    }
    rolling_volume_.push(bars[current_index].volume);

    // Wait for indicators to warm up
    if (rolling_volume_.size() < static_cast<size_t>(params_.at("volume_window"))) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }

    if (!should_participate(bars[current_index])) {
        return signal;
    }
    
    double inventory_skew = get_inventory_skew();
    
    // **FIXED**: Generate signals based on volatility and volume patterns instead of inventory
    // Since inventory tracking is not implemented, use a simpler approach
    double volatility = rolling_returns_.stddev();
    double avg_volume = rolling_volume_.mean();
    double volume_ratio = (avg_volume > 0) ? bars[current_index].volume / avg_volume : 0.0;
    
    // Generate signals when volatility is moderate and volume is increasing
    if (volatility > 0.0005 && volatility < adverse_selection_threshold_ && volume_ratio > 0.8) {
        // Simple momentum-based signal
        if (current_index > 0) {
            double price_change = (bars[current_index].close - bars[current_index - 1].close) / bars[current_index - 1].close;
            if (price_change > 0.001) {
                signal.type = StrategySignal::Type::BUY;
            } else if (price_change < -0.001) {
                signal.type = StrategySignal::Type::SELL;
            } else {
                diag_.drop(DropReason::THRESHOLD);
                return signal;
            }
        } else {
            diag_.drop(DropReason::THRESHOLD);
            return signal;
        }
    } else {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }

    signal.confidence = 0.6; // Fixed confidence since we're not using inventory_skew
    diag_.emitted++;
    return signal;
}

bool MarketMakingStrategy::should_participate(const Bar& bar) {
    double volatility = rolling_returns_.stddev();
    
    if (volatility > adverse_selection_threshold_) {
        diag_.drop(DropReason::THRESHOLD); 
        return false;
    }

    double avg_volume = rolling_volume_.mean();
    
    if (avg_volume > 0 && (bar.volume < avg_volume * min_volume_ratio_)) {
        diag_.drop(DropReason::ZERO_VOL);
        return false;
    }
    return true;
}

double MarketMakingStrategy::get_inventory_skew() const {
    if (max_inventory_ <= 0) return 0.0;
    double normalized_inventory = market_state_.inventory / max_inventory_;
    return -normalized_inventory * inventory_skew_mult_;
}

REGISTER_STRATEGY(MarketMakingStrategy, "MarketMaking");

} // namespace sentio

