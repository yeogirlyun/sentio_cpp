#pragma once

#include "sentio/base_strategy.hpp"
#include <algorithm>
#include <cmath>

namespace sentio::signal_utils {

/**
 * @brief Converts a strategy signal to probability
 * @param sig The strategy signal
 * @param conf_floor Minimum confidence floor
 * @return Probability value (0.0 to 1.0)
 */
inline double signal_to_probability(const StrategySignal& sig, double conf_floor = 0.0) {
    if (sig.confidence < conf_floor) return 0.5; // Neutral
    
    double probability;
    if (sig.type == StrategySignal::Type::BUY) {
        probability = 0.5 + sig.confidence * 0.5; // 0.5 to 1.0
    } else if (sig.type == StrategySignal::Type::SELL) {
        probability = 0.5 - sig.confidence * 0.5; // 0.0 to 0.5
    } else {
        probability = 0.5; // HOLD
    }
    
    return std::clamp(probability, 0.0, 1.0);
}

/**
 * @brief Converts a strategy signal to probability with custom scaling
 * @param sig The strategy signal
 * @param conf_floor Minimum confidence floor
 * @param buy_scale Scaling factor for buy signals
 * @param sell_scale Scaling factor for sell signals
 * @return Probability value (0.0 to 1.0)
 */
inline double signal_to_probability_custom(const StrategySignal& sig, double conf_floor = 0.0,
                                         double buy_scale = 0.5, double sell_scale = 0.5) {
    if (sig.confidence < conf_floor) return 0.5; // Neutral
    
    double probability;
    if (sig.type == StrategySignal::Type::BUY) {
        probability = 0.5 + sig.confidence * buy_scale; // 0.5 to 1.0
    } else if (sig.type == StrategySignal::Type::SELL) {
        probability = 0.5 - sig.confidence * sell_scale; // 0.0 to 0.5
    } else {
        probability = 0.5; // HOLD
    }
    
    return std::clamp(probability, 0.0, 1.0);
}

/**
 * @brief Validates if a signal has sufficient confidence
 * @param sig The strategy signal
 * @param conf_floor Minimum confidence floor
 * @return true if signal has sufficient confidence, false otherwise
 */
inline bool has_sufficient_confidence(const StrategySignal& sig, double conf_floor = 0.0) {
    return sig.confidence >= conf_floor;
}

/**
 * @brief Gets the signal strength (absolute confidence)
 * @param sig The strategy signal
 * @return Signal strength (0.0 to 1.0)
 */
inline double get_signal_strength(const StrategySignal& sig) {
    return std::abs(sig.confidence);
}

/**
 * @brief Determines if signal is a buy signal
 * @param sig The strategy signal
 * @return true if buy signal, false otherwise
 */
inline bool is_buy_signal(const StrategySignal& sig) {
    return sig.type == StrategySignal::Type::BUY;
}

/**
 * @brief Determines if signal is a sell signal
 * @param sig The strategy signal
 * @return true if sell signal, false otherwise
 */
inline bool is_sell_signal(const StrategySignal& sig) {
    return sig.type == StrategySignal::Type::SELL;
}

/**
 * @brief Determines if signal is a hold signal
 * @param sig The strategy signal
 * @return true if hold signal, false otherwise
 */
inline bool is_hold_signal(const StrategySignal& sig) {
    return sig.type == StrategySignal::Type::HOLD;
}

/**
 * @brief Gets the signal direction (-1 for sell, 0 for hold, +1 for buy)
 * @param sig The strategy signal
 * @return Signal direction
 */
inline int get_signal_direction(const StrategySignal& sig) {
    if (sig.type == StrategySignal::Type::BUY) return 1;
    if (sig.type == StrategySignal::Type::SELL) return -1;
    return 0; // HOLD
}

/**
 * @brief Applies confidence floor to a signal
 * @param sig The strategy signal
 * @param conf_floor Minimum confidence floor
 * @return Modified signal with confidence floor applied
 */
inline StrategySignal apply_confidence_floor(const StrategySignal& sig, double conf_floor = 0.0) {
    StrategySignal result = sig;
    if (sig.confidence < conf_floor) {
        result.type = StrategySignal::Type::HOLD;
        result.confidence = 0.0;
    }
    return result;
}

} // namespace sentio::signal_utils
