#include "sentio/cost_aware_gate.hpp"
// StrategySignal is defined in router.hpp which is already included via cost_aware_gate.hpp
#include <algorithm>
#include <cmath>

namespace sentio {

CostAwareGate::CostAwareGate(const CostAwareConfig& config) : config_(config) {
    // Initialize default cost curves for common instruments if not provided
    if (config_.instrument_costs.empty()) {
        config_.instrument_costs["QQQ"] = {2.0, 0.05, 5.0, 0.1};
        config_.instrument_costs["SPY"] = {1.5, 0.05, 4.0, 0.1};
        config_.instrument_costs["IWM"] = {2.5, 0.05, 6.0, 0.1};
        config_.instrument_costs["TLT"] = {3.0, 0.05, 7.0, 0.1};
        config_.instrument_costs["GLD"] = {2.0, 0.05, 5.0, 0.1};
    }
}

std::optional<StrategySignal> CostAwareGate::filter_signal(
    const StrategySignal& signal,
    const std::string& instrument,
    double current_price,
    double position_size
) const {
    if (!config_.enable_cost_filtering) {
        return signal;
    }

    // Check if signal should be rejected due to cost constraints
    if (should_reject_signal(signal, instrument, current_price, position_size)) {
        return std::nullopt;
    }

    // Calculate minimum confidence required
    double min_confidence = calculate_min_confidence(instrument, position_size, current_price);
    
    // If signal confidence is below threshold, reject it
    if (signal.confidence < min_confidence) {
        return std::nullopt;
    }

    // Signal passes cost analysis
    return signal;
}

double CostAwareGate::calculate_expected_cost(
    [[maybe_unused]] StrategySignal::Type signal_type,
    const std::string& instrument,
    double position_size,
    double current_price
) const {
    const auto& curve = get_cost_curve(instrument);
    
    // Base transaction cost
    double base_cost = curve.base_cost_bp;
    
    // Add slippage cost (increases with position size)
    double slippage_cost = calculate_slippage_cost(position_size, current_price);
    
    // Add market impact cost
    double impact_cost = calculate_market_impact(position_size, current_price);
    
    return base_cost + slippage_cost + impact_cost;
}

double CostAwareGate::calculate_min_confidence(
    const std::string& instrument,
    double position_size,
    double current_price
) const {
    const auto& curve = get_cost_curve(instrument);
    
    // Calculate expected cost
    double expected_cost = calculate_expected_cost(
        StrategySignal::Type::BUY, // Use BUY as reference
        instrument,
        position_size,
        current_price
    );
    
    // Minimum confidence should be proportional to expected cost
    // Higher costs require higher confidence
    double cost_factor = expected_cost / curve.base_cost_bp;
    double min_confidence = curve.confidence_threshold * cost_factor;
    
    // Ensure minimum confidence floor
    return std::max(min_confidence, config_.default_confidence_floor);
}

void CostAwareGate::update_cost_curve(
    const std::string& instrument,
    const std::vector<double>& recent_trades
) {
    if (recent_trades.empty()) return;
    
    // Calculate average P&L per trade
    double avg_pnl = 0.0;
    for (double pnl : recent_trades) {
        avg_pnl += pnl;
    }
    avg_pnl /= recent_trades.size();
    
    // Calculate P&L volatility
    double pnl_variance = 0.0;
    for (double pnl : recent_trades) {
        double diff = pnl - avg_pnl;
        pnl_variance += diff * diff;
    }
    pnl_variance /= recent_trades.size();
    double pnl_std = std::sqrt(pnl_variance);
    
    // Update cost curve based on recent performance
    auto& curve = config_.instrument_costs[instrument];
    
    // Adjust confidence threshold based on P&L consistency
    if (pnl_std > 0) {
        double sharpe_ratio = avg_pnl / pnl_std;
        curve.confidence_threshold = std::max(0.05, 0.1 / std::max(1.0, sharpe_ratio));
    }
    
    // Adjust base cost based on recent performance
    if (avg_pnl < 0) {
        curve.base_cost_bp *= 1.1; // Increase cost estimate if losing money
    } else if (avg_pnl > curve.min_expected_return_bp) {
        curve.base_cost_bp *= 0.95; // Decrease cost estimate if profitable
    }
}

bool CostAwareGate::should_reject_signal(
    const StrategySignal& signal,
    const std::string& instrument,
    double current_price,
    double position_size
) const {
    const auto& curve = get_cost_curve(instrument);
    
    // Check position size limits
    if (std::abs(position_size) > curve.max_position_size) {
        return true;
    }
    
    // Check minimum expected return
    double expected_cost = calculate_expected_cost(
        signal.type,
        instrument,
        position_size,
        current_price
    );
    
    // Estimate expected return based on signal confidence
    double estimated_return = signal.confidence * curve.min_expected_return_bp;
    
    // Reject if expected return is less than expected cost
    return estimated_return < expected_cost;
}

const CostAwareGate::CostCurve& CostAwareGate::get_cost_curve(const std::string& instrument) const {
    auto it = config_.instrument_costs.find(instrument);
    if (it != config_.instrument_costs.end()) {
        return it->second;
    }
    
    // Return default cost curve
    static CostCurve default_curve{config_.default_cost_bp, config_.default_confidence_floor, 5.0, 0.1};
    return default_curve;
}

double CostAwareGate::calculate_slippage_cost(double position_size, [[maybe_unused]] double current_price) const {
    // Slippage increases with position size (simplified model)
    double size_factor = std::abs(position_size) / 0.1; // Normalize to 10% position size
    return 0.5 * size_factor; // 0.5 bp per 10% position size
}

double CostAwareGate::calculate_market_impact(double position_size, [[maybe_unused]] double current_price) const {
    // Market impact increases quadratically with position size
    double size_factor = std::abs(position_size) / 0.1; // Normalize to 10% position size
    return 0.1 * size_factor * size_factor; // Quadratic impact
}

} // namespace sentio
