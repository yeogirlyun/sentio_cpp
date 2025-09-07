#pragma once
#include "signal_gate.hpp"
#include "router.hpp"
#include <unordered_map>
#include <vector>
#include <cmath>

namespace sentio {

/**
 * Cost-aware signal gate that filters signals based on expected transaction costs
 * and confidence thresholds derived from backtested cost curves.
 */
class CostAwareGate {
public:
    struct CostCurve {
        double base_cost_bp;           // Base transaction cost in basis points
        double confidence_threshold;   // Minimum confidence required to trade
        double min_expected_return_bp; // Minimum expected return to justify trade
        double max_position_size;      // Maximum position size as fraction of capital
    };

    struct CostAwareConfig {
        std::unordered_map<std::string, CostCurve> instrument_costs;
        double default_confidence_floor;
        double default_cost_bp;
        bool enable_cost_filtering;
        
        CostAwareConfig() 
            : default_confidence_floor(0.05)
            , default_cost_bp(2.0)
            , enable_cost_filtering(true) {}
    };

    explicit CostAwareGate(const CostAwareConfig& config = CostAwareConfig());

    /**
     * Filter signal based on cost analysis
     * @param signal Input signal to evaluate
     * @param instrument Instrument identifier
     * @param current_price Current market price
     * @param position_size Current position size
     * @return Filtered signal (may be modified or rejected)
     */
    std::optional<StrategySignal> filter_signal(
        const StrategySignal& signal,
        const std::string& instrument,
        double current_price,
        double position_size = 0.0
    ) const;

    /**
     * Calculate expected transaction cost for a signal
     * @param signal_type Type of signal (BUY/SELL)
     * @param instrument Instrument identifier
     * @param position_size Position size
     * @param current_price Current market price
     * @return Expected cost in basis points
     */
    double calculate_expected_cost(
        StrategySignal::Type signal_type,
        const std::string& instrument,
        double position_size,
        double current_price
    ) const;

    /**
     * Calculate minimum confidence required for profitable trade
     * @param instrument Instrument identifier
     * @param position_size Position size
     * @param current_price Current market price
     * @return Minimum confidence threshold
     */
    double calculate_min_confidence(
        const std::string& instrument,
        double position_size,
        double current_price
    ) const;

    /**
     * Update cost curve for an instrument based on recent performance
     * @param instrument Instrument identifier
     * @param recent_trades Vector of recent trade P&L data
     */
    void update_cost_curve(
        const std::string& instrument,
        const std::vector<double>& recent_trades
    );

    /**
     * Check if signal should be rejected due to cost constraints
     * @param signal Input signal
     * @param instrument Instrument identifier
     * @param current_price Current market price
     * @param position_size Current position size
     * @return True if signal should be rejected
     */
    bool should_reject_signal(
        const StrategySignal& signal,
        const std::string& instrument,
        double current_price,
        double position_size = 0.0
    ) const;

private:
    CostAwareConfig config_;
    
    // Helper methods
    const CostCurve& get_cost_curve(const std::string& instrument) const;
    double calculate_slippage_cost(double position_size, double current_price) const;
    double calculate_market_impact(double position_size, double current_price) const;
};

} // namespace sentio
