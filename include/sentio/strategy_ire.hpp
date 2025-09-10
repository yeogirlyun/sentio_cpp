#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/rules/integrated_rule_ensemble.hpp"
#include "sentio/strategy/intraday_position_governor.hpp"
#include "sentio/router.hpp"
#include "sentio/sizer.hpp"
#include <deque>

namespace sentio {

class IREStrategy : public BaseStrategy {
public:
    IREStrategy();
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    // The main entry point for the runner is now this function
    double calculate_target_weight(const std::vector<Bar>& bars, int i);
    
    // **NEW**: Implement probability-based signal interface
    double calculate_probability(const std::vector<Bar>& bars, int i) override;
    
    // **NEW**: Strategy-agnostic allocation interface
    std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, 
        int current_index,
        const std::string& base_symbol,
        const std::string& bull3x_symbol,
        const std::string& bear3x_symbol,
        const std::string& bear1x_symbol) override;
    
    RouterCfg get_router_config() const override;
    SizerCfg get_sizer_config() const override;
    bool requires_dynamic_allocation() const override { return true; }

    double get_latest_probability() const { return latest_probability_; }

private:
    void ensure_ensemble_built_(); // This can be deprecated or repurposed
    void ensure_governor_built_();
    
    // Legacy members
    std::unique_ptr<rules::IntegratedRuleEnsemble> ensemble_;
    float buy_lo_{0.60f}, buy_hi_{0.75f}, sell_hi_{0.40f}, sell_lo_{0.25f}, hysteresis_{0.02f};
    int warmup_{252};
    
    // Governor and new state
    std::unique_ptr<IntradayPositionGovernor> governor_;
    double latest_probability_{0.5};

    // State for the new Volatility-Adaptive Strategy
    std::deque<double> vol_return_history_;
    std::deque<double> vol_history_;
    std::deque<double> vwap_price_history_;
    std::deque<double> vwap_volume_history_;
    
    // State for the new Alpha Kernel
    std::deque<double> alpha_return_history_;

    // State for holding period and take-profit
    int last_trade_bar_{-1};
    int last_trade_direction_{0}; // -1 for short, 0 for flat, 1 for long
    double entry_price_{0.0};
    
    // **NEW**: State for Kelly Criterion
    std::deque<double> pnl_history_;
    
    // **NEW**: Kelly Criterion methods
    double calculate_kelly_fraction(double edge_probability, double confidence) const;
    void update_trade_performance(double realized_pnl);
    double get_win_loss_ratio() const;
    double get_win_rate() const;
    
    // **NEW**: Multi-Timeframe Alpha Kernel methods
    double calculate_multi_timeframe_alpha(const std::deque<double>& history) const;
    double calculate_single_alpha_probability(const std::deque<double>& history, int short_window, int long_window) const;
};

} // namespace sentio
