#pragma once
#include "base_strategy.hpp"
#include "rsi_prob.hpp"
#include "sizer.hpp"
#include <unordered_map>
#include <string>
#include <cmath>

namespace sentio {

class RSIStrategy final : public BaseStrategy {
public:
    RSIStrategy()
    : BaseStrategy("RSI_PROB"),
      rsi_period_(14),
      epsilon_(0.05),             // |w| < eps -> Neutral
      weight_clip_(1.0),
      alpha_(1.0),                // k = ln(2)*alpha ; alpha>1 => steeper
      long_symbol_("QQQ"),
      short_symbol_("SQQQ")
    {}

    ParameterMap get_default_params() const override {
        return {
            {"rsi_period", 14},
            {"epsilon", 0.05},
            {"weight_clip", 1.0},
            {"alpha", 1.0}
        };
    }

    ParameterSpace get_param_space() const override {
        ParameterSpace space;
        space["rsi_period"] = {ParamType::INT, 7, 21, 14};
        space["epsilon"] = {ParamType::FLOAT, 0.01, 0.2, 0.05};
        space["weight_clip"] = {ParamType::FLOAT, 0.5, 2.0, 1.0};
        space["alpha"] = {ParamType::FLOAT, 0.5, 3.0, 1.0};
        return space;
    }

    void apply_params() override {
        auto get = [&](const char* k, double d){
            auto it=params_.find(k); return (it==params_.end() || !std::isfinite(it->second))? d : it->second;
        };
        rsi_period_  = std::max(2, (int)std::llround(get("rsi_period", 14)));
        epsilon_     = std::max(0.0, std::min(0.5, get("epsilon", 0.05)));
        weight_clip_ = std::max(0.1, std::min(2.0, get("weight_clip", 1.0)));
        alpha_       = std::max(0.1, std::min(5.0, get("alpha", 1.0)));
    }

    double calculate_probability(const std::vector<Bar>& bars, int current_index) override {
        if (bars.empty() || current_index < 0 || current_index >= static_cast<int>(bars.size())) {
            diag_.drop(DropReason::MIN_BARS);
            return 0.5; // Neutral
        }
        
        // Need at least rsi_period_ bars to calculate RSI
        if (current_index < rsi_period_) {
            diag_.drop(DropReason::MIN_BARS);
            return 0.5; // Neutral during warmup
        }
        
        // Calculate RSI using the previous rsi_period_ bars
        double rsi = calculate_rsi_from_bars(bars, current_index - rsi_period_, current_index);
        
        // Apply sigmoid transformation
        double probability = rsi_to_prob_tuned(rsi, alpha_);
        
        // Update signal diagnostics
        if (probability != 0.5) {
            diag_.emitted++;
        }
        
        return probability;
    }

    std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, 
        int current_index,
        const std::string& /* base_symbol */,
        const std::string& /* bull3x_symbol */,
        const std::string& /* bear3x_symbol */,
        const std::string& /* bear1x_symbol */) override {
        
        std::vector<AllocationDecision> decisions;
        
        if (bars.empty() || current_index < 0 || current_index >= static_cast<int>(bars.size())) {
            return decisions;
        }
        
        double probability = calculate_probability(bars, current_index);
        double weight = 2.0 * (probability - 0.5); // Convert to [-1, 1]
        
        if (std::abs(weight) < epsilon_) {
            // Neutral signal
            AllocationDecision decision;
            decision.instrument = long_symbol_;
            decision.target_weight = 0.0;
            decision.confidence = 0.0;
            decision.reason = "RSI_NEUTRAL";
            decisions.push_back(decision);
        } else {
            // Clip weight
            if (weight > weight_clip_) weight = weight_clip_;
            if (weight < -weight_clip_) weight = -weight_clip_;
            
            AllocationDecision decision;
            if (weight > 0.0) {
                decision.instrument = long_symbol_;
                decision.target_weight = weight;
                decision.confidence = weight;
                decision.reason = "RSI_BULLISH";
            } else {
                decision.instrument = short_symbol_;
                decision.target_weight = weight;
                decision.confidence = -weight;
                decision.reason = "RSI_BEARISH";
            }
            decisions.push_back(decision);
        }
        
        return decisions;
    }

    RouterCfg get_router_config() const override {
        RouterCfg cfg;
        cfg.base_symbol = long_symbol_;
        cfg.bull3x = "TQQQ";
        cfg.bear3x = short_symbol_;
        cfg.bear1x = "PSQ";
        cfg.max_position_pct = 1.0;
        cfg.min_signal_strength = epsilon_;
        cfg.signal_multiplier = 1.0;
        return cfg;
    }

    SizerCfg get_sizer_config() const override {
        SizerCfg cfg;
        cfg.max_position_pct = 1.0;
        cfg.max_leverage = 3.0;
        cfg.volatility_target = 0.15;
        cfg.vol_lookback_days = 20;
        cfg.cash_reserve_pct = 0.05;
        cfg.fractional_allowed = true;
        cfg.min_notional = 1.0;
        return cfg;
    }

private:
    
    double calculate_rsi_from_bars(const std::vector<Bar>& bars, int start_idx, int end_idx) {
        if (start_idx < 0 || end_idx >= static_cast<int>(bars.size()) || start_idx >= end_idx) {
            return 50.0; // Neutral RSI
        }
        
        // Calculate price changes
        std::vector<double> gains, losses;
        for (int i = start_idx + 1; i <= end_idx; ++i) {
            double change = bars[i].close - bars[i-1].close;
            gains.push_back(change > 0 ? change : 0.0);
            losses.push_back(change < 0 ? -change : 0.0);
        }
        
        // Calculate initial averages (simple average for first period)
        double avg_gain = 0.0, avg_loss = 0.0;
        for (size_t i = 0; i < gains.size(); ++i) {
            avg_gain += gains[i];
            avg_loss += losses[i];
        }
        avg_gain /= gains.size();
        avg_loss /= losses.size();
        
        // Apply Wilder's smoothing for remaining periods
        for (size_t i = 0; i < gains.size(); ++i) {
            avg_gain = (avg_gain * (rsi_period_ - 1) + gains[i]) / rsi_period_;
            avg_loss = (avg_loss * (rsi_period_ - 1) + losses[i]) / rsi_period_;
        }
        
        // Calculate RSI
        if (avg_loss == 0.0) {
            return 100.0; // All gains, no losses
        }
        
        double rs = avg_gain / avg_loss;
        return 100.0 - (100.0 / (1.0 + rs));
    }

    // Params
    int    rsi_period_;
    double epsilon_;
    double weight_clip_;
    double alpha_;
    std::string long_symbol_;
    std::string short_symbol_;
};

} // namespace sentio
