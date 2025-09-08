#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace sentio {

// Advanced Sizer Configuration with Risk Controls
struct SizerCfg {
  bool fractional_allowed = true;
  double min_notional = 1.0;
  double max_leverage = 2.0;
  double max_position_pct = 0.25;
  double volatility_target = 0.15;
  bool allow_negative_cash = false;
  int vol_lookback_days = 20;
  double cash_reserve_pct = 0.05;
};

// Advanced Sizer Class with Multiple Constraints
class AdvancedSizer {
public:
  double calculate_volatility(const std::vector<Bar>& price_history, int lookback) const {
    if (price_history.size() < static_cast<size_t>(lookback)) return 0.05; // Default vol

    std::vector<double> returns;
    returns.reserve(lookback - 1);
    for (size_t i = price_history.size() - lookback + 1; i < price_history.size(); ++i) {
      double prev_close = price_history[i-1].close;
      if (prev_close > 0) {
        returns.push_back(price_history[i].close / prev_close - 1.0);
      }
    }
    
    if (returns.size() < 2) return 0.05;
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - mean) * (ret - mean);
    }
    variance /= returns.size();
    return std::sqrt(variance) * std::sqrt(252.0); // Annualized
  }

  // **MODIFIED**: Signature and logic updated for the ID-based, high-performance architecture.
  double calculate_target_quantity(const Portfolio& portfolio,
                                   const SymbolTable& ST,
                                   const std::vector<double>& last_prices,
                                   const std::string& instrument,
                                   double target_weight,
                                                                       [[maybe_unused]] const std::vector<Bar>& price_history,
                                   const SizerCfg& cfg) const {
    
    const double equity = equity_mark_to_market(portfolio, last_prices);
    int instrument_id = ST.get_id(instrument);

    if (equity <= 0 || instrument_id == -1 || last_prices[instrument_id] <= 0) {
        return 0.0;
    }
    double instrument_price = last_prices[instrument_id];

    // --- Calculate size based on multiple constraints ---
    double desired_notional = equity * std::abs(target_weight);

    // 1. Max Position Size Constraint
    desired_notional = std::min(desired_notional, equity * cfg.max_position_pct);

    // 2. Leverage Constraint
    double current_exposure = 0.0;
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        current_exposure += std::abs(portfolio.positions[sid].qty * last_prices[sid]);
    }
    double available_leverage_notional = (equity * cfg.max_leverage) - current_exposure;
    desired_notional = std::min(desired_notional, std::max(0.0, available_leverage_notional));

    // 4. Cash Constraint
    if (!cfg.allow_negative_cash) {
      double usable_cash = portfolio.cash * (1.0 - cfg.cash_reserve_pct);
      desired_notional = std::min(desired_notional, std::max(0.0, usable_cash));
    }
    
    if (desired_notional < cfg.min_notional) return 0.0;
    
    double qty = desired_notional / instrument_price;
    double final_qty = cfg.fractional_allowed ? qty : std::floor(qty);
    
    // Return with the correct sign (long/short)
    return (target_weight > 0) ? final_qty : -final_qty;
  }

  // **NEW**: Weight-to-shares helper for portfolio allocator integration
  long long target_shares_from_weight(double target_weight, double equity, double price, const SizerCfg& cfg) const {
    if (price <= 0 || equity <= 0) return 0;
    
    // weight = position_notional / equity ⇒ shares = weight * equity / price
    double desired_notional = target_weight * equity;
    long long shares = (long long)std::floor(std::abs(desired_notional) / price);
    
    // Apply min notional filter
    if (shares * price < cfg.min_notional) {
      shares = 0;
    }
    
    // Apply sign
    if (target_weight < 0) shares = -shares;
    
    return shares;
  }
};

} // namespace sentio