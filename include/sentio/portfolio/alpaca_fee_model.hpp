#pragma once
#include "fee_model.hpp"
#include <algorithm>

namespace sentio {

// Simplified Alpaca-like: $0 commission equities, SEC/TAF pass-through for sells
// Tweak the constants to match your latest schedule if needed.
class AlpacaEquityFeeModel : public IFeeModel {
public:
  // Per-share TAF for sells, SEC fee as bps of notional on sells (approx)
  double taf_per_share = 0.000119;    // $0.000119/share
  double sec_bps_sell  = 0.0000229;   // 0.00229% of notional on sells
  double min_fee       = 0.0;         // $0 min for commissionless
  bool   include_sec   = true;

  double commission(const TradeCtx& t) const override {
    // commission-free
    return min_fee;
  }

  double exchange_fees(const TradeCtx& t) const override {
    if (t.shares > 0) return 0.0; // buy: no SEC/TAF
    double taf = std::abs(t.shares) * taf_per_share;
    double sec = include_sec ? std::max(0.0, std::abs(t.notional) * sec_bps_sell) : 0.0;
    return taf + sec;
  }
};

} // namespace sentio
