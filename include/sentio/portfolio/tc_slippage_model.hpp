#pragma once
#include <algorithm>
#include <cmath>

namespace sentio {

struct LiquidityStats {
  double adv_notional;  // average daily $ volume
  double spread_bp;     // bid-ask spread in bps
  double vol_1d;        // 1-day realized vol (for impact)
};

struct TCModel {
  // Simple slippage model: half-spread + impact
  // slippage = 0.5*spread + k * (trade_notional / ADV) ^ alpha
  double k_impact = 25.0;  // bps at 100% ADV if alpha=0.5
  double alpha    = 0.5;   // square-root impact

  double slippage_bp(double trade_notional, const LiquidityStats& L) const {
    double half_spread = 0.5 * L.spread_bp;
    double adv_frac = (L.adv_notional > 0 ? std::min(1.0, trade_notional / L.adv_notional) : 1.0);
    double impact = k_impact * std::pow(adv_frac, alpha);
    return half_spread + impact; // total bps (one-way)
  }
};

} // namespace sentio
