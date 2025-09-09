# SOTA Portfolio Capital Routing System

**Generated**: 2025-09-09 00:36:20
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: Complete implementation of state-of-the-art linear policy with Merton/Kelly + Gârleanu-Pedersen for professional capital routing, replacing threshold-based decisions with utility-maximizing framework.

**Total Files**: 7

---

## 📋 **TABLE OF CONTENTS**

1. [include/sentio/alpha/sota_linear_policy.hpp](#file-1)
2. [include/sentio/portfolio/alpaca_fee_model.hpp](#file-2)
3. [include/sentio/portfolio/capital_manager.hpp](#file-3)
4. [include/sentio/portfolio/fee_model.hpp](#file-4)
5. [include/sentio/portfolio/portfolio_allocator.hpp](#file-5)
6. [include/sentio/portfolio/tc_slippage_model.hpp](#file-6)
7. [include/sentio/portfolio/utilization_governor.hpp](#file-7)

---

## 📄 **FILE 1 of 7**: include/sentio/alpha/sota_linear_policy.hpp

**File Information**:
- **Path**: `include/sentio/alpha/sota_linear_policy.hpp`

- **Size**: 84 lines
- **Modified**: 2025-09-09 00:34:01

- **Type**: .hpp

```text
#pragma once
#include <algorithm>
#include <cmath>

namespace sentio::alpha {

// Map probability to conditional mean return (per bar).
// Heuristic: if next-bar sign ~ Bernoulli(p_up), then E[r] ≈ k_ret * (2p-1) * sigma,
// where sigma is realized vol for the bar horizon; k_ret calibrates label/return link.
inline double prob_to_mu(double p_up, double sigma, double k_ret=1.0){
  p_up = std::clamp(p_up, 1e-6, 1.0-1e-6);
  return k_ret * (2.0*p_up - 1.0) * sigma;
}

struct SotaPolicyCfg {
  double gamma     = 4.0;   // risk aversion (higher => smaller positions)
  double k_ret     = 1.0;   // maps prob edge to mean return units
  double lam_tc    = 5e-4;  // turnover penalty (per unit weight change)
  double min_edge  = 0.0;   // optional: deadband in mu minus costs
  double max_abs_w = 0.50;  // per-name cap on weight (leverage)
};

// One-step SOTA linear policy with costs (1-asset version).
// Inputs: p_up in [0,1], sigma (per-bar vol), prev weight w_prev, est. one-shot cost in bps.
inline double sota_linear_weight(double p_up, double sigma, double w_prev, double cost_bps, const SotaPolicyCfg& cfg){
  sigma = std::max(1e-6, sigma);
  // 1) Aim (Merton/Kelly): w* = mu / (gamma * sigma^2)
  const double mu = prob_to_mu(p_up, sigma, cfg.k_ret);
  double w_aim = mu / (cfg.gamma * sigma * sigma);

  // 2) Cost-aware partial adjustment (Gârleanu–Pedersen style)
  // Solve: min_w  (gamma*sigma^2/2)*(w - w_aim)^2 + lam_tc*|w - w_prev|
  // Closed-form with L1 gives a soft-threshold around w_aim; approximate with shrinkage:
  const double k = cfg.gamma * sigma * sigma;
  double w_free = (k*w_aim + cfg.lam_tc*w_prev) / (k + cfg.lam_tc); // Ridge-like blend
  // Apply a small deadband if expected edge can't beat costs
  const double edge_bps = 1e4 * std::abs(mu); // rough edge proxy (bps)
  if (edge_bps < cost_bps + cfg.min_edge) {
    // shrink toward previous to avoid churn
    w_free = 0.5*w_prev + 0.5*w_free;
  }

  // 3) Cap leverage
  w_free = std::clamp(w_free, -cfg.max_abs_w, cfg.max_abs_w);
  return w_free;
}

// Decision helper (hold/long/short) — useful for audits:
enum class Dir { HOLD=0, LONG=+1, SHORT=-1 };
inline Dir direction_from_weight(double w, double tol=1e-3){
  if (w >  tol) return Dir::LONG;
  if (w < -tol) return Dir::SHORT;
  return Dir::HOLD;
}

// Multi-asset version: returns target weights vector given p_up vector and covariance
// This implements the multi-asset Merton rule with partial adjustment
inline std::vector<double> sota_multi_asset_weights(
    const std::vector<double>& p_up,
    const std::vector<double>& sigma,
    const std::vector<double>& w_prev,
    const std::vector<double>& cost_bps,
    const SotaPolicyCfg& cfg) {
    
    const int N = p_up.size();
    std::vector<double> w_target(N, 0.0);
    
    for (int i = 0; i < N; i++) {
        w_target[i] = sota_linear_weight(p_up[i], sigma[i], w_prev[i], cost_bps[i], cfg);
    }
    
    // Apply gross constraint (portfolio-level leverage cap)
    double gross = 0.0;
    for (double w : w_target) gross += std::abs(w);
    
    if (gross > cfg.max_abs_w * N) { // rough gross cap
        double scale = (cfg.max_abs_w * N) / gross;
        for (double& w : w_target) w *= scale;
    }
    
    return w_target;
}

} // namespace sentio::alpha

```

## 📄 **FILE 2 of 7**: include/sentio/portfolio/alpaca_fee_model.hpp

**File Information**:
- **Path**: `include/sentio/portfolio/alpaca_fee_model.hpp`

- **Size**: 30 lines
- **Modified**: 2025-09-08 21:40:10

- **Type**: .hpp

```text
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

```

## 📄 **FILE 3 of 7**: include/sentio/portfolio/capital_manager.hpp

**File Information**:
- **Path**: `include/sentio/portfolio/capital_manager.hpp`

- **Size**: 115 lines
- **Modified**: 2025-09-09 00:35:20

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>
#include "portfolio_allocator.hpp"
#include "utilization_governor.hpp"

namespace sentio {

// Forward declarations to avoid circular includes
class PositionSizer;
class BinaryRouter;
struct MarketSnapshot;

struct InstrumentState {
  // live state snapshot
  int64_t ts;
  double  price;
  double  vol_1d;
  double  adv_notional;
  double  spread_bp;
  long    shares;       // current position
  double  w_prev;       // last target weight
  bool    tradable{true};
};

struct RouterInputs {
  // Per-instrument p_up from Strategy (TFA/Kochi/RuleEnsemble)
  double p_up;
};

struct CapitalManagerConfig {
  AllocConfig alloc;
  // Map utilization governor into Router and Sizer:
  UtilGovConfig util;
};

struct CapitalDecision {
  std::vector<double> target_weights; // per instrument
  std::vector<long long> target_shares;
  std::vector<double> edge_bp;        // per-instrument edge in bps
  double gross{0.0};
  double total_cost_bp{0.0};          // estimated total cost
};

class CapitalManager {
public:
  CapitalManager(const CapitalManagerConfig& cfg,
                 PortfolioAllocator alloc)
   : cfg_(cfg), alloc_(std::move(alloc)) {}

  CapitalDecision decide(double equity,
                         const std::vector<InstrumentState>& S,
                         const std::vector<RouterInputs>& R,
                         UtilGovState& ug_state)
  {
    const int N = (int)S.size();
    std::vector<InstrumentInputs> X; X.reserve(N);
    for (int i=0;i<N;i++){
      X.push_back(InstrumentInputs{
        /*p_up   */ R[i].p_up,
        /*price  */ S[i].price,
        /*vol_1d */ S[i].vol_1d,
        /*spread */ S[i].spread_bp,
        /*adv    */ S[i].adv_notional,
        /*w_prev */ S[i].w_prev,
        /*pos_w  */ (S[i].price>0? (S[i].shares*S[i].price)/std::max(1e-9, equity) : 0.0),
        /*trad   */ S[i].tradable
      });
    }

    // Portfolio-level allocation with SOTA linear policy
    auto out = alloc_.allocate(X, cfg_.alloc);
    
    // Apply utilization governor: upscale/downscale weights via expo_shift
    const double expo_mul = std::clamp(1.0 + ug_state.expo_shift, 0.5, 1.5);
    for (double& w : out.w_target) w *= expo_mul;

    // Compute shares via weight-based sizing
    std::vector<long long> tgt_sh(N, 0);
    for (int i=0;i<N;i++){
      if (S[i].price > 0 && std::isfinite(out.w_target[i])) {
        double desired_notional = out.w_target[i] * equity;
        tgt_sh[i] = (long long)std::floor(desired_notional / S[i].price);
        // Apply round lot if needed (assume 1 for now)
        // Apply min notional filter (assume $50 from previous fix)
        if (std::abs(tgt_sh[i] * S[i].price) < 50.0) {
          tgt_sh[i] = 0;
        }
      }
    }

    // Report gross and cost estimates
    double gross=0.0, total_cost=0.0;
    for (int i=0;i<N;i++) {
      gross += std::abs(out.w_target[i]);
      // Estimate turnover cost
      double w_change = std::abs(out.w_target[i] - S[i].w_prev);
      total_cost += w_change * 5.0; // rough 5bp per weight change
    }

    return CapitalDecision{ 
      std::move(out.w_target), 
      std::move(tgt_sh), 
      std::move(out.edge_bp),
      gross,
      total_cost
    };
  }

private:
  CapitalManagerConfig cfg_;
  PortfolioAllocator alloc_;
};

} // namespace sentio

```

## 📄 **FILE 4 of 7**: include/sentio/portfolio/fee_model.hpp

**File Information**:
- **Path**: `include/sentio/portfolio/fee_model.hpp`

- **Size**: 21 lines
- **Modified**: 2025-09-08 21:40:10

- **Type**: .hpp

```text
#pragma once
#include <cstdint>

namespace sentio {

struct TradeCtx {
  double price;       // mid/exec price
  double notional;    // |shares| * price
  long   shares;      // signed
  bool   is_short;    // for borrow fees if modeled
};

class IFeeModel {
public:
  virtual ~IFeeModel() = default;
  virtual double commission(const TradeCtx& t) const = 0;  // $ cost
  virtual double exchange_fees(const TradeCtx& t) const { return 0.0; }
  virtual double borrow_fee_daily_bp(double notional_short) const { return 0.0; }
};

} // namespace sentio

```

## 📄 **FILE 5 of 7**: include/sentio/portfolio/portfolio_allocator.hpp

**File Information**:
- **Path**: `include/sentio/portfolio/portfolio_allocator.hpp`

- **Size**: 115 lines
- **Modified**: 2025-09-09 00:34:55

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "tc_slippage_model.hpp"
#include "fee_model.hpp"
#include "sentio/alpha/sota_linear_policy.hpp"

namespace sentio {

struct InstrumentInputs {
  // Inputs per instrument i
  double p_up;            // probability next bar up [0,1]
  double price;           // last price
  double vol_1d;          // daily vol (sigma)
  double spread_bp;       // bid-ask spread bps
  double adv_notional;    // ADV in $
  double w_prev;          // previous target weight
  double pos_weight;      // live position weight (for turnover)
  bool   tradable{true};  // liquidity/halts
};

struct AllocConfig {
  // SOTA linear policy config
  sentio::alpha::SotaPolicyCfg sota;
  
  // Portfolio-level constraints
  double max_gross       = 1.50;  // portfolio gross cap (150%)
  bool   long_only       = false; // allow shorts?
  
  // Legacy parameters (for backward compatibility)
  double risk_aversion   = 5.0;   // maps to sota.gamma
  double tc_lambda       = 2.0;   // maps to sota.lam_tc
  double max_weight_abs  = 0.20;  // maps to sota.max_abs_w
  double min_edge_bp     = 2.0;   // maps to sota.min_edge
};

struct AllocOutput {
  std::vector<double> w_target;     // target weights per instrument
  std::vector<double> edge_bp;      // per-instrument modeled edge bps
  double gross{0.0};
  double net_gross{0.0};
};

class PortfolioAllocator {
public:
  PortfolioAllocator(TCModel tc, const IFeeModel& fee) : tc_(tc), fee_(fee) {}

  AllocOutput allocate(const std::vector<InstrumentInputs>& X, const AllocConfig& cfg) const {
    const int N = (int)X.size();
    AllocOutput out; out.w_target.assign(N, 0.0); out.edge_bp.assign(N, 0.0);

    // Sync legacy config to SOTA policy config
    auto sota_cfg = cfg.sota;
    sota_cfg.gamma = cfg.risk_aversion;
    sota_cfg.lam_tc = cfg.tc_lambda * 1e-4; // convert to weight units
    sota_cfg.max_abs_w = cfg.max_weight_abs;
    sota_cfg.min_edge = cfg.min_edge_bp;

    // Prepare inputs for SOTA linear policy
    std::vector<double> p_up(N), sigma(N), w_prev(N), cost_bps(N);
    
    for (int i = 0; i < N; i++) {
      const auto& x = X[i];
      if (!x.tradable || !std::isfinite(x.p_up)) {
        p_up[i] = 0.5; // neutral if not tradable
        sigma[i] = 0.01; // small vol
        w_prev[i] = x.w_prev;
        cost_bps[i] = 1000.0; // high cost to discourage
        continue;
      }

      p_up[i] = x.p_up;
      sigma[i] = std::max(1e-6, x.vol_1d / std::sqrt(252.0)); // daily to per-bar vol
      w_prev[i] = x.w_prev;
      
      // Estimate total cost (slippage + fees)
      double notional_est = x.price * 1000.0; // estimate for small trade
      double slippage = tc_.slippage_bp(notional_est, {x.adv_notional, x.spread_bp, x.vol_1d});
      TradeCtx trade_ctx{x.price, notional_est, 1000, false};
      double fees_bp = 1e4 * (fee_.commission(trade_ctx) + fee_.exchange_fees(trade_ctx)) / notional_est;
      cost_bps[i] = slippage + fees_bp;
      
      // Store edge for reporting
      double mu = sentio::alpha::prob_to_mu(x.p_up, sigma[i], sota_cfg.k_ret);
      out.edge_bp[i] = 1e4 * mu - cost_bps[i]; // net edge in bps
    }

    // **SOTA LINEAR POLICY**: Merton/Kelly + Gârleanu-Pedersen
    out.w_target = sentio::alpha::sota_multi_asset_weights(p_up, sigma, w_prev, cost_bps, sota_cfg);
    
    // Apply long-only constraint if needed
    if (cfg.long_only) {
      for (double& w : out.w_target) w = std::max(0.0, w);
    }

    // Apply portfolio gross constraint
    double gross = 0.0; for (double w : out.w_target) gross += std::abs(w);
    if (gross > cfg.max_gross && gross > 0) {
      double scale = cfg.max_gross / gross;
      for (double& w : out.w_target) w *= scale;
      gross = cfg.max_gross;
    }

    out.gross = gross;
    out.net_gross = gross; // could subtract forecast TC here if desired
    return out;
  }

private:
  TCModel tc_;
  const IFeeModel& fee_;
};

} // namespace sentio

```

## 📄 **FILE 6 of 7**: include/sentio/portfolio/tc_slippage_model.hpp

**File Information**:
- **Path**: `include/sentio/portfolio/tc_slippage_model.hpp`

- **Size**: 27 lines
- **Modified**: 2025-09-08 21:40:10

- **Type**: .hpp

```text
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

```

## 📄 **FILE 7 of 7**: include/sentio/portfolio/utilization_governor.hpp

**File Information**:
- **Path**: `include/sentio/portfolio/utilization_governor.hpp`

- **Size**: 58 lines
- **Modified**: 2025-09-08 21:40:10

- **Type**: .hpp

```text
#pragma once
#include <algorithm>

namespace sentio {

struct UtilGovConfig {
  double target_gross = 0.60;  // target gross exposure (sum |weights|), e.g. 60%
  double kp_expo      = 0.05;  // proportional gain for exposure gap
  double target_tpd   = 40.0;  // trades per day target (optional)
  double kp_trades    = 0.02;  // proportional gain for trade gap
  float  max_shift    = 0.10f; // max threshold nudge
  double max_vol_adj  = 0.50;  // ±50% of vol target adjustment
};

struct UtilGovState {
  double expo_shift{0.0};      // maps to sizer's vol target multiplier
  float  buy_shift{0.0f};      // router threshold shift
  float  sell_shift{0.0f};
  double integ_expo{0.0};      // optional: add Ki later if needed
  double integ_trades{0.0};
};

class UtilizationGovernor {
public:
  explicit UtilizationGovernor(const UtilGovConfig& c) : cfg_(c) {}

  void daily_update(double realized_gross, int trades_today, UtilGovState& st){
    // Exposure control
    double e_err = cfg_.target_gross - realized_gross;
    st.expo_shift = std::clamp(st.expo_shift + cfg_.kp_expo * e_err,
                               -cfg_.max_vol_adj, cfg_.max_vol_adj);

    // Trades/day control → route thresholds
    double t_err = cfg_.target_tpd - trades_today;
    float delta  = (float)(cfg_.kp_trades * t_err);
    st.buy_shift  = clamp_shift(st.buy_shift  - 0.5f*delta);
    st.sell_shift = clamp_shift(st.sell_shift - 0.5f*delta);
  }

  void get_nudges(struct RouterNudges& nudges, const UtilGovState& st) const {
    nudges.buy_shift = st.buy_shift;
    nudges.sell_shift = st.sell_shift;
  }

private:
  float clamp_shift(float x) const {
    return std::clamp(x, -cfg_.max_shift, cfg_.max_shift);
  }
  UtilGovConfig cfg_;
};

// Forward declare RouterNudges for get_nudges method
struct RouterNudges {
  float buy_shift  = 0.f;
  float sell_shift = 0.f;
};

} // namespace sentio

```

