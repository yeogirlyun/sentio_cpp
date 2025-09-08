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
