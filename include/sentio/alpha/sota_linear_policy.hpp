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
