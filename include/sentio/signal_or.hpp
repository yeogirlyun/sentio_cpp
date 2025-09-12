#pragma once
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace sentio {

// Each rule emits a probability in [0,1] (>0.5 long bias, <0.5 short bias)
// and a confidence in [0,1] (0 = ignore, 1 = strong).
struct RuleOut {
  double p01;
  double conf01;
};

// Tuning knobs for OR behavior
struct OrCfg {
  double min_conf = 0.05;     // ignore components below this confidence
  double aggression = 0.85;   // 0..1, closer to 1 → stronger OR push
  double floor_eps = 1e-6;    // numerical safety
  double neutral_band = 0.015;// tiny band snapped to exact neutral
  double conflict_soften = 0.35; // 0..1 reduce both sides when both high
  size_t min_active = 1;      // require at least this many active rules
};

// Helper: clamp to [0,1]
inline double clamp01(double x) {
  if (!std::isfinite(x)) return 0.5;
  return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

// Noisy-OR on "evidence of long" and "evidence of short", combined safely.
inline double mix_signal_or(const std::vector<RuleOut>& rules, const OrCfg& cfg) {
  double prod_no_long  = 1.0;
  double prod_no_short = 1.0;
  size_t active = 0;

  for (auto r : rules) {
    double p = clamp01(r.p01);
    double c = clamp01(r.conf01);
    if (c < cfg.min_conf) continue;
    // Long evidence in [0,1]: map p ∈ [0.5,1] → [0,1]
    double e_long  = (p <= 0.5) ? 0.0 : (p - 0.5) * 2.0;
    double e_short = (p >= 0.5) ? 0.0 : (0.5 - p) * 2.0;
    // Confidence-weighted evidence
    e_long  = std::pow(e_long,  1.0 - cfg.aggression) * c;
    e_short = std::pow(e_short, 1.0 - cfg.aggression) * c;

    prod_no_long  *= (1.0 - std::max(cfg.floor_eps, e_long));
    prod_no_short *= (1.0 - std::max(cfg.floor_eps, e_short));
    active++;
  }

  if (active < cfg.min_active) return 0.5; // neutral if literally nothing active

  // Noisy-OR results (probability that at least one rule supports the side)
  double p_long  = 1.0 - prod_no_long;
  double p_short = 1.0 - prod_no_short;

  // If both sides are high (conflict), soften both so neutral can occur
  // only when truly balanced and confident on both sides.
  if (p_long > 0.0 && p_short > 0.0) {
    p_long  *= (1.0 - cfg.conflict_soften);
    p_short *= (1.0 - cfg.conflict_soften);
  }

  // Convert side probabilities back to a single p01 in [0,1]
  // Intuition: p = 0.5 + (p_long - p_short)/2, clipped and denoised.
  double p01 = 0.5 + 0.5 * (p_long - p_short);
  p01 = clamp01(p01);

  // Debounce micro-noise to exact neutral for stability near 0.5
  if (std::fabs(p01 - 0.5) < cfg.neutral_band) p01 = 0.5;

  return p01;
}

} // namespace sentio
