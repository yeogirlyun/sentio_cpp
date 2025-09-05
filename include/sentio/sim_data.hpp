#pragma once
#include <vector>
#include <cstdint>
#include <random>
#include "sanity.hpp"

namespace sentio {

// Generates a synthetic minute-bar series with regimes (trend, mean-revert, jump).
struct SimCfg {
  std::int64_t start_ts_utc{1'600'000'000};
  int minutes{500};
  double start_price{100.0};
  unsigned seed{42};
  // regime fractions (sum <= 1.0)
  double frac_trend{0.5};
  double frac_mr{0.4};
  double frac_jump{0.1};
  double vol_bps{15.0};    // base noise per min (bps)
};

std::vector<std::pair<std::int64_t, Bar>> generate_minute_series(const SimCfg& cfg);

} // namespace sentio
