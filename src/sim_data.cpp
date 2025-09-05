#include "sentio/sim_data.hpp"
#include <cmath>

namespace sentio {

std::vector<std::pair<std::int64_t, Bar>> generate_minute_series(const SimCfg& cfg) {
  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<double> z(0.0, 1.0);
  std::uniform_real_distribution<double> U(0.0, 1.0);

  std::vector<std::pair<std::int64_t, Bar>> out;
  out.reserve(cfg.minutes);

  double px = cfg.start_price;
  std::int64_t ts = cfg.start_ts_utc;

  for (int i=0;i<cfg.minutes;++i, ts+=60) {
    double u = U(rng);
    double ret = 0.0;

    if (u < cfg.frac_trend) {
      // trending drift + noise
      ret = 0.0002 + (cfg.vol_bps*1e-4) * z(rng);
    } else if (u < cfg.frac_trend + cfg.frac_mr) {
      // mean-reversion around 0 with lighter noise
      ret = -0.0001 + (cfg.vol_bps*0.6e-4) * z(rng);
    } else {
      // jump regime
      ret = (U(rng) < 0.5 ? +1 : -1) * (cfg.vol_bps*6e-4 + std::abs(z(rng))*cfg.vol_bps*3e-4);
    }

    double new_px = std::max(0.01, px * (1.0 + ret));
    double o = px;
    double c = new_px;
    double h = std::max(o, c) * (1.0 + std::abs((cfg.vol_bps*0.3e-4) * z(rng)));
    double l = std::min(o, c) * (1.0 - std::abs((cfg.vol_bps*0.3e-4) * z(rng)));
    // ensure consistency
    h = std::max({h, o, c});
    l = std::min({l, o, c});

    out.push_back({ts, Bar{o,h,l,c}});
    px = new_px;
  }
  return out;
}

} // namespace sentio
