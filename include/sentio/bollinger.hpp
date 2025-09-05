#pragma once
#include "rolling_stats.hpp"

namespace sentio {

struct Bollinger {
  RollingMeanVar mv;
  double k;
  double eps; // volatility floor to avoid zero-width bands

  Bollinger(int w, double k_=2.0, double eps_=1e-9) : mv(w), k(k_), eps(eps_) {}

  inline void step(double close, double& mid, double& lo, double& hi, double& sd_out){
    auto [m, var] = mv.push(close);
    double sd = std::sqrt(std::max(var, 0.0));
    if (sd < eps) sd = eps;         // <- floor
    mid = m; lo = m - k*sd; hi = m + k*sd;
    sd_out = sd;
  }
};

} // namespace sentio