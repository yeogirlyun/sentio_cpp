#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "of_index.hpp"

namespace sentio {

inline void precompute_mid_imb(const std::vector<Tick>& ticks,
                               std::vector<double>& mid,
                               std::vector<double>& imb)
{
    const int M = (int)ticks.size();
    mid.resize(M);
    imb.resize(M);
    for (int k=0; k<M; ++k) {
        double m = (ticks[k].bid_px + ticks[k].ask_px) * 0.5;
        double a = std::max(0.0, ticks[k].ask_sz);
        double b = std::max(0.0, ticks[k].bid_sz);
        double d = a + b;
        mid[k] = m;
        imb[k] = (d > 0.0) ? (a / d) : 0.5;   // neutral if zero depth
    }
}

} // namespace sentio