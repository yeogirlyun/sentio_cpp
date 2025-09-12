#pragma once
#include <cmath>

namespace sentio {

// Calibrated so that p(30)=0.8, p(50)=0.5, p(70)=0.2
// p(RSI) = 1 / (1 + exp( k * (RSI - 50) / 10 ) )
// Solve: 0.8 = 1/(1+exp(k*(30-50)/10)) -> k = ln(2) ≈ 0.693147
inline double rsi_to_prob(double rsi) {
    constexpr double k = 0.6931471805599453; // ln(2)
    double x = (rsi - 50.0) / 10.0;
    double e = std::exp(k * x);
    return 1.0 / (1.0 + e);
}

// Optionally expose a tunable steepness (k = ln(2)*alpha).
inline double rsi_to_prob_tuned(double rsi, double alpha) {
    double k = 0.6931471805599453 * alpha;
    double x = (rsi - 50.0) / 10.0;
    return 1.0 / (1.0 + std::exp(k * x));
}

} // namespace sentio
