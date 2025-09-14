#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace sentio::metrics {

inline double safe_log1p(double x) {
    // guard tiny negatives from rounding
    if (x <= -1.0) {
        throw std::domain_error("Return <= -100% encountered in log1p.");
    }
    return std::log1p(x);
}

struct MprParams {
    int trading_days_per_month = 21; // equities default
};

inline double compute_mpr_from_daily_returns(
    const std::vector<double>& daily_simple_returns,
    const MprParams& params = {}
) {
    if (daily_simple_returns.empty()) return 0.0;

    long double sum_log = 0.0L;
    for (double r : daily_simple_returns) {
        sum_log += static_cast<long double>(safe_log1p(r));
    }
    const long double mean_log = sum_log / static_cast<long double>(daily_simple_returns.size());
    const long double geo_daily = std::expm1(mean_log); // e^{mean_log} - 1
    const long double mpr = std::pow(1.0L + geo_daily, params.trading_days_per_month) - 1.0L;
    return static_cast<double>(mpr);
}

// Convenience: from equity curve (close-to-close)
inline double compute_mpr_from_equity(
    const std::vector<double>& daily_equity, // length N >= 2
    const MprParams& params = {}
) {
    if (daily_equity.size() < 2) return 0.0;
    std::vector<double> rets;
    rets.reserve(daily_equity.size() - 1);
    for (size_t i = 1; i < daily_equity.size(); ++i) {
        double prev = daily_equity[i-1], cur = daily_equity[i];
        if (!(std::isfinite(prev) && std::isfinite(cur)) || prev <= 0.0) {
            throw std::runtime_error("Non-finite or non-positive equity in compute_mpr_from_equity");
        }
        rets.push_back((cur / prev) - 1.0);
    }
    return compute_mpr_from_daily_returns(rets, params);
}

} // namespace sentio::metrics
