#include "sentio/virtual_market.hpp"
#include "sentio/leverage_pricing.hpp"
#include <cmath>

namespace sentio {

std::vector<Bar> generate_theoretical_leverage_series(const std::vector<Bar>& base_series,
                                                      double leverage_factor) {
    if (base_series.empty()) {
        return {};
    }

    // Prefer symbol-specific pricing when factor matches known ETFs
    std::string leverage_symbol;
    if (std::fabs(leverage_factor - 3.0) < 1e-6) {
        leverage_symbol = "TQQQ";   // 3x long
    } else if (std::fabs(leverage_factor + 3.0) < 1e-6) {
        leverage_symbol = "SQQQ";   // 3x short
    } else if (std::fabs(leverage_factor + 1.0) < 1e-6) {
        leverage_symbol = "PSQ";    // 1x inverse
    }

    TheoreticalLeveragePricer pricer;
    if (!leverage_symbol.empty()) {
        return pricer.generate_theoretical_series(leverage_symbol, base_series);
    }

    // Generic fallback: scale base returns by leverage_factor
    std::vector<Bar> out;
    out.reserve(base_series.size());

    Bar prev = base_series.front();
    Bar first = prev;
    out.push_back(first);

    for (size_t i = 1; i < base_series.size(); ++i) {
        const Bar& base_prev = base_series[i - 1];
        const Bar& base_cur = base_series[i];

        Bar lev = prev;
        lev.ts_utc = base_cur.ts_utc;
        lev.ts_utc_epoch = base_cur.ts_utc_epoch;

        double denom = (base_prev.close != 0.0) ? base_prev.close : 1e-8;
        double base_ret = (base_cur.close - base_prev.close) / denom;
        double lev_ret = leverage_factor * base_ret;

        double new_price = std::max(0.01, prev.close * (1.0 + lev_ret));
        lev.open = new_price;
        lev.high = new_price;
        lev.low = new_price;
        lev.close = new_price;
        lev.volume = base_cur.volume;

        out.push_back(lev);
        prev = lev;
    }

    return out;
}

} // namespace sentio


