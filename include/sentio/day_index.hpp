#pragma once
#include "core.hpp"
#include <vector>
#include <string_view>

namespace sentio {

// From minute bars with RFC3339 timestamps, return indices of first bar per day
inline std::vector<int> day_start_indices(const std::vector<Bar>& bars) {
    std::vector<int> starts;
    starts.reserve(bars.size() / 300 + 2);
    std::string last_day;
    for (int i=0; i<(int)bars.size(); ++i) {
        std::string_view ts(bars[i].ts_utc);
        if (ts.size() < 10) continue;
        std::string cur{ts.substr(0,10)};
        if (i == 0 || cur != last_day) {
            starts.push_back(i);
            last_day = std::move(cur);
        }
    }
    return starts;
}

} // namespace sentio

