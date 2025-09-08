#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include "core.hpp"
#include "orderflow_types.hpp"

namespace sentio {

struct BarTickSpan { 
    int start=-1, end=-1; // [start,end) ticks for this bar
};

inline std::vector<BarTickSpan> build_tick_spans(const std::vector<Bar>& bars,
                                                 const std::vector<Tick>& ticks)
{
    const int N = (int)bars.size();
    const int M = (int)ticks.size();
    std::vector<BarTickSpan> span(N);

    int i = 0, k = 0;
    int cur_start = 0;

    // assume bars have strictly increasing ts; ticks nondecreasing
    for (; i < N; ++i) {
        const int64_t ts = bars[i].ts_utc_epoch;
        // advance k until tick.ts > ts
        while (k < M && ticks[k].ts_utc_epoch <= ts) ++k;
        span[i].start = cur_start;
        span[i].end   = k;        // [cur_start, k) are ticks for bar i
        cur_start = k;
    }
    return span;
}

} // namespace sentio