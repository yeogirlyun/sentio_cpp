#pragma once
#include <vector>
#include <cstdint>

namespace sentio {

// Build a map from instrument rows to base rows via as-of (<= ts) alignment.
// base_ts, inst_ts must be monotonic non-decreasing (your loaders should ensure).
inline std::vector<int32_t> build_asof_index(const std::vector<std::int64_t>& base_ts,
                                             const std::vector<std::int64_t>& inst_ts) {
  std::vector<int32_t> idx(inst_ts.size(), -1);
  std::size_t j = 0;
  if (base_ts.empty()) return idx;
  for (std::size_t i = 0; i < inst_ts.size(); ++i) {
    auto t = inst_ts[i];
    while (j + 1 < base_ts.size() && base_ts[j + 1] <= t) ++j;
    idx[i] = static_cast<int32_t>(j);
  }
  return idx;
}

} // namespace sentio
