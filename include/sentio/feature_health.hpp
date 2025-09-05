#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sentio {

struct FeatureIssue {
  std::int64_t ts_utc{};
  std::string  kind;      // e.g., "NaN", "Gap", "Backwards_TS"
  std::string  detail;
};

struct FeatureHealthReport {
  std::vector<FeatureIssue> issues;
  bool ok() const { return issues.empty(); }
};

struct FeatureHealthCfg {
  // bar spacing in seconds (e.g., 60 for 1m). 0 = skip spacing checks.
  int expected_spacing_sec{60};
  bool check_nan{true};
  bool check_monotonic_time{true};
};

struct PricePoint { std::int64_t ts_utc{}; double close{}; };

FeatureHealthReport check_feature_health(const std::vector<PricePoint>& series,
                                         const FeatureHealthCfg& cfg);

} // namespace sentio
