#pragma once
#include "audit.hpp"
#include "core.hpp"
#include <unordered_map>
#include <vector>

namespace sentio {
// Rebuild positions/cash/equity from fills and compare to stored snapshots.
// Return true if match within epsilon; otherwise write a simple error to stderr.
bool replay_and_assert(Auditor& au, long long run_id, double eps=1e-6);

// Enhanced replay with market data for accurate final pricing
bool replay_and_assert_with_data(Auditor& au, long long run_id, 
                                 const std::unordered_map<std::string, std::vector<Bar>>& market_data,
                                 double eps=1e-6);
} // namespace sentio

