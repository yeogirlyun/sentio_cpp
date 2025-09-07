#include "sentio/feature/feature_feeder_guarded.hpp"
#include "sentio/feature/feature_builder_guarded.hpp"
#include "sentio/sym/symbol_utils.hpp"

namespace sentio {

bool FeatureFeederGuarded::infer_base_if_needed_(
  const std::unordered_map<std::string, std::vector<Bar>>& series,
  std::string& base_out)
{
  if (!base_out.empty()) { base_out = to_upper(base_out); return true; }
  // Prefer QQQ if present, else pick the first non-leveraged symbol
  if (series.find("QQQ") != series.end()) { base_out = "QQQ"; return true; }
  for (auto& kv : series) {
    if (!is_leveraged(kv.first)) { base_out = to_upper(kv.first); return true; }
  }
  return false;
}

bool FeatureFeederGuarded::initialize(const FeederInit& init) {
  prices_.clear();
  asof_.clear();
  base_ts_.clear();
  X_ = {};
  scaler_ = {};
  base_symU_.clear();

  // Cache the input prices (all symbols). We'll filter logically below.
  for (auto& kv : init.series) {
    prices_.emplace(to_upper(kv.first), kv.second);
  }

  // Decide base
  base_symU_ = init.base_symbol;
  if (!infer_base_if_needed_(prices_, base_symU_)) {
    // cannot proceed without a base
    return false;
  }

  // Build features **only** for base
  auto it = prices_.find(base_symU_);
  if (it == prices_.end() || it->second.empty()) return false;

  // Build base timestamp vector
  base_ts_.resize(it->second.size());
  for (std::size_t i=0; i<it->second.size(); ++i) base_ts_[i] = it->second[i].ts_epoch_us;

  // Strict guard: do not allow leveraged symbol to reach builder
  if (is_leveraged(base_symU_)) return false;

  X_ = build_features_for_base(base_symU_, it->second);
  if (X_.rows == 0) return false;

  // Fit scaler ONLY on base features; transform in-place
  scaler_.fit(X_.data.data(), X_.rows, X_.cols);
  scaler_.transform_inplace(X_.data.data(), X_.rows, X_.cols);

  // Build as-of maps for any instrument in the base family (leveraged or base itself)
  for (auto& kv : prices_) {
    const auto symU = kv.first;
    // Only build maps for instruments whose resolved base == base_symU_
    if (resolve_base(symU) != base_symU_) continue;

    // Build inst_ts
    std::vector<std::int64_t> inst_ts; inst_ts.reserve(kv.second.size());
    for (auto& b : kv.second) inst_ts.push_back(b.ts_epoch_us);

    asof_[symU] = build_asof_index(base_ts_, inst_ts);
  }

  return true;
}

bool FeatureFeederGuarded::allowed_for_exec(const std::string& symbol) const {
  const auto symU = to_upper(symbol);
  return resolve_base(symU) == base_symU_;
}

} // namespace sentio
