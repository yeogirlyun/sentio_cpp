#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "sentio/core/bar.hpp"
#include "sentio/feature/feature_matrix.hpp"
#include "sentio/feature/standard_scaler.hpp"
#include "sentio/sym/leverage_registry.hpp"
#include "sentio/exec/asof_index.hpp"

namespace sentio {

struct FeederInit {
  // All price series loaded (both base and leveraged allowed here)
  std::unordered_map<std::string, std::vector<Bar>> series; // symbol -> bars
  // The single base you want to signal on this run (e.g., "QQQ").
  // If empty, we'll infer it from presence (prefers QQQ if present).
  std::string base_symbol;
};

class FeatureFeederGuarded {
public:
  bool initialize(const FeederInit& init);

  const FeatureMatrix& features() const { return X_; }
  const StandardScaler& scaler() const { return scaler_; }
  const std::vector<std::int64_t>& base_ts() const { return base_ts_; }

  // For execution: map instrument rows to base rows
  // (present only for leveraged family members that exist in input)
  const std::vector<int32_t>* asof_map_for(const std::string& symbol) const {
    auto it = asof_.find(to_upper(symbol));
    if (it == asof_.end()) return nullptr;
    return &it->second;
  }

  // True if symbol is permitted for execution in this run (base or leverage family)
  bool allowed_for_exec(const std::string& symbol) const;

  const std::string& base() const { return base_symU_; }

private:
  std::string base_symU_;
  FeatureMatrix X_;
  StandardScaler scaler_;
  std::vector<std::int64_t> base_ts_;
  std::unordered_map<std::string, std::vector<int32_t>> asof_; // SYM -> asof index into base
  std::unordered_map<std::string, std::vector<Bar>> prices_;   // keep original price series

  bool infer_base_if_needed_(const std::unordered_map<std::string, std::vector<Bar>>& series,
                             std::string& base_out);
};

} // namespace sentio
