#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace sentio {

struct FeatureMatrix {
  int64_t rows{0}, cols{0};
  std::vector<float> data; // row-major [rows, cols]
};

struct IFeatureProvider {
  virtual ~IFeatureProvider() = default;
  virtual FeatureMatrix get_features_for(const std::string& symbol) = 0;
  virtual std::vector<std::string> feature_names() const = 0; // authoritative order in source
  virtual int seq_len() const = 0; // sequence length (warmup)
};

} // namespace sentio
