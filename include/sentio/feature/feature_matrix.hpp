#pragma once
#include <vector>
#include <cstdint>

namespace sentio {
struct FeatureMatrix {
  std::vector<float> data; // row-major [rows, cols]
  std::int64_t rows{0};
  std::int64_t cols{0};
  inline float* row_ptr(std::int64_t r) { return data.data() + r*cols; }
  inline const float* row_ptr(std::int64_t r) const { return data.data() + r*cols; }
};
}
