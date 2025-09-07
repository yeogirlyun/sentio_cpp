#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace sentio {

struct StandardScaler {
  std::vector<float> mean, inv_std;

  void fit(const float* X, std::int64_t rows, std::int64_t cols) {
    mean.assign(cols, 0.f);
    inv_std.assign(cols, 0.f);

    for (std::int64_t r=0; r<rows; ++r)
      for (std::int64_t c=0; c<cols; ++c)
        mean[c] += X[r*cols + c];
    for (std::int64_t c=0; c<cols; ++c) mean[c] /= std::max<std::int64_t>(1, rows);

    std::vector<double> var(cols, 0.0);
    for (std::int64_t r=0; r<rows; ++r)
      for (std::int64_t c=0; c<cols; ++c) {
        const double d = (double)X[r*cols + c] - (double)mean[c];
        var[c] += d*d;
      }
    for (std::int64_t c=0; c<cols; ++c) {
      const double sd = std::sqrt(std::max(1e-12, var[c] / std::max<std::int64_t>(1, rows)));
      inv_std[c] = (float)(1.0 / sd);
    }
  }

  void transform_inplace(float* X, std::int64_t rows, std::int64_t cols) const {
    for (std::int64_t r=0; r<rows; ++r)
      for (std::int64_t c=0; c<cols; ++c) {
        float& v = X[r*cols + c];
        v = (v - mean[c]) * inv_std[c];
      }
  }
};

} // namespace sentio
