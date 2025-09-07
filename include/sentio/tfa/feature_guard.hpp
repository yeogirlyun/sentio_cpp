#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace sentio::tfa {

struct FeatureGuard {
  int emit_from = 0;
  float pad_value = 0.0f;

  static inline bool is_finite(float x){
    return std::isfinite(x);
  }

  // returns mask: true = usable
  std::vector<uint8_t> build_mask_and_clean(float* X, int64_t rows, int64_t cols) const {
    std::vector<uint8_t> ok(rows, 0);
    // zero/pad early rows, mark them unusable
    for (int64_t r=0; r<std::min<int64_t>(emit_from, rows); ++r){
      for (int64_t c=0; c<cols; ++c) X[r*cols+c] = pad_value;
    }
    // after emit_from: sanitize NaN/Inf
    for (int64_t r=emit_from; r<rows; ++r){
      bool row_ok = true;
      for (int64_t c=0; c<cols; ++c){
        float& v = X[r*cols+c];
        if (!is_finite(v)) { v = 0.0f; row_ok = false; } // clean AND mark not-OK for signal
      }
      ok[r] = row_ok ? 1 : 0;
    }
    return ok;
  }
  
  // Overload for double vectors (from cached features)
  std::vector<uint8_t> build_mask_and_clean(std::vector<std::vector<double>>& features) const {
    const int64_t rows = features.size();
    const int64_t cols = rows > 0 ? features[0].size() : 0;
    std::vector<uint8_t> ok(rows, 0);
    
    // zero/pad early rows, mark them unusable
    for (int64_t r=0; r<std::min<int64_t>(emit_from, rows); ++r){
      for (int64_t c=0; c<cols; ++c) features[r][c] = pad_value;
    }
    
    // after emit_from: sanitize NaN/Inf
    for (int64_t r=emit_from; r<rows; ++r){
      bool row_ok = true;
      for (int64_t c=0; c<cols; ++c){
        double& v = features[r][c];
        if (!std::isfinite(v)) { v = 0.0; row_ok = false; } // clean AND mark not-OK for signal
      }
      ok[r] = row_ok ? 1 : 0;
    }
    return ok;
  }
};

} // namespace sentio::tfa
