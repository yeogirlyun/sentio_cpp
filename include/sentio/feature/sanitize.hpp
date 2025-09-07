#pragma once
#include <vector>
#include <cmath>
#include <cstdint>

namespace sentio {

// NaN/Inf sanitation before model (and a ready mask)
inline std::vector<uint8_t> sanitize_and_ready(float* X, int64_t rows, int64_t cols, int emit_from, float pad=0.0f){
  std::vector<uint8_t> ok((size_t)rows, 0);
  
  // Pad rows before emit_from
  for (int64_t r=0; r<std::min<int64_t>(emit_from, rows); ++r){
    for (int64_t c=0;c<cols;++c) X[r*cols+c] = pad;
  }
  
  // Sanitize and check rows from emit_from onward
  for (int64_t r=emit_from; r<rows; ++r){
    bool good=true;
    float* row = X + r*cols;
    for (int64_t c=0;c<cols;++c){
      float v=row[c];
      if (!std::isfinite(v)){ 
        row[c]=0.0f; 
        good=false; 
      }
    }
    ok[(size_t)r] = good ? 1 : 0;
  }
  return ok;
}

// Overload for cached features (vector<vector<double>>)
inline std::vector<uint8_t> sanitize_cached_features(std::vector<std::vector<double>>& features, int emit_from, double pad=0.0){
  std::vector<uint8_t> ok(features.size(), 0);
  
  if (features.empty()) return ok;
  
  size_t cols = features[0].size();
  
  // Pad rows before emit_from
  for (size_t r=0; r<std::min<size_t>(emit_from, features.size()); ++r){
    for (size_t c=0; c<cols; ++c) {
      features[r][c] = pad;
    }
  }
  
  // Sanitize and check rows from emit_from onward
  for (size_t r=emit_from; r<features.size(); ++r){
    bool good=true;
    for (size_t c=0; c<features[r].size(); ++c){
      double v = features[r][c];
      if (!std::isfinite(v)){ 
        features[r][c] = 0.0; 
        good=false; 
      }
    }
    ok[r] = good ? 1 : 0;
  }
  return ok;
}

} // namespace sentio
