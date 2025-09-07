#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <climits>

namespace sentio {

// HARDENED VERSION: Maps runtime-built feature matrix [N, F_src] into model-expected order [N, F_dst].
// Uses size_t arithmetic to prevent overflows and bounds checking
struct ColumnProjectorSafe {
  std::vector<int> map;
  size_t F_src{0}, F_dst{0};
  float fill_value{0.0f};

  static ColumnProjectorSafe make(const std::vector<std::string>& src, const std::vector<std::string>& dst, float fill=0.0f) {
    ColumnProjectorSafe p; 
    p.F_src=src.size(); 
    p.F_dst=dst.size(); 
    p.fill_value=fill;
    p.map.assign(p.F_dst, -1);
    
    std::unordered_map<std::string,int> pos; 
    pos.reserve(src.size()*2);
    for (size_t i=0; i<src.size(); ++i) pos[src[i]] = (int)i;
    
    int missing = 0;
    for (size_t j=0; j<dst.size(); ++j) {
      auto it=pos.find(dst[j]);
      p.map[j] = (it!=pos.end()) ? it->second : -1;
      if (it == pos.end()) missing++;
    }
    
    if (missing>0){
      std::cerr << "[ColumnProjectorSafe] Missing columns: " << missing << " will be filled with "
                << fill << "\n";
    }
    
    // Log extras (present in src but not expected by model)
    int extras=0;
    for (auto& kv : pos){
      if (std::find(dst.begin(), dst.end(), kv.first) == dst.end()) extras++;
    }
    if (extras>0){
      std::cerr << "[ColumnProjectorSafe] Extra columns in runtime features: " << extras
                << " (will be dropped)\n";
    }
    return p;
  }

  void project(const float* X, size_t rows, size_t Fsrc, std::vector<float>& out) const {
    if (Fsrc != F_src) throw std::runtime_error("ColumnProjectorSafe: F_src mismatch");
    
    // Check for potential overflow in total size calculation
    if (rows > 0 && F_dst > SIZE_MAX / rows) {
      throw std::runtime_error("ColumnProjectorSafe: output size overflow");
    }
    
    out.assign(rows*F_dst, fill_value);
    for (size_t r=0;r<rows;++r){
      const float* src = X + r*F_src;
      float* dst = out.data() + r*F_dst;
      for (size_t j=0;j<F_dst;++j){
        int si = map[j];
        dst[j] = (si>=0) ? src[(size_t)si] : fill_value;
      }
    }
  }
  
  // Legacy int64_t interface for backward compatibility with bounds checking
  void project(const float* X_src, int64_t N, int64_t F_src_legacy, std::vector<float>& X_out) const {
    if (N < 0 || F_src_legacy < 0) {
      throw std::runtime_error("ColumnProjectorSafe: negative dimensions");
    }
    project(X_src, (size_t)N, (size_t)F_src_legacy, X_out);
  }
  
  // Cached features version with bounds checking  
  void project_cached(const std::vector<std::vector<double>>& X_src_cached, std::vector<std::vector<double>>& X_out) const {
    if (X_src_cached.empty()) {
      X_out.clear();
      return;
    }
    size_t N = X_src_cached.size();
    size_t F_src_actual = X_src_cached[0].size();
    if (F_src_actual != this->F_src) {
      throw std::runtime_error("ColumnProjectorSafe::project_cached: F_src mismatch");
    }

    X_out.assign(N, std::vector<double>(F_dst, fill_value));
    for (size_t r = 0; r < N; ++r) {
      // Bounds check on input row
      if (X_src_cached[r].size() != F_src_actual) {
        throw std::runtime_error("ColumnProjectorSafe::project_cached: inconsistent row size");
      }
      for (size_t j = 0; j < F_dst; ++j) {
        int si = map[j];
        X_out[r][j] = (si >= 0) ? X_src_cached[r][(size_t)si] : fill_value;
      }
    }
  }
};

} // namespace sentio
