#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace sentio {

// Maps runtime-built feature matrix [N, F_src] into model-expected order [N, F_dst].
struct ColumnProjector {
  // dst[i] = index in src, or -1 if missing (will fill with fill_value)
  std::vector<int> map;          // length = F_dst
  std::vector<float> fill;       // length = F_dst (per-column fill)
  int64_t F_src{0}, F_dst{0};
  float fill_default{0.0f};

  static ColumnProjector make(
      const std::vector<std::string>& src_names,
      const std::vector<std::string>& dst_names,
      float fill_default = 0.0f)
  {
    ColumnProjector P; P.F_src = (int64_t)src_names.size(); P.F_dst = (int64_t)dst_names.size();
    P.fill_default = fill_default;
    P.map.assign(P.F_dst, -1);
    P.fill.assign(P.F_dst, fill_default);

    std::unordered_map<std::string, int> pos;
    pos.reserve(src_names.size()*2);
    for (int i=0;i<(int)src_names.size();++i) pos[src_names[i]] = i;

    int missing = 0;
    for (int j=0;j<(int)dst_names.size();++j){
      auto it = pos.find(dst_names[j]);
      if (it != pos.end()){
        P.map[j] = it->second;
      } else {
        P.map[j] = -1; // missing column: will be filled
        missing++;
      }
    }
    if (missing>0){
      std::cout << "[ColumnProjector] Missing columns: " << missing << " will be filled with "
                << fill_default << std::endl;
    }
    // Log extras (present in src but not expected by model)
    int extras=0;
    for (auto& kv : pos){
      if (std::find(dst_names.begin(), dst_names.end(), kv.first) == dst_names.end()) extras++;
    }
    if (extras>0){
      std::cout << "[ColumnProjector] Extra columns in runtime features: " << extras
                << " (will be dropped)" << std::endl;
    }
    return P;
  }

  // Apply to row-major [N, F_src] → [N, F_dst]
  void project(const float* X_src, int64_t N, int64_t F_src, std::vector<float>& X_out) const {
    if (F_src != this->F_src) {
      throw std::runtime_error("ColumnProjector: F_src mismatch");
    }
    X_out.assign((size_t)N * (size_t)F_dst, fill_default);
    for (int64_t r=0;r<N;++r){
      const float* src = X_src + r*F_src;
      float* dst = X_out.data() + r*F_dst;
      for (int j=0;j<F_dst;++j){
        int si = map[j];
        dst[j] = (si >= 0) ? src[si] : fill[j];
      }
    }
  }
  
  // Convenience method for vector<vector<double>> from cached features
  void project_cached(const std::vector<std::vector<double>>& features_in, 
                     std::vector<std::vector<double>>& features_out) const {
    const int64_t N = features_in.size();
    if (N == 0) return;
    
    features_out.resize(N);
    for (int64_t r = 0; r < N; ++r) {
      features_out[r].resize(F_dst, fill_default);
      for (int j = 0; j < F_dst; ++j) {
        int si = map[j];
        if (si >= 0 && si < (int)features_in[r].size()) {
          features_out[r][j] = features_in[r][si];
        } else {
          features_out[r][j] = fill[j];
        }
      }
    }
  }
};

} // namespace sentio
