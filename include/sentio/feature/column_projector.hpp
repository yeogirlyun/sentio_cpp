#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace sentio {

struct ColumnProjector {
  std::vector<int> src_to_dst; // index in src for each dst column; -1 -> pad
  float pad{0.0f};

  static ColumnProjector make(const std::vector<std::string>& src,
                              const std::vector<std::string>& dst,
                              float pad_value){
    std::unordered_map<std::string,int> pos;
    for (int i=0;i<(int)src.size();++i) pos[src[i]] = i;
    ColumnProjector P; P.pad = pad_value;
    P.src_to_dst.resize(dst.size(), -1);
    for (int j=0;j<(int)dst.size();++j){
      auto it = pos.find(dst[j]);
      P.src_to_dst[j] = (it==pos.end()) ? -1 : it->second;
    }
    return P;
  }

  void project(const float* X, size_t rows, size_t src_cols, std::vector<float>& Y) const {
    const size_t dst_cols = src_to_dst.size();
    Y.assign(rows * dst_cols, pad);
    for (size_t r=0;r<rows;++r){
      const float* src = X + r*src_cols;
      float* dst = Y.data() + r*dst_cols;
      for (size_t j=0;j<dst_cols;++j){
        int si = src_to_dst[j];
        if (si>=0) dst[j] = src[si];
      }
    }
  }
};

} // namespace sentio
