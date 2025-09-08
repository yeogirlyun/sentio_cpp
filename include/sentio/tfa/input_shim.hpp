#pragma once
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <unordered_map>

namespace sentio {

inline std::vector<float> shim_to_expected_input(const float* X_src,
                                                 int64_t rows,
                                                 int64_t F_src,
                                                 const std::vector<std::string>& runtime_names,
                                                 const std::vector<std::string>& expected_names,
                                                 int   F_expected,
                                                 float pad_value = 0.0f)
{
  // Fast path: exact match
  if (F_src == F_expected && runtime_names == expected_names) {
    return std::vector<float>(X_src, X_src + (size_t)rows*(size_t)F_src);
  }

  // Hotfix path: legacy model expects a leading 'ts' column
  const bool model_leads_with_ts = !expected_names.empty() && expected_names.front() == "ts";
  const bool runtime_has_ts      = !runtime_names.empty()   && runtime_names.front()  == "ts";
  if (model_leads_with_ts && !runtime_has_ts && F_src + 1 == F_expected) {
    std::vector<float> out((size_t)rows * (size_t)F_expected, pad_value);
    for (int64_t r=0; r<rows; ++r) {
      float* dst = out.data() + r*F_expected;
      std::memcpy(dst + 1, X_src + r*F_src, sizeof(float) * (size_t)F_src);
      dst[0] = 0.0f; // dummy ts
    }
    std::cerr << "[TFA] HOTFIX: injected dummy 'ts' col to satisfy legacy 56-dim model\n";
    return out;
  }

  // General name-based projection (drops extras, fills missing with pad)
  std::vector<float> out((size_t)rows * (size_t)F_expected, pad_value);
  // build index map
  std::unordered_map<std::string,int> pos;
  pos.reserve(runtime_names.size()*2);
  for (int i=0;i<(int)runtime_names.size();++i) pos[runtime_names[i]] = i;
  for (int64_t r=0; r<rows; ++r) {
    const float* src = X_src + r*F_src;
    float* dst = out.data() + r*F_expected;
    for (int j=0; j<F_expected; ++j) {
      auto it = pos.find(expected_names[j]);
      if (it != pos.end()) dst[j] = src[it->second];
    }
  }
  std::cerr << "[TFA] INFO: name-based projection applied (srcF="<<F_src<<" -> dstF="<<F_expected<<")\n";
  return out;
}

} // namespace sentio
