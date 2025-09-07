#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <cstring>

namespace sentio {

struct SafeMatrix {
  std::vector<float> buf;
  int64_t rows{0}, cols{0};

  void resize(int64_t r, int64_t c) {
    if (r < 0 || c < 0) throw std::runtime_error("SafeMatrix: negative shape");
    if (c > (int64_t)(std::numeric_limits<size_t>::max()/sizeof(float))/ (r>0?r:1))
      throw std::runtime_error("SafeMatrix: size overflow");
    rows = r; cols = c;
    buf.assign(static_cast<size_t>(r)*static_cast<size_t>(c), 0.0f);
  }

  inline float* row_ptr(int64_t r) {
    if ((uint64_t)r >= (uint64_t)rows) throw std::runtime_error("SafeMatrix: row OOB");
    return buf.data() + (size_t)r*(size_t)cols;
  }
  inline const float* row_ptr(int64_t r) const {
    if ((uint64_t)r >= (uint64_t)rows) throw std::runtime_error("SafeMatrix: row OOB");
    return buf.data() + (size_t)r*(size_t)cols;
  }
  
  // Convenience accessors
  float* data() { return buf.data(); }
  const float* data() const { return buf.data(); }
  size_t size() const { return buf.size(); }
  bool empty() const { return buf.empty(); }
};

} // namespace sentio
