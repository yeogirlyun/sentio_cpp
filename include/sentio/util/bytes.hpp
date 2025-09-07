#pragma once
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace sentio {

// Safe memory copy with bounds checking
inline void bytes_copy(void* dst, const void* src, size_t count){
  if (!dst || !src) throw std::runtime_error("bytes_copy: null ptr");
  if (count > 0) {
    std::memcpy(dst, src, count);
  }
}

// Safe memory set with bounds checking
inline void bytes_set(void* ptr, int value, size_t count) {
  if (!ptr) throw std::runtime_error("bytes_set: null ptr");
  if (count > 0) {
    std::memset(ptr, value, count);
  }
}

} // namespace sentio
