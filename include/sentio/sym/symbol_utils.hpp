#pragma once
#include <string>
#include <algorithm>

namespace sentio {
inline std::string to_upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::toupper(c); });
  return s;
}
}
