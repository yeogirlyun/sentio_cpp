// exec_types.hpp
#pragma once
#include <string>

namespace sentio {

struct ExecutionIntent {
  std::string base_symbol;     // e.g., "QQQ"
  std::string instrument;      // e.g., "TQQQ" or "SQQQ" or "QQQ"
  double      qty = 0.0;
  double      leverage = 1.0;  // informational; actual product carries leverage
  double      score = 0.0;     // signal strength
};

} // namespace sentio