// pnl_accounting.hpp
#pragma once
#include "core.hpp"  // Use Bar from core.hpp
#include <string>
#include <stdexcept>
#include <unordered_map>

namespace sentio {

class PriceBook {
public:
  // instrument -> latest bar (or map<ts,bar> for full history)
  const Bar* get_latest(const std::string& instrument) const;
  
  // Additional helper methods
  void upsert_latest(const std::string& instrument, const Bar& b);
  bool has_instrument(const std::string& instrument) const;
  std::size_t size() const;
};

// Use the instrument actually traded
double last_trade_price(const PriceBook& book, const std::string& instrument);

} // namespace sentio