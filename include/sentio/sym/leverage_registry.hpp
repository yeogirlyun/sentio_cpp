#pragma once
#include <string>
#include <unordered_map>
#include <mutex>
#include "sentio/sym/symbol_utils.hpp"

namespace sentio {

// Captures the leveraged instrument's relationship to a base ticker
struct LeverageSpec {
  std::string base;     // e.g., "QQQ"
  float factor{1.f};    // e.g., 3.0 for TQQQ, 1.0 for PSQ (but inverse)
  bool inverse{false};  // true for PSQ/SQQQ (short)
};

// Thread-safe global registry
class LeverageRegistry {
  std::unordered_map<std::string, LeverageSpec> map_; // key: UPPER(symbol)
  std::mutex mu_;
  LeverageRegistry() { seed_defaults_(); }

  void seed_defaults_() {
    // QQQ family (PSQ removed - moderate sell signals now use SHORT QQQ)
    map_.emplace("TQQQ", LeverageSpec{"QQQ", 3.f, false});
    map_.emplace("SQQQ", LeverageSpec{"QQQ", 3.f, true});
    // You can extend similarly for SPY, TSLA, BTC ETFs, etc.
    // Examples:
    // map_.emplace("UPRO", LeverageSpec{"SPY", 3.f, false});
    // map_.emplace("SPXU", LeverageSpec{"SPY", 3.f, true});
    // map_.emplace("TSLQ", LeverageSpec{"TSLA", 1.f, true});
  }

public:
  static LeverageRegistry& instance() {
    static LeverageRegistry x;
    return x;
  }

  void register_leveraged(const std::string& symbol, LeverageSpec spec) {
    std::lock_guard<std::mutex> lk(mu_);
    map_[to_upper(symbol)] = std::move(spec);
  }

  bool lookup(const std::string& symbol, LeverageSpec& out) const {
    const auto key = to_upper(symbol);
    auto it = map_.find(key);
    if (it == map_.end()) return false;
    out = it->second;
    return true;
  }
};

// Convenience helpers
inline bool is_leveraged(const std::string& symbol) {
  LeverageSpec tmp;
  return LeverageRegistry::instance().lookup(symbol, tmp);
}

inline std::string resolve_base(const std::string& symbol) {
  LeverageSpec tmp;
  if (LeverageRegistry::instance().lookup(symbol, tmp)) return tmp.base;
  return to_upper(symbol);
}

} // namespace sentio
