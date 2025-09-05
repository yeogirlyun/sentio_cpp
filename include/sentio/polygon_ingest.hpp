#pragma once
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace sentio {

// Polygon-like bar from feeds/adapters.
// ts can be epoch seconds (UTC), epoch milliseconds (UTC), or ISO8601 string.
struct ProviderBar {
  std::string symbol;                                   // instrument actually traded (QQQ/TQQQ/SQQQ/…)
  std::variant<std::int64_t, double, std::string> ts;   // epoch sec (int64), epoch ms (double), or ISO8601
  double open{};
  double high{};
  double low{};
  double close{};
  double volume{};
};

struct Bar { double open{}, high{}, low{}, close{}; };

class PriceBook {
public:
  void upsert_latest(const std::string& instrument, const Bar& b);
  const Bar* get_latest(const std::string& instrument) const;
  bool has_instrument(const std::string& instrument) const;
  std::size_t size() const;
};

std::size_t ingest_provider_bars(const std::vector<ProviderBar>& input, PriceBook& book);
bool        ingest_provider_bar(const ProviderBar& bar, PriceBook& book);

} // namespace sentio
