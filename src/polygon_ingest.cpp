#include "sentio/polygon_ingest.hpp"
#include "sentio/time_utils.hpp"
#include "sentio/rth_calendar.hpp"
#include "sentio/calendar_seed.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace sentio {

static const TradingCalendar& nyse_calendar() {
  static TradingCalendar cal = make_default_nyse_calendar();
  return cal;
}

static inline bool accept_bar_utc(std::int64_t ts_utc) {
  return nyse_calendar().is_rth_utc(ts_utc, "America/New_York");
}

static inline bool valid_bar(const ProviderBar& b) {
  if (!std::isfinite(b.open)  || !std::isfinite(b.high) ||
      !std::isfinite(b.low)   || !std::isfinite(b.close)||
      !std::isfinite(b.volume)) return false;
  if (b.open <= 0 || b.high <= 0 || b.low <= 0 || b.close <= 0) return false;
  if (b.low > b.high) return false;
  return !b.symbol.empty();
}

static inline void upsert_bar(PriceBook& book, const std::string& instrument, const ProviderBar& pb) {
  Bar b;
  b.open  = pb.open;
  b.high  = pb.high;
  b.low   = pb.low;
  b.close = pb.close;
  book.upsert_latest(instrument, b);
}

static inline std::int64_t to_epoch_s(const std::variant<std::int64_t, double, std::string>& ts) {
  auto tp = to_utc_sys_seconds(ts);
  return std::chrono::time_point_cast<std::chrono::seconds>(tp).time_since_epoch().count();
}

std::size_t ingest_provider_bars(const std::vector<ProviderBar>& input, PriceBook& book) {
  std::size_t accepted = 0;
  for (const auto& pb : input) {
    if (!valid_bar(pb)) continue;

    std::int64_t ts_utc{};
    try {
      ts_utc = to_epoch_s(pb.ts);
    } catch (...) {
      continue; // skip malformed time
    }

    if (!accept_bar_utc(ts_utc)) continue;

    upsert_bar(book, pb.symbol, pb);
    ++accepted;
  }
  return accepted;
}

bool ingest_provider_bar(const ProviderBar& bar, PriceBook& book) {
  return ingest_provider_bars(std::vector<ProviderBar>{bar}, book) == 1;
}

} // namespace sentio