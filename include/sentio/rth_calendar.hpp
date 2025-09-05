// rth_calendar.hpp
#pragma once
#include <chrono>
#include <string>
#include <unordered_set>
#include <unordered_map>

namespace sentio {

struct TradingCalendar {
  // Holidays: YYYYMMDD integers for fast lookups
  std::unordered_set<int> full_holidays;
  // Early closes (e.g., Black Friday): YYYYMMDD -> close second-of-day (e.g., 13:00 = 13*3600)
  std::unordered_map<int,int> early_close_sec;

  // Regular RTH bounds (seconds from midnight ET)
  static constexpr int RTH_OPEN_SEC  = 9*3600 + 30*60;  // 09:30:00
  static constexpr int RTH_CLOSE_SEC = 16*3600;         // 16:00:00

  // Return yyyymmdd in ET from a zoned_time (no allocations)
  static int yyyymmdd_from_local(const std::chrono::hh_mm_ss<std::chrono::seconds>& /*tod*/,
                                 std::chrono::year_month_day ymd) {
    int y = int(ymd.year());
    unsigned m = unsigned(ymd.month());
    unsigned d = unsigned(ymd.day());
    return y*10000 + int(m)*100 + int(d);
  }

  // Main predicate:
  //   ts_utc  = UTC wall clock in seconds since epoch
  //   tz_name = "America/New_York"
  bool is_rth_utc(std::int64_t ts_utc, const std::string& tz_name = "America/New_York") const;
};

} // namespace sentio