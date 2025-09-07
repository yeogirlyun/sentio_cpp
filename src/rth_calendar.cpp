// rth_calendar.cpp
#include "sentio/rth_calendar.hpp"
#include <chrono>
#include <string>
#include <stdexcept>
#include <iostream>
#include "cctz/time_zone.h"
#include "cctz/civil_time.h"

using namespace std::chrono;

namespace sentio {

bool TradingCalendar::is_rth_utc(std::int64_t ts_utc, [[maybe_unused]] const std::string& tz_name) const {
  using namespace std::chrono;
  
  // Convert to cctz time
  cctz::time_point<cctz::seconds> tp{cctz::seconds{ts_utc}};
  
  // Get UTC timezone
  cctz::time_zone utc_tz;
  if (!cctz::load_time_zone("UTC", &utc_tz)) {
    return false;
  }
  
  // Convert to UTC civil time
  auto lt = cctz::convert(tp, utc_tz);
  auto ct = cctz::civil_second(lt);
  
  // Check if weekend (Saturday = 6, Sunday = 0)
  auto wd = cctz::get_weekday(ct);
  if (wd == cctz::weekday::saturday || wd == cctz::weekday::sunday) {
    return false;
  }
  
  // Check if RTH in UTC
  // EST: 14:30-21:00 UTC (9:30 AM - 4:00 PM EST)
  // EDT: 13:30-20:00 UTC (9:30 AM - 4:00 PM EDT)
  int hour = ct.hour();
  int minute = ct.minute();
  int month = ct.month();
  
  // Simple DST check: April-October is EDT, rest is EST
  bool is_edt = (month >= 4 && month <= 10);
  
  if (is_edt) {
    // EDT: 13:30-20:00 UTC
    if (hour < 13 || (hour == 13 && minute < 30)) {
      return false;  // Before 13:30 UTC
    }
    if (hour >= 20) {
      return false;  // After 20:00 UTC
    }
  } else {
    // EST: 14:30-21:00 UTC
    if (hour < 14 || (hour == 14 && minute < 30)) {
      return false;  // Before 14:30 UTC
    }
    if (hour >= 21) {
      return false;  // After 21:00 UTC
    }
  }
  
  return true;
}

} // namespace sentio