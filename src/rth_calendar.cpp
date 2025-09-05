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

bool TradingCalendar::is_rth_utc(std::int64_t ts_utc, const std::string& tz_name) const {
  using namespace std::chrono;
  
  // Convert to cctz time
  cctz::time_point<cctz::seconds> tp{cctz::seconds{ts_utc}};
  
  // Get timezone
  cctz::time_zone tz;
  if (!cctz::load_time_zone(tz_name, &tz)) {
    return false;
  }
  
  // Convert to local time
  auto lt = cctz::convert(tp, tz);
  
  // Extract civil time components
  auto ct = cctz::civil_second(lt);
  
  // Check if weekend (Saturday = 6, Sunday = 0)
  auto wd = cctz::get_weekday(ct);
  if (wd == cctz::weekday::saturday || wd == cctz::weekday::sunday) {
    return false;
  }
  
  // Check if RTH (9:30 AM - 4:00 PM ET inclusive)
  if (ct.hour() < 9 || (ct.hour() == 9 && ct.minute() < 30)) {
    return false;  // Before 9:30 AM
  }
  if (ct.hour() > 16 || (ct.hour() == 16 && ct.minute() > 0)) {
    return false;  // After 4:00 PM
  }
  
  return true;
}

} // namespace sentio