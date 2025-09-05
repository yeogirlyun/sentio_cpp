// test_rth.cpp (use your preferred test framework)
#include "sentio/rth_calendar.hpp"
#include "sentio/calendar_seed.hpp"
#include <cassert>
#include "cctz/time_zone.h"
#include "cctz/civil_time.h"

using namespace std::chrono;

static std::int64_t to_epoch_utc(int y,int m,int d,int hh,int mm,int ss,const char* tz_name){
  using namespace std::chrono;
  // Interpret given local wall time (tz_name) and convert to UTC epoch
  cctz::time_zone tz;
  if (!cctz::load_time_zone(tz_name, &tz)) {
    return 0;
  }
  
  auto ct = cctz::civil_second(y, m, d, hh, mm, ss);
  auto tp = cctz::convert(ct, tz);
  return tp.time_since_epoch().count();
}

int main() {
  auto cal = sentio::make_default_nyse_calendar();

  // 2022-09-06 09:30:00 ET should be RTH (day after Labor Day)
  auto t1 = to_epoch_utc(2022,9,6,9,30,0,"America/New_York");
  assert(cal.is_rth_utc(t1));

  // 2022-11-25 12:59:59 ET (Black Friday early close @ 13:00) -> RTH
  auto t2 = to_epoch_utc(2022,11,25,12,59,59,"America/New_York");
  assert(cal.is_rth_utc(t2));

  // 2022-11-25 13:00:01 ET -> NOT RTH (just past early close)
  auto t3 = to_epoch_utc(2022,11,25,13,0,1,"America/New_York");
  assert(!cal.is_rth_utc(t3));

  // Labor Day 2022-09-05 10:00 ET -> NOT RTH (holiday)
  auto t4 = to_epoch_utc(2022,9,5,10,0,0,"America/New_York");
  assert(!cal.is_rth_utc(t4));

  // 2022-09-06 09:29:59 ET -> Pre-open (NOT RTH)
  auto t5 = to_epoch_utc(2022,9,6,9,29,59,"America/New_York");
  assert(!cal.is_rth_utc(t5));

  // 2022-09-06 16:00:00 ET -> Still RTH (inclusive)
  auto t6 = to_epoch_utc(2022,9,6,16,0,0,"America/New_York");
  assert(cal.is_rth_utc(t6));

  return 0;
}