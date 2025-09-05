// calendar_seed.hpp
#pragma once
#include "rth_calendar.hpp"

namespace sentio {

inline TradingCalendar make_default_nyse_calendar() {
  TradingCalendar c;

  // Full-day holidays (partial sample; fill your range robustly)
  // 2022: New Year (obs 2021-12-31), MLK 2022-01-17, Presidents 02-21,
  // Good Friday 04-15, Memorial 05-30, Juneteenth (obs 06-20), Independence 07-04,
  // Labor 09-05, Thanksgiving 11-24, Christmas (obs 2022-12-26)
  c.full_holidays.insert(20220117);
  c.full_holidays.insert(20220221);
  c.full_holidays.insert(20220415);
  c.full_holidays.insert(20220530);
  c.full_holidays.insert(20220620);
  c.full_holidays.insert(20220704);
  c.full_holidays.insert(20220905);
  c.full_holidays.insert(20221124);
  c.full_holidays.insert(20221226);

  // Early closes (sample): Black Friday 2022-11-25 @ 13:00 ET
  c.early_close_sec.emplace(20221125, 13*3600);

  return c;
}

} // namespace sentio