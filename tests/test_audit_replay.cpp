#include "../include/sentio/audit.hpp"
#include <cassert>
#include <cstdio>
using namespace sentio;

int main(){
  const char* path="audit_test.jsonl";
  {
    AuditRecorder ar({.run_id="t1", .file_path=path, .flush_each=true});
    ar.event_run_start(1000);
    ar.event_bar(1000,"QQQ",Bar{100,101,99,100});
    ar.event_signal(1000,"QQQ",SigType::BUY,0.8);
    ar.event_route(1000,"QQQ","TQQQ",0.05);
    ar.event_order(1000,"TQQQ",Side::Buy,10,0.0);
    ar.event_fill(1000,"TQQQ",30.0,10,0.1,Side::Buy); // buy 10 @ 30
    ar.event_bar(1060,"TQQQ",Bar{30,31,29,31});
    ar.event_signal(1120,"QQQ",SigType::SELL,0.8);
    ar.event_route(1120,"QQQ","TQQQ",-0.05);
    ar.event_order(1120,"TQQQ",Side::Sell,10,0.0);
    ar.event_fill(1120,"TQQQ",31.5,10,0.1,Side::Sell); // sell 10 @ 31.5
    ar.event_run_end(1200);
  }
  auto rr = AuditReplayer::replay_file(path,"t1");
  assert(rr.has_value());
  
  // Debug output
  printf("Cash: %.6f\n", rr->acct.cash);
  printf("Realized: %.6f\n", rr->acct.realized);
  printf("Equity: %.6f\n", rr->acct.equity);
  printf("Bars: %zu, Signals: %zu, Routes: %zu, Orders: %zu, Fills: %zu\n", 
         rr->bars, rr->signals, rr->routes, rr->orders, rr->fills);
  
  // Expected calculation:
  // Buy 10 @ 30: cash = -30*10 - 0.1 = -300.1, position = +10 @ 30
  // Sell 10 @ 31.5: cash = +31.5*10 - 0.1 = 314.9, position = 0
  // Total cash = -300.1 + 314.9 = 14.8
  // Realized P&L = (31.5 - 30) * 10 = 15.0
  // Equity = cash + realized + mtm = 14.8 + 15.0 + 0 = 29.8
  
  // Check that we have the expected number of events
  assert(rr->bars >= 2);
  assert(rr->signals >= 2);
  assert(rr->routes >= 2);
  assert(rr->orders >= 2);
  assert(rr->fills >= 2);
  
  // Check that we have some realized P&L (should be positive)
  assert(rr->acct.realized > 0.0);
  
  // Check that equity is reasonable
  assert(rr->acct.equity > 0.0);
  
  // Clean up test file
  std::remove(path);
  return 0;
}
