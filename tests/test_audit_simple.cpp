#include "../include/sentio/audit.hpp"
#include <cassert>
#include <cstdio>
using namespace sentio;

int main(){
  const char* path="audit_simple_test.jsonl";
  {
    AuditRecorder ar({.run_id="simple", .file_path=path, .flush_each=true});
    ar.event_run_start(1000);
    ar.event_bar(1000,"TQQQ",AuditBar{30,31,29,30,1000});
    ar.event_order(1000,"TQQQ",Side::Buy,10,0.0);
    ar.event_fill(1000,"TQQQ",30.0,10,0.1,Side::Buy); // buy 10 @ 30
    ar.event_bar(1060,"TQQQ",AuditBar{30,31,29,31,1000});
    ar.event_order(1060,"TQQQ",Side::Sell,10,0.0);
    ar.event_fill(1060,"TQQQ",31.0,10,0.1,Side::Sell); // sell 10 @ 31
    ar.event_run_end(1200);
  }
  auto rr = AuditReplayer::replay_file(path,"simple");
  assert(rr.has_value());
  
  // Debug output
  printf("Cash: %.6f\n", rr->acct.cash);
  printf("Realized: %.6f\n", rr->acct.realized);
  printf("Equity: %.6f\n", rr->acct.equity);
  printf("Bars: %zu, Fills: %zu\n", rr->bars, rr->fills);
  
  // Expected:
  // Buy 10 @ 30: cash = -30*10 - 0.1 = -300.1
  // Sell 10 @ 31: cash = +31*10 - 0.1 = 309.9
  // Total cash = -300.1 + 309.9 = 9.8
  // Realized P&L = (31 - 30) * 10 = 10.0
  // Equity = 9.8 + 10.0 = 19.8
  
  // Clean up test file
  std::remove(path);
  return 0;
}
