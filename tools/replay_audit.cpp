#include "../include/sentio/audit.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>

int main(int argc, char** argv){
  if (argc < 2){
    std::fprintf(stderr, "Usage: %s <audit.jsonl> [run_id]\n", argv[0]);
    return 1;
  }
  std::string path = argv[1];
  std::string run_id = (argc >= 3) ? argv[2] : std::string("");
  auto rr = sentio::AuditReplayer::replay_file(path, run_id);
  if (!rr.has_value()){
    std::fprintf(stderr, "Replay failed for %s\n", path.c_str());
    return 2;
  }
  const auto& r = *rr;
  std::printf("Replay OK: %s\n", path.c_str());
  std::printf("Bars=%zu Signals=%zu Routes=%zu Orders=%zu Fills=%zu\n", r.bars, r.signals, r.routes, r.orders, r.fills);
  std::printf("Cash=%.6f Realized=%.6f Equity=%.6f\n", r.acct.cash, r.acct.realized, r.acct.equity);
  return 0;
}


