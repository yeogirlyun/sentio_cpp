#include "../include/sentio/strategy_sma_cross.hpp"
#include "../include/sentio/signal_pipeline.hpp"
#include <cassert>
#include <vector>
#include <iostream>
#include <unordered_map>

using namespace sentio;

struct PB {
  void upsert_latest(const std::string& i, const Bar& b) { latest[i]=b; }
  const Bar* get_latest(const std::string& i) const { auto it=latest.find(i); return it==latest.end()?nullptr:&it->second; }
  bool has_instrument(const std::string& i) const { return latest.count(i)>0; }
  std::size_t size() const { return latest.size(); }
  std::unordered_map<std::string,Bar> latest;
};

int main() {
  std::cout << "🧪 Testing Signal Pipeline End-to-End\n";
  std::cout << "====================================\n\n";
  
  PB book;
  SMACrossCfg scfg{5, 10, 0.8};
  SMACrossStrategy strat(scfg);

  SignalTrace trace;
  PipelineCfg pcfg;
  pcfg.gate = GateCfg{true, 0, 0.01};
  pcfg.min_order_shares = 1.0;

  SignalPipeline pipe(&strat, pcfg, &book, &trace);
  // Simple account struct to avoid include conflicts
  struct SimpleAccount { double equity=100000; double cash=100000; };
  SimpleAccount acct;

  // Rising closes to trigger BUY cross; book price for routed instruments
  std::vector<double> closes;
  for (int i=0;i<30;++i) closes.push_back(100+i*0.5);

  int emits=0;
  for (int i=0;i<(int)closes.size();++i) {
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=1'000'000+i*60, .is_rth=true};
    Bar b{closes[i],closes[i],closes[i],closes[i]};
    // keep both QQQ and TQQQ book updated to avoid "no price"
    book.upsert_latest("QQQ", b);
    book.upsert_latest("TQQQ", Bar{b.open*3,b.high*3,b.low*3,b.close*3});
    auto out = pipe.on_bar(ctx,b,&acct);
    if (out.signal) ++emits;
  }

  std::cout << "📊 Pipeline Test Results:\n";
  std::cout << "  Signals emitted: " << emits << "\n";
  std::cout << "  Total trace rows: " << trace.rows().size() << "\n";
  std::cout << "  OK signals: " << trace.count(TraceReason::OK) << "\n";
  std::cout << "  No strategy output: " << trace.count(TraceReason::NO_STRATEGY_OUTPUT) << "\n";
  std::cout << "  Threshold too tight: " << trace.count(TraceReason::THRESHOLD_TOO_TIGHT) << "\n";
  std::cout << "  WARMUP: " << trace.count(TraceReason::WARMUP) << "\n";
  std::cout << "  NOT_RTH: " << trace.count(TraceReason::NOT_RTH) << "\n";
  std::cout << "  NAN_INPUT: " << trace.count(TraceReason::NAN_INPUT) << "\n";
  std::cout << "  Other reasons: " << trace.count(TraceReason::UNKNOWN) << "\n\n";

  assert(emits>=1);
  assert(trace.count(TraceReason::OK)>=1);
  
  std::cout << "✅ Test passed! Signal pipeline successfully emitted signals.\n";
  return 0;
}
