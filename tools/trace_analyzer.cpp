#include "../include/sentio/strategy_sma_cross.hpp"
#include "../include/sentio/signal_pipeline.hpp"
#include "../include/sentio/feature_health.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace sentio;

struct PB {
  void upsert_latest(const std::string& i, const Bar& b) { latest[i]=b; }
  const Bar* get_latest(const std::string& i) const { auto it=latest.find(i); return it==latest.end()?nullptr:&it->second; }
  bool has_instrument(const std::string& i) const { return latest.count(i)>0; }
  std::size_t size() const { return latest.size(); }
  std::unordered_map<std::string,Bar> latest;
};

const char* reason_to_string(TraceReason r) {
  switch (r) {
    case TraceReason::OK: return "OK";
    case TraceReason::NO_STRATEGY_OUTPUT: return "NO_STRATEGY_OUTPUT";
    case TraceReason::NOT_RTH: return "NOT_RTH";
    case TraceReason::HOLIDAY: return "HOLIDAY";
    case TraceReason::WARMUP: return "WARMUP";
    case TraceReason::NAN_INPUT: return "NAN_INPUT";
    case TraceReason::THRESHOLD_TOO_TIGHT: return "THRESHOLD_TOO_TIGHT";
    case TraceReason::COOLDOWN_ACTIVE: return "COOLDOWN_ACTIVE";
    case TraceReason::DUPLICATE_BAR_TS: return "DUPLICATE_BAR_TS";
    case TraceReason::EMPTY_PRICEBOOK: return "EMPTY_PRICEBOOK";
    case TraceReason::NO_PRICE_FOR_INSTRUMENT: return "NO_PRICE_FOR_INSTRUMENT";
    case TraceReason::ROUTER_REJECTED: return "ROUTER_REJECTED";
    case TraceReason::ORDER_QTY_LT_MIN: return "ORDER_QTY_LT_MIN";
    case TraceReason::UNKNOWN: return "UNKNOWN";
    default: return "UNKNOWN";
  }
}

void print_trace_summary(const SignalTrace& trace) {
  std::cout << "\n📊 TRACE SUMMARY\n";
  std::cout << "================\n";
  std::cout << "Total bars processed: " << trace.rows().size() << "\n";
  std::cout << "OK signals: " << trace.count(TraceReason::OK) << "\n";
  std::cout << "No strategy output: " << trace.count(TraceReason::NO_STRATEGY_OUTPUT) << "\n";
  std::cout << "Threshold too tight: " << trace.count(TraceReason::THRESHOLD_TOO_TIGHT) << "\n";
  std::cout << "WARMUP: " << trace.count(TraceReason::WARMUP) << "\n";
  std::cout << "NOT_RTH: " << trace.count(TraceReason::NOT_RTH) << "\n";
  std::cout << "NAN_INPUT: " << trace.count(TraceReason::NAN_INPUT) << "\n";
  std::cout << "COOLDOWN_ACTIVE: " << trace.count(TraceReason::COOLDOWN_ACTIVE) << "\n";
  std::cout << "DUPLICATE_BAR_TS: " << trace.count(TraceReason::DUPLICATE_BAR_TS) << "\n";
  std::cout << "EMPTY_PRICEBOOK: " << trace.count(TraceReason::EMPTY_PRICEBOOK) << "\n";
  std::cout << "NO_PRICE_FOR_INSTRUMENT: " << trace.count(TraceReason::NO_PRICE_FOR_INSTRUMENT) << "\n";
  std::cout << "ROUTER_REJECTED: " << trace.count(TraceReason::ROUTER_REJECTED) << "\n";
  std::cout << "ORDER_QTY_LT_MIN: " << trace.count(TraceReason::ORDER_QTY_LT_MIN) << "\n";
  std::cout << "UNKNOWN: " << trace.count(TraceReason::UNKNOWN) << "\n\n";
}

void print_csv_header() {
  std::cout << "ts_utc,instrument,routed,close,is_rth,warmed,inputs_finite,confidence,conf_after_gate,target_weight,last_px,order_qty,reason,note\n";
}

void print_trace_csv(const SignalTrace& trace, int limit = 50) {
  std::cout << "\n📋 LAST " << limit << " TRACE ROWS (CSV)\n";
  std::cout << "=====================================\n";
  print_csv_header();
  
  int start = std::max(0, (int)trace.rows().size() - limit);
  for (int i = start; i < (int)trace.rows().size(); ++i) {
    const auto& row = trace.rows()[i];
    std::cout << row.ts_utc << ","
              << row.instrument << ","
              << row.routed << ","
              << std::fixed << std::setprecision(2) << row.close << ","
              << (row.is_rth ? "true" : "false") << ","
              << (row.warmed ? "true" : "false") << ","
              << (row.inputs_finite ? "true" : "false") << ","
              << std::fixed << std::setprecision(3) << row.confidence << ","
              << row.conf_after_gate << ","
              << row.target_weight << ","
              << row.last_px << ","
              << row.order_qty << ","
              << reason_to_string(row.reason) << ","
              << row.note << "\n";
  }
}

void run_diagnostic_test() {
  std::cout << "🔍 SIGNAL PIPELINE DIAGNOSTIC TOOL\n";
  std::cout << "==================================\n\n";
  
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

  // Test with rising data to trigger signals
  std::vector<double> closes;
  for (int i=0;i<50;++i) closes.push_back(100+i*0.3);

  int emits=0;
  for (int i=0;i<(int)closes.size();++i) {
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=1'000'000+i*60, .is_rth=true};
    Bar b{closes[i],closes[i],closes[i],closes[i]};
    
    // Update price book for both QQQ and TQQQ
    book.upsert_latest("QQQ", b);
    book.upsert_latest("TQQQ", Bar{b.open*3,b.high*3,b.low*3,b.close*3});
    
    auto out = pipe.on_bar(ctx,b,&acct);
    if (out.signal) ++emits;
  }

  print_trace_summary(trace);
  print_trace_csv(trace, 20);
  
  std::cout << "\n🎯 DIAGNOSTIC COMPLETE\n";
  std::cout << "======================\n";
  std::cout << "Orders emitted: " << emits << "\n";
  std::cout << "Success rate: " << (trace.count(TraceReason::OK) * 100.0 / trace.rows().size()) << "%\n";
}

int main() {
  run_diagnostic_test();
  return 0;
}
