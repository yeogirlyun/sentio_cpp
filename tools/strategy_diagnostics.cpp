#include "../include/sentio/strategy_market_making.hpp"
#include "../include/sentio/signal_pipeline.hpp"
#include "../include/sentio/feature_health.hpp"
#include "../include/sentio/core.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <unordered_map>

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

void print_strategy_summary(const std::string& strategy_name, const SignalTrace& trace) {
  std::cout << "\n📊 " << strategy_name << " DIAGNOSTIC RESULTS\n";
  std::cout << "==========================================\n";
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
  std::cout << "UNKNOWN: " << trace.count(TraceReason::UNKNOWN) << "\n";
  
  double success_rate = trace.rows().size() > 0 ? 
    (trace.count(TraceReason::OK) * 100.0 / trace.rows().size()) : 0.0;
  std::cout << "Success rate: " << std::fixed << std::setprecision(2) << success_rate << "%\n";
}

void print_detailed_trace(const SignalTrace& trace, int limit = 20) {
  std::cout << "\n📋 DETAILED TRACE (Last " << limit << " bars)\n";
  std::cout << "============================================\n";
  std::cout << "ts_utc,instrument,close,is_rth,warmed,inputs_finite,confidence,conf_after_gate,reason\n";
  
  int start = std::max(0, (int)trace.rows().size() - limit);
  for (int i = start; i < (int)trace.rows().size(); ++i) {
    const auto& row = trace.rows()[i];
    std::cout << row.ts_utc << ","
              << row.instrument << ","
              << std::fixed << std::setprecision(2) << row.close << ","
              << (row.is_rth ? "true" : "false") << ","
              << (row.warmed ? "true" : "false") << ","
              << (row.inputs_finite ? "true" : "false") << ","
              << std::fixed << std::setprecision(3) << row.confidence << ","
              << row.conf_after_gate << ","
              << reason_to_string(row.reason) << "\n";
  }
}

void test_strategy(const std::string& strategy_name, BaseStrategy* strategy, const std::vector<double>& closes) {
  std::cout << "\n🔍 Testing " << strategy_name << " Strategy\n";
  std::cout << "=====================================\n";
  
  PB book;
  SignalTrace trace;
  PipelineCfg pcfg;
  pcfg.gate = GateCfg{true, 0, 0.01};  // Very permissive gate
  pcfg.min_order_shares = 1.0;

  SignalPipeline pipe(strategy, pcfg, &book, &trace);
  struct SimpleAccount { double equity=100000; double cash=100000; };
  SimpleAccount acct;

  int signals_emitted = 0;
  for (int i = 0; i < (int)closes.size(); ++i) {
    // Create Bar with proper structure
    Bar b;
    b.ts_utc = std::to_string(1'000'000 + i * 60);
    b.ts_nyt_epoch = 1'000'000 + i * 60;
    b.open = closes[i];
    b.high = closes[i] * 1.001;  // Slight high
    b.low = closes[i] * 0.999;   // Slight low
    b.close = closes[i];
    b.volume = 1000;
    
    // Update price book
    book.upsert_latest("QQQ", b);
    
    // Create TQQQ bar
    Bar tqqq_bar = b;
    tqqq_bar.open *= 3;
    tqqq_bar.high *= 3;
    tqqq_bar.low *= 3;
    tqqq_bar.close *= 3;
    book.upsert_latest("TQQQ", tqqq_bar);
    
    // Test strategy directly instead of through pipeline
    strategy->on_bar(b);
    auto signal = strategy->get_latest_signal();
    
    if (signal.has_value()) {
      signals_emitted++;
      std::cout << "  ✅ Signal at bar " << i << ": " 
                << (signal->type == StrategySignal::Type::BUY ? "BUY" : 
                    signal->type == StrategySignal::Type::SELL ? "SELL" : "HOLD")
                << " (conf: " << std::fixed << std::setprecision(3) << signal->confidence << ")\n";
    }
  }
  
  std::cout << "\nSignals emitted: " << signals_emitted << "\n";
  print_strategy_summary(strategy_name, trace);
  print_detailed_trace(trace, 15);
}

int main() {
  std::cout << "🔍 STRATEGY DIAGNOSTIC TOOL\n";
  std::cout << "===========================\n";
  std::cout << "Testing problematic strategies with detailed tracing\n\n";
  
  // Create realistic test data with some volatility
  std::vector<double> closes;
  double base_price = 100.0;
  for (int i = 0; i < 100; ++i) {
    // Add some realistic price movement with occasional spikes
    double trend = i * 0.1;  // Slight upward trend
    double noise = (rand() % 100 - 50) * 0.01;  // Random noise
    double spike = (i % 20 == 0) ? (rand() % 100 - 50) * 0.05 : 0;  // Occasional spikes
    closes.push_back(base_price + trend + noise + spike);
  }
  
  std::cout << "📈 Test data: " << closes.size() << " bars, price range: " 
            << *std::min_element(closes.begin(), closes.end()) << " - "
            << *std::max_element(closes.begin(), closes.end()) << "\n\n";
  
  // Test Market Making Strategy
  MarketMakingStrategy mm_strategy;
  test_strategy("Market Making", &mm_strategy, closes);
  
  std::cout << "\n🎯 DIAGNOSTIC COMPLETE\n";
  std::cout << "======================\n";
  std::cout << "Check the detailed traces above to identify why each strategy\n";
  std::cout << "is not generating signals. Look for patterns in the reason codes.\n";
  
  return 0;
}
