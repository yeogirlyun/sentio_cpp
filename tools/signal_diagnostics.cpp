#include "../include/sentio/strategy_sma_cross.hpp"
#include "../include/sentio/signal_engine.hpp"
#include "../include/sentio/rth_calendar.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace sentio;

struct BarData {
  double open, high, low, close;
  std::int64_t ts_utc;
  bool is_rth;
};

void print_health_summary(const SignalHealth& health) {
  std::cout << "\n📊 SIGNAL HEALTH SUMMARY\n";
  std::cout << "========================\n";
  std::cout << "Emitted: " << health.emitted.load() << "\n";
  std::cout << "Dropped: " << health.dropped.load() << "\n";
  std::cout << "\nDrop Reasons:\n";
  
  const char* reason_names[] = {
    "NONE", "NO_DATA", "NOT_RTH", "HOLIDAY", "WARMUP", 
    "NAN_INPUT", "THRESHOLD_TOO_TIGHT", "COOLDOWN_ACTIVE", 
    "DEBOUNCE", "DUPLICATE_BAR_TS", "UNKNOWN"
  };
  
  for (const auto& [reason, count] : health.by_reason) {
    if (count.load() > 0) {
      int idx = static_cast<int>(reason);
      if (idx >= 0 && idx < 11) {
        std::cout << "  " << reason_names[idx] << ": " << count.load() << "\n";
      }
    }
  }
}

void run_diagnostic_test() {
  std::cout << "🔍 SIGNAL DIAGNOSTIC TOOL\n";
  std::cout << "==========================\n\n";
  
  // Test 1: Basic SMA Cross with rising data
  std::cout << "Test 1: Rising data (should generate BUY signals)\n";
  std::cout << "------------------------------------------------\n";
  
  SMACrossCfg cfg{5, 10, 0.8};
  SMACrossStrategy strat(cfg);
  SignalHealth health1;
  GateCfg gate{.require_rth=false, .cooldown_bars=0, .min_conf=0.01};
  SignalEngine engine1(&strat, gate, &health1);

  std::vector<BarData> rising_data;
  for (int i = 0; i < 20; ++i) {
    double price = 100.0 + i * 0.5;
    rising_data.push_back({price, price, price, price, 1000000 + i * 60, true});
  }

  int signals_1 = 0;
  for (size_t i = 0; i < rising_data.size(); ++i) {
    const auto& bar = rising_data[i];
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=bar.ts_utc, .is_rth=bar.is_rth};
    Bar b{bar.open, bar.high, bar.low, bar.close};
    auto out = engine1.on_bar(ctx, b, true);
    
    if (out.signal) {
      signals_1++;
      std::cout << "  Bar " << std::setw(2) << i << ": " 
                << (out.signal->type == StrategySignal::Type::BUY ? "BUY" : "SELL")
                << " (conf: " << std::fixed << std::setprecision(2) << out.signal->confidence << ")\n";
    }
  }
  
  print_health_summary(health1);
  std::cout << "\nResult: " << signals_1 << " signals emitted\n\n";

  // Test 2: RTH validation test
  std::cout << "Test 2: RTH validation (should drop non-RTH bars)\n";
  std::cout << "------------------------------------------------\n";
  
  SMACrossStrategy strat2(cfg);
  SignalHealth health2;
  GateCfg gate_rth{.require_rth=true, .cooldown_bars=0, .min_conf=0.01};
  SignalEngine engine2(&strat2, gate_rth, &health2);

  std::vector<BarData> mixed_data;
  for (int i = 0; i < 15; ++i) {
    double price = 100.0 + i * 0.3;
    // Mix RTH and non-RTH bars
    bool is_rth = (i % 3 != 0); // Every 3rd bar is non-RTH
    mixed_data.push_back({price, price, price, price, 1000000 + i * 60, is_rth});
  }

  int signals_2 = 0;
  for (size_t i = 0; i < mixed_data.size(); ++i) {
    const auto& bar = mixed_data[i];
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=bar.ts_utc, .is_rth=bar.is_rth};
    Bar b{bar.open, bar.high, bar.low, bar.close};
    auto out = engine2.on_bar(ctx, b, true);
    
    if (out.signal) {
      signals_2++;
      std::cout << "  Bar " << std::setw(2) << i << " (RTH:" << (bar.is_rth ? "Y" : "N") << "): " 
                << (out.signal->type == StrategySignal::Type::BUY ? "BUY" : "SELL") << "\n";
    } else {
      std::cout << "  Bar " << std::setw(2) << i << " (RTH:" << (bar.is_rth ? "Y" : "N") << "): DROPPED\n";
    }
  }
  
  print_health_summary(health2);
  std::cout << "\nResult: " << signals_2 << " signals emitted\n\n";

  // Test 3: Threshold test
  std::cout << "Test 3: High threshold (should drop low confidence signals)\n";
  std::cout << "--------------------------------------------------------\n";
  
  SMACrossStrategy strat3(cfg);
  SignalHealth health3;
  GateCfg gate_high{.require_rth=false, .cooldown_bars=0, .min_conf=0.9}; // Very high threshold
  SignalEngine engine3(&strat3, gate_high, &health3);

  int signals_3 = 0;
  for (size_t i = 0; i < rising_data.size(); ++i) {
    const auto& bar = rising_data[i];
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=bar.ts_utc, .is_rth=bar.is_rth};
    Bar b{bar.open, bar.high, bar.low, bar.close};
    auto out = engine3.on_bar(ctx, b, true);
    
    if (out.signal) {
      signals_3++;
      std::cout << "  Bar " << std::setw(2) << i << ": " 
                << (out.signal->type == StrategySignal::Type::BUY ? "BUY" : "SELL")
                << " (conf: " << std::fixed << std::setprecision(2) << out.signal->confidence << ")\n";
    }
  }
  
  print_health_summary(health3);
  std::cout << "\nResult: " << signals_3 << " signals emitted\n\n";

  std::cout << "🎯 DIAGNOSTIC COMPLETE\n";
  std::cout << "======================\n";
  std::cout << "If you see 0 signals in any test, check the drop reasons above.\n";
  std::cout << "Common issues:\n";
  std::cout << "  - WARMUP: Not enough data for indicators\n";
  std::cout << "  - NOT_RTH: RTH validation too strict\n";
  std::cout << "  - THRESHOLD_TOO_TIGHT: Confidence threshold too high\n";
  std::cout << "  - NAN_INPUT: Invalid data in bars\n";
}

int main() {
  run_diagnostic_test();
  return 0;
}
