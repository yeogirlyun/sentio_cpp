#include "../include/sentio/strategy_sma_cross.hpp"
#include "../include/sentio/signal_engine.hpp"
#include "../include/sentio/rth_calendar.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace sentio;

// Example integration showing how to use the diagnostic system
// with existing QQQ data to surface exactly why signals are dropped

struct QQQBar {
  double open, high, low, close;
  std::int64_t ts_utc;
  double volume;
};

// Convert QQQ data to our diagnostic format
std::vector<QQQBar> load_qqq_sample() {
  std::vector<QQQBar> bars;
  
  // Sample QQQ data (rising trend to trigger golden cross)
  std::vector<double> prices = {
    100.0, 100.5, 101.0, 101.2, 101.5, 102.0, 102.3, 102.8, 103.0, 103.5,
    104.0, 104.2, 104.8, 105.0, 105.5, 106.0, 106.3, 106.8, 107.0, 107.5,
    108.0, 108.2, 108.8, 109.0, 109.5, 110.0, 110.3, 110.8, 111.0, 111.5
  };
  
  for (size_t i = 0; i < prices.size(); ++i) {
    double price = prices[i];
    bars.push_back({
      .open = price - 0.1,
      .high = price + 0.2,
      .low = price - 0.2,
      .close = price,
      .ts_utc = static_cast<std::int64_t>(1000000 + i * 60), // 1-minute bars
      .volume = static_cast<double>(1000000 + i * 10000)
    });
  }
  
  return bars;
}

void print_detailed_health(const SignalHealth& health) {
  std::cout << "\n🔍 DETAILED SIGNAL HEALTH ANALYSIS\n";
  std::cout << "===================================\n";
  std::cout << "Total Emitted: " << health.emitted.load() << "\n";
  std::cout << "Total Dropped: " << health.dropped.load() << "\n";
  
  if (health.dropped.load() > 0) {
    std::cout << "\nDrop Reason Breakdown:\n";
    std::cout << "---------------------\n";
    
    const char* reason_names[] = {
      "NONE", "NO_DATA", "NOT_RTH", "HOLIDAY", "WARMUP", 
      "NAN_INPUT", "THRESHOLD_TOO_TIGHT", "COOLDOWN_ACTIVE", 
      "DEBOUNCE", "DUPLICATE_BAR_TS", "UNKNOWN"
    };
    
    for (const auto& [reason, count] : health.by_reason) {
      if (count.load() > 0) {
        int idx = static_cast<int>(reason);
        if (idx >= 0 && idx < 11) {
          double percentage = (double)count.load() / health.dropped.load() * 100.0;
          std::cout << "  " << std::setw(20) << reason_names[idx] << ": " 
                    << std::setw(6) << count.load() 
                    << " (" << std::fixed << std::setprecision(1) << percentage << "%)\n";
        }
      }
    }
  }
}

void run_integration_test() {
  std::cout << "🚀 SIGNAL DIAGNOSTIC INTEGRATION TEST\n";
  std::cout << "=====================================\n\n";
  
  // Load sample data
  auto qqq_bars = load_qqq_sample();
  std::cout << "📊 Loaded " << qqq_bars.size() << " QQQ bars\n\n";
  
  // Test 1: Basic SMA Cross with permissive settings
  std::cout << "Test 1: SMA Cross with permissive settings\n";
  std::cout << "------------------------------------------\n";
  
  SMACrossCfg cfg{5, 15, 0.6}; // Fast=5, Slow=15, Conf=0.6
  SMACrossStrategy strat(cfg);
  SignalHealth health1;
  GateCfg gate1{.require_rth=false, .cooldown_bars=0, .min_conf=0.01};
  SignalEngine engine1(&strat, gate1, &health1);
  
  int signals_1 = 0;
  for (size_t i = 0; i < qqq_bars.size(); ++i) {
    const auto& bar = qqq_bars[i];
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=bar.ts_utc, .is_rth=true};
    Bar b{bar.open, bar.high, bar.low, bar.close};
    auto out = engine1.on_bar(ctx, b, true);
    
    if (out.signal) {
      signals_1++;
      std::cout << "  Bar " << std::setw(2) << i << ": " 
                << (out.signal->type == StrategySignal::Type::BUY ? "BUY" : "SELL")
                << " (conf: " << std::fixed << std::setprecision(2) << out.signal->confidence << ")\n";
    }
  }
  
  print_detailed_health(health1);
  std::cout << "\nResult: " << signals_1 << " signals emitted\n\n";
  
  // Test 2: RTH validation enabled
  std::cout << "Test 2: SMA Cross with RTH validation\n";
  std::cout << "-------------------------------------\n";
  
  SMACrossStrategy strat2(cfg);
  SignalHealth health2;
  GateCfg gate2{.require_rth=true, .cooldown_bars=0, .min_conf=0.01};
  SignalEngine engine2(&strat2, gate2, &health2);
  
  int signals_2 = 0;
  for (size_t i = 0; i < qqq_bars.size(); ++i) {
    const auto& bar = qqq_bars[i];
    // Simulate RTH validation (all bars are RTH in this example)
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=bar.ts_utc, .is_rth=true};
    Bar b{bar.open, bar.high, bar.low, bar.close};
    auto out = engine2.on_bar(ctx, b, true);
    
    if (out.signal) {
      signals_2++;
      std::cout << "  Bar " << std::setw(2) << i << ": " 
                << (out.signal->type == StrategySignal::Type::BUY ? "BUY" : "SELL")
                << " (conf: " << std::fixed << std::setprecision(2) << out.signal->confidence << ")\n";
    }
  }
  
  print_detailed_health(health2);
  std::cout << "\nResult: " << signals_2 << " signals emitted\n\n";
  
  // Test 3: High confidence threshold
  std::cout << "Test 3: SMA Cross with high confidence threshold\n";
  std::cout << "------------------------------------------------\n";
  
  SMACrossStrategy strat3(cfg);
  SignalHealth health3;
  GateCfg gate3{.require_rth=false, .cooldown_bars=0, .min_conf=0.9}; // Very high threshold
  SignalEngine engine3(&strat3, gate3, &health3);
  
  int signals_3 = 0;
  for (size_t i = 0; i < qqq_bars.size(); ++i) {
    const auto& bar = qqq_bars[i];
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=bar.ts_utc, .is_rth=true};
    Bar b{bar.open, bar.high, bar.low, bar.close};
    auto out = engine3.on_bar(ctx, b, true);
    
    if (out.signal) {
      signals_3++;
      std::cout << "  Bar " << std::setw(2) << i << ": " 
                << (out.signal->type == StrategySignal::Type::BUY ? "BUY" : "SELL")
                << " (conf: " << std::fixed << std::setprecision(2) << out.signal->confidence << ")\n";
    }
  }
  
  print_detailed_health(health3);
  std::cout << "\nResult: " << signals_3 << " signals emitted\n\n";
  
  // Summary
  std::cout << "📋 INTEGRATION TEST SUMMARY\n";
  std::cout << "===========================\n";
  std::cout << "Permissive settings: " << signals_1 << " signals\n";
  std::cout << "RTH validation:     " << signals_2 << " signals\n";
  std::cout << "High threshold:     " << signals_3 << " signals\n\n";
  
  std::cout << "🎯 KEY INSIGHTS:\n";
  std::cout << "- The diagnostic system successfully identifies why signals are dropped\n";
  std::cout << "- NO_DATA drops indicate warmup periods or strategy logic issues\n";
  std::cout << "- THRESHOLD_TOO_TIGHT drops show when confidence thresholds are too high\n";
  std::cout << "- NOT_RTH drops occur when RTH validation is too strict\n";
  std::cout << "- This system can be integrated into existing strategies to debug signal issues\n";
}

int main() {
  run_integration_test();
  return 0;
}
