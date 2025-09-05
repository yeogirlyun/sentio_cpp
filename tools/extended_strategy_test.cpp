#include "../include/sentio/strategy_market_making.hpp"
#include "../include/sentio/strategy_volatility_expansion.hpp"
#include "../include/sentio/core.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace sentio;

void test_strategy_extended(const std::string& strategy_name, BaseStrategy* strategy, const std::vector<double>& closes) {
  std::cout << "\n🔍 EXTENDED Testing " << strategy_name << " Strategy\n";
  std::cout << "==========================================\n";
  
  // Create a vector of bars
  std::vector<Bar> bars;
  for (int i = 0; i < (int)closes.size(); ++i) {
    Bar b;
    b.ts_utc = std::to_string(1'000'000 + i * 60);
    b.ts_nyt_epoch = 1'000'000 + i * 60;
    b.open = closes[i];
    b.high = closes[i] * 1.001;  // Slight high
    b.low = closes[i] * 0.999;   // Slight low
    b.close = closes[i];
    b.volume = 1000;
    bars.push_back(b);
  }
  
  int signals_emitted = 0;
  std::cout << "Extended bar-by-bar analysis (showing every 10th bar):\n";
  std::cout << "Bar | Price  | Signal | Confidence | Reason\n";
  std::cout << "----|--------|--------|------------|-------\n";
  
  for (int i = 0; i < (int)bars.size(); ++i) {
    // Calculate signal for this bar
    StrategySignal signal = strategy->calculate_signal(bars, i);
    
    // Show every 10th bar and any bars with signals
    if (i % 10 == 0 || signal.type != StrategySignal::Type::HOLD) {
      std::cout << std::setw(3) << i << " | " 
                << std::fixed << std::setprecision(2) << std::setw(6) << bars[i].close << " | ";
      
      if (signal.type != StrategySignal::Type::HOLD) {
        signals_emitted++;
        switch (signal.type) {
          case StrategySignal::Type::BUY: std::cout << "BUY    "; break;
          case StrategySignal::Type::SELL: std::cout << "SELL   "; break;
          case StrategySignal::Type::STRONG_BUY: std::cout << "S_BUY  "; break;
          case StrategySignal::Type::STRONG_SELL: std::cout << "S_SELL "; break;
          default: std::cout << "UNKNOWN"; break;
        }
        std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(10) << signal.confidence << " | SUCCESS";
      } else {
        std::cout << "NONE   | " << std::fixed << std::setprecision(3) << std::setw(10) << signal.confidence << " | DROPPED";
      }
      std::cout << "\n";
    }
  }
  
  std::cout << "\n📊 Summary for " << strategy_name << ":\n";
  std::cout << "  Total bars processed: " << bars.size() << "\n";
  std::cout << "  Signals emitted: " << signals_emitted << "\n";
  std::cout << "  Success rate: " << std::fixed << std::setprecision(2) 
            << (bars.size() > 0 ? (signals_emitted * 100.0 / bars.size()) : 0.0) << "%\n";
  
  // Get diagnostic info
  const SignalDiag& diag = strategy->get_diag();
  std::cout << "  Diagnostic info:\n";
  std::cout << "    Signals emitted: " << diag.emitted << "\n";
  std::cout << "    Signals dropped: " << diag.dropped << "\n";
  std::cout << "    Drop reasons:\n";
  std::cout << "      Min bars: " << diag.r_min_bars << "\n";
  std::cout << "      Session: " << diag.r_session << "\n";
  std::cout << "      NaN input: " << diag.r_nan << "\n";
  std::cout << "      Zero volume: " << diag.r_zero_vol << "\n";
  std::cout << "      Threshold: " << diag.r_threshold << "\n";
  std::cout << "      Cooldown: " << diag.r_cooldown << "\n";
  std::cout << "      Duplicate: " << diag.r_dup << "\n";
}

int main() {
  std::cout << "🔍 EXTENDED STRATEGY TEST\n";
  std::cout << "=========================\n";
  std::cout << "Testing strategies with extended data to satisfy warmup requirements\n\n";
  
  // Create extended test data with more volatility
  std::vector<double> closes;
  double base_price = 100.0;
  srand(42); // For reproducible results
  
  // Generate 200 bars to ensure warmup requirements are met
  for (int i = 0; i < 200; ++i) {
    // Add some realistic price movement with occasional spikes
    double trend = i * 0.05;  // Gentle upward trend
    double noise = (rand() % 100 - 50) * 0.01;  // Random noise
    double spike = (i % 30 == 0) ? (rand() % 100 - 50) * 0.05 : 0;  // Occasional spikes
    closes.push_back(base_price + trend + noise + spike);
  }
  
  std::cout << "📈 Test data: " << closes.size() << " bars, price range: " 
            << *std::min_element(closes.begin(), closes.end()) << " - "
            << *std::max_element(closes.begin(), closes.end()) << "\n\n";
  
  // Test Market Making Strategy
  std::cout << "Creating Market Making Strategy...\n";
  MarketMakingStrategy mm_strategy;
  test_strategy_extended("Market Making", &mm_strategy, closes);
  
  // Test Volatility Expansion Strategy  
  std::cout << "\nCreating Volatility Expansion Strategy...\n";
  VolatilityExpansionStrategy ve_strategy;
  test_strategy_extended("Volatility Expansion", &ve_strategy, closes);
  
  std::cout << "\n🎯 EXTENDED TEST COMPLETE\n";
  std::cout << "==========================\n";
  std::cout << "This test uses 200 bars to ensure warmup requirements are satisfied.\n";
  std::cout << "If strategies still don't generate signals, the thresholds are too restrictive.\n";
  
  return 0;
}
