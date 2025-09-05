#include "../include/sentio/strategy_market_making.hpp"
#include "../include/sentio/strategy_volatility_expansion.hpp"
#include "../include/sentio/core.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace sentio;

void test_strategy_detailed(const std::string& strategy_name, BaseStrategy* strategy, const std::vector<double>& closes) {
  std::cout << "\n🔍 DETAILED Testing " << strategy_name << " Strategy\n";
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
  std::cout << "Detailed bar-by-bar analysis:\n";
  std::cout << "Bar | Price  | Signal | Confidence | Reason\n";
  std::cout << "----|--------|--------|------------|-------\n";
  
  for (int i = 0; i < (int)bars.size(); ++i) {
    // Calculate signal for this bar
    StrategySignal signal = strategy->calculate_signal(bars, i);
    
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
    
    // Show first 10 bars and every 10th bar after that
    if (i >= 10 && i % 10 != 0) continue;
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
  std::cout << "🔍 DETAILED STRATEGY DIAGNOSTIC TOOL\n";
  std::cout << "====================================\n";
  std::cout << "Testing problematic strategies with detailed bar-by-bar analysis\n\n";
  
  // Create realistic test data with some volatility
  std::vector<double> closes;
  double base_price = 100.0;
  srand(42); // For reproducible results
  
  for (int i = 0; i < 50; ++i) { // Smaller dataset for detailed analysis
    // Add some realistic price movement with occasional spikes
    double trend = i * 0.2;  // Stronger upward trend
    double noise = (rand() % 100 - 50) * 0.02;  // More noise
    double spike = (i % 10 == 0) ? (rand() % 100 - 50) * 0.1 : 0;  // More frequent spikes
    closes.push_back(base_price + trend + noise + spike);
  }
  
  std::cout << "📈 Test data: " << closes.size() << " bars, price range: " 
            << *std::min_element(closes.begin(), closes.end()) << " - "
            << *std::max_element(closes.begin(), closes.end()) << "\n\n";
  
  // Test Market Making Strategy
  std::cout << "Creating Market Making Strategy...\n";
  MarketMakingStrategy mm_strategy;
  test_strategy_detailed("Market Making", &mm_strategy, closes);
  
  // Test Volatility Expansion Strategy  
  std::cout << "\nCreating Volatility Expansion Strategy...\n";
  VolatilityExpansionStrategy ve_strategy;
  test_strategy_detailed("Volatility Expansion", &ve_strategy, closes);
  
  std::cout << "\n🎯 DETAILED DIAGNOSTIC COMPLETE\n";
  std::cout << "================================\n";
  std::cout << "Check the detailed bar-by-bar analysis above to see exactly\n";
  std::cout << "what's happening in each bar and why signals are being dropped.\n";
  
  return 0;
}
