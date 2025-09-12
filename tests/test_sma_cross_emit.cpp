#include "../include/sentio/strategy_sma_cross.hpp"
#include "../include/sentio/signal_engine.hpp"
#include <cassert>
#include <vector>
#include <iostream>

using namespace sentio;

int main() {
  std::cout << "🧪 Testing SMA Cross Strategy Signal Emission\n";
  
  // Rising series should eventually trigger a BUY (golden cross)
  SMACrossCfg cfg{5, 10, 0.8};
  SMACrossStrategy strat(cfg);
  SignalHealth health;
  GateCfg gate{.cooldown_bars=0, .min_conf=0.01};
  SignalEngine engine(&strat, gate, &health);

  std::vector<double> closes;
  for (int i=0;i<25;++i) closes.push_back(100.0 + i*0.5);

  std::optional<StrategySignal> last;
  int emits=0;
  for (int i=0;i<(int)closes.size();++i) {
    StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=1'000'000+i*60, .is_rth=true};
    Bar b{"2024-01-01T09:30:00Z", 1000000+i*60, closes[i], closes[i], closes[i], closes[i], 1000};
    auto out = engine.on_bar(ctx, b, /*inputs_finite=*/true);
    if (out.signal) { 
      last = out.signal; 
      ++emits; 
      std::cout << "📈 Signal emitted at bar " << i << ": " 
                << (out.signal->type == StrategySignal::Type::BUY ? "BUY" : "SELL")
                << " (conf: " << out.signal->confidence << ")\n";
    }
  }

  std::cout << "📊 Final Results:\n";
  std::cout << "  Signals emitted: " << emits << "\n";
  std::cout << "  Signals dropped: " << health.dropped.load() << "\n";
  std::cout << "  Health by reason:\n";
  for (const auto& [reason, count] : health.by_reason) {
    if (count.load() > 0) {
      std::cout << "    " << static_cast<int>(reason) << ": " << count.load() << "\n";
    }
  }

  assert(emits >= 1);
  assert(last && (last->type == StrategySignal::Type::BUY));
  
  std::cout << "✅ Test passed! SMA Cross strategy successfully emitted signals.\n";
  return 0;
}
