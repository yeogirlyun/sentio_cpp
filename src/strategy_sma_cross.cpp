#include "sentio/strategy_sma_cross.hpp"
#include <cmath>

namespace sentio {

SMACrossStrategy::SMACrossStrategy(const SMACrossCfg& cfg)
  : cfg_(cfg), sma_fast_(cfg.fast), sma_slow_(cfg.slow) {}

void SMACrossStrategy::on_bar(const StrategyCtx& ctx, const Bar& b) {
  (void)ctx;
  sma_fast_.push(b.close);
  sma_slow_.push(b.close);

  if (!warmed_up()) { last_.reset(); return; }

  double f = sma_fast_.value();
  double s = sma_slow_.value();
  if (!std::isfinite(f) || !std::isfinite(s)) { last_.reset(); return; }

  // Detect cross (including equality toggle avoidance)
  bool golden_now  = (f >= s) && !(std::isfinite(last_fast_) && std::isfinite(last_slow_) && last_fast_ >= last_slow_);
  bool death_now   = (f <= s) && !(std::isfinite(last_fast_) && std::isfinite(last_slow_) && last_fast_ <= last_slow_);

  if (golden_now) {
    last_ = StrategySignal{StrategySignal::Type::BUY, cfg_.conf_fast_slow};
  } else if (death_now) {
    last_ = StrategySignal{StrategySignal::Type::SELL, cfg_.conf_fast_slow};
  } else {
    last_.reset(); // no edge this bar
  }

  last_fast_ = f; last_slow_ = s;
}

} // namespace sentio
