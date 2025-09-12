#include "sentio/signal_gate.hpp"
#include <cmath>
#include <algorithm>

namespace sentio {

SignalHealth::SignalHealth() {
  for (auto r :
       {DropReason::NONE, DropReason::WARMUP,
        DropReason::NAN_INPUT, DropReason::THRESHOLD_TOO_TIGHT, DropReason::COOLDOWN_ACTIVE,
        DropReason::DUPLICATE_BAR_TS}) {
    by_reason.emplace(r, 0ULL);
  }
}
void SignalHealth::incr_emit(){ emitted.fetch_add(1, std::memory_order_relaxed); }
void SignalHealth::incr_drop(DropReason r){
  dropped.fetch_add(1, std::memory_order_relaxed);
  auto it = by_reason.find(r);
  if (it != by_reason.end()) it->second.fetch_add(1, std::memory_order_relaxed);
}

SignalGate::SignalGate(const GateCfg& cfg, SignalHealth* health)
: cfg_(cfg), health_(health) {}

std::optional<double> SignalGate::accept(std::int64_t ts_utc_epoch,
                                         bool inputs_finite,
                                         bool warmed_up,
                                         double conf)
{
  // RTH filtering removed - accepting all trading hours data
  if (!inputs_finite)              { if (health_) health_->incr_drop(DropReason::NAN_INPUT); return std::nullopt; }
  if (!warmed_up)                  { if (health_) health_->incr_drop(DropReason::WARMUP);    return std::nullopt; }
  if (!(std::isfinite(conf)))      { if (health_) health_->incr_drop(DropReason::NAN_INPUT); return std::nullopt; }

  conf = std::clamp(conf, 0.0, 1.0);

  if (conf < cfg_.min_conf)        { if (health_) health_->incr_drop(DropReason::THRESHOLD_TOO_TIGHT); return std::nullopt; }

  // Cooldown (optional)
  if (cooldown_left_ > 0)          { --cooldown_left_; if (health_) health_->incr_drop(DropReason::COOLDOWN_ACTIVE); return std::nullopt; }

  // Debounce duplicate timestamps
  if (last_emit_ts_ == ts_utc_epoch){ if (health_) health_->incr_drop(DropReason::DUPLICATE_BAR_TS); return std::nullopt; }

  last_emit_ts_ = ts_utc_epoch;
  cooldown_left_ = cfg_.cooldown_bars;
  if (health_) health_->incr_emit();
  return conf;
}

} // namespace sentio
