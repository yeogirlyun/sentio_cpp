#include "sentio/signal_engine.hpp"

namespace sentio {

SignalEngine::SignalEngine(IStrategy* strat, const GateCfg& gate_cfg, SignalHealth* health)
: strat_(strat), gate_(gate_cfg, health), health_(health) {}

EngineOut SignalEngine::on_bar(const StrategyCtx& ctx, const Bar& b, bool inputs_finite) {
  strat_->on_bar(ctx, b);
  auto raw = strat_->latest();
  if (!raw.has_value()) {
    if (health_) health_->incr_drop(DropReason::NO_DATA);
    return {std::nullopt, DropReason::NO_DATA};
  }

  double conf = raw->confidence;
  auto conf2 = gate_.accept(ctx.ts_utc_epoch, ctx.is_rth, inputs_finite,
                            /*warmed_up=*/true, conf);
  if (!conf2) {
    // SignalGate already tallied reason; we return UNKNOWN to avoid double counting specific reason here.
    return {std::nullopt, DropReason::UNKNOWN};
  }

  StrategySignal out = *raw;
  out.confidence = *conf2;
  return {out, DropReason::NONE};
}

} // namespace sentio
