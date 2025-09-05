#pragma once
#include "strategy_base.hpp"
#include "signal_gate.hpp"

namespace sentio {

struct EngineOut {
  std::optional<StrategySignal> signal; // post-gate
  DropReason last_drop{DropReason::NONE};
};

class SignalEngine {
public:
  SignalEngine(IStrategy* strat, const GateCfg& gate_cfg, SignalHealth* health);
  EngineOut on_bar(const StrategyCtx& ctx, const Bar& b, bool inputs_finite=true);
private:
  IStrategy* strat_;
  SignalGate gate_;
  SignalHealth* health_;
};

} // namespace sentio
