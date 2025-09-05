#pragma once
#include "strategy_base.hpp"
#include "signal_gate.hpp"
#include "signal_trace.hpp"
#include <cstdint>
#include <string>
#include <optional>

namespace sentio {

// Forward declarations to avoid include conflicts
struct RouterCfg;
struct RouteDecision;
struct Order;
struct AccountSnapshot;
class PriceBook;

struct PipelineCfg {
  GateCfg gate;
  double min_order_shares{1.0};
};

struct PipelineOut {
  std::optional<StrategySignal> signal;
  TraceRow trace;
};

class SignalPipeline {
public:
  SignalPipeline(IStrategy* strat, const PipelineCfg& cfg, void* /*book*/, SignalTrace* trace)
  : strat_(strat), cfg_(cfg), trace_(trace), gate_(cfg.gate, nullptr) {}

  PipelineOut on_bar(const StrategyCtx& ctx, const Bar& b, const void* acct);
private:
  IStrategy* strat_;
  PipelineCfg cfg_;
  SignalTrace* trace_;
  SignalGate gate_;
};

} // namespace sentio
