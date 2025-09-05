#include "sentio/strategy_hybrid_ppo.hpp"
#include <algorithm>
#include <stdexcept>

namespace sentio {

HybridPPOStrategy::HybridPPOStrategy()
: BaseStrategy("HybridPPO"),
  cfg_(),
  handle_(ml::ModelRegistry::load_onnx("HybridPPO", cfg_.version, cfg_.artifacts_dir)),
  fpipe_(handle_.spec)
{}

HybridPPOStrategy::HybridPPOStrategy(const HybridPPOCfg& cfg)
: BaseStrategy("HybridPPO"),
  cfg_(cfg),
  handle_(ml::ModelRegistry::load_onnx("HybridPPO", cfg.version, cfg.artifacts_dir)),
  fpipe_(handle_.spec)
{}

void HybridPPOStrategy::set_raw_features(const std::vector<double>& raw) {
  raw_ = raw;
}

ParameterMap HybridPPOStrategy::get_default_params() const {
  return {
    {"conf_floor", cfg_.conf_floor}
  };
}

ParameterSpace HybridPPOStrategy::get_param_space() const {
  return {
    {"conf_floor", {ParamType::FLOAT, 0.0, 1.0, cfg_.conf_floor}}
  };
}

void HybridPPOStrategy::apply_params() {
  cfg_.conf_floor = params_["conf_floor"];
}

StrategySignal HybridPPOStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
  (void)bars; (void)current_index; // Features come from set_raw_features
  last_.reset();
  auto z = fpipe_.transform(raw_);
  if (!z) return StrategySignal{};

  auto out = handle_.model->predict(*z);
  if (!out) return StrategySignal{};

  StrategySignal sig = map_output(*out);
  // Safety: never emit if below floor
  if (sig.confidence < cfg_.conf_floor) return StrategySignal{};

  last_ = sig;
  return sig;
}

StrategySignal HybridPPOStrategy::map_output(const ml::ModelOutput& mo) const {
  StrategySignal s;
  // Discrete 3-way: SELL/HOLD/BUY
  int argmax = 0;
  for (int i=1;i<(int)mo.probs.size();++i) if (mo.probs[i] > mo.probs[argmax]) argmax = i;

  const auto& acts = handle_.spec.actions; // e.g., ["SELL","HOLD","BUY"]
  std::string a = (argmax<(int)acts.size()? acts[argmax] : "HOLD");
  float pmax = mo.probs.empty()? 0.0f : mo.probs[argmax];

  if (a=="BUY")       s.type = StrategySignal::Type::BUY;
  else if (a=="SELL") s.type = StrategySignal::Type::SELL;
  else                s.type = StrategySignal::Type::HOLD;

  // Confidence: calibrated pmax (basic identity)
  s.confidence = std::max(cfg_.conf_floor, (double)pmax);
  return s;
}


// Register the strategy with the factory
REGISTER_STRATEGY(HybridPPOStrategy, "hybrid_ppo")

} // namespace sentio
