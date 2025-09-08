#include "sentio/strategy_kochi_ppo.hpp"
#include <algorithm>
#include <stdexcept>

namespace sentio {

static ml::WindowSpec make_kochi_spec(const ml::ModelSpec& s){
  ml::WindowSpec w;
  // Kochi environment defaults to 20 window; allow metadata override if provided
  w.seq_len = s.seq_len > 0 ? s.seq_len : 20;
  w.layout  = s.input_layout.empty()? "BTF" : s.input_layout;
  w.feat_dim = (int)s.feature_names.size();
  w.mean = s.mean;
  w.std  = s.std;
  w.clip2 = s.clip2;
  return w;
}

KochiPPOStrategy::KochiPPOStrategy()
: BaseStrategy("KochiPPO")
, cfg_()
, handle_(ml::ModelRegistryTS::load_torchscript(cfg_.model_id, cfg_.version, cfg_.artifacts_dir, cfg_.use_cuda))
, window_(make_kochi_spec(handle_.spec))
{}

KochiPPOStrategy::KochiPPOStrategy(const KochiPPOCfg& cfg)
: BaseStrategy("KochiPPO")
, cfg_(cfg)
, handle_(ml::ModelRegistryTS::load_torchscript(cfg.model_id, cfg.version, cfg.artifacts_dir, cfg.use_cuda))
, window_(make_kochi_spec(handle_.spec))
{}

void KochiPPOStrategy::set_raw_features(const std::vector<double>& raw){
  // Expect exactly F features in model metadata order
  if ((int)raw.size() != window_.feat_dim()) return;
  window_.push(raw);
}

ParameterMap KochiPPOStrategy::get_default_params() const {
  return {
    {"conf_floor", cfg_.conf_floor}
  };
}

ParameterSpace KochiPPOStrategy::get_param_space() const {
  return {
    {"conf_floor", {ParamType::FLOAT, 0.0, 1.0, cfg_.conf_floor}}
  };
}

void KochiPPOStrategy::apply_params() {
  cfg_.conf_floor = params_["conf_floor"];
}

StrategySignal KochiPPOStrategy::map_output(const ml::ModelOutput& mo) const {
  StrategySignal s;
  // Assume discrete probs with actions in spec.actions. Default mapping SELL/HOLD/BUY
  int argmax = 0;
  for (int i=1;i<(int)mo.probs.size();++i) if (mo.probs[i] > mo.probs[argmax]) argmax = i;

  const auto& acts = handle_.spec.actions;
  std::string a = (argmax<(int)acts.size()? acts[argmax] : "HOLD");
  float pmax = mo.probs.empty()? 0.0f : mo.probs[argmax];

  if (a=="BUY")       s.type = StrategySignal::Type::BUY;
  else if (a=="SELL") s.type = StrategySignal::Type::SELL;
  else                 s.type = StrategySignal::Type::HOLD;

  s.confidence = std::max(cfg_.conf_floor, (double)pmax);
  return s;
}

StrategySignal KochiPPOStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
  (void)bars; (void)current_index; // features are streamed in via set_raw_features
  last_.reset();
  if (!window_.ready()) return StrategySignal{};

  auto in = window_.to_input();
  if (!in) return StrategySignal{};

  auto out = handle_.model->predict(*in, window_.seq_len(), window_.feat_dim(), handle_.spec.input_layout);
  if (!out) return StrategySignal{};

  auto sig = map_output(*out);
  if (sig.confidence < cfg_.conf_floor) return StrategySignal{};
  last_ = sig;
  return sig;
}

// Register with factory
REGISTER_STRATEGY(KochiPPOStrategy, "kochi_ppo");

} // namespace sentio


