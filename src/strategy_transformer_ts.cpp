#include "sentio/strategy_transformer_ts.hpp"
#include <algorithm>

namespace sentio {

static ml::WindowSpec make_spec(const ml::ModelSpec& s){
  ml::WindowSpec w;
  w.seq_len = s.seq_len>0? s.seq_len : 64;
  w.layout  = s.input_layout.empty()? "BTF" : s.input_layout;
  w.feat_dim = (int)s.feature_names.size();
  w.mean = s.mean; w.std = s.std; w.clip2 = s.clip2;
  return w;
}

TransformerSignalStrategyTS::TransformerSignalStrategyTS(const TransformerTSCfg& cfg)
: cfg_(cfg)
, handle_(ml::ModelRegistryTS::load_torchscript(cfg.model_id, cfg.version, cfg.artifacts_dir, cfg.use_cuda))
, window_(make_spec(handle_.spec))
{}

void TransformerSignalStrategyTS::set_raw_features(const std::vector<double>& raw){
  window_.push(raw);
}

StrategySignal TransformerSignalStrategyTS::map_output(const ml::ModelOutput& mo) const {
  StrategySignal s;
  if (mo.probs.empty()) { s.type=StrategySignal::Type::HOLD; s.confidence=cfg_.conf_floor; return s; }
  int argmax = 0; for (int i=1;i<(int)mo.probs.size();++i) if (mo.probs[i]>mo.probs[argmax]) argmax=i;
  float pmax = mo.probs[argmax];
  // default order ["SELL","HOLD","BUY"]
  if      (argmax==2) s.type = StrategySignal::Type::BUY;
  else if (argmax==0) s.type = StrategySignal::Type::SELL;
  else                s.type = StrategySignal::Type::HOLD;
  s.confidence = std::max(cfg_.conf_floor, (double)pmax);
  return s;
}

void TransformerSignalStrategyTS::on_bar(const StrategyCtx& ctx, const Bar& b){
  (void)ctx; (void)b;
  last_.reset();
  if (!window_.ready()) return;

  auto in = window_.to_input();
  if (!in) return;

  auto out = handle_.model->predict(*in, window_.seq_len(), window_.feat_dim(), handle_.spec.input_layout);
  if (!out) return;

  auto sig = map_output(*out);
  if (sig.confidence < cfg_.conf_floor) return;
  last_ = sig;
}

} // namespace sentio
