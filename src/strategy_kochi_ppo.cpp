#include "sentio/strategy_kochi_ppo.hpp"
#include "sentio/signal_utils.hpp"
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

double KochiPPOStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
  (void)bars; (void)current_index; // features are streamed in via set_raw_features
  last_.reset();
  if (!window_.ready()) return 0.5; // Neutral

  auto in = window_.to_input();
  if (!in) return 0.5; // Neutral

  auto out = handle_.model->predict(*in, window_.seq_len(), window_.feat_dim(), handle_.spec.input_layout);
  if (!out) return 0.5; // Neutral

  auto sig = map_output(*out);
  double probability = sentio::signal_utils::signal_to_probability(sig, cfg_.conf_floor);
  
  last_ = sig;
  return probability;
}

std::vector<BaseStrategy::AllocationDecision> KochiPPOStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // KochiPPO uses simple allocation based on signal strength
    if (probability > 0.6) {
        // Buy signal
        double conviction = (probability - 0.6) / 0.4; // Scale 0.6-1.0 to 0-1
        double base_weight = 0.3 + (conviction * 0.7); // 30-100% allocation
        
        decisions.push_back({base_symbol, base_weight, conviction, "KochiPPO buy: 100% QQQ"});
    } else if (probability < 0.4) {
        // Sell signal
        double conviction = (0.4 - probability) / 0.4; // Scale 0.0-0.4 to 0-1
        double base_weight = 0.3 + (conviction * 0.7); // 30-100% allocation
        
        // Use SHORT QQQ for moderate sell signals instead of PSQ
        decisions.push_back({base_symbol, -base_weight, conviction, "KochiPPO sell: SHORT QQQ"});
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "KochiPPO: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg KochiPPOStrategy::get_router_config() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}

SizerCfg KochiPPOStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.max_position_pct = 1.0; // 100% max position
    cfg.volatility_target = 0.15; // 15% volatility target
    return cfg;
}

// Register with factory
REGISTER_STRATEGY(KochiPPOStrategy, "kochi_ppo");

} // namespace sentio


