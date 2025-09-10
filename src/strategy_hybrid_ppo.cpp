#include "sentio/strategy_hybrid_ppo.hpp"
#include <algorithm>
#include <stdexcept>

namespace sentio {

HybridPPOStrategy::HybridPPOStrategy()
: BaseStrategy("HybridPPO"),
  cfg_(),
  handle_(ml::ModelRegistryTS::load_torchscript("HybridPPO", cfg_.version, cfg_.artifacts_dir, cfg_.use_cuda)),
  fpipe_(handle_.spec)
{}

HybridPPOStrategy::HybridPPOStrategy(const HybridPPOCfg& cfg)
: BaseStrategy("HybridPPO"),
  cfg_(cfg),
  handle_(ml::ModelRegistryTS::load_torchscript("HybridPPO", cfg.version, cfg.artifacts_dir, cfg.use_cuda)),
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

double HybridPPOStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
  (void)bars; (void)current_index; // Features come from set_raw_features
  last_.reset();
  auto z = fpipe_.transform(raw_);
  if (!z) return 0.5; // Neutral

  auto out = handle_.model->predict(*z, 1, (int)z->size(), "BF");
  if (!out) return 0.5; // Neutral

  StrategySignal sig = map_output(*out);
  // Safety: never emit if below floor
  if (sig.confidence < cfg_.conf_floor) return 0.5; // Neutral

  // Convert discrete signal to probability
  double probability;
  if (sig.type == StrategySignal::Type::BUY) {
    probability = 0.5 + sig.confidence * 0.5; // 0.5 to 1.0
  } else if (sig.type == StrategySignal::Type::SELL) {
    probability = 0.5 - sig.confidence * 0.5; // 0.0 to 0.5
  } else {
    probability = 0.5; // HOLD
  }

  last_ = sig;
  return probability;
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

std::vector<BaseStrategy::AllocationDecision> HybridPPOStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol,
    const std::string& bear1x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // HybridPPO uses simple allocation based on signal strength
    if (probability > 0.6) {
        // Buy signal
        double conviction = (probability - 0.6) / 0.4; // Scale 0.6-1.0 to 0-1
        double base_weight = 0.3 + (conviction * 0.7); // 30-100% allocation
        
        decisions.push_back({base_symbol, base_weight, conviction, "HybridPPO buy: 100% QQQ"});
    } else if (probability < 0.4) {
        // Sell signal
        double conviction = (0.4 - probability) / 0.4; // Scale 0.0-0.4 to 0-1
        double base_weight = 0.3 + (conviction * 0.7); // 30-100% allocation
        
        decisions.push_back({bear1x_symbol, base_weight, conviction, "HybridPPO sell: 100% PSQ"});
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol, bear1x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "HybridPPO: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg HybridPPOStrategy::get_router_config() const {
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    cfg.bear1x = "PSQ";
    return cfg;
}

SizerCfg HybridPPOStrategy::get_sizer_config() const {
    SizerCfg cfg;
    cfg.max_position_pct = 1.0; // 100% max position
    cfg.volatility_target = 0.15; // 15% volatility target
    return cfg;
}

// Register the strategy with the factory
// REGISTER_STRATEGY(HybridPPOStrategy, "hybrid_ppo")  // Disabled - not working

} // namespace sentio
