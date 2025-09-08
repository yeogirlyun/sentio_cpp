#include "sentio/strategy_tfa.hpp"
#include "sentio/tfa/feature_guard.hpp"
#include "sentio/tfa/signal_pipeline.hpp"
#include "sentio/tfa/artifacts_safe.hpp"
#include "sentio/feature/column_projector_safe.hpp"
#include "sentio/feature/name_diff.hpp"
#include "sentio/tfa/tfa_seq_context.hpp"
#include <algorithm>
#include <chrono>

namespace sentio {

static ml::WindowSpec make_spec(const ml::ModelSpec& s){
  ml::WindowSpec w;
  // TFA always uses sequence length of 64 (hardcoded for now since TorchScript doesn't store this)
  w.seq_len = 64;
  w.layout  = s.input_layout.empty()? "BTF" : s.input_layout;
  w.feat_dim = (int)s.feature_names.size();
  // Disable external normalization; model contains its own scaler
  w.mean.clear();
  w.std.clear();
  w.clip2 = s.clip2;
  return w;
}

TFAStrategy::TFAStrategy()
: BaseStrategy("TFA")
, cfg_()
, handle_(ml::ModelRegistryTS::load_torchscript(cfg_.model_id, cfg_.version, cfg_.artifacts_dir, cfg_.use_cuda))
, window_(make_spec(handle_.spec))
{}

TFAStrategy::TFAStrategy(const TFACfg& cfg)
: BaseStrategy("TFA")
, cfg_(cfg)
, handle_(ml::ModelRegistryTS::load_torchscript(cfg.model_id, cfg.version, cfg.artifacts_dir, cfg.use_cuda))
, window_(make_spec(handle_.spec))
{
  // Model loaded successfully
}

void TFAStrategy::apply_params() {
  cfg_.conf_floor = params_["conf_floor"];
}

void TFAStrategy::set_raw_features(const std::vector<double>& raw){
  static int feature_calls = 0;
  feature_calls++;

  // Initialize safe projector on first use
  if (!projector_initialized_) {
    try {
      std::string artifacts_path = cfg_.artifacts_dir + "/" + cfg_.model_id + "/" + cfg_.version + "/";
      auto artifacts = tfa::load_tfa_artifacts_safe(
        artifacts_path + "model.pt",
        artifacts_path + "feature_spec.json",
        artifacts_path + "model.meta.json"
      );

      const int F_expected = artifacts.get_expected_input_dim();
      const auto& expected_names = artifacts.get_expected_feature_names();
      if (F_expected != 55) {
        throw std::runtime_error("Unsupported model input_dim (expect exactly 55)");
      }

      auto runtime_names = tfa::feature_names_from_spec(artifacts.spec);
      float pad_value = artifacts.get_pad_value();

      std::cout << "[TFA] Creating safe ColumnProjector: runtime=" << runtime_names.size()
                << " -> expected=" << expected_names.size() << " features" << std::endl;

      projector_safe_ = std::make_unique<ColumnProjectorSafe>(
        ColumnProjectorSafe::make(runtime_names, expected_names, pad_value)
      );

      expected_feat_dim_ = F_expected;
      projector_initialized_ = true;
      std::cout << "[TFA] Safe projector initialized: expecting " << expected_feat_dim_ << " features" << std::endl;
    } catch (const std::exception& e) {
      std::cout << "[TFA] Failed to initialize safe projector: " << e.what() << std::endl;
      return;
    }
  }

  // Project raw -> expected order and sanitize, then push into window
  try {
    std::vector<float> proj_f;
    projector_safe_->project_double(raw.data(), 1, raw.size(), proj_f);

    std::vector<double> proj_d;
    proj_d.resize((size_t)expected_feat_dim_);
    for (int i = 0; i < expected_feat_dim_; ++i) {
      float v = (i < (int)proj_f.size() && std::isfinite(proj_f[(size_t)i])) ? proj_f[(size_t)i] : 0.0f;
      proj_d[(size_t)i] = static_cast<double>(v);
    }

    window_.push(proj_d);
  } catch (const std::exception& e) {
    if (feature_calls % 1000 == 0 || feature_calls <= 10) {
      std::cout << "[TFA] Projection error in set_raw_features: " << e.what() << std::endl;
    }
  }

  if (feature_calls % 1000 == 0 || feature_calls <= 10) {
    std::cout << "[DIAG] TFA set_raw_features: call=" << feature_calls
              << " raw.size()=" << raw.size()
              << " window_ready=" << (window_.ready()? 1:0)
              << " feat_dim=" << window_.feat_dim() << std::endl;
  }
}

StrategySignal TFAStrategy::map_output(const ml::ModelOutput& mo) const {
  StrategySignal s;
  // If explicit probabilities are provided
  if (!mo.probs.empty()) {
    if (mo.probs.size() == 1) {
      float prob = mo.probs[0];
      if (prob > 0.5f) {
        s.type = StrategySignal::Type::BUY;
        s.confidence = prob;
      } else {
        s.type = StrategySignal::Type::SELL;
        s.confidence = 1.0f - prob;
      }
      return s;
    }
    // 3-class path
    int argmax = 0; 
    for (int i=1;i<(int)mo.probs.size();++i) 
      if (mo.probs[i]>mo.probs[argmax]) argmax=i;
    float pmax = mo.probs[argmax];
    if      (argmax==2) s.type = StrategySignal::Type::BUY;
    else if (argmax==0) s.type = StrategySignal::Type::SELL;
    else                s.type = StrategySignal::Type::HOLD;
    s.confidence = std::max(cfg_.conf_floor, (double)pmax);
    return s;
  }

  // Fallback: logits-only path (binary)
  float logit = mo.score;
  float prob = 1.0f / (1.0f + std::exp(-logit));
  if (prob > 0.5f) {
    s.type = StrategySignal::Type::BUY;
    s.confidence = prob;
  } else {
    s.type = StrategySignal::Type::SELL;
    s.confidence = 1.0f - prob;
  }
  return s;
}

void TFAStrategy::on_bar(const StrategyCtx& ctx, const Bar& b){
  (void)ctx; (void)b;
  last_.reset();
  
  // Diagnostic: Check if window is ready
  if (!window_.ready()) {
    std::cout << "[TFA] Window not ready, required=" << window_.seq_len() << std::endl;
    return;
  }

  auto in = window_.to_input();
  if (!in) {
    std::cout << "[TFA] Failed to create input from window" << std::endl;
    return;
  }

  std::optional<ml::ModelOutput> out;
  try {
    out = handle_.model->predict(*in, window_.seq_len(), window_.feat_dim(), handle_.spec.input_layout);
  } catch (const std::exception& e) {
    std::cout << "[TFA] predict threw: " << e.what() << std::endl;
    return;
  }
  
  if (!out) {
    std::cout << "[TFA] Model prediction failed" << std::endl;
    return;
  }

  auto sig = map_output(*out);
  std::cout << "[TFA] Raw confidence=" << sig.confidence << ", floor=" << cfg_.conf_floor << std::endl;
  if (sig.confidence < cfg_.conf_floor) {
    std::cout << "[TFA] Signal dropped due to low confidence" << std::endl;
    return;
  }
  last_ = sig;
  std::cout << "[TFA] Signal generated: " << (sig.type == StrategySignal::Type::BUY ? "BUY" : 
                                              sig.type == StrategySignal::Type::SELL ? "SELL" : "HOLD") 
            << " conf=" << sig.confidence << std::endl;
}

ParameterMap TFAStrategy::get_default_params() const {
  return {
    {"conf_floor", cfg_.conf_floor}
  };
}

ParameterSpace TFAStrategy::get_param_space() const {
  return {
    {"conf_floor", {ParamType::FLOAT, 0.0, 1.0, cfg_.conf_floor}}
  };
}

StrategySignal TFAStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
  (void)current_index; // we will use bars.size() and a static cursor
  static int calls = 0;
  ++calls;
  last_.reset();

  // One-time: precompute probabilities over the whole series using the sequence context
  static bool seq_inited = false;
  static TfaSeqContext seq_ctx;
  static std::vector<float> probs_all;
  if (!seq_inited) {
    try {
      std::string artifacts_path = cfg_.artifacts_dir + "/" + cfg_.model_id + "/" + cfg_.version + "/";
      seq_ctx.load(artifacts_path + "model.pt",
                   artifacts_path + "feature_spec.json",
                   artifacts_path + "model.meta.json");
      // Assume base symbol is QQQ for this test run
      seq_ctx.forward_probs("QQQ", bars, probs_all);
      seq_inited = true;
      std::cout << "[TFA seq] precomputed probs: N=" << probs_all.size() << std::endl;
    } catch (const std::exception& e) {
      std::cout << "[TFA seq] init/forward failed: " << e.what() << std::endl;
      return StrategySignal{};
    }
  }

  // Maintain rolling threshold logic with cooldown based on precomputed prob at this call index
  float prob = (calls-1 < (int)probs_all.size()) ? probs_all[(size_t)(calls-1)] : 0.5f;

  static std::vector<float> p_hist; p_hist.reserve(4096);
  static int cooldown_long_until = -1;
  static int cooldown_short_until = -1;
  const int window = 250;
  const float q_long = 0.80f, q_short = 0.20f;
  const float floor_long = 0.55f, ceil_short = 0.45f;
  const int cooldown = 5;

  p_hist.push_back(prob);

  StrategySignal sig{}; sig.type = StrategySignal::Type::HOLD; sig.confidence = 0.0;

  if ((int)p_hist.size() >= std::max(window, seq_ctx.T)) {
    int end = (int)p_hist.size() - 1;
    int start = std::max(0, end - window + 1);
    std::vector<float> win(p_hist.begin() + start, p_hist.begin() + end + 1);

    int kL = (int)std::floor(q_long * (win.size() - 1));
    std::nth_element(win.begin(), win.begin() + kL, win.end());
    float thrL = std::max(floor_long, win[kL]);

    int kS = (int)std::floor(q_short * (win.size() - 1));
    std::nth_element(win.begin(), win.begin() + kS, win.end());
    float thrS = std::min(ceil_short, win[kS]);

    bool can_long = (calls >= cooldown_long_until);
    bool can_short = (calls >= cooldown_short_until);

    if (can_long && prob >= thrL) {
      sig.type = StrategySignal::Type::BUY;
      sig.confidence = prob;
      cooldown_long_until = calls + cooldown;
    } else if (can_short && prob <= thrS) {
      sig.type = StrategySignal::Type::SELL;
      sig.confidence = 1.0f - prob;
      cooldown_short_until = calls + cooldown;
    }

    if (calls % 64 == 0) {
      std::cout << "[TFA calc] prob=" << prob << " thrL=" << thrL << " thrS=" << thrS
                << " type=" << (sig.type==StrategySignal::Type::BUY?"BUY":sig.type==StrategySignal::Type::SELL?"SELL":"HOLD")
                << std::endl;
    }
  }

  if (sig.type == StrategySignal::Type::HOLD) return StrategySignal{};
  last_ = sig;
  return sig;
}

REGISTER_STRATEGY(TFAStrategy, "TFA");

} // namespace sentio
