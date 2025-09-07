#include "sentio/strategy_tfa.hpp"
#include "sentio/tfa/feature_guard.hpp"
#include "sentio/tfa/signal_pipeline.hpp"
#include "sentio/tfa/artifacts_loader.hpp"
#include "sentio/feature/column_projector.hpp"
#include "sentio/feature/name_diff.hpp"
#include <algorithm>
#include <chrono>

namespace sentio {

static ml::WindowSpec make_spec(const ml::ModelSpec& s){
  ml::WindowSpec w;
  // TFA always uses sequence length of 64 (hardcoded for now since TorchScript doesn't store this)
  w.seq_len = 64;
  w.layout  = s.input_layout.empty()? "BTF" : s.input_layout;
  w.feat_dim = (int)s.feature_names.size();
  w.mean = s.mean; w.std = s.std; w.clip2 = s.clip2;
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

void TFAStrategy::set_raw_features(const std::vector<double>& raw){
  static int feature_calls = 0;
  feature_calls++;
  
  // Store features in buffer for batch processing
  feature_buffer_.push_back(raw);
  
  // Detailed diagnostics
  if (feature_calls % 1000 == 0 || feature_calls <= 10) {
    std::cout << "[DIAG] TFA set_raw_features: call=" << feature_calls 
              << " raw.size()=" << raw.size() 
              << " buffer_size=" << feature_buffer_.size()
              << " expected_feat_dim=" << window_.feat_dim() << std::endl;
  }
}

StrategySignal TFAStrategy::map_output(const ml::ModelOutput& mo) const {
  StrategySignal s;
  if (mo.probs.empty()) { 
    s.type = StrategySignal::Type::HOLD; 
    s.confidence = cfg_.conf_floor; 
    return s; 
  }
  
  // Handle binary classifier output (TFA/TFB are binary classifiers)
  if (mo.probs.size() == 1) {
    float prob = mo.probs[0];
    if (prob > 0.5) {
      s.type = StrategySignal::Type::BUY;
      s.confidence = prob;
    } else {
      s.type = StrategySignal::Type::SELL;
      s.confidence = 1.0f - prob;
    }
  } else {
    // Handle 3-class output (for other models like HybridPPO)
    int argmax = 0; 
    for (int i=1;i<(int)mo.probs.size();++i) 
      if (mo.probs[i]>mo.probs[argmax]) argmax=i;
    float pmax = mo.probs[argmax];
    
    // default order ["SELL","HOLD","BUY"]
    if      (argmax==2) s.type = StrategySignal::Type::BUY;
    else if (argmax==0) s.type = StrategySignal::Type::SELL;
    else                s.type = StrategySignal::Type::HOLD;
    s.confidence = std::max(cfg_.conf_floor, (double)pmax);
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

  auto out = handle_.model->predict(*in, window_.seq_len(), window_.feat_dim(), handle_.spec.input_layout);
  
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

void TFAStrategy::apply_params() {
  cfg_.conf_floor = params_["conf_floor"];
}

StrategySignal TFAStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
  (void)bars; (void)current_index; // Features come from set_raw_features
  static int total_calls = 0;
  static int signals_emitted = 0;
  static bool pipeline_initialized = false;
  static tfa::TfaSignalPipeline pipeline;
  static tfa::FeatureGuard guard;
  
  total_calls++;
  last_.reset();
  
  // Initialize pipeline once
  if (!pipeline_initialized) {
    // Setup feature guard with emit_from (skip early bars)
    guard.emit_from = 64; // Need 64 bars for sequence model
    guard.pad_value = 0.0f;
    
    // Setup pipeline
    pipeline.model = handle_.model.get();
    pipeline.policy.min_prob = std::max(0.51f, static_cast<float>(cfg_.conf_floor)); // Sane threshold
    pipeline.cooldown.bars = 5; // Prevent rapid-fire signals
    
    pipeline_initialized = true;
    std::cout << "[DIAG] TFA Pipeline initialized: emit_from=" << guard.emit_from 
              << " min_prob=" << pipeline.policy.min_prob << std::endl;
  }
  
  // Need sufficient features for batch processing
  const size_t min_features = guard.emit_from + 100; // Extra buffer for stability
  if (feature_buffer_.size() < min_features) {
    diag_.drop(DropReason::MIN_BARS);
    if (total_calls % 1000 == 0) {
      std::cout << "[DIAG] TFA: buffer size=" << feature_buffer_.size() 
                << " need=" << min_features << " (waiting for more features)" << std::endl;
    }
    return StrategySignal{};
  }
  
  // Professional feature projection using ColumnProjector
  if (feature_buffer_.size() >= min_features) {
    
    // Initialize projector on first use
    if (!projector_initialized_) {
      try {
        // Load TFA artifacts with metadata validation
        std::string artifacts_path = cfg_.artifacts_dir + "/" + cfg_.model_id + "/" + cfg_.version + "/";
        auto artifacts = tfa::load_tfa_legacy(
          artifacts_path + "model.pt",
          artifacts_path + "metadata.json"
        );
        
        // Get runtime feature names (what we produce)
        std::vector<std::string> runtime_names = {
          // Price features (15)
          "ret_1m", "ret_5m", "ret_15m", "ret_30m", "ret_1h",
          "momentum_5", "momentum_10", "momentum_20",
          "volatility_10", "volatility_20", "volatility_30",
          "atr_14", "atr_21", "parkinson_vol", "garman_klass_vol",
          // Technical features (27)
          "rsi_14", "rsi_21", "rsi_30",
          "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
          "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
          "bb_upper_20", "bb_middle_20", "bb_lower_20",
          "bb_upper_50", "bb_middle_50", "bb_lower_50",
          "macd_line", "macd_signal", "macd_histogram",
          "stoch_k", "stoch_d", "williams_r", "cci_20", "adx_14",
          // Volume features (8)
          "volume_sma_10", "volume_sma_20", "volume_sma_50",
          "volume_roc", "obv", "vpt", "ad_line", "mfi_14",
          // Microstructure features (5)
          "spread_bp", "price_impact", "order_flow_imbalance", 
          "market_depth", "bid_ask_ratio"
        };
        
        // Get expected feature names from model
        auto expected_names = artifacts.get_expected_feature_names();
        float pad_value = artifacts.get_pad_value();
        
        std::cout << "[TFA] Creating ColumnProjector: " << runtime_names.size() 
                  << " → " << expected_names.size() << " features" << std::endl;
        
        // Show feature differences
        print_name_diff(runtime_names, expected_names);
        
        // Create projector
        projector_ = std::make_unique<ColumnProjector>(
          ColumnProjector::make(runtime_names, expected_names, pad_value)
        );
        
        projector_initialized_ = true;
        std::cout << "[TFA] ColumnProjector initialized successfully" << std::endl;
        
      } catch (const std::exception& e) {
        std::cout << "[TFA] Failed to initialize ColumnProjector: " << e.what() << std::endl;
        diag_.drop(DropReason::NAN_INPUT);
        return StrategySignal{};
      }
    }
    
    // Project features using the ColumnProjector
    std::vector<std::vector<double>> projected_features;
    try {
      // Use just the current feature for single-bar inference
      std::vector<std::vector<double>> current_batch = { feature_buffer_.back() };
      projector_->project_cached(current_batch, projected_features);
      
      if (projected_features.empty() || projected_features[0].size() != (size_t)window_.feat_dim()) {
        diag_.drop(DropReason::NAN_INPUT);
        if (total_calls % 1000 == 0) {
          std::cout << "[TFA] Projection failed: got=" << (projected_features.empty() ? 0 : projected_features[0].size())
                    << " expected=" << window_.feat_dim() << std::endl;
        }
        return StrategySignal{};
      }
      
    } catch (const std::exception& e) {
      diag_.drop(DropReason::NAN_INPUT);
      if (total_calls % 1000 == 0) {
        std::cout << "[TFA] Projection error: " << e.what() << std::endl;
      }
      return StrategySignal{};
    }
    
    // Convert projected features to float
    const auto& model_features = projected_features[0];
    std::vector<float> float_features(model_features.size());
    bool has_nan = false;
    for (size_t i = 0; i < model_features.size(); ++i) {
      if (!std::isfinite(model_features[i])) {
        has_nan = true;
        float_features[i] = 0.0f; // Replace NaN with 0
      } else {
        float_features[i] = static_cast<float>(model_features[i]);
      }
    }
    
    if (has_nan) {
      diag_.drop(DropReason::NAN_INPUT);
      if (total_calls % 1000 == 0) {
        std::cout << "[TFA] NaN/Inf features detected after projection" << std::endl;
      }
      return StrategySignal{};
    }
    
    // Single prediction with properly projected features
    // TFA model expects sequence dimension, so we need to create a sequence of length 64
    // For now, we'll replicate the current features to fill the sequence
    const int seq_len = 64; // TFA model requirement
    const int feat_dim = model_features.size();
    std::vector<float> sequence_features(seq_len * feat_dim);
    
    // Fill sequence with replicated current features (simple approach for single-bar inference)
    for (int t = 0; t < seq_len; ++t) {
      for (int f = 0; f < feat_dim; ++f) {
        sequence_features[t * feat_dim + f] = float_features[f];
      }
    }
    
    auto output = pipeline.model->predict(sequence_features, seq_len, feat_dim, "BTF");
    if (!output) {
      diag_.drop(DropReason::NAN_INPUT);
      if (total_calls % 1000 == 0) {
        std::cout << "[DIAG] TFA: Model prediction failed" << std::endl;
      }
      return StrategySignal{};
    }
    
    // Extract probability/score
    float prob = 0.5f;
    if (!output->probs.empty()) {
      prob = output->probs[0];
    } else {
      prob = 1.0f / (1.0f + std::exp(-output->score)); // Sigmoid
    }
    
    // Apply threshold
    if (prob < pipeline.policy.min_prob && (1.0f - prob) < pipeline.policy.min_prob) {
      diag_.drop(DropReason::THRESHOLD);
      if (total_calls % 1000 == 0) {
        std::cout << "[DIAG] TFA: Below threshold: prob=" << prob 
                  << " min_prob=" << pipeline.policy.min_prob << std::endl;
      }
      return StrategySignal{};
    }
    
    // Generate signal
    StrategySignal sig;
    sig.type = prob > 0.5 ? StrategySignal::Type::BUY : StrategySignal::Type::SELL;
    sig.confidence = prob > 0.5 ? prob : (1.0f - prob);
    
    signals_emitted++;
    diag_.emitted++;
    
    std::cout << "[DIAG] TFA: SIGNAL GENERATED! total=" << signals_emitted 
              << " conf=" << sig.confidence << " type=" 
              << (sig.type == StrategySignal::Type::BUY ? "BUY" : "SELL") 
              << " prob=" << prob << std::endl;
    
    last_ = sig;
    return sig;
  }
  
  // Not enough features yet
  diag_.drop(DropReason::MIN_BARS);
  return StrategySignal{};
}

REGISTER_STRATEGY(TFAStrategy, "TFA");

} // namespace sentio
