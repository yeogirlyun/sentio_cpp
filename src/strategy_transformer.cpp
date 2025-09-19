#include "sentio/strategy_transformer.hpp"
#include "sentio/time_utils.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <filesystem>

namespace sentio {

TransformerStrategy::TransformerStrategy()
    : BaseStrategy("Transformer")
    , cfg_()
{
    // CRITICAL: Ensure configuration has correct dimensions
    cfg_.feature_dim = 128;  // Must match FeaturePipeline::TOTAL_FEATURES
    cfg_.sequence_length = 64;
    cfg_.d_model = 256;
    cfg_.num_heads = 8;
    cfg_.num_layers = 6;
    cfg_.dropout = 0.1;
    cfg_.model_path = "artifacts/Transformer/v1/model.pt";
    cfg_.enable_online_training = true;
    
    std::cout << "🔧 Initializing TransformerStrategy with CORRECTED dimensions:" << std::endl;
    std::cout << "  sequence_length: " << cfg_.sequence_length << std::endl;
    std::cout << "  feature_dim: " << cfg_.feature_dim << std::endl;
    std::cout << "  d_model: " << cfg_.d_model << std::endl;
    
    initialize_model();
}

TransformerStrategy::TransformerStrategy(const TransformerCfg& cfg)
    : BaseStrategy("Transformer")
    , cfg_(cfg)
{
    // CRITICAL: Validate and correct configuration dimensions
    if (cfg_.feature_dim != 128) {
        std::cerr << "⚠️  WARNING: feature_dim was " << cfg_.feature_dim << ", correcting to 128" << std::endl;
        cfg_.feature_dim = 128;
    }
    
    if (!validate_configuration()) {
        std::cerr << "❌ Configuration validation failed! Using safe defaults." << std::endl;
        cfg_.feature_dim = 128;
        cfg_.sequence_length = 64;
    }
    
    std::cout << "🔧 Initializing TransformerStrategy with validated dimensions:" << std::endl;
    std::cout << "  sequence_length: " << cfg_.sequence_length << std::endl;
    std::cout << "  feature_dim: " << cfg_.feature_dim << std::endl;
    std::cout << "  d_model: " << cfg_.d_model << std::endl;
    
    initialize_model();
}

void TransformerStrategy::initialize_model() {
    try {
        // Create transformer configuration
        TransformerConfig model_config;
        model_config.feature_dim = cfg_.feature_dim;
        model_config.sequence_length = cfg_.sequence_length;
        model_config.d_model = cfg_.d_model;
        model_config.num_heads = cfg_.num_heads;
        model_config.num_layers = cfg_.num_layers;
        model_config.ffn_hidden = cfg_.ffn_hidden;
        model_config.dropout = cfg_.dropout;
        
        // Initialize model
        model_ = std::make_shared<TransformerModel>(model_config);
        
        // Try to load pre-trained model if it exists
        if (std::filesystem::exists(cfg_.model_path)) {
            std::cout << "Loading pre-trained transformer model from: " << cfg_.model_path << std::endl;
            model_->load_model(cfg_.model_path);
        } else {
            std::cout << "No pre-trained model found at: " << cfg_.model_path << std::endl;
            std::cout << "Using randomly initialized model" << std::endl;
        }
        
        model_->optimize_for_inference();
        
        // Initialize feature pipeline
        TransformerConfig::Features feature_config;
        feature_config.normalization = TransformerConfig::Features::NormalizationMethod::Z_SCORE;
        feature_config.decay_factor = 0.999f;
        feature_pipeline_ = std::make_unique<FeaturePipeline>(feature_config);
        
        // Initialize online trainer if enabled
        if (cfg_.enable_online_training) {
            OnlineTrainer::OnlineConfig trainer_config;
            trainer_config.update_interval_minutes = cfg_.update_interval_minutes;
            trainer_config.min_samples_for_update = cfg_.min_samples_for_update;
            trainer_config.base_learning_rate = 0.0001f;
            trainer_config.replay_buffer_size = 10000;
            trainer_config.enable_regime_detection = true;
            
            online_trainer_ = std::make_unique<OnlineTrainer>(model_, trainer_config);
            std::cout << "Online training enabled with " << cfg_.update_interval_minutes << " minute intervals" << std::endl;
        }
        
        model_initialized_ = true;
        std::cout << "Transformer strategy initialized successfully" << std::endl;
        std::cout << "Model parameters: " << model_->get_parameter_count() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize transformer strategy: " << e.what() << std::endl;
        model_initialized_ = false;
    }
}

ParameterMap TransformerStrategy::get_default_params() const {
    ParameterMap defaults;
    defaults["buy_threshold"] = cfg_.buy_threshold;
    defaults["sell_threshold"] = cfg_.sell_threshold;
    defaults["strong_threshold"] = cfg_.strong_threshold;
    defaults["conf_floor"] = cfg_.conf_floor;
    return defaults;
}

ParameterSpace TransformerStrategy::get_param_space() const {
    return {
        {"buy_threshold", {ParamType::FLOAT, 0.5, 0.8, 0.6}},
        {"sell_threshold", {ParamType::FLOAT, 0.2, 0.5, 0.4}},
        {"strong_threshold", {ParamType::FLOAT, 0.7, 0.9, 0.8}},
        {"conf_floor", {ParamType::FLOAT, 0.4, 0.6, 0.5}}
    };
}

void TransformerStrategy::apply_params() {
    if (params_.count("buy_threshold")) {
        cfg_.buy_threshold = static_cast<float>(params_["buy_threshold"]);
    }
    if (params_.count("sell_threshold")) {
        cfg_.sell_threshold = static_cast<float>(params_["sell_threshold"]);
    }
    if (params_.count("strong_threshold")) {
        cfg_.strong_threshold = static_cast<float>(params_["strong_threshold"]);
    }
    if (params_.count("conf_floor")) {
        cfg_.conf_floor = static_cast<float>(params_["conf_floor"]);
    }
}


void TransformerStrategy::update_bar_history(const Bar& bar) {
    bar_history_.push_back(bar);
    
    // Keep only the required sequence length + some buffer
    const size_t max_history = cfg_.sequence_length + 50;
    while (bar_history_.size() > max_history) {
        bar_history_.pop_front();
    }
}

std::vector<Bar> TransformerStrategy::convert_to_transformer_bars(const std::vector<Bar>& sentio_bars) const {
    std::vector<Bar> transformer_bars;
    transformer_bars.reserve(sentio_bars.size());
    
    for (const auto& sentio_bar : sentio_bars) {
        // Convert Sentio Bar to Transformer Bar format
        Bar transformer_bar;
        transformer_bar.open = sentio_bar.open;
        transformer_bar.high = sentio_bar.high;
        transformer_bar.low = sentio_bar.low;
        transformer_bar.close = sentio_bar.close;
        transformer_bar.volume = sentio_bar.volume;
        transformer_bar.ts_utc_epoch = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count(); // Use current time
        
        transformer_bars.push_back(transformer_bar);
    }
    
    return transformer_bars;
}

bool TransformerStrategy::validate_tensor_dimensions(const torch::Tensor& tensor, 
                                                   const std::vector<int64_t>& expected_dims,
                                                   const std::string& tensor_name) {
    if (tensor.sizes().size() != expected_dims.size()) {
        std::cerr << "Dimension count mismatch for " << tensor_name 
                  << ": expected " << expected_dims.size() 
                  << " dims, got " << tensor.sizes().size() << std::endl;
        return false;
    }
    
    for (size_t i = 0; i < expected_dims.size(); ++i) {
        if (tensor.size(i) != expected_dims[i]) {
            std::cerr << "Dimension " << i << " mismatch for " << tensor_name 
                      << ": expected " << expected_dims[i] 
                      << ", got " << tensor.size(i) << std::endl;
            return false;
        }
    }
    
    return true;
}

bool TransformerStrategy::validate_configuration() {
    // Check model configuration consistency
    if (cfg_.feature_dim != 128) {
        std::cerr << "WARNING: Feature dim should be 128, got " << cfg_.feature_dim << std::endl;
        return false;
    }
    
    if (cfg_.sequence_length <= 0 || cfg_.sequence_length > 1000) {
        std::cerr << "Invalid sequence length: " << cfg_.sequence_length << std::endl;
        return false;
    }
    
    if (cfg_.d_model % cfg_.num_heads != 0) {
        std::cerr << "d_model must be divisible by num_heads" << std::endl;
        return false;
    }
    
    // Check file paths
    if (!std::filesystem::exists(cfg_.model_path)) {
        std::cout << "Model file not found: " << cfg_.model_path << std::endl;
        std::cout << "Will use randomly initialized model" << std::endl;
    }
    
    return true;
}

TransformerFeatureMatrix TransformerStrategy::generate_features_for_bars(const std::vector<Bar>& bars, int end_index) {
    if (!model_initialized_ || bars.empty() || end_index < 0) {
        // Return properly shaped zero tensor: [1, sequence_length, feature_dim]
        return torch::zeros({1, cfg_.sequence_length, cfg_.feature_dim});
    }
    
    // Get sequence of bars
    std::vector<Bar> sequence_bars;
    int start_idx = std::max(0, end_index - cfg_.sequence_length + 1);
    
    for (int i = start_idx; i <= end_index && i < static_cast<int>(bars.size()); ++i) {
        sequence_bars.push_back(bars[i]);
    }
    
    // Convert to transformer bar format
    auto transformer_bars = convert_to_transformer_bars(sequence_bars);
    
    try {
        // CRITICAL FIX: Generate features for each bar in sequence individually
        std::vector<torch::Tensor> sequence_features;
        sequence_features.reserve(cfg_.sequence_length);
        
        // Generate features for each bar in the sequence
        for (size_t i = 0; i < transformer_bars.size(); ++i) {
            std::vector<Bar> single_bar = {transformer_bars[i]};
            auto bar_features = feature_pipeline_->generate_features(single_bar);
            
            // bar_features is [1, 128] - squeeze to get [128]
            auto squeezed_features = bar_features.squeeze(0);
            
            // Validate feature dimensions
            if (squeezed_features.size(0) != cfg_.feature_dim) {
                std::cerr << "Feature dimension mismatch: expected " << cfg_.feature_dim 
                          << ", got " << squeezed_features.size(0) << std::endl;
                return torch::zeros({1, cfg_.sequence_length, cfg_.feature_dim});
            }
            
            sequence_features.push_back(squeezed_features);
        }
        
        // Pad sequence if we don't have enough bars (pad at beginning with zeros)
        while (sequence_features.size() < cfg_.sequence_length) {
            sequence_features.insert(sequence_features.begin(), 
                                   torch::zeros({cfg_.feature_dim}));
        }
        
        // Take only the last sequence_length features if we have too many
        if (sequence_features.size() > cfg_.sequence_length) {
            sequence_features.erase(sequence_features.begin(), 
                                  sequence_features.end() - cfg_.sequence_length);
        }
        
        // Stack into proper tensor: [sequence_length, feature_dim]
        auto stacked_features = torch::stack(sequence_features, 0);
        
        // Add batch dimension: [1, sequence_length, feature_dim]
        auto batched_features = stacked_features.unsqueeze(0);
        
        // Validate final tensor shape using IntArrayRef comparison
        auto expected_shape = torch::IntArrayRef({1, cfg_.sequence_length, cfg_.feature_dim});
        if (batched_features.sizes() != expected_shape) {
            std::cerr << "Final tensor shape mismatch: expected [1, " 
                      << cfg_.sequence_length << ", " << cfg_.feature_dim 
                      << "], got " << batched_features.sizes() << std::endl;
            return torch::zeros({1, cfg_.sequence_length, cfg_.feature_dim});
        }
        
        return batched_features;
        
    } catch (const std::exception& e) {
        std::cerr << "Error generating features sequence: " << e.what() << std::endl;
        return torch::zeros({1, cfg_.sequence_length, cfg_.feature_dim});
    }
}

double TransformerStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (!model_initialized_ || bars.empty() || current_index < 0) {
        std::cerr << "Model not initialized or invalid input" << std::endl;
        return 0.5f;
    }
    
    try {
        // Generate features with proper dimensions
        auto features = generate_features_for_bars(bars, current_index);
        
        // Validate input tensor dimensions before model inference
        auto expected_shape = torch::IntArrayRef({1, cfg_.sequence_length, cfg_.feature_dim});
        if (features.sizes() != expected_shape) {
            std::cerr << "Feature tensor shape mismatch before inference: expected " 
                      << expected_shape << ", got " << features.sizes() << std::endl;
            return 0.5f;
        }
        
        // Ensure model is in eval mode
        model_->eval();
        torch::NoGradGuard no_grad;
        
        // Debug: Print tensor shapes before inference
        std::cout << "Input tensor shape: " << features.sizes() << std::endl;
        
        // Run inference
        auto prediction_tensor = model_->forward(features);
        
        // Debug: Print output tensor shape
        std::cout << "Output tensor shape: " << prediction_tensor.sizes() << std::endl;
        
        float raw_prediction = prediction_tensor.item<float>();
        
        // Debug: Print raw prediction
        std::cout << "Raw prediction: " << raw_prediction << std::endl;
        
        // Convert to probability using sigmoid (more stable than direct sigmoid)
        float probability = 1.0f / (1.0f + std::exp(-std::clamp(raw_prediction, -10.0f, 10.0f)));
        
        // Ensure probability is in valid range and not exactly neutral
        probability = std::clamp(probability, 0.01f, 0.99f);
        
        // Debug: Print final probability
        std::cout << "Final probability: " << probability << std::endl;
        
        // Log significant predictions (non-neutral)
        if (std::abs(probability - 0.5f) > 0.05f) {
            std::cout << "🎯 Non-neutral signal: " << probability << " (raw: " << raw_prediction << ")" << std::endl;
        }
        
        // Update performance metrics
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            current_metrics_.samples_processed++;
        }
        
        return probability;
        
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR in calculate_probability: " << e.what() << std::endl;
        
        // Log detailed tensor information for debugging
        try {
            auto features = generate_features_for_bars(bars, current_index);
            std::cerr << "Debug - Feature tensor shape: " << features.sizes() << std::endl;
            std::cerr << "Debug - Feature tensor dtype: " << features.dtype() << std::endl;
            std::cerr << "Debug - Model expects: [1, " << cfg_.sequence_length 
                      << ", " << cfg_.feature_dim << "]" << std::endl;
        } catch (...) {
            std::cerr << "Failed to generate debug information" << std::endl;
        }
        
        return 0.5f; // Only return neutral as last resort
    }
}

void TransformerStrategy::maybe_trigger_training() {
    if (!cfg_.enable_online_training || !online_trainer_ || is_training_.load()) {
        return;
    }
    
    try {
        // Check if we should update the model
        if (online_trainer_->should_update_model()) {
            std::cout << "Triggering transformer model update..." << std::endl;
            is_training_ = true;
            
            auto result = online_trainer_->update_model();
            if (result.success) {
                std::cout << "Model update completed successfully" << std::endl;
            } else {
                std::cerr << "Model update failed: " << result.error_message << std::endl;
            }
            
            is_training_ = false;
        }
        
        // Check for regime changes
        if (online_trainer_->detect_regime_change()) {
            std::cout << "Market regime change detected, adapting model..." << std::endl;
            online_trainer_->adapt_to_regime_change();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in maybe_trigger_training: " << e.what() << std::endl;
        is_training_ = false;
    }
}


std::vector<AllocationDecision> TransformerStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol, 
    const std::string& bear3x_symbol
) const {
    
    std::vector<AllocationDecision> decisions;
    
    if (!model_initialized_ || bars.empty() || current_index < 0) {
        return decisions;
    }
    
    // Get probability from strategy
    double probability = const_cast<TransformerStrategy*>(this)->calculate_probability(bars, current_index);
    
    // **PROFIT MAXIMIZATION**: Always deploy 100% of capital with maximum leverage
    if (probability > cfg_.strong_threshold) {
        // Strong buy: 100% TQQQ (3x leveraged long)
        decisions.push_back({bull3x_symbol, 1.0, "Transformer strong buy: 100% " + bull3x_symbol + " (3x leverage)"});
        
    } else if (probability > cfg_.buy_threshold) {
        // Moderate buy: 100% QQQ (1x long)
        decisions.push_back({base_symbol, 1.0, "Transformer moderate buy: 100% " + base_symbol});
        
    } else if (probability < (1.0f - cfg_.strong_threshold)) {
        // Strong sell: 100% SQQQ (3x leveraged short)
        decisions.push_back({bear3x_symbol, 1.0, "Transformer strong sell: 100% " + bear3x_symbol + " (3x inverse)"});
        
    } else if (probability < cfg_.sell_threshold) {
        // Weak sell: 100% PSQ (1x inverse)
        decisions.push_back({"PSQ", 1.0, "Transformer weak sell: 100% PSQ (1x inverse)"});
        
    } else {
        // Neutral: Stay in cash (rare case)
        // No positions needed
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol, "PSQ"};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, "Transformer: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

// Register the strategy
REGISTER_STRATEGY(TransformerStrategy, "transformer");

} // namespace sentio
