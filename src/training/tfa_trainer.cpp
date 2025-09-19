#include "sentio/training/tfa_trainer.hpp"
#include "sentio/feature/feature_from_spec.hpp"
#include "sentio/time_utils.hpp"

#include <torch/script.h>
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <cmath>

namespace sentio::training {

// ============================================================================
// TFATransformerImpl Implementation
// ============================================================================

TFATransformerImpl::TFATransformerImpl(int feature_dim, int sequence_length, 
                                       int d_model, int nhead, int num_layers, 
                                       int ffn_hidden, float dropout)
    : feature_dim_(feature_dim), sequence_length_(sequence_length), d_model_(d_model) {
    
    // Input projection: [batch, seq, features] -> [batch, seq, d_model]
    input_projection = register_module("input_projection", 
        torch::nn::Linear(feature_dim, d_model));
    
    // Transformer encoder
    auto encoder_layer = torch::nn::TransformerEncoderLayer(
        torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
            .dim_feedforward(ffn_hidden)
            .dropout(dropout)
            .activation(torch::kGELU)
    );
    
    transformer = register_module("transformer",
        torch::nn::TransformerEncoder(encoder_layer, num_layers));
    
    // Layer normalization
    layer_norm = register_module("layer_norm", torch::nn::LayerNorm(d_model));
    
    // Output projection: d_model -> 1 (binary classification)
    output_projection = register_module("output_projection",
        torch::nn::Linear(d_model, 1));
    
    // Dropout
    dropout = register_module("dropout", torch::nn::Dropout(dropout));
    
    // Create positional encoding
    create_positional_encoding();
}

void TFATransformerImpl::create_positional_encoding() {
    positional_encoding = torch::zeros({sequence_length_, d_model_});
    
    auto position = torch::arange(0, sequence_length_).unsqueeze(1).to(torch::kFloat);
    auto div_term = torch::exp(torch::arange(0, d_model_, 2).to(torch::kFloat) * 
                              -(std::log(10000.0) / d_model_));
    
    positional_encoding.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)}, 
                                  torch::sin(position * div_term));
    positional_encoding.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)}, 
                                  torch::cos(position * div_term));
    
    // Register as buffer (not a parameter)
    register_buffer("pos_encoding", positional_encoding);
}

torch::Tensor TFATransformerImpl::forward(torch::Tensor x) {
    // x shape: [batch, seq, features]
    auto batch_size = x.size(0);
    
    // Project to d_model
    x = input_projection(x);  // [batch, seq, d_model]
    
    // Add positional encoding
    x = x + positional_encoding.unsqueeze(0).expand({batch_size, -1, -1});
    
    // Apply dropout
    x = dropout(x);
    
    // Transformer expects [seq, batch, d_model]
    x = x.transpose(0, 1);
    
    // Pass through transformer
    x = transformer(x);  // [seq, batch, d_model]
    
    // Back to [batch, seq, d_model]
    x = x.transpose(0, 1);
    
    // Layer norm
    x = layer_norm(x);
    
    // Take the last timestep for prediction
    x = x.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});  // [batch, d_model]
    
    // Output projection
    x = output_projection(x);  // [batch, 1]
    
    return x.squeeze(-1);  // [batch]
}

// ============================================================================
// TFATrainer Implementation
// ============================================================================

TFATrainer::TFATrainer(const TFATrainingConfig& config) 
    : config_(config),
      model_(config.feature_dim, config.sequence_length, config.d_model,
             config.nhead, config.num_layers, config.ffn_hidden, config.dropout),
      optimizer_(model_->parameters(), torch::optim::AdamOptions(config.learning_rate)
                  .weight_decay(config.weight_decay)),
      device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
    
    // Move model to device
    model_->to(device_);
    
    std::cout << "🚀 TFA C++ Trainer initialized" << std::endl;
    std::cout << "📱 Device: " << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;
    std::cout << "🏗️  Model: " << config_.num_layers << " layers, " 
              << config_.d_model << " d_model, " << config_.nhead << " heads" << std::endl;
}

bool TFATrainer::train_from_bars(const std::vector<Bar>& bars, 
                                 const std::string& feature_spec_path) {
    std::cout << "📊 Training from " << bars.size() << " bars" << std::endl;
    
    // Extract features and create labels
    auto features = extract_features_from_bars(bars, feature_spec_path);
    auto labels = create_labels_from_bars(bars);
    
    if (features.size(0) == 0 || labels.size(0) == 0) {
        std::cerr << "❌ Failed to extract features or labels" << std::endl;
        return false;
    }
    
    std::cout << "✅ Features shape: " << features.sizes() << std::endl;
    std::cout << "✅ Labels shape: " << labels.sizes() << std::endl;
    
    // Prepare training data
    auto data = prepare_training_data(features, labels);
    
    // Reset metrics
    metrics_.reset();
    
    // Training loop
    model_->train();
    
    for (int epoch = 0; epoch < config_.epochs && !should_stop_; ++epoch) {
        // Train
        train_epoch(data.X_train, data.y_train);
        
        // Validate
        float val_loss = validate_epoch(data.X_val, data.y_val);
        
        // Update metrics
        metrics_.val_losses.push_back(val_loss);
        
        // Check for improvement
        if (val_loss < metrics_.best_val_loss - config_.min_delta) {
            metrics_.best_val_loss = val_loss;
            metrics_.best_epoch = epoch;
            metrics_.epochs_without_improvement = 0;
            
            // Save best model
            if (config_.save_checkpoints) {
                save_checkpoint(epoch, config_.output_dir + "/best_model.pt");
            }
        } else {
            metrics_.epochs_without_improvement++;
        }
        
        // Progress callback
        if (progress_callback_) {
            progress_callback_(epoch, metrics_);
        }
        
        // Print progress
        if (epoch % 10 == 0) {
            print_progress(epoch);
        }
        
        // Early stopping
        if (metrics_.epochs_without_improvement >= config_.patience) {
            std::cout << "🛑 Early stopping at epoch " << epoch << std::endl;
            break;
        }
        
        // Save checkpoint
        if (config_.save_checkpoints && epoch % config_.checkpoint_frequency == 0) {
            save_checkpoint(epoch, config_.output_dir + "/checkpoint_" + std::to_string(epoch) + ".pt");
        }
    }
    
    std::cout << "🎉 Training completed!" << std::endl;
    std::cout << "🏆 Best validation loss: " << metrics_.best_val_loss 
              << " at epoch " << metrics_.best_epoch << std::endl;
    
    return true;
}

bool TFATrainer::train_from_datasets(const std::vector<std::string>& dataset_paths,
                                    const std::vector<float>& weights,
                                    const std::string& feature_spec_path) {
    std::cout << "📊 Training from " << dataset_paths.size() << " datasets" << std::endl;
    
    // Load and combine all datasets
    std::vector<Bar> combined_bars;
    
    for (size_t i = 0; i < dataset_paths.size(); ++i) {
        // Load bars from CSV (reuse existing CSV loader)
        std::vector<Bar> dataset_bars;
        // TODO: Implement CSV loading or reuse existing loader
        
        // Apply weight by duplicating data
        float weight = (i < weights.size()) ? weights[i] : 1.0f;
        int duplications = static_cast<int>(std::round(weight));
        
        for (int d = 0; d < duplications; ++d) {
            combined_bars.insert(combined_bars.end(), dataset_bars.begin(), dataset_bars.end());
        }
        
        std::cout << "📁 Dataset " << i << ": " << dataset_bars.size() 
                  << " bars, weight: " << weight << std::endl;
    }
    
    // Shuffle combined data
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(combined_bars.begin(), combined_bars.end(), g);
    
    std::cout << "🔀 Combined dataset: " << combined_bars.size() << " bars" << std::endl;
    
    return train_from_bars(combined_bars, feature_spec_path);
}

void TFATrainer::update_realtime(const std::vector<Bar>& new_bars,
                                const std::string& feature_spec_path) {
    if (!config_.enable_realtime) {
        return;
    }
    
    // Add to buffer
    realtime_buffer_.insert(realtime_buffer_.end(), new_bars.begin(), new_bars.end());
    bars_since_update_ += new_bars.size();
    
    // Check if we should update
    if (bars_since_update_ >= config_.realtime_update_frequency) {
        std::cout << "🔄 Real-time model update with " << realtime_buffer_.size() << " bars" << std::endl;
        
        // Extract features and labels from buffer
        auto features = extract_features_from_bars(realtime_buffer_, feature_spec_path);
        auto labels = create_labels_from_bars(realtime_buffer_);
        
        if (features.size(0) > 0 && labels.size(0) > 0) {
            // Set lower learning rate for real-time updates
            for (auto& param_group : optimizer_.param_groups()) {
                static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(config_.realtime_learning_rate);
            }
            
            // Single epoch update
            model_->train();
            train_epoch(features, labels);
            
            // Restore original learning rate
            for (auto& param_group : optimizer_.param_groups()) {
                static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(config_.learning_rate);
            }
        }
        
        // Clear buffer and reset counter
        realtime_buffer_.clear();
        bars_since_update_ = 0;
    }
}

bool TFATrainer::export_torchscript(const std::string& output_path) {
    try {
        // Create output directory
        std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
        
        // Set model to eval mode
        model_->eval();
        
        // Create example input for tracing
        auto example_input = torch::randn({1, config_.sequence_length, config_.feature_dim}).to(device_);
        
        // Trace the model
        auto traced_model = torch::jit::trace(model_, example_input);
        
        // Save the traced model
        traced_model.save(output_path);
        
        std::cout << "💾 Model exported to: " << output_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to export model: " << e.what() << std::endl;
        return false;
    }
}

bool TFATrainer::export_metadata(const std::string& output_path) {
    try {
        nlohmann::json metadata;
        
        // Model metadata (new format)
        metadata["model_type"] = "TFA_CPP_Trained";
        metadata["feature_count"] = config_.feature_dim;
        metadata["sequence_length"] = config_.sequence_length;
        metadata["d_model"] = config_.d_model;
        metadata["nhead"] = config_.nhead;
        metadata["num_layers"] = config_.num_layers;
        metadata["ffn_hidden"] = config_.ffn_hidden;
        
        // Training metadata
        metadata["epochs_completed"] = metrics_.best_epoch;
        metadata["best_val_loss"] = metrics_.best_val_loss;
        metadata["training_device"] = device_.is_cuda() ? "cuda" : "cpu";
        metadata["cpp_trained"] = true;
        
        // Timestamp
        metadata["saved_at"] = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Save metadata
        std::ofstream file(output_path);
        file << metadata.dump(2);
        
        std::cout << "📄 Metadata exported to: " << output_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to export metadata: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Private Methods
// ============================================================================

TFATrainer::TrainingData TFATrainer::prepare_training_data(const torch::Tensor& features, 
                                                           const torch::Tensor& labels) {
    auto n_samples = features.size(0);
    
    // Calculate split indices
    int train_end = static_cast<int>(n_samples * config_.train_split);
    int val_end = train_end + static_cast<int>(n_samples * config_.val_split);
    
    TrainingData data;
    data.X_train = features.slice(0, 0, train_end).to(device_);
    data.y_train = labels.slice(0, 0, train_end).to(device_);
    
    data.X_val = features.slice(0, train_end, val_end).to(device_);
    data.y_val = labels.slice(0, train_end, val_end).to(device_);
    
    data.X_test = features.slice(0, val_end, n_samples).to(device_);
    data.y_test = labels.slice(0, val_end, n_samples).to(device_);
    
    std::cout << "📊 Data splits - Train: " << data.X_train.size(0) 
              << ", Val: " << data.X_val.size(0) 
              << ", Test: " << data.X_test.size(0) << std::endl;
    
    return data;
}

torch::Tensor TFATrainer::create_labels_from_bars(const std::vector<Bar>& bars) {
    std::vector<float> labels;
    labels.reserve(bars.size());
    
    // Create forward-looking labels (predict next bar's direction)
    for (size_t i = 0; i < bars.size() - 1; ++i) {
        double current_close = bars[i].close;
        double next_close = bars[i + 1].close;
        
        // Binary classification: 1 if price goes up, 0 if down
        float label = (next_close > current_close) ? 1.0f : 0.0f;
        labels.push_back(label);
    }
    
    // Last bar gets neutral label
    if (!bars.empty()) {
        labels.push_back(0.5f);
    }
    
    return torch::from_blob(labels.data(), {static_cast<long>(labels.size())}, torch::kFloat).clone();
}

torch::Tensor TFATrainer::extract_features_from_bars(const std::vector<Bar>& bars,
                                                    const std::string& feature_spec_path) {
    try {
        // Use Sentio's existing feature extraction
        auto feature_matrix = build_features_from_spec_json("QQQ", bars, 
            std::ifstream(feature_spec_path).rdbuf());
        
        if (feature_matrix.data.empty()) {
            std::cerr << "❌ Feature extraction failed" << std::endl;
            return torch::empty({0});
        }
        
        // Convert to tensor
        auto tensor = torch::from_blob(feature_matrix.data.data(), 
            {feature_matrix.rows, feature_matrix.cols}, torch::kFloat).clone();
        
        // Create sequences for transformer
        std::vector<torch::Tensor> sequences;
        for (int i = config_.sequence_length - 1; i < tensor.size(0); ++i) {
            auto seq = tensor.slice(0, i - config_.sequence_length + 1, i + 1);
            sequences.push_back(seq);
        }
        
        if (sequences.empty()) {
            return torch::empty({0});
        }
        
        return torch::stack(sequences);
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Feature extraction error: " << e.what() << std::endl;
        return torch::empty({0});
    }
}

void TFATrainer::train_epoch(const torch::Tensor& X_train, const torch::Tensor& y_train) {
    model_->train();
    float total_loss = 0.0f;
    int num_batches = 0;
    
    // Create batches
    auto dataset_size = X_train.size(0);
    for (int i = 0; i < dataset_size; i += config_.batch_size) {
        int batch_end = std::min(i + config_.batch_size, static_cast<int>(dataset_size));
        
        auto batch_X = X_train.slice(0, i, batch_end);
        auto batch_y = y_train.slice(0, i, batch_end);
        
        // Forward pass
        optimizer_.zero_grad();
        auto outputs = model_->forward(batch_X);
        
        // Calculate loss (binary cross entropy)
        auto loss = torch::binary_cross_entropy_with_logits(outputs, batch_y);
        
        // Backward pass
        loss.backward();
        
        // Gradient clipping
        torch::nn::utils::clip_grad_norm_(model_->parameters(), config_.grad_clip);
        
        optimizer_.step();
        
        total_loss += loss.item<float>();
        num_batches++;
    }
    
    float avg_loss = total_loss / num_batches;
    metrics_.train_losses.push_back(avg_loss);
}

float TFATrainer::validate_epoch(const torch::Tensor& X_val, const torch::Tensor& y_val) {
    model_->eval();
    torch::NoGradGuard no_grad;
    
    float total_loss = 0.0f;
    int num_batches = 0;
    
    auto dataset_size = X_val.size(0);
    for (int i = 0; i < dataset_size; i += config_.batch_size) {
        int batch_end = std::min(i + config_.batch_size, static_cast<int>(dataset_size));
        
        auto batch_X = X_val.slice(0, i, batch_end);
        auto batch_y = y_val.slice(0, i, batch_end);
        
        auto outputs = model_->forward(batch_X);
        auto loss = torch::binary_cross_entropy_with_logits(outputs, batch_y);
        
        total_loss += loss.item<float>();
        num_batches++;
    }
    
    return total_loss / num_batches;
}

void TFATrainer::save_checkpoint(int epoch, const std::string& path) {
    try {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        
        torch::serialize::OutputArchive archive;
        model_->save(archive);
        archive.save_to(path);
        
        std::cout << "💾 Checkpoint saved: " << path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to save checkpoint: " << e.what() << std::endl;
    }
}

void TFATrainer::print_progress(int epoch) {
    if (!metrics_.train_losses.empty() && !metrics_.val_losses.empty()) {
        std::cout << "📈 Epoch " << epoch 
                  << " | Train Loss: " << std::fixed << std::setprecision(4) << metrics_.train_losses.back()
                  << " | Val Loss: " << metrics_.val_losses.back()
                  << " | Best: " << metrics_.best_val_loss << std::endl;
    }
}

// Factory function
std::unique_ptr<TFATrainer> create_tfa_trainer(const TFATrainingConfig& config) {
    return std::make_unique<TFATrainer>(config);
}

} // namespace sentio::training
