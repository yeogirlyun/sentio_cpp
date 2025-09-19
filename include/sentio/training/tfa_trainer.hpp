#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <string>
#include <functional>

#include "sentio/core.hpp"
#include "sentio/feature/feature_from_spec.hpp"

namespace sentio::training {

// Training configuration
struct TFATrainingConfig {
    // Model architecture
    int feature_dim = 55;
    int sequence_length = 48;
    int d_model = 128;
    int nhead = 8;
    int num_layers = 3;
    int ffn_hidden = 256;
    float dropout = 0.1f;
    
    // Training parameters
    int batch_size = 32;
    int epochs = 100;
    float learning_rate = 0.001f;
    float weight_decay = 1e-4f;
    float grad_clip = 1.0f;
    
    // Data parameters
    float train_split = 0.8f;
    float val_split = 0.1f;
    float test_split = 0.1f;
    
    // Early stopping
    int patience = 10;
    float min_delta = 1e-4f;
    
    // Real-time training
    bool enable_realtime = false;
    int realtime_update_frequency = 100;  // Update every N bars
    float realtime_learning_rate = 0.0001f;
    
    // Output
    std::string output_dir = "artifacts/TFA/cpp_trained";
    bool save_checkpoints = true;
    int checkpoint_frequency = 10;  // Every N epochs
};

// Training metrics
struct TrainingMetrics {
    std::vector<float> train_losses;
    std::vector<float> val_losses;
    std::vector<float> train_accuracies;
    std::vector<float> val_accuracies;
    
    float best_val_loss = std::numeric_limits<float>::max();
    int best_epoch = 0;
    int epochs_without_improvement = 0;
    
    void reset() {
        train_losses.clear();
        val_losses.clear();
        train_accuracies.clear();
        val_accuracies.clear();
        best_val_loss = std::numeric_limits<float>::max();
        best_epoch = 0;
        epochs_without_improvement = 0;
    }
};

// TFA Transformer Model (PyTorch C++)
class TFATransformerImpl : public torch::nn::Module {
public:
    TFATransformerImpl(int feature_dim, int sequence_length, int d_model, 
                       int nhead, int num_layers, int ffn_hidden, float dropout);
    
    torch::Tensor forward(torch::Tensor x);
    
private:
    int feature_dim_, sequence_length_, d_model_;
    
    // Model components
    torch::nn::Linear input_projection{nullptr};
    torch::nn::TransformerEncoder transformer{nullptr};
    torch::nn::LayerNorm layer_norm{nullptr};
    torch::nn::Linear output_projection{nullptr};
    torch::nn::Dropout dropout{nullptr};
    
    // Positional encoding
    torch::Tensor positional_encoding;
    void create_positional_encoding();
};
TORCH_MODULE(TFATransformer);

// Main TFA Trainer Class
class TFATrainer {
public:
    explicit TFATrainer(const TFATrainingConfig& config = {});
    
    // Training from historical data
    bool train_from_bars(const std::vector<Bar>& bars, 
                        const std::string& feature_spec_path);
    
    // Training from multiple datasets (like Python multi-regime)
    bool train_from_datasets(const std::vector<std::string>& dataset_paths,
                            const std::vector<float>& weights,
                            const std::string& feature_spec_path);
    
    // Real-time training (incremental updates)
    void update_realtime(const std::vector<Bar>& new_bars,
                        const std::string& feature_spec_path);
    
    // Model export
    bool export_torchscript(const std::string& output_path);
    bool export_metadata(const std::string& output_path);
    
    // Training control
    void set_progress_callback(std::function<void(int epoch, const TrainingMetrics&)> callback);
    void stop_training() { should_stop_ = true; }
    
    // Getters
    const TrainingMetrics& get_metrics() const { return metrics_; }
    const TFATrainingConfig& get_config() const { return config_; }
    torch::Device get_device() const { return device_; }
    
private:
    TFATrainingConfig config_;
    TFATransformer model_;
    torch::optim::Adam optimizer_;
    torch::Device device_;
    TrainingMetrics metrics_;
    
    std::function<void(int, const TrainingMetrics&)> progress_callback_;
    bool should_stop_ = false;
    
    // Data preparation
    struct TrainingData {
        torch::Tensor X_train, y_train;
        torch::Tensor X_val, y_val;
        torch::Tensor X_test, y_test;
    };
    
    TrainingData prepare_training_data(const torch::Tensor& features, 
                                     const torch::Tensor& labels);
    
    torch::Tensor create_labels_from_bars(const std::vector<Bar>& bars);
    torch::Tensor extract_features_from_bars(const std::vector<Bar>& bars,
                                            const std::string& feature_spec_path);
    
    // Training loop
    void train_epoch(const torch::Tensor& X_train, const torch::Tensor& y_train);
    float validate_epoch(const torch::Tensor& X_val, const torch::Tensor& y_val);
    
    // Utilities
    void save_checkpoint(int epoch, const std::string& path);
    bool load_checkpoint(const std::string& path);
    void print_progress(int epoch);
    
    // Real-time components
    std::vector<Bar> realtime_buffer_;
    int bars_since_update_ = 0;
};

// Factory function for easy creation
std::unique_ptr<TFATrainer> create_tfa_trainer(const TFATrainingConfig& config = {});

} // namespace sentio::training
