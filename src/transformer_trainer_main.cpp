#include "sentio/transformer_model.hpp"
#include "sentio/feature_pipeline.hpp"
#include "sentio/online_trainer.hpp"
#include "sentio/transformer_strategy_core.hpp"
#include "sentio/core.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <random>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <limits>
#include <numeric>

namespace sentio {

struct TrainingConfig {
    std::string data_path = "data/equities/QQQ_1min.csv";
    std::string output_dir = "artifacts/Transformer/v1";
    int epochs = 20;
    int batch_size = 32;
    int sequence_length = 64;
    int feature_dim = 128;
    int d_model = 256;
    int num_heads = 8;
    int num_layers = 6;
    float learning_rate = 0.001f;
    float dropout = 0.1f;
    bool use_cuda = false;
    int validation_split_percent = 20;
    int progress_update_frequency = 10; // Update progress every N batches
};

class TransformerTrainer {
public:
    explicit TransformerTrainer(const TrainingConfig& config) : config_(config) {
        // Initialize model configuration
        model_config_.feature_dim = config.feature_dim;
        model_config_.sequence_length = config.sequence_length;
        model_config_.d_model = config.d_model;
        model_config_.num_heads = config.num_heads;
        model_config_.num_layers = config.num_layers;
        model_config_.dropout = config.dropout;
        
        // Initialize feature configuration
        feature_config_.normalization = TransformerConfig::Features::NormalizationMethod::Z_SCORE;
        feature_config_.decay_factor = 0.999f;
        
        std::cout << "Transformer Trainer initialized with:" << std::endl;
        std::cout << "  Feature dim: " << config.feature_dim << std::endl;
        std::cout << "  Sequence length: " << config.sequence_length << std::endl;
        std::cout << "  Model dim: " << config.d_model << std::endl;
        std::cout << "  Attention heads: " << config.num_heads << std::endl;
        std::cout << "  Layers: " << config.num_layers << std::endl;
        std::cout << "  Learning rate: " << config.learning_rate << std::endl;
    }
    
    bool load_market_data() {
        std::cout << "Loading market data from: " << config_.data_path << std::endl;
        
        if (!std::filesystem::exists(config_.data_path)) {
            std::cerr << "Error: Data file not found: " << config_.data_path << std::endl;
            return false;
        }
        
        std::ifstream file(config_.data_path);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open data file: " << config_.data_path << std::endl;
            return false;
        }
        
        std::string line;
        bool first_line = true;
        
        while (std::getline(file, line)) {
            if (first_line) {
                first_line = false;
                continue; // Skip header
            }
            
            std::istringstream ss(line);
            std::string token;
            std::vector<std::string> tokens;
            
            while (std::getline(ss, token, ',')) {
                tokens.push_back(token);
            }
            
            if (tokens.size() >= 5) {
                try {
                    Bar bar;
                    bar.open = std::stof(tokens[1]);   // Assuming: timestamp,open,high,low,close,volume
                    bar.high = std::stof(tokens[2]);
                    bar.low = std::stof(tokens[3]);
                    bar.close = std::stof(tokens[4]);
                    bar.volume = tokens.size() > 5 ? std::stof(tokens[5]) : 1000.0f;
                    bar.ts_utc_epoch = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count(); // Simplified
                    
                    market_data_.push_back(bar);
                } catch (const std::exception& e) {
                    // Skip invalid lines
                    continue;
                }
            }
        }
        
        std::cout << "Loaded " << market_data_.size() << " bars of market data" << std::endl;
        return !market_data_.empty();
    }
    
    void prepare_training_data() {
        std::cout << "Preparing training data with corrected dimensions..." << std::endl;
        
        // Initialize feature pipeline
        feature_pipeline_ = std::make_unique<FeaturePipeline>(feature_config_);
        
        // Generate features and labels
        for (size_t i = config_.sequence_length; i < market_data_.size() - 1; ++i) {
            // Generate feature sequence with proper dimensions
            std::vector<torch::Tensor> sequence_features;
            sequence_features.reserve(config_.sequence_length);
            
            // Generate features for each bar in the sequence
            for (int j = i - config_.sequence_length; j < static_cast<int>(i); ++j) {
                std::vector<Bar> single_bar = {market_data_[j]};
                auto bar_features = feature_pipeline_->generate_features(single_bar);
                
                // Ensure we get [128] features per bar
                auto squeezed = bar_features.squeeze(0);
                if (squeezed.size(0) != config_.feature_dim) {
                    std::cerr << "Feature dimension error: expected " << config_.feature_dim 
                              << ", got " << squeezed.size(0) << std::endl;
                    continue;
                }
                
                sequence_features.push_back(squeezed);
            }
            
            if (sequence_features.size() != config_.sequence_length) {
                continue; // Skip incomplete sequences
            }
            
            // Stack into [sequence_length, feature_dim] tensor
            auto features = torch::stack(sequence_features, 0);
            
            // Validate tensor shape
            if (features.sizes() != torch::IntArrayRef({config_.sequence_length, config_.feature_dim})) {
                std::cerr << "Training tensor shape error: " << features.sizes() << std::endl;
                continue;
            }
            
            // Calculate label (next bar return)
            float current_price = market_data_[i].close;
            float next_price = market_data_[i + 1].close;
            float label = (next_price - current_price) / current_price;
            
            training_samples_.emplace_back(features, label);
            
            // Debug first sample
            if (training_samples_.size() == 1) {
                std::cout << "First training sample shape: " << features.sizes() << std::endl;
                std::cout << "Feature dim: " << config_.feature_dim << std::endl;
                std::cout << "Sequence length: " << config_.sequence_length << std::endl;
            }
        }
        
        std::cout << "Generated " << training_samples_.size() << " training samples" << std::endl;
        
        // Split into training and validation
        size_t total_samples = training_samples_.size();
        size_t val_samples = (total_samples * config_.validation_split_percent) / 100;
        size_t train_samples = total_samples - val_samples;
        
        // Simple split - last 20% for validation
        train_samples_.assign(training_samples_.begin(), training_samples_.begin() + train_samples);
        val_samples_.assign(training_samples_.begin() + train_samples, training_samples_.end());
        
        std::cout << "Training samples: " << train_samples_.size() << std::endl;
        std::cout << "Validation samples: " << val_samples_.size() << std::endl;
    }
    
    bool train_model() {
        std::cout << "\nStarting model training..." << std::endl;
        std::cout << "=" << std::string(50, '=') << std::endl;
        
        // Create model
        model_ = std::make_shared<TransformerModel>(model_config_);
        std::cout << "Model created with " << model_->get_parameter_count() << " parameters" << std::endl;
        
        // Setup optimizer
        torch::optim::AdamW optimizer(model_->parameters(), torch::optim::AdamWOptions(config_.learning_rate));
        torch::nn::MSELoss criterion;
        
        // Training loop
        auto training_start = std::chrono::high_resolution_clock::now();
        
        for (int epoch = 0; epoch < config_.epochs; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            
            // Training phase
            model_->train();
            float train_loss = 0.0f;
            int train_batches = 0;
            
            // Shuffle training data
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(train_samples_.begin(), train_samples_.end(), g);
            
            // Process training batches
            for (size_t i = 0; i < train_samples_.size(); i += config_.batch_size) {
                size_t batch_end = std::min(i + config_.batch_size, train_samples_.size());
                
                // Create batch
                std::vector<torch::Tensor> batch_features;
                std::vector<float> batch_labels;
                
                for (size_t j = i; j < batch_end; ++j) {
                    batch_features.push_back(train_samples_[j].features);
                    batch_labels.push_back(train_samples_[j].label);
                }
                
                if (batch_features.empty()) continue;
                
                // Stack features into batch tensor
                auto features_batch = torch::stack(batch_features);
                auto labels_batch = torch::tensor(batch_labels);
                
                // Forward pass
                optimizer.zero_grad();
                auto predictions = model_->forward(features_batch);
                auto loss = criterion(predictions.squeeze(), labels_batch);
                
                // Backward pass
                loss.backward();
                
                // Gradient clipping
                torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);
                
                optimizer.step();
                
                train_loss += loss.item<float>();
                train_batches++;
                
                // Progress update within epoch
                if (train_batches % config_.progress_update_frequency == 0) {
                    float progress = (float)(i + batch_end - i) / train_samples_.size() * 100.0f;
                    std::cout << "\rEpoch " << (epoch + 1) << "/" << config_.epochs 
                              << " - Progress: " << std::fixed << std::setprecision(1) << progress << "% "
                              << "- Batch Loss: " << std::fixed << std::setprecision(6) << loss.item<float>()
                              << std::flush;
                }
            }
            
            float avg_train_loss = train_loss / train_batches;
            
            // Validation phase
            model_->eval();
            float val_loss = 0.0f;
            int val_batches = 0;
            
            torch::NoGradGuard no_grad;
            for (size_t i = 0; i < val_samples_.size(); i += config_.batch_size) {
                size_t batch_end = std::min(i + config_.batch_size, val_samples_.size());
                
                std::vector<torch::Tensor> batch_features;
                std::vector<float> batch_labels;
                
                for (size_t j = i; j < batch_end; ++j) {
                    batch_features.push_back(val_samples_[j].features);
                    batch_labels.push_back(val_samples_[j].label);
                }
                
                if (batch_features.empty()) continue;
                
                auto features_batch = torch::stack(batch_features);
                auto labels_batch = torch::tensor(batch_labels);
                
                auto predictions = model_->forward(features_batch);
                auto loss = criterion(predictions.squeeze(), labels_batch);
                
                val_loss += loss.item<float>();
                val_batches++;
            }
            
            float avg_val_loss = val_loss / val_batches;
            
            // Calculate epoch time
            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
            
            // Print epoch summary
            std::cout << "\rEpoch " << (epoch + 1) << "/" << config_.epochs 
                      << " - Train Loss: " << std::fixed << std::setprecision(6) << avg_train_loss
                      << " - Val Loss: " << std::fixed << std::setprecision(6) << avg_val_loss
                      << " - Time: " << epoch_duration.count() << "ms" << std::endl;
            
            // Save best model
            if (avg_val_loss < best_val_loss_) {
                best_val_loss_ = avg_val_loss;
                std::cout << "  → New best validation loss: " << std::fixed << std::setprecision(6) << best_val_loss_ << std::endl;
            }
        }
        
        auto training_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(training_end - training_start);
        
        std::cout << "\nTraining completed in " << total_duration.count() << " seconds" << std::endl;
        std::cout << "Best validation loss: " << std::fixed << std::setprecision(6) << best_val_loss_ << std::endl;
        
        return true;
    }
    
    bool save_model() {
        std::cout << "\nSaving trained model..." << std::endl;
        
        // Create output directory
        std::filesystem::create_directories(config_.output_dir);
        
        std::string model_path = config_.output_dir + "/model.pt";
        std::string metadata_path = config_.output_dir + "/model.meta.json";
        
        try {
            // Save model
            model_->save_model(model_path);
            
            // Create metadata
            std::ofstream meta_file(metadata_path);
            meta_file << "{\n";
            meta_file << "  \"feature_dim\": " << config_.feature_dim << ",\n";
            meta_file << "  \"sequence_length\": " << config_.sequence_length << ",\n";
            meta_file << "  \"d_model\": " << config_.d_model << ",\n";
            meta_file << "  \"num_heads\": " << config_.num_heads << ",\n";
            meta_file << "  \"num_layers\": " << config_.num_layers << ",\n";
            meta_file << "  \"model_type\": \"Transformer\",\n";
            meta_file << "  \"version\": \"v1\",\n";
            meta_file << "  \"best_val_loss\": " << best_val_loss_ << ",\n";
            meta_file << "  \"training_samples\": " << train_samples_.size() << ",\n";
            meta_file << "  \"validation_samples\": " << val_samples_.size() << "\n";
            meta_file << "}\n";
            meta_file.close();
            
            std::cout << "Model saved successfully:" << std::endl;
            std::cout << "  Model file: " << model_path << std::endl;
            std::cout << "  Metadata: " << metadata_path << std::endl;
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error saving model: " << e.what() << std::endl;
            return false;
        }
    }

private:
    TrainingConfig config_;
    TransformerConfig model_config_;
    TransformerConfig::Features feature_config_;
    
    std::vector<Bar> market_data_;
    std::vector<TrainingSample> training_samples_;
    std::vector<TrainingSample> train_samples_;
    std::vector<TrainingSample> val_samples_;
    
    std::shared_ptr<TransformerModel> model_;
    std::unique_ptr<FeaturePipeline> feature_pipeline_;
    
    float best_val_loss_ = std::numeric_limits<float>::max();
};

} // namespace sentio

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --data PATH          Path to market data CSV file (default: data/equities/QQQ_1min.csv)\n";
    std::cout << "  --output DIR         Output directory for trained model (default: artifacts/Transformer/v1)\n";
    std::cout << "  --epochs N           Number of training epochs (default: 20)\n";
    std::cout << "  --batch-size N       Batch size for training (default: 32)\n";
    std::cout << "  --sequence-length N  Sequence length for transformer (default: 64)\n";
    std::cout << "  --feature-dim N      Feature dimension (default: 128)\n";
    std::cout << "  --d-model N          Model dimension (default: 256)\n";
    std::cout << "  --num-heads N        Number of attention heads (default: 8)\n";
    std::cout << "  --num-layers N       Number of transformer layers (default: 6)\n";
    std::cout << "  --learning-rate F    Learning rate (default: 0.001)\n";
    std::cout << "  --help               Show this help message\n";
}

int main(int argc, char* argv[]) {
    sentio::TrainingConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--data" && i + 1 < argc) {
            config.data_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--sequence-length" && i + 1 < argc) {
            config.sequence_length = std::stoi(argv[++i]);
        } else if (arg == "--feature-dim" && i + 1 < argc) {
            config.feature_dim = std::stoi(argv[++i]);
        } else if (arg == "--d-model" && i + 1 < argc) {
            config.d_model = std::stoi(argv[++i]);
        } else if (arg == "--num-heads" && i + 1 < argc) {
            config.num_heads = std::stoi(argv[++i]);
        } else if (arg == "--num-layers" && i + 1 < argc) {
            config.num_layers = std::stoi(argv[++i]);
        } else if (arg == "--learning-rate" && i + 1 < argc) {
            config.learning_rate = std::stof(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    std::cout << "Sentio Transformer Strategy Training" << std::endl;
    std::cout << "====================================" << std::endl;
    
    try {
        sentio::TransformerTrainer trainer(config);
        
        // Load market data
        if (!trainer.load_market_data()) {
            std::cerr << "Failed to load market data" << std::endl;
            return 1;
        }
        
        // Prepare training data
        trainer.prepare_training_data();
        
        // Train model
        if (!trainer.train_model()) {
            std::cerr << "Training failed" << std::endl;
            return 1;
        }
        
        // Save model
        if (!trainer.save_model()) {
            std::cerr << "Failed to save model" << std::endl;
            return 1;
        }
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "TRAINING COMPLETE!" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Model saved to: " << config.output_dir << std::endl;
        std::cout << "You can now test the strategy with:" << std::endl;
        std::cout << "  ./sencli strattest transformer --mode historical --blocks 10" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
