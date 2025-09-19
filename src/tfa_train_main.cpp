#include "sentio/training/tfa_trainer.hpp"
#include "sentio/csv_loader.hpp"
#include "sentio/time_utils.hpp"

#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>

using namespace sentio::training;

void print_usage(const char* program_name) {
    std::cout << "🚀 Sentio C++ TFA Trainer\n" << std::endl;
    std::cout << "Usage: " << program_name << " [options]\n" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --data PATH              Path to training data CSV" << std::endl;
    std::cout << "  --feature-spec PATH      Path to feature specification JSON" << std::endl;
    std::cout << "  --output DIR             Output directory (default: artifacts/TFA/cpp_trained)" << std::endl;
    std::cout << "  --epochs N               Number of training epochs (default: 100)" << std::endl;
    std::cout << "  --batch-size N           Batch size (default: 32)" << std::endl;
    std::cout << "  --learning-rate FLOAT    Learning rate (default: 0.001)" << std::endl;
    std::cout << "  --sequence-length N      Sequence length (default: 48)" << std::endl;
    std::cout << "  --d-model N              Model dimension (default: 128)" << std::endl;
    std::cout << "  --nhead N                Number of attention heads (default: 8)" << std::endl;
    std::cout << "  --num-layers N           Number of transformer layers (default: 3)" << std::endl;
    std::cout << "  --enable-realtime        Enable real-time training mode" << std::endl;
    std::cout << "  --cuda                   Force CUDA usage (if available)" << std::endl;
    std::cout << "  --help                   Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  # Basic training" << std::endl;
    std::cout << "  " << program_name << " --data data/equities/QQQ_RTH_NH.csv --feature-spec configs/features/feature_spec_55_minimal.json" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Advanced training with custom parameters" << std::endl;
    std::cout << "  " << program_name << " --data data/equities/QQQ_RTH_NH.csv --feature-spec configs/features/feature_spec_55_minimal.json \\" << std::endl;
    std::cout << "                     --epochs 200 --batch-size 64 --learning-rate 0.0005 --d-model 256 --num-layers 6" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Real-time training mode" << std::endl;
    std::cout << "  " << program_name << " --data data/equities/QQQ_RTH_NH.csv --feature-spec configs/features/feature_spec_55_minimal.json \\" << std::endl;
    std::cout << "                     --enable-realtime --output artifacts/TFA/realtime" << std::endl;
}

class ProgressReporter {
public:
    ProgressReporter() : start_time_(std::chrono::steady_clock::now()) {}
    
    void operator()(int epoch, const TrainingMetrics& metrics) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        
        if (epoch % 10 == 0 || epoch < 10) {
            std::cout << "\n📊 Training Progress Report" << std::endl;
            std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
            std::cout << "🕒 Epoch: " << std::setw(4) << epoch 
                      << " | ⏱️  Elapsed: " << format_duration(elapsed) << std::endl;
            
            if (!metrics.train_losses.empty()) {
                std::cout << "📉 Train Loss: " << std::fixed << std::setprecision(6) 
                          << metrics.train_losses.back() << std::endl;
            }
            
            if (!metrics.val_losses.empty()) {
                std::cout << "📊 Val Loss:   " << std::fixed << std::setprecision(6) 
                          << metrics.val_losses.back() << std::endl;
            }
            
            std::cout << "🏆 Best Loss:  " << std::fixed << std::setprecision(6) 
                      << metrics.best_val_loss << " (epoch " << metrics.best_epoch << ")" << std::endl;
            
            std::cout << "⏳ No Improve: " << metrics.epochs_without_improvement << " epochs" << std::endl;
            
            // Progress bar
            if (epoch > 0) {
                float progress = static_cast<float>(epoch) / 100.0f;  // Assume max 100 for display
                progress = std::min(progress, 1.0f);
                
                int bar_width = 50;
                int filled = static_cast<int>(progress * bar_width);
                
                std::cout << "📈 Progress: [";
                for (int i = 0; i < bar_width; ++i) {
                    if (i < filled) std::cout << "█";
                    else std::cout << "░";
                }
                std::cout << "] " << std::fixed << std::setprecision(1) 
                          << (progress * 100.0f) << "%" << std::endl;
            }
            
            std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;
        }
    }
    
private:
    std::chrono::steady_clock::time_point start_time_;
    
    std::string format_duration(int64_t seconds) {
        int hours = seconds / 3600;
        int minutes = (seconds % 3600) / 60;
        int secs = seconds % 60;
        
        std::ostringstream oss;
        if (hours > 0) {
            oss << hours << "h " << minutes << "m " << secs << "s";
        } else if (minutes > 0) {
            oss << minutes << "m " << secs << "s";
        } else {
            oss << secs << "s";
        }
        return oss.str();
    }
};

int main(int argc, char* argv[]) {
    std::cout << "🚀 Sentio C++ TFA Trainer" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    
    // Parse command line arguments
    std::string data_path;
    std::string feature_spec_path;
    TFATrainingConfig config;
    bool force_cuda = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--data" && i + 1 < argc) {
            data_path = argv[++i];
        } else if (arg == "--feature-spec" && i + 1 < argc) {
            feature_spec_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--learning-rate" && i + 1 < argc) {
            config.learning_rate = std::stof(argv[++i]);
        } else if (arg == "--sequence-length" && i + 1 < argc) {
            config.sequence_length = std::stoi(argv[++i]);
        } else if (arg == "--d-model" && i + 1 < argc) {
            config.d_model = std::stoi(argv[++i]);
        } else if (arg == "--nhead" && i + 1 < argc) {
            config.nhead = std::stoi(argv[++i]);
        } else if (arg == "--num-layers" && i + 1 < argc) {
            config.num_layers = std::stoi(argv[++i]);
        } else if (arg == "--enable-realtime") {
            config.enable_realtime = true;
        } else if (arg == "--cuda") {
            force_cuda = true;
        } else {
            std::cerr << "❌ Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate required arguments
    if (data_path.empty() || feature_spec_path.empty()) {
        std::cerr << "❌ Missing required arguments: --data and --feature-spec" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // Check if files exist
    if (!std::filesystem::exists(data_path)) {
        std::cerr << "❌ Data file not found: " << data_path << std::endl;
        return 1;
    }
    
    if (!std::filesystem::exists(feature_spec_path)) {
        std::cerr << "❌ Feature spec file not found: " << feature_spec_path << std::endl;
        return 1;
    }
    
    // Print configuration
    std::cout << "📋 Training Configuration:" << std::endl;
    std::cout << "  📁 Data: " << data_path << std::endl;
    std::cout << "  🔧 Feature Spec: " << feature_spec_path << std::endl;
    std::cout << "  📂 Output: " << config.output_dir << std::endl;
    std::cout << "  🔄 Epochs: " << config.epochs << std::endl;
    std::cout << "  📦 Batch Size: " << config.batch_size << std::endl;
    std::cout << "  📈 Learning Rate: " << config.learning_rate << std::endl;
    std::cout << "  📏 Sequence Length: " << config.sequence_length << std::endl;
    std::cout << "  🏗️  Model Dim: " << config.d_model << std::endl;
    std::cout << "  🧠 Attention Heads: " << config.nhead << std::endl;
    std::cout << "  🏢 Layers: " << config.num_layers << std::endl;
    std::cout << "  ⚡ Real-time: " << (config.enable_realtime ? "Enabled" : "Disabled") << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;
    
    try {
        // Load training data
        std::cout << "📊 Loading training data..." << std::endl;
        auto bars = sentio::load_csv_bars(data_path);
        std::cout << "✅ Loaded " << bars.size() << " bars" << std::endl;
        
        if (bars.empty()) {
            std::cerr << "❌ No data loaded from file" << std::endl;
            return 1;
        }
        
        // Create trainer
        std::cout << "🏗️  Initializing trainer..." << std::endl;
        auto trainer = create_tfa_trainer(config);
        
        // Set progress callback
        ProgressReporter reporter;
        trainer->set_progress_callback(reporter);
        
        // Start training
        std::cout << "🚀 Starting training..." << std::endl;
        auto start_time = std::chrono::steady_clock::now();
        
        bool success = trainer->train_from_bars(bars, feature_spec_path);
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        
        if (success) {
            std::cout << "\n🎉 Training completed successfully!" << std::endl;
            std::cout << "⏱️  Total training time: " << duration << " seconds" << std::endl;
            
            // Export model
            std::string model_path = config.output_dir + "/model.pt";
            std::string metadata_path = config.output_dir + "/model.meta.json";
            
            std::cout << "💾 Exporting model..." << std::endl;
            trainer->export_torchscript(model_path);
            trainer->export_metadata(metadata_path);
            
            // Print final metrics
            const auto& metrics = trainer->get_metrics();
            std::cout << "\n📊 Final Training Metrics:" << std::endl;
            std::cout << "  🏆 Best Validation Loss: " << std::fixed << std::setprecision(6) 
                      << metrics.best_val_loss << std::endl;
            std::cout << "  🎯 Best Epoch: " << metrics.best_epoch << std::endl;
            std::cout << "  📈 Total Epochs: " << metrics.train_losses.size() << std::endl;
            
            std::cout << "\n✅ Model ready for use in Sentio!" << std::endl;
            std::cout << "📁 Model files:" << std::endl;
            std::cout << "  🤖 TorchScript: " << model_path << std::endl;
            std::cout << "  📄 Metadata: " << metadata_path << std::endl;
            
        } else {
            std::cerr << "❌ Training failed!" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
