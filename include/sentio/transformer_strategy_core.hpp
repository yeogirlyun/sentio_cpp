#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <shared_mutex>
#include <chrono>
#include <string>
#include <unordered_map>
#include <torch/torch.h>

namespace sentio {

// Forward declarations
struct Bar;
class Fill;
class BaseStrategy;
struct StrategySignal;

// ==================== Core Data Structures ====================

struct PriceFeatures {
    std::vector<float> ohlc_normalized;     // 4 features
    std::vector<float> returns;             // 5 features (1m, 5m, 15m, 1h, 4h)
    std::vector<float> log_returns;         // 5 features
    std::vector<float> moving_averages;     // 8 features (SMA/EMA 5,10,20,50)
    std::vector<float> bollinger_bands;     // 3 features (upper, lower, %B)
    std::vector<float> rsi_family;          // 4 features (RSI 14, Stoch RSI)
    std::vector<float> momentum;            // 6 features (ROC, Williams %R, etc.)
    std::vector<float> volatility;          // 5 features (ATR, realized vol, etc.)
    
    static constexpr int TOTAL_FEATURES = 40;
};

struct VolumeFeatures {
    std::vector<float> volume_indicators;   // 8 features (VWAP, OBV, etc.)
    std::vector<float> volume_ratios;       // 4 features (vol/avg_vol ratios)
    std::vector<float> price_volume;        // 4 features (PVT, MFI, etc.)
    std::vector<float> volume_profile;      // 4 features (VPOC, VAH, VAL, etc.)
    
    static constexpr int TOTAL_FEATURES = 20;
};

struct MicrostructureFeatures {
    std::vector<float> spread_metrics;      // 5 features (bid-ask spread analysis)
    std::vector<float> order_flow;          // 8 features (tick direction, etc.)
    std::vector<float> market_impact;       // 4 features (Kyle's lambda, etc.)
    std::vector<float> liquidity_metrics;   // 4 features (market depth, etc.)
    std::vector<float> regime_indicators;   // 4 features (volatility regime, etc.)
    
    static constexpr int TOTAL_FEATURES = 25;
};

struct CrossAssetFeatures {
    std::vector<float> correlation_features; // 5 features (SPY, VIX correlation)
    std::vector<float> sector_rotation;      // 5 features (sector momentum)
    std::vector<float> macro_indicators;     // 5 features (yield curve, etc.)
    
    static constexpr int TOTAL_FEATURES = 15;
};

struct TemporalFeatures {
    std::vector<float> time_of_day;         // 8 features (hour encoding)
    std::vector<float> day_of_week;         // 7 features (weekday encoding)
    std::vector<float> monthly_seasonal;    // 12 features (month encoding)
    std::vector<float> market_session;      // 1 feature (RTH/ETH indicator)
    
    static constexpr int TOTAL_FEATURES = 28;
};

using FeatureVector = torch::Tensor;
using TransformerFeatureMatrix = torch::Tensor;

// ==================== Configuration Structures ====================

struct TransformerConfig {
    // Model architecture
    int feature_dim = 128;
    int sequence_length = 64;
    int d_model = 256;
    int num_heads = 8;
    int num_layers = 6;
    int ffn_hidden = 1024;
    float dropout = 0.1f;
    
    // Performance requirements
    float max_inference_latency_ms = 1.0f;
    float max_memory_usage_mb = 1024.0f;
    
    // Training configuration
    struct Training {
        struct Offline {
            int batch_size = 256;
            float learning_rate = 0.001f;
            int max_epochs = 1000;
            int patience = 50;
            float min_delta = 1e-6f;
            std::string optimizer = "AdamW";
            float weight_decay = 1e-4f;
        } offline;
        
        struct Online {
            int update_interval_minutes = 60;
            float learning_rate = 0.0001f;
            int replay_buffer_size = 10000;
            bool enable_regime_detection = true;
            float regime_change_threshold = 0.15f;
            int min_samples_for_update = 1000;
        } online;
    } training;
    
    // Feature configuration
    struct Features {
        enum class NormalizationMethod {
            Z_SCORE, MIN_MAX, ROBUST, QUANTILE_UNIFORM
        };
        NormalizationMethod normalization = NormalizationMethod::Z_SCORE;
        float decay_factor = 0.999f;
    } features;
};

// ==================== Performance Metrics ====================

struct PerformanceMetrics {
    float avg_inference_latency_ms = 0.0f;
    float p95_inference_latency_ms = 0.0f;
    float p99_inference_latency_ms = 0.0f;
    float recent_accuracy = 0.0f;
    float rolling_sharpe_ratio = 0.0f;
    float current_drawdown = 0.0f;
    float memory_usage_mb = 0.0f;
    bool is_training_active = false;
    float training_loss = 0.0f;
    int samples_processed = 0;
};

struct ValidationMetrics {
    float directional_accuracy = 0.0f;
    float sharpe_ratio = 0.0f;
    float max_drawdown = 0.0f;
    float win_rate = 0.0f;
    float profit_factor = 0.0f;
    bool passes_validation = false;
};

struct TrainingResult {
    bool success = false;
    std::string model_path;
    ValidationMetrics validation_metrics;
    std::chrono::system_clock::time_point training_end_time;
    int total_epochs = 0;
};

// ==================== Model Status ====================

enum class ModelStatus {
    UNINITIALIZED,
    LOADING,
    READY,
    TRAINING,
    UPDATING,
    ERROR,
    DISABLED
};

struct UpdateResult {
    bool success = false;
    std::string error_message;
    ValidationMetrics post_update_metrics;
    std::chrono::milliseconds update_duration{0};
};

// ==================== Risk Management ====================

struct RiskLimits {
    float max_position_size = 1.0f;
    float max_daily_trades = 100.0f;
    float max_drawdown_threshold = 0.10f;
    float min_confidence_threshold = 0.6f;
};

struct Alert {
    std::string metric_name;
    float current_value;
    float threshold;
    std::chrono::system_clock::time_point timestamp;
    std::string message;
};

// ==================== Running Statistics for Feature Normalization ====================

class RunningStats {
public:
    RunningStats(float decay_factor = 0.999f) : decay_factor_(decay_factor) {}
    
    void update(float value) {
        if (!initialized_) {
            mean_ = value;
            var_ = 0.0f;
            min_ = max_ = value;
            initialized_ = true;
            count_ = 1;
        } else {
            // Exponential moving average
            mean_ = decay_factor_ * mean_ + (1.0f - decay_factor_) * value;
            float delta = value - mean_;
            var_ = decay_factor_ * var_ + (1.0f - decay_factor_) * delta * delta;
            min_ = std::min(min_, value);
            max_ = std::max(max_, value);
            count_++;
        }
    }
    
    float mean() const { return mean_; }
    float std() const { return std::sqrt(var_); }
    float min() const { return min_; }
    float max() const { return max_; }
    int count() const { return count_; }
    bool is_initialized() const { return initialized_; }
    
private:
    float decay_factor_;
    float mean_ = 0.0f;
    float var_ = 0.0f;
    float min_ = 0.0f;
    float max_ = 0.0f;
    int count_ = 0;
    bool initialized_ = false;
};

// ==================== Core Constants ====================

struct LatencyRequirements {
    static constexpr int MAX_INFERENCE_LATENCY_US = 1000;    // 1ms
    static constexpr int TARGET_INFERENCE_LATENCY_US = 500;  // 0.5ms
    static constexpr int MAX_FEATURE_GEN_LATENCY_US = 500;   // 0.5ms
    static constexpr int MAX_MODEL_UPDATE_TIME_S = 300;      // 5 minutes
    static constexpr int MAX_MEMORY_USAGE_MB = 1024;         // 1GB
    static constexpr int MAX_GPU_MEMORY_MB = 2048;           // 2GB
};

struct AccuracyRequirements {
    static constexpr float MIN_DIRECTIONAL_ACCURACY = 0.52f;
    static constexpr float TARGET_DIRECTIONAL_ACCURACY = 0.55f;
    static constexpr float MIN_SHARPE_RATIO = 1.0f;
    static constexpr float TARGET_SHARPE_RATIO = 2.0f;
    static constexpr float MAX_DRAWDOWN = 0.15f;
    static constexpr float MIN_WIN_RATE = 0.45f;
};

} // namespace sentio
