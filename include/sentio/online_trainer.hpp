#pragma once

#include "transformer_strategy_core.hpp"
#include "transformer_model.hpp"
#include <torch/torch.h>
#include <memory>
#include <atomic>
#include <mutex>
#include <vector>
#include <deque>
#include <chrono>
#include <numeric>
#include <thread>
#include <condition_variable>

namespace sentio {

// Simple training sample for online learning
struct TrainingSample {
    torch::Tensor features;
    float label;
    float weight = 1.0f;
    std::chrono::system_clock::time_point timestamp;
    
    TrainingSample(const torch::Tensor& f, float l, float w = 1.0f)
        : features(f.clone()), label(l), weight(w), 
          timestamp(std::chrono::system_clock::now()) {}
};

// Simple replay buffer for experience storage
class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity) : capacity_(capacity) {
        // Note: std::deque doesn't have reserve(), but that's okay
    }
    
    void add_sample(const TrainingSample& sample) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (samples_.size() >= capacity_) {
            samples_.pop_front();
        }
        samples_.push_back(sample);
    }
    
    std::vector<TrainingSample> sample_batch(size_t batch_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::vector<TrainingSample> batch;
        if (samples_.empty()) return batch;
        
        batch.reserve(std::min(batch_size, samples_.size()));
        
        // Simple sampling - take most recent samples
        size_t start_idx = samples_.size() > batch_size ? samples_.size() - batch_size : 0;
        for (size_t i = start_idx; i < samples_.size(); ++i) {
            batch.push_back(samples_[i]);
        }
        
        return batch;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return samples_.size();
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        samples_.clear();
    }

private:
    size_t capacity_;
    std::deque<TrainingSample> samples_;
    mutable std::mutex mutex_;
};

// Simple online trainer for the transformer model
class OnlineTrainer {
public:
    struct OnlineConfig {
        int update_interval_minutes = 60;
        int min_samples_for_update = 1000;
        float base_learning_rate = 0.0001f;
        int replay_buffer_size = 10000;
        bool enable_regime_detection = true;
        float regime_change_threshold = 0.15f;
        int regime_detection_window = 100;
        float validation_threshold = 0.02f;
        int validation_window = 500;
        int max_update_time_seconds = 300;
    };
    
    OnlineTrainer(std::shared_ptr<TransformerModel> model, const OnlineConfig& config)
        : model_(model), config_(config), 
          replay_buffer_(config.replay_buffer_size),
          last_update_time_(std::chrono::system_clock::now()) {}
    
    void add_training_sample(const torch::Tensor& features, float label, float weight = 1.0f) {
        TrainingSample sample(features, label, weight);
        replay_buffer_.add_sample(sample);
        samples_since_last_update_++;
    }
    
    bool should_update_model() const {
        auto now = std::chrono::system_clock::now();
        auto time_since_update = std::chrono::duration_cast<std::chrono::minutes>(
            now - last_update_time_).count();
        
        bool time_condition = time_since_update >= config_.update_interval_minutes;
        bool sample_condition = samples_since_last_update_ >= config_.min_samples_for_update;
        bool buffer_condition = static_cast<int>(replay_buffer_.size()) >= config_.min_samples_for_update;
        
        return time_condition && sample_condition && buffer_condition;
    }
    
    UpdateResult update_model() {
        // Simple implementation - just return success for now
        // In a full implementation, this would perform actual model updates
        last_update_time_ = std::chrono::system_clock::now();
        samples_since_last_update_ = 0;
        
        UpdateResult result;
        result.success = true;
        result.error_message = "";
        result.update_duration = std::chrono::milliseconds(100);
        
        return result;
    }
    
    bool detect_regime_change() const {
        // Simple regime detection - could be enhanced
        return false; // For now, always return false
    }
    
    void adapt_to_regime_change() {
        // Reset some internal state for regime adaptation
        samples_since_last_update_ = config_.min_samples_for_update;
    }
    
    PerformanceMetrics get_training_metrics() const {
        PerformanceMetrics metrics;
        metrics.samples_processed = replay_buffer_.size();
        metrics.is_training_active = false;
        metrics.training_loss = 0.0f;
        return metrics;
    }

private:
    std::shared_ptr<TransformerModel> model_;
    OnlineConfig config_;
    ReplayBuffer replay_buffer_;
    
    std::chrono::system_clock::time_point last_update_time_;
    std::atomic<int> samples_since_last_update_{0};
};

} // namespace sentio
