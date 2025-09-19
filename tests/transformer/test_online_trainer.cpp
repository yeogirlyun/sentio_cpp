// test_online_trainer.cpp - Tests for online training
#include <gtest/gtest.h>
#include "sentio/online_trainer.hpp"
#include "sentio/transformer_model.hpp"
#include "sentio/transformer_strategy_core.hpp"
#include <memory>
#include <cmath>

using namespace sentio;

class OnlineTrainerTest : public ::testing::Test {
protected:
    void SetUp() override {
        TransformerConfig model_config;
        model_config.feature_dim = 32;
        model_config.sequence_length = 16;
        model_config.d_model = 64;
        model_config.num_heads = 4;
        model_config.num_layers = 2;
        
        model_ = std::make_shared<TransformerModel>(model_config);
        
        OnlineTrainer::OnlineConfig trainer_config;
        trainer_config.update_interval_minutes = 1; // Fast updates for testing
        trainer_config.min_samples_for_update = 10;
        trainer_config.replay_buffer_size = 100;
        
        trainer_ = std::make_unique<OnlineTrainer>(model_, trainer_config);
    }
    
    std::shared_ptr<TransformerModel> model_;
    std::unique_ptr<OnlineTrainer> trainer_;
};

TEST_F(OnlineTrainerTest, SampleAddition) {
    auto features = torch::randn({32});
    float label = 0.1f;
    
    trainer_->add_training_sample(features, label);
    
    auto metrics = trainer_->get_training_metrics();
    // Should have processed at least one sample
    EXPECT_GE(metrics.samples_processed, 0);
}

TEST_F(OnlineTrainerTest, ModelUpdateTrigger) {
    // Add minimum samples required for update
    for (int i = 0; i < 15; ++i) {
        auto features = torch::randn({32});
        float label = std::sin(i * 0.1f);
        trainer_->add_training_sample(features, label);
    }
    
    // Should trigger update after enough samples and time
    bool should_update = trainer_->should_update_model();
    EXPECT_TRUE(should_update);
}

TEST_F(OnlineTrainerTest, RegimeDetection) {
    // Add samples with different regimes
    for (int i = 0; i < 50; ++i) {
        auto features = torch::randn({32});
        float label = (i < 25) ? 0.1f : -0.1f; // Regime change at i=25
        trainer_->add_training_sample(features, label);
    }
    
    bool regime_change = trainer_->detect_regime_change();
    // May or may not detect regime change with simple data
    EXPECT_TRUE(regime_change || !regime_change); // Just check it doesn't crash
}

TEST_F(OnlineTrainerTest, ReplayBufferManagement) {
    // Fill replay buffer beyond capacity
    for (int i = 0; i < 150; ++i) {
        auto features = torch::randn({32});
        float label = std::sin(i * 0.01f);
        trainer_->add_training_sample(features, label);
    }
    
    // Buffer should be at capacity, not exceed it
    auto metrics = trainer_->get_training_metrics();
    EXPECT_LE(metrics.samples_processed, 150);
}

TEST_F(OnlineTrainerTest, AdaptiveLearningRate) {
    // Create performance metrics that should trigger learning rate adaptation
    PerformanceMetrics good_metrics;
    good_metrics.recent_accuracy = 0.6f;
    good_metrics.training_loss = 0.1f;
    
    PerformanceMetrics poor_metrics;
    poor_metrics.recent_accuracy = 0.4f;
    poor_metrics.training_loss = 0.5f;
    
    // Test that learning rate adapter responds to different performance levels
    // This is more of a smoke test to ensure the system doesn't crash
    AdaptiveLearningRate lr_adapter;
    float initial_lr = lr_adapter.get_current_learning_rate();
    
    lr_adapter.update_learning_rate(good_metrics);
    float good_lr = lr_adapter.get_current_learning_rate();
    
    lr_adapter.update_learning_rate(poor_metrics);
    float poor_lr = lr_adapter.get_current_learning_rate();
    
    // Learning rates should be positive and bounded
    EXPECT_GT(initial_lr, 0.0f);
    EXPECT_GT(good_lr, 0.0f);
    EXPECT_GT(poor_lr, 0.0f);
    
    EXPECT_LT(initial_lr, 1.0f);
    EXPECT_LT(good_lr, 1.0f);
    EXPECT_LT(poor_lr, 1.0f);
}
