// test_feature_pipeline.cpp - Tests for feature generation
#include <gtest/gtest.h>
#include "sentio/feature_pipeline.hpp"
#include "sentio/transformer_strategy_core.hpp"
#include "sentio/core.hpp"
#include <chrono>
#include <cstdlib>
#include <cmath>

using namespace sentio;

using sentio::Bar;

class FeaturePipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.normalization = TransformerConfig::Features::NormalizationMethod::Z_SCORE;
        config_.decay_factor = 0.999f;
        
        pipeline_ = std::make_unique<FeaturePipeline>(config_);
        
        // Create sample data
        for (int i = 0; i < 100; ++i) {
            float price = 100.0f + std::sin(i * 0.1f) * 5.0f + (rand() % 100) * 0.01f;
            float volume = 1000.0f + (rand() % 500);
            Bar bar;
            bar.ts_utc = "";
            bar.ts_utc_epoch = static_cast<int64_t>(std::time(nullptr));
            bar.open = price * 0.999f;
            bar.high = price * 1.001f;
            bar.low = price * 0.998f;
            bar.close = price;
            bar.volume = static_cast<uint64_t>(volume);
            test_bars_.push_back(bar);
        }
    }
    
    TransformerConfig::Features config_;
    std::unique_ptr<FeaturePipeline> pipeline_;
    std::vector<Bar> test_bars_;
};

TEST_F(FeaturePipelineTest, FeatureGeneration) {
    auto features = pipeline_->generate_features(test_bars_);
    
    // Check feature matrix dimensions
    EXPECT_EQ(features.dim(), 2);
    EXPECT_EQ(features.size(0), 1); // Batch size 1
    EXPECT_GT(features.size(1), 0); // Should have features
    
    // Check features are finite
    EXPECT_TRUE(torch::all(torch::isfinite(features)).item<bool>());
}

TEST_F(FeaturePipelineTest, FeatureGenerationLatency) {
    // Warmup
    for (int i = 0; i < 10; ++i) {
        auto features = pipeline_->generate_features(test_bars_);
    }
    
    // Measure latency
    const int num_iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        auto features = pipeline_->generate_features(test_bars_);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float avg_latency_us = static_cast<float>(duration.count()) / num_iterations;
    
    std::cout << "Average feature generation latency: " << avg_latency_us << " microseconds" << std::endl;
    
    // Check latency requirement (should be < 500us)
    EXPECT_LT(avg_latency_us, 500.0f);
}

TEST_F(FeaturePipelineTest, CacheManagement) {
    // Test cache update
    for (const auto& bar : test_bars_) {
        pipeline_->update_feature_cache(bar);
    }
    
    // Test cache retrieval
    auto cached_bars = pipeline_->get_cached_bars(50);
    EXPECT_EQ(cached_bars.size(), 50);
    
    // Check that cached bars are the most recent
    for (size_t i = 0; i < cached_bars.size(); ++i) {
        size_t original_idx = test_bars_.size() - cached_bars.size() + i;
        EXPECT_FLOAT_EQ(cached_bars[i].close, test_bars_[original_idx].close);
    }
}

TEST_F(FeaturePipelineTest, FeatureNormalization) {
    // Generate features multiple times to test normalization adaptation
    for (int i = 0; i < 10; ++i) {
        auto features = pipeline_->generate_features(test_bars_);
        
        // Features should be normalized (roughly zero mean, unit variance)
        auto mean = torch::mean(features);
        auto std = torch::std(features);
        
        // After several iterations, check finite and reasonable bounds
        if (i > 5) {
            EXPECT_TRUE(std::isfinite(mean.item<float>()));
            EXPECT_TRUE(std::isfinite(std.item<float>()));
            EXPECT_LT(std::abs(mean.item<float>()), 5.0f);
            EXPECT_LT(std.item<float>(), 10.0f);
        }
    }
}
