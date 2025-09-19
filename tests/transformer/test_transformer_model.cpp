// test_transformer_model.cpp - Unit tests for transformer model
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "sentio/transformer_model.hpp"
#include "sentio/transformer_strategy_core.hpp"
#include <chrono>

using namespace sentio;

class TransformerModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.feature_dim = 32;
        config_.sequence_length = 16;
        config_.d_model = 64;
        config_.num_heads = 4;
        config_.num_layers = 2;
        config_.ffn_hidden = 128;
        config_.dropout = 0.1f;
    }
    
    TransformerConfig config_;
};

TEST_F(TransformerModelTest, ModelCreation) {
    TransformerModel model(config_);
    
    // Check parameter count
    size_t param_count = model.get_parameter_count();
    EXPECT_GT(param_count, 0);
    
    // Check memory usage
    size_t memory_usage = model.get_memory_usage_bytes();
    EXPECT_GT(memory_usage, 0);
    EXPECT_LT(memory_usage, 100 * 1024 * 1024); // Should be less than 100MB for small model
}

TEST_F(TransformerModelTest, ForwardPass) {
    TransformerModel model(config_);
    model.eval();
    
    // Create test input
    auto input = torch::randn({2, config_.sequence_length, config_.feature_dim});
    
    torch::NoGradGuard no_grad;
    auto output = model.forward(input);
    
    // Check output shape
    EXPECT_EQ(output.dim(), 2);
    EXPECT_EQ(output.size(0), 2); // Batch size
    EXPECT_EQ(output.size(1), 1); // Single output
    
    // Check output is finite
    EXPECT_TRUE(torch::all(torch::isfinite(output)).item<bool>());
}

TEST_F(TransformerModelTest, InferenceLatency) {
    TransformerModel model(config_);
    model.eval();
    model.optimize_for_inference();
    
    auto input = torch::randn({1, config_.sequence_length, config_.feature_dim});
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        torch::NoGradGuard no_grad;
        auto output = model.forward(input);
    }
    
    // Measure latency
    const int num_iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        torch::NoGradGuard no_grad;
        auto output = model.forward(input);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float avg_latency_us = static_cast<float>(duration.count()) / num_iterations;
    
    std::cout << "Average inference latency: " << avg_latency_us << " microseconds" << std::endl;
    
    // Check latency requirement (should be < 1ms = 1000us)
    EXPECT_LT(avg_latency_us, 1000.0f);
}

TEST_F(TransformerModelTest, ModelSaveLoad) {
    TransformerModel model1(config_);
    
    // Save model
    std::string model_path = "/tmp/test_model.pt";
    model1.save_model(model_path);
    
    TransformerModel model2(config_);
    model2.load_model(model_path);
    
    auto input = torch::randn({1, config_.sequence_length, config_.feature_dim});
    model1.eval();
    model2.eval();
    {
        torch::NoGradGuard no_grad;
        auto output1 = model1.forward(input);
        auto output2 = model2.forward(input);
        EXPECT_TRUE(torch::allclose(output1, output2, 1e-6));
    }
    std::remove(model_path.c_str());
}
