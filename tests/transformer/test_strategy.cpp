// test_strategy.cpp - Integration tests for the main strategy
#include <gtest/gtest.h>
#include "sentio/transformer_strategy.hpp"
#include "sentio/transformer_strategy_core.hpp"
#include <yaml-cpp/yaml.h>
#include <memory>

using namespace sentio;

// Mock classes for testing
struct StrategySignal {
    enum class Action { NONE, BUY, SELL };
    Action action = Action::NONE;
    float size = 0.0f;
    float confidence = 0.0f;
    std::string reason;
};

class Bar {
public:
    float open = 0.0f;
    float high = 0.0f;
    float low = 0.0f;
    float close = 0.0f;
    float volume = 0.0f;
    std::chrono::system_clock::time_point timestamp;
    
    Bar() = default;
    Bar(float o, float h, float l, float c, float v) 
        : open(o), high(h), low(l), close(c), volume(v),
          timestamp(std::chrono::system_clock::now()) {}
};

class Fill {
public:
    enum class Side { BUY, SELL };
    Side side = Side::BUY;
    float price = 0.0f;
    float quantity = 0.0f;
    std::chrono::system_clock::time_point timestamp;
    std::string symbol;
};

class TransformerStrategyTest : public ::testing::Test {
protected:
    void SetUp() override {
        TransformerConfig config;
        config.feature_dim = 32;
        config.sequence_length = 16;
        config.d_model = 64;
        config.num_heads = 4;
        config.num_layers = 2;
        
        strategy_ = std::make_unique<SentioTransformerStrategy>(config);
    }
    
    std::unique_ptr<SentioTransformerStrategy> strategy_;
};

TEST_F(TransformerStrategyTest, StrategyInitialization) {
    EXPECT_EQ(strategy_->get_strategy_name(), "transformer");
    EXPECT_EQ(strategy_->get_model_status(), ModelStatus::READY);
}

TEST_F(TransformerStrategyTest, ConfigurationLoading) {
    // Create test YAML configuration
    YAML::Node config;
    config["model"]["architecture"]["d_model"] = 128;
    config["model"]["architecture"]["num_heads"] = 8;
    
    // Should not throw
    EXPECT_NO_THROW(strategy_->configure(config));
}

TEST_F(TransformerStrategyTest, BarProcessing) {
    // Feed bars to build up history
    for (int i = 0; i < 20; ++i) {
        float price = 100.0f + i * 0.1f;
        Bar bar(price, price + 0.1f, price - 0.1f, price, 1000.0f);
        
        auto signal = strategy_->on_bar(bar);
        
        // Should not crash and return valid signal
        EXPECT_TRUE(signal.action == StrategySignal::Action::NONE ||
                   signal.action == StrategySignal::Action::BUY ||
                   signal.action == StrategySignal::Action::SELL);
    }
}

TEST_F(TransformerStrategyTest, RealTimeTraining) {
    strategy_->enable_real_time_training(true);
    EXPECT_TRUE(strategy_->is_training_enabled());
    
    strategy_->enable_real_time_training(false);
    EXPECT_FALSE(strategy_->is_training_enabled());
}

TEST_F(TransformerStrategyTest, PerformanceMetrics) {
    // Process some bars first
    for (int i = 0; i < 10; ++i) {
        float price = 100.0f + i * 0.1f;
        Bar bar(price, price + 0.1f, price - 0.1f, price, 1000.0f);
        strategy_->on_bar(bar);
    }
    
    auto metrics = strategy_->get_performance_metrics();
    
    // Check that metrics are initialized
    EXPECT_GE(metrics.avg_inference_latency_ms, 0.0f);
    EXPECT_GE(metrics.memory_usage_mb, 0.0f);
    EXPECT_GE(metrics.recent_accuracy, 0.0f);
    EXPECT_LE(metrics.recent_accuracy, 1.0f);
}

TEST_F(TransformerStrategyTest, FillProcessing) {
    // Process some bars first
    for (int i = 0; i < 5; ++i) {
        float price = 100.0f + i * 0.1f;
        Bar bar(price, price + 0.1f, price - 0.1f, price, 1000.0f);
        strategy_->on_bar(bar);
    }
    
    // Create a test fill
    Fill fill;
    fill.side = Fill::Side::BUY;
    fill.price = 100.5f;
    fill.quantity = 100.0f;
    fill.symbol = "TEST";
    
    // Should not crash
    EXPECT_NO_THROW(strategy_->on_fill(fill));
}

TEST_F(TransformerStrategyTest, MarketEvents) {
    // Test market open/close events
    EXPECT_NO_THROW(strategy_->on_market_open());
    EXPECT_NO_THROW(strategy_->on_market_close());
}

TEST_F(TransformerStrategyTest, ModelManagement) {
    // Test model save/load (using temporary file)
    std::string temp_path = "/tmp/test_transformer_model.pt";
    
    // Save model
    EXPECT_NO_THROW(strategy_->save_model(temp_path));
    
    // Load model
    EXPECT_NO_THROW(strategy_->load_model(temp_path));
    
    // Clean up
    std::remove(temp_path.c_str());
}

TEST_F(TransformerStrategyTest, RiskManagement) {
    // Create a risk manager to test
    RiskLimits limits;
    limits.max_position_size = 0.5f;
    limits.max_daily_trades = 10.0f;
    limits.min_confidence_threshold = 0.6f;
    
    ModelRiskManager risk_manager(limits);
    
    // Test prediction validation
    EXPECT_TRUE(risk_manager.validate_prediction(0.1f, 0.7f));  // Valid
    EXPECT_FALSE(risk_manager.validate_prediction(0.1f, 0.5f)); // Low confidence
    EXPECT_FALSE(risk_manager.validate_prediction(NAN, 0.7f));  // Invalid prediction
    
    // Test position limits
    EXPECT_TRUE(risk_manager.check_position_limits(0.3f));  // Within limits
    EXPECT_FALSE(risk_manager.check_position_limits(0.8f)); // Exceeds limits
    
    // Test trading frequency
    EXPECT_TRUE(risk_manager.check_trading_frequency()); // Should be OK initially
    
    // Log some trades
    for (int i = 0; i < 5; ++i) {
        risk_manager.log_trade();
    }
    
    EXPECT_TRUE(risk_manager.check_trading_frequency()); // Should still be OK
}
