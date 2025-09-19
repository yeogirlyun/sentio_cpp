// benchmark_transformer.cpp - Performance benchmarks
#include <chrono>
#include <iostream>
#include "sentio/transformer_strategy.hpp"
#include "sentio/transformer_strategy_core.hpp"
#include <memory>
#include <vector>
#include <cmath>

using namespace sentio;

// Mock classes for benchmarking
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

class BenchmarkFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        TransformerConfig config;
        config.feature_dim = 128;
        config.sequence_length = 64;
        config.d_model = 256;
        config.num_heads = 8;
        config.num_layers = 6;
        
        strategy_ = std::make_unique<SentioTransformerStrategy>(config);
        
        // Prepare test data
        for (int i = 0; i < 1000; ++i) {
            float price = 100.0f + std::sin(i * 0.1f) * 5.0f;
            test_bars_.emplace_back(price, price + 0.1f, price - 0.1f, price, 1000.0f);
        }
    }
    
    void TearDown(const benchmark::State& state) override {
        strategy_.reset();
    }
    
protected:
    std::unique_ptr<SentioTransformerStrategy> strategy_;
    std::vector<Bar> test_bars_;
};

BENCHMARK_F(BenchmarkFixture, BarProcessingLatency)(benchmark::State& state) {
    size_t bar_index = 0;
    
    for (auto _ : state) {
        auto signal = strategy_->on_bar(test_bars_[bar_index % test_bars_.size()]);
        bar_index++;
        
        // Prevent optimization
        benchmark::DoNotOptimize(signal);
    }
    
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_F(BenchmarkFixture, ModelInference)(benchmark::State& state) {
    // Build up some history first
    for (size_t i = 0; i < 70; ++i) {
        strategy_->on_bar(test_bars_[i]);
    }
    
    for (auto _ : state) {
        auto signal = strategy_->on_bar(test_bars_[70]);
        benchmark::DoNotOptimize(signal);
    }
    
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_F(BenchmarkFixture, MemoryUsage)(benchmark::State& state) {
    for (auto _ : state) {
        auto metrics = strategy_->get_performance_metrics();
        benchmark::DoNotOptimize(metrics.memory_usage_mb);
    }
}

BENCHMARK_F(BenchmarkFixture, FeatureGeneration)(benchmark::State& state) {
    // Build up history
    for (size_t i = 0; i < 100; ++i) {
        strategy_->on_bar(test_bars_[i]);
    }
    
    for (auto _ : state) {
        // This will trigger feature generation internally
        auto signal = strategy_->on_bar(test_bars_[100]);
        benchmark::DoNotOptimize(signal);
    }
    
    state.SetItemsProcessed(state.iterations());
}

// Register benchmarks
BENCHMARK_REGISTER_F(BenchmarkFixture, BarProcessingLatency)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(10000);

BENCHMARK_REGISTER_F(BenchmarkFixture, ModelInference)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000);

BENCHMARK_REGISTER_F(BenchmarkFixture, MemoryUsage)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BenchmarkFixture, FeatureGeneration)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(5000);

// Standalone benchmarks for individual components
static void BM_TransformerModelForward(benchmark::State& state) {
    TransformerConfig config;
    config.feature_dim = 128;
    config.sequence_length = 64;
    config.d_model = 256;
    config.num_heads = 8;
    config.num_layers = 6;
    
    TransformerModel model(config);
    model.eval();
    model.optimize_for_inference();
    
    auto input = torch::randn({1, config.sequence_length, config.feature_dim});
    
    for (auto _ : state) {
        torch::NoGradGuard no_grad;
        auto output = model.forward(input);
        benchmark::DoNotOptimize(output);
    }
    
    state.SetItemsProcessed(state.iterations());
}

static void BM_FeaturePipelineGeneration(benchmark::State& state) {
    TransformerConfig::Features config;
    config.normalization = TransformerConfig::Features::NormalizationMethod::Z_SCORE;
    config.decay_factor = 0.999f;
    
    FeaturePipeline pipeline(config);
    
    // Create test data
    std::vector<Bar> test_bars;
    for (int i = 0; i < 100; ++i) {
        float price = 100.0f + std::sin(i * 0.1f) * 5.0f;
        test_bars.emplace_back(price, price + 0.1f, price - 0.1f, price, 1000.0f);
    }
    
    for (auto _ : state) {
        auto features = pipeline.generate_features(test_bars);
        benchmark::DoNotOptimize(features);
    }
    
    state.SetItemsProcessed(state.iterations());
}

static void BM_RiskManagerValidation(benchmark::State& state) {
    RiskLimits limits;
    limits.max_position_size = 1.0f;
    limits.max_daily_trades = 100.0f;
    limits.min_confidence_threshold = 0.6f;
    
    ModelRiskManager risk_manager(limits);
    
    for (auto _ : state) {
        bool result = risk_manager.validate_prediction(0.1f, 0.7f);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
}

// Register standalone benchmarks
BENCHMARK(BM_TransformerModelForward)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000);

BENCHMARK(BM_FeaturePipelineGeneration)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(5000);

BENCHMARK(BM_RiskManagerValidation)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(100000);

// Memory benchmarks
static void BM_MemoryAllocation(benchmark::State& state) {
    for (auto _ : state) {
        TransformerConfig config;
        config.feature_dim = 64;
        config.sequence_length = 32;
        config.d_model = 128;
        config.num_heads = 4;
        config.num_layers = 2;
        
        auto strategy = std::make_unique<SentioTransformerStrategy>(config);
        benchmark::DoNotOptimize(strategy);
    }
    
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_MemoryAllocation)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// Scalability benchmarks
static void BM_BatchProcessing(benchmark::State& state) {
    const int batch_size = state.range(0);
    
    TransformerConfig config;
    config.feature_dim = 32;
    config.sequence_length = 16;
    config.d_model = 64;
    config.num_heads = 4;
    config.num_layers = 2;
    
    SentioTransformerStrategy strategy(config);
    
    // Prepare test data
    std::vector<Bar> test_bars;
    for (int i = 0; i < batch_size; ++i) {
        float price = 100.0f + i * 0.01f;
        test_bars.emplace_back(price, price + 0.1f, price - 0.1f, price, 1000.0f);
    }
    
    for (auto _ : state) {
        for (const auto& bar : test_bars) {
            auto signal = strategy.on_bar(bar);
            benchmark::DoNotOptimize(signal);
        }
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK(BM_BatchProcessing)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000);

// Main benchmark runner
BENCHMARK_MAIN();
