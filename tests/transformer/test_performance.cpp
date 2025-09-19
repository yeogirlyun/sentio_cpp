// test_performance.cpp - Performance and stress tests
#include <gtest/gtest.h>
#include "sentio/strategy_transformer.hpp"
#include "sentio/transformer_strategy_core.hpp"
#include "sentio/core.hpp"
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

using namespace sentio;

using sentio::Bar;
// Minimal StrategySignal alias if tests reference it; use local struct here if needed
struct StrategySignal { enum class Action { NONE, BUY, SELL }; Action action{Action::NONE}; float size{0.0f}; float confidence{0.0f}; std::string reason; };

class Fill {
public:
    enum class Side { BUY, SELL };
    Side side = Side::BUY;
    float price = 0.0f;
    float quantity = 0.0f;
    std::chrono::system_clock::time_point timestamp;
    std::string symbol;
};

class PerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        TransformerConfig config;
        config.max_inference_latency_ms = 1.0f;
        config.max_memory_usage_mb = 1024.0f;
        
        strategy_ = std::make_unique<SentioTransformerStrategy>(config);
        
        // Generate large dataset for stress testing
        test_bars_.reserve(10000);
        for (int i = 0; i < 10000; ++i) {
            float price = 100.0f + std::sin(i * 0.01f) * 10.0f + (rand() % 100) * 0.01f;
            float volume = 500.0f + (rand() % 1000);
            test_bars_.emplace_back(price * 0.999f, price * 1.001f, 
                                   price * 0.998f, price, volume);
        }
    }
    
    std::unique_ptr<SentioTransformerStrategy> strategy_;
    std::vector<Bar> test_bars_;
};

TEST_F(PerformanceTest, StressTestContinuousOperation) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Simulate continuous trading for extended period
    for (size_t i = 0; i < test_bars_.size(); ++i) {
        auto signal = strategy_->on_bar(test_bars_[i]);
        
        // Simulate fills occasionally
        if (signal.action != StrategySignal::Action::NONE && (i % 10 == 0)) {
            Fill fill;
            fill.side = (signal.action == StrategySignal::Action::BUY) ? 
                       Fill::Side::BUY : Fill::Side::SELL;
            fill.price = test_bars_[i].close;
            fill.quantity = signal.size;
            
            strategy_->on_fill(fill);
        }
        
        // Check performance every 1000 iterations
        if (i % 1000 == 0) {
            auto metrics = strategy_->get_performance_metrics();
            
            // Verify latency requirements
            EXPECT_LT(metrics.avg_inference_latency_ms, 2.0f); // Relaxed for stress test
            EXPECT_LT(metrics.p99_inference_latency_ms, 5.0f);
            
            // Verify memory usage
            EXPECT_LT(metrics.memory_usage_mb, 1500.0f); // Relaxed for stress test
            
            std::cout << "Progress: " << i << "/10000, "
                      << "Avg Latency: " << metrics.avg_inference_latency_ms << "ms, "
                      << "Memory: " << metrics.memory_usage_mb << "MB" << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time);
    
    std::cout << "Stress test completed in " << total_duration.count() << " seconds" << std::endl;
    
    auto final_metrics = strategy_->get_performance_metrics();
    std::cout << "Final metrics:" << std::endl;
    std::cout << "  Accuracy: " << final_metrics.recent_accuracy << std::endl;
    std::cout << "  Avg Latency: " << final_metrics.avg_inference_latency_ms << "ms" << std::endl;
    std::cout << "  P99 Latency: " << final_metrics.p99_inference_latency_ms << "ms" << std::endl;
    std::cout << "  Memory Usage: " << final_metrics.memory_usage_mb << "MB" << std::endl;
}

TEST_F(PerformanceTest, MemoryLeakTest) {
    // Measure initial memory
    auto initial_metrics = strategy_->get_performance_metrics();
    float initial_memory = initial_metrics.memory_usage_mb;
    
    // Run for many iterations
    for (int iteration = 0; iteration < 5; ++iteration) {
        for (size_t i = 0; i < 1000; ++i) {
            strategy_->on_bar(test_bars_[i % test_bars_.size()]);
        }
        
        // Force garbage collection (if applicable)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        auto current_metrics = strategy_->get_performance_metrics();
        float memory_growth = current_metrics.memory_usage_mb - initial_memory;
        
        std::cout << "Iteration " << iteration << ", Memory growth: " 
                  << memory_growth << "MB" << std::endl;
        
        // Memory growth should be bounded
        EXPECT_LT(memory_growth, 100.0f); // Allow 100MB growth max
    }
}

TEST_F(PerformanceTest, LatencyConsistency) {
    // Build up history
    for (size_t i = 0; i < 100; ++i) {
        strategy_->on_bar(test_bars_[i]);
    }
    
    std::vector<float> latencies;
    latencies.reserve(1000);
    
    // Measure latencies
    for (int i = 0; i < 1000; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        strategy_->on_bar(test_bars_[100 + (i % 1000)]);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        latencies.push_back(duration.count() / 1000.0f); // Convert to milliseconds
    }
    
    // Calculate statistics
    std::sort(latencies.begin(), latencies.end());
    
    float median = latencies[latencies.size() / 2];
    float p95 = latencies[static_cast<size_t>(latencies.size() * 0.95)];
    float p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];
    float max_latency = latencies.back();
    
    std::cout << "Latency statistics (ms):" << std::endl;
    std::cout << "  Median: " << median << std::endl;
    std::cout << "  P95: " << p95 << std::endl;
    std::cout << "  P99: " << p99 << std::endl;
    std::cout << "  Max: " << max_latency << std::endl;
    
    // Verify requirements
    EXPECT_LT(median, 1.0f);
    EXPECT_LT(p95, 2.0f);
    EXPECT_LT(p99, 5.0f);
    EXPECT_LT(max_latency, 10.0f); // Allow some outliers
}

TEST_F(PerformanceTest, ThroughputTest) {
    // Build up history
    for (size_t i = 0; i < 100; ++i) {
        strategy_->on_bar(test_bars_[i]);
    }
    
    const int num_predictions = 10000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_predictions; ++i) {
        strategy_->on_bar(test_bars_[100 + (i % 1000)]);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    float throughput = static_cast<float>(num_predictions) / (duration.count() / 1000.0f);
    
    std::cout << "Throughput: " << throughput << " predictions/second" << std::endl;
    
    // Should achieve at least 1000 predictions per second
    EXPECT_GT(throughput, 1000.0f);
}

TEST_F(PerformanceTest, ConcurrentAccess) {
    // Test thread safety with multiple threads accessing the strategy
    const int num_threads = 4;
    const int iterations_per_thread = 100;
    
    std::vector<std::thread> threads;
    std::atomic<int> successful_operations{0};
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, iterations_per_thread, &successful_operations]() {
            for (int i = 0; i < iterations_per_thread; ++i) {
                try {
                    size_t bar_idx = (t * iterations_per_thread + i) % test_bars_.size();
                    auto signal = strategy_->on_bar(test_bars_[bar_idx]);
                    successful_operations++;
                } catch (const std::exception& e) {
                    std::cerr << "Thread " << t << " error: " << e.what() << std::endl;
                }
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // All operations should succeed
    EXPECT_EQ(successful_operations.load(), num_threads * iterations_per_thread);
}

TEST_F(PerformanceTest, ResourceCleanup) {
    // Test that resources are properly cleaned up
    auto initial_metrics = strategy_->get_performance_metrics();
    
    // Create and destroy multiple strategies
    for (int i = 0; i < 10; ++i) {
        TransformerConfig config;
        config.feature_dim = 32;
        config.sequence_length = 16;
        
        auto temp_strategy = std::make_unique<SentioTransformerStrategy>(config);
        
        // Use the strategy briefly
        for (int j = 0; j < 10; ++j) {
            temp_strategy->on_bar(test_bars_[j]);
        }
        
        // Strategy should be destroyed here
    }
    
    // Memory usage should not have grown significantly
    auto final_metrics = strategy_->get_performance_metrics();
    float memory_growth = final_metrics.memory_usage_mb - initial_metrics.memory_usage_mb;
    
    std::cout << "Memory growth after resource cleanup test: " << memory_growth << "MB" << std::endl;
    
    // Allow some growth but not excessive
    EXPECT_LT(memory_growth, 50.0f);
}
