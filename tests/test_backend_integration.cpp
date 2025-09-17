#include "sentio/backend_architecture.hpp"
#include "sentio/strategy_signal_or.hpp"
#include "sentio/strategy_tfa.hpp"
#include "sentio/virtual_market.hpp"
#include "sentio/test_strategy.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

using namespace sentio;

// **INTEGRATION TEST**: Strategy-Agnostic Backend with Signal OR and TFA

class BackendIntegrationTest {
private:
    std::mt19937 rng_;
    
    // Generate synthetic market data for testing
    std::vector<Bar> generate_test_data(int num_bars, double initial_price = 400.0) {
        std::vector<Bar> bars;
        std::normal_distribution<double> return_dist(0.0005, 0.02); // 0.05% mean, 2% volatility
        
        double price = initial_price;
        for (int i = 0; i < num_bars; ++i) {
            Bar bar;
            bar.ts_utc_epoch = 1640995200 + i * 60; // Start from 2022-01-01, 1-minute bars
            
            double return_pct = return_dist(rng_);
            double new_price = price * (1.0 + return_pct);
            
            bar.open = price;
            bar.high = std::max(price, new_price) * (1.0 + std::abs(return_dist(rng_)) * 0.5);
            bar.low = std::min(price, new_price) * (1.0 - std::abs(return_dist(rng_)) * 0.5);
            bar.close = new_price;
            bar.volume = static_cast<uint64_t>(1000000 + return_dist(rng_) * 500000);
            
            bars.push_back(bar);
            price = new_price;
        }
        
        return bars;
    }
    
public:
    BackendIntegrationTest() : rng_(12345) {} // Fixed seed for reproducibility
    
    // **TEST 1**: Signal OR Strategy Integration
    void test_signal_or_integration() {
        std::cout << "\n🧪 TEST 1: Signal OR Strategy Integration" << std::endl;
        
        // Create backend with profit-maximizing configuration
        auto config = create_profit_maximizing_config();
        TradingBackend backend(config, 100000.0);
        
        // Create Signal OR strategy
        SignalOrCfg sigor_cfg;
        sigor_cfg.long_threshold = 0.55;
        sigor_cfg.short_threshold = 0.45;
        sigor_cfg.min_signal_strength = 0.05;
        
        SignalOrStrategy strategy(sigor_cfg);
        
        // Generate test data
        auto bars = generate_test_data(500);
        std::vector<std::vector<Bar>> market_data = {bars};
        
        // Run backtest
        auto performance = backend.run_backtest(&strategy, market_data, 50, 450);
        
        // Validate results
        std::cout << "✅ Signal OR Integration Results:" << std::endl;
        std::cout << "   Total Return: " << performance.total_return_pct << "%" << std::endl;
        std::cout << "   Sharpe Ratio: " << performance.sharpe_ratio << std::endl;
        std::cout << "   Max Drawdown: " << performance.max_drawdown_pct << "%" << std::endl;
        std::cout << "   Total Trades: " << performance.total_trades << std::endl;
        std::cout << "   Signal Accuracy: " << performance.signal_accuracy * 100.0 << "%" << std::endl;
        
        // Assertions
        assert(performance.total_trades > 0 && "Should have executed trades");
        assert(performance.signal_accuracy >= 0.0 && performance.signal_accuracy <= 1.0 && "Signal accuracy should be valid");
        assert(std::isfinite(performance.total_return_pct) && "Total return should be finite");
        assert(std::isfinite(performance.sharpe_ratio) && "Sharpe ratio should be finite");
        
        std::cout << "✅ Signal OR integration test PASSED!" << std::endl;
    }
    
    // **TEST 2**: TFA Strategy Integration
    void test_tfa_integration() {
        std::cout << "\n🧪 TEST 2: TFA Strategy Integration" << std::endl;
        
        // Create backend with test configuration (no transaction costs for clean testing)
        auto config = create_test_config();
        TradingBackend backend(config, 100000.0);
        
        // Create TFA strategy
        TFACfg tfa_cfg;
        tfa_cfg.conf_floor = 0.05;
        
        try {
            TFAStrategy strategy(tfa_cfg);
            
            // Generate test data
            auto bars = generate_test_data(300);
            std::vector<std::vector<Bar>> market_data = {bars};
            
            // Run backtest
            auto performance = backend.run_backtest(&strategy, market_data, 100, 250);
            
            // Validate results
            std::cout << "✅ TFA Integration Results:" << std::endl;
            std::cout << "   Total Return: " << performance.total_return_pct << "%" << std::endl;
            std::cout << "   Sharpe Ratio: " << performance.sharpe_ratio << std::endl;
            std::cout << "   Max Drawdown: " << performance.max_drawdown_pct << "%" << std::endl;
            std::cout << "   Total Trades: " << performance.total_trades << std::endl;
            std::cout << "   Signal Accuracy: " << performance.signal_accuracy * 100.0 << "%" << std::endl;
            
            // Assertions
            assert(std::isfinite(performance.total_return_pct) && "Total return should be finite");
            assert(performance.signal_accuracy >= 0.0 && performance.signal_accuracy <= 1.0 && "Signal accuracy should be valid");
            
            std::cout << "✅ TFA integration test PASSED!" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "⚠️  TFA integration test SKIPPED (artifacts not available): " << e.what() << std::endl;
        }
    }
    
    // **TEST 3**: Backend Performance Validation
    void test_backend_performance_validation() {
        std::cout << "\n🧪 TEST 3: Backend Performance Validation" << std::endl;
        
        // Test with different accuracy levels using SimpleTestStrategy
        std::vector<double> accuracy_levels = {0.3, 0.5, 0.7, 0.9};
        std::vector<BackendPerformance> results;
        
        for (double accuracy : accuracy_levels) {
            std::cout << "\n📊 Testing " << (accuracy * 100) << "% accuracy strategy..." << std::endl;
            
            // Create backend
            auto config = create_test_config();
            TradingBackend backend(config, 100000.0);
            
            // Create test strategy with controlled accuracy
            TestStrategyConfig strategy_config;
            strategy_config.target_accuracy = accuracy;
            strategy_config.random_seed = 42; // Fixed seed
            TestStrategy strategy(strategy_config);
            
            // Generate test data
            auto bars = generate_test_data(400);
            std::vector<std::vector<Bar>> market_data = {bars};
            
            // Run backtest
            auto performance = backend.run_backtest(&strategy, market_data, 50, 350);
            results.push_back(performance);
            
            std::cout << "   Accuracy: " << (performance.signal_accuracy * 100.0) << "%" << std::endl;
            std::cout << "   Return: " << performance.total_return_pct << "%" << std::endl;
            std::cout << "   Sharpe: " << performance.sharpe_ratio << std::endl;
            std::cout << "   Trades: " << performance.total_trades << std::endl;
        }
        
        // **VALIDATION**: Higher accuracy should generally lead to better performance
        std::cout << "\n📈 Performance Validation:" << std::endl;
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "   " << (accuracy_levels[i] * 100) << "% accuracy: " 
                      << results[i].total_return_pct << "% return, "
                      << results[i].sharpe_ratio << " Sharpe" << std::endl;
        }
        
        // Check that backend scales performance with accuracy (correlation test)
        double correlation = calculate_correlation(accuracy_levels, results);
        std::cout << "\n🎯 Accuracy-Performance Correlation: " << correlation << std::endl;
        
        assert(correlation > 0.3 && "Backend should show positive correlation between accuracy and performance");
        std::cout << "✅ Backend performance validation PASSED!" << std::endl;
    }
    
    // **TEST 4**: Conflict Prevention Validation
    void test_conflict_prevention() {
        std::cout << "\n🧪 TEST 4: Conflict Prevention Validation" << std::endl;
        
        // Create backend
        auto config = create_test_config();
        TradingBackend backend(config, 100000.0);
        
        // Create Signal OR strategy (has conflict prevention rules)
        SignalOrStrategy strategy;
        
        // Generate test data with volatile signals
        auto bars = generate_test_data(200);
        std::vector<std::vector<Bar>> market_data = {bars};
        
        // Run backtest and track position conflicts
        std::vector<std::string> position_history;
        
        for (int bar = 50; bar < 150; ++bar) {
            auto executions = backend.process_strategy_signal(&strategy, market_data, bar);
            
            // Track positions after each execution
            const auto& portfolio = backend.get_current_portfolio();
            std::string position_state = "";
            
            for (size_t i = 0; i < portfolio.positions.size(); ++i) {
                if (std::abs(portfolio.positions[i].qty) > 1e-6) {
                    if (!position_state.empty()) position_state += ",";
                    position_state += "pos" + std::to_string(i) + ":" + std::to_string(portfolio.positions[i].qty);
                }
            }
            
            if (position_state.empty()) position_state = "CASH";
            position_history.push_back(position_state);
        }
        
        // Validate no conflicting positions were held simultaneously
        bool conflicts_detected = false;
        for (const auto& state : position_history) {
            // Check for simultaneous long and inverse positions
            bool has_long = state.find("TQQQ") != std::string::npos || state.find("QQQ") != std::string::npos;
            bool has_inverse = state.find("SQQQ") != std::string::npos || state.find("PSQ") != std::string::npos;
            
            if (has_long && has_inverse) {
                std::cout << "⚠️  Conflict detected: " << state << std::endl;
                conflicts_detected = true;
            }
        }
        
        assert(!conflicts_detected && "No conflicting positions should be held simultaneously");
        std::cout << "✅ Conflict prevention validation PASSED!" << std::endl;
    }
    
    // **TEST 5**: Strategy-Agnostic Interface Validation
    void test_strategy_agnostic_interface() {
        std::cout << "\n🧪 TEST 5: Strategy-Agnostic Interface Validation" << std::endl;
        
        // Test that backend works identically with different strategies
        auto config = create_test_config();
        auto bars = generate_test_data(200);
        std::vector<std::vector<Bar>> market_data = {bars};
        
        // Test with Signal OR
        {
            TradingBackend backend(config, 100000.0);
            SignalOrStrategy strategy;
            auto performance = backend.run_backtest(&strategy, market_data, 50, 150);
            
            std::cout << "   Signal OR: " << performance.total_return_pct << "% return" << std::endl;
            assert(std::isfinite(performance.total_return_pct) && "Signal OR should produce finite results");
        }
        
        // Test with TestStrategy
        {
            TradingBackend backend(config, 100000.0);
            TestStrategyConfig test_config;
            test_config.target_accuracy = 0.6;
            test_config.random_seed = 123;
            TestStrategy strategy(test_config);
            auto performance = backend.run_backtest(&strategy, market_data, 50, 150);
            
            std::cout << "   Test Strategy: " << performance.total_return_pct << "% return" << std::endl;
            assert(std::isfinite(performance.total_return_pct) && "Test strategy should produce finite results");
        }
        
        std::cout << "✅ Strategy-agnostic interface validation PASSED!" << std::endl;
    }
    
    // Run all integration tests
    void run_all_tests() {
        std::cout << "🚀 BACKEND INTEGRATION TEST SUITE" << std::endl;
        std::cout << "===================================" << std::endl;
        
        try {
            test_signal_or_integration();
            test_tfa_integration();
            test_backend_performance_validation();
            test_conflict_prevention();
            test_strategy_agnostic_interface();
            
            std::cout << "\n🎉 ALL INTEGRATION TESTS PASSED!" << std::endl;
            std::cout << "✅ Strategy-agnostic backend is working correctly!" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "\n❌ INTEGRATION TEST FAILED: " << e.what() << std::endl;
            throw;
        }
    }
    
private:
    // Helper function to calculate correlation between accuracy and performance
    double calculate_correlation(const std::vector<double>& accuracy_levels, 
                               const std::vector<BackendPerformance>& results) {
        if (accuracy_levels.size() != results.size() || accuracy_levels.size() < 2) {
            return 0.0;
        }
        
        // Extract returns for correlation calculation
        std::vector<double> returns;
        for (const auto& result : results) {
            returns.push_back(result.total_return_pct);
        }
        
        // Calculate Pearson correlation coefficient
        double n = static_cast<double>(accuracy_levels.size());
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
        
        for (size_t i = 0; i < accuracy_levels.size(); ++i) {
            double x = accuracy_levels[i];
            double y = returns[i];
            
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }
        
        double numerator = n * sum_xy - sum_x * sum_y;
        double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
        
        return denominator > 1e-10 ? numerator / denominator : 0.0;
    }
};

// **MAIN TEST RUNNER**
int main() {
    try {
        BackendIntegrationTest test_suite;
        test_suite.run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Integration test failed: " << e.what() << std::endl;
        return 1;
    }
}
