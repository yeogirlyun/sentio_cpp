#pragma once
#include "sentio/core.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/feature_engineering/technical_indicators.hpp"
#include "sentio/feature_engineering/feature_normalizer.hpp"
#include "sentio/feature_cache.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <mutex>

namespace sentio {

struct FeatureMetrics {
    std::chrono::microseconds extraction_time{0};
    size_t features_extracted{0};
    size_t features_valid{0};
    size_t features_invalid{0};
    double extraction_rate{0.0}; // features per second
    std::chrono::steady_clock::time_point last_update;
};

struct FeatureHealthReport {
    bool is_healthy{false};
    std::vector<bool> feature_health;
    std::vector<double> feature_quality_scores;
    std::string health_summary;
    double overall_health_score{0.0};
};

class FeatureFeeder {
public:
    // Core functionality
    static std::vector<double> extract_features_from_bar(const Bar& bar, const std::string& strategy_name);
    static std::vector<double> extract_features_from_bars_with_index(const std::vector<Bar>& bars, int current_index, const std::string& strategy_name);
    static bool is_ml_strategy(const std::string& strategy_name);
    static void feed_features_to_strategy(BaseStrategy* strategy, const std::vector<Bar>& bars, int current_index, const std::string& strategy_name);
    
    // Enhanced functionality
    static void initialize_strategy(const std::string& strategy_name);
    static void cleanup_strategy(const std::string& strategy_name);
    
    // **STRATEGY ISOLATION**: Clear all state to prevent cross-strategy contamination
    static void reset_all_state();
    
    // Feature management
    static std::vector<double> get_cached_features(const std::string& strategy_name);
    static void cache_features(const std::string& strategy_name, const std::vector<double>& features);
    static void invalidate_cache(const std::string& strategy_name);
    
    // Performance monitoring
    static FeatureMetrics get_metrics(const std::string& strategy_name);
    static FeatureHealthReport get_health_report(const std::string& strategy_name);
    static void reset_metrics(const std::string& strategy_name);
    
    // Feature validation
    static bool validate_features(const std::vector<double>& features, const std::string& strategy_name);
    static std::vector<std::string> get_feature_names(const std::string& strategy_name);
    
    // Configuration
    static void set_feature_config(const std::string& strategy_name, const std::string& config_key, const std::string& config_value);
    static std::string get_feature_config(const std::string& strategy_name, const std::string& config_key);
    
    // Cached features (for performance)
    static bool load_feature_cache(const std::string& feature_file_path);
    static bool use_cached_features(bool enable = true);
    static bool has_cached_features();
    
    // Batch processing
    static std::vector<std::vector<double>> extract_features_from_bars(const std::vector<Bar>& bars, const std::string& strategy_name);
    static void feed_features_batch(BaseStrategy* strategy, const std::vector<Bar>& bars, const std::string& strategy_name);
    
    // Feature analysis
    static std::vector<double> get_feature_correlation(const std::string& strategy_name);
    static std::vector<double> get_feature_importance(const std::string& strategy_name);
    static void log_feature_performance(const std::string& strategy_name);
    
private:
    // Strategy-specific data
    struct StrategyData {
        std::unique_ptr<feature_engineering::TechnicalIndicatorCalculator> calculator;
        std::unique_ptr<feature_engineering::FeatureNormalizer> normalizer;
        std::vector<double> cached_features;
        FeatureMetrics metrics;
        std::chrono::steady_clock::time_point last_update;
        bool initialized{false};
        std::unordered_map<std::string, std::string> config;
    };
    
    static std::unordered_map<std::string, StrategyData> strategy_data_;
    static std::mutex data_mutex_;
    
    // Feature cache for performance
    static std::unique_ptr<FeatureCache> feature_cache_;
    static bool use_cached_features_;
    
    // Helper methods
    static StrategyData& get_strategy_data(const std::string& strategy_name);
    static void update_metrics(StrategyData& data, const std::vector<double>& features, std::chrono::microseconds extraction_time);
    static FeatureHealthReport calculate_health_report(const StrategyData& data, const std::vector<double>& features);
    static std::vector<std::string> get_strategy_feature_names(const std::string& strategy_name);
    static void initialize_strategy_data(const std::string& strategy_name);
};

} // namespace sentio