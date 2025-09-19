#pragma once

#include "transformer_strategy_core.hpp"
#include "core.hpp"  // For Bar definition
#include <torch/torch.h>
#include <vector>
#include <deque>
#include <memory>
#include <chrono>

namespace sentio {

// Simple feature pipeline for the transformer strategy
class FeaturePipeline {
public:
    static constexpr int MAX_GENERATION_TIME_US = 500;
    static constexpr int FEATURE_CACHE_SIZE = 10000;
    static constexpr int TOTAL_FEATURES = 128; // Original feature count
    static constexpr int ENHANCED_TOTAL_FEATURES = 173; // Enhanced feature count (128 + 45)
    
    explicit FeaturePipeline(const TransformerConfig::Features& config);
    
    // Main interface
    TransformerFeatureMatrix generate_features(const std::vector<Bar>& bars);
    TransformerFeatureMatrix generate_enhanced_features(const std::vector<Bar>& bars);
    void update_feature_cache(const Bar& new_bar);
    std::vector<Bar> get_cached_bars(int lookback_periods) const;

private:
    TransformerConfig::Features config_;
    std::deque<Bar> feature_cache_;
    std::vector<RunningStats> feature_stats_;
    
    // Feature generation methods
    std::vector<float> generate_price_features(const std::vector<Bar>& bars);
    std::vector<float> generate_volume_features(const std::vector<Bar>& bars);
    std::vector<float> generate_technical_features(const std::vector<Bar>& bars);
    std::vector<float> generate_temporal_features(const Bar& current_bar);
    
    // Enhanced feature groups (incremental to the 128 original features)
    std::vector<float> generate_momentum_persistence_features(const std::vector<Bar>& bars);
    std::vector<float> generate_volatility_regime_features(const std::vector<Bar>& bars);
    std::vector<float> generate_microstructure_features(const std::vector<Bar>& bars);
    std::vector<float> generate_options_features(const std::vector<Bar>& bars);
    
    // Technical indicators
    std::vector<float> calculate_sma(const std::vector<float>& prices, int period);
    std::vector<float> calculate_ema(const std::vector<float>& prices, int period);
    std::vector<float> calculate_rsi(const std::vector<float>& prices, int period = 14);
    
    // Utilities for enhanced features
    float calculate_realized_volatility(const std::vector<Bar>& bars, int period);
    float calculate_average_volume(const std::vector<Bar>& bars, int period);
    float calculate_skewness(const std::vector<float>& values);
    float calculate_kurtosis(const std::vector<float>& values);
    float calculate_trend_strength(const std::vector<Bar>& bars, int lookback);
    float calculate_volume_trend(const std::vector<Bar>& bars, int period);
    float calculate_volatility_persistence(const std::vector<Bar>& bars, int period);
    float calculate_volatility_clustering(const std::vector<Bar>& bars, int period);
    float calculate_mean_reversion_speed(const std::vector<Bar>& bars, int period);
    float calculate_gamma_exposure_proxy(const std::vector<Bar>& bars);
    float detect_unusual_volume_patterns(const std::vector<Bar>& bars, int period);
    float calculate_fear_greed_proxy(const std::vector<Bar>& bars);
    float calculate_resistance_breakthrough(const std::vector<Bar>& bars);
    
    // Normalization
    void update_feature_statistics(const std::vector<float>& features);
    std::vector<float> normalize_features(const std::vector<float>& features);
};

} // namespace sentio
