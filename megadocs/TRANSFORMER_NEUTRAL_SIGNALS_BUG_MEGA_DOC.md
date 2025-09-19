# Transformer Strategy Neutral Signals Bug Analysis

**Generated**: 2025-09-19 21:13:09
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: Comprehensive analysis of the transformer strategy bug causing neutral (0.5) signal emissions due to tensor dimension mismatch between feature pipeline and model architecture

**Total Files**: 11

---

## 🐛 **BUG REPORT**

# Bug Report: Transformer Strategy Emitting Neutral Signals (0.5)

## Summary
The Sentio Transformer Strategy is consistently emitting neutral signals (probability = 0.5) instead of meaningful directional predictions, resulting in zero trading activity during backtests.

## Bug Details

### Issue Description
- **Strategy Name**: `transformer`
- **Observed Behavior**: All signals have probability = 0.5 (neutral)
- **Expected Behavior**: Variable probabilities based on market conditions
- **Impact**: Strategy generates no trades, resulting in 0% returns
- **Severity**: High - Strategy is non-functional for trading

### Error Messages
```
Error in calculate_probability: mat1 and mat2 shapes cannot be multiplied (1x64 and 128x256)
Exception raised from meta at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/LinearAlgebra.cpp:194
```

### Root Cause Analysis
The primary issue appears to be a **tensor dimension mismatch** in the transformer model:

1. **Feature Generation**: The `FeaturePipeline` generates 128-dimensional feature vectors
2. **Model Configuration**: The trained model expects 64-dimensional input sequences
3. **Matrix Multiplication Error**: Attempting to multiply (1x64) with (128x256) tensors
4. **Fallback Behavior**: When the model fails, the strategy defaults to neutral (0.5) probability

### Technical Details

#### Dimension Mismatch
- **Generated Features**: `[128]` (128 features per bar)
- **Model Input Expected**: `[sequence_length=64, feature_dim=128]`
- **Actual Model Input**: `[1, 64]` (incorrect reshaping)
- **Model Weights**: First linear layer expects `[128, 256]`

#### Code Flow Issue
1. `FeaturePipeline::generate_features()` produces 128-dimensional vectors
2. `TransformerStrategy::calculate_probability()` reshapes incorrectly
3. Model forward pass fails with dimension mismatch
4. Exception caught, returns default 0.5 probability

### Test Results
- **Run ID**: 516262
- **Total Signals**: 8,600 signals generated
- **Signal Values**: All signals = 0.5 (neutral)
- **Trades Executed**: 0 trades
- **P&L**: $0.00 (no trading activity)

### Affected Components
1. **Feature Pipeline** (`src/feature_pipeline.cpp`)
2. **Transformer Model** (`src/transformer_model.cpp`)
3. **Strategy Implementation** (`src/strategy_transformer.cpp`)
4. **Model Training** (`src/transformer_trainer_main.cpp`)

### Configuration Issues
- **Training Config**: Model trained with incorrect feature dimensions
- **Runtime Config**: Feature pipeline and model dimension mismatch
- **Sequence Handling**: Incorrect tensor reshaping for transformer input

## Reproduction Steps
1. Build and run transformer trainer: `make build/transformer_trainer`
2. Train model: `./build/transformer_trainer --epochs 5 --data data/test_download/QQQ_RTH_NH.csv`
3. Run strategy test: `./sencli strattest transformer --mode historical --blocks 20`
4. Observe: All signals are 0.5, no trades executed

## Expected Fix
1. **Align Dimensions**: Ensure feature pipeline output matches model input expectations
2. **Fix Tensor Reshaping**: Correct the sequence tensor creation in `calculate_probability()`
3. **Update Training**: Retrain model with correct feature dimensions
4. **Add Validation**: Add dimension validation in model loading

## Priority
**HIGH** - Strategy is completely non-functional for trading purposes.

## Environment
- **OS**: macOS 14.6.0
- **Compiler**: g++ (C++20)
- **PyTorch**: LibTorch C++ API
- **Dataset**: QQQ_RTH_NH.csv (35.1 days, 9,600 bars)

## Additional Notes
- The strategy framework integration is working correctly
- Error handling prevents crashes but masks the underlying issue
- Training completes successfully but produces incompatible model
- All other Sentio systems (audit, position management) function normally

---
**Created**: 2025-09-19
**Reporter**: AI Assistant
**Status**: Open
**Assigned**: Development Team


---

## 📋 **TABLE OF CONTENTS**

1. [temp_mega_doc/bug_report_transformer_neutral_signals.md](#file-1)
2. [temp_mega_doc/feature_pipeline.cpp](#file-2)
3. [temp_mega_doc/feature_pipeline.hpp](#file-3)
4. [temp_mega_doc/online_trainer.hpp](#file-4)
5. [temp_mega_doc/strategy_initialization.cpp](#file-5)
6. [temp_mega_doc/strategy_transformer.cpp](#file-6)
7. [temp_mega_doc/strategy_transformer.hpp](#file-7)
8. [temp_mega_doc/transformer_model.cpp](#file-8)
9. [temp_mega_doc/transformer_model.hpp](#file-9)
10. [temp_mega_doc/transformer_strategy_core.hpp](#file-10)
11. [temp_mega_doc/transformer_trainer_main.cpp](#file-11)

---

## 📄 **FILE 1 of 11**: temp_mega_doc/bug_report_transformer_neutral_signals.md

**File Information**:
- **Path**: `temp_mega_doc/bug_report_transformer_neutral_signals.md`

- **Size**: 92 lines
- **Modified**: 2025-09-19 21:12:54

- **Type**: .md

```text
# Bug Report: Transformer Strategy Emitting Neutral Signals (0.5)

## Summary
The Sentio Transformer Strategy is consistently emitting neutral signals (probability = 0.5) instead of meaningful directional predictions, resulting in zero trading activity during backtests.

## Bug Details

### Issue Description
- **Strategy Name**: `transformer`
- **Observed Behavior**: All signals have probability = 0.5 (neutral)
- **Expected Behavior**: Variable probabilities based on market conditions
- **Impact**: Strategy generates no trades, resulting in 0% returns
- **Severity**: High - Strategy is non-functional for trading

### Error Messages
```
Error in calculate_probability: mat1 and mat2 shapes cannot be multiplied (1x64 and 128x256)
Exception raised from meta at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/LinearAlgebra.cpp:194
```

### Root Cause Analysis
The primary issue appears to be a **tensor dimension mismatch** in the transformer model:

1. **Feature Generation**: The `FeaturePipeline` generates 128-dimensional feature vectors
2. **Model Configuration**: The trained model expects 64-dimensional input sequences
3. **Matrix Multiplication Error**: Attempting to multiply (1x64) with (128x256) tensors
4. **Fallback Behavior**: When the model fails, the strategy defaults to neutral (0.5) probability

### Technical Details

#### Dimension Mismatch
- **Generated Features**: `[128]` (128 features per bar)
- **Model Input Expected**: `[sequence_length=64, feature_dim=128]`
- **Actual Model Input**: `[1, 64]` (incorrect reshaping)
- **Model Weights**: First linear layer expects `[128, 256]`

#### Code Flow Issue
1. `FeaturePipeline::generate_features()` produces 128-dimensional vectors
2. `TransformerStrategy::calculate_probability()` reshapes incorrectly
3. Model forward pass fails with dimension mismatch
4. Exception caught, returns default 0.5 probability

### Test Results
- **Run ID**: 516262
- **Total Signals**: 8,600 signals generated
- **Signal Values**: All signals = 0.5 (neutral)
- **Trades Executed**: 0 trades
- **P&L**: $0.00 (no trading activity)

### Affected Components
1. **Feature Pipeline** (`src/feature_pipeline.cpp`)
2. **Transformer Model** (`src/transformer_model.cpp`)
3. **Strategy Implementation** (`src/strategy_transformer.cpp`)
4. **Model Training** (`src/transformer_trainer_main.cpp`)

### Configuration Issues
- **Training Config**: Model trained with incorrect feature dimensions
- **Runtime Config**: Feature pipeline and model dimension mismatch
- **Sequence Handling**: Incorrect tensor reshaping for transformer input

## Reproduction Steps
1. Build and run transformer trainer: `make build/transformer_trainer`
2. Train model: `./build/transformer_trainer --epochs 5 --data data/test_download/QQQ_RTH_NH.csv`
3. Run strategy test: `./sencli strattest transformer --mode historical --blocks 20`
4. Observe: All signals are 0.5, no trades executed

## Expected Fix
1. **Align Dimensions**: Ensure feature pipeline output matches model input expectations
2. **Fix Tensor Reshaping**: Correct the sequence tensor creation in `calculate_probability()`
3. **Update Training**: Retrain model with correct feature dimensions
4. **Add Validation**: Add dimension validation in model loading

## Priority
**HIGH** - Strategy is completely non-functional for trading purposes.

## Environment
- **OS**: macOS 14.6.0
- **Compiler**: g++ (C++20)
- **PyTorch**: LibTorch C++ API
- **Dataset**: QQQ_RTH_NH.csv (35.1 days, 9,600 bars)

## Additional Notes
- The strategy framework integration is working correctly
- Error handling prevents crashes but masks the underlying issue
- Training completes successfully but produces incompatible model
- All other Sentio systems (audit, position management) function normally

---
**Created**: 2025-09-19
**Reporter**: AI Assistant
**Status**: Open
**Assigned**: Development Team

```

## 📄 **FILE 2 of 11**: temp_mega_doc/feature_pipeline.cpp

**File Information**:
- **Path**: `temp_mega_doc/feature_pipeline.cpp`

- **Size**: 479 lines
- **Modified**: 2025-09-19 21:12:54

- **Type**: .cpp

```text
#include "sentio/feature_pipeline.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

namespace sentio {

FeaturePipeline::FeaturePipeline(const TransformerConfig::Features& config)
    : config_(config) {
    // Note: std::deque doesn't have reserve(), but that's okay
    feature_stats_.resize(TOTAL_FEATURES, RunningStats(config.decay_factor));
}

TransformerFeatureMatrix FeaturePipeline::generate_features(const std::vector<Bar>& bars) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (bars.empty()) {
        return torch::zeros({1, TOTAL_FEATURES});
    }
    
    std::vector<float> all_features;
    all_features.reserve(TOTAL_FEATURES);
    
    // Generate different feature categories
    auto price_features = generate_price_features(bars);
    auto volume_features = generate_volume_features(bars);
    auto technical_features = generate_technical_features(bars);
    auto temporal_features = generate_temporal_features(bars.back());
    
    // Combine all features
    all_features.insert(all_features.end(), price_features.begin(), price_features.end());
    all_features.insert(all_features.end(), volume_features.begin(), volume_features.end());
    all_features.insert(all_features.end(), technical_features.begin(), technical_features.end());
    all_features.insert(all_features.end(), temporal_features.begin(), temporal_features.end());
    
    // Pad or truncate to exact feature count
    if (all_features.size() < TOTAL_FEATURES) {
        all_features.resize(TOTAL_FEATURES, 0.0f);
    } else if (all_features.size() > TOTAL_FEATURES) {
        all_features.resize(TOTAL_FEATURES);
    }
    
    // Update statistics and normalize
    update_feature_statistics(all_features);
    auto normalized_features = normalize_features(all_features);
    
    // Convert to tensor
    auto feature_tensor = torch::tensor(normalized_features).unsqueeze(0); // Add batch dimension
    
    // Check latency requirement
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (duration.count() > MAX_GENERATION_TIME_US) {
        std::cerr << "Feature generation exceeded latency requirement: " 
                  << duration.count() << "us > " << MAX_GENERATION_TIME_US << "us" << std::endl;
    }
    
    return feature_tensor;
}

void FeaturePipeline::update_feature_cache(const Bar& new_bar) {
    if (feature_cache_.size() >= FEATURE_CACHE_SIZE) {
        feature_cache_.pop_front();
    }
    feature_cache_.push_back(new_bar);
}

std::vector<Bar> FeaturePipeline::get_cached_bars(int lookback_periods) const {
    std::vector<Bar> bars;
    
    if (lookback_periods <= 0 || feature_cache_.empty()) {
        return bars;
    }
    
    int start_idx = std::max(0, static_cast<int>(feature_cache_.size()) - lookback_periods);
    bars.reserve(feature_cache_.size() - start_idx);
    
    for (size_t i = start_idx; i < feature_cache_.size(); ++i) {
        bars.push_back(feature_cache_[i]);
    }
    
    return bars;
}

std::vector<float> FeaturePipeline::generate_price_features(const std::vector<Bar>& bars) {
    std::vector<float> features;
    features.reserve(40); // Target 40 price features
    
    if (bars.empty()) {
        features.resize(40, 0.0f);
        return features;
    }
    
    const Bar& current = bars.back();
    
    // Basic OHLC normalized features
    features.push_back(current.open / current.close - 1.0f);
    features.push_back(current.high / current.close - 1.0f);
    features.push_back(current.low / current.close - 1.0f);
    features.push_back(0.0f); // close normalized to itself
    
    // Returns
    if (bars.size() >= 2) {
        features.push_back((current.close / bars[bars.size()-2].close) - 1.0f); // 1-period return
    } else {
        features.push_back(0.0f);
    }
    
    // Multi-period returns
    std::vector<int> periods = {5, 10, 20};
    for (int period : periods) {
        if (bars.size() > period) {
            float ret = (current.close / bars[bars.size()-1-period].close) - 1.0f;
            features.push_back(ret);
        } else {
            features.push_back(0.0f);
        }
    }
    
    // Log returns
    if (bars.size() >= 2) {
        features.push_back(std::log(current.close / bars[bars.size()-2].close));
    } else {
        features.push_back(0.0f);
    }
    
    // Extract close prices for technical indicators
    std::vector<float> closes;
    for (const auto& bar : bars) {
        closes.push_back(bar.close);
    }
    
    // Moving averages (SMA)
    std::vector<int> ma_periods = {5, 10, 20, 50};
    for (int period : ma_periods) {
        auto sma_values = calculate_sma(closes, period);
        if (!sma_values.empty()) {
            features.push_back((current.close / sma_values.back()) - 1.0f);
        } else {
            features.push_back(0.0f);
        }
    }
    
    // Moving averages (EMA)
    for (int period : ma_periods) {
        auto ema_values = calculate_ema(closes, period);
        if (!ema_values.empty()) {
            features.push_back((current.close / ema_values.back()) - 1.0f);
        } else {
            features.push_back(0.0f);
        }
    }
    
    // Volatility features
    if (bars.size() >= 10) {
        float sum_sq = 0.0f;
        float mean_return = 0.0f;
        
        // Calculate returns for volatility
        std::vector<float> returns;
        for (size_t i = bars.size() - 10; i < bars.size() - 1; ++i) {
            float ret = std::log(bars[i+1].close / bars[i].close);
            returns.push_back(ret);
            mean_return += ret;
        }
        mean_return /= returns.size();
        
        // Calculate standard deviation
        for (float ret : returns) {
            sum_sq += (ret - mean_return) * (ret - mean_return);
        }
        float vol = std::sqrt(sum_sq / (returns.size() - 1)) * std::sqrt(252); // Annualized
        features.push_back(vol);
    } else {
        features.push_back(0.0f);
    }
    
    // High-Low ratio
    features.push_back((current.high - current.low) / current.close);
    
    // Momentum features
    if (bars.size() >= 5) {
        float mom_5 = current.close - bars[bars.size()-6].close;
        features.push_back(mom_5 / current.close);
    } else {
        features.push_back(0.0f);
    }
    
    // Pad to exactly 40 features
    while (features.size() < 40) {
        features.push_back(0.0f);
    }
    features.resize(40);
    
    return features;
}

std::vector<float> FeaturePipeline::generate_volume_features(const std::vector<Bar>& bars) {
    std::vector<float> features;
    features.reserve(20); // Target 20 volume features
    
    if (bars.empty()) {
        features.resize(20, 0.0f);
        return features;
    }
    
    const Bar& current = bars.back();
    
    // Current volume normalized
    if (bars.size() >= 10) {
        float avg_vol = 0.0f;
        for (size_t i = bars.size() - 10; i < bars.size(); ++i) {
            avg_vol += bars[i].volume;
        }
        avg_vol /= 10.0f;
        features.push_back(current.volume / (avg_vol + 1e-8f));
    } else {
        features.push_back(1.0f);
    }
    
    // Volume ratios to different period averages
    std::vector<int> periods = {5, 10, 20};
    for (int period : periods) {
        if (bars.size() >= period) {
            float avg_vol = 0.0f;
            for (int i = 1; i <= period; ++i) {
                avg_vol += bars[bars.size() - i].volume;
            }
            avg_vol /= period;
            features.push_back(current.volume / (avg_vol + 1e-8f));
        } else {
            features.push_back(1.0f);
        }
    }
    
    // Volume trend
    if (bars.size() >= 10) {
        float recent_avg = 0.0f, older_avg = 0.0f;
        for (int i = 0; i < 5; ++i) {
            recent_avg += bars[bars.size() - 1 - i].volume;
            if (bars.size() > 10) {
                older_avg += bars[bars.size() - 6 - i].volume;
            }
        }
        recent_avg /= 5.0f;
        older_avg /= 5.0f;
        
        if (older_avg > 0) {
            features.push_back((recent_avg / older_avg) - 1.0f);
        } else {
            features.push_back(0.0f);
        }
    } else {
        features.push_back(0.0f);
    }
    
    // Volume volatility
    if (bars.size() >= 10) {
        float vol_mean = 0.0f;
        for (size_t i = bars.size() - 10; i < bars.size(); ++i) {
            vol_mean += bars[i].volume;
        }
        vol_mean /= 10.0f;
        
        float vol_var = 0.0f;
        for (size_t i = bars.size() - 10; i < bars.size(); ++i) {
            vol_var += (bars[i].volume - vol_mean) * (bars[i].volume - vol_mean);
        }
        vol_var /= 9.0f;
        float vol_std = std::sqrt(vol_var);
        features.push_back(vol_std / (vol_mean + 1e-8f));
    } else {
        features.push_back(0.0f);
    }
    
    // Pad to exactly 20 features
    while (features.size() < 20) {
        features.push_back(0.0f);
    }
    features.resize(20);
    
    return features;
}

std::vector<float> FeaturePipeline::generate_technical_features(const std::vector<Bar>& bars) {
    std::vector<float> features;
    features.reserve(40); // Target 40 technical features
    
    if (bars.empty()) {
        features.resize(40, 0.0f);
        return features;
    }
    
    // Extract close prices
    std::vector<float> closes;
    for (const auto& bar : bars) {
        closes.push_back(bar.close);
    }
    
    // RSI features
    auto rsi_14 = calculate_rsi(closes, 14);
    if (!rsi_14.empty()) {
        features.push_back(rsi_14.back() / 100.0f - 0.5f); // Normalize to [-0.5, 0.5]
    } else {
        features.push_back(0.0f);
    }
    
    auto rsi_7 = calculate_rsi(closes, 7);
    if (!rsi_7.empty()) {
        features.push_back(rsi_7.back() / 100.0f - 0.5f);
    } else {
        features.push_back(0.0f);
    }
    
    // Bollinger Bands approximation
    if (closes.size() >= 20) {
        auto sma_20 = calculate_sma(closes, 20);
        if (!sma_20.empty()) {
            // Calculate standard deviation
            float sum_sq = 0.0f;
            for (size_t i = closes.size() - 20; i < closes.size(); ++i) {
                float diff = closes[i] - sma_20.back();
                sum_sq += diff * diff;
            }
            float std_dev = std::sqrt(sum_sq / 20);
            
            float upper = sma_20.back() + 2.0f * std_dev;
            float lower = sma_20.back() - 2.0f * std_dev;
            
            // %B indicator
            float percent_b = (closes.back() - lower) / (upper - lower + 1e-8f);
            features.push_back(percent_b);
            
            // Bandwidth
            float bandwidth = (upper - lower) / sma_20.back();
            features.push_back(bandwidth);
        } else {
            features.push_back(0.5f);
            features.push_back(0.0f);
        }
    } else {
        features.push_back(0.5f);
        features.push_back(0.0f);
    }
    
    // Pad to exactly 40 features
    while (features.size() < 40) {
        features.push_back(0.0f);
    }
    features.resize(40);
    
    return features;
}

std::vector<float> FeaturePipeline::generate_temporal_features(const Bar& current_bar) {
    std::vector<float> features;
    features.reserve(28); // Target 28 temporal features
    
    auto time_t = static_cast<std::time_t>(current_bar.ts_utc_epoch);
    auto tm = *std::localtime(&time_t);
    
    // Hour of day (8 features - one-hot encoding for market hours)
    for (int h = 0; h < 8; ++h) {
        features.push_back((tm.tm_hour >= 9 + h && tm.tm_hour < 10 + h) ? 1.0f : 0.0f);
    }
    
    // Day of week (7 features)
    for (int d = 0; d < 7; ++d) {
        features.push_back((tm.tm_wday == d) ? 1.0f : 0.0f);
    }
    
    // Month (12 features)
    for (int m = 0; m < 12; ++m) {
        features.push_back((tm.tm_mon == m) ? 1.0f : 0.0f);
    }
    
    // Market session (1 feature - RTH vs ETH)
    bool is_rth = (tm.tm_hour >= 9 && tm.tm_hour < 16);
    features.push_back(is_rth ? 1.0f : 0.0f);
    
    // Ensure exactly 28 features
    features.resize(28, 0.0f);
    
    return features;
}

std::vector<float> FeaturePipeline::calculate_sma(const std::vector<float>& prices, int period) {
    std::vector<float> result;
    if (prices.size() < period) return result;
    
    result.reserve(prices.size());
    
    for (size_t i = period - 1; i < prices.size(); ++i) {
        float sum = std::accumulate(prices.begin() + i - period + 1, prices.begin() + i + 1, 0.0f);
        result.push_back(sum / period);
    }
    
    return result;
}

std::vector<float> FeaturePipeline::calculate_ema(const std::vector<float>& prices, int period) {
    std::vector<float> result;
    if (prices.empty()) return result;
    
    result.reserve(prices.size());
    float multiplier = 2.0f / (period + 1);
    result.push_back(prices[0]);
    
    for (size_t i = 1; i < prices.size(); ++i) {
        float ema_val = (prices[i] - result[i-1]) * multiplier + result[i-1];
        result.push_back(ema_val);
    }
    
    return result;
}

std::vector<float> FeaturePipeline::calculate_rsi(const std::vector<float>& prices, int period) {
    std::vector<float> result;
    if (prices.size() < period + 1) return result;
    
    std::vector<float> gains, losses;
    
    // Calculate price changes
    for (size_t i = 1; i < prices.size(); ++i) {
        float change = prices[i] - prices[i-1];
        gains.push_back(change > 0 ? change : 0);
        losses.push_back(change < 0 ? -change : 0);
    }
    
    // Calculate RSI
    auto avg_gains = calculate_ema(gains, period);
    auto avg_losses = calculate_ema(losses, period);
    
    result.reserve(avg_gains.size());
    for (size_t i = 0; i < avg_gains.size(); ++i) {
        if (avg_losses[i] == 0) {
            result.push_back(100.0f);
        } else {
            float rs = avg_gains[i] / avg_losses[i];
            result.push_back(100.0f - (100.0f / (1.0f + rs)));
        }
    }
    
    return result;
}

void FeaturePipeline::update_feature_statistics(const std::vector<float>& features) {
    for (size_t i = 0; i < features.size() && i < feature_stats_.size(); ++i) {
        if (!std::isnan(features[i]) && !std::isinf(features[i])) {
            feature_stats_[i].update(features[i]);
        }
    }
}

std::vector<float> FeaturePipeline::normalize_features(const std::vector<float>& features) {
    std::vector<float> normalized;
    normalized.reserve(features.size());
    
    for (size_t i = 0; i < features.size(); ++i) {
        if (i < feature_stats_.size() && feature_stats_[i].is_initialized()) {
            float value = features[i];
            if (!std::isnan(value) && !std::isinf(value)) {
                // Z-score normalization
                float normalized_value = (value - feature_stats_[i].mean()) / (feature_stats_[i].std() + 1e-8f);
                normalized.push_back(normalized_value);
            } else {
                normalized.push_back(0.0f);
            }
        } else {
            normalized.push_back(features[i]);
        }
    }
    
    return normalized;
}

} // namespace sentio

```

## 📄 **FILE 3 of 11**: temp_mega_doc/feature_pipeline.hpp

**File Information**:
- **Path**: `temp_mega_doc/feature_pipeline.hpp`

- **Size**: 48 lines
- **Modified**: 2025-09-19 21:12:54

- **Type**: .hpp

```text
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
    static constexpr int TOTAL_FEATURES = 128; // Target feature count
    
    explicit FeaturePipeline(const TransformerConfig::Features& config);
    
    // Main interface
    TransformerFeatureMatrix generate_features(const std::vector<Bar>& bars);
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
    
    // Technical indicators
    std::vector<float> calculate_sma(const std::vector<float>& prices, int period);
    std::vector<float> calculate_ema(const std::vector<float>& prices, int period);
    std::vector<float> calculate_rsi(const std::vector<float>& prices, int period = 14);
    
    // Normalization
    void update_feature_statistics(const std::vector<float>& features);
    std::vector<float> normalize_features(const std::vector<float>& features);
};

} // namespace sentio

```

## 📄 **FILE 4 of 11**: temp_mega_doc/online_trainer.hpp

**File Information**:
- **Path**: `temp_mega_doc/online_trainer.hpp`

- **Size**: 159 lines
- **Modified**: 2025-09-19 21:12:54

- **Type**: .hpp

```text
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

```

## 📄 **FILE 5 of 11**: temp_mega_doc/strategy_initialization.cpp

**File Information**:
- **Path**: `temp_mega_doc/strategy_initialization.cpp`

- **Size**: 30 lines
- **Modified**: 2025-09-19 21:12:54

- **Type**: .cpp

```text
#include "sentio/base_strategy.hpp"
// REMOVED: strategy_ire.hpp - unused legacy strategy
// REMOVED: strategy_bollinger_squeeze_breakout.hpp - unused legacy strategy
// REMOVED: strategy_kochi_ppo.hpp - unused legacy strategy
#include "sentio/strategy_tfa.hpp"
#include "sentio/strategy_signal_or.hpp"
#include "sentio/strategy_transformer.hpp"
// REMOVED: rsi_strategy.hpp - unused legacy strategy

namespace sentio {

/**
 * Initialize all strategies in the StrategyFactory
 * This replaces the StrategyRegistry system
 * 
 * Note: Individual strategy files use REGISTER_STRATEGY macro
 * which automatically registers strategies, so manual registration
 * is no longer needed here.
 */
bool initialize_strategies() {
    auto& factory = StrategyFactory::instance();
    
    // All strategies are now automatically registered via REGISTER_STRATEGY macro
    // in their respective source files. No manual registration needed.
    
    std::cout << "Registered " << factory.get_available_strategies().size() << " strategies" << std::endl;
    return true;
}

} // namespace sentio

```

## 📄 **FILE 6 of 11**: temp_mega_doc/strategy_transformer.cpp

**File Information**:
- **Path**: `temp_mega_doc/strategy_transformer.cpp`

- **Size**: 311 lines
- **Modified**: 2025-09-19 21:12:54

- **Type**: .cpp

```text
#include "sentio/strategy_transformer.hpp"
#include "sentio/time_utils.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <filesystem>

namespace sentio {

TransformerStrategy::TransformerStrategy()
    : BaseStrategy("Transformer")
    , cfg_()
{
    initialize_model();
}

TransformerStrategy::TransformerStrategy(const TransformerCfg& cfg)
    : BaseStrategy("Transformer")
    , cfg_(cfg)
{
    initialize_model();
}

void TransformerStrategy::initialize_model() {
    try {
        // Create transformer configuration
        TransformerConfig model_config;
        model_config.feature_dim = cfg_.feature_dim;
        model_config.sequence_length = cfg_.sequence_length;
        model_config.d_model = cfg_.d_model;
        model_config.num_heads = cfg_.num_heads;
        model_config.num_layers = cfg_.num_layers;
        model_config.ffn_hidden = cfg_.ffn_hidden;
        model_config.dropout = cfg_.dropout;
        
        // Initialize model
        model_ = std::make_shared<TransformerModel>(model_config);
        
        // Try to load pre-trained model if it exists
        if (std::filesystem::exists(cfg_.model_path)) {
            std::cout << "Loading pre-trained transformer model from: " << cfg_.model_path << std::endl;
            model_->load_model(cfg_.model_path);
        } else {
            std::cout << "No pre-trained model found at: " << cfg_.model_path << std::endl;
            std::cout << "Using randomly initialized model" << std::endl;
        }
        
        model_->optimize_for_inference();
        
        // Initialize feature pipeline
        TransformerConfig::Features feature_config;
        feature_config.normalization = TransformerConfig::Features::NormalizationMethod::Z_SCORE;
        feature_config.decay_factor = 0.999f;
        feature_pipeline_ = std::make_unique<FeaturePipeline>(feature_config);
        
        // Initialize online trainer if enabled
        if (cfg_.enable_online_training) {
            OnlineTrainer::OnlineConfig trainer_config;
            trainer_config.update_interval_minutes = cfg_.update_interval_minutes;
            trainer_config.min_samples_for_update = cfg_.min_samples_for_update;
            trainer_config.base_learning_rate = 0.0001f;
            trainer_config.replay_buffer_size = 10000;
            trainer_config.enable_regime_detection = true;
            
            online_trainer_ = std::make_unique<OnlineTrainer>(model_, trainer_config);
            std::cout << "Online training enabled with " << cfg_.update_interval_minutes << " minute intervals" << std::endl;
        }
        
        model_initialized_ = true;
        std::cout << "Transformer strategy initialized successfully" << std::endl;
        std::cout << "Model parameters: " << model_->get_parameter_count() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize transformer strategy: " << e.what() << std::endl;
        model_initialized_ = false;
    }
}

ParameterMap TransformerStrategy::get_default_params() const {
    ParameterMap defaults;
    defaults["buy_threshold"] = cfg_.buy_threshold;
    defaults["sell_threshold"] = cfg_.sell_threshold;
    defaults["strong_threshold"] = cfg_.strong_threshold;
    defaults["conf_floor"] = cfg_.conf_floor;
    return defaults;
}

ParameterSpace TransformerStrategy::get_param_space() const {
    return {
        {"buy_threshold", {ParamType::FLOAT, 0.5, 0.8, 0.6}},
        {"sell_threshold", {ParamType::FLOAT, 0.2, 0.5, 0.4}},
        {"strong_threshold", {ParamType::FLOAT, 0.7, 0.9, 0.8}},
        {"conf_floor", {ParamType::FLOAT, 0.4, 0.6, 0.5}}
    };
}

void TransformerStrategy::apply_params() {
    if (params_.count("buy_threshold")) {
        cfg_.buy_threshold = static_cast<float>(params_["buy_threshold"]);
    }
    if (params_.count("sell_threshold")) {
        cfg_.sell_threshold = static_cast<float>(params_["sell_threshold"]);
    }
    if (params_.count("strong_threshold")) {
        cfg_.strong_threshold = static_cast<float>(params_["strong_threshold"]);
    }
    if (params_.count("conf_floor")) {
        cfg_.conf_floor = static_cast<float>(params_["conf_floor"]);
    }
}


void TransformerStrategy::update_bar_history(const Bar& bar) {
    bar_history_.push_back(bar);
    
    // Keep only the required sequence length + some buffer
    const size_t max_history = cfg_.sequence_length + 50;
    while (bar_history_.size() > max_history) {
        bar_history_.pop_front();
    }
}

std::vector<Bar> TransformerStrategy::convert_to_transformer_bars(const std::vector<Bar>& sentio_bars) const {
    std::vector<Bar> transformer_bars;
    transformer_bars.reserve(sentio_bars.size());
    
    for (const auto& sentio_bar : sentio_bars) {
        // Convert Sentio Bar to Transformer Bar format
        Bar transformer_bar;
        transformer_bar.open = sentio_bar.open;
        transformer_bar.high = sentio_bar.high;
        transformer_bar.low = sentio_bar.low;
        transformer_bar.close = sentio_bar.close;
        transformer_bar.volume = sentio_bar.volume;
        transformer_bar.ts_utc_epoch = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count(); // Use current time
        
        transformer_bars.push_back(transformer_bar);
    }
    
    return transformer_bars;
}

TransformerFeatureMatrix TransformerStrategy::generate_features_for_bars(const std::vector<Bar>& bars, int end_index) {
    if (!model_initialized_ || bars.empty() || end_index < 0) {
        return torch::zeros({1, cfg_.feature_dim});
    }
    
    // Get the bars up to end_index
    std::vector<Bar> relevant_bars;
    int start_idx = std::max(0, end_index - cfg_.sequence_length + 1);
    
    for (int i = start_idx; i <= end_index && i < static_cast<int>(bars.size()); ++i) {
        relevant_bars.push_back(bars[i]);
    }
    
    // Convert to transformer bar format
    auto transformer_bars = convert_to_transformer_bars(relevant_bars);
    
    // Generate features using the pipeline
    try {
        auto features = feature_pipeline_->generate_features(transformer_bars);
        
        // Ensure we have the right sequence length
        if (features.size(1) < cfg_.sequence_length) {
            // Pad with zeros if we don't have enough history
            auto padded = torch::zeros({1, cfg_.sequence_length, cfg_.feature_dim});
            int start_pos = cfg_.sequence_length - features.size(1);
            padded.slice(1, start_pos, cfg_.sequence_length) = features.squeeze(0);
            return padded;
        } else if (features.size(1) > cfg_.sequence_length) {
            // Take the last sequence_length features
            return features.slice(1, features.size(1) - cfg_.sequence_length, features.size(1));
        }
        
        return features.unsqueeze(0); // Add batch dimension
        
    } catch (const std::exception& e) {
        std::cerr << "Error generating features: " << e.what() << std::endl;
        return torch::zeros({1, cfg_.sequence_length, cfg_.feature_dim});
    }
}

double TransformerStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
    if (!model_initialized_ || bars.empty() || current_index < 0) {
        return 0.5f; // Neutral
    }
    
    try {
        // Generate features for the current context
        auto features = generate_features_for_bars(bars, current_index);
        
        // Run inference
        model_->eval();
        torch::NoGradGuard no_grad;
        
        auto prediction_tensor = model_->forward(features);
        float raw_prediction = prediction_tensor.item<float>();
        
        // Convert to probability using sigmoid
        float probability = 1.0f / (1.0f + std::exp(-raw_prediction));
        
        // Clamp to reasonable bounds
        probability = std::max(0.01f, std::min(0.99f, probability));
        
        // Update performance metrics
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            current_metrics_.samples_processed++;
        }
        
        return probability;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in calculate_probability: " << e.what() << std::endl;
        return 0.5f; // Return neutral on error
    }
}

void TransformerStrategy::maybe_trigger_training() {
    if (!cfg_.enable_online_training || !online_trainer_ || is_training_.load()) {
        return;
    }
    
    try {
        // Check if we should update the model
        if (online_trainer_->should_update_model()) {
            std::cout << "Triggering transformer model update..." << std::endl;
            is_training_ = true;
            
            auto result = online_trainer_->update_model();
            if (result.success) {
                std::cout << "Model update completed successfully" << std::endl;
            } else {
                std::cerr << "Model update failed: " << result.error_message << std::endl;
            }
            
            is_training_ = false;
        }
        
        // Check for regime changes
        if (online_trainer_->detect_regime_change()) {
            std::cout << "Market regime change detected, adapting model..." << std::endl;
            online_trainer_->adapt_to_regime_change();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in maybe_trigger_training: " << e.what() << std::endl;
        is_training_ = false;
    }
}


std::vector<AllocationDecision> TransformerStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol, 
    const std::string& bear3x_symbol
) const {
    
    std::vector<AllocationDecision> decisions;
    
    if (!model_initialized_ || bars.empty() || current_index < 0) {
        return decisions;
    }
    
    // Get probability from strategy
    double probability = const_cast<TransformerStrategy*>(this)->calculate_probability(bars, current_index);
    
    // **PROFIT MAXIMIZATION**: Always deploy 100% of capital with maximum leverage
    if (probability > cfg_.strong_threshold) {
        // Strong buy: 100% TQQQ (3x leveraged long)
        decisions.push_back({bull3x_symbol, 1.0, "Transformer strong buy: 100% " + bull3x_symbol + " (3x leverage)"});
        
    } else if (probability > cfg_.buy_threshold) {
        // Moderate buy: 100% QQQ (1x long)
        decisions.push_back({base_symbol, 1.0, "Transformer moderate buy: 100% " + base_symbol});
        
    } else if (probability < (1.0f - cfg_.strong_threshold)) {
        // Strong sell: 100% SQQQ (3x leveraged short)
        decisions.push_back({bear3x_symbol, 1.0, "Transformer strong sell: 100% " + bear3x_symbol + " (3x inverse)"});
        
    } else if (probability < cfg_.sell_threshold) {
        // Weak sell: 100% PSQ (1x inverse)
        decisions.push_back({"PSQ", 1.0, "Transformer weak sell: 100% PSQ (1x inverse)"});
        
    } else {
        // Neutral: Stay in cash (rare case)
        // No positions needed
    }
    
    // Ensure all instruments are flattened if not in allocation
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol, "PSQ"};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, "Transformer: Flatten unused instrument"});
        }
    }
    
    return decisions;
}

// Register the strategy
REGISTER_STRATEGY(TransformerStrategy, "transformer");

} // namespace sentio

```

## 📄 **FILE 7 of 11**: temp_mega_doc/strategy_transformer.hpp

**File Information**:
- **Path**: `temp_mega_doc/strategy_transformer.hpp`

- **Size**: 93 lines
- **Modified**: 2025-09-19 21:12:54

- **Type**: .hpp

```text
#pragma once

#include "base_strategy.hpp"
#include "transformer_strategy_core.hpp"
#include "transformer_model.hpp"
#include "feature_pipeline.hpp"
#include "online_trainer.hpp"
#include "adaptive_allocation_manager.hpp"
#include <memory>
#include <deque>
#include <torch/torch.h>

namespace sentio {

struct TransformerCfg {
    // Model configuration
    int feature_dim = 128;
    int sequence_length = 64;
    int d_model = 256;
    int num_heads = 8;
    int num_layers = 6;
    int ffn_hidden = 1024;
    float dropout = 0.1f;
    
    // Strategy parameters
    float buy_threshold = 0.6f;
    float sell_threshold = 0.4f;
    float strong_threshold = 0.8f;
    float conf_floor = 0.5f;
    
    // Training parameters
    bool enable_online_training = true;
    int update_interval_minutes = 60;
    int min_samples_for_update = 1000;
    
    // Model paths
    std::string model_path = "artifacts/Transformer/v1/model.pt";
    std::string artifacts_dir = "artifacts/Transformer/";
    std::string version = "v1";
};

class TransformerStrategy : public BaseStrategy {
public:
    TransformerStrategy();
    explicit TransformerStrategy(const TransformerCfg& cfg);
    
    // BaseStrategy interface
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override;
    
    // Allocation decisions for profit maximization
    std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, 
        int current_index,
        const std::string& base_symbol = "QQQ",
        const std::string& bull3x_symbol = "TQQQ", 
        const std::string& bear3x_symbol = "SQQQ"
    ) const;

private:
    // Configuration
    TransformerCfg cfg_;
    
    // Core components
    std::shared_ptr<TransformerModel> model_;
    std::unique_ptr<FeaturePipeline> feature_pipeline_;
    std::unique_ptr<OnlineTrainer> online_trainer_;
    
    // Feature management
    std::deque<Bar> bar_history_;
    std::vector<double> current_features_;
    
    // Model state
    bool model_initialized_ = false;
    std::atomic<bool> is_training_{false};
    
    // Performance tracking
    mutable std::mutex metrics_mutex_;
    PerformanceMetrics current_metrics_;
    
    // Helper methods
    void initialize_model();
    void update_bar_history(const Bar& bar);
    void maybe_trigger_training();
    TransformerFeatureMatrix generate_features_for_bars(const std::vector<Bar>& bars, int end_index);
    
    // Feature conversion
    std::vector<Bar> convert_to_transformer_bars(const std::vector<Bar>& sentio_bars) const;
};

} // namespace sentio

```

## 📄 **FILE 8 of 11**: temp_mega_doc/transformer_model.cpp

**File Information**:
- **Path**: `temp_mega_doc/transformer_model.cpp`

- **Size**: 101 lines
- **Modified**: 2025-09-19 21:12:54

- **Type**: .cpp

```text
#include "sentio/transformer_model.hpp"
#include <cmath>
#include <iostream>

namespace sentio {

TransformerModel::TransformerModel(const TransformerConfig& config) : config_(config) {
    // Input projection
    input_projection_ = register_module("input_projection", 
        torch::nn::Linear(config.feature_dim, config.d_model));
    
    // Transformer encoder
    torch::nn::TransformerEncoderLayerOptions encoder_layer_options(config.d_model, config.num_heads);
    encoder_layer_options.dim_feedforward(config.d_model * 4);
    encoder_layer_options.dropout(config.dropout);
    // Note: batch_first option may not be available in all PyTorch versions
    // encoder_layer_options.batch_first(true);
    
    auto encoder_layer = torch::nn::TransformerEncoderLayer(encoder_layer_options);
    
    torch::nn::TransformerEncoderOptions encoder_options(encoder_layer, config.num_layers);
    transformer_ = register_module("transformer", torch::nn::TransformerEncoder(encoder_options));
    
    // Layer normalization
    layer_norm_ = register_module("layer_norm", torch::nn::LayerNorm(std::vector<int64_t>{config.d_model}));
    
    // Output projection
    output_projection_ = register_module("output_projection", 
        torch::nn::Linear(config.d_model, 1));
    
    // Dropout
    dropout_ = register_module("dropout", torch::nn::Dropout(config.dropout));
    
    // Create positional encoding
    create_positional_encoding();
}

void TransformerModel::create_positional_encoding() {
    pos_encoding_ = torch::zeros({config_.sequence_length, config_.d_model});
    
    auto position = torch::arange(0, config_.sequence_length).unsqueeze(1).to(torch::kFloat);
    auto div_term = torch::exp(torch::arange(0, config_.d_model, 2).to(torch::kFloat) * 
                              -(std::log(10000.0) / config_.d_model));
    
    pos_encoding_.slice(1, 0, config_.d_model, 2) = torch::sin(position * div_term);
    pos_encoding_.slice(1, 1, config_.d_model, 2) = torch::cos(position * div_term);
    
    pos_encoding_ = pos_encoding_.unsqueeze(0); // Add batch dimension
}

torch::Tensor TransformerModel::forward(const torch::Tensor& input) {
    // input shape: [batch_size, sequence_length, feature_dim]
    // auto batch_size = input.size(0);  // Unused for now
    auto seq_len = input.size(1);
    
    // Input projection
    auto x = input_projection_->forward(input);
    
    // Add positional encoding
    auto pos_enc = pos_encoding_.slice(1, 0, seq_len).to(x.device());
    x = x + pos_enc;
    x = dropout_->forward(x);
    
    // Transformer encoding
    x = transformer_->forward(x);
    
    // Layer normalization
    x = layer_norm_->forward(x);
    
    // Global average pooling
    x = torch::mean(x, 1); // [batch_size, d_model]
    
    // Output projection
    auto output = output_projection_->forward(x); // [batch_size, 1]
    
    return output;
}

void TransformerModel::save_model(const std::string& path) {
    torch::save(shared_from_this(), path);
}

void TransformerModel::load_model(const std::string& path) {
    auto model_ptr = shared_from_this();
    torch::load(model_ptr, path);
}

void TransformerModel::optimize_for_inference() {
    eval();
    // Additional optimizations could be added here
}

size_t TransformerModel::get_parameter_count() const {
    size_t count = 0;
    for (const auto& param : parameters()) {
        count += param.numel();
    }
    return count;
}

} // namespace sentio

```

## 📄 **FILE 9 of 11**: temp_mega_doc/transformer_model.hpp

**File Information**:
- **Path**: `temp_mega_doc/transformer_model.hpp`

- **Size**: 42 lines
- **Modified**: 2025-09-19 21:12:54

- **Type**: .hpp

```text
#pragma once

#include "transformer_strategy_core.hpp"
#include <torch/torch.h>
#include <memory>
#include <string>

namespace sentio {

// Simple transformer model implementation for Sentio
class TransformerModel : public torch::nn::Module {
public:
    explicit TransformerModel(const TransformerConfig& config);
    
    // Forward pass
    torch::Tensor forward(const torch::Tensor& input);
    
    // Model management
    void save_model(const std::string& path);
    void load_model(const std::string& path);
    void optimize_for_inference();
    
    // Utilities
    size_t get_parameter_count() const;
    
private:
    TransformerConfig config_;
    
    // Model components
    torch::nn::Linear input_projection_{nullptr};
    torch::nn::TransformerEncoder transformer_{nullptr};
    torch::nn::LayerNorm layer_norm_{nullptr};
    torch::nn::Linear output_projection_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    
    // Positional encoding
    torch::Tensor pos_encoding_;
    
    void create_positional_encoding();
};

} // namespace sentio
```

## 📄 **FILE 10 of 11**: temp_mega_doc/transformer_strategy_core.hpp

**File Information**:
- **Path**: `temp_mega_doc/transformer_strategy_core.hpp`

- **Size**: 251 lines
- **Modified**: 2025-09-19 21:12:54

- **Type**: .hpp

```text
#pragma once

#include <memory>
#include <vector>
#include <atomic>
#include <shared_mutex>
#include <chrono>
#include <string>
#include <unordered_map>
#include <torch/torch.h>

namespace sentio {

// Forward declarations
struct Bar;
class Fill;
class BaseStrategy;
struct StrategySignal;

// ==================== Core Data Structures ====================

struct PriceFeatures {
    std::vector<float> ohlc_normalized;     // 4 features
    std::vector<float> returns;             // 5 features (1m, 5m, 15m, 1h, 4h)
    std::vector<float> log_returns;         // 5 features
    std::vector<float> moving_averages;     // 8 features (SMA/EMA 5,10,20,50)
    std::vector<float> bollinger_bands;     // 3 features (upper, lower, %B)
    std::vector<float> rsi_family;          // 4 features (RSI 14, Stoch RSI)
    std::vector<float> momentum;            // 6 features (ROC, Williams %R, etc.)
    std::vector<float> volatility;          // 5 features (ATR, realized vol, etc.)
    
    static constexpr int TOTAL_FEATURES = 40;
};

struct VolumeFeatures {
    std::vector<float> volume_indicators;   // 8 features (VWAP, OBV, etc.)
    std::vector<float> volume_ratios;       // 4 features (vol/avg_vol ratios)
    std::vector<float> price_volume;        // 4 features (PVT, MFI, etc.)
    std::vector<float> volume_profile;      // 4 features (VPOC, VAH, VAL, etc.)
    
    static constexpr int TOTAL_FEATURES = 20;
};

struct MicrostructureFeatures {
    std::vector<float> spread_metrics;      // 5 features (bid-ask spread analysis)
    std::vector<float> order_flow;          // 8 features (tick direction, etc.)
    std::vector<float> market_impact;       // 4 features (Kyle's lambda, etc.)
    std::vector<float> liquidity_metrics;   // 4 features (market depth, etc.)
    std::vector<float> regime_indicators;   // 4 features (volatility regime, etc.)
    
    static constexpr int TOTAL_FEATURES = 25;
};

struct CrossAssetFeatures {
    std::vector<float> correlation_features; // 5 features (SPY, VIX correlation)
    std::vector<float> sector_rotation;      // 5 features (sector momentum)
    std::vector<float> macro_indicators;     // 5 features (yield curve, etc.)
    
    static constexpr int TOTAL_FEATURES = 15;
};

struct TemporalFeatures {
    std::vector<float> time_of_day;         // 8 features (hour encoding)
    std::vector<float> day_of_week;         // 7 features (weekday encoding)
    std::vector<float> monthly_seasonal;    // 12 features (month encoding)
    std::vector<float> market_session;      // 1 feature (RTH/ETH indicator)
    
    static constexpr int TOTAL_FEATURES = 28;
};

using FeatureVector = torch::Tensor;
using TransformerFeatureMatrix = torch::Tensor;

// ==================== Configuration Structures ====================

struct TransformerConfig {
    // Model architecture
    int feature_dim = 128;
    int sequence_length = 64;
    int d_model = 256;
    int num_heads = 8;
    int num_layers = 6;
    int ffn_hidden = 1024;
    float dropout = 0.1f;
    
    // Performance requirements
    float max_inference_latency_ms = 1.0f;
    float max_memory_usage_mb = 1024.0f;
    
    // Training configuration
    struct Training {
        struct Offline {
            int batch_size = 256;
            float learning_rate = 0.001f;
            int max_epochs = 1000;
            int patience = 50;
            float min_delta = 1e-6f;
            std::string optimizer = "AdamW";
            float weight_decay = 1e-4f;
        } offline;
        
        struct Online {
            int update_interval_minutes = 60;
            float learning_rate = 0.0001f;
            int replay_buffer_size = 10000;
            bool enable_regime_detection = true;
            float regime_change_threshold = 0.15f;
            int min_samples_for_update = 1000;
        } online;
    } training;
    
    // Feature configuration
    struct Features {
        enum class NormalizationMethod {
            Z_SCORE, MIN_MAX, ROBUST, QUANTILE_UNIFORM
        };
        NormalizationMethod normalization = NormalizationMethod::Z_SCORE;
        float decay_factor = 0.999f;
    } features;
};

// ==================== Performance Metrics ====================

struct PerformanceMetrics {
    float avg_inference_latency_ms = 0.0f;
    float p95_inference_latency_ms = 0.0f;
    float p99_inference_latency_ms = 0.0f;
    float recent_accuracy = 0.0f;
    float rolling_sharpe_ratio = 0.0f;
    float current_drawdown = 0.0f;
    float memory_usage_mb = 0.0f;
    bool is_training_active = false;
    float training_loss = 0.0f;
    int samples_processed = 0;
};

struct ValidationMetrics {
    float directional_accuracy = 0.0f;
    float sharpe_ratio = 0.0f;
    float max_drawdown = 0.0f;
    float win_rate = 0.0f;
    float profit_factor = 0.0f;
    bool passes_validation = false;
};

struct TrainingResult {
    bool success = false;
    std::string model_path;
    ValidationMetrics validation_metrics;
    std::chrono::system_clock::time_point training_end_time;
    int total_epochs = 0;
};

// ==================== Model Status ====================

enum class ModelStatus {
    UNINITIALIZED,
    LOADING,
    READY,
    TRAINING,
    UPDATING,
    ERROR,
    DISABLED
};

struct UpdateResult {
    bool success = false;
    std::string error_message;
    ValidationMetrics post_update_metrics;
    std::chrono::milliseconds update_duration{0};
};

// ==================== Risk Management ====================

struct RiskLimits {
    float max_position_size = 1.0f;
    float max_daily_trades = 100.0f;
    float max_drawdown_threshold = 0.10f;
    float min_confidence_threshold = 0.6f;
};

struct Alert {
    std::string metric_name;
    float current_value;
    float threshold;
    std::chrono::system_clock::time_point timestamp;
    std::string message;
};

// ==================== Running Statistics for Feature Normalization ====================

class RunningStats {
public:
    RunningStats(float decay_factor = 0.999f) : decay_factor_(decay_factor) {}
    
    void update(float value) {
        if (!initialized_) {
            mean_ = value;
            var_ = 0.0f;
            min_ = max_ = value;
            initialized_ = true;
            count_ = 1;
        } else {
            // Exponential moving average
            mean_ = decay_factor_ * mean_ + (1.0f - decay_factor_) * value;
            float delta = value - mean_;
            var_ = decay_factor_ * var_ + (1.0f - decay_factor_) * delta * delta;
            min_ = std::min(min_, value);
            max_ = std::max(max_, value);
            count_++;
        }
    }
    
    float mean() const { return mean_; }
    float std() const { return std::sqrt(var_); }
    float min() const { return min_; }
    float max() const { return max_; }
    int count() const { return count_; }
    bool is_initialized() const { return initialized_; }
    
private:
    float decay_factor_;
    float mean_ = 0.0f;
    float var_ = 0.0f;
    float min_ = 0.0f;
    float max_ = 0.0f;
    int count_ = 0;
    bool initialized_ = false;
};

// ==================== Core Constants ====================

struct LatencyRequirements {
    static constexpr int MAX_INFERENCE_LATENCY_US = 1000;    // 1ms
    static constexpr int TARGET_INFERENCE_LATENCY_US = 500;  // 0.5ms
    static constexpr int MAX_FEATURE_GEN_LATENCY_US = 500;   // 0.5ms
    static constexpr int MAX_MODEL_UPDATE_TIME_S = 300;      // 5 minutes
    static constexpr int MAX_MEMORY_USAGE_MB = 1024;         // 1GB
    static constexpr int MAX_GPU_MEMORY_MB = 2048;           // 2GB
};

struct AccuracyRequirements {
    static constexpr float MIN_DIRECTIONAL_ACCURACY = 0.52f;
    static constexpr float TARGET_DIRECTIONAL_ACCURACY = 0.55f;
    static constexpr float MIN_SHARPE_RATIO = 1.0f;
    static constexpr float TARGET_SHARPE_RATIO = 2.0f;
    static constexpr float MAX_DRAWDOWN = 0.15f;
    static constexpr float MIN_WIN_RATE = 0.45f;
};

} // namespace sentio

```

## 📄 **FILE 11 of 11**: temp_mega_doc/transformer_trainer_main.cpp

**File Information**:
- **Path**: `temp_mega_doc/transformer_trainer_main.cpp`

- **Size**: 457 lines
- **Modified**: 2025-09-19 21:12:54

- **Type**: .cpp

```text
#include "sentio/transformer_model.hpp"
#include "sentio/feature_pipeline.hpp"
#include "sentio/online_trainer.hpp"
#include "sentio/transformer_strategy_core.hpp"
#include "sentio/core.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <random>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <limits>
#include <numeric>

namespace sentio {

struct TrainingConfig {
    std::string data_path = "data/equities/QQQ_1min.csv";
    std::string output_dir = "artifacts/Transformer/v1";
    int epochs = 20;
    int batch_size = 32;
    int sequence_length = 64;
    int feature_dim = 128;
    int d_model = 256;
    int num_heads = 8;
    int num_layers = 6;
    float learning_rate = 0.001f;
    float dropout = 0.1f;
    bool use_cuda = false;
    int validation_split_percent = 20;
    int progress_update_frequency = 10; // Update progress every N batches
};

class TransformerTrainer {
public:
    explicit TransformerTrainer(const TrainingConfig& config) : config_(config) {
        // Initialize model configuration
        model_config_.feature_dim = config.feature_dim;
        model_config_.sequence_length = config.sequence_length;
        model_config_.d_model = config.d_model;
        model_config_.num_heads = config.num_heads;
        model_config_.num_layers = config.num_layers;
        model_config_.dropout = config.dropout;
        
        // Initialize feature configuration
        feature_config_.normalization = TransformerConfig::Features::NormalizationMethod::Z_SCORE;
        feature_config_.decay_factor = 0.999f;
        
        std::cout << "Transformer Trainer initialized with:" << std::endl;
        std::cout << "  Feature dim: " << config.feature_dim << std::endl;
        std::cout << "  Sequence length: " << config.sequence_length << std::endl;
        std::cout << "  Model dim: " << config.d_model << std::endl;
        std::cout << "  Attention heads: " << config.num_heads << std::endl;
        std::cout << "  Layers: " << config.num_layers << std::endl;
        std::cout << "  Learning rate: " << config.learning_rate << std::endl;
    }
    
    bool load_market_data() {
        std::cout << "Loading market data from: " << config_.data_path << std::endl;
        
        if (!std::filesystem::exists(config_.data_path)) {
            std::cerr << "Error: Data file not found: " << config_.data_path << std::endl;
            return false;
        }
        
        std::ifstream file(config_.data_path);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open data file: " << config_.data_path << std::endl;
            return false;
        }
        
        std::string line;
        bool first_line = true;
        
        while (std::getline(file, line)) {
            if (first_line) {
                first_line = false;
                continue; // Skip header
            }
            
            std::istringstream ss(line);
            std::string token;
            std::vector<std::string> tokens;
            
            while (std::getline(ss, token, ',')) {
                tokens.push_back(token);
            }
            
            if (tokens.size() >= 5) {
                try {
                    Bar bar;
                    bar.open = std::stof(tokens[1]);   // Assuming: timestamp,open,high,low,close,volume
                    bar.high = std::stof(tokens[2]);
                    bar.low = std::stof(tokens[3]);
                    bar.close = std::stof(tokens[4]);
                    bar.volume = tokens.size() > 5 ? std::stof(tokens[5]) : 1000.0f;
                    bar.ts_utc_epoch = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count(); // Simplified
                    
                    market_data_.push_back(bar);
                } catch (const std::exception& e) {
                    // Skip invalid lines
                    continue;
                }
            }
        }
        
        std::cout << "Loaded " << market_data_.size() << " bars of market data" << std::endl;
        return !market_data_.empty();
    }
    
    void prepare_training_data() {
        std::cout << "Preparing training data..." << std::endl;
        
        // Initialize feature pipeline
        feature_pipeline_ = std::make_unique<FeaturePipeline>(feature_config_);
        
        // Generate features and labels
        for (size_t i = config_.sequence_length; i < market_data_.size() - 1; ++i) {
            // Generate features for each bar in the sequence
            std::vector<torch::Tensor> sequence_features;
            
            for (int j = i - config_.sequence_length; j < static_cast<int>(i); ++j) {
                std::vector<Bar> single_bar = {market_data_[j]};
                auto bar_features = feature_pipeline_->generate_features(single_bar);
                sequence_features.push_back(bar_features.squeeze(0));
            }
            
            // Stack features into sequence tensor [sequence_length, feature_dim]
            auto features = torch::stack(sequence_features);
            
            // Debug: Print feature dimensions for first sample
            if (training_samples_.empty()) {
                std::cout << "Feature tensor shape: " << features.sizes() << std::endl;
                std::cout << "Individual bar feature shape: " << sequence_features[0].sizes() << std::endl;
                
                // Update model configuration based on actual feature dimensions
                auto actual_feature_dim = sequence_features[0].size(0);
                if (actual_feature_dim != config_.feature_dim) {
                    std::cout << "Updating feature_dim from " << config_.feature_dim << " to " << actual_feature_dim << std::endl;
                    config_.feature_dim = actual_feature_dim;
                    model_config_.feature_dim = actual_feature_dim;
                }
            }
            
            // Calculate label (next bar return)
            float current_price = market_data_[i].close;
            float next_price = market_data_[i + 1].close;
            float label = (next_price - current_price) / current_price;
            
            training_samples_.emplace_back(features, label);
        }
        
        std::cout << "Generated " << training_samples_.size() << " training samples" << std::endl;
        
        // Split into training and validation
        size_t total_samples = training_samples_.size();
        size_t val_samples = (total_samples * config_.validation_split_percent) / 100;
        size_t train_samples = total_samples - val_samples;
        
        // Simple split - last 20% for validation
        train_samples_.assign(training_samples_.begin(), training_samples_.begin() + train_samples);
        val_samples_.assign(training_samples_.begin() + train_samples, training_samples_.end());
        
        std::cout << "Training samples: " << train_samples_.size() << std::endl;
        std::cout << "Validation samples: " << val_samples_.size() << std::endl;
    }
    
    bool train_model() {
        std::cout << "\nStarting model training..." << std::endl;
        std::cout << "=" << std::string(50, '=') << std::endl;
        
        // Create model
        model_ = std::make_shared<TransformerModel>(model_config_);
        std::cout << "Model created with " << model_->get_parameter_count() << " parameters" << std::endl;
        
        // Setup optimizer
        torch::optim::AdamW optimizer(model_->parameters(), torch::optim::AdamWOptions(config_.learning_rate));
        torch::nn::MSELoss criterion;
        
        // Training loop
        auto training_start = std::chrono::high_resolution_clock::now();
        
        for (int epoch = 0; epoch < config_.epochs; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            
            // Training phase
            model_->train();
            float train_loss = 0.0f;
            int train_batches = 0;
            
            // Shuffle training data
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(train_samples_.begin(), train_samples_.end(), g);
            
            // Process training batches
            for (size_t i = 0; i < train_samples_.size(); i += config_.batch_size) {
                size_t batch_end = std::min(i + config_.batch_size, train_samples_.size());
                
                // Create batch
                std::vector<torch::Tensor> batch_features;
                std::vector<float> batch_labels;
                
                for (size_t j = i; j < batch_end; ++j) {
                    batch_features.push_back(train_samples_[j].features);
                    batch_labels.push_back(train_samples_[j].label);
                }
                
                if (batch_features.empty()) continue;
                
                // Stack features into batch tensor
                auto features_batch = torch::stack(batch_features);
                auto labels_batch = torch::tensor(batch_labels);
                
                // Forward pass
                optimizer.zero_grad();
                auto predictions = model_->forward(features_batch);
                auto loss = criterion(predictions.squeeze(), labels_batch);
                
                // Backward pass
                loss.backward();
                
                // Gradient clipping
                torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);
                
                optimizer.step();
                
                train_loss += loss.item<float>();
                train_batches++;
                
                // Progress update within epoch
                if (train_batches % config_.progress_update_frequency == 0) {
                    float progress = (float)(i + batch_end - i) / train_samples_.size() * 100.0f;
                    std::cout << "\rEpoch " << (epoch + 1) << "/" << config_.epochs 
                              << " - Progress: " << std::fixed << std::setprecision(1) << progress << "% "
                              << "- Batch Loss: " << std::fixed << std::setprecision(6) << loss.item<float>()
                              << std::flush;
                }
            }
            
            float avg_train_loss = train_loss / train_batches;
            
            // Validation phase
            model_->eval();
            float val_loss = 0.0f;
            int val_batches = 0;
            
            torch::NoGradGuard no_grad;
            for (size_t i = 0; i < val_samples_.size(); i += config_.batch_size) {
                size_t batch_end = std::min(i + config_.batch_size, val_samples_.size());
                
                std::vector<torch::Tensor> batch_features;
                std::vector<float> batch_labels;
                
                for (size_t j = i; j < batch_end; ++j) {
                    batch_features.push_back(val_samples_[j].features);
                    batch_labels.push_back(val_samples_[j].label);
                }
                
                if (batch_features.empty()) continue;
                
                auto features_batch = torch::stack(batch_features);
                auto labels_batch = torch::tensor(batch_labels);
                
                auto predictions = model_->forward(features_batch);
                auto loss = criterion(predictions.squeeze(), labels_batch);
                
                val_loss += loss.item<float>();
                val_batches++;
            }
            
            float avg_val_loss = val_loss / val_batches;
            
            // Calculate epoch time
            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
            
            // Print epoch summary
            std::cout << "\rEpoch " << (epoch + 1) << "/" << config_.epochs 
                      << " - Train Loss: " << std::fixed << std::setprecision(6) << avg_train_loss
                      << " - Val Loss: " << std::fixed << std::setprecision(6) << avg_val_loss
                      << " - Time: " << epoch_duration.count() << "ms" << std::endl;
            
            // Save best model
            if (avg_val_loss < best_val_loss_) {
                best_val_loss_ = avg_val_loss;
                std::cout << "  → New best validation loss: " << std::fixed << std::setprecision(6) << best_val_loss_ << std::endl;
            }
        }
        
        auto training_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(training_end - training_start);
        
        std::cout << "\nTraining completed in " << total_duration.count() << " seconds" << std::endl;
        std::cout << "Best validation loss: " << std::fixed << std::setprecision(6) << best_val_loss_ << std::endl;
        
        return true;
    }
    
    bool save_model() {
        std::cout << "\nSaving trained model..." << std::endl;
        
        // Create output directory
        std::filesystem::create_directories(config_.output_dir);
        
        std::string model_path = config_.output_dir + "/model.pt";
        std::string metadata_path = config_.output_dir + "/model.meta.json";
        
        try {
            // Save model
            model_->save_model(model_path);
            
            // Create metadata
            std::ofstream meta_file(metadata_path);
            meta_file << "{\n";
            meta_file << "  \"feature_dim\": " << config_.feature_dim << ",\n";
            meta_file << "  \"sequence_length\": " << config_.sequence_length << ",\n";
            meta_file << "  \"d_model\": " << config_.d_model << ",\n";
            meta_file << "  \"num_heads\": " << config_.num_heads << ",\n";
            meta_file << "  \"num_layers\": " << config_.num_layers << ",\n";
            meta_file << "  \"model_type\": \"Transformer\",\n";
            meta_file << "  \"version\": \"v1\",\n";
            meta_file << "  \"best_val_loss\": " << best_val_loss_ << ",\n";
            meta_file << "  \"training_samples\": " << train_samples_.size() << ",\n";
            meta_file << "  \"validation_samples\": " << val_samples_.size() << "\n";
            meta_file << "}\n";
            meta_file.close();
            
            std::cout << "Model saved successfully:" << std::endl;
            std::cout << "  Model file: " << model_path << std::endl;
            std::cout << "  Metadata: " << metadata_path << std::endl;
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error saving model: " << e.what() << std::endl;
            return false;
        }
    }

private:
    TrainingConfig config_;
    TransformerConfig model_config_;
    TransformerConfig::Features feature_config_;
    
    std::vector<Bar> market_data_;
    std::vector<TrainingSample> training_samples_;
    std::vector<TrainingSample> train_samples_;
    std::vector<TrainingSample> val_samples_;
    
    std::shared_ptr<TransformerModel> model_;
    std::unique_ptr<FeaturePipeline> feature_pipeline_;
    
    float best_val_loss_ = std::numeric_limits<float>::max();
};

} // namespace sentio

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --data PATH          Path to market data CSV file (default: data/equities/QQQ_1min.csv)\n";
    std::cout << "  --output DIR         Output directory for trained model (default: artifacts/Transformer/v1)\n";
    std::cout << "  --epochs N           Number of training epochs (default: 20)\n";
    std::cout << "  --batch-size N       Batch size for training (default: 32)\n";
    std::cout << "  --sequence-length N  Sequence length for transformer (default: 64)\n";
    std::cout << "  --feature-dim N      Feature dimension (default: 128)\n";
    std::cout << "  --d-model N          Model dimension (default: 256)\n";
    std::cout << "  --num-heads N        Number of attention heads (default: 8)\n";
    std::cout << "  --num-layers N       Number of transformer layers (default: 6)\n";
    std::cout << "  --learning-rate F    Learning rate (default: 0.001)\n";
    std::cout << "  --help               Show this help message\n";
}

int main(int argc, char* argv[]) {
    sentio::TrainingConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--data" && i + 1 < argc) {
            config.data_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--sequence-length" && i + 1 < argc) {
            config.sequence_length = std::stoi(argv[++i]);
        } else if (arg == "--feature-dim" && i + 1 < argc) {
            config.feature_dim = std::stoi(argv[++i]);
        } else if (arg == "--d-model" && i + 1 < argc) {
            config.d_model = std::stoi(argv[++i]);
        } else if (arg == "--num-heads" && i + 1 < argc) {
            config.num_heads = std::stoi(argv[++i]);
        } else if (arg == "--num-layers" && i + 1 < argc) {
            config.num_layers = std::stoi(argv[++i]);
        } else if (arg == "--learning-rate" && i + 1 < argc) {
            config.learning_rate = std::stof(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    std::cout << "Sentio Transformer Strategy Training" << std::endl;
    std::cout << "====================================" << std::endl;
    
    try {
        sentio::TransformerTrainer trainer(config);
        
        // Load market data
        if (!trainer.load_market_data()) {
            std::cerr << "Failed to load market data" << std::endl;
            return 1;
        }
        
        // Prepare training data
        trainer.prepare_training_data();
        
        // Train model
        if (!trainer.train_model()) {
            std::cerr << "Training failed" << std::endl;
            return 1;
        }
        
        // Save model
        if (!trainer.save_model()) {
            std::cerr << "Failed to save model" << std::endl;
            return 1;
        }
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "TRAINING COMPLETE!" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Model saved to: " << config.output_dir << std::endl;
        std::cout << "You can now test the strategy with:" << std::endl;
        std::cout << "  ./sencli strattest transformer --mode historical --blocks 10" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

```

