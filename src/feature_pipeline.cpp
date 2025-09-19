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

TransformerFeatureMatrix FeaturePipeline::generate_enhanced_features(const std::vector<Bar>& bars) {
    auto start_time = std::chrono::high_resolution_clock::now();
    if (bars.empty()) {
        return torch::zeros({1, ENHANCED_TOTAL_FEATURES});
    }
    // Original 128
    auto base_tensor = generate_features(bars);
    auto base_vec = std::vector<float>(base_tensor.data_ptr<float>(), base_tensor.data_ptr<float>() + TOTAL_FEATURES);

    // Additional feature blocks
    auto momentum = generate_momentum_persistence_features(bars);
    auto volreg   = generate_volatility_regime_features(bars);
    auto micro    = generate_microstructure_features(bars);
    auto options  = generate_options_features(bars);

    std::vector<float> all_features;
    all_features.reserve(ENHANCED_TOTAL_FEATURES);
    all_features.insert(all_features.end(), base_vec.begin(), base_vec.end());
    all_features.insert(all_features.end(), momentum.begin(), momentum.end());
    all_features.insert(all_features.end(), volreg.begin(), volreg.end());
    all_features.insert(all_features.end(), micro.begin(), micro.end());
    all_features.insert(all_features.end(), options.begin(), options.end());

    if (all_features.size() < ENHANCED_TOTAL_FEATURES) {
        all_features.resize(ENHANCED_TOTAL_FEATURES, 0.0f);
    } else if (all_features.size() > ENHANCED_TOTAL_FEATURES) {
        all_features.resize(ENHANCED_TOTAL_FEATURES);
    }

    // For enhanced features we reuse the normalization path but stats are sized to TOTAL_FEATURES.
    // We therefore normalize only the first TOTAL_FEATURES with running stats; rest are left as-is.
    for (size_t i = 0; i < std::min(all_features.size(), feature_stats_.size()); ++i) {
        if (!std::isnan(all_features[i]) && !std::isinf(all_features[i])) {
            feature_stats_[i].update(all_features[i]);
            float z = (all_features[i] - feature_stats_[i].mean()) / (feature_stats_[i].std() + 1e-8f);
            all_features[i] = z;
        } else {
            all_features[i] = 0.0f;
        }
    }

    auto tensor = torch::tensor(all_features).unsqueeze(0);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    if (duration.count() > MAX_GENERATION_TIME_US) {
        std::cerr << "Enhanced feature generation exceeded latency: "
                  << duration.count() << "us > " << MAX_GENERATION_TIME_US << "us" << std::endl;
    }
    return tensor;
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

// ===================== Enhanced feature blocks =====================

std::vector<float> FeaturePipeline::generate_momentum_persistence_features(const std::vector<Bar>& bars) {
    std::vector<float> features;
    features.reserve(15);
    if (bars.size() < 50) { features.resize(15, 0.0f); return features; }

    std::vector<int> periods = {5, 10, 20, 50};
    int consistent_momentum = 0;
    for (int period : periods) {
        if (bars.size() > static_cast<size_t>(period)) {
            float momentum = (bars.back().close / bars[bars.size()-1-period].close) - 1.0f;
            features.push_back(momentum);
            if (momentum > 0.02f) consistent_momentum++; else if (momentum < -0.02f) consistent_momentum--;
        } else { features.push_back(0.0f); }
    }
    features.push_back(static_cast<float>(consistent_momentum) / 4.0f);

    // Price acceleration
    if (bars.size() >= 3) {
        float accel = (bars.back().close - bars[bars.size()-2].close) - (bars[bars.size()-2].close - bars[bars.size()-3].close);
        features.push_back(accel / std::max(1e-8f, static_cast<float>(bars.back().close)));
    } else { features.push_back(0.0f); }

    // Trend strength (R^2)
    if (bars.size() >= 20) { features.push_back(calculate_trend_strength(bars, 20)); } else { features.push_back(0.0f); }

    // Volume-momentum divergence
    if (bars.size() >= 10) {
        float price_mom = (bars.back().close / bars[bars.size()-10].close) - 1.0f;
        float vol_tr = calculate_volume_trend(bars, 10);
        features.push_back(price_mom * vol_tr);
    } else { features.push_back(0.0f); }

    // Momentum decay
    std::vector<int> decay_periods = {2, 5, 10};
    for (int period : decay_periods) {
        if (bars.size() > static_cast<size_t>(period*2)) {
            float recent_m = (bars.back().close / bars[bars.size()-1-period].close) - 1.0f;
            float older_m  = (bars[bars.size()-1-period].close / bars[bars.size()-1-period*2].close) - 1.0f;
            features.push_back(recent_m - older_m);
        } else { features.push_back(0.0f); }
    }

    // Resistance breakthrough
    if (bars.size() >= 50) { features.push_back(calculate_resistance_breakthrough(bars)); } else { features.push_back(0.0f); }

    features.resize(15, 0.0f);
    return features;
}

std::vector<float> FeaturePipeline::generate_volatility_regime_features(const std::vector<Bar>& bars) {
    std::vector<float> features; features.reserve(12);
    if (bars.size() < 30) { features.resize(12, 0.0f); return features; }

    std::vector<int> vol_periods = {5, 10, 20, 50};
    for (int p : vol_periods) { features.push_back(calculate_realized_volatility(bars, p)); }

    float current_vol = calculate_realized_volatility(bars, 10);
    float historical_vol = calculate_realized_volatility(bars, 50);
    float vol_regime = current_vol / (historical_vol + 1e-8f);
    features.push_back(vol_regime);

    if (bars.size() >= 20) {
        float up=0.0f, dn=0.0f; int up_c=0, dn_c=0;
        for (size_t i = bars.size()-20; i < bars.size()-1; ++i) {
            float r = std::log(bars[i+1].close / bars[i].close);
            if (r>0){ up += r*r; up_c++; } else { dn += r*r; dn_c++; }
        }
        float up_vol = up_c>0? std::sqrt(up/up_c) : 0.0f;
        float dn_vol = dn_c>0? std::sqrt(dn/dn_c) : 0.0f;
        features.push_back(up_vol);
        features.push_back(dn_vol);
        features.push_back((dn_vol - up_vol) / (dn_vol + up_vol + 1e-8f));
    } else { features.insert(features.end(), {0.0f,0.0f,0.0f}); }

    std::vector<int> persistence_periods = {3,7,15};
    for (int p : persistence_periods) { features.push_back(calculate_volatility_persistence(bars, p)); }
    features.push_back(calculate_volatility_clustering(bars, 20));
    features.resize(12, 0.0f);
    return features;
}

std::vector<float> FeaturePipeline::generate_microstructure_features(const std::vector<Bar>& bars) {
    std::vector<float> features; features.reserve(10);
    if (bars.empty()) { features.resize(10, 0.0f); return features; }

    const Bar& cur = bars.back();
    features.push_back((cur.high - cur.low) / std::max(1e-8f, static_cast<float>(cur.close)));

    if (bars.size() >= 2) {
        float price_change = (cur.close - bars[bars.size()-2].close) / std::max(1e-8f, static_cast<float>(bars[bars.size()-2].close));
        float vol_factor = cur.volume / (calculate_average_volume(bars, 10) + 1e-8f);
        features.push_back(price_change / (vol_factor + 1e-8f));
    } else { features.push_back(0.0f); }

    float pr = cur.high - cur.low; features.push_back(pr>0? cur.volume / pr : 0.0f);
    float pos = (cur.close - cur.low) / (pr + 1e-8f); features.push_back(pos);
    if (bars.size() >= 2) { features.push_back(cur.close > bars[bars.size()-2].close ? 1.0f : -1.0f); } else { features.push_back(0.0f); }
    float high_vol_proxy = (cur.high - cur.close) / (pr + 1e-8f);
    float low_vol_proxy  = (cur.close - cur.low) / (pr + 1e-8f);
    features.push_back(high_vol_proxy); features.push_back(low_vol_proxy);
    if (bars.size() >= 10) { features.push_back(calculate_mean_reversion_speed(bars, 10)); } else { features.push_back(0.0f); }
    if (bars.size() >= 5) {
        float expected = calculate_average_volume(bars, 5);
        float surprise = (cur.volume - expected) / (expected + 1e-8f);
        features.push_back(surprise);
    } else { features.push_back(0.0f); }
    if (bars.size() >= 2 && cur.volume > 0) {
        float lambda_proxy = std::fabs(cur.close - bars[bars.size()-2].close) / (cur.volume + 1e-8f);
        features.push_back(lambda_proxy);
    } else { features.push_back(0.0f); }

    features.resize(10, 0.0f);
    return features;
}

std::vector<float> FeaturePipeline::generate_options_features(const std::vector<Bar>& bars) {
    std::vector<float> features; features.reserve(8);
    if (bars.size() >= 22) {
        float realized_vol = calculate_realized_volatility(bars, 22) * std::sqrt(252.0f);
        float vix_proxy = realized_vol * 100.0f;
        features.push_back(vix_proxy);
        float short_vol = calculate_realized_volatility(bars, 5) * std::sqrt(252.0f) * 100.0f;
        float long_vol  = calculate_realized_volatility(bars, 44) * std::sqrt(252.0f) * 100.0f;
        float term = (long_vol - short_vol) / (short_vol + 1e-8f);
        features.push_back(term);
    } else { features.push_back(20.0f); features.push_back(0.0f); }

    if (bars.size() >= 10) {
        float down_v=0.0f, up_v=0.0f;
        for (size_t i = bars.size()-10; i < bars.size()-1; ++i) {
            if (bars[i+1].close < bars[i].close) down_v += bars[i+1].volume; else up_v += bars[i+1].volume;
        }
        features.push_back(down_v / (up_v + 1e-8f));
    } else { features.push_back(1.0f); }

    if (bars.size() >= 30) {
        std::vector<float> rets; rets.reserve(29);
        for (size_t i = bars.size()-30; i < bars.size()-1; ++i) { rets.push_back(std::log(bars[i+1].close / bars[i].close)); }
        features.push_back(calculate_skewness(rets));
        features.push_back(calculate_kurtosis(rets));
    } else { features.push_back(0.0f); features.push_back(3.0f); }

    if (bars.size() >= 5) { features.push_back(calculate_gamma_exposure_proxy(bars)); } else { features.push_back(0.0f); }
    if (bars.size() >= 20) { features.push_back(detect_unusual_volume_patterns(bars, 20)); } else { features.push_back(0.0f); }
    features.push_back(calculate_fear_greed_proxy(bars));
    features.resize(8, 0.0f);
    return features;
}

// ===================== Enhanced feature utilities =====================

float FeaturePipeline::calculate_realized_volatility(const std::vector<Bar>& bars, int period) {
    if (bars.size() < static_cast<size_t>(period+1)) return 0.0f;
    float sum=0.0f; for (size_t i = bars.size()-period-1; i < bars.size()-1; ++i) {
        float r = std::log(bars[i+1].close / bars[i].close); sum += r*r; }
    return std::sqrt(sum / period);
}

float FeaturePipeline::calculate_average_volume(const std::vector<Bar>& bars, int period) {
    if (bars.size() < static_cast<size_t>(period)) return bars.back().volume;
    double s=0.0; for (size_t i = bars.size()-period; i < bars.size(); ++i) s += bars[i].volume; return static_cast<float>(s/period);
}

float FeaturePipeline::calculate_skewness(const std::vector<float>& v) {
    if (v.size() < 3) return 0.0f;
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double m2=0.0, m3=0.0; for (double x : v){ double d=x-mean; m2+=d*d; m3+=d*d*d; }
    m2/=v.size(); m3/=v.size(); double s = std::sqrt(m2); return s>0 ? static_cast<float>(m3/(s*s*s)) : 0.0f;
}

float FeaturePipeline::calculate_kurtosis(const std::vector<float>& v) {
    if (v.size() < 4) return 3.0f;
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double m2=0.0, m4=0.0; for (double x : v){ double d=x-mean; m2+=d*d; m4+=d*d*d*d; }
    m2/=v.size(); m4/=v.size(); return m2>0 ? static_cast<float>(m4/(m2*m2)) : 3.0f;
}

float FeaturePipeline::calculate_trend_strength(const std::vector<Bar>& bars, int lookback) {
    if (bars.size() < static_cast<size_t>(lookback)) return 0.0f;
    // Simple linear regression R^2 over closes
    int n = lookback; double sumx=0,sumy=0,sumxy=0,sumxx=0;
    for (int i=0;i<n;++i){ double x=i; double y=bars[bars.size()-lookback+i].close; sumx+=x; sumy+=y; sumxy+=x*y; sumxx+=x*x; }
    double xbar=sumx/n, ybar=sumy/n; double ssxy=sumxy - n*xbar*ybar; double ssxx=sumxx - n*xbar*xbar;
    if (ssxx==0) return 0.0f; double beta=ssxy/ssxx; double sst=0, sse=0;
    for (int i=0;i<n;++i){ double x=i; double y=bars[bars.size()-lookback+i].close; double yhat=ybar+beta*(x-xbar); sst+=(y-ybar)*(y-ybar); sse+=(y-yhat)*(y-yhat);} 
    return sst>0? static_cast<float>(1.0 - sse/sst) : 0.0f;
}

float FeaturePipeline::calculate_volume_trend(const std::vector<Bar>& bars, int period) {
    if (bars.size() < static_cast<size_t>(period*2)) return 0.0f;
    double recent=0, older=0; for (int i=0;i<period;++i){ recent += bars[bars.size()-1-i].volume; older += bars[bars.size()-1-period-i].volume; }
    recent/=period; older/=period; return older>0? static_cast<float>(recent/older - 1.0) : 0.0f;
}

float FeaturePipeline::calculate_volatility_persistence(const std::vector<Bar>& bars, int period) {
    if (bars.size() < static_cast<size_t>(period+1)) return 0.0f;
    std::vector<float> r; r.reserve(period);
    for (size_t i = bars.size()-period-1; i < bars.size()-1; ++i) r.push_back(std::fabs(std::log(bars[i+1].close / bars[i].close)));
    if (r.size()<2) return 0.0f; double mean = std::accumulate(r.begin(), r.end(), 0.0) / r.size(); double acf=0.0, denom=0.0;
    for (size_t i=1;i<r.size();++i){ acf += (r[i]-mean)*(r[i-1]-mean); }
    for (size_t i=0;i<r.size();++i){ denom += (r[i]-mean)*(r[i]-mean); }
    return denom>0? static_cast<float>(acf/denom) : 0.0f;
}

float FeaturePipeline::calculate_volatility_clustering(const std::vector<Bar>& bars, int period) {
    if (bars.size() < static_cast<size_t>(period+1)) return 0.0f;
    // Proxy: mean absolute return vs its std over window
    double sum=0.0; std::vector<float> absr; absr.reserve(period);
    for (size_t i = bars.size()-period-1; i < bars.size()-1; ++i){ float ar = std::fabs(std::log(bars[i+1].close / bars[i].close)); absr.push_back(ar); sum+=ar; }
    double mean = sum/absr.size(); double var=0.0; for (float x:absr){ var += (x-mean)*(x-mean);} var/=std::max<size_t>(1,absr.size()-1);
    return mean>0? static_cast<float>(std::sqrt(var)/mean) : 0.0f;
}

float FeaturePipeline::calculate_mean_reversion_speed(const std::vector<Bar>& bars, int period) {
    if (bars.size() < static_cast<size_t>(period+1)) return 0.0f;
    // OU speed proxy using lag-1 autocorrelation of returns
    double sum=0.0; std::vector<float> r; r.reserve(period);
    for (size_t i = bars.size()-period-1; i < bars.size()-1; ++i) r.push_back(std::log(bars[i+1].close / bars[i].close));
    double mean = std::accumulate(r.begin(), r.end(), 0.0) / r.size(); double num=0.0, den=0.0;
    for (size_t i=1;i<r.size();++i){ num += (r[i]-mean)*(r[i-1]-mean); }
    for (size_t i=0;i<r.size();++i){ den += (r[i]-mean)*(r[i]-mean); }
    double rho = den>0? num/den : 0.0; return static_cast<float>(1.0 - rho);
}

float FeaturePipeline::calculate_gamma_exposure_proxy(const std::vector<Bar>& bars) {
    if (bars.size() < 5) return 0.0f; // simple curvature proxy
    // Use 2nd derivative magnitude of price over last 5
    double curv=0.0; for (size_t i = bars.size()-5; i < bars.size()-2; ++i){ double a = bars[i+2].close - 2*bars[i+1].close + bars[i].close; curv += std::fabs(a); }
    return static_cast<float>(curv / 3.0 / std::max(1e-6f, static_cast<float>(bars.back().close)));
}

float FeaturePipeline::detect_unusual_volume_patterns(const std::vector<Bar>& bars, int period) {
    if (bars.size() < static_cast<size_t>(period)) return 0.0f;
    double mean = 0.0; for (size_t i = bars.size()-period; i < bars.size(); ++i) mean += bars[i].volume; mean/=period;
    double var=0.0; for (size_t i = bars.size()-period; i < bars.size(); ++i){ double d=bars[i].volume-mean; var+=d*d; }
    var/=std::max(1, period-1); double stdv=std::sqrt(var);
    return stdv>0? static_cast<float>((bars.back().volume - mean)/(stdv + 1e-8)) : 0.0f;
}

float FeaturePipeline::calculate_fear_greed_proxy(const std::vector<Bar>& bars) {
    if (bars.size() < 10) return 0.5f;
    // Combine normalized return momentum and volume surprise
    float mom = (bars.back().close / bars[bars.size()-10].close) - 1.0f;
    float vol_surprise = detect_unusual_volume_patterns(bars, 10);
    float scaled = 0.5f + std::tanh(5.0f * (mom + 0.1f*vol_surprise)) * 0.5f;
    return std::clamp(scaled, 0.0f, 1.0f);
}

float FeaturePipeline::calculate_resistance_breakthrough(const std::vector<Bar>& bars) {
    if (bars.size() < 50) return 0.0f;
    float max_price = bars[bars.size()-50].close;
    for (size_t i = bars.size()-49; i < bars.size()-1; ++i) max_price = std::max(max_price, static_cast<float>(bars[i].close));
    float diff = static_cast<float>(bars.back().close) - max_price; return diff / std::max(1e-8f, max_price);
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
