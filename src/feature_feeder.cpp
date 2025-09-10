#include "sentio/feature_feeder.hpp"
// TFB strategy removed - focusing on TFA only
#include "sentio/strategy_tfa.hpp"
#include "sentio/strategy_transformer_ts.hpp"
#include "sentio/strategy_kochi_ppo.hpp"
#include "sentio/feature_builder.hpp"
#include "sentio/feature_engineering/kochi_features.hpp"
#include "sentio/feature_utils.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <algorithm>

namespace sentio {

// Static member definitions
std::unordered_map<std::string, FeatureFeeder::StrategyData> FeatureFeeder::strategy_data_;
std::mutex FeatureFeeder::data_mutex_;
std::unique_ptr<FeatureCache> FeatureFeeder::feature_cache_;
bool FeatureFeeder::use_cached_features_ = false;

bool FeatureFeeder::is_ml_strategy(const std::string& strategy_name) {
    return strategy_name == "TFA" || strategy_name == "tfa" ||
           strategy_name == "transformer" ||
           strategy_name == "hybrid_ppo" ||
           strategy_name == "kochi_ppo";
}

void FeatureFeeder::initialize_strategy(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    if (sentio::FeatureUtils::has_strategy_data(strategy_data_, strategy_name)) {
        return; // Already initialized
    }
    
    initialize_strategy_data(strategy_name);
}

void FeatureFeeder::initialize_strategy_data(const std::string& strategy_name) {
    StrategyData data;
    
    // Create technical indicator calculator
    data.calculator = std::make_unique<feature_engineering::TechnicalIndicatorCalculator>();
    
    // Create feature normalizer
    data.normalizer = std::make_unique<feature_engineering::FeatureNormalizer>(252); // 1 year window
    
    // Set default configuration
    data.config["normalization_method"] = "robust";
    data.config["outlier_threshold"] = "3.0";
    data.config["winsorize_percentile"] = "0.05";
    data.config["enable_caching"] = "true";
    
    data.initialized = true;
    data.last_update = std::chrono::steady_clock::now();
    
    strategy_data_[strategy_name] = std::move(data);
    
    std::cout << "Initialized FeatureFeeder for strategy: " << strategy_name << std::endl;
}

void FeatureFeeder::cleanup_strategy(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    strategy_data_.erase(strategy_name);
}

std::vector<double> FeatureFeeder::extract_features_from_bar(const Bar& bar, const std::string& strategy_name) {
    if (!is_ml_strategy(strategy_name)) {
        return {};
    }
    
    // Initialize if not already done
    if (!sentio::FeatureUtils::has_strategy_data(strategy_data_, strategy_name)) {
        initialize_strategy(strategy_name);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Kochi PPO uses its own feature set; need at least a small history.
        if (strategy_name == "kochi_ppo") {
            std::vector<Bar> hist = {bar};
            auto features = feature_engineering::calculate_kochi_features(hist, 0);
            if (features.empty()) return {};
            auto end_time_metrics = std::chrono::high_resolution_clock::now();
            auto extraction_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time_metrics - start_time);
            auto& data = get_strategy_data(strategy_name);
            update_metrics(data, features, extraction_time);
            return features;
        }

        // Get strategy data
        auto& data = get_strategy_data(strategy_name);
        
        // For single bar, we need at least some history
        // This is a limitation - we need multiple bars for most indicators
        // For now, return empty vector if we don't have enough history
        if (!data.calculator) {
            return {};
        }
        
        // Create a minimal bar history for calculation
        std::vector<Bar> bar_history = {bar};
        
        // Calculate features
        auto features = data.calculator->calculate_all_features(bar_history, 0);
        
        // Normalize features
        if (data.normalizer && !features.empty()) {
            features = data.normalizer->normalize_features(features);
        }
        
        // Validate features
        if (!validate_features(features, strategy_name)) {
            return {};
        }
        
        // Update metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto extraction_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        update_metrics(data, features, extraction_time);
        
        return features;
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting features for " << strategy_name << ": " << e.what() << std::endl;
        return {};
    }
}

std::vector<std::vector<double>> FeatureFeeder::extract_features_from_bars(const std::vector<Bar>& bars, const std::string& strategy_name) {
    if (!is_ml_strategy(strategy_name) || bars.empty()) {
        return {};
    }
    
    // Initialize if not already done
    if (!sentio::FeatureUtils::has_strategy_data(strategy_data_, strategy_name)) {
        initialize_strategy(strategy_name);
    }
    
    std::vector<std::vector<double>> all_features;
    all_features.reserve(bars.size());
    
    try {
        auto& data = get_strategy_data(strategy_name);
        
        if (!data.calculator) {
            return {};
        }
        
        // Extract features for each bar
        for (int i = 0; i < static_cast<int>(bars.size()); ++i) {
            auto features = data.calculator->calculate_all_features(bars, i);
            
            // Normalize features
            if (data.normalizer && !features.empty()) {
                features = data.normalizer->normalize_features(features);
            }
            
            all_features.push_back(features);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting features from bars for " << strategy_name << ": " << e.what() << std::endl;
        return {};
    }
    
    return all_features;
}

std::vector<double> FeatureFeeder::extract_features_from_bars_with_index(const std::vector<Bar>& bars, int current_index, const std::string& strategy_name) {
    
    if (!is_ml_strategy(strategy_name) || bars.empty() || current_index < 0 || current_index >= static_cast<int>(bars.size())) {
        return {};
    }
    
    // Initialize if not already done
    if (!sentio::FeatureUtils::has_strategy_data(strategy_data_, strategy_name)) {
        initialize_strategy(strategy_name);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Get strategy data
        auto& data = get_strategy_data(strategy_name);
        
        if (!data.calculator) {
            return {};
        }
        
        // Use cached features if available, otherwise calculate
        std::vector<double> features;
        if (use_cached_features_ && feature_cache_ && feature_cache_->has_features(current_index)) {
            features = feature_cache_->get_features(current_index);
            static int cache_calls = 0;
            cache_calls++;
            if (cache_calls <= 5) {
                std::cout << "[DEBUG] After cache get_features: " << features.size() << " features" << std::endl;
            }
        } else {
            // Calculate features using full bar history up to current_index
            features = data.calculator->calculate_all_features(bars, current_index);
        }
        
        // Normalize features (skip normalization for cached features as they're pre-processed)
        bool used_cache = (use_cached_features_ && feature_cache_ && feature_cache_->has_features(current_index));
        if (data.normalizer && !features.empty() && !used_cache) {
            size_t before_norm = features.size();
            features = data.normalizer->normalize_features(features);
            static int norm_calls = 0;
            norm_calls++;
            if (norm_calls <= 5) {
                std::cout << "[DEBUG] After normalize: " << before_norm << " -> " << features.size() << " features" << std::endl;
            }
        } else if (used_cache) {
            static int cache_skip_calls = 0;
            cache_skip_calls++;
            if (cache_skip_calls <= 5) {
                std::cout << "[DEBUG] Skipping normalization for cached features: " << features.size() << " features" << std::endl;
            }
        }
        
        // Validate features (bypass validation for cached features as they're pre-validated)
        if (used_cache) {
            static int cache_bypass_calls = 0;
            cache_bypass_calls++;
            if (cache_bypass_calls <= 5) {
                std::cout << "[DEBUG] Bypassing validation for cached features: " << features.size() << " features" << std::endl;
            }
        } else {
            bool valid = validate_features(features, strategy_name);
            static int val_calls = 0;
            val_calls++;
            if (val_calls <= 5) {
                std::cout << "[DEBUG] Validation result: " << (valid ? "PASS" : "FAIL") << " for " << features.size() << " features" << std::endl;
            }
            if (!valid) {
                return {};
            }
        }
        
        // Update metrics
        auto end_time_metrics = std::chrono::high_resolution_clock::now();
        auto extraction_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time_metrics - start_time);
        update_metrics(data, features, extraction_time);
        return features;
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting features for " << strategy_name << " at index " << current_index << ": " << e.what() << std::endl;
        return {};
    }
}

void FeatureFeeder::feed_features_to_strategy(BaseStrategy* strategy, const std::vector<Bar>& bars, int current_index, const std::string& strategy_name) {
    
    if (!is_ml_strategy(strategy_name) || !strategy) {
        return;
    }
    
    // Initialize if not already done
    if (!sentio::FeatureUtils::has_strategy_data(strategy_data_, strategy_name)) {
        initialize_strategy(strategy_name);
    }
    
    try {
        // Extract features using full bar history (required for technical indicators)
        auto features = extract_features_from_bars_with_index(bars, current_index, strategy_name);
        
        static int feature_extract_calls = 0;
        feature_extract_calls++;
        
        if (feature_extract_calls % 1000 == 0 || feature_extract_calls <= 10) {
            std::cout << "[DIAG] FeatureFeeder extract: call=" << feature_extract_calls 
                      << " features.size()=" << features.size() 
                      << " current_index=" << current_index << std::endl;
        }
        
        if (features.empty()) {
            if (feature_extract_calls % 1000 == 0 || feature_extract_calls <= 10) {
                std::cout << "[DIAG] FeatureFeeder: Features EMPTY at call=" << feature_extract_calls << std::endl;
            }
            return;
        }
        
        // Cast to specific strategy type and feed features
        static int strategy_check_calls = 0;
        strategy_check_calls++;
        
        if (strategy_check_calls % 1000 == 0 || strategy_check_calls <= 10) {
            std::cout << "[DIAG] FeatureFeeder strategy check: call=" << strategy_check_calls 
                      << " strategy_name='" << strategy_name << "'" << std::endl;
        }
        
        if (strategy_name == "TFA" || strategy_name == "tfa") {
            auto* tfa = dynamic_cast<TFAStrategy*>(strategy);
            if (tfa) {
                static int tfa_feed_calls = 0;
                tfa_feed_calls++;
                
                if (tfa_feed_calls % 1000 == 0 || tfa_feed_calls <= 10) {
                    std::cout << "[DIAG] FeatureFeeder TFA: call=" << tfa_feed_calls 
                              << " features.size()=" << features.size() << std::endl;
                }
                
                tfa->set_raw_features(features);
            } else {
                static int cast_fail = 0;
                cast_fail++;
                if (cast_fail <= 10) {
                    std::cout << "[DIAG] FeatureFeeder: TFA cast failed! call=" << cast_fail << std::endl;
                }
            }
        } else if (strategy_name == "transformer") {
            auto* tf = dynamic_cast<TransformerSignalStrategyTS*>(strategy);
            if (tf) {
                tf->set_raw_features(features);
            }
        } else if (strategy_name == "kochi_ppo") {
            auto* kp = dynamic_cast<KochiPPOStrategy*>(strategy);
            if (kp) {
                kp->set_raw_features(features);
            }
        }
        
        // Cache features
        cache_features(strategy_name, features);
        
    } catch (const std::exception& e) {
        std::cerr << "Error feeding features to strategy " << strategy_name << ": " << e.what() << std::endl;
    }
}

void FeatureFeeder::feed_features_batch(BaseStrategy* strategy, const std::vector<Bar>& bars, const std::string& strategy_name) {
    if (!is_ml_strategy(strategy_name) || !strategy || bars.empty()) {
        return;
    }
    
    // Initialize if not already done
    if (!sentio::FeatureUtils::has_strategy_data(strategy_data_, strategy_name)) {
        initialize_strategy(strategy_name);
    }
    
    try {
        // Extract features for all bars
        auto all_features = extract_features_from_bars(bars, strategy_name);
        
        if (all_features.empty()) {
            return;
        }
        
        // Feed features to strategy
        for (size_t i = 0; i < all_features.size(); ++i) {
            if (!all_features[i].empty()) {
                feed_features_to_strategy(strategy, bars, i, strategy_name);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error feeding features batch to strategy " << strategy_name << ": " << e.what() << std::endl;
    }
}

std::vector<double> FeatureFeeder::get_cached_features(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        return it->second.cached_features;
    }
    
    return {};
}

void FeatureFeeder::cache_features(const std::string& strategy_name, const std::vector<double>& features) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        it->second.cached_features = features;
        it->second.last_update = std::chrono::steady_clock::now();
    }
}

void FeatureFeeder::invalidate_cache(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        it->second.cached_features.clear();
    }
}

FeatureMetrics FeatureFeeder::get_metrics(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        return it->second.metrics;
    }
    
    return FeatureMetrics{};
}

FeatureHealthReport FeatureFeeder::get_health_report(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        return calculate_health_report(it->second, it->second.cached_features);
    }
    
    return FeatureHealthReport{};
}

void FeatureFeeder::reset_metrics(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        it->second.metrics = FeatureMetrics{};
    }
}

bool FeatureFeeder::validate_features(const std::vector<double>& features, const std::string& strategy_name) {
    if (features.empty()) {
        return false;
    }
    
    // Check if all features are finite
    int non_finite_count = 0;
    for (size_t i = 0; i < features.size(); ++i) {
        if (!std::isfinite(features[i])) {
            non_finite_count++;
        }
    }
    
    // Feature validation completed
    
    if (non_finite_count > 0) {
        return false;
    }
    
    // Check feature count; for Kochi we compare against its own names
    auto expected_names = (strategy_name == "kochi_ppo")
        ? feature_engineering::kochi_feature_names()
        : get_feature_names(strategy_name);
    if (features.size() != expected_names.size()) {
        return false;
    }
    
    return true;
}

std::vector<std::string> FeatureFeeder::get_feature_names(const std::string& strategy_name) {
    return get_strategy_feature_names(strategy_name);
}

std::vector<std::string> FeatureFeeder::get_strategy_feature_names(const std::string& strategy_name) {
    if (strategy_name == "TFA" || strategy_name == "tfa" || 
        strategy_name == "transformer") {
        // Return the exact 55 features that TechnicalIndicatorCalculator provides
        return {
            // Price features (15)
            "ret_1m", "ret_5m", "ret_15m", "ret_30m", "ret_1h",
            "momentum_5", "momentum_10", "momentum_20",
            "volatility_10", "volatility_20", "volatility_30",
            "atr_14", "atr_21", "parkinson_vol", "garman_klass_vol",
            
            // Technical features (27) - Note: Actually 27, not 25 as the comment in calculator says
            "rsi_14", "rsi_21", "rsi_30",
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
            "bb_upper_20", "bb_middle_20", "bb_lower_20",
            "bb_upper_50", "bb_middle_50", "bb_lower_50",
            "macd_line", "macd_signal", "macd_histogram",
            "stoch_k", "stoch_d", "williams_r", "cci_20", "adx_14",
            
            // Volume features (8)
            "volume_sma_10", "volume_sma_20", "volume_sma_50",
            "volume_roc", "obv", "vpt", "ad_line", "mfi_14",
            
            // Microstructure features (5)
            "spread_bp", "price_impact", "order_flow_imbalance", "market_depth", "bid_ask_ratio"
        };
    }
    if (strategy_name == "kochi_ppo") {
        return feature_engineering::kochi_feature_names();
    }
    
    return {};
}

void FeatureFeeder::set_feature_config(const std::string& strategy_name, const std::string& config_key, const std::string& config_value) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        it->second.config[config_key] = config_value;
    }
}

std::string FeatureFeeder::get_feature_config(const std::string& strategy_name, const std::string& config_key) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        auto config_it = it->second.config.find(config_key);
        if (config_it != it->second.config.end()) {
            return config_it->second;
        }
    }
    
    return "";
}

std::vector<double> FeatureFeeder::get_feature_correlation([[maybe_unused]] const std::string& strategy_name) {
    // Placeholder implementation
    // In practice, this would calculate correlation between features
    return {};
}

std::vector<double> FeatureFeeder::get_feature_importance([[maybe_unused]] const std::string& strategy_name) {
    // Placeholder implementation
    // In practice, this would calculate feature importance scores
    return {};
}

void FeatureFeeder::log_feature_performance(const std::string& strategy_name) {
    auto metrics = get_metrics(strategy_name);
    auto health = get_health_report(strategy_name);
    
    std::cout << "FeatureFeeder Performance for " << strategy_name << ":" << std::endl;
    std::cout << "  Extraction time: " << metrics.extraction_time.count() << " microseconds" << std::endl;
    std::cout << "  Features extracted: " << metrics.features_extracted << std::endl;
    std::cout << "  Features valid: " << metrics.features_valid << std::endl;
    std::cout << "  Features invalid: " << metrics.features_invalid << std::endl;
    std::cout << "  Extraction rate: " << metrics.extraction_rate << " features/sec" << std::endl;
    std::cout << "  Health status: " << (health.is_healthy ? "HEALTHY" : "UNHEALTHY") << std::endl;
    std::cout << "  Overall health score: " << health.overall_health_score << std::endl;
}

FeatureFeeder::StrategyData& FeatureFeeder::get_strategy_data(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it == strategy_data_.end()) {
        initialize_strategy_data(strategy_name);
        it = strategy_data_.find(strategy_name);
    }
    
    return it->second;
}

void FeatureFeeder::update_metrics(StrategyData& data, const std::vector<double>& features, std::chrono::microseconds extraction_time) {
    data.metrics.extraction_time = extraction_time;
    data.metrics.features_extracted = features.size();
    data.metrics.features_valid = features.size(); // Assuming all features are valid at this point
    data.metrics.features_invalid = 0;
    
    if (extraction_time.count() > 0) {
        data.metrics.extraction_rate = static_cast<double>(features.size()) / (extraction_time.count() / 1000000.0);
    }
    
    data.metrics.last_update = std::chrono::steady_clock::now();
}

FeatureHealthReport FeatureFeeder::calculate_health_report([[maybe_unused]] const StrategyData& data, const std::vector<double>& features) {
    FeatureHealthReport report;
    
    if (features.empty()) {
        report.is_healthy = false;
        report.health_summary = "No features available";
        return report;
    }
    
    report.feature_health.resize(features.size(), true);
    report.feature_quality_scores.resize(features.size(), 1.0);
    
    // Check for NaN or infinite values
    for (size_t i = 0; i < features.size(); ++i) {
        if (!std::isfinite(features[i])) {
            report.feature_health[i] = false;
            report.feature_quality_scores[i] = 0.0;
        }
    }
    
    // Calculate overall health
    size_t healthy_features = std::count(report.feature_health.begin(), report.feature_health.end(), true);
    report.overall_health_score = static_cast<double>(healthy_features) / features.size();
    report.is_healthy = report.overall_health_score > 0.8; // 80% threshold
    
    if (report.is_healthy) {
        report.health_summary = "All features are healthy";
    } else {
        report.health_summary = "Some features are unhealthy";
    }
    
    return report;
}

// Cached features implementation
bool FeatureFeeder::load_feature_cache(const std::string& feature_file_path) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    feature_cache_ = std::make_unique<FeatureCache>();
    if (feature_cache_->load_from_csv(feature_file_path)) {
        std::cout << "FeatureFeeder: Successfully loaded feature cache from " << feature_file_path << std::endl;
        return true;
    } else {
        feature_cache_.reset();
        std::cerr << "FeatureFeeder: Failed to load feature cache from " << feature_file_path << std::endl;
        return false;
    }
}

bool FeatureFeeder::use_cached_features(bool enable) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    use_cached_features_ = enable;
    std::cout << "FeatureFeeder: Cached features " << (enable ? "ENABLED" : "DISABLED") << std::endl;
    return true;
}

bool FeatureFeeder::has_cached_features() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return feature_cache_ != nullptr;
}

} // namespace sentio