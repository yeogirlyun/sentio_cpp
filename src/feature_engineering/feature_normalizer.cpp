#include "sentio/feature_engineering/feature_normalizer.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

namespace sentio {
namespace feature_engineering {

FeatureNormalizer::FeatureNormalizer(size_t window_size) 
    : window_size_(window_size) {
    // Initialize with empty stats - updated to match actual feature count
    stats_.resize(55); // Updated from 52 to 55 to match actual feature count
    feature_history_.resize(55);
}

std::vector<double> FeatureNormalizer::normalize_features(const std::vector<double>& features) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (features.empty()) {
        return features;
    }
    
    // Update statistics with new features
    update_stats(features);
    
    // Apply robust normalization
    return robust_normalize(features);
}

std::vector<double> FeatureNormalizer::denormalize_features(const std::vector<double>& normalized_features) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (normalized_features.empty() || stats_.empty()) {
        return normalized_features;
    }
    
    std::vector<double> denormalized;
    denormalized.reserve(normalized_features.size());
    
    for (size_t i = 0; i < normalized_features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && stat.std > 0) {
            double denorm = (normalized_features[i] * stat.std) + stat.mean;
            denormalized.push_back(denorm);
        } else {
            denormalized.push_back(normalized_features[i]);
        }
    }
    
    return denormalized;
}

void FeatureNormalizer::update_stats(const std::vector<double>& features) {
    for (size_t i = 0; i < features.size() && i < feature_history_.size(); ++i) {
        update_feature_stats(i, features[i]);
    }
}

void FeatureNormalizer::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    for (auto& stat : stats_) {
        stat = NormalizationStats{};
    }
    
    for (auto& history : feature_history_) {
        history.clear();
    }
}

std::vector<double> FeatureNormalizer::z_score_normalize(const std::vector<double>& features) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (features.empty() || stats_.empty()) {
        return features;
    }
    
    std::vector<double> normalized;
    normalized.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && stat.std > 0) {
            double normalized_value = (features[i] - stat.mean) / stat.std;
            normalized.push_back(normalized_value);
        } else {
            normalized.push_back(features[i]);
        }
    }
    
    return normalized;
}

std::vector<double> FeatureNormalizer::min_max_normalize(const std::vector<double>& features) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (features.empty() || stats_.empty()) {
        return features;
    }
    
    std::vector<double> normalized;
    normalized.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && (stat.max - stat.min) > 0) {
            double normalized_value = (features[i] - stat.min) / (stat.max - stat.min);
            normalized.push_back(normalized_value);
        } else {
            normalized.push_back(features[i]);
        }
    }
    
    return normalized;
}

std::vector<double> FeatureNormalizer::robust_normalize(const std::vector<double>& features) {
    // Note: Mutex is already held by caller (normalize_features)
    
    if (features.empty() || stats_.empty()) {
        return features;
    }
    
    std::vector<double> normalized;
    normalized.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && stat.std > 0) {
            // Use robust statistics (median and MAD)
            double robust_mean = calculate_robust_mean(feature_history_[i]);
            double robust_std = calculate_robust_std(feature_history_[i], robust_mean);
            
            if (robust_std > 0) {
                double normalized_value = (features[i] - robust_mean) / robust_std;
                normalized.push_back(normalized_value);
            } else {
                normalized.push_back(features[i]);
            }
        } else {
            normalized.push_back(features[i]);
        }
    }
    
    return normalized;
}

std::vector<double> FeatureNormalizer::clip_outliers(const std::vector<double>& features, double threshold) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (features.empty() || stats_.empty()) {
        return features;
    }
    
    std::vector<double> clipped;
    clipped.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && stat.std > 0) {
            double upper_bound = stat.mean + (threshold * stat.std);
            double lower_bound = stat.mean - (threshold * stat.std);
            
            double clipped_value = std::clamp(features[i], lower_bound, upper_bound);
            clipped.push_back(clipped_value);
        } else {
            clipped.push_back(features[i]);
        }
    }
    
    return clipped;
}

std::vector<double> FeatureNormalizer::winsorize(const std::vector<double>& features, double percentile) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (features.empty() || stats_.empty()) {
        return features;
    }
    
    std::vector<double> winsorized;
    winsorized.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < feature_history_.size(); ++i) {
        if (feature_history_[i].size() < 2) {
            winsorized.push_back(features[i]);
            continue;
        }
        
        // Calculate percentiles from history
        auto sorted_values = feature_history_[i];
        sort_values(sorted_values);
        
        double lower_percentile = calculate_percentile(sorted_values, percentile);
        double upper_percentile = calculate_percentile(sorted_values, 1.0 - percentile);
        
        double winsorized_value = std::clamp(features[i], lower_percentile, upper_percentile);
        winsorized.push_back(winsorized_value);
    }
    
    return winsorized;
}

bool FeatureNormalizer::is_normalized(const std::vector<double>& features) const {
    if (features.empty()) {
        return true;
    }
    
    // Check if features are in reasonable normalized range [-5, 5]
    for (double feature : features) {
        if (std::abs(feature) > 5.0) {
            return false;
        }
    }
    
    return true;
}

std::vector<bool> FeatureNormalizer::get_outlier_mask(const std::vector<double>& features, double threshold) const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    std::vector<bool> outlier_mask;
    outlier_mask.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && stat.std > 0) {
            double upper_bound = stat.mean + (threshold * stat.std);
            double lower_bound = stat.mean - (threshold * stat.std);
            
            bool is_outlier = (features[i] < lower_bound) || (features[i] > upper_bound);
            outlier_mask.push_back(is_outlier);
        } else {
            outlier_mask.push_back(false);
        }
    }
    
    return outlier_mask;
}

NormalizationStats FeatureNormalizer::get_stats(size_t feature_index) const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (feature_index < stats_.size()) {
        return stats_[feature_index];
    }
    
    return NormalizationStats{};
}

std::vector<NormalizationStats> FeatureNormalizer::get_all_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void FeatureNormalizer::set_window_size(size_t window_size) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    window_size_ = window_size;
    
    // Trim existing histories to new window size
    for (auto& history : feature_history_) {
        while (history.size() > window_size_) {
            history.pop_front();
        }
    }
}

void FeatureNormalizer::set_outlier_threshold(double threshold) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    outlier_threshold_ = threshold;
}

void FeatureNormalizer::set_winsorize_percentile(double percentile) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    winsorize_percentile_ = percentile;
}

void FeatureNormalizer::update_feature_stats(size_t feature_index, double value) {
    if (feature_index >= feature_history_.size()) {
        return;
    }
    
    auto& history = feature_history_[feature_index];
    auto& stat = stats_[feature_index];
    
    // Add new value to history
    history.push_back(value);
    
    // Trim history to window size
    while (history.size() > window_size_) {
        history.pop_front();
    }
    
    // Update statistics
    if (history.size() == 1) {
        stat.mean = value;
        stat.std = 0.0;
        stat.min = value;
        stat.max = value;
        stat.count = 1;
    } else {
        // Calculate running statistics
        double sum = std::accumulate(history.begin(), history.end(), 0.0);
        stat.mean = sum / history.size();
        
        double variance = 0.0;
        for (double v : history) {
            double diff = v - stat.mean;
            variance += diff * diff;
        }
        stat.std = std::sqrt(variance / (history.size() - 1));
        
        stat.min = *std::min_element(history.begin(), history.end());
        stat.max = *std::max_element(history.begin(), history.end());
        stat.count = history.size();
    }
}

double FeatureNormalizer::calculate_robust_mean(const std::deque<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    
    auto sorted_values = values;
    sort_values(sorted_values);
    
    size_t n = sorted_values.size();
    if (n % 2 == 0) {
        return (sorted_values[n/2 - 1] + sorted_values[n/2]) / 2.0;
    } else {
        return sorted_values[n/2];
    }
}

double FeatureNormalizer::calculate_robust_std(const std::deque<double>& values, double mean) {
    if (values.size() < 2) {
        return 0.0;
    }
    
    // Calculate Median Absolute Deviation (MAD)
    std::vector<double> deviations;
    deviations.reserve(values.size());
    
    for (double value : values) {
        deviations.push_back(std::abs(value - mean));
    }
    
    std::sort(deviations.begin(), deviations.end());
    
    double mad = deviations[deviations.size() / 2];
    
    // Convert MAD to standard deviation approximation
    return mad * 1.4826;
}

double FeatureNormalizer::calculate_percentile(const std::deque<double>& values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }
    
    size_t index = static_cast<size_t>(percentile * (values.size() - 1));
    index = std::min(index, values.size() - 1);
    
    return values[index];
}

void FeatureNormalizer::sort_values(std::deque<double>& values) {
    std::sort(values.begin(), values.end());
}

} // namespace feature_engineering
} // namespace sentio
