#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <deque>
#include <mutex>

namespace sentio {
namespace feature_engineering {

struct NormalizationStats {
    double mean{0.0};
    double std{0.0};
    double min{0.0};
    double max{0.0};
    size_t count{0};
};

class FeatureNormalizer {
public:
    FeatureNormalizer(size_t window_size = 252);
    
    // Normalization methods
    std::vector<double> normalize_features(const std::vector<double>& features);
    std::vector<double> denormalize_features(const std::vector<double>& normalized_features);
    
    // Statistics management
    void update_stats(const std::vector<double>& features);
    void reset_stats();
    
    // Feature-specific normalization
    std::vector<double> z_score_normalize(const std::vector<double>& features);
    std::vector<double> min_max_normalize(const std::vector<double>& features);
    std::vector<double> robust_normalize(const std::vector<double>& features);
    
    // Outlier handling
    std::vector<double> clip_outliers(const std::vector<double>& features, double threshold = 3.0);
    std::vector<double> winsorize(const std::vector<double>& features, double percentile = 0.05);
    
    // Validation
    bool is_normalized(const std::vector<double>& features) const;
    std::vector<bool> get_outlier_mask(const std::vector<double>& features, double threshold = 3.0) const;
    
    // Statistics access
    NormalizationStats get_stats(size_t feature_index) const;
    std::vector<NormalizationStats> get_all_stats() const;
    
    // Configuration
    void set_window_size(size_t window_size);
    void set_outlier_threshold(double threshold);
    void set_winsorize_percentile(double percentile);
    
private:
    size_t window_size_;
    double outlier_threshold_{3.0};
    double winsorize_percentile_{0.05};
    
    std::vector<std::deque<double>> feature_history_;
    std::vector<NormalizationStats> stats_;
    mutable std::mutex stats_mutex_;
    
    void update_feature_stats(size_t feature_index, double value);
    double calculate_robust_mean(const std::deque<double>& values);
    double calculate_robust_std(const std::deque<double>& values, double mean);
    double calculate_percentile(const std::deque<double>& values, double percentile);
    void sort_values(std::deque<double>& values);
};

} // namespace feature_engineering
} // namespace sentio
