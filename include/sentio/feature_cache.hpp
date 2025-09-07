#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace sentio {

/**
 * FeatureCache loads and provides access to pre-computed features
 * This eliminates the need for expensive real-time technical indicator calculations
 */
class FeatureCache {
public:
    /**
     * Load pre-computed features from CSV file
     * @param feature_file_path Path to the feature CSV file (e.g., QQQ_RTH_features.csv)
     * @return true if loaded successfully
     */
    bool load_from_csv(const std::string& feature_file_path);

    /**
     * Get features for a specific bar index
     * @param bar_index The bar index (0-based)
     * @return Vector of 55 features, or empty vector if not found
     */
    std::vector<double> get_features(int bar_index) const;

    /**
     * Check if features are available for a given bar index
     */
    bool has_features(int bar_index) const;

    /**
     * Get the total number of bars with features
     */
    size_t get_bar_count() const;

    /**
     * Get the recommended starting bar index (after warmup)
     */
    int get_recommended_start_bar() const;

    /**
     * Get feature names in order
     */
    const std::vector<std::string>& get_feature_names() const;

private:
    // Map from bar_index to feature vector
    std::unordered_map<int, std::vector<double>> features_by_bar_;
    
    // Feature names in order
    std::vector<std::string> feature_names_;
    
    // Recommended starting bar (after warmup period)
    int recommended_start_bar_ = 300;
    
    // Total number of bars
    size_t total_bars_ = 0;
};

} // namespace sentio
