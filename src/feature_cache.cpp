#include "sentio/feature_cache.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace sentio {

bool FeatureCache::load_from_csv(const std::string& feature_file_path) {
    std::ifstream file(feature_file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open feature file: " << feature_file_path << std::endl;
        return false;
    }

    std::string line;
    bool is_header = true;
    size_t lines_processed = 0;

    while (std::getline(file, line)) {
        if (is_header) {
            // Parse header to get feature names
            std::stringstream ss(line);
            std::string column;
            
            // Skip bar_index and timestamp columns
            std::getline(ss, column, ','); // bar_index
            std::getline(ss, column, ','); // timestamp
            
            // Read feature names
            feature_names_.clear();
            while (std::getline(ss, column, ',')) {
                feature_names_.push_back(column);
            }
            
            std::cout << "FeatureCache: Loaded " << feature_names_.size() << " feature names" << std::endl;
            is_header = false;
            continue;
        }

        // Parse data line
        std::stringstream ss(line);
        std::string cell;
        
        // Read bar_index
        if (!std::getline(ss, cell, ',')) continue;
        int bar_index = std::stoi(cell);
        
        // Skip timestamp
        if (!std::getline(ss, cell, ',')) continue;
        
        // Read features
        std::vector<double> features;
        features.reserve(feature_names_.size());
        
        while (std::getline(ss, cell, ',')) {
            features.push_back(std::stod(cell));
        }
        
        // Verify feature count  
        if (features.size() != feature_names_.size()) {
            std::cerr << "CRITICAL: Bar " << bar_index << " has " << features.size() 
                      << " features, expected " << feature_names_.size() << std::endl;
            std::cerr << "CSV line: " << line << std::endl;
            std::cerr << "This will cause missing features in get_features()!" << std::endl;
            continue;  // This is the bug - skipping bars causes missing data
        }
        
        // Debug output for first few bars
        if (lines_processed < 3) {
            std::cout << "[DEBUG] Bar " << bar_index << ": loaded " << features.size() 
                      << " features (expected " << feature_names_.size() << ")" << std::endl;
        }
        
        // Store features
        features_by_bar_[bar_index] = std::move(features);
        lines_processed++;
        
        // Progress reporting
        if (lines_processed % 50000 == 0) {
            std::cout << "FeatureCache: Loaded " << lines_processed << " bars..." << std::endl;
        }
    }

    total_bars_ = lines_processed;
    file.close();

    std::cout << "FeatureCache: Successfully loaded " << total_bars_ << " bars with " 
              << feature_names_.size() << " features each" << std::endl;
    std::cout << "FeatureCache: Recommended starting bar: " << recommended_start_bar_ << std::endl;
    
    return true;
}

std::vector<double> FeatureCache::get_features(int bar_index) const {
    auto it = features_by_bar_.find(bar_index);
    if (it != features_by_bar_.end()) {
        static int get_calls = 0;
        get_calls++;
        if (get_calls <= 5) {
            std::cout << "[DEBUG] FeatureCache::get_features(" << bar_index 
                      << ") returning " << it->second.size() << " features" << std::endl;
        }
        return it->second;
    }
    std::cout << "[ERROR] FeatureCache::get_features(" << bar_index 
              << ") - bar not found! Returning empty vector." << std::endl;
    return {}; // Return empty vector if not found
}

bool FeatureCache::has_features(int bar_index) const {
    return features_by_bar_.find(bar_index) != features_by_bar_.end();
}

size_t FeatureCache::get_bar_count() const {
    return total_bars_;
}

int FeatureCache::get_recommended_start_bar() const {
    return recommended_start_bar_;
}

const std::vector<std::string>& FeatureCache::get_feature_names() const {
    return feature_names_;
}

} // namespace sentio
