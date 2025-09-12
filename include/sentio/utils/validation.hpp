#pragma once

#include <vector>
#include <string>

namespace sentio::utils {

/**
 * @brief Validates if a series has sufficient data for the given index range
 * @param series The data series to validate
 * @param start_idx The starting index
 * @param end_idx The ending index (optional, defaults to series size)
 * @return true if series has sufficient data, false otherwise
 */
template<typename T>
bool has_sufficient_data(const std::vector<T>& series, int start_idx, int end_idx = -1) {
    if (end_idx == -1) end_idx = static_cast<int>(series.size());
    return static_cast<int>(series.size()) > start_idx && start_idx >= 0 && end_idx > start_idx;
}

/**
 * @brief Validates if a series has sufficient data for the given start index
 * @param series The data series to validate
 * @param start_idx The starting index
 * @return true if series has sufficient data, false otherwise
 */
template<typename T>
bool has_sufficient_data(const std::vector<T>& series, int start_idx) {
    return static_cast<int>(series.size()) > start_idx && start_idx >= 0;
}

/**
 * @brief Validates if a series has sufficient data for volume window
 * @param series The data series to validate
 * @param vol_win The volume window size
 * @return true if series has sufficient data, false otherwise
 */
template<typename T>
bool has_volume_window_data(const std::vector<T>& series, int vol_win) {
    return static_cast<int>(series.size()) > vol_win;
}

/**
 * @brief Safely extracts a sub-range from a series with validation
 * @param series The source series
 * @param start_idx The starting index
 * @param end_idx The ending index
 * @return A new vector containing the sub-range, or empty vector if invalid
 */
template<typename T>
std::vector<T> extract_range(const std::vector<T>& series, int start_idx, int end_idx = -1) {
    if (!has_sufficient_data(series, start_idx, end_idx)) {
        return {};
    }
    
    if (end_idx == -1) end_idx = static_cast<int>(series.size());
    int actual_end = std::min(end_idx, static_cast<int>(series.size()));
    
    return std::vector<T>(series.begin() + start_idx, series.begin() + actual_end);
}

/**
 * @brief Safely extracts multiple sub-ranges from multiple series
 * @param series_vector Vector of series to extract from
 * @param start_idx The starting index
 * @param end_idx The ending index
 * @return Vector of extracted ranges
 */
template<typename T>
std::vector<std::vector<T>> extract_multiple_ranges(const std::vector<std::vector<T>>& series_vector, 
                                                   int start_idx, int end_idx = -1) {
    std::vector<std::vector<T>> result;
    result.reserve(series_vector.size());
    
    for (const auto& series : series_vector) {
        result.emplace_back(extract_range(series, start_idx, end_idx));
    }
    
    return result;
}

} // namespace sentio::utils
