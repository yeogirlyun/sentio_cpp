#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

namespace sentio::utils {

/**
 * @brief Formats a value with specified precision
 * @param value The value to format
 * @param precision Number of decimal places
 * @return Formatted string
 */
template<typename T>
std::string format_precision(T value, int precision = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

/**
 * @brief Formats a percentage value
 * @param value The value to format as percentage
 * @param precision Number of decimal places
 * @return Formatted percentage string
 */
template<typename T>
std::string format_percentage(T value, int precision = 2) {
    return format_precision(value * 100.0, precision) + "%";
}

/**
 * @brief Formats a vector of values as a comma-separated string
 * @param values The vector of values
 * @param precision Number of decimal places
 * @return Comma-separated string
 */
template<typename T>
std::string format_vector(const std::vector<T>& values, int precision = 2) {
    if (values.empty()) return "[]";
    
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << format_precision(values[i], precision);
    }
    oss << "]";
    return oss.str();
}

/**
 * @brief Formats a key-value pair
 * @param key The key
 * @param value The value
 * @param precision Number of decimal places for numeric values
 * @return Formatted key-value string
 */
template<typename T>
std::string format_key_value(const std::string& key, T value, int precision = 2) {
    return key + ": " + format_precision(value, precision);
}

/**
 * @brief Formats multiple key-value pairs
 * @param pairs Vector of key-value pairs
 * @param precision Number of decimal places for numeric values
 * @return Comma-separated key-value string
 */
template<typename T>
std::string format_key_values(const std::vector<std::pair<std::string, T>>& pairs, int precision = 2) {
    if (pairs.empty()) return "";
    
    std::ostringstream oss;
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << format_key_value(pairs[i].first, pairs[i].second, precision);
    }
    return oss.str();
}

/**
 * @brief Formats a result summary with key metrics
 * @param metrics Map of metric names to values
 * @param precision Number of decimal places
 * @return Formatted summary string
 */
template<typename T>
std::string format_result_summary(const std::vector<std::pair<std::string, T>>& metrics, int precision = 2) {
    return format_key_values(metrics, precision);
}

} // namespace sentio::utils
