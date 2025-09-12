#pragma once

#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>

namespace sentio::rules::utils {

/**
 * @brief Validates if there's sufficient data for volume window calculations
 * @param data_size Current data size
 * @param vol_win Volume window size
 * @return true if sufficient data, false otherwise
 */
inline bool has_volume_window_data(int data_size, int vol_win) {
    return data_size > vol_win;
}

/**
 * @brief Validates if there's sufficient data for volume window calculations
 * @param data_size Current data size
 * @param vol_win Volume window size
 * @return true if sufficient data, false otherwise
 */
template<typename Container>
inline bool has_volume_window_data(const Container& data, int vol_win) {
    return static_cast<int>(data.size()) > vol_win;
}

/**
 * @brief Safely manages a sliding window with volume window validation
 * @tparam T Data type
 */
template<typename T>
class SlidingWindow {
private:
    std::deque<T> window_;
    int max_size_;
    T sum_{};
    T sum_squared_{};
    
public:
    explicit SlidingWindow(int max_size) : max_size_(max_size) {}
    
    void push(T value) {
        window_.push_back(value);
        sum_ += value;
        sum_squared_ += value * value;
        
        if (static_cast<int>(window_.size()) > max_size_) {
            T front_value = window_.front();
            window_.pop_front();
            sum_ -= front_value;
            sum_squared_ -= front_value * front_value;
        }
    }
    
    bool has_sufficient_data() const {
        return static_cast<int>(window_.size()) > max_size_;
    }
    
    T mean() const {
        return window_.empty() ? T{} : sum_ / static_cast<T>(window_.size());
    }
    
    T variance() const {
        if (window_.empty()) return T{};
        T m = mean();
        return std::max(T{}, sum_squared_ / static_cast<T>(window_.size()) - m * m);
    }
    
    T standard_deviation() const {
        return std::sqrt(variance());
    }
    
    size_t size() const { return window_.size(); }
    bool empty() const { return window_.empty(); }
};

/**
 * @brief Calculates rolling statistics with volume window validation
 * @param data Input data
 * @param vol_win Volume window size
 * @param output_mean Output mean values
 * @param output_variance Output variance values
 */
template<typename T>
void calculate_rolling_stats(const std::vector<T>& data, int vol_win,
                           std::vector<T>& output_mean, std::vector<T>& output_variance) {
    int N = static_cast<int>(data.size());
    output_mean.assign(N, T{});
    output_variance.assign(N, T{});
    
    SlidingWindow<T> window(vol_win);
    
    for (int i = 0; i < N; ++i) {
        window.push(data[i]);
        
        if (window.has_sufficient_data()) {
            output_mean[i] = window.mean();
            output_variance[i] = window.variance();
        }
    }
}

/**
 * @brief Calculates rolling mean with volume window validation
 * @param data Input data
 * @param vol_win Volume window size
 * @param output Output mean values
 */
template<typename T>
void calculate_rolling_mean(const std::vector<T>& data, int vol_win, std::vector<T>& output) {
    int N = static_cast<int>(data.size());
    output.assign(N, T{});
    
    SlidingWindow<T> window(vol_win);
    
    for (int i = 0; i < N; ++i) {
        window.push(data[i]);
        
        if (window.has_sufficient_data()) {
            output[i] = window.mean();
        }
    }
}

/**
 * @brief Calculates rolling variance with volume window validation
 * @param data Input data
 * @param vol_win Volume window size
 * @param output Output variance values
 */
template<typename T>
void calculate_rolling_variance(const std::vector<T>& data, int vol_win, std::vector<T>& output) {
    int N = static_cast<int>(data.size());
    output.assign(N, T{});
    
    SlidingWindow<T> window(vol_win);
    
    for (int i = 0; i < N; ++i) {
        window.push(data[i]);
        
        if (window.has_sufficient_data()) {
            output[i] = window.variance();
        }
    }
}

} // namespace sentio::rules::utils
