#pragma once

#include "sentio/core.hpp"
#include <string>
#include <vector>
#include <map>

namespace sentio {

/**
 * @brief Loader for pre-generated future QQQ data files
 * 
 * This class provides access to the 10 pre-generated future QQQ tracks,
 * each representing 4 weeks (28 days) of different market regimes.
 * 
 * Track Distribution:
 * - Tracks 1, 4, 7, 10: Normal regime (4 tracks)
 * - Tracks 2, 5, 8: Volatile regime (3 tracks)  
 * - Tracks 3, 6, 9: Trending regime (3 tracks)
 */
class FutureQQQLoader {
public:
    /**
     * @brief Market regime types available in future QQQ data
     */
    enum class Regime {
        NORMAL,
        VOLATILE, 
        TRENDING
    };

    /**
     * @brief Load a specific future QQQ track
     * @param track_id Track ID (1-10)
     * @return Vector of bars for the track
     */
    static std::vector<Bar> load_track(int track_id);

    /**
     * @brief Load a random track for the specified regime
     * @param regime Market regime to load
     * @param seed Random seed for reproducible selection (optional)
     * @return Vector of bars for a random track of the specified regime
     */
    static std::vector<Bar> load_regime_track(Regime regime, int seed = -1);

    /**
     * @brief Load a random track for the specified regime (string version)
     * @param regime_str Market regime string ("normal", "volatile", "trending")
     * @param seed Random seed for reproducible selection (optional)
     * @return Vector of bars for a random track of the specified regime
     */
    static std::vector<Bar> load_regime_track(const std::string& regime_str, int seed = -1);

    /**
     * @brief Get all track IDs for a specific regime
     * @param regime Market regime
     * @return Vector of track IDs for the regime
     */
    static std::vector<int> get_regime_tracks(Regime regime);

    /**
     * @brief Convert regime string to enum
     * @param regime_str Regime string ("normal", "volatile", "trending")
     * @return Regime enum value
     */
    static Regime string_to_regime(const std::string& regime_str);

    /**
     * @brief Get the base directory for future QQQ data
     * @return Path to future QQQ data directory
     */
    static std::string get_data_directory();

    /**
     * @brief Check if all future QQQ tracks are available
     * @return True if all 10 tracks are accessible
     */
    static bool validate_tracks();

private:
    /**
     * @brief Get the file path for a specific track
     * @param track_id Track ID (1-10)
     * @return Full path to the CSV file
     */
    static std::string get_track_file_path(int track_id);

    /**
     * @brief Regime to track mapping
     */
    static const std::map<Regime, std::vector<int>> regime_tracks_;
};

} // namespace sentio
