#include "sentio/future_qqq_loader.hpp"
#include "sentio/csv_loader.hpp"
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>

namespace sentio {

// Define regime to track mapping
const std::map<FutureQQQLoader::Regime, std::vector<int>> FutureQQQLoader::regime_tracks_ = {
    {Regime::NORMAL, {1, 4, 7, 10}},     // 4 normal tracks
    {Regime::VOLATILE, {2, 5, 8}},       // 3 volatile tracks
    {Regime::TRENDING, {3, 6, 9}}        // 3 trending tracks
};

std::vector<Bar> FutureQQQLoader::load_track(int track_id) {
    if (track_id < 1 || track_id > 10) {
        throw std::invalid_argument("Track ID must be between 1 and 10, got: " + std::to_string(track_id));
    }

    std::string file_path = get_track_file_path(track_id);
    
    if (!std::filesystem::exists(file_path)) {
        throw std::runtime_error("Future QQQ track file not found: " + file_path);
    }

    std::cout << "📊 Loading future QQQ track " << track_id << " from: " << file_path << std::endl;

    // Use existing CSV loader to load the data
    std::vector<Bar> bars;
    bool success = load_csv(file_path, bars);
    
    if (!success || bars.empty()) {
        throw std::runtime_error("Failed to load future QQQ track " + std::to_string(track_id));
    }

    std::cout << "✅ Loaded " << bars.size() << " bars from future QQQ track " << track_id << std::endl;
    return bars;
}

std::vector<Bar> FutureQQQLoader::load_regime_track(Regime regime, int seed) {
    auto track_ids = get_regime_tracks(regime);
    
    if (track_ids.empty()) {
        throw std::runtime_error("No tracks available for the specified regime");
    }

    // Select random track from the regime
    int selected_track;
    if (seed >= 0) {
        // Use seed for reproducible selection
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, track_ids.size() - 1);
        selected_track = track_ids[dist(rng)];
    } else {
        // Use random device for true randomness
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> dist(0, track_ids.size() - 1);
        selected_track = track_ids[dist(rng)];
    }

    std::string regime_name;
    switch (regime) {
        case Regime::NORMAL: regime_name = "normal"; break;
        case Regime::VOLATILE: regime_name = "volatile"; break;
        case Regime::TRENDING: regime_name = "trending"; break;
    }

    std::cout << "🎯 Selected track " << selected_track << " for " << regime_name << " regime" << std::endl;
    return load_track(selected_track);
}

std::vector<Bar> FutureQQQLoader::load_regime_track(const std::string& regime_str, int seed) {
    Regime regime = string_to_regime(regime_str);
    return load_regime_track(regime, seed);
}

std::vector<int> FutureQQQLoader::get_regime_tracks(Regime regime) {
    auto it = regime_tracks_.find(regime);
    if (it != regime_tracks_.end()) {
        return it->second;
    }
    return {};
}

FutureQQQLoader::Regime FutureQQQLoader::string_to_regime(const std::string& regime_str) {
    if (regime_str == "normal") {
        return Regime::NORMAL;
    } else if (regime_str == "volatile") {
        return Regime::VOLATILE;
    } else if (regime_str == "trending") {
        return Regime::TRENDING;
    } else {
        // Default to normal for unknown regimes (including "bear", "bull", etc.)
        std::cout << "⚠️  Unknown regime '" << regime_str << "', defaulting to 'normal'" << std::endl;
        return Regime::NORMAL;
    }
}

std::string FutureQQQLoader::get_data_directory() {
    return "data/future_qqq";
}

bool FutureQQQLoader::validate_tracks() {
    std::string base_dir = get_data_directory();
    
    if (!std::filesystem::exists(base_dir)) {
        std::cerr << "❌ Future QQQ data directory not found: " << base_dir << std::endl;
        return false;
    }

    // Check all 10 tracks
    for (int i = 1; i <= 10; ++i) {
        std::string file_path = get_track_file_path(i);
        if (!std::filesystem::exists(file_path)) {
            std::cerr << "❌ Future QQQ track " << i << " not found: " << file_path << std::endl;
            return false;
        }
    }

    std::cout << "✅ All 10 future QQQ tracks validated successfully" << std::endl;
    return true;
}

std::string FutureQQQLoader::get_track_file_path(int track_id) {
    std::string base_dir = get_data_directory();
    
    // Format track ID with leading zero (e.g., "01", "02", ..., "10")
    std::string track_id_str = (track_id < 10) ? "0" + std::to_string(track_id) : std::to_string(track_id);
    
    return base_dir + "/future_qqq_track_" + track_id_str + ".csv";
}

} // namespace sentio
