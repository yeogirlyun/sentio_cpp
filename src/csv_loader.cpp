#include "sentio/csv_loader.hpp"
#include "sentio/binio.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctz/time_zone.h>
#include <cctz/civil_time.h>

namespace sentio {

bool load_csv(const std::string& filename, std::vector<Bar>& out) {
    // Try binary cache first for performance
    std::string bin_filename = filename.substr(0, filename.find_last_of('.')) + ".bin";
    auto cached = load_bin(bin_filename);
    if (!cached.empty()) {
        out = std::move(cached);
        return true;
    }
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string timestamp_str, symbol, open_str, high_str, low_str, close_str, volume_str;
        
        std::getline(ss, timestamp_str, ',');
        std::getline(ss, symbol, ',');
        std::getline(ss, open_str, ',');
        std::getline(ss, high_str, ',');
        std::getline(ss, low_str, ',');
        std::getline(ss, close_str, ',');
        std::getline(ss, volume_str, ',');
        
        Bar bar;
        bar.ts_utc = timestamp_str;
        
        // **MODIFIED**: Parse ISO 8601 timestamp using cctz
        try {
            // Parse the RFC3339 / ISO 8601 timestamp string (e.g., "2023-10-27T09:30:00-04:00")
            cctz::time_zone tz;
            if (cctz::load_time_zone("America/New_York", &tz)) {
                cctz::time_point<cctz::seconds> tp;
                if (cctz::parse("%Y-%m-%dT%H:%M:%S%Ez", timestamp_str, tz, &tp)) {
                    // Convert to UTC epoch seconds
                    bar.ts_nyt_epoch = tp.time_since_epoch().count();
                } else {
                    // Try alternative format
                    if (cctz::parse("%Y-%m-%d %H:%M:%S%Ez", timestamp_str, tz, &tp)) {
                        bar.ts_nyt_epoch = tp.time_since_epoch().count();
                    } else {
                        bar.ts_nyt_epoch = 0; // Could not parse
                    }
                }
            } else {
                bar.ts_nyt_epoch = 0; // Could not load timezone
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse timestamp '" << timestamp_str << "'. Error: " << e.what() << std::endl;
            bar.ts_nyt_epoch = 0;
        }
        
        try {
            bar.open = std::stod(open_str);
            bar.high = std::stod(high_str);
            bar.low = std::stod(low_str);
            bar.close = std::stod(close_str);
            bar.volume = std::stoull(volume_str);
            out.push_back(bar);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse bar data line: " << line << std::endl;
        }
    }
    
    // Save to binary cache for next time
    if (!out.empty()) {
        save_bin(bin_filename, out);
    }
    
    return true;
}

} // namespace sentio