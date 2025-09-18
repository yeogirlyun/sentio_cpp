#include "sentio/csv_loader.hpp"
#include "sentio/binio.hpp"
#include "sentio/global_leverage_config.hpp"
#include "sentio/leverage_aware_csv_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <cctz/time_zone.h>
#include <cctz/civil_time.h>

namespace sentio {

bool load_csv(const std::string& filename, std::vector<Bar>& out) {
    // Check if this is a leverage instrument and theoretical pricing is enabled
    if (GlobalLeverageConfig::is_theoretical_leverage_pricing_enabled()) {
        // Extract symbol from filename
        std::string symbol;
        size_t last_slash = filename.find_last_of("/\\");
        size_t last_dot = filename.find_last_of(".");
        if (last_slash != std::string::npos && last_dot != std::string::npos && last_dot > last_slash) {
            symbol = filename.substr(last_slash + 1, last_dot - last_slash - 1);
            // Remove any suffix like _NH_ALIGNED
            size_t underscore = symbol.find('_');
            if (underscore != std::string::npos) {
                symbol = symbol.substr(0, underscore);
            }
        }
        
        // If this is a leverage instrument, use theoretical pricing
        if (symbol == "TQQQ" || symbol == "SQQQ" || symbol == "PSQ") {
            std::cout << "🧮 Using theoretical pricing for " << symbol << " (based on QQQ)" << std::endl;
            return load_csv_leverage_aware(symbol, out);
        }
    }
    
    namespace fs = std::filesystem;
    
    // **SMART FRESHNESS-BASED LOADING**: Choose between CSV and binary based on file timestamps
    std::string bin_filename = filename.substr(0, filename.find_last_of('.')) + ".bin";
    
    bool csv_exists = fs::exists(filename);
    bool bin_exists = fs::exists(bin_filename);
    
    // **FRESHNESS COMPARISON**: Use binary only if it's newer than CSV
    bool use_binary = false;
    if (bin_exists && csv_exists) {
        auto csv_time = fs::last_write_time(filename);
        auto bin_time = fs::last_write_time(bin_filename);
        use_binary = (bin_time >= csv_time);
        
        if (use_binary) {
            std::cout << "📦 Using cached binary data (fresher than CSV): " << bin_filename << std::endl;
        } else {
            std::cout << "🔄 CSV file is newer than binary cache, reloading: " << filename << std::endl;
        }
    } else if (bin_exists && !csv_exists) {
        use_binary = true;
        std::cout << "📦 Using binary data (CSV not found): " << bin_filename << std::endl;
    } else if (!bin_exists && csv_exists) {
        use_binary = false;
        std::cout << "📄 Loading CSV data (no binary cache): " << filename << std::endl;
    } else {
        std::cerr << "❌ Neither CSV nor binary file exists: " << filename << std::endl;
        return false;
    }
    
    // **LOAD FROM BINARY**: Use cached binary if it's fresher
    if (use_binary) {
        auto cached = load_bin(bin_filename);
        if (!cached.empty()) {
            out = std::move(cached);
            return true;
        } else {
            std::cerr << "⚠️  Binary cache corrupted, falling back to CSV: " << bin_filename << std::endl;
        }
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
        
        // Standard Polygon format: timestamp,symbol,open,high,low,close,volume
        std::getline(ss, timestamp_str, ',');
        std::getline(ss, symbol, ',');
        std::getline(ss, open_str, ',');
        std::getline(ss, high_str, ',');
        std::getline(ss, low_str, ',');
        std::getline(ss, close_str, ',');
        std::getline(ss, volume_str, ',');
        
        Bar bar;
        bar.ts_utc = timestamp_str;
        
        // **MODIFIED**: Parse ISO 8601 timestamp directly as UTC
        try {
            // Parse the RFC3339 / ISO 8601 timestamp string (e.g., "2023-10-27T13:30:00Z")
            cctz::time_zone utc_tz;
            if (cctz::load_time_zone("UTC", &utc_tz)) {
                cctz::time_point<cctz::seconds> utc_tp;
                if (cctz::parse("%Y-%m-%dT%H:%M:%S%Ez", timestamp_str, utc_tz, &utc_tp)) {
                    bar.ts_utc_epoch = utc_tp.time_since_epoch().count();
                } else {
                    // Try alternative format with Z suffix
                    if (cctz::parse("%Y-%m-%dT%H:%M:%SZ", timestamp_str, utc_tz, &utc_tp)) {
                        bar.ts_utc_epoch = utc_tp.time_since_epoch().count();
                    } else {
                        // Try space format with timezone
                        if (cctz::parse("%Y-%m-%d %H:%M:%S%Ez", timestamp_str, utc_tz, &utc_tp)) {
                            bar.ts_utc_epoch = utc_tp.time_since_epoch().count();
                        } else {
                            // Try space format without timezone (assume UTC)
                            if (cctz::parse("%Y-%m-%d %H:%M:%S", timestamp_str, utc_tz, &utc_tp)) {
                                bar.ts_utc_epoch = utc_tp.time_since_epoch().count();
                            } else {
                                bar.ts_utc_epoch = 0;
                            }
                        }
                    }
                }
            } else {
                bar.ts_utc_epoch = 0; // Could not load timezone
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse timestamp '" << timestamp_str << "'. Error: " << e.what() << std::endl;
            bar.ts_utc_epoch = 0;
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
    
    // **REGENERATE BINARY CACHE**: Save to binary cache for next time
    if (!out.empty()) {
        save_bin(bin_filename, out);
        std::cout << "💾 Regenerated binary cache: " << bin_filename << " (" << out.size() << " bars)" << std::endl;
    }
    
    return true;
}

} // namespace sentio