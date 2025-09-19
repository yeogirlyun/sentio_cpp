#pragma once
#include <string>
#include <cstdint>

namespace sentio {

/**
 * Dataset Metadata for Audit Traceability
 * 
 * This structure captures comprehensive information about the dataset used
 * in a backtest, enabling exact reproduction and verification.
 */
struct DatasetMetadata {
    // Source type classification
    std::string source_type = "unknown";        // "historical_csv", "future_qqq_track", "mars_simulation", etc.
    
    // File information
    std::string file_path = "";                 // Full path to the data file
    std::string file_hash = "";                 // SHA256 hash for integrity verification
    
    // Regime/track information (for AI tests)
    std::string track_id = "";                  // For future QQQ tracks (e.g., "track_01")
    std::string regime = "";                    // For AI regime tests (e.g., "normal", "volatile", "trending")
    std::string mode = "hybrid";                // Strategy execution mode ("historical", "hybrid", "live", etc.)
    
    // Dataset characteristics
    int bars_count = 0;                         // Total number of bars in the dataset
    std::int64_t time_range_start = 0;          // First timestamp in the dataset (milliseconds)
    std::int64_t time_range_end = 0;            // Last timestamp in the dataset (milliseconds)
    
    // Trading Block information (canonical evaluation units)
    int available_trading_blocks = 0;           // How many complete 480-bar Trading Blocks available
    int trading_blocks_tested = 0;              // How many Trading Blocks were actually tested
    double dataset_trading_days = 0.0;          // Total trading days equivalent (bars ÷ 390)
    double dataset_calendar_days = 0.0;         // Calendar days span of the dataset
    
    // Performance context
    std::string frequency = "1min";             // Bar frequency (1min, 5min, etc.)
    std::string market_hours = "RTH";           // Regular Trading Hours or Extended
    
    // Helper methods
    bool is_valid() const {
        return !source_type.empty() && source_type != "unknown" && bars_count > 0;
    }
    
    std::string to_json() const {
        std::string json = "{";
        json += "\"source_type\":\"" + source_type + "\",";
        json += "\"file_path\":\"" + file_path + "\",";
        json += "\"file_hash\":\"" + file_hash + "\",";
        json += "\"track_id\":\"" + track_id + "\",";
        json += "\"regime\":\"" + regime + "\",";
        json += "\"bars_count\":" + std::to_string(bars_count) + ",";
        json += "\"time_range_start\":" + std::to_string(time_range_start) + ",";
        json += "\"time_range_end\":" + std::to_string(time_range_end) + ",";
        // Trading Block metadata
        json += "\"available_trading_blocks\":" + std::to_string(available_trading_blocks) + ",";
        json += "\"trading_blocks_tested\":" + std::to_string(trading_blocks_tested) + ",";
        json += "\"dataset_trading_days\":" + std::to_string(dataset_trading_days) + ",";
        json += "\"dataset_calendar_days\":" + std::to_string(dataset_calendar_days) + ",";
        json += "\"frequency\":\"" + frequency + "\",";
        json += "\"market_hours\":\"" + market_hours + "\"";
        json += "}";
        return json;
    }
    
    // Calculate Trading Block availability from dataset
    void calculate_trading_blocks(int block_size = 480, int warmup_bars = 250) {
        if (bars_count > warmup_bars) {
            int usable_bars = bars_count - warmup_bars;
            available_trading_blocks = usable_bars / block_size;
        } else {
            available_trading_blocks = 0;
        }
        
        // Calculate trading days equivalent (assuming 390 bars per trading day)
        dataset_trading_days = static_cast<double>(bars_count) / 390.0;
        
        // Calculate calendar days from timestamp range
        if (time_range_end > time_range_start) {
            dataset_calendar_days = static_cast<double>(time_range_end - time_range_start) / (24 * 60 * 60 * 1000);
        }
    }
};

} // namespace sentio
