#pragma once
#include <string>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <vector>
#include <set>
#include <cstdint>

namespace sentio::metrics {

// Convert UTC timestamp (milliseconds) to NYSE session date (YYYY-MM-DD)
inline std::string timestamp_to_session_date(std::int64_t ts_millis, const std::string& calendar = "XNYS") {
    // For simplicity, assume NYSE calendar (UTC-5/UTC-4 depending on DST)
    // In production, you'd want proper timezone handling
    std::time_t time_t = ts_millis / 1000;
    
    // Convert to Eastern Time (approximate - doesn't handle DST perfectly)
    // For production use, consider using a proper timezone library
    time_t -= 5 * 3600; // UTC-5 (EST) - this is a simplification
    
    std::tm* tm_info = std::gmtime(&time_t);
    std::ostringstream oss;
    oss << std::put_time(tm_info, "%Y-%m-%d");
    return oss.str();
}

// Count unique trading sessions in a vector of timestamps
inline int count_trading_days(const std::vector<std::int64_t>& timestamps, const std::string& calendar = "XNYS") {
    std::set<std::string> unique_dates;
    for (std::int64_t ts : timestamps) {
        unique_dates.insert(timestamp_to_session_date(ts, calendar));
    }
    return static_cast<int>(unique_dates.size());
}

} // namespace sentio::metrics
