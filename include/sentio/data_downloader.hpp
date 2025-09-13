#pragma once
#include <string>
#include <vector>

namespace sentio {

// **DATA DOWNLOAD UTILITIES**: Extracted from poly_fetch for reuse in sentio_cli

/**
 * Calculate start date for data download based on time period
 * @param years Number of years to go back (0 = ignore)
 * @param months Number of months to go back (0 = ignore) 
 * @param days Number of days to go back (0 = ignore)
 * @return Start date string in YYYY-MM-DD format
 */
std::string calculate_start_date(int years, int months, int days);

/**
 * Get yesterday's date in YYYY-MM-DD format
 * @return Yesterday's date string
 */
std::string get_yesterday_date();

/**
 * Get current date in YYYY-MM-DD format
 * @return Current date string
 */
std::string get_current_date();

/**
 * Map symbol to family name for poly_fetch
 * @param symbol Symbol like "QQQ", "BTC", "TSLA"
 * @return Family name like "qqq", "bitcoin", "tesla"
 */
std::string symbol_to_family(const std::string& symbol);

/**
 * Get symbols for a given family
 * @param family Family name like "qqq", "bitcoin", "tesla"
 * @return Vector of symbols in that family
 */
std::vector<std::string> get_family_symbols(const std::string& family);

/**
 * Download data for a symbol family using Polygon API
 * @param symbol Primary symbol (e.g., "QQQ")
 * @param years Number of years to download (0 = ignore)
 * @param months Number of months to download (0 = ignore)
 * @param days Number of days to download (0 = ignore)
 * @param timespan "day", "hour", or "minute"
 * @param multiplier Aggregation multiplier (default: 1)
 * @param exclude_holidays Whether to exclude holidays (adds _NH suffix)
 * @param output_dir Output directory (default: "data/equities")
 * @return True if successful, false otherwise
 */
bool download_symbol_data(const std::string& symbol,
                         int years = 0,
                         int months = 0, 
                         int days = 0,
                         const std::string& timespan = "day",
                         int multiplier = 1,
                         bool exclude_holidays = false,
                         const std::string& output_dir = "data/equities");

} // namespace sentio
