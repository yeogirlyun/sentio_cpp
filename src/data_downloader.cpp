#include "sentio/data_downloader.hpp"
#include "sentio/polygon_client.hpp"
#include <iostream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <algorithm>

namespace sentio {

std::string get_yesterday_date() {
    std::time_t now = std::time(nullptr);
    std::time_t yesterday = now - 24 * 60 * 60; // Subtract 1 day in seconds
    
    std::tm* tm_yesterday = std::gmtime(&yesterday);
    std::ostringstream oss;
    oss << std::put_time(tm_yesterday, "%Y-%m-%d");
    return oss.str();
}

std::string get_current_date() {
    std::time_t now = std::time(nullptr);
    std::tm* tm_now = std::gmtime(&now);
    std::ostringstream oss;
    oss << std::put_time(tm_now, "%Y-%m-%d");
    return oss.str();
}

std::string calculate_start_date(int years, int months, int days) {
    std::time_t now = std::time(nullptr);
    std::time_t yesterday = now - 24 * 60 * 60; // Start from yesterday
    
    std::tm* tm_start = std::gmtime(&yesterday);
    
    if (years > 0) {
        tm_start->tm_year -= years;
    } else if (months > 0) {
        tm_start->tm_mon -= months;
        if (tm_start->tm_mon < 0) {
            tm_start->tm_mon += 12;
            tm_start->tm_year--;
        }
    } else if (days > 0) {
        tm_start->tm_mday -= days;
        // Let mktime handle month/year overflow
        std::mktime(tm_start);
    } else {
        // Default: 3 years (now explicit default)
        tm_start->tm_year -= 3;
    }
    
    std::ostringstream oss;
    oss << std::put_time(tm_start, "%Y-%m-%d");
    return oss.str();
}

std::string symbol_to_family(const std::string& symbol) {
    std::string upper_symbol = symbol;
    std::transform(upper_symbol.begin(), upper_symbol.end(), upper_symbol.begin(), ::toupper);
    
    if (upper_symbol == "QQQ" || upper_symbol == "TQQQ" || upper_symbol == "SQQQ") {
        return "qqq";
    } else if (upper_symbol == "BTC" || upper_symbol == "BTCUSD" || upper_symbol == "ETH" || upper_symbol == "ETHUSD") {
        return "bitcoin";
    } else if (upper_symbol == "TSLA" || upper_symbol == "TSLQ") {
        return "tesla";
    } else {
        return "custom";
    }
}

std::vector<std::string> get_family_symbols(const std::string& family) {
    if (family == "qqq") {
        return {"QQQ", "TQQQ", "SQQQ"};
    } else if (family == "bitcoin") {
        return {"X:BTCUSD", "X:ETHUSD"};
    } else if (family == "tesla") {
        return {"TSLA", "TSLQ"};
    } else {
        return {}; // Empty for custom family
    }
}

bool download_symbol_data(const std::string& symbol,
                         int years,
                         int months,
                         int days,
                         const std::string& timespan,
                         int multiplier,
                         bool exclude_holidays,
                         const std::string& output_dir) {
    
    // **POLYGON API KEY CHECK**
    const char* key = std::getenv("POLYGON_API_KEY");
    if (!key || std::string(key).empty()) {
        std::cerr << "❌ Error: POLYGON_API_KEY environment variable not set" << std::endl;
        std::cerr << "   Please set your Polygon API key: export POLYGON_API_KEY=your_key_here" << std::endl;
        return false;
    }
    
    std::string api_key = key;
    PolygonClient cli(api_key);
    
    // **FAMILY DETECTION**
    std::string family = symbol_to_family(symbol);
    std::vector<std::string> symbols;
    
    if (family == "custom") {
        // Single symbol download
        symbols = {symbol};
        std::cout << "📊 Downloading data for symbol: " << symbol << std::endl;
    } else {
        // Family download
        symbols = get_family_symbols(family);
        std::cout << "📊 Downloading data for " << family << " family: ";
        for (size_t i = 0; i < symbols.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << symbols[i];
        }
        std::cout << std::endl;
    }
    
    // **DATE CALCULATION**
    std::string from = calculate_start_date(years, months, days);
    std::string to = get_yesterday_date();
    
    std::cout << "📅 Current date: " << get_current_date() << std::endl;
    std::cout << "📅 Downloading ";
    if (years > 0) {
        std::cout << years << " year" << (years > 1 ? "s" : "");
    } else if (months > 0) {
        std::cout << months << " month" << (months > 1 ? "s" : "");
    } else if (days > 0) {
        std::cout << days << " day" << (days > 1 ? "s" : "");
    } else {
        std::cout << "3 years (default)";
    }
    std::cout << " of data: " << from << " to " << to << std::endl;
    std::cout << "📈 Timespan: " << timespan << " (multiplier: " << multiplier << ")" << std::endl;
    
    // **DOWNLOAD EACH SYMBOL**
    bool all_success = true;
    for (const auto& sym : symbols) {
        std::cout << "\\n🔄 Downloading " << sym << "..." << std::endl;
        
        AggsQuery q;
        q.symbol = sym;
        q.from = from;
        q.to = to;
        q.timespan = timespan;
        q.multiplier = multiplier;
        q.adjusted = true;
        q.sort = "asc";
        
        try {
            auto bars = cli.get_aggs_all(q);
            
            if (bars.empty()) {
                std::cerr << "⚠️  Warning: No data received for " << sym << std::endl;
                continue;
            }
            
            // **FILE NAMING**
            std::string suffix;
            if (exclude_holidays) suffix += "_NH";
            std::string fname = output_dir + "/" + sym + suffix + ".csv";
            
            // **WRITE CSV**
            cli.write_csv(fname, sym, bars, exclude_holidays);
            
            std::cout << "✅ Wrote " << bars.size() << " bars -> " << fname << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Error downloading " << sym << ": " << e.what() << std::endl;
            all_success = false;
        }
    }
    
    if (all_success) {
        std::cout << "\\n🎉 Download completed successfully!" << std::endl;
        std::cout << "💡 Tip: The smart data loading system will automatically use this fresh data on your next run." << std::endl;
    } else {
        std::cout << "\\n⚠️  Download completed with some errors. Check the output above." << std::endl;
    }
    
    return all_success;
}

} // namespace sentio
