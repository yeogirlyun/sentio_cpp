#include "sentio/polygon_client.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <cctz/time_zone.h>
#include <cctz/civil_time.h>
#include <fstream>
#include <thread>
#include <chrono>
#include <sstream>
#include <iomanip>

using json = nlohmann::json;
namespace sentio {

static size_t write_cb(void* contents, size_t size, size_t nmemb, void* userp) {
  size_t total = size * nmemb;
  std::string* s = static_cast<std::string*>(userp);
  s->append(static_cast<char*>(contents), total);
  return total;
}

static std::string rfc3339_nyc_from_epoch_ms(long long ms) {
  using namespace std::chrono;
  cctz::time_zone tz;
  if (!cctz::load_time_zone("America/New_York", &tz)) {
    return "1970-01-01T00:00:00+00:00"; // fallback
  }
  
  cctz::time_point<cctz::seconds> tp{cctz::seconds{ms / 1000}};
  auto lt = cctz::convert(tp, tz);
  auto ct = cctz::civil_second(lt);
  
  std::ostringstream oss;
  oss << std::setfill('0') 
      << std::setw(4) << ct.year() << "-"
      << std::setw(2) << ct.month() << "-"
      << std::setw(2) << ct.day() << "T"
      << std::setw(2) << ct.hour() << ":"
      << std::setw(2) << ct.minute() << ":"
      << std::setw(2) << ct.second() << "-04:00"; // EDT offset
  
  return oss.str();
}

// **NEW**: Robust RTH check directly from a UTC timestamp.
// This avoids error-prone string parsing and ensures correct time zone handling.
static bool is_rth_nyc_from_utc_ms(long long utc_ms) {
    cctz::time_zone tz;
    if (!cctz::load_time_zone("America/New_York", &tz)) {
        return false;
    }
    
    cctz::time_point<cctz::seconds> tp{cctz::seconds{utc_ms / 1000}};
    auto lt = cctz::convert(tp, tz);
    auto ct = cctz::civil_second(lt);
    
    // Check if weekend (Saturday = 6, Sunday = 0)
    auto wd = cctz::get_weekday(ct);
    if (wd == cctz::weekday::saturday || wd == cctz::weekday::sunday) {
        return false;
    }
    
    // Check if RTH (9:30 AM - 4:00 PM ET)
    if (ct.hour() < 9 || (ct.hour() == 9 && ct.minute() < 30)) {
        return false;  // Before 9:30 AM
    }
    if (ct.hour() >= 16) {
        return false;  // After 4:00 PM
    }
    
    return true;
}

static bool is_us_market_holiday_2022_2025(int year, int month, int day) {
  // Simple holiday check for common US market holidays
  // This is a simplified version - for production use, integrate with the full calendar system
  
  // New Year's Day (observed)
  if (month == 1 && day == 1) return true;
  if (month == 1 && day == 2) return true; // observed if Jan 1 is Sunday
  
  // MLK Day (3rd Monday in January)
  if (month == 1 && day >= 15 && day <= 21) {
    // Simple check - this could be more precise
    return true;
  }
  
  // Presidents Day (3rd Monday in February)
  if (month == 2 && day >= 15 && day <= 21) {
    return true;
  }
  
  // Good Friday (varies by year)
  if (year == 2022 && month == 4 && day == 15) return true;
  if (year == 2023 && month == 4 && day == 7) return true;
  if (year == 2024 && month == 3 && day == 29) return true;
  if (year == 2025 && month == 4 && day == 18) return true;
  
  // Memorial Day (last Monday in May)
  if (month == 5 && day >= 25 && day <= 31) {
    return true;
  }
  
  // Juneteenth (observed)
  if (month == 6 && day == 19) return true;
  if (month == 6 && day == 20) return true; // observed if Jun 19 is Sunday
  
  // Independence Day (observed)
  if (month == 7 && day == 4) return true;
  if (month == 7 && day == 5) return true; // observed if Jul 4 is Sunday
  
  // Labor Day (1st Monday in September)
  if (month == 9 && day >= 1 && day <= 7) {
    return true;
  }
  
  // Thanksgiving (4th Thursday in November)
  if (month == 11 && day >= 22 && day <= 28) {
    return true;
  }
  
  // Christmas (observed)
  if (month == 12 && day == 25) return true;
  if (month == 12 && day == 26) return true; // observed if Dec 25 is Sunday
  
  return false;
}

PolygonClient::PolygonClient(std::string api_key) : api_key_(std::move(api_key)) {}

std::string PolygonClient::get_(const std::string& url) {
    // This function's implementation remains unchanged.
    CURL* curl = curl_easy_init();
    std::string buffer;
    if (!curl) return buffer;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    struct curl_slist* headers = nullptr;
    std::string auth = "Authorization: Bearer " + api_key_;
    headers = curl_slist_append(headers, auth.c_str());
    headers = curl_slist_append(headers, "Accept: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return buffer;
}

std::vector<AggBar> PolygonClient::get_aggs_all(const AggsQuery& q, int max_pages) {
    // This function's implementation remains unchanged.
    std::vector<AggBar> out;
    std::string base = "https://api.polygon.io/v2/aggs/ticker/" + q.symbol + "/range/" + std::to_string(q.multiplier) + "/" + q.timespan + "/" + q.from + "/" + q.to + "?adjusted=" + (q.adjusted?"true":"false") + "&sort=" + q.sort + "&limit=" + std::to_string(q.limit);
    std::string url = base;
    for (int page=0; page<max_pages; ++page) {
        std::string body = get_(url);
        if (body.empty()) break;
        auto j = json::parse(body, nullptr, false);
        if (j.is_discarded()) break;
        if (j.contains("results")) {
            for (auto& r : j["results"]) {
                out.push_back({r.value("t", 0LL), r.value("o", 0.0), r.value("h", 0.0), r.value("l", 0.0), r.value("c", 0.0), r.value("v", 0.0)});
            }
        }
        if (!j.contains("next_url")) break;
        url = j["next_url"].get<std::string>();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return out;
}

void PolygonClient::write_csv(const std::string& out_path,const std::string& symbol,
                              const std::vector<AggBar>& bars, bool rth_only, bool exclude_holidays) {
  std::ofstream f(out_path);
  f << "timestamp,symbol,open,high,low,close,volume\n";
  for (auto& a: bars) {
    // **MODIFIED**: RTH and holiday filtering is now done directly on the UTC timestamp
    // before any string conversion, making it much more reliable.

    if (rth_only && !is_rth_nyc_from_utc_ms(a.ts_ms)) {
        continue;
    }
    
    if (exclude_holidays) {
        cctz::time_zone tz;
        if (cctz::load_time_zone("America/New_York", &tz)) {
            cctz::time_point<cctz::seconds> tp{cctz::seconds{a.ts_ms / 1000}};
            auto lt = cctz::convert(tp, tz);
            auto ct = cctz::civil_second(lt);
            
            if (is_us_market_holiday_2022_2025(ct.year(), ct.month(), ct.day())) {
                continue;
            }
        }
    }
    
    // The timestamp is converted to a NYT string only for writing to the CSV
    std::string ts_str = rfc3339_nyc_from_epoch_ms(a.ts_ms);

    f << ts_str << ',' << symbol << ','
      << a.open << ',' << a.high << ',' << a.low << ',' << a.close << ',' << a.volume << '\n';
  }
}

} // namespace sentio
