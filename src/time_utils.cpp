#include "sentio/time_utils.hpp"
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <algorithm>

#if __has_include(<chrono>)
  #include <chrono>
  using namespace std::chrono;
#else
  #error "Need C++20 <chrono>. If missing tzdb, you can still normalize UTC here and handle ET in calendar."
#endif

namespace sentio {

bool iso8601_looks_like(const std::string& s) {
  // Very light heuristic: "YYYY-MM-DDTHH:MM" and either 'Z' or +/-HH:MM
  return s.size() >= 16 && s[4]=='-' && s[7]=='-' && (s.find('T') != std::string::npos);
}

bool epoch_ms_suspected(double v_ms) {
  // If it's larger than ~1e12 it's probably ms; (1e12 sec is ~31k years)
  return std::isfinite(v_ms) && v_ms > 1.0e11;
}

static inline sys_seconds parse_iso8601_to_utc(const std::string& s) {
  // Minimal ISO8601 handling: require offset or 'Z'.
  // For robustness use Howard Hinnant's date::parse with %FT%T%Ez.
  // Here we support the common forms: 2022-09-06T13:30:00Z and 2022-09-06T09:30:00-04:00
  // We'll implement a tiny parser that splits offset and adjusts.
  auto posT = s.find('T');
  if (posT == std::string::npos) throw std::runtime_error("ISO8601 missing T");
  // Find offset start: last char 'Z' or last '+'/'-'
  int sign = 0;
  int oh=0, om=0;
  bool zulu = false;
  std::size_t offPos = s.rfind('Z');
  if (offPos != std::string::npos && offPos > posT) {
    zulu = true;
  } else {
    std::size_t plus = s.rfind('+');
    std::size_t minus= s.rfind('-');
    std::size_t off  = std::string::npos;
    if (plus!=std::string::npos && plus>posT) { off=plus; sign=+1; }
    else if (minus!=std::string::npos && minus>posT) { off=minus; sign=-1; }
    if (off==std::string::npos) throw std::runtime_error("ISO8601 missing offset/Z");
    // parse HH:MM
    if (off+3 >= s.size()) throw std::runtime_error("Bad offset");
    oh = std::stoi(s.substr(off+1,2));
    if (off+6 <= s.size() && s[off+3]==':') om = std::stoi(s.substr(off+4,2));
  }

  // parse date/time parts (seconds optional)
  int Y = std::stoi(s.substr(0,4));
  int M = std::stoi(s.substr(5,2));
  int D = std::stoi(s.substr(8,2));
  int h = std::stoi(s.substr(posT+1,2));
  int m = std::stoi(s.substr(posT+4,2));
  int sec = 0;
  if (posT+6 < s.size() && s[posT+6]==':') {
    sec = std::stoi(s.substr(posT+7,2));
  }

  // Treat parsed time as local-time-with-offset; compute UTC by subtracting offset
  using namespace std::chrono;
  sys_days sd = sys_days(std::chrono::year{Y}/M/D);
  seconds local = hours{h} + minutes{m} + seconds{sec};
  seconds off = seconds{ (oh*3600 + om*60) * (zulu ? 0 : sign) };
  // If sign=+1 (e.g., +09:00), local = UTC + offset => UTC = local - offset
  seconds utc_sec = local - off;
  return sys_seconds{sd.time_since_epoch() + utc_sec};
}

std::chrono::sys_seconds to_utc_sys_seconds(const std::variant<std::int64_t, double, std::string>& ts) {
  if (std::holds_alternative<std::int64_t>(ts)) {
    // epoch seconds
    return std::chrono::sys_seconds{std::chrono::seconds{std::get<std::int64_t>(ts)}};
  }
  if (std::holds_alternative<double>(ts)) {
    // Could be epoch ms or sec (float). Prefer ms detection and round down.
    double v = std::get<double>(ts);
    if (!std::isfinite(v)) throw std::runtime_error("Non-finite epoch");
    if (epoch_ms_suspected(v)) {
      auto s = static_cast<std::int64_t>(v / 1000.0);
      return std::chrono::sys_seconds{std::chrono::seconds{s}};
    } else {
      auto s = static_cast<std::int64_t>(v);
      return std::chrono::sys_seconds{std::chrono::seconds{s}};
    }
  }
  const std::string& s = std::get<std::string>(ts);
  if (!iso8601_looks_like(s)) {
    // fall back: try integer seconds in string
    std::int64_t v{};
    auto sv = std::string_view{s};
    if (auto [p, ec] = std::from_chars(sv.data(), sv.data()+sv.size(), v); ec == std::errc{}) {
      return std::chrono::sys_seconds{std::chrono::seconds{v}};
    }
    throw std::runtime_error("Unrecognized timestamp format: " + s);
  }
  return parse_iso8601_to_utc(s);
}

} // namespace sentio