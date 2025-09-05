#pragma once
#include <chrono>
#include <string>
#include <variant>

namespace sentio {

// Normalize various timestamp representations to UTC epoch seconds.
std::chrono::sys_seconds to_utc_sys_seconds(const std::variant<std::int64_t, double, std::string>& ts);

// Helpers exposed for tests
bool iso8601_looks_like(const std::string& s);
bool epoch_ms_suspected(double v_ms);

} // namespace sentio