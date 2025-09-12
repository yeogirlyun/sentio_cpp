#pragma once
#include <string>
#include <cstdint>

namespace sentio {

enum class Side { Buy, Sell, Neutral };

struct Signal {
    std::string  symbol;  // e.g., "QQQ", "SQQQ"
    Side         side;    // Buy/Sell/Neutral
    double       weight;  // [-1, +1]
    std::int64_t ts;      // epoch millis or bar index
};

} // namespace sentio