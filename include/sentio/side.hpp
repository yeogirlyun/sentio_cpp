#pragma once
#include <cstdint>
#include <string>

namespace sentio {

enum class PositionSide : int8_t { Flat=0, Long=1, Short=-1 };

inline std::string to_string(PositionSide s) {
    switch (s) { 
        case PositionSide::Flat: return "FLAT";
        case PositionSide::Long: return "LONG";
        case PositionSide::Short: return "SHORT"; 
    }
    return "UNKNOWN";
}

struct Qty { double shares{0.0}; };       // positive magnitude
struct Price { double px{0.0}; };         // last/avg price as needed

struct ExposureKey {
    std::string account;   // e.g., "alpaca:primary"
    std::string family;    // e.g., "QQQ*"  (see family mapper below)

    bool operator==(const ExposureKey& o) const {
        return account==o.account && family==o.family;
    }
};

struct ExposureKeyHash {
    size_t operator()(ExposureKey const& k) const noexcept {
        std::hash<std::string> h;
        return (h(k.account)*1315423911u) ^ h(k.family);
    }
};

} // namespace sentio
