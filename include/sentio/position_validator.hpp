#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include <string>
#include <unordered_set>
#include <vector>

namespace sentio {

// **NEW**: Conflicting position detection
const std::unordered_set<std::string> LONG_ETFS = {"QQQ", "TQQQ"};
const std::unordered_set<std::string> INVERSE_ETFS = {"PSQ", "SQQQ"};

inline bool has_conflicting_positions(const Portfolio& pf, const SymbolTable& ST) {
    bool has_long = false;
    bool has_inverse = false;
    
    for (size_t sid = 0; sid < pf.positions.size(); ++sid) {
        if (std::abs(pf.positions[sid].qty) > 1e-6) {
            const std::string& symbol = ST.get_symbol(sid);
            if (LONG_ETFS.count(symbol)) {
                has_long = true;
            }
            if (INVERSE_ETFS.count(symbol)) {
                has_inverse = true;
            }
        }
    }
    
    // **FIXED**: Only conflict if we have BOTH long AND inverse positions
    // TQQQ + QQQ is allowed (both long), PSQ + SQQQ is allowed (both inverse)
    return has_long && has_inverse;
}

inline std::string get_conflicting_symbols(const Portfolio& pf, const SymbolTable& ST) {
    std::vector<std::string> long_positions;
    std::vector<std::string> inverse_positions;
    
    for (size_t sid = 0; sid < pf.positions.size(); ++sid) {
        if (std::abs(pf.positions[sid].qty) > 1e-6) {
            const std::string& symbol = ST.get_symbol(sid);
            if (LONG_ETFS.count(symbol)) {
                long_positions.push_back(symbol);
            }
            if (INVERSE_ETFS.count(symbol)) {
                inverse_positions.push_back(symbol);
            }
        }
    }
    
    std::string result;
    if (!long_positions.empty()) {
        result += "Long: " + std::to_string(long_positions.size()) + " positions";
    }
    if (!inverse_positions.empty()) {
        if (!result.empty()) result += ", ";
        result += "Inverse: " + std::to_string(inverse_positions.size()) + " positions";
    }
    return result;
}

} // namespace sentio
