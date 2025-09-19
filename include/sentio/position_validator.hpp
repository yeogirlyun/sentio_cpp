#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include <string>
#include <unordered_set>
#include <vector>

namespace sentio {

// **UPDATED**: Conflicting position detection - PSQ is inverse ETF, not short position
const std::unordered_set<std::string> LONG_ETFS = {"QQQ", "TQQQ"};
const std::unordered_set<std::string> INVERSE_ETFS = {"SQQQ", "PSQ"}; // PSQ restored as inverse ETF

inline bool has_conflicting_positions(const Portfolio& pf, const SymbolTable& ST) {
    bool has_long_etf = false;
    bool has_inverse_etf = false;
    bool has_short_qqq = false;
    
    for (size_t sid = 0; sid < pf.positions.size(); ++sid) {
        const auto& pos = pf.positions[sid];
        if (std::abs(pos.qty) > 1e-6) {
            const std::string& symbol = ST.get_symbol(sid);
            
            if (LONG_ETFS.count(symbol)) {
                if (pos.qty > 0) {
                    has_long_etf = true;
                } else {
                    // SHORT QQQ or SHORT TQQQ
                    has_short_qqq = true;
                }
            }
            if (INVERSE_ETFS.count(symbol)) {
                has_inverse_etf = true;
            }
        }
    }
    
    // **CONFLICT RULES**:
    // 1. Long ETF (QQQ+, TQQQ+) conflicts with Inverse ETF (SQQQ) or SHORT QQQ (QQQ-, TQQQ-)
    // 2. SHORT QQQ (QQQ-, TQQQ-) conflicts with Long ETF (QQQ+, TQQQ+)
    // 3. Inverse ETF (SQQQ) conflicts with Long ETF (QQQ+, TQQQ+)
    return (has_long_etf && (has_inverse_etf || has_short_qqq)) || 
           (has_short_qqq && has_long_etf);
}

inline std::string get_conflicting_symbols(const Portfolio& pf, const SymbolTable& ST) {
    std::vector<std::string> long_positions;
    std::vector<std::string> short_positions;
    std::vector<std::string> inverse_positions;
    
    for (size_t sid = 0; sid < pf.positions.size(); ++sid) {
        const auto& pos = pf.positions[sid];
        if (std::abs(pos.qty) > 1e-6) {
            const std::string& symbol = ST.get_symbol(sid);
            if (LONG_ETFS.count(symbol)) {
                if (pos.qty > 0) {
                    long_positions.push_back(symbol + "(+" + std::to_string((int)pos.qty) + ")");
                } else {
                    short_positions.push_back("SHORT " + symbol + "(" + std::to_string((int)pos.qty) + ")");
                }
            }
            if (INVERSE_ETFS.count(symbol)) {
                inverse_positions.push_back(symbol + "(" + std::to_string((int)pos.qty) + ")");
            }
        }
    }
    
    std::string result;
    if (!long_positions.empty()) {
        result += "Long: ";
        for (size_t i = 0; i < long_positions.size(); ++i) {
            if (i > 0) result += ", ";
            result += long_positions[i];
        }
    }
    if (!short_positions.empty()) {
        if (!result.empty()) result += "; ";
        result += "Short: ";
        for (size_t i = 0; i < short_positions.size(); ++i) {
            if (i > 0) result += ", ";
            result += short_positions[i];
        }
    }
    if (!inverse_positions.empty()) {
        if (!result.empty()) result += "; ";
        result += "Inverse: ";
        for (size_t i = 0; i < inverse_positions.size(); ++i) {
            if (i > 0) result += ", ";
            result += inverse_positions[i];
        }
    }
    return result;
}

} // namespace sentio
