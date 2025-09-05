#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include "core.hpp"
#include "symbol_table.hpp"

namespace sentio {

// **MODIFIED**: The Pricebook ensures that all symbol prices are correctly aligned in time.
// It works by advancing an index for each symbol's data series until its timestamp
// matches or just passes the timestamp of the base symbol's current bar.
// This is both fast (amortized O(1)) and correct, handling missing data gracefully.
struct Pricebook {
    const int base_id;
    const SymbolTable& ST;
    const std::vector<std::vector<Bar>>& S; // Reference to all series data
    std::vector<int> idx;                   // Rolling index for each symbol's series
    std::vector<double> last_px;            // Stores the last known price for each symbol ID

    Pricebook(int base, const SymbolTable& st, const std::vector<std::vector<Bar>>& series)
      : base_id(base), ST(st), S(series), idx(S.size(), 0), last_px(S.size(), 0.0) {}

    // Advances the index 'j' for a given series 'V' to the bar corresponding to 'base_ts'
    inline void advance_to_ts(const std::vector<Bar>& V, int& j, int64_t base_ts) {
        const int n = (int)V.size();
        // Move index forward as long as the *next* bar is still at or before the target time
        while (j + 1 < n && V[j + 1].ts_nyt_epoch <= base_ts) {
            ++j;
        }
    }

    // Syncs all symbol prices to the timestamp of the i-th bar of the base symbol
    inline void sync_to_base_i(int i) {
        if (S[base_id].empty()) return;
        const int64_t ts = S[base_id][i].ts_nyt_epoch;
        
        for (int sid = 0; sid < (int)S.size(); ++sid) {
            if (!S[sid].empty()) {
                advance_to_ts(S[sid], idx[sid], ts);
                last_px[sid] = S[sid][idx[sid]].close;
            }
        }
    }

    // **NEW**: Helper to get the last prices as a map for components needing string keys
    inline std::unordered_map<std::string, double> last_px_map() const {
        std::unordered_map<std::string, double> price_map;
        for (int sid = 0; sid < (int)last_px.size(); ++sid) {
            if (last_px[sid] > 0.0) {
                price_map[ST.get_symbol(sid)] = last_px[sid];
            }
        }
        return price_map;
    }
};

} // namespace sentio