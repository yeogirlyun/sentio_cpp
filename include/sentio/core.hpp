#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath> // For std::sqrt

namespace sentio {

// Forward declarations
struct SymbolTable;

struct Bar {
    std::string ts_utc;
    int64_t ts_utc_epoch;
    double open, high, low, close;
    uint64_t volume;
};

// **MODIFIED**: This struct now holds a vector of Positions, indexed by symbol ID for performance.
struct Position { 
    double qty = 0.0; 
    double avg_price = 0.0; 
};

struct Portfolio {
    double cash = 100000.0;
    std::vector<Position> positions; // Indexed by symbol ID

    Portfolio() = default;
    explicit Portfolio(size_t num_symbols) : positions(num_symbols) {}
};

// **MODIFIED**: Vector-based functions are now the primary way to manage the portfolio.
inline void apply_fill(Portfolio& pf, int sid, double qty_delta, double price) {
    if (sid < 0 || static_cast<size_t>(sid) >= pf.positions.size()) {
        return; // Invalid symbol ID
    }
    
    pf.cash -= qty_delta * price;
    auto& pos = pf.positions[sid];
    
    double new_qty = pos.qty + qty_delta;
    if (std::abs(new_qty) < 1e-9) { // Position is closed
        pos.qty = 0.0;
        pos.avg_price = 0.0;
    } else {
        if (pos.qty * new_qty >= 0) { // Increasing position or opening a new one
            pos.avg_price = (pos.avg_price * pos.qty + price * qty_delta) / new_qty;
        }
        // If flipping from long to short or vice-versa, the new avg_price is just the fill price.
        else if (pos.qty * qty_delta < 0) {
             pos.avg_price = price;
        }
        pos.qty = new_qty;
    }
}

// Helper function to check if a position exists (non-zero quantity)
inline bool has_position(const Position& pos) {
    return std::abs(pos.qty) > 1e-9;
}

// Helper function to get position exposure (always positive)
inline double position_exposure(const Position& pos) {
    return std::abs(pos.qty);
}

inline double equity_mark_to_market(const Portfolio& pf, const std::vector<double>& last_prices) {
    double eq = pf.cash;
    for (size_t sid = 0; sid < pf.positions.size(); ++sid) {
        if (has_position(pf.positions[sid]) && sid < last_prices.size()) {
            eq += pf.positions[sid].qty * last_prices[sid];
        }
    }
    return eq;
}


// **REMOVED**: Old, simplistic Direction and StratSignal types are now deprecated.

} // namespace sentio