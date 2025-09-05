#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include <string>
#include <vector>
#include <unordered_set>

namespace sentio {

enum class PortfolioState { Neutral, LongOnly, ShortOnly };
enum class RequiredAction { None, CloseLong, CloseShort };
enum class Direction { Long, Short }; // Keep for simple directional logic

const std::unordered_set<std::string> LONG_INSTRUMENTS = {"QQQ", "TQQQ", "TSLA"};
const std::unordered_set<std::string> SHORT_INSTRUMENTS = {"SQQQ", "PSQ", "TSLQ"};

class PositionManager {
private:
    PortfolioState state = PortfolioState::Neutral;
    int bars_since_flip = 0;
    [[maybe_unused]] const int cooldown_period = 5;
    
public:
    // **MODIFIED**: Logic restored to work with the new ID-based portfolio structure.
    void update_state(const Portfolio& portfolio, const SymbolTable& ST, const std::vector<double>& last_prices) {
        bars_since_flip++;
        double long_exposure = 0.0;
        double short_exposure = 0.0;

        for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
            const auto& pos = portfolio.positions[sid];
            if (std::abs(pos.qty) < 1e-6) continue;

            const std::string& symbol = ST.get_symbol(sid);
            double exposure = pos.qty * last_prices[sid];

            if (LONG_INSTRUMENTS.count(symbol)) {
                long_exposure += exposure;
            }
            if (SHORT_INSTRUMENTS.count(symbol)) {
                short_exposure += exposure; // Will be negative for short positions
            }
        }
        
        PortfolioState old_state = state;
        
        if (long_exposure > 100 && std::abs(short_exposure) < 100) {
            state = PortfolioState::LongOnly;
        } else if (std::abs(short_exposure) > 100 && long_exposure < 100) {
            state = PortfolioState::ShortOnly;
        } else {
            state = PortfolioState::Neutral;
        }

        if (state != old_state) {
            bars_since_flip = 0;
        }
    }
    
    // ... other methods remain the same ...
};

} // namespace sentio