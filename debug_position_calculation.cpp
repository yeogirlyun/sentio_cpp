#include <iostream>
#include <map>
#include <string>
#include <cmath>

int main() {
    // Simulate the position reconstruction logic
    std::map<std::string, double> final_positions;
    std::map<std::string, double> avg_prices;
    
    // Sample trades from the end of the run (in reverse order to simulate chronological)
    struct Trade {
        std::string symbol;
        std::string side;
        double qty;
        double price;
    };
    
    // Last 10 trades in chronological order
    Trade trades[] = {
        {"QQQ", "BUY", 0.0441616034933592, 412.0},
        {"QQQ", "SELL", 19.2405404348433, 412.0},
        {"QQQ", "BUY", 19.2405133519147, 412.0},
        {"QQQ", "SELL", 19.2627228407081, 411.0},
        {"QQQ", "BUY", 19.2849322974528, 411.0},
        {"QQQ", "SELL", 0.066379202620837, 414.0},
        {"QQQ", "SELL", 19.2185501117295, 411.0},
        {"QQQ", "BUY", 19.1632368080129, 416.0},
        {"QQQ", "SELL", 19.151553285137, 411.0},
        {"QQQ", "BUY", 0.0489969628574674, 413.0}
    };
    
    std::cout << "=== POSITION RECONSTRUCTION DEBUG ===" << std::endl;
    
    for (int i = 0; i < 10; ++i) {
        const Trade& trade = trades[i];
        
        double pos_delta = (trade.side == "BUY") ? trade.qty : -trade.qty;
        double current_qty = final_positions[trade.symbol];
        double new_qty = current_qty + pos_delta;
        
        std::cout << "Trade " << (i+1) << ": " << trade.side << " " << trade.qty << " @ " << trade.price;
        std::cout << " | Pos Delta: " << pos_delta << " | Current: " << current_qty << " | New: " << new_qty;
        
        if (std::abs(new_qty) < 1e-9) { // Position closed
            std::cout << " | CLOSED";
            final_positions.erase(trade.symbol);
            avg_prices.erase(trade.symbol);
        } else {
            // Update average price (VWAP)
            if (current_qty * pos_delta >= 0) { // Increasing position
                avg_prices[trade.symbol] = (avg_prices[trade.symbol] * current_qty + trade.price * pos_delta) / new_qty;
            } else { // Reducing or flipping position
                avg_prices[trade.symbol] = trade.price; // New cost basis is the flip price
            }
            final_positions[trade.symbol] = new_qty;
            std::cout << " | Avg Price: " << avg_prices[trade.symbol];
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n=== FINAL POSITIONS ===" << std::endl;
    for (const auto& [symbol, qty] : final_positions) {
        std::cout << symbol << ": " << qty << " shares @ avg " << avg_prices[symbol] << std::endl;
    }
    
    return 0;
}
