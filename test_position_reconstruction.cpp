#include <iostream>
#include <map>
#include <string>
#include <cmath>
#include <sqlite3.h>

int main() {
    sqlite3* db;
    int rc = sqlite3_open("audit/sentio_audit.sqlite3", &db);
    if (rc) {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return 1;
    }
    
    std::string run_id = "350169";
    
    // 1. Reconstruct final positions by replaying all fills for the run
    std::map<std::string, double> final_positions; // symbol -> quantity
    std::map<std::string, double> avg_prices;      // symbol -> avg_entry_price
    sqlite3_stmt* fill_st = nullptr;
    const char* fill_sql = "SELECT symbol, side, qty, price FROM audit_events WHERE run_id=? AND kind='FILL' ORDER BY ts_millis ASC";
    sqlite3_prepare_v2(db, fill_sql, -1, &fill_st, nullptr);
    sqlite3_bind_text(fill_st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);

    int fill_count = 0;
    while (sqlite3_step(fill_st) == SQLITE_ROW) {
        const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text(fill_st, 0));
        const char* side = reinterpret_cast<const char*>(sqlite3_column_text(fill_st, 1));
        double qty = sqlite3_column_double(fill_st, 2);
        double price = sqlite3_column_double(fill_st, 3);
        
        if (symbol && side) {
            double pos_delta = (strcmp(side, "BUY") == 0) ? qty : -qty;
            double current_qty = final_positions[symbol];
            double new_qty = current_qty + pos_delta;

            if (std::abs(new_qty) < 1e-9) { // Position closed
                final_positions.erase(symbol);
                avg_prices.erase(symbol);
            } else {
                // Update average price (VWAP)
                if (current_qty * pos_delta >= 0) { // Increasing position
                    avg_prices[symbol] = (avg_prices[symbol] * current_qty + price * pos_delta) / new_qty;
                } else { // Reducing or flipping position
                    avg_prices[symbol] = price; // New cost basis is the flip price
                }
                final_positions[symbol] = new_qty;
            }
            
            fill_count++;
            if (fill_count % 1000 == 0) {
                std::cout << "Processed " << fill_count << " fills, current QQQ position: " << final_positions["QQQ"] << std::endl;
            }
        }
    }
    sqlite3_finalize(fill_st);
    
    std::cout << "\n=== FINAL POSITIONS AFTER ALL FILLS ===" << std::endl;
    for (const auto& [symbol, qty] : final_positions) {
        std::cout << symbol << ": " << qty << " shares @ avg " << avg_prices[symbol] << std::endl;
    }
    
    // 2. Get the last known price for each open position
    std::map<std::string, double> last_prices;
    for (auto const& [symbol, qty] : final_positions) {
        sqlite3_stmt* price_st = nullptr;
        const char* price_sql = "SELECT price FROM audit_events WHERE run_id=? AND symbol=? AND kind IN ('FILL', 'BAR') ORDER BY ts_millis DESC LIMIT 1";
        sqlite3_prepare_v2(db, price_sql, -1, &price_st, nullptr);
        sqlite3_bind_text(price_st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(price_st, 2, symbol.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(price_st) == SQLITE_ROW) {
            last_prices[symbol] = sqlite3_column_double(price_st, 0);
            std::cout << "Last price for " << symbol << ": " << last_prices[symbol] << std::endl;
        }
        sqlite3_finalize(price_st);
    }

    // 3. Calculate total unrealized P&L and add it to the realized P&L
    double unrealized_pnl_sum = 0.0;
    for (auto const& [symbol, qty] : final_positions) {
        if (last_prices.count(symbol)) {
            double unrealized_pnl = (last_prices[symbol] - avg_prices[symbol]) * qty;
            unrealized_pnl_sum += unrealized_pnl;
            std::cout << symbol << " unrealized P&L: (" << last_prices[symbol] << " - " << avg_prices[symbol] << ") * " << qty << " = " << unrealized_pnl << std::endl;
        }
    }
    
    std::cout << "\nTotal unrealized P&L: " << unrealized_pnl_sum << std::endl;
    
    sqlite3_close(db);
    return 0;
}
