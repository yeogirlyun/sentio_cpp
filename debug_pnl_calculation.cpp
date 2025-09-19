#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <sqlite3.h>
#include <cmath>

struct TradeData {
    std::string timestamp;
    std::string symbol;
    std::string side;
    double qty;
    double price;
    double pnl_delta;
};

std::vector<TradeData> extract_trades(const std::string& db_path, const std::string& run_id) {
    std::vector<TradeData> trades;
    
    sqlite3* db;
    int rc = sqlite3_open(db_path.c_str(), &db);
    if (rc != SQLITE_OK) {
        std::cerr << "Cannot open database: " << sqlite3_errmsg(db) << std::endl;
        return trades;
    }
    
    const char* sql = "SELECT ts_millis, symbol, side, qty, price, pnl_delta FROM audit_events WHERE run_id = ? AND kind = 'FILL' ORDER BY ts_millis ASC";
    sqlite3_stmt* stmt;
    
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "SQL prepare error: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return trades;
    }
    
    sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        TradeData trade;
        
        std::int64_t ts = sqlite3_column_int64(stmt, 0);
        std::time_t time_t = ts / 1000;
        char timestamp_str[32];
        std::strftime(timestamp_str, sizeof(timestamp_str), "%Y-%m-%d %H:%M:%S", std::gmtime(&time_t));
        trade.timestamp = timestamp_str;
        
        const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        const char* side = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        
        trade.symbol = symbol ? symbol : "";
        trade.side = side ? side : "";
        trade.qty = sqlite3_column_double(stmt, 3);
        trade.price = sqlite3_column_double(stmt, 4);
        trade.pnl_delta = sqlite3_column_double(stmt, 5);
        
        trades.push_back(trade);
    }
    
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    
    return trades;
}

int main() {
    std::cout << "=== P&L CALCULATION DEBUG ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    std::string db_path = "audit/sentio_audit.sqlite3";
    std::string run_id = "475482";
    
    std::vector<TradeData> trades = extract_trades(db_path, run_id);
    std::cout << "Total trades: " << trades.size() << std::endl;
    
    // Method 1: Cash flow tracking (what my canonical calculation did)
    double starting_cash = 100000.0;
    double running_cash = starting_cash;
    double total_cash_flow = 0.0;
    
    for (const auto& trade : trades) {
        double trade_value = trade.qty * trade.price;
        bool is_buy = (trade.side == "BUY");
        double cash_delta = is_buy ? -trade_value : trade_value;
        
        running_cash += cash_delta;
        total_cash_flow += cash_delta;
    }
    
    std::cout << "\n=== METHOD 1: CASH FLOW TRACKING ===" << std::endl;
    std::cout << "Starting Cash: $" << starting_cash << std::endl;
    std::cout << "Final Cash: $" << running_cash << std::endl;
    std::cout << "Total Cash Flow: $" << total_cash_flow << std::endl;
    std::cout << "Cash-based P&L: $" << running_cash - starting_cash << std::endl;
    std::cout << "Cash-based Return: " << ((running_cash - starting_cash) / starting_cash) * 100 << "%" << std::endl;
    
    // Method 2: P&L delta accumulation (what audit system does)
    double cumulative_pnl_delta = 0.0;
    for (const auto& trade : trades) {
        cumulative_pnl_delta += trade.pnl_delta;
    }
    
    std::cout << "\n=== METHOD 2: P&L DELTA ACCUMULATION ===" << std::endl;
    std::cout << "Cumulative P&L Delta: $" << cumulative_pnl_delta << std::endl;
    std::cout << "PnL-based Return: " << (cumulative_pnl_delta / starting_cash) * 100 << "%" << std::endl;
    
    // Method 3: Position tracking with mark-to-market (what position-history does)
    std::map<std::string, double> positions; // symbol -> quantity
    std::map<std::string, double> avg_prices; // symbol -> average price
    double realized_pnl = 0.0;
    double final_cash = starting_cash;
    
    for (const auto& trade : trades) {
        bool is_buy = (trade.side == "BUY");
        double trade_value = trade.qty * trade.price;
        
        // Update cash
        final_cash += is_buy ? -trade_value : trade_value;
        
        // Update positions
        std::string symbol = trade.symbol;
        double old_qty = positions[symbol];
        double new_qty = old_qty + (is_buy ? trade.qty : -trade.qty);
        
        if (std::abs(new_qty) < 1e-6) {
            // Position closed - calculate realized P&L
            if (std::abs(old_qty) > 1e-6) {
                double avg_price = avg_prices[symbol];
                double pnl = old_qty * (trade.price - avg_price);
                if (!is_buy) pnl = -pnl; // Adjust for sell
                realized_pnl += pnl;
            }
            positions.erase(symbol);
            avg_prices.erase(symbol);
        } else {
            if (old_qty * new_qty >= 0 && std::abs(old_qty) > 1e-6) {
                // Same direction - update VWAP
                avg_prices[symbol] = (avg_prices[symbol] * std::abs(old_qty) + trade.price * trade.qty) / std::abs(new_qty);
            } else {
                // New position or flipping direction
                avg_prices[symbol] = trade.price;
            }
            positions[symbol] = new_qty;
        }
    }
    
    // Calculate unrealized P&L for open positions
    double unrealized_pnl = 0.0;
    double total_position_value = 0.0;
    
    for (const auto& [symbol, qty] : positions) {
        if (std::abs(qty) > 1e-6) {
            double avg_price = avg_prices[symbol];
            // Use last trade price as current market price
            double current_price = avg_price;
            for (int i = trades.size() - 1; i >= 0; i--) {
                if (trades[i].symbol == symbol) {
                    current_price = trades[i].price;
                    break;
                }
            }
            
            double market_value = qty * current_price;
            double unrealized = qty * (current_price - avg_price);
            
            total_position_value += market_value;
            unrealized_pnl += unrealized;
        }
    }
    
    double final_equity = final_cash + total_position_value;
    double total_pnl_method3 = realized_pnl + unrealized_pnl;
    
    std::cout << "\n=== METHOD 3: POSITION TRACKING ===" << std::endl;
    std::cout << "Final Cash: $" << final_cash << std::endl;
    std::cout << "Position Value: $" << total_position_value << std::endl;
    std::cout << "Final Equity: $" << final_equity << std::endl;
    std::cout << "Realized P&L: $" << realized_pnl << std::endl;
    std::cout << "Unrealized P&L: $" << unrealized_pnl << std::endl;
    std::cout << "Total P&L: $" << total_pnl_method3 << std::endl;
    std::cout << "Position-based Return: " << ((final_equity - starting_cash) / starting_cash) * 100 << "%" << std::endl;
    std::cout << "Open Positions: " << positions.size() << std::endl;
    
    // Show open positions
    for (const auto& [symbol, qty] : positions) {
        if (std::abs(qty) > 1e-6) {
            std::cout << "  " << symbol << ": " << qty << " @ $" << avg_prices[symbol] << std::endl;
        }
    }
    
    // Compare all methods
    std::cout << "\n=== COMPARISON OF ALL METHODS ===" << std::endl;
    std::cout << "┌─────────────────────────┬─────────────────┬─────────────────┬─────────────────┐" << std::endl;
    std::cout << "│ Method                  │ Final Value     │ P&L             │ Return %        │" << std::endl;
    std::cout << "├─────────────────────────┼─────────────────┼─────────────────┼─────────────────┤" << std::endl;
    std::cout << "│ Cash Flow Tracking      │ $" << std::setw(14) << running_cash 
              << " │ $" << std::setw(14) << (running_cash - starting_cash)
              << " │ " << std::setw(14) << ((running_cash - starting_cash) / starting_cash) * 100 << "% │" << std::endl;
    std::cout << "│ P&L Delta Accumulation  │ $" << std::setw(14) << (starting_cash + cumulative_pnl_delta)
              << " │ $" << std::setw(14) << cumulative_pnl_delta
              << " │ " << std::setw(14) << (cumulative_pnl_delta / starting_cash) * 100 << "% │" << std::endl;
    std::cout << "│ Position Tracking       │ $" << std::setw(14) << final_equity
              << " │ $" << std::setw(14) << (final_equity - starting_cash)
              << " │ " << std::setw(14) << ((final_equity - starting_cash) / starting_cash) * 100 << "% │" << std::endl;
    std::cout << "└─────────────────────────┴─────────────────┴─────────────────┴─────────────────┘" << std::endl;
    
    // Expected values from our previous analysis
    std::cout << "\n=== COMPARISON WITH EXPECTED VALUES ===" << std::endl;
    std::cout << "Position-History reported: -20.42% return, $79,579.39 final equity" << std::endl;
    std::cout << "Audit Summarize reported: -20.33% return, P&L of -$20,333.37" << std::endl;
    
    // The correct method should be Position Tracking (Method 3)
    std::cout << "\n=== CONCLUSION ===" << std::endl;
    std::cout << "Method 3 (Position Tracking) is the correct approach." << std::endl;
    std::cout << "This matches the position-history calculation methodology." << std::endl;
    std::cout << "The discrepancy suggests different position tracking implementations." << std::endl;
    
    return 0;
}
