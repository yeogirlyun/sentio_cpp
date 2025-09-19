#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sqlite3.h>
#include <cmath>

// Simple structure to hold trade data
struct TradeData {
    std::string timestamp;
    std::string symbol;
    std::string side;
    double qty;
    double price;
    double pnl_delta;
};

// Simple structure to hold run info
struct RunInfo {
    std::string run_id;
    std::string strategy;
    int trading_days;
    double starting_capital = 100000.0;
};

// Extract trade data from audit database
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
        
        // Convert timestamp to string
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

// Get run information
RunInfo get_run_info(const std::string& db_path, const std::string& run_id) {
    RunInfo info;
    info.run_id = run_id;
    
    sqlite3* db;
    int rc = sqlite3_open(db_path.c_str(), &db);
    if (rc != SQLITE_OK) {
        return info;
    }
    
    const char* sql = "SELECT strategy, run_trading_days FROM audit_runs WHERE run_id = ?";
    sqlite3_stmt* stmt;
    
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        sqlite3_close(db);
        return info;
    }
    
    sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* strategy = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        info.strategy = strategy ? strategy : "";
        info.trading_days = sqlite3_column_int(stmt, 1);
    }
    
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    
    return info;
}

// Build equity curve from trades
std::vector<std::pair<std::string, double>> build_equity_curve(const std::vector<TradeData>& trades, double starting_capital) {
    std::vector<std::pair<std::string, double>> equity_curve;
    
    double running_cash = starting_capital;
    double running_equity = starting_capital;
    
    // Add starting point
    if (!trades.empty()) {
        equity_curve.push_back({trades[0].timestamp.substr(0, 10) + " 00:00:00", starting_capital});
    }
    
    for (const auto& trade : trades) {
        // Update cash based on trade
        double trade_value = trade.qty * trade.price;
        bool is_buy = (trade.side == "BUY");
        double cash_delta = is_buy ? -trade_value : trade_value;
        
        running_cash += cash_delta;
        
        // For simplicity, assume no open positions (all trades are closed immediately)
        // In reality, we'd need to track positions and mark-to-market
        running_equity = running_cash;
        
        equity_curve.push_back({trade.timestamp, running_equity});
    }
    
    return equity_curve;
}

// Calculate canonical metrics
struct CanonicalMetrics {
    double total_return_pct;
    double monthly_projected_return_pct;
    double daily_trades;
    int total_trades;
    int trading_days;
    double final_equity;
    double total_pnl;
};

CanonicalMetrics calculate_canonical_metrics(const std::vector<TradeData>& trades, const RunInfo& info) {
    CanonicalMetrics metrics = {};
    
    metrics.total_trades = static_cast<int>(trades.size());
    metrics.trading_days = info.trading_days;
    
    if (trades.empty()) {
        return metrics;
    }
    
    // Build simplified equity curve
    auto equity_curve = build_equity_curve(trades, info.starting_capital);
    
    metrics.final_equity = equity_curve.back().second;
    metrics.total_pnl = metrics.final_equity - info.starting_capital;
    metrics.total_return_pct = (metrics.total_pnl / info.starting_capital) * 100.0;
    
    // Calculate MPR from total return and trading days
    if (metrics.trading_days > 0) {
        double daily_growth_rate = std::pow(1.0 + (metrics.total_pnl / info.starting_capital), 1.0 / metrics.trading_days) - 1.0;
        double monthly_return = std::pow(1.0 + daily_growth_rate, 21.0) - 1.0; // 21 trading days per month
        metrics.monthly_projected_return_pct = monthly_return * 100.0;
    }
    
    // Calculate daily trades
    metrics.daily_trades = (metrics.trading_days > 0) ? 
        static_cast<double>(metrics.total_trades) / metrics.trading_days : 0.0;
    
    return metrics;
}

int main() {
    std::cout << "=== REAL-WORLD METRICS VALIDATION ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    // Configuration
    std::string db_path = "audit/sentio_audit.sqlite3";
    std::string run_id = "475482"; // sigor test run
    
    // Extract data from audit database
    std::cout << "Extracting data from audit database..." << std::endl;
    RunInfo run_info = get_run_info(db_path, run_id);
    std::vector<TradeData> trades = extract_trades(db_path, run_id);
    
    std::cout << "Run ID: " << run_info.run_id << std::endl;
    std::cout << "Strategy: " << run_info.strategy << std::endl;
    std::cout << "Trading Days: " << run_info.trading_days << std::endl;
    std::cout << "Total Trades: " << trades.size() << std::endl;
    
    // Calculate canonical metrics from raw trade data
    CanonicalMetrics canonical = calculate_canonical_metrics(trades, run_info);
    
    std::cout << "\n=== CANONICAL METRICS (from raw trades) ===" << std::endl;
    std::cout << "Final Equity: $" << canonical.final_equity << std::endl;
    std::cout << "Total P&L: $" << canonical.total_pnl << std::endl;
    std::cout << "Total Return: " << canonical.total_return_pct << "%" << std::endl;
    std::cout << "Monthly Projected Return: " << canonical.monthly_projected_return_pct << "%" << std::endl;
    std::cout << "Daily Trades: " << canonical.daily_trades << std::endl;
    
    // Compare with known values from our previous analysis
    std::cout << "\n=== COMPARISON WITH PREVIOUS ANALYSIS ===" << std::endl;
    std::cout << "┌─────────────────────────┬─────────────────┬─────────────────┬─────────────────┐" << std::endl;
    std::cout << "│ Metric                  │ Canonical (New) │ Position-History│ Audit Summarize │" << std::endl;
    std::cout << "├─────────────────────────┼─────────────────┼─────────────────┼─────────────────┤" << std::endl;
    std::cout << "│ Total Return            │ " << std::setw(14) << canonical.total_return_pct << "% │ " 
              << std::setw(14) << "-20.42" << "% │ " 
              << std::setw(14) << "-20.33" << "% │" << std::endl;
    std::cout << "│ Monthly Proj. Return    │ " << std::setw(14) << canonical.monthly_projected_return_pct << "% │ " 
              << std::setw(14) << "-11.04" << "% │ " 
              << std::setw(14) << "-10.99" << "% │" << std::endl;
    std::cout << "│ Daily Trades            │ " << std::setw(14) << canonical.daily_trades << "  │ " 
              << std::setw(14) << "124.51" << "  │ " 
              << std::setw(14) << "124.5" << "  │" << std::endl;
    std::cout << "└─────────────────────────┴─────────────────┴─────────────────┴─────────────────┘" << std::endl;
    
    // Calculate discrepancies
    double return_disc_ph = std::abs(canonical.total_return_pct - (-20.42));
    double return_disc_audit = std::abs(canonical.total_return_pct - (-20.33));
    double mpr_disc_ph = std::abs(canonical.monthly_projected_return_pct - (-11.04));
    double mpr_disc_audit = std::abs(canonical.monthly_projected_return_pct - (-10.99));
    
    std::cout << "\n=== DISCREPANCY ANALYSIS ===" << std::endl;
    std::cout << "Total Return discrepancy (vs Position-History): " << return_disc_ph << "%" << std::endl;
    std::cout << "Total Return discrepancy (vs Audit): " << return_disc_audit << "%" << std::endl;
    std::cout << "MPR discrepancy (vs Position-History): " << mpr_disc_ph << "%" << std::endl;
    std::cout << "MPR discrepancy (vs Audit): " << mpr_disc_audit << "%" << std::endl;
    
    // Validation results
    std::cout << "\n=== VALIDATION RESULTS ===" << std::endl;
    bool return_consistent = (return_disc_audit < 0.5 && return_disc_ph < 0.5);
    bool mpr_consistent = (mpr_disc_audit < 1.0 && mpr_disc_ph < 1.0);
    
    std::cout << "Total Return Consistency: " << (return_consistent ? "✅ PASS" : "❌ FAIL") << std::endl;
    std::cout << "MPR Consistency: " << (mpr_consistent ? "✅ PASS" : "❌ FAIL") << std::endl;
    
    if (return_consistent && mpr_consistent) {
        std::cout << "\n🎉 SUCCESS: All three methods show consistent results!" << std::endl;
        std::cout << "The audit system is working correctly for dataset traceability." << std::endl;
    } else {
        std::cout << "\n⚠️  ISSUES DETECTED: Further investigation needed." << std::endl;
    }
    
    return 0;
}
