#include "sentio/replay.hpp"
#include "sentio/core.hpp"
#include <sqlite3.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <climits>

namespace sentio {

bool replay_and_assert(Auditor& au, long long run_id, double eps) {
  // Replay portfolio state from audit logs and verify against snapshots
  
  // 1. Load all fills for this run ordered by timestamp
  std::vector<std::tuple<long long, std::string, double, double, double, double>> fills; // ts_ms, symbol, qty, price, fees, slippage_bp
  
  sqlite3_stmt* stmt = nullptr;
  const char* sql = R"(
    SELECT f.ts_utc, f.symbol, f.qty, f.price, f.fees, f.slippage_bp
    FROM fills f 
    WHERE f.run_id = ? 
    ORDER BY f.ts_utc
  )";
  
  if (sqlite3_prepare_v2(au.db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    std::cerr << "Failed to prepare fills query: " << sqlite3_errmsg(au.db) << std::endl;
    return false;
  }
  
  sqlite3_bind_int64(stmt, 1, run_id);
  
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const char* ts_utc_cstr = (const char*)sqlite3_column_text(stmt, 0);
    const char* symbol_cstr = (const char*)sqlite3_column_text(stmt, 1);
    std::string ts_utc = ts_utc_cstr ? ts_utc_cstr : "";
    std::string symbol = symbol_cstr ? symbol_cstr : "";
    double qty = sqlite3_column_double(stmt, 2);
    double price = sqlite3_column_double(stmt, 3);
    double fees = sqlite3_column_double(stmt, 4);
    double slippage_bp = sqlite3_column_double(stmt, 5);
    
    if (!symbol.empty() && !ts_utc.empty()) {
      // Use a simple counter for ordering since we don't have ts_ms
      static long long counter = 0;
      fills.emplace_back(counter++, symbol, qty, price, fees, slippage_bp);
    }
  }
  sqlite3_finalize(stmt);
  
  // 2. Load snapshots for verification
  std::vector<std::tuple<long long, double, double>> snapshots; // ts_ms, cash, equity
  
  sql = "SELECT ts_utc, cash, equity FROM snapshots WHERE run_id = ? ORDER BY ts_utc";
  if (sqlite3_prepare_v2(au.db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    std::cerr << "Failed to prepare snapshots query: " << sqlite3_errmsg(au.db) << std::endl;
    return false;
  }
  
  sqlite3_bind_int64(stmt, 1, run_id);
  
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    // Use counter for ordering since we're using ts_utc
    static long long snap_counter = 0;
    double cash = sqlite3_column_double(stmt, 1);
    double equity = sqlite3_column_double(stmt, 2);
    snapshots.emplace_back(snap_counter++, cash, equity);
  }
  sqlite3_finalize(stmt);
  
  // 3. Replay portfolio state
  Portfolio replay_pf{}; // Start with default initial cash
  std::unordered_map<std::string, double> last_prices;
  
  size_t fill_idx = 0;
  size_t snap_idx = 0;
  bool verification_passed = true;
  
  std::cout << "Replaying " << fills.size() << " fills and verifying " 
            << snapshots.size() << " snapshots for run " << run_id << std::endl;
  
  // Process fills and check snapshots
  while (fill_idx < fills.size() || snap_idx < snapshots.size()) {
    long long next_fill_ts = (fill_idx < fills.size()) ? std::get<0>(fills[fill_idx]) : LLONG_MAX;
    long long next_snap_ts = (snap_idx < snapshots.size()) ? std::get<0>(snapshots[snap_idx]) : LLONG_MAX;
    
    if (next_fill_ts <= next_snap_ts) {
      // Process fill with fees and slippage (matching original run)
      auto [ts_ms, symbol, qty, price, fees, slippage_bp] = fills[fill_idx];
      // Convert symbol string to symbol ID (assuming 0 for QQQ, 1 for TQQQ, 2 for SQQQ)
      int symbol_id = 0;
      if (symbol == "TQQQ") symbol_id = 1;
      else if (symbol == "SQQQ") symbol_id = 2;
      apply_fill(replay_pf, symbol_id, qty, price);
      replay_pf.cash -= fees;  // Apply transaction fees
      last_prices[symbol] = price;
      fill_idx++;
      
      std::cout << "Fill: " << symbol << " qty=" << qty << " price=" << price 
                << " fees=" << fees << " cash=" << replay_pf.cash << std::endl;
    } else {
      // Verify snapshot
      auto [ts_ms, expected_cash, expected_equity] = snapshots[snap_idx];
      double actual_cash = replay_pf.cash;
      // Convert last_prices map to vector
      std::vector<double> last_prices_vec(3, 0.0);
      for (const auto& [sym, price] : last_prices) {
        int symbol_id = 0;
        if (sym == "TQQQ") symbol_id = 1;
        else if (sym == "SQQQ") symbol_id = 2;
        if (static_cast<size_t>(symbol_id) < last_prices_vec.size()) {
          last_prices_vec[symbol_id] = price;
        }
      }
      double actual_equity = equity_mark_to_market(replay_pf, last_prices_vec);
      
      double cash_diff = std::abs(actual_cash - expected_cash);
      double equity_diff = std::abs(actual_equity - expected_equity);
      
      std::cout << "Snapshot verification at ts=" << ts_ms << ":" << std::endl;
      std::cout << "  Cash: expected=" << expected_cash << " actual=" << actual_cash 
                << " diff=" << cash_diff << std::endl;
      std::cout << "  Equity: expected=" << expected_equity << " actual=" << actual_equity 
                << " diff=" << equity_diff << std::endl;
      
      if (cash_diff > eps || equity_diff > eps) {
        std::cerr << "VERIFICATION FAILED: Differences exceed tolerance " << eps << std::endl;
        verification_passed = false;
      }
      
      snap_idx++;
    }
  }
  
  // 4. Final verification summary
  std::cout << "Replay completed. Final portfolio state:" << std::endl;
  std::cout << "  Cash: $" << replay_pf.cash << std::endl;
  std::cout << "  Positions:" << std::endl;
  for (size_t sym = 0; sym < replay_pf.positions.size(); ++sym) {
    const auto& qty = replay_pf.positions[sym].qty;
    if (std::abs(qty) > 1e-6) {
      std::cout << "    " << sym << ": " << qty << " shares" << std::endl;
    }
  }
  // Convert last_prices map to vector
  std::vector<double> last_prices_vec(3, 0.0);
  for (const auto& [sym, price] : last_prices) {
    int symbol_id = 0;
    if (sym == "TQQQ") symbol_id = 1;
    else if (sym == "SQQQ") symbol_id = 2;
    if (static_cast<size_t>(symbol_id) < last_prices_vec.size()) {
      last_prices_vec[symbol_id] = price;
    }
  }
  std::cout << "  Total Equity: $" << equity_mark_to_market(replay_pf, last_prices_vec) << std::endl;
  
  if (verification_passed) {
    std::cout << "✅ REPLAY VERIFICATION PASSED" << std::endl;
  } else {
    std::cout << "❌ REPLAY VERIFICATION FAILED" << std::endl;
  }
  
  return verification_passed;
}

bool replay_and_assert_with_data(Auditor& au, long long run_id, 
                                 const std::unordered_map<std::string, std::vector<Bar>>& market_data,
                                 double eps) {
  // Enhanced replay that uses market data for accurate final pricing
  
  // 1. Load all fills for this run ordered by timestamp
  std::vector<std::tuple<std::string, std::string, double, double, double, double>> fills; // ts_utc, symbol, qty, price, fees, slippage_bp
  
  sqlite3_stmt* stmt = nullptr;
  const char* sql = R"(
    SELECT f.ts_utc, f.symbol, f.qty, f.price, f.fees, f.slippage_bp
    FROM fills f 
    WHERE f.run_id = ? 
    ORDER BY f.ts_utc
  )";
  
  if (sqlite3_prepare_v2(au.db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    std::cerr << "Failed to prepare fills query: " << sqlite3_errmsg(au.db) << std::endl;
    return false;
  }
  
  sqlite3_bind_int64(stmt, 1, run_id);
  
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const char* ts_utc_cstr = (const char*)sqlite3_column_text(stmt, 0);
    const char* symbol_cstr = (const char*)sqlite3_column_text(stmt, 1);
    std::string ts_utc = ts_utc_cstr ? ts_utc_cstr : "";
    std::string symbol = symbol_cstr ? symbol_cstr : "";
    double qty = sqlite3_column_double(stmt, 2);
    double price = sqlite3_column_double(stmt, 3);
    double fees = sqlite3_column_double(stmt, 4);
    double slippage_bp = sqlite3_column_double(stmt, 5);
    
    if (!symbol.empty() && !ts_utc.empty()) {
      fills.emplace_back(ts_utc, symbol, qty, price, fees, slippage_bp);
    }
  }
  sqlite3_finalize(stmt);
  
  // 2. Load snapshots for verification
  std::vector<std::tuple<std::string, double, double>> snapshots; // ts_utc, cash, equity
  
  sql = "SELECT ts_utc, cash, equity FROM snapshots WHERE run_id = ? ORDER BY ts_utc";
  if (sqlite3_prepare_v2(au.db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    std::cerr << "Failed to prepare snapshots query: " << sqlite3_errmsg(au.db) << std::endl;
    return false;
  }
  
  sqlite3_bind_int64(stmt, 1, run_id);
  
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const char* ts_utc_cstr = (const char*)sqlite3_column_text(stmt, 0);
    std::string ts_utc = ts_utc_cstr ? ts_utc_cstr : "";
    double cash = sqlite3_column_double(stmt, 1);
    double equity = sqlite3_column_double(stmt, 2);
    snapshots.emplace_back(ts_utc, cash, equity);
  }
  sqlite3_finalize(stmt);
  
  if (snapshots.empty()) {
    std::cerr << "No snapshots found for run " << run_id << std::endl;
    return false;
  }
  
  // 3. Replay portfolio state
  Portfolio replay_pf{}; // Start with default initial cash
  std::unordered_map<std::string, double> current_prices;
  
  std::cout << "Enhanced replay: processing " << fills.size() << " fills and verifying " 
            << snapshots.size() << " snapshots for run " << run_id << std::endl;
  
  // Process all fills chronologically
  for (const auto& [ts_utc, symbol, qty, price, fees, slippage_bp] : fills) {
    // Convert symbol string to symbol ID (assuming 0 for QQQ, 1 for TQQQ, 2 for SQQQ)
    int symbol_id = 0;
    if (symbol == "TQQQ") symbol_id = 1;
    else if (symbol == "SQQQ") symbol_id = 2;
    apply_fill(replay_pf, symbol_id, qty, price);
    replay_pf.cash -= fees;  // Apply transaction fees
    current_prices[symbol] = price;  // Update price as of this fill
    
    std::cout << "Fill: " << symbol << " qty=" << qty << " price=" << price 
              << " fees=" << fees << " cash=" << replay_pf.cash << std::endl;
  }
  
  // 4. Get final prices from market data at the last snapshot timestamp
  const std::string& final_timestamp = std::get<0>(snapshots.back());
  std::unordered_map<std::string, double> final_prices;
  
  std::cout << "\nGetting final prices at timestamp: " << final_timestamp << std::endl;
  
  for (const auto& [symbol, bars] : market_data) {
    // Find the bar at or before the final timestamp
    double final_price = 0.0;
    for (int i = (int)bars.size() - 1; i >= 0; --i) {
      if (bars[i].ts_utc <= final_timestamp) {
        final_price = bars[i].close;
        break;
      }
    }
    
    if (final_price > 0.0) {
      final_prices[symbol] = final_price;
      std::cout << "Final price for " << symbol << ": " << final_price << std::endl;
    }
  }
  
  // 5. Calculate final equity using correct final prices
  double final_cash = replay_pf.cash;
  // Convert final_prices map to vector
  std::vector<double> final_prices_vec(3, 0.0);
  for (const auto& [sym, price] : final_prices) {
    int symbol_id = 0;
    if (sym == "TQQQ") symbol_id = 1;
    else if (sym == "SQQQ") symbol_id = 2;
    if (static_cast<size_t>(symbol_id) < final_prices_vec.size()) {
      final_prices_vec[symbol_id] = price;
    }
  }
  double final_equity = equity_mark_to_market(replay_pf, final_prices_vec);
  
  std::cout << "\nFinal portfolio state with correct pricing:" << std::endl;
  std::cout << "  Cash: $" << final_cash << std::endl;
  std::cout << "  Positions:" << std::endl;
  for (size_t sym = 0; sym < replay_pf.positions.size(); ++sym) {
    const auto& qty = replay_pf.positions[sym].qty;
    if (std::abs(qty) > 1e-6) {
      double price = (sym < final_prices_vec.size()) ? final_prices_vec[sym] : 0.0;
      double position_value = qty * price;
      std::cout << "    " << sym << ": " << qty << " shares @ $" << price 
                << " = $" << position_value << std::endl;
    }
  }
  std::cout << "  Total Equity: $" << final_equity << std::endl;
  
  // 6. Verify against final snapshot
  const auto& [expected_ts, expected_cash, expected_equity] = snapshots.back();
  
  double cash_diff = std::abs(final_cash - expected_cash);
  double equity_diff = std::abs(final_equity - expected_equity);
  
  std::cout << "\nFinal verification:" << std::endl;
  std::cout << "  Expected Cash: $" << expected_cash << " | Actual: $" << final_cash 
            << " | Diff: $" << cash_diff << std::endl;
  std::cout << "  Expected Equity: $" << expected_equity << " | Actual: $" << final_equity 
            << " | Diff: $" << equity_diff << std::endl;
  
  bool verification_passed = (cash_diff <= eps && equity_diff <= eps);
  
  if (verification_passed) {
    std::cout << "✅ ENHANCED REPLAY VERIFICATION PASSED" << std::endl;
    
    // Calculate and display performance metrics
    double total_return = (final_equity / 100000.0) - 1.0;
    double monthly_return = std::pow(1.0 + total_return, 1.0/12.0) - 1.0;
    
    std::cout << "Performance Metrics:" << std::endl;
    std::cout << "  Total Return: " << (total_return * 100) << "%" << std::endl;
    std::cout << "  Monthly Return: " << (monthly_return * 100) << "%" << std::endl;
    
  } else {
    std::cout << "❌ ENHANCED REPLAY VERIFICATION FAILED" << std::endl;
    std::cout << "Cash tolerance: " << eps << ", Equity tolerance: " << eps << std::endl;
  }
  
  return verification_passed;
}

} // namespace sentio

