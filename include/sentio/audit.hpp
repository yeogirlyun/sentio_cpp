#pragma once
#include <sqlite3.h>
#include <string>
#include <optional>

namespace sentio {

struct Auditor {
  sqlite3* db{};
  long long run_id{-1};
  // prepared statements (optional fast path)
  sqlite3_stmt* st_sig{nullptr};
  sqlite3_stmt* st_router{nullptr};
  sqlite3_stmt* st_ord{nullptr};
  sqlite3_stmt* st_fill{nullptr};
  sqlite3_stmt* st_snap{nullptr};
  sqlite3_stmt* st_metrics{nullptr};

  bool open(const std::string& path);
  void close();
  bool ensure_schema(); // creates tables if not exists
  bool begin_tx();
  bool commit_tx();
  bool prepare_hot();
  void finalize_hot();

  // start a run (kind: backtest | wf-train | wf-oos | oos-2w)
  bool start_run(const std::string& kind, const std::string& strategy_name,
                 const std::string& params_json, const std::string& data_hash,
                 std::optional<long long> seed, std::optional<std::string> notes);

  long long insert_signal(const std::string& ts, const std::string& base_sym,
                          const std::string& side, double price, double score);

  long long insert_router(long long signal_id, const std::string& policy,
                          const std::string& instrument, double lev, double weight, const std::string& notes);

  long long insert_order(const std::string& ts, const std::string& instrument, const std::string& side,
                         double qty, const std::string& order_type, double price, const std::string& status,
                         double leverage_used);

  bool insert_fill(long long order_id, const std::string& ts, const std::string& instrument,
                   double qty, double price, double fees, double slippage_bp);

  bool insert_snapshot(const std::string& ts, double cash, double equity,
                       double gross, double net, double pnl, double dd);

  bool insert_metrics(int bars,int trades,double ret_total,double ret_ann,double vol_ann,double sharpe,double mdd,double monthly_proj,double daily_trades);

  // Fast-path versions using prepared statements
  long long insert_signal_fast(const std::string& ts,const std::string& base_sym,const std::string& side,double price,double score);
  long long insert_router_fast(long long signal_id,const std::string& policy,const std::string& instrument,double lev,double weight,const std::string& notes);
  long long insert_order_fast(const std::string& ts,const std::string& instrument,const std::string& side,double qty,const std::string& order_type,double price,const std::string& status,double leverage_used);
  bool insert_fill_fast(long long order_id,const std::string& ts,const std::string& instrument,double qty,double price,double fees,double slippage_bp);
  bool insert_snapshot_fast(const std::string& ts,double cash,double equity,double gross,double net,double pnl,double dd);
  bool insert_metrics_upsert(int bars,int trades,double ret_total,double ret_ann,double vol_ann,double sharpe,double mdd,double monthly_proj,double daily_trades);

  // Compact dictionary helpers
  int upsert_symbol_id(const std::string& sym);
  int upsert_policy_id(const std::string& name);
  int upsert_instrument_id(const std::string& sym);

  // Compact insert methods (prefer these for new writes)
  long long insert_signal_compact(long long ts_ms, int symbol_id, const char* side, double price, double score);
  long long insert_router_compact(long long signal_id, int policy_id, int instrument_id, double lev, double weight);
  long long insert_order_compact(long long ts_ms, int symbol_id, const char* side, double qty, const char* order_type, double price, const char* status, double leverage_used);
  bool insert_fill_compact(long long order_id, long long ts_ms, int symbol_id, double qty, double price, double fees, double slippage_bp);
  bool insert_snapshot_compact(long long ts_ms, double cash, double equity, double gross, double net, double pnl, double dd);
};

} // namespace sentio

