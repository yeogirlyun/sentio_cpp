#include "sentio/audit.hpp"
#include <cassert>
#include <iostream>

namespace sentio {

static bool exec(sqlite3* db, const char* sql) {
  char* err=nullptr;
  int rc = sqlite3_exec(db, sql, nullptr, nullptr, &err);
  if (rc!=SQLITE_OK) { if (err){ sqlite3_free(err);} return false; }
  return true;
}

bool Auditor::open(const std::string& path) {
  if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) return false;
  exec(db, "PRAGMA journal_mode=WAL;");
  exec(db, "PRAGMA synchronous=NORMAL;");
  exec(db, "PRAGMA temp_store=MEMORY;");
  exec(db, "PRAGMA mmap_size=268435456;");
  return true;
}
void Auditor::close() { if (db) sqlite3_close(db), db=nullptr; }

bool Auditor::ensure_schema() {
  const char* ddl =
    "PRAGMA journal_mode=WAL;"
    "CREATE TABLE IF NOT EXISTS audit_runs("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " created_at_utc TEXT NOT NULL DEFAULT (datetime('now')),"
    " kind TEXT NOT NULL, strategy_name TEXT NOT NULL, params_json TEXT NOT NULL,"
    " data_hash TEXT NOT NULL, code_commit TEXT, seed INTEGER, notes TEXT );"
    "CREATE TABLE IF NOT EXISTS signals("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,"
    " ts_utc TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL, price REAL NOT NULL,"
    " score REAL, confidence REAL, features_json TEXT );"
    "CREATE TABLE IF NOT EXISTS router_decisions("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL, signal_id INTEGER NOT NULL,"
    " policy TEXT NOT NULL, instrument TEXT NOT NULL, target_leverage REAL NOT NULL, target_weight REAL, notes TEXT );"
    "CREATE TABLE IF NOT EXISTS orders("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,"
    " ts_utc TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,"
    " qty REAL NOT NULL, price REAL, order_type TEXT NOT NULL, status TEXT NOT NULL,"
    " instrument TEXT, leverage_used REAL );"
    "CREATE TABLE IF NOT EXISTS fills("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL, order_id INTEGER,"
    " ts_utc TEXT NOT NULL, symbol TEXT NOT NULL, qty REAL NOT NULL, price REAL NOT NULL,"
    " fees REAL DEFAULT 0.0, slippage_bp REAL DEFAULT 0.0 );"
    "CREATE TABLE IF NOT EXISTS snapshots("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL, ts_utc TEXT NOT NULL,"
    " cash REAL NOT NULL, equity REAL NOT NULL, gross_exposure REAL NOT NULL, net_exposure REAL NOT NULL,"
    " pnl REAL NOT NULL, drawdown REAL NOT NULL );"
    "CREATE TABLE IF NOT EXISTS run_metrics("
    " run_id INTEGER PRIMARY KEY, bars INTEGER NOT NULL, trades INTEGER NOT NULL, ret_total REAL NOT NULL,"
    " ret_ann REAL NOT NULL, vol_ann REAL NOT NULL, sharpe REAL NOT NULL, mdd REAL NOT NULL,"
    " monthly_proj REAL, daily_trades REAL );"
    // Dictionary tables for compact audit (optional, for performance)
    "CREATE TABLE IF NOT EXISTS dict_symbol("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, sym TEXT UNIQUE NOT NULL );"
    "CREATE TABLE IF NOT EXISTS dict_policy("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL );"
    "CREATE TABLE IF NOT EXISTS dict_instrument("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, sym TEXT UNIQUE NOT NULL );"
    // Add compact columns to existing tables (optional, backward compatible)
    "ALTER TABLE signals ADD COLUMN ts_ms INTEGER DEFAULT NULL;"
    "ALTER TABLE signals ADD COLUMN symbol_id INTEGER DEFAULT NULL;"
    "ALTER TABLE router_decisions ADD COLUMN policy_id INTEGER DEFAULT NULL;"
    "ALTER TABLE router_decisions ADD COLUMN instrument_id INTEGER DEFAULT NULL;"
    "ALTER TABLE orders ADD COLUMN ts_ms INTEGER DEFAULT NULL;"
    "ALTER TABLE orders ADD COLUMN symbol_id INTEGER DEFAULT NULL;"
    "ALTER TABLE fills ADD COLUMN ts_ms INTEGER DEFAULT NULL;"
    "ALTER TABLE fills ADD COLUMN symbol_id INTEGER DEFAULT NULL;"
    "ALTER TABLE snapshots ADD COLUMN ts_ms INTEGER DEFAULT NULL;";
  return exec(db, ddl);
}

bool Auditor::begin_tx(){
  return exec(db, "BEGIN IMMEDIATE TRANSACTION;");
}
bool Auditor::commit_tx(){
  return exec(db, "COMMIT;");
}

bool Auditor::prepare_hot() {
  const char* sql_sig = "INSERT INTO signals(run_id,ts_utc,symbol,side,price,score) VALUES(?,?,?,?,?,?);";
  const char* sql_router="INSERT INTO router_decisions(run_id,signal_id,policy,instrument,target_leverage,target_weight,notes) VALUES(?,?,?,?,?,?,?);";
  const char* sql_ord = "INSERT INTO orders(run_id,ts_utc,symbol,side,qty,price,order_type,status,instrument,leverage_used) VALUES(?,?,?,?,?,?,?,?,?,?);";
  const char* sql_fill= "INSERT INTO fills(run_id,order_id,ts_utc,symbol,qty,price,fees,slippage_bp) VALUES(?,?,?,?,?,?,?,?);";
  const char* sql_snap= "INSERT INTO snapshots(run_id,ts_utc,cash,equity,gross_exposure,net_exposure,pnl,drawdown) VALUES(?,?,?,?,?,?,?,?);";
  const char* sql_metric="INSERT INTO run_metrics(run_id,bars,trades,ret_total,ret_ann,vol_ann,sharpe,mdd,monthly_proj,daily_trades) VALUES(?,?,?,?,?,?,?,?,?,?) "
                         "ON CONFLICT(run_id) DO UPDATE SET bars=excluded.bars,trades=excluded.trades,ret_total=excluded.ret_total,ret_ann=excluded.ret_ann,vol_ann=excluded.vol_ann,sharpe=excluded.sharpe,mdd=excluded.mdd,monthly_proj=excluded.monthly_proj,daily_trades=excluded.daily_trades;";
  if (sqlite3_prepare_v2(db, sql_sig, -1, &st_sig, nullptr) != SQLITE_OK) return false;
  if (sqlite3_prepare_v2(db, sql_router, -1, &st_router, nullptr) != SQLITE_OK) return false;
  if (sqlite3_prepare_v2(db, sql_ord, -1, &st_ord, nullptr) != SQLITE_OK) return false;
  if (sqlite3_prepare_v2(db, sql_fill, -1, &st_fill, nullptr) != SQLITE_OK) return false;
  if (sqlite3_prepare_v2(db, sql_snap, -1, &st_snap, nullptr) != SQLITE_OK) return false;
  if (sqlite3_prepare_v2(db, sql_metric, -1, &st_metrics, nullptr) != SQLITE_OK) return false;
  return true;
}

void Auditor::finalize_hot() {
  if (st_sig) sqlite3_finalize(st_sig);
  if (st_router) sqlite3_finalize(st_router);
  if (st_ord) sqlite3_finalize(st_ord);
  if (st_fill) sqlite3_finalize(st_fill);
  if (st_snap) sqlite3_finalize(st_snap);
  if (st_metrics) sqlite3_finalize(st_metrics);
  st_sig=st_router=st_ord=st_fill=st_snap=st_metrics=nullptr;
}

long long Auditor::insert_signal_fast(const std::string& ts,const std::string& base_sym,const std::string& side,double price,double score){
  sqlite3_reset(st_sig); sqlite3_clear_bindings(st_sig);
  sqlite3_bind_int64(st_sig,1,run_id);
  sqlite3_bind_text (st_sig,2,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_sig,3,base_sym.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_sig,4,side.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st_sig,5,price);
  sqlite3_bind_double(st_sig,6,score);
  if (sqlite3_step(st_sig) != SQLITE_DONE) return -1;
  return sqlite3_last_insert_rowid(db);
}

long long Auditor::insert_router_fast(long long signal_id,const std::string& policy,const std::string& instrument,double lev,double weight,const std::string& notes){
  sqlite3_reset(st_router); sqlite3_clear_bindings(st_router);
  sqlite3_bind_int64(st_router,1,run_id);
  sqlite3_bind_int64(st_router,2,signal_id);
  sqlite3_bind_text (st_router,3,policy.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_router,4,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st_router,5,lev);
  sqlite3_bind_double(st_router,6,weight);
  sqlite3_bind_text (st_router,7,notes.c_str(),-1,SQLITE_TRANSIENT);
  if (sqlite3_step(st_router) != SQLITE_DONE) return -1;
  return sqlite3_last_insert_rowid(db);
}

long long Auditor::insert_order_fast(const std::string& ts,const std::string& instrument,const std::string& side,double qty,const std::string& order_type,double price,const std::string& status,double leverage_used){
  sqlite3_reset(st_ord); sqlite3_clear_bindings(st_ord);
  sqlite3_bind_int64(st_ord,1,run_id);
  sqlite3_bind_text (st_ord,2,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_ord,3,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_ord,4,side.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st_ord,5,qty);
  sqlite3_bind_double(st_ord,6,price);
  sqlite3_bind_text (st_ord,7,order_type.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_ord,8,status.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_ord,9,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st_ord,10,leverage_used);
  if (sqlite3_step(st_ord) != SQLITE_DONE) return -1;
  return sqlite3_last_insert_rowid(db);
}

bool Auditor::insert_fill_fast(long long order_id,const std::string& ts,const std::string& instrument,double qty,double price,double fees,double slippage_bp){
  sqlite3_reset(st_fill); sqlite3_clear_bindings(st_fill);
  sqlite3_bind_int64(st_fill,1,run_id);
  sqlite3_bind_int64(st_fill,2,order_id);
  sqlite3_bind_text (st_fill,3,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st_fill,4,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st_fill,5,qty);
  sqlite3_bind_double(st_fill,6,price);
  sqlite3_bind_double(st_fill,7,fees);
  sqlite3_bind_double(st_fill,8,slippage_bp);
  return sqlite3_step(st_fill) == SQLITE_DONE;
}

bool Auditor::insert_snapshot_fast(const std::string& ts,double cash,double equity,double gross,double net,double pnl,double dd){
  sqlite3_reset(st_snap); sqlite3_clear_bindings(st_snap);
  sqlite3_bind_int64(st_snap,1,run_id);
  sqlite3_bind_text (st_snap,2,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st_snap,3,cash);
  sqlite3_bind_double(st_snap,4,equity);
  sqlite3_bind_double(st_snap,5,gross);
  sqlite3_bind_double(st_snap,6,net);
  sqlite3_bind_double(st_snap,7,pnl);
  sqlite3_bind_double(st_snap,8,dd);
  return sqlite3_step(st_snap) == SQLITE_DONE;
}

bool Auditor::insert_metrics_upsert(int bars,int trades,double ret_total,double ret_ann,double vol_ann,double sharpe,double mdd,double monthly_proj,double daily_trades){
  sqlite3_reset(st_metrics); sqlite3_clear_bindings(st_metrics);
  sqlite3_bind_int64(st_metrics,1,run_id);
  sqlite3_bind_int (st_metrics,2,bars);
  sqlite3_bind_int (st_metrics,3,trades);
  sqlite3_bind_double(st_metrics,4,ret_total);
  sqlite3_bind_double(st_metrics,5,ret_ann);
  sqlite3_bind_double(st_metrics,6,vol_ann);
  sqlite3_bind_double(st_metrics,7,sharpe);
  sqlite3_bind_double(st_metrics,8,mdd);
  sqlite3_bind_double(st_metrics,9,monthly_proj);
  sqlite3_bind_double(st_metrics,10,daily_trades);
  return sqlite3_step(st_metrics) == SQLITE_DONE;
}

bool Auditor::start_run(const std::string& kind, const std::string& strategy_name,
                 const std::string& params_json, const std::string& data_hash,
                 std::optional<long long> seed, std::optional<std::string> notes) {
  sqlite3_stmt* st=nullptr;
  const char* sql = "INSERT INTO audit_runs(kind,strategy_name,params_json,data_hash,seed,notes) VALUES(?,?,?,?,?,?);";
  if (sqlite3_prepare_v2(db, sql, -1, &st, nullptr) != SQLITE_OK) return false;
  sqlite3_bind_text(st,1,kind.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,2,strategy_name.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,3,params_json.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,4,data_hash.c_str(),-1,SQLITE_TRANSIENT);
  if (seed) sqlite3_bind_int64(st,5,*seed); else sqlite3_bind_null(st,5);
  if (notes) sqlite3_bind_text(st,6,notes->c_str(),-1,SQLITE_TRANSIENT); else sqlite3_bind_null(st,6);
  int rc = sqlite3_step(st); sqlite3_finalize(st);
  if (rc != SQLITE_DONE) return false;
  run_id = sqlite3_last_insert_rowid(db);
  return true;
}

long long Auditor::insert_signal(const std::string& ts,const std::string& base_sym,const std::string& side,double price,double score) {
  sqlite3_stmt* st=nullptr;
  const char* sql="INSERT INTO signals(run_id,ts_utc,symbol,side,price,score) VALUES(?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_text(st,2,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,3,base_sym.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,4,side.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,5,price);
  sqlite3_bind_double(st,6,score);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return (rc==SQLITE_DONE)? sqlite3_last_insert_rowid(db) : -1;
}

long long Auditor::insert_router(long long signal_id,const std::string& policy,const std::string& instrument,double lev,double weight,const std::string& notes){
  sqlite3_stmt* st=nullptr;
  const char* sql="INSERT INTO router_decisions(run_id,signal_id,policy,instrument,target_leverage,target_weight,notes) VALUES(?,?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int64(st,2,signal_id);
  sqlite3_bind_text(st,3,policy.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,4,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,5,lev);
  sqlite3_bind_double(st,6,weight);
  sqlite3_bind_text(st,7,notes.c_str(),-1,SQLITE_TRANSIENT);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return (rc==SQLITE_DONE)? sqlite3_last_insert_rowid(db) : -1;
}

long long Auditor::insert_order(const std::string& ts,const std::string& instrument,const std::string& side,double qty,const std::string& order_type,double price,const std::string& status,double leverage_used){
  sqlite3_stmt* st=nullptr;
  const char* sql="INSERT INTO orders(run_id,ts_utc,symbol,side,qty,price,order_type,status,instrument,leverage_used) VALUES(?,?,?,?,?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_text(st,2,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,3,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,4,side.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,5,qty);
  sqlite3_bind_double(st,6,price);
  sqlite3_bind_text(st,7,order_type.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,8,status.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,9,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,10,leverage_used);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return (rc==SQLITE_DONE)? sqlite3_last_insert_rowid(db) : -1;
}

bool Auditor::insert_fill(long long order_id,const std::string& ts,const std::string& instrument,double qty,double price,double fees,double slippage_bp){
  sqlite3_stmt* st=nullptr;
  const char* sql="INSERT INTO fills(run_id,order_id,ts_utc,symbol,qty,price,fees,slippage_bp) VALUES(?,?,?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int64(st,2,order_id);
  sqlite3_bind_text(st,3,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,4,instrument.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,5,qty);
  sqlite3_bind_double(st,6,price);
  sqlite3_bind_double(st,7,fees);
  sqlite3_bind_double(st,8,slippage_bp);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return rc==SQLITE_DONE;
}

bool Auditor::insert_snapshot(const std::string& ts,double cash,double equity,double gross,double net,double pnl,double dd){
  sqlite3_stmt* st=nullptr;
  const char* sql="INSERT INTO snapshots(run_id,ts_utc,cash,equity,gross_exposure,net_exposure,pnl,drawdown) VALUES(?,?,?,?,?,?,?,?);";
  if (sqlite3_prepare_v2(db, sql, -1, &st, nullptr) != SQLITE_OK) return false;
  std::cerr << "Auditor: run_id = " << run_id << std::endl;
  std::cerr << "Auditor: timestamp = '" << ts << "'" << std::endl;
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_text(st,2,ts.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,3,cash);
  sqlite3_bind_double(st,4,equity);
  sqlite3_bind_double(st,5,gross);
  sqlite3_bind_double(st,6,net);
  sqlite3_bind_double(st,7,pnl);
  sqlite3_bind_double(st,8,dd);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return rc==SQLITE_DONE;
}

bool Auditor::insert_metrics(int bars,int trades,double ret_total,double ret_ann,double vol_ann,double sharpe,double mdd,double monthly_proj,double daily_trades){
  sqlite3_stmt* st=nullptr;
  const char* sql="INSERT INTO run_metrics(run_id,bars,trades,ret_total,ret_ann,vol_ann,sharpe,mdd,monthly_proj,daily_trades) VALUES(?,?,?,?,?,?,?,?,?,?) "
                  "ON CONFLICT(run_id) DO UPDATE SET bars=excluded.bars,trades=excluded.trades,ret_total=excluded.ret_total,ret_ann=excluded.ret_ann,vol_ann=excluded.vol_ann,sharpe=excluded.sharpe,mdd=excluded.mdd,monthly_proj=excluded.monthly_proj,daily_trades=excluded.daily_trades;";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int(st,2,bars);
  sqlite3_bind_int(st,3,trades);
  sqlite3_bind_double(st,4,ret_total);
  sqlite3_bind_double(st,5,ret_ann);
  sqlite3_bind_double(st,6,vol_ann);
  sqlite3_bind_double(st,7,sharpe);
  sqlite3_bind_double(st,8,mdd);
  sqlite3_bind_double(st,9,monthly_proj);
  sqlite3_bind_double(st,10,daily_trades);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return rc==SQLITE_DONE;
}

} // namespace sentio
 
// Compact helpers impl
namespace sentio {

int Auditor::upsert_symbol_id(const std::string& sym){
  sqlite3_stmt* st=nullptr; int id=-1;
  sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO dict_symbol(sym) VALUES(?1);", -1, &st, nullptr);
  sqlite3_bind_text(st,1,sym.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_step(st); sqlite3_finalize(st);
  sqlite3_prepare_v2(db, "SELECT id FROM dict_symbol WHERE sym=?1;", -1, &st, nullptr);
  sqlite3_bind_text(st,1,sym.c_str(),-1,SQLITE_TRANSIENT);
  if (sqlite3_step(st)==SQLITE_ROW) id=sqlite3_column_int(st,0);
  sqlite3_finalize(st);
  return id;
}

int Auditor::upsert_policy_id(const std::string& name){
  sqlite3_stmt* st=nullptr; int id=-1;
  sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO dict_policy(name) VALUES(?1);", -1, &st, nullptr);
  sqlite3_bind_text(st,1,name.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_step(st); sqlite3_finalize(st);
  sqlite3_prepare_v2(db, "SELECT id FROM dict_policy WHERE name=?1;", -1, &st, nullptr);
  sqlite3_bind_text(st,1,name.c_str(),-1,SQLITE_TRANSIENT);
  if (sqlite3_step(st)==SQLITE_ROW) id=sqlite3_column_int(st,0);
  sqlite3_finalize(st);
  return id;
}

int Auditor::upsert_instrument_id(const std::string& sym){
  sqlite3_stmt* st=nullptr; int id=-1;
  sqlite3_prepare_v2(db, "INSERT OR IGNORE INTO dict_instrument(sym) VALUES(?1);", -1, &st, nullptr);
  sqlite3_bind_text(st,1,sym.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_step(st); sqlite3_finalize(st);
  sqlite3_prepare_v2(db, "SELECT id FROM dict_instrument WHERE sym=?1;", -1, &st, nullptr);
  sqlite3_bind_text(st,1,sym.c_str(),-1,SQLITE_TRANSIENT);
  if (sqlite3_step(st)==SQLITE_ROW) id=sqlite3_column_int(st,0);
  sqlite3_finalize(st);
  return id;
}

long long Auditor::insert_signal_compact(long long ts_ms, int symbol_id, const char* side, double price, double score){
  sqlite3_stmt* st=nullptr;
  const char* sql = "INSERT INTO signals(run_id, ts_ms, symbol_id, side, price, score) VALUES(?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int64(st,2,ts_ms);
  sqlite3_bind_int  (st,3,symbol_id);
  sqlite3_bind_text (st,4,side,-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,5,price);
  sqlite3_bind_double(st,6,score);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return (rc==SQLITE_DONE)? sqlite3_last_insert_rowid(db) : -1;
}

long long Auditor::insert_router_compact(long long signal_id, int policy_id, int instrument_id, double lev, double weight){
  sqlite3_stmt* st=nullptr;
  const char* sql = "INSERT INTO router_decisions(run_id, signal_id, policy_id, instrument_id, target_leverage, target_weight) VALUES(?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int64(st,2,signal_id);
  sqlite3_bind_int  (st,3,policy_id);
  sqlite3_bind_int  (st,4,instrument_id);
  sqlite3_bind_double(st,5,lev);
  sqlite3_bind_double(st,6,weight);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return (rc==SQLITE_DONE)? sqlite3_last_insert_rowid(db) : -1;
}

long long Auditor::insert_order_compact(long long ts_ms, int symbol_id, const char* side, double qty, const char* order_type, double price, const char* status, double leverage_used){
  sqlite3_stmt* st=nullptr;
  const char* sql = "INSERT INTO orders(run_id, ts_ms, symbol_id, side, qty, price, order_type, status, leverage_used) VALUES(?,?,?,?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int64(st,2,ts_ms);
  sqlite3_bind_int  (st,3,symbol_id);
  sqlite3_bind_text (st,4,side,-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,5,qty);
  sqlite3_bind_double(st,6,price);
  sqlite3_bind_text (st,7,order_type,-1,SQLITE_TRANSIENT);
  sqlite3_bind_text (st,8,status,-1,SQLITE_TRANSIENT);
  sqlite3_bind_double(st,9,leverage_used);
  int rc=sqlite3_step(st); sqlite3_finalize(st);
  return (rc==SQLITE_DONE)? sqlite3_last_insert_rowid(db) : -1;
}

bool Auditor::insert_fill_compact(long long order_id, long long ts_ms, int symbol_id, double qty, double price, double fees, double slippage_bp){
  sqlite3_stmt* st=nullptr;
  const char* sql = "INSERT INTO fills(run_id, order_id, ts_ms, symbol_id, qty, price, fees, slippage_bp) VALUES(?,?,?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int64(st,2,order_id);
  sqlite3_bind_int64(st,3,ts_ms);
  sqlite3_bind_int  (st,4,symbol_id);
  sqlite3_bind_double(st,5,qty);
  sqlite3_bind_double(st,6,price);
  sqlite3_bind_double(st,7,fees);
  sqlite3_bind_double(st,8,slippage_bp);
  return sqlite3_step(st) == SQLITE_DONE;
}

bool Auditor::insert_snapshot_compact(long long ts_ms, double cash, double equity, double gross, double net, double pnl, double dd){
  sqlite3_stmt* st=nullptr;
  const char* sql = "INSERT INTO snapshots(run_id, ts_ms, cash, equity, gross_exposure, net_exposure, pnl, drawdown) VALUES(?,?,?,?,?,?,?,?);";
  sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
  sqlite3_bind_int64(st,1,run_id);
  sqlite3_bind_int64(st,2,ts_ms);
  sqlite3_bind_double(st,3,cash);
  sqlite3_bind_double(st,4,equity);
  sqlite3_bind_double(st,5,gross);
  sqlite3_bind_double(st,6,net);
  sqlite3_bind_double(st,7,pnl);
  sqlite3_bind_double(st,8,dd);
  return sqlite3_step(st) == SQLITE_DONE;
}

} // namespace sentio

