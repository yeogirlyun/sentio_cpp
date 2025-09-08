#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <optional>
#include <unordered_map>
#include <vector>

namespace sentio {

// --- Core enums/structs ----
enum class SigType : uint8_t { BUY=0, STRONG_BUY=1, SELL=2, STRONG_SELL=3, HOLD=4 };
enum class Side    : uint8_t { Buy=0, Sell=1 };

// Simple Bar structure for audit system (avoiding conflicts with core.hpp)
struct AuditBar { 
  double open{}, high{}, low{}, close{}, volume{};
};

struct AuditPosition { 
  double qty{0.0}; 
  double avg_px{0.0}; 
};

struct AccountState {
  double cash{0.0};
  double realized{0.0};
  double equity{0.0};
  // computed: equity = cash + realized + sum(qty * mark_px)
};

struct AuditConfig {
  std::string run_id;           // stable id for this run
  std::string file_path;        // where JSONL events are appended
  bool        flush_each=true;  // fsync-ish (fflush) after each write
};

// --- Recorder: append events to JSONL ---
class AuditRecorder {
public:
  explicit AuditRecorder(const AuditConfig& cfg);
  ~AuditRecorder();

  // lifecycle
  void event_run_start(std::int64_t ts_utc, const std::string& meta_json="{}");
  void event_run_end(std::int64_t ts_utc, const std::string& meta_json="{}");

  // data plane
  void event_bar   (std::int64_t ts_utc, const std::string& instrument, const AuditBar& b);
  void event_signal(std::int64_t ts_utc, const std::string& base_symbol, SigType type, double confidence);
  void event_route (std::int64_t ts_utc, const std::string& base_symbol, const std::string& instrument, double target_weight);
  void event_order (std::int64_t ts_utc, const std::string& instrument, Side side, double qty, double limit_px);
  void event_fill  (std::int64_t ts_utc, const std::string& instrument, double price, double qty, double fees, Side side);
  void event_snapshot(std::int64_t ts_utc, const AccountState& acct);
  void event_metric (std::int64_t ts_utc, const std::string& key, double value);

  // Extended events with chain id and richer context for precise trade linking and P/L deltas.
  void event_signal_ex(std::int64_t ts_utc, const std::string& base_symbol, SigType type, double confidence,
                       const std::string& chain_id);
  void event_route_ex (std::int64_t ts_utc, const std::string& base_symbol, const std::string& instrument, double target_weight,
                       const std::string& chain_id);
  void event_order_ex (std::int64_t ts_utc, const std::string& instrument, Side side, double qty, double limit_px,
                       const std::string& chain_id);
  void event_fill_ex  (std::int64_t ts_utc, const std::string& instrument, double price, double qty, double fees, Side side,
                       double realized_pnl_delta, double equity_after, double position_after,
                       const std::string& chain_id);

  // Get current config (for creating new instances)
  AuditConfig get_config() const { return {run_id_, file_path_, flush_each_}; }

private:
  std::string run_id_;
  std::string file_path_;
  std::FILE*  fp_{nullptr};
  std::uint64_t seq_{0};
  bool flush_each_;
  void write_line_(const std::string& s);
  static std::string sha1_hex_(const std::string& s); // tiny local impl
  static std::string json_escape_(const std::string& s);
};

// --- Replayer: read JSONL, rebuild state, recompute P&L, verify ---
struct ReplayResult {
  // recomputed
  std::unordered_map<std::string, AuditPosition> positions;
  AccountState acct{};
  std::size_t  bars{0}, signals{0}, routes{0}, orders{0}, fills{0};
  // mismatches discovered
  std::vector<std::string> issues;
};

class AuditReplayer {
public:
  // price map can be filled from bar events; you may also inject EOD marks
  struct PriceBook { std::unordered_map<std::string, double> last_px; };

  // replay the file; return recomputed account/pnl from fills + marks
  static std::optional<ReplayResult> replay_file(const std::string& file_path,
                                                 const std::string& run_id_expect = "");
private:
  static bool apply_bar_(PriceBook& pb, const std::string& instrument, const AuditBar& b);
  static void mark_to_market_(const PriceBook& pb, ReplayResult& rr);
  static void apply_fill_(ReplayResult& rr, const std::string& inst, double px, double qty, double fees, Side side);
};

} // namespace sentio