#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <optional>

namespace sentio {

enum class TraceReason : uint16_t {
  OK = 0,
  NO_STRATEGY_OUTPUT,
  NOT_RTH,
  HOLIDAY,
  WARMUP,
  NAN_INPUT,
  THRESHOLD_TOO_TIGHT,
  COOLDOWN_ACTIVE,
  DUPLICATE_BAR_TS,
  EMPTY_PRICEBOOK,
  NO_PRICE_FOR_INSTRUMENT,
  ROUTER_REJECTED,
  ORDER_QTY_LT_MIN,
  UNKNOWN
};

struct TraceRow {
  std::int64_t ts_utc{};
  std::string  instrument;     // stream instrument (e.g., QQQ)
  std::string  routed;         // routed instrument (e.g., TQQQ/SQQQ)
  double       close{};
  bool         is_rth{true};
  bool         warmed{true};
  bool         inputs_finite{true};
  double       confidence{0.0};              // raw strategy conf
  double       conf_after_gate{0.0};         // post gate
  double       target_weight{0.0};           // router decision
  double       last_px{0.0};                 // last price for routed
  double       order_qty{0.0};
  TraceReason  reason{TraceReason::UNKNOWN};
  std::string  note;                         // optional detail
};

class SignalTrace {
public:
  void push(const TraceRow& r) { rows_.push_back(r); }
  const std::vector<TraceRow>& rows() const { return rows_; }
  // useful summaries
  std::size_t count(TraceReason r) const;
private:
  std::vector<TraceRow> rows_;
};

} // namespace sentio
