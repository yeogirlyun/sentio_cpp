#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <atomic>
#include <optional>

namespace sentio {

enum class DropReason : uint16_t {
  NONE=0, WARMUP, NAN_INPUT, THRESHOLD_TOO_TIGHT,
  COOLDOWN_ACTIVE, DUPLICATE_BAR_TS
};

struct SignalHealth {
  std::atomic<uint64_t> emitted{0};
  std::atomic<uint64_t> dropped{0};
  std::unordered_map<DropReason, std::atomic<uint64_t>> by_reason;
  SignalHealth();
  void incr_emit();
  void incr_drop(DropReason r);
};

struct GateCfg { 
  int cooldown_bars=0; 
  double min_conf=0.05; 
};

class SignalGate {
public:
  explicit SignalGate(const GateCfg& cfg, SignalHealth* health);
  // Returns nullopt if dropped; otherwise passes through with possibly clamped confidence.
  std::optional<double> accept(std::int64_t ts_utc_epoch,
                               bool inputs_finite,
                               bool warmed_up,
                               double confidence);
private:
  GateCfg cfg_;
  SignalHealth* health_;
  std::int64_t last_emit_ts_{-1};
  int cooldown_left_{0};
};

} // namespace sentio
