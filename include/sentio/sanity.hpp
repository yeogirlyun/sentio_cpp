#pragma once
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <unordered_map>

namespace sentio {

// Reuse your existing simple types
struct Bar { double open{}, high{}, low{}, close{}; };

// Signal types for strategy system
enum class SigType : uint8_t { BUY=0, STRONG_BUY=1, SELL=2, STRONG_SELL=3, HOLD=4 };

struct SanityIssue {
  enum class Severity { Warn, Error, Fatal };
  Severity severity{Severity::Error};
  std::string where;      // subsystem (DATA/FEATURE/STRAT/ROUTER/EXEC/PnL/AUDIT)
  std::string what;       // human message
  std::int64_t ts_utc{0}; // when applicable
};

struct SanityReport {
  std::vector<SanityIssue> issues;
  bool ok() const;                 // == no Error/Fatal
  std::size_t errors() const;
  std::size_t fatals() const;
  void add(SanityIssue::Severity sev, std::string where, std::string what, std::int64_t ts=0);
};

// Minimal interfaces (match your existing ones)
class PriceBook {
public:
  virtual void upsert_latest(const std::string& instrument, const Bar& b) = 0;
  virtual const Bar* get_latest(const std::string& instrument) const = 0;
  virtual bool has_instrument(const std::string& instrument) const = 0;
  virtual std::size_t size() const = 0;
  virtual ~PriceBook() = default;
};

struct Position { double qty{0.0}; double avg_px{0.0}; };
struct AccountState { double cash{0.0}; double realized{0.0}; double equity{0.0}; };

struct AuditEventCounts {
  std::size_t bars{0}, signals{0}, routes{0}, orders{0}, fills{0};
};

// Contracts you can call from tests or at the end of a run
namespace sanity {

// Data layer
void check_bar_monotonic(const std::vector<std::pair<std::int64_t, Bar>>& bars,
                         int expected_spacing_sec,
                         SanityReport& rep);

void check_bar_values_finite(const std::vector<std::pair<std::int64_t, Bar>>& bars,
                             SanityReport& rep);

// PriceBook coherence
void check_pricebook_coherence(const PriceBook& pb,
                               const std::vector<std::string>& required_instruments,
                               SanityReport& rep);

// Strategy/Routing layer
void check_signal_confidence_range(double conf, SanityReport& rep, std::int64_t ts);
void check_routed_instrument_has_price(const PriceBook& pb,
                                       const std::string& routed,
                                       SanityReport& rep, std::int64_t ts);

// Execution layer
void check_order_qty_min(double qty, double min_shares,
                         SanityReport& rep, std::int64_t ts);
void check_order_side_qty_sign_consistency(const std::string& side, double qty,
                                           SanityReport& rep, std::int64_t ts);

// P&L invariants
void check_equity_consistency(const AccountState& acct,
                              const std::unordered_map<std::string, Position>& pos,
                              const PriceBook& pb,
                              SanityReport& rep);

// Audit correlations
void check_audit_counts(const AuditEventCounts& c,
                        SanityReport& rep);

} // namespace sanity

// Lightweight runtime guard macros (no external deps)
#define SENTIO_ASSERT_FINITE(val, where, rep, ts) \
  do { if (!std::isfinite(val)) { (rep).add(SanityIssue::Severity::Fatal, (where), "non-finite value: " #val, (ts)); } } while(0)

} // namespace sentio
