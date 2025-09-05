#include "sentio/sanity.hpp"
#include <cmath>
#include <algorithm>

namespace sentio {

// PriceBook is now abstract - implementations must be provided by concrete classes

bool SanityReport::ok() const {
  for (auto& i : issues) if (i.severity != SanityIssue::Severity::Warn) return false;
  return true;
}
std::size_t SanityReport::errors() const {
  return std::count_if(issues.begin(), issues.end(), [](auto& i){
    return i.severity==SanityIssue::Severity::Error;
  });
}
std::size_t SanityReport::fatals() const {
  return std::count_if(issues.begin(), issues.end(), [](auto& i){
    return i.severity==SanityIssue::Severity::Fatal;
  });
}
void SanityReport::add(SanityIssue::Severity sev, std::string where, std::string what, std::int64_t ts){
  issues.push_back({sev,std::move(where),std::move(what),ts});
}

namespace sanity {

void check_bar_monotonic(const std::vector<std::pair<std::int64_t, Bar>>& bars,
                         int expected_spacing_sec,
                         SanityReport& rep)
{
  if (bars.empty()) return;
  for (std::size_t i=1;i<bars.size();++i){
    auto prev = bars[i-1].first;
    auto cur  = bars[i].first;
    if (cur <= prev) {
      rep.add(SanityIssue::Severity::Fatal, "DATA", "non-increasing timestamp", cur);
    }
    if (expected_spacing_sec>0) {
      auto gap = cur - prev;
      if (gap != expected_spacing_sec) {
        rep.add(SanityIssue::Severity::Error, "DATA",
          "unexpected spacing: got "+std::to_string((long long)gap)+"s expected "+std::to_string(expected_spacing_sec)+"s", cur);
      }
    }
    const Bar& b = bars[i].second;
    if (!(b.low <= b.high)) {
      rep.add(SanityIssue::Severity::Error, "DATA", "bar.low > bar.high", cur);
    }
  }
}

void check_bar_values_finite(const std::vector<std::pair<std::int64_t, Bar>>& bars,
                             SanityReport& rep)
{
  for (auto& it : bars) {
    auto ts = it.first; const Bar& b = it.second;
    if (!(std::isfinite(b.open) && std::isfinite(b.high) && std::isfinite(b.low) && std::isfinite(b.close))) {
      rep.add(SanityIssue::Severity::Fatal, "DATA", "non-finite OHLC", ts);
    }
    if (b.open<=0 || b.high<=0 || b.low<=0 || b.close<=0) {
      rep.add(SanityIssue::Severity::Error, "DATA", "non-positive price", ts);
    }
  }
}

void check_pricebook_coherence(const PriceBook& pb,
                               const std::vector<std::string>& required_instruments,
                               SanityReport& rep)
{
  for (auto& inst : required_instruments) {
    if (!pb.has_instrument(inst)) {
      rep.add(SanityIssue::Severity::Error, "DATA", "PriceBook missing instrument: "+inst);
    } else {
      auto* b = pb.get_latest(inst);
      if (!b || !std::isfinite(b->close)) {
        rep.add(SanityIssue::Severity::Error, "DATA", "PriceBook non-finite last close for: "+inst);
      }
    }
  }
}

void check_signal_confidence_range(double conf, SanityReport& rep, std::int64_t ts) {
  if (!(conf>=0.0 && conf<=1.0)) {
    rep.add(SanityIssue::Severity::Error, "STRAT", "signal confidence out of [0,1]", ts);
  }
}

void check_routed_instrument_has_price(const PriceBook& pb,
                                       const std::string& routed,
                                       SanityReport& rep, std::int64_t ts)
{
  if (routed.empty()) {
    rep.add(SanityIssue::Severity::Fatal, "ROUTER", "empty routed instrument", ts);
    return;
  }
  if (!pb.has_instrument(routed)) {
    rep.add(SanityIssue::Severity::Error, "ROUTER", "routed instrument missing in PriceBook: "+routed, ts);
  } else if (auto* b = pb.get_latest(routed); !b || !std::isfinite(b->close)) {
    rep.add(SanityIssue::Severity::Error, "ROUTER", "routed instrument has non-finite price: "+routed, ts);
  }
}

void check_order_qty_min(double qty, double min_shares,
                         SanityReport& rep, std::int64_t ts)
{
  if (!(std::isfinite(qty))) {
    rep.add(SanityIssue::Severity::Fatal, "EXEC", "order qty non-finite", ts);
    return;
  }
  if (qty != 0.0 && std::abs(qty) < min_shares) {
    rep.add(SanityIssue::Severity::Warn, "EXEC", "order qty < min_shares", ts);
  }
}

void check_order_side_qty_sign_consistency(const std::string& side, double qty,
                                           SanityReport& rep, std::int64_t ts)
{
  if (side=="BUY" && qty<0)  rep.add(SanityIssue::Severity::Error, "EXEC", "BUY with negative qty", ts);
  if (side=="SELL"&& qty>0)  rep.add(SanityIssue::Severity::Error, "EXEC", "SELL with positive qty", ts);
}

void check_equity_consistency(const AccountState& acct,
                              const std::unordered_map<std::string, Position>& pos,
                              const PriceBook& pb,
                              SanityReport& rep)
{
  if (!std::isfinite(acct.cash) || !std::isfinite(acct.realized) || !std::isfinite(acct.equity)) {
    rep.add(SanityIssue::Severity::Fatal, "PnL", "non-finite account values");
    return;
  }
  // recompute mark-to-market
  double mtm = 0.0;
  for (auto& kv : pos) {
    const auto& inst = kv.first;
    const auto& p = kv.second;
    if (!std::isfinite(p.qty) || !std::isfinite(p.avg_px)) {
      rep.add(SanityIssue::Severity::Fatal, "PnL", "non-finite position for "+inst);
      continue;
    }
    auto* b = pb.get_latest(inst);
    if (!b) continue;
    mtm += p.qty * b->close;
  }
  double equity_calc = acct.cash + acct.realized + mtm;
  if (std::isfinite(equity_calc) && std::abs(equity_calc - acct.equity) > 1e-6) {
    rep.add(SanityIssue::Severity::Error, "PnL", "equity mismatch (calc vs recorded) diff="+std::to_string(equity_calc - acct.equity));
  }
}

void check_audit_counts(const AuditEventCounts& c, SanityReport& rep) {
  if (c.orders < c.fills) {
    rep.add(SanityIssue::Severity::Error, "AUDIT", "fills exceed orders");
  }
  // Loose ratios to catch obviously broken runs
  if (c.routes && c.orders==0) {
    rep.add(SanityIssue::Severity::Warn, "AUDIT", "routes exist but no orders");
  }
  if (c.signals && c.routes==0) {
    rep.add(SanityIssue::Severity::Warn, "AUDIT", "signals exist but no routes");
  }
}

} // namespace sanity
} // namespace sentio
