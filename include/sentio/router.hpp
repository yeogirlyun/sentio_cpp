#pragma once
#include <cstdint>
#include <optional>
#include <string>

namespace sentio {

enum class OrderSide { Buy, Sell };

struct StrategySignal {
  enum class Type { BUY, STRONG_BUY, SELL, STRONG_SELL, HOLD };
  Type   type{Type::HOLD};
  double confidence{0.0}; // 0..1
};

struct RouterCfg {
  double min_signal_strength = 0.10; // below -> ignore
  double signal_multiplier   = 1.00; // scales target weight
  double max_position_pct    = 0.05; // +/- 5%
  bool   require_rth         = true; // assume ingest enforces RTH
  // family config
  std::string base_symbol{"QQQ"};
  std::string bull3x{"TQQQ"};
  std::string bear3x{"SQQQ"};
  std::string bear1x{"PSQ"};
  // sizing
  double min_shares = 1.0;
  double lot_size   = 1.0; // for ETFs typically 1
};

struct RouteDecision {
  std::string instrument;
  double      target_weight; // [-max, +max]
};

struct AccountSnapshot { double equity{0.0}; double cash{0.0}; };

struct Order {
  std::string instrument;
  OrderSide   side{OrderSide::Buy};
  double      qty{0.0};
  double      notional{0.0};
  double      limit_price{0.0}; // 0 = market
  std::int64_t ts_utc{0};
  std::string signal_id;
};

class PriceBook; // fwd
double last_trade_price(const PriceBook& book, const std::string& instrument);

std::optional<RouteDecision> route(const StrategySignal& s, const RouterCfg& cfg, const std::string& base_symbol);

// High-level convenience: route + size into a market order
Order route_and_create_order(const std::string& signal_id,
                             const StrategySignal& sig,
                             const RouterCfg& cfg,
                             const std::string& base_symbol,
                             const PriceBook& book,
                             const AccountSnapshot& acct,
                             std::int64_t ts_utc);

} // namespace sentio