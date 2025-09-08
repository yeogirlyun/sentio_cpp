#pragma once
#include <cstdint>

namespace sentio {

struct TradeCtx {
  double price;       // mid/exec price
  double notional;    // |shares| * price
  long   shares;      // signed
  bool   is_short;    // for borrow fees if modeled
};

class IFeeModel {
public:
  virtual ~IFeeModel() = default;
  virtual double commission(const TradeCtx& t) const = 0;  // $ cost
  virtual double exchange_fees(const TradeCtx& t) const { return 0.0; }
  virtual double borrow_fee_daily_bp(double notional_short) const { return 0.0; }
};

} // namespace sentio
