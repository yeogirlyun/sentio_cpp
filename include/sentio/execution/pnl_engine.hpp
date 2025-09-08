#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <cmath>
#include <optional>
#include <algorithm>
#include <stdexcept>

namespace sentio {

struct Fill {
  int64_t ts{0};
  double  price{0.0};
  double  qty{0.0};  // +buy, -sell
  double  fee{0.0};
  std::string venue;
};

struct BarPx {
  int64_t ts{0};
  double close{0.0};
  double bid{0.0};
  double ask{0.0};
};

struct Lot {
  double qty{0.0};
  double px{0.0};
  int64_t ts{0};
};

struct PnLSnapshot {
  int64_t ts{0};
  double position{0.0};
  double avg_price{0.0};
  double cash{0.0};
  double realized{0.0};
  double unrealized{0.0};
  double equity{0.0};
  double last_price{0.0};
};

struct FeeModel {
  double per_share{0.0};
  double bps_notional{0.0};
  double min_fee{0.0};
  double compute(double price, double qty) const {
    double absqty = std::abs(qty);
    double f = per_share * absqty + (bps_notional * (price * absqty));
    return std::max(f, min_fee);
  }
};

struct SlippageModel {
  double bps{0.0};
  double apply(double ref_price, double qty) const {
    double sgn = (qty >= 0.0 ? 1.0 : -1.0);
    double mult = 1.0 + sgn * bps;
    return ref_price * mult;
  }
};

class PnLEngine {
public:
  enum class PriceMode { Close, Mid, Bid, Ask };

  explicit PnLEngine(double start_cash = 100000.0)
    : cash_(start_cash), equity_(start_cash) {}

  void set_price_mode(PriceMode m) { price_mode_ = m; }
  void set_fee_model(const FeeModel& f) { fee_model_ = f; auto_fee_ = true; }
  void set_slippage_model(const SlippageModel& s) { slippage_ = s; }

  void on_fill(const Fill& fill) {
    if (std::abs(fill.qty) < 1e-12) return;
    const double qty = fill.qty;
    const double px  = fill.price;
    double fee = fill.fee;
    if (auto_fee_) fee = fee_model_.compute(px, qty);

    double remaining = qty;
    if (same_sign(remaining, position_)) {
      lots_.push_back(Lot{remaining, px, fill.ts});
      position_ += remaining;
      cash_ -= px * qty;
      cash_ -= fee;
    } else {
      while (std::abs(remaining) > 1e-12 && !lots_.empty() && opposite_sign(remaining, lots_.front().qty)) {
        Lot &lot = lots_.front();
        double close_qty = std::min(std::abs(remaining), std::abs(lot.qty));
        double dq = (lot.qty > 0 ? +close_qty : -close_qty);
        realized_ += (px - lot.px) * dq;
        cash_ -= px * (-dq);
        lot.qty -= dq;
        remaining += dq;
        if (std::abs(lot.qty) <= 1e-12) lots_.erase(lots_.begin());
      }
      if (std::abs(remaining) > 1e-12) {
        lots_.push_back(Lot{remaining, px, fill.ts});
        position_ += remaining;
        cash_ -= px * remaining;
      } else {
        position_ = sum_position_from_lots();
      }
      cash_ -= fee;
    }
    avg_price_ = compute_signed_avg();
  }

  void on_bar(const BarPx& bar) {
    last_price_ = reference_price(bar);
    unrealized_ = position_ * (last_price_ - avg_price_);
    equity_     = cash_ + position_ * last_price_;
    snapshots_.push_back(PnLSnapshot{bar.ts, position_, avg_price_, cash_, realized_, unrealized_, equity_, last_price_});
  }

  void reset(double start_cash = 100000.0) {
    lots_.clear(); snapshots_.clear();
    position_=0; avg_price_=0; cash_=start_cash; realized_=0; unrealized_=0; equity_=start_cash; last_price_=0;
  }

  const std::vector<PnLSnapshot>& snapshots() const { return snapshots_; }
  double position()  const { return position_; }
  double avg_price() const { return avg_price_; }
  double cash()      const { return cash_; }
  double realized()  const { return realized_; }
  double unrealized()const { return unrealized_; }
  double equity()    const { return equity_; }

private:
  static bool same_sign(double a, double b){ return (a>=0 && b>=0) || (a<=0 && b<=0); }
  static bool opposite_sign(double a, double b){ return (a>=0 && b<=0) || (a<=0 && b>=0); }

  double compute_signed_avg() const {
    double num = 0.0, den = 0.0;
    for (const auto &l : lots_) { num += l.px * l.qty; den += l.qty; }
    if (std::abs(den) < 1e-12) return 0.0;
    return num / den;
  }
  double sum_position_from_lots() const { double s=0.0; for (const auto &l : lots_) s += l.qty; return s; }

  double reference_price(const BarPx& bar) const {
    switch (price_mode_) {
      case PriceMode::Close: return bar.close;
      case PriceMode::Mid:   return (bar.bid>0 && bar.ask>0) ? 0.5*(bar.bid+bar.ask) : bar.close;
      case PriceMode::Bid:   return (bar.bid>0) ? bar.bid : bar.close;
      case PriceMode::Ask:   return (bar.ask>0) ? bar.ask : bar.close;
    }
    return bar.close;
  }

private:
  std::vector<Lot> lots_;
  std::vector<PnLSnapshot> snapshots_;

  double position_{0.0};
  double avg_price_{0.0};
  double cash_{0.0};
  double realized_{0.0};
  double unrealized_{0.0};
  double equity_{0.0};
  double last_price_{0.0};

  PriceMode price_mode_{PriceMode::Close};
  FeeModel fee_model_{};
  SlippageModel slippage_{};
  bool auto_fee_{false};
};

} // namespace sentio
