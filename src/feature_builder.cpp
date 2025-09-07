#include "sentio/feature_builder.hpp"
#include <algorithm>

namespace sentio {

FeatureBuilder::FeatureBuilder(FeaturePlan plan, FeatureBuilderCfg cfg)
: plan_(std::move(plan))
, cfg_(cfg)
, sma_fast_(cfg.sma_fast)
, sma_slow_(cfg.sma_slow)
, vol_rtn_(std::max(2, cfg.vol_window))
, rsi_(cfg.rsi_period)
{}

void FeatureBuilder::reset(){
  *this = FeatureBuilder(plan_, cfg_);
}

void FeatureBuilder::on_bar(const Bar& b, const std::optional<MicroTick>& mt) {
  ++bars_seen_;

  // --- returns ---
  if (!close_q_.empty()) {
    double prev = close_q_.back();
    last_ret_1m_ = (b.close - prev) / std::max(1e-12, prev);
    vol_rtn_.push(last_ret_1m_);
  } else {
    last_ret_1m_ = 0.0;
  }

  close_q_.push_back(b.close);
  if (close_q_.size() > (size_t)std::max({cfg_.ret_5m_window, cfg_.sma_slow, cfg_.rsi_period+1})) {
    close_q_.pop_front();
  }

  if (close_q_.size() >= (size_t)cfg_.ret_5m_window+1) {
    double prev5 = close_q_[close_q_.size()-(cfg_.ret_5m_window+1)];
    last_ret_5m_ = (b.close - prev5) / std::max(1e-12, prev5);
  }

  // --- RSI ---
  if (close_q_.size() >= 2) {
    double prev = close_q_[close_q_.size()-2];
    rsi_.push(prev, b.close);
    last_rsi_ = rsi_.value();
  }

  // --- SMA fast/slow ---
  sma_fast_.push(b.close);
  sma_slow_.push(b.close);
  last_sma_fast_ = sma_fast_.full()? sma_fast_.mean() : NAN;
  last_sma_slow_ = sma_slow_.full()? sma_slow_.mean() : NAN;

  // --- Volatility (stdev of 1m returns) ---
  last_vol_1m_ = vol_rtn_.full()? vol_rtn_.stdev() : NAN;

  // --- Spread bp ---
  if (mt && std::isfinite(mt->bid) && std::isfinite(mt->ask)) {
    double mid = 0.5*(mt->bid + mt->ask);
    if (mid>0) last_spread_bp_ = 1e4 * (mt->ask - mt->bid) / mid;
  } else {
    // Proxy from high/low as a fallback (intrabar range proxy), otherwise default
    double mid = (b.high + b.low) * 0.5;
    if (mid>0) last_spread_bp_ = 1e4 * (b.high - b.low) / std::max(1e-12, mid) * 0.1; // scaled
    else       last_spread_bp_ = cfg_.default_spread_bp;
  }

  // clamp any negatives/NaNs on first bars
  if (!std::isfinite(last_ret_5m_)) last_ret_5m_ = 0.0;
  if (!std::isfinite(last_rsi_))    last_rsi_    = 50.0;
  if (!std::isfinite(last_sma_fast_)) last_sma_fast_ = b.close;
  if (!std::isfinite(last_sma_slow_)) last_sma_slow_ = b.close;
  if (!std::isfinite(last_vol_1m_))   last_vol_1m_   = 0.0;
  if (!std::isfinite(last_spread_bp_)) last_spread_bp_ = cfg_.default_spread_bp;
}

bool FeatureBuilder::ready() const {
  // Require the slowest indicator to be ready (SMA slow & RSI & vol window)
  bool sma_ok = sma_slow_.full();
  bool rsi_ok = rsi_.ready();
  bool vol_ok = vol_rtn_.full();
  // ret_5m needs at least 5+1 bars, covered by SMA slow usually; but check anyway
  bool r5_ok  = close_q_.size() >= (size_t)cfg_.ret_5m_window+1;
  return sma_ok && rsi_ok && vol_ok && r5_ok
      && finite(last_ret_1m_) && finite(last_ret_5m_) && finite(last_rsi_)
      && finite(last_sma_fast_) && finite(last_sma_slow_) && finite(last_vol_1m_) && finite(last_spread_bp_);
}

std::optional<std::vector<double>> FeatureBuilder::build() const {
  if (!ready()) return std::nullopt;
  std::vector<double> out; out.reserve(plan_.names.size());

  for (const auto& name : plan_.names) {
    if (name=="ret_1m")      out.push_back(last_ret_1m_);
    else if (name=="ret_5m") out.push_back(last_ret_5m_);
    else if (name=="rsi_14") out.push_back(last_rsi_);
    else if (name=="sma_10") out.push_back(last_sma_fast_);
    else if (name=="sma_30") out.push_back(last_sma_slow_);
    else if (name=="vol_1m") out.push_back(last_vol_1m_);
    else if (name=="spread_bp") out.push_back(last_spread_bp_);
    else {
      // Unknown feature: fail closed so you'll notice in tests
      throw std::runtime_error("FeatureBuilder: unsupported feature name: " + name);
    }
  }
  return out;
}

} // namespace sentio
