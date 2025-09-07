#pragma once
#include "sentio/core.hpp"
#include <deque>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <optional>
#include <cmath>
#include <stdexcept>

namespace sentio {

// Optional microstructure snapshot (pass when you have it; else omit)
struct MicroTick { double bid{NAN}, ask{NAN}; };

// A compiled plan for the features (from metadata)
struct FeaturePlan {
  std::vector<std::string> names; // in metadata order
  // sanity: names.size() must match metadata.feature_names
};

// Builder config
struct FeatureBuilderCfg {
  int rsi_period{14};
  int sma_fast{10};
  int sma_slow{30};
  int ret_5m_window{5};      // number of 1m bars
  // If you want volatility as stdev of 1m returns over N bars, set vol_window>1
  int vol_window{20};        // stdev window (bars)
  // spread fallback (bps) when no bid/ask and proxy not computable
  double default_spread_bp{1.5};
};

// Rolling helpers (small, header-only for speed)
class RollingMean {
  double sum_{0}; std::deque<double> q_;
  size_t W_;
public:
  explicit RollingMean(size_t W): W_(W) {}
  void push(double x){ sum_ += x; q_.push_back(x); if(q_.size()>W_){ sum_-=q_.front(); q_.pop_front(); } }
  bool full() const { return q_.size()==W_; }
  double mean() const { return q_.empty()? NAN : (sum_/double(q_.size())); }
  size_t size() const { return q_.size(); }
};

class RollingStdWindow {
  std::vector<double> buf_;
  size_t W_, i_{0}, n_{0};
  double sum_{0}, sumsq_{0};
public:
  explicit RollingStdWindow(size_t W): buf_(W, 0.0), W_(W) {}
  inline void push(double x){
    if (n_ < W_) { 
      buf_[n_++] = x; 
      sum_ += x; 
      sumsq_ += x*x; 
      if (n_ == W_) i_ = 0; 
    } else { 
      double old = buf_[i_]; 
      buf_[i_] = x; 
      sum_ += x - old; 
      sumsq_ += x*x - old*old; 
      if (++i_ == W_) i_ = 0; 
    }
  }
  inline bool full() const { return n_ == W_; }
  inline double stdev() const { 
    if (n_ < 2) return NAN; 
    double m = sum_ / n_; 
    return std::sqrt(std::max(0.0, sumsq_ / n_ - m * m)); 
  }
  inline size_t size() const { return n_; }
};

class RollingRSI {
  // Wilder's RSI with smoothing; requires first 'period' values to bootstrap
  int period_; bool boot_{true}; int boot_count_{0};
  double up_{0}, dn_{0};
public:
  explicit RollingRSI(int p): period_(p) {}
  // x = current close, px = previous close
  void push(double px, double x){
    double chg = x - px;
    double u = chg>0? chg:0; double d = chg<0? -chg:0;
    if (boot_){
      up_ += u; dn_ += d; ++boot_count_;
      if (boot_count_ == period_) {
        up_ /= period_; dn_ /= period_; boot_ = false;
      }
    } else {
      up_ = (up_*(period_-1) + u) / period_;
      dn_ = (dn_*(period_-1) + d) / period_;
    }
  }
  bool ready() const { return !boot_; }
  double value() const {
    if (boot_) return NAN;
    if (dn_==0) return 100.0;
    double rs = up_/dn_;
    return 100.0 - 100.0/(1.0+rs);
  }
};

class FeatureBuilder {
public:
  FeatureBuilder(FeaturePlan plan, FeatureBuilderCfg cfg);

  // Feed one 1m bar (RTH-filtered) plus optional bid/ask for spread
  void on_bar(const Bar& b, const std::optional<MicroTick>& mt = std::nullopt);

  // True when all requested features can be computed *and* are finite
  bool ready() const;

  // Returns features in the exact metadata order (size == plan.names.size()).
  // Will return std::nullopt if not ready().
  std::optional<std::vector<double>> build() const;

  // Resets internal buffers
  void reset();

  // Accessors (useful in tests)
  size_t bars_seen() const { return bars_seen_; }

private:
  FeaturePlan plan_;
  FeatureBuilderCfg cfg_;

  // Internal state
  size_t bars_seen_{0};
  std::deque<double> close_q_;             // last N closes for ret/RSI
  RollingMean sma_fast_, sma_slow_;
  RollingStdWindow vol_rtn_;               // stdev of 1m returns (O(1) implementation)
  RollingRSI  rsi_;

  // Cached per-bar computations
  double last_ret_1m_{NAN};
  double last_ret_5m_{NAN};
  double last_rsi_{NAN};
  double last_sma_fast_{NAN};
  double last_sma_slow_{NAN};
  double last_vol_1m_{NAN};
  double last_spread_bp_{NAN};

  // helpers
  static inline bool finite(double x){ return std::isfinite(x); }
};

} // namespace sentio
