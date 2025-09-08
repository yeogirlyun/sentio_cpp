#pragma once
#include "irule.hpp"
#include <vector>
#include <algorithm>

namespace sentio::rules {

struct OpeningRangeBreakoutRule : IRuleStrategy {
  int or_bars{30}; double thr{0.000};
  std::vector<double> hi_, lo_;
  const char* name() const override { return "OPENING_RANGE_BRK"; }
  int warmup() const override { return or_bars+1; }

  std::optional<RuleOutput> eval(const BarsView& b, int64_t i) override {
    if ((int)hi_.size()!=b.n) build_(b);
    if (i < warmup()) return std::nullopt;
    double hh=hi_[i], ll=lo_[i], px=b.close[i];
    int sig = (px >= hh*(1.0+thr))? +1 : (px <= ll*(1.0-thr)? -1 : 0);
    return RuleOutput{std::nullopt, sig, std::nullopt, 0.7f};
  }

  void build_(const BarsView& b){
    int N=b.n; hi_.assign(N,b.high[0]); lo_.assign(N,b.low[0]);
    double hh = -1e300, ll = 1e300;
    for(int i=0;i<N;i++){
      if (i<or_bars){ hh=std::max(hh,b.high[i]); ll=std::min(ll,b.low[i]); }
      hi_[i]=hh; lo_[i]=ll;
    }
  }
};

} // namespace sentio::rules


