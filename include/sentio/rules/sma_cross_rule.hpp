#pragma once
#include "irule.hpp"
#include <vector>
#include <deque>
#include <algorithm>

namespace sentio::rules {

struct SMACrossRule : IRuleStrategy {
  int fast{10}, slow{20};
  std::vector<double> sma_f_, sma_s_;
  const char* name() const override { return "SMA_CROSS"; }
  int warmup() const override { return std::max(fast, slow); }

  std::optional<RuleOutput> eval(const BarsView& b, int64_t i) override {
    if ((int)sma_f_.size()!=b.n) build_(b);
    if (i < warmup()) return std::nullopt;
    int sig = (sma_f_[i]>sma_s_[i]) ? +1 : (sma_f_[i]<sma_s_[i] ? -1 : 0);
    return RuleOutput{std::nullopt, sig, std::nullopt, 0.6f};
  }

  static void roll_sma_(const double* x, int n, int w, std::vector<double>& out){
    out.assign(n,0.0);
    double s=0; for(int i=0;i<n;i++){ s+=x[i]; if(i>=w) s-=x[i-w]; out[i]=(i>=w-1)? s/w : (i>0? out[i-1]:x[0]); }
  }
  void build_(const BarsView& b){
    sma_f_.assign(b.n,0.0); sma_s_.assign(b.n,0.0);
    roll_sma_(b.close,b.n,fast,sma_f_); roll_sma_(b.close,b.n,slow,sma_s_);
  }
};

} // namespace sentio::rules


