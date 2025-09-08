#pragma once
#include "irule.hpp"
#include <vector>
#include <deque>
#include <cmath>

namespace sentio::rules {

struct OFIProxyRule : IRuleStrategy {
  int vol_win{20}; double k{1.0};
  std::vector<double> ofi_;
  const char* name() const override { return "OFI_PROXY"; }
  int warmup() const override { return vol_win+1; }

  std::optional<RuleOutput> eval(const BarsView& b, int64_t i) override {
    if ((int)ofi_.size()!=b.n) build_(b);
    if (i < warmup()) return std::nullopt;
    float s = (float)ofi_[i];
    float p = 1.f/(1.f+std::exp(-k*s));
    return RuleOutput{p, std::nullopt, s, 0.6f};
  }

  void build_(const BarsView& b){
    int N=b.n; ofi_.assign(N,0.0);
    std::vector<double> lr(N,0); for(int i=1;i<N;i++) lr[i]=std::log(std::max(1e-12,b.close[i]))-std::log(std::max(1e-12,b.close[i-1]));
    std::deque<double> q; double s=0,s2=0;
    for(int i=0;i<N;i++){
      double x = lr[i]*b.volume[i];
      q.push_back(x); s+=x; s2+=x*x;
      if((int)q.size()>vol_win){ double z=q.front(); q.pop_front(); s-=z; s2-=z*z; }
      double m = q.empty()?0.0:s/q.size();
      double v = q.empty()?0.0:std::max(0.0, s2/q.size() - m*m);
      ofi_[i] = (v>0? (x-m)/std::sqrt(v) : 0.0);
    }
  }
};

} // namespace sentio::rules


