#pragma once
#include "irule.hpp"
#include <vector>
#include <deque>
#include <cmath>

namespace sentio::rules {

struct VWAPReversionRule : IRuleStrategy {
  int win{20}; double z_lo{-1.0}, z_hi{+1.0};
  std::vector<double> vwap_, sd_;
  const char* name() const override { return "VWAP_REVERSION"; }
  int warmup() const override { return win+1; }

  std::optional<RuleOutput> eval(const BarsView& b, int64_t i) override {
    if ((int)vwap_.size()!=b.n) build_(b);
    if (i < warmup()) return std::nullopt;
    double z=(b.close[i]-vwap_[i])/(sd_[i]+1e-9);
    int sig = (z<=z_lo)? +1 : (z>=z_hi? -1 : 0);
    return RuleOutput{std::nullopt, sig, (float)z, 0.6f};
  }

  void build_(const BarsView& b){
    int N=b.n; vwap_.assign(N,0); sd_.assign(N,1.0);
    std::deque<double> qv,qpv; double sv=0, spv=0; std::deque<double> qdiff; double s2=0;
    for(int i=0;i<N;i++){
      double pv=b.close[i]*b.volume[i];
      qv.push_back(b.volume[i]); qpv.push_back(pv); sv+=b.volume[i]; spv+=pv;
      if((int)qv.size()>win){ sv-=qv.front(); spv-=qpv.front(); qv.pop_front(); qpv.pop_front(); }
      vwap_[i] = (sv>0? spv/sv : b.close[i]);

      double d=b.close[i]-vwap_[i];
      qdiff.push_back(d); s2+=d*d;
      if((int)qdiff.size()>win){ double z=qdiff.front(); qdiff.pop_front(); s2-=z*z; }
      sd_[i] = std::sqrt(std::max(0.0, s2/std::max(1,(int)qdiff.size())));
    }
  }
};

} // namespace sentio::rules


