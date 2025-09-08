#pragma once
#include "irule.hpp"
#include <vector>
#include <deque>
#include <cmath>

namespace sentio::rules {

struct MomentumVolumeRule : IRuleStrategy {
  int mom_win{10}, vol_win{20}; double vol_z{0.0};
  std::vector<double> mom_, vol_ma_, vol_sd_;
  const char* name() const override { return "MOMENTUM_VOLUME"; }
  int warmup() const override { return std::max(mom_win, vol_win)+1; }

  std::optional<RuleOutput> eval(const BarsView& b, int64_t i) override {
    if ((int)mom_.size()!=b.n) build_(b);
    if (i < warmup()) return std::nullopt;
    double mz = (b.volume[i]-vol_ma_[i])/(vol_sd_[i]+1e-9);
    int sig = 0;
    if (mom_[i]>0 && mz>=vol_z) sig=+1;
    else if (mom_[i]<0 && mz>=vol_z) sig=-1;
    return RuleOutput{std::nullopt, sig, (float)mom_[i], 0.6f};
  }

  void build_(const BarsView& b){
    int N=b.n; mom_.assign(N,0); vol_ma_.assign(N,0); vol_sd_.assign(N,1);
    std::vector<double> logc(N,0); logc[0]=std::log(std::max(1e-12,b.close[0]));
    for(int i=1;i<N;i++) logc[i]=std::log(std::max(1e-12,b.close[i]));
    for(int i=0;i<N;i++){ int j=std::max(0,i-mom_win); mom_[i]=logc[i]-logc[j]; }
    std::deque<double> q; double s=0,s2=0;
    for(int i=0;i<N;i++){
      q.push_back(b.volume[i]); s+=b.volume[i]; s2+=b.volume[i]*b.volume[i];
      if((int)q.size()>vol_win){ double z=q.front(); q.pop_front(); s-=z; s2-=z*z; }
      if (!q.empty()){
        double m=s/q.size(); double v=std::max(0.0, s2/q.size() - m*m);
        vol_ma_[i]=m; vol_sd_[i]=std::sqrt(v);
      }
    }
  }
};

} // namespace sentio::rules


