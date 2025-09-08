#pragma once
#include "irule.hpp"
#include <vector>
#include <deque>
#include <cmath>

namespace sentio::rules {

struct BBandsSqueezeBreakoutRule : IRuleStrategy {
  int win{20}; double k{2.0}; double squeeze_thr{0.8};
  std::vector<double> ma_, sd_, upper_, lower_, bw_, med_bw_;
  const char* name() const override { return "BBANDS_SQUEEZE_BRK"; }
  int warmup() const override { return win+1; }

  std::optional<RuleOutput> eval(const BarsView& b, int64_t i) override {
    if ((int)ma_.size()!=b.n) build_(b);
    if (i < warmup()) return std::nullopt;
    bool squeeze = (bw_[i] < squeeze_thr * med_bw_[i]);
    int sig = 0;
    if (squeeze){
      if (b.close[i] > upper_[i]) sig = +1;
      else if (b.close[i] < lower_[i]) sig = -1;
    }
    return RuleOutput{std::nullopt, sig, std::nullopt, 0.7f};
  }

  void build_(const BarsView& b){
    int N=b.n; ma_.assign(N,0); sd_.assign(N,0); upper_.assign(N,0); lower_.assign(N,0); bw_.assign(N,0); med_bw_.assign(N,0);
    std::deque<double> q; double s=0,s2=0; std::deque<double> bwq;
    for(int i=0;i<N;i++){
      q.push_back(b.close[i]); s+=b.close[i]; s2+=b.close[i]*b.close[i];
      if((int)q.size()>win){ double z=q.front(); q.pop_front(); s-=z; s2-=z*z; }
      if ((int)q.size()==win){
        double m=s/win, v=std::max(0.0, s2/win - m*m), sd=std::sqrt(v);
        ma_[i]=m; sd_[i]=sd; upper_[i]=m+k*sd; lower_[i]=m-k*sd; bw_[i]=(upper_[i]-lower_[i])/(m+1e-9);
      } else { ma_[i]=(i?ma_[i-1]:b.close[0]); sd_[i]=(i?sd_[i-1]:0.0); upper_[i]=(i?upper_[i-1]:b.close[0]); lower_[i]=(i?lower_[i-1]:b.close[0]); bw_[i]=(i?bw_[i-1]:0.0); }
      if ((int)bwq.size()==win) bwq.pop_front(); bwq.push_back(bw_[i]);
      double m=0; for(double x:bwq) m+=x; med_bw_[i]=(bwq.empty()? bw_[i] : m/bwq.size());
    }
  }
};

} // namespace sentio::rules


