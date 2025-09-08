#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

namespace sentio::rules {

struct DiversityCfg {
  int   window = 512;
  float shrink = 0.10f;
  float min_w  = 0.10f;
  float max_w  = 3.00f;
};

class DiversityWeighter {
public:
  DiversityWeighter(int members, const DiversityCfg& c = {})
  : M_(members), cfg_(c), mean_(members,0.f), var_(members,1e-4f) {}

  void update(const std::vector<float>& p){
    if ((int)p.size()!=M_) return;
    hist_.push_back(p);
    if ((int)hist_.size()>cfg_.window) hist_.erase(hist_.begin());
    for (int k=0;k<M_;++k){
      mean_[k] = 0.f; for (auto& r: hist_) mean_[k]+=r[k]; mean_[k]/=hist_.size();
      float v=0.f; for (auto& r: hist_){ float d=r[k]-mean_[k]; v+=d*d; }
      var_[k] = std::max(v/std::max(1,(int)hist_.size()-1), 1e-6f);
    }
  }

  std::vector<float> weights() const {
    std::vector<float> w(M_, 1.f);
    for (int k=0;k<M_;++k){
      float inv = 1.f / std::max(var_[k], 1e-6f);
      float ws = (1-cfg_.shrink)*inv + cfg_.shrink*1.f;
      w[k] = std::clamp(ws, cfg_.min_w, cfg_.max_w);
    }
    return w;
  }

private:
  int M_;
  DiversityCfg cfg_{};
  std::vector<std::vector<float>> hist_;
  std::vector<float> mean_, var_;
};

} // namespace sentio::rules


