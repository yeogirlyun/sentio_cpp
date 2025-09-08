#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace sentio::rules {

struct PlattCfg {
  int   window = 4096;
  float lr     = 0.02f;
  float l2     = 1e-3f;
  float clip   = 10.f;
};

class OnlinePlatt {
public:
  explicit OnlinePlatt(const PlattCfg& c = {}) : cfg_(c) {}

  void update(float z, float target){
    z = std::clamp(z, -cfg_.clip, cfg_.clip);
    float yhat = sigmoid(a_*z + b_);
    float grad_a = (yhat - target)*z + cfg_.l2*a_;
    float grad_b = (yhat - target)     + cfg_.l2*b_;
    a_ -= cfg_.lr * grad_a;
    b_ -= cfg_.lr * grad_b;
    zs_.push_back(z); ys_.push_back(target);
    if ((int)zs_.size()>cfg_.window){ zs_.erase(zs_.begin()); ys_.erase(ys_.begin()); }
  }

  float calibrate_from_p(float p) const {
    p = std::clamp(p, 1e-6f, 1.f-1e-6f);
    float z = std::log(p/(1.f-p));
    float zc = std::clamp(a_*z + b_, -cfg_.clip, cfg_.clip);
    return sigmoid(zc);
  }

private:
  static float sigmoid(float x){ return 1.f/(1.f+std::exp(-x)); }
  PlattCfg cfg_{};
  float a_{1.f}, b_{0.f};
  std::vector<float> zs_, ys_;
};

} // namespace sentio::rules


