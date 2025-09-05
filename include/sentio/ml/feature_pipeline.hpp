#pragma once
#include "iml_model.hpp"
#include <vector>
#include <string>
#include <optional>
#include <cmath>

namespace sentio::ml {

struct FeaturePipeline {
  // Pre-sized to spec.feature_names.size()
  std::vector<float> buf;

  explicit FeaturePipeline(const ModelSpec& spec)
  : buf(spec.feature_names.size(), 0.0f), spec_(&spec) {}

  // raw must match spec.feature_names order/length
  // Applies (x-mean)/std then clips to [clip_lo, clip_hi]
  // Returns pointer to internal buffer if successful, nullptr if failed
  const std::vector<float>* transform(const std::vector<double>& raw) {
    auto N = spec_->feature_names.size();
    if (raw.size()!=N) return nullptr;
    const double lo = spec_->clip2.size()==2 ? spec_->clip2[0] : -5.0;
    const double hi = spec_->clip2.size()==2 ? spec_->clip2[1] :  5.0;
    for (size_t i=0;i<N;++i) {
      double x = raw[i];
      double m = (i<spec_->mean.size()? spec_->mean[i] : 0.0);
      double s = (i<spec_->std.size()?  spec_->std[i]  : 1.0);
      double z = s>0 ? (x-m)/s : x-m;
      if (z<lo) z=lo; if (z>hi) z=hi;
      buf[i] = static_cast<float>(z);
    }
    return &buf;
  }

private:
  const ModelSpec* spec_;
};

} // namespace sentio::ml
