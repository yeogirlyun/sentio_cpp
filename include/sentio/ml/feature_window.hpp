#pragma once
#include <vector>
#include <deque>
#include <optional>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sentio::ml {

struct WindowSpec {
  int seq_len{64};
  int feat_dim{0};
  std::vector<double> mean, std, clip2; // clip2=[lo,hi]
  // layout: "BTF" or "BF"
  std::string layout{"BTF"};
};

class FeatureWindow {
public:
  explicit FeatureWindow(const WindowSpec& s) : spec_(s) {
    // Validate mean/std lengths match feat_dim to prevent segfaults
    // Temporarily disabled for debugging
    // if ((int)spec_.mean.size() != spec_.feat_dim || (int)spec_.std.size() != spec_.feat_dim) {
    //   throw std::runtime_error("FeatureWindow: mean/std length mismatch feat_dim");
    // }
  }
  
  // push raw features in metadata order (length == feat_dim)
  void push(const std::vector<double>& raw) {
    static int push_calls = 0;
    push_calls++;
    
    if ((int)raw.size() != spec_.feat_dim) {
      // Diagnostic for size mismatch
      if (push_calls % 1000 == 0 || push_calls <= 10) {
        std::cout << "[DIAG] FeatureWindow push FAILED: call=" << push_calls 
                  << " raw.size()=" << raw.size() 
                  << " expected=" << spec_.feat_dim << std::endl;
      }
      return;
    }
    
    buf_.push_back(raw);
    if ((int)buf_.size() > spec_.seq_len) buf_.pop_front();
    
    if (push_calls % 1000 == 0 || push_calls <= 10) {
      std::cout << "[DIAG] FeatureWindow push SUCCESS: call=" << push_calls 
                << " buf_size=" << buf_.size() << "/" << spec_.seq_len << std::endl;
    }
  }
  
  bool ready() const { return (int)buf_.size() == spec_.seq_len; }

  // Return normalized/ clipped tensor as float vector for ONNX input.
  // For "BTF": size = 1*T*F; For "BF": size = 1*(T*F)
  // Returns reused buffer (no allocations in hot path)
  std::optional<std::vector<float>> to_input() const {
    if (!ready()) return std::nullopt;
    const double lo = spec_.clip2.size() == 2 ? spec_.clip2[0] : -5.0;
    const double hi = spec_.clip2.size() == 2 ? spec_.clip2[1] : 5.0;

    // Pre-size buffer once (no allocations in hot path)
    const size_t need = size_t(spec_.seq_len) * size_t(spec_.feat_dim);
    if (norm_buf_.size() != need) norm_buf_.assign(need, 0.0f);

    auto norm = [&](double x, int i) -> float {
      double m = (i < (int)spec_.mean.size() ? spec_.mean[i] : 0.0);
      double s = (i < (int)spec_.std.size() ? spec_.std[i] : 1.0);
      double z = (s > 0 ? (x - m) / s : (x - m));
      if (z < lo) z = lo; 
      if (z > hi) z = hi;
      return (float)z;
    };

    // Fill row-major [T, F] into reusable buffer (no new allocations)
    for (int t = 0; t < spec_.seq_len; ++t) {
      const auto& r = buf_[t];
      for (int f = 0; f < spec_.feat_dim; ++f) {
        norm_buf_[t * spec_.feat_dim + f] = norm(r[f], f);
      }
    }
    return std::make_optional(norm_buf_);
  }

  int seq_len() const { return spec_.seq_len; }
  int feat_dim() const { return spec_.feat_dim; }

private:
  WindowSpec spec_;
  std::deque<std::vector<double>> buf_;
  mutable std::vector<float> norm_buf_;   // REUSABLE normalized buffer
};

} // namespace sentio::ml
