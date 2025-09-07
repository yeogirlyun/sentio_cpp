#pragma once
#include "sentio/ml/iml_model.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace sentio::tfa {

struct DropCounters {
  int64_t total{0}, not_ready{0}, nan_row{0}, low_conf{0}, session{0}, volume{0}, cooldown{0}, dup{0};
  void log() const {
    std::cout << "[SIG TFA] total="<<total
              << " not_ready="<<not_ready
              << " nan_row="<<nan_row
              << " low_conf="<<low_conf
              << " session="<<session
              << " volume="<<volume
              << " cooldown="<<cooldown
              << " dup="<<dup << std::endl;
  }
};

struct ThresholdPolicy {
  // Either fixed threshold or rolling quantile
  float min_prob = 0.55f;      // fixed
  int   q_window = 0;          // if >0, use quantile
  float q_level  = 0.75f;      // 75th percentile

  std::vector<uint8_t> filter(const std::vector<float>& prob) const {
    const int64_t N = (int64_t)prob.size();
    std::vector<uint8_t> keep(N, 0);
    if (q_window <= 0){
      for (int64_t i=0;i<N;++i) keep[i] = (prob[i] >= min_prob) ? 1 : 0;
      return keep;
    }
    // rolling quantile
    std::vector<float> win; win.reserve(q_window);
    for (int64_t i=0;i<N;++i){
      int64_t a = std::max<int64_t>(0, i - q_window + 1);
      win.clear();
      for (int64_t j=a;j<=i;++j) win.push_back(prob[j]);
      std::nth_element(win.begin(), win.begin() + (int)(q_level*(win.size()-1)), win.end());
      float thr = win[(int)(q_level*(win.size()-1))];
      keep[i] = (prob[i] >= thr) ? 1 : 0;
    }
    return keep;
  }
};

struct Cooldown {
  int bars = 0; // e.g., 5
  // returns mask where entry allowed, tracking last accepted index
  std::vector<uint8_t> apply(const std::vector<uint8_t>& keep) const {
    if (bars <= 0) return keep;
    std::vector<uint8_t> out(keep.size(), 0);
    int64_t next_ok = 0;
    for (int64_t i=0;i<(int64_t)keep.size(); ++i){
      if (i < next_ok) continue;
      if (keep[i]){ out[i]=1; next_ok = i + bars; }
    }
    return out;
  }
};

// Minimal session & volume filters, customize to your data
struct RowFilters {
  bool rth_only = false;
  double min_volume = 0.0;
  std::vector<uint8_t> session_mask; // 1 if allowed (precomputed per row)
  std::vector<double>  volumes;      // per row

  void ensure_sizes(int64_t N){
    if ((int64_t)session_mask.size()!=N) session_mask.assign(N,1);
    if ((int64_t)volumes.size()!=N) volumes.assign(N, 1.0);
  }
};

struct TfaSignalPipeline {
  ml::IModel* model{nullptr}; // Model interface, returns score/prob
  ThresholdPolicy policy;
  Cooldown cooldown;
  RowFilters rowf;

  struct Result {
    std::vector<uint8_t> emit;     // 1=emit entry
    std::vector<float>   prob;     // model output (after activation)
    DropCounters drops;
  };

  static inline float sigmoid(float x){ return 1.f / (1.f + std::exp(-x)); }

  Result run(float* X, int64_t rows, int64_t cols,
             const std::vector<uint8_t>& ready_mask,
             bool model_outputs_logit=true)
  {
    Result R;
    R.prob.assign(rows, 0.f);
    R.emit.assign(rows, 0);
    R.drops.total = rows;

    // 1) forward model using IModel interface
    for (int64_t i=0; i<rows; ++i){
      std::vector<float> features(cols);
      for (int64_t j=0; j<cols; ++j) {
        features[j] = X[i*cols + j];
      }
      
      auto output = model->predict(features, 1, cols, "BF"); // Single row prediction
      if (output && !output->probs.empty()) {
        float v = output->probs[0]; // Assume single output probability
        R.prob[i] = model_outputs_logit ? sigmoid(v) : v;
      } else if (output) {
        float v = output->score; // Fallback to score
        R.prob[i] = model_outputs_logit ? sigmoid(v) : v;
      } else {
        R.prob[i] = 0.5f; // Default neutral probability
      }
    }

    // 2) ready vs not_ready
    std::vector<uint8_t> keep(rows, 0);
    for (int64_t i=0;i<rows;++i){
      if (!ready_mask[i]) { R.drops.not_ready++; continue; }
      keep[i] = 1;
    }

    // 3) session / volume
    rowf.ensure_sizes(rows);
    for (int64_t i=0;i<rows;++i){
      if (!keep[i]) continue;
      if (!rowf.session_mask[i]){ keep[i]=0; R.drops.session++; continue; }
      if (rowf.volumes[i] <= rowf.min_volume){ keep[i]=0; R.drops.volume++; continue; }
    }

    // 4) thresholding
    auto conf_keep = policy.filter(R.prob);
    for (int64_t i=0;i<rows;++i){
      if (!keep[i]) continue;
      if (!conf_keep[i]){ keep[i]=0; R.drops.low_conf++; }
    }

    // 5) cooldown
    keep = cooldown.apply(keep);
    // count cooldown drops (approx)
    // (We can estimate: entries removed between pre/post)
    // For transparency, compute:
    {
      int64_t pre=0, post=0;
      for (auto v: conf_keep) if (v) pre++;
      for (auto v: keep) if (v) post++;
      R.drops.cooldown += std::max<int64_t>(0, pre - post);
    }

    R.emit = std::move(keep);
    return R;
  }
  
  // Overload for vector<vector<double>> from cached features
  Result run_cached(const std::vector<std::vector<double>>& features,
                    const std::vector<uint8_t>& ready_mask,
                    bool model_outputs_logit=true)
  {
    const int64_t rows = features.size();
    if (rows == 0) {
      Result R;
      R.drops.total = 0;
      return R;
    }
    
    const int64_t cols = features[0].size();
    if (cols == 0) {
      Result R;
      R.drops.total = rows;
      R.drops.nan_row = rows;
      return R;
    }
    
    // Safety check: ensure all rows have same column count
    for (int64_t r = 0; r < rows; ++r) {
      if ((int64_t)features[r].size() != cols) {
        std::cout << "[ERROR] TfaSignalPipeline: Row " << r << " has " << features[r].size() 
                  << " features, expected " << cols << std::endl;
        Result R;
        R.drops.total = rows;
        R.drops.nan_row = rows;
        return R;
      }
    }
    
    // Convert to float array for model
    std::vector<float> X_flat(rows * cols);
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        X_flat[r * cols + c] = static_cast<float>(features[r][c]);
      }
    }
    
    return run(X_flat.data(), rows, cols, ready_mask, model_outputs_logit);
  }
};

} // namespace sentio::tfa
