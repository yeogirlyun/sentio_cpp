#pragma once
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "sentio/feature/column_projector.hpp"

namespace kochi {

struct KochiBinaryContext {
  torch::jit::Module model;
  int Fk{0};
  int T{0};
  int emit_from{0};
  float pad{0.f};
  std::vector<std::string> expected_names;
  sentio::ColumnProjector projector;

  static nlohmann::json jload(const std::string &p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("missing json: " + p);
    nlohmann::json j;
    f >> j;
    return j;
  }

  void load(const std::string &model_pt, const std::string &meta_json, const std::vector<std::string> &runtime_names) {
    model = torch::jit::load(model_pt, torch::kCPU);
    model.eval();
    auto meta = jload(meta_json);
    auto ex = meta["expects"];
    Fk = ex["input_dim"].get<int>();
    T = ex["seq_len"].get<int>();
    emit_from = ex["emit_from"].get<int>();
    pad = ex["pad_value"].get<float>();
    expected_names = ex["feature_names"].get<std::vector<std::string>>();
    projector = sentio::ColumnProjector::make(runtime_names, expected_names, pad);
  }

  void forward(const float *Xsrc,
               int64_t N,
               int64_t F_runtime,
               std::vector<float> &p_up_out,
               const std::string &audit_jsonl = "") {
    std::vector<float> X;
    projector.project(Xsrc, (size_t)N, (size_t)F_runtime, X);
    const float *Xp = X.data();

    p_up_out.assign((size_t)N, 0.5f);

    torch::NoGradGuard ng;
    torch::InferenceMode im;
    const int64_t start = std::max<int64_t>(emit_from, T);
    const int64_t last = N - 1;
    const int64_t B = 256;

    std::ofstream audit;
    if (!audit_jsonl.empty()) audit.open(audit_jsonl);

    for (int64_t i = start; i <= last;) {
      int64_t L = std::min<int64_t>(B, last - i + 1);
      auto t = torch::empty({L, T, Fk}, torch::kFloat32);
      float *dst = t.data_ptr<float>();
      for (int64_t k = 0; k < L; ++k) {
        int64_t end = i + k, lo = end - T;
        std::memcpy(dst + k * T * Fk, Xp + lo * Fk, sizeof(float) * (size_t)(T * Fk));
      }
      auto y = model.forward({t}).toTensor();
      if (y.dim() == 2 && y.size(1) == 1) y = y.squeeze(1);
      auto logits = y.contiguous().data_ptr<float>();
      for (int64_t k = 0; k < L; ++k) {
        float lg = logits[k];
        float p = 1.f / (1.f + std::exp(-lg));
        size_t idx = (size_t)(i + k);
        p_up_out[idx] = p;
        if (audit.is_open()) {
          audit << "{\"i\":" << idx << ",\"logit\":" << lg << ",\"p_up\":" << p << "}\n";
        }
      }
      i += L;
    }
  }
};

} // namespace kochi


