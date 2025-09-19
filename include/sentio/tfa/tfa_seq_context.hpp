#pragma once
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <cstring>

#include "sentio/feature/feature_from_spec.hpp"
#include "sentio/feature/column_projector.hpp"
#include "sentio/feature/sanitize.hpp"

namespace sentio {

struct TfaSeqContext {
  torch::jit::Module model;  nlohmann::json spec, meta;
  std::vector<std::string> runtime_names, expected_names;
  int F{55}, T{64}, emit_from{64}; float pad_value{0.f};

  static nlohmann::json load_json(const std::string& p){ std::ifstream f(p); nlohmann::json j; f>>j; return j; }
  static std::vector<std::string> names_from_spec(const nlohmann::json& spec){
    std::vector<std::string> out; out.reserve(spec["features"].size());
    for (auto& f: spec["features"]){
      if (f.contains("name")) out.push_back(f["name"].get<std::string>());
      else {
        std::string op=f["op"].get<std::string>(), src=f.value("source","");
        std::string w=f.contains("window")?std::to_string((int)f["window"]):"";
        std::string k=f.contains("k")?std::to_string((float)f["k"]):"";
        out.push_back(op+"_"+src+"_"+w+"_"+k);
      }
    }
    return out;
  }

  void load(const std::string& model_pt, const std::string& spec_json, const std::string& meta_json){
    model = torch::jit::load(model_pt, torch::kCPU); model.eval();
    spec  = load_json(spec_json);
    meta  = load_json(meta_json);

    runtime_names  = names_from_spec(spec);
    
    // Handle both old (v1) and new (v2_m4_optimized) metadata formats
    if (meta.contains("expects")) {
      // Old format (v1)
      expected_names = meta["expects"]["feature_names"].get<std::vector<std::string>>();
      F         = meta["expects"]["input_dim"].get<int>();
      if (meta["expects"].contains("seq_len")) T = meta["expects"]["seq_len"].get<int>();
      emit_from = meta["expects"]["emit_from"].get<int>();
      pad_value = meta["expects"]["pad_value"].get<float>();
    } else {
      // New format (v2_m4_optimized)
      F = meta["feature_count"].get<int>();
      T = meta["sequence_length"].get<int>();
      // For new format, use runtime names as expected names
      expected_names = runtime_names;
      emit_from = T; // Use sequence length as emit_from
      pad_value = 0.0f; // Default pad value
    }

    if (F!=55) std::cerr << "[WARN] model F="<<F<<" expected 55\n";
  }

  template<class Bars>
  void forward_probs(const std::string& symbol, const Bars& bars, std::vector<float>& probs_out)
  {
    // Build features [N,F]
    auto X = sentio::build_features_from_spec_json(symbol, bars, spec.dump());
    // Project if needed
    std::vector<float> Xproj;
    const float* Xp = X.data.data(); int64_t Fs = X.cols;
    if (!(Fs==F && runtime_names==expected_names)){
      auto proj = sentio::ColumnProjector::make(runtime_names, expected_names, pad_value);
      proj.project(X.data.data(), (size_t)X.rows, (size_t)X.cols, Xproj);
      Xp = Xproj.data(); Fs = F;
    }

    // Sanitize
    auto ready = sentio::sanitize_and_ready(const_cast<float*>(Xp), X.rows, Fs, emit_from, pad_value);

    // Slide windows → batch inference
    probs_out.assign((size_t)X.rows, 0.5f);
    torch::NoGradGuard ng; torch::InferenceMode im;
    const int64_t B = 256;
    const int64_t start = std::max<int64_t>({emit_from, T-1});
    const int64_t last  = X.rows - 1;

    for (int64_t i=start; i<=last; ){
      int64_t j = std::min<int64_t>(last+1, i+B);
      int64_t L = j - i;
      auto t = torch::empty({L, T, F}, torch::kFloat32);
      float* dst = t.data_ptr<float>();
      for (int64_t k=0;k<L;++k){
        int64_t idx=i+k, lo=idx-T+1;
        std::memcpy(dst + k*T*F, Xp + lo*F, sizeof(float)*(size_t)(T*F));
      }
      auto y = model.forward({t}).toTensor(); // [L,1] logits
      if (y.dim()==2 && y.size(1)==1) y=y.squeeze(1);
      float* lp = y.contiguous().data_ptr<float>();
      for (int64_t k=0;k<L;++k)
        probs_out[(size_t)(i+k)] = 1.f/(1.f+std::exp(-lp[k])); // sigmoid
      i = j;
    }

    // Stats
    float pmin=1.f, pmax=0.f, ps=0.f; int64_t cnt=0;
    for (int64_t i=start;i<(int64_t)probs_out.size();++i){ pmin=std::min(pmin,probs_out[i]); pmax=std::max(pmax,probs_out[i]); ps+=probs_out[i]; cnt++; }
  }
};

} // namespace sentio


