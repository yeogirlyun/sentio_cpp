#pragma once
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <cstring>
#include <algorithm>

#include "sentio/feature/feature_provider.hpp"
#include "sentio/feature/column_projector.hpp"

namespace kochi {

struct KochiPPOContext {
  torch::jit::Module actor;         // TorchScript; expects [B,T,Fk]; outputs logits [B,A]
  int Fk{0}, T{0}, A{0}, emit_from{0}; float pad{0.f};
  std::vector<std::string> expected_names;
  sentio::ColumnProjector projector;

  static nlohmann::json jload(const std::string& p){
    std::ifstream f(p); if(!f) throw std::runtime_error("missing json: " + p);
    nlohmann::json j; f>>j; return j;
  }

  void load(const std::string& actor_pt, const std::string& meta_json,
            const std::vector<std::string>& runtime_names){
    actor = torch::jit::load(actor_pt, torch::kCPU); actor.eval();
    auto meta = jload(meta_json);
    T   = meta["expects"]["seq_len"].get<int>();
    Fk  = meta["expects"]["input_dim"].get<int>();
    A   = meta["expects"]["num_actions"].get<int>();
    emit_from = meta["expects"]["emit_from"].get<int>();
    pad = meta["expects"]["pad_value"].get<float>();
    expected_names = meta["expects"]["feature_names"].get<std::vector<std::string>>();
    if ((int)expected_names.size()!=Fk) throw std::runtime_error("meta names vs input_dim mismatch");
    projector = sentio::ColumnProjector::make(runtime_names, expected_names, pad);
  }

  // Perform batched sliding-window inference; write actions & probs
  void forward(const sentio::FeatureMatrix& Xsrc,
               std::vector<int>& actions,
               std::vector<std::array<float,3>>& probs, // assuming 3 actions
               const std::string& audit_path = "")
  {
    if (Xsrc.cols <= 0 || Xsrc.rows < T) throw std::runtime_error("insufficient features");
    // 1) Align columns
    std::vector<float> X; projector.project(Xsrc.data.data(), (size_t)Xsrc.rows, (size_t)Xsrc.cols, X);
    const float* Xp = X.data(); const int64_t N = Xsrc.rows;

    // 2) Slide windows and infer
    actions.assign((size_t)N, 0);
    probs.assign((size_t)N, {0.f,0.f,0.f});

    torch::NoGradGuard ng; torch::InferenceMode im;
    const int64_t start = std::max(emit_from, T);   // require full window
    const int64_t last  = N - 1;
    const int64_t B     = 256;

    std::ofstream audit;
    if (!audit_path.empty()) audit.open(audit_path);

    for (int64_t i=start; i<=last; ){
      int64_t L = std::min<int64_t>(B, last - i + 1);
      auto t = torch::empty({L, T, Fk}, torch::kFloat32);
      float* dst = t.data_ptr<float>();
      for (int64_t k=0;k<L;++k){
        int64_t end=i+k, lo=end-T;
        std::memcpy(dst + k*T*Fk, Xp + lo*Fk, sizeof(float)*(size_t)(T*Fk));
      }
      auto logits = actor.forward({t}).toTensor(); // [L, A]
      auto p = torch::softmax(logits, 1).contiguous(); // [L,A]
      auto acc = p.accessor<float,2>();

      for (int64_t k=0;k<L;++k){
        int a=0; float best=-1e9f;
        for (int j=0;j<A;++j){ float pj=acc[k][j]; if (pj>best){best=pj; a=j;} }
        actions[(size_t)(i+k)] = a;
        std::array<float,3> pr{0.f,0.f,0.f};
        for (int j=0;j<std::min(A,3);++j) pr[j]=acc[k][j];
        probs[(size_t)(i+k)] = pr;

        if (audit.is_open()){
          nlohmann::json rec = {
            {"i", (int)(i+k)},
            {"action", a},
            {"probs", {pr[0], pr[1], pr[2]}}
          };
          audit << rec.dump() << "\n";
        }
      }
      i += L;
    }
  }
};

} // namespace kochi
