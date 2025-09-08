#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "kochi/kochi_binary_context.hpp"
#include "sentio/feature/csv_feature_provider.hpp"

struct BinarizedDecision {
  enum Side { HOLD = 0, BUY = 1, SELL = 2 } side;
  float strength; // 0..1
};

static inline BinarizedDecision decide_from_pup(float p, float buy_lo = 0.60f, float buy_hi = 0.75f,
                                                float sell_hi = 0.40f, float sell_lo = 0.25f) {
  if (p >= buy_hi)
    return {BinarizedDecision::BUY, std::min(1.f, (p - buy_hi) / (1.f - buy_hi))};
  if (p >= buy_lo)
    return {BinarizedDecision::BUY, std::min(1.f, (p - buy_lo) / (buy_hi - buy_lo))};
  if (p <= sell_lo)
    return {BinarizedDecision::SELL, std::min(1.f, (sell_lo - p) / sell_lo)};
  if (p <= sell_hi)
    return {BinarizedDecision::SELL, std::min(1.f, (sell_hi - p) / (sell_hi - 0.5f))};
  return {BinarizedDecision::HOLD, 0.f};
}

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "Usage: kochi_bin_runner <symbol> <features_csv> <model_pt> <meta_json> <audit_dir>\n";
    return 1;
  }
  std::string symbol = argv[1];
  std::string features_csv = argv[2];
  std::string model_pt = argv[3];
  std::string meta_json = argv[4];
  std::string audit_dir = argv[5];

  // Read meta.json for T/seq_len
  std::ifstream jf(meta_json);
  if (!jf) {
    std::cerr << "Missing meta json: " << meta_json << "\n";
    return 1;
  }
  nlohmann::json meta; jf >> meta;
  int T = meta["expects"]["seq_len"].get<int>();

  sentio::CsvFeatureProvider provider(features_csv, /*T=*/T);
  auto X = provider.get_features_for(symbol);
  auto runtime_names = provider.feature_names();

  kochi::KochiBinaryContext ctx;
  ctx.load(model_pt, meta_json, runtime_names);

  std::filesystem::create_directories(audit_dir);
  long long ts_epoch = std::time(nullptr);
  std::string audit_file = audit_dir + "/kochi_bin_temporal_" + std::to_string(ts_epoch) + ".jsonl";

  std::vector<float> p_up;
  ctx.forward(X.data.data(), X.rows, X.cols, p_up, audit_file);

  // Map to BUY/SELL/HOLD counts (router thresholds)
  size_t start = (size_t)std::max(ctx.emit_from, ctx.T);
  long long buy_cnt = 0, sell_cnt = 0, hold_cnt = 0;
  for (size_t i = start; i < p_up.size(); ++i) {
    BinarizedDecision d = decide_from_pup(p_up[i]);
    if (d.side == BinarizedDecision::BUY) buy_cnt++;
    else if (d.side == BinarizedDecision::SELL) sell_cnt++;
    else hold_cnt++;
  }

  std::cerr << "[KOCHI-BIN] bars=" << p_up.size() << " start=" << start
            << " buy=" << buy_cnt << " sell=" << sell_cnt << " hold=" << hold_cnt
            << " T=" << ctx.T << " Fk=" << ctx.Fk << "\n";

  return 0;
}


