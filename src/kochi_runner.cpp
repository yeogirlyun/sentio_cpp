#include "sentio/feature/csv_feature_provider.hpp"
#include "kochi/kochi_ppo_context.hpp"
#include <filesystem>
#include <iostream>

namespace kochi {

int run_kochi_ppo(const std::string& symbol,
                  const std::string& feature_csv,
                  const std::string& actor_pt,
                  const std::string& actor_meta_json,
                  const std::string& audit_dir,
                  int kochi_T,
                  int cooldown_bars = 5)
{
  sentio::CsvFeatureProvider provider(feature_csv, /*T=*/kochi_T);
  auto X = provider.get_features_for(symbol);
  auto runtime_names = provider.feature_names();

  KochiPPOContext ctx;
  ctx.load(actor_pt, actor_meta_json, runtime_names);

  std::filesystem::create_directories(audit_dir);
  long long ts_epoch = std::time(nullptr);
  std::string audit_file = audit_dir + "/kochi_ppo_temporal_" + std::to_string(ts_epoch) + ".jsonl";

  std::vector<int> actions;
  std::vector<std::array<float,3>> probs;
  ctx.forward(X, actions, probs, audit_file);

  // Simple execution: position = {-1,0,+1}
  int pos = 0; int64_t emitted=0, considered=0; int cooldown_drops=0;
  const size_t start = (size_t)std::max(ctx.emit_from, ctx.T);
  int64_t next_ok = 0;

  for (size_t i=start;i<actions.size();++i){
    considered++;
    int a = actions[i]; // 0=HOLD 1=LONG 2=SHORT
    if ((int64_t)i < next_ok){ cooldown_drops++; continue; }
    int target = (a==1) ? +1 : (a==2 ? -1 : pos);
    if (target != pos){
      pos = target;
      emitted++;
      next_ok = (int64_t)i + cooldown_bars;
    }
  }

  std::cerr << "[KOCHI] considered="<<considered<<" emitted="<<emitted
            << " cooldown_drops="<<cooldown_drops
            << " T="<<ctx.T<<" Fk="<<ctx.Fk<<"\n";
  return 0;
}

} // namespace kochi
