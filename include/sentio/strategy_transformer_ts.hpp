#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/ml/model_registry.hpp"
#include "sentio/ml/feature_window.hpp"
#include <optional>

namespace sentio {

struct TransformerTSCfg {
  std::string artifacts_dir{"artifacts"};
  std::string model_id{"TransAlpha"};
  std::string version{"v1"};
  bool use_cuda{false};
  double conf_floor{0.05};
};

class TransformerSignalStrategyTS final : public IStrategy {
public:
  explicit TransformerSignalStrategyTS(const TransformerTSCfg& cfg);

  void set_raw_features(const std::vector<double>& raw);
  void on_bar(const StrategyCtx& ctx, const Bar& b) override;
  std::optional<StrategySignal> latest() const override { return last_; }

private:
  TransformerTSCfg cfg_;
  ml::ModelHandle handle_;
  ml::FeatureWindow window_;
  std::optional<StrategySignal> last_;
  StrategySignal map_output(const ml::ModelOutput& mo) const;
};

} // namespace sentio
