#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/ml/model_registry.hpp"
#include "sentio/ml/feature_pipeline.hpp"
#include <memory>

namespace sentio {

struct HybridPPOCfg {
  std::string artifacts_dir{"artifacts"};
  std::string version{"v1"};
  double conf_floor{0.05}; // gate safety: below -> no signal
};

class HybridPPOStrategy final : public BaseStrategy {
public:
  HybridPPOStrategy(); // Default constructor for factory
  explicit HybridPPOStrategy(const HybridPPOCfg& cfg);

  // BaseStrategy interface
  ParameterMap get_default_params() const override;
  ParameterSpace get_param_space() const override;
  void apply_params() override;
  StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;

  // Feed features (same order/length as metadata feature_names).
  void set_raw_features(const std::vector<double>& raw);

private:
  HybridPPOCfg cfg_;
  std::optional<StrategySignal> last_;
  ml::ModelHandle handle_;
  ml::FeaturePipeline fpipe_;
  std::vector<double> raw_;

  StrategySignal map_output(const ml::ModelOutput& mo) const;
};

} // namespace sentio
