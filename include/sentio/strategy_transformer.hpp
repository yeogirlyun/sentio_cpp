#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/ml/model_registry.hpp"
#include "sentio/ml/feature_window.hpp"
#include <optional>
#include <memory>

namespace sentio {

struct TransformerCfg {
  std::string artifacts_dir{"artifacts"};
  std::string model_id{"TransAlpha"};
  std::string version{"v1"};
  double conf_floor{0.05};
};

class TransformerSignalStrategy final : public BaseStrategy {
public:
  TransformerSignalStrategy(); // Default constructor for factory
  explicit TransformerSignalStrategy(const TransformerCfg& cfg);

  // BaseStrategy interface
  ParameterMap get_default_params() const override;
  ParameterSpace get_param_space() const override;
  void apply_params() override;
  StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;

  // Push one feature vector per bar, metadata order
  void set_raw_features(const std::vector<double>& raw);

private:
  TransformerCfg cfg_;
  ml::ModelHandle handle_;
  ml::FeatureWindow window_;
  std::optional<StrategySignal> last_;

  StrategySignal map_output(const ml::ModelOutput& mo) const;
  ml::WindowSpec make_window_spec(const ml::ModelSpec& spec) const;
};

} // namespace sentio
