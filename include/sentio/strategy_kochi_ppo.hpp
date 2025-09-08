#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/ml/model_registry.hpp"
#include "sentio/ml/feature_window.hpp"
#include <optional>

namespace sentio {

struct KochiPPOCfg {
  std::string artifacts_dir{"artifacts"};
  std::string model_id{"KochiPPO"};
  std::string version{"v1"};
  bool use_cuda{false};
  double conf_floor{0.05};
};

class KochiPPOStrategy final : public BaseStrategy {
public:
  KochiPPOStrategy();
  explicit KochiPPOStrategy(const KochiPPOCfg& cfg);

  // BaseStrategy interface
  ParameterMap get_default_params() const override;
  ParameterSpace get_param_space() const override;
  void apply_params() override;
  StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;

  // Feed one bar worth of raw features (metadata order, length must match)
  void set_raw_features(const std::vector<double>& raw);

private:
  KochiPPOCfg cfg_;
  ml::ModelHandle handle_;
  ml::FeatureWindow window_;
  std::optional<StrategySignal> last_;

  StrategySignal map_output(const ml::ModelOutput& mo) const;
};

} // namespace sentio


