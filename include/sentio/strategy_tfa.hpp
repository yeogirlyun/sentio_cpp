#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/ml/model_registry.hpp"
#include "sentio/ml/feature_window.hpp"
#include "sentio/feature/column_projector.hpp"
#include "sentio/feature/column_projector_safe.hpp"
#include "sentio/tfa/tfa_seq_context.hpp"
#include "sentio/router.hpp"
#include "sentio/sizer.hpp"
#include <optional>
#include <memory>

namespace sentio {

struct TFACfg {
  std::string artifacts_dir{"artifacts"};
  std::string model_id{"TFA"};
  std::string version{"v1"};
  bool use_cuda{false};
  double conf_floor{0.05};
};

class TFAStrategy final : public BaseStrategy {
public:
  TFAStrategy(); // Default constructor for factory
  explicit TFAStrategy(const TFACfg& cfg);

  void set_raw_features(const std::vector<double>& raw);
  void on_bar(const StrategyCtx& ctx, const Bar& b);
  std::optional<StrategySignal> latest() const { return last_; }
  
  // BaseStrategy virtual methods
  ParameterMap get_default_params() const override;
  ParameterSpace get_param_space() const override;
  void apply_params() override;
  double calculate_probability(const std::vector<Bar>& bars, int current_index) override;
  
  // **NEW**: Strategy-agnostic allocation interface
  std::vector<AllocationDecision> get_allocation_decisions(
      const std::vector<Bar>& bars, 
      int current_index,
      const std::string& base_symbol,
      const std::string& bull3x_symbol,
      const std::string& bear3x_symbol) override;
  
    RouterCfg get_router_config() const override;
    // REMOVED: get_sizer_config() - No artificial limits allowed for profit maximization
    
    // **ARCHITECTURAL COMPLIANCE**: TFA requires dynamic allocation for profit maximization
    bool requires_dynamic_allocation() const override { return true; }

private:
  TFACfg cfg_;
  ml::ModelHandle handle_;
  ml::FeatureWindow window_;
  std::optional<StrategySignal> last_;
  std::vector<std::vector<double>> feature_buffer_;
  StrategySignal map_output(const ml::ModelOutput& mo) const;
  
  // Feature projection system
  mutable std::unique_ptr<ColumnProjector> projector_;
  mutable std::unique_ptr<ColumnProjectorSafe> projector_safe_;
  mutable bool projector_initialized_{false};
  mutable int expected_feat_dim_{56};
  
  // CRITICAL FIX: Move static state from calculate_probability to class members
  // This prevents data leakage between different test runs and ensures deterministic behavior
  mutable int probability_calls_{0};
  mutable bool seq_context_initialized_{false};
  mutable TfaSeqContext seq_context_;
  mutable std::vector<float> precomputed_probabilities_;
  mutable std::vector<float> probability_history_;
  mutable int cooldown_long_until_{-1};
  mutable int cooldown_short_until_{-1};
};

} // namespace sentio
