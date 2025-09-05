#pragma once
#include "base_strategy.hpp"
#include "indicators.hpp"
#include <optional>

namespace sentio {

struct SMACrossCfg {
  int fast = 10;
  int slow = 30;
  double conf_fast_slow = 0.7; // confidence when cross happens
};

class SMACrossStrategy final : public IStrategy {
public:
  explicit SMACrossStrategy(const SMACrossCfg& cfg);
  void on_bar(const StrategyCtx& ctx, const Bar& b) override;
  std::optional<StrategySignal> latest() const override { return last_; }
  bool warmed_up() const { return sma_fast_.ready() && sma_slow_.ready(); }
private:
  SMACrossCfg cfg_;
  SMA sma_fast_, sma_slow_;
  double last_fast_{NAN}, last_slow_{NAN};
  std::optional<StrategySignal> last_;
};

} // namespace sentio
