#pragma once
#include <cstdint>
#include <optional>

namespace sentio::rules {

struct RuleOutput {
  std::optional<float> p_up;     // [0,1]
  std::optional<int>   signal;   // {-1,0,+1}
  std::optional<float> score;    // unbounded or [-1,1]
  std::optional<float> strength; // [0,1]
};

struct BarsView {
  const int64_t* ts;
  const double*  open;
  const double*  high;
  const double*  low;
  const double*  close;
  const double*  volume;
  int64_t        n;
};

struct IRuleStrategy {
  virtual ~IRuleStrategy() = default;
  virtual std::optional<RuleOutput> eval(const BarsView& bars, int64_t i) = 0;
  virtual int  warmup() const { return 20; }
  virtual const char* name() const { return "UnnamedRule"; }
};

} // namespace sentio::rules


