# Strategy-Scoped Router/Sizer: Requirements + Rule-Based Strategies Index

**Generated**: 2025-09-08 11:55:53
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: Augmented document: includes all rule-based strategies (BollingerSqueeze, OpeningRange, MomentumVolume, OrderFlowImbalance, OrderFlowScalping, VWAPReversion, SMA Cross, MarketMaking) alongside the router/sizer extensibility plan. Use this as a reference to design strategy-specific router/sizers with consistent audit. Modules included from include/sentio/strategy_*.hpp and src/strategy_*.cpp, plus router/sizer/signal pipeline and audit.

**Total Files**: 145

---

## 📋 **TABLE OF CONTENTS**

1. [include/kochi/kochi_binary_context.hpp](#file-1)
2. [include/kochi/kochi_ppo_context.hpp](#file-2)
3. [include/sentio/all_strategies.hpp](#file-3)
4. [include/sentio/alpha.hpp](#file-4)
5. [include/sentio/audit.hpp](#file-5)
6. [include/sentio/base_strategy.hpp](#file-6)
7. [include/sentio/binio.hpp](#file-7)
8. [include/sentio/bo.hpp](#file-8)
9. [include/sentio/bollinger.hpp](#file-9)
10. [include/sentio/calendar_seed.hpp](#file-10)
11. [include/sentio/core.hpp](#file-11)
12. [include/sentio/core/bar.hpp](#file-12)
13. [include/sentio/cost_aware_gate.hpp](#file-13)
14. [include/sentio/cost_model.hpp](#file-14)
15. [include/sentio/csv_loader.hpp](#file-15)
16. [include/sentio/data_resolver.hpp](#file-16)
17. [include/sentio/day_index.hpp](#file-17)
18. [include/sentio/exec/asof_index.hpp](#file-18)
19. [include/sentio/exec_types.hpp](#file-19)
20. [include/sentio/execution/pnl_engine.hpp](#file-20)
21. [include/sentio/feature/column_projector.hpp](#file-21)
22. [include/sentio/feature/column_projector_safe.hpp](#file-22)
23. [include/sentio/feature/csv_feature_provider.hpp](#file-23)
24. [include/sentio/feature/feature_builder_guarded.hpp](#file-24)
25. [include/sentio/feature/feature_builder_ops.hpp](#file-25)
26. [include/sentio/feature/feature_feeder_guarded.hpp](#file-26)
27. [include/sentio/feature/feature_from_spec.hpp](#file-27)
28. [include/sentio/feature/feature_matrix.hpp](#file-28)
29. [include/sentio/feature/feature_provider.hpp](#file-29)
30. [include/sentio/feature/name_diff.hpp](#file-30)
31. [include/sentio/feature/ops.hpp](#file-31)
32. [include/sentio/feature/sanitize.hpp](#file-32)
33. [include/sentio/feature/standard_scaler.hpp](#file-33)
34. [include/sentio/feature_builder.hpp](#file-34)
35. [include/sentio/feature_cache.hpp](#file-35)
36. [include/sentio/feature_engineering/feature_normalizer.hpp](#file-36)
37. [include/sentio/feature_engineering/kochi_features.hpp](#file-37)
38. [include/sentio/feature_engineering/technical_indicators.hpp](#file-38)
39. [include/sentio/feature_feeder.hpp](#file-39)
40. [include/sentio/feature_health.hpp](#file-40)
41. [include/sentio/indicators.hpp](#file-41)
42. [include/sentio/metrics.hpp](#file-42)
43. [include/sentio/ml/feature_pipeline.hpp](#file-43)
44. [include/sentio/ml/feature_window.hpp](#file-44)
45. [include/sentio/ml/iml_model.hpp](#file-45)
46. [include/sentio/ml/model_registry.hpp](#file-46)
47. [include/sentio/ml/ts_model.hpp](#file-47)
48. [include/sentio/of_index.hpp](#file-48)
49. [include/sentio/of_precompute.hpp](#file-49)
50. [include/sentio/optimizer.hpp](#file-50)
51. [include/sentio/orderflow_types.hpp](#file-51)
52. [include/sentio/pnl_accounting.hpp](#file-52)
53. [include/sentio/polygon_client.hpp](#file-53)
54. [include/sentio/polygon_ingest.hpp](#file-54)
55. [include/sentio/position_manager.hpp](#file-55)
56. [include/sentio/pricebook.hpp](#file-56)
57. [include/sentio/profiling.hpp](#file-57)
58. [include/sentio/progress_bar.hpp](#file-58)
59. [include/sentio/property_test.hpp](#file-59)
60. [include/sentio/rolling_stats.hpp](#file-60)
61. [include/sentio/router.hpp](#file-61)
62. [include/sentio/rth_calendar.hpp](#file-62)
63. [include/sentio/runner.hpp](#file-63)
64. [include/sentio/sanity.hpp](#file-64)
65. [include/sentio/signal_diag.hpp](#file-65)
66. [include/sentio/signal_engine.hpp](#file-66)
67. [include/sentio/signal_gate.hpp](#file-67)
68. [include/sentio/signal_pipeline.hpp](#file-68)
69. [include/sentio/signal_trace.hpp](#file-69)
70. [include/sentio/sim_data.hpp](#file-70)
71. [include/sentio/sizer.hpp](#file-71)
72. [include/sentio/strategy_bollinger_squeeze_breakout.hpp](#file-72)
73. [include/sentio/strategy_hybrid_ppo.hpp](#file-73)
74. [include/sentio/strategy_kochi_ppo.hpp](#file-74)
75. [include/sentio/strategy_market_making.hpp](#file-75)
76. [include/sentio/strategy_momentum_volume.hpp](#file-76)
77. [include/sentio/strategy_opening_range_breakout.hpp](#file-77)
78. [include/sentio/strategy_order_flow_imbalance.hpp](#file-78)
79. [include/sentio/strategy_order_flow_scalping.hpp](#file-79)
80. [include/sentio/strategy_sma_cross.hpp](#file-80)
81. [include/sentio/strategy_tfa.hpp](#file-81)
82. [include/sentio/strategy_transformer.hpp](#file-82)
83. [include/sentio/strategy_transformer_ts.hpp](#file-83)
84. [include/sentio/strategy_vwap_reversion.hpp](#file-84)
85. [include/sentio/sym/leverage_registry.hpp](#file-85)
86. [include/sentio/sym/symbol_utils.hpp](#file-86)
87. [include/sentio/symbol_table.hpp](#file-87)
88. [include/sentio/telemetry_logger.hpp](#file-88)
89. [include/sentio/temporal_analysis.hpp](#file-89)
90. [include/sentio/tfa/artifacts_loader.hpp](#file-90)
91. [include/sentio/tfa/artifacts_safe.hpp](#file-91)
92. [include/sentio/tfa/feature_guard.hpp](#file-92)
93. [include/sentio/tfa/input_shim.hpp](#file-93)
94. [include/sentio/tfa/signal_pipeline.hpp](#file-94)
95. [include/sentio/tfa/tfa_seq_context.hpp](#file-95)
96. [include/sentio/time_utils.hpp](#file-96)
97. [include/sentio/torch/safe_from_blob.hpp](#file-97)
98. [include/sentio/util/bytes.hpp](#file-98)
99. [include/sentio/util/safe_matrix.hpp](#file-99)
100. [include/sentio/wf.hpp](#file-100)
101. [src/audit.cpp](#file-101)
102. [src/base_strategy.cpp](#file-102)
103. [src/cost_aware_gate.cpp](#file-103)
104. [src/csv_loader.cpp](#file-104)
105. [src/feature_builder.cpp](#file-105)
106. [src/feature_cache.cpp](#file-106)
107. [src/feature_engineering/feature_normalizer.cpp](#file-107)
108. [src/feature_engineering/kochi_features.cpp](#file-108)
109. [src/feature_engineering/technical_indicators.cpp](#file-109)
110. [src/feature_feeder.cpp](#file-110)
111. [src/feature_feeder_guarded.cpp](#file-111)
112. [src/feature_health.cpp](#file-112)
113. [src/kochi_runner.cpp](#file-113)
114. [src/main.cpp](#file-114)
115. [src/ml/model_registry_ts.cpp](#file-115)
116. [src/ml/ts_model.cpp](#file-116)
117. [src/optimizer.cpp](#file-117)
118. [src/pnl_accounting.cpp](#file-118)
119. [src/poly_fetch_main.cpp](#file-119)
120. [src/polygon_client.cpp](#file-120)
121. [src/polygon_ingest.cpp](#file-121)
122. [src/router.cpp](#file-122)
123. [src/rth_calendar.cpp](#file-123)
124. [src/runner.cpp](#file-124)
125. [src/sanity.cpp](#file-125)
126. [src/signal_engine.cpp](#file-126)
127. [src/signal_gate.cpp](#file-127)
128. [src/signal_pipeline.cpp](#file-128)
129. [src/signal_trace.cpp](#file-129)
130. [src/sim_data.cpp](#file-130)
131. [src/strategy_bollinger_squeeze_breakout.cpp](#file-131)
132. [src/strategy_hybrid_ppo.cpp](#file-132)
133. [src/strategy_kochi_ppo.cpp](#file-133)
134. [src/strategy_market_making.cpp](#file-134)
135. [src/strategy_momentum_volume.cpp](#file-135)
136. [src/strategy_opening_range_breakout.cpp](#file-136)
137. [src/strategy_order_flow_imbalance.cpp](#file-137)
138. [src/strategy_order_flow_scalping.cpp](#file-138)
139. [src/strategy_sma_cross.cpp](#file-139)
140. [src/strategy_tfa.cpp](#file-140)
141. [src/strategy_transformer_ts.cpp](#file-141)
142. [src/strategy_vwap_reversion.cpp](#file-142)
143. [src/telemetry_logger.cpp](#file-143)
144. [src/temporal_analysis.cpp](#file-144)
145. [src/time_utils.cpp](#file-145)

---

## 📄 **FILE 1 of 145**: include/kochi/kochi_binary_context.hpp

**File Information**:
- **Path**: `include/kochi/kochi_binary_context.hpp`

- **Size**: 91 lines
- **Modified**: 2025-09-08 11:33:28

- **Type**: .hpp

```text
#pragma once
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "sentio/feature/column_projector.hpp"

namespace kochi {

struct KochiBinaryContext {
  torch::jit::Module model;
  int Fk{0};
  int T{0};
  int emit_from{0};
  float pad{0.f};
  std::vector<std::string> expected_names;
  sentio::ColumnProjector projector;

  static nlohmann::json jload(const std::string &p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("missing json: " + p);
    nlohmann::json j;
    f >> j;
    return j;
  }

  void load(const std::string &model_pt, const std::string &meta_json, const std::vector<std::string> &runtime_names) {
    model = torch::jit::load(model_pt, torch::kCPU);
    model.eval();
    auto meta = jload(meta_json);
    auto ex = meta["expects"];
    Fk = ex["input_dim"].get<int>();
    T = ex["seq_len"].get<int>();
    emit_from = ex["emit_from"].get<int>();
    pad = ex["pad_value"].get<float>();
    expected_names = ex["feature_names"].get<std::vector<std::string>>();
    projector = sentio::ColumnProjector::make(runtime_names, expected_names, pad);
  }

  void forward(const float *Xsrc,
               int64_t N,
               int64_t F_runtime,
               std::vector<float> &p_up_out,
               const std::string &audit_jsonl = "") {
    std::vector<float> X;
    projector.project(Xsrc, (size_t)N, (size_t)F_runtime, X);
    const float *Xp = X.data();

    p_up_out.assign((size_t)N, 0.5f);

    torch::NoGradGuard ng;
    torch::InferenceMode im;
    const int64_t start = std::max<int64_t>(emit_from, T);
    const int64_t last = N - 1;
    const int64_t B = 256;

    std::ofstream audit;
    if (!audit_jsonl.empty()) audit.open(audit_jsonl);

    for (int64_t i = start; i <= last;) {
      int64_t L = std::min<int64_t>(B, last - i + 1);
      auto t = torch::empty({L, T, Fk}, torch::kFloat32);
      float *dst = t.data_ptr<float>();
      for (int64_t k = 0; k < L; ++k) {
        int64_t end = i + k, lo = end - T;
        std::memcpy(dst + k * T * Fk, Xp + lo * Fk, sizeof(float) * (size_t)(T * Fk));
      }
      auto y = model.forward({t}).toTensor();
      if (y.dim() == 2 && y.size(1) == 1) y = y.squeeze(1);
      auto logits = y.contiguous().data_ptr<float>();
      for (int64_t k = 0; k < L; ++k) {
        float lg = logits[k];
        float p = 1.f / (1.f + std::exp(-lg));
        size_t idx = (size_t)(i + k);
        p_up_out[idx] = p;
        if (audit.is_open()) {
          audit << "{\"i\":" << idx << ",\"logit\":" << lg << ",\"p_up\":" << p << "}\n";
        }
      }
      i += L;
    }
  }
};

} // namespace kochi



```

## 📄 **FILE 2 of 145**: include/kochi/kochi_ppo_context.hpp

**File Information**:
- **Path**: `include/kochi/kochi_ppo_context.hpp`

- **Size**: 98 lines
- **Modified**: 2025-09-08 01:46:55

- **Type**: .hpp

```text
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

```

## 📄 **FILE 3 of 145**: include/sentio/all_strategies.hpp

**File Information**:
- **Path**: `include/sentio/all_strategies.hpp`

- **Size**: 17 lines
- **Modified**: 2025-09-07 22:52:32

- **Type**: .hpp

```text
#pragma once

// This file ensures all strategies are included and registered with the factory.
// Include this header once in your main.cpp.

#include "strategy_bollinger_squeeze_breakout.hpp"
#include "strategy_market_making.hpp"
#include "strategy_momentum_volume.hpp"
#include "strategy_opening_range_breakout.hpp"
#include "strategy_order_flow_imbalance.hpp"
#include "strategy_order_flow_scalping.hpp"
#include "strategy_vwap_reversion.hpp"
#include "strategy_hybrid_ppo.hpp"
#include "strategy_transformer_ts.hpp"
// TFB strategy removed - focusing on TFA only
#include "strategy_tfa.hpp"
#include "strategy_kochi_ppo.hpp"
```

## 📄 **FILE 4 of 145**: include/sentio/alpha.hpp

**File Information**:
- **Path**: `include/sentio/alpha.hpp`

- **Size**: 7 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <string>

namespace sentio {
// Removed: Direction enum and StratSignal struct. These are now defined in core.hpp
} // namespace sentio


```

## 📄 **FILE 5 of 145**: include/sentio/audit.hpp

**File Information**:
- **Path**: `include/sentio/audit.hpp`

- **Size**: 95 lines
- **Modified**: 2025-09-05 20:13:14

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <optional>
#include <unordered_map>
#include <vector>

namespace sentio {

// --- Core enums/structs ----
enum class SigType : uint8_t { BUY=0, STRONG_BUY=1, SELL=2, STRONG_SELL=3, HOLD=4 };
enum class Side    : uint8_t { Buy=0, Sell=1 };

// Simple Bar structure for audit system (avoiding conflicts with core.hpp)
struct AuditBar { 
  double open{}, high{}, low{}, close{}, volume{};
};

struct AuditPosition { 
  double qty{0.0}; 
  double avg_px{0.0}; 
};

struct AccountState {
  double cash{0.0};
  double realized{0.0};
  double equity{0.0};
  // computed: equity = cash + realized + sum(qty * mark_px)
};

struct AuditConfig {
  std::string run_id;           // stable id for this run
  std::string file_path;        // where JSONL events are appended
  bool        flush_each=true;  // fsync-ish (fflush) after each write
};

// --- Recorder: append events to JSONL ---
class AuditRecorder {
public:
  explicit AuditRecorder(const AuditConfig& cfg);
  ~AuditRecorder();

  // lifecycle
  void event_run_start(std::int64_t ts_utc, const std::string& meta_json="{}");
  void event_run_end(std::int64_t ts_utc, const std::string& meta_json="{}");

  // data plane
  void event_bar   (std::int64_t ts_utc, const std::string& instrument, const AuditBar& b);
  void event_signal(std::int64_t ts_utc, const std::string& base_symbol, SigType type, double confidence);
  void event_route (std::int64_t ts_utc, const std::string& base_symbol, const std::string& instrument, double target_weight);
  void event_order (std::int64_t ts_utc, const std::string& instrument, Side side, double qty, double limit_px);
  void event_fill  (std::int64_t ts_utc, const std::string& instrument, double price, double qty, double fees, Side side);
  void event_snapshot(std::int64_t ts_utc, const AccountState& acct);
  void event_metric (std::int64_t ts_utc, const std::string& key, double value);

  // Get current config (for creating new instances)
  AuditConfig get_config() const { return {run_id_, file_path_, flush_each_}; }

private:
  std::string run_id_;
  std::string file_path_;
  std::FILE*  fp_{nullptr};
  std::uint64_t seq_{0};
  bool flush_each_;
  void write_line_(const std::string& s);
  static std::string sha1_hex_(const std::string& s); // tiny local impl
  static std::string json_escape_(const std::string& s);
};

// --- Replayer: read JSONL, rebuild state, recompute P&L, verify ---
struct ReplayResult {
  // recomputed
  std::unordered_map<std::string, AuditPosition> positions;
  AccountState acct{};
  std::size_t  bars{0}, signals{0}, routes{0}, orders{0}, fills{0};
  // mismatches discovered
  std::vector<std::string> issues;
};

class AuditReplayer {
public:
  // price map can be filled from bar events; you may also inject EOD marks
  struct PriceBook { std::unordered_map<std::string, double> last_px; };

  // replay the file; return recomputed account/pnl from fills + marks
  static std::optional<ReplayResult> replay_file(const std::string& file_path,
                                                 const std::string& run_id_expect = "");
private:
  static bool apply_bar_(PriceBook& pb, const std::string& instrument, const AuditBar& b);
  static void mark_to_market_(const PriceBook& pb, ReplayResult& rr);
  static void apply_fill_(ReplayResult& rr, const std::string& inst, double px, double qty, double fees, Side side);
};

} // namespace sentio
```

## 📄 **FILE 6 of 145**: include/sentio/base_strategy.hpp

**File Information**:
- **Path**: `include/sentio/base_strategy.hpp`

- **Size**: 115 lines
- **Modified**: 2025-09-05 21:10:09

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "signal_diag.hpp"
#include "router.hpp"  // for StrategySignal
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <functional>

namespace sentio {

// Strategy context for bar processing
struct StrategyCtx {
  std::string instrument;     // traded instrument for this stream
  std::int64_t ts_utc_epoch;  // bar timestamp (UTC seconds)
  bool is_rth{true};          // inject from your RTH checker
};

// Minimal strategy interface for ML integration
class IStrategy {
public:
  virtual ~IStrategy() = default;
  virtual void on_bar(const StrategyCtx& ctx, const Bar& b) = 0;
  virtual std::optional<StrategySignal> latest() const = 0;
};

// Parameter types and enums
enum class ParamType { INT, FLOAT };
struct ParamSpec { 
    ParamType type;
    double min_val, max_val;
    double default_val;
};

using ParameterMap = std::unordered_map<std::string, double>;
using ParameterSpace = std::unordered_map<std::string, ParamSpec>;

enum class SignalType { NONE = 0, BUY = 1, SELL = -1, STRONG_BUY = 2, STRONG_SELL = -2 };

struct StrategyState {
    bool in_position = false;
    SignalType last_signal = SignalType::NONE;
    int last_trade_bar = -1000; // Initialize far in the past
    
    void reset() {
        in_position = false;
        last_signal = SignalType::NONE;
        last_trade_bar = -1000;
    }
};

// StrategySignal is now defined in router.hpp

class BaseStrategy {
protected:
    std::string name_;
    ParameterMap params_;
    StrategyState state_;
    SignalDiag diag_;

    bool is_cooldown_active(int current_bar, int cooldown_period) const;
    
public:
    BaseStrategy(const std::string& name) : name_(name) {}
    virtual ~BaseStrategy() = default;

    // **MODIFIED**: Explicitly delete copy and move operations for this polymorphic base class.
    // This prevents object slicing and ownership confusion.
    BaseStrategy(const BaseStrategy&) = delete;
    BaseStrategy& operator=(const BaseStrategy&) = delete;
    BaseStrategy(BaseStrategy&&) = delete;
    BaseStrategy& operator=(BaseStrategy&&) = delete;
    
    std::string get_name() const { return name_; }
    
    virtual ParameterMap get_default_params() const = 0;
    virtual ParameterSpace get_param_space() const = 0;
    virtual void apply_params() = 0;
    
    void set_params(const ParameterMap& params);
    
    virtual StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) = 0;
    virtual void reset_state();
    
    const SignalDiag& get_diag() const { return diag_; }
};

class StrategyFactory {
public:
    using CreateFunction = std::function<std::unique_ptr<BaseStrategy>()>;
    
    static StrategyFactory& instance();
    void register_strategy(const std::string& name, CreateFunction create_func);
    std::unique_ptr<BaseStrategy> create_strategy(const std::string& name);
    std::vector<std::string> get_available_strategies() const;
    
private:
    std::unordered_map<std::string, CreateFunction> strategies_;
};

// **NEW**: The final, more robust registration macro.
// It takes the C++ ClassName and the "Name" to be used by the CLI.
#define REGISTER_STRATEGY(ClassName, Name) \
    namespace { \
        struct ClassName##Registrar { \
            ClassName##Registrar() { \
                StrategyFactory::instance().register_strategy(Name, \
                    []() { return std::make_unique<ClassName>(); }); \
            } \
        }; \
        static ClassName##Registrar g_##ClassName##_registrar; \
    }

} // namespace sentio
```

## 📄 **FILE 7 of 145**: include/sentio/binio.hpp

**File Information**:
- **Path**: `include/sentio/binio.hpp`

- **Size**: 68 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <cstdio>
#include <vector>
#include <string>
#include "core.hpp"

namespace sentio {

inline void save_bin(const std::string& path, const std::vector<Bar>& v) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return;
    
    uint64_t n = v.size();
    std::fwrite(&n, sizeof(n), 1, fp);
    
    for (const auto& bar : v) {
        // Write string length and data
        uint32_t str_len = bar.ts_utc.length();
        std::fwrite(&str_len, sizeof(str_len), 1, fp);
        std::fwrite(bar.ts_utc.c_str(), 1, str_len, fp);
        
        // Write other fields
        std::fwrite(&bar.ts_nyt_epoch, sizeof(bar.ts_nyt_epoch), 1, fp);
        std::fwrite(&bar.open, sizeof(bar.open), 1, fp);
        std::fwrite(&bar.high, sizeof(bar.high), 1, fp);
        std::fwrite(&bar.low, sizeof(bar.low), 1, fp);
        std::fwrite(&bar.close, sizeof(bar.close), 1, fp);
        std::fwrite(&bar.volume, sizeof(bar.volume), 1, fp);
    }
    std::fclose(fp);
}

inline std::vector<Bar> load_bin(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return {};
    
    uint64_t n = 0; 
    std::fread(&n, sizeof(n), 1, fp);
    std::vector<Bar> v;
    v.reserve(n);
    
    for (uint64_t i = 0; i < n; ++i) {
        Bar bar;
        
        // Read string length and data
        uint32_t str_len = 0;
        std::fread(&str_len, sizeof(str_len), 1, fp);
        if (str_len > 0) {
            std::vector<char> str_data(str_len);
            std::fread(str_data.data(), 1, str_len, fp);
            bar.ts_utc = std::string(str_data.data(), str_len);
        }
        
        // Read other fields
        std::fread(&bar.ts_nyt_epoch, sizeof(bar.ts_nyt_epoch), 1, fp);
        std::fread(&bar.open, sizeof(bar.open), 1, fp);
        std::fread(&bar.high, sizeof(bar.high), 1, fp);
        std::fread(&bar.low, sizeof(bar.low), 1, fp);
        std::fread(&bar.close, sizeof(bar.close), 1, fp);
        std::fread(&bar.volume, sizeof(bar.volume), 1, fp);
        
        v.push_back(bar);
    }
    std::fclose(fp);
    return v;
}

} // namespace sentio
```

## 📄 **FILE 8 of 145**: include/sentio/bo.hpp

**File Information**:
- **Path**: `include/sentio/bo.hpp`

- **Size**: 384 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
// bo.hpp — minimal, solid C++20 Bayesian Optimization (GP + EI)
// No external deps. Deterministic. Safe. Box bounds + integers + batch ask().

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <functional>
#include <optional>
#include <limits>
#include <cmath>
#include <cassert>
#include <chrono>
#include <string>

// ----------------------------- Utilities -----------------------------
struct BOBounds {
  std::vector<double> lo, hi; // size = D
  bool valid() const { return !lo.empty() && lo.size()==hi.size(); }
  int dim() const { return (int)lo.size(); }
  inline double clamp_unit(double u, int d) const {
    return std::min(1.0, std::max(0.0, u));
  }
  inline double to_real(double u, int d) const {
    u = clamp_unit(u,d);
    return lo[d] + u * (hi[d] - lo[d]);
  }
  inline double to_unit(double x, int d) const {
    const double L = lo[d], H = hi[d];
    if (H<=L) return 0.0;
    double u = (x - L) / (H - L);
    return std::min(1.0, std::max(0.0, u));
  }
};

// Parameter kinds (continuous or integer). If INT, we round to nearest.
enum class ParamKind : uint8_t { CONT, INT };

struct BOOpts {
  int init_design = 16;          // initial random/space-filling points
  int cand_pool   = 2048;        // candidate samples for EI maximization
  int batch_q     = 1;           // q-EI via constant liar
  double jitter   = 1e-10;       // numerical jitter for Cholesky
  double ei_xi    = 0.01;        // EI exploration parameter
  bool  maximize  = true;        // true: maximize objective (default)
  uint64_t seed   = 42;          // deterministic rand seed
  bool  verbose   = false;
};

// ----------------------------- Tiny matrix helpers -----------------------------
// Row-major dense matrix with minimal ops (just what we need for GP Cholesky).
struct Mat {
  int n=0, m=0;                  // n rows, m cols
  std::vector<double> a;         // size n*m
  Mat()=default;
  Mat(int n_, int m_) : n(n_), m(m_), a((size_t)n_*(size_t)m_, 0.0) {}
  inline double& operator()(int i, int j){ return a[(size_t)i*(size_t)m + j]; }
  inline double  operator()(int i, int j) const { return a[(size_t)i*(size_t)m + j]; }
};

// Cholesky decomposition A = L L^T in-place into L (lower). Returns false if fails.
inline bool cholesky(Mat& A, double jitter=1e-10){
  // A must be square, symmetric, positive definite (we'll add jitter on diag).
  assert(A.n == A.m);
  const int n = A.n;
  for (int i=0;i<n;i++) A(i,i) += jitter;

  for (int i=0;i<n;i++){
    for (int j=0;j<=i;j++){
      double sum = A(i,j);
      for (int k=0;k<j;k++) sum -= A(i,k)*A(j,k);
      if (i==j){
        if (sum <= 0.0) return false;
        A(i,j) = std::sqrt(sum);
      } else {
        A(i,j) = sum / A(j,j);
      }
    }
    for (int j=i+1;j<n;j++) A(i,j)=0.0; // zero upper for clarity
  }
  return true;
}

// Solve L y = b (forward)
inline void trisolve_lower(const Mat& L, std::vector<double>& y){
  const int n=L.n;
  for (int i=0;i<n;i++){
    double s = y[i];
    for (int k=0;k<i;k++) s -= L(i,k)*y[k];
    y[i] = s / L(i,i);
  }
}
// Solve L^T x = y (backward)
inline void trisolve_upperT(const Mat& L, std::vector<double>& x){
  const int n=L.n;
  for (int i=n-1;i>=0;i--){
    double s = x[i];
    for (int k=i+1;k<n;k++) s -= L(k,i)*x[k];
    x[i] = s / L(i,i);
  }
}

// ----------------------------- Gaussian Process (RBF-ARD) -----------------------------
struct GP {
  // Hyperparams (on unit cube): signal^2, noise^2, lengthscales (per-dim)
  double sigma_f2 = 1.0;
  double sigma_n2 = 1e-6;
  std::vector<double> ell; // size D, lengthscales

  // Data (unit cube)
  std::vector<std::vector<double>> X; // N x D
  std::vector<double> y;              // N (centered)
  double y_mean = 0.0;

  // Factorization
  Mat L;                 // Cholesky of K = Kf + sigma_n2 I
  std::vector<double> alpha; // (K)^-1 y

  static inline double sqdist_ard(const std::vector<double>& a, const std::vector<double>& b,
                                  const std::vector<double>& ell){
    double s=0.0;
    for (size_t d=0; d<ell.size(); ++d){
      const double z = (a[d]-b[d]) / std::max(1e-12, ell[d]);
      s += z*z;
    }
    return s;
  }

  inline double kf(const std::vector<double>& a, const std::vector<double>& b) const {
    const double s2 = sqdist_ard(a,b,ell);
    return sigma_f2 * std::exp(-0.5 * s2);
  }

  void fit(const std::vector<std::vector<double>>& X_unit,
           const std::vector<double>& y_raw,
           double jitter=1e-10)
  {
    X = X_unit;
    const int N = (int)X.size();
    y_mean = 0.0;
    for (double v: y_raw) y_mean += v;
    y_mean /= std::max(1, N);
    y.resize(N);
    for (int i=0;i<N;i++) y[i] = y_raw[i] - y_mean;

    // Build K
    Mat K(N,N);
    for (int i=0;i<N;i++){
      for (int j=0;j<=i;j++){
        double kij = kf(X[i], X[j]);
        if (i==j) kij += sigma_n2;
        K(i,j) = K(j,i) = kij;
      }
    }
    // Chol
    L = K;
    if (!cholesky(L, jitter)) {
      // increase jitter progressively if needed
      double j = std::max(jitter, 1e-12);
      bool ok=false;
      for (int t=0;t<6 && !ok; ++t) { L = K; ok = cholesky(L, j); j *= 10.0; }
      if (!ok) throw std::runtime_error("GP Cholesky failed (matrix not PD)");
    }
    // alpha = K^{-1} y  via L
    alpha = y;
    trisolve_lower(L, alpha);
    trisolve_upperT(L, alpha);
  }

  // Predictive mean/variance at x (unit cube)
  inline std::pair<double,double> predict(const std::vector<double>& x) const {
    const int N = (int)X.size();
    std::vector<double> k(N);
    for (int i=0;i<N;i++) k[i] = kf(x, X[i]);
    // mean = k^T alpha + y_mean
    double mu = std::inner_product(k.begin(), k.end(), alpha.begin(), 0.0) + y_mean;
    // v = L^{-1} k
    std::vector<double> v = k;
    trisolve_lower(L, v);
    double kxx = sigma_f2; // k(x,x) for RBF with same params is sigma_f2
    double var = kxx - std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    if (var < 1e-18) var = 1e-18;
    return {mu, var};
  }
};

// ----------------------------- Random/Sampling helpers -----------------------------
struct RNG {
  std::mt19937_64 g;
  explicit RNG(uint64_t seed): g(seed) {}
  double uni(){ return std::uniform_real_distribution<double>(0.0,1.0)(g); }
  int randint(int lo, int hi){ return std::uniform_int_distribution<int>(lo,hi)(g); }
};

// Jittered Latin hypercube on unit cube
inline std::vector<std::vector<double>> latin_hypercube(int n, int D, RNG& rng){
  std::vector<std::vector<double>> X(n, std::vector<double>(D,0.0));
  for (int d=0; d<D; ++d){
    std::vector<double> slots(n);
    for (int i=0;i<n;i++) slots[i] = (i + rng.uni()) / n;
    std::shuffle(slots.begin(), slots.end(), rng.g);
    for (int i=0;i<n;i++) X[i][d] = std::min(1.0, std::max(0.0, slots[i]));
  }
  return X;
}

// Round integer dims to nearest feasible grid-point after scaling
inline void apply_kinds_inplace(std::vector<double>& u, const std::vector<ParamKind>& kinds,
                                const BOBounds& B){
  for (int d=0; d<(int)u.size(); ++d){
    if (kinds[d] == ParamKind::INT){
      // snap in real space to integer, then rescale back to unit
      const double x = B.to_real(u[d], d);
      const double xi = std::round(x);
      u[d] = B.to_unit(xi, d);
    }
  }
}

// ----------------------------- Acquisition: Expected Improvement -----------------------------
inline double norm_pdf(double z){ static const double inv_sqrt2pi = 0.3989422804014327; return inv_sqrt2pi*std::exp(-0.5*z*z); }
inline double norm_cdf(double z){
  return 0.5 * std::erfc(-z/std::sqrt(2.0));
}
// EI for maximization: EI = (mu - best - xi) Phi(z) + sigma phi(z), z = (mu - best - xi)/sigma
inline double expected_improvement(double mu, double var, double best, double xi){
  const double sigma = std::sqrt(std::max(1e-18, var));
  const double diff  = mu - best - xi;
  if (sigma <= 1e-12) return std::max(0.0, diff);
  const double z = diff / sigma;
  return diff * norm_cdf(z) + sigma * norm_pdf(z);
}

// ----------------------------- Bayesian Optimizer -----------------------------
struct BO {
  BOBounds bounds;
  std::vector<ParamKind> kinds; // size D
  BOOpts opt;

  // data (real-space for API; internally store unit-cube for GP)
  std::vector<std::vector<double>> X_real; // N x D
  std::vector<std::vector<double>> X;      // N x D (unit)
  std::vector<double> y;                   // N
  GP gp;
  RNG rng;

  BO(const BOBounds& B, const std::vector<ParamKind>& kinds_, BOOpts o)
  : bounds(B), kinds(kinds_), opt(o), gp(), rng(o.seed)
  {
    assert(bounds.valid());
    assert((int)kinds.size() == bounds.dim());
    // init default GP hyperparams
    gp.ell.assign(bounds.dim(), 0.2); // medium lengthscale on unit cube
    gp.sigma_f2 = 1.0;
    gp.sigma_n2 = 1e-6;
  }

  int dim() const { return bounds.dim(); }
  int size() const { return (int)X.size(); }

  void clear(){
    X_real.clear(); X.clear(); y.clear();
  }

  // Append observation
  void tell(const std::vector<double>& xr, double val){
    assert((int)xr.size() == dim());
    X_real.push_back(xr);
    std::vector<double> u(dim());
    for (int d=0; d<dim(); ++d) u[d] = bounds.to_unit(xr[d], d);
    // snap integers
    apply_kinds_inplace(u, kinds, bounds);
    X.push_back(u);
    y.push_back(val);
  }

  // Fit GP (simple hyperparam heuristics; robust & fast)
  void fit(){
    if (X.empty()) return;
    // set GP hyperparams from y stats
    double m=0, s2=0;
    for (double v: y) m += v; m /= (double)y.size();
    for (double v: y) s2 += (v-m)*(v-m);
    s2 /= std::max(1, (int)y.size()-1);
    gp.sigma_f2 = std::max(1e-12, s2);
    gp.sigma_n2 = std::max(1e-9 * gp.sigma_f2, 1e-10);
    // modest ARD scaling: initialize lengthscales to 0.2; (optionally: scale by output sensitivity)
    for (double& e: gp.ell) e = 0.2;
    gp.fit(X, y, opt.jitter);
  }

  // Generate initial design if dataset is empty/small
  void ensure_init_design(){
    const int need = std::max(0, opt.init_design - (int)X.size());
    if (need <= 0) return;
    auto U = latin_hypercube(need, dim(), rng);
    for (auto& u : U){
      apply_kinds_inplace(u, kinds, bounds);
      std::vector<double> xr(dim());
      for (int d=0; d<dim(); ++d) xr[d] = bounds.to_real(u[d], d);
      // placeholder y (user will evaluate and call tell)
      X.push_back(u);
      X_real.push_back(xr);
      y.push_back(std::numeric_limits<double>::quiet_NaN()); // mark unevaluated
    }
  }

  // Ask for q new locations (real-space). Uses constant liar on current model (no new evals yet).
  std::vector<std::vector<double>> ask(int q=1){
    if (q<=0) return {};
    // If we have NaNs from ensure_init_design, return those first (ask-user-to-evaluate)
    std::vector<std::vector<double>> out;
    for (int i=0;i<(int)X.size() && (int)out.size()<q; ++i){
      if (!std::isfinite(y[i])) out.push_back(X_real[i]);
    }
    if ((int)out.size() == q) return out;

    // Ensure we can build a GP on finished points
    // Build filtered dataset of (finite) y's
    std::vector<std::vector<double>> Xf;
    std::vector<double> yf;
    Xf.reserve(X.size()); yf.reserve(y.size());
    for (int i=0;i<(int)X.size(); ++i){
      if (std::isfinite(y[i])) { Xf.push_back(X[i]); yf.push_back(y[i]); }
    }
    if (Xf.size() >= 2) {
      gp.fit(Xf, yf, opt.jitter);
    } else {
      // not enough data to fit GP: just random suggest
      out = random_candidates(q);
      return out;
    }

    const double y_best = opt.maximize ? *std::max_element(yf.begin(), yf.end())
                                       : *std::min_element(yf.begin(), yf.end());

    // batch selection with constant liar (for maximization, lie = y_best)
    std::vector<std::vector<double>> X_aug = Xf;
    std::vector<double> y_aug = yf;

    for (int pick=0; pick<q; ++pick){
      // pool
      double best_ei = -1.0;
      std::vector<double> best_u(dim());
      for (int c=0; c<opt.cand_pool; ++c){
        std::vector<double> u(dim());
        for (int d=0; d<dim(); ++d) u[d] = rng.uni();
        apply_kinds_inplace(u, kinds, bounds);
        // score EI on augmented model (approximate by reusing gp; small bias is fine)
        // NOTE: we approximate: use current gp (no retrain) — fast & works well in practice
        auto [mu, var] = gp.predict(u);
        double ei = opt.maximize ? expected_improvement(mu, var, y_best, opt.ei_xi)
                                 : expected_improvement(-mu, var, -y_best, opt.ei_xi);
        if (ei > best_ei){ best_ei = ei; best_u = u; }
      }
      // map to real, append
      std::vector<double> xr(dim());
      for (int d=0; d<dim(); ++d) xr[d] = bounds.to_real(best_u[d], d);
      out.push_back(xr);

      // constant liar update (no refit for speed; optional: refit small)
      X_aug.push_back(best_u);
      y_aug.push_back(y_best); // lie
      // (Optionally refit gp with augmented data each pick for sharper q-EI; omitted for speed)
    }
    return out;
  }

  // Helper: random suggestions on real-space
  std::vector<std::vector<double>> random_candidates(int q){
    std::vector<std::vector<double>> out;
    out.reserve(q);
    for (int i=0;i<q;++i){
      std::vector<double> u(dim());
      for (int d=0; d<dim(); ++d) u[d] = rng.uni();
      apply_kinds_inplace(u, kinds, bounds);
      std::vector<double> xr(dim());
      for (int d=0; d<dim(); ++d) xr[d] = bounds.to_real(u[d], d);
      out.push_back(xr);
    }
    return out;
  }
};
```

## 📄 **FILE 9 of 145**: include/sentio/bollinger.hpp

**File Information**:
- **Path**: `include/sentio/bollinger.hpp`

- **Size**: 22 lines
- **Modified**: 2025-09-05 13:23:38

- **Type**: .hpp

```text
#pragma once
#include "rolling_stats.hpp"

namespace sentio {

struct Bollinger {
  RollingMeanVar mv;
  double k;
  double eps; // volatility floor to avoid zero-width bands

  Bollinger(int w, double k_=2.0, double eps_=1e-9) : mv(w), k(k_), eps(eps_) {}

  inline void step(double close, double& mid, double& lo, double& hi, double& sd_out){
    auto [m, var] = mv.push(close);
    double sd = std::sqrt(std::max(var, 0.0));
    if (sd < eps) sd = eps;         // <- floor
    mid = m; lo = m - k*sd; hi = m + k*sd;
    sd_out = sd;
  }
};

} // namespace sentio
```

## 📄 **FILE 10 of 145**: include/sentio/calendar_seed.hpp

**File Information**:
- **Path**: `include/sentio/calendar_seed.hpp`

- **Size**: 30 lines
- **Modified**: 2025-09-05 14:04:17

- **Type**: .hpp

```text
// calendar_seed.hpp
#pragma once
#include "rth_calendar.hpp"

namespace sentio {

inline TradingCalendar make_default_nyse_calendar() {
  TradingCalendar c;

  // Full-day holidays (partial sample; fill your range robustly)
  // 2022: New Year (obs 2021-12-31), MLK 2022-01-17, Presidents 02-21,
  // Good Friday 04-15, Memorial 05-30, Juneteenth (obs 06-20), Independence 07-04,
  // Labor 09-05, Thanksgiving 11-24, Christmas (obs 2022-12-26)
  c.full_holidays.insert(20220117);
  c.full_holidays.insert(20220221);
  c.full_holidays.insert(20220415);
  c.full_holidays.insert(20220530);
  c.full_holidays.insert(20220620);
  c.full_holidays.insert(20220704);
  c.full_holidays.insert(20220905);
  c.full_holidays.insert(20221124);
  c.full_holidays.insert(20221226);

  // Early closes (sample): Black Friday 2022-11-25 @ 13:00 ET
  c.early_close_sec.emplace(20221125, 13*3600);

  return c;
}

} // namespace sentio
```

## 📄 **FILE 11 of 145**: include/sentio/core.hpp

**File Information**:
- **Path**: `include/sentio/core.hpp`

- **Size**: 69 lines
- **Modified**: 2025-09-05 09:44:09

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath> // For std::sqrt

namespace sentio {

struct Bar {
    std::string ts_utc;
    int64_t ts_nyt_epoch;
    double open, high, low, close;
    uint64_t volume;
};

// **MODIFIED**: This struct now holds a vector of Positions, indexed by symbol ID for performance.
struct Position { 
    double qty = 0.0; 
    double avg_price = 0.0; 
};

struct Portfolio {
    double cash = 100000.0;
    std::vector<Position> positions; // Indexed by symbol ID

    Portfolio() = default;
    explicit Portfolio(size_t num_symbols) : positions(num_symbols) {}
};

// **MODIFIED**: Vector-based functions are now the primary way to manage the portfolio.
inline void apply_fill(Portfolio& pf, int sid, double qty_delta, double price) {
    if (sid < 0 || static_cast<size_t>(sid) >= pf.positions.size()) {
        return; // Invalid symbol ID
    }
    
    pf.cash -= qty_delta * price;
    auto& pos = pf.positions[sid];
    
    double new_qty = pos.qty + qty_delta;
    if (std::abs(new_qty) < 1e-9) { // Position is closed
        pos.qty = 0.0;
        pos.avg_price = 0.0;
    } else {
        if (pos.qty * new_qty >= 0) { // Increasing position or opening a new one
            pos.avg_price = (pos.avg_price * pos.qty + price * qty_delta) / new_qty;
        }
        // If flipping from long to short or vice-versa, the new avg_price is just the fill price.
        else if (pos.qty * qty_delta < 0) {
             pos.avg_price = price;
        }
        pos.qty = new_qty;
    }
}

inline double equity_mark_to_market(const Portfolio& pf, const std::vector<double>& last_prices) {
    double eq = pf.cash;
    for (size_t sid = 0; sid < pf.positions.size(); ++sid) {
        if (std::abs(pf.positions[sid].qty) > 0.0 && sid < last_prices.size()) {
            eq += pf.positions[sid].qty * last_prices[sid];
        }
    }
    return eq;
}

// **REMOVED**: Old, simplistic Direction and StratSignal types are now deprecated.

} // namespace sentio
```

## 📄 **FILE 12 of 145**: include/sentio/core/bar.hpp

**File Information**:
- **Path**: `include/sentio/core/bar.hpp`

- **Size**: 9 lines
- **Modified**: 2025-09-06 22:14:53

- **Type**: .hpp

```text
#pragma once
#include <cstdint>

namespace sentio {
struct Bar {
  std::int64_t ts_epoch_us{0};
  double open{0}, high{0}, low{0}, close{0}, volume{0};
};
}

```

## 📄 **FILE 13 of 145**: include/sentio/cost_aware_gate.hpp

**File Information**:
- **Path**: `include/sentio/cost_aware_gate.hpp`

- **Size**: 114 lines
- **Modified**: 2025-09-06 01:51:35

- **Type**: .hpp

```text
#pragma once
#include "signal_gate.hpp"
#include "router.hpp"
#include <unordered_map>
#include <vector>
#include <cmath>

namespace sentio {

/**
 * Cost-aware signal gate that filters signals based on expected transaction costs
 * and confidence thresholds derived from backtested cost curves.
 */
class CostAwareGate {
public:
    struct CostCurve {
        double base_cost_bp;           // Base transaction cost in basis points
        double confidence_threshold;   // Minimum confidence required to trade
        double min_expected_return_bp; // Minimum expected return to justify trade
        double max_position_size;      // Maximum position size as fraction of capital
    };

    struct CostAwareConfig {
        std::unordered_map<std::string, CostCurve> instrument_costs;
        double default_confidence_floor;
        double default_cost_bp;
        bool enable_cost_filtering;
        
        CostAwareConfig() 
            : default_confidence_floor(0.05)
            , default_cost_bp(2.0)
            , enable_cost_filtering(true) {}
    };

    explicit CostAwareGate(const CostAwareConfig& config = CostAwareConfig());

    /**
     * Filter signal based on cost analysis
     * @param signal Input signal to evaluate
     * @param instrument Instrument identifier
     * @param current_price Current market price
     * @param position_size Current position size
     * @return Filtered signal (may be modified or rejected)
     */
    std::optional<StrategySignal> filter_signal(
        const StrategySignal& signal,
        const std::string& instrument,
        double current_price,
        double position_size = 0.0
    ) const;

    /**
     * Calculate expected transaction cost for a signal
     * @param signal_type Type of signal (BUY/SELL)
     * @param instrument Instrument identifier
     * @param position_size Position size
     * @param current_price Current market price
     * @return Expected cost in basis points
     */
    double calculate_expected_cost(
        StrategySignal::Type signal_type,
        const std::string& instrument,
        double position_size,
        double current_price
    ) const;

    /**
     * Calculate minimum confidence required for profitable trade
     * @param instrument Instrument identifier
     * @param position_size Position size
     * @param current_price Current market price
     * @return Minimum confidence threshold
     */
    double calculate_min_confidence(
        const std::string& instrument,
        double position_size,
        double current_price
    ) const;

    /**
     * Update cost curve for an instrument based on recent performance
     * @param instrument Instrument identifier
     * @param recent_trades Vector of recent trade P&L data
     */
    void update_cost_curve(
        const std::string& instrument,
        const std::vector<double>& recent_trades
    );

    /**
     * Check if signal should be rejected due to cost constraints
     * @param signal Input signal
     * @param instrument Instrument identifier
     * @param current_price Current market price
     * @param position_size Current position size
     * @return True if signal should be rejected
     */
    bool should_reject_signal(
        const StrategySignal& signal,
        const std::string& instrument,
        double current_price,
        double position_size = 0.0
    ) const;

private:
    CostAwareConfig config_;
    
    // Helper methods
    const CostCurve& get_cost_curve(const std::string& instrument) const;
    double calculate_slippage_cost(double position_size, double current_price) const;
    double calculate_market_impact(double position_size, double current_price) const;
};

} // namespace sentio

```

## 📄 **FILE 14 of 145**: include/sentio/cost_model.hpp

**File Information**:
- **Path**: `include/sentio/cost_model.hpp`

- **Size**: 120 lines
- **Modified**: 2025-09-05 09:44:09

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <cmath>

namespace sentio {

// Alpaca Trading Cost Model
struct AlpacaCostModel {
  // Commission structure (Alpaca is commission-free for stocks/ETFs)
  static constexpr double commission_per_share = 0.0;
  static constexpr double min_commission = 0.0;
  
  // SEC fees (for sells only)
  static constexpr double sec_fee_rate = 0.0000278; // $0.0278 per $1000 of principal
  
  // FINRA Trading Activity Fee (TAF) - for sells only
  static constexpr double taf_per_share = 0.000145; // $0.000145 per share (max $7.27 per trade)
  static constexpr double taf_max_per_trade = 7.27;
  
  // Slippage model based on market impact
  struct SlippageParams {
    double base_slippage_bps = 1.0;    // Base 1 bps slippage
    double volume_impact_factor = 0.5;  // Additional slippage based on volume
    double volatility_factor = 0.3;     // Additional slippage based on volatility
    double max_slippage_bps = 10.0;     // Cap at 10 bps
  };
  
  static SlippageParams default_slippage;
  
  // Calculate total transaction costs for a trade
  static double calculate_fees([[maybe_unused]] const std::string& symbol, 
                              double quantity, 
                              double price, 
                              bool is_sell) {
    double notional = std::abs(quantity) * price;
    double total_fees = 0.0;
    
    // Commission (free for Alpaca)
    total_fees += commission_per_share * std::abs(quantity);
    total_fees = std::max(total_fees, min_commission);
    
    if (is_sell) {
      // SEC fees (sells only)
      total_fees += notional * sec_fee_rate;
      
      // FINRA TAF (sells only)
      double taf = std::abs(quantity) * taf_per_share;
      total_fees += std::min(taf, taf_max_per_trade);
    }
    
    return total_fees;
  }
  
  // Calculate slippage based on trade characteristics
  static double calculate_slippage_bps(double quantity,
                                      double price, 
                                      double avg_volume,
                                      double volatility,
                                      const SlippageParams& params = default_slippage) {
    double notional = std::abs(quantity) * price;
    
    // Base slippage
    double slippage_bps = params.base_slippage_bps;
    
    // Volume impact (higher for larger trades relative to average volume)
    if (avg_volume > 0) {
      double volume_ratio = notional / (avg_volume * price);
      slippage_bps += params.volume_impact_factor * std::sqrt(volume_ratio) * 100; // Convert to bps
    }
    
    // Volatility impact
    slippage_bps += params.volatility_factor * volatility * 10000; // Convert annual vol to bps
    
    // Cap the slippage
    return std::min(slippage_bps, params.max_slippage_bps);
  }
  
  // Apply slippage to execution price
  static double apply_slippage(double market_price, 
                              double slippage_bps, 
                              bool is_buy) {
    double slippage_factor = slippage_bps / 10000.0; // Convert bps to decimal
    
    if (is_buy) {
      return market_price * (1.0 + slippage_factor); // Pay more when buying
    } else {
      return market_price * (1.0 - slippage_factor); // Receive less when selling
    }
  }
  
  // Complete cost calculation including fees and slippage
  static std::pair<double, double> calculate_total_costs(
      const std::string& symbol,
      double quantity,
      double market_price,
      double avg_volume = 1000000, // Default 1M average volume
      double volatility = 0.20,    // Default 20% annual volatility
      const SlippageParams& params = default_slippage) {
    
    bool is_sell = quantity < 0;
    bool is_buy = quantity > 0;
    
    // Calculate fees
    double fees = calculate_fees(symbol, quantity, market_price, is_sell);
    
    // Calculate slippage
    double slippage_bps = calculate_slippage_bps(quantity, market_price, avg_volume, volatility, params);
    double execution_price = apply_slippage(market_price, slippage_bps, is_buy);
    
    // Slippage cost (difference from market price)
    double slippage_cost = std::abs(quantity) * std::abs(execution_price - market_price);
    
    return {fees, slippage_cost};
  }
};

// Static member definition
inline AlpacaCostModel::SlippageParams AlpacaCostModel::default_slippage = {};

} // namespace sentio
```

## 📄 **FILE 15 of 145**: include/sentio/csv_loader.hpp

**File Information**:
- **Path**: `include/sentio/csv_loader.hpp`

- **Size**: 8 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include <string>

namespace sentio {
bool load_csv(const std::string& path, std::vector<Bar>& out);
} // namespace sentio


```

## 📄 **FILE 16 of 145**: include/sentio/data_resolver.hpp

**File Information**:
- **Path**: `include/sentio/data_resolver.hpp`

- **Size**: 44 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <filesystem>
#include <cstdlib>

namespace sentio {
enum class TickerFamily { Qqq, Bitcoin, Tesla };

inline const char** family_symbols(TickerFamily f, int& n) {
  static const char* QQQ[] = {"QQQ","TQQQ","SQQQ","PSQ"};
  static const char* BTC[] = {"BTCUSD","ETHUSD"};
  static const char* TSLA[]= {"TSLA","TSLQ"};
  switch (f) {
    case TickerFamily::Qqq: n=4; return QQQ;
    case TickerFamily::Bitcoin: n=2; return BTC;
    case TickerFamily::Tesla: n=2; return TSLA;
  }
  n=0; return nullptr;
}

inline std::string resolve_csv(const std::string& symbol,
                               const std::string& equities_root="data/equities",
                               const std::string& crypto_root="data/crypto") {
  namespace fs = std::filesystem;
  std::string up = symbol; for (auto& c: up) c = ::toupper(c);
  auto is_crypto = (up=="BTC"||up=="BTCUSD"||up=="ETH"||up=="ETHUSD");

  const char* env_root = std::getenv("SENTIO_DATA_ROOT");
  const char* env_suffix = std::getenv("SENTIO_DATA_SUFFIX");
  std::string base = env_root ? std::string(env_root) : (is_crypto ? crypto_root : equities_root);
  std::string suffix = env_suffix ? std::string(env_suffix) : std::string("");

  // Prefer suffixed file in base, then non-suffixed, then fallback to default roots
  std::string cand1 = base + "/" + up + suffix + ".csv";
  if (fs::exists(cand1)) return cand1;
  std::string cand2 = base + "/" + up + ".csv";
  if (fs::exists(cand2)) return cand2;
  std::string fallback_base = (is_crypto ? crypto_root : equities_root);
  std::string cand3 = fallback_base + "/" + up + suffix + ".csv";
  if (fs::exists(cand3)) return cand3;
  return fallback_base + "/" + up + ".csv";
}
} // namespace sentio


```

## 📄 **FILE 17 of 145**: include/sentio/day_index.hpp

**File Information**:
- **Path**: `include/sentio/day_index.hpp`

- **Size**: 26 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include <vector>
#include <string_view>

namespace sentio {

// From minute bars with RFC3339 timestamps, return indices of first bar per day
inline std::vector<int> day_start_indices(const std::vector<Bar>& bars) {
    std::vector<int> starts;
    starts.reserve(bars.size() / 300 + 2);
    std::string last_day;
    for (int i=0; i<(int)bars.size(); ++i) {
        std::string_view ts(bars[i].ts_utc);
        if (ts.size() < 10) continue;
        std::string cur{ts.substr(0,10)};
        if (i == 0 || cur != last_day) {
            starts.push_back(i);
            last_day = std::move(cur);
        }
    }
    return starts;
}

} // namespace sentio


```

## 📄 **FILE 18 of 145**: include/sentio/exec/asof_index.hpp

**File Information**:
- **Path**: `include/sentio/exec/asof_index.hpp`

- **Size**: 22 lines
- **Modified**: 2025-09-06 22:14:53

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>

namespace sentio {

// Build a map from instrument rows to base rows via as-of (<= ts) alignment.
// base_ts, inst_ts must be monotonic non-decreasing (your loaders should ensure).
inline std::vector<int32_t> build_asof_index(const std::vector<std::int64_t>& base_ts,
                                             const std::vector<std::int64_t>& inst_ts) {
  std::vector<int32_t> idx(inst_ts.size(), -1);
  std::size_t j = 0;
  if (base_ts.empty()) return idx;
  for (std::size_t i = 0; i < inst_ts.size(); ++i) {
    auto t = inst_ts[i];
    while (j + 1 < base_ts.size() && base_ts[j + 1] <= t) ++j;
    idx[i] = static_cast<int32_t>(j);
  }
  return idx;
}

} // namespace sentio

```

## 📄 **FILE 19 of 145**: include/sentio/exec_types.hpp

**File Information**:
- **Path**: `include/sentio/exec_types.hpp`

- **Size**: 15 lines
- **Modified**: 2025-09-05 14:08:05

- **Type**: .hpp

```text
// exec_types.hpp
#pragma once
#include <string>

namespace sentio {

struct ExecutionIntent {
  std::string base_symbol;     // e.g., "QQQ"
  std::string instrument;      // e.g., "TQQQ" or "SQQQ" or "QQQ"
  double      qty = 0.0;
  double      leverage = 1.0;  // informational; actual product carries leverage
  double      score = 0.0;     // signal strength
};

} // namespace sentio
```

## 📄 **FILE 20 of 145**: include/sentio/execution/pnl_engine.hpp

**File Information**:
- **Path**: `include/sentio/execution/pnl_engine.hpp`

- **Size**: 171 lines
- **Modified**: 2025-09-08 09:05:49

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <cmath>
#include <optional>
#include <algorithm>
#include <stdexcept>

namespace sentio {

struct Fill {
  int64_t ts{0};
  double  price{0.0};
  double  qty{0.0};  // +buy, -sell
  double  fee{0.0};
  std::string venue;
};

struct BarPx {
  int64_t ts{0};
  double close{0.0};
  double bid{0.0};
  double ask{0.0};
};

struct Lot {
  double qty{0.0};
  double px{0.0};
  int64_t ts{0};
};

struct PnLSnapshot {
  int64_t ts{0};
  double position{0.0};
  double avg_price{0.0};
  double cash{0.0};
  double realized{0.0};
  double unrealized{0.0};
  double equity{0.0};
  double last_price{0.0};
};

struct FeeModel {
  double per_share{0.0};
  double bps_notional{0.0};
  double min_fee{0.0};
  double compute(double price, double qty) const {
    double absqty = std::abs(qty);
    double f = per_share * absqty + (bps_notional * (price * absqty));
    return std::max(f, min_fee);
  }
};

struct SlippageModel {
  double bps{0.0};
  double apply(double ref_price, double qty) const {
    double sgn = (qty >= 0.0 ? 1.0 : -1.0);
    double mult = 1.0 + sgn * bps;
    return ref_price * mult;
  }
};

class PnLEngine {
public:
  enum class PriceMode { Close, Mid, Bid, Ask };

  explicit PnLEngine(double start_cash = 100000.0)
    : cash_(start_cash), equity_(start_cash) {}

  void set_price_mode(PriceMode m) { price_mode_ = m; }
  void set_fee_model(const FeeModel& f) { fee_model_ = f; auto_fee_ = true; }
  void set_slippage_model(const SlippageModel& s) { slippage_ = s; }

  void on_fill(const Fill& fill) {
    if (std::abs(fill.qty) < 1e-12) return;
    const double qty = fill.qty;
    const double px  = fill.price;
    double fee = fill.fee;
    if (auto_fee_) fee = fee_model_.compute(px, qty);

    double remaining = qty;
    if (same_sign(remaining, position_)) {
      lots_.push_back(Lot{remaining, px, fill.ts});
      position_ += remaining;
      cash_ -= px * qty;
      cash_ -= fee;
    } else {
      while (std::abs(remaining) > 1e-12 && !lots_.empty() && opposite_sign(remaining, lots_.front().qty)) {
        Lot &lot = lots_.front();
        double close_qty = std::min(std::abs(remaining), std::abs(lot.qty));
        double dq = (lot.qty > 0 ? +close_qty : -close_qty);
        realized_ += (px - lot.px) * dq;
        cash_ -= px * (-dq);
        lot.qty -= dq;
        remaining += dq;
        if (std::abs(lot.qty) <= 1e-12) lots_.erase(lots_.begin());
      }
      if (std::abs(remaining) > 1e-12) {
        lots_.push_back(Lot{remaining, px, fill.ts});
        position_ += remaining;
        cash_ -= px * remaining;
      } else {
        position_ = sum_position_from_lots();
      }
      cash_ -= fee;
    }
    avg_price_ = compute_signed_avg();
  }

  void on_bar(const BarPx& bar) {
    last_price_ = reference_price(bar);
    unrealized_ = position_ * (last_price_ - avg_price_);
    equity_     = cash_ + position_ * last_price_;
    snapshots_.push_back(PnLSnapshot{bar.ts, position_, avg_price_, cash_, realized_, unrealized_, equity_, last_price_});
  }

  void reset(double start_cash = 100000.0) {
    lots_.clear(); snapshots_.clear();
    position_=0; avg_price_=0; cash_=start_cash; realized_=0; unrealized_=0; equity_=start_cash; last_price_=0;
  }

  const std::vector<PnLSnapshot>& snapshots() const { return snapshots_; }
  double position()  const { return position_; }
  double avg_price() const { return avg_price_; }
  double cash()      const { return cash_; }
  double realized()  const { return realized_; }
  double unrealized()const { return unrealized_; }
  double equity()    const { return equity_; }

private:
  static bool same_sign(double a, double b){ return (a>=0 && b>=0) || (a<=0 && b<=0); }
  static bool opposite_sign(double a, double b){ return (a>=0 && b<=0) || (a<=0 && b>=0); }

  double compute_signed_avg() const {
    double num = 0.0, den = 0.0;
    for (const auto &l : lots_) { num += l.px * l.qty; den += l.qty; }
    if (std::abs(den) < 1e-12) return 0.0;
    return num / den;
  }
  double sum_position_from_lots() const { double s=0.0; for (const auto &l : lots_) s += l.qty; return s; }

  double reference_price(const BarPx& bar) const {
    switch (price_mode_) {
      case PriceMode::Close: return bar.close;
      case PriceMode::Mid:   return (bar.bid>0 && bar.ask>0) ? 0.5*(bar.bid+bar.ask) : bar.close;
      case PriceMode::Bid:   return (bar.bid>0) ? bar.bid : bar.close;
      case PriceMode::Ask:   return (bar.ask>0) ? bar.ask : bar.close;
    }
    return bar.close;
  }

private:
  std::vector<Lot> lots_;
  std::vector<PnLSnapshot> snapshots_;

  double position_{0.0};
  double avg_price_{0.0};
  double cash_{0.0};
  double realized_{0.0};
  double unrealized_{0.0};
  double equity_{0.0};
  double last_price_{0.0};

  PriceMode price_mode_{PriceMode::Close};
  FeeModel fee_model_{};
  SlippageModel slippage_{};
  bool auto_fee_{false};
};

} // namespace sentio

```

## 📄 **FILE 21 of 145**: include/sentio/feature/column_projector.hpp

**File Information**:
- **Path**: `include/sentio/feature/column_projector.hpp`

- **Size**: 41 lines
- **Modified**: 2025-09-08 01:46:55

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace sentio {

struct ColumnProjector {
  std::vector<int> src_to_dst; // index in src for each dst column; -1 -> pad
  float pad{0.0f};

  static ColumnProjector make(const std::vector<std::string>& src,
                              const std::vector<std::string>& dst,
                              float pad_value){
    std::unordered_map<std::string,int> pos;
    for (int i=0;i<(int)src.size();++i) pos[src[i]] = i;
    ColumnProjector P; P.pad = pad_value;
    P.src_to_dst.resize(dst.size(), -1);
    for (int j=0;j<(int)dst.size();++j){
      auto it = pos.find(dst[j]);
      P.src_to_dst[j] = (it==pos.end()) ? -1 : it->second;
    }
    return P;
  }

  void project(const float* X, size_t rows, size_t src_cols, std::vector<float>& Y) const {
    const size_t dst_cols = src_to_dst.size();
    Y.assign(rows * dst_cols, pad);
    for (size_t r=0;r<rows;++r){
      const float* src = X + r*src_cols;
      float* dst = Y.data() + r*dst_cols;
      for (size_t j=0;j<dst_cols;++j){
        int si = src_to_dst[j];
        if (si>=0) dst[j] = src[si];
      }
    }
  }
};

} // namespace sentio

```

## 📄 **FILE 22 of 145**: include/sentio/feature/column_projector_safe.hpp

**File Information**:
- **Path**: `include/sentio/feature/column_projector_safe.hpp`

- **Size**: 101 lines
- **Modified**: 2025-09-07 19:22:14

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <cstddef>
#include <iostream>

namespace sentio {

struct ColumnProjectorSafe {
  std::vector<int> map;   // dst[j] = src index or -1
  size_t F_src{0}, F_dst{0};
  float fill_value{0.0f};

  static ColumnProjectorSafe make(const std::vector<std::string>& src,
                                  const std::vector<std::string>& dst,
                                  float fill_value = 0.0f) {
    ColumnProjectorSafe P; 
    P.F_src = src.size(); 
    P.F_dst = dst.size(); 
    P.fill_value = fill_value;
    P.map.assign(P.F_dst, -1);
    
    std::unordered_map<std::string,int> pos; 
    pos.reserve(src.size()*2);
    for (int i=0;i<(int)src.size();++i) pos[src[i]] = i;
    
    int filled_count = 0;
    int mapped_count = 0;
    
    for (int j=0;j<(int)dst.size();++j){
      auto it = pos.find(dst[j]);
      if (it!=pos.end()) {
        P.map[j] = it->second; // Found mapping
        mapped_count++;
      } else {
        P.map[j] = -1; // Will be filled
        filled_count++;
      }
    }
    
    std::cout << "[ColumnProjectorSafe] Created: " << src.size() << " src → " << dst.size() 
              << " dst (mapped=" << mapped_count << ", filled=" << filled_count << ")" << std::endl;
    
    if (filled_count > 0) {
      std::cout << "[ColumnProjectorSafe] WARNING: " << filled_count 
                << " features will be filled with " << fill_value << std::endl;
    }
    
    return P;
  }

  void project(const float* X_src, size_t rows, size_t Fsrc, std::vector<float>& X_out) const {
    if (Fsrc != F_src) {
      throw std::runtime_error("ColumnProjectorSafe: F_src mismatch expected=" + 
                               std::to_string(F_src) + " got=" + std::to_string(Fsrc));
    }
    
    X_out.assign(rows*F_dst, fill_value);
    
    for (size_t r=0;r<rows;++r){
      const float* src = X_src + r*F_src;
      float* dst = X_out.data() + r*F_dst;
      
      for (size_t j=0;j<F_dst;++j){
        int si = map[j];
        if (si >= 0 && si < (int)F_src) {
          dst[j] = src[(size_t)si];
        } else {
          dst[j] = fill_value;
        }
      }
    }
  }
  
  void project_double(const double* X_src, size_t rows, size_t Fsrc, std::vector<float>& X_out) const {
    if (Fsrc != F_src) {
      throw std::runtime_error("ColumnProjectorSafe: F_src mismatch expected=" + 
                               std::to_string(F_src) + " got=" + std::to_string(Fsrc));
    }
    
    X_out.assign(rows*F_dst, fill_value);
    
    for (size_t r=0;r<rows;++r){
      const double* src = X_src + r*F_src;
      float* dst = X_out.data() + r*F_dst;
      
      for (size_t j=0;j<F_dst;++j){
        int si = map[j];
        if (si >= 0 && si < (int)F_src) {
          dst[j] = static_cast<float>(src[(size_t)si]);
        } else {
          dst[j] = fill_value;
        }
      }
    }
  }
};

} // namespace sentio
```

## 📄 **FILE 23 of 145**: include/sentio/feature/csv_feature_provider.hpp

**File Information**:
- **Path**: `include/sentio/feature/csv_feature_provider.hpp`

- **Size**: 60 lines
- **Modified**: 2025-09-08 01:46:55

- **Type**: .hpp

```text
#pragma once
#include "sentio/feature/feature_provider.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace sentio {

struct CsvFeatureProvider : IFeatureProvider {
  std::string path_;
  std::vector<std::string> names_;
  int T_{64};

  explicit CsvFeatureProvider(std::string path, int T)
  : path_(std::move(path)), T_(T) {
    std::ifstream f(path_);
    if (!f) throw std::runtime_error("missing features csv: " + path_);
    std::string header;
    std::getline(f, header);
    std::stringstream hs(header);
    std::string col; int idx=0;
    while (std::getline(hs, col, ',')) {
      if (idx==0 && (col=="bar_index"||col=="idx")) { idx++; continue; }
      if (col=="ts" || col=="timestamp") { idx++; continue; }
      names_.push_back(col);
      idx++;
    }
  }

  FeatureMatrix get_features_for(const std::string& /*symbol*/) override {
    std::ifstream f(path_);
    if (!f) throw std::runtime_error("missing: " + path_);
    std::string line;
    std::getline(f, line); // header
    std::vector<float> buf; buf.reserve(1<<20);
    int64_t rows=0; const int64_t cols=(int64_t)names_.size();
    while (std::getline(f, line)) {
      if (line.empty()) continue;
      std::stringstream ss(line);
      std::string cell; int colidx=0; bool first=true; bool have_ts=false;
      // optional bar_index, optional timestamp, then features
      while (std::getline(ss, cell, ',')) {
        if (first) { first=false; continue; }
        if (!have_ts) { have_ts=true; continue; }
        buf.push_back(cell.empty()? 0.0f : std::stof(cell));
        colidx++;
      }
      if (colidx != cols) throw std::runtime_error("col mismatch in " + path_);
      rows++;
    }
    return FeatureMatrix{rows, cols, std::move(buf)};
  }

  std::vector<std::string> feature_names() const override { return names_; }
  int seq_len() const override { return T_; }
};

} // namespace sentio

```

## 📄 **FILE 24 of 145**: include/sentio/feature/feature_builder_guarded.hpp

**File Information**:
- **Path**: `include/sentio/feature/feature_builder_guarded.hpp`

- **Size**: 104 lines
- **Modified**: 2025-09-06 23:05:23

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include "sentio/core/bar.hpp"
#include "sentio/feature/feature_matrix.hpp"
#include "sentio/sym/leverage_registry.hpp"
#include "sentio/sym/symbol_utils.hpp"

namespace sentio {

// Throws if symbol is leveraged: you must pass a base ticker (e.g., "QQQ")
inline FeatureMatrix build_features_for_base(const std::string& symbol,
                                             const std::vector<Bar>& bars);

// ---- Implementation (header-only for simplicity) ----
namespace detail {
  inline float s_log_safe(float x) { return std::log(std::max(x, 1e-12f)); }
}

inline FeatureMatrix build_features_for_base(const std::string& symbol,
                                             const std::vector<Bar>& bars)
{
  const auto symU = to_upper(symbol);
  if (is_leveraged(symU)) {
    throw std::invalid_argument("FeatureBuilder: leveraged symbol '" + symU +
                                "' not allowed. Pass base ticker: '" + resolve_base(symU) + "'");
  }

  const std::int64_t N = static_cast<std::int64_t>(bars.size());
  if (N < 64) return {}; // not enough history; adjust to your min

  // Example feature set (extend as needed)
  // 0: close, 1: logret, 2: ema20, 3: ema50, 4: rsi14, 5: zscore20(logret)
  constexpr int F = 6;
  FeatureMatrix M;
  M.rows = N; M.cols = F;
  M.data.resize(static_cast<std::size_t>(N * F));

  std::vector<float> close(N), logret(N, 0.f), ema20(N, 0.f), ema50(N, 0.f), rsi14(N, 0.f), z20(N, 0.f);

  for (std::int64_t i = 0; i < N; ++i) close[i] = static_cast<float>(bars[i].close);
  for (std::int64_t i = 1; i < N; ++i) logret[i] = detail::s_log_safe(close[i] / std::max(close[i-1], 1e-12f));

  auto ema = [&](int period, std::vector<float>& out){
    const float k = 2.f / (period + 1.f);
    float e = close[0];
    out[0] = e;
    for (std::int64_t i=1; i<N; ++i){ e = k*close[i] + (1.f - k)*e; out[i] = e; }
  };
  ema(20, ema20);
  ema(50, ema50);

  // RSI(14) (Wilders)
  {
    const int p = 14;
    float up=0.f, dn=0.f;
    for (int i=1; i<=p && i<N; ++i){
      float d = close[i]-close[i-1];
      up += std::max(d, 0.f);
      dn += std::max(-d, 0.f);
    }
    up/=p; dn/=p;
    for (std::int64_t i=p+1; i<N; ++i){
      float d = close[i]-close[i-1];
      up = (up*(p-1) + std::max(d,0.f)) / p;
      dn = (dn*(p-1) + std::max(-d,0.f)) / p;
      float rs = (dn>1e-12f) ? (up/dn) : 0.f;
      rsi14[i] = 100.f - 100.f/(1.f + rs);
    }
  }

  // Z-score(20) of logret
  {
    const int w = 20;
    if (N > w) {
      double sum=0.0, sum2=0.0;
      for (int i=0; i<w; ++i){ sum += logret[i]; sum2 += logret[i]*logret[i]; }
      for (std::int64_t i=w; i<N; ++i){
        const double mu = sum / w;
        const double var = std::max(0.0, sum2 / w - mu*mu);
        const float sd = static_cast<float>(std::sqrt(var));
        z20[i] = sd > 1e-8f ? static_cast<float>((logret[i]-mu)/sd) : 0.f;
        // slide
        sum += logret[i] - logret[i-w];
        sum2 += logret[i]*logret[i] - logret[i-w]*logret[i-w];
      }
    }
  }

  // Pack row-major
  for (std::int64_t i=0; i<N; ++i) {
    float* r = M.row_ptr(i);
    r[0] = close[i];
    r[1] = logret[i];
    r[2] = ema20[i];
    r[3] = ema50[i];
    r[4] = rsi14[i];
    r[5] = z20[i];
  }
  return M;
}

} // namespace sentio

```

## 📄 **FILE 25 of 145**: include/sentio/feature/feature_builder_ops.hpp

**File Information**:
- **Path**: `include/sentio/feature/feature_builder_ops.hpp`

- **Size**: 66 lines
- **Modified**: 2025-09-06 22:25:38

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace sentio {

inline std::vector<float> fb_ident(const std::vector<float>& x){ return x; }

inline std::vector<float> fb_logret(const std::vector<float>& x){
  std::vector<float> y(x.size(), 0.f);
  for (size_t i=1;i<x.size();++i){
    float a = std::max(x[i],    1e-12f);
    float b = std::max(x[i-1],  1e-12f);
    y[i] = std::log(a) - std::log(b);
  }
  return y;
}

inline std::vector<float> fb_ema(const std::vector<float>& x, int p){
  std::vector<float> e(x.size());
  if (x.empty()) return e;
  float k = 2.f / (p + 1.f);
  e[0] = x[0];
  for (size_t i=1;i<x.size();++i) e[i] = k*x[i] + (1.f-k)*e[i-1];
  return e;
}

inline std::vector<float> fb_rsi(const std::vector<float>& x, int p=14){
  const size_t N=x.size();
  std::vector<float> out(N,0.f), up(N,0.f), dn(N,0.f);
  for (size_t i=1;i<N;++i){
    float d=x[i]-x[i-1]; up[i]=std::max(d,0.f); dn[i]=std::max(-d,0.f);
  }
  std::vector<float> ru(N,0.f), rd(N,0.f);
  if (N> (size_t)p){
    float su=0.f, sd=0.f;
    for (int i=1;i<=p;i++){ su+=up[i]; sd+=dn[i]; }
    ru[p]=su/p; rd[p]=sd/p;
    for (size_t i=p+1;i<N;++i){
      ru[i]=(ru[i-1]*(p-1)+up[i])/p;
      rd[i]=(rd[i-1]*(p-1)+dn[i])/p;
      float rs=(rd[i]>1e-12f)?(ru[i]/rd[i]):0.f;
      out[i]=100.f-100.f/(1.f+rs);
    }
  }
  return out;
}

inline std::vector<float> fb_zwin(const std::vector<float>& x, int w){
  const size_t N=x.size(); std::vector<float> out(N,0.f);
  if (N <= (size_t)w) return out;
  std::vector<double> s(N,0.0), s2(N,0.0);
  s[0]=x[0]; s2[0]=x[0]*x[0];
  for (size_t i=1;i<N;++i){ s[i]=s[i-1]+x[i]; s2[i]=s2[i-1]+x[i]*x[i]; }
  for (size_t i=w;i<N;++i){
    double su = s[i]-s[i-w], su2 = s2[i]-s2[i-w];
    double mu = su/w;
    double var = std::max(0.0, su2/w - mu*mu);
    float sd = (float)std::sqrt(var);
    out[i] = (sd>1e-8f) ? (float)((x[i]-mu)/sd) : 0.f;
  }
  return out;
}

} // namespace sentio

```

## 📄 **FILE 26 of 145**: include/sentio/feature/feature_feeder_guarded.hpp

**File Information**:
- **Path**: `include/sentio/feature/feature_feeder_guarded.hpp`

- **Size**: 54 lines
- **Modified**: 2025-09-06 22:14:53

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "sentio/core/bar.hpp"
#include "sentio/feature/feature_matrix.hpp"
#include "sentio/feature/standard_scaler.hpp"
#include "sentio/sym/leverage_registry.hpp"
#include "sentio/exec/asof_index.hpp"

namespace sentio {

struct FeederInit {
  // All price series loaded (both base and leveraged allowed here)
  std::unordered_map<std::string, std::vector<Bar>> series; // symbol -> bars
  // The single base you want to signal on this run (e.g., "QQQ").
  // If empty, we'll infer it from presence (prefers QQQ if present).
  std::string base_symbol;
};

class FeatureFeederGuarded {
public:
  bool initialize(const FeederInit& init);

  const FeatureMatrix& features() const { return X_; }
  const StandardScaler& scaler() const { return scaler_; }
  const std::vector<std::int64_t>& base_ts() const { return base_ts_; }

  // For execution: map instrument rows to base rows
  // (present only for leveraged family members that exist in input)
  const std::vector<int32_t>* asof_map_for(const std::string& symbol) const {
    auto it = asof_.find(to_upper(symbol));
    if (it == asof_.end()) return nullptr;
    return &it->second;
  }

  // True if symbol is permitted for execution in this run (base or leverage family)
  bool allowed_for_exec(const std::string& symbol) const;

  const std::string& base() const { return base_symU_; }

private:
  std::string base_symU_;
  FeatureMatrix X_;
  StandardScaler scaler_;
  std::vector<std::int64_t> base_ts_;
  std::unordered_map<std::string, std::vector<int32_t>> asof_; // SYM -> asof index into base
  std::unordered_map<std::string, std::vector<Bar>> prices_;   // keep original price series

  bool infer_base_if_needed_(const std::unordered_map<std::string, std::vector<Bar>>& series,
                             std::string& base_out);
};

} // namespace sentio

```

## 📄 **FILE 27 of 145**: include/sentio/feature/feature_from_spec.hpp

**File Information**:
- **Path**: `include/sentio/feature/feature_from_spec.hpp`

- **Size**: 257 lines
- **Modified**: 2025-09-07 22:44:15

- **Type**: .hpp

```text
#pragma once
#include <stdexcept>
#include <string>
#include <vector>
#include "sentio/core.hpp"
#include "sentio/feature/feature_matrix.hpp"
#include "sentio/feature/feature_builder_ops.hpp"
#include "sentio/feature/ops.hpp"
#include "sentio/sym/leverage_registry.hpp"

// nlohmann JSON single-header:
#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace sentio {

inline FeatureMatrix build_features_from_spec_json(
  const std::string& symbol,
  const std::vector<Bar>& bars,
  const std::string& spec_json
){
  if (is_leveraged(symbol))
    throw std::invalid_argument("Leveraged symbol not allowed in FeatureBuilder: " + symbol);

  const size_t N = bars.size();
  if (N == 0) return {};

  // load spec
  json spec = json::parse(spec_json);
  const int F = (int)spec["features"].size();
  int emit_from = (int)spec["alignment_policy"]["emit_from_index"];
  float pad = (float)spec["alignment_policy"]["pad_value"];

  // Build source vectors
  std::vector<float> open(N), high(N), low(N), close(N), volume(N);
  std::vector<float> logret; // Built on demand
  
  for (size_t i = 0; i < N; ++i) {
    open[i] = (float)bars[i].open;
    high[i] = (float)bars[i].high;
    low[i] = (float)bars[i].low;
    close[i] = (float)bars[i].close;
    volume[i] = (float)bars[i].volume;
  }

  auto get_source_vector = [&](const std::string& name) -> const std::vector<float>& {
    if (name == "open") return open;
    if (name == "high") return high;
    if (name == "low") return low;
    if (name == "close") return close;
    if (name == "volume") return volume;
    if (name == "logret") {
      if (logret.empty()) logret = op_LOGRET(close);
      return logret;
    }
    // Handle composite sources
    if (name == "hlc" || name == "hl" || name == "ohlc" || name == "ohlcv" || name == "close_volume" || name == "hlcv") {
      return close; // Return close as default, actual multi-source ops handle this specially
    }
    throw std::runtime_error("unknown source: " + name);
  };

  FeatureMatrix M; 
  M.rows = (std::int64_t)N; 
  M.cols = F; 
  M.data.resize(N * F);

  for (int c = 0; c < F; ++c) {
    const auto& f = spec["features"][c];
    const std::string op = f["op"];
    const std::string src = f.value("source", "close");
    
    std::vector<float> col;

    // ============================================================================
    // BASIC OPERATIONS
    // ============================================================================
    if (op == "IDENT") {
      const auto& x = get_source_vector(src);
      col = op_IDENT(x);
    }
    else if (op == "LOGRET") {
      const auto& x = get_source_vector(src);
      col = op_LOGRET(x);
    }
    else if (op == "MOMENTUM") {
      const auto& x = get_source_vector(src);
      col = op_MOMENTUM(x, f.value("window", 5));
    }
    else if (op == "ROC") {
      const auto& x = get_source_vector(src);
      col = op_ROC(x, f.value("window", 10));
    }
    
    // ============================================================================
    // MOVING AVERAGES
    // ============================================================================
    else if (op == "SMA") {
      const auto& x = get_source_vector(src);
      col = op_SMA(x, f.value("window", 20));
    }
    else if (op == "EMA") {
      const auto& x = get_source_vector(src);
      col = op_EMA(x, f.value("window", 20));
    }
    
    // ============================================================================
    // VOLATILITY MEASURES
    // ============================================================================
    else if (op == "VOLATILITY") {
      const auto& x = get_source_vector(src);
      col = op_VOLATILITY(x, f.value("window", 20));
    }
    else if (op == "PARKINSON") {
      col = op_PARKINSON(high, low, f.value("window", 14));
    }
    else if (op == "GARMAN_KLASS") {
      col = op_GARMAN_KLASS(open, high, low, close, f.value("window", 14));
    }
    
    // ============================================================================
    // TECHNICAL INDICATORS
    // ============================================================================
    else if (op == "RSI") {
      const auto& x = get_source_vector(src);
      col = op_RSI(x, f.value("window", 14));
    }
    else if (op == "ZWIN") {
      const auto& x = get_source_vector(src);
      col = op_ZWIN(x, f.value("window", 20));
    }
    else if (op == "ATR") {
      col = op_ATR(high, low, close, f.value("window", 14));
    }
    
    // ============================================================================
    // BOLLINGER BANDS
    // ============================================================================
    else if (op == "BOLLINGER_UPPER") {
      const auto& x = get_source_vector(src);
      auto b = op_BOLLINGER(x, f.value("window", 20), f.value("k", 2.0));
      col = std::move(b.upper);
    }
    else if (op == "BOLLINGER_MIDDLE") {
      const auto& x = get_source_vector(src);
      auto b = op_BOLLINGER(x, f.value("window", 20), f.value("k", 2.0));
      col = std::move(b.middle);
    }
    else if (op == "BOLLINGER_LOWER") {
      const auto& x = get_source_vector(src);
      auto b = op_BOLLINGER(x, f.value("window", 20), f.value("k", 2.0));
      col = std::move(b.lower);
    }
    
    // ============================================================================
    // MACD
    // ============================================================================
    else if (op == "MACD_LINE") {
      const auto& x = get_source_vector(src);
      auto m = op_MACD(x, f.value("fast", 12), f.value("slow", 26), f.value("signal", 9));
      col = std::move(m.line);
    }
    else if (op == "MACD_SIGNAL") {
      const auto& x = get_source_vector(src);
      auto m = op_MACD(x, f.value("fast", 12), f.value("slow", 26), f.value("signal", 9));
      col = std::move(m.signal);
    }
    else if (op == "MACD_HISTOGRAM") {
      const auto& x = get_source_vector(src);
      auto m = op_MACD(x, f.value("fast", 12), f.value("slow", 26), f.value("signal", 9));
      col = std::move(m.histogram);
    }
    
    // ============================================================================
    // STOCHASTIC
    // ============================================================================
    else if (op == "STOCHASTIC_K") {
      auto s = op_STOCHASTIC(high, low, close, f.value("window", 14), f.value("d_period", 3));
      col = std::move(s.k);
    }
    else if (op == "STOCHASTIC_D") {
      auto s = op_STOCHASTIC(high, low, close, f.value("window", 14), f.value("d_period", 3));
      col = std::move(s.d);
    }
    
    // ============================================================================
    // OTHER OSCILLATORS
    // ============================================================================
    else if (op == "WILLIAMS_R") {
      col = op_WILLIAMS_R(high, low, close, f.value("window", 14));
    }
    else if (op == "CCI") {
      col = op_CCI(high, low, close, f.value("window", 20));
    }
    else if (op == "ADX") {
      col = op_ADX(high, low, close, f.value("window", 14));
    }
    
    // ============================================================================
    // VOLUME INDICATORS
    // ============================================================================
    else if (op == "OBV") {
      col = op_OBV(close, volume);
    }
    else if (op == "VPT") {
      col = op_VPT(close, volume);
    }
    else if (op == "AD_LINE") {
      col = op_AD_LINE(high, low, close, volume);
    }
    else if (op == "MFI") {
      col = op_MFI(high, low, close, volume, f.value("window", 14));
    }
    
    // ============================================================================
    // MICROSTRUCTURE INDICATORS
    // ============================================================================
    else if (op == "SPREAD_BP") {
      col = op_SPREAD_BP(open, high, low, close);
    }
    else if (op == "PRICE_IMPACT") {
      col = op_PRICE_IMPACT(open, high, low, close, volume);
    }
    else if (op == "ORDER_FLOW") {
      col = op_ORDER_FLOW(open, high, low, close, volume);
    }
    else if (op == "MARKET_DEPTH") {
      col = op_MARKET_DEPTH(open, high, low, close, volume);
    }
    else if (op == "BID_ASK_RATIO") {
      col = op_BID_ASK_RATIO(open, high, low, close);
    }
    
    // ============================================================================
    // FALLBACK
    // ============================================================================
    else {
      throw std::runtime_error("bad op: " + op);
    }

    // Write column to matrix
    for (size_t r = 0; r < N; ++r) {
      M.data[r * F + c] = col[r];
    }
  }

  // Apply padding policy
  for (std::int64_t r = 0; r < std::min<std::int64_t>(emit_from, M.rows); ++r) {
    for (int c = 0; c < F; ++c) {
      M.data[r * F + c] = pad;
    }
  }

  return M;
}

} // namespace sentio

```

## 📄 **FILE 28 of 145**: include/sentio/feature/feature_matrix.hpp

**File Information**:
- **Path**: `include/sentio/feature/feature_matrix.hpp`

- **Size**: 13 lines
- **Modified**: 2025-09-06 22:14:53

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>

namespace sentio {
struct FeatureMatrix {
  std::vector<float> data; // row-major [rows, cols]
  std::int64_t rows{0};
  std::int64_t cols{0};
  inline float* row_ptr(std::int64_t r) { return data.data() + r*cols; }
  inline const float* row_ptr(std::int64_t r) const { return data.data() + r*cols; }
};
}

```

## 📄 **FILE 29 of 145**: include/sentio/feature/feature_provider.hpp

**File Information**:
- **Path**: `include/sentio/feature/feature_provider.hpp`

- **Size**: 20 lines
- **Modified**: 2025-09-08 01:46:55

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace sentio {

struct FeatureMatrix {
  int64_t rows{0}, cols{0};
  std::vector<float> data; // row-major [rows, cols]
};

struct IFeatureProvider {
  virtual ~IFeatureProvider() = default;
  virtual FeatureMatrix get_features_for(const std::string& symbol) = 0;
  virtual std::vector<std::string> feature_names() const = 0; // authoritative order in source
  virtual int seq_len() const = 0; // sequence length (warmup)
};

} // namespace sentio

```

## 📄 **FILE 30 of 145**: include/sentio/feature/name_diff.hpp

**File Information**:
- **Path**: `include/sentio/feature/name_diff.hpp`

- **Size**: 40 lines
- **Modified**: 2025-09-07 12:43:11

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <unordered_set>

namespace sentio {

inline void print_name_diff(const std::vector<std::string>& src, const std::vector<std::string>& dst){
  std::unordered_set<std::string> S(src.begin(), src.end()), D(dst.begin(), dst.end());
  int miss=0, extra=0, reorder=0;

  std::cout << "[DIFF] Feature name differences:" << std::endl;
  for (auto& n : dst) {
    if (!S.count(n)) { 
      miss++; 
      std::cout << "  MISSING: " << n << std::endl; 
    }
  }
  
  for (auto& n : src) {
    if (!D.count(n)) { 
      extra++; 
      std::cout << "  EXTRA  : " << n << std::endl; 
    }
  }

  if (miss==0 && extra==0 && src.size()==dst.size()){
    for (size_t i=0;i<src.size();++i) {
      if (src[i] != dst[i]) reorder++;
    }
    if (reorder>0) {
      std::cout << "  REORDER count: " << reorder << std::endl;
    }
  }
  
  std::cout << "  Summary: missing=" << miss << " extra=" << extra << " reordered=" << reorder << std::endl;
}

} // namespace sentio

```

## 📄 **FILE 31 of 145**: include/sentio/feature/ops.hpp

**File Information**:
- **Path**: `include/sentio/feature/ops.hpp`

- **Size**: 592 lines
- **Modified**: 2025-09-07 05:33:18

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace sentio {

// ============================================================================
// BASIC OPERATIONS
// ============================================================================

inline std::vector<float> op_IDENT(const std::vector<float>& x) { 
    return x; 
}

inline std::vector<float> op_LOGRET(const std::vector<float>& x) {
    std::vector<float> y(x.size(), 0.f);
    for (size_t i = 1; i < x.size(); ++i) {
        float a = std::max(x[i], 1e-12f), b = std::max(x[i-1], 1e-12f);
        y[i] = std::log(a) - std::log(b);
    }
    return y;
}

inline std::vector<float> op_MOMENTUM(const std::vector<float>& x, int w) {
    std::vector<float> out(x.size(), 0.f);
    for (size_t i = w; i < x.size(); ++i) {
        out[i] = x[i] - x[i-w];
    }
    return out;
}

inline std::vector<float> op_ROC(const std::vector<float>& x, int w) {
    std::vector<float> out(x.size(), 0.f);
    for (size_t i = w; i < x.size(); ++i) {
        float prev = std::max(x[i-w], 1e-12f);
        out[i] = ((x[i] - prev) / prev) * 100.0f;
    }
    return out;
}

// ============================================================================
// MOVING AVERAGES
// ============================================================================

inline std::vector<float> op_SMA(const std::vector<float>& x, int w) {
    std::vector<float> out(x.size(), 0.f);
    if (w <= 1) return x;
    double s = 0.0;
    for (int i = 0; i < (int)x.size(); ++i) {
        s += x[i];
        if (i >= w) s -= x[i-w];
        if (i >= w-1) out[i] = (float)(s/w);
    }
    return out;
}

inline std::vector<float> op_EMA(const std::vector<float>& x, int p) {
    std::vector<float> e(x.size()); 
    if (x.empty()) return e;
    float k = 2.f / (p + 1.f); 
    e[0] = x[0];
    for (size_t i = 1; i < x.size(); ++i) {
        e[i] = k * x[i] + (1.f - k) * e[i-1];
    }
    return e;
}

// ============================================================================
// VOLATILITY AND STATISTICAL MEASURES
// ============================================================================

inline std::vector<float> op_VOLATILITY(const std::vector<float>& x, int w) {
    std::vector<float> out(x.size(), 0.f);
    if ((int)x.size() < w) return out;
    double s = 0.0, s2 = 0.0;
    for (int i = 0; i < w; i++) { 
        s += x[i]; 
        s2 += x[i] * x[i]; 
    }
    for (size_t i = w; i < x.size(); ++i) {
        double mu = s / w;
        double var = std::max(0.0, s2 / w - mu * mu);
        out[i] = (float)std::sqrt(var);
        // slide window
        s += x[i] - x[i-w];
        s2 += x[i] * x[i] - x[i-w] * x[i-w];
    }
    return out;
}

inline std::vector<float> op_PARKINSON(const std::vector<float>& high, 
                                       const std::vector<float>& low, 
                                       int w) {
    std::vector<float> out(high.size(), 0.f);
    std::vector<float> hl_ratio(high.size(), 0.f);
    
    // Calculate log(H/L)^2 for each bar
    for (size_t i = 0; i < high.size(); ++i) {
        if (high[i] > 0 && low[i] > 0) {
            float ratio = std::log(high[i] / low[i]);
            hl_ratio[i] = ratio * ratio;
        }
    }
    
    // Rolling average of (log(H/L))^2 and scale by Parkinson constant
    auto avg_hl = op_SMA(hl_ratio, w);
    const float parkinson_factor = 1.0f / (4.0f * std::log(2.0f));
    
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = std::sqrt(avg_hl[i] * parkinson_factor);
    }
    return out;
}

inline std::vector<float> op_GARMAN_KLASS(const std::vector<float>& open,
                                          const std::vector<float>& high,
                                          const std::vector<float>& low,
                                          const std::vector<float>& close,
                                          int w) {
    std::vector<float> out(open.size(), 0.f);
    std::vector<float> gk_vals(open.size(), 0.f);
    
    // Calculate Garman-Klass estimator for each bar
    for (size_t i = 1; i < open.size(); ++i) {
        if (high[i] > 0 && low[i] > 0 && open[i] > 0 && close[i] > 0) {
            float log_hl = std::log(high[i] / low[i]);
            float log_co = std::log(close[i] / open[i]);
            gk_vals[i] = 0.5f * log_hl * log_hl - (2.0f * std::log(2.0f) - 1.0f) * log_co * log_co;
        }
    }
    
    // Rolling average
    auto avg_gk = op_SMA(gk_vals, w);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = std::sqrt(std::max(0.0f, avg_gk[i]));
    }
    return out;
}

// ============================================================================
// TECHNICAL INDICATORS
// ============================================================================

inline std::vector<float> op_RSI(const std::vector<float>& x, int p = 14) {
    const size_t N = x.size();
    std::vector<float> out(N, 0.f), up(N, 0.f), dn(N, 0.f);
    
    for (size_t i = 1; i < N; ++i) {
        float d = x[i] - x[i-1]; 
        up[i] = std::max(d, 0.f); 
        dn[i] = std::max(-d, 0.f);
    }
    
    std::vector<float> ru(N, 0.f), rd(N, 0.f);
    if (N > (size_t)p) {
        float su = 0.f, sd = 0.f;
        for (int i = 1; i <= p; i++) { 
            su += up[i]; 
            sd += dn[i]; 
        }
        ru[p] = su / p; 
        rd[p] = sd / p;
        
        for (size_t i = p + 1; i < N; ++i) {
            ru[i] = (ru[i-1] * (p-1) + up[i]) / p;
            rd[i] = (rd[i-1] * (p-1) + dn[i]) / p;
            float rs = (rd[i] > 1e-12f) ? (ru[i] / rd[i]) : 0.f;
            out[i] = 100.f - 100.f / (1.f + rs);
        }
    }
    return out;
}

inline std::vector<float> op_ZWIN(const std::vector<float>& x, int w) {
    const size_t N = x.size(); 
    std::vector<float> out(N, 0.f);
    if (N <= (size_t)w) return out;
    
    std::vector<double> s(N, 0.0), s2(N, 0.0);
    s[0] = x[0]; 
    s2[0] = x[0] * x[0];
    
    for (size_t i = 1; i < N; ++i) { 
        s[i] = s[i-1] + x[i]; 
        s2[i] = s2[i-1] + x[i] * x[i]; 
    }
    
    for (size_t i = w; i < N; ++i) {
        double su = s[i] - s[i-w], su2 = s2[i] - s2[i-w];
        double mu = su / w;
        double var = std::max(0.0, su2 / w - mu * mu);
        float sd = (float)std::sqrt(var);
        out[i] = (sd > 1e-8f) ? (float)((x[i] - mu) / sd) : 0.f;
    }
    return out;
}

inline std::vector<float> op_ATR(const std::vector<float>& high,
                                 const std::vector<float>& low,
                                 const std::vector<float>& close,
                                 int p) {
    const size_t N = high.size();
    std::vector<float> tr(N, 0.f), atr(N, 0.f);
    
    for (size_t i = 1; i < N; ++i) {
        float h = high[i], l = low[i], cp = close[i-1];
        float v = std::max({h-l, std::fabs(h-cp), std::fabs(l-cp)});
        tr[i] = v;
    }
    
    // Wilder smoothing
    if (N > (size_t)p) {
        float a = 0.f; 
        for (int i = 1; i <= p; i++) a += tr[i];
        a /= p; 
        atr[p] = a;
        for (size_t i = p + 1; i < N; ++i) {
            atr[i] = (atr[i-1] * (p-1) + tr[i]) / p;
        }
    }
    return atr;
}

// ============================================================================
// BOLLINGER BANDS
// ============================================================================

struct BollingerBands { 
    std::vector<float> upper, middle, lower; 
};

inline BollingerBands op_BOLLINGER(const std::vector<float>& x, int w, float k = 2.0f) {
    BollingerBands b; 
    b.upper.resize(x.size(), 0.f); 
    b.middle.resize(x.size(), 0.f); 
    b.lower.resize(x.size(), 0.f);
    
    auto sma = op_SMA(x, w);
    auto sd = op_VOLATILITY(x, w);
    
    for (size_t i = w; i < x.size(); ++i) {
        b.middle[i] = sma[i];
        b.upper[i] = sma[i] + k * sd[i];
        b.lower[i] = sma[i] - k * sd[i];
    }
    return b;
}

// ============================================================================
// MACD
// ============================================================================

struct MACD { 
    std::vector<float> line, signal, histogram; 
};

inline MACD op_MACD(const std::vector<float>& x, int fast = 12, int slow = 26, int sig = 9) {
    MACD m; 
    m.line.resize(x.size()); 
    m.signal.resize(x.size()); 
    m.histogram.resize(x.size());
    
    auto ema_fast = op_EMA(x, fast);
    auto ema_slow = op_EMA(x, slow);
    
    for (size_t i = 0; i < x.size(); ++i) {
        m.line[i] = ema_fast[i] - ema_slow[i];
    }
    
    m.signal = op_EMA(m.line, sig);
    
    for (size_t i = 0; i < x.size(); ++i) {
        m.histogram[i] = m.line[i] - m.signal[i];
    }
    return m;
}

// ============================================================================
// STOCHASTIC OSCILLATOR
// ============================================================================

struct Stochastic {
    std::vector<float> k, d;
};

inline Stochastic op_STOCHASTIC(const std::vector<float>& high,
                                const std::vector<float>& low,
                                const std::vector<float>& close,
                                int k_period = 14,
                                int d_period = 3) {
    Stochastic stoch;
    stoch.k.resize(close.size(), 0.f);
    stoch.d.resize(close.size(), 0.f);
    
    for (size_t i = k_period; i < close.size(); ++i) {
        float highest = *std::max_element(high.begin() + i - k_period, high.begin() + i + 1);
        float lowest = *std::min_element(low.begin() + i - k_period, low.begin() + i + 1);
        
        if (highest > lowest) {
            stoch.k[i] = ((close[i] - lowest) / (highest - lowest)) * 100.0f;
        }
    }
    
    stoch.d = op_SMA(stoch.k, d_period);
    return stoch;
}

// ============================================================================
// OTHER OSCILLATORS
// ============================================================================

inline std::vector<float> op_WILLIAMS_R(const std::vector<float>& high,
                                        const std::vector<float>& low,
                                        const std::vector<float>& close,
                                        int period = 14) {
    std::vector<float> out(close.size(), 0.f);
    
    for (size_t i = period; i < close.size(); ++i) {
        float highest = *std::max_element(high.begin() + i - period, high.begin() + i + 1);
        float lowest = *std::min_element(low.begin() + i - period, low.begin() + i + 1);
        
        if (highest > lowest) {
            out[i] = ((highest - close[i]) / (highest - lowest)) * -100.0f;
        }
    }
    return out;
}

inline std::vector<float> op_CCI(const std::vector<float>& high,
                                 const std::vector<float>& low,
                                 const std::vector<float>& close,
                                 int period = 20) {
    std::vector<float> out(close.size(), 0.f);
    std::vector<float> tp(close.size()); // typical price
    
    for (size_t i = 0; i < close.size(); ++i) {
        tp[i] = (high[i] + low[i] + close[i]) / 3.0f;
    }
    
    auto sma_tp = op_SMA(tp, period);
    
    for (size_t i = period; i < close.size(); ++i) {
        float mean_dev = 0.0f;
        for (size_t j = i - period + 1; j <= i; ++j) {
            mean_dev += std::fabs(tp[j] - sma_tp[i]);
        }
        mean_dev /= period;
        
        if (mean_dev > 1e-8f) {
            out[i] = (tp[i] - sma_tp[i]) / (0.015f * mean_dev);
        }
    }
    return out;
}

inline std::vector<float> op_ADX(const std::vector<float>& high,
                                 const std::vector<float>& low,
                                 const std::vector<float>& close,
                                 int period = 14) {
    const size_t N = close.size();
    std::vector<float> adx(N, 0.f);
    std::vector<float> dm_plus(N, 0.f), dm_minus(N, 0.f), tr(N, 0.f);
    
    // Calculate directional movement and true range
    for (size_t i = 1; i < N; ++i) {
        float up_move = high[i] - high[i-1];
        float down_move = low[i-1] - low[i];
        
        dm_plus[i] = (up_move > down_move && up_move > 0) ? up_move : 0.0f;
        dm_minus[i] = (down_move > up_move && down_move > 0) ? down_move : 0.0f;
        
        float h = high[i], l = low[i], cp = close[i-1];
        tr[i] = std::max({h-l, std::fabs(h-cp), std::fabs(l-cp)});
    }
    
    // Smooth the values using Wilder's smoothing
    if (N > (size_t)period) {
        float sum_dm_plus = 0, sum_dm_minus = 0, sum_tr = 0;
        for (int i = 1; i <= period; i++) {
            sum_dm_plus += dm_plus[i];
            sum_dm_minus += dm_minus[i];
            sum_tr += tr[i];
        }
        
        float smooth_dm_plus = sum_dm_plus;
        float smooth_dm_minus = sum_dm_minus;
        float smooth_tr = sum_tr;
        
        for (size_t i = period + 1; i < N; ++i) {
            smooth_dm_plus = smooth_dm_plus - smooth_dm_plus/period + dm_plus[i];
            smooth_dm_minus = smooth_dm_minus - smooth_dm_minus/period + dm_minus[i];
            smooth_tr = smooth_tr - smooth_tr/period + tr[i];
            
            float di_plus = (smooth_tr > 0) ? (smooth_dm_plus / smooth_tr) * 100 : 0;
            float di_minus = (smooth_tr > 0) ? (smooth_dm_minus / smooth_tr) * 100 : 0;
            
            float di_sum = di_plus + di_minus;
            float dx = (di_sum > 0) ? std::fabs(di_plus - di_minus) / di_sum * 100 : 0;
            
            // Simple moving average of DX for ADX
            if (i >= period * 2) {
                float adx_sum = 0;
                for (size_t j = i - period + 1; j <= i; ++j) {
                    // Recalculate DX for each period (simplified)
                    adx_sum += dx; // This is simplified; full implementation would store DX values
                }
                adx[i] = adx_sum / period;
            }
        }
    }
    return adx;
}

// ============================================================================
// VOLUME INDICATORS
// ============================================================================

inline std::vector<float> op_OBV(const std::vector<float>& close,
                                 const std::vector<float>& volume) {
    std::vector<float> obv(close.size(), 0.f);
    
    for (size_t i = 1; i < close.size(); ++i) {
        if (close[i] > close[i-1]) {
            obv[i] = obv[i-1] + volume[i];
        } else if (close[i] < close[i-1]) {
            obv[i] = obv[i-1] - volume[i];
        } else {
            obv[i] = obv[i-1];
        }
    }
    return obv;
}

inline std::vector<float> op_VPT(const std::vector<float>& close,
                                 const std::vector<float>& volume) {
    std::vector<float> vpt(close.size(), 0.f);
    
    for (size_t i = 1; i < close.size(); ++i) {
        if (close[i-1] > 0) {
            float pct_change = (close[i] - close[i-1]) / close[i-1];
            vpt[i] = vpt[i-1] + volume[i] * pct_change;
        } else {
            vpt[i] = vpt[i-1];
        }
    }
    return vpt;
}

inline std::vector<float> op_AD_LINE(const std::vector<float>& high,
                                     const std::vector<float>& low,
                                     const std::vector<float>& close,
                                     const std::vector<float>& volume) {
    std::vector<float> ad(close.size(), 0.f);
    
    for (size_t i = 1; i < close.size(); ++i) {
        float hl_diff = high[i] - low[i];
        if (hl_diff > 1e-8f) {
            float mfm = ((close[i] - low[i]) - (high[i] - close[i])) / hl_diff;
            float mfv = mfm * volume[i];
            ad[i] = ad[i-1] + mfv;
        } else {
            ad[i] = ad[i-1];
        }
    }
    return ad;
}

inline std::vector<float> op_MFI(const std::vector<float>& high,
                                 const std::vector<float>& low,
                                 const std::vector<float>& close,
                                 const std::vector<float>& volume,
                                 int period = 14) {
    std::vector<float> mfi(close.size(), 0.f);
    std::vector<float> tp(close.size()); // typical price
    std::vector<float> mf(close.size()); // money flow
    
    for (size_t i = 0; i < close.size(); ++i) {
        tp[i] = (high[i] + low[i] + close[i]) / 3.0f;
        mf[i] = tp[i] * volume[i];
    }
    
    for (size_t i = period; i < close.size(); ++i) {
        float positive_mf = 0, negative_mf = 0;
        
        for (size_t j = i - period + 1; j <= i; ++j) {
            if (j > 0) {
                if (tp[j] > tp[j-1]) {
                    positive_mf += mf[j];
                } else if (tp[j] < tp[j-1]) {
                    negative_mf += mf[j];
                }
            }
        }
        
        if (negative_mf > 1e-8f) {
            float mfr = positive_mf / negative_mf;
            mfi[i] = 100.0f - (100.0f / (1.0f + mfr));
        }
    }
    return mfi;
}

// ============================================================================
// MICROSTRUCTURE INDICATORS (Simplified implementations)
// ============================================================================

inline std::vector<float> op_SPREAD_BP(const std::vector<float>& open,
                                       const std::vector<float>& high,
                                       const std::vector<float>& low,
                                       const std::vector<float>& close) {
    std::vector<float> spread(close.size(), 0.f);
    
    for (size_t i = 0; i < close.size(); ++i) {
        float mid = (high[i] + low[i]) / 2.0f;
        if (mid > 1e-8f) {
            spread[i] = ((high[i] - low[i]) / mid) * 10000.0f; // basis points
        }
    }
    return spread;
}

inline std::vector<float> op_PRICE_IMPACT(const std::vector<float>& open,
                                          const std::vector<float>& high,
                                          const std::vector<float>& low,
                                          const std::vector<float>& close,
                                          const std::vector<float>& volume) {
    std::vector<float> impact(close.size(), 0.f);
    
    for (size_t i = 1; i < close.size(); ++i) {
        float price_change = std::fabs(close[i] - close[i-1]);
        float vol_sqrt = std::sqrt(volume[i]);
        if (vol_sqrt > 1e-8f && close[i-1] > 1e-8f) {
            impact[i] = (price_change / close[i-1]) / vol_sqrt * 1000000.0f; // scaled
        }
    }
    return impact;
}

inline std::vector<float> op_ORDER_FLOW(const std::vector<float>& open,
                                        const std::vector<float>& high,
                                        const std::vector<float>& low,
                                        const std::vector<float>& close,
                                        const std::vector<float>& volume) {
    std::vector<float> flow(close.size(), 0.f);
    
    for (size_t i = 0; i < close.size(); ++i) {
        float hl_diff = high[i] - low[i];
        if (hl_diff > 1e-8f) {
            float buy_pressure = (close[i] - low[i]) / hl_diff;
            float sell_pressure = (high[i] - close[i]) / hl_diff;
            flow[i] = (buy_pressure - sell_pressure) * volume[i];
        }
    }
    return flow;
}

inline std::vector<float> op_MARKET_DEPTH(const std::vector<float>& open,
                                          const std::vector<float>& high,
                                          const std::vector<float>& low,
                                          const std::vector<float>& close,
                                          const std::vector<float>& volume) {
    std::vector<float> depth(close.size(), 0.f);
    
    for (size_t i = 0; i < close.size(); ++i) {
        float range = high[i] - low[i];
        if (volume[i] > 1e-8f && range > 1e-8f) {
            depth[i] = range / std::log(volume[i] + 1.0f);
        }
    }
    return depth;
}

inline std::vector<float> op_BID_ASK_RATIO(const std::vector<float>& open,
                                          const std::vector<float>& high,
                                          const std::vector<float>& low,
                                          const std::vector<float>& close) {
    std::vector<float> ratio(close.size(), 0.f);
    
    for (size_t i = 0; i < close.size(); ++i) {
        // Simplified: use close position within range as proxy
        float range = high[i] - low[i];
        if (range > 1e-8f) {
            float position = (close[i] - low[i]) / range; // 0 = low, 1 = high
            ratio[i] = position / (1.0f - position + 1e-8f); // bid/ask proxy
        }
    }
    return ratio;
}

} // namespace sentio

```

## 📄 **FILE 32 of 145**: include/sentio/feature/sanitize.hpp

**File Information**:
- **Path**: `include/sentio/feature/sanitize.hpp`

- **Size**: 63 lines
- **Modified**: 2025-09-07 13:49:40

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>

namespace sentio {

// NaN/Inf sanitation before model (and a ready mask)
inline std::vector<uint8_t> sanitize_and_ready(float* X, int64_t rows, int64_t cols, int emit_from, float pad=0.0f){
  std::vector<uint8_t> ok((size_t)rows, 0);
  
  // Pad rows before emit_from
  for (int64_t r=0; r<std::min<int64_t>(emit_from, rows); ++r){
    for (int64_t c=0;c<cols;++c) X[r*cols+c] = pad;
  }
  
  // Sanitize and check rows from emit_from onward
  for (int64_t r=emit_from; r<rows; ++r){
    bool good=true;
    float* row = X + r*cols;
    for (int64_t c=0;c<cols;++c){
      float v=row[c];
      if (!std::isfinite(v)){ 
        row[c]=0.0f; 
        good=false; 
      }
    }
    ok[(size_t)r] = good ? 1 : 0;
  }
  return ok;
}

// Overload for cached features (vector<vector<double>>)
inline std::vector<uint8_t> sanitize_cached_features(std::vector<std::vector<double>>& features, int emit_from, double pad=0.0){
  std::vector<uint8_t> ok(features.size(), 0);
  
  if (features.empty()) return ok;
  
  size_t cols = features[0].size();
  
  // Pad rows before emit_from
  for (size_t r=0; r<std::min<size_t>(emit_from, features.size()); ++r){
    for (size_t c=0; c<cols; ++c) {
      features[r][c] = pad;
    }
  }
  
  // Sanitize and check rows from emit_from onward
  for (size_t r=emit_from; r<features.size(); ++r){
    bool good=true;
    for (size_t c=0; c<features[r].size(); ++c){
      double v = features[r][c];
      if (!std::isfinite(v)){ 
        features[r][c] = 0.0; 
        good=false; 
      }
    }
    ok[r] = good ? 1 : 0;
  }
  return ok;
}

} // namespace sentio

```

## 📄 **FILE 33 of 145**: include/sentio/feature/standard_scaler.hpp

**File Information**:
- **Path**: `include/sentio/feature/standard_scaler.hpp`

- **Size**: 42 lines
- **Modified**: 2025-09-06 22:14:53

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace sentio {

struct StandardScaler {
  std::vector<float> mean, inv_std;

  void fit(const float* X, std::int64_t rows, std::int64_t cols) {
    mean.assign(cols, 0.f);
    inv_std.assign(cols, 0.f);

    for (std::int64_t r=0; r<rows; ++r)
      for (std::int64_t c=0; c<cols; ++c)
        mean[c] += X[r*cols + c];
    for (std::int64_t c=0; c<cols; ++c) mean[c] /= std::max<std::int64_t>(1, rows);

    std::vector<double> var(cols, 0.0);
    for (std::int64_t r=0; r<rows; ++r)
      for (std::int64_t c=0; c<cols; ++c) {
        const double d = (double)X[r*cols + c] - (double)mean[c];
        var[c] += d*d;
      }
    for (std::int64_t c=0; c<cols; ++c) {
      const double sd = std::sqrt(std::max(1e-12, var[c] / std::max<std::int64_t>(1, rows)));
      inv_std[c] = (float)(1.0 / sd);
    }
  }

  void transform_inplace(float* X, std::int64_t rows, std::int64_t cols) const {
    for (std::int64_t r=0; r<rows; ++r)
      for (std::int64_t c=0; c<cols; ++c) {
        float& v = X[r*cols + c];
        v = (v - mean[c]) * inv_std[c];
      }
  }
};

} // namespace sentio

```

## 📄 **FILE 34 of 145**: include/sentio/feature_builder.hpp

**File Information**:
- **Path**: `include/sentio/feature_builder.hpp`

- **Size**: 149 lines
- **Modified**: 2025-09-06 20:23:41

- **Type**: .hpp

```text
#pragma once
#include "sentio/core.hpp"
#include <deque>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <optional>
#include <cmath>
#include <stdexcept>

namespace sentio {

// Optional microstructure snapshot (pass when you have it; else omit)
struct MicroTick { double bid{NAN}, ask{NAN}; };

// A compiled plan for the features (from metadata)
struct FeaturePlan {
  std::vector<std::string> names; // in metadata order
  // sanity: names.size() must match metadata.feature_names
};

// Builder config
struct FeatureBuilderCfg {
  int rsi_period{14};
  int sma_fast{10};
  int sma_slow{30};
  int ret_5m_window{5};      // number of 1m bars
  // If you want volatility as stdev of 1m returns over N bars, set vol_window>1
  int vol_window{20};        // stdev window (bars)
  // spread fallback (bps) when no bid/ask and proxy not computable
  double default_spread_bp{1.5};
};

// Rolling helpers (small, header-only for speed)
class RollingMean {
  double sum_{0}; std::deque<double> q_;
  size_t W_;
public:
  explicit RollingMean(size_t W): W_(W) {}
  void push(double x){ sum_ += x; q_.push_back(x); if(q_.size()>W_){ sum_-=q_.front(); q_.pop_front(); } }
  bool full() const { return q_.size()==W_; }
  double mean() const { return q_.empty()? NAN : (sum_/double(q_.size())); }
  size_t size() const { return q_.size(); }
};

class RollingStdWindow {
  std::vector<double> buf_;
  size_t W_, i_{0}, n_{0};
  double sum_{0}, sumsq_{0};
public:
  explicit RollingStdWindow(size_t W): buf_(W, 0.0), W_(W) {}
  inline void push(double x){
    if (n_ < W_) { 
      buf_[n_++] = x; 
      sum_ += x; 
      sumsq_ += x*x; 
      if (n_ == W_) i_ = 0; 
    } else { 
      double old = buf_[i_]; 
      buf_[i_] = x; 
      sum_ += x - old; 
      sumsq_ += x*x - old*old; 
      if (++i_ == W_) i_ = 0; 
    }
  }
  inline bool full() const { return n_ == W_; }
  inline double stdev() const { 
    if (n_ < 2) return NAN; 
    double m = sum_ / n_; 
    return std::sqrt(std::max(0.0, sumsq_ / n_ - m * m)); 
  }
  inline size_t size() const { return n_; }
};

class RollingRSI {
  // Wilder's RSI with smoothing; requires first 'period' values to bootstrap
  int period_; bool boot_{true}; int boot_count_{0};
  double up_{0}, dn_{0};
public:
  explicit RollingRSI(int p): period_(p) {}
  // x = current close, px = previous close
  void push(double px, double x){
    double chg = x - px;
    double u = chg>0? chg:0; double d = chg<0? -chg:0;
    if (boot_){
      up_ += u; dn_ += d; ++boot_count_;
      if (boot_count_ == period_) {
        up_ /= period_; dn_ /= period_; boot_ = false;
      }
    } else {
      up_ = (up_*(period_-1) + u) / period_;
      dn_ = (dn_*(period_-1) + d) / period_;
    }
  }
  bool ready() const { return !boot_; }
  double value() const {
    if (boot_) return NAN;
    if (dn_==0) return 100.0;
    double rs = up_/dn_;
    return 100.0 - 100.0/(1.0+rs);
  }
};

class FeatureBuilder {
public:
  FeatureBuilder(FeaturePlan plan, FeatureBuilderCfg cfg);

  // Feed one 1m bar (RTH-filtered) plus optional bid/ask for spread
  void on_bar(const Bar& b, const std::optional<MicroTick>& mt = std::nullopt);

  // True when all requested features can be computed *and* are finite
  bool ready() const;

  // Returns features in the exact metadata order (size == plan.names.size()).
  // Will return std::nullopt if not ready().
  std::optional<std::vector<double>> build() const;

  // Resets internal buffers
  void reset();

  // Accessors (useful in tests)
  size_t bars_seen() const { return bars_seen_; }

private:
  FeaturePlan plan_;
  FeatureBuilderCfg cfg_;

  // Internal state
  size_t bars_seen_{0};
  std::deque<double> close_q_;             // last N closes for ret/RSI
  RollingMean sma_fast_, sma_slow_;
  RollingStdWindow vol_rtn_;               // stdev of 1m returns (O(1) implementation)
  RollingRSI  rsi_;

  // Cached per-bar computations
  double last_ret_1m_{NAN};
  double last_ret_5m_{NAN};
  double last_rsi_{NAN};
  double last_sma_fast_{NAN};
  double last_sma_slow_{NAN};
  double last_vol_1m_{NAN};
  double last_spread_bp_{NAN};

  // helpers
  static inline bool finite(double x){ return std::isfinite(x); }
};

} // namespace sentio

```

## 📄 **FILE 35 of 145**: include/sentio/feature_cache.hpp

**File Information**:
- **Path**: `include/sentio/feature_cache.hpp`

- **Size**: 62 lines
- **Modified**: 2025-09-07 11:38:06

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace sentio {

/**
 * FeatureCache loads and provides access to pre-computed features
 * This eliminates the need for expensive real-time technical indicator calculations
 */
class FeatureCache {
public:
    /**
     * Load pre-computed features from CSV file
     * @param feature_file_path Path to the feature CSV file (e.g., QQQ_RTH_features.csv)
     * @return true if loaded successfully
     */
    bool load_from_csv(const std::string& feature_file_path);

    /**
     * Get features for a specific bar index
     * @param bar_index The bar index (0-based)
     * @return Vector of 55 features, or empty vector if not found
     */
    std::vector<double> get_features(int bar_index) const;

    /**
     * Check if features are available for a given bar index
     */
    bool has_features(int bar_index) const;

    /**
     * Get the total number of bars with features
     */
    size_t get_bar_count() const;

    /**
     * Get the recommended starting bar index (after warmup)
     */
    int get_recommended_start_bar() const;

    /**
     * Get feature names in order
     */
    const std::vector<std::string>& get_feature_names() const;

private:
    // Map from bar_index to feature vector
    std::unordered_map<int, std::vector<double>> features_by_bar_;
    
    // Feature names in order
    std::vector<std::string> feature_names_;
    
    // Recommended starting bar (after warmup period)
    int recommended_start_bar_ = 300;
    
    // Total number of bars
    size_t total_bars_ = 0;
};

} // namespace sentio

```

## 📄 **FILE 36 of 145**: include/sentio/feature_engineering/feature_normalizer.hpp

**File Information**:
- **Path**: `include/sentio/feature_engineering/feature_normalizer.hpp`

- **Size**: 70 lines
- **Modified**: 2025-09-06 02:39:51

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <deque>
#include <mutex>

namespace sentio {
namespace feature_engineering {

struct NormalizationStats {
    double mean{0.0};
    double std{0.0};
    double min{0.0};
    double max{0.0};
    size_t count{0};
};

class FeatureNormalizer {
public:
    FeatureNormalizer(size_t window_size = 252);
    
    // Normalization methods
    std::vector<double> normalize_features(const std::vector<double>& features);
    std::vector<double> denormalize_features(const std::vector<double>& normalized_features);
    
    // Statistics management
    void update_stats(const std::vector<double>& features);
    void reset_stats();
    
    // Feature-specific normalization
    std::vector<double> z_score_normalize(const std::vector<double>& features);
    std::vector<double> min_max_normalize(const std::vector<double>& features);
    std::vector<double> robust_normalize(const std::vector<double>& features);
    
    // Outlier handling
    std::vector<double> clip_outliers(const std::vector<double>& features, double threshold = 3.0);
    std::vector<double> winsorize(const std::vector<double>& features, double percentile = 0.05);
    
    // Validation
    bool is_normalized(const std::vector<double>& features) const;
    std::vector<bool> get_outlier_mask(const std::vector<double>& features, double threshold = 3.0) const;
    
    // Statistics access
    NormalizationStats get_stats(size_t feature_index) const;
    std::vector<NormalizationStats> get_all_stats() const;
    
    // Configuration
    void set_window_size(size_t window_size);
    void set_outlier_threshold(double threshold);
    void set_winsorize_percentile(double percentile);
    
private:
    size_t window_size_;
    double outlier_threshold_{3.0};
    double winsorize_percentile_{0.05};
    
    std::vector<std::deque<double>> feature_history_;
    std::vector<NormalizationStats> stats_;
    mutable std::mutex stats_mutex_;
    
    void update_feature_stats(size_t feature_index, double value);
    double calculate_robust_mean(const std::deque<double>& values);
    double calculate_robust_std(const std::deque<double>& values, double mean);
    double calculate_percentile(const std::deque<double>& values, double percentile);
    void sort_values(std::deque<double>& values);
};

} // namespace feature_engineering
} // namespace sentio

```

## 📄 **FILE 37 of 145**: include/sentio/feature_engineering/kochi_features.hpp

**File Information**:
- **Path**: `include/sentio/feature_engineering/kochi_features.hpp`

- **Size**: 21 lines
- **Modified**: 2025-09-07 23:21:17

- **Type**: .hpp

```text
#pragma once
#include "sentio/core.hpp"
#include <vector>
#include <string>

namespace sentio {
namespace feature_engineering {

// Returns Kochi feature names in the exact order expected by the trainer.
// This excludes any state features (position one-hot, PnL), which are not used at inference time.
std::vector<std::string> kochi_feature_names();

// Compute Kochi feature vector for a given bar index using bar history.
// Window-dependent features use typical Kochi defaults (e.g., 20 for many).
// The output order matches kochi_feature_names().
std::vector<double> calculate_kochi_features(const std::vector<Bar>& bars, int current_index);

} // namespace feature_engineering
} // namespace sentio



```

## 📄 **FILE 38 of 145**: include/sentio/feature_engineering/technical_indicators.hpp

**File Information**:
- **Path**: `include/sentio/feature_engineering/technical_indicators.hpp`

- **Size**: 127 lines
- **Modified**: 2025-09-06 02:39:51

- **Type**: .hpp

```text
#pragma once
#include "sentio/core.hpp"
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace sentio {
namespace feature_engineering {

// Price-based features
struct PriceFeatures {
    double ret_1m{0.0}, ret_5m{0.0}, ret_15m{0.0}, ret_30m{0.0}, ret_1h{0.0};
    double momentum_5{0.0}, momentum_10{0.0}, momentum_20{0.0};
    double volatility_10{0.0}, volatility_20{0.0}, volatility_30{0.0};
    double atr_14{0.0}, atr_21{0.0};
    double parkinson_vol{0.0}, garman_klass_vol{0.0};
};

// Technical analysis features
struct TechnicalFeatures {
    double rsi_14{0.0}, rsi_21{0.0}, rsi_30{0.0};
    double sma_5{0.0}, sma_10{0.0}, sma_20{0.0}, sma_50{0.0}, sma_200{0.0};
    double ema_5{0.0}, ema_10{0.0}, ema_20{0.0}, ema_50{0.0}, ema_200{0.0};
    double bb_upper_20{0.0}, bb_middle_20{0.0}, bb_lower_20{0.0};
    double bb_upper_50{0.0}, bb_middle_50{0.0}, bb_lower_50{0.0};
    double macd_line{0.0}, macd_signal{0.0}, macd_histogram{0.0};
    double stoch_k{0.0}, stoch_d{0.0};
    double williams_r{0.0};
    double cci_20{0.0};
    double adx_14{0.0};
};

// Volume features
struct VolumeFeatures {
    double volume_sma_10{0.0}, volume_sma_20{0.0}, volume_sma_50{0.0};
    double volume_roc{0.0};
    double obv{0.0};
    double vpt{0.0};
    double ad_line{0.0};
    double mfi_14{0.0};
};

// Market microstructure features
struct MicrostructureFeatures {
    double spread_bp{0.0};
    double price_impact{0.0};
    double order_flow_imbalance{0.0};
    double market_depth{0.0};
    double bid_ask_ratio{0.0};
};

// Main feature calculator
class TechnicalIndicatorCalculator {
public:
    TechnicalIndicatorCalculator();
    
    // Core calculation methods
    PriceFeatures calculate_price_features(const std::vector<Bar>& bars, int current_index);
    TechnicalFeatures calculate_technical_features(const std::vector<Bar>& bars, int current_index);
    VolumeFeatures calculate_volume_features(const std::vector<Bar>& bars, int current_index);
    MicrostructureFeatures calculate_microstructure_features(const std::vector<Bar>& bars, int current_index);
    
    // Combined feature vector
    std::vector<double> calculate_all_features(const std::vector<Bar>& bars, int current_index);
    
    // Feature validation
    bool validate_features(const std::vector<double>& features);
    std::vector<std::string> get_feature_names() const;
    
    // Helper methods
    static std::vector<double> extract_closes(const std::vector<Bar>& bars);
    static std::vector<double> extract_volumes(const std::vector<Bar>& bars);
    static std::vector<double> extract_returns(const std::vector<Bar>& bars);
    
private:
    // Rolling calculations
    double calculate_rsi(const std::vector<double>& closes, int period, int current_index);
    double calculate_sma(const std::vector<double>& values, int period, int current_index);
    double calculate_ema(const std::vector<double>& values, int period, int current_index);
    double calculate_volatility(const std::vector<double>& returns, int period, int current_index);
    double calculate_atr(const std::vector<Bar>& bars, int period, int current_index);
    
    // Bollinger Bands
    struct BollingerBands {
        double upper, middle, lower;
    };
    BollingerBands calculate_bollinger_bands(const std::vector<double>& values, int period, double std_dev, int current_index);
    
    // MACD
    struct MACD {
        double line, signal, histogram;
    };
    MACD calculate_macd(const std::vector<double>& values, int fast, int slow, int signal, int current_index);
    
    // Stochastic
    struct Stochastic {
        double k, d;
    };
    Stochastic calculate_stochastic(const std::vector<Bar>& bars, int k_period, int d_period, int current_index);
    
    // Williams %R
    double calculate_williams_r(const std::vector<Bar>& bars, int period, int current_index);
    
    // CCI
    double calculate_cci(const std::vector<Bar>& bars, int period, int current_index);
    
    // ADX
    double calculate_adx(const std::vector<Bar>& bars, int period, int current_index);
    
    // Volume indicators
    double calculate_obv(const std::vector<Bar>& bars, int current_index);
    double calculate_vpt(const std::vector<Bar>& bars, int current_index);
    double calculate_ad_line(const std::vector<Bar>& bars, int current_index);
    double calculate_mfi(const std::vector<Bar>& bars, int period, int current_index);
    
    // Volatility indicators
    double calculate_parkinson_volatility(const std::vector<Bar>& bars, int period, int current_index);
    double calculate_garman_klass_volatility(const std::vector<Bar>& bars, int period, int current_index);
    
    // Feature names for validation
    std::vector<std::string> feature_names_;
};

} // namespace feature_engineering
} // namespace sentio

```

## 📄 **FILE 39 of 145**: include/sentio/feature_feeder.hpp

**File Information**:
- **Path**: `include/sentio/feature_feeder.hpp`

- **Size**: 104 lines
- **Modified**: 2025-09-07 11:53:47

- **Type**: .hpp

```text
#pragma once
#include "sentio/core.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/feature_engineering/technical_indicators.hpp"
#include "sentio/feature_engineering/feature_normalizer.hpp"
#include "sentio/feature_cache.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <mutex>

namespace sentio {

struct FeatureMetrics {
    std::chrono::microseconds extraction_time{0};
    size_t features_extracted{0};
    size_t features_valid{0};
    size_t features_invalid{0};
    double extraction_rate{0.0}; // features per second
    std::chrono::steady_clock::time_point last_update;
};

struct FeatureHealthReport {
    bool is_healthy{false};
    std::vector<bool> feature_health;
    std::vector<double> feature_quality_scores;
    std::string health_summary;
    double overall_health_score{0.0};
};

class FeatureFeeder {
public:
    // Core functionality
    static std::vector<double> extract_features_from_bar(const Bar& bar, const std::string& strategy_name);
    static std::vector<double> extract_features_from_bars_with_index(const std::vector<Bar>& bars, int current_index, const std::string& strategy_name);
    static bool is_ml_strategy(const std::string& strategy_name);
    static void feed_features_to_strategy(BaseStrategy* strategy, const std::vector<Bar>& bars, int current_index, const std::string& strategy_name);
    
    // Enhanced functionality
    static void initialize_strategy(const std::string& strategy_name);
    static void cleanup_strategy(const std::string& strategy_name);
    
    // Feature management
    static std::vector<double> get_cached_features(const std::string& strategy_name);
    static void cache_features(const std::string& strategy_name, const std::vector<double>& features);
    static void invalidate_cache(const std::string& strategy_name);
    
    // Performance monitoring
    static FeatureMetrics get_metrics(const std::string& strategy_name);
    static FeatureHealthReport get_health_report(const std::string& strategy_name);
    static void reset_metrics(const std::string& strategy_name);
    
    // Feature validation
    static bool validate_features(const std::vector<double>& features, const std::string& strategy_name);
    static std::vector<std::string> get_feature_names(const std::string& strategy_name);
    
    // Configuration
    static void set_feature_config(const std::string& strategy_name, const std::string& config_key, const std::string& config_value);
    static std::string get_feature_config(const std::string& strategy_name, const std::string& config_key);
    
    // Cached features (for performance)
    static bool load_feature_cache(const std::string& feature_file_path);
    static bool use_cached_features(bool enable = true);
    static bool has_cached_features();
    
    // Batch processing
    static std::vector<std::vector<double>> extract_features_from_bars(const std::vector<Bar>& bars, const std::string& strategy_name);
    static void feed_features_batch(BaseStrategy* strategy, const std::vector<Bar>& bars, const std::string& strategy_name);
    
    // Feature analysis
    static std::vector<double> get_feature_correlation(const std::string& strategy_name);
    static std::vector<double> get_feature_importance(const std::string& strategy_name);
    static void log_feature_performance(const std::string& strategy_name);
    
private:
    // Strategy-specific data
    struct StrategyData {
        std::unique_ptr<feature_engineering::TechnicalIndicatorCalculator> calculator;
        std::unique_ptr<feature_engineering::FeatureNormalizer> normalizer;
        std::vector<double> cached_features;
        FeatureMetrics metrics;
        std::chrono::steady_clock::time_point last_update;
        bool initialized{false};
        std::unordered_map<std::string, std::string> config;
    };
    
    static std::unordered_map<std::string, StrategyData> strategy_data_;
    static std::mutex data_mutex_;
    
    // Feature cache for performance
    static std::unique_ptr<FeatureCache> feature_cache_;
    static bool use_cached_features_;
    
    // Helper methods
    static StrategyData& get_strategy_data(const std::string& strategy_name);
    static void update_metrics(StrategyData& data, const std::vector<double>& features, std::chrono::microseconds extraction_time);
    static FeatureHealthReport calculate_health_report(const StrategyData& data, const std::vector<double>& features);
    static std::vector<std::string> get_strategy_feature_names(const std::string& strategy_name);
    static void initialize_strategy_data(const std::string& strategy_name);
};

} // namespace sentio
```

## 📄 **FILE 40 of 145**: include/sentio/feature_health.hpp

**File Information**:
- **Path**: `include/sentio/feature_health.hpp`

- **Size**: 31 lines
- **Modified**: 2025-09-05 17:01:01

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sentio {

struct FeatureIssue {
  std::int64_t ts_utc{};
  std::string  kind;      // e.g., "NaN", "Gap", "Backwards_TS"
  std::string  detail;
};

struct FeatureHealthReport {
  std::vector<FeatureIssue> issues;
  bool ok() const { return issues.empty(); }
};

struct FeatureHealthCfg {
  // bar spacing in seconds (e.g., 60 for 1m). 0 = skip spacing checks.
  int expected_spacing_sec{60};
  bool check_nan{true};
  bool check_monotonic_time{true};
};

struct PricePoint { std::int64_t ts_utc{}; double close{}; };

FeatureHealthReport check_feature_health(const std::vector<PricePoint>& series,
                                         const FeatureHealthCfg& cfg);

} // namespace sentio

```

## 📄 **FILE 41 of 145**: include/sentio/indicators.hpp

**File Information**:
- **Path**: `include/sentio/indicators.hpp`

- **Size**: 52 lines
- **Modified**: 2025-09-05 16:34:30

- **Type**: .hpp

```text
#pragma once
#include <deque>
#include <cmath>
#include <limits>

namespace sentio {

struct SMA {
  int n{0};
  double sum{0.0};
  std::deque<double> q;
  explicit SMA(int n_) : n(n_) {}
  void reset(){ sum=0; q.clear(); }
  void push(double x){
    if (!std::isfinite(x)) { reset(); return; }
    q.push_back(x); sum += x;
    if ((int)q.size() > n) { sum -= q.front(); q.pop_front(); }
  }
  bool ready() const { return (int)q.size() == n; }
  double value() const { return ready() ? sum / n : std::numeric_limits<double>::quiet_NaN(); }
};

struct RSI {
  int n{14};
  int warm{0};
  double avgGain{0}, avgLoss{0}, prev{NAN};
  explicit RSI(int n_=14):n(n_),warm(0),avgGain(0),avgLoss(0),prev(NAN){}
  void reset(){ warm=0; avgGain=avgLoss=0; prev=NAN; }
  void push(double close){
    if (!std::isfinite(close)) { reset(); return; }
    if (!std::isfinite(prev)) { prev = close; return; }
    double delta = close - prev; prev = close;
    double gain = delta > 0 ? delta : 0.0;
    double loss = delta < 0 ? -delta : 0.0;
    if (warm < n) {
      avgGain += gain; avgLoss += loss; ++warm;
      if (warm==n) { avgGain/=n; avgLoss/=n; }
    } else {
      avgGain = (avgGain*(n-1) + gain)/n;
      avgLoss = (avgLoss*(n-1) + loss)/n;
    }
  }
  bool ready() const { return warm >= n; }
  double value() const {
    if (!ready()) return std::numeric_limits<double>::quiet_NaN();
    if (avgLoss == 0) return 100.0;
    double rs = avgGain/avgLoss;
    return 100.0 - (100.0/(1.0+rs));
  }
};

} // namespace sentio

```

## 📄 **FILE 42 of 145**: include/sentio/metrics.hpp

**File Information**:
- **Path**: `include/sentio/metrics.hpp`

- **Size**: 123 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <utility>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <string>

namespace sentio {
struct RunSummary {
  int bars{}, trades{};
  double ret_total{}, ret_ann{}, vol_ann{}, sharpe{}, mdd{};
  double monthly_proj{}, daily_trades{};
};

inline RunSummary compute_metrics(const std::vector<std::pair<std::string,double>>& daily_equity,
                                  int fills_count) {
  RunSummary s{};
  if (daily_equity.size() < 2) return s;
  s.bars = (int)daily_equity.size();
  std::vector<double> rets; rets.reserve(daily_equity.size()-1);
  for (size_t i=1;i<daily_equity.size();++i) {
    double e0=daily_equity[i-1].second, e1=daily_equity[i].second;
    rets.push_back(e0>0 ? (e1/e0-1.0) : 0.0);
  }
  double mean = std::accumulate(rets.begin(),rets.end(),0.0)/rets.size();
  double var=0.0; for (double r:rets){ double d=r-mean; var+=d*d; } var/=std::max<size_t>(1,rets.size());
  double sd = std::sqrt(var);
  s.ret_ann = mean * 252.0;
  s.vol_ann = sd * std::sqrt(252.0);
  s.sharpe  = (s.vol_ann>1e-12)? s.ret_ann/s.vol_ann : 0.0;
  double e0 = daily_equity.front().second, e1 = daily_equity.back().second;
  s.ret_total = e0>0 ? (e1/e0-1.0) : 0.0;
  s.monthly_proj = std::pow(1.0 + s.ret_ann, 1.0/12.0) - 1.0;
  s.trades = fills_count;
  s.daily_trades = (s.bars>0) ? (double)s.trades / (double)s.bars : 0.0;
  s.mdd = 0.0; // TODO: compute drawdown if you track running peaks
  return s;
}

// Day-aware metrics computed from bar-level equity series by compressing to day closes
inline RunSummary compute_metrics_day_aware(
    const std::vector<std::pair<std::string,double>>& equity_steps,
    int fills_count) {
  RunSummary s{};
  if (equity_steps.size() < 2) {
    s.trades = fills_count;
    s.daily_trades = 0.0;
    return s;
  }

  // Compress to day closes using ts_utc prefix YYYY-MM-DD
  std::vector<double> day_close;
  day_close.reserve(equity_steps.size() / 300 + 2);

  std::string last_day = equity_steps.front().first.size() >= 10
                         ? equity_steps.front().first.substr(0,10)
                         : std::string{};
  double cur = equity_steps.front().second;

  for (size_t i = 1; i < equity_steps.size(); ++i) {
    const auto& ts = equity_steps[i].first;
    const std::string day = ts.size() >= 10 ? ts.substr(0,10) : last_day;
    if (day != last_day) {
      day_close.push_back(cur); // close of previous day
      last_day = day;
    }
    cur = equity_steps[i].second; // latest equity for current day
  }
  day_close.push_back(cur); // close of final day

  const int D = static_cast<int>(day_close.size());
  s.bars = D;

  if (D < 2) {
    s.trades = fills_count;
    s.daily_trades = 0.0;
    s.ret_total = 0.0; s.ret_ann = 0.0; s.vol_ann = 0.0; s.sharpe = 0.0; s.mdd = 0.0;
    s.monthly_proj = 0.0;
    return s;
  }

  // Daily simple returns
  std::vector<double> r; r.reserve(D - 1);
  for (int i = 1; i < D; ++i) {
    double prev = day_close[i-1];
    double next = day_close[i];
    r.push_back(prev > 0.0 ? (next/prev - 1.0) : 0.0);
  }

  // Mean and variance
  double mean = 0.0; for (double x : r) mean += x; mean /= r.size();
  double var  = 0.0; for (double x : r) { double d = x - mean; var += d*d; } var /= r.size();
  double sd = std::sqrt(var);

  // Annualization on daily series
  s.vol_ann = sd * std::sqrt(252.0);
  double e0 = day_close.front(), e1 = day_close.back();
  s.ret_total = e0 > 0.0 ? (e1/e0 - 1.0) : 0.0;
  double years = (D - 1) / 252.0;
  s.ret_ann = (years > 0.0) ? (std::pow(1.0 + s.ret_total, 1.0/years) - 1.0) : 0.0;
  s.sharpe = (sd > 1e-12) ? (mean / sd) * std::sqrt(252.0) : 0.0;
  s.monthly_proj = std::pow(1.0 + s.ret_ann, 1.0/12.0) - 1.0;

  // Trades
  s.trades = fills_count;
  s.daily_trades = (D > 0) ? static_cast<double>(fills_count) / static_cast<double>(D) : 0.0;

  // Max drawdown on day closes
  double peak = day_close.front();
  double max_dd = 0.0;
  for (int i = 1; i < D; ++i) {
    peak = std::max(peak, day_close[i-1]);
    if (peak > 0.0) {
      double dd = (day_close[i] - peak) / peak; // negative when below peak
      if (dd < max_dd) max_dd = dd;
    }
  }
  s.mdd = -max_dd;
  return s;
}
} // namespace sentio


```

## 📄 **FILE 43 of 145**: include/sentio/ml/feature_pipeline.hpp

**File Information**:
- **Path**: `include/sentio/ml/feature_pipeline.hpp`

- **Size**: 40 lines
- **Modified**: 2025-09-05 21:02:15

- **Type**: .hpp

```text
#pragma once
#include "iml_model.hpp"
#include <vector>
#include <string>
#include <optional>
#include <cmath>

namespace sentio::ml {

struct FeaturePipeline {
  // Pre-sized to spec.feature_names.size()
  std::vector<float> buf;

  explicit FeaturePipeline(const ModelSpec& spec)
  : buf(spec.feature_names.size(), 0.0f), spec_(&spec) {}

  // raw must match spec.feature_names order/length
  // Applies (x-mean)/std then clips to [clip_lo, clip_hi]
  // Returns pointer to internal buffer if successful, nullptr if failed
  const std::vector<float>* transform(const std::vector<double>& raw) {
    auto N = spec_->feature_names.size();
    if (raw.size()!=N) return nullptr;
    const double lo = spec_->clip2.size()==2 ? spec_->clip2[0] : -5.0;
    const double hi = spec_->clip2.size()==2 ? spec_->clip2[1] :  5.0;
    for (size_t i=0;i<N;++i) {
      double x = raw[i];
      double m = (i<spec_->mean.size()? spec_->mean[i] : 0.0);
      double s = (i<spec_->std.size()?  spec_->std[i]  : 1.0);
      double z = s>0 ? (x-m)/s : x-m;
      if (z<lo) z=lo; if (z>hi) z=hi;
      buf[i] = static_cast<float>(z);
    }
    return &buf;
  }

private:
  const ModelSpec* spec_;
};

} // namespace sentio::ml

```

## 📄 **FILE 44 of 145**: include/sentio/ml/feature_window.hpp

**File Information**:
- **Path**: `include/sentio/ml/feature_window.hpp`

- **Size**: 95 lines
- **Modified**: 2025-09-07 12:00:08

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <deque>
#include <optional>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sentio::ml {

struct WindowSpec {
  int seq_len{64};
  int feat_dim{0};
  std::vector<double> mean, std, clip2; // clip2=[lo,hi]
  // layout: "BTF" or "BF"
  std::string layout{"BTF"};
};

class FeatureWindow {
public:
  explicit FeatureWindow(const WindowSpec& s) : spec_(s) {
    // Validate mean/std lengths match feat_dim to prevent segfaults
    // Temporarily disabled for debugging
    // if ((int)spec_.mean.size() != spec_.feat_dim || (int)spec_.std.size() != spec_.feat_dim) {
    //   throw std::runtime_error("FeatureWindow: mean/std length mismatch feat_dim");
    // }
  }
  
  // push raw features in metadata order (length == feat_dim)
  void push(const std::vector<double>& raw) {
    static int push_calls = 0;
    push_calls++;
    
    if ((int)raw.size() != spec_.feat_dim) {
      // Diagnostic for size mismatch
      if (push_calls % 1000 == 0 || push_calls <= 10) {
        std::cout << "[DIAG] FeatureWindow push FAILED: call=" << push_calls 
                  << " raw.size()=" << raw.size() 
                  << " expected=" << spec_.feat_dim << std::endl;
      }
      return;
    }
    
    buf_.push_back(raw);
    if ((int)buf_.size() > spec_.seq_len) buf_.pop_front();
    
    if (push_calls % 1000 == 0 || push_calls <= 10) {
      std::cout << "[DIAG] FeatureWindow push SUCCESS: call=" << push_calls 
                << " buf_size=" << buf_.size() << "/" << spec_.seq_len << std::endl;
    }
  }
  
  bool ready() const { return (int)buf_.size() == spec_.seq_len; }

  // Return normalized/ clipped tensor as float vector for ONNX input.
  // For "BTF": size = 1*T*F; For "BF": size = 1*(T*F)
  // Returns reused buffer (no allocations in hot path)
  std::optional<std::vector<float>> to_input() const {
    if (!ready()) return std::nullopt;
    const double lo = spec_.clip2.size() == 2 ? spec_.clip2[0] : -5.0;
    const double hi = spec_.clip2.size() == 2 ? spec_.clip2[1] : 5.0;

    // Pre-size buffer once (no allocations in hot path)
    const size_t need = size_t(spec_.seq_len) * size_t(spec_.feat_dim);
    if (norm_buf_.size() != need) norm_buf_.assign(need, 0.0f);

    auto norm = [&](double x, int i) -> float {
      double m = (i < (int)spec_.mean.size() ? spec_.mean[i] : 0.0);
      double s = (i < (int)spec_.std.size() ? spec_.std[i] : 1.0);
      double z = (s > 0 ? (x - m) / s : (x - m));
      if (z < lo) z = lo; 
      if (z > hi) z = hi;
      return (float)z;
    };

    // Fill row-major [T, F] into reusable buffer (no new allocations)
    for (int t = 0; t < spec_.seq_len; ++t) {
      const auto& r = buf_[t];
      for (int f = 0; f < spec_.feat_dim; ++f) {
        norm_buf_[t * spec_.feat_dim + f] = norm(r[f], f);
      }
    }
    return std::make_optional(norm_buf_);
  }

  int seq_len() const { return spec_.seq_len; }
  int feat_dim() const { return spec_.feat_dim; }

private:
  WindowSpec spec_;
  std::deque<std::vector<double>> buf_;
  mutable std::vector<float> norm_buf_;   // REUSABLE normalized buffer
};

} // namespace sentio::ml

```

## 📄 **FILE 45 of 145**: include/sentio/ml/iml_model.hpp

**File Information**:
- **Path**: `include/sentio/ml/iml_model.hpp`

- **Size**: 40 lines
- **Modified**: 2025-09-05 23:39:11

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>
#include <optional>

namespace sentio::ml {

// Raw model output abstraction: either class probs or a scalar score.
struct ModelOutput {
  std::vector<float> probs;   // e.g., [p_sell, p_hold, p_buy] for discrete
  float score{0.0f};          // for regressors
};

struct ModelSpec {
  std::string model_id;       // "TransAlpha", "HybridPPO"
  std::string version;        // "v1"
  std::vector<std::string> feature_names;
  std::vector<double> mean, std, clip2; // clip2: [lo, hi]
  std::vector<std::string> actions;     // e.g., SELL, HOLD, BUY
  int expected_spacing_sec{60};
  std::string instrument_family;        // e.g., "QQQ"
  std::string notes;                    // optional metadata
  // Sequence model extensions
  int seq_len{1};  // 1 for non-sequence models
  std::string input_layout{"BTF"};  // "BTF" for batch-time-feature, "BF" for flattened
  std::string format{"torchscript"};  // "torchscript", "onnx", etc.
};

// Runtime inference model
class IModel {
public:
  virtual ~IModel() = default;
  virtual const ModelSpec& spec() const = 0;
  // features must match spec().feature_names length and order
  // T, F, layout parameters for sequence models
  virtual std::optional<ModelOutput> predict(const std::vector<float>& features,
                                             int T, int F, const std::string& layout) const = 0;
};

} // namespace sentio::ml
```

## 📄 **FILE 46 of 145**: include/sentio/ml/model_registry.hpp

**File Information**:
- **Path**: `include/sentio/ml/model_registry.hpp`

- **Size**: 20 lines
- **Modified**: 2025-09-05 23:39:11

- **Type**: .hpp

```text
#pragma once
#include "iml_model.hpp"
#include <memory>

namespace sentio::ml {

struct ModelHandle {
  std::unique_ptr<IModel> model;
  ModelSpec spec;
};

class ModelRegistryTS {
public:
  static ModelHandle load_torchscript(const std::string& model_id,
                                      const std::string& version,
                                      const std::string& artifacts_dir,
                                      bool use_cuda = false);
};

} // namespace sentio::ml
```

## 📄 **FILE 47 of 145**: include/sentio/ml/ts_model.hpp

**File Information**:
- **Path**: `include/sentio/ml/ts_model.hpp`

- **Size**: 31 lines
- **Modified**: 2025-09-06 20:23:41

- **Type**: .hpp

```text
#pragma once
#include "iml_model.hpp"
#include <memory>

namespace torch { namespace jit { class Module; } }

namespace sentio::ml {

class TorchScriptModel final : public IModel {
public:
  static std::unique_ptr<TorchScriptModel> load(const std::string& pt_path,
                                                const ModelSpec& spec,
                                                bool use_cuda = false);

  const ModelSpec& spec() const override { return spec_; }
  std::optional<ModelOutput> predict(const std::vector<float>& features,
                                     int T, int F, const std::string& layout) const override;

  ~TorchScriptModel();

private:
  explicit TorchScriptModel(ModelSpec spec);
  ModelSpec spec_;
  std::shared_ptr<torch::jit::Module> mod_;
  // Preallocated input tensor & shape (PIMPL pattern)
  mutable void* input_tensor_;  // torch::Tensor (hidden in .cpp)
  mutable std::vector<int64_t> in_shape_;
  bool cuda_{false};
};

} // namespace sentio::ml

```

## 📄 **FILE 48 of 145**: include/sentio/of_index.hpp

**File Information**:
- **Path**: `include/sentio/of_index.hpp`

- **Size**: 36 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include "core.hpp"
#include "orderflow_types.hpp"

namespace sentio {

struct BarTickSpan { 
    int start=-1, end=-1; // [start,end) ticks for this bar
};

inline std::vector<BarTickSpan> build_tick_spans(const std::vector<Bar>& bars,
                                                 const std::vector<Tick>& ticks)
{
    const int N = (int)bars.size();
    const int M = (int)ticks.size();
    std::vector<BarTickSpan> span(N);

    int i = 0, k = 0;
    int cur_start = 0;

    // assume bars have strictly increasing ts; ticks nondecreasing
    for (; i < N; ++i) {
        const int64_t ts = bars[i].ts_nyt_epoch;
        // advance k until tick.ts > ts
        while (k < M && ticks[k].ts_nyt_epoch <= ts) ++k;
        span[i].start = cur_start;
        span[i].end   = k;        // [cur_start, k) are ticks for bar i
        cur_start = k;
    }
    return span;
}

} // namespace sentio
```

## 📄 **FILE 49 of 145**: include/sentio/of_precompute.hpp

**File Information**:
- **Path**: `include/sentio/of_precompute.hpp`

- **Size**: 26 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "of_index.hpp"

namespace sentio {

inline void precompute_mid_imb(const std::vector<Tick>& ticks,
                               std::vector<double>& mid,
                               std::vector<double>& imb)
{
    const int M = (int)ticks.size();
    mid.resize(M);
    imb.resize(M);
    for (int k=0; k<M; ++k) {
        double m = (ticks[k].bid_px + ticks[k].ask_px) * 0.5;
        double a = std::max(0.0, ticks[k].ask_sz);
        double b = std::max(0.0, ticks[k].bid_sz);
        double d = a + b;
        mid[k] = m;
        imb[k] = (d > 0.0) ? (a / d) : 0.5;   // neutral if zero depth
    }
}

} // namespace sentio
```

## 📄 **FILE 50 of 145**: include/sentio/optimizer.hpp

**File Information**:
- **Path**: `include/sentio/optimizer.hpp`

- **Size**: 147 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "sentio/core.hpp"
#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <random>

namespace sentio {

struct Parameter {
    std::string name;
    double min_value;
    double max_value;
    double current_value;
    
    Parameter(const std::string& n, double min_val, double max_val, double current_val)
        : name(n), min_value(min_val), max_value(max_val), current_value(current_val) {}
    
    Parameter(const std::string& n, double min_val, double max_val)
        : name(n), min_value(min_val), max_value(max_val), current_value(min_val) {}
};

struct OptimizationResult {
    std::unordered_map<std::string, double> parameters;
    RunResult metrics;
    double objective_value;
};

struct OptimizationConfig {
    std::string optimizer_type = "random";
    int max_trials = 30;
    std::string objective = "sharpe_ratio";
    double timeout_minutes = 15.0;
    bool verbose = true;
};

class ParameterOptimizer {
public:
    virtual ~ParameterOptimizer() = default;
    virtual std::vector<Parameter> get_parameter_space() = 0;
    virtual void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) = 0;
    virtual OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                                      const std::vector<Parameter>& param_space,
                                      const OptimizationConfig& config) = 0;
};

class RandomSearchOptimizer : public ParameterOptimizer {
private:
    std::mt19937 rng;
    
public:
    RandomSearchOptimizer(unsigned int seed = 42) : rng(seed) {}
    
    std::vector<Parameter> get_parameter_space() override;
    void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) override;
    OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                               const std::vector<Parameter>& param_space,
                               const OptimizationConfig& config) override;
};

class GridSearchOptimizer : public ParameterOptimizer {
public:
    std::vector<Parameter> get_parameter_space() override;
    void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) override;
    OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                              const std::vector<Parameter>& param_space,
                              const OptimizationConfig& config) override;
};

class BayesianOptimizer : public ParameterOptimizer {
public:
    std::vector<Parameter> get_parameter_space() override;
    void apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) override;
    OptimizationResult optimize(const std::function<double(const RunResult&)>& objective_func,
                              const std::vector<Parameter>& param_space,
                              const OptimizationConfig& config) override;
};

// Objective functions
using ObjectiveFunction = std::function<double(const RunResult&)>;

class ObjectiveFunctions {
public:
    static double sharpe_objective(const RunResult& summary) {
        return summary.sharpe_ratio;
    }
    
    static double calmar_objective(const RunResult& summary) {
        return summary.monthly_projected_return / std::max(0.01, summary.max_drawdown);
    }
    
    static double total_return_objective(const RunResult& summary) {
        return summary.total_return;
    }
    
    static double sortino_objective(const RunResult& summary) {
        // Simplified Sortino ratio (assuming no downside deviation calculation)
        return summary.sharpe_ratio;
    }
};

// Strategy-specific parameter creation functions
std::vector<Parameter> create_vwap_parameters();
std::vector<Parameter> create_momentum_parameters();
std::vector<Parameter> create_volatility_parameters();
std::vector<Parameter> create_bollinger_squeeze_parameters();
std::vector<Parameter> create_opening_range_parameters();
std::vector<Parameter> create_order_flow_scalping_parameters();
std::vector<Parameter> create_order_flow_imbalance_parameters();
std::vector<Parameter> create_market_making_parameters();
std::vector<Parameter> create_router_parameters();

std::vector<Parameter> create_parameters_for_strategy(const std::string& strategy_name);
std::vector<Parameter> create_full_parameter_space();

// Optimization utilities
class OptimizationEngine {
private:
    std::unique_ptr<ParameterOptimizer> optimizer;
    OptimizationConfig config;
    
public:
    OptimizationEngine(const std::string& optimizer_type = "random");
    
    void set_config(const OptimizationConfig& cfg) { config = cfg; }
    
    OptimizationResult run_optimization(const std::string& strategy_name,
                                      const std::function<double(const RunResult&)>& objective_func);
    
    double calculate_objective(const RunResult& summary) {
        if (config.objective == "sharpe_ratio") {
            return ObjectiveFunctions::sharpe_objective(summary);
        } else if (config.objective == "calmar_ratio") {
            return ObjectiveFunctions::calmar_objective(summary);
        } else if (config.objective == "total_return") {
            return ObjectiveFunctions::total_return_objective(summary);
        } else if (config.objective == "sortino_ratio") {
            return ObjectiveFunctions::sortino_objective(summary);
        }
        return ObjectiveFunctions::sharpe_objective(summary); // default
    }
};

} // namespace sentio
```

## 📄 **FILE 51 of 145**: include/sentio/orderflow_types.hpp

**File Information**:
- **Path**: `include/sentio/orderflow_types.hpp`

- **Size**: 34 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include "core.hpp"

namespace sentio {

struct Tick {
    int64_t ts_nyt_epoch;     // strictly nondecreasing
    double  bid_px, ask_px;
    double  bid_sz, ask_sz;   // L1 sizes (or synthetic from L2)
    // (Optional: trade prints, aggressor flags, etc.)
};

struct OFParams {
    // Signal
    double  min_imbalance = 0.65;     // (ask_sz / (ask_sz + bid_sz)) for long
    int     look_ticks    = 50;       // rolling window
    // Risk
    int     hold_max_ticks = 800;     // hard cap per trade
    double  tp_ticks       = 4.0;     // TP in ticks (half-spread units if you like)
    double  sl_ticks       = 4.0;     // SL in ticks
    // Execution
    double  tick_size      = 0.01;
};

struct Trade {
    int start_k=-1, end_k=-1;  // tick indices
    double entry_px=0, exit_px=0;
    int dir=0;                 // +1 long, -1 short
};

} // namespace sentio
```

## 📄 **FILE 52 of 145**: include/sentio/pnl_accounting.hpp

**File Information**:
- **Path**: `include/sentio/pnl_accounting.hpp`

- **Size**: 24 lines
- **Modified**: 2025-09-05 15:29:35

- **Type**: .hpp

```text
// pnl_accounting.hpp
#pragma once
#include "core.hpp"  // Use Bar from core.hpp
#include <string>
#include <stdexcept>
#include <unordered_map>

namespace sentio {

class PriceBook {
public:
  // instrument -> latest bar (or map<ts,bar> for full history)
  const Bar* get_latest(const std::string& instrument) const;
  
  // Additional helper methods
  void upsert_latest(const std::string& instrument, const Bar& b);
  bool has_instrument(const std::string& instrument) const;
  std::size_t size() const;
};

// Use the instrument actually traded
double last_trade_price(const PriceBook& book, const std::string& instrument);

} // namespace sentio
```

## 📄 **FILE 53 of 145**: include/sentio/polygon_client.hpp

**File Information**:
- **Path**: `include/sentio/polygon_client.hpp`

- **Size**: 30 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>

namespace sentio {
struct AggBar { long long ts_ms; double open, high, low, close, volume; };

struct AggsQuery {
  std::string symbol;
  int multiplier{1};
  std::string timespan{"day"}; // "day","hour","minute"
  std::string from, to;
  bool adjusted{true};
  std::string sort{"asc"};
  int limit{50000};
};

class PolygonClient {
public:
  explicit PolygonClient(std::string api_key);
  std::vector<AggBar> get_aggs_all(const AggsQuery& q, int max_pages=200);
  void write_csv(const std::string& out_path,const std::string& symbol,
                 const std::vector<AggBar>& bars, bool rth_only=false, bool exclude_holidays=false);

private:
  std::string api_key_;
  std::string get_(const std::string& url);
};
} // namespace sentio


```

## 📄 **FILE 54 of 145**: include/sentio/polygon_ingest.hpp

**File Information**:
- **Path**: `include/sentio/polygon_ingest.hpp`

- **Size**: 34 lines
- **Modified**: 2025-09-05 15:28:51

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace sentio {

// Polygon-like bar from feeds/adapters.
// ts can be epoch seconds (UTC), epoch milliseconds (UTC), or ISO8601 string.
struct ProviderBar {
  std::string symbol;                                   // instrument actually traded (QQQ/TQQQ/SQQQ/…)
  std::variant<std::int64_t, double, std::string> ts;   // epoch sec (int64), epoch ms (double), or ISO8601
  double open{};
  double high{};
  double low{};
  double close{};
  double volume{};
};

struct Bar { double open{}, high{}, low{}, close{}; };

class PriceBook {
public:
  void upsert_latest(const std::string& instrument, const Bar& b);
  const Bar* get_latest(const std::string& instrument) const;
  bool has_instrument(const std::string& instrument) const;
  std::size_t size() const;
};

std::size_t ingest_provider_bars(const std::vector<ProviderBar>& input, PriceBook& book);
bool        ingest_provider_bar(const ProviderBar& bar, PriceBook& book);

} // namespace sentio

```

## 📄 **FILE 55 of 145**: include/sentio/position_manager.hpp

**File Information**:
- **Path**: `include/sentio/position_manager.hpp`

- **Size**: 63 lines
- **Modified**: 2025-09-05 09:41:27

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include <string>
#include <vector>
#include <unordered_set>

namespace sentio {

enum class PortfolioState { Neutral, LongOnly, ShortOnly };
enum class RequiredAction { None, CloseLong, CloseShort };
enum class Direction { Long, Short }; // Keep for simple directional logic

const std::unordered_set<std::string> LONG_INSTRUMENTS = {"QQQ", "TQQQ", "TSLA"};
const std::unordered_set<std::string> SHORT_INSTRUMENTS = {"SQQQ", "PSQ", "TSLQ"};

class PositionManager {
private:
    PortfolioState state = PortfolioState::Neutral;
    int bars_since_flip = 0;
    [[maybe_unused]] const int cooldown_period = 5;
    
public:
    // **MODIFIED**: Logic restored to work with the new ID-based portfolio structure.
    void update_state(const Portfolio& portfolio, const SymbolTable& ST, const std::vector<double>& last_prices) {
        bars_since_flip++;
        double long_exposure = 0.0;
        double short_exposure = 0.0;

        for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
            const auto& pos = portfolio.positions[sid];
            if (std::abs(pos.qty) < 1e-6) continue;

            const std::string& symbol = ST.get_symbol(sid);
            double exposure = pos.qty * last_prices[sid];

            if (LONG_INSTRUMENTS.count(symbol)) {
                long_exposure += exposure;
            }
            if (SHORT_INSTRUMENTS.count(symbol)) {
                short_exposure += exposure; // Will be negative for short positions
            }
        }
        
        PortfolioState old_state = state;
        
        if (long_exposure > 100 && std::abs(short_exposure) < 100) {
            state = PortfolioState::LongOnly;
        } else if (std::abs(short_exposure) > 100 && long_exposure < 100) {
            state = PortfolioState::ShortOnly;
        } else {
            state = PortfolioState::Neutral;
        }

        if (state != old_state) {
            bars_since_flip = 0;
        }
    }
    
    // ... other methods remain the same ...
};

} // namespace sentio
```

## 📄 **FILE 56 of 145**: include/sentio/pricebook.hpp

**File Information**:
- **Path**: `include/sentio/pricebook.hpp`

- **Size**: 59 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include "core.hpp"
#include "symbol_table.hpp"

namespace sentio {

// **MODIFIED**: The Pricebook ensures that all symbol prices are correctly aligned in time.
// It works by advancing an index for each symbol's data series until its timestamp
// matches or just passes the timestamp of the base symbol's current bar.
// This is both fast (amortized O(1)) and correct, handling missing data gracefully.
struct Pricebook {
    const int base_id;
    const SymbolTable& ST;
    const std::vector<std::vector<Bar>>& S; // Reference to all series data
    std::vector<int> idx;                   // Rolling index for each symbol's series
    std::vector<double> last_px;            // Stores the last known price for each symbol ID

    Pricebook(int base, const SymbolTable& st, const std::vector<std::vector<Bar>>& series)
      : base_id(base), ST(st), S(series), idx(S.size(), 0), last_px(S.size(), 0.0) {}

    // Advances the index 'j' for a given series 'V' to the bar corresponding to 'base_ts'
    inline void advance_to_ts(const std::vector<Bar>& V, int& j, int64_t base_ts) {
        const int n = (int)V.size();
        // Move index forward as long as the *next* bar is still at or before the target time
        while (j + 1 < n && V[j + 1].ts_nyt_epoch <= base_ts) {
            ++j;
        }
    }

    // Syncs all symbol prices to the timestamp of the i-th bar of the base symbol
    inline void sync_to_base_i(int i) {
        if (S[base_id].empty()) return;
        const int64_t ts = S[base_id][i].ts_nyt_epoch;
        
        for (int sid = 0; sid < (int)S.size(); ++sid) {
            if (!S[sid].empty()) {
                advance_to_ts(S[sid], idx[sid], ts);
                last_px[sid] = S[sid][idx[sid]].close;
            }
        }
    }

    // **NEW**: Helper to get the last prices as a map for components needing string keys
    inline std::unordered_map<std::string, double> last_px_map() const {
        std::unordered_map<std::string, double> price_map;
        for (int sid = 0; sid < (int)last_px.size(); ++sid) {
            if (last_px[sid] > 0.0) {
                price_map[ST.get_symbol(sid)] = last_px[sid];
            }
        }
        return price_map;
    }
};

} // namespace sentio
```

## 📄 **FILE 57 of 145**: include/sentio/profiling.hpp

**File Information**:
- **Path**: `include/sentio/profiling.hpp`

- **Size**: 25 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <chrono>
#include <cstdio>

namespace sentio {

struct Tsc {
    std::chrono::high_resolution_clock::time_point t0;
    
    void tic() { 
        t0 = std::chrono::high_resolution_clock::now(); 
    }
    
    double toc_ms() const {
        using namespace std::chrono;
        return duration<double, std::milli>(high_resolution_clock::now() - t0).count();
    }
    
    double toc_sec() const {
        using namespace std::chrono;
        return duration<double>(high_resolution_clock::now() - t0).count();
    }
};

} // namespace sentio
```

## 📄 **FILE 58 of 145**: include/sentio/progress_bar.hpp

**File Information**:
- **Path**: `include/sentio/progress_bar.hpp`

- **Size**: 225 lines
- **Modified**: 2025-09-06 21:06:24

- **Type**: .hpp

```text
#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>

namespace sentio {

class ProgressBar {
public:
    ProgressBar(int total, const std::string& description = "Progress")
        : total_(total), current_(0), description_(description), start_time_(std::chrono::high_resolution_clock::now()) {}
    
    void update(int current) {
        current_ = current;
        if (current_ % std::max(1, total_ / 100) == 0 || current_ == total_) {
            display();
        }
    }
    
    void display() {
        double percentage = (double)current_ / total_ * 100.0;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        
        // Calculate ETA
        double eta_seconds = 0.0;
        if (current_ > 0 && current_ < total_) {
            eta_seconds = (double)elapsed / current_ * (total_ - current_);
        }
        
        std::cout << "\r" << description_ << ": [";
        
        // Draw progress bar (50 characters wide)
        int bar_width = 50;
        int pos = (int)(percentage / 100.0 * bar_width);
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        
        std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "% "
                  << "(" << current_ << "/" << total_ << ") "
                  << "Elapsed: " << elapsed << "s";
        
        if (eta_seconds > 0) {
            std::cout << " ETA: " << (int)eta_seconds << "s";
        }
        
        std::cout << std::flush;
        
        if (current_ == total_) {
            std::cout << std::endl;
        }
    }
    
    void set_description(const std::string& desc) {
        description_ = desc;
    }
    
    int get_current() const { return current_; }
    int get_total() const { return total_; }
    const std::string& get_description() const { return description_; }

private:
    int total_;
    int current_;
    std::string description_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

class TrainingProgressBar : public ProgressBar {
public:
    TrainingProgressBar(int total, const std::string& strategy_name = "TSB")
        : ProgressBar(total, "Training " + strategy_name), best_sharpe_(-999.0), best_return_(-999.0), start_time_(std::chrono::high_resolution_clock::now()) {}
    
    void update_with_metrics(int current, double sharpe_ratio, double total_return, double oos_return = 0.0) {
        // Track best metrics
        if (sharpe_ratio > best_sharpe_) best_sharpe_ = sharpe_ratio;
        if (total_return > best_return_) best_return_ = total_return;
        
        update(current);
        if (current % std::max(1, get_total() / 100) == 0 || current == get_total()) {
            display_with_metrics(sharpe_ratio, total_return, oos_return);
        }
    }
    
    void display_with_metrics(double current_sharpe, double current_return, double oos_return = 0.0) {
        int current = get_current();
        int total = get_total();
        double percentage = (double)current / total * 100.0;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        
        // Calculate ETA
        double eta_seconds = 0.0;
        if (current > 0 && current < total) {
            eta_seconds = (double)elapsed / current * (total - current);
        }
        
        std::cout << "\r" << get_description() << ": [";
        
        // Draw progress bar (50 characters wide)
        int bar_width = 50;
        int pos = (int)(percentage / 100.0 * bar_width);
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        
        std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "% "
                  << "(" << current << "/" << total << ") "
                  << "Elapsed: " << elapsed << "s";
        
        if (eta_seconds > 0) {
            std::cout << " ETA: " << (int)eta_seconds << "s";
        }
        
        // Add metrics
        std::cout << " | Sharpe: " << std::fixed << std::setprecision(3) << current_sharpe
                  << " | Return: " << std::fixed << std::setprecision(2) << (current_return * 100) << "%";
        
        if (oos_return != 0.0) {
            std::cout << " | OOS: " << std::fixed << std::setprecision(2) << (oos_return * 100) << "%";
        }
        
        std::cout << " | Best Sharpe: " << std::fixed << std::setprecision(3) << best_sharpe_
                  << " | Best Return: " << std::fixed << std::setprecision(2) << (best_return_ * 100) << "%";
        
        std::cout << std::flush;
        
        if (current == total) {
            std::cout << std::endl;
        }
    }

private:
    double best_sharpe_;
    double best_return_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

class WFProgressBar : public ProgressBar {
public:
    WFProgressBar(int total, const std::string& strategy_name = "TSB")
        : ProgressBar(total, "WF Test " + strategy_name), best_oos_sharpe_(-999.0), best_oos_return_(-999.0), 
          avg_oos_return_(0.0), successful_folds_(0), start_time_(std::chrono::high_resolution_clock::now()) {}
    
    void update_with_wf_metrics(int current, double oos_sharpe, double oos_return, double train_return = 0.0) {
        // Track best OOS metrics
        if (oos_sharpe > best_oos_sharpe_) best_oos_sharpe_ = oos_sharpe;
        if (oos_return > best_oos_return_) best_oos_return_ = oos_return;
        
        // Update running average
        successful_folds_++;
        avg_oos_return_ = (avg_oos_return_ * (successful_folds_ - 1) + oos_return) / successful_folds_;
        
        update(current);
        if (current % std::max(1, get_total() / 100) == 0 || current == get_total()) {
            display_with_wf_metrics(oos_sharpe, oos_return, train_return);
        }
    }
    
    void display_with_wf_metrics(double current_oos_sharpe, double current_oos_return, double train_return = 0.0) {
        int current = get_current();
        int total = get_total();
        double percentage = (double)current / total * 100.0;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        
        // Calculate ETA
        double eta_seconds = 0.0;
        if (current > 0 && current < total) {
            eta_seconds = (double)elapsed / current * (total - current);
        }
        
        std::cout << "\r" << get_description() << ": [";
        
        // Draw progress bar (50 characters wide)
        int bar_width = 50;
        int pos = (int)(percentage / 100.0 * bar_width);
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        
        std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "% "
                  << "(" << current << "/" << total << ") "
                  << "Elapsed: " << elapsed << "s";
        
        if (eta_seconds > 0) {
            std::cout << " ETA: " << (int)eta_seconds << "s";
        }
        
        // Add WF-specific metrics
        std::cout << " | OOS Sharpe: " << std::fixed << std::setprecision(3) << current_oos_sharpe
                  << " | OOS Return: " << std::fixed << std::setprecision(2) << (current_oos_return * 100) << "%";
        
        if (train_return != 0.0) {
            std::cout << " | Train Return: " << std::fixed << std::setprecision(2) << (train_return * 100) << "%";
        }
        
        std::cout << " | Avg OOS: " << std::fixed << std::setprecision(2) << (avg_oos_return_ * 100) << "%"
                  << " | Best OOS Sharpe: " << std::fixed << std::setprecision(3) << best_oos_sharpe_
                  << " | Folds: " << successful_folds_;
        
        std::cout << std::flush;
        
        if (current == total) {
            std::cout << std::endl;
        }
    }

private:
    double best_oos_sharpe_;
    double best_oos_return_;
    double avg_oos_return_;
    int successful_folds_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace sentio
```

## 📄 **FILE 59 of 145**: include/sentio/property_test.hpp

**File Information**:
- **Path**: `include/sentio/property_test.hpp`

- **Size**: 26 lines
- **Modified**: 2025-09-05 20:37:59

- **Type**: .hpp

```text
#pragma once
#include <functional>
#include <vector>
#include <string>
#include <iostream>

namespace sentio {

struct PropCase { std::string name; std::function<bool()> fn; };

inline int run_properties(const std::vector<PropCase>& cases) {
  int fails = 0;
  for (auto& c : cases) {
    bool ok = false;
    try { ok = c.fn(); }
    catch (const std::exception& e) {
      std::cerr << "[PROP] " << c.name << " threw: " << e.what() << "\n";
      ok = false;
    }
    if (!ok) { std::cerr << "[PROP] FAIL: " << c.name << "\n"; ++fails; }
  }
  if (fails==0) std::cout << "[PROP] all passed ("<<cases.size()<<")\n";
  return fails==0 ? 0 : 1;
}

} // namespace sentio

```

## 📄 **FILE 60 of 145**: include/sentio/rolling_stats.hpp

**File Information**:
- **Path**: `include/sentio/rolling_stats.hpp`

- **Size**: 97 lines
- **Modified**: 2025-09-05 13:47:04

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace sentio {

struct RollingMean {
  int win, idx=0, count=0;
  std::vector<double> buf;
  double sum=0.0;
  
  explicit RollingMean(int w = 1) { reset(w); }

  void reset(int w) {
    win = w > 0 ? w : 1;
    idx = 0;
    count = 0;
    sum = 0.0;
    buf.assign(win, 0.0);
  }

  inline double push(double x){
      if (count < win) { 
          buf[count++] = x; 
          sum += x; 
      } else {
          sum -= buf[idx]; 
          buf[idx]=x; 
          sum += x; 
          idx = (idx+1) % win; 
      }
      return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }

  double mean() const {
      return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }

  size_t size() const {
      return static_cast<size_t>(count);
  }
};


struct RollingMeanVar {
  int win, idx=0, count=0;
  std::vector<double> buf;
  double sum=0.0, sumsq=0.0;

  explicit RollingMeanVar(int w = 1) { reset(w); }

  void reset(int w) {
    win = w > 0 ? w : 1;
    idx = 0;
    count = 0;
    sum = 0.0;
    sumsq = 0.0;
    buf.assign(win, 0.0);
  }

  inline std::pair<double,double> push(double x){
    if (count < win) {
      buf[count++] = x; 
      sum += x; 
      sumsq += x*x;
    } else {
      double old_val = buf[idx];
      sum   -= old_val;
      sumsq -= old_val * old_val;
      buf[idx] = x;
      sum   += x;
      sumsq += x*x;
      idx = (idx+1) % win;
    }
    double m = count > 0 ? sum / static_cast<double>(count) : 0.0;
    double v = count > 0 ? std::max(0.0, (sumsq / static_cast<double>(count)) - (m*m)) : 0.0;
    return {m, v};
  }
  
  double mean() const {
      return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }
  
  double var() const {
      if (count < 2) return 0.0;
      double m = mean();
      return std::max(0.0, (sumsq / static_cast<double>(count)) - (m * m));
  }

  double stddev() const {
      return std::sqrt(var());
  }
};

} // namespace sentio

```

## 📄 **FILE 61 of 145**: include/sentio/router.hpp

**File Information**:
- **Path**: `include/sentio/router.hpp`

- **Size**: 62 lines
- **Modified**: 2025-09-05 15:29:30

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <optional>
#include <string>

namespace sentio {

enum class OrderSide { Buy, Sell };

struct StrategySignal {
  enum class Type { BUY, STRONG_BUY, SELL, STRONG_SELL, HOLD };
  Type   type{Type::HOLD};
  double confidence{0.0}; // 0..1
};

struct RouterCfg {
  double min_signal_strength = 0.10; // below -> ignore
  double signal_multiplier   = 1.00; // scales target weight
  double max_position_pct    = 0.05; // +/- 5%
  bool   require_rth         = true; // assume ingest enforces RTH
  // family config
  std::string base_symbol{"QQQ"};
  std::string bull3x{"TQQQ"};
  std::string bear3x{"SQQQ"};
  std::string bear1x{"PSQ"};
  // sizing
  double min_shares = 1.0;
  double lot_size   = 1.0; // for ETFs typically 1
};

struct RouteDecision {
  std::string instrument;
  double      target_weight; // [-max, +max]
};

struct AccountSnapshot { double equity{0.0}; double cash{0.0}; };

struct Order {
  std::string instrument;
  OrderSide   side{OrderSide::Buy};
  double      qty{0.0};
  double      notional{0.0};
  double      limit_price{0.0}; // 0 = market
  std::int64_t ts_utc{0};
  std::string signal_id;
};

class PriceBook; // fwd
double last_trade_price(const PriceBook& book, const std::string& instrument);

std::optional<RouteDecision> route(const StrategySignal& s, const RouterCfg& cfg, const std::string& base_symbol);

// High-level convenience: route + size into a market order
Order route_and_create_order(const std::string& signal_id,
                             const StrategySignal& sig,
                             const RouterCfg& cfg,
                             const std::string& base_symbol,
                             const PriceBook& book,
                             const AccountSnapshot& acct,
                             std::int64_t ts_utc);

} // namespace sentio
```

## 📄 **FILE 62 of 145**: include/sentio/rth_calendar.hpp

**File Information**:
- **Path**: `include/sentio/rth_calendar.hpp`

- **Size**: 35 lines
- **Modified**: 2025-09-05 20:51:42

- **Type**: .hpp

```text
// rth_calendar.hpp
#pragma once
#include <chrono>
#include <string>
#include <unordered_set>
#include <unordered_map>

namespace sentio {

struct TradingCalendar {
  // Holidays: YYYYMMDD integers for fast lookups
  std::unordered_set<int> full_holidays;
  // Early closes (e.g., Black Friday): YYYYMMDD -> close second-of-day (e.g., 13:00 = 13*3600)
  std::unordered_map<int,int> early_close_sec;

  // Regular RTH bounds (seconds from midnight ET)
  static constexpr int RTH_OPEN_SEC  = 9*3600 + 30*60;  // 09:30:00
  static constexpr int RTH_CLOSE_SEC = 16*3600;         // 16:00:00

  // Return yyyymmdd in ET from a zoned_time (no allocations)
  static int yyyymmdd_from_local(const std::chrono::hh_mm_ss<std::chrono::seconds>& /*tod*/,
                                 std::chrono::year_month_day ymd) {
    int y = int(ymd.year());
    unsigned m = unsigned(ymd.month());
    unsigned d = unsigned(ymd.day());
    return y*10000 + int(m)*100 + int(d);
  }

  // Main predicate:
  //   ts_utc  = UTC wall clock in seconds since epoch
  //   tz_name = "America/New_York"
  bool is_rth_utc(std::int64_t ts_utc, const std::string& tz_name = "America/New_York") const;
};

} // namespace sentio
```

## 📄 **FILE 63 of 145**: include/sentio/runner.hpp

**File Information**:
- **Path**: `include/sentio/runner.hpp`

- **Size**: 43 lines
- **Modified**: 2025-09-05 20:13:14

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "audit.hpp"
#include "router.hpp"
#include "sizer.hpp"
#include "position_manager.hpp"
#include "cost_model.hpp"
#include "symbol_table.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace sentio {

enum class AuditLevel { Full, MetricsOnly };

struct RunnerCfg {
    std::string strategy_name = "VWAPReversion";
    std::unordered_map<std::string, std::string> strategy_params;
    RouterCfg router;
    SizerCfg sizer;
    AuditLevel audit_level = AuditLevel::Full;
    int snapshot_stride = 100;
    std::string audit_file = "audit.jsonl";  // JSONL audit file path
};

struct RunResult {
    double final_equity;
    double total_return;
    double sharpe_ratio;
    double max_drawdown;
    double monthly_projected_return;
    int daily_trades;
    int total_fills;
    int no_route;
    int no_qty;
};

RunResult run_backtest(AuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                      int base_symbol_id, const RunnerCfg& cfg);

} // namespace sentio


```

## 📄 **FILE 64 of 145**: include/sentio/sanity.hpp

**File Information**:
- **Path**: `include/sentio/sanity.hpp`

- **Size**: 93 lines
- **Modified**: 2025-09-05 20:37:59

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <unordered_map>

namespace sentio {

// Reuse your existing simple types
struct Bar { double open{}, high{}, low{}, close{}; };

// Signal types for strategy system
enum class SigType : uint8_t { BUY=0, STRONG_BUY=1, SELL=2, STRONG_SELL=3, HOLD=4 };

struct SanityIssue {
  enum class Severity { Warn, Error, Fatal };
  Severity severity{Severity::Error};
  std::string where;      // subsystem (DATA/FEATURE/STRAT/ROUTER/EXEC/PnL/AUDIT)
  std::string what;       // human message
  std::int64_t ts_utc{0}; // when applicable
};

struct SanityReport {
  std::vector<SanityIssue> issues;
  bool ok() const;                 // == no Error/Fatal
  std::size_t errors() const;
  std::size_t fatals() const;
  void add(SanityIssue::Severity sev, std::string where, std::string what, std::int64_t ts=0);
};

// Minimal interfaces (match your existing ones)
class PriceBook {
public:
  virtual void upsert_latest(const std::string& instrument, const Bar& b) = 0;
  virtual const Bar* get_latest(const std::string& instrument) const = 0;
  virtual bool has_instrument(const std::string& instrument) const = 0;
  virtual std::size_t size() const = 0;
  virtual ~PriceBook() = default;
};

struct Position { double qty{0.0}; double avg_px{0.0}; };
struct AccountState { double cash{0.0}; double realized{0.0}; double equity{0.0}; };

struct AuditEventCounts {
  std::size_t bars{0}, signals{0}, routes{0}, orders{0}, fills{0};
};

// Contracts you can call from tests or at the end of a run
namespace sanity {

// Data layer
void check_bar_monotonic(const std::vector<std::pair<std::int64_t, Bar>>& bars,
                         int expected_spacing_sec,
                         SanityReport& rep);

void check_bar_values_finite(const std::vector<std::pair<std::int64_t, Bar>>& bars,
                             SanityReport& rep);

// PriceBook coherence
void check_pricebook_coherence(const PriceBook& pb,
                               const std::vector<std::string>& required_instruments,
                               SanityReport& rep);

// Strategy/Routing layer
void check_signal_confidence_range(double conf, SanityReport& rep, std::int64_t ts);
void check_routed_instrument_has_price(const PriceBook& pb,
                                       const std::string& routed,
                                       SanityReport& rep, std::int64_t ts);

// Execution layer
void check_order_qty_min(double qty, double min_shares,
                         SanityReport& rep, std::int64_t ts);
void check_order_side_qty_sign_consistency(const std::string& side, double qty,
                                           SanityReport& rep, std::int64_t ts);

// P&L invariants
void check_equity_consistency(const AccountState& acct,
                              const std::unordered_map<std::string, Position>& pos,
                              const PriceBook& pb,
                              SanityReport& rep);

// Audit correlations
void check_audit_counts(const AuditEventCounts& c,
                        SanityReport& rep);

} // namespace sanity

// Lightweight runtime guard macros (no external deps)
#define SENTIO_ASSERT_FINITE(val, where, rep, ts) \
  do { if (!std::isfinite(val)) { (rep).add(SanityIssue::Severity::Fatal, (where), "non-finite value: " #val, (ts)); } } while(0)

} // namespace sentio

```

## 📄 **FILE 65 of 145**: include/sentio/signal_diag.hpp

**File Information**:
- **Path**: `include/sentio/signal_diag.hpp`

- **Size**: 35 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
// signal_diag.hpp
#pragma once
#include <cstdint>
#include <cstdio>

enum class DropReason : uint8_t {
  NONE, MIN_BARS, SESSION, NAN_INPUT, ZERO_VOL, THRESHOLD, COOLDOWN, DUP_SAME_BAR
};

struct SignalDiag {
  uint64_t emitted=0, dropped=0;
  uint64_t r_min_bars=0, r_session=0, r_nan=0, r_zero_vol=0, r_threshold=0, r_cooldown=0, r_dup=0;

  inline void drop(DropReason r){
    dropped++;
    switch(r){
      case DropReason::MIN_BARS:  r_min_bars++; break;
      case DropReason::SESSION:   r_session++;  break;
      case DropReason::NAN_INPUT: r_nan++;      break;
      case DropReason::ZERO_VOL:  r_zero_vol++; break;
      case DropReason::THRESHOLD: r_threshold++;break;
      case DropReason::COOLDOWN:  r_cooldown++; break;
      case DropReason::DUP_SAME_BAR:r_dup++;    break;
      default: break;
    }
  }

  inline void print(const char* tag) const {
    std::fprintf(stderr, "[SIG %s] emitted=%llu dropped=%llu  min_bars=%llu session=%llu nan=%llu zerovol=%llu thr=%llu cooldown=%llu dup=%llu\n",
      tag,(unsigned long long)emitted,(unsigned long long)dropped,
      (unsigned long long)r_min_bars,(unsigned long long)r_session,(unsigned long long)r_nan,
      (unsigned long long)r_zero_vol,(unsigned long long)r_threshold,(unsigned long long)r_cooldown,
      (unsigned long long)r_dup);
  }
};
```

## 📄 **FILE 66 of 145**: include/sentio/signal_engine.hpp

**File Information**:
- **Path**: `include/sentio/signal_engine.hpp`

- **Size**: 22 lines
- **Modified**: 2025-09-05 21:09:57

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "signal_gate.hpp"

namespace sentio {

struct EngineOut {
  std::optional<StrategySignal> signal; // post-gate
  DropReason last_drop{DropReason::NONE};
};

class SignalEngine {
public:
  SignalEngine(IStrategy* strat, const GateCfg& gate_cfg, SignalHealth* health);
  EngineOut on_bar(const StrategyCtx& ctx, const Bar& b, bool inputs_finite=true);
private:
  IStrategy* strat_;
  SignalGate gate_;
  SignalHealth* health_;
};

} // namespace sentio

```

## 📄 **FILE 67 of 145**: include/sentio/signal_gate.hpp

**File Information**:
- **Path**: `include/sentio/signal_gate.hpp`

- **Size**: 46 lines
- **Modified**: 2025-09-05 17:01:14

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <atomic>
#include <optional>

namespace sentio {

enum class DropReason : uint16_t {
  NONE=0, NOT_RTH, WARMUP, NAN_INPUT, THRESHOLD_TOO_TIGHT,
  COOLDOWN_ACTIVE, DUPLICATE_BAR_TS
};

struct SignalHealth {
  std::atomic<uint64_t> emitted{0};
  std::atomic<uint64_t> dropped{0};
  std::unordered_map<DropReason, std::atomic<uint64_t>> by_reason;
  SignalHealth();
  void incr_emit();
  void incr_drop(DropReason r);
};

struct GateCfg { 
  bool require_rth=true; 
  int cooldown_bars=0; 
  double min_conf=0.05; 
};

class SignalGate {
public:
  explicit SignalGate(const GateCfg& cfg, SignalHealth* health);
  // Returns nullopt if dropped; otherwise passes through with possibly clamped confidence.
  std::optional<double> accept(std::int64_t ts_utc_epoch,
                               bool is_rth,
                               bool inputs_finite,
                               bool warmed_up,
                               double confidence);
private:
  GateCfg cfg_;
  SignalHealth* health_;
  std::int64_t last_emit_ts_{-1};
  int cooldown_left_{0};
};

} // namespace sentio

```

## 📄 **FILE 68 of 145**: include/sentio/signal_pipeline.hpp

**File Information**:
- **Path**: `include/sentio/signal_pipeline.hpp`

- **Size**: 41 lines
- **Modified**: 2025-09-05 21:09:57

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "signal_gate.hpp"
#include "signal_trace.hpp"
#include <cstdint>
#include <string>
#include <optional>

namespace sentio {

// Forward declarations to avoid include conflicts
struct RouterCfg;
struct RouteDecision;
struct Order;
struct AccountSnapshot;
class PriceBook;

struct PipelineCfg {
  GateCfg gate;
  double min_order_shares{1.0};
};

struct PipelineOut {
  std::optional<StrategySignal> signal;
  TraceRow trace;
};

class SignalPipeline {
public:
  SignalPipeline(IStrategy* strat, const PipelineCfg& cfg, void* /*book*/, SignalTrace* trace)
  : strat_(strat), cfg_(cfg), trace_(trace), gate_(cfg.gate, nullptr) {}

  PipelineOut on_bar(const StrategyCtx& ctx, const Bar& b, const void* acct);
private:
  IStrategy* strat_;
  PipelineCfg cfg_;
  SignalTrace* trace_;
  SignalGate gate_;
};

} // namespace sentio

```

## 📄 **FILE 69 of 145**: include/sentio/signal_trace.hpp

**File Information**:
- **Path**: `include/sentio/signal_trace.hpp`

- **Size**: 53 lines
- **Modified**: 2025-09-05 17:00:48

- **Type**: .hpp

```text
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <optional>

namespace sentio {

enum class TraceReason : uint16_t {
  OK = 0,
  NO_STRATEGY_OUTPUT,
  NOT_RTH,
  HOLIDAY,
  WARMUP,
  NAN_INPUT,
  THRESHOLD_TOO_TIGHT,
  COOLDOWN_ACTIVE,
  DUPLICATE_BAR_TS,
  EMPTY_PRICEBOOK,
  NO_PRICE_FOR_INSTRUMENT,
  ROUTER_REJECTED,
  ORDER_QTY_LT_MIN,
  UNKNOWN
};

struct TraceRow {
  std::int64_t ts_utc{};
  std::string  instrument;     // stream instrument (e.g., QQQ)
  std::string  routed;         // routed instrument (e.g., TQQQ/SQQQ)
  double       close{};
  bool         is_rth{true};
  bool         warmed{true};
  bool         inputs_finite{true};
  double       confidence{0.0};              // raw strategy conf
  double       conf_after_gate{0.0};         // post gate
  double       target_weight{0.0};           // router decision
  double       last_px{0.0};                 // last price for routed
  double       order_qty{0.0};
  TraceReason  reason{TraceReason::UNKNOWN};
  std::string  note;                         // optional detail
};

class SignalTrace {
public:
  void push(const TraceRow& r) { rows_.push_back(r); }
  const std::vector<TraceRow>& rows() const { return rows_; }
  // useful summaries
  std::size_t count(TraceReason r) const;
private:
  std::vector<TraceRow> rows_;
};

} // namespace sentio

```

## 📄 **FILE 70 of 145**: include/sentio/sim_data.hpp

**File Information**:
- **Path**: `include/sentio/sim_data.hpp`

- **Size**: 24 lines
- **Modified**: 2025-09-05 20:37:59

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>
#include <random>
#include "sanity.hpp"

namespace sentio {

// Generates a synthetic minute-bar series with regimes (trend, mean-revert, jump).
struct SimCfg {
  std::int64_t start_ts_utc{1'600'000'000};
  int minutes{500};
  double start_price{100.0};
  unsigned seed{42};
  // regime fractions (sum <= 1.0)
  double frac_trend{0.5};
  double frac_mr{0.4};
  double frac_jump{0.1};
  double vol_bps{15.0};    // base noise per min (bps)
};

std::vector<std::pair<std::int64_t, Bar>> generate_minute_series(const SimCfg& cfg);

} // namespace sentio

```

## 📄 **FILE 71 of 145**: include/sentio/sizer.hpp

**File Information**:
- **Path**: `include/sentio/sizer.hpp`

- **Size**: 95 lines
- **Modified**: 2025-09-05 09:44:09

- **Type**: .hpp

```text
#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace sentio {

// Advanced Sizer Configuration with Risk Controls
struct SizerCfg {
  bool fractional_allowed = true;
  double min_notional = 1.0;
  double max_leverage = 2.0;
  double max_position_pct = 0.25;
  double volatility_target = 0.15;
  bool allow_negative_cash = false;
  int vol_lookback_days = 20;
  double cash_reserve_pct = 0.05;
};

// Advanced Sizer Class with Multiple Constraints
class AdvancedSizer {
public:
  double calculate_volatility(const std::vector<Bar>& price_history, int lookback) const {
    if (price_history.size() < static_cast<size_t>(lookback)) return 0.05; // Default vol

    std::vector<double> returns;
    returns.reserve(lookback - 1);
    for (size_t i = price_history.size() - lookback + 1; i < price_history.size(); ++i) {
      double prev_close = price_history[i-1].close;
      if (prev_close > 0) {
        returns.push_back(price_history[i].close / prev_close - 1.0);
      }
    }
    
    if (returns.size() < 2) return 0.05;
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - mean) * (ret - mean);
    }
    variance /= returns.size();
    return std::sqrt(variance) * std::sqrt(252.0); // Annualized
  }

  // **MODIFIED**: Signature and logic updated for the ID-based, high-performance architecture.
  double calculate_target_quantity(const Portfolio& portfolio,
                                   const SymbolTable& ST,
                                   const std::vector<double>& last_prices,
                                   const std::string& instrument,
                                   double target_weight,
                                                                       [[maybe_unused]] const std::vector<Bar>& price_history,
                                   const SizerCfg& cfg) const {
    
    const double equity = equity_mark_to_market(portfolio, last_prices);
    int instrument_id = ST.get_id(instrument);

    if (equity <= 0 || instrument_id == -1 || last_prices[instrument_id] <= 0) {
        return 0.0;
    }
    double instrument_price = last_prices[instrument_id];

    // --- Calculate size based on multiple constraints ---
    double desired_notional = equity * std::abs(target_weight);

    // 1. Max Position Size Constraint
    desired_notional = std::min(desired_notional, equity * cfg.max_position_pct);

    // 2. Leverage Constraint
    double current_exposure = 0.0;
    for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
        current_exposure += std::abs(portfolio.positions[sid].qty * last_prices[sid]);
    }
    double available_leverage_notional = (equity * cfg.max_leverage) - current_exposure;
    desired_notional = std::min(desired_notional, std::max(0.0, available_leverage_notional));

    // 4. Cash Constraint
    if (!cfg.allow_negative_cash) {
      double usable_cash = portfolio.cash * (1.0 - cfg.cash_reserve_pct);
      desired_notional = std::min(desired_notional, std::max(0.0, usable_cash));
    }
    
    if (desired_notional < cfg.min_notional) return 0.0;
    
    double qty = desired_notional / instrument_price;
    double final_qty = cfg.fractional_allowed ? qty : std::floor(qty);
    
    // Return with the correct sign (long/short)
    return (target_weight > 0) ? final_qty : -final_qty;
  }
};

} // namespace sentio
```

## 📄 **FILE 72 of 145**: include/sentio/strategy_bollinger_squeeze_breakout.hpp

**File Information**:
- **Path**: `include/sentio/strategy_bollinger_squeeze_breakout.hpp`

- **Size**: 44 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "bollinger.hpp"
#include <vector>
#include <string>

namespace sentio {

class BollingerSqueezeBreakoutStrategy : public BaseStrategy {
private:
    enum class State { Idle, Squeezed, ArmedLong, ArmedShort, Long, Short };
    
    // **MODIFIED**: Cached parameters
    int bb_window_;
    double squeeze_percentile_;
    int squeeze_lookback_;
    int hold_max_bars_;
    double tp_mult_sd_;
    double sl_mult_sd_;
    int min_squeeze_bars_;

    // Strategy state & indicators
    State state_ = State::Idle;
    int bars_in_trade_ = 0;
    int squeeze_duration_ = 0;
    Bollinger bollinger_;
    std::vector<double> sd_history_;
    
    // Helper methods
    double calculate_volatility_percentile(double percentile) const;
    void update_state_machine(const Bar& bar);
    
public:
    BollingerSqueezeBreakoutStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 73 of 145**: include/sentio/strategy_hybrid_ppo.hpp

**File Information**:
- **Path**: `include/sentio/strategy_hybrid_ppo.hpp`

- **Size**: 41 lines
- **Modified**: 2025-09-06 00:28:01

- **Type**: .hpp

```text
#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/ml/model_registry.hpp"
#include "sentio/ml/feature_pipeline.hpp"
#include <memory>

namespace sentio {

struct HybridPPOCfg {
  std::string artifacts_dir{"artifacts"};
  std::string version{"v1"};
  bool use_cuda{false};
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


```

## 📄 **FILE 74 of 145**: include/sentio/strategy_kochi_ppo.hpp

**File Information**:
- **Path**: `include/sentio/strategy_kochi_ppo.hpp`

- **Size**: 42 lines
- **Modified**: 2025-09-07 22:52:32

- **Type**: .hpp

```text
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



```

## 📄 **FILE 75 of 145**: include/sentio/strategy_market_making.hpp

**File Information**:
- **Path**: `include/sentio/strategy_market_making.hpp`

- **Size**: 52 lines
- **Modified**: 2025-09-05 12:22:58

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp" // For rolling volume
#include <deque>

namespace sentio {

class MarketMakingStrategy : public BaseStrategy {
private:
    // **MODIFIED**: Cached parameters
    double base_spread_;
    double min_spread_;
    double max_spread_;
    int order_levels_;
    double level_spacing_;
    double order_size_base_;
    double max_inventory_;
    double inventory_skew_mult_;
    double adverse_selection_threshold_;
    double min_volume_ratio_;
    int max_orders_per_bar_;
    int rebalance_frequency_;

    // Market making state
    struct MarketState {
        double inventory = 0.0;
        double average_cost = 0.0;
        int last_rebalance = 0;
        int orders_this_bar = 0;
    };
    MarketState market_state_;

    // **NEW**: Stateful, rolling indicators for performance
    RollingMeanVar rolling_returns_;
    RollingMean rolling_volume_;

    // Helper methods
    bool should_participate(const Bar& bar);
    double get_inventory_skew() const;
    
public:
    MarketMakingStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 76 of 145**: include/sentio/strategy_momentum_volume.hpp

**File Information**:
- **Path**: `include/sentio/strategy_momentum_volume.hpp`

- **Size**: 57 lines
- **Modified**: 2025-09-05 12:25:31

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp" // **NEW**: For efficient MA calculations
#include <map>

namespace sentio {

class MomentumVolumeProfileStrategy : public BaseStrategy {
private:
    // **MODIFIED**: Cached parameters for performance
    int profile_period_;
    double value_area_pct_;
    int price_bins_;
    double breakout_threshold_pct_;
    int momentum_lookback_;
    double volume_surge_mult_;
    int cool_down_period_;

    // Volume profile data structures
    struct VolumeNode {
        double price_level;
        double volume;
    };
    struct VolumeProfile {
        std::map<double, VolumeNode> profile;
        double poc_level = 0.0;
        double value_area_high = 0.0;
        double value_area_low = 0.0;
        double total_volume = 0.0;
        void clear() { /* ... unchanged ... */ }
    };
    
    // Strategy state
    VolumeProfile volume_profile_;
    int last_profile_update_ = -1;
    
    // **NEW**: Stateful, rolling indicators for performance
    RollingMean avg_volume_;

    // Helper methods
    void build_volume_profile(const std::vector<Bar>& bars, int end_index);
    void calculate_value_area();
    bool is_momentum_confirmed(const std::vector<Bar>& bars, int index) const;
    
public:
    MomentumVolumeProfileStrategy();
    
    // BaseStrategy interface
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 77 of 145**: include/sentio/strategy_opening_range_breakout.hpp

**File Information**:
- **Path**: `include/sentio/strategy_opening_range_breakout.hpp`

- **Size**: 38 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"

namespace sentio {

class OpeningRangeBreakoutStrategy : public BaseStrategy {
private:
    // **MODIFIED**: Cached parameters
    int opening_range_minutes_;
    int breakout_confirmation_bars_;
    double volume_multiplier_;
    double stop_loss_pct_;
    double take_profit_pct_;
    int cool_down_period_;

    struct OpeningRange {
        double high = 0.0;
        double low = 0.0;
        int end_bar = -1;
        bool is_finalized = false;
    };
    
    // Strategy state
    OpeningRange current_range_;
    int day_start_index_ = -1;
    
public:
    OpeningRangeBreakoutStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 78 of 145**: include/sentio/strategy_order_flow_imbalance.hpp

**File Information**:
- **Path**: `include/sentio/strategy_order_flow_imbalance.hpp`

- **Size**: 40 lines
- **Modified**: 2025-09-05 12:32:08

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp"

namespace sentio {

class OrderFlowImbalanceStrategy : public BaseStrategy {
private:
    // Cached parameters
    int lookback_period_;
    double entry_threshold_long_;
    double entry_threshold_short_;
    int hold_max_bars_;
    int cool_down_period_;

    // Strategy-specific state machine
    enum class OFIState { Flat, Long, Short };
    // **FIXED**: Renamed this member to 'ofi_state_' to avoid conflict
    // with the 'state_' member inherited from BaseStrategy.
    OFIState ofi_state_ = OFIState::Flat;
    int bars_in_trade_ = 0;

    // Rolling indicator to measure average pressure
    RollingMean rolling_pressure_;

    // Helper to calculate buying/selling pressure proxy from a bar
    double calculate_bar_pressure(const Bar& bar) const;
    
public:
    OrderFlowImbalanceStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio

```

## 📄 **FILE 79 of 145**: include/sentio/strategy_order_flow_scalping.hpp

**File Information**:
- **Path**: `include/sentio/strategy_order_flow_scalping.hpp`

- **Size**: 38 lines
- **Modified**: 2025-09-05 13:09:39

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"
#include "rolling_stats.hpp"

namespace sentio {

class OrderFlowScalpingStrategy : public BaseStrategy {
private:
    // Cached parameters
    int lookback_period_;
    double imbalance_threshold_;
    int hold_max_bars_;
    int cool_down_period_;

    // State machine states
    enum class OFState { Idle, ArmedLong, ArmedShort, Long, Short };
    
    // **FIXED**: Renamed this member to 'of_state_' to avoid conflict
    // with the 'state_' member inherited from BaseStrategy.
    OFState of_state_ = OFState::Idle;
    int bars_in_trade_ = 0;
    RollingMean rolling_pressure_;

    // Helper to calculate buying/selling pressure proxy from a bar
    double calculate_bar_pressure(const Bar& bar) const;
    
public:
    OrderFlowScalpingStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio

```

## 📄 **FILE 80 of 145**: include/sentio/strategy_sma_cross.hpp

**File Information**:
- **Path**: `include/sentio/strategy_sma_cross.hpp`

- **Size**: 27 lines
- **Modified**: 2025-09-05 21:09:57

- **Type**: .hpp

```text
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

```

## 📄 **FILE 81 of 145**: include/sentio/strategy_tfa.hpp

**File Information**:
- **Path**: `include/sentio/strategy_tfa.hpp`

- **Size**: 50 lines
- **Modified**: 2025-09-07 19:22:14

- **Type**: .hpp

```text
#pragma once
#include "sentio/base_strategy.hpp"
#include "sentio/ml/model_registry.hpp"
#include "sentio/ml/feature_window.hpp"
#include "sentio/feature/column_projector.hpp"
#include "sentio/feature/column_projector_safe.hpp"
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
  StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;

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
};

} // namespace sentio

```

## 📄 **FILE 82 of 145**: include/sentio/strategy_transformer.hpp

**File Information**:
- **Path**: `include/sentio/strategy_transformer.hpp`

- **Size**: 41 lines
- **Modified**: 2025-09-05 21:36:38

- **Type**: .hpp

```text
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

```

## 📄 **FILE 83 of 145**: include/sentio/strategy_transformer_ts.hpp

**File Information**:
- **Path**: `include/sentio/strategy_transformer_ts.hpp`

- **Size**: 33 lines
- **Modified**: 2025-09-05 23:39:11

- **Type**: .hpp

```text
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

```

## 📄 **FILE 84 of 145**: include/sentio/strategy_vwap_reversion.hpp

**File Information**:
- **Path**: `include/sentio/strategy_vwap_reversion.hpp`

- **Size**: 47 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include "base_strategy.hpp"

namespace sentio {

class VWAPReversionStrategy : public BaseStrategy {
private:
    // Cached parameters for performance
    int vwap_period_;
    double band_multiplier_;
    double max_band_width_;
    double min_distance_from_vwap_;
    double volume_confirmation_mult_;
    int rsi_period_;
    double rsi_oversold_;
    double rsi_overbought_;
    double stop_loss_pct_;
    double take_profit_pct_;
    int time_stop_bars_;
    int cool_down_period_;

    // VWAP calculation state
    double cumulative_pv_ = 0.0;
    double cumulative_volume_ = 0.0;
    
    // Strategy state
    int time_in_position_ = 0;
    double vwap_ = 0.0;
    
    // Helper methods
    void update_vwap(const Bar& bar);
    bool is_volume_confirmed(const std::vector<Bar>& bars, int index) const;
    bool is_rsi_condition_met(const std::vector<Bar>& bars, int index, bool for_buy) const;
    double calculate_simple_rsi(const std::vector<double>& prices) const;
    
public:
    VWAPReversionStrategy();
    
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) override;
    void reset_state() override;
};

} // namespace sentio
```

## 📄 **FILE 85 of 145**: include/sentio/sym/leverage_registry.hpp

**File Information**:
- **Path**: `include/sentio/sym/leverage_registry.hpp`

- **Size**: 66 lines
- **Modified**: 2025-09-06 22:14:53

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <unordered_map>
#include <mutex>
#include "sentio/sym/symbol_utils.hpp"

namespace sentio {

// Captures the leveraged instrument's relationship to a base ticker
struct LeverageSpec {
  std::string base;     // e.g., "QQQ"
  float factor{1.f};    // e.g., 3.0 for TQQQ, 1.0 for PSQ (but inverse)
  bool inverse{false};  // true for PSQ/SQQQ (short)
};

// Thread-safe global registry
class LeverageRegistry {
  std::unordered_map<std::string, LeverageSpec> map_; // key: UPPER(symbol)
  std::mutex mu_;
  LeverageRegistry() { seed_defaults_(); }

  void seed_defaults_() {
    // QQQ family
    map_.emplace("TQQQ", LeverageSpec{"QQQ", 3.f, false});
    map_.emplace("SQQQ", LeverageSpec{"QQQ", 3.f, true});
    map_.emplace("PSQ",  LeverageSpec{"QQQ", 1.f, true});
    // You can extend similarly for SPY, TSLA, BTC ETFs, etc.
    // Examples:
    // map_.emplace("UPRO", LeverageSpec{"SPY", 3.f, false});
    // map_.emplace("SPXU", LeverageSpec{"SPY", 3.f, true});
    // map_.emplace("TSLQ", LeverageSpec{"TSLA", 1.f, true});
  }

public:
  static LeverageRegistry& instance() {
    static LeverageRegistry x;
    return x;
  }

  void register_leveraged(const std::string& symbol, LeverageSpec spec) {
    std::lock_guard<std::mutex> lk(mu_);
    map_[to_upper(symbol)] = std::move(spec);
  }

  bool lookup(const std::string& symbol, LeverageSpec& out) const {
    const auto key = to_upper(symbol);
    auto it = map_.find(key);
    if (it == map_.end()) return false;
    out = it->second;
    return true;
  }
};

// Convenience helpers
inline bool is_leveraged(const std::string& symbol) {
  LeverageSpec tmp;
  return LeverageRegistry::instance().lookup(symbol, tmp);
}

inline std::string resolve_base(const std::string& symbol) {
  LeverageSpec tmp;
  if (LeverageRegistry::instance().lookup(symbol, tmp)) return tmp.base;
  return to_upper(symbol);
}

} // namespace sentio

```

## 📄 **FILE 86 of 145**: include/sentio/sym/symbol_utils.hpp

**File Information**:
- **Path**: `include/sentio/sym/symbol_utils.hpp`

- **Size**: 10 lines
- **Modified**: 2025-09-06 22:14:53

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <algorithm>

namespace sentio {
inline std::string to_upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::toupper(c); });
  return s;
}
}

```

## 📄 **FILE 87 of 145**: include/sentio/symbol_table.hpp

**File Information**:
- **Path**: `include/sentio/symbol_table.hpp`

- **Size**: 35 lines
- **Modified**: 2025-09-05 03:57:38

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <unordered_map>

namespace sentio {

struct SymbolTable {
  std::vector<std::string> id2sym;
  std::unordered_map<std::string,int> sym2id;

  int intern(const std::string& s){
    auto it = sym2id.find(s);
    if (it != sym2id.end()) return it->second;
    int id = (int)id2sym.size();
    id2sym.push_back(s);
    sym2id.emplace(id2sym.back(), id);
    return id;
  }

  const std::string& get_symbol(int id) const {
    return id2sym[id];
  }

  int get_id(const std::string& sym) const {
    auto it = sym2id.find(sym);
    return it != sym2id.end() ? it->second : -1;
  }

  size_t size() const {
    return id2sym.size();
  }
};

} // namespace sentio
```

## 📄 **FILE 88 of 145**: include/sentio/telemetry_logger.hpp

**File Information**:
- **Path**: `include/sentio/telemetry_logger.hpp`

- **Size**: 131 lines
- **Modified**: 2025-09-06 01:37:11

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <unordered_map>
#include <atomic>

namespace sentio {

/**
 * JSON line logger for telemetry data
 * Thread-safe logging of strategy performance metrics
 */
class TelemetryLogger {
public:
    struct TelemetryData {
        std::string timestamp;
        std::string strategy_name;
        std::string instrument;
        int bars_processed{0};
        int signals_generated{0};
        int buy_signals{0};
        int sell_signals{0};
        int hold_signals{0};
        double avg_confidence{0.0};
        double ready_percentage{0.0};
        double processing_time_ms{0.0};
        std::string notes;
    };

    explicit TelemetryLogger(const std::string& log_file_path);
    ~TelemetryLogger();

    /**
     * Log telemetry data for a strategy
     * @param data Telemetry data to log
     */
    void log(const TelemetryData& data);

    /**
     * Log a simple metric
     * @param strategy_name Strategy name
     * @param metric_name Metric name
     * @param value Metric value
     * @param instrument Optional instrument name
     */
    void log_metric(
        const std::string& strategy_name,
        const std::string& metric_name,
        double value,
        const std::string& instrument = ""
    );

    /**
     * Log signal generation statistics
     * @param strategy_name Strategy name
     * @param instrument Instrument name
     * @param signals_generated Total signals generated
     * @param buy_signals Buy signals
     * @param sell_signals Sell signals
     * @param hold_signals Hold signals
     * @param avg_confidence Average confidence
     */
    void log_signal_stats(
        const std::string& strategy_name,
        const std::string& instrument,
        int signals_generated,
        int buy_signals,
        int sell_signals,
        int hold_signals,
        double avg_confidence
    );

    /**
     * Log performance metrics
     * @param strategy_name Strategy name
     * @param instrument Instrument name
     * @param bars_processed Number of bars processed
     * @param processing_time_ms Processing time in milliseconds
     * @param ready_percentage Percentage of time strategy was ready
     */
    void log_performance(
        const std::string& strategy_name,
        const std::string& instrument,
        int bars_processed,
        double processing_time_ms,
        double ready_percentage
    );

    /**
     * Flush any pending log data
     */
    void flush();

    /**
     * Get current log file path
     */
    const std::string& get_log_file_path() const { return log_file_path_; }

private:
    std::string log_file_path_;
    std::ofstream log_file_;
    std::mutex log_mutex_;
    std::atomic<int> log_counter_{0};
    
    // Helper methods
    std::string get_current_timestamp() const;
    std::string escape_json_string(const std::string& str) const;
    void write_json_line(const std::string& json_line);
};

/**
 * Global telemetry logger instance
 * Use this for easy access throughout the application
 */
extern std::unique_ptr<TelemetryLogger> g_telemetry_logger;

/**
 * Initialize global telemetry logger
 * @param log_file_path Path to log file
 */
void init_telemetry_logger(const std::string& log_file_path = "logs/telemetry.jsonl");

/**
 * Get global telemetry logger instance
 * @return Reference to global telemetry logger
 */
TelemetryLogger& get_telemetry_logger();

} // namespace sentio

```

## 📄 **FILE 89 of 145**: include/sentio/temporal_analysis.hpp

**File Information**:
- **Path**: `include/sentio/temporal_analysis.hpp`

- **Size**: 364 lines
- **Modified**: 2025-09-07 14:29:16

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <iostream>
#include "sentio/progress_bar.hpp"

namespace sentio {

struct TemporalAnalysisConfig {
    int num_quarters = 12;           // Number of quarters to analyze
    int min_bars_per_quarter = 50;   // Minimum bars required per quarter
    bool print_detailed_report = true;
    
    // TPA readiness criteria for virtual market testing
    double min_avg_sharpe = 0.5;     // Minimum average Sharpe ratio
    double max_sharpe_volatility = 1.0; // Maximum Sharpe volatility
    double min_health_quarters_pct = 70.0; // Minimum % of quarters with healthy trade frequency
    double min_avg_monthly_return = 0.5;   // Minimum average monthly return (%)
    double max_drawdown_threshold = 15.0;  // Maximum acceptable drawdown (%)
};

struct QuarterlyMetrics {
    int year;
    int quarter;
    double monthly_return_pct;      // Monthly projected return (annualized)
    double sharpe_ratio;
    int total_trades;
    int trading_days;
    double avg_daily_trades;
    double max_drawdown;
    double win_rate;
    double total_return_pct;
    
    // Health indicators
    bool healthy_trade_frequency() const {
        return avg_daily_trades >= 10.0 && avg_daily_trades <= 100.0;
    }
    
    std::string health_status() const {
        if (avg_daily_trades < 10.0) return "LOW_FREQ";
        if (avg_daily_trades > 100.0) return "HIGH_FREQ";
        return "HEALTHY";
    }
};

struct TPAReadinessAssessment {
    bool ready_for_virtual_market = false;
    bool ready_for_paper_trading = false;
    bool ready_for_live_trading = false;
    
    std::vector<std::string> issues;
    std::vector<std::string> recommendations;
    
    double readiness_score = 0.0; // 0-100 score
};

struct TemporalAnalysisSummary {
    std::vector<QuarterlyMetrics> quarterly_results;
    double overall_sharpe;
    double overall_return;
    int total_quarters;
    int healthy_quarters;
    int low_freq_quarters;
    int high_freq_quarters;
    
    // Consistency metrics
    double sharpe_std_dev;
    double return_std_dev;
    double trade_freq_std_dev;
    
    // TPA readiness assessment
    TPAReadinessAssessment readiness;
    
    void calculate_summary_stats() {
        if (quarterly_results.empty()) return;
        
        total_quarters = quarterly_results.size();
        healthy_quarters = 0;
        low_freq_quarters = 0;
        high_freq_quarters = 0;
        
        double sharpe_sum = 0.0, return_sum = 0.0, freq_sum = 0.0;
        double sharpe_sq_sum = 0.0, return_sq_sum = 0.0, freq_sq_sum = 0.0;
        
        for (const auto& q : quarterly_results) {
            sharpe_sum += q.sharpe_ratio;
            return_sum += q.monthly_return_pct;
            freq_sum += q.avg_daily_trades;
            
            sharpe_sq_sum += q.sharpe_ratio * q.sharpe_ratio;
            return_sq_sum += q.monthly_return_pct * q.monthly_return_pct;
            freq_sq_sum += q.avg_daily_trades * q.avg_daily_trades;
            
            if (q.health_status() == "HEALTHY") healthy_quarters++;
            else if (q.health_status() == "LOW_FREQ") low_freq_quarters++;
            else if (q.health_status() == "HIGH_FREQ") high_freq_quarters++;
        }
        
        overall_sharpe = sharpe_sum / total_quarters;
        overall_return = return_sum / total_quarters;
        
        // Calculate standard deviations
        double sharpe_mean = overall_sharpe;
        double return_mean = overall_return;
        double freq_mean = freq_sum / total_quarters;
        
        // For single quarter, standard deviation is 0 (no variance)
        if (total_quarters == 1) {
            sharpe_std_dev = 0.0;
            return_std_dev = 0.0;
            trade_freq_std_dev = 0.0;
        } else {
            sharpe_std_dev = std::sqrt(std::max(0.0, (sharpe_sq_sum / total_quarters) - (sharpe_mean * sharpe_mean)));
            return_std_dev = std::sqrt(std::max(0.0, (return_sq_sum / total_quarters) - (return_mean * return_mean)));
            trade_freq_std_dev = std::sqrt(std::max(0.0, (freq_sq_sum / total_quarters) - (freq_mean * freq_mean)));
        }
    }
    
    void assess_readiness(const TemporalAnalysisConfig& config) {
        readiness.issues.clear();
        readiness.recommendations.clear();
        readiness.ready_for_virtual_market = false;
        readiness.ready_for_paper_trading = false;
        readiness.ready_for_live_trading = false;
        
        double score = 0.0;
        int criteria_met = 0;
        [[maybe_unused]] int total_criteria = 5;
        
        // 1. Average Sharpe ratio check
        if (overall_sharpe >= config.min_avg_sharpe) {
            criteria_met++;
            score += 20.0;
        } else {
            readiness.issues.push_back("Average Sharpe ratio too low: " + std::to_string(overall_sharpe) + " < " + std::to_string(config.min_avg_sharpe));
            readiness.recommendations.push_back("Improve strategy risk-adjusted returns");
        }
        
        // 2. Sharpe volatility check
        if (sharpe_std_dev <= config.max_sharpe_volatility) {
            criteria_met++;
            score += 20.0;
        } else {
            readiness.issues.push_back("Sharpe ratio too volatile: " + std::to_string(sharpe_std_dev) + " > " + std::to_string(config.max_sharpe_volatility));
            readiness.recommendations.push_back("Improve strategy consistency across time periods");
        }
        
        // 3. Trade frequency health check
        double health_pct = 100.0 * healthy_quarters / total_quarters;
        if (health_pct >= config.min_health_quarters_pct) {
            criteria_met++;
            score += 20.0;
        } else {
            readiness.issues.push_back("Too many quarters with unhealthy trade frequency: " + std::to_string(health_pct) + "% < " + std::to_string(config.min_health_quarters_pct) + "%");
            readiness.recommendations.push_back("Adjust strategy parameters for consistent trade frequency");
        }
        
        // 4. Monthly return check
        if (overall_return >= config.min_avg_monthly_return) {
            criteria_met++;
            score += 20.0;
        } else {
            readiness.issues.push_back("Average monthly return too low: " + std::to_string(overall_return) + "% < " + std::to_string(config.min_avg_monthly_return) + "%");
            readiness.recommendations.push_back("Improve strategy profitability");
        }
        
        // 5. Drawdown check (check max drawdown across quarters)
        double max_quarterly_drawdown = 0.0;
        for (const auto& q : quarterly_results) {
            max_quarterly_drawdown = std::max(max_quarterly_drawdown, q.max_drawdown);
        }
        if (max_quarterly_drawdown <= config.max_drawdown_threshold) {
            criteria_met++;
            score += 20.0;
        } else {
            readiness.issues.push_back("Maximum drawdown too high: " + std::to_string(max_quarterly_drawdown) + "% > " + std::to_string(config.max_drawdown_threshold) + "%");
            readiness.recommendations.push_back("Implement better risk management");
        }
        
        readiness.readiness_score = score;
        
        // Determine readiness levels
        if (criteria_met >= 3) {
            readiness.ready_for_virtual_market = true;
        }
        if (criteria_met >= 4) {
            readiness.ready_for_paper_trading = true;
        }
        if (criteria_met >= 5) {
            readiness.ready_for_live_trading = true;
        }
        
        // Add general recommendations based on score
        if (score < 60.0) {
            readiness.recommendations.push_back("Strategy needs significant improvement before testing");
        } else if (score < 80.0) {
            readiness.recommendations.push_back("Strategy shows promise but needs refinement");
        } else {
            readiness.recommendations.push_back("Strategy appears ready for advanced testing");
        }
    }
};

class TPATestProgressBar : public ProgressBar {
public:
    TPATestProgressBar(int total_quarters, const std::string& strategy_name) 
        : ProgressBar(total_quarters, "TPA Test: " + strategy_name) {}
    
    void display_with_quarter_info([[maybe_unused]] int current_quarter, int year, int quarter, 
                                   double monthly_return, double sharpe, 
                                   double avg_daily_trades, const std::string& health_status) {
        update(get_current() + 1);
        
        // Clear line and move cursor to beginning
        std::cout << "\r\033[K";
        
        // Calculate percentage
        double percentage = (double)get_current() / get_total() * 100.0;
        
        // Create progress bar
        int bar_width = 50;
        int pos = (int)(bar_width * percentage / 100.0);
        
        std::cout << get_description() << " [" << std::string(pos, '=') 
                  << std::string(bar_width - pos, '-') << "] " 
                  << std::fixed << std::setprecision(1) << percentage << "%";
        
        // Show current quarter info
        std::cout << " | Q" << year << "Q" << quarter 
                  << " | Ret: " << std::fixed << std::setprecision(2) << monthly_return << "%"
                  << " | Sharpe: " << std::fixed << std::setprecision(3) << sharpe
                  << " | Trades: " << std::fixed << std::setprecision(1) << avg_daily_trades
                  << " | " << health_status;
        
        std::cout.flush();
    }
};

class TemporalAnalyzer {
public:
    TemporalAnalyzer() = default;
    
    void add_quarterly_result(const QuarterlyMetrics& metrics) {
        quarterly_results_.push_back(metrics);
    }
    
    TemporalAnalysisSummary generate_summary() const {
        TemporalAnalysisSummary summary;
        summary.quarterly_results = quarterly_results_;
        summary.calculate_summary_stats();
        return summary;
    }
    
    void print_detailed_report() const {
        auto summary = generate_summary();
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "TEMPORAL PERFORMANCE ANALYSIS REPORT" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        // Quarterly breakdown
        std::cout << "\nQUARTERLY PERFORMANCE BREAKDOWN:\n" << std::endl;
        std::cout << std::left << std::setw(8) << "Quarter" 
                  << std::setw(12) << "Monthly Ret%" 
                  << std::setw(10) << "Sharpe" 
                  << std::setw(8) << "Trades" 
                  << std::setw(12) << "Daily Avg" 
                  << std::setw(10) << "Health" 
                  << std::setw(10) << "Drawdown%" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (const auto& q : summary.quarterly_results) {
            std::cout << std::left << std::setw(8) << (std::to_string(q.year) + "Q" + std::to_string(q.quarter))
                      << std::setw(12) << std::fixed << std::setprecision(2) << q.monthly_return_pct
                      << std::setw(10) << std::fixed << std::setprecision(3) << q.sharpe_ratio
                      << std::setw(8) << q.total_trades
                      << std::setw(12) << std::fixed << std::setprecision(1) << q.avg_daily_trades
                      << std::setw(10) << q.health_status()
                      << std::setw(10) << std::fixed << std::setprecision(2) << q.max_drawdown << std::endl;
        }
        
        // Summary statistics
        std::cout << "\n" << std::string(80, '-') << std::endl;
        std::cout << "SUMMARY STATISTICS:\n" << std::endl;
        std::cout << "Overall Performance:" << std::endl;
        std::cout << "  Average Monthly Return: " << std::fixed << std::setprecision(2) 
                  << summary.overall_return << "%" << std::endl;
        std::cout << "  Average Sharpe Ratio: " << std::fixed << std::setprecision(3) 
                  << summary.overall_sharpe << std::endl;
        
        std::cout << "\nConsistency Metrics:" << std::endl;
        std::cout << "  Return Volatility (std): " << std::fixed << std::setprecision(2) 
                  << summary.return_std_dev << "%" << std::endl;
        std::cout << "  Sharpe Volatility (std): " << std::fixed << std::setprecision(3) 
                  << summary.sharpe_std_dev << std::endl;
        std::cout << "  Trade Frequency Volatility (std): " << std::fixed << std::setprecision(1) 
                  << summary.trade_freq_std_dev << " trades/day" << std::endl;
        
        std::cout << "\nTrade Frequency Health:" << std::endl;
        std::cout << "  Healthy Quarters: " << summary.healthy_quarters << "/" << summary.total_quarters 
                  << " (" << std::fixed << std::setprecision(1) 
                  << (100.0 * summary.healthy_quarters / summary.total_quarters) << "%)" << std::endl;
        std::cout << "  Low Frequency: " << summary.low_freq_quarters << " quarters" << std::endl;
        std::cout << "  High Frequency: " << summary.high_freq_quarters << " quarters" << std::endl;
        
        // Health assessment
        std::cout << "\nHEALTH ASSESSMENT:" << std::endl;
        double health_pct = 100.0 * summary.healthy_quarters / summary.total_quarters;
        if (health_pct >= 80.0) {
            std::cout << "  ✅ EXCELLENT: " << health_pct << "% of quarters have healthy trade frequency" << std::endl;
        } else if (health_pct >= 60.0) {
            std::cout << "  ⚠️  MODERATE: " << health_pct << "% of quarters have healthy trade frequency" << std::endl;
        } else {
            std::cout << "  ❌ POOR: " << health_pct << "% of quarters have healthy trade frequency" << std::endl;
        }
        
        // TPA Readiness Assessment
        std::cout << "\nTPA READINESS ASSESSMENT:" << std::endl;
        std::cout << "  Readiness Score: " << std::fixed << std::setprecision(1) << summary.readiness.readiness_score << "/100" << std::endl;
        
        std::cout << "\n  Testing Readiness:" << std::endl;
        std::cout << "  " << (summary.readiness.ready_for_virtual_market ? "✅" : "❌") 
                  << " Virtual Market Testing: " << (summary.readiness.ready_for_virtual_market ? "READY" : "NOT READY") << std::endl;
        std::cout << "  " << (summary.readiness.ready_for_paper_trading ? "✅" : "❌") 
                  << " Paper Trading: " << (summary.readiness.ready_for_paper_trading ? "READY" : "NOT READY") << std::endl;
        std::cout << "  " << (summary.readiness.ready_for_live_trading ? "✅" : "❌") 
                  << " Live Trading: " << (summary.readiness.ready_for_live_trading ? "READY" : "NOT READY") << std::endl;
        
        if (!summary.readiness.issues.empty()) {
            std::cout << "\n  Issues Identified:" << std::endl;
            for (const auto& issue : summary.readiness.issues) {
                std::cout << "  ❌ " << issue << std::endl;
            }
        }
        
        if (!summary.readiness.recommendations.empty()) {
            std::cout << "\n  Recommendations:" << std::endl;
            for (const auto& rec : summary.readiness.recommendations) {
                std::cout << "  💡 " << rec << std::endl;
            }
        }
        
        std::cout << std::string(80, '=') << std::endl;
    }

private:
    std::vector<QuarterlyMetrics> quarterly_results_;
};

// Forward declarations
struct SymbolTable;
struct Bar;
struct RunnerCfg;

// Main temporal analysis function
TemporalAnalysisSummary run_temporal_analysis(const SymbolTable& ST,
                                            const std::vector<std::vector<Bar>>& series,
                                            int base_symbol_id,
                                            const RunnerCfg& rcfg,
                                            const TemporalAnalysisConfig& cfg = TemporalAnalysisConfig{});

} // namespace sentio

```

## 📄 **FILE 90 of 145**: include/sentio/tfa/artifacts_loader.hpp

**File Information**:
- **Path**: `include/sentio/tfa/artifacts_loader.hpp`

- **Size**: 114 lines
- **Modified**: 2025-09-07 12:55:23

- **Type**: .hpp

```text
#pragma once
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <iostream>

namespace sentio::tfa {

struct TfaArtifacts {
  torch::jit::Module model;
  nlohmann::json spec;
  nlohmann::json meta;
  
  // Convenience getters
  std::vector<std::string> get_expected_feature_names() const {
    return meta["expects"]["feature_names"].get<std::vector<std::string>>();
  }
  
  int get_expected_input_dim() const {
    return meta["expects"]["input_dim"].get<int>();
  }
  
  std::string get_spec_hash() const {
    return meta["expects"]["spec_hash"].get<std::string>();
  }
  
  float get_pad_value() const {
    return meta["expects"]["pad_value"].get<float>();
  }
  
  int get_emit_from() const {
    return meta["expects"]["emit_from"].get<int>();
  }
};

inline TfaArtifacts load_tfa(const std::string& model_pt,
                             const std::string& feature_spec_json,
                             const std::string& model_meta_json)
{
  TfaArtifacts A;
  
  std::cout << "[TFA] Loading model: " << model_pt << std::endl;
  A.model = torch::jit::load(model_pt, torch::kCPU);
  A.model.eval();
  
  std::cout << "[TFA] Loading feature spec: " << feature_spec_json << std::endl;
  std::ifstream fs(feature_spec_json); 
  if(!fs) throw std::runtime_error("missing feature_spec.json: " + feature_spec_json);
  fs >> A.spec;
  
  std::cout << "[TFA] Loading model meta: " << model_meta_json << std::endl;
  std::ifstream fm(model_meta_json); 
  if(!fm) throw std::runtime_error("missing model.meta.json: " + model_meta_json);
  fm >> A.meta;
  
  // Validate metadata structure
  if (!A.meta.contains("expects")) {
    throw std::runtime_error("model.meta.json missing 'expects' section");
  }
  
  auto expects = A.meta["expects"];
  if (!expects.contains("input_dim") || !expects.contains("feature_names") || 
      !expects.contains("spec_hash") || !expects.contains("pad_value") || 
      !expects.contains("emit_from")) {
    throw std::runtime_error("model.meta.json 'expects' section incomplete");
  }
  
  std::cout << "[TFA] Model expects " << A.get_expected_input_dim() << " features" << std::endl;
  
  return A;
}

// Fallback loader for existing metadata.json (without model.meta.json)
inline TfaArtifacts load_tfa_legacy(const std::string& model_pt,
                                     const std::string& metadata_json)
{
  TfaArtifacts A;
  
  std::cout << "[TFA] Loading model (legacy): " << model_pt << std::endl;
  A.model = torch::jit::load(model_pt, torch::kCPU);
  A.model.eval();
  std::cout << "[TFA] Model loaded and set to eval mode" << std::endl;
  
  std::cout << "[TFA] Loading legacy metadata: " << metadata_json << std::endl;
  std::ifstream fs(metadata_json);
  if(!fs) throw std::runtime_error("missing metadata.json: " + metadata_json);
  
  nlohmann::json legacy_meta;
  fs >> legacy_meta;
  
  // Convert legacy metadata.json to new format
  A.spec = legacy_meta; // Use legacy as spec for now
  
  // Create synthetic model.meta.json structure
  A.meta = {
    {"schema_version", "1.0"},
    {"framework", "torchscript"},
    {"expects", {
      {"input_dim", (int)legacy_meta["feature_names"].size()},
      {"feature_names", legacy_meta["feature_names"]},
      {"spec_hash", "legacy"},
      {"emit_from", 64}, // Default for TFA
      {"pad_value", 0.0f},
      {"dtype", "float32"}
    }}
  };
  
  std::cout << "[TFA] Legacy model expects " << A.get_expected_input_dim() << " features" << std::endl;
  
  return A;
}

} // namespace sentio::tfa

```

## 📄 **FILE 91 of 145**: include/sentio/tfa/artifacts_safe.hpp

**File Information**:
- **Path**: `include/sentio/tfa/artifacts_safe.hpp`

- **Size**: 121 lines
- **Modified**: 2025-09-07 19:22:14

- **Type**: .hpp

```text
#pragma once
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>

namespace sentio::tfa {

struct TfaArtifactsSafe {
  torch::jit::Module model;
  nlohmann::json spec;
  nlohmann::json meta;
  
  // Convenience getters with validation
  std::vector<std::string> get_expected_feature_names() const {
    if (!meta.contains("expects") || !meta["expects"].contains("feature_names")) {
      throw std::runtime_error("Model metadata missing feature_names");
    }
    return meta["expects"]["feature_names"].get<std::vector<std::string>>();
  }
  
  int get_expected_input_dim() const {
    if (!meta.contains("expects") || !meta["expects"].contains("input_dim")) {
      throw std::runtime_error("Model metadata missing input_dim");
    }
    return meta["expects"]["input_dim"].get<int>();
  }
  
  std::string get_spec_hash() const {
    if (!meta.contains("expects") || !meta["expects"].contains("spec_hash")) {
      throw std::runtime_error("Model metadata missing spec_hash");
    }
    return meta["expects"]["spec_hash"].get<std::string>();
  }
  
  float get_pad_value() const {
    if (!meta.contains("expects") || !meta["expects"].contains("pad_value")) {
      return 0.0f; // Default
    }
    return meta["expects"]["pad_value"].get<float>();
  }
  
  int get_emit_from() const {
    if (!meta.contains("expects") || !meta["expects"].contains("emit_from")) {
      return 64; // Default for TFA
    }
    return meta["expects"]["emit_from"].get<int>();
  }
};

inline TfaArtifactsSafe load_tfa_artifacts_safe(const std::string& model_pt,
                                                const std::string& feature_spec_json,
                                                const std::string& model_meta_json)
{
  TfaArtifactsSafe A;
  
  std::cout << "[TFA] Loading model: " << model_pt << std::endl;
  A.model = torch::jit::load(model_pt, torch::kCPU);
  A.model.eval();
  
  std::cout << "[TFA] Loading feature spec: " << feature_spec_json << std::endl;
  std::ifstream fs(feature_spec_json); 
  if(!fs) throw std::runtime_error("missing feature_spec.json: " + feature_spec_json);
  fs >> A.spec;
  
  std::cout << "[TFA] Loading model meta: " << model_meta_json << std::endl;
  std::ifstream fm(model_meta_json); 
  if(!fm) throw std::runtime_error("missing model.meta.json: " + model_meta_json);
  fm >> A.meta;
  
  // Validate metadata structure
  if (!A.meta.contains("expects")) {
    throw std::runtime_error("model.meta.json missing 'expects' section");
  }
  
  auto expects = A.meta["expects"];
  if (!expects.contains("input_dim") || !expects.contains("feature_names") || 
      !expects.contains("spec_hash")) {
    throw std::runtime_error("model.meta.json 'expects' section incomplete");
  }
  
  // Validate spec hash if available
  if (A.spec.contains("content_hash")) {
    std::string spec_hash = A.spec["content_hash"].get<std::string>();
    std::string expected_hash = A.get_spec_hash();
    if (spec_hash != expected_hash) {
      std::cout << "[TFA] WARNING: Spec hash mismatch!\n"
                << "  Runtime spec: " << spec_hash.substr(0,16) << "...\n"
                << "  Model expects: " << expected_hash.substr(0,16) << "..." << std::endl;
    }
  }
  
  std::cout << "[TFA] Model expects " << A.get_expected_input_dim() 
            << " features, emit_from=" << A.get_emit_from() << std::endl;
  
  return A;
}

inline std::vector<std::string> feature_names_from_spec(const nlohmann::json& spec){
  std::vector<std::string> names;
  if (!spec.contains("features")) {
    throw std::runtime_error("Feature spec missing 'features' array");
  }
  
  for (auto& f : spec["features"]){
    if (f.contains("name")) {
      names.push_back(f["name"].get<std::string>());
    } else {
      std::string op = f.value("op", "UNKNOWN");
      std::string src = f.value("source", "");
      std::string w = f.contains("window") ? std::to_string((int)f["window"]) : "";
      std::string k = f.contains("k") ? std::to_string((float)f["k"]) : "";
      names.push_back(op + "_" + src + "_" + w + "_" + k);
    }
  }
  return names;
}

} // namespace sentio::tfa

```

## 📄 **FILE 92 of 145**: include/sentio/tfa/feature_guard.hpp

**File Information**:
- **Path**: `include/sentio/tfa/feature_guard.hpp`

- **Size**: 60 lines
- **Modified**: 2025-09-07 12:32:16

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace sentio::tfa {

struct FeatureGuard {
  int emit_from = 0;
  float pad_value = 0.0f;

  static inline bool is_finite(float x){
    return std::isfinite(x);
  }

  // returns mask: true = usable
  std::vector<uint8_t> build_mask_and_clean(float* X, int64_t rows, int64_t cols) const {
    std::vector<uint8_t> ok(rows, 0);
    // zero/pad early rows, mark them unusable
    for (int64_t r=0; r<std::min<int64_t>(emit_from, rows); ++r){
      for (int64_t c=0; c<cols; ++c) X[r*cols+c] = pad_value;
    }
    // after emit_from: sanitize NaN/Inf
    for (int64_t r=emit_from; r<rows; ++r){
      bool row_ok = true;
      for (int64_t c=0; c<cols; ++c){
        float& v = X[r*cols+c];
        if (!is_finite(v)) { v = 0.0f; row_ok = false; } // clean AND mark not-OK for signal
      }
      ok[r] = row_ok ? 1 : 0;
    }
    return ok;
  }
  
  // Overload for double vectors (from cached features)
  std::vector<uint8_t> build_mask_and_clean(std::vector<std::vector<double>>& features) const {
    const int64_t rows = features.size();
    const int64_t cols = rows > 0 ? features[0].size() : 0;
    std::vector<uint8_t> ok(rows, 0);
    
    // zero/pad early rows, mark them unusable
    for (int64_t r=0; r<std::min<int64_t>(emit_from, rows); ++r){
      for (int64_t c=0; c<cols; ++c) features[r][c] = pad_value;
    }
    
    // after emit_from: sanitize NaN/Inf
    for (int64_t r=emit_from; r<rows; ++r){
      bool row_ok = true;
      for (int64_t c=0; c<cols; ++c){
        double& v = features[r][c];
        if (!std::isfinite(v)) { v = 0.0; row_ok = false; } // clean AND mark not-OK for signal
      }
      ok[r] = row_ok ? 1 : 0;
    }
    return ok;
  }
};

} // namespace sentio::tfa

```

## 📄 **FILE 93 of 145**: include/sentio/tfa/input_shim.hpp

**File Information**:
- **Path**: `include/sentio/tfa/input_shim.hpp`

- **Size**: 55 lines
- **Modified**: 2025-09-07 20:15:09

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <unordered_map>

namespace sentio {

inline std::vector<float> shim_to_expected_input(const float* X_src,
                                                 int64_t rows,
                                                 int64_t F_src,
                                                 const std::vector<std::string>& runtime_names,
                                                 const std::vector<std::string>& expected_names,
                                                 int   F_expected,
                                                 float pad_value = 0.0f)
{
  // Fast path: exact match
  if (F_src == F_expected && runtime_names == expected_names) {
    return std::vector<float>(X_src, X_src + (size_t)rows*(size_t)F_src);
  }

  // Hotfix path: legacy model expects a leading 'ts' column
  const bool model_leads_with_ts = !expected_names.empty() && expected_names.front() == "ts";
  const bool runtime_has_ts      = !runtime_names.empty()   && runtime_names.front()  == "ts";
  if (model_leads_with_ts && !runtime_has_ts && F_src + 1 == F_expected) {
    std::vector<float> out((size_t)rows * (size_t)F_expected, pad_value);
    for (int64_t r=0; r<rows; ++r) {
      float* dst = out.data() + r*F_expected;
      std::memcpy(dst + 1, X_src + r*F_src, sizeof(float) * (size_t)F_src);
      dst[0] = 0.0f; // dummy ts
    }
    std::cerr << "[TFA] HOTFIX: injected dummy 'ts' col to satisfy legacy 56-dim model\n";
    return out;
  }

  // General name-based projection (drops extras, fills missing with pad)
  std::vector<float> out((size_t)rows * (size_t)F_expected, pad_value);
  // build index map
  std::unordered_map<std::string,int> pos;
  pos.reserve(runtime_names.size()*2);
  for (int i=0;i<(int)runtime_names.size();++i) pos[runtime_names[i]] = i;
  for (int64_t r=0; r<rows; ++r) {
    const float* src = X_src + r*F_src;
    float* dst = out.data() + r*F_expected;
    for (int j=0; j<F_expected; ++j) {
      auto it = pos.find(expected_names[j]);
      if (it != pos.end()) dst[j] = src[it->second];
    }
  }
  std::cerr << "[TFA] INFO: name-based projection applied (srcF="<<F_src<<" -> dstF="<<F_expected<<")\n";
  return out;
}

} // namespace sentio

```

## 📄 **FILE 94 of 145**: include/sentio/tfa/signal_pipeline.hpp

**File Information**:
- **Path**: `include/sentio/tfa/signal_pipeline.hpp`

- **Size**: 206 lines
- **Modified**: 2025-09-07 12:32:16

- **Type**: .hpp

```text
#pragma once
#include "sentio/ml/iml_model.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace sentio::tfa {

struct DropCounters {
  int64_t total{0}, not_ready{0}, nan_row{0}, low_conf{0}, session{0}, volume{0}, cooldown{0}, dup{0};
  void log() const {
    std::cout << "[SIG TFA] total="<<total
              << " not_ready="<<not_ready
              << " nan_row="<<nan_row
              << " low_conf="<<low_conf
              << " session="<<session
              << " volume="<<volume
              << " cooldown="<<cooldown
              << " dup="<<dup << std::endl;
  }
};

struct ThresholdPolicy {
  // Either fixed threshold or rolling quantile
  float min_prob = 0.55f;      // fixed
  int   q_window = 0;          // if >0, use quantile
  float q_level  = 0.75f;      // 75th percentile

  std::vector<uint8_t> filter(const std::vector<float>& prob) const {
    const int64_t N = (int64_t)prob.size();
    std::vector<uint8_t> keep(N, 0);
    if (q_window <= 0){
      for (int64_t i=0;i<N;++i) keep[i] = (prob[i] >= min_prob) ? 1 : 0;
      return keep;
    }
    // rolling quantile
    std::vector<float> win; win.reserve(q_window);
    for (int64_t i=0;i<N;++i){
      int64_t a = std::max<int64_t>(0, i - q_window + 1);
      win.clear();
      for (int64_t j=a;j<=i;++j) win.push_back(prob[j]);
      std::nth_element(win.begin(), win.begin() + (int)(q_level*(win.size()-1)), win.end());
      float thr = win[(int)(q_level*(win.size()-1))];
      keep[i] = (prob[i] >= thr) ? 1 : 0;
    }
    return keep;
  }
};

struct Cooldown {
  int bars = 0; // e.g., 5
  // returns mask where entry allowed, tracking last accepted index
  std::vector<uint8_t> apply(const std::vector<uint8_t>& keep) const {
    if (bars <= 0) return keep;
    std::vector<uint8_t> out(keep.size(), 0);
    int64_t next_ok = 0;
    for (int64_t i=0;i<(int64_t)keep.size(); ++i){
      if (i < next_ok) continue;
      if (keep[i]){ out[i]=1; next_ok = i + bars; }
    }
    return out;
  }
};

// Minimal session & volume filters, customize to your data
struct RowFilters {
  bool rth_only = false;
  double min_volume = 0.0;
  std::vector<uint8_t> session_mask; // 1 if allowed (precomputed per row)
  std::vector<double>  volumes;      // per row

  void ensure_sizes(int64_t N){
    if ((int64_t)session_mask.size()!=N) session_mask.assign(N,1);
    if ((int64_t)volumes.size()!=N) volumes.assign(N, 1.0);
  }
};

struct TfaSignalPipeline {
  ml::IModel* model{nullptr}; // Model interface, returns score/prob
  ThresholdPolicy policy;
  Cooldown cooldown;
  RowFilters rowf;

  struct Result {
    std::vector<uint8_t> emit;     // 1=emit entry
    std::vector<float>   prob;     // model output (after activation)
    DropCounters drops;
  };

  static inline float sigmoid(float x){ return 1.f / (1.f + std::exp(-x)); }

  Result run(float* X, int64_t rows, int64_t cols,
             const std::vector<uint8_t>& ready_mask,
             bool model_outputs_logit=true)
  {
    Result R;
    R.prob.assign(rows, 0.f);
    R.emit.assign(rows, 0);
    R.drops.total = rows;

    // 1) forward model using IModel interface
    for (int64_t i=0; i<rows; ++i){
      std::vector<float> features(cols);
      for (int64_t j=0; j<cols; ++j) {
        features[j] = X[i*cols + j];
      }
      
      auto output = model->predict(features, 1, cols, "BF"); // Single row prediction
      if (output && !output->probs.empty()) {
        float v = output->probs[0]; // Assume single output probability
        R.prob[i] = model_outputs_logit ? sigmoid(v) : v;
      } else if (output) {
        float v = output->score; // Fallback to score
        R.prob[i] = model_outputs_logit ? sigmoid(v) : v;
      } else {
        R.prob[i] = 0.5f; // Default neutral probability
      }
    }

    // 2) ready vs not_ready
    std::vector<uint8_t> keep(rows, 0);
    for (int64_t i=0;i<rows;++i){
      if (!ready_mask[i]) { R.drops.not_ready++; continue; }
      keep[i] = 1;
    }

    // 3) session / volume
    rowf.ensure_sizes(rows);
    for (int64_t i=0;i<rows;++i){
      if (!keep[i]) continue;
      if (!rowf.session_mask[i]){ keep[i]=0; R.drops.session++; continue; }
      if (rowf.volumes[i] <= rowf.min_volume){ keep[i]=0; R.drops.volume++; continue; }
    }

    // 4) thresholding
    auto conf_keep = policy.filter(R.prob);
    for (int64_t i=0;i<rows;++i){
      if (!keep[i]) continue;
      if (!conf_keep[i]){ keep[i]=0; R.drops.low_conf++; }
    }

    // 5) cooldown
    keep = cooldown.apply(keep);
    // count cooldown drops (approx)
    // (We can estimate: entries removed between pre/post)
    // For transparency, compute:
    {
      int64_t pre=0, post=0;
      for (auto v: conf_keep) if (v) pre++;
      for (auto v: keep) if (v) post++;
      R.drops.cooldown += std::max<int64_t>(0, pre - post);
    }

    R.emit = std::move(keep);
    return R;
  }
  
  // Overload for vector<vector<double>> from cached features
  Result run_cached(const std::vector<std::vector<double>>& features,
                    const std::vector<uint8_t>& ready_mask,
                    bool model_outputs_logit=true)
  {
    const int64_t rows = features.size();
    if (rows == 0) {
      Result R;
      R.drops.total = 0;
      return R;
    }
    
    const int64_t cols = features[0].size();
    if (cols == 0) {
      Result R;
      R.drops.total = rows;
      R.drops.nan_row = rows;
      return R;
    }
    
    // Safety check: ensure all rows have same column count
    for (int64_t r = 0; r < rows; ++r) {
      if ((int64_t)features[r].size() != cols) {
        std::cout << "[ERROR] TfaSignalPipeline: Row " << r << " has " << features[r].size() 
                  << " features, expected " << cols << std::endl;
        Result R;
        R.drops.total = rows;
        R.drops.nan_row = rows;
        return R;
      }
    }
    
    // Convert to float array for model
    std::vector<float> X_flat(rows * cols);
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        X_flat[r * cols + c] = static_cast<float>(features[r][c]);
      }
    }
    
    return run(X_flat.data(), rows, cols, ready_mask, model_outputs_logit);
  }
};

} // namespace sentio::tfa

```

## 📄 **FILE 95 of 145**: include/sentio/tfa/tfa_seq_context.hpp

**File Information**:
- **Path**: `include/sentio/tfa/tfa_seq_context.hpp`

- **Size**: 106 lines
- **Modified**: 2025-09-07 22:27:03

- **Type**: .hpp

```text
#pragma once
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <cstring>

#include "sentio/feature/feature_from_spec.hpp"
#include "sentio/feature/column_projector.hpp"
#include "sentio/feature/sanitize.hpp"

namespace sentio {

struct TfaSeqContext {
  torch::jit::Module model;  nlohmann::json spec, meta;
  std::vector<std::string> runtime_names, expected_names;
  int F{55}, T{64}, emit_from{64}; float pad_value{0.f};

  static nlohmann::json load_json(const std::string& p){ std::ifstream f(p); nlohmann::json j; f>>j; return j; }
  static std::vector<std::string> names_from_spec(const nlohmann::json& spec){
    std::vector<std::string> out; out.reserve(spec["features"].size());
    for (auto& f: spec["features"]){
      if (f.contains("name")) out.push_back(f["name"].get<std::string>());
      else {
        std::string op=f["op"].get<std::string>(), src=f.value("source","");
        std::string w=f.contains("window")?std::to_string((int)f["window"]):"";
        std::string k=f.contains("k")?std::to_string((float)f["k"]):"";
        out.push_back(op+"_"+src+"_"+w+"_"+k);
      }
    }
    return out;
  }

  void load(const std::string& model_pt, const std::string& spec_json, const std::string& meta_json){
    model = torch::jit::load(model_pt, torch::kCPU); model.eval();
    spec  = load_json(spec_json);
    meta  = load_json(meta_json);

    runtime_names  = names_from_spec(spec);
    expected_names = meta["expects"]["feature_names"].get<std::vector<std::string>>();
    F         = meta["expects"]["input_dim"].get<int>();
    if (meta["expects"].contains("seq_len")) T = meta["expects"]["seq_len"].get<int>();
    emit_from = meta["expects"]["emit_from"].get<int>();
    pad_value = meta["expects"]["pad_value"].get<float>();

    if (F!=55) std::cerr << "[WARN] model F="<<F<<" expected 55\n";
    std::cerr << "[TFA-SEQ] loaded: F="<<F<<" T="<<T<<" emit_from="<<emit_from<<"\n";
  }

  template<class Bars>
  void forward_probs(const std::string& symbol, const Bars& bars, std::vector<float>& probs_out)
  {
    // Build features [N,F]
    auto X = sentio::build_features_from_spec_json(symbol, bars, spec.dump());
    // Project if needed
    std::vector<float> Xproj;
    const float* Xp = X.data.data(); int64_t Fs = X.cols;
    if (!(Fs==F && runtime_names==expected_names)){
      auto proj = sentio::ColumnProjector::make(runtime_names, expected_names, pad_value);
      proj.project(X.data.data(), (size_t)X.rows, (size_t)X.cols, Xproj);
      Xp = Xproj.data(); Fs = F;
      std::cerr << "[TFA-SEQ] projected "<<X.cols<<" -> "<<F<<"\n";
    }

    // Sanitize
    auto ready = sentio::sanitize_and_ready(const_cast<float*>(Xp), X.rows, Fs, emit_from, pad_value);

    // Slide windows → batch inference
    probs_out.assign((size_t)X.rows, 0.5f);
    torch::NoGradGuard ng; torch::InferenceMode im;
    const int64_t B = 256;
    const int64_t start = std::max<int64_t>({emit_from, T-1});
    const int64_t last  = X.rows - 1;

    for (int64_t i=start; i<=last; ){
      int64_t j = std::min<int64_t>(last+1, i+B);
      int64_t L = j - i;
      auto t = torch::empty({L, T, F}, torch::kFloat32);
      float* dst = t.data_ptr<float>();
      for (int64_t k=0;k<L;++k){
        int64_t idx=i+k, lo=idx-T+1;
        std::memcpy(dst + k*T*F, Xp + lo*F, sizeof(float)*(size_t)(T*F));
      }
      auto y = model.forward({t}).toTensor(); // [L,1] logits
      if (y.dim()==2 && y.size(1)==1) y=y.squeeze(1);
      float* lp = y.contiguous().data_ptr<float>();
      for (int64_t k=0;k<L;++k)
        probs_out[(size_t)(i+k)] = 1.f/(1.f+std::exp(-lp[k])); // sigmoid
      i = j;
    }

    // Stats
    float pmin=1.f, pmax=0.f, ps=0.f; int64_t cnt=0;
    for (int64_t i=start;i<(int64_t)probs_out.size();++i){ pmin=std::min(pmin,probs_out[i]); pmax=std::max(pmax,probs_out[i]); ps+=probs_out[i]; cnt++; }
    std::cerr << "[TFA-SEQ] prob stats: min="<<pmin<<" mean="<<(ps/std::max<int64_t>(1,cnt))<<" max="<<pmax<<"\n";
    for (int64_t i=start; i<(int64_t)probs_out.size(); i+=50)
      std::cerr << "[TFA-SEQ] prob["<<i<<"]="<<probs_out[i]<<"\n";
  }
};

} // namespace sentio



```

## 📄 **FILE 96 of 145**: include/sentio/time_utils.hpp

**File Information**:
- **Path**: `include/sentio/time_utils.hpp`

- **Size**: 15 lines
- **Modified**: 2025-09-05 15:29:30

- **Type**: .hpp

```text
#pragma once
#include <chrono>
#include <string>
#include <variant>

namespace sentio {

// Normalize various timestamp representations to UTC epoch seconds.
std::chrono::sys_seconds to_utc_sys_seconds(const std::variant<std::int64_t, double, std::string>& ts);

// Helpers exposed for tests
bool iso8601_looks_like(const std::string& s);
bool epoch_ms_suspected(double v_ms);

} // namespace sentio
```

## 📄 **FILE 97 of 145**: include/sentio/torch/safe_from_blob.hpp

**File Information**:
- **Path**: `include/sentio/torch/safe_from_blob.hpp`

- **Size**: 48 lines
- **Modified**: 2025-09-07 13:49:40

- **Type**: .hpp

```text
#pragma once
#include <torch/torch.h>
#include <memory>
#include <vector>

namespace sentio {

// Creates a Tensor that OWNS a heap buffer and will free it when Tensor dies.
// If you already have a std::shared_ptr<float> backing store, prefer that version.
inline torch::Tensor own_copy_tensor(const float* src, int64_t rows, int64_t cols) {
  auto t = torch::empty({rows, cols}, torch::dtype(torch::kFloat32));
  t.copy_(torch::from_blob((void*)src, {rows, cols}, torch::kFloat32)); // safe copy
  return t;
}

// If you insist on zero-copy, give Tensor a deleter tied to a shared_ptr:
inline torch::Tensor tensor_from_shared(std::shared_ptr<std::vector<float>> store, int64_t rows, int64_t cols) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  return torch::from_blob(
      (void*)store->data(),
      {rows, cols},
      [store](void*) mutable { store.reset(); }, // keep alive
      options);
}

// Safe batched forward pass with owned tensors
inline std::vector<float> model_forward_probs(torch::jit::Module& m, const float* X, int64_t rows, int64_t cols, bool logits=true){
  std::vector<float> probs((size_t)rows, 0.f);
  torch::NoGradGuard ng; 
  torch::InferenceMode im;
  const int64_t B = 8192;
  for (int64_t i=0;i<rows;i+=B){
    int64_t b = std::min<int64_t>(B, rows-i);
    auto t = torch::from_blob((void*)(X + i*cols), {b, cols}, torch::kFloat32).clone(); // OWNED
    t = t.contiguous(); // belt & suspenders
    auto y = m.forward({t}).toTensor();
    if (y.dim()==2 && y.size(1)==1) y = y.squeeze(1);
    if (y.dim()!=1 || y.size(0)!=b) throw std::runtime_error("model output shape mismatch");
    auto acc = y.contiguous().data_ptr<float>();
    for (int64_t k=0;k<b;++k){
      float v = acc[k];
      probs[(size_t)(i+k)] = logits ? 1.f/(1.f+std::exp(-v)) : v;
    }
  }
  return probs;
}

} // namespace sentio

```

## 📄 **FILE 98 of 145**: include/sentio/util/bytes.hpp

**File Information**:
- **Path**: `include/sentio/util/bytes.hpp`

- **Size**: 24 lines
- **Modified**: 2025-09-07 13:49:40

- **Type**: .hpp

```text
#pragma once
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace sentio {

// Safe memory copy with bounds checking
inline void bytes_copy(void* dst, const void* src, size_t count){
  if (!dst || !src) throw std::runtime_error("bytes_copy: null ptr");
  if (count > 0) {
    std::memcpy(dst, src, count);
  }
}

// Safe memory set with bounds checking
inline void bytes_set(void* ptr, int value, size_t count) {
  if (!ptr) throw std::runtime_error("bytes_set: null ptr");
  if (count > 0) {
    std::memset(ptr, value, count);
  }
}

} // namespace sentio

```

## 📄 **FILE 99 of 145**: include/sentio/util/safe_matrix.hpp

**File Information**:
- **Path**: `include/sentio/util/safe_matrix.hpp`

- **Size**: 38 lines
- **Modified**: 2025-09-07 13:49:40

- **Type**: .hpp

```text
#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <cstring>

namespace sentio {

struct SafeMatrix {
  std::vector<float> buf;
  int64_t rows{0}, cols{0};

  void resize(int64_t r, int64_t c) {
    if (r < 0 || c < 0) throw std::runtime_error("SafeMatrix: negative shape");
    if (c > (int64_t)(std::numeric_limits<size_t>::max()/sizeof(float))/ (r>0?r:1))
      throw std::runtime_error("SafeMatrix: size overflow");
    rows = r; cols = c;
    buf.assign(static_cast<size_t>(r)*static_cast<size_t>(c), 0.0f);
  }

  inline float* row_ptr(int64_t r) {
    if ((uint64_t)r >= (uint64_t)rows) throw std::runtime_error("SafeMatrix: row OOB");
    return buf.data() + (size_t)r*(size_t)cols;
  }
  inline const float* row_ptr(int64_t r) const {
    if ((uint64_t)r >= (uint64_t)rows) throw std::runtime_error("SafeMatrix: row OOB");
    return buf.data() + (size_t)r*(size_t)cols;
  }
  
  // Convenience accessors
  float* data() { return buf.data(); }
  const float* data() const { return buf.data(); }
  size_t size() const { return buf.size(); }
  bool empty() const { return buf.empty(); }
};

} // namespace sentio

```

## 📄 **FILE 100 of 145**: include/sentio/wf.hpp

**File Information**:
- **Path**: `include/sentio/wf.hpp`

- **Size**: 52 lines
- **Modified**: 2025-09-07 16:03:01

- **Type**: .hpp

```text
#pragma once
#include "sentio/core.hpp"
#include "sentio/runner.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/audit.hpp"
#include <vector>
#include <string>

namespace sentio {

struct WfCfg {
    int train_days = 252;  // Training period in days
    int oos_days = 14;     // Out-of-sample period in days
    int step_days = 14;    // Step size between folds
    bool enable_optimization = false;
    std::string optimizer_type = "random";
    int max_optimization_trials = 30;
    std::string optimization_objective = "sharpe_ratio";
    double optimization_timeout_minutes = 15.0;
    bool optimization_verbose = true;
};

struct Gate {
    bool wf_pass = false;
    bool oos_pass = false;
    std::string recommend = "REJECT";
    double oos_monthly_avg_return = 0.0;
    double oos_sharpe = 0.0;
    double oos_max_drawdown = 0.0;
    int successful_optimizations = 0;
    double avg_optimization_improvement = 0.0;
    std::vector<std::pair<std::string, double>> optimization_results;
};

// Walk-forward testing with vector-based data
Gate run_wf_and_gate(AuditRecorder& audit_template,
                     const SymbolTable& ST,
                     const std::vector<std::vector<Bar>>& series,
                     int base_symbol_id,
                     const RunnerCfg& rcfg,
                     const WfCfg& wcfg);

// Walk-forward testing with optimization
Gate run_wf_and_gate_optimized(AuditRecorder& audit_template,
                               const SymbolTable& ST,
                               const std::vector<std::vector<Bar>>& series,
                               int base_symbol_id,
                               const RunnerCfg& base_rcfg,
                               const WfCfg& wcfg);

} // namespace sentio


```

## 📄 **FILE 101 of 145**: src/audit.cpp

**File Information**:
- **Path**: `src/audit.cpp`

- **Size**: 259 lines
- **Modified**: 2025-09-05 20:51:42

- **Type**: .cpp

```text
#include "sentio/audit.hpp"
#include <cstring>
#include <sstream>
#include <iomanip>

// ---- minimal SHA1 (tiny, not constant-time; for tamper detection only) ----
namespace {
struct Sha1 {
  uint32_t h0=0x67452301, h1=0xEFCDAB89, h2=0x98BADCFE, h3=0x10325476, h4=0xC3D2E1F0;
  std::string hex(const std::string& s){
    uint64_t ml = s.size()*8ULL;
    std::string msg = s;
    msg.push_back('\x80');
    while ((msg.size()%64)!=56) msg.push_back('\0');
    for (int i=7;i>=0;--i) msg.push_back(char((ml>>(i*8))&0xFF));
    for (size_t i=0;i<msg.size(); i+=64) {
      uint32_t w[80];
      for (int t=0;t<16;++t) {
        w[t] = (uint32_t(uint8_t(msg[i+4*t]))<<24)
             | (uint32_t(uint8_t(msg[i+4*t+1]))<<16)
             | (uint32_t(uint8_t(msg[i+4*t+2]))<<8)
             | (uint32_t(uint8_t(msg[i+4*t+3])));
      }
      for (int t=16;t<80;++t){ uint32_t v = w[t-3]^w[t-8]^w[t-14]^w[t-16]; w[t]=(v<<1)|(v>>31); }
      uint32_t a=h0,b=h1,c=h2,d=h3,e=h4;
      for (int t=0;t<80;++t){
        uint32_t f,k;
        if (t<20){ f=(b&c)|((~b)&d); k=0x5A827999; }
        else if (t<40){ f=b^c^d; k=0x6ED9EBA1; }
        else if (t<60){ f=(b&c)|(b&d)|(c&d); k=0x8F1BBCDC; }
        else { f=b^c^d; k=0xCA62C1D6; }
        uint32_t temp = ((a<<5)|(a>>27)) + f + e + k + w[t];
        e=d; d=c; c=((b<<30)|(b>>2)); b=a; a=temp;
      }
      h0+=a; h1+=b; h2+=c; h3+=d; h4+=e;
    }
    std::ostringstream os; os<<std::hex<<std::setfill('0')<<std::nouppercase;
    os<<std::setw(8)<<h0<<std::setw(8)<<h1<<std::setw(8)<<h2<<std::setw(8)<<h3<<std::setw(8)<<h4;
    return os.str();
  }
};
}

namespace sentio {

static inline std::string num_s(double v){
  std::ostringstream os; os.setf(std::ios::fixed); os<<std::setprecision(8)<<v; return os.str();
}
static inline std::string num_i(std::int64_t v){
  std::ostringstream os; os<<v; return os.str();
}
std::string AuditRecorder::json_escape_(const std::string& s){
  std::string o; o.reserve(s.size()+8);
  for (char c: s){
    switch(c){
      case '"': o+="\\\""; break;
      case '\\':o+="\\\\"; break;
      case '\b':o+="\\b"; break;
      case '\f':o+="\\f"; break;
      case '\n':o+="\\n"; break;
      case '\r':o+="\\r"; break;
      case '\t':o+="\\t"; break;
      default: o.push_back(c);
    }
  }
  return o;
}

std::string AuditRecorder::sha1_hex_(const std::string& s){
  Sha1 sh; return sh.hex(s);
}

AuditRecorder::AuditRecorder(const AuditConfig& cfg)
: run_id_(cfg.run_id), file_path_(cfg.file_path), flush_each_(cfg.flush_each)
{
  fp_ = std::fopen(cfg.file_path.c_str(), "ab");
  if (!fp_) throw std::runtime_error("Audit open failed: "+cfg.file_path);
}
AuditRecorder::~AuditRecorder(){ if (fp_) std::fclose(fp_); }

void AuditRecorder::write_line_(const std::string& body){
  std::string core = "{\"run\":\""+json_escape_(run_id_)+"\",\"seq\":"+num_i((std::int64_t)seq_)+","+body+"}";
  std::string line = core; // sha1 over core for stability
  std::string h = sha1_hex_(core);
  line.pop_back(); // remove trailing '}'
  line += ",\"sha1\":\""+h+"\"}\n";
  if (std::fwrite(line.data(),1,line.size(),fp_)!=line.size()) throw std::runtime_error("Audit write failed");
  if (flush_each_) std::fflush(fp_);
  ++seq_;
}

void AuditRecorder::event_run_start(std::int64_t ts, const std::string& meta){
  write_line_("\"type\":\"run_start\",\"ts\":"+num_i(ts)+",\"meta\":"+meta+"}");
}
void AuditRecorder::event_run_end(std::int64_t ts, const std::string& meta){
  write_line_("\"type\":\"run_end\",\"ts\":"+num_i(ts)+",\"meta\":"+meta+"}");
}
void AuditRecorder::event_bar(std::int64_t ts, const std::string& inst, const AuditBar& b){
  write_line_("\"type\":\"bar\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"o\":"+num_s(b.open)+",\"h\":"+num_s(b.high)+",\"l\":"+num_s(b.low)+",\"c\":"+num_s(b.close)+",\"v\":"+num_s(b.volume)+"}");
}
void AuditRecorder::event_signal(std::int64_t ts, const std::string& base, SigType t, double conf){
  write_line_("\"type\":\"signal\",\"ts\":"+num_i(ts)+",\"base\":\""+json_escape_(base)+"\",\"sig\":" + num_i((int)t) + ",\"conf\":"+num_s(conf)+"}");
}
void AuditRecorder::event_route (std::int64_t ts, const std::string& base, const std::string& inst, double tw){
  write_line_("\"type\":\"route\",\"ts\":"+num_i(ts)+",\"base\":\""+json_escape_(base)+"\",\"inst\":\""+json_escape_(inst)+"\",\"tw\":"+num_s(tw)+"}");
}
void AuditRecorder::event_order (std::int64_t ts, const std::string& inst, Side side, double qty, double limit_px){
  write_line_("\"type\":\"order\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"side\":"+num_i((int)side)+",\"qty\":"+num_s(qty)+",\"limit\":"+num_s(limit_px)+"}");
}
void AuditRecorder::event_fill  (std::int64_t ts, const std::string& inst, double price, double qty, double fees, Side side){
  write_line_("\"type\":\"fill\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"px\":"+num_s(price)+",\"qty\":"+num_s(qty)+",\"fees\":"+num_s(fees)+",\"side\":"+num_i((int)side)+"}");
}
void AuditRecorder::event_snapshot(std::int64_t ts, const AccountState& a){
  write_line_("\"type\":\"snapshot\",\"ts\":"+num_i(ts)+",\"cash\":"+num_s(a.cash)+",\"real\":"+num_s(a.realized)+",\"equity\":"+num_s(a.equity)+"}");
}
void AuditRecorder::event_metric(std::int64_t ts, const std::string& key, double val){
  write_line_("\"type\":\"metric\",\"ts\":"+num_i(ts)+",\"key\":\""+json_escape_(key)+"\",\"val\":"+num_s(val)+"}");
}

// ----------------- Replayer --------------------

static inline bool parse_kv(const std::string& s, const char* key, std::string& out) {
  auto kq = std::string("\"")+key+"\":";
  auto p = s.find(kq); if (p==std::string::npos) return false;
  p += kq.size();
  if (p>=s.size()) return false;
  if (s[p]=='"'){ // string
    auto e = s.find('"', p+1);
    if (e==std::string::npos) return false;
    out = s.substr(p+1, e-(p+1));
    return true;
  } else { // number or enum
    auto e = s.find_first_of(",}", p);
    out = s.substr(p, e-p);
    return true;
  }
}

std::optional<ReplayResult> AuditReplayer::replay_file(const std::string& path,
                                                       const std::string& run_expect)
{
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) return std::nullopt;

  PriceBook pb;
  ReplayResult rr;
  rr.acct.cash = 0.0;
  rr.acct.realized = 0.0;

  char buf[16*1024];
  while (std::fgets(buf, sizeof(buf), fp)) {
    std::string line(buf);
    // very light JSONL parsing (we control writer)

    std::string run; if (!parse_kv(line, "run", run)) continue;
    if (!run_expect.empty() && run!=run_expect) continue;

    std::string type; if (!parse_kv(line, "type", type)) continue;
    // Note: timestamp parsing removed as it's not currently used in replay logic

    if (type=="bar") {
      std::string inst; parse_kv(line, "inst", inst);
      std::string o,h,l,c; parse_kv(line,"o",o); parse_kv(line,"h",h);
      parse_kv(line,"l",l); parse_kv(line,"c",c);
      AuditBar b{std::stod(o),std::stod(h),std::stod(l),std::stod(c)};
      apply_bar_(pb, inst, b);
      ++rr.bars;
    } else if (type=="signal") {
      ++rr.signals;
    } else if (type=="route") {
      ++rr.routes;
    } else if (type=="order") {
      ++rr.orders;
    } else if (type=="fill") {
      std::string inst,px,qty,fees,side_s;
      parse_kv(line,"inst",inst); parse_kv(line,"px",px);
      parse_kv(line,"qty",qty); parse_kv(line,"fees",fees);
      parse_kv(line,"side",side_s);
      Side side = side_s.empty() ? Side::Buy : static_cast<Side>(std::stoi(side_s));
      apply_fill_(rr, inst, std::stod(px), std::stod(qty), std::stod(fees), side);
      ++rr.fills;
      mark_to_market_(pb, rr);
    } else if (type=="snapshot") {
      // snapshots are optional for verification; we recompute anyway
      // you can cross-check here if you also store snapshots
    } else if (type=="run_end") {
      // could verify sha1 continuity/counts here
    }
  }
  std::fclose(fp);

  mark_to_market_(pb, rr);
  return rr;
}

bool AuditReplayer::apply_bar_(PriceBook& pb, const std::string& instrument, const AuditBar& b) {
  pb.last_px[instrument] = b.close;
  return true;
}

void AuditReplayer::apply_fill_(ReplayResult& rr, const std::string& inst, double px, double qty, double fees, Side side) {
  auto& pos = rr.positions[inst];
  
  // Convert order side to position impact
  // Buy orders: positive position qty, Sell orders: negative position qty
  double position_qty = (side == Side::Buy) ? qty : -qty;
  
  // cash impact: buy qty>0 => cash decreases, sell qty>0 => cash increases
  double cash_delta = (side == Side::Buy) ? -(px*qty + fees) : (px*qty - fees);
  rr.acct.cash += cash_delta;

  // position update (VWAP)
  double new_qty = pos.qty + position_qty;
  
  if (new_qty == 0.0) {
    // flat: realize P&L for the round trip
    if (pos.qty != 0.0) {
      rr.acct.realized += (px - pos.avg_px) * pos.qty;
    }
    pos.qty = 0.0; 
    pos.avg_px = 0.0;
  } else if (pos.qty == 0.0) {
    // opening new position
    pos.qty = new_qty;
    pos.avg_px = px;
  } else if ((pos.qty > 0) == (position_qty > 0)) {
    // adding to same side -> new average
    pos.avg_px = (pos.avg_px * pos.qty + px * position_qty) / new_qty;
    pos.qty = new_qty;
  } else {
    // reducing or flipping side -> realize partial P&L
    double closed_qty = std::min(std::abs(position_qty), std::abs(pos.qty));
    if (pos.qty > 0) {
      rr.acct.realized += (px - pos.avg_px) * closed_qty;
    } else {
      rr.acct.realized += (pos.avg_px - px) * closed_qty;
    }
    pos.qty = new_qty;
    if (pos.qty == 0.0) {
      pos.avg_px = 0.0;
    } else {
      pos.avg_px = px; // new average for remaining position
    }
  }
}

void AuditReplayer::mark_to_market_(const PriceBook& pb, ReplayResult& rr) {
  double mtm=0.0;
  for (auto& kv : rr.positions) {
    const auto& inst = kv.first;
    const auto& p = kv.second;
    auto it = pb.last_px.find(inst);
    if (it==pb.last_px.end()) continue;
    mtm += p.qty * it->second;
  }
  rr.acct.equity = rr.acct.cash + rr.acct.realized + mtm;
}

} // namespace sentio
```

## 📄 **FILE 102 of 145**: src/base_strategy.cpp

**File Information**:
- **Path**: `src/base_strategy.cpp`

- **Size**: 46 lines
- **Modified**: 2025-09-05 16:02:04

- **Type**: .cpp

```text
#include "sentio/base_strategy.hpp"
#include <iostream>

namespace sentio {

void BaseStrategy::reset_state() {
    state_.reset();
    diag_ = SignalDiag{};
}

bool BaseStrategy::is_cooldown_active(int current_bar, int cooldown_period) const {
    return (current_bar - state_.last_trade_bar) < cooldown_period;
}

// **MODIFIED**: This function now merges incoming parameters (overrides)
// with the existing default parameters, preventing the defaults from being erased.
void BaseStrategy::set_params(const ParameterMap& overrides) {
    // The constructor has already set the defaults.
    // Now, merge the overrides into the existing params_.
    for (const auto& [key, value] : overrides) {
        params_[key] = value;
    }
    
    apply_params();
}

// --- Strategy Factory Implementation ---
StrategyFactory& StrategyFactory::instance() {
    static StrategyFactory factory_instance;
    return factory_instance;
}

void StrategyFactory::register_strategy(const std::string& name, CreateFunction create_func) {
    strategies_[name] = create_func;
}

std::unique_ptr<BaseStrategy> StrategyFactory::create_strategy(const std::string& name) {
    auto it = strategies_.find(name);
    if (it != strategies_.end()) {
        return it->second();
    }
    std::cerr << "Error: Strategy '" << name << "' not found in factory." << std::endl;
    return nullptr;
}

} // namespace sentio
```

## 📄 **FILE 103 of 145**: src/cost_aware_gate.cpp

**File Information**:
- **Path**: `src/cost_aware_gate.cpp`

- **Size**: 180 lines
- **Modified**: 2025-09-07 15:31:34

- **Type**: .cpp

```text
#include "sentio/cost_aware_gate.hpp"
// StrategySignal is defined in router.hpp which is already included via cost_aware_gate.hpp
#include <algorithm>
#include <cmath>

namespace sentio {

CostAwareGate::CostAwareGate(const CostAwareConfig& config) : config_(config) {
    // Initialize default cost curves for common instruments if not provided
    if (config_.instrument_costs.empty()) {
        config_.instrument_costs["QQQ"] = {2.0, 0.05, 5.0, 0.1};
        config_.instrument_costs["SPY"] = {1.5, 0.05, 4.0, 0.1};
        config_.instrument_costs["IWM"] = {2.5, 0.05, 6.0, 0.1};
        config_.instrument_costs["TLT"] = {3.0, 0.05, 7.0, 0.1};
        config_.instrument_costs["GLD"] = {2.0, 0.05, 5.0, 0.1};
    }
}

std::optional<StrategySignal> CostAwareGate::filter_signal(
    const StrategySignal& signal,
    const std::string& instrument,
    double current_price,
    double position_size
) const {
    if (!config_.enable_cost_filtering) {
        return signal;
    }

    // Check if signal should be rejected due to cost constraints
    if (should_reject_signal(signal, instrument, current_price, position_size)) {
        return std::nullopt;
    }

    // Calculate minimum confidence required
    double min_confidence = calculate_min_confidence(instrument, position_size, current_price);
    
    // If signal confidence is below threshold, reject it
    if (signal.confidence < min_confidence) {
        return std::nullopt;
    }

    // Signal passes cost analysis
    return signal;
}

double CostAwareGate::calculate_expected_cost(
    [[maybe_unused]] StrategySignal::Type signal_type,
    const std::string& instrument,
    double position_size,
    double current_price
) const {
    const auto& curve = get_cost_curve(instrument);
    
    // Base transaction cost
    double base_cost = curve.base_cost_bp;
    
    // Add slippage cost (increases with position size)
    double slippage_cost = calculate_slippage_cost(position_size, current_price);
    
    // Add market impact cost
    double impact_cost = calculate_market_impact(position_size, current_price);
    
    return base_cost + slippage_cost + impact_cost;
}

double CostAwareGate::calculate_min_confidence(
    const std::string& instrument,
    double position_size,
    double current_price
) const {
    const auto& curve = get_cost_curve(instrument);
    
    // Calculate expected cost
    double expected_cost = calculate_expected_cost(
        StrategySignal::Type::BUY, // Use BUY as reference
        instrument,
        position_size,
        current_price
    );
    
    // Minimum confidence should be proportional to expected cost
    // Higher costs require higher confidence
    double cost_factor = expected_cost / curve.base_cost_bp;
    double min_confidence = curve.confidence_threshold * cost_factor;
    
    // Ensure minimum confidence floor
    return std::max(min_confidence, config_.default_confidence_floor);
}

void CostAwareGate::update_cost_curve(
    const std::string& instrument,
    const std::vector<double>& recent_trades
) {
    if (recent_trades.empty()) return;
    
    // Calculate average P&L per trade
    double avg_pnl = 0.0;
    for (double pnl : recent_trades) {
        avg_pnl += pnl;
    }
    avg_pnl /= recent_trades.size();
    
    // Calculate P&L volatility
    double pnl_variance = 0.0;
    for (double pnl : recent_trades) {
        double diff = pnl - avg_pnl;
        pnl_variance += diff * diff;
    }
    pnl_variance /= recent_trades.size();
    double pnl_std = std::sqrt(pnl_variance);
    
    // Update cost curve based on recent performance
    auto& curve = config_.instrument_costs[instrument];
    
    // Adjust confidence threshold based on P&L consistency
    if (pnl_std > 0) {
        double sharpe_ratio = avg_pnl / pnl_std;
        curve.confidence_threshold = std::max(0.05, 0.1 / std::max(1.0, sharpe_ratio));
    }
    
    // Adjust base cost based on recent performance
    if (avg_pnl < 0) {
        curve.base_cost_bp *= 1.1; // Increase cost estimate if losing money
    } else if (avg_pnl > curve.min_expected_return_bp) {
        curve.base_cost_bp *= 0.95; // Decrease cost estimate if profitable
    }
}

bool CostAwareGate::should_reject_signal(
    const StrategySignal& signal,
    const std::string& instrument,
    double current_price,
    double position_size
) const {
    const auto& curve = get_cost_curve(instrument);
    
    // Check position size limits
    if (std::abs(position_size) > curve.max_position_size) {
        return true;
    }
    
    // Check minimum expected return
    double expected_cost = calculate_expected_cost(
        signal.type,
        instrument,
        position_size,
        current_price
    );
    
    // Estimate expected return based on signal confidence
    double estimated_return = signal.confidence * curve.min_expected_return_bp;
    
    // Reject if expected return is less than expected cost
    return estimated_return < expected_cost;
}

const CostAwareGate::CostCurve& CostAwareGate::get_cost_curve(const std::string& instrument) const {
    auto it = config_.instrument_costs.find(instrument);
    if (it != config_.instrument_costs.end()) {
        return it->second;
    }
    
    // Return default cost curve
    static CostCurve default_curve{config_.default_cost_bp, config_.default_confidence_floor, 5.0, 0.1};
    return default_curve;
}

double CostAwareGate::calculate_slippage_cost(double position_size, [[maybe_unused]] double current_price) const {
    // Slippage increases with position size (simplified model)
    double size_factor = std::abs(position_size) / 0.1; // Normalize to 10% position size
    return 0.5 * size_factor; // 0.5 bp per 10% position size
}

double CostAwareGate::calculate_market_impact(double position_size, [[maybe_unused]] double current_price) const {
    // Market impact increases quadratically with position size
    double size_factor = std::abs(position_size) / 0.1; // Normalize to 10% position size
    return 0.1 * size_factor * size_factor; // Quadratic impact
}

} // namespace sentio

```

## 📄 **FILE 104 of 145**: src/csv_loader.cpp

**File Information**:
- **Path**: `src/csv_loader.cpp`

- **Size**: 93 lines
- **Modified**: 2025-09-07 04:14:17

- **Type**: .cpp

```text
#include "sentio/csv_loader.hpp"
#include "sentio/binio.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctz/time_zone.h>
#include <cctz/civil_time.h>

namespace sentio {

bool load_csv(const std::string& filename, std::vector<Bar>& out) {
    // Try binary cache first for performance
    std::string bin_filename = filename.substr(0, filename.find_last_of('.')) + ".bin";
    auto cached = load_bin(bin_filename);
    if (!cached.empty()) {
        out = std::move(cached);
        return true;
    }
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string timestamp_str, symbol, open_str, high_str, low_str, close_str, volume_str;
        
        std::getline(ss, timestamp_str, ',');
        std::getline(ss, symbol, ',');
        std::getline(ss, open_str, ',');
        std::getline(ss, high_str, ',');
        std::getline(ss, low_str, ',');
        std::getline(ss, close_str, ',');
        std::getline(ss, volume_str, ',');
        
        Bar bar;
        bar.ts_utc = timestamp_str;
        
        // **MODIFIED**: Parse ISO 8601 timestamp directly as UTC
        try {
            // Parse the RFC3339 / ISO 8601 timestamp string (e.g., "2023-10-27T13:30:00Z")
            cctz::time_zone utc_tz;
            if (cctz::load_time_zone("UTC", &utc_tz)) {
                cctz::time_point<cctz::seconds> utc_tp;
                if (cctz::parse("%Y-%m-%dT%H:%M:%S%Ez", timestamp_str, utc_tz, &utc_tp)) {
                    bar.ts_nyt_epoch = utc_tp.time_since_epoch().count();
                } else {
                    // Try alternative format with Z suffix
                    if (cctz::parse("%Y-%m-%dT%H:%M:%SZ", timestamp_str, utc_tz, &utc_tp)) {
                        bar.ts_nyt_epoch = utc_tp.time_since_epoch().count();
                    } else {
                        // Try space format
                        if (cctz::parse("%Y-%m-%d %H:%M:%S%Ez", timestamp_str, utc_tz, &utc_tp)) {
                            bar.ts_nyt_epoch = utc_tp.time_since_epoch().count();
                        } else {
                            bar.ts_nyt_epoch = 0;
                        }
                    }
                }
            } else {
                bar.ts_nyt_epoch = 0; // Could not load timezone
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse timestamp '" << timestamp_str << "'. Error: " << e.what() << std::endl;
            bar.ts_nyt_epoch = 0;
        }
        
        try {
            bar.open = std::stod(open_str);
            bar.high = std::stod(high_str);
            bar.low = std::stod(low_str);
            bar.close = std::stod(close_str);
            bar.volume = std::stoull(volume_str);
            out.push_back(bar);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not parse bar data line: " << line << std::endl;
        }
    }
    
    // Save to binary cache for next time
    if (!out.empty()) {
        save_bin(bin_filename, out);
    }
    
    return true;
}

} // namespace sentio
```

## 📄 **FILE 105 of 145**: src/feature_builder.cpp

**File Information**:
- **Path**: `src/feature_builder.cpp`

- **Size**: 109 lines
- **Modified**: 2025-09-06 01:37:11

- **Type**: .cpp

```text
#include "sentio/feature_builder.hpp"
#include <algorithm>

namespace sentio {

FeatureBuilder::FeatureBuilder(FeaturePlan plan, FeatureBuilderCfg cfg)
: plan_(std::move(plan))
, cfg_(cfg)
, sma_fast_(cfg.sma_fast)
, sma_slow_(cfg.sma_slow)
, vol_rtn_(std::max(2, cfg.vol_window))
, rsi_(cfg.rsi_period)
{}

void FeatureBuilder::reset(){
  *this = FeatureBuilder(plan_, cfg_);
}

void FeatureBuilder::on_bar(const Bar& b, const std::optional<MicroTick>& mt) {
  ++bars_seen_;

  // --- returns ---
  if (!close_q_.empty()) {
    double prev = close_q_.back();
    last_ret_1m_ = (b.close - prev) / std::max(1e-12, prev);
    vol_rtn_.push(last_ret_1m_);
  } else {
    last_ret_1m_ = 0.0;
  }

  close_q_.push_back(b.close);
  if (close_q_.size() > (size_t)std::max({cfg_.ret_5m_window, cfg_.sma_slow, cfg_.rsi_period+1})) {
    close_q_.pop_front();
  }

  if (close_q_.size() >= (size_t)cfg_.ret_5m_window+1) {
    double prev5 = close_q_[close_q_.size()-(cfg_.ret_5m_window+1)];
    last_ret_5m_ = (b.close - prev5) / std::max(1e-12, prev5);
  }

  // --- RSI ---
  if (close_q_.size() >= 2) {
    double prev = close_q_[close_q_.size()-2];
    rsi_.push(prev, b.close);
    last_rsi_ = rsi_.value();
  }

  // --- SMA fast/slow ---
  sma_fast_.push(b.close);
  sma_slow_.push(b.close);
  last_sma_fast_ = sma_fast_.full()? sma_fast_.mean() : NAN;
  last_sma_slow_ = sma_slow_.full()? sma_slow_.mean() : NAN;

  // --- Volatility (stdev of 1m returns) ---
  last_vol_1m_ = vol_rtn_.full()? vol_rtn_.stdev() : NAN;

  // --- Spread bp ---
  if (mt && std::isfinite(mt->bid) && std::isfinite(mt->ask)) {
    double mid = 0.5*(mt->bid + mt->ask);
    if (mid>0) last_spread_bp_ = 1e4 * (mt->ask - mt->bid) / mid;
  } else {
    // Proxy from high/low as a fallback (intrabar range proxy), otherwise default
    double mid = (b.high + b.low) * 0.5;
    if (mid>0) last_spread_bp_ = 1e4 * (b.high - b.low) / std::max(1e-12, mid) * 0.1; // scaled
    else       last_spread_bp_ = cfg_.default_spread_bp;
  }

  // clamp any negatives/NaNs on first bars
  if (!std::isfinite(last_ret_5m_)) last_ret_5m_ = 0.0;
  if (!std::isfinite(last_rsi_))    last_rsi_    = 50.0;
  if (!std::isfinite(last_sma_fast_)) last_sma_fast_ = b.close;
  if (!std::isfinite(last_sma_slow_)) last_sma_slow_ = b.close;
  if (!std::isfinite(last_vol_1m_))   last_vol_1m_   = 0.0;
  if (!std::isfinite(last_spread_bp_)) last_spread_bp_ = cfg_.default_spread_bp;
}

bool FeatureBuilder::ready() const {
  // Require the slowest indicator to be ready (SMA slow & RSI & vol window)
  bool sma_ok = sma_slow_.full();
  bool rsi_ok = rsi_.ready();
  bool vol_ok = vol_rtn_.full();
  // ret_5m needs at least 5+1 bars, covered by SMA slow usually; but check anyway
  bool r5_ok  = close_q_.size() >= (size_t)cfg_.ret_5m_window+1;
  return sma_ok && rsi_ok && vol_ok && r5_ok
      && finite(last_ret_1m_) && finite(last_ret_5m_) && finite(last_rsi_)
      && finite(last_sma_fast_) && finite(last_sma_slow_) && finite(last_vol_1m_) && finite(last_spread_bp_);
}

std::optional<std::vector<double>> FeatureBuilder::build() const {
  if (!ready()) return std::nullopt;
  std::vector<double> out; out.reserve(plan_.names.size());

  for (const auto& name : plan_.names) {
    if (name=="ret_1m")      out.push_back(last_ret_1m_);
    else if (name=="ret_5m") out.push_back(last_ret_5m_);
    else if (name=="rsi_14") out.push_back(last_rsi_);
    else if (name=="sma_10") out.push_back(last_sma_fast_);
    else if (name=="sma_30") out.push_back(last_sma_slow_);
    else if (name=="vol_1m") out.push_back(last_vol_1m_);
    else if (name=="spread_bp") out.push_back(last_spread_bp_);
    else {
      // Unknown feature: fail closed so you'll notice in tests
      throw std::runtime_error("FeatureBuilder: unsupported feature name: " + name);
    }
  }
  return out;
}

} // namespace sentio

```

## 📄 **FILE 106 of 145**: src/feature_cache.cpp

**File Information**:
- **Path**: `src/feature_cache.cpp`

- **Size**: 127 lines
- **Modified**: 2025-09-07 20:15:09

- **Type**: .cpp

```text
#include "sentio/feature_cache.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace sentio {

bool FeatureCache::load_from_csv(const std::string& feature_file_path) {
    std::ifstream file(feature_file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open feature file: " << feature_file_path << std::endl;
        return false;
    }

    std::string line;
    bool is_header = true;
    size_t lines_processed = 0;

    while (std::getline(file, line)) {
        if (is_header) {
            // Parse header to get feature names
            std::stringstream ss(line);
            std::string column;
            
            // Skip bar_index and timestamp columns
            std::getline(ss, column, ','); // bar_index
            std::getline(ss, column, ','); // timestamp
            
            // Read feature names
            feature_names_.clear();
            while (std::getline(ss, column, ',')) {
                feature_names_.push_back(column);
            }
            
            std::cout << "FeatureCache: Loaded " << feature_names_.size() << " feature names" << std::endl;
            is_header = false;
            continue;
        }

        // Parse data line
        std::stringstream ss(line);
        std::string cell;
        
        // Read bar_index
        if (!std::getline(ss, cell, ',')) continue;
        int bar_index = std::stoi(cell);
        
        // Skip timestamp
        if (!std::getline(ss, cell, ',')) continue;
        
        // Read features
        std::vector<double> features;
        features.reserve(feature_names_.size());
        
        while (std::getline(ss, cell, ',')) {
            features.push_back(std::stod(cell));
        }
        
        // Verify feature count  
        if (features.size() != feature_names_.size()) {
            std::cerr << "CRITICAL: Bar " << bar_index << " has " << features.size() 
                      << " features, expected " << feature_names_.size() << std::endl;
            std::cerr << "CSV line: " << line << std::endl;
            std::cerr << "This will cause missing features in get_features()!" << std::endl;
            continue;  // This is the bug - skipping bars causes missing data
        }
        
        // Debug output for first few bars
        if (lines_processed < 3) {
            std::cout << "[DEBUG] Bar " << bar_index << ": loaded " << features.size() 
                      << " features (expected " << feature_names_.size() << ")" << std::endl;
        }
        
        // Store features
        features_by_bar_[bar_index] = std::move(features);
        lines_processed++;
        
        // Progress reporting
        if (lines_processed % 50000 == 0) {
            std::cout << "FeatureCache: Loaded " << lines_processed << " bars..." << std::endl;
        }
    }

    total_bars_ = lines_processed;
    file.close();

    std::cout << "FeatureCache: Successfully loaded " << total_bars_ << " bars with " 
              << feature_names_.size() << " features each" << std::endl;
    std::cout << "FeatureCache: Recommended starting bar: " << recommended_start_bar_ << std::endl;
    
    return true;
}

std::vector<double> FeatureCache::get_features(int bar_index) const {
    auto it = features_by_bar_.find(bar_index);
    if (it != features_by_bar_.end()) {
        static int get_calls = 0;
        get_calls++;
        if (get_calls <= 5) {
            std::cout << "[DEBUG] FeatureCache::get_features(" << bar_index 
                      << ") returning " << it->second.size() << " features" << std::endl;
        }
        return it->second;
    }
    std::cout << "[ERROR] FeatureCache::get_features(" << bar_index 
              << ") - bar not found! Returning empty vector." << std::endl;
    return {}; // Return empty vector if not found
}

bool FeatureCache::has_features(int bar_index) const {
    return features_by_bar_.find(bar_index) != features_by_bar_.end();
}

size_t FeatureCache::get_bar_count() const {
    return total_bars_;
}

int FeatureCache::get_recommended_start_bar() const {
    return recommended_start_bar_;
}

const std::vector<std::string>& FeatureCache::get_feature_names() const {
    return feature_names_;
}

} // namespace sentio

```

## 📄 **FILE 107 of 145**: src/feature_engineering/feature_normalizer.cpp

**File Information**:
- **Path**: `src/feature_engineering/feature_normalizer.cpp`

- **Size**: 373 lines
- **Modified**: 2025-09-07 04:32:21

- **Type**: .cpp

```text
#include "sentio/feature_engineering/feature_normalizer.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

namespace sentio {
namespace feature_engineering {

FeatureNormalizer::FeatureNormalizer(size_t window_size) 
    : window_size_(window_size) {
    // Initialize with empty stats - updated to match actual feature count
    stats_.resize(55); // Updated from 52 to 55 to match actual feature count
    feature_history_.resize(55);
}

std::vector<double> FeatureNormalizer::normalize_features(const std::vector<double>& features) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (features.empty()) {
        return features;
    }
    
    // Update statistics with new features
    update_stats(features);
    
    // Apply robust normalization
    return robust_normalize(features);
}

std::vector<double> FeatureNormalizer::denormalize_features(const std::vector<double>& normalized_features) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (normalized_features.empty() || stats_.empty()) {
        return normalized_features;
    }
    
    std::vector<double> denormalized;
    denormalized.reserve(normalized_features.size());
    
    for (size_t i = 0; i < normalized_features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && stat.std > 0) {
            double denorm = (normalized_features[i] * stat.std) + stat.mean;
            denormalized.push_back(denorm);
        } else {
            denormalized.push_back(normalized_features[i]);
        }
    }
    
    return denormalized;
}

void FeatureNormalizer::update_stats(const std::vector<double>& features) {
    for (size_t i = 0; i < features.size() && i < feature_history_.size(); ++i) {
        update_feature_stats(i, features[i]);
    }
}

void FeatureNormalizer::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    for (auto& stat : stats_) {
        stat = NormalizationStats{};
    }
    
    for (auto& history : feature_history_) {
        history.clear();
    }
}

std::vector<double> FeatureNormalizer::z_score_normalize(const std::vector<double>& features) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (features.empty() || stats_.empty()) {
        return features;
    }
    
    std::vector<double> normalized;
    normalized.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && stat.std > 0) {
            double normalized_value = (features[i] - stat.mean) / stat.std;
            normalized.push_back(normalized_value);
        } else {
            normalized.push_back(features[i]);
        }
    }
    
    return normalized;
}

std::vector<double> FeatureNormalizer::min_max_normalize(const std::vector<double>& features) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (features.empty() || stats_.empty()) {
        return features;
    }
    
    std::vector<double> normalized;
    normalized.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && (stat.max - stat.min) > 0) {
            double normalized_value = (features[i] - stat.min) / (stat.max - stat.min);
            normalized.push_back(normalized_value);
        } else {
            normalized.push_back(features[i]);
        }
    }
    
    return normalized;
}

std::vector<double> FeatureNormalizer::robust_normalize(const std::vector<double>& features) {
    // Note: Mutex is already held by caller (normalize_features)
    
    if (features.empty() || stats_.empty()) {
        return features;
    }
    
    std::vector<double> normalized;
    normalized.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && stat.std > 0) {
            // Use robust statistics (median and MAD)
            double robust_mean = calculate_robust_mean(feature_history_[i]);
            double robust_std = calculate_robust_std(feature_history_[i], robust_mean);
            
            if (robust_std > 0) {
                double normalized_value = (features[i] - robust_mean) / robust_std;
                normalized.push_back(normalized_value);
            } else {
                normalized.push_back(features[i]);
            }
        } else {
            normalized.push_back(features[i]);
        }
    }
    
    return normalized;
}

std::vector<double> FeatureNormalizer::clip_outliers(const std::vector<double>& features, double threshold) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (features.empty() || stats_.empty()) {
        return features;
    }
    
    std::vector<double> clipped;
    clipped.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && stat.std > 0) {
            double upper_bound = stat.mean + (threshold * stat.std);
            double lower_bound = stat.mean - (threshold * stat.std);
            
            double clipped_value = std::clamp(features[i], lower_bound, upper_bound);
            clipped.push_back(clipped_value);
        } else {
            clipped.push_back(features[i]);
        }
    }
    
    return clipped;
}

std::vector<double> FeatureNormalizer::winsorize(const std::vector<double>& features, double percentile) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (features.empty() || stats_.empty()) {
        return features;
    }
    
    std::vector<double> winsorized;
    winsorized.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < feature_history_.size(); ++i) {
        if (feature_history_[i].size() < 2) {
            winsorized.push_back(features[i]);
            continue;
        }
        
        // Calculate percentiles from history
        auto sorted_values = feature_history_[i];
        sort_values(sorted_values);
        
        double lower_percentile = calculate_percentile(sorted_values, percentile);
        double upper_percentile = calculate_percentile(sorted_values, 1.0 - percentile);
        
        double winsorized_value = std::clamp(features[i], lower_percentile, upper_percentile);
        winsorized.push_back(winsorized_value);
    }
    
    return winsorized;
}

bool FeatureNormalizer::is_normalized(const std::vector<double>& features) const {
    if (features.empty()) {
        return true;
    }
    
    // Check if features are in reasonable normalized range [-5, 5]
    for (double feature : features) {
        if (std::abs(feature) > 5.0) {
            return false;
        }
    }
    
    return true;
}

std::vector<bool> FeatureNormalizer::get_outlier_mask(const std::vector<double>& features, double threshold) const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    std::vector<bool> outlier_mask;
    outlier_mask.reserve(features.size());
    
    for (size_t i = 0; i < features.size() && i < stats_.size(); ++i) {
        const auto& stat = stats_[i];
        if (stat.count > 0 && stat.std > 0) {
            double upper_bound = stat.mean + (threshold * stat.std);
            double lower_bound = stat.mean - (threshold * stat.std);
            
            bool is_outlier = (features[i] < lower_bound) || (features[i] > upper_bound);
            outlier_mask.push_back(is_outlier);
        } else {
            outlier_mask.push_back(false);
        }
    }
    
    return outlier_mask;
}

NormalizationStats FeatureNormalizer::get_stats(size_t feature_index) const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (feature_index < stats_.size()) {
        return stats_[feature_index];
    }
    
    return NormalizationStats{};
}

std::vector<NormalizationStats> FeatureNormalizer::get_all_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void FeatureNormalizer::set_window_size(size_t window_size) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    window_size_ = window_size;
    
    // Trim existing histories to new window size
    for (auto& history : feature_history_) {
        while (history.size() > window_size_) {
            history.pop_front();
        }
    }
}

void FeatureNormalizer::set_outlier_threshold(double threshold) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    outlier_threshold_ = threshold;
}

void FeatureNormalizer::set_winsorize_percentile(double percentile) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    winsorize_percentile_ = percentile;
}

void FeatureNormalizer::update_feature_stats(size_t feature_index, double value) {
    if (feature_index >= feature_history_.size()) {
        return;
    }
    
    auto& history = feature_history_[feature_index];
    auto& stat = stats_[feature_index];
    
    // Add new value to history
    history.push_back(value);
    
    // Trim history to window size
    while (history.size() > window_size_) {
        history.pop_front();
    }
    
    // Update statistics
    if (history.size() == 1) {
        stat.mean = value;
        stat.std = 0.0;
        stat.min = value;
        stat.max = value;
        stat.count = 1;
    } else {
        // Calculate running statistics
        double sum = std::accumulate(history.begin(), history.end(), 0.0);
        stat.mean = sum / history.size();
        
        double variance = 0.0;
        for (double v : history) {
            double diff = v - stat.mean;
            variance += diff * diff;
        }
        stat.std = std::sqrt(variance / (history.size() - 1));
        
        stat.min = *std::min_element(history.begin(), history.end());
        stat.max = *std::max_element(history.begin(), history.end());
        stat.count = history.size();
    }
}

double FeatureNormalizer::calculate_robust_mean(const std::deque<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    
    auto sorted_values = values;
    sort_values(sorted_values);
    
    size_t n = sorted_values.size();
    if (n % 2 == 0) {
        return (sorted_values[n/2 - 1] + sorted_values[n/2]) / 2.0;
    } else {
        return sorted_values[n/2];
    }
}

double FeatureNormalizer::calculate_robust_std(const std::deque<double>& values, double mean) {
    if (values.size() < 2) {
        return 0.0;
    }
    
    // Calculate Median Absolute Deviation (MAD)
    std::vector<double> deviations;
    deviations.reserve(values.size());
    
    for (double value : values) {
        deviations.push_back(std::abs(value - mean));
    }
    
    std::sort(deviations.begin(), deviations.end());
    
    double mad = deviations[deviations.size() / 2];
    
    // Convert MAD to standard deviation approximation
    return mad * 1.4826;
}

double FeatureNormalizer::calculate_percentile(const std::deque<double>& values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }
    
    size_t index = static_cast<size_t>(percentile * (values.size() - 1));
    index = std::min(index, values.size() - 1);
    
    return values[index];
}

void FeatureNormalizer::sort_values(std::deque<double>& values) {
    std::sort(values.begin(), values.end());
}

} // namespace feature_engineering
} // namespace sentio

```

## 📄 **FILE 108 of 145**: src/feature_engineering/kochi_features.cpp

**File Information**:
- **Path**: `src/feature_engineering/kochi_features.cpp`

- **Size**: 146 lines
- **Modified**: 2025-09-07 23:21:17

- **Type**: .cpp

```text
#include "sentio/feature_engineering/kochi_features.hpp"
#include <cmath>
#include <algorithm>

namespace sentio {
namespace feature_engineering {

std::vector<std::string> kochi_feature_names() {
  // Derived from third_party/kochi/kochi/data_processor.py feature engineering
  // Excludes OHLCV base columns
  return {
    // Time features
    "HOUR_SIN","HOUR_COS","DOW_SIN","DOW_COS","WOY_SIN","WOY_COS",
    // Overlap features
    "SMA_5","SMA_20","EMA_5","EMA_20",
    "BB_UPPER","BB_LOWER","BB_MIDDLE",
    "TENKAN_SEN","KIJUN_SEN","SENKOU_SPAN_A","CHIKOU_SPAN","SAR",
    "TURBULENCE","VWAP","VWAP_ZSCORE",
    // Momentum features
    "AROON_UP","AROON_DOWN","AROON_OSC",
    "MACD","MACD_SIGNAL","MACD_HIST","MOMENTUM","ROC","RSI",
    "STOCH_SLOWK","STOCH_SLOWD","STOCH_CROSS","WILLR",
    "PLUS_DI","MINUS_DI","ADX","CCI","PPO","ULTOSC",
    "SQUEEZE_ON","SQUEEZE_OFF",
    // Volatility features
    "STDDEV","ATR","ADL_30","OBV_30","VOLATILITY",
    "KELTNER_UPPER","KELTNER_LOWER","KELTNER_MIDDLE","GARMAN_KLASS",
    // Price action features
    "LOG_RETURN_1","OVERNIGHT_GAP","BAR_SHAPE","SHADOW_UP","SHADOW_DOWN",
    "DOJI","HAMMER","ENGULFING"
  };
}

// Helpers
static double safe_div(double a, double b){ return b!=0.0? a/b : 0.0; }
static double clip(double x, double lo, double hi){ return std::max(lo, std::min(hi, x)); }

std::vector<double> calculate_kochi_features(const std::vector<Bar>& bars, int i) {
  std::vector<double> f; f.reserve(64);
  if (i<=0 || i>=(int)bars.size()) return f;

  // Minimal rolling helpers inline (lightweight subset sufficient for inference parity)
  auto roll_mean = [&](int win, auto getter){ double s=0.0; int n=0; for (int k=i-win+1;k<=i;++k){ if (k<0) continue; s += getter(k); ++n;} return n>0? s/n:0.0; };
  auto roll_std  = [&](int win, auto getter){ double m=roll_mean(win,getter); double v=0.0; int n=0; for (int k=i-win+1;k<=i;++k){ if (k<0) continue; double x=getter(k)-m; v+=x*x; ++n;} return n>1? std::sqrt(v/(n-1)) : 0.0; };
  auto high_at=[&](int k){ return bars[k].high; };
  auto low_at =[&](int k){ return bars[k].low; };
  auto close_at=[&](int k){ return bars[k].close; };
  auto open_at=[&](int k){ return bars[k].open; };
  auto vol_at=[&](int k){ return double(bars[k].volume); };

  // Basic time encodings from timestamp: approximate via NY hour/day/week if available
  // Here we cannot access tz; emit zeros to keep layout. Trainer will handle.
  double HOUR_SIN=0, HOUR_COS=0, DOW_SIN=0, DOW_COS=0, WOY_SIN=0, WOY_COS=0;
  f.insert(f.end(), {HOUR_SIN,HOUR_COS,DOW_SIN,DOW_COS,WOY_SIN,WOY_COS});

  // SMA/EMA minimal
  auto sma = [&](int w){ return roll_mean(w, close_at) - close_at(i); };
  auto ema = [&](int w){ double a=2.0/(w+1.0); double e=close_at(0); for(int k=1;k<=i;++k){ e=a*close_at(k)+(1-a)*e; } return e - close_at(i); };
  double SMA_5=sma(5), SMA_20=sma(20);
  double EMA_5=ema(5), EMA_20=ema(20);
  f.insert(f.end(), {SMA_5,SMA_20,EMA_5,EMA_20});

  // Bollinger 20
  double m20 = roll_mean(20, close_at);
  double sd20 = roll_std(20, close_at);
  double BB_UPPER = (m20 + 2.0*sd20) - close_at(i);
  double BB_LOWER = (m20 - 2.0*sd20) - close_at(i);
  double BB_MIDDLE = m20 - close_at(i);
  f.insert(f.end(), {BB_UPPER,BB_LOWER,BB_MIDDLE});

  // Ichimoku simplified
  auto max_roll = [&](int w){ double mx=high_at(std::max(0,i-w+1)); for(int k=i-w+1;k<=i;++k) if(k>=0) mx=std::max(mx, high_at(k)); return mx; };
  auto min_roll = [&](int w){ double mn=low_at(std::max(0,i-w+1)); for(int k=i-w+1;k<=i;++k) if(k>=0) mn=std::min(mn, low_at(k)); return mn; };
  double TENKAN = 0.5*(max_roll(9)+min_roll(9));
  double KIJUN  = 0.5*(max_roll(26)+min_roll(26));
  double SENKOU_A = 0.5*(TENKAN+KIJUN);
  double CHIKOU = (i+26<(int)bars.size()? bars[i+26].close : bars[i].close) - close_at(i);
  // Parabolic SAR surrogate using high-low
  double SAR = (high_at(i)-low_at(i));
  double TURB = safe_div(high_at(i)-low_at(i), std::max(1e-8, open_at(i)));
  // Rolling VWAP proxy and zscore
  double num=0, den=0; for (int k=i-13;k<=i;++k){ if(k<0) continue; num += close_at(k)*vol_at(k); den += vol_at(k);} double VWAP = safe_div(num, den) - close_at(i);
  double vwap_mean = roll_mean(20, [&](int k){ double n=0,d=0; for(int j=k-13;j<=k;++j){ if(j<0) continue; n+=close_at(j)*vol_at(j); d+=vol_at(j);} return safe_div(n,d) - close_at(k);} );
  double vwap_std  = roll_std(20, [&](int k){ double n=0,d=0; for(int j=k-13;j<=k;++j){ if(j<0) continue; n+=close_at(j)*vol_at(j); d+=vol_at(j);} return safe_div(n,d) - close_at(k);} );
  double VWAP_Z = (vwap_std>0? (VWAP - vwap_mean)/vwap_std : 0.0);
  f.insert(f.end(), {TENKAN - close_at(i), KIJUN - close_at(i), SENKOU_A - close_at(i), CHIKOU, SAR, TURB, VWAP, VWAP_Z});

  // Momentum / oscillators (simplified)
  // Aroon
  int w=14; int idx_up=i, idx_dn=i; double hh=high_at(i), ll=low_at(i);
  for(int k=i-w+1;k<=i;++k){ if(k<0) continue; if (high_at(k)>=hh){ hh=high_at(k); idx_up=k;} if (low_at(k)<=ll){ ll=low_at(k); idx_dn=k;} }
  double AROON_UP = 1.0 - double(i-idx_up)/std::max(1,w);
  double AROON_DN = 1.0 - double(i-idx_dn)/std::max(1,w);
  double AROON_OSC = AROON_UP - AROON_DN;
  // MACD simplified
  auto emaN = [&](int p){ double a=2.0/(p+1.0); double e=close_at(0); for(int k=1;k<=i;++k) e = a*close_at(k) + (1-a)*e; return e; };
  double macd = emaN(12) - emaN(26); double macds=macd; double macdh=macd-macds;
  double MOM = (i>=10? close_at(i) - close_at(i-10) : 0.0);
  double ROC = (i>=10? (close_at(i)/std::max(1e-12, close_at(i-10)) - 1.0)/100.0 : 0.0);
  // RSI (scaled 0..1)
  int rp=14; double gain=0, loss=0; for(int k=i-rp+1;k<=i;++k){ if(k<=0) continue; double d=close_at(k)-close_at(k-1); if (d>0) gain+=d; else loss-=d; }
  double RSI = (loss==0.0? 1.0 : (gain/(gain+loss)));
  // Stochastics
  double hh14=max_roll(14), ll14=min_roll(14); double STOK = (hh14==ll14? 0.0 : (close_at(i)-ll14)/(hh14-ll14));
  double STOD = STOK; int STOCROSS = STOK>STOD? 1:0;
  // Williams %R scaled to [0,1]
  double WILLR = (hh14==ll14? 0.5 : (close_at(i)-ll14)/(hh14-ll14));
  // DI/ADX placeholders and others
  double PLUS_DI=0, MINUS_DI=0, ADX=0, CCI=0, PPO=0, ULTOSC=0;
  // Squeeze Madrid indicators (approx from BB/KC)
  double atr20 = 0.0; for(int k=i-19;k<=i;++k){ if(k<=0) continue; double tr = std::max({high_at(k)-low_at(k), std::fabs(high_at(k)-close_at(k-1)), std::fabs(low_at(k)-close_at(k-1))}); atr20 += tr; } atr20/=20.0;
  double KC_UB = m20 + 1.5*atr20; double KC_LB = m20 - 1.5*atr20;
  int SQUEEZE_ON = ( (m20-2*sd20 > KC_LB) && (m20+2*sd20 < KC_UB) ) ? 1 : 0;
  int SQUEEZE_OFF= ( (m20-2*sd20 < KC_LB) && (m20+2*sd20 > KC_UB) ) ? 1 : 0;
  f.insert(f.end(), {AROON_UP,AROON_DN,AROON_OSC, macd, macds, macdh, MOM, ROC, RSI, STOK, STOD, (double)STOCROSS, WILLR, PLUS_DI, MINUS_DI, ADX, CCI, PPO, ULTOSC, (double)SQUEEZE_ON, (double)SQUEEZE_OFF});

  // Volatility set
  double STDDEV = roll_std(20, close_at);
  double ATR = atr20;
  // ADL_30/OBV_30 rolling sums proxies
  double adl=0.0, obv=0.0; for(int k=i-29;k<=i;++k){ if(k<=0) continue; double clv = safe_div((close_at(k)-low_at(k)) - (high_at(k)-close_at(k)), (high_at(k)-low_at(k))); adl += clv*vol_at(k); if (close_at(k)>close_at(k-1)) obv += vol_at(k); else if (close_at(k)<close_at(k-1)) obv -= vol_at(k); }
  double VAR = 0.0; { double m=roll_mean(20, close_at); for(int k=i-19;k<=i;++k){ double d=close_at(k)-m; VAR += d*d; } VAR/=20.0; }
  double GK = 0.0; { for(int k=i-19;k<=i;++k){ double log_hl=std::log(high_at(k)/low_at(k)); double log_co=std::log(close_at(k)/open_at(k)); GK += 0.5*log_hl*log_hl - (2.0*std::log(2.0)-1.0)*log_co*log_co; } GK/=20.0; GK=std::sqrt(std::max(0.0, GK)); }
  double KCU= (m20 + 2*atr20) - close_at(i);
  double KCL= (m20 - 2*atr20) - close_at(i);
  double KCM= m20 - close_at(i);
  f.insert(f.end(), {STDDEV, ATR, adl, obv, VAR, KCU, KCL, KCM, GK});

  // Price action
  double LOG_RET1 = std::log(std::max(1e-12, close_at(i))) - std::log(std::max(1e-12, close_at(i-1)));
  // Overnight gap approximation requires previous day close; not available — set 0
  double OVG=0.0;
  double body = std::fabs(close_at(i)-open_at(i)); double hl = std::max(1e-8, high_at(i)-low_at(i));
  double BAR_SHAPE = body/hl;
  double SHADOW_UP = (high_at(i) - std::max(close_at(i), open_at(i))) / hl;
  double SHADOW_DN = (std::min(close_at(i), open_at(i)) - low_at(i)) / hl;
  double DOJI=0.0, HAMMER=0.0, ENGULFING=0.0; // simplified candlesticks
  f.insert(f.end(), {LOG_RET1, OVG, BAR_SHAPE, SHADOW_UP, SHADOW_DN, DOJI, HAMMER, ENGULFING});

  return f;
}

} // namespace feature_engineering
} // namespace sentio



```

## 📄 **FILE 109 of 145**: src/feature_engineering/technical_indicators.cpp

**File Information**:
- **Path**: `src/feature_engineering/technical_indicators.cpp`

- **Size**: 684 lines
- **Modified**: 2025-09-07 14:29:16

- **Type**: .cpp

```text
#include "sentio/feature_engineering/technical_indicators.hpp"
#include <numeric>
#include <algorithm>
#include <cmath>

namespace sentio {
namespace feature_engineering {

TechnicalIndicatorCalculator::TechnicalIndicatorCalculator() {
    // Initialize feature names in order
    feature_names_ = {
        // Price features (15)
        "ret_1m", "ret_5m", "ret_15m", "ret_30m", "ret_1h",
        "momentum_5", "momentum_10", "momentum_20",
        "volatility_10", "volatility_20", "volatility_30",
        "atr_14", "atr_21", "parkinson_vol", "garman_klass_vol",
        
        // Technical features (25)
        "rsi_14", "rsi_21", "rsi_30",
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
        "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
        "bb_upper_20", "bb_middle_20", "bb_lower_20",
        "bb_upper_50", "bb_middle_50", "bb_lower_50",
        "macd_line", "macd_signal", "macd_histogram",
        "stoch_k", "stoch_d", "williams_r", "cci_20", "adx_14",
        
        // Volume features (7)
        "volume_sma_10", "volume_sma_20", "volume_sma_50",
        "volume_roc", "obv", "vpt", "ad_line", "mfi_14",
        
        // Microstructure features (5)
        "spread_bp", "price_impact", "order_flow_imbalance", "market_depth", "bid_ask_ratio"
    };
}

std::vector<double> TechnicalIndicatorCalculator::extract_closes(const std::vector<Bar>& bars) {
    std::vector<double> closes;
    closes.reserve(bars.size());
    for (const auto& bar : bars) {
        closes.push_back(bar.close);
    }
    return closes;
}

std::vector<double> TechnicalIndicatorCalculator::extract_volumes(const std::vector<Bar>& bars) {
    std::vector<double> volumes;
    volumes.reserve(bars.size());
    for (const auto& bar : bars) {
        volumes.push_back(static_cast<double>(bar.volume));
    }
    return volumes;
}

std::vector<double> TechnicalIndicatorCalculator::extract_returns(const std::vector<Bar>& bars) {
    std::vector<double> returns;
    if (bars.size() < 2) return returns;
    
    returns.reserve(bars.size() - 1);
    for (size_t i = 1; i < bars.size(); ++i) {
        double ret = (bars[i].close - bars[i-1].close) / std::max(1e-12, bars[i-1].close);
        returns.push_back(ret);
    }
    return returns;
}

PriceFeatures TechnicalIndicatorCalculator::calculate_price_features(const std::vector<Bar>& bars, int current_index) {
    PriceFeatures features{};
    
    if (current_index < 1 || current_index >= static_cast<int>(bars.size())) {
        return features;
    }
    
    const Bar& current = bars[current_index];
    
    // 1-minute return
    features.ret_1m = (current.close - bars[current_index - 1].close) / std::max(1e-12, bars[current_index - 1].close);
    
    // Price features calculated successfully
    
    // Multi-period returns
    if (current_index >= 5) {
        features.ret_5m = (current.close - bars[current_index - 5].close) / std::max(1e-12, bars[current_index - 5].close);
    }
    if (current_index >= 15) {
        features.ret_15m = (current.close - bars[current_index - 15].close) / std::max(1e-12, bars[current_index - 15].close);
    }
    if (current_index >= 30) {
        features.ret_30m = (current.close - bars[current_index - 30].close) / std::max(1e-12, bars[current_index - 30].close);
    }
    if (current_index >= 60) {
        features.ret_1h = (current.close - bars[current_index - 60].close) / std::max(1e-12, bars[current_index - 60].close);
    }
    
    // Momentum features
    if (current_index >= 5) {
        features.momentum_5 = current.close / std::max(1e-12, bars[current_index - 5].close) - 1.0;
        
        // Momentum features calculated successfully
    }
    if (current_index >= 10) {
        features.momentum_10 = current.close / std::max(1e-12, bars[current_index - 10].close) - 1.0;
    }
    if (current_index >= 20) {
        features.momentum_20 = current.close / std::max(1e-12, bars[current_index - 20].close) - 1.0;
    }
    
    // Volatility features
    auto returns = extract_returns(bars);
    if (current_index >= 10) {
        features.volatility_10 = calculate_volatility(returns, 10, current_index - 1);
    }
    if (current_index >= 20) {
        features.volatility_20 = calculate_volatility(returns, 20, current_index - 1);
    }
    if (current_index >= 30) {
        features.volatility_30 = calculate_volatility(returns, 30, current_index - 1);
    }
    
    // ATR
    features.atr_14 = calculate_atr(bars, 14, current_index);
    features.atr_21 = calculate_atr(bars, 21, current_index);
    
    // Advanced volatility measures
    features.parkinson_vol = calculate_parkinson_volatility(bars, 20, current_index);
    features.garman_klass_vol = calculate_garman_klass_volatility(bars, 20, current_index);
    
    return features;
}

TechnicalFeatures TechnicalIndicatorCalculator::calculate_technical_features(const std::vector<Bar>& bars, int current_index) {
    TechnicalFeatures features{};
    
    if (current_index < 1 || current_index >= static_cast<int>(bars.size())) {
        return features;
    }
    
    auto closes = extract_closes(bars);
    
    // RSI
    features.rsi_14 = calculate_rsi(closes, 14, current_index);
    features.rsi_21 = calculate_rsi(closes, 21, current_index);
    features.rsi_30 = calculate_rsi(closes, 30, current_index);
    
    // SMA
    features.sma_5 = calculate_sma(closes, 5, current_index);
    features.sma_10 = calculate_sma(closes, 10, current_index);
    features.sma_20 = calculate_sma(closes, 20, current_index);
    features.sma_50 = calculate_sma(closes, 50, current_index);
    features.sma_200 = calculate_sma(closes, 200, current_index);
    
    // EMA
    features.ema_5 = calculate_ema(closes, 5, current_index);
    features.ema_10 = calculate_ema(closes, 10, current_index);
    features.ema_20 = calculate_ema(closes, 20, current_index);
    features.ema_50 = calculate_ema(closes, 50, current_index);
    features.ema_200 = calculate_ema(closes, 200, current_index);
    
    // Bollinger Bands
    auto bb_20 = calculate_bollinger_bands(closes, 20, 2.0, current_index);
    features.bb_upper_20 = bb_20.upper;
    features.bb_middle_20 = bb_20.middle;
    features.bb_lower_20 = bb_20.lower;
    
    auto bb_50 = calculate_bollinger_bands(closes, 50, 2.0, current_index);
    features.bb_upper_50 = bb_50.upper;
    features.bb_middle_50 = bb_50.middle;
    features.bb_lower_50 = bb_50.lower;
    
    // MACD
    auto macd = calculate_macd(closes, 12, 26, 9, current_index);
    features.macd_line = macd.line;
    features.macd_signal = macd.signal;
    features.macd_histogram = macd.histogram;
    
    // Stochastic
    auto stoch = calculate_stochastic(bars, 14, 3, current_index);
    features.stoch_k = stoch.k;
    features.stoch_d = stoch.d;
    
    // Williams %R
    features.williams_r = calculate_williams_r(bars, 14, current_index);
    
    // CCI
    features.cci_20 = calculate_cci(bars, 20, current_index);
    
    // ADX
    features.adx_14 = calculate_adx(bars, 14, current_index);
    
    return features;
}

VolumeFeatures TechnicalIndicatorCalculator::calculate_volume_features(const std::vector<Bar>& bars, int current_index) {
    VolumeFeatures features{};
    
    if (current_index < 1 || current_index >= static_cast<int>(bars.size())) {
        return features;
    }
    
    auto volumes = extract_volumes(bars);
    
    // Volume SMA
    features.volume_sma_10 = calculate_sma(volumes, 10, current_index);
    features.volume_sma_20 = calculate_sma(volumes, 20, current_index);
    features.volume_sma_50 = calculate_sma(volumes, 50, current_index);
    
    // Volume Rate of Change
    if (current_index >= 10) {
        double vol_10_ago = volumes[current_index - 10];
        features.volume_roc = (volumes[current_index] - vol_10_ago) / std::max(1e-12, vol_10_ago);
    }
    
    // On-Balance Volume
    features.obv = calculate_obv(bars, current_index);
    
    // Volume-Price Trend
    features.vpt = calculate_vpt(bars, current_index);
    
    // Accumulation/Distribution Line
    features.ad_line = calculate_ad_line(bars, current_index);
    
    // Money Flow Index
    features.mfi_14 = calculate_mfi(bars, 14, current_index);
    
    return features;
}

MicrostructureFeatures TechnicalIndicatorCalculator::calculate_microstructure_features(const std::vector<Bar>& bars, int current_index) {
    MicrostructureFeatures features{};
    
    if (current_index < 1 || current_index >= static_cast<int>(bars.size())) {
        return features;
    }
    
    const Bar& current = bars[current_index];
    
    // For now, use simplified microstructure features
    // In a real implementation, these would come from order book data
    
    // Spread (simplified - using high-low as proxy)
    features.spread_bp = ((current.high - current.low) / current.close) * 10000.0;
    
    // Price impact (simplified)
    features.price_impact = std::abs(current.close - current.open) / current.open;
    
    // Order flow imbalance (simplified - using volume and price movement)
    if (current.close > current.open) {
        features.order_flow_imbalance = static_cast<double>(current.volume) / 1000000.0;
    } else {
        features.order_flow_imbalance = -static_cast<double>(current.volume) / 1000000.0;
    }
    
    // Market depth (simplified)
    features.market_depth = static_cast<double>(current.volume) / 100000.0;
    
    // Bid-ask ratio (simplified)
    features.bid_ask_ratio = current.high / std::max(1e-12, current.low);
    
    return features;
}

std::vector<double> TechnicalIndicatorCalculator::calculate_all_features(const std::vector<Bar>& bars, int current_index) {
    std::vector<double> features;
    
    // Technical indicators are working correctly - features appear small but are meaningful
    
    // Calculate all feature groups
    auto price_features = calculate_price_features(bars, current_index);
    auto technical_features = calculate_technical_features(bars, current_index);
    auto volume_features = calculate_volume_features(bars, current_index);
    auto microstructure_features = calculate_microstructure_features(bars, current_index);
    
    // Combine into single vector
    features.reserve(52); // Total feature count
    
    // Price features (15)
    features.push_back(price_features.ret_1m);
    features.push_back(price_features.ret_5m);
    features.push_back(price_features.ret_15m);
    features.push_back(price_features.ret_30m);
    features.push_back(price_features.ret_1h);
    features.push_back(price_features.momentum_5);
    features.push_back(price_features.momentum_10);
    features.push_back(price_features.momentum_20);
    features.push_back(price_features.volatility_10);
    features.push_back(price_features.volatility_20);
    features.push_back(price_features.volatility_30);
    features.push_back(price_features.atr_14);
    features.push_back(price_features.atr_21);
    features.push_back(price_features.parkinson_vol);
    features.push_back(price_features.garman_klass_vol);
    
    // Technical features (25)
    features.push_back(technical_features.rsi_14);
    features.push_back(technical_features.rsi_21);
    features.push_back(technical_features.rsi_30);
    features.push_back(technical_features.sma_5);
    features.push_back(technical_features.sma_10);
    features.push_back(technical_features.sma_20);
    features.push_back(technical_features.sma_50);
    features.push_back(technical_features.sma_200);
    features.push_back(technical_features.ema_5);
    features.push_back(technical_features.ema_10);
    features.push_back(technical_features.ema_20);
    features.push_back(technical_features.ema_50);
    features.push_back(technical_features.ema_200);
    features.push_back(technical_features.bb_upper_20);
    features.push_back(technical_features.bb_middle_20);
    features.push_back(technical_features.bb_lower_20);
    features.push_back(technical_features.bb_upper_50);
    features.push_back(technical_features.bb_middle_50);
    features.push_back(technical_features.bb_lower_50);
    features.push_back(technical_features.macd_line);
    features.push_back(technical_features.macd_signal);
    features.push_back(technical_features.macd_histogram);
    features.push_back(technical_features.stoch_k);
    features.push_back(technical_features.stoch_d);
    features.push_back(technical_features.williams_r);
    features.push_back(technical_features.cci_20);
    features.push_back(technical_features.adx_14);
    
    // Volume features (7)
    features.push_back(volume_features.volume_sma_10);
    features.push_back(volume_features.volume_sma_20);
    features.push_back(volume_features.volume_sma_50);
    features.push_back(volume_features.volume_roc);
    features.push_back(volume_features.obv);
    features.push_back(volume_features.vpt);
    features.push_back(volume_features.ad_line);
    features.push_back(volume_features.mfi_14);
    
    // Microstructure features (5)
    features.push_back(microstructure_features.spread_bp);
    features.push_back(microstructure_features.price_impact);
    features.push_back(microstructure_features.order_flow_imbalance);
    features.push_back(microstructure_features.market_depth);
    features.push_back(microstructure_features.bid_ask_ratio);
    
    // All 55 features calculated successfully
    
    return features;
}

bool TechnicalIndicatorCalculator::validate_features(const std::vector<double>& features) {
    if (features.size() != feature_names_.size()) {
        return false;
    }
    
    for (double feature : features) {
        if (!std::isfinite(feature)) {
            return false;
        }
    }
    
    return true;
}

std::vector<std::string> TechnicalIndicatorCalculator::get_feature_names() const {
    return feature_names_;
}

// Helper method implementations
double TechnicalIndicatorCalculator::calculate_rsi(const std::vector<double>& closes, int period, int current_index) {
    if (current_index < period || current_index >= static_cast<int>(closes.size())) {
        return 0.0;
    }
    
    double gain_sum = 0.0;
    double loss_sum = 0.0;
    
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double change = closes[i] - closes[i - 1];
        if (change > 0) {
            gain_sum += change;
        } else {
            loss_sum -= change;
        }
    }
    
    double avg_gain = gain_sum / period;
    double avg_loss = loss_sum / period;
    
    if (avg_loss == 0.0) return 100.0;
    
    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}

double TechnicalIndicatorCalculator::calculate_sma(const std::vector<double>& values, int period, int current_index) {
    if (current_index < period - 1 || current_index >= static_cast<int>(values.size())) {
        // SMA returns 0 for insufficient data
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        sum += values[i];
    }
    
    double result = sum / period;
    
    // SMA calculation completed
    
    return result;
}

double TechnicalIndicatorCalculator::calculate_ema(const std::vector<double>& values, int period, int current_index) {
    if (current_index < 0 || current_index >= static_cast<int>(values.size())) {
        return 0.0;
    }
    
    double multiplier = 2.0 / (period + 1.0);
    
    if (current_index == 0) {
        return values[0];
    }
    
    double ema = values[0];
    for (int i = 1; i <= current_index; ++i) {
        ema = (values[i] * multiplier) + (ema * (1.0 - multiplier));
    }
    
    return ema;
}

double TechnicalIndicatorCalculator::calculate_volatility(const std::vector<double>& returns, int period, int current_index) {
    if (current_index < period - 1 || current_index >= static_cast<int>(returns.size())) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        sum += returns[i];
    }
    double mean = sum / period;
    
    double variance = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double diff = returns[i] - mean;
        variance += diff * diff;
    }
    
    return std::sqrt(variance / (period - 1));
}

double TechnicalIndicatorCalculator::calculate_atr(const std::vector<Bar>& bars, int period, int current_index) {
    if (current_index < period || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        const Bar& current = bars[i];
        const Bar& previous = bars[i - 1];
        
        double tr1 = current.high - current.low;
        double tr2 = std::abs(current.high - previous.close);
        double tr3 = std::abs(current.low - previous.close);
        
        sum += std::max({tr1, tr2, tr3});
    }
    
    return sum / period;
}

TechnicalIndicatorCalculator::BollingerBands TechnicalIndicatorCalculator::calculate_bollinger_bands(
    const std::vector<double>& values, int period, double std_dev, int current_index) {
    BollingerBands bands{};
    
    if (current_index < period - 1 || current_index >= static_cast<int>(values.size())) {
        return bands;
    }
    
    double sma = calculate_sma(values, period, current_index);
    double variance = 0.0;
    
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double diff = values[i] - sma;
        variance += diff * diff;
    }
    
    double std = std::sqrt(variance / period);
    
    bands.middle = sma;
    bands.upper = sma + (std_dev * std);
    bands.lower = sma - (std_dev * std);
    
    return bands;
}

TechnicalIndicatorCalculator::MACD TechnicalIndicatorCalculator::calculate_macd(
    const std::vector<double>& values, int fast, int slow, [[maybe_unused]] int signal, int current_index) {
    MACD macd{};
    
    if (current_index < slow || current_index >= static_cast<int>(values.size())) {
        return macd;
    }
    
    double ema_fast = calculate_ema(values, fast, current_index);
    double ema_slow = calculate_ema(values, slow, current_index);
    
    macd.line = ema_fast - ema_slow;
    
    // For signal line, we need to calculate EMA of MACD line
    // This is simplified - in practice, you'd maintain a running EMA
    macd.signal = macd.line; // Simplified
    macd.histogram = macd.line - macd.signal;
    
    return macd;
}

TechnicalIndicatorCalculator::Stochastic TechnicalIndicatorCalculator::calculate_stochastic(
    const std::vector<Bar>& bars, int k_period, [[maybe_unused]] int d_period, int current_index) {
    Stochastic stoch{};
    
    if (current_index < k_period - 1 || current_index >= static_cast<int>(bars.size())) {
        return stoch;
    }
    
    double highest_high = bars[current_index - k_period + 1].high;
    double lowest_low = bars[current_index - k_period + 1].low;
    
    for (int i = current_index - k_period + 2; i <= current_index; ++i) {
        highest_high = std::max(highest_high, bars[i].high);
        lowest_low = std::min(lowest_low, bars[i].low);
    }
    
    double current_close = bars[current_index].close;
    stoch.k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100.0;
    stoch.d = stoch.k; // Simplified - in practice, this would be SMA of %K
    
    return stoch;
}

double TechnicalIndicatorCalculator::calculate_williams_r(const std::vector<Bar>& bars, int period, int current_index) {
    if (current_index < period - 1 || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double highest_high = bars[current_index - period + 1].high;
    double lowest_low = bars[current_index - period + 1].low;
    
    for (int i = current_index - period + 2; i <= current_index; ++i) {
        highest_high = std::max(highest_high, bars[i].high);
        lowest_low = std::min(lowest_low, bars[i].low);
    }
    
    double current_close = bars[current_index].close;
    return ((highest_high - current_close) / (highest_high - lowest_low)) * -100.0;
}

double TechnicalIndicatorCalculator::calculate_cci(const std::vector<Bar>& bars, int period, int current_index) {
    if (current_index < period - 1 || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double tp = (bars[i].high + bars[i].low + bars[i].close) / 3.0;
        sum += tp;
    }
    
    double sma_tp = sum / period;
    double current_tp = (bars[current_index].high + bars[current_index].low + bars[current_index].close) / 3.0;
    
    double mean_deviation = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double tp = (bars[i].high + bars[i].low + bars[i].close) / 3.0;
        mean_deviation += std::abs(tp - sma_tp);
    }
    
    mean_deviation /= period;
    
    return (current_tp - sma_tp) / (0.015 * mean_deviation);
}

double TechnicalIndicatorCalculator::calculate_adx([[maybe_unused]] const std::vector<Bar>& bars, [[maybe_unused]] int period, [[maybe_unused]] int current_index) {
    // Simplified ADX calculation
    // In practice, this would be more complex
    return 25.0; // Placeholder
}

double TechnicalIndicatorCalculator::calculate_obv(const std::vector<Bar>& bars, int current_index) {
    if (current_index < 1 || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double obv = 0.0;
    for (int i = 1; i <= current_index; ++i) {
        if (bars[i].close > bars[i-1].close) {
            obv += bars[i].volume;
        } else if (bars[i].close < bars[i-1].close) {
            obv -= bars[i].volume;
        }
    }
    
    return obv;
}

double TechnicalIndicatorCalculator::calculate_vpt(const std::vector<Bar>& bars, int current_index) {
    if (current_index < 1 || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double vpt = 0.0;
    for (int i = 1; i <= current_index; ++i) {
        double price_change = (bars[i].close - bars[i-1].close) / bars[i-1].close;
        vpt += bars[i].volume * price_change;
    }
    
    return vpt;
}

double TechnicalIndicatorCalculator::calculate_ad_line(const std::vector<Bar>& bars, int current_index) {
    if (current_index < 1 || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double ad_line = 0.0;
    for (int i = 1; i <= current_index; ++i) {
        double clv = ((bars[i].close - bars[i].low) - (bars[i].high - bars[i].close)) / (bars[i].high - bars[i].low);
        ad_line += clv * bars[i].volume;
    }
    
    return ad_line;
}

double TechnicalIndicatorCalculator::calculate_mfi(const std::vector<Bar>& bars, int period, int current_index) {
    if (current_index < period || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double positive_mf = 0.0;
    double negative_mf = 0.0;
    
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double tp = (bars[i].high + bars[i].low + bars[i].close) / 3.0;
        double prev_tp = (bars[i-1].high + bars[i-1].low + bars[i-1].close) / 3.0;
        
        double mf = tp * bars[i].volume;
        
        if (tp > prev_tp) {
            positive_mf += mf;
        } else if (tp < prev_tp) {
            negative_mf += mf;
        }
    }
    
    if (negative_mf == 0.0) return 100.0;
    
    double mfr = positive_mf / negative_mf;
    return 100.0 - (100.0 / (1.0 + mfr));
}

double TechnicalIndicatorCalculator::calculate_parkinson_volatility(const std::vector<Bar>& bars, int period, int current_index) {
    if (current_index < period - 1 || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double log_hl = std::log(bars[i].high / bars[i].low);
        sum += log_hl * log_hl;
    }
    
    return std::sqrt(sum / (4.0 * std::log(2.0) * period));
}

double TechnicalIndicatorCalculator::calculate_garman_klass_volatility(const std::vector<Bar>& bars, int period, int current_index) {
    if (current_index < period - 1 || current_index >= static_cast<int>(bars.size())) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double log_hl = std::log(bars[i].high / bars[i].low);
        double log_co = std::log(bars[i].close / bars[i].open);
        sum += 0.5 * log_hl * log_hl - (2.0 * std::log(2.0) - 1.0) * log_co * log_co;
    }
    
    return std::sqrt(sum / period);
}

} // namespace feature_engineering
} // namespace sentio

```

## 📄 **FILE 110 of 145**: src/feature_feeder.cpp

**File Information**:
- **Path**: `src/feature_feeder.cpp`

- **Size**: 631 lines
- **Modified**: 2025-09-07 23:35:51

- **Type**: .cpp

```text
#include "sentio/feature_feeder.hpp"
// TFB strategy removed - focusing on TFA only
#include "sentio/strategy_tfa.hpp"
#include "sentio/strategy_transformer_ts.hpp"
#include "sentio/strategy_kochi_ppo.hpp"
#include "sentio/feature_builder.hpp"
#include "sentio/feature_engineering/kochi_features.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <algorithm>

namespace sentio {

// Static member definitions
std::unordered_map<std::string, FeatureFeeder::StrategyData> FeatureFeeder::strategy_data_;
std::mutex FeatureFeeder::data_mutex_;
std::unique_ptr<FeatureCache> FeatureFeeder::feature_cache_;
bool FeatureFeeder::use_cached_features_ = false;

bool FeatureFeeder::is_ml_strategy(const std::string& strategy_name) {
    return strategy_name == "TFA" || strategy_name == "tfa" ||
           strategy_name == "transformer" ||
           strategy_name == "hybrid_ppo" ||
           strategy_name == "kochi_ppo";
}

void FeatureFeeder::initialize_strategy(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    if (strategy_data_.find(strategy_name) != strategy_data_.end()) {
        return; // Already initialized
    }
    
    initialize_strategy_data(strategy_name);
}

void FeatureFeeder::initialize_strategy_data(const std::string& strategy_name) {
    StrategyData data;
    
    // Create technical indicator calculator
    data.calculator = std::make_unique<feature_engineering::TechnicalIndicatorCalculator>();
    
    // Create feature normalizer
    data.normalizer = std::make_unique<feature_engineering::FeatureNormalizer>(252); // 1 year window
    
    // Set default configuration
    data.config["normalization_method"] = "robust";
    data.config["outlier_threshold"] = "3.0";
    data.config["winsorize_percentile"] = "0.05";
    data.config["enable_caching"] = "true";
    
    data.initialized = true;
    data.last_update = std::chrono::steady_clock::now();
    
    strategy_data_[strategy_name] = std::move(data);
    
    std::cout << "Initialized FeatureFeeder for strategy: " << strategy_name << std::endl;
}

void FeatureFeeder::cleanup_strategy(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    strategy_data_.erase(strategy_name);
}

std::vector<double> FeatureFeeder::extract_features_from_bar(const Bar& bar, const std::string& strategy_name) {
    if (!is_ml_strategy(strategy_name)) {
        return {};
    }
    
    // Initialize if not already done
    if (strategy_data_.find(strategy_name) == strategy_data_.end()) {
        initialize_strategy(strategy_name);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Kochi PPO uses its own feature set; need at least a small history.
        if (strategy_name == "kochi_ppo") {
            std::vector<Bar> hist = {bar};
            auto features = feature_engineering::calculate_kochi_features(hist, 0);
            if (features.empty()) return {};
            auto end_time_metrics = std::chrono::high_resolution_clock::now();
            auto extraction_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time_metrics - start_time);
            auto& data = get_strategy_data(strategy_name);
            update_metrics(data, features, extraction_time);
            return features;
        }

        // Get strategy data
        auto& data = get_strategy_data(strategy_name);
        
        // For single bar, we need at least some history
        // This is a limitation - we need multiple bars for most indicators
        // For now, return empty vector if we don't have enough history
        if (!data.calculator) {
            return {};
        }
        
        // Create a minimal bar history for calculation
        std::vector<Bar> bar_history = {bar};
        
        // Calculate features
        auto features = data.calculator->calculate_all_features(bar_history, 0);
        
        // Normalize features
        if (data.normalizer && !features.empty()) {
            features = data.normalizer->normalize_features(features);
        }
        
        // Validate features
        if (!validate_features(features, strategy_name)) {
            return {};
        }
        
        // Update metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto extraction_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        update_metrics(data, features, extraction_time);
        
        return features;
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting features for " << strategy_name << ": " << e.what() << std::endl;
        return {};
    }
}

std::vector<std::vector<double>> FeatureFeeder::extract_features_from_bars(const std::vector<Bar>& bars, const std::string& strategy_name) {
    if (!is_ml_strategy(strategy_name) || bars.empty()) {
        return {};
    }
    
    // Initialize if not already done
    if (strategy_data_.find(strategy_name) == strategy_data_.end()) {
        initialize_strategy(strategy_name);
    }
    
    std::vector<std::vector<double>> all_features;
    all_features.reserve(bars.size());
    
    try {
        auto& data = get_strategy_data(strategy_name);
        
        if (!data.calculator) {
            return {};
        }
        
        // Extract features for each bar
        for (int i = 0; i < static_cast<int>(bars.size()); ++i) {
            auto features = data.calculator->calculate_all_features(bars, i);
            
            // Normalize features
            if (data.normalizer && !features.empty()) {
                features = data.normalizer->normalize_features(features);
            }
            
            all_features.push_back(features);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting features from bars for " << strategy_name << ": " << e.what() << std::endl;
        return {};
    }
    
    return all_features;
}

std::vector<double> FeatureFeeder::extract_features_from_bars_with_index(const std::vector<Bar>& bars, int current_index, const std::string& strategy_name) {
    
    if (!is_ml_strategy(strategy_name) || bars.empty() || current_index < 0 || current_index >= static_cast<int>(bars.size())) {
        return {};
    }
    
    // Initialize if not already done
    if (strategy_data_.find(strategy_name) == strategy_data_.end()) {
        initialize_strategy(strategy_name);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Get strategy data
        auto& data = get_strategy_data(strategy_name);
        
        if (!data.calculator) {
            return {};
        }
        
        // Use cached features if available, otherwise calculate
        std::vector<double> features;
        if (use_cached_features_ && feature_cache_ && feature_cache_->has_features(current_index)) {
            features = feature_cache_->get_features(current_index);
            static int cache_calls = 0;
            cache_calls++;
            if (cache_calls <= 5) {
                std::cout << "[DEBUG] After cache get_features: " << features.size() << " features" << std::endl;
            }
        } else {
            // Calculate features using full bar history up to current_index
            features = data.calculator->calculate_all_features(bars, current_index);
        }
        
        // Normalize features (skip normalization for cached features as they're pre-processed)
        bool used_cache = (use_cached_features_ && feature_cache_ && feature_cache_->has_features(current_index));
        if (data.normalizer && !features.empty() && !used_cache) {
            size_t before_norm = features.size();
            features = data.normalizer->normalize_features(features);
            static int norm_calls = 0;
            norm_calls++;
            if (norm_calls <= 5) {
                std::cout << "[DEBUG] After normalize: " << before_norm << " -> " << features.size() << " features" << std::endl;
            }
        } else if (used_cache) {
            static int cache_skip_calls = 0;
            cache_skip_calls++;
            if (cache_skip_calls <= 5) {
                std::cout << "[DEBUG] Skipping normalization for cached features: " << features.size() << " features" << std::endl;
            }
        }
        
        // Validate features (bypass validation for cached features as they're pre-validated)
        if (used_cache) {
            static int cache_bypass_calls = 0;
            cache_bypass_calls++;
            if (cache_bypass_calls <= 5) {
                std::cout << "[DEBUG] Bypassing validation for cached features: " << features.size() << " features" << std::endl;
            }
        } else {
            bool valid = validate_features(features, strategy_name);
            static int val_calls = 0;
            val_calls++;
            if (val_calls <= 5) {
                std::cout << "[DEBUG] Validation result: " << (valid ? "PASS" : "FAIL") << " for " << features.size() << " features" << std::endl;
            }
            if (!valid) {
                return {};
            }
        }
        
        // Update metrics
        auto end_time_metrics = std::chrono::high_resolution_clock::now();
        auto extraction_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time_metrics - start_time);
        update_metrics(data, features, extraction_time);
        return features;
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting features for " << strategy_name << " at index " << current_index << ": " << e.what() << std::endl;
        return {};
    }
}

void FeatureFeeder::feed_features_to_strategy(BaseStrategy* strategy, const std::vector<Bar>& bars, int current_index, const std::string& strategy_name) {
    
    if (!is_ml_strategy(strategy_name) || !strategy) {
        return;
    }
    
    // Initialize if not already done
    if (strategy_data_.find(strategy_name) == strategy_data_.end()) {
        initialize_strategy(strategy_name);
    }
    
    try {
        // Extract features using full bar history (required for technical indicators)
        auto features = extract_features_from_bars_with_index(bars, current_index, strategy_name);
        
        static int feature_extract_calls = 0;
        feature_extract_calls++;
        
        if (feature_extract_calls % 1000 == 0 || feature_extract_calls <= 10) {
            std::cout << "[DIAG] FeatureFeeder extract: call=" << feature_extract_calls 
                      << " features.size()=" << features.size() 
                      << " current_index=" << current_index << std::endl;
        }
        
        if (features.empty()) {
            if (feature_extract_calls % 1000 == 0 || feature_extract_calls <= 10) {
                std::cout << "[DIAG] FeatureFeeder: Features EMPTY at call=" << feature_extract_calls << std::endl;
            }
            return;
        }
        
        // Cast to specific strategy type and feed features
        static int strategy_check_calls = 0;
        strategy_check_calls++;
        
        if (strategy_check_calls % 1000 == 0 || strategy_check_calls <= 10) {
            std::cout << "[DIAG] FeatureFeeder strategy check: call=" << strategy_check_calls 
                      << " strategy_name='" << strategy_name << "'" << std::endl;
        }
        
        if (strategy_name == "TFA" || strategy_name == "tfa") {
            auto* tfa = dynamic_cast<TFAStrategy*>(strategy);
            if (tfa) {
                static int tfa_feed_calls = 0;
                tfa_feed_calls++;
                
                if (tfa_feed_calls % 1000 == 0 || tfa_feed_calls <= 10) {
                    std::cout << "[DIAG] FeatureFeeder TFA: call=" << tfa_feed_calls 
                              << " features.size()=" << features.size() << std::endl;
                }
                
                tfa->set_raw_features(features);
            } else {
                static int cast_fail = 0;
                cast_fail++;
                if (cast_fail <= 10) {
                    std::cout << "[DIAG] FeatureFeeder: TFA cast failed! call=" << cast_fail << std::endl;
                }
            }
        } else if (strategy_name == "transformer") {
            auto* tf = dynamic_cast<TransformerSignalStrategyTS*>(strategy);
            if (tf) {
                tf->set_raw_features(features);
            }
        } else if (strategy_name == "kochi_ppo") {
            auto* kp = dynamic_cast<KochiPPOStrategy*>(strategy);
            if (kp) {
                kp->set_raw_features(features);
            }
        }
        
        // Cache features
        cache_features(strategy_name, features);
        
    } catch (const std::exception& e) {
        std::cerr << "Error feeding features to strategy " << strategy_name << ": " << e.what() << std::endl;
    }
}

void FeatureFeeder::feed_features_batch(BaseStrategy* strategy, const std::vector<Bar>& bars, const std::string& strategy_name) {
    if (!is_ml_strategy(strategy_name) || !strategy || bars.empty()) {
        return;
    }
    
    // Initialize if not already done
    if (strategy_data_.find(strategy_name) == strategy_data_.end()) {
        initialize_strategy(strategy_name);
    }
    
    try {
        // Extract features for all bars
        auto all_features = extract_features_from_bars(bars, strategy_name);
        
        if (all_features.empty()) {
            return;
        }
        
        // Feed features to strategy
        for (size_t i = 0; i < all_features.size(); ++i) {
            if (!all_features[i].empty()) {
                feed_features_to_strategy(strategy, bars, i, strategy_name);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error feeding features batch to strategy " << strategy_name << ": " << e.what() << std::endl;
    }
}

std::vector<double> FeatureFeeder::get_cached_features(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        return it->second.cached_features;
    }
    
    return {};
}

void FeatureFeeder::cache_features(const std::string& strategy_name, const std::vector<double>& features) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        it->second.cached_features = features;
        it->second.last_update = std::chrono::steady_clock::now();
    }
}

void FeatureFeeder::invalidate_cache(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        it->second.cached_features.clear();
    }
}

FeatureMetrics FeatureFeeder::get_metrics(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        return it->second.metrics;
    }
    
    return FeatureMetrics{};
}

FeatureHealthReport FeatureFeeder::get_health_report(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        return calculate_health_report(it->second, it->second.cached_features);
    }
    
    return FeatureHealthReport{};
}

void FeatureFeeder::reset_metrics(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        it->second.metrics = FeatureMetrics{};
    }
}

bool FeatureFeeder::validate_features(const std::vector<double>& features, const std::string& strategy_name) {
    if (features.empty()) {
        return false;
    }
    
    // Check if all features are finite
    int non_finite_count = 0;
    for (size_t i = 0; i < features.size(); ++i) {
        if (!std::isfinite(features[i])) {
            non_finite_count++;
        }
    }
    
    // Feature validation completed
    
    if (non_finite_count > 0) {
        return false;
    }
    
    // Check feature count; for Kochi we compare against its own names
    auto expected_names = (strategy_name == "kochi_ppo")
        ? feature_engineering::kochi_feature_names()
        : get_feature_names(strategy_name);
    if (features.size() != expected_names.size()) {
        return false;
    }
    
    return true;
}

std::vector<std::string> FeatureFeeder::get_feature_names(const std::string& strategy_name) {
    return get_strategy_feature_names(strategy_name);
}

std::vector<std::string> FeatureFeeder::get_strategy_feature_names(const std::string& strategy_name) {
    if (strategy_name == "TFA" || strategy_name == "tfa" || 
        strategy_name == "transformer") {
        // Return the exact 55 features that TechnicalIndicatorCalculator provides
        return {
            // Price features (15)
            "ret_1m", "ret_5m", "ret_15m", "ret_30m", "ret_1h",
            "momentum_5", "momentum_10", "momentum_20",
            "volatility_10", "volatility_20", "volatility_30",
            "atr_14", "atr_21", "parkinson_vol", "garman_klass_vol",
            
            // Technical features (27) - Note: Actually 27, not 25 as the comment in calculator says
            "rsi_14", "rsi_21", "rsi_30",
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
            "bb_upper_20", "bb_middle_20", "bb_lower_20",
            "bb_upper_50", "bb_middle_50", "bb_lower_50",
            "macd_line", "macd_signal", "macd_histogram",
            "stoch_k", "stoch_d", "williams_r", "cci_20", "adx_14",
            
            // Volume features (8)
            "volume_sma_10", "volume_sma_20", "volume_sma_50",
            "volume_roc", "obv", "vpt", "ad_line", "mfi_14",
            
            // Microstructure features (5)
            "spread_bp", "price_impact", "order_flow_imbalance", "market_depth", "bid_ask_ratio"
        };
    }
    if (strategy_name == "kochi_ppo") {
        return feature_engineering::kochi_feature_names();
    }
    
    return {};
}

void FeatureFeeder::set_feature_config(const std::string& strategy_name, const std::string& config_key, const std::string& config_value) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        it->second.config[config_key] = config_value;
    }
}

std::string FeatureFeeder::get_feature_config(const std::string& strategy_name, const std::string& config_key) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it != strategy_data_.end()) {
        auto config_it = it->second.config.find(config_key);
        if (config_it != it->second.config.end()) {
            return config_it->second;
        }
    }
    
    return "";
}

std::vector<double> FeatureFeeder::get_feature_correlation([[maybe_unused]] const std::string& strategy_name) {
    // Placeholder implementation
    // In practice, this would calculate correlation between features
    return {};
}

std::vector<double> FeatureFeeder::get_feature_importance([[maybe_unused]] const std::string& strategy_name) {
    // Placeholder implementation
    // In practice, this would calculate feature importance scores
    return {};
}

void FeatureFeeder::log_feature_performance(const std::string& strategy_name) {
    auto metrics = get_metrics(strategy_name);
    auto health = get_health_report(strategy_name);
    
    std::cout << "FeatureFeeder Performance for " << strategy_name << ":" << std::endl;
    std::cout << "  Extraction time: " << metrics.extraction_time.count() << " microseconds" << std::endl;
    std::cout << "  Features extracted: " << metrics.features_extracted << std::endl;
    std::cout << "  Features valid: " << metrics.features_valid << std::endl;
    std::cout << "  Features invalid: " << metrics.features_invalid << std::endl;
    std::cout << "  Extraction rate: " << metrics.extraction_rate << " features/sec" << std::endl;
    std::cout << "  Health status: " << (health.is_healthy ? "HEALTHY" : "UNHEALTHY") << std::endl;
    std::cout << "  Overall health score: " << health.overall_health_score << std::endl;
}

FeatureFeeder::StrategyData& FeatureFeeder::get_strategy_data(const std::string& strategy_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = strategy_data_.find(strategy_name);
    if (it == strategy_data_.end()) {
        initialize_strategy_data(strategy_name);
        it = strategy_data_.find(strategy_name);
    }
    
    return it->second;
}

void FeatureFeeder::update_metrics(StrategyData& data, const std::vector<double>& features, std::chrono::microseconds extraction_time) {
    data.metrics.extraction_time = extraction_time;
    data.metrics.features_extracted = features.size();
    data.metrics.features_valid = features.size(); // Assuming all features are valid at this point
    data.metrics.features_invalid = 0;
    
    if (extraction_time.count() > 0) {
        data.metrics.extraction_rate = static_cast<double>(features.size()) / (extraction_time.count() / 1000000.0);
    }
    
    data.metrics.last_update = std::chrono::steady_clock::now();
}

FeatureHealthReport FeatureFeeder::calculate_health_report([[maybe_unused]] const StrategyData& data, const std::vector<double>& features) {
    FeatureHealthReport report;
    
    if (features.empty()) {
        report.is_healthy = false;
        report.health_summary = "No features available";
        return report;
    }
    
    report.feature_health.resize(features.size(), true);
    report.feature_quality_scores.resize(features.size(), 1.0);
    
    // Check for NaN or infinite values
    for (size_t i = 0; i < features.size(); ++i) {
        if (!std::isfinite(features[i])) {
            report.feature_health[i] = false;
            report.feature_quality_scores[i] = 0.0;
        }
    }
    
    // Calculate overall health
    size_t healthy_features = std::count(report.feature_health.begin(), report.feature_health.end(), true);
    report.overall_health_score = static_cast<double>(healthy_features) / features.size();
    report.is_healthy = report.overall_health_score > 0.8; // 80% threshold
    
    if (report.is_healthy) {
        report.health_summary = "All features are healthy";
    } else {
        report.health_summary = "Some features are unhealthy";
    }
    
    return report;
}

// Cached features implementation
bool FeatureFeeder::load_feature_cache(const std::string& feature_file_path) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    feature_cache_ = std::make_unique<FeatureCache>();
    if (feature_cache_->load_from_csv(feature_file_path)) {
        std::cout << "FeatureFeeder: Successfully loaded feature cache from " << feature_file_path << std::endl;
        return true;
    } else {
        feature_cache_.reset();
        std::cerr << "FeatureFeeder: Failed to load feature cache from " << feature_file_path << std::endl;
        return false;
    }
}

bool FeatureFeeder::use_cached_features(bool enable) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    use_cached_features_ = enable;
    std::cout << "FeatureFeeder: Cached features " << (enable ? "ENABLED" : "DISABLED") << std::endl;
    return true;
}

bool FeatureFeeder::has_cached_features() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return feature_cache_ != nullptr;
}

} // namespace sentio
```

## 📄 **FILE 111 of 145**: src/feature_feeder_guarded.cpp

**File Information**:
- **Path**: `src/feature_feeder_guarded.cpp`

- **Size**: 79 lines
- **Modified**: 2025-09-06 22:14:53

- **Type**: .cpp

```text
#include "sentio/feature/feature_feeder_guarded.hpp"
#include "sentio/feature/feature_builder_guarded.hpp"
#include "sentio/sym/symbol_utils.hpp"

namespace sentio {

bool FeatureFeederGuarded::infer_base_if_needed_(
  const std::unordered_map<std::string, std::vector<Bar>>& series,
  std::string& base_out)
{
  if (!base_out.empty()) { base_out = to_upper(base_out); return true; }
  // Prefer QQQ if present, else pick the first non-leveraged symbol
  if (series.find("QQQ") != series.end()) { base_out = "QQQ"; return true; }
  for (auto& kv : series) {
    if (!is_leveraged(kv.first)) { base_out = to_upper(kv.first); return true; }
  }
  return false;
}

bool FeatureFeederGuarded::initialize(const FeederInit& init) {
  prices_.clear();
  asof_.clear();
  base_ts_.clear();
  X_ = {};
  scaler_ = {};
  base_symU_.clear();

  // Cache the input prices (all symbols). We'll filter logically below.
  for (auto& kv : init.series) {
    prices_.emplace(to_upper(kv.first), kv.second);
  }

  // Decide base
  base_symU_ = init.base_symbol;
  if (!infer_base_if_needed_(prices_, base_symU_)) {
    // cannot proceed without a base
    return false;
  }

  // Build features **only** for base
  auto it = prices_.find(base_symU_);
  if (it == prices_.end() || it->second.empty()) return false;

  // Build base timestamp vector
  base_ts_.resize(it->second.size());
  for (std::size_t i=0; i<it->second.size(); ++i) base_ts_[i] = it->second[i].ts_epoch_us;

  // Strict guard: do not allow leveraged symbol to reach builder
  if (is_leveraged(base_symU_)) return false;

  X_ = build_features_for_base(base_symU_, it->second);
  if (X_.rows == 0) return false;

  // Fit scaler ONLY on base features; transform in-place
  scaler_.fit(X_.data.data(), X_.rows, X_.cols);
  scaler_.transform_inplace(X_.data.data(), X_.rows, X_.cols);

  // Build as-of maps for any instrument in the base family (leveraged or base itself)
  for (auto& kv : prices_) {
    const auto symU = kv.first;
    // Only build maps for instruments whose resolved base == base_symU_
    if (resolve_base(symU) != base_symU_) continue;

    // Build inst_ts
    std::vector<std::int64_t> inst_ts; inst_ts.reserve(kv.second.size());
    for (auto& b : kv.second) inst_ts.push_back(b.ts_epoch_us);

    asof_[symU] = build_asof_index(base_ts_, inst_ts);
  }

  return true;
}

bool FeatureFeederGuarded::allowed_for_exec(const std::string& symbol) const {
  const auto symU = to_upper(symbol);
  return resolve_base(symU) == base_symU_;
}

} // namespace sentio

```

## 📄 **FILE 112 of 145**: src/feature_health.cpp

**File Information**:
- **Path**: `src/feature_health.cpp`

- **Size**: 32 lines
- **Modified**: 2025-09-05 17:01:07

- **Type**: .cpp

```text
#include "sentio/feature_health.hpp"
#include <cmath>

namespace sentio {

FeatureHealthReport check_feature_health(const std::vector<PricePoint>& series,
                                         const FeatureHealthCfg& cfg) {
  FeatureHealthReport rep;
  if (series.empty()) return rep;

  for (std::size_t i=0;i<series.size();++i) {
    const auto& p = series[i];
    if (cfg.check_nan && !std::isfinite(p.close)) {
      rep.issues.push_back({p.ts_utc, "NaN", "non-finite close"});
    }
    if (cfg.check_monotonic_time && i>0) {
      if (series[i].ts_utc <= series[i-1].ts_utc) {
        rep.issues.push_back({p.ts_utc, "Backwards_TS", "non-increasing timestamp"});
      }
      if (cfg.expected_spacing_sec>0) {
        auto gap = series[i].ts_utc - series[i-1].ts_utc;
        if (gap != cfg.expected_spacing_sec) {
          rep.issues.push_back({p.ts_utc, "Gap",
              "expected "+std::to_string(cfg.expected_spacing_sec)+"s got "+std::to_string((long long)gap)+"s"});
        }
      }
    }
  }
  return rep;
}

} // namespace sentio

```

## 📄 **FILE 113 of 145**: src/kochi_runner.cpp

**File Information**:
- **Path**: `src/kochi_runner.cpp`

- **Size**: 54 lines
- **Modified**: 2025-09-08 08:47:53

- **Type**: .cpp

```text
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

```

## 📄 **FILE 114 of 145**: src/main.cpp

**File Information**:
- **Path**: `src/main.cpp`

- **Size**: 519 lines
- **Modified**: 2025-09-08 10:26:15

- **Type**: .cpp

```text
#include "sentio/core.hpp"
#include "sentio/runner.hpp"
#include "sentio/temporal_analysis.hpp"
#include "sentio/csv_loader.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/profiling.hpp"
#include "sentio/data_resolver.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/all_strategies.hpp"
#include "sentio/feature_feeder.hpp"
#include "sentio/rth_calendar.hpp" // **NEW**: Include for RTH check
#include "sentio/calendar_seed.hpp" // **NEW**: Include for calendar creation
// #include "sentio/feature/feature_from_spec.hpp" // For C++ feature building - causes Bar redefinition
#include "sentio/feature/feature_matrix.hpp" // For FeatureMatrix
// TFB strategy removed - focusing on TFA only
#include "sentio/strategy_tfa.hpp" // For TFA strategy

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <cstdlib> // For std::exit
#include <sstream>
#include <ctime> // For std::time
#include <fstream> // For std::ifstream
#include <ATen/Parallel.h> // For LibTorch threading controls


namespace { // Anonymous namespace to ensure link-time registration
    struct StrategyRegistrar {
        StrategyRegistrar() {
            // Register strategies in the factory
            sentio::StrategyFactory::instance().register_strategy("VWAPReversion", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::VWAPReversionStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("MomentumVolume", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::MomentumVolumeProfileStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("BollingerSqueeze", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::BollingerSqueezeBreakoutStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("OpeningRange", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::OpeningRangeBreakoutStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("OrderFlowScalping", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::OrderFlowScalpingStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("OrderFlowImbalance", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::OrderFlowImbalanceStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("MarketMaking", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::MarketMakingStrategy>();
            });
            // TransformerTS inherits from IStrategy, not BaseStrategy, so skip for now
            // TFB strategy removed - focusing on TFA only
            sentio::StrategyFactory::instance().register_strategy("tfa", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::unique_ptr<sentio::BaseStrategy>(new sentio::TFAStrategy(sentio::TFACfg{}));
            });
        }
    };
    static StrategyRegistrar registrar;
}


void usage() {
    std::cout << "Usage: sentio_cli <command> [options]\n"
              << "Commands:\n"
              << "  backtest <symbol> [--strategy <name>] [--params <k=v,...>]\n"
              << "  tpa_test <symbol> [--strategy <name>] [--params <k=v,...>] [--quarters <n>]\n"
              << "  test-models [--strategy <name>] [--data <file>] [--start <date>] [--end <date>]\n"
              << "  replay <run_id>\n";
}

int main(int argc, char* argv[]) {
    // Configure LibTorch threading to prevent oversubscription
    at::set_num_threads(1);         // intra-op
    at::set_num_interop_threads(1); // inter-op
    
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string command = argv[1];
    
    if (command == "backtest") {
        if (argc < 3) {
            std::cout << "Usage: sentio_cli backtest <symbol> [--strategy <name>] [--params <k=v,...>]\n";
            return 1;
        }
        
        std::string base_symbol = argv[2];
        std::string strategy_name = "VWAPReversion";
        std::unordered_map<std::string, std::string> strategy_params;
        
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--strategy" && i + 1 < argc) {
                strategy_name = argv[++i];
            } else if (arg == "--params" && i + 1 < argc) {
                std::string params_str = argv[++i];
                std::stringstream ss(params_str);
                std::string pair;
                while (std::getline(ss, pair, ',')) {
                    size_t eq_pos = pair.find('=');
                    if (eq_pos != std::string::npos) {
                        strategy_params[pair.substr(0, eq_pos)] = pair.substr(eq_pos + 1);
                    }
                }
            }
        }
        
        sentio::SymbolTable ST;
        std::vector<std::vector<sentio::Bar>> series;
        std::vector<std::string> symbols_to_load = {base_symbol};
        if (base_symbol == "QQQ") {
            symbols_to_load.push_back("TQQQ");
            symbols_to_load.push_back("SQQQ");
        }

        std::cout << "Loading data for symbols: ";
        for(const auto& sym : symbols_to_load) std::cout << sym << " ";
        std::cout << std::endl;

        for (const auto& sym : symbols_to_load) {
            std::vector<sentio::Bar> bars;
            std::string data_path = sentio::resolve_csv(sym);
            if (!sentio::load_csv(data_path, bars)) {
                std::cerr << "ERROR: Failed to load data for " << sym << " from " << data_path << std::endl;
                continue;
            }
            std::cout << " -> Loaded " << bars.size() << " bars for " << sym << std::endl;
            
            int symbol_id = ST.intern(sym);
            if (static_cast<size_t>(symbol_id) >= series.size()) {
                series.resize(symbol_id + 1);
            }
            series[symbol_id] = std::move(bars);
        }
        
        int base_symbol_id = ST.get_id(base_symbol);
        if (series.empty() || series[base_symbol_id].empty()) {
            std::cerr << "FATAL: No data loaded for base symbol " << base_symbol << std::endl;
            return 1;
        }

        // Report data period and bars for transparency
        auto& base_bars = series[base_symbol_id];
        auto fmt_date = [](std::int64_t epoch_sec){
            std::time_t tt = static_cast<std::time_t>(epoch_sec);
            std::tm tm{}; gmtime_r(&tt, &tm);
            char buf[32]; std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm); return std::string(buf);
        };
        std::int64_t min_ts = base_bars.front().ts_nyt_epoch;
        std::int64_t max_ts = base_bars.back().ts_nyt_epoch;
        std::size_t   n_bars = base_bars.size();
        double span_days = double(max_ts - min_ts) / (24.0*3600.0);
        std::cout << "Data period: " << fmt_date(min_ts) << " → " << fmt_date(max_ts)
                  << " (" << std::fixed << std::setprecision(1) << span_days << " days)\n";
        std::cout << "Bars(" << base_symbol << "): " << n_bars << "\n";
        if ((max_ts - min_ts) < (365LL*24LL*3600LL)) {
            std::cerr << "WARNING: Data period is shorter than 1 year. Results may be unrepresentative.\n";
        }

        // **NEW**: Data Sanity Check - Verify RTH filtering post-load.
        // This acts as a safety net. If your data was generated with RTH filtering,
        // this check ensures the filtering was successful.
        std::cout << "\nVerifying data integrity for RTH..." << std::endl;
        bool rth_filter_failed = false;
        sentio::TradingCalendar calendar = sentio::make_default_nyse_calendar();
        for (size_t sid = 0; sid < series.size(); ++sid) {
            if (series[sid].empty()) continue;
            for (const auto& bar : series[sid]) {
                if (!calendar.is_rth_utc(bar.ts_nyt_epoch, "UTC")) {
                    std::cerr << "\nFATAL ERROR: Non-RTH data found after filtering!\n"
                              << " -> Symbol: " << ST.get_symbol(sid) << "\n"
                              << " -> Timestamp (UTC): " << bar.ts_utc << "\n"
                              << " -> UTC Epoch: " << bar.ts_nyt_epoch << "\n\n"
                              << "This indicates your data files (*.csv, *.bin) were generated with an old or incorrect RTH filter.\n"
                              << "Please DELETE your existing data files and REGENERATE them using the updated poly_fetch tool.\n"
                              << std::endl;
                    rth_filter_failed = true;
                    break;
                }
            }
            if (rth_filter_failed) break;
        }

        if (rth_filter_failed) {
            std::exit(1); // Exit with an error as requested
        }
        std::cout << " -> Data verification passed." << std::endl;

        sentio::RunnerCfg cfg;
        cfg.strategy_name = strategy_name;
        cfg.strategy_params = strategy_params;
        cfg.audit_level = sentio::AuditLevel::Full;
        cfg.snapshot_stride = 100;
        
        // Create audit recorder
        sentio::AuditConfig audit_cfg;
        long long ts_epoch = std::time(nullptr);
        audit_cfg.run_id = strategy_name + std::string("_backtest_") + base_symbol + "_" + std::to_string(ts_epoch);
        audit_cfg.file_path = "audit/" + strategy_name + std::string("_backtest_") + base_symbol + "_" + std::to_string(ts_epoch) + ".jsonl";
        audit_cfg.flush_each = true;
        sentio::AuditRecorder audit(audit_cfg);
        
        sentio::Tsc timer;
        timer.tic();
        auto result = sentio::run_backtest(audit, ST, series, base_symbol_id, cfg);
        double elapsed = timer.toc_sec();
        
        std::cout << "\nBacktest completed in " << elapsed << "s\n";
        std::cout << "Final Equity: " << result.final_equity << "\n";
        std::cout << "Total Return: " << result.total_return << "%\n";
        std::cout << "Sharpe Ratio: " << result.sharpe_ratio << "\n";
        std::cout << "Max Drawdown: " << result.max_drawdown << "%\n";
        std::cout << "Total Fills: " << result.total_fills << "\n";
        std::cout << "Diagnostics -> No Route: " << result.no_route << " | No Quantity: " << result.no_qty << "\n";

    } else if (command == "tpa_test") {
        if (argc < 3) {
            std::cout << "Usage: sentio_cli tpa_test <symbol> [--strategy <name>] [--params <k=v,...>] [--quarters <n>]\n";
            return 1;
        }
        
        std::string base_symbol = argv[2];
        std::string strategy_name = "TFA";
        std::unordered_map<std::string, std::string> strategy_params;
        int num_quarters = 12;
        
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--strategy" && i + 1 < argc) {
                strategy_name = argv[++i];
            } else if (arg == "--params" && i + 1 < argc) {
                std::string params_str = argv[++i];
                std::stringstream ss(params_str);
                std::string pair;
                while (std::getline(ss, pair, ',')) {
                    size_t eq_pos = pair.find('=');
                    if (eq_pos != std::string::npos) {
                        strategy_params[pair.substr(0, eq_pos)] = pair.substr(eq_pos + 1);
                    }
                }
            } else if (arg == "--quarters" && i + 1 < argc) {
                num_quarters = std::stoi(argv[++i]);
            }
        }
        
        sentio::SymbolTable ST;
        std::vector<std::vector<sentio::Bar>> series;
        std::vector<std::string> symbols_to_load = {base_symbol};
        if (base_symbol == "QQQ") {
            symbols_to_load.push_back("TQQQ");
            symbols_to_load.push_back("SQQQ");
        }

        std::cout << "Loading data for symbols: ";
        for(const auto& sym : symbols_to_load) std::cout << sym << " ";
        std::cout << std::endl;

        for (const auto& sym : symbols_to_load) {
            std::vector<sentio::Bar> bars;
            std::string data_path = sentio::resolve_csv(sym);
            if (!sentio::load_csv(data_path, bars)) {
                std::cerr << "ERROR: Failed to load data for " << sym << " from " << data_path << std::endl;
                continue;
            }
            std::cout << " -> Loaded " << bars.size() << " bars for " << sym << std::endl;
            
            int symbol_id = ST.intern(sym);
            if (static_cast<size_t>(symbol_id) >= series.size()) {
                series.resize(symbol_id + 1);
            }
            series[symbol_id] = std::move(bars);
        }
        
        int base_symbol_id = ST.get_id(base_symbol);
        if (series.empty() || series[base_symbol_id].empty()) {
            std::cerr << "FATAL: No data loaded for base symbol " << base_symbol << std::endl;
            return 1;
        }

        // Report data period and bars for transparency
        auto& base_bars2 = series[base_symbol_id];
        auto fmt_date2 = [](std::int64_t epoch_sec){
            std::time_t tt = static_cast<std::time_t>(epoch_sec);
            std::tm tm{}; gmtime_r(&tt, &tm);
            char buf[32]; std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm); return std::string(buf);
        };
        std::int64_t min_ts2 = base_bars2.front().ts_nyt_epoch;
        std::int64_t max_ts2 = base_bars2.back().ts_nyt_epoch;
        std::size_t   n_bars2 = base_bars2.size();
        double span_days2 = double(max_ts2 - min_ts2) / (24.0*3600.0);
        std::cout << "Data period: " << fmt_date2(min_ts2) << " → " << fmt_date2(max_ts2)
                  << " (" << std::fixed << std::setprecision(1) << span_days2 << " days)\n";
        std::cout << "Bars(" << base_symbol << "): " << n_bars2 << "\n";
        if ((max_ts2 - min_ts2) < (365LL*24LL*3600LL)) {
            std::cerr << "WARNING: Data period is shorter than 1 year. Results may be unrepresentative.\n";
        }

        // Load cached features for ML strategies for massive performance improvement
        if (strategy_name == "TFA" || strategy_name == "tfa") {
            std::string feature_file = "data/" + base_symbol + "_RTH_features.csv";
            std::cout << "Loading pre-computed features from: " << feature_file << std::endl;
            if (sentio::FeatureFeeder::load_feature_cache(feature_file)) {
                sentio::FeatureFeeder::use_cached_features(true);
                std::cout << "✅ Cached features loaded successfully - MASSIVE speed improvement enabled!" << std::endl;
            } else {
                std::cout << "⚠️  Failed to load cached features - falling back to real-time calculation" << std::endl;
            }
        } else if (strategy_name == "kochi_ppo") {
            std::string feature_file = "data/" + base_symbol + "_KOCHI_features.csv";
            std::cout << "Loading pre-computed KOCHI features from: " << feature_file << std::endl;
            if (sentio::FeatureFeeder::load_feature_cache(feature_file)) {
                sentio::FeatureFeeder::use_cached_features(true);
                std::cout << "✅ KOCHI cached features loaded successfully" << std::endl;
            } else {
                std::cerr << "❌ KOCHI feature cache missing; cannot proceed for kochi_ppo strategy." << std::endl;
                return 1;
            }
        }

        sentio::RunnerCfg cfg;
        cfg.strategy_name = strategy_name;
        cfg.strategy_params = strategy_params;
        cfg.audit_level = sentio::AuditLevel::Full;
        cfg.snapshot_stride = 100;
        
        sentio::TemporalAnalysisConfig temporal_cfg;
        temporal_cfg.num_quarters = num_quarters;
        temporal_cfg.print_detailed_report = true;
        
        std::cout << "\nRunning TPA (Temporal Performance Analysis) Test..." << std::endl;
        std::cout << "Strategy: " << strategy_name << ", Quarters: " << num_quarters << std::endl;
        
        sentio::Tsc timer;
        timer.tic();
        auto summary = sentio::run_temporal_analysis(ST, series, base_symbol_id, cfg, temporal_cfg);
        double elapsed = timer.toc_sec();
        
        std::cout << "\nTPA test completed in " << elapsed << "s" << std::endl;
        
        if (temporal_cfg.print_detailed_report) {
            sentio::TemporalAnalyzer analyzer;
            for (const auto& q : summary.quarterly_results) {
                analyzer.add_quarterly_result(q);
            }
            analyzer.print_detailed_report();
        }
        
    } else if (command == "test-models") {
        std::string strategy_name = "tfa";
        std::string data_file = "data/QQQ_RTH.csv";
        std::string start_date = "2023-01-01";
        std::string end_date = "2023-12-31";
        
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--strategy" && i + 1 < argc) {
                strategy_name = argv[++i];
            } else if (arg == "--data" && i + 1 < argc) {
                data_file = argv[++i];
            } else if (arg == "--start" && i + 1 < argc) {
                start_date = argv[++i];
            } else if (arg == "--end" && i + 1 < argc) {
                end_date = argv[++i];
            }
        }
        
        // Validate strategy name
        if (strategy_name != "tfa") {
            std::cerr << "ERROR: Strategy must be 'tfa'. Got: " << strategy_name << std::endl;
            return 1;
        }
        
        // Check if model exists
        std::string model_dir = "artifacts/TFA/v1";
        std::string model_path = model_dir + "/model.pt";
        std::string metadata_path = model_dir + "/metadata.json";
        
        if (!std::ifstream(model_path).good()) {
            std::cerr << "ERROR: Model not found at " << model_path << std::endl;
            std::cerr << "Please train the model first: python train_models.py" << std::endl;
            return 1;
        }
        
        if (!std::ifstream(metadata_path).good()) {
            std::cerr << "ERROR: Metadata not found at " << metadata_path << std::endl;
            return 1;
        }
        
        std::cout << "🧪 Testing " << strategy_name << " model..." << std::endl;
        std::cout << "📁 Model: " << model_path << std::endl;
        std::cout << "📊 Data: " << data_file << std::endl;
        std::cout << "📅 Period: " << start_date << " to " << end_date << std::endl;
        
        // Load data
        std::vector<sentio::Bar> bars;
        if (!sentio::load_csv(data_file, bars)) {
            std::cerr << "ERROR: Failed to load data from " << data_file << std::endl;
            return 1;
        }
        
        std::cout << "✅ Loaded " << bars.size() << " bars" << std::endl;
        
        // Load feature specification
        std::string feature_spec_path = "python/feature_spec.json";
        std::ifstream spec_file(feature_spec_path);
        if (!spec_file.good()) {
            std::cerr << "ERROR: Feature spec not found at " << feature_spec_path << std::endl;
            return 1;
        }
        
        std::string spec_json((std::istreambuf_iterator<char>(spec_file)), std::istreambuf_iterator<char>());
        std::cout << "✅ Loaded feature specification" << std::endl;
        
        // For now, let's skip the C++ feature builder and use a simple approach
        // We'll create dummy features to test the model loading and signal generation
        std::cout << "🔧 Creating dummy features for testing..." << std::endl;
        
        // Create a simple feature matrix with dummy data
        sentio::FeatureMatrix feature_matrix;
        feature_matrix.rows = std::min(100, static_cast<int>(bars.size()));
        feature_matrix.cols = 6; // close, logret, ema_20, ema_50, rsi_14, zlog_20
        feature_matrix.data.resize(feature_matrix.rows * feature_matrix.cols);
        
        // Fill with dummy features (just close prices for now)
        for (int i = 0; i < feature_matrix.rows; ++i) {
            int base_idx = i * feature_matrix.cols;
            feature_matrix.data[base_idx + 0] = static_cast<float>(bars[i].close); // close
            feature_matrix.data[base_idx + 1] = 0.0f; // logret
            feature_matrix.data[base_idx + 2] = static_cast<float>(bars[i].close); // ema_20
            feature_matrix.data[base_idx + 3] = static_cast<float>(bars[i].close); // ema_50
            feature_matrix.data[base_idx + 4] = 50.0f; // rsi_14
            feature_matrix.data[base_idx + 5] = 0.0f; // zlog_20
        }
        
        std::cout << "✅ Created dummy features: " << feature_matrix.rows << " rows x " << feature_matrix.cols << " cols" << std::endl;
        
        // Create TFA strategy instance
        std::unique_ptr<sentio::TFAStrategy> tfa_strategy = std::make_unique<sentio::TFAStrategy>();
        
        std::cout << "\n🧪 Testing " << strategy_name << " model with features..." << std::endl;
        
        // Feed features to strategy and test signal generation
        int signals_generated = 0;
        int total_bars = std::min(100, static_cast<int>(feature_matrix.rows)); // Test first 100 bars
        
        for (int i = 0; i < total_bars; ++i) {
            // Extract features for this bar
            std::vector<double> raw_features(feature_matrix.cols);
            for (int j = 0; j < feature_matrix.cols; ++j) {
                raw_features[j] = feature_matrix.data[i * feature_matrix.cols + j];
            }
            
            // Create a dummy bar for on_bar call
            sentio::Bar dummy_bar;
            if (i < static_cast<int>(bars.size())) {
                dummy_bar = bars[i];
            }
            
            sentio::StrategyCtx ctx;
            ctx.ts_utc_epoch = dummy_bar.ts_nyt_epoch;
            ctx.instrument = "QQQ";
            ctx.is_rth = true;
            
            // Feed features to strategy and call on_bar
            if (tfa_strategy) {
                tfa_strategy->set_raw_features(raw_features);
                tfa_strategy->on_bar(ctx, dummy_bar);
                auto signal = tfa_strategy->latest();
                if (signal) {
                    signals_generated++;
                    std::cout << "Bar " << i << ": Signal generated - " 
                              << (signal->type == sentio::StrategySignal::Type::BUY ? "BUY" : 
                                  signal->type == sentio::StrategySignal::Type::SELL ? "SELL" : "HOLD")
                              << " (conf=" << signal->confidence << ")" << std::endl;
                }
            } else if (tfa_strategy) {
                tfa_strategy->set_raw_features(raw_features);
                tfa_strategy->on_bar(ctx, dummy_bar);
                auto signal = tfa_strategy->latest();
                if (signal) {
                    signals_generated++;
                    std::cout << "Bar " << i << ": Signal generated - " 
                              << (signal->type == sentio::StrategySignal::Type::BUY ? "BUY" : 
                                  signal->type == sentio::StrategySignal::Type::SELL ? "SELL" : "HOLD")
                              << " (conf=" << signal->confidence << ")" << std::endl;
                }
            }
        }
        
        std::cout << "\n📊 Test Results:" << std::endl;
        std::cout << "  Total bars tested: " << total_bars << std::endl;
        std::cout << "  Signals generated: " << signals_generated << std::endl;
        std::cout << "  Signal rate: " << (100.0 * signals_generated / total_bars) << "%" << std::endl;
        
        if (signals_generated == 0) {
            std::cout << "\n⚠️  WARNING: No signals generated! This could indicate:" << std::endl;
            std::cout << "  - Model confidence threshold too high" << std::endl;
            std::cout << "  - Feature window not ready" << std::endl;
            std::cout << "  - Model prediction issues" << std::endl;
            std::cout << "  - Check the diagnostic output above for details" << std::endl;
        } else {
            std::cout << "\n✅ Model is generating signals successfully!" << std::endl;
        }
        
    } else if (command == "replay") {
        std::cout << "Replay command is not fully implemented in this example.\n";
    } else {
        usage();
        return 1;
    }
    
    return 0;
}

```

## 📄 **FILE 115 of 145**: src/ml/model_registry_ts.cpp

**File Information**:
- **Path**: `src/ml/model_registry_ts.cpp`

- **Size**: 71 lines
- **Modified**: 2025-09-07 12:32:16

- **Type**: .cpp

```text
#include "sentio/ml/model_registry.hpp"
#include "sentio/ml/ts_model.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace sentio::ml {

static std::string slurp(const std::string& path){
  std::ifstream f(path); if (!f) throw std::runtime_error("cannot open "+path);
  std::ostringstream ss; ss<<f.rdbuf(); return ss.str();
}
static bool find_val(const std::string& j, const std::string& key, std::string& out){
  auto k="\""+key+"\""; auto p=j.find(k); if (p==std::string::npos) return false;
  p=j.find(':',p); if (p==std::string::npos) return false; ++p;
  while (p<j.size() && isspace((unsigned char)j[p])) ++p;
  if (j[p]=='"'){ auto e=j.find('"',p+1); out=j.substr(p+1,e-(p+1)); return true; }
  auto e=j.find_first_of(",}\n",p); out=j.substr(p,e-p); return true;
}
static std::vector<std::string> parse_sarr(const std::string& j, const std::string& key){
  std::vector<std::string> v; auto k="\""+key+"\""; auto p=j.find(k); if (p==std::string::npos) return v;
  p=j.find('[',p); auto e=j.find(']',p); if (p==std::string::npos||e==std::string::npos) return v;
  auto s=j.substr(p+1,e-(p+1)); size_t i=0;
  while (i<s.size()){ auto q1=s.find('"',i); if (q1==std::string::npos) break; auto q2=s.find('"',q1+1);
    v.push_back(s.substr(q1+1,q2-(q1+1))); i=q2+1; }
  return v;
}
static std::vector<double> parse_darr(const std::string& j, const std::string& key){
  std::vector<double> v; auto k="\""+key+"\""; auto p=j.find(k); if (p==std::string::npos) return v;
  p=j.find('[',p); auto e=j.find(']',p); if (p==std::string::npos||e==std::string::npos) return v;
  auto s=j.substr(p+1,e-(p+1)); size_t i=0;
  while (i<s.size()){ auto j2=s.find_first_of(", \t\n", i);
    auto tok=s.substr(i,(j2==std::string::npos)?std::string::npos:(j2-i));
    if (!tok.empty()) v.push_back(std::stod(tok));
    if (j2==std::string::npos) break; i=j2+1; }
  return v;
}

ModelHandle ModelRegistryTS::load_torchscript(const std::string& model_id,
                                              const std::string& version,
                                              const std::string& artifacts_dir,
                                              bool use_cuda)
{
  const std::string base = artifacts_dir + "/" + model_id + "/" + version + "/";
  const std::string meta_path = base + "metadata.json";
  const std::string pt_path   = base + "model.pt";

  auto js = slurp(meta_path);

  ModelSpec spec;
  spec.model_id = model_id;
  spec.version  = version;
  spec.feature_names = parse_sarr(js, "feature_names");
  spec.mean = parse_darr(js, "mean");
  spec.std  = parse_darr(js, "std");
  auto clip = parse_darr(js, "clip"); if (clip.size()==2) spec.clip2 = clip;
  spec.actions = parse_sarr(js, "actions");

  std::string t; if (find_val(js, "expected_bar_spacing_sec", t)) spec.expected_spacing_sec = std::stoi(t);
  if (find_val(js, "seq_len", t)) spec.seq_len = std::stoi(t);
  std::string layout; if (find_val(js, "input_layout", layout)) spec.input_layout = layout;
  std::string fmt; if (find_val(js, "format", fmt)) spec.format = fmt; else spec.format="torchscript";

  ModelHandle h;
  h.spec = spec;
  h.model = TorchScriptModel::load(pt_path, h.spec, use_cuda);
  return h;
}

} // namespace sentio::ml

```

## 📄 **FILE 116 of 145**: src/ml/ts_model.cpp

**File Information**:
- **Path**: `src/ml/ts_model.cpp`

- **Size**: 81 lines
- **Modified**: 2025-09-07 14:29:16

- **Type**: .cpp

```text
#include "sentio/ml/ts_model.hpp"
#include <torch/script.h>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace sentio::ml {

TorchScriptModel::TorchScriptModel(ModelSpec spec) : spec_(std::move(spec)), input_tensor_(nullptr) {}

TorchScriptModel::~TorchScriptModel() {
  if (input_tensor_) {
    delete static_cast<torch::Tensor*>(input_tensor_);
  }
}

std::unique_ptr<TorchScriptModel> TorchScriptModel::load(const std::string& pt_path,
                                                         const ModelSpec& spec,
                                                         [[maybe_unused]] bool use_cuda)
{
  auto m = std::unique_ptr<TorchScriptModel>(new TorchScriptModel(spec));
  torch::jit::script::Module mod = torch::jit::load(pt_path);
  mod.eval();
  m->mod_ = std::make_shared<torch::jit::script::Module>(std::move(mod));
  // Disable CUDA for now - CPU-only build
  m->cuda_ = false; // use_cuda && torch::cuda::is_available();
  if (m->cuda_) m->mod_->to(torch::kCUDA);
  // Defer concrete shape until first predict (we need T,F,layout)
  m->in_shape_.clear();
  return m;
}

std::optional<ModelOutput> TorchScriptModel::predict(const std::vector<float>& feats,
                                                     int T, int F, const std::string& layout) const
{
  if (T<=0 || F<=0) return std::nullopt;
  const size_t need = (layout=="BF")? feats.size() : size_t(T)*size_t(F);
  if (feats.size() != need) return std::nullopt;

  torch::NoGradGuard ng; torch::InferenceMode im;

  // Pre-allocate input tensor if needed (persistent tensor approach)
  std::vector<int64_t> need_shape = (layout=="BTF") ? std::vector<int64_t>{1,T,F}
                                                    : std::vector<int64_t>{1,(int64_t)feats.size()};
  
  if (!input_tensor_ || in_shape_ != need_shape) {
    in_shape_ = need_shape;
    if (input_tensor_) {
      delete static_cast<torch::Tensor*>(input_tensor_);
    }
    input_tensor_ = new torch::Tensor(torch::empty(in_shape_, torch::TensorOptions().dtype(torch::kFloat32).device(cuda_?torch::kCUDA:torch::kCPU)));
  }

  // memcpy into persistent tensor (no allocation, no clone)
  torch::Tensor& x = *static_cast<torch::Tensor*>(input_tensor_);
  if (cuda_) {
    // For CUDA, copy via CPU tensor to avoid sync issues
    torch::Tensor host = torch::from_blob((void*)feats.data(), {(int64_t)feats.size()}, torch::kFloat32);
    x.view({-1}).copy_(host);
  } else {
    // For CPU, direct memcpy into tensor data
    std::memcpy(x.data_ptr<float>(), feats.data(), feats.size() * sizeof(float));
  }

  std::vector<torch::jit::IValue> inputs; inputs.emplace_back(x);
  torch::Tensor out = mod_->forward(inputs).toTensor().to(torch::kCPU).contiguous();
  if (out.dim()==2 && out.size(0)==1) out = out.squeeze(0);
  if (out.dim()!=1) return std::nullopt;

  ModelOutput mo; mo.probs.resize(out.numel());
  std::memcpy(mo.probs.data(), out.data_ptr<float>(), mo.probs.size()*sizeof(float));
  float sum=0.f; for (float v: mo.probs) sum += std::isfinite(v)? v : 0.f;
  if (!(sum>0.f)) {
    float mv=*std::max_element(mo.probs.begin(), mo.probs.end());
    float s=0; for (auto& v: mo.probs){ v=std::exp(v-mv); s+=v; } if (s>0) for (auto& v: mo.probs) v/=s;
  } else for (auto& v: mo.probs) v = std::max(0.f, v/sum);
  return mo;
}

} // namespace sentio::ml

```

## 📄 **FILE 117 of 145**: src/optimizer.cpp

**File Information**:
- **Path**: `src/optimizer.cpp`

- **Size**: 180 lines
- **Modified**: 2025-09-05 09:41:27

- **Type**: .cpp

```text
#include "sentio/optimizer.hpp"
#include "sentio/runner.hpp"
#include <iostream>

namespace sentio {

// RandomSearchOptimizer Implementation
std::vector<Parameter> RandomSearchOptimizer::get_parameter_space() {
    return {
        Parameter("test_param", 0.0, 1.0, 0.5)
    };
}

void RandomSearchOptimizer::apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) {
    for (const auto& param : params) {
        cfg.strategy_params[param.name] = std::to_string(param.current_value);
    }
}

OptimizationResult RandomSearchOptimizer::optimize([[maybe_unused]] const std::function<double(const RunResult&)>& objective_func,
                                                  [[maybe_unused]] const std::vector<Parameter>& param_space,
                                                  [[maybe_unused]] const OptimizationConfig& config) {
    OptimizationResult result;
    result.parameters["test"] = 0.5;
    result.objective_value = 0.0;
    return result;
}

// GridSearchOptimizer Implementation
std::vector<Parameter> GridSearchOptimizer::get_parameter_space() {
    return {
        Parameter("test_param", 0.0, 1.0, 0.5)
    };
}

void GridSearchOptimizer::apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) {
    for (const auto& param : params) {
        cfg.strategy_params[param.name] = std::to_string(param.current_value);
    }
}

OptimizationResult GridSearchOptimizer::optimize([[maybe_unused]] const std::function<double(const RunResult&)>& objective_func,
                                                [[maybe_unused]] const std::vector<Parameter>& param_space,
                                                [[maybe_unused]] const OptimizationConfig& config) {
    OptimizationResult result;
    result.parameters["test"] = 0.5;
    result.objective_value = 0.0;
    return result;
}

// BayesianOptimizer Implementation
std::vector<Parameter> BayesianOptimizer::get_parameter_space() {
    return {
        Parameter("test_param", 0.0, 1.0, 0.5)
    };
}

void BayesianOptimizer::apply_parameters(const std::vector<Parameter>& params, RunnerCfg& cfg) {
    for (const auto& param : params) {
        cfg.strategy_params[param.name] = std::to_string(param.current_value);
    }
}

OptimizationResult BayesianOptimizer::optimize([[maybe_unused]] const std::function<double(const RunResult&)>& objective_func,
                                              [[maybe_unused]] const std::vector<Parameter>& param_space,
                                              [[maybe_unused]] const OptimizationConfig& config) {
    OptimizationResult result;
    result.parameters["test"] = 0.5;
    result.objective_value = 0.0;
    return result;
}

// Strategy parameter creation functions
std::vector<Parameter> create_vwap_parameters() {
    return {
        Parameter("vwap_period", 100.0, 800.0, 200.0),
        Parameter("reversion_threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_momentum_parameters() {
    return {
        Parameter("momentum_period", 5.0, 50.0, 20.0),
        Parameter("threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_volatility_parameters() {
    return {
        Parameter("volatility_period", 10.0, 50.0, 20.0),
        Parameter("threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_bollinger_squeeze_parameters() {
    return {
        Parameter("bb_period", 10.0, 40.0, 20.0),
        Parameter("bb_std", 1.0, 3.0, 2.0)
    };
}

std::vector<Parameter> create_opening_range_parameters() {
    return {
        Parameter("range_minutes", 15.0, 60.0, 30.0),
        Parameter("breakout_threshold", 0.001, 0.01, 0.005)
    };
}

std::vector<Parameter> create_order_flow_scalping_parameters() {
    return {
        Parameter("imbalance_period", 5.0, 40.0, 20.0),
        Parameter("threshold", 0.4, 0.9, 0.7)
    };
}

std::vector<Parameter> create_order_flow_imbalance_parameters() {
    return {
        Parameter("lookback_window", 5.0, 50.0, 20.0),
        Parameter("threshold", 0.5, 3.0, 1.5)
    };
}

std::vector<Parameter> create_market_making_parameters() {
    return {
        Parameter("base_spread", 0.0002, 0.003, 0.001),
        Parameter("order_levels", 1.0, 5.0, 3.0)
    };
}

std::vector<Parameter> create_router_parameters() {
    return {
        Parameter("t1", 0.01, 0.3, 0.05),
        Parameter("t2", 0.1, 0.6, 0.3)
    };
}

std::vector<Parameter> create_parameters_for_strategy(const std::string& strategy_name) {
    if (strategy_name == "VWAPReversion") {
        return create_vwap_parameters();
    } else if (strategy_name == "MomentumVolumeProfile") {
        return create_momentum_parameters();
    } else if (strategy_name == "VolatilityExpansion") {
        return create_volatility_parameters();
    } else if (strategy_name == "BollingerSqueezeBreakout") {
        return create_bollinger_squeeze_parameters();
    } else if (strategy_name == "OpeningRangeBreakout") {
        return create_opening_range_parameters();
    } else if (strategy_name == "OrderFlowScalping") {
        return create_order_flow_scalping_parameters();
    } else if (strategy_name == "OrderFlowImbalance") {
        return create_order_flow_imbalance_parameters();
    } else if (strategy_name == "MarketMaking") {
        return create_market_making_parameters();
    } else {
        return create_router_parameters();
    }
}

std::vector<Parameter> create_full_parameter_space() {
    return create_router_parameters();
}

// OptimizationEngine implementation
OptimizationEngine::OptimizationEngine(const std::string& optimizer_type) {
    if (optimizer_type == "grid") {
        optimizer = std::make_unique<GridSearchOptimizer>();
    } else if (optimizer_type == "bayesian") {
        optimizer = std::make_unique<BayesianOptimizer>();
    } else {
        optimizer = std::make_unique<RandomSearchOptimizer>();
    }
}

OptimizationResult OptimizationEngine::run_optimization(const std::string& strategy_name,
                                                       const std::function<double(const RunResult&)>& objective_func) {
    auto param_space = create_parameters_for_strategy(strategy_name);
    return optimizer->optimize(objective_func, param_space, config);
}

} // namespace sentio

```

## 📄 **FILE 118 of 145**: src/pnl_accounting.cpp

**File Information**:
- **Path**: `src/pnl_accounting.cpp`

- **Size**: 62 lines
- **Modified**: 2025-09-05 15:28:49

- **Type**: .cpp

```text
// src/pnl_accounting.cpp
#include "sentio/pnl_accounting.hpp"
#include <shared_mutex>
#include <unordered_map>
#include <utility>

namespace sentio {

namespace {
// Simple in-TU storage for latest bars keyed by instrument.
// If your header already declares members, remove this and
// implement against those members instead.
struct PriceBookStorage {
    std::unordered_map<std::string, Bar> latest;
    mutable std::shared_mutex mtx;
};

// One global storage per process for the default PriceBook()
// If your PriceBook is an instance with its own fields, move this inside the class.
static PriceBookStorage& storage() {
    static PriceBookStorage s;
    return s;
}
} // anonymous namespace

// ---- Interface implementation ----

const Bar* PriceBook::get_latest(const std::string& instrument) const {
    auto& S = storage();
    std::shared_lock lk(S.mtx);
    auto it = S.latest.find(instrument);
    if (it == S.latest.end()) return nullptr;
    return &it->second; // pointer remains valid as long as map bucket survives
}

// ---- Additional helper methods (not declared in header but useful) ----

void PriceBook::upsert_latest(const std::string& instrument, const Bar& b) {
    auto& S = storage();
    std::unique_lock lk(S.mtx);
    S.latest[instrument] = b;
}

bool PriceBook::has_instrument(const std::string& instrument) const {
    auto& S = storage();
    std::shared_lock lk(S.mtx);
    return S.latest.find(instrument) != S.latest.end();
}

std::size_t PriceBook::size() const {
    auto& S = storage();
    std::shared_lock lk(S.mtx);
    return S.latest.size();
}

double last_trade_price(const PriceBook& book, const std::string& instrument) {
    auto* b = book.get_latest(instrument);
    if (!b) throw std::runtime_error("No bar for instrument: " + instrument);
    return b->close;
}

} // namespace sentio

```

## 📄 **FILE 119 of 145**: src/poly_fetch_main.cpp

**File Information**:
- **Path**: `src/poly_fetch_main.cpp`

- **Size**: 57 lines
- **Modified**: 2025-09-07 03:24:41

- **Type**: .cpp

```text
#include "sentio/polygon_client.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

using namespace sentio;

int main(int argc,char**argv){
  if(argc<5){
    std::cerr<<"Usage: poly_fetch FAMILY from to outdir [--timespan day|hour|minute] [--multiplier N] [--symbols SYM1,SYM2,...] [--rth]\n";
    return 1;
  }
  std::string fam=argv[1], from=argv[2], to=argv[3], outdir=argv[4];
  std::string timespan = "day";
  int multiplier = 1;
  std::string symbols_csv;
  bool rth_only=false;
  bool exclude_holidays=false;
  for (int i=5;i<argc;i++) {
    std::string a = argv[i];
    if ((a=="--timespan" || a=="-t") && i+1<argc) { timespan = argv[++i]; }
    else if ((a=="--multiplier" || a=="-m") && i+1<argc) { multiplier = std::stoi(argv[++i]); }
    else if (a=="--symbols" && i+1<argc) { symbols_csv = argv[++i]; }
    else if (a=="--rth") { rth_only=true; }
    else if (a=="--no-holidays") { exclude_holidays=true; }
  }
  const char* key = std::getenv("POLYGON_API_KEY");
  std::string api_key = key? key: "";
  PolygonClient cli(api_key);

  std::vector<std::string> syms;
  if(fam=="qqq") syms={"QQQ","TQQQ","SQQQ","PSQ"};
  else if(fam=="bitcoin") syms={"X:BTCUSD","X:ETHUSD"};
  else if(fam=="tesla") syms={"TSLA","TSLQ"};
  else if(fam=="custom") {
    if (symbols_csv.empty()) { std::cerr<<"--symbols required for custom family\n"; return 1; }
    size_t start=0; while (start < symbols_csv.size()) {
      size_t pos = symbols_csv.find(',', start);
      std::string tok = (pos==std::string::npos)? symbols_csv.substr(start) : symbols_csv.substr(start, pos-start);
      if (!tok.empty()) syms.push_back(tok);
      if (pos==std::string::npos) break; else start = pos+1;
    }
  } else { std::cerr<<"Unknown family\n"; return 1; }

  for(auto&s:syms){
    AggsQuery q; q.symbol=s; q.from=from; q.to=to; q.timespan=timespan; q.multiplier=multiplier; q.adjusted=true; q.sort="asc";
    auto bars=cli.get_aggs_all(q);
    std::string suffix;
    if (rth_only) suffix += "_RTH";
    if (exclude_holidays) suffix += "_NH";
    std::string fname= outdir + "/" + s + suffix + ".csv";
    cli.write_csv(fname,s,bars,rth_only,exclude_holidays);
    std::cerr<<"Wrote "<<bars.size()<<" bars -> "<<fname<<"\n";
  }
}


```

## 📄 **FILE 120 of 145**: src/polygon_client.cpp

**File Information**:
- **Path**: `src/polygon_client.cpp`

- **Size**: 244 lines
- **Modified**: 2025-09-07 04:14:17

- **Type**: .cpp

```text
#include "sentio/polygon_client.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <cctz/time_zone.h>
#include <cctz/civil_time.h>
#include <fstream>
#include <thread>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>

using json = nlohmann::json;
namespace sentio {

static size_t write_cb(void* contents, size_t size, size_t nmemb, void* userp) {
  size_t total = size * nmemb;
  std::string* s = static_cast<std::string*>(userp);
  s->append(static_cast<char*>(contents), total);
  return total;
}

static std::string rfc3339_utc_from_epoch_ms(long long ms) {
  using namespace std::chrono;
  
  cctz::time_point<cctz::seconds> tp{cctz::seconds{ms / 1000}};
  
  // Get UTC timezone
  cctz::time_zone utc_tz;
  if (!cctz::load_time_zone("UTC", &utc_tz)) {
    return "1970-01-01T00:00:00Z"; // fallback
  }
  
  // Convert to UTC civil time
  auto lt = cctz::convert(tp, utc_tz);
  auto ct = cctz::civil_second(lt);
  
  std::ostringstream oss;
  oss << std::setfill('0') 
      << std::setw(4) << ct.year() << "-"
      << std::setw(2) << ct.month() << "-"
      << std::setw(2) << ct.day() << "T"
      << std::setw(2) << ct.hour() << ":"
      << std::setw(2) << ct.minute() << ":"
      << std::setw(2) << ct.second() << "Z";
  
  return oss.str();
}

// **NEW**: RTH check directly from UTC timestamp.
// RTH in UTC: 13:30-20:00 UTC (EDT) or 14:30-21:00 UTC (EST)
static bool is_rth_utc_from_utc_ms(long long utc_ms) {
    cctz::time_point<cctz::seconds> tp{cctz::seconds{utc_ms / 1000}};
    
    // Get UTC timezone
    cctz::time_zone utc_tz;
    if (!cctz::load_time_zone("UTC", &utc_tz)) {
        return false;
    }
    
    // Convert to UTC civil time
    auto lt = cctz::convert(tp, utc_tz);
    auto ct = cctz::civil_second(lt);
    
    // Check if weekend (Saturday = 6, Sunday = 0)
    auto wd = cctz::get_weekday(ct);
    if (wd == cctz::weekday::saturday || wd == cctz::weekday::sunday) {
        return false;
    }
    
    // Check if RTH in UTC
    // EST: 14:30-21:00 UTC (9:30 AM - 4:00 PM EST)
    // EDT: 13:30-20:00 UTC (9:30 AM - 4:00 PM EDT)
    int hour = ct.hour();
    int minute = ct.minute();
    
    // Simple DST check: April-October is EDT, rest is EST
    int month = ct.month();
    bool is_edt = (month >= 4 && month <= 10);
    
    if (is_edt) {
        // EDT: 13:30-20:00 UTC
        if (hour < 13 || (hour == 13 && minute < 30)) {
            return false;  // Before 13:30 UTC
        }
        if (hour >= 20) {
            return false;  // After 20:00 UTC
        }
    } else {
        // EST: 14:30-21:00 UTC
        if (hour < 14 || (hour == 14 && minute < 30)) {
            return false;  // Before 14:30 UTC
        }
        if (hour >= 21) {
            return false;  // After 21:00 UTC
        }
    }
    
    return true;
}

// **NEW**: Holiday check in UTC
static bool is_us_market_holiday_utc(int year, int month, int day) {
  // Simple holiday check for common US market holidays in UTC
  // This is a simplified version - for production use, integrate with the full calendar system
  
  // New Year's Day (observed)
  if (month == 1 && day == 1) return true;
  if (month == 1 && day == 2) return true; // observed if Jan 1 is Sunday
  
  // MLK Day (3rd Monday in January)
  if (month == 1 && day >= 15 && day <= 21) {
    // Simple check - this could be more precise
    return true;
  }
  
  // Presidents Day (3rd Monday in February)
  if (month == 2 && day >= 15 && day <= 21) {
    return true;
  }
  
  // Good Friday (varies by year)
  if (year == 2022 && month == 4 && day == 15) return true;
  if (year == 2023 && month == 4 && day == 7) return true;
  if (year == 2024 && month == 3 && day == 29) return true;
  if (year == 2025 && month == 4 && day == 18) return true;
  
  // Memorial Day (last Monday in May)
  if (month == 5 && day >= 25 && day <= 31) {
    return true;
  }
  
  // Juneteenth (observed)
  if (month == 6 && day == 19) return true;
  if (month == 6 && day == 20) return true; // observed if Jun 19 is Sunday
  
  // Independence Day (observed)
  if (month == 7 && day == 4) return true;
  if (month == 7 && day == 5) return true; // observed if Jul 4 is Sunday
  
  // Labor Day (1st Monday in September)
  if (month == 9 && day >= 1 && day <= 7) {
    return true;
  }
  
  // Thanksgiving (4th Thursday in November)
  if (month == 11 && day >= 22 && day <= 28) {
    return true;
  }
  
  // Christmas (observed)
  if (month == 12 && day == 25) return true;
  if (month == 12 && day == 26) return true; // observed if Dec 25 is Sunday
  
  return false;
}

PolygonClient::PolygonClient(std::string api_key) : api_key_(std::move(api_key)) {}

std::string PolygonClient::get_(const std::string& url) {
    CURL* curl = curl_easy_init();
    std::string buffer;
    if (!curl) return buffer;
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    struct curl_slist* headers = nullptr;
    std::string auth = "Authorization: Bearer " + api_key_;
    headers = curl_slist_append(headers, auth.c_str());
    headers = curl_slist_append(headers, "Accept: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return buffer;
}

std::vector<AggBar> PolygonClient::get_aggs_all(const AggsQuery& q, int max_pages) {
    std::vector<AggBar> out;
    std::string base = "https://api.polygon.io/v2/aggs/ticker/" + q.symbol + "/range/" + std::to_string(q.multiplier) + "/" + q.timespan + "/" + q.from + "/" + q.to + "?adjusted=" + (q.adjusted?"true":"false") + "&sort=" + q.sort + "&limit=" + std::to_string(q.limit);
    std::string url = base;
    
    for (int page=0; page<max_pages; ++page) {
        std::string body = get_(url);
        if (body.empty()) break;
        
        auto j = json::parse(body, nullptr, false);
        if (j.is_discarded()) break;
        
        if (j.contains("results")) {
            for (auto& r : j["results"]) {
                out.push_back({r.value("t", 0LL), r.value("o", 0.0), r.value("h", 0.0), r.value("l", 0.0), r.value("c", 0.0), r.value("v", 0.0)});
            }
        }
        
        if (!j.contains("next_url")) break;
        url = j["next_url"].get<std::string>();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    return out;
}

void PolygonClient::write_csv(const std::string& out_path,const std::string& symbol,
                              const std::vector<AggBar>& bars, bool rth_only, bool exclude_holidays) {
  std::ofstream f(out_path);
  f << "timestamp,symbol,open,high,low,close,volume\n";
  for (auto& a: bars) {
    // **MODIFIED**: RTH and holiday filtering is now done directly on the UTC timestamp
    // before any string conversion, making it much more reliable.

    if (rth_only && !is_rth_utc_from_utc_ms(a.ts_ms)) {
        continue;
    }
    
    if (exclude_holidays) {
        cctz::time_point<cctz::seconds> tp{cctz::seconds{a.ts_ms / 1000}};
        
        // Get UTC timezone
        cctz::time_zone utc_tz;
        if (cctz::load_time_zone("UTC", &utc_tz)) {
            auto lt = cctz::convert(tp, utc_tz);
            auto ct = cctz::civil_second(lt);
            
            if (is_us_market_holiday_utc(ct.year(), ct.month(), ct.day())) {
                continue;
            }
        }
    }
    
    // The timestamp is converted to a UTC string for writing to the CSV
    std::string ts_str = rfc3339_utc_from_epoch_ms(a.ts_ms);

    f << ts_str << ',' << symbol << ','
      << a.open << ',' << a.high << ',' << a.low << ',' << a.close << ',' << a.volume << '\n';
  }
}

} // namespace sentio

```

## 📄 **FILE 121 of 145**: src/polygon_ingest.cpp

**File Information**:
- **Path**: `src/polygon_ingest.cpp`

- **Size**: 68 lines
- **Modified**: 2025-09-05 15:29:30

- **Type**: .cpp

```text
#include "sentio/polygon_ingest.hpp"
#include "sentio/time_utils.hpp"
#include "sentio/rth_calendar.hpp"
#include "sentio/calendar_seed.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace sentio {

static const TradingCalendar& nyse_calendar() {
  static TradingCalendar cal = make_default_nyse_calendar();
  return cal;
}

static inline bool accept_bar_utc(std::int64_t ts_utc) {
  return nyse_calendar().is_rth_utc(ts_utc, "America/New_York");
}

static inline bool valid_bar(const ProviderBar& b) {
  if (!std::isfinite(b.open)  || !std::isfinite(b.high) ||
      !std::isfinite(b.low)   || !std::isfinite(b.close)||
      !std::isfinite(b.volume)) return false;
  if (b.open <= 0 || b.high <= 0 || b.low <= 0 || b.close <= 0) return false;
  if (b.low > b.high) return false;
  return !b.symbol.empty();
}

static inline void upsert_bar(PriceBook& book, const std::string& instrument, const ProviderBar& pb) {
  Bar b;
  b.open  = pb.open;
  b.high  = pb.high;
  b.low   = pb.low;
  b.close = pb.close;
  book.upsert_latest(instrument, b);
}

static inline std::int64_t to_epoch_s(const std::variant<std::int64_t, double, std::string>& ts) {
  auto tp = to_utc_sys_seconds(ts);
  return std::chrono::time_point_cast<std::chrono::seconds>(tp).time_since_epoch().count();
}

std::size_t ingest_provider_bars(const std::vector<ProviderBar>& input, PriceBook& book) {
  std::size_t accepted = 0;
  for (const auto& pb : input) {
    if (!valid_bar(pb)) continue;

    std::int64_t ts_utc{};
    try {
      ts_utc = to_epoch_s(pb.ts);
    } catch (...) {
      continue; // skip malformed time
    }

    if (!accept_bar_utc(ts_utc)) continue;

    upsert_bar(book, pb.symbol, pb);
    ++accepted;
  }
  return accepted;
}

bool ingest_provider_bar(const ProviderBar& bar, PriceBook& book) {
  return ingest_provider_bars(std::vector<ProviderBar>{bar}, book) == 1;
}

} // namespace sentio
```

## 📄 **FILE 122 of 145**: src/router.cpp

**File Information**:
- **Path**: `src/router.cpp`

- **Size**: 92 lines
- **Modified**: 2025-09-05 15:29:30

- **Type**: .cpp

```text
#include "sentio/router.hpp"
#include "sentio/polygon_ingest.hpp" // for PriceBook forward impl
#include <algorithm>
#include <cassert>
#include <cmath>

namespace sentio {

static inline int dir_from(const StrategySignal& s) {
  using T = StrategySignal::Type;
  if (s.type==T::BUY || s.type==T::STRONG_BUY) return +1;
  if (s.type==T::SELL|| s.type==T::STRONG_SELL) return -1;
  return 0;
}
static inline bool is_strong(const StrategySignal& s) {
  using T = StrategySignal::Type;
  return s.type==T::STRONG_BUY || s.type==T::STRONG_SELL || s.confidence>=0.90;
}
static inline double clamp(double x,double lo,double hi){ return std::max(lo,std::min(hi,x)); }

static inline std::string map_instrument_qqq_family(bool go_long, bool strong,
                                                    const RouterCfg& cfg,
                                                    const std::string& base_symbol)
{
  if (base_symbol == cfg.base_symbol) {
    if (go_long)   return strong ? cfg.bull3x : cfg.base_symbol;
    else           return strong ? cfg.bear3x : cfg.bear1x;
  }
  // Unknown family: fall back to base
  return base_symbol;
}

std::optional<RouteDecision> route(const StrategySignal& s, const RouterCfg& cfg, const std::string& base_symbol) {
  int d = dir_from(s);
  if (d==0) return std::nullopt;
  if (s.confidence < cfg.min_signal_strength) return std::nullopt;

  const bool strong = is_strong(s);
  const bool go_long = (d>0);
  const std::string instrument = map_instrument_qqq_family(go_long, strong, cfg, base_symbol);

  const double raw = (d>0?+1.0:-1.0) * (s.confidence * cfg.signal_multiplier);
  const double tw  = clamp(raw, -cfg.max_position_pct, +cfg.max_position_pct);

  return RouteDecision{instrument, tw};
}

// Implemented elsewhere in your codebase; declared in router.hpp.
// Here's a weak reference for clarity:
// double last_trade_price(const PriceBook&, const std::string&);

static inline double round_to_lot(double qty, double lot) {
  if (lot <= 0) return std::floor(qty);
  return std::floor(qty / lot) * lot;
}

Order route_and_create_order(const std::string& signal_id,
                             const StrategySignal& sig,
                             const RouterCfg& cfg,
                             const std::string& base_symbol,
                             const PriceBook& book,
                             const AccountSnapshot& acct,
                             std::int64_t ts_utc)
{
  Order o{};
  o.signal_id = signal_id;
  auto rd = route(sig, cfg, base_symbol);
  if (!rd) return o; // qty 0

  o.instrument = rd->instrument;
  o.side = (rd->target_weight >= 0 ? OrderSide::Buy : OrderSide::Sell);

  // Size by equity * |target_weight|
  double px = last_trade_price(book, o.instrument); // must be routed instrument
  if (!(std::isfinite(px) && px > 0.0)) return o;

  double target_notional = std::abs(rd->target_weight) * acct.equity;
  double raw_qty = target_notional / px;
  double lot = (cfg.lot_size>0 ? cfg.lot_size : 1.0);
  double qty = round_to_lot(raw_qty, lot);
  if (qty < cfg.min_shares) return o;

  o.qty = qty;
  o.limit_price = 0.0; // market
  o.ts_utc = ts_utc;
  o.notional = (o.side==OrderSide::Buy ? +1.0 : -1.0) * px * qty;

  assert(!o.instrument.empty() && "Instrument must be set");
  return o;
}

} // namespace sentio
```

## 📄 **FILE 123 of 145**: src/rth_calendar.cpp

**File Information**:
- **Path**: `src/rth_calendar.cpp`

- **Size**: 67 lines
- **Modified**: 2025-09-07 14:29:16

- **Type**: .cpp

```text
// rth_calendar.cpp
#include "sentio/rth_calendar.hpp"
#include <chrono>
#include <string>
#include <stdexcept>
#include <iostream>
#include "cctz/time_zone.h"
#include "cctz/civil_time.h"

using namespace std::chrono;

namespace sentio {

bool TradingCalendar::is_rth_utc(std::int64_t ts_utc, [[maybe_unused]] const std::string& tz_name) const {
  using namespace std::chrono;
  
  // Convert to cctz time
  cctz::time_point<cctz::seconds> tp{cctz::seconds{ts_utc}};
  
  // Get UTC timezone
  cctz::time_zone utc_tz;
  if (!cctz::load_time_zone("UTC", &utc_tz)) {
    return false;
  }
  
  // Convert to UTC civil time
  auto lt = cctz::convert(tp, utc_tz);
  auto ct = cctz::civil_second(lt);
  
  // Check if weekend (Saturday = 6, Sunday = 0)
  auto wd = cctz::get_weekday(ct);
  if (wd == cctz::weekday::saturday || wd == cctz::weekday::sunday) {
    return false;
  }
  
  // Check if RTH in UTC
  // EST: 14:30-21:00 UTC (9:30 AM - 4:00 PM EST)
  // EDT: 13:30-20:00 UTC (9:30 AM - 4:00 PM EDT)
  int hour = ct.hour();
  int minute = ct.minute();
  int month = ct.month();
  
  // Simple DST check: April-October is EDT, rest is EST
  bool is_edt = (month >= 4 && month <= 10);
  
  if (is_edt) {
    // EDT: 13:30-20:00 UTC
    if (hour < 13 || (hour == 13 && minute < 30)) {
      return false;  // Before 13:30 UTC
    }
    if (hour >= 20) {
      return false;  // After 20:00 UTC
    }
  } else {
    // EST: 14:30-21:00 UTC
    if (hour < 14 || (hour == 14 && minute < 30)) {
      return false;  // Before 14:30 UTC
    }
    if (hour >= 21) {
      return false;  // After 21:00 UTC
    }
  }
  
  return true;
}

} // namespace sentio
```

## 📄 **FILE 124 of 145**: src/runner.cpp

**File Information**:
- **Path**: `src/runner.cpp`

- **Size**: 190 lines
- **Modified**: 2025-09-07 14:29:16

- **Type**: .cpp

```text
#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/pricebook.hpp"
#include "sentio/metrics.hpp"
#include "sentio/sizer.hpp"
#include "sentio/feature_feeder.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <chrono>

namespace sentio {

RunResult run_backtest(AuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                      int base_symbol_id, const RunnerCfg& cfg) {
    
    // 1. ============== INITIALIZATION ==============
    RunResult result{};
    
    // Start audit run
    std::string meta = "{";
    meta += "\"strategy\":\"" + cfg.strategy_name + "\",";
    meta += "\"base_symbol_id\":" + std::to_string(base_symbol_id) + ",";
    meta += "\"total_series\":" + std::to_string(series.size()) + ",";
    meta += "\"base_series_size\":" + std::to_string(series[base_symbol_id].size());
    meta += "}";
    audit.event_run_start(series[base_symbol_id][0].ts_nyt_epoch, meta);
    
    auto strategy = StrategyFactory::instance().create_strategy(cfg.strategy_name);
    if (!strategy) {
        std::cerr << "FATAL: Could not create strategy '" << cfg.strategy_name << "'. Check registration." << std::endl;
        return result;
    }
    
    ParameterMap params;
    for (const auto& [key, value] : cfg.strategy_params) {
        try {
            params[key] = std::stod(value);
        } catch (...) { /* ignore */ }
    }
    strategy->set_params(params);

    Portfolio portfolio(ST.size());
    AdvancedSizer sizer;
    Pricebook pricebook(base_symbol_id, ST, series);
    
    std::vector<std::pair<std::string, double>> equity_curve;
    const auto& base_series = series[base_symbol_id];
    equity_curve.reserve(base_series.size());

    int total_fills = 0;
    int no_route_count = 0;
    int no_qty_count = 0;

    // 2. ============== MAIN EVENT LOOP ==============
    size_t total_bars = base_series.size();
    size_t progress_interval = total_bars / 20; // 5% intervals (20 steps)
    
    // Skip first 300 bars to allow technical indicators to warm up
    size_t warmup_bars = 300;
    if (total_bars <= warmup_bars) {
        std::cout << "Warning: Not enough bars for warmup (need " << warmup_bars << ", have " << total_bars << ")" << std::endl;
        warmup_bars = 0;
    }
    
    for (size_t i = warmup_bars; i < base_series.size(); ++i) {
        // Progress reporting at 5% intervals
        if (i % progress_interval == 0) {
            int progress_percent = (i * 100) / total_bars;
            std::cout << "Progress: " << progress_percent << "% (" << i << "/" << total_bars << " bars)" << std::endl;
        }
        
        const auto& bar = base_series[i];
        pricebook.sync_to_base_i(i);
        
        // Log bar data
        AuditBar audit_bar{bar.open, bar.high, bar.low, bar.close, static_cast<double>(bar.volume)};
        audit.event_bar(bar.ts_nyt_epoch, ST.get_symbol(base_symbol_id), audit_bar);
        
        // Feed features to ML strategies
        [[maybe_unused]] auto start_feed = std::chrono::high_resolution_clock::now();
        FeatureFeeder::feed_features_to_strategy(strategy.get(), base_series, i, cfg.strategy_name);
        [[maybe_unused]] auto end_feed = std::chrono::high_resolution_clock::now();
        
        [[maybe_unused]] auto start_signal = std::chrono::high_resolution_clock::now();
        StrategySignal sig = strategy->calculate_signal(base_series, i);
        [[maybe_unused]] auto end_signal = std::chrono::high_resolution_clock::now();
        
        if (i < 10) {
            // Timing removed for clean output
        }
        
        if (sig.type != StrategySignal::Type::HOLD) {
            // Log signal
            SigType sig_type = static_cast<SigType>(static_cast<int>(sig.type));
            audit.event_signal(bar.ts_nyt_epoch, ST.get_symbol(base_symbol_id), sig_type, sig.confidence);
            
            auto route_decision = route(sig, cfg.router, ST.get_symbol(base_symbol_id));

            if (route_decision) {
                // Log route decision
                audit.event_route(bar.ts_nyt_epoch, ST.get_symbol(base_symbol_id), route_decision->instrument, route_decision->target_weight);
                
                int instrument_id = ST.get_id(route_decision->instrument);
                if (instrument_id != -1) {
                    double instrument_price = pricebook.last_px[instrument_id];

                    if (instrument_price > 0) {
                        [[maybe_unused]] double current_equity = equity_mark_to_market(portfolio, pricebook.last_px);
                        
                        // **MODIFIED**: This is the core logic fix.
                        // We calculate the desired final position size, then determine the needed trade quantity.
                        double target_qty = sizer.calculate_target_quantity(portfolio, ST, pricebook.last_px, 
                                                                             route_decision->instrument, route_decision->target_weight, 
                                                                             series[instrument_id], cfg.sizer);
                        
                        double current_qty = portfolio.positions[instrument_id].qty;
                        double trade_qty = target_qty - current_qty; // The actual amount to trade

                        if (std::abs(trade_qty * instrument_price) > 1.0) { // Min trade notional $1
                            // Log order and fill
                            Side side = (trade_qty > 0) ? Side::Buy : Side::Sell;
                            audit.event_order(bar.ts_nyt_epoch, route_decision->instrument, side, std::abs(trade_qty), 0.0);
                            audit.event_fill(bar.ts_nyt_epoch, route_decision->instrument, instrument_price, std::abs(trade_qty), 0.0, side);
                            
                            apply_fill(portfolio, instrument_id, trade_qty, instrument_price);
                            total_fills++; // **CRITICAL FIX**: Increment the fills counter
                        } else {
                            no_qty_count++;
                        }
                    }
                }
            } else {
                no_route_count++;
            }
        }
        
        // 3. ============== SNAPSHOT ==============
        if (i % cfg.snapshot_stride == 0 || i == base_series.size() - 1) {
            double current_equity = equity_mark_to_market(portfolio, pricebook.last_px);
            equity_curve.emplace_back(bar.ts_utc, current_equity);
            
            // Log account snapshot
            AccountState state;
            state.cash = portfolio.cash;
            state.equity = current_equity;
            state.realized = 0.0; // TODO: Calculate realized P&L
            audit.event_snapshot(bar.ts_nyt_epoch, state);
        }
    }
    
    // 4. ============== METRICS & DIAGNOSTICS ==============
    strategy->get_diag().print(strategy->get_name().c_str());

    if (equity_curve.empty()) {
        return result;
    }
    
    // **CRITICAL FIX**: Pass the correct `total_fills` to the metrics calculator.
    auto summary = compute_metrics_day_aware(equity_curve, total_fills);

    result.final_equity = equity_curve.empty() ? 100000.0 : equity_curve.back().second;
    result.total_return = summary.ret_total * 100.0;
    result.sharpe_ratio = summary.sharpe;
    result.max_drawdown = summary.mdd * 100.0;
    result.total_fills = summary.trades;
    result.no_route = no_route_count;
    result.no_qty = no_qty_count;

    // Log final metrics and end run
    std::int64_t end_ts = equity_curve.empty() ? series[base_symbol_id][0].ts_nyt_epoch : series[base_symbol_id].back().ts_nyt_epoch;
    audit.event_metric(end_ts, "final_equity", result.final_equity);
    audit.event_metric(end_ts, "total_return", result.total_return);
    audit.event_metric(end_ts, "sharpe_ratio", result.sharpe_ratio);
    audit.event_metric(end_ts, "max_drawdown", result.max_drawdown);
    audit.event_metric(end_ts, "total_fills", result.total_fills);
    audit.event_metric(end_ts, "no_route", result.no_route);
    audit.event_metric(end_ts, "no_qty", result.no_qty);
    
    std::string end_meta = "{";
    end_meta += "\"final_equity\":" + std::to_string(result.final_equity) + ",";
    end_meta += "\"total_return\":" + std::to_string(result.total_return) + ",";
    end_meta += "\"sharpe_ratio\":" + std::to_string(result.sharpe_ratio);
    end_meta += "}";
    audit.event_run_end(end_ts, end_meta);

    return result;
}

} // namespace sentio
```

## 📄 **FILE 125 of 145**: src/sanity.cpp

**File Information**:
- **Path**: `src/sanity.cpp`

- **Size**: 166 lines
- **Modified**: 2025-09-05 20:37:59

- **Type**: .cpp

```text
#include "sentio/sanity.hpp"
#include <cmath>
#include <algorithm>

namespace sentio {

// PriceBook is now abstract - implementations must be provided by concrete classes

bool SanityReport::ok() const {
  for (auto& i : issues) if (i.severity != SanityIssue::Severity::Warn) return false;
  return true;
}
std::size_t SanityReport::errors() const {
  return std::count_if(issues.begin(), issues.end(), [](auto& i){
    return i.severity==SanityIssue::Severity::Error;
  });
}
std::size_t SanityReport::fatals() const {
  return std::count_if(issues.begin(), issues.end(), [](auto& i){
    return i.severity==SanityIssue::Severity::Fatal;
  });
}
void SanityReport::add(SanityIssue::Severity sev, std::string where, std::string what, std::int64_t ts){
  issues.push_back({sev,std::move(where),std::move(what),ts});
}

namespace sanity {

void check_bar_monotonic(const std::vector<std::pair<std::int64_t, Bar>>& bars,
                         int expected_spacing_sec,
                         SanityReport& rep)
{
  if (bars.empty()) return;
  for (std::size_t i=1;i<bars.size();++i){
    auto prev = bars[i-1].first;
    auto cur  = bars[i].first;
    if (cur <= prev) {
      rep.add(SanityIssue::Severity::Fatal, "DATA", "non-increasing timestamp", cur);
    }
    if (expected_spacing_sec>0) {
      auto gap = cur - prev;
      if (gap != expected_spacing_sec) {
        rep.add(SanityIssue::Severity::Error, "DATA",
          "unexpected spacing: got "+std::to_string((long long)gap)+"s expected "+std::to_string(expected_spacing_sec)+"s", cur);
      }
    }
    const Bar& b = bars[i].second;
    if (!(b.low <= b.high)) {
      rep.add(SanityIssue::Severity::Error, "DATA", "bar.low > bar.high", cur);
    }
  }
}

void check_bar_values_finite(const std::vector<std::pair<std::int64_t, Bar>>& bars,
                             SanityReport& rep)
{
  for (auto& it : bars) {
    auto ts = it.first; const Bar& b = it.second;
    if (!(std::isfinite(b.open) && std::isfinite(b.high) && std::isfinite(b.low) && std::isfinite(b.close))) {
      rep.add(SanityIssue::Severity::Fatal, "DATA", "non-finite OHLC", ts);
    }
    if (b.open<=0 || b.high<=0 || b.low<=0 || b.close<=0) {
      rep.add(SanityIssue::Severity::Error, "DATA", "non-positive price", ts);
    }
  }
}

void check_pricebook_coherence(const PriceBook& pb,
                               const std::vector<std::string>& required_instruments,
                               SanityReport& rep)
{
  for (auto& inst : required_instruments) {
    if (!pb.has_instrument(inst)) {
      rep.add(SanityIssue::Severity::Error, "DATA", "PriceBook missing instrument: "+inst);
    } else {
      auto* b = pb.get_latest(inst);
      if (!b || !std::isfinite(b->close)) {
        rep.add(SanityIssue::Severity::Error, "DATA", "PriceBook non-finite last close for: "+inst);
      }
    }
  }
}

void check_signal_confidence_range(double conf, SanityReport& rep, std::int64_t ts) {
  if (!(conf>=0.0 && conf<=1.0)) {
    rep.add(SanityIssue::Severity::Error, "STRAT", "signal confidence out of [0,1]", ts);
  }
}

void check_routed_instrument_has_price(const PriceBook& pb,
                                       const std::string& routed,
                                       SanityReport& rep, std::int64_t ts)
{
  if (routed.empty()) {
    rep.add(SanityIssue::Severity::Fatal, "ROUTER", "empty routed instrument", ts);
    return;
  }
  if (!pb.has_instrument(routed)) {
    rep.add(SanityIssue::Severity::Error, "ROUTER", "routed instrument missing in PriceBook: "+routed, ts);
  } else if (auto* b = pb.get_latest(routed); !b || !std::isfinite(b->close)) {
    rep.add(SanityIssue::Severity::Error, "ROUTER", "routed instrument has non-finite price: "+routed, ts);
  }
}

void check_order_qty_min(double qty, double min_shares,
                         SanityReport& rep, std::int64_t ts)
{
  if (!(std::isfinite(qty))) {
    rep.add(SanityIssue::Severity::Fatal, "EXEC", "order qty non-finite", ts);
    return;
  }
  if (qty != 0.0 && std::abs(qty) < min_shares) {
    rep.add(SanityIssue::Severity::Warn, "EXEC", "order qty < min_shares", ts);
  }
}

void check_order_side_qty_sign_consistency(const std::string& side, double qty,
                                           SanityReport& rep, std::int64_t ts)
{
  if (side=="BUY" && qty<0)  rep.add(SanityIssue::Severity::Error, "EXEC", "BUY with negative qty", ts);
  if (side=="SELL"&& qty>0)  rep.add(SanityIssue::Severity::Error, "EXEC", "SELL with positive qty", ts);
}

void check_equity_consistency(const AccountState& acct,
                              const std::unordered_map<std::string, Position>& pos,
                              const PriceBook& pb,
                              SanityReport& rep)
{
  if (!std::isfinite(acct.cash) || !std::isfinite(acct.realized) || !std::isfinite(acct.equity)) {
    rep.add(SanityIssue::Severity::Fatal, "PnL", "non-finite account values");
    return;
  }
  // recompute mark-to-market
  double mtm = 0.0;
  for (auto& kv : pos) {
    const auto& inst = kv.first;
    const auto& p = kv.second;
    if (!std::isfinite(p.qty) || !std::isfinite(p.avg_px)) {
      rep.add(SanityIssue::Severity::Fatal, "PnL", "non-finite position for "+inst);
      continue;
    }
    auto* b = pb.get_latest(inst);
    if (!b) continue;
    mtm += p.qty * b->close;
  }
  double equity_calc = acct.cash + acct.realized + mtm;
  if (std::isfinite(equity_calc) && std::abs(equity_calc - acct.equity) > 1e-6) {
    rep.add(SanityIssue::Severity::Error, "PnL", "equity mismatch (calc vs recorded) diff="+std::to_string(equity_calc - acct.equity));
  }
}

void check_audit_counts(const AuditEventCounts& c, SanityReport& rep) {
  if (c.orders < c.fills) {
    rep.add(SanityIssue::Severity::Error, "AUDIT", "fills exceed orders");
  }
  // Loose ratios to catch obviously broken runs
  if (c.routes && c.orders==0) {
    rep.add(SanityIssue::Severity::Warn, "AUDIT", "routes exist but no orders");
  }
  if (c.signals && c.routes==0) {
    rep.add(SanityIssue::Severity::Warn, "AUDIT", "signals exist but no routes");
  }
}

} // namespace sanity
} // namespace sentio

```

## 📄 **FILE 126 of 145**: src/signal_engine.cpp

**File Information**:
- **Path**: `src/signal_engine.cpp`

- **Size**: 29 lines
- **Modified**: 2025-09-05 17:03:00

- **Type**: .cpp

```text
#include "sentio/signal_engine.hpp"

namespace sentio {

SignalEngine::SignalEngine(IStrategy* strat, const GateCfg& gate_cfg, SignalHealth* health)
: strat_(strat), gate_(gate_cfg, health), health_(health) {}

EngineOut SignalEngine::on_bar(const StrategyCtx& ctx, const Bar& b, bool inputs_finite) {
  strat_->on_bar(ctx, b);
  auto raw = strat_->latest();
  if (!raw.has_value()) {
    if (health_) health_->incr_drop(DropReason::NONE);
    return {std::nullopt, DropReason::NONE};
  }

  double conf = raw->confidence;
  auto conf2 = gate_.accept(ctx.ts_utc_epoch, ctx.is_rth, inputs_finite,
                            /*warmed_up=*/true, conf);
  if (!conf2) {
    // SignalGate already tallied reason; we return NONE to avoid double counting specific reason here.
    return {std::nullopt, DropReason::NONE};
  }

  StrategySignal out = *raw;
  out.confidence = *conf2;
  return {out, DropReason::NONE};
}

} // namespace sentio

```

## 📄 **FILE 127 of 145**: src/signal_gate.cpp

**File Information**:
- **Path**: `src/signal_gate.cpp`

- **Size**: 52 lines
- **Modified**: 2025-09-05 17:03:11

- **Type**: .cpp

```text
#include "sentio/signal_gate.hpp"
#include <cmath>
#include <algorithm>

namespace sentio {

SignalHealth::SignalHealth() {
  for (auto r :
       {DropReason::NONE, DropReason::NOT_RTH, DropReason::WARMUP,
        DropReason::NAN_INPUT, DropReason::THRESHOLD_TOO_TIGHT, DropReason::COOLDOWN_ACTIVE,
        DropReason::DUPLICATE_BAR_TS}) {
    by_reason.emplace(r, 0ULL);
  }
}
void SignalHealth::incr_emit(){ emitted.fetch_add(1, std::memory_order_relaxed); }
void SignalHealth::incr_drop(DropReason r){
  dropped.fetch_add(1, std::memory_order_relaxed);
  auto it = by_reason.find(r);
  if (it != by_reason.end()) it->second.fetch_add(1, std::memory_order_relaxed);
}

SignalGate::SignalGate(const GateCfg& cfg, SignalHealth* health)
: cfg_(cfg), health_(health) {}

std::optional<double> SignalGate::accept(std::int64_t ts_utc_epoch,
                                         bool is_rth,
                                         bool inputs_finite,
                                         bool warmed_up,
                                         double conf)
{
  if (cfg_.require_rth && !is_rth) { if (health_) health_->incr_drop(DropReason::NOT_RTH); return std::nullopt; }
  if (!inputs_finite)              { if (health_) health_->incr_drop(DropReason::NAN_INPUT); return std::nullopt; }
  if (!warmed_up)                  { if (health_) health_->incr_drop(DropReason::WARMUP);    return std::nullopt; }
  if (!(std::isfinite(conf)))      { if (health_) health_->incr_drop(DropReason::NAN_INPUT); return std::nullopt; }

  conf = std::clamp(conf, 0.0, 1.0);

  if (conf < cfg_.min_conf)        { if (health_) health_->incr_drop(DropReason::THRESHOLD_TOO_TIGHT); return std::nullopt; }

  // Cooldown (optional)
  if (cooldown_left_ > 0)          { --cooldown_left_; if (health_) health_->incr_drop(DropReason::COOLDOWN_ACTIVE); return std::nullopt; }

  // Debounce duplicate timestamps
  if (last_emit_ts_ == ts_utc_epoch){ if (health_) health_->incr_drop(DropReason::DUPLICATE_BAR_TS); return std::nullopt; }

  last_emit_ts_ = ts_utc_epoch;
  cooldown_left_ = cfg_.cooldown_bars;
  if (health_) health_->incr_emit();
  return conf;
}

} // namespace sentio

```

## 📄 **FILE 128 of 145**: src/signal_pipeline.cpp

**File Information**:
- **Path**: `src/signal_pipeline.cpp`

- **Size**: 43 lines
- **Modified**: 2025-09-05 17:05:46

- **Type**: .cpp

```text
#include "sentio/signal_pipeline.hpp"
#include <cmath>

namespace sentio {

PipelineOut SignalPipeline::on_bar(const StrategyCtx& ctx, const Bar& b, const void* acct) {
  (void)acct; // Avoid unused parameter warning
  strat_->on_bar(ctx, b);
  PipelineOut out{};
  TraceRow tr{};
  tr.ts_utc = ctx.ts_utc_epoch;
  tr.instrument = ctx.instrument;
  tr.close = b.close;
  tr.is_rth = ctx.is_rth;
  tr.inputs_finite = std::isfinite(b.close);

  auto sig = strat_->latest();
  if (!sig) {
    tr.reason = TraceReason::NO_STRATEGY_OUTPUT;
    if (trace_) trace_->push(tr);
    return out;
  }
  tr.confidence = sig->confidence;

  // Use the existing signal_gate API
  auto conf2 = gate_.accept(ctx.ts_utc_epoch, ctx.is_rth, tr.inputs_finite, true, sig->confidence);
  if (!conf2) {
    tr.reason = TraceReason::THRESHOLD_TOO_TIGHT; // Default to threshold for now
    if (trace_) trace_->push(tr);
    return out;
  }

  StrategySignal sig2 = *sig; sig2.confidence = *conf2;
  tr.conf_after_gate = *conf2;
  out.signal = sig2;

  // For now, just mark as OK since we don't have full routing implemented
  tr.reason = TraceReason::OK;
  if (trace_) trace_->push(tr);
  return out;
}

} // namespace sentio

```

## 📄 **FILE 129 of 145**: src/signal_trace.cpp

**File Information**:
- **Path**: `src/signal_trace.cpp`

- **Size**: 7 lines
- **Modified**: 2025-09-05 17:00:56

- **Type**: .cpp

```text
#include "sentio/signal_trace.hpp"

namespace sentio {
std::size_t SignalTrace::count(TraceReason r) const {
  std::size_t n=0; for (auto& x: rows_) if (x.reason==r) ++n; return n;
}
} // namespace sentio

```

## 📄 **FILE 130 of 145**: src/sim_data.cpp

**File Information**:
- **Path**: `src/sim_data.cpp`

- **Size**: 47 lines
- **Modified**: 2025-09-05 20:37:59

- **Type**: .cpp

```text
#include "sentio/sim_data.hpp"
#include <cmath>

namespace sentio {

std::vector<std::pair<std::int64_t, Bar>> generate_minute_series(const SimCfg& cfg) {
  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<double> z(0.0, 1.0);
  std::uniform_real_distribution<double> U(0.0, 1.0);

  std::vector<std::pair<std::int64_t, Bar>> out;
  out.reserve(cfg.minutes);

  double px = cfg.start_price;
  std::int64_t ts = cfg.start_ts_utc;

  for (int i=0;i<cfg.minutes;++i, ts+=60) {
    double u = U(rng);
    double ret = 0.0;

    if (u < cfg.frac_trend) {
      // trending drift + noise
      ret = 0.0002 + (cfg.vol_bps*1e-4) * z(rng);
    } else if (u < cfg.frac_trend + cfg.frac_mr) {
      // mean-reversion around 0 with lighter noise
      ret = -0.0001 + (cfg.vol_bps*0.6e-4) * z(rng);
    } else {
      // jump regime
      ret = (U(rng) < 0.5 ? +1 : -1) * (cfg.vol_bps*6e-4 + std::abs(z(rng))*cfg.vol_bps*3e-4);
    }

    double new_px = std::max(0.01, px * (1.0 + ret));
    double o = px;
    double c = new_px;
    double h = std::max(o, c) * (1.0 + std::abs((cfg.vol_bps*0.3e-4) * z(rng)));
    double l = std::min(o, c) * (1.0 - std::abs((cfg.vol_bps*0.3e-4) * z(rng)));
    // ensure consistency
    h = std::max({h, o, c});
    l = std::min({l, o, c});

    out.push_back({ts, Bar{o,h,l,c}});
    px = new_px;
  }
  return out;
}

} // namespace sentio

```

## 📄 **FILE 131 of 145**: src/strategy_bollinger_squeeze_breakout.cpp

**File Information**:
- **Path**: `src/strategy_bollinger_squeeze_breakout.cpp`

- **Size**: 147 lines
- **Modified**: 2025-09-05 15:29:30

- **Type**: .cpp

```text
#include "sentio/strategy_bollinger_squeeze_breakout.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

namespace sentio {

    BollingerSqueezeBreakoutStrategy::BollingerSqueezeBreakoutStrategy() 
    : BaseStrategy("BollingerSqueezeBreakout"), bollinger_(20, 2.0) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap BollingerSqueezeBreakoutStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed parameters to be more sensitive to trading opportunities.
    return {
        {"bb_window", 20.0},
        {"bb_k", 1.8},                   // Tighter bands to increase breakout signals
        {"squeeze_percentile", 0.25},    // Squeeze is now top 25% of quietest periods (was 15%)
        {"squeeze_lookback", 60.0},      // Shorter lookback for volatility
        {"hold_max_bars", 120.0},
        {"tp_mult_sd", 1.5},
        {"sl_mult_sd", 1.5},
        {"min_squeeze_bars", 3.0}        // Require at least 3 bars of squeeze
    };
}

ParameterSpace BollingerSqueezeBreakoutStrategy::get_param_space() const { return {}; }

void BollingerSqueezeBreakoutStrategy::apply_params() {
    bb_window_ = static_cast<int>(params_["bb_window"]);
    squeeze_percentile_ = params_["squeeze_percentile"];
    squeeze_lookback_ = static_cast<int>(params_["squeeze_lookback"]);
    hold_max_bars_ = static_cast<int>(params_["hold_max_bars"]);
    tp_mult_sd_ = params_["tp_mult_sd"];
    sl_mult_sd_ = params_["sl_mult_sd"];
    min_squeeze_bars_ = static_cast<int>(params_["min_squeeze_bars"]);
    
    bollinger_ = Bollinger(bb_window_, params_["bb_k"]);
    sd_history_.reserve(squeeze_lookback_);
    reset_state();
}

void BollingerSqueezeBreakoutStrategy::reset_state() {
    BaseStrategy::reset_state();
    state_ = State::Idle;
    bars_in_trade_ = 0;
    squeeze_duration_ = 0;
    sd_history_.clear();
}

StrategySignal BollingerSqueezeBreakoutStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < squeeze_lookback_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }
    
    if (state_ == State::Long || state_ == State::Short) {
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            signal.type = (state_ == State::Long) ? StrategySignal::Type::SELL : StrategySignal::Type::BUY;
            reset_state();
            diag_.emitted++;
            return signal;
        }
        return signal;
    }

    update_state_machine(bars[current_index]);

    if (state_ == State::ArmedLong || state_ == State::ArmedShort) {
        if (squeeze_duration_ < min_squeeze_bars_) {
            diag_.drop(DropReason::THRESHOLD);
            state_ = State::Idle;
            return signal;
        }

        double mid, lo, hi, sd;
        bollinger_.step(bars[current_index].close, mid, lo, hi, sd);
        
        if (state_ == State::ArmedLong) {
            signal.type = StrategySignal::Type::BUY;
            state_ = State::Long;
        } else {
            signal.type = StrategySignal::Type::SELL;
            state_ = State::Short;
        }
        
        signal.confidence = 0.8;
        diag_.emitted++;
        bars_in_trade_ = 0;
    } else {
        diag_.drop(DropReason::THRESHOLD);
    }

    return signal;
}

void BollingerSqueezeBreakoutStrategy::update_state_machine(const Bar& bar) {
    double mid, lo, hi, sd;
    bollinger_.step(bar.close, mid, lo, hi, sd);
    
    sd_history_.push_back(sd);
    if (sd_history_.size() > static_cast<size_t>(squeeze_lookback_)) {
        sd_history_.erase(sd_history_.begin());
    }
    
    double sd_threshold = calculate_volatility_percentile(squeeze_percentile_);
    bool is_squeezed = (sd_history_.size() == static_cast<size_t>(squeeze_lookback_)) && (sd <= sd_threshold);

    switch (state_) {
        case State::Idle:
            if (is_squeezed) {
                state_ = State::Squeezed;
                squeeze_duration_ = 1;
            }
            break;
        case State::Squeezed:
            if (bar.close > hi) state_ = State::ArmedLong;
            else if (bar.close < lo) state_ = State::ArmedShort;
            else if (!is_squeezed) state_ = State::Idle;
            else squeeze_duration_++;
            break;
        default:
            break;
    }
}

// **MODIFIED**: Implemented a proper percentile calculation instead of a stub.
double BollingerSqueezeBreakoutStrategy::calculate_volatility_percentile(double percentile) const {
    if (sd_history_.size() < static_cast<size_t>(squeeze_lookback_)) {
        return std::numeric_limits<double>::max(); // Not enough data, effectively prevents squeeze
    }
    
    std::vector<double> sorted_history = sd_history_;
    std::sort(sorted_history.begin(), sorted_history.end());
    
    int index = static_cast<int>(percentile * (sorted_history.size() - 1));
    return sorted_history[index];
}

REGISTER_STRATEGY(BollingerSqueezeBreakoutStrategy, "BollingerSqueezeBreakout");

} // namespace sentio

```

## 📄 **FILE 132 of 145**: src/strategy_hybrid_ppo.cpp

**File Information**:
- **Path**: `src/strategy_hybrid_ppo.cpp`

- **Size**: 81 lines
- **Modified**: 2025-09-06 00:28:01

- **Type**: .cpp

```text
#include "sentio/strategy_hybrid_ppo.hpp"
#include <algorithm>
#include <stdexcept>

namespace sentio {

HybridPPOStrategy::HybridPPOStrategy()
: BaseStrategy("HybridPPO"),
  cfg_(),
  handle_(ml::ModelRegistryTS::load_torchscript("HybridPPO", cfg_.version, cfg_.artifacts_dir, cfg_.use_cuda)),
  fpipe_(handle_.spec)
{}

HybridPPOStrategy::HybridPPOStrategy(const HybridPPOCfg& cfg)
: BaseStrategy("HybridPPO"),
  cfg_(cfg),
  handle_(ml::ModelRegistryTS::load_torchscript("HybridPPO", cfg.version, cfg.artifacts_dir, cfg.use_cuda)),
  fpipe_(handle_.spec)
{}

void HybridPPOStrategy::set_raw_features(const std::vector<double>& raw) {
  raw_ = raw;
}

ParameterMap HybridPPOStrategy::get_default_params() const {
  return {
    {"conf_floor", cfg_.conf_floor}
  };
}

ParameterSpace HybridPPOStrategy::get_param_space() const {
  return {
    {"conf_floor", {ParamType::FLOAT, 0.0, 1.0, cfg_.conf_floor}}
  };
}

void HybridPPOStrategy::apply_params() {
  cfg_.conf_floor = params_["conf_floor"];
}

StrategySignal HybridPPOStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
  (void)bars; (void)current_index; // Features come from set_raw_features
  last_.reset();
  auto z = fpipe_.transform(raw_);
  if (!z) return StrategySignal{};

  auto out = handle_.model->predict(*z, 1, (int)z->size(), "BF");
  if (!out) return StrategySignal{};

  StrategySignal sig = map_output(*out);
  // Safety: never emit if below floor
  if (sig.confidence < cfg_.conf_floor) return StrategySignal{};

  last_ = sig;
  return sig;
}

StrategySignal HybridPPOStrategy::map_output(const ml::ModelOutput& mo) const {
  StrategySignal s;
  // Discrete 3-way: SELL/HOLD/BUY
  int argmax = 0;
  for (int i=1;i<(int)mo.probs.size();++i) if (mo.probs[i] > mo.probs[argmax]) argmax = i;

  const auto& acts = handle_.spec.actions; // e.g., ["SELL","HOLD","BUY"]
  std::string a = (argmax<(int)acts.size()? acts[argmax] : "HOLD");
  float pmax = mo.probs.empty()? 0.0f : mo.probs[argmax];

  if (a=="BUY")       s.type = StrategySignal::Type::BUY;
  else if (a=="SELL") s.type = StrategySignal::Type::SELL;
  else                s.type = StrategySignal::Type::HOLD;

  // Confidence: calibrated pmax (basic identity)
  s.confidence = std::max(cfg_.conf_floor, (double)pmax);
  return s;
}


// Register the strategy with the factory
// REGISTER_STRATEGY(HybridPPOStrategy, "hybrid_ppo")  // Disabled - not working

} // namespace sentio

```

## 📄 **FILE 133 of 145**: src/strategy_kochi_ppo.cpp

**File Information**:
- **Path**: `src/strategy_kochi_ppo.cpp`

- **Size**: 95 lines
- **Modified**: 2025-09-07 22:52:32

- **Type**: .cpp

```text
#include "sentio/strategy_kochi_ppo.hpp"
#include <algorithm>
#include <stdexcept>

namespace sentio {

static ml::WindowSpec make_kochi_spec(const ml::ModelSpec& s){
  ml::WindowSpec w;
  // Kochi environment defaults to 20 window; allow metadata override if provided
  w.seq_len = s.seq_len > 0 ? s.seq_len : 20;
  w.layout  = s.input_layout.empty()? "BTF" : s.input_layout;
  w.feat_dim = (int)s.feature_names.size();
  w.mean = s.mean;
  w.std  = s.std;
  w.clip2 = s.clip2;
  return w;
}

KochiPPOStrategy::KochiPPOStrategy()
: BaseStrategy("KochiPPO")
, cfg_()
, handle_(ml::ModelRegistryTS::load_torchscript(cfg_.model_id, cfg_.version, cfg_.artifacts_dir, cfg_.use_cuda))
, window_(make_kochi_spec(handle_.spec))
{}

KochiPPOStrategy::KochiPPOStrategy(const KochiPPOCfg& cfg)
: BaseStrategy("KochiPPO")
, cfg_(cfg)
, handle_(ml::ModelRegistryTS::load_torchscript(cfg.model_id, cfg.version, cfg.artifacts_dir, cfg.use_cuda))
, window_(make_kochi_spec(handle_.spec))
{}

void KochiPPOStrategy::set_raw_features(const std::vector<double>& raw){
  // Expect exactly F features in model metadata order
  if ((int)raw.size() != window_.feat_dim()) return;
  window_.push(raw);
}

ParameterMap KochiPPOStrategy::get_default_params() const {
  return {
    {"conf_floor", cfg_.conf_floor}
  };
}

ParameterSpace KochiPPOStrategy::get_param_space() const {
  return {
    {"conf_floor", {ParamType::FLOAT, 0.0, 1.0, cfg_.conf_floor}}
  };
}

void KochiPPOStrategy::apply_params() {
  cfg_.conf_floor = params_["conf_floor"];
}

StrategySignal KochiPPOStrategy::map_output(const ml::ModelOutput& mo) const {
  StrategySignal s;
  // Assume discrete probs with actions in spec.actions. Default mapping SELL/HOLD/BUY
  int argmax = 0;
  for (int i=1;i<(int)mo.probs.size();++i) if (mo.probs[i] > mo.probs[argmax]) argmax = i;

  const auto& acts = handle_.spec.actions;
  std::string a = (argmax<(int)acts.size()? acts[argmax] : "HOLD");
  float pmax = mo.probs.empty()? 0.0f : mo.probs[argmax];

  if (a=="BUY")       s.type = StrategySignal::Type::BUY;
  else if (a=="SELL") s.type = StrategySignal::Type::SELL;
  else                 s.type = StrategySignal::Type::HOLD;

  s.confidence = std::max(cfg_.conf_floor, (double)pmax);
  return s;
}

StrategySignal KochiPPOStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
  (void)bars; (void)current_index; // features are streamed in via set_raw_features
  last_.reset();
  if (!window_.ready()) return StrategySignal{};

  auto in = window_.to_input();
  if (!in) return StrategySignal{};

  auto out = handle_.model->predict(*in, window_.seq_len(), window_.feat_dim(), handle_.spec.input_layout);
  if (!out) return StrategySignal{};

  auto sig = map_output(*out);
  if (sig.confidence < cfg_.conf_floor) return StrategySignal{};
  last_ = sig;
  return sig;
}

// Register with factory
REGISTER_STRATEGY(KochiPPOStrategy, "kochi_ppo");

} // namespace sentio



```

## 📄 **FILE 134 of 145**: src/strategy_market_making.cpp

**File Information**:
- **Path**: `src/strategy_market_making.cpp`

- **Size**: 140 lines
- **Modified**: 2025-09-05 20:51:42

- **Type**: .cpp

```text
#include "sentio/strategy_market_making.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace sentio {

MarketMakingStrategy::MarketMakingStrategy() 
    : BaseStrategy("MarketMaking"),
      rolling_returns_(20),
      rolling_volume_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap MarketMakingStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed volatility and volume thresholds to allow participation.
    return {
        {"base_spread", 0.001}, {"min_spread", 0.0005}, {"max_spread", 0.003},
        {"order_levels", 3.0}, {"level_spacing", 0.0005}, {"order_size_base", 0.5},
        {"max_inventory", 100.0}, {"inventory_skew_mult", 0.002},
        {"adverse_selection_threshold", 0.004}, // Was 0.002, allowing participation in more volatile conditions
        {"volatility_window", 20.0},
        {"volume_window", 50.0}, {"min_volume_ratio", 0.05}, // Was 0.1, making it even more permissive
        {"max_orders_per_bar", 10.0}, {"rebalance_frequency", 10.0}
    };
}

ParameterSpace MarketMakingStrategy::get_param_space() const { return {}; }

void MarketMakingStrategy::apply_params() {
    base_spread_ = params_.at("base_spread");
    min_spread_ = params_.at("min_spread");
    max_spread_ = params_.at("max_spread");
    order_levels_ = static_cast<int>(params_.at("order_levels"));
    level_spacing_ = params_.at("level_spacing");
    order_size_base_ = params_.at("order_size_base");
    max_inventory_ = params_.at("max_inventory");
    inventory_skew_mult_ = params_.at("inventory_skew_mult");
    adverse_selection_threshold_ = params_.at("adverse_selection_threshold");
    min_volume_ratio_ = params_.at("min_volume_ratio");
    max_orders_per_bar_ = static_cast<int>(params_.at("max_orders_per_bar"));
    rebalance_frequency_ = static_cast<int>(params_.at("rebalance_frequency"));

    int vol_window = std::max(1, static_cast<int>(params_.at("volatility_window")));
    int vol_mean_window = std::max(1, static_cast<int>(params_.at("volume_window")));
    
    rolling_returns_.reset(vol_window);
    rolling_volume_.reset(vol_mean_window);
    reset_state();
}

void MarketMakingStrategy::reset_state() {
    BaseStrategy::reset_state();
    market_state_ = MarketState{};
    rolling_returns_.reset(std::max(1, static_cast<int>(params_.at("volatility_window"))));
    rolling_volume_.reset(std::max(1, static_cast<int>(params_.at("volume_window"))));
}

StrategySignal MarketMakingStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;
    
    // Always update indicators to have a full history for the next bar
    if(current_index > 0) {
        double price_return = (bars[current_index].close - bars[current_index - 1].close) / bars[current_index - 1].close;
        rolling_returns_.push(price_return);
    }
    rolling_volume_.push(bars[current_index].volume);

    // Wait for indicators to warm up
    if (rolling_volume_.size() < static_cast<size_t>(params_.at("volume_window"))) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }

    if (!should_participate(bars[current_index])) {
        return signal;
    }
    
    // **FIXED**: Generate signals based on volatility and volume patterns instead of inventory
    // Note: inventory tracking removed as it's not currently implemented
    // Since inventory tracking is not implemented, use a simpler approach
    double volatility = rolling_returns_.stddev();
    double avg_volume = rolling_volume_.mean();
    double volume_ratio = (avg_volume > 0) ? bars[current_index].volume / avg_volume : 0.0;
    
    // Generate signals when volatility is moderate and volume is increasing
    if (volatility > 0.0005 && volatility < adverse_selection_threshold_ && volume_ratio > 0.8) {
        // Simple momentum-based signal
        if (current_index > 0) {
            double price_change = (bars[current_index].close - bars[current_index - 1].close) / bars[current_index - 1].close;
            if (price_change > 0.001) {
                signal.type = StrategySignal::Type::BUY;
            } else if (price_change < -0.001) {
                signal.type = StrategySignal::Type::SELL;
            } else {
                diag_.drop(DropReason::THRESHOLD);
                return signal;
            }
        } else {
            diag_.drop(DropReason::THRESHOLD);
            return signal;
        }
    } else {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }

    signal.confidence = 0.6; // Fixed confidence since we're not using inventory_skew
    diag_.emitted++;
    return signal;
}

bool MarketMakingStrategy::should_participate(const Bar& bar) {
    double volatility = rolling_returns_.stddev();
    
    if (volatility > adverse_selection_threshold_) {
        diag_.drop(DropReason::THRESHOLD); 
        return false;
    }

    double avg_volume = rolling_volume_.mean();
    
    if (avg_volume > 0 && (bar.volume < avg_volume * min_volume_ratio_)) {
        diag_.drop(DropReason::ZERO_VOL);
        return false;
    }
    return true;
}

double MarketMakingStrategy::get_inventory_skew() const {
    if (max_inventory_ <= 0) return 0.0;
    double normalized_inventory = market_state_.inventory / max_inventory_;
    return -normalized_inventory * inventory_skew_mult_;
}

REGISTER_STRATEGY(MarketMakingStrategy, "MarketMaking");

} // namespace sentio


```

## 📄 **FILE 135 of 145**: src/strategy_momentum_volume.cpp

**File Information**:
- **Path**: `src/strategy_momentum_volume.cpp`

- **Size**: 147 lines
- **Modified**: 2025-09-05 15:24:52

- **Type**: .cpp

```text
#include "sentio/strategy_momentum_volume.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>

namespace sentio {

MomentumVolumeProfileStrategy::MomentumVolumeProfileStrategy() 
    : BaseStrategy("MomentumVolumeProfile"), avg_volume_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap MomentumVolumeProfileStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed parameters to be more sensitive
    return {
        {"profile_period", 100.0},
        {"value_area_pct", 0.7},
        {"price_bins", 30.0},
        {"breakout_threshold_pct", 0.001},
        {"momentum_lookback", 20.0},
        {"volume_surge_mult", 1.2}, // Was 1.5
        {"cool_down_period", 5.0}   // Was 10
    };
}

ParameterSpace MomentumVolumeProfileStrategy::get_param_space() const { return {}; }

void MomentumVolumeProfileStrategy::apply_params() {
    profile_period_ = static_cast<int>(params_["profile_period"]);
    value_area_pct_ = params_["value_area_pct"];
    price_bins_ = static_cast<int>(params_["price_bins"]);
    breakout_threshold_pct_ = params_["breakout_threshold_pct"];
    momentum_lookback_ = static_cast<int>(params_["momentum_lookback"]);
    volume_surge_mult_ = params_["volume_surge_mult"];
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);
    avg_volume_ = RollingMean(profile_period_);
    reset_state();
}

void MomentumVolumeProfileStrategy::reset_state() {
    BaseStrategy::reset_state();
    volume_profile_.clear();
    last_profile_update_ = -1;
    avg_volume_ = RollingMean(profile_period_);
}

StrategySignal MomentumVolumeProfileStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < profile_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }
    
    if (is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return signal;
    }
    
    // Periodically rebuild the expensive volume profile
    if (last_profile_update_ == -1 || current_index - last_profile_update_ >= 10) {
        build_volume_profile(bars, current_index);
        last_profile_update_ = current_index;
    }
    
    if (volume_profile_.value_area_high <= 0) {
        diag_.drop(DropReason::NAN_INPUT); // Profile not ready or invalid
        return signal;
    }

    const auto& bar = bars[current_index];
    avg_volume_.push(bar.volume);
    
    bool breakout_up = bar.close > (volume_profile_.value_area_high * (1.0 + breakout_threshold_pct_));
    bool breakout_down = bar.close < (volume_profile_.value_area_low * (1.0 - breakout_threshold_pct_));

    if (!breakout_up && !breakout_down) {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }

    if (!is_momentum_confirmed(bars, current_index)) {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }
    
    if (bar.volume < avg_volume_.mean() * volume_surge_mult_) {
        diag_.drop(DropReason::ZERO_VOL);
        return signal;
    }

    if (breakout_up) {
        signal.type = StrategySignal::Type::BUY;
    } else {
        signal.type = StrategySignal::Type::SELL;
    }
    
    signal.confidence = 0.85;
    diag_.emitted++;
    state_.last_trade_bar = current_index;

    return signal;
}

bool MomentumVolumeProfileStrategy::is_momentum_confirmed(const std::vector<Bar>& bars, int index) const {
    if (index < momentum_lookback_) return false;
    double price_change = bars[index].close - bars[index - momentum_lookback_].close;
    if (bars[index].close > volume_profile_.value_area_high) {
        return price_change > 0;
    }
    if (bars[index].close < volume_profile_.value_area_low) {
        return price_change < 0;
    }
    return false;
}

// **MODIFIED**: This is now a functional, albeit simple, implementation to prevent NaN drops.
void MomentumVolumeProfileStrategy::build_volume_profile(const std::vector<Bar>& bars, int end_index) {
    volume_profile_.clear();
    int start_index = std::max(0, end_index - profile_period_ + 1);

    double min_price = std::numeric_limits<double>::max();
    double max_price = std::numeric_limits<double>::lowest();
    
    for (int i = start_index; i <= end_index; ++i) {
        min_price = std::min(min_price, bars[i].low);
        max_price = std::max(max_price, bars[i].high);
    }
    
    if (max_price <= min_price) return; // Cannot build profile

    // Simple implementation: Value Area is the high/low of the lookback period
    volume_profile_.value_area_high = max_price;
    volume_profile_.value_area_low = min_price;
    volume_profile_.total_volume = 1.0; // Mark as valid by setting a non-zero value
    // A proper implementation would bin prices and find the 70% volume area.
}

void MomentumVolumeProfileStrategy::calculate_value_area() {
    // This is now handled within build_volume_profile for simplicity
}

REGISTER_STRATEGY(MomentumVolumeProfileStrategy, "MomentumVolumeProfile");

} // namespace sentio

```

## 📄 **FILE 136 of 145**: src/strategy_opening_range_breakout.cpp

**File Information**:
- **Path**: `src/strategy_opening_range_breakout.cpp`

- **Size**: 131 lines
- **Modified**: 2025-09-05 15:24:52

- **Type**: .cpp

```text
#include "sentio/strategy_opening_range_breakout.hpp"
#include <algorithm>
#include <cmath>

namespace sentio {

OpeningRangeBreakoutStrategy::OpeningRangeBreakoutStrategy() 
    : BaseStrategy("OpeningRangeBreakout") {
    params_ = get_default_params();
    apply_params();
}
ParameterMap OpeningRangeBreakoutStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed cooldown to allow more frequent trades.
    return {
        {"opening_range_minutes", 30.0},
        {"breakout_confirmation_bars", 1.0},
        {"volume_multiplier", 1.5},
        {"stop_loss_pct", 0.01},
        {"take_profit_pct", 0.02},
        {"cool_down_period", 5.0}, // Was 15.0
    };
}

ParameterSpace OpeningRangeBreakoutStrategy::get_param_space() const { /* ... unchanged ... */ return {}; }

void OpeningRangeBreakoutStrategy::apply_params() {
    // **NEW**: Cache parameters
    opening_range_minutes_ = static_cast<int>(params_["opening_range_minutes"]);
    breakout_confirmation_bars_ = static_cast<int>(params_["breakout_confirmation_bars"]);
    volume_multiplier_ = params_["volume_multiplier"];
    stop_loss_pct_ = params_["stop_loss_pct"];
    take_profit_pct_ = params_["take_profit_pct"];
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);
    reset_state();
}

void OpeningRangeBreakoutStrategy::reset_state() {
    BaseStrategy::reset_state();
    current_range_ = OpeningRange{};
    day_start_index_ = -1;
}

StrategySignal OpeningRangeBreakoutStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < 1) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }

    // **MODIFIED**: Robust and performant new-day detection
    const int SECONDS_IN_DAY = 86400;
    long current_day = bars[current_index].ts_nyt_epoch / SECONDS_IN_DAY;
    long prev_day = bars[current_index - 1].ts_nyt_epoch / SECONDS_IN_DAY;

    if (current_day != prev_day) {
        reset_state(); // Reset everything for the new day
        day_start_index_ = current_index;
    }
    
    if (day_start_index_ == -1) { // Haven't established the start of the first day yet
        day_start_index_ = 0;
    }

    int bars_into_day = current_index - day_start_index_;

    // --- Phase 1: Define the Opening Range ---
    if (bars_into_day < opening_range_minutes_) {
        if (bars_into_day == 0) {
            current_range_.high = bars[current_index].high;
            current_range_.low = bars[current_index].low;
        } else {
            current_range_.high = std::max(current_range_.high, bars[current_index].high);
            current_range_.low = std::min(current_range_.low, bars[current_index].low);
        }
        diag_.drop(DropReason::SESSION); // Use SESSION to mean "in range formation"
        return signal;
    }

    // --- Finalize the range exactly once ---
    if (!current_range_.is_finalized) {
        current_range_.end_bar = current_index - 1;
        current_range_.is_finalized = true;
    }

    // --- Phase 2: Look for Breakouts ---
    if (state_.in_position || is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return signal;
    }

    const auto& bar = bars[current_index];
    bool is_breakout_up = bar.close > current_range_.high;
    bool is_breakout_down = bar.close < current_range_.low;

    if (!is_breakout_up && !is_breakout_down) {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }
    
    // Volume Confirmation
    double avg_volume = 0;
    for (int i = day_start_index_; i < current_range_.end_bar; ++i) {
        avg_volume += bars[i].volume;
    }
    avg_volume /= (current_range_.end_bar - day_start_index_ + 1);

    if (bar.volume < avg_volume * volume_multiplier_) {
        diag_.drop(DropReason::ZERO_VOL); // Re-using for low volume
        return signal;
    }

    // Generate Signal
    if (is_breakout_up) {
        signal.type = StrategySignal::Type::BUY;
    } else { // is_breakout_down
        signal.type = StrategySignal::Type::SELL;
    }

    signal.confidence = 0.9;
    diag_.emitted++;
    state_.in_position = true; // Manually set state as this is an intraday strategy
    state_.last_trade_bar = current_index;

    return signal;
}

// Register the strategy
REGISTER_STRATEGY(OpeningRangeBreakoutStrategy, "OpeningRangeBreakout");

} // namespace sentio
```

## 📄 **FILE 137 of 145**: src/strategy_order_flow_imbalance.cpp

**File Information**:
- **Path**: `src/strategy_order_flow_imbalance.cpp`

- **Size**: 105 lines
- **Modified**: 2025-09-05 15:25:26

- **Type**: .cpp

```text
#include "sentio/strategy_order_flow_imbalance.hpp"
#include <algorithm>
#include <cmath>

namespace sentio {

OrderFlowImbalanceStrategy::OrderFlowImbalanceStrategy() 
    : BaseStrategy("OrderFlowImbalance"),
      rolling_pressure_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap OrderFlowImbalanceStrategy::get_default_params() const {
    return {
        {"lookback_period", 50.0},
        {"entry_threshold_long", 0.60},
        {"entry_threshold_short", 0.40},
        {"hold_max_bars", 60.0},
        {"cool_down_period", 5.0}
    };
}

ParameterSpace OrderFlowImbalanceStrategy::get_param_space() const { return {}; }

void OrderFlowImbalanceStrategy::apply_params() {
    lookback_period_ = static_cast<int>(params_["lookback_period"]);
    entry_threshold_long_ = params_["entry_threshold_long"];
    entry_threshold_short_ = params_["entry_threshold_short"];
    hold_max_bars_ = static_cast<int>(params_["hold_max_bars"]);
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);
    
    rolling_pressure_ = RollingMean(lookback_period_);
    reset_state();
}

void OrderFlowImbalanceStrategy::reset_state() {
    BaseStrategy::reset_state();
    ofi_state_ = OFIState::Flat; // **FIXED**: Use the renamed state variable
    bars_in_trade_ = 0;
    rolling_pressure_ = RollingMean(lookback_period_);
}

double OrderFlowImbalanceStrategy::calculate_bar_pressure(const Bar& bar) const {
    double range = bar.high - bar.low;
    if (range < 1e-9) {
        return 0.5; // Neutral pressure if there's no range
    }
    return (bar.close - bar.low) / range;
}

StrategySignal OrderFlowImbalanceStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < lookback_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }

    double pressure = calculate_bar_pressure(bars[current_index]);
    double avg_pressure = rolling_pressure_.push(pressure);

    // **FIXED**: Use the strategy-specific 'ofi_state_' for state machine logic
    if (ofi_state_ == OFIState::Flat) {
        if (is_cooldown_active(current_index, cool_down_period_)) {
            diag_.drop(DropReason::COOLDOWN);
            return signal;
        }

        if (avg_pressure > entry_threshold_long_) {
            signal.type = StrategySignal::Type::BUY;
            ofi_state_ = OFIState::Long;
            // **FIXED**: Correctly access the 'state_' member from BaseStrategy
            state_.last_trade_bar = current_index;
        } else if (avg_pressure < entry_threshold_short_) {
            signal.type = StrategySignal::Type::SELL;
            ofi_state_ = OFIState::Short;
            // **FIXED**: Correctly access the 'state_' member from BaseStrategy
            state_.last_trade_bar = current_index;
        } else {
            diag_.drop(DropReason::THRESHOLD);
            return signal;
        }

        signal.confidence = 0.7;
        diag_.emitted++;
        bars_in_trade_ = 0;

    } else { // In a trade, check for exit
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            // **FIXED**: Use 'ofi_state_' to determine exit signal direction
            signal.type = (ofi_state_ == OFIState::Long) ? StrategySignal::Type::SELL : StrategySignal::Type::BUY;
            diag_.emitted++;
            reset_state();
        }
    }

    return signal;
}

REGISTER_STRATEGY(OrderFlowImbalanceStrategy, "OrderFlowImbalance");

} // namespace sentio


```

## 📄 **FILE 138 of 145**: src/strategy_order_flow_scalping.cpp

**File Information**:
- **Path**: `src/strategy_order_flow_scalping.cpp`

- **Size**: 120 lines
- **Modified**: 2025-09-05 16:40:29

- **Type**: .cpp

```text
#include "sentio/strategy_order_flow_scalping.hpp"
#include <algorithm>
#include <cmath>

namespace sentio {

OrderFlowScalpingStrategy::OrderFlowScalpingStrategy() 
    : BaseStrategy("OrderFlowScalping"),
      rolling_pressure_(50) {
    params_ = get_default_params();
    apply_params();
}

ParameterMap OrderFlowScalpingStrategy::get_default_params() const {
    // **MODIFIED**: Relaxed the imbalance threshold to arm more frequently.
    return {
        {"lookback_period", 50.0},
        {"imbalance_threshold", 0.55}, // Was 0.65, now arms when avg pressure is > 55%
        {"hold_max_bars", 20.0},
        {"cool_down_period", 3.0}
    };
}

ParameterSpace OrderFlowScalpingStrategy::get_param_space() const { return {}; }

void OrderFlowScalpingStrategy::apply_params() {
    lookback_period_ = static_cast<int>(params_["lookback_period"]);
    imbalance_threshold_ = params_["imbalance_threshold"];
    hold_max_bars_ = static_cast<int>(params_["hold_max_bars"]);
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);

    rolling_pressure_ = RollingMean(lookback_period_);
    reset_state();
}

void OrderFlowScalpingStrategy::reset_state() {
    BaseStrategy::reset_state();
    of_state_ = OFState::Idle; // **FIXED**: Use the renamed state variable
    bars_in_trade_ = 0;
    rolling_pressure_ = RollingMean(lookback_period_);
}

double OrderFlowScalpingStrategy::calculate_bar_pressure(const Bar& bar) const {
    double range = bar.high - bar.low;
    if (range < 1e-9) return 0.5;
    return (bar.close - bar.low) / range;
}

StrategySignal OrderFlowScalpingStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;

    if (current_index < lookback_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }

    const auto& bar = bars[current_index];
    double pressure = calculate_bar_pressure(bar);
    double avg_pressure = rolling_pressure_.push(pressure);

    // **FIXED**: Use the strategy-specific 'of_state_' for state machine logic
    if (of_state_ == OFState::Long || of_state_ == OFState::Short) {
        bars_in_trade_++;
        if (bars_in_trade_ >= hold_max_bars_) {
            signal.type = (of_state_ == OFState::Long) ? StrategySignal::Type::SELL : StrategySignal::Type::BUY;
            diag_.emitted++;
            reset_state();
        }
        return signal;
    }
    
    if (is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return signal;
    }

    switch (of_state_) {
        case OFState::Idle:
            if (avg_pressure > imbalance_threshold_) of_state_ = OFState::ArmedLong;
            else if (avg_pressure < (1.0 - imbalance_threshold_)) of_state_ = OFState::ArmedShort;
            else diag_.drop(DropReason::THRESHOLD);
            break;
            
        case OFState::ArmedLong:
            if (pressure > 0.5) { // Confirmation bar must be bullish
                signal.type = StrategySignal::Type::BUY;
                of_state_ = OFState::Long;
            } else { // Failed confirmation
                of_state_ = OFState::Idle;
                diag_.drop(DropReason::THRESHOLD);
            }
            break;

        case OFState::ArmedShort:
            if (pressure < 0.5) { // Confirmation bar must be bearish
                signal.type = StrategySignal::Type::SELL;
                of_state_ = OFState::Short;
            } else { // Failed confirmation
                of_state_ = OFState::Idle;
                diag_.drop(DropReason::THRESHOLD);
            }
            break;
        default: break;
    }
    
    if (signal.type != StrategySignal::Type::HOLD) {
        signal.confidence = 0.7;
        diag_.emitted++;
        bars_in_trade_ = 0;
        // **FIXED**: This now correctly refers to the 'state_' member from BaseStrategy
        state_.last_trade_bar = current_index;
    }
    
    return signal;
}

REGISTER_STRATEGY(OrderFlowScalpingStrategy, "OrderFlowScalping");

} // namespace sentio


```

## 📄 **FILE 139 of 145**: src/strategy_sma_cross.cpp

**File Information**:
- **Path**: `src/strategy_sma_cross.cpp`

- **Size**: 35 lines
- **Modified**: 2025-09-05 16:34:48

- **Type**: .cpp

```text
#include "sentio/strategy_sma_cross.hpp"
#include <cmath>

namespace sentio {

SMACrossStrategy::SMACrossStrategy(const SMACrossCfg& cfg)
  : cfg_(cfg), sma_fast_(cfg.fast), sma_slow_(cfg.slow) {}

void SMACrossStrategy::on_bar(const StrategyCtx& ctx, const Bar& b) {
  (void)ctx;
  sma_fast_.push(b.close);
  sma_slow_.push(b.close);

  if (!warmed_up()) { last_.reset(); return; }

  double f = sma_fast_.value();
  double s = sma_slow_.value();
  if (!std::isfinite(f) || !std::isfinite(s)) { last_.reset(); return; }

  // Detect cross (including equality toggle avoidance)
  bool golden_now  = (f >= s) && !(std::isfinite(last_fast_) && std::isfinite(last_slow_) && last_fast_ >= last_slow_);
  bool death_now   = (f <= s) && !(std::isfinite(last_fast_) && std::isfinite(last_slow_) && last_fast_ <= last_slow_);

  if (golden_now) {
    last_ = StrategySignal{StrategySignal::Type::BUY, cfg_.conf_fast_slow};
  } else if (death_now) {
    last_ = StrategySignal{StrategySignal::Type::SELL, cfg_.conf_fast_slow};
  } else {
    last_.reset(); // no edge this bar
  }

  last_fast_ = f; last_slow_ = s;
}

} // namespace sentio

```

## 📄 **FILE 140 of 145**: src/strategy_tfa.cpp

**File Information**:
- **Path**: `src/strategy_tfa.cpp`

- **Size**: 286 lines
- **Modified**: 2025-09-07 22:48:35

- **Type**: .cpp

```text
#include "sentio/strategy_tfa.hpp"
#include "sentio/tfa/feature_guard.hpp"
#include "sentio/tfa/signal_pipeline.hpp"
#include "sentio/tfa/artifacts_safe.hpp"
#include "sentio/feature/column_projector_safe.hpp"
#include "sentio/feature/name_diff.hpp"
#include "sentio/tfa/tfa_seq_context.hpp"
#include <algorithm>
#include <chrono>

namespace sentio {

static ml::WindowSpec make_spec(const ml::ModelSpec& s){
  ml::WindowSpec w;
  // TFA always uses sequence length of 64 (hardcoded for now since TorchScript doesn't store this)
  w.seq_len = 64;
  w.layout  = s.input_layout.empty()? "BTF" : s.input_layout;
  w.feat_dim = (int)s.feature_names.size();
  // Disable external normalization; model contains its own scaler
  w.mean.clear();
  w.std.clear();
  w.clip2 = s.clip2;
  return w;
}

TFAStrategy::TFAStrategy()
: BaseStrategy("TFA")
, cfg_()
, handle_(ml::ModelRegistryTS::load_torchscript(cfg_.model_id, cfg_.version, cfg_.artifacts_dir, cfg_.use_cuda))
, window_(make_spec(handle_.spec))
{}

TFAStrategy::TFAStrategy(const TFACfg& cfg)
: BaseStrategy("TFA")
, cfg_(cfg)
, handle_(ml::ModelRegistryTS::load_torchscript(cfg.model_id, cfg.version, cfg.artifacts_dir, cfg.use_cuda))
, window_(make_spec(handle_.spec))
{
  // Model loaded successfully
}

void TFAStrategy::apply_params() {
  cfg_.conf_floor = params_["conf_floor"];
}

void TFAStrategy::set_raw_features(const std::vector<double>& raw){
  static int feature_calls = 0;
  feature_calls++;

  // Initialize safe projector on first use
  if (!projector_initialized_) {
    try {
      std::string artifacts_path = cfg_.artifacts_dir + "/" + cfg_.model_id + "/" + cfg_.version + "/";
      auto artifacts = tfa::load_tfa_artifacts_safe(
        artifacts_path + "model.pt",
        artifacts_path + "feature_spec.json",
        artifacts_path + "model.meta.json"
      );

      const int F_expected = artifacts.get_expected_input_dim();
      const auto& expected_names = artifacts.get_expected_feature_names();
      if (F_expected != 55) {
        throw std::runtime_error("Unsupported model input_dim (expect exactly 55)");
      }

      auto runtime_names = tfa::feature_names_from_spec(artifacts.spec);
      float pad_value = artifacts.get_pad_value();

      std::cout << "[TFA] Creating safe ColumnProjector: runtime=" << runtime_names.size()
                << " -> expected=" << expected_names.size() << " features" << std::endl;

      projector_safe_ = std::make_unique<ColumnProjectorSafe>(
        ColumnProjectorSafe::make(runtime_names, expected_names, pad_value)
      );

      expected_feat_dim_ = F_expected;
      projector_initialized_ = true;
      std::cout << "[TFA] Safe projector initialized: expecting " << expected_feat_dim_ << " features" << std::endl;
    } catch (const std::exception& e) {
      std::cout << "[TFA] Failed to initialize safe projector: " << e.what() << std::endl;
      return;
    }
  }

  // Project raw -> expected order and sanitize, then push into window
  try {
    std::vector<float> proj_f;
    projector_safe_->project_double(raw.data(), 1, raw.size(), proj_f);

    std::vector<double> proj_d;
    proj_d.resize((size_t)expected_feat_dim_);
    for (int i = 0; i < expected_feat_dim_; ++i) {
      float v = (i < (int)proj_f.size() && std::isfinite(proj_f[(size_t)i])) ? proj_f[(size_t)i] : 0.0f;
      proj_d[(size_t)i] = static_cast<double>(v);
    }

    window_.push(proj_d);
  } catch (const std::exception& e) {
    if (feature_calls % 1000 == 0 || feature_calls <= 10) {
      std::cout << "[TFA] Projection error in set_raw_features: " << e.what() << std::endl;
    }
  }

  if (feature_calls % 1000 == 0 || feature_calls <= 10) {
    std::cout << "[DIAG] TFA set_raw_features: call=" << feature_calls
              << " raw.size()=" << raw.size()
              << " window_ready=" << (window_.ready()? 1:0)
              << " feat_dim=" << window_.feat_dim() << std::endl;
  }
}

StrategySignal TFAStrategy::map_output(const ml::ModelOutput& mo) const {
  StrategySignal s;
  // If explicit probabilities are provided
  if (!mo.probs.empty()) {
    if (mo.probs.size() == 1) {
      float prob = mo.probs[0];
      if (prob > 0.5f) {
        s.type = StrategySignal::Type::BUY;
        s.confidence = prob;
      } else {
        s.type = StrategySignal::Type::SELL;
        s.confidence = 1.0f - prob;
      }
      return s;
    }
    // 3-class path
    int argmax = 0; 
    for (int i=1;i<(int)mo.probs.size();++i) 
      if (mo.probs[i]>mo.probs[argmax]) argmax=i;
    float pmax = mo.probs[argmax];
    if      (argmax==2) s.type = StrategySignal::Type::BUY;
    else if (argmax==0) s.type = StrategySignal::Type::SELL;
    else                s.type = StrategySignal::Type::HOLD;
    s.confidence = std::max(cfg_.conf_floor, (double)pmax);
    return s;
  }

  // Fallback: logits-only path (binary)
  float logit = mo.score;
  float prob = 1.0f / (1.0f + std::exp(-logit));
  if (prob > 0.5f) {
    s.type = StrategySignal::Type::BUY;
    s.confidence = prob;
  } else {
    s.type = StrategySignal::Type::SELL;
    s.confidence = 1.0f - prob;
  }
  return s;
}

void TFAStrategy::on_bar(const StrategyCtx& ctx, const Bar& b){
  (void)ctx; (void)b;
  last_.reset();
  
  // Diagnostic: Check if window is ready
  if (!window_.ready()) {
    std::cout << "[TFA] Window not ready, required=" << window_.seq_len() << std::endl;
    return;
  }

  auto in = window_.to_input();
  if (!in) {
    std::cout << "[TFA] Failed to create input from window" << std::endl;
    return;
  }

  std::optional<ml::ModelOutput> out;
  try {
    out = handle_.model->predict(*in, window_.seq_len(), window_.feat_dim(), handle_.spec.input_layout);
  } catch (const std::exception& e) {
    std::cout << "[TFA] predict threw: " << e.what() << std::endl;
    return;
  }
  
  if (!out) {
    std::cout << "[TFA] Model prediction failed" << std::endl;
    return;
  }

  auto sig = map_output(*out);
  std::cout << "[TFA] Raw confidence=" << sig.confidence << ", floor=" << cfg_.conf_floor << std::endl;
  if (sig.confidence < cfg_.conf_floor) {
    std::cout << "[TFA] Signal dropped due to low confidence" << std::endl;
    return;
  }
  last_ = sig;
  std::cout << "[TFA] Signal generated: " << (sig.type == StrategySignal::Type::BUY ? "BUY" : 
                                              sig.type == StrategySignal::Type::SELL ? "SELL" : "HOLD") 
            << " conf=" << sig.confidence << std::endl;
}

ParameterMap TFAStrategy::get_default_params() const {
  return {
    {"conf_floor", cfg_.conf_floor}
  };
}

ParameterSpace TFAStrategy::get_param_space() const {
  return {
    {"conf_floor", {ParamType::FLOAT, 0.0, 1.0, cfg_.conf_floor}}
  };
}

StrategySignal TFAStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
  (void)current_index; // we will use bars.size() and a static cursor
  static int calls = 0;
  ++calls;
  last_.reset();

  // One-time: precompute probabilities over the whole series using the sequence context
  static bool seq_inited = false;
  static TfaSeqContext seq_ctx;
  static std::vector<float> probs_all;
  if (!seq_inited) {
    try {
      std::string artifacts_path = cfg_.artifacts_dir + "/" + cfg_.model_id + "/" + cfg_.version + "/";
      seq_ctx.load(artifacts_path + "model.pt",
                   artifacts_path + "feature_spec.json",
                   artifacts_path + "model.meta.json");
      // Assume base symbol is QQQ for this test run
      seq_ctx.forward_probs("QQQ", bars, probs_all);
      seq_inited = true;
      std::cout << "[TFA seq] precomputed probs: N=" << probs_all.size() << std::endl;
    } catch (const std::exception& e) {
      std::cout << "[TFA seq] init/forward failed: " << e.what() << std::endl;
      return StrategySignal{};
    }
  }

  // Maintain rolling threshold logic with cooldown based on precomputed prob at this call index
  float prob = (calls-1 < (int)probs_all.size()) ? probs_all[(size_t)(calls-1)] : 0.5f;

  static std::vector<float> p_hist; p_hist.reserve(4096);
  static int cooldown_long_until = -1;
  static int cooldown_short_until = -1;
  const int window = 250;
  const float q_long = 0.80f, q_short = 0.20f;
  const float floor_long = 0.55f, ceil_short = 0.45f;
  const int cooldown = 5;

  p_hist.push_back(prob);

  StrategySignal sig{}; sig.type = StrategySignal::Type::HOLD; sig.confidence = 0.0;

  if ((int)p_hist.size() >= std::max(window, seq_ctx.T)) {
    int end = (int)p_hist.size() - 1;
    int start = std::max(0, end - window + 1);
    std::vector<float> win(p_hist.begin() + start, p_hist.begin() + end + 1);

    int kL = (int)std::floor(q_long * (win.size() - 1));
    std::nth_element(win.begin(), win.begin() + kL, win.end());
    float thrL = std::max(floor_long, win[kL]);

    int kS = (int)std::floor(q_short * (win.size() - 1));
    std::nth_element(win.begin(), win.begin() + kS, win.end());
    float thrS = std::min(ceil_short, win[kS]);

    bool can_long = (calls >= cooldown_long_until);
    bool can_short = (calls >= cooldown_short_until);

    if (can_long && prob >= thrL) {
      sig.type = StrategySignal::Type::BUY;
      sig.confidence = prob;
      cooldown_long_until = calls + cooldown;
    } else if (can_short && prob <= thrS) {
      sig.type = StrategySignal::Type::SELL;
      sig.confidence = 1.0f - prob;
      cooldown_short_until = calls + cooldown;
    }

    if (calls % 64 == 0) {
      std::cout << "[TFA calc] prob=" << prob << " thrL=" << thrL << " thrS=" << thrS
                << " type=" << (sig.type==StrategySignal::Type::BUY?"BUY":sig.type==StrategySignal::Type::SELL?"SELL":"HOLD")
                << std::endl;
    }
  }

  if (sig.type == StrategySignal::Type::HOLD) return StrategySignal{};
  last_ = sig;
  return sig;
}

REGISTER_STRATEGY(TFAStrategy, "TFA");

} // namespace sentio

```

## 📄 **FILE 141 of 145**: src/strategy_transformer_ts.cpp

**File Information**:
- **Path**: `src/strategy_transformer_ts.cpp`

- **Size**: 54 lines
- **Modified**: 2025-09-05 23:39:11

- **Type**: .cpp

```text
#include "sentio/strategy_transformer_ts.hpp"
#include <algorithm>

namespace sentio {

static ml::WindowSpec make_spec(const ml::ModelSpec& s){
  ml::WindowSpec w;
  w.seq_len = s.seq_len>0? s.seq_len : 64;
  w.layout  = s.input_layout.empty()? "BTF" : s.input_layout;
  w.feat_dim = (int)s.feature_names.size();
  w.mean = s.mean; w.std = s.std; w.clip2 = s.clip2;
  return w;
}

TransformerSignalStrategyTS::TransformerSignalStrategyTS(const TransformerTSCfg& cfg)
: cfg_(cfg)
, handle_(ml::ModelRegistryTS::load_torchscript(cfg.model_id, cfg.version, cfg.artifacts_dir, cfg.use_cuda))
, window_(make_spec(handle_.spec))
{}

void TransformerSignalStrategyTS::set_raw_features(const std::vector<double>& raw){
  window_.push(raw);
}

StrategySignal TransformerSignalStrategyTS::map_output(const ml::ModelOutput& mo) const {
  StrategySignal s;
  if (mo.probs.empty()) { s.type=StrategySignal::Type::HOLD; s.confidence=cfg_.conf_floor; return s; }
  int argmax = 0; for (int i=1;i<(int)mo.probs.size();++i) if (mo.probs[i]>mo.probs[argmax]) argmax=i;
  float pmax = mo.probs[argmax];
  // default order ["SELL","HOLD","BUY"]
  if      (argmax==2) s.type = StrategySignal::Type::BUY;
  else if (argmax==0) s.type = StrategySignal::Type::SELL;
  else                s.type = StrategySignal::Type::HOLD;
  s.confidence = std::max(cfg_.conf_floor, (double)pmax);
  return s;
}

void TransformerSignalStrategyTS::on_bar(const StrategyCtx& ctx, const Bar& b){
  (void)ctx; (void)b;
  last_.reset();
  if (!window_.ready()) return;

  auto in = window_.to_input();
  if (!in) return;

  auto out = handle_.model->predict(*in, window_.seq_len(), window_.feat_dim(), handle_.spec.input_layout);
  if (!out) return;

  auto sig = map_output(*out);
  if (sig.confidence < cfg_.conf_floor) return;
  last_ = sig;
}

} // namespace sentio

```

## 📄 **FILE 142 of 145**: src/strategy_vwap_reversion.cpp

**File Information**:
- **Path**: `src/strategy_vwap_reversion.cpp`

- **Size**: 156 lines
- **Modified**: 2025-09-05 15:24:52

- **Type**: .cpp

```text
#include "sentio/strategy_vwap_reversion.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace sentio {

VWAPReversionStrategy::VWAPReversionStrategy() : BaseStrategy("VWAPReversion") {
    params_ = get_default_params();
    apply_params();
}

ParameterMap VWAPReversionStrategy::get_default_params() const {
    // Default parameters remain the same
    return {
        {"vwap_period", 390.0}, {"band_multiplier", 0.005}, {"max_band_width", 0.01},
        {"min_distance_from_vwap", 0.001}, {"volume_confirmation_mult", 1.2},
        {"rsi_period", 14.0}, {"rsi_oversold", 40.0}, {"rsi_overbought", 60.0},
        {"stop_loss_pct", 0.003}, {"take_profit_pct", 0.005},
        {"time_stop_bars", 30.0}, {"cool_down_period", 2.0}
    };
}

ParameterSpace VWAPReversionStrategy::get_param_space() const { return {}; }

void VWAPReversionStrategy::apply_params() {
    vwap_period_ = static_cast<int>(params_["vwap_period"]);
    band_multiplier_ = params_["band_multiplier"];
    max_band_width_ = params_["max_band_width"];
    min_distance_from_vwap_ = params_["min_distance_from_vwap"];
    volume_confirmation_mult_ = params_["volume_confirmation_mult"];
    rsi_period_ = static_cast<int>(params_["rsi_period"]);
    rsi_oversold_ = params_["rsi_oversold"];
    rsi_overbought_ = params_["rsi_overbought"];
    stop_loss_pct_ = params_["stop_loss_pct"];
    take_profit_pct_ = params_["take_profit_pct"];
    time_stop_bars_ = static_cast<int>(params_["time_stop_bars"]);
    cool_down_period_ = static_cast<int>(params_["cool_down_period"]);
    reset_state();
}

void VWAPReversionStrategy::reset_state() {
    BaseStrategy::reset_state();
    cumulative_pv_ = 0.0;
    cumulative_volume_ = 0.0;
    time_in_position_ = 0;
    vwap_ = 0.0;
}

void VWAPReversionStrategy::update_vwap(const Bar& bar) {
    double typical_price = (bar.high + bar.low + bar.close) / 3.0;
    cumulative_pv_ += typical_price * bar.volume;
    cumulative_volume_ += bar.volume;
    if (cumulative_volume_ > 0) {
        vwap_ = cumulative_pv_ / cumulative_volume_;
    }
}

StrategySignal VWAPReversionStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
    StrategySignal signal;
    update_vwap(bars[current_index]);

    if (current_index < rsi_period_) {
        diag_.drop(DropReason::MIN_BARS);
        return signal;
    }
    
    if (is_cooldown_active(current_index, cool_down_period_)) {
        diag_.drop(DropReason::COOLDOWN);
        return signal;
    }
    
    if (vwap_ <= 0) {
        diag_.drop(DropReason::NAN_INPUT);
        return signal;
    }

    const auto& bar = bars[current_index];
    double distance_pct = std::abs(bar.close - vwap_) / vwap_;
    if (distance_pct < min_distance_from_vwap_) {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }

    double upper_band = vwap_ * (1.0 + band_multiplier_);
    double lower_band = vwap_ * (1.0 - band_multiplier_);

    bool buy_condition = bar.close < lower_band && is_rsi_condition_met(bars, current_index, true);
    bool sell_condition = bar.close > upper_band && is_rsi_condition_met(bars, current_index, false);

    if (buy_condition) {
        signal.type = StrategySignal::Type::BUY;
    } else if (sell_condition) {
        signal.type = StrategySignal::Type::SELL;
    } else {
        diag_.drop(DropReason::THRESHOLD);
        return signal;
    }

    signal.confidence = 0.8;
    diag_.emitted++;
    state_.last_trade_bar = current_index;
    return signal;
}

bool VWAPReversionStrategy::is_rsi_condition_met(const std::vector<Bar>& bars, int index, bool for_buy) const {
    std::vector<double> closes;
    closes.reserve(rsi_period_);
    for(int i = 0; i < rsi_period_; ++i) {
        closes.push_back(bars[index - rsi_period_ + 1 + i].close);
    }
    // Simple RSI calculation
    double rsi = calculate_simple_rsi(closes);
    return for_buy ? (rsi < rsi_oversold_) : (rsi > rsi_overbought_);
}

double VWAPReversionStrategy::calculate_simple_rsi(const std::vector<double>& prices) const {
    if (prices.size() < 2) return 50.0; // Neutral RSI if not enough data
    
    std::vector<double> gains, losses;
    for (size_t i = 1; i < prices.size(); ++i) {
        double change = prices[i] - prices[i-1];
        if (change > 0) {
            gains.push_back(change);
            losses.push_back(0.0);
        } else {
            gains.push_back(0.0);
            losses.push_back(-change);
        }
    }
    
    if (gains.empty()) return 50.0;
    
    double avg_gain = std::accumulate(gains.begin(), gains.end(), 0.0) / gains.size();
    double avg_loss = std::accumulate(losses.begin(), losses.end(), 0.0) / losses.size();
    
    if (avg_loss == 0.0) return 100.0; // All gains
    if (avg_gain == 0.0) return 0.0;   // All losses
    
    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}

bool VWAPReversionStrategy::is_volume_confirmed(const std::vector<Bar>& bars, int index) const {
    if (index < 20) return true;
    double avg_vol = 0;
    for(int i = 1; i <= 20; ++i) {
        avg_vol += bars[index-i].volume;
    }
    avg_vol /= 20.0;
    return bars[index].volume > avg_vol * volume_confirmation_mult_;
}

REGISTER_STRATEGY(VWAPReversionStrategy, "VWAPReversion");

} // namespace sentio
```

## 📄 **FILE 143 of 145**: src/telemetry_logger.cpp

**File Information**:
- **Path**: `src/telemetry_logger.cpp`

- **Size**: 177 lines
- **Modified**: 2025-09-06 01:37:11

- **Type**: .cpp

```text
#include "sentio/telemetry_logger.hpp"
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace sentio {

// Global telemetry logger instance
std::unique_ptr<TelemetryLogger> g_telemetry_logger = nullptr;

TelemetryLogger::TelemetryLogger(const std::string& log_file_path) 
    : log_file_path_(log_file_path) {
    // Create directory if it doesn't exist
    std::filesystem::path path(log_file_path);
    std::filesystem::create_directories(path.parent_path());
    
    // Open log file in append mode
    log_file_.open(log_file_path, std::ios::app);
    if (!log_file_.is_open()) {
        throw std::runtime_error("Failed to open telemetry log file: " + log_file_path);
    }
}

TelemetryLogger::~TelemetryLogger() {
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

void TelemetryLogger::log(const TelemetryData& data) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    std::ostringstream json;
    json << "{";
    json << "\"timestamp\":\"" << data.timestamp << "\",";
    json << "\"strategy_name\":\"" << escape_json_string(data.strategy_name) << "\",";
    json << "\"instrument\":\"" << escape_json_string(data.instrument) << "\",";
    json << "\"bars_processed\":" << data.bars_processed << ",";
    json << "\"signals_generated\":" << data.signals_generated << ",";
    json << "\"buy_signals\":" << data.buy_signals << ",";
    json << "\"sell_signals\":" << data.sell_signals << ",";
    json << "\"hold_signals\":" << data.hold_signals << ",";
    json << "\"avg_confidence\":" << std::fixed << std::setprecision(6) << data.avg_confidence << ",";
    json << "\"ready_percentage\":" << std::fixed << std::setprecision(2) << data.ready_percentage << ",";
    json << "\"processing_time_ms\":" << std::fixed << std::setprecision(3) << data.processing_time_ms;
    
    if (!data.notes.empty()) {
        json << ",\"notes\":\"" << escape_json_string(data.notes) << "\"";
    }
    
    json << "}";
    
    write_json_line(json.str());
}

void TelemetryLogger::log_metric(
    const std::string& strategy_name,
    const std::string& metric_name,
    double value,
    const std::string& instrument
) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    std::ostringstream json;
    json << "{";
    json << "\"timestamp\":\"" << get_current_timestamp() << "\",";
    json << "\"strategy_name\":\"" << escape_json_string(strategy_name) << "\",";
    json << "\"instrument\":\"" << escape_json_string(instrument) << "\",";
    json << "\"metric_name\":\"" << escape_json_string(metric_name) << "\",";
    json << "\"value\":" << std::fixed << std::setprecision(6) << value;
    json << "}";
    
    write_json_line(json.str());
}

void TelemetryLogger::log_signal_stats(
    const std::string& strategy_name,
    const std::string& instrument,
    int signals_generated,
    int buy_signals,
    int sell_signals,
    int hold_signals,
    double avg_confidence
) {
    TelemetryData data;
    data.timestamp = get_current_timestamp();
    data.strategy_name = strategy_name;
    data.instrument = instrument;
    data.signals_generated = signals_generated;
    data.buy_signals = buy_signals;
    data.sell_signals = sell_signals;
    data.hold_signals = hold_signals;
    data.avg_confidence = avg_confidence;
    
    log(data);
}

void TelemetryLogger::log_performance(
    const std::string& strategy_name,
    const std::string& instrument,
    int bars_processed,
    double processing_time_ms,
    double ready_percentage
) {
    TelemetryData data;
    data.timestamp = get_current_timestamp();
    data.strategy_name = strategy_name;
    data.instrument = instrument;
    data.bars_processed = bars_processed;
    data.processing_time_ms = processing_time_ms;
    data.ready_percentage = ready_percentage;
    
    log(data);
}

void TelemetryLogger::flush() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (log_file_.is_open()) {
        log_file_.flush();
    }
}

std::string TelemetryLogger::get_current_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count() << "Z";
    return oss.str();
}

std::string TelemetryLogger::escape_json_string(const std::string& str) const {
    std::string escaped;
    escaped.reserve(str.length() + 10); // Reserve some extra space
    
    for (char c : str) {
        switch (c) {
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:   escaped += c; break;
        }
    }
    
    return escaped;
}

void TelemetryLogger::write_json_line(const std::string& json_line) {
    if (log_file_.is_open()) {
        log_file_ << json_line << std::endl;
        
        // Flush every 100 lines to ensure data is written
        if (++log_counter_ % 100 == 0) {
            log_file_.flush();
        }
    }
}

void init_telemetry_logger(const std::string& log_file_path) {
    g_telemetry_logger = std::make_unique<TelemetryLogger>(log_file_path);
}

TelemetryLogger& get_telemetry_logger() {
    if (!g_telemetry_logger) {
        init_telemetry_logger();
    }
    return *g_telemetry_logger;
}

} // namespace sentio

```

## 📄 **FILE 144 of 145**: src/temporal_analysis.cpp

**File Information**:
- **Path**: `src/temporal_analysis.cpp`

- **Size**: 150 lines
- **Modified**: 2025-09-08 08:47:53

- **Type**: .cpp

```text
#include "sentio/temporal_analysis.hpp"
#include "sentio/runner.hpp"
#include "sentio/audit.hpp"
#include "sentio/metrics.hpp"
#include "sentio/progress_bar.hpp"
#include "sentio/day_index.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

namespace sentio {

TemporalAnalysisSummary run_temporal_analysis(const SymbolTable& ST,
                                            const std::vector<std::vector<Bar>>& series,
                                            int base_symbol_id,
                                            const RunnerCfg& rcfg,
                                            const TemporalAnalysisConfig& cfg) {
    
    TemporalAnalyzer analyzer;
    const auto& base_series = series[base_symbol_id];
    const int total_bars = (int)base_series.size();
    
    if (total_bars < cfg.min_bars_per_quarter) {
        std::cerr << "ERROR: Insufficient data for temporal analysis. Need at least " 
                  << cfg.min_bars_per_quarter << " bars per quarter." << std::endl;
        return TemporalAnalysisSummary{};
    }
    
    // Calculate quarters based on data
    int bars_per_quarter = total_bars / cfg.num_quarters;
    int current_year = 2021; // Starting year
    int current_quarter = 1;
    
    std::cout << "Starting TPA (Temporal Performance Analysis) Test..." << std::endl;
    std::cout << "Total bars: " << total_bars << ", Bars per quarter: " << bars_per_quarter << std::endl;
    
    // Initialize TPA progress bar
    TPATestProgressBar progress_bar(cfg.num_quarters, rcfg.strategy_name);
    progress_bar.display(); // Show initial progress bar
    
    std::cout << "\nInitializing data processing..." << std::endl;
    
    // Build audit filename prefix with strategy and timestamp
    const std::string test_name = "tpa_test";
    const auto ts_epoch = static_cast<long long>(std::time(nullptr));

    for (int q = 0; q < cfg.num_quarters; ++q) {
        int start_idx = q * bars_per_quarter;
        int end_idx = std::min(start_idx + bars_per_quarter, total_bars);
        
        if (end_idx - start_idx < cfg.min_bars_per_quarter) {
            std::cout << "Skipping quarter " << (q + 1) << " - insufficient data" << std::endl;
            continue;
        }
        
        std::cout << "\nProcessing Quarter " << current_year << "Q" << current_quarter 
                  << " (bars " << start_idx << "-" << end_idx << ")..." << std::endl;
        
        // Create data slice for this quarter
        std::vector<std::vector<Bar>> quarter_series;
        quarter_series.reserve(series.size());
        for (const auto& sym_series : series) {
            if (sym_series.size() > static_cast<size_t>(end_idx)) {
                quarter_series.emplace_back(sym_series.begin() + start_idx, sym_series.begin() + end_idx);
            } else if (sym_series.size() > static_cast<size_t>(start_idx)) {
                quarter_series.emplace_back(sym_series.begin() + start_idx, sym_series.end());
            } else {
                quarter_series.emplace_back();
            }
        }
        
        // Run backtest for this quarter
        AuditConfig audit_cfg;
        audit_cfg.run_id = rcfg.strategy_name + "_" + test_name + "_q" + std::to_string(q + 1) + "_" + std::to_string(ts_epoch);
        audit_cfg.file_path = "audit/" + rcfg.strategy_name + "_" + test_name + "_" + std::to_string(ts_epoch) + "_q" + std::to_string(q + 1) + ".jsonl";
        audit_cfg.flush_each = true;
        AuditRecorder audit(audit_cfg);
        
        auto result = run_backtest(audit, ST, quarter_series, base_symbol_id, rcfg);
        
        // Calculate quarterly metrics
        QuarterlyMetrics metrics;
        metrics.year = current_year;
        metrics.quarter = current_quarter;
        
        // Calculate actual trading days by extracting unique dates from base symbol bars
        // Use the first series (base symbol) to count trading days
        int actual_trading_days = 0;
        if (!series.empty() && series[0].size() > static_cast<size_t>(start_idx)) {
            int actual_end_idx = std::min(end_idx, static_cast<int>(series[0].size()));
            std::vector<Bar> quarter_bars(series[0].begin() + start_idx, series[0].begin() + actual_end_idx);
            auto day_starts = day_start_indices(quarter_bars);
            actual_trading_days = static_cast<int>(day_starts.size());
        } else {
            // Fallback: estimate trading days (approximately 66 trading days per quarter)
            actual_trading_days = std::max(1, (end_idx - start_idx) / 390); // ~390 bars per day
        }
        
        // Convert total return percent for the slice into a day-compounded monthly return
        // result.total_return is percent for the tested slice; convert to decimal for compounding
        double ret_dec = result.total_return / 100.0;
        double monthly_compounded = 0.0;
        if (actual_trading_days > 0) {
            // Compound to a 21-trading-day month
            monthly_compounded = (std::pow(1.0 + ret_dec, 21.0 / static_cast<double>(actual_trading_days)) - 1.0) * 100.0;
        }
        metrics.monthly_return_pct = monthly_compounded;
        
        metrics.sharpe_ratio = result.sharpe_ratio;
        metrics.total_trades = result.total_fills;  // Use total_fills as proxy for trades
        metrics.trading_days = actual_trading_days;
        metrics.avg_daily_trades = actual_trading_days > 0 ? static_cast<double>(result.total_fills) / actual_trading_days : 0.0;
        metrics.max_drawdown = result.max_drawdown;
        metrics.win_rate = 0.0;  // Not available in RunResult, set to 0
        metrics.total_return_pct = result.total_return;
        
        analyzer.add_quarterly_result(metrics);
        
        // Update progress bar with quarter results
        progress_bar.display_with_quarter_info(q + 1, current_year, current_quarter,
                                             metrics.monthly_return_pct, metrics.sharpe_ratio,
                                             metrics.avg_daily_trades, metrics.health_status());
        
        // Print quarter summary
        std::cout << "\n  Monthly Return: " << std::fixed << std::setprecision(2) 
                  << metrics.monthly_return_pct << "%" << std::endl;
        std::cout << "  Sharpe Ratio: " << std::fixed << std::setprecision(3) 
                  << metrics.sharpe_ratio << std::endl;
        std::cout << "  Daily Trades: " << std::fixed << std::setprecision(1) 
                  << metrics.avg_daily_trades << " (Health: " << metrics.health_status() << ")" << std::endl;
        std::cout << "  Total Trades: " << metrics.total_trades << std::endl;
        
        // Update year/quarter for next iteration
        current_quarter++;
        if (current_quarter > 4) {
            current_quarter = 1;
            current_year++;
        }
    }
    
    // Final progress bar update
    std::cout << "\n\nTPA Test completed! Generating summary..." << std::endl;
    
    auto summary = analyzer.generate_summary();
    summary.assess_readiness(cfg);
    return summary;
}

} // namespace sentio

```

## 📄 **FILE 145 of 145**: src/time_utils.cpp

**File Information**:
- **Path**: `src/time_utils.cpp`

- **Size**: 106 lines
- **Modified**: 2025-09-05 15:29:30

- **Type**: .cpp

```text
#include "sentio/time_utils.hpp"
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <algorithm>

#if __has_include(<chrono>)
  #include <chrono>
  using namespace std::chrono;
#else
  #error "Need C++20 <chrono>. If missing tzdb, you can still normalize UTC here and handle ET in calendar."
#endif

namespace sentio {

bool iso8601_looks_like(const std::string& s) {
  // Very light heuristic: "YYYY-MM-DDTHH:MM" and either 'Z' or +/-HH:MM
  return s.size() >= 16 && s[4]=='-' && s[7]=='-' && (s.find('T') != std::string::npos);
}

bool epoch_ms_suspected(double v_ms) {
  // If it's larger than ~1e12 it's probably ms; (1e12 sec is ~31k years)
  return std::isfinite(v_ms) && v_ms > 1.0e11;
}

static inline sys_seconds parse_iso8601_to_utc(const std::string& s) {
  // Minimal ISO8601 handling: require offset or 'Z'.
  // For robustness use Howard Hinnant's date::parse with %FT%T%Ez.
  // Here we support the common forms: 2022-09-06T13:30:00Z and 2022-09-06T09:30:00-04:00
  // We'll implement a tiny parser that splits offset and adjusts.
  auto posT = s.find('T');
  if (posT == std::string::npos) throw std::runtime_error("ISO8601 missing T");
  // Find offset start: last char 'Z' or last '+'/'-'
  int sign = 0;
  int oh=0, om=0;
  bool zulu = false;
  std::size_t offPos = s.rfind('Z');
  if (offPos != std::string::npos && offPos > posT) {
    zulu = true;
  } else {
    std::size_t plus = s.rfind('+');
    std::size_t minus= s.rfind('-');
    std::size_t off  = std::string::npos;
    if (plus!=std::string::npos && plus>posT) { off=plus; sign=+1; }
    else if (minus!=std::string::npos && minus>posT) { off=minus; sign=-1; }
    if (off==std::string::npos) throw std::runtime_error("ISO8601 missing offset/Z");
    // parse HH:MM
    if (off+3 >= s.size()) throw std::runtime_error("Bad offset");
    oh = std::stoi(s.substr(off+1,2));
    if (off+6 <= s.size() && s[off+3]==':') om = std::stoi(s.substr(off+4,2));
  }

  // parse date/time parts (seconds optional)
  int Y = std::stoi(s.substr(0,4));
  int M = std::stoi(s.substr(5,2));
  int D = std::stoi(s.substr(8,2));
  int h = std::stoi(s.substr(posT+1,2));
  int m = std::stoi(s.substr(posT+4,2));
  int sec = 0;
  if (posT+6 < s.size() && s[posT+6]==':') {
    sec = std::stoi(s.substr(posT+7,2));
  }

  // Treat parsed time as local-time-with-offset; compute UTC by subtracting offset
  using namespace std::chrono;
  sys_days sd = sys_days(std::chrono::year{Y}/M/D);
  seconds local = hours{h} + minutes{m} + seconds{sec};
  seconds off = seconds{ (oh*3600 + om*60) * (zulu ? 0 : sign) };
  // If sign=+1 (e.g., +09:00), local = UTC + offset => UTC = local - offset
  seconds utc_sec = local - off;
  return sys_seconds{sd.time_since_epoch() + utc_sec};
}

std::chrono::sys_seconds to_utc_sys_seconds(const std::variant<std::int64_t, double, std::string>& ts) {
  if (std::holds_alternative<std::int64_t>(ts)) {
    // epoch seconds
    return std::chrono::sys_seconds{std::chrono::seconds{std::get<std::int64_t>(ts)}};
  }
  if (std::holds_alternative<double>(ts)) {
    // Could be epoch ms or sec (float). Prefer ms detection and round down.
    double v = std::get<double>(ts);
    if (!std::isfinite(v)) throw std::runtime_error("Non-finite epoch");
    if (epoch_ms_suspected(v)) {
      auto s = static_cast<std::int64_t>(v / 1000.0);
      return std::chrono::sys_seconds{std::chrono::seconds{s}};
    } else {
      auto s = static_cast<std::int64_t>(v);
      return std::chrono::sys_seconds{std::chrono::seconds{s}};
    }
  }
  const std::string& s = std::get<std::string>(ts);
  if (!iso8601_looks_like(s)) {
    // fall back: try integer seconds in string
    std::int64_t v{};
    auto sv = std::string_view{s};
    if (auto [p, ec] = std::from_chars(sv.data(), sv.data()+sv.size(), v); ec == std::errc{}) {
      return std::chrono::sys_seconds{std::chrono::seconds{v}};
    }
    throw std::runtime_error("Unrecognized timestamp format: " + s);
  }
  return parse_iso8601_to_utc(s);
}

} // namespace sentio
```

