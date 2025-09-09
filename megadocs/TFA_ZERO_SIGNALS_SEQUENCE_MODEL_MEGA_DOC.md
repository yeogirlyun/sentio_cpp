## TFA Zero-Signals Bug (Sequence Model) — Mega Doc

### Summary
- After migrating TFA to a 64×55 Transformer sequence path with strict schema (55 features, no ts) and using cached features, the TPA test emits zero signals/trades across all quarters.
- Features are loaded correctly (55/55), pushed into `FeatureWindow`, and the model artifacts are found and validated.
- Despite this, no BUY/SELL signals are produced in the TPA loop.

### Impact
- TFA strategy currently produces no trades in TPA, blocking validation of monthly returns, Sharpe, and audit replay.

### Environment
- Host: macOS (darwin 24.6.0), Apple Silicon
- Repo: sentio_cpp
- Artifacts: `artifacts/TFA/v1/model.pt`, `feature_spec.json`, `model.meta.json`
- Feature cache: `data/QQQ_RTH_features.csv` (55 features; validated)

### Reproduction Steps
1. Ensure feature cache exists:
   - `data/QQQ_RTH_features.csv` (55 features; header `ts,<55 names>`)
2. Ensure TFA artifacts exist under `artifacts/TFA/v1/`:
   - `model.pt`, `feature_spec.json`, `model.meta.json` with `input_dim=55`
3. Build and run:
   - `make -j4 build/sentio_cli`
   - `./build/sentio_cli tpa_test QQQ --strategy tfa --days 1`

### Observed Output (excerpts)
```text
FeatureCache: Loaded 55 feature names
FeatureCache: Successfully loaded 1564 bars with 55 features each
FeatureCache: Recommended starting bar: 300
FeatureFeeder: Cached features ENABLED
✅ Cached features loaded successfully - MASSIVE speed improvement enabled!
...
[TFA] Model expects 55 features, emit_from=64
[TFA] Creating safe ColumnProjector: runtime=55 -> expected=55 features
[ColumnProjectorSafe] Created: 55 src → 55 dst (mapped=55, filled=0)
[TFA] Safe projector initialized: expecting 55 features
[DIAG] FeatureWindow push SUCCESS: call=1 buf_size=1/64
...
[SIG TFA] emitted=0 dropped=0  min_bars=0 session=0 nan=0 zerovol=0 thr=0 cooldown=0 dup=0
TPA Test: tfa [...] | Trades: 0.0 | LOW_FREQ
```

Notes:
- Feature cache path confirms correct 55-dim features.
- Projector built (no fill-ins), FeatureWindow is filling to 64 frames.
- No signals reported; trades remain zero.

### Expected Behavior
- Once `FeatureWindow` reaches sequence length (64) and per-bar inference runs, the model should output logits/probabilities; confidence threshold is default low (`conf_floor=0.05`), so BUY/SELL signals should be emitted on at least some bars.

### Diagnostics Added
- TFA now pushes projected frames into `FeatureWindow` in `set_raw_features`.
- `calculate_signal` uses `window_.to_input()` -> TorchScript predict -> `map_output`.
- Logging added for projector init, window readiness, predict, and inferred confidence/type.

### Relevant Code Excerpts

```140:182:src/strategy_tfa.cpp
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

  auto out = handle_.model->predict(*in, window_.seq_len(), window_.feat_dim(), handle_.spec.input_layout);
  
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
```

```182:246:src/strategy_tfa.cpp
StrategySignal TFAStrategy::calculate_signal(const std::vector<Bar>& bars, int current_index) {
  (void)bars; (void)current_index; // Features come from set_raw_features projected into window_
  static int calls = 0;
  ++calls;
  if (calls <= 5 || calls % 64 == 0) {
    std::cout << "[TFA calc] enter calls=" << calls << " ready=" << (window_.ready()?1:0) << std::endl;
  }
  last_.reset();

  if (!window_.ready()) {
    if (calls <= 5 || calls % 64 == 0) {
      std::cout << "[TFA calc] not ready" << std::endl;
    }
    return StrategySignal{};
  }

  auto in = window_.to_input();
  if (!in) {
    if (calls <= 5 || calls % 64 == 0) {
      std::cout << "[TFA calc] to_input failed" << std::endl;
    }
    return StrategySignal{};
  }

  auto out = handle_.model->predict(*in, window_.seq_len(), window_.feat_dim(), handle_.spec.input_layout);
  if (!out) {
    if (calls <= 5 || calls % 64 == 0) {
      std::cout << "[TFA calc] predict failed" << std::endl;
    }
    return StrategySignal{};
  }

  if (calls <= 5 || calls % 64 == 0) {
    std::cout << "[TFA calc] probs_sz=" << out->probs.size() << " score=" << out->score << std::endl;
  }

  auto sig = map_output(*out);
  if (calls % 64 == 0) {
    std::cout << "[TFA calc] conf=" << sig.confidence << " type="
              << (sig.type == StrategySignal::Type::BUY ? "BUY" : sig.type == StrategySignal::Type::SELL ? "SELL" : "HOLD")
              << std::endl;
  }
  if (sig.confidence < cfg_.conf_floor) {
    return StrategySignal{};
  }
  last_ = sig;
  return sig;
}
```

```66:92:src/runner.cpp
for (size_t i = warmup_bars; i < base_series.size(); ++i) {
  if (i % progress_interval == 0) {
    int progress_percent = (i * 100) / total_bars;
    std::cout << "Progress: " << progress_percent << "% (" << i << "/" << total_bars << " bars)" << std::endl;
  }
  const auto& bar = base_series[i];
  pricebook.sync_to_base_i(i);
  AuditBar audit_bar{bar.open, bar.high, bar.low, bar.close, static_cast<double>(bar.volume)};
  audit.event_bar(bar.ts_nyt_epoch, ST.get_symbol(base_symbol_id), audit_bar);
  FeatureFeeder::feed_features_to_strategy(strategy.get(), base_series, i, cfg.strategy_name);
  StrategySignal sig = strategy->calculate_signal(base_series, i);
  if (sig.type != StrategySignal::Type::HOLD) {
    // routing + fills (omitted)
  }
}
```

### Observations
- Feature flow is correct: cache -> FeatureFeeder -> TFAStrategy::set_raw_features -> `FeatureWindow` buffer grows.
- Projector is initialized once per quarter; expected_feat_dim=55, mapped=55, filled=0.
- No `[TFA calc]` diagnostics appear in logs during TPA despite being added in `calculate_signal`. This hints at either:
  - The TPA path isn’t invoking our `calculate_signal` (contradicts `runner.cpp`), or
  - Logging is suppressed or drowned out, or
  - `calculate_signal` returns immediately before prints (e.g., window not ready) — but FeatureWindow shows buf_size reaching 64.

### Hypotheses (Root Cause)
1. Hidden control flow mismatch: TPA pipeline might gate signals elsewhere (e.g., `SignalEngine` path) while metrics counters report zero. However, `runner.cpp` calls `calculate_signal` directly.
2. Inference result is always failing before logging (e.g., `to_input` or `predict` returning null). No such diagnostics surfaced (we’d see `predict failed`).
3. Output mapping path always returns below `conf_floor` and thus returns default `StrategySignal{}` (HOLD). Unlikely with binary sigmoid since max(conf, 1-conf) ≥ 0.5.
4. Console slice/log volume hides `calculate_signal` diagnostics. Needs focused, earlier prints.

### Next Steps (Proposed Fix/Debugging Plan)
- Add guaranteed first-call prints inside `calculate_signal` right before returning for each branch and re-run a single quarter to reduce noise.
- Temporarily raise `conf_floor` logging detail and log `window_.ready()` transitions.
- Add a unit smoke-test runner that calls `set_raw_features` 64 times and then `calculate_signal` on a small synthetic bar set to exercise the path deterministically.
- If prints still absent, instrument `run_backtest` to log just before/after calling `calculate_signal` to confirm invocation.

### Artifacts & Config
- Model: `artifacts/TFA/v1/model.pt`
- Meta: `artifacts/TFA/v1/model.meta.json` (expects 55 features)
- Spec: `artifacts/TFA/v1/feature_spec.json`
- Feature cache: `data/QQQ_RTH_features.csv` (55 names; cache loader validated)

### Status
- Reproducible; pending deeper instrumentation to locate the exact early return in `calculate_signal` or confirm call path under TPA.

