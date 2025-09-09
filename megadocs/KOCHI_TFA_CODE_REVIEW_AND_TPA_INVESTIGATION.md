# KOCHI_PPO vs TFA: Architecture, Training/Inference, Cache Policy, and TPA Investigation

## Scope
- Compare architecture parity between `TFA` and `KochiPPO` end-to-end
- Identify bugs/inefficiencies in training and inference
- Enforce cache-only training policy and audit runtime cache usage
- Investigate why `KochiPPO` emitted no trades in initial TPA run

## Key Components and Files
- Trainers (Python)
  - TFA: `sentio_trainer/trainers/tfa_seq.py` (seq transformer), `sentio_trainer/trainers/tfa_fast.py`
  - Kochi: `sentio_trainer/trainers/kochi_ppo.py`
  - Feature cache helpers: `sentio_trainer/utils/feature_cache.py`, `tools/generate_kochi_feature_cache.py`
  - Kochi features (standalone): `sentio_trainer/utils/kochi_features.py`
- C++ Runtime
  - Strategy classes: `src/strategy_tfa.cpp`, `src/strategy_kochi_ppo.cpp`, headers in `include/sentio/*.hpp`
  - Feature feeder/cache: `src/feature_feeder.cpp`, `src/feature_cache.cpp`, `include/sentio/feature_cache.hpp`
  - TorchScript model loading: `src/ml/model_registry_ts.cpp`, `include/sentio/ml/model_registry.hpp`
  - CLI/TPA entrypoint: `src/main.cpp`

## Architecture Parity Checklist
- Model export format: TorchScript PT with sidecar `metadata.json`
  - TFA: yes (seq_len, feature_names, mean/std, input layout)
  - KochiPPO: yes (seq_len, feature_names, mean/std, clip, actions, input layout). Model id `KochiPPO`.
- Feature normalization at inference: done in C++ `FeatureWindow` using metadata `mean/std/clip2`
  - TFA: yes
  - KochiPPO: yes
- Input layout: `BTF` (Batch, Time, Features)
  - TFA: yes
  - KochiPPO: yes (wrapper `WrappedNoScaler` expects flattened input; C++ reshapes accordingly)
- Sequence window length and readiness gating
  - TFA: gates on `seq_len` readiness
  - KochiPPO: gates on `seq_len` readiness via `ml::FeatureWindow::ready()`
- Feature cache policy
  - Training: cache-only enforced for both TFA and KochiPPO
  - Inference (C++): uses cache if loaded; for KochiPPO in TPA path, cache is required and hard-fails if missing

## Training Path Review
- TFA (`tfa_seq.py`)
  - Reads bars for labels only; features loaded via `load_cached_features()`; hard error if missing
  - Scaler computed on train split; spec used only for metadata capture
  - Potential inefficiency: duplicated spec hashing and file IO per run — acceptable; minor
- KochiPPO (`kochi_ppo.py`)
  - Requires KOCHI cache; searches bars dir and its parent; raises if missing
  - Simple CNN classifier producing 3-class logits [SELL, HOLD, BUY]
  - Potential improvements:
    - Class imbalance handling (weights) and label smoothing
    - Better optimizer schedule; early stopping
    - Deterministic seeds for reproducibility
    - Windowed sample construction could be vectorized further, but acceptable for dataset sizes tested

## Inference Path Review (C++)
- TorchScript metadata parsing: `src/ml/model_registry_ts.cpp`
  - Reads feature_names, mean, std, clip, actions, seq_len, input_layout
  - Risk: simplistic JSON parser (string scanning). Consider migrating to `nlohmann::json` for robustness
- FeatureWindow (`include/sentio/ml/feature_window.hpp`)
  - Normalization applied per metadata; reusable buffer avoids allocations
  - Safety checks on mean/std sizes are currently commented out — re-enable once metadata is stable
- FeatureFeeder (`src/feature_feeder.cpp`)
  - Cache path: if enabled and present, feeds cached features; skips normalization/validation (assumed pre-validated)
  - Non-cache path: computes features; normalized via FeatureNormalizer for non-ML strategies
  - Improvement: unify validation and add optional asserts on feature count per strategy
- Strategy mapping
  - TFA: maps logits→thresholded buy/sell internally (reference `strategy_tfa.cpp`)
  - KochiPPO: `map_output()` selects argmax over probs and maps via `actions` in metadata; applies `conf_floor`

## Initial TPA Findings for KochiPPO (no trades)
- Observation
  - TPA on QQQ (4 quarters) shows healthy feature extraction but 0.00% return and Sharpe 0.000; signals were not materialized into trades
- Likely Causes
  1) Output interpretation mismatch
     - TorchScript wrapper outputs raw logits; C++ calls `model->predict`, which returns probabilities in `ml::ModelOutput::probs`
     - Verify that `TorchScriptModel::predict` applies softmax when model output is logits. If not, `probs` may be unnormalized logits, causing `pmax` to be near 0 after any downstream normalization
  2) High confidence floor
     - `KochiPPOStrategy` uses `conf_floor` as minimum `pmax` to emit (default 0.05). If logits→softmax missing, `pmax` could be < 0.05, suppressing signals
  3) Action mapping mismatch
     - `metadata.json` actions order must be ["SELL","HOLD","BUY"] — confirm this is exactly how training produced labels. If order differs, argmax may map to HOLD frequently
  4) Window not fully ready vs. progress
     - The window fills and is ready; DIAG logs confirm push success. Not likely the cause

## Actionable Checks and Fixes
- Verify TorchScript prediction output
  - Inspect `src/ml/ts_model.cpp` for whether `softmax` is applied to model outputs (logits) before populating `ModelOutput::probs`
  - If not applied, add softmax in C++ predict, or export from Python with softmax head
- Lower confidence floor for testing
  - In `KochiPPOStrategy`, set `conf_floor` param to 0.0 during test to see if trades appear
- Confirm `actions` ordering
  - Open `artifacts/KochiPPO/v1/metadata.json` and ensure actions are `["SELL","HOLD","BUY"]`
  - If different, adjust `map_output` or ensure training writes the same ordering
- Add DIAG dump of first N `mo.probs` per bar when no signals are emitted to confirm distribution

## Performance and Reliability Improvements
- Use `nlohmann::json` for model metadata parsing (robustness)
- Re-enable `FeatureWindow` mean/std length checks
- Add end-to-end parity test
  - Load metadata, create synthetic features with known distribution, and verify identical normalization between Python and C++
- Trainer reproducibility: set seeds for NumPy and Torch

## Repro Steps
1) Generate Kochi cache
   - `python tools/generate_kochi_feature_cache.py --symbol QQQ --bars data/equities/QQQ.csv --outdir data`
2) Train Kochi
   - `PYTHONPATH=. python -c "from sentio_trainer.trainers import train_kochi_ppo as t; t(symbol='QQQ', bars_csv='data/equities/QQQ.csv', out_dir='artifacts/KochiPPO/v1')"`
3) Build & run TPA
   - `make build/sentio_cli -j4`
   - `build/sentio_cli tpa_test QQQ --strategy kochi_ppo --quarters 4`
4) Compare with TFA
   - Ensure `data/QQQ_RTH_features.csv` exists
   - `build/sentio_cli tpa_test QQQ --strategy TFA --quarters 4`

## Appendix: Source Modules
- Trainers
  - `sentio_trainer/trainers/tfa_seq.py`
  - `sentio_trainer/trainers/tfa_fast.py`
  - `sentio_trainer/trainers/kochi_ppo.py`
  - `sentio_trainer/utils/feature_cache.py`
  - `sentio_trainer/utils/kochi_features.py`
  - `tools/generate_kochi_feature_cache.py`
- C++ Runtime
  - `src/strategy_tfa.cpp`, `include/sentio/strategy_tfa.hpp`
  - `src/strategy_kochi_ppo.cpp`, `include/sentio/strategy_kochi_ppo.hpp`
  - `src/feature_feeder.cpp`, `include/sentio/feature_cache.hpp`, `src/feature_cache.cpp`
  - `src/ml/model_registry_ts.cpp`, `include/sentio/ml/model_registry.hpp`, `include/sentio/ml/feature_window.hpp`
  - `src/main.cpp`
