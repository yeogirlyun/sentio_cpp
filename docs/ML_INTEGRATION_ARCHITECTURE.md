# ML Integration Architecture

## Overview

The Sentio C++ framework supports seamless integration of Python-trained ML models through a clean, production-grade architecture that separates training (Python) from inference (C++). This hybrid approach leverages Python's ML ecosystem for model development while maintaining C++'s performance for real-time trading.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SENTIO ML PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  OFFLINE TRAINING (Python)           │    ONLINE INFERENCE (C++)             │
│                                       │                                        │
│  ┌─────────────────────┐              │    ┌─────────────────────┐           │
│  │ sentio_trainer/     │              │    │ C++ Strategies      │           │
│  │ ├── trainers/       │              │    │ ├── TFA Strategy    │           │
│  │ │   └── tfa_fast.py │              │    │ ├── HybridPPO       │           │
│  │ ├── cli.py          │              │    │ └── Transformer     │           │
│  │ └── models/         │              │    └─────────────────────┘           │
│  └─────────────────────┘              │                │                       │
│           │                           │                ▼                       │
│           ▼                           │    ┌─────────────────────┐           │
│  ┌─────────────────────┐              │    │ Model Registry      │           │
│  │ Feature Engineering │◀─────────────┼────│ ├── TorchScript      │           │
│  │ (C++ via pybind)    │              │    │ ├── ONNX Runtime    │           │
│  └─────────────────────┘              │    │ └── Metadata        │           │
│           │                           │    └─────────────────────┘           │
│           ▼                           │                │                       │
│  ┌─────────────────────┐              │                ▼                       │
│  │ Model Training      │              │    ┌─────────────────────┐           │
│  │ ├── MLP/Transformer │              │    │ Signal Generation   │           │
│  │ ├── Loss/Optimizer  │              │    │ ├── Feature Window  │           │
│  │ └── TorchScript     │              │    │ ├── Model Inference │           │
│  └─────────────────────┘              │    │ └── Signal Pipeline │           │
│           │                           │    └─────────────────────┘           │
│           ▼                           │                                        │
│  ┌─────────────────────┐              │                                        │
│  │ Artifact Export     │──────────────┼────────────────────────────────────────│
│  │ ├── model.pt        │              │                                        │
│  │ ├── metadata.json   │              │                                        │
│  │ └── feature_spec    │              │                                        │
│  └─────────────────────┘              │                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Offline Training Infrastructure (`sentio_trainer/`)

**⚠️ CRITICAL: This directory contains the Python training codebase for all ML strategies. It must be maintained in the repository at all times.**

#### Core Training Modules:
- **`sentio_trainer/trainers/tfa_fast.py`**: Fast TFA model trainer with C++ feature parity
- **`sentio_trainer/cli.py`**: Command-line interface for training workflows
- **`configs/tfa.yaml`**: TFA training configuration
- **`python/feature_spec.json`**: Feature specification for 55-feature models
- **`train_models.py`**: Main training script (referenced by C++ error messages)

#### Key Features:
- **C++ Feature Parity**: Uses `sentio_features` pybind module to ensure identical feature engineering
- **Fast Training**: Memory-mapped arrays, multi-process DataLoader, optimized for Apple Silicon (MPS)
- **TorchScript Export**: Exports `.pt` models compatible with C++ inference
- **Artifact Generation**: Creates `model.pt`, `metadata.json`, and `feature_spec.json`

#### Usage:
```bash
# Install dependencies
pip install torch numpy pyyaml

# Train TFA model
python train_models.py --config configs/tfa.yaml

# Output artifacts to: artifacts/TFA/v1/
```

### 2. C++ ML Interfaces (`include/sentio/ml/`)

- **`model_registry_ts.hpp`**: TorchScript model loading and management
- **`ts_model.hpp`**: TorchScript model wrapper with safe execution
- **`feature_window.hpp`**: Sequence model window management
- **`onnx_model.hpp`**: ONNX Runtime wrapper (for HybridPPO)

### 3. Strategy Integration

- **`strategy_tfa.hpp`**: TFA transformer strategy (TorchScript)
- **`strategy_hybrid_ppo.hpp`**: HybridPPO strategy (ONNX)
- **`strategy_transformer_ts.hpp`**: Generic transformer strategy
- Plugs directly into existing Sentio pipeline (RTH, SignalGate, Router, PriceBook, Audit)
- Uses standardized `StrategySignal` interface

### 4. Artifact Structure

```
artifacts/
  TFA/                    # Transformer Financial Alpha (TorchScript)
    v1/
      model.pt            # TorchScript model file
      metadata.json       # Model metadata with feature specs
      feature_spec.json   # Feature engineering specification
      model.meta.json     # Extended metadata for C++ loader
  HybridPPO/             # Hybrid PPO strategy (ONNX)
    v1/
      model.onnx          # ONNX model file
      metadata.json       # Feature schema, normalization, actions
  TransformerTS/         # Generic transformer strategies
    v1/
      model.pt            # TorchScript model
      metadata.json       # Model specifications
```

## Usage

### Training (Python)

#### TFA Strategy Training:
```bash
# 1. Install dependencies
pip install torch numpy pyyaml

# 2. Train TFA model using C++ feature parity
python train_models.py --config configs/tfa.yaml

# 3. Artifacts created:
#    artifacts/TFA/v1/model.pt         # TorchScript model
#    artifacts/TFA/v1/metadata.json    # Model specifications
#    artifacts/TFA/v1/feature_spec.json # Feature engineering spec
```

#### Manual Training (Advanced):
```python
from sentio_trainer.trainers.tfa_fast import train_tfa_fast

# Train with custom parameters
train_tfa_fast(
    symbol="QQQ",
    bars_csv="data/equities/QQQ_RTH_NH.csv",
    feature_spec="python/feature_spec.json",
    out_dir="artifacts/TFA/v1",
    epochs=10,
    batch_size=8192,
    lr=1e-3
)
```

#### HybridPPO Training (Legacy):
```python
# Train your model
model = train_hybrid_ppo()

# Export to ONNX
torch.onnx.export(model, dummy_input, "artifacts/HybridPPO/v1/model.onnx")

# Create metadata
metadata = {
    "model_id": "HybridPPO",
    "version": "v1",
    "feature_names": ["ret_1m", "ret_5m", "rsi_14", ...],
    "mean": [0.0, 0.0, 50.0, ...],
    "std": [1.0, 1.0, 20.0, ...],
    "actions": ["SELL", "HOLD", "BUY"]
}
```

### Inference (C++)

#### TFA Strategy Usage:
```cpp
// Create TFA strategy (loads model.pt automatically)
sentio::TFACfg cfg;
cfg.model_id = "TFA";
cfg.version = "v1";
cfg.artifacts_dir = "artifacts";
auto strategy = std::make_unique<sentio::TFAStrategy>(cfg);

// Features are calculated automatically from bars
// via FeatureFeeder and 55-feature pipeline

// Process bar
StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=ts, .is_rth=true};
Bar b{open, high, low, close};
strategy->on_bar(ctx, b);

// Get signal (BUY/SELL/HOLD)
auto signal = strategy->latest();
```

#### HybridPPO Strategy Usage:
```cpp
// Create HybridPPO strategy (loads model.onnx)
HybridPPOCfg cfg;
cfg.artifacts_dir = "artifacts";
cfg.version = "v1";
auto strategy = std::make_unique<HybridPPOStrategy>(cfg);

// Set features manually
std::vector<double> raw_features = {0.0, 0.0, 50.0, ...};
strategy->set_raw_features(raw_features);

// Process bar
StrategyCtx ctx{.instrument="QQQ", .ts_utc_epoch=ts, .is_rth=true};
Bar b{open, high, low, close};
strategy->on_bar(ctx, b);

// Get signal
auto signal = strategy->latest();
```

### CLI Usage

```bash
# Train TFA model
python train_models.py --config configs/tfa.yaml

# Test TFA strategy (requires trained model)
build/sentio_cli tpa_test QQQ --strategy tfa --days 5

# Run backtest with TFA
build/sentio_cli backtest QQQ --strategy tfa

# Run backtest with HybridPPO (legacy)
build/sentio_cli backtest QQQ --strategy hybrid_ppo

# Run with custom parameters
build/sentio_cli backtest QQQ --strategy tfa --params conf_floor=0.1
```

## Features

### ✅ Production-Ready

- **Zero-copy feature pipeline**: Pre-allocated buffers for performance
- **Safe fallback**: Works without ONNX Runtime (returns HOLD)
- **Deterministic**: No RNG in inference, fixed thread counts
- **Hot reload**: Version-based model switching

### ✅ Pluggable Architecture

- **Any ML model**: PPO, LSTM, XGBoost, Random Forest
- **Standardized interface**: Same C++ code for all models
- **Easy integration**: Just export to ONNX + metadata

### ✅ Robust Error Handling

- **Missing models**: Graceful fallback to HOLD
- **Feature mismatches**: Validation and error reporting
- **ONNX errors**: Safe exception handling

## Testing

### Unit Tests

```bash
# Test HybridPPO strategy (fallback mode)
build/test_hybrid_ppo

# Test all ML components
make test_ml
```

### Integration Tests

```bash
# Test with real data
build/sentio_cli backtest QQQ --strategy hybrid_ppo --verbose

# Test audit trail
build/test_audit_replay
```

## Performance

### Latency Optimization

- **Pre-allocated buffers**: No dynamic allocation in hot path
- **ONNX Runtime**: Optimized inference engine
- **Feature caching**: Reuse transformed features when possible

### Memory Efficiency

- **Minimal overhead**: ~1KB per model instance
- **Shared sessions**: Reuse ONNX sessions across strategies
- **Lazy loading**: Load models only when needed

## Security

### Model Validation

- **Content hashing**: Verify model integrity
- **Schema validation**: Ensure feature compatibility
- **Version control**: Track model changes

### Runtime Safety

- **Input validation**: Check feature dimensions
- **Output bounds**: Clamp confidence values
- **Error isolation**: Fail gracefully without crashing

## Future Extensions

### Additional Models

- **TreeProba**: XGBoost/LightGBM/Random Forest
- **LSTM**: Temporal sequence models
- **CNN**: Convolutional feature extractors

### Advanced Features

- **Ensemble methods**: Multiple model voting
- **Online learning**: Incremental model updates
- **A/B testing**: Model comparison framework

## Troubleshooting

### Common Issues

1. **Model not found**: Check artifacts directory and version
2. **Feature mismatch**: Verify metadata.json schema
3. **ONNX errors**: Ensure model compatibility with ONNX Runtime

### Debug Mode

```cpp
// Enable verbose logging
HybridPPOCfg cfg;
cfg.verbose = true;
auto strategy = std::make_unique<HybridPPOStrategy>(cfg);
```

## Contributing

### Adding New ML Strategies

1. Create strategy class inheriting from `IStrategy`
2. Implement `map_output()` for your model type
3. Add to `all_strategies.hpp`
4. Register with `REGISTER_STRATEGY` macro
5. Add unit tests

### Model Export Guidelines

1. Use ONNX opset version 17
2. Include comprehensive metadata
3. Test with C++ fallback mode
4. Validate feature schemas
5. Document model assumptions
