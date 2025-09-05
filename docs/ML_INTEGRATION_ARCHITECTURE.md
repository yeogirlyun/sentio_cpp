# ML Integration Architecture

## Overview

The Sentio C++ framework now supports seamless integration of Python-trained ML models through a clean, production-grade architecture that separates training (Python) from inference (C++).

## Architecture

```
Python Training (Offline)          C++ Inference (Online)
┌─────────────────────┐           ┌─────────────────────┐
│ 1. Train Model      │           │ 1. Load ONNX Model  │
│ 2. Export to ONNX   │──────────▶│ 2. Apply Features   │
│ 3. Create Metadata  │           │ 3. Generate Signals │
└─────────────────────┘           └─────────────────────┘
```

## Key Components

### 1. ML Interfaces (`include/sentio/ml/`)

- **`iml_model.hpp`**: Base interfaces for ML models
- **`onnx_model.hpp`**: ONNX Runtime wrapper with safe fallback
- **`feature_pipeline.hpp`**: Feature normalization and clipping
- **`model_registry.hpp`**: Artifact loading and model management

### 2. Strategy Integration

- **`strategy_hybrid_ppo.hpp`**: HybridPPO strategy wrapper
- Plugs directly into existing Sentio pipeline (RTH, SignalGate, Router, PriceBook, Audit)
- Uses standardized `StrategySignal` interface

### 3. Artifact Structure

```
artifacts/
  HybridPPO/
    v1/
      model.onnx          # ONNX model file
      metadata.json       # Feature schema, normalization, actions
    v2/
      model.onnx
      metadata.json
  TreeAlpha/
    v3/
      model.onnx
      metadata.json
```

## Usage

### Training (Python)

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

```cpp
// Create strategy
HybridPPOCfg cfg;
cfg.artifacts_dir = "artifacts";
cfg.version = "v1";
auto strategy = std::make_unique<HybridPPOStrategy>(cfg);

// Set features
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
# Run backtest with HybridPPO
build/sentio_cli backtest QQQ --strategy hybrid_ppo

# Run with custom parameters
build/sentio_cli backtest QQQ --strategy hybrid_ppo --params conf_floor=0.1
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
