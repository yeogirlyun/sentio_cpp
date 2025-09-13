# Sentio Trainer - Offline ML Training Infrastructure

**⚠️ CRITICAL: This directory contains the Python training codebase for all ML strategies. It must NEVER be deleted and must be maintained in the repository at all times.**

## Overview

The `sentio_trainer/` directory provides the complete offline training infrastructure for machine learning strategies in the Sentio trading system. This enables the Python-to-C++ ML pipeline where models are trained offline in Python and deployed for real-time inference in C++.

## Architecture

```
sentio_trainer/
├── __init__.py                    # Package initialization
├── cli.py                         # Command-line training interface
├── trainers/                      # Strategy-specific trainers
│   ├── __init__.py
│   └── tfa_fast.py               # Fast TFA model trainer
└── README.md                     # This file

Related directories:
├── configs/tfa.yaml              # Training configurations
├── python/feature_spec.json      # Feature engineering specifications
└── train_models.py               # Main training entry point
```

## Key Features

### C++ Feature Parity
- Uses `sentio_features` pybind11 module to ensure identical feature engineering between Python training and C++ inference
- Eliminates Python/C++ feature drift that can cause model failures

### High Performance Training
- Memory-mapped numpy arrays for efficient data loading
- Multi-process DataLoader for parallel training
- Optimized for Apple Silicon (MPS), CUDA, and CPU
- Fast TorchScript compilation and export

### Production Integration
- Exports TorchScript `.pt` models compatible with C++ inference
- Generates comprehensive metadata for C++ model loading
- Creates feature specifications for runtime validation

## Usage

### Train TFA Model
```bash
# Install dependencies
pip install torch numpy pyyaml

# Train using configuration
python train_models.py --config configs/tfa.yaml

# Output artifacts:
# artifacts/TFA/v1/model.pt         # TorchScript model
# artifacts/TFA/v1/metadata.json    # Model specifications  
# artifacts/TFA/v1/feature_spec.json # Feature engineering spec
```

### Test C++ Integration
```bash
# After training, test C++ inference
build/sentio_cli tpa_test QQQ --strategy tfa --days 5
```

## Training Process

1. **Data Loading**: Fast CSV parsing of OHLCV bars
2. **Feature Engineering**: Uses C++ FeatureBuilder via pybind for parity
3. **Label Generation**: Forward-looking returns or price differences
4. **Model Training**: MLP/Transformer with optimized training loop
5. **TorchScript Export**: Baked-in normalization for C++ simplicity
6. **Artifact Generation**: Model, metadata, and feature specifications

## Model Artifacts

### `model.pt` (TorchScript Model)
- Contains trained neural network with baked-in feature normalization
- Ready for C++ inference via LibTorch
- Outputs raw logits (C++ applies sigmoid/thresholding)

### `metadata.json` (Model Specifications)
- Feature names and dimensions
- Input/output specifications
- Training metadata and timestamps

### `feature_spec.json` (Feature Engineering)
- Complete specification for 55-feature pipeline
- Used by both Python training and C++ inference
- Ensures feature consistency across environments

## Integration with C++

The trained models integrate seamlessly with C++ strategies:

```cpp
// C++ strategy automatically loads trained model
sentio::TFACfg cfg;
cfg.model_id = "TFA";
cfg.version = "v1";
auto strategy = std::make_unique<sentio::TFAStrategy>(cfg);

// Features calculated automatically, model inference runs in real-time
```

## Adding New Trainers

To add a new ML strategy trainer:

1. Create `sentio_trainer/trainers/new_strategy.py`
2. Implement training function with consistent signature
3. Add configuration in `configs/new_strategy.yaml`
4. Update `train_models.py` to support new strategy
5. Ensure C++ strategy exists to consume the model

## Maintenance Requirements

- **Never delete this directory** - it's essential for ML model training
- Keep Python dependencies minimal and well-documented
- Maintain feature parity with C++ via pybind11 module
- Version control all training configurations
- Test training pipeline with each major change

## Dependencies

- **PyTorch**: Model training and TorchScript export
- **NumPy**: Efficient array operations and data loading
- **PyYAML**: Configuration file parsing
- **sentio_features**: C++ feature builder via pybind11 (critical for parity)

## Performance Notes

- Use `batch_size >= 8192` for optimal GPU utilization
- Set `num_workers` to CPU core count for fast data loading
- Enable PyTorch compilation for 10-20% speedup
- Use MPS backend on Apple Silicon for GPU acceleration

## Troubleshooting

### Common Issues

1. **Import Error: sentio_features**
   - Ensure C++ pybind module is built and installed
   - Run `pip install -e .` from project root after building

2. **Feature Dimension Mismatch**
   - Verify `feature_spec.json` matches C++ feature pipeline
   - Check emit_from_index alignment between training and inference

3. **TorchScript Export Failure**
   - Use `torch.jit.trace()` fallback if `torch.jit.script()` fails
   - Ensure model uses supported PyTorch operations

### Debug Mode

Enable verbose logging in training:
```python
train_tfa_fast(
    symbol="QQQ",
    bars_csv="data/equities/QQQ_NH.csv",
    feature_spec="python/feature_spec.json",
    # ... other params
    epochs=1,  # Quick test run
    batch_size=1024,  # Smaller batches for debugging
)
```

## Critical Importance

This training infrastructure is **essential** for the ML trading system. Without it:
- No new ML models can be trained
- Existing models cannot be retrained or updated  
- The Python-to-C++ ML pipeline breaks completely
- TFA and other ML strategies become unusable

**Always maintain this directory in the repository and ensure it stays functional.**
