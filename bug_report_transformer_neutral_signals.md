# Bug Report: Transformer Strategy Emitting Neutral Signals (0.5)

## Summary
The Sentio Transformer Strategy is consistently emitting neutral signals (probability = 0.5) instead of meaningful directional predictions, resulting in zero trading activity during backtests.

## Bug Details

### Issue Description
- **Strategy Name**: `transformer`
- **Observed Behavior**: All signals have probability = 0.5 (neutral)
- **Expected Behavior**: Variable probabilities based on market conditions
- **Impact**: Strategy generates no trades, resulting in 0% returns
- **Severity**: High - Strategy is non-functional for trading

### Error Messages
```
Error in calculate_probability: mat1 and mat2 shapes cannot be multiplied (1x64 and 128x256)
Exception raised from meta at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/LinearAlgebra.cpp:194
```

### Root Cause Analysis
The primary issue appears to be a **tensor dimension mismatch** in the transformer model:

1. **Feature Generation**: The `FeaturePipeline` generates 128-dimensional feature vectors
2. **Model Configuration**: The trained model expects 64-dimensional input sequences
3. **Matrix Multiplication Error**: Attempting to multiply (1x64) with (128x256) tensors
4. **Fallback Behavior**: When the model fails, the strategy defaults to neutral (0.5) probability

### Technical Details

#### Dimension Mismatch
- **Generated Features**: `[128]` (128 features per bar)
- **Model Input Expected**: `[sequence_length=64, feature_dim=128]`
- **Actual Model Input**: `[1, 64]` (incorrect reshaping)
- **Model Weights**: First linear layer expects `[128, 256]`

#### Code Flow Issue
1. `FeaturePipeline::generate_features()` produces 128-dimensional vectors
2. `TransformerStrategy::calculate_probability()` reshapes incorrectly
3. Model forward pass fails with dimension mismatch
4. Exception caught, returns default 0.5 probability

### Test Results
- **Run ID**: 516262
- **Total Signals**: 8,600 signals generated
- **Signal Values**: All signals = 0.5 (neutral)
- **Trades Executed**: 0 trades
- **P&L**: $0.00 (no trading activity)

### Affected Components
1. **Feature Pipeline** (`src/feature_pipeline.cpp`)
2. **Transformer Model** (`src/transformer_model.cpp`)
3. **Strategy Implementation** (`src/strategy_transformer.cpp`)
4. **Model Training** (`src/transformer_trainer_main.cpp`)

### Configuration Issues
- **Training Config**: Model trained with incorrect feature dimensions
- **Runtime Config**: Feature pipeline and model dimension mismatch
- **Sequence Handling**: Incorrect tensor reshaping for transformer input

## Reproduction Steps
1. Build and run transformer trainer: `make build/transformer_trainer`
2. Train model: `./build/transformer_trainer --epochs 5 --data data/test_download/QQQ_RTH_NH.csv`
3. Run strategy test: `./sencli strattest transformer --mode historical --blocks 20`
4. Observe: All signals are 0.5, no trades executed

## Expected Fix
1. **Align Dimensions**: Ensure feature pipeline output matches model input expectations
2. **Fix Tensor Reshaping**: Correct the sequence tensor creation in `calculate_probability()`
3. **Update Training**: Retrain model with correct feature dimensions
4. **Add Validation**: Add dimension validation in model loading

## Priority
**HIGH** - Strategy is completely non-functional for trading purposes.

## Environment
- **OS**: macOS 14.6.0
- **Compiler**: g++ (C++20)
- **PyTorch**: LibTorch C++ API
- **Dataset**: QQQ_RTH_NH.csv (35.1 days, 9,600 bars)

## Additional Notes
- The strategy framework integration is working correctly
- Error handling prevents crashes but masks the underlying issue
- Training completes successfully but produces incompatible model
- All other Sentio systems (audit, position management) function normally

---
**Created**: 2025-09-19
**Reporter**: AI Assistant
**Status**: Open
**Assigned**: Development Team
