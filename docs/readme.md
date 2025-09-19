# Sentio C++ Trading System

A high-performance quantitative trading system built in C++ with strategy-agnostic architecture, comprehensive audit capabilities, and ML integration for systematic trading.

> **📖 For complete usage instructions, see the [Sentio User Guide](sentio_user_guide.md) which provides comprehensive documentation for both CLI and audit systems.**

## Quick Start

### Prerequisites
 - macOS (tested on macOS 24.6.0)
 - C++20 compatible compiler (AppleClang 17+ recommended)
 - CMake 3.16+
 - LibTorch (PyTorch C++); set Torch_DIR to your Python torch cmake path
 - Polygon.io API key (for market data)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/sentio_cpp.git
cd sentio_cpp
```

2. **Set up environment**
```bash
export POLYGON_API_KEY="your_api_key_here"
```

3. **Build the system**
```bash
cmake -S /Users/yeogirlyun/C++/sentio_cpp -B /Users/yeogirlyun/C++/sentio_cpp/build -DCMAKE_BUILD_TYPE=Release -DTorch_DIR=/Users/yeogirlyun/C++/sentio_cpp/MarS/mars_env/lib/python3.13/site-packages/torch/share/cmake/Torch
cmake --build /Users/yeogirlyun/C++/sentio_cpp/build --target sentio_cli sentio_audit -j8
```

Binaries:
- `/Users/yeogirlyun/C++/sentio_cpp/build/sentio_cli`
- `/Users/yeogirlyun/C++/sentio_cpp/build/sentio_audit`

4. **Download market data (Polygon subcommand)**
```bash
export POLYGON_API_KEY="your_api_key_here"
./build/sentio_cli download QQQ --family --period 3y --timespan minute
```

5. **Run your first strategy test**
```bash
./build/sentio_cli strattest ire QQQ --comprehensive
```

## Core Features

### 🚀 Strategy-Agnostic Architecture
- **Dynamic Strategy Registration**: Add new strategies via configuration files
- **Unified Signal Interface**: All strategies output 0-1 probability signals
- **Extensible Framework**: Seamless integration of new trading strategies
- **Performance Optimized**: C++20 with aggressive optimization flags

### 📊 Comprehensive Audit System
- **Complete Traceability**: Every signal, order, and trade is logged
- **JSONL Format**: Human-readable audit files with SHA1 integrity
- **Replay Capability**: Perfect reproduction of any trading session
- **P&L Verification**: Accurate profit/loss calculation and validation

### 🎯 Leverage Trading Support
- **Multiple Instruments**: QQQ (1x), PSQ (1x inverse), TQQQ (3x), SQQQ (3x inverse)
- **Dynamic Allocation**: Automatic instrument selection based on signal strength
- **Risk Management**: Position limits, volatility targeting, correlation controls
- **Performance Optimization**: Maximize capital efficiency for optimal returns

### 🤖 Machine Learning Integration
- **Offline Training**: Python-based model training with C++ inference
- **Multiple ML Strategies**: IRE, TFA, PPO, Transformer models
- **Feature Engineering**: 55-feature pipeline with technical indicators
- **Model Deployment**: TorchScript and ONNX runtime support

### 🔍 Signal Diagnostics
- **Real-time Monitoring**: Comprehensive signal generation diagnostics
- **Drop Reason Analysis**: Detailed breakdown of signal filtering
- **Performance Metrics**: Signal quality, execution rates, P&L tracking
- **Troubleshooting Tools**: Built-in diagnostic tools for signal issues

## Available Strategies

| Strategy | Type | Description | Instruments |
|----------|------|-------------|-------------|
| **IRE** | ML Ensemble | Integrated Rule Ensemble with dynamic leverage | QQQ, PSQ, TQQQ, SQQQ |
| **TFA** | Transformer | Deep learning transformer for sequence modeling | QQQ, PSQ |
| **BollingerSqueezeBreakout** | Technical | Bollinger Bands squeeze and breakout strategy | QQQ, PSQ |
| **MomentumVolume** | Technical | Price momentum combined with volume analysis | QQQ, PSQ |
| **VWAPReversion** | Technical | Mean reversion around Volume Weighted Average Price | QQQ, PSQ |
| **OrderFlowImbalance** | Microstructure | Market microstructure analysis | QQQ, PSQ |
| **MarketMaking** | Microstructure | Bid-ask spread capture strategy | QQQ, PSQ |
| **OrderFlowScalping** | High-Frequency | Scalping based on order flow patterns | QQQ, PSQ |
| **OpeningRangeBreakout** | Technical | Breakout strategy using opening range patterns | QQQ, PSQ |
| **KochiPPO** | Reinforcement Learning | Proximal Policy Optimization for trading | QQQ, PSQ |
| **HybridPPO** | Hybrid ML | Multiple ML techniques combined | QQQ, PSQ |

## Usage Examples

### Strategy Testing
```bash
# Basic strategy test
./build/sentio_cli strattest ire QQQ

# Comprehensive robustness test
./build/sentio_cli strattest ire QQQ --comprehensive --stress-test

# Quick development test
./build/sentio_cli strattest tfa QQQ --quick --duration 1d

# Canonical Trading Block evaluation (preferred)
./build/sentio_cli strattest tfa QQQ --blocks 20 --block-size 480
```

### Audit Analysis
```bash
# Show performance summary
./build/sentio_audit summarize

# Analyze trade execution
./build/sentio_audit trade-flow --buy --sell

# Review signal quality
./build/sentio_audit signal-flow --max 100

# Check position history
./build/sentio_audit position-history --max 50
```

### Transformer: Train and Evaluate
```bash
# Discover trainer/evaluator options
./build/transformer_trainer_main --help
./build/transformer_evaluator_main --help

# Example: quick training run (adjust data path/flags per --help)
./build/transformer_trainer_main --data data/equities/QQQ_RTH_NH.csv --epochs 20

# Example: evaluate a saved run or predictions (see --help for exact args)
./build/transformer_evaluator_main --input results/preds_labels.csv
```

### Data Management (Polygon)
```bash
# Download latest data
./build/sentio_cli download QQQ --family --period 1y --timespan minute

# Download specific timespan
./build/sentio_cli download SPY --period 6m --timespan day

# Check system status
./build/sentio_cli probe
```

## Configuration

### Strategy Configuration (`configs/strategies.json`)
```json
{
  "strategies": [
    {
      "name": "IRE",
      "class": "IREStrategy",
      "enabled": true,
      "parameters": {
        "lookback_period": 20,
        "volatility_target": 0.15,
        "max_position_pct": 0.8
      }
    }
  ]
}
```

### TFA Configuration (`configs/tfa.yaml`)
```yaml
model:
  sequence_length: 50
  attention_heads: 8
  hidden_dim: 256
  num_layers: 6

training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  validation_split: 0.2
```

## Performance Metrics

### System Performance
- **Signal Generation**: < 1ms per bar
- **Order Processing**: < 100μs per order
- **Audit Logging**: < 10μs per event
- **Memory Usage**: < 100MB base, < 1MB per 10K events

### Trading Performance
- **Target Daily Trades**: 10-100 trades/day
- **Sharpe Ratio**: Optimized for highest Sharpe score
- **Monthly Return**: Maximize monthly projected return rate
- **Drawdown Control**: Dynamic risk management

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SENTIO TRADING SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  DATA LAYER          │  STRATEGY LAYER        │  EXECUTION LAYER             │
│  ├── Polygon API     │  ├── IRE Strategy     │  ├── Router                  │
│  ├── Data Downloader │  ├── TFA Strategy     │  ├── Sizer                   │
│  ├── Data Aligner    │  ├── Technical Strate │  ├── Order Management        │
│  └── Binary Cache    │  └── ML Strategies    │  └── Position Management     │
├─────────────────────────────────────────────────────────────────────────────┤
│  AUDIT LAYER         │  DIAGNOSTICS LAYER    │  ML LAYER                    │
│  ├── Event Logging   │  ├── Signal Diagnostics│  ├── Feature Engineering     │
│  ├── P&L Tracking    │  ├── Performance Metrics│  ├── Model Training        │
│  ├── Replay Engine   │  └── Troubleshooting   │  ├── Model Deployment        │
│  └── Analysis Tools  │                       │  └── Inference Engine       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Development

### Building from Source
```bash
# Configure (Release)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTorch_DIR=$PYTORCH_CMAKE

# Build key targets
cmake --build build --target sentio_cli sentio_audit -j8

# Other targets
cmake --build build --target transformer_trainer_main transformer_evaluator_main transformer_tests -j8

# Run tests
./build/transformer_tests
```

### Adding New Strategies
1. Create strategy class inheriting from `BaseStrategy`
2. Implement required virtual methods
3. Add to `configs/strategies.json`
4. Register with `REGISTER_STRATEGY` macro
5. Add unit tests

### Code Quality
- **C++20 Standard**: Modern C++ features and best practices
- **Memory Safety**: RAII, smart pointers, no raw pointers
- **Performance**: Aggressive optimization, SIMD instructions
- **Testing**: Comprehensive unit and integration tests
- **Documentation**: Inline documentation and architecture guides

## Troubleshooting

### Common Issues

1. **No signals generated**
   - Check warmup period configuration
   - Verify RTH (Regular Trading Hours) settings
   - Check data quality and completeness

2. **Signals but no trades**
   - Verify router configuration
   - Check sizer parameters
   - Review order execution logic

3. **High drop rate**
   - Analyze drop reason breakdown
   - Adjust gate parameters
   - Review strategy logic

4. **Performance issues**
   - Check memory usage patterns
   - Verify optimization flags
   - Review data access patterns

### Diagnostic Tools
```bash
# Strategy validation
./build/sentio_cli audit-validate

# System health check
./build/sentio_cli probe

# Comprehensive strategy test
./build/sentio_cli strattest ire QQQ --comprehensive

# Audit analysis
./build/sentio_audit summarize --detailed
```

## Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Run full test suite
5. Submit pull request

### Code Standards
- Follow C++20 best practices
- Maintain comprehensive test coverage
- Document public APIs
- Use consistent naming conventions
- Optimize for performance

### Testing Requirements
- Unit tests for all new components
- Integration tests for system interactions
- Performance benchmarks for critical paths
- Sanity checks for data integrity

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Documentation**: See `docs/architecture.md` for detailed architecture
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join community discussions via GitHub Discussions
- **Wiki**: Check the project wiki for additional resources

## Acknowledgments

- **Polygon.io**: Market data provider
- **PyTorch**: ML framework for model training
- **ONNX Runtime**: Model inference engine
- **C++20 Standard**: Modern C++ features and performance

---

**Note**: This system is designed for research and educational purposes. Always test thoroughly before using in production trading environments.
