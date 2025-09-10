# Sentio C++ Architecture Document

## Overview

Sentio is a high-performance quantitative trading system built in C++ that implements a strategy-agnostic architecture for systematic trading. The system is designed to maximize profit through sophisticated signal generation, dynamic allocation, and comprehensive audit capabilities.

## Core Architecture Principles

### 1. Strategy Agnostic Design
- **Runner/Router/Sizer Independence**: Core execution components are completely decoupled from strategy-specific logic
- **Dynamic Strategy Registration**: Strategies are loaded from configuration files without code modifications
- **Unified Signal Interface**: All strategies output probability-based signals (0-1) for consistent processing
- **Extensible Framework**: New strategies integrate seamlessly without architectural changes

### 2. Profit Maximization Mandate
- **Primary Goal**: Maximize monthly projected return rate
- **Capital Efficiency**: Optimize for highest Sharpe score and healthy daily trading range (10-100 trades/day)
- **SOTA Optimization**: Remove artificial constraints, let utility-maximizing frameworks determine optimal allocation
- **Performance Over Safety**: Bias towards aggressive profit-seeking parameters

## System Components

### 1. Strategy Layer

#### Base Strategy Interface
```cpp
class BaseStrategy {
public:
    // Core signal generation - all strategies must implement
    virtual double calculate_probability(const std::vector<Bar>& bars, int current_index) = 0;
    
    // Strategy-agnostic allocation decisions
    virtual std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, int current_index,
        const std::string& base_symbol, const std::string& bull3x_symbol,
        const std::string& bear3x_symbol, const std::string& bear1x_symbol) = 0;
    
    // Strategy-specific configurations
    virtual RouterCfg get_router_config() const = 0;
    virtual SizerCfg get_sizer_config() const = 0;
    
    // Dynamic allocation requirements
    virtual bool requires_dynamic_allocation() const { return false; }
};
```

#### Strategy Signal Standardization
- **Probability Range**: All signals output 0-1 probability where:
  - `1.0` = Very strong buy signal
  - `0.8-1.0` = Strong buy
  - `0.6-0.8` = Buy
  - `0.4-0.6` = Hold/Neutral
  - `0.2-0.4` = Sell
  - `0.0-0.2` = Strong sell
  - `0.0` = Very strong sell signal

#### Registered Strategies
1. **IRE (Integrated Rule Ensemble)**: ML-based ensemble with dynamic leverage optimization
2. **TFA (Transformer Financial Analysis)**: Deep learning transformer for sequence modeling
3. **BollingerSqueezeBreakout**: Technical analysis strategy using Bollinger Bands
4. **MomentumVolume**: Price momentum combined with volume analysis
5. **VWAPReversion**: Mean reversion around Volume Weighted Average Price
6. **OrderFlowImbalance**: Market microstructure analysis
7. **MarketMaking**: Bid-ask spread capture strategy
8. **OrderFlowScalping**: High-frequency scalping based on order flow
9. **OpeningRangeBreakout**: Breakout strategy using opening range patterns
10. **KochiPPO**: Reinforcement learning using Proximal Policy Optimization
11. **HybridPPO**: Hybrid approach combining multiple ML techniques

### 2. Execution Layer

#### Runner (Strategy Agnostic)
```cpp
// Core execution loop - completely strategy agnostic
for (const auto& decision : allocation_decisions) {
    if (std::abs(decision.target_weight) > 1e-6) {
        execute_target_position(decision.instrument, decision.target_weight, 
                              portfolio, ST, pricebook, sizer, cfg, series, 
                              bar, chain_id, audit, logging_enabled, total_fills);
    }
}
```

#### Router
- **Instrument Selection**: Routes signals to appropriate instruments (QQQ, PSQ, TQQQ, SQQQ)
- **Leverage Management**: Handles 1x, 3x leveraged instruments
- **Risk Controls**: Implements position limits and exposure controls

#### Sizer
- **Position Sizing**: Calculates optimal position sizes based on volatility targets
- **Risk Management**: Implements maximum position percentage limits
- **Dynamic Allocation**: Adjusts sizing based on market conditions

### 3. Signal Diagnostics System

#### Signal Pipeline Architecture
```
Bar Data → Feature Extraction → Strategy Signal → Signal Gate → Signal Trace → Router
```

#### Diagnostic Components
1. **SignalGate**: Filters and validates signals before execution
2. **SignalTrace**: Records signal history for analysis
3. **SignalDiag**: Provides real-time signal diagnostics

#### Diagnostic Output Format
```
Signal Diagnostics for [Strategy]:
├── Signal Generation: [Status]
├── Feature Pipeline: [Status]
├── Signal Validation: [Status]
├── Execution Pipeline: [Status]
└── Performance Metrics: [Status]
```

#### Common Signal Issues
- **Feature Pipeline Failures**: Missing or corrupted feature data
- **Signal Validation Errors**: Invalid signal ranges or formats
- **Execution Pipeline Issues**: Router or sizer failures
- **Performance Degradation**: Suboptimal signal quality

#### Troubleshooting Guide
- **No Signals Generated**: Check warmup period, RTH configuration, data quality
- **Low Signal Rate**: Analyze drop reasons, adjust gate parameters
- **Signals But No Trades**: Verify router/sizer configuration
- **High Drop Rate**: Review gate configuration and strategy logic

### 4. Audit System

#### Audit Architecture
```
Execution Events → AuditRecorder → JSONL Files → AuditReplayer → Analysis Tools
```

#### Event Types
1. **Signal Events**: Strategy signal generation and validation
2. **Order Events**: Order creation, modification, and cancellation
3. **Fill Events**: Trade execution with P&L calculation
4. **Position Events**: Position changes and portfolio updates

#### Audit File Format (JSONL)
```json
{"ts": 1640995200, "type": "signal", "strategy": "IRE", "symbol": "QQQ", "signal": "BUY", "probability": 0.85, "chain_id": "1640995200:123"}
{"ts": 1640995200, "type": "order", "symbol": "QQQ", "side": "BUY", "quantity": 100, "price": 450.25}
{"ts": 1640995200, "type": "fill", "symbol": "QQQ", "quantity": 100, "price": 450.25, "pnl_d": 1250.50}
```

#### Audit Analysis Tools
- **`audit_cli.py`**: Unified Python interface for audit analysis
- **`cmd_replay`**: Replay audit events with performance metrics
- **`cmd_latest`**: Analyze most recent audit file
- **`cmd_trades`**: Export trades to CSV format
- **`cmd_analyze`**: Comprehensive performance analysis

#### Instrument Distribution Analysis
```python
def analyze_instrument_distribution(audit_file):
    """Analyze instrument distribution, P&L, and win rates"""
    instruments = defaultdict(lambda: {'trades': 0, 'pnl': 0.0, 'wins': 0})
    
    for event in parse_audit_events(audit_file):
        if event['type'] == 'fill':
            symbol = event['symbol']
            pnl = event['pnl_d']
            instruments[symbol]['trades'] += 1
            instruments[symbol]['pnl'] += pnl
            if pnl > 0:
                instruments[symbol]['wins'] += 1
    
    return instruments
```

### 5. Leverage Trading System

#### Leveraged Instruments
- **QQQ**: 1x NASDAQ-100 ETF (base instrument)
- **PSQ**: 1x inverse NASDAQ-100 ETF (bear 1x)
- **TQQQ**: 3x leveraged NASDAQ-100 ETF (bull 3x)
- **SQQQ**: 3x inverse NASDAQ-100 ETF (bear 3x)

#### Dynamic Leverage Allocation
```cpp
// IRE Strategy Example
std::vector<AllocationDecision> IREStrategy::get_allocation_decisions(...) {
    double probability = calculate_probability(bars, current_index);
    
    if (probability > 0.8) {
        // Strong bullish signal - allocate to TQQQ
        return {{bull3x_symbol, 0.8, probability, "Strong bullish signal"}};
    } else if (probability < 0.2) {
        // Strong bearish signal - allocate to SQQQ
        return {{bear3x_symbol, 0.8, 1.0-probability, "Strong bearish signal"}};
    } else if (probability > 0.6) {
        // Moderate bullish signal - allocate to QQQ
        return {{base_symbol, 0.6, probability, "Moderate bullish signal"}};
    } else if (probability < 0.4) {
        // Moderate bearish signal - allocate to PSQ
        return {{bear1x_symbol, 0.6, 1.0-probability, "Moderate bearish signal"}};
    }
    
    return {}; // No allocation for neutral signals
}
```

#### Leverage Risk Management
- **Position Limits**: Maximum exposure per instrument
- **Volatility Targeting**: Dynamic sizing based on instrument volatility
- **Correlation Controls**: Limits on correlated positions
- **Drawdown Protection**: Automatic position reduction during drawdowns

### 6. Polygon Interface

#### Data Pipeline
```
Polygon API → Data Downloader → Data Aligner → Binary Cache → Strategy Processing
```

#### Data Downloader (`tools/data_downloader.py`)
```python
def download_symbol_data(symbol, years=3, api_key=None):
    """Download historical data from Polygon.io"""
    start_date = datetime.now() - timedelta(days=years*365)
    end_date = datetime.now()
    
    # Download bars with RTH filtering
    bars = polygon_client.get_bars(symbol, start_date, end_date, 
                                 timespan='minute', adjusted=True)
    
    # Filter for Regular Trading Hours
    rth_bars = filter_rth_bars(bars)
    
    # Save to CSV format
    save_bars_to_csv(rth_bars, f"{symbol}_RTH_NH.csv")
```

#### Data Alignment (`tools/align_bars.py`)
```python
def align_bars(symbols, output_dir="data"):
    """Align timestamps across multiple symbols"""
    all_bars = {}
    for symbol in symbols:
        bars = load_bars_from_csv(f"{symbol}_RTH_NH.csv")
        all_bars[symbol] = bars
    
    # Find common timestamps
    common_timestamps = find_common_timestamps(all_bars)
    
    # Align all symbols to common timestamps
    aligned_bars = align_to_timestamps(all_bars, common_timestamps)
    
    # Save aligned data
    for symbol, bars in aligned_bars.items():
        save_bars_to_csv(bars, f"{symbol}.csv")
```

#### Binary Cache System
- **Format**: Custom binary format for fast loading
- **Compression**: LZ4 compression for storage efficiency
- **Indexing**: Fast timestamp-based lookups
- **Validation**: SHA1 checksums for data integrity

### 7. Machine Learning Approaches

#### Offline Training Pipeline
```
Historical Data → Feature Engineering → Model Training → Model Validation → Model Deployment
```

#### ML Strategy Types

##### 1. IRE (Integrated Rule Ensemble)
```cpp
class IREStrategy : public BaseStrategy {
private:
    std::unique_ptr<ml::ModelHandle> model_;
    ml::FeaturePipeline feature_pipeline_;
    
public:
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override {
        // Extract features
        auto features = feature_pipeline_.extract_features(bars, current_index);
        
        // Get model prediction
        auto output = model_->predict(features);
        
        // Convert to probability
        return sigmoid(output.prediction);
    }
};
```

##### 2. TFA (Transformer Financial Analysis)
```cpp
class TFAStrategy : public BaseStrategy {
private:
    std::unique_ptr<ml::TransformerModel> transformer_;
    ml::SequenceContext seq_context_;
    
public:
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override {
        // Build sequence context
        seq_context_.update(bars, current_index);
        
        // Get transformer prediction
        auto output = transformer_->forward(seq_context_);
        
        // Extract probability
        return output.probability;
    }
};
```

##### 3. PPO Strategies (KochiPPO, HybridPPO)
```cpp
class KochiPPOStrategy : public BaseStrategy {
private:
    std::unique_ptr<ml::PPOAgent> ppo_agent_;
    ml::ActionSpace action_space_;
    
public:
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override {
        // Get state representation
        auto state = extract_state(bars, current_index);
        
        // Get PPO action
        auto action = ppo_agent_->act(state);
        
        // Convert action to probability
        return action_to_probability(action);
    }
};
```

#### Feature Engineering
```cpp
class FeaturePipeline {
public:
    struct FeatureSet {
        std::vector<double> technical_indicators;
        std::vector<double> volume_metrics;
        std::vector<double> volatility_measures;
        std::vector<double> momentum_signals;
    };
    
    FeatureSet extract_features(const std::vector<Bar>& bars, int current_index) {
        FeatureSet features;
        
        // Technical indicators
        features.technical_indicators = {
            calculate_rsi(bars, current_index, 14),
            calculate_sma(bars, current_index, 20),
            calculate_bollinger_position(bars, current_index, 20, 2.0)
        };
        
        // Volume metrics
        features.volume_metrics = {
            calculate_volume_ratio(bars, current_index, 20),
            calculate_vwap_deviation(bars, current_index, 20)
        };
        
        // Volatility measures
        features.volatility_measures = {
            calculate_atr(bars, current_index, 14),
            calculate_volatility(bars, current_index, 20)
        };
        
        // Momentum signals
        features.momentum_signals = {
            calculate_momentum(bars, current_index, 10),
            calculate_rate_of_change(bars, current_index, 5)
        };
        
        return features;
    }
};
```

#### Model Training Infrastructure
```python
# tools/ml_training.py
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.feature_pipeline = FeaturePipeline(config.features)
        self.model = self.create_model(config.model_type)
    
    def train(self, training_data, validation_data):
        """Train model with offline data"""
        # Feature extraction
        X_train = self.feature_pipeline.extract_features(training_data)
        y_train = self.extract_labels(training_data)
        
        # Model training
        self.model.fit(X_train, y_train)
        
        # Validation
        X_val = self.feature_pipeline.extract_features(validation_data)
        y_val = self.extract_labels(validation_data)
        
        # Evaluate performance
        metrics = self.model.evaluate(X_val, y_val)
        
        return metrics
    
    def export_model(self, output_path):
        """Export trained model for C++ deployment"""
        self.model.save(output_path)
        self.feature_pipeline.save_config(f"{output_path}_features.json")
```

#### Model Deployment
```cpp
class ModelHandle {
public:
    static std::unique_ptr<ModelHandle> load(const std::string& model_path) {
        auto handle = std::make_unique<ModelHandle>();
        handle->load_model(model_path);
        handle->load_feature_config(f"{model_path}_features.json");
        return handle;
    }
    
    ModelOutput predict(const std::vector<double>& features) {
        // Preprocess features
        auto processed_features = preprocess_features(features);
        
        // Run inference
        auto output = run_inference(processed_features);
        
        // Postprocess output
        return postprocess_output(output);
    }
};
```

### 8. Sanity System

#### Comprehensive Validation Framework
The sanity system provides drop-in C++20 validation checks across the entire pipeline from data ingestion through P&L calculation.

#### Core Components
1. **Sanity Framework**: `SanityIssue` and `SanityReport` for structured error reporting
2. **Deterministic Simulator**: Generates synthetic minute-bar series with realistic market regimes
3. **Property Testing Harness**: Fuzz-like testing framework for invariant validation
4. **Integration Examples**: Complete workflow demonstrations

#### High-Value Bug Detection
- **Time Integrity Issues**: Non-monotonic timestamps, incorrect bar spacing
- **Data Quality Problems**: NaN/Infinity propagation, negative prices, invalid OHLC relationships
- **Instrument Mismatches**: Routed to instruments not in PriceBook
- **Order Execution Errors**: BUY orders with negative quantities, sub-minimum share quantities
- **P&L Calculation Issues**: Equity != cash + realized + mark-to-market
- **Audit Trail Problems**: Fills exceeding orders, missing event sequences

#### Usage Patterns
```cpp
// During data ingestion
SanityReport rep;
sanity::check_bar_monotonic(bars, 60, rep);
sanity::check_bar_values_finite(bars, rep);

// During strategy execution
sanity::check_signal_confidence_range(signal.confidence, rep, timestamp);
sanity::check_routed_instrument_has_price(pricebook, routed_instrument, rep, timestamp);

// During order execution
sanity::check_order_qty_min(qty, min_shares, rep, timestamp);
sanity::check_order_side_qty_sign_consistency(side, qty, rep, timestamp);

// End-of-run validation
sanity::check_equity_consistency(account, positions, pricebook, rep);
sanity::check_audit_counts(event_counts, rep);
```

## Performance Optimization

### 1. Compilation Optimization
- **Release Builds**: `-O3 -march=native -DNDEBUG`
- **Link Time Optimization**: `-flto`
- **Profile Guided Optimization**: `-fprofile-use`

### 2. Runtime Optimization
- **Memory Pool Allocation**: Pre-allocated memory pools for frequent allocations
- **SIMD Instructions**: Vectorized operations for numerical computations
- **Cache-Friendly Data Structures**: Optimized memory layout for hot paths
- **Lock-Free Data Structures**: Minimize synchronization overhead

### 3. Data Access Optimization
- **Binary Cache**: Fast binary format for historical data
- **Memory Mapping**: Memory-mapped files for large datasets
- **Compression**: LZ4 compression for storage efficiency
- **Indexing**: Fast timestamp-based lookups

## Configuration Management

### 1. Strategy Configuration (`configs/strategies.json`)
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
    },
    {
      "name": "TFA",
      "class": "TFAStrategy",
      "enabled": true,
      "parameters": {
        "sequence_length": 50,
        "attention_heads": 8,
        "max_position_pct": 0.6
      }
    }
  ]
}
```

### 2. TFA Configuration (`configs/tfa.yaml`)
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

features:
  technical_indicators: true
  volume_metrics: true
  volatility_measures: true
  momentum_signals: true
```

## Strategy Evaluation Framework

### 1. Signal Quality Evaluation

#### Universal Evaluation Metrics
The system includes a comprehensive evaluation framework for all probability-based trading strategies:

```python
# Core evaluation components
from sentio_trainer.utils.strategy_evaluation import StrategyEvaluator

# Signal quality metrics
evaluator = StrategyEvaluator("StrategyName")
results = evaluator.evaluate_strategy_signal(predictions, actual_returns)
```

#### Evaluation Dimensions
1. **Signal Quality**: Probability range, mean, standard deviation, signal strength
2. **Calibration**: How well probabilities match actual frequencies
3. **Information Content**: Log loss, Brier score, AUC, information ratio
4. **Trading Performance**: Accuracy, precision, recall, F1 score, Sharpe ratio
5. **Overall Assessment**: Weighted score with rating (Excellent/Good/Fair/Poor)

#### Calibration Analysis
- **Calibration Error**: < 0.05 = Excellent, 0.05-0.10 = Good, > 0.10 = Poor
- **Information Ratio**: > 0.2 = High, 0.1-0.2 = Moderate, < 0.1 = Low
- **AUC Score**: 0.5 = Random, 1.0 = Perfect prediction

### 2. Backend Trading System Evaluation

#### Complete Pipeline Evaluation
The system evaluates the complete trading pipeline beyond signal quality:

```
Signal (Probability) → Router → Sizer → Runner → Actual PnL
     ↑                    ↑        ↑        ↑
  Signal              Portfolio    Risk     Execution
  Evaluation          Management   Management Management
```

#### Backend Components Evaluation
1. **Router Performance**: Instrument selection effectiveness, PnL per instrument
2. **Sizer Performance**: Position sizing optimization, risk-adjusted returns
3. **Runner Performance**: Execution costs, slippage, timing efficiency
4. **Signal Effectiveness**: Correlation between signal strength and actual profits

#### Performance Metrics
- **Overall Performance**: Total PnL, win rate, Sharpe ratio, max drawdown, Calmar ratio
- **Execution Quality**: Commission rate, slippage rate, execution cost per trade
- **Risk Management**: Value at Risk (VaR), expected shortfall, turnover rate
- **Signal Correlation**: Signal-to-PnL correlation, signal effectiveness

### 3. Virtual Market Testing (VMTest)

#### VMTest Overview
VMTest provides comprehensive virtual market simulation for strategy testing using multiple data generation approaches. The system supports both synthetic data generation and integration with the MarS (Market Simulation Engine) for realistic market microstructure modeling.

#### VMTest Architecture
```
Data Generation Layer → Strategy Execution → Monte Carlo Testing → Results
         ↑                    ↑                    ↑              ↑
    MarS Engine          SentioStrategy       Statistical    Performance
    Fast Historical      Real Runner          Analysis        Metrics
    Synthetic Data       Integration
```

#### Data Generation Approaches

##### 1. Fast Historical Bridge
- **Purpose**: Instant generation of realistic market data based on historical patterns
- **Speed**: < 1 second per simulation
- **Realism**: Uses actual historical QQQ patterns for volatility, volume, and intraday behavior
- **Time Handling**: Generates timestamps from today's market open (9:30 AM ET)
- **Pattern Analysis**: Extracts mean return, volatility, volume patterns, and hourly multipliers

##### 2. MarS Integration
- **Purpose**: AI-powered market simulation with realistic microstructure
- **Features**: Order-level simulation, market maker behavior, realistic spreads
- **Historical Context**: Uses HistoricalContextAgent for realistic starting conditions
- **AI Continuation**: Optional MarS AI for sophisticated market behavior
- **Performance**: High-quality simulation with realistic market dynamics

##### 3. Synthetic Data Generation
- **Purpose**: Basic synthetic data for rapid testing
- **Speed**: Very fast generation
- **Use Case**: Quick validation and debugging
- **Limitations**: Less realistic than historical or MarS data

#### VMTest CLI Commands

##### Basic VM Test (Fast Historical)
```bash
# Standard VM test with fast historical data
./build/sentio_cli vmtest IRE QQQ --days 30 --simulations 100

# Extended testing with custom parameters
./build/sentio_cli vmtest IRE QQQ --days 70 --simulations 100 --params '{"buy_hi": 0.6, "sell_lo": 0.4}'

# Custom historical data source
./build/sentio_cli vmtest IRE QQQ --days 14 --simulations 50 --historical-data data/equities/QQQ_RTH_NH.csv
```

##### MarS-Powered VM Test
```bash
# MarS simulation with AI
./build/sentio_cli marstest IRE QQQ --days 7 --simulations 20 --regime normal --use-mars-ai

# MarS with historical context
./build/sentio_cli marstest TFA QQQ --days 14 --simulations 10 --regime volatile --historical-data data/equities/QQQ_RTH_NH.csv
```

##### Fast Historical Test
```bash
# Direct fast historical test
./build/sentio_cli fasttest IRE QQQ --historical-data data/equities/QQQ_RTH_NH.csv --continuation-minutes 1440 --simulations 50
```

#### VMTest Parameters

##### Common Parameters
- **--days <n>**: Number of days to simulate (default: 30)
- **--hours <n>**: Number of hours to simulate (alternative to days)
- **--simulations <n>**: Number of Monte Carlo simulations (default: 100)
- **--params <json>**: Strategy parameters as JSON string
- **--historical-data <file>**: Historical data file for pattern analysis

##### MarS-Specific Parameters
- **--regime <type>**: Market regime (normal, bull_trending, bear_trending, sideways_low_vol, volatile)
- **--use-mars-ai**: Enable MarS AI for sophisticated market behavior
- **--continuation-minutes <n>**: Minutes to simulate beyond historical data

#### VMTest Output Metrics
- **Return Statistics**: Mean, median, standard deviation, min/max returns
- **Confidence Intervals**: 5th, 25th, 75th, 95th percentiles
- **Probability Analysis**: Probability of profit across simulations
- **Performance Metrics**: 
  - Mean Sharpe Ratio
  - Mean MPR (Monthly Projected Return)
  - Mean Daily Trades
- **Signal Diagnostics**: Signal generation and validation metrics
- **Data Quality**: Generated data statistics (price range, volume range, time range)

#### MarS Integration Details

##### Historical Context Agent
```python
class HistoricalContextAgent:
    """Provides realistic starting conditions for MarS simulations"""
    
    def __init__(self, symbol, historical_bars, continuation_minutes):
        self.symbol = symbol
        self.historical_bars = historical_bars
        self.continuation_minutes = continuation_minutes
        
        # Analyze historical patterns
        self.mean_return, self.volatility, self.mean_volume = \
            self._analyze_historical_patterns(historical_bars)
    
    def generate_continuation_orders(self, time):
        """Generate realistic orders based on historical patterns"""
        # Use historical volatility and volume patterns
        # Generate market-making orders with realistic spreads
        # Transition smoothly from historical to synthetic data
```

##### Fast Historical Bridge
```python
def generate_realistic_bars(patterns, start_price, duration_minutes):
    """Generate realistic bars instantly using historical patterns"""
    
    # Use today's market open time
    market_open = get_today_market_open()
    
    # Generate bars with historical patterns
    for i in range(num_bars):
        # Apply hourly volume and volatility multipliers
        volume_multiplier = patterns.hourly_volume_multipliers[hour]
        volatility_multiplier = patterns.hourly_volatility_multipliers[hour]
        
        # Generate realistic price movement
        price_change = np.random.normal(patterns.mean_return, 
                                      patterns.volatility * volatility_multiplier)
        
        # Generate volume with time-of-day patterns
        volume = int(patterns.mean_volume * volume_multiplier * random_factor)
        
        # Create bar with realistic OHLC relationships
        bar = create_bar_with_realistic_ohlc(current_price, price_change, volume)
```

#### VMTest Performance Characteristics

##### Speed Comparison
- **Fast Historical**: < 1 second per simulation
- **MarS (No AI)**: 10-30 seconds per simulation
- **MarS (With AI)**: 30-120 seconds per simulation
- **Synthetic Data**: < 0.1 seconds per simulation

##### Data Quality Ranking
1. **MarS with AI**: Highest realism, sophisticated market behavior
2. **MarS without AI**: High realism, basic market microstructure
3. **Fast Historical**: Good realism, instant generation
4. **Synthetic Data**: Basic realism, fastest generation

##### Use Case Recommendations
- **Development/Testing**: Fast Historical or Synthetic Data
- **Strategy Validation**: MarS without AI
- **Production Simulation**: MarS with AI
- **Quick Diagnostics**: Synthetic Data

### 4. Evaluation Integration

#### Automatic Evaluation During Training
```python
# TFA trainer includes automatic evaluation
python train_models.py --config configs/tfa.yaml
# Output: Comprehensive evaluation metrics + results saved to evaluation_results.json
```

#### Standalone Evaluation Tools
```python
# CLI tool for strategy evaluation
python sentio_trainer/evaluate_strategies.py single --data strategy_data.json --name "MyStrategy"
python sentio_trainer/evaluate_strategies.py compare --data-files strategy1.json strategy2.json
```

#### Programmatic Evaluation
```python
# Quick evaluation
from sentio_trainer.utils.strategy_evaluation import quick_evaluate
results = quick_evaluate(predictions, actual_returns, "StrategyName")

# Detailed evaluation
evaluator = StrategyEvaluator("StrategyName")
results = evaluator.evaluate_strategy_signal(predictions, actual_returns, verbose=True)
evaluator.save_results("results.json")
```

### 5. Evaluation Data Requirements

#### Input Format
- **Predictions**: Raw model outputs (logits), converted to probabilities
- **Actual Returns**: Binary values (1 = price up, 0 = price down)
- **Data Sources**: JSON, NPZ, or CSV formats supported

#### Evaluation Standards
- **Sufficient Data**: Minimum 1000 samples for reliable metrics
- **Balanced Classes**: Avoid extreme class imbalance
- **Clean Data**: Remove outliers and invalid predictions
- **Consistent Time Periods**: Compare strategies on identical data

## Testing and Validation

### 1. Unit Testing
- **Strategy Testing**: Individual strategy signal generation
- **Component Testing**: Router, sizer, and audit components
- **Integration Testing**: End-to-end system testing

### 2. Backtesting Framework
- **Temporal Performance Analysis (TPA)**: Comprehensive backtesting with multiple time periods
- **Performance Metrics**: Sharpe ratio, maximum drawdown, monthly returns
- **Risk Analysis**: Position sizing, leverage analysis, correlation studies

### 3. Live Trading Validation
- **Paper Trading**: Simulated trading with real market data
- **Gradual Deployment**: Phased rollout with position limits
- **Performance Monitoring**: Real-time performance tracking

## Deployment Architecture

### 1. Development Environment
- **Local Development**: macOS development environment
- **Version Control**: Git with GitHub integration
- **Build System**: Makefile-based build system
- **Testing**: Automated testing pipeline

### 2. Production Environment
- **Cloud Deployment**: AWS/GCP cloud infrastructure
- **Containerization**: Docker containers for deployment
- **Monitoring**: Real-time performance monitoring
- **Alerting**: Automated alerting for system issues

### 3. Data Pipeline
- **Real-time Data**: Polygon.io API integration
- **Historical Data**: Comprehensive historical data storage
- **Data Validation**: Automated data quality checks
- **Backup Systems**: Redundant data storage

## Diagnostic Strategy Framework

### 1. Diagnostic Strategy Requirements

The system includes a comprehensive diagnostic strategy framework designed to validate system components and provide baseline performance metrics.

#### Diagnostic Strategy Specifications
- **Purpose**: System validation and infrastructure testing
- **Signal Generation**: RSI-based with aggressive thresholds
- **Frequency**: Minimum 100 signals per day
- **Leverage Support**: QQQ (40%), TQQQ (30%), SQQQ (30%)
- **Objective**: System diagnostics, NOT profit optimization

#### Technical Implementation
```cpp
class DiagnosticStrategy : public BaseStrategy {
private:
    std::vector<double> price_history_;
    int rsi_period_;
    int signal_count_;
    double last_rsi_;
    std::vector<std::string> leverage_symbols_;
    int current_symbol_index_;
    
public:
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override {
        // Calculate RSI from price history
        double rsi = calculate_rsi(price_history_, rsi_period_);
        
        // Generate signals based on RSI thresholds
        if (rsi < 30) return 0.8;      // Strong buy
        if (rsi > 70) return 0.2;      // Strong sell
        if (rsi < 50) return 0.6;      // Moderate buy
        if (rsi > 50) return 0.4;      // Moderate sell
        return 0.5;                    // Neutral
    }
    
    std::vector<AllocationDecision> get_allocation_decisions(...) override {
        // Rotate through leverage symbols
        std::string symbol = select_leverage_ticker();
        double weight = (probability - 0.5) * 2.0; // Convert to -1 to 1
        
        return {{symbol, weight, confidence, "Diagnostic signal"}};
    }
};
```

#### Validation Criteria
- **Signal Generation**: ≥100 signals per day
- **Signal Distribution**: 40% QQQ, 30% TQQQ, 30% SQQQ
- **System Integration**: Successful VM test execution
- **Infrastructure Validation**: Confirms Runner, Router, Sizer functionality

### 2. System Validation Workflow

#### Diagnostic Testing Pipeline
```
Diagnostic Strategy → VM Test → System Validation → Performance Baseline
        ↑              ↑              ↑                    ↑
    RSI Signals    MarS/Fast      Component         Expected
    Generation     Historical     Validation        Metrics
```

#### Validation Steps
1. **Signal Generation Test**: Verify diagnostic strategy generates expected signal frequency
2. **VM Test Integration**: Confirm VM test infrastructure processes signals correctly
3. **Runner Integration**: Validate signal execution and trade generation
4. **Router Validation**: Test multi-symbol routing (QQQ, TQQQ, SQQQ)
5. **Sizer Validation**: Confirm position sizing calculations
6. **Performance Baseline**: Establish expected performance metrics

#### Diagnostic Value
- **Infrastructure Verification**: Confirms all system components are functional
- **Performance Baseline**: Establishes expected signal generation rates
- **Comparison Framework**: Enables comparison with other strategies
- **Debugging Tool**: Helps isolate system vs. strategy issues

## Future Enhancements

### 1. Advanced ML Techniques
- **Reinforcement Learning**: Advanced RL algorithms for strategy optimization
- **Ensemble Methods**: Combining multiple ML models
- **Online Learning**: Continuous model updates with new data
- **Transfer Learning**: Leveraging pre-trained models

### 2. Risk Management
- **Dynamic Risk Controls**: Adaptive risk management based on market conditions
- **Portfolio Optimization**: Multi-strategy portfolio optimization
- **Stress Testing**: Comprehensive stress testing framework
- **Regulatory Compliance**: Automated compliance monitoring

### 3. Performance Optimization
- **GPU Acceleration**: CUDA-based numerical computations
- **Distributed Computing**: Multi-node computation for large-scale analysis
- **Real-time Processing**: Sub-millisecond signal processing
- **Advanced Caching**: Intelligent caching strategies

## Conclusion

The Sentio C++ architecture provides a robust, scalable, and profit-maximizing framework for quantitative trading. The strategy-agnostic design ensures that new strategies can be easily integrated without architectural changes, while the comprehensive audit and diagnostics systems provide full visibility into system performance. The ML integration enables sophisticated signal generation, and the leverage trading system maximizes capital efficiency for optimal returns.

The system is designed to continuously evolve and improve, with a focus on maximizing profit through advanced signal generation, dynamic allocation, and comprehensive risk management. The architecture supports both research and production environments, with robust testing and validation frameworks ensuring reliable performance in live trading scenarios.
