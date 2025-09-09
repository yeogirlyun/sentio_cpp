# TFA Feature Dimension Mismatch Bug Report

**Date**: January 7, 2025  
**Status**: 🔴 **CRITICAL** - Prevents TFA strategy from generating signals  
**Impact**: Complete failure of TFA strategy signal generation  
**Priority**: P0 - System Breaking  

## Executive Summary

The TFA (Transformer Financial Alpha) strategy is experiencing a critical dimension mismatch bug that prevents signal generation. While the FeatureCache successfully loads 56 features from the pre-computed CSV file, the FeatureFeeder extracts only 55 features, causing a mismatch with the trained TFA model which expects exactly 56 features.

**Key Symptoms:**
- FeatureCache: ✅ Reports "Successfully loaded 1564 bars with 56 features each"
- FeatureFeeder: ❌ Extracts only 55 features (`features.size()=55`)
- TFA Model: ❌ Expects 56 features (`expected_feat_dim=56`)
- Result: **Zero signals generated** due to dimension mismatch

## Root Cause Analysis

### Primary Issue
The bug exists in the **feature extraction pipeline** between the FeatureCache and FeatureFeeder. Despite the FeatureCache correctly loading 56 features from the CSV file, when `get_features()` is called, it returns only 55 features.

### Evidence from Debug Logs
```
FeatureCache: Loaded 56 feature names
FeatureCache: Successfully loaded 1564 bars with 56 features each
[DIAG] FeatureFeeder extract: call=1 features.size()=55 current_index=0
[DIAG] TFA set_raw_features: call=1 raw.size()=55 buffer_size=1 expected_feat_dim=56
```

### Feature Cache CSV Format
The generated CSV has the correct format:
- **Header**: `bar_index,timestamp,close,logret,ema_2,ema_3,...,rsi_short_2` (58 total columns)
- **Data**: 1564 rows × 58 columns (bar_index + timestamp + 56 features)
- **Expected**: 56 features after skipping bar_index and timestamp

### Suspected Bug Location
The issue is likely in `src/feature_cache.cpp` in either:
1. **Header parsing logic** (lines 26-34) - may be incorrectly counting feature names
2. **Data parsing logic** (lines 56-58) - may be dropping a feature during CSV parsing
3. **Feature validation logic** (lines 61-65) - may have an off-by-one error

## Detailed Investigation

### CSV File Analysis
```bash
# Header analysis
head -1 data/QQQ_RTH_features.csv | tr ',' '\n' | wc -l
# Output: 58 (correct: bar_index + timestamp + 56 features)

# Data row analysis  
head -2 data/QQQ_RTH_features.csv | tail -1 | tr ',' '\n' | wc -l
# Output: 58 (matches header)
```

### Feature Names from Cache
The FeatureCache reports loading **56 feature names** correctly:
```
close, logret, ema_2, ema_3, ema_5, ema_8, ema_10, ema_12, ema_15, ema_20,
ema_25, ema_26, ema_30, ema_35, ema_40, ema_45, ema_50, ema_55, ema_60, 
ema_70, ema_80, ema_90, ema_100, ema_120, ema_150, ema_200, rsi_2, rsi_3,
rsi_5, rsi_7, rsi_9, rsi_14, rsi_21, rsi_25, rsi_30, zlog_5, zlog_10, 
zlog_15, zlog_20, zlog_25, zlog_30, zlog_40, zlog_50, zlog_60, zlog_80,
zlog_100, ema_short_1, ema_short_2, ema_short_3, ema_short_4, ema_short_5,
ema_short_6, ema_short_7, ema_short_8, rsi_short_1, rsi_short_2
```

### Model Expectations
The TFA model metadata confirms it expects **56 features**:
```json
"expects": {
  "input_dim": 56,
  "feature_names": [...], // 56 features listed
  "emit_from": 64,
  "pad_value": 0.0
}
```

## Impact Assessment

### Immediate Impact
- **TFA Strategy**: Complete failure - generates 0 signals across all quarters
- **Trading Performance**: 0% returns, 0 trades/day
- **System Status**: TFA marked as "NOT READY" for any trading

### Downstream Effects
- **Feature Cache Performance**: Despite 100-1000x speedup being available, dimension mismatch prevents utilization
- **ML Pipeline**: Breaks the entire Python training → C++ inference pipeline for TFA
- **Production Readiness**: Blocks TFA from live trading deployment

## Detailed Source Code Analysis

### 1. Feature Cache Loading (`src/feature_cache.cpp`)

**Header Parsing Logic:**
```cpp
// Lines 26-34: Parse header to get feature names
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
```

**Data Parsing Logic:**
```cpp
// Lines 42-58: Parse data line
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
```

**Validation Logic:**
```cpp
// Lines 61-65: Verify feature count
if (features.size() != feature_names_.size()) {
    std::cerr << "Warning: Bar " << bar_index << " has " << features.size() 
              << " features, expected " << feature_names_.size() << std::endl;
    continue;  // ⚠️ POTENTIAL BUG: Skips entire bar if mismatch
}
```

### 2. Feature Feeder Integration (`src/feature_feeder.cpp`)

**Feature Extraction Call:**
```cpp
// Lines 182-187: Use cached features if available
std::vector<double> features;
if (use_cached_features_ && feature_cache_ && feature_cache_->has_features(current_index)) {
    features = feature_cache_->get_features(current_index);  // ⚠️ Returns 55 instead of 56
} else {
    features = data.calculator->calculate_all_features(bars, current_index);
}
```

### 3. TFA Strategy Reception (`src/strategy_tfa.cpp`)

**Feature Reception:**
```cpp
// Lines 38-50: Receive features from FeatureFeeder
void TFAStrategy::set_raw_features(const std::vector<double>& raw) {
    static int feature_calls = 0;
    feature_calls++;
    
    feature_buffer_.push_back(raw);
    
    if (feature_calls % 1000 == 0 || feature_calls <= 10) {
        std::cout << "[DIAG] TFA set_raw_features: call=" << feature_calls 
                  << " raw.size()=" << raw.size()           // Shows 55
                  << " buffer_size=" << feature_buffer_.size()
                  << " expected_feat_dim=" << window_.feat_dim() << std::endl;  // Shows 56
    }
}
```

## Hypothesis & Testing Plan

### Primary Hypothesis
The bug is in the CSV parsing logic in `feature_cache.cpp`. Possible causes:

1. **Off-by-one error** in the `while (std::getline(ss, cell, ','))` loop
2. **Trailing comma issue** causing one feature to be missed
3. **Empty cell handling** dropping a feature silently
4. **Feature validation** incorrectly rejecting valid bars

### Testing Strategy
1. **Add debug output** to `feature_cache.cpp` to show exact feature counts during parsing
2. **Inspect raw CSV data** for trailing commas or empty cells
3. **Verify feature extraction** by printing all 58 columns during parsing
4. **Compare header vs data** feature counts line by line

### Proposed Fix Approach
1. **Immediate**: Add comprehensive debugging to `get_features()` method
2. **Investigate**: Check for parsing edge cases (empty cells, trailing commas)
3. **Validate**: Ensure header parsing matches data parsing exactly
4. **Test**: Verify feature count consistency across all 1564 bars

## Environment & Configuration

### System Configuration
- **OS**: macOS 14.6.0 (Darwin)
- **Compiler**: Clang with C++20
- **Build Type**: Release (-O3 -DNDEBUG)
- **Cache File**: `data/QQQ_RTH_features.csv` (941KB, 1564 rows × 58 columns)

### Feature Specification
- **Source**: `configs/features/feature_spec_55_minimal.json`
- **Operations**: IDENT, LOGRET, EMA, RSI, ZWIN (pybind-supported operations only)
- **Sources**: close only (pybind limitation)
- **Generated Features**: 56 technical indicators

### Model Configuration
- **Framework**: TorchScript (.pt)
- **Input Dimension**: 56 features
- **Sequence Length**: 64 timesteps
- **Architecture**: MLP with feature scaling

## Temporary Workarounds

### None Available
- **Feature padding**: Not viable (would corrupt model input)
- **Dimension reduction**: Would require model retraining
- **Real-time calculation**: Defeats purpose of feature caching

### Alternative Approaches
1. **Retrain model** with 55 features (significant effort)
2. **Fix feature generation** to produce exactly 55 features (regression)
3. **Fix extraction bug** to properly return 56 features (preferred)

## Next Steps

### Immediate Actions (P0)
1. **Debug feature extraction** by adding detailed logging to `feature_cache.cpp`
2. **Identify exact location** where 56 features become 55 features
3. **Fix the parsing bug** to ensure consistent feature dimensions

### Validation Steps
1. **Verify feature count** matches between cache loading and extraction
2. **Test TFA signal generation** with corrected feature dimensions
3. **Confirm model predictions** work correctly with proper input

### Long-term Prevention
1. **Add unit tests** for feature cache loading/extraction
2. **Implement dimension validation** at multiple pipeline stages
3. **Create integration tests** for ML feature pipeline

---

## Appendix: Complete Source Code Modules

### A1. Feature Cache Implementation (`src/feature_cache.cpp`)

```cpp
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
            std::cerr << "Warning: Bar " << bar_index << " has " << features.size() 
                      << " features, expected " << feature_names_.size() << std::endl;
            continue;
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
        return it->second;
    }
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

### A2. Feature Cache Header (`include/sentio/feature_cache.hpp`)

```cpp
#pragma once

#include <vector>
#include <string>
#include <unordered_map>

namespace sentio {

class FeatureCache {
public:
    FeatureCache() = default;
    ~FeatureCache() = default;
    
    // Load features from CSV file
    bool load_from_csv(const std::string& feature_file_path);
    
    // Get features for a specific bar index
    std::vector<double> get_features(int bar_index) const;
    
    // Check if features exist for a bar index
    bool has_features(int bar_index) const;
    
    // Get total number of bars loaded
    size_t get_bar_count() const;
    
    // Get recommended starting bar for analysis
    int get_recommended_start_bar() const;
    
    // Get feature names
    const std::vector<std::string>& get_feature_names() const;
    
private:
    std::unordered_map<int, std::vector<double>> features_by_bar_;
    std::vector<std::string> feature_names_;
    size_t total_bars_ = 0;
    int recommended_start_bar_ = 300;  // Conservative default for technical indicators
};

} // namespace sentio
```

### A3. Feature Feeder Implementation (Relevant Section)

```cpp
// From src/feature_feeder.cpp lines 159-209
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
            features = feature_cache_->get_features(current_index);  // ⚠️ BUG: Returns 55 instead of 56
        } else {
            // Calculate features using full bar history up to current_index
            features = data.calculator->calculate_all_features(bars, current_index);
        }
        
        // Normalize features
        if (data.normalizer && !features.empty()) {
            features = data.normalizer->normalize_features(features);
        }
        
        // Validate features
        if (!validate_features(features, strategy_name)) {
            return {};
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
```

### A4. TFA Strategy Feature Reception

```cpp
// From src/strategy_tfa.cpp lines 38-60
void TFAStrategy::set_raw_features(const std::vector<double>& raw){
  static int feature_calls = 0;
  feature_calls++;
  
  // Store features in buffer for batch processing
  feature_buffer_.push_back(raw);
  
  // Detailed diagnostics
  if (feature_calls % 1000 == 0 || feature_calls <= 10) {
    std::cout << "[DIAG] TFA set_raw_features: call=" << feature_calls 
              << " raw.size()=" << raw.size() 
              << " buffer_size=" << feature_buffer_.size()
              << " expected_feat_dim=" << window_.feat_dim() << std::endl;
    
    if (raw.size() != (size_t)window_.feat_dim()) {
      std::cout << "[ERROR] Feature dimension mismatch! Expected: " 
                << window_.feat_dim() << ", Got: " << raw.size() << std::endl;
    }
  }
}
```

### A5. Feature Cache Generator (`tools/generate_feature_cache.py`)

```python
#!/usr/bin/env python3
import argparse, json, hashlib, pathlib, numpy as np
import sentio_features as sf  # pybind bridge to your C++ FeatureBuilder

def spec_with_hash(p):
    raw = pathlib.Path(p).read_bytes()
    spec = json.loads(raw)
    spec["content_hash"] = "sha256:" + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return spec

def load_bars(csv_path):
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    
    # Handle different CSV column formats
    if "ts_nyt_epoch" in arr.dtype.names:
        ts = arr["ts_nyt_epoch"].astype(np.int64)
    elif "ts" in arr.dtype.names:
        ts = arr["ts"].astype(np.int64)
    else:
        raise ValueError(f"No timestamp column found. Available columns: {arr.dtype.names}")
    
    return (
        ts,
        arr["open"].astype(np.float64),
        arr["high"].astype(np.float64),
        arr["low"].astype(np.float64),
        arr["close"].astype(np.float64),
        arr["volume"].astype(np.float64),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="Base ticker, e.g. QQQ")
    ap.add_argument("--bars", required=True, help="CSV with columns: ts,open,high,low,close,volume")
    ap.add_argument("--spec", required=True, help="feature_spec_55.json")
    ap.add_argument("--outdir", default="data", help="output dir for features files")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    spec = spec_with_hash(args.spec); spec_json = json.dumps(spec, sort_keys=True)
    ts, o, h, l, c, v = load_bars(args.bars)

    print(f"[FeatureCache] Building features for {args.symbol} with {len(ts)} bars...")
    # Build features via C++ → parity with runtime
    X = sf.build_features_from_spec(args.symbol, ts, o, h, l, c, v, spec_json).astype(np.float32)
    N, F = X.shape
    names = [f.get("name", f'{f["op"]}_{f.get("source","")}_{f.get("window","")}_{f.get("k","")}') for f in spec["features"]]

    print(f"[FeatureCache] Generated features: {N} rows x {F} features")
    print(f"[FeatureCache] Feature stats: min={X.min():.6f}, max={X.max():.6f}, mean={X.mean():.6f}, std={X.std():.6f}")

    # Save CSV (bar_index + timestamp + features) — header: bar_index,timestamp,<names...>
    csv_path = outdir / f"{args.symbol}_RTH_features.csv"
    header = "bar_index,timestamp," + ",".join(names)
    M = np.empty((N, F+2), dtype=np.float32)
    M[:, 0] = np.arange(N).astype(np.float64)  # bar_index
    M[:, 1] = ts.astype(np.float64)  # timestamp
    M[:, 2:] = X  # features
    np.savetxt(csv_path, M, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"✅ CSV saved: {csv_path}")

    # Save NPY for fast memmap reuse (just the features, not ts)
    npy_path = outdir / f"{args.symbol}_RTH_features.npy"
    np.save(npy_path, X, allow_pickle=False)
    print(f"✅ NPY saved: {npy_path}")

    # Save meta to help consumers validate shape/order
    meta = {
        "schema_version":"1.0",
        "symbol": args.symbol,
        "rows": int(N), "cols": int(F),
        "feature_names": names,
        "spec_hash": spec["content_hash"],
        "emit_from": int(spec["alignment_policy"]["emit_from_index"])
    }
    json.dump(meta, open(outdir / f"{args.symbol}_RTH_features.meta.json","w"), indent=2)
    print(f"✅ META saved: {outdir / (args.symbol + '_RTH_features.meta.json')}")

if __name__ == "__main__":
    main()
```

### A6. Feature Specification (`configs/features/feature_spec_55_minimal.json`)

```json
{
  "schema_version": "1.0",
  "features": [
    {"name":"close",   "op":"IDENT",  "source":"close"},
    {"name":"logret",  "op":"LOGRET", "source":"close"},
    {"name":"ema_2",   "op":"EMA",    "source":"close","window":2},
    {"name":"ema_3",   "op":"EMA",    "source":"close","window":3},
    {"name":"ema_5",   "op":"EMA",    "source":"close","window":5},
    {"name":"ema_8",   "op":"EMA",    "source":"close","window":8},
    {"name":"ema_10",  "op":"EMA",    "source":"close","window":10},
    {"name":"ema_12",  "op":"EMA",    "source":"close","window":12},
    {"name":"ema_15",  "op":"EMA",    "source":"close","window":15},
    {"name":"ema_20",  "op":"EMA",    "source":"close","window":20},
    {"name":"ema_25",  "op":"EMA",    "source":"close","window":25},
    {"name":"ema_26",  "op":"EMA",    "source":"close","window":26},
    {"name":"ema_30",  "op":"EMA",    "source":"close","window":30},
    {"name":"ema_35",  "op":"EMA",    "source":"close","window":35},
    {"name":"ema_40",  "op":"EMA",    "source":"close","window":40},
    {"name":"ema_45",  "op":"EMA",    "source":"close","window":45},
    {"name":"ema_50",  "op":"EMA",    "source":"close","window":50},
    {"name":"ema_55",  "op":"EMA",    "source":"close","window":55},
    {"name":"ema_60",  "op":"EMA",    "source":"close","window":60},
    {"name":"ema_70",  "op":"EMA",    "source":"close","window":70},
    {"name":"ema_80",  "op":"EMA",    "source":"close","window":80},
    {"name":"ema_90",  "op":"EMA",    "source":"close","window":90},
    {"name":"ema_100", "op":"EMA",    "source":"close","window":100},
    {"name":"ema_120", "op":"EMA",    "source":"close","window":120},
    {"name":"ema_150", "op":"EMA",    "source":"close","window":150},
    {"name":"ema_200", "op":"EMA",    "source":"close","window":200},
    {"name":"rsi_2",   "op":"RSI",    "source":"close","window":2},
    {"name":"rsi_3",   "op":"RSI",    "source":"close","window":3},
    {"name":"rsi_5",   "op":"RSI",    "source":"close","window":5},
    {"name":"rsi_7",   "op":"RSI",    "source":"close","window":7},
    {"name":"rsi_9",   "op":"RSI",    "source":"close","window":9},
    {"name":"rsi_14",  "op":"RSI",    "source":"close","window":14},
    {"name":"rsi_21",  "op":"RSI",    "source":"close","window":21},
    {"name":"rsi_25",  "op":"RSI",    "source":"close","window":25},
    {"name":"rsi_30",  "op":"RSI",    "source":"close","window":30},
    {"name":"zlog_5",  "op":"ZWIN",   "source":"logret","window":5},
    {"name":"zlog_10", "op":"ZWIN",   "source":"logret","window":10},
    {"name":"zlog_15", "op":"ZWIN",   "source":"logret","window":15},
    {"name":"zlog_20", "op":"ZWIN",   "source":"logret","window":20},
    {"name":"zlog_25", "op":"ZWIN",   "source":"logret","window":25},
    {"name":"zlog_30", "op":"ZWIN",   "source":"logret","window":30},
    {"name":"zlog_40", "op":"ZWIN",   "source":"logret","window":40},
    {"name":"zlog_50", "op":"ZWIN",   "source":"logret","window":50},
    {"name":"zlog_60", "op":"ZWIN",   "source":"logret","window":60},
    {"name":"zlog_80", "op":"ZWIN",   "source":"logret","window":80},
    {"name":"zlog_100","op":"ZWIN",   "source":"logret","window":100},
    {"name":"ema_short_1", "op":"EMA", "source":"close","window":4},
    {"name":"ema_short_2", "op":"EMA", "source":"close","window":6},
    {"name":"ema_short_3", "op":"EMA", "source":"close","window":7},
    {"name":"ema_short_4", "op":"EMA", "source":"close","window":9},
    {"name":"ema_short_5", "op":"EMA", "source":"close","window":11},
    {"name":"ema_short_6", "op":"EMA", "source":"close","window":13},
    {"name":"ema_short_7", "op":"EMA", "source":"close","window":14},
    {"name":"ema_short_8", "op":"EMA", "source":"close","window":16},
    {"name":"rsi_short_1", "op":"RSI", "source":"close","window":4},
    {"name":"rsi_short_2", "op":"RSI", "source":"close","window":6}
  ],
  "alignment_policy": { "emit_from_index": 64, "pad_value": 0.0 },
  "dtype": "float32"
}
```

### A7. TFA Model Metadata (`artifacts/TFA/v1/metadata.json`)

```json
{
  "schema_version": "1.0",
  "saved_at": 1757231725,
  "framework": "torchscript",
  "expects": {
    "input_dim": 56,
    "feature_names": [
      "close", "logret", "ema_2", "ema_3", "ema_5", "ema_8", "ema_10", "ema_12",
      "ema_15", "ema_20", "ema_25", "ema_26", "ema_30", "ema_35", "ema_40", 
      "ema_45", "ema_50", "ema_55", "ema_60", "ema_70", "ema_80", "ema_90",
      "ema_100", "ema_120", "ema_150", "ema_200", "rsi_2", "rsi_3", "rsi_5",
      "rsi_7", "rsi_9", "rsi_14", "rsi_21", "rsi_25", "rsi_30", "zlog_5",
      "zlog_10", "zlog_15", "zlog_20", "zlog_25", "zlog_30", "zlog_40",
      "zlog_50", "zlog_60", "zlog_80", "zlog_100", "ema_short_1", "ema_short_2",
      "ema_short_3", "ema_short_4", "ema_short_5", "ema_short_6", "ema_short_7",
      "ema_short_8", "rsi_short_1", "rsi_short_2"
    ],
    "spec_hash": "sha256:d9399c3155f1f4c16d7d8a7a1f73aea9183f7113d043fe01b94e9d08664a5ad5",
    "emit_from": 64,
    "pad_value": 0.0,
    "dtype": "float32",
    "output": "logit"
  }
}
```

---

**Report Generated**: January 7, 2025  
**Next Update**: Upon bug resolution  
**Assigned**: Development Team  
**Priority**: P0 - Critical System Failure  

