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

  // PERFORMANCE NOTE: Current implementation processes features column-by-column
  // which causes cache misses due to row-major memory layout. For optimal performance,
  // consider refactoring to process row-by-row: calculate all features for row r,
  // then write them contiguously to M.data[r * F + c] before moving to next row.
  // This would require stateful indicator objects (EMA_Calculator, RSI_Calculator, etc.)
  // and significant refactoring of the op_* functions.
  
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
