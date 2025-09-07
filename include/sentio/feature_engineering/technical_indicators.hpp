#pragma once
#include "sentio/core.hpp"
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace sentio {
namespace feature_engineering {

// Price-based features
struct PriceFeatures {
    double ret_1m{0.0}, ret_5m{0.0}, ret_15m{0.0}, ret_30m{0.0}, ret_1h{0.0};
    double momentum_5{0.0}, momentum_10{0.0}, momentum_20{0.0};
    double volatility_10{0.0}, volatility_20{0.0}, volatility_30{0.0};
    double atr_14{0.0}, atr_21{0.0};
    double parkinson_vol{0.0}, garman_klass_vol{0.0};
};

// Technical analysis features
struct TechnicalFeatures {
    double rsi_14{0.0}, rsi_21{0.0}, rsi_30{0.0};
    double sma_5{0.0}, sma_10{0.0}, sma_20{0.0}, sma_50{0.0}, sma_200{0.0};
    double ema_5{0.0}, ema_10{0.0}, ema_20{0.0}, ema_50{0.0}, ema_200{0.0};
    double bb_upper_20{0.0}, bb_middle_20{0.0}, bb_lower_20{0.0};
    double bb_upper_50{0.0}, bb_middle_50{0.0}, bb_lower_50{0.0};
    double macd_line{0.0}, macd_signal{0.0}, macd_histogram{0.0};
    double stoch_k{0.0}, stoch_d{0.0};
    double williams_r{0.0};
    double cci_20{0.0};
    double adx_14{0.0};
};

// Volume features
struct VolumeFeatures {
    double volume_sma_10{0.0}, volume_sma_20{0.0}, volume_sma_50{0.0};
    double volume_roc{0.0};
    double obv{0.0};
    double vpt{0.0};
    double ad_line{0.0};
    double mfi_14{0.0};
};

// Market microstructure features
struct MicrostructureFeatures {
    double spread_bp{0.0};
    double price_impact{0.0};
    double order_flow_imbalance{0.0};
    double market_depth{0.0};
    double bid_ask_ratio{0.0};
};

// Main feature calculator
class TechnicalIndicatorCalculator {
public:
    TechnicalIndicatorCalculator();
    
    // Core calculation methods
    PriceFeatures calculate_price_features(const std::vector<Bar>& bars, int current_index);
    TechnicalFeatures calculate_technical_features(const std::vector<Bar>& bars, int current_index);
    VolumeFeatures calculate_volume_features(const std::vector<Bar>& bars, int current_index);
    MicrostructureFeatures calculate_microstructure_features(const std::vector<Bar>& bars, int current_index);
    
    // Combined feature vector
    std::vector<double> calculate_all_features(const std::vector<Bar>& bars, int current_index);
    
    // Feature validation
    bool validate_features(const std::vector<double>& features);
    std::vector<std::string> get_feature_names() const;
    
    // Helper methods
    static std::vector<double> extract_closes(const std::vector<Bar>& bars);
    static std::vector<double> extract_volumes(const std::vector<Bar>& bars);
    static std::vector<double> extract_returns(const std::vector<Bar>& bars);
    
private:
    // Rolling calculations
    double calculate_rsi(const std::vector<double>& closes, int period, int current_index);
    double calculate_sma(const std::vector<double>& values, int period, int current_index);
    double calculate_ema(const std::vector<double>& values, int period, int current_index);
    double calculate_volatility(const std::vector<double>& returns, int period, int current_index);
    double calculate_atr(const std::vector<Bar>& bars, int period, int current_index);
    
    // Bollinger Bands
    struct BollingerBands {
        double upper, middle, lower;
    };
    BollingerBands calculate_bollinger_bands(const std::vector<double>& values, int period, double std_dev, int current_index);
    
    // MACD
    struct MACD {
        double line, signal, histogram;
    };
    MACD calculate_macd(const std::vector<double>& values, int fast, int slow, int signal, int current_index);
    
    // Stochastic
    struct Stochastic {
        double k, d;
    };
    Stochastic calculate_stochastic(const std::vector<Bar>& bars, int k_period, int d_period, int current_index);
    
    // Williams %R
    double calculate_williams_r(const std::vector<Bar>& bars, int period, int current_index);
    
    // CCI
    double calculate_cci(const std::vector<Bar>& bars, int period, int current_index);
    
    // ADX
    double calculate_adx(const std::vector<Bar>& bars, int period, int current_index);
    
    // Volume indicators
    double calculate_obv(const std::vector<Bar>& bars, int current_index);
    double calculate_vpt(const std::vector<Bar>& bars, int current_index);
    double calculate_ad_line(const std::vector<Bar>& bars, int current_index);
    double calculate_mfi(const std::vector<Bar>& bars, int period, int current_index);
    
    // Volatility indicators
    double calculate_parkinson_volatility(const std::vector<Bar>& bars, int period, int current_index);
    double calculate_garman_klass_volatility(const std::vector<Bar>& bars, int period, int current_index);
    
    // Feature names for validation
    std::vector<std::string> feature_names_;
};

} // namespace feature_engineering
} // namespace sentio
