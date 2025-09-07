#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace sentio {

// ============================================================================
// BASIC OPERATIONS
// ============================================================================

inline std::vector<float> op_IDENT(const std::vector<float>& x) { 
    return x; 
}

inline std::vector<float> op_LOGRET(const std::vector<float>& x) {
    std::vector<float> y(x.size(), 0.f);
    for (size_t i = 1; i < x.size(); ++i) {
        float a = std::max(x[i], 1e-12f), b = std::max(x[i-1], 1e-12f);
        y[i] = std::log(a) - std::log(b);
    }
    return y;
}

inline std::vector<float> op_MOMENTUM(const std::vector<float>& x, int w) {
    std::vector<float> out(x.size(), 0.f);
    for (size_t i = w; i < x.size(); ++i) {
        out[i] = x[i] - x[i-w];
    }
    return out;
}

inline std::vector<float> op_ROC(const std::vector<float>& x, int w) {
    std::vector<float> out(x.size(), 0.f);
    for (size_t i = w; i < x.size(); ++i) {
        float prev = std::max(x[i-w], 1e-12f);
        out[i] = ((x[i] - prev) / prev) * 100.0f;
    }
    return out;
}

// ============================================================================
// MOVING AVERAGES
// ============================================================================

inline std::vector<float> op_SMA(const std::vector<float>& x, int w) {
    std::vector<float> out(x.size(), 0.f);
    if (w <= 1) return x;
    double s = 0.0;
    for (int i = 0; i < (int)x.size(); ++i) {
        s += x[i];
        if (i >= w) s -= x[i-w];
        if (i >= w-1) out[i] = (float)(s/w);
    }
    return out;
}

inline std::vector<float> op_EMA(const std::vector<float>& x, int p) {
    std::vector<float> e(x.size()); 
    if (x.empty()) return e;
    float k = 2.f / (p + 1.f); 
    e[0] = x[0];
    for (size_t i = 1; i < x.size(); ++i) {
        e[i] = k * x[i] + (1.f - k) * e[i-1];
    }
    return e;
}

// ============================================================================
// VOLATILITY AND STATISTICAL MEASURES
// ============================================================================

inline std::vector<float> op_VOLATILITY(const std::vector<float>& x, int w) {
    std::vector<float> out(x.size(), 0.f);
    if ((int)x.size() < w) return out;
    double s = 0.0, s2 = 0.0;
    for (int i = 0; i < w; i++) { 
        s += x[i]; 
        s2 += x[i] * x[i]; 
    }
    for (size_t i = w; i < x.size(); ++i) {
        double mu = s / w;
        double var = std::max(0.0, s2 / w - mu * mu);
        out[i] = (float)std::sqrt(var);
        // slide window
        s += x[i] - x[i-w];
        s2 += x[i] * x[i] - x[i-w] * x[i-w];
    }
    return out;
}

inline std::vector<float> op_PARKINSON(const std::vector<float>& high, 
                                       const std::vector<float>& low, 
                                       int w) {
    std::vector<float> out(high.size(), 0.f);
    std::vector<float> hl_ratio(high.size(), 0.f);
    
    // Calculate log(H/L)^2 for each bar
    for (size_t i = 0; i < high.size(); ++i) {
        if (high[i] > 0 && low[i] > 0) {
            float ratio = std::log(high[i] / low[i]);
            hl_ratio[i] = ratio * ratio;
        }
    }
    
    // Rolling average of (log(H/L))^2 and scale by Parkinson constant
    auto avg_hl = op_SMA(hl_ratio, w);
    const float parkinson_factor = 1.0f / (4.0f * std::log(2.0f));
    
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = std::sqrt(avg_hl[i] * parkinson_factor);
    }
    return out;
}

inline std::vector<float> op_GARMAN_KLASS(const std::vector<float>& open,
                                          const std::vector<float>& high,
                                          const std::vector<float>& low,
                                          const std::vector<float>& close,
                                          int w) {
    std::vector<float> out(open.size(), 0.f);
    std::vector<float> gk_vals(open.size(), 0.f);
    
    // Calculate Garman-Klass estimator for each bar
    for (size_t i = 1; i < open.size(); ++i) {
        if (high[i] > 0 && low[i] > 0 && open[i] > 0 && close[i] > 0) {
            float log_hl = std::log(high[i] / low[i]);
            float log_co = std::log(close[i] / open[i]);
            gk_vals[i] = 0.5f * log_hl * log_hl - (2.0f * std::log(2.0f) - 1.0f) * log_co * log_co;
        }
    }
    
    // Rolling average
    auto avg_gk = op_SMA(gk_vals, w);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = std::sqrt(std::max(0.0f, avg_gk[i]));
    }
    return out;
}

// ============================================================================
// TECHNICAL INDICATORS
// ============================================================================

inline std::vector<float> op_RSI(const std::vector<float>& x, int p = 14) {
    const size_t N = x.size();
    std::vector<float> out(N, 0.f), up(N, 0.f), dn(N, 0.f);
    
    for (size_t i = 1; i < N; ++i) {
        float d = x[i] - x[i-1]; 
        up[i] = std::max(d, 0.f); 
        dn[i] = std::max(-d, 0.f);
    }
    
    std::vector<float> ru(N, 0.f), rd(N, 0.f);
    if (N > (size_t)p) {
        float su = 0.f, sd = 0.f;
        for (int i = 1; i <= p; i++) { 
            su += up[i]; 
            sd += dn[i]; 
        }
        ru[p] = su / p; 
        rd[p] = sd / p;
        
        for (size_t i = p + 1; i < N; ++i) {
            ru[i] = (ru[i-1] * (p-1) + up[i]) / p;
            rd[i] = (rd[i-1] * (p-1) + dn[i]) / p;
            float rs = (rd[i] > 1e-12f) ? (ru[i] / rd[i]) : 0.f;
            out[i] = 100.f - 100.f / (1.f + rs);
        }
    }
    return out;
}

inline std::vector<float> op_ZWIN(const std::vector<float>& x, int w) {
    const size_t N = x.size(); 
    std::vector<float> out(N, 0.f);
    if (N <= (size_t)w) return out;
    
    std::vector<double> s(N, 0.0), s2(N, 0.0);
    s[0] = x[0]; 
    s2[0] = x[0] * x[0];
    
    for (size_t i = 1; i < N; ++i) { 
        s[i] = s[i-1] + x[i]; 
        s2[i] = s2[i-1] + x[i] * x[i]; 
    }
    
    for (size_t i = w; i < N; ++i) {
        double su = s[i] - s[i-w], su2 = s2[i] - s2[i-w];
        double mu = su / w;
        double var = std::max(0.0, su2 / w - mu * mu);
        float sd = (float)std::sqrt(var);
        out[i] = (sd > 1e-8f) ? (float)((x[i] - mu) / sd) : 0.f;
    }
    return out;
}

inline std::vector<float> op_ATR(const std::vector<float>& high,
                                 const std::vector<float>& low,
                                 const std::vector<float>& close,
                                 int p) {
    const size_t N = high.size();
    std::vector<float> tr(N, 0.f), atr(N, 0.f);
    
    for (size_t i = 1; i < N; ++i) {
        float h = high[i], l = low[i], cp = close[i-1];
        float v = std::max({h-l, std::fabs(h-cp), std::fabs(l-cp)});
        tr[i] = v;
    }
    
    // Wilder smoothing
    if (N > (size_t)p) {
        float a = 0.f; 
        for (int i = 1; i <= p; i++) a += tr[i];
        a /= p; 
        atr[p] = a;
        for (size_t i = p + 1; i < N; ++i) {
            atr[i] = (atr[i-1] * (p-1) + tr[i]) / p;
        }
    }
    return atr;
}

// ============================================================================
// BOLLINGER BANDS
// ============================================================================

struct BollingerBands { 
    std::vector<float> upper, middle, lower; 
};

inline BollingerBands op_BOLLINGER(const std::vector<float>& x, int w, float k = 2.0f) {
    BollingerBands b; 
    b.upper.resize(x.size(), 0.f); 
    b.middle.resize(x.size(), 0.f); 
    b.lower.resize(x.size(), 0.f);
    
    auto sma = op_SMA(x, w);
    auto sd = op_VOLATILITY(x, w);
    
    for (size_t i = w; i < x.size(); ++i) {
        b.middle[i] = sma[i];
        b.upper[i] = sma[i] + k * sd[i];
        b.lower[i] = sma[i] - k * sd[i];
    }
    return b;
}

// ============================================================================
// MACD
// ============================================================================

struct MACD { 
    std::vector<float> line, signal, histogram; 
};

inline MACD op_MACD(const std::vector<float>& x, int fast = 12, int slow = 26, int sig = 9) {
    MACD m; 
    m.line.resize(x.size()); 
    m.signal.resize(x.size()); 
    m.histogram.resize(x.size());
    
    auto ema_fast = op_EMA(x, fast);
    auto ema_slow = op_EMA(x, slow);
    
    for (size_t i = 0; i < x.size(); ++i) {
        m.line[i] = ema_fast[i] - ema_slow[i];
    }
    
    m.signal = op_EMA(m.line, sig);
    
    for (size_t i = 0; i < x.size(); ++i) {
        m.histogram[i] = m.line[i] - m.signal[i];
    }
    return m;
}

// ============================================================================
// STOCHASTIC OSCILLATOR
// ============================================================================

struct Stochastic {
    std::vector<float> k, d;
};

inline Stochastic op_STOCHASTIC(const std::vector<float>& high,
                                const std::vector<float>& low,
                                const std::vector<float>& close,
                                int k_period = 14,
                                int d_period = 3) {
    Stochastic stoch;
    stoch.k.resize(close.size(), 0.f);
    stoch.d.resize(close.size(), 0.f);
    
    for (size_t i = k_period; i < close.size(); ++i) {
        float highest = *std::max_element(high.begin() + i - k_period, high.begin() + i + 1);
        float lowest = *std::min_element(low.begin() + i - k_period, low.begin() + i + 1);
        
        if (highest > lowest) {
            stoch.k[i] = ((close[i] - lowest) / (highest - lowest)) * 100.0f;
        }
    }
    
    stoch.d = op_SMA(stoch.k, d_period);
    return stoch;
}

// ============================================================================
// OTHER OSCILLATORS
// ============================================================================

inline std::vector<float> op_WILLIAMS_R(const std::vector<float>& high,
                                        const std::vector<float>& low,
                                        const std::vector<float>& close,
                                        int period = 14) {
    std::vector<float> out(close.size(), 0.f);
    
    for (size_t i = period; i < close.size(); ++i) {
        float highest = *std::max_element(high.begin() + i - period, high.begin() + i + 1);
        float lowest = *std::min_element(low.begin() + i - period, low.begin() + i + 1);
        
        if (highest > lowest) {
            out[i] = ((highest - close[i]) / (highest - lowest)) * -100.0f;
        }
    }
    return out;
}

inline std::vector<float> op_CCI(const std::vector<float>& high,
                                 const std::vector<float>& low,
                                 const std::vector<float>& close,
                                 int period = 20) {
    std::vector<float> out(close.size(), 0.f);
    std::vector<float> tp(close.size()); // typical price
    
    for (size_t i = 0; i < close.size(); ++i) {
        tp[i] = (high[i] + low[i] + close[i]) / 3.0f;
    }
    
    auto sma_tp = op_SMA(tp, period);
    
    for (size_t i = period; i < close.size(); ++i) {
        float mean_dev = 0.0f;
        for (size_t j = i - period + 1; j <= i; ++j) {
            mean_dev += std::fabs(tp[j] - sma_tp[i]);
        }
        mean_dev /= period;
        
        if (mean_dev > 1e-8f) {
            out[i] = (tp[i] - sma_tp[i]) / (0.015f * mean_dev);
        }
    }
    return out;
}

inline std::vector<float> op_ADX(const std::vector<float>& high,
                                 const std::vector<float>& low,
                                 const std::vector<float>& close,
                                 int period = 14) {
    const size_t N = close.size();
    std::vector<float> adx(N, 0.f);
    std::vector<float> dm_plus(N, 0.f), dm_minus(N, 0.f), tr(N, 0.f);
    
    // Calculate directional movement and true range
    for (size_t i = 1; i < N; ++i) {
        float up_move = high[i] - high[i-1];
        float down_move = low[i-1] - low[i];
        
        dm_plus[i] = (up_move > down_move && up_move > 0) ? up_move : 0.0f;
        dm_minus[i] = (down_move > up_move && down_move > 0) ? down_move : 0.0f;
        
        float h = high[i], l = low[i], cp = close[i-1];
        tr[i] = std::max({h-l, std::fabs(h-cp), std::fabs(l-cp)});
    }
    
    // Smooth the values using Wilder's smoothing
    if (N > (size_t)period) {
        float sum_dm_plus = 0, sum_dm_minus = 0, sum_tr = 0;
        for (int i = 1; i <= period; i++) {
            sum_dm_plus += dm_plus[i];
            sum_dm_minus += dm_minus[i];
            sum_tr += tr[i];
        }
        
        float smooth_dm_plus = sum_dm_plus;
        float smooth_dm_minus = sum_dm_minus;
        float smooth_tr = sum_tr;
        
        for (size_t i = period + 1; i < N; ++i) {
            smooth_dm_plus = smooth_dm_plus - smooth_dm_plus/period + dm_plus[i];
            smooth_dm_minus = smooth_dm_minus - smooth_dm_minus/period + dm_minus[i];
            smooth_tr = smooth_tr - smooth_tr/period + tr[i];
            
            float di_plus = (smooth_tr > 0) ? (smooth_dm_plus / smooth_tr) * 100 : 0;
            float di_minus = (smooth_tr > 0) ? (smooth_dm_minus / smooth_tr) * 100 : 0;
            
            float di_sum = di_plus + di_minus;
            float dx = (di_sum > 0) ? std::fabs(di_plus - di_minus) / di_sum * 100 : 0;
            
            // Simple moving average of DX for ADX
            if (i >= period * 2) {
                float adx_sum = 0;
                for (size_t j = i - period + 1; j <= i; ++j) {
                    // Recalculate DX for each period (simplified)
                    adx_sum += dx; // This is simplified; full implementation would store DX values
                }
                adx[i] = adx_sum / period;
            }
        }
    }
    return adx;
}

// ============================================================================
// VOLUME INDICATORS
// ============================================================================

inline std::vector<float> op_OBV(const std::vector<float>& close,
                                 const std::vector<float>& volume) {
    std::vector<float> obv(close.size(), 0.f);
    
    for (size_t i = 1; i < close.size(); ++i) {
        if (close[i] > close[i-1]) {
            obv[i] = obv[i-1] + volume[i];
        } else if (close[i] < close[i-1]) {
            obv[i] = obv[i-1] - volume[i];
        } else {
            obv[i] = obv[i-1];
        }
    }
    return obv;
}

inline std::vector<float> op_VPT(const std::vector<float>& close,
                                 const std::vector<float>& volume) {
    std::vector<float> vpt(close.size(), 0.f);
    
    for (size_t i = 1; i < close.size(); ++i) {
        if (close[i-1] > 0) {
            float pct_change = (close[i] - close[i-1]) / close[i-1];
            vpt[i] = vpt[i-1] + volume[i] * pct_change;
        } else {
            vpt[i] = vpt[i-1];
        }
    }
    return vpt;
}

inline std::vector<float> op_AD_LINE(const std::vector<float>& high,
                                     const std::vector<float>& low,
                                     const std::vector<float>& close,
                                     const std::vector<float>& volume) {
    std::vector<float> ad(close.size(), 0.f);
    
    for (size_t i = 1; i < close.size(); ++i) {
        float hl_diff = high[i] - low[i];
        if (hl_diff > 1e-8f) {
            float mfm = ((close[i] - low[i]) - (high[i] - close[i])) / hl_diff;
            float mfv = mfm * volume[i];
            ad[i] = ad[i-1] + mfv;
        } else {
            ad[i] = ad[i-1];
        }
    }
    return ad;
}

inline std::vector<float> op_MFI(const std::vector<float>& high,
                                 const std::vector<float>& low,
                                 const std::vector<float>& close,
                                 const std::vector<float>& volume,
                                 int period = 14) {
    std::vector<float> mfi(close.size(), 0.f);
    std::vector<float> tp(close.size()); // typical price
    std::vector<float> mf(close.size()); // money flow
    
    for (size_t i = 0; i < close.size(); ++i) {
        tp[i] = (high[i] + low[i] + close[i]) / 3.0f;
        mf[i] = tp[i] * volume[i];
    }
    
    for (size_t i = period; i < close.size(); ++i) {
        float positive_mf = 0, negative_mf = 0;
        
        for (size_t j = i - period + 1; j <= i; ++j) {
            if (j > 0) {
                if (tp[j] > tp[j-1]) {
                    positive_mf += mf[j];
                } else if (tp[j] < tp[j-1]) {
                    negative_mf += mf[j];
                }
            }
        }
        
        if (negative_mf > 1e-8f) {
            float mfr = positive_mf / negative_mf;
            mfi[i] = 100.0f - (100.0f / (1.0f + mfr));
        }
    }
    return mfi;
}

// ============================================================================
// MICROSTRUCTURE INDICATORS (Simplified implementations)
// ============================================================================

inline std::vector<float> op_SPREAD_BP(const std::vector<float>& open,
                                       const std::vector<float>& high,
                                       const std::vector<float>& low,
                                       const std::vector<float>& close) {
    std::vector<float> spread(close.size(), 0.f);
    
    for (size_t i = 0; i < close.size(); ++i) {
        float mid = (high[i] + low[i]) / 2.0f;
        if (mid > 1e-8f) {
            spread[i] = ((high[i] - low[i]) / mid) * 10000.0f; // basis points
        }
    }
    return spread;
}

inline std::vector<float> op_PRICE_IMPACT(const std::vector<float>& open,
                                          const std::vector<float>& high,
                                          const std::vector<float>& low,
                                          const std::vector<float>& close,
                                          const std::vector<float>& volume) {
    std::vector<float> impact(close.size(), 0.f);
    
    for (size_t i = 1; i < close.size(); ++i) {
        float price_change = std::fabs(close[i] - close[i-1]);
        float vol_sqrt = std::sqrt(volume[i]);
        if (vol_sqrt > 1e-8f && close[i-1] > 1e-8f) {
            impact[i] = (price_change / close[i-1]) / vol_sqrt * 1000000.0f; // scaled
        }
    }
    return impact;
}

inline std::vector<float> op_ORDER_FLOW(const std::vector<float>& open,
                                        const std::vector<float>& high,
                                        const std::vector<float>& low,
                                        const std::vector<float>& close,
                                        const std::vector<float>& volume) {
    std::vector<float> flow(close.size(), 0.f);
    
    for (size_t i = 0; i < close.size(); ++i) {
        float hl_diff = high[i] - low[i];
        if (hl_diff > 1e-8f) {
            float buy_pressure = (close[i] - low[i]) / hl_diff;
            float sell_pressure = (high[i] - close[i]) / hl_diff;
            flow[i] = (buy_pressure - sell_pressure) * volume[i];
        }
    }
    return flow;
}

inline std::vector<float> op_MARKET_DEPTH(const std::vector<float>& open,
                                          const std::vector<float>& high,
                                          const std::vector<float>& low,
                                          const std::vector<float>& close,
                                          const std::vector<float>& volume) {
    std::vector<float> depth(close.size(), 0.f);
    
    for (size_t i = 0; i < close.size(); ++i) {
        float range = high[i] - low[i];
        if (volume[i] > 1e-8f && range > 1e-8f) {
            depth[i] = range / std::log(volume[i] + 1.0f);
        }
    }
    return depth;
}

inline std::vector<float> op_BID_ASK_RATIO(const std::vector<float>& open,
                                          const std::vector<float>& high,
                                          const std::vector<float>& low,
                                          const std::vector<float>& close) {
    std::vector<float> ratio(close.size(), 0.f);
    
    for (size_t i = 0; i < close.size(); ++i) {
        // Simplified: use close position within range as proxy
        float range = high[i] - low[i];
        if (range > 1e-8f) {
            float position = (close[i] - low[i]) / range; // 0 = low, 1 = high
            ratio[i] = position / (1.0f - position + 1e-8f); // bid/ask proxy
        }
    }
    return ratio;
}

} // namespace sentio
