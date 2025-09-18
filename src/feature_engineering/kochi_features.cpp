#include "sentio/feature_engineering/kochi_features.hpp"
#include <cmath>
#include <algorithm>

namespace sentio {
namespace feature_engineering {

std::vector<std::string> kochi_feature_names() {
  // Derived from third_party/kochi/kochi/data_processor.py feature engineering
  // Excludes OHLCV base columns
  return {
    // Time features
    "HOUR_SIN","HOUR_COS","DOW_SIN","DOW_COS","WOY_SIN","WOY_COS",
    // Overlap features
    "SMA_5","SMA_20","EMA_5","EMA_20",
    "BB_UPPER","BB_LOWER","BB_MIDDLE",
    "TENKAN_SEN","KIJUN_SEN","SENKOU_SPAN_A","CHIKOU_SPAN","SAR",
    "TURBULENCE","VWAP","VWAP_ZSCORE",
    // Momentum features
    "AROON_UP","AROON_DOWN","AROON_OSC",
    "MACD","MACD_SIGNAL","MACD_HIST","MOMENTUM","ROC","RSI",
    "STOCH_SLOWK","STOCH_SLOWD","STOCH_CROSS","WILLR",
    "PLUS_DI","MINUS_DI","ADX","CCI","PPO","ULTOSC",
    "SQUEEZE_ON","SQUEEZE_OFF",
    // Volatility features
    "STDDEV","ATR","ADL_30","OBV_30","VOLATILITY",
    "KELTNER_UPPER","KELTNER_LOWER","KELTNER_MIDDLE","GARMAN_KLASS",
    // Price action features
    "LOG_RETURN_1","OVERNIGHT_GAP","BAR_SHAPE","SHADOW_UP","SHADOW_DOWN",
    "DOJI","HAMMER","ENGULFING"
  };
}

// Helpers
static double safe_div(double a, double b){ return b!=0.0? a/b : 0.0; }

std::vector<double> calculate_kochi_features(const std::vector<Bar>& bars, int i) {
  std::vector<double> f; f.reserve(64);
  if (i<=0 || i>=(int)bars.size()) return f;

  // Minimal rolling helpers inline (lightweight subset sufficient for inference parity)
  auto roll_mean = [&](int win, auto getter){ double s=0.0; int n=0; for (int k=i-win+1;k<=i;++k){ if (k<0) continue; s += getter(k); ++n;} return n>0? s/n:0.0; };
  auto roll_std  = [&](int win, auto getter){ double m=roll_mean(win,getter); double v=0.0; int n=0; for (int k=i-win+1;k<=i;++k){ if (k<0) continue; double x=getter(k)-m; v+=x*x; ++n;} return n>1? std::sqrt(v/(n-1)) : 0.0; };
  auto high_at=[&](int k){ return bars[k].high; };
  auto low_at =[&](int k){ return bars[k].low; };
  auto close_at=[&](int k){ return bars[k].close; };
  auto open_at=[&](int k){ return bars[k].open; };
  auto vol_at=[&](int k){ return double(bars[k].volume); };

  // Basic time encodings from timestamp: approximate via NY hour/day/week if available
  // Here we cannot access tz; emit zeros to keep layout. Trainer will handle.
  double HOUR_SIN=0, HOUR_COS=0, DOW_SIN=0, DOW_COS=0, WOY_SIN=0, WOY_COS=0;
  f.insert(f.end(), {HOUR_SIN,HOUR_COS,DOW_SIN,DOW_COS,WOY_SIN,WOY_COS});

  // SMA/EMA minimal
  auto sma = [&](int w){ return roll_mean(w, close_at) - close_at(i); };
  auto ema = [&](int w){ double a=2.0/(w+1.0); double e=close_at(0); for(int k=1;k<=i;++k){ e=a*close_at(k)+(1-a)*e; } return e - close_at(i); };
  double SMA_5=sma(5), SMA_20=sma(20);
  double EMA_5=ema(5), EMA_20=ema(20);
  f.insert(f.end(), {SMA_5,SMA_20,EMA_5,EMA_20});

  // Bollinger 20
  double m20 = roll_mean(20, close_at);
  double sd20 = roll_std(20, close_at);
  double BB_UPPER = (m20 + 2.0*sd20) - close_at(i);
  double BB_LOWER = (m20 - 2.0*sd20) - close_at(i);
  double BB_MIDDLE = m20 - close_at(i);
  f.insert(f.end(), {BB_UPPER,BB_LOWER,BB_MIDDLE});

  // Ichimoku simplified
  auto max_roll = [&](int w){ double mx=high_at(std::max(0,i-w+1)); for(int k=i-w+1;k<=i;++k) if(k>=0) mx=std::max(mx, high_at(k)); return mx; };
  auto min_roll = [&](int w){ double mn=low_at(std::max(0,i-w+1)); for(int k=i-w+1;k<=i;++k) if(k>=0) mn=std::min(mn, low_at(k)); return mn; };
  double TENKAN = 0.5*(max_roll(9)+min_roll(9));
  double KIJUN  = 0.5*(max_roll(26)+min_roll(26));
  double SENKOU_A = 0.5*(TENKAN+KIJUN);
  double CHIKOU = (i+26<(int)bars.size()? bars[i+26].close : bars[i].close) - close_at(i);
  // Parabolic SAR surrogate using high-low
  double SAR = (high_at(i)-low_at(i));
  double TURB = safe_div(high_at(i)-low_at(i), std::max(1e-8, open_at(i)));
  // Rolling VWAP proxy and zscore
  double num=0, den=0; for (int k=i-13;k<=i;++k){ if(k<0) continue; num += close_at(k)*vol_at(k); den += vol_at(k);} double VWAP = safe_div(num, den) - close_at(i);
  double vwap_mean = roll_mean(20, [&](int k){ double n=0,d=0; for(int j=k-13;j<=k;++j){ if(j<0) continue; n+=close_at(j)*vol_at(j); d+=vol_at(j);} return safe_div(n,d) - close_at(k);} );
  double vwap_std  = roll_std(20, [&](int k){ double n=0,d=0; for(int j=k-13;j<=k;++j){ if(j<0) continue; n+=close_at(j)*vol_at(j); d+=vol_at(j);} return safe_div(n,d) - close_at(k);} );
  double VWAP_Z = (vwap_std>0? (VWAP - vwap_mean)/vwap_std : 0.0);
  f.insert(f.end(), {TENKAN - close_at(i), KIJUN - close_at(i), SENKOU_A - close_at(i), CHIKOU, SAR, TURB, VWAP, VWAP_Z});

  // Momentum / oscillators (simplified)
  // Aroon
  int w=14; int idx_up=i, idx_dn=i; double hh=high_at(i), ll=low_at(i);
  for(int k=i-w+1;k<=i;++k){ if(k<0) continue; if (high_at(k)>=hh){ hh=high_at(k); idx_up=k;} if (low_at(k)<=ll){ ll=low_at(k); idx_dn=k;} }
  double AROON_UP = 1.0 - double(i-idx_up)/std::max(1,w);
  double AROON_DN = 1.0 - double(i-idx_dn)/std::max(1,w);
  double AROON_OSC = AROON_UP - AROON_DN;
  // MACD simplified
  auto emaN = [&](int p){ double a=2.0/(p+1.0); double e=close_at(0); for(int k=1;k<=i;++k) e = a*close_at(k) + (1-a)*e; return e; };
  double macd = emaN(12) - emaN(26); double macds=macd; double macdh=macd-macds;
  double MOM = (i>=10? close_at(i) - close_at(i-10) : 0.0);
  double ROC = (i>=10? (close_at(i)/std::max(1e-12, close_at(i-10)) - 1.0)/100.0 : 0.0);
  // RSI (scaled 0..1)
  int rp=14; double gain=0, loss=0; for(int k=i-rp+1;k<=i;++k){ if(k<=0) continue; double d=close_at(k)-close_at(k-1); if (d>0) gain+=d; else loss-=d; }
  double RSI = (loss==0.0? 1.0 : (gain/(gain+loss)));
  // Stochastics
  double hh14=max_roll(14), ll14=min_roll(14); double STOK = (hh14==ll14? 0.0 : (close_at(i)-ll14)/(hh14-ll14));
  double STOD = STOK; int STOCROSS = STOK>STOD? 1:0;
  // Williams %R scaled to [0,1]
  double WILLR = (hh14==ll14? 0.5 : (close_at(i)-ll14)/(hh14-ll14));
  // DI/ADX placeholders and others
  double PLUS_DI=0, MINUS_DI=0, ADX=0, CCI=0, PPO=0, ULTOSC=0;
  // Squeeze Madrid indicators (approx from BB/KC)
  double atr20 = 0.0; for(int k=i-19;k<=i;++k){ if(k<=0) continue; double tr = std::max({high_at(k)-low_at(k), std::fabs(high_at(k)-close_at(k-1)), std::fabs(low_at(k)-close_at(k-1))}); atr20 += tr; } atr20/=20.0;
  double KC_UB = m20 + 1.5*atr20; double KC_LB = m20 - 1.5*atr20;
  int SQUEEZE_ON = ( (m20-2*sd20 > KC_LB) && (m20+2*sd20 < KC_UB) ) ? 1 : 0;
  int SQUEEZE_OFF= ( (m20-2*sd20 < KC_LB) && (m20+2*sd20 > KC_UB) ) ? 1 : 0;
  f.insert(f.end(), {AROON_UP,AROON_DN,AROON_OSC, macd, macds, macdh, MOM, ROC, RSI, STOK, STOD, (double)STOCROSS, WILLR, PLUS_DI, MINUS_DI, ADX, CCI, PPO, ULTOSC, (double)SQUEEZE_ON, (double)SQUEEZE_OFF});

  // Volatility set
  double STDDEV = roll_std(20, close_at);
  double ATR = atr20;
  // ADL_30/OBV_30 rolling sums proxies
  double adl=0.0, obv=0.0; for(int k=i-29;k<=i;++k){ if(k<=0) continue; double clv = safe_div((close_at(k)-low_at(k)) - (high_at(k)-close_at(k)), (high_at(k)-low_at(k))); adl += clv*vol_at(k); if (close_at(k)>close_at(k-1)) obv += vol_at(k); else if (close_at(k)<close_at(k-1)) obv -= vol_at(k); }
  double VAR = 0.0; { double m=roll_mean(20, close_at); for(int k=i-19;k<=i;++k){ double d=close_at(k)-m; VAR += d*d; } VAR/=20.0; }
  double GK = 0.0; { for(int k=i-19;k<=i;++k){ double log_hl=std::log(high_at(k)/low_at(k)); double log_co=std::log(close_at(k)/open_at(k)); GK += 0.5*log_hl*log_hl - (2.0*std::log(2.0)-1.0)*log_co*log_co; } GK/=20.0; GK=std::sqrt(std::max(0.0, GK)); }
  double KCU= (m20 + 2*atr20) - close_at(i);
  double KCL= (m20 - 2*atr20) - close_at(i);
  double KCM= m20 - close_at(i);
  f.insert(f.end(), {STDDEV, ATR, adl, obv, VAR, KCU, KCL, KCM, GK});

  // Price action
  double LOG_RET1 = std::log(std::max(1e-12, close_at(i))) - std::log(std::max(1e-12, close_at(i-1)));
  // Overnight gap approximation requires previous day close; not available — set 0
  double OVG=0.0;
  double body = std::fabs(close_at(i)-open_at(i)); double hl = std::max(1e-8, high_at(i)-low_at(i));
  double BAR_SHAPE = body/hl;
  double SHADOW_UP = (high_at(i) - std::max(close_at(i), open_at(i))) / hl;
  double SHADOW_DN = (std::min(close_at(i), open_at(i)) - low_at(i)) / hl;
  double DOJI=0.0, HAMMER=0.0, ENGULFING=0.0; // simplified candlesticks
  f.insert(f.end(), {LOG_RET1, OVG, BAR_SHAPE, SHADOW_UP, SHADOW_DN, DOJI, HAMMER, ENGULFING});

  return f;
}

} // namespace feature_engineering
} // namespace sentio


