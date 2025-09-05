// volatility_expansion.hpp
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "rth_calendar.hpp"
#include "signal_diag.hpp"

namespace sentio {


// Bar is defined in core.hpp

inline double true_range(double h,double l,double pc){
  double r1=h-l, r2=std::fabs(h-pc), r3=std::fabs(l-pc);
  return std::max(r1, std::max(r2,r3));
}

struct VEParams {
  // Relaxed defaults that actually fire on QQQ minute bars
  int    atr_window   = 14;
  double atr_alpha    = 2.0 / (14 + 1.0);
  int    lookback_hh  = 20;
  int    lookback_ll  = 20;
  double breakout_k   = 0.75;    // was 1.0 → easier
  int    hold_max_bars= 160;
  double tp_atr_mult  = 1.5;
  double sl_atr_mult  = 1.0;
  bool   require_rth  = true;
  int    cooldown_bars= 5;       // avoid immediate re-entries
};

// SignalType is defined in core.hpp

// StrategySignal is defined in base_strategy.hpp

struct VEResult { std::vector<int> entry_idx; std::vector<int> exit_idx; std::vector<int> dir; };

struct RollingHHLL {
  int w; std::vector<double> hi, lo; int idx=0, cnt=0;
  RollingHHLL(int win): w(win), hi(win, -INFINITY), lo(win, +INFINITY){}
  inline std::pair<double,double> push(double H,double L){
    hi[idx]=H; lo[idx]=L; idx=(idx+1)%w; if (cnt<w) ++cnt;
    double HH=-INFINITY, LL=+INFINITY;
    for(int k=0;k<cnt;++k){ HH=std::max(HH,hi[k]); LL=std::min(LL,lo[k]); }
    return {HH,LL};
  }
};

// VolatilityExpansionStrategy class is defined in strategy_volatility_expansion.hpp

} // namespace sentio