#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace sentio {

struct RollingMean {
  int win, idx=0, count=0;
  std::vector<double> buf;
  double sum=0.0;
  
  explicit RollingMean(int w = 1) { reset(w); }

  void reset(int w) {
    win = w > 0 ? w : 1;
    idx = 0;
    count = 0;
    sum = 0.0;
    buf.assign(win, 0.0);
  }

  inline double push(double x){
      if (count < win) { 
          buf[count++] = x; 
          sum += x; 
      } else {
          sum -= buf[idx]; 
          buf[idx]=x; 
          sum += x; 
          idx = (idx+1) % win; 
      }
      return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }

  double mean() const {
      return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }

  size_t size() const {
      return static_cast<size_t>(count);
  }
};


struct RollingMeanVar {
  int win, idx=0, count=0;
  std::vector<double> buf;
  double sum=0.0, sumsq=0.0;

  explicit RollingMeanVar(int w = 1) { reset(w); }

  void reset(int w) {
    win = w > 0 ? w : 1;
    idx = 0;
    count = 0;
    sum = 0.0;
    sumsq = 0.0;
    buf.assign(win, 0.0);
  }

  inline std::pair<double,double> push(double x){
    if (count < win) {
      buf[count++] = x; 
      sum += x; 
      sumsq += x*x;
    } else {
      double old_val = buf[idx];
      sum   -= old_val;
      sumsq -= old_val * old_val;
      buf[idx] = x;
      sum   += x;
      sumsq += x*x;
      idx = (idx+1) % win;
    }
    double m = count > 0 ? sum / static_cast<double>(count) : 0.0;
    double v = count > 0 ? std::max(0.0, (sumsq / static_cast<double>(count)) - (m*m)) : 0.0;
    return {m, v};
  }
  
  double mean() const {
      return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }
  
  double var() const {
      if (count < 2) return 0.0;
      double m = mean();
      return std::max(0.0, (sumsq / static_cast<double>(count)) - (m * m));
  }

  double stddev() const {
      return std::sqrt(var());
  }
};

} // namespace sentio
