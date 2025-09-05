#pragma once
#include <deque>
#include <cmath>
#include <limits>

namespace sentio {

struct SMA {
  int n{0};
  double sum{0.0};
  std::deque<double> q;
  explicit SMA(int n_) : n(n_) {}
  void reset(){ sum=0; q.clear(); }
  void push(double x){
    if (!std::isfinite(x)) { reset(); return; }
    q.push_back(x); sum += x;
    if ((int)q.size() > n) { sum -= q.front(); q.pop_front(); }
  }
  bool ready() const { return (int)q.size() == n; }
  double value() const { return ready() ? sum / n : std::numeric_limits<double>::quiet_NaN(); }
};

struct RSI {
  int n{14};
  int warm{0};
  double avgGain{0}, avgLoss{0}, prev{NAN};
  explicit RSI(int n_=14):n(n_),warm(0),avgGain(0),avgLoss(0),prev(NAN){}
  void reset(){ warm=0; avgGain=avgLoss=0; prev=NAN; }
  void push(double close){
    if (!std::isfinite(close)) { reset(); return; }
    if (!std::isfinite(prev)) { prev = close; return; }
    double delta = close - prev; prev = close;
    double gain = delta > 0 ? delta : 0.0;
    double loss = delta < 0 ? -delta : 0.0;
    if (warm < n) {
      avgGain += gain; avgLoss += loss; ++warm;
      if (warm==n) { avgGain/=n; avgLoss/=n; }
    } else {
      avgGain = (avgGain*(n-1) + gain)/n;
      avgLoss = (avgLoss*(n-1) + loss)/n;
    }
  }
  bool ready() const { return warm >= n; }
  double value() const {
    if (!ready()) return std::numeric_limits<double>::quiet_NaN();
    if (avgLoss == 0) return 100.0;
    double rs = avgGain/avgLoss;
    return 100.0 - (100.0/(1.0+rs));
  }
};

} // namespace sentio
