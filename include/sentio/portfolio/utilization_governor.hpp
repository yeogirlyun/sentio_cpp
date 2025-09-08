#pragma once
#include <algorithm>

namespace sentio {

struct UtilGovConfig {
  double target_gross = 0.60;  // target gross exposure (sum |weights|), e.g. 60%
  double kp_expo      = 0.05;  // proportional gain for exposure gap
  double target_tpd   = 40.0;  // trades per day target (optional)
  double kp_trades    = 0.02;  // proportional gain for trade gap
  float  max_shift    = 0.10f; // max threshold nudge
  double max_vol_adj  = 0.50;  // ±50% of vol target adjustment
};

struct UtilGovState {
  double expo_shift{0.0};      // maps to sizer's vol target multiplier
  float  buy_shift{0.0f};      // router threshold shift
  float  sell_shift{0.0f};
  double integ_expo{0.0};      // optional: add Ki later if needed
  double integ_trades{0.0};
};

class UtilizationGovernor {
public:
  explicit UtilizationGovernor(const UtilGovConfig& c) : cfg_(c) {}

  void daily_update(double realized_gross, int trades_today, UtilGovState& st){
    // Exposure control
    double e_err = cfg_.target_gross - realized_gross;
    st.expo_shift = std::clamp(st.expo_shift + cfg_.kp_expo * e_err,
                               -cfg_.max_vol_adj, cfg_.max_vol_adj);

    // Trades/day control → route thresholds
    double t_err = cfg_.target_tpd - trades_today;
    float delta  = (float)(cfg_.kp_trades * t_err);
    st.buy_shift  = clamp_shift(st.buy_shift  - 0.5f*delta);
    st.sell_shift = clamp_shift(st.sell_shift - 0.5f*delta);
  }

  void get_nudges(struct RouterNudges& nudges, const UtilGovState& st) const {
    nudges.buy_shift = st.buy_shift;
    nudges.sell_shift = st.sell_shift;
  }

private:
  float clamp_shift(float x) const {
    return std::clamp(x, -cfg_.max_shift, cfg_.max_shift);
  }
  UtilGovConfig cfg_;
};

// Forward declare RouterNudges for get_nudges method
struct RouterNudges {
  float buy_shift  = 0.f;
  float sell_shift = 0.f;
};

} // namespace sentio
