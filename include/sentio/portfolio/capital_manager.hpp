#pragma once
#include <vector>
#include <cstdint>
#include "portfolio_allocator.hpp"
#include "utilization_governor.hpp"

namespace sentio {

// Forward declarations to avoid circular includes
class PositionSizer;
class BinaryRouter;
struct MarketSnapshot;

struct InstrumentState {
  // live state snapshot
  int64_t ts;
  double  price;
  double  vol_1d;
  double  adv_notional;
  double  spread_bp;
  long    shares;       // current position
  double  w_prev;       // last target weight
  bool    tradable{true};
};

struct RouterInputs {
  // Per-instrument p_up from Strategy (TFA/Kochi/RuleEnsemble)
  double p_up;
};

struct CapitalManagerConfig {
  AllocConfig alloc;
  // Map utilization governor into Router and Sizer:
  UtilGovConfig util;
};

struct CapitalDecision {
  std::vector<double> target_weights; // per instrument
  std::vector<long long> target_shares;
  std::vector<double> edge_bp;        // per-instrument edge in bps
  double gross{0.0};
  double total_cost_bp{0.0};          // estimated total cost
};

class CapitalManager {
public:
  CapitalManager(const CapitalManagerConfig& cfg,
                 PortfolioAllocator alloc)
   : cfg_(cfg), alloc_(std::move(alloc)) {}

  CapitalDecision decide(double equity,
                         const std::vector<InstrumentState>& S,
                         const std::vector<RouterInputs>& R,
                         UtilGovState& ug_state)
  {
    const int N = (int)S.size();
    std::vector<InstrumentInputs> X; X.reserve(N);
    for (int i=0;i<N;i++){
      X.push_back(InstrumentInputs{
        /*p_up   */ R[i].p_up,
        /*price  */ S[i].price,
        /*vol_1d */ S[i].vol_1d,
        /*spread */ S[i].spread_bp,
        /*adv    */ S[i].adv_notional,
        /*w_prev */ S[i].w_prev,
        /*pos_w  */ (S[i].price>0? (S[i].shares*S[i].price)/std::max(1e-9, equity) : 0.0),
        /*trad   */ S[i].tradable
      });
    }

    // Portfolio-level allocation with SOTA linear policy
    auto out = alloc_.allocate(X, cfg_.alloc);
    
    // Apply utilization governor: upscale/downscale weights via expo_shift
    const double expo_mul = std::clamp(1.0 + ug_state.expo_shift, 0.5, 1.5);
    for (double& w : out.w_target) w *= expo_mul;

    // Compute shares via weight-based sizing
    std::vector<long long> tgt_sh(N, 0);
    for (int i=0;i<N;i++){
      if (S[i].price > 0 && std::isfinite(out.w_target[i])) {
        double desired_notional = out.w_target[i] * equity;
        tgt_sh[i] = (long long)std::floor(desired_notional / S[i].price);
        // Apply round lot if needed (assume 1 for now)
        // Apply min notional filter (assume $50 from previous fix)
        if (std::abs(tgt_sh[i] * S[i].price) < 50.0) {
          tgt_sh[i] = 0;
        }
      }
    }

    // Report gross and cost estimates
    double gross=0.0, total_cost=0.0;
    for (int i=0;i<N;i++) {
      gross += std::abs(out.w_target[i]);
      // Estimate turnover cost
      double w_change = std::abs(out.w_target[i] - S[i].w_prev);
      total_cost += w_change * 5.0; // rough 5bp per weight change
    }

    return CapitalDecision{ 
      std::move(out.w_target), 
      std::move(tgt_sh), 
      std::move(out.edge_bp),
      gross,
      total_cost
    };
  }

private:
  CapitalManagerConfig cfg_;
  PortfolioAllocator alloc_;
};

} // namespace sentio
