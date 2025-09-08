#pragma once
#include "irule.hpp"
#include <algorithm>
#include <cmath>

namespace sentio::rules {

inline float logistic(float x, float k=1.2f){ return 1.f/(1.f+std::exp(-k*x)); }

inline float signal_to_p(float sig_pm, float strength=1.f){
  strength=std::clamp(strength,0.f,1.f);
  if (sig_pm>0)  return std::min(1.f, 0.5f + 0.5f*strength);
  if (sig_pm<0)  return std::max(0.f, 0.5f - 0.5f*strength);
  return 0.5f;
}

inline float to_probability(const RuleOutput& out, float k_logistic=1.2f){
  if (out.p_up)   return std::clamp(*out.p_up, 0.f, 1.f);
  if (out.signal) return signal_to_p((float)*out.signal, out.strength.value_or(1.f));
  if (out.score)  return logistic(*out.score, k_logistic);
  return 0.5f;
}

} // namespace sentio::rules


