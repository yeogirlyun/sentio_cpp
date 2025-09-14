#pragma once
#include "irule.hpp"
#include "adapters.hpp"
#include "online_platt_calibrator.hpp"
#include "diversity_weighter.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace sentio::rules {

struct EnsembleConfig {
  float score_logistic_k = 1.2f;
  int   reliability_window = 512;
  std::vector<float> base_weights; // size=rules; default=1
  float agreement_boost = 0.25f;   // scales (p-0.5)
  int   min_rules = 1;
  float eps_clip = 1e-4f;
  bool  use_diversity = true;
  DiversityCfg diversity;
  bool  use_platt = true;
  PlattCfg platt;
};

struct EnsembleMeta {
  int    n_used{0};
  float  agreement{-1.f};
  float  variance{0.f};
  float  weighted_mean{0.f};
  std::vector<float> member_p;
  std::vector<float> member_w;
};

class IntegratedRuleEnsemble {
public:
  IntegratedRuleEnsemble(std::vector<std::unique_ptr<IRuleStrategy>> rules, EnsembleConfig cfg)
  : cfg_(std::move(cfg)), rules_(std::move(rules)),
    div_((int)rules_.size(), cfg_.diversity), platt_(cfg_.platt)
  {
    const size_t R = rules_.size();
    if (cfg_.base_weights.empty()) cfg_.base_weights.assign(R, 1.0f);
    reli_brier_.assign(R, {}); reli_hits_.assign(R, {});
  }

  int warmup() const {
    int w = 0; for (auto& r : rules_) w = std::max(w, r->warmup());
    w = std::max(w, cfg_.reliability_window);
    w = std::max(w, cfg_.diversity.window);
    return w;
  }

  std::optional<float> eval(const BarsView& b, int64_t i,
                            std::optional<float> realized_next_logret,
                            EnsembleMeta* meta=nullptr)
  {
    if (i < warmup()) return std::nullopt;

    raw_p_.clear(); w_eff_.clear();
    const size_t R = rules_.size();

    for (size_t r=0;r<R;r++){
      auto out = rules_[r]->eval(b, i);
      if (!out) { raw_p_.push_back(std::numeric_limits<float>::quiet_NaN()); continue; }
      float p = to_probability(*out, cfg_.score_logistic_k);
      raw_p_.push_back(std::clamp(p, cfg_.eps_clip, 1.f-cfg_.eps_clip));
    }
    if (cfg_.use_diversity){
      std::vector<float> snap(R, 0.5f);
      for (size_t r=0;r<R;r++) snap[r] = std::isfinite(raw_p_[r])? raw_p_[r] : 0.5f;
      div_.update(snap);
      auto w_div = div_.weights();
      for (size_t r=0;r<R;r++){
        if (!std::isfinite(raw_p_[r])) continue;
        float w = cfg_.base_weights[r] * reliability_weight_(r) * w_div[r];
        w_eff_.push_back(std::max(0.f,w));
      }
    } else {
      for (size_t r=0;r<R;r++){
        if (!std::isfinite(raw_p_[r])) continue;
        float w = cfg_.base_weights[r] * reliability_weight_(r);
        w_eff_.push_back(std::max(0.f,w));
      }
    }

    vec_p_.clear(); vec_w_.clear();
    for (size_t r=0, k=0; r<R; r++){
      if (!std::isfinite(raw_p_[r])) continue;
      vec_p_.push_back(raw_p_[r]);
      vec_w_.push_back(w_eff_[k++]);
    }
    if ((int)vec_p_.size() < cfg_.min_rules) return std::nullopt;

    float sw=0.f, sp=0.f;
    for (size_t k=0;k<vec_p_.size();k++){ sw += vec_w_[k]; sp += vec_w_[k]*vec_p_[k]; }
    if (sw<=0.f) return std::nullopt;
    float p_mean = sp / sw;

    float vote = 0.f; for (float p: vec_p_) vote += (p>=0.5f? +1.f : -1.f);
    float agree = std::fabs(vote) / (float)vec_p_.size();
    float boost = 1.f + cfg_.agreement_boost * (agree - 0.5f);
    float p_boosted = std::clamp(0.5f + (p_mean - 0.5f)*boost, cfg_.eps_clip, 1.f - cfg_.eps_clip);

    float p_final = p_boosted;
    if (cfg_.use_platt){ p_final = platt_.calibrate_from_p(p_boosted); }

    if (meta){
      meta->n_used = (int)vec_p_.size();
      meta->agreement = agree; meta->weighted_mean = p_mean;
      float m=0; for(float p:vec_p_) m+=p; m/=vec_p_.size();
      float v=0; for(float p:vec_p_){ float d=p-m; v+=d*d; } v/=std::max<size_t>(1,vec_p_.size()-1);
      meta->variance = v; meta->member_p = vec_p_; meta->member_w = vec_w_;
    }

    if (realized_next_logret){
      float target = (*realized_next_logret > 0.f) ? 1.f : 0.f;
      for (size_t r=0;r<R;r++){
        auto out = rules_[r]->eval(b, i);
        if (!out) continue;
        float p = to_probability(*out, cfg_.score_logistic_k);
        update_reliability_(r, p, target);
      }
      if (cfg_.use_platt){
        float pb = std::clamp(p_boosted, 1e-6f, 1.f-1e-6f);
        float zb = std::log(pb/(1.f-pb));
        platt_.update(zb, target);
      }
    }
    return p_final;
  }

private:
  float reliability_weight_(size_t r) const {
    const auto& B = reli_brier_[r]; const auto& H = reli_hits_[r];
    if (B.empty()) return 1.0f;
    float brier = mean_(B);
    float w_brier = std::clamp(1.5f - brier*3.0f, 0.25f, 2.0f);
    float hit = H.empty()? 0.5f : mean_(H);
    float w_hit = std::clamp(0.5f + (hit-0.5f)*1.0f, 0.25f, 2.0f);
    return 0.5f*w_brier + 0.5f*w_hit;
  }
  void update_reliability_(size_t r, float p, float target){
    float brier = (p - target)*(p - target);
    push_window_(reli_brier_[r], brier, cfg_.reliability_window);
    float hit = ((p>=0.5f) == (target>=0.5f)) ? 1.f : 0.f;
    push_window_(reli_hits_[r], hit, cfg_.reliability_window);
  }
  static void push_window_(std::vector<float>& v, float x, int W){ v.push_back(x); if ((int)v.size()>W) v.erase(v.begin()); }
  static float mean_(const std::vector<float>& v){ if (v.empty()) return 0.f; float s=0.f; for(float x:v) s+=x; return s/v.size(); }

  EnsembleConfig cfg_;
  std::vector<std::unique_ptr<IRuleStrategy>> rules_;
  std::vector<std::vector<float>> reli_brier_, reli_hits_;
  DiversityWeighter div_;
  OnlinePlatt       platt_;
  std::vector<float> raw_p_, w_eff_, vec_p_, vec_w_;
};

} // namespace sentio::rules


