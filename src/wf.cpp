// wf.cpp
// Walk-Forward driver with safe training loop, always-OOS policy, and robust windowing.
// Drop-in: requires C++20. Plug your own hooks where marked TODO.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

// -------------------------------
// Minimal date helper (day granularity)
// Replace with your own if you already have Date utilities.
// -------------------------------
struct Date {
  // days since epoch (UTC or NYT midnight—consistent across your dataset)
  int64_t days{};
  friend bool operator<(const Date& a, const Date& b){ return a.days < b.days; }
  friend bool operator<=(const Date& a, const Date& b){ return a.days <= b.days; }
  friend bool operator>(const Date& a, const Date& b){ return a.days > b.days; }
  friend bool operator>=(const Date& a, const Date& b){ return a.days >= b.days; }
  friend bool operator==(const Date& a, const Date& b){ return a.days == b.days; }
};
inline Date operator+(Date d, int add_days){ d.days += add_days; return d; }
inline Date days(int n){ return Date{n}; } // convenience

// -------------------------------
// WF config and data slice types
// -------------------------------
struct WfCfg {
  int train_days = 252 * 2; // ~2y
  int oos_days   = 63;      // ~1 quarter
  int step_days  = 63;      // slide by a quarter
};

struct DataSlice {
  // Your implementation should carry pointers/indices into your bar arrays.
  // For this template, we just keep [start,end) in "days since epoch".
  Date start{};
  Date end{};
  bool has_data() const { return start < end; }
};

// -------------------------------
// Params & hashing (replace with your strategy params)
// -------------------------------
struct Params {
  // Example parameters — replace with your actual fields.
  int   short_win = 20;
  int   long_win  = 100;
  double stop_bps = 50.0;
  friend bool operator==(const Params& a, const Params& b){
    return a.short_win==b.short_win && a.long_win==b.long_win && std::fabs(a.stop_bps-b.stop_bps)<1e-12;
  }
};

struct ParamsHash {
  size_t operator()(const Params& p) const noexcept {
    // simple FNV-1a style
    auto mix = [](size_t h, size_t v){ return (h ^ v) * 1099511628211ULL; };
    size_t h = 1469598103934665603ULL;
    h = mix(h, std::hash<int>{}(p.short_win));
    h = mix(h, std::hash<int>{}(p.long_win));
    h = mix(h, std::hash<long long>{}((long long)std::llround(p.stop_bps*10000)));
    return h;
  }
};

// -------------------------------
// WF window builder (bulletproof)
// -------------------------------
inline void validate_wf_cfg(const WfCfg& w){
  if (w.train_days <= 0) throw std::invalid_argument("train_days must be > 0");
  if (w.oos_days   <= 0) throw std::invalid_argument("oos_days must be > 0");
  if (w.step_days  <= 0) throw std::invalid_argument("step_days must be > 0");
}

inline std::vector<std::tuple<Date,Date,Date>>  // {train_start, train_end, oos_end}
make_wf_windows(Date start, Date end, const WfCfg& w){
  validate_wf_cfg(w);
  std::vector<std::tuple<Date,Date,Date>> out;
  if (!(start < end)) return out;

  Date cur = start;
  const int step = std::max(1, w.step_days);

  for (size_t guard=0; cur < end; ++guard){
    auto train_end = cur + w.train_days;
    auto oos_end   = train_end + w.oos_days;
    if (!(train_end < end) || !(oos_end <= end)) break;

    out.emplace_back(cur, train_end, oos_end);

    Date next = cur + step;
    if (!(next > cur)) throw std::runtime_error("WF window builder: next <= cur (no progress)");
    cur = next;

    if (guard > 10'000'000ULL) throw std::runtime_error("WF window builder: guard tripped");
  }
  return out;
}

// -------------------------------
// Training criteria & result
// -------------------------------
struct TrainCriteria {
  int    max_iters      = 200;    // absolute upper bound
  int    patience       = 20;     // stop after this many non-improving iters
  double min_delta      = 1e-4;   // min improvement to reset patience
  double target_metric  = 0.0;    // stop early if >= target (0 disables)
  int64_t time_budget_ms = 0;     // wall-clock cap (0 disables)
};

struct TrainResult {
  bool   success = false;   // met target or improved at least once
  int    iters   = 0;
  double best_metric = -1e300;
  Params best_params;
};

// -------------------------------
// Progress guard (Debug use)
// -------------------------------
struct ProgressGuard {
  uint64_t same_count = 0, limit = 1000000ULL;
  uint64_t last_token = 0;
  void step(uint64_t token){
    if (token == last_token) { if (++same_count > limit) throw std::runtime_error("ProgressGuard: no progress"); }
    else { last_token = token; same_count = 0; }
  }
};

// -------------------------------
// Safe training loop
// - evaluate: Params -> metric (higher is better)
// - propose_next: (last_params, last_metric) -> new Params (should usually differ)
// -------------------------------
inline TrainResult train_until(
    const TrainCriteria& cfg,
    Params init,
    const std::function<double(const Params&)>& evaluate,
    const std::function<Params(const Params&, double)>& propose_next)
{
  using clock = std::chrono::high_resolution_clock;
  const auto t0 = clock::now();

  TrainResult out;
  out.best_params = init;

  int no_improve = 0;
  int iter = 0;

  std::unordered_set<Params, ParamsHash> seen; seen.reserve(cfg.max_iters*2);
  ProgressGuard guard;

  Params cur = init;
  double cur_metric = evaluate(cur);
  out.best_metric = cur_metric;
  bool improved_once = false;

  seen.insert(cur);
  guard.step(ParamsHash{}(cur));

  if (cfg.target_metric > 0.0 && cur_metric >= cfg.target_metric) {
    out.success = true; out.iters = 1; return out;
  }

  for (iter = 2; iter <= cfg.max_iters; ++iter) {
    // Time budget check
    if (cfg.time_budget_ms > 0) {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
      if (ms >= cfg.time_budget_ms) break;
    }

    // Propose new params
    Params next = propose_next(cur, cur_metric);

    // If duplicate, try one more proposal; if still dup → accept best-so-far
    if (!seen.insert(next).second) {
      next = propose_next(next, cur_metric);
      if (!seen.insert(next).second) break;
    }

    double next_metric = evaluate(next);

    // Improvement logic
    if (next_metric >= out.best_metric + cfg.min_delta) {
      out.best_metric = next_metric;
      out.best_params = next;
      improved_once = true;
      no_improve = 0;
    } else {
      if (++no_improve >= cfg.patience) break;
    }

    if (cfg.target_metric > 0.0 && out.best_metric >= cfg.target_metric) break;

    // Guard against “no progress”
    guard.step(ParamsHash{}(next) ^ (uint64_t)iter);

    cur = next;
    cur_metric = next_metric;
  }

  out.iters = iter - 1;
  out.success = improved_once || (cfg.target_metric > 0.0 && out.best_metric >= cfg.target_metric);
  return out;
}

// -------------------------------
// OOS result container
// -------------------------------
struct OOSResult {
  bool   valid  = true;
  double metric = std::nan("");   // e.g., Sharpe/return
  std::string reason = "ok";
};

// -------------------------------
// Fold outcome (training + OOS); OOS always runs unless invalid data/params
// -------------------------------
struct FoldOutcome {
  bool training_success = false;
  int  train_iters      = 0;
  double train_metric_best = -1e300;

  Params params_used;
  bool   params_from_prev = false;

  bool   oos_valid = false;
  double oos_metric = std::nan("");
  std::string oos_reason = "uninit";
};

// -------------------------------
// Run a single WF fold
// Hooks you must provide via std::function (plug at call site):
//   train_metric(train_slice, params) -> double
//   run_oos(oos_slice, params) -> OOSResult
//   propose(last_params, last_metric) -> Params
//   params_valid(params) -> bool
//   baseline_params() -> Params
// -------------------------------
inline FoldOutcome run_wf_fold(
    const DataSlice& train, const DataSlice& oos,
    const Params& init, const TrainCriteria& crit,
    const std::optional<Params>& prev_fold_params,

    const std::function<double(const DataSlice&, const Params&)>& train_metric,
    const std::function<OOSResult(const DataSlice&, const Params&)>& run_oos,
    const std::function<Params(const Params&, double)>& propose,
    const std::function<bool(const Params&)>& params_valid,
    const std::function<Params(void)>& baseline_params
){
  FoldOutcome R{};

  auto eval_train = [&](const Params& p)->double { return train_metric(train, p); };
  auto tr = train_until(crit, init, eval_train, propose);

  R.training_success   = tr.success;
  R.train_iters        = tr.iters;
  R.train_metric_best  = tr.best_metric;
  R.params_used        = tr.best_params;

  // Fallback params if training didn't meet criteria
  if (!R.training_success) {
    if (prev_fold_params) {
      R.params_used      = *prev_fold_params;
      R.params_from_prev = true;
    } else if (!params_valid(R.params_used)) {
      R.params_used = baseline_params();
    }
  }

  // OOS always runs unless we truly cannot evaluate
  if (!oos.has_data()) {
    R.oos_valid = false;
    R.oos_reason = "no_data";
    return R;
  }
  if (!params_valid(R.params_used)) {
    R.oos_valid = false;
    R.oos_reason = "invalid_params";
    return R;
  }

  auto res = run_oos(oos, R.params_used);
  R.oos_valid  = res.valid;
  R.oos_metric = res.metric;
  R.oos_reason = res.reason.empty()? "ok" : res.reason;

  return R;
}

// -------------------------------
// Orchestrate full WF run (optionally parallel across folds)
// -------------------------------
struct WfRunSummary {
  int folds = 0;
  int folds_ok = 0;
  double oos_metric_sum = 0.0;
  std::vector<FoldOutcome> outcomes;
};

inline WfRunSummary run_walk_forward(
    Date global_start, Date global_end, const WfCfg& cfg,
    const Params& init, const TrainCriteria& crit,
    const std::function<DataSlice(Date,Date)>& get_slice, // build slices from dates
    const std::function<double(const DataSlice&, const Params&)>& train_metric,
    const std::function<OOSResult(const DataSlice&, const Params&)>& run_oos,
    const std::function<Params(const Params&, double)>& propose,
    const std::function<bool(const Params&)>& params_valid,
    const std::function<Params(void)>& baseline_params,
    int max_concurrency = std::max(1u, std::thread::hardware_concurrency()-1)
){
  auto windows = make_wf_windows(global_start, global_end, cfg);
  const int K = (int)windows.size();
  WfRunSummary sum; sum.folds = K; sum.outcomes.resize(K);

  std::mutex m_prev;
  std::optional<Params> prev_params{};

  // simple async throttle
  std::deque<std::pair<int, std::future<FoldOutcome>>> q;

  auto submit = [&](int idx){
    auto [ts, tr_end, oos_end] = windows[idx];
    DataSlice train{ts, tr_end};
    DataSlice oos  {tr_end, oos_end};

    Params init_for_fold = init;
    {
      std::lock_guard<std::mutex> lk(m_prev);
      if (prev_params) init_for_fold = *prev_params; // warm start with last
    }

    return std::async(std::launch::async, [=, &get_slice, &train_metric, &run_oos, &propose, &params_valid, &baseline_params]{
      // materialize slices (or pass views) from your data store
      auto train_slice = get_slice(train.start, train.end);
      auto oos_slice   = get_slice(oos.start, oos.end);

      auto out = run_wf_fold(train_slice, oos_slice, init_for_fold, crit, prev_params,
                             train_metric, run_oos, propose, params_valid, baseline_params);
      return out;
    });
  };

  auto drain_one = [&](){
    auto idx = q.front().first;
    auto& fut = q.front().second;
    auto out = fut.get();
    sum.outcomes[idx] = out;
    if (out.oos_valid) { sum.folds_ok++; sum.oos_metric_sum += out.oos_metric; }
    // record last successful params as warm start
    if (out.training_success) {
      std::lock_guard<std::mutex> lk(m_prev);
      prev_params = out.params_used;
    }
    q.pop_front();
  };

  for (int i=0; i<K; ++i) {
    q.emplace_back(i, submit(i));
    if ((int)q.size() >= max_concurrency) {
      q.front().second.wait();
      drain_one();
    }
  }
  while (!q.empty()) {
    q.front().second.wait();
    drain_one();
  }

  return sum;
}

// -------------------------------
// Example usage (wire your hooks here)
// Remove main() and integrate into your app as needed.
// -------------------------------
#ifdef WF_EXAMPLE_MAIN
// Mock hooks (replace with real ones)
static double mock_train_metric(const DataSlice& s, const Params& p){
  // pretend Sharpe improves with longer windows up to a point
  double len = double(s.end.days - s.start.days);
  double score = (p.long_win > p.short_win ? 1.0 : -1.0) * std::tanh(len/500.0) - std::abs(p.stop_bps-50.0)/200.0;
  return score;
}
static OOSResult mock_run_oos(const DataSlice& s, const Params& p){
  (void)p;
  if (!s.has_data()) return {false, std::nan(""), "no_data"};
  double len = double(s.end.days - s.start.days);
  return {true, 0.5 * std::tanh(len/200.0), "ok"};
}
static Params mock_propose(const Params& last, double){
  Params n = last;
  // simple random-ish walk (deterministic tweak here)
  n.short_win = std::max(5, std::min(60, n.short_win + ((n.short_win%2)? +1 : -1)));
  n.long_win  = std::max(20, std::min(240, n.long_win  + ((n.long_win%3)? +2 : -2)));
  n.stop_bps  = std::clamp(n.stop_bps + ((n.stop_bps>50)? -5.0 : +5.0), 5.0, 200.0);
  return n;
}
static bool mock_params_valid(const Params& p){
  return p.short_win > 0 && p.long_win > p.short_win && p.stop_bps > 0.0;
}
static Params mock_baseline(){ return Params{20, 100, 50.0}; }
static DataSlice mock_get_slice(Date a, Date b){ return DataSlice{a,b}; }

int main(){
  WfCfg cfg; cfg.train_days=240; cfg.oos_days=60; cfg.step_days=60;
  TrainCriteria crit; crit.max_iters=60; crit.patience=10; crit.min_delta=1e-3; crit.target_metric=0.0;
  Params init{20,100,50.0};

  auto sum = run_walk_forward(
    /*global_start*/ Date{0}, /*global_end*/ Date{2000}, cfg, init, crit,
    mock_get_slice,
    mock_train_metric,
    mock_run_oos,
    mock_propose,
    mock_params_valid,
    mock_baseline,
    /*max_concurrency*/ 2
  );

  std::cout << "WF folds=" << sum.folds
            << " ok=" << sum.folds_ok
            << " avg_oos=" << (sum.folds_ok? sum.oos_metric_sum / sum.folds_ok : 0.0)
            << "\n";
  return 0;
}
#endif

// -------------------------------
// Bridge functions to match existing API
// -------------------------------
#include "sentio/wf.hpp"
#include "sentio/runner.hpp"
#include "sentio/audit.hpp"
#include "sentio/metrics.hpp"
#include <iostream>
#include <algorithm>
#include <vector>

namespace sentio {

// Convert bar index to date (simplified)
Date bar_index_to_date(int index) {
    // Assuming 390 bars per day (6.5 hours * 60 minutes)
    return Date{index / 390};
}

// Convert date to bar index (simplified)
int date_to_bar_index(Date date) {
    return date.days * 390;
}

// Bridge function to match existing API
Gate run_wf_and_gate(AuditRecorder& audit_template,
                     const SymbolTable& ST,
                     const std::vector<std::vector<Bar>>& series,
                     int base_symbol_id,
                     const RunnerCfg& rcfg,
                     const WfCfg& wcfg) {
    
    Gate gate;
    
    // Validate step_days to prevent infinite loop
    if (wcfg.step_days <= 0) {
        std::cerr << "ERROR: Walk-forward step_days must be positive. Got: "
                  << wcfg.step_days << std::endl;
        gate.recommend = "REJECT - INVALID CONFIG";
        return gate;
    }
    
    const auto& base_series = series[base_symbol_id];
    const int total_bars = (int)base_series.size();
    if (total_bars < wcfg.train_days + wcfg.oos_days) {
        std::cerr << "Insufficient data for walk-forward test" << std::endl;
        return gate;
    }
    
    std::vector<double> oos_returns;
    [[maybe_unused]] int successful_folds = 0;
    
    // Walk-forward testing
    int total_iterations = (total_bars - wcfg.train_days - wcfg.oos_days) / wcfg.step_days;
    int current_iteration = 0;
    
    std::cerr << "Starting WF test with " << total_iterations << " iterations..." << std::endl;
    
    for (int start_idx = 0; start_idx < total_bars - wcfg.train_days - wcfg.oos_days; start_idx += wcfg.step_days) {
        current_iteration++;
        if (current_iteration % 10 == 0) {
            std::cerr << "Progress: " << current_iteration << "/" << total_iterations << " iterations completed" << std::endl;
        }
        
        int train_end = start_idx + wcfg.train_days;
        int oos_start = train_end;
        int oos_end = std::min(oos_start + wcfg.oos_days, total_bars);
        
        if (oos_end - oos_start < 5) continue; // Skip if insufficient OOS data
        
        // Create training data slice
        std::vector<std::vector<Bar>> train_series = series;
        for (auto& sym_series : train_series) {
            if (sym_series.size() > static_cast<size_t>(train_end)) {
                sym_series.resize(train_end);
            }
        }
        
        // Create OOS data slice
        std::vector<std::vector<Bar>> oos_series = series;
        for (auto& sym_series : oos_series) {
            if (sym_series.size() > static_cast<size_t>(oos_end)) {
                sym_series.erase(sym_series.begin(), sym_series.begin() + oos_start);
                sym_series.resize(oos_end - oos_start);
            }
        }
        
        // Run training backtest
        if (current_iteration % 50 == 0) {
            std::cerr << "Running training backtest for iteration " << current_iteration << "..." << std::endl;
        }
        
        // Create new audit recorder for training
        AuditConfig train_cfg = audit_template.get_config();
        train_cfg.run_id = "wf_train_" + std::to_string(start_idx);
        train_cfg.file_path = "audit/wf_train_" + std::to_string(start_idx) + ".jsonl";
        AuditRecorder au_train(train_cfg);
        
        auto train_result = run_backtest(au_train, ST, train_series, base_symbol_id, rcfg);
        
        // Always run OOS backtest (removed gate mechanism)
        AuditConfig oos_cfg = audit_template.get_config();
        oos_cfg.run_id = "wf_oos_" + std::to_string(start_idx);
        oos_cfg.file_path = "audit/wf_oos_" + std::to_string(start_idx) + ".jsonl";
        AuditRecorder au_oos(oos_cfg);
        
        auto oos_result = run_backtest(au_oos, ST, oos_series, base_symbol_id, rcfg);
        
        // Store OOS results
        oos_returns.push_back(oos_result.total_return);
        successful_folds++;
        
        // Track if any training period was profitable (for WF pass criteria)
        if (train_result.sharpe_ratio > 0.1 && train_result.total_return > -0.1) {
            gate.wf_pass = true;
        }
    }
    
    // Calculate OOS statistics
    if (!oos_returns.empty()) {
        double avg_return = 0.0;
        for (double ret : oos_returns) avg_return += ret;
        avg_return /= oos_returns.size();
        
        gate.oos_monthly_avg_return = avg_return * 12.0; // Annualized
        gate.oos_pass = gate.oos_monthly_avg_return > 0.0;
        
        // Calculate Sharpe ratio (simplified)
        double variance = 0.0;
        for (double ret : oos_returns) {
            double diff = ret - avg_return;
            variance += diff * diff;
        }
        variance /= oos_returns.size();
        double std_dev = std::sqrt(variance);
        
        if (std_dev > 0.0) {
            gate.oos_sharpe = avg_return / std_dev * std::sqrt(252.0);
        }
        
        // Determine recommendation
        if (gate.wf_pass && gate.oos_pass) {
            gate.recommend = "PAPER";
        } else if (gate.wf_pass) {
            gate.recommend = "ITERATE";
        } else {
            gate.recommend = "REJECT";
        }
    }
    
    return gate;
}

// Walk-forward testing with optimization (simplified for now)
Gate run_wf_and_gate_optimized(AuditRecorder& audit_template,
                               const SymbolTable& ST,
                               const std::vector<std::vector<Bar>>& series,
                               int base_symbol_id,
                               const RunnerCfg& base_rcfg,
                               const WfCfg& wcfg) {
    // For now, just run the basic WF without optimization
    // TODO: Implement parameter optimization
    return run_wf_and_gate(audit_template, ST, series, base_symbol_id, base_rcfg, wcfg);
}

} // namespace sentio