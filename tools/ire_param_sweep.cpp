#include "sentio/core.hpp"
#include "sentio/csv_loader.hpp"
#include "sentio/data_resolver.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/runner.hpp"
#include "sentio/temporal_analysis.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace sentio;

static bool load_family(const std::string& base_symbol,
                        SymbolTable& ST,
                        std::vector<std::vector<Bar>>& series)
{
  std::vector<std::string> symbols_to_load = {base_symbol, "TQQQ", "SQQQ"};
  for (const auto& sym : symbols_to_load) {
    std::vector<Bar> bars;
    std::string data_path = resolve_csv(sym);
    if (!load_csv(data_path, bars)) {
      std::cerr << "ERROR: Failed to load data for " << sym << " from " << data_path << std::endl;
      return false;
    }
    int symbol_id = ST.intern(sym);
    if (static_cast<size_t>(symbol_id) >= series.size()) series.resize(symbol_id + 1);
    series[symbol_id] = std::move(bars);
  }
  return true;
}

static std::vector<std::vector<Bar>> slice_series(const std::vector<std::vector<Bar>>& series,
                                                  int base_sid,
                                                  int start_idx,
                                                  int end_idx)
{
  std::vector<std::vector<Bar>> out; out.reserve(series.size());
  for (const auto& s : series) {
    if ((int)s.size() <= start_idx) { out.emplace_back(); continue; }
    int e = std::min(end_idx, (int)s.size());
    out.emplace_back(s.begin() + start_idx, s.begin() + e);
  }
  return out;
}

int main(int argc, char** argv){
  std::string base_symbol = "QQQ";
  int test_quarters = 1;
  if (argc > 1) base_symbol = argv[1];
  if (argc > 2) test_quarters = std::max(1, std::atoi(argv[2]));

  SymbolTable ST; std::vector<std::vector<Bar>> series;
  if (!load_family(base_symbol, ST, series)) return 1;
  int base_sid = ST.get_id(base_symbol);
  const auto& base = series[base_sid];
  if (base.empty()) { std::cerr << "No data for base." << std::endl; return 1; }

  // Quarter slicing
  int total_bars = (int)base.size();
  int approx_quarters = std::max(1, total_bars / std::max(1, total_bars / 4));
  // pick bars_per_quarter ~ total_bars / 4 if test_quarters==1; more generally use 4 logical quarters per year approximation
  // Use the same bars_per_quarter as main TPA: divide equally by (train+test)
  int total_quarters = 12; // assume 12 logical quarters (~3 years on minute bars)
  int bars_per_quarter = total_bars / total_quarters;
  int test_bars = std::min(total_bars, bars_per_quarter * test_quarters);
  int train_bars = std::max(0, total_bars - test_bars);

  auto train_series = slice_series(series, base_sid, 0, train_bars);
  auto test_series  = slice_series(series, base_sid, train_bars, total_bars);

  // Parameter grids
  std::vector<double> buy_hi_grid   = {0.75, 0.80, 0.85};
  std::vector<double> sell_lo_grid  = {0.25, 0.20, 0.15};
  std::vector<double> strong_short_conf_grid = {0.85, 0.90, 0.95};

  struct Candidate { double buy_hi, sell_lo, short_conf; double score; double avg_trades; double avg_monthly; double sharpe; } best{0,0,0,-1,0,0,0};

  for (double bh : buy_hi_grid){
    for (double sl : sell_lo_grid){
      for (double sc : strong_short_conf_grid){
        RunnerCfg rcfg; rcfg.strategy_name = "IRE";
        rcfg.strategy_params = { {"buy_hi", std::to_string(bh)}, {"sell_lo", std::to_string(sl)} };
        rcfg.router.ire_min_conf_strong_short = sc;

        TemporalAnalysisConfig tcfg; tcfg.num_quarters = std::max(1, (int)std::round((double)train_bars / std::max(1, bars_per_quarter)));
        tcfg.print_detailed_report = false;
        auto train_summary = run_temporal_analysis(ST, train_series, base_sid, rcfg, tcfg);
        double avg_trades = 0.0; double avg_monthly = 0.0; double avg_sharpe = 0.0;
        if (!train_summary.quarterly_results.empty()){
          for (const auto& q : train_summary.quarterly_results){ avg_trades += q.avg_daily_trades; avg_monthly += q.monthly_return_pct; avg_sharpe += q.sharpe_ratio; }
          avg_trades /= train_summary.quarterly_results.size();
          avg_monthly /= train_summary.quarterly_results.size();
          avg_sharpe /= train_summary.quarterly_results.size();
        }
        // Objective: within [80,120] daily trades, maximize avg_monthly with Sharpe bonus
        double trade_penalty = 0.0;
        if (avg_trades < 80) trade_penalty = (80 - avg_trades);
        else if (avg_trades > 120) trade_penalty = (avg_trades - 120);
        double score = avg_monthly + 0.5 * avg_sharpe - 0.1 * trade_penalty;
        if (score > best.score){ best = {bh, sl, sc, score, avg_trades, avg_monthly, avg_sharpe}; }
        std::cout << "Grid bh="<<bh<<" sl="<<sl<<" sc="<<sc<<" -> trades="<<avg_trades<<" mret="<<avg_monthly<<" sharpe="<<avg_sharpe<<" score="<<score<<"\n";
      }
    }
  }

  std::cout << "\nBest params: buy_hi="<<best.buy_hi<<" sell_lo="<<best.sell_lo<<" short_conf="<<best.short_conf
            <<" | trades="<<best.avg_trades<<" mret="<<best.avg_monthly<<" sharpe="<<best.sharpe<<"\n";

  // Run test with best params on most recent quarters
  RunnerCfg rcfg_best; rcfg_best.strategy_name = "IRE";
  rcfg_best.strategy_params = { {"buy_hi", std::to_string(best.buy_hi)}, {"sell_lo", std::to_string(best.sell_lo)} };
  rcfg_best.router.ire_min_conf_strong_short = best.short_conf;
  TemporalAnalysisConfig tcfg_test; tcfg_test.num_quarters = test_quarters; tcfg_test.print_detailed_report = true;
  auto test_summary = run_temporal_analysis(ST, test_series, base_sid, rcfg_best, tcfg_test);
  test_summary.assess_readiness(tcfg_test);
  return 0;
}


