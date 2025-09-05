#include "../include/sentio/sanity.hpp"
#include "../include/sentio/sim_data.hpp"
#include "../include/sentio/property_test.hpp"
#include <unordered_map>

// Test PriceBook implementation that works with sanity checks
struct TestPriceBook : sentio::PriceBook {
  std::unordered_map<std::string, sentio::Bar> latest;
  void upsert_latest(const std::string& i, const sentio::Bar& b) override { latest[i]=b; }
  const sentio::Bar* get_latest(const std::string& i) const override {
    auto it=latest.find(i); return it==latest.end()?nullptr:&it->second;
  }
  bool has_instrument(const std::string& i) const override { return latest.count(i)>0; }
  std::size_t size() const override { return latest.size(); }
};

int main(){
  using namespace sentio;

  std::cout << "Starting sanity test..." << std::endl;

  // 1) Generate clean synthetic data
  SimCfg cfg; cfg.minutes=300; cfg.seed=1337;
  auto bars = generate_minute_series(cfg);
  std::cout << "Generated " << bars.size() << " bars" << std::endl;

  // 2) Data sanity
  SanityReport rep;
  sanity::check_bar_monotonic(bars, /*expected 60 sec*/60, rep);
  sanity::check_bar_values_finite(bars, rep);
  std::cout << "Data sanity check: " << (rep.ok() ? "PASS" : "FAIL") << std::endl;
  if (!rep.ok()) {
    std::cout << "Issues found: " << rep.errors() << " errors, " << rep.fatals() << " fatals" << std::endl;
    return 1;
  }

  // 3) Load into PriceBook (as if routed to QQQ/TQQQ)
  TestPriceBook pb;
  for (auto& it : bars) {
    pb.upsert_latest("QQQ", it.second);
    // make TQQQ a simple 3x price proxy for the test
    auto b = it.second; b.open*=3; b.high*=3; b.low*=3; b.close*=3;
    pb.upsert_latest("TQQQ", b);
  }
  std::cout << "Loaded " << pb.size() << " instruments into PriceBook" << std::endl;
  sanity::check_pricebook_coherence(pb, {"QQQ","TQQQ"}, rep);
  std::cout << "PriceBook coherence check: " << (rep.ok() ? "PASS" : "FAIL") << std::endl;
  if (!rep.ok()) {
    std::cout << "Issues found: " << rep.errors() << " errors, " << rep.fatals() << " fatals" << std::endl;
    return 1;
  }

  // 4) P&L invariant property: equity must equal cash+realized+mtm
  AccountState acct{.cash=10'000.0, .realized=0.0, .equity=10'000.0};
  std::unordered_map<std::string, Position> pos;
  // open a fake long 100 sh TQQQ @ first close
  double first_px = pb.get_latest("TQQQ")->close; // latest after loop — for demo only
  pos["TQQQ"] = Position{100.0, first_px};
  std::cout << "Created position: 100 TQQQ @ " << first_px << std::endl;
  // mark with latest
  sanity::check_equity_consistency(acct, pos, pb, rep);
  std::cout << "Initial equity check: " << (rep.ok() ? "PASS" : "FAIL") << std::endl;
  // This will likely flag mismatch because we didn't recompute equity;
  // fix by recomputing equity for the test:
  double mtm = pos["TQQQ"].qty * pb.get_latest("TQQQ")->close;
  acct.equity = acct.cash + acct.realized + mtm;
  std::cout << "Recomputed equity: " << acct.equity << " (cash=" << acct.cash << ", realized=" << acct.realized << ", mtm=" << mtm << ")" << std::endl;
  rep.issues.clear();
  sanity::check_equity_consistency(acct, pos, pb, rep);
  std::cout << "Final equity check: " << (rep.ok() ? "PASS" : "FAIL") << std::endl;
  if (!rep.ok()) {
    std::cout << "Issues found: " << rep.errors() << " errors, " << rep.fatals() << " fatals" << std::endl;
    return 1;
  }

  // 5) Quick property pack
  std::cout << "Running property tests..." << std::endl;
  std::vector<PropCase> props;
  props.push_back({"bar spacing is 60s",
    [&](){
      for (size_t i=1;i<bars.size();++i)
        if (bars[i].first - bars[i-1].first != 60) return false;
      return true;
    }});
  props.push_back({"prices are positive", [&](){
      for (auto& p : bars) {
        auto b = p.second;
        if (b.open<=0 || b.high<=0 || b.low<=0 || b.close<=0) return false;
      } return true;
    }});

  int result = run_properties(props);
  std::cout << "Property tests completed with result: " << result << std::endl;
  return result;
}
