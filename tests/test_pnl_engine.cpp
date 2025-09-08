#include "../include/sentio/execution/pnl_engine.hpp"
#include <cassert>
#include <vector>
using namespace sentio;

int main() {
  PnLEngine pnl(100000.0);
  pnl.set_price_mode(PnLEngine::PriceMode::Close);

  std::vector<BarPx> bars = {{1,100.0,0,0},{2,101.0,0,0},{3,102.0,0,0}};

  pnl.on_fill(Fill{1,100.0, +10.0, 0.0, "SIM"});
  pnl.on_bar(bars[0]);
  pnl.on_bar(bars[1]);
  pnl.on_fill(Fill{3,102.0, -10.0, 0.0, "SIM"});
  pnl.on_bar(bars[2]);

  auto s = pnl.snapshots().back();
  assert(std::abs(s.realized - 200.0) < 1e-6);
  assert(std::abs(s.position) < 1e-9);
  assert(s.equity > 100000.0);
  return 0;
}
