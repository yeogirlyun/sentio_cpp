#include "sentio/signal_trace.hpp"

namespace sentio {
std::size_t SignalTrace::count(TraceReason r) const {
  std::size_t n=0; for (auto& x: rows_) if (x.reason==r) ++n; return n;
}
} // namespace sentio
