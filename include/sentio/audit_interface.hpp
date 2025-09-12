#pragma once
#include <string>
#include <cstdint>
#include "sentio/signal_diag.hpp"

namespace sentio {

// Forward declarations
struct Bar;
struct AccountState;
enum class SigType : uint8_t;
enum class Side : uint8_t;

// Common interface for all audit recorders
class IAuditRecorder {
public:
    virtual ~IAuditRecorder() = default;
    
    // Run lifecycle events
    virtual void event_run_start(std::int64_t ts, const std::string& meta) = 0;
    virtual void event_run_end(std::int64_t ts, const std::string& meta) = 0;
    
    // Market data events
    virtual void event_bar(std::int64_t ts, const std::string& inst, double open, double high, double low, double close, double volume) = 0;
    
    // Signal events
    virtual void event_signal(std::int64_t ts, const std::string& base, SigType t, double conf) = 0;
    virtual void event_signal_ex(std::int64_t ts, const std::string& base, SigType t, double conf, const std::string& chain_id) = 0;
    
    // Trading events
    virtual void event_route(std::int64_t ts, const std::string& base, const std::string& inst, double tw) = 0;
    virtual void event_route_ex(std::int64_t ts, const std::string& base, const std::string& inst, double tw, const std::string& chain_id) = 0;
    virtual void event_order(std::int64_t ts, const std::string& inst, Side side, double qty, double limit_px) = 0;
    virtual void event_order_ex(std::int64_t ts, const std::string& inst, Side side, double qty, double limit_px, const std::string& chain_id) = 0;
    virtual void event_fill(std::int64_t ts, const std::string& inst, double price, double qty, double fees, Side side) = 0;
    virtual void event_fill_ex(std::int64_t ts, const std::string& inst, double price, double qty, double fees, Side side, 
                              double realized_pnl_delta, double equity_after, double position_after, const std::string& chain_id) = 0;
    
    // Portfolio events
    virtual void event_snapshot(std::int64_t ts, const AccountState& a) = 0;
    
    // Metric events
    virtual void event_metric(std::int64_t ts, const std::string& key, double val) = 0;
    
    // Signal diagnostics events
    virtual void event_signal_diag(std::int64_t ts, const std::string& strategy_name, 
                                  const SignalDiag& diag) = 0;
    virtual void event_signal_drop(std::int64_t ts, const std::string& strategy_name, 
                                  const std::string& symbol, DropReason reason, 
                                  const std::string& chain_id, const std::string& note = "") = 0;
};

} // namespace sentio
