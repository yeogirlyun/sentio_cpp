#pragma once
#include "core.hpp"
#include "audit.hpp"
#include "router.hpp"
#include "safe_sizer.hpp"
// REMOVED: position_manager.hpp - unused legacy file
#include "cost_model.hpp"
#include "symbol_table.hpp"
#include "dataset_metadata.hpp"
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration for canonical evaluation
#include "canonical_evaluation.hpp"

namespace sentio {
    class StrategyProfiler;  // Forward declaration
}

namespace sentio {

enum class AuditLevel { Full, MetricsOnly };

struct RunnerCfg {
    std::string strategy_name = "VWAPReversion";
    std::unordered_map<std::string, std::string> strategy_params;
    RouterCfg router;
    SafeSizerConfig sizer;
    AuditLevel audit_level = AuditLevel::Full;
    int snapshot_stride = 100;
    std::string audit_file = "audit.jsonl";  // JSONL audit file path
    bool skip_audit_run_creation = false;  // Skip audit run creation (for block processing)
};

// NEW: This struct holds the RAW output from a backtest simulation.
// It does not contain any calculated performance metrics.
struct BacktestOutput {
    std::vector<std::pair<std::string, double>> equity_curve;
    // Canonical: raw timestamps aligned with equity_curve entries (milliseconds since epoch)
    std::vector<std::int64_t> equity_curve_ts_ms;
    int total_fills = 0;
    int no_route_events = 0;
    int no_qty_events = 0;
    int run_trading_days = 0;
};

// REMOVED: The old RunResult struct is now obsolete.
// struct RunResult { ... };

// CHANGED: run_backtest now returns the raw BacktestOutput and accepts dataset metadata.
BacktestOutput run_backtest(IAuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                           int base_symbol_id, const RunnerCfg& cfg, const DatasetMetadata& dataset_meta = {}, 
                           StrategyProfiler* persistent_profiler = nullptr);

// NEW: Canonical evaluation using Trading Block system for deterministic performance measurement
CanonicalReport run_canonical_backtest(IAuditRecorder& audit, const SymbolTable& ST, 
                                      const std::vector<std::vector<Bar>>& series, int base_symbol_id, 
                                      const RunnerCfg& cfg, const DatasetMetadata& dataset_meta, 
                                      const TradingBlockConfig& block_config);

} // namespace sentio

