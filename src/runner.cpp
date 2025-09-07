#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/pricebook.hpp"
#include "sentio/metrics.hpp"
#include "sentio/sizer.hpp"
#include "sentio/feature_feeder.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <chrono>

namespace sentio {

RunResult run_backtest(AuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                      int base_symbol_id, const RunnerCfg& cfg) {
    
    // 1. ============== INITIALIZATION ==============
    RunResult result{};
    
    // Start audit run
    std::string meta = "{";
    meta += "\"strategy\":\"" + cfg.strategy_name + "\",";
    meta += "\"base_symbol_id\":" + std::to_string(base_symbol_id) + ",";
    meta += "\"total_series\":" + std::to_string(series.size()) + ",";
    meta += "\"base_series_size\":" + std::to_string(series[base_symbol_id].size());
    meta += "}";
    audit.event_run_start(series[base_symbol_id][0].ts_nyt_epoch, meta);
    
    auto strategy = StrategyFactory::instance().create_strategy(cfg.strategy_name);
    if (!strategy) {
        std::cerr << "FATAL: Could not create strategy '" << cfg.strategy_name << "'. Check registration." << std::endl;
        return result;
    }
    
    ParameterMap params;
    for (const auto& [key, value] : cfg.strategy_params) {
        try {
            params[key] = std::stod(value);
        } catch (...) { /* ignore */ }
    }
    strategy->set_params(params);

    Portfolio portfolio(ST.size());
    AdvancedSizer sizer;
    Pricebook pricebook(base_symbol_id, ST, series);
    
    std::vector<std::pair<std::string, double>> equity_curve;
    const auto& base_series = series[base_symbol_id];
    equity_curve.reserve(base_series.size());

    int total_fills = 0;
    int no_route_count = 0;
    int no_qty_count = 0;

    // 2. ============== MAIN EVENT LOOP ==============
    size_t total_bars = base_series.size();
    size_t progress_interval = total_bars / 20; // 5% intervals (20 steps)
    
    // Skip first 300 bars to allow technical indicators to warm up
    size_t warmup_bars = 300;
    if (total_bars <= warmup_bars) {
        std::cout << "Warning: Not enough bars for warmup (need " << warmup_bars << ", have " << total_bars << ")" << std::endl;
        warmup_bars = 0;
    }
    
    for (size_t i = warmup_bars; i < base_series.size(); ++i) {
        // Progress reporting at 5% intervals
        if (i % progress_interval == 0) {
            int progress_percent = (i * 100) / total_bars;
            std::cout << "Progress: " << progress_percent << "% (" << i << "/" << total_bars << " bars)" << std::endl;
        }
        
        const auto& bar = base_series[i];
        pricebook.sync_to_base_i(i);
        
        // Log bar data
        AuditBar audit_bar{bar.open, bar.high, bar.low, bar.close, static_cast<double>(bar.volume)};
        audit.event_bar(bar.ts_nyt_epoch, ST.get_symbol(base_symbol_id), audit_bar);
        
        // Feed features to ML strategies
        [[maybe_unused]] auto start_feed = std::chrono::high_resolution_clock::now();
        FeatureFeeder::feed_features_to_strategy(strategy.get(), base_series, i, cfg.strategy_name);
        [[maybe_unused]] auto end_feed = std::chrono::high_resolution_clock::now();
        
        [[maybe_unused]] auto start_signal = std::chrono::high_resolution_clock::now();
        StrategySignal sig = strategy->calculate_signal(base_series, i);
        [[maybe_unused]] auto end_signal = std::chrono::high_resolution_clock::now();
        
        if (i < 10) {
            // Timing removed for clean output
        }
        
        if (sig.type != StrategySignal::Type::HOLD) {
            // Log signal
            SigType sig_type = static_cast<SigType>(static_cast<int>(sig.type));
            audit.event_signal(bar.ts_nyt_epoch, ST.get_symbol(base_symbol_id), sig_type, sig.confidence);
            
            auto route_decision = route(sig, cfg.router, ST.get_symbol(base_symbol_id));

            if (route_decision) {
                // Log route decision
                audit.event_route(bar.ts_nyt_epoch, ST.get_symbol(base_symbol_id), route_decision->instrument, route_decision->target_weight);
                
                int instrument_id = ST.get_id(route_decision->instrument);
                if (instrument_id != -1) {
                    double instrument_price = pricebook.last_px[instrument_id];

                    if (instrument_price > 0) {
                        [[maybe_unused]] double current_equity = equity_mark_to_market(portfolio, pricebook.last_px);
                        
                        // **MODIFIED**: This is the core logic fix.
                        // We calculate the desired final position size, then determine the needed trade quantity.
                        double target_qty = sizer.calculate_target_quantity(portfolio, ST, pricebook.last_px, 
                                                                             route_decision->instrument, route_decision->target_weight, 
                                                                             series[instrument_id], cfg.sizer);
                        
                        double current_qty = portfolio.positions[instrument_id].qty;
                        double trade_qty = target_qty - current_qty; // The actual amount to trade

                        if (std::abs(trade_qty * instrument_price) > 1.0) { // Min trade notional $1
                            // Log order and fill
                            Side side = (trade_qty > 0) ? Side::Buy : Side::Sell;
                            audit.event_order(bar.ts_nyt_epoch, route_decision->instrument, side, std::abs(trade_qty), 0.0);
                            audit.event_fill(bar.ts_nyt_epoch, route_decision->instrument, instrument_price, std::abs(trade_qty), 0.0, side);
                            
                            apply_fill(portfolio, instrument_id, trade_qty, instrument_price);
                            total_fills++; // **CRITICAL FIX**: Increment the fills counter
                        } else {
                            no_qty_count++;
                        }
                    }
                }
            } else {
                no_route_count++;
            }
        }
        
        // 3. ============== SNAPSHOT ==============
        if (i % cfg.snapshot_stride == 0 || i == base_series.size() - 1) {
            double current_equity = equity_mark_to_market(portfolio, pricebook.last_px);
            equity_curve.emplace_back(bar.ts_utc, current_equity);
            
            // Log account snapshot
            AccountState state;
            state.cash = portfolio.cash;
            state.equity = current_equity;
            state.realized = 0.0; // TODO: Calculate realized P&L
            audit.event_snapshot(bar.ts_nyt_epoch, state);
        }
    }
    
    // 4. ============== METRICS & DIAGNOSTICS ==============
    strategy->get_diag().print(strategy->get_name().c_str());

    if (equity_curve.empty()) {
        return result;
    }
    
    // **CRITICAL FIX**: Pass the correct `total_fills` to the metrics calculator.
    auto summary = compute_metrics_day_aware(equity_curve, total_fills);

    result.final_equity = equity_curve.empty() ? 100000.0 : equity_curve.back().second;
    result.total_return = summary.ret_total * 100.0;
    result.sharpe_ratio = summary.sharpe;
    result.max_drawdown = summary.mdd * 100.0;
    result.total_fills = summary.trades;
    result.no_route = no_route_count;
    result.no_qty = no_qty_count;

    // Log final metrics and end run
    std::int64_t end_ts = equity_curve.empty() ? series[base_symbol_id][0].ts_nyt_epoch : series[base_symbol_id].back().ts_nyt_epoch;
    audit.event_metric(end_ts, "final_equity", result.final_equity);
    audit.event_metric(end_ts, "total_return", result.total_return);
    audit.event_metric(end_ts, "sharpe_ratio", result.sharpe_ratio);
    audit.event_metric(end_ts, "max_drawdown", result.max_drawdown);
    audit.event_metric(end_ts, "total_fills", result.total_fills);
    audit.event_metric(end_ts, "no_route", result.no_route);
    audit.event_metric(end_ts, "no_qty", result.no_qty);
    
    std::string end_meta = "{";
    end_meta += "\"final_equity\":" + std::to_string(result.final_equity) + ",";
    end_meta += "\"total_return\":" + std::to_string(result.total_return) + ",";
    end_meta += "\"sharpe_ratio\":" + std::to_string(result.sharpe_ratio);
    end_meta += "}";
    audit.event_run_end(end_ts, end_meta);

    return result;
}

} // namespace sentio