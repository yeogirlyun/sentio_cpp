#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/strategy_ire.hpp"
#include "sentio/pricebook.hpp"
#include "sentio/metrics.hpp"
#include "sentio/sizer.hpp"
#include "sentio/cost_model.hpp"
#include "sentio/feature_feeder.hpp"
#include "sentio/position_validator.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <chrono>

namespace sentio {

// **RENOVATED HELPER**: Execute target position for any instrument
static void execute_target_position(const std::string& instrument, double target_weight,            
                                   Portfolio& portfolio, const SymbolTable& ST, const Pricebook& pricebook,                    
                                   const AdvancedSizer& sizer, const RunnerCfg& cfg,                
                                   const std::vector<std::vector<Bar>>& series, const Bar& bar,     
                                   const std::string& chain_id, IAuditRecorder& audit, bool logging_enabled, int& total_fills) {
    
    int instrument_id = ST.get_id(instrument);
    if (instrument_id == -1) return;
    
    double instrument_price = pricebook.last_px[instrument_id];
    if (instrument_price <= 0) return;

    // Calculate target quantity using sizer
    double target_qty = sizer.calculate_target_quantity(portfolio, ST, pricebook.last_px, 
                                                       instrument, target_weight, 
                                                       series[instrument_id], cfg.sizer);
    
    // **CONFLICT PREVENTION**: Strategy-level conflict prevention should prevent conflicts
    // No need for smart conflict resolution since strategy checks existing positions
    
    double current_qty = portfolio.positions[instrument_id].qty;
    double trade_qty = target_qty - current_qty;

    // **PROFIT MAXIMIZATION**: Execute any meaningful trade (no dust filter for Governor)
    if (std::abs(trade_qty * instrument_price) > 10.0) { // $10 minimum
        Side side = (trade_qty > 0) ? Side::Buy : Side::Sell;
        
        if (logging_enabled) {
            audit.event_order_ex(bar.ts_utc_epoch, instrument, side, std::abs(trade_qty), 0.0, chain_id);
        }

        // Calculate realized P&L for position changes
        double realized_delta = 0.0;
        const auto& pos_before = portfolio.positions[instrument_id];
        double closing = 0.0;
        if (pos_before.qty > 0 && trade_qty < 0) closing = std::min(std::abs(trade_qty), pos_before.qty);
        if (pos_before.qty < 0 && trade_qty > 0) closing = std::min(std::abs(trade_qty), std::abs(pos_before.qty));
        if (closing > 0.0) {
            if (pos_before.qty > 0) realized_delta = (instrument_price - pos_before.avg_price) * closing;
            else                    realized_delta = (pos_before.avg_price - instrument_price) * closing;
        }

        // **ZERO COSTS FOR TESTING**: Remove transaction costs and slippage
        double fees = 0.0;
        double slippage_cost = 0.0;
        double exec_px = instrument_price; // Perfect execution at market price
        
        apply_fill(portfolio, instrument_id, trade_qty, exec_px);
        // portfolio.cash -= fees; // No fees charged
        
        double equity_after = equity_mark_to_market(portfolio, pricebook.last_px);
        double pos_after = portfolio.positions[instrument_id].qty;
        
        if (logging_enabled) {
            audit.event_fill_ex(bar.ts_utc_epoch, instrument, exec_px, std::abs(trade_qty), fees, side,
                               realized_delta, equity_after, pos_after, chain_id);
        }
        total_fills++;
    }
}

RunResult run_backtest(IAuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                      int base_symbol_id, const RunnerCfg& cfg) {
    
    // 1. ============== INITIALIZATION ==============
    RunResult result{};
    
    const bool logging_enabled = (cfg.audit_level == AuditLevel::Full);
    // Start audit run
    std::string meta = "{";
    meta += "\"strategy\":\"" + cfg.strategy_name + "\",";
    meta += "\"base_symbol_id\":" + std::to_string(base_symbol_id) + ",";
    meta += "\"total_series\":" + std::to_string(series.size()) + ",";
    meta += "\"base_series_size\":" + std::to_string(series[base_symbol_id].size());
    meta += "}";
    if (logging_enabled) audit.event_run_start(series[base_symbol_id][0].ts_utc_epoch, meta);
    
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
    double cumulative_realized_pnl = 0.0;  // Track cumulative realized P&L for audit transparency

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
        
        const auto& bar = base_series[i];
        
        // **DAY TRADING RULE**: Intraday risk management
        if (i > warmup_bars) {
            // Extract hour and minute from UTC timestamp for market close detection
            // Market closes at 4:00 PM ET = 20:00 UTC (EDT) or 21:00 UTC (EST)
            time_t raw_time = bar.ts_utc_epoch;
            struct tm* utc_tm = gmtime(&raw_time);
            int hour_utc = utc_tm->tm_hour;
            int minute_utc = utc_tm->tm_min;
            int month = utc_tm->tm_mon + 1; // tm_mon is 0-based
            
            // Simple DST check: April-October is EDT (20:00 close), rest is EST (21:00 close)
            bool is_edt = (month >= 4 && month <= 10);
            int market_close_hour = is_edt ? 20 : 21;
            
            // Calculate minutes until market close
            int current_minutes = hour_utc * 60 + minute_utc;
            int close_minutes = market_close_hour * 60; // Market close time in minutes
            int minutes_to_close = close_minutes - current_minutes;
            
            // Handle day wrap-around (shouldn't happen with RTH data, but safety check)
            if (minutes_to_close < -300) minutes_to_close += 24 * 60;
            
            // **MANDATORY POSITION CLOSURE**: 10 minutes before market close
            if (minutes_to_close <= 10 && minutes_to_close > 0) {
                for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
                    if (portfolio.positions[sid].qty != 0.0) {
                        double close_qty = -portfolio.positions[sid].qty; // Close entire position
                        double close_price = pricebook.last_px[sid];
                        
                        if (close_price > 0 && std::abs(close_qty * close_price) > 10.0) {
                            std::string inst = ST.get_symbol(sid);
                            Side close_side = (close_qty > 0) ? Side::Buy : Side::Sell;
                            
                            if (logging_enabled) {
                                audit.event_order_ex(bar.ts_utc_epoch, inst, close_side, std::abs(close_qty), 0.0, "EOD_MANDATORY_CLOSE");
                            }
                            
                            // **ZERO COSTS FOR TESTING**: Perfect execution for EOD close
                            double fees = 0.0;
                            double exec_px = close_price; // Perfect execution at market price
                            
                            double realized_pnl = (portfolio.positions[sid].qty > 0) 
                                ? (exec_px - portfolio.positions[sid].avg_price) * std::abs(close_qty)
                                : (portfolio.positions[sid].avg_price - exec_px) * std::abs(close_qty);
                            
                            apply_fill(portfolio, sid, close_qty, exec_px);
                            // portfolio.cash -= fees; // No fees for EOD close
                            
                            double eq_after = equity_mark_to_market(portfolio, pricebook.last_px);
                            if (logging_enabled) {
                                audit.event_fill_ex(bar.ts_utc_epoch, inst, exec_px, std::abs(close_qty), fees, close_side, realized_pnl, eq_after, 0.0, "EOD_MANDATORY_CLOSE");
                            }
                            total_fills++;
                        }
                    }
                }
            }
        }
        
        // **RENOVATED**: Governor handles day trading automatically - no manual time logic needed
        pricebook.sync_to_base_i(i);
        
        // Log bar data
        AuditBar audit_bar{bar.open, bar.high, bar.low, bar.close, static_cast<double>(bar.volume)};
        if (logging_enabled) audit.event_bar(bar.ts_utc_epoch, ST.get_symbol(base_symbol_id), audit_bar.open, audit_bar.high, audit_bar.low, audit_bar.close, audit_bar.volume);
        
        // Feed features to ML strategies
        [[maybe_unused]] auto start_feed = std::chrono::high_resolution_clock::now();
        FeatureFeeder::feed_features_to_strategy(strategy.get(), base_series, i, cfg.strategy_name);
        [[maybe_unused]] auto end_feed = std::chrono::high_resolution_clock::now();
        
        // **RENOVATED ARCHITECTURE**: Governor-based target weight system
        [[maybe_unused]] auto start_signal = std::chrono::high_resolution_clock::now();
        
        // **STRATEGY-AGNOSTIC**: Get allocation decisions from strategy
        std::string chain_id = std::to_string(bar.ts_utc_epoch) + ":" + std::to_string((long long)i);
        
        // Get strategy-specific router configuration
        RouterCfg strategy_router = strategy->get_router_config();
        
        // Get allocation decisions from strategy
        auto allocation_decisions = strategy->get_allocation_decisions(
            base_series, i, 
            ST.get_symbol(base_symbol_id),  // base_symbol
            strategy_router.bull3x,         // bull3x_symbol  
            strategy_router.bear3x,         // bear3x_symbol
            strategy_router.bear1x          // bear1x_symbol
        );
        
        // Log signal for diagnostics
        if (logging_enabled) {
            double probability = strategy->calculate_probability(base_series, i);
            std::string signal_desc = strategy->get_signal_description(probability);
            SigType sig_type = SigType::HOLD;
            if (signal_desc == "STRONG_BUY") sig_type = SigType::STRONG_BUY;
            else if (signal_desc == "BUY") sig_type = SigType::BUY;
            else if (signal_desc == "SELL") sig_type = SigType::SELL;
            else if (signal_desc == "STRONG_SELL") sig_type = SigType::STRONG_SELL;
            
            audit.event_signal_ex(bar.ts_utc_epoch, ST.get_symbol(base_symbol_id), 
                                sig_type, probability, chain_id);
        }
        
        [[maybe_unused]] auto end_signal = std::chrono::high_resolution_clock::now();
        
        // **CONFLICT RESOLUTION**: Close existing conflicting positions first
        int conflicts_detected = 0;
        int conflicts_resolved = 0;
        
        if (has_conflicting_positions(portfolio, ST)) {
            conflicts_detected++;
            
            // Close all conflicting positions
            for (size_t sid = 0; sid < portfolio.positions.size(); ++sid) {
                if (std::abs(portfolio.positions[sid].qty) > 1e-6) {
                    const std::string& symbol = ST.get_symbol(sid);
                    if (LONG_ETFS.count(symbol) || INVERSE_ETFS.count(symbol)) {
                        double close_qty = -portfolio.positions[sid].qty;
                        double close_price = pricebook.last_px[sid];
                        
                        if (close_price > 0 && std::abs(close_qty * close_price) > 10.0) {
                            Side close_side = (close_qty > 0) ? Side::Buy : Side::Sell;
                            
                            if (logging_enabled) {
                                audit.event_order_ex(bar.ts_utc_epoch, symbol, close_side, std::abs(close_qty), 0.0, "CONFLICT_RESOLUTION");
                            }
                            
                            // Perfect execution for conflict resolution
                            double fees = 0.0;
                            double exec_px = close_price;
                            
                            double realized_pnl = (portfolio.positions[sid].qty > 0) 
                                ? (exec_px - portfolio.positions[sid].avg_price) * std::abs(close_qty)
                                : (portfolio.positions[sid].avg_price - exec_px) * std::abs(close_qty);
                            
                            apply_fill(portfolio, sid, close_qty, exec_px);
                            
                            if (logging_enabled) {
                                double equity_after = equity_mark_to_market(portfolio, pricebook.last_px);
                                double position_after = portfolio.positions[sid].qty;
                                audit.event_fill_ex(bar.ts_utc_epoch, symbol, exec_px, std::abs(close_qty), fees, close_side, 
                                                  realized_pnl, equity_after, position_after, "CONFLICT_RESOLUTION");
                            }
                            total_fills++;
                            conflicts_resolved++;
                        }
                    }
                }
            }
        }
        
        // **STRATEGY-AGNOSTIC EXECUTION**: Execute all allocation decisions from strategy
        for (const auto& decision : allocation_decisions) {
            if (std::abs(decision.target_weight) > 1e-6) { // Only execute non-zero weights
                execute_target_position(decision.instrument, decision.target_weight, portfolio, ST, pricebook, sizer, 
                                      cfg, series, bar, chain_id, audit, logging_enabled, total_fills);
            }
        }
        
        // **CRITICAL SANITY CHECK**: Validate no conflicting positions exist after execution
        int post_execution_conflicts = 0;
        if (has_conflicting_positions(portfolio, ST)) {
            post_execution_conflicts++;
        }
        
        // **AUDIT CONFLICT STATISTICS**: Log conflict resolution metrics
        if (logging_enabled && (conflicts_detected > 0 || conflicts_resolved > 0 || post_execution_conflicts > 0)) {
            std::string conflict_stats = "CONFLICTS_DETECTED:" + std::to_string(conflicts_detected) + 
                                       ",CONFLICTS_RESOLVED:" + std::to_string(conflicts_resolved) + 
                                       ",POST_EXECUTION_CONFLICTS:" + std::to_string(post_execution_conflicts);
            audit.event_signal_drop(bar.ts_utc_epoch, cfg.strategy_name, "CONFLICT_STATS", 
                                  DropReason::NONE, chain_id, conflict_stats);
        }
        
        // 3. ============== SNAPSHOT ==============
        if (i % cfg.snapshot_stride == 0 || i == base_series.size() - 1) {
            double current_equity = equity_mark_to_market(portfolio, pricebook.last_px);
            equity_curve.emplace_back(bar.ts_utc, current_equity);
            
            // Log account snapshot
            // Calculate actual position value and track cumulative realized P&L  
            double position_value = current_equity - portfolio.cash;
            
            AccountState state;
            state.cash = portfolio.cash;
            state.equity = current_equity;
            state.realized = cumulative_realized_pnl; // Track actual cumulative realized P&L
            if (logging_enabled) audit.event_snapshot(bar.ts_utc_epoch, state);
        }
    }
    
    // 4. ============== METRICS & DIAGNOSTICS ==============
    strategy->get_diag().print(strategy->get_name().c_str());
    
    // Log signal diagnostics to audit trail
    if (logging_enabled) {
        audit.event_signal_diag(series[base_symbol_id].back().ts_utc_epoch, 
                               cfg.strategy_name, strategy->get_diag());
    }

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
    std::int64_t end_ts = equity_curve.empty() ? series[base_symbol_id][0].ts_utc_epoch : series[base_symbol_id].back().ts_utc_epoch;
    if (logging_enabled) {
        audit.event_metric(end_ts, "final_equity", result.final_equity);
        audit.event_metric(end_ts, "total_return", result.total_return);
        audit.event_metric(end_ts, "sharpe_ratio", result.sharpe_ratio);
        audit.event_metric(end_ts, "max_drawdown", result.max_drawdown);
        audit.event_metric(end_ts, "total_fills", result.total_fills);
        audit.event_metric(end_ts, "no_route", result.no_route);
        audit.event_metric(end_ts, "no_qty", result.no_qty);
    }
    
    std::string end_meta = "{";
    end_meta += "\"final_equity\":" + std::to_string(result.final_equity) + ",";
    end_meta += "\"total_return\":" + std::to_string(result.total_return) + ",";
    end_meta += "\"sharpe_ratio\":" + std::to_string(result.sharpe_ratio);
    end_meta += "}";
    if (logging_enabled) audit.event_run_end(end_ts, end_meta);

    return result;
}

} // namespace sentio