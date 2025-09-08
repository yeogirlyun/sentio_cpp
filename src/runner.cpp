#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/strategy_ire.hpp"
#include "sentio/pricebook.hpp"
#include "sentio/metrics.hpp"
#include "sentio/sizer.hpp"
#include "sentio/cost_model.hpp"
#include "sentio/feature_feeder.hpp"
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
                                   const std::string& chain_id, AuditRecorder& audit, bool logging_enabled, int& total_fills) {
    
    int instrument_id = ST.get_id(instrument);
    if (instrument_id == -1) return;
    
    double instrument_price = pricebook.last_px[instrument_id];
    if (instrument_price <= 0) return;

    // Calculate target quantity using sizer
    double target_qty = sizer.calculate_target_quantity(portfolio, ST, pricebook.last_px, 
                                                       instrument, target_weight, 
                                                       series[instrument_id], cfg.sizer);
    
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

RunResult run_backtest(AuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
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
        // **DEBUG**: Check main loop entry
        if (i < warmup_bars + 3) {
            std::cout << "MAIN LOOP i=" << i << " warmup=" << warmup_bars << " strategy=" << cfg.strategy_name << std::endl;
        }
        
        // Progress reporting at 5% intervals
        if (i % progress_interval == 0) {
            int progress_percent = (i * 100) / total_bars;
            std::cout << "Progress: " << progress_percent << "% (" << i << "/" << total_bars << " bars)" << std::endl;
        }
        
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
        if (logging_enabled) audit.event_bar(bar.ts_utc_epoch, ST.get_symbol(base_symbol_id), audit_bar);
        
        // Feed features to ML strategies
        [[maybe_unused]] auto start_feed = std::chrono::high_resolution_clock::now();
        FeatureFeeder::feed_features_to_strategy(strategy.get(), base_series, i, cfg.strategy_name);
        [[maybe_unused]] auto end_feed = std::chrono::high_resolution_clock::now();
        
        // **RENOVATED ARCHITECTURE**: Governor-based target weight system
        [[maybe_unused]] auto start_signal = std::chrono::high_resolution_clock::now();
        
        bool is_ire = (cfg.strategy_name == "IRE");
        double target_weight = 0.0;
        std::string target_instrument = ST.get_symbol(base_symbol_id);
        std::string chain_id = std::to_string(bar.ts_utc_epoch) + ":" + std::to_string((long long)i);
        std::vector<std::pair<std::string, double>> allocations; // For dynamic leverage
        
        // **DEBUG**: Check strategy name and path selection
        if (i < 5) {
            std::cout << "Strategy: " << cfg.strategy_name << " is_ire=" << is_ire << std::endl;
        }
        
        if (is_ire) {
            // **DYNAMIC LEVERAGE OPTIMIZATION**: Strategy provides probability, runner optimizes allocation
            auto* ire_strategy = dynamic_cast<IREStrategy*>(strategy.get());
            if (ire_strategy) {
                // 1. GET PURE PROBABILITY SIGNAL FROM STRATEGY
                ire_strategy->calculate_target_weight(base_series, i); // Updates internal probability
                double probability = ire_strategy->get_latest_probability();
                
                // 2. DYNAMIC LEVERAGE ALLOCATION LOGIC BASED ON SIGNAL STRENGTH
                // **FIXED**: Use outer scope allocations vector (don't create new one)
                
                if (probability > 0.80) {
                    // **STRONG BUY**: High conviction - aggressive leverage
                    double conviction = (probability - 0.80) / 0.20; // 0-1 scale within strong range
                    double base_weight = 0.6 + (conviction * 0.4); // 60-100% allocation
                    allocations.push_back({cfg.router.bull3x, base_weight * 0.7}); // 70% TQQQ (3x leverage)
                    allocations.push_back({ST.get_symbol(base_symbol_id), base_weight * 0.3}); // 30% QQQ (1x)
                } 
                else if (probability > 0.55) {
                    // **MODERATE BUY**: Good conviction - conservative allocation
                    double conviction = (probability - 0.55) / 0.25; // 0-1 scale within moderate range
                    double base_weight = 0.3 + (conviction * 0.3); // 30-60% allocation
                    allocations.push_back({ST.get_symbol(base_symbol_id), base_weight}); // 100% QQQ (1x)
                }
                else if (probability < 0.20) {
                    // **STRONG SELL**: High conviction - aggressive inverse leverage
                    double conviction = (0.20 - probability) / 0.20; // 0-1 scale within strong range
                    double base_weight = 0.6 + (conviction * 0.4); // 60-100% allocation
                    allocations.push_back({cfg.router.bear3x, base_weight}); // 100% SQQQ (3x inverse)
                }
                else if (probability < 0.45) {
                    // **MODERATE SELL**: Good conviction - conservative inverse
                    double conviction = (0.45 - probability) / 0.25; // 0-1 scale within moderate range  
                    double base_weight = 0.3 + (conviction * 0.3); // 30-60% allocation
                    allocations.push_back({"PSQ", base_weight}); // 100% PSQ (1x inverse)
                }
                // **NEUTRAL ZONE** (0.45-0.55): No allocations = stay flat
                
                // 3. ENSURE ALL INSTRUMENTS ARE FLATTENED IF NOT IN ALLOCATION
                std::vector<std::string> all_instruments = {ST.get_symbol(base_symbol_id), cfg.router.bull3x, cfg.router.bear3x, "PSQ"};
                for (const auto& inst : all_instruments) {
                    bool found = false;
                    for (const auto& alloc : allocations) {
                        if (alloc.first == inst) { found = true; break; }
                    }
                    if (!found) allocations.push_back({inst, 0.0}); // Flatten unused instruments
                }
                
                // **DEBUG**: Log allocation decisions
                if (i < 10 || i % 1000 == 0) {
                    std::cout << "DYNAMIC LEVERAGE i=" << i << " prob=" << probability << " allocations_size=" << allocations.size();
                    for (const auto& alloc : allocations) {
                        std::cout << " " << alloc.first << ":" << alloc.second;
                    }
                    std::cout << std::endl;
                }
                
                // Log probability for diagnostics
                if (logging_enabled) {
                    audit.event_signal_ex(bar.ts_utc_epoch, ST.get_symbol(base_symbol_id), 
                                        SigType::HOLD, probability, chain_id);
                }
            }
        } else {
            // **LEGACY**: Old signal routing for non-IRE strategies
            StrategySignal sig = strategy->calculate_signal(base_series, i);
            if (sig.type != StrategySignal::Type::HOLD) {
                SigType sig_type = static_cast<SigType>(static_cast<int>(sig.type));
                if (logging_enabled) audit.event_signal_ex(bar.ts_utc_epoch, ST.get_symbol(base_symbol_id), sig_type, sig.confidence, chain_id);
                
                auto route_decision = route(sig, cfg.router, ST.get_symbol(base_symbol_id));
                if (route_decision) {
                    target_weight = route_decision->target_weight;
                    target_instrument = route_decision->instrument;
                }
            }
        }
        
        [[maybe_unused]] auto end_signal = std::chrono::high_resolution_clock::now();
        
        // **DYNAMIC EXECUTION**: Execute allocations (IRE) or legacy target weight
        if (is_ire) {
            // **DEBUG**: Confirm execution block is reached
            if (i < 10 || i % 1000 == 0) {
                std::cout << "EXECUTION BLOCK REACHED i=" << i << " allocations_size=" << allocations.size() << std::endl;
            }
            
            // **DYNAMIC LEVERAGE**: Execute all allocation decisions from probability-based logic above
            for (const auto& [instrument, weight] : allocations) {
                // **DEBUG**: Check every weight
                if (i < 10 || i % 1000 == 0) {
                    std::cout << "CHECKING: " << instrument << " weight=" << weight << " abs=" << std::abs(weight) << " threshold=" << 1e-6;
                }
                
                if (std::abs(weight) > 1e-6) { // Only execute non-zero weights
                    // **DEBUG**: Log execution attempts BEFORE calling function
                    if (i < 10 || i % 1000 == 0) {
                        std::cout << " → EXECUTING fills_before=" << total_fills;
                    }
                    
                    execute_target_position(instrument, weight, portfolio, ST, pricebook, sizer, 
                                          cfg, series, bar, chain_id, audit, logging_enabled, total_fills);
                    
                    // **DEBUG**: Log execution results AFTER calling function
                    if (i < 10 || i % 1000 == 0) {
                        std::cout << " fills_after=" << total_fills;
                    }
                } else {
                    if (i < 10 || i % 1000 == 0) {
                        std::cout << " → SKIPPED";
                    }
                }
                
                if (i < 10 || i % 1000 == 0) {
                    std::cout << std::endl;
                }
            }
        } else {
            // **LEGACY**: Single instrument execution for other strategies
            if (std::abs(target_weight) > 1e-6) {
                execute_target_position(target_instrument, target_weight, portfolio, ST, pricebook, sizer,
                                      cfg, series, bar, chain_id, audit, logging_enabled, total_fills);
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
            if (logging_enabled) audit.event_snapshot(bar.ts_utc_epoch, state);
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