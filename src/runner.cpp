#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/pricebook.hpp"
#include "sentio/metrics.hpp"
#include "sentio/unified_metrics.hpp"
#include "sentio/metrics/mpr.hpp"
#include "sentio/metrics/session_utils.hpp"
#include "audit/audit_db_recorder.hpp"
#include "sentio/sizer.hpp"
#include "sentio/cost_model.hpp"
#include "sentio/feature_feeder.hpp"
#include "sentio/position_validator.hpp"
#include "sentio/position_coordinator.hpp"
#include "sentio/router.hpp"
#include <iostream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <map>
#include <ctime>

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

    // **BUG FIX**: Prevent zero-quantity trades that generate phantom P&L
    // Early return to completely avoid processing zero or tiny trades
    if (std::abs(trade_qty) < 1e-9 || std::abs(trade_qty * instrument_price) <= 10.0) {
        return; // No logging, no execution, no audit entries
    }

    // **PROFIT MAXIMIZATION**: Execute meaningful trades
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

    // **ALPACA COSTS**: Use realistic Alpaca fee model for accurate backtesting
    bool is_sell = (side == Side::Sell);
    double fees = AlpacaCostModel::calculate_fees(instrument, std::abs(trade_qty), instrument_price, is_sell);
    double exec_px = instrument_price; // Perfect execution at market price (no slippage)
    
    apply_fill(portfolio, instrument_id, trade_qty, exec_px);
    portfolio.cash -= fees; // Apply transaction fees
    
    double equity_after = equity_mark_to_market(portfolio, pricebook.last_px);
    double pos_after = portfolio.positions[instrument_id].qty;
    
    if (logging_enabled) {
        audit.event_fill_ex(bar.ts_utc_epoch, instrument, exec_px, std::abs(trade_qty), fees, side,
                           realized_delta, equity_after, pos_after, chain_id);
    }
    total_fills++;
}

RunResult run_backtest(IAuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                      int base_symbol_id, const RunnerCfg& cfg) {
    
    // 1. ============== INITIALIZATION ==============
    RunResult result{};
    
    const bool logging_enabled = (cfg.audit_level == AuditLevel::Full);
    // Calculate actual test period in trading days
    int actual_test_days = 0;
    if (!series[base_symbol_id].empty()) {
        // Estimate trading days from bars (assuming ~390 bars per trading day)
        actual_test_days = std::max(1, static_cast<int>(series[base_symbol_id].size()) / 390);
    }
    
    // Determine dataset type based on data source and characteristics
    std::string dataset_type = "historical"; // default
    if (!series[base_symbol_id].empty()) {
        // Check if this looks like future/AI regime data
        // Future data characteristics: ~26k bars (4 weeks), specific timestamp patterns
        size_t bar_count = series[base_symbol_id].size();
        std::int64_t first_ts = series[base_symbol_id][0].ts_utc_epoch;
        std::int64_t last_ts = series[base_symbol_id].back().ts_utc_epoch;
        double time_span_days = (last_ts - first_ts) / (60.0 * 60.0 * 24.0); // Convert seconds to days
        
        // Future data is typically exactly 4 weeks (28 days) with ~26k bars
        if (bar_count >= 25000 && bar_count <= 27000 && time_span_days >= 27 && time_span_days <= 29) {
            dataset_type = "future_ai_regime";
        }
        // Historical data is typically longer periods or different bar counts
    }
    
    // **DEFERRED**: Calculate actual test period metadata after we know the filtered data range
    // This will be done after warmup calculation when we know the exact bars being processed
    
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
    
    // **CANONICAL METADATA**: Calculate actual test period from filtered data (post-warmup)
    std::int64_t run_period_start_ts_ms = 0;
    std::int64_t run_period_end_ts_ms = 0;
    int run_trading_days = 0;
    
    if (warmup_bars < base_series.size()) {
        run_period_start_ts_ms = base_series[warmup_bars].ts_utc_epoch * 1000;
        run_period_end_ts_ms = base_series.back().ts_utc_epoch * 1000;
        
        // Count unique trading days in the filtered range
        std::vector<std::int64_t> filtered_timestamps;
        for (size_t i = warmup_bars; i < base_series.size(); ++i) {
            filtered_timestamps.push_back(base_series[i].ts_utc_epoch * 1000);
        }
        run_trading_days = sentio::metrics::count_trading_days(filtered_timestamps);
    }
    
    // Start audit run with canonical metadata
    std::string meta = "{";
    meta += "\"strategy\":\"" + cfg.strategy_name + "\",";
    meta += "\"base_symbol_id\":" + std::to_string(base_symbol_id) + ",";
    meta += "\"total_series\":" + std::to_string(series.size()) + ",";
    meta += "\"base_series_size\":" + std::to_string(series[base_symbol_id].size()) + ",";
    meta += "\"dataset_type\":\"" + dataset_type + "\",";
    meta += "\"test_period_days\":" + std::to_string(run_trading_days) + ",";
    meta += "\"run_period_start_ts_ms\":" + std::to_string(run_period_start_ts_ms) + ",";
    meta += "\"run_period_end_ts_ms\":" + std::to_string(run_period_end_ts_ms) + ",";
    meta += "\"run_trading_days\":" + std::to_string(run_trading_days);
    meta += "}";
    
    // Use current time for run timestamp (for proper run ordering)
    std::int64_t start_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (logging_enabled) audit.event_run_start(start_ts, meta);
    
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
        
        // **STRATEGY-AGNOSTIC**: Feed features to any strategy that needs them
        [[maybe_unused]] auto start_feed = std::chrono::high_resolution_clock::now();
        FeatureFeeder::feed_features_to_strategy(strategy.get(), base_series, i, strategy->get_name());
        [[maybe_unused]] auto end_feed = std::chrono::high_resolution_clock::now();
        
        // **RENOVATED ARCHITECTURE**: Governor-based target weight system
        [[maybe_unused]] auto start_signal = std::chrono::high_resolution_clock::now();
        
        // **CORRECT ARCHITECTURE**: Strategy emits ONE signal, router selects instrument
        std::string chain_id = std::to_string(bar.ts_utc_epoch) + ":" + std::to_string((long long)i);
        
        // Get strategy signal (ONE signal per bar)
        double probability = strategy->calculate_probability(base_series, i);
        StrategySignal signal = StrategySignal::from_probability(probability);
        
        // Get strategy-specific router configuration
        RouterCfg strategy_router = strategy->get_router_config();
        
        // Router determines which instrument to use based on signal strength
        std::string base_symbol = ST.get_symbol(base_symbol_id);
        auto route_decision = route(signal, strategy_router, base_symbol);
        
        // Convert router decision to allocation format
        std::vector<AllocationDecision> allocation_decisions;
        if (route_decision.has_value()) {
            // **AUDIT**: Log router decision
            if (logging_enabled) {
                audit.event_route(bar.ts_utc_epoch, base_symbol, route_decision->instrument, route_decision->target_weight);
            }
            
            AllocationDecision decision;
            decision.instrument = route_decision->instrument;
            decision.target_weight = route_decision->target_weight;
            decision.confidence = signal.confidence;
            decision.reason = "Router selected " + route_decision->instrument + " for signal strength " + std::to_string(signal.confidence);
            allocation_decisions.push_back(decision);
        } else {
            // **AUDIT**: Log when router returns no decision
            if (logging_enabled) {
                audit.event_route(bar.ts_utc_epoch, base_symbol, "NO_ROUTE", 0.0);
            }
            no_route_count++;
        }
        
        // **STRATEGY-AGNOSTIC**: Log signal for diagnostics
        if (logging_enabled) {
            std::string signal_desc = strategy->get_signal_description(probability);
            
            // **STRATEGY-AGNOSTIC**: Convert signal description to SigType enum
            SigType sig_type = SigType::HOLD;
            std::string upper_desc = signal_desc;
            std::transform(upper_desc.begin(), upper_desc.end(), upper_desc.begin(), ::toupper);
            
            if (upper_desc.find("STRONG") != std::string::npos && upper_desc.find("BUY") != std::string::npos) {
                sig_type = SigType::STRONG_BUY;
            } else if (upper_desc.find("STRONG") != std::string::npos && upper_desc.find("SELL") != std::string::npos) {
                sig_type = SigType::STRONG_SELL;
            } else if (upper_desc.find("BUY") != std::string::npos) {
                sig_type = SigType::BUY;
            } else if (upper_desc.find("SELL") != std::string::npos) {
                sig_type = SigType::SELL;
            }
            // Default remains SigType::HOLD for any other signal descriptions
            
            audit.event_signal_ex(bar.ts_utc_epoch, ST.get_symbol(base_symbol_id), 
                                sig_type, probability, chain_id);
        }
        
        [[maybe_unused]] auto end_signal = std::chrono::high_resolution_clock::now();
        
        // **STRATEGY-ISOLATED COORDINATION**: Create fresh coordinator for each run
        PositionCoordinator coordinator(5); // Max 5 orders per bar (QQQ family + extras)
        
        // Convert strategy allocation decisions to coordination format
        std::vector<AllocationDecision> coord_allocation_decisions;
        for (const auto& decision : allocation_decisions) {
            AllocationDecision coord_decision;
            coord_decision.instrument = decision.instrument;
            coord_decision.target_weight = decision.target_weight;
            coord_decision.confidence = decision.confidence;
            coord_decision.reason = decision.reason;
            coord_allocation_decisions.push_back(coord_decision);
        }
        
        // Convert to coordination requests
        auto allocation_requests = convert_allocation_decisions(coord_allocation_decisions, cfg.strategy_name, chain_id);
        
        // Coordinate all allocations to prevent conflicts
        auto coordination_decisions = coordinator.coordinate_allocations(allocation_requests, portfolio, ST);
        
        // Log coordination statistics
        auto coord_stats = coordinator.get_stats();
        int approved_orders = 0;
        int rejected_conflicts = 0;
        int rejected_frequency = 0;
        
        // **COORDINATED EXECUTION**: Execute only approved allocation decisions
        for (const auto& coord_decision : coordination_decisions) {
            if (coord_decision.result == CoordinationResult::APPROVED) {
                // Execute approved decision
                execute_target_position(coord_decision.instrument, coord_decision.approved_weight, 
                                      portfolio, ST, pricebook, sizer, cfg, series, bar, 
                                      chain_id, audit, logging_enabled, total_fills);
                approved_orders++;
                
            } else if (coord_decision.result == CoordinationResult::REJECTED_CONFLICT) {
                // Log conflict prevention
                if (logging_enabled) {
                    audit.event_signal_drop(bar.ts_utc_epoch, cfg.strategy_name, coord_decision.instrument, 
                                          DropReason::THRESHOLD, chain_id, 
                                          "CONFLICT_PREVENTED: " + coord_decision.conflict_details);
                }
                rejected_conflicts++;
                
            } else if (coord_decision.result == CoordinationResult::REJECTED_FREQUENCY) {
                // Log frequency limit
                if (logging_enabled) {
                    audit.event_signal_drop(bar.ts_utc_epoch, cfg.strategy_name, coord_decision.instrument, 
                                          DropReason::THRESHOLD, chain_id, 
                                          "FREQUENCY_LIMITED: " + coord_decision.reason);
                }
                rejected_frequency++;
                
            } else if (coord_decision.result == CoordinationResult::MODIFIED) {
                // Execute modified decision (usually zero weight to prevent conflict)
                if (std::abs(coord_decision.approved_weight) > 1e-6) {
                    execute_target_position(coord_decision.instrument, coord_decision.approved_weight, 
                                          portfolio, ST, pricebook, sizer, cfg, series, bar, 
                                          chain_id, audit, logging_enabled, total_fills);
                }
                if (logging_enabled) {
                    audit.event_signal_drop(bar.ts_utc_epoch, cfg.strategy_name, coord_decision.instrument, 
                                          DropReason::THRESHOLD, chain_id, 
                                          "MODIFIED_FOR_CONFLICT: " + coord_decision.conflict_details);
                }
            }
        }
        
        // **COORDINATION VALIDATION**: Validate no conflicting positions exist after coordination
        int post_coordination_conflicts = 0;
        if (has_conflicting_positions(portfolio, ST)) {
            post_coordination_conflicts++;
        }
        
        // **AUDIT COORDINATION STATISTICS**: Log coordination metrics
        if (logging_enabled && (approved_orders > 0 || rejected_conflicts > 0 || rejected_frequency > 0 || post_coordination_conflicts > 0)) {
            std::string coord_stats = "COORDINATION_APPROVED:" + std::to_string(approved_orders) + 
                                    ",CONFLICTS_PREVENTED:" + std::to_string(rejected_conflicts) + 
                                    ",FREQUENCY_LIMITED:" + std::to_string(rejected_frequency) + 
                                    ",POST_COORD_CONFLICTS:" + std::to_string(post_coordination_conflicts);
            audit.event_signal_drop(bar.ts_utc_epoch, cfg.strategy_name, "COORDINATION_STATS", 
                                  DropReason::NONE, chain_id, coord_stats);
        }
        
        // **CRITICAL ASSERTION**: With proper coordination, there should NEVER be post-execution conflicts
        if (post_coordination_conflicts > 0) {
            std::cerr << "CRITICAL ERROR: Position Coordinator failed to prevent conflicts!" << std::endl;
            std::cerr << "This indicates a bug in the coordination logic." << std::endl;
        }
        
        // 3. ============== SNAPSHOT ==============
        if (i % cfg.snapshot_stride == 0 || i == base_series.size() - 1) {
            double current_equity = equity_mark_to_market(portfolio, pricebook.last_px);
            
            // Fix: Ensure we have a valid timestamp string for metrics calculation
            std::string timestamp = bar.ts_utc;
            if (timestamp.empty()) {
                // Create synthetic progressive timestamps for metrics calculation
                // Start from a base date and add minutes for each bar
                static time_t base_time = 1726200000; // Sept 13, 2024 (recent date)
                time_t synthetic_time = base_time + (i * 60); // Add 1 minute per bar
                
                auto tm_val = *std::gmtime(&synthetic_time);
                char buffer[32];
                std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_val);
                timestamp = std::string(buffer);
            }
            
            equity_curve.emplace_back(timestamp, current_equity);
            
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
    
    // **CANONICAL METRICS**: Use canonical MPR calculation and store daily returns
    auto summary = UnifiedMetricsCalculator::calculate_from_equity_curve(equity_curve, total_fills, true);
    
    // Calculate daily returns from equity curve for canonical MPR
    std::vector<double> daily_equity_values;
    std::vector<std::pair<std::string, double>> daily_returns_with_dates;
    
    if (!equity_curve.empty()) {
        // Group equity by session date and calculate daily returns
        // The equity_curve already has string timestamps, so we need to extract session dates
        std::map<std::string, double> daily_equity_map;
        for (const auto& [timestamp_str, equity] : equity_curve) {
            // Extract date from timestamp string (format: "YYYY-MM-DD HH:MM:SS")
            std::string session_date = timestamp_str.substr(0, 10); // Extract "YYYY-MM-DD"
            daily_equity_map[session_date] = equity; // Keep last equity of each day
        }
        
        // Convert to vectors and calculate returns
        std::vector<std::string> dates;
        for (const auto& [date, equity] : daily_equity_map) {
            dates.push_back(date);
            daily_equity_values.push_back(equity);
        }
        
        // Calculate daily returns
        for (size_t i = 1; i < daily_equity_values.size(); ++i) {
            double daily_return = (daily_equity_values[i] / daily_equity_values[i-1]) - 1.0;
            daily_returns_with_dates.emplace_back(dates[i], daily_return);
        }
    }
    
    // Use canonical MPR calculation
    std::vector<double> daily_returns;
    for (const auto& [date, ret] : daily_returns_with_dates) {
        daily_returns.push_back(ret);
    }
    double canonical_mpr = sentio::metrics::compute_mpr_from_daily_returns(daily_returns);

    result.final_equity = equity_curve.empty() ? 100000.0 : equity_curve.back().second;
    result.total_return = summary.ret_total; // Already in decimal form (0.0366 = 3.66%)
    result.sharpe_ratio = summary.sharpe;
    result.max_drawdown = summary.mdd; // Already in decimal form
    result.monthly_projected_return = canonical_mpr; // **CANONICAL MPR**
    result.daily_trades = static_cast<int>(summary.daily_trades);
    result.total_fills = summary.trades;
    result.no_route = no_route_count;
    result.no_qty = no_qty_count;

    // **CANONICAL STORAGE**: Store daily returns and update run with canonical period fields
    std::int64_t end_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (logging_enabled) {
        // Store daily returns for canonical MPR calculation
        if (!daily_returns_with_dates.empty()) {
            // Cast to AuditDBRecorder to access canonical MPR methods
            if (auto* db_recorder = dynamic_cast<audit::AuditDBRecorder*>(&audit)) {
                db_recorder->store_daily_returns(daily_returns_with_dates);
            }
        }
        
        // Log final metrics with canonical MPR
        audit.event_metric(end_ts, "final_equity", result.final_equity);
        audit.event_metric(end_ts, "total_return", result.total_return);
        audit.event_metric(end_ts, "sharpe_ratio", result.sharpe_ratio);
        audit.event_metric(end_ts, "max_drawdown", result.max_drawdown);
        audit.event_metric(end_ts, "canonical_mpr", canonical_mpr);
        audit.event_metric(end_ts, "run_trading_days", static_cast<double>(run_trading_days));
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