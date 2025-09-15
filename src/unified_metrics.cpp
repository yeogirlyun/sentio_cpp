#include "sentio/unified_metrics.hpp"
#include "sentio/metrics.hpp"
#include "sentio/metrics/mpr.hpp"
#include "sentio/cost_model.hpp"
#include "sentio/side.hpp"
#include "audit/audit_db.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <map>
#include <ctime>
#include <sstream>
#include <iomanip>

namespace sentio {

UnifiedMetricsReport UnifiedMetricsCalculator::calculate_metrics(const BacktestOutput& output) {
    UnifiedMetricsReport report{};
    if (output.equity_curve.empty()) {
        return report;
    }

    // 1. Use the existing, robust day-aware calculation
    RunSummary summary = calculate_from_equity_curve(output.equity_curve, output.total_fills);

    // 2. Daily returns are implicit in compute_metrics_day_aware; no separate derivation needed
    
    // 3. Populate the final, unified report
    report.final_equity = output.equity_curve.back().second;
    report.total_return = summary.ret_total;
    report.sharpe_ratio = summary.sharpe;
    report.max_drawdown = summary.mdd;
    report.monthly_projected_return = summary.monthly_proj;
    report.total_fills = output.total_fills;
    report.avg_daily_trades = output.run_trading_days > 0 ? 
        static_cast<double>(output.total_fills) / output.run_trading_days : 0.0;
        
    return report;
}

RunSummary UnifiedMetricsCalculator::calculate_from_equity_curve(
    const std::vector<std::pair<std::string, double>>& equity_curve,
    int fills_count,
    bool include_fees
) {
    // Use the authoritative compute_metrics_day_aware function
    // This ensures consistency across all systems
    return compute_metrics_day_aware(equity_curve, fills_count);
}

RunSummary UnifiedMetricsCalculator::calculate_from_audit_events(
    const std::vector<audit::Event>& events,
    double initial_capital,
    bool include_fees
) {
    // Independent BAR-driven daily-close reconstruction (UTC day buckets)
    double cash = initial_capital;
    std::unordered_map<std::string, double> positions;   // symbol -> qty
    std::unordered_map<std::string, double> last_prices; // symbol -> last price

    // Session-aware trading day bucketing: map timestamps to US/Eastern calendar day.
    // Approximate EST/EDT by using UTC-5h in winter (our current datasets span Jan–Feb).
    // For broader periods, this can be extended to handle DST.
    const std::int64_t eastern_offset_ms = 5LL * 60LL * 60LL * 1000LL; // UTC-5
    auto ts_to_day = [eastern_offset_ms](std::int64_t ts_ms) -> std::string {
        std::int64_t shifted = ts_ms - eastern_offset_ms;
        std::time_t secs = static_cast<std::time_t>(shifted / 1000);
        std::tm* tm_utc = std::gmtime(&secs);
        char buf[16];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d", tm_utc);
        return std::string(buf);
    };

    auto compute_mtm = [&]() -> double {
        double mtm = cash;
        for (const auto& kv : positions) {
            const std::string& sym = kv.first;
            double qty = kv.second;
            auto it = last_prices.find(sym);
            if (std::abs(qty) > 1e-8 && it != last_prices.end()) {
                mtm += qty * it->second;
            }
        }
        return mtm;
    };

    std::vector<std::pair<std::string,double>> daily_equity; // (YYYY-MM-DD 23:59:59, equity)
    std::string current_day;
    double last_mtm = initial_capital;

    int fills_count = 0;

    for (const auto& e : events) {
        if (e.kind == "FILL") {
            // Fees
            double fees = include_fees ? calculate_transaction_fees(e.symbol, (e.side == "SELL" ? -e.qty : e.qty), e.price) : 0.0;
            // Cash
            if (e.side == "SELL") {
                cash += e.price * e.qty - fees;
            } else {
                cash -= e.price * e.qty + fees;
            }
            // Position update
            double delta = (e.side == "SELL") ? -e.qty : e.qty;
            positions[e.symbol] += delta;
            // Update last trade price for symbol
            last_prices[e.symbol] = e.price;
            ++fills_count;
            // Do not emit a day point here; wait for BAR to set day boundary
        } else if (e.kind == "BAR") {
            // Update price book
            last_prices[e.symbol] = e.price;
            // Determine day from BAR timestamp
            std::string day = ts_to_day(e.ts_millis);
            // If day changed, emit previous day's close using last_mtm
            if (!current_day.empty() && day != current_day) {
                daily_equity.emplace_back(current_day + " 23:59:59", last_mtm);
                current_day = day;
            } else if (current_day.empty()) {
                current_day = day;
            }
            // Recompute MTM after this BAR
            last_mtm = compute_mtm();
        }
    }

    // Emit final day close
    if (!current_day.empty()) {
        daily_equity.emplace_back(current_day + " 23:59:59", last_mtm);
    } else {
        // No BARs; fallback to a start/end two-point flat series
        daily_equity.emplace_back("1970-01-01 23:59:59", initial_capital);
        daily_equity.emplace_back("1970-01-02 23:59:59", initial_capital);
    }

    return calculate_from_equity_curve(daily_equity, fills_count, include_fees);
}

std::vector<std::pair<std::string, double>> UnifiedMetricsCalculator::reconstruct_equity_curve_from_events(
    const std::vector<audit::Event>& events,
    double initial_capital,
    bool include_fees
) {
    std::vector<std::pair<std::string, double>> equity_curve;
    
    // Track portfolio state
    double cash = initial_capital;
    std::unordered_map<std::string, double> positions; // symbol -> quantity
    std::unordered_map<std::string, double> avg_prices; // symbol -> average price
    std::unordered_map<std::string, double> last_prices; // symbol -> last known price
    
    // Process events chronologically
    for (const auto& event : events) {
        if (event.kind == "FILL") {
            const std::string& symbol = event.symbol;
            double quantity = event.qty;
            double price = event.price;
            bool is_sell = (event.side == "SELL");
            
            // Convert order side to position impact
            double position_delta = is_sell ? -quantity : quantity;
            
            // Calculate transaction fees if enabled
            double fees = 0.0;
            if (include_fees) {
                fees = calculate_transaction_fees(symbol, position_delta, price);
            }
            
            // Update cash (buy decreases cash, sell increases cash)
            double cash_delta = is_sell ? (price * quantity - fees) : -(price * quantity + fees);
            cash += cash_delta;
            
            // Update position using VWAP
            double current_qty = positions[symbol];
            double new_qty = current_qty + position_delta;
            
            if (std::abs(new_qty) < 1e-8) {
                // Position closed
                positions[symbol] = 0.0;
                avg_prices[symbol] = 0.0;
            } else if (std::abs(current_qty) < 1e-8) {
                // Opening new position
                positions[symbol] = new_qty;
                avg_prices[symbol] = price;
            } else if ((current_qty > 0) == (position_delta > 0)) {
                // Adding to same side - update VWAP
                avg_prices[symbol] = (avg_prices[symbol] * current_qty + price * position_delta) / new_qty;
                positions[symbol] = new_qty;
            } else {
                // Reducing or flipping position
                positions[symbol] = new_qty;
                if (std::abs(new_qty) > 1e-8) {
                    avg_prices[symbol] = price; // New average for remaining position
                }
            }
            
            // Update last known price
            last_prices[symbol] = price;
            
            // Calculate mark-to-market equity
            double mtm_value = cash;
            for (const auto& [sym, qty] : positions) {
                if (std::abs(qty) > 1e-8 && last_prices.count(sym)) {
                    mtm_value += qty * last_prices[sym];
                }
            }
            
            // Add to equity curve with ISO timestamp for proper day compression
            std::time_t secs = static_cast<std::time_t>(event.ts_millis / 1000);
            std::tm* tm_utc = std::gmtime(&secs);
            char buf[32];
            std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm_utc);
            std::string timestamp(buf);
            equity_curve.emplace_back(timestamp, mtm_value);
        }
        else if (event.kind == "BAR") {
            // Update last known prices from bar data
            // This ensures mark-to-market calculations use current prices
            last_prices[event.symbol] = event.price;
            
            // Recalculate mark-to-market equity with updated prices
            double mtm_value = cash;
            for (const auto& [sym, qty] : positions) {
                if (std::abs(qty) > 1e-8 && last_prices.count(sym)) {
                    mtm_value += qty * last_prices[sym];
                }
            }
            
            // Add to equity curve if we have positions or this is a significant update
            if (!positions.empty() || equity_curve.empty()) {
                std::time_t secs = static_cast<std::time_t>(event.ts_millis / 1000);
                std::tm* tm_utc = std::gmtime(&secs);
                char buf[32];
                std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm_utc);
                std::string timestamp(buf);
                equity_curve.emplace_back(timestamp, mtm_value);
            }
        }
    }
    
    // Ensure we have at least initial and final equity points
    if (equity_curve.empty()) {
        equity_curve.emplace_back("start", initial_capital);
        equity_curve.emplace_back("end", initial_capital);
    } else if (equity_curve.size() == 1) {
        equity_curve.emplace_back("end", equity_curve[0].second);
    }
    
    return equity_curve;
}

double UnifiedMetricsCalculator::calculate_transaction_fees(
    const std::string& symbol,
    double quantity,
    double price
) {
    bool is_sell = (quantity < 0);
    return AlpacaCostModel::calculate_fees(symbol, std::abs(quantity), price, is_sell);
}

bool UnifiedMetricsCalculator::validate_metrics_consistency(
    const RunSummary& metrics1,
    const RunSummary& metrics2,
    double tolerance_pct
) {
    auto within_tolerance = [tolerance_pct](double a, double b) -> bool {
        if (std::abs(a) < 1e-8 && std::abs(b) < 1e-8) return true;
        double diff_pct = std::abs(a - b) / std::max(std::abs(a), std::abs(b)) * 100.0;
        return diff_pct <= tolerance_pct;
    };
    
    return within_tolerance(metrics1.ret_total, metrics2.ret_total) &&
           within_tolerance(metrics1.ret_ann, metrics2.ret_ann) &&
           within_tolerance(metrics1.monthly_proj, metrics2.monthly_proj) &&
           within_tolerance(metrics1.sharpe, metrics2.sharpe) &&
           within_tolerance(metrics1.mdd, metrics2.mdd) &&
           (metrics1.trades == metrics2.trades);
}

} // namespace sentio
