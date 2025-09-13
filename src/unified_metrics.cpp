#include "sentio/unified_metrics.hpp"
#include "sentio/metrics.hpp"
#include "sentio/cost_model.hpp"
#include "sentio/side.hpp"
#include "audit/audit_db.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace sentio {

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
    // Reconstruct equity curve from audit events
    auto equity_curve = reconstruct_equity_curve_from_events(events, initial_capital, include_fees);
    
    // Count fill events for trade statistics
    int fills_count = 0;
    for (const auto& event : events) {
        if (event.kind == "FILL") {
            fills_count++;
        }
    }
    
    // Use unified calculation method
    return calculate_from_equity_curve(equity_curve, fills_count, include_fees);
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
            
            // Add to equity curve (convert timestamp from millis to string)
            std::string timestamp = std::to_string(event.ts_millis);
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
                std::string timestamp = std::to_string(event.ts_millis);
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
