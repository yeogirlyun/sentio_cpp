#pragma once
#include "core.hpp"
#include "symbol_table.hpp"
#include "position_validator.hpp"
#include "time_utils.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>

namespace sentio {

// **SAFE SIZER CONFIGURATION**: Enforces all trading system constraints
struct SafeSizerConfig {
    // **CASH MANAGEMENT**
    double min_cash_reserve_pct = 0.02;      // Minimum 2% cash reserve (relaxed)
    double max_position_pct = 0.95;          // Maximum 95% of available cash per position
    double max_total_leverage = 1.0;         // Maximum 1x leverage (cash account)
    
    // **CONFLICT PREVENTION**
    bool prevent_conflicting_positions = true; // Prevent long/inverse conflicts
    
    // **TRADE FREQUENCY CONTROL**
    int max_trades_per_bar = 1;              // Maximum 1 trade per bar
    
    // **EOD MANAGEMENT**
    bool force_eod_closure = true;           // Force EOD closure of all positions
    int eod_closure_minutes = 30;            // Start closure 30 minutes before close
    
    // **BASIC FILTERS**
    bool fractional_allowed = true;
    double min_notional = 10.0;              // Minimum $10 trade size
    double min_price = 0.01;                 // Minimum price validation
};

// **TRADE FREQUENCY TRACKER**: Prevents multiple trades per bar
class TradeFrequencyTracker {
private:
    mutable std::unordered_map<int64_t, int> trades_per_timestamp_;
    mutable int64_t last_cleanup_timestamp_ = 0;
    
public:
    bool can_trade_at_timestamp(int64_t timestamp, int max_trades_per_bar) const {
        // Clean up old entries periodically
        if (timestamp > last_cleanup_timestamp_ + 86400) { // Daily cleanup
            cleanup_old_entries(timestamp);
            last_cleanup_timestamp_ = timestamp;
        }
        
        int current_trades = trades_per_timestamp_[timestamp];
        return current_trades < max_trades_per_bar;
    }
    
    void record_trade(int64_t timestamp) {
        trades_per_timestamp_[timestamp]++;
    }
    
    void cleanup_old_entries(int64_t current_timestamp) const {
        auto it = trades_per_timestamp_.begin();
        while (it != trades_per_timestamp_.end()) {
            if (current_timestamp - it->first > 86400) { // Remove entries older than 1 day
                it = trades_per_timestamp_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    int get_trades_at_timestamp(int64_t timestamp) const {
        auto it = trades_per_timestamp_.find(timestamp);
        return (it != trades_per_timestamp_.end()) ? it->second : 0;
    }
};

// **EOD MANAGER**: Handles end-of-day position closure
class EODManager {
private:
    SafeSizerConfig config_;
    
public:
    EODManager(const SafeSizerConfig& config) : config_(config) {}
    
    bool is_eod_closure_time(int64_t timestamp) const {
        if (!config_.force_eod_closure) return false;
        
        // Convert timestamp to market time and check if within closure window
        auto market_timing = get_market_timing(timestamp);
        int minutes_to_close = get_minutes_to_market_close(timestamp, market_timing);
        
        return minutes_to_close <= config_.eod_closure_minutes && minutes_to_close >= 0;
    }
    
    bool should_close_all_positions(int64_t timestamp) const {
        return is_eod_closure_time(timestamp);
    }
    
private:
    struct MarketTiming {
        int market_close_hour_utc = 20;  // 4 PM EDT = 20:00 UTC
        int market_close_minute_utc = 0;
    };
    
    MarketTiming get_market_timing([[maybe_unused]] int64_t timestamp) const {
        // For now, use standard US market hours
        // Could be enhanced to handle different markets/timezones
        return MarketTiming{};
    }
    
    int get_minutes_to_market_close(int64_t timestamp, const MarketTiming& timing) const {
        time_t raw_time = timestamp;
        struct tm* utc_tm = gmtime(&raw_time);
        
        int current_minutes = utc_tm->tm_hour * 60 + utc_tm->tm_min;
        int close_minutes = timing.market_close_hour_utc * 60 + timing.market_close_minute_utc;
        
        int minutes_to_close = close_minutes - current_minutes;
        
        // Handle day wrap-around
        if (minutes_to_close < -300) {
            minutes_to_close += 24 * 60;
        }
        
        return minutes_to_close;
    }
};

// **SAFE SIZER**: Comprehensive sizer with all safety constraints
class SafeSizer {
private:
    SafeSizerConfig config_;
    TradeFrequencyTracker frequency_tracker_;
    EODManager eod_manager_;
    
    // Conflict detection sets
    const std::unordered_set<std::string> LONG_ETFS = {"QQQ", "TQQQ"};
    const std::unordered_set<std::string> INVERSE_ETFS = {"PSQ", "SQQQ"};
    
public:
    SafeSizer(const SafeSizerConfig& config = SafeSizerConfig{}) 
        : config_(config), eod_manager_(config) {}
    
    // **MAIN SIZING FUNCTION**: Safe position sizing with all constraints
    double calculate_target_quantity(
        const Portfolio& portfolio,
        const SymbolTable& ST,
        const std::vector<double>& last_prices,
        const std::string& instrument,
        double target_weight,
        int64_t timestamp,
        [[maybe_unused]] const std::vector<Bar>& price_history = {}) const {
        
        // **VALIDATION**: Basic input validation
        int instrument_id = ST.get_id(instrument);
        if (instrument_id == -1 || instrument_id >= (int)last_prices.size()) {
            return 0.0;
        }
        
        double instrument_price = last_prices[instrument_id];
        if (instrument_price <= config_.min_price) {
            return 0.0;
        }
        
        // **EOD CHECK**: Force closure of all positions during EOD window
        if (eod_manager_.should_close_all_positions(timestamp)) {
            // **CRITICAL FIX**: Always return 0 to close position, never negative quantities
            return 0.0; // Close position to zero (no shorts allowed)
        }
        
        // **FREQUENCY CHECK**: Enforce trade frequency limits
        if (!frequency_tracker_.can_trade_at_timestamp(timestamp, config_.max_trades_per_bar)) {
            return portfolio.positions[instrument_id].qty; // Hold current position
        }
        
        // **CASH VALIDATION**: Calculate available cash for trading
        double available_cash = calculate_available_cash(portfolio, last_prices);
        if (available_cash <= 0) {
            return portfolio.positions[instrument_id].qty; // Hold current position
        }
        
        // **CONFLICT CHECK**: Prevent conflicting positions
        if (config_.prevent_conflicting_positions) {
            if (would_create_conflict(portfolio, ST, last_prices, instrument, target_weight)) {
                // Close conflicting positions first by returning current quantity
                return portfolio.positions[instrument_id].qty;
            }
        }
        
        // **POSITION SIZING**: Calculate safe position size
        double target_qty = calculate_safe_quantity(
            portfolio, last_prices, instrument_price, target_weight, available_cash);
        
        // **CRITICAL SAFETY CHECK**: NEVER allow short positions (negative quantities)
        // We have inverse ETFs (SQQQ, PSQ) for bearish exposure - no shorts needed
        if (target_qty < 0.0) {
            std::cout << "WARNING: SafeSizer prevented short position in " << instrument 
                      << " (target_qty=" << target_qty << ") - using 0 instead" << std::endl;
            return 0.0;
        }
        
        return target_qty;
    }
    
    // **RECORD TRADE**: Must be called after successful trade execution
    void record_trade_execution(int64_t timestamp) {
        frequency_tracker_.record_trade(timestamp);
    }
    
    // **DIAGNOSTICS**: Get current system state
    bool can_trade_now(int64_t timestamp) const {
        return frequency_tracker_.can_trade_at_timestamp(timestamp, config_.max_trades_per_bar) &&
               !eod_manager_.should_close_all_positions(timestamp);
    }
    
    int get_trades_at_timestamp(int64_t timestamp) const {
        return frequency_tracker_.get_trades_at_timestamp(timestamp);
    }
    
    bool is_eod_closure_active(int64_t timestamp) const {
        return eod_manager_.is_eod_closure_time(timestamp);
    }

private:
    // **CASH CALCULATION**: Calculate available cash for new positions
    double calculate_available_cash(const Portfolio& portfolio, const std::vector<double>& last_prices) const {
        // **CRITICAL FIX**: Use actual cash balance, not total equity
        double cash = portfolio.cash;
        
        // **SAFETY CHECK**: Must have positive cash to trade
        if (cash <= 0) return 0.0;
        
        // **CASH RESERVE**: Maintain minimum cash reserve based on starting capital
        // Use a fixed reserve amount to avoid the equity calculation trap
        double min_cash_required = 100000.0 * config_.min_cash_reserve_pct; // Fixed starting capital
        double available_for_trading = std::max(0.0, cash - min_cash_required);
        
        return available_for_trading;
    }
    
    // **CONFLICT DETECTION**: Check if position would create conflicts
    bool would_create_conflict(
        const Portfolio& portfolio,
        const SymbolTable& ST,
        const std::vector<double>& last_prices,
        const std::string& instrument,
        double target_weight) const {
        
        if (!config_.prevent_conflicting_positions || std::abs(target_weight) < 1e-6) {
            return false;
        }
        
        // Determine what type of position is being requested
        bool requesting_long = (LONG_ETFS.count(instrument) && target_weight > 0);
        bool requesting_inverse = (INVERSE_ETFS.count(instrument) && target_weight > 0);
        bool requesting_short = (LONG_ETFS.count(instrument) && target_weight < 0);
        
        // Check existing positions for conflicts
        for (size_t i = 0; i < portfolio.positions.size() && i < last_prices.size(); ++i) {
            const auto& pos = portfolio.positions[i];
            if (std::abs(pos.qty) < 1e-6) continue; // No position
            
            std::string existing_symbol = ST.get_symbol(i);
            bool existing_long = (LONG_ETFS.count(existing_symbol) && pos.qty > 0);
            bool existing_inverse = (INVERSE_ETFS.count(existing_symbol) && pos.qty > 0);
            bool existing_short = (LONG_ETFS.count(existing_symbol) && pos.qty < 0);
            
            // Check for conflicts
            if ((requesting_long && (existing_inverse || existing_short)) ||
                (requesting_inverse && (existing_long || existing_short)) ||
                (requesting_short && (existing_long || existing_inverse))) {
                return true; // Conflict detected
            }
        }
        
        return false; // No conflicts
    }
    
    // **SAFE QUANTITY CALCULATION**: Calculate position size with all safety checks
    double calculate_safe_quantity(
        const Portfolio& portfolio,
        const std::vector<double>& last_prices,
        double instrument_price,
        double target_weight,
        double available_cash) const {
        
        if (std::abs(target_weight) < 1e-6) return 0.0;
        
        // **CRITICAL FIX**: Use available cash as the base, not total equity
        // This prevents the compounding leverage effect that caused negative cash
        
        // **CASH-BASED SIZING**: Calculate position size based on available cash
        double desired_notional = available_cash * std::abs(target_weight);
        
        // **POSITION SIZE LIMIT**: Limit maximum position size (optional additional safety)
        double max_position_value = available_cash * config_.max_position_pct;
        
        // Use the most restrictive constraint (cash is already the limiting factor)
        double final_notional = std::min(desired_notional, max_position_value);
        
        // **MINIMUM NOTIONAL CHECK**
        if (final_notional < config_.min_notional) return 0.0;
        
        // Calculate quantity
        double qty = final_notional / instrument_price;
        
        // Apply fractional/integer constraint
        if (!config_.fractional_allowed) {
            qty = std::floor(qty);
        }
        
        // **CRITICAL FIX**: NEVER create short positions
        // In our system, negative target_weight should never happen because
        // we use inverse ETFs (SQQQ, PSQ) for bearish exposure
        if (target_weight < 0) {
            std::cout << "ERROR: SafeSizer received negative target_weight=" << target_weight 
                      << " - this should never happen with inverse ETFs. Returning 0." << std::endl;
            return 0.0;
        }
        
        // Always return positive quantity (long positions only)
        return qty;
    }
};

} // namespace sentio
