#include "sentio/strategy_ire.hpp"
#include "sentio/position_validator.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace sentio {

// Helper to calculate the mean of a deque
double calculate_mean(const std::deque<double>& data) {
    if (data.empty()) return 0.0;
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

// Helper to calculate the standard deviation of a deque
double calculate_stddev(const std::deque<double>& data, double mean) {
    if (data.size() < 2) return 0.0;
    double sq_sum = 0.0;
    for (const auto& value : data) {
        sq_sum += (value - mean) * (value - mean);
    }
    return std::sqrt(sq_sum / data.size());
}

// **ENHANCED**: Multi-Timeframe Alpha Kernel Implementation


IREStrategy::IREStrategy() : BaseStrategy("IRE") {
    apply_params();
}

ParameterMap IREStrategy::get_default_params() const {
  return { {"buy_lo", 0.60}, {"buy_hi", 0.75}, {"sell_hi", 0.40}, {"sell_lo", 0.25} };
}

ParameterSpace IREStrategy::get_param_space() const { return {}; }

void IREStrategy::apply_params() {
    vol_return_history_.clear();
    vol_history_.clear();
    vwap_price_history_.clear();
    vwap_volume_history_.clear();
    alpha_return_history_.clear(); // Initialize new state
    last_trade_bar_ = -1;
    last_trade_direction_ = 0;
    entry_price_ = 0.0; // Initialize new state
    pnl_history_.clear(); // Initialize Kelly state
}

double IREStrategy::calculate_probability(const std::vector<Bar>& bars, int current_index) {
  // Use the existing calculate_target_weight method which already returns probability
  return calculate_target_weight(bars, current_index);
}

std::vector<BaseStrategy::AllocationDecision> IREStrategy::get_allocation_decisions(
    const std::vector<Bar>& bars, 
    int current_index,
    const std::string& base_symbol,
    const std::string& bull3x_symbol,
    const std::string& bear3x_symbol) {
    
    std::vector<AllocationDecision> decisions;
    
    // Get probability from strategy
    double probability = calculate_probability(bars, current_index);
    
    // **SIMPLIFIED ALLOCATION LOGIC**: Direct probability-based allocation
    if (probability > 0.80) {
        // **STRONG BUY**: High conviction - aggressive leverage
        double conviction = (probability - 0.80) / 0.20; // 0-1 scale within strong range
        double base_weight = 0.6 + (conviction * 0.4); // 60-100% allocation
        
        decisions.push_back({bull3x_symbol, base_weight * 0.7, conviction, "Strong buy: 70% TQQQ"});
        decisions.push_back({base_symbol, base_weight * 0.3, conviction, "Strong buy: 30% QQQ"});
    } 
    else if (probability > 0.55) {
        // **MODERATE BUY**: Good conviction - conservative allocation
        double conviction = (probability - 0.55) / 0.25; // 0-1 scale within moderate range
        double base_weight = 0.3 + (conviction * 0.3); // 30-60% allocation
        
        decisions.push_back({base_symbol, base_weight, conviction, "Moderate buy: 100% QQQ"});
    }
    else if (probability < 0.20) {
        // **STRONG SELL**: High conviction - aggressive inverse leverage
        double conviction = (0.20 - probability) / 0.20; // 0-1 scale within strong range
        double base_weight = 0.6 + (conviction * 0.4); // 60-100% allocation
        
        decisions.push_back({bear3x_symbol, base_weight, conviction, "Strong sell: 100% SQQQ"});
    }
    else if (probability < 0.45) {
        // **MODERATE SELL**: Good conviction - conservative inverse
        double conviction = (0.45 - probability) / 0.25; // 0-1 scale within moderate range  
        double base_weight = 0.3 + (conviction * 0.3); // 30-60% allocation
        
        // Use SHORT QQQ for moderate sell signals instead of PSQ
        decisions.push_back({base_symbol, -base_weight, conviction, "Moderate sell: SHORT QQQ"});
    }
    // **NEUTRAL ZONE** (0.45-0.55): No allocations = stay flat
    
    // **ENSURE ALL INSTRUMENTS ARE FLATTENED IF NOT IN ALLOCATION**
    std::vector<std::string> all_instruments = {base_symbol, bull3x_symbol, bear3x_symbol};
    for (const auto& inst : all_instruments) {
        bool found = false;
        for (const auto& decision : decisions) {
            if (decision.instrument == inst) { found = true; break; }
        }
        if (!found) {
            decisions.push_back({inst, 0.0, 0.0, "Flatten unused instrument"});
        }
    }
    
    return decisions;
}

RouterCfg IREStrategy::get_router_config() const {
    // IRE uses default router configuration
    RouterCfg cfg;
    cfg.bull3x = "TQQQ";
    cfg.bear3x = "SQQQ";
    // Note: moderate sell signals now use SHORT QQQ instead of PSQ
    return cfg;
}

SizerCfg IREStrategy::get_sizer_config() const {
    // IRE uses default sizer configuration
    SizerCfg cfg;
    cfg.max_position_pct = 1.0; // 100% max position
    cfg.volatility_target = 0.15; // 15% volatility target
    return cfg;
}

double IREStrategy::calculate_target_weight(const std::vector<Bar>& bars, int i) {
    ensure_governor_built_();
    const int WARMUP_PERIOD = 60;
    if (i < 1) return 0.0;

    // --- Maintain Rolling History Windows ---
    double log_return = std::log(bars[i].close / bars[i-1].close);
    vol_return_history_.push_back(log_return);
    alpha_return_history_.push_back(log_return);
    if (vol_return_history_.size() > WARMUP_PERIOD) vol_return_history_.pop_front();
    
    // **ENHANCED**: Maintain longer history for multi-timeframe analysis (30 minutes)
    if (alpha_return_history_.size() > 30) alpha_return_history_.pop_front();
    
    vwap_price_history_.push_back(bars[i].close);
    vwap_volume_history_.push_back(bars[i].volume);
    if (vwap_price_history_.size() > 20) {
        vwap_price_history_.pop_front();
        vwap_volume_history_.pop_front();
    }
    
    if (i < WARMUP_PERIOD) return 0.0;

    // --- Dynamic Take-Profit Logic ---
    if (last_trade_direction_ != 0 && entry_price_ > 0) {
        double current_pnl = (bars[i].close - entry_price_) * last_trade_direction_;
        // Target profit is 1.5x the recent volatility
        double take_profit_threshold = calculate_stddev(vol_return_history_, 0.0) * entry_price_ * 1.5;
        if (current_pnl > take_profit_threshold) {
            // **NEW**: Record trade performance for Kelly Criterion
            update_trade_performance(current_pnl);
            
            last_trade_direction_ = 0; // Reset trade state
            entry_price_ = 0.0;
            return 0.0; // Signal to flatten position and lock in profit
        }
    }

    // --- REGIME DETECTION ---
    double vol_mean = calculate_mean(vol_return_history_);
    double current_vol = calculate_stddev(vol_return_history_, vol_mean);
    vol_history_.push_back(current_vol);
    if (vol_history_.size() > WARMUP_PERIOD) vol_history_.pop_front();
    double avg_vol = calculate_mean(vol_history_);
    bool is_high_volatility = current_vol > (avg_vol * 2.0);

    // --- SIGNAL CALCULATION ---
    double regime_probability = 0.5;
    if (is_high_volatility) {
        double momentum = (bars[i].close / bars[i-10].close) - 1.0;
        double volume_mean = calculate_mean(vwap_volume_history_);
        if (bars[i].volume > volume_mean * 2.5) { 
            if (momentum > 0.0015) regime_probability = 0.80;
            else if (momentum < -0.0015) regime_probability = 0.20;
        }
    } else {
        double sum_pv = 0.0, sum_vol = 0.0;
        for (size_t k = 0; k < vwap_price_history_.size(); ++k) {
            sum_pv += vwap_price_history_[k] * vwap_volume_history_[k];
            sum_vol += vwap_volume_history_[k];
        }
        double vwap = (sum_vol > 0) ? sum_pv / sum_vol : bars[i].close;
        double sq_diff_sum = 0.0;
        for (size_t k = 0; k < vwap_price_history_.size(); ++k) {
            sq_diff_sum += (vwap_price_history_[k] - vwap) * (vwap_price_history_[k] - vwap);
        }
        double std_dev_from_vwap = std::sqrt(sq_diff_sum / vwap_price_history_.size());
        if (std_dev_from_vwap > 0) {
            double z_score = (bars[i].close - vwap) / std_dev_from_vwap;
            regime_probability = 0.5 - std::clamp(z_score / 5.0, -0.40, 0.40);
        }
    }

    // **REVERTED**: Simple 20-minute MA momentum signal (PROVEN 3.21% PERFORMER)
    double momentum_signal = 0.5; // Default neutral
    if (i >= 20) {
        double ma_20 = 0.0;
        for (int j = i - 19; j <= i; ++j) {
            ma_20 += bars[j].close;
        }
        ma_20 /= 20.0;
        
        // Simple momentum: current price vs 20-min MA
        double momentum = (bars[i].close - ma_20) / ma_20;
        momentum_signal = 0.5 + std::clamp(momentum * 25.0, -0.4, 0.4); // Scale momentum to probability
    }
    
    latest_probability_ = momentum_signal;
    
    // **FIXED**: Increment signal diagnostics counter
    diag_.emitted++;
    
    // **SIMPLIFIED**: Strategy only provides probability - runner handles allocation
    static int debug_count = 0;
    if (debug_count < 5 || debug_count % 1000 == 0) {
        double momentum_pct = 0.0;
        if (i >= 20) {
            double ma_20_debug = 0.0;
            for (int j = i - 19; j <= i; ++j) ma_20_debug += bars[j].close;
            ma_20_debug /= 20.0;
            momentum_pct = (bars[i].close - ma_20_debug) / ma_20_debug * 100.0;
        }
    }
    debug_count++;
    
    // **FIXED**: Return the calculated momentum signal probability
    return momentum_signal; // Return actual probability for signal generation
}

void IREStrategy::ensure_governor_built_() {
    if (governor_) return;
    IntradayPositionGovernor::Config gov_config;
    gov_config.lookback_window = 45;        // Shorter for more responsiveness to Alpha Kernel
    gov_config.buy_percentile = params_["buy_hi"];   // Use actual strategy parameter
    gov_config.sell_percentile = params_["sell_lo"]; // Use actual strategy parameter
    gov_config.max_base_weight = 1.0; 
    gov_config.min_abs_edge = 0.03;         // Lower threshold to allow Alpha Kernel through
    governor_ = std::make_unique<IntradayPositionGovernor>(gov_config);
}

// ensure_ensemble_built is no longer needed but kept for compatibility
void IREStrategy::ensure_ensemble_built_() {}

// **NEW**: Kelly Criterion Implementation
double IREStrategy::calculate_kelly_fraction(double edge_probability, double confidence) const {
    if (pnl_history_.size() < 10) {
        // Not enough trade history, use moderate sizing
        return 0.8; // 80% of normal position size during learning phase
    }
    
    double win_rate = get_win_rate();
    double win_loss_ratio = get_win_loss_ratio();
    
    if (win_loss_ratio <= 0.0 || win_rate <= 0.0) {
        // Poor historical performance, use minimal sizing
        return 0.25;
    }
    
    // Kelly formula: f = (bp - q) / b
    // where b = win_loss_ratio, p = edge_probability, q = 1-p
    double p = edge_probability;
    double q = 1.0 - p;
    double b = win_loss_ratio;
    
    double kelly_f = (b * p - q) / b;
    
    // Apply confidence scaling and safety constraints
    double scaled_kelly = kelly_f * confidence * 0.5; // 50% of full Kelly for more aggressive sizing (half-Kelly)
    
    // Clamp to reasonable range - more aggressive bounds
    return std::clamp(scaled_kelly, 0.3, 3.0); // Min 30%, Max 300% of base position
}

void IREStrategy::update_trade_performance(double realized_pnl) {
    pnl_history_.push_back(realized_pnl);
    
    // Maintain rolling window of last 50 trades for Kelly calculation
    if (pnl_history_.size() > 50) {
        pnl_history_.pop_front();
    }
}

double IREStrategy::get_win_loss_ratio() const {
    if (pnl_history_.size() < 5) return 1.0; // Default ratio
    
    double total_wins = 0.0;
    double total_losses = 0.0;
    int win_count = 0;
    int loss_count = 0;
    
    for (double pnl : pnl_history_) {
        if (pnl > 0) {
            total_wins += pnl;
            win_count++;
        } else if (pnl < 0) {
            total_losses += std::abs(pnl);
            loss_count++;
        }
    }
    
    if (loss_count == 0) return 2.0; // No losses, assume good ratio
    if (win_count == 0) return 0.5; // No wins, conservative ratio
    
    double avg_win = total_wins / win_count;
    double avg_loss = total_losses / loss_count;
    
    return avg_win / avg_loss;
}

double IREStrategy::get_win_rate() const {
    if (pnl_history_.size() < 5) return 0.55; // Default optimistic win rate
    
    int wins = 0;
    for (double pnl : pnl_history_) {
        if (pnl > 0) wins++;
    }
    
    return static_cast<double>(wins) / pnl_history_.size();
}

// **NEW**: Multi-Timeframe Alpha Kernel Implementation
double IREStrategy::calculate_multi_timeframe_alpha(const std::deque<double>& history) const {
    if (history.size() < 30) return 0.5; // Need sufficient history for all timeframes
    
    // **Ultra-Fast (3-8 min)**: Captures immediate momentum and noise
    double short_alpha = calculate_single_alpha_probability(history, 3, 8);
    
    // **Medium-Term (5-15 min)**: Identifies core intraday moves (original alpha)
    double medium_alpha = calculate_single_alpha_probability(history, 5, 15);
    
    // **Long-Term (10-30 min)**: Detects broader intraday trend
    double long_alpha = calculate_single_alpha_probability(history, 10, 30);
    
    // **Hierarchical Blending**: Weight shorter timeframes more heavily for day trading
    // short_alpha * 0.5 + medium_alpha * 0.3 + long_alpha * 0.2 = active but trend-aware
    double ensemble_alpha = (short_alpha * 0.5) + (medium_alpha * 0.3) + (long_alpha * 0.2);
    
    return ensemble_alpha;
}

double IREStrategy::calculate_single_alpha_probability(const std::deque<double>& history, int short_window, int long_window) const {
    if (history.size() < static_cast<size_t>(long_window)) return 0.5;

    // Velocity (short-term trend direction)
    double recent_mean = 0.0;
    for(size_t i = history.size() - short_window; i < history.size(); ++i) {
        recent_mean += history[i];
    }
    recent_mean /= short_window;

    // Acceleration (how the trend is changing)
    double older_mean = 0.0;
    for(size_t i = history.size() - long_window; i < history.size() - short_window; ++i) {
        older_mean += history[i];
    }
    older_mean /= (long_window - short_window);
    
    // A positive acceleration means the upward trend is strengthening (or downward is weakening)
    double acceleration = recent_mean - older_mean;

    // Combine velocity and acceleration for a forward-looking forecast
    double forecast = (recent_mean * 0.7) + (acceleration * 0.3);
    
    // **ENHANCED**: Adaptive scaling factor based on timeframe
    double timeframe_scale = 5000.0;
    if (short_window <= 3) {
        // Ultra-fast: More aggressive for scalping
        timeframe_scale = 8000.0;
    } else if (short_window >= 10) {
        // Long-term: More conservative for trend
        timeframe_scale = 3000.0;
    }
    
    // Convert the forecast into a probability between 0 and 1
    return 0.5 + std::clamp(forecast * timeframe_scale, -0.48, 0.48);
}

REGISTER_STRATEGY(IREStrategy, "ire");

} // namespace sentio