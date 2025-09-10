#include "sentio/virtual_market.hpp"
#include "sentio/strategy_registry.hpp"
#include "sentio/runner.hpp"
#include "sentio/temporal_analysis.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/audit.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/mars_data_loader.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>

namespace sentio {

VirtualMarketEngine::VirtualMarketEngine() 
    : rng_(std::chrono::steady_clock::now().time_since_epoch().count()) {
    initialize_market_regimes();
    
    // Initialize base prices for common symbols (updated to match real market levels)
    base_prices_["QQQ"] = 458.0; // Updated to match real QQQ levels
    base_prices_["SPY"] = 450.0;
    base_prices_["AAPL"] = 175.0;
    base_prices_["MSFT"] = 350.0;
    base_prices_["TSLA"] = 250.0;
    base_prices_["PSQ"] = 380.0;  // Inverse QQQ
    base_prices_["TQQQ"] = 120.0; // 3x QQQ
    base_prices_["SQQQ"] = 120.0; // 3x inverse QQQ
}

void VirtualMarketEngine::initialize_market_regimes() {
    market_regimes_ = {
        {"BULL_TRENDING", 0.015, 0.02, 0.3, 1.2, 60},
        {"BEAR_TRENDING", 0.025, -0.015, 0.2, 1.5, 45},
        {"SIDEWAYS_LOW_VOL", 0.008, 0.001, 0.8, 0.8, 90},
        {"SIDEWAYS_HIGH_VOL", 0.020, 0.002, 0.6, 1.3, 30},
        {"VOLATILE_BREAKOUT", 0.035, 0.025, 0.1, 2.0, 15},
        {"VOLATILE_BREAKDOWN", 0.040, -0.030, 0.1, 2.2, 20},
        {"NORMAL_MARKET", 0.008, 0.001, 0.5, 1.0, 120}
    };
    
    // Select initial regime
    select_new_regime();
}

void VirtualMarketEngine::select_new_regime() {
    // Weight regimes by probability (normal market is most common)
    std::vector<double> weights = {0.15, 0.10, 0.20, 0.15, 0.05, 0.05, 0.30};
    
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    int regime_index = dist(rng_);
    
    current_regime_ = market_regimes_[regime_index];
    regime_start_time_ = std::chrono::system_clock::now();
    
    std::cout << "🔄 Switched to market regime: " << current_regime_.name 
              << " (vol=" << std::fixed << std::setprecision(3) << current_regime_.volatility
              << ", trend=" << current_regime_.trend << ")" << std::endl;
}

bool VirtualMarketEngine::should_change_regime() {
    if (current_regime_.name.empty()) return true;
    
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - regime_start_time_);
    return elapsed.count() >= current_regime_.duration_minutes;
}

std::vector<Bar> VirtualMarketEngine::generate_market_data(const std::string& symbol, 
                                                         int periods, 
                                                         int interval_seconds) {
    std::vector<Bar> bars;
    bars.reserve(periods);
    
    auto current_time = std::chrono::system_clock::now();
    double current_price = base_prices_[symbol];
    
    for (int i = 0; i < periods; ++i) {
        // Change regime periodically
        if (i % 100 == 0) {
            select_new_regime();
        }
        
        Bar bar = generate_market_bar(symbol, current_time);
        bars.push_back(bar);
        
        current_price = bar.close;
        current_time += std::chrono::seconds(interval_seconds);
    }
    
    std::cout << "✅ Generated " << bars.size() << " bars for " << symbol << std::endl;
    return bars;
}

Bar VirtualMarketEngine::generate_market_bar(const std::string& symbol, 
                                           std::chrono::system_clock::time_point timestamp) {
    double current_price = base_prices_[symbol];
    
    // Calculate price movement
    double price_move = calculate_price_movement(symbol, current_price, current_regime_);
    double new_price = current_price * (1 + price_move);
    
    // Generate OHLC from price movement
    double open_price = current_price;
    double close_price = new_price;
    
    // High and low with realistic intrabar movement
    double intrabar_range = std::abs(price_move) * (0.3 + (rng_() % 50) / 100.0);
    double high_price = std::max(open_price, close_price) + current_price * intrabar_range * (rng_() % 100) / 100.0;
    double low_price = std::min(open_price, close_price) - current_price * intrabar_range * (rng_() % 100) / 100.0;
    
    // Realistic volume model based on real QQQ data
    int base_volume = 80000 + (rng_() % 120000); // 80K-200K base volume
    double volume_multiplier = current_regime_.volume_multiplier * (1 + std::abs(price_move) * 2.0);
    
    // Add time-of-day volume patterns (higher volume during market open/close)
    double time_multiplier = 1.0;
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);
    std::tm* tm_info = std::localtime(&time_t);
    int hour = tm_info->tm_hour;
    
    // Market hours volume patterns (9:30-16:00 ET)
    if (hour >= 9 && hour <= 16) {
        if (hour == 9 || hour == 16) {
            time_multiplier = 1.5; // Higher volume at open/close
        } else if (hour == 10 || hour == 15) {
            time_multiplier = 1.2; // Moderate volume
        } else {
            time_multiplier = 1.0; // Normal volume
        }
    } else {
        time_multiplier = 0.3; // Lower volume outside market hours
    }
    
    int volume = static_cast<int>(base_volume * volume_multiplier * time_multiplier);
    
    Bar bar;
    bar.ts_utc_epoch = std::chrono::duration_cast<std::chrono::seconds>(timestamp.time_since_epoch()).count();
    bar.open = open_price;
    bar.high = high_price;
    bar.low = low_price;
    bar.close = close_price;
    bar.volume = volume;
    
    return bar;
}

double VirtualMarketEngine::calculate_price_movement(const std::string& symbol, 
                                                    double current_price,
                                                    const MarketRegime& regime) {
    // Calculate price movement components
    double dt = 1.0 / (252 * 390); // 1 minute as fraction of trading year
    
    // Trend component
    double trend_move = regime.trend * dt;
    
    // Random walk component
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    double volatility_move = regime.volatility * std::sqrt(dt) * normal_dist(rng_);
    
    // Mean reversion component
    double base_price = base_prices_[symbol];
    double reversion_move = regime.mean_reversion * (base_price - current_price) * dt * 0.01;
    
    // Microstructure noise
    double microstructure_noise = normal_dist(rng_) * 0.0005;
    
    return trend_move + volatility_move + reversion_move + microstructure_noise;
}

std::vector<VirtualMarketEngine::VMSimulationResult> VirtualMarketEngine::run_monte_carlo_simulation(
    const VMTestConfig& config,
    std::unique_ptr<BaseStrategy> strategy,
    const RunnerCfg& runner_cfg) {
    
    std::vector<VMSimulationResult> results;
    results.reserve(config.simulations);
    
    int periods = config.hours > 0 ? config.hours * 60 : config.days * 390;
    
    std::cout << "🎲 Running " << config.simulations << " Monte Carlo simulations..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < config.simulations; ++i) {
        // Progress reporting
        if ((i + 1) % 25 == 0 || i == 0) {
            auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
            auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            double progress_pct = ((i + 1) / static_cast<double>(config.simulations)) * 100;
            double eta = elapsed_sec / ((i + 1) / static_cast<double>(config.simulations)) - elapsed_sec;
            
            std::cout << "📊 Progress: " << std::fixed << std::setprecision(1) << progress_pct 
                      << "% | Sim " << std::setw(4) << (i + 1) << "/" << std::setw(4) << config.simulations
                      << " | Elapsed: " << std::setw(5) << elapsed_sec << "s"
                      << " | ETA: " << std::setw(5) << static_cast<int>(eta) << "s" << std::endl;
        }
        
        // Generate synthetic market data
        std::vector<Bar> market_data = generate_market_data(config.symbol, periods, 60);
        
        // Create a new strategy instance for this simulation
        auto sim_strategy = StrategyFactory::instance().create_strategy(config.strategy_name);
        
        // Run single simulation
        VMSimulationResult result = run_single_simulation(market_data, 
                                                         std::move(sim_strategy),
                                                         runner_cfg, 
                                                         config.initial_capital);
        results.push_back(result);
    }
    
    auto total_time = std::chrono::high_resolution_clock::now() - start_time;
    auto total_sec = std::chrono::duration_cast<std::chrono::seconds>(total_time).count();
    
    std::cout << "⏱️  Completed " << config.simulations << " simulations in " 
              << total_sec << " seconds" << std::endl;
    
    return results;
}

std::vector<VirtualMarketEngine::VMSimulationResult> VirtualMarketEngine::run_mars_monte_carlo_simulation(
    const VMTestConfig& config,
    std::unique_ptr<BaseStrategy> strategy,
    const RunnerCfg& runner_cfg,
    const std::string& market_regime) {
    
    std::vector<VMSimulationResult> results;
    results.reserve(config.simulations);
    
    std::cout << "🚀 Starting MarS Monte Carlo simulation..." << std::endl;
    std::cout << "📊 Strategy: " << config.strategy_name << std::endl;
    std::cout << "📈 Symbol: " << config.symbol << std::endl;
    std::cout << "⏱️  Duration: " << config.days << " days" << std::endl;
    std::cout << "🎲 Simulations: " << config.simulations << std::endl;
    std::cout << "🌊 Market Regime: " << market_regime << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < config.simulations; ++i) {
        if (i % std::max(1, config.simulations / 10) == 0) {
            auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
            auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            auto eta = (elapsed_sec * config.simulations) / (i + 1) - elapsed_sec;
            
            std::cout << "📊 Progress: " << std::fixed << std::setprecision(1) 
                      << (100.0 * i / config.simulations) << "% | Sim " 
                      << std::setw(4) << (i + 1) << "/" << std::setw(4) << config.simulations
                      << " | Elapsed: " << std::setw(6) << elapsed_sec << "s | ETA: " 
                      << std::setw(6) << eta << "s" << std::endl;
        }
        
        try {
            // Generate MarS data for this simulation
            int periods = config.days * 24 * 60; // Convert days to minutes
            auto mars_data = MarsDataLoader::load_mars_data(
                config.symbol, periods, 60, 1, market_regime
            );
            
            if (mars_data.empty()) {
                std::cerr << "Warning: No MarS data generated for simulation " << (i + 1) << std::endl;
                results.push_back(VMSimulationResult{}); // Add empty result
                continue;
            }
            
            // Create new strategy instance for this simulation
            auto strategy_instance = StrategyFactory::instance().create_strategy(
                config.strategy_name
            );
            
            if (!strategy_instance) {
                std::cerr << "Error: Could not create strategy instance for simulation " << (i + 1) << std::endl;
                results.push_back(VMSimulationResult{}); // Add empty result
                continue;
            }
            
            // Run simulation with MarS data
            auto result = run_single_simulation(
                mars_data, std::move(strategy_instance), runner_cfg, config.initial_capital
            );
            
            results.push_back(result);
            
        } catch (const std::exception& e) {
            std::cerr << "Error in MarS simulation " << (i + 1) << ": " << e.what() << std::endl;
            results.push_back(VMSimulationResult{}); // Add empty result
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    std::cout << "⏱️  Completed " << config.simulations << " MarS simulations in " 
              << total_time << " seconds" << std::endl;
    
    return results;
}

std::vector<VirtualMarketEngine::VMSimulationResult> VirtualMarketEngine::run_mars_vm_test(
    const std::string& strategy_name,
    const std::string& symbol,
    int days,
    int simulations,
    const std::string& market_regime,
    const std::string& params_json) {
    
    // Create VM test configuration
    VMTestConfig config;
    config.strategy_name = strategy_name;
    config.symbol = symbol;
    config.days = days;
    config.simulations = simulations;
    config.params_json = params_json;
    config.initial_capital = 100000.0;
    
    // Create strategy instance
    auto strategy = StrategyFactory::instance().create_strategy(strategy_name);
    if (!strategy) {
        std::cerr << "Error: Could not create strategy " << strategy_name << std::endl;
        return {};
    }
    
    // Create runner configuration
    RunnerCfg runner_cfg;
    runner_cfg.strategy_name = strategy_name;
    runner_cfg.strategy_params["buy_hi"] = "0.6";
    runner_cfg.strategy_params["sell_lo"] = "0.4";
    
    // Run MarS Monte Carlo simulation
    return run_mars_monte_carlo_simulation(config, std::move(strategy), runner_cfg, market_regime);
}

std::vector<VirtualMarketEngine::VMSimulationResult> VirtualMarketEngine::run_fast_historical_test(
    const std::string& strategy_name,
    const std::string& symbol,
    const std::string& historical_data_file,
    int continuation_minutes,
    int simulations,
    const std::string& params_json) {
    
    std::vector<VMSimulationResult> results;
    results.reserve(simulations);
    
    std::cout << "⚡ Starting Fast Historical Test..." << std::endl;
    std::cout << "📊 Strategy: " << strategy_name << std::endl;
    std::cout << "📈 Symbol: " << symbol << std::endl;
    std::cout << "📊 Historical data: " << historical_data_file << std::endl;
    std::cout << "⏱️  Continuation: " << continuation_minutes << " minutes" << std::endl;
    std::cout << "🎲 Simulations: " << simulations << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < simulations; ++i) {
        if (i % std::max(1, simulations / 10) == 0) {
            auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
            auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            auto eta = (elapsed_sec * simulations) / (i + 1) - elapsed_sec;
            
            std::cout << "📊 Progress: " << std::fixed << std::setprecision(1) 
                      << (100.0 * i / simulations) << "% | Sim " 
                      << std::setw(4) << (i + 1) << "/" << std::setw(4) << simulations
                      << " | Elapsed: " << std::setw(6) << elapsed_sec << "s | ETA: " 
                      << std::setw(6) << eta << "s" << std::endl;
        }
        
        try {
            // Generate fast historical data for this simulation
            auto fast_data = MarsDataLoader::load_fast_historical_data(
                symbol, historical_data_file, continuation_minutes
            );
            
            if (fast_data.empty()) {
                std::cerr << "Warning: No fast historical data generated for simulation " << (i + 1) << std::endl;
                results.push_back(VMSimulationResult{}); // Add empty result
                continue;
            }
            
            // Create new strategy instance for this simulation
            auto strategy_instance = StrategyFactory::instance().create_strategy(
                strategy_name
            );
            
            if (!strategy_instance) {
                std::cerr << "Error: Could not create strategy instance for simulation " << (i + 1) << std::endl;
                results.push_back(VMSimulationResult{}); // Add empty result
                continue;
            }
            
            // Create runner configuration
            RunnerCfg runner_cfg;
            runner_cfg.strategy_name = strategy_name;
            runner_cfg.strategy_params["buy_hi"] = "0.6";
            runner_cfg.strategy_params["sell_lo"] = "0.4";
            
            // Run simulation with fast historical data
            auto result = run_single_simulation(
                fast_data, std::move(strategy_instance), runner_cfg, 100000.0
            );
            
            results.push_back(result);
            
        } catch (const std::exception& e) {
            std::cerr << "Error in fast historical simulation " << (i + 1) << ": " << e.what() << std::endl;
            results.push_back(VMSimulationResult{}); // Add empty result
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    std::cout << "⏱️  Completed " << simulations << " fast historical simulations in " 
              << total_time << " seconds" << std::endl;
    
    // Print comprehensive report
    VMTestConfig config;
    config.strategy_name = strategy_name;
    config.symbol = symbol;
    config.simulations = simulations;
    config.params_json = params_json;
    config.initial_capital = 100000.0;
    print_simulation_report(results, config);
    
    return results;
}

VirtualMarketEngine::VMSimulationResult VirtualMarketEngine::run_single_simulation(
    const std::vector<Bar>& market_data,
    std::unique_ptr<BaseStrategy> strategy,
    const RunnerCfg& runner_cfg,
    double initial_capital) {
    
    // REAL INTEGRATION: Use actual Runner component
    VMSimulationResult result;
    
    try {
        // 1. Create SymbolTable for the test symbol
        SymbolTable ST;
        int symbol_id = ST.intern("QQQ"); // Use QQQ as the base symbol
        
        // 2. Create series data structure (single symbol)
        std::vector<std::vector<Bar>> series(1);
        series[0] = market_data;
        
        // 3. Create audit recorder with minimal config for VM test
        AuditConfig audit_cfg;
        audit_cfg.run_id = "vmtest_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        audit_cfg.file_path = "/dev/null"; // Don't write audit files for VM test
        audit_cfg.flush_each = false;
        
        AuditRecorder audit(audit_cfg);
        
        // 4. Run REAL backtest using actual Runner
        RunResult run_result = run_backtest(audit, ST, series, symbol_id, runner_cfg);
        
        // 5. Extract performance metrics from real results
        result.total_return = run_result.total_return;
        result.final_capital = initial_capital * (1 + run_result.total_return);
        result.sharpe_ratio = run_result.sharpe_ratio;
        result.max_drawdown = run_result.max_drawdown;
        result.win_rate = 0.0; // Not available in RunResult, calculate separately if needed
        result.total_trades = run_result.total_fills;
        result.monthly_projected_return = run_result.monthly_projected_return;
        result.daily_trades = static_cast<double>(run_result.daily_trades);
        
    } catch (const std::exception& e) {
        std::cerr << "❌ VM simulation failed: " << e.what() << std::endl;
        
        // Return zero results on error
        result.total_return = 0.0;
        result.final_capital = initial_capital;
        result.sharpe_ratio = 0.0;
        result.max_drawdown = 0.0;
        result.win_rate = 0.0;
        result.total_trades = 0;
        result.monthly_projected_return = 0.0;
        result.daily_trades = 0.0;
    }
    
    return result;
}

VirtualMarketEngine::VMSimulationResult VirtualMarketEngine::calculate_performance_metrics(
    const std::vector<double>& returns,
    const std::vector<int>& trades,
    double initial_capital) {
    
    VMSimulationResult result;
    
    if (returns.empty()) {
        result.total_return = 0.0;
        result.final_capital = initial_capital;
        result.sharpe_ratio = 0.0;
        result.max_drawdown = 0.0;
        result.win_rate = 0.0;
        result.total_trades = 0;
        result.monthly_projected_return = 0.0;
        result.daily_trades = 0.0;
        return result;
    }
    
    // Calculate basic metrics
    result.total_return = returns.back();
    result.final_capital = initial_capital * (1 + result.total_return);
    result.total_trades = trades.size();
    
    // Calculate Sharpe ratio
    if (returns.size() > 1) {
        std::vector<double> daily_returns;
        for (size_t i = 1; i < returns.size(); ++i) {
            daily_returns.push_back(returns[i] - returns[i-1]);
        }
        
        double mean_return = std::accumulate(daily_returns.begin(), daily_returns.end(), 0.0) / daily_returns.size();
        double variance = 0.0;
        for (double ret : daily_returns) {
            variance += (ret - mean_return) * (ret - mean_return);
        }
        double std_dev = std::sqrt(variance / daily_returns.size());
        
        result.sharpe_ratio = (std_dev > 0) ? (mean_return / std_dev * std::sqrt(252)) : 0.0;
    }
    
    // Calculate max drawdown
    double peak = initial_capital;
    double max_dd = 0.0;
    for (double ret : returns) {
        double current_value = initial_capital * (1 + ret);
        if (current_value > peak) {
            peak = current_value;
        }
        double dd = (peak - current_value) / peak;
        max_dd = std::max(max_dd, dd);
    }
    result.max_drawdown = max_dd;
    
    // Calculate win rate
    int winning_trades = std::count_if(trades.begin(), trades.end(), [](int trade) { return trade > 0; });
    result.win_rate = (trades.empty()) ? 0.0 : static_cast<double>(winning_trades) / trades.size();
    
    // Calculate monthly projected return
    double annual_return = result.total_return * (252.0 / returns.size());
    result.monthly_projected_return = annual_return / 12.0;
    
    // Calculate daily trades
    result.daily_trades = static_cast<double>(result.total_trades) / returns.size();
    
    return result;
}

void VirtualMarketEngine::print_simulation_report(const std::vector<VMSimulationResult>& results,
                                                 const VMTestConfig& config) {
    if (results.empty()) {
        std::cout << "❌ No simulation results to report" << std::endl;
        return;
    }
    
    // Calculate statistics
    std::vector<double> returns;
    std::vector<double> sharpe_ratios;
    std::vector<double> monthly_returns;
    std::vector<double> daily_trades;
    
    for (const auto& result : results) {
        returns.push_back(result.total_return);
        sharpe_ratios.push_back(result.sharpe_ratio);
        monthly_returns.push_back(result.monthly_projected_return);
        daily_trades.push_back(result.daily_trades);
    }
    
    // Sort for percentiles
    std::sort(returns.begin(), returns.end());
    std::sort(sharpe_ratios.begin(), sharpe_ratios.end());
    std::sort(monthly_returns.begin(), monthly_returns.end());
    std::sort(daily_trades.begin(), daily_trades.end());
    
    // Calculate statistics
    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double median_return = returns[returns.size() / 2];
    double std_return = 0.0;
    for (double ret : returns) {
        std_return += (ret - mean_return) * (ret - mean_return);
    }
    std_return = std::sqrt(std_return / returns.size());
    
    double mean_sharpe = std::accumulate(sharpe_ratios.begin(), sharpe_ratios.end(), 0.0) / sharpe_ratios.size();
    double mean_mpr = std::accumulate(monthly_returns.begin(), monthly_returns.end(), 0.0) / monthly_returns.size();
    double mean_daily_trades = std::accumulate(daily_trades.begin(), daily_trades.end(), 0.0) / daily_trades.size();
    
    // Probability analysis
    int profitable_sims = std::count_if(returns.begin(), returns.end(), [](double ret) { return ret > 0; });
    double prob_profit = static_cast<double>(profitable_sims) / returns.size() * 100;
    
    // Print report
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "📊 VIRTUAL MARKET TEST RESULTS" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Strategy:                 " << config.strategy_name << std::endl;
    std::cout << "Symbol:                   " << config.symbol << std::endl;
    std::cout << "Simulations:              " << config.simulations << std::endl;
    std::cout << "Simulation Period:        " << (config.hours > 0 ? std::to_string(config.hours) + " hours" : std::to_string(config.days) + " days") << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "📈 RETURN STATISTICS" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Mean Return:              " << std::setw(8) << std::fixed << std::setprecision(2) << mean_return * 100 << "%" << std::endl;
    std::cout << "Median Return:            " << std::setw(8) << std::fixed << std::setprecision(2) << median_return * 100 << "%" << std::endl;
    std::cout << "Standard Deviation:       " << std::setw(8) << std::fixed << std::setprecision(2) << std_return * 100 << "%" << std::endl;
    std::cout << "Minimum Return:           " << std::setw(8) << std::fixed << std::setprecision(2) << returns.front() * 100 << "%" << std::endl;
    std::cout << "Maximum Return:           " << std::setw(8) << std::fixed << std::setprecision(2) << returns.back() * 100 << "%" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "📊 CONFIDENCE INTERVALS" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "5th Percentile:           " << std::setw(8) << std::fixed << std::setprecision(2) << returns[static_cast<size_t>(returns.size() * 0.05)] * 100 << "%" << std::endl;
    std::cout << "25th Percentile:          " << std::setw(8) << std::fixed << std::setprecision(2) << returns[static_cast<size_t>(returns.size() * 0.25)] * 100 << "%" << std::endl;
    std::cout << "75th Percentile:          " << std::setw(8) << std::fixed << std::setprecision(2) << returns[static_cast<size_t>(returns.size() * 0.75)] * 100 << "%" << std::endl;
    std::cout << "95th Percentile:          " << std::setw(8) << std::fixed << std::setprecision(2) << returns[static_cast<size_t>(returns.size() * 0.95)] * 100 << "%" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "🎯 PROBABILITY ANALYSIS" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Probability of Profit:    " << std::setw(8) << std::fixed << std::setprecision(1) << prob_profit << "%" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "📋 ADDITIONAL METRICS" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Mean Sharpe Ratio:        " << std::setw(8) << std::fixed << std::setprecision(2) << mean_sharpe << std::endl;
    std::cout << "Mean MPR (Monthly):       " << std::setw(8) << std::fixed << std::setprecision(2) << mean_mpr * 100 << "%" << std::endl;
    std::cout << "Mean Daily Trades:        " << std::setw(8) << std::fixed << std::setprecision(1) << mean_daily_trades << std::endl;
    std::cout << std::string(70, '=') << std::endl << std::endl;
}

// VMTestRunner implementation
int VMTestRunner::run_vmtest(const VirtualMarketEngine::VMTestConfig& config) {
    std::cout << "🚀 Starting Virtual Market Test..." << std::endl;
    std::cout << "📊 Strategy: " << config.strategy_name << std::endl;
    std::cout << "📈 Symbol: " << config.symbol << std::endl;
    std::cout << "⏱️  Duration: " << (config.hours > 0 ? std::to_string(config.hours) + " hours" : std::to_string(config.days) + " days") << std::endl;
    std::cout << "🎲 Simulations: " << config.simulations << std::endl;
    std::cout << "⚡ Fast Mode: " << (config.fast_mode ? "enabled" : "disabled") << std::endl;
    
    // Create strategy
    auto strategy = create_strategy(config.strategy_name, config.params_json);
    if (!strategy) {
        std::cerr << "❌ Failed to create strategy: " << config.strategy_name << std::endl;
        return 1;
    }
    
    // Load strategy configuration
    RunnerCfg runner_cfg = load_strategy_config(config.strategy_name);
    
    // Run Monte Carlo simulation
    auto results = vm_engine_.run_monte_carlo_simulation(config, std::move(strategy), runner_cfg);
    
    // Print comprehensive report
    vm_engine_.print_simulation_report(results, config);
    
    return 0;
}

std::unique_ptr<BaseStrategy> VMTestRunner::create_strategy(const std::string& strategy_name,
                                                           const std::string& params_json) {
    // Use StrategyFactory to create strategy (same as Runner)
    return StrategyFactory::instance().create_strategy(strategy_name);
}

RunnerCfg VMTestRunner::load_strategy_config(const std::string& strategy_name) {
    RunnerCfg cfg;
    cfg.strategy_name = strategy_name;
    cfg.audit_level = AuditLevel::Full;
    cfg.snapshot_stride = 100;
    
    // Load configuration from JSON files
    // TODO: Implement JSON configuration loading
    
    return cfg;
}

} // namespace sentio
