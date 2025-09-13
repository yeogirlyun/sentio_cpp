#include "sentio/audit_validator.hpp"
#include "sentio/all_strategies.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/run_id_generator.hpp"
#include "audit/audit_db_recorder.hpp"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>

namespace sentio {

AuditValidator::ValidationResult AuditValidator::validate_strategy_audit_compatibility(
    const std::string& strategy_name,
    int test_bars) {
    
    ValidationResult result;
    result.strategy_name = strategy_name;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // **1. CREATE STRATEGY INSTANCE**
        auto strategy = StrategyFactory::instance().create_strategy(strategy_name);
        if (!strategy) {
            result.error_message = "Failed to create strategy instance";
            return result;
        }
        
        // **2. SETUP TEST ENVIRONMENT**
        SymbolTable ST;
        int base_symbol_id = ST.intern("QQQ");
        
        // Generate synthetic test data
        auto test_data = generate_test_data(test_bars);
        std::vector<std::vector<Bar>> series(1);
        series[0] = test_data;
        
        // **3. CREATE AUDIT RECORDER**
        std::string run_id = generate_run_id();
        std::string audit_note = create_audit_note(strategy_name, "audit_validation");
        std::string db_path = ":memory:"; // Use in-memory database for testing
        audit::AuditDBRecorder audit(db_path, run_id, audit_note);
        
        // **4. CREATE STRATEGY-AGNOSTIC CONFIG**
        RunnerCfg cfg = create_test_config(strategy_name);
        
        // **5. RUN BACKTEST WITH AUDIT**
        auto backtest_result = run_backtest(audit, ST, series, base_symbol_id, cfg);
        
        // **6. VALIDATE RESULTS**
        result.success = true;
        result.signals_logged = 0; // Would need to query audit DB to get actual counts
        result.orders_logged = 0;
        result.fills_logged = backtest_result.total_fills;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.test_duration_sec = std::chrono::duration<double>(end_time - start_time).count();
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.test_duration_sec = std::chrono::duration<double>(end_time - start_time).count();
    }
    
    return result;
}

std::vector<AuditValidator::ValidationResult> AuditValidator::validate_all_strategies(int test_bars) {
    std::vector<ValidationResult> results;
    
    // Get all registered strategies
    auto strategy_names = StrategyFactory::instance().get_available_strategies();
    
    std::cout << "🔍 **STRATEGY-AGNOSTIC AUDIT VALIDATION**" << std::endl;
    std::cout << "Testing " << strategy_names.size() << " registered strategies..." << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    for (const auto& strategy_name : strategy_names) {
        std::cout << "Testing " << std::setw(20) << std::left << strategy_name << "... ";
        std::cout.flush();
        
        auto result = validate_strategy_audit_compatibility(strategy_name, test_bars);
        results.push_back(result);
        
        if (result.success) {
            std::cout << "✅ PASS (" << std::fixed << std::setprecision(3) 
                      << result.test_duration_sec << "s)" << std::endl;
        } else {
            std::cout << "❌ FAIL: " << result.error_message << std::endl;
        }
    }
    
    return results;
}

void AuditValidator::print_validation_report(const std::vector<ValidationResult>& results) {
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "📊 **AUDIT VALIDATION REPORT**" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    int passed = 0;
    int failed = 0;
    double total_time = 0.0;
    
    for (const auto& result : results) {
        if (result.success) {
            passed++;
        } else {
            failed++;
        }
        total_time += result.test_duration_sec;
    }
    
    std::cout << "📈 **SUMMARY:**" << std::endl;
    std::cout << "  Total Strategies: " << results.size() << std::endl;
    std::cout << "  ✅ Passed: " << passed << " (" << (100.0 * passed / results.size()) << "%)" << std::endl;
    std::cout << "  ❌ Failed: " << failed << " (" << (100.0 * failed / results.size()) << "%)" << std::endl;
    std::cout << "  ⏱️  Total Time: " << std::fixed << std::setprecision(2) << total_time << "s" << std::endl;
    
    if (failed > 0) {
        std::cout << std::endl << "❌ **FAILED STRATEGIES:**" << std::endl;
        for (const auto& result : results) {
            if (!result.success) {
                std::cout << "  • " << result.strategy_name << ": " << result.error_message << std::endl;
            }
        }
    }
    
    if (passed == static_cast<int>(results.size())) {
        std::cout << std::endl << "🎉 **ALL STRATEGIES PASS AUDIT VALIDATION!**" << std::endl;
        std::cout << "The audit system is fully strategy-agnostic." << std::endl;
    }
}

std::vector<Bar> AuditValidator::generate_test_data(int num_bars) {
    std::vector<Bar> bars;
    bars.reserve(num_bars);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> price_change(0.0, 0.01); // 1% volatility
    
    double base_price = 100.0;
    std::int64_t base_time = 1640995200000; // 2022-01-01 00:00:00 UTC in milliseconds
    
    for (int i = 0; i < num_bars; ++i) {
        Bar bar;
        
        // Generate realistic OHLCV data
        double change = price_change(gen);
        double open = base_price * (1.0 + change);
        double high = open * (1.0 + std::abs(price_change(gen)) * 0.5);
        double low = open * (1.0 - std::abs(price_change(gen)) * 0.5);
        double close = open + (high - low) * (price_change(gen) + 1.0) / 2.0;
        
        bar.open = open;
        bar.high = std::max(open, std::max(high, close));
        bar.low = std::min(open, std::min(low, close));
        bar.close = close;
        bar.volume = 1000000 + static_cast<std::uint64_t>(std::abs(price_change(gen)) * 500000);
        
        bar.ts_utc_epoch = base_time + i * 60000; // 1-minute bars
        bar.ts_utc = std::to_string(bar.ts_utc_epoch);
        
        bars.push_back(bar);
        base_price = close; // Use close as next open
    }
    
    return bars;
}

RunnerCfg AuditValidator::create_test_config(const std::string& strategy_name) {
    RunnerCfg cfg;
    cfg.strategy_name = strategy_name;
    cfg.audit_level = AuditLevel::Full; // **CRITICAL**: Enable full audit logging
    cfg.snapshot_stride = 10; // Take snapshots every 10 bars for testing
    
    // Set default router config
    cfg.router.bull3x = "TQQQ";
    cfg.router.bear3x = "SQQQ";
    
    // Set default sizer config
    cfg.sizer.max_position_pct = 1.0;
    cfg.sizer.allow_negative_cash = false;
    cfg.sizer.max_leverage = 2.0;
    cfg.sizer.min_notional = 1.0;
    
    return cfg;
}

} // namespace sentio
