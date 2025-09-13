#include "sentio/core.hpp"
#include "sentio/runner.hpp"
#include "sentio/temporal_analysis.hpp"
#include "sentio/csv_loader.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/profiling.hpp"
#include "sentio/data_resolver.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/all_strategies.hpp"
#include "sentio/feature_feeder.hpp"
#include "sentio/data_downloader.hpp"
#include "sentio/audit_validator.hpp"
#include "sentio/feature/feature_matrix.hpp"
#include "sentio/strategy_tfa.hpp"
#include "sentio/virtual_market.hpp"
#include "sentio/unified_strategy_tester.hpp"
#include "sentio/cli_helpers.hpp"

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <cstdlib>
#include <sstream>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <ATen/Parallel.h>
#include <nlohmann/json.hpp>

static bool verify_series_alignment(const sentio::SymbolTable& ST,
                                    const std::vector<std::vector<sentio::Bar>>& series,
                                    int base_sid){
    const auto& base = series[base_sid];
    if (base.empty()) return false;
    const size_t N = base.size();
    bool ok = true;
    for (size_t sid = 0; sid < series.size(); ++sid) {
        if (sid == (size_t)base_sid) continue;
        const auto& s = series[sid];
        if (s.empty()) continue; // allow missing non-base
        if (s.size() != N) {
            std::cerr << "FATAL: Alignment check failed: " << ST.get_symbol((int)sid)
                      << " bars=" << s.size() << " != base(" << ST.get_symbol(base_sid)
                      << ") bars=" << N << "\n";
            ok = false;
            continue;
        }
        // Check timestamp alignment for first and last bars
        if (s[0].ts_utc_epoch != base[0].ts_utc_epoch) {
            std::cerr << "FATAL: Start timestamp mismatch: " << ST.get_symbol((int)sid)
                      << " vs " << ST.get_symbol(base_sid) << "\n";
            ok = false;
        }
        if (s[N-1].ts_utc_epoch != base[N-1].ts_utc_epoch) {
            std::cerr << "FATAL: End timestamp mismatch: " << ST.get_symbol((int)sid)
                      << " vs " << ST.get_symbol(base_sid) << "\n";
            ok = false;
        }
    }
    return ok;
}

void usage() {
    std::cout << "Usage: sentio_cli <command> [options]\n\n"
              << "STRATEGY TESTING:\n"
              << "  strattest <strategy> <symbol> [options]    Unified strategy robustness testing\n"
              << "\n"
              << "DATA MANAGEMENT:\n"
              << "  download <symbol> [options]               Download historical data from Polygon.io\n"
              << "  probe                                     Show data availability and system status\n"
              << "\n"
              << "DEVELOPMENT & VALIDATION:\n"
              << "  audit-validate                            Validate strategies with audit system\n"
              << "\n"
              << "Global Options:\n"
              << "  --help, -h                                Show command-specific help\n"
              << "  --verbose, -v                             Enable verbose output\n"
              << "  --output <format>                         Output format: console|json|csv\n"
              << "\n"
              << "Examples:\n"
              << "  sentio_cli strattest momentum QQQ --mode hybrid --duration 1w\n"
              << "  sentio_cli strattest ire QQQ --comprehensive --stress-test\n"
              << "  sentio_cli download QQQ --period 3y\n"
              << "  sentio_cli probe\n"
              << "\n"
              << "Use 'sentio_cli <command> --help' for command-specific options.\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // Initialize strategies using factory pattern
    if (!sentio::initialize_strategies()) {
        std::cerr << "Warning: Failed to initialize strategies" << std::endl;
    }
    
    if (argc < 2) {
        usage();
        return 1;
    }

    // Parse arguments using CLI helpers
    auto args = sentio::CLIHelpers::parse_arguments(argc, argv);
    std::string command = args.command;
    
    // Handle global help
    if (command.empty() || (args.help_requested && command.empty())) {
        usage();
        return 0;
    }
    
    if (command == "strattest") {
        if (args.help_requested) {
            sentio::CLIHelpers::print_help("strattest", 
                "sentio_cli strattest <strategy> <symbol> [options]",
                {
                    "--mode <mode>              Simulation mode: monte-carlo|historical|ai-regime|hybrid (default: hybrid)",
                    "--simulations <n>          Number of simulations (default: 50)",
                    "--duration <period>        Test duration: 1h, 4h, 1d, 5d, 1w, 1m (default: 5d)",
                    "--historical-data <file>   Historical data file (auto-detect if not specified)",
                    "--regime <regime>          Market regime: normal|volatile|trending|bear|bull (default: normal)",
                    "--stress-test              Enable stress testing scenarios",
                    "--regime-switching         Test across multiple market regimes",
                    "--liquidity-stress         Simulate low liquidity conditions",
                    "--alpaca-fees              Use Alpaca fee structure (default: true)",
                    "--alpaca-limits            Apply Alpaca position/order limits",
                    "--confidence <level>       Confidence level: 90|95|99 (default: 95)",
                    "--output <format>          Output format: console|json|csv (default: console)",
                    "--save-results <file>      Save detailed results to file",
                    "--benchmark <symbol>       Benchmark symbol (default: SPY)",
                    "--quick                    Quick mode: fewer simulations, faster execution",
                    "--comprehensive            Comprehensive mode: extensive testing scenarios",
                    "--params <json>            Strategy parameters as JSON string (default: '{}')"
                },
                {
                    "sentio_cli strattest momentum QQQ --mode hybrid --duration 1w",
                    "sentio_cli strattest ire QQQ --comprehensive --stress-test",
                    "sentio_cli strattest momentum QQQ --mode monte-carlo --simulations 100",
                    "sentio_cli strattest ire SPY --mode ai-regime --regime volatile"
                });
            return 0;
        }
        
        if (!sentio::CLIHelpers::validate_required_args(args, 2, 
            "sentio_cli strattest <strategy> <symbol> [options]")) {
            return 1;
        }
        
        std::string strategy_name = args.positional_args[0];
        std::string symbol = args.positional_args[1];
        
        // Validate inputs
        if (!sentio::CLIHelpers::is_valid_strategy_name(strategy_name)) {
            sentio::CLIHelpers::print_error("Invalid strategy name: " + strategy_name);
            return 1;
        }
        
        if (!sentio::CLIHelpers::is_valid_symbol(symbol)) {
            sentio::CLIHelpers::print_error("Invalid symbol: " + symbol);
            return 1;
        }
        
        // Build test configuration
        sentio::UnifiedStrategyTester::TestConfig config;
        config.strategy_name = strategy_name;
        config.symbol = symbol;
        
        // Parse options
        std::string mode_str = sentio::CLIHelpers::get_option(args, "mode", "hybrid");
        config.mode = sentio::UnifiedStrategyTester::parse_test_mode(mode_str);
        
        config.simulations = sentio::CLIHelpers::get_int_option(args, "simulations", 50);
        config.duration = sentio::CLIHelpers::get_option(args, "duration", "5d");
        config.historical_data_file = sentio::CLIHelpers::get_option(args, "historical-data");
        config.regime = sentio::CLIHelpers::get_option(args, "regime", "normal");
        
        config.stress_test = sentio::CLIHelpers::get_flag(args, "stress-test");
        config.regime_switching = sentio::CLIHelpers::get_flag(args, "regime-switching");
        config.liquidity_stress = sentio::CLIHelpers::get_flag(args, "liquidity-stress");
        
        config.alpaca_fees = !sentio::CLIHelpers::get_flag(args, "no-alpaca-fees"); // Default true
        config.alpaca_limits = sentio::CLIHelpers::get_flag(args, "alpaca-limits");
        config.paper_validation = sentio::CLIHelpers::get_flag(args, "paper-validation");
        
        int confidence_pct = sentio::CLIHelpers::get_int_option(args, "confidence", 95);
        config.confidence_level = confidence_pct / 100.0;
        
        config.output_format = sentio::CLIHelpers::get_option(args, "output", "console");
        config.save_results_file = sentio::CLIHelpers::get_option(args, "save-results");
        config.benchmark_symbol = sentio::CLIHelpers::get_option(args, "benchmark", "SPY");
        
        config.quick_mode = sentio::CLIHelpers::get_flag(args, "quick");
        config.comprehensive_mode = sentio::CLIHelpers::get_flag(args, "comprehensive");
        
        config.params_json = sentio::CLIHelpers::get_option(args, "params", "{}");
        
        // Adjust simulations based on mode flags
        if (config.quick_mode && config.simulations == 50) {
            config.simulations = 20;
        } else if (config.comprehensive_mode && config.simulations == 50) {
            config.simulations = 100;
        }
        
        // Run unified strategy test
        try {
            sentio::UnifiedStrategyTester tester;
            auto report = tester.run_comprehensive_test(config);
            
            // Display results
            if (config.output_format == "console") {
                tester.print_robustness_report(report, config);
            } else {
                // Save to file or output in other formats
                if (!config.save_results_file.empty()) {
                    if (tester.save_report(report, config, config.save_results_file, config.output_format)) {
                        std::cout << "Results saved to: " << config.save_results_file << std::endl;
                    } else {
                        std::cerr << "Failed to save results to: " << config.save_results_file << std::endl;
                        return 1;
                    }
                }
            }
            
            // Return deployment readiness as exit code (0 = ready, 1 = not ready)
            return report.ready_for_deployment ? 0 : 1;
            
        } catch (const std::exception& e) {
            std::cerr << "Error running strategy test: " << e.what() << std::endl;
            return 1;
        }
        
    } else if (command == "probe") {
        std::cout << "=== SENTIO SYSTEM PROBE ===\n\n";
        
        // Show available strategies
        std::cout << "📊 Available Strategies (" << sentio::StrategyFactory::instance().get_available_strategies().size() << " total):\n";
        std::cout << "=====================\n";
        for (const auto& strategy_name : sentio::StrategyFactory::instance().get_available_strategies()) {
            std::cout << "  • " << strategy_name << "\n";
        }
        std::cout << "\n";
        
        // Helper function to get file info
        auto get_file_info = [](const std::string& path) -> std::pair<bool, std::pair<std::string, std::string>> {
            std::ifstream file(path);
            if (!file.good()) {
                return {false, {"", ""}};
            }
            
            std::string line;
            std::getline(file, line); // Skip header
            
            std::string start_time = "N/A", end_time = "N/A";
            if (std::getline(file, line)) {
                std::istringstream iss(line);
                std::getline(iss, start_time, ',');
                
                // Find last valid data line (skip any trailing header lines)
                std::string current_line = line;
                while (std::getline(file, line)) {
                    // Skip lines that start with "timestamp" (trailing headers in aligned files)
                    if (line.find("timestamp,") != 0 && !line.empty()) {
                        current_line = line;
                    }
                }
                
                if (!current_line.empty()) {
                    std::istringstream iss2(current_line);
                    std::getline(iss2, end_time, ',');
                }
            }
            
            return {true, {start_time, end_time}};
        };
        
        // Check data availability for key symbols
        std::vector<std::string> symbols = {"QQQ", "SPY", "AAPL", "MSFT", "TSLA"};
        std::cout << "📈 Data Availability Check:\n";
        std::cout << "==========================\n";
        
        bool daily_aligned = true, minute_aligned = true;
        
        for (const auto& symbol : symbols) {
            std::cout << "Symbol: " << symbol << "\n";
            
            // Check daily data
            std::string daily_path = "data/equities/" + symbol + "_daily.csv";
            auto [daily_exists, daily_range] = get_file_info(daily_path);
            
            std::cout << "  📅 Daily:  ";
            if (daily_exists) {
                std::cout << "✅ Available (" << daily_range.first << " to " << daily_range.second << ")\n";
            } else {
                std::cout << "❌ Missing\n";
                daily_aligned = false;
            }
            
            // Check minute data
            std::string minute_path = "data/equities/" + symbol + "_NH.csv";
            auto [minute_exists, minute_range] = get_file_info(minute_path);
            
            std::cout << "  ⏰ Minute: ";
            if (minute_exists) {
                std::cout << "✅ Available (" << minute_range.first << " to " << minute_range.second << ")\n";
            } else {
                std::cout << "❌ Missing\n";
                minute_aligned = false;
            }
            
            std::cout << "\n";
        }
        
        // Summary
        std::cout << "📋 Summary:\n";
        std::cout << "===========\n";
        if (daily_aligned && minute_aligned) {
            std::cout << "  🎉 All data is properly aligned and ready for strategy testing!\n";
            std::cout << "  📋 Ready to run: ./build/sentio_cli strattest ire QQQ --comprehensive\n";
        }
        
        std::cout << "\n";
        return 0;
        
    } else if (command == "audit-validate") {
        std::cout << "🔍 **STRATEGY-AGNOSTIC AUDIT VALIDATION**" << std::endl;
        std::cout << "Validating that all registered strategies work with the audit system..." << std::endl;
        std::cout << std::endl;
        
        // Run validation for all strategies
        auto results = sentio::AuditValidator::validate_all_strategies(50); // Test with 50 bars
        
        // Print comprehensive report
        std::cout << "📊 **AUDIT VALIDATION RESULTS**" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        int passed = 0, failed = 0;
        for (const auto& result : results) {
            if (result.success) {
                std::cout << "✅ " << result.strategy_name << " - PASSED" << std::endl;
                passed++;
            } else {
                std::cout << "❌ " << result.strategy_name << " - FAILED: " << result.error_message << std::endl;
                failed++;
            }
        }
        
        std::cout << std::string(50, '=') << std::endl;
        std::cout << "📈 Summary: " << passed << " passed, " << failed << " failed" << std::endl;
        
        if (failed == 0) {
            std::cout << "🎉 All strategies are audit-compatible!" << std::endl;
            return 0;
        } else {
            std::cout << "⚠️  Some strategies need fixes before audit compatibility" << std::endl;
            return 1;
        }
        
    } else if (command == "download") {
        if (args.help_requested) {
            sentio::CLIHelpers::print_help("download",
                "sentio_cli download <symbol> [options]",
                {
                    "--period <period>          Time period: 1y, 6m, 3m, 1m, 2w, 5d (default: 3y)",
                    "--timespan <span>          Data resolution: day|hour|minute (default: minute)",
                    "--holidays                 Include market holidays (default: exclude)",
                    "--output <dir>             Output directory (default: data/equities/)",
                    "--family                   Download symbol family (QQQ -> QQQ,TQQQ,SQQQ)",
                    "--force                    Overwrite existing files"
                },
                {
                    "sentio_cli download QQQ --period 3y",
                    "sentio_cli download SPY --period 1y --timespan day",
                    "sentio_cli download QQQ --family --period 6m"
                });
            return 0;
        }
        
        if (!sentio::CLIHelpers::validate_required_args(args, 1,
            "sentio_cli download <symbol> [options]")) {
            return 1;
        }
        
        std::string symbol = args.positional_args[0];
        
        // Validate symbol
        if (!sentio::CLIHelpers::is_valid_symbol(symbol)) {
            sentio::CLIHelpers::print_error("Invalid symbol: " + symbol);
            return 1;
        }
        
        // Parse options
        std::string period = sentio::CLIHelpers::get_option(args, "period", "3y");
        std::string timespan = sentio::CLIHelpers::get_option(args, "timespan", "minute");
        bool include_holidays = sentio::CLIHelpers::get_flag(args, "holidays");
        std::string output_dir = sentio::CLIHelpers::get_option(args, "output", "data/equities/");
        bool download_family = sentio::CLIHelpers::get_flag(args, "family");
        // bool force_overwrite = sentio::CLIHelpers::get_flag(args, "force"); // TODO: Implement force overwrite
        
        // Build symbol list
        std::vector<std::string> symbols_to_download = {symbol};
        if (download_family && symbol == "QQQ") {
            symbols_to_download.push_back("TQQQ");
            symbols_to_download.push_back("SQQQ");
        }
        
        // Convert period to days
        int days = sentio::CLIHelpers::parse_period_to_days(period);
        
        std::cout << "📥 Downloading data for: ";
        for (const auto& sym : symbols_to_download) {
            std::cout << sym << " ";
        }
        std::cout << std::endl;
        std::cout << "⏱️  Period: " << period << " (" << days << " days)" << std::endl;
        std::cout << "📊 Timespan: " << timespan << std::endl;
        std::cout << "🏖️  Holidays: " << (include_holidays ? "included" : "excluded") << std::endl;
        
        try {
            for (const auto& sym : symbols_to_download) {
                std::cout << "\n📈 Downloading " << sym << "..." << std::endl;
                
                bool success = sentio::download_symbol_data(
                    sym, 0, 0, days, timespan, 1, !include_holidays, output_dir
                );
                
                if (success) {
                    std::cout << "✅ " << sym << " downloaded successfully" << std::endl;
                } else {
                    std::cout << "❌ Failed to download " << sym << std::endl;
                    return 1;
                }
            }
            
            std::cout << "\n🎉 All downloads completed successfully!" << std::endl;
            return 0;
            
        } catch (const std::exception& e) {
            std::cerr << "Error downloading data: " << e.what() << std::endl;
            return 1;
        }
        
    } else {
        usage();
        return 1;
    }
    
    return 0;
}
