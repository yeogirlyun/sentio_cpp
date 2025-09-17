#include "sentio/canonical_evaluation.hpp"
#include "sentio/dataset_metadata.hpp"
#include "sentio/csv_loader.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/all_strategies.hpp"
#include "sentio/data_downloader.hpp"
#include "sentio/cli_helpers.hpp"
#include "sentio/mars_data_loader.hpp"
#include "sentio/run_id_generator.hpp"
#include "sentio/runner.hpp"
#include "sentio/virtual_market.hpp"
#include "audit/audit_db_recorder.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <ctime>
#include <fstream>
#include <iomanip>


void usage() {
    std::cout << "Usage: sentio_cli <command> [options]\n\n"
              << "STRATEGY TESTING:\n"
              << "  strattest <strategy> [symbol] [options]    Unified strategy robustness testing (symbol defaults to QQQ)\n"
              << "  list-strategies [options]                  List all available strategies\n"
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
              << "  sentio_cli list-strategies --format table --verbose\n"
              << "  sentio_cli strattest momentum --mode hybrid --blocks 20\n"
              << "  sentio_cli strattest ire --comprehensive --stress-test\n"
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
                "sentio_cli strattest <strategy> [symbol] [options]",
                {
                    "=== DATA MODES ===",
                    "--mode <mode>              Data mode: historical|simulation|ai-regime (default: simulation)",
                    "                           • historical: Real QQQ market data (data/equities/)",
                    "                           • simulation: MarS pre-generated tracks (data/future_qqq/)",  
                    "                           • ai-regime: Real-time MarS generation (may take 30-60s)",
                    "",
                    "=== TRADING BLOCK CONFIGURATION ===",
                    "--blocks <n>               Number of Trading Blocks to test (default: 10)",
                    "--block-size <bars>        Bars per Trading Block (default: 480 ≈ 8hrs)",
                    "",
                    "=== REGIME & TRACK OPTIONS ===",
                    "--regime <regime>          Market regime: normal|volatile|trending (default: normal)",
                    "                           • For simulation: selects appropriate track",
                    "                           • For ai-regime: configures MarS generation",
                    "--track <n>                Force specific track (1-10, simulation mode only)",
                    "--alpaca-fees              Use Alpaca fee structure (default: true)",
                    "--alpaca-limits            Apply Alpaca position/order limits",
                    "--confidence <level>       Confidence level: 90|95|99 (default: 95)",
                    "--output <format>          Output format: console|json|csv (default: console)",
                    "--save-results <file>      Save detailed results to file",
                    "--benchmark <symbol>       Benchmark symbol (default: SPY)",
                    "--quick                    Quick mode: 10 TB test",
                    "--comprehensive            Comprehensive mode: 60 TB test (3 months)",
                    "--monthly                  Monthly benchmark: 20 TB test (1 month)",
                    "--params <json>            Strategy parameters as JSON string (default: '{}')",
                    "",
                    "Trading Block Info: 1 TB = 480 bars ≈ 8 hours, 10 TB ≈ 2 weeks, 20 TB ≈ 1 month",
                    "Note: Symbol defaults to QQQ if not specified"
                },
                {
                    "# Historical mode (real market data)",
                    "sentio_cli strattest ire --mode historical --blocks 20",
                    "",
                    "# Simulation mode (MarS pre-generated tracks)", 
                    "sentio_cli strattest momentum --mode simulation --regime volatile --track 2",
                    "sentio_cli strattest tfa --mode simulation --regime trending --blocks 30",
                    "",
                    "# AI Regime mode (real-time MarS generation)",
                    "sentio_cli strattest ire --mode ai-regime --regime volatile --blocks 15",
                    "sentio_cli strattest mm --mode ai-regime --regime normal --monthly"
                });
            return 0;
        }
        
        if (!sentio::CLIHelpers::validate_required_args(args, 1, 
            "sentio_cli strattest <strategy> [symbol] [options]")) {
            return 1;
        }
        
        std::string strategy_name = args.positional_args[0];
        std::string symbol = (args.positional_args.size() > 1) ? args.positional_args[1] : "QQQ";
        
        // Validate command options
        std::vector<std::string> allowed_options = {
            "mode", "blocks", "block-size", "regime", "track", "confidence", 
            "output", "save-results", "benchmark", "params"
        };
        std::vector<std::string> allowed_flags = {
            "alpaca-fees", "alpaca-limits", "quick", "comprehensive", "monthly"
        };
        
        if (!sentio::CLIHelpers::validate_options(args, "strattest", allowed_options, allowed_flags)) {
            return 1;
        }
        
        // Validate inputs
        if (!sentio::CLIHelpers::is_valid_strategy_name(strategy_name)) {
            sentio::CLIHelpers::print_error("Invalid strategy name: " + strategy_name);
            return 1;
        }
        
        if (!sentio::CLIHelpers::is_valid_symbol(symbol)) {
            sentio::CLIHelpers::print_error("Invalid symbol: " + symbol);
            return 1;
        }
        
        // Parse mode and Trading Block configuration  
        std::string mode_str = sentio::CLIHelpers::get_option(args, "mode", "simulation");
        
        // Parse Trading Block configuration (canonical evaluation only)
        int num_blocks = 10;  // Default to 10 TB
        int block_size = 480; // Default to 480 bars per TB (8 hours)
        
        // Check for preset modes first
        if (sentio::CLIHelpers::get_flag(args, "quick")) {
            num_blocks = 10;  // Quick: 10 TB ≈ 2 weeks
        } else if (sentio::CLIHelpers::get_flag(args, "comprehensive")) {
            num_blocks = 60;  // Comprehensive: 60 TB ≈ 3 months
        } else if (sentio::CLIHelpers::get_flag(args, "monthly")) {
            num_blocks = 20;  // Monthly: 20 TB ≈ 1 month
        }
        
        // Allow explicit override of defaults
        num_blocks = sentio::CLIHelpers::get_int_option(args, "blocks", num_blocks);
        block_size = sentio::CLIHelpers::get_int_option(args, "block-size", block_size);
        
        // Show Trading Block configuration
        std::cout << "\n📊 **TRADING BLOCK CONFIGURATION**" << std::endl;
        std::cout << "Strategy: " << strategy_name << " (" << symbol << ")" << std::endl;
        std::cout << "Mode: " << mode_str << std::endl;
        std::cout << "Trading Blocks: " << num_blocks << " TB × " << block_size << " bars" << std::endl;
        std::cout << "Total Duration: " << (num_blocks * block_size) << " bars ≈ " 
                  << std::fixed << std::setprecision(1) << ((num_blocks * block_size) / 390.0) << " trading days" << std::endl;
        std::cout << "Equivalent: ~" << std::fixed << std::setprecision(1) << ((num_blocks * block_size) / 60.0 / 8.0) << " trading days (8hrs/day)" << std::endl;
        
        if (num_blocks >= 20) {
            std::cout << "📈 20TB Benchmark: Available (monthly performance measurement)" << std::endl;
        } else {
            std::cout << "ℹ️  For monthly benchmark (20TB), use --monthly or --blocks 20" << std::endl;
        }
        std::cout << std::endl;

        // Check if we should use the new Trading Block canonical evaluation
        bool use_canonical_evaluation = (num_blocks > 0 && block_size > 0);
        
        if (use_canonical_evaluation) {
            std::cout << "\n🎯 **CANONICAL TRADING BLOCK EVALUATION**" << std::endl;
            std::cout << "Using deterministic Trading Block system instead of legacy duration-based testing" << std::endl;
            std::cout << std::endl;
            
            // Use new canonical evaluation system
            try {
                // Create Trading Block configuration
                sentio::TradingBlockConfig block_config;
                block_config.block_size = block_size;
                block_config.num_blocks = num_blocks;
                
                std::cout << "🚀 Loading data for canonical evaluation..." << std::endl;
                
                // Load data using the profit-maximizing pipeline (QQQ family for leverage)
                sentio::SymbolTable ST;
                std::vector<std::vector<sentio::Bar>> series;
                
                // **PROFIT MAXIMIZATION**: Load full QQQ family for maximum leverage
                int base_symbol_id = ST.intern(symbol);
                int tqqq_id = ST.intern("TQQQ");  // 3x leveraged long
                int sqqq_id = ST.intern("SQQQ");  // 3x leveraged short  
                int psq_id = ST.intern("PSQ");    // 1x inverse
                
                series.resize(4);  // QQQ family
                
                // Select data source based on mode
                std::string data_file;
                std::string dataset_type;
                std::vector<sentio::Bar> bars;
                
                if (mode_str == "historical") {
                    // Mode 1: Use real QQQ RTH historical data from equities folder
                    data_file = "data/equities/QQQ_RTH_NH.csv";
                    dataset_type = "real_historical_qqq_rth";
                    std::cout << "📊 Loading " << symbol << " RTH historical data..." << std::endl;
                    std::cout << "📁 Dataset: " << data_file << " (Real Market Data - RTH Only)" << std::endl;
                    
                } else if (mode_str == "simulation") {
                    // Mode 2: Use MarS pre-generated future QQQ simulation data
                    std::string regime = sentio::CLIHelpers::get_option(args, "regime", "normal");
                    int track_num = sentio::CLIHelpers::get_int_option(args, "track", 1);
                    
                    // Select track based on regime preference
                    if (regime == "volatile") {
                        track_num = (track_num <= 3) ? (2 + (track_num-1)*3) : 2; // Tracks 2,5,8
                    } else if (regime == "trending") {
                        track_num = (track_num <= 3) ? (3 + (track_num-1)*3) : 3; // Tracks 3,6,9
                    } else {
                        // Normal regime: tracks 1,4,7,10
                        if (track_num > 4) track_num = 1;
                        int normal_tracks[] = {1, 4, 7, 10};
                        track_num = normal_tracks[(track_num-1) % 4];
                    }
                    
                    char track_file[64];
                    snprintf(track_file, sizeof(track_file), "data/future_qqq/future_qqq_track_%02d.csv", track_num);
                    data_file = track_file;
                    dataset_type = "mars_simulation_" + regime;
                    
                    std::cout << "📊 Loading " << symbol << " MarS simulation data..." << std::endl;
                    std::cout << "🎯 Simulation Regime: " << regime << " (Track " << track_num << ")" << std::endl;
                    std::cout << "📁 Dataset: " << data_file << " (MarS Simulation)" << std::endl;
                    
                } else if (mode_str == "ai-regime") {
                    // Mode 3: Generate real-time AI data using MarS
                    std::string regime = sentio::CLIHelpers::get_option(args, "regime", "normal");
                    int required_bars = (num_blocks * block_size) + 250; // Include warmup bars
                    // MarS generates about 0.67 bars per minute, so multiply by 1.5 to ensure enough data
                    int duration_minutes = static_cast<int>(required_bars * 1.5);
                    
                    data_file = "temp_ai_regime_" + symbol + "_" + std::to_string(std::time(nullptr)) + ".json";
                    dataset_type = "mars_ai_realtime_" + regime;
                    
                    std::cout << "🤖 Generating " << symbol << " AI regime data..." << std::endl;
                    std::cout << "🎯 AI Regime: " << regime << " (" << duration_minutes << " minutes)" << std::endl;
                    std::cout << "⚡ Real-time generation - this may take 30-60 seconds..." << std::endl;
                    
                    // Generate MarS data in real-time
                    sentio::MarsDataLoader mars_loader;
                    bars = mars_loader.load_mars_data(symbol, duration_minutes, 60, 1, regime);
                    
                    if (bars.empty()) {
                        throw std::runtime_error("Failed to generate AI regime data with MarS");
                    }
                    
                    std::cout << "✅ Generated " << bars.size() << " bars with AI regime: " << regime << std::endl;
                    data_file = "AI-Generated (" + regime + " regime)";
                    
                } else {
                    throw std::runtime_error("Invalid mode: " + mode_str + ". Use: historical, simulation, or ai-regime");
                }
                
                // Load data (if not already loaded by AI regime mode)
                if (mode_str != "ai-regime") {
                    bool load_success = sentio::load_csv(data_file, bars);
                    if (!load_success || bars.empty()) {
                        throw std::runtime_error("Failed to load data from " + data_file);
                    }
                }
                
                std::cout << "✅ Loaded " << bars.size() << " bars" << std::endl;
                
                // Prepare series data structure with leverage instruments for profit maximization
                series.resize(ST.size());
                series[base_symbol_id] = bars;
                
                // **PROFIT MAXIMIZATION**: Generate theoretical leverage data for maximum capital deployment
                std::cout << "🚀 Generating theoretical leverage data for maximum profit..." << std::endl;
                series[tqqq_id] = generate_theoretical_leverage_series(bars, 3.0);   // 3x leveraged long
                series[sqqq_id] = generate_theoretical_leverage_series(bars, -3.0);  // 3x leveraged short  
                series[psq_id] = generate_theoretical_leverage_series(bars, -1.0);   // 1x inverse
                
                std::cout << "✅ TQQQ theoretical data generated (" << series[tqqq_id].size() << " bars, 3x leverage)" << std::endl;
                std::cout << "✅ SQQQ theoretical data generated (" << series[sqqq_id].size() << " bars, -3x leverage)" << std::endl;
                std::cout << "✅ PSQ theoretical data generated (" << series[psq_id].size() << " bars, -1x leverage)" << std::endl;
                
                // Create comprehensive dataset metadata with ISO timestamps
                sentio::DatasetMetadata dataset_meta;
                dataset_meta.source_type = dataset_type;
                dataset_meta.file_path = data_file;
                dataset_meta.bars_count = static_cast<int>(bars.size());
                dataset_meta.mode = mode_str;
                dataset_meta.regime = sentio::CLIHelpers::get_option(args, "regime", "normal");
                
                if (!bars.empty()) {
                    dataset_meta.time_range_start = bars.front().ts_utc_epoch * 1000;
                    dataset_meta.time_range_end = bars.back().ts_utc_epoch * 1000;
                    
                    // Convert timestamps to ISO format for display
                    auto start_time = std::time_t(bars.front().ts_utc_epoch);
                    auto end_time = std::time_t(bars.back().ts_utc_epoch);
                    
                    char start_iso[32], end_iso[32];
                    std::strftime(start_iso, sizeof(start_iso), "%Y-%m-%dT%H:%M:%S", std::gmtime(&start_time));
                    std::strftime(end_iso, sizeof(end_iso), "%Y-%m-%dT%H:%M:%S", std::gmtime(&end_time));
                    
                    std::cout << "📅 Dataset Period: " << start_iso << " to " << end_iso << " UTC" << std::endl;
                    
                    // Calculate and display test period (using most recent data)
                    size_t warmup_bars = 250;
                    size_t test_bars = num_blocks * block_size;
                    if (bars.size() > warmup_bars + test_bars) {
                        // Start from the end and work backwards to get the most recent data
                        size_t test_end_idx = bars.size() - 1;
                        size_t test_start_idx = test_end_idx - test_bars + 1;
                        
                        auto test_start_time = std::time_t(bars[test_start_idx].ts_utc_epoch);
                        auto test_end_time = std::time_t(bars[test_end_idx].ts_utc_epoch);
                        
                        char test_start_iso[32], test_end_iso[32];
                        std::strftime(test_start_iso, sizeof(test_start_iso), "%Y-%m-%dT%H:%M:%S", std::gmtime(&test_start_time));
                        std::strftime(test_end_iso, sizeof(test_end_iso), "%Y-%m-%dT%H:%M:%S", std::gmtime(&test_end_time));
                        
                        std::cout << "🎯 Test Period: " << test_start_iso << " to " << test_end_iso << " UTC" << std::endl;
                        std::cout << "⚡ Warmup Bars: " << warmup_bars << " (excluded from test)" << std::endl;
                    }
                }
                dataset_meta.calculate_trading_blocks(block_config.block_size);
                
                // Create audit recorder
                std::cout << "🔍 Initializing audit system..." << std::endl;
                std::string audit_db_path = "audit/sentio_audit.sqlite3";
                std::string run_id = sentio::generate_run_id();
                auto audit_recorder = std::make_unique<audit::AuditDBRecorder>(audit_db_path, run_id, "Trading Block canonical evaluation");
                
                // Create runner configuration
                sentio::RunnerCfg runner_cfg;
                runner_cfg.strategy_name = strategy_name;
                // Pass mode through strategy parameters (safer than adding to struct)
                runner_cfg.strategy_params["mode"] = mode_str;
                
                std::cout << "🎯 Running canonical Trading Block evaluation..." << std::endl;
                
                // Run the canonical evaluation!
                auto canonical_report = sentio::run_canonical_backtest(
                    *audit_recorder, 
                    ST, 
                    series, 
                    base_symbol_id, 
                    runner_cfg, 
                    dataset_meta, 
                    block_config
                );
                
                std::cout << "\n✅ **CANONICAL EVALUATION COMPLETED SUCCESSFULLY!**" << std::endl;
                std::cout << "🎉 Trading Block system is now fully operational!" << std::endl;
                std::cout << "\nUse './saudit summarize' to verify results in audit system" << std::endl;
                
                return 0; // Exit here - canonical evaluation complete
            
        } catch (const std::exception& e) {
                std::cerr << "❌ Canonical evaluation failed: " << e.what() << std::endl;
            return 1;
            }
        }
        
        // All Trading Block evaluations should succeed and return above
        std::cerr << "❌ Unexpected: Trading Block configuration invalid" << std::endl;
        return 1;
        
    } else if (command == "probe") {
        // ANSI color codes
        const std::string RESET = "\033[0m";
        const std::string BOLD = "\033[1m";
        const std::string DIM = "\033[2m";
        const std::string BLUE = "\033[34m";
        const std::string GREEN = "\033[32m";
        const std::string RED = "\033[31m";
        const std::string YELLOW = "\033[33m";
        const std::string CYAN = "\033[36m";
        const std::string MAGENTA = "\033[35m";
        const std::string WHITE = "\033[37m";
        const std::string BG_BLUE = "\033[44m";
        
        // Enhanced header
        std::cout << "\n" << BOLD << BG_BLUE << WHITE << "╔══════════════════════════════════════════════════════════════════════════════════╗" << RESET << std::endl;
        std::cout << BOLD << BG_BLUE << WHITE << "║                           🔍 SENTIO SYSTEM PROBE                                ║" << RESET << std::endl;
        std::cout << BOLD << BG_BLUE << WHITE << "╚══════════════════════════════════════════════════════════════════════════════════╝" << RESET << std::endl;
        
        auto strategies = sentio::StrategyFactory::instance().get_available_strategies();
        
        // Show available strategies
        std::cout << "\n" << BOLD << CYAN << "📊 AVAILABLE STRATEGIES" << RESET << " " << DIM << "(" << strategies.size() << " total)" << RESET << std::endl;
        std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
        
        // Display strategies in a nice grid format
        int count = 0;
        for (const auto& strategy_name : strategies) {
            if (count % 3 == 0) {
                std::cout << "│ ";
            }
            std::cout << MAGENTA << "• " << strategy_name << RESET;
            
            // Pad to make columns align
            int padding = 25 - strategy_name.length();
            for (int i = 0; i < padding; ++i) {
                std::cout << " ";
            }
            
            count++;
            if (count % 3 == 0) {
                std::cout << "│" << std::endl;
            }
        }
        
        // Handle remaining strategies if not divisible by 3
        if (count % 3 != 0) {
            int remaining = 3 - (count % 3);
            for (int i = 0; i < remaining; ++i) {
                for (int j = 0; j < 25; ++j) {
                    std::cout << " ";
                }
            }
            std::cout << "│" << std::endl;
        }
        
        std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
        
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
        std::cout << "\n" << BOLD << CYAN << "📈 DATA AVAILABILITY CHECK" << RESET << std::endl;
        std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
        
        bool daily_aligned = true, minute_aligned = true;
        
        for (const auto& symbol : symbols) {
            std::cout << "│ " << BOLD << "Symbol: " << BLUE << symbol << RESET << std::endl;
            
            // Check daily data
            std::string daily_path = "data/equities/" + symbol + "_daily.csv";
            auto [daily_exists, daily_range] = get_file_info(daily_path);
            
            std::cout << "│   📅 Daily:  ";
            if (daily_exists) {
                std::cout << GREEN << "✅ Available" << RESET << " " << DIM << "(" << daily_range.first << " to " << daily_range.second << ")" << RESET << std::endl;
            } else {
                std::cout << RED << "❌ Missing" << RESET << std::endl;
                daily_aligned = false;
            }
            
            // Check minute data
            std::string minute_path = "data/equities/" + symbol + "_NH.csv";
            auto [minute_exists, minute_range] = get_file_info(minute_path);
            
            std::cout << "│   ⏰ Minute: ";
            if (minute_exists) {
                std::cout << GREEN << "✅ Available" << RESET << " " << DIM << "(" << minute_range.first << " to " << minute_range.second << ")" << RESET << std::endl;
            } else {
                std::cout << RED << "❌ Missing" << RESET << std::endl;
                minute_aligned = false;
            }
            
            std::cout << "│" << std::endl;
        }
        
        std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
        
        // Summary
        std::cout << "\n" << BOLD << CYAN << "📋 SYSTEM STATUS" << RESET << std::endl;
        std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
        
        if (daily_aligned && minute_aligned) {
            std::cout << "│ " << GREEN << "🎉 ALL SYSTEMS READY" << RESET << " - Data is properly aligned for strategy testing" << std::endl;
            std::cout << "│" << std::endl;
            std::cout << "│ " << BOLD << "Quick Start Commands:" << RESET << std::endl;
            std::cout << "│   " << CYAN << "• ./build/sentio_cli strattest ire --mode simulation --blocks 10" << RESET << std::endl;
            std::cout << "│   " << CYAN << "• ./build/sentio_cli strattest tfa --mode historical --blocks 20" << RESET << std::endl;
            std::cout << "│   " << CYAN << "• ./saudit list --limit 10" << RESET << std::endl;
        } else {
            std::cout << "│ " << YELLOW << "⚠️  PARTIAL DATA AVAILABILITY" << RESET << " - Some data files are missing" << std::endl;
            std::cout << "│" << std::endl;
            std::cout << "│ " << BOLD << "Recommended Actions:" << RESET << std::endl;
            if (!daily_aligned) {
                std::cout << "│   " << RED << "• Download missing daily data files" << RESET << std::endl;
            }
            if (!minute_aligned) {
                std::cout << "│   " << RED << "• Download missing minute data files" << RESET << std::endl;
            }
            std::cout << "│   " << CYAN << "• Use: ./build/sentio_cli download <SYMBOL> --period 3y" << RESET << std::endl;
        }
        
        std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
        
        return 0;
        
    } else if (command == "list-strategies") {
        if (args.help_requested) {
            sentio::CLIHelpers::print_help("list-strategies",
                "sentio_cli list-strategies [options]",
                {
                    "--format <format>          Output format: table|list|json (default: table)",
                    "--category <cat>           Filter by category: momentum|mean-reversion|ml|all (default: all)",
                    "--verbose                  Show detailed strategy information"
                },
                {
                    "sentio_cli list-strategies",
                    "sentio_cli list-strategies --format list",
                    "sentio_cli list-strategies --category ml --verbose"
                });
            return 0;
        }
        
        // Validate command options
        std::vector<std::string> allowed_options = {"format", "category"};
        std::vector<std::string> allowed_flags = {"verbose"};
        
        if (!sentio::CLIHelpers::validate_options(args, "list-strategies", allowed_options, allowed_flags)) {
            return 1;
        }
        
        // Get available strategies from factory
        auto& factory = sentio::StrategyFactory::instance();
        auto strategies = factory.get_available_strategies();
        
        if (strategies.empty()) {
            std::cout << "❌ No strategies available" << std::endl;
            return 1;
        }
        
        // Color constants
        const std::string BOLD = "\033[1m";
        const std::string CYAN = "\033[36m";
        const std::string GREEN = "\033[32m";
        const std::string BLUE = "\033[34m";
        const std::string RESET = "\033[0m";
        
        std::string format = sentio::CLIHelpers::get_option(args, "format", "table");
        std::string category = sentio::CLIHelpers::get_option(args, "category", "all");
        bool verbose = sentio::CLIHelpers::get_flag(args, "verbose");
        
        // Strategy categorization (based on common knowledge)
        std::unordered_map<std::string, std::vector<std::string>> strategy_categories = {
            {"momentum", {"ire", "bollinger_squeeze_breakout"}},
            {"mean-reversion", {"rsi"}},
            {"ml", {"tfa", "kochi_ppo"}},
            {"signal", {"signal_or"}}
        };
        
        // Filter strategies by category if specified
        std::vector<std::string> filtered_strategies = strategies;
        if (category != "all") {
            auto it = strategy_categories.find(category);
            if (it != strategy_categories.end()) {
                filtered_strategies.clear();
                for (const auto& strategy : strategies) {
                    if (std::find(it->second.begin(), it->second.end(), strategy) != it->second.end()) {
                        filtered_strategies.push_back(strategy);
                    }
                }
            }
        }
        
        if (format == "json") {
            std::cout << "{\n";
            std::cout << "  \"strategies\": [\n";
            for (size_t i = 0; i < filtered_strategies.size(); ++i) {
                std::cout << "    \"" << filtered_strategies[i] << "\"";
                if (i < filtered_strategies.size() - 1) std::cout << ",";
                std::cout << "\n";
            }
            std::cout << "  ],\n";
            std::cout << "  \"total\": " << filtered_strategies.size() << ",\n";
            std::cout << "  \"category\": \"" << category << "\"\n";
            std::cout << "}\n";
            
        } else if (format == "list") {
            std::cout << "📋 Available Strategies (" << filtered_strategies.size() << "):\n";
            for (const auto& strategy : filtered_strategies) {
                std::cout << "  • " << strategy << "\n";
            }
            
        } else { // table format (default)
            std::cout << "\n" << BOLD << CYAN << "📋 AVAILABLE STRATEGIES" << RESET << std::endl;
            std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
            std::cout << "│ " << BOLD << "Strategy Name" << RESET << "                    │ " << BOLD << "Category" << RESET << "        │ " << BOLD << "Description" << RESET << "                    │" << std::endl;
            std::cout << "├─────────────────────────────────────────────────────────────────────────────────┤" << std::endl;
            
            for (const auto& strategy : filtered_strategies) {
                std::string category_name = "Other";
                std::string description = "Trading strategy";
                
                // Determine category and description
                if (strategy == "ire") {
                    category_name = "Momentum";
                    description = "Intelligent Regime Engine";
                } else if (strategy == "bollinger_squeeze_breakout") {
                    category_name = "Momentum";
                    description = "Bollinger Band breakout";
                } else if (strategy == "rsi") {
                    category_name = "Mean Reversion";
                    description = "RSI-based reversion";
                } else if (strategy == "tfa") {
                    category_name = "Machine Learning";
                    description = "Transformer-based forecasting";
                } else if (strategy == "kochi_ppo") {
                    category_name = "Machine Learning";
                    description = "PPO reinforcement learning";
                } else if (strategy == "signal_or") {
                    category_name = "Signal";
                    description = "Signal combination logic";
                }
                
                printf("│ %-30s │ %-15s │ %-30s │\n", 
                       strategy.c_str(), category_name.c_str(), description.c_str());
            }
            
            std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
            
            if (verbose) {
                std::cout << "\n" << BOLD << CYAN << "📖 STRATEGY DETAILS" << RESET << std::endl;
                std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
                
                for (const auto& strategy : filtered_strategies) {
                    std::cout << "│ " << BOLD << BLUE << strategy << RESET << std::endl;
                    
                    if (strategy == "ire") {
                        std::cout << "│   📊 Intelligent Regime Engine - Advanced momentum strategy" << std::endl;
                        std::cout << "│   🎯 Uses regime detection and adaptive parameters" << std::endl;
                        std::cout << "│   💡 Best for: Trending markets, volatile conditions" << std::endl;
                    } else if (strategy == "bollinger_squeeze_breakout") {
                        std::cout << "│   📊 Bollinger Band Squeeze Breakout strategy" << std::endl;
                        std::cout << "│   🎯 Detects low volatility periods and trades breakouts" << std::endl;
                        std::cout << "│   💡 Best for: Range-bound markets, volatility expansion" << std::endl;
                    } else if (strategy == "rsi") {
                        std::cout << "│   📊 RSI Mean Reversion strategy" << std::endl;
                        std::cout << "│   🎯 Uses RSI overbought/oversold signals" << std::endl;
                        std::cout << "│   💡 Best for: Range-bound markets, contrarian trading" << std::endl;
                    } else if (strategy == "tfa") {
                        std::cout << "│   🤖 Transformer-based Forecasting Algorithm" << std::endl;
                        std::cout << "│   🎯 Uses deep learning for price prediction" << std::endl;
                        std::cout << "│   💡 Best for: Complex patterns, adaptive to market changes" << std::endl;
                    } else if (strategy == "kochi_ppo") {
                        std::cout << "│   🤖 Kochi PPO Reinforcement Learning strategy" << std::endl;
                        std::cout << "│   🎯 Uses Proximal Policy Optimization for trading decisions" << std::endl;
                        std::cout << "│   💡 Best for: Adaptive learning, complex market dynamics" << std::endl;
                    } else if (strategy == "signal_or") {
                        std::cout << "│   🔗 Signal OR combination strategy" << std::endl;
                        std::cout << "│   🎯 Combines multiple signal sources with OR logic" << std::endl;
                        std::cout << "│   💡 Best for: Signal aggregation, multi-strategy approaches" << std::endl;
                    }
                    
                    std::cout << "│" << std::endl;
                }
                
                std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
            }
        }
        
        std::cout << "\n" << BOLD << GREEN << "💡 Usage Examples:" << RESET << std::endl;
        std::cout << "  " << CYAN << "• sentio_cli strattest ire --mode simulation --blocks 10" << RESET << std::endl;
        std::cout << "  " << CYAN << "• sentio_cli strattest tfa --mode historical --blocks 20" << RESET << std::endl;
        std::cout << "  " << CYAN << "• sentio_cli strattest rsi --mode ai-regime --regime volatile" << RESET << std::endl;
        
        return 0;
        
    } else if (command == "audit-validate") {
        std::cout << "🔍 **STRATEGY-AGNOSTIC AUDIT VALIDATION**" << std::endl;
        std::cout << "⚠️  Audit validation feature removed during cleanup" << std::endl;
        return 0;
        
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
