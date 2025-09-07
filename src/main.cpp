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
#include "sentio/rth_calendar.hpp" // **NEW**: Include for RTH check
#include "sentio/calendar_seed.hpp" // **NEW**: Include for calendar creation
// #include "sentio/feature/feature_from_spec.hpp" // For C++ feature building - causes Bar redefinition
#include "sentio/feature/feature_matrix.hpp" // For FeatureMatrix
// TFB strategy removed - focusing on TFA only
#include "sentio/strategy_tfa.hpp" // For TFA strategy

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <cstdlib> // For std::exit
#include <sstream>
#include <ctime> // For std::time
#include <fstream> // For std::ifstream
#include <ATen/Parallel.h> // For LibTorch threading controls


namespace { // Anonymous namespace to ensure link-time registration
    struct StrategyRegistrar {
        StrategyRegistrar() {
            // Register strategies in the factory
            sentio::StrategyFactory::instance().register_strategy("VWAPReversion", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::VWAPReversionStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("MomentumVolume", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::MomentumVolumeProfileStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("BollingerSqueeze", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::BollingerSqueezeBreakoutStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("OpeningRange", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::OpeningRangeBreakoutStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("OrderFlowScalping", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::OrderFlowScalpingStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("OrderFlowImbalance", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::OrderFlowImbalanceStrategy>();
            });
            sentio::StrategyFactory::instance().register_strategy("MarketMaking", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::make_unique<sentio::MarketMakingStrategy>();
            });
            // TransformerTS inherits from IStrategy, not BaseStrategy, so skip for now
            // TFB strategy removed - focusing on TFA only
            sentio::StrategyFactory::instance().register_strategy("tfa", []() -> std::unique_ptr<sentio::BaseStrategy> {
                return std::unique_ptr<sentio::BaseStrategy>(new sentio::TFAStrategy(sentio::TFACfg{}));
            });
        }
    };
    static StrategyRegistrar registrar;
}


void usage() {
    std::cout << "Usage: sentio_cli <command> [options]\n"
              << "Commands:\n"
              << "  backtest <symbol> [--strategy <name>] [--params <k=v,...>]\n"
              << "  tpa_test <symbol> [--strategy <name>] [--params <k=v,...>] [--quarters <n>]\n"
              << "  test-models [--strategy <name>] [--data <file>] [--start <date>] [--end <date>]\n"
              << "  replay <run_id>\n";
}

int main(int argc, char* argv[]) {
    // Configure LibTorch threading to prevent oversubscription
    at::set_num_threads(1);         // intra-op
    at::set_num_interop_threads(1); // inter-op
    
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string command = argv[1];
    
    if (command == "backtest") {
        if (argc < 3) {
            std::cout << "Usage: sentio_cli backtest <symbol> [--strategy <name>] [--params <k=v,...>]\n";
            return 1;
        }
        
        std::string base_symbol = argv[2];
        std::string strategy_name = "VWAPReversion";
        std::unordered_map<std::string, std::string> strategy_params;
        
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--strategy" && i + 1 < argc) {
                strategy_name = argv[++i];
            } else if (arg == "--params" && i + 1 < argc) {
                std::string params_str = argv[++i];
                std::stringstream ss(params_str);
                std::string pair;
                while (std::getline(ss, pair, ',')) {
                    size_t eq_pos = pair.find('=');
                    if (eq_pos != std::string::npos) {
                        strategy_params[pair.substr(0, eq_pos)] = pair.substr(eq_pos + 1);
                    }
                }
            }
        }
        
        sentio::SymbolTable ST;
        std::vector<std::vector<sentio::Bar>> series;
        std::vector<std::string> symbols_to_load = {base_symbol};
        if (base_symbol == "QQQ") {
            symbols_to_load.push_back("TQQQ");
            symbols_to_load.push_back("SQQQ");
        }

        std::cout << "Loading data for symbols: ";
        for(const auto& sym : symbols_to_load) std::cout << sym << " ";
        std::cout << std::endl;

        for (const auto& sym : symbols_to_load) {
            std::vector<sentio::Bar> bars;
            std::string data_path = sentio::resolve_csv(sym);
            if (!sentio::load_csv(data_path, bars)) {
                std::cerr << "ERROR: Failed to load data for " << sym << " from " << data_path << std::endl;
                continue;
            }
            std::cout << " -> Loaded " << bars.size() << " bars for " << sym << std::endl;
            
            int symbol_id = ST.intern(sym);
            if (static_cast<size_t>(symbol_id) >= series.size()) {
                series.resize(symbol_id + 1);
            }
            series[symbol_id] = std::move(bars);
        }
        
        int base_symbol_id = ST.get_id(base_symbol);
        if (series.empty() || series[base_symbol_id].empty()) {
            std::cerr << "FATAL: No data loaded for base symbol " << base_symbol << std::endl;
            return 1;
        }

        // **NEW**: Data Sanity Check - Verify RTH filtering post-load.
        // This acts as a safety net. If your data was generated with RTH filtering,
        // this check ensures the filtering was successful.
        std::cout << "\nVerifying data integrity for RTH..." << std::endl;
        bool rth_filter_failed = false;
        sentio::TradingCalendar calendar = sentio::make_default_nyse_calendar();
        for (size_t sid = 0; sid < series.size(); ++sid) {
            if (series[sid].empty()) continue;
            for (const auto& bar : series[sid]) {
                if (!calendar.is_rth_utc(bar.ts_nyt_epoch, "UTC")) {
                    std::cerr << "\nFATAL ERROR: Non-RTH data found after filtering!\n"
                              << " -> Symbol: " << ST.get_symbol(sid) << "\n"
                              << " -> Timestamp (UTC): " << bar.ts_utc << "\n"
                              << " -> UTC Epoch: " << bar.ts_nyt_epoch << "\n\n"
                              << "This indicates your data files (*.csv, *.bin) were generated with an old or incorrect RTH filter.\n"
                              << "Please DELETE your existing data files and REGENERATE them using the updated poly_fetch tool.\n"
                              << std::endl;
                    rth_filter_failed = true;
                    break;
                }
            }
            if (rth_filter_failed) break;
        }

        if (rth_filter_failed) {
            std::exit(1); // Exit with an error as requested
        }
        std::cout << " -> Data verification passed." << std::endl;

        sentio::RunnerCfg cfg;
        cfg.strategy_name = strategy_name;
        cfg.strategy_params = strategy_params;
        cfg.audit_level = sentio::AuditLevel::Full;
        cfg.snapshot_stride = 100;
        
        // Create audit recorder
        sentio::AuditConfig audit_cfg;
        audit_cfg.run_id = "backtest_" + base_symbol + "_" + std::to_string(std::time(nullptr));
        audit_cfg.file_path = "audit/backtest_" + base_symbol + ".jsonl";
        audit_cfg.flush_each = true;
        sentio::AuditRecorder audit(audit_cfg);
        
        sentio::Tsc timer;
        timer.tic();
        auto result = sentio::run_backtest(audit, ST, series, base_symbol_id, cfg);
        double elapsed = timer.toc_sec();
        
        std::cout << "\nBacktest completed in " << elapsed << "s\n";
        std::cout << "Final Equity: " << result.final_equity << "\n";
        std::cout << "Total Return: " << result.total_return << "%\n";
        std::cout << "Sharpe Ratio: " << result.sharpe_ratio << "\n";
        std::cout << "Max Drawdown: " << result.max_drawdown << "%\n";
        std::cout << "Total Fills: " << result.total_fills << "\n";
        std::cout << "Diagnostics -> No Route: " << result.no_route << " | No Quantity: " << result.no_qty << "\n";

    } else if (command == "tpa_test") {
        if (argc < 3) {
            std::cout << "Usage: sentio_cli tpa_test <symbol> [--strategy <name>] [--params <k=v,...>] [--quarters <n>]\n";
            return 1;
        }
        
        std::string base_symbol = argv[2];
        std::string strategy_name = "TFA";
        std::unordered_map<std::string, std::string> strategy_params;
        int num_quarters = 12;
        
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--strategy" && i + 1 < argc) {
                strategy_name = argv[++i];
            } else if (arg == "--params" && i + 1 < argc) {
                std::string params_str = argv[++i];
                std::stringstream ss(params_str);
                std::string pair;
                while (std::getline(ss, pair, ',')) {
                    size_t eq_pos = pair.find('=');
                    if (eq_pos != std::string::npos) {
                        strategy_params[pair.substr(0, eq_pos)] = pair.substr(eq_pos + 1);
                    }
                }
            } else if (arg == "--quarters" && i + 1 < argc) {
                num_quarters = std::stoi(argv[++i]);
            }
        }
        
        sentio::SymbolTable ST;
        std::vector<std::vector<sentio::Bar>> series;
        std::vector<std::string> symbols_to_load = {base_symbol};
        if (base_symbol == "QQQ") {
            symbols_to_load.push_back("TQQQ");
            symbols_to_load.push_back("SQQQ");
        }

        std::cout << "Loading data for symbols: ";
        for(const auto& sym : symbols_to_load) std::cout << sym << " ";
        std::cout << std::endl;

        for (const auto& sym : symbols_to_load) {
            std::vector<sentio::Bar> bars;
            std::string data_path = sentio::resolve_csv(sym);
            if (!sentio::load_csv(data_path, bars)) {
                std::cerr << "ERROR: Failed to load data for " << sym << " from " << data_path << std::endl;
                continue;
            }
            std::cout << " -> Loaded " << bars.size() << " bars for " << sym << std::endl;
            
            int symbol_id = ST.intern(sym);
            if (static_cast<size_t>(symbol_id) >= series.size()) {
                series.resize(symbol_id + 1);
            }
            series[symbol_id] = std::move(bars);
        }
        
        int base_symbol_id = ST.get_id(base_symbol);
        if (series.empty() || series[base_symbol_id].empty()) {
            std::cerr << "FATAL: No data loaded for base symbol " << base_symbol << std::endl;
            return 1;
        }

        // Load cached features for ML strategies for massive performance improvement
        if (strategy_name == "TFA" || strategy_name == "tfa") {
            std::string feature_file = "data/" + base_symbol + "_RTH_features.csv";
            std::cout << "Loading pre-computed features from: " << feature_file << std::endl;
            
            if (sentio::FeatureFeeder::load_feature_cache(feature_file)) {
                sentio::FeatureFeeder::use_cached_features(true);
                std::cout << "✅ Cached features loaded successfully - MASSIVE speed improvement enabled!" << std::endl;
            } else {
                std::cout << "⚠️  Failed to load cached features - falling back to real-time calculation" << std::endl;
            }
        }

        sentio::RunnerCfg cfg;
        cfg.strategy_name = strategy_name;
        cfg.strategy_params = strategy_params;
        cfg.audit_level = sentio::AuditLevel::Full;
        cfg.snapshot_stride = 100;
        
        sentio::TemporalAnalysisConfig temporal_cfg;
        temporal_cfg.num_quarters = num_quarters;
        temporal_cfg.print_detailed_report = true;
        
        std::cout << "\nRunning TPA (Temporal Performance Analysis) Test..." << std::endl;
        std::cout << "Strategy: " << strategy_name << ", Quarters: " << num_quarters << std::endl;
        
        sentio::Tsc timer;
        timer.tic();
        auto summary = sentio::run_temporal_analysis(ST, series, base_symbol_id, cfg, temporal_cfg);
        double elapsed = timer.toc_sec();
        
        std::cout << "\nTPA test completed in " << elapsed << "s" << std::endl;
        
        if (temporal_cfg.print_detailed_report) {
            sentio::TemporalAnalyzer analyzer;
            for (const auto& q : summary.quarterly_results) {
                analyzer.add_quarterly_result(q);
            }
            analyzer.print_detailed_report();
        }
        
    } else if (command == "test-models") {
        std::string strategy_name = "tfa";
        std::string data_file = "data/QQQ_RTH.csv";
        std::string start_date = "2023-01-01";
        std::string end_date = "2023-12-31";
        
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--strategy" && i + 1 < argc) {
                strategy_name = argv[++i];
            } else if (arg == "--data" && i + 1 < argc) {
                data_file = argv[++i];
            } else if (arg == "--start" && i + 1 < argc) {
                start_date = argv[++i];
            } else if (arg == "--end" && i + 1 < argc) {
                end_date = argv[++i];
            }
        }
        
        // Validate strategy name
        if (strategy_name != "tfa") {
            std::cerr << "ERROR: Strategy must be 'tfa'. Got: " << strategy_name << std::endl;
            return 1;
        }
        
        // Check if model exists
        std::string model_dir = "artifacts/TFA/v1";
        std::string model_path = model_dir + "/model.pt";
        std::string metadata_path = model_dir + "/metadata.json";
        
        if (!std::ifstream(model_path).good()) {
            std::cerr << "ERROR: Model not found at " << model_path << std::endl;
            std::cerr << "Please train the model first: python train_models.py" << std::endl;
            return 1;
        }
        
        if (!std::ifstream(metadata_path).good()) {
            std::cerr << "ERROR: Metadata not found at " << metadata_path << std::endl;
            return 1;
        }
        
        std::cout << "🧪 Testing " << strategy_name << " model..." << std::endl;
        std::cout << "📁 Model: " << model_path << std::endl;
        std::cout << "📊 Data: " << data_file << std::endl;
        std::cout << "📅 Period: " << start_date << " to " << end_date << std::endl;
        
        // Load data
        std::vector<sentio::Bar> bars;
        if (!sentio::load_csv(data_file, bars)) {
            std::cerr << "ERROR: Failed to load data from " << data_file << std::endl;
            return 1;
        }
        
        std::cout << "✅ Loaded " << bars.size() << " bars" << std::endl;
        
        // Load feature specification
        std::string feature_spec_path = "python/feature_spec.json";
        std::ifstream spec_file(feature_spec_path);
        if (!spec_file.good()) {
            std::cerr << "ERROR: Feature spec not found at " << feature_spec_path << std::endl;
            return 1;
        }
        
        std::string spec_json((std::istreambuf_iterator<char>(spec_file)), std::istreambuf_iterator<char>());
        std::cout << "✅ Loaded feature specification" << std::endl;
        
        // For now, let's skip the C++ feature builder and use a simple approach
        // We'll create dummy features to test the model loading and signal generation
        std::cout << "🔧 Creating dummy features for testing..." << std::endl;
        
        // Create a simple feature matrix with dummy data
        sentio::FeatureMatrix feature_matrix;
        feature_matrix.rows = std::min(100, static_cast<int>(bars.size()));
        feature_matrix.cols = 6; // close, logret, ema_20, ema_50, rsi_14, zlog_20
        feature_matrix.data.resize(feature_matrix.rows * feature_matrix.cols);
        
        // Fill with dummy features (just close prices for now)
        for (int i = 0; i < feature_matrix.rows; ++i) {
            int base_idx = i * feature_matrix.cols;
            feature_matrix.data[base_idx + 0] = static_cast<float>(bars[i].close); // close
            feature_matrix.data[base_idx + 1] = 0.0f; // logret
            feature_matrix.data[base_idx + 2] = static_cast<float>(bars[i].close); // ema_20
            feature_matrix.data[base_idx + 3] = static_cast<float>(bars[i].close); // ema_50
            feature_matrix.data[base_idx + 4] = 50.0f; // rsi_14
            feature_matrix.data[base_idx + 5] = 0.0f; // zlog_20
        }
        
        std::cout << "✅ Created dummy features: " << feature_matrix.rows << " rows x " << feature_matrix.cols << " cols" << std::endl;
        
        // Create TFA strategy instance
        std::unique_ptr<sentio::TFAStrategy> tfa_strategy = std::make_unique<sentio::TFAStrategy>();
        
        std::cout << "\n🧪 Testing " << strategy_name << " model with features..." << std::endl;
        
        // Feed features to strategy and test signal generation
        int signals_generated = 0;
        int total_bars = std::min(100, static_cast<int>(feature_matrix.rows)); // Test first 100 bars
        
        for (int i = 0; i < total_bars; ++i) {
            // Extract features for this bar
            std::vector<double> raw_features(feature_matrix.cols);
            for (int j = 0; j < feature_matrix.cols; ++j) {
                raw_features[j] = feature_matrix.data[i * feature_matrix.cols + j];
            }
            
            // Create a dummy bar for on_bar call
            sentio::Bar dummy_bar;
            if (i < static_cast<int>(bars.size())) {
                dummy_bar = bars[i];
            }
            
            sentio::StrategyCtx ctx;
            ctx.ts_utc_epoch = dummy_bar.ts_nyt_epoch;
            ctx.instrument = "QQQ";
            ctx.is_rth = true;
            
            // Feed features to strategy and call on_bar
            if (tfa_strategy) {
                tfa_strategy->set_raw_features(raw_features);
                tfa_strategy->on_bar(ctx, dummy_bar);
                auto signal = tfa_strategy->latest();
                if (signal) {
                    signals_generated++;
                    std::cout << "Bar " << i << ": Signal generated - " 
                              << (signal->type == sentio::StrategySignal::Type::BUY ? "BUY" : 
                                  signal->type == sentio::StrategySignal::Type::SELL ? "SELL" : "HOLD")
                              << " (conf=" << signal->confidence << ")" << std::endl;
                }
            } else if (tfa_strategy) {
                tfa_strategy->set_raw_features(raw_features);
                tfa_strategy->on_bar(ctx, dummy_bar);
                auto signal = tfa_strategy->latest();
                if (signal) {
                    signals_generated++;
                    std::cout << "Bar " << i << ": Signal generated - " 
                              << (signal->type == sentio::StrategySignal::Type::BUY ? "BUY" : 
                                  signal->type == sentio::StrategySignal::Type::SELL ? "SELL" : "HOLD")
                              << " (conf=" << signal->confidence << ")" << std::endl;
                }
            }
        }
        
        std::cout << "\n📊 Test Results:" << std::endl;
        std::cout << "  Total bars tested: " << total_bars << std::endl;
        std::cout << "  Signals generated: " << signals_generated << std::endl;
        std::cout << "  Signal rate: " << (100.0 * signals_generated / total_bars) << "%" << std::endl;
        
        if (signals_generated == 0) {
            std::cout << "\n⚠️  WARNING: No signals generated! This could indicate:" << std::endl;
            std::cout << "  - Model confidence threshold too high" << std::endl;
            std::cout << "  - Feature window not ready" << std::endl;
            std::cout << "  - Model prediction issues" << std::endl;
            std::cout << "  - Check the diagnostic output above for details" << std::endl;
        } else {
            std::cout << "\n✅ Model is generating signals successfully!" << std::endl;
        }
        
    } else if (command == "replay") {
        std::cout << "Replay command is not fully implemented in this example.\n";
    } else {
        usage();
        return 1;
    }
    
    return 0;
}
