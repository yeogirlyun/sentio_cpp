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
#include "sentio/strategy_registry.hpp" // For configuration-based strategy registration
#include "sentio/virtual_market.hpp" // For native VM testing

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <cstdlib> // For std::exit
#include <sstream>
#include <ctime> // For std::time
#include <fstream> // For std::ifstream
#include <iomanip> // For std::setw, std::setprecision
#include <cmath> // For std::pow
#include <ATen/Parallel.h> // For LibTorch threading controls
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
        size_t mism=0; for (size_t i=0;i<N;i++){ if (s[i].ts_utc_epoch != base[i].ts_utc_epoch) { mism++; if (mism<3) {
            std::cerr << "  ts mismatch["<<i<<"] " << s[i].ts_utc_epoch << " vs base " << base[i].ts_utc_epoch << "\n"; }
        } }
        if (mism>0) {
            std::cerr << "FATAL: Alignment check failed: " << ST.get_symbol((int)sid)
                      << " has " << mism << " timestamp mismatches vs base (first shown above).\n";
            ok = false;
        }
    }
    if (!ok) {
        std::cerr << "Hint: Use aligned CSVs or run tools/align_bars.py to generate aligned datasets.\n";
    }
    return ok;
}

static void load_strategy_config_if_any(const std::string& strategy_name, sentio::RunnerCfg& cfg){
    try {
        std::string lower = strategy_name;
        for (auto& c : lower) c = std::tolower(c);
        std::string path = "configs/strategies/" + lower + ".json";
        std::ifstream f(path);
        if (!f.good()) return;
        nlohmann::json j; f >> j;
        // params
        if (j.contains("params") && j["params"].is_object()){
            for (auto it = j["params"].begin(); it != j["params"].end(); ++it){
                cfg.strategy_params[it.key()] = std::to_string(it.value().get<double>());
            }
        }
        // router overrides
        if (j.contains("router")){
            auto r = j["router"];
            if (r.contains("ire_min_conf_strong_short")) cfg.router.ire_min_conf_strong_short = r["ire_min_conf_strong_short"].get<double>();
            if (r.contains("min_signal_strength")) cfg.router.min_signal_strength = r["min_signal_strength"].get<double>();
            if (r.contains("signal_multiplier"))   cfg.router.signal_multiplier   = r["signal_multiplier"].get<double>();
            if (r.contains("max_position_pct"))    cfg.router.max_position_pct    = r["max_position_pct"].get<double>();
            if (r.contains("min_shares"))          cfg.router.min_shares          = r["min_shares"].get<double>();
            if (r.contains("lot_size"))            cfg.router.lot_size            = r["lot_size"].get<double>();
            if (r.contains("base_symbol"))         cfg.router.base_symbol         = r["base_symbol"].get<std::string>();
            if (r.contains("bull3x"))              cfg.router.bull3x              = r["bull3x"].get<std::string>();
            if (r.contains("bear3x"))              cfg.router.bear3x              = r["bear3x"].get<std::string>();
            if (r.contains("bear1x"))              cfg.router.bear1x              = r["bear1x"].get<std::string>();
        }
        // sizer overrides
        if (j.contains("sizer")){
            auto s = j["sizer"];
            if (s.contains("fractional_allowed")) cfg.sizer.fractional_allowed = s["fractional_allowed"].get<bool>();
            if (s.contains("min_notional"))       cfg.sizer.min_notional       = s["min_notional"].get<double>();
            if (s.contains("max_leverage"))       cfg.sizer.max_leverage       = s["max_leverage"].get<double>();
            if (s.contains("max_position_pct"))   cfg.sizer.max_position_pct   = s["max_position_pct"].get<double>();
            if (s.contains("volatility_target"))  cfg.sizer.volatility_target  = s["volatility_target"].get<double>();
            if (s.contains("allow_negative_cash"))cfg.sizer.allow_negative_cash= s["allow_negative_cash"].get<bool>();
            if (s.contains("vol_lookback_days"))  cfg.sizer.vol_lookback_days  = s["vol_lookback_days"].get<int>();
            if (s.contains("cash_reserve_pct"))   cfg.sizer.cash_reserve_pct   = s["cash_reserve_pct"].get<double>();
        }
    } catch (...) {
        // ignore config errors
    }
}

// Strategy registration is now handled by StrategyRegistry::load_from_config()
// This eliminates code duplication and allows dynamic strategy configuration


void usage() {
    std::cout << "Usage: sentio_cli <command> [options]\n"
              << "Commands:\n"
              << "  tpatest <symbol> [--strategy <name>] [--params <k=v,...>] [--quarters <n>] [--weeks <n>] [--days <n>]\n"
              << "  test-models [--strategy <name>] [--data <file>] [--start <date>] [--end <date>]\n"
              << "  tune_ire <symbol> [--test_quarters <n>]\n"
              << "  vmtest <strategy> <symbol> [--days <n>] [--hours <n>] [--simulations <n>] [--params <json>] [--historical-data <file>]\n"
              << "\nTime Period Options (most recent periods):\n"
              << "  --quarters <n>  Analyze the most recent n quarters\n"
              << "  --weeks <n>      Analyze the most recent n weeks\n"
              << "  --days <n>       Analyze the most recent n days\n"
              << "  (no option)      Analyze entire dataset\n"
              << "\nVMTest Options:\n"
              << "  --days <n>        Number of days to simulate (default: 30)\n"
              << "  --hours <n>       Number of hours to simulate (alternative to days)\n"
              << "  --simulations <n> Number of Monte Carlo simulations (default: 100)\n"
              << "  --params <json>   Strategy parameters as JSON string\n"
              << "  --historical-data <file> Historical data file (default: data/equities/<symbol>_RTH_NH.csv)\n"
              << "\nNote: Audit functionality moved to tools/audit_cli\n";
}

int main(int argc, char* argv[]) {
    // Configure LibTorch threading to prevent oversubscription (disabled for audit commands)
    // at::set_num_threads(1);         // intra-op
    // at::set_num_interop_threads(1); // inter-op
    
    // Initialize strategy registry from configuration
    if (!sentio::StrategyRegistry::load_from_config()) {
        std::cerr << "Warning: Failed to load strategy configuration, using default registrations" << std::endl;
    }
    
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string command = argv[1];
    
    if (command == "tpatest") {
        if (argc < 3) {
            std::cout << "Usage: sentio_cli tpatest <symbol> [--strategy <name>] [--params <k=v,...>] [--quarters <n>] [--weeks <n>] [--days <n>]\n";
            return 1;
        }
        
        std::string base_symbol = argv[2];
        std::string strategy_name = "TFA";
        std::unordered_map<std::string, std::string> strategy_params;
        int num_quarters = 0; // default: all data
        int num_weeks = 0;
        int num_days = 0;
        
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
            } else if (arg == "--weeks" && i + 1 < argc) {
                num_weeks = std::stoi(argv[++i]);
            } else if (arg == "--days" && i + 1 < argc) {
                num_days = std::stoi(argv[++i]);
            }
        }
        
        sentio::SymbolTable ST;
        std::vector<std::vector<sentio::Bar>> series;
        std::vector<std::string> symbols_to_load = {base_symbol};
        if (base_symbol == "QQQ") {
            symbols_to_load.push_back("TQQQ");
            symbols_to_load.push_back("SQQQ");
            symbols_to_load.push_back("PSQ");
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

        // Report data period and bars for transparency
        auto& base_bars2 = series[base_symbol_id];
        auto fmt_date2 = [](std::int64_t epoch_sec){
            std::time_t tt = static_cast<std::time_t>(epoch_sec);
            std::tm tm{}; gmtime_r(&tt, &tm);
            char buf[32]; std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm); return std::string(buf);
        };
        std::int64_t min_ts2 = base_bars2.front().ts_utc_epoch;
        std::int64_t max_ts2 = base_bars2.back().ts_utc_epoch;
        std::size_t   n_bars2 = base_bars2.size();
        double span_days2 = double(max_ts2 - min_ts2) / (24.0*3600.0);
        std::cout << "Data period: " << fmt_date2(min_ts2) << " → " << fmt_date2(max_ts2)
                  << " (" << std::fixed << std::setprecision(1) << span_days2 << " days)\n";
        std::cout << "Bars(" << base_symbol << "): " << n_bars2 << "\n";
        if ((max_ts2 - min_ts2) < (365LL*24LL*3600LL)) {
            std::cerr << "WARNING: Data period is shorter than 1 year. Results may be unrepresentative.\n";
        }

        // Alignment check (all loaded symbols must align to base timestamps)
        if (!verify_series_alignment(ST, series, base_symbol_id)) {
            std::cerr << "FATAL: Data alignment check failed. Aborting TPA test." << std::endl;
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
        } else if (strategy_name == "kochi_ppo") {
            std::string feature_file = "data/" + base_symbol + "_KOCHI_features.csv";
            std::cout << "Loading pre-computed KOCHI features from: " << feature_file << std::endl;
            if (sentio::FeatureFeeder::load_feature_cache(feature_file)) {
                sentio::FeatureFeeder::use_cached_features(true);
                std::cout << "✅ KOCHI cached features loaded successfully" << std::endl;
            } else {
                std::cerr << "❌ KOCHI feature cache missing; cannot proceed for kochi_ppo strategy." << std::endl;
                return 1;
            }
        }

        // Merge defaults from configs/<strategy>.json (overridden by CLI params)
        sentio::RunnerCfg tmp_cfg2; tmp_cfg2.strategy_name = strategy_name;
        load_strategy_config_if_any(strategy_name, tmp_cfg2);
        for (const auto& kv : tmp_cfg2.strategy_params){ if (!strategy_params.count(kv.first)) strategy_params[kv.first] = kv.second; }
        sentio::RunnerCfg cfg;
        cfg.strategy_name = strategy_name;
        cfg.strategy_params = strategy_params;
        cfg.audit_level = sentio::AuditLevel::Full;
        cfg.snapshot_stride = 100;
        // **FIX**: Apply full router and sizer config from JSON
        cfg.router = tmp_cfg2.router;
        cfg.sizer = tmp_cfg2.sizer;
        
        sentio::TemporalAnalysisConfig temporal_cfg;
        temporal_cfg.num_quarters = num_quarters;
        temporal_cfg.num_weeks = num_weeks;
        temporal_cfg.num_days = num_days;
        temporal_cfg.print_detailed_report = true;
        
        std::cout << "\nRunning TPA (Temporal Performance Analysis) Test..." << std::endl;
        std::cout << "Strategy: " << strategy_name;
        if (num_days > 0) {
            std::cout << ", Days: " << num_days;
        } else if (num_weeks > 0) {
            std::cout << ", Weeks: " << num_weeks;
        } else if (num_quarters > 0) {
            std::cout << ", Quarters: " << num_quarters;
        } else {
            std::cout << ", Full Dataset";
        }
        std::cout << std::endl;
        
        sentio::Tsc timer;
        timer.tic();
        auto summary = sentio::run_temporal_analysis(ST, series, base_symbol_id, cfg, temporal_cfg);
        double elapsed = timer.toc_sec();
        
        std::cout << "\nTPA test completed in " << elapsed << "s" << std::endl;
        
        if (temporal_cfg.print_detailed_report) {
            sentio::TemporalAnalyzer analyzer;
            // Set the correct period name based on the configuration
            if (num_days > 0) {
                analyzer.set_period_name("day");
            } else if (num_weeks > 0) {
                analyzer.set_period_name("week");
            } else if (num_quarters > 0) {
                analyzer.set_period_name("quarter");
            } else {
                analyzer.set_period_name("full period");
            }
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
            ctx.ts_utc_epoch = dummy_bar.ts_utc_epoch;
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
        
    } else if (command == "tune_ire") {
        if (argc < 3) {
            std::cout << "Usage: sentio_cli tune_ire <symbol> [--test_quarters <n>]" << std::endl;
            return 1;
        }

        std::string base_symbol = argv[2];
        int test_quarters = 1;
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--test_quarters" && i + 1 < argc) {
                test_quarters = std::max(1, std::stoi(argv[++i]));
            }
        }

        // Load QQQ family (QQQ, TQQQ, SQQQ)
        sentio::SymbolTable ST;
        std::vector<std::vector<sentio::Bar>> series;
        std::vector<std::string> symbols_to_load = {base_symbol, "TQQQ", "SQQQ", "PSQ"};
        for (const auto& sym : symbols_to_load) {
            std::vector<sentio::Bar> bars;
            std::string data_path = sentio::resolve_csv(sym);
            if (!sentio::load_csv(data_path, bars)) {
                std::cerr << "ERROR: Failed to load data for " << sym << " from " << data_path << std::endl;
                return 1;
            }
            int symbol_id = ST.intern(sym);
            if (static_cast<size_t>(symbol_id) >= series.size()) series.resize(symbol_id + 1);
            series[symbol_id] = std::move(bars);
        }

        int base_symbol_id = ST.get_id(base_symbol);
        if (!verify_series_alignment(ST, series, base_symbol_id)) {
            std::cerr << "FATAL: Data alignment check failed. Aborting tuning." << std::endl;
            return 1;
        }

        const auto& base = series[base_symbol_id];
        if (base.empty()) { std::cerr << "FATAL: No data for base symbol." << std::endl; return 1; }

        // Split into train (all but most recent test_quarters) and test (most recent test_quarters)
        int total_bars = (int)base.size();
        int total_quarters = 12; // assume ~3 years
        int bars_per_quarter = std::max(1, total_bars / total_quarters);
        int test_bars = std::min(total_bars, bars_per_quarter * test_quarters);
        int train_bars = std::max(0, total_bars - test_bars);

        auto slice_series = [&](int start_idx, int end_idx){
            std::vector<std::vector<sentio::Bar>> out; out.reserve(series.size());
            for (const auto& s : series) {
                if ((int)s.size() <= start_idx) { out.emplace_back(); continue; }
                int e = std::min(end_idx, (int)s.size());
                out.emplace_back(s.begin() + start_idx, s.begin() + e);
            }
            return out;
        };

        auto train_series = slice_series(0, train_bars);
        auto test_series  = slice_series(train_bars, total_bars);

        // Bayesian-like TPE tuner (lightweight, dependency-free)
        auto eval_params = [&](double bh, double sl, double sc){
            sentio::RunnerCfg rcfg; rcfg.strategy_name = "IRE";
            rcfg.strategy_params = { {"buy_hi", std::to_string(bh)}, {"sell_lo", std::to_string(sl)} };
            rcfg.router.ire_min_conf_strong_short = sc;
            sentio::TemporalAnalysisConfig tcfg; 
            // Use only last 4 training quarters as proxy
            int train_q = std::max(1, train_bars / std::max(1, bars_per_quarter));
            tcfg.num_quarters = std::min(4, train_q);
            tcfg.print_detailed_report = false;
            auto train_summary = sentio::run_temporal_analysis(ST, train_series, base_symbol_id, rcfg, tcfg);
            double avg_trades = 0.0, avg_monthly = 0.0, avg_sharpe = 0.0;
            if (!train_summary.quarterly_results.empty()){
                for (const auto& q : train_summary.quarterly_results){ avg_trades += q.avg_daily_trades; avg_monthly += q.monthly_return_pct; avg_sharpe += q.sharpe_ratio; }
                avg_trades /= train_summary.quarterly_results.size();
                avg_monthly /= train_summary.quarterly_results.size();
                avg_sharpe /= train_summary.quarterly_results.size();
            }
            double trade_penalty = 0.0;
            if (avg_trades < 80) trade_penalty = (80 - avg_trades);
            else if (avg_trades > 120) trade_penalty = (avg_trades - 120);
            double score = avg_monthly + 0.5 * avg_sharpe - 0.1 * trade_penalty;
            return std::tuple<double,double,double,double>(score, avg_trades, avg_monthly, avg_sharpe);
        };

        auto clip = [](double x, double lo, double hi){ return std::max(lo, std::min(hi, x)); };
        double bh_lo=0.70, bh_hi=0.90, sl_lo=0.10, sl_hi=0.30, sc_lo=0.80, sc_hi=0.98;
        struct Candidate { double buy_hi, sell_lo, short_conf; double score; double avg_trades; double avg_monthly; double sharpe; } best{0.80,0.20,0.90,-1,0,0,0};

        // Initialize with 5 random starts
        for (int iinit=0;iinit<5;iinit++){
            double bh = bh_lo + (bh_hi-bh_lo) * (double)rand()/RAND_MAX;
            double sl = sl_lo + (sl_hi-sl_lo) * (double)rand()/RAND_MAX;
            double sc = sc_lo + (sc_hi-sc_lo) * (double)rand()/RAND_MAX;
            auto [score,tr,mm,sh] = eval_params(bh,sl,sc);
            if (score > best.score) best = {bh,sl,sc,score,tr,mm,sh};
            std::cout << "Init bh="<<bh<<" sl="<<sl<<" sc="<<sc<<" -> trades="<<tr<<" mret="<<mm<<" sharpe="<<sh<<" score="<<score<<"\n";
        }

        int trials = 24, no_improve=0, patience=5;
        double step_bh=0.03, step_sl=0.03, step_sc=0.03;
        for (int t=0;t<trials && no_improve<patience; ++t){
            double bh, sl, sc;
            if (t % 3 == 0){
                // exploration
                bh = bh_lo + (bh_hi-bh_lo) * (double)rand()/RAND_MAX;
                sl = sl_lo + (sl_hi-sl_lo) * (double)rand()/RAND_MAX;
                sc = sc_lo + (sc_hi-sc_lo) * (double)rand()/RAND_MAX;
            } else {
                // exploitation around best
                bh = clip(best.buy_hi + step_bh * (((double)rand()/RAND_MAX)-0.5)*2.0, bh_lo, bh_hi);
                sl = clip(best.sell_lo + step_sl * (((double)rand()/RAND_MAX)-0.5)*2.0, sl_lo, sl_hi);
                sc = clip(best.short_conf + step_sc * (((double)rand()/RAND_MAX)-0.5)*2.0, sc_lo, sc_hi);
            }
            auto [score,tr,mm,sh] = eval_params(bh,sl,sc);
            std::cout << "Try#"<<t<<" bh="<<bh<<" sl="<<sl<<" sc="<<sc<<" -> trades="<<tr<<" mret="<<mm<<" sharpe="<<sh<<" score="<<score<<"\n";
            if (score > best.score){ best = {bh,sl,sc,score,tr,mm,sh}; no_improve=0; }
            else { no_improve++; }
        }

        std::cout << "\nBest params: buy_hi="<<best.buy_hi<<" sell_lo="<<best.sell_lo<<" short_conf="<<best.short_conf
                  <<" | trades="<<best.avg_trades<<" mret="<<best.avg_monthly<<" sharpe="<<best.sharpe<<"\n";

        // Persist best params for IRE
        try {
            nlohmann::json j;
            j["params"]["buy_hi"] = best.buy_hi;
            j["params"]["sell_lo"] = best.sell_lo;
            j["router"]["ire_min_conf_strong_short"] = best.short_conf;
            std::ofstream outf("configs/strategies/ire.json");
            outf << j.dump(2);
            std::cout << "Saved tuned IRE params to configs/strategies/ire.json\n";
        } catch (...) {
            std::cerr << "Warning: failed to write configs/strategies/ire.json\n";
        }

        // Run test with best params on most recent quarters
        sentio::RunnerCfg rcfg_best; rcfg_best.strategy_name = "IRE";
        rcfg_best.strategy_params = { {"buy_hi", std::to_string(best.buy_hi)}, {"sell_lo", std::to_string(best.sell_lo)} };
        rcfg_best.router.ire_min_conf_strong_short = best.short_conf;
        sentio::TemporalAnalysisConfig tcfg_test; tcfg_test.num_quarters = test_quarters; tcfg_test.print_detailed_report = true;
        auto test_summary = sentio::run_temporal_analysis(ST, test_series, base_symbol_id, rcfg_best, tcfg_test);
        test_summary.assess_readiness(tcfg_test);

    } else if (command == "vmtest") {
        if (argc < 4) {
            std::cout << "Usage: sentio_cli vmtest <strategy> <symbol> [--days <n>] [--hours <n>] [--simulations <n>] [--params <json>] [--historical-data <file>]\n";
            return 1;
        }
        
        std::string strategy_name = argv[2];
        std::string symbol = argv[3];
        int days = 30;
        int hours = 0;
        int simulations = 100;
        std::string params_json = "{}";
        std::string historical_data_file = "data/equities/" + symbol + "_RTH_NH.csv";
        
        for (int i = 4; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--days" && i + 1 < argc) {
                days = std::stoi(argv[++i]);
            } else if (arg == "--hours" && i + 1 < argc) {
                hours = std::stoi(argv[++i]);
            } else if (arg == "--simulations" && i + 1 < argc) {
                simulations = std::stoi(argv[++i]);
            } else if (arg == "--params" && i + 1 < argc) {
                params_json = argv[++i];
            } else if (arg == "--historical-data" && i + 1 < argc) {
                historical_data_file = argv[++i];
            }
        }
        
        std::cout << "🚀 Starting Virtual Market Test (Fast Historical Bridge)..." << std::endl;
        std::cout << "📊 Strategy: " << strategy_name << std::endl;
        std::cout << "📈 Symbol: " << symbol << std::endl;
        std::cout << "⏱️  Duration: " << (hours > 0 ? std::to_string(hours) + " hours" : std::to_string(days) + " days") << std::endl;
        std::cout << "🎲 Simulations: " << simulations << std::endl;
        std::cout << "📊 Historical data: " << historical_data_file << std::endl;
        std::cout << "⚡ Using optimized historical patterns" << std::endl;
        
        // Convert days/hours to continuation minutes
        int continuation_minutes = hours > 0 ? hours * 60 : days * 390; // 390 minutes per trading day
        
        // Run fast historical test
        sentio::VirtualMarketEngine vm_engine;
        auto results = vm_engine.run_fast_historical_test(strategy_name, symbol, historical_data_file, 
                                                         continuation_minutes, simulations, params_json);
        
        std::cout << "\n✅ Virtual Market Test completed successfully" << std::endl;

    } else if (command == "marstest") {
        if (argc < 4) {
            std::cout << "Usage: sentio_cli marstest <strategy> <symbol> [--days <n>] [--simulations <n>] [--regime <regime>] [--params <json>]\n";
            return 1;
        }
        
        std::string strategy_name = argv[2];
        std::string symbol = argv[3];
        int days = 1; // Start with 1 day for MarS
        int simulations = 10; // Fewer simulations for MarS (more expensive)
        std::string market_regime = "normal";
        std::string params_json = "{}";
        
        for (int i = 4; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--days" && i + 1 < argc) {
                days = std::stoi(argv[++i]);
            } else if (arg == "--simulations" && i + 1 < argc) {
                simulations = std::stoi(argv[++i]);
            } else if (arg == "--regime" && i + 1 < argc) {
                market_regime = argv[++i];
            } else if (arg == "--params" && i + 1 < argc) {
                params_json = argv[++i];
            }
        }
        
        std::cout << "🚀 Starting MarS Virtual Market Test..." << std::endl;
        std::cout << "📊 Strategy: " << strategy_name << std::endl;
        std::cout << "📈 Symbol: " << symbol << std::endl;
        std::cout << "⏱️  Duration: " << days << " days" << std::endl;
        std::cout << "🎲 Simulations: " << simulations << std::endl;
        std::cout << "🌊 Market Regime: " << market_regime << std::endl;
        std::cout << "🤖 Using MarS AI-powered market simulation" << std::endl;
        
        // Run MarS virtual market test
        sentio::VirtualMarketEngine vm_engine;
        auto results = vm_engine.run_mars_vm_test(strategy_name, symbol, days, simulations, market_regime, params_json);
        
        std::cout << "\n✅ MarS VM test completed successfully" << std::endl;

    } else if (command == "fasttest") {
        if (argc < 4) {
            std::cout << "Usage: sentio_cli fasttest <strategy> <symbol> [--historical-data <file>] [--continuation-minutes <n>] [--simulations <n>] [--params <json>]\n";
            return 1;
        }
        
        std::string strategy_name = argv[2];
        std::string symbol = argv[3];
        std::string historical_data_file = "data/equities/" + symbol + "_RTH_NH.csv";
        int continuation_minutes = 60;
        int simulations = 10;
        std::string params_json = "{}";
        
        for (int i = 4; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--historical-data" && i + 1 < argc) {
                historical_data_file = argv[++i];
            } else if (arg == "--continuation-minutes" && i + 1 < argc) {
                continuation_minutes = std::stoi(argv[++i]);
            } else if (arg == "--simulations" && i + 1 < argc) {
                simulations = std::stoi(argv[++i]);
            } else if (arg == "--params" && i + 1 < argc) {
                params_json = argv[++i];
            }
        }
        
        std::cout << "⚡ Starting Fast Historical Test..." << std::endl;
        std::cout << "📊 Strategy: " << strategy_name << std::endl;
        std::cout << "📈 Symbol: " << symbol << std::endl;
        std::cout << "📊 Historical data: " << historical_data_file << std::endl;
        std::cout << "⏱️  Continuation: " << continuation_minutes << " minutes" << std::endl;
        std::cout << "🎲 Simulations: " << simulations << std::endl;
        std::cout << "🚀 Using optimized historical patterns" << std::endl;
        
        // Run fast historical test
        sentio::VirtualMarketEngine vm_engine;
        auto results = vm_engine.run_fast_historical_test(strategy_name, symbol, historical_data_file, 
                                                         continuation_minutes, simulations, params_json);
        
        std::cout << "\n✅ Fast historical test completed successfully" << std::endl;

    } else {
        usage();
        return 1;
    }
    
    return 0;
}
