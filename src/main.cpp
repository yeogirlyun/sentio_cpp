#include "sentio/core.hpp"
#include "sentio/runner.hpp"
#include "sentio/wf.hpp"
#include "sentio/csv_loader.hpp"
#include "sentio/symbol_table.hpp"
#include "sentio/profiling.hpp"
#include "sentio/data_resolver.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/all_strategies.hpp"
#include "sentio/rth_calendar.hpp" // **NEW**: Include for RTH check
#include "sentio/calendar_seed.hpp" // **NEW**: Include for calendar creation

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <cstdlib> // For std::exit
#include <sstream>
#include <ctime> // For std::time


namespace { // Anonymous namespace to ensure link-time registration
    struct StrategyRegistrar {
        StrategyRegistrar() {
            auto force_link_vwap = std::make_unique<sentio::VWAPReversionStrategy>();
            auto force_link_momentum = std::make_unique<sentio::MomentumVolumeProfileStrategy>();
            auto force_link_bollinger = std::make_unique<sentio::BollingerSqueezeBreakoutStrategy>();
            auto force_link_opening = std::make_unique<sentio::OpeningRangeBreakoutStrategy>();
            auto force_link_scalping = std::make_unique<sentio::OrderFlowScalpingStrategy>();
            auto force_link_imbalance = std::make_unique<sentio::OrderFlowImbalanceStrategy>();
            auto force_link_market = std::make_unique<sentio::MarketMakingStrategy>();
            auto force_link_hybrid_ppo = std::make_unique<sentio::HybridPPOStrategy>(sentio::HybridPPOCfg{});
        }
    };
    static StrategyRegistrar registrar;
}


void usage() {
    std::cout << "Usage: sentio_cli <command> [options]\n"
              << "Commands:\n"
              << "  backtest <symbol> [--strategy <name>] [--params <k=v,...>]\n"
              << "  wf <symbol> [--strategy <name>] [--params <k=v,...>]\n"
              << "  replay <run_id>\n";
}

int main(int argc, char* argv[]) {
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
                if (!calendar.is_rth_utc(bar.ts_nyt_epoch, "America/New_York")) {
                    std::cerr << "\nFATAL ERROR: Non-RTH data found after filtering!\n"
                              << " -> Symbol: " << ST.get_symbol(sid) << "\n"
                              << " -> Timestamp (UTC): " << bar.ts_utc << "\n"
                              << " -> NYT Epoch: " << bar.ts_nyt_epoch << "\n\n"
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

    } else if (command == "wf") {
        std::cout << "Walk-forward command is not fully implemented in this example.\n";
    } else if (command == "replay") {
        std::cout << "Replay command is not fully implemented in this example.\n";
    } else {
        usage();
        return 1;
    }
    
    return 0;
}
