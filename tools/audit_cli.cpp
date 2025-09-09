#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <nlohmann/json.hpp>

void usage() {
    std::cout << "Usage: audit_cli <command> [options]\n"
              << "Commands:\n"
              << "  replay <audit_file.jsonl> [--summary] [--trades] [--metrics]\n"
              << "  format <audit_file.jsonl> [--output <output_file>] [--type <txt|csv>] [--trades-only]\n"
              << "  trades <audit_file.jsonl> [--output <output_file>]\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string command = argv[1];
    
    if (command == "replay") {
        if (argc < 3) {
            std::cout << "Usage: audit_cli replay <audit_file.jsonl> [--summary] [--trades] [--metrics]\n";
            return 1;
        }
        
        std::string audit_file = argv[2];
        bool show_summary = false, show_trades = false, show_metrics = false;
        
        // Parse flags
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--summary") show_summary = true;
            else if (arg == "--trades") show_trades = true;
            else if (arg == "--metrics") show_metrics = true;
        }
        
        // Default to showing everything if no specific flags
        if (!show_summary && !show_trades && !show_metrics) {
            show_summary = show_trades = show_metrics = true;
        }
        
        // Process audit file
        std::ifstream file(audit_file);
        if (!file.is_open()) {
            std::cerr << "ERROR: Cannot open audit file: " << audit_file << std::endl;
            return 1;
        }
        
        std::vector<nlohmann::json> trades, snapshots, signals;
        std::string run_id, strategy_name;
        int total_records = 0;
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            total_records++;
            
            try {
                // Parse the line - handle the {data},"sha1":{hash} format
                size_t sha1_pos = line.find("},\"sha1\":");
                std::string json_part;
                if (sha1_pos != std::string::npos) {
                    json_part = line.substr(0, sha1_pos + 1);
                } else {
                    // Try alternative format {data},sha1:{hash}
                    sha1_pos = line.find("\",\"sha1\":");
                    if (sha1_pos != std::string::npos) {
                        json_part = line.substr(0, sha1_pos + 1) + "}";
                    } else {
                        // Fallback - try to parse entire line
                        json_part = line;
                    }
                }
                
                nlohmann::json record = nlohmann::json::parse(json_part);
                
                std::string type = record.value("type", "");
                if (run_id.empty()) run_id = record.value("run", "");
                
                if (type == "run_start") {
                    auto meta = record.value("meta", nlohmann::json::object());
                    strategy_name = meta.value("strategy", "");
                } else if (type == "trade") {
                    trades.push_back(record);
                } else if (type == "snapshot") {
                    snapshots.push_back(record);
                } else if (type == "signal") {
                    signals.push_back(record);
                }
            } catch (const std::exception& e) {
                // Skip malformed lines
                continue;
            }
        }
        file.close();
        
        // Display results
        if (show_summary) {
            std::cout << "=== AUDIT REPLAY SUMMARY ===" << std::endl;
            std::cout << "Run ID: " << run_id << std::endl;
            std::cout << "Strategy: " << strategy_name << std::endl;
            std::cout << "Total Records: " << total_records << std::endl;
            std::cout << "Trades: " << trades.size() << std::endl;
            std::cout << "Snapshots: " << snapshots.size() << std::endl;
            std::cout << "Signals: " << signals.size() << std::endl;
            std::cout << std::endl;
        }
        
        if (show_metrics && !snapshots.empty()) {
            auto initial = snapshots.front();
            auto final = snapshots.back();
            
            double initial_equity = initial.value("equity", 100000.0);
            double final_equity = final.value("equity", 100000.0);
            double total_return = (final_equity - initial_equity) / initial_equity;
            double monthly_return = std::pow(final_equity / initial_equity, 1.0/3.0) - 1.0;
            
            std::cout << "=== PERFORMANCE METRICS ===" << std::endl;
            std::cout << "Initial Equity: $" << std::fixed << std::setprecision(2) << initial_equity << std::endl;
            std::cout << "Final Equity: $" << std::fixed << std::setprecision(2) << final_equity << std::endl;
            std::cout << "Total Return: " << std::fixed << std::setprecision(4) << total_return << " (" << total_return*100 << "%)" << std::endl;
            std::cout << "Monthly Return: " << std::fixed << std::setprecision(4) << monthly_return << " (" << monthly_return*100 << "%)" << std::endl;
            
            if (!trades.empty()) {
                std::cout << "Total Trades: " << trades.size() << std::endl;
                std::cout << "Avg Trades/Day: " << std::fixed << std::setprecision(1) << trades.size() / 63.0 << std::endl;
            }
            std::cout << std::endl;
        }
        
        if (show_trades && !trades.empty()) {
            std::cout << "=== RECENT TRADES (Last 10) ===" << std::endl;
            std::cout << "Time                Side Instr   Quantity    Price      PnL" << std::endl;
            std::cout << "----------------------------------------------------------------" << std::endl;
            
            size_t start_idx = trades.size() > 10 ? trades.size() - 10 : 0;
            for (size_t i = start_idx; i < trades.size(); i++) {
                auto& trade = trades[i];
                
                std::string side = trade.value("side", "");
                std::string inst = trade.value("inst", "");
                double qty = trade.value("qty", 0.0);
                double price = trade.value("price", 0.0);
                double pnl = trade.value("pnl", 0.0);
                int64_t ts = trade.value("ts", 0);
                
                // Convert timestamp
                std::time_t tt = static_cast<std::time_t>(ts);
                std::tm tm{};
                gmtime_r(&tt, &tm);
                char time_buf[32];
                std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M", &tm);
                
                std::cout << std::setw(19) << time_buf
                          << std::setw(5) << side
                          << std::setw(7) << inst
                          << std::setw(10) << std::fixed << std::setprecision(0) << qty
                          << std::setw(10) << std::fixed << std::setprecision(2) << price
                          << std::setw(10) << std::fixed << std::setprecision(2) << pnl
                          << std::endl;
            }
        }

    } else if (command == "format") {
        if (argc < 3) {
            std::cout << "Usage: audit_cli format <audit_file.jsonl> [--output <file>] [--type <txt|csv>]\n";
            return 1;
        }
        
        std::string audit_file = argv[2];
        
        // Extract base filename and generate descriptive output name
        std::string base_name = audit_file;
        
        // Remove path if present
        size_t last_slash = base_name.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            base_name = base_name.substr(last_slash + 1);
        }
        
        // Remove .jsonl extension
        if (base_name.size() > 6 && base_name.substr(base_name.size() - 6) == ".jsonl") {
            base_name = base_name.substr(0, base_name.size() - 6);
        }
        
        std::string output_file = "audit/" + base_name + "_human_readable.txt";
        std::string format_type = "txt";
        bool trades_only = false;
        
        // Parse options (allow override)
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--output" && i + 1 < argc) {
                output_file = argv[++i];
            } else if (arg == "--type" && i + 1 < argc) {
                format_type = argv[++i];
                // Update default extension based on type
                if (format_type == "csv" && output_file.find("_human_readable.txt") != std::string::npos) {
                    output_file = "audit/" + base_name + "_data.csv";
                }
            } else if (arg == "--trades-only") {
                trades_only = true;
                if (output_file.find("_human_readable.txt") != std::string::npos) {
                    output_file = "audit/" + base_name + "_trades_only.txt";
                }
            }
        }
        
        // Process audit file
        std::ifstream file(audit_file);
        if (!file.is_open()) {
            std::cerr << "ERROR: Cannot open audit file: " << audit_file << std::endl;
            return 1;
        }
        
        std::ostream* out = &std::cout;
        std::ofstream out_file;
        if (!output_file.empty()) {
            out_file.open(output_file);
            if (out_file.is_open()) {
                out = &out_file;
            } else {
                std::cerr << "ERROR: Cannot create output file: " << output_file << std::endl;
                return 1;
            }
        }
        
        // Write header
        if (format_type == "csv") {
            *out << "Timestamp,Type,Symbol,Side,Quantity,Price,Trade_PnL,Cash,Realized_PnL,Unrealized_PnL,Total_Equity" << std::endl;
        } else if (trades_only) {
            *out << "TRADES ONLY - AUDIT LOG" << std::endl;
            *out << "=======================" << std::endl;
            *out << std::endl;
            *out << "Format: [#] TIMESTAMP | TICKER | BUY/SELL | QUANTITY @ PRICE | EQUITY_AFTER" << std::endl;
            *out << "---------------------------------------------------------------------------------" << std::endl;
            *out << std::endl;
        } else {
            *out << "HUMAN-READABLE AUDIT LOG" << std::endl;
            *out << "========================" << std::endl;
            *out << std::endl;
        }
        
        std::string line;
        int line_num = 0;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            line_num++;
            
            try {
                // Parse the line - handle the {data},"sha1":{hash} format
                size_t sha1_pos = line.find("},\"sha1\":");
                std::string json_part;
                if (sha1_pos != std::string::npos) {
                    json_part = line.substr(0, sha1_pos + 1);
                } else {
                    // Try alternative format {data},sha1:{hash}
                    sha1_pos = line.find("\",\"sha1\":");
                    if (sha1_pos != std::string::npos) {
                        json_part = line.substr(0, sha1_pos + 1) + "}";
                    } else {
                        // Fallback - try to parse entire line
                        json_part = line;
                    }
                }
                
                nlohmann::json record = nlohmann::json::parse(json_part);
                
                std::string type = record.value("type", "");
                int64_t ts = record.value("ts", 0);
                
                // Convert timestamp
                std::time_t tt = static_cast<std::time_t>(ts);
                std::tm tm{};
                gmtime_r(&tt, &tm);
                char time_buf[32];
                std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &tm);
                
                if (format_type == "csv") {
                    *out << time_buf << "," << type;
                    
                    if (type == "fill") {
                        std::string side_str = (record.value("side", 0) == 0) ? "BUY" : "SELL";
                        *out << "," << record.value("inst", "")
                             << "," << side_str
                             << "," << record.value("qty", 0.0)
                             << "," << record.value("px", 0.0)
                             << "," << record.value("pnl_d", 0.0)
                             << ",,";
                    } else if (type == "snapshot") {
                        *out << ",,,,,"
                             << "," << record.value("cash", 0.0)
                             << "," << record.value("equity", 0.0)
                             << "," << record.value("real", 0.0);
                    } else {
                        *out << ",,,,,,,";
                    }
                    *out << std::endl;
                } else if (trades_only) {
                    // Trades-only format
                    if (type == "fill") {
                        std::string side_str = (record.value("side", 0) == 0) ? "BUY" : "SELL";
                        *out << "[" << std::setw(4) << line_num << "] " << time_buf << " | ";
                        *out << std::setw(5) << record.value("inst", "") << " | ";
                        *out << std::setw(4) << side_str << " | ";
                        *out << std::setw(8) << std::fixed << std::setprecision(0) << record.value("qty", 0.0) << " @ ";
                        *out << "$" << std::fixed << std::setprecision(2) << record.value("px", 0.0) << " | ";
                        *out << "Equity: $" << record.value("eq_after", 0.0);
                        if (record.contains("pnl_d") && record.value("pnl_d", 0.0) != 0.0) {
                            *out << " | P&L: $" << record.value("pnl_d", 0.0);
                        }
                        *out << std::endl;
                    }
                    // Skip all other record types in trades-only mode
                } else {
                    // Human readable format
                    *out << "[" << std::setw(4) << line_num << "] " << time_buf << " ";
                    
                    if (type == "run_start") {
                        auto meta = record.value("meta", nlohmann::json::object());
                        *out << "RUN START - Strategy: " << meta.value("strategy", "")
                             << ", Series: " << meta.value("total_series", 0) << std::endl;
                    } else if (type == "fill") {
                        std::string side_str = (record.value("side", 0) == 0) ? "BUY" : "SELL";
                        *out << "TRADE - " << side_str << " "
                             << record.value("qty", 0.0) << " " << record.value("inst", "")
                             << " @ $" << std::fixed << std::setprecision(2) << record.value("px", 0.0)
                             << " (P&L: $" << record.value("pnl_d", 0.0) << ")" << std::endl;
                    } else if (type == "snapshot") {
                        double cash = record.value("cash", 0.0);
                        double equity = record.value("equity", 0.0);
                        double realized = record.value("real", 0.0);
                        double unrealized = equity - cash - realized;
                        
                        *out << "PORTFOLIO - Cash: $" << std::fixed << std::setprecision(2) << cash
                             << ", Realized P&L: $" << realized
                             << ", Unrealized P&L: $" << unrealized 
                             << ", Total Equity: $" << equity
                             << " (Cash + Realized + Unrealized = " << (cash + realized + unrealized) << ")" << std::endl;
                    } else if (type == "signal") {
                        *out << "SIGNAL - " << record.value("inst", "")
                             << " p=" << std::fixed << std::setprecision(3) << record.value("p", 0.0)
                             << " conf=" << record.value("conf", 0.0) << std::endl;
                    } else if (type == "bar") {
                        *out << "BAR - " << record.value("inst", "")
                             << " O:" << std::fixed << std::setprecision(2) << record.value("o", 0.0)
                             << " H:" << record.value("h", 0.0)
                             << " L:" << record.value("l", 0.0)
                             << " C:" << record.value("c", 0.0)
                             << " V:" << std::fixed << std::setprecision(0) << record.value("v", 0.0) << std::endl;
                    } else {
                        *out << type << " - " << record.dump() << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                if (format_type != "csv") {
                    *out << "[" << std::setw(4) << line_num << "] ERROR: Malformed line" << std::endl;
                }
            }
        }
        file.close();
        
        if (out_file.is_open()) {
            out_file.close();
            std::cout << "Human-readable audit log written to: " << output_file << std::endl;
        }
    } else if (command == "trades") {
        if (argc < 3) {
            std::cout << "Usage: audit_cli trades <audit_file.jsonl> [--output <file>]\n";
            return 1;
        }
        
        std::string audit_file = argv[2];
        
        // Extract base filename and generate descriptive output name
        std::string base_name = audit_file;
        
        // Remove path if present
        size_t last_slash = base_name.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            base_name = base_name.substr(last_slash + 1);
        }
        
        // Remove .jsonl extension
        if (base_name.size() > 6 && base_name.substr(base_name.size() - 6) == ".jsonl") {
            base_name = base_name.substr(0, base_name.size() - 6);
        }
        
        std::string output_file = "audit/" + base_name + "_trades_only.txt";
        
        // Parse options (allow override)
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--output" && i + 1 < argc) {
                output_file = argv[++i];
            }
        }
        
        // Process audit file for trades only
        std::ifstream file(audit_file);
        if (!file.is_open()) {
            std::cerr << "ERROR: Cannot open audit file: " << audit_file << std::endl;
            return 1;
        }
        
        std::ofstream out_file;
        std::ostream* out = &std::cout;
        
        if (!output_file.empty()) {
            out_file.open(output_file);
            if (!out_file.is_open()) {
                std::cerr << "ERROR: Cannot create output file: " << output_file << std::endl;
                return 1;
            }
            out = &out_file;
        }
        
        // Write header for trades-only
        *out << "TRADES ONLY - AUDIT LOG" << std::endl;
        *out << "=======================" << std::endl;
        *out << std::endl;
        *out << "Format: [#] TIMESTAMP | TICKER | BUY/SELL | QUANTITY @ PRICE | EQUITY_AFTER | P&L" << std::endl;
        *out << "-------------------------------------------------------------------------------------" << std::endl;
        *out << std::endl;
        
        std::string line;
        int line_num = 0;
        int trade_count = 0;
        
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            line_num++;
            
            try {
                // Parse the line - handle the {data},"sha1":{hash} format
                size_t sha1_pos = line.find("},\"sha1\":");
                std::string json_part;
                if (sha1_pos != std::string::npos) {
                    json_part = line.substr(0, sha1_pos + 1);
                } else {
                    // Try alternative format {data},sha1:{hash}
                    sha1_pos = line.find("\",\"sha1\":");
                    if (sha1_pos != std::string::npos) {
                        json_part = line.substr(0, sha1_pos + 1) + "}";
                    } else {
                        // Fallback - treat as complete JSON
                        json_part = line;
                    }
                }
                
                nlohmann::json record = nlohmann::json::parse(json_part);
                std::string type = record.value("type", "");
                
                if (type == "fill") {
                    trade_count++;
                    
                    // Extract timestamp
                    std::int64_t ts_epoch = record.value("ts", 0);
                    std::time_t tt = static_cast<std::time_t>(ts_epoch);
                    std::tm tm{};
                    gmtime_r(&tt, &tm);
                    char time_buf[32];
                    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &tm);
                    
                    // Format trade details
                    std::string side_str = (record.value("side", 0) == 0) ? "BUY" : "SELL";
                    *out << "[" << std::setw(4) << trade_count << "] " << time_buf << " | ";
                    *out << std::setw(5) << record.value("inst", "") << " | ";
                    *out << std::setw(4) << side_str << " | ";
                    *out << std::setw(8) << std::fixed << std::setprecision(0) << record.value("qty", 0.0) << " @ ";
                    *out << "$" << std::fixed << std::setprecision(2) << record.value("px", 0.0) << " | ";
                    *out << "Equity: $" << record.value("eq_after", 0.0);
                    
                    if (record.contains("pnl_d") && record.value("pnl_d", 0.0) != 0.0) {
                        *out << " | P&L: $" << record.value("pnl_d", 0.0);
                    }
                    *out << std::endl;
                }
            } catch (const std::exception& e) {
                // Skip malformed lines
                continue;
            }
        }
        file.close();
        
        if (trade_count == 0) {
            *out << "No trades found in audit file." << std::endl;
        } else {
            *out << std::endl;
            *out << "Total trades: " << trade_count << std::endl;
        }
        
        if (out_file.is_open()) {
            out_file.close();
            std::cout << "Trades-only audit log written to: " << output_file << std::endl;
            std::cout << "Total trades found: " << trade_count << std::endl;
        }
        
    } else {
        usage();
        return 1;
    }
    
    return 0;
}
