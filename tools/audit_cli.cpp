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
              << "  format <audit_file.jsonl> [--output <output_file>] [--type <txt|csv>]\n";
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
        std::string output_file = "";
        std::string format_type = "txt";
        
        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--output" && i + 1 < argc) {
                output_file = argv[++i];
            } else if (arg == "--type" && i + 1 < argc) {
                format_type = argv[++i];
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
            *out << "Timestamp,Type,Symbol,Side,Quantity,Price,PnL,Cash,Equity,Real_Position" << std::endl;
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
                    
                    if (type == "trade") {
                        *out << "," << record.value("inst", "")
                             << "," << record.value("side", "")
                             << "," << record.value("qty", 0.0)
                             << "," << record.value("price", 0.0)
                             << "," << record.value("pnl", 0.0)
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
                } else {
                    // Human readable format
                    *out << "[" << std::setw(4) << line_num << "] " << time_buf << " ";
                    
                    if (type == "run_start") {
                        auto meta = record.value("meta", nlohmann::json::object());
                        *out << "RUN START - Strategy: " << meta.value("strategy", "")
                             << ", Series: " << meta.value("total_series", 0) << std::endl;
                    } else if (type == "trade") {
                        *out << "TRADE - " << record.value("side", "") << " "
                             << record.value("qty", 0.0) << " " << record.value("inst", "")
                             << " @ $" << std::fixed << std::setprecision(2) << record.value("price", 0.0)
                             << " (PnL: $" << record.value("pnl", 0.0) << ")" << std::endl;
                    } else if (type == "snapshot") {
                        *out << "PORTFOLIO - Cash: $" << std::fixed << std::setprecision(2) << record.value("cash", 0.0)
                             << ", Equity: $" << record.value("equity", 0.0)
                             << ", Positions: $" << record.value("real", 0.0) << std::endl;
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
    } else {
        usage();
        return 1;
    }
    
    return 0;
}
