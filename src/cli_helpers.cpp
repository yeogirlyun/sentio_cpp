#include "sentio/cli_helpers.hpp"
#include "sentio/base_strategy.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <regex>
#include <filesystem>
#include <sstream>

namespace sentio {

CLIHelpers::ParsedArgs CLIHelpers::parse_arguments(int argc, char* argv[]) {
    ParsedArgs args;
    
    if (argc < 2) {
        return args;
    }
    
    args.command = argv[1];
    
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        
        // Check for help flag
        if (arg == "--help" || arg == "-h") {
            args.help_requested = true;
            continue;
        }
        
        // Check for verbose flag
        if (arg == "--verbose" || arg == "-v") {
            args.verbose = true;
            args.flags["verbose"] = true;
            continue;
        }
        
        // Check for options (--key value or --key=value)
        if (arg.starts_with("--")) {
            std::string key, value;
            
            size_t eq_pos = arg.find('=');
            if (eq_pos != std::string::npos) {
                // --key=value format
                key = arg.substr(2, eq_pos - 2);
                value = arg.substr(eq_pos + 1);
            } else {
                // --key value format
                key = arg.substr(2);
                if (i + 1 < argc && !(argv[i + 1][0] == '-')) {
                    value = argv[++i];
                } else {
                    // Flag without value
                    args.flags[normalize_option_key(key)] = true;
                    continue;
                }
            }
            
            args.options[normalize_option_key(key)] = value;
        }
        // Check for short options (-k value)
        else if (arg.starts_with("-") && arg.length() == 2) {
            std::string key = arg.substr(1);
            std::string value;
            
            if (i + 1 < argc && !(argv[i + 1][0] == '-')) {
                value = argv[++i];
                args.options[normalize_option_key(key)] = value;
            } else {
                args.flags[normalize_option_key(key)] = true;
            }
        }
        // Positional argument
        else {
            args.positional_args.push_back(arg);
        }
    }
    
    return args;
}

std::string CLIHelpers::get_option(const ParsedArgs& args, const std::string& key, 
                                  const std::string& default_value) {
    std::string normalized_key = normalize_option_key(key);
    auto it = args.options.find(normalized_key);
    return (it != args.options.end()) ? it->second : default_value;
}

int CLIHelpers::get_int_option(const ParsedArgs& args, const std::string& key, int default_value) {
    std::string value = get_option(args, key);
    if (value.empty()) {
        return default_value;
    }
    
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        std::cerr << "Warning: Invalid integer value '" << value << "' for option --" << key 
                  << ", using default " << default_value << std::endl;
        return default_value;
    }
}

double CLIHelpers::get_double_option(const ParsedArgs& args, const std::string& key, double default_value) {
    std::string value = get_option(args, key);
    if (value.empty()) {
        return default_value;
    }
    
    try {
        return std::stod(value);
    } catch (const std::exception&) {
        std::cerr << "Warning: Invalid double value '" << value << "' for option --" << key 
                  << ", using default " << default_value << std::endl;
        return default_value;
    }
}

bool CLIHelpers::get_flag(const ParsedArgs& args, const std::string& key) {
    std::string normalized_key = normalize_option_key(key);
    auto it = args.flags.find(normalized_key);
    return (it != args.flags.end()) ? it->second : false;
}

int CLIHelpers::parse_period_to_days(const std::string& period) {
    std::regex period_regex(R"((\d+)([yMwdh]))");
    std::smatch match;
    
    if (!std::regex_match(period, match, period_regex)) {
        std::cerr << "Warning: Invalid period format '" << period << "', using default 30d" << std::endl;
        return 30;
    }
    
    int value = std::stoi(match[1].str());
    char unit = match[2].str()[0];
    
    switch (unit) {
        case 'y': return value * 365;
        case 'M': return value * 30;
        case 'w': return value * 7;
        case 'd': return value;
        case 'h': return std::max(1, value / 24);
        default: return 30;
    }
}

int CLIHelpers::parse_period_to_minutes(const std::string& period) {
    std::regex period_regex(R"((\d+)([yMwdhm]))");
    std::smatch match;
    
    if (!std::regex_match(period, match, period_regex)) {
        std::cerr << "Warning: Invalid period format '" << period << "', using default 1d" << std::endl;
        return 390; // 1 trading day
    }
    
    int value = std::stoi(match[1].str());
    char unit = match[2].str()[0];
    
    switch (unit) {
        case 'y': return value * 252 * 390; // 252 trading days per year
        case 'M': return value * 22 * 390;  // ~22 trading days per month
        case 'w': return value * 5 * 390;   // 5 trading days per week
        case 'd': return value * 390;       // 390 minutes per trading day
        case 'h': return value * 60;
        case 'm': return value;
        default: return 390;
    }
}

bool CLIHelpers::validate_required_args(const ParsedArgs& args, int min_required, 
                                        const std::string& usage_msg) {
    if (static_cast<int>(args.positional_args.size()) < min_required) {
        print_error("Insufficient arguments", usage_msg);
        return false;
    }
    return true;
}

void CLIHelpers::print_help(const std::string& command, const std::string& usage,
                           const std::vector<std::string>& options,
                           const std::vector<std::string>& examples) {
    (void)command; // Suppress unused parameter warning
    std::cout << "Usage: " << usage << std::endl << std::endl;
    
    if (!options.empty()) {
        std::cout << "Options:" << std::endl;
        for (const auto& option : options) {
            std::cout << "  " << option << std::endl;
        }
        std::cout << std::endl;
    }
    
    if (!examples.empty()) {
        std::cout << "Examples:" << std::endl;
        for (const auto& example : examples) {
            std::cout << "  " << example << std::endl;
        }
        std::cout << std::endl;
    }
}

void CLIHelpers::print_error(const std::string& error_msg, const std::string& usage_hint) {
    std::cerr << "Error: " << error_msg << std::endl;
    if (!usage_hint.empty()) {
        std::cerr << "Usage: " << usage_hint << std::endl;
    }
    std::cerr << "Use --help for more information." << std::endl;
}

bool CLIHelpers::is_valid_symbol(const std::string& symbol) {
    // Basic symbol validation: 1-5 uppercase letters
    std::regex symbol_regex(R"([A-Z]{1,5})");
    return std::regex_match(symbol, symbol_regex);
}

bool CLIHelpers::is_valid_strategy_name(const std::string& strategy_name) {
    // Basic strategy name validation: alphanumeric and underscores
    std::regex strategy_regex(R"([a-zA-Z][a-zA-Z0-9_]*)");
    return std::regex_match(strategy_name, strategy_regex);
}

std::vector<std::string> CLIHelpers::get_available_strategies() {
    // This would ideally query the StrategyFactory for available strategies
    // For now, return a hardcoded list of common strategies
    return {
        "ire", "momentum", "mean_reversion", "rsi", "sma_cross", 
        "bollinger", "macd", "stochastic", "williams_r"
    };
}

std::string CLIHelpers::format_duration(int minutes) {
    if (minutes < 60) {
        return std::to_string(minutes) + "m";
    } else if (minutes < 390) {
        return std::to_string(minutes / 60) + "h";
    } else if (minutes < 390 * 7) {
        double days = static_cast<double>(minutes) / 390.0;
        if (days == static_cast<int>(days)) {
            return std::to_string(static_cast<int>(days)) + "d";
        } else {
            return std::to_string(days).substr(0, 4) + "d";
        }
    } else if (minutes < 390 * 30) {
        return std::to_string(minutes / (390 * 7)) + "w";
    } else {
        return std::to_string(minutes / (390 * 22)) + "M";
    }
}

std::vector<std::string> CLIHelpers::parse_list(const std::string& list_str, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(list_str);
    std::string item;
    
    while (std::getline(ss, item, delimiter)) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    
    return result;
}

bool CLIHelpers::file_exists(const std::string& filepath) {
    return std::filesystem::exists(filepath);
}

std::string CLIHelpers::get_default_data_file(const std::string& symbol, const std::string& suffix) {
    return "data/equities/" + symbol + suffix;
}

std::string CLIHelpers::normalize_option_key(const std::string& key) {
    std::string normalized = key;
    
    // Remove leading dashes
    while (normalized.starts_with("-")) {
        normalized = normalized.substr(1);
    }
    
    // Convert to lowercase
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    
    return normalized;
}

bool CLIHelpers::is_option(const std::string& arg) {
    return arg.starts_with("-");
}

bool CLIHelpers::is_flag(const std::string& arg) {
    return arg.starts_with("--") && arg.find('=') == std::string::npos;
}

bool CLIHelpers::validate_options(const ParsedArgs& args, const std::string& command,
                                 const std::vector<std::string>& allowed_options,
                                 const std::vector<std::string>& allowed_flags) {
    // Check for unknown options
    for (const auto& [option, value] : args.options) {
        bool found = false;
        for (const auto& allowed : allowed_options) {
            if (normalize_option_key(option) == normalize_option_key(allowed)) {
                found = true;
                break;
            }
        }
        if (!found) {
            print_unknown_option_error(option, command, allowed_options, allowed_flags);
            return false;
        }
    }
    
    // Check for unknown flags
    for (const auto& [flag, value] : args.flags) {
        // Skip global flags
        if (flag == "verbose" || flag == "help" || flag == "h" || flag == "v") {
            continue;
        }
        
        bool found = false;
        for (const auto& allowed : allowed_flags) {
            if (normalize_option_key(flag) == normalize_option_key(allowed)) {
                found = true;
                break;
            }
        }
        if (!found) {
            print_unknown_option_error(flag, command, allowed_options, allowed_flags);
            return false;
        }
    }
    
    return true;
}

void CLIHelpers::print_unknown_option_error(const std::string& unknown_option,
                                           const std::string& command,
                                           const std::vector<std::string>& allowed_options,
                                           const std::vector<std::string>& allowed_flags) {
    std::cout << "\033[31m❌ ERROR:\033[0m Unknown option '--" << unknown_option << "' for command '" << command << "'" << std::endl;
    std::cout << std::endl;
    
    // Find similar options (simple string matching)
    std::vector<std::string> suggestions;
    std::string normalized_unknown = normalize_option_key(unknown_option);
    
    // Check for partial matches in allowed options
    for (const auto& allowed : allowed_options) {
        std::string normalized_allowed = normalize_option_key(allowed);
        if (normalized_allowed.find(normalized_unknown) != std::string::npos ||
            normalized_unknown.find(normalized_allowed) != std::string::npos) {
            suggestions.push_back(allowed);
        }
    }
    
    // Check for partial matches in allowed flags
    for (const auto& allowed : allowed_flags) {
        std::string normalized_allowed = normalize_option_key(allowed);
        if (normalized_allowed.find(normalized_unknown) != std::string::npos ||
            normalized_unknown.find(normalized_allowed) != std::string::npos) {
            suggestions.push_back(allowed);
        }
    }
    
    if (!suggestions.empty()) {
        std::cout << "\033[33m💡 Did you mean:\033[0m" << std::endl;
        for (const auto& suggestion : suggestions) {
            std::cout << "  --" << suggestion << std::endl;
        }
        std::cout << std::endl;
    }
    
    std::cout << "\033[36m📋 Available options for '" << command << "':\033[0m" << std::endl;
    
    if (!allowed_options.empty()) {
        std::cout << "\033[1mOptions (require values):\033[0m" << std::endl;
        for (const auto& option : allowed_options) {
            std::cout << "  --" << option << " <value>" << std::endl;
        }
        std::cout << std::endl;
    }
    
    if (!allowed_flags.empty()) {
        std::cout << "\033[1mFlags (no values):\033[0m" << std::endl;
        for (const auto& flag : allowed_flags) {
            std::cout << "  --" << flag << std::endl;
        }
        std::cout << std::endl;
    }
    
    std::cout << "\033[1mGlobal options:\033[0m" << std::endl;
    std::cout << "  --help, -h                Show command help" << std::endl;
    std::cout << "  --verbose, -v             Enable verbose output" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Use 'sentio_cli " << command << " --help' for detailed usage information." << std::endl;
}

} // namespace sentio
