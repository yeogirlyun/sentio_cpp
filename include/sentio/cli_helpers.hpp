#pragma once

#include <string>
#include <vector>
#include <map>
#include <optional>

namespace sentio {

/**
 * CLI Argument Parsing Helpers
 * 
 * Provides standardized argument parsing for canonical sentio_cli interface
 */
class CLIHelpers {
public:
    struct ParsedArgs {
        std::string command;
        std::vector<std::string> positional_args;
        std::map<std::string, std::string> options;
        std::map<std::string, bool> flags;
        bool help_requested = false;
        bool verbose = false;
    };
    
    /**
     * Parse command line arguments into structured format
     */
    static ParsedArgs parse_arguments(int argc, char* argv[]);
    
    /**
     * Get string option with default value
     */
    static std::string get_option(const ParsedArgs& args, const std::string& key, 
                                 const std::string& default_value = "");
    
    /**
     * Get integer option with default value
     */
    static int get_int_option(const ParsedArgs& args, const std::string& key, int default_value = 0);
    
    /**
     * Get double option with default value
     */
    static double get_double_option(const ParsedArgs& args, const std::string& key, double default_value = 0.0);
    
    /**
     * Get boolean flag
     */
    static bool get_flag(const ParsedArgs& args, const std::string& key);
    
    /**
     * Parse period string (e.g., "3y", "6m", "2w", "5d", "4h") to days
     */
    static int parse_period_to_days(const std::string& period);
    
    /**
     * Parse period string to minutes
     */
    static int parse_period_to_minutes(const std::string& period);
    
    /**
     * Validate required positional arguments
     */
    static bool validate_required_args(const ParsedArgs& args, int min_required, 
                                      const std::string& usage_msg = "");
    
    /**
     * Print standardized help message
     */
    static void print_help(const std::string& command, const std::string& usage, 
                          const std::vector<std::string>& options,
                          const std::vector<std::string>& examples);
    
    /**
     * Print error message with usage hint
     */
    static void print_error(const std::string& error_msg, const std::string& usage_hint = "");
    
    /**
     * Validate symbol format
     */
    static bool is_valid_symbol(const std::string& symbol);
    
    /**
     * Validate strategy name format
     */
    static bool is_valid_strategy_name(const std::string& strategy_name);
    
    /**
     * Get available strategies list
     */
    static std::vector<std::string> get_available_strategies();
    
    /**
     * Format duration for display (e.g., 1800 minutes -> "30h" or "1.25d")
     */
    static std::string format_duration(int minutes);
    
    /**
     * Parse comma-separated list
     */
    static std::vector<std::string> parse_list(const std::string& list_str, char delimiter = ',');
    
    /**
     * Validate file path exists
     */
    static bool file_exists(const std::string& filepath);
    
    /**
     * Get default data file for symbol
     */
    static std::string get_default_data_file(const std::string& symbol, const std::string& suffix = "_NH.csv");

private:
    /**
     * Normalize option key (remove leading dashes, convert to lowercase)
     */
    static std::string normalize_option_key(const std::string& key);
    
    /**
     * Check if argument is an option (starts with -)
     */
    static bool is_option(const std::string& arg);
    
    /**
     * Check if argument is a flag (starts with -- and has no value)
     */
    static bool is_flag(const std::string& arg);
};

} // namespace sentio
