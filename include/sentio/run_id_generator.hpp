#pragma once

#include <string>

namespace sentio {

/**
 * Generate a unique 6-digit run ID
 * Format: NNNNNN (e.g., "123456")
 * 
 * Uses a combination of timestamp and random number to ensure uniqueness
 */
std::string generate_run_id();

/**
 * Create a descriptive note for the audit system
 * Format: "Strategy: <strategy_name>, Test: <test_type>, Period: <period_info>"
 */
std::string create_audit_note(const std::string& strategy_name, 
                             const std::string& test_type, 
                             const std::string& period_info = "");

} // namespace sentio
