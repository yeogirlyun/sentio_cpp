#include "sentio/backend_audit_events.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace sentio {

using json = nlohmann::json;

// **BACKEND AUDIT RECORDER IMPLEMENTATION**

BackendAuditRecorder::BackendAuditRecorder(audit::DB* db, const std::string& run_id, bool enabled)
    : audit_db_(db), current_run_id_(run_id), logging_enabled_(enabled) {
    if (!audit_db_) {
        throw std::runtime_error("BackendAuditRecorder: audit database is null");
    }
}

void BackendAuditRecorder::record_backend_stage(const BackendAuditEvent& event) {
    if (!logging_enabled_ || !audit_db_) return;
    
    // Convert backend stage to string
    std::string stage_name;
    switch (event.stage) {
        case BackendStage::STRATEGY_SIGNAL_GENERATION: stage_name = "STRATEGY_SIGNAL_GENERATION"; break;
        case BackendStage::ALLOCATION_DECISION_MAKING: stage_name = "ALLOCATION_DECISION_MAKING"; break;
        case BackendStage::CONFLICT_RESOLUTION: stage_name = "CONFLICT_RESOLUTION"; break;
        case BackendStage::POSITION_SIZING: stage_name = "POSITION_SIZING"; break;
        case BackendStage::EXECUTION_ROUTING: stage_name = "EXECUTION_ROUTING"; break;
        case BackendStage::PORTFOLIO_UPDATE: stage_name = "PORTFOLIO_UPDATE"; break;
        default: stage_name = "UNKNOWN_STAGE"; break;
    }
    
    // Create audit event
    audit::Event audit_event;
    audit_event.run_id = event.run_id;
    audit_event.ts_millis = event.timestamp;
    audit_event.kind = "BACKEND_STAGE";
    audit_event.symbol = event.base_symbol;
    audit_event.side = event.success ? "SUCCESS" : "FAILURE";
    audit_event.qty = event.processing_time_us;
    audit_event.price = 0.0;
    audit_event.pnl_delta = 0.0;
    audit_event.weight = 0.0;
    audit_event.prob = 0.0;
    audit_event.reason = stage_name + " | " + event.strategy_name;
    
    // Pack detailed data into note field
    json note_data;
    note_data["chain_id"] = event.chain_id;
    note_data["stage"] = stage_name;
    note_data["strategy"] = event.strategy_name;
    note_data["stage_data"] = event.stage_data;
    note_data["processing_time_us"] = event.processing_time_us;
    note_data["success"] = event.success;
    if (!event.error_message.empty()) {
        note_data["error"] = event.error_message;
    }
    if (!event.notes.empty()) {
        note_data["notes"] = event.notes;
    }
    
    audit_event.note = note_data.dump();
    
    // Record to audit database
    audit_db_->append_event(audit_event);
}

void BackendAuditRecorder::record_strategy_signal_processed(const StrategySignalAuditEvent& event) {
    if (!logging_enabled_ || !audit_db_) return;
    
    // Create audit event for strategy signal processing
    audit::Event audit_event;
    audit_event.run_id = event.run_id;
    audit_event.ts_millis = event.timestamp;
    audit_event.kind = "STRATEGY_SIGNAL_PROCESSED";
    audit_event.symbol = event.base_symbol;
    audit_event.side = "PROCESSED";
    audit_event.qty = static_cast<double>(event.raw_decisions.size());
    audit_event.price = event.signal_strength;
    audit_event.pnl_delta = 0.0;
    audit_event.weight = 0.0;
    audit_event.prob = event.signal_strength;
    audit_event.reason = event.strategy_name + " | " + event.signal_reason;
    
    // Pack detailed data into note field
    json note_data;
    note_data["chain_id"] = event.chain_id;
    note_data["strategy"] = event.strategy_name;
    note_data["signal_strength"] = event.signal_strength;
    note_data["signal_reason"] = event.signal_reason;
    note_data["raw_decisions"] = serialize_allocation_decisions(event.raw_decisions);
    note_data["processed_decisions"] = serialize_allocation_decisions(event.processed_decisions);
    note_data["portfolio_state"] = serialize_portfolio_state(event.current_portfolio_state);
    note_data["backend_modifications"] = event.backend_modifications;
    note_data["strategy_compute_time_us"] = event.strategy_compute_time_us;
    note_data["backend_process_time_us"] = event.backend_process_time_us;
    
    audit_event.note = note_data.dump();
    
    // Record to audit database
    audit_db_->append_event(audit_event);
}

void BackendAuditRecorder::record_allocation_decision(const AllocationDecisionAuditEvent& event) {
    if (!logging_enabled_ || !audit_db_) return;
    
    // Create audit event for allocation decision
    audit::Event audit_event;
    audit_event.run_id = event.run_id;
    audit_event.ts_millis = event.timestamp;
    audit_event.kind = "ALLOCATION_DECISION";
    audit_event.symbol = ""; // Multiple symbols involved
    audit_event.side = "DECISION";
    audit_event.qty = static_cast<double>(event.output_decisions.size());
    audit_event.price = static_cast<double>(event.conflicts_resolved_count);
    audit_event.pnl_delta = 0.0;
    audit_event.weight = 0.0;
    audit_event.prob = 0.0;
    audit_event.reason = event.strategy_name + " | " + event.decision_algorithm;
    
    // Pack detailed data into note field
    json note_data;
    note_data["chain_id"] = event.chain_id;
    note_data["strategy"] = event.strategy_name;
    note_data["decision_algorithm"] = event.decision_algorithm;
    note_data["input_portfolio"] = serialize_portfolio_state(event.input_portfolio);
    note_data["input_decisions"] = serialize_allocation_decisions(event.input_decisions);
    note_data["output_decisions"] = serialize_allocation_decisions(event.output_decisions);
    note_data["conflicts_detected"] = event.conflicts_detected;
    note_data["conflicts_resolved"] = event.conflicts_resolved;
    note_data["resolution_reasoning"] = event.resolution_reasoning;
    note_data["processing_time_us"] = event.processing_time_us;
    note_data["decisions_modified"] = event.decisions_modified;
    note_data["conflicts_resolved_count"] = event.conflicts_resolved_count;
    
    // Add conflict analysis
    json conflict_analysis;
    for (const auto& [instrument, score] : event.conflict_analysis) {
        conflict_analysis[instrument] = score;
    }
    note_data["conflict_analysis"] = conflict_analysis;
    
    audit_event.note = note_data.dump();
    
    // Record to audit database
    audit_db_->append_event(audit_event);
}

void BackendAuditRecorder::record_backend_execution_summary(const BackendExecutionSummaryEvent& event) {
    if (!logging_enabled_ || !audit_db_) return;
    
    // Create audit event for backend execution summary
    audit::Event audit_event;
    audit_event.run_id = event.run_id;
    audit_event.ts_millis = event.timestamp;
    audit_event.kind = "BACKEND_EXECUTION_SUMMARY";
    audit_event.symbol = ""; // Multiple symbols involved
    audit_event.side = "SUMMARY";
    audit_event.qty = static_cast<double>(event.orders_executed);
    audit_event.price = event.execution_success_rate;
    audit_event.pnl_delta = event.realized_pnl_delta;
    audit_event.weight = (event.portfolio_value_after - event.portfolio_value_before) / event.portfolio_value_before;
    audit_event.prob = event.execution_success_rate;
    audit_event.reason = event.strategy_name + " | Backend Execution Summary";
    
    // Pack detailed data into note field
    json note_data;
    note_data["chain_id"] = event.chain_id;
    note_data["strategy"] = event.strategy_name;
    note_data["total_processing_time_us"] = event.total_processing_time_us;
    note_data["stage_timings"] = serialize_stage_timings(event.stage_timings);
    note_data["signals_processed"] = event.signals_processed;
    note_data["decisions_generated"] = event.decisions_generated;
    note_data["conflicts_resolved"] = event.conflicts_resolved;
    note_data["orders_executed"] = event.orders_executed;
    note_data["fills_completed"] = event.fills_completed;
    note_data["portfolio_value_before"] = event.portfolio_value_before;
    note_data["portfolio_value_after"] = event.portfolio_value_after;
    note_data["realized_pnl_delta"] = event.realized_pnl_delta;
    note_data["unrealized_pnl_delta"] = event.unrealized_pnl_delta;
    note_data["signal_to_execution_latency_us"] = event.signal_to_execution_latency_us;
    note_data["execution_success_rate"] = event.execution_success_rate;
    note_data["performance_notes"] = event.performance_notes;
    
    audit_event.note = note_data.dump();
    
    // Record to audit database
    audit_db_->append_event(audit_event);
}

// **CONVENIENCE METHODS**

void BackendAuditRecorder::record_signal_generation_start(const std::string& chain_id, 
                                                         const std::string& strategy_name,
                                                         const std::string& base_symbol, 
                                                         std::int64_t timestamp) {
    BackendAuditEvent event;
    event.timestamp = timestamp;
    event.run_id = current_run_id_;
    event.chain_id = chain_id;
    event.stage = BackendStage::STRATEGY_SIGNAL_GENERATION;
    event.strategy_name = strategy_name;
    event.base_symbol = base_symbol;
    event.stage_data = "{\"phase\":\"start\"}";
    event.notes = "Strategy signal generation started";
    event.success = true;
    
    record_backend_stage(event);
}

void BackendAuditRecorder::record_signal_generation_complete(const std::string& chain_id, 
                                                           double signal_strength,
                                                           const std::string& reason, 
                                                           double compute_time_us, 
                                                           std::int64_t timestamp) {
    BackendAuditEvent event;
    event.timestamp = timestamp;
    event.run_id = current_run_id_;
    event.chain_id = chain_id;
    event.stage = BackendStage::STRATEGY_SIGNAL_GENERATION;
    event.strategy_name = ""; // Will be filled by caller
    event.base_symbol = ""; // Will be filled by caller
    
    json stage_data;
    stage_data["phase"] = "complete";
    stage_data["signal_strength"] = signal_strength;
    stage_data["reason"] = reason;
    event.stage_data = stage_data.dump();
    
    event.notes = "Strategy signal generation completed";
    event.processing_time_us = compute_time_us;
    event.success = true;
    
    record_backend_stage(event);
}

void BackendAuditRecorder::record_allocation_start(const std::string& chain_id, 
                                                   const std::string& strategy_name,
                                                   const std::vector<BaseStrategy::AllocationDecision>& input_decisions,
                                                   const std::map<std::string, double>& portfolio_state, 
                                                   std::int64_t timestamp) {
    BackendAuditEvent event;
    event.timestamp = timestamp;
    event.run_id = current_run_id_;
    event.chain_id = chain_id;
    event.stage = BackendStage::ALLOCATION_DECISION_MAKING;
    event.strategy_name = strategy_name;
    event.base_symbol = ""; // Multiple symbols
    
    json stage_data;
    stage_data["phase"] = "start";
    stage_data["input_decisions_count"] = input_decisions.size();
    stage_data["portfolio_instruments"] = portfolio_state.size();
    event.stage_data = stage_data.dump();
    
    event.notes = "Allocation decision making started";
    event.success = true;
    
    record_backend_stage(event);
}

void BackendAuditRecorder::record_allocation_complete(const std::string& chain_id,
                                                     const std::vector<BaseStrategy::AllocationDecision>& output_decisions,
                                                     const std::vector<std::string>& conflicts_resolved,
                                                     const std::string& reasoning, 
                                                     double process_time_us, 
                                                     std::int64_t timestamp) {
    BackendAuditEvent event;
    event.timestamp = timestamp;
    event.run_id = current_run_id_;
    event.chain_id = chain_id;
    event.stage = BackendStage::ALLOCATION_DECISION_MAKING;
    event.strategy_name = ""; // Will be filled by caller
    event.base_symbol = ""; // Multiple symbols
    
    json stage_data;
    stage_data["phase"] = "complete";
    stage_data["output_decisions_count"] = output_decisions.size();
    stage_data["conflicts_resolved_count"] = conflicts_resolved.size();
    stage_data["reasoning"] = reasoning;
    event.stage_data = stage_data.dump();
    
    event.notes = "Allocation decision making completed";
    event.processing_time_us = process_time_us;
    event.success = true;
    
    record_backend_stage(event);
}

void BackendAuditRecorder::record_execution_start(const std::string& chain_id, 
                                                  const std::string& strategy_name,
                                                  double portfolio_value_before, 
                                                  std::int64_t timestamp) {
    BackendAuditEvent event;
    event.timestamp = timestamp;
    event.run_id = current_run_id_;
    event.chain_id = chain_id;
    event.stage = BackendStage::EXECUTION_ROUTING;
    event.strategy_name = strategy_name;
    event.base_symbol = ""; // Multiple symbols
    
    json stage_data;
    stage_data["phase"] = "start";
    stage_data["portfolio_value_before"] = portfolio_value_before;
    event.stage_data = stage_data.dump();
    
    event.notes = "Backend execution started";
    event.success = true;
    
    record_backend_stage(event);
}

void BackendAuditRecorder::record_execution_complete(const std::string& chain_id, 
                                                    double portfolio_value_after,
                                                    double realized_pnl_delta, 
                                                    double unrealized_pnl_delta,
                                                    int orders_executed, 
                                                    int fills_completed,
                                                    double total_latency_us, 
                                                    std::int64_t timestamp) {
    BackendAuditEvent event;
    event.timestamp = timestamp;
    event.run_id = current_run_id_;
    event.chain_id = chain_id;
    event.stage = BackendStage::EXECUTION_ROUTING;
    event.strategy_name = ""; // Will be filled by caller
    event.base_symbol = ""; // Multiple symbols
    
    json stage_data;
    stage_data["phase"] = "complete";
    stage_data["portfolio_value_after"] = portfolio_value_after;
    stage_data["realized_pnl_delta"] = realized_pnl_delta;
    stage_data["unrealized_pnl_delta"] = unrealized_pnl_delta;
    stage_data["orders_executed"] = orders_executed;
    stage_data["fills_completed"] = fills_completed;
    event.stage_data = stage_data.dump();
    
    event.notes = "Backend execution completed";
    event.processing_time_us = total_latency_us;
    event.success = true;
    
    record_backend_stage(event);
}

// **HELPER METHODS FOR JSON SERIALIZATION**

std::string BackendAuditRecorder::serialize_allocation_decisions(const std::vector<BaseStrategy::AllocationDecision>& decisions) {
    json decisions_json = json::array();
    
    for (const auto& decision : decisions) {
        json decision_json;
        decision_json["instrument"] = decision.instrument;
        decision_json["target_weight"] = decision.target_weight;
        decision_json["confidence"] = decision.confidence;
        decision_json["reasoning"] = decision.reason;
        decisions_json.push_back(decision_json);
    }
    
    return decisions_json.dump();
}

std::string BackendAuditRecorder::serialize_portfolio_state(const std::map<std::string, double>& portfolio) {
    json portfolio_json;
    for (const auto& [instrument, position] : portfolio) {
        portfolio_json[instrument] = position;
    }
    return portfolio_json.dump();
}

std::string BackendAuditRecorder::serialize_stage_timings(const std::map<std::string, double>& timings) {
    json timings_json;
    for (const auto& [stage, time_us] : timings) {
        timings_json[stage] = time_us;
    }
    return timings_json.dump();
}

} // namespace sentio
