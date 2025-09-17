#pragma once
#include "audit/audit_db.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/allocation_manager.hpp"
#include <string>
#include <vector>
#include <map>

namespace sentio {

// **ENHANCED AUDIT EVENTS FOR STRATEGY-AGNOSTIC BACKEND**

// Backend pipeline stage tracking
enum class BackendStage {
    STRATEGY_SIGNAL_GENERATION,
    ALLOCATION_DECISION_MAKING,
    CONFLICT_RESOLUTION,
    POSITION_SIZING,
    EXECUTION_ROUTING,
    PORTFOLIO_UPDATE
};

// Enhanced event types for backend operations
struct BackendAuditEvent {
    std::int64_t timestamp;
    std::string run_id;
    std::string chain_id;
    BackendStage stage;
    std::string strategy_name;
    std::string base_symbol;
    
    // Stage-specific data (JSON serialized)
    std::string stage_data;
    std::string notes;
    
    // Performance tracking
    double processing_time_us = 0.0;
    bool success = true;
    std::string error_message;
};

// Strategy signal with backend context
struct StrategySignalAuditEvent {
    std::int64_t timestamp;
    std::string run_id;
    std::string chain_id;
    std::string strategy_name;
    std::string base_symbol;
    
    // Original strategy signal
    double signal_strength;
    std::string signal_reason;
    
    // Backend processing context
    std::vector<BaseStrategy::AllocationDecision> raw_decisions;
    std::map<std::string, double> current_portfolio_state;
    
    // Backend modifications
    std::vector<BaseStrategy::AllocationDecision> processed_decisions;
    std::string backend_modifications;
    
    // Timing
    double strategy_compute_time_us = 0.0;
    double backend_process_time_us = 0.0;
};

// Allocation manager decision audit
struct AllocationDecisionAuditEvent {
    std::int64_t timestamp;
    std::string run_id;
    std::string chain_id;
    std::string strategy_name;
    
    // Input state
    std::map<std::string, double> input_portfolio;
    std::vector<BaseStrategy::AllocationDecision> input_decisions;
    
    // Allocation manager analysis
    std::string decision_algorithm;
    std::map<std::string, double> conflict_analysis;
    std::vector<std::string> conflicts_detected;
    std::vector<std::string> conflicts_resolved;
    
    // Output decisions
    std::vector<BaseStrategy::AllocationDecision> output_decisions;
    std::string resolution_reasoning;
    
    // Performance metrics
    double processing_time_us = 0.0;
    int decisions_modified = 0;
    int conflicts_resolved_count = 0;
};

// Backend execution summary
struct BackendExecutionSummaryEvent {
    std::int64_t timestamp;
    std::string run_id;
    std::string chain_id;
    std::string strategy_name;
    
    // Pipeline performance
    double total_processing_time_us = 0.0;
    std::map<std::string, double> stage_timings;
    
    // Execution results
    int signals_processed = 0;
    int decisions_generated = 0;
    int conflicts_resolved = 0;
    int orders_executed = 0;
    int fills_completed = 0;
    
    // Portfolio impact
    double portfolio_value_before = 0.0;
    double portfolio_value_after = 0.0;
    double realized_pnl_delta = 0.0;
    double unrealized_pnl_delta = 0.0;
    
    // Backend efficiency metrics
    double signal_to_execution_latency_us = 0.0;
    double execution_success_rate = 0.0;
    std::string performance_notes;
};

// **BACKEND AUDIT RECORDER**: Enhanced audit recording for backend operations
class BackendAuditRecorder {
private:
    audit::DB* audit_db_;
    std::string current_run_id_;
    bool logging_enabled_;
    
public:
    BackendAuditRecorder(audit::DB* db, const std::string& run_id, bool enabled = true);
    
    // Backend pipeline events
    void record_backend_stage(const BackendAuditEvent& event);
    void record_strategy_signal_processed(const StrategySignalAuditEvent& event);
    void record_allocation_decision(const AllocationDecisionAuditEvent& event);
    void record_backend_execution_summary(const BackendExecutionSummaryEvent& event);
    
    // Convenience methods for common backend operations
    void record_signal_generation_start(const std::string& chain_id, const std::string& strategy_name, 
                                       const std::string& base_symbol, std::int64_t timestamp);
    void record_signal_generation_complete(const std::string& chain_id, double signal_strength, 
                                          const std::string& reason, double compute_time_us, std::int64_t timestamp);
    
    void record_allocation_start(const std::string& chain_id, const std::string& strategy_name,
                                const std::vector<BaseStrategy::AllocationDecision>& input_decisions, 
                                const std::map<std::string, double>& portfolio_state, std::int64_t timestamp);
    void record_allocation_complete(const std::string& chain_id, 
                                   const std::vector<BaseStrategy::AllocationDecision>& output_decisions,
                                   const std::vector<std::string>& conflicts_resolved,
                                   const std::string& reasoning, double process_time_us, std::int64_t timestamp);
    
    void record_execution_start(const std::string& chain_id, const std::string& strategy_name,
                               double portfolio_value_before, std::int64_t timestamp);
    void record_execution_complete(const std::string& chain_id, double portfolio_value_after,
                                  double realized_pnl_delta, double unrealized_pnl_delta,
                                  int orders_executed, int fills_completed, 
                                  double total_latency_us, std::int64_t timestamp);
    
    // State and configuration
    void set_logging_enabled(bool enabled) { logging_enabled_ = enabled; }
    bool is_logging_enabled() const { return logging_enabled_; }
    std::string get_current_run_id() const { return current_run_id_; }
    
private:
    // Helper methods for JSON serialization
    std::string serialize_allocation_decisions(const std::vector<BaseStrategy::AllocationDecision>& decisions);
    std::string serialize_portfolio_state(const std::map<std::string, double>& portfolio);
    std::string serialize_stage_timings(const std::map<std::string, double>& timings);
};

// **BACKEND AUDIT ANALYZER**: Query and analyze backend audit data
class BackendAuditAnalyzer {
private:
    audit::DB* audit_db_;
    
public:
    explicit BackendAuditAnalyzer(audit::DB* db) : audit_db_(db) {}
    
    // Backend performance analysis
    struct BackendPerformanceReport {
        std::string run_id;
        std::string strategy_name;
        
        // Timing analysis
        double avg_signal_generation_time_us = 0.0;
        double avg_allocation_decision_time_us = 0.0;
        double avg_execution_time_us = 0.0;
        double avg_total_pipeline_time_us = 0.0;
        
        // Efficiency metrics
        double signal_to_execution_success_rate = 0.0;
        int total_conflicts_detected = 0;
        int total_conflicts_resolved = 0;
        double conflict_resolution_rate = 0.0;
        
        // Portfolio impact
        double total_realized_pnl_from_backend = 0.0;
        double total_unrealized_pnl_from_backend = 0.0;
        int total_backend_executions = 0;
        
        // Backend vs Strategy comparison
        double backend_efficiency_score = 0.0; // 0-100 scale
        std::string performance_summary;
    };
    
    BackendPerformanceReport analyze_backend_performance(const std::string& run_id);
    
    // Signal flow analysis through backend
    struct SignalFlowReport {
        std::string run_id;
        std::string strategy_name;
        
        int signals_generated = 0;
        int signals_processed_by_backend = 0;
        int allocation_decisions_made = 0;
        int conflicts_detected = 0;
        int conflicts_resolved = 0;
        int orders_executed = 0;
        int fills_completed = 0;
        
        // Flow efficiency
        double signal_to_order_conversion_rate = 0.0;
        double order_to_fill_success_rate = 0.0;
        double end_to_end_success_rate = 0.0;
        
        std::vector<std::string> bottlenecks_identified;
        std::string flow_summary;
    };
    
    SignalFlowReport analyze_signal_flow(const std::string& run_id);
    
    // Conflict resolution analysis
    struct ConflictResolutionReport {
        std::string run_id;
        std::string strategy_name;
        
        std::map<std::string, int> conflict_types_detected;
        std::map<std::string, int> resolution_methods_used;
        std::map<std::string, double> resolution_success_rates;
        
        double avg_resolution_time_us = 0.0;
        double portfolio_impact_from_conflicts = 0.0;
        
        std::vector<std::string> most_common_conflicts;
        std::vector<std::string> resolution_recommendations;
        std::string conflict_summary;
    };
    
    ConflictResolutionReport analyze_conflict_resolution(const std::string& run_id);
};

} // namespace sentio
