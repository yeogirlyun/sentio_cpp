#pragma once
#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <unordered_map>
#include <atomic>

namespace sentio {

/**
 * JSON line logger for telemetry data
 * Thread-safe logging of strategy performance metrics
 */
class TelemetryLogger {
public:
    struct TelemetryData {
        std::string timestamp;
        std::string strategy_name;
        std::string instrument;
        int bars_processed{0};
        int signals_generated{0};
        int buy_signals{0};
        int sell_signals{0};
        int hold_signals{0};
        double avg_confidence{0.0};
        double ready_percentage{0.0};
        double processing_time_ms{0.0};
        std::string notes;
    };

    explicit TelemetryLogger(const std::string& log_file_path);
    ~TelemetryLogger();

    /**
     * Log telemetry data for a strategy
     * @param data Telemetry data to log
     */
    void log(const TelemetryData& data);

    /**
     * Log a simple metric
     * @param strategy_name Strategy name
     * @param metric_name Metric name
     * @param value Metric value
     * @param instrument Optional instrument name
     */
    void log_metric(
        const std::string& strategy_name,
        const std::string& metric_name,
        double value,
        const std::string& instrument = ""
    );

    /**
     * Log signal generation statistics
     * @param strategy_name Strategy name
     * @param instrument Instrument name
     * @param signals_generated Total signals generated
     * @param buy_signals Buy signals
     * @param sell_signals Sell signals
     * @param hold_signals Hold signals
     * @param avg_confidence Average confidence
     */
    void log_signal_stats(
        const std::string& strategy_name,
        const std::string& instrument,
        int signals_generated,
        int buy_signals,
        int sell_signals,
        int hold_signals,
        double avg_confidence
    );

    /**
     * Log performance metrics
     * @param strategy_name Strategy name
     * @param instrument Instrument name
     * @param bars_processed Number of bars processed
     * @param processing_time_ms Processing time in milliseconds
     * @param ready_percentage Percentage of time strategy was ready
     */
    void log_performance(
        const std::string& strategy_name,
        const std::string& instrument,
        int bars_processed,
        double processing_time_ms,
        double ready_percentage
    );

    /**
     * Flush any pending log data
     */
    void flush();

    /**
     * Get current log file path
     */
    const std::string& get_log_file_path() const { return log_file_path_; }

private:
    std::string log_file_path_;
    std::ofstream log_file_;
    std::mutex log_mutex_;
    std::atomic<int> log_counter_{0};
    
    // Helper methods
    std::string get_current_timestamp() const;
    std::string escape_json_string(const std::string& str) const;
    void write_json_line(const std::string& json_line);
};

/**
 * Global telemetry logger instance
 * Use this for easy access throughout the application
 */
extern std::unique_ptr<TelemetryLogger> g_telemetry_logger;

/**
 * Initialize global telemetry logger
 * @param log_file_path Path to log file
 */
void init_telemetry_logger(const std::string& log_file_path = "logs/telemetry.jsonl");

/**
 * Get global telemetry logger instance
 * @return Reference to global telemetry logger
 */
TelemetryLogger& get_telemetry_logger();

} // namespace sentio
