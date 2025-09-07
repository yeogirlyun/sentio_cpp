#include "sentio/telemetry_logger.hpp"
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace sentio {

// Global telemetry logger instance
std::unique_ptr<TelemetryLogger> g_telemetry_logger = nullptr;

TelemetryLogger::TelemetryLogger(const std::string& log_file_path) 
    : log_file_path_(log_file_path) {
    // Create directory if it doesn't exist
    std::filesystem::path path(log_file_path);
    std::filesystem::create_directories(path.parent_path());
    
    // Open log file in append mode
    log_file_.open(log_file_path, std::ios::app);
    if (!log_file_.is_open()) {
        throw std::runtime_error("Failed to open telemetry log file: " + log_file_path);
    }
}

TelemetryLogger::~TelemetryLogger() {
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

void TelemetryLogger::log(const TelemetryData& data) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    std::ostringstream json;
    json << "{";
    json << "\"timestamp\":\"" << data.timestamp << "\",";
    json << "\"strategy_name\":\"" << escape_json_string(data.strategy_name) << "\",";
    json << "\"instrument\":\"" << escape_json_string(data.instrument) << "\",";
    json << "\"bars_processed\":" << data.bars_processed << ",";
    json << "\"signals_generated\":" << data.signals_generated << ",";
    json << "\"buy_signals\":" << data.buy_signals << ",";
    json << "\"sell_signals\":" << data.sell_signals << ",";
    json << "\"hold_signals\":" << data.hold_signals << ",";
    json << "\"avg_confidence\":" << std::fixed << std::setprecision(6) << data.avg_confidence << ",";
    json << "\"ready_percentage\":" << std::fixed << std::setprecision(2) << data.ready_percentage << ",";
    json << "\"processing_time_ms\":" << std::fixed << std::setprecision(3) << data.processing_time_ms;
    
    if (!data.notes.empty()) {
        json << ",\"notes\":\"" << escape_json_string(data.notes) << "\"";
    }
    
    json << "}";
    
    write_json_line(json.str());
}

void TelemetryLogger::log_metric(
    const std::string& strategy_name,
    const std::string& metric_name,
    double value,
    const std::string& instrument
) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    std::ostringstream json;
    json << "{";
    json << "\"timestamp\":\"" << get_current_timestamp() << "\",";
    json << "\"strategy_name\":\"" << escape_json_string(strategy_name) << "\",";
    json << "\"instrument\":\"" << escape_json_string(instrument) << "\",";
    json << "\"metric_name\":\"" << escape_json_string(metric_name) << "\",";
    json << "\"value\":" << std::fixed << std::setprecision(6) << value;
    json << "}";
    
    write_json_line(json.str());
}

void TelemetryLogger::log_signal_stats(
    const std::string& strategy_name,
    const std::string& instrument,
    int signals_generated,
    int buy_signals,
    int sell_signals,
    int hold_signals,
    double avg_confidence
) {
    TelemetryData data;
    data.timestamp = get_current_timestamp();
    data.strategy_name = strategy_name;
    data.instrument = instrument;
    data.signals_generated = signals_generated;
    data.buy_signals = buy_signals;
    data.sell_signals = sell_signals;
    data.hold_signals = hold_signals;
    data.avg_confidence = avg_confidence;
    
    log(data);
}

void TelemetryLogger::log_performance(
    const std::string& strategy_name,
    const std::string& instrument,
    int bars_processed,
    double processing_time_ms,
    double ready_percentage
) {
    TelemetryData data;
    data.timestamp = get_current_timestamp();
    data.strategy_name = strategy_name;
    data.instrument = instrument;
    data.bars_processed = bars_processed;
    data.processing_time_ms = processing_time_ms;
    data.ready_percentage = ready_percentage;
    
    log(data);
}

void TelemetryLogger::flush() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (log_file_.is_open()) {
        log_file_.flush();
    }
}

std::string TelemetryLogger::get_current_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count() << "Z";
    return oss.str();
}

std::string TelemetryLogger::escape_json_string(const std::string& str) const {
    std::string escaped;
    escaped.reserve(str.length() + 10); // Reserve some extra space
    
    for (char c : str) {
        switch (c) {
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:   escaped += c; break;
        }
    }
    
    return escaped;
}

void TelemetryLogger::write_json_line(const std::string& json_line) {
    if (log_file_.is_open()) {
        log_file_ << json_line << std::endl;
        
        // Flush every 100 lines to ensure data is written
        if (++log_counter_ % 100 == 0) {
            log_file_.flush();
        }
    }
}

void init_telemetry_logger(const std::string& log_file_path) {
    g_telemetry_logger = std::make_unique<TelemetryLogger>(log_file_path);
}

TelemetryLogger& get_telemetry_logger() {
    if (!g_telemetry_logger) {
        init_telemetry_logger();
    }
    return *g_telemetry_logger;
}

} // namespace sentio
