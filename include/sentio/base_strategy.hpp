#pragma once
#include "core.hpp"
#include "signal_diag.hpp"
#include "router.hpp"  // for StrategySignal
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <functional>

namespace sentio {

// Forward declarations
struct RouterCfg;
struct SizerCfg;

// Strategy context for bar processing
struct StrategyCtx {
  std::string instrument;     // traded instrument for this stream
  std::int64_t ts_utc_epoch;  // bar timestamp (UTC seconds)
  bool is_rth{true};          // inject from your RTH checker
};

// Minimal strategy interface for ML integration
class IStrategy {
public:
  virtual ~IStrategy() = default;
  virtual void on_bar(const StrategyCtx& ctx, const Bar& b) = 0;
  virtual std::optional<StrategySignal> latest() const = 0;
};

// Parameter types and enums
enum class ParamType { INT, FLOAT };
struct ParamSpec { 
    ParamType type;
    double min_val, max_val;
    double default_val;
};

using ParameterMap = std::unordered_map<std::string, double>;
using ParameterSpace = std::unordered_map<std::string, ParamSpec>;

enum class SignalType { NONE = 0, BUY = 1, SELL = -1, STRONG_BUY = 2, STRONG_SELL = -2 };

struct StrategyState {
    bool in_position = false;
    SignalType last_signal = SignalType::NONE;
    int last_trade_bar = -1000; // Initialize far in the past
    
    void reset() {
        in_position = false;
        last_signal = SignalType::NONE;
        last_trade_bar = -1000;
    }
};

// StrategySignal is now defined in router.hpp

class BaseStrategy {
protected:
    std::string name_;
    ParameterMap params_;
    StrategyState state_;
    SignalDiag diag_;

    bool is_cooldown_active(int current_bar, int cooldown_period) const;
    
public:
    BaseStrategy(const std::string& name) : name_(name) {}
    virtual ~BaseStrategy() = default;

    // **MODIFIED**: Explicitly delete copy and move operations for this polymorphic base class.
    // This prevents object slicing and ownership confusion.
    BaseStrategy(const BaseStrategy&) = delete;
    BaseStrategy& operator=(const BaseStrategy&) = delete;
    BaseStrategy(BaseStrategy&&) = delete;
    BaseStrategy& operator=(BaseStrategy&&) = delete;
    
    std::string get_name() const { return name_; }
    
    virtual ParameterMap get_default_params() const = 0;
    virtual ParameterSpace get_param_space() const = 0;
    virtual void apply_params() = 0;
    
    void set_params(const ParameterMap& params);
    
    // **NEW**: Primary method - strategies should implement this to return probability (0-1)
    virtual double calculate_probability(const std::vector<Bar>& bars, int current_index) = 0;
    
    // **NEW**: Wrapper method that converts probability to StrategySignal
    StrategySignal calculate_signal(const std::vector<Bar>& bars, int current_index) {
        double prob = calculate_probability(bars, current_index);
        return StrategySignal::from_probability(prob);
    }
    
    // **NEW**: Strategy-agnostic allocation interface
    struct AllocationDecision {
        std::string instrument;
        double target_weight; // -1.0 to 1.0
        double confidence;    // 0.0 to 1.0
        std::string reason;   // Human-readable reason for allocation
    };
    
    // **NEW**: Get allocation decisions for this strategy
    virtual std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, 
        int current_index,
        const std::string& base_symbol,
        const std::string& bull3x_symbol,
        const std::string& bear3x_symbol,
        const std::string& bear1x_symbol) = 0;
    
    // **NEW**: Get strategy-specific router configuration
    virtual RouterCfg get_router_config() const = 0;
    
    // **NEW**: Get strategy-specific sizer configuration  
    virtual SizerCfg get_sizer_config() const = 0;
    
    // **NEW**: Check if strategy requires special handling (e.g., dynamic leverage)
    virtual bool requires_dynamic_allocation() const { return false; }
    
    // **NEW**: Get strategy-specific signal processing (for audit/logging)
    virtual std::string get_signal_description(double probability) const {
        if (probability > 0.8) return "STRONG_BUY";
        if (probability > 0.6) return "BUY";
        if (probability < 0.2) return "STRONG_SELL";
        if (probability < 0.4) return "SELL";
        return "HOLD";
    }
    virtual void reset_state();
    
    const SignalDiag& get_diag() const { return diag_; }
};

class StrategyFactory {
public:
    using CreateFunction = std::function<std::unique_ptr<BaseStrategy>()>;
    
    static StrategyFactory& instance();
    void register_strategy(const std::string& name, CreateFunction create_func);
    std::unique_ptr<BaseStrategy> create_strategy(const std::string& name);
    std::vector<std::string> get_available_strategies() const;
    
private:
    std::unordered_map<std::string, CreateFunction> strategies_;
};

// **NEW**: The final, more robust registration macro.
// It takes the C++ ClassName and the "Name" to be used by the CLI.
#define REGISTER_STRATEGY(ClassName, Name) \
    namespace { \
        struct ClassName##Registrar { \
            ClassName##Registrar() { \
                StrategyFactory::instance().register_strategy(Name, \
                    []() { return std::make_unique<ClassName>(); }); \
            } \
        }; \
        static ClassName##Registrar g_##ClassName##_registrar; \
    }

} // namespace sentio