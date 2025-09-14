#pragma once

namespace sentio {

// Global configuration for leverage pricing system
class GlobalLeverageConfig {
private:
    static bool use_theoretical_leverage_pricing_;
    
public:
    // Enable theoretical leverage pricing globally (default: true)
    static void enable_theoretical_leverage_pricing(bool enable = true) {
        use_theoretical_leverage_pricing_ = enable;
    }
    
    static bool is_theoretical_leverage_pricing_enabled() {
        return use_theoretical_leverage_pricing_;
    }
};

} // namespace sentio
