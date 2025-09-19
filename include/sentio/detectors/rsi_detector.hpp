#pragma once
#include "sentio/signal_utils.hpp"
#include "sentio/rsi_prob.hpp"

namespace sentio::detectors {

class RsiDetector final : public IDetector {
public:
    explicit RsiDetector(int period = 14, double alpha = 1.0)
        : period_(period), alpha_(alpha) {}

    std::string_view name() const override { return "RSI_REVERSION"; }
    int warmup_period() const override { return period_ + 1; }

    DetectorResult score(const std::vector<Bar>& bars, int idx) override {
        if (idx < warmup_period()) return {0.5, 0, name()};
        double rsi = calculate_rsi(bars, idx);
        double probability = rsi_to_prob_tuned(rsi, alpha_);
        int direction = (probability > 0.55) ? 1 : (probability < 0.45 ? -1 : 0);
        return {probability, direction, name()};
    }

private:
    int period_;
    double alpha_;

    double calculate_rsi(const std::vector<Bar>& bars, int end_idx) {
        double avg_gain = 0.0, avg_loss = 0.0;
        int start_idx = end_idx - period_;
        for (int i = start_idx + 1; i <= end_idx; ++i) {
            double change = bars[i].close - bars[i - 1].close;
            if (change > 0) avg_gain += change; else avg_loss -= change;
        }
        avg_gain /= period_;
        avg_loss /= period_;
        if (avg_loss < 1e-9) return 100.0;
        double rs = avg_gain / avg_loss;
        return 100.0 - (100.0 / (1.0 + rs));
    }
};

} // namespace sentio::detectors


