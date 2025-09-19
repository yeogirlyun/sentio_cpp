#pragma once
#include "sentio/signal_utils.hpp"
#include "sentio/bollinger.hpp"
#include <algorithm>
#include <vector>

namespace sentio::detectors {

class BollingerDetector final : public IDetector {
private:
    enum class State { Idle, Squeezed, ArmedLong, ArmedShort };
    State state_ = State::Idle;

    int bb_window_;
    int squeeze_lookback_;
    double squeeze_percentile_;
    int min_squeeze_bars_;

    Bollinger boll_
;
    std::vector<double> sd_history_;
    int squeeze_duration_ = 0;

public:
    explicit BollingerDetector(int bb_win = 20, int sqz_lookback = 60, double sqz_pct = 0.25, int min_sqz_bars = 3)
        : bb_window_(bb_win), squeeze_lookback_(sqz_lookback), squeeze_percentile_(sqz_pct), min_squeeze_bars_(min_sqz_bars),
          boll_(bb_win, 2.0) {}

    std::string_view name() const override { return "BOLLINGER_SQZ_BREAKOUT"; }
    int warmup_period() const override { return std::max(bb_window_, squeeze_lookback_) + 1; }

    DetectorResult score(const std::vector<Bar>& bars, int idx) override {
        const auto& bar = bars[idx];
        double mid, lo, hi, sd;
        boll_.step(bar.close, mid, lo, hi, sd);

        if (sd_history_.size() >= static_cast<size_t>(squeeze_lookback_)) sd_history_.erase(sd_history_.begin());
        sd_history_.push_back(sd);

        update_state_machine(bar, mid, lo, hi, sd);

        double probability = 0.5; int direction = 0;
        if (state_ == State::ArmedLong) { probability = 0.85; direction = 1; state_ = State::Idle; }
        else if (state_ == State::ArmedShort) { probability = 0.15; direction = -1; state_ = State::Idle; }
        return {probability, direction, name()};
    }

private:
    void update_state_machine(const Bar& bar, double mid, double lo, double hi, double sd) {
        if (sd_history_.size() < static_cast<size_t>(squeeze_lookback_)) return;
        auto sds = sd_history_;
        std::sort(sds.begin(), sds.end());
        double sd_threshold = sds[static_cast<size_t>(sds.size() * squeeze_percentile_)];
        bool squeezed = (sd <= sd_threshold);

        if (state_ == State::Idle && squeezed) {
            state_ = State::Squeezed; squeeze_duration_ = 1;
        } else if (state_ == State::Squeezed) {
            if (!squeezed) state_ = State::Idle;
            else if (squeeze_duration_ >= min_squeeze_bars_) {
                if (bar.close > hi) state_ = State::ArmedLong;
                else if (bar.close < lo) state_ = State::ArmedShort;
            }
            squeeze_duration_++;
        }
    }
};

} // namespace sentio::detectors


