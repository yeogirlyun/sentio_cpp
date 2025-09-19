#pragma once
#include "sentio/signal_utils.hpp"
#include "sentio/rules/momentum_volume_rule.hpp"

namespace sentio::detectors {

class MomentumVolumeDetector final : public IDetector {
public:
    MomentumVolumeDetector() = default;
    std::string_view name() const override { return "MOMENTUM_VOLUME"; }
    int warmup_period() const override { return rule_.warmup(); }
    DetectorResult score(const std::vector<Bar>& bars, int idx) override {
        rules::BarsView v{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0};
        to_view(bars, v);
        auto out = rule_.eval(v, idx);
        if (!out) return {0.5, 0, name()};
        double p = out->p_up ? static_cast<double>(*out->p_up) : 0.5 + 0.5 * (out->signal.value_or(0));
        int dir = out->signal.value_or(0);
        return {p, dir, name()};
    }
private:
    rules::MomentumVolumeRule rule_;
    static void to_view(const std::vector<Bar>& b, rules::BarsView& v){
        // This quick adapter uses contiguous vectors to temporary buffers
        static std::vector<int64_t> ts; static std::vector<double> open,high,low,close,vol;
        size_t N=b.size(); ts.resize(N); open.resize(N); high.resize(N); low.resize(N); close.resize(N); vol.resize(N);
        for(size_t i=0;i<N;i++){ ts[i]=b[i].ts_utc_epoch; open[i]=b[i].open; high[i]=b[i].high; low[i]=b[i].low; close[i]=b[i].close; vol[i]=b[i].volume; }
        v.ts=ts.data(); v.open=open.data(); v.high=high.data(); v.low=low.data(); v.close=close.data(); v.volume=vol.data(); v.n=(int64_t)N;
    }
};

} // namespace sentio::detectors


