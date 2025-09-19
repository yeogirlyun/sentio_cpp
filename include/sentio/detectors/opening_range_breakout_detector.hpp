#pragma once
#include "sentio/signal_utils.hpp"
#include "sentio/rules/opening_range_breakout_rule.hpp"

namespace sentio::detectors {

class OpeningRangeBreakoutDetector final : public IDetector {
public:
    OpeningRangeBreakoutDetector() = default;
    std::string_view name() const override { return "OPENING_RANGE_BRK"; }
    int warmup_period() const override { return rule_.warmup(); }
    DetectorResult score(const std::vector<Bar>& bars, int idx) override {
        rules::BarsView v{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0}; to_view(bars, v);
        auto out = rule_.eval(v, idx);
        if (!out) return {0.5, 0, name()};
        int dir = out->signal.value_or(0);
        double p = (dir>0)? 0.8 : (dir<0? 0.2 : 0.5);
        return {p, dir, name()};
    }
private:
    rules::OpeningRangeBreakoutRule rule_;
    static void to_view(const std::vector<Bar>& b, rules::BarsView& v){
        static std::vector<int64_t> ts; static std::vector<double> open,high,low,close,vol;
        size_t N=b.size(); ts.resize(N); open.resize(N); high.resize(N); low.resize(N); close.resize(N); vol.resize(N);
        for(size_t i=0;i<N;i++){ ts[i]=b[i].ts_utc_epoch; open[i]=b[i].open; high[i]=b[i].high; low[i]=b[i].low; close[i]=b[i].close; vol[i]=b[i].volume; }
        v.ts=ts.data(); v.open=open.data(); v.high=high.data(); v.low=low.data(); v.close=close.data(); v.volume=vol.data(); v.n=(int64_t)N;
    }
};

} // namespace sentio::detectors


