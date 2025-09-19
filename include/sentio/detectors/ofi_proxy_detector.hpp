#pragma once
#include "sentio/signal_utils.hpp"
#include "sentio/rules/ofi_proxy_rule.hpp"

namespace sentio::detectors {

class OFIProxyDetector final : public IDetector {
public:
    OFIProxyDetector() = default;
    std::string_view name() const override { return "OFI_PROXY"; }
    int warmup_period() const override { return rule_.warmup(); }
    DetectorResult score(const std::vector<Bar>& bars, int idx) override {
        rules::BarsView v{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0}; to_view(bars, v);
        auto out = rule_.eval(v, idx);
        if (!out) return {0.5, 0, name()};
        double p = out->p_up.value_or(0.5);
        int dir = out->signal.value_or( (p>0.55) ? 1 : (p<0.45 ? -1 : 0) );
        return {p, dir, name()};
    }
private:
    rules::OFIProxyRule rule_;
    static void to_view(const std::vector<Bar>& b, rules::BarsView& v){
        static std::vector<int64_t> ts; static std::vector<double> open,high,low,close,vol;
        size_t N=b.size(); ts.resize(N); open.resize(N); high.resize(N); low.resize(N); close.resize(N); vol.resize(N);
        for(size_t i=0;i<N;i++){ ts[i]=b[i].ts_utc_epoch; open[i]=b[i].open; high[i]=b[i].high; low[i]=b[i].low; close[i]=b[i].close; vol[i]=b[i].volume; }
        v.ts=ts.data(); v.open=open.data(); v.high=high.data(); v.low=low.data(); v.close=close.data(); v.volume=vol.data(); v.n=(int64_t)N;
    }
};

} // namespace sentio::detectors


