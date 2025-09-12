#include <cassert>
#include <vector>
#include "sentio/rsi_strategy.hpp"
#include "sentio/rsi_prob.hpp"

using namespace sentio;

static std::vector<Bar> synth(std::int64_t t0, int n, double step=1.0) {
    std::vector<Bar> v; v.reserve(n);
    double px=100.0;
    for (int i=0;i<n;++i) {
        double prev=px;
        px += (i%2==0 ? -step : +step); // oscillate to traverse RSI range
        Bar bar;
        bar.ts_utc_epoch = t0 + i*60000;
        bar.open = prev;
        bar.high = std::max(prev, px);
        bar.low = std::min(prev, px);
        bar.close = px;
        bar.volume = 1000;
        v.push_back(bar);
    }
    return v;
}

int main() {
    // Mapping exactness
    assert(std::abs(rsi_to_prob(30.0) - 0.8) < 1e-12);
    assert(std::abs(rsi_to_prob(50.0) - 0.5) < 1e-12);
    assert(std::abs(rsi_to_prob(70.0) - 0.2) < 1e-12);

    RSIStrategy s;
    s.set_params({{"rsi_period",14},{"epsilon",0.0},{"alpha",1.0}});
    s.apply_params();

    auto warm = synth(0, 14);
    long c=0; 
    for (int i = 0; i < static_cast<int>(warm.size()); ++i) {
        double prob = s.calculate_probability(warm, i);
        if (prob != 0.5) ++c; // Count non-neutral signals
    }
    assert(c==0); // no signals during warmup

    auto run = synth(14*60000, 400, 2.0);
    long signals=0;
    double min_prob=1.0, max_prob=0.0;
    for (int i = 0; i < static_cast<int>(run.size()); ++i) {
        double prob = s.calculate_probability(run, i);
        if (prob != 0.5) {
            ++signals;
            min_prob = std::min(min_prob, prob);
            max_prob = std::max(max_prob, prob);
        }
        assert(prob >= 0.0 && prob <= 1.0);
    }
    assert(signals > 0);
    return 0;
}
