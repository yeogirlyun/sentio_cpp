#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "sentio/core.hpp"
#include "sentio/feature/feature_from_spec.hpp"
#include "sentio/feature_engineering/kochi_features.hpp"

namespace py = pybind11;

static std::vector<sentio::Bar> bars_from_numpy(
    py::array_t<long long> ts,
    py::array_t<double> open,
    py::array_t<double> high,
    py::array_t<double> low,
    py::array_t<double> close,
    py::array_t<double> volume)
{
  auto rts = ts.unchecked<1>();
  auto ro = open.unchecked<1>();
  auto rh = high.unchecked<1>();
  auto rl = low.unchecked<1>();
  auto rc = close.unchecked<1>();
  auto rv = volume.unchecked<1>();
  ssize_t N = ts.shape(0);
  std::vector<sentio::Bar> v; v.resize(N);
  for (ssize_t i=0;i<N;++i){
    v[i].ts_utc = ""; // optional
    v[i].ts_utc_epoch = rts(i);
    v[i].open = ro(i);
    v[i].high = rh(i);
    v[i].low  = rl(i);
    v[i].close= rc(i);
    v[i].volume = (uint64_t)rv(i);
  }
  return v;
}

PYBIND11_MODULE(sentio_features, m) {
  m.doc() = "Sentio feature builders (TFA + Kochi)";

  m.def("build_features_from_spec", [](const std::string& symbol,
                                        py::array_t<long long> ts,
                                        py::array_t<double> open,
                                        py::array_t<double> high,
                                        py::array_t<double> low,
                                        py::array_t<double> close,
                                        py::array_t<double> volume,
                                        const std::string& spec_json){
    auto bars = bars_from_numpy(ts, open, high, low, close, volume);
    auto M = sentio::build_features_from_spec_json(symbol, bars, spec_json);
    py::array_t<float> out({M.rows, (long long)M.cols});
    auto r = out.mutable_unchecked<2>();
    for (long long i=0;i<M.rows;++i){
      for (int j=0;j<M.cols;++j){ r(i,j) = M.data[i*M.cols + j]; }
    }
    return out;
  },
  py::arg("symbol"), py::arg("ts"), py::arg("open"), py::arg("high"), py::arg("low"), py::arg("close"), py::arg("volume"), py::arg("spec_json"));

  m.def("kochi_feature_names", [](){
    return sentio::feature_engineering::kochi_feature_names();
  });

  m.def("build_kochi_features", [](py::array_t<long long> ts,
                                    py::array_t<double> open,
                                    py::array_t<double> high,
                                    py::array_t<double> low,
                                    py::array_t<double> close,
                                    py::array_t<double> volume){
    auto bars = bars_from_numpy(ts, open, high, low, close, volume);
    std::vector<std::vector<double>> rows; rows.reserve(bars.size());
    for (int i=0;i<(int)bars.size();++i){
      auto f = sentio::feature_engineering::calculate_kochi_features(bars, i);
      rows.push_back(std::move(f));
    }
    // pack into numpy
    int64_t N = (int64_t)rows.size();
    int64_t F = rows.empty()? 0 : (int64_t)rows[0].size();
    py::array_t<float> out({N, F});
    auto r = out.mutable_unchecked<2>();
    for (int64_t i=0;i<N;++i){ for (int64_t j=0;j<F;++j){ r(i,j) = (float)rows[i][j]; } }
    return out;
  }, py::arg("ts"), py::arg("open"), py::arg("high"), py::arg("low"), py::arg("close"), py::arg("volume"));
}
