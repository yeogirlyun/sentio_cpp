#include "sentio/ml/onnx_model.hpp"
#include <stdexcept>
#include <algorithm>

#ifdef SENTIO_WITH_ONNXRUNTIME
  #include <onnxruntime_cxx_api.h>
#endif

namespace sentio::ml {

OnnxModel::OnnxModel(ModelSpec spec) : spec_(std::move(spec)) {}

#ifdef SENTIO_WITH_ONNXRUNTIME
struct OnnxModel::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "sentio"};
  Ort::Session session{nullptr};
  Ort::MemoryInfo mem{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::vector<int64_t> input_shape; // [1, N] or [N]
  std::vector<const char*> input_names, output_names;

  Impl(const std::string& path, int intra, int inter) {
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(intra);
    opt.SetInterOpNumThreads(inter);
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = Ort::Session(env, path.c_str(), opt);

    size_t n_in = session.GetInputCount();
    size_t n_out= session.GetOutputCount();
    input_names.reserve(n_in);
    output_names.reserve(n_out);
    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i=0;i<n_in;++i){
      auto name = session.GetInputNameAllocated(i, alloc);
      input_names.push_back(strdup(name.get()));
      auto info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
      input_shape = info.GetShape();
    }
    for (size_t i=0;i<n_out;++i){
      auto name = session.GetOutputNameAllocated(i, alloc);
      output_names.push_back(strdup(name.get()));
    }
  }

  ~Impl(){
    for (auto p: input_names) free(const_cast<char*>(p));
    for (auto p: output_names) free(const_cast<char*>(p));
  }
};
#endif

std::unique_ptr<OnnxModel> OnnxModel::load(const std::string& path, const ModelSpec& spec,
                                           int intra, int inter)
{
  auto m = std::unique_ptr<OnnxModel>(new OnnxModel(spec));
#ifdef SENTIO_WITH_ONNXRUNTIME
  m->impl_ = std::make_unique<Impl>(path, intra, inter);
#else
  (void)path; (void)intra; (void)inter;
  // Build without ONNX Runtime -> inference unavailable at runtime.
#endif
  return m;
}

std::optional<ModelOutput> OnnxModel::predict(const std::vector<float>& feats) const {
  if (feats.size() != spec_.feature_names.size()) return std::nullopt;

#ifndef SENTIO_WITH_ONNXRUNTIME
  // Fallback (useful in unit tests without ORT): return HOLD with 0.5 confidence
  ModelOutput out; out.probs = {0.25f, 0.5f, 0.25f}; out.score = 0.0f;
  return out;
#else
  auto& s = *impl_;
  // ONNX expects contiguous float tensor, shape [1, N]
  std::vector<int64_t> shape{1, (int64_t)feats.size()};
  Ort::Value input = Ort::Value::CreateTensor<float>(s.mem, const_cast<float*>(feats.data()),
                                                     feats.size(), shape.data(), shape.size());
  auto outv = s.session.Run(Ort::RunOptions{nullptr},
                            s.input_names.data(), &input, 1,
                            s.output_names.data(), s.output_names.size());
  // Assume single output: logits/probs [1, K]
  if (outv.empty() || !outv.front().IsTensor()) return std::nullopt;
  float* p = outv.front().GetTensorMutableData<float>();
  size_t K = outv.front().GetTensorTypeAndShapeInfo().GetElementCount();
  ModelOutput out; out.probs.resize(K);
  // If logits, you may apply softmax; assume probs here:
  float sum=0.f; for (size_t i=0;i<K;++i){ out.probs[i]=p[i]; sum+=out.probs[i]; }
  if (sum>0){ for (auto& v: out.probs) v/=sum; }
  return out;
#endif
}

} // namespace sentio::ml
