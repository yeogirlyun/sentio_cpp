#include "sentio/ml/ts_model.hpp"
#include <torch/script.h>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace sentio::ml {

TorchScriptModel::TorchScriptModel(ModelSpec spec) : spec_(std::move(spec)), input_tensor_(nullptr) {}

TorchScriptModel::~TorchScriptModel() {
  if (input_tensor_) {
    delete static_cast<torch::Tensor*>(input_tensor_);
  }
}

std::unique_ptr<TorchScriptModel> TorchScriptModel::load(const std::string& pt_path,
                                                         const ModelSpec& spec,
                                                         [[maybe_unused]] bool use_cuda)
{
  auto m = std::unique_ptr<TorchScriptModel>(new TorchScriptModel(spec));
  torch::jit::script::Module mod = torch::jit::load(pt_path);
  mod.eval();
  m->mod_ = std::make_shared<torch::jit::script::Module>(std::move(mod));
  // Disable CUDA for now - CPU-only build
  m->cuda_ = false; // use_cuda && torch::cuda::is_available();
  if (m->cuda_) m->mod_->to(torch::kCUDA);
  // Defer concrete shape until first predict (we need T,F,layout)
  m->in_shape_.clear();
  return m;
}

std::optional<ModelOutput> TorchScriptModel::predict(const std::vector<float>& feats,
                                                     int T, int F, const std::string& layout) const
{
  if (T<=0 || F<=0) return std::nullopt;
  const size_t need = (layout=="BF")? feats.size() : size_t(T)*size_t(F);
  if (feats.size() != need) return std::nullopt;

  torch::NoGradGuard ng; torch::InferenceMode im;

  // Pre-allocate input tensor if needed (persistent tensor approach)
  std::vector<int64_t> need_shape = (layout=="BTF") ? std::vector<int64_t>{1,T,F}
                                                    : std::vector<int64_t>{1,(int64_t)feats.size()};
  
  if (!input_tensor_ || in_shape_ != need_shape) {
    in_shape_ = need_shape;
    if (input_tensor_) {
      delete static_cast<torch::Tensor*>(input_tensor_);
    }
    input_tensor_ = new torch::Tensor(torch::empty(in_shape_, torch::TensorOptions().dtype(torch::kFloat32).device(cuda_?torch::kCUDA:torch::kCPU)));
  }

  // memcpy into persistent tensor (no allocation, no clone)
  torch::Tensor& x = *static_cast<torch::Tensor*>(input_tensor_);
  if (cuda_) {
    // For CUDA, copy via CPU tensor to avoid sync issues
    torch::Tensor host = torch::from_blob((void*)feats.data(), {(int64_t)feats.size()}, torch::kFloat32);
    x.view({-1}).copy_(host);
  } else {
    // For CPU, direct memcpy into tensor data
    std::memcpy(x.data_ptr<float>(), feats.data(), feats.size() * sizeof(float));
  }

  std::vector<torch::jit::IValue> inputs; inputs.emplace_back(x);
  torch::Tensor out = mod_->forward(inputs).toTensor().to(torch::kCPU).contiguous();
  if (out.dim()==2 && out.size(0)==1) out = out.squeeze(0);
  if (out.dim()!=1) return std::nullopt;

  ModelOutput mo; mo.probs.resize(out.numel());
  std::memcpy(mo.probs.data(), out.data_ptr<float>(), mo.probs.size()*sizeof(float));
  float sum=0.f; for (float v: mo.probs) sum += std::isfinite(v)? v : 0.f;
  if (!(sum>0.f)) {
    float mv=*std::max_element(mo.probs.begin(), mo.probs.end());
    float s=0; for (auto& v: mo.probs){ v=std::exp(v-mv); s+=v; } if (s>0) for (auto& v: mo.probs) v/=s;
  } else for (auto& v: mo.probs) v = std::max(0.f, v/sum);
  return mo;
}

} // namespace sentio::ml
