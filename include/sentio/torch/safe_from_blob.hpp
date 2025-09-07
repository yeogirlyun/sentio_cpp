#pragma once
#include <torch/torch.h>
#include <memory>
#include <vector>

namespace sentio {

// Creates a Tensor that OWNS a heap buffer and will free it when Tensor dies.
// If you already have a std::shared_ptr<float> backing store, prefer that version.
inline torch::Tensor own_copy_tensor(const float* src, int64_t rows, int64_t cols) {
  auto t = torch::empty({rows, cols}, torch::dtype(torch::kFloat32));
  t.copy_(torch::from_blob((void*)src, {rows, cols}, torch::kFloat32)); // safe copy
  return t;
}

// If you insist on zero-copy, give Tensor a deleter tied to a shared_ptr:
inline torch::Tensor tensor_from_shared(std::shared_ptr<std::vector<float>> store, int64_t rows, int64_t cols) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  return torch::from_blob(
      (void*)store->data(),
      {rows, cols},
      [store](void*) mutable { store.reset(); }, // keep alive
      options);
}

// Safe batched forward pass with owned tensors
inline std::vector<float> model_forward_probs(torch::jit::Module& m, const float* X, int64_t rows, int64_t cols, bool logits=true){
  std::vector<float> probs((size_t)rows, 0.f);
  torch::NoGradGuard ng; 
  torch::InferenceMode im;
  const int64_t B = 8192;
  for (int64_t i=0;i<rows;i+=B){
    int64_t b = std::min<int64_t>(B, rows-i);
    auto t = torch::from_blob((void*)(X + i*cols), {b, cols}, torch::kFloat32).clone(); // OWNED
    t = t.contiguous(); // belt & suspenders
    auto y = m.forward({t}).toTensor();
    if (y.dim()==2 && y.size(1)==1) y = y.squeeze(1);
    if (y.dim()!=1 || y.size(0)!=b) throw std::runtime_error("model output shape mismatch");
    auto acc = y.contiguous().data_ptr<float>();
    for (int64_t k=0;k<b;++k){
      float v = acc[k];
      probs[(size_t)(i+k)] = logits ? 1.f/(1.f+std::exp(-v)) : v;
    }
  }
  return probs;
}

} // namespace sentio
