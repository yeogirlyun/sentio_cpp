#pragma once
#include "iml_model.hpp"
#include <memory>

namespace torch { namespace jit { class Module; } }

namespace sentio::ml {

class TorchScriptModel final : public IModel {
public:
  static std::unique_ptr<TorchScriptModel> load(const std::string& pt_path,
                                                const ModelSpec& spec,
                                                bool use_cuda = false);

  const ModelSpec& spec() const override { return spec_; }
  std::optional<ModelOutput> predict(const std::vector<float>& features,
                                     int T, int F, const std::string& layout) const override;

  ~TorchScriptModel();

private:
  explicit TorchScriptModel(ModelSpec spec);
  ModelSpec spec_;
  std::shared_ptr<torch::jit::Module> mod_;
  // Preallocated input tensor & shape (PIMPL pattern)
  mutable void* input_tensor_;  // torch::Tensor (hidden in .cpp)
  mutable std::vector<int64_t> in_shape_;
  bool cuda_{false};
};

} // namespace sentio::ml
