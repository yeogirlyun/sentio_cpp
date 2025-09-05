#pragma once
#include "iml_model.hpp"
#include <memory>

namespace sentio::ml {

class OnnxModel final : public IModel {
public:
  // Throws on load error (missing files, shape mismatch, etc.)
  static std::unique_ptr<OnnxModel> load(const std::string& onnx_path,
                                         const ModelSpec& spec,
                                         int intra_threads = 1,
                                         int inter_threads = 1);

  const ModelSpec& spec() const override { return spec_; }
  std::optional<ModelOutput> predict(const std::vector<float>& features) const override;

private:
  explicit OnnxModel(ModelSpec spec);
  ModelSpec spec_;
#ifdef SENTIO_WITH_ONNXRUNTIME
  // forward-declared to avoid including heavy headers here
  struct Impl;
  std::unique_ptr<Impl> impl_;
#endif
};

} // namespace sentio::ml
