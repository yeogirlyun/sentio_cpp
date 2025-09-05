#pragma once
#include "iml_model.hpp"
#include <memory>
#include <unordered_map>

namespace sentio::ml {

struct ModelHandle {
  std::unique_ptr<IModel> model;
  ModelSpec spec;
};

class ModelRegistry {
public:
  // Load and memoize by (model_id, version)
  static ModelHandle load_onnx(const std::string& model_id,
                               const std::string& version,
                               const std::string& artifacts_dir);
};

} // namespace sentio::ml
