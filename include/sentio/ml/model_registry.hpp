#pragma once
#include "iml_model.hpp"
#include <memory>

namespace sentio::ml {

struct ModelHandle {
  std::unique_ptr<IModel> model;
  ModelSpec spec;
};

class ModelRegistryTS {
public:
  static ModelHandle load_torchscript(const std::string& model_id,
                                      const std::string& version,
                                      const std::string& artifacts_dir,
                                      bool use_cuda = false);
};

} // namespace sentio::ml