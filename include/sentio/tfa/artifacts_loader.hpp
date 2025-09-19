#pragma once
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <iostream>

namespace sentio::tfa {

struct TfaArtifacts {
  torch::jit::Module model;
  nlohmann::json spec;
  nlohmann::json meta;
  
  // Convenience getters
  std::vector<std::string> get_expected_feature_names() const {
    return meta["expects"]["feature_names"].get<std::vector<std::string>>();
  }
  
  int get_expected_input_dim() const {
    return meta["expects"]["input_dim"].get<int>();
  }
  
  std::string get_spec_hash() const {
    return meta["expects"]["spec_hash"].get<std::string>();
  }
  
  float get_pad_value() const {
    return meta["expects"]["pad_value"].get<float>();
  }
  
  int get_emit_from() const {
    return meta["expects"]["emit_from"].get<int>();
  }
};

inline TfaArtifacts load_tfa(const std::string& model_pt,
                             const std::string& feature_spec_json,
                             const std::string& model_meta_json)
{
  TfaArtifacts A;
  
  A.model = torch::jit::load(model_pt, torch::kCPU);
  A.model.eval();
  
  std::ifstream fs(feature_spec_json); 
  if(!fs) throw std::runtime_error("missing feature_spec.json: " + feature_spec_json);
  fs >> A.spec;
  
  std::ifstream fm(model_meta_json); 
  if(!fm) throw std::runtime_error("missing model.meta.json: " + model_meta_json);
  fm >> A.meta;
  
  // Validate metadata structure
  if (!A.meta.contains("expects")) {
    throw std::runtime_error("model.meta.json missing 'expects' section");
  }
  
  auto expects = A.meta["expects"];
  if (!expects.contains("input_dim") || !expects.contains("feature_names") || 
      !expects.contains("spec_hash") || !expects.contains("pad_value") || 
      !expects.contains("emit_from")) {
    throw std::runtime_error("model.meta.json 'expects' section incomplete");
  }
  
  
  return A;
}

// Fallback loader for existing metadata.json (without model.meta.json)
inline TfaArtifacts load_tfa_legacy(const std::string& model_pt,
                                     const std::string& metadata_json)
{
  TfaArtifacts A;
  
  A.model = torch::jit::load(model_pt, torch::kCPU);
  A.model.eval();
  
  std::ifstream fs(metadata_json);
  if(!fs) throw std::runtime_error("missing metadata.json: " + metadata_json);
  
  nlohmann::json legacy_meta;
  fs >> legacy_meta;
  
  // Convert legacy metadata.json to new format
  A.spec = legacy_meta; // Use legacy as spec for now
  
  // Create synthetic model.meta.json structure
  A.meta = {
    {"schema_version", "1.0"},
    {"framework", "torchscript"},
    {"expects", {
      {"input_dim", (int)legacy_meta["feature_names"].size()},
      {"feature_names", legacy_meta["feature_names"]},
      {"spec_hash", "legacy"},
      {"emit_from", 64}, // Default for TFA
      {"pad_value", 0.0f},
      {"dtype", "float32"}
    }}
  };
  
  
  return A;
}

} // namespace sentio::tfa
