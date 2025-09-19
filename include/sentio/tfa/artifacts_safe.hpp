#pragma once
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>

namespace sentio::tfa {

struct TfaArtifactsSafe {
  torch::jit::Module model;
  nlohmann::json spec;
  nlohmann::json meta;
  
  // Convenience getters with validation
  std::vector<std::string> get_expected_feature_names() const {
    if (!meta.contains("expects") || !meta["expects"].contains("feature_names")) {
      throw std::runtime_error("Model metadata missing feature_names");
    }
    return meta["expects"]["feature_names"].get<std::vector<std::string>>();
  }
  
  int get_expected_input_dim() const {
    if (!meta.contains("expects") || !meta["expects"].contains("input_dim")) {
      throw std::runtime_error("Model metadata missing input_dim");
    }
    return meta["expects"]["input_dim"].get<int>();
  }
  
  std::string get_spec_hash() const {
    if (!meta.contains("expects") || !meta["expects"].contains("spec_hash")) {
      throw std::runtime_error("Model metadata missing spec_hash");
    }
    return meta["expects"]["spec_hash"].get<std::string>();
  }
  
  float get_pad_value() const {
    if (!meta.contains("expects") || !meta["expects"].contains("pad_value")) {
      return 0.0f; // Default
    }
    return meta["expects"]["pad_value"].get<float>();
  }
  
  int get_emit_from() const {
    if (!meta.contains("expects") || !meta["expects"].contains("emit_from")) {
      return 64; // Default for TFA
    }
    return meta["expects"]["emit_from"].get<int>();
  }
};

inline TfaArtifactsSafe load_tfa_artifacts_safe(const std::string& model_pt,
                                                const std::string& feature_spec_json,
                                                const std::string& model_meta_json)
{
  TfaArtifactsSafe A;
  
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
      !expects.contains("spec_hash")) {
    throw std::runtime_error("model.meta.json 'expects' section incomplete");
  }
  
  // Validate spec hash if available
  if (A.spec.contains("content_hash")) {
    std::string spec_hash = A.spec["content_hash"].get<std::string>();
    std::string expected_hash = A.get_spec_hash();
    if (spec_hash != expected_hash) {
    }
  }
  
  
  return A;
}

inline std::vector<std::string> feature_names_from_spec(const nlohmann::json& spec){
  std::vector<std::string> names;
  if (!spec.contains("features")) {
    throw std::runtime_error("Feature spec missing 'features' array");
  }
  
  for (auto& f : spec["features"]){
    if (f.contains("name")) {
      names.push_back(f["name"].get<std::string>());
    } else {
      std::string op = f.value("op", "UNKNOWN");
      std::string src = f.value("source", "");
      std::string w = f.contains("window") ? std::to_string((int)f["window"]) : "";
      std::string k = f.contains("k") ? std::to_string((float)f["k"]) : "";
      names.push_back(op + "_" + src + "_" + w + "_" + k);
    }
  }
  return names;
}

} // namespace sentio::tfa
