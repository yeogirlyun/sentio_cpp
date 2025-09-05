#include "sentio/ml/model_registry.hpp"
#include "sentio/ml/onnx_model.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace sentio::ml {

static std::string slurp(const std::string& path){
  std::ifstream f(path); if (!f) throw std::runtime_error("cannot open "+path);
  std::ostringstream ss; ss<<f.rdbuf(); return ss.str();
}

// Ultra-light JSON getters (expects flat metadata per schema)
static bool find_value(const std::string& j, const std::string& key, std::string& out) {
  auto k = "\""+key+"\"";
  auto p = j.find(k); if (p==std::string::npos) return false;
  p = j.find(':', p); if (p==std::string::npos) return false; ++p;
  while (p<j.size() && (j[p]==' ')) ++p;
  if (j[p]=='"'){ auto e=j.find('"',p+1); out=j.substr(p+1,e-(p+1)); return true; }
  auto e=j.find_first_of(",}\n", p); out=j.substr(p, e-p); return true;
}

static std::vector<std::string> parse_str_array(const std::string& j, const std::string& key){
  std::vector<std::string> v;
  auto k="\""+key+"\""; auto p=j.find(k); if (p==std::string::npos) return v;
  p=j.find('[',p); auto e=j.find(']',p); if (p==std::string::npos||e==std::string::npos) return v;
  auto s=j.substr(p+1, e-(p+1));
  size_t i=0; while (i<s.size()){
    auto q1=s.find('"',i); if (q1==std::string::npos) break;
    auto q2=s.find('"',q1+1); v.push_back(s.substr(q1+1,q2-(q1+1))); i=q2+1;
  }
  return v;
}

static std::vector<double> parse_num_array(const std::string& j, const std::string& key){
  std::vector<double> v;
  auto k="\""+key+"\""; auto p=j.find(k); if (p==std::string::npos) return v;
  p=j.find('[',p); auto e=j.find(']',p); if (p==std::string::npos||e==std::string::npos) return v;
  auto s=j.substr(p+1, e-(p+1));
  size_t i=0; while (i<s.size()){
    auto j2=s.find_first_of(", \t\n", i); auto tok=s.substr(i,(j2==std::string::npos)?std::string::npos:(j2-i));
    if (!tok.empty()) v.push_back(std::stod(tok));
    if (j2==std::string::npos) break; i=j2+1;
  }
  return v;
}

ModelHandle ModelRegistry::load_onnx(const std::string& model_id,
                                     const std::string& version,
                                     const std::string& artifacts_dir)
{
  ModelSpec spec;
  spec.model_id = model_id;
  spec.version  = version;
  const std::string base = artifacts_dir + "/" + model_id + "/" + version + "/";
  const std::string meta_path = base + "metadata.json";
  const std::string onnx_path = base + "model.onnx";

  auto js = slurp(meta_path);
  spec.feature_names = parse_str_array(js, "feature_names");
  spec.mean = parse_num_array(js, "mean");
  spec.std  = parse_num_array(js, "std");
  auto clip = parse_num_array(js, "clip");
  if (clip.size()==2) spec.clip2 = clip;
  spec.actions = parse_str_array(js, "actions");
  std::string esp; if (find_value(js, "expected_bar_spacing_sec", esp)) spec.expected_spacing_sec = std::stoi(esp);
  find_value(js, "instrument_family", spec.instrument_family);
  find_value(js, "notes", spec.notes);

  ModelHandle h;
  h.spec = spec;
  h.model = OnnxModel::load(onnx_path, h.spec);
  return h;
}

} // namespace sentio::ml
