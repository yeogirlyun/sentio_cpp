#include "sentio/ml/model_registry.hpp"
#include "sentio/ml/ts_model.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace sentio::ml {

static std::string slurp(const std::string& path){
  std::ifstream f(path); if (!f) throw std::runtime_error("cannot open "+path);
  std::ostringstream ss; ss<<f.rdbuf(); return ss.str();
}
static bool find_val(const std::string& j, const std::string& key, std::string& out){
  auto k="\""+key+"\""; auto p=j.find(k); if (p==std::string::npos) return false;
  p=j.find(':',p); if (p==std::string::npos) return false; ++p;
  while (p<j.size() && isspace((unsigned char)j[p])) ++p;
  if (j[p]=='"'){ auto e=j.find('"',p+1); out=j.substr(p+1,e-(p+1)); return true; }
  auto e=j.find_first_of(",}\n",p); out=j.substr(p,e-p); return true;
}
static std::vector<std::string> parse_sarr(const std::string& j, const std::string& key){
  std::vector<std::string> v; auto k="\""+key+"\""; auto p=j.find(k); if (p==std::string::npos) return v;
  p=j.find('[',p); auto e=j.find(']',p); if (p==std::string::npos||e==std::string::npos) return v;
  auto s=j.substr(p+1,e-(p+1)); size_t i=0;
  while (i<s.size()){ auto q1=s.find('"',i); if (q1==std::string::npos) break; auto q2=s.find('"',q1+1);
    v.push_back(s.substr(q1+1,q2-(q1+1))); i=q2+1; }
  return v;
}
static std::vector<double> parse_darr(const std::string& j, const std::string& key){
  std::vector<double> v; auto k="\""+key+"\""; auto p=j.find(k); if (p==std::string::npos) return v;
  p=j.find('[',p); auto e=j.find(']',p); if (p==std::string::npos||e==std::string::npos) return v;
  auto s=j.substr(p+1,e-(p+1)); size_t i=0;
  while (i<s.size()){ auto j2=s.find_first_of(", \t\n", i);
    auto tok=s.substr(i,(j2==std::string::npos)?std::string::npos:(j2-i));
    if (!tok.empty()) v.push_back(std::stod(tok));
    if (j2==std::string::npos) break; i=j2+1; }
  return v;
}

ModelHandle ModelRegistryTS::load_torchscript(const std::string& model_id,
                                              const std::string& version,
                                              const std::string& artifacts_dir,
                                              bool use_cuda)
{
  const std::string base = artifacts_dir + "/" + model_id + "/" + version + "/";
  const std::string meta_path = base + "metadata.json";
  const std::string pt_path   = base + "model.pt";

  auto js = slurp(meta_path);

  ModelSpec spec;
  spec.model_id = model_id;
  spec.version  = version;
  spec.feature_names = parse_sarr(js, "feature_names");
  spec.mean = parse_darr(js, "mean");
  spec.std  = parse_darr(js, "std");
  auto clip = parse_darr(js, "clip"); if (clip.size()==2) spec.clip2 = clip;
  spec.actions = parse_sarr(js, "actions");

  std::string t; if (find_val(js, "expected_bar_spacing_sec", t)) spec.expected_spacing_sec = std::stoi(t);
  if (find_val(js, "seq_len", t)) spec.seq_len = std::stoi(t);
  std::string layout; if (find_val(js, "input_layout", layout)) spec.input_layout = layout;
  std::string fmt; if (find_val(js, "format", fmt)) spec.format = fmt; else spec.format="torchscript";

  ModelHandle h;
  h.spec = spec;
  h.model = TorchScriptModel::load(pt_path, h.spec, use_cuda);
  return h;
}

} // namespace sentio::ml
