#include "sentio/rules/registry.hpp"
#include "sentio/rules/integrated_rule_ensemble.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using json = nlohmann::json;

struct Data { std::vector<long long> ts; std::vector<double> o,h,l,c,v; };

static Data load_csv_simple(const std::string& path){
  Data d; std::ifstream f(path); if(!f){ std::cerr<<"Missing csv: "<<path<<"\n"; return d; }
  std::string line; if(!std::getline(f,line)) return d; // header
  while (std::getline(f,line)){
    if(line.empty()) continue; std::stringstream ss(line); std::string cell; int col=0; long long ts=0; double o=0,h=0,l=0,c=0,v=0;
    while(std::getline(ss,cell,',')){
      switch(col){
        case 0: ts = std::stoll(cell); break;
        case 1: o = std::stod(cell); break;
        case 2: h = std::stod(cell); break;
        case 3: l = std::stod(cell); break;
        case 4: c = std::stod(cell); break;
        case 5: v = std::stod(cell); break;
      }
      col++;
    }
    if(col>=6){ d.ts.push_back(ts); d.o.push_back(o); d.h.push_back(h); d.l.push_back(l); d.c.push_back(c); d.v.push_back(v); }
  }
  return d;
}

static sentio::rules::BarsView as_view(const Data& d){
  return sentio::rules::BarsView{ d.ts.data(), d.o.data(), d.h.data(), d.l.data(), d.c.data(), d.v.data(), (long long)d.c.size() };
}

int main(int argc, char** argv){
  std::string cfg_path="configs/strategies/rule_ensemble.json";
  std::string csv_path="data/QQQ.csv";
  for(int i=1;i<argc;i++){
    std::string a=argv[i];
    if (a=="--cfg" && i+1<argc) cfg_path=argv[++i];
    else if (a=="--csv" && i+1<argc) csv_path=argv[++i];
  }
  json cfg = json::parse(std::ifstream(cfg_path));

  std::vector<std::unique_ptr<sentio::rules::IRuleStrategy>> rulesv;
  for (auto& nm : cfg["rules"]) {
    auto r = sentio::rules::make_rule(nm.get<std::string>());
    if (r) rulesv.push_back(std::move(r));
  }

  sentio::rules::EnsembleConfig ec;
  ec.score_logistic_k = cfg.value("score_logistic_k", 1.2f);
  ec.reliability_window = cfg.value("reliability_window", 512);
  ec.agreement_boost    = cfg.value("agreement_boost", 0.25f);
  ec.min_rules          = cfg.value("min_rules", 1);
  if (cfg.contains("base_weights")) for (auto& w : cfg["base_weights"]) ec.base_weights.push_back(w.get<float>());

  sentio::rules::IntegratedRuleEnsemble ers(std::move(rulesv), ec);

  Data d = load_csv_simple(csv_path);
  auto view = as_view(d);
  std::vector<float> lr(view.n,0.f);
  for (long long i=1;i<view.n;i++) lr[i] = (float)(std::log(std::max(1e-9, view.close[i])) - std::log(std::max(1e-9, view.close[i-1])));

  long long start = ers.warmup();
  long long used=0; double sprob=0.0; long long cnt=0;
  for (long long i=start;i<view.n;i++){
    sentio::rules::EnsembleMeta meta;
    auto p = ers.eval(view, i, (i+1<view.n? std::optional<float>(lr[i+1]) : std::nullopt), &meta);
    if (!p) continue; used++; sprob += *p; cnt++;
  }
  std::cerr << "[RuleEnsemble] n=" << view.n << " used="<<used<<" mean_p="<<(cnt? sprob/cnt:0.0) << "\n";
  return 0;
}


