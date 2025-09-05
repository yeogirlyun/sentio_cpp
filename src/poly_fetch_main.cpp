#include "sentio/polygon_client.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

using namespace sentio;

int main(int argc,char**argv){
  if(argc<5){
    std::cerr<<"Usage: poly_fetch FAMILY from to outdir [--timespan day|hour|minute] [--multiplier N] [--symbols SYM1,SYM2,...] [--rth]\n";
    return 1;
  }
  std::string fam=argv[1], from=argv[2], to=argv[3], outdir=argv[4];
  std::string timespan = "day";
  int multiplier = 1;
  std::string symbols_csv;
  bool rth_only=false;
  bool exclude_holidays=false;
  for (int i=5;i<argc;i++) {
    std::string a = argv[i];
    if ((a=="--timespan" || a=="-t") && i+1<argc) { timespan = argv[++i]; }
    else if ((a=="--multiplier" || a=="-m") && i+1<argc) { multiplier = std::stoi(argv[++i]); }
    else if (a=="--symbols" && i+1<argc) { symbols_csv = argv[++i]; }
    else if (a=="--rth") { rth_only=true; }
    else if (a=="--no-holidays") { exclude_holidays=true; }
  }
  const char* key = std::getenv("POLYGON_API_KEY");
  std::string api_key = key? key: "";
  PolygonClient cli(api_key);

  std::vector<std::string> syms;
  if(fam=="qqq") syms={"QQQ","TQQQ","SQQQ","PSQ"};
  else if(fam=="bitcoin") syms={"X:BTCUSD","X:ETHUSD"};
  else if(fam=="tesla") syms={"TSLA","TSLQ"};
  else if(fam=="custom") {
    if (symbols_csv.empty()) { std::cerr<<"--symbols required for custom family\n"; return 1; }
    size_t start=0; while (start < symbols_csv.size()) {
      size_t pos = symbols_csv.find(',', start);
      std::string tok = (pos==std::string::npos)? symbols_csv.substr(start) : symbols_csv.substr(start, pos-start);
      if (!tok.empty()) syms.push_back(tok);
      if (pos==std::string::npos) break; else start = pos+1;
    }
  } else { std::cerr<<"Unknown family\n"; return 1; }

  for(auto&s:syms){
    AggsQuery q; q.symbol=s; q.from=from; q.to=to; q.timespan=timespan; q.multiplier=multiplier; q.adjusted=true; q.sort="asc";
    auto bars=cli.get_aggs_all(q);
    std::string suffix;
    if (rth_only) suffix += "_RTH";
    if (exclude_holidays) suffix += "_NH";
    std::string fname= outdir + "/" + s + suffix + ".csv";
    cli.write_csv(fname,s,bars,rth_only,exclude_holidays);
    std::cerr<<"Wrote "<<bars.size()<<" bars -> "<<fname<<"\n";
  }
}

