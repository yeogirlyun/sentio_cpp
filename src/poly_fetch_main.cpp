#include "sentio/polygon_client.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>

using namespace sentio;

std::string get_yesterday_date() {
    std::time_t now = std::time(nullptr);
    std::time_t yesterday = now - 24 * 60 * 60; // Subtract 1 day in seconds
    
    std::tm* tm_yesterday = std::gmtime(&yesterday);
    std::ostringstream oss;
    oss << std::put_time(tm_yesterday, "%Y-%m-%d");
    return oss.str();
}

std::string get_current_date() {
    std::time_t now = std::time(nullptr);
    std::tm* tm_now = std::gmtime(&now);
    std::ostringstream oss;
    oss << std::put_time(tm_now, "%Y-%m-%d");
    return oss.str();
}

std::string calculate_start_date(int years, int months, int days) {
    std::time_t now = std::time(nullptr);
    std::time_t yesterday = now - 24 * 60 * 60; // Start from yesterday
    
    std::tm* tm_start = std::gmtime(&yesterday);
    
    if (years > 0) {
        tm_start->tm_year -= years;
    } else if (months > 0) {
        tm_start->tm_mon -= months;
        if (tm_start->tm_mon < 0) {
            tm_start->tm_mon += 12;
            tm_start->tm_year--;
        }
    } else if (days > 0) {
        tm_start->tm_mday -= days;
        // Let mktime handle month/year overflow
        std::mktime(tm_start);
    } else {
        // Default: 3 years
        tm_start->tm_year -= 3;
    }
    
    std::ostringstream oss;
    oss << std::put_time(tm_start, "%Y-%m-%d");
    return oss.str();
}

int main(int argc,char**argv){
  if(argc<3){
    std::cerr<<"Usage: poly_fetch FAMILY outdir [--years N] [--months N] [--days N] [--timespan day|hour|minute] [--multiplier N] [--symbols SYM1,SYM2,...] [--no-holidays] [--rth-only]\n";
    std::cerr<<"       poly_fetch FAMILY from to outdir [--timespan day|hour|minute] [--multiplier N] [--symbols SYM1,SYM2,...] [--no-holidays] [--rth-only]\n";
    std::cerr<<"Examples:\n";
    std::cerr<<"  poly_fetch qqq data/equities --years 3 --no-holidays --rth-only\n";
    std::cerr<<"  poly_fetch qqq 2022-01-01 2025-01-10 data/equities --timespan minute --rth-only\n";
    return 1;
  }
  
  std::string fam=argv[1];
  std::string from, to, outdir;
  
  // Check if we're using time range options (new format) or explicit dates (old format)
  bool use_time_range = false;
  int years = 0, months = 0, days = 0;
  
  if (argc >= 3) {
    // Check if second argument is a directory (new format) or a date (old format)
    std::string second_arg = argv[2];
    if (second_arg.find('/') != std::string::npos || second_arg == "data" || second_arg == "data/equities") {
      // New format: FAMILY outdir [time options]
      outdir = second_arg;
      use_time_range = true;
    } else if (argc >= 5) {
      // Old format: FAMILY from to outdir
      from = argv[2];
      to = argv[3];
      outdir = argv[4];
    } else {
      std::cerr<<"Error: Invalid arguments. Use --help for usage.\n";
      return 1;
    }
  }
  
  std::string timespan = "day";
  int multiplier = 1;
  std::string symbols_csv;
  bool exclude_holidays=false;
  bool rth_only=false;
  
  int start_idx = use_time_range ? 3 : 5;
  for (int i=start_idx;i<argc;i++) {
    std::string a = argv[i];
    if (a=="--years" && i+1<argc) { years = std::stoi(argv[++i]); }
    else if (a=="--months" && i+1<argc) { months = std::stoi(argv[++i]); }
    else if (a=="--days" && i+1<argc) { days = std::stoi(argv[++i]); }
    else if ((a=="--timespan" || a=="-t") && i+1<argc) { timespan = argv[++i]; }
    else if ((a=="--multiplier" || a=="-m") && i+1<argc) { multiplier = std::stoi(argv[++i]); }
    else if (a=="--symbols" && i+1<argc) { symbols_csv = argv[++i]; }
    else if (a=="--rth-only") { rth_only=true; }
    else if (a=="--no-holidays") { exclude_holidays=true; }
  }
  
  // Calculate dates if using time range options
  if (use_time_range) {
    from = calculate_start_date(years, months, days);
    to = get_yesterday_date();
    std::cerr<<"Current date: " << get_current_date() << "\n";
    std::cerr<<"Downloading " << (years > 0 ? std::to_string(years) + " years" : 
                                  months > 0 ? std::to_string(months) + " months" : 
                                  days > 0 ? std::to_string(days) + " days" : "3 years (default)") 
             << " of data: " << from << " to " << to << "\n";
  }
  const char* key = std::getenv("POLYGON_API_KEY");
  std::string api_key = key? key: "";
  PolygonClient cli(api_key);

  std::vector<std::string> syms;
  if(fam=="qqq") syms={"QQQ","TQQQ","SQQQ"};
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
    cli.write_csv(fname,s,bars,exclude_holidays,rth_only);
    std::cerr<<"Wrote "<<bars.size()<<" bars -> "<<fname<<"\n";
  }
}

