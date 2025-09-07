#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <unordered_set>

namespace sentio {

inline void print_name_diff(const std::vector<std::string>& src, const std::vector<std::string>& dst){
  std::unordered_set<std::string> S(src.begin(), src.end()), D(dst.begin(), dst.end());
  int miss=0, extra=0, reorder=0;

  std::cout << "[DIFF] Feature name differences:" << std::endl;
  for (auto& n : dst) {
    if (!S.count(n)) { 
      miss++; 
      std::cout << "  MISSING: " << n << std::endl; 
    }
  }
  
  for (auto& n : src) {
    if (!D.count(n)) { 
      extra++; 
      std::cout << "  EXTRA  : " << n << std::endl; 
    }
  }

  if (miss==0 && extra==0 && src.size()==dst.size()){
    for (size_t i=0;i<src.size();++i) {
      if (src[i] != dst[i]) reorder++;
    }
    if (reorder>0) {
      std::cout << "  REORDER count: " << reorder << std::endl;
    }
  }
  
  std::cout << "  Summary: missing=" << miss << " extra=" << extra << " reordered=" << reorder << std::endl;
}

} // namespace sentio
