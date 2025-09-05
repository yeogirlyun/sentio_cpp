#pragma once
#include <vector>
#include <string>
#include <unordered_map>

namespace sentio {

struct SymbolTable {
  std::vector<std::string> id2sym;
  std::unordered_map<std::string,int> sym2id;

  int intern(const std::string& s){
    auto it = sym2id.find(s);
    if (it != sym2id.end()) return it->second;
    int id = (int)id2sym.size();
    id2sym.push_back(s);
    sym2id.emplace(id2sym.back(), id);
    return id;
  }

  const std::string& get_symbol(int id) const {
    return id2sym[id];
  }

  int get_id(const std::string& sym) const {
    auto it = sym2id.find(sym);
    return it != sym2id.end() ? it->second : -1;
  }

  size_t size() const {
    return id2sym.size();
  }
};

} // namespace sentio