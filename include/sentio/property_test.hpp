#pragma once
#include <functional>
#include <vector>
#include <string>
#include <iostream>

namespace sentio {

struct PropCase { std::string name; std::function<bool()> fn; };

inline int run_properties(const std::vector<PropCase>& cases) {
  int fails = 0;
  for (auto& c : cases) {
    bool ok = false;
    try { ok = c.fn(); }
    catch (const std::exception& e) {
      std::cerr << "[PROP] " << c.name << " threw: " << e.what() << "\n";
      ok = false;
    }
    if (!ok) { std::cerr << "[PROP] FAIL: " << c.name << "\n"; ++fails; }
  }
  if (fails==0) std::cout << "[PROP] all passed ("<<cases.size()<<")\n";
  return fails==0 ? 0 : 1;
}

} // namespace sentio
