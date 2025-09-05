#pragma once
#include "core.hpp"
#include <string>

namespace sentio {
bool load_csv(const std::string& path, std::vector<Bar>& out);
} // namespace sentio

