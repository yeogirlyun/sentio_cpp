#pragma once
#include "sentio/core.hpp"
#include <vector>
#include <string>

namespace sentio {
namespace feature_engineering {

// Returns Kochi feature names in the exact order expected by the trainer.
// This excludes any state features (position one-hot, PnL), which are not used at inference time.
std::vector<std::string> kochi_feature_names();

// Compute Kochi feature vector for a given bar index using bar history.
// Window-dependent features use typical Kochi defaults (e.g., 20 for many).
// The output order matches kochi_feature_names().
std::vector<double> calculate_kochi_features(const std::vector<Bar>& bars, int current_index);

} // namespace feature_engineering
} // namespace sentio


