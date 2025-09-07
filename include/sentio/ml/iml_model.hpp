#pragma once
#include <string>
#include <vector>
#include <optional>

namespace sentio::ml {

// Raw model output abstraction: either class probs or a scalar score.
struct ModelOutput {
  std::vector<float> probs;   // e.g., [p_sell, p_hold, p_buy] for discrete
  float score{0.0f};          // for regressors
};

struct ModelSpec {
  std::string model_id;       // "TransAlpha", "HybridPPO"
  std::string version;        // "v1"
  std::vector<std::string> feature_names;
  std::vector<double> mean, std, clip2; // clip2: [lo, hi]
  std::vector<std::string> actions;     // e.g., SELL, HOLD, BUY
  int expected_spacing_sec{60};
  std::string instrument_family;        // e.g., "QQQ"
  std::string notes;                    // optional metadata
  // Sequence model extensions
  int seq_len{1};  // 1 for non-sequence models
  std::string input_layout{"BTF"};  // "BTF" for batch-time-feature, "BF" for flattened
  std::string format{"torchscript"};  // "torchscript", "onnx", etc.
};

// Runtime inference model
class IModel {
public:
  virtual ~IModel() = default;
  virtual const ModelSpec& spec() const = 0;
  // features must match spec().feature_names length and order
  // T, F, layout parameters for sequence models
  virtual std::optional<ModelOutput> predict(const std::vector<float>& features,
                                             int T, int F, const std::string& layout) const = 0;
};

} // namespace sentio::ml