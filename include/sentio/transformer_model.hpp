#pragma once

#include "transformer_strategy_core.hpp"
#include <torch/torch.h>
#include <memory>
#include <string>

namespace sentio {

// Simple transformer model implementation for Sentio
class TransformerModel : public torch::nn::Module {
public:
    explicit TransformerModel(const TransformerConfig& config);
    
    // Forward pass
    torch::Tensor forward(const torch::Tensor& input);
    
    // Model management
    void save_model(const std::string& path);
    void load_model(const std::string& path);
    void optimize_for_inference();
    
    // Utilities
    size_t get_parameter_count() const;
    size_t get_memory_usage_bytes() const;
    
private:
    TransformerConfig config_;
    
    // Model components
    torch::nn::Linear input_projection_{nullptr};
    torch::nn::TransformerEncoder transformer_{nullptr};
    torch::nn::LayerNorm layer_norm_{nullptr};
    torch::nn::Linear output_projection_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    
    // Positional encoding
    torch::Tensor pos_encoding_;
    
    void create_positional_encoding();
};

} // namespace sentio