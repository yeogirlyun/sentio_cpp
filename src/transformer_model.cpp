#include "sentio/transformer_model.hpp"
#include <cmath>
#include <iostream>

namespace sentio {

TransformerModel::TransformerModel(const TransformerConfig& config) : config_(config) {
    // Input projection
    input_projection_ = register_module("input_projection", 
        torch::nn::Linear(config.feature_dim, config.d_model));
    
    // Transformer encoder
    torch::nn::TransformerEncoderLayerOptions encoder_layer_options(config.d_model, config.num_heads);
    encoder_layer_options.dim_feedforward(config.d_model * 4);
    encoder_layer_options.dropout(config.dropout);
    // Note: batch_first option may not be available in all PyTorch versions
    // encoder_layer_options.batch_first(true);
    
    auto encoder_layer = torch::nn::TransformerEncoderLayer(encoder_layer_options);
    
    torch::nn::TransformerEncoderOptions encoder_options(encoder_layer, config.num_layers);
    transformer_ = register_module("transformer", torch::nn::TransformerEncoder(encoder_options));
    
    // Layer normalization
    layer_norm_ = register_module("layer_norm", torch::nn::LayerNorm(std::vector<int64_t>{config.d_model}));
    
    // Output projection
    output_projection_ = register_module("output_projection", 
        torch::nn::Linear(config.d_model, 1));
    
    // Dropout
    dropout_ = register_module("dropout", torch::nn::Dropout(config.dropout));
    
    // Create positional encoding
    create_positional_encoding();
}

void TransformerModel::create_positional_encoding() {
    pos_encoding_ = torch::zeros({config_.sequence_length, config_.d_model});
    
    auto position = torch::arange(0, config_.sequence_length).unsqueeze(1).to(torch::kFloat);
    auto div_term = torch::exp(torch::arange(0, config_.d_model, 2).to(torch::kFloat) * 
                              -(std::log(10000.0) / config_.d_model));
    
    pos_encoding_.slice(1, 0, config_.d_model, 2) = torch::sin(position * div_term);
    pos_encoding_.slice(1, 1, config_.d_model, 2) = torch::cos(position * div_term);
    
    pos_encoding_ = pos_encoding_.unsqueeze(0); // Add batch dimension
}

torch::Tensor TransformerModel::forward(const torch::Tensor& input) {
    // input shape: [batch_size, sequence_length, feature_dim]
    // auto batch_size = input.size(0);  // Unused for now
    auto seq_len = input.size(1);
    
    // Input projection
    auto x = input_projection_->forward(input);
    
    // Add positional encoding
    auto pos_enc = pos_encoding_.slice(1, 0, seq_len).to(x.device());
    x = x + pos_enc;
    x = dropout_->forward(x);
    
    // Transformer encoding
    x = transformer_->forward(x);
    
    // Layer normalization
    x = layer_norm_->forward(x);
    
    // Global average pooling
    x = torch::mean(x, 1); // [batch_size, d_model]
    
    // Output projection
    auto output = output_projection_->forward(x); // [batch_size, 1]
    
    return output;
}

void TransformerModel::save_model(const std::string& path) {
    torch::serialize::OutputArchive archive;
    this->save(archive);
    archive.save_to(path);
}

void TransformerModel::load_model(const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    this->load(archive);
}

void TransformerModel::optimize_for_inference() {
    eval();
    // Additional optimizations could be added here
}

size_t TransformerModel::get_parameter_count() const {
    size_t count = 0;
    for (const auto& param : parameters()) {
        count += param.numel();
    }
    return count;
}

size_t TransformerModel::get_memory_usage_bytes() const {
    size_t bytes = 0;
    for (const auto& param : parameters()) {
        bytes += param.nbytes();
    }
    return bytes;
}

} // namespace sentio
