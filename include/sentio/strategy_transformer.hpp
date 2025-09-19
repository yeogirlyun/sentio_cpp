#pragma once

#include "base_strategy.hpp"
#include "transformer_strategy_core.hpp"
#include "transformer_model.hpp"
#include "feature_pipeline.hpp"
#include "online_trainer.hpp"
#include "adaptive_allocation_manager.hpp"
#include <memory>
#include <deque>
#include <torch/torch.h>

namespace sentio {

struct TransformerCfg {
    // Model configuration
    int feature_dim = 128;
    int sequence_length = 64;
    int d_model = 256;
    int num_heads = 8;
    int num_layers = 6;
    int ffn_hidden = 1024;
    float dropout = 0.1f;
    
    // Strategy parameters
    float buy_threshold = 0.6f;
    float sell_threshold = 0.4f;
    float strong_threshold = 0.8f;
    float conf_floor = 0.5f;
    
    // Training parameters
    bool enable_online_training = true;
    int update_interval_minutes = 60;
    int min_samples_for_update = 1000;
    
    // Model paths
    std::string model_path = "artifacts/Transformer/v1/model.pt";
    std::string artifacts_dir = "artifacts/Transformer/";
    std::string version = "v1";
};

class TransformerStrategy : public BaseStrategy {
public:
    TransformerStrategy();
    explicit TransformerStrategy(const TransformerCfg& cfg);
    
    // BaseStrategy interface
    ParameterMap get_default_params() const override;
    ParameterSpace get_param_space() const override;
    void apply_params() override;
    double calculate_probability(const std::vector<Bar>& bars, int current_index) override;
    
    // Allocation decisions for profit maximization
    std::vector<AllocationDecision> get_allocation_decisions(
        const std::vector<Bar>& bars, 
        int current_index,
        const std::string& base_symbol = "QQQ",
        const std::string& bull3x_symbol = "TQQQ", 
        const std::string& bear3x_symbol = "SQQQ"
    ) const;

private:
    // Configuration
    TransformerCfg cfg_;
    
    // Core components
    std::shared_ptr<TransformerModel> model_;
    std::unique_ptr<FeaturePipeline> feature_pipeline_;
    std::unique_ptr<OnlineTrainer> online_trainer_;
    
    // Feature management
    std::deque<Bar> bar_history_;
    std::vector<double> current_features_;
    
    // Model state
    bool model_initialized_ = false;
    std::atomic<bool> is_training_{false};
    
    // Performance tracking
    mutable std::mutex metrics_mutex_;
    PerformanceMetrics current_metrics_;
    
    // Helper methods
    void initialize_model();
    void update_bar_history(const Bar& bar);
    void maybe_trigger_training();
    TransformerFeatureMatrix generate_features_for_bars(const std::vector<Bar>& bars, int end_index);
    
    // Validation methods
    bool validate_tensor_dimensions(const torch::Tensor& tensor, 
                                   const std::vector<int64_t>& expected_dims,
                                   const std::string& tensor_name);
    bool validate_configuration();
    
    // Feature conversion
    std::vector<Bar> convert_to_transformer_bars(const std::vector<Bar>& sentio_bars) const;
};

} // namespace sentio
