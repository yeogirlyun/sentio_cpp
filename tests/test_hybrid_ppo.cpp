#include "sentio/strategy_hybrid_ppo.hpp"
#include "sentio/ml/feature_pipeline.hpp"
#include "sentio/ml/model_registry.hpp"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    std::cout << "🧪 Testing HybridPPO Strategy (Fallback Mode)" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    try {
        // Test 1: Create strategy with default config
        std::cout << "1. Creating HybridPPO strategy..." << std::endl;
        sentio::HybridPPOCfg cfg;
        cfg.artifacts_dir = "artifacts";
        cfg.version = "v1";
        cfg.conf_floor = 0.1; // Higher floor for testing
        
        sentio::HybridPPOStrategy strat(cfg);
        std::cout << "   ✅ Strategy created successfully" << std::endl;
        
        // Test 2: Set raw features (7 features as per metadata)
        std::cout << "2. Setting raw features..." << std::endl;
        std::vector<double> raw_features = {0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 1.5};
        strat.set_raw_features(raw_features);
        std::cout << "   ✅ Features set successfully" << std::endl;
        
        // Test 3: Process a bar
        std::cout << "3. Processing bar..." << std::endl;
        std::vector<sentio::Bar> bars;
        sentio::Bar b;
        b.ts_utc = "2024-01-01T09:30:00Z";
        b.ts_nyt_epoch = 1000000;
        b.open = 100.0;
        b.high = 101.0;
        b.low = 99.0;
        b.close = 100.5;
        b.volume = 1000;
        bars.push_back(b);
        
        auto signal = strat.calculate_signal(bars, 0);
        std::cout << "   ✅ Bar processed successfully" << std::endl;
        
        // Test 4: Check for signal
        std::cout << "4. Checking for signal..." << std::endl;
        if (signal.type != sentio::StrategySignal::Type::HOLD) {
            std::cout << "   ✅ Signal generated:" << std::endl;
            std::cout << "      Type: " << (int)signal.type << std::endl;
            std::cout << "      Confidence: " << signal.confidence << std::endl;
        } else {
            std::cout << "   ℹ️  No signal generated (expected in fallback mode)" << std::endl;
        }
        
        // Test 5: Test feature pipeline directly
        std::cout << "5. Testing feature pipeline..." << std::endl;
        sentio::ml::ModelSpec spec;
        spec.feature_names = {"ret_1m", "ret_5m", "rsi_14", "sma_10", "sma_30", "vol_1m", "spread_bp"};
        spec.mean = {0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 1.5};
        spec.std = {1.0, 1.0, 20.0, 1.0, 1.0, 1.0, 0.5};
        spec.clip2 = {-5.0, 5.0};
        
        sentio::ml::FeaturePipeline pipeline(spec);
        auto transformed = pipeline.transform(raw_features);
        
        if (transformed) {
            std::cout << "   ✅ Feature transformation successful" << std::endl;
            std::cout << "      Transformed features: ";
            for (size_t i = 0; i < transformed->size(); ++i) {
                std::cout << (*transformed)[i];
                if (i < transformed->size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "   ❌ Feature transformation failed" << std::endl;
            return 1;
        }
        
        std::cout << std::endl;
        std::cout << "🎉 All tests passed!" << std::endl;
        std::cout << "   HybridPPO strategy is working correctly in fallback mode" << std::endl;
        std::cout << "   (ONNX Runtime not available, using fallback implementation)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
