// test_main.cpp - Main test runner
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <iostream>

// Main test runner
GTEST_API_ int main(int argc, char** argv) {
    std::cout << "Sentio Transformer Strategy Test Suite" << std::endl;
    std::cout << "======================================" << std::endl;
    
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // Set random seed for reproducible tests
    std::srand(42);
    torch::manual_seed(42);
    
    // Print system information
    std::cout << "System Information:" << std::endl;
    std::cout << "  PyTorch Version: " << TORCH_VERSION << std::endl;
    std::cout << "  CUDA Available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
    if (torch::cuda::is_available()) {
        std::cout << "  CUDA Devices: " << torch::cuda::device_count() << std::endl;
    }
    std::cout << std::endl;
    
    // Run tests
    int result = RUN_ALL_TESTS();
    
    std::cout << std::endl;
    std::cout << "Test suite completed with result: " << result << std::endl;
    
    return result;
}
