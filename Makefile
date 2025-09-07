# Sentio C++ Makefile
# C++20 backtesting framework

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -march=native -DNDEBUG -ffast-math -fno-math-errno -fno-trapping-math -DSENTIO_WITH_LIBTORCH -flto
CXXFLAGS_DEBUG = -std=c++20 -Wall -Wextra -O1 -g -DSENTIO_WITH_LIBTORCH -fsanitize=address,undefined -fno-omit-frame-pointer
INCLUDES = -Iinclude -Ithird_party/nlohmann/include -I/opt/homebrew/include -I/opt/homebrew/lib/python3.12/site-packages/torch/include -I/opt/homebrew/lib/python3.12/site-packages/torch/include/torch/csrc/api/include
LIBS = -lsqlite3 -lcurl -L/opt/homebrew/lib -lcctz -L/opt/homebrew/lib/python3.12/site-packages/torch/lib -ltorch -ltorch_cpu -lc10
LIBS_DEBUG = -lsqlite3 -lcurl -L/opt/homebrew/lib -lcctz -L/opt/homebrew/lib/python3.12/site-packages/torch/lib -ltorch -ltorch_cpu -lc10 -fsanitize=address,undefined

# Python module flags
PYTHON_INCLUDES = -Iinclude -Ithird_party/nlohmann/include -I/opt/homebrew/include -I/opt/homebrew/lib/python3.13/site-packages/pybind11/include -I/opt/homebrew/lib/python3.13/site-packages/numpy/core/include -I/Library/Frameworks/Python.framework/Versions/3.13/include/python3.13
PYTHON_LIBS = -L/opt/homebrew/lib -L/Library/Frameworks/Python.framework/Versions/3.13/lib -lpython3.13

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# Source files (exclude poly_fetch_main.cpp and test_rth.cpp from main executable)
SOURCES = $(filter-out $(SRC_DIR)/poly_fetch_main.cpp $(SRC_DIR)/test_rth.cpp, $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/ml/*.cpp) $(wildcard $(SRC_DIR)/feature_engineering/*.cpp))
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Main targets
MAIN_TARGET = $(BUILD_DIR)/sentio_cli
POLY_TARGET = $(BUILD_DIR)/poly_fetch
LIBRARY_TARGET = $(BUILD_DIR)/libsentio.a
TEST_TARGET = $(BUILD_DIR)/test_sma_cross
PIPELINE_TEST_TARGET = $(BUILD_DIR)/test_pipeline_emits
AUDIT_TEST_TARGET = $(BUILD_DIR)/test_audit_replay
AUDIT_SIMPLE_TEST_TARGET = $(BUILD_DIR)/test_audit_simple
SANITY_TEST_TARGET = $(BUILD_DIR)/test_sanity_end_to_end
PERF_TEST_TARGET = $(BUILD_DIR)/test_tfa_performance
PRECOMPUTE_FEATURES_TARGET = $(BUILD_DIR)/precompute_features
PROD_PERF_TEST_TARGET = $(BUILD_DIR)/test_production_performance
PROGRESS_BAR_TEST_TARGET = $(BUILD_DIR)/test_progress_bar
WF_PROGRESS_TEST_TARGET = $(BUILD_DIR)/test_wf_progress
SANITY_INTEGRATION_TARGET = $(BUILD_DIR)/sanity_integration_example
HYBRID_PPO_TEST_TARGET = $(BUILD_DIR)/test_hybrid_ppo
LEVERAGE_TEST_TARGET = $(BUILD_DIR)/test_leverage
BUILDER_GUARD_TEST_TARGET = $(BUILD_DIR)/test_builder_guard
LEVERAGE_EXAMPLE_TARGET = $(BUILD_DIR)/leverage_guarded_example
TFA_CORRUPTION_TEST_TARGET = $(BUILD_DIR)/test_tfa_corruption
PYTHON_MODULE_TARGET = sentio_features.cpython-313-darwin.so

# Default target
all: $(MAIN_TARGET) $(POLY_TARGET) $(TEST_TARGET) $(PIPELINE_TEST_TARGET) $(AUDIT_TEST_TARGET) $(AUDIT_SIMPLE_TEST_TARGET) $(SANITY_TEST_TARGET) $(SANITY_INTEGRATION_TARGET) $(HYBRID_PPO_TEST_TARGET) $(TS_TEST_TARGET) $(TS_PARITY_TEST_TARGET) $(FEATURE_BUILDER_TEST_TARGET) $(PERF_TEST_TARGET) $(PROD_PERF_TEST_TARGET) $(PROGRESS_BAR_TEST_TARGET) $(WF_PROGRESS_TEST_TARGET) $(LEVERAGE_TEST_TARGET) $(BUILDER_GUARD_TEST_TARGET) $(LEVERAGE_EXAMPLE_TARGET) $(PYTHON_MODULE_TARGET)

# Main executable
$(MAIN_TARGET): $(OBJECTS)
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Poly fetch executable
$(POLY_TARGET): $(OBJ_DIR)/poly_fetch_main.o $(OBJ_DIR)/polygon_client.o $(OBJ_DIR)/csv_loader.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Static library
$(LIBRARY_TARGET): $(OBJECTS)
	@echo "Creating $@"
	@mkdir -p $(BUILD_DIR)
	ar rcs $@ $^


# Unit test
$(TEST_TARGET): tests/test_sma_cross_emit.cpp $(OBJ_DIR)/signal_gate.o $(OBJ_DIR)/strategy_sma_cross.o $(OBJ_DIR)/signal_engine.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)


# Pipeline test
$(PIPELINE_TEST_TARGET): tests/test_pipeline_emits.cpp $(OBJ_DIR)/signal_gate.o $(OBJ_DIR)/strategy_sma_cross.o $(OBJ_DIR)/signal_engine.o $(OBJ_DIR)/signal_pipeline.o $(OBJ_DIR)/signal_trace.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)


# Audit test
$(AUDIT_TEST_TARGET): tests/test_audit_replay.cpp $(OBJ_DIR)/audit.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Audit simple test
$(AUDIT_SIMPLE_TEST_TARGET): tests/test_audit_simple.cpp $(OBJ_DIR)/audit.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Sanity test
$(SANITY_TEST_TARGET): tests/test_sanity_end_to_end.cpp $(OBJ_DIR)/sanity.o $(OBJ_DIR)/sim_data.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Performance test
$(PERF_TEST_TARGET): tests/test_tfa_performance.cpp $(OBJ_DIR)/ml/model_registry_ts.o $(OBJ_DIR)/ml/ts_model.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Feature precomputation tool
$(PRECOMPUTE_FEATURES_TARGET): tools/precompute_features.cpp $(OBJ_DIR)/csv_loader.o $(OBJ_DIR)/feature_engineering/technical_indicators.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Production performance test
$(PROD_PERF_TEST_TARGET): tests/test_production_performance.cpp $(OBJ_DIR)/ml/model_registry_ts.o $(OBJ_DIR)/ml/ts_model.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Progress bar test
$(PROGRESS_BAR_TEST_TARGET): tests/test_progress_bar.cpp
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# WF progress bar test
$(WF_PROGRESS_TEST_TARGET): tests/test_wf_progress.cpp
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# Sanity integration example
$(SANITY_INTEGRATION_TARGET): tools/sanity_integration_example.cpp $(OBJ_DIR)/sanity.o $(OBJ_DIR)/sim_data.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

$(HYBRID_PPO_TEST_TARGET): tests/test_hybrid_ppo.cpp $(OBJ_DIR)/strategy_hybrid_ppo.o $(OBJ_DIR)/base_strategy.o $(OBJ_DIR)/ml/model_registry_ts.o $(OBJ_DIR)/ml/ts_model.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# TorchScript test targets
TS_TEST_TARGET = $(BUILD_DIR)/test_torchscript
TS_PARITY_TEST_TARGET = $(BUILD_DIR)/test_ts_parity
FEATURE_BUILDER_TEST_TARGET = $(BUILD_DIR)/test_feature_builder

$(TS_TEST_TARGET): tools/test_torchscript.cpp $(OBJ_DIR)/ml/ts_model.o $(OBJ_DIR)/ml/model_registry_ts.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

$(TS_PARITY_TEST_TARGET): tests/test_ts_parity.cpp $(OBJ_DIR)/ml/ts_model.o $(OBJ_DIR)/ml/model_registry_ts.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

$(FEATURE_BUILDER_TEST_TARGET): tests/test_feature_builder.cpp $(OBJ_DIR)/feature_builder.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Leverage protection test targets
$(LEVERAGE_TEST_TARGET): tests/test_leverage.cpp
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

$(BUILDER_GUARD_TEST_TARGET): tests/test_builder_guard.cpp
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

$(LEVERAGE_EXAMPLE_TARGET): examples/leverage_guarded_example.cpp $(OBJ_DIR)/feature_feeder_guarded.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

$(TFA_CORRUPTION_TEST_TARGET): tests/test_tfa_corruption.cpp
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# Python module
$(PYTHON_MODULE_TARGET): bindings/featurebridge.cpp
	@echo "Building Python module $@"
	$(CXX) -std=c++17 -O3 -shared -fPIC $(PYTHON_INCLUDES) -o $@ $< $(PYTHON_LIBS)

# Object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<"
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@


# Clean targets
clean:
	@echo "Cleaning object files"
	rm -rf $(OBJ_DIR)

clean-all:
	@echo "Cleaning all build artifacts"
	rm -rf $(BUILD_DIR)

# Install dependencies (macOS)
install-deps:
	@echo "Installing dependencies..."
	brew install sqlite3

# Test compilation
test-compile: $(OBJECTS)
	@echo "Compilation test passed"

# Debug build
debug: CXXFLAGS += -DDEBUG -g3
debug: $(MAIN_TARGET) $(POLY_TARGET)

# Sanitized debug build (for malloc corruption detection)
debug-sanitized: CXXFLAGS = $(CXXFLAGS_DEBUG) -fstack-protector-strong -D_GLIBCXX_ASSERTIONS
debug-sanitized: LIBS = $(LIBS_DEBUG)
debug-sanitized: clean $(MAIN_TARGET)

# Release build (safe optimization for LibTorch compatibility)
release: CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -DNDEBUG -DSENTIO_WITH_LIBTORCH -fstack-protector-strong
release: LIBS = -lsqlite3 -lcurl -L/opt/homebrew/lib -lcctz -L/opt/homebrew/lib/python3.12/site-packages/torch/lib -ltorch -ltorch_cpu -lc10
release: clean $(MAIN_TARGET) $(POLY_TARGET)

# Aggressive release build (may cause malloc corruption with LibTorch)
release-fast: CXXFLAGS += -O3 -DNDEBUG -ffast-math -flto -march=native
release-fast: clean $(MAIN_TARGET) $(POLY_TARGET)

# Run tests
test: $(MAIN_TARGET) $(TEST_TARGET)
	@echo "Running basic tests..."
	@$(MAIN_TARGET) backtest QQQ --strategy MarketMaking 2>/dev/null | grep -E "(Final Equity|Total Return|Sharpe Ratio|Total Fills)" || echo "Test completed"
	@echo "Running signal emission test..."
	@$(TEST_TARGET)

# Run pipeline test
pipeline-test: $(PIPELINE_TEST_TARGET)
	@echo "Running pipeline test..."
	@$(PIPELINE_TEST_TARGET)

# Run audit test
audit-test: $(AUDIT_TEST_TARGET)
	@echo "Running audit replay test..."
	@$(AUDIT_TEST_TARGET)

# Run sanity test
sanity-test: $(SANITY_TEST_TARGET)
	@echo "Running sanity end-to-end test..."
	@$(SANITY_TEST_TARGET)

# Run sanity integration example
sanity-integration: $(SANITY_INTEGRATION_TARGET)
	@echo "Running sanity integration example..."
	@$(SANITY_INTEGRATION_TARGET)

# Run TorchScript test
test-torchscript: $(TS_TEST_TARGET)
	@echo "Running TorchScript test..."
	@$(TS_TEST_TARGET)

# Run TorchScript parity test
test-ts-parity: $(TS_PARITY_TEST_TARGET)
	@echo "Running TorchScript parity test..."
	@$(TS_PARITY_TEST_TARGET)

# Run FeatureBuilder test
test-feature-builder: $(FEATURE_BUILDER_TEST_TARGET)
	@echo "Running FeatureBuilder test..."
	@$(FEATURE_BUILDER_TEST_TARGET)

# Run all ML tests
test-ml: test-torchscript test-ts-parity test-feature-builder
	@echo "All ML tests completed"

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build all executables (default)"
	@echo "  clean        - Remove object files"
	@echo "  clean-all    - Remove all build artifacts"
	@echo "  debug        - Build with debug flags"
	@echo "  release      - Build with release optimization"
	@echo "  test         - Run basic tests"
	@echo "  pipeline-test - Run pipeline test"
	@echo "  audit-test - Run audit replay test"
	@echo "  sanity-test - Run sanity end-to-end test"
	@echo "  sanity-integration - Run sanity integration example"
	@echo "  test-torchscript - Run TorchScript test"
	@echo "  test-ts-parity - Run TorchScript parity test"
	@echo "  test-feature-builder - Run FeatureBuilder test"
	@echo "  test-ml - Run all ML tests"
	@echo "  test-compile - Test compilation only"
	@echo "  install-deps - Install dependencies (macOS)"
	@echo "  help         - Show this help"

# Phony targets
.PHONY: all clean clean-all debug release test pipeline-test audit-test sanity-test sanity-integration test-torchscript test-ts-parity test-feature-builder test-ml test-compile install-deps help
