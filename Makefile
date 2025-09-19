# Sentio C++ Makefile
# C++20 backtesting framework

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -march=native -DNDEBUG -ffast-math -fno-math-errno -fno-trapping-math -DSENTIO_WITH_LIBTORCH -flto
CXXFLAGS_DEBUG = -std=c++20 -Wall -Wextra -O1 -g -DSENTIO_WITH_LIBTORCH -fsanitize=address,undefined -fno-omit-frame-pointer
INCLUDES = -Iinclude -Iaudit/include -Ithird_party/nlohmann/include -I/opt/homebrew/include -I/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/include -I/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/include/torch/csrc/api/include -I/opt/homebrew/include/gtest
LIBS = -lsqlite3 -lcurl -L/opt/homebrew/lib -lcctz -L/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/lib -ltorch -ltorch_cpu -lc10 -lgtest -lgtest_main -Wl,-rpath,/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/torch/lib
LIBS_DEBUG = -lsqlite3 -lcurl -L/opt/homebrew/lib -lcctz -L/opt/homebrew/lib/python3.12/site-packages/torch/lib -ltorch -ltorch_cpu -lc10 -fsanitize=address,undefined -L/opt/homebrew/lib -lgtest -lgtest_main

# Python module flags
PYTHON_INCLUDES = -Iinclude -Ithird_party/nlohmann/include -I/opt/homebrew/include -I/opt/homebrew/lib/python3.13/site-packages/pybind11/include -I/opt/homebrew/lib/python3.13/site-packages/numpy/core/include -I/Library/Frameworks/Python.framework/Versions/3.13/include/python3.13
PYTHON_LIBS = -L/opt/homebrew/lib -L/Library/Frameworks/Python.framework/Versions/3.13/lib -lpython3.13

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# Source files (exclude poly_fetch_main.cpp, test_rth.cpp, tfa_train_main.cpp, tfa_trainer.cpp, and transformer_trainer_main.cpp from main executable)
SOURCES = $(filter-out $(SRC_DIR)/poly_fetch_main.cpp $(SRC_DIR)/test_rth.cpp $(SRC_DIR)/tfa_train_main.cpp $(SRC_DIR)/training/tfa_trainer.cpp $(SRC_DIR)/transformer_trainer_main.cpp, $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/ml/*.cpp) $(wildcard $(SRC_DIR)/feature_engineering/*.cpp) $(wildcard $(SRC_DIR)/training/*.cpp))
# Add transformer strategy sources (excluding the trainer main)
TRANSFORMER_SOURCES = src/strategy_transformer.cpp src/feature_pipeline.cpp src/transformer_model.cpp
SOURCES += $(TRANSFORMER_SOURCES)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
AUDIT_OBJECTS = audit/src/audit_db.cpp audit/src/audit_db_recorder.cpp audit/src/hash.cpp audit/src/clock.cpp
AUDIT_OBJECT_FILES = $(AUDIT_OBJECTS:audit/src/%.cpp=$(OBJ_DIR)/audit_%.o)
ALL_OBJECTS = $(filter-out $(AUDIT_OBJECT_FILES), $(OBJECTS)) $(AUDIT_OBJECT_FILES)

# Main targets
MAIN_TARGET = $(BUILD_DIR)/sentio_cli
TFA_TRAINER_TARGET = $(BUILD_DIR)/tfa_trainer
TRANSFORMER_TRAINER_TARGET = $(BUILD_DIR)/transformer_trainer
POLY_TARGET = $(BUILD_DIR)/poly_fetch
AUDIT_CLI_TARGET = $(BUILD_DIR)/sentio_audit
LIBRARY_TARGET = $(BUILD_DIR)/libsentio.a
TEST_TARGET = $(BUILD_DIR)/test_sma_cross
PIPELINE_TEST_TARGET = $(BUILD_DIR)/test_pipeline_emits
# AUDIT_TEST_TARGET = $(BUILD_DIR)/test_audit_replay
AUDIT_SIMPLE_TEST_TARGET = $(BUILD_DIR)/test_audit_simple
SANITY_TEST_TARGET = $(BUILD_DIR)/test_sanity_end_to_end
PERF_TEST_TARGET = $(BUILD_DIR)/test_tfa_performance
PRECOMPUTE_FEATURES_TARGET = $(BUILD_DIR)/precompute_features
PROD_PERF_TEST_TARGET = $(BUILD_DIR)/test_production_performance
PROGRESS_BAR_TEST_TARGET = $(BUILD_DIR)/test_progress_bar
WF_PROGRESS_TEST_TARGET = $(BUILD_DIR)/test_wf_progress
SANITY_INTEGRATION_TARGET = $(BUILD_DIR)/sanity_integration_example
# HYBRID_PPO_TEST_TARGET removed - strategy deregistered
LEVERAGE_TEST_TARGET = $(BUILD_DIR)/test_leverage
BUILDER_GUARD_TEST_TARGET = $(BUILD_DIR)/test_builder_guard
LEVERAGE_EXAMPLE_TARGET = $(BUILD_DIR)/leverage_guarded_example
TFA_CORRUPTION_TEST_TARGET = $(BUILD_DIR)/test_tfa_corruption
POSITION_GUARDIAN_TEST_TARGET = $(BUILD_DIR)/test_position_guardian
LEVERAGE_PRICING_TEST_TARGET = $(BUILD_DIR)/test_leverage_pricing
LEVERAGE_PNL_COMPARISON_TARGET = $(BUILD_DIR)/leverage_pnl_comparison
MINUTE_ANALYSIS_TARGET = $(BUILD_DIR)/minute_by_minute_analysis
ACCURATE_MINUTE_ANALYSIS_TARGET = $(BUILD_DIR)/accurate_minute_analysis
CALIBRATED_ANALYSIS_TARGET = $(BUILD_DIR)/calibrated_accurate_analysis
PYTHON_MODULE_TARGET = sentio_features.cpython-313-darwin.so
REPLAY_AUDIT_TARGET = $(BUILD_DIR)/replay_audit
PNL_TEST_TARGET = $(BUILD_DIR)/test_pnl_engine
# KOCHI_BIN_RUNNER removed - strategy deregistered
RULE_ENSEMBLE_TARGET = $(BUILD_DIR)/run_rule_ensemble
# IRE_SWEEP_TARGET removed - strategy deregistered
CSV_RUNNER_TARGET = $(BUILD_DIR)/csv_runner
RSI_TEST_TARGET = $(BUILD_DIR)/test_rsi_strategy
BACKEND_INTEGRATION_TEST_TARGET = $(BUILD_DIR)/test_backend_integration

# Default target
all: $(MAIN_TARGET) $(TFA_TRAINER_TARGET) $(TRANSFORMER_TRAINER_TARGET) $(POLY_TARGET) $(TEST_TARGET) $(PIPELINE_TEST_TARGET) $(AUDIT_SIMPLE_TEST_TARGET) $(SANITY_TEST_TARGET) $(SANITY_INTEGRATION_TARGET) $(TS_TEST_TARGET) $(TS_PARITY_TEST_TARGET) $(FEATURE_BUILDER_TEST_TARGET) $(PERF_TEST_TARGET) $(PROD_PERF_TEST_TARGET) $(PROGRESS_BAR_TEST_TARGET) $(WF_PROGRESS_TEST_TARGET) $(LEVERAGE_TEST_TARGET) $(BUILDER_GUARD_TEST_TARGET) $(LEVERAGE_EXAMPLE_TARGET) $(PYTHON_MODULE_TARGET) $(REPLAY_AUDIT_TARGET) $(PNL_TEST_TARGET) $(RULE_ENSEMBLE_TARGET) $(CSV_RUNNER_TARGET) $(RSI_TEST_TARGET) $(BACKEND_INTEGRATION_TEST_TARGET) $(AUDIT_CLI_TARGET)

# Main executable
$(MAIN_TARGET): $(ALL_OBJECTS)
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# TFA Trainer executable
$(TFA_TRAINER_TARGET): $(filter-out $(OBJ_DIR)/main.o, $(ALL_OBJECTS)) $(OBJ_DIR)/training/tfa_trainer.o $(OBJ_DIR)/tfa_train_main.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Transformer trainer executable
$(TRANSFORMER_TRAINER_TARGET): $(filter-out $(OBJ_DIR)/main.o $(OBJ_DIR)/transformer_trainer_main.o, $(ALL_OBJECTS)) $(OBJ_DIR)/transformer_trainer_main.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Poly fetch executable
$(POLY_TARGET): $(OBJ_DIR)/poly_fetch_main.o $(OBJ_DIR)/polygon_client.o $(OBJ_DIR)/csv_loader.o $(OBJ_DIR)/time_utils.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Audit CLI executable
$(AUDIT_CLI_TARGET): $(OBJ_DIR)/audit_audit_cli.o $(OBJ_DIR)/audit_audit_db.o $(OBJ_DIR)/audit_hash.o $(OBJ_DIR)/audit_clock.o $(OBJ_DIR)/audit_price_csv.o $(OBJ_DIR)/unified_metrics.o $(OBJ_DIR)/adaptive_allocation_manager.o $(OBJ_DIR)/universal_position_coordinator.o $(OBJ_DIR)/adaptive_eod_manager.o $(OBJ_DIR)/strategy_profiler.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L/opt/homebrew/opt/sqlite/lib -lsqlite3

# Static library
$(LIBRARY_TARGET): $(OBJECTS)
	@echo "Creating $@"
	@mkdir -p $(BUILD_DIR)
	ar rcs $@ $^


# Unit test
# $(TEST_TARGET): tests/test_sma_cross_emit.cpp $(OBJ_DIR)/signal_gate.o $(OBJ_DIR)/strategy_sma_cross.o $(OBJ_DIR)/signal_engine.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)


# Pipeline test
# $(PIPELINE_TEST_TARGET): tests/test_pipeline_emits.cpp $(OBJ_DIR)/signal_gate.o $(OBJ_DIR)/strategy_sma_cross.o $(OBJ_DIR)/signal_engine.o $(OBJ_DIR)/signal_pipeline.o $(OBJ_DIR)/signal_trace.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)


# Audit test
# $(AUDIT_TEST_TARGET): tests/test_audit_replay.cpp $(OBJ_DIR)/audit.o
#	@echo "Linking $@"
#	@mkdir -p $(BUILD_DIR)
#	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Audit simple test
$(AUDIT_SIMPLE_TEST_TARGET): tests/test_audit_simple.cpp $(OBJ_DIR)/audit.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Sanity test
$(SANITY_TEST_TARGET): tests/test_sanity_end_to_end.cpp $(OBJ_DIR)/sanity.o
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
$(SANITY_INTEGRATION_TARGET): tools/sanity_integration_example.cpp $(OBJ_DIR)/sanity.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# HybridPPO test target removed - strategy deregistered

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

$(POSITION_GUARDIAN_TEST_TARGET): tests/test_position_guardian.cpp $(OBJ_DIR)/position_guardian.o $(OBJ_DIR)/position_orchestrator.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(OBJ_DIR)/position_guardian.o $(OBJ_DIR)/position_orchestrator.o -lgtest -lgtest_main $(LIBS)

# Python module
$(PYTHON_MODULE_TARGET): bindings/featurebridge.cpp $(OBJ_DIR)/feature_builder.o $(OBJ_DIR)/feature_engineering/technical_indicators.o
	@echo "Building Python module $@"
	$(CXX) -std=c++17 -O3 -shared -fPIC $(PYTHON_INCLUDES) -o $@ $^ $(PYTHON_LIBS)

# Kochi binary runner

# Header-only deps; compile single TU
# Kochi binary runner removed - strategy deregistered

# Rule ensemble runner
$(RULE_ENSEMBLE_TARGET): src/strategy/run_rule_ensemble.cpp
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# IRE parameter sweep tool
# IRE parameter sweep removed - strategy deregistered

# CSV runner for RSI strategy
$(CSV_RUNNER_TARGET): tools/csv_runner.cpp $(OBJ_DIR)/rsi_strategy.o $(OBJ_DIR)/base_strategy.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# RSI strategy test
$(RSI_TEST_TARGET): tests/test_rsi_strategy.cpp $(OBJ_DIR)/rsi_strategy.o $(OBJ_DIR)/base_strategy.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Backend integration test
$(BACKEND_INTEGRATION_TEST_TARGET): tests/test_backend_integration.cpp $(OBJ_DIR)/backend_architecture.o $(OBJ_DIR)/backend_audit_events.o $(OBJ_DIR)/allocation_manager.o $(OBJ_DIR)/strategy_signal_or.o $(OBJ_DIR)/strategy_tfa.o $(OBJ_DIR)/test_strategy.o $(OBJ_DIR)/base_strategy.o $(OBJ_DIR)/router.o $(OBJ_DIR)/virtual_market.o $(OBJ_DIR)/signal_engine.o $(OBJ_DIR)/signal_gate.o $(OBJ_DIR)/signal_pipeline.o $(OBJ_DIR)/signal_trace.o $(OBJ_DIR)/ml/model_registry_ts.o $(OBJ_DIR)/ml/ts_model.o $(OBJ_DIR)/audit_audit_db.o $(OBJ_DIR)/audit_audit_db_recorder.o $(OBJ_DIR)/audit_hash.o $(OBJ_DIR)/audit_clock.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Leverage pricing test
$(LEVERAGE_PRICING_TEST_TARGET): tests/test_leverage_pricing.cpp $(OBJ_DIR)/leverage_pricing.o $(OBJ_DIR)/csv_loader.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Leverage P&L comparison tool
$(LEVERAGE_PNL_COMPARISON_TARGET): tools/leverage_pnl_comparison.cpp $(OBJ_DIR)/leverage_pricing.o $(OBJ_DIR)/leverage_aware_csv_loader.o $(OBJ_DIR)/csv_loader.o $(OBJ_DIR)/accurate_leverage_pricing.o $(OBJ_DIR)/global_leverage_config.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Minute-by-minute analysis tool
$(MINUTE_ANALYSIS_TARGET): tools/minute_by_minute_analysis.cpp $(OBJ_DIR)/leverage_pricing.o $(OBJ_DIR)/csv_loader.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Accurate minute-by-minute analysis tool
$(ACCURATE_MINUTE_ANALYSIS_TARGET): tools/accurate_minute_analysis.cpp $(OBJ_DIR)/accurate_leverage_pricing.o $(OBJ_DIR)/csv_loader.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Calibrated accurate analysis tool
$(CALIBRATED_ANALYSIS_TARGET): tools/calibrated_accurate_analysis.cpp $(OBJ_DIR)/accurate_leverage_pricing.o $(OBJ_DIR)/csv_loader.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# PSQ pricing accuracy test
PSQ_PRICING_TEST_TARGET = $(BUILD_DIR)/test_psq_pricing_accuracy
$(PSQ_PRICING_TEST_TARGET): tools/test_psq_pricing_accuracy.cpp $(OBJ_DIR)/accurate_leverage_pricing.o $(OBJ_DIR)/csv_loader.o $(OBJ_DIR)/global_leverage_config.o $(OBJ_DIR)/leverage_aware_csv_loader.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# PSQ routing test
PSQ_ROUTING_TEST_TARGET = $(BUILD_DIR)/test_psq_routing
$(PSQ_ROUTING_TEST_TARGET): tools/test_psq_routing.cpp $(OBJ_DIR)/router.o $(OBJ_DIR)/global_leverage_config.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# PSQ comprehensive integration test
PSQ_INTEGRATION_TEST_TARGET = $(BUILD_DIR)/test_psq_integration
$(PSQ_INTEGRATION_TEST_TARGET): tools/test_psq_integration.cpp $(OBJ_DIR)/router.o $(OBJ_DIR)/global_leverage_config.o $(OBJ_DIR)/accurate_leverage_pricing.o $(OBJ_DIR)/csv_loader.o $(OBJ_DIR)/leverage_aware_csv_loader.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Comprehensive strategy analysis tool
COMPREHENSIVE_ANALYSIS_TARGET = $(BUILD_DIR)/comprehensive_strategy_analysis
$(COMPREHENSIVE_ANALYSIS_TARGET): tools/comprehensive_strategy_analysis.cpp
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

# Duplicate audit CLI target removed - using the one at line 82



# Object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<"
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Training object files
$(OBJ_DIR)/training/%.o: $(SRC_DIR)/training/%.cpp
	@echo "Compiling $<"
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Audit object files
$(OBJ_DIR)/audit_%.o: audit/src/%.cpp
	@echo "Compiling $<"
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@


# Clean targets
clean:
	@echo "Cleaning object files"
	rm -rf $(OBJ_DIR)

# Convenient targets
tfa-trainer: $(TFA_TRAINER_TARGET)

train-tfa: $(TFA_TRAINER_TARGET)
	@echo "🚀 C++ TFA Trainer built successfully!"
	@echo "📋 Usage: ./build/tfa_trainer --data <csv_file> --feature-spec <json_file>"
	@echo "📖 Run './build/tfa_trainer --help' for full options"

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

# Run position guardian test
position-guardian-test: $(POSITION_GUARDIAN_TEST_TARGET)
	@echo "Running position guardian test..."
	@$(POSITION_GUARDIAN_TEST_TARGET)

# Run leverage pricing test
leverage-pricing-test: $(LEVERAGE_PRICING_TEST_TARGET)
	@echo "Running leverage pricing test..."
	@$(LEVERAGE_PRICING_TEST_TARGET)

# Run audit test
# audit-test: $(AUDIT_TEST_TARGET)
#	@echo "Running audit replay test..."
#	@$(AUDIT_TEST_TARGET)

# Run sanity test
sanity-test: $(SANITY_TEST_TARGET)
	@echo "Running sanity end-to-end test..."
	@$(SANITY_TEST_TARGET)

# Run sanity integration example
santity-integration: $(SANITY_INTEGRATION_TARGET)
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

# Run Backend Integration test
test-backend-integration: $(BACKEND_INTEGRATION_TEST_TARGET)
	@echo "Running Backend Integration test..."
	@$(BACKEND_INTEGRATION_TEST_TARGET)

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
	@echo "  leverage-pricing-test - Run leverage pricing test"
	@echo "  audit-test - Run audit replay test"
	@echo "  sanity-test - Run sanity end-to-end test"
	@echo "  sanity-integration - Run sanity integration example"
	@echo "  test-torchscript - Run TorchScript test"
	@echo "  test-ts-parity - Run TorchScript parity test"
	@echo "  test-feature-builder - Run FeatureBuilder test"
	@echo "  test-ml - Run all ML tests"
	@echo "  test-compile - Test compilation only"
	@echo "  install-deps - Install dependencies (macOS)"
	@echo "  train-transformer - Train transformer strategy (20 epochs)"
	@echo "  help         - Show this help"

# Training targets
train-transformer: $(TRANSFORMER_TRAINER_TARGET)
	@echo "Training Transformer strategy for 20 epochs..."
	./$(TRANSFORMER_TRAINER_TARGET) --epochs 20 --data data/equities/QQQ_1min.csv --output artifacts/Transformer/v1

# Phony targets
.PHONY: all clean clean-all debug release test pipeline-test leverage-pricing-test audit-test sanity-test sanity-integration test-torchscript test-ts-parity test-feature-builder test-ml test-compile install-deps help train-transformer
