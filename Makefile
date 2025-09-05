# Sentio C++ Makefile
# C++20 backtesting framework

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -g
INCLUDES = -Iinclude -Ithird_party/nlohmann/include -I/opt/homebrew/include
LIBS = -lsqlite3 -lcurl -L/opt/homebrew/lib -lcctz

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# Source files (exclude poly_fetch_main.cpp and test_rth.cpp from main executable)
SOURCES = $(filter-out $(SRC_DIR)/poly_fetch_main.cpp $(SRC_DIR)/test_rth.cpp, $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/ml/*.cpp))
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
SANITY_INTEGRATION_TARGET = $(BUILD_DIR)/sanity_integration_example
HYBRID_PPO_TEST_TARGET = $(BUILD_DIR)/test_hybrid_ppo

# Default target
all: $(MAIN_TARGET) $(POLY_TARGET) $(TEST_TARGET) $(PIPELINE_TEST_TARGET) $(AUDIT_TEST_TARGET) $(AUDIT_SIMPLE_TEST_TARGET) $(SANITY_TEST_TARGET) $(SANITY_INTEGRATION_TARGET) $(HYBRID_PPO_TEST_TARGET)

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

# Sanity integration example
$(SANITY_INTEGRATION_TARGET): tools/sanity_integration_example.cpp $(OBJ_DIR)/sanity.o $(OBJ_DIR)/sim_data.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

$(HYBRID_PPO_TEST_TARGET): tests/test_hybrid_ppo.cpp $(OBJ_DIR)/strategy_hybrid_ppo.o $(OBJ_DIR)/ml/model_registry.o $(OBJ_DIR)/ml/onnx_model.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

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

# Release build
release: CXXFLAGS += -O3 -DNDEBUG
release: clean $(MAIN_TARGET) $(POLY_TARGET)

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
	@echo "  test-compile - Test compilation only"
	@echo "  install-deps - Install dependencies (macOS)"
	@echo "  help         - Show this help"

# Phony targets
.PHONY: all clean clean-all debug release test pipeline-test audit-test sanity-test sanity-integration test-compile install-deps help
