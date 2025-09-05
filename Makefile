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
SOURCES = $(filter-out $(SRC_DIR)/poly_fetch_main.cpp $(SRC_DIR)/test_rth.cpp, $(wildcard $(SRC_DIR)/*.cpp))
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Main targets
MAIN_TARGET = $(BUILD_DIR)/sentio_cli
POLY_TARGET = $(BUILD_DIR)/poly_fetch
LIBRARY_TARGET = $(BUILD_DIR)/libsentio.a
DIAGNOSTIC_TARGET = $(BUILD_DIR)/signal_diagnostics
TEST_TARGET = $(BUILD_DIR)/test_sma_cross
INTEGRATION_TARGET = $(BUILD_DIR)/integration_example

# Default target
all: $(MAIN_TARGET) $(POLY_TARGET) $(DIAGNOSTIC_TARGET) $(TEST_TARGET) $(INTEGRATION_TARGET)

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

# Diagnostic tool
$(DIAGNOSTIC_TARGET): tools/signal_diagnostics.cpp $(OBJ_DIR)/signal_gate.o $(OBJ_DIR)/strategy_sma_cross.o $(OBJ_DIR)/signal_engine.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Unit test
$(TEST_TARGET): tests/test_sma_cross_emit.cpp $(OBJ_DIR)/signal_gate.o $(OBJ_DIR)/strategy_sma_cross.o $(OBJ_DIR)/signal_engine.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Integration example
$(INTEGRATION_TARGET): tools/integration_example.cpp $(OBJ_DIR)/signal_gate.o $(OBJ_DIR)/strategy_sma_cross.o $(OBJ_DIR)/signal_engine.o
	@echo "Linking $@"
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<"
	@mkdir -p $(OBJ_DIR)
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

# Run diagnostics
diagnose: $(DIAGNOSTIC_TARGET)
	@echo "Running signal diagnostics..."
	@$(DIAGNOSTIC_TARGET)

# Run integration example
integration: $(INTEGRATION_TARGET)
	@echo "Running integration example..."
	@$(INTEGRATION_TARGET)

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build all executables (default)"
	@echo "  clean        - Remove object files"
	@echo "  clean-all    - Remove all build artifacts"
	@echo "  debug        - Build with debug flags"
	@echo "  release      - Build with release optimization"
	@echo "  test         - Run basic tests"
	@echo "  diagnose     - Run signal diagnostics"
	@echo "  integration  - Run integration example"
	@echo "  test-compile - Test compilation only"
	@echo "  install-deps - Install dependencies (macOS)"
	@echo "  help         - Show this help"

# Phony targets
.PHONY: all clean clean-all debug release test diagnose integration test-compile install-deps help
