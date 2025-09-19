#!/bin/bash
# build.sh - Main build script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
BUILD_TESTS=ON
BUILD_DOCS=OFF
CLEAN_BUILD=false
INSTALL_DEPS=false
NUM_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -t, --type TYPE         Build type: Debug, Release, RelWithDebInfo (default: Release)
    -j, --jobs NUM          Number of parallel jobs (default: $NUM_JOBS)
    -c, --clean             Clean build directory before building
    --no-tests              Disable test building
    --docs                  Enable documentation building
    --install-deps          Install system dependencies
    --cuda                  Enable CUDA support (if available)
    --profile               Enable profiling build

Examples:
    $0                      # Release build with tests
    $0 -t Debug -c          # Debug build, clean first
    $0 --no-tests --docs    # Release build with docs, no tests
    $0 --install-deps       # Install dependencies and build

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -j|--jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        --no-tests)
            BUILD_TESTS=OFF
            shift
            ;;
        --docs)
            BUILD_DOCS=ON
            shift
            ;;
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --cuda)
            ENABLE_CUDA=ON
            shift
            ;;
        --profile)
            ENABLE_PROFILING=ON
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate build type
case $BUILD_TYPE in
    Debug|Release|RelWithDebInfo|MinSizeRel)
        ;;
    *)
        print_error "Invalid build type: $BUILD_TYPE"
        print_info "Valid types: Debug, Release, RelWithDebInfo, MinSizeRel"
        exit 1
        ;;
esac

print_info "Starting Sentio Transformer Strategy build"
print_info "Build type: $BUILD_TYPE"
print_info "Parallel jobs: $NUM_JOBS"
print_info "Tests enabled: $BUILD_TESTS"
print_info "Docs enabled: $BUILD_DOCS"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

# Function to install dependencies
install_dependencies() {
    print_info "Installing system dependencies..."
    
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            git \
            libtorch-dev \
            libyaml-cpp-dev \
            libgtest-dev \
            libomp-dev \
            python3-dev \
            python3-pip \
            doxygen \
            graphviz \
            clang-format \
            clang-tidy \
            valgrind
            
    elif command -v yum &> /dev/null; then
        # RHEL/CentOS
        sudo yum install -y \
            gcc-c++ \
            cmake3 \
            git \
            yaml-cpp-devel \
            gtest-devel \
            openmp-devel \
            python3-devel \
            doxygen \
            graphviz \
            clang-tools-extra \
            valgrind-devel
            
    elif command -v brew &> /dev/null; then
        # macOS
        brew install \
            cmake \
            yaml-cpp \
            googletest \
            libomp \
            doxygen \
            graphviz \
            clang-format \
            llvm
            
    else
        print_warning "Unknown package manager. Please install dependencies manually."
        print_info "Required: cmake, libtorch, yaml-cpp, googletest, openmp"
    fi
    
    # Install PyTorch C++ (if not found by system package manager)
    if ! pkg-config --exists torch 2>/dev/null; then
        print_info "Installing PyTorch C++ from source..."
        cd /tmp
        if [[ "$OSTYPE" == "darwin"* ]]; then
            wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.0.0.zip
            unzip libtorch-macos-2.0.0.zip
        else
            wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
            unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip
        fi
        sudo cp -r libtorch/include/* /usr/local/include/
        sudo cp -r libtorch/lib/* /usr/local/lib/
        if command -v ldconfig &> /dev/null; then
            sudo ldconfig
        fi
        rm -rf libtorch*
        cd -
    fi
}

# Function to setup build directory
setup_build_dir() {
    if [[ "$CLEAN_BUILD" == true ]] && [[ -d "$BUILD_DIR" ]]; then
        print_info "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
    fi
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
}

# Function to configure CMake
configure_cmake() {
    print_info "Configuring CMake..."
    
    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        -DBUILD_TESTS="$BUILD_TESTS"
        -DBUILD_DOCS="$BUILD_DOCS"
    )
    
    if [[ -n "$ENABLE_CUDA" ]]; then
        CMAKE_ARGS+=(-DENABLE_CUDA="$ENABLE_CUDA")
    fi
    
    if [[ -n "$ENABLE_PROFILING" ]]; then
        CMAKE_ARGS+=(-DENABLE_PROFILING="$ENABLE_PROFILING")
    fi
    
    # Add custom PyTorch path if needed
    if [[ -n "$TORCH_PATH" ]]; then
        CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="$TORCH_PATH")
    fi
    
    cmake "${CMAKE_ARGS[@]}" "$PROJECT_ROOT"
}

# Function to build project
build_project() {
    print_info "Building project with $NUM_JOBS parallel jobs..."
    make -j"$NUM_JOBS"
}

# Function to run tests
run_tests() {
    if [[ "$BUILD_TESTS" == "ON" ]]; then
        print_info "Running tests..."
        ctest --output-on-failure --parallel "$NUM_JOBS"
    fi
}

# Function to build documentation
build_docs() {
    if [[ "$BUILD_DOCS" == "ON" ]]; then
        print_info "Building documentation..."
        make docs
    fi
}

# Function to install project
install_project() {
    if [[ "$BUILD_TYPE" == "Release" ]]; then
        print_info "Installing project..."
        sudo make install
    fi
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    
    if [[ "$INSTALL_DEPS" == true ]]; then
        install_dependencies
    fi
    
    setup_build_dir
    configure_cmake
    build_project
    run_tests
    build_docs
    
    print_success "Build completed successfully!"
    print_info "Build artifacts are in: $BUILD_DIR"
    
    if [[ "$BUILD_TESTS" == "ON" ]]; then
        print_info "Test executable: $BUILD_DIR/transformer_tests"
    fi
    
    if [[ "$BUILD_DOCS" == "ON" ]]; then
        print_info "Documentation: $BUILD_DIR/docs/html/index.html"
    fi
    
    # Show binary sizes
    print_info "Binary sizes:"
    ls -lh "$BUILD_DIR"/{libsentio_transformer.a,train_transformer,backtest_transformer} 2>/dev/null || true
}

# Error handling
trap 'print_error "Build failed at line $LINENO"' ERR

# Run main function
main "$@"
