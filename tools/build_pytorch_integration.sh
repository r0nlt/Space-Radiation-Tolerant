#!/bin/bash
# PyTorch Integration Build Script
# Author: Rishab Nuguru
# License: AGPL v3 license

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect PyTorch installation
detect_pytorch() {
    print_info "Detecting PyTorch installation..."

    # Check for PyTorch via pip
    if command_exists python3; then
        if python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
            PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
            print_success "Found PyTorch via pip: $PYTORCH_VERSION"
            return 0
        fi
    fi

    # Check for LibTorch installation
    PYTORCH_ROOT=""

    # Check common LibTorch locations
    if [ -d "$PWD/libtorch" ]; then
        PYTORCH_ROOT="$PWD/libtorch"
        print_success "Found local libtorch installation: $PYTORCH_ROOT"
    elif [ -n "$PYTORCH_ROOT" ]; then
        print_success "Using PYTORCH_ROOT: $PYTORCH_ROOT"
    elif [ -d "/usr/local/libtorch" ]; then
        PYTORCH_ROOT="/usr/local/libtorch"
        print_success "Found system libtorch: $PYTORCH_ROOT"
    elif [ -d "/opt/libtorch" ]; then
        PYTORCH_ROOT="/opt/libtorch"
        print_success "Found system libtorch: $PYTORCH_ROOT"
    else
        print_warning "PyTorch not found in common locations"
        return 1
    fi

    return 0
}

# Function to download LibTorch if needed
download_libtorch() {
    print_info "Downloading LibTorch..."

    # Detect OS and architecture
    OS=$(uname -s)
    ARCH=$(uname -m)

    if [ "$OS" = "Darwin" ]; then
        if [ "$ARCH" = "x86_64" ]; then
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip"
        elif [ "$ARCH" = "arm64" ]; then
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip"
        else
            print_error "Unsupported architecture: $ARCH"
            return 1
        fi
    elif [ "$OS" = "Linux" ]; then
        if [ "$ARCH" = "x86_64" ]; then
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
        else
            print_error "Unsupported architecture: $ARCH"
            return 1
        fi
    else
        print_error "Unsupported OS: $OS"
        return 1
    fi

    # Download and extract
    print_info "Downloading from: $LIBTORCH_URL"
    curl -L -o libtorch.zip "$LIBTORCH_URL"
    unzip -q libtorch.zip
    rm libtorch.zip

    PYTORCH_ROOT="$PWD/libtorch"
    print_success "LibTorch downloaded to: $PYTORCH_ROOT"
}

# Function to build the project with PyTorch
build_with_pytorch() {
    print_info "Building project with PyTorch integration..."

    # Clean up any existing build files in current directory
    if [ -f "CMakeCache.txt" ]; then
        print_info "Cleaning up existing build files..."
        rm -f CMakeCache.txt Makefile
        rm -rf CMakeFiles
    fi

    # Create build directory
    mkdir -p build_pytorch
    cd build_pytorch

    # Configure with CMake
    if [ -n "$PYTORCH_ROOT" ]; then
        print_info "Configuring with PyTorch at: $PYTORCH_ROOT"
        cmake -DCMAKE_PREFIX_PATH="$PYTORCH_ROOT" \
              -DENABLE_PYTORCH=ON \
              -DBUILD_PYTHON_BINDINGS=ON \
              -DBUILD_TESTING=ON \
              -DCMAKE_BUILD_TYPE=Release \
              ..
    else
        print_info "Configuring with system PyTorch"
        cmake -DENABLE_PYTORCH=ON \
              -DBUILD_PYTHON_BINDINGS=ON \
              -DBUILD_TESTING=ON \
              -DCMAKE_BUILD_TYPE=Release \
              ..
    fi

    # Build
    print_info "Building project..."
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

    print_success "Build completed successfully"

    # Return to original directory
    cd ..
}

# Function to run PyTorch integration tests
run_pytorch_tests() {
    print_info "Running PyTorch integration tests..."

    cd build_pytorch

    # Run C++ tests
    if [ -f "test/pytorch_integration_test" ]; then
        print_info "Running C++ PyTorch integration test..."
        ./test/pytorch_integration_test
    fi

    # Run Python tests if available
    if [ -f "test/test_pytorch_integration_comprehensive.py" ]; then
        print_info "Running Python PyTorch integration tests..."
        cd test
        python3 test_pytorch_integration_comprehensive.py
        cd ..
    fi

    # Run example
    if [ -f "examples/pytorch_integration_example.py" ]; then
        print_info "Running PyTorch integration example..."
        cd examples
        python3 pytorch_integration_example.py
        cd ..
    fi

    print_success "All PyTorch integration tests completed"

    # Return to original directory
    cd ..
}

# Function to install Python package
install_python_package() {
    print_info "Installing Python package..."

    cd build_pytorch

    # Copy Python files
    mkdir -p rad_ml
    cp ../python/rad_ml/*.py rad_ml/ 2>/dev/null || true
    cp ../python/examples/*.py examples/ 2>/dev/null || true

    # Create __init__.py if it doesn't exist
    if [ ! -f "rad_ml/__init__.py" ]; then
        echo "# PyTorch Integration Package" > rad_ml/__init__.py
    fi

    print_success "Python package installed"

    # Return to original directory
    cd ..
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --detect-only     Only detect PyTorch installation"
    echo "  --download        Download LibTorch if not found"
    echo "  --build-only      Only build the project"
    echo "  --test-only       Only run tests"
    echo "  --install-only    Only install Python package"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Full build and test"
    echo "  $0 --detect-only   # Only detect PyTorch"
    echo "  $0 --download      # Download LibTorch and build"
}

# Main script
main() {
    print_info "=== PyTorch Integration Build Script ==="

    # Parse command line arguments
    DETECT_ONLY=false
    DOWNLOAD_LIBTORCH=false
    BUILD_ONLY=false
    TEST_ONLY=false
    INSTALL_ONLY=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --detect-only)
                DETECT_ONLY=true
                shift
                ;;
            --download)
                DOWNLOAD_LIBTORCH=true
                shift
                ;;
            --build-only)
                BUILD_ONLY=true
                shift
                ;;
            --test-only)
                TEST_ONLY=true
                shift
                ;;
            --install-only)
                INSTALL_ONLY=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Detect PyTorch
    if ! detect_pytorch; then
        if [ "$DOWNLOAD_LIBTORCH" = true ]; then
            download_libtorch
        else
            print_error "PyTorch not found. Use --download to download LibTorch."
            exit 1
        fi
    fi

    if [ "$DETECT_ONLY" = true ]; then
        print_success "PyTorch detection completed"
        exit 0
    fi

    # Build project
    if [ "$BUILD_ONLY" = true ] || [ "$TEST_ONLY" = false ] && [ "$INSTALL_ONLY" = false ]; then
        build_with_pytorch
    fi

    # Install Python package
    if [ "$INSTALL_ONLY" = true ] || [ "$TEST_ONLY" = false ]; then
        install_python_package
    fi

    # Run tests
    if [ "$TEST_ONLY" = true ] || [ "$BUILD_ONLY" = false ] && [ "$INSTALL_ONLY" = false ]; then
        run_pytorch_tests
    fi

    print_success "=== PyTorch Integration Build Completed Successfully ==="
}

# Run main function
main "$@"
