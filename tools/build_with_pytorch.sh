#!/bin/bash

# Build script for rad_ml with PyTorch support
# This script helps set up and build the project with PyTorch integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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
    print_status "Detecting PyTorch installation..."

    # Check common PyTorch installation paths
    PYTORCH_PATHS=(
        "/usr/local/libtorch"
        "/opt/libtorch"
        "/usr/local/opt/libtorch"
        "/opt/homebrew/opt/libtorch"
        "$HOME/libtorch"
        "$HOME/pytorch"
    )

    for path in "${PYTORCH_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/include/torch/torch.h" ]; then
            print_success "Found PyTorch at: $path"
            export PYTORCH_ROOT="$path"
            return 0
        fi
    done

    # Check if PyTorch is installed via pip (for Python bindings)
    if command_exists python3; then
        if python3 -c "import torch; print(torch.__file__)" >/dev/null 2>&1; then
            PYTORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
            if [ -d "$PYTORCH_PATH" ]; then
                print_success "Found PyTorch via Python at: $PYTORCH_PATH"
                export PYTORCH_ROOT="$PYTORCH_PATH"
                return 0
            fi
        fi
    fi

    print_warning "PyTorch not found in common locations"
    return 1
}

# Function to download PyTorch
download_pytorch() {
    print_status "Downloading PyTorch..."

    # Detect OS and architecture
    OS=$(uname -s)
    ARCH=$(uname -m)

    if [ "$OS" = "Darwin" ]; then
        if [ "$ARCH" = "arm64" ]; then
            URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip"
        else
            URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip"
        fi
    elif [ "$OS" = "Linux" ]; then
        if [ "$ARCH" = "x86_64" ]; then
            URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
        else
            print_error "Unsupported architecture: $ARCH"
            return 1
        fi
    else
        print_error "Unsupported OS: $OS"
        return 1
    fi

    # Create libtorch directory
    mkdir -p libtorch

    # Download and extract
    print_status "Downloading from: $URL"
    curl -L "$URL" -o libtorch.zip

    if [ $? -eq 0 ]; then
        print_status "Extracting PyTorch..."
        unzip -q libtorch.zip -d .
        rm libtorch.zip

        # Find the extracted directory
        for dir in libtorch-*; do
            if [ -d "$dir" ]; then
                mv "$dir" libtorch
                break
            fi
        done

        export PYTORCH_ROOT="$PWD/libtorch"
        print_success "PyTorch downloaded and extracted to: $PYTORCH_ROOT"
        return 0
    else
        print_error "Failed to download PyTorch"
        return 1
    fi
}

# Function to build the project
build_project() {
    print_status "Building rad_ml with PyTorch support..."

    # Create build directory
    mkdir -p build_pytorch
    cd build_pytorch

    # Configure with CMake
    print_status "Configuring with CMake..."
    cmake .. \
        -DENABLE_PYTORCH=ON \
        -DBUILD_TESTING=ON \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

    if [ $? -ne 0 ]; then
        print_error "CMake configuration failed"
        return 1
    fi

    # Build
    print_status "Building..."
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

    if [ $? -ne 0 ]; then
        print_error "Build failed"
        return 1
    fi

    print_success "Build completed successfully!"

    # Run tests
    print_status "Running tests..."
    ctest --output-on-failure

    cd ..
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -d, --download      Download PyTorch if not found"
    echo "  -b, --build-only    Only build, don't download"
    echo "  -t, --test-only     Only run tests"
    echo ""
    echo "Environment variables:"
    echo "  PYTORCH_ROOT        Path to PyTorch installation"
    echo "  CMAKE_BUILD_TYPE    Build type (Debug, Release, etc.)"
}

# Main script
main() {
    print_status "rad_ml PyTorch Build Script"
    print_status "=========================="

    # Parse command line arguments
    DOWNLOAD_PYTORCH=false
    BUILD_ONLY=false
    TEST_ONLY=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -d|--download)
                DOWNLOAD_PYTORCH=true
                shift
                ;;
            -b|--build-only)
                BUILD_ONLY=true
                shift
                ;;
            -t|--test-only)
                TEST_ONLY=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Check if we're in the right directory
    if [ ! -f "CMakeLists.txt" ]; then
        print_error "CMakeLists.txt not found. Please run this script from the project root."
        exit 1
    fi

    # Detect or download PyTorch
    if [ "$DOWNLOAD_PYTORCH" = true ]; then
        # Force download even if detected
        print_status "Forcing PyTorch download..."
        download_pytorch
    elif ! detect_pytorch; then
        print_error "PyTorch not found. Use -d to download automatically or set PYTORCH_ROOT."
        exit 1
    fi

    # Build the project
    if [ "$TEST_ONLY" = false ]; then
        build_project
    fi

    # Run tests only
    if [ "$TEST_ONLY" = true ]; then
        if [ -d "build_pytorch" ]; then
            cd build_pytorch
            print_status "Running tests..."
            ctest --output-on-failure
            cd ..
        else
            print_error "Build directory not found. Run build first."
            exit 1
        fi
    fi

    print_success "PyTorch integration setup completed!"
    print_status "You can now use the rad_ml framework with PyTorch support."
}

# Run main function
main "$@"
