#!/bin/bash

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

print_status "LibTorch Standalone Test Script"
print_status "================================"

# Check if we're in the right directory
if [ ! -f "test_libtorch_standalone.cpp" ]; then
    print_error "test_libtorch_standalone.cpp not found. Please run this script from the project root."
    exit 1
fi

# Check if CMake is available
if ! command -v cmake &> /dev/null; then
    print_error "CMake is not installed. Please install CMake first."
    exit 1
fi

# Check if we have a libtorch installation
PYTORCH_ROOT=""
if [ -d "libtorch" ]; then
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
    print_warning "No libtorch installation found. Trying to find via CMake..."
fi

# Create build directory
print_status "Creating build directory..."
mkdir -p build_libtorch_test
cd build_libtorch_test

# Copy the test files to build directory
cp ../test_libtorch_standalone.cpp .
cp ../CMakeLists_test_libtorch.txt CMakeLists.txt

# Configure with CMake
print_status "Configuring with CMake..."
if [ -n "$PYTORCH_ROOT" ]; then
    cmake -DCMAKE_PREFIX_PATH="$PYTORCH_ROOT" .
else
    cmake .
fi

if [ $? -ne 0 ]; then
    print_error "CMake configuration failed"
    print_status "Trying alternative approach..."

    # Try with explicit Torch path
    if [ -d "../libtorch" ]; then
        cmake -DCMAKE_PREFIX_PATH="../libtorch" \
            -DTorch_DIR="../libtorch/share/cmake/Torch" .
    else
        print_error "Could not configure CMake. Please ensure libtorch is properly installed."
        exit 1
    fi
fi

# Build the test
print_status "Building test executable..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

if [ $? -ne 0 ]; then
    print_error "Build failed"
    exit 1
fi

print_success "Build completed successfully!"

# Run the test
print_status "Running LibTorch standalone test..."
print_status "=================================="

# Set library path for macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="$PYTORCH_ROOT/lib:$DYLD_LIBRARY_PATH"
    print_status "Set DYLD_LIBRARY_PATH to: $DYLD_LIBRARY_PATH"
fi

./test_libtorch_standalone

if [ $? -eq 0 ]; then
    print_success "LibTorch test completed successfully!"
    print_status "LibTorch is working correctly on your system."
else
    print_error "LibTorch test failed!"
    print_status "Please check the error messages above."
fi

# Clean up test file
if [ -f "test_model.pt" ]; then
    rm test_model.pt
    print_status "Cleaned up test model file"
fi

cd ..
print_status "Test completed. Build directory: build_libtorch_test/"
