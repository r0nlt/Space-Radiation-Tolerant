#!/bin/bash

# LibTorch Comprehensive Test Runner
# This script builds and runs all LibTorch integration tests for the rad_ml framework
#
# @author Rishab Nuguru
# @copyright © 2025 Rishab Nuguru
# @license AGPL v3 license

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="build_libtorch_tests"
TEST_RESULTS_DIR="libtorch_test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Test executables
TESTS=(
    "libtorch_resilience_test"
    "libtorch_radiation_integration_test"
    "pytorch_integration_test"
    "libtorch_macos_compatibility_test"
)

# Python tests
PYTHON_TESTS=(
    "libtorch_python_resilience_test.py"
    "libtorch_macos_python_test.py"
)

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

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check if CMake is available
    if ! command -v cmake &> /dev/null; then
        print_error "CMake is not installed or not in PATH"
        exit 1
    fi

    # Check if make is available
    if ! command -v make &> /dev/null; then
        print_error "Make is not installed or not in PATH"
        exit 1
    fi

    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed or not in PATH"
        exit 1
    fi

    # Check if PyTorch is available in Python
    if ! python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" &> /dev/null; then
        print_warning "PyTorch not found in Python environment"
    else
        print_success "PyTorch found in Python environment"

        # Check CUDA availability
        if python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null | grep -q "True"; then
            print_success "CUDA is available"
        else
            print_warning "CUDA not available - will use CPU-only tests"
        fi
    fi

    print_success "Prerequisites check completed"
}

# Function to create build directory
setup_build_environment() {
    print_status "Setting up build environment..."

    # Create build directory
    if [ -d "$BUILD_DIR" ]; then
        print_warning "Build directory already exists, cleaning..."
        rm -rf "$BUILD_DIR"
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Create results directory
    mkdir -p "../$TEST_RESULTS_DIR"

    print_success "Build environment setup completed"
}

# Function to configure CMake
configure_cmake() {
    print_status "Configuring CMake..."

    # Get the path to LibTorch
    LIBTORCH_PATH=""

    # Check common LibTorch installation paths
    if [ -d "/usr/local/libtorch" ]; then
        LIBTORCH_PATH="/usr/local/libtorch"
    elif [ -d "$HOME/libtorch" ]; then
        LIBTORCH_PATH="$HOME/libtorch"
    elif [ -d "../libtorch" ]; then
        LIBTORCH_PATH="../libtorch"
    else
        print_warning "LibTorch not found in common locations"
        print_status "Please set LIBTORCH_PATH environment variable"
        if [ -n "$LIBTORCH_PATH" ]; then
            print_status "Using LIBTORCH_PATH: $LIBTORCH_PATH"
        else
            print_error "LibTorch path not found. Please install LibTorch or set LIBTORCH_PATH"
            exit 1
        fi
    fi

    # Configure CMake
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_PYTORCH=ON \
        -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" \
        -DRAD_ML_PYTORCH_ENABLED=ON \
        -DBUILD_TESTING=ON \
        -DCMAKE_CXX_STANDARD=17 \
        -DUSE_CUDA=OFF

    print_success "CMake configuration completed"
}

# Function to build tests
build_tests() {
    print_status "Building LibTorch tests..."

    # Build all tests
    make -j$(nproc) libtorch_resilience_test libtorch_radiation_integration_test pytorch_integration_test libtorch_macos_compatibility_test

    print_success "Test build completed"
}

# Function to run C++ tests
run_cpp_tests() {
    print_status "Running C++ LibTorch tests..."

    local test_results_file="../$TEST_RESULTS_DIR/cpp_test_results_$TIMESTAMP.txt"
    local all_passed=true

    for test in "${TESTS[@]}"; do
        if [ -f "./$test" ]; then
            print_status "Running $test..."
            echo "=== $test ===" >> "$test_results_file"

            if ./"$test" 2>&1 | tee -a "$test_results_file"; then
                print_success "$test passed"
            else
                print_error "$test failed"
                all_passed=false
            fi

            echo "" >> "$test_results_file"
            echo "========================================" >> "$test_results_file"
            echo "" >> "$test_results_file"
        else
            print_error "Test executable $test not found"
            all_passed=false
        fi
    done

    if [ "$all_passed" = true ]; then
        print_success "All C++ tests completed"
        return 0
    else
        print_error "Some C++ tests failed"
        return 1
    fi
}

# Function to run Python tests
run_python_tests() {
    print_status "Running Python LibTorch tests..."

    local test_results_file="../$TEST_RESULTS_DIR/python_test_results_$TIMESTAMP.txt"
    local all_passed=true

    cd ..  # Go back to test directory

    for test in "${PYTHON_TESTS[@]}"; do
        if [ -f "$test" ]; then
            print_status "Running $test..."
            echo "=== $test ===" >> "$TEST_RESULTS_DIR/$test_results_file"

            if python3 "$test" 2>&1 | tee -a "$TEST_RESULTS_DIR/$test_results_file"; then
                print_success "$test passed"
            else
                print_error "$test failed"
                all_passed=false
            fi

            echo "" >> "$TEST_RESULTS_DIR/$test_results_file"
            echo "========================================" >> "$TEST_RESULTS_DIR/$test_results_file"
            echo "" >> "$TEST_RESULTS_DIR/$test_results_file"
        else
            print_error "Python test $test not found"
            all_passed=false
        fi
    done

    if [ "$all_passed" = true ]; then
        print_success "All Python tests completed"
        return 0
    else
        print_error "Some Python tests failed"
        return 1
    fi
}

# Function to run standalone LibTorch tests
run_standalone_tests() {
    print_status "Running standalone LibTorch tests..."

    local test_results_file="../$TEST_RESULTS_DIR/standalone_test_results_$TIMESTAMP.txt"

    cd ..  # Go back to test directory

    # Run standalone C++ test if available
    if [ -f "test_libtorch_standalone.cpp" ]; then
        print_status "Building standalone C++ test..."
        g++ -std=c++17 -I"$LIBTORCH_PATH/include" -L"$LIBTORCH_PATH/lib" \
            -ltorch -ltorch_cpu -ltorch_cuda -ltorch_cudart \
            test_libtorch_standalone.cpp -o test_libtorch_standalone

        if [ -f "./test_libtorch_standalone" ]; then
            print_status "Running standalone C++ test..."
            echo "=== Standalone C++ Test ===" >> "$TEST_RESULTS_DIR/$test_results_file"
            ./test_libtorch_standalone 2>&1 | tee -a "$TEST_RESULTS_DIR/$test_results_file"
            print_success "Standalone C++ test completed"
        fi
    fi

    # Run standalone Python test if available
    if [ -f "test_libtorch_python.py" ]; then
        print_status "Running standalone Python test..."
        echo "=== Standalone Python Test ===" >> "$TEST_RESULTS_DIR/$test_results_file"
        python3 test_libtorch_python.py 2>&1 | tee -a "$TEST_RESULTS_DIR/$test_results_file"
        print_success "Standalone Python test completed"
    fi
}

# Function to generate test report
generate_report() {
    print_status "Generating test report..."

    local report_file="$TEST_RESULTS_DIR/libtorch_test_report_$TIMESTAMP.md"

    cat > "$report_file" << EOF
# LibTorch Integration Test Report

**Generated:** $(date)
**Timestamp:** $TIMESTAMP

## Test Summary

### C++ Tests
EOF

    # Add C++ test results
    for test in "${TESTS[@]}"; do
        if [ -f "$BUILD_DIR/$test" ]; then
            echo "- ✅ $test: Built successfully" >> "$report_file"
        else
            echo "- ❌ $test: Build failed" >> "$report_file"
        fi
    done

    cat >> "$report_file" << EOF

### Python Tests
EOF

    # Add Python test results
    for test in "${PYTHON_TESTS[@]}"; do
        if [ -f "$test" ]; then
            echo "- ✅ $test: Available" >> "$report_file"
        else
            echo "- ❌ $test: Not found" >> "$report_file"
        fi
    done

    cat >> "$report_file" << EOF

## System Information

- **OS:** $(uname -s)
- **Architecture:** $(uname -m)
- **CMake Version:** $(cmake --version | head -n1)
- **Make Version:** $(make --version | head -n1)
- **Python Version:** $(python3 --version)

## LibTorch Information

EOF

    # Add LibTorch version info if available
    if python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
        python3 -c "import torch; print(f'- **PyTorch Version:** {torch.__version__}')" >> "$report_file"
        if torch.cuda.is_available(): print(f'- **CUDA Available:** Yes'); print(f'- **CUDA Version:** {torch.version.cuda}')" 2>/dev/null >> "$report_file" || echo "- **CUDA Available:** No" >> "$report_file"
    else
        echo "- **PyTorch:** Not available in Python environment" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

## Test Results

Detailed test results are available in the following files:
- C++ Tests: \`cpp_test_results_$TIMESTAMP.txt\`
- Python Tests: \`python_test_results_$TIMESTAMP.txt\`
- Standalone Tests: \`standalone_test_results_$TIMESTAMP.txt\`

## Recommendations

1. Ensure all tests pass before deploying to production
2. Monitor test results for any regressions
3. Update LibTorch version as needed
4. Verify CUDA compatibility if using GPU acceleration

EOF

    print_success "Test report generated: $report_file"
}

# Function to cleanup
cleanup() {
    print_status "Cleaning up..."

    # Remove test executables
    if [ -f "test_libtorch_standalone" ]; then
        rm -f test_libtorch_standalone
    fi

    # Remove test files
    rm -f test_model_*.pt test_tensor_*.pt

    print_success "Cleanup completed"
}

# Main execution
main() {
    print_status "Starting LibTorch comprehensive test suite..."

    # Store original directory
    ORIGINAL_DIR=$(pwd)

    # Check prerequisites
    check_prerequisites

    # Setup build environment
    setup_build_environment

    # Configure and build
    configure_cmake
    build_tests

    # Run tests
    local cpp_result=0
    local python_result=0
    local standalone_result=0

    run_cpp_tests || cpp_result=1
    run_python_tests || python_result=1
    run_standalone_tests || standalone_result=1

    # Generate report
    generate_report

    # Cleanup
    cd "$ORIGINAL_DIR"
    cleanup

    # Final summary
    print_status "Test execution completed"
    print_status "Results saved in: $TEST_RESULTS_DIR/"

    if [ $cpp_result -eq 0 ] && [ $python_result -eq 0 ] && [ $standalone_result -eq 0 ]; then
        print_success "All LibTorch tests passed successfully! 🎉"
        exit 0
    else
        print_error "Some tests failed. Check the results for details."
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "LibTorch Comprehensive Test Runner"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --clean        Clean build directory before building"
        echo "  --cpp-only     Run only C++ tests"
        echo "  --python-only  Run only Python tests"
        echo "  --standalone   Run only standalone tests"
        echo ""
        echo "Environment Variables:"
        echo "  LIBTORCH_PATH  Path to LibTorch installation"
        echo ""
        exit 0
        ;;
    --clean)
        print_status "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
        ;;
    --cpp-only)
        check_prerequisites
        setup_build_environment
        configure_cmake
        build_tests
        run_cpp_tests
        generate_report
        cleanup
        exit $?
        ;;
    --python-only)
        check_prerequisites
        run_python_tests
        generate_report
        cleanup
        exit $?
        ;;
    --standalone)
        check_prerequisites
        run_standalone_tests
        generate_report
        cleanup
        exit $?
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
