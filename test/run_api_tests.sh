#!/bin/bash
# Script to run all API tests for the Space-Radiation-Tolerant framework

set -e  # Exit on error

# Color formatting for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Running Space-Radiation-Tolerant API Tests${NC}"
echo "=================================="

# Determine script path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Build directory location
BUILD_DIR="$ROOT_DIR/build"
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${BLUE}Creating build directory...${NC}"
    mkdir -p "$BUILD_DIR"
fi

# CD to build directory
cd "$BUILD_DIR"

# Check if CMake is installed
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: CMake is not installed. Please install CMake and try again.${NC}"
    exit 1
fi

# Check if the build directory exists and has CMake files
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    echo -e "${BLUE}Configuring CMake...${NC}"
    cmake "$ROOT_DIR"
fi

# Run C++ API tests
echo -e "${BLUE}Running C++ API tests...${NC}"
if ! cmake --build . --target run_api_tests; then
    echo -e "${RED}C++ API tests failed.${NC}"
    C_TESTS_FAILED=1
else
    echo -e "${GREEN}C++ API tests passed.${NC}"
    C_TESTS_FAILED=0
fi

# Run Python API tests
echo -e "${BLUE}Running Python API tests...${NC}"
cd "$ROOT_DIR"
if ! python -m pytest python/rad_ml/test_unified_defense.py -v; then
    echo -e "${RED}Python API tests failed.${NC}"
    PYTHON_TESTS_FAILED=1
else
    echo -e "${GREEN}Python API tests passed.${NC}"
    PYTHON_TESTS_FAILED=0
fi

# Run TensorFlow example
echo -e "${BLUE}Running TensorFlow example...${NC}"
if ! python "$ROOT_DIR/examples/tensorflow_radiation_hardening.py"; then
    echo -e "${RED}TensorFlow example failed.${NC}"
    TF_FAILED=1
else
    echo -e "${GREEN}TensorFlow example passed.${NC}"
    TF_FAILED=0
fi

# Run PyTorch example
echo -e "${BLUE}Running PyTorch example...${NC}"
if ! python "$ROOT_DIR/examples/pytorch_radiation_hardening.py"; then
    echo -e "${RED}PyTorch example failed.${NC}"
    PT_FAILED=1
else
    echo -e "${GREEN}PyTorch example passed.${NC}"
    PT_FAILED=0
fi

# Output summary
echo ""
echo -e "${BLUE}Test Summary:${NC}"
echo "=================================="
[ $C_TESTS_FAILED -eq 0 ] && echo -e "${GREEN}✓${NC} C++ API Tests" || echo -e "${RED}✗${NC} C++ API Tests"
[ $PYTHON_TESTS_FAILED -eq 0 ] && echo -e "${GREEN}✓${NC} Python API Tests" || echo -e "${RED}✗${NC} Python API Tests"
[ $TF_FAILED -eq 0 ] && echo -e "${GREEN}✓${NC} TensorFlow Example" || echo -e "${RED}✗${NC} TensorFlow Example"
[ $PT_FAILED -eq 0 ] && echo -e "${GREEN}✓${NC} PyTorch Example" || echo -e "${RED}✗${NC} PyTorch Example"

# Final result
if [ $C_TESTS_FAILED -eq 0 ] && [ $PYTHON_TESTS_FAILED -eq 0 ] && [ $TF_FAILED -eq 0 ] && [ $PT_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
