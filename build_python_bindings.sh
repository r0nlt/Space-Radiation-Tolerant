#!/bin/bash
# Build script for the rad_ml Python bindings
# Author: Rishab Nuguru
# License: AGPL v3 license

# Parse arguments
DEBUG=0
CLEAN=0
MINIMAL=1

while [[ $# -gt 0 ]]; do
  case $1 in
    --debug)
      DEBUG=1
      shift
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --full)
      MINIMAL=0
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# Set debug flags
if [ $DEBUG -eq 1 ]; then
  set -x  # Print commands as they execute
  CMAKE_BUILD_TYPE="Debug"
  VERBOSE=1
else
  CMAKE_BUILD_TYPE="Release"
fi

# Exit on error
set -e

# Clean if requested
if [ $CLEAN -eq 1 ]; then
  echo "Cleaning build directory..."
  rm -rf build_python
fi

# Create build directory if it doesn't exist
mkdir -p build_python
cd build_python

# Configure CMake with Python bindings enabled
if [ $MINIMAL -eq 1 ]; then
  echo "Using minimal Python bindings"
  cmake .. -DBUILD_PYTHON_BINDINGS=ON -DUSE_MINIMAL_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -B.
else
  echo "Using full Python bindings"
  cmake .. -DBUILD_PYTHON_BINDINGS=ON -DUSE_MINIMAL_PYTHON_BINDINGS=OFF -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -B.
fi

# Build the Python module
if [ $DEBUG -eq 1 ]; then
  cmake --build . -j4 --verbose
else
  cmake --build . -j4
fi

# List the built files to help with debugging
if [ $DEBUG -eq 1 ]; then
  echo "Looking for built Python module..."
  find . -name "*.so" -o -name "*.dylib" -o -name "*.pyd"
  echo "Contents of rad_ml directory:"
  ls -la rad_ml || echo "rad_ml directory not found"
fi

# Install the Python module in development mode
cd ../python
pip install -e .

echo ""
echo "======================================================================"
echo "Python bindings for rad_ml have been built and installed successfully!"
echo "You can now import rad_ml in your Python code."
echo ""
echo "To run the examples:"
echo "  python -m examples.tmr_example"
echo ""
echo "If you encounter issues, try running with debug flags:"
echo "  ./build_python_bindings.sh --debug"
echo ""
echo "Or to build full bindings (requires fixing C++ code):"
echo "  ./build_python_bindings.sh --full"
echo "======================================================================"
