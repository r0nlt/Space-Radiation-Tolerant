#!/bin/bash
# Direct build script for minimal Python bindings
# Author: Rishab Nuguru
# License: AGPL v3 license

set -e  # Exit on error

# Detect Python
PYTHON_BIN=$(which python3)
echo "Using Python: $PYTHON_BIN"

# Get Python include directory and library flags - use Python's own tool
PYTHON_INCLUDES=$($PYTHON_BIN -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_CONFIG=$($PYTHON_BIN-config --ldflags)
PYTHON_EXTENSION=$($PYTHON_BIN -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

echo "Python includes: $PYTHON_INCLUDES"
echo "Python config flags: $PYTHON_CONFIG"
echo "Python extension: $PYTHON_EXTENSION"

# Ensure pybind11 is available
if [ ! -d "third_party/pybind11/include" ]; then
    echo "pybind11 not found in third_party/pybind11/include"
    echo "Checking if it can be found via pip..."
    PYBIND11_INCLUDES=$($PYTHON_BIN -c "import pybind11; print(pybind11.get_include())")
    if [ -z "$PYBIND11_INCLUDES" ]; then
        echo "pybind11 not found via pip either. Please install it:"
        echo "pip install pybind11"
        exit 1
    fi
    echo "Using pybind11 from pip: $PYBIND11_INCLUDES"
else
    PYBIND11_INCLUDES="third_party/pybind11/include"
    echo "Using pybind11 from third_party: $PYBIND11_INCLUDES"
fi

# Create output directories
mkdir -p build/rad_ml
mkdir -p build/examples
echo "Created output directories"

# Set compiler flags
CXX=${CXX:-clang++}
CXXFLAGS="-std=c++17 -O3 -Wall -shared -fPIC"
INCLUDES="-I$PYTHON_INCLUDES -I$PYBIND11_INCLUDES"

# Build the module
echo "Building _core module..."
$CXX $CXXFLAGS $INCLUDES python/rad_ml_minimal.cpp -o build/rad_ml/_core$PYTHON_EXTENSION $PYTHON_CONFIG

# Copy Python files
echo "Copying Python package files..."
cp python/rad_ml/*.py build/rad_ml/
cp python/examples/*.py build/examples/

# Report success
if [ -f "build/rad_ml/_core$PYTHON_EXTENSION" ]; then
    echo "Build successful!"
    echo "Module: build/rad_ml/_core$PYTHON_EXTENSION"
    ls -la build/rad_ml/

    echo ""
    echo "======================================================================"
    echo "To use the module, add it to your Python path:"
    echo "export PYTHONPATH=$PWD/build:\$PYTHONPATH"
    echo ""
    echo "Or install it in development mode:"
    echo "cd python && pip install -e ."
    echo ""
    echo "You can then import the module:"
    echo "import rad_ml"
    echo "======================================================================"
else
    echo "Build failed!"
    exit 1
fi
