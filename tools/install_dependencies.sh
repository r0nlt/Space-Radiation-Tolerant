#!/bin/bash

echo "=== Radiation Tolerant ML Dependencies Installer ==="
echo "This script will install required dependencies for the project."

# Detect OS
OS=$(uname)
if [ "$OS" == "Darwin" ]; then
    echo "Detected macOS system"

    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    # Install dependencies with Homebrew
    echo "Installing dependencies with Homebrew..."
    brew install cmake eigen boost googletest lmdb

    echo "Dependencies installed successfully!"

elif [ "$OS" == "Linux" ]; then
    echo "Detected Linux system"

    # Check for apt-get (Debian/Ubuntu)
    if command -v apt-get &> /dev/null; then
        echo "Installing dependencies with apt-get..."
        sudo apt-get update
        sudo apt-get install -y cmake libboost-all-dev libeigen3-dev libgtest-dev build-essential liblmdb-dev

        # Build GTest if needed
        if [ ! -f "/usr/lib/libgtest.a" ] && [ -d "/usr/src/gtest" ]; then
            echo "Building Google Test..."
            cd /usr/src/gtest
            sudo cmake .
            sudo make
            sudo cp lib/*.a /usr/lib/
        fi

        echo "Dependencies installed successfully!"

    # Check for yum (CentOS/RHEL/Fedora)
    elif command -v yum &> /dev/null; then
        echo "Installing dependencies with yum..."
        sudo yum install -y cmake boost-devel eigen3-devel gtest-devel gcc-c++ lmdb-devel
        echo "Dependencies installed successfully!"

    # Check for dnf (newer Fedora)
    elif command -v dnf &> /dev/null; then
        echo "Installing dependencies with dnf..."
        sudo dnf install -y cmake boost-devel eigen3-devel gtest-devel gcc-c++ lmdb-devel
        echo "Dependencies installed successfully!"

    else
        echo "Unsupported Linux distribution. Please install the following packages manually:"
        echo "- CMake"
        echo "- Boost (system, filesystem)"
        echo "- Eigen3"
        echo "- Google Test"
        echo "- LMDB development libraries"
        echo "- C++ compiler (gcc/g++)"
    fi

else
    echo "Unsupported operating system: $OS"
    echo "Please install the following dependencies manually:"
    echo "- CMake"
    echo "- Boost (system, filesystem)"
    echo "- Eigen3"
    echo "- Google Test"
    echo "- LMDB development libraries"
    echo "- C++ compiler"
fi

echo ""
echo "After installing dependencies, run: ./tools/build_all.sh"
