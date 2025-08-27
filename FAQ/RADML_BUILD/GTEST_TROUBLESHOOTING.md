# GTest Troubleshooting Guide

## Overview
This guide covers common GTest (Google Test) integration issues and their solutions in the Radiation Tolerant ML Framework. Many of the test are being updated as well.

## Table of Contents
1. [Common GTest Issues](#common-gtest-issues)
2. [Solution](#solution)
3. [Manual Installation Methods](#manual-installation-methods)
4. [Troubleshooting Steps](#troubleshooting-steps)
5. [Build System Integration](#build-system-integration)

## Common GTest Issues

### Problem 1: Hardcoded Paths
**Issue**: CMake files with hardcoded GTest paths like:
```cmake
# ❌ BAD: Hardcoded paths
if(EXISTS "/usr/local/Cellar/googletest/1.17.0/include/gtest/gtest.h")
    set(GTEST_INCLUDE_DIRS "/usr/local/Cellar/googletest/1.17.0/include")
    set(GTEST_LIBRARIES "/usr/local/Cellar/googletest/1.17.0/lib/libgtest.a")
```

**Problems**:
- Not portable across systems
- Breaks when GTest is updated
- Different paths on different OS (macOS vs Linux)
- Hard to maintain

### Problem 2: Missing GTest Installation
**Issue**: GTest not found on the system
**Symptoms**:
```
-- GTest not found, some tests may not compile
CMake Error: Could not find GTest
```

### Problem 3: Inconsistent Detection Methods
**Issue**: Different systems use different package managers
- **macOS**: Homebrew, MacPorts
- **Linux**: apt, yum, pkg-config
- **Windows**: vcpkg, Conan

## Solution

### Multi-Strategy GTest Detection
Implemented a robust, multi-layered approach:

```cmake
# ✅ GOOD: Multi-strategy detection
option(DOWNLOAD_GTEST "Download and build GTest if not found" ON)

# Strategy 1: Standard CMake find_package
find_package(GTest QUIET)

# Strategy 2: pkg-config (Linux)
if(NOT GTEST_FOUND)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(GTEST QUIET gtest)
    endif()
endif()

# Strategy 3: Common installation paths
if(NOT GTEST_FOUND)
    set(GTEST_SEARCH_PATHS
        "/usr/local/include"
        "/usr/include"
        "/opt/local/include"
        "/opt/homebrew/include"
        "/usr/local/opt/googletest/include"
        "/usr/local/Cellar/googletest/*/include"
        "/opt/homebrew/Cellar/googletest/*/include"
    )
    # ... search logic
endif()

# Strategy 4: Download and build
if(NOT GTEST_FOUND AND DOWNLOAD_GTEST)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
endif()
```

### Proper CMake Target Creation
```cmake
if(GTEST_FOUND)
    # Create proper imported targets
    if(NOT TARGET GTest::GTest)
        add_library(GTest::GTest UNKNOWN IMPORTED)
        set_target_properties(GTest::GTest PROPERTIES
            IMPORTED_LOCATION "${GTEST_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}"
        )
    endif()

    if(NOT TARGET GTest::Main)
        add_library(GTest::Main UNKNOWN IMPORTED)
        set_target_properties(GTest::Main PROPERTIES
            IMPORTED_LOCATION "${GTEST_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES GTest::GTest
        )
    endif()
endif()
```

## Manual Installation Methods

### macOS (Homebrew)
```bash
# Install GTest
brew install googletest

# Verify installation
ls /usr/local/include/gtest/
ls /usr/local/lib/libgtest*
```

### macOS (MacPorts)
```bash
# Install GTest
sudo port install googletest

# Verify installation
ls /opt/local/include/gtest/
ls /opt/local/lib/libgtest*
```

### Linux (Ubuntu/Debian)
```bash
# Install GTest
sudo apt-get install libgtest-dev

# Build GTest (required on some systems)
cd /usr/src/googletest
sudo cmake .
sudo make
sudo cp lib/*.a /usr/lib
```

### Linux (CentOS/RHEL)
```bash
# Install GTest
sudo yum install gtest-devel

# Or build from source
git clone https://github.com/google/googletest.git
cd googletest
mkdir build && cd build
cmake ..
make
sudo make install
```

### Windows (vcpkg)
```bash
# Install GTest
vcpkg install gtest

# Use in CMake
cmake .. -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake
```

## Troubleshooting Steps

### Step 1: Check System Installation
```bash
# Check if GTest is installed
find /usr -name "gtest.h" 2>/dev/null
find /usr/local -name "gtest.h" 2>/dev/null
find /opt -name "gtest.h" 2>/dev/null

# Check libraries
find /usr -name "libgtest*" 2>/dev/null
find /usr/local -name "libgtest*" 2>/dev/null
find /opt -name "libgtest*" 2>/dev/null
```

### Step 2: Verify CMake Detection
```bash
# Run CMake with verbose output
cmake .. -DBUILD_TESTING=ON --debug-find

# Look for GTest detection messages
grep -i "gtest\|googletest" CMakeCache.txt
```

### Step 3: Check Environment Variables
```bash
# Set GTest paths manually if needed
export GTEST_ROOT=/path/to/gtest
export GTEST_INCLUDE_DIR=/path/to/gtest/include
export GTEST_LIBRARY_DIR=/path/to/gtest/lib

# Re-run CMake
cmake .. -DBUILD_TESTING=ON
```

### Step 4: Enable Download Option
```bash
# Force download and build
cmake .. -DBUILD_TESTING=ON -DDOWNLOAD_GTEST=ON
```

## Build System Integration

### Using GTest in Your Tests
```cpp
#include <gtest/gtest.h>

TEST(MyTestSuite, MyTest) {
    EXPECT_EQ(2 + 2, 4);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

### CMake Test Configuration
```cmake
# Enable testing
enable_testing()

# Add test executable
add_executable(my_test my_test.cpp)

# Link with GTest
target_link_libraries(my_test GTest::GTest GTest::Main)

# Add test
add_test(NAME MyTest COMMAND my_test)
```

### Running Tests
```bash
# Build tests
make

# Run all tests
ctest

# Run specific test
./my_test

# Run with verbose output
ctest --verbose
```

## Success Indicators

### ✅ Good CMake Output
```
-- GTest found successfully!
--   Include directories: /usr/local/include
--   Libraries: /usr/local/lib/libgtest.a
--   Imported targets: GTest::GTest, GTest::Main
-- Testing enabled: ON
```

### ✅ Successful Build
```
-- Configuring done
-- Generating done
-- Build files have been written to: /path/to/build
```

### ✅ Tests Compile
```
[100%] Built target my_test
[100%] Built target all
```

## Common Error Messages and Solutions

### Error: "Could not find GTest"
**Solution**: Use our multi-strategy detection or install GTest manually

### Error: "GTest not found, testing will be disabled"
**Solution**: Check GTest installation or enable `DOWNLOAD_GTEST=ON`

### Error: "Cannot find -lgtest"
**Solution**: GTest libraries not in standard library path, check installation

### Error: "gtest/gtest.h: No such file or directory"
**Solution**: GTest headers not in include path, check installation

## Best Practices

### 1. Always Use Imported Targets
```cmake
# ✅ GOOD
target_link_libraries(my_test GTest::GTest GTest::Main)

# ❌ BAD
target_link_libraries(my_test gtest gtest_main)
```

### 2. Check GTest Version Compatibility
```cmake
# Ensure compatible version
find_package(GTest 1.8.0 REQUIRED)
```

### 3. Use FetchContent for Fallback
```cmake
# Automatic download if not found
option(DOWNLOAD_GTEST "Download GTest if not found" ON)
```

### 4. Test Your GTest Integration
```cpp
// Simple test to verify GTest works
TEST(GTestIntegration, BasicFunctionality) {
    EXPECT_TRUE(true);
    EXPECT_FALSE(false);
    EXPECT_EQ(1, 1);
}
```

## GTest Usage Instructions

### Running Tests with CTest
```bash
# Run all tests
ctest

# Run with verbose output
ctest --verbose

# Run with output on failure
ctest --output-on-failure

# Run specific test by name
ctest -R "test_name"

# Run tests in parallel
ctest -j4

# Stop on first failure
ctest --stop-on-failure
```

### Running Individual GTest Executables
```bash
# Run specific test executable
./enhanced_tmr_test
./framework_verification_test
./monte_carlo_validation

# Run with GTest options
./enhanced_tmr_test --help
./enhanced_tmr_test --gtest_list_tests
./enhanced_tmr_test --gtest_filter="TestSuite.*"
```

### GTest Command Line Options
```bash
# List all available tests
--gtest_list_tests

# Run specific test suite
--gtest_filter="TestSuiteName.*"

# Run specific test
--gtest_filter="TestSuiteName.TestName"

# Run tests matching pattern
--gtest_filter="*Test*"

# Exclude tests
--gtest_filter="-*SlowTest*"

# Run tests multiple times
--gtest_repeat=5

# Shuffle test order
--gtest_shuffle

# Set random seed
--gtest_random_seed=42

# Run with death test style
--gtest_death_test_style=threadsafe

# Generate XML output
--gtest_output=xml:results.xml
```

### Available Tests in Your Framework
```bash
# TMR Protection Tests
./enhanced_tmr_test                    # Triple Modular Redundancy tests
./enhanced_tmr_test --gtest_list_tests # List available TMR tests

# Framework Validation Tests
./framework_verification_test          # Framework verification tests
./framework_verification_test --gtest_list_tests

# Monte Carlo Simulation Tests
./monte_carlo_validation              # Monte Carlo validation tests
./space_monte_carlo_validation        # Space-optimized Monte Carlo tests

# Scientific Validation Tests
./scientific_validation_test          # Scientific validation across environments
./realistic_space_validation          # Realistic space environment tests

# Radiation Stress Tests
./radiation_stress_test               # Radiation stress testing
./systematic_fault_test               # Systematic fault injection tests

# Modern Features Tests
./modern_features_test                # Modern framework features
./modern_features_test --gtest_list_tests

# NASA/ESA Standard Tests
./nasa_esa_standard_test             # NASA/ESA standard compliance tests
```

### Common GTest Commands
```bash
# List available tests in any executable
./enhanced_tmr_test --gtest_list_tests
./modern_features_test --gtest_list_tests
./framework_verification_test --gtest_list_tests

# Run specific test patterns
./enhanced_tmr_test --gtest_filter="*SingleBit*"
./systematic_fault_test --gtest_filter="*BYTE_ERROR*"

# Run tests multiple times
./framework_verification_test --gtest_repeat=3

# Run with specific seed for reproducibility
./scientific_validation_test --gtest_random_seed=123

# Generate test list for CI/CD
./monte_carlo_validation --gtest_list_tests
```

---

## Building Tests Guide

### Building Your Tests
```bash
# Create and enter build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DBUILD_TESTING=ON

# Build all tests
make -j4

# Build specific test
make enhanced_tmr_test
make framework_verification_test
```

### Running Tests in Your Framework
```bash
# Run all tests with CTest
ctest

# Run specific test executable
./enhanced_tmr_test
./framework_verification_test
./monte_carlo_validation

# Run with verbose output
ctest --verbose

# Run with output on failure
ctest --output-on-failure
```

### Test File Locations
```
build_test/
├── enhanced_tmr_test              # TMR protection tests (112KB)
├── framework_verification_test    # Framework validation (153KB)
├── monte_carlo_validation        # Monte Carlo simulation tests (422KB)
├── space_monte_carlo_validation  # Space-optimized Monte Carlo (100KB)
├── realistic_space_validation    # Realistic space environment (289KB)
├── scientific_validation_test    # Scientific validation tests (418KB)
├── radiation_stress_test         # Radiation stress tests (172KB)
├── systematic_fault_test         # Systematic fault injection (275KB)
├── modern_features_test          # Modern framework features (422KB)
└── nasa_esa_standard_test       # NASA/ESA standard tests (665KB)
```

### Test Categories
- **TMR Tests**: Triple Modular Redundancy protection mechanisms
- **Framework Tests**: Core framework validation and verification
- **Monte Carlo Tests**: Statistical validation and simulation
- **Scientific Tests**: Physics-based radiation environment validation
- **Stress Tests**: Radiation stress and fault injection testing
- **Modern Tests**: Advanced framework features and capabilities
- **Standard Tests**: NASA/ESA compliance and industry standards

### Non-GTest Executables (Standalone Tests), will later Be added.
```bash
# AI Database Tests
./ai_native_database_test              # AI-native database functionality
./simple_ai_database_test              # Simple AI database tests
./simple_ai_database_test_minimal      # Minimal AI database tests

# Neural Network Tests
./ml_radiation_training_test           # ML radiation training tests
./trained_network_radiation_test       # Trained network radiation tests
./neural_network_validation            # Neural network validation
./monte_carlo_neuralnetwork            # Monte Carlo Neural Network Validation

# Cross-Validation Tests
./cross_validation_test                # Cross-validation testing
./training_loop_diagnostic             # Training loop diagnostics

# Quantum Field Tests
./quantum_field_test                   # Quantum field theory tests
./quantum_field_validation_test        # Quantum field validation
./quantum_stability_test               # Quantum stability tests

# LibTorch Integration Tests
./libtorch_radiation_integration_test  # PyTorch radiation integration
./libtorch_macos_compatibility_test    # macOS compatibility tests
./libtorch_resilience_test             # PyTorch resilience tests

# Database Tests
./lmdb_basic_test                      # LMDB basic functionality
./lmdb_datacenter_demo                 # LMDB datacenter demo

```


---

*Last updated: v1.0.2.2 - GTest integration fixes implemented*
