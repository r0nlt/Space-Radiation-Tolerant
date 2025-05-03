cmake_minimum_required(VERSION 3.10)
project(rad_tolerant_ml VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(RAD_ML_ENABLE_WARNINGS "Enable all warnings" ON)
option(RAD_ML_BUILD_STRESS_TEST "Build extreme radiation stress test" ON)

# Add compiler warnings
if(RAD_ML_ENABLE_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-Wall -Wextra -Wpedantic)
    elseif(MSVC)
        add_compile_options(/W4)
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Make sure directories exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples/enhanced_features_test)

# Add executables for tests
add_executable(simple_test examples/enhanced_features_test/simple_test.cpp)
add_executable(approx_tmr_test examples/enhanced_features_test/approx_tmr_test.cpp)

# Build stress test if enabled
if(RAD_ML_BUILD_STRESS_TEST)
    add_executable(extreme_stress_test examples/enhanced_features_test/extreme_stress_test.cpp)
endif()

# Enable testing
enable_testing()

# Add tests
add_test(NAME SimpleTest COMMAND simple_test)
add_test(NAME ApproxTMRTest COMMAND approx_tmr_test)
if(RAD_ML_BUILD_STRESS_TEST)
    add_test(NAME ExtremeStressTest COMMAND extreme_stress_test)
endif() 