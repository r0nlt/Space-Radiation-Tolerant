cmake_minimum_required(VERSION 3.10)
project(RadiationTolerantML)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Add include directories
include_directories(include src)

# Directory structure validation
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/validation)

# Add NASA/ESA standard test
add_executable(nasa_esa_standard_test 
    src/nasa_esa_standard_runner.cpp
    src/validation/nasa_esa_validation_protocol.cpp
)
target_include_directories(nasa_esa_standard_test PRIVATE include src)
target_link_libraries(nasa_esa_standard_test ${CMAKE_THREAD_LIBS_INIT})

# Enable testing
enable_testing()

# Add test
add_test(NAME nasa_esa_standard_test COMMAND nasa_esa_standard_test)

# Print configuration summary
message(STATUS "Configured RadiationTolerantML NASA/ESA Standard Test:")
message(STATUS "  C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Testing: Enabled") 