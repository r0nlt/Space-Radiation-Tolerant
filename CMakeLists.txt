cmake_minimum_required(VERSION 3.10)
project(RadiationTolerantML)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Try to find Eigen3 (if available)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND AND NOT EIGEN3_FOUND)
    # If Eigen3 is not found, set up include paths for our stubbed version
    message(STATUS "Eigen3 not found, using minimal stub implementation")
    include_directories(${CMAKE_SOURCE_DIR}/third_party/eigen_stub)
    add_definitions(-DUSE_EIGEN_STUB)
else()
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
endif()

# Add include directories
include_directories(include src)

# Add src subdirectory which contains the library implementations
add_subdirectory(src)

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
# Tests
if(BUILD_TESTS)
    message(STATUS "Building tests")
    # Enable testing
    enable_testing()
    # Find GTest
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})
    
    # Add test executable
    add_executable(rad_ml_framework_test
        test/tmr/rad_ml_framework_test.cpp
    )
    
    # Link libraries
    target_link_libraries(rad_ml_framework_test
        rad_ml_tmr  # Add our new TMR implementation library
        ${GTEST_BOTH_LIBRARIES}
        pthread
    )
    if(Eigen3_FOUND OR EIGEN3_FOUND)
        target_link_libraries(rad_ml_framework_test Eigen3::Eigen)
    endif()
    
    # Add test
    add_test(NAME rad_ml_framework_test COMMAND rad_ml_framework_test)
    
    # Add model accuracy test
    add_executable(model_accuracy_test
        test/model_accuracy_test.cpp
    )
    
    # Link libraries for model accuracy test
    target_link_libraries(model_accuracy_test
        rad_ml_tmr
        pthread
    )
    if(Eigen3_FOUND OR EIGEN3_FOUND)
        target_link_libraries(model_accuracy_test Eigen3::Eigen)
    endif()
    
    # Add test
    add_test(NAME model_accuracy_test COMMAND model_accuracy_test)

    # Add simple model accuracy test
    add_executable(simple_model_accuracy_test
        test/simple_model_accuracy_test.cpp
    )

    # Link libraries for simple model accuracy test
    target_link_libraries(simple_model_accuracy_test
        rad_ml_tmr
        pthread
    )
    if(Eigen3_FOUND OR EIGEN3_FOUND)
        target_link_libraries(simple_model_accuracy_test Eigen3::Eigen)
    endif()

    # Add test
    add_test(NAME simple_model_accuracy_test COMMAND simple_model_accuracy_test)

    # Add comprehensive model test
    add_executable(comprehensive_model_test
        test/comprehensive_model_test.cpp
    )

    # Link libraries for comprehensive model test
    target_link_libraries(comprehensive_model_test
        rad_ml_tmr
        pthread
    )
    if(Eigen3_FOUND OR EIGEN3_FOUND)
        target_link_libraries(comprehensive_model_test Eigen3::Eigen)
    endif()

    # Add test
    add_test(NAME comprehensive_model_test COMMAND comprehensive_model_test)
endif()

# Examples
if(BUILD_EXAMPLES)
    message(STATUS "Building examples")
    # Add demo executable
    add_executable(rad_tolerant_ml_demo
        examples/orbit_simulation/rad_tolerant_ml_demo.cpp
    )
    
    # Link libraries
    target_link_libraries(rad_tolerant_ml_demo
        rad_ml_tmr  # Add our TMR implementation library
        pthread
    )
    if(Eigen3_FOUND OR EIGEN3_FOUND)
        target_link_libraries(rad_tolerant_ml_demo Eigen3::Eigen)
    endif()
endif()
