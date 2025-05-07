cmake_minimum_required(VERSION 3.10)
project(RadiationTolerantML)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Try to find Eigen3 (if available)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Add additional hints for Eigen3 location
set(EIGEN3_ADDITIONAL_HINTS
  "/usr/local/Cellar/eigen/3.4.0_1/include/eigen3"
  "/usr/local/include/eigen3"
  "/opt/homebrew/include/eigen3"
)

# Set Eigen3 include dir directly if it's not found through find_package
if(EXISTS "/usr/local/Cellar/eigen/3.4.0_1/include/eigen3")
  set(EIGEN3_INCLUDE_DIR "/usr/local/Cellar/eigen/3.4.0_1/include/eigen3")
  set(EIGEN3_FOUND TRUE)
endif()

find_package(Eigen3 QUIET)

# Always create the Eigen3::Eigen target if EIGEN3_FOUND is TRUE
if(NOT TARGET Eigen3::Eigen AND EIGEN3_FOUND)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
  )
endif()

if(NOT Eigen3_FOUND AND NOT EIGEN3_FOUND)
    # Try to find Eigen3 in additional locations
    find_path(EIGEN3_INCLUDE_DIR
      NAMES Eigen/Core
      PATHS ${EIGEN3_ADDITIONAL_HINTS}
    )
    
    if(EIGEN3_INCLUDE_DIR)
        set(EIGEN3_FOUND TRUE)
        message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
        if(NOT TARGET Eigen3::Eigen)
            add_library(Eigen3::Eigen INTERFACE IMPORTED)
            set_target_properties(Eigen3::Eigen PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
            )
        endif()
    else()
        # If Eigen3 is not found, set up include paths for our stubbed version
        message(STATUS "Eigen3 not found, using minimal stub implementation")
        include_directories(${CMAKE_SOURCE_DIR}/third_party/eigen_stub)
        add_definitions(-DUSE_EIGEN_STUB)
    endif()
else()
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
endif()

# Create the Eigen3::Eigen target as an alias to our bundled version if using stub
if(NOT TARGET Eigen3::Eigen AND NOT EIGEN3_FOUND)
    add_library(eigen3_stub INTERFACE)
    target_include_directories(eigen3_stub INTERFACE ${CMAKE_SOURCE_DIR}/third_party/eigen_stub)
    add_library(Eigen3::Eigen ALIAS eigen3_stub)
endif()

# Add include directories
include_directories(include src)

# Add src subdirectory which contains the library implementations
add_subdirectory(src)

# Create missing directories if needed
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/validation)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/testing)

# Ensure the src/testing directory exists
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/src/testing)
    file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/src/testing)
endif()

# Add fault injection implementation
add_library(rad_ml_testing
    src/testing/fault_injection.cpp
)
target_include_directories(rad_ml_testing PRIVATE include)

# Add NASA/ESA standard test
add_executable(nasa_esa_standard_test 
    src/nasa_esa_standard_runner.cpp
    src/validation/nasa_esa_validation_protocol.cpp
)
target_include_directories(nasa_esa_standard_test PRIVATE include src)
target_link_libraries(nasa_esa_standard_test ${CMAKE_THREAD_LIBS_INIT})

# Add our verification test
add_executable(framework_verification_test
    test/verification/framework_verification_test.cpp
)

# Add our scientific validation test
add_executable(scientific_validation_test
    test/verification/scientific_validation_test.cpp
)

# Add our radiation stress test
add_executable(radiation_stress_test
    test/verification/radiation_stress_test.cpp
)

# Add systematic fault test
add_executable(systematic_fault_test
    test/verification/systematic_fault_test.cpp
)

# Add our enhanced TMR test
add_executable(enhanced_tmr_test
    test/verification/enhanced_tmr_test.cpp
)

# Add include directories for our verification tests
target_include_directories(framework_verification_test PRIVATE include src)
target_include_directories(scientific_validation_test PRIVATE include src)
target_include_directories(radiation_stress_test PRIVATE include src)
target_include_directories(systematic_fault_test PRIVATE include src)
target_include_directories(enhanced_tmr_test PRIVATE include src)

# Link libraries for our verification tests
target_link_libraries(framework_verification_test
    rad_ml_tmr
    ${CMAKE_THREAD_LIBS_INIT}
)

target_link_libraries(scientific_validation_test
    rad_ml_tmr
    ${CMAKE_THREAD_LIBS_INIT}
)

target_link_libraries(radiation_stress_test
    rad_ml_tmr
    ${CMAKE_THREAD_LIBS_INIT}
)

target_link_libraries(systematic_fault_test
    rad_ml_tmr
    rad_ml_testing
    ${CMAKE_THREAD_LIBS_INIT}
)

target_link_libraries(enhanced_tmr_test
    rad_ml_tmr
    ${CMAKE_THREAD_LIBS_INIT}
)

if(Eigen3_FOUND OR EIGEN3_FOUND)
    target_link_libraries(framework_verification_test Eigen3::Eigen)
    target_link_libraries(scientific_validation_test Eigen3::Eigen)
    target_link_libraries(radiation_stress_test Eigen3::Eigen)
    target_link_libraries(systematic_fault_test Eigen3::Eigen)
    target_link_libraries(enhanced_tmr_test Eigen3::Eigen)
endif()

# Enable testing
enable_testing()

# Add tests
add_test(NAME nasa_esa_standard_test COMMAND nasa_esa_standard_test)
add_test(NAME framework_verification_test COMMAND framework_verification_test)
add_test(NAME scientific_validation_test COMMAND scientific_validation_test)
add_test(NAME radiation_stress_test COMMAND radiation_stress_test)
add_test(NAME systematic_fault_test COMMAND systematic_fault_test)
add_test(NAME enhanced_tmr_test COMMAND enhanced_tmr_test)

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
