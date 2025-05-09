cmake_minimum_required(VERSION 3.10)
project(RadiationTolerantML)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

# Try to find Eigen3 (if available)
if (NOT EIGEN3_INCLUDE_DIR)
  # Look for it in some common places
  find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core
            PATHS "${CMAKE_SOURCE_DIR}/third_party/eigen"
                  "/usr/local/include/eigen3"
                  "/usr/include/eigen3"
            DOC "Eigen include directory")
  if (EIGEN3_INCLUDE_DIR)
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
  else()
    message(WARNING "Eigen3 not found. Some functionality may be disabled.")
  endif()
endif()

# Include directories
include_directories(include)
if (EIGEN3_INCLUDE_DIR)
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Enable testing
enable_testing()

# Define libraries
add_library(rad_ml_testing STATIC
    src/testing/fault_injection.cpp
    src/testing/protection_techniques.cpp
)

# Define NASA/ESA Standard Test Runner
add_executable(nasa_esa_standard_test
    src/nasa_esa_standard_runner.cpp
    src/validation/nasa_esa_validation_protocol.cpp
)

target_link_libraries(nasa_esa_standard_test
    rad_ml_testing
    ${CMAKE_THREAD_LIBS_INIT}
)

# Add TMR library
add_subdirectory(src/tmr)

# Add executable for Monte Carlo validation test
add_executable(monte_carlo_validation test/verification/monte_carlo_validation.cpp)
target_link_libraries(monte_carlo_validation rad_ml_tmr rad_ml_testing)
add_test(NAME monte_carlo_validation COMMAND monte_carlo_validation)

# Add executable for space-optimized Monte Carlo validation
add_executable(space_monte_carlo_validation test/verification/space_monte_carlo_validation.cpp)
target_link_libraries(space_monte_carlo_validation rad_ml_tmr rad_ml_testing)
target_compile_definitions(space_monte_carlo_validation PRIVATE ENABLE_TESTING=1)
add_test(NAME space_monte_carlo_validation COMMAND space_monte_carlo_validation)

# Add the realistic space validation test
add_executable(realistic_space_validation test/verification/realistic_space_validation.cpp)
target_link_libraries(realistic_space_validation rad_ml_tmr rad_ml_testing)
target_compile_definitions(realistic_space_validation PRIVATE ENABLE_TESTING=1)
add_test(NAME realistic_space_validation COMMAND realistic_space_validation)

# Add other verification tests
add_executable(framework_verification_test test/verification/framework_verification_test.cpp)
target_link_libraries(framework_verification_test rad_ml_tmr rad_ml_testing)
add_test(NAME framework_verification_test COMMAND framework_verification_test)

add_executable(enhanced_tmr_test test/verification/enhanced_tmr_test.cpp)
target_link_libraries(enhanced_tmr_test rad_ml_tmr rad_ml_testing)
add_test(NAME enhanced_tmr_test COMMAND enhanced_tmr_test)

add_executable(scientific_validation_test test/verification/scientific_validation_test.cpp)
target_link_libraries(scientific_validation_test rad_ml_tmr rad_ml_testing)
add_test(NAME scientific_validation_test COMMAND scientific_validation_test)

add_executable(radiation_stress_test test/verification/radiation_stress_test.cpp)
target_link_libraries(radiation_stress_test rad_ml_tmr rad_ml_testing)
add_test(NAME radiation_stress_test COMMAND radiation_stress_test)

add_executable(systematic_fault_test test/verification/systematic_fault_test.cpp)
target_link_libraries(systematic_fault_test rad_ml_tmr rad_ml_testing)
add_test(NAME systematic_fault_test COMMAND systematic_fault_test)

add_executable(modern_features_test test/verification/modern_features_test.cpp)
target_link_libraries(modern_features_test rad_ml_tmr rad_ml_testing)
add_test(NAME modern_features_test COMMAND modern_features_test)

# Add the neural network validation test
add_executable(neural_network_validation
  test/verification/monte_carlo_neuralnetwork.cpp
)
target_link_libraries(neural_network_validation
  PRIVATE
  rad_ml_tmr
  rad_ml_testing
)

# Add the neural Monte Carlo validation test
add_executable(monte_carlo_neuralnetwork
  test/verification/monte_carlo_neuralnetwork.cpp
)
target_link_libraries(monte_carlo_neuralnetwork
  PRIVATE
  rad_ml_tmr
  rad_ml_testing
)

# Optional: Link OpenCV if visualization is enabled
option(ENABLE_VISUALIZATION "Enable visualization with OpenCV" OFF)
if(ENABLE_VISUALIZATION)
  find_package(OpenCV REQUIRED)
  target_link_libraries(neural_network_validation PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(neural_network_validation PRIVATE ENABLE_VISUALIZATION)
  target_link_libraries(monte_carlo_neuralnetwork PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(monte_carlo_neuralnetwork PRIVATE ENABLE_VISUALIZATION)
endif()

# Display configuration summary
set(BUILD_TYPE ${CMAKE_BUILD_TYPE})
if(NOT BUILD_TYPE)
  set(BUILD_TYPE "Release")
endif()

# Output configuration information
message(STATUS "Configured RadiationTolerantML NASA/ESA Standard Test:")
message(STATUS "  C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${BUILD_TYPE}")

# Enable testing flag if needed
if(BUILD_TESTING)
  message(STATUS "  Testing: Enabled")
  add_definitions(-DENABLE_TESTING=1)
endif()
