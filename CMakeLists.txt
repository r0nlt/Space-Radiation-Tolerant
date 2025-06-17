cmake_minimum_required(VERSION 3.10)
project(RadiationTolerantML)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set modern CMake policies
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

# Include the IDE integration system
include(cmake/RadTolerantIDE.cmake)

# Build options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(ENABLE_VISUALIZATION "Enable visualization with OpenCV" OFF)
option(BUILD_TESTING "Build with testing enabled" ON)
option(USE_MINIMAL_PYTHON_BINDINGS "Use minimal Python bindings to avoid compilation errors" ON)
option(ENABLE_IDE_INTEGRATION "Enable IDE integration features" ON)

# Find required packages
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# Find GoogleTest
find_package(GTest)
if(NOT GTEST_FOUND)
  message(STATUS "GTest not found via find_package, trying manual paths")
  if(EXISTS "/usr/local/Cellar/googletest/1.17.0/include/gtest/gtest.h")
    set(GTEST_INCLUDE_DIRS "/usr/local/Cellar/googletest/1.17.0/include")
    set(GTEST_LIBRARIES "/usr/local/Cellar/googletest/1.17.0/lib/libgtest.a" "/usr/local/Cellar/googletest/1.17.0/lib/libgtest_main.a")
    set(GTEST_FOUND TRUE)
    message(STATUS "Found GTest at /usr/local/Cellar/googletest")
  endif()
endif()

if(GTEST_FOUND)
  include_directories(${GTEST_INCLUDE_DIRS})
  message(STATUS "GTest include directories: ${GTEST_INCLUDE_DIRS}")
  message(STATUS "GTest libraries: ${GTEST_LIBRARIES}")
else()
  message(WARNING "GTest not found, some tests may not compile")
endif()

# Use modern Boost find_package approach
if(BUILD_PYTHON_BINDINGS)
  # For Python bindings, we may not need Boost
else()
  find_package(Boost REQUIRED COMPONENTS system filesystem)
endif()

# Setup Eigen includes properly
if(EIGEN3_INCLUDE_DIR)
  # Include system Eigen headers with BEFORE to override any local headers
  include_directories(BEFORE ${EIGEN3_INCLUDE_DIR})
  message(STATUS "Using Eigen3 from: ${EIGEN3_INCLUDE_DIR}")
else()
  # Fallback to default locations
  include_directories(BEFORE /usr/local/include/eigen3 /usr/include/eigen3)
  message(STATUS "Using default Eigen3 locations")
endif()

# Project includes
include_directories(include)

# Enable testing
enable_testing()

# Define libraries
add_library(rad_ml_testing STATIC
    src/testing/fault_injection.cpp
    src/testing/protection_techniques.cpp
)

# Define the quantum field theory library
add_library(rad_ml_qft STATIC
    src/rad_ml/physics/quantum_field_theory.cpp
    src/rad_ml/physics/quantum_models.cpp
    src/rad_ml/physics/quantum_enhanced_radiation.cpp
)

target_link_libraries(rad_ml_qft
    ${CMAKE_THREAD_LIBS_INIT}
    Eigen3::Eigen
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

# Add research components
add_subdirectory(src/rad_ml/research)

# Add examples
add_subdirectory(examples)

# Add executable for Monte Carlo validation test
add_executable(monte_carlo_validation test/verification/monte_carlo_validation.cpp)
target_link_libraries(monte_carlo_validation rad_ml_tmr rad_ml_testing rad_ml_qft)
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

add_executable(scientific_validation_test
    test/scientific_validation/scientific_validation_test.cpp
)

target_link_libraries(scientific_validation_test
    rad_ml_qft
    rad_ml_research
    ${CMAKE_THREAD_LIBS_INIT}
)

target_include_directories(scientific_validation_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/eigen
)

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

# Add the IEEE-754 aware TMR voting test
add_executable(ieee754_tmr_test test/verification/ieee754_tmr_test.cpp)
target_link_libraries(ieee754_tmr_test rad_ml_tmr rad_ml_testing)
target_compile_definitions(ieee754_tmr_test PRIVATE ENABLE_TESTING=1)
add_test(NAME ieee754_tmr_test COMMAND ieee754_tmr_test)

# Add the quantum field validation test
add_executable(quantum_field_validation_test test/verification/quantum_field_validation_test.cpp)
target_link_libraries(quantum_field_validation_test
    rad_ml_tmr
    rad_ml_testing
    rad_ml_qft
)
add_test(NAME quantum_field_validation_test COMMAND quantum_field_validation_test)

# Add the realistic LEO mission test
add_executable(realistic_leo_mission_test test/verification/realistic_leo_mission_test.cpp)
target_link_libraries(realistic_leo_mission_test
    rad_ml_tmr
    rad_ml_testing
    rad_ml_qft
)
add_test(NAME realistic_leo_mission_test COMMAND realistic_leo_mission_test)

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

# Add the quantum field theory example
add_executable(quantum_field_example
  examples/quantum_field_example.cpp
)
target_link_libraries(quantum_field_example
  PRIVATE
  rad_ml_tmr
  rad_ml_testing
  rad_ml_qft
)

# Optional: Link OpenCV if visualization is enabled
if(ENABLE_VISUALIZATION)
  find_package(OpenCV REQUIRED)
  target_link_libraries(neural_network_validation PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(neural_network_validation PRIVATE ENABLE_VISUALIZATION)
  target_link_libraries(monte_carlo_neuralnetwork PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(monte_carlo_neuralnetwork PRIVATE ENABLE_VISUALIZATION)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
  message(STATUS "Building Python bindings")

  # Include pybind11
  add_subdirectory(third_party/pybind11)

  # Find Python
  find_package(Python COMPONENTS Interpreter Development REQUIRED)

  # Build the Python module - choose between minimal and full versions
  if(USE_MINIMAL_PYTHON_BINDINGS)
    message(STATUS "Using minimal Python bindings to avoid compilation errors")
    pybind11_add_module(_core python/rad_ml_minimal.cpp)
    # Minimal bindings don't need to link against any libraries
  else()
    message(STATUS "Using full Python bindings")
    pybind11_add_module(_core python/rad_ml_core.cpp)
    target_link_libraries(_core PRIVATE rad_ml_tmr rad_ml_testing rad_ml_qft)
  endif()

  # Set the output directory for the Python module
  set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/rad_ml"
  )

  # Create the Python package directory
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/rad_ml)

  # Copy Python files for development mode
  file(GLOB PYTHON_FILES "python/rad_ml/*.py")
  foreach(PYTHON_FILE ${PYTHON_FILES})
    get_filename_component(FILE_NAME ${PYTHON_FILE} NAME)
    configure_file(${PYTHON_FILE} ${CMAKE_BINARY_DIR}/rad_ml/${FILE_NAME} COPYONLY)
  endforeach()

  # Create __init__.py in the build directory
  configure_file(python/rad_ml/__init__.py ${CMAKE_BINARY_DIR}/rad_ml/__init__.py COPYONLY)

  # Create examples directory
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/examples)

  # Copy examples
  file(GLOB EXAMPLE_FILES "python/examples/*.py")
  foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
    get_filename_component(FILE_NAME ${EXAMPLE_FILE} NAME)
    configure_file(${EXAMPLE_FILE} ${CMAKE_BINARY_DIR}/examples/${FILE_NAME} COPYONLY)
  endforeach()
endif()

# Enable IDE integration if requested
if(ENABLE_IDE_INTEGRATION)
  configure_all_ide_integrations()

  # Add custom target to run clang-format over source files
  find_program(CLANG_FORMAT_EXE NAMES "clang-format")
  if(CLANG_FORMAT_EXE)
    # Get all source files
    file(GLOB_RECURSE ALL_SOURCE_FILES
      "${CMAKE_SOURCE_DIR}/src/*.cpp"
      "${CMAKE_SOURCE_DIR}/src/*.hpp"
      "${CMAKE_SOURCE_DIR}/include/*.hpp"
      "${CMAKE_SOURCE_DIR}/examples/*.cpp"
      "${CMAKE_SOURCE_DIR}/test/*.cpp"
    )

    add_custom_target(
      format
      COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${ALL_SOURCE_FILES}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Running clang-format to format all source files"
      VERBATIM
    )

    message(STATUS "Added 'format' target using clang-format: ${CLANG_FORMAT_EXE}")
  endif()
endif()

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Display configuration summary
set(BUILD_TYPE ${CMAKE_BUILD_TYPE})
if(NOT BUILD_TYPE)
  set(BUILD_TYPE "Release")
endif()

message(STATUS "")
message(STATUS "==== Radiation Tolerant ML Framework Configuration ====")
message(STATUS "Build type:                 ${BUILD_TYPE}")
message(STATUS "C++ Standard:               C++17")
message(STATUS "Python bindings:            ${BUILD_PYTHON_BINDINGS}")
message(STATUS "IDE integration:            ${ENABLE_IDE_INTEGRATION}")
message(STATUS "Testing enabled:            ${BUILD_TESTING}")
message(STATUS "Visualization:              ${ENABLE_VISUALIZATION}")
message(STATUS "======================================================")
