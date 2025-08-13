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
option(ENABLE_PYTORCH "Enable PyTorch integration" OFF)

# Find required packages
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# Find PyTorch if enabled
if(ENABLE_PYTORCH)
  # Add cmake modules path
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

  # Find PyTorch
  find_package(PyTorch)

  if(PyTorch_FOUND)
    message(STATUS "PyTorch found: ${PyTorch_LIBRARY}")
    message(STATUS "PyTorch include directory: ${PyTorch_INCLUDE_DIR}")
  else()
    message(WARNING "PyTorch not found. Set PyTorch_ROOT or PYTORCH_ROOT environment variable.")
    message(WARNING "You can download LibTorch from: https://pytorch.org/get-started/locally/")
  endif()
endif()

# Find LMDB
find_path(LMDB_INCLUDE_DIR
    NAMES lmdb.h
    HINTS
        /usr/local/opt/lmdb/include
        /usr/local/include
        /usr/include
        /opt/local/include
        /opt/homebrew/include
)

find_library(LMDB_LIBRARY
    NAMES lmdb
    HINTS
        /usr/local/opt/lmdb/lib
        /usr/local/lib
        /usr/lib
        /opt/local/lib
        /opt/homebrew/lib
)

if(LMDB_INCLUDE_DIR AND LMDB_LIBRARY)
    set(LMDB_FOUND TRUE)
    message(STATUS "Found LMDB: ${LMDB_LIBRARY}")
    message(STATUS "LMDB include directory: ${LMDB_INCLUDE_DIR}")

    # Create imported target for LMDB
    add_library(LMDB::LMDB UNKNOWN IMPORTED)
    set_target_properties(LMDB::LMDB PROPERTIES
        IMPORTED_LOCATION "${LMDB_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LMDB_INCLUDE_DIR}"
    )
else()
    set(LMDB_FOUND FALSE)
    message(WARNING "LMDB not found. AI-native database features will be disabled.")
endif()

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

# Add PyTorch integration if enabled
if(ENABLE_PYTORCH)
  add_subdirectory(src/rad_ml/pytorch)
endif()

# Add examples
add_subdirectory(examples)

# Add executable for Monte Carlo validation test
add_executable(monte_carlo_validation test/verification/monte_carlo_validation.cpp)
target_link_libraries(monte_carlo_validation rad_ml_tmr rad_ml_testing rad_ml_qft)
if(ENABLE_PYTORCH AND PyTorch_FOUND)
  target_link_libraries(monte_carlo_validation rad_ml_pytorch)
endif()
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

# Add radiation environment validation executable
add_executable(radiation_environment_validation
  test/verification/radiation_environment_validation.cpp
)
target_link_libraries(radiation_environment_validation
  PRIVATE
  rad_ml_tmr
  rad_ml_testing
)
add_test(NAME radiation_environment_validation COMMAND radiation_environment_validation)

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

# Add the ML Radiation Training test
add_executable(ml_radiation_training_test
  test/ml_radiation_training_test.cpp
)
target_link_libraries(ml_radiation_training_test
  PRIVATE
  rad_ml_tmr
  rad_ml_testing
  rad_ml_research
  rad_ml_qft
  gtest
  gtest_main
  ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories(ml_radiation_training_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
add_test(NAME ml_radiation_training_test COMMAND ml_radiation_training_test)

# Add the Cross Validation test
add_executable(cross_validation_test
  test/cross_validation_test.cpp
)
target_link_libraries(cross_validation_test
  PRIVATE
  rad_ml_tmr
  rad_ml_testing
  rad_ml_research
  rad_ml_qft
  gtest
  gtest_main
  ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories(cross_validation_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
add_test(NAME cross_validation_test COMMAND cross_validation_test)

# Add the Training Loop Diagnostic test
add_executable(training_loop_diagnostic
  training_loop_diagnostic.cpp
)
target_link_libraries(training_loop_diagnostic
  PRIVATE
  rad_ml_tmr
  rad_ml_testing
  rad_ml_research
  rad_ml_qft
  gtest
  gtest_main
  ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories(training_loop_diagnostic PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
add_test(NAME training_loop_diagnostic COMMAND training_loop_diagnostic)

# Add the Simple Radiation test
add_executable(simple_radiation_test
  test/simple_radiation_test.cpp
)
target_link_libraries(simple_radiation_test
  PRIVATE
  rad_ml_tmr
  rad_ml_testing
  rad_ml_research
  rad_ml_qft
  gtest
  gtest_main
  ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories(simple_radiation_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
add_test(NAME simple_radiation_test COMMAND simple_radiation_test)

# Add the Weight Diagnostic test
add_executable(weight_diagnostic_test
  weight_diagnostic_test.cpp
)
target_link_libraries(weight_diagnostic_test
  PRIVATE
  rad_ml_tmr
  rad_ml_testing
  rad_ml_research
  rad_ml_qft
  gtest
  gtest_main
  ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories(weight_diagnostic_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
add_test(NAME weight_diagnostic_test COMMAND weight_diagnostic_test)

# Add the Trained Network Radiation test
add_executable(trained_network_radiation_test
  trained_network_radiation_test.cpp
)
target_link_libraries(trained_network_radiation_test
  PRIVATE
  rad_ml_tmr
  rad_ml_testing
  rad_ml_research
  rad_ml_qft
  gtest
  gtest_main
  ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories(trained_network_radiation_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
add_test(NAME trained_network_radiation_test COMMAND trained_network_radiation_test)

# Add the Trivial Radiation test
add_executable(trivial_radiation_test
  trivial_radiation_test.cpp
)
target_link_libraries(trivial_radiation_test
  PRIVATE
  rad_ml_tmr
  rad_ml_testing
  rad_ml_research
  rad_ml_qft
  gtest
  gtest_main
  ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories(trivial_radiation_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
add_test(NAME trivial_radiation_test COMMAND trivial_radiation_test)

# Add PyTorch integration test if enabled
if(ENABLE_PYTORCH AND PyTorch_FOUND)
  add_executable(pytorch_integration_test
    test/pytorch_integration_test.cpp
  )
  target_link_libraries(pytorch_integration_test
    PRIVATE
    rad_ml_pytorch
    ${CMAKE_THREAD_LIBS_INIT}
  )
  target_include_directories(pytorch_integration_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  )
  add_test(NAME pytorch_integration_test COMMAND pytorch_integration_test)

  message(STATUS "Added PyTorch integration test")

  # Add LibTorch resilience test
  add_executable(libtorch_resilience_test
    test/libtorch_resilience_test.cpp
  )
  target_link_libraries(libtorch_resilience_test
    PRIVATE
    rad_ml_pytorch
    ${CMAKE_THREAD_LIBS_INIT}
  )
  target_include_directories(libtorch_resilience_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${PyTorch_INCLUDE_DIR}
    ${PyTorch_INCLUDE_DIR}/torch/csrc/api/include
  )
  target_compile_definitions(libtorch_resilience_test PRIVATE
    RAD_ML_PYTORCH_ENABLED=1
  )
  add_test(NAME libtorch_resilience_test COMMAND libtorch_resilience_test)

  # Add LibTorch radiation integration test
  add_executable(libtorch_radiation_integration_test
    test/libtorch_radiation_integration_test.cpp
  )
  target_link_libraries(libtorch_radiation_integration_test
    PRIVATE
    rad_ml_pytorch
    ${CMAKE_THREAD_LIBS_INIT}
  )
  target_include_directories(libtorch_radiation_integration_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${PyTorch_INCLUDE_DIR}
    ${PyTorch_INCLUDE_DIR}/torch/csrc/api/include
  )
  target_compile_definitions(libtorch_radiation_integration_test PRIVATE
    RAD_ML_PYTORCH_ENABLED=1
  )
  add_test(NAME libtorch_radiation_integration_test COMMAND libtorch_radiation_integration_test)

  # Add LibTorch macOS compatibility test
  add_executable(libtorch_macos_compatibility_test
    test/libtorch_macos_compatibility_test.cpp
  )
  target_link_libraries(libtorch_macos_compatibility_test
    PRIVATE
    rad_ml_pytorch
    ${CMAKE_THREAD_LIBS_INIT}
  )
  target_include_directories(libtorch_macos_compatibility_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${PyTorch_INCLUDE_DIR}
    ${PyTorch_INCLUDE_DIR}/torch/csrc/api/include
  )
  target_compile_definitions(libtorch_macos_compatibility_test PRIVATE
    RAD_ML_PYTORCH_ENABLED=1
  )
  add_test(NAME libtorch_macos_compatibility_test COMMAND libtorch_macos_compatibility_test)

  message(STATUS "Added LibTorch test suite (resilience, radiation integration, macOS compatibility)")
endif()

# Add LMDB basic test if LMDB is available
if(LMDB_FOUND)
    add_executable(lmdb_basic_test
        test/lmdb_basic_test.cpp
    )
    target_link_libraries(lmdb_basic_test
        PRIVATE
        LMDB::LMDB
    )

    # Set C++17 standard for filesystem support
    target_compile_features(lmdb_basic_test PRIVATE cxx_std_17)

    add_test(NAME lmdb_basic_test COMMAND lmdb_basic_test)

    message(STATUS "Added LMDB basic test executable")

    # Add Datacenter LMDB Demo
    add_executable(lmdb_datacenter_demo
        test/lmdb_datacenter_demo.cpp
    )
    target_link_libraries(lmdb_datacenter_demo
        PRIVATE
        LMDB::LMDB
    )
    target_compile_features(lmdb_datacenter_demo PRIVATE cxx_std_17)

    add_test(NAME lmdb_datacenter_demo COMMAND lmdb_datacenter_demo)

    message(STATUS "Added LMDB Datacenter Demo")
endif()

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

  # Include pybind11 - try to find it first
  if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/pybind11")
    add_subdirectory(third_party/pybind11)
  else()
    # Try to find pybind11 via find_package
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
      message(WARNING "pybind11 not found. Python bindings will be disabled.")
      set(BUILD_PYTHON_BINDINGS OFF)
    endif()
  endif()

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
message(STATUS "PyTorch integration:        ${ENABLE_PYTORCH}")
if(ENABLE_PYTORCH)
  message(STATUS "PyTorch found:              ${PyTorch_FOUND}")
endif()
message(STATUS "IDE integration:            ${ENABLE_IDE_INTEGRATION}")
message(STATUS "Testing enabled:            ${BUILD_TESTING}")
message(STATUS "Visualization:              ${ENABLE_VISUALIZATION}")
message(STATUS "======================================================")

# C++17 Template Solutions and AI Database Tests
add_executable(simple_ai_database_test_minimal test/simple_ai_database_test_minimal.cpp)
target_link_libraries(simple_ai_database_test_minimal ${LMDB_LIBRARY})

# Add AI Native Database Test
if(LMDB_FOUND)
    add_executable(ai_native_database_test
        test/ai_native_database_test.cpp
        src/rad_ml/storage/ai_native_database.cpp
    )
    target_link_libraries(ai_native_database_test
        PRIVATE
        LMDB::LMDB
        rad_ml_research
    )
    target_include_directories(ai_native_database_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_features(ai_native_database_test PRIVATE cxx_std_17)
    add_test(NAME ai_native_database_test COMMAND ai_native_database_test)

    # Add Simple AI Database Test
    add_executable(simple_ai_database_test
        test/simple_ai_database_test.cpp
        src/rad_ml/storage/simple_ai_native_database.cpp
    )
    target_link_libraries(simple_ai_database_test
        PRIVATE
        LMDB::LMDB
    )
    target_include_directories(simple_ai_database_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_features(simple_ai_database_test PRIVATE cxx_std_17)
    add_test(NAME simple_ai_database_test COMMAND simple_ai_database_test)

    message(STATUS "Added AI Database test executables")
endif()

# Custom clean target to remove all generated executables and build artifacts
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E remove_directory build
    COMMAND ${CMAKE_COMMAND} -E remove_directory build_test
    COMMAND ${CMAKE_COMMAND} -E remove_directory build_python
    COMMAND ${CMAKE_COMMAND} -E remove_directory build2
    COMMAND ${CMAKE_COMMAND} -E remove_directory build_full
    COMMAND ${CMAKE_COMMAND} -E remove_directory build_dev
    COMMAND ${CMAKE_COMMAND} -E remove_directory build_minimal
    # Remove specific executables that might be in root directory
    COMMAND ${CMAKE_COMMAND} -E remove -f monte_carlo_validation
    COMMAND ${CMAKE_COMMAND} -E remove -f space_monte_carlo_validation
    COMMAND ${CMAKE_COMMAND} -E remove -f realistic_space_validation
    COMMAND ${CMAKE_COMMAND} -E remove -f framework_verification_test
    COMMAND ${CMAKE_COMMAND} -E remove -f enhanced_tmr_test
    COMMAND ${CMAKE_COMMAND} -E remove -f scientific_validation_test
    COMMAND ${CMAKE_COMMAND} -E remove -f radiation_stress_test
    COMMAND ${CMAKE_COMMAND} -E remove -f systematic_fault_test
    COMMAND ${CMAKE_COMMAND} -E remove -f modern_features_test
    COMMAND ${CMAKE_COMMAND} -E remove -f ieee754_tmr_test
    COMMAND ${CMAKE_COMMAND} -E remove -f quantum_field_validation_test
    COMMAND ${CMAKE_COMMAND} -E remove -f realistic_leo_mission_test
    COMMAND ${CMAKE_COMMAND} -E remove -f neural_network_validation
    COMMAND ${CMAKE_COMMAND} -E remove -f monte_carlo_neuralnetwork
    COMMAND ${CMAKE_COMMAND} -E remove -f quantum_field_example
    COMMAND ${CMAKE_COMMAND} -E remove -f ml_radiation_training_test
    COMMAND ${CMAKE_COMMAND} -E remove -f cross_validation_test
    COMMAND ${CMAKE_COMMAND} -E remove -f training_loop_diagnostic
    COMMAND ${CMAKE_COMMAND} -E remove -f simple_radiation_test
    COMMAND ${CMAKE_COMMAND} -E remove -f weight_diagnostic_test
    COMMAND ${CMAKE_COMMAND} -E remove -f trained_network_radiation_test
    COMMAND ${CMAKE_COMMAND} -E remove -f trivial_radiation_test
    COMMAND ${CMAKE_COMMAND} -E remove -f pytorch_integration_test
    COMMAND ${CMAKE_COMMAND} -E remove -f libtorch_resilience_test
    COMMAND ${CMAKE_COMMAND} -E remove -f libtorch_radiation_integration_test
    COMMAND ${CMAKE_COMMAND} -E remove -f libtorch_macos_compatibility_test
    COMMAND ${CMAKE_COMMAND} -E remove -f lmdb_basic_test
    COMMAND ${CMAKE_COMMAND} -E remove -f lmdb_datacenter_demo
    COMMAND ${CMAKE_COMMAND} -E remove -f simple_ai_database_test_minimal
    COMMAND ${CMAKE_COMMAND} -E remove -f ai_native_database_test
    COMMAND ${CMAKE_COMMAND} -E remove -f simple_ai_database_test
    COMMAND ${CMAKE_COMMAND} -E remove -f nasa_esa_standard_test
    # Remove debug executables
    COMMAND ${CMAKE_COMMAND} -E remove -f galois_field_debug
    COMMAND ${CMAKE_COMMAND} -E remove -f galois_field_correct_debug
    COMMAND ${CMAKE_COMMAND} -E remove -f galois_field_detailed_debug
    COMMAND ${CMAKE_COMMAND} -E remove -f galois_field_final_debug
    COMMAND ${CMAKE_COMMAND} -E remove -f galois_field_fixed_debug
    COMMAND ${CMAKE_COMMAND} -E remove -f galois_field_step_debug
    COMMAND ${CMAKE_COMMAND} -E remove -f galois_field_working_debug
    COMMAND ${CMAKE_COMMAND} -E remove -f minimal_adaptive_test
    COMMAND ${CMAKE_COMMAND} -E remove -f simple_adaptive_test
    COMMAND ${CMAKE_COMMAND} -E remove -f simple_galois_test
    COMMAND ${CMAKE_COMMAND} -E remove -f simple_shift_test
    COMMAND ${CMAKE_COMMAND} -E remove -f rs_memory_test
    COMMAND ${CMAKE_COMMAND} -E remove -f test_monte_carlo
    # Remove Python cache and build artifacts
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/rad_ml/__pycache__
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/__pycache__
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/test_tensor.pt
    # Remove database files
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/test_ai_db
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/test_async_db
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/test_error_db
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/test_types_db
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/test_minimal_db
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/compression_validation_db
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/optimal_anomaly_db
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/optimal_compression_db
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/quick_test_db
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/large_dataset_test_db
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/trained_vae_test_db
    # Remove example executables
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/mission_simulator/mission_simulator
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/simple_nn/simple_nn
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/architecture_test
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/residual_network_test
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/auto_arch_search_example
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/radiation_aware_training_example
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/multi_particle_simulation_example
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/vae_example
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/vae_space_mission_test
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/vae_validation_test
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/vae_production_example
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/vae_comprehensive_test
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/vae_monte_carlo_tuning_example
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/vae_quick_tuning_demo
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/quantum_enhanced_radiation_test
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/vae_production_database_example
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/vae_database_trained_test
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/vae_simple_usage_example
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/quick_tuning_test
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/compression_validation_test
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/large_dataset_compression_test
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/modern_cpp_training_example
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/cpu_optimized_training
    COMMAND ${CMAKE_COMMAND} -E remove -f examples/mnist_training_example
    COMMENT "Cleaning all build artifacts, executables, and database files"
    VERBATIM
)

# Custom clean target to remove additional artifacts (not overriding the default clean)
add_custom_target(clean_extras
    COMMAND ${CMAKE_COMMAND} -E echo "Running additional cleanup..."
    # Remove common executables
    COMMAND ${CMAKE_COMMAND} -E remove -f monte_carlo_validation
    COMMAND ${CMAKE_COMMAND} -E remove -f space_monte_carlo_validation
    COMMAND ${CMAKE_COMMAND} -E remove -f framework_verification_test
    COMMAND ${CMAKE_COMMAND} -E remove -f enhanced_tmr_test
    COMMAND ${CMAKE_COMMAND} -E remove -f pytorch_integration_test
    COMMAND ${CMAKE_COMMAND} -E remove -f libtorch_resilience_test
    COMMAND ${CMAKE_COMMAND} -E remove -f libtorch_radiation_integration_test
    COMMAND ${CMAKE_COMMAND} -E remove -f libtorch_macos_compatibility_test
    COMMAND ${CMAKE_COMMAND} -E remove -f ai_native_database_test
    COMMAND ${CMAKE_COMMAND} -E remove -f simple_ai_database_test
    COMMAND ${CMAKE_COMMAND} -E remove -f simple_ai_database_test_minimal
    COMMAND ${CMAKE_COMMAND} -E remove -f rs_memory_test
    COMMAND ${CMAKE_COMMAND} -E remove -f galois_field_*_debug
    COMMAND ${CMAKE_COMMAND} -E remove -f minimal_adaptive_test
    COMMAND ${CMAKE_COMMAND} -E remove -f simple_adaptive_test
    COMMAND ${CMAKE_COMMAND} -E remove -f simple_galois_test
    COMMAND ${CMAKE_COMMAND} -E remove -f simple_shift_test
    COMMAND ${CMAKE_COMMAND} -E remove -f test_monte_carlo
    # Remove Python artifacts
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/rad_ml/__pycache__
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/test_tensor.pt
    # Remove specific untracked executables that commonly appear
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/galois_field_correct_debug
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/galois_field_debug
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/galois_field_detailed_debug
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/galois_field_final_debug
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/galois_field_fixed_debug
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/galois_field_step_debug
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/galois_field_working_debug
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/minimal_adaptive_test
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/rs_memory_test
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/simple_adaptive_test
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/simple_galois_test
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/simple_shift_test
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_SOURCE_DIR}/test_monte_carlo
    COMMENT "Cleaning additional build artifacts and untracked executables"
    VERBATIM
)
