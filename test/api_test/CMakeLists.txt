cmake_minimum_required(VERSION 3.10)

# Find Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add the executable
add_executable(custom_defense_api_test
    custom_defense_api_test.cpp
)

# Link against Google Test and the project libraries
target_link_libraries(custom_defense_api_test
    ${GTEST_LIBRARIES}
    pthread
    rad_ml_core
    rad_ml_api
    rad_ml_neural
    rad_ml_tmr
    rad_ml_sim
    rad_ml_hw
)

# Add the test to CTest
add_test(NAME custom_defense_api_test COMMAND custom_defense_api_test)

# Set the test timeout (2 minutes)
set_tests_properties(custom_defense_api_test PROPERTIES TIMEOUT 120)

# Create a custom target to run only the API tests
add_custom_target(run_api_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -R "custom_defense_api_test" --output-on-failure
    DEPENDS custom_defense_api_test
    COMMENT "Running API tests..."
)

# Create a custom target to run the Python tests
add_custom_target(run_python_tests
    COMMAND ${CMAKE_COMMAND} -E echo "Running Python API tests..."
    COMMAND pytest ${CMAKE_SOURCE_DIR}/python/rad_ml/test_unified_defense.py -v
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running Python API tests..."
)

# Add all tests to a single target
add_custom_target(run_all_api_tests
    DEPENDS run_api_tests run_python_tests
    COMMENT "Running all API tests..."
)
