# Validation tests for the rad-tolerant ML framework

# Include OpenCV if available for visualization
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    add_definitions(-DENABLE_VISUALIZATION)
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(STATUS "OpenCV not found. Visualization will be disabled.")
endif()

# Neural Network Validation Test
add_executable(neural_network_validation neural_network_validation.cpp)
target_link_libraries(neural_network_validation rad_tolerant_ml)

if(OpenCV_FOUND)
    target_link_libraries(neural_network_validation ${OpenCV_LIBS})
endif()

# Monte Carlo Validation Test
add_executable(monte_carlo_validation monte_carlo_validation.cpp)
target_link_libraries(monte_carlo_validation rad_tolerant_ml)

if(OpenCV_FOUND)
    target_link_libraries(monte_carlo_validation ${OpenCV_LIBS})
endif()

# Add the tests to CTest
add_test(NAME neural_network_validation COMMAND neural_network_validation)
add_test(NAME monte_carlo_validation COMMAND monte_carlo_validation) 