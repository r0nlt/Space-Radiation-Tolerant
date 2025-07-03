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

# Crystal Lattice Validation Test
add_executable(crystal_lattice_validation_test crystal_lattice_validation_test.cpp)
target_link_libraries(crystal_lattice_validation_test
    rad_ml_tmr
    rad_ml_testing
    rad_ml_qft
    rad_ml_research
)

if(OpenCV_FOUND)
    target_link_libraries(crystal_lattice_validation_test ${OpenCV_LIBS})
endif()

# Add the tests to CTest
add_test(NAME crystal_lattice_validation_test COMMAND crystal_lattice_validation_test)
