cmake_minimum_required(VERSION 3.10)
project(rad_tolerant_ml VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add include directory to include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Set compiler options based on compiler
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

# Add the source directory
add_subdirectory(src)

# Add validation tools subdirectory
add_subdirectory(src/validation)

# Add enhanced features test executable
add_executable(enhanced_features_test src/enhanced_features_test.cpp)
target_include_directories(enhanced_features_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Add mission simulation test executable
add_executable(mission_simulation_test src/mission_simulation_test.cpp)
target_include_directories(mission_simulation_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Add industry standard validation test executable
add_executable(industry_standard_test src/industry_standard_test.cpp)
target_include_directories(industry_standard_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Add stress test executable if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/extreme_stress_test.cpp")
    add_executable(extreme_stress_test src/extreme_stress_test.cpp)
    target_include_directories(extreme_stress_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Add simple test executable if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/simple_test.cpp")
    add_executable(simple_test src/simple_test.cpp)
    target_include_directories(simple_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Enable testing
enable_testing()
add_test(NAME enhanced_features_test COMMAND enhanced_features_test)
add_test(NAME mission_simulation_test COMMAND mission_simulation_test)
add_test(NAME industry_standard_test COMMAND industry_standard_test)

# If stress test exists, add it as a test
if(TARGET extreme_stress_test)
    add_test(NAME extreme_stress_test COMMAND extreme_stress_test)
endif()

# If simple test exists, add it as a test
if(TARGET simple_test)
    add_test(NAME simple_test COMMAND simple_test)
endif() 