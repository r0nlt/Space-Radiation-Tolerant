cmake_minimum_required(VERSION 3.15)
project(RadToleranceML VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(RAD_ML_BUILD_TESTS "Build tests" ON)
option(RAD_ML_BUILD_EXAMPLES "Build examples" ON)
option(RAD_ML_ENABLE_WARNINGS "Enable all warnings" ON)
option(RAD_ML_BUILD_STRESS_TEST "Build radiation stress test" ON)

# Add compiler warnings
if(RAD_ML_ENABLE_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    elseif(MSVC)
        add_compile_options(/W4 /WX)
    endif()
endif()

# Include directories
include_directories(include)

# Add subdirectories
add_subdirectory(src)

if(RAD_ML_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(RAD_ML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Build the stress test
if(RAD_ML_BUILD_STRESS_TEST)
    add_executable(stress_test stress_test.cpp)
    target_link_libraries(stress_test PRIVATE rad_ml)
endif() 