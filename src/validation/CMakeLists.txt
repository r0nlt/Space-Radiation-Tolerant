# Validation tools for NASA/ESA radiation tolerance protocol
cmake_minimum_required(VERSION 3.10)

# Add the validation header
set(VALIDATION_HEADERS
    radiation_accuracy_validator.hpp
    radiation_validation_report.hpp
    cross_section_utils.hpp
)

# Validation test executable
add_executable(radiation_validation_test
    radiation_validation_test.cpp
)

target_include_directories(radiation_validation_test 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(radiation_validation_test
    PRIVATE
)

# Install
install(TARGETS radiation_validation_test
    RUNTIME DESTINATION bin
)

# Add radiation SEU validation benchmark executable
add_executable(seu_benchmark
    seu_benchmark.cpp
)

target_include_directories(seu_benchmark
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(seu_benchmark
    PRIVATE
)

# Install
install(TARGETS seu_benchmark
    RUNTIME DESTINATION bin
) 