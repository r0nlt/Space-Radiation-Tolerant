add_library(rad_ml INTERFACE)
target_include_directories(rad_ml INTERFACE ${CMAKE_SOURCE_DIR}/include)

# Create directories if they don't exist
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/src/tmr)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/src/testing)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/src/memory) 