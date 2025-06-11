# File: src/rad_ml/research/CMakeLists.txt

# Set sources for research components
set(RAD_ML_RESEARCH_SOURCES
    architecture_tester.cpp
    auto_arch_search.cpp
    radiation_aware_training.cpp
    variational_autoencoder.cpp
    # Other source files will be added as they are implemented
)

# Create research library
add_library(rad_ml_research ${RAD_ML_RESEARCH_SOURCES})

# Set include directories
target_include_directories(rad_ml_research
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# Link dependencies - remove non-existent dependencies for now
# to get the build working
target_link_libraries(rad_ml_research
    PUBLIC
)

# Export library
install(TARGETS rad_ml_research
    EXPORT RadMLTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Build tests if enabled
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
