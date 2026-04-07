# File: src/rad_ml/research/CMakeLists.txt

# Set sources for research components
set(RAD_ML_RESEARCH_SOURCES
    architecture_tester.cpp
    auto_arch/io.cpp
    auto_arch/generation.cpp
    auto_arch/variation.cpp
    auto_arch/grid.cpp
    auto_arch/random.cpp
    auto_arch/evolutionary.cpp
    auto_arch/genetic_operators.cpp
    auto_arch/auto_arch_search.cpp
    radiation_aware_training.cpp
    variational_autoencoder.cpp
)

if(LMDB_FOUND)
    list(APPEND RAD_ML_RESEARCH_SOURCES vae_auto_tuner.cpp)
endif()

# Create research library
add_library(rad_ml_research ${RAD_ML_RESEARCH_SOURCES})

# Set include directories
target_include_directories(rad_ml_research
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# Link dependencies
target_link_libraries(rad_ml_research
    PUBLIC
        Eigen3::Eigen
)

if(LMDB_FOUND)
    target_link_libraries(rad_ml_research PUBLIC LMDB::LMDB)
endif()

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
