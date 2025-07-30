# PyTorch Integration Library
# This library provides PyTorch integration for the rad_ml framework

# Set sources for PyTorch integration
set(RAD_ML_PYTORCH_SOURCES
    pytorch_integration.cpp
    pytorch_tensor_wrapper.cpp
    pytorch_model_protection.cpp
    pytorch_radiation_hardening.cpp
)

# Create PyTorch integration library
add_library(rad_ml_pytorch ${RAD_ML_PYTORCH_SOURCES})

# Set include directories
target_include_directories(rad_ml_pytorch
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/rad_ml/pytorch
)

# Link dependencies
target_link_libraries(rad_ml_pytorch
    PUBLIC
        rad_ml_tmr
        rad_ml_testing
        rad_ml_research
)

# Link PyTorch if found
if(PyTorch_FOUND)
    # Add PyTorch include directories - need both root and API directories
    target_include_directories(rad_ml_pytorch
        PRIVATE
            ${PyTorch_INCLUDE_DIRS}
            ${PyTorch_INCLUDE_DIRS}/..
    )

    # Try to link PyTorch libraries
    if(PyTorch_LIBRARIES)
        target_link_libraries(rad_ml_pytorch
            PUBLIC
                ${PyTorch_LIBRARIES}
        )
    else()
        # Fallback: try to link the PyTorch target
        target_link_libraries(rad_ml_pytorch
            PUBLIC
                PyTorch::PyTorch
        )
    endif()

    # Add PyTorch-specific compile definitions
    target_compile_definitions(rad_ml_pytorch
        PRIVATE
            RAD_ML_PYTORCH_ENABLED=1
    )

    message(STATUS "PyTorch integration enabled")
    message(STATUS "PyTorch include dirs: ${PyTorch_INCLUDE_DIRS}")
    message(STATUS "PyTorch libraries: ${PyTorch_LIBRARIES}")
else()
    # Disable PyTorch features if not found
    target_compile_definitions(rad_ml_pytorch
        PRIVATE
            RAD_ML_PYTORCH_ENABLED=0
    )

    message(STATUS "PyTorch not found - PyTorch integration disabled")
endif()

# Set C++ standard
set_target_properties(rad_ml_pytorch PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Export library
install(TARGETS rad_ml_pytorch
    EXPORT RadMLTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)
