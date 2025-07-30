# FindPyTorch.cmake - Find PyTorch (LibTorch) installation
#
# This module finds the PyTorch C++ library (LibTorch) and sets up the necessary
# variables and targets for use in CMake projects.
#
# Variables defined:
#   PyTorch_FOUND - True if PyTorch was found
#   PyTorch_INCLUDE_DIRS - PyTorch include directories
#   PyTorch_LIBRARIES - PyTorch libraries
#   PyTorch_VERSION - PyTorch version
#
# Targets defined:
#   PyTorch::PyTorch - Imported target for PyTorch

# Set minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Try to find PyTorch via Python first
find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    # Get PyTorch path from Python
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__file__)"
        OUTPUT_VARIABLE PYTORCH_PYTHON_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(PYTORCH_PYTHON_PATH)
        # Extract the directory containing PyTorch
        get_filename_component(PYTORCH_ROOT "${PYTORCH_PYTHON_PATH}" DIRECTORY)
        get_filename_component(PYTORCH_ROOT "${PYTORCH_ROOT}" DIRECTORY)
        get_filename_component(PYTORCH_ROOT "${PYTORCH_ROOT}" DIRECTORY)
        get_filename_component(PYTORCH_ROOT "${PYTORCH_ROOT}" DIRECTORY)

        message(STATUS "Found PyTorch via Python at: ${PYTORCH_ROOT}")
    endif()
endif()

# Find PyTorch installation - look for the root include directory
find_path(PyTorch_INCLUDE_DIR
    NAMES torch/torch.h
    HINTS
        ${PYTORCH_ROOT}/include
        ${PyTorch_ROOT}/include
        $ENV{PYTORCH_ROOT}/include
        /usr/local/include
        /usr/include
        /opt/local/include
        /opt/homebrew/include
        /usr/local/opt/pytorch/include
        /usr/local/opt/libtorch/include
        /opt/homebrew/opt/libtorch/include
    PATH_SUFFIXES
        torch
        pytorch
        libtorch
)

# If not found, try to find it in the Python site-packages
if(NOT PyTorch_INCLUDE_DIR AND PYTORCH_ROOT)
    find_path(PyTorch_INCLUDE_DIR
        NAMES torch/torch.h
        HINTS
            ${PYTORCH_ROOT}/include
        PATH_SUFFIXES
            torch
            pytorch
            libtorch
    )
endif()

# Find PyTorch libraries
find_library(PyTorch_LIBRARY
    NAMES torch torch_cpu torch_cuda
    HINTS
        ${PYTORCH_ROOT}/lib
        ${PyTorch_ROOT}/lib
        $ENV{PYTORCH_ROOT}/lib
        /usr/local/lib
        /usr/lib
        /opt/local/lib
        /opt/homebrew/lib
        /usr/local/opt/pytorch/lib
        /usr/local/opt/libtorch/lib
        /opt/homebrew/opt/libtorch/lib
)

# Find additional PyTorch libraries
find_library(PyTorch_C10_LIBRARY
    NAMES c10 c10_cpu c10_cuda
    HINTS
        ${PYTORCH_ROOT}/lib
        ${PyTorch_ROOT}/lib
        $ENV{PYTORCH_ROOT}/lib
        /usr/local/lib
        /usr/lib
        /opt/local/lib
        /opt/homebrew/lib
        /usr/local/opt/pytorch/lib
        /usr/local/opt/libtorch/lib
        /opt/homebrew/opt/libtorch/lib
)

# Check if PyTorch was found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PyTorch
    REQUIRED_VARS PyTorch_INCLUDE_DIR PyTorch_LIBRARY
    VERSION_VAR PyTorch_VERSION
)

# Set up variables
if(PyTorch_FOUND)
    set(PyTorch_INCLUDE_DIRS ${PyTorch_INCLUDE_DIR})
    set(PyTorch_LIBRARIES ${PyTorch_LIBRARY})

    # Add c10 library if found
    if(PyTorch_C10_LIBRARY)
        list(APPEND PyTorch_LIBRARIES ${PyTorch_C10_LIBRARY})
    endif()

    # Create imported target
    if(NOT TARGET PyTorch::PyTorch)
        add_library(PyTorch::PyTorch UNKNOWN IMPORTED)
        set_target_properties(PyTorch::PyTorch PROPERTIES
            IMPORTED_LOCATION "${PyTorch_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${PyTorch_INCLUDE_DIR}"
        )

        # Add c10 library to the target if found
        if(PyTorch_C10_LIBRARY)
            set_target_properties(PyTorch::PyTorch PROPERTIES
                INTERFACE_LINK_LIBRARIES "${PyTorch_C10_LIBRARY}"
            )
        endif()
    endif()

    # Print status
    message(STATUS "Found PyTorch: ${PyTorch_LIBRARY}")
    message(STATUS "PyTorch include directory: ${PyTorch_INCLUDE_DIR}")
    if(PyTorch_C10_LIBRARY)
        message(STATUS "PyTorch C10 library: ${PyTorch_C10_LIBRARY}")
    endif()
else()
    message(WARNING "PyTorch not found. PyTorch integration will be disabled.")
endif()

# Mark variables as advanced
mark_as_advanced(PyTorch_INCLUDE_DIR PyTorch_LIBRARY PyTorch_C10_LIBRARY)
