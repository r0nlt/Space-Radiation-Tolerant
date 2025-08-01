# FindPyTorch.cmake
# Custom PyTorch finder that works with Homebrew and other installations

# Prevent infinite recursion
if(PYTORCH_FIND_QUIETLY)
    return()
endif()

# Set flag to prevent recursion
set(PYTORCH_FIND_QUIETLY TRUE)

# Try to find PyTorch via Homebrew first (most common on macOS)
find_path(PYTORCH_HOMEBREW_ROOT
    NAMES lib/libtorch.dylib
    PATHS
        /usr/local/opt/pytorch
        /opt/homebrew/opt/pytorch
    NO_DEFAULT_PATH
)

if(PYTORCH_HOMEBREW_ROOT)
    message(STATUS "Found PyTorch via Homebrew at: ${PYTORCH_HOMEBREW_ROOT}")

    # Set up PyTorch variables for Homebrew installation
    set(PyTorch_FOUND TRUE)
    set(PyTorch_INCLUDE_DIR ${PYTORCH_HOMEBREW_ROOT}/include)
    set(PyTorch_LIBRARY ${PYTORCH_HOMEBREW_ROOT}/lib/libtorch.dylib)

    # Find all required libraries
    find_library(PYTORCH_C10_LIBRARY
        NAMES libc10.dylib
        PATHS ${PYTORCH_HOMEBREW_ROOT}/lib
        NO_DEFAULT_PATH
    )

    find_library(PYTORCH_CPU_LIBRARY
        NAMES libtorch_cpu.dylib
        PATHS ${PYTORCH_HOMEBREW_ROOT}/lib
        NO_DEFAULT_PATH
    )

    find_library(PYTORCH_GLOBAL_DEPS_LIBRARY
        NAMES libtorch_global_deps.dylib
        PATHS ${PYTORCH_HOMEBREW_ROOT}/lib
        NO_DEFAULT_PATH
    )

    # Build library list
    set(PyTorch_LIBRARIES ${PyTorch_LIBRARY})
    if(PYTORCH_C10_LIBRARY)
        list(APPEND PyTorch_LIBRARIES ${PYTORCH_C10_LIBRARY})
    endif()
    if(PYTORCH_CPU_LIBRARY)
        list(APPEND PyTorch_LIBRARIES ${PYTORCH_CPU_LIBRARY})
    endif()
    if(PYTORCH_GLOBAL_DEPS_LIBRARY)
        list(APPEND PyTorch_LIBRARIES ${PYTORCH_GLOBAL_DEPS_LIBRARY})
    endif()

    message(STATUS "PyTorch include directory: ${PyTorch_INCLUDE_DIR}")
    message(STATUS "PyTorch libraries: ${PyTorch_LIBRARIES}")

else()
    # Fallback: Try to find PyTorch in common locations
    find_path(PyTorch_INCLUDE_DIR
        NAMES torch/torch.h
        PATHS
            $ENV{PYTORCH_ROOT}/include
            /usr/local/include
            /opt/local/include
            /usr/include
        PATH_SUFFIXES
            torch
            pytorch
    )

    find_library(PyTorch_LIBRARY
        NAMES libtorch.dylib libtorch.so
        PATHS
            $ENV{PYTORCH_ROOT}/lib
            /usr/local/lib
            /opt/local/lib
            /usr/lib
    )

    if(PyTorch_INCLUDE_DIR AND PyTorch_LIBRARY)
        set(PyTorch_FOUND TRUE)
        set(PyTorch_LIBRARIES ${PyTorch_LIBRARY})

        # Try to find additional libraries
        get_filename_component(PYTORCH_LIB_DIR "${PyTorch_LIBRARY}" DIRECTORY)

        find_library(PYTORCH_C10_LIBRARY
            NAMES libc10.dylib libc10.so
            PATHS ${PYTORCH_LIB_DIR}
            NO_DEFAULT_PATH
        )

        find_library(PYTORCH_CPU_LIBRARY
            NAMES libtorch_cpu.dylib libtorch_cpu.so
            PATHS ${PYTORCH_LIB_DIR}
            NO_DEFAULT_PATH
        )

        if(PYTORCH_C10_LIBRARY)
            list(APPEND PyTorch_LIBRARIES ${PYTORCH_C10_LIBRARY})
        endif()
        if(PYTORCH_CPU_LIBRARY)
            list(APPEND PyTorch_LIBRARIES ${PYTORCH_CPU_LIBRARY})
        endif()

        message(STATUS "Found PyTorch: ${PyTorch_LIBRARY}")
        message(STATUS "PyTorch include directory: ${PyTorch_INCLUDE_DIR}")
        message(STATUS "PyTorch libraries: ${PyTorch_LIBRARIES}")
    endif()
endif()

# Handle the case where PyTorch is not found
if(NOT PyTorch_FOUND)
    message(WARNING "PyTorch not found. Install via Homebrew: brew install pytorch")
    message(WARNING "Or download from: https://pytorch.org/get-started/locally/")
endif()

# Mark variables as advanced
mark_as_advanced(
    PyTorch_LIBRARY
    PyTorch_INCLUDE_DIR
    PyTorch_LIBRARIES
    PYTORCH_C10_LIBRARY
    PYTORCH_CPU_LIBRARY
    PYTORCH_GLOBAL_DEPS_LIBRARY
    PYTORCH_HOMEBREW_ROOT
    PYTORCH_LIB_DIR
)
