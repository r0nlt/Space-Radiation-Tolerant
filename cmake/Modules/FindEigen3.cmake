# - Try to find Eigen3 include dirs
#
# The following variables are set:
#   EIGEN3_FOUND        - System has Eigen3
#   EIGEN3_INCLUDE_DIRS - Eigen3 include directories
#   EIGEN3_INCLUDE_DIR  - Eigen3 include directory (same as EIGEN3_INCLUDE_DIRS for backward compatibility)
#
# The following cache variables are also available:
#   EIGEN3_INCLUDE_DIR  - Directory containing the Eigen3 header files

# Check if EIGEN3_INCLUDE_DIR is already defined
if (EIGEN3_INCLUDE_DIR)
  set(EIGEN3_FOUND TRUE)
  return()
endif()

# Common paths for Eigen3
set(_EIGEN3_COMMON_INCLUDE_DIRS
  /usr/include/eigen3
  /usr/local/include/eigen3
  /opt/local/include/eigen3
  ${CMAKE_INSTALL_PREFIX}/include/eigen3
  /opt/homebrew/include/eigen3
)

# Try to find the Eigen3 include dir
find_path(EIGEN3_INCLUDE_DIR
  NAMES Eigen/Core
  PATHS ${_EIGEN3_COMMON_INCLUDE_DIRS}
)

# Handle the QUIETLY and REQUIRED arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen3 DEFAULT_MSG EIGEN3_INCLUDE_DIR)

# Define output variables
if(EIGEN3_FOUND)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
  
  # Create an interface target for Eigen3
  if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIRS}"
    )
  endif()
endif()

# Mark as advanced
mark_as_advanced(EIGEN3_INCLUDE_DIR) 