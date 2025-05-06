add_library(rad_ml_tmr
    adaptive_protection.cpp
)

target_include_directories(rad_ml_tmr PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Handle Eigen dependency
if(Eigen3_FOUND OR EIGEN3_FOUND)
    target_link_libraries(rad_ml_tmr PUBLIC Eigen3::Eigen)
endif()

set_target_properties(rad_ml_tmr PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
) 