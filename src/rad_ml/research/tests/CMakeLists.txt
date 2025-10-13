# File: src/rad_ml/research/tests/CMakeLists.txt

set(RAD_ML_RESEARCH_TEST_SOURCES
    architecture_tester_test.cpp
    residual_network_test.cpp
    evolutionary_ga_test.cpp
    ieee_qrs_ga_qd_validation_test.cpp
)

# Create test executable
add_executable(rad_ml_research_tests ${RAD_ML_RESEARCH_TEST_SOURCES})

# Link dependencies - only link to libraries that exist
target_link_libraries(rad_ml_research_tests
    PRIVATE
        rad_ml_research
)

# Only link against GTest if it's found
if(GTEST_FOUND)
    target_link_libraries(rad_ml_research_tests
        PRIVATE
            ${GTEST_LIBRARIES}
            GTest::gtest_main
    )
else()
    message(WARNING "GTest not found, rad_ml_research_tests may not compile")
endif()

# Add tests to CTest
add_test(NAME RadML_Research_Tests COMMAND rad_ml_research_tests)
