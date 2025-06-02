find_package(GTest REQUIRED)

add_executable(space_containers_test
    fixed_vector_test.cpp
    radiation_hardened_vector_test.cpp
    fixed_map_test.cpp
)

target_link_libraries(space_containers_test
    PRIVATE
    space_containers
    GTest::GTest
    GTest::Main
)

add_test(NAME space_containers_test COMMAND space_containers_test)
