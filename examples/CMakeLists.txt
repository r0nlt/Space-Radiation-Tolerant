add_subdirectory(simple_nn)
add_subdirectory(mission_simulator)

# Add the architecture testing example
add_executable(architecture_test architecture_test.cpp)
target_link_libraries(architecture_test
    rad_ml_research
    rad_ml_tmr
    rad_ml_testing
    rad_ml_qft
)

# Add the residual network example
add_executable(residual_network_test residual_network_test.cpp)
target_link_libraries(residual_network_test
    rad_ml_research
    rad_ml_tmr
    rad_ml_testing
    rad_ml_qft
)

# Add auto_arch_search_example
add_executable(auto_arch_search_example auto_arch_search_example.cpp)
target_link_libraries(auto_arch_search_example rad_ml_research rad_ml_tmr)

# Add radiation_aware_training_example
add_executable(radiation_aware_training_example radiation_aware_training_example.cpp)
target_link_libraries(radiation_aware_training_example rad_ml_research)

# Add multi-particle simulation example
add_executable(multi_particle_simulation_example multi_particle_simulation_example.cpp)
target_link_libraries(multi_particle_simulation_example
    rad_ml_qft
    rad_ml_tmr
    rad_ml_testing
    rad_ml_research
)
target_include_directories(multi_particle_simulation_example PRIVATE ${CMAKE_SOURCE_DIR})

# Add VAE example
add_executable(vae_example vae_example.cpp)
target_link_libraries(vae_example
    rad_ml_research
    rad_ml_tmr
    rad_ml_testing
)
target_include_directories(vae_example PRIVATE ${CMAKE_SOURCE_DIR})

# Add comprehensive VAE space mission test
add_executable(vae_space_mission_test vae_space_mission_test.cpp)
target_link_libraries(vae_space_mission_test
    rad_ml_research
    rad_ml_tmr
    rad_ml_testing
)
target_include_directories(vae_space_mission_test PRIVATE ${CMAKE_SOURCE_DIR})

# Add tests
add_test(NAME architecture_test_run COMMAND architecture_test)
add_test(NAME residual_network_test_run COMMAND residual_network_test)
add_test(NAME auto_arch_search_example_run COMMAND auto_arch_search_example)
add_test(NAME radiation_aware_training_example_run COMMAND radiation_aware_training_example)
add_test(NAME multi_particle_simulation_example_test COMMAND multi_particle_simulation_example)
add_test(NAME vae_example_run COMMAND vae_example)
add_test(NAME vae_space_mission_test_run COMMAND vae_space_mission_test)

# Installation
install(TARGETS architecture_test residual_network_test auto_arch_search_example radiation_aware_training_example multi_particle_simulation_example vae_example vae_space_mission_test
    DESTINATION bin/examples)

# These will be enabled as they are implemented
# add_subdirectory(fault_injection)
# add_subdirectory(radiation_sim)
