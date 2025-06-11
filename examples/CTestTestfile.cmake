# CMake generated Testfile for 
# Source directory: /Users/rishabnuguru/space/examples
# Build directory: /Users/rishabnuguru/space/examples
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(architecture_test_run "/Users/rishabnuguru/space/examples/architecture_test")
set_tests_properties(architecture_test_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;78;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
add_test(residual_network_test_run "/Users/rishabnuguru/space/examples/residual_network_test")
set_tests_properties(residual_network_test_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;79;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
add_test(auto_arch_search_example_run "/Users/rishabnuguru/space/examples/auto_arch_search_example")
set_tests_properties(auto_arch_search_example_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;80;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
add_test(radiation_aware_training_example_run "/Users/rishabnuguru/space/examples/radiation_aware_training_example")
set_tests_properties(radiation_aware_training_example_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;81;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
add_test(multi_particle_simulation_example_test "/Users/rishabnuguru/space/examples/multi_particle_simulation_example")
set_tests_properties(multi_particle_simulation_example_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;82;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
add_test(vae_example_run "/Users/rishabnuguru/space/examples/vae_example")
set_tests_properties(vae_example_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;83;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
add_test(vae_space_mission_test_run "/Users/rishabnuguru/space/examples/vae_space_mission_test")
set_tests_properties(vae_space_mission_test_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;84;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
add_test(vae_validation_test_run "/Users/rishabnuguru/space/examples/vae_validation_test")
set_tests_properties(vae_validation_test_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;85;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
add_test(vae_production_example_run "/Users/rishabnuguru/space/examples/vae_production_example")
set_tests_properties(vae_production_example_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;86;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
add_test(vae_comprehensive_test "/Users/rishabnuguru/space/examples/vae_comprehensive_test")
set_tests_properties(vae_comprehensive_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;87;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
subdirs("simple_nn")
subdirs("mission_simulator")
