# CMake generated Testfile for 
# Source directory: /Users/rishabnuguru/space/examples
# Build directory: /Users/rishabnuguru/space/examples
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(architecture_test_run "/Users/rishabnuguru/space/examples/architecture_test")
set_tests_properties(architecture_test_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;59;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
add_test(residual_network_test_run "/Users/rishabnuguru/space/examples/residual_network_test")
set_tests_properties(residual_network_test_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;60;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
add_test(auto_arch_search_example_run "/Users/rishabnuguru/space/examples/auto_arch_search_example")
set_tests_properties(auto_arch_search_example_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;61;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
add_test(radiation_aware_training_example_run "/Users/rishabnuguru/space/examples/radiation_aware_training_example")
set_tests_properties(radiation_aware_training_example_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;62;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
add_test(multi_particle_simulation_example_test "/Users/rishabnuguru/space/examples/multi_particle_simulation_example")
set_tests_properties(multi_particle_simulation_example_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;63;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
add_test(vae_example_run "/Users/rishabnuguru/space/examples/vae_example")
set_tests_properties(vae_example_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;64;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
add_test(vae_space_mission_test_run "/Users/rishabnuguru/space/examples/vae_space_mission_test")
set_tests_properties(vae_space_mission_test_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/examples/CMakeLists.txt;65;add_test;/Users/rishabnuguru/space/examples/CMakeLists.txt;0;")
subdirs("simple_nn")
subdirs("mission_simulator")
