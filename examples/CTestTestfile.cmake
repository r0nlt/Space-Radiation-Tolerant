# CMake generated Testfile for 
# Source directory: /Users/rishabnuguru/Space-Radiation-Tolerant/examples
# Build directory: /Users/rishabnuguru/Space-Radiation-Tolerant/examples
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(architecture_test_run "/Users/rishabnuguru/Space-Radiation-Tolerant/examples/architecture_test")
set_tests_properties(architecture_test_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/Space-Radiation-Tolerant/examples/CMakeLists.txt;31;add_test;/Users/rishabnuguru/Space-Radiation-Tolerant/examples/CMakeLists.txt;0;")
add_test(residual_network_test_run "/Users/rishabnuguru/Space-Radiation-Tolerant/examples/residual_network_test")
set_tests_properties(residual_network_test_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/Space-Radiation-Tolerant/examples/CMakeLists.txt;32;add_test;/Users/rishabnuguru/Space-Radiation-Tolerant/examples/CMakeLists.txt;0;")
add_test(auto_arch_search_example_run "/Users/rishabnuguru/Space-Radiation-Tolerant/examples/auto_arch_search_example")
set_tests_properties(auto_arch_search_example_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/Space-Radiation-Tolerant/examples/CMakeLists.txt;33;add_test;/Users/rishabnuguru/Space-Radiation-Tolerant/examples/CMakeLists.txt;0;")
add_test(radiation_aware_training_example_run "/Users/rishabnuguru/Space-Radiation-Tolerant/examples/radiation_aware_training_example")
set_tests_properties(radiation_aware_training_example_run PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/Space-Radiation-Tolerant/examples/CMakeLists.txt;34;add_test;/Users/rishabnuguru/Space-Radiation-Tolerant/examples/CMakeLists.txt;0;")
subdirs("simple_nn")
subdirs("mission_simulator")
