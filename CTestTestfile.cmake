# CMake generated Testfile for 
# Source directory: /Users/rishabnuguru/Space-Radiation-Tolerant
# Build directory: /Users/rishabnuguru/Space-Radiation-Tolerant
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(monte_carlo_validation "/Users/rishabnuguru/Space-Radiation-Tolerant/monte_carlo_validation")
set_tests_properties(monte_carlo_validation PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;113;add_test;/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;0;")
add_test(space_monte_carlo_validation "/Users/rishabnuguru/Space-Radiation-Tolerant/space_monte_carlo_validation")
set_tests_properties(space_monte_carlo_validation PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;119;add_test;/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;0;")
add_test(realistic_space_validation "/Users/rishabnuguru/Space-Radiation-Tolerant/realistic_space_validation")
set_tests_properties(realistic_space_validation PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;125;add_test;/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;0;")
add_test(framework_verification_test "/Users/rishabnuguru/Space-Radiation-Tolerant/framework_verification_test")
set_tests_properties(framework_verification_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;130;add_test;/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;0;")
add_test(enhanced_tmr_test "/Users/rishabnuguru/Space-Radiation-Tolerant/enhanced_tmr_test")
set_tests_properties(enhanced_tmr_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;134;add_test;/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;0;")
add_test(scientific_validation_test "/Users/rishabnuguru/Space-Radiation-Tolerant/scientific_validation_test")
set_tests_properties(scientific_validation_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;138;add_test;/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;0;")
add_test(radiation_stress_test "/Users/rishabnuguru/Space-Radiation-Tolerant/radiation_stress_test")
set_tests_properties(radiation_stress_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;142;add_test;/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;0;")
add_test(systematic_fault_test "/Users/rishabnuguru/Space-Radiation-Tolerant/systematic_fault_test")
set_tests_properties(systematic_fault_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;146;add_test;/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;0;")
add_test(modern_features_test "/Users/rishabnuguru/Space-Radiation-Tolerant/modern_features_test")
set_tests_properties(modern_features_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;150;add_test;/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;0;")
add_test(quantum_field_validation_test "/Users/rishabnuguru/Space-Radiation-Tolerant/quantum_field_validation_test")
set_tests_properties(quantum_field_validation_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;159;add_test;/Users/rishabnuguru/Space-Radiation-Tolerant/CMakeLists.txt;0;")
subdirs("src/tmr")
subdirs("src/rad_ml/research")
subdirs("examples")
