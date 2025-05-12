# CMake generated Testfile for 
# Source directory: /Users/rishabnuguru/rad-tolerant-ml
# Build directory: /Users/rishabnuguru/rad-tolerant-ml
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(monte_carlo_validation "/Users/rishabnuguru/rad-tolerant-ml/monte_carlo_validation")
set_tests_properties(monte_carlo_validation PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;109;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(space_monte_carlo_validation "/Users/rishabnuguru/rad-tolerant-ml/space_monte_carlo_validation")
set_tests_properties(space_monte_carlo_validation PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;115;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(realistic_space_validation "/Users/rishabnuguru/rad-tolerant-ml/realistic_space_validation")
set_tests_properties(realistic_space_validation PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;121;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(framework_verification_test "/Users/rishabnuguru/rad-tolerant-ml/framework_verification_test")
set_tests_properties(framework_verification_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;126;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(enhanced_tmr_test "/Users/rishabnuguru/rad-tolerant-ml/enhanced_tmr_test")
set_tests_properties(enhanced_tmr_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;130;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(scientific_validation_test "/Users/rishabnuguru/rad-tolerant-ml/scientific_validation_test")
set_tests_properties(scientific_validation_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;134;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(radiation_stress_test "/Users/rishabnuguru/rad-tolerant-ml/radiation_stress_test")
set_tests_properties(radiation_stress_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;138;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(systematic_fault_test "/Users/rishabnuguru/rad-tolerant-ml/systematic_fault_test")
set_tests_properties(systematic_fault_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;142;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(modern_features_test "/Users/rishabnuguru/rad-tolerant-ml/modern_features_test")
set_tests_properties(modern_features_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;146;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(quantum_field_validation_test "/Users/rishabnuguru/rad-tolerant-ml/quantum_field_validation_test")
set_tests_properties(quantum_field_validation_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;155;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
subdirs("src/tmr")
subdirs("src/rad_ml/research")
subdirs("examples")
