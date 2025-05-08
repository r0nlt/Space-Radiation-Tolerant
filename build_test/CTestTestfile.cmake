# CMake generated Testfile for 
# Source directory: /Users/rishabnuguru/rad-tolerant-ml
# Build directory: /Users/rishabnuguru/rad-tolerant-ml/build_test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(monte_carlo_validation "/Users/rishabnuguru/rad-tolerant-ml/build_test/monte_carlo_validation")
set_tests_properties(monte_carlo_validation PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;59;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(space_monte_carlo_validation "/Users/rishabnuguru/rad-tolerant-ml/build_test/space_monte_carlo_validation")
set_tests_properties(space_monte_carlo_validation PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;65;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(realistic_space_validation "/Users/rishabnuguru/rad-tolerant-ml/build_test/realistic_space_validation")
set_tests_properties(realistic_space_validation PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;71;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(framework_verification_test "/Users/rishabnuguru/rad-tolerant-ml/build_test/framework_verification_test")
set_tests_properties(framework_verification_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;76;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(enhanced_tmr_test "/Users/rishabnuguru/rad-tolerant-ml/build_test/enhanced_tmr_test")
set_tests_properties(enhanced_tmr_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;80;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(scientific_validation_test "/Users/rishabnuguru/rad-tolerant-ml/build_test/scientific_validation_test")
set_tests_properties(scientific_validation_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;84;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(radiation_stress_test "/Users/rishabnuguru/rad-tolerant-ml/build_test/radiation_stress_test")
set_tests_properties(radiation_stress_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;88;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(systematic_fault_test "/Users/rishabnuguru/rad-tolerant-ml/build_test/systematic_fault_test")
set_tests_properties(systematic_fault_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;92;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
add_test(modern_features_test "/Users/rishabnuguru/rad-tolerant-ml/build_test/modern_features_test")
set_tests_properties(modern_features_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;96;add_test;/Users/rishabnuguru/rad-tolerant-ml/CMakeLists.txt;0;")
subdirs("src/tmr")
