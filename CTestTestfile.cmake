# CMake generated Testfile for 
# Source directory: /Users/rishabnuguru/space
# Build directory: /Users/rishabnuguru/space
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(monte_carlo_validation "/Users/rishabnuguru/space/monte_carlo_validation")
set_tests_properties(monte_carlo_validation PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/CMakeLists.txt;129;add_test;/Users/rishabnuguru/space/CMakeLists.txt;0;")
add_test(space_monte_carlo_validation "/Users/rishabnuguru/space/space_monte_carlo_validation")
set_tests_properties(space_monte_carlo_validation PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/CMakeLists.txt;135;add_test;/Users/rishabnuguru/space/CMakeLists.txt;0;")
add_test(realistic_space_validation "/Users/rishabnuguru/space/realistic_space_validation")
set_tests_properties(realistic_space_validation PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/CMakeLists.txt;141;add_test;/Users/rishabnuguru/space/CMakeLists.txt;0;")
add_test(framework_verification_test "/Users/rishabnuguru/space/framework_verification_test")
set_tests_properties(framework_verification_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/CMakeLists.txt;146;add_test;/Users/rishabnuguru/space/CMakeLists.txt;0;")
add_test(enhanced_tmr_test "/Users/rishabnuguru/space/enhanced_tmr_test")
set_tests_properties(enhanced_tmr_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/CMakeLists.txt;150;add_test;/Users/rishabnuguru/space/CMakeLists.txt;0;")
add_test(scientific_validation_test "/Users/rishabnuguru/space/scientific_validation_test")
set_tests_properties(scientific_validation_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/CMakeLists.txt;154;add_test;/Users/rishabnuguru/space/CMakeLists.txt;0;")
add_test(radiation_stress_test "/Users/rishabnuguru/space/radiation_stress_test")
set_tests_properties(radiation_stress_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/CMakeLists.txt;158;add_test;/Users/rishabnuguru/space/CMakeLists.txt;0;")
add_test(systematic_fault_test "/Users/rishabnuguru/space/systematic_fault_test")
set_tests_properties(systematic_fault_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/CMakeLists.txt;162;add_test;/Users/rishabnuguru/space/CMakeLists.txt;0;")
add_test(modern_features_test "/Users/rishabnuguru/space/modern_features_test")
set_tests_properties(modern_features_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/CMakeLists.txt;166;add_test;/Users/rishabnuguru/space/CMakeLists.txt;0;")
add_test(ieee754_tmr_test "/Users/rishabnuguru/space/ieee754_tmr_test")
set_tests_properties(ieee754_tmr_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/CMakeLists.txt;172;add_test;/Users/rishabnuguru/space/CMakeLists.txt;0;")
add_test(quantum_field_validation_test "/Users/rishabnuguru/space/quantum_field_validation_test")
set_tests_properties(quantum_field_validation_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/CMakeLists.txt;181;add_test;/Users/rishabnuguru/space/CMakeLists.txt;0;")
add_test(realistic_leo_mission_test "/Users/rishabnuguru/space/realistic_leo_mission_test")
set_tests_properties(realistic_leo_mission_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/rishabnuguru/space/CMakeLists.txt;190;add_test;/Users/rishabnuguru/space/CMakeLists.txt;0;")
subdirs("src/tmr")
subdirs("src/rad_ml/research")
subdirs("examples")
