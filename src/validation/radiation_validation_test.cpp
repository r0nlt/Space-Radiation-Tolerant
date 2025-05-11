/**
 * Copyright (C) 2025 Rishab Nuguru
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */
#include "validation/radiation_accuracy_validator.hpp"
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

/**
 * @brief Command-line test program for NASA/ESA radiation validation protocol
 * 
 * This program runs a comprehensive test of the radiation-tolerant ML framework
 * using the NASA/ESA standard validation protocol. It generates reports comparing
 * the framework's predictions against industry-standard tools and flight data.
 */
int main(int argc, char* argv[]) {
    // Parse command line arguments
    bool run_all_environments = true;
    int duration_days = 30;
    std::string output_path = "radiation_validation_results.csv";
    std::string report_path = "radiation_validation_report.txt";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Radiation Tolerance Framework NASA/ESA Validation Tool\n\n"
                      << "Usage: " << argv[0] << " [options]\n\n"
                      << "Options:\n"
                      << "  --days N           Set simulation duration to N days (default: 30)\n"
                      << "  --csv PATH         Set CSV output path (default: radiation_validation_results.csv)\n"
                      << "  --report PATH      Set report output path (default: radiation_validation_report.txt)\n"
                      << "  --environment ENV  Test only specified environment (default: all)\n"
                      << "                     Valid environments: LEO, VAN_ALLEN, LUNAR, INTERPLANETARY, JUPITER\n"
                      << "  --help, -h         Show this help message\n";
            return 0;
        } else if (arg == "--days" && i + 1 < argc) {
            duration_days = std::stoi(argv[++i]);
        } else if (arg == "--csv" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--report" && i + 1 < argc) {
            report_path = argv[++i];
        } else if (arg == "--environment" && i + 1 < argc) {
            run_all_environments = false;
            // Environment will be parsed below
        }
    }
    
    // Print header
    std::cout << "┌──────────────────────────────────────────────────────────┐\n";
    std::cout << "│      NASA/ESA RADIATION TOLERANCE VALIDATION PROTOCOL    │\n";
    std::cout << "└──────────────────────────────────────────────────────────┘\n\n";
    
    // Show configuration
    std::cout << "Configuration:\n"
              << "  Duration:      " << duration_days << " days\n"
              << "  CSV Output:    " << output_path << "\n"
              << "  Report Output: " << report_path << "\n"
              << "  Environments:  " << (run_all_environments ? "All" : "Specific") << "\n\n";
    
    // Initialize validator
    std::cout << "Initializing validation framework..." << std::endl;
    rad_ml::validation::RadiationAccuracyValidator validator;
    
    if (!validator.initialize()) {
        std::cerr << "Error: Failed to initialize radiation validation framework." << std::endl;
        return 1;
    }
    
    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run validation
    std::cout << "Running validation tests (this may take some time)..." << std::endl;
    
    std::map<rad_ml::validation::StandardEnvironment, 
             std::map<std::string, rad_ml::validation::ValidationResult>> all_results;
    
    if (run_all_environments) {
        // Run all environments
        all_results = validator.validateAllEnvironments(duration_days);
    } else {
        // Run only specified environment
        rad_ml::validation::StandardEnvironment env = 
            rad_ml::validation::StandardEnvironment::LOW_EARTH_ORBIT;
        
        // Find environment argument
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--environment" && i + 1 < argc) {
                std::string env_name = argv[++i];
                
                if (env_name == "LEO") {
                    env = rad_ml::validation::StandardEnvironment::LOW_EARTH_ORBIT;
                } else if (env_name == "VAN_ALLEN") {
                    env = rad_ml::validation::StandardEnvironment::VAN_ALLEN_BELT;
                } else if (env_name == "LUNAR") {
                    env = rad_ml::validation::StandardEnvironment::LUNAR_ORBIT;
                } else if (env_name == "INTERPLANETARY") {
                    env = rad_ml::validation::StandardEnvironment::INTERPLANETARY;
                } else if (env_name == "JUPITER") {
                    env = rad_ml::validation::StandardEnvironment::JUPITER_EUROPA;
                } else {
                    std::cerr << "Warning: Unknown environment '" << env_name 
                              << "', using LEO instead." << std::endl;
                }
                
                break;
            }
        }
        
        // Run selected environment
        all_results[env] = validator.validateEnvironment(env, duration_days);
    }
    
    // Record end time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time).count();
    
    // Export results
    std::cout << "Exporting results to CSV: " << output_path << std::endl;
    if (!validator.exportResultsToCSV(all_results, output_path)) {
        std::cerr << "Error: Failed to write CSV output file." << std::endl;
    }
    
    // Generate report
    std::cout << "Generating validation report: " << report_path << std::endl;
    std::string report = validator.generateReport(all_results);
    
    // Write report to file
    std::ofstream report_file(report_path);
    if (report_file.is_open()) {
        report_file << report;
        report_file.close();
    } else {
        std::cerr << "Error: Failed to write report file." << std::endl;
    }
    
    // Calculate statistics
    int total_tests = 0;
    int passing_tests = 0;
    double total_accuracy = 0.0;
    double total_protection_efficiency = 0.0;
    
    for (const auto& env_pair : all_results) {
        for (const auto& prot_pair : env_pair.second) {
            const auto& result = prot_pair.second;
            
            total_tests++;
            if (result.within_tolerance) {
                passing_tests++;
            }
            
            total_accuracy += result.accuracy_percentage;
            total_protection_efficiency += result.protection_efficiency;
        }
    }
    
    // Print summary
    std::cout << "\n┌──────────────────────────────────────────────────────────┐\n";
    std::cout << "│                     VALIDATION SUMMARY                    │\n";
    std::cout << "└──────────────────────────────────────────────────────────┘\n\n";
    
    std::cout << "Total tests:           " << total_tests << "\n";
    std::cout << "Passing tests:         " << passing_tests << " (" 
              << std::fixed << std::setprecision(1) 
              << (total_tests > 0 ? (passing_tests * 100.0 / total_tests) : 0.0) << "%)\n";
    
    if (total_tests > 0) {
        std::cout << "Average accuracy:      " 
                 << std::fixed << std::setprecision(2)
                 << (total_accuracy / total_tests) << "%\n";
        std::cout << "Protection efficiency: " 
                 << std::fixed << std::setprecision(2)
                 << (total_protection_efficiency / total_tests) << "%\n";
    }
    
    std::cout << "\nTotal runtime: " << duration << " seconds\n";
    std::cout << "\nFull report saved to: " << report_path << "\n";
    std::cout << "Raw data saved to:    " << output_path << "\n\n";
    
    return 0;
} 
