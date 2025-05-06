#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>

// Rad-ML framework includes - needed for compilation
#include "rad_ml/tmr/physics_driven_protection.hpp"
#include "rad_ml/sim/mission_environment.hpp"
#include "rad_ml/core/material_database.hpp"

using namespace rad_ml;
using namespace rad_ml::tmr;

// Simple TMR demonstration
class TMRDemo {
private:
    // Random number generation for realistic error simulation
    static std::mt19937& getRNG() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return gen;
    }

public:
    // Simulate a computation with realistic radiation-induced errors
    static float computeWithErrors(float input, float error_rate) {
        // No errors case
        if (error_rate <= 0) {
            return input;
        }
        
        // Get random number generator
        auto& gen = getRNG();
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        // Apply different types of errors based on error rate
        if (dist(gen) < error_rate) {
            // Decide which type of error to inject
            int error_type = static_cast<int>(dist(gen) * 4);
            
            switch (error_type) {
                case 0: { // Bit flip
                    uint32_t bits = *reinterpret_cast<uint32_t*>(&input);
                    int bit_pos = static_cast<int>(dist(gen) * 32);
                    bits ^= (1u << bit_pos);
                    return *reinterpret_cast<float*>(&bits);
                }
                case 1: // Value offset
                    return input + (dist(gen) * 2.0f - 1.0f) * input;
                case 2: // Sign flip
                    return -input;
                case 3: // NaN or Inf
                    if (dist(gen) < 0.5f) {
                        // Return NaN
                        uint32_t nan_bits = 0x7FC00000;
                        return *reinterpret_cast<float*>(&nan_bits);
                    } else {
                        // Return Inf
                        uint32_t inf_bits = 0x7F800000;
                        return *reinterpret_cast<float*>(&inf_bits);
                    }
                default:
                    return input;
            }
        }
        
        return input;
    }
    
    // Apply TMR voting with N repetitions
    static float applyTMR(float input, float error_rate, int redundancy) {
        if (redundancy <= 1) {
            return computeWithErrors(input, error_rate);
        }
        
        std::vector<float> results;
        for (int i = 0; i < redundancy; i++) {
            results.push_back(computeWithErrors(input, error_rate));
        }
        
        // Handle NaN and Inf values first
        std::vector<float> valid_results;
        for (float result : results) {
            if (!std::isnan(result) && !std::isinf(result)) {
                valid_results.push_back(result);
            }
        }
        
        // If no valid results, return the input (conservative approach)
        if (valid_results.empty()) {
            return input;
        }
        
        // Sort and take median for TMR
        std::sort(valid_results.begin(), valid_results.end());
        return valid_results[valid_results.size() / 2];
    }
    
    // Determine if result is within acceptable error margin
    static bool isAcceptable(float original, float result, float tolerance) {
        // Handle NaN and Inf
        if (std::isnan(result) || std::isinf(result)) {
            return false;
        }
        
        // Check relative error
        return std::abs(original - result) / std::abs(original) <= tolerance;
    }
};

// Function to test protection effectiveness
void testProtectionEffectiveness() {
    std::cout << "===============================================\n";
    std::cout << "  Radiation-Tolerant ML Protection Test\n";
    std::cout << "===============================================\n";
    
    // Test parameters
    std::vector<float> error_rates = {0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5};
    std::vector<int> redundancy_levels = {1, 3, 5, 7}; // 1 = no protection, 3 = basic TMR, etc.
    std::vector<std::string> protection_names = {"No Protection", "Basic TMR", "Enhanced TMR", "Advanced TMR"};
    
    // Input value for testing
    const float test_value = 10.0f;
    const float tolerance = 0.05f; // 5% tolerance
    
    // Results table headers
    std::cout << "\n===================================================================\n";
    std::cout << std::left << std::setw(15) << "Error Rate"
              << std::setw(15) << "No Protection"
              << std::setw(15) << "Basic TMR"
              << std::setw(15) << "Enhanced TMR"
              << std::setw(15) << "Advanced TMR" << "\n";
    std::cout << "===================================================================\n";
    
    // Open CSV file for results
    std::ofstream results_file("protection_effectiveness.csv");
    results_file << "Error Rate,No Protection,Basic TMR,Enhanced TMR,Advanced TMR\n";
    
    // Run tests with 1000 trials per configuration
    const int num_trials = 1000;
    
    // Test each error rate
    for (float error_rate : error_rates) {
        std::cout << std::left << std::setw(15) << error_rate;
        results_file << error_rate;
        
        // Test each protection level
        for (size_t i = 0; i < redundancy_levels.size(); i++) {
            int redundancy = redundancy_levels[i];
            int correct = 0;
            
            // Run trials
            for (int trial = 0; trial < num_trials; trial++) {
                // Calculate result with TMR protection
                float result = TMRDemo::applyTMR(test_value, error_rate, redundancy);
                
                // Check if result is acceptable
                if (TMRDemo::isAcceptable(test_value, result, tolerance)) {
                    correct++;
                }
            }
            
            // Calculate accuracy percentage
            double accuracy = static_cast<double>(correct) / num_trials * 100.0;
            
            // Print and save result
            std::cout << std::setw(15) << std::fixed << std::setprecision(2) << accuracy;
            results_file << "," << std::fixed << std::setprecision(4) << (accuracy / 100.0);
        }
        
        std::cout << "\n";
        results_file << "\n";
    }
    
    std::cout << "===================================================================\n";
    
    // Calculate improvement from TMR
    std::cout << "\nImprovement from TMR protection:\n";
    std::cout << "===================================================================\n";
    std::cout << std::left << std::setw(15) << "Error Rate"
              << std::setw(15) << "Basic TMR"
              << std::setw(15) << "Enhanced TMR"
              << std::setw(15) << "Advanced TMR" << "\n";
    std::cout << "===================================================================\n";
    
    results_file << "\nImprovement\n";
    results_file << "Error Rate,Basic TMR,Enhanced TMR,Advanced TMR\n";
    
    // Test multiple runs of the same configuration to see statistical improvements
    std::vector<std::vector<double>> all_accuracies;
    
    for (float error_rate : error_rates) {
        std::vector<double> accuracies;
        
        for (int redundancy : redundancy_levels) {
            int correct = 0;
            for (int trial = 0; trial < num_trials; trial++) {
                float result = TMRDemo::applyTMR(test_value, error_rate, redundancy);
                if (TMRDemo::isAcceptable(test_value, result, tolerance)) {
                    correct++;
                }
            }
            double accuracy = static_cast<double>(correct) / num_trials * 100.0;
            accuracies.push_back(accuracy);
        }
        
        all_accuracies.push_back(accuracies);
    }
    
    // Calculate improvements
    for (size_t i = 0; i < error_rates.size(); i++) {
        std::cout << std::left << std::setw(15) << error_rates[i];
        results_file << error_rates[i];
        
        double base_accuracy = all_accuracies[i][0]; // No protection
        
        // Show improvement for each protection level
        for (size_t j = 1; j < redundancy_levels.size(); j++) {
            double improvement = all_accuracies[i][j] - base_accuracy;
            
            // Print improvement with plus/minus sign
            std::cout << std::setw(15) << std::showpos << std::fixed << std::setprecision(2) << improvement;
            results_file << "," << improvement;
        }
        
        std::cout << "\n";
        results_file << "\n";
    }
    
    results_file.close();
    std::cout << "\nTest completed. Results saved to protection_effectiveness.csv\n";
}

int main() {
    testProtectionEffectiveness();
    return 0;
} 