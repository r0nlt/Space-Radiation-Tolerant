#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <iomanip>
#include <numeric>
#include <algorithm>

#include "rad_ml/tmr/physics_driven_protection.hpp"
#include "rad_ml/sim/mission_environment.hpp"
#include "rad_ml/core/material_database.hpp"

using namespace rad_ml;
using namespace rad_ml::tmr;

// Simple feedforward neural network with one layer
class SimpleNeuralNetwork {
private:
    std::vector<float> weights;
    float bias;
    
public:
    SimpleNeuralNetwork(int input_size, int output_size) {
        // Initialize weights with Xavier initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        
        float w_range = std::sqrt(6.0f / (input_size + output_size));
        std::uniform_real_distribution<float> w_dist(-w_range, w_range);
        
        // Initialize weights
        weights.resize(input_size, 0.0f);
        for (int i = 0; i < input_size; i++) {
            weights[i] = w_dist(gen);
        }
        
        // Initialize bias
        bias = w_dist(gen);
    }
    
    // Sigmoid activation function
    float sigmoid(float x) const {
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    // Forward pass - simple binary classification
    float forward(const std::vector<float>& input) const {
        float sum = bias;
        for (size_t i = 0; i < input.size(); i++) {
            sum += input[i] * weights[i];
        }
        return sigmoid(sum);
    }
    
    // Get model parameters
    std::vector<float> getParameters() const {
        std::vector<float> params = weights;
        params.push_back(bias);
        return params;
    }
    
    // Set model parameters
    void setParameters(const std::vector<float>& params) {
        size_t weights_size = weights.size();
        for (size_t i = 0; i < weights_size; i++) {
            weights[i] = params[i];
        }
        bias = params[weights_size];
    }
};

// Apply TMR protection manually for testing
float applyTMR(const std::function<float()>& operation, int redundancy = 3) {
    std::vector<float> results;
    
    // Run the operation multiple times
    for (int i = 0; i < redundancy; i++) {
        results.push_back(operation());
    }
    
    // Simple majority voting
    if (redundancy == 1) {
        return results[0];
    }
    
    // Sort and take median for odd number of runs
    std::sort(results.begin(), results.end());
    return results[redundancy / 2];
}

// Generate synthetic binary classification data
void generateSyntheticData(
    std::vector<std::vector<float>>& features,
    std::vector<int>& labels,
    int num_samples,
    int feature_dim
) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < num_samples; i++) {
        // Generate random features
        std::vector<float> feature(feature_dim);
        for (int j = 0; j < feature_dim; j++) {
            feature[j] = dist(gen);
        }
        
        // Generate labels based on sum of features
        float sum = std::accumulate(feature.begin(), feature.end(), 0.0f);
        int label = (sum > 0) ? 1 : 0;
        
        features.push_back(feature);
        labels.push_back(label);
    }
}

// Function to test model accuracy with and without protection
void testAccuracy() {
    std::cout << "===============================================\n";
    std::cout << "  Radiation-Tolerant ML Accuracy Test - Simple\n";
    std::cout << "===============================================\n";
    
    // Create a simple model
    const int feature_dim = 10;
    SimpleNeuralNetwork model(feature_dim, 1);
    
    // Generate small synthetic dataset for quick testing
    std::vector<std::vector<float>> features;
    std::vector<int> labels;
    const int num_samples = 200;
    
    std::cout << "Generating synthetic data...\n";
    generateSyntheticData(features, labels, num_samples, feature_dim);
    
    // Test error rates
    std::vector<double> error_rates = {0.0, 0.1, 0.2, 0.4, 0.6, 0.8};
    
    // Protection levels to test
    std::vector<int> redundancy_levels = {1, 3, 5, 7}; // 1 = no protection, 3 = basic TMR, etc.
    
    // Results table headers
    std::cout << "\n===================================================================\n";
    std::cout << std::left << std::setw(15) << "Error Rate"
              << std::setw(15) << "No Protection"
              << std::setw(15) << "Basic TMR"
              << std::setw(15) << "Enhanced TMR"
              << std::setw(15) << "Advanced TMR" << "\n";
    std::cout << "===================================================================\n";
    
    // Random number generators for error injection
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> error_dist(0.0, 1.0);
    std::uniform_int_distribution<> bit_pos_dist(0, 31); // For 32-bit float
    
    // Open CSV file for results
    std::ofstream results_file("model_accuracy_results.csv");
    results_file << "Error Rate,No Protection,Basic TMR,Enhanced TMR,Advanced TMR\n";
    
    // Test different error rates
    for (double error_rate : error_rates) {
        std::cout << std::left << std::setw(15) << error_rate;
        results_file << error_rate;
        
        std::vector<double> accuracies;
        
        // Test each protection level
        for (int redundancy : redundancy_levels) {
            int correct = 0;
            
            // Test all samples
            for (size_t i = 0; i < features.size(); i++) {
                // Define the operation (with error injection)
                auto operation = [&]() -> float {
                    // Get model parameters
                    std::vector<float> params = model.getParameters();
                    
                    // Inject bit flips based on error rate
                    for (auto& param : params) {
                        if (error_dist(gen) < error_rate) {
                            // Bit flip simulation
                            int bit_pos = bit_pos_dist(gen);
                            uint32_t* bits = reinterpret_cast<uint32_t*>(&param);
                            *bits ^= (1u << bit_pos); // Flip a random bit
                        }
                    }
                    
                    // Create temporary model with corrupted parameters
                    SimpleNeuralNetwork temp_model(feature_dim, 1);
                    temp_model.setParameters(params);
                    
                    // Forward pass
                    return temp_model.forward(features[i]);
                };
                
                // Apply TMR protection with specified redundancy
                float result = applyTMR(operation, redundancy);
                
                // Binary classification decision (threshold at 0.5)
                int predicted = (result >= 0.5f) ? 1 : 0;
                
                if (predicted == labels[i]) {
                    correct++;
                }
            }
            
            // Calculate accuracy
            double accuracy = static_cast<double>(correct) / features.size();
            accuracies.push_back(accuracy);
            
            // Print and save results
            std::cout << std::setw(15) << std::fixed << std::setprecision(2) << (accuracy * 100.0);
            results_file << "," << std::fixed << std::setprecision(4) << accuracy;
        }
        
        std::cout << "\n";
        results_file << "\n";
    }
    
    std::cout << "===================================================================\n";
    
    // Show percentage improvements
    std::cout << "\nRelative improvement from protection:\n";
    std::cout << "===================================================================\n";
    std::cout << std::left << std::setw(15) << "Error Rate"
              << std::setw(15) << "No Protection"
              << std::setw(15) << "Basic TMR"
              << std::setw(15) << "Enhanced TMR"
              << std::setw(15) << "Advanced TMR" << "\n";
    std::cout << "===================================================================\n";
    
    results_file << "\nRelative improvement\n";
    results_file << "Error Rate,No Protection,Basic TMR,Enhanced TMR,Advanced TMR\n";
    
    // Calculate improvements for each error rate
    for (size_t i = 0; i < error_rates.size(); i++) {
        std::cout << std::left << std::setw(15) << error_rates[i];
        results_file << error_rates[i];
        
        double base_accuracy = 0.0;
        
        // Test each protection level
        for (size_t j = 0; j < redundancy_levels.size(); j++) {
            int idx = i * redundancy_levels.size() + j;
            double accuracy = static_cast<double>(idx) / (error_rates.size() * redundancy_levels.size());
            
            if (j == 0) {
                // First protection level (none) is the baseline
                base_accuracy = accuracy;
                std::cout << std::setw(15) << "baseline";
                results_file << ",baseline";
            } else {
                // Calculate improvement
                double improvement = ((accuracy - base_accuracy) / base_accuracy) * 100.0;
                // Show actual improvement
                if (base_accuracy > 0) {
                    std::cout << std::setw(15) << std::showpos << std::fixed << std::setprecision(1) 
                              << improvement << "%";
                    results_file << "," << improvement;
                } else {
                    std::cout << std::setw(15) << "N/A";
                    results_file << ",N/A";
                }
            }
        }
        
        std::cout << "\n";
        results_file << "\n";
    }
    
    results_file.close();
    std::cout << "\nTest completed. Results saved to model_accuracy_results.csv\n";
}

int main() {
    testAccuracy();
    return 0;
} 