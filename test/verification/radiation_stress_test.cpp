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
/**
 * @file radiation_stress_test.cpp
 * @brief Stress test for the radiation-tolerant ML framework under extreme radiation conditions
 * 
 * This test simulates extreme radiation environments to evaluate the framework's
 * robustness and error correction capabilities under severe conditions.
 */

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <bitset>
#include <cmath>

#include "../../include/rad_ml/tmr/adaptive_protection.hpp"
#include "../../include/rad_ml/sim/mission_environment.hpp"

// Utility function to flip bits in a floating point value
// This simulates bit flips caused by radiation
template<typename T>
T simulateBitFlip(T value, int numBitFlips, bool adjacentBits = false) {
    static_assert(std::is_floating_point<T>::value, "Only floating point types are supported");
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, sizeof(T) * 8 - 1);
    
    // Get raw bit representation
    std::bitset<sizeof(T) * 8> bits = 
        *reinterpret_cast<std::bitset<sizeof(T) * 8>*>(&value);
    
    if (adjacentBits && numBitFlips > 1) {
        // Flip adjacent bits (common in some radiation effects)
        int startBit = dis(gen);
        for (int i = 0; i < numBitFlips; i++) {
            int bitToFlip = (startBit + i) % (sizeof(T) * 8);
            bits.flip(bitToFlip);
        }
    } else {
        // Flip random bits
        for (int i = 0; i < numBitFlips; i++) {
            bits.flip(dis(gen));
        }
    }
    
    // Convert back to floating point
    value = *reinterpret_cast<T*>(&bits);
    return value;
}

// Simple neural network class with TMR protection 
class RadiationTestNetwork {
private:
    // Network parameters
    std::vector<std::vector<float>> weights1;
    std::vector<float> biases1;
    std::vector<std::vector<float>> weights2;
    std::vector<float> biases2;
    
    // Triple redundant copies for radiation protection
    std::vector<std::vector<float>> weights1_copy2;
    std::vector<std::vector<float>> weights1_copy3;
    std::vector<float> biases1_copy2;
    std::vector<float> biases1_copy3;
    std::vector<std::vector<float>> weights2_copy2;
    std::vector<std::vector<float>> weights2_copy3;
    std::vector<float> biases2_copy2;
    std::vector<float> biases2_copy3;
    
    // Protection status
    bool protectionEnabled;
    
    // Error tracking
    int totalErrors;
    int correctedErrors;
    
    // ReLU activation function
    float relu(float x) {
        return x > 0 ? x : 0;
    }
    
public:
    RadiationTestNetwork(int inputSize, int hiddenSize, int outputSize) : 
        protectionEnabled(true), totalErrors(0), correctedErrors(0) {
        
        // Initialize with a random seed for reproducibility
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-1.0, 1.0);
        
        // Initialize weights and biases
        weights1.resize(inputSize, std::vector<float>(hiddenSize));
        biases1.resize(hiddenSize);
        weights2.resize(hiddenSize, std::vector<float>(outputSize));
        biases2.resize(outputSize);
        
        // Initialize with random values
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weights1[i][j] = dis(gen);
            }
        }
        
        for (int i = 0; i < hiddenSize; i++) {
            biases1[i] = dis(gen);
            for (int j = 0; j < outputSize; j++) {
                weights2[i][j] = dis(gen);
            }
        }
        
        for (int i = 0; i < outputSize; i++) {
            biases2[i] = dis(gen);
        }
        
        // Create redundant copies
        weights1_copy2 = weights1;
        weights1_copy3 = weights1;
        biases1_copy2 = biases1;
        biases1_copy3 = biases1;
        weights2_copy2 = weights2;
        weights2_copy3 = weights2;
        biases2_copy2 = biases2;
        biases2_copy3 = biases2;
    }
    
    // Forward pass with radiation protection
    std::vector<float> forward(const std::vector<float>& input, float radiationIntensity) {
        // Apply radiation effects
        applyRadiationEffects(radiationIntensity);
        
        // First layer
        std::vector<float> hidden(biases1.size());
        for (size_t i = 0; i < hidden.size(); i++) {
            // Voter for bias
            float bias = voteMajority(biases1[i], biases1_copy2[i], biases1_copy3[i]);
            
            hidden[i] = bias;
            for (size_t j = 0; j < input.size(); j++) {
                // Voter for weight
                float weight = voteMajority(
                    weights1[j][i], weights1_copy2[j][i], weights1_copy3[j][i]);
                
                hidden[i] += input[j] * weight;
            }
            hidden[i] = relu(hidden[i]);
        }
        
        // Second layer
        std::vector<float> output(biases2.size());
        for (size_t i = 0; i < output.size(); i++) {
            // Voter for bias
            float bias = voteMajority(biases2[i], biases2_copy2[i], biases2_copy3[i]);
            
            output[i] = bias;
            for (size_t j = 0; j < hidden.size(); j++) {
                // Voter for weight
                float weight = voteMajority(
                    weights2[j][i], weights2_copy2[j][i], weights2_copy3[j][i]);
                
                output[i] += hidden[j] * weight;
            }
            // No activation for output layer (for regression)
        }
        
        return output;
    }
    
    // Apply radiation effects to the model parameters
    void applyRadiationEffects(float radiationIntensity) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 1.0);
        std::uniform_int_distribution<int> bitDis(1, 3);
        
        // Probability of a bit flip is proportional to radiation intensity
        float flipProb = radiationIntensity * 0.01; // Scale factor to get reasonable probabilities
        
        // Apply to weights1
        for (size_t i = 0; i < weights1.size(); i++) {
            for (size_t j = 0; j < weights1[i].size(); j++) {
                if (dis(gen) < flipProb) {
                    totalErrors++;
                    int bitFlips = bitDis(gen); // Number of bits to flip (1-3)
                    weights1[i][j] = simulateBitFlip(weights1[i][j], bitFlips);
                }
                if (dis(gen) < flipProb) {
                    totalErrors++;
                    int bitFlips = bitDis(gen);
                    weights1_copy2[i][j] = simulateBitFlip(weights1_copy2[i][j], bitFlips, true);
                }
                if (dis(gen) < flipProb * 0.5) { // Less likely to damage all three copies
                    totalErrors++;
                    int bitFlips = bitDis(gen);
                    weights1_copy3[i][j] = simulateBitFlip(weights1_copy3[i][j], bitFlips);
                }
            }
        }
        
        // Apply to biases1
        for (size_t i = 0; i < biases1.size(); i++) {
            if (dis(gen) < flipProb) {
                totalErrors++;
                int bitFlips = bitDis(gen);
                biases1[i] = simulateBitFlip(biases1[i], bitFlips);
            }
            if (dis(gen) < flipProb) {
                totalErrors++;
                int bitFlips = bitDis(gen);
                biases1_copy2[i] = simulateBitFlip(biases1_copy2[i], bitFlips, true);
            }
            if (dis(gen) < flipProb * 0.5) {
                totalErrors++;
                int bitFlips = bitDis(gen);
                biases1_copy3[i] = simulateBitFlip(biases1_copy3[i], bitFlips);
            }
        }
        
        // Apply to weights2
        for (size_t i = 0; i < weights2.size(); i++) {
            for (size_t j = 0; j < weights2[i].size(); j++) {
                if (dis(gen) < flipProb) {
                    totalErrors++;
                    int bitFlips = bitDis(gen);
                    weights2[i][j] = simulateBitFlip(weights2[i][j], bitFlips);
                }
                if (dis(gen) < flipProb) {
                    totalErrors++;
                    int bitFlips = bitDis(gen);
                    weights2_copy2[i][j] = simulateBitFlip(weights2_copy2[i][j], bitFlips, true);
                }
                if (dis(gen) < flipProb * 0.5) {
                    totalErrors++;
                    int bitFlips = bitDis(gen);
                    weights2_copy3[i][j] = simulateBitFlip(weights2_copy3[i][j], bitFlips);
                }
            }
        }
        
        // Apply to biases2
        for (size_t i = 0; i < biases2.size(); i++) {
            if (dis(gen) < flipProb) {
                totalErrors++;
                int bitFlips = bitDis(gen);
                biases2[i] = simulateBitFlip(biases2[i], bitFlips);
            }
            if (dis(gen) < flipProb) {
                totalErrors++;
                int bitFlips = bitDis(gen);
                biases2_copy2[i] = simulateBitFlip(biases2_copy2[i], bitFlips, true);
            }
            if (dis(gen) < flipProb * 0.5) {
                totalErrors++;
                int bitFlips = bitDis(gen);
                biases2_copy3[i] = simulateBitFlip(biases2_copy3[i], bitFlips);
            }
        }
    }
    
    // Majority voter for TMR
    float voteMajority(float a, float b, float c) {
        if (!protectionEnabled) {
            return a; // Return first copy without voting if protection is disabled
        }
        
        // Count number of exact matches
        int matches = 0;
        if (a == b) matches++;
        if (a == c) matches++;
        if (b == c) matches++;
        
        // Fast path for exact matches
        if (matches > 0) {
            if (a == b) {
                correctedErrors++;
                return a;
            } else if (a == c) {
                correctedErrors++;
                return a;
            } else { // b == c
                correctedErrors++;
                return b;
            }
        }
        
        // If no exact matches, use closest match approach
        // Convert to bits for bitwise voting
        uint32_t a_bits = *reinterpret_cast<uint32_t*>(&a);
        uint32_t b_bits = *reinterpret_cast<uint32_t*>(&b);
        uint32_t c_bits = *reinterpret_cast<uint32_t*>(&c);
        
        // Bitwise voting for each bit
        uint32_t result_bits = 0;
        for (int i = 0; i < 32; i++) {
            uint32_t bit_a = (a_bits >> i) & 1;
            uint32_t bit_b = (b_bits >> i) & 1;
            uint32_t bit_c = (c_bits >> i) & 1;
            
            // Majority vote for this bit
            uint32_t majority_bit = (bit_a & bit_b) | (bit_a & bit_c) | (bit_b & bit_c);
            result_bits |= (majority_bit << i);
            
            if (bit_a != majority_bit || bit_b != majority_bit || bit_c != majority_bit) {
                correctedErrors++;
            }
        }
        
        return *reinterpret_cast<float*>(&result_bits);
    }
    
    // Reset error counters
    void resetErrorCounters() {
        totalErrors = 0;
        correctedErrors = 0;
    }
    
    // Get error statistics
    std::pair<int, int> getErrorStats() {
        return {totalErrors, correctedErrors};
    }
    
    // Toggle protection
    void setProtection(bool enabled) {
        protectionEnabled = enabled;
    }
    
    // Check if protection is enabled
    bool isProtectionEnabled() {
        return protectionEnabled;
    }
};

// Function to run radiation tests with a network
std::vector<std::vector<float>> runRadiationTest(
    RadiationTestNetwork& network, 
    float radiationIntensity, 
    int numSamples, 
    bool protectionEnabled,
    std::string& csvOutput) {
    
    // Configure network protection
    network.setProtection(protectionEnabled);
    network.resetErrorCounters();
    
    // Statistics
    std::vector<float> mseValues;
    std::vector<std::vector<float>> detailedResults;
    
    // Create output file
    std::ofstream outFile(csvOutput);
    outFile << "sample_id,radiation_intensity,protection,input_size,output_size,mse,raw_error,corrected_error\n";
    
    // Create test inputs
    std::vector<std::vector<float>> testInputs;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    
    // Generate test inputs
    for (int i = 0; i < numSamples; i++) {
        std::vector<float> input(8); // 8-dimensional input
        for (auto& val : input) {
            val = dis(gen);
        }
        testInputs.push_back(input);
    }
    
    // Get reference outputs without radiation
    network.setProtection(true);
    std::vector<std::vector<float>> referenceOutputs;
    for (const auto& input : testInputs) {
        referenceOutputs.push_back(network.forward(input, 0.0)); // No radiation
    }
    
    // Reset error counters
    network.resetErrorCounters();
    
    // Configure protection for the test
    network.setProtection(protectionEnabled);
    
    // Run tests with radiation
    for (int i = 0; i < numSamples; i++) {
        // Forward pass with radiation
        auto output = network.forward(testInputs[i], radiationIntensity);
        auto referenceOutput = referenceOutputs[i];
        
        // Calculate MSE
        float mse = 0.0f;
        for (size_t j = 0; j < output.size(); j++) {
            float diff = output[j] - referenceOutput[j];
            mse += diff * diff;
        }
        mse /= output.size();
        mseValues.push_back(mse);
        
        // Get error statistics
        auto [totalErrors, correctedErrors] = network.getErrorStats();
        
        // Record detailed results
        std::vector<float> result = {
            static_cast<float>(i),               // Sample ID
            radiationIntensity,                  // Radiation intensity
            static_cast<float>(protectionEnabled), // Protection status
            static_cast<float>(testInputs[i].size()), // Input size
            static_cast<float>(output.size()),   // Output size
            mse,                                // MSE
            static_cast<float>(totalErrors),     // Total errors
            static_cast<float>(correctedErrors)  // Corrected errors
        };
        detailedResults.push_back(result);
        
        // Write to CSV
        outFile << i << ","
                << radiationIntensity << ","
                << (protectionEnabled ? "enabled" : "disabled") << ","
                << testInputs[i].size() << ","
                << output.size() << ","
                << mse << ","
                << totalErrors << ","
                << correctedErrors << "\n";
    }
    
    outFile.close();
    
    return detailedResults;
}

// Function to create a Python visualization script
void createVisualizationScript() {
    std::ofstream pyFile("radiation_stress_visualization.py");
    
    pyFile << R"(
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np

# Find all CSV files with radiation test results
csv_files = glob.glob("radiation_test_*.csv")

# Process each file
for csv_file in csv_files:
    try:
        # Extract test parameters from filename
        filename = os.path.basename(csv_file)
        params = filename.replace("radiation_test_", "").replace(".csv", "").split("_")
        radiation_intensity = float(params[0])
        protection_status = params[1]
        
        # Read data
        df = pd.read_csv(csv_file)
        
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Radiation Test Results - Intensity: {radiation_intensity}, Protection: {protection_status}", 
                     fontsize=16)
        
        # Plot 1: MSE over samples
        axs[0, 0].plot(df['sample_id'], df['mse'], 'b-')
        axs[0, 0].set_title('MSE over Samples')
        axs[0, 0].set_xlabel('Sample ID')
        axs[0, 0].set_ylabel('Mean Squared Error')
        axs[0, 0].grid(True)
        
        # Plot 2: Errors over samples
        axs[0, 1].plot(df['sample_id'], df['raw_error'], 'r-', label='Raw Errors')
        if protection_status == "enabled":
            axs[0, 1].plot(df['sample_id'], df['corrected_error'], 'g-', label='Corrected Errors')
        axs[0, 1].set_title('Errors over Samples')
        axs[0, 1].set_xlabel('Sample ID')
        axs[0, 1].set_ylabel('Number of Errors')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # Plot 3: Histogram of MSE
        axs[1, 0].hist(df['mse'], bins=20, alpha=0.7, color='blue')
        axs[1, 0].set_title('MSE Distribution')
        axs[1, 0].set_xlabel('MSE')
        axs[1, 0].set_ylabel('Frequency')
        axs[1, 0].grid(True)
        
        # Plot 4: Error correction effectiveness (if protection enabled)
        if protection_status == "enabled":
            correction_rate = df['corrected_error'].sum() / max(df['raw_error'].sum(), 1) * 100
            labels = ['Corrected', 'Uncorrected']
            sizes = [correction_rate, 100 - correction_rate]
            axs[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', 
                          colors=['green', 'red'], startangle=90)
            axs[1, 1].set_title('Error Correction Rate')
        else:
            axs[1, 1].text(0.5, 0.5, 'Protection Disabled\nNo Error Correction',
                          horizontalalignment='center', verticalalignment='center',
                          transform=axs[1, 1].transAxes, fontsize=14)
            axs[1, 1].set_title('Error Correction Rate')
            axs[1, 1].axis('off')
        
        # Add statistics as text
        stats_text = f"Statistics:\n"
        stats_text += f"Total Samples: {len(df)}\n"
        stats_text += f"Avg MSE: {df['mse'].mean():.6f}\n"
        stats_text += f"Total Raw Errors: {df['raw_error'].sum()}\n"
        if protection_status == "enabled":
            stats_text += f"Total Corrected Errors: {df['corrected_error'].sum()}\n"
            stats_text += f"Correction Rate: {correction_rate:.2f}%"
        
        fig.text(0.5, 0.01, stats_text, horizontalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(f"radiation_test_{radiation_intensity}_{protection_status}.png", dpi=300)
        plt.close()
        
        print(f"Created visualization for {csv_file}")
        
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

print("Visualization complete. Check the generated PNG files.")
)";
    
    pyFile.close();
}

int main() {
    std::cout << "Starting radiation stress test..." << std::endl;
    
    // Test parameters
    std::vector<float> radiationIntensities = {10.0, 50.0, 100.0}; // Very high radiation levels
    int numSamples = 100;
    
    // Create a network
    std::cout << "Creating neural network for radiation testing..." << std::endl;
    RadiationTestNetwork network(8, 16, 4); // 8 inputs, 16 hidden, 4 outputs
    
    // Create Python visualization script
    createVisualizationScript();
    
    // Run tests with protection enabled
    std::cout << "Running tests with protection enabled..." << std::endl;
    for (auto intensity : radiationIntensities) {
        std::string csvFilename = "radiation_test_" + std::to_string(static_cast<int>(intensity)) + "_enabled.csv";
        std::cout << "Testing radiation intensity: " << intensity << "..." << std::endl;
        runRadiationTest(network, intensity, numSamples, true, csvFilename);
    }
    
    // Run tests with protection disabled
    std::cout << "Running tests with protection disabled..." << std::endl;
    for (auto intensity : radiationIntensities) {
        std::string csvFilename = "radiation_test_" + std::to_string(static_cast<int>(intensity)) + "_disabled.csv";
        std::cout << "Testing radiation intensity: " << intensity << "..." << std::endl;
        runRadiationTest(network, intensity, numSamples, false, csvFilename);
    }
    
    std::cout << "Radiation stress test completed." << std::endl;
    std::cout << "Results saved to CSV files. Run 'python radiation_stress_visualization.py' to generate visualizations." << std::endl;
    
    return 0;
} 
