/**
 * @file systematic_fault_test.cpp
 * @brief Systematic fault injection testing for the radiation-tolerant ML framework
 * 
 * This test performs systematic fault injection using different error patterns
 * to evaluate the framework's error detection and correction capabilities.
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
#include <memory>

#include "../../include/rad_ml/tmr/adaptive_protection.hpp"
#include "../../include/rad_ml/sim/mission_environment.hpp"
#include "../../include/rad_ml/testing/fault_injection.hpp"

using namespace rad_ml::testing;

// Simple neural network class with TMR protection - similar to RadiationTestNetwork
// but enhanced with systematic fault injection capabilities
class FaultInjectionTestNetwork {
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
    
    // Network dimensions
    int inputSize;
    int hiddenSize;
    int outputSize;
    
    // ReLU activation function
    float relu(float x) {
        return x > 0 ? x : 0;
    }
    
public:
    FaultInjectionTestNetwork(int inputSize, int hiddenSize, int outputSize) : 
        protectionEnabled(true), totalErrors(0), correctedErrors(0),
        inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize) {
        
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
    
    // Forward pass with protection
    std::vector<float> forward(const std::vector<float>& input) {
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
    
    // Apply systematic fault injection
    int applyFaultInjection(SystematicFaultInjector& injector, 
                           SystematicFaultInjector::FaultPattern pattern,
                           int coverage_percentage) {
        resetErrorCounters();
        int total_injected = 0;
        
        // Calculate number of parameters to affect
        int total_weights = weights1.size() * weights1[0].size() + 
                           weights2.size() * weights2[0].size();
        int total_biases = biases1.size() + biases2.size();
        int total_params = total_weights + total_biases;
        
        int num_to_affect = (total_params * coverage_percentage) / 100;
        
        // Track which parameters have been affected
        std::vector<bool> affected(total_params, false);
        
        // Random number generator for parameter selection
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> param_dis(0, total_params - 1);
        
        // Apply faults to random parameters
        for (int i = 0; i < num_to_affect; i++) {
            // Select a random parameter that hasn't been affected yet
            int param_idx;
            do {
                param_idx = param_dis(gen);
            } while (affected[param_idx]);
            
            affected[param_idx] = true;
            
            // Determine which parameter to affect
            if (param_idx < weights1.size() * weights1[0].size()) {
                // weights1
                int row = param_idx / weights1[0].size();
                int col = param_idx % weights1[0].size();
                
                // Determine which copy to affect
                std::uniform_int_distribution<> copy_dis(0, 2);
                int copy = copy_dis(gen);
                
                if (copy == 0) {
                    weights1[row][col] = injector.injectFault(weights1[row][col], pattern);
                } else if (copy == 1) {
                    weights1_copy2[row][col] = injector.injectFault(weights1_copy2[row][col], pattern);
                } else {
                    weights1_copy3[row][col] = injector.injectFault(weights1_copy3[row][col], pattern);
                }
                
                totalErrors++;
                total_injected++;
            } else if (param_idx < weights1.size() * weights1[0].size() + biases1.size()) {
                // biases1
                int idx = param_idx - weights1.size() * weights1[0].size();
                
                // Determine which copy to affect
                std::uniform_int_distribution<> copy_dis(0, 2);
                int copy = copy_dis(gen);
                
                if (copy == 0) {
                    biases1[idx] = injector.injectFault(biases1[idx], pattern);
                } else if (copy == 1) {
                    biases1_copy2[idx] = injector.injectFault(biases1_copy2[idx], pattern);
                } else {
                    biases1_copy3[idx] = injector.injectFault(biases1_copy3[idx], pattern);
                }
                
                totalErrors++;
                total_injected++;
            } else if (param_idx < weights1.size() * weights1[0].size() + biases1.size() + 
                       weights2.size() * weights2[0].size()) {
                // weights2
                int offset = weights1.size() * weights1[0].size() + biases1.size();
                int idx = param_idx - offset;
                int row = idx / weights2[0].size();
                int col = idx % weights2[0].size();
                
                // Determine which copy to affect
                std::uniform_int_distribution<> copy_dis(0, 2);
                int copy = copy_dis(gen);
                
                if (copy == 0) {
                    weights2[row][col] = injector.injectFault(weights2[row][col], pattern);
                } else if (copy == 1) {
                    weights2_copy2[row][col] = injector.injectFault(weights2_copy2[row][col], pattern);
                } else {
                    weights2_copy3[row][col] = injector.injectFault(weights2_copy3[row][col], pattern);
                }
                
                totalErrors++;
                total_injected++;
            } else {
                // biases2
                int offset = weights1.size() * weights1[0].size() + biases1.size() + 
                             weights2.size() * weights2[0].size();
                int idx = param_idx - offset;
                
                // Determine which copy to affect
                std::uniform_int_distribution<> copy_dis(0, 2);
                int copy = copy_dis(gen);
                
                if (copy == 0) {
                    biases2[idx] = injector.injectFault(biases2[idx], pattern);
                } else if (copy == 1) {
                    biases2_copy2[idx] = injector.injectFault(biases2_copy2[idx], pattern);
                } else {
                    biases2_copy3[idx] = injector.injectFault(biases2_copy3[idx], pattern);
                }
                
                totalErrors++;
                total_injected++;
            }
        }
        
        return total_injected;
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
    
    // Get network dimensions
    std::tuple<int, int, int> getDimensions() {
        return {inputSize, hiddenSize, outputSize};
    }
    
    // Get count of affected parameters
    std::pair<int, int> getAffectedCounts() {
        // Count weights and biases that differ between copies
        int weights_affected = 0;
        int biases_affected = 0;
        
        // Check weights1
        for (size_t i = 0; i < weights1.size(); i++) {
            for (size_t j = 0; j < weights1[i].size(); j++) {
                if (weights1[i][j] != weights1_copy2[i][j] || 
                    weights1[i][j] != weights1_copy3[i][j] ||
                    weights1_copy2[i][j] != weights1_copy3[i][j]) {
                    weights_affected++;
                }
            }
        }
        
        // Check weights2
        for (size_t i = 0; i < weights2.size(); i++) {
            for (size_t j = 0; j < weights2[i].size(); j++) {
                if (weights2[i][j] != weights2_copy2[i][j] || 
                    weights2[i][j] != weights2_copy3[i][j] ||
                    weights2_copy2[i][j] != weights2_copy3[i][j]) {
                    weights_affected++;
                }
            }
        }
        
        // Check biases1
        for (size_t i = 0; i < biases1.size(); i++) {
            if (biases1[i] != biases1_copy2[i] || 
                biases1[i] != biases1_copy3[i] ||
                biases1_copy2[i] != biases1_copy3[i]) {
                biases_affected++;
            }
        }
        
        // Check biases2
        for (size_t i = 0; i < biases2.size(); i++) {
            if (biases2[i] != biases2_copy2[i] || 
                biases2[i] != biases2_copy3[i] ||
                biases2_copy2[i] != biases2_copy3[i]) {
                biases_affected++;
            }
        }
        
        return {weights_affected, biases_affected};
    }
};

// Function to run systematic fault injection test
FaultInjectionResult runSystematicFaultTest(
    FaultInjectionTestNetwork& network,
    SystematicFaultInjector& injector,
    SystematicFaultInjector::FaultPattern pattern,
    int coverage_percentage,
    bool protectionEnabled)
{
    // Configure network protection
    network.setProtection(protectionEnabled);
    network.resetErrorCounters();
    
    // Get network dimensions
    auto [inputSize, hiddenSize, outputSize] = network.getDimensions();
    
    // Create result object
    FaultInjectionResult result;
    result.pattern_name = SystematicFaultInjector::patternToString(pattern);
    result.coverage_percentage = coverage_percentage;
    result.protection_enabled = protectionEnabled;
    result.input_size = inputSize;
    result.hidden_size = hiddenSize;
    result.output_size = outputSize;
    
    // Create test inputs
    std::vector<std::vector<float>> testInputs;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    
    const int numSamples = 50;
    
    // Generate test inputs
    for (int i = 0; i < numSamples; i++) {
        std::vector<float> input(inputSize);
        for (auto& val : input) {
            val = dis(gen);
        }
        testInputs.push_back(input);
    }
    
    // Get reference outputs before fault injection
    network.setProtection(true); // Ensure protection for reference
    std::vector<std::vector<float>> referenceOutputs;
    for (const auto& input : testInputs) {
        referenceOutputs.push_back(network.forward(input));
    }
    
    // Inject faults according to pattern and coverage
    result.total_injected_faults = network.applyFaultInjection(injector, pattern, coverage_percentage);
    
    // Get counts of affected parameters
    auto [weights_affected, biases_affected] = network.getAffectedCounts();
    result.total_weights_affected = weights_affected;
    result.total_biases_affected = biases_affected;
    
    // Configure protection for the test
    network.setProtection(protectionEnabled);
    
    // Run tests with injected faults
    double total_mse = 0.0;
    for (int i = 0; i < numSamples; i++) {
        // Forward pass with injected faults
        auto output = network.forward(testInputs[i]);
        auto referenceOutput = referenceOutputs[i];
        
        // Calculate MSE
        float mse = 0.0f;
        for (size_t j = 0; j < output.size(); j++) {
            float diff = output[j] - referenceOutput[j];
            mse += diff * diff;
        }
        mse /= output.size();
        total_mse += mse;
    }
    
    // Calculate average MSE
    result.mean_squared_error = total_mse / numSamples;
    
    // Get error statistics
    auto [totalErrors, correctedErrors] = network.getErrorStats();
    result.detected_faults = totalErrors;
    result.corrected_faults = correctedErrors;
    
    // Calculate correction rate
    if (totalErrors > 0) {
        result.correction_rate = 100.0 * static_cast<double>(correctedErrors) / totalErrors;
    } else {
        result.correction_rate = 100.0; // If no errors, 100% correction rate
    }
    
    return result;
}

// Create a Python visualization script
void createVisualizationScript() {
    std::ofstream pyFile("systematic_fault_visualization.py");
    
    pyFile << R"(
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Read the data
df = pd.read_csv("systematic_fault_results.csv")

# Set up the plotting style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (15, 10)

# Create a figure for fault pattern comparison
plt.figure()

# Group by pattern and protection status, calculate mean MSE
pattern_mse = df.groupby(['pattern', 'protection'])['mse'].mean().unstack()

# Plot pattern comparison
ax = pattern_mse.plot(kind='bar', color=['red', 'green'])
plt.title('MSE by Fault Pattern and Protection Status', fontsize=16)
plt.xlabel('Fault Pattern', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.xticks(rotation=45)
plt.legend(['Protection Disabled', 'Protection Enabled'])
plt.tight_layout()
plt.savefig('mse_by_pattern.png', dpi=300)

# Create a figure for correction rate by pattern
plt.figure()
protection_enabled = df[df['protection'] == 'enabled']
sns.barplot(x='pattern', y='correction_rate', data=protection_enabled)
plt.title('Error Correction Rate by Fault Pattern', fontsize=16)
plt.xlabel('Fault Pattern', fontsize=14)
plt.ylabel('Correction Rate (%)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('correction_by_pattern.png', dpi=300)

# Create a figure for MSE vs. coverage percentage
plt.figure()
coverage_plot = sns.lineplot(x='coverage', y='mse', hue='protection', 
                             style='pattern', data=df, markers=True)
plt.title('MSE vs. Coverage Percentage', fontsize=16)
plt.xlabel('Coverage Percentage', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.tight_layout()
plt.savefig('mse_vs_coverage.png', dpi=300)

# Create a heatmap for pattern effectiveness
plt.figure(figsize=(12, 8))
pivot_table = df.pivot_table(values='mse', 
                            index='pattern', 
                            columns=['protection', 'coverage'])
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title('MSE Heatmap by Pattern, Protection and Coverage', fontsize=16)
plt.tight_layout()
plt.savefig('mse_heatmap.png', dpi=300)

# Summary statistics
print("\nSummary Statistics:")
print(df.groupby(['pattern', 'protection'])['mse', 'correction_rate'].agg(['mean', 'std', 'min', 'max']))

# Create scatter plot of injected faults vs correction rate
plt.figure()
sns.scatterplot(x='injected_faults', y='correction_rate', hue='pattern', 
                size='coverage', sizes=(50, 200), data=protection_enabled)
plt.title('Error Correction Rate vs. Injected Faults', fontsize=16)
plt.xlabel('Number of Injected Faults', fontsize=14)
plt.ylabel('Correction Rate (%)', fontsize=14)
plt.tight_layout()
plt.savefig('correction_vs_faults.png', dpi=300)

print("Visualizations complete. Check the PNG files.")
)";
    
    pyFile.close();
}

int main() {
    std::cout << "Starting systematic fault injection test..." << std::endl;
    
    // Create the fault injector
    SystematicFaultInjector injector;
    
    // Set a fixed seed for reproducibility
    injector.setSeed(42);
    
    // Create a network for testing
    FaultInjectionTestNetwork network(8, 16, 4); // 8 inputs, 16 hidden, 4 outputs
    
    // Create result file
    std::ofstream outFile("systematic_fault_results.csv");
    outFile << FaultInjectionResult::getCsvHeader() << std::endl;
    
    // Define fault patterns to test
    std::vector<SystematicFaultInjector::FaultPattern> patterns = {
        SystematicFaultInjector::SINGLE_BIT,
        SystematicFaultInjector::ADJACENT_BITS,
        SystematicFaultInjector::BYTE_ERROR,
        SystematicFaultInjector::WORD_ERROR,
        SystematicFaultInjector::STUCK_AT_ZERO,
        SystematicFaultInjector::STUCK_AT_ONE,
        SystematicFaultInjector::ROW_COLUMN,
        SystematicFaultInjector::BURST_ERROR
    };
    
    // Coverage percentages to test
    std::vector<int> coverages = {1, 5, 10, 20, 30};
    
    // Run tests with protection enabled and disabled
    for (auto pattern : patterns) {
        std::cout << "Testing pattern: " << SystematicFaultInjector::patternToString(pattern) << std::endl;
        
        for (int coverage : coverages) {
            std::cout << "  Coverage: " << coverage << "%" << std::endl;
            
            // Test with protection disabled
            std::cout << "    Protection disabled..." << std::endl;
            auto result_disabled = runSystematicFaultTest(
                network, injector, pattern, coverage, false);
            outFile << result_disabled.toCsvRow() << std::endl;
            
            // Test with protection enabled
            std::cout << "    Protection enabled..." << std::endl;
            auto result_enabled = runSystematicFaultTest(
                network, injector, pattern, coverage, true);
            outFile << result_enabled.toCsvRow() << std::endl;
            
            // Print summary for this test
            std::cout << "    MSE (disabled): " << result_disabled.mean_squared_error << std::endl;
            std::cout << "    MSE (enabled): " << result_enabled.mean_squared_error << std::endl;
            std::cout << "    Correction rate: " << result_enabled.correction_rate << "%" << std::endl;
        }
    }
    
    outFile.close();
    
    // Create the visualization script
    createVisualizationScript();
    
    std::cout << "Systematic fault injection test completed." << std::endl;
    std::cout << "Results saved to 'systematic_fault_results.csv'." << std::endl;
    std::cout << "Run 'python systematic_fault_visualization.py' to generate visualizations." << std::endl;
    
    return 0;
} 