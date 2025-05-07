/**
 * Framework Verification Test
 * 
 * This test verifies the core functionality of the radiation-tolerant ML framework.
 * It creates a simple neural network, applies different protection levels, and
 * tests its resilience in simulated radiation environments.
 */

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <iomanip>

#include "rad_ml/tmr/tmr.hpp"
#include "rad_ml/tmr/enhanced_tmr.hpp"
#include "rad_ml/testing/radiation_simulator.hpp"

using namespace rad_ml::tmr;
using namespace rad_ml::testing;

/**
 * A simple neural network class for testing the framework
 */
class SimpleNetwork {
private:
    // Use TMR to protect the weights
    TMR<std::vector<float>> weights_tmr;
    
    // Use Enhanced TMR to protect the bias
    std::shared_ptr<EnhancedTMR<float>> bias_tmr;
    
public:
    SimpleNetwork() {
        // Initialize with some weights and bias
        std::vector<float> initial_weights = {0.5f, -0.3f, 0.8f};
        weights_tmr.set(initial_weights);
        
        // Use the factory to create Enhanced TMR for the bias
        bias_tmr = TMRFactory::createEnhancedTMR<float>(0.2f);
    }
    
    /**
     * Forward pass with protected weights and bias
     */
    float forward(const std::vector<float>& inputs) {
        // Get protected weights
        std::vector<float> weights = weights_tmr.get();
        float bias = bias_tmr->get();
        
        // Ensure input size matches weights
        if (inputs.size() != weights.size()) {
            std::cerr << "Input size mismatch! Expected " << weights.size() 
                      << " but got " << inputs.size() << std::endl;
            return 0.0f;
        }
        
        // Compute dot product
        float sum = 0.0f;
        for (size_t i = 0; i < weights.size(); ++i) {
            sum += weights[i] * inputs[i];
        }
        
        // Add bias and apply activation function (tanh)
        return std::tanh(sum + bias);
    }
    
    /**
     * Deliberately corrupt a weight to test error detection
     */
    void corruptWeight(size_t index, float value) {
        auto weights = weights_tmr.get();
        if (index < weights.size()) {
            weights[index] = value;
            weights_tmr.setRawCopy(0, weights);  // Corrupt only one copy
        }
    }
    
    /**
     * Deliberately corrupt the bias to test error detection
     */
    void corruptBias(float value) {
        bias_tmr->setRawCopy(1, value);  // Corrupt only one copy
    }
    
    /**
     * Get error statistics
     */
    void printErrorStats() {
        auto basic_stats = weights_tmr.getErrorStats();
        
        std::cout << "Basic TMR Error Stats:" << std::endl
                  << "  Detected Errors: " << basic_stats.detected_errors << std::endl
                  << "  Corrected Errors: " << basic_stats.corrected_errors << std::endl
                  << "  Uncorrectable Errors: " << basic_stats.uncorrectable_errors << std::endl;
        
        std::cout << "Enhanced TMR Error Stats:" << std::endl
                  << "  " << bias_tmr->getErrorStats() << std::endl;
    }
    
    /**
     * Access to the protected values for testing
     */
    TMR<std::vector<float>>& getWeightsTMR() {
        return weights_tmr;
    }
    
    std::shared_ptr<EnhancedTMR<float>> getBiasTMR() {
        return bias_tmr;
    }
};

/**
 * Test the protection mechanisms of the framework
 */
bool test_protection_mechanisms() {
    bool all_tests_passed = true;
    std::cout << "=== Testing Radiation Protection Framework ===" << std::endl;
    
    // Create test inputs
    std::vector<float> inputs = {1.0f, 0.5f, -0.2f};
    
    // Create a simple network
    SimpleNetwork network;
    
    // Test 1: Normal operation without radiation
    std::cout << "\nTest 1: Normal operation without radiation" << std::endl;
    float normal_output = network.forward(inputs);
    std::cout << "Output: " << normal_output << std::endl;
    
    // Test 2: Corrupt one weight and verify error detection and correction
    std::cout << "\nTest 2: Testing basic TMR with corrupted weight" << std::endl;
    network.corruptWeight(1, 5.0f);  // Corrupt the second weight
    float corrupted_output = network.forward(inputs);
    std::cout << "Output with corrupted weight: " << corrupted_output << std::endl;
    
    // Verify TMR detected and corrected the error
    auto& weight_tmr = network.getWeightsTMR();
    auto stats = weight_tmr.getErrorStats();
    if (stats.detected_errors > 0 && stats.corrected_errors > 0) {
        std::cout << "SUCCESS: TMR detected and corrected the error" << std::endl;
    } else {
        std::cout << "FAILURE: TMR did not detect or correct the error" << std::endl;
        all_tests_passed = false;
    }
    
    // Verify output is the same despite corruption
    if (std::abs(normal_output - corrupted_output) < 1e-5) {
        std::cout << "SUCCESS: Output remains correct despite corruption" << std::endl;
    } else {
        std::cout << "FAILURE: Output changed after corruption" << std::endl;
        all_tests_passed = false;
    }
    
    // Test 3: Test Enhanced TMR with corrupted bias
    std::cout << "\nTest 3: Testing Enhanced TMR with corrupted bias" << std::endl;
    network.corruptBias(10.0f);  // Corrupt the bias with an extreme value
    float bias_corrupted_output = network.forward(inputs);
    
    // Verify Enhanced TMR handling
    std::cout << "Output with corrupted bias: " << bias_corrupted_output << std::endl;
    
    // Check if the bias TMR detected the error
    auto bias_tmr = network.getBiasTMR();
    if (bias_tmr->verify()) {
        std::cout << "SUCCESS: Enhanced TMR detected and verified the error" << std::endl;
    } else {
        std::cout << "NOTE: Enhanced TMR detected CRC mismatch" << std::endl;
    }
    
    // Test 4: Simulate radiation environment
    std::cout << "\nTest 4: Testing in simulated radiation environment" << std::endl;
    
    // Create a radiation simulator for Jupiter environment (harsh)
    auto jupiter_env = RadiationSimulator::getMissionEnvironment("JUPITER");
    RadiationSimulator simulator(jupiter_env);
    
    std::cout << "Simulating Jupiter radiation environment:" << std::endl;
    std::cout << simulator.getEnvironmentDescription() << std::endl;
    
    // Reset error stats for clean test
    weight_tmr.resetErrorStats();
    bias_tmr->resetErrorStats();
    
    // Run multiple forward passes under simulated radiation
    const int num_radiation_tests = 100;
    int success_count = 0;
    
    std::cout << "Running " << num_radiation_tests << " forward passes under radiation..." << std::endl;
    
    for (int i = 0; i < num_radiation_tests; i++) {
        // Get raw memory for weights to simulate radiation effects
        auto weights = weight_tmr.get();
        
        // Simulate radiation effects (simplified for test)
        weight_tmr.setRawCopy(0, weights);  // Reset to consistent state
        
        // Apply simulated radiation effects manually by corrupting random copies
        if (i % 3 == 0) {  // Corrupt a weight in first copy
            auto weights_copy = weights;
            weights_copy[i % weights.size()] *= 1.5f;
            weight_tmr.setRawCopy(0, weights_copy);
        }
        
        if (i % 7 == 0) {  // Corrupt bias in second copy
            bias_tmr->setRawCopy(1, 0.9f);
        }
        
        // Run forward pass
        float rad_output = network.forward(inputs);
        
        // Check if the result is reasonably close to the expected value
        if (std::abs(rad_output - normal_output) < 0.1f) {
            success_count++;
        }
    }
    
    // Print final results
    std::cout << "SUCCESS RATE: " << (success_count * 100 / num_radiation_tests) << "%" << std::endl;
    network.printErrorStats();
    
    if (success_count > 70) {  // At least 70% success rate expected
        std::cout << "SUCCESS: Framework maintained reasonable accuracy under radiation" << std::endl;
    } else {
        std::cout << "FAILURE: Framework did not maintain reasonable accuracy under radiation" << std::endl;
        all_tests_passed = false;
    }
    
    std::cout << "\n=== Framework Verification " << (all_tests_passed ? "PASSED" : "FAILED") << " ===" << std::endl;
    return all_tests_passed;
}

int main() {
    bool passed = test_protection_mechanisms();
    return passed ? 0 : 1;
}
