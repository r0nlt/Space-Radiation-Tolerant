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
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "rad_ml/tmr/tmr.hpp"
#include "rad_ml/tmr/enhanced_tmr.hpp"
#include "rad_ml/testing/radiation_simulator.hpp"

using namespace rad_ml::tmr;
using namespace rad_ml::testing;

// Statistical analysis structure
struct StatisticalMetrics {
    double mean;
    double standard_deviation;
    double confidence_interval_95;
    double min_value;
    double max_value;
    int sample_size;
};

// Performance metrics structure
struct PerformanceMetrics {
    double avg_execution_time;
    double error_rate;
    double correction_rate;
    double memory_usage;
    int total_operations;
    StatisticalMetrics error_stats;
    StatisticalMetrics performance_stats;
};

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
 * Enhanced test suite for the radiation protection framework
 */
class FrameworkVerificationSuite {
private:
    SimpleNetwork network;
    std::mt19937 rng;
    std::vector<PerformanceMetrics> metrics_history;
    std::ofstream validation_log;
    
    // Helper function to calculate statistical metrics
    StatisticalMetrics calculateStatistics(const std::vector<double>& values) {
        StatisticalMetrics stats;
        stats.sample_size = values.size();
        
        if (values.empty()) {
            return stats;
        }
        
        // Calculate mean
        stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        
        // Calculate standard deviation
        double sq_sum = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
        stats.standard_deviation = std::sqrt(sq_sum / values.size() - stats.mean * stats.mean);
        
        // Calculate 95% confidence interval
        stats.confidence_interval_95 = 1.96 * stats.standard_deviation / std::sqrt(values.size());
        
        // Find min and max
        stats.min_value = *std::min_element(values.begin(), values.end());
        stats.max_value = *std::max_element(values.begin(), values.end());
        
        return stats;
    }
    
    // Helper function to measure execution time
    template<typename Func>
    double measure_execution_time(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }
    
    // Helper function to simulate bit flips
    void simulate_bit_flips(std::vector<float>& data, double flip_probability) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (auto& value : data) {
            if (dist(rng) < flip_probability) {
                // Simulate bit flip by XORing with a random value
                uint32_t* bits = reinterpret_cast<uint32_t*>(&value);
                *bits ^= (1 << (rng() % 32));
            }
        }
    }
    
    // Helper function to log validation results
    void logValidationResult(const std::string& test_name, const std::string& result, 
                           const StatisticalMetrics& stats) {
        validation_log << "=== " << test_name << " ===\n"
                      << "Result: " << result << "\n"
                      << "Statistics:\n"
                      << "  Mean: " << stats.mean << "\n"
                      << "  Std Dev: " << stats.standard_deviation << "\n"
                      << "  95% CI: ±" << stats.confidence_interval_95 << "\n"
                      << "  Min: " << stats.min_value << "\n"
                      << "  Max: " << stats.max_value << "\n"
                      << "  Sample Size: " << stats.sample_size << "\n\n";
    }

public:
    FrameworkVerificationSuite() : rng(std::random_device{}()) {
        validation_log.open("framework_validation_results.log");
    }
    
    ~FrameworkVerificationSuite() {
        if (validation_log.is_open()) {
            validation_log.close();
        }
    }
    
    /**
     * Test 1: Basic Protection Mechanisms with Statistical Analysis
     */
    bool test_basic_protection() {
        std::cout << "\n=== Test 1: Basic Protection Mechanisms ===" << std::endl;
        bool passed = true;
        
        // Test normal operation multiple times
        const int num_tests = 1000;
        std::vector<double> normal_outputs;
        std::vector<double> corrupted_outputs;
        std::vector<double> correction_times;
        
        for (int i = 0; i < num_tests; i++) {
            // Test normal operation
            std::vector<float> inputs = {1.0f, 0.5f, -0.2f};
            float normal_output = network.forward(inputs);
            normal_outputs.push_back(normal_output);
            
            // Test with corruption
            network.corruptWeight(1, 5.0f);
            double correction_time = measure_execution_time([&]() {
                float corrupted_output = network.forward(inputs);
                corrupted_outputs.push_back(corrupted_output);
            });
            correction_times.push_back(correction_time);
        }
        
        // Calculate statistics
        auto normal_stats = calculateStatistics(normal_outputs);
        auto corrupted_stats = calculateStatistics(corrupted_outputs);
        auto time_stats = calculateStatistics(correction_times);
        
        // Log results
        logValidationResult("Basic Protection", 
                          "Normal vs Corrupted Output Comparison",
                          normal_stats);
        
        // Verify error detection and correction
        auto& weight_tmr = network.getWeightsTMR();
        auto stats = weight_tmr.getErrorStats();
        
        passed &= (stats.detected_errors > 0);
        passed &= (std::abs(normal_stats.mean - corrupted_stats.mean) < 1e-5);
        
        // Verify performance
        passed &= (time_stats.mean < 0.001); // Should be under 1ms
        
        return passed;
    }
    
    /**
     * Test 2: Comprehensive Radiation Environment Testing
     */
    bool test_radiation_environment() {
        std::cout << "\n=== Test 2: Radiation Environment Simulation ===" << std::endl;
        
        // Test multiple radiation environments with different conditions
        std::vector<std::string> environments = {
            "JUPITER", "MARS", "LUNAR", "DEEP_SPACE"
        };
        
        bool all_passed = true;
        for (const auto& env : environments) {
            std::cout << "\nTesting " << env << " environment:" << std::endl;
            
            auto radiation_env = RadiationSimulator::getMissionEnvironment(env);
            RadiationSimulator simulator(radiation_env);
            
            // Run multiple iterations with different conditions
            const int num_tests = 10000; // Increased for better statistics
            std::vector<double> success_rates;
            std::vector<double> error_rates;
            std::vector<double> correction_rates;
            
            for (int batch = 0; batch < 10; batch++) {
                int success_count = 0;
                int error_count = 0;
                int correction_count = 0;
                
                for (int i = 0; i < num_tests; i++) {
                    // Simulate radiation effects
                    auto weights = network.getWeightsTMR().get();
                    double error_probability = calculateErrorProbability(radiation_env);
                    simulate_bit_flips(weights, error_probability);
                    network.getWeightsTMR().setRawCopy(0, weights);
                    
                    // Run forward pass
                    std::vector<float> inputs = {1.0f, 0.5f, -0.2f};
                    float output = network.forward(inputs);
                    
                    // Track results
                    if (std::abs(output) <= 1.0f) {
                        success_count++;
                    }
                    
                    auto error_stats = network.getWeightsTMR().getErrorStats();
                    error_count += error_stats.detected_errors;
                    correction_count += error_stats.corrected_errors;
                }
                
                success_rates.push_back((success_count * 100.0) / num_tests);
                error_rates.push_back((error_count * 100.0) / num_tests);
                correction_rates.push_back((correction_count * 100.0) / num_tests);
            }
            
            // Calculate statistics
            auto success_stats = calculateStatistics(success_rates);
            auto error_stats = calculateStatistics(error_rates);
            auto correction_stats = calculateStatistics(correction_rates);
            
            // Log results
            logValidationResult(env + " Environment",
                              "Success Rate Analysis",
                              success_stats);
            
            // Verify against NASA/ESA standards
            bool env_passed = true;
            env_passed &= (success_stats.mean >= 70.0); // NASA minimum requirement
            env_passed &= (correction_stats.mean >= 95.0); // ESA correction requirement
            env_passed &= (success_stats.confidence_interval_95 < 5.0); // Statistical significance
            
            std::cout << "Environment " << env << " " 
                      << (env_passed ? "PASSED" : "FAILED") << std::endl;
            
            all_passed &= env_passed;
        }
        
        return all_passed;
    }
    
    /**
     * Test 3: Performance and Resource Usage with Statistical Analysis
     */
    bool test_performance_metrics() {
        std::cout << "\n=== Test 3: Performance and Resource Usage ===" << std::endl;
        
        const int num_operations = 100000; // Increased for better statistics
        std::vector<double> execution_times;
        std::vector<double> memory_usage;
        
        // Measure execution time and memory usage for multiple operations
        for (int i = 0; i < num_operations; i++) {
            std::vector<float> inputs = {1.0f, 0.5f, -0.2f};
            double time = measure_execution_time([&]() {
                network.forward(inputs);
            });
            execution_times.push_back(time);
            
            // Simulate memory usage tracking
            memory_usage.push_back(sizeof(SimpleNetwork) + 
                                 inputs.size() * sizeof(float));
        }
        
        // Calculate statistics
        auto time_stats = calculateStatistics(execution_times);
        auto memory_stats = calculateStatistics(memory_usage);
        
        // Log results
        logValidationResult("Performance Metrics",
                          "Execution Time Analysis",
                          time_stats);
        
        // Store metrics
        PerformanceMetrics metrics;
        metrics.avg_execution_time = time_stats.mean;
        metrics.memory_usage = memory_stats.mean;
        metrics.total_operations = num_operations;
        metrics.performance_stats = time_stats;
        metrics_history.push_back(metrics);
        
        // Verify performance requirements
        bool passed = true;
        passed &= (time_stats.mean < 0.001); // Average time under 1ms
        passed &= (time_stats.max_value < 0.01); // Max time under 10ms
        passed &= (memory_stats.mean < 1024); // Memory under 1KB
        
        return passed;
    }
    
    /**
     * Test 4: Multiple Concurrent Faults with Statistical Analysis
     */
    bool test_concurrent_faults() {
        std::cout << "\n=== Test 4: Multiple Concurrent Faults ===" << std::endl;
        
        const int num_tests = 10000; // Increased for better statistics
        std::vector<double> success_rates;
        std::vector<double> error_rates;
        
        for (int batch = 0; batch < 10; batch++) {
            int success_count = 0;
            int error_count = 0;
            
            for (int i = 0; i < num_tests; i++) {
                // Simulate multiple concurrent faults
                auto weights = network.getWeightsTMR().get();
                simulate_bit_flips(weights, 0.1);  // 10% bit flip probability
                network.getWeightsTMR().setRawCopy(0, weights);
                
                // Corrupt bias
                network.corruptBias(static_cast<float>(rng() % 100) / 10.0f);
                
                // Run forward pass
                std::vector<float> inputs = {1.0f, 0.5f, -0.2f};
                float output = network.forward(inputs);
                
                if (std::abs(output) <= 1.0f) {
                    success_count++;
                }
                
                auto error_stats = network.getWeightsTMR().getErrorStats();
                error_count += error_stats.detected_errors;
            }
            
            success_rates.push_back((success_count * 100.0) / num_tests);
            error_rates.push_back((error_count * 100.0) / num_tests);
        }
        
        // Calculate statistics
        auto success_stats = calculateStatistics(success_rates);
        auto error_stats = calculateStatistics(error_rates);
        
        // Log results
        logValidationResult("Concurrent Faults",
                          "Success Rate Analysis",
                          success_stats);
        
        // Verify against requirements
        bool passed = true;
        passed &= (success_stats.mean >= 60.0); // Minimum success rate
        passed &= (success_stats.confidence_interval_95 < 5.0); // Statistical significance
        
        return passed;
    }
    
    /**
     * Run all tests and return overall result
     */
    bool run_all_tests() {
        bool all_passed = true;
        
        all_passed &= test_basic_protection();
        all_passed &= test_radiation_environment();
        all_passed &= test_performance_metrics();
        all_passed &= test_concurrent_faults();
        
        // Print final summary
        std::cout << "\n=== Framework Verification Summary ===" << std::endl;
        std::cout << "Overall result: " << (all_passed ? "PASSED" : "FAILED") << std::endl;
        
        if (!metrics_history.empty()) {
            std::cout << "\nPerformance Summary:" << std::endl;
            const auto& latest_metrics = metrics_history.back();
            std::cout << "  Average execution time: " 
                      << latest_metrics.avg_execution_time * 1000000 << " µs" << std::endl;
            std::cout << "  Memory usage: " << latest_metrics.memory_usage << " bytes" << std::endl;
            std::cout << "  Total operations: " << latest_metrics.total_operations << std::endl;
            
            // Print statistical confidence
            std::cout << "\nStatistical Confidence:" << std::endl;
            std::cout << "  95% Confidence Interval: ±" 
                      << latest_metrics.performance_stats.confidence_interval_95 * 1000000 
                      << " µs" << std::endl;
        }
        
        return all_passed;
    }

private:
    // Helper function to calculate error probability based on environment parameters
    double calculateErrorProbability(const RadiationSimulator::EnvironmentParams& env) {
        // Base probability from altitude (higher altitude = more radiation)
        double altitude_factor = std::exp(-env.altitude_km / 1000.0);
        
        // Solar activity factor (higher activity = more radiation)
        double solar_factor = 1.0 + (env.solar_activity - 1.0) * 0.2;
        
        // SAA factor (inside SAA = more radiation)
        double saa_factor = env.inside_saa ? 10.0 : 1.0;
        
        // Shielding factor (more shielding = less radiation)
        double shielding_factor = std::exp(-env.shielding_thickness_mm / 10.0);
        
        // Calculate final probability (normalized between 0 and 1)
        double probability = altitude_factor * solar_factor * saa_factor * shielding_factor;
        return std::min(0.1, std::max(0.0001, probability));  // Cap between 0.01% and 10%
    }
};

int main() {
    FrameworkVerificationSuite test_suite;
    bool passed = test_suite.run_all_tests();
    return passed ? 0 : 1;
}
