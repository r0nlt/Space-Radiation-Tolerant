/**
 * @file neural_network_validation.cpp
 * @brief Validation of neural network radiation tolerance in space environments
 * 
 * This file implements a validation test for neural networks operating in
 * various space radiation environments, testing the effectiveness of
 * radiation hardening techniques.
 */

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <random>
#include <chrono>
#include <functional>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cmath>

#include "rad_ml/neural/protected_neural_network.hpp"
#include "rad_ml/neural/multi_bit_protection.hpp"
#include "rad_ml/neural/radiation_environment.hpp"

// Optional debugging visualization
#ifdef ENABLE_VISUALIZATION
#include <opencv2/opencv.hpp>
#endif

// Use appropriate namespaces
using namespace rad_ml::neural;

/**
 * @brief Structure to hold test results
 */
struct TestResult {
    std::string test_name;
    double accuracy_baseline;
    double accuracy_under_radiation;
    double protection_effectiveness;
    size_t errors_detected;
    size_t errors_corrected;
    double execution_time_ms;
};

/**
 * @brief Structure to represent a sample dataset
 */
struct Dataset {
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> outputs;
    
    // Add a new sample
    void addSample(const std::vector<float>& input, const std::vector<float>& output) {
        inputs.push_back(input);
        outputs.push_back(output);
    }
    
    // Get dataset size
    size_t size() const {
        return inputs.size();
    }
};

/**
 * @brief Generate a synthetic dataset for testing
 * 
 * @param input_size Input feature size
 * @param output_size Output size
 * @param num_samples Number of samples to generate
 * @param seed Random seed for reproducibility
 * @return Dataset with synthetic samples
 */
Dataset generateSyntheticDataset(
    size_t input_size,
    size_t output_size,
    size_t num_samples,
    uint64_t seed = 42
) {
    Dataset dataset;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Create a simple pattern: sum of inputs determines pattern
    for (size_t i = 0; i < num_samples; ++i) {
        std::vector<float> input;
        for (size_t j = 0; j < input_size; ++j) {
            input.push_back(dist(rng));
        }
        
        // Create simple output based on input
        std::vector<float> output(output_size, 0.0f);
        float sum = std::accumulate(input.begin(), input.end(), 0.0f);
        
        if (output_size >= 2) {
            // For 2+ outputs, use thresholds on sum
            if (sum > 0.5f) {
                output[0] = 1.0f;
            } else if (sum < -0.5f) {
                output[1] = 1.0f;
            } else {
                if (output_size > 2) {
                    output[2] = 1.0f;
                } else {
                    output[0] = 0.5f;
                    output[1] = 0.5f;
                }
            }
        } else {
            // For single output, use a continuous function
            output[0] = std::tanh(sum);
        }
        
        dataset.addSample(input, output);
    }
    
    return dataset;
}

/**
 * @brief Calculate accuracy between predicted and actual outputs
 * 
 * @param predicted Predicted outputs
 * @param actual Actual outputs
 * @param threshold Classification threshold
 * @return Accuracy as a percentage
 */
double calculateAccuracy(
    const std::vector<std::vector<float>>& predicted,
    const std::vector<std::vector<float>>& actual,
    float threshold = 0.5f
) {
    if (predicted.size() != actual.size() || predicted.empty()) {
        return 0.0;
    }
    
    size_t correct = 0;
    size_t total = predicted.size();
    
    for (size_t i = 0; i < total; ++i) {
        const auto& pred = predicted[i];
        const auto& act = actual[i];
        
        if (pred.size() != act.size()) {
            continue;
        }
        
        // For multi-class, find index of max value
        if (pred.size() > 1) {
            size_t pred_max_idx = std::max_element(pred.begin(), pred.end()) - pred.begin();
            size_t act_max_idx = std::max_element(act.begin(), act.end()) - act.begin();
            
            if (pred_max_idx == act_max_idx) {
                correct++;
            }
        } 
        // For binary classification, use threshold
        else if (pred.size() == 1) {
            bool pred_class = pred[0] >= threshold;
            bool act_class = act[0] >= threshold;
            
            if (pred_class == act_class) {
                correct++;
            }
        }
    }
    
    return 100.0 * static_cast<double>(correct) / static_cast<double>(total);
}

/**
 * @brief Test neural network in a specific space radiation environment
 * 
 * @param mission Space mission profile
 * @param protection_level Protection level to test
 * @param dataset Test dataset
 * @return Test results
 */
TestResult testInSpaceEnvironment(
    SpaceMission mission,
    ProtectionLevel protection_level,
    const Dataset& dataset
) {
    TestResult result;
    result.test_name = "Mission: " + RadiationEnvironment(mission).getMissionName();
    
    // Create neural network with the specified protection level
    const size_t input_size = dataset.inputs[0].size();
    const size_t output_size = dataset.outputs[0].size();
    const std::vector<size_t> layer_sizes = {input_size, 16, output_size};
    
    ProtectedNeuralNetwork<float> network(layer_sizes, protection_level);
    
    // Set activation functions
    auto relu = [](float x) { return x > 0 ? x : 0; };
    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    
    network.setActivationFunction(0, relu);
    network.setActivationFunction(1, sigmoid);
    
    // Train the network (simple training loop for demonstration)
    // Note: In a real application, use a proper training algorithm
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<size_t> dist(0, dataset.size() - 1);
    
    for (int epoch = 0; epoch < 100; ++epoch) {
        size_t idx = dist(rng);
        std::vector<float> prediction = network.forward(dataset.inputs[idx]);
        
        // Simple update rule (not real backprop, just for demonstration)
        // Weights would be updated here in a real training scenario
    }
    
    // Baseline accuracy (no radiation)
    std::vector<std::vector<float>> baseline_predictions;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (const auto& input : dataset.inputs) {
        std::vector<float> prediction = network.forward(input);
        baseline_predictions.push_back(prediction);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    result.execution_time_ms = duration.count();
    
    result.accuracy_baseline = calculateAccuracy(baseline_predictions, dataset.outputs);
    
    // Create radiation environment
    RadiationEnvironment environment(mission);
    
    // Test with radiation
    std::vector<std::vector<float>> radiation_predictions;
    
    // Reset error statistics
    network.resetErrorStats();
    
    // Create orbital trajectory for testing
    std::vector<OrbitalPosition> trajectory;
    
    // Generate a trajectory covering different radiation environments
    double inclination = (mission == SpaceMission::LEO_EQUATORIAL) ? 10.0 : 
                        (mission == SpaceMission::LEO_POLAR) ? 85.0 : 45.0;
                        
    double altitude = (mission == SpaceMission::LEO_EQUATORIAL || 
                      mission == SpaceMission::LEO_POLAR) ? 400.0 : 
                      (mission == SpaceMission::MEO) ? 20000.0 : 
                      (mission == SpaceMission::GEO) ? 35786.0 : 1000.0;
    
    for (int i = 0; i < 36; ++i) {
        double time_hours = i * 0.25; // 15-minute intervals
        trajectory.push_back(environment.generateOrbitPosition(inclination, altitude, time_hours));
    }
    
    // Add South Atlantic Anomaly position if applicable
    if (mission == SpaceMission::LEO_EQUATORIAL || mission == SpaceMission::LEO_POLAR) {
        trajectory.push_back({-30.0, -30.0, altitude}); // SAA position
    }
    
    // Test along the trajectory
    std::vector<double> radiation_levels;
    for (const auto& position : trajectory) {
        double radiation_level = environment.calculateSEUProbability(position);
        radiation_levels.push_back(radiation_level);
        
        // Scale radiation level to something reasonable for the simulation
        // Actual SEU probabilities are very small, so we scale for the test
        double scaled_radiation = radiation_level * 1e7;
        
        // Apply radiation effects to the network
        network.applyRadiationEffects(scaled_radiation, static_cast<uint64_t>(position.latitude * 1000));
        
        // Get predictions under radiation
        for (size_t i = 0; i < std::min(size_t(5), dataset.size()); ++i) {
            const auto& input = dataset.inputs[i];
            std::vector<float> prediction = network.forward(input, scaled_radiation);
            
            if (radiation_predictions.size() <= i) {
                radiation_predictions.push_back(prediction);
            } else {
                // Average predictions along trajectory
                for (size_t j = 0; j < prediction.size(); ++j) {
                    radiation_predictions[i][j] = (radiation_predictions[i][j] + prediction[j]) / 2.0f;
                }
            }
        }
    }
    
    // Fill in remaining predictions if needed
    for (size_t i = radiation_predictions.size(); i < dataset.size(); ++i) {
        radiation_predictions.push_back(network.forward(dataset.inputs[i]));
    }
    
    // Calculate accuracy under radiation
    result.accuracy_under_radiation = calculateAccuracy(radiation_predictions, dataset.outputs);
    
    // Get error statistics
    auto [detected_errors, corrected_errors] = network.getErrorStats();
    result.errors_detected = detected_errors;
    result.errors_corrected = corrected_errors;
    
    // Calculate protection effectiveness
    if (result.accuracy_baseline > 0) {
        result.protection_effectiveness = 
            (result.accuracy_under_radiation / result.accuracy_baseline) * 100.0;
    } else {
        result.protection_effectiveness = 0.0;
    }
    
#ifdef ENABLE_VISUALIZATION
    // Create a heatmap visualization of radiation along the trajectory
    cv::Mat heatmap(400, 800, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Normalize radiation levels for visualization
    double max_level = *std::max_element(radiation_levels.begin(), radiation_levels.end());
    for (size_t i = 0; i < trajectory.size(); ++i) {
        const auto& pos = trajectory[i];
        double level = radiation_levels[i] / max_level;
        
        // Convert lat/lon to image coordinates
        int x = static_cast<int>((pos.longitude + 180) / 360.0 * 800);
        int y = static_cast<int>((90 - pos.latitude) / 180.0 * 400);
        
        // Draw radiation level
        cv::circle(heatmap, cv::Point(x, y), 5, 
                cv::Scalar(0, 255 * (1 - level), 255 * level), -1);
    }
    
    // Save the heatmap
    std::string filename = "radiation_heatmap_" + 
                         std::to_string(static_cast<int>(mission)) + ".png";
    cv::imwrite(filename, heatmap);
#endif
    
    return result;
}

/**
 * @brief Test neural network under radiation with different protection levels
 * 
 * @param mission Space mission profile
 * @param dataset Test dataset
 * @return Vector of test results for each protection level
 */
std::vector<TestResult> testProtectionLevels(
    SpaceMission mission,
    const Dataset& dataset
) {
    std::vector<TestResult> results;
    
    // Test different protection levels
    std::vector<ProtectionLevel> protection_levels = {
        ProtectionLevel::NONE,
        ProtectionLevel::CHECKSUM_ONLY,
        ProtectionLevel::SELECTIVE_TMR,
        ProtectionLevel::FULL_TMR,
        ProtectionLevel::ADAPTIVE_TMR,
        ProtectionLevel::SPACE_OPTIMIZED
    };
    
    for (auto level : protection_levels) {
        TestResult result = testInSpaceEnvironment(mission, level, dataset);
        
        // Add protection level to test name
        switch (level) {
            case ProtectionLevel::NONE:
                result.test_name += " / No Protection";
                break;
            case ProtectionLevel::CHECKSUM_ONLY:
                result.test_name += " / Checksum Only";
                break;
            case ProtectionLevel::SELECTIVE_TMR:
                result.test_name += " / Selective TMR";
                break;
            case ProtectionLevel::FULL_TMR:
                result.test_name += " / Full TMR";
                break;
            case ProtectionLevel::ADAPTIVE_TMR:
                result.test_name += " / Adaptive TMR";
                break;
            case ProtectionLevel::SPACE_OPTIMIZED:
                result.test_name += " / Space-Optimized TMR";
                break;
        }
        
        results.push_back(result);
    }
    
    return results;
}

/**
 * @brief Print test results in a table
 * 
 * @param results Vector of test results
 */
void printResults(const std::vector<TestResult>& results) {
    // Table header
    std::cout << "\n" << std::string(100, '-') << "\n";
    std::cout << "| " << std::left << std::setw(40) << "Test" 
              << " | " << std::setw(10) << "Baseline"
              << " | " << std::setw(10) << "Radiation"
              << " | " << std::setw(10) << "Effect."
              << " | " << std::setw(10) << "Errors"
              << " | " << std::setw(10) << "Fixed"
              << " |\n";
    std::cout << std::string(100, '-') << "\n";
    
    // Table rows
    for (const auto& result : results) {
        std::cout << "| " << std::left << std::setw(40) << result.test_name
                  << " | " << std::fixed << std::setprecision(2) << std::setw(10) << result.accuracy_baseline
                  << " | " << std::setw(10) << result.accuracy_under_radiation
                  << " | " << std::setw(10) << result.protection_effectiveness
                  << " | " << std::setw(10) << result.errors_detected
                  << " | " << std::setw(10) << result.errors_corrected
                  << " |\n";
    }
    
    std::cout << std::string(100, '-') << "\n";
}

/**
 * @brief Compare multi-bit upset patterns in neural network weights
 */
void testMultiBitUpsets() {
    std::cout << "\n=== Multi-Bit Upset Test ===\n";
    
    // Test value
    float test_value = 3.14159f;
    
    // Test different upset types
    std::vector<MultibitUpsetType> upset_types = {
        MultibitUpsetType::SINGLE_BIT,
        MultibitUpsetType::ADJACENT_BITS,
        MultibitUpsetType::ROW_UPSET,
        MultibitUpsetType::COLUMN_UPSET,
        MultibitUpsetType::RANDOM_MULTI
    };
    
    std::vector<std::string> type_names = {
        "Single Bit",
        "Adjacent Bits",
        "Row Upset",
        "Column Upset",
        "Random Multi"
    };
    
    std::cout << "\nOriginal value: " << test_value << "\n";
    std::cout << "Bit pattern: " << std::bitset<32>(*(uint32_t*)&test_value) << "\n\n";
    
    for (size_t i = 0; i < upset_types.size(); ++i) {
        auto type = upset_types[i];
        std::cout << "Testing " << type_names[i] << " upsets:\n";
        
        for (int j = 0; j < 5; ++j) {
            float corrupted = MultibitProtection<float>::applyMultiBitErrors(
                test_value, 1.0, type, j + 42);
            
            // Skip if no change (unlikely with error_rate=1.0)
            if (corrupted == test_value) continue;
            
            std::cout << "  Corrupted value: " << corrupted << "\n";
            std::cout << "  Bit pattern: " << std::bitset<32>(*(uint32_t*)&corrupted) << "\n";
            
            // Create protected value and try to correct
            MultibitProtection<float> protected_value(corrupted, ECCCodingScheme::HAMMING);
            protected_value.setValue(test_value); // Set original value to calculate ECC
            
            // Manually corrupt the value with same pattern
            *(float*)&protected_value = corrupted;
            
            // Try to correct
            float corrected = protected_value.getValue();
            bool fixed = (corrected == test_value);
            
            std::cout << "  Corrected: " << (fixed ? "YES" : "NO") 
                      << " - Value: " << corrected << "\n\n";
        }
    }
}

/**
 * @brief Main function
 */
int main(int argc, char** argv) {
    std::cout << "\n=== Neural Network Radiation Tolerance Validation ===\n";
    
    // Create test dataset
    Dataset dataset = generateSyntheticDataset(4, 3, 100);
    
    // Test in different space environments
    std::vector<SpaceMission> missions = {
        SpaceMission::LEO_EQUATORIAL,
        SpaceMission::LEO_POLAR,
        SpaceMission::MEO,
        SpaceMission::GEO,
        SpaceMission::LUNAR,
        SpaceMission::MARS,
        SpaceMission::JUPITER,
        SpaceMission::SOLAR_PROBE
    };
    
    std::vector<TestResult> all_results;
    
    for (auto mission : missions) {
        auto results = testProtectionLevels(mission, dataset);
        all_results.insert(all_results.end(), results.begin(), results.end());
    }
    
    // Print results
    printResults(all_results);
    
    // Test multi-bit upset patterns
    testMultiBitUpsets();
    
    return 0;
} 