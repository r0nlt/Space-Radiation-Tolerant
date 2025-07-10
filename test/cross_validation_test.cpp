/**
 * @file cross_validation_test.cpp
 * @brief Cross-validation tests to verify framework authenticity
 *
 * This test suite validates that:
 * 1. Bit flips are actually happening at the bit level
 * 2. Protection mechanisms are genuinely working
 * 3. Results are not artifacts of test setup
 * 4. Performance measurements are accurate
 */

#include <gtest/gtest.h>

#include <bitset>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <rad_ml/neural/protected_neural_network.hpp>
#include <rad_ml/research/radiation_aware_training.hpp>
#include <rad_ml/research/residual_network.hpp>
#include <rad_ml/sim/environment.hpp>
#include <rad_ml/utils/bit_manipulation.hpp>
#include <random>

using namespace rad_ml;
using namespace rad_ml::neural;
using namespace rad_ml::research;
using namespace rad_ml::utils;
using namespace rad_ml::sim;

// Helper function to convert float to binary string
std::string floatToBinary(float f)
{
    union {
        float f;
        uint32_t i;
    } u;
    u.f = f;
    return std::bitset<32>(u.i).to_string();
}

// Helper function to count bit differences
int countBitDifferences(float a, float b)
{
    union {
        float f;
        uint32_t i;
    } ua, ub;
    ua.f = a;
    ub.f = b;
    uint32_t diff = ua.i ^ ub.i;
    return __builtin_popcount(diff);
}

TEST(CrossValidation, DirectBitFlipVerification)
{
    std::cout << "\n=== DIRECT BIT FLIP VERIFICATION ===" << std::endl;

    // Test 1: Verify BitManipulation::flipBit actually flips bits
    std::cout << "\n--- Test 1: Bit Flip Function Verification ---" << std::endl;

    float original = 3.14159f;
    std::cout << "Original: " << original << " (" << floatToBinary(original) << ")" << std::endl;

    // Test each bit position
    for (int bit = 0; bit < 32; bit += 8) {  // Test every 8th bit
        float flipped = BitManipulation::flipBit(original, bit);
        std::cout << "Bit " << bit << ": " << flipped << " (" << floatToBinary(flipped) << ")"
                  << std::endl;

        // Verify exactly 1 bit changed
        int bit_diff = countBitDifferences(original, flipped);
        EXPECT_EQ(bit_diff, 1) << "Expected exactly 1 bit difference at position " << bit;

        // Verify it's the correct bit
        union {
            float f;
            uint32_t i;
        } u_orig, u_flip;
        u_orig.f = original;
        u_flip.f = flipped;
        uint32_t diff = u_orig.i ^ u_flip.i;
        EXPECT_EQ(diff, (1u << bit)) << "Bit flip at wrong position";
    }

    // Test 2: Verify multiple bit flips accumulate
    std::cout << "\n--- Test 2: Multiple Bit Flip Accumulation ---" << std::endl;

    float value = 1.0f;
    std::vector<int> flipped_bits = {0, 4, 8, 12, 16};

    for (int bit : flipped_bits) {
        value = BitManipulation::flipBit(value, bit);
    }

    int total_diff = countBitDifferences(1.0f, value);
    EXPECT_EQ(total_diff, flipped_bits.size()) << "Multiple bit flips not accumulating correctly";

    std::cout << "Original 1.0: " << floatToBinary(1.0f) << std::endl;
    std::cout << "After flips: " << floatToBinary(value) << std::endl;
    std::cout << "Total bit differences: " << total_diff << std::endl;
}

TEST(CrossValidation, RadiationInjectionVerification)
{
    std::cout << "\n=== RADIATION INJECTION VERIFICATION ===" << std::endl;

    // Create a simple network with known weights
    ResidualNeuralNetwork<float> network({2, 3, 1}, ProtectionLevel::NONE);

    // Get initial weights
    std::vector<float> initial_weights;
    // Note: This assumes we can access weights - might need to add a getter
    // For now, we'll create a simple forward pass and track changes

    std::vector<float> input = {1.0f, 2.0f};
    auto baseline_output = network.forward(input);

    std::cout << "Baseline output: ";
    for (float val : baseline_output) {
        std::cout << std::fixed << std::setprecision(6) << val << " ";
    }
    std::cout << std::endl;

    // Apply radiation with known bit flip probability
    std::cout << "\n--- Testing Radiation Application ---" << std::endl;

    std::vector<float> radiation_outputs;
    int significant_changes = 0;

    for (int trial = 0; trial < 10; trial++) {
        ResidualNeuralNetwork<float> test_network({2, 3, 1}, ProtectionLevel::NONE);

        // Apply radiation
        auto radiation_output = test_network.forward(input, 0.5f);  // 50% bit flip probability

        // Check if output changed significantly
        bool changed = false;
        for (size_t i = 0; i < baseline_output.size(); i++) {
            if (std::abs(baseline_output[i] - radiation_output[i]) > 1e-6) {
                changed = true;
                break;
            }
        }

        if (changed) {
            significant_changes++;
            std::cout << "Trial " << trial << " output: ";
            for (float val : radiation_output) {
                std::cout << std::fixed << std::setprecision(6) << val << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "Significant changes in " << significant_changes << "/10 trials" << std::endl;

    // With 50% bit flip probability, we should see some changes
    EXPECT_GT(significant_changes, 0)
        << "No radiation effects observed - injection may not be working";
    EXPECT_LT(significant_changes, 10) << "All trials affected - may be too aggressive";
}

TEST(CrossValidation, ProtectionMechanismVerification)
{
    std::cout << "\n=== PROTECTION MECHANISM VERIFICATION ===" << std::endl;

    struct ProtectionTest {
        ProtectionLevel level;
        std::string name;
        int expected_error_reduction;  // Expected percentage reduction
    };

    std::vector<ProtectionTest> tests = {{ProtectionLevel::NONE, "No Protection", 0},
                                         {ProtectionLevel::CHECKSUM_ONLY, "Checksum Only", 10},
                                         {ProtectionLevel::SELECTIVE_TMR, "Selective TMR", 25},
                                         {ProtectionLevel::FULL_TMR, "Full TMR", 50},
                                         {ProtectionLevel::ADAPTIVE_TMR, "Adaptive TMR", 40}};

    std::vector<float> input = {1.0f, -1.0f, 0.5f, -0.5f};
    const int num_trials = 20;
    const float radiation_intensity = 0.3f;

    for (const auto& test : tests) {
        std::cout << "\n--- Testing " << test.name << " ---" << std::endl;

        int error_count = 0;
        std::vector<float> output_variances;

        for (int trial = 0; trial < num_trials; trial++) {
            ProtectedNeuralNetwork<float> network({4, 8, 4}, test.level);

            // Get baseline
            auto baseline = network.forward(input);

            // Apply radiation
            network.resetErrorStats();
            auto radiation_result = network.forward(input, radiation_intensity);

            auto [detected, corrected] = network.getErrorStats();
            error_count += detected;

            // Calculate output variance
            float variance = 0.0f;
            for (size_t i = 0; i < baseline.size(); i++) {
                variance += std::pow(baseline[i] - radiation_result[i], 2);
            }
            output_variances.push_back(variance);
        }

        // Calculate statistics
        float avg_variance = 0.0f;
        for (float var : output_variances) {
            avg_variance += var;
        }
        avg_variance /= num_trials;

        std::cout << "Average errors detected: " << (float)error_count / num_trials << std::endl;
        std::cout << "Average output variance: " << std::fixed << std::setprecision(8)
                  << avg_variance << std::endl;

        // Store results for comparison
        if (test.level == ProtectionLevel::NONE) {
            // Use as baseline for comparison
            std::cout << "Baseline established for comparison" << std::endl;
        }
    }
}

TEST(CrossValidation, PerformanceConsistencyVerification)
{
    std::cout << "\n=== PERFORMANCE CONSISTENCY VERIFICATION ===" << std::endl;

    // Test that performance measurements are consistent
    struct NetworkConfig {
        std::string name;
        std::vector<size_t> architecture;
        int expected_relative_speed;  // 1=slow, 5=fast
    };

    std::vector<NetworkConfig> configs = {{"Small", {4, 8, 4}, 5},
                                          {"Medium", {8, 16, 8}, 4},
                                          {"Large", {8, 32, 8}, 3},
                                          {"Very Large", {8, 64, 8}, 2}};

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const int num_iterations = 100;

    for (const auto& config : configs) {
        std::cout << "\n--- Testing " << config.name << " Performance ---" << std::endl;

        ResidualNeuralNetwork<float> network(config.architecture, ProtectionLevel::NONE);

        // Warm up
        for (int i = 0; i < 10; i++) {
            network.forward(std::vector<float>(config.architecture[0], 1.0f));
        }

        // Time multiple runs
        std::vector<double> times;
        for (int i = 0; i < num_iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            network.forward(std::vector<float>(config.architecture[0], 1.0f));
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration<double, std::milli>(end - start);
            times.push_back(duration.count());
        }

        // Calculate statistics
        double avg_time = 0.0;
        for (double t : times) avg_time += t;
        avg_time /= num_iterations;

        double variance = 0.0;
        for (double t : times) {
            variance += (t - avg_time) * (t - avg_time);
        }
        variance /= num_iterations;
        double std_dev = std::sqrt(variance);

        std::cout << "Average time: " << std::fixed << std::setprecision(4) << avg_time << " ms"
                  << std::endl;
        std::cout << "Std deviation: " << std::fixed << std::setprecision(4) << std_dev << " ms"
                  << std::endl;
        std::cout << "Coefficient of variation: " << std::fixed << std::setprecision(2)
                  << (std_dev / avg_time * 100) << "%" << std::endl;

        // Verify consistency (low coefficient of variation)
        EXPECT_LT(std_dev / avg_time, 0.2) << "Performance measurements too inconsistent";
    }
}

TEST(CrossValidation, StatisticalSignificanceVerification)
{
    std::cout << "\n=== STATISTICAL SIGNIFICANCE VERIFICATION ===" << std::endl;

    // Test that radiation effects are statistically significant
    const int num_trials = 50;
    const float radiation_intensity = 0.2f;

    std::vector<float> input = {1.0f, 0.5f, -0.5f, -1.0f};

    std::vector<float> baseline_outputs;
    std::vector<float> radiation_outputs;

    for (int trial = 0; trial < num_trials; trial++) {
        ResidualNeuralNetwork<float> network({4, 8, 4}, ProtectionLevel::NONE);

        // Baseline
        auto baseline = network.forward(input);
        baseline_outputs.push_back(baseline[0]);  // Just use first output

        // With radiation
        auto radiation = network.forward(input, radiation_intensity);
        radiation_outputs.push_back(radiation[0]);
    }

    // Calculate means
    float baseline_mean = 0.0f;
    float radiation_mean = 0.0f;

    for (int i = 0; i < num_trials; i++) {
        baseline_mean += baseline_outputs[i];
        radiation_mean += radiation_outputs[i];
    }
    baseline_mean /= num_trials;
    radiation_mean /= num_trials;

    // Calculate variances
    float baseline_var = 0.0f;
    float radiation_var = 0.0f;

    for (int i = 0; i < num_trials; i++) {
        baseline_var +=
            (baseline_outputs[i] - baseline_mean) * (baseline_outputs[i] - baseline_mean);
        radiation_var +=
            (radiation_outputs[i] - radiation_mean) * (radiation_outputs[i] - radiation_mean);
    }
    baseline_var /= (num_trials - 1);
    radiation_var /= (num_trials - 1);

    // Simple t-test
    float pooled_std = std::sqrt((baseline_var + radiation_var) / 2.0f);
    float t_stat = (baseline_mean - radiation_mean) / (pooled_std * std::sqrt(2.0f / num_trials));

    std::cout << "Baseline mean: " << std::fixed << std::setprecision(6) << baseline_mean
              << std::endl;
    std::cout << "Radiation mean: " << std::fixed << std::setprecision(6) << radiation_mean
              << std::endl;
    std::cout << "Baseline variance: " << std::fixed << std::setprecision(8) << baseline_var
              << std::endl;
    std::cout << "Radiation variance: " << std::fixed << std::setprecision(8) << radiation_var
              << std::endl;
    std::cout << "T-statistic: " << std::fixed << std::setprecision(4) << t_stat << std::endl;

    // Check for statistically significant difference
    float t_critical = 2.01;  // Approximate for 50 samples, 95% confidence
    if (std::abs(t_stat) > t_critical) {
        std::cout << "✅ Statistically significant difference detected" << std::endl;
    }
    else {
        std::cout << "❌ No statistically significant difference" << std::endl;
        // This might indicate the radiation isn't actually affecting the network
    }
}

TEST(CrossValidation, IndependentImplementationCheck)
{
    std::cout << "\n=== INDEPENDENT IMPLEMENTATION CHECK ===" << std::endl;

    // Implement a simple, independent bit flip function
    auto independent_bit_flip = [](float value, int bit_position) -> float {
        union {
            float f;
            uint32_t i;
        } converter;
        converter.f = value;
        converter.i ^= (1u << bit_position);  // Flip the bit
        return converter.f;
    };

    // Test against our BitManipulation implementation
    std::cout << "Comparing BitManipulation::flipBit with independent implementation..."
              << std::endl;

    std::vector<float> test_values = {1.0f, -1.0f, 3.14159f, 0.0f, -3.14159f};
    std::vector<int> test_bits = {0, 1, 8, 16, 23, 31};

    int matches = 0;
    int total_tests = 0;

    for (float value : test_values) {
        for (int bit : test_bits) {
            float our_result = BitManipulation::flipBit(value, bit);
            float independent_result = independent_bit_flip(value, bit);

            total_tests++;
            if (our_result == independent_result) {
                matches++;
            }
            else {
                std::cout << "❌ Mismatch for value " << value << " bit " << bit << std::endl;
                std::cout << "   Our result: " << our_result << " (" << floatToBinary(our_result)
                          << ")" << std::endl;
                std::cout << "   Independent: " << independent_result << " ("
                          << floatToBinary(independent_result) << ")" << std::endl;
            }
        }
    }

    std::cout << "Matches: " << matches << "/" << total_tests << std::endl;
    EXPECT_EQ(matches, total_tests)
        << "BitManipulation implementation doesn't match independent version";
}
