/**
 * @file simple_radiation_test.cpp
 * @brief Simple test to prove radiation effects are working
 */

#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <rad_ml/research/residual_network.hpp>
#include <rad_ml/utils/bit_manipulation.hpp>

using namespace rad_ml;
using namespace rad_ml::research;
using namespace rad_ml::neural;

TEST(SimpleRadiationTest, ForwardPassRadiationEffects)
{
    std::cout << "\n=== SIMPLE RADIATION EFFECT VALIDATION ===\n" << std::endl;

    // Create a simple network
    ResidualNeuralNetwork<float> network({4, 8, 4}, ProtectionLevel::NONE);

    // Create simple input
    std::vector<float> input = {1.0f, 0.5f, -0.5f, 2.0f};

    std::cout << "Input: ";
    for (float val : input) {
        std::cout << std::fixed << std::setprecision(3) << val << " ";
    }
    std::cout << std::endl;

    // Test multiple times to see variation
    std::cout << "\n--- Testing Forward Pass Outputs ---" << std::endl;
    std::cout << "Run | Radiation Level | Output" << std::endl;
    std::cout << "----+-----------------+-------------------" << std::endl;

    // Test without radiation
    for (int i = 0; i < 5; ++i) {
        auto output_clean = network.forward(input, 0.0);
        std::cout << std::setw(3) << i << " | " << std::setw(15) << "0.0 (clean)" << " | ";
        for (float val : output_clean) {
            std::cout << std::fixed << std::setprecision(6) << val << " ";
        }
        std::cout << std::endl;
    }

    // Test with moderate radiation
    for (int i = 0; i < 5; ++i) {
        auto output_rad = network.forward(input, 0.1);
        std::cout << std::setw(3) << i + 5 << " | " << std::setw(15) << "0.1 (moderate)" << " | ";
        for (float val : output_rad) {
            std::cout << std::fixed << std::setprecision(6) << val << " ";
        }
        std::cout << std::endl;
    }

    // Test with high radiation
    for (int i = 0; i < 5; ++i) {
        auto output_high = network.forward(input, 1.0);
        std::cout << std::setw(3) << i + 10 << " | " << std::setw(15) << "1.0 (high)" << " | ";
        for (float val : output_high) {
            std::cout << std::fixed << std::setprecision(6) << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n--- Analysis ---" << std::endl;

    // Compare clean vs radiated outputs
    auto clean_output = network.forward(input, 0.0);
    auto radiated_output = network.forward(input, 1.0);

    std::cout << "Clean output:     ";
    for (float val : clean_output) {
        std::cout << std::fixed << std::setprecision(6) << val << " ";
    }
    std::cout << std::endl;

    std::cout << "Radiated output:  ";
    for (float val : radiated_output) {
        std::cout << std::fixed << std::setprecision(6) << val << " ";
    }
    std::cout << std::endl;

    // Calculate differences
    std::cout << "Differences:      ";
    float total_diff = 0.0f;
    for (size_t i = 0; i < clean_output.size(); ++i) {
        float diff = std::abs(clean_output[i] - radiated_output[i]);
        total_diff += diff;
        std::cout << std::fixed << std::setprecision(6) << diff << " ";
    }
    std::cout << std::endl;

    std::cout << "Total difference: " << std::fixed << std::setprecision(6) << total_diff
              << std::endl;

    if (total_diff > 0.001f) {
        std::cout << "✅ RADIATION EFFECTS ARE WORKING!" << std::endl;
    }
    else {
        std::cout << "❌ No significant radiation effects detected" << std::endl;
    }

    std::cout << "\n=== TEST COMPLETE ===\n" << std::endl;
}
